#include "egmgpu_partitioned_structural_holes.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef EASYGRAPH_ENABLE_GPU
#include "egmgpu_partition_loader.h"

#include <functions/partitioned/structural_holes_partitioned.cuh>
#include <runtime/native_runtime.h>
#include <runtime/partitioned_device_state.h>
#include <runtime/partitioned_ego_plan.h>
#include <runtime/partitioned_ego_state.h>
#include <runtime/partitioned_host_plan_cache.h>
#include <runtime/partitioned_host_plan_registry.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#endif

namespace {

#ifdef EASYGRAPH_ENABLE_GPU
using easygraph::egmgpu::MultiPartitionGraph;
using easygraph::egmgpu::PartitionCSR;
using gpu_easygraph::runtime::EgoHostPartition;
using gpu_easygraph::runtime::EgoInputPartitionView;
using gpu_easygraph::runtime::EgoPartitionDeviceUpload;
using gpu_easygraph::runtime::EgoPartitionDeviceView;
using gpu_easygraph::runtime::EgoPartitionWorkspace;
using gpu_easygraph::runtime::PartitionedEgoMetric;
using gpu_easygraph::runtime::PartitionedEgoPlan;
using gpu_easygraph::runtime::PartitionedHostPlanAlgorithm;
using gpu_easygraph::runtime::PartitionedHostPlanCache;
using gpu_easygraph::runtime::PartitionedHostPlanCacheKey;
using gpu_easygraph::runtime::ResidentEgoPartitionState;

typedef std::chrono::steady_clock SteadyClock;

double elapsed_seconds(
    const SteadyClock::time_point& begin,
    const SteadyClock::time_point& end) {
    return std::chrono::duration<double>(end - begin).count();
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
}

class CudaEventTimer {
public:
    CudaEventTimer() : begin_(nullptr), end_(nullptr) {
        check_cuda(cudaEventCreate(&begin_), "create ego timing event");
        try {
            check_cuda(cudaEventCreate(&end_), "create ego timing event");
        } catch (...) {
            cudaEventDestroy(begin_);
            throw;
        }
    }
    ~CudaEventTimer() {
        if (begin_) cudaEventDestroy(begin_);
        if (end_) cudaEventDestroy(end_);
    }
    void start() {
        check_cuda(cudaEventRecord(begin_, 0), "start ego kernel timer");
    }
    double finish() {
        check_cuda(cudaEventRecord(end_, 0), "stop ego kernel timer");
        check_cuda(cudaEventSynchronize(end_), "synchronize ego kernel timer");
        float milliseconds = 0.0F;
        check_cuda(cudaEventElapsedTime(&milliseconds, begin_, end_),
                   "read ego kernel timer");
        return static_cast<double>(milliseconds) * 1.0e-3;
    }

private:
    cudaEvent_t begin_;
    cudaEvent_t end_;
};

const char* metric_name(PartitionedEgoMetric metric) {
    switch (metric) {
        case PartitionedEgoMetric::EffectiveSize: return "effective_size";
        case PartitionedEgoMetric::Efficiency: return "efficiency";
        case PartitionedEgoMetric::Constraint: return "constraint";
        case PartitionedEgoMetric::Hierarchy: return "hierarchy";
    }
    throw std::invalid_argument("unknown partitioned structural-hole metric");
}

gpu_easygraph::partitioned::EgoMetric device_metric(
    PartitionedEgoMetric metric) {
    switch (metric) {
        case PartitionedEgoMetric::EffectiveSize:
            return gpu_easygraph::partitioned::EgoMetric::EffectiveSize;
        case PartitionedEgoMetric::Efficiency:
            return gpu_easygraph::partitioned::EgoMetric::Efficiency;
        case PartitionedEgoMetric::Constraint:
            return gpu_easygraph::partitioned::EgoMetric::Constraint;
        case PartitionedEgoMetric::Hierarchy:
            return gpu_easygraph::partitioned::EgoMetric::Hierarchy;
    }
    throw std::invalid_argument("unknown partitioned structural-hole metric");
}

std::vector<int> requested_devices(
    const MultiPartitionGraph& graph,
    std::vector<int> requested) {
    const gpu_easygraph::runtime::RuntimeStatus status =
        gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running) {
        throw std::runtime_error(
            "EGMGPU native runtime must be started before partitioned "
            "structural-hole execution");
    }
    if (requested.empty()) {
        requested.reserve(graph.partitions().size());
        for (std::size_t rank = 0; rank < graph.partitions().size(); ++rank) {
            requested.push_back(graph.partitions()[rank].device_id);
        }
    }
    std::sort(requested.begin(), requested.end());
    if (std::adjacent_find(requested.begin(), requested.end()) !=
        requested.end() || requested.size() != graph.partitions().size()) {
        throw std::invalid_argument(
            "partitioned structural-hole execution requires one distinct "
            "device per manifest partition");
    }
    for (std::size_t rank = 0; rank < requested.size(); ++rank) {
        if (!std::binary_search(
                status.device_ids.begin(), status.device_ids.end(),
                requested[rank])) {
            throw std::out_of_range(
                "partitioned structural-hole device is outside the runtime");
        }
        if (graph.partitions()[rank].device_id != requested[rank]) {
            throw std::invalid_argument(
                "partition manifest device order differs from the request");
        }
    }
    return requested;
}

std::vector<std::uint64_t> parse_nodes(
    const py::object& object,
    std::uint64_t vertex_count) {
    std::vector<std::uint64_t> nodes;
    if (object.is_none()) {
        if (vertex_count > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error("structural-hole output exceeds host memory");
        }
        nodes.resize(static_cast<std::size_t>(vertex_count));
        for (std::size_t node = 0; node < nodes.size(); ++node) {
            nodes[node] = static_cast<std::uint64_t>(node);
        }
        return nodes;
    }
    std::unordered_set<std::uint64_t> seen;
    const bool scalar = py::isinstance<py::int_>(object);
    const py::object values_object = scalar
        ? py::reinterpret_borrow<py::object>(py::make_tuple(object))
        : py::reinterpret_borrow<py::object>(object);
    const py::iterable values =
        py::reinterpret_borrow<py::iterable>(values_object);
    for (py::handle value : values) {
        const std::int64_t node = py::cast<std::int64_t>(value);
        if (node < 0 || static_cast<std::uint64_t>(node) >= vertex_count) {
            throw py::key_error("node is outside the partitioned graph domain");
        }
        if (seen.insert(static_cast<std::uint64_t>(node)).second) {
            nodes.push_back(static_cast<std::uint64_t>(node));
        }
    }
    return nodes;
}

std::string semantic_key(
    const MultiPartitionGraph& graph,
    const std::vector<int>& devices) {
    std::string key = "partitioned-ego-statistics-v1|graph=" +
        graph.graph_id() + "|generation=" + std::to_string(graph.generation()) +
        "|view=" + graph.view_signature() + "|topology=" +
        graph.topology_signature() + "|orientation=" +
        graph.view_orientation() + "|devices=";
    for (std::size_t index = 0; index < devices.size(); ++index) {
        if (index) key.push_back(',');
        key += std::to_string(devices[index]);
    }
    return key;
}

std::vector<EgoInputPartitionView> plan_input_views(
    const MultiPartitionGraph& graph) {
    std::vector<EgoInputPartitionView> result(graph.partitions().size());
    for (std::size_t rank = 0; rank < graph.partitions().size(); ++rank) {
        const PartitionCSR& source = graph.partitions()[rank];
        source.require_narrow_offsets("structural_holes");
        EgoInputPartitionView& target = result[rank];
        target.partition_id = source.partition_id;
        target.device_id = source.device_id;
        if (source.owned_vertex_count > source.local_to_global.size() ||
            source.owned_vertex_count > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::logic_error(
                "validated partition lost its owned-prefix invariant");
        }
        target.owned_vertex_count = static_cast<std::size_t>(
            source.owned_vertex_count);
        target.offsets = &source.offsets.view();
        target.indices = &source.indices.view();
        target.local_to_global = &source.local_to_global.view();
    }
    return result;
}

struct RankResult {
    std::vector<double> values;
    double kernel_seconds;
    double gather_seconds;
    std::size_t resident_bytes;
    std::size_t workspace_bytes;
    std::size_t kernel_launches;
    std::size_t selection_upload_count;
    std::size_t selection_upload_bytes;
    std::size_t result_gather_count;
    bool state_cache_hit;
    std::string phase;

    RankResult()
        : kernel_seconds(0.0),
          gather_seconds(0.0),
          resident_bytes(0U),
          workspace_bytes(0U),
          kernel_launches(0U),
          selection_upload_count(0U),
          selection_upload_bytes(0U),
          result_gather_count(0U),
          state_cache_hit(false),
          phase("not_started") {}
};

py::dict run_partitioned_structural_hole_impl(
    PartitionedEgoMetric metric,
    const std::shared_ptr<MultiPartitionGraph>& graph,
    const py::object& nodes_object,
    const py::object& weight_object,
    std::vector<int> requested_device_ids,
    bool return_diagnostics,
    std::string* stage,
    bool fused = false,
    const std::string& result_format = "dict") {
    *stage = "validate_input";
    if (result_format != "dict" && result_format != "dense") {
        throw std::invalid_argument("result_format must be 'dict' or 'dense'");
    }
    const bool dense_result = result_format == "dense";
    if (!graph || graph->vertex_count() == 0U) {
        throw std::invalid_argument(
            "partitioned structural-hole graph must be nonempty");
    }
    for (const auto& part : graph->partitions()) part.require_narrow_offsets("structural_holes");
    if (!graph->directed() || graph->view_orientation() != "outgoing" ||
        graph->weighted() || !weight_object.is_none()) {
        throw std::invalid_argument(
            "partitioned structural-hole execution requires an unweighted "
            "directed outgoing CSR manifest and weight=None");
    }
    const std::vector<int> devices = requested_devices(
        *graph, std::move(requested_device_ids));
    const std::vector<std::uint64_t> requested_nodes = parse_nodes(
        nodes_object, graph->vertex_count());
    const std::size_t metric_count = fused ? 4U : 1U;
    const std::string key = semantic_key(*graph, devices);
    const PartitionedHostPlanCacheKey cache_key =
        gpu_easygraph::runtime::make_partitioned_host_plan_cache_key(
            PartitionedHostPlanAlgorithm::Ego,
            graph->graph_id(),
            graph->generation(),
            graph->view_signature(),
            graph->topology_signature(),
            false,
            std::string(),
            graph->vertex_count(),
            graph->adjacency_count(),
            devices);

    *stage = "prepare_shared_ego_state";
    PartitionedHostPlanCache<PartitionedEgoPlan>::AcquireResult acquisition;
    const SteadyClock::time_point preparation_begin = SteadyClock::now();
    {
        py::gil_scoped_release release;
        acquisition =
            gpu_easygraph::runtime::partitioned_ego_host_plan_cache().acquire(
            cache_key,
            [graph, key]() {
                return gpu_easygraph::runtime::
                    build_partitioned_ego_plan_from_views(
                        plan_input_views(*graph), graph->vertex_count(), key);
            });
    }
    const double state_preparation_seconds = elapsed_seconds(
        preparation_begin, SteadyClock::now());
    const std::shared_ptr<const PartitionedEgoPlan> plan = acquisition.plan;

    std::vector<std::vector<int> > selected_local(devices.size());
    std::vector<std::vector<std::size_t> > selected_positions(devices.size());
    for (std::size_t position = 0; position < requested_nodes.size(); ++position) {
        const std::size_t global = static_cast<std::size_t>(
            requested_nodes[position]);
        const int rank = plan->owner_partition[global];
        const int local = plan->owner_local_id[global];
        if (rank < 0 || local < 0 ||
            static_cast<std::size_t>(rank) >= devices.size()) {
            throw std::logic_error("shared ego plan has an invalid owner map");
        }
        selected_local[static_cast<std::size_t>(rank)].push_back(local);
        selected_positions[static_cast<std::size_t>(rank)].push_back(position);
    }

    const std::shared_ptr<std::vector<RankResult> > ranks(
        new std::vector<RankResult>(devices.size()));
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.reserve(devices.size());
    const std::uint64_t graph_identity =
        gpu_easygraph::runtime::stable_signature64(graph->graph_id());
    const std::uint64_t view_signature =
        gpu_easygraph::runtime::stable_signature64(key);
    const std::uint64_t generation = graph->generation();
    const gpu_easygraph::partitioned::EgoMetric kernel_metric =
        device_metric(metric);
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        const std::vector<int> selection = selected_local[rank];
        tasks.push_back(
            [plan, devices, ranks, graph_identity, generation, view_signature,
             kernel_metric, selection, rank, fused, metric_count](int worker_device) {
                RankResult& result = (*ranks)[rank];
                result.phase = "acquire_immutable_ego_state";
                if (worker_device != devices[rank]) {
                    throw std::runtime_error(
                        "partitioned ego task ran on the wrong fixed worker");
                }
                if (selection.empty()) {
                    result.phase = "complete_empty_shard";
                    return;
                }
                const EgoHostPartition& host = plan->partitions[rank];
                EgoPartitionDeviceUpload upload;
                upload.partition_id = host.partition_id;
                upload.device_id = host.device_id;
                upload.graph_identity = graph_identity;
                upload.generation = generation;
                upload.ego_view_signature = view_signature;
                upload.device_group = devices;
                upload.semantic_key = plan->semantic_key;
                upload.host = &host;
                std::shared_ptr<ResidentEgoPartitionState> state =
                    gpu_easygraph::runtime::partitioned_ego_state_registry().acquire(
                        upload, &result.state_cache_hit);
                std::shared_ptr<EgoPartitionWorkspace> workspace =
                    gpu_easygraph::runtime::partitioned_ego_state_registry().workspace(
                        state);
                workspace->ensure(selection.size(), metric_count * selection.size());
                result.resident_bytes = state->resident_bytes();
                result.workspace_bytes = workspace->capacity_bytes();
                result.phase = "copy_selected_owner_ids";
                check_cuda(cudaMemcpy(
                    workspace->selected_nodes(), selection.data(),
                    selection.size() * sizeof(int), cudaMemcpyHostToDevice),
                    "copy selected ego owner IDs");
                result.selection_upload_count = 1U;
                result.selection_upload_bytes = selection.size() * sizeof(int);
                result.phase = fused ? "evaluate_fused_metrics" : "evaluate_metric";
                CudaEventTimer timer;
                timer.start();
                const EgoPartitionDeviceView& view = state->view();
                if (fused) {
                    gpu_easygraph::partitioned::evaluate_partitioned_ego_metrics(
                        view.offsets, view.mutual, view.neighbor_max_scale,
                        view.common_product, view.indirect, view.degree, view.sum_scale,
                        workspace->selected_nodes(), selection.size(),
                        view.owned_vertex_count, workspace->result(), 0);
                } else {
                    gpu_easygraph::partitioned::evaluate_partitioned_ego_metric(
                        view.offsets, view.mutual, view.neighbor_max_scale,
                        view.common_product, view.indirect, view.degree, view.sum_scale,
                        workspace->selected_nodes(), selection.size(),
                        view.owned_vertex_count, kernel_metric, workspace->result(), 0);
                }
                result.kernel_seconds = timer.finish();
                result.kernel_launches = 1U;
                result.phase = "gather_compact_result";
                const SteadyClock::time_point gather_begin = SteadyClock::now();
                result.values.resize(metric_count * selection.size());
                check_cuda(cudaMemcpy(
                    result.values.data(), workspace->result(),
                    result.values.size() * sizeof(double), cudaMemcpyDeviceToHost),
                    "gather partitioned ego values");
                result.result_gather_count = 1U;
                result.gather_seconds = elapsed_seconds(
                    gather_begin, SteadyClock::now());
                result.phase = "complete";
            });
    }

    *stage = "execute_worker_gang";
    const SteadyClock::time_point device_begin = SteadyClock::now();
    {
        py::gil_scoped_release release;
        try {
            gpu_easygraph::runtime::runtime_manager().execute_gang(
                devices, std::move(tasks));
        } catch (const std::exception& error) {
            std::string context("partitioned structural-hole worker failure (");
            for (std::size_t rank = 0; rank < ranks->size(); ++rank) {
                if (rank) context += ", ";
                context += "rank " + std::to_string(rank) + ":" +
                    (*ranks)[rank].phase;
            }
            context += "): ";
            context += error.what();
            throw std::runtime_error(context);
        }
    }
    const double device_processing_seconds = elapsed_seconds(
        device_begin, SteadyClock::now());

    *stage = "merge_owner_shards";
    const SteadyClock::time_point merge_begin = SteadyClock::now();
    std::vector<double> merged(metric_count * requested_nodes.size(), 0.0);
    {
        py::gil_scoped_release release;
        for (std::size_t rank = 0; rank < devices.size(); ++rank) {
            const std::size_t shard_count = selected_positions[rank].size();
            if ((*ranks)[rank].values.size() != metric_count * shard_count) {
                throw std::runtime_error(
                    "partitioned ego shard returned an invalid result length");
            }
            for (std::size_t metric_index = 0; metric_index < metric_count; ++metric_index) {
                for (std::size_t index = 0; index < shard_count; ++index) {
                    merged[metric_index * requested_nodes.size() + selected_positions[rank][index]] =
                        (*ranks)[rank].values[metric_index * shard_count + index];
                }
            }
        }
    }
    const double native_merge_seconds = elapsed_seconds(
        merge_begin, SteadyClock::now());

    py::dict out;
    const auto materialize_dense = [&]() {
        py::array_t<std::uint64_t> dense_nodes(requested_nodes.size());
        py::array_t<double> dense_values = fused
            ? py::array_t<double>(py::array::ShapeContainer{
                static_cast<py::ssize_t>(metric_count),
                static_cast<py::ssize_t>(requested_nodes.size())})
            : py::array_t<double>(merged.size());
        std::copy(requested_nodes.begin(), requested_nodes.end(), dense_nodes.mutable_data());
        std::copy(merged.begin(), merged.end(), dense_values.mutable_data());
        out["nodes_dense"] = std::move(dense_nodes);
        out["values_dense"] = std::move(dense_values);
    };
    *stage = dense_result ? "materialize_numpy_arrays" : "materialize_python_mapping";
    const SteadyClock::time_point materialization_begin = SteadyClock::now();
    if (dense_result) {
        // All values have already been computed, gathered and merged. Dense
        // output changes only host representation, without scalar boxing.
        materialize_dense();
    } else {
        py::dict values;
        for (std::size_t metric_index = 0; metric_index < metric_count; ++metric_index) {
            py::dict metric_values;
            for (std::size_t index = 0; index < requested_nodes.size(); ++index) {
                metric_values[py::int_(requested_nodes[index])] =
                    py::float_(merged[metric_index * requested_nodes.size() + index]);
            }
            if (fused) {
                values[metric_name(static_cast<PartitionedEgoMetric>(metric_index))] =
                    std::move(metric_values);
            } else {
                values = std::move(metric_values);
            }
        }
        out["values"] = std::move(values);
    }
    const double python_materialization_seconds = elapsed_seconds(
        materialization_begin, SteadyClock::now());

    double kernel_seconds = 0.0;
    double output_gather_seconds = 0.0;
    std::uint64_t aggregate_resident_bytes = 0U;
    std::uint64_t aggregate_workspace_bytes = 0U;
    std::uint64_t remote_incidence_count = 0U;
    std::uint64_t unique_remote_rows = 0U;
    std::uint64_t reused_remote_rows = 0U;
    std::uint64_t remote_row_adjacency_entries = 0U;
    std::uint64_t kernel_launches = 0U;
    std::uint64_t selection_upload_count = 0U;
    std::uint64_t selection_upload_bytes = 0U;
    std::uint64_t result_gather_count = 0U;
    std::uint64_t result_gather_bytes = 0U;
    py::list per_device;
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        const RankResult& result = (*ranks)[rank];
        const EgoHostPartition& host = plan->partitions[rank];
        kernel_seconds = std::max(kernel_seconds, result.kernel_seconds);
        output_gather_seconds = std::max(
            output_gather_seconds, result.gather_seconds);
        aggregate_resident_bytes += result.resident_bytes;
        aggregate_workspace_bytes += result.workspace_bytes;
        kernel_launches += result.kernel_launches;
        selection_upload_count += result.selection_upload_count;
        selection_upload_bytes += result.selection_upload_bytes;
        result_gather_count += result.result_gather_count;
        result_gather_bytes += result.values.size() * sizeof(double);
        remote_incidence_count += host.remote_incidence_count;
        unique_remote_rows += host.unique_remote_rows;
        reused_remote_rows += host.reused_remote_rows;
        remote_row_adjacency_entries += host.remote_row_adjacency_entries;
        if (return_diagnostics) {
            py::dict item;
            item["device_id"] = devices[rank];
            item["selected_node_count"] = selected_local[rank].size();
            item["kernel_seconds"] = result.kernel_seconds;
            item["output_gather_seconds"] = result.gather_seconds;
            item["resident_ego_state_bytes"] = result.resident_bytes;
            item["workspace_capacity_bytes"] = result.workspace_bytes;
            item["ego_state_cache_hit"] = result.state_cache_hit;
            if (fused) {
                item["kernel_launches"] = result.kernel_launches;
                item["selection_upload_count"] = result.selection_upload_count;
                item["selection_upload_bytes"] = result.selection_upload_bytes;
                item["result_gather_count"] = result.result_gather_count;
                item["result_gather_bytes"] = result.values.size() * sizeof(double);
            }
            item["remote_incidence_count"] = host.remote_incidence_count;
            item["unique_remote_rows"] = host.unique_remote_rows;
            item["reused_remote_rows"] = host.reused_remote_rows;
            per_device.append(item);
        }
    }

    const gpu_easygraph::runtime::RuntimeStatus runtime =
        gpu_easygraph::runtime::runtime_manager().status();
    out["result_format"] = result_format;
    out["function"] = fused ? "ego_metrics" : metric_name(metric);
    if (fused) {
        out["metric_count"] = metric_count;
        out["metric_names"] = std::vector<std::string>{
            "effective_size", "efficiency", "constraint", "hierarchy"};
        out["kernel_launches"] = kernel_launches;
        out["selection_upload_count"] = selection_upload_count;
        out["selection_upload_bytes"] = selection_upload_bytes;
        out["result_gather_count"] = result_gather_count;
        out["result_gather_bytes"] = result_gather_bytes;
    }
    out["devices_used"] = devices;
    out["execution_mode"] = "partitioned_vertex_shard";
    out["communication_backend"] = "none";
    out["runtime_backend"] = std::string(
        gpu_easygraph::runtime::communication_backend_name(runtime.backend));
    out["boundary_state_path"] = "host_precomputed_owner_incidence";
    out["state_preparation_seconds"] = state_preparation_seconds;
    out["host_plan_cache_hit"] = acquisition.cache_hit;
    out["host_plan_cache_builder"] = acquisition.builder;
    out["host_plan_cache_waiter"] = acquisition.waiter;
    out["host_plan_build_seconds"] = acquisition.build_seconds_measured
        ? py::cast(acquisition.build_seconds) : py::none();
    out["host_plan_cache_entries"] = acquisition.entry_count;
    out["device_processing_seconds"] = device_processing_seconds;
    out["kernel_seconds"] = kernel_seconds;
    out["output_gather_seconds"] = output_gather_seconds;
    out["native_merge_seconds"] = native_merge_seconds;
    out["python_materialization_seconds"] = python_materialization_seconds;
    out["logical_communication_bytes"] = 0U;
    out["physical_transfer_bytes"] = 0U;
    out["logical_undirected_edges"] = plan->logical_edge_count;
    out["duplicate_directed_entries_removed"] =
        plan->duplicate_directed_entries_removed;
    out["host_ego_plan_bytes"] = plan->host_bytes();
    out["aggregate_resident_ego_state_bytes"] = aggregate_resident_bytes;
    out["aggregate_workspace_capacity_bytes"] = aggregate_workspace_bytes;
    out["remote_incidence_count"] = remote_incidence_count;
    out["unique_remote_rows"] = unique_remote_rows;
    out["reused_remote_rows"] = reused_remote_rows;
    out["remote_row_adjacency_entries"] = remote_row_adjacency_entries;
    if (return_diagnostics) {
        if (!dense_result) materialize_dense();
        out["per_device"] = std::move(per_device);
        out["semantic_key"] = plan->semantic_key;
    }
    *stage = "complete";
    return out;
}

py::dict run_partitioned_structural_hole(
    PartitionedEgoMetric metric,
    const std::shared_ptr<MultiPartitionGraph>& graph,
    const py::object& nodes,
    const py::object& weight,
    const std::vector<int>& devices,
    bool return_diagnostics,
    bool fused = false,
    const std::string& result_format = "dict") {
    std::string stage("enter");
    try {
        return run_partitioned_structural_hole_impl(
            metric, graph, nodes, weight, devices, return_diagnostics, &stage, fused, result_format);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string("partitioned ") + (fused ? "ego_metrics" : metric_name(metric)) + " at " +
            stage + ": " + error.what());
    }
}

py::dict partitioned_effective_size(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    const py::object& nodes,
    const py::object& weight,
    const std::vector<int>& devices,
    bool return_diagnostics,
    const std::string& result_format) {
    return run_partitioned_structural_hole(
        PartitionedEgoMetric::EffectiveSize, graph, nodes, weight, devices,
        return_diagnostics, false, result_format);
}

py::dict partitioned_efficiency(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    const py::object& nodes,
    const py::object& weight,
    const std::vector<int>& devices,
    bool return_diagnostics,
    const std::string& result_format) {
    return run_partitioned_structural_hole(
        PartitionedEgoMetric::Efficiency, graph, nodes, weight, devices,
        return_diagnostics, false, result_format);
}

py::dict partitioned_constraint(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    const py::object& nodes,
    const py::object& weight,
    const std::vector<int>& devices,
    bool return_diagnostics,
    const std::string& result_format) {
    return run_partitioned_structural_hole(
        PartitionedEgoMetric::Constraint, graph, nodes, weight, devices,
        return_diagnostics, false, result_format);
}

py::dict partitioned_hierarchy(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    const py::object& nodes,
    const py::object& weight,
    const std::vector<int>& devices,
    bool return_diagnostics,
    const std::string& result_format) {
    return run_partitioned_structural_hole(
        PartitionedEgoMetric::Hierarchy, graph, nodes, weight, devices,
        return_diagnostics, false, result_format);
}

py::dict partitioned_ego_metrics(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    const py::object& nodes,
    const std::vector<int>& devices,
    bool return_diagnostics,
    const std::string& result_format) {
    return run_partitioned_structural_hole(
        PartitionedEgoMetric::EffectiveSize, graph, nodes, py::none(), devices,
        return_diagnostics, true, result_format);
}
#endif

}  // namespace

void bind_egmgpu_partitioned_structural_holes(py::module_& module) {
#ifdef EASYGRAPH_ENABLE_GPU
    module.def(
        "cpp_egmgpu_partitioned_effective_size",
        &partitioned_effective_size,
        py::arg("partitioned_graph"),
        py::arg("nodes") = py::none(),
        py::arg("weight") = py::none(),
        py::arg("devices") = std::vector<int>(),
        py::arg("return_diagnostics") = false,
        py::arg("result_format") = "dict");
    module.def(
        "cpp_egmgpu_partitioned_efficiency",
        &partitioned_efficiency,
        py::arg("partitioned_graph"),
        py::arg("nodes") = py::none(),
        py::arg("weight") = py::none(),
        py::arg("devices") = std::vector<int>(),
        py::arg("return_diagnostics") = false,
        py::arg("result_format") = "dict");
    module.def(
        "cpp_egmgpu_partitioned_constraint",
        &partitioned_constraint,
        py::arg("partitioned_graph"),
        py::arg("nodes") = py::none(),
        py::arg("weight") = py::none(),
        py::arg("devices") = std::vector<int>(),
        py::arg("return_diagnostics") = false,
        py::arg("result_format") = "dict");
    module.def(
        "cpp_egmgpu_partitioned_hierarchy",
        &partitioned_hierarchy,
        py::arg("partitioned_graph"),
        py::arg("nodes") = py::none(),
        py::arg("weight") = py::none(),
        py::arg("devices") = std::vector<int>(),
        py::arg("return_diagnostics") = false,
        py::arg("result_format") = "dict");
    module.def(
        "cpp_egmgpu_partitioned_ego_metrics",
        &partitioned_ego_metrics,
        py::arg("partitioned_graph"),
        py::arg("nodes") = py::none(),
        py::arg("devices") = std::vector<int>(),
        py::arg("return_diagnostics") = false,
        py::arg("result_format") = "dict",
        "Evaluate effective_size, efficiency, constraint and hierarchy in one owner-incidence scan.");
#else
    (void)module;
#endif
}
