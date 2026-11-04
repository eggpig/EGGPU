#include "egmgpu_partitioned_pagerank.h"

#ifdef EASYGRAPH_ENABLE_GPU
#include "egmgpu_partition_loader.h"
#include "egmgpu_partition_array_scope.h"
#include "egmgpu_owner_output_team_py.h"
#include "egmgpu_pr_host_output.h"
#include "egmgpu_host_halo_team.h"

#include <functions/partitioned/pagerank_partitioned.cuh>
#include <runtime/collective_transport.h>
#include <runtime/numa_pinned_host.h>
#include <runtime/device_phase_timer.h>
#include <runtime/cooperative_barrier.h>
#include <runtime/native_runtime.h>
#include <runtime/partitioned_device_state.h>
#include <runtime/partitioned_bfs_state.h>
#include <runtime/partitioned_ego_state.h>
#include <runtime/partitioned_host_plan_registry.h>
#include <runtime/partitioned_pagerank_reference.h>
#include <runtime/pagerank_cusparse_transpose.h>
#include <runtime/partitioned_sssp_state.h>
#include <runtime/partitioned_wcc_state.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#endif

namespace {

#ifdef EASYGRAPH_ENABLE_GPU
using easygraph::egmgpu::MultiPartitionGraph;
using easygraph::egmgpu::PartitionCSR;
using easygraph::egmgpu::PartitionWeightType;
using gpu_easygraph::runtime::CommunicationBackend;
using gpu_easygraph::runtime::CollectiveContext;
using gpu_easygraph::runtime::CooperativeBarrier;
using gpu_easygraph::runtime::DeviceGroup;
using gpu_easygraph::runtime::DeviceTopology;
using gpu_easygraph::runtime::PageRankHostPartition;
using gpu_easygraph::runtime::PageRankOutgoingPartition;
using gpu_easygraph::runtime::PartitionDeviceUpload;
using gpu_easygraph::runtime::PartitionWorkspace;
using gpu_easygraph::runtime::ResidentPartitionView;
using gpu_easygraph::runtime::SharedArrayCallStats;
using gpu_easygraph::runtime::WorkerStreams;

typedef std::chrono::steady_clock SteadyClock;

double elapsed_seconds(
    const SteadyClock::time_point& begin,
    const SteadyClock::time_point& end) {
    return std::chrono::duration<double>(end - begin).count();
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorName(status) +
        " (" + cudaGetErrorString(status) + ")");
}

struct CudaEventTimer {
    CudaEventTimer() : begin(nullptr), end(nullptr) {
        check_cuda(cudaEventCreate(&begin), "create PageRank timing event");
        try {
            check_cuda(cudaEventCreate(&end), "create PageRank timing event");
        } catch (...) {
            cudaEventDestroy(begin);
            throw;
        }
    }
    ~CudaEventTimer() {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }
    cudaEvent_t begin;
    cudaEvent_t end;
};

double finish_cuda_timer(CudaEventTimer& timer, cudaStream_t stream) {
    check_cuda(cudaEventRecord(timer.end, stream), "record PageRank timing end");
    check_cuda(cudaEventSynchronize(timer.end), "synchronize PageRank timing event");
    float milliseconds = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, timer.begin, timer.end),
        "read PageRank timing event");
    return static_cast<double>(milliseconds) / 1000.0;
}

std::uint64_t graph_identity(const std::string& graph_id) {
    return gpu_easygraph::runtime::stable_signature64(graph_id);
}

std::vector<int> requested_devices(
    const MultiPartitionGraph& graph, const py::object& request) {
    const gpu_easygraph::runtime::RuntimeStatus status =
        gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running) {
        throw std::runtime_error(
            "EGMGPU native runtime must be started before partitioned PageRank");
    }
    std::vector<int> devices = request.is_none()
        ? std::vector<int>()
        : request.cast<std::vector<int> >();
    if (devices.empty()) {
        const std::vector<PartitionCSR>& partitions = graph.partitions();
        devices.reserve(partitions.size());
        for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
            devices.push_back(partitions[rank].device_id);
        }
    }
    std::sort(devices.begin(), devices.end());
    if (std::adjacent_find(devices.begin(), devices.end()) != devices.end()) {
        throw std::invalid_argument(
            "partitioned PageRank devices must be distinct");
    }
    if (devices.size() != graph.partitions().size()) {
        throw std::invalid_argument(
            "partitioned PageRank requires one device for every partition");
    }
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        if (!std::binary_search(
                status.device_ids.begin(), status.device_ids.end(), devices[rank])) {
            throw std::out_of_range(
                "partitioned PageRank device is outside the active runtime");
        }
        if (graph.partitions()[rank].device_id != devices[rank]) {
            throw std::invalid_argument(
                "partition manifest device order differs from the requested group");
        }
    }
    return devices;
}

std::vector<PageRankOutgoingPartition> outgoing_partitions(
    const MultiPartitionGraph& graph) {
    if (graph.view_orientation() != "outgoing" &&
        graph.view_orientation() != "undirected") {
        throw std::invalid_argument(
            "partitioned PageRank currently requires source-owned outgoing CSR");
    }
    const std::vector<PartitionCSR>& partitions = graph.partitions();
    std::vector<PageRankOutgoingPartition> result(partitions.size());
    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        const PartitionCSR& source = partitions[rank];
        PageRankOutgoingPartition& target = result[rank];
        target.partition_id = source.partition_id;
        target.device_id = source.device_id;
        target.owned_vertex_count = static_cast<std::size_t>(
            source.owned_vertex_count);
        target.outgoing_offsets.assign(
            source.offsets.begin(), source.offsets.end());
        target.outgoing_destinations.assign(
            source.indices.begin(), source.indices.end());
        target.local_to_global.assign(
            source.local_to_global.begin(), source.local_to_global.end());
        target.owners.assign(source.owners.begin(), source.owners.end());
        if (source.weight_type == PartitionWeightType::kFloat64) {
            target.outgoing_weights.assign(
                source.weights_float64.begin(), source.weights_float64.end());
        } else if (source.weight_type == PartitionWeightType::kFloat32) {
            target.outgoing_weights.reserve(source.weights_float32.size());
            for (std::size_t edge = 0; edge < source.weights_float32.size(); ++edge) {
                target.outgoing_weights.push_back(
                    static_cast<double>(source.weights_float32[edge]));
            }
        }
    }
    return result;
}

struct HaloCopy {
    int owner_rank;
    int owner_device;
    std::size_t owner_local;
    std::size_t destination_local;
};

struct PreparedPullState {
    std::shared_ptr<MultiPartitionGraph> graph_owner;
    std::vector<PageRankHostPartition> partitions;
    std::vector<std::vector<HaloCopy> > halo_plan;
    std::vector<PartitionDeviceUpload> uploads;
    std::once_flag nccl_halo_plan_checked;
    std::once_flag sparse_halo_plan_once;
    std::shared_ptr<const gpu_easygraph::runtime::SparseHaloLayout> retained_sparse_halo_plan;
    std::once_flag host_halo_plan_checked;
    bool has_dangling = false;
    std::string preparation_method;
    std::string host_weight_storage;
    std::string device_metadata_policy;
    std::string host_transpose_policy;
    std::string transpose_engine;
    gpu_easygraph::runtime::PageRankCuSparseTransposeMetrics cusparse_transpose_metrics;
    std::string host_transpose_method;
    gpu_easygraph::runtime::WholeContiguousTransposeTimings whole_transpose_timings;
    gpu_easygraph::runtime::PackedRouteTransposeTimings packed_transpose_timings;
    std::uint64_t host_incoming_weight_bytes = 0;
    double outgoing_copy_seconds = 0.0;
    double incoming_view_seconds = 0.0;
    double halo_plan_seconds = 0.0;
    double upload_plan_seconds = 0.0;
};

std::string pagerank_device_metadata_policy() {
    const char* setting = std::getenv("EGMGPU_PAGERANK_DEVICE_METADATA");
    const std::string policy = setting ? setting : "full";
    if (policy != "full" && policy != "pull_only")
        throw std::invalid_argument("EGMGPU_PAGERANK_DEVICE_METADATA must be full or pull_only");
    return policy;
}

std::string pagerank_host_weight_storage() {
    const char* value = std::getenv("EGMGPU_PAGERANK_HOST_WEIGHT_STORAGE");
    const std::string policy = value ? value : "materialized";
    if (policy != "materialized" && policy != "implicit")
        throw std::invalid_argument("EGMGPU_PAGERANK_HOST_WEIGHT_STORAGE must be materialized or implicit");
    return policy;
}

std::string pagerank_host_transpose_policy() {
    const char* value = std::getenv("EGMGPU_PAGERANK_HOST_TRANSPOSE");
    const std::string policy = value ? value : "general";
    if (policy != "general" && policy != "auto")
        throw std::invalid_argument("EGMGPU_PAGERANK_HOST_TRANSPOSE must be general or auto");
    return policy;
}

std::string pagerank_transpose_engine() {
    const char* setting = std::getenv("EGMGPU_PAGERANK_TRANSPOSE_ENGINE");
    const std::string engine = setting ? setting : "cpu";
    if (engine != "cpu" && engine != "cusparse")
        throw std::invalid_argument("EGMGPU_PAGERANK_TRANSPOSE_ENGINE must be cpu or cusparse");
    return engine;
}

void add_host_plan_metrics(py::dict& out, const PreparedPullState& state) {
    out["host_transpose_engine"] = state.transpose_engine;
    if (state.host_transpose_method == "whole_contiguous_cusparse") {
        const auto& measured = state.cusparse_transpose_metrics;
        py::dict phase;
        phase["metadata_and_denominator"] = measured.metadata_and_denominator_seconds;
        phase["host_csc_allocation"] = measured.host_csc_allocation_seconds;
        phase["context_handle_stream"] = measured.context_handle_stream_seconds;
        phase["device_array_allocation"] = measured.device_array_allocation_seconds;
        phase["h2d_wall"] = measured.h2d_wall_seconds;
        phase["workspace_plan_allocation"] = measured.workspace_plan_allocation_seconds;
        phase["conversion_synchronized_wall"] = measured.conversion_synchronized_wall_seconds;
        phase["d2h_wall"] = measured.d2h_wall_seconds;
        phase["device_release"] = measured.device_release_seconds;
        phase["total"] = measured.total_seconds;
        out["host_cusparse_transpose_original_build_seconds"] = std::move(phase);
        py::dict resources;
        resources["logical_device"] = measured.logical_device;
        resources["cusparse_version"] = measured.cusparse_version;
        resources["conversion_performed"] = measured.conversion_performed;
        resources["workspace_bytes"] = measured.workspace_bytes;
        resources["temporary_explicit_device_peak_bytes"] = measured.temporary_explicit_device_peak_bytes;
        resources["h2d_bytes"] = measured.h2d_bytes;
        resources["d2h_bytes"] = measured.d2h_bytes;
        resources["temporary_bytes_are_current_resident"] = false;
        resources["timing_domain"] = "host wall; synchronized conversion is not pure GPU time";
        out["cusparse_transpose_original_build"] = std::move(resources);
    }
    out["host_weight_storage"] = state.host_weight_storage;
    out["device_metadata_policy"] = state.device_metadata_policy;
    std::uint64_t upload_metadata_bytes = 0U;
    for (const auto& upload : state.uploads)
        upload_metadata_bytes += upload.local_to_global.size() * sizeof(std::uint64_t)
            + upload.owners.size() * sizeof(int);
    out["host_upload_metadata_bytes"] = upload_metadata_bytes;
    out["host_upload_metadata_original_sha_input_bytes"] = upload_metadata_bytes;
    out["host_transpose"] = state.host_transpose_policy;
    out["host_transpose_method"] = state.host_transpose_method;
    out["host_incoming_weight_bytes"] = state.host_incoming_weight_bytes;
    py::dict phases;
    phases["outgoing_copy"] = state.outgoing_copy_seconds;
    phases["incoming_view"] = state.incoming_view_seconds;
    phases["halo_plan"] = state.halo_plan_seconds;
    phases["upload_plan"] = state.upload_plan_seconds;
    out["host_plan_original_build_seconds"] = std::move(phases);
    if (state.host_transpose_method == "packed_destination_owned") {
        const auto& measured = state.packed_transpose_timings;
        py::dict phase;
        phase["metadata"] = measured.metadata_seconds;
        phase["route_pack"] = measured.route_preparation_seconds;
        phase["degree_and_halo_count"] = measured.degree_and_halo_count_seconds;
        phase["prefix_halo_sort_allocation"] = measured.prefix_halo_sort_allocation_seconds;
        phase["incoming_csr_scatter"] = measured.scatter_seconds;
        phase["inner_total"] = measured.inner_total_seconds;
        phase["outer_other"] = state.incoming_view_seconds - measured.inner_total_seconds;
        out["host_packed_transpose_original_build_seconds"] = std::move(phase);
        py::dict resources;
        resources["route_temporary_bytes"] = measured.route_temporary_bytes;
        resources["count_scratch_temporary_bytes"] = measured.count_scratch_temporary_bytes;
        resources["scatter_scratch_temporary_bytes"] = measured.scatter_scratch_temporary_bytes;
        resources["route_build_temporary_payload_bytes"] = measured.route_build_peak_temporary_payload_bytes;
        resources["prefix_temporary_payload_bytes"] = measured.prefix_peak_temporary_payload_bytes;
        resources["scatter_temporary_payload_bytes"] = measured.scatter_temporary_payload_bytes;
        resources["stage_capacity_max_temporary_payload_bytes"] = measured.packed_boundary_peak_temporary_payload_bytes;
        resources["temporary_payload_upper_bound_bytes"] = measured.temporary_payload_upper_bound_bytes;
        resources["scope"] = "CPU temporary array capacity model only; excludes input/output arrays and allocator metadata; stage maximum is not an observed process peak";
        out["packed_transpose_original_build"] = std::move(resources);
    }
    if (state.host_transpose_method == "whole_contiguous") {
        py::dict whole;
        whole["metadata"] = state.whole_transpose_timings.metadata_seconds;
        whole["degree_count"] = state.whole_transpose_timings.degree_count_seconds;
        whole["prefix_and_allocation"] = state.whole_transpose_timings.prefix_and_allocation_seconds;
        whole["scatter"] = state.whole_transpose_timings.scatter_seconds;
        out["host_whole_transpose_original_build_seconds"] = std::move(whole);
    }
}

PartitionDeviceUpload make_upload(
    const MultiPartitionGraph& graph,
    const PageRankHostPartition& partition,
    const std::vector<int>& devices, const std::string& device_metadata_policy);

std::mutex& prepared_state_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, std::shared_ptr<PreparedPullState> >&
prepared_state_cache() {
    static std::unordered_map<std::string, std::shared_ptr<PreparedPullState> > cache;
    return cache;
}

std::string prepared_state_key(
    const MultiPartitionGraph& graph, const std::vector<int>& devices) {
    std::string key = graph.graph_id() + "\n" +
        std::to_string(graph.generation()) + "\n" + graph.view_signature();
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        key += ":" + std::to_string(devices[rank]);
    }
    const char* view_policy = std::getenv("EGMGPU_PAGERANK_VIEW");
    key += view_policy ? view_policy : "transpose";
    const char* share_csr = std::getenv("EGMGPU_PAGERANK_SHARE_CSR");
    key += share_csr ? share_csr : "0";
    key += ":host-weights=" + pagerank_host_weight_storage();
    key += ":host-transpose=" + pagerank_host_transpose_policy();
    key += ":transpose-engine=" + pagerank_transpose_engine();
    const std::string device_metadata = pagerank_device_metadata_policy();
    if (device_metadata != "full") key += ":device-metadata=" + device_metadata;
    return key;
}

std::vector<std::vector<HaloCopy> > make_halo_plan(
    const std::vector<PageRankHostPartition>& partitions) {
    std::vector<std::vector<HaloCopy> > plan(partitions.size());
    for (std::size_t destination_rank = 0;
         destination_rank < partitions.size(); ++destination_rank) {
        const PageRankHostPartition& destination = partitions[destination_rank];
        for (std::size_t local = destination.owned_vertex_count;
             local < destination.local_to_global.size(); ++local) {
            const int owner_rank = destination.owners[local];
            const PageRankHostPartition& owner =
                partitions[static_cast<std::size_t>(owner_rank)];
            const std::vector<std::uint64_t>::const_iterator owner_end =
                owner.local_to_global.begin() +
                static_cast<std::ptrdiff_t>(owner.owned_vertex_count);
            const std::vector<std::uint64_t>::const_iterator found =
                std::lower_bound(
                    owner.local_to_global.begin(), owner_end,
                    destination.local_to_global[local]);
            if (found == owner_end ||
                *found != destination.local_to_global[local]) {
                throw std::logic_error("partitioned PageRank halo owner lookup failed");
            }
            HaloCopy copy;
            copy.owner_rank = owner_rank;
            copy.owner_device = owner.device_id;
            copy.owner_local = static_cast<std::size_t>(
                found - owner.local_to_global.begin());
            copy.destination_local = local;
            plan[destination_rank].push_back(copy);
        }
    }
    return plan;
}

std::shared_ptr<PreparedPullState> prepare_pull_state(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    const std::vector<int>& devices,
    bool* cache_hit, int colocated_device = -1) {
    const std::string key = prepared_state_key(*graph, devices) +
        (colocated_device < 0 ? "" : ":serial-device=" + std::to_string(colocated_device));
    {
        std::lock_guard<std::mutex> lock(prepared_state_mutex());
        const std::unordered_map<
            std::string,
            std::shared_ptr<PreparedPullState> >::const_iterator found =
            prepared_state_cache().find(key);
        if (found != prepared_state_cache().end()) {
            if (cache_hit) *cache_hit = true;
            return found->second;
        }
    }
    std::shared_ptr<PreparedPullState> built(new PreparedPullState());
    built->graph_owner = graph;
    built->host_weight_storage = pagerank_host_weight_storage();
    built->device_metadata_policy = pagerank_device_metadata_policy();
    built->host_transpose_policy = pagerank_host_transpose_policy();
    built->transpose_engine = pagerank_transpose_engine();
    if (built->transpose_engine == "cusparse") {
        const char* setting = std::getenv("EGMGPU_PAGERANK_VIEW");
        const std::string requested_view = setting ? setting : "transpose";
        if (built->host_transpose_policy != "auto" || graph->partitions().size() != 1U ||
            graph->weighted() || requested_view != "transpose")
            throw std::invalid_argument("cuSPARSE transpose requires HOST_TRANSPOSE=auto, one partition, unweighted graph and VIEW=transpose");
    }
    const bool materialize_unit_weights = built->host_weight_storage == "materialized";
    auto phase_begin = SteadyClock::now();
    std::vector<PageRankOutgoingPartition> outgoing =
        outgoing_partitions(*graph);
    built->outgoing_copy_seconds = elapsed_seconds(phase_begin, SteadyClock::now());
    phase_begin = SteadyClock::now();
    const char* view_setting = std::getenv("EGMGPU_PAGERANK_VIEW");
    const std::string view_policy = view_setting ? view_setting : "transpose";
    if (view_policy != "transpose" && view_policy != "reuse_undirected")
        throw std::invalid_argument("EGMGPU_PAGERANK_VIEW must be transpose or reuse_undirected");
    if (view_policy == "reuse_undirected" && !graph->directed() &&
            graph->view_orientation() == "undirected") {
        built->partitions = gpu_easygraph::runtime::reuse_symmetric_page_rank_view(std::move(outgoing), materialize_unit_weights);
        built->preparation_method = "reuse_undirected";
        built->host_transpose_method = "reuse_undirected";
    } else {
        if (built->host_transpose_policy == "auto" && outgoing.size() == 1U) {
            if (built->transpose_engine == "cusparse") {
                built->partitions = gpu_easygraph::runtime::build_whole_contiguous_page_rank_cusparse(
                    outgoing, graph->vertex_count(), materialize_unit_weights,
                    colocated_device < 0 ? devices.front() : colocated_device,
                    &built->cusparse_transpose_metrics);
                built->host_transpose_method = "whole_contiguous_cusparse";
            } else {
            built->partitions = gpu_easygraph::runtime::build_whole_contiguous_page_rank(
                outgoing, graph->vertex_count(), materialize_unit_weights, &built->whole_transpose_timings);
            built->host_transpose_method = "whole_contiguous";
            }
        } else if (built->host_transpose_policy == "auto" && outgoing.size() > 1U) {
            built->partitions = gpu_easygraph::runtime::build_destination_owned_page_rank_packed(
                outgoing, graph->vertex_count(), materialize_unit_weights,
                &built->packed_transpose_timings);
            built->host_transpose_method = "packed_destination_owned";
        } else {
            built->partitions = gpu_easygraph::runtime::build_destination_owned_page_rank(
                outgoing, graph->vertex_count(), materialize_unit_weights);
            built->host_transpose_method = "general_destination_owned";
        }
        built->preparation_method = "transpose";
    }
    built->incoming_view_seconds = elapsed_seconds(phase_begin, SteadyClock::now());
    phase_begin = SteadyClock::now();
    built->halo_plan = make_halo_plan(built->partitions);
    for (const auto& partition : built->partitions) {
        built->host_incoming_weight_bytes += partition.incoming_weights.size() * sizeof(double);
        for (double weight : partition.owned_out_weight) {
            if (!(weight > 0.0)) built->has_dangling = true;
        }
    }
    built->halo_plan_seconds = elapsed_seconds(phase_begin, SteadyClock::now());
    phase_begin = SteadyClock::now();
    built->uploads.reserve(devices.size());
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        built->uploads.push_back(make_upload(*graph, built->partitions[rank], devices,
                                             built->device_metadata_policy));
        PartitionDeviceUpload& upload = built->uploads.back();
        if (built->transpose_engine == "cusparse")
            upload.view_signature += ":cusparse-transpose-v11";
        const char* share_csr = std::getenv("EGMGPU_PAGERANK_SHARE_CSR");
        if (built->preparation_method == "reuse_undirected" && share_csr && std::string(share_csr) == "1") {
            const auto& original = graph->partitions()[rank];
            upload.offsets_sha256 = original.offsets_sha256;
            upload.indices_sha256 = original.indices_sha256;
            upload.topology_compatibility = easygraph::egmgpu::partition_topology_compatibility(
                *graph, static_cast<int>(rank), devices);
            upload.view_signature += ":shared-csr";
        }
        upload.halo_owner_offsets.push_back(0U);
        for (std::size_t owner = 0; owner < devices.size(); ++owner) {
            for (const HaloCopy& copy : built->halo_plan[rank]) {
                if (static_cast<std::size_t>(copy.owner_rank) != owner) continue;
                upload.halo_source_local.push_back(static_cast<int>(copy.owner_local));
                upload.halo_destination_local.push_back(static_cast<int>(copy.destination_local));
            }
            upload.halo_owner_offsets.push_back(upload.halo_source_local.size());
        }
        if (colocated_device >= 0) {
            upload.device_id = colocated_device;
            upload.device_group.assign(1U, colocated_device);
            upload.view_signature += ":serial-colocated";
        }
    }
    built->upload_plan_seconds = elapsed_seconds(phase_begin, SteadyClock::now());
    std::lock_guard<std::mutex> lock(prepared_state_mutex());
    std::unordered_map<
        std::string,
        std::shared_ptr<PreparedPullState> >::const_iterator raced =
        prepared_state_cache().find(key);
    if (raced != prepared_state_cache().end()) {
        if (cache_hit) *cache_hit = true;
        return raced->second;
    }
    // Retain only the current generation of this logical graph.
    for (std::unordered_map<
             std::string,
             std::shared_ptr<PreparedPullState> >::iterator item =
             prepared_state_cache().begin();
         item != prepared_state_cache().end();) {
        if (item->second->graph_owner->graph_id() == graph->graph_id() &&
            item->second->graph_owner->generation() != graph->generation()) {
            item = prepared_state_cache().erase(item);
        } else {
            ++item;
        }
    }
    prepared_state_cache().insert(std::make_pair(key, built));
    if (cache_hit) *cache_hit = false;
    return built;
}

void clear_prepared_state_cache() {
    std::lock_guard<std::mutex> lock(prepared_state_mutex());
    prepared_state_cache().clear();
}

std::size_t prepared_state_count() {
    std::lock_guard<std::mutex> lock(prepared_state_mutex());
    return prepared_state_cache().size();
}

std::string pagerank_output_mode() {
    const char* setting = std::getenv("EGMGPU_PAGERANK_OUTPUT");
    const std::string mode = setting ? setting : "copy";
    if (mode != "copy" && mode != "move" && mode != "pinned")
        throw std::invalid_argument("EGMGPU_PAGERANK_OUTPUT must be copy, move or pinned");
    return mode;
}

py::array_t<double> pagerank_host_array(std::vector<double>& values, bool move) {
    if (!move) {
        py::array_t<double> result(values.size());
        std::copy(values.begin(), values.end(), result.mutable_data());
        return result;
    }
    std::unique_ptr<std::vector<double> > storage(new std::vector<double>(std::move(values)));
    double* data = storage->data();
    const py::ssize_t count = static_cast<py::ssize_t>(storage->size());
    py::capsule owner(storage.get(), [](void* pointer) {
        delete static_cast<std::vector<double>*>(pointer);
    });
    storage.release();
    return py::array_t<double>({count}, {sizeof(double)}, data, owner);
}

// The prepared partition plan proves a sorted, disjoint cover of the global
// vertex domain. Every result element is assigned exactly once below.
struct PageRankHostResult {
    std::vector<double> zeroed;
    std::unique_ptr<easygraph::egmgpu::PrHostOutput> direct;
    std::size_t count;
    bool direct_assembly;

    PageRankHostResult(std::size_t size, bool requested, bool move) : count(size) {
        const char* setting = std::getenv("EGMGPU_PAGERANK_RESULT_ASSEMBLY");
        const std::string mode = setting ? setting : "zeroed";
        if (mode != "zeroed" && mode != "direct")
            throw std::invalid_argument("EGMGPU_PAGERANK_RESULT_ASSEMBLY must be zeroed or direct");
        direct_assembly = mode == "direct";
        if (direct_assembly) direct.reset(new easygraph::egmgpu::PrHostOutput(count, requested && move));
        else zeroed.resize(count);
    }

    void assign(const PageRankHostPartition& part, const double* source) {
        const std::size_t size = part.owned_vertex_count;
        if (!size) return;
        double* destination = direct_assembly ? direct->data() : zeroed.data();
        if (direct_assembly && part.local_to_global[size - 1U] -
                part.local_to_global[0] == size - 1U) {
            std::memcpy(destination + part.local_to_global[0], source, size * sizeof(double));
        } else {
            for (std::size_t local = 0; local < size; ++local)
                destination[part.local_to_global[local]] = source[local];
        }
    }

    void assign_team(const PageRankHostPartition& part, const double* source,
                     int width, easygraph::egmgpu::owner_output::Observation& observation) {
        double* destination = direct_assembly ? direct->data() : zeroed.data();
        const std::size_t size = part.owned_vertex_count;
        const bool contiguous = size && direct_assembly &&
            part.local_to_global[size - 1U] - part.local_to_global[0] == size - 1U;
        easygraph::egmgpu::owner_output::run(size, width, observation,
            [&](std::size_t begin, std::size_t end) noexcept {
                if (begin == end) return;
                if (contiguous) {
                    std::memcpy(destination + part.local_to_global[0] + begin,
                        source + begin, (end - begin) * sizeof(double));
                } else {
                    for (std::size_t local = begin; local < end; ++local)
                        destination[part.local_to_global[local]] = source[local];
                }
            });
    }

    py::array_t<double> materialize(bool move) {
        if (!direct_assembly) return pagerank_host_array(zeroed, move);
        if (!move) {
            py::array_t<double> result(count);
            if (count) std::memcpy(result.mutable_data(), direct->data(), count * sizeof(double));
            return result;
        }
        return direct->publish();
    }
};

struct PageRankPinnedOutput {
    std::mutex mutex;
    std::vector<std::shared_ptr<gpu_easygraph::runtime::PinnedHostBuffer> > buffers;
};

struct PageRankOutputLease {
    std::shared_ptr<PageRankPinnedOutput> state;
    std::unique_lock<std::mutex> lock;
    bool cache_hit = false;
    std::size_t new_bytes = 0U;
    double preparation_wall = 0.0;
    std::string placement = "caller";

    void acquire(const std::string& mode, const std::vector<int>& devices,
                 const std::vector<PageRankHostPartition>& parts, bool serial) {
        if (mode != "pinned") return;
        const char* setting = std::getenv("EGMGPU_PAGERANK_PINNED_PLACEMENT");
        placement = setting ? setting : "caller";
        if (placement != "caller" && placement != "numa_local")
            throw std::invalid_argument("EGMGPU_PAGERANK_PINNED_PLACEMENT must be caller or numa_local");
        const auto begin = SteadyClock::now();
        std::string key = serial ? "pr:output:serial:" : "pr:output:parallel:";
        key += placement + ":";
        for (int device : devices) key += std::to_string(device) + ":";
        {
            py::gil_scoped_release release;
            state = std::static_pointer_cast<PageRankPinnedOutput>(
                gpu_easygraph::runtime::runtime_manager().acquire_resident_resource(
                    key, []() { return std::shared_ptr<void>(new PageRankPinnedOutput()); }, &cache_hit));
            lock = std::unique_lock<std::mutex>(state->mutex);
            state->buffers.resize(parts.size());
            for (std::size_t rank = 0; rank < parts.size(); ++rank) {
                const std::size_t bytes = parts[rank].owned_vertex_count * sizeof(double);
                auto& buffer = state->buffers[rank];
                if (!buffer || buffer->size() < bytes) {
                    buffer = placement == "numa_local"
                        ? gpu_easygraph::runtime::make_gpu_local_pinned_host_buffer(
                              bytes, devices[serial ? 0U : rank])
                        : std::make_shared<gpu_easygraph::runtime::PinnedHostBuffer>(bytes);
                    new_bytes += bytes;
                }
            }
        }
        preparation_wall = elapsed_seconds(begin, SteadyClock::now());
    }
};

struct PageRankHostHaloLease {
    std::shared_ptr<PageRankPinnedOutput> state;
    std::unique_lock<std::mutex> lock;
    bool cache_hit = false;
    std::size_t required_bytes = 0U, capacity_bytes = 0U, new_bytes = 0U;
    std::size_t replaced_bytes = 0U, allocation_peak_bytes = 0U;
    double preparation_wall = 0.0;
    void acquire(const std::vector<int>& devices, const std::vector<PageRankHostPartition>& parts) {
        const auto begin = SteadyClock::now();
        std::string key = "pr:host-halo:pinned:gpu-numa:";
        for (int device : devices) key += std::to_string(device) + ":";
        py::gil_scoped_release release;
        state = std::static_pointer_cast<PageRankPinnedOutput>(
            gpu_easygraph::runtime::runtime_manager().acquire_resident_resource(
                key, []() { return std::shared_ptr<void>(new PageRankPinnedOutput()); }, &cache_hit));
        lock = std::unique_lock<std::mutex>(state->mutex);
        state->buffers.resize(parts.size()*2U);
        for (const auto& buffer : state->buffers) if (buffer) capacity_bytes += buffer->size();
        allocation_peak_bytes = capacity_bytes;
        for (std::size_t rank=0; rank<parts.size(); ++rank) {
            const std::size_t sizes[2] = {parts[rank].owned_vertex_count,
                parts[rank].local_to_global.size()-parts[rank].owned_vertex_count};
            for (std::size_t role=0; role<2U; ++role) {
                const std::size_t bytes = sizes[role]*sizeof(double);
                required_bytes += bytes;
                auto& buffer = state->buffers[2U*rank+role];
                if (!buffer || buffer->size()<bytes) {
                    const std::size_t old = buffer ? buffer->size() : 0U;
                    auto replacement = gpu_easygraph::runtime::make_gpu_local_pinned_host_buffer(bytes, devices[rank]);
                    allocation_peak_bytes = std::max(allocation_peak_bytes, capacity_bytes+bytes);
                    buffer = std::move(replacement);
                    capacity_bytes += bytes-old; new_bytes += bytes; replaced_bytes += old;
                }
            }
        }
        preparation_wall = elapsed_seconds(begin, SteadyClock::now());
    }
};

struct RankState {
    easygraph::egmgpu::owner_output::Observation host_output_threads;
    double host_output_assembly_seconds = 0.0;
    std::shared_ptr<ResidentPartitionView> view;
    std::shared_ptr<PartitionWorkspace> workspace;
    std::vector<double> owned_result;
    std::shared_ptr<gpu_easygraph::runtime::PinnedHostBuffer> pinned_result;
    double* result_data() {
        return pinned_result ? static_cast<double*>(pinned_result->data()) : owned_result.data();
    }
    const double* result_data() const {
        return pinned_result ? static_cast<const double*>(pinned_result->data()) : owned_result.data();
    }
    std::vector<double> staged_contribution;
    std::shared_ptr<gpu_easygraph::runtime::PinnedHostBuffer> pinned_contribution, pinned_halo;
    easygraph::egmgpu::owner_output::Observation host_halo_threads;
    double host_halo_publication_wall = 0.0;
    double state_upload_seconds;
    double kernel_seconds;
    double communication_seconds;
    double output_gather_seconds;
    std::size_t resident_bytes;
    bool device_view_cache_hit;
    SharedArrayCallStats shared_arrays;
    gpu_easygraph::runtime::DeviceStateProfile device_state_profile;
    // Disjoint host wall intervals on this rank. Device event intervals below
    // are a separate diagnostic: neither is summed across ranks.
    double initialization_wall = 0.0;
    double contribution_wall = 0.0;
    double halo_wall = 0.0;
    double pull_wall = 0.0;
    double reduction_wall = 0.0;
    double nccl_halo_scratch_prepare_wall = 0.0;
    double nccl_halo_scratch_release_wall = 0.0;
    double nccl_halo_pack_wall = 0.0;
    double nccl_halo_publication_wall = 0.0;
    double nccl_halo_collective_wall = 0.0;
    double nccl_halo_gather_wall = 0.0;
    std::size_t nccl_halo_scratch_bytes = 0U;
    std::size_t nccl_sparse_send_buffer_bytes = 0U;
    std::size_t nccl_sparse_map_bytes = 0U;
    std::size_t nccl_sparse_scratch_bytes = 0U;
    std::size_t nccl_sparse_query_local_bytes = 0U;
    std::size_t nccl_sparse_map_new_bytes = 0U, nccl_sparse_map_h2d_bytes = 0U;
    std::size_t nccl_sparse_send_new_bytes = 0U;
    bool nccl_sparse_map_cache_hit = false, nccl_sparse_send_cache_hit = false, sparse_workspace_retained = false;
    double staging_owned_d2h_wall = 0.0;
    double staging_cpu_gather_wall = 0.0;
    double staging_halo_h2d_wall = 0.0;
    double kernel_init = 0.0;
    double kernel_contribution = 0.0;
    double kernel_pull = 0.0;
    double halo_device = 0.0;
    double scalar_d2h_device = 0.0;
    double output_d2h_device = 0.0;
    double task_enter_offset = 0.0;
    double task_exit_offset = 0.0;
    double peer_enable_wall = 0.0;
    double streams_acquire_wall = 0.0;

    RankState()
        : state_upload_seconds(0.0),
          kernel_seconds(0.0),
          communication_seconds(0.0),
          output_gather_seconds(0.0),
          resident_bytes(0),
          device_view_cache_hit(false) {}
};

void add_device_projection_metrics(py::dict& out, const std::vector<RankState>& states) {
    py::list partitions;
    std::uint64_t metadata_total = 0U, omitted_total = 0U, resident_total = 0U;
    for (const auto& state : states) {
        const auto& view = state.view->view();
        const std::uint64_t ids = view.local_to_global
            ? view.local_vertex_count * sizeof(std::uint64_t) : 0U;
        const std::uint64_t owners = view.owners
            ? view.local_vertex_count * sizeof(int) : 0U;
        const std::uint64_t full_bytes = view.local_vertex_count
            * (sizeof(std::uint64_t) + sizeof(int));
        const std::uint64_t actual_resident = state.view->resident_bytes()
            + state.workspace->capacity_bytes();
        py::dict item;
        item["partition_id"] = view.partition_id;
        item["device_id"] = view.device_id;
        item["policy"] = state.view->device_metadata_policy();
        item["owned_vertices"] = view.owned_vertex_count;
        item["local_vertices"] = view.local_vertex_count;
        item["incoming_arcs"] = view.adjacency_count;
        item["heavy_rows"] = view.pagerank_heavy_row_count;
        item["has_device_weight_array"] = view.weights != nullptr;
        item["weights_device_bytes"] = view.weights ? view.adjacency_count * sizeof(double) : 0U;
        item["local_to_global_device_bytes"] = ids;
        item["owners_device_bytes"] = owners;
        item["metadata_omitted_bytes"] = full_bytes - ids - owners;
        item["view_and_workspace_device_bytes"] = actual_resident;
        partitions.append(item);
        metadata_total += ids + owners;
        omitted_total += full_bytes - ids - owners;
        resident_total += actual_resident;
    }
    out["device_projection_partitions"] = partitions;
    out["device_metadata_array_bytes"] = metadata_total;
    out["device_metadata_omitted_bytes"] = omitted_total;
    out["projection_view_and_workspace_device_bytes"] = resident_total;
    out["device_projection_byte_scope"] = "Referenced current views/workspaces; omitted IDs/owners counted from actual null device pointers. Excludes separately retained states/catalog entries, runtime/NCCL and transient preparation; not physical VRAM peak or actual DMA traffic avoided on a cache hit.";
}

struct SharedPageRankExecution {
    SharedPageRankExecution(
        std::size_t count,
        const std::shared_ptr<CollectiveContext>& collective_value)
        : barrier(count),
          collective(collective_value),
          rank_states(count),
          local_dangling(count, 0.0),
          local_residual(count, 0.0),
          global_dangling(0.0),
          global_residual(0.0),
          iterations(0),
          converged(false),
          communication_bytes(0) {}

    CooperativeBarrier barrier;
    SteadyClock::time_point dispatch_begin;
    // Separate bulk transport from scalar reductions. The experimental bulk
    // route keeps scalar reduction on the host to change one data path only.
    std::shared_ptr<CollectiveContext> collective;
    std::shared_ptr<CollectiveContext> halo_collective;
    std::size_t halo_stride = 0U;
    bool variable_halo = false;
    bool sparse_halo = false;
    bool retain_sparse_state = false;
    std::string sparse_plan_key;
    std::shared_ptr<const gpu_easygraph::runtime::SparseHaloLayout> sparse_layout;
    gpu_easygraph::runtime::VariableGatherLayout halo_layout;
    std::vector<std::size_t> halo_slot_offsets;
    bool skip_nccl_halo = false;
    bool host_pinned_halo = false;
    int host_halo_team_width = 1;
    std::vector<RankState> rank_states;
    std::vector<double> local_dangling;
    std::vector<double> local_residual;
    double global_dangling;
    double global_residual;
    int iterations;
    bool converged;
    std::uint64_t communication_bytes;
};

PartitionDeviceUpload make_upload(
    const MultiPartitionGraph& graph,
    const PageRankHostPartition& partition,
    const std::vector<int>& devices, const std::string& device_metadata_policy) {
    PartitionDeviceUpload upload;
    upload.partition_id = partition.partition_id;
    upload.device_id = partition.device_id;
    upload.graph_identity = graph_identity(graph.graph_id());
    upload.graph_identity_text = graph.graph_id();
    upload.generation = graph.generation();
    upload.device_group = devices;
    upload.view_signature = graph.view_signature() + ":pagerank-pull-incoming";
    upload.orientation = "incoming";
    upload.owned_vertex_count = partition.owned_vertex_count;
    upload.local_vertex_count = partition.local_to_global.size();
    upload.device_metadata_policy = device_metadata_policy;
    upload.offsets = partition.incoming_offsets;
    upload.indices = partition.incoming_sources;
    if (graph.weighted()) upload.weights = partition.incoming_weights;
    upload.offsets_sha256 = easygraph::egmgpu::sha256_vector(upload.offsets);
    upload.indices_sha256 = easygraph::egmgpu::sha256_vector(upload.indices);
    upload.topology_compatibility =
        easygraph::egmgpu::partition_topology_compatibility(graph, partition.partition_id, devices) +
        ":incoming=" + upload.offsets_sha256 + ":" + upload.indices_sha256;
    upload.weights_sha256 = easygraph::egmgpu::sha256_vector(upload.weights);
    upload.weight_compatibility = upload.topology_compatibility + ":raw-f64:weight-key=" + graph.weight_key();
    if (device_metadata_policy == "full") {
        upload.local_to_global = partition.local_to_global;
        upload.owners = partition.owners;
    }
    upload.owned_out_weight = partition.owned_out_weight;
    for (std::size_t row = 0; row < upload.owned_vertex_count; ++row) {
        if (upload.offsets[row + 1U] - upload.offsets[row] >= gpu_easygraph::partitioned::kPageRankHeavyRowDegree) {
            upload.pagerank_heavy_rows.push_back(static_cast<int>(row));
        }
    }
    if (device_metadata_policy == "full") {
        upload.local_to_global_sha256 =
            easygraph::egmgpu::sha256_vector(upload.local_to_global);
        upload.owners_sha256 = easygraph::egmgpu::sha256_vector(upload.owners);
        upload.ownership_compatibility =
            easygraph::egmgpu::partition_ownership_compatibility_for_arrays(
                graph, partition.partition_id, devices,
                upload.local_to_global_sha256, upload.owners_sha256,
                upload.owned_vertex_count, upload.local_to_global.size());
    }
    return upload;
}

void host_scalar_sum(
    const std::shared_ptr<SharedPageRankExecution>& shared,
    std::size_t rank,
    bool dangling) {
    shared->barrier.arrive_and_wait();
    if (rank == 0) {
        double total = 0.0;
        const std::vector<double>& values = dangling
            ? shared->local_dangling
            : shared->local_residual;
        for (std::size_t index = 0; index < values.size(); ++index) {
            total += values[index];
        }
        if (dangling) {
            shared->global_dangling = total;
        } else {
            shared->global_residual = total;
        }
    }
    shared->barrier.arrive_and_wait();
}

void reduce_scalar(
    const std::shared_ptr<SharedPageRankExecution>& shared,
    PartitionWorkspace& workspace,
    WorkerStreams& streams,
    int worker_device,
    std::size_t rank,
    bool dangling) {
    double* device_scalar = workspace.scalars() + (dangling ? 0U : 1U);
    if (shared->collective) {
        shared->collective->all_reduce(
            worker_device,
            device_scalar,
            device_scalar,
            1U,
            gpu_easygraph::runtime::CollectiveDataType::Float64,
            gpu_easygraph::runtime::CollectiveReduce::Sum,
            streams.transfer());
        check_cuda(
            cudaMemcpyAsync(
                dangling ? &shared->local_dangling[rank]
                         : &shared->local_residual[rank],
                device_scalar,
                sizeof(double),
                cudaMemcpyDeviceToHost,
                streams.transfer()),
            "copy PageRank collective scalar");
        check_cuda(
            cudaStreamSynchronize(streams.transfer()),
            "synchronize PageRank collective");
        shared->barrier.arrive_and_wait();
        if (rank == 0) {
            if (dangling) {
                shared->global_dangling = shared->local_dangling[0];
            } else {
                shared->global_residual = shared->local_residual[0];
            }
        }
        shared->barrier.arrive_and_wait();
    } else {
        host_scalar_sum(shared, rank, dangling);
    }
}

// Contributions remain immutable from their publication barrier until every
// destination has completed pull. The residual reduction is also the release
// barrier for those reads; a second halo barrier is therefore unnecessary.
void run_streamed_iterations(
    const std::shared_ptr<SharedPageRankExecution>& shared,
    const PreparedPullState& prepared, std::size_t rank,
    WorkerStreams& streams, int worker_device,
    double alpha, int max_iterations, double tolerance,
    std::size_t vertex_count_value, bool profile, bool peer_transport) {
    using gpu_easygraph::runtime::DevicePhaseTimer;
    RankState& state = shared->rank_states[rank];
    PartitionWorkspace& workspace = *state.workspace;
    const auto& view = state.view->view();
    const auto& upload = prepared.uploads[rank];
    const std::size_t ranks = shared->rank_states.size();
    const double n = static_cast<double>(vertex_count_value);
    DevicePhaseTimer init_timer(profile), contribution_timer(profile),
        pull_timer(profile), halo_timer(profile), scalar_timer(profile),
        output_timer(profile);
    auto begin = SteadyClock::now();
    std::vector<double> staged_halo;
    const bool nccl_halo = static_cast<bool>(shared->halo_collective);
    const bool sparse_halo = nccl_halo && shared->sparse_halo;
    const bool pinned_halo = shared->host_pinned_halo;
    std::vector<const double*> host_sources;
    if (pinned_halo) {
        for (const auto& rank_state : shared->rank_states)
            host_sources.push_back(static_cast<const double*>(rank_state.pinned_contribution->data()));
    }
    std::unique_ptr<gpu_easygraph::runtime::DeviceAllocation> halo_scratch;
    double* gathered = nullptr;
    std::unique_ptr<gpu_easygraph::runtime::DeviceAllocation> sparse_send, sparse_source, sparse_destination;
    const int* sparse_source_ptr=nullptr;const int* sparse_packed_ptr=nullptr;
    double* sparse_send_ptr=nullptr;
    const auto* sparse_rank = sparse_halo ? &shared->sparse_layout->ranks[rank] : nullptr;
    if (nccl_halo) {
        const auto allocation_begin = SteadyClock::now();
        if (sparse_halo) {
            const std::size_t count=sparse_rank->source_indices.size();
            if (shared->retain_sparse_state) {
                auto maps=state.view->ensure_sparse_halo_maps(shared->sparse_plan_key,
                    sparse_rank->source_indices,sparse_rank->packed_indices,streams.transfer());
                state.nccl_sparse_map_cache_hit=maps.cache_hit;state.nccl_sparse_map_new_bytes=maps.new_bytes;
                state.nccl_sparse_map_h2d_bytes=maps.h2d_bytes;state.nccl_sparse_map_bytes=maps.bytes;
                state.nccl_sparse_send_new_bytes=workspace.ensure_sparse_send_capacity(count);
                state.nccl_sparse_send_cache_hit=state.nccl_sparse_send_new_bytes==0;
                state.nccl_sparse_send_buffer_bytes=workspace.sparse_send_capacity_bytes();
                sparse_source_ptr=maps.source;sparse_packed_ptr=maps.packed;sparse_send_ptr=workspace.sparse_send();
                state.nccl_sparse_scratch_bytes=maps.bytes+workspace.sparse_send_capacity_bytes();
            } else if (count) {
                sparse_send.reset(new gpu_easygraph::runtime::DeviceAllocation(worker_device,count*sizeof(double)));
                sparse_source.reset(new gpu_easygraph::runtime::DeviceAllocation(worker_device,count*sizeof(int)));
                sparse_destination.reset(new gpu_easygraph::runtime::DeviceAllocation(worker_device,count*sizeof(int)));
                check_cuda(cudaMemcpyAsync(sparse_source->data(),sparse_rank->source_indices.data(),count*sizeof(int),cudaMemcpyHostToDevice,streams.transfer()),"upload sparse halo source map");
                check_cuda(cudaMemcpyAsync(sparse_destination->data(),sparse_rank->packed_indices.data(),count*sizeof(int),cudaMemcpyHostToDevice,streams.transfer()),"upload sparse halo packed map");
                state.nccl_sparse_send_buffer_bytes=sparse_send->bytes();
                state.nccl_sparse_map_bytes=sparse_source->bytes()+sparse_destination->bytes();
                state.nccl_sparse_scratch_bytes=state.nccl_sparse_send_buffer_bytes+state.nccl_sparse_map_bytes;
                state.nccl_sparse_query_local_bytes=state.nccl_sparse_scratch_bytes;
                state.nccl_sparse_map_new_bytes=state.nccl_sparse_map_h2d_bytes=state.nccl_sparse_map_bytes;
                state.nccl_sparse_send_new_bytes=state.nccl_sparse_send_buffer_bytes;
                sparse_source_ptr=static_cast<const int*>(sparse_source->data());
                sparse_packed_ptr=static_cast<const int*>(sparse_destination->data());sparse_send_ptr=static_cast<double*>(sparse_send->data());
            }
            state.resident_bytes=state.view->resident_bytes()+workspace.capacity_bytes();
            check_cuda(cudaStreamSynchronize(streams.transfer()),"complete sparse halo mapping upload");
        } else {
            const std::size_t bytes = shared->halo_slot_offsets.back() * sizeof(double);
            halo_scratch.reset(new gpu_easygraph::runtime::DeviceAllocation(worker_device, bytes));
            gathered = static_cast<double*>(halo_scratch->data());
            check_cuda(cudaMemsetAsync(gathered, 0, bytes, streams.transfer()), "zero PageRank NCCL padding");
            state.nccl_halo_scratch_bytes = halo_scratch->bytes();
        }
        state.nccl_halo_scratch_prepare_wall = elapsed_seconds(allocation_begin, SteadyClock::now());
    }
    if (!peer_transport && !nccl_halo && !pinned_halo && !shared->skip_nccl_halo && ranks > 1U) {
        state.staged_contribution.resize(view.owned_vertex_count);
        staged_halo.resize(view.local_vertex_count - view.owned_vertex_count);
    }
    init_timer.begin(streams.compute());
    gpu_easygraph::partitioned::initialize_owned_rank(
        workspace.current_rank(), view.owned_vertex_count, 1.0 / n,
        streams.compute());
    init_timer.end(streams.compute());
    check_cuda(cudaStreamSynchronize(streams.compute()), "complete PageRank initialization");
    state.kernel_init = init_timer.seconds();
    // Every rank allocates successfully before any rank enters a collective;
    // cancellation of this barrier propagates allocation failures to the gang.
    if (nccl_halo) shared->barrier.arrive_and_wait();
    state.initialization_wall = elapsed_seconds(begin, SteadyClock::now());

    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        begin = SteadyClock::now();
        contribution_timer.begin(streams.compute());
        gpu_easygraph::partitioned::prepare_page_rank_contribution(
            workspace.current_rank(), view.owned_out_weight,
            workspace.local_contribution(), view.owned_vertex_count,
            workspace.scalars(), streams.compute());
        contribution_timer.end(streams.compute());
        if (prepared.has_dangling && !shared->collective) {
            scalar_timer.begin(streams.compute());
            check_cuda(cudaMemcpyAsync(&shared->local_dangling[rank],
                workspace.scalars(), sizeof(double), cudaMemcpyDeviceToHost,
                streams.compute()), "copy PageRank dangling sum");
            scalar_timer.end(streams.compute());
        }
        check_cuda(cudaStreamSynchronize(streams.compute()), "complete PageRank contribution");
        state.kernel_contribution += contribution_timer.seconds();
        if (prepared.has_dangling && !shared->collective)
            state.scalar_d2h_device += scalar_timer.seconds();
        state.contribution_wall += elapsed_seconds(begin, SteadyClock::now());

        begin = SteadyClock::now();
        if (prepared.has_dangling) {
            if (ranks == 1U) shared->global_dangling = shared->local_dangling[rank];
            else reduce_scalar(shared, workspace, streams, worker_device, rank, true);
        } else if (ranks > 1U && peer_transport) {
            // No dangling mass exists in this immutable graph. This one
            // barrier publishes all owned contributions and workspaces.
            shared->barrier.arrive_and_wait();
        }
        halo_timer.begin(streams.transfer());
        if (nccl_halo) {
            const auto pack_begin = SteadyClock::now();
            if (sparse_halo) {
                if (!sparse_rank->source_indices.empty())
                    gpu_easygraph::partitioned::gather_peer_page_rank_halo(
                        workspace.local_contribution(),sparse_source_ptr,
                        sparse_packed_ptr,sparse_rank->source_indices.size(),
                        sparse_send_ptr,streams.transfer());
            } else if (view.owned_vertex_count != 0U)
                check_cuda(cudaMemcpyAsync(gathered + shared->halo_slot_offsets[rank],
                    workspace.local_contribution(), view.owned_vertex_count * sizeof(double),
                    cudaMemcpyDeviceToDevice, streams.transfer()), "pack PageRank owned NCCL slot");
            check_cuda(cudaStreamSynchronize(streams.transfer()), "complete PageRank NCCL slot publication");
            state.nccl_halo_pack_wall += elapsed_seconds(pack_begin, SteadyClock::now());
            const auto publication_begin = SteadyClock::now();
            shared->barrier.arrive_and_wait();
            state.nccl_halo_publication_wall += elapsed_seconds(publication_begin, SteadyClock::now());
            const auto collective_begin = SteadyClock::now();
            // NCCL permits a separate communicator per fixed GPU thread. Each
            // worker issues one AllGather or a thread-local group of broadcasts; do not span a
            // GroupStart/End across threads. Single-thread multi-device launch
            // would instead require all operations in one NCCL group.
            if (sparse_halo)
                shared->halo_collective->exchange_sparse_halo_f64(worker_device,
                    sparse_send_ptr,
                    workspace.local_contribution(),*shared->sparse_layout,streams.transfer());
            else if (shared->variable_halo)
                shared->halo_collective->all_gather_variable_f64(worker_device,
                    gathered, shared->halo_layout, streams.transfer());
            else
                shared->halo_collective->all_gather(worker_device,
                    gathered + shared->halo_slot_offsets[rank], gathered, shared->halo_stride,
                    gpu_easygraph::runtime::CollectiveDataType::Float64, streams.transfer());
            check_cuda(cudaStreamSynchronize(streams.transfer()), "complete PageRank NCCL allgather");
            state.nccl_halo_collective_wall += elapsed_seconds(collective_begin, SteadyClock::now());
            const auto gather_begin = SteadyClock::now();
            if (!sparse_halo) for (std::size_t owner = 0; owner < ranks; ++owner) {
                const std::size_t first = upload.halo_owner_offsets[owner];
                const std::size_t count = upload.halo_owner_offsets[owner + 1U] - first;
                if (count == 0U) continue;
                // These pointers refer to THIS GPU's receive buffer. The same
                // index kernel is reused without remote peer memory accesses.
                gpu_easygraph::partitioned::gather_peer_page_rank_halo(
                    gathered + shared->halo_slot_offsets[owner],
                    view.halo_source_local + first, view.halo_destination_local + first,
                    count, workspace.local_contribution(), streams.transfer());
            }
            check_cuda(cudaStreamSynchronize(streams.transfer()), "complete PageRank local NCCL halo gather");
            if (!sparse_halo) state.nccl_halo_gather_wall += elapsed_seconds(gather_begin, SteadyClock::now());
        } else if (shared->skip_nccl_halo) {
            // The explicit route is vacuous for P1 or a globally empty halo.
            // A single rank with no halo MUST still join a nonempty collective.
        } else if (!peer_transport && ranks > 1U) {
            const auto download_begin = SteadyClock::now();
            double* host_owned = pinned_halo ? static_cast<double*>(state.pinned_contribution->data()) : state.staged_contribution.data();
            double* host_halo = pinned_halo ? static_cast<double*>(state.pinned_halo->data()) : staged_halo.data();
            const std::size_t halo_count = view.local_vertex_count-view.owned_vertex_count;
            if (view.owned_vertex_count)
                check_cuda(cudaMemcpyAsync(host_owned,
                    workspace.local_contribution(), view.owned_vertex_count * sizeof(double),
                    cudaMemcpyDeviceToHost, streams.transfer()), "stage PageRank owned contributions");
            check_cuda(cudaStreamSynchronize(streams.transfer()), "complete PageRank host contributions");
            state.staging_owned_d2h_wall += elapsed_seconds(download_begin, SteadyClock::now());
            // All D2H copies are complete before any CPU reads another rank.
            const auto publication_begin = SteadyClock::now();
            shared->barrier.arrive_and_wait();
            state.host_halo_publication_wall += elapsed_seconds(publication_begin, SteadyClock::now());
            const auto gather_begin = SteadyClock::now();
            if (pinned_halo) {
                easygraph::egmgpu::host_halo::gather(upload.halo_owner_offsets,
                    upload.halo_source_local, upload.halo_destination_local, host_sources,
                    view.owned_vertex_count, host_halo, shared->host_halo_team_width, state.host_halo_threads);
            } else {
                for (const HaloCopy& copy : prepared.halo_plan[rank])
                    host_halo[copy.destination_local-view.owned_vertex_count] =
                        shared->rank_states[copy.owner_rank].staged_contribution[copy.owner_local];
            }
            state.staging_cpu_gather_wall += elapsed_seconds(gather_begin, SteadyClock::now());
            const auto upload_begin = SteadyClock::now();
            if (halo_count)
                check_cuda(cudaMemcpyAsync(workspace.local_contribution() + view.owned_vertex_count,
                    host_halo, halo_count * sizeof(double),
                    cudaMemcpyHostToDevice, streams.transfer()), "stage PageRank halo contributions");
            check_cuda(cudaStreamSynchronize(streams.transfer()), "complete PageRank host halo upload");
            state.staging_halo_h2d_wall += elapsed_seconds(upload_begin, SteadyClock::now());
        } else for (std::size_t owner = 0; owner < ranks; ++owner) {
            const std::size_t first = upload.halo_owner_offsets[owner];
            const std::size_t count = upload.halo_owner_offsets[owner + 1U] - first;
            if (count == 0U) continue;
            gpu_easygraph::partitioned::gather_peer_page_rank_halo(
                shared->rank_states[owner].workspace->local_contribution(),
                view.halo_source_local + first, view.halo_destination_local + first,
                count, workspace.local_contribution(), streams.transfer());
        }
        halo_timer.end(streams.transfer());
        if (ranks > 1U || profile)
            check_cuda(cudaStreamSynchronize(streams.transfer()), "complete PageRank halo");
        state.halo_device += halo_timer.seconds();
        state.halo_wall += elapsed_seconds(begin, SteadyClock::now());

        begin = SteadyClock::now();
        const double base = (1.0 - alpha + alpha * shared->global_dangling) / n;
        pull_timer.begin(streams.compute());
        gpu_easygraph::partitioned::pull_page_rank_rows(
            view.offsets, view.indices, view.weights, workspace.local_contribution(),
            workspace.current_rank(), workspace.next_rank(), view.owned_vertex_count,
            alpha, base, workspace.scalars() + 1U, streams.compute(),
            view.pagerank_heavy_rows, view.pagerank_heavy_row_count);
        pull_timer.end(streams.compute());
        if (!shared->collective) {
            scalar_timer.begin(streams.compute());
            check_cuda(cudaMemcpyAsync(&shared->local_residual[rank],
                workspace.scalars() + 1U, sizeof(double), cudaMemcpyDeviceToHost,
                streams.compute()), "copy PageRank residual");
            scalar_timer.end(streams.compute());
        }
        check_cuda(cudaStreamSynchronize(streams.compute()), "complete PageRank pull");
        state.kernel_pull += pull_timer.seconds();
        if (!shared->collective) state.scalar_d2h_device += scalar_timer.seconds();
        state.pull_wall += elapsed_seconds(begin, SteadyClock::now());

        begin = SteadyClock::now();
        workspace.advance_rank();
        if (shared->collective) {
            reduce_scalar(shared, workspace, streams, worker_device, rank, false);
            if (rank == 0U) {
                shared->iterations = iteration;
                shared->converged = shared->global_residual <= n * tolerance;
            }
            shared->barrier.arrive_and_wait();
        } else {
            if (ranks > 1U) shared->barrier.arrive_and_wait();
            if (rank == 0U) {
                double sum = 0.0;
                for (double value : shared->local_residual) sum += value;
                shared->global_residual = sum;
                shared->iterations = iteration;
                shared->converged = sum <= n * tolerance;
            }
            if (ranks > 1U) shared->barrier.arrive_and_wait();
        }
        state.reduction_wall += elapsed_seconds(begin, SteadyClock::now());
        if (shared->converged) break;
    }
    begin = SteadyClock::now();
    if (!state.pinned_result) state.owned_result.resize(view.owned_vertex_count);
    output_timer.begin(streams.compute());
    if (view.owned_vertex_count != 0U) {
        check_cuda(cudaMemcpyAsync(state.result_data(), workspace.current_rank(),
            view.owned_vertex_count * sizeof(double), cudaMemcpyDeviceToHost,
            streams.compute()), "gather partitioned PageRank owned ranks");
    }
    output_timer.end(streams.compute());
    check_cuda(cudaStreamSynchronize(streams.compute()), "complete PageRank result transfer");
    state.output_d2h_device = output_timer.seconds();
    state.output_gather_seconds = elapsed_seconds(begin, SteadyClock::now());
    if (halo_scratch || sparse_send || sparse_source || sparse_destination) {
        // All collective/gather/output work has completed on this fixed device.
        // Query-local scratch does not become an unreported retained allocation.
        const auto release_begin = SteadyClock::now();
        halo_scratch.reset();sparse_send.reset();sparse_source.reset();sparse_destination.reset();
        state.nccl_halo_scratch_release_wall = elapsed_seconds(release_begin, SteadyClock::now());
    }
    state.kernel_seconds = state.kernel_init + state.kernel_contribution + state.kernel_pull;
    state.communication_seconds = state.halo_wall + state.reduction_wall;
}

#include "egmgpu_pagerank_serial.inc"

py::dict run_partitioned_pagerank(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    double alpha,
    int max_iterations,
    double tolerance,
    py::object device_request) {
    if (!graph) throw std::invalid_argument("partitioned PageRank graph is null");
    if (!(alpha > 0.0 && alpha < 1.0) || !std::isfinite(alpha) ||
        max_iterations <= 0 || tolerance <= 0.0 || !std::isfinite(tolerance)) {
        throw std::invalid_argument("invalid partitioned PageRank parameters");
    }
    if (graph->vertex_count() == 0) {
        throw std::invalid_argument(
            "native partition manifests do not encode empty graph partitions");
    }
    const std::vector<int> devices = requested_devices(*graph, device_request);
    const DeviceGroup active_group =
        gpu_easygraph::runtime::runtime_manager().device_group();
    const DeviceTopology topology =
        gpu_easygraph::runtime::runtime_manager().topology();
    CommunicationBackend backend = active_group.backend();
    if (devices.size() == 1U) backend = CommunicationBackend::None;
    const bool peer_available = backend != CommunicationBackend::HostStaging &&
        topology.is_fully_peer_connected(DeviceGroup(devices, backend));
    const char* halo_setting = std::getenv("EGMGPU_PAGERANK_HALO_EXCHANGE");
    const std::string halo_exchange = halo_setting ? halo_setting : "peer_gather";
    if (halo_exchange != "peer_gather" && halo_exchange != "scalar_copy" && halo_exchange != "nccl_allgather" && halo_exchange != "host_pinned_parallel" && halo_exchange != "nccl_variable_gather" && halo_exchange != "nccl_sparse_halo") {
        throw std::invalid_argument("EGMGPU_PAGERANK_HALO_EXCHANGE must be peer_gather, scalar_copy, nccl_allgather, nccl_variable_gather, nccl_sparse_halo or host_pinned_parallel");
    }
    const bool variable_halo_requested = halo_exchange == "nccl_variable_gather";
    const bool sparse_halo_requested = halo_exchange == "nccl_sparse_halo";
    const char* sparse_state_setting=std::getenv("EGMGPU_PAGERANK_SPARSE_STATE");
    const std::string sparse_state=sparse_state_setting ? sparse_state_setting : "call";
    if (sparse_state!="call" && sparse_state!="retained") throw std::invalid_argument("EGMGPU_PAGERANK_SPARSE_STATE must be call or retained");
    const bool retain_sparse_state=sparse_state=="retained";
    if (retain_sparse_state && (!sparse_halo_requested || !gpu_easygraph::runtime::managed_workspace_per_worker()))
        throw std::invalid_argument("retained sparse state requires nccl_sparse_halo and per_worker workspace leases");
    const bool nccl_halo_requested = halo_exchange == "nccl_allgather" || variable_halo_requested || sparse_halo_requested;
    const bool host_halo_requested = halo_exchange == "host_pinned_parallel";
    if (nccl_halo_requested && devices.size() > 1U && backend != CommunicationBackend::Nccl)
        throw std::invalid_argument("NCCL bulk halo requires explicit NCCL runtime; no fallback");
    const bool use_peer = peer_available && !nccl_halo_requested && !host_halo_requested;
    const bool use_peer_gather = use_peer && halo_exchange == "peer_gather";
    const char* schedule_setting = std::getenv("EGMGPU_PAGERANK_SCHEDULE");
    const std::string schedule = schedule_setting ? schedule_setting : "legacy";
    if (schedule != "legacy" && schedule != "streamed")
        throw std::invalid_argument("EGMGPU_PAGERANK_SCHEDULE must be legacy or streamed");
    const bool streamed = schedule == "streamed";
    if ((nccl_halo_requested || host_halo_requested) && !streamed)
        throw std::invalid_argument("explicit bulk halo requires streamed PageRank");
    const std::string output_mode = pagerank_output_mode();
    const bool numpy_output_requested = easygraph::egmgpu::pr_numpy_output_requested();
    if (!streamed && output_mode == "pinned")
        throw std::invalid_argument("pinned PageRank output requires streamed schedule");
    if (streamed && devices.size() > 1U && use_peer && !use_peer_gather)
        throw std::invalid_argument("streamed peer PageRank requires peer_gather transport");
    const char* profile_setting = std::getenv("EGMGPU_PAGERANK_PROFILE");
    const bool profile = profile_setting && std::string(profile_setting) == "1";
    const char* scalar_setting = std::getenv("EGMGPU_PAGERANK_SCALAR_REDUCTION");
    const std::string scalar_policy = scalar_setting ? scalar_setting : "auto";
    if (scalar_policy != "auto" && scalar_policy != "host" && scalar_policy != "nccl")
        throw std::invalid_argument("EGMGPU_PAGERANK_SCALAR_REDUCTION must be auto, host, or nccl");
    if ((nccl_halo_requested || host_halo_requested) && scalar_policy != "host")
        throw std::invalid_argument("explicit bulk halo experiment requires host scalar reduction");
    if (scalar_policy == "nccl" && devices.size() > 1U && backend != CommunicationBackend::Nccl)
        throw std::invalid_argument("NCCL scalar reduction requires an NCCL runtime group");
    // Streamed iterations already make the convergence scalars available to
    // fixed workers on one shared host. Sum those values there; a scalar NCCL
    // collective would transfer and synchronize the same eight-byte state
    // again. The explicit NCCL policy remains available for matched ablations.
    const bool nccl_scalars = devices.size() > 1U && backend == CommunicationBackend::Nccl &&
        (scalar_policy == "nccl" || (scalar_policy == "auto" && !streamed));

    std::shared_ptr<PreparedPullState> prepared_state;
    bool host_view_cache_hit = false;
    const SteadyClock::time_point preparation_begin = SteadyClock::now();
    {
        py::gil_scoped_release release;
        prepared_state = prepare_pull_state(
            graph, devices, &host_view_cache_hit);
    }
    const std::vector<PageRankHostPartition>& partitions =
        prepared_state->partitions;
    const std::vector<std::vector<HaloCopy> >& halo_plan =
        prepared_state->halo_plan;
    const double state_preparation_seconds = elapsed_seconds(
        preparation_begin, SteadyClock::now());
    std::size_t halo_stride = 0U;
    std::uint64_t global_halo_count = 0U;
    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        halo_stride = std::max(halo_stride, partitions[rank].owned_vertex_count);
        global_halo_count += prepared_state->halo_plan[rank].size();
    }
    std::vector<std::size_t> owned_counts;
    for (const auto& part : partitions) owned_counts.push_back(part.owned_vertex_count);
    const auto packed_layout=gpu_easygraph::runtime::variable_gather_layout(owned_counts);
    if (packed_layout.total_values!=graph->vertex_count())
        throw std::invalid_argument("PR owned domains do not cover graph");
    const bool nccl_halo_active = nccl_halo_requested && devices.size() > 1U && global_halo_count != 0U;
    const bool host_halo_active = host_halo_requested && devices.size()>1U && global_halo_count!=0U;
    const int host_halo_width = host_halo_active ? easygraph::egmgpu::owner_output::requested_width(devices.size()) : 0;
    cpu_set_t host_halo_caller_mask; CPU_ZERO(&host_halo_caller_mask);
    bool host_halo_plan_cache_hit = true;
    const auto host_plan_begin = SteadyClock::now();
    if (host_halo_active) {
        const char* wait_policy = std::getenv("OMP_WAIT_POLICY");
        if (!wait_policy || std::string(wait_policy)!="PASSIVE")
            throw std::invalid_argument("host_pinned_parallel requires OMP_WAIT_POLICY=PASSIVE before process start");
        if (sched_getaffinity(0,sizeof(host_halo_caller_mask),&host_halo_caller_mask)!=0)
            throw std::runtime_error("cannot observe host halo caller CPU budget");
        std::call_once(prepared_state->host_halo_plan_checked, [&]() {
            host_halo_plan_cache_hit = false;
            std::vector<std::size_t> counts;
            for (const auto& part : partitions) counts.push_back(part.owned_vertex_count);
            for (std::size_t rank=0; rank<partitions.size(); ++rank) {
                const auto& upload = prepared_state->uploads[rank];
                easygraph::egmgpu::host_halo::validate(upload.halo_owner_offsets,
                    upload.halo_source_local, upload.halo_destination_local, counts,
                    partitions[rank].owned_vertex_count,
                    partitions[rank].local_to_global.size()-partitions[rank].owned_vertex_count);
            }
        });
    }
    const double host_halo_plan_seconds = host_halo_active ? elapsed_seconds(host_plan_begin,SteadyClock::now()) : 0.0;
    if (nccl_halo_active) {
        if (!variable_halo_requested && !sparse_halo_requested && halo_stride > std::numeric_limits<std::size_t>::max() / devices.size() / sizeof(double))
            throw std::overflow_error("PageRank padded NCCL buffer size overflow");
        // Validate cached owner/slot metadata once. Padding is never a source.
        std::call_once(prepared_state->nccl_halo_plan_checked, [&]() {
            for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
                const auto& upload = prepared_state->uploads[rank];
                const auto& offsets = upload.halo_owner_offsets;
                if (offsets.size() != devices.size() + 1U || offsets.front() != 0U ||
                    offsets.back() != upload.halo_source_local.size() ||
                    upload.halo_source_local.size() != upload.halo_destination_local.size())
                    throw std::invalid_argument("Invalid PageRank NCCL halo index shape");
                for (std::size_t owner = 0; owner < devices.size(); ++owner) {
                    if (offsets[owner] > offsets[owner + 1U] || offsets[owner + 1U] > offsets.back())
                        throw std::invalid_argument("Invalid PageRank NCCL halo owner interval");
                    for (std::size_t i = offsets[owner]; i < offsets[owner + 1U]; ++i) {
                        const int source = upload.halo_source_local[i];
                        const int destination = upload.halo_destination_local[i];
                        if (source < 0 || static_cast<std::size_t>(source) >= partitions[owner].owned_vertex_count ||
                            destination < 0 || static_cast<std::size_t>(destination) < partitions[rank].owned_vertex_count ||
                            static_cast<std::size_t>(destination) >= partitions[rank].local_to_global.size())
                            throw std::invalid_argument("NCCL halo index reads padding or invalid destination");
                    }
                }
            }
        });
    }
    std::shared_ptr<const gpu_easygraph::runtime::SparseHaloLayout> sparse_layout;
    double sparse_plan_seconds=0.0;bool sparse_plan_cache_hit=false;
    if (sparse_halo_requested && nccl_halo_active) {
        const auto begin=SteadyClock::now();
        std::vector<std::size_t> local_counts;
        std::vector<gpu_easygraph::runtime::SparseHaloRequestView> requests;
        for (const auto& part:partitions) local_counts.push_back(part.local_to_global.size());
        for (const auto& upload:prepared_state->uploads)
            requests.push_back({&upload.halo_owner_offsets,&upload.halo_source_local,&upload.halo_destination_local});
        {
            py::gil_scoped_release release;
            const auto build=[&]() {return std::make_shared<const gpu_easygraph::runtime::SparseHaloLayout>(
                gpu_easygraph::runtime::sparse_halo_layout(owned_counts,local_counts,requests));};
            if (retain_sparse_state) {
                bool constructed=false;
                std::call_once(prepared_state->sparse_halo_plan_once,[&]() {
                    auto next=build();prepared_state->retained_sparse_halo_plan=std::move(next);constructed=true;
                });
                sparse_layout=prepared_state->retained_sparse_halo_plan;sparse_plan_cache_hit=!constructed;
            } else sparse_layout=build();
        }
        if (sparse_layout->halo_values!=global_halo_count) throw std::logic_error("sparse halo request coverage differs");
        sparse_plan_seconds=elapsed_seconds(begin,SteadyClock::now());
    }
    std::shared_ptr<CollectiveContext> acquired_collective;
    bool collective_context_cache_hit = false;
    const auto collective_begin = SteadyClock::now();
    if (nccl_scalars || nccl_halo_active) {
        std::string resource_key = "nccl-communicator";
        for (int device : devices) resource_key += ":" + std::to_string(device);
        py::gil_scoped_release release;
        acquired_collective = std::static_pointer_cast<CollectiveContext>(
            gpu_easygraph::runtime::runtime_manager().acquire_resident_resource(
                resource_key, [devices]() -> std::shared_ptr<void> {
                    return std::make_shared<CollectiveContext>(
                        DeviceGroup(devices, CommunicationBackend::Nccl));
                }, &collective_context_cache_hit));
    }
    const std::shared_ptr<CollectiveContext> collective = nccl_scalars ? acquired_collective : nullptr;
    const double collective_context_acquire_seconds = (nccl_scalars || nccl_halo_active)
        ? elapsed_seconds(collective_begin, SteadyClock::now()) : 0.0;
    const bool output_team16 = easygraph::egmgpu::owner_output::enabled();
    if (output_team16 && !streamed)
        throw std::invalid_argument("core PR output team16 requires streamed execution");
    const int output_team_width = output_team16
        ? easygraph::egmgpu::owner_output::requested_width(devices.size()) : 1;
    const auto output_allocation_begin = SteadyClock::now();
    const std::shared_ptr<PageRankHostResult> worker_result = output_team16
        ? std::make_shared<PageRankHostResult>(static_cast<std::size_t>(graph->vertex_count()), numpy_output_requested, output_mode != "copy")
        : std::shared_ptr<PageRankHostResult>();
    const double host_output_allocation_seconds = output_team16
        ? elapsed_seconds(output_allocation_begin, SteadyClock::now()) : 0.0;
    const std::shared_ptr<SharedPageRankExecution> shared(
        new SharedPageRankExecution(partitions.size(), collective));
    shared->halo_collective = nccl_halo_active ? acquired_collective : nullptr;
    shared->halo_stride = halo_stride;
    shared->variable_halo = variable_halo_requested;
    shared->sparse_halo=sparse_halo_requested;shared->sparse_layout=sparse_layout;
    shared->retain_sparse_state=retain_sparse_state;
    shared->sparse_plan_key=prepared_state_key(*graph,devices);
    shared->halo_layout = packed_layout;
    if (variable_halo_requested || sparse_halo_requested) shared->halo_slot_offsets = packed_layout.offsets;
    else for (std::size_t rank=0;rank<=devices.size();++rank)
        shared->halo_slot_offsets.push_back(rank*halo_stride);
    const std::size_t halo_receive_values=sparse_halo_requested ? 0U : shared->halo_slot_offsets.back();
    shared->skip_nccl_halo = (nccl_halo_requested && !nccl_halo_active) || (host_halo_requested && !host_halo_active);
    shared->host_pinned_halo = host_halo_active;
    shared->host_halo_team_width = host_halo_width;
    PageRankHostHaloLease halo_lease;
    if (host_halo_active) {
        halo_lease.acquire(devices, partitions);
        for (std::size_t rank=0; rank<partitions.size(); ++rank) {
            shared->rank_states[rank].pinned_contribution = halo_lease.state->buffers[2U*rank];
            shared->rank_states[rank].pinned_halo = halo_lease.state->buffers[2U*rank+1U];
        }
    }
    PageRankOutputLease output_lease;
    output_lease.acquire(output_mode, devices, partitions, false);
    if (output_lease.state)
        for (std::size_t rank = 0; rank < partitions.size(); ++rank)
            shared->rank_states[rank].pinned_result = output_lease.state->buffers[rank];
    const std::shared_ptr<const std::vector<std::vector<HaloCopy> > >
        halo_state(prepared_state, &prepared_state->halo_plan);
    const std::size_t vertex_count_value =
        static_cast<std::size_t>(graph->vertex_count());

    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.reserve(partitions.size());
    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        tasks.push_back(
            [shared, halo_state, prepared_state, devices, topology,
             use_peer, use_peer_gather, streamed, profile, alpha, max_iterations, tolerance, rank,
             vertex_count_value, worker_result, output_team_width](int worker_device) {
                RankState& task_state = shared->rank_states[rank];
                task_state.task_enter_offset = elapsed_seconds(
                    shared->dispatch_begin, SteadyClock::now());
                gpu_easygraph::runtime::run_cooperative_rank(
                    shared->barrier,
                    [shared, halo_state, prepared_state, devices, topology,
                     use_peer, use_peer_gather, streamed, profile, alpha, max_iterations, tolerance, rank,
                     worker_device, vertex_count_value, worker_result, output_team_width]() {
                        const PartitionDeviceUpload& upload = prepared_state->uploads[rank];
                        RankState& state = shared->rank_states[rank];
                        if (worker_device != upload.device_id) {
                            throw std::runtime_error(
                                "partitioned PageRank ran on the wrong worker");
                        }
                        const auto peer_begin = SteadyClock::now();
                        if (use_peer && devices.size() > 1U) {
                            gpu_easygraph::runtime::enable_worker_peer_access(
                                DeviceGroup(devices, CommunicationBackend::PeerToPeer),
                                topology);
                        }
                        state.peer_enable_wall = elapsed_seconds(peer_begin, SteadyClock::now());
                        gpu_easygraph::runtime::PartitionedDeviceStateRegistry& registry =
                            gpu_easygraph::runtime::partitioned_device_state_registry();
                        const SteadyClock::time_point upload_begin =
                            SteadyClock::now();
                        gpu_easygraph::runtime::begin_device_state_profile(profile);
                        state.view = registry.acquire(
                            upload, &state.device_view_cache_hit,
                            &state.shared_arrays);
                        state.workspace = registry.workspace(state.view);
                        state.sparse_workspace_retained=registry.is_pooled_workspace(state.workspace);
                        state.device_state_profile =
                            gpu_easygraph::runtime::end_device_state_profile();
                        state.state_upload_seconds = elapsed_seconds(
                            upload_begin, SteadyClock::now());
                        state.resident_bytes = state.view->resident_bytes() +
                            state.workspace->capacity_bytes();
                        const gpu_easygraph::runtime::PartitionDeviceView& view =
                            state.view->view();
                        PartitionWorkspace& workspace = *state.workspace;
                        const auto streams_begin = SteadyClock::now();
                        WorkerStreams& streams =
                            gpu_easygraph::runtime::worker_streams();
                        state.streams_acquire_wall = elapsed_seconds(streams_begin, SteadyClock::now());
                        if (streamed) {
                            run_streamed_iterations(shared, *prepared_state, rank,
                                streams, worker_device, alpha, max_iterations,
                                tolerance, vertex_count_value, profile, use_peer_gather);
                            if (worker_result) {
                                const auto output_begin = SteadyClock::now();
                                worker_result->assign_team(prepared_state->partitions[rank],
                                    state.result_data(), output_team_width, state.host_output_threads);
                                state.host_output_assembly_seconds = elapsed_seconds(
                                    output_begin, SteadyClock::now());
                            }
                            return;
                        }
                        const double initial = 1.0 /
                            static_cast<double>(vertex_count_value);
                        CudaEventTimer initialization_timer;
                        check_cuda(
                            cudaEventRecord(
                                initialization_timer.begin, streams.compute()),
                            "record PageRank initialization start");
                        gpu_easygraph::partitioned::initialize_owned_rank(
                            workspace.current_rank(), view.owned_vertex_count,
                            initial, streams.compute());
                        state.kernel_seconds += finish_cuda_timer(
                            initialization_timer, streams.compute());

                        for (int iteration = 1;
                             iteration <= max_iterations; ++iteration) {
                            CudaEventTimer contribution_timer;
                            check_cuda(
                                cudaEventRecord(
                                    contribution_timer.begin, streams.compute()),
                                "record PageRank contribution start");
                            gpu_easygraph::partitioned::prepare_page_rank_contribution(
                                workspace.current_rank(), view.owned_out_weight,
                                workspace.local_contribution(),
                                view.owned_vertex_count, workspace.scalars(),
                                streams.compute());
                            if (!shared->collective) {
                                check_cuda(
                                    cudaMemcpyAsync(
                                        &shared->local_dangling[rank],
                                        workspace.scalars(), sizeof(double),
                                        cudaMemcpyDeviceToHost, streams.compute()),
                                    "copy PageRank dangling sum");
                            }
                            check_cuda(
                                cudaStreamSynchronize(streams.compute()),
                                "synchronize PageRank contribution");
                            state.kernel_seconds += finish_cuda_timer(
                                contribution_timer, streams.compute());

                            const SteadyClock::time_point communication_begin =
                                SteadyClock::now();
                            reduce_scalar(
                                shared, workspace, streams, worker_device,
                                rank, true);
                            if (use_peer) {
                                shared->barrier.arrive_and_wait();
                                if (use_peer_gather) {
                                    for (std::size_t owner = 0; owner < devices.size(); ++owner) {
                                        const std::size_t begin = upload.halo_owner_offsets[owner];
                                        const std::size_t count = upload.halo_owner_offsets[owner + 1U] - begin;
                                        if (count == 0) continue;
                                        gpu_easygraph::partitioned::gather_peer_page_rank_halo(
                                            shared->rank_states[owner].workspace->local_contribution(),
                                            view.halo_source_local + begin,
                                            view.halo_destination_local + begin, count,
                                            workspace.local_contribution(), streams.transfer());
                                    }
                                } else {
                                for (std::size_t index = 0;
                                     index < (*halo_state)[rank].size(); ++index) {
                                    const HaloCopy& copy = (*halo_state)[rank][index];
                                    const PartitionWorkspace& owner_workspace =
                                        *shared->rank_states[
                                             static_cast<std::size_t>(copy.owner_rank)]
                                             .workspace;
                                    gpu_easygraph::runtime::peer_copy_async(
                                        workspace.local_contribution() +
                                            copy.destination_local,
                                        worker_device,
                                        owner_workspace.local_contribution() +
                                            copy.owner_local,
                                        copy.owner_device,
                                        sizeof(double), streams.transfer());
                                }
                                }
                                check_cuda(
                                    cudaStreamSynchronize(streams.transfer()),
                                    "synchronize PageRank halo peer copies");
                            } else {
                                // Host staging transports only owned contributions.
                                // Each destination gathers its exact halo after every
                                // owner has published a contiguous host segment.
                                std::vector<double>& owner_values =
                                    shared->rank_states[rank].staged_contribution;
                                owner_values.resize(view.owned_vertex_count);
                                if (!owner_values.empty()) {
                                    check_cuda(
                                        cudaMemcpy(
                                            owner_values.data(),
                                            workspace.local_contribution(),
                                            owner_values.size() * sizeof(double),
                                            cudaMemcpyDeviceToHost),
                                        "stage PageRank owned contributions");
                                }
                                shared->barrier.arrive_and_wait();
                                std::vector<double> halo((*halo_state)[rank].size());
                                for (std::size_t index = 0;
                                     index < (*halo_state)[rank].size(); ++index) {
                                    const HaloCopy& copy = (*halo_state)[rank][index];
                                    halo[index] = shared->rank_states[
                                        static_cast<std::size_t>(copy.owner_rank)]
                                        .staged_contribution[copy.owner_local];
                                }
                                if (!halo.empty()) {
                                    check_cuda(
                                        cudaMemcpy(
                                            workspace.local_contribution() +
                                                view.owned_vertex_count,
                                            halo.data(), halo.size() * sizeof(double),
                                            cudaMemcpyHostToDevice),
                                        "copy staged PageRank halo");
                                }
                            }
                            shared->barrier.arrive_and_wait();
                            state.communication_seconds += elapsed_seconds(
                                communication_begin, SteadyClock::now());

                            const double vertex_count =
                                static_cast<double>(vertex_count_value);
                            const double base = (1.0 - alpha) / vertex_count +
                                alpha * shared->global_dangling / vertex_count;
                            CudaEventTimer pull_timer;
                            check_cuda(
                                cudaEventRecord(pull_timer.begin, streams.compute()),
                                "record PageRank pull start");
                            gpu_easygraph::partitioned::pull_page_rank_rows(
                                view.offsets, view.indices, view.weights,
                                workspace.local_contribution(),
                                workspace.current_rank(), workspace.next_rank(),
                                view.owned_vertex_count, alpha, base,
                                workspace.scalars() + 1U, streams.compute(),
                                view.pagerank_heavy_rows, view.pagerank_heavy_row_count);
                            if (!shared->collective) {
                                check_cuda(
                                    cudaMemcpyAsync(
                                        &shared->local_residual[rank],
                                        workspace.scalars() + 1U, sizeof(double),
                                        cudaMemcpyDeviceToHost, streams.compute()),
                                    "copy PageRank residual");
                            }
                            check_cuda(
                                cudaStreamSynchronize(streams.compute()),
                                "synchronize PageRank pull");
                            state.kernel_seconds += finish_cuda_timer(
                                pull_timer, streams.compute());

                            const SteadyClock::time_point reduction_begin =
                                SteadyClock::now();
                            reduce_scalar(
                                shared, workspace, streams, worker_device,
                                rank, false);
                            state.communication_seconds += elapsed_seconds(
                                reduction_begin, SteadyClock::now());
                            workspace.advance_rank();
                            if (rank == 0) {
                                shared->iterations = iteration;
                                shared->converged =
                                    shared->global_residual <=
                                    vertex_count * tolerance;
                            }
                            shared->barrier.arrive_and_wait();
                            if (shared->converged) break;
                        }

                        const SteadyClock::time_point gather_begin =
                            SteadyClock::now();
                        state.owned_result.resize(view.owned_vertex_count);
                        if (!state.owned_result.empty()) {
                            check_cuda(
                                cudaMemcpy(
                                    state.owned_result.data(),
                                    workspace.current_rank(),
                                    state.owned_result.size() * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                                "gather partitioned PageRank owned ranks");
                        }
                        state.output_gather_seconds = elapsed_seconds(
                            gather_begin, SteadyClock::now());
                    });
                task_state.task_exit_offset = elapsed_seconds(
                    shared->dispatch_begin, SteadyClock::now());
            });
    }
    const SteadyClock::time_point device_begin = SteadyClock::now();
    shared->dispatch_begin = device_begin;
    auto gil_released = device_begin;
    auto gang_returned = device_begin;
    {
        py::gil_scoped_release release;
        gil_released = SteadyClock::now();
        gpu_easygraph::runtime::runtime_manager().execute_gang(
            devices, std::move(tasks));
        gang_returned = SteadyClock::now();
    }
    const auto device_end = SteadyClock::now();
    const double cooperative_execution_seconds = elapsed_seconds(
        device_begin, device_end);

    const SteadyClock::time_point merge_begin = SteadyClock::now();
    std::shared_ptr<PageRankHostResult> result = worker_result;
    if (!result) result = std::make_shared<PageRankHostResult>(
        static_cast<std::size_t>(graph->vertex_count()), numpy_output_requested, output_mode != "copy");
    double kernel_seconds = 0.0;
    double communication_seconds = 0.0;
    double device_processing_seconds = 0.0;
    double state_upload_seconds = 0.0;
    double output_gather_seconds = 0.0;
    std::uint64_t resident_bytes = 0;
    std::vector<std::uint64_t> resident_device_bytes;
    resident_device_bytes.reserve(partitions.size());
    bool device_view_cache_hit = true;
    std::uint64_t shared_array_hits = 0;
    std::uint64_t shared_array_new_bytes = 0;
    std::uint64_t shared_array_retained_bytes = 0;
    std::vector<std::uint64_t> shared_array_retained_device_bytes;
    shared_array_retained_device_bytes.reserve(partitions.size());
    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        RankState& state = shared->rank_states[rank];
        if (!worker_result) {
            const auto output_begin = SteadyClock::now();
            easygraph::egmgpu::owner_output::observe_original(state.host_output_threads);
            result->assign(partitions[rank], state.result_data());
            state.host_output_assembly_seconds = elapsed_seconds(output_begin, SteadyClock::now());
        }
        kernel_seconds = std::max(kernel_seconds, state.kernel_seconds);
        communication_seconds = std::max(
            communication_seconds, state.communication_seconds);
        device_processing_seconds = std::max(
            device_processing_seconds,
            streamed ? state.initialization_wall + state.contribution_wall +
                state.halo_wall + state.pull_wall + state.reduction_wall :
                state.kernel_seconds + state.communication_seconds);
        state_upload_seconds = std::max(
            state_upload_seconds, state.state_upload_seconds);
        output_gather_seconds = std::max(
            output_gather_seconds, state.output_gather_seconds);
        resident_bytes += static_cast<std::uint64_t>(state.resident_bytes);
        resident_device_bytes.push_back(
            static_cast<std::uint64_t>(state.resident_bytes));
        device_view_cache_hit =
            device_view_cache_hit && state.device_view_cache_hit;
        shared_array_hits += static_cast<std::uint64_t>(
            state.shared_arrays.hits);
        shared_array_new_bytes += static_cast<std::uint64_t>(
            state.shared_arrays.new_bytes);
        shared_array_retained_bytes += static_cast<std::uint64_t>(
            state.shared_arrays.retained_bytes);
        shared_array_retained_device_bytes.push_back(
            static_cast<std::uint64_t>(state.shared_arrays.retained_bytes));
    }
    const double native_merge_seconds = elapsed_seconds(
        merge_begin, SteadyClock::now());

    std::uint64_t halo_values = 0;
    for (std::size_t rank = 0; rank < halo_plan.size(); ++rank) {
        halo_values += static_cast<std::uint64_t>(halo_plan[rank].size());
    }
    const std::uint64_t iteration_count =
        static_cast<std::uint64_t>(shared->iterations);
    const std::uint64_t scalar_bytes_per_iteration =
        static_cast<std::uint64_t>(partitions.size()) *
        (streamed && !prepared_state->has_dangling ? 1U : 2U) * sizeof(double);
    const std::uint64_t peer_to_peer_bytes = use_peer
        ? iteration_count * halo_values * sizeof(double)
        : 0U;
    const std::uint64_t host_staging_bytes = nccl_halo_active || shared->skip_nccl_halo || use_peer || devices.size() == 1U
        ? 0U
        : iteration_count *
              (graph->vertex_count() * sizeof(double) +
               halo_values * sizeof(double));
    const std::uint64_t collective_payload_bytes = collective
        ? iteration_count * scalar_bytes_per_iteration
        : 0U;
    const std::uint64_t host_scalar_bytes = collective
        ? 0U
        : iteration_count * scalar_bytes_per_iteration;
    const std::uint64_t nccl_halo_remote_logical_bytes = nccl_halo_active
        ? iteration_count * (sparse_halo_requested ? global_halo_count : (devices.size()-1U)*halo_receive_values) * sizeof(double) : 0U;
    const std::uint64_t communication_bytes = peer_to_peer_bytes +
        host_staging_bytes + collective_payload_bytes + host_scalar_bytes + nccl_halo_remote_logical_bytes;

    const auto materialize_begin = SteadyClock::now();
    py::array_t<double> values = result->materialize(output_mode != "copy");
    const double native_materialize_seconds = elapsed_seconds(materialize_begin, SteadyClock::now());
    py::dict out;
    out["values_dense"] = values;
    out["output_mode"] = output_mode;
    out["result_assembly"] = result->direct_assembly ? "direct" : "zeroed";
    out["pr_native_output_ownership"] = easygraph::egmgpu::pr_host_output_proof(
        numpy_output_requested, result->direct_assembly, output_mode != "copy", graph->vertex_count());
    out["host_pinned_output_prepare_seconds"] = output_lease.preparation_wall;
    out["host_pinned_output_cache_hit"] = output_lease.cache_hit;
    out["host_pinned_output_new_bytes"] = output_lease.new_bytes;
    out["host_pinned_output_placement"] = output_lease.placement;
    out["iterations"] = shared->iterations;
    out["converged"] = shared->converged;
    out["residual"] = shared->global_residual;
    out["devices_used"] = devices;
    const CommunicationBackend observed_backend = devices.size() == 1U
        ? CommunicationBackend::None : (collective || nccl_halo_active) ? CommunicationBackend::Nccl
        : use_peer ? CommunicationBackend::PeerToPeer : CommunicationBackend::HostStaging;
    out["communication_backend"] = std::string(
        gpu_easygraph::runtime::communication_backend_name(observed_backend));
    out["runtime_communication_backend"] = std::string(
        gpu_easygraph::runtime::communication_backend_name(active_group.backend()));
    out["scalar_reduction_policy"] = scalar_policy;
    out["scalar_reduction_backend"] = devices.size() == 1U ? "local" : collective ? "nccl" : "host";
    out["collective_context_used"] = static_cast<bool>(acquired_collective);
    out["collective_context_cache_hit"] = collective_context_cache_hit;
    out["collective_context_acquire_seconds"] = collective_context_acquire_seconds;
    out["requested_halo_exchange"] = halo_exchange;
    out["host_halo_active"] = host_halo_active;
    out["host_halo_plan_cache_hit"] = host_halo_active && host_halo_plan_cache_hit;
    out["host_halo_plan_validation_seconds"] = host_halo_plan_seconds;
    out["host_halo_pinned_cache_hit"] = host_halo_active && halo_lease.cache_hit;
    out["host_halo_pinned_prepare_seconds"] = halo_lease.preparation_wall;
    out["host_halo_pinned_required_bytes"] = halo_lease.required_bytes;
    out["host_halo_pinned_live_bytes"] = halo_lease.capacity_bytes;
    out["host_halo_pinned_capacity_bytes"] = halo_lease.capacity_bytes;
    out["host_halo_pinned_new_bytes"] = halo_lease.new_bytes;
    out["host_halo_pinned_replaced_bytes"] = halo_lease.replaced_bytes;
    out["host_halo_pinned_prepare_peak_bytes"] = halo_lease.allocation_peak_bytes;
    out["host_halo_requested_threads_per_rank"] = host_halo_width;
    out["host_halo_total_thread_budget"] = host_halo_width*devices.size();
    out["host_halo_caller_allowed_cpus"] = easygraph::egmgpu::owner_output::cpu_list(host_halo_caller_mask);
    out["host_halo_wait_policy"] = host_halo_active ? "PASSIVE" : "not_active";
    out["host_halo_index_policy"] = host_halo_active ? "existing_owner_grouped_int32" : "not_active";
    out["host_halo_pinned_scope"] = "Retained host pinned owned/halo buffers, capacity reused by device group under query mutex. GPU-NUMA allocation with calling mask restored. Required is current values; live/capacity include unused retained capacity. Preparation peak counts explicit pinned allocation sizes, excludes CUDA metadata/output/NCCL buffers.";
    out["host_halo_thread_observation_scope"] = "Last iteration CPU masks; actual width and affinity errors checked every iteration. Sum per-rank requested width is 16; phase durations overlap across ranks.";
    out["halo_exchange"] = host_halo_requested
        ? (host_halo_active ? "host_pinned_parallel" : devices.size()==1U ? "local_skip" : "global_no_halo_skip")
        : nccl_halo_requested
        ? (nccl_halo_active ? halo_exchange.c_str() : devices.size() == 1U ? "local_skip" : "global_no_halo_skip")
        : (use_peer ? halo_exchange : "host_staging");
    out["nccl_halo_active"] = nccl_halo_active;
    out["nccl_halo_stride"] = nccl_halo_active && !variable_halo_requested && !sparse_halo_requested ? halo_stride : 0U;
    out["nccl_halo_collective"] = nccl_halo_active ? (sparse_halo_requested ? "grouped_send_recv" : variable_halo_requested ? "grouped_broadcast" : "allgather") : "inactive";
    out["nccl_halo_layout"] = nccl_halo_active ? (sparse_halo_requested ? "direct_existing_halo_runs" : variable_halo_requested ? "packed_owned_prefix" : "padded_equal_stride") : "inactive";
    out["nccl_halo_owned_counts"] = owned_counts;
    out["nccl_halo_slot_offsets"] = sparse_halo_requested ? std::vector<std::size_t>() : shared->halo_slot_offsets;
    out["nccl_halo_receive_values_per_device"] = nccl_halo_active ? halo_receive_values : 0U;
    out["nccl_halo_collective_groups_per_rank"] = nccl_halo_active && !sparse_halo_requested ? iteration_count : 0U;
    out["nccl_halo_padding_values_per_iteration"] = nccl_halo_active && !sparse_halo_requested
        ? halo_receive_values - graph->vertex_count() : 0U;
    out["nccl_halo_buffer_bytes_per_device"] = nccl_halo_active ? halo_receive_values * sizeof(double) : 0U;
    out["nccl_halo_aggregate_buffer_bytes"] = nccl_halo_active ? devices.size() * halo_receive_values * sizeof(double) : 0U;
    out["nccl_halo_payload_input_bytes"] = nccl_halo_active ? iteration_count * (sparse_halo_requested ? global_halo_count : halo_receive_values) * sizeof(double) : 0U;
    out["nccl_halo_payload_output_bytes"] = nccl_halo_active ? iteration_count * (sparse_halo_requested ? global_halo_count : devices.size()*halo_receive_values) * sizeof(double) : 0U;
    out["nccl_halo_remote_logical_bytes"] = nccl_halo_remote_logical_bytes;
    out["nccl_halo_d2d_owned_pack_bytes"] = nccl_halo_active && !sparse_halo_requested ? iteration_count * graph->vertex_count() * sizeof(double) : 0U;
    out["nccl_halo_gather_read_bytes"] = nccl_halo_active && !sparse_halo_requested ? iteration_count * global_halo_count * sizeof(double) : 0U;
    out["nccl_halo_gather_write_bytes"] = nccl_halo_active && !sparse_halo_requested ? iteration_count * global_halo_count * sizeof(double) : 0U;
    out["nccl_halo_collective_calls_per_rank"] = nccl_halo_active && !sparse_halo_requested ? iteration_count * (variable_halo_requested ? packed_layout.positive_ranks : 1U) : 0U;
    std::vector<std::size_t> sparse_send_values,sparse_receive_values,sparse_map_bytes,sparse_send_bytes,sparse_scratch_bytes,sparse_send_runs,sparse_receive_runs;
    std::vector<std::uint64_t> sparse_groups;
    std::vector<std::size_t> sparse_map_h2d,sparse_map_new,sparse_send_new,sparse_query_local;
    std::vector<bool> sparse_map_hits,sparse_send_hits;
    std::size_t sparse_scratch_sum=0,sparse_scratch_max=0,sparse_maps_sum=0,sparse_upload_sum=0,sparse_query_local_sum=0,sparse_retained_sum=0,sparse_transient_workspace_sum=0;
    for (std::size_t rank=0;rank<devices.size();++rank) {
        const auto* r=sparse_layout ? &sparse_layout->ranks[rank] : nullptr;
        const std::size_t send=r ? r->source_indices.size() : 0U,recv=r ? r->receive_values : 0U;
        sparse_send_values.push_back(send);sparse_receive_values.push_back(recv);
        const auto& observed=shared->rank_states[rank];
        sparse_map_bytes.push_back(observed.nccl_sparse_map_bytes);sparse_send_bytes.push_back(observed.nccl_sparse_send_buffer_bytes);
        sparse_scratch_bytes.push_back(observed.nccl_sparse_scratch_bytes);
        sparse_map_h2d.push_back(observed.nccl_sparse_map_h2d_bytes);sparse_upload_sum+=sparse_map_h2d.back();
        sparse_map_new.push_back(observed.nccl_sparse_map_new_bytes);sparse_send_new.push_back(observed.nccl_sparse_send_new_bytes);
        sparse_query_local.push_back(observed.nccl_sparse_query_local_bytes);sparse_query_local_sum+=sparse_query_local.back();
        if (retain_sparse_state) {
            sparse_retained_sum+=observed.nccl_sparse_map_bytes;
            if (observed.sparse_workspace_retained) sparse_retained_sum+=observed.nccl_sparse_send_buffer_bytes;
            else sparse_transient_workspace_sum+=observed.nccl_sparse_send_buffer_bytes;
        }
        sparse_map_hits.push_back(observed.nccl_sparse_map_cache_hit);sparse_send_hits.push_back(observed.nccl_sparse_send_cache_hit);
        sparse_scratch_sum+=sparse_scratch_bytes.back();sparse_scratch_max=std::max(sparse_scratch_max,sparse_scratch_bytes.back());sparse_maps_sum+=sparse_map_bytes.back();
        sparse_send_runs.push_back(r ? r->sends.size() : 0U);sparse_receive_runs.push_back(r ? r->receives.size() : 0U);
        sparse_groups.push_back(r && (!r->sends.empty() || !r->receives.empty()) ? iteration_count : 0U);
    }
    out["nccl_sparse_active"]=nccl_halo_active && sparse_halo_requested;
    out["nccl_sparse_plan_seconds"]=sparse_plan_seconds;
    out["nccl_sparse_plan_cache_hit"]=sparse_plan_cache_hit;
    out["nccl_sparse_state_policy"]=sparse_state;
    out["nccl_sparse_plan_key"]=shared->sparse_plan_key;
    out["nccl_sparse_map_h2d_bytes_by_rank"]=sparse_map_h2d;
    out["nccl_sparse_map_new_bytes_by_rank"]=sparse_map_new;
    out["nccl_sparse_send_new_bytes_by_rank"]=sparse_send_new;
    out["nccl_sparse_map_cache_hits"]=sparse_map_hits;out["nccl_sparse_send_cache_hits"]=sparse_send_hits;
    out["nccl_sparse_query_local_bytes_by_rank"]=sparse_query_local;
    out["nccl_sparse_query_local_sum_bytes"]=sparse_query_local_sum;
    out["nccl_sparse_current_retained_bytes"]=sparse_retained_sum;
    out["nccl_sparse_transient_workspace_bytes"]=sparse_transient_workspace_sum;
    out["nccl_sparse_plan_host_payload_bytes"]=sparse_layout ? sparse_layout->host_payload_bytes : 0U;
    out["nccl_sparse_validation_bitmap_peak_bytes"]=sparse_layout ? sparse_layout->validation_peak_bytes : 0U;
    out["nccl_sparse_send_values_by_rank"]=sparse_send_values;out["nccl_sparse_receive_values_by_rank"]=sparse_receive_values;
    out["nccl_sparse_send_runs_by_rank"]=sparse_send_runs;out["nccl_sparse_receive_runs_by_rank"]=sparse_receive_runs;
    out["nccl_sparse_groups_by_rank"]=sparse_groups;
    out["nccl_sparse_map_bytes_by_rank"]=sparse_map_bytes;out["nccl_sparse_send_buffer_bytes_by_rank"]=sparse_send_bytes;
    out["nccl_sparse_scratch_bytes_by_rank"]=sparse_scratch_bytes;
    out["nccl_sparse_scratch_live_sum_bytes"]=sparse_scratch_sum;out["nccl_sparse_scratch_peak_per_device_bytes"]=sparse_scratch_max;
    out["nccl_sparse_map_h2d_bytes_per_call"]=sparse_upload_sum;
    out["nccl_sparse_pack_logical_read_bytes"]=sparse_layout ? iteration_count*global_halo_count*sizeof(double) : 0U;
    out["nccl_sparse_pack_logical_write_bytes"]=sparse_layout ? iteration_count*global_halo_count*sizeof(double) : 0U;
    out["nccl_sparse_direct_receive_logical_write_bytes"]=sparse_layout ? iteration_count*global_halo_count*sizeof(double) : 0U;
    out["nccl_sparse_transfer_runs"]=sparse_layout ? sparse_layout->transfer_runs : 0U;
    out["nccl_sparse_scope"]="call mode creates a CPU transpose and two int32 GPU maps plus f64 send buffer per call. retained mode keeps the immutable host plan under the full snapshot/view key, maps under the PR view owner, and mutable send capacity under the exclusive per_worker workspace lease; active leases use a transient workspace. Retained maps/workspace appear in unique-allocation inventory and ordinary eviction/close. New/reused capacities and actual H2D are separate. No cross-weight-generation sharing claim. NCCL writes existing halo slots directly. Logical send volume is 8H per round, not PCIe traffic. Sizes exclude NCCL internals/allocator backing/process VRAM; retained budget is not a total VRAM hard cap.";
    out["nccl_halo_buffer_scope"] = "All-gather receive scratch and sparse call-mode maps/send values are query-local and extra to retained inventory. Sparse retained-mode maps and mutable send capacity belong to view/workspace owners, are included in unique-allocation inventory, and release at idle eviction/close. Current view/workspace-plus-query-local bytes never double count retained maps/send. Explicit capacities exclude NCCL internal scratch, allocator backing and total physical VRAM peak.";
    out["communication_bytes_scope"] = "Algorithm payload accounting; NCCL remote logical delivery includes padding only for equal-stride AllGather; variable gather publishes all actual owned values to all ranks; sparse halo sends only requested halo values. These are not physical wire/PCIe bytes. NCCL may internally stage through host. Our explicit owned-D2H/CPU-halo-gather/H2D path is bypassed only when nccl_halo_active.";
    out["execution_schedule"] = schedule;
    out["device_phase_profiling"] = streamed && profile;
    out["has_dangling_vertices"] = prepared_state->has_dangling;
    out["preparation_method"] = prepared_state->preparation_method;
    out["csr_shared_across_functions"] = !prepared_state->uploads.front().topology_compatibility.empty();
    out["native_materialize_seconds"] = native_materialize_seconds;
    out["core_output_team_version"] = "stage4_core_owner_output_v1";
    out["core_output_team_policy"] = output_team16 ? "team16" : "original";
    // Original PR copies on one caller; team16 copies concurrently on owners.
    out["core_output_workers"] = output_team16 ? devices.size() : 1U;
    out["core_output_requested_per_worker"] = output_team_width;
    out["core_output_thread_budget"] = output_team16 ? devices.size() * output_team_width : 1U;
    out["host_output_allocation_seconds"] = host_output_allocation_seconds;
    out["host_output_index_bytes"] = 0U;
    py::list output_observations;
    for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
        auto observation = easygraph::egmgpu::owner_output::to_python(
            shared->rank_states[rank].host_output_threads);
        observation["rank"] = rank;
        observation["host_output_bytes"] = partitions[rank].owned_vertex_count * sizeof(double);
        observation["assembly_seconds"] = shared->rank_states[rank].host_output_assembly_seconds;
        output_observations.append(observation);
    }
    out["core_output_observations"] = output_observations;
    if (streamed) {
        py::list stages;
        for (std::size_t rank = 0; rank < partitions.size(); ++rank) {
            const auto& s = shared->rank_states[rank];
            py::dict item;
            item["device"] = devices[rank];
            item["state_acquire_wall"] = s.state_upload_seconds;
            item["initialization_wall"] = s.initialization_wall;
            item["contribution_wall"] = s.contribution_wall;
            item["halo_wall"] = s.halo_wall;
            item["nccl_halo_scratch_prepare_wall"] = s.nccl_halo_scratch_prepare_wall;
            item["nccl_halo_scratch_release_wall"] = s.nccl_halo_scratch_release_wall;
            item["nccl_halo_pack_wall"] = s.nccl_halo_pack_wall;
            item["nccl_halo_publication_wall"] = s.nccl_halo_publication_wall;
            item["nccl_halo_collective_wall"] = s.nccl_halo_collective_wall;
            item["nccl_halo_gather_wall"] = s.nccl_halo_gather_wall;
            item["nccl_halo_scratch_bytes"] = s.nccl_halo_scratch_bytes;
            item["nccl_sparse_send_buffer_bytes"]=s.nccl_sparse_send_buffer_bytes;
            item["nccl_sparse_map_bytes"]=s.nccl_sparse_map_bytes;
            item["nccl_sparse_scratch_bytes"]=s.nccl_sparse_scratch_bytes;
            item["nccl_sparse_query_local_bytes"]=s.nccl_sparse_query_local_bytes;
            item["nccl_sparse_workspace_retained"]=retain_sparse_state && s.sparse_workspace_retained;
            item["nccl_sparse_transient_workspace_bytes"]=retain_sparse_state && !s.sparse_workspace_retained ? s.nccl_sparse_send_buffer_bytes : 0U;
            item["nccl_sparse_map_new_bytes"]=s.nccl_sparse_map_new_bytes;
            item["nccl_sparse_map_h2d_bytes"]=s.nccl_sparse_map_h2d_bytes;
            item["nccl_sparse_send_new_bytes"]=s.nccl_sparse_send_new_bytes;
            item["nccl_sparse_map_cache_hit"]=s.nccl_sparse_map_cache_hit;
            item["nccl_sparse_send_cache_hit"]=s.nccl_sparse_send_cache_hit;
            item["current_view_workspace_plus_nccl_scratch_bytes"] = s.resident_bytes + s.nccl_halo_scratch_bytes + s.nccl_sparse_query_local_bytes;
            item["host_halo_publication_wall"] = s.host_halo_publication_wall;
            item["host_halo_threads"] = easygraph::egmgpu::owner_output::to_python(s.host_halo_threads);
            item["host_halo_owned_pinned_capacity_bytes"] = s.pinned_contribution ? s.pinned_contribution->size() : 0U;
            item["host_halo_received_pinned_capacity_bytes"] = s.pinned_halo ? s.pinned_halo->size() : 0U;
            item["staging_owned_d2h_wall"] = s.staging_owned_d2h_wall;
            item["staging_cpu_gather_wall"] = s.staging_cpu_gather_wall;
            item["staging_halo_h2d_wall"] = s.staging_halo_h2d_wall;
            item["pull_wall"] = s.pull_wall;
            item["reduction_wall"] = s.reduction_wall;
            item["output_wall"] = s.output_gather_seconds;
            item["host_output_assembly_wall"] = s.host_output_assembly_seconds;
            item["host_output_within_gang"] = output_team16;
            if (profile) {
                item["allocation_wall"] = s.device_state_profile.malloc_wall_seconds;
                item["allocation_bytes"] = s.device_state_profile.malloc_bytes;
                item["allocation_count"] = s.device_state_profile.malloc_count;
                item["h2d_wall"] = s.device_state_profile.h2d_wall_seconds;
                item["h2d_device"] = s.device_state_profile.h2d_device_seconds;
                item["h2d_bytes"] = s.device_state_profile.h2d_bytes;
                item["h2d_count"] = s.device_state_profile.h2d_count;
                item["kernel_init"] = s.kernel_init;
                item["kernel_contribution"] = s.kernel_contribution;
                item["kernel_pull"] = s.kernel_pull;
                item["halo_device"] = s.halo_device;
                item["scalar_d2h_device"] = s.scalar_d2h_device;
                item["output_d2h_device"] = s.output_d2h_device;
            }
            stages.append(item);
        }
        out["rank_stage_seconds"] = stages;
        out["timing_contract"] = "rank wall phases are disjoint; staging detail is nested within halo_wall; host output assembly is within gang only for team16 and within caller merge for original; compute CUDA events exclude scalar copies; halo stream elapsed can include host gaps; cross-rank maxima are not additive";
    }
    const char* pull_setting = std::getenv("EGMGPU_PAGERANK_PULL");
    out["pull_row_policy"] = pull_setting ? pull_setting : "split";
    out["halo_values_per_iteration"] = halo_values;
    std::uint64_t halo_submissions = 0U;
    for (const PartitionDeviceUpload& upload : prepared_state->uploads) {
        for (std::size_t owner = 0; owner < devices.size(); ++owner) {
            halo_submissions += upload.halo_owner_offsets[owner + 1U] > upload.halo_owner_offsets[owner];
        }
    }
    out["peer_halo_submissions_per_iteration"] = use_peer
        ? (use_peer_gather ? halo_submissions : halo_values) : 0U;
    out["state_preparation_seconds"] = state_preparation_seconds;
    out["host_view_cache_hit"] = host_view_cache_hit;
    add_host_plan_metrics(out, *prepared_state);
    add_device_projection_metrics(out, shared->rank_states);
    out["device_view_cache_hit"] = device_view_cache_hit;
    out["state_upload_seconds"] = state_upload_seconds;
    out["kernel_seconds"] = kernel_seconds;
    if (streamed && !profile) out["kernel_seconds"] = py::none();
    out["device_processing_seconds"] = device_processing_seconds;
    out["device_processing_timing_domain"] = streamed
        ? "maximum per-rank sum of disjoint execution wall phases; excludes state acquisition and output"
        : "legacy per-rank kernel-event plus communication-wall sum";
    out["cooperative_execution_seconds"] = cooperative_execution_seconds;
    py::dict lifecycle;
    lifecycle["gil_release_wall"] = elapsed_seconds(device_begin, gil_released);
    lifecycle["execute_gang_wall"] = elapsed_seconds(gil_released, gang_returned);
    lifecycle["gil_reacquire_wall"] = elapsed_seconds(gang_returned, device_end);
    lifecycle["time_origin"] = "cooperative execution begin; per-rank offsets share the caller steady clock";
    py::list rank_lifecycle;
    for (std::size_t rank = 0; rank < shared->rank_states.size(); ++rank) {
        const auto& state = shared->rank_states[rank];
        py::dict item;
        item["rank"] = rank;
        item["device"] = devices[rank];
        item["task_enter_offset"] = state.task_enter_offset;
        item["task_exit_offset"] = state.task_exit_offset;
        item["task_wall"] = state.task_exit_offset - state.task_enter_offset;
        item["peer_enable_wall"] = state.peer_enable_wall;
        item["state_acquire_wall"] = state.state_upload_seconds;
        item["streams_acquire_wall"] = state.streams_acquire_wall;
        rank_lifecycle.append(item);
    }
    lifecycle["ranks"] = rank_lifecycle;
    out["cooperative_lifecycle_seconds"] = lifecycle;
    out["communication_seconds"] = communication_seconds;
    out["output_gather_seconds"] = output_gather_seconds;
    out["native_merge_seconds"] = native_merge_seconds;
    out["communication_bytes"] = communication_bytes;
    out["peer_to_peer_bytes"] = peer_to_peer_bytes;
    out["host_staging_bytes"] = host_staging_bytes;
    out["collective_payload_bytes"] = collective_payload_bytes;
    out["host_scalar_bytes"] = host_scalar_bytes;
    out["resident_device_bytes"] = resident_device_bytes;
    out["aggregate_resident_device_bytes"] = resident_bytes;
    out["shared_array_hits"] = shared_array_hits;
    out["shared_array_new_bytes"] = shared_array_new_bytes;
    out["shared_array_retained_bytes"] = shared_array_retained_bytes;
    out["shared_array_retained_device_bytes"] =
        shared_array_retained_device_bytes;
    return out;
}


// Qualification-only: retain two concurrent workspace leases on each fixed
// worker, verify distinct mutable storage, and attempt eviction while held.
// This diagnostic never runs inside a numerical or performance measurement.
py::dict probe_sparse_retained_leases(const std::shared_ptr<MultiPartitionGraph>& graph,
                                    py::object device_request) {
    if (!graph || !gpu_easygraph::runtime::managed_workspace_per_worker())
        throw std::invalid_argument("probe requires a graph and per_worker policy");
    const auto devices=requested_devices(*graph,device_request);
    bool host_hit=false;auto prepared=prepare_pull_state(graph,devices,&host_hit);
    py::dict out;out["schema"]="eggpu.sparse_retained_lease_probe.v1";
    if (!prepared->retained_sparse_halo_plan) {
        if (devices.size()!=1) {
            for (const auto& plan:prepared->halo_plan) if (!plan.empty())
                throw std::invalid_argument("probe requires a completed retained sparse query");
        }
        out["status"]="SKIP_NO_HALOS";return out;
    }
    struct Proof {std::size_t protected_bytes=0,transient_bytes=0;bool distinct=false,
        unchanged=false,skipped_view=false,skipped_workspace=false,rejected_key=false;};
    auto proofs=std::make_shared<std::vector<Proof>>(devices.size());
    const auto key=prepared_state_key(*graph,devices);
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    for (std::size_t rank=0;rank<devices.size();++rank) tasks.push_back(
        [prepared,proofs,key,rank](int worker_device) {
            auto& registry=gpu_easygraph::runtime::partitioned_device_state_registry();
            bool hit=false;auto view=registry.acquire(prepared->uploads[rank],&hit);
            if (!hit || worker_device!=prepared->uploads[rank].device_id)
                throw std::runtime_error("probe expected resident warmed view");
            auto& streams=gpu_easygraph::runtime::worker_streams();
            const auto& plan=prepared->retained_sparse_halo_plan->ranks[rank];
            auto maps=view->ensure_sparse_halo_maps(key,plan.source_indices,plan.packed_indices,streams.transfer());
            if (!maps.cache_hit) throw std::runtime_error("probe expected retained map hit");
            auto& proof=(*proofs)[rank];const auto before_bytes=view->resident_bytes();
            try {view->ensure_sparse_halo_maps(key+"/wrong",plan.source_indices,plan.packed_indices,streams.transfer());}
            catch (const std::invalid_argument&) {proof.rejected_key=true;}
            if (!proof.rejected_key || view->resident_bytes()!=before_bytes)
                throw std::runtime_error("mismatched immutable plan published state");
            auto first=registry.workspace(view);
            if (first->ensure_sparse_send_capacity(plan.source_indices.size()))
                throw std::runtime_error("probe expected warmed pooled send capacity");
            proof.protected_bytes=first->sparse_send_capacity_bytes();
            std::vector<unsigned char> before(proof.protected_bytes),after(proof.protected_bytes);
            if (proof.protected_bytes) check_cuda(cudaMemcpy(before.data(),first->sparse_send(),before.size(),cudaMemcpyDeviceToHost),"probe save held send values");
            auto second=registry.workspace(view);
            second->ensure_sparse_send_capacity(plan.source_indices.size());
            proof.transient_bytes=second->sparse_send_capacity_bytes();
            proof.distinct=first.get()!=second.get() && (!proof.protected_bytes || first->sparse_send()!=second->sparse_send());
            if (proof.transient_bytes) check_cuda(cudaMemset(second->sparse_send(),0x42,proof.transient_bytes),"probe write transient send");
            check_cuda(cudaDeviceSynchronize(),"probe synchronize temporary write");
            if (proof.protected_bytes) check_cuda(cudaMemcpy(after.data(),first->sparse_send(),after.size(),cudaMemcpyDeviceToHost),"probe reread held send");
            proof.unchanged=before==after;
            gpu_easygraph::runtime::ManagedEvictRecord evicted;
            registry.managed_evict({view->managed_owner_id(),first->managed_owner_id()},false,evicted);
            proof.skipped_view=std::find(evicted.skipped_active.begin(),evicted.skipped_active.end(),view->managed_owner_id())!=evicted.skipped_active.end();
            proof.skipped_workspace=std::find(evicted.skipped_active.begin(),evicted.skipped_active.end(),first->managed_owner_id())!=evicted.skipped_active.end();
            if (!proof.distinct || !proof.unchanged || !proof.skipped_view || !proof.skipped_workspace)
                throw std::runtime_error("active sparse lease qualification failed");
        });
    {py::gil_scoped_release release;gpu_easygraph::runtime::runtime_manager().execute_gang(devices,std::move(tasks));}
    py::list rows;
    for (std::size_t rank=0;rank<devices.size();++rank) {
        const auto& proof=(*proofs)[rank];py::dict row;row["device"]=devices[rank];
        row["protected_send_bytes"]=proof.protected_bytes;row["transient_send_bytes"]=proof.transient_bytes;
        row["distinct_buffers"]=proof.distinct;row["held_values_unchanged"]=proof.unchanged;
        row["active_view_eviction_skipped"]=proof.skipped_view;row["active_workspace_eviction_skipped"]=proof.skipped_workspace;
        row["wrong_plan_rejected_without_publication"]=proof.rejected_key;rows.append(row);
    }
    out["status"]="PASS_ACTIVE_LEASE_PROBE";out["ranks"]=rows;return out;
}

py::dict clear_partitioned_state(py::object device_request) {
    const gpu_easygraph::runtime::RuntimeStatus status =
        gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running) {
        throw std::runtime_error(
            "EGMGPU native runtime must be started before clearing device state");
    }
    std::vector<int> devices = device_request.is_none()
        ? status.device_ids
        : device_request.cast<std::vector<int> >();
    std::sort(devices.begin(), devices.end());
    if (devices.empty() ||
        std::adjacent_find(devices.begin(), devices.end()) != devices.end()) {
        throw std::invalid_argument(
            "partitioned state clear requires distinct devices");
    }
    struct ClearStats {
        std::size_t views;
        std::size_t workspaces;
        std::size_t shared_entries;
        std::size_t shared_bytes;
        std::size_t ego_states;
        std::size_t ego_workspaces;
        ClearStats()
            : views(0U),
              workspaces(0U),
              shared_entries(0U),
              shared_bytes(0U),
              ego_states(0U),
              ego_workspaces(0U) {}
    };
    const std::shared_ptr<std::vector<ClearStats> > released(
        new std::vector<ClearStats>(devices.size()));
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.reserve(devices.size());
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        tasks.push_back([released, rank](int) {
            gpu_easygraph::runtime::PartitionedDeviceStateRegistry& registry =
                gpu_easygraph::runtime::partitioned_device_state_registry();
            gpu_easygraph::runtime::PartitionedWccStateRegistry& wcc =
                gpu_easygraph::runtime::partitioned_wcc_state_registry();
            gpu_easygraph::runtime::PartitionedBfsStateRegistry& bfs =
                gpu_easygraph::runtime::partitioned_bfs_state_registry();
            gpu_easygraph::runtime::PartitionedSsspStateRegistry& sssp =
                gpu_easygraph::runtime::partitioned_sssp_state_registry();
            gpu_easygraph::runtime::PartitionedEgoStateRegistry& ego =
                gpu_easygraph::runtime::partitioned_ego_state_registry();
            ClearStats& stats = (*released)[rank];
            stats.ego_states = ego.resident_state_count();
            stats.ego_workspaces = ego.workspace_count();
            stats.views = registry.resident_view_count() +
                wcc.resident_view_count() + bfs.resident_view_count() +
                sssp.resident_view_count() + stats.ego_states;
            stats.workspaces = registry.workspace_count() +
                wcc.workspace_count() + bfs.workspace_count() +
                sssp.workspace_count() + stats.ego_workspaces;
            const gpu_easygraph::runtime::SharedArrayCatalogStats shared =
                gpu_easygraph::runtime::shared_device_array_catalog().stats();
            stats.shared_entries = shared.retained_entries;
            stats.shared_bytes = shared.retained_bytes;
            registry.clear();
            wcc.clear();
            bfs.clear();
            sssp.clear();
            ego.clear();
            gpu_easygraph::runtime::shared_device_array_catalog().clear();
        });
    }
    const std::size_t host_states = prepared_state_count();
    gpu_easygraph::runtime::PartitionedHostPlanCacheClearResult host_plans;
    {
        py::gil_scoped_release release;
        gpu_easygraph::runtime::runtime_manager().execute_gang(
            devices, std::move(tasks));
        clear_prepared_state_cache();
        host_plans =
            gpu_easygraph::runtime::clear_partitioned_host_plan_caches();
    }
    std::size_t device_states = 0;
    std::size_t device_workspaces = 0;
    std::size_t shared_entries = 0;
    std::size_t shared_bytes = 0;
    std::size_t ego_states = 0;
    std::size_t ego_workspaces = 0;
    for (std::size_t rank = 0; rank < released->size(); ++rank) {
        device_states += (*released)[rank].views;
        device_workspaces += (*released)[rank].workspaces;
        shared_entries += (*released)[rank].shared_entries;
        shared_bytes += (*released)[rank].shared_bytes;
        ego_states += (*released)[rank].ego_states;
        ego_workspaces += (*released)[rank].ego_workspaces;
    }
    py::dict out;
    out["devices"] = devices;
    out["released_device_views"] = device_states;
    out["released_device_workspaces"] = device_workspaces;
    // "Cleared" describes removal from the retention catalog.  External
    // shared_ptr leases, if any, keep their allocations alive until the lease
    // ends, so these values are not claimed as immediately freed bytes.
    out["cleared_shared_catalog_entries"] = shared_entries;
    out["cleared_shared_catalog_bytes"] = shared_bytes;
    out["released_ego_device_states"] = ego_states;
    out["released_ego_device_workspaces"] = ego_workspaces;
    out["released_prepared_pagerank_host_views"] = host_states;
    out["released_bfs_host_plans"] = host_plans.bfs;
    out["released_wcc_host_plans"] = host_plans.wcc;
    out["released_sssp_host_plans"] = host_plans.sssp;
    out["released_ego_host_plans"] = host_plans.ego;
    out["released_host_execution_plans"] = host_plans.total;
    out["released_host_views"] = host_states + host_plans.total;
    return out;
}

py::dict lifecycle_host_plan_stats(
    const gpu_easygraph::runtime::PartitionedHostPlanCacheStats& stats) {
    py::dict out;
    out["hits"] = stats.hits;
    out["builders"] = stats.builders;
    out["waiters"] = stats.waiters;
    out["build_failures"] = stats.build_failures;
    out["evictions"] = stats.evictions;
    out["generation_invalidations"] = stats.generation_invalidations;
    out["entries"] = stats.entries;
    out["capacity"] = stats.capacity;
    return out;
}

py::dict partitioned_state_stats(py::object device_request) {
    const gpu_easygraph::runtime::RuntimeStatus status =
        gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running) {
        throw std::runtime_error(
            "EGMGPU native runtime must be started before reading device state");
    }
    std::vector<int> devices = device_request.is_none()
        ? status.device_ids
        : device_request.cast<std::vector<int> >();
    std::sort(devices.begin(), devices.end());
    if (devices.empty() ||
        std::adjacent_find(devices.begin(), devices.end()) != devices.end()) {
        throw std::invalid_argument(
            "partitioned state stats require distinct devices");
    }
    struct WorkerStats {
        std::size_t pagerank_views;
        std::size_t pagerank_workspaces;
        std::size_t wcc_views;
        std::size_t wcc_workspaces;
        std::size_t bfs_views;
        std::size_t bfs_workspaces;
        std::size_t sssp_views;
        std::size_t sssp_workspaces;
        std::size_t ego_states;
        std::size_t ego_workspaces;
        gpu_easygraph::runtime::SharedArrayCatalogStats shared;
    };
    const std::shared_ptr<std::vector<WorkerStats> > observed(
        new std::vector<WorkerStats>(devices.size()));
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.reserve(devices.size());
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        tasks.push_back([observed, rank](int) {
            WorkerStats& item = (*observed)[rank];
            gpu_easygraph::runtime::PartitionedDeviceStateRegistry& pagerank =
                gpu_easygraph::runtime::partitioned_device_state_registry();
            gpu_easygraph::runtime::PartitionedWccStateRegistry& wcc =
                gpu_easygraph::runtime::partitioned_wcc_state_registry();
            gpu_easygraph::runtime::PartitionedBfsStateRegistry& bfs =
                gpu_easygraph::runtime::partitioned_bfs_state_registry();
            gpu_easygraph::runtime::PartitionedSsspStateRegistry& sssp =
                gpu_easygraph::runtime::partitioned_sssp_state_registry();
            gpu_easygraph::runtime::PartitionedEgoStateRegistry& ego =
                gpu_easygraph::runtime::partitioned_ego_state_registry();
            item.pagerank_views = pagerank.resident_view_count();
            item.pagerank_workspaces = pagerank.workspace_count();
            item.wcc_views = wcc.resident_view_count();
            item.wcc_workspaces = wcc.workspace_count();
            item.bfs_views = bfs.resident_view_count();
            item.bfs_workspaces = bfs.workspace_count();
            item.sssp_views = sssp.resident_view_count();
            item.sssp_workspaces = sssp.workspace_count();
            item.ego_states = ego.resident_state_count();
            item.ego_workspaces = ego.workspace_count();
            item.shared =
                gpu_easygraph::runtime::shared_device_array_catalog().stats();
        });
    }
    {
        py::gil_scoped_release release;
        gpu_easygraph::runtime::runtime_manager().execute_gang(
            devices, std::move(tasks));
    }
    py::list per_device;
    std::uint64_t aggregate_shared_bytes = 0U;
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        const WorkerStats& item = (*observed)[rank];
        py::dict value;
        value["device_id"] = devices[rank];
        value["pagerank_views"] = item.pagerank_views;
        value["pagerank_workspaces"] = item.pagerank_workspaces;
        value["wcc_views"] = item.wcc_views;
        value["wcc_workspaces"] = item.wcc_workspaces;
        value["bfs_views"] = item.bfs_views;
        value["bfs_workspaces"] = item.bfs_workspaces;
        value["sssp_views"] = item.sssp_views;
        value["sssp_workspaces"] = item.sssp_workspaces;
        value["ego_states"] = item.ego_states;
        value["ego_workspaces"] = item.ego_workspaces;
        value["shared_budget_bytes"] = item.shared.budget_bytes;
        value["shared_retained_bytes"] = item.shared.retained_bytes;
        value["shared_retained_entries"] = item.shared.retained_entries;
        value["shared_active_entries"] = item.shared.active_entries;
        value["shared_hits"] = item.shared.hits;
        value["shared_misses"] = item.shared.misses;
        value["shared_evictions"] = item.shared.evictions;
        value["shared_allocated_bytes"] = item.shared.allocated_bytes;
        aggregate_shared_bytes += item.shared.retained_bytes;
        per_device.append(value);
    }
    py::dict out;
    out["devices"] = devices;
    out["per_device"] = per_device;
    out["aggregate_shared_retained_bytes"] = aggregate_shared_bytes;
    out["prepared_pagerank_host_views"] = prepared_state_count();
    const gpu_easygraph::runtime::PartitionedHostPlanCacheRegistryStats
        host_plans = gpu_easygraph::runtime::partitioned_host_plan_cache_stats();
    const std::size_t host_plan_entries = host_plans.bfs.entries +
        host_plans.wcc.entries + host_plans.sssp.entries +
        host_plans.ego.entries;
    py::dict host_entries;
    host_entries["bfs"] = host_plans.bfs.entries;
    host_entries["wcc"] = host_plans.wcc.entries;
    host_entries["sssp"] = host_plans.sssp.entries;
    host_entries["ego"] = host_plans.ego.entries;
    host_entries["total"] = host_plan_entries;
    out["host_plan_entries"] = std::move(host_entries);
    py::dict host_caches;
    host_caches["bfs"] = lifecycle_host_plan_stats(host_plans.bfs);
    host_caches["wcc"] = lifecycle_host_plan_stats(host_plans.wcc);
    host_caches["sssp"] = lifecycle_host_plan_stats(host_plans.sssp);
    host_caches["ego"] = lifecycle_host_plan_stats(host_plans.ego);
    out["host_plan_caches"] = std::move(host_caches);
    return out;
}
// Added inside the existing binding translation unit's anonymous namespace.
using gpu_easygraph::runtime::ManagedStateInventory;
using gpu_easygraph::runtime::ManagedEvictRecord;

const char* managed_scope_text() {
    return "Request-boundary retained BFS/PR/SSSP/WCC view and workspace allocations plus all retained shared-catalog allocations, deduplicated by actual device+pointer. Excludes CUDA context, streams, communication-library/external scratch, host memory, unretained in-flight temporaries and construction peaks. Call only between completed requests under the session request lock.";
}

std::vector<int> managed_requested_devices(py::object request) {
    const auto status = gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running) throw std::runtime_error("Native runtime is not running");
    std::vector<int> devices = request.is_none() ? status.device_ids : request.cast<std::vector<int> >();
    std::sort(devices.begin(), devices.end());
    if (devices.empty() || std::adjacent_find(devices.begin(), devices.end()) != devices.end())
        throw std::invalid_argument("Managed state requires nonempty distinct devices");
    for (int device : devices)
        if (std::find(status.device_ids.begin(), status.device_ids.end(), device) == status.device_ids.end())
            throw std::invalid_argument("Managed state device is outside the running worker group");
    return devices;
}

ManagedStateInventory collect_managed_state(int device) {
    ManagedStateInventory result(device);
    gpu_easygraph::runtime::partitioned_bfs_state_registry().append_managed_inventory(result);
    gpu_easygraph::runtime::partitioned_device_state_registry().append_managed_inventory(result);
    gpu_easygraph::runtime::partitioned_sssp_state_registry().append_managed_inventory(result);
    gpu_easygraph::runtime::partitioned_wcc_state_registry().append_managed_inventory(result);
    gpu_easygraph::runtime::shared_device_array_catalog().append_managed_inventory(result);
    const auto dynamic = gpu_easygraph::runtime::dynamic_weight_counters();
    for (const auto& item : dynamic) result.counters["dynamic." + item.first] = item.second;
    return result;
}

py::dict managed_inventory_python(const std::vector<ManagedStateInventory>& inventories) {
    py::dict out;
    out["schema"] = "eggpu.native_managed_state.four_core.v1";
    out["algorithms"] = std::vector<std::string>{"bfs", "pr", "sssp", "wcc"};
    out["scope"] = managed_scope_text();
    out["construction_peak_bounded"] = false;
    out["query_boundary_required"] = true;
    out["workspace_policy"] = gpu_easygraph::runtime::managed_workspace_per_worker() ? "per_worker" : "per_view";
    py::list per_device, devices;
    std::uint64_t total = 0;
    bool has_active = false, has_untracked = false;
    for (const auto& inventory : inventories) {
        devices.append(inventory.device_id);
        py::dict device;
        device["device_id"] = inventory.device_id;
        device["managed_resident_bytes"] = inventory.bytes();
        const auto observed = gpu_easygraph::runtime::device_allocation_observation(inventory.device_id);
        py::dict observation;
        observation["scope"] = "All live DeviceAllocation objects; excludes CUDA context, raw allocations and external-library scratch; observed, not budget-enforced";
        observation["live_bytes"] = observed.live_bytes;
        observation["peak_bytes"] = observed.peak_bytes;
        observation["window_peak_bytes"] = observed.window_peak_bytes;
        observation["window_start_bytes"] = observed.window_start_bytes;
        observation["window_id"] = observed.window_id;
        observation["live_count"] = observed.live_count;
        observation["allocation_count"] = observed.allocation_count;
        observation["free_count"] = observed.free_count;
        observation["allocated_bytes"] = observed.allocated_bytes;
        observation["freed_bytes"] = observed.freed_bytes;
        observation["free_errors"] = observed.free_errors;
        device["allocation_observation"] = observation;
        total += inventory.bytes();
        py::list owners, allocations;
        std::uint64_t workspace_bytes = 0, view_bytes = 0, idle_catalog_bytes = 0;
        std::size_t views = 0, workspaces = 0;
        for (const auto& item : inventory.owners) {
            const auto& value = item.second;
            py::dict owner;
            owner["owner_id"] = value.owner_id;
            owner["kind"] = value.kind;
            owner["algorithm"] = value.algorithm;
            owner["graph_id"] = value.graph_id;
            owner["parent_owner_id"] = value.parent_owner_id;
            owner["policy"] = value.policy;
            owner["device_id"] = value.device_id;
            owner["partition_id"] = value.partition_id;
            owner["generation"] = value.generation;
            owner["active_leases"] = value.active_leases;
            has_active = has_active || value.active_leases != 0;
            std::uint64_t owned_bytes = 0;
            for (const auto& allocation : inventory.allocations)
                if (allocation.second.owners.count(value.owner_id)) owned_bytes += allocation.second.bytes;
            owner["referenced_bytes"] = owned_bytes;
            owners.append(owner);
            views += value.kind == "view" ? 1U : 0U;
            workspaces += value.kind == "workspace" ? 1U : 0U;
        }
        for (const auto& item : inventory.allocations) {
            const auto& value = item.second;
            py::dict allocation;
            std::ostringstream address, identity;
            address << "0x" << std::hex << value.pointer;
            identity << value.device_id << ':' << address.str() << ':' << value.allocation_generation;
            allocation["allocation_id"] = identity.str();
            allocation["allocation_generation"] = value.allocation_generation;
            allocation["pointer"] = address.str();
            allocation["device_id"] = value.device_id;
            allocation["bytes"] = value.bytes;
            allocation["owners"] = std::vector<std::string>(value.owners.begin(), value.owners.end());
            allocation["components"] = std::vector<std::string>(value.components.begin(), value.components.end());
            allocation["catalog_retained"] = value.catalog_retained;
            const bool idle = value.catalog_retained && value.catalog_external_references == 0;
            allocation["catalog_only_idle"] = idle;
            allocation["catalog_external_references"] = value.catalog_external_references;
            allocation["inventoried_shared_references"] = value.inventoried_shared_references;
            const std::size_t untracked = value.catalog_retained &&
                value.catalog_external_references > value.inventoried_shared_references ?
                value.catalog_external_references - value.inventoried_shared_references : 0U;
            allocation["untracked_shared_references"] = untracked;
            has_untracked = has_untracked || untracked != 0;
            bool workspace = false, view = false;
            for (const auto& id : value.owners) {
                workspace = workspace || inventory.owners.at(id).kind == "workspace";
                view = view || inventory.owners.at(id).kind == "view";
            }
            workspace_bytes += workspace ? value.bytes : 0U;
            view_bytes += view ? value.bytes : 0U;
            idle_catalog_bytes += idle ? value.bytes : 0U;
            allocations.append(allocation);
        }
        py::dict counters;
        for (const auto& item : inventory.counters) counters[py::str(item.first)] = item.second;
        device["owners"] = owners;
        device["allocations"] = allocations;
        device["view_count"] = views;
        device["workspace_count"] = workspaces;
        device["workspace_bytes"] = workspace_bytes;
        device["view_referenced_union_bytes"] = view_bytes;
        device["catalog_idle_bytes"] = idle_catalog_bytes;
        device["counters"] = counters;
        per_device.append(device);
    }
    out["devices"] = devices;
    out["per_device"] = per_device;
    out["managed_resident_bytes"] = total;
    out["has_active_leases"] = has_active;
    out["has_untracked_shared_references"] = has_untracked;
    return out;
}

void reset_allocation_observation(py::object request) {
    const auto devices = managed_requested_devices(request);
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    for (std::size_t rank = 0; rank < devices.size(); ++rank)
        tasks.push_back([](int device) { gpu_easygraph::runtime::reset_device_allocation_observation(device); });
    py::gil_scoped_release release;
    gpu_easygraph::runtime::runtime_manager().execute_gang(devices, std::move(tasks));
}

py::dict managed_state_inventory(py::object request) {
    const auto devices = managed_requested_devices(request);
    auto inventories = std::make_shared<std::vector<ManagedStateInventory> >(devices.size());
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    for (std::size_t rank = 0; rank < devices.size(); ++rank)
        tasks.push_back([inventories, rank](int device) { (*inventories)[rank] = collect_managed_state(device); });
    {
        py::gil_scoped_release release;
        gpu_easygraph::runtime::runtime_manager().execute_gang(devices, std::move(tasks));
    }
    return managed_inventory_python(*inventories);
}

py::dict managed_evict(const std::vector<std::string>& owner_ids, py::object request,
                       bool release_idle, bool release_workspaces) {
    const auto devices = managed_requested_devices(request);
    const std::set<std::string> selected(owner_ids.begin(), owner_ids.end());
    if (selected.size() != owner_ids.size()) throw std::invalid_argument("Duplicate managed owner IDs");
    auto before = std::make_shared<std::vector<ManagedStateInventory> >(devices.size());
    auto after = std::make_shared<std::vector<ManagedStateInventory> >(devices.size());
    auto receipts = std::make_shared<std::vector<ManagedEvictRecord> >(devices.size());
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    for (std::size_t rank = 0; rank < devices.size(); ++rank)
        tasks.push_back([before, after, receipts, selected, rank, release_idle, release_workspaces](int device) {
            (*before)[rank] = collect_managed_state(device);
            auto& receipt = (*receipts)[rank];
            gpu_easygraph::runtime::partitioned_bfs_state_registry().managed_evict(selected, release_workspaces, receipt);
            gpu_easygraph::runtime::partitioned_device_state_registry().managed_evict(selected, release_workspaces, receipt);
            gpu_easygraph::runtime::partitioned_sssp_state_registry().managed_evict(selected, release_workspaces, receipt);
            gpu_easygraph::runtime::partitioned_wcc_state_registry().managed_evict(selected, release_workspaces, receipt);
            if (release_idle) gpu_easygraph::runtime::shared_device_array_catalog().release_managed_idle(receipt);
            (*after)[rank] = collect_managed_state(device);
        });
    {
        py::gil_scoped_release release;
        gpu_easygraph::runtime::runtime_manager().execute_gang(devices, std::move(tasks));
    }
    std::set<std::string> recognized;
    py::list per_device;
    std::vector<std::string> removed, skipped, workspaces, unknown;
    std::uint64_t before_bytes = 0, after_bytes = 0, catalog_bytes = 0;
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        const auto& receipt = (*receipts)[rank];
        for (const auto& owner : (*before)[rank].owners)
            if (owner.second.kind == "view" || owner.second.kind == "workspace") recognized.insert(owner.first);
        removed.insert(removed.end(), receipt.removed.begin(), receipt.removed.end());
        skipped.insert(skipped.end(), receipt.skipped_active.begin(), receipt.skipped_active.end());
        workspaces.insert(workspaces.end(), receipt.released_workspaces.begin(), receipt.released_workspaces.end());
        py::dict row;
        row["device_id"] = devices[rank];
        row["before_bytes"] = (*before)[rank].bytes();
        row["after_bytes"] = (*after)[rank].bytes();
        row["removed_owner_ids"] = receipt.removed;
        row["released_workspace_owner_ids"] = receipt.released_workspaces;
        row["skipped_active_owner_ids"] = receipt.skipped_active;
        row["idle_catalog_released_bytes"] = receipt.idle_catalog_released_bytes;
        row["idle_catalog_released_entries"] = receipt.idle_catalog_released_entries;
        before_bytes += (*before)[rank].bytes();
        after_bytes += (*after)[rank].bytes();
        catalog_bytes += receipt.idle_catalog_released_bytes;
        per_device.append(row);
    }
    for (const auto& id : owner_ids) if (!recognized.count(id)) unknown.push_back(id);
    if (after_bytes > before_bytes) throw std::runtime_error("Managed eviction unexpectedly increased retained allocations");
    py::dict result;
    result["schema"] = "eggpu.native_managed_evict.v1";
    result["scope"] = managed_scope_text();
    result["requested_owner_ids"] = owner_ids;
    result["removed_owner_ids"] = removed;
    result["released_workspace_owner_ids"] = workspaces;
    result["skipped_active_owner_ids"] = skipped;
    result["unknown_owner_ids"] = unknown;
    result["before_bytes"] = before_bytes;
    result["after_bytes"] = after_bytes;
    result["released_bytes"] = before_bytes - after_bytes;
    result["idle_catalog_released_bytes"] = catalog_bytes;
    result["per_device"] = per_device;
    result["after_inventory"] = managed_inventory_python(*after);
    return result;
}

#endif

}  // namespace

void bind_egmgpu_partitioned_pagerank(py::module_& module) {
#ifdef EASYGRAPH_ENABLE_GPU
    module.def(
        "cpp_egmgpu_partitioned_pagerank",
        &run_partitioned_pagerank,
        py::arg("partitioned_graph"),
        py::arg("alpha") = 0.85,
        py::arg("max_iterations") = 200,
        py::arg("tolerance") = 1e-6,
        py::arg("devices") = py::none(),
        "Run destination-owned pull PageRank on native graph partitions.");
    module.def("cpp_egmgpu_partitioned_pagerank_serial", &run_serial_partitioned_pagerank,
        py::arg("partitioned_graph"), py::arg("alpha") = 0.85,
        py::arg("max_iterations") = 200, py::arg("tolerance") = 1e-6,
        py::arg("device") = 0,
        "Execute the same destination partitions sequentially on one GPU, preserving global iteration semantics.");
    module.def(
        "cpp_egmgpu_clear_partitioned_state",
        &clear_partitioned_state,
        py::arg("devices") = py::none(),
        "Release all partitioned worker state and retained host plans.");
    module.def("cpp_egmgpu_probe_sparse_retained_leases", &probe_sparse_retained_leases,
        py::arg("partitioned_graph"),py::arg("devices")=py::none(),
        "Qualification-only active lease/mismatched-plan probe after a retained sparse query.");
    module.def("cpp_egmgpu_reset_allocation_observation", &reset_allocation_observation,
        py::arg("devices") = py::none(),
        "Reset DeviceAllocation construction-peak windows at a completed request boundary.");
    module.def("cpp_egmgpu_managed_state_inventory", &managed_state_inventory,
        py::arg("devices") = py::none(),
        "Read retained BFS/PR/SSSP/WCC and catalog allocation identities at a completed request boundary.");
    module.def("cpp_egmgpu_managed_evict", &managed_evict,
        py::arg("owner_ids"), py::arg("devices") = py::none(),
        py::arg("release_idle") = true, py::arg("release_workspaces") = false,
        "Release selected idle view/workspace owners and optionally all idle scratch; preserve host plans.");
    module.def(
        "cpp_egmgpu_partitioned_state_stats",
        &partitioned_state_stats,
        py::arg("devices") = py::none(),
        "Read unified partitioned device-state and host-plan statistics.");
#else
    (void)module;
#endif
}
