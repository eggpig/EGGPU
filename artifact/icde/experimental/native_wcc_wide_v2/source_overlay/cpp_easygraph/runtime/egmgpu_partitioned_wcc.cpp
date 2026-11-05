#include "egmgpu_partitioned_wcc.h"
#include "egmgpu_wcc_host_finalize.h"
#include "egmgpu_wcc_public_output.h"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifdef EASYGRAPH_ENABLE_GPU
#include "egmgpu_partition_loader.h"
#include "egmgpu_wcc_host_assembly.h"
#include "egmgpu_owner_output_team_py.h"
#include "egmgpu_partition_array_scope.h"

#include <functions/partitioned/wcc_partitioned.cuh>
#include <runtime/collective_transport.h>
#include <runtime/cooperative_barrier.h>
#include <runtime/algorithm_control_policy.h>
#include <runtime/wcc_pinned_route_pool.h>
#include <runtime/native_runtime.h>
#include <runtime/partitioned_device_state.h>
#include <runtime/partitioned_host_plan_registry.h>
#include <runtime/partitioned_wcc_plan.h>
#include <runtime/partitioned_wcc_state.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#endif

namespace {

#ifdef EASYGRAPH_ENABLE_GPU
using easygraph::egmgpu::MultiPartitionGraph;
using easygraph::egmgpu::PartitionCSR;
using gpu_easygraph::runtime::CommunicationBackend;
using gpu_easygraph::runtime::CooperativeBarrier;
using gpu_easygraph::runtime::DeviceGroup;
using gpu_easygraph::runtime::DeviceTopology;
using gpu_easygraph::runtime::PartitionedWccPlan;
using gpu_easygraph::runtime::PinnedHostBuffer;
using gpu_easygraph::runtime::ResidentWccPartitionView;
using gpu_easygraph::runtime::SharedArrayCallStats;
using gpu_easygraph::runtime::WccHostPartition;
using gpu_easygraph::runtime::WccPartitionDeviceView;
using gpu_easygraph::runtime::WccPartitionUpload;
using gpu_easygraph::runtime::WccPartitionWorkspace;
using gpu_easygraph::runtime::WccHaloRoute;
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

// Fixed per query before worker dispatch; use one setting for an entire A/B process.
bool single_partition_candidate_enabled() {
    const char* value = std::getenv("EGMGPU_SINGLE_PARTITION_ORDERING");
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) == "1") return true;
    throw std::invalid_argument("EGMGPU_SINGLE_PARTITION_ORDERING must be 0 or 1");
}

class CudaEventTimer {
public:
    explicit CudaEventTimer(const char* context, bool enabled = true)
        : begin_(nullptr), end_(nullptr), context_(context), enabled_(enabled) {
        if (!enabled_) return;
        check_cuda(cudaEventCreate(&begin_), "create WCC timing event");
        try {
            check_cuda(cudaEventCreate(&end_), "create WCC timing event");
        } catch (...) {
            cudaEventDestroy(begin_);
            throw;
        }
    }
    ~CudaEventTimer() {
        if (begin_) cudaEventDestroy(begin_);
        if (end_) cudaEventDestroy(end_);
    }
    void start(cudaStream_t stream) {
        if (!enabled_) return;
        check_cuda(cudaEventRecord(begin_, stream), context_);
    }
    double finish(cudaStream_t stream, bool require_completion = false) {
        if (!enabled_) {
            if (require_completion) check_cuda(cudaStreamSynchronize(stream), context_);
            return -1.0;  // No device sample; MeasuredSeconds remains unmeasured.
        }
        check_cuda(cudaEventRecord(end_, stream), context_);
        check_cuda(cudaEventSynchronize(end_), context_);
        float milliseconds = 0.0F;
        check_cuda(cudaEventElapsedTime(&milliseconds, begin_, end_), context_);
        return static_cast<double>(milliseconds) / 1000.0;
    }

private:
    cudaEvent_t begin_;
    cudaEvent_t end_;
    const char* context_;
    bool enabled_;
};

struct MeasuredSeconds {
    bool measured;
    double seconds;

    MeasuredSeconds() : measured(false), seconds(0.0) {}
    void add(double value) {
        if (value < 0.0) return;
        measured = true;
        seconds += value;
    }
};

py::object timing_value(const MeasuredSeconds& value) {
    if (!value.measured) return py::none();
    return py::float_(value.seconds);
}

std::vector<int> requested_devices(
    const MultiPartitionGraph& graph,
    const std::vector<int>& request,
    std::string* stage) {
    *stage = "resolve_devices.runtime_status";
    const gpu_easygraph::runtime::RuntimeStatus status =
        gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running) {
        throw std::runtime_error(
            "EGMGPU native runtime must be started before partitioned WCC");
    }
    *stage = "resolve_devices.copy_request";
    std::vector<int> devices(request);
    *stage = "resolve_devices.manifest_partitions";
    if (devices.empty()) {
        devices.reserve(graph.partitions().size());
        for (std::size_t rank = 0; rank < graph.partitions().size(); ++rank) {
            devices.push_back(graph.partitions()[rank].device_id);
        }
    }
    *stage = "resolve_devices.sort";
    std::sort(devices.begin(), devices.end());
    if (std::adjacent_find(devices.begin(), devices.end()) != devices.end()) {
        throw std::invalid_argument("partitioned WCC devices must be distinct");
    }
    *stage = "resolve_devices.cardinality";
    if (devices.size() != graph.partitions().size()) {
        throw std::invalid_argument(
            "partitioned WCC requires one device for every partition");
    }
    *stage = "resolve_devices.membership";
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        if (!std::binary_search(
                status.device_ids.begin(), status.device_ids.end(), devices[rank])) {
            throw std::out_of_range(
                "partitioned WCC device is outside the active runtime");
        }
        if (graph.partitions()[rank].device_id != devices[rank]) {
            throw std::invalid_argument(
                "partition manifest device order differs from the requested group");
        }
    }
    return devices;
}

std::vector<WccHostPartition> host_partitions(
    const MultiPartitionGraph& graph) {
    if (graph.directed() || graph.view_orientation() != "undirected") {
        throw std::invalid_argument(
            "partitioned WCC requires a validated undirected CSR view");
    }
    std::vector<WccHostPartition> result(graph.partitions().size());
    for (std::size_t rank = 0; rank < graph.partitions().size(); ++rank) {
        const PartitionCSR& source = graph.partitions()[rank];
        WccHostPartition& target = result[rank];
        target.partition_id = source.partition_id;
        target.device_id = source.device_id;
        target.owned_vertex_count = static_cast<std::size_t>(
            source.owned_vertex_count);
        target.wide_offsets = source.wide_offsets;
        if (source.wide_offsets) target.offsets64.assign(source.offsets64.begin(), source.offsets64.end());
        else target.offsets.assign(source.offsets.begin(), source.offsets.end());
        target.indices.assign(source.indices.begin(), source.indices.end());
        target.local_to_global.assign(
            source.local_to_global.begin(), source.local_to_global.end());
        target.owners.assign(source.owners.begin(), source.owners.end());
    }
    return result;
}

WccPartitionUpload make_upload(
    const MultiPartitionGraph& graph,
    const PartitionedWccPlan& plan,
    const std::vector<int>& devices,
    std::size_t rank,
    bool forest_contraction, bool boundary_csr) {
    const WccHostPartition& partition = plan.partitions[rank];
    WccPartitionUpload upload;
    upload.partition_id = partition.partition_id;
    upload.device_id = partition.device_id;
    upload.graph_identity = gpu_easygraph::runtime::stable_signature64(
        graph.graph_id());
    upload.graph_identity_text = graph.graph_id();
    upload.generation = graph.generation();
    upload.device_group = devices;
    upload.view_signature = graph.view_signature() +
        (forest_contraction ? ":wcc-local-forest-v1" : ":wcc-owner-halo-v1");
    if (boundary_csr) upload.view_signature += ":csr-boundary-v1";
    upload.forest_contraction = forest_contraction;
    upload.boundary_csr = boundary_csr;
    upload.owned_vertex_count = partition.owned_vertex_count;
    upload.wide_offsets = partition.wide_offsets;
    upload.offsets = partition.offsets;
    upload.offsets64 = partition.offsets64;
    upload.indices = partition.indices;
    upload.local_to_global = partition.local_to_global;
    upload.owners = partition.owners;
    if (forest_contraction) {
        for (std::size_t row = 0; row < partition.owned_vertex_count; ++row) {
            for (std::int64_t edge = partition.offset(row); edge < partition.offset(row + 1U); ++edge) {
                const int neighbor = partition.indices[edge];
                if (static_cast<std::size_t>(neighbor) >= partition.owned_vertex_count) {
                    ++upload.boundary_edge_count;
                    if (!boundary_csr) {
                        upload.boundary_sources.push_back(static_cast<int>(row));
                        upload.boundary_neighbors.push_back(neighbor);
                    }
                }
            }
        }
    }
    const PartitionCSR& manifest_partition = graph.partitions().at(rank);
    upload.offsets_sha256 = manifest_partition.offsets_sha256;
    upload.indices_sha256 = manifest_partition.indices_sha256;
    upload.local_to_global_sha256 =
        manifest_partition.local_to_global_sha256;
    upload.owners_sha256 = manifest_partition.owners_sha256;
    upload.topology_compatibility =
        easygraph::egmgpu::partition_topology_compatibility(
            graph, partition.partition_id, devices);
    upload.ownership_compatibility =
        easygraph::egmgpu::partition_ownership_compatibility(
            graph, partition.partition_id, devices);
    upload.outgoing_owner_local_ids.resize(plan.partitions.size());
    upload.incoming_halo_local_ids.resize(plan.partitions.size());
    for (std::size_t peer = 0; peer < plan.partitions.size(); ++peer) {
        upload.outgoing_owner_local_ids[peer] =
            plan.route(rank, peer).source_owned_local_ids;
        upload.incoming_halo_local_ids[peer] =
            plan.route(peer, rank).destination_halo_local_ids;
    }
    return upload;
}

// Copy-mode preserves the original by-value task captures for the ablation.
// Reuse-mode captures only an immutable plan-owned upload description.
struct WccUploadHolder {
    std::shared_ptr<const WccPartitionUpload> prepared;
    WccPartitionUpload owned;
    const WccPartitionUpload& get() const {
        return prepared ? *prepared : owned;
    }
};

struct RankTiming {
    MeasuredSeconds kernel;
    MeasuredSeconds initialization;
    MeasuredSeconds local_union;
    MeasuredSeconds local_hook_compress;
    MeasuredSeconds changed_clear;
    MeasuredSeconds changed_d2h;
    MeasuredSeconds halo_scatter;
    MeasuredSeconds boundary_combine;
    MeasuredSeconds component_projection;
    MeasuredSeconds forest_compression;
    MeasuredSeconds state_acquire;
    MeasuredSeconds peer_enable;
    MeasuredSeconds pack;
    MeasuredSeconds transfer;
    MeasuredSeconds wait;
    MeasuredSeconds communication;
    MeasuredSeconds output_gather;
    MeasuredSeconds host_owner_assembly;
    MeasuredSeconds host_output_ready_wait;
};

struct RankState {
    easygraph::egmgpu::owner_output::Observation core_output_observation;
    std::shared_ptr<ResidentWccPartitionView> view;
    std::shared_ptr<WccPartitionWorkspace> workspace;
    std::vector<std::uint64_t> owned_labels;
    std::shared_ptr<PinnedHostBuffer> pinned_labels;
    gpu_easygraph::runtime::GpuLocalPinnedAllocationInfo host_output_info;
    bool host_output_new_allocation;
    std::size_t host_owner_assembly_bytes = 0;
    std::size_t output_tile_begin = 0, output_tile_end = 0;
    RankTiming timing;
    int host_changed;
    std::size_t resident_bytes;
    bool device_view_cache_hit;
    SharedArrayCallStats shared_arrays;

    RankState()
        : host_output_new_allocation(false), host_changed(0),
          resident_bytes(0),
          device_view_cache_hit(false) {}

    const std::uint64_t* labels_data() const {
        return pinned_labels ? static_cast<const std::uint64_t*>(pinned_labels->data())
                             : owned_labels.data();
    }
};

struct WccOutputOptions {
    std::string transfer, placement, merge, assembly;
    bool tiled() const { return assembly == "tiled"; }
    bool worker_assembly() const { return assembly == "worker" || tiled(); }
    bool pinned() const { return transfer == "pinned"; }
    bool canonical() const { return merge == "canonical" || merge == "canonical_direct"; }
    bool direct() const { return merge == "canonical_direct"; }
};

// Workspace host buffers outlive a gang. Hold these leases on the CALLER
// through host merge, so another gang cannot overwrite a buffer being read.
// Sorting device IDs avoids lock-order inversion; disjoint GPU groups retain
// concurrency. Workers never acquire or release these caller-owned locks.
struct WccHostOutputLease {
    std::vector<std::unique_lock<std::mutex> > locks;
    double wait_seconds = 0.0;

    void acquire(const std::vector<int>& devices, bool pinned) {
        if (!pinned) return;
        static std::mutex registry_mutex;
        static std::map<int, std::shared_ptr<std::mutex> > registry;
        std::vector<int> ordered = devices;
        std::sort(ordered.begin(), ordered.end());
        std::vector<std::shared_ptr<std::mutex> > selected;
        {
            std::lock_guard<std::mutex> guard(registry_mutex);
            for (int device : ordered) {
                auto& mutex = registry[device];
                if (!mutex) mutex = std::make_shared<std::mutex>();
                selected.push_back(mutex);
            }
        }
        const auto begin = SteadyClock::now();
        for (const auto& mutex : selected) locks.emplace_back(*mutex);
        wait_seconds = elapsed_seconds(begin, SteadyClock::now());
    }

    void release() { locks.clear(); }
};

WccOutputOptions wcc_output_options(bool forest) {
    auto setting = [](const char* name, const char* fallback) {
        const char* value = std::getenv(name);
        return std::string(value ? value : fallback);
    };
    WccOutputOptions result;
    result.transfer = setting("EGMGPU_WCC_HOST_OUTPUT", "pageable");
    result.placement = setting("EGMGPU_WCC_PINNED_PLACEMENT", "caller");
    result.merge = setting("EGMGPU_WCC_RESULT_MERGE", "legacy");
    result.assembly = setting("EGMGPU_WCC_HOST_ASSEMBLY", "caller");
    if (result.transfer != "pageable" && result.transfer != "pinned")
        throw std::invalid_argument("EGMGPU_WCC_HOST_OUTPUT must be pageable or pinned");
    if (result.placement != "caller" && result.placement != "numa_local")
        throw std::invalid_argument("EGMGPU_WCC_PINNED_PLACEMENT must be caller or numa_local");
    if (result.merge != "legacy" && result.merge != "canonical" && result.merge != "canonical_direct")
        throw std::invalid_argument("EGMGPU_WCC_RESULT_MERGE must be legacy, canonical, or canonical_direct");
    if (result.assembly != "caller" && result.assembly != "worker" && result.assembly != "tiled")
        throw std::invalid_argument("EGMGPU_WCC_HOST_ASSEMBLY must be caller, worker, or tiled");
    if (result.tiled() && !result.pinned())
        throw std::invalid_argument("tiled WCC host assembly requires pinned rank outputs");
    if (result.worker_assembly() && (!forest || !result.direct()))
        throw std::invalid_argument("worker WCC host assembly requires forest and canonical_direct");
    if (result.canonical() && !forest)
        throw std::invalid_argument("canonical WCC merge requires the converged forest algorithm");
    return result;
}

bool wcc_boundary_csr(bool forest) {
    const char* value = std::getenv("EGMGPU_WCC_BOUNDARY_REPRESENTATION");
    const std::string mode = value ? value : "coo";
    if (mode != "coo" && mode != "csr")
        throw std::invalid_argument("EGMGPU_WCC_BOUNDARY_REPRESENTATION must be coo or csr");
    if (mode == "csr" && !forest)
        throw std::invalid_argument("CSR boundary representation requires the forest algorithm");
    return mode == "csr";
}

void combine_forest_boundary(const WccPartitionDeviceView& view,
                            WccPartitionWorkspace& workspace, cudaStream_t stream) {
    if (view.boundary_csr) {
        // Skip a graph with no boundary, including whole1, just like COO.
        if (view.boundary_edge_count == 0U) return;
        gpu_easygraph::partitioned::combine_wcc_boundary_components_csr(
            view.offsets, view.indices, view.owned_vertex_count,
            workspace.local_parents(), workspace.labels(), workspace.changed(), stream);
    } else {
        gpu_easygraph::partitioned::combine_wcc_boundary_components(
            view.boundary_sources, view.boundary_neighbors, view.boundary_edge_count,
            workspace.local_parents(), workspace.labels(), workspace.changed(), stream);
    }
}

bool wcc_unique_local_edges(bool forest) {
    const char* value = std::getenv("EGMGPU_WCC_LOCAL_UNION");
    const std::string mode = value ? value : "all_arcs";
    if (mode != "all_arcs" && mode != "unique_edges")
        throw std::invalid_argument("EGMGPU_WCC_LOCAL_UNION must be all_arcs or unique_edges");
    if (mode == "unique_edges" && !forest)
        throw std::invalid_argument("unique-edge WCC union requires forest algorithm");
    return mode == "unique_edges";
}

void prepare_wcc_host_output(RankState& state, const WccOutputOptions& output) {
    if (!output.pinned()) return;
    state.host_output_new_allocation = state.workspace->ensure_host_output(
        state.view->view().owned_vertex_count, output.placement);
    state.pinned_labels = state.workspace->host_output();
    state.host_output_info = state.workspace->host_output_info();
}

void append_wcc_output_metadata(py::dict& item, const RankState& state) {
    item["core_output_thread_observation"] = easygraph::egmgpu::owner_output::to_python(state.core_output_observation);
    item["host_output_new_allocation"] = state.host_output_new_allocation;
    item["host_owner_assembly_wall_seconds"] = timing_value(state.timing.host_owner_assembly);
    item["host_owner_assembly_output_bytes"] = state.host_owner_assembly_bytes;
    item["host_output_readiness_wait_seconds"] = timing_value(state.timing.host_output_ready_wait);
    item["output_tile_begin"] = state.output_tile_begin;
    item["output_tile_end"] = state.output_tile_end;
    item["host_output_pinned_bytes"] = state.pinned_labels ? state.pinned_labels->size() : 0U;
    if (state.pinned_labels) {
        const auto& info = state.host_output_info;
        item["host_output_numa_node"] = info.numa_node;
        item["host_output_allocation_cpu"] = info.allocation_cpu;
        item["host_output_pci_bus_id"] = info.pci_bus_id;
    }
}

using WccPinnedRoutes = gpu_easygraph::runtime::WccPinnedRoutePool<PinnedHostBuffer>;

struct SharedWccExecution {
    SharedWccExecution(
        std::size_t count,
        const std::shared_ptr<WccPinnedRoutes>& pinned)
        : barrier(count),
          ranks(count),
          any_changed(false),
          iterations(0),
          converged(false),
          phases(count, "not_started"),
          host_buffers(pinned->buffers), output_sources(count, nullptr) {}

    // Preserve the serial-partition entry's uncached constructor contract.
    SharedWccExecution(std::size_t count, const PartitionedWccPlan& plan, bool staging)
        : SharedWccExecution(count, std::make_shared<WccPinnedRoutes>()) {
        if (staging) {
            WccPinnedRoutes routes;
            routes.ensure(plan);
            host_buffers = std::move(routes.buffers);
        } else host_buffers.resize(count * count);
    }

    CooperativeBarrier barrier;
    const bool control_fusion_requested = gpu_easygraph::runtime::algorithm_candidate_flag(
        "EGGPU_ALGORITHM_CONTROL_FUSION");
    bool control_fusion_used = false;
    std::uint64_t control_reduction_calls = 0;
    std::vector<RankState> ranks;
    bool any_changed;
    int iterations;
    bool converged;
    std::vector<std::string> phases;
    std::vector<std::shared_ptr<PinnedHostBuffer> > host_buffers;
    std::vector<const std::uint64_t*> output_sources;
    // Independent query output: never retained in a reusable workspace/cache.
    std::unique_ptr<std::uint64_t[]> worker_labels;
    double* public_labels = nullptr; // Borrowed from caller/GIL-owned request storage.
    easygraph::egmgpu::WccPublicObservation public_observation;
    bool core_output_team16 = false;
    int core_output_width = 1;
    std::size_t core_output_workers = 1;
    std::shared_ptr<const easygraph::egmgpu::owner_output::CpuPlacement> core_output_placement;
    MeasuredSeconds worker_result_allocation;
    std::size_t worker_result_allocation_bytes = 0;
};

void prepare_wcc_worker_result(
    const PartitionedWccPlan& plan, SharedWccExecution& shared,
    const WccOutputOptions& output) {
    if (!output.worker_assembly()) return;
    const auto begin = SteadyClock::now();
    // Preserve the v9 whole1/pageable vector move: no second result array.
    if (plan.partitions.size() != 1U || output.pinned()) {
        if (plan.vertex_count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t))
            throw std::overflow_error("WCC result exceeds host address space");
        const auto count = static_cast<std::size_t>(plan.vertex_count);
        shared.worker_labels.reset(new std::uint64_t[count]);
        shared.worker_result_allocation_bytes = count * sizeof(std::uint64_t);
    }
    // new[] is intentionally uninitialized; first-touch happens in workers.
    shared.worker_result_allocation.add(elapsed_seconds(begin, SteadyClock::now()));
}

void assemble_wcc_worker_result(
    const WccHostPartition& partition, SharedWccExecution& shared, RankState& state, std::size_t output_rank = 0U) {
    if ((!shared.worker_labels && !shared.public_labels) || !shared.converged) return;
    const auto begin = SteadyClock::now();
    if (shared.public_labels) {
        auto write = [&](std::size_t first,std::size_t last) {
            if(first==last)return;
            easygraph::egmgpu::assemble_wcc_public_owned(partition.local_to_global.data()+first,
                last-first,state.labels_data()+first,shared.public_labels,
                shared.worker_result_allocation_bytes/sizeof(double),shared.public_observation);
        };
        if(shared.core_output_team16)
            easygraph::egmgpu::owner_output::run(partition.owned_vertex_count,shared.core_output_width,
                state.core_output_observation,write,shared.core_output_placement->for_worker(output_rank));
        else {easygraph::egmgpu::owner_output::observe_original(state.core_output_observation);write(0,partition.owned_vertex_count);}
        state.host_owner_assembly_bytes=partition.owned_vertex_count*sizeof(double);
        state.timing.host_owner_assembly.add(elapsed_seconds(begin,SteadyClock::now()));
        return;
    }
    if (shared.core_output_team16) {
        // Disjoint local slices preserve the immutable plan's unique owner
        // writes. The parallel-region join retains the existing lease/output
        // lifetime and adds no GPU synchronization or extra result allocation.
        easygraph::egmgpu::owner_output::run(partition.owned_vertex_count,
            shared.core_output_width, state.core_output_observation,
            [&](std::size_t first, std::size_t last) {
                if (first == last) return;
                easygraph::egmgpu::assemble_wcc_owned_host(
                    partition.local_to_global.data()+first, last-first,
                    state.labels_data()+first, shared.worker_labels.get());
            }, shared.core_output_placement->for_worker(output_rank));
    } else {
        easygraph::egmgpu::owner_output::observe_original(state.core_output_observation);
        easygraph::egmgpu::assemble_wcc_owned_host(partition.local_to_global.data(),
            partition.owned_vertex_count, state.labels_data(), shared.worker_labels.get());
    }
    state.host_owner_assembly_bytes = partition.owned_vertex_count * sizeof(std::uint64_t);
    state.timing.host_owner_assembly.add(elapsed_seconds(begin, SteadyClock::now()));
}

void assemble_wcc_tiled_result(
    const PartitionedWccPlan& plan, SharedWccExecution& shared,
    RankState& state, std::size_t output_rank, std::size_t output_workers) {
    if ((!shared.worker_labels && !shared.public_labels) || !shared.converged) return;
    const auto begin = SteadyClock::now();
    if(shared.public_labels) {
        const auto bounds=easygraph::egmgpu::wcc_output_tile_bounds(shared.public_labels,
            static_cast<std::size_t>(plan.vertex_count),output_rank,output_workers);
        auto write=[&](std::size_t first,std::size_t last) {
            for(std::size_t tile=first;tile<last;++tile){
                const auto local=easygraph::egmgpu::wcc_output_tile_bounds(shared.public_labels+bounds.first,
                    bounds.second-bounds.first,tile,static_cast<std::size_t>(shared.core_output_width));
                easygraph::egmgpu::assemble_wcc_public_tiled(plan.output_source_for_global.data(),
                    shared.output_sources.data(),shared.public_labels,bounds.first+local.first,
                    bounds.first+local.second,plan.vertex_count,shared.public_observation);
            }
        };
        if(shared.core_output_team16)
            easygraph::egmgpu::owner_output::run(shared.core_output_width,shared.core_output_width,
                state.core_output_observation,write,shared.core_output_placement->for_worker(output_rank));
        else {easygraph::egmgpu::owner_output::observe_original(state.core_output_observation);write(0,1);}
        state.output_tile_begin=bounds.first;state.output_tile_end=bounds.second;
        state.host_owner_assembly_bytes=(bounds.second-bounds.first)*sizeof(double);
        state.timing.host_owner_assembly.add(elapsed_seconds(begin,SteadyClock::now()));
        return;
    }
    const auto bounds = easygraph::egmgpu::wcc_output_tile_bounds(
        shared.worker_labels.get(), static_cast<std::size_t>(plan.vertex_count),
        output_rank, output_workers);
    if (shared.core_output_team16) {
        // Keep the preselected output-owner tile and its existing gather index.
        // Split this tile's physical cache lines with the already proven helper;
        // adjacent teams still write no common line across rank boundaries.
        const std::size_t width = static_cast<std::size_t>(shared.core_output_width);
        easygraph::egmgpu::owner_output::run(width, shared.core_output_width,
            state.core_output_observation, [&](std::size_t first, std::size_t last) {
                for (std::size_t tile=first; tile<last; ++tile) {
                    const auto local = easygraph::egmgpu::wcc_output_tile_bounds(
                        shared.worker_labels.get()+bounds.first, bounds.second-bounds.first,
                        tile, width);
                    easygraph::egmgpu::assemble_wcc_tiled_host(plan.output_source_for_global.data(),
                        shared.output_sources.data(), shared.worker_labels.get(),
                        bounds.first+local.first, bounds.first+local.second);
                }
            }, shared.core_output_placement->for_worker(output_rank));
    } else {
        easygraph::egmgpu::owner_output::observe_original(state.core_output_observation);
        easygraph::egmgpu::assemble_wcc_tiled_host(plan.output_source_for_global.data(),
            shared.output_sources.data(), shared.worker_labels.get(), bounds.first, bounds.second);
    }
    state.output_tile_begin = bounds.first; state.output_tile_end = bounds.second;
    state.host_owner_assembly_bytes = (bounds.second-bounds.first)*sizeof(std::uint64_t);
    state.timing.host_owner_assembly.add(elapsed_seconds(begin, SteadyClock::now()));
}

void append_wcc_assembly_metadata(
    py::dict& out, const SharedWccExecution& shared, const WccOutputOptions& output,
    const PartitionedWccPlan& plan, bool plan_builder) {
    MeasuredSeconds maximum, total;
    std::size_t written = 0;
    for (const auto& rank : shared.ranks) {
        written += rank.host_owner_assembly_bytes;
        if (!rank.timing.host_owner_assembly.measured) continue;
        const auto value = rank.timing.host_owner_assembly.seconds;
        maximum.measured = true;
        maximum.seconds = std::max(maximum.seconds, value);
        total.add(value);
    }
    out["host_assembly_strategy"] = output.assembly;
    out["core_output_team_version"] = "stage4_core_owner_output_v1";
    out["core_output_team_policy"] = shared.core_output_team16 ? "team16" : "original";
    out["core_output_workers"] = shared.core_output_workers;
    out["core_output_requested_per_worker"] = shared.core_output_width;
    out["core_output_cpu_placement"] = easygraph::egmgpu::owner_output::placement_to_python(*shared.core_output_placement);
    py::list core_output_observations;
    for (const auto& state : shared.ranks)
        core_output_observations.append(easygraph::egmgpu::owner_output::to_python(state.core_output_observation));
    out["core_output_observations"] = std::move(core_output_observations);
    out["host_assembly_effective"] = !output.worker_assembly() ? "caller" :
        (shared.worker_result_allocation_bytes ? output.assembly : "whole_pageable_move");
    out["wcc_output_partition_version"] = "stage4_host_tiles_v1";
    out["host_output_index_retained_bytes"] = plan.output_source_for_global.capacity()*sizeof(std::uint64_t);
    out["host_output_index_build_this_call_seconds"] = plan_builder ? plan.output_index_build_seconds : 0.0;
    out["host_output_index_build_timing_scope"] = "nested within host-plan build and state_preparation; zero on cache hit";
    out["host_output_tiled_readiness_barriers"] = output.tiled() && plan.partitions.size()>1U ? 1 : 0;
    out["host_output_tile_cacheline_bytes"] = output.tiled() ? 64 : 0;
    out["host_result_allocation_wall_seconds"] = timing_value(shared.worker_result_allocation);
    out["host_result_new_allocation_bytes"] = output.worker_assembly()
        ? py::cast(shared.worker_result_allocation_bytes) : py::none();
    out["worker_host_assembly_max_partition_wall_seconds"] = timing_value(maximum);
    out["worker_host_assembly_sum_partition_wall_seconds"] = timing_value(total);
    out["worker_host_assembly_output_bytes"] = written;
    out["worker_output_timing_semantics"] =
        "host_result_allocation is caller wall before gang, no initialization; "
        "per-partition owner assembly is worker host wall after completed D2H; "
        "device_processing_seconds includes worker assembly; max/sum are not additive critical-path stages; "
        "output bytes count host destination writes, not GPU allocations or measured memory traffic";
}

void timed_barrier(
    const std::shared_ptr<SharedWccExecution>& shared,
    RankTiming* timing,
    bool measure) {
    const SteadyClock::time_point begin = SteadyClock::now();
    shared->barrier.arrive_and_wait();
    if (measure) {
        timing->wait.add(elapsed_seconds(begin, SteadyClock::now()));
    }
}

// Replaces only a control reduction's arrival + rank0 + publication pair.
// GPU producer/transfer completion and payload-reuse barriers are untouched.
template <typename Reduction>
void timed_control_reduction(
    const std::shared_ptr<SharedWccExecution>& shared,
    RankTiming* timing, bool measure, std::size_t rank, Reduction reduce) {
    if (!shared->control_fusion_used) {
        timed_barrier(shared, timing, measure);
        if (rank == 0) reduce();
        timed_barrier(shared, timing, measure);
        return;
    }
    const auto begin = SteadyClock::now();
    shared->barrier.arrive_and_wait_then([&]() {
        reduce();
        ++shared->control_reduction_calls;
    });
    if (measure) timing->wait.add(elapsed_seconds(begin, SteadyClock::now()));
}

void pack_routes(
    const WccPartitionDeviceView& view,
    WccPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankTiming* timing,
    bool profile) {
    bool launched = false;
    CudaEventTimer timer("measure WCC pack", profile);
    for (std::size_t destination = 0;
         destination < view.outgoing_counts.size(); ++destination) {
        const std::size_t count = view.outgoing_counts[destination];
        if (!count) continue;
        if (!launched) {
            // labels are produced on compute; pack is a distinct stream.
            // Profile-on already waited at the preceding local timer.
            if (!profile) check_cuda(cudaStreamSynchronize(streams.compute()),
                "publish WCC owner labels to pack stream");
            timer.start(streams.pack());
            launched = true;
        }
        gpu_easygraph::partitioned::pack_wcc_owner_labels(
            workspace.labels(), view.outgoing_owner_local_ids[destination],
            workspace.send_buffer(destination), count, streams.pack());
    }
    // Another device/transfer stream will read these send buffers next.
    if (launched) timing->pack.add(timer.finish(streams.pack(), true));
}

void scatter_routes(
    const WccPartitionDeviceView& view,
    WccPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankTiming* timing,
    bool profile) {
    bool launched = false;
    CudaEventTimer timer("measure WCC halo scatter", profile);
    for (std::size_t source = 0; source < view.incoming_counts.size(); ++source) {
        const std::size_t count = view.incoming_counts[source];
        if (!count) continue;
        if (!launched) {
            timer.start(streams.compute());
            launched = true;
        }
        gpu_easygraph::partitioned::scatter_wcc_halo_labels(
            workspace.receive_buffer(source), view.incoming_halo_local_ids[source],
            workspace.labels(), count, streams.compute());
    }
    if (launched) {
        const double seconds = timer.finish(streams.compute());
        timing->kernel.add(seconds);
        timing->halo_scatter.add(seconds);
    }
}

void peer_transfer_routes(
    const std::shared_ptr<SharedWccExecution>& shared,
    const PartitionedWccPlan& plan,
    std::size_t rank,
    int worker_device,
    WorkerStreams& streams,
    RankTiming* timing,
    bool profile) {
    bool copied = false;
    CudaEventTimer timer("measure WCC peer transfer", profile);
    for (std::size_t source = 0; source < plan.partitions.size(); ++source) {
        const std::size_t count = plan.route(source, rank).size();
        if (!count) continue;
        if (!copied) {
            timer.start(streams.transfer());
            copied = true;
        }
        gpu_easygraph::runtime::peer_copy_async(
            shared->ranks[rank].workspace->receive_buffer(source),
            worker_device,
            shared->ranks[source].workspace->send_buffer(rank),
            plan.partitions[source].device_id,
            count * sizeof(std::uint64_t),
            streams.transfer());
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer(), true));
}

void stage_outgoing_routes(
    const std::shared_ptr<SharedWccExecution>& shared,
    const PartitionedWccPlan& plan,
    std::size_t rank,
    WorkerStreams& streams,
    RankTiming* timing,
    bool profile) {
    bool copied = false;
    CudaEventTimer timer("measure WCC D2H staging", profile);
    for (std::size_t destination = 0;
         destination < plan.partitions.size(); ++destination) {
        const std::size_t count = plan.route(rank, destination).size();
        if (!count) continue;
        if (!copied) {
            timer.start(streams.transfer());
            copied = true;
        }
        check_cuda(
            cudaMemcpyAsync(
                shared->host_buffers[
                    rank * plan.partitions.size() + destination]->data(),
                shared->ranks[rank].workspace->send_buffer(destination),
                count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
                streams.transfer()),
            "stage WCC owner labels to host");
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer(), true));
}

void stage_incoming_routes(
    const std::shared_ptr<SharedWccExecution>& shared,
    const PartitionedWccPlan& plan,
    std::size_t rank,
    WorkerStreams& streams,
    RankTiming* timing,
    bool profile) {
    bool copied = false;
    CudaEventTimer timer("measure WCC H2D staging", profile);
    for (std::size_t source = 0; source < plan.partitions.size(); ++source) {
        const std::size_t count = plan.route(source, rank).size();
        if (!count) continue;
        if (!copied) {
            timer.start(streams.transfer());
            copied = true;
        }
        check_cuda(
            cudaMemcpyAsync(
                shared->ranks[rank].workspace->receive_buffer(source),
                shared->host_buffers[
                    source * plan.partitions.size() + rank]->data(),
                count * sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                streams.transfer()),
            "stage WCC halo labels to device");
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer(), true));
}

struct NativeWccResult {
    std::vector<std::uint64_t> labels;
    std::unique_ptr<std::uint64_t[]> uninitialized_labels;
    std::size_t label_count = 0;
    std::vector<std::vector<std::uint64_t> > components;
    double merge_seconds;
    std::uint64_t resident_bytes;
    MeasuredSeconds kernel;
    MeasuredSeconds pack;
    MeasuredSeconds transfer;
    MeasuredSeconds wait;
    MeasuredSeconds communication;
    MeasuredSeconds output_gather;
};

void update_critical_path(
    MeasuredSeconds* aggregate, const MeasuredSeconds& rank) {
    if (!rank.measured) return;
    aggregate->measured = true;
    aggregate->seconds = std::max(aggregate->seconds, rank.seconds);
}

NativeWccResult merge_wcc_result(
    const PartitionedWccPlan& plan,
    const std::shared_ptr<SharedWccExecution>& shared,
    bool dense_output, bool canonical_output = false, bool direct_output = false) {
    const SteadyClock::time_point begin = SteadyClock::now();
    if (plan.vertex_count > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("WCC result exceeds host address space");
    }
    NativeWccResult result;
    result.resident_bytes = 0;
    const std::size_t count = static_cast<std::size_t>(plan.vertex_count);
    result.label_count = count;
    std::vector<std::uint64_t> labels;
    std::uint64_t* destination = nullptr;
    // The validated owner sets partition [0,N), each owner list is sorted,
    // and convergence of the forest projection gives canonical global labels.
    // No host parent chase or value-initialization is needed in this mode.
    const bool worker_assembled = static_cast<bool>(shared->worker_labels) || shared->public_labels;
    const bool move_whole = !worker_assembled && canonical_output && plan.partitions.size() == 1U &&
        !shared->ranks[0].pinned_labels;
    if (shared->public_labels) {
        // Final FP64 values already exist; retain only native stage metadata.
    } else if (worker_assembled) {
        result.uninitialized_labels = std::move(shared->worker_labels);
        destination = result.uninitialized_labels.get();
    } else if (move_whole) {
        labels = std::move(shared->ranks[0].owned_labels);
        destination = labels.data();
    } else if (canonical_output) {
        result.uninitialized_labels.reset(new std::uint64_t[count]);
        destination = result.uninitialized_labels.get();
    } else {
        labels.resize(count);
        destination = labels.data();
    }
    for (std::size_t rank = 0; rank < plan.partitions.size(); ++rank) {
        const WccHostPartition& partition = plan.partitions[rank];
        const RankState& state = shared->ranks[rank];
        if (!move_whole && !worker_assembled) {
            if (canonical_output && plan.partitions.size() == 1U) {
                std::memcpy(destination, state.labels_data(), count * sizeof(std::uint64_t));
            } else if (direct_output && partition.owned_vertex_count > 0U &&
                       partition.local_to_global[partition.owned_vertex_count - 1U] -
                           partition.local_to_global[0] == partition.owned_vertex_count - 1U) {
                // The validated owner IDs are sorted and distinct. Equal
                // endpoint span and cardinality prove the complete interval;
                // no per-query owner scan is needed. Empty ranks copy nothing.
                std::memcpy(destination + partition.local_to_global[0],
                    state.labels_data(), partition.owned_vertex_count * sizeof(std::uint64_t));
            } else {
                for (std::size_t local = 0; local < partition.owned_vertex_count; ++local) {
                    destination[static_cast<std::size_t>(partition.local_to_global[local])] =
                        state.labels_data()[local];
                }
            }
        }
        result.resident_bytes += static_cast<std::uint64_t>(state.resident_bytes);
        update_critical_path(&result.kernel, state.timing.kernel);
        update_critical_path(&result.pack, state.timing.pack);
        update_critical_path(&result.transfer, state.timing.transfer);
        update_critical_path(&result.wait, state.timing.wait);
        update_critical_path(&result.communication, state.timing.communication);
        update_critical_path(&result.output_gather, state.timing.output_gather);
    }
    for (std::size_t vertex = 0; !canonical_output && vertex < labels.size(); ++vertex) {
        std::uint64_t current = labels[vertex];
        std::size_t hops = 0;
        while (current < labels.size() && labels[static_cast<std::size_t>(current)] <
                   current && hops++ < labels.size()) {
            current = labels[static_cast<std::size_t>(current)];
        }
        if (current >= labels.size()) {
            throw std::runtime_error("partitioned WCC produced an invalid label");
        }
        labels[vertex] = current;
    }
    if (dense_output) {
        result.labels = std::move(labels);
        result.merge_seconds = elapsed_seconds(begin, SteadyClock::now());
        return result;
    }
    std::map<std::uint64_t, std::vector<std::uint64_t> > grouped;
    for (std::size_t vertex = 0; vertex < count; ++vertex) {
        grouped[destination[vertex]].push_back(static_cast<std::uint64_t>(vertex));
    }
    result.components.reserve(grouped.size());
    for (std::map<std::uint64_t, std::vector<std::uint64_t> >::iterator item =
             grouped.begin(); item != grouped.end(); ++item) {
        result.components.push_back(std::move(item->second));
    }
    result.merge_seconds = elapsed_seconds(begin, SteadyClock::now());
    return result;
}

py::array_t<std::uint64_t> materialize_wcc_labels(NativeWccResult& native) {
    const py::ssize_t count = static_cast<py::ssize_t>(native.label_count);
    if (native.uninitialized_labels) {
        const auto* data = native.uninitialized_labels.get();
        py::capsule owner(native.uninitialized_labels.get(), [](void* pointer) {
            delete[] static_cast<std::uint64_t*>(pointer);
        });
        native.uninitialized_labels.release();
        return py::array_t<std::uint64_t>(
            {count}, {static_cast<py::ssize_t>(sizeof(std::uint64_t))}, data, owner);
    }
    std::unique_ptr<std::vector<std::uint64_t> > storage(
        new std::vector<std::uint64_t>(std::move(native.labels)));
    const auto* data = storage->data();
    py::capsule owner(storage.get(), [](void* pointer) {
        delete static_cast<std::vector<std::uint64_t>*>(pointer);
    });
    storage.release();
    return py::array_t<std::uint64_t>(
        {count}, {static_cast<py::ssize_t>(sizeof(std::uint64_t))}, data, owner);
}

#include "egmgpu_wcc_serial.inc"

py::dict run_partitioned_wcc_impl(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    int max_iterations,
    const std::vector<int>& requested_device_ids,
    bool dense_output, bool public_fp64_output,
    std::string* stage) {
    *stage = "validate_input";
    if (!graph) throw std::invalid_argument("partitioned WCC graph is null");
    if (graph->vertex_count() == 0 || max_iterations <= 0) {
        throw std::invalid_argument("invalid partitioned WCC graph or iteration limit");
    }
    const char* reuse_setting = std::getenv("EGMGPU_WCC_UPLOAD_REUSE");
    const std::string reuse_mode = reuse_setting ? reuse_setting : "1";
    if (reuse_mode != "0" && reuse_mode != "1") {
        throw std::invalid_argument("EGMGPU_WCC_UPLOAD_REUSE must be 0 or 1");
    }
    const bool reuse_uploads = reuse_mode == "1";
    const char* profile_setting = std::getenv("EGMGPU_WCC_PROFILE");
    const std::string profile_mode = profile_setting ? profile_setting : "1";
    if (profile_mode != "0" && profile_mode != "1") {
        throw std::invalid_argument("EGMGPU_WCC_PROFILE must be 0 or 1");
    }
    const bool profile = profile_mode == "1";
    const char* algorithm_setting = std::getenv("EGMGPU_WCC_ALGORITHM");
    const std::string algorithm = algorithm_setting ? algorithm_setting : "label";
    if (algorithm != "label" && algorithm != "forest") {
        throw std::invalid_argument("EGMGPU_WCC_ALGORITHM must be label or forest");
    }
    const bool forest_contraction = algorithm == "forest";
    const bool boundary_csr = wcc_boundary_csr(forest_contraction);
    const WccOutputOptions output = wcc_output_options(forest_contraction);
    if(public_fp64_output && (!dense_output || !forest_contraction || !output.canonical() || !output.worker_assembly() || !output.pinned()))
        throw std::invalid_argument("WCC public_fp64_output requires canonical forest dense pinned worker or tiled assembly");
    const bool core_output_team16 = easygraph::egmgpu::owner_output::enabled(true);
    if (core_output_team16 && (!output.worker_assembly() || !output.pinned()))
        throw std::invalid_argument("WCC core output team16 requires pinned worker or tiled output");
    const bool unique_local_edges = wcc_unique_local_edges(forest_contraction);
    *stage = "resolve_devices";
    const std::vector<int> devices = requested_devices(
        *graph, requested_device_ids, stage);
    const int core_output_width = core_output_team16
        ? easygraph::egmgpu::owner_output::requested_width(devices.size()) : 1;
    const auto core_output_placement = std::make_shared<const easygraph::egmgpu::owner_output::CpuPlacement>(
        easygraph::egmgpu::owner_output::parse_placement(core_output_team16, devices.size()));
    WccHostOutputLease output_lease;
    {
        py::gil_scoped_release release;
        output_lease.acquire(devices, output.pinned());
    }
    *stage = "inspect_topology";
    const DeviceTopology topology =
        gpu_easygraph::runtime::runtime_manager().topology();
    const CommunicationBackend runtime_backend =
        gpu_easygraph::runtime::runtime_manager().device_group().backend();
    CommunicationBackend backend = CommunicationBackend::None;
    if (devices.size() > 1U) {
        const bool peer_available = topology.is_fully_peer_connected(
            DeviceGroup(devices, CommunicationBackend::PeerToPeer));
        backend = runtime_backend != CommunicationBackend::HostStaging &&
                peer_available
            ? CommunicationBackend::PeerToPeer
            : CommunicationBackend::HostStaging;
    }

    *stage = "prepare_host_plan";
    const SteadyClock::time_point preparation_begin = SteadyClock::now();
    gpu_easygraph::runtime::PartitionedHostPlanCache<
        PartitionedWccPlan>::AcquireResult plan_acquisition;
    try {
        const gpu_easygraph::runtime::PartitionedHostPlanCacheKey plan_key =
            gpu_easygraph::runtime::make_partitioned_host_plan_cache_key(
                gpu_easygraph::runtime::PartitionedHostPlanAlgorithm::Wcc,
                graph->graph_id(),
                graph->generation(),
                graph->view_signature(),
                graph->topology_signature(),
                graph->weighted(),
                graph->weight_signature(),
                graph->vertex_count(),
                graph->adjacency_count(),
                devices,
                std::string(reuse_uploads ? "prepared-uploads" : "copy-uploads") +
                    (forest_contraction ? ":forest" : ":label") +
                    (boundary_csr ? ":csr-boundary-v1" : "") +
                    (output.tiled() ? ":host-output-tiles-v1" : ""));
        py::gil_scoped_release release;
        plan_acquisition =
            gpu_easygraph::runtime::partitioned_wcc_host_plan_cache().acquire(
                plan_key,
                [&graph, &devices, reuse_uploads, forest_contraction, boundary_csr, output]() {
                    std::shared_ptr<PartitionedWccPlan> prepared(
                        new PartitionedWccPlan(gpu_easygraph::runtime::
                            build_partitioned_wcc_plan(
                                host_partitions(*graph), graph->vertex_count())));
                    if (output.tiled()) gpu_easygraph::runtime::prepare_wcc_output_gather_index(*prepared);
                    if (reuse_uploads) {
                        prepared->prepared_uploads.reserve(devices.size());
                        for (std::size_t rank = 0; rank < devices.size(); ++rank) {
                            prepared->prepared_uploads.push_back(
                                std::make_shared<WccPartitionUpload>(make_upload(
                                    *graph, *prepared, devices, rank, forest_contraction, boundary_csr)));
                        }
                    }
                    return std::shared_ptr<const PartitionedWccPlan>(prepared);
                });
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string("prepare partitioned WCC host plan: ") + error.what());
    }
    const double state_preparation_seconds = elapsed_seconds(
        preparation_begin, SteadyClock::now());
    const std::shared_ptr<const PartitionedWccPlan> plan =
        plan_acquisition.plan;
    *stage = "prepare_pinned_routes";
    const bool pinned_reuse_requested = gpu_easygraph::runtime::algorithm_candidate_flag(
        "EGGPU_WCC_PINNED_ROUTE_REUSE");
    const bool use_host_staging = backend == CommunicationBackend::HostStaging;
    const bool pinned_reuse_used = pinned_reuse_requested && use_host_staging;
    std::string pinned_key("wcc:routes:pinned:v1:");
    for (int device : devices) pinned_key += std::to_string(device) + ":";
    pinned_key += "host-staging";
    const auto pinned_begin = SteadyClock::now();
    std::shared_ptr<WccPinnedRoutes> pinned;
    // Caller owns the lease; GPU workers only borrow buffers. The pool has no
    // graph/result owner. Every active route prefix is overwritten before use.
    std::unique_lock<std::mutex> pinned_lease;
    bool pinned_cache_hit = false;
    double pinned_lease_wait_seconds = 0.0;
    double pinned_allocation_seconds = 0.0;
    std::size_t pinned_new_allocations = 0;
    std::size_t pinned_new_bytes = 0;
    std::size_t pinned_allocated_bytes = 0;
    std::size_t pinned_required_bytes = 0;
    {
        py::gil_scoped_release release;
        if (pinned_reuse_used) {
            pinned = std::static_pointer_cast<WccPinnedRoutes>(
                gpu_easygraph::runtime::runtime_manager().acquire_resident_resource(
                    pinned_key, []() { return std::shared_ptr<void>(new WccPinnedRoutes()); },
                    &pinned_cache_hit));
        } else pinned.reset(new WccPinnedRoutes());
        const auto lease_begin = SteadyClock::now();
        pinned_lease = std::unique_lock<std::mutex>(pinned->lease_mutex);
        pinned_lease_wait_seconds = elapsed_seconds(lease_begin, SteadyClock::now());
        const std::size_t old_allocations = pinned->allocation_count;
        const std::size_t old_bytes = pinned->retained_bytes;
        const std::size_t old_allocated = pinned->allocated_bytes;
        const double old_seconds = pinned->allocation_seconds;
        if (use_host_staging) pinned_required_bytes = pinned->ensure(*plan);
        pinned_new_allocations = pinned->allocation_count - old_allocations;
        pinned_new_bytes = pinned->retained_bytes - old_bytes;
        pinned_allocated_bytes = pinned->allocated_bytes - old_allocated;
        pinned_allocation_seconds = pinned->allocation_seconds - old_seconds;
    }
    const double pinned_preparation_seconds = elapsed_seconds(pinned_begin, SteadyClock::now());
    const std::size_t pinned_retained_bytes = pinned->retained_bytes;
    const auto shared_begin = SteadyClock::now();
    *stage = "allocate_shared_execution";
    std::shared_ptr<SharedWccExecution> shared;
    try {
        shared.reset(new SharedWccExecution(devices.size(), pinned));
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string("allocate partitioned WCC shared execution for ") +
            std::to_string(devices.size()) + " ranks: " + error.what());
    }

    const double shared_construction_seconds = elapsed_seconds(shared_begin, SteadyClock::now());
    shared->control_fusion_used = shared->control_fusion_requested && devices.size() > 1U;
    shared->core_output_team16 = core_output_team16;
    shared->core_output_width = core_output_width;
    shared->core_output_workers = devices.size();
    shared->core_output_placement = core_output_placement;
    std::unique_ptr<easygraph::egmgpu::WccPublicOutput> public_output;
    if(public_fp64_output){
        *stage="allocate_public_worker_result";
        const auto begin=SteadyClock::now();
        public_output.reset(new easygraph::egmgpu::WccPublicOutput(plan->vertex_count));
        shared->public_labels=public_output->data();
        shared->worker_result_allocation_bytes=plan->vertex_count*sizeof(double);
        shared->worker_result_allocation.add(elapsed_seconds(begin,SteadyClock::now()));
    } else if (output.worker_assembly()) {
        *stage = "allocate_worker_host_result";
        py::gil_scoped_release release;
        prepare_wcc_worker_result(*plan, *shared, output);
    }
    *stage = "prepare_worker_tasks";
    const bool single_partition_ordering = single_partition_candidate_enabled() &&
        devices.size() == 1U && plan->partitions.size() == 1U;
    const SteadyClock::time_point task_preparation_begin = SteadyClock::now();
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    try {
      tasks.reserve(devices.size());
      for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        WccUploadHolder upload_holder;
        if (reuse_uploads) upload_holder.prepared = plan->prepared_uploads.at(rank);
        else upload_holder.owned = make_upload(*graph, *plan, devices, rank, forest_contraction, boundary_csr);
        tasks.push_back(
            [shared, plan, upload_holder, devices, topology, backend, max_iterations, profile, output, unique_local_edges, single_partition_ordering,
             rank](int worker_device) {
                gpu_easygraph::runtime::run_cooperative_rank(
                    shared->barrier,
                    [shared, plan, upload_holder, devices, topology, backend, profile, output, unique_local_edges, single_partition_ordering,
                     max_iterations, rank, worker_device]() {
                        const WccPartitionUpload& upload = upload_holder.get();
                        shared->phases[rank] = "worker_start";
                        if (worker_device != upload.device_id) {
                            throw std::runtime_error(
                                "partitioned WCC ran on the wrong worker");
                        }
                        if (backend == CommunicationBackend::PeerToPeer) {
                            const SteadyClock::time_point peer_begin = SteadyClock::now();
                            shared->phases[rank] = "enable_peer_access";
                            gpu_easygraph::runtime::enable_worker_peer_access(
                                DeviceGroup(
                                    devices, CommunicationBackend::PeerToPeer),
                                topology);
                            shared->ranks[rank].timing.peer_enable.add(
                                elapsed_seconds(peer_begin, SteadyClock::now()));
                        }
                        RankState& state = shared->ranks[rank];
                        shared->phases[rank] = "acquire_resident_state";
                        gpu_easygraph::runtime::PartitionedWccStateRegistry& registry =
                            gpu_easygraph::runtime::partitioned_wcc_state_registry();
                        const SteadyClock::time_point acquire_begin = SteadyClock::now();
                        state.view = registry.acquire(
                            upload, &state.device_view_cache_hit,
                            &state.shared_arrays);
                        state.workspace = registry.workspace(state.view);
                        prepare_wcc_host_output(state, output);
                        state.timing.state_acquire.add(elapsed_seconds(acquire_begin, SteadyClock::now()));
                        state.resident_bytes = state.view->resident_bytes() +
                            state.workspace->capacity_bytes();
                        const WccPartitionDeviceView& view = state.view->view();
                        WccPartitionWorkspace& workspace = *state.workspace;
                        WorkerStreams& streams =
                            gpu_easygraph::runtime::worker_streams();
                        if (single_partition_ordering &&
                            (view.local_vertex_count != view.owned_vertex_count ||
                             view.boundary_edge_count != 0U))
                            throw std::logic_error("P1 WCC requires a complete owner with no boundary edges");
                        const bool nonempty = view.owned_vertex_count != 0;
                        if (nonempty) {
                            shared->phases[rank] = "initialize_labels";
                            CudaEventTimer initialize_timer(
                                "measure WCC initialization", profile);
                            initialize_timer.start(streams.compute());
                            gpu_easygraph::partitioned::initialize_wcc_labels(
                                view.local_to_global, workspace.labels(),
                                view.local_vertex_count, streams.compute());
                            if (view.forest_contraction) {
                                gpu_easygraph::partitioned::initialize_wcc_local_parents(
                                    workspace.local_parents(), view.owned_vertex_count, streams.compute());
                            }
                            const double initialize_seconds = initialize_timer.finish(streams.compute());
                            state.timing.initialization.add(initialize_seconds);
                            state.timing.kernel.add(initialize_seconds);
                            if (view.forest_contraction) {
                                shared->phases[rank] = "union_local_forest";
                                CudaEventTimer union_timer("measure WCC local union", profile);
                                union_timer.start(streams.compute());
                                gpu_easygraph::partitioned::union_wcc_local_edges(
                                    view.offsets, view.indices, workspace.local_parents(),
                                    view.owned_vertex_count, streams.compute(), unique_local_edges);
                                const double union_seconds = union_timer.finish(streams.compute());
                                state.timing.local_union.add(union_seconds);
                                state.timing.kernel.add(union_seconds);
                                CudaEventTimer compression_timer("measure WCC forest compression", profile);
                                compression_timer.start(streams.compute());
                                gpu_easygraph::partitioned::compress_wcc_local_parents(
                                    workspace.local_parents(), view.owned_vertex_count, streams.compute());
                                const double compression_seconds = compression_timer.finish(streams.compute());
                                state.timing.forest_compression.add(compression_seconds);
                                state.timing.kernel.add(compression_seconds);
                                CudaEventTimer projection_timer("measure WCC initial component projection", profile);
                                projection_timer.start(streams.compute());
                                gpu_easygraph::partitioned::project_wcc_component_labels(
                                    workspace.local_parents(), workspace.labels(),
                                    view.owned_vertex_count, streams.compute());
                                const double projection_seconds = projection_timer.finish(streams.compute());
                                state.timing.component_projection.add(projection_seconds);
                                state.timing.kernel.add(projection_seconds);
                            }
                        }
                        timed_barrier(
                            shared, &state.timing, devices.size() > 1U);

                        for (int iteration = 1;
                             iteration <= max_iterations; ++iteration) {
                            if (nonempty) {
                                shared->phases[rank] = "local_hook_compress";
                                CudaEventTimer local_timer(
                                    "measure WCC local hooking", profile);
                                local_timer.start(streams.compute());
                                gpu_easygraph::partitioned::clear_wcc_changed(
                                    workspace.changed(), streams.compute());
                                if (!view.forest_contraction) {
                                    gpu_easygraph::partitioned::local_wcc_hook_compress(
                                        view.offsets, view.indices, view.owners,
                                        view.local_to_global, workspace.labels(),
                                        view.owned_vertex_count, view.partition_id,
                                        workspace.changed(), streams.compute());
                                }
                                const double local_seconds = local_timer.finish(streams.compute());
                                if (view.forest_contraction) state.timing.changed_clear.add(local_seconds);
                                else state.timing.local_hook_compress.add(local_seconds);
                                state.timing.kernel.add(local_seconds);
                            }

                            const SteadyClock::time_point communication_begin =
                                SteadyClock::now();
                            shared->phases[rank] = "owner_halo_exchange";
                            pack_routes(
                                view, workspace, streams, &state.timing, profile);
                            if (backend == CommunicationBackend::HostStaging) {
                                stage_outgoing_routes(
                                    shared, *plan, rank, streams, &state.timing, profile);
                            }
                            timed_barrier(
                                shared, &state.timing, devices.size() > 1U);
                            if (backend == CommunicationBackend::PeerToPeer) {
                                peer_transfer_routes(
                                    shared, *plan, rank, worker_device, streams,
                                    &state.timing, profile);
                            } else if (backend == CommunicationBackend::HostStaging) {
                                stage_incoming_routes(
                                    shared, *plan, rank, streams, &state.timing, profile);
                            }
                            scatter_routes(
                                view, workspace, streams, &state.timing, profile);
                            timed_barrier(
                                shared, &state.timing, devices.size() > 1U);
                            if (profile && devices.size() > 1U) {
                                state.timing.communication.add(elapsed_seconds(
                                    communication_begin, SteadyClock::now()));
                            }

                            if (nonempty) {
                                shared->phases[rank] = "boundary_min_combine";
                                CudaEventTimer boundary_timer(
                                    "measure WCC boundary min-combine", profile);
                                boundary_timer.start(streams.compute());
                                if (view.forest_contraction) {
                                    combine_forest_boundary(view, workspace, streams.compute());
                                } else {
                                    gpu_easygraph::partitioned::boundary_wcc_hook_compress(
                                        view.offsets, view.indices, view.owners,
                                        view.local_to_global, workspace.labels(),
                                        view.owned_vertex_count, view.partition_id,
                                        workspace.changed(), streams.compute());
                                }
                                const double boundary_seconds = boundary_timer.finish(streams.compute());
                                state.timing.boundary_combine.add(boundary_seconds);
                                state.timing.kernel.add(boundary_seconds);
                                // P1 has no boundary updates after initial forest projection.
                                if (view.forest_contraction && !single_partition_ordering) {
                                    CudaEventTimer projection_timer("measure WCC component projection", profile);
                                    projection_timer.start(streams.compute());
                                    gpu_easygraph::partitioned::project_wcc_component_labels(
                                        workspace.local_parents(), workspace.labels(),
                                        view.owned_vertex_count, streams.compute());
                                    const double projection_seconds = projection_timer.finish(streams.compute());
                                    state.timing.component_projection.add(projection_seconds);
                                    state.timing.kernel.add(projection_seconds);
                                }
                                const SteadyClock::time_point changed_begin = SteadyClock::now();
                                check_cuda(
                                    cudaMemcpyAsync(
                                        &state.host_changed, workspace.changed(),
                                        sizeof(int), cudaMemcpyDeviceToHost,
                                        streams.compute()),
                                    "copy WCC changed flag");
                                check_cuda(
                                    cudaStreamSynchronize(streams.compute()),
                                    "synchronize WCC changed flag");
                                state.timing.changed_d2h.add(elapsed_seconds(changed_begin, SteadyClock::now()));
                            } else {
                                state.host_changed = 0;
                            }
                            shared->phases[rank] = "global_convergence";
                            timed_control_reduction(
                                shared, &state.timing, devices.size() > 1U, rank, [&]() {
                                shared->any_changed = false;
                                for (std::size_t peer = 0;
                                     peer < shared->ranks.size(); ++peer) {
                                    shared->any_changed = shared->any_changed ||
                                        shared->ranks[peer].host_changed != 0;
                                }
                                shared->iterations = iteration;
                                shared->converged = !shared->any_changed;
                            });
                            if (shared->converged) break;
                        }

                        if (nonempty) {
                            shared->phases[rank] = "output_gather";
                            const SteadyClock::time_point gather_begin =
                                SteadyClock::now();
                            if (output.pinned()) {
                                check_cuda(cudaMemcpyAsync(state.pinned_labels->data(), workspace.labels(),
                                    view.owned_vertex_count * sizeof(std::uint64_t),
                                    cudaMemcpyDeviceToHost, streams.compute()), "gather pinned WCC labels");
                                check_cuda(cudaStreamSynchronize(streams.compute()), "complete pinned WCC labels");
                            } else {
                                state.owned_labels.resize(view.owned_vertex_count);
                                check_cuda(cudaMemcpy(state.owned_labels.data(), workspace.labels(),
                                    view.owned_vertex_count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
                                    "gather partitioned WCC labels");
                            }
                            state.timing.output_gather.add(elapsed_seconds(
                                gather_begin, SteadyClock::now()));
                        }
                        if (shared->worker_labels || shared->public_labels) {
                            if (output.tiled()) {
                                // Every rank publishes its completed pinned D2H buffer.
                                // The barrier makes all pointer/buffer writes visible;
                                // the caller-held output lease excludes later reuse.
                                shared->output_sources[rank] = state.labels_data();
                                shared->phases[rank] = "host_output_readiness";
                                const auto ready_begin = SteadyClock::now();
                                if (devices.size()>1U) shared->barrier.arrive_and_wait();
                                state.timing.host_output_ready_wait.add(elapsed_seconds(ready_begin, SteadyClock::now()));
                                shared->phases[rank] = "host_output_tiled_gather";
                                assemble_wcc_tiled_result(*plan, *shared, state, rank, devices.size());
                            } else {
                                shared->phases[rank] = "host_owner_assembly";
                                assemble_wcc_worker_result(plan->partitions[rank], *shared, state, rank);
                            }
                        }
                        shared->phases[rank] = "complete";
                    });
            });
      }
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string("prepare partitioned WCC worker tasks: ") + error.what());
    }

    *stage = "execute_worker_gang";
    const double host_task_preparation_seconds = elapsed_seconds(
        task_preparation_begin, SteadyClock::now());
    const SteadyClock::time_point device_begin = SteadyClock::now();
    // Once tasks can produce DMA, an exceptional exit must never make this
    // runtime-retained pool reusable. Only the full success boundary clears it.
    if (pinned_reuse_used) pinned->begin_query();
    {
        py::gil_scoped_release release;
        try {
            gpu_easygraph::runtime::runtime_manager().execute_gang(
                devices, std::move(tasks));
        } catch (const std::exception& error) {
            std::string context("partitioned WCC worker failure (");
            for (std::size_t rank = 0; rank < shared->phases.size(); ++rank) {
                if (rank) context += ", ";
                context += "rank " + std::to_string(rank) + ":" +
                    shared->phases[rank];
            }
            context += "): ";
            context += error.what();
            throw std::runtime_error(context);
        }
    }
    const double device_processing_seconds = elapsed_seconds(
        device_begin, SteadyClock::now());
    if (!shared->converged) {
        throw std::runtime_error(
            "partitioned WCC did not converge within max_iterations");
    }

    // Every producer has completed before any cross-label certificate read.
    double public_certificate_seconds=0.0;
    easygraph::egmgpu::owner_output::Observation certificate_threads;
    if(public_output){
        *stage="certify_public_worker_result";
        std::vector<int> cpus;
        if(core_output_placement->explicit_binding)
            for(const auto& group:core_output_placement->workers)cpus.insert(cpus.end(),group.begin(),group.end());
        const auto begin=SteadyClock::now();
        {
            py::gil_scoped_release release;
            public_output->certify(shared->public_labels,shared->public_observation,
                core_output_team16?16:1,certificate_threads,core_output_placement->explicit_binding?&cpus:nullptr);
        }
        public_certificate_seconds=elapsed_seconds(begin,SteadyClock::now());
    }
    *stage = "merge_native_result";
    NativeWccResult native;
    {
        py::gil_scoped_release release;
        native = merge_wcc_result(*plan, shared, dense_output, output.canonical(), output.direct());
        output_lease.release();
    }

    *stage = "materialize_components";
    const SteadyClock::time_point materialization_begin = SteadyClock::now();
    py::list components;
    py::array labels_dense;
    if(public_output) labels_dense=public_output->publish();
    else if (dense_output) {
        labels_dense = materialize_wcc_labels(native);
    }
    for (std::size_t component = 0;
         component < native.components.size(); ++component) {
        py::set vertices;
        for (std::size_t index = 0;
             index < native.components[component].size(); ++index) {
            vertices.add(py::int_(native.components[component][index]));
        }
        components.append(vertices);
    }
    const double python_materialization_seconds = elapsed_seconds(
        materialization_begin, SteadyClock::now());

    *stage = "materialize_diagnostics";
    py::list per_device;
    py::list resident_device_bytes;
    std::uint64_t shared_array_hits = 0U;
    std::uint64_t shared_array_new_bytes = 0U;
    std::uint64_t shared_array_retained_bytes = 0U;
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        const RankState& state = shared->ranks[rank];
        py::dict item;
        item["device_id"] = devices[rank];
        append_wcc_output_metadata(item, state);
        item["kernel_seconds"] = timing_value(state.timing.kernel);
        item["initialization_cuda_seconds"] = timing_value(state.timing.initialization);
        item["local_union_cuda_seconds"] = timing_value(state.timing.local_union);
        item["local_hook_compress_cuda_seconds"] = timing_value(state.timing.local_hook_compress);
        item["changed_clear_cuda_seconds"] = timing_value(state.timing.changed_clear);
        item["changed_d2h_wall_seconds"] = timing_value(state.timing.changed_d2h);
        item["halo_scatter_cuda_seconds"] = timing_value(state.timing.halo_scatter);
        item["boundary_combine_cuda_seconds"] = timing_value(state.timing.boundary_combine);
        item["forest_compression_cuda_seconds"] = timing_value(state.timing.forest_compression);
        item["component_projection_cuda_seconds"] = timing_value(state.timing.component_projection);
        item["state_acquire_wall_seconds"] = timing_value(state.timing.state_acquire);
        item["peer_enable_wall_seconds"] = timing_value(state.timing.peer_enable);
        item["boundary_edge_count"] = state.view->view().boundary_edge_count;
        item["pack_seconds"] = timing_value(state.timing.pack);
        item["transfer_seconds"] = timing_value(state.timing.transfer);
        item["wait_seconds"] = timing_value(state.timing.wait);
        item["communication_seconds"] =
            timing_value(state.timing.communication);
        item["output_gather_seconds"] =
            timing_value(state.timing.output_gather);
        item["resident_device_bytes"] =
            static_cast<std::uint64_t>(state.resident_bytes);
        item["device_view_cache_hit"] = state.device_view_cache_hit;
        item["shared_array_hits"] = static_cast<std::uint64_t>(
            state.shared_arrays.hits);
        item["shared_array_new_bytes"] = static_cast<std::uint64_t>(
            state.shared_arrays.new_bytes);
        item["shared_array_retained_bytes"] = static_cast<std::uint64_t>(
            state.shared_arrays.retained_bytes);
        shared_array_hits += state.shared_arrays.hits;
        shared_array_new_bytes += state.shared_arrays.new_bytes;
        shared_array_retained_bytes += state.shared_arrays.retained_bytes;
        resident_device_bytes.append(
            static_cast<std::uint64_t>(state.resident_bytes));
        per_device.append(item);
    }

    const std::uint64_t logical_bytes =
        static_cast<std::uint64_t>(shared->iterations) *
        plan->halo_value_count * sizeof(std::uint64_t);
    const std::uint64_t physical_bytes = backend ==
            CommunicationBackend::HostStaging
        ? logical_bytes * 2U
        : logical_bytes;
    py::dict out;
    if(public_output){
        out["wcc_worker_public_output"]=easygraph::egmgpu::wcc_public_output_proof(
            plan->vertex_count,shared->public_observation,certificate_threads.actual);
        out["wcc_public_certificate_seconds"]=public_certificate_seconds;
        out["wcc_public_certificate_thread_observation"]=easygraph::egmgpu::owner_output::to_python(certificate_threads);
        out["output_representation"]="owning_fp64_dense";
    }
    if (dense_output) out["labels_dense"] = std::move(labels_dense);
    else out["components"] = std::move(components);
    py::dict reduction_proof;
    reduction_proof["version"] = "cooperative_control_completion_v1";
    reduction_proof["flag"] = "EGGPU_ALGORITHM_CONTROL_FUSION";
    reduction_proof["requested"] = shared->control_fusion_requested;
    reduction_proof["used"] = shared->control_fusion_used;
    reduction_proof["eligible"] = devices.size() > 1U;
    reduction_proof["reason"] = !shared->control_fusion_requested ? "disabled" :
        (shared->control_fusion_used ? "multi_rank_control_completion" : "single_rank_or_unsupported_route");
    reduction_proof["completed_reductions"] = shared->control_reduction_calls;
    reduction_proof["barriers_elided_per_rank"] = shared->control_reduction_calls;
    reduction_proof["scope"] = "multi-rank global convergence; no GPU completion or payload lifetime boundary removed";
    out["algorithm_control_reduction"] = reduction_proof;
    out["result_format"] = dense_output ? "dense_labels" : "components";
    out["upload_descriptor_reuse"] = reuse_uploads;
    out["boundary_representation"] = boundary_csr ? "csr" : "coo";
    out["wcc_algorithm"] = algorithm;
    out["host_output_mode"] = output.transfer;
    out["host_output_pinned_placement"] = output.pinned() ? output.placement : "not_applicable";
    out["result_merge_mode"] = output.merge;
    append_wcc_assembly_metadata(out, *shared, output, *plan, plan_acquisition.builder);
    out["local_union_mode"] = unique_local_edges ? "unique_edges" : "all_arcs";
    out["result_ownership"] = "independent complete host labels; retained across queries and runtime shutdown";
    out["host_output_lease_wait_seconds"] = output_lease.wait_seconds;
    out["single_partition_projection_reuse"] = forest_contraction && single_partition_ordering;
    out["profile_enabled"] = profile;
    out["device_phase_timing_semantics"] = profile
        ? "CUDA events with phase completion waits"
        : "unmeasured; only stream publication and convergence/output completion waits";
    out["host_task_preparation_seconds"] = host_task_preparation_seconds;
    out["iterations"] = shared->iterations;
    out["converged"] = shared->converged;
    out["devices_used"] = devices;
    out["communication_backend"] = std::string(
        gpu_easygraph::runtime::communication_backend_name(backend));
    out["state_preparation_seconds"] = state_preparation_seconds;
    out["host_plan_cache_hit"] = plan_acquisition.cache_hit;
    out["host_plan_cache_builder"] = plan_acquisition.builder;
    out["host_plan_cache_waiter"] = plan_acquisition.waiter;
    if (plan_acquisition.build_seconds_measured) {
        out["host_plan_build_seconds"] = plan_acquisition.build_seconds;
    } else {
        out["host_plan_build_seconds"] = py::none();
    }
    out["host_plan_cache_entries"] = plan_acquisition.entry_count;
    out["device_processing_seconds"] = device_processing_seconds;
    out["kernel_seconds"] = timing_value(native.kernel);
    out["pack_seconds"] = timing_value(native.pack);
    out["transfer_seconds"] = timing_value(native.transfer);
    out["wait_seconds"] = timing_value(native.wait);
    out["communication_seconds"] = timing_value(native.communication);
    out["output_gather_seconds"] = timing_value(native.output_gather);
    out["native_merge_seconds"] = native.merge_seconds;
    out["python_materialization_seconds"] = python_materialization_seconds;
    out["logical_communication_bytes"] = logical_bytes;
    out["physical_transfer_bytes"] = physical_bytes;
    out["resident_device_bytes"] = resident_device_bytes;
    out["aggregate_resident_device_bytes"] = native.resident_bytes;
    out["shared_array_hits"] = shared_array_hits;
    out["shared_array_new_bytes"] = shared_array_new_bytes;
    out["shared_array_retained_bytes"] = shared_array_retained_bytes;
    out["per_device"] = per_device;
    // All accepted GPU work, full host merge and canonical certification are
    // complete. Retire query aliases before making reusable storage available.
    // Off-arm cudaFreeHost costs are measured here, within the public call.
    *stage = "release_pinned_routes";
    const auto pinned_release_begin = SteadyClock::now();
    {
        py::gil_scoped_release release;
        if (pinned_reuse_used) pinned->complete_query();
        shared->host_buffers.clear();
        pinned_lease.unlock();
        pinned.reset();
    }
    const double pinned_release_seconds = elapsed_seconds(pinned_release_begin, SteadyClock::now());
    py::dict pinned_proof;
    pinned_proof["version"] = "wcc_runtime_pinned_routes_v2";
    pinned_proof["flag"] = "EGGPU_WCC_PINNED_ROUTE_REUSE";
    pinned_proof["requested"] = pinned_reuse_requested;
    pinned_proof["eligible"] = use_host_staging;
    pinned_proof["used"] = pinned_reuse_used;
    pinned_proof["cache_hit"] = pinned_cache_hit;
    pinned_proof["reason"] = !pinned_reuse_requested ? "disabled" :
        (use_host_staging ? "host_staging_runtime_owned_pool" : "no_host_staging_routes");
    pinned_proof["key"] = pinned_reuse_used ? pinned_key : "not_retained";
    pinned_proof["required_host_bytes"] = pinned_required_bytes;
    pinned_proof["retained_host_bytes"] = pinned_retained_bytes;
    pinned_proof["retained_after_call_host_bytes"] = pinned_reuse_used ? pinned_retained_bytes : 0U;
    pinned_proof["new_allocation_count"] = pinned_new_allocations;
    pinned_proof["capacity_growth_host_bytes"] = pinned_new_bytes;
    pinned_proof["allocated_host_bytes"] = pinned_allocated_bytes;
    pinned_proof["preparation_seconds"] = pinned_preparation_seconds;
    pinned_proof["lease_wait_seconds"] = pinned_lease_wait_seconds;
    pinned_proof["allocation_and_growth_release_seconds"] = pinned_allocation_seconds;
    pinned_proof["query_release_seconds"] = pinned_release_seconds;
    pinned_proof["shared_construction_seconds"] = shared_construction_seconds;
    pinned_proof["query_completed_before_reuse"] = pinned_reuse_used;
    pinned_proof["failed_query_policy"] = pinned_reuse_used
        ? "retain_pool_and_reject_reuse_until_runtime_teardown" : "not_applicable_no_runtime_pool";
    pinned_proof["scope"] = "Pinned HOST route storage only, not device bytes or result arrays; runtime-stop release excluded from query timer; lease_wait/allocation nested in preparation; public certificate unchanged";
    out["wcc_pinned_routes"] = pinned_proof;
    py::list offset_layouts;
    for (const auto& part : graph->partitions()) {
        py::dict row; row["partition_id"] = part.partition_id;
        row["dtype"] = part.wide_offsets ? "int64" : "int32";
        row["bytes"] = part.offset_payload_bytes();
        row["sha256"] = part.offsets_sha256;
        offset_layouts.append(row);
    }
    out["wcc_offset_layouts"] = offset_layouts;
    out["wcc_offset_policy"] = "adaptive_signed_i32_i64_wcc_v1";
    *stage = "complete";
    return out;
}

py::dict run_partitioned_wcc(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    int max_iterations,
    const std::vector<int>& requested_device_ids,
    bool dense_output, bool public_fp64_output) {
    std::string stage("enter");
    try {
        return run_partitioned_wcc_impl(
            graph, max_iterations, requested_device_ids, dense_output, public_fp64_output, &stage);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "partitioned WCC at " + stage + ": " + error.what());
    }
}

py::dict run_serial_partitioned_wcc_core_output_guard(
    const std::shared_ptr<MultiPartitionGraph>& graph, int max_iterations,
    int device, bool dense_output) {
    return run_serial_partitioned_wcc(graph, max_iterations, device, dense_output);
}
#endif

}  // namespace

void bind_egmgpu_partitioned_wcc(py::module_& module) {
    module.def("cpp_egmgpu_finalize_wcc_uint64", &easygraph::egmgpu::finalize_wcc_uint64,
        py::arg("labels"), py::arg("count"),
        "Certify a complete native uint64 WCC label vector and materialize fresh owning float64 values on CPU.");
#ifdef EASYGRAPH_ENABLE_GPU
    module.def("cpp_egmgpu_partitioned_wcc_serial", &run_serial_partitioned_wcc_core_output_guard,
        py::arg("partitioned_graph"), py::arg("max_iterations") = 100000,
        py::arg("device") = 0, py::arg("dense_output") = true,
        "Run the same WCC local-forest partitions sequentially on one GPU.");
    module.def(
        "cpp_egmgpu_partitioned_wcc",
        &run_partitioned_wcc,
        py::arg("partitioned_graph"),
        py::arg("max_iterations") = 100000,
        py::arg("devices") = std::vector<int>(),
        py::arg("dense_output") = false,
        py::arg("public_fp64_output") = false,
        "Run owner-partitioned weakly connected components; optional worker FP64 output preserves the full canonical certificate.");
#else
    (void)module;
#endif
}
