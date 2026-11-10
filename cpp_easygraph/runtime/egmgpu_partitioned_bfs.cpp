#include "egmgpu_partitioned_bfs.h"
#include "egmgpu_bfs_host_finalize.h"
#include "egmgpu_bfs_public_output.h"

#ifdef EASYGRAPH_ENABLE_GPU
#include "egmgpu_partition_loader.h"
#include "egmgpu_partition_array_scope.h"
#include "egmgpu_bfs_host_assembly.h"
#include "egmgpu_bfs_thread_usage.h"
#include "egmgpu_owner_output_team_py.h"

#include <functions/partitioned/bfs_partitioned.cuh>
#include <runtime/collective_transport.h>
#include <runtime/cooperative_barrier.h>
#include <runtime/algorithm_control_policy.h>
#include <runtime/device_phase_timer.h>
#include <runtime/native_runtime.h>
#include <runtime/numa_pinned_host.h>
#include <runtime/partitioned_bfs_plan.h>
#include <runtime/partitioned_bfs_state.h>
#include <runtime/partitioned_host_plan_registry.h>

#include <cuda_runtime_api.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <map>
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
using gpu_easygraph::runtime::BfsHostPartition;
using gpu_easygraph::runtime::BfsDenseRouteLayout;
using gpu_easygraph::runtime::BfsFrontierKind;
using gpu_easygraph::runtime::BfsFrontierMetrics;
using gpu_easygraph::runtime::BfsFrontierPolicy;
using gpu_easygraph::runtime::BfsFrontierThresholds;
using gpu_easygraph::runtime::BfsOwnerRoute;
using gpu_easygraph::runtime::BfsPartitionDeviceView;
using gpu_easygraph::runtime::BfsPartitionUpload;
using gpu_easygraph::runtime::BfsPartitionWorkspace;
using gpu_easygraph::runtime::CommunicationBackend;
using gpu_easygraph::runtime::CooperativeBarrier;
using gpu_easygraph::runtime::DeviceGroup;
using gpu_easygraph::runtime::DeviceTopology;
using gpu_easygraph::runtime::PartitionedBfsPlan;
using gpu_easygraph::runtime::PinnedHostBuffer;
using gpu_easygraph::runtime::ResidentBfsPartitionView;
using gpu_easygraph::runtime::SharedArrayCallStats;
using gpu_easygraph::runtime::WorkerStreams;
namespace owner_output = easygraph::egmgpu::owner_output;

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

bool bfs_profile_setting(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (!value) return default_value;
    if (std::string(value) == "0") return false;
    if (std::string(value) == "1") return true;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

bool bfs_direct_result_assembly() {
    const char* value = std::getenv("EGMGPU_BFS_RESULT_ASSEMBLY");
    if (!value || std::string(value) == "zeroed") return false;
    if (std::string(value) == "direct") return true;
    throw std::invalid_argument("EGMGPU_BFS_RESULT_ASSEMBLY must be zeroed or direct");
}

bool bfs_numa_local_output() {
    const char* value = std::getenv("EGMGPU_BFS_PINNED_PLACEMENT");
    if (!value || std::string(value) == "caller") return false;
    if (std::string(value) == "numa_local") return true;
    throw std::invalid_argument("EGMGPU_BFS_PINNED_PLACEMENT must be caller or numa_local");
}

struct BfsDenseDirection {
    bool pull = false;
    int lanes = 1;
};

bool bfs_worker_gpu_numa() {
    const char* setting = std::getenv("EGMGPU_BFS_WORKER_CPU_PLACEMENT");
    const std::string value = setting ? setting : "none";
    if (value == "none") return false;
    if (value == "gpu_numa") return true;
    throw std::invalid_argument("EGMGPU_BFS_WORKER_CPU_PLACEMENT must be none or gpu_numa");
}

py::object bfs_worker_cpu_placement_dict(const gpu_easygraph::runtime::GpuWorkerCpuPlacementInfo& info) {
    if (!info.observed) return py::none();
    py::dict out;
    out["logical_device"] = info.logical_device; out["target_numa_node"] = info.numa_node;
    out["pci_bus_id"] = info.pci_bus_id;
    out["cpu_before"] = info.cpu_before; out["cpu_during"] = info.cpu_during; out["cpu_after"] = info.cpu_after;
    out["mask_before"] = info.mask_before; out["mask_during"] = info.mask_during; out["mask_after"] = info.mask_after;
    return std::move(out);
}

enum class BfsHostAssembly { Legacy, Caller, Workers };

BfsHostAssembly bfs_host_assembly(bool move_output, bool direct_assembly) {
    const char* setting = std::getenv("EGMGPU_BFS_HOST_ASSEMBLY");
    const std::string mode = setting ? setting : "legacy";
    if (mode == "legacy") return BfsHostAssembly::Legacy;
    if (mode != "caller" && mode != "workers")
        throw std::invalid_argument("EGMGPU_BFS_HOST_ASSEMBLY must be legacy, caller, or workers");
    if (!move_output || !direct_assembly)
        throw std::invalid_argument("BFS caller/workers host assembly requires move output and direct result assembly");
    return mode == "caller" ? BfsHostAssembly::Caller : BfsHostAssembly::Workers;
}

const char* bfs_host_assembly_name(BfsHostAssembly mode) {
    return mode == BfsHostAssembly::Workers ? "workers" : mode == BfsHostAssembly::Caller ? "caller" : "legacy";
}

std::unique_ptr<std::int64_t[]> allocate_bfs_host_output(
    std::uint64_t count, BfsHostAssembly mode) {
    if (mode == BfsHostAssembly::Legacy) return std::unique_ptr<std::int64_t[]>();
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t))
        throw std::overflow_error("BFS result exceeds host address space");
    // Allocation belongs to the query. Pages are first touched by the actual
    // owner copy, without pre-faulting or initialization outside its interval.
    return std::unique_ptr<std::int64_t[]>(new std::int64_t[static_cast<std::size_t>(count)]);
}

int bfs_sparse_ctas_per_row(const std::string& edge_schedule) {
    const char* setting = std::getenv("EGMGPU_BFS_SPARSE_SCHEDULE");
    const std::string mode = setting ? setting : "warp";
    if (mode == "warp") return 0;
    if (mode != "cta1" && mode != "cta4")
        throw std::invalid_argument("EGMGPU_BFS_SPARSE_SCHEDULE must be warp, cta1, or cta4");
    if (edge_schedule != "warp")
        throw std::invalid_argument("BFS high-degree CTA modes require EDGE_SCHEDULE=warp");
    return mode == "cta1" ? 1 : 4;
}

const char* bfs_sparse_schedule_name(int ctas_per_row) {
    return ctas_per_row == 4 ? "cta4" : ctas_per_row == 1 ? "cta1" : "warp";
}

BfsDenseDirection bfs_dense_direction(const MultiPartitionGraph& graph) {
    BfsDenseDirection result;
    const char* direction = std::getenv("EGMGPU_BFS_DENSE_DIRECTION");
    const std::string mode = direction ? direction : "push";
    if (mode != "push" && mode != "pull")
        throw std::invalid_argument("EGMGPU_BFS_DENSE_DIRECTION must be push or pull");
    result.pull = mode == "pull";
    if (result.pull && (graph.directed() || graph.view_orientation() != "undirected"))
        throw std::invalid_argument("BFS dense pull requires a validated symmetric undirected CSR; directed pull needs a separate incoming view");
    const char* lanes = std::getenv("EGMGPU_BFS_PULL_LANES");
    const std::string width = lanes ? lanes : "1";
    if (width != "1" && width != "8" && width != "32")
        throw std::invalid_argument("EGMGPU_BFS_PULL_LANES must be 1, 8, or 32");
    result.lanes = std::stoi(width);
    return result;
}

// Fixed runtime workers serialize tasks on each thread. Capture the query's
// explicit policy once, and restore it even when a cooperative rank fails.
thread_local bool bfs_worker_profile_enabled = true;
class BfsWorkerProfileScope {
public:
    explicit BfsWorkerProfileScope(bool enabled)
        : previous_(bfs_worker_profile_enabled) {
        bfs_worker_profile_enabled = enabled;
    }
    ~BfsWorkerProfileScope() { bfs_worker_profile_enabled = previous_; }
private:
    bool previous_;
};

struct MeasuredSeconds {
    bool measured;
    double seconds;

    MeasuredSeconds() : measured(false), seconds(0.0) {}
    void add(double value) {
        measured = true;
        seconds += value;
    }
    void add(const MeasuredSeconds& value) {
        if (value.measured) add(value.seconds);
    }
};

// Fixed per query before worker dispatch; use one setting for an entire A/B process.
bool single_partition_candidate_enabled() {
    const char* value = std::getenv("EGMGPU_SINGLE_PARTITION_ORDERING");
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) == "1") return true;
    throw std::invalid_argument("EGMGPU_SINGLE_PARTITION_ORDERING must be 0 or 1");
}

class CudaEventTimer {
public:
    explicit CudaEventTimer(const char* context)
        : begin_(nullptr), end_(nullptr), context_(context),
          enabled_(bfs_worker_profile_enabled) {
        if (!enabled_) return;
        check_cuda(cudaEventCreate(&begin_), "create BFS timing event");
        try {
            check_cuda(cudaEventCreate(&end_), "create BFS timing event");
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
        if (enabled_) check_cuda(cudaEventRecord(begin_, stream), context_);
    }
    MeasuredSeconds finish(cudaStream_t stream, bool require_completion = true) {
        MeasuredSeconds result;
        if (!enabled_) {
            // Existing callers use this completion before switching streams,
            // publishing peer buffers, or reading D2H counters. Keep those
            // dependencies; removing a redundant wait is a separate factor.
            if (require_completion) check_cuda(cudaStreamSynchronize(stream), context_);
            return result;
        }
        check_cuda(cudaEventRecord(end_, stream), context_);
        check_cuda(cudaEventSynchronize(end_), context_);
        float milliseconds = 0.0F;
        check_cuda(cudaEventElapsedTime(&milliseconds, begin_, end_), context_);
        result.add(static_cast<double>(milliseconds) / 1000.0);
        return result;
    }
private:
    cudaEvent_t begin_;
    cudaEvent_t end_;
    const char* context_;
    bool enabled_;
};

py::dict bfs_usage_sample_dict(const easygraph::egmgpu::BfsThreadUsageSample& sample) {
    py::dict out;
    out["steady_ns"] = sample.steady_ns; out["monotonic_ns"] = sample.monotonic_ns;
    out["thread_cpu_ns"] = sample.thread_cpu_ns;
    out["user_cpu_us"] = sample.user_cpu_us; out["system_cpu_us"] = sample.system_cpu_us;
    out["minor_faults"] = sample.minor_faults; out["major_faults"] = sample.major_faults;
    out["voluntary_switches"] = sample.voluntary_switches;
    out["involuntary_switches"] = sample.involuntary_switches;
    out["tid"] = sample.tid; out["cpu"] = sample.cpu;
    return out;
}

py::object bfs_usage_window_dict(const easygraph::egmgpu::BfsThreadUsageWindow& window) {
    if (!window.measured) return py::none();
    const auto& a = window.begin; const auto& b = window.end;
    py::dict out;
    out["begin"] = bfs_usage_sample_dict(a); out["end"] = bfs_usage_sample_dict(b);
    out["steady_wall_seconds"] = (b.steady_ns - a.steady_ns) / 1e9;
    out["monotonic_wall_seconds"] = (b.monotonic_ns - a.monotonic_ns) / 1e9;
    out["thread_cpu_seconds"] = (b.thread_cpu_ns - a.thread_cpu_ns) / 1e9;
    out["user_cpu_seconds"] = (b.user_cpu_us - a.user_cpu_us) / 1e6;
    out["system_cpu_seconds"] = (b.system_cpu_us - a.system_cpu_us) / 1e6;
    out["minor_faults"] = b.minor_faults - a.minor_faults;
    out["major_faults"] = b.major_faults - a.major_faults;
    out["voluntary_switches"] = b.voluntary_switches - a.voluntary_switches;
    out["involuntary_switches"] = b.involuntary_switches - a.involuntary_switches;
    out["wall_minus_thread_cpu_seconds"] =
        (b.steady_ns - a.steady_ns - b.thread_cpu_ns + a.thread_cpu_ns) / 1e9;
    return std::move(out);
}

py::object timing_value(const MeasuredSeconds& value) {
    if (!value.measured) return py::none();
    return py::float_(value.seconds);
}

std::vector<int> requested_devices(
    const MultiPartitionGraph& graph, const py::object& request) {
    const gpu_easygraph::runtime::RuntimeStatus status =
        gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running) {
        throw std::runtime_error(
            "EGMGPU native runtime must be started before partitioned BFS");
    }
    std::vector<int> devices = request.is_none()
        ? std::vector<int>()
        : request.cast<std::vector<int> >();
    if (devices.empty()) {
        devices.reserve(graph.partitions().size());
        for (std::size_t rank = 0; rank < graph.partitions().size(); ++rank) {
            devices.push_back(graph.partitions()[rank].device_id);
        }
    }
    std::sort(devices.begin(), devices.end());
    if (std::adjacent_find(devices.begin(), devices.end()) != devices.end()) {
        throw std::invalid_argument("partitioned BFS devices must be distinct");
    }
    if (devices.size() != graph.partitions().size()) {
        throw std::invalid_argument(
            "partitioned BFS requires one device for every partition");
    }
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        if (!std::binary_search(
                status.device_ids.begin(), status.device_ids.end(),
                devices[rank])) {
            throw std::out_of_range(
                "partitioned BFS device is outside the active runtime");
        }
    }
    std::vector<int> manifest_devices;
    manifest_devices.reserve(graph.partitions().size());
    for (std::size_t rank = 0; rank < graph.partitions().size(); ++rank) {
        manifest_devices.push_back(graph.partitions()[rank].device_id);
    }
    std::sort(manifest_devices.begin(), manifest_devices.end());
    if (manifest_devices != devices) {
        throw std::invalid_argument(
            "partitioned BFS requested devices must exactly match the "
            "manifest device set");
    }
    return devices;
}

std::vector<BfsHostPartition> host_partitions(
    const MultiPartitionGraph& graph) {
    if (graph.view_orientation() != "outgoing" &&
        graph.view_orientation() != "undirected") {
        throw std::invalid_argument(
            "partitioned BFS requires an outgoing or undirected CSR view");
    }
    std::vector<BfsHostPartition> result(graph.partitions().size());
    for (std::size_t rank = 0; rank < graph.partitions().size(); ++rank) {
        const PartitionCSR& source = graph.partitions()[rank];
        BfsHostPartition& target = result[rank];
        target.partition_id = source.partition_id;
        target.device_id = source.device_id;
        target.owned_vertex_count = static_cast<std::size_t>(
            source.owned_vertex_count);
        target.offsets.assign(source.offsets.begin(), source.offsets.end());
        target.indices.assign(source.indices.begin(), source.indices.end());
        target.local_to_global.assign(
            source.local_to_global.begin(), source.local_to_global.end());
        target.owners.assign(source.owners.begin(), source.owners.end());
    }
    return result;
}

BfsPartitionUpload make_upload(
    const MultiPartitionGraph& graph,
    const PartitionedBfsPlan& plan,
    const std::vector<int>& devices,
    std::size_t rank) {
    const BfsHostPartition& partition = plan.partitions[rank];
    const std::size_t rank_count = plan.partitions.size();
    BfsPartitionUpload upload;
    upload.partition_id = partition.partition_id;
    upload.device_id = partition.device_id;
    upload.graph_identity = gpu_easygraph::runtime::stable_signature64(
        graph.graph_id());
    upload.graph_identity_text = graph.graph_id();
    upload.generation = graph.generation();
    upload.device_group = devices;
    upload.view_signature = graph.view_signature() +
        ":bfs-owner-frontier-v2";
    upload.owned_vertex_count = partition.owned_vertex_count;
    upload.offsets = partition.offsets;
    upload.indices = partition.indices;
    upload.owners = partition.owners;
    const PartitionCSR& manifest_partition = graph.partitions().at(rank);
    upload.offsets_sha256 = manifest_partition.offsets_sha256;
    upload.indices_sha256 = manifest_partition.indices_sha256;
    upload.owners_sha256 = manifest_partition.owners_sha256;
    upload.topology_compatibility =
        easygraph::egmgpu::partition_topology_compatibility(
            graph, partition.partition_id, devices);
    upload.ownership_compatibility =
        easygraph::egmgpu::partition_ownership_compatibility(
            graph, partition.partition_id, devices);
    upload.halo_route_slots.assign(partition.local_to_global.size(), -1);
    upload.outgoing_route_offsets.push_back(0U);
    for (std::size_t destination = 0; destination < rank_count;
         ++destination) {
        const BfsOwnerRoute& route = plan.route(rank, destination);
        for (std::size_t index = 0; index < route.capacity(); ++index) {
            const int halo_local = route.source_halo_local_ids[index];
            if (halo_local < 0 ||
                static_cast<std::size_t>(halo_local) >=
                    upload.halo_route_slots.size() ||
                upload.halo_route_slots[static_cast<std::size_t>(halo_local)] !=
                    -1) {
                throw std::logic_error(
                    "partitioned BFS route contains an invalid halo local ID");
            }
            const int slot = static_cast<int>(
                upload.route_owner_local_ids.size());
            upload.halo_route_slots[static_cast<std::size_t>(halo_local)] =
                slot;
            upload.route_destination_ranks.push_back(
                static_cast<int>(destination));
            upload.route_owner_local_ids.push_back(
                route.destination_owner_local_ids[index]);
        }
        upload.outgoing_route_offsets.push_back(
            upload.route_owner_local_ids.size());
    }
    for (std::size_t local = partition.owned_vertex_count;
         local < partition.local_to_global.size(); ++local) {
        if (upload.halo_route_slots[local] < 0) {
            throw std::logic_error(
                "partitioned BFS halo is absent from its owner route");
        }
    }
    upload.incoming_route_offsets.push_back(0U);
    for (std::size_t source = 0; source < rank_count; ++source) {
        upload.incoming_route_offsets.push_back(
            upload.incoming_route_offsets.back() +
            plan.route(source, rank).capacity());
    }
    const BfsDenseRouteLayout dense_layout =
        gpu_easygraph::runtime::build_bfs_dense_route_layout(plan, rank);
    upload.outgoing_dense_word_offsets =
        dense_layout.outgoing_word_offsets;
    upload.incoming_dense_word_offsets =
        dense_layout.incoming_word_offsets;
    return upload;
}

struct RankTiming {
    MeasuredSeconds kernel;
    MeasuredSeconds sparse_kernel;
    MeasuredSeconds dense_kernel;
    MeasuredSeconds pack;
    MeasuredSeconds transfer;
    MeasuredSeconds wait;
    MeasuredSeconds communication;
    MeasuredSeconds output_gather;
    MeasuredSeconds sparse_round;
    MeasuredSeconds dense_round;
    MeasuredSeconds frontier_switch;
    MeasuredSeconds sparse_communication;
    MeasuredSeconds dense_communication;
};

struct RankState {
    std::shared_ptr<ResidentBfsPartitionView> view;
    std::shared_ptr<BfsPartitionWorkspace> workspace;
    std::vector<int> send_counts;
    std::vector<std::int64_t> owned_distances;
    RankTiming timing;
    std::size_t current_count;
    int next_count;
    std::uint64_t logical_messages;
    std::uint64_t control_transfer_bytes;
    std::uint64_t sparse_logical_bytes;
    std::uint64_t sparse_payload_transfer_bytes;
    std::uint64_t dense_logical_bytes;
    std::uint64_t dense_payload_transfer_bytes;
    std::uint64_t sparse_rounds;
    std::uint64_t dense_rounds;
    std::uint64_t frontier_switches;
    BfsFrontierKind last_kind;
    bool have_last_kind;
    std::size_t resident_bytes;
    double state_upload_seconds;
    bool device_view_cache_hit;
    bool warp_edges = false;
    int sparse_ctas_per_row = 0;
    std::size_t high_degree_workspace_bytes = 0;
    std::uint64_t reached_count = 0;
    double host_assembly_seconds = 0.0;
    owner_output::Observation core_output_observation;
    gpu_easygraph::runtime::GpuWorkerCpuPlacementInfo cpu_placement;
    easygraph::egmgpu::BfsThreadUsageWindow worker_usage, before_output_usage;
    easygraph::egmgpu::BfsThreadUsageWindow output_gather_usage, host_assembly_usage;
    bool direct_routes = false;
    bool move_output = false;
    bool trim_barriers = false;
    bool single_partition_ordering = false;
    SharedArrayCallStats shared_arrays;

    RankState()
        : current_count(0),
          next_count(0),
          logical_messages(0),
          control_transfer_bytes(0),
          sparse_logical_bytes(0),
          sparse_payload_transfer_bytes(0),
          dense_logical_bytes(0),
          dense_payload_transfer_bytes(0),
          sparse_rounds(0),
          dense_rounds(0),
          frontier_switches(0),
          last_kind(BfsFrontierKind::Sparse),
          have_last_kind(false),
          resident_bytes(0),
          state_upload_seconds(0.0),
          device_view_cache_hit(false) {}
};

struct BfsPinnedHostState {
    std::mutex lease_mutex;
    std::vector<std::shared_ptr<PinnedHostBuffer> > host_payloads;
    std::vector<std::shared_ptr<PinnedHostBuffer> > host_counts;
    std::vector<std::shared_ptr<PinnedHostBuffer> > output_buffers;
    std::vector<gpu_easygraph::runtime::GpuLocalPinnedAllocationInfo> output_placement;
    std::size_t allocation_count = 0;
    std::size_t retained_bytes = 0;

    void grow(std::shared_ptr<PinnedHostBuffer>& buffer, std::size_t bytes) {
        if (bytes == 0 || (buffer && buffer->size() >= bytes)) return;
        const std::size_t old_bytes = buffer ? buffer->size() : 0U;
        std::shared_ptr<PinnedHostBuffer> replacement(new PinnedHostBuffer(bytes));
        buffer = std::move(replacement);
        retained_bytes += bytes - old_bytes;
        ++allocation_count;
    }

    void ensure(const PartitionedBfsPlan& plan, bool use_host_staging,
                bool numa_local_output = false, int colocated_device = -1) {
        const std::size_t count = plan.partitions.size();
        host_counts.resize(count);
        output_buffers.resize(count);
        output_placement.resize(count);
        host_payloads.resize(count * count);
        for (std::size_t rank = 0; rank < count; ++rank) {
            grow(host_counts[rank], (count + 1U) * sizeof(int));
            const std::size_t bytes = plan.partitions[rank].owned_vertex_count * sizeof(std::int64_t);
            if (numa_local_output && bytes && (!output_buffers[rank] || output_buffers[rank]->size() < bytes)) {
                const std::size_t previous = output_buffers[rank] ? output_buffers[rank]->size() : 0;
                output_buffers[rank] = gpu_easygraph::runtime::make_gpu_local_pinned_host_buffer(
                    bytes, colocated_device >= 0 ? colocated_device : plan.partitions[rank].device_id,
                    &output_placement[rank]);
                retained_bytes += bytes - previous;
                ++allocation_count;
            } else if (!numa_local_output) {
                const bool allocation_needed = bytes && (!output_buffers[rank] || output_buffers[rank]->size() < bytes);
                grow(output_buffers[rank], bytes);
                if (allocation_needed) output_placement[rank].allocation_cpu = sched_getcpu();
            }
        }
        if (use_host_staging) {
            for (std::size_t source = 0; source < count; ++source) {
                const BfsDenseRouteLayout dense_layout =
                    gpu_easygraph::runtime::build_bfs_dense_route_layout(plan, source);
                for (std::size_t destination = 0; destination < count; ++destination) {
                    const std::size_t capacity = plan.route(source, destination).capacity();
                    const std::size_t dense_words =
                        dense_layout.outgoing_word_offsets[destination + 1U] -
                        dense_layout.outgoing_word_offsets[destination];
                    grow(host_payloads[source * count + destination], std::max(
                        capacity * sizeof(int), dense_words * sizeof(std::uint32_t)));
                }
            }
        }
    }
};

struct SharedBfsExecution {
    struct FrontierDecision {
        std::uint64_t iteration;
        BfsFrontierKind kind;
        BfsFrontierMetrics metrics;
        std::size_t active_vertices;
        double estimated_active_edges;
    };

    SharedBfsExecution(
        const PartitionedBfsPlan& plan,
        const std::shared_ptr<BfsPinnedHostState>& pinned,
        BfsFrontierPolicy requested_policy,
        const BfsFrontierThresholds& calibrated_thresholds)
        : barrier(plan.partitions.size()),
          ranks(plan.partitions.size()),
          global_active(0),
          iterations(0),
          converged(false),
          policy(requested_policy),
          thresholds(calibrated_thresholds),
          selected_kind(BfsFrontierKind::Sparse),
          previous_kind(BfsFrontierKind::Sparse),
          phases(plan.partitions.size(), "not_started"),
          host_payloads(pinned->host_payloads),
          host_counts(pinned->host_counts),
          output_buffers(pinned->output_buffers) {}

    CooperativeBarrier barrier;
    const bool control_fusion_requested = gpu_easygraph::runtime::algorithm_candidate_flag(
        "EGGPU_ALGORITHM_CONTROL_FUSION");
    bool control_fusion_used = false;
    std::uint64_t control_reduction_calls = 0;
    std::vector<RankState> ranks;
    std::size_t global_active;
    std::uint64_t iterations;
    bool converged;
    BfsFrontierPolicy policy;
    BfsFrontierThresholds thresholds;
    BfsFrontierKind selected_kind;
    BfsFrontierKind previous_kind;
    std::vector<FrontierDecision> decisions;
    std::vector<std::string> phases;
    std::vector<std::shared_ptr<PinnedHostBuffer> > host_payloads;
    std::vector<std::shared_ptr<PinnedHostBuffer> > host_counts;
    std::vector<std::shared_ptr<PinnedHostBuffer> > output_buffers;
    BfsHostAssembly host_assembly = BfsHostAssembly::Legacy;
    bool core_output_team16 = false;
    int core_output_width = 1;
    std::size_t core_output_workers = 1U;
    std::shared_ptr<const owner_output::CpuPlacement> core_output_placement;
    bool host_usage_profile = false;
    bool worker_gpu_numa = false;
    gpu_easygraph::runtime::GpuWorkerCpuPlacementInfo serial_cpu_placement;
    easygraph::egmgpu::BfsThreadUsageWindow serial_worker_usage, serial_before_output_usage, serial_output_usage;
    std::unique_ptr<std::int64_t[]> assembled_distances;
    double* public_distances = nullptr; // Borrowed from caller-owned request storage.
    easygraph::egmgpu::BfsPublicObservation public_output_observation;
};

void assemble_bfs_rank_output(const PartitionedBfsPlan& plan,
    const std::shared_ptr<SharedBfsExecution>& shared, std::size_t rank) {
    RankState& state = shared->ranks[rank];
    if (shared->host_usage_profile) state.host_assembly_usage.start();
    const auto begin = SteadyClock::now();
    const auto& partition = plan.partitions[rank];
    const std::int64_t* observed = static_cast<const std::int64_t*>(
        shared->output_buffers[rank] ? shared->output_buffers[rank]->data() : nullptr);
    if (shared->public_distances) {
        auto write = [&](std::size_t first, std::size_t last) {
            if (first == last) return;
            shared->public_output_observation.add(easygraph::egmgpu::assemble_bfs_public_host(
                partition.local_to_global.data()+first, last-first, observed+first,
                shared->public_distances, plan.vertex_count));
        };
        if (shared->core_output_team16)
            owner_output::run(partition.owned_vertex_count, shared->core_output_width,
                state.core_output_observation, write,
                shared->core_output_placement->for_worker(shared->core_output_workers == 1U ? 0U : rank));
        else {
            owner_output::observe_original(state.core_output_observation);
            write(0U, partition.owned_vertex_count);
        }
    } else if (shared->core_output_team16) {
        // The plan gives every global ID one owner. Each team member receives
        // disjoint owner-local IDs; the join precedes worker completion and
        // caller-held pinned-buffer lease release. All-reached is an OWNER
        // property, not the original count reinterpreted on a shorter slice.
        const bool all_reached = state.reached_count == partition.owned_vertex_count;
        owner_output::run(partition.owned_vertex_count, shared->core_output_width,
            state.core_output_observation, [&](std::size_t first, std::size_t last) {
                if (first == last) return;
                easygraph::egmgpu::assemble_bfs_owned_host(
                    partition.local_to_global.data() + first, last-first,
                    observed + first, shared->assembled_distances.get(),
                    all_reached ? last-first : 0U);
            }, shared->core_output_placement->for_worker(shared->core_output_workers == 1U ? 0U : rank));
    } else {
        owner_output::observe_original(state.core_output_observation);
        easygraph::egmgpu::assemble_bfs_owned_host(partition.local_to_global.data(),
            partition.owned_vertex_count, observed, shared->assembled_distances.get(), state.reached_count);
    }
    state.host_assembly_seconds = elapsed_seconds(begin, SteadyClock::now());
    if (shared->host_usage_profile) state.host_assembly_usage.finish();
}

void timed_barrier(
    const std::shared_ptr<SharedBfsExecution>& shared,
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
    const std::shared_ptr<SharedBfsExecution>& shared,
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

void clear_and_expand_sparse(
    const BfsPartitionDeviceView& view,
    BfsPartitionWorkspace& workspace,
    RankState* state,
    std::int64_t next_distance,
    WorkerStreams& streams) {
    CudaEventTimer timer("measure BFS sparse expansion");
    timer.start(streams.compute());
    if (workspace.next_count()) {
        check_cuda(
            cudaMemsetAsync(
                workspace.next_count(), 0, sizeof(int), streams.compute()),
            "clear BFS next frontier count");
    }
    const std::size_t route_capacity = view.outgoing_route_offsets.back();
    if (route_capacity != 0 && !state->direct_routes) {
        check_cuda(
            cudaMemsetAsync(
                workspace.send_flags(), 0,
                route_capacity * sizeof(int), streams.compute()),
            "clear BFS sparse route flags");
    }
    check_cuda(
        cudaMemsetAsync(
            workspace.send_counts(), 0,
            state->send_counts.size() * sizeof(int), streams.compute()),
        "clear BFS sparse route counts");
    if (state->current_count != 0) {
        gpu_easygraph::partitioned::expand_bfs_sparse(
            view.offsets,
            view.indices,
            view.owners,
            view.halo_route_slots,
            workspace.distances(),
            workspace.current_frontier(),
            state->current_count,
            workspace.next_frontier(),
            workspace.next_count(),
            workspace.send_flags(),
            view.owned_vertex_count,
            view.partition_id,
            next_distance,
            streams.compute(),
            state->warp_edges,
            state->direct_routes ? view.route_destination_ranks : nullptr,
            state->direct_routes ? view.route_owner_local_ids : nullptr,
            state->direct_routes ? view.outgoing_route_offsets_device : nullptr,
            state->direct_routes ? workspace.send_values(0) : nullptr,
            state->direct_routes ? workspace.send_counts() : nullptr,
            state->sparse_ctas_per_row,
            workspace.high_degree_rows(), workspace.high_degree_control(),
            workspace.high_degree_grid_limit());
    }
    const MeasuredSeconds seconds = timer.finish(streams.compute(), !state->single_partition_ordering);
    state->timing.kernel.add(seconds);
    state->timing.sparse_kernel.add(seconds);
}

void convert_sparse_frontier_to_bitmap(
    const BfsPartitionDeviceView& view,
    BfsPartitionWorkspace& workspace,
    RankState* state,
    WorkerStreams& streams) {
    if (view.owned_vertex_count == 0) return;
    CudaEventTimer timer("measure BFS sparse-to-dense frontier conversion");
    timer.start(streams.compute());
    gpu_easygraph::partitioned::build_bfs_frontier_bitmap(
        workspace.current_frontier(),
        state->current_count,
        workspace.current_bitmap(),
        view.owned_vertex_count,
        streams.compute());
    state->timing.frontier_switch.add(timer.finish(streams.compute(), !state->single_partition_ordering));
}

void clear_and_expand_dense(
    const BfsPartitionDeviceView& view,
    BfsPartitionWorkspace& workspace,
    RankState* state,
    std::int64_t next_distance,
    WorkerStreams& streams) {
    CudaEventTimer timer("measure BFS dense bitmap expansion");
    timer.start(streams.compute());
    if (workspace.next_count()) {
        check_cuda(
            cudaMemsetAsync(
                workspace.next_count(), 0, sizeof(int), streams.compute()),
            "clear BFS dense next frontier count");
    }
    const std::size_t owned_words =
        gpu_easygraph::runtime::bfs_bitmap_word_count(
            view.owned_vertex_count);
    if (owned_words != 0) {
        check_cuda(
            cudaMemsetAsync(
                workspace.next_bitmap(), 0,
                owned_words * sizeof(std::uint32_t), streams.compute()),
            "clear BFS dense next frontier bitmap");
    }
    const std::size_t outgoing_words =
        view.outgoing_dense_word_offsets.back();
    if (outgoing_words != 0) {
        check_cuda(
            cudaMemsetAsync(
                workspace.dense_send_bitmap(0), 0,
                outgoing_words * sizeof(std::uint32_t), streams.compute()),
            "clear BFS dense outgoing owner bitmaps");
    }
    if (state->current_count != 0 && view.owned_vertex_count != 0) {
        gpu_easygraph::partitioned::expand_bfs_dense(
            view.offsets,
            view.indices,
            view.owners,
            view.halo_route_slots,
            view.route_destination_ranks,
            view.route_owner_local_ids,
            view.outgoing_dense_word_offsets_device,
            workspace.distances(),
            workspace.current_bitmap(),
            workspace.next_bitmap(),
            workspace.next_frontier(),
            workspace.next_count(),
            outgoing_words == 0 ? nullptr : workspace.dense_send_bitmap(0),
            view.owned_vertex_count,
            view.partition_id,
            next_distance,
            streams.compute(),
            state->warp_edges,
            state->direct_routes ? workspace.send_flags() : nullptr);
    }
    const MeasuredSeconds seconds = timer.finish(streams.compute(), !state->single_partition_ordering);
    state->timing.kernel.add(seconds);
    state->timing.dense_kernel.add(seconds);
}

void clear_and_expand_pull(
    const BfsPartitionDeviceView& view, BfsPartitionWorkspace& workspace,
    RankState* state, std::int64_t next_distance, WorkerStreams& streams,
    int pull_lanes) {
    CudaEventTimer timer("measure BFS dense pull expansion");
    timer.start(streams.compute());
    if (workspace.next_count()) check_cuda(cudaMemsetAsync(
        workspace.next_count(), 0, sizeof(int), streams.compute()), "clear BFS pull count");
    const std::size_t words = gpu_easygraph::runtime::bfs_bitmap_word_count(view.owned_vertex_count);
    if (words) check_cuda(cudaMemsetAsync(workspace.next_bitmap(), 0,
        words * sizeof(std::uint32_t), streams.compute()), "clear BFS pull next bitmap");
    gpu_easygraph::partitioned::expand_bfs_pull(
        view.offsets, view.indices, view.owners, view.halo_route_slots,
        view.route_destination_ranks, view.route_owner_local_ids,
        view.outgoing_dense_word_offsets_device, workspace.distances(),
        workspace.current_bitmap(), workspace.dense_send_bitmap(0),
        workspace.next_bitmap(), workspace.next_frontier(), workspace.next_count(),
        view.owned_vertex_count, view.partition_id, next_distance,
        streams.compute(), pull_lanes);
    const MeasuredSeconds seconds = timer.finish(streams.compute(), !state->single_partition_ordering);
    state->timing.kernel.add(seconds);
    state->timing.dense_kernel.add(seconds);
}

void stage_pull_frontier(
    const std::shared_ptr<SharedBfsExecution>& shared,
    const PartitionedBfsPlan& plan, std::size_t rank,
    WorkerStreams& streams, RankTiming* timing) {
    const std::size_t words = gpu_easygraph::runtime::bfs_bitmap_word_count(
        plan.partitions[rank].owned_vertex_count);
    if (!words || plan.partitions.size() == 1U) return;
    CudaEventTimer timer("measure BFS pull frontier D2H"); timer.start(streams.transfer());
    for (std::size_t destination = 0; destination < plan.partitions.size(); ++destination) {
        if (destination == rank) continue;
        // This existing buffer was sized for a bitmap in rank's owner domain.
        check_cuda(cudaMemcpyAsync(shared->host_payloads[destination * plan.partitions.size() + rank]->data(),
            shared->ranks[rank].workspace->current_bitmap(), words * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost, streams.transfer()), "stage immutable BFS pull frontier");
    }
    timing->transfer.add(timer.finish(streams.transfer()));
}

void acquire_pull_frontiers(
    const std::shared_ptr<SharedBfsExecution>& shared,
    const PartitionedBfsPlan& plan, std::size_t rank, int worker_device,
    CommunicationBackend backend, WorkerStreams& streams, RankTiming* timing) {
    BfsPartitionWorkspace& workspace = *shared->ranks[rank].workspace;
    bool copied = false; CudaEventTimer timer("measure BFS pull frontier replica transfer");
    for (std::size_t source = 0; source < plan.partitions.size(); ++source) {
        const std::size_t words = workspace.dense_send_word_count(source);
        if (source == rank || !words) continue;
        if (!copied) { timer.start(streams.transfer()); copied = true; }
        if (backend == CommunicationBackend::PeerToPeer) {
            gpu_easygraph::runtime::peer_copy_async(workspace.dense_send_bitmap(source), worker_device,
                shared->ranks[source].workspace->current_bitmap(), plan.partitions[source].device_id,
                words * sizeof(std::uint32_t), streams.transfer());
        } else if (backend == CommunicationBackend::HostStaging) {
            check_cuda(cudaMemcpyAsync(workspace.dense_send_bitmap(source),
                shared->host_payloads[rank * plan.partitions.size() + source]->data(),
                words * sizeof(std::uint32_t), cudaMemcpyHostToDevice, streams.transfer()),
                "receive immutable BFS pull frontier");
        }
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer()));
}

void compact_routes(
    const BfsPartitionDeviceView& view,
    BfsPartitionWorkspace& workspace,
    RankTiming* timing,
    WorkerStreams& streams) {
    bool launched = false;
    CudaEventTimer timer("measure BFS sparse route compaction");
    for (std::size_t destination = 0;
         destination + 1U < view.outgoing_route_offsets.size();
         ++destination) {
        const std::size_t begin = view.outgoing_route_offsets[destination];
        const std::size_t capacity =
            view.outgoing_route_offsets[destination + 1U] - begin;
        if (capacity == 0) continue;
        if (!launched) {
            timer.start(streams.pack());
            launched = true;
        }
        gpu_easygraph::partitioned::compact_bfs_route(
            workspace.send_flags() + begin,
            view.route_owner_local_ids + begin,
            workspace.send_values(destination),
            capacity,
            workspace.send_counts() + destination,
            streams.pack());
    }
    if (launched) timing->pack.add(timer.finish(streams.pack()));
}

void copy_send_counts_to_host(
    const std::shared_ptr<SharedBfsExecution>& shared,
    std::size_t rank,
    BfsPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankState* state) {
    // P1 has no remote owner. The host counts remain their initialized zeros.
    if (state->single_partition_ordering) return;
    const std::size_t count = state->send_counts.size();
    CudaEventTimer timer("measure BFS route count transfer");
    timer.start(streams.transfer());
    check_cuda(
        cudaMemcpyAsync(
            shared->host_counts[rank]->data(),
            workspace.send_counts(),
            count * sizeof(int),
            cudaMemcpyDeviceToHost,
            streams.transfer()),
        "copy BFS route counts to host");
    state->timing.transfer.add(timer.finish(streams.transfer()));
    state->control_transfer_bytes +=
        static_cast<std::uint64_t>(count * sizeof(int));
    const int* observed = static_cast<const int*>(
        shared->host_counts[rank]->data());
    for (std::size_t peer = 0; peer < count; ++peer) {
        const std::size_t capacity =
            state->view->view().outgoing_route_offsets[peer + 1U] -
            state->view->view().outgoing_route_offsets[peer];
        if (observed[peer] < 0 ||
            static_cast<std::size_t>(observed[peer]) > capacity) {
            throw std::runtime_error(
                "partitioned BFS produced an invalid compact route count");
        }
        state->send_counts[peer] = observed[peer];
        state->logical_messages += static_cast<std::uint64_t>(observed[peer]);
    }
}

void stage_outgoing_payloads(
    const std::shared_ptr<SharedBfsExecution>& shared,
    const PartitionedBfsPlan& plan,
    std::size_t rank,
    BfsPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankTiming* timing) {
    bool copied = false;
    CudaEventTimer timer("measure BFS D2H staging");
    for (std::size_t destination = 0;
         destination < plan.partitions.size(); ++destination) {
        const int count = shared->ranks[rank].send_counts[destination];
        if (count == 0) continue;
        if (!copied) {
            timer.start(streams.transfer());
            copied = true;
        }
        check_cuda(
            cudaMemcpyAsync(
                shared->host_payloads[
                    rank * plan.partitions.size() + destination]->data(),
                workspace.send_values(destination),
                static_cast<std::size_t>(count) * sizeof(int),
                cudaMemcpyDeviceToHost,
                streams.transfer()),
            "stage BFS frontier to host");
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer()));
}

void peer_transfer_payloads(
    const std::shared_ptr<SharedBfsExecution>& shared,
    const PartitionedBfsPlan& plan,
    std::size_t rank,
    int worker_device,
    WorkerStreams& streams,
    RankTiming* timing) {
    bool copied = false;
    CudaEventTimer timer("measure BFS peer transfer");
    for (std::size_t source = 0; source < plan.partitions.size(); ++source) {
        const int count = shared->ranks[source].send_counts[rank];
        if (count == 0) continue;
        if (!copied) {
            timer.start(streams.transfer());
            copied = true;
        }
        gpu_easygraph::runtime::peer_copy_async(
            shared->ranks[rank].workspace->receive_values(source),
            worker_device,
            shared->ranks[source].workspace->send_values(rank),
            plan.partitions[source].device_id,
            static_cast<std::size_t>(count) * sizeof(int),
            streams.transfer());
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer()));
}

void stage_incoming_payloads(
    const std::shared_ptr<SharedBfsExecution>& shared,
    const PartitionedBfsPlan& plan,
    std::size_t rank,
    BfsPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankTiming* timing) {
    bool copied = false;
    CudaEventTimer timer("measure BFS H2D staging");
    for (std::size_t source = 0; source < plan.partitions.size(); ++source) {
        const int count = shared->ranks[source].send_counts[rank];
        if (count == 0) continue;
        if (!copied) {
            timer.start(streams.transfer());
            copied = true;
        }
        check_cuda(
            cudaMemcpyAsync(
                workspace.receive_values(source),
                shared->host_payloads[
                    source * plan.partitions.size() + rank]->data(),
                static_cast<std::size_t>(count) * sizeof(int),
                cudaMemcpyHostToDevice,
                streams.transfer()),
            "stage BFS frontier to owner device");
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer()));
}

void apply_owner_candidates(
    const std::shared_ptr<SharedBfsExecution>& shared,
    const PartitionedBfsPlan& plan,
    std::size_t rank,
    const BfsPartitionDeviceView& view,
    BfsPartitionWorkspace& workspace,
    std::int64_t next_distance,
    WorkerStreams& streams,
    RankTiming* timing) {
    bool launched = false;
    CudaEventTimer timer("measure BFS owner combine");
    for (std::size_t source = 0; source < plan.partitions.size(); ++source) {
        const int count = shared->ranks[source].send_counts[rank];
        if (count == 0) continue;
        if (!launched) {
            timer.start(streams.compute());
            launched = true;
        }
        gpu_easygraph::partitioned::apply_bfs_owner_candidates(
            workspace.receive_values(source),
            static_cast<std::size_t>(count),
            workspace.distances(),
            workspace.next_frontier(),
            workspace.next_count(),
            view.owned_vertex_count,
            next_distance,
            streams.compute());
    }
    if (launched) timing->kernel.add(timer.finish(streams.compute()));
}

void account_dense_payload(
    const PartitionedBfsPlan& plan,
    CommunicationBackend backend,
    RankState* state) {
    if (state->current_count == 0) return;
    std::uint64_t bytes = 0;
    for (std::size_t destination = 0;
         destination < plan.partitions.size(); ++destination) {
        bytes += static_cast<std::uint64_t>(
            state->workspace->dense_send_word_count(destination) *
            sizeof(std::uint32_t));
    }
    state->dense_logical_bytes += bytes;
    state->dense_payload_transfer_bytes +=
        backend == CommunicationBackend::HostStaging ? bytes * 2U : bytes;
}

void stage_dense_outgoing_bitmaps(
    const std::shared_ptr<SharedBfsExecution>& shared,
    const PartitionedBfsPlan& plan,
    std::size_t rank,
    BfsPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankTiming* timing) {
    if (shared->ranks[rank].current_count == 0) return;
    bool copied = false;
    CudaEventTimer timer("measure BFS dense bitmap D2H staging");
    for (std::size_t destination = 0;
         destination < plan.partitions.size(); ++destination) {
        const std::size_t words =
            workspace.dense_send_word_count(destination);
        if (words == 0) continue;
        if (!copied) {
            timer.start(streams.transfer());
            copied = true;
        }
        check_cuda(
            cudaMemcpyAsync(
                shared->host_payloads[
                    rank * plan.partitions.size() + destination]->data(),
                workspace.dense_send_bitmap(destination),
                words * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost,
                streams.transfer()),
            "stage BFS dense owner bitmap to host");
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer()));
}

void peer_transfer_dense_bitmaps(
    const std::shared_ptr<SharedBfsExecution>& shared,
    const PartitionedBfsPlan& plan,
    std::size_t rank,
    int worker_device,
    WorkerStreams& streams,
    RankTiming* timing) {
    bool copied = false;
    CudaEventTimer timer("measure BFS dense bitmap peer transfer");
    for (std::size_t source = 0; source < plan.partitions.size(); ++source) {
        if (shared->ranks[source].current_count == 0) continue;
        const std::size_t words =
            shared->ranks[source].workspace->dense_send_word_count(rank);
        if (words == 0) continue;
        if (shared->ranks[rank].workspace->dense_receive_word_count(source) !=
            words) {
            throw std::logic_error(
                "BFS dense sender and receiver bitmap sizes disagree");
        }
        if (!copied) {
            timer.start(streams.transfer());
            copied = true;
        }
        gpu_easygraph::runtime::peer_copy_async(
            shared->ranks[rank].workspace->dense_receive_bitmap(source),
            worker_device,
            shared->ranks[source].workspace->dense_send_bitmap(rank),
            plan.partitions[source].device_id,
            words * sizeof(std::uint32_t),
            streams.transfer());
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer()));
}

void stage_dense_incoming_bitmaps(
    const std::shared_ptr<SharedBfsExecution>& shared,
    const PartitionedBfsPlan& plan,
    std::size_t rank,
    BfsPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankTiming* timing) {
    bool copied = false;
    CudaEventTimer timer("measure BFS dense bitmap H2D staging");
    for (std::size_t source = 0; source < plan.partitions.size(); ++source) {
        if (shared->ranks[source].current_count == 0) continue;
        const std::size_t words =
            shared->ranks[source].workspace->dense_send_word_count(rank);
        if (words == 0) continue;
        if (workspace.dense_receive_word_count(source) != words) {
            throw std::logic_error(
                "BFS dense staged bitmap sizes disagree");
        }
        if (!copied) {
            timer.start(streams.transfer());
            copied = true;
        }
        check_cuda(
            cudaMemcpyAsync(
                workspace.dense_receive_bitmap(source),
                shared->host_payloads[
                    source * plan.partitions.size() + rank]->data(),
                words * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice,
                streams.transfer()),
            "stage BFS dense owner bitmap to device");
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer()));
}

void apply_dense_owner_bitmaps(
    const std::shared_ptr<SharedBfsExecution>& shared,
    const PartitionedBfsPlan& plan,
    std::size_t rank,
    const BfsPartitionDeviceView& view,
    BfsPartitionWorkspace& workspace,
    std::int64_t next_distance,
    WorkerStreams& streams,
    RankTiming* timing) {
    bool launched = false;
    CudaEventTimer timer("measure BFS dense owner bitmap combine");
    for (std::size_t source = 0; source < plan.partitions.size(); ++source) {
        if (shared->ranks[source].current_count == 0) continue;
        const std::size_t words = workspace.dense_receive_word_count(source);
        if (words == 0) continue;
        if (!launched) {
            timer.start(streams.compute());
            launched = true;
        }
        gpu_easygraph::partitioned::apply_bfs_owner_bitmap(
            workspace.dense_receive_bitmap(source),
            words,
            workspace.distances(),
            workspace.next_bitmap(),
            workspace.next_frontier(),
            workspace.next_count(),
            view.owned_vertex_count,
            next_distance,
            streams.compute());
    }
    if (launched) {
        const MeasuredSeconds seconds = timer.finish(streams.compute());
        timing->kernel.add(seconds);
        timing->dense_kernel.add(seconds);
    }
}

BfsFrontierMetrics frontier_metrics(
    const PartitionedBfsPlan& plan,
    const std::shared_ptr<SharedBfsExecution>& shared,
    std::size_t* active_vertices,
    double* estimated_active_edges) {
    std::size_t active = 0;
    double estimated_edges = 0.0;
    std::uint64_t adjacency = 0;
    for (std::size_t rank = 0; rank < plan.partitions.size(); ++rank) {
        const BfsHostPartition& partition = plan.partitions[rank];
        active += shared->ranks[rank].current_count;
        adjacency += static_cast<std::uint64_t>(partition.indices.size());
        if (partition.owned_vertex_count != 0) {
            estimated_edges +=
                static_cast<double>(shared->ranks[rank].current_count) *
                static_cast<double>(partition.indices.size()) /
                static_cast<double>(partition.owned_vertex_count);
        }
    }
    BfsFrontierMetrics metrics;
    metrics.active_vertex_fraction = plan.vertex_count == 0
        ? 0.0
        : std::min(
            1.0,
            static_cast<double>(active) /
                static_cast<double>(plan.vertex_count));
    metrics.estimated_active_edge_fraction = adjacency == 0
        ? 0.0
        : std::min(1.0, estimated_edges / static_cast<double>(adjacency));
    metrics.boundary_edge_fraction = adjacency == 0
        ? 0.0
        : std::min(
            1.0,
            static_cast<double>(plan.boundary_adjacency_count) /
                static_cast<double>(adjacency));
    *active_vertices = active;
    *estimated_active_edges = estimated_edges;
    return metrics;
}

void copy_next_count_to_host(
    const std::shared_ptr<SharedBfsExecution>& shared,
    std::size_t rank,
    BfsPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankState* state) {
    if (!workspace.next_count()) {
        state->next_count = 0;
        return;
    }
    const std::size_t offset = state->send_counts.size();
    int* destination = static_cast<int*>(shared->host_counts[rank]->data()) +
        offset;
    // This is the P1 publication boundary: expansion and D2H share one stream.
    const cudaStream_t count_stream = state->single_partition_ordering
        ? streams.compute() : streams.transfer();
    CudaEventTimer timer("measure BFS next-count transfer");
    timer.start(count_stream);
    check_cuda(
        cudaMemcpyAsync(
            destination,
            workspace.next_count(),
            sizeof(int),
            cudaMemcpyDeviceToHost,
            count_stream),
        "copy BFS next frontier count to host");
    state->timing.transfer.add(timer.finish(count_stream));
    state->control_transfer_bytes += sizeof(int);
    if (*destination < 0 ||
        static_cast<std::size_t>(*destination) >
            state->view->view().owned_vertex_count) {
        throw std::runtime_error(
            "partitioned BFS produced an invalid next frontier count");
    }
    state->next_count = *destination;
}

struct NativeBfsResult {
    std::vector<std::int64_t> distances;
    std::unique_ptr<std::int64_t[]> direct_distances;
    std::size_t direct_distance_count = 0;
    bool direct_assembly = false;
    double merge_seconds;
    std::uint64_t reachable = 0;
    std::uint64_t resident_bytes;
    double state_upload_seconds;
    bool device_view_cache_hit;
    MeasuredSeconds kernel;
    MeasuredSeconds pack;
    MeasuredSeconds transfer;
    MeasuredSeconds wait;
    MeasuredSeconds communication;
    MeasuredSeconds output_gather;
    MeasuredSeconds sparse_kernel;
    MeasuredSeconds dense_kernel;
    MeasuredSeconds sparse_round;
    MeasuredSeconds dense_round;
    MeasuredSeconds frontier_switch;
    MeasuredSeconds sparse_communication;
    MeasuredSeconds dense_communication;
};

void update_critical_path(
    MeasuredSeconds* aggregate, const MeasuredSeconds& rank) {
    if (!rank.measured) return;
    aggregate->measured = true;
    aggregate->seconds = std::max(aggregate->seconds, rank.seconds);
}

NativeBfsResult merge_bfs_result(
    const PartitionedBfsPlan& plan,
    const std::shared_ptr<SharedBfsExecution>& shared,
    bool move_output, bool direct_assembly = false) {
    const SteadyClock::time_point begin = SteadyClock::now();
    if (plan.vertex_count > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("BFS result exceeds host address space");
    }
    NativeBfsResult result;
    std::vector<std::vector<std::int64_t> > owned;
    result.direct_assembly = move_output && direct_assembly;
    if (shared->public_distances) {
        shared->public_output_observation.verify(plan.vertex_count);
    } else if (result.direct_assembly) {
        // The immutable host plan already proves a disjoint cover. Every
        // output position is written below, so an earlier zero fill is unused.
        result.direct_distance_count = static_cast<std::size_t>(plan.vertex_count);
        if (shared->host_assembly == BfsHostAssembly::Legacy)
            result.direct_distances.reset(new std::int64_t[result.direct_distance_count]);
        else if (shared->host_assembly == BfsHostAssembly::Caller) {
            for (std::size_t rank = 0; rank < plan.partitions.size(); ++rank)
                assemble_bfs_rank_output(plan, shared, rank);
        }
        if (shared->host_assembly != BfsHostAssembly::Legacy)
            result.direct_distances = std::move(shared->assembled_distances);
    } else if (move_output) result.distances.resize(static_cast<std::size_t>(plan.vertex_count));
    else owned.resize(plan.partitions.size());
    result.resident_bytes = 0;
    result.state_upload_seconds = 0.0;
    result.device_view_cache_hit = true;
    for (std::size_t rank = 0; rank < plan.partitions.size(); ++rank) {
        const BfsHostPartition& partition = plan.partitions[rank];
        const RankState& state = shared->ranks[rank];
        if (shared->host_assembly != BfsHostAssembly::Legacy) {
            result.reachable += state.reached_count;
        } else if (move_output) {
            // The validated partition plan owns every global vertex exactly
            // once. Scatter while the caller still holds the pinned lease.
            const std::int64_t* observed = static_cast<const std::int64_t*>(
                shared->output_buffers[rank] ? shared->output_buffers[rank]->data() : nullptr);
            std::int64_t* destination = result.direct_assembly
                ? result.direct_distances.get() : result.distances.data();
            const std::size_t count = partition.owned_vertex_count;
            // The cached plan also proves sorted, unique owner IDs. Endpoints
            // suffice to recognize a range; arbitrary layouts keep scatter.
            const bool contiguous = result.direct_assembly && count &&
                partition.local_to_global[count-1] - partition.local_to_global[0] == count-1;
            if (contiguous) {
                destination += partition.local_to_global[0];
                std::uint64_t reached_count = 0;
                for (std::size_t local = 0; local < count; ++local) {
                    const std::int64_t distance = observed[local];
                    const bool reached = distance != gpu_easygraph::partitioned::kBfsUnvisitedDistance;
                    destination[local] = reached ? distance : -1;
                    reached_count += reached ? 1U : 0U;
                }
                result.reachable += reached_count;
            } else {
                for (std::size_t local = 0; local < count; ++local) {
                    const std::int64_t distance = observed[local];
                    const bool reached = distance != gpu_easygraph::partitioned::kBfsUnvisitedDistance;
                    destination[partition.local_to_global[local]] = reached ? distance : -1;
                    result.reachable += reached ? 1U : 0U;
                }
            }
        } else {
            if (state.owned_distances.size() != partition.owned_vertex_count) {
                throw std::runtime_error("partitioned BFS owner result has the wrong length");
            }
            owned[rank] = state.owned_distances;
        }
        result.resident_bytes += static_cast<std::uint64_t>(
            state.resident_bytes);
        result.state_upload_seconds = std::max(
            result.state_upload_seconds, state.state_upload_seconds);
        result.device_view_cache_hit = result.device_view_cache_hit &&
            state.device_view_cache_hit;
        update_critical_path(&result.kernel, state.timing.kernel);
        update_critical_path(&result.pack, state.timing.pack);
        update_critical_path(&result.transfer, state.timing.transfer);
        update_critical_path(&result.wait, state.timing.wait);
        update_critical_path(
            &result.communication, state.timing.communication);
        update_critical_path(
            &result.output_gather, state.timing.output_gather);
        update_critical_path(
            &result.sparse_kernel, state.timing.sparse_kernel);
        update_critical_path(
            &result.dense_kernel, state.timing.dense_kernel);
        update_critical_path(
            &result.sparse_round, state.timing.sparse_round);
        update_critical_path(
            &result.dense_round, state.timing.dense_round);
        update_critical_path(
            &result.frontier_switch, state.timing.frontier_switch);
        update_critical_path(
            &result.sparse_communication,
            state.timing.sparse_communication);
        update_critical_path(
            &result.dense_communication,
            state.timing.dense_communication);
    }
    if (!move_output) result.distances =
        gpu_easygraph::runtime::merge_partitioned_bfs_owner_distances(
            plan,
            owned,
            gpu_easygraph::partitioned::kBfsUnvisitedDistance);
    result.merge_seconds = elapsed_seconds(begin, SteadyClock::now());
    return result;
}

py::array_t<std::int64_t> move_bfs_result_array(NativeBfsResult* result) {
    if (result->direct_assembly) {
        std::int64_t* data = result->direct_distances.get();
        const py::ssize_t count = static_cast<py::ssize_t>(result->direct_distance_count);
        py::capsule owner(data, [](void* pointer) {
            delete[] static_cast<std::int64_t*>(pointer);
        });
        result->direct_distances.release();
        return py::array_t<std::int64_t>({count}, {sizeof(std::int64_t)}, data, owner);
    }
    std::unique_ptr<std::vector<std::int64_t> > storage(
        new std::vector<std::int64_t>(std::move(result->distances)));
    std::int64_t* data = storage->data();
    const py::ssize_t count = static_cast<py::ssize_t>(storage->size());
    py::capsule owner(storage.get(), [](void* pointer) {
        delete static_cast<std::vector<std::int64_t>*>(pointer);
    });
    storage.release();
    return py::array_t<std::int64_t>({count}, {sizeof(std::int64_t)}, data, owner);
}

py::dict run_partitioned_bfs(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    std::uint64_t source,
    std::uint64_t max_iterations,
    const std::string& frontier_policy_text,
    py::object device_request,
    double dense_vertex_enter,
    double dense_vertex_exit,
    double dense_edge_enter,
    double dense_edge_exit,
    double dense_boundary_enter,
    bool dense_output, bool public_fp64_output) {
    if (!graph) throw std::invalid_argument("partitioned BFS graph is null");
    const bool profile = bfs_profile_setting("EGMGPU_BFS_PROFILE", true);
    const char* edge_schedule_env = std::getenv("EGMGPU_BFS_EDGE_SCHEDULE");
    const std::string edge_schedule = edge_schedule_env ? edge_schedule_env : "thread";
    if (edge_schedule != "thread" && edge_schedule != "warp") {
        throw std::invalid_argument("EGMGPU_BFS_EDGE_SCHEDULE must be thread or warp");
    }
    const int sparse_ctas_per_row = bfs_sparse_ctas_per_row(edge_schedule);

    const char* route_env = std::getenv("EGMGPU_BFS_ROUTE_MODE");
    const std::string route_mode = route_env ? route_env : "compact";
    if (route_mode != "compact" && route_mode != "direct")
        throw std::invalid_argument("EGMGPU_BFS_ROUTE_MODE must be compact or direct");
    const char* output_env = std::getenv("EGMGPU_BFS_OUTPUT_MODE");
    const std::string output_mode = output_env ? output_env : "copy";
    if (output_mode != "copy" && output_mode != "move")
        throw std::invalid_argument("EGMGPU_BFS_OUTPUT_MODE must be copy or move");
    const char* sync_env = std::getenv("EGMGPU_BFS_SYNC_MODE");
    const std::string sync_mode = sync_env ? sync_env : "standard";
    if (sync_mode != "standard" && sync_mode != "trim")
        throw std::invalid_argument("EGMGPU_BFS_SYNC_MODE must be standard or trim");
    const bool move_output = dense_output && output_mode == "move";
    const bool direct_assembly = bfs_direct_result_assembly();
    const BfsHostAssembly host_assembly = bfs_host_assembly(move_output, direct_assembly);
    if (public_fp64_output && (!dense_output || !move_output || !direct_assembly || host_assembly != BfsHostAssembly::Workers))
        throw std::invalid_argument("BFS public_fp64_output requires complete dense move output and direct worker assembly");
    const bool core_output_team16 = owner_output::enabled(true);
    if (core_output_team16 && host_assembly != BfsHostAssembly::Workers)
        throw std::invalid_argument("BFS core output team16 requires workers host assembly");
    const bool numa_local_output = bfs_numa_local_output();
    const BfsDenseDirection dense_direction = bfs_dense_direction(*graph);

    const gpu_easygraph::runtime::BfsFrontierPolicy frontier_policy =
        gpu_easygraph::runtime::parse_bfs_frontier_policy(
            frontier_policy_text);
    BfsFrontierThresholds frontier_thresholds;
    frontier_thresholds.dense_vertex_enter = dense_vertex_enter;
    frontier_thresholds.dense_vertex_exit = dense_vertex_exit;
    frontier_thresholds.dense_edge_enter = dense_edge_enter;
    frontier_thresholds.dense_edge_exit = dense_edge_exit;
    frontier_thresholds.dense_boundary_enter = dense_boundary_enter;
    gpu_easygraph::runtime::validate_bfs_frontier_thresholds(
        frontier_thresholds);
    const std::vector<int> devices = requested_devices(
        *graph, device_request);
    const int core_output_width = core_output_team16 ? owner_output::requested_width(devices.size()) : 1;
    const auto core_output_placement = std::make_shared<const owner_output::CpuPlacement>(
        owner_output::parse_placement(core_output_team16, devices.size()));
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

    const SteadyClock::time_point preparation_begin = SteadyClock::now();
    const gpu_easygraph::runtime::PartitionedHostPlanCacheKey plan_key =
        gpu_easygraph::runtime::make_partitioned_host_plan_cache_key(
            gpu_easygraph::runtime::PartitionedHostPlanAlgorithm::Bfs,
            graph->graph_id(),
            graph->generation(),
            graph->view_signature(),
            graph->topology_signature(),
            graph->weighted(),
            graph->weight_signature(),
            graph->vertex_count(),
            graph->adjacency_count(),
            devices);
    gpu_easygraph::runtime::PartitionedHostPlanCache<
        PartitionedBfsPlan>::AcquireResult plan_acquisition;
    {
        py::gil_scoped_release release;
        plan_acquisition =
            gpu_easygraph::runtime::partitioned_bfs_host_plan_cache().acquire(
                plan_key,
                [&graph, &devices]() {
                    std::shared_ptr<PartitionedBfsPlan> prepared(
                        new PartitionedBfsPlan(gpu_easygraph::runtime::
                            build_partitioned_bfs_plan(
                                host_partitions(*graph), graph->vertex_count())));
                    prepared->prepared_uploads.reserve(prepared->partitions.size());
                    for (std::size_t rank = 0; rank < prepared->partitions.size(); ++rank) {
                        prepared->prepared_uploads.push_back(
                            std::make_shared<BfsPartitionUpload>(make_upload(
                                *graph, *prepared, devices, rank)));
                    }
                    return std::shared_ptr<const PartitionedBfsPlan>(prepared);
                });
    }
    const std::shared_ptr<const PartitionedBfsPlan> plan =
        plan_acquisition.plan;
    const double state_preparation_seconds = elapsed_seconds(
        preparation_begin, SteadyClock::now());
    const std::size_t source_rank = plan->owner_rank(source);
    const std::size_t source_local = plan->owner_local(source);
    if (max_iterations == 0) max_iterations = graph->vertex_count();
    if (max_iterations == 0) {
        throw std::invalid_argument(
            "partitioned BFS requires a positive iteration limit");
    }

    const SteadyClock::time_point pinned_begin = SteadyClock::now();
    std::string pinned_key("bfs:pinned:v1:");
    for (int device : devices) pinned_key += std::to_string(device) + ":";
    const bool use_host_staging = backend == CommunicationBackend::HostStaging;
    pinned_key += use_host_staging ? "staging" : "device";
    pinned_key += numa_local_output ? ":numa-local-output" : ":caller-output";
    const char* pinned_reuse_setting = std::getenv("EGMGPU_BFS_PINNED_REUSE");
    const bool pinned_reuse = !pinned_reuse_setting ||
        std::string(pinned_reuse_setting) != "0";
    std::shared_ptr<BfsPinnedHostState> pinned;
    // This lease belongs to the calling thread. GPU workers never unlock it.
    std::unique_lock<std::mutex> pinned_lease;
    bool pinned_cache_hit = false;
    double pinned_lease_wait_seconds = 0.0;
    std::size_t pinned_new_allocations = 0;
    {
        py::gil_scoped_release release;
        if (pinned_reuse) {
            pinned = std::static_pointer_cast<BfsPinnedHostState>(
                gpu_easygraph::runtime::runtime_manager().acquire_resident_resource(
                    pinned_key,
                    []() { return std::shared_ptr<void>(new BfsPinnedHostState()); },
                    &pinned_cache_hit));
        } else {
            pinned.reset(new BfsPinnedHostState());
        }
        const SteadyClock::time_point lease_begin = SteadyClock::now();
        pinned_lease = std::unique_lock<std::mutex>(pinned->lease_mutex);
        pinned_lease_wait_seconds = elapsed_seconds(lease_begin, SteadyClock::now());
        const std::size_t allocations_before = pinned->allocation_count;
        pinned->ensure(*plan, use_host_staging, numa_local_output);
        pinned_new_allocations = pinned->allocation_count - allocations_before;
    }
    const double host_pinned_preparation_seconds = elapsed_seconds(
        pinned_begin, SteadyClock::now());
    const std::size_t pinned_retained_bytes = pinned->retained_bytes;
    const auto result_allocation_begin = SteadyClock::now();
    std::unique_ptr<easygraph::egmgpu::BfsPublicOutput> public_output;
    std::unique_ptr<std::int64_t[]> assembled_distances;
    if (public_fp64_output) public_output.reset(new easygraph::egmgpu::BfsPublicOutput(plan->vertex_count));
    else assembled_distances = allocate_bfs_host_output(plan->vertex_count, host_assembly);
    const double host_result_allocation_seconds = elapsed_seconds(result_allocation_begin, SteadyClock::now());
    const SteadyClock::time_point task_preparation_begin = SteadyClock::now();
    const std::shared_ptr<SharedBfsExecution> shared(
        new SharedBfsExecution(*plan, pinned, frontier_policy, frontier_thresholds));
    shared->control_fusion_used = shared->control_fusion_requested && devices.size() > 1U;
    shared->host_assembly = host_assembly;
    shared->core_output_team16 = core_output_team16;
    shared->core_output_width = core_output_width;
    shared->core_output_workers = devices.size();
    shared->core_output_placement = core_output_placement;
    shared->host_usage_profile = bfs_profile_setting("EGMGPU_BFS_HOST_USAGE_PROFILE", false);
    shared->worker_gpu_numa = bfs_worker_gpu_numa();
    shared->assembled_distances = std::move(assembled_distances);
    if (public_output) shared->public_distances = public_output->data();
    const bool single_partition_ordering = single_partition_candidate_enabled() &&
        devices.size() == 1U && plan->partitions.size() == 1U;
    for (RankState& state : shared->ranks) {
        state.warp_edges = edge_schedule == "warp";
        state.sparse_ctas_per_row = sparse_ctas_per_row;
        state.direct_routes = route_mode == "direct";
        state.move_output = move_output;
        state.trim_barriers = sync_mode == "trim";
        state.single_partition_ordering = single_partition_ordering;
    }
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.reserve(devices.size());
    std::vector<std::size_t> execution_ranks;
    execution_ranks.reserve(devices.size());
    for (std::size_t execution_index = 0;
         execution_index < devices.size(); ++execution_index) {
        std::size_t rank = 0;
        while (rank < plan->partitions.size() &&
               plan->partitions[rank].device_id != devices[execution_index]) {
            ++rank;
        }
        if (rank == plan->partitions.size()) {
            throw std::logic_error(
                "partitioned BFS could not map a device to its partition");
        }
        execution_ranks.push_back(rank);
    }
    for (std::size_t execution_index = 0;
         execution_index < execution_ranks.size(); ++execution_index) {
        const std::size_t rank = execution_ranks[execution_index];
        const std::shared_ptr<const BfsPartitionUpload> upload_ptr =
            plan->prepared_uploads.at(rank);
        tasks.push_back(
            [shared, plan, upload_ptr, devices, topology, backend, source_rank,
             source_local, max_iterations, rank, profile, dense_direction](int worker_device) {
                BfsWorkerProfileScope profile_scope(profile);
                gpu_easygraph::runtime::run_cooperative_rank(
                    shared->barrier,
                    [shared, plan, upload_ptr, devices, topology, backend,
                     source_rank, source_local, max_iterations, rank,
                     worker_device, dense_direction]() {
                        if (shared->host_usage_profile) shared->ranks[rank].worker_usage.start();
                        gpu_easygraph::runtime::run_with_gpu_worker_cpu_placement(
                            worker_device, shared->worker_gpu_numa, shared->host_usage_profile,
                            &shared->ranks[rank].cpu_placement, [&]() {
                        const BfsPartitionUpload& upload = *upload_ptr;
                        shared->phases[rank] = "worker_start";
                        if (worker_device != upload.device_id) {
                            throw std::runtime_error(
                                "partitioned BFS ran on the wrong worker");
                        }
                        if (backend == CommunicationBackend::PeerToPeer) {
                            shared->phases[rank] = "enable_peer_access";
                            gpu_easygraph::runtime::enable_worker_peer_access(
                                DeviceGroup(
                                    devices,
                                    CommunicationBackend::PeerToPeer),
                                topology);
                        }
                        RankState& state = shared->ranks[rank];
                        state.send_counts.assign(devices.size(), 0);
                        shared->phases[rank] = "acquire_resident_state";
                        const SteadyClock::time_point upload_begin =
                            SteadyClock::now();
                        gpu_easygraph::runtime::PartitionedBfsStateRegistry&
                            registry = gpu_easygraph::runtime::
                                partitioned_bfs_state_registry();
                        state.view = registry.acquire(
                            upload, &state.device_view_cache_hit,
                            &state.shared_arrays);
                        state.workspace = registry.workspace(state.view);
                        if (state.sparse_ctas_per_row)
                            state.workspace->ensure_high_degree(state.view->view());
                        state.high_degree_workspace_bytes = state.workspace->high_degree_capacity_bytes();
                        state.state_upload_seconds = elapsed_seconds(
                            upload_begin, SteadyClock::now());
                        state.resident_bytes = state.view->resident_bytes() +
                            state.workspace->capacity_bytes();
                        const BfsPartitionDeviceView& view =
                            state.view->view();
                        BfsPartitionWorkspace& workspace = *state.workspace;
                        WorkerStreams& streams =
                            gpu_easygraph::runtime::worker_streams();

                        if (state.single_partition_ordering &&
                            (view.local_vertex_count != view.owned_vertex_count ||
                             view.outgoing_route_offsets.back() != 0U))
                            throw std::logic_error("P1 BFS requires a complete owner with no remote routes");
                        shared->phases[rank] = "initialize_frontiers";
                        if (state.direct_routes && view.outgoing_route_offsets.back() != 0) {
                            CudaEventTimer timer("measure BFS remote discovery initialization");
                            timer.start(streams.compute());
                            check_cuda(cudaMemsetAsync(workspace.send_flags(), 0,
                                view.outgoing_route_offsets.back() * sizeof(int), streams.compute()),
                                "clear per-query BFS remote discoveries");
                            state.timing.kernel.add(timer.finish(streams.compute(), !state.single_partition_ordering));
                        }
                        if (view.owned_vertex_count != 0) {
                            CudaEventTimer timer(
                                "measure BFS initialization");
                            timer.start(streams.compute());
                            const int local = rank == source_rank
                                ? static_cast<int>(source_local)
                                : -1;
                            gpu_easygraph::partitioned::initialize_bfs_frontiers(
                                workspace.distances(),
                                workspace.current_frontier(),
                                workspace.current_bitmap(),
                                view.owned_vertex_count,
                                local,
                                streams.compute());
                            state.timing.kernel.add(
                                timer.finish(streams.compute(), !state.single_partition_ordering));
                            state.current_count = rank == source_rank ? 1U : 0U;
                            state.reached_count = state.current_count;
                        }
                        // The loop-entry barrier already publishes all
                        // initialization; standard retains its historical duplicate.
                        if (!state.trim_barriers) timed_barrier(
                            shared, &state.timing, devices.size() > 1U);

                        for (std::uint64_t iteration = 1;
                             iteration <= max_iterations; ++iteration) {
                            // Every rank enters the decision with its current
                            // owner-frontier count visible to the decision callback.
                            timed_control_reduction(
                                shared, &state.timing, devices.size() > 1U, rank, [&]() {
                                std::size_t active_vertices = 0;
                                double estimated_active_edges = 0.0;
                                const BfsFrontierMetrics metrics =
                                    frontier_metrics(
                                        *plan,
                                        shared,
                                        &active_vertices,
                                        &estimated_active_edges);
                                shared->selected_kind =
                                    gpu_easygraph::runtime::
                                        select_bfs_frontier_kind(
                                            shared->policy,
                                            metrics,
                                            shared->thresholds,
                                            shared->previous_kind);
                                shared->previous_kind = shared->selected_kind;
                                SharedBfsExecution::FrontierDecision decision;
                                decision.iteration = iteration;
                                decision.kind = shared->selected_kind;
                                decision.metrics = metrics;
                                decision.active_vertices = active_vertices;
                                decision.estimated_active_edges =
                                    estimated_active_edges;
                                shared->decisions.push_back(decision);
                            });
                            const BfsFrontierKind kind = shared->selected_kind;
                            if (state.have_last_kind && kind != state.last_kind) {
                                ++state.frontier_switches;
                                if (kind == BfsFrontierKind::Dense) {
                                    shared->phases[rank] =
                                        "sparse_to_dense_frontier";
                                    convert_sparse_frontier_to_bitmap(
                                        view, workspace, &state, streams);
                                }
                            }
                            state.last_kind = kind;
                            state.have_last_kind = true;

                            const SteadyClock::time_point round_begin =
                                SteadyClock::now();
                            if (kind == BfsFrontierKind::Sparse) {
                                ++state.sparse_rounds;
                                shared->phases[rank] = "local_sparse_expand";
                                clear_and_expand_sparse(
                                    view,
                                    workspace,
                                    &state,
                                    static_cast<std::int64_t>(iteration),
                                    streams);

                                const SteadyClock::time_point
                                    communication_begin = SteadyClock::now();
                                shared->phases[rank] = "compact_owner_routes";
                                if (!state.direct_routes) compact_routes(
                                    view, workspace, &state.timing, streams);
                                copy_send_counts_to_host(
                                    shared,
                                    rank,
                                    workspace,
                                    streams,
                                    &state);
                                std::uint64_t round_bytes = 0;
                                for (std::size_t peer = 0;
                                     peer < state.send_counts.size(); ++peer) {
                                    round_bytes += static_cast<std::uint64_t>(
                                        state.send_counts[peer]) * sizeof(int);
                                }
                                state.sparse_logical_bytes += round_bytes;
                                state.sparse_payload_transfer_bytes +=
                                    backend == CommunicationBackend::HostStaging
                                    ? round_bytes * 2U
                                    : round_bytes;
                                if (backend ==
                                    CommunicationBackend::HostStaging) {
                                    stage_outgoing_payloads(
                                        shared,
                                        *plan,
                                        rank,
                                        workspace,
                                        streams,
                                        &state.timing);
                                }
                                timed_barrier(
                                    shared,
                                    &state.timing,
                                    devices.size() > 1U);

                                shared->phases[rank] =
                                    "sparse_owner_frontier_transfer";
                                if (backend ==
                                    CommunicationBackend::PeerToPeer) {
                                    peer_transfer_payloads(
                                        shared,
                                        *plan,
                                        rank,
                                        worker_device,
                                        streams,
                                        &state.timing);
                                } else if (backend ==
                                    CommunicationBackend::HostStaging) {
                                    stage_incoming_payloads(
                                        shared,
                                        *plan,
                                        rank,
                                        workspace,
                                        streams,
                                        &state.timing);
                                }
                                timed_barrier(
                                    shared,
                                    &state.timing,
                                    devices.size() > 1U);

                                shared->phases[rank] =
                                    "sparse_owner_discovery_combine";
                                apply_owner_candidates(
                                    shared,
                                    *plan,
                                    rank,
                                    view,
                                    workspace,
                                    static_cast<std::int64_t>(iteration),
                                    streams,
                                    &state.timing);
                                copy_next_count_to_host(
                                    shared,
                                    rank,
                                    workspace,
                                    streams,
                                    &state);
                                timed_barrier(
                                    shared,
                                    &state.timing,
                                    devices.size() > 1U);
                                if (devices.size() > 1U) {
                                    const double seconds = elapsed_seconds(
                                        communication_begin,
                                        SteadyClock::now());
                                    state.timing.communication.add(seconds);
                                    state.timing.sparse_communication.add(
                                        seconds);
                                }
                                state.timing.sparse_round.add(elapsed_seconds(
                                    round_begin, SteadyClock::now()));
                            } else if (dense_direction.pull) {
                                ++state.dense_rounds;
                                const SteadyClock::time_point communication_begin = SteadyClock::now();
                                shared->phases[rank] = "publish_pull_frontier";
                                if (backend == CommunicationBackend::HostStaging)
                                    stage_pull_frontier(shared, *plan, rank, streams, &state.timing);
                                // Every sparse->dense bitmap conversion is complete before
                                // another device may read this rank's current bitmap.
                                timed_barrier(shared, &state.timing, devices.size() > 1U);
                                shared->phases[rank] = "acquire_pull_frontiers";
                                acquire_pull_frontiers(shared, *plan, rank, worker_device,
                                    backend, streams, &state.timing);
                                const std::uint64_t bytes = gpu_easygraph::runtime::bfs_bitmap_word_count(
                                    view.owned_vertex_count) * sizeof(std::uint32_t) * (devices.size() - 1U);
                                state.dense_logical_bytes += bytes;
                                state.dense_payload_transfer_bytes += backend == CommunicationBackend::HostStaging ? bytes * 2U : bytes;
                                if (devices.size() > 1U) {
                                    const double seconds = elapsed_seconds(communication_begin, SteadyClock::now());
                                    state.timing.communication.add(seconds); state.timing.dense_communication.add(seconds);
                                }
                                shared->phases[rank] = "local_dense_pull";
                                clear_and_expand_pull(view, workspace, &state,
                                    static_cast<std::int64_t>(iteration), streams, dense_direction.lanes);
                                copy_next_count_to_host(shared, rank, workspace, streams, &state);
                                // No current bitmap is recycled until all receivers have
                                // completed their reads and every owner has its next count.
                                timed_barrier(shared, &state.timing, devices.size() > 1U);
                                state.timing.dense_round.add(elapsed_seconds(round_begin, SteadyClock::now()));
                            } else {
                                ++state.dense_rounds;
                                shared->phases[rank] =
                                    "local_dense_bitmap_expand";
                                clear_and_expand_dense(
                                    view,
                                    workspace,
                                    &state,
                                    static_cast<std::int64_t>(iteration),
                                    streams);
                                account_dense_payload(
                                    *plan, backend, &state);

                                const SteadyClock::time_point
                                    communication_begin = SteadyClock::now();
                                if (backend ==
                                    CommunicationBackend::HostStaging) {
                                    shared->phases[rank] =
                                        "stage_dense_owner_bitmaps";
                                    stage_dense_outgoing_bitmaps(
                                        shared,
                                        *plan,
                                        rank,
                                        workspace,
                                        streams,
                                        &state.timing);
                                }
                                timed_barrier(
                                    shared,
                                    &state.timing,
                                    devices.size() > 1U);

                                shared->phases[rank] =
                                    "dense_owner_bitmap_transfer";
                                if (backend ==
                                    CommunicationBackend::PeerToPeer) {
                                    peer_transfer_dense_bitmaps(
                                        shared,
                                        *plan,
                                        rank,
                                        worker_device,
                                        streams,
                                        &state.timing);
                                } else if (backend ==
                                    CommunicationBackend::HostStaging) {
                                    stage_dense_incoming_bitmaps(
                                        shared,
                                        *plan,
                                        rank,
                                        workspace,
                                        streams,
                                        &state.timing);
                                }
                                timed_barrier(
                                    shared,
                                    &state.timing,
                                    devices.size() > 1U);

                                shared->phases[rank] =
                                    "dense_owner_bitmap_combine";
                                apply_dense_owner_bitmaps(
                                    shared,
                                    *plan,
                                    rank,
                                    view,
                                    workspace,
                                    static_cast<std::int64_t>(iteration),
                                    streams,
                                    &state.timing);
                                copy_next_count_to_host(
                                    shared,
                                    rank,
                                    workspace,
                                    streams,
                                    &state);
                                timed_barrier(
                                    shared,
                                    &state.timing,
                                    devices.size() > 1U);
                                if (devices.size() > 1U) {
                                    const double seconds = elapsed_seconds(
                                        communication_begin,
                                        SteadyClock::now());
                                    state.timing.communication.add(seconds);
                                    state.timing.dense_communication.add(
                                        seconds);
                                }
                                state.timing.dense_round.add(elapsed_seconds(
                                    round_begin, SteadyClock::now()));
                            }

                            state.reached_count += static_cast<std::uint64_t>(state.next_count);
                            shared->phases[rank] = "global_termination";
                            if (rank == 0) {
                                shared->global_active = 0;
                                for (std::size_t peer = 0;
                                     peer < shared->ranks.size(); ++peer) {
                                    shared->global_active +=
                                        static_cast<std::size_t>(
                                            shared->ranks[peer].next_count);
                                }
                                shared->iterations = iteration;
                                shared->converged =
                                    shared->global_active == 0;
                            }
                            timed_barrier(
                                shared, &state.timing, devices.size() > 1U);
                            if (shared->converged) break;
                            state.current_count = static_cast<std::size_t>(
                                state.next_count);
                            workspace.advance_frontier();
                            // The next iteration's entry barrier publishes the
                            // count/frontier swap before rank zero reads it.
                            if (!state.trim_barriers) timed_barrier(
                                shared, &state.timing, devices.size() > 1U);
                        }

                        if (shared->host_usage_profile) {
                            state.before_output_usage.begin = state.worker_usage.begin;
                            state.before_output_usage.finish();
                            state.output_gather_usage.begin = state.before_output_usage.end;
                        }
                        if (view.owned_vertex_count != 0) {
                            shared->phases[rank] = "output_gather";
                            CudaEventTimer timer(
                                "measure BFS output gather");
                            timer.start(streams.transfer());
                            check_cuda(
                                cudaMemcpyAsync(
                                    shared->output_buffers[rank]->data(),
                                    workspace.distances(),
                                    view.owned_vertex_count *
                                        sizeof(std::int64_t),
                                    cudaMemcpyDeviceToHost,
                                    streams.transfer()),
                                "gather partitioned BFS owner distances");
                            state.timing.output_gather.add(
                                timer.finish(streams.transfer()));
                            const std::int64_t* observed =
                                static_cast<const std::int64_t*>(
                                    shared->output_buffers[rank]->data());
                            if (!state.move_output) state.owned_distances.assign(
                                observed,
                                observed + view.owned_vertex_count);
                        }
                        if (shared->host_usage_profile) state.output_gather_usage.finish();
                        if (shared->host_assembly == BfsHostAssembly::Workers) {
                            shared->phases[rank] = "host_owner_assembly";
                            assemble_bfs_rank_output(*plan, shared, rank);
                        }
                        shared->phases[rank] = "complete";
                        });
                        if (shared->host_usage_profile) shared->ranks[rank].worker_usage.finish();
                    });
            });
    }

    const double host_task_preparation_seconds = elapsed_seconds(
        task_preparation_begin, SteadyClock::now());
    const SteadyClock::time_point device_begin = SteadyClock::now();
    {
        py::gil_scoped_release release;
        try {
            gpu_easygraph::runtime::runtime_manager().execute_gang(
                devices, std::move(tasks));
        } catch (const std::exception& error) {
            std::string context("partitioned BFS worker failure (");
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
            "partitioned BFS did not finish within max_iterations");
    }

    NativeBfsResult native;
    double host_pinned_release_seconds = 0.0;
    {
        py::gil_scoped_release release;
        native = merge_bfs_result(*plan, shared, move_output, direct_assembly);
        if (!pinned_reuse) {
            const SteadyClock::time_point release_begin = SteadyClock::now();
            shared->host_payloads.clear();
            shared->host_counts.clear();
            shared->output_buffers.clear();
            pinned_lease.unlock();
            pinned.reset();
            host_pinned_release_seconds = elapsed_seconds(
                release_begin, SteadyClock::now());
        }
    }

    const SteadyClock::time_point materialization_begin = SteadyClock::now();
    py::dict values;
    py::array values_dense;
    std::uint64_t reachable = 0;
    if (public_output) {
        reachable = native.reachable;
        if (shared->public_output_observation.unreachable.load() != plan->vertex_count - reachable)
            throw std::runtime_error("BFS worker FP64 output and frontier reachability disagree");
        values_dense = public_output->publish(shared->public_output_observation);
    } else if (move_output) {
        reachable = native.reachable;
        values_dense = move_bfs_result_array(&native);
    } else if (dense_output) {
        values_dense = py::array_t<std::int64_t>(native.distances.size());
        std::copy(native.distances.begin(), native.distances.end(),
                  static_cast<std::int64_t*>(values_dense.mutable_data()));
        reachable = static_cast<std::uint64_t>(std::count_if(
            native.distances.begin(), native.distances.end(),
            [](std::int64_t distance) { return distance >= 0; }));
    } else {
        for (std::size_t vertex = 0; vertex < native.distances.size(); ++vertex) {
            if (native.distances[vertex] >= 0) {
                values[py::int_(static_cast<std::uint64_t>(vertex))] =
                    py::int_(native.distances[vertex]);
                ++reachable;
            }
        }
    }
    const double python_materialization_seconds = elapsed_seconds(
        materialization_begin, SteadyClock::now());

    py::list per_device;
    py::list resident_device_bytes;
    std::uint64_t logical_messages = 0;
    std::uint64_t control_transfer_bytes = 0;
    std::uint64_t sparse_logical_bytes = 0;
    std::uint64_t sparse_payload_transfer_bytes = 0;
    std::uint64_t dense_logical_bytes = 0;
    std::uint64_t dense_payload_transfer_bytes = 0;
    std::uint64_t shared_array_hits = 0U;
    std::uint64_t shared_array_new_bytes = 0U;
    std::uint64_t shared_array_retained_bytes = 0U;
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        const RankState& state = shared->ranks[rank];
        logical_messages += state.logical_messages;
        control_transfer_bytes += state.control_transfer_bytes;
        sparse_logical_bytes += state.sparse_logical_bytes;
        sparse_payload_transfer_bytes +=
            state.sparse_payload_transfer_bytes;
        dense_logical_bytes += state.dense_logical_bytes;
        dense_payload_transfer_bytes +=
            state.dense_payload_transfer_bytes;
        py::dict item;
        item["device_id"] = plan->partitions[rank].device_id;
        item["host_assembly_seconds"] = state.host_assembly_seconds;
        item["core_output_thread_observation"] = owner_output::to_python(state.core_output_observation);
        item["worker_cpu_placement"] = bfs_worker_cpu_placement_dict(state.cpu_placement);
        if (shared->host_usage_profile) {
            py::dict usage;
            usage["worker_total"] = bfs_usage_window_dict(state.worker_usage);
            usage["before_output"] = bfs_usage_window_dict(state.before_output_usage);
            usage["output_gather"] = bfs_usage_window_dict(state.output_gather_usage);
            usage["host_owner_assembly"] = bfs_usage_window_dict(state.host_assembly_usage);
            item["thread_usage"] = std::move(usage);
        }
        item["reachable_owned_vertex_count"] = state.reached_count;
        item["output_pinned_numa_node"] = pinned->output_placement[rank].numa_node;
        item["output_pinned_allocation_cpu"] = pinned->output_placement[rank].allocation_cpu;
        item["output_pinned_allowed_node_cpus"] = pinned->output_placement[rank].allowed_node_cpus;
        item["kernel_seconds"] = timing_value(state.timing.kernel);
        item["pack_seconds"] = timing_value(state.timing.pack);
        item["transfer_seconds"] = timing_value(state.timing.transfer);
        item["wait_seconds"] = timing_value(state.timing.wait);
        item["communication_seconds"] =
            timing_value(state.timing.communication);
        item["output_gather_seconds"] =
            timing_value(state.timing.output_gather);
        item["sparse_kernel_seconds"] =
            timing_value(state.timing.sparse_kernel);
        item["dense_kernel_seconds"] =
            timing_value(state.timing.dense_kernel);
        item["sparse_frontier_seconds"] =
            timing_value(state.timing.sparse_round);
        item["dense_frontier_seconds"] =
            timing_value(state.timing.dense_round);
        item["frontier_switch_seconds"] =
            timing_value(state.timing.frontier_switch);
        item["sparse_communication_seconds"] =
            timing_value(state.timing.sparse_communication);
        item["dense_communication_seconds"] =
            timing_value(state.timing.dense_communication);
        item["sparse_rounds"] = state.sparse_rounds;
        item["dense_rounds"] = state.dense_rounds;
        item["frontier_switches"] = state.frontier_switches;
        item["sparse_logical_communication_bytes"] =
            state.sparse_logical_bytes;
        item["sparse_payload_transfer_bytes"] =
            state.sparse_payload_transfer_bytes;
        item["dense_logical_communication_bytes"] =
            state.dense_logical_bytes;
        item["dense_payload_transfer_bytes"] =
            state.dense_payload_transfer_bytes;
        item["resident_device_bytes"] =
            static_cast<std::uint64_t>(state.resident_bytes);
        item["high_degree_workspace_bytes"] = state.high_degree_workspace_bytes;
        resident_device_bytes.append(
            static_cast<std::uint64_t>(state.resident_bytes));
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
        per_device.append(item);
    }

    const std::uint64_t logical_bytes =
        sparse_logical_bytes + dense_logical_bytes;
    const std::uint64_t payload_transfer_bytes =
        sparse_payload_transfer_bytes + dense_payload_transfer_bytes;
    py::dict timing_sources;
    timing_sources["kernel"] = "cuda_event";
    timing_sources["pack"] = "cuda_event";
    timing_sources["transfer"] = "cuda_event";
    timing_sources["output_gather"] = "cuda_event";
    timing_sources["wait"] = "steady_clock_barrier";
    timing_sources["communication"] = "steady_clock_exchange_phase";
    timing_sources["native_merge"] = "steady_clock_host";
    timing_sources["sparse_frontier"] = "steady_clock_round";
    timing_sources["dense_frontier"] = "steady_clock_round";
    timing_sources["frontier_switch"] = "cuda_event";
    timing_sources["state_upload"] = "steady_clock_state_acquire";
    if (!profile) {
        for (const char* field : {"kernel", "pack", "transfer", "output_gather", "frontier_switch"})
            timing_sources[field] = py::none();
    }

    std::uint64_t sparse_rounds = 0;
    std::uint64_t dense_rounds = 0;
    std::uint64_t frontier_switches = 0;
    py::list frontier_trace;
    BfsFrontierKind previous_trace_kind = BfsFrontierKind::Sparse;
    bool have_previous_trace_kind = false;
    for (std::size_t index = 0; index < shared->decisions.size(); ++index) {
        const SharedBfsExecution::FrontierDecision& decision =
            shared->decisions[index];
        if (decision.kind == BfsFrontierKind::Sparse) ++sparse_rounds;
        else ++dense_rounds;
        if (have_previous_trace_kind &&
            decision.kind != previous_trace_kind) {
            ++frontier_switches;
        }
        previous_trace_kind = decision.kind;
        have_previous_trace_kind = true;
        py::dict trace;
        trace["iteration"] = decision.iteration;
        trace["frontier_kind"] = std::string(
            gpu_easygraph::runtime::bfs_frontier_kind_name(decision.kind));
        trace["active_vertices"] = decision.active_vertices;
        trace["estimated_active_edges"] = decision.estimated_active_edges;
        trace["active_vertex_fraction"] =
            decision.metrics.active_vertex_fraction;
        trace["estimated_active_edge_fraction"] =
            decision.metrics.estimated_active_edge_fraction;
        trace["boundary_edge_fraction"] =
            decision.metrics.boundary_edge_fraction;
        frontier_trace.append(trace);
    }
    py::dict threshold_report;
    threshold_report["dense_vertex_enter"] =
        frontier_thresholds.dense_vertex_enter;
    threshold_report["dense_vertex_exit"] =
        frontier_thresholds.dense_vertex_exit;
    threshold_report["dense_edge_enter"] =
        frontier_thresholds.dense_edge_enter;
    threshold_report["dense_edge_exit"] =
        frontier_thresholds.dense_edge_exit;
    threshold_report["dense_boundary_enter"] =
        frontier_thresholds.dense_boundary_enter;

    py::dict out;
    if (dense_output) {
        out["values_dense"] = std::move(values_dense);
        out["unreachable_distance"] = public_fp64_output ? py::object(py::float_(std::numeric_limits<double>::infinity())) : py::object(py::int_(-1));
    } else {
        out["values"] = std::move(values);
    }
    if (public_fp64_output) {
        out["bfs_worker_public_output"] = easygraph::egmgpu::bfs_public_output_proof(
            plan->vertex_count, shared->public_output_observation);
        out["output_representation"] = "owning_fp64_dense";
    }
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
    reduction_proof["scope"] = "multi-rank frontier-kind decision; no GPU completion or payload lifetime boundary removed";
    out["algorithm_control_reduction"] = reduction_proof;
    out["result_format"] = dense_output ? "dense" : "dict";
    out["edge_schedule"] = edge_schedule;
    out["sparse_schedule"] = bfs_sparse_schedule_name(sparse_ctas_per_row);
    out["high_degree_threshold"] = gpu_easygraph::partitioned::kBfsHighDegreeThreshold;
    out["route_mode"] = route_mode;
    out["output_mode"] = move_output ? "move" : "copy";
    out["result_assembly"] = native.direct_assembly ? "direct" : "zeroed";
    out["host_assembly"] = bfs_host_assembly_name(host_assembly);
    out["core_output_team_version"] = "stage4_core_owner_output_v1";
    out["core_output_team_policy"] = core_output_team16 ? "team16" : "original";
    out["core_output_workers"] = devices.size();
    out["core_output_requested_per_worker"] = core_output_width;
    out["core_output_cpu_placement"] = owner_output::placement_to_python(*core_output_placement);
    py::list core_output_observations;
    for (const auto& state : shared->ranks)
        core_output_observations.append(owner_output::to_python(state.core_output_observation));
    out["core_output_observations"] = std::move(core_output_observations);
    out["host_assembly_timing_scope"] = host_assembly == BfsHostAssembly::Workers
        ? "per-rank CPU owner copy/normalization including output first-touch, nested in device_processing wall"
        : "caller CPU owner copy/normalization including output first-touch, nested in native_merge wall";
    out["device_processing_timing_scope"] = "runtime gang wall including GPU execution, coordination, D2H and selected worker CPU output assembly";
    out["output_pinned_placement"] = numa_local_output ? "numa_local" : "caller";
    out["dense_direction"] = dense_direction.pull ? "pull" : "push";
    out["pull_lanes"] = dense_direction.lanes;
    out["sync_mode"] = sync_mode;
    out["profile_enabled"] = profile;
    out["host_usage_profile_enabled"] = shared->host_usage_profile;
    out["worker_cpu_placement_policy"] = shared->worker_gpu_numa ? "gpu_numa" : "none";
    out["worker_cpu_placement_scope"] = "per actual worker task, current allowed CPU mask intersect GPU NUMA CPUs; restore mask before task returns; discovery/bind/restore are inside gang; caller-side output assembly is outside this policy";
    if (shared->host_usage_profile) {
        out["host_usage_scope"] = "diagnostic-only RUSAGE_THREAD and CLOCK_THREAD_CPUTIME_ID; before_output/output_gather are nested in worker_total; host_owner_assembly is nested only in workers mode, caller mode is on another thread after worker completion; wall minus CPU includes blocking, waits and descheduling, not an attributed scheduler delay";
        out["diagnostic_performance_eligible"] = false;
    }
    out["profiling_scope"] = "optional CUDA event diagnostics; host wall intervals remain measured";
    out["source"] = source;
    out["reachable_vertex_count"] = reachable;
    out["unreachable_vertex_count"] = graph->vertex_count() - reachable;
    out["iterations"] = shared->iterations;
    out["converged"] = shared->converged;
    out["devices_used"] = devices;
    out["communication_backend"] = std::string(
        gpu_easygraph::runtime::communication_backend_name(backend));
    out["runtime_communication_backend"] = std::string(
        gpu_easygraph::runtime::communication_backend_name(runtime_backend));
    out["frontier_policy"] = std::string(
        gpu_easygraph::runtime::bfs_frontier_policy_name(frontier_policy));
    out["adaptive_frontier_available"] =
        gpu_easygraph::runtime::bfs_dense_frontier_available();
    out["frontier_thresholds"] = threshold_report;
    out["frontier_threshold_source"] = "binding_arguments";
    out["frontier_trace"] = frontier_trace;
    out["sparse_rounds"] = sparse_rounds;
    out["dense_rounds"] = dense_rounds;
    out["frontier_switches"] = frontier_switches;
    out["state_preparation_seconds"] = state_preparation_seconds;
    out["host_task_preparation_seconds"] = host_task_preparation_seconds;
    out["host_result_allocation_seconds"] = host_result_allocation_seconds;
    out["host_pinned_preparation_seconds"] = host_pinned_preparation_seconds;
    out["host_pinned_release_seconds"] = host_pinned_release_seconds;
    out["host_pinned_lease_wait_seconds"] = pinned_lease_wait_seconds;
    out["host_pinned_reuse"] = pinned_reuse;
    out["host_pinned_cache_hit"] = pinned_cache_hit;
    out["host_pinned_new_allocations"] = pinned_new_allocations;
    out["host_pinned_working_bytes"] = pinned_retained_bytes;
    out["host_pinned_retained_bytes"] = pinned_reuse ? pinned_retained_bytes : 0U;
    out["host_upload_descriptor_cache_hit"] = plan_acquisition.cache_hit;
    out["host_plan_cache_hit"] = plan_acquisition.cache_hit;
    out["host_plan_cache_builder"] = plan_acquisition.builder;
    out["host_plan_cache_waiter"] = plan_acquisition.waiter;
    if (plan_acquisition.build_seconds_measured) {
        out["host_plan_build_seconds"] = plan_acquisition.build_seconds;
    } else {
        out["host_plan_build_seconds"] = py::none();
    }
    out["host_plan_cache_entries"] = plan_acquisition.entry_count;
    out["state_upload_seconds"] = native.state_upload_seconds;
    out["device_view_cache_hit"] = native.device_view_cache_hit;
    out["device_processing_seconds"] = device_processing_seconds;
    out["kernel_seconds"] = timing_value(native.kernel);
    out["pack_seconds"] = timing_value(native.pack);
    out["transfer_seconds"] = timing_value(native.transfer);
    out["collective_seconds"] = py::none();
    out["wait_seconds"] = timing_value(native.wait);
    out["communication_seconds"] = timing_value(native.communication);
    out["output_gather_seconds"] = timing_value(native.output_gather);
    out["native_merge_seconds"] = native.merge_seconds;
    out["python_materialization_seconds"] =
        python_materialization_seconds;
    out["sparse_kernel_seconds"] = timing_value(native.sparse_kernel);
    out["dense_kernel_seconds"] = timing_value(native.dense_kernel);
    out["sparse_frontier_seconds"] = timing_value(native.sparse_round);
    out["dense_frontier_seconds"] = timing_value(native.dense_round);
    out["frontier_switch_seconds"] = timing_value(native.frontier_switch);
    out["sparse_communication_seconds"] =
        timing_value(native.sparse_communication);
    out["dense_communication_seconds"] =
        timing_value(native.dense_communication);
    out["sparse_logical_communication_bytes"] = sparse_logical_bytes;
    out["sparse_logical_messages"] = logical_messages;
    out["dense_logical_communication_bytes"] = dense_logical_bytes;
    out["sparse_payload_transfer_bytes"] = sparse_payload_transfer_bytes;
    out["dense_payload_transfer_bytes"] = dense_payload_transfer_bytes;
    out["logical_communication_bytes"] = logical_bytes;
    out["payload_transfer_bytes"] = payload_transfer_bytes;
    out["control_transfer_bytes"] = control_transfer_bytes;
    out["physical_transfer_bytes"] =
        payload_transfer_bytes + control_transfer_bytes;
    out["single_partition_ordering"] = single_partition_ordering;
    out["output_gather_bytes"] =
        graph->vertex_count() * sizeof(std::int64_t);
    out["resident_device_bytes"] = resident_device_bytes;
    out["aggregate_resident_device_bytes"] = native.resident_bytes;
    out["shared_array_hits"] = shared_array_hits;
    out["shared_array_new_bytes"] = shared_array_new_bytes;
    out["shared_array_retained_bytes"] = shared_array_retained_bytes;
    out["timing_sources"] = timing_sources;
    out["per_device"] = per_device;
    return out;
}

#include "egmgpu_bfs_serial.inc"

py::dict run_serial_partitioned_bfs_core_output_guard(
    const std::shared_ptr<MultiPartitionGraph>& graph, std::uint64_t source,
    std::uint64_t max_iterations, const std::string& frontier_policy_text,
    int device, double dense_vertex_enter, double dense_vertex_exit,
    double dense_edge_enter, double dense_edge_exit, double dense_boundary_enter) {
    return run_serial_partitioned_bfs(graph, source, max_iterations, frontier_policy_text,
        device, dense_vertex_enter, dense_vertex_exit, dense_edge_enter, dense_edge_exit,
        dense_boundary_enter);
}

py::dict host_plan_cache_stats_dict(
    const gpu_easygraph::runtime::PartitionedHostPlanCacheStats& stats) {
    py::dict result;
    result["hits"] = stats.hits;
    result["builders"] = stats.builders;
    result["waiters"] = stats.waiters;
    result["build_failures"] = stats.build_failures;
    result["evictions"] = stats.evictions;
    result["generation_invalidations"] = stats.generation_invalidations;
    result["entries"] = stats.entries;
    result["capacity"] = stats.capacity;
    return result;
}

py::dict clear_partitioned_host_plan_cache() {
    const gpu_easygraph::runtime::PartitionedHostPlanCacheClearResult cleared =
        gpu_easygraph::runtime::clear_partitioned_host_plan_caches();
    py::dict result;
    result["bfs"] = cleared.bfs;
    result["wcc"] = cleared.wcc;
    result["sssp"] = cleared.sssp;
    result["ego"] = cleared.ego;
    result["total"] = cleared.total;
    return result;
}

py::dict partitioned_host_plan_cache_stats() {
    const gpu_easygraph::runtime::PartitionedHostPlanCacheRegistryStats stats =
        gpu_easygraph::runtime::partitioned_host_plan_cache_stats();
    py::dict result;
    result["bfs"] = host_plan_cache_stats_dict(stats.bfs);
    result["wcc"] = host_plan_cache_stats_dict(stats.wcc);
    result["sssp"] = host_plan_cache_stats_dict(stats.sssp);
    result["ego"] = host_plan_cache_stats_dict(stats.ego);
    return result;
}
#endif

}  // namespace

void bind_egmgpu_partitioned_bfs(py::module_& module) {
    module.def("cpp_egmgpu_finalize_bfs_int64", &easygraph::egmgpu::finalize_bfs_int64,
        py::arg("distances"), py::arg("count"),
        "Normalize a complete native int64 BFS vector into fresh owning float64 values on CPU; uncertified values retain the Python fallback.");
#ifdef EASYGRAPH_ENABLE_GPU
    module.def(
        "cpp_egmgpu_partitioned_bfs",
        &run_partitioned_bfs,
        py::arg("partitioned_graph"),
        py::arg("source"),
        py::arg("max_iterations") = 0,
        py::arg("frontier_policy") = "adaptive",
        py::arg("devices") = py::none(),
        py::arg("dense_vertex_enter") = 0.08,
        py::arg("dense_vertex_exit") = 0.04,
        py::arg("dense_edge_enter") = 0.10,
        py::arg("dense_edge_exit") = 0.05,
        py::arg("dense_boundary_enter") = 0.05,
        py::arg("dense_output") = false,
        py::arg("public_fp64_output") = false,
        "Run owner-partitioned single-source BFS on fixed GPU workers; optional public FP64 output fuses worker assembly.");
    module.def("cpp_egmgpu_partitioned_bfs_serial", &run_serial_partitioned_bfs_core_output_guard,
        py::arg("partitioned_graph"), py::arg("source"), py::arg("max_iterations")=0,
        py::arg("frontier_policy")="adaptive", py::arg("device")=0,
        py::arg("dense_vertex_enter")=0.08, py::arg("dense_vertex_exit")=0.04,
        py::arg("dense_edge_enter")=0.10, py::arg("dense_edge_exit")=0.05,
        py::arg("dense_boundary_enter")=0.05,
        "Run the same BFS owner partitions sequentially on one GPU with complete per-level owner exchange.");
    module.def(
        "cpp_egmgpu_clear_partitioned_host_plan_cache",
        &clear_partitioned_host_plan_cache,
        "Clear cached BFS, WCC, SSSP, and structural-hole host plans.");
    module.def(
        "cpp_egmgpu_partitioned_host_plan_cache_stats",
        &partitioned_host_plan_cache_stats,
        "Return host execution-plan cache counters by algorithm.");
#else
    (void)module;
#endif
}
