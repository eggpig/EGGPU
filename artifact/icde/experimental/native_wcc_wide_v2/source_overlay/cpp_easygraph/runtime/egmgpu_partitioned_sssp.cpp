#include "egmgpu_partitioned_sssp.h"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "egmgpu_sssp_host_output.h"

#ifdef EASYGRAPH_ENABLE_GPU
#include "egmgpu_partition_loader.h"
#include "egmgpu_partition_array_scope.h"

#include <functions/partitioned/sssp_partitioned.cuh>
#include <functions/partitioned/single_owner_sssp.cuh>
#include <runtime/sssp_numeric_domain.h>
#include <runtime/collective_transport.h>
#include <runtime/cooperative_barrier.h>
#include <runtime/algorithm_control_policy.h>
#include <runtime/native_runtime.h>
#include <runtime/partitioned_host_plan_registry.h>
#include <runtime/partitioned_sssp_plan.h>
#include <runtime/partitioned_sssp_state.h>
#include <runtime/sssp_execution_policy.h>
#include <runtime/numa_pinned_host.h>

#include <cuda_runtime_api.h>
#include <sched.h>
#include <cerrno>
#if EASYGRAPH_USE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
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
using gpu_easygraph::partitioned::SsspRelaxPolicy;
using gpu_easygraph::partitioned::SsspControlPolicy;
using gpu_easygraph::partitioned::SsspKernelPolicy;
using gpu_easygraph::partitioned::SsspAtomicMinPolicy;
using gpu_easygraph::runtime::CommunicationBackend;
using gpu_easygraph::runtime::CooperativeBarrier;
using gpu_easygraph::runtime::DeviceGroup;
using gpu_easygraph::runtime::DeviceTopology;
using gpu_easygraph::runtime::PartitionedSsspPlan;
using gpu_easygraph::runtime::PinnedHostBuffer;
using gpu_easygraph::runtime::ResidentSsspPartitionView;
using gpu_easygraph::runtime::SharedArrayCallStats;
using gpu_easygraph::runtime::SsspHostPartition;
using gpu_easygraph::runtime::SsspOwnerRoute;
using gpu_easygraph::runtime::SsspPartitionDeviceView;
using gpu_easygraph::runtime::SsspPartitionUpload;
using gpu_easygraph::runtime::SsspWeightEncoding;
using gpu_easygraph::runtime::sssp_weight_encoding_name;
using gpu_easygraph::runtime::SsspUploadBuildStats;
using gpu_easygraph::runtime::SsspPartitionWorkspace;
using gpu_easygraph::runtime::SsspRelaxation;
using gpu_easygraph::runtime::SsspWireEncoding;
using gpu_easygraph::runtime::sssp_wire_encoding_name;
using gpu_easygraph::runtime::sssp_wire_bytes;
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

bool sssp_profile_enabled() {
    const char* value = std::getenv("EGMGPU_SSSP_PROFILE");
    if (!value || std::string(value) == "1") return true;
    if (std::string(value) == "0") return false;
    throw std::invalid_argument("EGMGPU_SSSP_PROFILE must be 0 or 1");
}

bool sssp_option(const char* name) {
    const char* value = std::getenv(name);
    if (!value || std::string(value) == "1") return true;
    if (std::string(value) == "0") return false;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}

bool sssp_worker_output_enabled() {
    const char* value = std::getenv("EGMGPU_SSSP_HOST_ASSEMBLY");
    const std::string policy = value ? value : "caller";
    if (policy == "worker") return true;
    if (policy == "caller") return false;
    throw std::invalid_argument("EGMGPU_SSSP_HOST_ASSEMBLY must be caller or worker");
}

bool sssp_output_team16_enabled() {
    const char* raw = std::getenv("EGMGPU_SSSP_OUTPUT_TEAM");
    const std::string value = raw ? raw : "worker1";
    if (value == "worker1") return false;
    if (value == "team16") {
#if EASYGRAPH_USE_OPENMP
        return true;
#else
        throw std::invalid_argument("SSSP team16 requires this build to enable OpenMP");
#endif
    }
    throw std::invalid_argument("EGMGPU_SSSP_OUTPUT_TEAM must be worker1 or team16");
}

struct SsspOutputThreadObservation {
    cpu_set_t mask;
    int cpu = -1;
    int error = 0;
    SsspOutputThreadObservation() { CPU_ZERO(&mask); }
    void capture() {
        cpu = sched_getcpu();
        if (sched_getaffinity(0, sizeof(mask), &mask) != 0) error = errno;
    }
};

std::string sssp_pinned_placement() {
    const char* value = std::getenv("EGMGPU_SSSP_PINNED_PLACEMENT");
    const std::string policy = value ? value : "caller";
    if (policy == "caller" || policy == "numa_local") return policy;
    throw std::invalid_argument("EGMGPU_SSSP_PINNED_PLACEMENT must be caller or numa_local");
}

SsspWeightEncoding sssp_weight_encoding() {
    const char* raw = std::getenv("EGMGPU_SSSP_WEIGHT_ENCODING");
    const std::string value = raw ? raw : "f64";
    if (value == "f64") return SsspWeightEncoding::F64;
    if (value == "u8_exact") return SsspWeightEncoding::U8Exact;
    throw std::invalid_argument("EGMGPU_SSSP_WEIGHT_ENCODING must be f64 or u8_exact");
}

SsspWireEncoding sssp_wire_encoding() {
    const char* raw=std::getenv("EGMGPU_SSSP_WIRE_ENCODING");
    const std::string value=raw ? raw : "f64";
    if (value=="f64") return SsspWireEncoding::F64;
    if (value=="u32_exact") return SsspWireEncoding::U32Exact;
    throw std::invalid_argument("EGMGPU_SSSP_WIRE_ENCODING must be f64 or u32_exact");
}

std::string sssp_weight_identity() {
    const char* value = std::getenv("EGMGPU_SSSP_WEIGHT_IDENTITY");
    const std::string policy = value ? value : "manifest_f64";
    if (policy == "compute" || policy == "manifest_f64") return policy;
    throw std::invalid_argument(
        "EGMGPU_SSSP_WEIGHT_IDENTITY must be compute or manifest_f64");
}

SsspKernelPolicy sssp_kernel_policy() {
    SsspKernelPolicy policy;
    const char* relax_env = std::getenv("EGMGPU_SSSP_RELAX_POLICY");
    const std::string relax = relax_env ? relax_env : "thread";
    if (relax == "warp") policy.relax = SsspRelaxPolicy::Warp;
    else if (relax != "thread") throw std::invalid_argument(
        "EGMGPU_SSSP_RELAX_POLICY must be thread or warp");
    const char* control_env = std::getenv("EGMGPU_SSSP_CONTROL_POLICY");
    const std::string control = control_env ? control_env : "atomic";
    if (control == "warp") policy.control = SsspControlPolicy::Warp;
    else if (control != "atomic") throw std::invalid_argument(
        "EGMGPU_SSSP_CONTROL_POLICY must be atomic or warp");
    const char* atomic_env = std::getenv("EGMGPU_SSSP_ATOMIC_MIN");
    const std::string atomic_min = atomic_env ? atomic_env : "cas";
    if (atomic_min == "native_u64") policy.atomic_min = SsspAtomicMinPolicy::NativeU64;
    else if (atomic_min != "cas") throw std::invalid_argument(
        "EGMGPU_SSSP_ATOMIC_MIN must be cas or native_u64");
    return policy;
}

enum class SsspExchangeSync { Legacy, Trim };

SsspExchangeSync sssp_exchange_sync_policy() {
    const char* value = std::getenv("EGMGPU_SSSP_EXCHANGE_SYNC");
    const std::string policy = value ? value : "legacy";
    if (policy == "legacy") return SsspExchangeSync::Legacy;
    if (policy == "trim") return SsspExchangeSync::Trim;
    throw std::invalid_argument(
        "EGMGPU_SSSP_EXCHANGE_SYNC must be legacy or trim");
}

using gpu_easygraph::runtime::SsspExecutionPolicy;

SsspExecutionPolicy resolve_sssp_execution_policy(const py::object& requested) {
    SsspExecutionPolicy policy;
    if (requested.is_none()) {
        const char* local = std::getenv("EGGPU_SINGLE_OWNER_SSSP");
        if (local && std::string(local) != "0" && std::string(local) != "1")
            throw std::invalid_argument("EGGPU_SINGLE_OWNER_SSSP must be 0 or 1");
        policy.single_owner_execution = local && std::string(local) == "1" ? "arena_v3_if_eligible" : "off";
        // Compatibility only: legacy calls retain their process environment.
        policy.weight_encoding = sssp_weight_encoding_name(sssp_weight_encoding());
        policy.wire_encoding = sssp_wire_encoding_name(sssp_wire_encoding());
        policy.output_team = sssp_output_team16_enabled() ? "team16" : "worker1";
        const auto kernel = sssp_kernel_policy();
        policy.atomic_min = kernel.atomic_min == SsspAtomicMinPolicy::NativeU64 ? "native_u64" : "cas";
        policy.relax_policy = kernel.relax == SsspRelaxPolicy::Warp ? "warp" : "thread";
        policy.control_policy = kernel.control == SsspControlPolicy::Warp ? "warp" : "atomic";
        policy.exchange_sync = sssp_exchange_sync_policy() == SsspExchangeSync::Trim ? "trim" : "legacy";
        policy.pinned_placement = sssp_pinned_placement();
        policy.host_assembly = sssp_worker_output_enabled() ? "worker" : "caller";
        policy.weight_identity = sssp_weight_identity();
        policy.profile = sssp_profile_enabled();
        policy.pinned_reuse = sssp_option("EGMGPU_SSSP_PINNED_REUSE");
        policy.prepared_uploads = sssp_option("EGMGPU_SSSP_PREPARED_UPLOADS");
        policy.allow_undirected = sssp_option("EGMGPU_SSSP_ALLOW_UNDIRECTED");
    } else {
        if (!py::isinstance<py::dict>(requested))
            throw std::invalid_argument("SSSP execution_policy must be dict or None");
        const py::dict values = requested.cast<py::dict>();
        for (const auto& item : values) {
            const std::string key = py::cast<std::string>(item.first);
            std::string* string_target = nullptr;
            bool* bool_target = nullptr;
            if (key == "single_owner_execution") string_target = &policy.single_owner_execution;
            else if (key == "weight_encoding") string_target = &policy.weight_encoding;
            else if (key == "wire_encoding") string_target = &policy.wire_encoding;
            else if (key == "output_team") string_target = &policy.output_team;
            else if (key == "atomic_min") string_target = &policy.atomic_min;
            else if (key == "exchange_sync") string_target = &policy.exchange_sync;
            else if (key == "pinned_placement") string_target = &policy.pinned_placement;
            else if (key == "host_assembly") string_target = &policy.host_assembly;
            else if (key == "relax_policy") string_target = &policy.relax_policy;
            else if (key == "control_policy") string_target = &policy.control_policy;
            else if (key == "weight_identity") string_target = &policy.weight_identity;
            else if (key == "profile") bool_target = &policy.profile;
            else if (key == "pinned_reuse") bool_target = &policy.pinned_reuse;
            else if (key == "prepared_uploads") bool_target = &policy.prepared_uploads;
            else if (key == "allow_undirected") bool_target = &policy.allow_undirected;
            else throw std::invalid_argument("unknown SSSP execution_policy field: " + key);
            if (string_target) {
                if (!py::isinstance<py::str>(item.second))
                    throw std::invalid_argument("SSSP execution_policy string field: " + key);
                *string_target = py::cast<std::string>(item.second);
            } else {
                if (!py::isinstance<py::bool_>(item.second))
                    throw std::invalid_argument("SSSP execution_policy boolean field: " + key);
                *bool_target = py::cast<bool>(item.second);
            }
        }
    }
    policy.validate();
#if !EASYGRAPH_USE_OPENMP
    if (policy.output_team == "team16")
        throw std::invalid_argument("SSSP team16 requires this build to enable OpenMP");
#endif
    return policy;
}

py::dict sssp_execution_policy_document(const SsspExecutionPolicy& p) {
    py::dict out;
    out["single_owner_execution"]=p.single_owner_execution;
    out["weight_encoding"]=p.weight_encoding; out["wire_encoding"]=p.wire_encoding;
    out["output_team"]=p.output_team; out["atomic_min"]=p.atomic_min;
    out["exchange_sync"]=p.exchange_sync; out["pinned_placement"]=p.pinned_placement;
    out["host_assembly"]=p.host_assembly; out["relax_policy"]=p.relax_policy;
    out["control_policy"]=p.control_policy; out["weight_identity"]=p.weight_identity;
    out["profile"]=p.profile; out["pinned_reuse"]=p.pinned_reuse;
    out["prepared_uploads"]=p.prepared_uploads; out["allow_undirected"]=p.allow_undirected;
    return out;
}

void validate_sssp_view(const MultiPartitionGraph& graph, bool allow_undirected) {
    const bool orientation_ok = graph.view_orientation() == "outgoing" ||
        (allow_undirected && graph.view_orientation() == "undirected");
    if (!graph.weighted() || !orientation_ok || graph.weight_key().empty()) {
        throw std::invalid_argument(
            "partitioned SSSP requires weighted outgoing CSR or enabled weighted undirected CSR");
    }
}

// Fixed per query before worker dispatch; use one setting for an entire A/B process.
bool single_partition_candidate_enabled() {
    const char* value = std::getenv("EGMGPU_SINGLE_PARTITION_ORDERING");
    if (!value || std::string(value) == "0") return false;
    if (std::string(value) == "1") return true;
    throw std::invalid_argument("EGMGPU_SINGLE_PARTITION_ORDERING must be 0 or 1");
}

bool sssp_direct_frontier_requested() {
    const char* raw = std::getenv("EGGPU_SSSP_DIRECT_FRONTIER");
    const std::string value = raw ? raw : "0";
    if (value == "0") return false;
    if (value == "1") return true;
    throw std::invalid_argument("EGGPU_SSSP_DIRECT_FRONTIER must be 0 or 1");
}

class CudaEventTimer {
public:
    explicit CudaEventTimer(const char* context, bool enabled)
        : begin_(nullptr), end_(nullptr), context_(context), enabled_(enabled) {
        if (!enabled_) return;
        check_cuda(cudaEventCreate(&begin_), "create SSSP timing event");
        try {
            check_cuda(cudaEventCreate(&end_), "create SSSP timing event");
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
    double finish(cudaStream_t stream, bool require_completion = true) {
        if (!enabled_) {
            // Old event completion was also the publication boundary between
            // compute/pack/transfer streams and for host scalar consumption.
            // This first off mode removes timing events, preserving each wait.
            if (require_completion) check_cuda(cudaStreamSynchronize(stream), context_);
            return -1.0;
        }
        check_cuda(cudaEventRecord(end_, stream), context_);
        check_cuda(cudaEventSynchronize(end_), context_);
        float milliseconds = 0.0F;
        check_cuda(
            cudaEventElapsedTime(&milliseconds, begin_, end_), context_);
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
        if (value < 0.0) return;  // No CUDA event observation in profile-off mode.
        measured = true;
        seconds += value;
    }
};

py::object timing_value(const MeasuredSeconds& value) {
    return value.measured ? py::object(py::float_(value.seconds)) : py::none();
}

std::vector<int> requested_devices(
    const MultiPartitionGraph& graph,
    const std::vector<int>& requested) {
    const gpu_easygraph::runtime::RuntimeStatus status =
        gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running) {
        throw std::runtime_error(
            "EGMGPU native runtime must be started before partitioned SSSP");
    }
    std::vector<int> devices(requested);
    if (devices.empty()) {
        devices.reserve(graph.partitions().size());
        for (std::size_t rank = 0; rank < graph.partitions().size(); ++rank) {
            devices.push_back(graph.partitions()[rank].device_id);
        }
    }
    std::sort(devices.begin(), devices.end());
    if (std::adjacent_find(devices.begin(), devices.end()) != devices.end()) {
        throw std::invalid_argument("partitioned SSSP devices must be distinct");
    }
    if (devices.size() != graph.partitions().size()) {
        throw std::invalid_argument(
            "partitioned SSSP requires one device for every partition");
    }
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        if (!std::binary_search(
                status.device_ids.begin(), status.device_ids.end(),
                devices[rank])) {
            throw std::out_of_range(
                "partitioned SSSP device is outside the active runtime");
        }
    }
    std::vector<int> manifest_devices;
    for (std::size_t rank = 0; rank < graph.partitions().size(); ++rank) {
        manifest_devices.push_back(graph.partitions()[rank].device_id);
    }
    std::sort(manifest_devices.begin(), manifest_devices.end());
    if (manifest_devices != devices) {
        throw std::invalid_argument(
            "partitioned SSSP devices must match the manifest device set");
    }
    return devices;
}

std::vector<SsspHostPartition> host_partitions(
    const MultiPartitionGraph& graph) {
    std::vector<SsspHostPartition> result(graph.partitions().size());
    for (std::size_t rank = 0; rank < graph.partitions().size(); ++rank) {
        const PartitionCSR& source = graph.partitions()[rank];
        source.require_narrow_offsets("sssp");
        SsspHostPartition& target = result[rank];
        target.partition_id = source.partition_id;
        target.device_id = source.device_id;
        target.owned_vertex_count = static_cast<std::size_t>(
            source.owned_vertex_count);
        target.offsets.assign(source.offsets.begin(), source.offsets.end());
        target.indices.assign(source.indices.begin(), source.indices.end());
        target.local_to_global.assign(
            source.local_to_global.begin(), source.local_to_global.end());
        target.owners.assign(source.owners.begin(), source.owners.end());
        if (source.weight_type == easygraph::egmgpu::PartitionWeightType::kFloat64) {
            target.weights = source.weights_float64;
        } else {
            target.weights.assign(
                source.weights_float32.begin(), source.weights_float32.end());
        }
    }
    return result;
}

SsspPartitionUpload make_upload(
    const MultiPartitionGraph& graph,
    const PartitionedSsspPlan& plan,
    const std::vector<int>& devices,
    std::size_t rank,
    const std::string& weight_identity,
    SsspUploadBuildStats* stats) {
    const auto descriptor_begin = SteadyClock::now();
    const SsspHostPartition& partition = plan.partitions[rank];
    const std::size_t rank_count = plan.partitions.size();
    SsspPartitionUpload upload;
    upload.partition_id = partition.partition_id;
    upload.device_id = partition.device_id;
    upload.graph_identity = gpu_easygraph::runtime::stable_signature64(
        graph.graph_id());
    upload.graph_identity_text = graph.graph_id();
    upload.generation = graph.generation();
    upload.device_group = devices;
    upload.view_signature = graph.view_signature() + ":" +
        graph.weight_signature() + ":sssp-delta-owner-v1:weight_encoding=" + sssp_weight_encoding_name(plan.weight_encoding) + ":wire-v1=" + sssp_wire_encoding_name(plan.wire_encoding);
    upload.owned_vertex_count = partition.owned_vertex_count;
    const auto copy_begin = SteadyClock::now();
    upload.offsets = partition.offsets;
    upload.indices = partition.indices;
    upload.weight_encoding = plan.weight_encoding;
    upload.wire_encoding = plan.wire_encoding;
    upload.wire_distance_bound = plan.wire_distance_bound;
    if (plan.weight_encoding == SsspWeightEncoding::U8Exact) upload.weights_u8 = partition.weights_u8;
    else upload.weights = partition.weights;
    upload.owners = partition.owners;
    const double copy_seconds = elapsed_seconds(copy_begin, SteadyClock::now());
    const PartitionCSR& manifest_partition = graph.partitions().at(rank);
    const char* packet_setting = std::getenv("EGGPU_NATIVE_SSSP_PACKET");
    const std::string packet_policy = packet_setting ? packet_setting : "exact_base";
    if (packet_policy != "exact_base" && packet_policy != "off")
        throw std::invalid_argument("EGGPU_NATIVE_SSSP_PACKET must be exact_base or off");
    if (packet_policy == "exact_base" && plan.weight_encoding == SsspWeightEncoding::F64 &&
        manifest_partition.weight_type == easygraph::egmgpu::PartitionWeightType::kFloat64 &&
        manifest_partition.native_weight_patch) {
        const auto& patch = *manifest_partition.native_weight_patch;
        auto packet = std::make_shared<gpu_easygraph::runtime::ExactWeightPatch>();
        packet->base_sha256 = patch.base_weight_sha256;
        packet->target_sha256 = patch.target_weight_sha256;
        packet->positions = patch.positions;
        packet->values = patch.values;
        upload.native_weight_patch = packet;
    }
    upload.offsets_sha256 = manifest_partition.offsets_sha256;
    upload.indices_sha256 = manifest_partition.indices_sha256;
    // F64 weights are copied unchanged from the immutable loader-owned
    // array. Reuse its byte identity under the same loader policy as offsets
    // and indices. Float32 normalization changes bytes and requires a new F64
    // digest. Weight type, not vector emptiness, also covers empty F64 owners.
    const auto digest_begin = SteadyClock::now();
    const std::size_t weight_bytes = upload.weights.size() * sizeof(double);
    if (plan.weight_encoding == SsspWeightEncoding::U8Exact) {
        // Deterministic exact transform identity: verified immutable source
        // digest plus encoding/version in compatibility. This is not a claim
        // that the source digest hashes the newly encoded bytes, and needs no
        // second content scan or hash of all edges.
        upload.weights_sha256 = manifest_partition.weights_sha256;
        stats->weight_manifest_bytes += upload.weights_u8.size();
        ++stats->weight_manifest_descriptors;
    } else if (weight_identity == "manifest_f64" &&
        manifest_partition.weight_type == easygraph::egmgpu::PartitionWeightType::kFloat64) {
        upload.weights_sha256 = manifest_partition.weights_sha256;
        stats->weight_manifest_bytes += weight_bytes;
        ++stats->weight_manifest_descriptors;
    } else {
        upload.weights_sha256 = easygraph::egmgpu::sha256_vector(upload.weights);
        stats->weight_hash_bytes += weight_bytes;
        ++stats->weight_hash_descriptors;
        if (manifest_partition.weight_type == easygraph::egmgpu::PartitionWeightType::kFloat32)
            ++stats->weight_normalized_float32_descriptors;
    }
    const double digest_seconds = elapsed_seconds(digest_begin, SteadyClock::now());
    upload.owners_sha256 = manifest_partition.owners_sha256;
    upload.topology_compatibility =
        easygraph::egmgpu::partition_topology_compatibility(
            graph, partition.partition_id, devices);
    upload.weight_compatibility =
        upload.topology_compatibility + ":outgoing=" + upload.offsets_sha256 + ":" + upload.indices_sha256 +
        ":weight-key=" + graph.weight_key() + ":encoding-v1=" + sssp_weight_encoding_name(plan.weight_encoding);
    upload.ownership_compatibility =
        easygraph::egmgpu::partition_ownership_compatibility(
            graph, partition.partition_id, devices);
    const auto routes_begin = SteadyClock::now();
    upload.halo_route_slots.assign(partition.local_to_global.size(), -1);
    upload.outgoing_route_offsets.push_back(0U);
    for (std::size_t destination = 0; destination < rank_count;
         ++destination) {
        const SsspOwnerRoute& route = plan.route(rank, destination);
        for (std::size_t index = 0; index < route.capacity(); ++index) {
            const int halo_local = route.source_halo_local_ids[index];
            if (halo_local < 0 ||
                static_cast<std::size_t>(halo_local) >=
                    upload.halo_route_slots.size() ||
                upload.halo_route_slots[static_cast<std::size_t>(halo_local)] !=
                    -1) {
                throw std::logic_error("invalid SSSP halo route");
            }
            upload.halo_route_slots[static_cast<std::size_t>(halo_local)] =
                static_cast<int>(upload.route_owner_local_ids.size());
            upload.route_destination_ranks.push_back(
                static_cast<int>(destination));
            upload.route_owner_local_ids.push_back(
                route.destination_owner_local_ids[index]);
        }
        upload.outgoing_route_offsets.push_back(
            upload.route_owner_local_ids.size());
    }
    upload.incoming_route_offsets.push_back(0U);
    for (std::size_t source = 0; source < rank_count; ++source) {
        upload.incoming_route_offsets.push_back(
            upload.incoming_route_offsets.back() +
            plan.route(source, rank).capacity());
    }
    const auto descriptor_end = SteadyClock::now();
    const double routes_seconds = elapsed_seconds(routes_begin, descriptor_end);
    stats->copy_seconds += copy_seconds;
    stats->weight_digest_seconds += digest_seconds;
    stats->routes_seconds += routes_seconds;
    stats->other_seconds += elapsed_seconds(descriptor_begin, descriptor_end) -
        copy_seconds - digest_seconds - routes_seconds;
    ++stats->descriptors;
    return upload;
}

template <typename T>
std::size_t array_capacity_bytes(const std::vector<T>& values) {
    return values.capacity() * sizeof(T);
}

std::size_t upload_array_capacity_bytes(const SsspPartitionUpload& upload) {
    return array_capacity_bytes(upload.device_group) +
        array_capacity_bytes(upload.offsets) + array_capacity_bytes(upload.indices) +
        array_capacity_bytes(upload.weights) + array_capacity_bytes(upload.weights_u8) + array_capacity_bytes(upload.owners) +
        array_capacity_bytes(upload.halo_route_slots) +
        array_capacity_bytes(upload.route_destination_ranks) +
        array_capacity_bytes(upload.route_owner_local_ids) +
        array_capacity_bytes(upload.outgoing_route_offsets) +
        array_capacity_bytes(upload.incoming_route_offsets);
}

std::size_t plan_array_capacity_bytes(const PartitionedSsspPlan& plan) {
    std::size_t bytes = array_capacity_bytes(plan.owner_by_global);
    for (const auto& part : plan.partitions) {
        bytes += array_capacity_bytes(part.offsets) + array_capacity_bytes(part.indices) +
            array_capacity_bytes(part.weights) + array_capacity_bytes(part.weights_u8) + array_capacity_bytes(part.local_to_global) +
            array_capacity_bytes(part.owners);
    }
    for (const auto& route : plan.routes) {
        bytes += array_capacity_bytes(route.source_halo_local_ids) +
            array_capacity_bytes(route.destination_owner_local_ids);
    }
    return bytes;
}

struct SsspUploadHolder {
    std::shared_ptr<const SsspPartitionUpload> prepared;
    SsspPartitionUpload owned;
    const SsspPartitionUpload& get() const { return prepared ? *prepared : owned; }
};

py::dict descriptor_stats(const SsspUploadBuildStats& stats) {
    py::dict out;
    out["copy_seconds"] = stats.copy_seconds;
    out["weight_digest_seconds"] = stats.weight_digest_seconds;
    out["routes_seconds"] = stats.routes_seconds;
    out["other_seconds"] = stats.other_seconds;
    out["descriptors"] = stats.descriptors;
    out["weight_hash_bytes"] = stats.weight_hash_bytes;
    out["weight_hash_descriptors"] = stats.weight_hash_descriptors;
    out["weight_manifest_bytes"] = stats.weight_manifest_bytes;
    out["weight_manifest_descriptors"] = stats.weight_manifest_descriptors;
    out["weight_normalized_float32_descriptors"] = stats.weight_normalized_float32_descriptors;
    return out;
}

struct RankTiming {
    bool profile = true;
    bool single_partition_ordering = false;
    std::uint64_t barrier_count = 0;
    MeasuredSeconds kernel;
    MeasuredSeconds pack;
    MeasuredSeconds transfer;
    MeasuredSeconds wait;
    MeasuredSeconds communication;
    MeasuredSeconds output_gather;
};

struct RankState {
    gpu_easygraph::runtime::SingleOwnerSsspStats single_owner_stats;
    std::shared_ptr<ResidentSsspPartitionView> view;
    std::shared_ptr<SsspPartitionWorkspace> workspace;
    std::vector<int> send_counts;
    std::vector<double> owned_distances;
    std::uint64_t reachable_owned = 0;
    double host_output_assembly_seconds = 0.0;
    int host_output_team_actual = 0;
    std::vector<SsspOutputThreadObservation> host_output_threads;
    RankTiming timing;
    SsspKernelPolicy kernel_policy;
    bool direct_frontier = false;
    gpu_easygraph::partitioned::SsspFrontierEpoch frontier_epoch;
    std::uint64_t direct_frontier_rounds = 0;
    std::uint64_t frontier_epoch_clears = 0;
    std::uint64_t direct_frontier_vertices = 0;
    std::size_t direct_frontier_peak = 0;
    std::size_t current_count;
    int next_count;
    std::uint64_t local_minimum_bucket;
    std::uint64_t logical_messages;
    std::uint64_t control_transfer_bytes;
    std::size_t resident_bytes;
    double state_upload_seconds;
    double worker_task_capture_wall_seconds = 0.0;
    std::uint64_t heavy_rounds_executed = 0;
    std::uint64_t heavy_rounds_skipped = 0;
    bool device_view_cache_hit;
    SharedArrayCallStats shared_arrays;

    RankState()
        : current_count(0),
          next_count(0),
          local_minimum_bucket(gpu_easygraph::partitioned::kSsspNoBucket),
          logical_messages(0),
          control_transfer_bytes(0),
          resident_bytes(0),
          state_upload_seconds(0.0),
          device_view_cache_hit(false) {}
};

#include "egmgpu_sssp_pinned.inc"

struct SharedSsspExecution {
    SharedSsspExecution(const PartitionedSsspPlan& plan,
        const std::shared_ptr<SsspPinnedHostState>& pinned,
        const SsspKernelPolicy& kernels, SsspExchangeSync sync_policy)
        : barrier(plan.partitions.size()), ranks(plan.partitions.size()),
          exchange_sync(sync_policy),
          global_bucket(0), global_frontier(0), bucket_rounds(0), light_rounds(0),
          converged(false), phases(plan.partitions.size(), "not_started"),
          pinned_owner(pinned), host_payloads(pinned->host_payloads),
          host_counts(pinned->host_counts), host_minimum_buckets(pinned->host_minimum_buckets),
          output_buffers(pinned->output_buffers) {
        for (auto& state : ranks) state.kernel_policy = kernels;
    }

    CooperativeBarrier barrier;
    const bool control_fusion_requested = gpu_easygraph::runtime::algorithm_candidate_flag(
        "EGGPU_ALGORITHM_CONTROL_FUSION");
    bool control_fusion_used = false;
    std::uint64_t control_reduction_calls = 0;
    const bool skip_empty_heavy_requested = gpu_easygraph::runtime::algorithm_candidate_flag(
        "EGGPU_SSSP_SKIP_EMPTY_HEAVY");
    bool heavy_phase_observed = false;
    bool skip_empty_heavy_eligible = false;
    bool skip_empty_heavy_used = false;
    std::vector<RankState> ranks;
    const SsspExchangeSync exchange_sync;
    std::uint64_t global_bucket;
    std::size_t global_frontier;
    std::uint64_t bucket_rounds;
    std::uint64_t light_rounds;
    bool converged;
    std::vector<std::string> phases;
    std::shared_ptr<SsspPinnedHostState> pinned_owner;
    std::vector<std::shared_ptr<PinnedHostBuffer> > host_payloads;
    std::vector<std::shared_ptr<PinnedHostBuffer> > host_counts;
    std::vector<std::shared_ptr<PinnedHostBuffer> > host_minimum_buckets;
    std::vector<std::shared_ptr<PinnedHostBuffer> > output_buffers;
    // The loader/plan gives each global vertex exactly one owner. Each worker
    // writes only that owner's entries; this storage belongs to one query.
    // Borrowed only until the caller-owned request storage is published.
    double* worker_distances = nullptr;
    bool single_owner_execution = false;
    bool worker_output = false;
    bool output_team16 = false;
    int output_team_requested = 1;
    double host_output_allocation_seconds = 0.0;
};

void assemble_sssp_owner_output(const SsspHostPartition& partition,
    SharedSsspExecution& shared, RankState& state, const double* observed) {
    const auto begin = SteadyClock::now();
    if (shared.worker_output) {
        double* result = shared.worker_distances;
        const int requested = shared.output_team_requested;
        state.host_output_threads.resize(static_cast<std::size_t>(requested));
        std::uint64_t reachable = 0;
        if (shared.output_team16) {
#if EASYGRAPH_USE_OPENMP
            // Each local ID has one writer and each global ID has one owner.
            // The implicit parallel-region join precedes worker completion and
            // thus the caller's pinned_lease unlock. No Python/CUDA calls here.
#pragma omp parallel num_threads(requested) reduction(+:reachable)
            {
                const int thread = omp_get_thread_num();
                state.host_output_threads[thread].capture();
#pragma omp single
                state.host_output_team_actual = omp_get_num_threads();
#pragma omp for schedule(static)
                for (std::int64_t local = 0;
                     local < static_cast<std::int64_t>(partition.owned_vertex_count); ++local) {
                    const double value = observed[local];
                    result[partition.local_to_global[local]] = value;
                    reachable += std::isfinite(value);
                }
            }
#endif
        } else {
            state.host_output_team_actual = 1;
            state.host_output_threads[0].capture();
            for (std::size_t local = 0; local < partition.owned_vertex_count; ++local) {
                const double value = observed[local];
                result[partition.local_to_global[local]] = value;
                reachable += std::isfinite(value);
            }
        }
        if (state.host_output_team_actual != requested)
            throw std::runtime_error("SSSP output team actual width differs from requested budget");
        for (const auto& thread : state.host_output_threads)
            if (thread.error || thread.cpu < 0)
                throw std::runtime_error("SSSP output team CPU observation failed");
        state.reachable_owned = reachable;
    } else {
        state.owned_distances.assign(observed, observed + partition.owned_vertex_count);
    }
    state.host_output_assembly_seconds = elapsed_seconds(begin, SteadyClock::now());
}


void timed_barrier(
    const std::shared_ptr<SharedSsspExecution>& shared,
    RankTiming* timing,
    bool measure) {
    const SteadyClock::time_point begin = SteadyClock::now();
    shared->barrier.arrive_and_wait();
    ++timing->barrier_count;
    if (measure) timing->wait.add(elapsed_seconds(begin, SteadyClock::now()));
}

// Replaces only a control reduction's arrival + rank0 + publication pair.
// GPU producer/transfer completion and payload-reuse barriers are untouched.
template <typename Reduction>
void timed_control_reduction(
    const std::shared_ptr<SharedSsspExecution>& shared,
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
    ++timing->barrier_count;
    if (measure) timing->wait.add(elapsed_seconds(begin, SteadyClock::now()));
}

void reset_relaxation_round(
    const SsspPartitionDeviceView& view,
    SsspPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankTiming* timing, RankState* direct_state = nullptr) {
    CudaEventTimer timer("measure SSSP relaxation reset", timing->profile);
    timer.start(streams.compute());
    // Query-local epoch starts at zero, even when workspace storage is reused.
    // The old reset completion also orders these writes before any producer.
    bool reset_next_flags = true;
    if (direct_state) {
        reset_next_flags = direct_state->frontier_epoch.advance();
        if (reset_next_flags && view.owned_vertex_count != 0) ++direct_state->frontier_epoch_clears;
        ++direct_state->direct_frontier_rounds;
        check_cuda(cudaMemsetAsync(workspace.frontier_count(), 0, sizeof(int),
            streams.compute()), "clear SSSP direct frontier count");
    }
    if (reset_next_flags && view.owned_vertex_count != 0) {
        check_cuda(
            cudaMemsetAsync(
                workspace.next_flags(), 0,
                view.owned_vertex_count * sizeof(int), streams.compute()),
            "clear SSSP next flags");
    }
    const std::size_t outgoing = view.outgoing_route_offsets.back();
    gpu_easygraph::partitioned::reset_sssp_remote_candidates(
        workspace.remote_candidates(), outgoing, streams.compute());
    const std::size_t ranks = view.outgoing_route_offsets.size() - 1U;
    check_cuda(
        cudaMemsetAsync(
            workspace.send_counts(), 0, ranks * sizeof(int),
            streams.compute()),
        "clear SSSP send counts");
    timing->kernel.add(timer.finish(streams.compute(), !timing->single_partition_ordering));
}

gpu_easygraph::partitioned::SsspNextFrontier next_frontier_descriptor(
    SsspPartitionWorkspace& workspace, const RankState& state, bool light) {
    return state.direct_frontier && light
        ? gpu_easygraph::partitioned::SsspNextFrontier(
            workspace.next_frontier(), workspace.frontier_count(), state.frontier_epoch.value)
        : gpu_easygraph::partitioned::SsspNextFrontier();
}

void relax_edges(
    const SsspPartitionDeviceView& view,
    SsspPartitionWorkspace& workspace,
    RankState* state,
    std::uint64_t bucket,
    double delta,
    bool light,
    WorkerStreams& streams) {
    if (state->current_count == 0) return;
    CudaEventTimer timer(
        light ? "measure SSSP light relaxation"
              : "measure SSSP heavy relaxation", state->timing.profile);
    timer.start(streams.compute());
    gpu_easygraph::partitioned::relax_sssp_delta_edges(
        view.offsets, view.indices, view.weights, view.owners,
        view.halo_route_slots, workspace.distances(), workspace.settled(),
        workspace.current_frontier(), state->current_count,
        workspace.next_flags(), workspace.r_flags(),
        workspace.remote_candidates(), view.owned_vertex_count,
        view.partition_id, bucket, delta, light, streams.compute(), state->kernel_policy.relax, state->kernel_policy.atomic_min, view.weight_encoding,
        next_frontier_descriptor(workspace, *state, light));
    state->timing.kernel.add(timer.finish(streams.compute(), !state->timing.single_partition_ordering));
}

void compact_routes(
    const SsspPartitionDeviceView& view,
    SsspPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankTiming* timing,
    SsspControlPolicy control) {
    bool launched = false;
    CudaEventTimer timer("measure SSSP remote min compaction", timing->profile);
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
        gpu_easygraph::partitioned::compact_sssp_relaxation_route(
            workspace.remote_candidates() + begin,
            view.route_owner_local_ids + begin,
            workspace.send_values(destination), capacity,
            workspace.send_counts() + destination, streams.pack(), control, view.wire_encoding, view.wire_distance_bound);
    }
    if (launched) timing->pack.add(timer.finish(streams.pack()));
}

void copy_send_counts(
    const std::shared_ptr<SharedSsspExecution>& shared,
    std::size_t rank,
    SsspPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankState* state) {
    const std::size_t rank_count = state->send_counts.size();
    CudaEventTimer timer("measure SSSP route-count transfer", state->timing.profile);
    timer.start(streams.transfer());
    check_cuda(
        cudaMemcpyAsync(
            shared->host_counts[rank]->data(), workspace.send_counts(),
            rank_count * sizeof(int), cudaMemcpyDeviceToHost,
            streams.transfer()),
        "copy SSSP route counts to host");
    state->timing.transfer.add(timer.finish(streams.transfer()));
    state->control_transfer_bytes += rank_count * sizeof(int);
    const int* observed = static_cast<const int*>(
        shared->host_counts[rank]->data());
    for (std::size_t destination = 0; destination < rank_count;
         ++destination) {
        const std::size_t capacity =
            state->view->view().outgoing_route_offsets[destination + 1U] -
            state->view->view().outgoing_route_offsets[destination];
        if (observed[destination] < 0 ||
            static_cast<std::size_t>(observed[destination]) > capacity) {
            throw std::runtime_error(
                "partitioned SSSP produced an invalid route count");
        }
        state->send_counts[destination] = observed[destination];
        state->logical_messages += observed[destination];
    }
}

void stage_outgoing(
    const std::shared_ptr<SharedSsspExecution>& shared,
    const PartitionedSsspPlan& plan,
    std::size_t rank,
    SsspPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankTiming* timing) {
    bool copied = false;
    CudaEventTimer timer("measure SSSP D2H staging", timing->profile);
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
                static_cast<std::size_t>(count) * sssp_wire_bytes(plan.wire_encoding),
                cudaMemcpyDeviceToHost, streams.transfer()),
            "stage SSSP relaxations to host");
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer()));
}

void peer_transfer(
    const std::shared_ptr<SharedSsspExecution>& shared,
    const PartitionedSsspPlan& plan,
    std::size_t rank,
    int worker_device,
    WorkerStreams& streams,
    RankTiming* timing) {
    bool copied = false;
    CudaEventTimer timer("measure SSSP peer transfer", timing->profile);
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
            static_cast<std::size_t>(count) * sssp_wire_bytes(plan.wire_encoding),
            streams.transfer());
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer()));
}

void stage_incoming(
    const std::shared_ptr<SharedSsspExecution>& shared,
    const PartitionedSsspPlan& plan,
    std::size_t rank,
    SsspPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankTiming* timing) {
    bool copied = false;
    CudaEventTimer timer("measure SSSP H2D staging", timing->profile);
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
                static_cast<std::size_t>(count) * sssp_wire_bytes(plan.wire_encoding),
                cudaMemcpyHostToDevice, streams.transfer()),
            "stage SSSP relaxations to owner device");
    }
    if (copied) timing->transfer.add(timer.finish(streams.transfer()));
}

void apply_owner_relaxations(
    const std::shared_ptr<SharedSsspExecution>& shared,
    const PartitionedSsspPlan& plan,
    std::size_t rank,
    const SsspPartitionDeviceView& view,
    SsspPartitionWorkspace& workspace,
    std::uint64_t bucket,
    double delta,
    bool enqueue,
    WorkerStreams& streams,
    RankTiming* timing) {
    bool launched = false;
    CudaEventTimer timer("measure SSSP owner min-combine", timing->profile);
    for (std::size_t source = 0; source < plan.partitions.size(); ++source) {
        const int count = shared->ranks[source].send_counts[rank];
        if (count == 0) continue;
        if (!launched) {
            timer.start(streams.compute());
            launched = true;
        }
        gpu_easygraph::partitioned::apply_sssp_owner_relaxations(
            workspace.receive_values(source),
            static_cast<std::size_t>(count), workspace.distances(),
            workspace.settled(), workspace.next_flags(),
            view.owned_vertex_count, bucket, delta, enqueue,
            streams.compute(), shared->ranks[rank].kernel_policy.atomic_min, view.wire_encoding,
            next_frontier_descriptor(workspace, shared->ranks[rank], enqueue));
    }
    if (launched) timing->kernel.add(timer.finish(streams.compute()));
}

void compact_vertex_flags(
    const int* flags,
    int* frontier,
    std::size_t owned,
    SsspPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankTiming* timing,
    SsspControlPolicy control) {
    CudaEventTimer timer("measure SSSP vertex compaction", timing->profile);
    timer.start(streams.compute());
    check_cuda(
        cudaMemsetAsync(
            workspace.frontier_count(), 0, sizeof(int), streams.compute()),
        "clear SSSP frontier count");
    gpu_easygraph::partitioned::compact_sssp_vertex_flags(
        flags, frontier, owned, workspace.frontier_count(),
        streams.compute(), control);
    timing->kernel.add(timer.finish(streams.compute(), !timing->single_partition_ordering));
}

void copy_frontier_count(
    const std::shared_ptr<SharedSsspExecution>& shared,
    std::size_t rank,
    SsspPartitionWorkspace& workspace,
    WorkerStreams& streams,
    RankState* state) {
    int* destination = static_cast<int*>(shared->host_counts[rank]->data()) +
        state->send_counts.size();
    // Ordered after reset/relax/compact, with one wait before host consumption.
    const cudaStream_t count_stream = state->timing.single_partition_ordering
        ? streams.compute() : streams.transfer();
    CudaEventTimer timer("measure SSSP frontier-count transfer", state->timing.profile);
    timer.start(count_stream);
    check_cuda(
        cudaMemcpyAsync(
            destination, workspace.frontier_count(), sizeof(int),
            cudaMemcpyDeviceToHost, count_stream),
        "copy SSSP frontier count to host");
    state->timing.transfer.add(timer.finish(count_stream));
    state->control_transfer_bytes += sizeof(int);
    if (*destination < 0 ||
        static_cast<std::size_t>(*destination) >
            state->view->view().owned_vertex_count) {
        throw std::runtime_error(
            "partitioned SSSP produced an invalid frontier count");
    }
    state->next_count = *destination;
}

void exchange_relaxations(
    const std::shared_ptr<SharedSsspExecution>& shared,
    const PartitionedSsspPlan& plan,
    std::size_t rank,
    int worker_device,
    CommunicationBackend backend,
    const SsspPartitionDeviceView& view,
    SsspPartitionWorkspace& workspace,
    std::uint64_t bucket,
    double delta,
    bool enqueue,
    WorkerStreams& streams,
    RankState* state,
    bool measure_wait) {
    // Every relaxation is already local when the graph has one owner.
    if (state->timing.single_partition_ordering) return;
    const SteadyClock::time_point communication_begin = SteadyClock::now();
    compact_routes(view, workspace, streams, &state->timing, state->kernel_policy.control);
    copy_send_counts(shared, rank, workspace, streams, state);
    if (backend == CommunicationBackend::HostStaging) {
        stage_outgoing(
            shared, plan, rank, workspace, streams, &state->timing);
    }
    timed_barrier(shared, &state->timing, measure_wait);
    if (backend == CommunicationBackend::PeerToPeer) {
        peer_transfer(
            shared, plan, rank, worker_device, streams, &state->timing);
    } else if (backend == CommunicationBackend::HostStaging) {
        stage_incoming(
            shared, plan, rank, workspace, streams, &state->timing);
    }
    if (shared->exchange_sync == SsspExchangeSync::Legacy) {
        timed_barrier(shared, &state->timing, measure_wait);
    }
    apply_owner_relaxations(
        shared, plan, rank, view, workspace, bucket, delta, enqueue,
        streams, &state->timing);
    if (shared->exchange_sync == SsspExchangeSync::Legacy) {
        timed_barrier(shared, &state->timing, measure_wait);
    }
    // Trim keeps the send-ready barrier above. Each receiver's transfer and
    // apply have completed locally; neither writes another rank's send buffer.
    // The caller's light frontier-publication barrier or heavy settle barrier
    // completes this round before any sender reuses its counts/payload.
    if (measure_wait) {
        state->timing.communication.add(elapsed_seconds(
            communication_begin, SteadyClock::now()));
    }
}

void find_local_minimum_bucket(
    const std::shared_ptr<SharedSsspExecution>& shared,
    std::size_t rank,
    const SsspPartitionDeviceView& view,
    SsspPartitionWorkspace& workspace,
    double delta,
    WorkerStreams& streams,
    RankState* state) {
    CudaEventTimer kernel_timer("measure SSSP minimum bucket", state->timing.profile);
    kernel_timer.start(streams.compute());
    check_cuda(
        cudaMemsetAsync(
            workspace.minimum_bucket(), 0xff, sizeof(std::uint64_t),
            streams.compute()),
        "clear SSSP minimum bucket");
    gpu_easygraph::partitioned::find_sssp_minimum_bucket(
        workspace.distances(), workspace.settled(),
        workspace.minimum_bucket(), view.owned_vertex_count, delta,
        streams.compute(), state->kernel_policy.control);
    state->timing.kernel.add(kernel_timer.finish(streams.compute(), !state->timing.single_partition_ordering));
    const cudaStream_t count_stream = state->timing.single_partition_ordering
        ? streams.compute() : streams.transfer();
    CudaEventTimer transfer_timer("measure SSSP bucket transfer", state->timing.profile);
    transfer_timer.start(count_stream);
    check_cuda(
        cudaMemcpyAsync(
            shared->host_minimum_buckets[rank]->data(),
            workspace.minimum_bucket(), sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost, count_stream),
        "copy SSSP minimum bucket to host");
    state->timing.transfer.add(transfer_timer.finish(count_stream));
    state->control_transfer_bytes += sizeof(std::uint64_t);
    state->local_minimum_bucket = *static_cast<const std::uint64_t*>(
        shared->host_minimum_buckets[rank]->data());
}

#include "egmgpu_partitioned_sssp_serial.inc"

struct NativeSsspResult {
    std::vector<double> distances;
    double merge_seconds;
    std::uint64_t resident_bytes;
    double state_upload_seconds;
    bool device_view_cache_hit;
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

void aggregate_rank_time(
    MeasuredSeconds* aggregate, const MeasuredSeconds& rank, bool serial) {
    if (serial) {
        if (rank.measured) aggregate->add(rank.seconds);
    } else {
        update_critical_path(aggregate, rank);
    }
}

NativeSsspResult merge_result(
    const PartitionedSsspPlan& plan,
    const std::shared_ptr<SharedSsspExecution>& shared,
    bool serial) {
    const SteadyClock::time_point begin = SteadyClock::now();
    NativeSsspResult result;
    result.resident_bytes = 0;
    result.state_upload_seconds = 0.0;
    result.device_view_cache_hit = true;
    std::vector<std::vector<double> > owned;
    if (!shared->worker_output) owned.resize(plan.partitions.size());
    for (std::size_t rank = 0; rank < plan.partitions.size(); ++rank) {
        const RankState& state = shared->ranks[rank];
        if (!shared->worker_output && state.owned_distances.size() !=
            plan.partitions[rank].owned_vertex_count) {
            throw std::runtime_error(
                "partitioned SSSP owner result has the wrong length");
        }
        if (!shared->worker_output) owned[rank] = state.owned_distances;
        result.resident_bytes += state.resident_bytes;
        result.state_upload_seconds = serial
            ? result.state_upload_seconds + state.state_upload_seconds
            : std::max(result.state_upload_seconds, state.state_upload_seconds);
        result.device_view_cache_hit = result.device_view_cache_hit &&
            state.device_view_cache_hit;
        aggregate_rank_time(&result.kernel, state.timing.kernel, serial);
        aggregate_rank_time(&result.pack, state.timing.pack, serial);
        aggregate_rank_time(&result.transfer, state.timing.transfer, serial);
        aggregate_rank_time(&result.wait, state.timing.wait, serial);
        aggregate_rank_time(&result.communication, state.timing.communication, serial);
        aggregate_rank_time(&result.output_gather, state.timing.output_gather, serial);
    }
    if (!shared->worker_output) {
        result.distances =
            gpu_easygraph::runtime::merge_partitioned_sssp_owner_distances(plan, owned);
    }
    result.merge_seconds = elapsed_seconds(begin, SteadyClock::now());
    return result;
}

py::dict run_partitioned_sssp_impl(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    std::uint64_t source,
    double delta,
    std::uint64_t max_bucket_rounds,
    const std::vector<int>& requested_device_ids,
    bool serial_partitions,
    bool dense_output,
    const py::object& requested_policy) {
    const auto native_begin = SteadyClock::now();
    const SsspExecutionPolicy policy = resolve_sssp_execution_policy(requested_policy);
    const bool profile = policy.profile;
    const bool default_round_limit = max_bucket_rounds == 0;
    const double requested_delta = delta;
    const std::uint64_t requested_bucket_round_limit = max_bucket_rounds;
    SsspKernelPolicy kernel_policy;
    kernel_policy.relax = policy.relax_policy == "warp" ? SsspRelaxPolicy::Warp : SsspRelaxPolicy::Thread;
    kernel_policy.control = policy.control_policy == "warp" ? SsspControlPolicy::Warp : SsspControlPolicy::Atomic;
    kernel_policy.atomic_min = policy.atomic_min == "native_u64" ? SsspAtomicMinPolicy::NativeU64 : SsspAtomicMinPolicy::Cas;
    const SsspExchangeSync exchange_sync = policy.exchange_sync == "trim" ? SsspExchangeSync::Trim : SsspExchangeSync::Legacy;
    const bool pinned_reuse = policy.pinned_reuse;
    const std::string weight_identity = policy.weight_identity;
    const SsspWeightEncoding weight_encoding = policy.weight_encoding == "u8_exact" ? SsspWeightEncoding::U8Exact : SsspWeightEncoding::F64;
    const SsspWireEncoding wire_encoding = policy.wire_encoding == "u32_exact" ? SsspWireEncoding::U32Exact : SsspWireEncoding::F64;
    const bool prepared_uploads = policy.prepared_uploads;
    const bool allow_undirected = policy.allow_undirected;
    const bool worker_output = policy.host_assembly == "worker";
    const bool output_team16 = policy.output_team == "team16";
    const std::string pinned_placement = policy.pinned_placement;
    if (!graph) throw std::invalid_argument("partitioned SSSP graph is null");
    for (const auto& part : graph->partitions()) part.require_narrow_offsets("sssp");
    if (!std::isfinite(delta) || delta <= 0.0) {
        throw std::invalid_argument("partitioned SSSP delta must be positive");
    }
    validate_sssp_view(*graph, allow_undirected);
    std::vector<int> devices;
    if (serial_partitions) {
        const auto status = gpu_easygraph::runtime::runtime_manager().status();
        if (!status.running || requested_device_ids.size() != 1U ||
            !std::binary_search(status.device_ids.begin(), status.device_ids.end(),
                requested_device_ids.front())) {
            throw std::invalid_argument("serial SSSP requires one active runtime device");
        }
        devices = requested_device_ids;
    } else {
        devices = requested_devices(*graph, requested_device_ids);
    }
    if (output_team16 && (devices.empty() || 16U % devices.size() != 0U))
        throw std::invalid_argument("SSSP team16 requires a GPU worker count dividing 16");
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

    SsspUploadBuildStats descriptor_this_call;
    const SteadyClock::time_point preparation_begin = SteadyClock::now();
    const char* certification_setting = std::getenv("EGGPU_SSSP_WEIGHT_CERTIFICATION");
    const std::string weight_certification = certification_setting ? certification_setting : "full";
    if (weight_certification != "full" && weight_certification != "native_delta" && weight_certification != "adaptive")
        throw std::invalid_argument("EGGPU_SSSP_WEIGHT_CERTIFICATION requires full, native_delta or adaptive");
    std::uint64_t delta_maximum_density_ppm = 0;
    std::string delta_calibration_sha256;
    std::string certification_variant = weight_certification;
    if (weight_certification == "adaptive") {
        const char* threshold = std::getenv("EGGPU_SSSP_DELTA_MAX_DENSITY_PPM");
        const char* calibration = std::getenv("EGGPU_SSSP_DELTA_CALIBRATION_SHA256");
        if (!threshold || !*threshold || !calibration)
            throw std::invalid_argument("Adaptive SSSP certification requires a frozen density threshold and calibration SHA");
        for (const char* digit = threshold; *digit; ++digit)
            if (*digit < '0' || *digit > '9')
                throw std::invalid_argument("SSSP delta density must be an integer in [0,1000000] ppm");
        errno = 0;
        char* end = nullptr;
        const auto parsed = std::strtoull(threshold, &end, 10);
        if (errno || !end || *end || parsed > 1000000)
            throw std::invalid_argument("SSSP delta density must be an integer in [0,1000000] ppm");
        delta_maximum_density_ppm = static_cast<std::uint64_t>(parsed);
        delta_calibration_sha256 = calibration;
        if (delta_calibration_sha256.size() != 64 ||
            delta_calibration_sha256.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::invalid_argument("SSSP delta calibration identity must be a SHA-256 hex digest");
        // Keep this policy identity stable when consecutive commits choose
        // different routes, so the exact direct parent remains eligible.
        certification_variant += ":density-v1:ppm=" + std::to_string(delta_maximum_density_ppm) +
            ":calibration=" + delta_calibration_sha256;
    }
    const std::string plan_variant =
        std::string(prepared_uploads ? "prepared-v4" : "per-query-v4") +
        (serial_partitions ? ":serial:" : ":parallel:") + weight_identity + ":weight_encoding=" + sssp_weight_encoding_name(weight_encoding) + ":wire-v1=" + sssp_wire_encoding_name(wire_encoding) + ":weight_certification=" + certification_variant;
    const gpu_easygraph::runtime::PartitionedHostPlanCacheKey plan_key =
        gpu_easygraph::runtime::make_partitioned_host_plan_cache_key(
            gpu_easygraph::runtime::PartitionedHostPlanAlgorithm::Sssp,
            graph->graph_id(),
            graph->generation(),
            graph->view_signature(),
            graph->topology_signature(),
            graph->weighted(),
            graph->weight_signature(),
            graph->vertex_count(),
            graph->adjacency_count(),
            devices, plan_variant);
    gpu_easygraph::runtime::PartitionedHostPlanCache<
        PartitionedSsspPlan>::AcquireResult plan_acquisition;
    bool parent_topology_lookup_attempted = false;
    double parent_topology_lookup_seconds = 0.0;
    {
        py::gil_scoped_release release;
        auto& host_cache = gpu_easygraph::runtime::partitioned_sssp_host_plan_cache();
        std::shared_ptr<const PartitionedSsspPlan> parent_plan;
        // First-stage scope: the native producer proves unchanged topology.
        // Imported snapshots and other representations retain the full builder.
        if (!serial_partitions && prepared_uploads &&
            weight_encoding == SsspWeightEncoding::F64 && wire_encoding == SsspWireEncoding::F64 &&
            graph->snapshot_origin() == "native_memory") {
            parent_topology_lookup_attempted = true;
            const auto lookup_begin = SteadyClock::now();
            std::vector<std::pair<int, int> > partition_devices;
            for (const auto& part : graph->partitions())
                partition_devices.emplace_back(part.partition_id, part.device_id);
            const auto& commit = graph->native_commit_profile();
            parent_plan = host_cache.find_completed(
                [&](const gpu_easygraph::runtime::PartitionedHostPlanCacheKey& cached_key,
                    const PartitionedSsspPlan& cached_plan) {
                    return gpu_easygraph::runtime::matches_sssp_direct_parent(
                        cached_key, cached_plan, plan_key, commit.base_graph_id,
                        commit.base_view_signature, partition_devices);
                });
            parent_topology_lookup_seconds = elapsed_seconds(lookup_begin, SteadyClock::now());
        }
        plan_acquisition =
            host_cache.acquire(
                plan_key,
                [&graph, &devices, &descriptor_this_call, &parent_plan, prepared_uploads, serial_partitions, weight_identity, weight_encoding, wire_encoding, weight_certification, delta_maximum_density_ppm]() {
                    std::shared_ptr<PartitionedSsspPlan> prepared;
                    if (parent_plan) {
                        const auto candidate_positions = graph->native_commit_profile().changed_positions;
                        const bool selected_delta = weight_certification == "native_delta" ||
                            (weight_certification == "adaptive" && gpu_easygraph::runtime::sssp_native_delta_density_eligible(
                                candidate_positions, graph->adjacency_count(), delta_maximum_density_ppm));
                        // Select before constructing/copying any delta packet.
                        if (selected_delta) {
                            std::vector<gpu_easygraph::runtime::SsspNativeWeightDelta> deltas(graph->partitions().size());
                            std::vector<std::string> parent_hashes, current_hashes;
                            if (parent_plan->prepared_uploads.size() != deltas.size())
                                throw std::logic_error("Native SSSP certification parent descriptor count mismatch");
                            for (std::size_t rank = 0; rank < graph->partitions().size(); ++rank) {
                                const auto& part = graph->partitions()[rank];
                                if (!parent_plan->prepared_uploads[rank] ||
                                    part.weight_type != easygraph::egmgpu::PartitionWeightType::kFloat64)
                                    throw std::logic_error("Native SSSP certification requires exact F64 parent descriptors");
                                parent_hashes.push_back(parent_plan->prepared_uploads[rank]->weights_sha256);
                                current_hashes.push_back(part.weights_sha256);
                                if (!part.native_weight_patch) continue;
                                const auto& patch = *part.native_weight_patch;
                                deltas[rank].base_weight_sha256 = patch.base_weight_sha256;
                                deltas[rank].target_weight_sha256 = patch.target_weight_sha256;
                                deltas[rank].positions = patch.positions;
                                deltas[rank].values = patch.values;
                            }
                            gpu_easygraph::runtime::validate_sssp_native_delta_identity(
                                parent_hashes, current_hashes, deltas, graph->native_commit_profile().changed_positions);
                            prepared = std::make_shared<PartitionedSsspPlan>(gpu_easygraph::runtime::
                                rebuild_sssp_f64_from_native_delta(*parent_plan, deltas));
                        } else {
                            std::vector<std::vector<double> > current_weights;
                            current_weights.reserve(graph->partitions().size());
                            for (const auto& part : graph->partitions())
                                current_weights.push_back(part.weights_float64);
                            prepared = std::make_shared<PartitionedSsspPlan>(gpu_easygraph::runtime::
                                rebuild_sssp_f64_from_parent(*parent_plan, std::move(current_weights)));
                        }
                        prepared->weight_certification_candidate_positions = candidate_positions;
                        prepared->weight_certification_selected_route = selected_delta ? "native_delta" : "full";
                        prepared->weight_certification_selection_reason = weight_certification == "adaptive" ?
                            (selected_delta ? "at_or_below_density_threshold" : "above_density_threshold") : "forced_policy";
                    } else {
                        prepared = std::make_shared<PartitionedSsspPlan>(gpu_easygraph::runtime::
                            build_partitioned_sssp_plan(host_partitions(*graph),
                                graph->vertex_count(), weight_encoding, wire_encoding));
                    }
                    prepared->host_plan_array_capacity_bytes = plan_array_capacity_bytes(*prepared);
                    if (prepared_uploads) {
                        prepared->prepared_uploads.reserve(prepared->partitions.size());
                        for (std::size_t rank = 0; rank < prepared->partitions.size(); ++rank) {
                            SsspPartitionUpload upload = make_upload(*graph, *prepared,
                                devices, rank, weight_identity, &descriptor_this_call);
                            if (serial_partitions) {
                                upload.device_id = devices.front();
                                upload.view_signature += ":same-device-serial-v2";
                            }
                            prepared->prepared_upload_array_capacity_bytes +=
                                upload_array_capacity_bytes(upload);
                            prepared->prepared_uploads.push_back(
                                std::make_shared<const SsspPartitionUpload>(std::move(upload)));
                        }
                        prepared->prepared_upload_build = descriptor_this_call;
                    }
                    return std::shared_ptr<const PartitionedSsspPlan>(prepared);
                });
        // End the temporary parent lease before any GPU task is constructed.
        // The new plan owns its arrays and no descriptor/packet from the parent.
        parent_plan.reset();
    }
    const std::shared_ptr<const PartitionedSsspPlan> plan =
        plan_acquisition.plan;
    const auto preparation_end = SteadyClock::now();
    const double state_preparation_seconds = elapsed_seconds(
        preparation_begin, preparation_end);
    const std::size_t source_rank = plan->owner_rank(source);
    const std::size_t source_local = plan->owner_local(source);
    if (graph->vertex_count() == std::numeric_limits<std::uint64_t>::max())
        throw std::invalid_argument("SSSP vertex count exceeds safe round-limit arithmetic");
    if (max_bucket_rounds == 0) {
        max_bucket_rounds = graph->vertex_count() + 1U;
    }
    const std::uint64_t max_light_rounds = graph->vertex_count() + 1U;

    const auto pinned_begin = SteadyClock::now();
    const bool staging = backend == CommunicationBackend::HostStaging;
    std::string pinned_key = std::string("sssp:pinned:v4:") +
        (serial_partitions ? "serial:" : "parallel:");
    for (int device : devices) pinned_key += std::to_string(device) + ":";
    pinned_key += "p" + std::to_string(plan->partitions.size()) +
        (staging ? ":staging:" : ":device:") + pinned_placement + ":wire-v1=" + sssp_wire_encoding_name(wire_encoding);
    std::shared_ptr<SsspPinnedHostState> pinned;
    std::unique_lock<std::mutex> pinned_lease;
    bool pinned_cache_hit = false;
    double pinned_lease_wait_seconds = 0.0;
    std::size_t pinned_new_allocations = 0;
    std::size_t pinned_retained_bytes = 0;
    {
        py::gil_scoped_release release;
        if (pinned_reuse) {
            pinned = std::static_pointer_cast<SsspPinnedHostState>(
                gpu_easygraph::runtime::runtime_manager().acquire_resident_resource(
                    pinned_key, [](){ return std::shared_ptr<void>(new SsspPinnedHostState()); },
                    &pinned_cache_hit));
        } else pinned.reset(new SsspPinnedHostState());
        const auto wait_begin = SteadyClock::now();
        pinned_lease = std::unique_lock<std::mutex>(pinned->lease_mutex);
        pinned_lease_wait_seconds = elapsed_seconds(wait_begin, SteadyClock::now());
        const std::size_t before = pinned->allocation_count;
        pinned->ensure(*plan, staging, pinned_placement == "numa_local",
            serial_partitions ? devices.front() : -1);
        pinned_new_allocations = pinned->allocation_count - before;
        pinned_retained_bytes = pinned->retained_bytes;
    }
    const auto shared_allocation_begin = SteadyClock::now();
    const double pinned_preparation_seconds = elapsed_seconds(pinned_begin, shared_allocation_begin);
    const std::shared_ptr<SharedSsspExecution> shared(new SharedSsspExecution(*plan, pinned, kernel_policy, exchange_sync));
    const bool single_partition_ordering = single_partition_candidate_enabled() &&
        !serial_partitions && devices.size() == 1U && plan->partitions.size() == 1U;
    for (auto& state : shared->ranks)
        state.timing.single_partition_ordering = single_partition_ordering;
    const auto& single = plan->partitions.front();
    shared->single_owner_execution = gpu_easygraph::runtime::single_owner_sssp_eligible(
        policy.single_owner_execution == "arena_v3_if_eligible", serial_partitions,
        devices.size(), plan->partitions.size(), single.owned_vertex_count,
        single.local_to_global.size(), single.indices.size(), plan->route_capacity,
        weight_encoding == SsspWeightEncoding::F64, wire_encoding == SsspWireEncoding::F64,
        profile, default_round_limit, delta == 16.0 && policy.atomic_min == "native_u64" &&
        policy.relax_policy == "warp" && policy.control_policy == "warp");
    const bool direct_frontier_requested = sssp_direct_frontier_requested();
    const bool direct_frontier_used = direct_frontier_requested &&
        !serial_partitions && !shared->single_owner_execution;
    for (auto& state : shared->ranks) state.direct_frontier = direct_frontier_used;
    // Decide from the current immutable plan, after owner capability selection.
    // Existing IEEE owner execution ignores fixed delta and retains its gate.
    const auto numeric_domain = gpu_easygraph::runtime::select_sssp_numeric_domain(
        plan->vertex_count, plan->maximum_integer_weight, plan->noninteger_weight_count,
        plan->weight_statistics_available, requested_delta);
    if (!shared->single_owner_execution) delta = numeric_domain.effective_delta;
    shared->heavy_phase_observed = !serial_partitions && !shared->single_owner_execution;
    shared->skip_empty_heavy_eligible = gpu_easygraph::runtime::sssp_skip_empty_heavy_eligible(
        shared->heavy_phase_observed, plan->weight_statistics_available,
        plan->maximum_integer_weight, numeric_domain);
    shared->skip_empty_heavy_used = shared->skip_empty_heavy_requested && shared->skip_empty_heavy_eligible;
    shared->control_fusion_used = shared->control_fusion_requested &&
        !serial_partitions && !shared->single_owner_execution && devices.size() > 1U;
    const bool numpy_output_requested = easygraph::egmgpu::sssp_numpy_output_requested();
    const bool numpy_output_used = numpy_output_requested && dense_output && worker_output;
    std::unique_ptr<easygraph::egmgpu::SsspHostOutput> worker_output_storage;
    shared->worker_output = worker_output;
    shared->output_team16 = output_team16;
    // devices.size()==1 for same-device serial partitions, irrespective of P.
    shared->output_team_requested = output_team16 ? 16 / static_cast<int>(devices.size()) : 1;
    if (worker_output) {
        const auto begin = SteadyClock::now();
        // Intentionally no initialization pass. Every owned entry is written
        // after the worker's completed D2H, including unreachable +inf values.
        worker_output_storage.reset(new easygraph::egmgpu::SsspHostOutput(graph->vertex_count(), numpy_output_used));
        shared->worker_distances = worker_output_storage->data();
        shared->host_output_allocation_seconds = elapsed_seconds(begin, SteadyClock::now());
    }
    const auto shared_allocation_end = SteadyClock::now();
    double upload_descriptor_seconds = 0.0;
    double task_capture_seconds = 0.0;
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.reserve(devices.size());
    if (serial_partitions) {
        std::vector<SsspUploadHolder> uploads;
        uploads.reserve(plan->partitions.size());
        for (std::size_t rank = 0; rank < plan->partitions.size(); ++rank) {
            const auto begin = SteadyClock::now();
            SsspUploadHolder holder;
            if (prepared_uploads) holder.prepared = plan->prepared_uploads.at(rank);
            else {
                holder.owned = make_upload(*graph, *plan, devices, rank, weight_identity, &descriptor_this_call);
                holder.owned.device_id = devices.front();
                holder.owned.view_signature += ":same-device-serial-v2";
            }
            upload_descriptor_seconds += elapsed_seconds(begin, SteadyClock::now());
            uploads.push_back(std::move(holder));
        }
        const auto capture_begin = SteadyClock::now();
        tasks.push_back([shared, plan, uploads, source_rank, source_local,
            delta, max_bucket_rounds, max_light_rounds, profile](int worker_device) {
            // Legacy copies owned descriptors; prepared mode copies only the
            // shared const references, while retaining plan and rank identity.
            std::vector<SsspUploadHolder> worker_uploads;
            worker_uploads.reserve(uploads.size());
            for (std::size_t rank = 0; rank < uploads.size(); ++rank) {
                const auto begin = SteadyClock::now();
                worker_uploads.push_back(uploads[rank]);
                shared->ranks[rank].worker_task_capture_wall_seconds =
                    elapsed_seconds(begin, SteadyClock::now());
            }
            run_serial_sssp_gang(shared, *plan, worker_uploads, source_rank,
                source_local, delta, max_bucket_rounds, max_light_rounds,
                worker_device, profile);
        });
        task_capture_seconds += elapsed_seconds(capture_begin, SteadyClock::now());
    } else {
    std::vector<std::size_t> execution_ranks;
    for (std::size_t index = 0; index < devices.size(); ++index) {
        std::size_t rank = 0;
        while (rank < plan->partitions.size() &&
               plan->partitions[rank].device_id != devices[index]) {
            ++rank;
        }
        if (rank == plan->partitions.size()) {
            throw std::logic_error(
                "partitioned SSSP could not map device to partition");
        }
        execution_ranks.push_back(rank);
    }

    for (std::size_t execution_index = 0;
         execution_index < execution_ranks.size(); ++execution_index) {
        const std::size_t rank = execution_ranks[execution_index];
        const auto upload_descriptor_begin = SteadyClock::now();
        SsspUploadHolder upload_holder;
        if (prepared_uploads) upload_holder.prepared = plan->prepared_uploads.at(rank);
        else upload_holder.owned = make_upload(
            *graph, *plan, devices, rank, weight_identity, &descriptor_this_call);
        const auto capture_begin = SteadyClock::now();
        upload_descriptor_seconds += elapsed_seconds(
            upload_descriptor_begin, capture_begin);
        tasks.push_back(
            [shared, plan, upload_holder, devices, topology, backend, source_rank,
             source_local, delta, max_bucket_rounds, max_light_rounds,
             rank, profile](int worker_device) {
                const auto worker_capture_begin = SteadyClock::now();
                gpu_easygraph::runtime::run_cooperative_rank(
                    shared->barrier,
                    [shared, plan, upload_holder, devices, topology, backend,
                     source_rank, source_local, delta, max_bucket_rounds,
                     max_light_rounds, rank, worker_device, profile,
                     worker_capture_begin]() {
                        const SsspPartitionUpload& upload = upload_holder.get();
                        shared->ranks[rank].worker_task_capture_wall_seconds =
                            elapsed_seconds(worker_capture_begin, SteadyClock::now());
                        shared->ranks[rank].timing.profile = profile;
                        shared->phases[rank] = "worker_start";
                        if (worker_device != upload.device_id) {
                            throw std::runtime_error(
                                "partitioned SSSP ran on wrong worker");
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
                        gpu_easygraph::runtime::PartitionedSsspStateRegistry&
                            registry = gpu_easygraph::runtime::
                                partitioned_sssp_state_registry();
                        state.view = registry.acquire(
                            upload, &state.device_view_cache_hit,
                            &state.shared_arrays);
                        state.workspace = registry.workspace(state.view);
                        state.state_upload_seconds = elapsed_seconds(
                            upload_begin, SteadyClock::now());
                        state.resident_bytes = state.view->resident_bytes() +
                            state.workspace->capacity_bytes();
                        const SsspPartitionDeviceView& view =
                            state.view->view();
                        SsspPartitionWorkspace& workspace = *state.workspace;
                        WorkerStreams& streams =
                            gpu_easygraph::runtime::worker_streams();

                        if (state.timing.single_partition_ordering &&
                            (view.local_vertex_count != view.owned_vertex_count ||
                             view.outgoing_route_offsets.back() != 0U))
                            throw std::logic_error("P1 SSSP requires a complete owner with no remote routes");
                        if (shared->single_owner_execution) {
                            if (view.local_vertex_count != view.owned_vertex_count || view.outgoing_route_offsets.back() != 0U)
                                throw std::logic_error("Single-owner engine cannot consume a halo view");
                            shared->phases[rank] = "single_owner_arena_execution";
                            const auto scratch_begin = SteadyClock::now();
                            workspace.ensure_single_owner_scratch(view.owned_vertex_count);
                            state.state_upload_seconds += elapsed_seconds(scratch_begin, SteadyClock::now());
                            state.resident_bytes = state.view->resident_bytes() + workspace.capacity_bytes();
                            const auto execution_begin = SteadyClock::now();
                            check_cuda(gpu_easygraph::partitioned::run_single_owner_sssp(
                                view.offsets, view.indices, static_cast<const double*>(view.weights),
                                static_cast<int>(view.owned_vertex_count), static_cast<int>(plan->partitions[rank].indices.size()),
                                static_cast<int>(source_local), workspace.single_owner_scratch(),
                                workspace.single_owner_scratch_capacity(), workspace.distances(), streams.compute(),
                                &state.single_owner_stats), "single-owner existing Arena SSSP");
                            state.single_owner_stats.execution_wall_seconds = elapsed_seconds(execution_begin, SteadyClock::now());
                            state.control_transfer_bytes += state.single_owner_stats.control_d2h_bytes + state.single_owner_stats.source_h2d_bytes;
                            shared->converged = true;
                        } else {
                        shared->phases[rank] = "initialize_delta_state";
                        if (view.owned_vertex_count != 0) {
                            CudaEventTimer timer(
                                "measure SSSP initialization", state.timing.profile);
                            timer.start(streams.compute());
                            gpu_easygraph::partitioned::initialize_sssp_delta(
                                workspace.distances(), workspace.settled(),
                                workspace.r_flags(), view.owned_vertex_count,
                                rank == source_rank
                                    ? static_cast<int>(source_local)
                                    : -1,
                                streams.compute());
                            state.timing.kernel.add(
                                timer.finish(streams.compute(), !state.timing.single_partition_ordering));
                        }
                        timed_barrier(
                            shared, &state.timing, devices.size() > 1U);

                        for (std::uint64_t bucket_round = 0;
                             bucket_round < max_bucket_rounds;
                             ++bucket_round) {
                            const std::uint64_t bucket =
                                shared->global_bucket;
                            shared->phases[rank] = "select_delta_bucket";
                            if (view.owned_vertex_count != 0) {
                                check_cuda(
                                    cudaMemsetAsync(
                                        workspace.frontier_count(), 0,
                                        sizeof(int), streams.compute()),
                                    "clear SSSP selected frontier count");
                                check_cuda(
                                    cudaMemsetAsync(
                                        workspace.r_flags(), 0,
                                        view.owned_vertex_count * sizeof(int),
                                        streams.compute()),
                                    "clear SSSP R flags");
                                CudaEventTimer timer(
                                    "measure SSSP bucket selection", state.timing.profile);
                                timer.start(streams.compute());
                                gpu_easygraph::partitioned::
                                    select_sssp_bucket_frontier(
                                        workspace.distances(),
                                        workspace.settled(),
                                        workspace.current_frontier(),
                                        workspace.frontier_count(),
                                        view.owned_vertex_count, bucket, delta,
                                        streams.compute(), state.kernel_policy.control);
                                state.timing.kernel.add(
                                    timer.finish(streams.compute(), !state.timing.single_partition_ordering));
                                copy_frontier_count(
                                    shared, rank, workspace, streams, &state);
                            } else {
                                // No owned vertices implies an empty frontier.
                                // Avoid an unsequenced compute memset / transfer
                                // D2H pair for this host control value.
                                state.next_count = 0;
                            }
                            state.current_count = state.next_count;
                            timed_control_reduction(
                                shared, &state.timing, devices.size() > 1U, rank, [&]() {
                                shared->global_frontier = 0;
                                for (std::size_t peer = 0;
                                     peer < shared->ranks.size(); ++peer) {
                                    shared->global_frontier +=
                                        shared->ranks[peer].current_count;
                                }
                            });
                            if (shared->global_frontier == 0) {
                                throw std::runtime_error(
                                    "selected SSSP bucket is globally empty");
                            }

                            std::uint64_t light_round = 0;
                            while (shared->global_frontier != 0) {
                                if (++light_round > max_light_rounds) {
                                    throw std::runtime_error(
                                        "SSSP light-edge closure exceeded "
                                        "the correctness guard");
                                }
                                shared->phases[rank] =
                                    "light_relaxation_and_precombine";
                                reset_relaxation_round(
                                    view, workspace, streams, &state.timing,
                                    state.direct_frontier ? &state : nullptr);
                                relax_edges(
                                    view, workspace, &state, bucket, delta,
                                    true, streams);
                                exchange_relaxations(
                                    shared, *plan, rank, worker_device,
                                    backend, view, workspace, bucket, delta,
                                    true, streams, &state,
                                    devices.size() > 1U);
                                if (state.direct_frontier) {
                                    // Preserve the old compaction's completion boundary:
                                    // transfer-stream count reads must follow all producers.
                                    // P1 ordering already reads the count on compute.
                                    CudaEventTimer timer("measure SSSP direct frontier publication", state.timing.profile);
                                    timer.start(streams.compute());
                                    state.timing.kernel.add(timer.finish(streams.compute(), !state.timing.single_partition_ordering));
                                } else {
                                    compact_vertex_flags(
                                        workspace.next_flags(),
                                        workspace.next_frontier(),
                                        view.owned_vertex_count, workspace,
                                        streams, &state.timing, state.kernel_policy.control);
                                }
                                copy_frontier_count(
                                    shared, rank, workspace, streams, &state);
                                if (state.direct_frontier) {
                                    state.direct_frontier_vertices += static_cast<std::uint64_t>(state.next_count);
                                    state.direct_frontier_peak = std::max(state.direct_frontier_peak,
                                        static_cast<std::size_t>(state.next_count));
                                }
                                timed_control_reduction(
                                    shared, &state.timing, devices.size() > 1U, rank, [&]() {
                                    shared->global_frontier = 0;
                                    for (std::size_t peer = 0;
                                         peer < shared->ranks.size(); ++peer) {
                                        shared->global_frontier +=
                                            shared->ranks[peer].next_count;
                                    }
                                    ++shared->light_rounds;
                                });
                                state.current_count =
                                    static_cast<std::size_t>(state.next_count);
                                workspace.advance_frontier();
                            }

                            if (shared->skip_empty_heavy_used) {
                                // Current exact maximum <= effective delta proves E_heavy empty.
                                // Keep R/settlement and the subsequent global publication barrier.
                                // All light producers and incoming applies completed before the
                                // light-count reduction; no send buffer is reused early here.
                                shared->phases[rank] = "certified_empty_heavy_elided";
                                ++state.heavy_rounds_skipped;
                            } else {
                                shared->phases[rank] = "heavy_relaxation";
                                compact_vertex_flags(
                                    workspace.r_flags(),
                                    workspace.current_frontier(),
                                    view.owned_vertex_count, workspace, streams,
                                    &state.timing, state.kernel_policy.control);
                                copy_frontier_count(
                                    shared, rank, workspace, streams, &state);
                                state.current_count = state.next_count;
                                reset_relaxation_round(
                                    view, workspace, streams, &state.timing);
                                relax_edges(
                                    view, workspace, &state, bucket, delta, false,
                                    streams);
                                exchange_relaxations(
                                    shared, *plan, rank, worker_device, backend,
                                    view, workspace, bucket, delta, false, streams,
                                    &state, devices.size() > 1U);
                                ++state.heavy_rounds_executed;
                            }
                            if (view.owned_vertex_count != 0) {
                                CudaEventTimer timer(
                                    "measure SSSP settle bucket", state.timing.profile);
                                timer.start(streams.compute());
                                gpu_easygraph::partitioned::
                                    settle_sssp_r_vertices(
                                        workspace.r_flags(),
                                        workspace.settled(),
                                        view.owned_vertex_count,
                                        streams.compute());
                                state.timing.kernel.add(
                                    timer.finish(streams.compute(), !state.timing.single_partition_ordering));
                            }
                            timed_barrier(
                                shared, &state.timing, devices.size() > 1U);

                            shared->phases[rank] = "global_empty_bucket";
                            find_local_minimum_bucket(
                                shared, rank, view, workspace, delta, streams,
                                &state);
                            timed_control_reduction(
                                shared, &state.timing, devices.size() > 1U, rank, [&]() {
                                shared->global_bucket =
                                    gpu_easygraph::partitioned::kSsspNoBucket;
                                for (std::size_t peer = 0;
                                     peer < shared->ranks.size(); ++peer) {
                                    shared->global_bucket = std::min(
                                        shared->global_bucket,
                                        shared->ranks[peer].
                                            local_minimum_bucket);
                                }
                                ++shared->bucket_rounds;
                                shared->converged = shared->global_bucket ==
                                    gpu_easygraph::partitioned::kSsspNoBucket;
                            });
                            if (shared->converged) break;
                        }
                        if (!shared->converged) {
                            throw std::runtime_error(
                                "partitioned SSSP exceeded max_bucket_rounds");
                        }

                        }  // Original partitioned execution branch.
                        if (view.owned_vertex_count != 0) {
                            shared->phases[rank] = "output_gather";
                            CudaEventTimer timer(
                                "measure SSSP output gather", state.timing.profile);
                            timer.start(streams.transfer());
                            check_cuda(
                                cudaMemcpyAsync(
                                    shared->output_buffers[rank]->data(),
                                    workspace.distances(),
                                    view.owned_vertex_count * sizeof(double),
                                    cudaMemcpyDeviceToHost,
                                    streams.transfer()),
                                "gather partitioned SSSP distances");
                            state.timing.output_gather.add(
                                timer.finish(streams.transfer()));
                            const double* observed =
                                static_cast<const double*>(
                                    shared->output_buffers[rank]->data());
                            assemble_sssp_owner_output(plan->partitions[rank],
                                *shared, state, observed);
                        }
                        shared->phases[rank] = "complete";
                    });
            });
        task_capture_seconds += elapsed_seconds(capture_begin, SteadyClock::now());
    }
    }

    const SteadyClock::time_point device_begin = SteadyClock::now();
    {
        py::gil_scoped_release release;
        try {
            gpu_easygraph::runtime::runtime_manager().execute_gang(
                devices, std::move(tasks));
        } catch (const std::exception& error) {
            std::string context("partitioned SSSP worker failure (");
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
    const auto device_end = SteadyClock::now();
    const auto output_placement_snapshot = pinned->output_placement;
    // Worker results are already copied into per-request owning storage.
    // CPU merge and capsule publication never reference this scratch pool.
    pinned_lease.unlock();
    const double device_processing_seconds = elapsed_seconds(
        device_begin, device_end);

    NativeSsspResult native;
    {
        py::gil_scoped_release release;
        native = merge_result(*plan, shared, serial_partitions);
    }
    const SteadyClock::time_point materialization_begin = SteadyClock::now();
    py::object values;
    std::uint64_t reachable = 0;
    if (worker_output) {
        for (const auto& rank : shared->ranks) reachable += rank.reachable_owned;
    }
    if (dense_output) {
        if (worker_output) values = worker_output_storage->publish();
        else {
            for (double distance : native.distances)
                if (std::isfinite(distance)) ++reachable;
            values = sssp_owning_host_array(native.distances);
        }
    } else {
        py::dict eager;
        const double* distances = worker_output ? shared->worker_distances : native.distances.data();
        for (std::size_t vertex = 0; vertex < graph->vertex_count(); ++vertex) {
            if (std::isfinite(distances[vertex])) {
                eager[py::int_(static_cast<std::uint64_t>(vertex))] = py::float_(distances[vertex]);
                if (!worker_output) ++reachable;
            }
        }
        values = eager;
    }
    const auto materialization_end = SteadyClock::now();
    const double python_materialization_seconds = elapsed_seconds(
        materialization_begin, materialization_end);

    py::list per_device;
    py::list per_partition;
    py::list resident_device_bytes;
    py::list resident_partition_bytes;
    std::uint64_t messages = 0;
    std::uint64_t control_bytes = 0;
    std::uint64_t barrier_count_total = 0;
    std::uint64_t shared_array_hits = 0U;
    std::uint64_t shared_array_new_bytes = 0U;
    std::uint64_t shared_array_retained_bytes = 0U;
    for (std::size_t rank = 0; rank < plan->partitions.size(); ++rank) {
        const RankState& state = shared->ranks[rank];
        messages += state.logical_messages;
        control_bytes += state.control_transfer_bytes;
        py::dict item;
        item["partition_id"] = state.view->view().partition_id;
        item["device_id"] = state.view->view().device_id;
        item["owned_vertex_count"] = state.view->view().owned_vertex_count;
        item["local_vertex_count"] = state.view->view().local_vertex_count;
        item["adjacency_count"] = state.view->view().adjacency_count;
        py::dict frontier;
        frontier["used"] = state.direct_frontier;
        frontier["rounds"] = state.direct_frontier_rounds;
        frontier["epoch_clear_count"] = state.frontier_epoch_clears;
        frontier["epoch_clear_vertices"] = state.frontier_epoch_clears * state.view->view().owned_vertex_count;
        frontier["enqueued_vertices"] = state.direct_frontier_vertices;
        frontier["peak_next_count"] = state.direct_frontier_peak;
        frontier["capacity_vertices"] = state.view->view().owned_vertex_count;
        frontier["light_next_full_scan_rounds"] = shared->single_owner_execution || state.direct_frontier || state.view->view().owned_vertex_count == 0U ? 0U : shared->light_rounds;
        frontier["light_next_full_scan_rounds_avoided"] = state.view->view().owned_vertex_count == 0U ? 0U : state.direct_frontier_rounds;
        frontier["light_next_full_scan_vertices_avoided"] = state.direct_frontier_rounds * state.view->view().owned_vertex_count;
        frontier["light_membership_clear_rounds_avoided"] = state.view->view().owned_vertex_count == 0U ? 0U : state.direct_frontier_rounds - state.frontier_epoch_clears;
        frontier["light_membership_clear_vertices_avoided"] = (state.direct_frontier_rounds - state.frontier_epoch_clears) * state.view->view().owned_vertex_count;
        frontier["additional_device_bytes"] = 0U;
        item["sssp_frontier_queue"] = frontier;
        item["weight_encoding"] = sssp_weight_encoding_name(state.view->view().weight_encoding);
        item["weight_device_bytes"] = state.view->view().weight_bytes;
        item["wire_encoding"] = sssp_wire_encoding_name(state.view->view().wire_encoding);
        item["wire_bytes_per_message"] = sssp_wire_bytes(state.view->view().wire_encoding);
        item["wire_send_capacity_bytes"] = state.view->view().outgoing_route_offsets.back() * sssp_wire_bytes(plan->wire_encoding);
        item["wire_receive_capacity_bytes"] = state.view->view().incoming_route_offsets.back() * sssp_wire_bytes(plan->wire_encoding);
        item["worker_task_capture_wall_seconds"] =
            state.worker_task_capture_wall_seconds;
        item["host_output_assembly_seconds"] = state.host_output_assembly_seconds;
        item["host_output_team_actual"] = state.host_output_team_actual;
        item["host_output_bytes"] = state.view->view().owned_vertex_count * sizeof(double);
        py::list output_threads;
        for (const auto& thread : state.host_output_threads) {
            py::dict observation;
            observation["cpu"] = thread.cpu;
            observation["allowed_cpus"] = gpu_easygraph::runtime::gpu_cpu_mask_values(thread.mask);
            output_threads.append(observation);
        }
        item["host_output_threads"] = output_threads;
        item["reachable_owned_vertex_count"] = worker_output ? py::cast(state.reachable_owned) : py::none();
        const auto& placement = output_placement_snapshot[rank];
        item["host_output_pinned_numa_node"] = placement.numa_node;
        item["host_output_pinned_allocation_cpu"] = placement.allocation_cpu;
        item["host_output_pinned_pci_bus_id"] = placement.pci_bus_id;
        item["kernel_seconds"] = timing_value(state.timing.kernel);
        item["pack_seconds"] = timing_value(state.timing.pack);
        item["transfer_seconds"] = timing_value(state.timing.transfer);
        item["wait_seconds"] = timing_value(state.timing.wait);
        item["barrier_count"] = state.timing.barrier_count;
        item["heavy_phase_observed"] = shared->heavy_phase_observed;
        item["heavy_rounds_executed"] = shared->heavy_phase_observed ? py::cast(state.heavy_rounds_executed) : py::none();
        item["heavy_rounds_skipped"] = shared->heavy_phase_observed ? py::cast(state.heavy_rounds_skipped) : py::none();
        barrier_count_total += state.timing.barrier_count;
        item["communication_seconds"] =
            timing_value(state.timing.communication);
        item["output_gather_seconds"] =
            timing_value(state.timing.output_gather);
        item["resident_device_bytes"] =
            static_cast<std::uint64_t>(state.resident_bytes);
        resident_partition_bytes.append(
            static_cast<std::uint64_t>(state.resident_bytes));
        if (!serial_partitions) resident_device_bytes.append(
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
        per_partition.append(item);
        if (!serial_partitions) per_device.append(item);
    }
    if (serial_partitions) {
        py::dict item;
        item["device_id"] = devices.front();
        item["partition_count"] = plan->partitions.size();
        item["kernel_seconds"] = timing_value(native.kernel);
        item["pack_seconds"] = timing_value(native.pack);
        item["transfer_seconds"] = timing_value(native.transfer);
        item["wait_seconds"] = timing_value(native.wait);
        item["barrier_count"] = barrier_count_total;
        item["communication_seconds"] = timing_value(native.communication);
        item["output_gather_seconds"] = timing_value(native.output_gather);
        item["resident_device_bytes"] = native.resident_bytes;
        item["device_view_cache_hit"] = native.device_view_cache_hit;
        item["shared_array_hits"] = shared_array_hits;
        item["shared_array_new_bytes"] = shared_array_new_bytes;
        per_device.append(item);
        resident_device_bytes.append(native.resident_bytes);
    }
    const std::uint64_t logical_bytes = messages * sssp_wire_bytes(plan->wire_encoding);
    const std::uint64_t payload_bytes = backend ==
            CommunicationBackend::HostStaging
        ? logical_bytes * 2U
        : logical_bytes;
    py::dict timing_sources;
    timing_sources["kernel"] = profile ? py::object(py::str("cuda_event")) : py::none();
    timing_sources["pack"] = profile ? py::object(py::str("cuda_event")) : py::none();
    timing_sources["transfer"] = profile ? py::object(py::str("cuda_event")) : py::none();
    timing_sources["output_gather"] = profile ? py::object(py::str("cuda_event")) : py::none();
    timing_sources["wait"] = serial_partitions
        ? "zero_no_multi_rank_barrier" : "steady_clock_barrier";
    timing_sources["communication"] = "steady_clock_exchange_phase";
    timing_sources["native_merge"] = "steady_clock_host";

    py::dict out;
    out[dense_output ? "values_dense" : "values"] = std::move(values);
    out["sssp_native_output_ownership"] = easygraph::egmgpu::sssp_host_output_proof(
        numpy_output_requested, dense_output, worker_output, graph->vertex_count());
    py::dict reduction_proof;
    reduction_proof["version"] = "cooperative_control_completion_v1";
    reduction_proof["flag"] = "EGGPU_ALGORITHM_CONTROL_FUSION";
    reduction_proof["requested"] = shared->control_fusion_requested;
    reduction_proof["used"] = shared->control_fusion_used;
    reduction_proof["eligible"] = !serial_partitions && !shared->single_owner_execution && devices.size() > 1U;
    reduction_proof["reason"] = !shared->control_fusion_requested ? "disabled" :
        (shared->control_fusion_used ? "multi_rank_control_completion" : "single_rank_or_unsupported_route");
    reduction_proof["completed_reductions"] = shared->control_reduction_calls;
    reduction_proof["barriers_elided_per_rank"] = shared->control_reduction_calls;
    reduction_proof["scope"] = "multi-rank selected frontier, light frontier and minimum-bucket decisions; no GPU completion or payload lifetime boundary removed";
    out["algorithm_control_reduction"] = reduction_proof;
    py::dict heavy_proof;
    heavy_proof["version"] = "sssp_current_maximum_empty_heavy_v1";
    heavy_proof["flag"] = "EGGPU_SSSP_SKIP_EMPTY_HEAVY";
    heavy_proof["requested"] = shared->skip_empty_heavy_requested;
    heavy_proof["eligible"] = shared->skip_empty_heavy_eligible;
    heavy_proof["used"] = shared->skip_empty_heavy_used;
    heavy_proof["current_statistics_available"] = plan->weight_statistics_available;
    heavy_proof["current_maximum_weight"] = plan->maximum_integer_weight;
    heavy_proof["effective_delta"] = delta;
    heavy_proof["all_edges_light"] = numeric_domain.all_edges_light;
    heavy_proof["phase_counts_observed"] = shared->heavy_phase_observed;
    heavy_proof["heavy_rounds_executed"] = shared->heavy_phase_observed ? py::cast(shared->ranks.front().heavy_rounds_executed) : py::none();
    heavy_proof["heavy_rounds_skipped"] = shared->heavy_phase_observed ? py::cast(shared->ranks.front().heavy_rounds_skipped) : py::none();
    heavy_proof["scope"] = "partitioned gang route only; skip R compaction/count, reset, heavy traversal and exchange; retain light closure, R settlement and global minimum";
    out["sssp_empty_heavy"] = heavy_proof;
    out["output_representation"] = dense_output ? "owning_fp64_dense" : "eager_dict";
    out["host_assembly_policy"] = worker_output ? "worker" : "caller";
    out["host_output_team_version"] = "stage4_owner_team_v1";
    out["host_output_team_policy"] = output_team16 ? "team16" : "worker1";
    out["host_output_team_requested_per_worker"] = shared->output_team_requested;
    out["host_output_worker_count"] = devices.size();
    out["host_output_thread_budget"] = devices.size() * shared->output_team_requested;
    out["host_output_index_bytes"] = 0U;
    out["host_output_allocation_seconds"] = shared->host_output_allocation_seconds;
    out["host_output_allocation_bytes"] = worker_output ? graph->vertex_count() * sizeof(double) : 0U;
    out["host_output_timing_scope"] = "Owner assembly is nested in gang wall after completed D2H; allocation is nested in host_shared_allocation. Independent query output never aliases reusable pinned scratch.";
    out["host_owning_dense_output_bytes"] = dense_output ? graph->vertex_count() * sizeof(double) : 0U;
    out["pinned_reuse_enabled"] = pinned_reuse;
    out["host_pinned_cache_hit"] = pinned_cache_hit;
    out["host_pinned_new_allocations"] = pinned_new_allocations;
    out["host_pinned_retained_bytes"] = pinned_retained_bytes;
    out["host_pinned_preparation_seconds"] = pinned_preparation_seconds;
    out["host_pinned_lease_wait_seconds"] = pinned_lease_wait_seconds;
    out["host_pinned_pool_key"] = pinned_key;
    out["host_pinned_placement"] = pinned_placement;
    py::dict frontier_policy;
    frontier_policy["version"] = "sssp_light_epoch_queue_v1";
    frontier_policy["requested"] = direct_frontier_requested;
    frontier_policy["used"] = direct_frontier_used;
    frontier_policy["reason"] = !direct_frontier_requested ? "disabled" :
        (serial_partitions ? "serial_execution_unchanged" :
        (shared->single_owner_execution ? "single_owner_engine_unchanged" : "cooperative_delta_epoch_queue"));
    frontier_policy["flag"] = "EGGPU_SSSP_DIRECT_FRONTIER";
    frontier_policy["default"] = "0";
    frontier_policy["additional_device_bytes"] = 0U;
    frontier_policy["scope"] = "light next-frontier only; existing R/bucket/minimum/remote-route scans and heavy reset retained";
    frontier_policy["publication"] = "all producers complete before scalar count read and next-round consumption";
    out["sssp_frontier_queue"] = frontier_policy;
    out["sssp_kernel_version"] = "v5_candidate";
    out["sssp_sync_version"] = "stage4_sync_v1";
    out["exchange_sync_policy"] = exchange_sync == SsspExchangeSync::Trim
        ? "trim" : "legacy";
    out["exchange_sync_scope"] = serial_partitions
        ? "serial_no_cooperative_barriers"
        : "cooperative_exchange_only; caller_round_completion_preserved";
    out["barrier_count"] = shared->ranks.front().timing.barrier_count;
    out["barrier_count_total"] = barrier_count_total;
    out["atomic_min_version"] = "stage4_atomic_min_v1";
    out["atomic_min_policy"] = kernel_policy.atomic_min == SsspAtomicMinPolicy::NativeU64 ? "native_u64" : "cas";
    out["atomic_min_scope"] = "request-local local-distance, remote-candidate, owner-combine; finite nonnegative FP64 and +inf";
    out["relax_policy"] = kernel_policy.relax == SsspRelaxPolicy::Warp ? "warp" : "thread";
    out["control_policy"] = kernel_policy.control == SsspControlPolicy::Warp ? "warp" : "atomic";
    out["execution_policy_version"] = "egmgpu.sssp-execution-policy/v1";
    out["execution_policy_source"] = requested_policy.is_none() ? "environment_compatibility" : "explicit";
    out["execution_policy"] = sssp_execution_policy_document(policy);
    out["kernel_policy_scope"] = "request-local; same partition, FP64, delta and synchronization; control covers select/flags/route/minimum";
    out["device_phase_profiling"] = profile;
    out["single_partition_ordering"] = single_partition_ordering;
    out["profiling_sync_policy"] = single_partition_ordering && !profile
        ? "single_owner_compute_stream_then_control_or_output_completion"
        : "same_finish_boundaries_event_or_stream_sync";
    out["source"] = source;
    out["delta"] = delta;
    out["reachable_vertex_count"] = reachable;
    out["unreachable_vertex_count"] = graph->vertex_count() - reachable;
    out["bucket_rounds"] = shared->bucket_rounds;
    out["light_relaxation_rounds"] = shared->light_rounds;
    out["converged"] = shared->converged;
    out["devices_used"] = devices;
    out["execution_mode"] = serial_partitions
        ? "serial_partitions_one_gpu" : "cooperative";
    out["partition_count"] = plan->partitions.size();
    out["communication_backend"] = serial_partitions && plan->partitions.size() > 1U
        ? "same_device" : std::string(
            gpu_easygraph::runtime::communication_backend_name(backend));
    out["rank_timing_aggregation"] = serial_partitions
        ? "sum of sequential partition spans; nested in gang wall"
        : "max per metric across ranks; not an additive critical path";
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
    out["parent_topology_reuse_version"] = "native_direct_parent_owned_copy_v1";
    out["parent_topology_lookup_attempted"] = parent_topology_lookup_attempted;
    out["parent_topology_lookup_seconds"] = parent_topology_lookup_seconds;
    out["parent_topology_reused_first_prepare"] = plan->parent_topology_reused;
    out["parent_topology_reused_this_call"] = plan_acquisition.builder && plan->parent_topology_reused;
    out["parent_topology_copied_bytes_this_call"] = plan_acquisition.builder ? plan->parent_topology_copied_bytes : 0;
    out["weight_validation_edges_this_call"] = plan_acquisition.builder ? plan->weight_validation_edges : 0;
    out["weight_delta_positions_this_call"] = plan_acquisition.builder ? plan->weight_delta_positions : 0;
    out["weight_delta_certification_used_this_call"] = plan_acquisition.builder && plan->weight_delta_certification_used;
    out["weight_delta_rescan_required_this_call"] = plan_acquisition.builder && plan->weight_delta_rescan_required;
    out["weight_certification_policy"] = weight_certification;
    out["weight_certification_selection_version"] = "native_delta_density_ppm_v1";
    out["weight_certification_selected_route_this_call"] = plan_acquisition.builder ?
        plan->weight_certification_selected_route : "cached_plan";
    out["weight_certification_selection_reason_this_call"] = plan_acquisition.builder ?
        plan->weight_certification_selection_reason : "cached_plan";
    out["weight_certification_candidate_positions_this_call"] = plan_acquisition.builder ?
        plan->weight_certification_candidate_positions : 0;
    out["weight_certification_maximum_density_ppm"] = delta_maximum_density_ppm;
    out["weight_certification_calibration_sha256"] = delta_calibration_sha256;
    out["weight_certification_version"] = "native_complete_delta_exact_statistics_v1";
    out["parent_topology_scope"] =
        "Completed exact-parent plan borrowed from the existing four-entry numeric-plan cache only during preparation; "
        "owned copies of topology/owner/routes; exact native delta weight certification, full scan fallback on lost maximum; fresh current-identity descriptors; "
        "no new cache/index or parent chain. Copied bytes count topology numeric element payload, not allocation capacity or peak.";
    out["host_plan_variant"] = plan_variant;
    out["prepared_uploads_enabled"] = prepared_uploads;
    out["weight_identity_policy"] = weight_identity;
    out["weight_identity_scope"] = "F64 reuses immutable source identity; normalized F32 may hash F64; u8 exact deterministic identity is original source digest plus explicit encoding-v1, not an encoded-payload digest";
    out["weight_encoding_version"] = "stage4_u8_exact_v1";
    out["weight_encoding"] = sssp_weight_encoding_name(plan->weight_encoding);
    out["weight_decode_arithmetic"] = "typed weight load to FP64; FP64 distance state/atomic unchanged; independent certified wire encoding";
    out["wire_encoding_version"] = "stage4_u32_exact_wire_v1";
    out["wire_encoding"] = sssp_wire_encoding_name(plan->wire_encoding);
    out["wire_bytes_per_message"] = sssp_wire_bytes(plan->wire_encoding);
    out["wire_integer_bound_certified"] = plan->wire_integer_bound_certified;
    out["wire_maximum_weight"] = plan->maximum_integer_weight;
    out["wire_distance_bound"] = std::isfinite(plan->wire_distance_bound) ? py::cast(plan->wire_distance_bound) : py::none();
    out["wire_bound_filter_scope"] = "pack candidates above (N-1)*maxW pruned identically for f64/u32 when certified; state arithmetic remains FP64";
    out["logical_relaxation_messages"] = messages;
    out["logical_equivalent_f64_payload_bytes"] = messages * sizeof(SsspRelaxation);
    out["weight_validation_and_encoding_first_prepare_seconds"] = plan->weight_validation_and_encoding_seconds;
    out["weight_validation_and_encoding_this_call_seconds"] = plan_acquisition.builder ? plan->weight_validation_and_encoding_seconds : 0.0;
    out["weight_encoding_time_scope"] = "existing plan edge endpoint/finite validation fused with exact-domain certification and encoding; nested in host-plan cold preparation";
    std::uint64_t actual_weight_bytes = 0;
    for (const auto& rank_state : shared->ranks) actual_weight_bytes += rank_state.view->view().weight_bytes;
    out["weight_device_bytes"] = actual_weight_bytes;
    out["weight_source_fp64_bytes"] = graph->adjacency_count() * sizeof(double);
    out["weight_storage_scope"] = "resident edge weights only; original loader/host-plan FP64 input retained; no claim about total device peak or input-file compression";
    out["weighted_undirected_enabled"] = allow_undirected;
    out["input_view_orientation"] = graph->view_orientation();
    out["host_upload_descriptor_cache_hit"] = prepared_uploads && plan_acquisition.cache_hit;
    out["descriptor_build_this_call"] = descriptor_stats(descriptor_this_call);
    out["descriptor_first_prepare"] = descriptor_stats(plan->prepared_upload_build);
    out["descriptor_build_timing_scope"] = prepared_uploads
        ? "nested in host_plan on builder only; warm hit builds zero descriptors"
        : "nested in per-query upload_descriptor; owned descriptors copied into tasks";
    out["host_plan_array_capacity_bytes"] = plan->host_plan_array_capacity_bytes;
    out["prepared_upload_array_capacity_bytes"] = plan->prepared_upload_array_capacity_bytes;
    out["host_plan_retained_array_capacity_bytes"] = plan->host_plan_array_capacity_bytes +
        plan->prepared_upload_array_capacity_bytes;
    out["host_retained_bytes_scope"] =
        "current cached plan numeric vector capacities including prepared descriptors; "
        "excludes native graph, packet vectors, other plans, transient closures, strings and allocator overhead; not RSS/peak";
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
    out["logical_communication_bytes"] = logical_bytes;
    out["payload_transfer_bytes"] = payload_bytes;
    out["control_transfer_bytes"] = control_bytes;
    out["physical_transfer_bytes"] = payload_bytes + control_bytes;
    out["same_device_transfer_bytes"] = serial_partitions ? payload_bytes : 0U;
    out["inter_device_transfer_bytes"] = serial_partitions ? 0U : payload_bytes;
    out["output_gather_bytes"] = graph->vertex_count() * sizeof(double);
    out["resident_device_bytes"] = resident_device_bytes;
    out["resident_partition_bytes"] = resident_partition_bytes;
    out["aggregate_resident_device_bytes"] = native.resident_bytes;
    out["shared_array_hits"] = shared_array_hits;
    out["shared_array_new_bytes"] = shared_array_new_bytes;
    out["shared_array_retained_bytes"] = shared_array_retained_bytes;
    out["timing_sources"] = timing_sources;
    out["per_device"] = per_device;
    out["per_partition"] = per_partition;

    // Mutually exclusive host wall intervals. Device-event and per-rank
    // fields are nested observations and must not be added to this ledger.
    py::dict host_phases;
    host_phases["entry"] = elapsed_seconds(native_begin, preparation_begin);
    host_phases["host_plan"] = state_preparation_seconds;
    host_phases["request_setup"] = elapsed_seconds(
        preparation_end, pinned_begin);
    host_phases["host_pinned_preparation"] = pinned_preparation_seconds;
    host_phases["host_shared_allocation"] = elapsed_seconds(
        shared_allocation_begin, shared_allocation_end);
    host_phases["upload_descriptor"] = upload_descriptor_seconds;
    host_phases["task_capture"] = task_capture_seconds;
    host_phases["task_setup_and_local_descriptor_cleanup"] = elapsed_seconds(
        shared_allocation_end, device_begin) - upload_descriptor_seconds -
        task_capture_seconds;
    host_phases[serial_partitions ? "serial_worker" : "cooperative_gang"] =
        device_processing_seconds;
    host_phases["host_merge"] = elapsed_seconds(device_end, materialization_begin);
    host_phases["python_materialization"] = python_materialization_seconds;
    const auto& local = shared->ranks.front().single_owner_stats;
    out["single_owner_execution_requested"] = policy.single_owner_execution;
    out["single_owner_execution_eligible"] = shared->single_owner_execution;
    out["single_owner_execution_used"] = local.used;
    out["single_owner_execution_version"] = "borrowed_arena_ieee_v3_native_owner_v1";
    out["single_owner_engine"] = local.used
        ? (local.sparse_frontier ? "corrected_warp_frontier_v2" : "device_delta_ieee_v3")
        : "partitioned_delta";
    std::size_t retained_single_owner_scratch = 0;
    for (const auto& rank : shared->ranks)
        if (rank.workspace) retained_single_owner_scratch += rank.workspace->single_owner_scratch_capacity();
    out["single_owner_scratch_capacity_bytes"] = retained_single_owner_scratch;
    out["single_owner_scratch_used_capacity_bytes"] = local.scratch_capacity_bytes;
    out["single_owner_maximum_degree"] = local.maximum_degree >= 0 ? py::cast(local.maximum_degree) : py::none();
    out["single_owner_rounds"] = local.used ? py::cast(local.rounds) : py::none();
    out["single_owner_queued_rounds"] = local.used ? py::cast(local.queued_rounds) : py::none();
    out["single_owner_control_checkpoints"] = local.used ? py::cast(local.checkpoints) : py::none();
    out["single_owner_control_d2h_bytes"] = local.control_d2h_bytes;
    out["single_owner_source_h2d_bytes"] = local.source_h2d_bytes;
    out["single_owner_kernel_launches"] = local.kernel_launches;
    out["single_owner_stream_synchronizations"] = local.stream_synchronizations;
    out["single_owner_execution_wall_seconds"] = local.used ? py::cast(local.execution_wall_seconds) : py::none();
    out["single_owner_execution_time_scope"] = "worker wall around borrowed-engine call including structure check/control copies/checkpoint completion; excludes scratch allocation and full output gather; nested in cooperative_gang";
    out["single_owner_workspace_scope"] = "Native workspace-owned DeviceAllocation; included in capacity/inventory/release; no private allocator, graph copy, output alias or answer cache";
    out["single_owner_distance_contract"] = "finite nonnegative FP64 inputs; IEEE +inf unreachable and unrepresentable path sums; full fresh distances every call";
    out["requested_delta"] = requested_delta;
    out["effective_delta"] = local.used ? py::none() : py::cast(delta);
    out["numeric_domain_version"] = "nonnegative_fp64_all_light_fallback_v1";
    out["numeric_domain_route"] = local.used ? "single_owner_ieee" : numeric_domain.route;
    out["numeric_domain_reason"] = local.used ? "existing_single_owner_ieee_engine" : numeric_domain.reason;
    out["numeric_maximum_weight"] = plan->maximum_integer_weight;
    out["numeric_vertex_count"] = plan->vertex_count;
    out["numeric_statistics_available"] = plan->weight_statistics_available;
    out["requested_max_bucket_rounds"] = requested_bucket_round_limit;
    out["effective_max_bucket_rounds"] = local.used ? py::none() : py::cast(max_bucket_rounds);
    out["effective_max_light_rounds"] = local.used ? py::none() : py::cast(max_light_rounds);
    out["default_bucket_round_limit"] = default_round_limit;
    out["numeric_noninteger_weight_count"] = plan->noninteger_weight_count;
    out["numeric_integer_bound_used"] = !local.used && numeric_domain.integer_bound_used;
    out["numeric_all_edges_light"] = !local.used && numeric_domain.all_edges_light;
    out["numeric_delta_changed"] = !local.used && delta != requested_delta;
    out["numeric_distance_semantics"] = "nonnegative binary64 rounded additions; +inf for unreachable or unrepresentable path sums";
    py::dict effective;
    effective["engine"] = out["single_owner_engine"];
    effective["weight_encoding"] = policy.weight_encoding;
    effective["checkpoint_iterations"] = local.used ? py::cast(16) : py::none();
    effective["vector_width"] = local.used ? py::cast(local.sparse_frontier ? 4 : 32) : py::none();
    effective["blocks_per_sm"] = local.used ? py::cast(8) : py::none();
    effective["sm_count"] = local.used ? py::cast(local.sm_count) : py::none();
    effective["launch_cap"] = local.used ? py::cast(8 * local.sm_count) : py::none();
    effective["structural_rule"] = "n>=256 && m<=4*n && current_max_out_degree<=128 -> corrected frontier; otherwise IEEE device delta";
    effective["fixed_delta_applied"] = !local.used;
    effective["requested_delta"] = requested_delta;
    effective["effective_delta"] = out["effective_delta"];
    effective["requested_delta_applied"] = !local.used && delta == requested_delta;
    effective["numeric_domain_route"] = out["numeric_domain_route"];
    effective["legacy_kernel_controls_applied"] = !local.used;
    effective["partitioned_ordering_applied"] = !local.used && single_partition_ordering;
    out["execution_policy_effective"] = effective;
    if (local.used) {
        // Requested policy remains separately auditable. These observed fields
        // report the active implementation, never pretend old bucket controls ran.
        out["delta"] = py::none();
        out["bucket_rounds"] = py::none();
        out["light_relaxation_rounds"] = py::none();
        out["atomic_min_policy"] = local.sparse_frontier ? "cas" : "native_u64";
        out["relax_policy"] = local.sparse_frontier ? "logical_warp4" : "logical_warp32";
        out["control_policy"] = "device_resident_checkpoint16";
        out["sssp_kernel_version"] = "existing_arena_ieee_v3_borrowed_native_v1";
        out["sssp_sync_version"] = "single_owner_stream_checkpoint16_v1";
        out["exchange_sync_policy"] = "not_applicable_single_owner";
        out["exchange_sync_scope"] = "No remote owner, halo exchange or cooperative round protocol";
        out["atomic_min_scope"] = "Fresh single-owner IEEE FP64 distances; actual CAS frontier or unsigned IEEE minimum device delta";
        out["weight_decode_arithmetic"] = "Native FP64 weights and distance state; actual minimum primitive recorded in atomic_min_policy";
        out["kernel_policy_scope"] = "Explicit single_owner_execution supersedes requested delta/atomic/relax/control; actual parameters in execution_policy_effective";
        out["profiling_sync_policy"] = "compute_stream_checkpoint16_then_complete_output";
    }

    const auto metadata_end = SteadyClock::now();
    host_phases["metadata"] = elapsed_seconds(materialization_end, metadata_end);
    out["host_phase_wall_seconds"] = host_phases;
    out["native_accounted_seconds"] = elapsed_seconds(native_begin, metadata_end);
    out["host_shared_allocation_seconds"] = host_phases["host_shared_allocation"];
    out["upload_descriptor_seconds"] = upload_descriptor_seconds;
    out["task_capture_seconds"] = task_capture_seconds;
    return out;
}

py::dict run_partitioned_sssp(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    std::uint64_t source,
    double delta,
    std::uint64_t max_bucket_rounds,
    const std::vector<int>& requested_device_ids,
    bool dense_output,
    const py::object& execution_policy) {
    // The wrapper encloses destruction of query-local descriptors, buffers,
    // and distance vectors in the implementation as well as eager output.
    const auto begin = SteadyClock::now();
    py::dict out = run_partitioned_sssp_impl(
        graph, source, delta, max_bucket_rounds, requested_device_ids, false, dense_output, execution_policy);
    const double outer_seconds = elapsed_seconds(begin, SteadyClock::now());
    const double accounted = out["native_accounted_seconds"].cast<double>();
    py::dict host_phases = out["host_phase_wall_seconds"].cast<py::dict>();
    host_phases["native_outer_tail"] = outer_seconds - accounted;
    out["native_outer_seconds"] = outer_seconds;
    out["native_outer_boundary"] =
        "C++ entry through complete owning output and implementation-local cleanup; "
        "final timing metadata and pybind return conversion excluded";
    return out;
}

py::dict run_serial_partitioned_sssp(
    const std::shared_ptr<MultiPartitionGraph>& graph,
    std::uint64_t source,
    double delta,
    std::uint64_t max_bucket_rounds,
    int device,
    bool dense_output,
    const py::object& execution_policy) {
    const auto begin = SteadyClock::now();
    py::dict out = run_partitioned_sssp_impl(
        graph, source, delta, max_bucket_rounds, std::vector<int>{device}, true, dense_output, execution_policy);
    const double outer_seconds = elapsed_seconds(begin, SteadyClock::now());
    const double accounted = out["native_accounted_seconds"].cast<double>();
    py::dict host_phases = out["host_phase_wall_seconds"].cast<py::dict>();
    host_phases["native_outer_tail"] = outer_seconds - accounted;
    out["native_outer_seconds"] = outer_seconds;
    out["native_outer_boundary"] =
        "C++ entry through complete owning output and implementation-local cleanup; "
        "final timing metadata and pybind return conversion excluded";
    return out;
}
#endif

}  // namespace

void bind_egmgpu_partitioned_sssp(py::module_& module) {
#ifdef EASYGRAPH_ENABLE_GPU
    module.def("cpp_egmgpu_sssp_execution_policy", [](const py::object& policy) {
        return sssp_execution_policy_document(resolve_sssp_execution_policy(policy));
    }, py::arg("execution_policy"), "Normalize an explicit SSSP policy without CUDA initialization.");
    module.def(
        "cpp_egmgpu_partitioned_sssp",
        &run_partitioned_sssp,
        py::arg("partitioned_graph"),
        py::arg("source"),
        py::arg("delta") = 1.0,
        py::arg("max_bucket_rounds") = 0,
        py::arg("devices") = std::vector<int>(),
        py::arg("dense_output") = false,
        py::arg("execution_policy") = py::none(),
        "Run nonnegative owner-partitioned delta-stepping SSSP on fixed "
        "GPU workers.");
    module.def(
        "cpp_egmgpu_partitioned_sssp_serial",
        &run_serial_partitioned_sssp,
        py::arg("partitioned_graph"),
        py::arg("source"),
        py::arg("delta") = 1.0,
        py::arg("max_bucket_rounds") = 0,
        py::arg("device") = 0,
        py::arg("dense_output") = false,
        py::arg("execution_policy") = py::none(),
        "Run the same owner partitions and delta-round propagation sequentially "
        "on one fixed GPU worker, returning the complete owning result.");
#else
    (void)module;
#endif
}
