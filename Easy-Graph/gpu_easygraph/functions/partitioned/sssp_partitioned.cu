#include "partitioned/sssp_partitioned.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace gpu_easygraph {
namespace partitioned {
namespace {

__device__ std::uint64_t bucket_of(double distance, double delta) {
    if (!isfinite(distance)) return kSsspNoBucket;
    const double quotient = floor(distance / delta);
    if (quotient >= static_cast<double>(kSsspNoBucket - 1U)) {
        return kSsspNoBucket - 1U;
    }
    return static_cast<std::uint64_t>(quotient);
}

template <bool UseNativeMin>
__device__ bool atomic_min_nonnegative(double* address, double value) {
    unsigned long long* raw = reinterpret_cast<unsigned long long*>(address);
    unsigned long long observed = *raw;
    if (UseNativeMin) {
        // Epoch values only decrease from +inf or +0. A stale larger observed
        // value can cause extra work, but cannot suppress a true improvement.
        // This predicate also rejects +inf and NaN candidates.
        if (!(value < __longlong_as_double(observed))) return false;
        // Nonnegative finite FP64 and +inf have the same unsigned bit order.
        // Canonicalize signed zero before encoding it as an unsigned integer.
        const double canonical = value == 0.0 ? 0.0 : value;
        const unsigned long long incoming =
            static_cast<unsigned long long>(__double_as_longlong(canonical));
        const unsigned long long previous = atomicMin(raw, incoming);
        return incoming < previous;
    }
    while (value < __longlong_as_double(observed)) {
        const unsigned long long previous = atomicCAS(
            raw, observed, __double_as_longlong(value));
        if (previous == observed) return true;
        observed = previous;
    }
    return false;
}

__global__ void initialize_kernel(
    double* distances,
    int* settled,
    int* r_flags,
    std::size_t count,
    int source_local) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        // The owner thread initializes its own source distance as well.
        distances[index] = static_cast<int>(index) == source_local
            ? 0.0 : INFINITY;
        settled[index] = 0;
        r_flags[index] = 0;
    }
}

__global__ void select_bucket_kernel(
    const double* distances,
    const int* settled,
    int* frontier,
    int* frontier_count,
    std::size_t count,
    std::uint64_t bucket,
    double delta) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && settled[index] == 0 &&
        bucket_of(distances[index], delta) == bucket) {
        const int position = atomicAdd(frontier_count, 1);
        frontier[position] = static_cast<int>(index);
    }
}

__global__ void reset_candidates_kernel(double* values, std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = INFINITY;
}

template <bool UseNativeMin, typename Weight, bool DirectQueue>
__global__ void relax_edges_kernel(
    const int* offsets,
    const int* indices,
    const Weight* weights,
    const int* owners,
    const int* halo_route_slots,
    double* distances,
    const int* settled,
    const int* frontier,
    std::size_t frontier_count,
    int* next_flags,
    int* r_flags,
    double* remote_candidates,
    std::size_t owned_count,
    int partition_id,
    std::uint64_t current_bucket,
    double delta,
    bool light, SsspNextFrontier next) {
    const std::size_t frontier_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (frontier_index >= frontier_count) return;
    const int source = frontier[frontier_index];
    if (source < 0 || static_cast<std::size_t>(source) >= owned_count ||
        settled[source] != 0) {
        return;
    }
    if (light) atomicExch(r_flags + source, 1);
    const double source_distance = distances[source];
    for (int edge = offsets[source]; edge < offsets[source + 1]; ++edge) {
        const double weight = static_cast<double>(weights[edge]);
        if ((weight <= delta) != light) continue;
        const double candidate = source_distance + weight;
        if (!isfinite(candidate)) continue;
        const int target = indices[edge];
        if (owners[target] == partition_id) {
            if (target < 0 || static_cast<std::size_t>(target) >= owned_count ||
                settled[target] != 0) {
                continue;
            }
            if (atomic_min_nonnegative<UseNativeMin>(distances + target, candidate) &&
                light && bucket_of(candidate, delta) == current_bucket) {
                mark_sssp_next_vertex<DirectQueue>(target, next_flags, next);
            }
        } else {
            const int slot = halo_route_slots[target];
            if (slot >= 0) {
                atomic_min_nonnegative<UseNativeMin>(remote_candidates + slot, candidate);
            }
        }
    }
}

template <typename Wire>
__device__ Wire encode_sssp_wire(int owner, double distance);
template <>
__device__ gpu_easygraph::runtime::SsspRelaxation encode_sssp_wire<gpu_easygraph::runtime::SsspRelaxation>(int owner,double distance) {
    gpu_easygraph::runtime::SsspRelaxation out;
    out.owner_local_id=owner; out.reserved=0; out.distance=distance; return out;
}
template <>
__device__ gpu_easygraph::runtime::SsspRelaxationU32 encode_sssp_wire<gpu_easygraph::runtime::SsspRelaxationU32>(int owner,double distance) {
    gpu_easygraph::runtime::SsspRelaxationU32 out;
    out.owner_local_id=owner; out.distance=static_cast<std::uint32_t>(distance); return out;
}

template <typename Wire>
__global__ void compact_route_kernel(
    const double* candidates,
    const int* owner_local_ids,
    Wire* compact,
    std::size_t count,
    int* compact_count, double maximum_distance) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && isfinite(candidates[index]) && candidates[index] <= maximum_distance) {
        const int position = atomicAdd(compact_count, 1);
        compact[position] = encode_sssp_wire<Wire>(owner_local_ids[index],candidates[index]);
    }
}

template <bool UseNativeMin, typename Wire, bool DirectQueue>
__global__ void apply_owner_kernel(
    const Wire* candidates,
    std::size_t count,
    double* distances,
    const int* settled,
    int* next_flags,
    std::size_t owned_count,
    std::uint64_t current_bucket,
    double delta,
    bool enqueue_current_bucket, SsspNextFrontier next) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int target = candidates[index].owner_local_id;
    if (target < 0 || static_cast<std::size_t>(target) >= owned_count ||
        settled[target] != 0) {
        return;
    }
    const double candidate = static_cast<double>(candidates[index].distance);
    if (atomic_min_nonnegative<UseNativeMin>(distances + target, candidate) &&
        enqueue_current_bucket &&
        bucket_of(candidate, delta) == current_bucket) {
        mark_sssp_next_vertex<DirectQueue>(target, next_flags, next);
    }
}

__global__ void compact_flags_kernel(
    const int* flags,
    int* frontier,
    std::size_t count,
    int* frontier_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && flags[index] != 0) {
        const int position = atomicAdd(frontier_count, 1);
        frontier[position] = static_cast<int>(index);
    }
}

__global__ void settle_kernel(
    const int* r_flags, int* settled, std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && r_flags[index] != 0) settled[index] = 1;
}

__global__ void minimum_bucket_kernel(
    const double* distances,
    const int* settled,
    std::uint64_t* minimum,
    std::size_t count,
    double delta) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && settled[index] == 0) {
        const std::uint64_t bucket = bucket_of(distances[index], delta);
        if (bucket != kSsspNoBucket) {
            atomicMin(
                reinterpret_cast<unsigned long long*>(minimum),
                static_cast<unsigned long long>(bucket));
        }
    }
}

#include "sssp_warp_kernels.inc"

unsigned int grid_size(std::size_t count) {
    return static_cast<unsigned int>((count + 255U) / 256U);
}

void check_launch(const char* operation) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

}  // namespace

void initialize_sssp_delta(
    double* distances,
    int* settled,
    int* r_flags,
    std::size_t owned_count,
    int source_local,
    cudaStream_t stream) {
    if (owned_count == 0) {
        if (source_local >= 0) {
            throw std::invalid_argument(
                "empty SSSP partition cannot own the source");
        }
        return;
    }
    if (!distances || !settled || !r_flags || source_local < -1 ||
        (source_local >= 0 &&
         static_cast<std::size_t>(source_local) >= owned_count)) {
        throw std::invalid_argument("invalid SSSP initialization state");
    }
    initialize_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
        distances, settled, r_flags, owned_count, source_local);
    check_launch("partitioned SSSP initialization");
}

void select_sssp_bucket_frontier(
    const double* distances,
    const int* settled,
    int* frontier,
    int* frontier_count,
    std::size_t owned_count,
    std::uint64_t bucket,
    double delta,
    cudaStream_t stream,
    SsspControlPolicy policy) {
    if (owned_count == 0) return;
    if (!distances || !settled || !frontier || !frontier_count ||
        bucket == kSsspNoBucket || !std::isfinite(delta) || delta <= 0.0) {
        throw std::invalid_argument("invalid SSSP bucket selection state");
    }
    if (policy == SsspControlPolicy::Warp) {
        select_bucket_warp_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
            distances, settled, frontier, frontier_count, owned_count, bucket, delta);
    } else {
        select_bucket_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
            distances, settled, frontier, frontier_count, owned_count, bucket, delta);
    }
    check_launch("partitioned SSSP bucket selection");
}

void reset_sssp_remote_candidates(
    double* candidates,
    std::size_t count,
    cudaStream_t stream) {
    if (count == 0) return;
    if (!candidates) {
        throw std::invalid_argument("invalid SSSP remote candidate state");
    }
    reset_candidates_kernel<<<grid_size(count), 256, 0, stream>>>(
        candidates, count);
    check_launch("partitioned SSSP remote candidate reset");
}

template <typename Weight, bool DirectQueue>
void relax_sssp_delta_edges_typed(
    const int* offsets,
    const int* indices,
    const Weight* weights,
    const int* owners,
    const int* halo_route_slots,
    double* distances,
    const int* settled,
    const int* frontier,
    std::size_t frontier_count,
    int* next_flags,
    int* r_flags,
    double* remote_candidates,
    std::size_t owned_count,
    int partition_id,
    std::uint64_t current_bucket,
    double delta,
    bool light,
    cudaStream_t stream,
    SsspRelaxPolicy policy,
    SsspAtomicMinPolicy atomic_min_policy, SsspNextFrontier next) {
    if (frontier_count == 0) return;
    if (!offsets || !owners || !halo_route_slots || !distances || !settled ||
        !frontier || !next_flags || !r_flags || owned_count == 0 ||
        frontier_count > owned_count || partition_id < 0 ||
        current_bucket == kSsspNoBucket || !std::isfinite(delta) ||
        delta <= 0.0) {
        throw std::invalid_argument("invalid SSSP relaxation state");
    }
    if (atomic_min_policy == SsspAtomicMinPolicy::NativeU64) {
    if (policy == SsspRelaxPolicy::Warp) {
        const unsigned int blocks = static_cast<unsigned int>((frontier_count + 7U) / 8U);
        relax_edges_warp_kernel<true, Weight, DirectQueue><<<blocks, 256, 0, stream>>>(
            offsets, indices, weights, owners, halo_route_slots, distances,
            settled, frontier, frontier_count, next_flags, r_flags,
            remote_candidates, owned_count, partition_id, current_bucket, delta, light, next);
    } else {
        relax_edges_kernel<true, Weight, DirectQueue><<<grid_size(frontier_count), 256, 0, stream>>>(
            offsets, indices, weights, owners, halo_route_slots, distances,
            settled, frontier, frontier_count, next_flags, r_flags,
            remote_candidates, owned_count, partition_id, current_bucket, delta, light, next);
    }
    } else {
    if (policy == SsspRelaxPolicy::Warp) {
        const unsigned int blocks = static_cast<unsigned int>((frontier_count + 7U) / 8U);
        relax_edges_warp_kernel<false, Weight, DirectQueue><<<blocks, 256, 0, stream>>>(
            offsets, indices, weights, owners, halo_route_slots, distances,
            settled, frontier, frontier_count, next_flags, r_flags,
            remote_candidates, owned_count, partition_id, current_bucket, delta, light, next);
    } else {
        relax_edges_kernel<false, Weight, DirectQueue><<<grid_size(frontier_count), 256, 0, stream>>>(
            offsets, indices, weights, owners, halo_route_slots, distances,
            settled, frontier, frontier_count, next_flags, r_flags,
            remote_candidates, owned_count, partition_id, current_bucket, delta, light, next);
    }
    }
    check_launch(light ? "partitioned SSSP light relaxation"
                       : "partitioned SSSP heavy relaxation");
}

void relax_sssp_delta_edges(
    const int* offsets,
    const int* indices,
    const void* weights,
    const int* owners,
    const int* halo_route_slots,
    double* distances,
    const int* settled,
    const int* frontier,
    std::size_t frontier_count,
    int* next_flags,
    int* r_flags,
    double* remote_candidates,
    std::size_t owned_count,
    int partition_id,
    std::uint64_t current_bucket,
    double delta,
    bool light,
    cudaStream_t stream,
    SsspRelaxPolicy policy,
    SsspAtomicMinPolicy atomic_min_policy,
    gpu_easygraph::runtime::SsspWeightEncoding weight_encoding, SsspNextFrontier next) {
    using gpu_easygraph::runtime::SsspWeightEncoding;
    if (next.epoch < 0 || (next.epoch > 0 && (!light || !next.vertices || !next.count)))
        throw std::invalid_argument("invalid SSSP direct frontier descriptor");
    if (weight_encoding == SsspWeightEncoding::U8Exact) {
        if (next.epoch > 0)
            relax_sssp_delta_edges_typed<std::uint8_t, true>(offsets, indices, static_cast<const std::uint8_t*>(weights), owners, halo_route_slots,
                distances, settled, frontier, frontier_count, next_flags, r_flags, remote_candidates,
                owned_count, partition_id, current_bucket, delta, light, stream, policy, atomic_min_policy, next);
        else
            relax_sssp_delta_edges_typed<std::uint8_t, false>(offsets, indices, static_cast<const std::uint8_t*>(weights), owners, halo_route_slots,
                distances, settled, frontier, frontier_count, next_flags, r_flags, remote_candidates,
                owned_count, partition_id, current_bucket, delta, light, stream, policy, atomic_min_policy, next);
    }
    else if (weight_encoding == SsspWeightEncoding::F64) {
        if (next.epoch > 0)
            relax_sssp_delta_edges_typed<double, true>(offsets, indices, static_cast<const double*>(weights), owners, halo_route_slots,
                distances, settled, frontier, frontier_count, next_flags, r_flags, remote_candidates,
                owned_count, partition_id, current_bucket, delta, light, stream, policy, atomic_min_policy, next);
        else
            relax_sssp_delta_edges_typed<double, false>(offsets, indices, static_cast<const double*>(weights), owners, halo_route_slots,
                distances, settled, frontier, frontier_count, next_flags, r_flags, remote_candidates,
                owned_count, partition_id, current_bucket, delta, light, stream, policy, atomic_min_policy, next);
    }
    else throw std::invalid_argument("unknown SSSP kernel weight encoding");
}

template <typename Wire>
void compact_route_typed(const double* candidates,const int* owner_local_ids,void* compact,
    std::size_t capacity,int* compact_count,cudaStream_t stream,SsspControlPolicy policy,double bound) {
    Wire* output=static_cast<Wire*>(compact);
    if (policy==SsspControlPolicy::Warp)
        compact_route_warp_kernel<Wire><<<grid_size(capacity),256,0,stream>>>(candidates,owner_local_ids,output,capacity,compact_count,bound);
    else
        compact_route_kernel<Wire><<<grid_size(capacity),256,0,stream>>>(candidates,owner_local_ids,output,capacity,compact_count,bound);
}

void compact_sssp_relaxation_route(const double* candidates,const int* owner_local_ids,void* compact,
    std::size_t capacity,int* compact_count,cudaStream_t stream,SsspControlPolicy policy,
    gpu_easygraph::runtime::SsspWireEncoding wire_encoding,double maximum_distance) {
    if (capacity==0) return;
    if (!candidates || !owner_local_ids || !compact || !compact_count)
        throw std::invalid_argument("invalid SSSP route compaction state");
    using namespace gpu_easygraph::runtime;
    if (wire_encoding==SsspWireEncoding::U32Exact)
        compact_route_typed<SsspRelaxationU32>(candidates,owner_local_ids,compact,capacity,compact_count,stream,policy,maximum_distance);
    else
        compact_route_typed<SsspRelaxation>(candidates,owner_local_ids,compact,capacity,compact_count,stream,policy,maximum_distance);
    check_launch("partitioned SSSP route compaction");
}

template <typename Wire, bool DirectQueue>
void apply_owner_typed(const void* candidates,std::size_t count,double* distances,const int* settled,
    int* next_flags,std::size_t owned_count,std::uint64_t current_bucket,double delta,
    bool enqueue_current_bucket,cudaStream_t stream,SsspAtomicMinPolicy atomic_min_policy, SsspNextFrontier next) {
    const Wire* input=static_cast<const Wire*>(candidates);
    if (atomic_min_policy==SsspAtomicMinPolicy::NativeU64)
        apply_owner_kernel<true,Wire,DirectQueue><<<grid_size(count),256,0,stream>>>(input,count,distances,settled,next_flags,owned_count,current_bucket,delta,enqueue_current_bucket,next);
    else
        apply_owner_kernel<false,Wire,DirectQueue><<<grid_size(count),256,0,stream>>>(input,count,distances,settled,next_flags,owned_count,current_bucket,delta,enqueue_current_bucket,next);
}

void apply_sssp_owner_relaxations(const void* candidates,std::size_t count,double* distances,
    const int* settled,int* next_flags,std::size_t owned_count,std::uint64_t current_bucket,
    double delta,bool enqueue_current_bucket,cudaStream_t stream,SsspAtomicMinPolicy atomic_min_policy,
    gpu_easygraph::runtime::SsspWireEncoding wire_encoding, SsspNextFrontier next) {
    if (count==0) return;
    if (!candidates || !distances || !settled || !next_flags || owned_count==0 ||
        current_bucket==kSsspNoBucket || !std::isfinite(delta) || delta<=0.)
        throw std::invalid_argument("invalid SSSP owner combine state");
    if (next.epoch < 0 || (next.epoch > 0 && (!enqueue_current_bucket || !next.vertices || !next.count)))
        throw std::invalid_argument("invalid SSSP direct owner frontier descriptor");
    using namespace gpu_easygraph::runtime;
    if (wire_encoding==SsspWireEncoding::U32Exact) {
        if (next.epoch > 0)
            apply_owner_typed<SsspRelaxationU32,true>(candidates,count,distances,settled,next_flags,owned_count,current_bucket,delta,enqueue_current_bucket,stream,atomic_min_policy,next);
        else
            apply_owner_typed<SsspRelaxationU32,false>(candidates,count,distances,settled,next_flags,owned_count,current_bucket,delta,enqueue_current_bucket,stream,atomic_min_policy,next);
    }
    else {
        if (next.epoch > 0)
            apply_owner_typed<SsspRelaxation,true>(candidates,count,distances,settled,next_flags,owned_count,current_bucket,delta,enqueue_current_bucket,stream,atomic_min_policy,next);
        else
            apply_owner_typed<SsspRelaxation,false>(candidates,count,distances,settled,next_flags,owned_count,current_bucket,delta,enqueue_current_bucket,stream,atomic_min_policy,next);
    }
    check_launch("partitioned SSSP owner min-combine");
}

void compact_sssp_vertex_flags(
    const int* flags,
    int* frontier,
    std::size_t owned_count,
    int* frontier_count,
    cudaStream_t stream,
    SsspControlPolicy policy) {
    if (owned_count == 0) return;
    if (!flags || !frontier || !frontier_count) {
        throw std::invalid_argument("invalid SSSP vertex compaction state");
    }
    if (policy == SsspControlPolicy::Warp) {
        compact_flags_warp_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
            flags, frontier, owned_count, frontier_count);
    } else {
        compact_flags_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
            flags, frontier, owned_count, frontier_count);
    }
    check_launch("partitioned SSSP vertex compaction");
}

void settle_sssp_r_vertices(
    const int* r_flags,
    int* settled,
    std::size_t owned_count,
    cudaStream_t stream) {
    if (owned_count == 0) return;
    if (!r_flags || !settled) {
        throw std::invalid_argument("invalid SSSP settle state");
    }
    settle_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
        r_flags, settled, owned_count);
    check_launch("partitioned SSSP settle bucket");
}

void find_sssp_minimum_bucket(
    const double* distances,
    const int* settled,
    std::uint64_t* minimum_bucket,
    std::size_t owned_count,
    double delta,
    cudaStream_t stream,
    SsspControlPolicy policy) {
    if (owned_count == 0) return;
    if (!distances || !settled || !minimum_bucket ||
        !std::isfinite(delta) || delta <= 0.0) {
        throw std::invalid_argument("invalid SSSP minimum bucket state");
    }
    if (policy == SsspControlPolicy::Warp) {
        minimum_bucket_warp_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
            distances, settled, minimum_bucket, owned_count, delta);
    } else {
        minimum_bucket_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
            distances, settled, minimum_bucket, owned_count, delta);
    }
    check_launch("partitioned SSSP minimum bucket");
}

}  // namespace partitioned
}  // namespace gpu_easygraph
