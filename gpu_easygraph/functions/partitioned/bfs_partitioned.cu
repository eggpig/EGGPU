#include "partitioned/bfs_partitioned.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <string>

namespace gpu_easygraph {
namespace partitioned {
namespace {

__global__ void initialize_bfs_kernel(
    std::int64_t* distances,
    int* frontier,
    std::uint32_t* bitmap,
    std::size_t count,
    int source_local) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        // Each distance has one writer, including a source in another block.
        distances[index] = static_cast<int>(index) == source_local
            ? 0 : kBfsUnvisitedDistance;
    }
    if (index == 0 && source_local >= 0) {
        frontier[0] = source_local;
        atomicOr(
            bitmap + static_cast<std::size_t>(source_local) / 32U,
            static_cast<std::uint32_t>(1U) <<
                (static_cast<unsigned int>(source_local) & 31U));
    }
}

__global__ void list_to_bitmap_kernel(
    const int* frontier,
    std::size_t count,
    std::uint32_t* bitmap,
    std::size_t owned_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int vertex = frontier[index];
    if (vertex >= 0 && static_cast<std::size_t>(vertex) < owned_count) {
        atomicOr(
            bitmap + static_cast<std::size_t>(vertex) / 32U,
            static_cast<std::uint32_t>(1U) <<
                (static_cast<unsigned int>(vertex) & 31U));
    }
}

__device__ bool claim_distance(
    std::int64_t* address, std::int64_t distance) {
    unsigned long long* raw = reinterpret_cast<unsigned long long*>(address);
    const unsigned long long previous = atomicCAS(
        raw,
        static_cast<unsigned long long>(kBfsUnvisitedDistance),
        static_cast<unsigned long long>(distance));
    return previous ==
        static_cast<unsigned long long>(kBfsUnvisitedDistance);
}

template <bool DirectRoutes>
__device__ __forceinline__ void visit_sparse_target(
    int target, const int* owners, const int* halo_route_slots,
    std::int64_t* distances, int* next_frontier, int* next_count,
    int* send_flags, std::size_t owned_count, int partition_id,
    std::int64_t next_distance, const int* route_destination_ranks,
    const int* route_owner_local_ids, const std::size_t* outgoing_route_offsets,
    int* direct_send_values, int* direct_send_counts) {
    if (owners[target] == partition_id) {
        if (target >= 0 && static_cast<std::size_t>(target) < owned_count &&
            claim_distance(distances + target, next_distance)) {
            const int position = atomicAdd(next_count, 1);
            next_frontier[position] = target;
        }
    } else {
        const int slot = halo_route_slots[target];
        if (slot < 0) return;
        if (DirectRoutes) {
            // Each source rank emits each halo target once per query.
            if (atomicCAS(send_flags + slot, 0, 1) == 0) {
                const int destination = route_destination_ranks[slot];
                const int position = atomicAdd(direct_send_counts + destination, 1);
                direct_send_values[outgoing_route_offsets[destination] + position] =
                    route_owner_local_ids[slot];
            }
        } else atomicExch(send_flags + slot, 1);
    }
}

template <unsigned int LaneCount, bool DirectRoutes, bool QueueHigh = false>
__global__ void expand_sparse_kernel(
    const int* offsets,
    const int* indices,
    const int* owners,
    const int* halo_route_slots,
    std::int64_t* distances,
    const int* current_frontier,
    std::size_t current_count,
    int* next_frontier,
    int* next_count,
    int* send_flags,
    std::size_t owned_count,
    int partition_id,
    std::int64_t next_distance,
    const int* route_destination_ranks,
    const int* route_owner_local_ids,
    const std::size_t* outgoing_route_offsets,
    int* direct_send_values,
    int* direct_send_counts,
    int* high_degree_rows = nullptr,
    int* high_degree_control = nullptr) {
    const std::size_t frontier_index =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / LaneCount;
    if (frontier_index >= current_count) return;
    const int source = current_frontier[frontier_index];
    if (source < 0 || static_cast<std::size_t>(source) >= owned_count) return;
    using EdgeIndex = typename std::conditional<LaneCount == 1U, int, std::int64_t>::type;
    const unsigned int lane = threadIdx.x % LaneCount;
    if (QueueHigh && static_cast<std::size_t>(offsets[source + 1] - offsets[source]) >= kBfsHighDegreeThreshold) {
        if (lane == 0) high_degree_rows[atomicAdd(high_degree_control, 1)] = source;
        return;
    }
    for (EdgeIndex edge = static_cast<EdgeIndex>(offsets[source]) + lane;
         edge < offsets[source + 1]; edge += LaneCount) {
        visit_sparse_target<DirectRoutes>(indices[edge], owners, halo_route_slots,
            distances, next_frontier, next_count, send_flags, owned_count,
            partition_id, next_distance, route_destination_ranks,
            route_owner_local_ids, outgoing_route_offsets, direct_send_values,
            direct_send_counts);
    }
}

template <bool DirectRoutes>
__global__ void expand_sparse_high_degree_kernel(
    const int* offsets, const int* indices, const int* owners,
    const int* halo_route_slots, std::int64_t* distances,
    int* next_frontier, int* next_count, int* send_flags,
    std::size_t owned_count, int partition_id, std::int64_t next_distance,
    const int* route_destination_ranks, const int* route_owner_local_ids,
    const std::size_t* outgoing_route_offsets, int* direct_send_values,
    int* direct_send_counts, const int* high_degree_rows, int* high_degree_control,
    int CtasPerRow) {
    __shared__ int task;
    // The preceding classifier completed on this stream; count and row IDs
    // are immutable. Only the ticket at control[1] changes here.
    const int task_count = high_degree_control[0] * CtasPerRow;
    for (;;) {
        if (threadIdx.x == 0) task = atomicAdd(high_degree_control + 1, 1);
        __syncthreads();
        if (task >= task_count) return;
        const int source = high_degree_rows[task / CtasPerRow];
        const int slice = task % CtasPerRow;
        for (std::int64_t edge = static_cast<std::int64_t>(offsets[source]) +
                 slice * blockDim.x + threadIdx.x;
             edge < offsets[source + 1]; edge += CtasPerRow * blockDim.x) {
            visit_sparse_target<DirectRoutes>(indices[edge], owners, halo_route_slots,
                distances, next_frontier, next_count, send_flags, owned_count,
                partition_id, next_distance, route_destination_ranks,
                route_owner_local_ids, outgoing_route_offsets, direct_send_values,
                direct_send_counts);
        }
        __syncthreads();
    }
}

__global__ void compact_route_kernel(
    const int* flags,
    const int* owner_local_ids,
    int* compact_owner_local_ids,
    std::size_t count,
    int* compact_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && flags[index] != 0) {
        const int position = atomicAdd(compact_count, 1);
        compact_owner_local_ids[position] = owner_local_ids[index];
    }
}

__global__ void apply_candidates_kernel(
    const int* owner_local_ids,
    std::size_t count,
    std::int64_t* distances,
    int* next_frontier,
    int* next_count,
    std::size_t owned_count,
    std::int64_t next_distance) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int target = owner_local_ids[index];
    if (target >= 0 && static_cast<std::size_t>(target) < owned_count &&
        claim_distance(distances + target, next_distance)) {
        const int position = atomicAdd(next_count, 1);
        next_frontier[position] = target;
    }
}

__global__ void apply_candidates_device_count_kernel(
    const int* owner_local_ids, const int* device_count,
    std::int64_t* distances, int* next_frontier, int* next_count,
    std::size_t owned_count, std::int64_t next_distance) {
    const std::size_t count = static_cast<std::size_t>(*device_count);
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const int target = owner_local_ids[index];
        if (target >= 0 && static_cast<std::size_t>(target) < owned_count &&
            claim_distance(distances + target, next_distance)) {
            const int position = atomicAdd(next_count, 1);
            next_frontier[position] = target;
        }
    }
}

template <unsigned int LaneCount>
__global__ void expand_dense_kernel(
    const int* offsets,
    const int* indices,
    const int* owners,
    const int* halo_route_slots,
    const int* route_destination_ranks,
    const int* route_owner_local_ids,
    const std::size_t* outgoing_dense_word_offsets,
    std::int64_t* distances,
    const std::uint32_t* current_bitmap,
    std::uint32_t* next_bitmap,
    int* next_frontier,
    int* next_count,
    std::uint32_t* dense_send_bitmap,
    std::size_t owned_count,
    int partition_id,
    std::int64_t next_distance,
    int* persistent_remote_seen) {
    const std::size_t source =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / LaneCount;
    if (source >= owned_count) return;
    const std::uint32_t source_mask =
        static_cast<std::uint32_t>(1U) << (source & 31U);
    if ((current_bitmap[source / 32U] & source_mask) == 0U) return;
    using EdgeIndex = typename std::conditional<LaneCount == 1U, int, std::int64_t>::type;
    const unsigned int lane = threadIdx.x % LaneCount;
    for (EdgeIndex edge = static_cast<EdgeIndex>(offsets[source]) + lane;
         edge < offsets[source + 1U]; edge += LaneCount) {
        const int target = indices[edge];
        if (owners[target] == partition_id) {
            if (target >= 0 && static_cast<std::size_t>(target) < owned_count &&
                claim_distance(distances + target, next_distance)) {
                atomicOr(
                    next_bitmap + static_cast<std::size_t>(target) / 32U,
                    static_cast<std::uint32_t>(1U) <<
                        (static_cast<unsigned int>(target) & 31U));
                const int position = atomicAdd(next_count, 1);
                next_frontier[position] = target;
            }
        } else {
            const int slot = halo_route_slots[target];
            if (slot < 0) continue;
            if (persistent_remote_seen &&
                atomicCAS(persistent_remote_seen + slot, 0, 1) != 0) continue;
            const int destination = route_destination_ranks[slot];
            const int owner_local = route_owner_local_ids[slot];
            const std::size_t word =
                outgoing_dense_word_offsets[destination] +
                static_cast<std::size_t>(owner_local) / 32U;
            atomicOr(
                dense_send_bitmap + word,
                static_cast<std::uint32_t>(1U) <<
                    (static_cast<unsigned int>(owner_local) & 31U));
        }
    }
}

__global__ void apply_owner_bitmap_kernel(
    const std::uint32_t* owner_bitmap,
    std::size_t bitmap_words,
    std::int64_t* distances,
    std::uint32_t* next_bitmap,
    int* next_frontier,
    int* next_count,
    std::size_t owned_count,
    std::int64_t next_distance) {
    const std::size_t target =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (target >= owned_count || target / 32U >= bitmap_words) return;
    const std::uint32_t mask =
        static_cast<std::uint32_t>(1U) << (target & 31U);
    if ((owner_bitmap[target / 32U] & mask) != 0U &&
        claim_distance(distances + target, next_distance)) {
        atomicOr(next_bitmap + target / 32U, mask);
        const int position = atomicAdd(next_count, 1);
        next_frontier[position] = static_cast<int>(target);
    }
}

template<unsigned int LaneCount>
__global__ void expand_pull_kernel(
    const int* offsets, const int* indices, const int* owners,
    const int* halo_route_slots, const int* route_destination_ranks,
    const int* route_owner_local_ids, const std::size_t* owner_word_offsets,
    std::int64_t* distances, const std::uint32_t* current_bitmap,
    const std::uint32_t* remote_frontiers, std::uint32_t* next_bitmap,
    int* next_frontier, int* next_count, std::size_t owned_count,
    int partition_id, std::int64_t next_distance) {
    const std::size_t row = (static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x)/LaneCount;
    if(row>=owned_count || distances[row]!=kBfsUnvisitedDistance)return;
    const unsigned lane=threadIdx.x%LaneCount;
    const unsigned base=(threadIdx.x%32U)/LaneCount*LaneCount;
    const unsigned mask=static_cast<unsigned>(((1ULL<<LaneCount)-1ULL)<<base);
    bool discovered=false;
    for(std::int64_t begin=offsets[row];begin<offsets[row+1];begin+=LaneCount){
        const auto edge=begin+lane;bool found=false;
        if(edge<offsets[row+1]){
            const int neighbor=indices[edge];
            if(owners[neighbor]==partition_id){
                found=(current_bitmap[static_cast<std::size_t>(neighbor)/32U]&(1U<<(neighbor&31)))!=0;
            }else{
                const int slot=halo_route_slots[neighbor];
                const int owner=route_destination_ranks[slot],local=route_owner_local_ids[slot];
                found=(remote_frontiers[owner_word_offsets[owner]+static_cast<std::size_t>(local)/32U]&(1U<<(local&31)))!=0;
            }
        }
        discovered=LaneCount==1U?found:__any_sync(mask,found);
        if(discovered)break;
    }
    if(discovered && lane==0){
        // Exactly one row group owns this distance; there are no owner-candidate
        // writers in a pull level. Neighbor tests read only the frozen bitmap.
        distances[row]=next_distance;
        atomicOr(next_bitmap+row/32U,1U<<(row&31U));
        const int position=atomicAdd(next_count,1);
        next_frontier[position]=static_cast<int>(row);
    }
}

unsigned int grid_size(std::size_t count) {
    return static_cast<unsigned int>((count + 255U) / 256U);
}

void check_launch(const char* operation) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " +
            cudaGetErrorString(status));
    }
}

}  // namespace

void initialize_bfs_frontiers(
    std::int64_t* distances,
    int* current_frontier,
    std::uint32_t* current_bitmap,
    std::size_t owned_count,
    int source_local,
    cudaStream_t stream) {
    if (owned_count == 0) {
        if (source_local >= 0) {
            throw std::invalid_argument(
                "empty BFS partition cannot own the source");
        }
        return;
    }
    if (!distances || !current_frontier || !current_bitmap ||
        source_local < -1 ||
        (source_local >= 0 &&
         static_cast<std::size_t>(source_local) >= owned_count)) {
        throw std::invalid_argument("invalid BFS frontier initialization");
    }
    const std::size_t bitmap_words = owned_count / 32U +
        (owned_count % 32U == 0U ? 0U : 1U);
    const cudaError_t cleared = cudaMemsetAsync(
        current_bitmap, 0, bitmap_words * sizeof(std::uint32_t), stream);
    if (cleared != cudaSuccess) {
        throw std::runtime_error(
            std::string("clear BFS source bitmap failed: ") +
            cudaGetErrorString(cleared));
    }
    initialize_bfs_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
        distances, current_frontier, current_bitmap, owned_count, source_local);
    check_launch("partitioned BFS initialization");
}

void build_bfs_frontier_bitmap(
    const int* current_frontier,
    std::size_t current_count,
    std::uint32_t* current_bitmap,
    std::size_t owned_count,
    cudaStream_t stream) {
    if (owned_count == 0) {
        if (current_count != 0) {
            throw std::invalid_argument(
                "empty BFS partition has a nonempty frontier");
        }
        return;
    }
    if (!current_bitmap || current_count > owned_count ||
        (current_count != 0 && !current_frontier)) {
        throw std::invalid_argument("invalid BFS bitmap conversion state");
    }
    const std::size_t bitmap_words = owned_count / 32U +
        (owned_count % 32U == 0U ? 0U : 1U);
    const cudaError_t cleared = cudaMemsetAsync(
        current_bitmap, 0, bitmap_words * sizeof(std::uint32_t), stream);
    if (cleared != cudaSuccess) {
        throw std::runtime_error(
            std::string("clear BFS frontier bitmap failed: ") +
            cudaGetErrorString(cleared));
    }
    if (current_count != 0) {
        list_to_bitmap_kernel<<<grid_size(current_count), 256, 0, stream>>>(
            current_frontier,
            current_count,
            current_bitmap,
            owned_count);
        check_launch("partitioned BFS list-to-bitmap conversion");
    }
}

void expand_bfs_sparse(
    const int* offsets,
    const int* indices,
    const int* owners,
    const int* halo_route_slots,
    std::int64_t* distances,
    const int* current_frontier,
    std::size_t current_count,
    int* next_frontier,
    int* next_count,
    int* send_flags,
    std::size_t owned_count,
    int partition_id,
    std::int64_t next_distance,
    cudaStream_t stream,
    bool warp_edges,
    const int* route_destination_ranks,
    const int* route_owner_local_ids,
    const std::size_t* outgoing_route_offsets,
    int* direct_send_values,
    int* direct_send_counts,
    int sparse_ctas_per_row,
    int* high_degree_rows,
    int* high_degree_control,
    int high_degree_grid_limit) {
    if (current_count == 0) return;
    if (!offsets || !owners || !halo_route_slots || !distances ||
        !current_frontier || !next_frontier || !next_count ||
        owned_count == 0 || current_count > owned_count ||
        partition_id < 0 || next_distance <= 0) {
        throw std::invalid_argument("invalid sparse BFS expansion state");
    }
    // A null indices pointer is valid only for a zero-edge partition, which
    // cannot dereference it because every CSR row is empty.
    if (sparse_ctas_per_row && high_degree_rows) {
        const cudaError_t cleared = cudaMemsetAsync(high_degree_control, 0,
            2U * sizeof(int), stream);
        if (cleared != cudaSuccess) throw std::runtime_error(
            std::string("clear high-degree BFS queue failed: ") + cudaGetErrorString(cleared));
        const unsigned int high_grid = static_cast<unsigned int>(
            std::min(static_cast<std::size_t>(high_degree_grid_limit),
                current_count * sparse_ctas_per_row));
        if (direct_send_counts) {
            expand_sparse_kernel<32U, true, true><<<grid_size(current_count * 32U), 256, 0, stream>>>(
                offsets, indices, owners, halo_route_slots, distances,
                current_frontier, current_count, next_frontier, next_count,
                send_flags, owned_count, partition_id, next_distance,
                route_destination_ranks, route_owner_local_ids, outgoing_route_offsets,
                direct_send_values, direct_send_counts, high_degree_rows, high_degree_control);
            check_launch("partitioned BFS active high-degree classification");
            expand_sparse_high_degree_kernel<true><<<high_grid, 256, 0, stream>>>(
                offsets, indices, owners, halo_route_slots, distances,
                next_frontier, next_count, send_flags, owned_count, partition_id,
                next_distance, route_destination_ranks, route_owner_local_ids,
                outgoing_route_offsets, direct_send_values, direct_send_counts,
                high_degree_rows, high_degree_control, sparse_ctas_per_row);
        } else {
            expand_sparse_kernel<32U, false, true><<<grid_size(current_count * 32U), 256, 0, stream>>>(
                offsets, indices, owners, halo_route_slots, distances,
                current_frontier, current_count, next_frontier, next_count,
                send_flags, owned_count, partition_id, next_distance,
                route_destination_ranks, route_owner_local_ids, outgoing_route_offsets,
                direct_send_values, direct_send_counts, high_degree_rows, high_degree_control);
            check_launch("partitioned BFS active high-degree classification");
            expand_sparse_high_degree_kernel<false><<<high_grid, 256, 0, stream>>>(
                offsets, indices, owners, halo_route_slots, distances,
                next_frontier, next_count, send_flags, owned_count, partition_id,
                next_distance, route_destination_ranks, route_owner_local_ids,
                outgoing_route_offsets, direct_send_values, direct_send_counts,
                high_degree_rows, high_degree_control, sparse_ctas_per_row);
        }
        check_launch("partitioned BFS bounded high-degree CTA expansion");
        return;
    }
    if (direct_send_counts) {
        if (warp_edges) {
            expand_sparse_kernel<32U, true><<<grid_size(current_count * 32U), 256, 0, stream>>>(
            offsets, indices, owners, halo_route_slots, distances,
            current_frontier, current_count, next_frontier, next_count,
            send_flags, owned_count, partition_id, next_distance,
            route_destination_ranks, route_owner_local_ids,
            outgoing_route_offsets, direct_send_values, direct_send_counts);
        } else {
            expand_sparse_kernel<1U, true><<<grid_size(current_count), 256, 0, stream>>>(
            offsets, indices, owners, halo_route_slots, distances,
            current_frontier, current_count, next_frontier, next_count,
            send_flags, owned_count, partition_id, next_distance,
            route_destination_ranks, route_owner_local_ids,
            outgoing_route_offsets, direct_send_values, direct_send_counts);
        }
    } else {
        if (warp_edges) {
            expand_sparse_kernel<32U, false><<<grid_size(current_count * 32U), 256, 0, stream>>>(
            offsets, indices, owners, halo_route_slots, distances,
            current_frontier, current_count, next_frontier, next_count,
            send_flags, owned_count, partition_id, next_distance,
            route_destination_ranks, route_owner_local_ids,
            outgoing_route_offsets, direct_send_values, direct_send_counts);
        } else {
            expand_sparse_kernel<1U, false><<<grid_size(current_count), 256, 0, stream>>>(
            offsets, indices, owners, halo_route_slots, distances,
            current_frontier, current_count, next_frontier, next_count,
            send_flags, owned_count, partition_id, next_distance,
            route_destination_ranks, route_owner_local_ids,
            outgoing_route_offsets, direct_send_values, direct_send_counts);
        }
    }
    check_launch("partitioned BFS sparse expansion");
}

void expand_bfs_pull(
    const int* offsets,const int* indices,const int* owners,
    const int* halo_route_slots,const int* route_destination_ranks,
    const int* route_owner_local_ids,const std::size_t* owner_word_offsets,
    std::int64_t* distances,const std::uint32_t* current_bitmap,
    const std::uint32_t* remote_frontiers,std::uint32_t* next_bitmap,
    int* next_frontier,int* next_count,std::size_t owned_count,
    int partition_id,std::int64_t next_distance,cudaStream_t stream,int pull_lanes) {
    if(!owned_count)return;
    if(pull_lanes!=1 && pull_lanes!=8 && pull_lanes!=32)throw std::invalid_argument("BFS pull lanes must be 1, 8, or 32");
    #define LAUNCH_PULL(LANES) expand_pull_kernel<LANES><<<grid_size(owned_count*LANES),256,0,stream>>>(offsets,indices,owners,halo_route_slots,route_destination_ranks,route_owner_local_ids,owner_word_offsets,distances,current_bitmap,remote_frontiers,next_bitmap,next_frontier,next_count,owned_count,partition_id,next_distance)
    if(pull_lanes==1){LAUNCH_PULL(1U);}else if(pull_lanes==8){LAUNCH_PULL(8U);}else{LAUNCH_PULL(32U);}
    #undef LAUNCH_PULL
    check_launch("partitioned BFS undirected pull expansion");
}

void compact_bfs_route(
    const int* flags,
    const int* owner_local_ids,
    int* compact_owner_local_ids,
    std::size_t route_capacity,
    int* compact_count,
    cudaStream_t stream) {
    if (route_capacity == 0) return;
    if (!flags || !owner_local_ids || !compact_owner_local_ids ||
        !compact_count) {
        throw std::invalid_argument("invalid sparse BFS route compaction");
    }
    compact_route_kernel<<<grid_size(route_capacity), 256, 0, stream>>>(
        flags,
        owner_local_ids,
        compact_owner_local_ids,
        route_capacity,
        compact_count);
    check_launch("partitioned BFS route compaction");
}

void apply_bfs_owner_candidates(
    const int* owner_local_ids,
    std::size_t count,
    std::int64_t* distances,
    int* next_frontier,
    int* next_count,
    std::size_t owned_count,
    std::int64_t next_distance,
    cudaStream_t stream) {
    if (count == 0) return;
    if (!owner_local_ids || !distances || !next_frontier || !next_count ||
        owned_count == 0 || next_distance <= 0) {
        throw std::invalid_argument("invalid BFS owner candidate state");
    }
    apply_candidates_kernel<<<grid_size(count), 256, 0, stream>>>(
        owner_local_ids,
        count,
        distances,
        next_frontier,
        next_count,
        owned_count,
        next_distance);
    check_launch("partitioned BFS owner combine");
}

void apply_bfs_owner_candidates_device_count(
    const int* owner_local_ids, const int* count,
    std::size_t route_capacity, std::int64_t* distances,
    int* next_frontier, int* next_count, std::size_t owned_count,
    std::int64_t next_distance, cudaStream_t stream) {
    if (route_capacity == 0) return;
    if (!owner_local_ids || !count || !distances || !next_frontier || !next_count ||
        owned_count == 0 || next_distance <= 0)
        throw std::invalid_argument("invalid device-count BFS owner candidates");
    // A bounded grid strides over actual candidates, never over unused route
    // capacity. This changes only queue-length publication, not claim semantics.
    const std::size_t work = route_capacity < 65536U ? route_capacity : 65536U;
    apply_candidates_device_count_kernel<<<grid_size(work), 256, 0, stream>>>(
        owner_local_ids, count, distances, next_frontier, next_count,
        owned_count, next_distance);
    check_launch("partitioned BFS same-device owner combine");
}

void expand_bfs_dense(
    const int* offsets,
    const int* indices,
    const int* owners,
    const int* halo_route_slots,
    const int* route_destination_ranks,
    const int* route_owner_local_ids,
    const std::size_t* outgoing_dense_word_offsets,
    std::int64_t* distances,
    const std::uint32_t* current_bitmap,
    std::uint32_t* next_bitmap,
    int* next_frontier,
    int* next_count,
    std::uint32_t* dense_send_bitmap,
    std::size_t owned_count,
    int partition_id,
    std::int64_t next_distance,
    cudaStream_t stream,
    bool warp_edges,
    int* persistent_remote_seen) {
    if (owned_count == 0) return;
    if (!offsets || !owners || !halo_route_slots ||
        !outgoing_dense_word_offsets || !distances || !current_bitmap ||
        !next_bitmap || !next_frontier || !next_count || partition_id < 0 ||
        next_distance <= 0) {
        throw std::invalid_argument("invalid dense BFS expansion state");
    }
    // dense_send_bitmap may be null for a one-rank graph with no halo.  The
    // validated CSR then cannot enter the remote branch.
    if (warp_edges) {
        expand_dense_kernel<32U><<<grid_size(owned_count * 32U), 256, 0, stream>>>(
            offsets,
            indices,
            owners,
            halo_route_slots,
            route_destination_ranks,
            route_owner_local_ids,
            outgoing_dense_word_offsets,
            distances,
            current_bitmap,
            next_bitmap,
            next_frontier,
            next_count,
            dense_send_bitmap,
            owned_count,
            partition_id,
            next_distance,
            persistent_remote_seen);
    } else {
        expand_dense_kernel<1U><<<grid_size(owned_count), 256, 0, stream>>>(
            offsets,
            indices,
            owners,
            halo_route_slots,
            route_destination_ranks,
            route_owner_local_ids,
            outgoing_dense_word_offsets,
            distances,
            current_bitmap,
            next_bitmap,
            next_frontier,
            next_count,
            dense_send_bitmap,
            owned_count,
            partition_id,
            next_distance,
            persistent_remote_seen);
    }
    check_launch("partitioned BFS dense expansion");
}

void apply_bfs_owner_bitmap(
    const std::uint32_t* owner_bitmap,
    std::size_t bitmap_words,
    std::int64_t* distances,
    std::uint32_t* next_bitmap,
    int* next_frontier,
    int* next_count,
    std::size_t owned_count,
    std::int64_t next_distance,
    cudaStream_t stream) {
    if (bitmap_words == 0 || owned_count == 0) return;
    if (!owner_bitmap || !distances || !next_bitmap || !next_frontier ||
        !next_count || next_distance <= 0) {
        throw std::invalid_argument("invalid BFS owner bitmap state");
    }
    apply_owner_bitmap_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
        owner_bitmap,
        bitmap_words,
        distances,
        next_bitmap,
        next_frontier,
        next_count,
        owned_count,
        next_distance);
    check_launch("partitioned BFS dense owner combine");
}

}  // namespace partitioned
}  // namespace gpu_easygraph
