// All launches use 256 threads: every hardware warp is complete, including
// the final data tile. Out-of-range lanes participate with a false predicate
// or the minimum identity; no lane exits before the warp collective.
__device__ int reserve_sssp_warp_position(bool selected, int* counter) {
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int selected_mask = __ballot_sync(0xffffffffU, selected);
    if (selected_mask == 0U) return -1;
    const int leader = __ffs(selected_mask) - 1;
    int base = 0;
    if (static_cast<int>(lane) == leader) {
        base = atomicAdd(counter, __popc(selected_mask));
    }
    base = __shfl_sync(0xffffffffU, base, leader);
    const unsigned int preceding = selected_mask & ((1U << lane) - 1U);
    return selected ? base + __popc(preceding) : -1;
}

__global__ void select_bucket_warp_kernel(
    const double* distances, const int* settled, int* frontier,
    int* frontier_count, std::size_t count, std::uint64_t bucket,
    double delta) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool selected = index < count && settled[index] == 0 &&
        bucket_of(distances[index], delta) == bucket;
    const int position = reserve_sssp_warp_position(selected, frontier_count);
    if (selected) frontier[position] = static_cast<int>(index);
}

__global__ void compact_flags_warp_kernel(
    const int* flags, int* frontier, std::size_t count, int* frontier_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool selected = index < count && flags[index] != 0;
    const int position = reserve_sssp_warp_position(selected, frontier_count);
    if (selected) frontier[position] = static_cast<int>(index);
}

template <typename Wire>
__global__ void compact_route_warp_kernel(
    const double* candidates, const int* owner_local_ids,
    Wire* compact,
    std::size_t count, int* compact_count, double maximum_distance) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const double candidate = index < count ? candidates[index] : INFINITY;
    const bool selected = isfinite(candidate) && candidate <= maximum_distance;
    const int position = reserve_sssp_warp_position(selected, compact_count);
    if (selected) {
        compact[position] = encode_sssp_wire<Wire>(owner_local_ids[index],candidate);
    }
}

__global__ void minimum_bucket_warp_kernel(
    const double* distances, const int* settled, std::uint64_t* minimum,
    std::size_t count, double delta) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned long long bucket = static_cast<unsigned long long>(kSsspNoBucket);
    if (index < count && settled[index] == 0) {
        bucket = static_cast<unsigned long long>(bucket_of(distances[index], delta));
    }
    for (int offset = 16; offset != 0; offset /= 2) {
        const unsigned long long other = __shfl_down_sync(0xffffffffU, bucket, offset);
        if (other < bucket) bucket = other;
    }
    if ((threadIdx.x & 31U) == 0U && bucket != kSsspNoBucket) {
        atomicMin(reinterpret_cast<unsigned long long*>(minimum), bucket);
    }
}

template <bool UseNativeMin, typename Weight, bool DirectQueue>
__global__ void relax_edges_warp_kernel(
    const int* offsets, const int* indices, const Weight* weights,
    const int* owners, const int* halo_route_slots, double* distances,
    const int* settled, const int* frontier, std::size_t frontier_count,
    int* next_flags, int* r_flags, double* remote_candidates,
    std::size_t owned_count, int partition_id,
    std::uint64_t current_bucket, double delta, bool light, SsspNextFrontier next) {
    const unsigned int lane = threadIdx.x & 31U;
    const std::size_t frontier_index =
        static_cast<std::size_t>(blockIdx.x) * (blockDim.x / 32U) + threadIdx.x / 32U;
    // This predicate is uniform across the full warp.
    if (frontier_index >= frontier_count) return;
    int source = lane == 0U ? frontier[frontier_index] : 0;
    source = __shfl_sync(0xffffffffU, source, 0);
    int valid = 0;
    if (lane == 0U) {
        valid = source >= 0 && static_cast<std::size_t>(source) < owned_count &&
            settled[source] == 0;
    }
    valid = __shfl_sync(0xffffffffU, valid, 0);
    if (!valid) return;
    double source_distance = 0.0;
    int row_begin = 0, row_end = 0;
    if (lane == 0U) {
        if (light) atomicExch(r_flags + source, 1);
        source_distance = distances[source];
        row_begin = offsets[source];
        row_end = offsets[source + 1];
    }
    source_distance = __shfl_sync(0xffffffffU, source_distance, 0);
    row_begin = __shfl_sync(0xffffffffU, row_begin, 0);
    row_end = __shfl_sync(0xffffffffU, row_end, 0);
    // size_t avoids signed overflow on the final +32 near an int32 CSR bound.
    for (std::size_t edge = static_cast<std::size_t>(row_begin) + lane;
         edge < static_cast<std::size_t>(row_end); edge += 32U) {
        const double weight = static_cast<double>(weights[edge]);
        if ((weight <= delta) != light) continue;
        const double candidate = source_distance + weight;
        if (!isfinite(candidate)) continue;
        const int target = indices[edge];
        if (owners[target] == partition_id) {
            if (target < 0 || static_cast<std::size_t>(target) >= owned_count ||
                settled[target] != 0) continue;
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
