#include "partitioned/wcc_partitioned.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace gpu_easygraph {
namespace partitioned {
namespace {

__device__ unsigned long long atomic_min_u64(
    unsigned long long* address, unsigned long long value) {
    unsigned long long observed = *address;
    while (value < observed) {
        const unsigned long long previous = atomicCAS(address, observed, value);
        if (previous == observed) break;
        observed = previous;
    }
    return observed;
}

__global__ void initialize_labels_kernel(
    const std::uint64_t* local_to_global,
    std::uint64_t* labels,
    std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) labels[index] = local_to_global[index];
}

__device__ int find_owned_local(
    const std::uint64_t* local_to_global,
    std::size_t owned_count,
    std::uint64_t global_id) {
    std::size_t low = 0;
    std::size_t high = owned_count;
    while (low < high) {
        const std::size_t middle = low + (high - low) / 2U;
        const std::uint64_t observed = local_to_global[middle];
        if (observed < global_id) {
            low = middle + 1U;
        } else {
            high = middle;
        }
    }
    return low < owned_count && local_to_global[low] == global_id
        ? static_cast<int>(low)
        : -1;
}

__global__ void hook_kernel(
    const int* offsets,
    const int* indices,
    const int* owners,
    std::uint64_t* labels,
    std::size_t owned_count,
    int partition_id,
    bool boundary,
    int* changed) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= owned_count) return;
    for (int edge = offsets[row]; edge < offsets[row + 1U]; ++edge) {
        const int neighbor = indices[edge];
        const bool is_boundary = owners[neighbor] != partition_id;
        if (is_boundary != boundary) continue;
        const std::uint64_t source_label = labels[row];
        const std::uint64_t neighbor_label = labels[neighbor];
        const std::uint64_t candidate = source_label < neighbor_label
            ? source_label
            : neighbor_label;
        if (candidate < source_label) {
            const unsigned long long previous = atomic_min_u64(
                reinterpret_cast<unsigned long long*>(labels + row),
                static_cast<unsigned long long>(candidate));
            if (candidate < previous) atomicExch(changed, 1);
        }
        if (!is_boundary && static_cast<std::size_t>(neighbor) < owned_count &&
            candidate < neighbor_label) {
            const unsigned long long previous = atomic_min_u64(
                reinterpret_cast<unsigned long long*>(labels + neighbor),
                static_cast<unsigned long long>(candidate));
            if (candidate < previous) atomicExch(changed, 1);
        }
    }
}

__global__ void compress_kernel(
    const std::uint64_t* local_to_global,
    std::uint64_t* labels,
    std::size_t owned_count,
    int* changed) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= owned_count) return;
    std::uint64_t value = labels[row];
    // Each hop strictly decreases a global label.  The bound prevents a
    // malformed device state from spinning; later global iterations continue
    // compression when a chain is longer.
    for (int hop = 0; hop < 64; ++hop) {
        const int parent_local = find_owned_local(
            local_to_global, owned_count, value);
        if (parent_local < 0) break;
        const std::uint64_t parent = labels[parent_local];
        if (parent >= value) break;
        value = parent;
    }
    if (value < labels[row]) {
        labels[row] = value;
        atomicExch(changed, 1);
    }
}

__global__ void pack_kernel(
    const std::uint64_t* labels,
    const int* local_ids,
    std::uint64_t* packed,
    std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) packed[index] = labels[local_ids[index]];
}

__global__ void initialize_local_parents_kernel(int* parents, std::size_t count) {
    const std::size_t row = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (row < count) parents[row] = static_cast<int>(row);
}

// Every link strictly decreases an owner-local ID. CAS only hooks a root;
// concurrent path halving preserves connectivity and cannot introduce cycles.
__device__ int local_root(int* parents, int vertex) {
    volatile int* observed = parents;
    int parent = observed[vertex];
    while (parent != vertex) {
        const int grandparent = observed[parent];
        if (grandparent != parent) atomicMin(parents + vertex, grandparent);
        vertex = parent;
        parent = grandparent;
    }
    return vertex;
}

template <bool UniqueEdges>
__global__ void union_local_edges_kernel(const int* offsets, const int* indices,
                                        int* parents, std::size_t owned_count) {
    const std::size_t thread = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t row = thread / 32U;
    if (row >= owned_count) return;
    const int lane = threadIdx.x & 31;
    for (std::int64_t edge = static_cast<std::int64_t>(offsets[row]) + lane;
         edge < offsets[row + 1U]; edge += 32) {
        const int neighbor = indices[edge];
        if (static_cast<std::size_t>(neighbor) >= owned_count || neighbor == row) continue;
        // The undirected contract stores both orientations. One owner-local
        // orientation suffices for union; duplicate parallel edges remain safe.
        if (UniqueEdges && static_cast<std::size_t>(neighbor) < row) continue;
        int left = local_root(parents, static_cast<int>(row));
        int right = local_root(parents, neighbor);
        while (left != right) {
            const int high = left > right ? left : right;
            const int low = left < right ? left : right;
            const int prior = atomicCAS(parents + high, high, low);
            if (prior == high || prior == low) break;
            left = local_root(parents, prior);
            right = local_root(parents, low);
        }
    }
}

__global__ void compress_local_parents_kernel(int* parents, std::size_t count) {
    const std::size_t row = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (row < count) atomicMin(parents + row, local_root(parents, static_cast<int>(row)));
}

__global__ void project_component_labels_kernel(const int* parents,
                                               std::uint64_t* labels,
                                               std::size_t count) {
    const std::size_t row = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (row < count && parents[row] != row) labels[row] = labels[parents[row]];
}

__global__ void combine_boundary_components_kernel(
    const int* sources, const int* neighbors, std::size_t count,
    const int* parents, std::uint64_t* labels, int* changed) {
    const std::size_t edge = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (edge >= count) return;
    const int root = parents[sources[edge]];
    const unsigned long long candidate = labels[neighbors[edge]];
    if (candidate < labels[root]) {
        const unsigned long long previous = atomic_min_u64(
            reinterpret_cast<unsigned long long*>(labels + root), candidate);
        if (candidate < previous) atomicExch(changed, 1);
    }
}

__global__ void combine_boundary_components_csr_kernel(
    const int* offsets, const int* indices, std::size_t owned_count,
    const int* parents, std::uint64_t* labels, int* changed) {
    const std::size_t thread = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t row = thread / 32U;
    if (row >= owned_count) return;
    const int lane = threadIdx.x & 31;
    // Parent compression has completed before this phase. Each complete warp
    // reuses its row's fixed root and CSR interval; halo labels are the same
    // completed receive snapshot used by the COO implementation.
    const int root = __shfl_sync(0xffffffffU, lane == 0 ? parents[row] : 0, 0);
    const int begin = __shfl_sync(0xffffffffU, lane == 0 ? offsets[row] : 0, 0);
    const int end = __shfl_sync(0xffffffffU, lane == 0 ? offsets[row + 1U] : 0, 0);
    for (std::int64_t edge = static_cast<std::int64_t>(begin) + lane; edge < end; edge += 32) {
        const int neighbor = indices[edge];
        if (static_cast<std::size_t>(neighbor) < owned_count) continue;
        const unsigned long long candidate = labels[neighbor];
        if (candidate < labels[root]) {
            const unsigned long long previous = atomic_min_u64(
                reinterpret_cast<unsigned long long*>(labels + root), candidate);
            if (candidate < previous) atomicExch(changed, 1);
        }
    }
}

__global__ void scatter_kernel(
    const std::uint64_t* packed,
    const int* local_ids,
    std::uint64_t* labels,
    std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) labels[local_ids[index]] = packed[index];
}

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

void hook_compress(
    const int* offsets,
    const int* indices,
    const int* owners,
    const std::uint64_t* local_to_global,
    std::uint64_t* labels,
    std::size_t owned_count,
    int partition_id,
    bool boundary,
    int* changed,
    cudaStream_t stream) {
    if (owned_count == 0) return;
    if (!offsets || !owners || !local_to_global || !labels || !changed) {
        throw std::invalid_argument("invalid partitioned WCC hook state");
    }
    hook_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
        offsets, indices, owners, labels, owned_count, partition_id,
        boundary, changed);
    check_launch(boundary ? "partitioned WCC boundary hook"
                          : "partitioned WCC local hook");
    compress_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
        local_to_global, labels, owned_count, changed);
    check_launch("partitioned WCC label compression");
}

}  // namespace

void initialize_wcc_labels(
    const std::uint64_t* local_to_global,
    std::uint64_t* labels,
    std::size_t local_count,
    cudaStream_t stream) {
    if (local_count == 0) return;
    if (!local_to_global || !labels) {
        throw std::invalid_argument("invalid partitioned WCC label state");
    }
    initialize_labels_kernel<<<grid_size(local_count), 256, 0, stream>>>(
        local_to_global, labels, local_count);
    check_launch("partitioned WCC label initialization");
}

void clear_wcc_changed(int* changed, cudaStream_t stream) {
    if (!changed) return;
    const cudaError_t status = cudaMemsetAsync(changed, 0, sizeof(int), stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("clear partitioned WCC changed flag failed: ") +
            cudaGetErrorString(status));
    }
}

void initialize_wcc_local_parents(int* parents, std::size_t count, cudaStream_t stream) {
    if (!count) return;
    initialize_local_parents_kernel<<<grid_size(count), 256, 0, stream>>>(parents, count);
    check_launch("initialize WCC local parents");
}
void union_wcc_local_edges(const int* offsets, const int* indices, int* parents,
                          std::size_t count, cudaStream_t stream, bool unique_edges) {
    if (!count) return;
    if (unique_edges) {
        union_local_edges_kernel<true><<<grid_size(count * 32U), 256, 0, stream>>>(
            offsets, indices, parents, count);
    } else {
        union_local_edges_kernel<false><<<grid_size(count * 32U), 256, 0, stream>>>(
            offsets, indices, parents, count);
    }
    check_launch("union WCC local edges");
}
void compress_wcc_local_parents(int* parents, std::size_t count, cudaStream_t stream) {
    if (!count) return;
    compress_local_parents_kernel<<<grid_size(count), 256, 0, stream>>>(parents, count);
    check_launch("compress WCC local parents");
}
void project_wcc_component_labels(const int* parents, std::uint64_t* labels,
                                 std::size_t count, cudaStream_t stream) {
    if (!count) return;
    project_component_labels_kernel<<<grid_size(count), 256, 0, stream>>>(parents, labels, count);
    check_launch("project WCC component labels");
}
void combine_wcc_boundary_components(const int* sources, const int* neighbors,
    std::size_t count, const int* parents, std::uint64_t* labels,
    int* changed, cudaStream_t stream) {
    if (!count) return;
    combine_boundary_components_kernel<<<grid_size(count), 256, 0, stream>>>(
        sources, neighbors, count, parents, labels, changed);
    check_launch("combine WCC boundary components");
}

void combine_wcc_boundary_components_csr(const int* offsets, const int* indices,
    std::size_t owned_count, const int* parents, std::uint64_t* labels,
    int* changed, cudaStream_t stream) {
    if (!owned_count) return;
    combine_boundary_components_csr_kernel<<<grid_size(owned_count * 32U), 256, 0, stream>>>(
        offsets, indices, owned_count, parents, labels, changed);
    check_launch("combine WCC boundary components from CSR");
}

void local_wcc_hook_compress(
    const int* offsets,
    const int* indices,
    const int* owners,
    const std::uint64_t* local_to_global,
    std::uint64_t* labels,
    std::size_t owned_count,
    int partition_id,
    int* changed,
    cudaStream_t stream) {
    hook_compress(
        offsets, indices, owners, local_to_global, labels, owned_count,
        partition_id, false, changed, stream);
}

void boundary_wcc_hook_compress(
    const int* offsets,
    const int* indices,
    const int* owners,
    const std::uint64_t* local_to_global,
    std::uint64_t* labels,
    std::size_t owned_count,
    int partition_id,
    int* changed,
    cudaStream_t stream) {
    hook_compress(
        offsets, indices, owners, local_to_global, labels, owned_count,
        partition_id, true, changed, stream);
}

void pack_wcc_owner_labels(
    const std::uint64_t* labels,
    const int* owner_local_ids,
    std::uint64_t* packed,
    std::size_t count,
    cudaStream_t stream) {
    if (count == 0) return;
    if (!labels || !owner_local_ids || !packed) {
        throw std::invalid_argument("invalid partitioned WCC pack state");
    }
    pack_kernel<<<grid_size(count), 256, 0, stream>>>(
        labels, owner_local_ids, packed, count);
    check_launch("partitioned WCC owner-label pack");
}

void scatter_wcc_halo_labels(
    const std::uint64_t* packed,
    const int* halo_local_ids,
    std::uint64_t* labels,
    std::size_t count,
    cudaStream_t stream) {
    if (count == 0) return;
    if (!packed || !halo_local_ids || !labels) {
        throw std::invalid_argument("invalid partitioned WCC halo scatter state");
    }
    scatter_kernel<<<grid_size(count), 256, 0, stream>>>(
        packed, halo_local_ids, labels, count);
    check_launch("partitioned WCC halo scatter");
}

}  // namespace partitioned
}  // namespace gpu_easygraph
