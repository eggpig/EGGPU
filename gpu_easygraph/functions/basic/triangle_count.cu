#include "triangle_count.cuh"
#include <algorithm>
#include <cuda_runtime.h>

namespace gpu_easygraph {
namespace {
// A warp owns one forward edge; lanes divide the shorter adjacency list.
// Only edges whose unique root belongs to this rank are scheduled.
__global__ void triangle_owner_kernel(const int* row, const int* col,
    const int* sources, int edge_begin, int edge_count, unsigned long long* total) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int warps = (blockDim.x * gridDim.x) >> 5;
    unsigned long long count = 0;
    for (long long e = warp; e < edge_count; e += warps) {
        const int u = sources[e], v = col[edge_begin + e];
        int small_begin = row[u], small_end = row[u + 1];
        int big_begin = row[v], big_end = row[v + 1];
        if (small_end - small_begin > big_end - big_begin) {
            const int a = small_begin, b = small_end;
            small_begin = big_begin; small_end = big_end;
            big_begin = a; big_end = b;
        }
        for (int i = small_begin + lane; i < small_end; i += 32) {
            const int target = col[i];
            int lo = big_begin, hi = big_end;
            while (lo < hi) {
                const int mid = lo + (hi - lo) / 2;
                if (col[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            if (lo < big_end && col[lo] == target) ++count;
        }
    }
    for (int delta = 16; delta; delta >>= 1)
        count += __shfl_down_sync(0xffffffffU, count, delta);
    __shared__ unsigned long long partial[8];
    if (lane == 0) partial[threadIdx.x >> 5] = count;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long sum = 0;
        for (int w = 0; w < 8; ++w) sum += partial[w];
        if (sum) atomicAdd(total, sum);
    }
}
}
void launch_triangle_count(const int* row, const int* col, const int* sources,
    int edge_begin, int edge_count, unsigned long long* total, cudaStream_t stream) {
    if (!edge_count) return;
    const int blocks = std::max(1, std::min(65535, (edge_count - 1) / 8 + 1));
    triangle_owner_kernel<<<blocks, 256, 0, stream>>>(row, col, sources, edge_begin, edge_count, total);
}
}
