#include "partitioned/pagerank_partitioned.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gpu_easygraph {
namespace partitioned {
namespace {

__device__ double atomic_add_double(double* address, double value) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    unsigned long long* raw = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *raw;
    unsigned long long assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(
            raw, assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (old != assumed);
    return __longlong_as_double(old);
#endif
}

// The pull launch has 256 threads. All lanes participate, including the
// incomplete final block, so one block publishes one residual update.
__device__ void accumulate_block_residual(double value, double* residual) {
    __shared__ double warp_totals[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    if (lane == 0) warp_totals[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < 8 ? warp_totals[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) atomic_add_double(residual, value);
    }
}

__global__ void initialize_kernel(double* values, std::size_t count, double value) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = value;
}

__global__ void peer_halo_kernel(
    const double* owner, const int* source, const int* destination,
    std::size_t count, double* contribution) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) contribution[destination[index]] = owner[source[index]];
}

__global__ void contribution_kernel(
    const double* rank,
    const double* out_weight,
    double* contribution,
    std::size_t count,
    double* dangling_sum) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double dangling = 0.0;
    bool is_dangling = false;
    if (index < count) {
        const double denominator = out_weight[index];
        is_dangling = !(denominator > 0.0);
        if (!is_dangling) {
            contribution[index] = rank[index] / denominator;
        } else {
            contribution[index] = 0.0;
            dangling = rank[index];
        }
    }
    // Most graph rows are not dangling. Such warps skip both the reduction
    // and the atomic; mixed warps publish a single sum instead of per vertex.
    if (__ballot_sync(0xffffffffU, is_dangling) != 0U) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            dangling += __shfl_down_sync(0xffffffffU, dangling, offset);
        }
        if ((threadIdx.x & 31) == 0) atomic_add_double(dangling_sum, dangling);
    }
}

template <bool Weighted>
__global__ void pull_kernel(
    const int* offsets,
    const int* sources,
    const double* weights,
    const double* contribution,
    const double* current,
    double* next,
    std::size_t count,
    double alpha,
    double base,
    double* residual,
    bool split_heavy_rows) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double local_residual = 0.0;
    if (row < count && (!split_heavy_rows || offsets[row + 1U] - offsets[row] < kPageRankHeavyRowDegree)) {
        double incoming = 0.0;
        for (int edge = offsets[row]; edge < offsets[row + 1U]; ++edge) {
            if (Weighted) incoming += contribution[sources[edge]] * weights[edge];
            else incoming += contribution[sources[edge]];
        }
        const double value = base + alpha * incoming;
        next[row] = value;
        local_residual = fabs(value - current[row]);
    }
    accumulate_block_residual(local_residual, residual);
}

template <bool Weighted>
__global__ void pull_heavy_rows_kernel(
    const int* offsets, const int* sources, const double* weights,
    const double* contribution, const double* current, double* next,
    const int* heavy_rows, std::size_t heavy_count,
    double alpha, double base, double* residual) {
    const std::size_t work = static_cast<std::size_t>(blockIdx.x) * 8U + threadIdx.x / 32U;
    const unsigned int lane = threadIdx.x & 31U;
    double incoming = 0.0;
    const int row = work < heavy_count ? heavy_rows[work] : -1;
    if (row >= 0) {
        for (std::size_t edge = static_cast<std::size_t>(offsets[row]) + lane;
             edge < static_cast<std::size_t>(offsets[row + 1]); edge += 32U) {
            if (Weighted) incoming += contribution[sources[edge]] * weights[edge];
            else incoming += contribution[sources[edge]];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        incoming += __shfl_down_sync(0xffffffffU, incoming, offset);
    }
    double local_residual = 0.0;
    if (row >= 0 && lane == 0U) {
        const double value = base + alpha * incoming;
        next[row] = value;
        local_residual = fabs(value - current[row]);
    }
    accumulate_block_residual(local_residual, residual);
}

// Fixed subwarps cover every destination row. All lanes stay active for the
// block residual reduction, including the incomplete final row group.
template <bool Weighted, int Width>
__global__ void pull_vector_rows_kernel(
    const int* offsets, const int* sources, const double* weights,
    const double* contribution, const double* current, double* next,
    std::size_t count, double alpha, double base, double* residual) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row = tid / Width;
    const int lane = threadIdx.x % Width;
    double incoming = 0.0;
    if (row < count) {
        for (std::size_t edge = static_cast<std::size_t>(offsets[row]) + lane;
             edge < static_cast<std::size_t>(offsets[row + 1U]); edge += Width) {
            if (Weighted) incoming += contribution[sources[edge]] * weights[edge];
            else incoming += contribution[sources[edge]];
        }
    }
    for (int offset = Width / 2; offset > 0; offset >>= 1)
        incoming += __shfl_down_sync(0xffffffffU, incoming, offset, Width);
    double local_residual = 0.0;
    if (row < count && lane == 0) {
        const double value = base + alpha * incoming;
        next[row] = value;
        local_residual = fabs(value - current[row]);
    }
    accumulate_block_residual(local_residual, residual);
}

void check_launch(const char* operation) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

unsigned int grid_size(std::size_t count) {
    const std::size_t blocks = (count + 255U) / 256U;
    return static_cast<unsigned int>(blocks == 0 ? 1U : blocks);
}

template <bool Weighted>
void launch_pull_rows(
    const int* offsets, const int* sources, const double* weights,
    const double* contribution, const double* current, double* next,
    std::size_t count, double alpha, double base, double* residual,
    cudaStream_t stream, const int* heavy_rows, std::size_t heavy_count) {
    const char* setting = std::getenv("EGMGPU_PAGERANK_PULL");
    const std::string policy = setting ? setting : "split";
    if (policy != "split") {
        if (policy == "vector8")
            pull_vector_rows_kernel<Weighted, 8><<<grid_size(count * 8U), 256, 0, stream>>>(
                offsets, sources, weights, contribution, current, next, count, alpha, base, residual);
        else if (policy == "vector16")
            pull_vector_rows_kernel<Weighted, 16><<<grid_size(count * 16U), 256, 0, stream>>>(
                offsets, sources, weights, contribution, current, next, count, alpha, base, residual);
        else if (policy == "vector32")
            pull_vector_rows_kernel<Weighted, 32><<<grid_size(count * 32U), 256, 0, stream>>>(
                offsets, sources, weights, contribution, current, next, count, alpha, base, residual);
        else throw std::invalid_argument("EGMGPU_PAGERANK_PULL must be split, vector8, vector16 or vector32");
        check_launch("partitioned PageRank vector pull");
        return;
    }
    pull_kernel<Weighted><<<grid_size(count), 256, 0, stream>>>(
        offsets, sources, weights, contribution, current, next, count,
        alpha, base, residual, heavy_count != 0U);
    check_launch("partitioned PageRank pull");
    if (heavy_count != 0U) {
        pull_heavy_rows_kernel<Weighted><<<static_cast<unsigned int>((heavy_count + 7U) / 8U), 256, 0, stream>>>(
            offsets, sources, weights, contribution, current, next,
            heavy_rows, heavy_count, alpha, base, residual);
        check_launch("partitioned PageRank heavy row pull");
    }
}

}  // namespace

void gather_peer_page_rank_halo(
    const double* owner_contribution, const int* source_local,
    const int* destination_local, std::size_t count,
    double* contribution, cudaStream_t stream) {
    if (count == 0) return;
    peer_halo_kernel<<<grid_size(count), 256, 0, stream>>>(
        owner_contribution, source_local, destination_local, count, contribution);
    check_launch("partitioned PageRank peer halo gather");
}

void initialize_owned_rank(
    double* rank, std::size_t owned_count, double value, cudaStream_t stream) {
    if (owned_count == 0) return;
    if (!rank) {
        throw std::invalid_argument("PageRank initialization requires owned vertices");
    }
    initialize_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
        rank, owned_count, value);
    check_launch("partitioned PageRank initialization");
}

void prepare_page_rank_contribution(
    const double* rank,
    const double* out_weight,
    double* contribution,
    std::size_t owned_count,
    double* dangling_sum,
    cudaStream_t stream) {
    if (!dangling_sum ||
        (owned_count != 0 && (!rank || !out_weight || !contribution))) {
        throw std::invalid_argument("invalid partitioned PageRank contribution");
    }
    const cudaError_t clear = cudaMemsetAsync(
        dangling_sum, 0, sizeof(double), stream);
    if (clear != cudaSuccess) {
        throw std::runtime_error(
            std::string("clear PageRank dangling sum failed: ") +
            cudaGetErrorString(clear));
    }
    if (owned_count != 0) {
        contribution_kernel<<<grid_size(owned_count), 256, 0, stream>>>(
            rank, out_weight, contribution, owned_count, dangling_sum);
        check_launch("partitioned PageRank contribution");
    }
}

void pull_page_rank_rows(
    const int* incoming_offsets,
    const int* incoming_sources,
    const double* incoming_weights,
    const double* local_contribution,
    const double* current_rank,
    double* next_rank,
    std::size_t owned_count,
    double alpha,
    double base,
    double* residual,
    cudaStream_t stream,
    const int* heavy_rows,
    std::size_t heavy_row_count) {
    if (!residual ||
        (owned_count != 0 &&
         (!incoming_offsets || !local_contribution || !current_rank ||
          !next_rank))) {
        throw std::invalid_argument("invalid partitioned PageRank pull input");
    }
    const cudaError_t clear = cudaMemsetAsync(residual, 0, sizeof(double), stream);
    if (clear != cudaSuccess) {
        throw std::runtime_error(
            std::string("clear PageRank residual failed: ") +
            cudaGetErrorString(clear));
    }
    if (owned_count != 0) {
        if (incoming_weights) {
            launch_pull_rows<true>(
                incoming_offsets, incoming_sources, incoming_weights, local_contribution,
                current_rank, next_rank, owned_count, alpha, base, residual,
                stream, heavy_rows, heavy_row_count);
        } else {
            launch_pull_rows<false>(
                incoming_offsets, incoming_sources, nullptr, local_contribution,
                current_rank, next_rank, owned_count, alpha, base, residual,
                stream, heavy_rows, heavy_row_count);
        }
    }
}

void pull_page_rank_rows(
    const int* incoming_offsets, const int* incoming_sources,
    const double* incoming_weights, const double* local_contribution,
    const double* current_rank, double* next_rank, std::size_t owned_count,
    double alpha, double base, double* residual, cudaStream_t stream) {
    pull_page_rank_rows(incoming_offsets, incoming_sources, incoming_weights,
        local_contribution, current_rank, next_rank, owned_count, alpha, base,
        residual, stream, nullptr, 0U);
}

}  // namespace partitioned
}  // namespace gpu_easygraph
