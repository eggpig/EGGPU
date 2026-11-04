#include "partitioned/structural_holes_partitioned.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gpu_easygraph {
namespace partitioned {
namespace {

__device__ __forceinline__ double warp_sum(double value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ double local_constraint(
    std::uint8_t mutual,
    double indirect,
    int scale) {
    const double normalized =
        (static_cast<double>(mutual) + indirect) /
        static_cast<double>(scale);
    return normalized * normalized;
}

__global__ void evaluate_ego_kernel(
    const int* __restrict__ offsets,
    const std::uint8_t* __restrict__ mutual,
    const std::uint8_t* __restrict__ neighbor_max_scale,
    const std::uint64_t* __restrict__ common_product,
    const double* __restrict__ indirect,
    const int* __restrict__ degree,
    const int* __restrict__ sum_scale,
    const int* __restrict__ selected_nodes,
    std::size_t selected_count,
    int metric,
    double* __restrict__ result) {
    const int lane = threadIdx.x & 31;
    const std::size_t global_thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t warp = global_thread >> 5U;
    const std::size_t warp_count =
        (static_cast<std::size_t>(gridDim.x) * blockDim.x) >> 5U;
    for (std::size_t selected = warp; selected < selected_count;
         selected += warp_count) {
        const int node = selected_nodes[selected];
        const int node_degree = degree[node];
        const int scale = sum_scale[node];
        if (node_degree == 0) {
            if (lane == 0) {
                result[selected] = metric == static_cast<int>(EgoMetric::Hierarchy)
                    ? 0.0 : nan("");
            }
            continue;
        }
        const int begin = offsets[node];
        const int end = offsets[node + 1];
        double local = 0.0;
        if (metric == static_cast<int>(EgoMetric::EffectiveSize) ||
            metric == static_cast<int>(EgoMetric::Efficiency)) {
            for (int edge = begin + lane; edge < end; edge += 32) {
                const double denominator = static_cast<double>(scale) *
                    static_cast<double>(neighbor_max_scale[edge]);
                local += 1.0 -
                    static_cast<double>(common_product[edge]) / denominator;
            }
            local = warp_sum(local);
            if (lane == 0) {
                result[selected] =
                    metric == static_cast<int>(EgoMetric::Efficiency)
                    ? local / static_cast<double>(scale)
                    : local;
            }
            continue;
        }
        for (int edge = begin + lane; edge < end; edge += 32) {
            local += local_constraint(
                mutual[edge], indirect[edge], scale);
        }
        double total = warp_sum(local);
        total = __shfl_sync(0xffffffffu, total, 0);
        if (metric == static_cast<int>(EgoMetric::Constraint)) {
            if (lane == 0) result[selected] = total;
            continue;
        }
        if (node_degree <= 1 || total <= 0.0) {
            if (lane == 0) result[selected] = 0.0;
            continue;
        }
        double hierarchy = 0.0;
        for (int edge = begin + lane; edge < end; edge += 32) {
            const double probability = local_constraint(
                mutual[edge], indirect[edge], scale) / total;
            hierarchy += probability *
                log(probability * static_cast<double>(node_degree)) /
                log(static_cast<double>(node_degree));
        }
        hierarchy = warp_sum(hierarchy);
        if (lane == 0) result[selected] = hierarchy;
    }
}

__global__ void evaluate_ego_metrics_kernel(
    const int* __restrict__ offsets,
    const std::uint8_t* __restrict__ mutual,
    const std::uint8_t* __restrict__ neighbor_max_scale,
    const std::uint64_t* __restrict__ common_product,
    const double* __restrict__ indirect,
    const int* __restrict__ degree,
    const int* __restrict__ sum_scale,
    const int* __restrict__ selected_nodes,
    std::size_t selected_count,
    double* __restrict__ result) {
    const int lane = threadIdx.x & 31;
    const std::size_t global_thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t warp = global_thread >> 5U;
    const std::size_t warp_count =
        (static_cast<std::size_t>(gridDim.x) * blockDim.x) >> 5U;
    for (std::size_t selected = warp; selected < selected_count;
         selected += warp_count) {
        const int node = selected_nodes[selected];
        const int node_degree = degree[node];
        if (node_degree == 0) {
            if (lane == 0) {
                result[selected] = nan("");
                result[selected_count + selected] = nan("");
                result[2U * selected_count + selected] = nan("");
                result[3U * selected_count + selected] = 0.0;
            }
            continue;
        }
        const int scale = sum_scale[node];
        double effective = 0.0;
        double constraint = 0.0;
        double weighted_log = 0.0;
        for (int edge = offsets[node] + lane; edge < offsets[node + 1]; edge += 32) {
            const double denominator = static_cast<double>(scale) *
                static_cast<double>(neighbor_max_scale[edge]);
            effective += 1.0 - static_cast<double>(common_product[edge]) / denominator;
            const double c = local_constraint(mutual[edge], indirect[edge], scale);
            constraint += c;
            // Extend x*log(x) continuously at zero without computing log(0).
            if (c > 0.0) weighted_log += c * log(c);
        }
        effective = warp_sum(effective);
        constraint = warp_sum(constraint);
        weighted_log = warp_sum(weighted_log);
        if (lane == 0) {
            result[selected] = effective;
            result[selected_count + selected] = effective / static_cast<double>(scale);
            result[2U * selected_count + selected] = constraint;
            double hierarchy = 0.0;
            if (node_degree > 1 && constraint > 0.0) {
                const double log_degree = log(static_cast<double>(node_degree));
                hierarchy = (weighted_log / constraint - log(constraint) + log_degree) /
                    log_degree;
            }
            result[3U * selected_count + selected] = hierarchy;
        }
    }
}

void check_launch(const char* operation) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

}  // namespace

void evaluate_partitioned_ego_metric(
    const int* offsets,
    const std::uint8_t* mutual,
    const std::uint8_t* neighbor_max_scale,
    const std::uint64_t* common_product,
    const double* indirect,
    const int* degree,
    const int* sum_scale,
    const int* selected_nodes,
    std::size_t selected_count,
    std::size_t owned_vertex_count,
    EgoMetric metric,
    double* result,
    cudaStream_t stream) {
    if (selected_count == 0U) return;
    if (!offsets || !degree || !sum_scale || !selected_nodes || !result) {
        throw std::invalid_argument("invalid partitioned ego kernel state");
    }
    (void)owned_vertex_count;
    const int metric_value = static_cast<int>(metric);
    if (metric_value < static_cast<int>(EgoMetric::EffectiveSize) ||
        metric_value > static_cast<int>(EgoMetric::Hierarchy)) {
        throw std::invalid_argument("unknown partitioned ego metric");
    }
    const std::size_t warps = selected_count;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::size_t>(65535U, (warps * 32U + 255U) / 256U));
    evaluate_ego_kernel<<<blocks, 256, 0, stream>>>(
        offsets, mutual, neighbor_max_scale, common_product, indirect,
        degree, sum_scale, selected_nodes, selected_count, metric_value,
        result);
    check_launch("partitioned structural-hole reduction");
}

void evaluate_partitioned_ego_metrics(
    const int* offsets,
    const std::uint8_t* mutual,
    const std::uint8_t* neighbor_max_scale,
    const std::uint64_t* common_product,
    const double* indirect,
    const int* degree,
    const int* sum_scale,
    const int* selected_nodes,
    std::size_t selected_count,
    std::size_t owned_vertex_count,
    double* result,
    cudaStream_t stream) {
    if (selected_count == 0U) return;
    if (!offsets || !degree || !sum_scale || !selected_nodes || !result) {
        throw std::invalid_argument("invalid partitioned ego fused kernel state");
    }
    (void)owned_vertex_count;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::size_t>(65535U, (selected_count * 32U + 255U) / 256U));
    evaluate_ego_metrics_kernel<<<blocks, 256, 0, stream>>>(
        offsets, mutual, neighbor_max_scale, common_product, indirect,
        degree, sum_scale, selected_nodes, selected_count, result);
    check_launch("partitioned structural-hole fused reduction");
}

}  // namespace partitioned
}  // namespace gpu_easygraph
