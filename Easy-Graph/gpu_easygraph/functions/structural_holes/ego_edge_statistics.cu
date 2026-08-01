#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#include "buffer_cache.h"
#include "device_graph_cache.h"
#include "err.h"

namespace gpu_easygraph {
namespace {

enum class EgoMetric {
    EffectiveSize,
    Constraint,
    Hierarchy,
};

struct EgoEdgeWorkspace {
    PersistentDeviceBuffer d_degree;
    PersistentDeviceBuffer d_mutual;
    PersistentDeviceBuffer d_sum_scale;
    PersistentDeviceBuffer d_max_scale;
    PersistentDeviceBuffer d_common_product;
    PersistentDeviceBuffer d_indirect;
    PersistentDeviceBuffer d_aux;
    PersistentDeviceBuffer d_result;
    PersistentPinnedBuffer h_result;

    std::uint64_t graph_id = 0;
    const int* host_V = nullptr;
    const int* host_E = nullptr;
    const int* host_forward_V = nullptr;
    const int* host_forward_E = nullptr;
    int num_nodes = -1;
    int num_entries = -1;
    int num_undirected_edges = -1;
    int device_id = -1;
    bool statistics_ready = false;

    void reset() {
        d_degree.reset();
        d_mutual.reset();
        d_sum_scale.reset();
        d_max_scale.reset();
        d_common_product.reset();
        d_indirect.reset();
        d_aux.reset();
        d_result.reset();
        h_result.reset();
        graph_id = 0;
        host_V = nullptr;
        host_E = nullptr;
        host_forward_V = nullptr;
        host_forward_E = nullptr;
        num_nodes = -1;
        num_entries = -1;
        num_undirected_edges = -1;
        statistics_ready = false;
    }

    bool matches(
        std::uint64_t candidate_graph_id,
        const int* candidate_V,
        const int* candidate_E,
        const int* candidate_forward_V,
        const int* candidate_forward_E,
        int candidate_num_nodes,
        int candidate_num_entries,
        int candidate_num_undirected_edges,
        int candidate_device
    ) const {
        return statistics_ready &&
            graph_id == candidate_graph_id &&
            host_V == candidate_V &&
            host_E == candidate_E &&
            host_forward_V == candidate_forward_V &&
            host_forward_E == candidate_forward_E &&
            num_nodes == candidate_num_nodes &&
            num_entries == candidate_num_entries &&
            num_undirected_edges == candidate_num_undirected_edges &&
            device_id == candidate_device;
    }
};

static thread_local EgoEdgeWorkspace g_ego_workspace;

static int status_from_cuda(cudaError_t status) {
    if (status == cudaSuccess) return EG_GPU_SUCC;
    if (status == cudaErrorMemoryAllocation) {
        return EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM;
    }
    return EG_GPU_DEVICE_ERR;
}

static int ensure_buffer(PersistentDeviceBuffer& buffer, std::size_t bytes) {
    return buffer.ensure_bytes(std::max<std::size_t>(bytes, 1));
}

__device__ __forceinline__ bool row_contains_sorted(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int row,
    int target
) {
    int lo = offsets[row];
    int hi = offsets[row + 1] - 1;
    while (lo <= hi) {
        const int mid = lo + ((hi - lo) >> 1);
        const int value = indices[mid];
        if (value == target) return true;
        if (value < target) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return false;
}

__global__ void initialize_edge_statistics(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ forward_offsets,
    const int* __restrict__ forward_indices,
    std::uint8_t* __restrict__ mutual,
    int* __restrict__ sum_scale,
    int* __restrict__ max_scale,
    int num_nodes
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int u = tid; u < num_nodes; u += stride) {
        for (int edge = forward_offsets[u];
             edge < forward_offsets[u + 1];
             ++edge) {
            const int v = forward_indices[edge];
            const int value =
                static_cast<int>(row_contains_sorted(offsets, indices, u, v)) +
                static_cast<int>(row_contains_sorted(offsets, indices, v, u));
            mutual[edge] = static_cast<std::uint8_t>(value);
            atomicAdd(sum_scale + u, value);
            atomicAdd(sum_scale + v, value);
            atomicMax(max_scale + u, value);
            atomicMax(max_scale + v, value);
        }
    }
}

__global__ void accumulate_triangle_statistics(
    const int* __restrict__ forward_offsets,
    const int* __restrict__ forward_indices,
    const std::uint8_t* __restrict__ mutual,
    const int* __restrict__ sum_scale,
    unsigned long long* __restrict__ common_product,
    double* __restrict__ indirect,
    int num_nodes
) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = global_thread >> 5;
    const int warp_count = (blockDim.x * gridDim.x) >> 5;
    if (warp_count <= 0) return;

    for (int u = warp_id; u < num_nodes; u += warp_count) {
        for (int uv = forward_offsets[u] + lane;
             uv < forward_offsets[u + 1];
             uv += 32) {
            const int v = forward_indices[uv];
            int uw = forward_offsets[u];
            int vw = forward_offsets[v];
            const int uw_end = forward_offsets[u + 1];
            const int vw_end = forward_offsets[v + 1];
            const int mutual_uv = static_cast<int>(mutual[uv]);

            while (uw < uw_end && vw < vw_end) {
                const int u_neighbor = forward_indices[uw];
                const int v_neighbor = forward_indices[vw];
                if (u_neighbor == v_neighbor) {
                    const int w = u_neighbor;
                    const int mutual_uw = static_cast<int>(mutual[uw]);
                    const int mutual_vw = static_cast<int>(mutual[vw]);

                    atomicAdd(
                        common_product + uv,
                        static_cast<unsigned long long>(mutual_uw * mutual_vw));
                    atomicAdd(
                        common_product + uw,
                        static_cast<unsigned long long>(mutual_uv * mutual_vw));
                    atomicAdd(
                        common_product + vw,
                        static_cast<unsigned long long>(mutual_uv * mutual_uw));

                    if (sum_scale[w] > 0) {
                        atomicAdd(
                            indirect + uv,
                            static_cast<double>(mutual_uw * mutual_vw) /
                                static_cast<double>(sum_scale[w]));
                    }
                    if (sum_scale[v] > 0) {
                        atomicAdd(
                            indirect + uw,
                            static_cast<double>(mutual_uv * mutual_vw) /
                                static_cast<double>(sum_scale[v]));
                    }
                    if (sum_scale[u] > 0) {
                        atomicAdd(
                            indirect + vw,
                            static_cast<double>(mutual_uv * mutual_uw) /
                                static_cast<double>(sum_scale[u]));
                    }
                    ++uw;
                    ++vw;
                } else if (u_neighbor < v_neighbor) {
                    ++uw;
                } else {
                    ++vw;
                }
            }
        }
    }
}

__device__ __forceinline__ double warp_sum(double value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void reduce_effective_size(
    const int* __restrict__ forward_offsets,
    const int* __restrict__ forward_indices,
    const int* __restrict__ sum_scale,
    const int* __restrict__ max_scale,
    const unsigned long long* __restrict__ common_product,
    double* __restrict__ result,
    int num_nodes
) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = global_thread >> 5;
    const int warp_count = (blockDim.x * gridDim.x) >> 5;
    if (warp_count <= 0) return;

    for (int u = warp_id; u < num_nodes; u += warp_count) {
        double local_u = 0.0;
        for (int edge = forward_offsets[u] + lane;
             edge < forward_offsets[u + 1];
             edge += 32) {
            const int v = forward_indices[edge];
            const double common =
                static_cast<double>(common_product[edge]);
            const double term_u =
                1.0 - common /
                    static_cast<double>(sum_scale[u] * max_scale[v]);
            const double term_v =
                1.0 - common /
                    static_cast<double>(sum_scale[v] * max_scale[u]);
            local_u += term_u;
            atomicAdd(result + v, term_v);
        }
        local_u = warp_sum(local_u);
        if (lane == 0 && local_u != 0.0) {
            atomicAdd(result + u, local_u);
        }
    }
}

__device__ __forceinline__ double local_constraint_value(
    std::uint8_t mutual,
    double indirect,
    int scale
) {
    const double total =
        static_cast<double>(mutual) + indirect;
    const double normalized = total / static_cast<double>(scale);
    return normalized * normalized;
}

__global__ void reduce_constraint(
    const int* __restrict__ forward_offsets,
    const int* __restrict__ forward_indices,
    const std::uint8_t* __restrict__ mutual,
    const int* __restrict__ sum_scale,
    const double* __restrict__ indirect,
    double* __restrict__ result,
    int num_nodes
) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = global_thread >> 5;
    const int warp_count = (blockDim.x * gridDim.x) >> 5;
    if (warp_count <= 0) return;

    for (int u = warp_id; u < num_nodes; u += warp_count) {
        double local_u = 0.0;
        for (int edge = forward_offsets[u] + lane;
             edge < forward_offsets[u + 1];
             edge += 32) {
            const int v = forward_indices[edge];
            const double c_u = local_constraint_value(
                mutual[edge], indirect[edge], sum_scale[u]);
            const double c_v = local_constraint_value(
                mutual[edge], indirect[edge], sum_scale[v]);
            local_u += c_u;
            atomicAdd(result + v, c_v);
        }
        local_u = warp_sum(local_u);
        if (lane == 0 && local_u != 0.0) {
            atomicAdd(result + u, local_u);
        }
    }
}

__global__ void reduce_hierarchy(
    const int* __restrict__ forward_offsets,
    const int* __restrict__ forward_indices,
    const int* __restrict__ degree,
    const std::uint8_t* __restrict__ mutual,
    const int* __restrict__ sum_scale,
    const double* __restrict__ indirect,
    const double* __restrict__ total_constraint,
    double* __restrict__ result,
    int num_nodes
) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = global_thread >> 5;
    const int warp_count = (blockDim.x * gridDim.x) >> 5;
    if (warp_count <= 0) return;

    for (int u = warp_id; u < num_nodes; u += warp_count) {
        double local_u = 0.0;
        for (int edge = forward_offsets[u] + lane;
             edge < forward_offsets[u + 1];
             edge += 32) {
            const int v = forward_indices[edge];
            if (degree[u] > 1 && total_constraint[u] > 0.0) {
                const double c_u = local_constraint_value(
                    mutual[edge], indirect[edge], sum_scale[u]);
                const double p_u = c_u / total_constraint[u];
                local_u += p_u *
                    log(p_u * static_cast<double>(degree[u])) /
                    log(static_cast<double>(degree[u]));
            }
            if (degree[v] > 1 && total_constraint[v] > 0.0) {
                const double c_v = local_constraint_value(
                    mutual[edge], indirect[edge], sum_scale[v]);
                const double p_v = c_v / total_constraint[v];
                const double value_v = p_v *
                    log(p_v * static_cast<double>(degree[v])) /
                    log(static_cast<double>(degree[v]));
                atomicAdd(result + v, value_v);
            }
        }
        local_u = warp_sum(local_u);
        if (lane == 0 && local_u != 0.0) {
            atomicAdd(result + u, local_u);
        }
    }
}

__global__ void finalize_result(
    const int* __restrict__ offsets,
    const int* __restrict__ degree,
    const double* __restrict__ total_constraint,
    double* __restrict__ result,
    int num_nodes,
    int metric
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int u = tid; u < num_nodes; u += stride) {
        if (metric == static_cast<int>(EgoMetric::Hierarchy)) {
            if (degree[u] <= 1 ||
                total_constraint == nullptr ||
                total_constraint[u] <= 0.0) {
                result[u] = 0.0;
            }
        } else if (offsets[u + 1] == offsets[u]) {
            result[u] = nan("");
        }
    }
}

static int run_ego_metric(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& forward_V,
    const std::vector<int>& forward_E,
    const std::vector<int>& degree,
    std::uint64_t graph_id,
    EgoMetric metric,
    std::vector<double>& result,
    double* kernel_seconds
) {
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    if (V.empty() || forward_V.empty()) return EG_GPU_DEVICE_ERR;

    const std::size_t n_size = V.size() - 1;
    if (n_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        E.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        forward_E.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        forward_V.size() != V.size() ||
        degree.size() != n_size ||
        V.back() != static_cast<int>(E.size()) ||
        forward_V.back() != static_cast<int>(forward_E.size())) {
        return EG_GPU_DEVICE_ERR;
    }

    const int num_nodes = static_cast<int>(n_size);
    const int num_entries = static_cast<int>(E.size());
    const int num_undirected_edges = static_cast<int>(forward_E.size());
    int device_id = -1;
    int rc = status_from_cuda(cudaGetDevice(&device_id));
    if (rc != EG_GPU_SUCC) return rc;

    EgoEdgeWorkspace& ws = g_ego_workspace;
    const bool cache_hit = ws.matches(
        graph_id,
        V.data(),
        E.data(),
        forward_V.data(),
        forward_E.data(),
        num_nodes,
        num_entries,
        num_undirected_edges,
        device_id);
    if (!cache_hit) {
        ws.reset();
        ws.graph_id = graph_id;
        ws.host_V = V.data();
        ws.host_E = E.data();
        ws.host_forward_V = forward_V.data();
        ws.host_forward_E = forward_E.data();
        ws.num_nodes = num_nodes;
        ws.num_entries = num_entries;
        ws.num_undirected_edges = num_undirected_edges;
        ws.device_id = device_id;
    }

    DeviceCsrView directed_view;
    DeviceCsrView forward_view;
    rc = acquire_device_csr(
        V.data(), E.data(), nullptr, num_nodes, num_entries, false,
        &directed_view);
    if (rc != EG_GPU_SUCC) return rc;
    rc = acquire_device_csr(
        forward_V.data(), forward_E.data(), nullptr, num_nodes,
        num_undirected_edges, false, &forward_view);
    if (rc != EG_GPU_SUCC) return rc;

    const std::size_t node_int_bytes =
        static_cast<std::size_t>(num_nodes) * sizeof(int);
    const std::size_t node_double_bytes =
        static_cast<std::size_t>(num_nodes) * sizeof(double);
    const std::size_t edge_byte_bytes =
        static_cast<std::size_t>(num_undirected_edges) *
        sizeof(std::uint8_t);
    const std::size_t edge_u64_bytes =
        static_cast<std::size_t>(num_undirected_edges) *
        sizeof(unsigned long long);
    const std::size_t edge_double_bytes =
        static_cast<std::size_t>(num_undirected_edges) * sizeof(double);

    for (const int status : {
             ensure_buffer(ws.d_degree, node_int_bytes),
             ensure_buffer(ws.d_mutual, edge_byte_bytes),
             ensure_buffer(ws.d_sum_scale, node_int_bytes),
             ensure_buffer(ws.d_max_scale, node_int_bytes),
             ensure_buffer(ws.d_common_product, edge_u64_bytes),
             ensure_buffer(ws.d_indirect, edge_double_bytes),
             ensure_buffer(ws.d_aux, node_double_bytes),
             ensure_buffer(ws.d_result, node_double_bytes)}) {
        if (status != EG_GPU_SUCC) return status;
    }

    if (!cache_hit) {
        rc = status_from_cuda(cudaMemcpy(
            ws.d_degree.data(),
            degree.data(),
            node_int_bytes,
            cudaMemcpyHostToDevice));
        if (rc != EG_GPU_SUCC) return rc;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    rc = status_from_cuda(cudaEventCreate(&start));
    if (rc != EG_GPU_SUCC) return rc;
    rc = status_from_cuda(cudaEventCreate(&stop));
    if (rc != EG_GPU_SUCC) {
        cudaEventDestroy(start);
        return rc;
    }

    auto fail = [&](cudaError_t status) {
        const int value = status_from_cuda(status);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return value;
    };

    cudaError_t status = cudaEventRecord(start, 0);
    if (status != cudaSuccess) return fail(status);

    const int block = 256;
    const int node_grid = std::max(
        1, std::min(65535, (num_nodes + block - 1) / block));
    const long long requested_warp_grid =
        (static_cast<long long>(num_nodes) * 32LL + block - 1) / block;
    const int warp_grid = static_cast<int>(
        std::max(1LL, std::min(65535LL, requested_warp_grid)));

    if (!cache_hit) {
        status = cudaMemset(
            ws.d_sum_scale.data(), 0, node_int_bytes);
        if (status != cudaSuccess) return fail(status);
        status = cudaMemset(ws.d_max_scale.data(), 0, node_int_bytes);
        if (status != cudaSuccess) return fail(status);
        status = cudaMemset(
            ws.d_common_product.data(), 0, edge_u64_bytes);
        if (status != cudaSuccess) return fail(status);
        status = cudaMemset(ws.d_indirect.data(), 0, edge_double_bytes);
        if (status != cudaSuccess) return fail(status);

        initialize_edge_statistics<<<node_grid, block>>>(
            directed_view.d_V,
            directed_view.d_E,
            forward_view.d_V,
            forward_view.d_E,
            ws.d_mutual.as<std::uint8_t>(),
            ws.d_sum_scale.as<int>(),
            ws.d_max_scale.as<int>(),
            num_nodes);
        status = cudaGetLastError();
        if (status != cudaSuccess) return fail(status);

        accumulate_triangle_statistics<<<warp_grid, block>>>(
            forward_view.d_V,
            forward_view.d_E,
            ws.d_mutual.as<std::uint8_t>(),
            ws.d_sum_scale.as<int>(),
            ws.d_common_product.as<unsigned long long>(),
            ws.d_indirect.as<double>(),
            num_nodes);
        status = cudaGetLastError();
        if (status != cudaSuccess) return fail(status);
    }

    status = cudaMemset(ws.d_result.data(), 0, node_double_bytes);
    if (status != cudaSuccess) return fail(status);

    if (metric == EgoMetric::EffectiveSize) {
        reduce_effective_size<<<warp_grid, block>>>(
            forward_view.d_V,
            forward_view.d_E,
            ws.d_sum_scale.as<int>(),
            ws.d_max_scale.as<int>(),
            ws.d_common_product.as<unsigned long long>(),
            ws.d_result.as<double>(),
            num_nodes);
    } else {
        double* constraint_target = ws.d_result.as<double>();
        if (metric == EgoMetric::Hierarchy) {
            status = cudaMemset(ws.d_aux.data(), 0, node_double_bytes);
            if (status != cudaSuccess) return fail(status);
            constraint_target = ws.d_aux.as<double>();
        }
        reduce_constraint<<<warp_grid, block>>>(
            forward_view.d_V,
            forward_view.d_E,
            ws.d_mutual.as<std::uint8_t>(),
            ws.d_sum_scale.as<int>(),
            ws.d_indirect.as<double>(),
            constraint_target,
            num_nodes);
        status = cudaGetLastError();
        if (status != cudaSuccess) return fail(status);

        if (metric == EgoMetric::Hierarchy) {
            reduce_hierarchy<<<warp_grid, block>>>(
                forward_view.d_V,
                forward_view.d_E,
                ws.d_degree.as<int>(),
                ws.d_mutual.as<std::uint8_t>(),
                ws.d_sum_scale.as<int>(),
                ws.d_indirect.as<double>(),
                ws.d_aux.as<double>(),
                ws.d_result.as<double>(),
                num_nodes);
        }
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) return fail(status);

    finalize_result<<<node_grid, block>>>(
        directed_view.d_V,
        ws.d_degree.as<int>(),
        metric == EgoMetric::Hierarchy ? ws.d_aux.as<double>() : nullptr,
        ws.d_result.as<double>(),
        num_nodes,
        static_cast<int>(metric));
    status = cudaGetLastError();
    if (status != cudaSuccess) return fail(status);

    status = cudaEventRecord(stop, 0);
    if (status != cudaSuccess) return fail(status);
    status = cudaEventSynchronize(stop);
    if (status != cudaSuccess) return fail(status);

    float elapsed_ms = 0.0f;
    status = cudaEventElapsedTime(&elapsed_ms, start, stop);
    if (status != cudaSuccess) return fail(status);
    if (kernel_seconds != nullptr) {
        *kernel_seconds = static_cast<double>(elapsed_ms) * 1e-3;
    }
    if (!cache_hit) ws.statistics_ready = true;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.resize(static_cast<std::size_t>(num_nodes));
    const int pinned_rc = ws.h_result.ensure_bytes(node_double_bytes);
    if (pinned_rc == EG_GPU_SUCC) {
        status = cudaMemcpy(
            ws.h_result.data(),
            ws.d_result.data(),
            node_double_bytes,
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return status_from_cuda(status);
        std::copy(
            ws.h_result.as<double>(),
            ws.h_result.as<double>() + num_nodes,
            result.begin());
    } else {
        status = cudaMemcpy(
            result.data(),
            ws.d_result.data(),
            node_double_bytes,
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return status_from_cuda(status);
    }
    return EG_GPU_SUCC;
}

} // namespace

int effective_size_ego_edge_statistics(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& forward_V,
    const std::vector<int>& forward_E,
    const std::vector<int>& degree,
    std::uint64_t graph_id,
    std::vector<double>& result,
    double* kernel_seconds
) {
    return run_ego_metric(
        V, E, forward_V, forward_E, degree, graph_id,
        EgoMetric::EffectiveSize, result, kernel_seconds);
}

int constraint_ego_edge_statistics(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& forward_V,
    const std::vector<int>& forward_E,
    const std::vector<int>& degree,
    std::uint64_t graph_id,
    std::vector<double>& result,
    double* kernel_seconds
) {
    return run_ego_metric(
        V, E, forward_V, forward_E, degree, graph_id,
        EgoMetric::Constraint, result, kernel_seconds);
}

int hierarchy_ego_edge_statistics(
    const std::vector<int>& V,
    const std::vector<int>& E,
    const std::vector<int>& forward_V,
    const std::vector<int>& forward_E,
    const std::vector<int>& degree,
    std::uint64_t graph_id,
    std::vector<double>& result,
    double* kernel_seconds
) {
    return run_ego_metric(
        V, E, forward_V, forward_E, degree, graph_id,
        EgoMetric::Hierarchy, result, kernel_seconds);
}

} // namespace gpu_easygraph
