#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"
#define NODES_PER_BLOCK 1

namespace gpu_easygraph {

enum norm_t { SUM = 0, MAX = 1 };

static __device__ bool has_out_edge(
    const int* V,
    const int* E,
    int u,
    int v
) {
    int lo = V[u];
    int hi = V[u + 1] - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        int value = E[mid];
        if (value == v) return true;
        if (value < v) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return false;
}

static __device__ double mutual_weight(
    const int* V,
    const int* E,
    const double* W,
    int u,
    int v
) {
    double a_uv = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        if (E[i] == v) {
            a_uv = W[i];
            break;
        }
    }
    return a_uv;
}

static __device__ double normalized_mutual_weight(
    const int* V,
    const int* E,
    const double* W, 
    int u,
    int v,
    norm_t norm
) {
    double weight_uv = mutual_weight(V, E, W, u, v);

    double scale = 0.0;
    if (norm == SUM) {
        for (int i = V[u]; i < V[u+1]; i++) {
            int neighbor = E[i];
            double weight_uw = mutual_weight(V, E, W, u, neighbor);
            scale += weight_uw;
        }
    } else if (norm == MAX) {
        for (int i = V[u]; i < V[u+1]; i++) {
            int neighbor = E[i];
            double weight_uw = mutual_weight(V, E, W, u, neighbor);
            scale = fmax(scale,weight_uw);
        }
    }
    return (scale==0.0) ? 0.0 : (weight_uv / scale);
}

static __device__ double directed_mutual_weight(
    const int* V,
    const int* E,
    const double* W,
    int u,
    int v
) {
    double a_uv = 0.0, a_vu = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        if (E[i] == v) {
            a_uv = W[i];
            break;
        }
    }
    for (int i = V[v]; i < V[v+1]; i++) {
        if (E[i] == u) {
            a_vu = W[i];
            break;
        }
    }
    return a_uv + a_vu;
}

static __device__ double directed_normalized_mutual_weight(
    const int* V,
    const int* E,
    const int* in_V,
    const int* in_E,
    const int* row, 
    const int* col, 
    const double* W, 
    int num_edges,
    int u,
    int v,
    norm_t norm
) {
    double weight_uv = directed_mutual_weight(V, E, W, u, v);

    double scale = 0.0;
    if(norm==SUM){
        for (int i = V[u]; i < V[u+1]; i++) {
            int neighbor = E[i];
            double weight_uw = directed_mutual_weight(V, E, W, u, neighbor);
            scale += weight_uw;
        }

        for (int i = in_V[u]; i < in_V[u + 1]; i++) {
            int neighbor = in_E[i];
            if (has_out_edge(V, E, u, neighbor)) continue;
            double weight_wu = directed_mutual_weight(V, E, W, u, neighbor);
            scale += weight_wu;
        }
    }else if(norm==MAX){
        for (int i = V[u]; i < V[u+1]; i++) {
            int neighbor = E[i];
            double weight_uw = directed_mutual_weight(V, E, W, u, neighbor);
            scale = fmax(scale,weight_uw);
        }

        for (int i = in_V[u]; i < in_V[u + 1]; i++) {
            int neighbor = in_E[i];
            if (has_out_edge(V, E, u, neighbor)) continue;
            double weight_wu = directed_mutual_weight(V, E, W, u, neighbor);
            scale = fmax(scale,weight_wu);
        }
    }
    return (scale==0.0) ? 0.0 : (weight_uv / scale);
}

static __device__ double redundancy(
    const int* V,
    const int* E,
    const double* W,
    const int num_nodes,
    int u,
    int v
) {
    double r = 0.0;
    for (int i = V[u]; i < V[u + 1]; i++) {
        int w = E[i];
        r += normalized_mutual_weight(V, E, W, u, w, SUM) * normalized_mutual_weight(V, E, W, v, w, MAX);
    }
    return 1-r;
}

static __device__ int directed_unweighted_mutual_count(
    const int* V,
    const int* E,
    int u,
    int v
) {
    int count = has_out_edge(V, E, u, v) ? 1 : 0;
    count += has_out_edge(V, E, v, u) ? 1 : 0;
    return count;
}

static __device__ double directed_unweighted_normalized_sum(
    const int* V,
    const int* E,
    const int* in_V,
    int u,
    int v
) {
    int scale = (V[u + 1] - V[u]) + (in_V[u + 1] - in_V[u]);
    if (scale <= 0) return 0.0;
    return (double)directed_unweighted_mutual_count(V, E, u, v) / (double)scale;
}

static __device__ double directed_unweighted_normalized_max(
    const int* V,
    const int* E,
    const int* in_V,
    int u,
    int v
) {
    int degree = (V[u + 1] - V[u]) + (in_V[u + 1] - in_V[u]);
    if (degree <= 0) return 0.0;

    int max_scale = 1;
    for (int i = V[u]; i < V[u + 1]; i++) {
        int w = E[i];
        if (has_out_edge(V, E, w, u)) {
            max_scale = 2;
            break;
        }
    }

    return (double)directed_unweighted_mutual_count(V, E, u, v) / (double)max_scale;
}

static __device__ double directed_unweighted_normalized_sum_fast(
    const int* V,
    const int* E,
    const int* sum_scale,
    int u,
    int v
) {
    int scale = sum_scale[u];
    if (scale <= 0) return 0.0;
    return (double)directed_unweighted_mutual_count(V, E, u, v) / (double)scale;
}

static __device__ double directed_unweighted_normalized_max_fast(
    const int* V,
    const int* E,
    const int* max_scale,
    int u,
    int v
) {
    int scale = max_scale[u];
    if (scale <= 0) return 0.0;
    return (double)directed_unweighted_mutual_count(V, E, u, v) / (double)scale;
}

static __device__ double directed_unweighted_redundancy_scan_v(
    const int* V,
    const int* E,
    const int* in_V,
    const int* in_E,
    const int* directed_sum_scale,
    const int* directed_max_scale,
    int u,
    int v
) {
    // For hub nodes, scanning all neighbors of u for every v degenerates to
    // O(deg(u)^2).  The same sum can be evaluated by scanning v's ego
    // neighbors and checking membership in u's neighborhood.
    int sum_scale_u = directed_sum_scale[u];
    int max_scale_v = directed_max_scale[v];
    if (sum_scale_u <= 0 || max_scale_v <= 0) return 1.0;

    double r = 0.0;
    for (int i = V[v]; i < V[v + 1]; ++i) {
        int w = E[i];
        int count_uw = directed_unweighted_mutual_count(V, E, u, w);
        if (count_uw == 0) continue;
        int count_vw = directed_unweighted_mutual_count(V, E, v, w);
        r += ((double)count_uw / (double)sum_scale_u) *
             ((double)count_vw / (double)max_scale_v);
    }
    for (int i = in_V[v]; i < in_V[v + 1]; ++i) {
        int w = in_E[i];
        if (has_out_edge(V, E, v, w)) continue;
        int count_uw = directed_unweighted_mutual_count(V, E, u, w);
        if (count_uw == 0) continue;
        int count_vw = directed_unweighted_mutual_count(V, E, v, w);
        r += ((double)count_uw / (double)sum_scale_u) *
             ((double)count_vw / (double)max_scale_v);
    }
    return 1.0 - r;
}

__global__ void directed_unweighted_scale_precompute(
    const int* V,
    const int* E,
    const int* in_V,
    const int* in_E,
    int num_nodes,
    int* sum_scale,
    int* max_scale
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;

    int out_degree = V[u + 1] - V[u];
    int in_degree = in_V[u + 1] - in_V[u];
    sum_scale[u] = out_degree + in_degree;

    int max_value = (sum_scale[u] > 0) ? 1 : 0;
    for (int i = V[u]; i < V[u + 1]; ++i) {
        int w = E[i];
        if (has_out_edge(V, E, w, u)) {
            max_value = 2;
            break;
        }
    }
    max_scale[u] = max_value;
}


__inline__ __device__ double warp_reduce_sum(double val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ double block_reduce_sum(double val)
{
    val = warp_reduce_sum(val);

    __shared__ double shared[32];
    int warp_id = threadIdx.x / warpSize;
    if (threadIdx.x % warpSize == 0)
    {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        val = (threadIdx.x < (blockDim.x / warpSize)) ? shared[threadIdx.x] : 0.0;
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void calculate_effective_size(
    const int* __restrict__ V,
    const int* __restrict__ E,
    const double* __restrict__ W,
    const int num_nodes,
    const int* __restrict__ node_mask,
    double* __restrict__ effective_size_results,
    bool is_unweighted
) {
    int u = blockIdx.x;
    if (u >= num_nodes || !node_mask[u]) return;

    int neighbor_start = V[u];
    int neighbor_end = V[u + 1];
    int degree = neighbor_end - neighbor_start;

    int threads_per_block = blockDim.x;

    if (degree == 0) {
        if (threadIdx.x == 0) effective_size_results[u] = NAN;
        return;
    }

    double redundancy_sum = 0.0;
    if (is_unweighted) {
        double neighbor_edges = 0.0;
        for (int idx = threadIdx.x; idx < degree; idx += threads_per_block) {
            int v = E[neighbor_start + idx];
            for (int j = V[v]; j < V[v + 1]; j++) {
                int w = E[j];
                if (has_out_edge(V, E, u, w)) {
                    neighbor_edges += 1.0;
                }
            }
        }
        neighbor_edges = block_reduce_sum(neighbor_edges);
        if (threadIdx.x == 0) {
            effective_size_results[u] = (double)degree - neighbor_edges / (double)degree;
        }
        return;
    }

    for (int idx = threadIdx.x; idx < degree; idx += threads_per_block) {
        int i = neighbor_start + idx;
        int v = E[i];
        if (v != u) {
            double r = 0.0;
            for (int j = V[v]; j < V[v + 1]; j++) {
                int w = E[j];
                r += normalized_mutual_weight(V, E, W, u, w, SUM) * 
                     normalized_mutual_weight(V, E, W, v, w, MAX);
            }
            redundancy_sum += 1 - r;
        }
    }
    redundancy_sum = block_reduce_sum(redundancy_sum);

    if (threadIdx.x == 0) {
        effective_size_results[u] = redundancy_sum;
    }
}

static __device__ double directed_redundancy(
    const int* V,
    const int* E,
    const int* in_V,
    const int* in_E,
    const int* row,
    const int* col,
    const double* W,
    const int num_nodes,
    const int num_edges,
    int u,
    int v,
    bool is_unweighted,
    const int* directed_sum_scale,
    const int* directed_max_scale
) {
    if (is_unweighted) {
        int degree_u = directed_sum_scale ? directed_sum_scale[u] : 0;
        int degree_v = directed_sum_scale ? directed_sum_scale[v] : 0;
        if (degree_v > 0 && (degree_u >= 256 || degree_v < degree_u)) {
            return directed_unweighted_redundancy_scan_v(
                V, E, in_V, in_E, directed_sum_scale, directed_max_scale, u, v);
        }
    }

    double r = 0.0;
    for (int i = V[u]; i < V[u + 1]; i++) {
        int w = E[i];
        double sum_factor = is_unweighted
            ? directed_unweighted_normalized_sum_fast(V, E, directed_sum_scale, u, w)
            : directed_normalized_mutual_weight(V, E, in_V, in_E, row,col,W,num_edges, u, w,SUM);
        double max_factor = is_unweighted
            ? directed_unweighted_normalized_max_fast(V, E, directed_max_scale, v, w)
            : directed_normalized_mutual_weight(V, E, in_V, in_E, row,col,W, num_edges, v,w,MAX);
        r += sum_factor * max_factor;
    }
    for (int i = in_V[u]; i < in_V[u + 1]; i++) {
        int w = in_E[i];
        if (has_out_edge(V, E, u, w)) continue;
        double sum_factor = is_unweighted
            ? directed_unweighted_normalized_sum_fast(V, E, directed_sum_scale, u, w)
            : directed_normalized_mutual_weight(V, E, in_V, in_E, row,col,W,num_edges, u, w,SUM);
        double max_factor = is_unweighted
            ? directed_unweighted_normalized_max_fast(V, E, directed_max_scale, v, w)
            : directed_normalized_mutual_weight(V, E, in_V, in_E, row,col,W, num_edges, v,w,MAX);
        r += sum_factor * max_factor;
    }
    return 1-r;
}

__global__ void directed_calculate_effective_size(
    const int* V,
    const int* E,
    const int* in_V,
    const int* in_E,
    const int* row,
    const int* col,
    const double* W, 
    const int num_nodes,
    const int num_edges,
    const int* node_mask,
    double* effective_size_results,
    bool is_unweighted,
    const int* directed_sum_scale,
    const int* directed_max_scale
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes || !node_mask[u]) return;

    // Match EasyGraph's directed effective_size guard: G[u] is the outgoing
    // adjacency, so nodes with no outgoing edges return NaN even if they have
    // incoming neighbors.
    if (V[u + 1] == V[u]) {
        effective_size_results[u] = NAN;
        return;
    }

    double redundancy_sum = 0.0;
    bool is_nan = true;

    for (int i = V[u]; i < V[u + 1]; i++) {
        int v = E[i];
        if (v == u) continue;
        is_nan = false;
        redundancy_sum += directed_redundancy(V,E,in_V,in_E,row,col,W,num_nodes,num_edges,u,v,is_unweighted,directed_sum_scale,directed_max_scale);
    }
    for (int i = in_V[u]; i < in_V[u + 1]; i++) {
        int v = in_E[i];
        if (has_out_edge(V, E, u, v)) continue;
        is_nan = false;
        redundancy_sum += directed_redundancy(V,E,in_V,in_E,row,col,W,num_nodes,num_edges,u,v,is_unweighted,directed_sum_scale,directed_max_scale);
    }
    effective_size_results[u] = is_nan ? NAN : redundancy_sum;
}

__global__ void directed_unweighted_calculate_effective_size_block(
    const int* __restrict__ V,
    const int* __restrict__ E,
    const int* __restrict__ in_V,
    const int* __restrict__ in_E,
    const int num_nodes,
    const int* __restrict__ node_mask,
    double* __restrict__ effective_size_results,
    const int* __restrict__ directed_sum_scale,
    const int* __restrict__ directed_max_scale
) {
    int u = blockIdx.x;
    if (u >= num_nodes || !node_mask[u]) return;

    // Preserve EasyGraph's directed semantics: nodes without outgoing edges
    // return NaN even when they have incoming neighbors.
    if (V[u + 1] == V[u]) {
        if (threadIdx.x == 0) effective_size_results[u] = NAN;
        return;
    }

    double redundancy_sum = 0.0;
    bool saw_neighbor = false;

    int out_start = V[u];
    int out_end = V[u + 1];
    int out_degree = out_end - out_start;
    for (int idx = threadIdx.x; idx < out_degree; idx += blockDim.x) {
        int v = E[out_start + idx];
        if (v == u) continue;
        saw_neighbor = true;
        redundancy_sum += directed_redundancy(
            V, E, in_V, in_E, nullptr, nullptr, nullptr, num_nodes, 0,
            u, v, true, directed_sum_scale, directed_max_scale);
    }

    int in_start = in_V[u];
    int in_end = in_V[u + 1];
    int in_degree = in_end - in_start;
    for (int idx = threadIdx.x; idx < in_degree; idx += blockDim.x) {
        int v = in_E[in_start + idx];
        if (has_out_edge(V, E, u, v)) continue;
        saw_neighbor = true;
        redundancy_sum += directed_redundancy(
            V, E, in_V, in_E, nullptr, nullptr, nullptr, num_nodes, 0,
            u, v, true, directed_sum_scale, directed_max_scale);
    }

    int local_seen = saw_neighbor ? 1 : 0;
    int any_seen = __syncthreads_or(local_seen);
    redundancy_sum = block_reduce_sum(redundancy_sum);
    if (threadIdx.x == 0) {
        effective_size_results[u] = any_seen ? redundancy_sum : NAN;
    }
}

static bool should_use_directed_unweighted_block_kernel(
    const int* V,
    int num_nodes,
    int num_edges
) {
    int max_out_degree = 0;
    long long active_nodes = 0;
    for (int u = 0; u < num_nodes; ++u) {
        int degree = V[u + 1] - V[u];
        if (degree > 0) ++active_nodes;
        if (degree > max_out_degree) max_out_degree = degree;
    }
    // The block-per-node path pays more fixed scheduling cost, so keep the old
    // one-thread-per-node kernel for tiny/sparse directed graphs.
    return max_out_degree >= 256 || (num_edges >= 50000 && active_nodes >= 1024);
}


int cuda_effective_size(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const int* in_V,
    _IN_ const int* in_E,
    _IN_ const int* row,
    _IN_ const int* col,
    _IN_ const double* W,
    _IN_ int num_nodes,
    _IN_ int num_edges,
    _IN_ bool is_directed,
    _IN_ bool is_unweighted,
    _IN_ int* node_mask,
    _OUT_ double* effective_size_results,
    _OUT_ double* kernel_seconds
) {
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    int block_size = 128;
    int grid_size = (num_nodes + block_size - 1) / block_size;

    int* d_V;
    int* d_E;
    int* d_in_V;
    int* d_in_E;
    int* d_row;
    int* d_col;
    double* d_W;
    int* d_node_mask;
    int* d_directed_sum_scale = nullptr;
    int* d_directed_max_scale = nullptr;
    double* d_effective_size_results;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, (num_nodes+1) * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_in_V, (num_nodes+1) * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_in_E, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_row, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_col, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, num_edges * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_node_mask, num_nodes * sizeof(int)));
    if (is_directed && is_unweighted) {
        EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_directed_sum_scale, num_nodes * sizeof(int)));
        EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_directed_max_scale, num_nodes * sizeof(int)));
    }
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_effective_size_results, num_nodes * sizeof(double)));

    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, (num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_in_V, in_V, (num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_in_E, in_E, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_row, row, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_col, col, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_node_mask, node_mask, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, num_edges * sizeof(double), cudaMemcpyHostToDevice));

    EXIT_IF_CUDA_FAILED(cudaEventCreate(&kernel_start));
    EXIT_IF_CUDA_FAILED(cudaEventCreate(&kernel_stop));
    EXIT_IF_CUDA_FAILED(cudaEventRecord(kernel_start, 0));

    if(is_directed){
        if (is_unweighted) {
            directed_unweighted_scale_precompute<<<grid_size, block_size>>>(
                d_V, d_E, d_in_V, d_in_E, num_nodes, d_directed_sum_scale, d_directed_max_scale);
            EXIT_IF_CUDA_FAILED(cudaGetLastError());
            if (should_use_directed_unweighted_block_kernel(V, num_nodes, num_edges)) {
                directed_unweighted_calculate_effective_size_block<<<num_nodes, block_size>>>(
                    d_V, d_E, d_in_V, d_in_E, num_nodes, d_node_mask,
                    d_effective_size_results, d_directed_sum_scale, d_directed_max_scale);
            } else {
                directed_calculate_effective_size<<<grid_size, block_size>>>(
                    d_V, d_E, d_in_V, d_in_E, d_row, d_col, d_W, num_nodes, num_edges,
                    d_node_mask, d_effective_size_results, is_unweighted,
                    d_directed_sum_scale, d_directed_max_scale);
            }
        } else {
            directed_calculate_effective_size<<<grid_size, block_size>>>(
                d_V, d_E, d_in_V, d_in_E, d_row, d_col, d_W, num_nodes, num_edges,
                d_node_mask, d_effective_size_results, is_unweighted,
                d_directed_sum_scale, d_directed_max_scale);
        }
    }else{
        int block_size = 128; 
        int grid_size = (num_nodes + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK;
        calculate_effective_size<<<grid_size, block_size>>>(d_V, d_E, d_W, num_nodes, d_node_mask, d_effective_size_results, is_unweighted);
        
    }
    EXIT_IF_CUDA_FAILED(cudaGetLastError());
    EXIT_IF_CUDA_FAILED(cudaEventRecord(kernel_stop, 0));
    EXIT_IF_CUDA_FAILED(cudaEventSynchronize(kernel_stop));
    if (kernel_seconds != nullptr) {
        float elapsed_ms = 0.0f;
        EXIT_IF_CUDA_FAILED(cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop));
        *kernel_seconds = (double)elapsed_ms * 1e-3;
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(effective_size_results, d_effective_size_results, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));

exit:
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_in_V);
    cudaFree(d_in_E);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_W);
    cudaFree(d_node_mask);
    cudaFree(d_directed_sum_scale);
    cudaFree(d_directed_max_scale);
    if (kernel_start != nullptr) cudaEventDestroy(kernel_start);
    if (kernel_stop != nullptr) cudaEventDestroy(kernel_stop);
    cudaFree(d_effective_size_results);

    if (cuda_ret != cudaSuccess) {
        switch (cuda_ret) {
            case cudaErrorMemoryAllocation:
                EG_ret = EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM;
                break;
            default:
                EG_ret = EG_GPU_DEVICE_ERR;
                break;
        }
    }

    return EG_ret;
}

} // namespace gpu_easygraph
