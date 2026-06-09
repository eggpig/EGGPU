#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <stdlib.h>

#include "common.h"
#define NODES_PER_BLOCK 1

namespace gpu_easygraph {

enum norm_t { SUM = 0, MAX = 1 };

static inline bool env_flag_enabled(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return default_value;
    if (raw[0] == '\0') return default_value;
    if (raw[0] == '0') return false;
    if (raw[0] == 'f' || raw[0] == 'F' || raw[0] == 'n' || raw[0] == 'N') return false;
    return true;
}

static inline bool constraint_use_smaller_intersection(
    int num_nodes,
    int num_edges,
    bool is_directed
) {
    const char* raw = std::getenv("EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION");
    if (raw != nullptr && raw[0] != '\0') {
        if (std::strcmp(raw, "AUTO") != 0 && std::strcmp(raw, "auto") != 0) {
            return env_flag_enabled("EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION", true);
        }
    }
    if (is_directed || num_nodes <= 0) return false;

    // The smaller-side intersection path is best on low-degree graphs, where it
    // cuts binary-search probes substantially.  On denser medium graphs the
    // extra branch and reversed lookup pattern can lose to the original scan.
    const double avg_slots_per_node = (double)num_edges / (double)num_nodes;
    return avg_slots_per_node <= 7.5 || num_nodes >= 500000;
}

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

static __device__ double unweighted_local_constraint(
    const int* V,
    const int* E,
    int u,
    int v,
    bool use_smaller_intersection
) {
    int deg_u = V[u + 1] - V[u];
    if (deg_u <= 0) return 0.0;

    double common_sum = 0.0;
    int deg_v = V[v + 1] - V[v];

    if (use_smaller_intersection && deg_v > 0 && deg_v < deg_u) {
        for (int i = V[v]; i < V[v + 1]; i++) {
            int w = E[i];
            if (has_out_edge(V, E, u, w)) {
                int deg_w = V[w + 1] - V[w];
                if (deg_w > 0) common_sum += 1.0 / (double)deg_w;
            }
        }
    } else {
        for (int i = V[u]; i < V[u + 1]; i++) {
            int w = E[i];
            if (has_out_edge(V, E, w, v)) {
                int deg_w = V[w + 1] - V[w];
                if (deg_w > 0) common_sum += 1.0 / (double)deg_w;
            }
        }
    }

    double value = (1.0 + common_sum) / (double)deg_u;
    return value * value;
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

static __device__ double local_constraint(
    const int* V,
    const int* E,
    const double* W,
    int u,
    int v
) {
    double direct = normalized_mutual_weight(V,E,W,u,v,SUM);
    double indirect = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        int neighbor = E[i];
        double norm_uw = normalized_mutual_weight(V, E, W, u, neighbor,SUM);
        double norm_wv = normalized_mutual_weight(V, E, W, neighbor, v,SUM);
        indirect += norm_uw * norm_wv;
    }
    double local_constraint_of_uv = (direct + indirect) * (direct + indirect);
    return local_constraint_of_uv;
}

__global__ void calculate_constraints(
    const int* __restrict__ V,
    const int* __restrict__ E,
    const double* __restrict__ W, 
    const int num_nodes, 
    const int* __restrict__ node_mask,
    double* __restrict__ constraint_results,
    bool is_unweighted,
    bool use_smaller_intersection
) {
    int start_node = blockIdx.x * NODES_PER_BLOCK;
    int end_node = min(start_node + NODES_PER_BLOCK, num_nodes);

    for (int v = start_node; v < end_node; ++v) {
        if (!node_mask[v]) continue;

        // Match EasyGraph's constraint guard: nodes with empty G[v] return NaN.
        if (V[v + 1] == V[v]) {
            if (threadIdx.x == 0) constraint_results[v] = NAN;
            continue;
        }

        double constraint_of_v = 0.0;
        bool is_nan = true;

        __shared__ double shared_constraint[256];
        double local_sum = 0.0;

        for (int i = V[v] + threadIdx.x; i < V[v + 1]; i += blockDim.x) {
            is_nan = false;
            int neighbor = E[i];
            local_sum += is_unweighted
                ? unweighted_local_constraint(V, E, v, neighbor, use_smaller_intersection)
                : local_constraint(V, E, W, v, neighbor);
        }

        shared_constraint[threadIdx.x] = local_sum;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                shared_constraint[threadIdx.x] += shared_constraint[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            constraint_results[v] = (is_nan) ? NAN : shared_constraint[0];
        }
    }
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

static __device__ double directed_unweighted_local_constraint_scan_v(
    const int* V,
    const int* E,
    const int* in_V,
    const int* in_E,
    const int* directed_sum_scale,
    int u,
    int v
) {
    // Equivalent to summing over all neighbors of u, but much cheaper for
    // high-degree u because only nodes adjacent to v can contribute.
    int scale_u = directed_sum_scale[u];
    if (scale_u <= 0) return 0.0;

    double direct = (double)directed_unweighted_mutual_count(V, E, u, v) /
                    (double)scale_u;
    double indirect = 0.0;

    for (int i = V[v]; i < V[v + 1]; ++i) {
        int w = E[i];
        int count_uw = directed_unweighted_mutual_count(V, E, u, w);
        if (count_uw == 0) continue;
        int scale_w = directed_sum_scale[w];
        if (scale_w <= 0) continue;
        int count_wv = directed_unweighted_mutual_count(V, E, w, v);
        indirect += ((double)count_uw / (double)scale_u) *
                    ((double)count_wv / (double)scale_w);
    }
    for (int i = in_V[v]; i < in_V[v + 1]; ++i) {
        int w = in_E[i];
        if (has_out_edge(V, E, v, w)) continue;
        int count_uw = directed_unweighted_mutual_count(V, E, u, w);
        if (count_uw == 0) continue;
        int scale_w = directed_sum_scale[w];
        if (scale_w <= 0) continue;
        int count_wv = directed_unweighted_mutual_count(V, E, w, v);
        indirect += ((double)count_uw / (double)scale_u) *
                    ((double)count_wv / (double)scale_w);
    }

    double value = direct + indirect;
    return value * value;
}

__global__ void constraint_directed_unweighted_sum_scale_precompute(
    const int* V,
    const int* in_V,
    int num_nodes,
    int* sum_scale
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;
    sum_scale[u] = (V[u + 1] - V[u]) + (in_V[u + 1] - in_V[u]);
}

static __device__ double directed_local_constraint(
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
    bool is_unweighted,
    const int* directed_sum_scale
) {
    if (is_unweighted) {
        int degree_u = directed_sum_scale ? directed_sum_scale[u] : 0;
        int degree_v = directed_sum_scale ? directed_sum_scale[v] : 0;
        if (degree_v > 0 && (degree_u >= 256 || degree_v < degree_u)) {
            return directed_unweighted_local_constraint_scan_v(
                V, E, in_V, in_E, directed_sum_scale, u, v);
        }
    }

    double direct = is_unweighted
        ? directed_unweighted_normalized_sum_fast(V, E, directed_sum_scale, u, v)
        : directed_normalized_mutual_weight(V,E,in_V,in_E,row,col,W,num_edges,u,v,SUM);
    double indirect = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        int neighbor = E[i];
        double norm_uw = is_unweighted
            ? directed_unweighted_normalized_sum_fast(V, E, directed_sum_scale, u, neighbor)
            : directed_normalized_mutual_weight(V, E, in_V, in_E, row, col, W, num_edges, u, neighbor,SUM);
        double norm_wv = is_unweighted
            ? directed_unweighted_normalized_sum_fast(V, E, directed_sum_scale, neighbor, v)
            : directed_normalized_mutual_weight(V, E, in_V, in_E, row, col, W, num_edges, neighbor, v,SUM);
        indirect += norm_uw * norm_wv;
    }

    for (int i = in_V[u]; i < in_V[u + 1]; i++) {
        int neighbor = in_E[i];
        if (has_out_edge(V, E, u, neighbor)) continue;
        double norm_uw = is_unweighted
            ? directed_unweighted_normalized_sum_fast(V, E, directed_sum_scale, u, neighbor)
            : directed_normalized_mutual_weight(V, E, in_V, in_E, row, col, W, num_edges, u, neighbor,SUM);
        double norm_wv = is_unweighted
            ? directed_unweighted_normalized_sum_fast(V, E, directed_sum_scale, neighbor, v)
            : directed_normalized_mutual_weight(V, E, in_V, in_E, row, col, W, num_edges, neighbor, v,SUM);
        indirect += norm_uw * norm_wv;
    }
    double local_constraint_of_uv = (direct + indirect) * (direct + indirect);
    return local_constraint_of_uv;
}

__global__ void directed_calculate_constraints(
    const int* V,
    const int* E,
    const int* in_V,
    const int* in_E,
    const int* row, 
    const int* col, 
    const double* W,  
    int num_nodes,
    int num_edges,
    int* node_mask,
    double* constraint_results,
    bool is_unweighted,
    const int* directed_sum_scale
) {
    int start_node = blockIdx.x * NODES_PER_BLOCK;
    int end_node = min(start_node + NODES_PER_BLOCK, num_nodes);

    for (int v = start_node; v < end_node; ++v) {
        if (!node_mask[v]) continue;

        // For directed graphs EasyGraph checks the outgoing adjacency G[v],
        // not the union of predecessors and successors.
        if (V[v + 1] == V[v]) {
            if (threadIdx.x == 0) constraint_results[v] = NAN;
            continue;
        }

        double constraint_of_v = 0.0;
        bool is_nan = true;

        __shared__ double shared_constraint[256];
        double local_sum = 0.0;

        for (int i = V[v] + threadIdx.x; i < V[v + 1]; i += blockDim.x) {
            is_nan = false;
            int neighbor = E[i];
            local_sum += directed_local_constraint(V, E, in_V, in_E, row, col, W, num_edges, v, neighbor, is_unweighted, directed_sum_scale);
        }

        for (int i = in_V[v] + threadIdx.x; i < in_V[v + 1]; i += blockDim.x) {
            int neighbor = in_E[i];
            if (has_out_edge(V, E, v, neighbor)) continue;
            is_nan = false;
            local_sum += directed_local_constraint(V, E, in_V, in_E, row, col, W, num_edges, v, neighbor, is_unweighted, directed_sum_scale);
        }

        shared_constraint[threadIdx.x] = local_sum;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                shared_constraint[threadIdx.x] += shared_constraint[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            constraint_results[v] = (is_nan) ? NAN : shared_constraint[0];
        }
    }
}


int cuda_constraint(
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
    _OUT_ double* constraint_results,
    _OUT_ double* kernel_seconds
) {
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    
    int* d_V;
    int* d_E;
    int* d_in_V;
    int* d_in_E;
    int* d_row;
    int* d_col;
    double* d_W;
    int* d_node_mask;
    int* d_directed_sum_scale = nullptr;
    double* d_constraint_results;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    int block_size = 128;
    int grid_size = (num_nodes + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK;
    bool use_smaller_intersection = constraint_use_smaller_intersection(
        num_nodes, num_edges, is_directed);

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
    }
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_constraint_results, num_nodes * sizeof(double)));

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
            int scale_grid = (num_nodes + block_size - 1) / block_size;
            constraint_directed_unweighted_sum_scale_precompute<<<scale_grid, block_size>>>(
                d_V, d_in_V, num_nodes, d_directed_sum_scale);
            EXIT_IF_CUDA_FAILED(cudaGetLastError());
        }
        directed_calculate_constraints<<<grid_size, block_size>>>(d_V, d_E, d_in_V, d_in_E, d_row, d_col, d_W, num_nodes, num_edges, d_node_mask, d_constraint_results, is_unweighted, d_directed_sum_scale);
    }else{
        calculate_constraints<<<grid_size, block_size>>>(d_V, d_E, d_W, num_nodes, d_node_mask, d_constraint_results, is_unweighted, use_smaller_intersection);
    }
    EXIT_IF_CUDA_FAILED(cudaGetLastError());
    EXIT_IF_CUDA_FAILED(cudaEventRecord(kernel_stop, 0));
    EXIT_IF_CUDA_FAILED(cudaEventSynchronize(kernel_stop));
    if (kernel_seconds != nullptr) {
        float elapsed_ms = 0.0f;
        EXIT_IF_CUDA_FAILED(cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop));
        *kernel_seconds = (double)elapsed_ms * 1e-3;
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(constraint_results, d_constraint_results, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));
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
    if (kernel_start != nullptr) cudaEventDestroy(kernel_start);
    if (kernel_stop != nullptr) cudaEventDestroy(kernel_stop);
    cudaFree(d_constraint_results);
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
