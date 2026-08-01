#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <vector>

#include "common.h"
#include "device_graph_cache.h"
#define NODES_PER_BLOCK 1

namespace gpu_easygraph {

enum norm_t { SUM = 0, MAX = 1 };
static constexpr int DIRECTED_BLOCK_DEGREE_THRESHOLD = 128;

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
    int v
) {
    int deg_u = V[u + 1] - V[u];
    if (deg_u <= 0) return 0.0;

    double common_sum = 0.0;
    for (int i = V[u]; i < V[u + 1]; i++) {
        int w = E[i];
        if (has_out_edge(V, E, w, v)) {
            int deg_w = V[w + 1] - V[w];
            if (deg_w > 0) common_sum += 1.0 / (double)deg_w;
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
    // Equivalent local-constraint sum with neighbor-side scanning.  This avoids
    // quadratic hub-node scans on sparse, skewed directed graphs.
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

__global__ void hierarchy_directed_unweighted_sum_scale_precompute(
    const int* V,
    const int* in_V,
    int num_nodes,
    int* sum_scale
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;
    sum_scale[u] = (V[u + 1] - V[u]) + (in_V[u + 1] - in_V[u]);
}

static __device__ double atomicAdd (
    _OUT_ double* address, 
    _IN_ double val
)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
			__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
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

__global__ void calculate_hierarchy(
    const int* V, 
    const int* E, 
    const double* W,
    int num_nodes,
    const int* node_mask,
    double* hierarchy_results,
    bool is_unweighted
) {
    int v = blockIdx.x;
    if (v >= num_nodes || !node_mask[v]) return;

    extern __shared__ double shared[];

    int n = V[v + 1] - V[v]; 
    if (n <= 1) {
        if (threadIdx.x == 0) hierarchy_results[v] = 0.0;
        return;
    }

    double local_C = 0.0;
    for (int i = V[v] + threadIdx.x; i < V[v + 1]; i += blockDim.x) {
        int w = E[i];
        local_C += is_unweighted
            ? unweighted_local_constraint(V, E, v, w)
            : local_constraint(V, E, W, v, w);
    }
    shared[threadIdx.x] = local_C;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    double C = shared[0];
    if (C <= 0.0) {
        if (threadIdx.x == 0) hierarchy_results[v] = 0.0;
        return;
    }

    double local_H = 0.0;
    double log_n = log((double)n);
    for (int i = V[v] + threadIdx.x; i < V[v + 1]; i += blockDim.x) {
        int w = E[i];
        double c = is_unweighted
            ? unweighted_local_constraint(V, E, v, w)
            : local_constraint(V, E, W, v, w);
        if (c > 0.0) {
            double p = c / C;
            local_H += p * log(p * (double)n) / log_n;
        }
    }
    shared[threadIdx.x] = local_H;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) hierarchy_results[v] = shared[0]; 
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

__global__ void directed_unweighted_calculate_hierarchy_thread(
    const int* __restrict__ V,
    const int* __restrict__ E,
    const int* __restrict__ in_V,
    const int* __restrict__ in_E,
    int num_nodes,
    const int* __restrict__ node_mask,
    double* __restrict__ hierarchy_results,
    const int* __restrict__ directed_sum_scale,
    int block_degree_threshold
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes || !node_mask[v]) return;

    int combined_degree =
        (V[v + 1] - V[v]) + (in_V[v + 1] - in_V[v]);
    if (combined_degree >= block_degree_threshold) return;

    int neighbor_count = V[v + 1] - V[v];
    for (int i = in_V[v]; i < in_V[v + 1]; ++i) {
        int w = in_E[i];
        if (!has_out_edge(V, E, v, w)) ++neighbor_count;
    }
    if (neighbor_count <= 1) {
        hierarchy_results[v] = 0.0;
        return;
    }

    double C = 0.0;
    for (int i = V[v]; i < V[v + 1]; ++i) {
        int w = E[i];
        C += directed_local_constraint(
            V, E, in_V, in_E, nullptr, nullptr, nullptr, 0, v, w, true,
            directed_sum_scale);
    }
    for (int i = in_V[v]; i < in_V[v + 1]; ++i) {
        int w = in_E[i];
        if (!has_out_edge(V, E, v, w)) {
            C += directed_local_constraint(
                V, E, in_V, in_E, nullptr, nullptr, nullptr, 0, v, w, true,
                directed_sum_scale);
        }
    }
    if (C <= 0.0) {
        hierarchy_results[v] = 0.0;
        return;
    }

    double H = 0.0;
    double log_n = log((double)neighbor_count);
    for (int i = V[v]; i < V[v + 1]; ++i) {
        int w = E[i];
        double c = directed_local_constraint(
            V, E, in_V, in_E, nullptr, nullptr, nullptr, 0, v, w, true,
            directed_sum_scale);
        if (c > 0.0) {
            double p = c / C;
            H += p * log(p * (double)neighbor_count) / log_n;
        }
    }
    for (int i = in_V[v]; i < in_V[v + 1]; ++i) {
        int w = in_E[i];
        if (!has_out_edge(V, E, v, w)) {
            double c = directed_local_constraint(
                V, E, in_V, in_E, nullptr, nullptr, nullptr, 0, v, w, true,
                directed_sum_scale);
            if (c > 0.0) {
                double p = c / C;
                H += p * log(p * (double)neighbor_count) / log_n;
            }
        }
    }
    hierarchy_results[v] = H;
}

__global__ void directed_calculate_hierarchy(
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
    double* hierarchy_results,
    bool is_unweighted,
    const int* directed_sum_scale,
    const int* block_nodes,
    int block_node_count
) {
    int work_index = blockIdx.x;
    if (block_nodes != nullptr && work_index >= block_node_count) return;
    int v = block_nodes == nullptr ? work_index : block_nodes[work_index];
    if (v >= num_nodes || !node_mask[v]) return;

    extern __shared__ double shared[];

    int neighbor_count = V[v + 1] - V[v];  
    for (int i = in_V[v]; i < in_V[v + 1]; i++) {
        int w = in_E[i];
        if (!has_out_edge(V, E, v, w)) neighbor_count++;
    }

    if (neighbor_count <= 1) {
        if (threadIdx.x == 0) hierarchy_results[v] = 0.0;
        return;
    }

    double local_C = 0.0;
    for (int i = V[v] + threadIdx.x; i < V[v + 1]; i += blockDim.x) {
        int w = E[i];
        local_C += directed_local_constraint(V, E, in_V, in_E, row, col, W, num_edges, v, w, is_unweighted, directed_sum_scale);
    }
    for (int i = in_V[v] + threadIdx.x; i < in_V[v + 1]; i += blockDim.x) {
        int w = in_E[i];
        if (!has_out_edge(V, E, v, w)) {
            local_C += directed_local_constraint(V, E, in_V, in_E, row, col, W, num_edges, v, w, is_unweighted, directed_sum_scale);
        }
    }

    shared[threadIdx.x] = local_C;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    double C = shared[0];
    if (C <= 0.0) {
        if (threadIdx.x == 0) hierarchy_results[v] = 0.0;
        return;
    }

    double local_H = 0.0;
    double log_n = log((double)neighbor_count);
    for (int i = V[v] + threadIdx.x; i < V[v + 1]; i += blockDim.x) {
        int w = E[i];
        double c = directed_local_constraint(V, E, in_V, in_E, row, col, W, num_edges, v, w, is_unweighted, directed_sum_scale);
        if (c > 0.0) {
            double p = c / C;
            local_H += p * log(p * (double)neighbor_count) / log_n;
        }
    }
    for (int i = in_V[v] + threadIdx.x; i < in_V[v + 1]; i += blockDim.x) {
        int w = in_E[i];
        if (!has_out_edge(V, E, v, w)) {
            double c = directed_local_constraint(V, E, in_V, in_E, row, col, W, num_edges, v, w, is_unweighted, directed_sum_scale);
            if (c > 0.0) {
                double p = c / C;
                local_H += p * log(p * (double)neighbor_count) / log_n;
            }
        }
    }

    shared[threadIdx.x] = local_H;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) hierarchy_results[v] = shared[0];
}

static std::vector<int> directed_unweighted_block_nodes(
    const int* V,
    const int* in_V,
    const int* node_mask,
    int num_nodes,
    int degree_threshold
) {
    std::vector<int> nodes;
    nodes.reserve((size_t)num_nodes / 64 + 1);
    for (int v = 0; v < num_nodes; ++v) {
        if (!node_mask[v]) continue;
        long long combined_degree =
            (long long)(V[v + 1] - V[v]) +
            (long long)(in_V[v + 1] - in_V[v]);
        if (combined_degree >= degree_threshold) nodes.push_back(v);
    }
    return nodes;
}

int cuda_hierarchy(
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
    _OUT_ double* hierarchy_results,
    _OUT_ double* kernel_seconds
){
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    int min_grid_size = 0;
    int block_size = 0;
    
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, calculate_hierarchy, 0, 0);
    int grid_size = (num_nodes + block_size - 1) / block_size;
    
    int* d_V = nullptr;
    int* d_E = nullptr;
    int* d_in_V = nullptr;
    int* d_in_E = nullptr;
    int* d_row = nullptr;
    int* d_col = nullptr;
    double* d_W = nullptr;
    int* d_node_mask = nullptr;
    int* d_directed_sum_scale = nullptr;
    int* d_block_nodes = nullptr;
    double* d_hierarchy_results = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    DeviceCsrView out_view;
    DeviceCsrView in_view;
    const bool use_cached_csr = is_unweighted;
    std::vector<int> block_nodes;
    if (is_directed && is_unweighted) {
        block_nodes = directed_unweighted_block_nodes(
            V, in_V, node_mask, num_nodes, DIRECTED_BLOCK_DEGREE_THRESHOLD);
    }

    if (use_cached_csr) {
        EG_ret = acquire_device_csr(
            V, E, nullptr, num_nodes, num_edges, false, &out_view);
        if (EG_ret != EG_GPU_SUCC) goto exit;
        d_V = out_view.d_V;
        d_E = out_view.d_E;
        if (is_directed) {
            EG_ret = acquire_device_csr(
                in_V, in_E, nullptr, num_nodes, num_edges, false, &in_view);
            if (EG_ret != EG_GPU_SUCC) goto exit;
            d_in_V = in_view.d_V;
            d_in_E = in_view.d_E;
        }
    } else {
        EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, (num_nodes+1) * sizeof(int)));
        EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, num_edges * sizeof(int)));
        EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_in_V, (num_nodes+1) * sizeof(int)));
        EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_in_E, num_edges * sizeof(int)));
        EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_row, num_edges * sizeof(int)));
        EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_col, num_edges * sizeof(int)));
        EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, num_edges * sizeof(double)));
    }
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_node_mask, num_nodes * sizeof(int)));
    if (is_directed && is_unweighted) {
        EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_directed_sum_scale, num_nodes * sizeof(int)));
        if (!block_nodes.empty()) {
            EXIT_IF_CUDA_FAILED(cudaMalloc(
                (void**)&d_block_nodes, block_nodes.size() * sizeof(int)));
        }
    }
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_hierarchy_results, num_nodes * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMemset(d_hierarchy_results, 0, num_nodes * sizeof(double)));

    if (!use_cached_csr) {
        EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, (num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
        EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, num_edges * sizeof(int), cudaMemcpyHostToDevice));
        EXIT_IF_CUDA_FAILED(cudaMemcpy(d_in_V, in_V, (num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
        EXIT_IF_CUDA_FAILED(cudaMemcpy(d_in_E, in_E, num_edges * sizeof(int), cudaMemcpyHostToDevice));
        EXIT_IF_CUDA_FAILED(cudaMemcpy(d_row, row, num_edges * sizeof(int), cudaMemcpyHostToDevice));
        EXIT_IF_CUDA_FAILED(cudaMemcpy(d_col, col, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    }
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_node_mask, node_mask, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    if (!block_nodes.empty()) {
        EXIT_IF_CUDA_FAILED(cudaMemcpy(
            d_block_nodes, block_nodes.data(), block_nodes.size() * sizeof(int),
            cudaMemcpyHostToDevice));
    }
    if (!is_unweighted) {
        EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, num_edges * sizeof(double), cudaMemcpyHostToDevice));
    }

    EXIT_IF_CUDA_FAILED(cudaEventCreate(&kernel_start));
    EXIT_IF_CUDA_FAILED(cudaEventCreate(&kernel_stop));
    EXIT_IF_CUDA_FAILED(cudaEventRecord(kernel_start, 0));

    if(is_directed){
        int block_size = 128;
        int shared_memory_size = sizeof(double) * block_size; 
        if (is_unweighted) {
            int scale_grid = (num_nodes + block_size - 1) / block_size;
            hierarchy_directed_unweighted_sum_scale_precompute<<<scale_grid, block_size>>>(
                d_V, d_in_V, num_nodes, d_directed_sum_scale);
            EXIT_IF_CUDA_FAILED(cudaGetLastError());
            directed_unweighted_calculate_hierarchy_thread<<<
                scale_grid, block_size>>>(
                d_V, d_E, d_in_V, d_in_E, num_nodes, d_node_mask,
                d_hierarchy_results, d_directed_sum_scale,
                DIRECTED_BLOCK_DEGREE_THRESHOLD);
            EXIT_IF_CUDA_FAILED(cudaGetLastError());
            if (!block_nodes.empty()) {
                directed_calculate_hierarchy<<<
                    (int)block_nodes.size(), block_size, shared_memory_size>>>(
                    d_V, d_E, d_in_V, d_in_E, d_row, d_col, d_W, num_nodes,
                    num_edges, d_node_mask, d_hierarchy_results, true,
                    d_directed_sum_scale, d_block_nodes,
                    (int)block_nodes.size());
            }
        } else {
            directed_calculate_hierarchy<<<
                num_nodes, block_size, shared_memory_size>>>(
                d_V, d_E, d_in_V, d_in_E, d_row, d_col, d_W, num_nodes,
                num_edges, d_node_mask, d_hierarchy_results, false,
                d_directed_sum_scale, nullptr, num_nodes);
        }
    }else{
        int block_size = 128;
        int grid_size = num_nodes; 
        int shared_memory_size = sizeof(double) * block_size; 
        calculate_hierarchy<<<grid_size, block_size, shared_memory_size>>>(d_V, d_E, d_W, num_nodes, d_node_mask, d_hierarchy_results, is_unweighted);
    }
    EXIT_IF_CUDA_FAILED(cudaGetLastError());
    EXIT_IF_CUDA_FAILED(cudaEventRecord(kernel_stop, 0));
    EXIT_IF_CUDA_FAILED(cudaEventSynchronize(kernel_stop));
    if (kernel_seconds != nullptr) {
        float elapsed_ms = 0.0f;
        EXIT_IF_CUDA_FAILED(cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop));
        *kernel_seconds = (double)elapsed_ms * 1e-3;
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(hierarchy_results, d_hierarchy_results, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));
exit:
    if (!use_cached_csr) {
        cudaFree(d_V);
        cudaFree(d_E);
        cudaFree(d_in_V);
        cudaFree(d_in_E);
    }
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_W);
    cudaFree(d_node_mask);
    cudaFree(d_directed_sum_scale);
    cudaFree(d_block_nodes);
    if (kernel_start != nullptr) cudaEventDestroy(kernel_start);
    if (kernel_stop != nullptr) cudaEventDestroy(kernel_stop);
    cudaFree(d_hierarchy_results);
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
}
