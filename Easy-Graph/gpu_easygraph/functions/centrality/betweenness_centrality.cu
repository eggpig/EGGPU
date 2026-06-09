#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <algorithm>

#include "buffer_cache.h"
#include "common.h"
#include "device_graph_cache.h"

namespace gpu_easygraph {

static inline int ensure_device_buffer(PersistentDeviceBuffer& buf, std::size_t bytes, int* eg_ret) {
    int rc = buf.ensure_bytes(bytes);
    if (rc != EG_GPU_SUCC && eg_ret != nullptr) {
        *eg_ret = rc;
    }
    return rc;
}

struct BcRuntime {
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    bool ready = false;
    ~BcRuntime() {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
    }
};

static __device__ double atomicAddDouble (
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



static __device__ double atomicMinDouble (
    _OUT_ double *address, 
    _IN_ double val
)
{
	unsigned long long ret = __double_as_longlong(*address);
	while (val < __longlong_as_double(ret))
	{
		unsigned long long old = ret;
		if ((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
			break;
	}
	return __longlong_as_double(ret);
}



static __global__ void d_calc_min_edge (
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ double* d_min_edge
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;

    for (int u = tid; u < len_V; u += tnum) {
		double curr_min = EG_DOUBLE_INF;
        int edge_start = d_V[u];
        int edge_end = d_V[u + 1];
		for(int v = edge_start; v < edge_end; ++v) {
            curr_min = min(curr_min, d_W[v]);
		}
		d_min_edge[u] = curr_min;
    }
}



static __global__ void d_dijkstra_bc (
    _IN_ int* d_curr_node,
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _IN_ double* d_min_edge,
    _IN_ int* d_sources,
    _BUFFER_ double* d_dist_2D,
    _BUFFER_ double* d_sigma_2D,
    _BUFFER_ double* d_delta_2D,
    _BUFFER_ int* d_U_2D,
    _BUFFER_ int* d_F_2D,
    _BUFFER_ int* d_st_2D,
    _BUFFER_ int* d_st_idx_2D,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int warp_size,
    _IN_ int endpoints,
    _OUT_ double* d_BC
)
{
    //for (int s_idx = blockIdx.x; s_idx < len_sources; s_idx += gridDim.x) {
    while (1) {
        __shared__ int curr_node;
        if (threadIdx.x == 0) {
            curr_node = atomicAdd(d_curr_node, 1);
        }
        __syncthreads();

        if (curr_node >= len_sources) {
            break;
        }

        int s = d_sources[curr_node];

        double* d_dist = d_dist_2D + blockIdx.x * len_V;
        double* d_sigma = d_sigma_2D + blockIdx.x * len_V;
        double* d_delta = d_delta_2D + blockIdx.x * len_V;

        int* d_U = d_U_2D + blockIdx.x * len_V;
        int* d_F = d_F_2D + blockIdx.x * len_V;
        int* d_st = d_st_2D + blockIdx.x * len_V;
        int* d_st_idx = d_st_idx_2D + blockIdx.x * (len_V + 2);

        __shared__ int len_F;
        __shared__ int len_st;
        __shared__ int len_st_idx;
        __shared__ double delta;

        for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
            d_dist[i] = EG_DOUBLE_INF;
            d_sigma[i] = 0;
            d_delta[i] = 0;

            d_U[i] = 1;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            d_dist[s] = 0;
            d_sigma[s] = 1;

            d_U[s] = 0;
            d_F[0] = s;
            len_F = 1;
            d_st[0] = s;
            len_st = 1;
            d_st_idx[0] = 0;
            d_st_idx[1] = 1;
            len_st_idx = 2;

            delta = 0.0;
        }
        __syncthreads();

        while (delta < EG_DOUBLE_INF) {
            for (int j = threadIdx.x; j < len_F * warp_size; j += blockDim.x) {
                int f = d_F[j / warp_size];
                int edge_start = d_V[f];
                int edge_end = d_V[f + 1];
                double dist = d_dist[f];
                for (int e = j % warp_size; e < edge_end - edge_start; e += warp_size) {
                    int adj = d_E[e + edge_start];
                    double relax_w = dist + d_W[e + edge_start];
                    atomicMinDouble(d_dist + adj, relax_w);
                }
                __threadfence_block();
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                delta = EG_DOUBLE_INF;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
                double dist_i = d_dist[i];
                if (d_U[i] == 1 && dist_i < EG_DOUBLE_INF) {
                    atomicMinDouble(&delta, dist_i + d_min_edge[i]);
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
				len_F = 0;
			}
			__syncthreads();

            for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
                double dist_i = d_dist[i];
                if (d_U[i] && dist_i < delta && dist_i < EG_DOUBLE_INF) {
                    d_U[i] = 0;
                    int f_idx = atomicAdd(&len_F, 1);
                    d_F[f_idx] = i;
                }
            }
            __syncthreads();

            for (int i = threadIdx.x; i < len_F; i += blockDim.x) {
                int st_idx = atomicAdd(&len_st, 1);
                d_st[st_idx] = d_F[i];
            }
            __syncthreads();
            
            if (threadIdx.x == 0) {
                d_st_idx[len_st_idx] = d_st_idx[len_st_idx - 1] + len_F;
                ++len_st_idx;
            }
            __syncthreads();
        }
        // calculate single source shortest path END

        // calculate sigma START
        for (int curr_lvl = 0; curr_lvl + 1 < len_st_idx; ++curr_lvl) {
            int lvl_start = d_st_idx[curr_lvl];
            int lvl_end = d_st_idx[curr_lvl + 1];
            for (int j = threadIdx.x; j < (lvl_end - lvl_start) * warp_size; j += blockDim.x) {
                int v = d_st[lvl_start + j / warp_size];
                double dist_v = d_dist[v];
                int edge_start = d_V[v];
                int edge_end = d_V[v + 1];
                for (int e = j % warp_size; e < edge_end - edge_start; e += warp_size) {
                    int adj = d_E[e + edge_start];
                    if (dist_v + d_W[e + edge_start] == d_dist[adj]) {
                        atomicAddDouble(d_sigma + adj, d_sigma[v]);
                    }
                }
                __threadfence_block();
            }
            __syncthreads();
        }
        // calculate sigma END

        __shared__ int depth, st_start, st_end;
        if (threadIdx.x == 0) {
            depth = len_st_idx - 1;
        }
        __syncthreads();

        if (threadIdx.x == 0 && endpoints) {
            atomicAddDouble(d_BC + s, d_st_idx[depth] - 1);
        }
        __syncthreads();

        while (depth > 0) {
            if (threadIdx.x == 0) {
                st_start = d_st_idx[depth - 1];
                st_end = d_st_idx[depth];
            }
            __syncthreads();

            for (int j = threadIdx.x; j < (st_end - st_start) * warp_size; j += blockDim.x) {
                int pred = d_st[st_start + j / warp_size];
                int edge_start = d_V[pred];
                int edge_end = d_V[pred + 1];
                double pred_sigma = d_sigma[pred];
                double pred_dist = d_dist[pred];

                for (int e = j % warp_size; e < edge_end - edge_start; e += warp_size) {
                    int succ = d_E[e + edge_start];
                    double weight = d_W[e + edge_start];
                    double succ_dist = d_dist[succ];
                    if (succ_dist == pred_dist + weight) {
                        atomicAddDouble(d_delta + pred, 
                                pred_sigma / d_sigma[succ] * (1 + d_delta[succ]));
                    }
                }
                __threadfence_block();
            }
            __syncthreads();

            for (int i = threadIdx.x; i < st_end - st_start; i += blockDim.x) {
                int pred = d_st[st_start + i];
                if (s != pred) {
                    atomicAddDouble(d_BC + pred, d_delta[pred] + endpoints);
                }
            }
            __syncthreads();


            if (threadIdx.x == 0) {
                --depth;
            }
            __syncthreads();
        }
    }
}

static __global__ void d_rescale(
    _IN_ int len_V,
    _IN_ double scale,
    _OUT_ double* d_BC
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;

    for (int u = tid; u < len_V; u += tnum) {
        d_BC[u] *= scale;
    }
}



static double calc_scale(
    _IN_ int len_V,
    _IN_ int is_directed,
    _IN_ int normalized,
    _IN_ int endpoints
)
{
    double scale = 1.0;
    if (normalized) {
        if (endpoints) {
            if (len_V < 2) {
                scale = 1.0;
            } else {
                scale = 1.0 / (double(len_V) * (len_V - 1));
            }
        } else if (len_V <= 2) {
            scale = 1.0;
        } else {
            scale = 1.0 / ((double(len_V) - 1) * (len_V - 2));
        }
    } else {
        if (!is_directed) {
            scale = 0.5;
        } else {
            scale = 1.0;
        }
    }
    return scale;
}

static int memory_aware_source_blocks(int requested, int len_V)
{
    const char* explicit_cap = getenv("EASYGRAPH_GPU_BC_MAX_CONCURRENT_SOURCES");
    if (explicit_cap != nullptr) {
        int parsed = atoi(explicit_cap);
        if (parsed > 0) return std::max(1, std::min(requested, parsed));
    }

    // Brandes keeps several per-source dense arrays.  On smaller GPUs this can
    // turn sampled BC into a memory benchmark, so cap concurrency only when the
    // workspace would exceed a conservative budget.  Users can override with
    // EASYGRAPH_GPU_BC_MAX_CONCURRENT_SOURCES.
    std::size_t per_source_bytes =
        (std::size_t)len_V * (3 * sizeof(double) + 4 * sizeof(int)) +
        (std::size_t)(len_V + 2) * sizeof(int);
    std::size_t free_b = 0, total_b = 0;
    cudaError_t mem_ret = cudaMemGetInfo(&free_b, &total_b);
    if (mem_ret != cudaSuccess || per_source_bytes == 0) {
        (void)cudaGetLastError();
        return requested;
    }
    double frac = 0.20;
    const char* frac_env = getenv("EASYGRAPH_GPU_BC_WORKSPACE_FRACTION");
    if (frac_env != nullptr) {
        double parsed = atof(frac_env);
        if (parsed > 0.0 && parsed < 0.95) frac = parsed;
    }
    std::size_t budget = (std::size_t)((double)free_b * frac);
    int by_mem = (int)std::max<std::size_t>(1, budget / per_source_bytes);
    return std::max(1, std::min(requested, by_mem));
}



int cuda_betweenness_centrality (
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const double* W,
    _IN_ const int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int warp_size,
    _IN_ int is_directed,
    _IN_ int normalized,
    _IN_ int endpoints,
    _OUT_ double* BC,
    _OUT_ double* kernel_seconds
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;

    int min_edge_block_size;
    int min_edge_grid_size;
    int dijkstra_block_size;
    int dijkstra_grid_size;
    int rescale_block_size;
    int rescale_grid_size;

    cudaOccupancyMaxPotentialBlockSize(&min_edge_grid_size, &min_edge_block_size, d_calc_min_edge, 0, 0); 
    cudaOccupancyMaxPotentialBlockSize(&dijkstra_grid_size, &dijkstra_block_size, d_dijkstra_bc, 0, 0); 
    cudaOccupancyMaxPotentialBlockSize(&rescale_grid_size, &rescale_block_size, d_rescale, 0, 0); 
    if (len_sources > 0 && dijkstra_grid_size > len_sources) {
        dijkstra_grid_size = len_sources;
    }
    dijkstra_grid_size = memory_aware_source_blocks(dijkstra_grid_size, len_V);
    if (dijkstra_grid_size < 1) {
        dijkstra_grid_size = 1;
    }
    
    double scale = calc_scale(len_V, is_directed, normalized, endpoints);

    thread_local PersistentDeviceBuffer b_curr_node;
    thread_local PersistentDeviceBuffer b_sources;
    thread_local PersistentDeviceBuffer b_U_2D;
    thread_local PersistentDeviceBuffer b_F_2D;
    thread_local PersistentDeviceBuffer b_st_2D;
    thread_local PersistentDeviceBuffer b_st_idx_2D;
    thread_local PersistentDeviceBuffer b_min_edge;
    thread_local PersistentDeviceBuffer b_dist_2D;
    thread_local PersistentDeviceBuffer b_sigma_2D;
    thread_local PersistentDeviceBuffer b_delta_2D;
    thread_local PersistentDeviceBuffer b_BC;
    thread_local PersistentPinnedBuffer h_sources_stage;
    thread_local BcRuntime runtime;

    int *d_curr_node = NULL;
    int *d_V = NULL, *d_E = NULL, *d_sources= NULL;
    int *d_U_2D = NULL, *d_F_2D = NULL, *d_st_2D = NULL, *d_st_idx_2D = NULL;
    double *d_W = NULL, *d_min_edge = NULL, *d_dist_2D = NULL,
            *d_sigma_2D = NULL, *d_delta_2D = NULL, *d_BC = NULL;
    const int* h_sources = nullptr;
    bool reg_sources = false;
    DeviceCsrView graph_view;

    if (ensure_device_buffer(b_curr_node, sizeof(int), &EG_ret) != EG_GPU_SUCC) goto exit;
    EG_ret = acquire_device_csr(V, E, W, len_V, len_E, true, &graph_view);
    if (EG_ret != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_sources, sizeof(int) * len_sources, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_U_2D, sizeof(int) * dijkstra_grid_size * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_F_2D, sizeof(int) * dijkstra_grid_size * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_st_2D, sizeof(int) * dijkstra_grid_size * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_st_idx_2D, sizeof(int) * dijkstra_grid_size * (len_V + 2), &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_min_edge, sizeof(double) * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_dist_2D, sizeof(double) * dijkstra_grid_size * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_sigma_2D, sizeof(double) * dijkstra_grid_size * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_delta_2D, sizeof(double) * dijkstra_grid_size * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_BC, sizeof(double) * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;

    d_curr_node = b_curr_node.as<int>();
    d_V = graph_view.d_V;
    d_E = graph_view.d_E;
    d_sources = b_sources.as<int>();
    d_U_2D = b_U_2D.as<int>();
    d_F_2D = b_F_2D.as<int>();
    d_st_2D = b_st_2D.as<int>();
    d_st_idx_2D = b_st_idx_2D.as<int>();
    d_W = graph_view.d_W;
    d_min_edge = b_min_edge.as<double>();
    d_dist_2D = b_dist_2D.as<double>();
    d_sigma_2D = b_sigma_2D.as<double>();
    d_delta_2D = b_delta_2D.as<double>();
    d_BC = b_BC.as<double>();

    EXIT_IF_CUDA_FAILED(cudaMemset(d_curr_node, 0, sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMemset(d_BC, 0, sizeof(double) * len_V));
    h_sources = prepare_h2d_source(sources, len_sources, h_sources_stage, &reg_sources);
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_sources, h_sources, sizeof(int) * len_sources, cudaMemcpyHostToDevice));

    if (!runtime.ready) {
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.start_event));
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.stop_event));
        runtime.ready = true;
    }
    EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.start_event));

    d_calc_min_edge<<<min_edge_grid_size, min_edge_block_size>>>(d_V, d_E, d_W, len_V, len_E, d_min_edge);
    EXIT_IF_CUDA_FAILED(cudaGetLastError());

    d_dijkstra_bc<<<dijkstra_grid_size, dijkstra_block_size>>>(d_curr_node, d_V, d_E, d_W, d_min_edge,
                                            d_sources, d_dist_2D, d_sigma_2D, d_delta_2D, d_U_2D,
                                            d_F_2D, d_st_2D, d_st_idx_2D, len_V, len_E, len_sources,
                                            warp_size, endpoints, d_BC);
    EXIT_IF_CUDA_FAILED(cudaGetLastError());

    if (scale != 1.0) {
        d_rescale<<<rescale_grid_size, rescale_block_size>>>(len_V, scale, d_BC);
        EXIT_IF_CUDA_FAILED(cudaGetLastError());
    }

    EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.stop_event));
    EXIT_IF_CUDA_FAILED(cudaEventSynchronize(runtime.stop_event));
    if (kernel_seconds != nullptr) {
        float elapsed_ms = 0.0f;
        EXIT_IF_CUDA_FAILED(cudaEventElapsedTime(&elapsed_ms, runtime.start_event, runtime.stop_event));
        *kernel_seconds = (double)elapsed_ms * 1e-3;
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(BC, d_BC, sizeof(double) * len_V, cudaMemcpyDeviceToHost));

exit:
    release_h2d_source(h_sources, reg_sources);

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
