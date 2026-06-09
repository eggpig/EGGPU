#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "buffer_cache.h"
#include "common.h"
#include "device_graph_cache.h"

namespace gpu_easygraph {

struct ClosenessRuntime {
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    ~ClosenessRuntime() {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
    }
};

struct ClosenessWorkspace {
    PersistentDeviceBuffer d_sources;
    PersistentDeviceBuffer d_min_edge;
    PersistentDeviceBuffer d_dist_2D;
    PersistentDeviceBuffer d_U_2D;
    PersistentDeviceBuffer d_F_2D;
    PersistentDeviceBuffer d_CC;
    PersistentDeviceBuffer d_bfs_dist_2D;
    PersistentDeviceBuffer d_bfs_frontier_a_2D;
    PersistentDeviceBuffer d_bfs_frontier_b_2D;
};

static thread_local ClosenessWorkspace g_closeness_ws;

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

static __global__ void d_bfs_cc (
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ int* d_sources,
    _BUFFER_ int* d_dist_2D,
    _BUFFER_ int* d_frontier_a_2D,
    _BUFFER_ int* d_frontier_b_2D,
    _IN_ int len_V,
    _IN_ int len_sources,
    _OUT_ double* d_CC
)
{
    for (int s_idx = blockIdx.x; s_idx < len_sources; s_idx += gridDim.x) {
        int s = d_sources[s_idx];
        int* d_dist = d_dist_2D + blockIdx.x * len_V;
        int* d_frontier_a = d_frontier_a_2D + blockIdx.x * len_V;
        int* d_frontier_b = d_frontier_b_2D + blockIdx.x * len_V;

        __shared__ int frontier_size;
        __shared__ int next_size;
        __shared__ int depth;
        __shared__ int reachable_cnt;
        __shared__ unsigned long long dist_accum;

        for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
            d_dist[i] = -1;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            d_dist[s] = 0;
            d_frontier_a[0] = s;
            frontier_size = 1;
            depth = 0;
            reachable_cnt = 1;
            dist_accum = 0;
        }
        __syncthreads();

        while (frontier_size > 0) {
            int* curr_frontier = (depth & 1) ? d_frontier_b : d_frontier_a;
            int* next_frontier = (depth & 1) ? d_frontier_a : d_frontier_b;

            if (threadIdx.x == 0) {
                next_size = 0;
            }
            __syncthreads();

            int next_depth = depth + 1;
            for (int i = threadIdx.x; i < frontier_size; i += blockDim.x) {
                int u = curr_frontier[i];
                int edge_start = d_V[u];
                int edge_end = d_V[u + 1];
                for (int e = edge_start; e < edge_end; ++e) {
                    int v = d_E[e];
                    if (atomicCAS(d_dist + v, -1, next_depth) == -1) {
                        int out_idx = atomicAdd(&next_size, 1);
                        next_frontier[out_idx] = v;
                        atomicAdd(&reachable_cnt, 1);
                        atomicAdd(&dist_accum, (unsigned long long)next_depth);
                    }
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                frontier_size = next_size;
                depth = next_depth;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            d_CC[s_idx] = dist_accum == 0 ? 0.0 :
                                (double)(reachable_cnt - 1) *
                                (double)(reachable_cnt - 1) /
                                ((len_V - 1) * (double)dist_accum);
        }
        __syncthreads();
    }
}

static __global__ void d_dijkstra_cc (
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _IN_ double* d_min_edge,
    _IN_ int* d_sources,
    _BUFFER_ double* d_dist_2D,
    _BUFFER_ int* d_U_2D,
    _BUFFER_ int* d_F_2D,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int warp_size,
    _OUT_ double* d_CC
)
{
    for (int s_idx = blockIdx.x; s_idx < len_sources; s_idx += gridDim.x) {
        int s = d_sources[s_idx];

        int* d_U = d_U_2D + blockIdx.x * len_V;
        int* d_F = d_F_2D + blockIdx.x * len_V;
        double* d_dist = d_dist_2D + blockIdx.x * len_V;

        __shared__ int len_F;
        __shared__ double delta;
        __shared__ double dist_accum;
        __shared__ int reachable_cnt;

        for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
            d_U[i] = 1;
            d_dist[i] = EG_DOUBLE_INF;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            d_dist[s] = 0.0;
            d_F[0] = s;
            len_F = 1;
            delta = 0.0;
            dist_accum = 0.0;
            reachable_cnt = 0;
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
                if (d_U[i] && dist_i <= delta && dist_i < EG_DOUBLE_INF) {
                    d_U[i] = 0;
                    int f_idx = atomicAdd(&len_F, 1);
                    d_F[f_idx] = i;

                    atomicAdd(&reachable_cnt, 1);
                    atomicAddDouble(&dist_accum, d_dist[i]);
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            d_CC[s_idx] = dist_accum == 0.0 ? 0.0 :
                                (double)(reachable_cnt - 1) * 
                                (double)(reachable_cnt - 1) /
                                ((len_V - 1) * dist_accum);
        }
        __syncthreads();
    }
}



// we here use CSR to represent a graph
int cuda_closeness_centrality (
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const double* W,
    _IN_ const int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int warp_size,
    _IN_ bool unweighted,
    _OUT_ double* CC,
    _OUT_ double* kernel_seconds
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    ClosenessRuntime runtime;
    DeviceCsrView graph_view;

    int min_edge_block_size;
    int min_edge_grid_size;
    int dijkstra_block_size;
    int dijkstra_grid_size;
    int bfs_block_size;
    int bfs_grid_size;
    int rc = EG_GPU_SUCC;
    int* d_sources = nullptr;
    double* d_CC = nullptr;

    cudaOccupancyMaxPotentialBlockSize(&min_edge_grid_size, &min_edge_block_size, d_calc_min_edge, 0, 0); 
    cudaOccupancyMaxPotentialBlockSize(&dijkstra_grid_size, &dijkstra_block_size, d_dijkstra_cc, 0, 0); 
    cudaOccupancyMaxPotentialBlockSize(&bfs_grid_size, &bfs_block_size, d_bfs_cc, 0, 0);
    if (len_sources <= 0) {
        if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
        return EG_GPU_SUCC;
    }
    if (dijkstra_grid_size > len_sources) {
        dijkstra_grid_size = len_sources;
    }
    if (bfs_grid_size > len_sources) {
        bfs_grid_size = len_sources;
    }

    rc = acquire_device_csr(V, E, W, len_V, len_E, !unweighted, &graph_view);
    if (rc != EG_GPU_SUCC) {
        EG_ret = rc;
        goto exit;
    }
    rc = g_closeness_ws.d_sources.ensure_bytes(sizeof(int) * len_sources);
    if (rc != EG_GPU_SUCC) {
        EG_ret = rc;
        goto exit;
    }
    rc = g_closeness_ws.d_CC.ensure_bytes(sizeof(double) * len_sources);
    if (rc != EG_GPU_SUCC) {
        EG_ret = rc;
        goto exit;
    }

    d_sources = g_closeness_ws.d_sources.as<int>();
    d_CC = g_closeness_ws.d_CC.as<double>();
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_sources, sources, sizeof(int) * len_sources, cudaMemcpyHostToDevice));

    if (unweighted) {
        rc = g_closeness_ws.d_bfs_dist_2D.ensure_bytes(sizeof(int) * bfs_grid_size * len_V);
        if (rc != EG_GPU_SUCC) {
            EG_ret = rc;
            goto exit;
        }
        rc = g_closeness_ws.d_bfs_frontier_a_2D.ensure_bytes(sizeof(int) * bfs_grid_size * len_V);
        if (rc != EG_GPU_SUCC) {
            EG_ret = rc;
            goto exit;
        }
        rc = g_closeness_ws.d_bfs_frontier_b_2D.ensure_bytes(sizeof(int) * bfs_grid_size * len_V);
        if (rc != EG_GPU_SUCC) {
            EG_ret = rc;
            goto exit;
        }
    } else {
        rc = g_closeness_ws.d_U_2D.ensure_bytes(sizeof(int) * dijkstra_grid_size * len_V);
        if (rc != EG_GPU_SUCC) {
            EG_ret = rc;
            goto exit;
        }
        rc = g_closeness_ws.d_F_2D.ensure_bytes(sizeof(int) * dijkstra_grid_size * len_V);
        if (rc != EG_GPU_SUCC) {
            EG_ret = rc;
            goto exit;
        }
        rc = g_closeness_ws.d_min_edge.ensure_bytes(sizeof(double) * len_V);
        if (rc != EG_GPU_SUCC) {
            EG_ret = rc;
            goto exit;
        }
        rc = g_closeness_ws.d_dist_2D.ensure_bytes(sizeof(double) * dijkstra_grid_size * len_V);
        if (rc != EG_GPU_SUCC) {
            EG_ret = rc;
            goto exit;
        }
    }

    if (kernel_seconds != nullptr) {
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.start_event));
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.stop_event));
        EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.start_event));
    }

    if (unweighted) {
        d_bfs_cc<<<bfs_grid_size, bfs_block_size>>>(
            graph_view.d_V, graph_view.d_E, d_sources,
            g_closeness_ws.d_bfs_dist_2D.as<int>(),
            g_closeness_ws.d_bfs_frontier_a_2D.as<int>(),
            g_closeness_ws.d_bfs_frontier_b_2D.as<int>(),
            len_V, len_sources, d_CC);
        EXIT_IF_CUDA_FAILED(cudaGetLastError());
    } else {
        d_calc_min_edge<<<min_edge_grid_size, min_edge_block_size>>>(
            graph_view.d_V, graph_view.d_E, graph_view.d_W, len_V, len_E,
            g_closeness_ws.d_min_edge.as<double>());
        EXIT_IF_CUDA_FAILED(cudaGetLastError());

        d_dijkstra_cc<<<dijkstra_grid_size, dijkstra_block_size>>>(
            graph_view.d_V, graph_view.d_E, graph_view.d_W,
            g_closeness_ws.d_min_edge.as<double>(), d_sources,
            g_closeness_ws.d_dist_2D.as<double>(),
            g_closeness_ws.d_U_2D.as<int>(),
            g_closeness_ws.d_F_2D.as<int>(),
            len_V, len_E, len_sources, warp_size, d_CC);
        EXIT_IF_CUDA_FAILED(cudaGetLastError());
    }

    if (kernel_seconds != nullptr) {
        EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.stop_event));
        EXIT_IF_CUDA_FAILED(cudaEventSynchronize(runtime.stop_event));
        float elapsed_ms = 0.0f;
        EXIT_IF_CUDA_FAILED(cudaEventElapsedTime(&elapsed_ms, runtime.start_event, runtime.stop_event));
        *kernel_seconds = (double)elapsed_ms * 1e-3;
    }
    EXIT_IF_CUDA_FAILED(cudaMemcpy(CC, d_CC, sizeof(double) * len_sources, cudaMemcpyDeviceToHost));

exit:
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
