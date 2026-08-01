#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdlib.h>
#include <vector>

#include "buffer_cache.h"
#include "common.h"
#include "device_graph_cache.h"

namespace gpu_easygraph {

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

static __device__ bool atomicMinDoubleUpdate(
    _OUT_ double* address,
    _IN_ double val
)
{
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    while (val < __longlong_as_double(old)) {
        unsigned long long assumed = old;
        unsigned long long desired = __double_as_longlong(val);
        old = atomicCAS(address_as_ull, assumed, desired);
        if (old == assumed) return true;
    }
    return false;
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



static __global__ void d_sssp_dijkstra (
    _IN_ int* d_curr_node,
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _IN_ double* d_min_edge,
    _IN_ int* d_sources,
    _OUT_ double* d_dist_2D,
    _BUFFER_ int* d_U_2D,
    _BUFFER_ int* d_F_2D,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int target,
    _IN_ int warp_size
)
{
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

        double* d_dist = d_dist_2D + curr_node * len_V;
        int* d_U = d_U_2D + blockIdx.x * len_V;
        int* d_F = d_F_2D + blockIdx.x * len_V;

        __shared__ int len_F;
        __shared__ double delta;
        __shared__ int target_cnt;

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
            target_cnt = 0;
        }
        __syncthreads();

        while (delta < EG_DOUBLE_INF && target_cnt == 0) {
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
                    target_cnt += i == target;
                }
            }
            __syncthreads();
        }

        __syncthreads();
    }
}

static __global__ void d_sssp_delta_init(
    _IN_ int* d_sources,
    _OUT_ double* d_dist,
    _OUT_ int* d_U,
    _OUT_ int* d_F,
    _OUT_ int* d_len_F,
    _IN_ int len_V,
    _IN_ int source_row
)
{
    int s = d_sources[source_row];
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < len_V;
         i += blockDim.x * gridDim.x) {
        d_U[i] = 1;
        d_dist[i] = (i == s) ? 0.0 : EG_DOUBLE_INF;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_F[0] = s;
        *d_len_F = 1;
    }
}

static __global__ void d_sssp_delta_relax(
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _IN_ int* d_F,
    _IN_ int len_F,
    _OUT_ double* d_dist,
    _IN_ int warp_size
)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tnum = (long long)blockDim.x * gridDim.x;
    long long work = (long long)len_F * (long long)warp_size;

    for (long long j = tid; j < work; j += tnum) {
        int f = d_F[j / warp_size];
        int edge_start = d_V[f];
        int edge_end = d_V[f + 1];
        double dist = d_dist[f];
        if (dist >= EG_DOUBLE_INF) continue;
        for (int e = (int)(j % warp_size); e < edge_end - edge_start; e += warp_size) {
            int edge_idx = e + edge_start;
            int adj = d_E[edge_idx];
            double relax_w = dist + d_W[edge_idx];
            atomicMinDouble(d_dist + adj, relax_w);
        }
    }
}

static __global__ void d_sssp_delta_reduce_partial(
    _IN_ double* d_dist,
    _IN_ int* d_U,
    _IN_ double* d_min_edge,
    _OUT_ double* d_partial,
    _IN_ int len_V
)
{
    __shared__ double s_min[256];
    double local = EG_DOUBLE_INF;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;

    for (int i = tid; i < len_V; i += tnum) {
        double dist_i = d_dist[i];
        if (d_U[i] == 1 && dist_i < EG_DOUBLE_INF) {
            double cand = dist_i + d_min_edge[i];
            if (cand < local) local = cand;
        }
    }

    s_min[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            double other = s_min[threadIdx.x + stride];
            if (other < s_min[threadIdx.x]) s_min[threadIdx.x] = other;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_partial[blockIdx.x] = s_min[0];
    }
}

static __global__ void d_sssp_delta_build_frontier(
    _OUT_ int* d_U,
    _IN_ double* d_dist,
    _IN_ double delta,
    _OUT_ int* d_F,
    _OUT_ int* d_len_F,
    _IN_ int len_V
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;

    for (int i = tid; i < len_V; i += tnum) {
        double dist_i = d_dist[i];
        if (d_U[i] == 1 && dist_i <= delta && dist_i < EG_DOUBLE_INF) {
            d_U[i] = 0;
            int f_idx = atomicAdd(d_len_F, 1);
            if (f_idx < len_V) {
                d_F[f_idx] = i;
            }
        }
    }
}

static __global__ void d_sssp_frontier_init(
    _IN_ int* d_sources,
    _OUT_ double* d_dist,
    _OUT_ int* d_frontier,
    _OUT_ int* d_len_F,
    _IN_ int len_V,
    _IN_ int source_row
)
{
    int s = d_sources[source_row];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tnum = blockDim.x * gridDim.x;
    for (int i = tid; i < len_V; i += tnum) {
        d_dist[i] = (i == s) ? 0.0 : EG_DOUBLE_INF;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_frontier[0] = s;
        *d_len_F = 1;
    }
}

static __global__ void d_sssp_frontier_relax(
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _IN_ int* d_frontier,
    _IN_ int len_F,
    _OUT_ double* d_dist,
    _OUT_ int* d_next,
    _OUT_ int* d_next_len,
    _OUT_ int* d_overflow,
    _IN_ int frontier_cap,
    _IN_ int warp_size
)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tnum = (long long)blockDim.x * gridDim.x;
    long long work = (long long)len_F * (long long)warp_size;
    for (long long j = tid; j < work; j += tnum) {
        int u = d_frontier[j / warp_size];
        int edge_start = d_V[u];
        int edge_end = d_V[u + 1];
        double du = d_dist[u];
        if (du >= EG_DOUBLE_INF) continue;
        for (int e = (int)(j % warp_size); e < edge_end - edge_start; e += warp_size) {
            int edge_idx = edge_start + e;
            int v = d_E[edge_idx];
            double nd = du + d_W[edge_idx];
            if (atomicMinDoubleUpdate(d_dist + v, nd)) {
                int pos = atomicAdd(d_next_len, 1);
                if (pos < frontier_cap) {
                    d_next[pos] = v;
                } else {
                    atomicExch(d_overflow, 1);
                }
            }
        }
    }
}

static __global__ void d_sssp_unweighted_bfs(
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ int* d_sources,
    _OUT_ int* d_dist_2D,
    _IN_ int len_V,
    _IN_ int len_sources,
    _IN_ int target
)
{
    int source_row = blockIdx.x;
    if (source_row >= len_sources) return;

    int* d_dist = d_dist_2D + source_row * len_V;
    for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
        d_dist[i] = -1;
    }
    __syncthreads();

    __shared__ int level;
    __shared__ int active;
    __shared__ int target_found;
    if (threadIdx.x == 0) {
        int s = d_sources[source_row];
        d_dist[s] = 0;
        level = 0;
        active = 1;
        target_found = (target >= 0 && s == target) ? 1 : 0;
    }
    __syncthreads();

    while (active > 0 && target_found == 0) {
        __shared__ int next_active;
        if (threadIdx.x == 0) {
            next_active = 0;
        }
        __syncthreads();

        for (int u = threadIdx.x; u < len_V; u += blockDim.x) {
            if (d_dist[u] != level) continue;
            int edge_start = d_V[u];
            int edge_end = d_V[u + 1];
            for (int e = edge_start; e < edge_end; ++e) {
                int v = d_E[e];
                if (atomicCAS(d_dist + v, -1, level + 1) == -1) {
                    atomicAdd(&next_active, 1);
                    if (target >= 0 && v == target) {
                        atomicExch(&target_found, 1);
                    }
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            active = next_active;
            ++level;
        }
        __syncthreads();
    }
}

static __global__ void d_bfs_frontier_init(
    _IN_ int* d_sources,
    _OUT_ int* d_dist,
    _OUT_ int* d_frontier,
    _OUT_ int* d_len_F,
    _IN_ int len_V,
    _IN_ int source_row
)
{
    int s = d_sources[source_row];
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < len_V;
         i += blockDim.x * gridDim.x) {
        d_dist[i] = (i == s) ? 0 : -1;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_frontier[0] = s;
        *d_len_F = 1;
    }
}

static __global__ void d_bfs_expand_frontier(
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ int* d_frontier,
    _IN_ int len_F,
    _OUT_ int* d_dist,
    _OUT_ int* d_next,
    _OUT_ int* d_next_len,
    _OUT_ int* d_target_found,
    _IN_ int target,
    _IN_ int level,
    _IN_ int warp_size
)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tnum = (long long)blockDim.x * gridDim.x;
    long long work = (long long)len_F * (long long)warp_size;
    for (long long j = tid; j < work; j += tnum) {
        int u = d_frontier[j / warp_size];
        int edge_start = d_V[u];
        int edge_end = d_V[u + 1];
        for (int e = (int)(j % warp_size); e < edge_end - edge_start; e += warp_size) {
            int v = d_E[edge_start + e];
            if (atomicCAS(d_dist + v, -1, level + 1) == -1) {
                int pos = atomicAdd(d_next_len, 1);
                d_next[pos] = v;
                if (target >= 0 && v == target) {
                    atomicExch(d_target_found, 1);
                }
            }
        }
    }
}

static __global__ void d_bfs_int_to_double(
    _IN_ int* d_dist_int,
    _OUT_ double* d_dist_double,
    _IN_ int total
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;
    for (int i = tid; i < total; i += tnum) {
        int d = d_dist_int[i];
        d_dist_double[i] = (d < 0) ? EG_DOUBLE_INF : (double)d;
    }
}

static __global__ void d_bellman_build_edge_src(
    _IN_ int* d_V,
    _OUT_ int* d_edge_src,
    _IN_ int len_V
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;
    for (int u = tid; u < len_V; u += tnum) {
        int begin = d_V[u];
        int end = d_V[u + 1];
        for (int e = begin; e < end; ++e) {
            d_edge_src[e] = u;
        }
    }
}

static __global__ void d_bellman_init(
    _IN_ int* d_sources,
    _OUT_ double* d_dist,
    _IN_ int len_V,
    _IN_ int len_sources
)
{
    long long total = (long long)len_sources * (long long)len_V;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tnum = (long long)blockDim.x * gridDim.x;
    for (long long idx = tid; idx < total; idx += tnum) {
        int row = (int)(idx / len_V);
        int col = (int)(idx - (long long)row * len_V);
        int source = d_sources[row];
        d_dist[idx] = (col == source) ? 0.0 : EG_DOUBLE_INF;
    }
}

static __global__ void d_bellman_relax_edges(
    _IN_ int* d_edge_src,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _OUT_ double* d_dist,
    _OUT_ int* d_changed,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources
)
{
    long long total = (long long)len_sources * (long long)len_E;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tnum = (long long)blockDim.x * gridDim.x;
    for (long long idx = tid; idx < total; idx += tnum) {
        int row = (int)(idx / len_E);
        int edge = (int)(idx - (long long)row * len_E);
        long long row_base = (long long)row * (long long)len_V;
        int u = d_edge_src[edge];
        int v = d_E[edge];
        double du = d_dist[row_base + u];
        if (du >= EG_DOUBLE_INF) continue;
        double nd = du + d_W[edge];
        if (atomicMinDoubleUpdate(d_dist + row_base + v, nd)) {
            atomicExch(d_changed, 1);
        }
    }
}

static inline int ensure_device_buffer(PersistentDeviceBuffer& buf, std::size_t bytes, int* eg_ret) {
    int rc = buf.ensure_bytes(bytes);
    if (rc != EG_GPU_SUCC && eg_ret != nullptr) {
        *eg_ret = rc;
    }
    return rc;
}

struct SsspRuntime {
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    bool ready = false;
    ~SsspRuntime() {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
    }
};

static inline bool env_flag_enabled(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return default_value;
    return raw[0] == '1' || raw[0] == 'T' || raw[0] == 't' ||
           raw[0] == 'Y' || raw[0] == 'y' || raw[0] == 'O' || raw[0] == 'o';
}

static inline int env_int_value(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return default_value;
    long parsed = std::strtol(raw, nullptr, 10);
    if (parsed <= 0) return default_value;
    return (int)parsed;
}

static inline bool use_parallel_delta_sssp(int len_V, int len_E) {
    if (env_flag_enabled("EASYGRAPH_GPU_SSSP_LEGACY_DENSE", false)) {
        return false;
    }
    int min_edges = env_int_value("EASYGRAPH_GPU_SSSP_PARALLEL_DELTA_MIN_EDGES", 300000);
    int min_vertices = env_int_value("EASYGRAPH_GPU_SSSP_PARALLEL_DELTA_MIN_VERTICES", 50000);
    return len_E >= min_edges || len_V >= min_vertices;
}

static inline bool use_frontier_weighted_sssp(
    const HostCsrStats& stats,
    int len_V,
    int len_E,
    int len_sources
) {
    return should_use_weighted_frontier_sssp(
        stats, len_V, len_E, len_sources);
}

static inline bool use_small_scan_bfs(int len_V, int len_E, int len_sources) {
    if (!env_flag_enabled("EASYGRAPH_GPU_BFS_SCAN_SMALL", true)) {
        return false;
    }
    if (len_sources <= 0 || len_sources > 65535) {
        return false;
    }
    int max_vertices = env_int_value("EASYGRAPH_GPU_BFS_SCAN_SMALL_MAX_VERTICES", 20000);
    int max_edges = env_int_value("EASYGRAPH_GPU_BFS_SCAN_SMALL_MAX_EDGE_SLOTS", 250000);
    return len_V <= max_vertices && len_E <= max_edges;
}

// we here use CSR to represent a graph
int cuda_sssp_dijkstra(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const double* W,
    _IN_ const int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int target,
    _IN_ int warp_size,
    _OUT_ double* res,
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

    cudaOccupancyMaxPotentialBlockSize(&min_edge_grid_size, &min_edge_block_size, d_calc_min_edge, 0, 0); 
    cudaOccupancyMaxPotentialBlockSize(&dijkstra_grid_size, &dijkstra_block_size, d_sssp_dijkstra, 0, 0); 
    if (len_sources > 0 && dijkstra_grid_size > len_sources) {
        dijkstra_grid_size = len_sources;
    }
    if (dijkstra_grid_size < 1) {
        dijkstra_grid_size = 1;
    }

    thread_local PersistentDeviceBuffer b_curr_node;
    thread_local PersistentDeviceBuffer b_sources;
    thread_local PersistentDeviceBuffer b_U_2D;
    thread_local PersistentDeviceBuffer b_F_2D;
    thread_local PersistentDeviceBuffer b_partial_min;
    thread_local PersistentDeviceBuffer b_len_F;
    thread_local PersistentDeviceBuffer b_min_edge;
    thread_local PersistentDeviceBuffer b_dist_2D;
    thread_local PersistentPinnedBuffer h_sources_stage;
    thread_local SsspRuntime runtime;
    int *d_curr_node = NULL;
    int *d_V = NULL, *d_E = NULL, *d_sources= NULL;
    int *d_U_2D = NULL, *d_F_2D = NULL, *d_len_F = NULL;
    double *d_W = NULL, *d_min_edge = NULL, *d_dist_2D = NULL;
    double *d_partial_min = NULL;
    bool reg_sources = false;
    const int* h_sources = nullptr;
    bool frontier_sssp = false;
    bool parallel_delta = !frontier_sssp && use_parallel_delta_sssp(len_V, len_E);
    int parallel_block_size = 256;
    int parallel_grid_size = std::min(65535, std::max(1, (len_V + parallel_block_size - 1) / parallel_block_size));
    int reduce_grid_size = std::min(4096, parallel_grid_size);
    int frontier_cap = std::max(1, len_E);
    std::vector<double> h_partial_min;
    bool graph_changed = true;
    DeviceCsrView graph_view;

    if (ensure_device_buffer(b_curr_node, sizeof(int), &EG_ret) != EG_GPU_SUCC) goto exit;
    EG_ret = acquire_device_csr(V, E, W, len_V, len_E, true, &graph_view);
    if (EG_ret != EG_GPU_SUCC) goto exit;
    frontier_sssp = use_frontier_weighted_sssp(
        graph_view.stats, len_V, len_E, len_sources);
    parallel_delta = !frontier_sssp && use_parallel_delta_sssp(len_V, len_E);
    if (ensure_device_buffer(b_sources, sizeof(int) * len_sources, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(
            b_U_2D,
            sizeof(int) * (frontier_sssp ? frontier_cap : (parallel_delta ? len_V : dijkstra_grid_size * len_V)),
            &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(
            b_F_2D,
            sizeof(int) * (frontier_sssp ? frontier_cap : (parallel_delta ? len_V : dijkstra_grid_size * len_V)),
            &EG_ret) != EG_GPU_SUCC) goto exit;
    if (parallel_delta && ensure_device_buffer(b_partial_min, sizeof(double) * reduce_grid_size, &EG_ret) != EG_GPU_SUCC) goto exit;
    if ((parallel_delta || frontier_sssp) && ensure_device_buffer(b_len_F, sizeof(int), &EG_ret) != EG_GPU_SUCC) goto exit;
    if (!frontier_sssp && ensure_device_buffer(b_min_edge, sizeof(double) * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_dist_2D, sizeof(double) * len_sources * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;

    d_curr_node = b_curr_node.as<int>();
    d_V = graph_view.d_V;
    d_E = graph_view.d_E;
    d_sources = b_sources.as<int>();
    d_U_2D = b_U_2D.as<int>();
    d_F_2D = b_F_2D.as<int>();
    d_partial_min = b_partial_min.as<double>();
    d_len_F = b_len_F.as<int>();
    d_W = graph_view.d_W;
    d_min_edge = frontier_sssp ? nullptr : b_min_edge.as<double>();
    d_dist_2D = b_dist_2D.as<double>();
    graph_changed = graph_view.structure_changed || graph_view.weights_changed;

    EXIT_IF_CUDA_FAILED(cudaMemset(d_curr_node, 0, sizeof(int)));
    h_sources = prepare_h2d_source(sources, len_sources, h_sources_stage, &reg_sources);
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_sources, h_sources, sizeof(int) * len_sources, cudaMemcpyHostToDevice));

    if (!runtime.ready) {
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.start_event));
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.stop_event));
        runtime.ready = true;
    }
    EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.start_event));

    if (!frontier_sssp && graph_changed) {
        d_calc_min_edge<<<min_edge_grid_size, min_edge_block_size>>>(d_V, d_E, d_W, len_V, len_E, d_min_edge);
        EXIT_IF_CUDA_FAILED(cudaGetLastError());
    }

    if (frontier_sssp) {
        int* d_frontier = d_U_2D;
        int* d_next = d_F_2D;
        int* d_overflow = d_curr_node;
        for (int row = 0; row < len_sources; ++row) {
            double* d_dist = d_dist_2D + (std::size_t)row * (std::size_t)len_V;
            d_sssp_frontier_init<<<parallel_grid_size, parallel_block_size>>>(
                d_sources, d_dist, d_frontier, d_len_F, len_V, row);
            EXIT_IF_CUDA_FAILED(cudaGetLastError());

            int h_len_F = 1;
            int iter = 0;
            int* curr_frontier = d_frontier;
            int* next_frontier = d_next;
            while (h_len_F > 0) {
                int zero = 0;
                EXIT_IF_CUDA_FAILED(cudaMemcpy(d_len_F, &zero, sizeof(int), cudaMemcpyHostToDevice));
                EXIT_IF_CUDA_FAILED(cudaMemcpy(d_overflow, &zero, sizeof(int), cudaMemcpyHostToDevice));
                int relax_grid = std::min(
                    65535,
                    std::max(1, (int)(((long long)h_len_F * (long long)warp_size + parallel_block_size - 1) / parallel_block_size))
                );
                d_sssp_frontier_relax<<<relax_grid, parallel_block_size>>>(
                    d_V, d_E, d_W, curr_frontier, h_len_F, d_dist,
                    next_frontier, d_len_F, d_overflow, frontier_cap, warp_size);
                EXIT_IF_CUDA_FAILED(cudaGetLastError());
                int h_overflow = 0;
                EXIT_IF_CUDA_FAILED(cudaMemcpy(&h_len_F, d_len_F, sizeof(int), cudaMemcpyDeviceToHost));
                EXIT_IF_CUDA_FAILED(cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
                if (h_overflow || h_len_F > frontier_cap) {
                    EG_ret = EG_GPU_DEVICE_ERR;
                    goto exit;
                }
                std::swap(curr_frontier, next_frontier);
                ++iter;
                if (iter > len_V) {
                    EG_ret = EG_GPU_DEVICE_ERR;
                    goto exit;
                }
            }
        }
    } else if (parallel_delta) {
        h_partial_min.assign(reduce_grid_size, EG_DOUBLE_INF);
        for (int row = 0; row < len_sources; ++row) {
            double* d_dist = d_dist_2D + (std::size_t)row * (std::size_t)len_V;
            d_sssp_delta_init<<<parallel_grid_size, parallel_block_size>>>(
                d_sources, d_dist, d_U_2D, d_F_2D, d_len_F, len_V, row);
            EXIT_IF_CUDA_FAILED(cudaGetLastError());

            int h_len_F = 1;
            int iter = 0;
            while (h_len_F > 0) {
                int relax_grid = std::min(
                    65535,
                    std::max(1, (int)(((long long)h_len_F * (long long)warp_size + parallel_block_size - 1) / parallel_block_size))
                );
                d_sssp_delta_relax<<<relax_grid, parallel_block_size>>>(
                    d_V, d_E, d_W, d_F_2D, h_len_F, d_dist, warp_size);
                EXIT_IF_CUDA_FAILED(cudaGetLastError());

                d_sssp_delta_reduce_partial<<<reduce_grid_size, parallel_block_size>>>(
                    d_dist, d_U_2D, d_min_edge, d_partial_min, len_V);
                EXIT_IF_CUDA_FAILED(cudaGetLastError());
                EXIT_IF_CUDA_FAILED(cudaMemcpy(
                    h_partial_min.data(), d_partial_min, sizeof(double) * reduce_grid_size,
                    cudaMemcpyDeviceToHost));

                double h_delta = EG_DOUBLE_INF;
                for (double v : h_partial_min) {
                    if (v < h_delta) h_delta = v;
                }
                if (h_delta >= EG_DOUBLE_INF) {
                    break;
                }

                EXIT_IF_CUDA_FAILED(cudaMemset(d_len_F, 0, sizeof(int)));
                d_sssp_delta_build_frontier<<<parallel_grid_size, parallel_block_size>>>(
                    d_U_2D, d_dist, h_delta, d_F_2D, d_len_F, len_V);
                EXIT_IF_CUDA_FAILED(cudaGetLastError());
                EXIT_IF_CUDA_FAILED(cudaMemcpy(&h_len_F, d_len_F, sizeof(int), cudaMemcpyDeviceToHost));

                ++iter;
                if (iter > len_V) {
                    EG_ret = EG_GPU_DEVICE_ERR;
                    goto exit;
                }
            }
        }
    } else {
        d_sssp_dijkstra<<<dijkstra_grid_size, dijkstra_block_size>>>(d_curr_node, d_V, d_E, d_W, d_min_edge, d_sources,
                                        d_dist_2D, d_U_2D, d_F_2D, len_V, len_E, len_sources, target, warp_size);
        EXIT_IF_CUDA_FAILED(cudaGetLastError());
    }
    EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.stop_event));
    EXIT_IF_CUDA_FAILED(cudaEventSynchronize(runtime.stop_event));
    if (kernel_seconds != nullptr) {
        float elapsed_ms = 0.0f;
        EXIT_IF_CUDA_FAILED(cudaEventElapsedTime(&elapsed_ms, runtime.start_event, runtime.stop_event));
        *kernel_seconds = (double)elapsed_ms * 1e-3;
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(res, d_dist_2D, sizeof(double) * len_sources * len_V, cudaMemcpyDeviceToHost));

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

int cuda_sssp_bellman_ford(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const double* W,
    _IN_ const int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int target,
    _OUT_ double* res,
    _OUT_ double* kernel_seconds
)
{
    (void)target;
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;

    if (len_sources <= 0 || len_V <= 0) {
        return EG_GPU_SUCC;
    }

    thread_local PersistentDeviceBuffer b_sources;
    thread_local PersistentDeviceBuffer b_edge_src;
    thread_local PersistentDeviceBuffer b_dist_2D;
    thread_local PersistentDeviceBuffer b_changed;
    thread_local PersistentPinnedBuffer h_sources_stage;
    thread_local SsspRuntime runtime;

    int* d_V = nullptr;
    int* d_E = nullptr;
    double* d_W = nullptr;
    int* d_sources = nullptr;
    int* d_edge_src = nullptr;
    double* d_dist_2D = nullptr;
    int* d_changed = nullptr;
    const int* h_sources = nullptr;
    bool reg_sources = false;
    DeviceCsrView graph_view;

    const std::size_t total_dist = (std::size_t)len_sources * (std::size_t)len_V;
    const long long relax_work = (long long)len_sources * (long long)len_E;
    const int block = 256;
    const int init_grid = std::min(
        65535,
        std::max(1, (int)(((long long)total_dist + block - 1) / block))
    );
    const int edge_grid = std::min(
        65535,
        std::max(1, (int)(((long long)len_E + block - 1) / block))
    );
    const int relax_grid = std::min(
        65535,
        std::max(1, (int)((relax_work + block - 1) / block))
    );

    EG_ret = acquire_device_csr(V, E, W, len_V, len_E, true, &graph_view);
    if (EG_ret != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_sources, sizeof(int) * len_sources, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_edge_src, sizeof(int) * std::max(1, len_E), &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_dist_2D, sizeof(double) * total_dist, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_changed, sizeof(int), &EG_ret) != EG_GPU_SUCC) goto exit;

    d_V = graph_view.d_V;
    d_E = graph_view.d_E;
    d_W = graph_view.d_W;
    d_sources = b_sources.as<int>();
    d_edge_src = b_edge_src.as<int>();
    d_dist_2D = b_dist_2D.as<double>();
    d_changed = b_changed.as<int>();

    h_sources = prepare_h2d_source(sources, len_sources, h_sources_stage, &reg_sources);
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_sources, h_sources, sizeof(int) * len_sources, cudaMemcpyHostToDevice));

    if (!runtime.ready) {
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.start_event));
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.stop_event));
        runtime.ready = true;
    }
    EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.start_event));

    if (len_E > 0) {
        d_bellman_build_edge_src<<<edge_grid, block>>>(d_V, d_edge_src, len_V);
        EXIT_IF_CUDA_FAILED(cudaGetLastError());
    }
    d_bellman_init<<<init_grid, block>>>(d_sources, d_dist_2D, len_V, len_sources);
    EXIT_IF_CUDA_FAILED(cudaGetLastError());

    if (len_E > 0) {
        int h_changed = 0;
        for (int iter = 0; iter < len_V - 1; ++iter) {
            EXIT_IF_CUDA_FAILED(cudaMemset(d_changed, 0, sizeof(int)));
            d_bellman_relax_edges<<<relax_grid, block>>>(
                d_edge_src, d_E, d_W, d_dist_2D, d_changed,
                len_V, len_E, len_sources);
            EXIT_IF_CUDA_FAILED(cudaGetLastError());
            EXIT_IF_CUDA_FAILED(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_changed == 0) break;
        }

        EXIT_IF_CUDA_FAILED(cudaMemset(d_changed, 0, sizeof(int)));
        d_bellman_relax_edges<<<relax_grid, block>>>(
            d_edge_src, d_E, d_W, d_dist_2D, d_changed,
            len_V, len_E, len_sources);
        EXIT_IF_CUDA_FAILED(cudaGetLastError());
        EXIT_IF_CUDA_FAILED(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_changed != 0) {
            EG_ret = EG_GPU_NEGATIVE_CYCLE;
            goto timed_exit;
        }
    }

timed_exit:
    EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.stop_event));
    EXIT_IF_CUDA_FAILED(cudaEventSynchronize(runtime.stop_event));
    if (kernel_seconds != nullptr) {
        float elapsed_ms = 0.0f;
        EXIT_IF_CUDA_FAILED(cudaEventElapsedTime(&elapsed_ms, runtime.start_event, runtime.stop_event));
        *kernel_seconds = (double)elapsed_ms * 1e-3;
    }

    if (EG_ret == EG_GPU_SUCC) {
        EXIT_IF_CUDA_FAILED(cudaMemcpy(res, d_dist_2D, sizeof(double) * total_dist, cudaMemcpyDeviceToHost));
    }

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

int cuda_sssp_unweighted_bfs(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int target,
    _OUT_ double* res,
    _OUT_ double* kernel_seconds
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;

    if (len_sources <= 0 || len_V <= 0) {
        return EG_GPU_SUCC;
    }

    thread_local PersistentDeviceBuffer b_sources;
    thread_local PersistentDeviceBuffer b_dist_int;
    thread_local PersistentDeviceBuffer b_dist_double;
    thread_local PersistentDeviceBuffer b_frontier_a;
    thread_local PersistentDeviceBuffer b_frontier_b;
    thread_local PersistentDeviceBuffer b_len_F;
    thread_local PersistentDeviceBuffer b_next_len;
    thread_local PersistentDeviceBuffer b_target_found;
    thread_local PersistentPinnedBuffer h_sources_stage;
    thread_local SsspRuntime runtime;

    int* d_V = nullptr;
    int* d_E = nullptr;
    int* d_sources = nullptr;
    int* d_dist_int = nullptr;
    double* d_dist_double = nullptr;
    int* d_frontier_a = nullptr;
    int* d_frontier_b = nullptr;
    int* d_len_F = nullptr;
    int* d_next_len = nullptr;
    int* d_target_found = nullptr;
    const int* h_sources = nullptr;
    bool reg_sources = false;
    int convert_block = 256;
    int convert_grid = 1;
    int bfs_block = 256;
    int bfs_grid = std::min(65535, std::max(1, (len_V + bfs_block - 1) / bfs_block));
    int warp_size = 32;
    bool scan_bfs = use_small_scan_bfs(len_V, len_E, len_sources);
    DeviceCsrView graph_view;

    const std::size_t total = (std::size_t)len_sources * (std::size_t)len_V;
    EG_ret = acquire_device_csr(V, E, nullptr, len_V, len_E, false, &graph_view);
    if (EG_ret != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_sources, sizeof(int) * len_sources, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_dist_int, sizeof(int) * total, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_dist_double, sizeof(double) * total, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_frontier_a, sizeof(int) * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_frontier_b, sizeof(int) * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_len_F, sizeof(int), &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_next_len, sizeof(int), &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_target_found, sizeof(int), &EG_ret) != EG_GPU_SUCC) goto exit;

    d_V = graph_view.d_V;
    d_E = graph_view.d_E;
    d_sources = b_sources.as<int>();
    d_dist_int = b_dist_int.as<int>();
    d_dist_double = b_dist_double.as<double>();
    d_frontier_a = b_frontier_a.as<int>();
    d_frontier_b = b_frontier_b.as<int>();
    d_len_F = b_len_F.as<int>();
    d_next_len = b_next_len.as<int>();
    d_target_found = b_target_found.as<int>();

    h_sources = prepare_h2d_source(sources, len_sources, h_sources_stage, &reg_sources);
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_sources, h_sources, sizeof(int) * len_sources, cudaMemcpyHostToDevice));

    if (!runtime.ready) {
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.start_event));
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.stop_event));
        runtime.ready = true;
    }
    EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.start_event));

    if (scan_bfs) {
        d_sssp_unweighted_bfs<<<len_sources, bfs_block>>>(
            d_V, d_E, d_sources, d_dist_int, len_V, len_sources, target);
        EXIT_IF_CUDA_FAILED(cudaGetLastError());
    } else {
        for (int row = 0; row < len_sources; ++row) {
            int* d_dist = d_dist_int + (std::size_t)row * (std::size_t)len_V;
            d_bfs_frontier_init<<<bfs_grid, bfs_block>>>(
                d_sources, d_dist, d_frontier_a, d_len_F, len_V, row);
            EXIT_IF_CUDA_FAILED(cudaGetLastError());

            int h_len_F = 1;
            int h_target_found = 0;
            int level = 0;
            int iter = 0;
            int* curr_frontier = d_frontier_a;
            int* next_frontier = d_frontier_b;
            while (h_len_F > 0 && h_target_found == 0) {
                EXIT_IF_CUDA_FAILED(cudaMemset(d_next_len, 0, sizeof(int)));
                EXIT_IF_CUDA_FAILED(cudaMemset(d_target_found, 0, sizeof(int)));
                int expand_grid = std::min(
                    65535,
                    std::max(1, (int)(((long long)h_len_F * (long long)warp_size + bfs_block - 1) / bfs_block))
                );
                d_bfs_expand_frontier<<<expand_grid, bfs_block>>>(
                    d_V, d_E, curr_frontier, h_len_F, d_dist,
                    next_frontier, d_next_len, d_target_found,
                    target, level, warp_size);
                EXIT_IF_CUDA_FAILED(cudaGetLastError());
                EXIT_IF_CUDA_FAILED(cudaMemcpy(&h_len_F, d_next_len, sizeof(int), cudaMemcpyDeviceToHost));
                if (target >= 0) {
                    EXIT_IF_CUDA_FAILED(cudaMemcpy(&h_target_found, d_target_found, sizeof(int), cudaMemcpyDeviceToHost));
                }
                std::swap(curr_frontier, next_frontier);
                ++level;
                ++iter;
                if (iter > len_V) {
                    EG_ret = EG_GPU_DEVICE_ERR;
                    goto exit;
                }
            }
        }
    }

    convert_grid = std::min<int>((int)((total + convert_block - 1) / convert_block), 65535);
    if (convert_grid < 1) convert_grid = 1;
    d_bfs_int_to_double<<<convert_grid, convert_block>>>(d_dist_int, d_dist_double, (int)total);
    EXIT_IF_CUDA_FAILED(cudaGetLastError());

    EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.stop_event));
    EXIT_IF_CUDA_FAILED(cudaEventSynchronize(runtime.stop_event));
    if (kernel_seconds != nullptr) {
        float elapsed_ms = 0.0f;
        EXIT_IF_CUDA_FAILED(cudaEventElapsedTime(&elapsed_ms, runtime.start_event, runtime.stop_event));
        *kernel_seconds = (double)elapsed_ms * 1e-3;
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(res, d_dist_double, sizeof(double) * total, cudaMemcpyDeviceToHost));

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
