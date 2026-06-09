#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <stdlib.h>
#include <string>

#include "buffer_cache.h"
#include "common.h"
#include "device_graph_cache.h"

namespace gpu_easygraph {

static inline bool env_flag_enabled(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return default_value;
    return raw[0] == '1' || raw[0] == 'T' || raw[0] == 't' ||
           raw[0] == 'Y' || raw[0] == 'y' || raw[0] == 'O' || raw[0] == 'o';
}

static inline int env_int_value(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return default_value;
    if (std::strcmp(raw, "AUTO") == 0 || std::strcmp(raw, "auto") == 0) {
        return default_value;
    }
    char* end = nullptr;
    long parsed = std::strtol(raw, &end, 10);
    if (end == raw || parsed < 0) return default_value;
    if (parsed > 2000000000L) return 2000000000;
    return (int)parsed;
}

static inline double env_double_value(const char* name, double default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) return default_value;
    char* end = nullptr;
    double parsed = std::strtod(raw, &end);
    if (end == raw || parsed < 0.0) return default_value;
    return parsed;
}

static inline bool should_use_kcore_single_block(
    const HostCsrStats& stats,
    int len_V,
    int len_E
) {
    if (!env_flag_enabled("EASYGRAPH_GPU_KCORE_SINGLE_BLOCK", true)) return false;
    if (len_V > env_int_value("EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MAX_NODES", 50000)) return false;
    if (len_E > env_int_value("EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MAX_EDGE_SLOTS", 700000)) return false;

    const int easy_node_cut = env_int_value("EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_EASY_MAX_NODES", 20000);
    if (len_V <= easy_node_cut) return true;
    if (!stats.ready) return false;

    const double min_avg_degree = env_double_value(
        "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_AVG_DEGREE",
        10.0
    );
    const int min_max_degree = env_int_value(
        "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE",
        2000000000
    );
    return stats.avg_degree >= min_avg_degree || stats.max_degree >= min_max_degree;
}

static __global__ void d_calc_deg(
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ int* d_deg
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;
    for (int u = tid; u < len_V; u += tnum) {
        d_deg[u] = d_V[u + 1] - d_V[u];
    }
}

static __global__ void d_k_core_scan_enqueue(
    _IN_ int* d_deg,
    _OUT_ int* d_core,
    _OUT_ int* d_processed,
    _IN_ int len_V,
    _IN_ int level,
    _OUT_ int* d_queue,
    _OUT_ int* d_queue_len,
    _OUT_ int* d_processed_count
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;
    for (int u = tid; u < len_V; u += tnum) {
        if (d_processed[u] == 0 && d_deg[u] <= level) {
            if (atomicCAS(d_processed + u, 0, 1) == 0) {
                d_core[u] = level;
                int pos = atomicAdd(d_queue_len, 1);
                if (pos < len_V) d_queue[pos] = u;
                atomicAdd(d_processed_count, 1);
            }
        }
    }
}

static __global__ void d_k_core_process_queue(
    _IN_ int* d_V,
    _IN_ int* d_E,
    _OUT_ int* d_deg,
    _OUT_ int* d_core,
    _OUT_ int* d_processed,
    _IN_ int len_V,
    _IN_ int level,
    _OUT_ int* d_queue,
    _IN_ int start,
    _IN_ int end,
    _OUT_ int* d_queue_len,
    _OUT_ int* d_processed_count
)
{
    const int warp_size = 32;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tnum = (long long)blockDim.x * gridDim.x;
    long long work = (long long)(end - start) * warp_size;

    for (long long j = tid; j < work; j += tnum) {
        int q_idx = start + (int)(j / warp_size);
        int lane = (int)(j % warp_size);
        int v = d_queue[q_idx];
        int edge_start = d_V[v];
        int edge_end = d_V[v + 1];
        for (int e = edge_start + lane; e < edge_end; e += warp_size) {
            int nbr = d_E[e];
            if (d_processed[nbr] != 0) continue;
            int old = atomicSub(d_deg + nbr, 1);
            if (old == level + 1) {
                if (atomicCAS(d_processed + nbr, 0, 1) == 0) {
                    d_core[nbr] = level;
                    int pos = atomicAdd(d_queue_len, 1);
                    if (pos < len_V) d_queue[pos] = nbr;
                    atomicAdd(d_processed_count, 1);
                }
            } else if (old <= level) {
                atomicAdd(d_deg + nbr, 1);
            }
        }
    }
}

static __global__ void d_k_core_find_min_unprocessed(
    _IN_ int* d_deg,
    _IN_ int* d_processed,
    _IN_ int len_V,
    _OUT_ int* d_min_level
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;
    int local_min = 0x7fffffff;

    for (int u = tid; u < len_V; u += tnum) {
        if (d_processed[u] == 0) {
            int deg = d_deg[u];
            if (deg < local_min) local_min = deg;
        }
    }

    __shared__ int s_min[256];
    s_min[threadIdx.x] = local_min;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && s_min[threadIdx.x + stride] < s_min[threadIdx.x]) {
            s_min[threadIdx.x] = s_min[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicMin(d_min_level, s_min[0]);
    }
}

static __global__ void d_k_core_single_block(
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ int len_V,
    _OUT_ int* d_deg,
    _OUT_ int* d_core,
    _OUT_ int* d_processed,
    _OUT_ int* d_queue
)
{
    const int warp_size = 32;
    const int tid = threadIdx.x;
    __shared__ int s_level;
    __shared__ int s_processed_count;
    __shared__ int s_queue_len;
    __shared__ int s_min_level;

    if (tid == 0) {
        s_level = 0;
        s_processed_count = 0;
    }

    for (int u = tid; u < len_V; u += blockDim.x) {
        d_deg[u] = d_V[u + 1] - d_V[u];
        d_core[u] = 0;
        d_processed[u] = 0;
    }
    __syncthreads();

    while (s_processed_count < len_V && s_level <= len_V) {
        if (tid == 0) {
            s_queue_len = 0;
        }
        __syncthreads();

        for (int u = tid; u < len_V; u += blockDim.x) {
            if (d_processed[u] == 0 && d_deg[u] <= s_level) {
                if (atomicCAS(d_processed + u, 0, 1) == 0) {
                    d_core[u] = s_level;
                    int pos = atomicAdd(&s_queue_len, 1);
                    if (pos < len_V) d_queue[pos] = u;
                    atomicAdd(&s_processed_count, 1);
                }
            }
        }
        __syncthreads();

        int offset = 0;
        while (offset < s_queue_len) {
            int end = s_queue_len;
            long long work = (long long)(end - offset) * (long long)warp_size;
            for (long long j = tid; j < work; j += blockDim.x) {
                int q_idx = offset + (int)(j / warp_size);
                int lane = (int)(j % warp_size);
                int v = d_queue[q_idx];
                int edge_start = d_V[v];
                int edge_end = d_V[v + 1];
                for (int e = edge_start + lane; e < edge_end; e += warp_size) {
                    int nbr = d_E[e];
                    if (d_processed[nbr] != 0) continue;
                    int old = atomicSub(d_deg + nbr, 1);
                    if (old == s_level + 1) {
                        if (atomicCAS(d_processed + nbr, 0, 1) == 0) {
                            d_core[nbr] = s_level;
                            int pos = atomicAdd(&s_queue_len, 1);
                            if (pos < len_V) d_queue[pos] = nbr;
                            atomicAdd(&s_processed_count, 1);
                        }
                    } else if (old <= s_level) {
                        atomicAdd(d_deg + nbr, 1);
                    }
                }
            }
            __syncthreads();
            offset = end;
        }

        if (s_processed_count < len_V) {
            if (tid == 0) {
                s_min_level = 0x7fffffff;
            }
            __syncthreads();
            for (int u = tid; u < len_V; u += blockDim.x) {
                if (d_processed[u] == 0) {
                    atomicMin(&s_min_level, d_deg[u]);
                }
            }
            __syncthreads();
            if (tid == 0) {
                if (s_min_level != 0x7fffffff && s_min_level > s_level + 1) {
                    s_level = s_min_level;
                } else {
                    ++s_level;
                }
            }
            __syncthreads();
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

struct KCoreRuntime {
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    bool ready = false;
    ~KCoreRuntime() {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
    }
};


int cuda_k_core (
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ int* k_core_res,
    _OUT_ double* kernel_seconds
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;

    int calc_deg_block_size = 256;
    int calc_deg_grid_size;
    int block_size = 256;
    int single_block_size = env_int_value("EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS", 1024);
    if (single_block_size < 32) single_block_size = 32;
    if (single_block_size > 1024) single_block_size = 1024;
    single_block_size = ((single_block_size + 31) / 32) * 32;
    int grid_size = std::min(65535, std::max(1, (len_V + block_size - 1) / block_size));

    calc_deg_grid_size = std::min(65535, std::max(1, (len_V + calc_deg_block_size - 1) / calc_deg_block_size));

    int processed_count = 0, level = 0;

    thread_local PersistentDeviceBuffer b_deg;
    thread_local PersistentDeviceBuffer b_core;
    thread_local PersistentDeviceBuffer b_processed;
    thread_local PersistentDeviceBuffer b_queue;
    thread_local PersistentDeviceBuffer b_queue_len;
    thread_local PersistentDeviceBuffer b_processed_count;
    thread_local PersistentDeviceBuffer b_min_level;
    thread_local KCoreRuntime runtime;

    int *d_V = NULL, *d_E = NULL, *d_deg = NULL, *d_core = NULL,
        *d_processed = NULL, *d_queue = NULL, *d_queue_len = NULL,
        *d_processed_count = NULL, *d_min_level = NULL;
    DeviceCsrView graph_view;

    EG_ret = acquire_device_csr(V, E, nullptr, len_V, len_E, false, &graph_view);
    if (EG_ret != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_deg, sizeof(int) * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_core, sizeof(int) * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_processed, sizeof(int) * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_queue, sizeof(int) * len_V, &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_queue_len, sizeof(int), &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_processed_count, sizeof(int), &EG_ret) != EG_GPU_SUCC) goto exit;
    if (ensure_device_buffer(b_min_level, sizeof(int), &EG_ret) != EG_GPU_SUCC) goto exit;

    d_V = graph_view.d_V;
    d_E = graph_view.d_E;
    d_deg = b_deg.as<int>();
    d_core = b_core.as<int>();
    d_processed = b_processed.as<int>();
    d_queue = b_queue.as<int>();
    d_queue_len = b_queue_len.as<int>();
    d_processed_count = b_processed_count.as<int>();
    d_min_level = b_min_level.as<int>();

    if (!runtime.ready) {
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.start_event));
        EXIT_IF_CUDA_FAILED(cudaEventCreate(&runtime.stop_event));
        runtime.ready = true;
    }
    EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.start_event));

    if (should_use_kcore_single_block(graph_view.stats, len_V, len_E)) {
        d_k_core_single_block<<<1, single_block_size>>>(
            d_V, d_E, len_V, d_deg, d_core, d_processed, d_queue);
        EXIT_IF_CUDA_FAILED(cudaGetLastError());
        goto timed_copy;
    }

    d_calc_deg<<<calc_deg_grid_size, calc_deg_block_size>>>(d_V, d_E, len_V, len_E, d_deg);
    EXIT_IF_CUDA_FAILED(cudaGetLastError());
    EXIT_IF_CUDA_FAILED(cudaMemset(d_core, 0, sizeof(int) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMemset(d_processed, 0, sizeof(int) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMemset(d_processed_count, 0, sizeof(int)));

    while (processed_count < len_V) {
        int queue_len = 0;
        int offset = 0;
        EXIT_IF_CUDA_FAILED(cudaMemset(d_queue_len, 0, sizeof(int)));

        d_k_core_scan_enqueue<<<grid_size, block_size>>>(
            d_deg, d_core, d_processed, len_V, level, d_queue, d_queue_len, d_processed_count);
        EXIT_IF_CUDA_FAILED(cudaGetLastError());
        EXIT_IF_CUDA_FAILED(cudaMemcpy(&queue_len, d_queue_len, sizeof(int), cudaMemcpyDeviceToHost));

        while (offset < queue_len) {
            int segment = queue_len - offset;
            int process_grid = std::min(
                65535,
                std::max(1, (int)(((long long)segment * 32LL + block_size - 1) / block_size))
            );
            d_k_core_process_queue<<<process_grid, block_size>>>(
                d_V, d_E, d_deg, d_core, d_processed, len_V, level,
                d_queue, offset, queue_len, d_queue_len, d_processed_count);
            EXIT_IF_CUDA_FAILED(cudaGetLastError());
            offset = queue_len;
            EXIT_IF_CUDA_FAILED(cudaMemcpy(&queue_len, d_queue_len, sizeof(int), cudaMemcpyDeviceToHost));
            if (queue_len > len_V) {
                EG_ret = EG_GPU_DEVICE_ERR;
                goto exit;
            }
        }

        EXIT_IF_CUDA_FAILED(cudaMemcpy(&processed_count, d_processed_count, sizeof(int), cudaMemcpyDeviceToHost));

        if (processed_count < len_V) {
            int h_min_level = 0x7fffffff;
            EXIT_IF_CUDA_FAILED(cudaMemcpy(d_min_level, &h_min_level, sizeof(int), cudaMemcpyHostToDevice));
            d_k_core_find_min_unprocessed<<<grid_size, block_size>>>(
                d_deg, d_processed, len_V, d_min_level);
            EXIT_IF_CUDA_FAILED(cudaGetLastError());
            EXIT_IF_CUDA_FAILED(cudaMemcpy(&h_min_level, d_min_level, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_min_level != 0x7fffffff && h_min_level > level + 1) {
                level = h_min_level;
            } else {
                ++level;
            }
        }
        if (level > len_V) {
            EG_ret = EG_GPU_DEVICE_ERR;
            goto exit;
        }
    }

timed_copy:
    EXIT_IF_CUDA_FAILED(cudaEventRecord(runtime.stop_event));
    EXIT_IF_CUDA_FAILED(cudaEventSynchronize(runtime.stop_event));
    if (kernel_seconds != nullptr) {
        float elapsed_ms = 0.0f;
        EXIT_IF_CUDA_FAILED(cudaEventElapsedTime(&elapsed_ms, runtime.start_event, runtime.stop_event));
        *kernel_seconds = (double)elapsed_ms * 1e-3;
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(k_core_res, d_core, sizeof(int) * len_V, cudaMemcpyDeviceToHost));

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
