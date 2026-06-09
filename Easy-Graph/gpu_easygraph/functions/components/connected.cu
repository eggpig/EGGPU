#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "components/connected.cuh"
#include "buffer_cache.h"
#include "device_graph_cache.h"
#include "err.h"

namespace gpu_easygraph {

namespace {

struct ConnectedWorkspace {
    PersistentDeviceBuffer d_parent;
    PersistentDeviceBuffer d_changed;
    PersistentDeviceBuffer d_labels;
    PersistentDeviceBuffer d_rev_V;
    PersistentDeviceBuffer d_rev_E;
    PersistentDeviceBuffer d_forward;
    PersistentDeviceBuffer d_backward;
    PersistentDeviceBuffer d_comp;
    PersistentDeviceBuffer d_valid;
    PersistentDeviceBuffer d_frontier;
    PersistentDeviceBuffer d_next_frontier;
    PersistentDeviceBuffer d_next_size;
    PersistentDeviceBuffer d_indeg;
    PersistentDeviceBuffer d_outdeg;
    PersistentDeviceBuffer d_trim_count;
    PersistentDeviceBuffer d_next_cid;
    PersistentDeviceBuffer d_pivot;
    PersistentDeviceBuffer d_pivot_score;
    PersistentPinnedBuffer h_rev_V_stage;
    PersistentPinnedBuffer h_rev_E_stage;

    std::vector<int> cached_rev_V;
    std::vector<int> cached_rev_E;
    const int* rev_sig_V = nullptr;
    const int* rev_sig_E = nullptr;
    int rev_sig_n = -1;
    int rev_sig_m = -1;
    int rev_sig_v0 = 0;
    int rev_sig_vn = 0;
    int rev_sig_e0 = 0;
    int rev_sig_em = 0;
    bool rev_ready = false;

    cudaEvent_t ev_begin = nullptr;
    cudaEvent_t ev_end = nullptr;
    bool runtime_ready = false;

    ~ConnectedWorkspace() {
        if (ev_begin != nullptr) cudaEventDestroy(ev_begin);
        if (ev_end != nullptr) cudaEventDestroy(ev_end);
    }
};

static thread_local ConnectedWorkspace g_conn_ws;

__global__ void k_init_parents(int* parent, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n) parent[u] = u;
}

template <int SampleCount>
__global__ void k_sample_hook(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int* __restrict__ parent,
    int n,
    int* __restrict__ changed
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    int begin = __ldg(&row_ptr[u]);
    int end = __ldg(&row_ptr[u + 1]);
    #pragma unroll
    for (int t = 0; t < SampleCount; ++t) {
        int p = begin + t;
        if (p >= end) break;
        int v = __ldg(&col_idx[p]);
        int pu = parent[u];
        int pv = parent[v];
        while (pu != pv) {
            int hi = pu > pv ? pu : pv;
            int lo = pu ^ pv ^ hi;
            int old = atomicMin(&parent[hi], lo);
            if (old == hi) {
                atomicExch(changed, 1);
                break;
            }
            pu = parent[pu];
            pv = parent[pv];
        }
    }
}

__global__ void k_hook_csr_edges(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int n,
    int* __restrict__ parent,
    int* __restrict__ changed
) {
    const int warp_size = 32;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long tnum = (long long)blockDim.x * gridDim.x;
    long long work = (long long)n * (long long)warp_size;

    for (long long j = tid; j < work; j += tnum) {
        int u = (int)(j / warp_size);
        int lane = (int)(j % warp_size);
        int begin = __ldg(&row_ptr[u]);
        int end = __ldg(&row_ptr[u + 1]);
        for (int p = begin + lane; p < end; p += warp_size) {
            int v = __ldg(&col_idx[p]);
            if (u == v || v < 0 || v >= n) continue;
            int pu = parent[u];
            int pv = parent[v];
            while (pu != pv) {
                int hi = pu > pv ? pu : pv;
                int lo = pu ^ pv ^ hi;
                int old = atomicMin(&parent[hi], lo);
                if (old == hi) {
                    atomicExch(changed, 1);
                    break;
                }
                pu = parent[pu];
                pv = parent[pv];
            }
        }
    }
}

__global__ void k_compress(int* parent, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    int p = parent[u];
    int gp = parent[p];
    if (p != gp) parent[u] = gp;
}

__global__ void k_assign_roots(const int* parent, int* labels, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n) labels[u] = parent[u];
}

__global__ void k_init_array_int(int* a, int value, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n) a[u] = value;
}

__global__ void k_mark_valid_nodes(int* valid, int value, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n) valid[u] = value;
}

__global__ void k_find_pivot_min(const int* __restrict__ valid, int n, int* pivot) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n && valid[u] == 1) atomicMin(pivot, u);
}

__global__ void k_find_pivot_max_degree(
    const int* __restrict__ valid,
    const int* __restrict__ indeg,
    const int* __restrict__ outdeg,
    int n,
    unsigned long long* best
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n || valid[u] != 1) return;
    int raw_in = indeg[u];
    int raw_out = outdeg[u];
    unsigned int din = (unsigned int)(raw_in > 0 ? raw_in : 0);
    unsigned int dout = (unsigned int)(raw_out > 0 ? raw_out : 0);
    unsigned int score = din + dout;
    if (score < din) score = 0xffffffffu;
    unsigned int tie = 0xffffffffu - (unsigned int)u;
    unsigned long long key = ((unsigned long long)score << 32) | (unsigned long long)tie;
    atomicMax(best, key);
}

__device__ __forceinline__ int warp_agg_push_int(int success, int* __restrict__ counter) {
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, success);
    if (!mask) return -1;
    int lane = threadIdx.x & 31;
    int leader = __ffs(mask) - 1;
    int votes = __popc(mask);
    int base = 0;
    if (lane == leader) base = atomicAdd(counter, votes);
    base = __shfl_sync(active, base, leader);
    int offset = __popc(mask & ((1u << lane) - 1));
    return success ? base + offset : -1;
}

__global__ void k_scc_bfs_expand(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int* __restrict__ visited,
    const int* __restrict__ frontier,
    int frontier_size,
    int* __restrict__ next_frontier,
    int* __restrict__ next_size,
    const int* __restrict__ valid,
    int visit_mark
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = tid >> 5;
    int warp_count = (blockDim.x * gridDim.x) >> 5;
    if (warp_count <= 0) return;
    for (int idx = warp_id; idx < frontier_size; idx += warp_count) {
        int u = frontier[idx];
        int begin = __ldg(&row_ptr[u]);
        int end = __ldg(&row_ptr[u + 1]);
        for (int p = begin + lane; p < end; p += 32) {
            int v = __ldg(&col_idx[p]);
            int ok = 0;
            if (valid[v] == 1) {
                int old = atomicExch(visited + v, visit_mark);
                ok = old != visit_mark;
            }
            int pos = warp_agg_push_int(ok, next_size);
            if (pos >= 0) next_frontier[pos] = v;
        }
    }
}

__global__ void k_scc_intersect_assign(
    const int* __restrict__ forward,
    const int* __restrict__ backward,
    int* __restrict__ valid,
    int* __restrict__ comp,
    int n,
    int cid,
    int visit_mark
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n && valid[u] == 1 && forward[u] == visit_mark && backward[u] == visit_mark) {
        comp[u] = cid;
        valid[u] = 0;
    }
}

__global__ void k_outdeg(const int* row_ptr, int* outdeg, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n) outdeg[u] = row_ptr[u + 1] - row_ptr[u];
}

__global__ void k_trim_singletons_assign(
    const int* __restrict__ indeg,
    const int* __restrict__ outdeg,
    int* __restrict__ valid,
    int* __restrict__ comp,
    int n,
    int* __restrict__ next_cid,
    int* __restrict__ trimmed
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n && valid[u] == 1 && (indeg[u] == 0 || outdeg[u] == 0)) {
        int cid = atomicAdd(next_cid, 1);
        comp[u] = cid;
        valid[u] = 0;
        if (trimmed != nullptr) atomicAdd(trimmed, 1);
    }
}

__global__ void k_active_degrees(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const int* __restrict__ valid,
    int* __restrict__ indeg,
    int* __restrict__ outdeg,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;
    for (int u = tid; u < n; u += tnum) {
        if (valid[u] != 1) {
            outdeg[u] = 0;
            continue;
        }
        int local_out = 0;
        int begin = __ldg(&row_ptr[u]);
        int end = __ldg(&row_ptr[u + 1]);
        for (int p = begin; p < end; ++p) {
            int v = __ldg(&col_idx[p]);
            if (v >= 0 && v < n && valid[v] == 1) {
                ++local_out;
                atomicAdd(indeg + v, 1);
            }
        }
        outdeg[u] = local_out;
    }
}

static bool env_flag_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::toupper(c);
    });
    return s == "1" || s == "TRUE" || s == "ON" || s == "YES";
}

static int env_int_value(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value) return default_value;
    if (parsed < 0) return 0;
    if (parsed > 1000000) return 1000000;
    return (int)parsed;
}

static bool reverse_signature_matches(
    const ConnectedWorkspace& ws,
    const int* V,
    const int* E,
    int n,
    int m
) {
    if (!ws.rev_ready) return false;
    if (ws.rev_sig_V != V || ws.rev_sig_E != E) return false;
    if (ws.rev_sig_n != n || ws.rev_sig_m != m) return false;
    if (n <= 0) return true;
    if (ws.rev_sig_v0 != V[0] || ws.rev_sig_vn != V[n]) return false;
    if (m > 0 && (ws.rev_sig_e0 != E[0] || ws.rev_sig_em != E[m - 1])) return false;
    return true;
}

static void update_reverse_signature(
    ConnectedWorkspace& ws,
    const int* V,
    const int* E,
    int n,
    int m
) {
    ws.rev_sig_V = V;
    ws.rev_sig_E = E;
    ws.rev_sig_n = n;
    ws.rev_sig_m = m;
    ws.rev_sig_v0 = (n > 0) ? V[0] : 0;
    ws.rev_sig_vn = (n > 0) ? V[n] : 0;
    ws.rev_sig_e0 = (m > 0) ? E[0] : 0;
    ws.rev_sig_em = (m > 0) ? E[m - 1] : 0;
    ws.rev_ready = true;
}

static void build_reverse_csr_cached(
    ConnectedWorkspace& ws,
    const int* V,
    const int* E,
    int n,
    int m
) {
    if (reverse_signature_matches(ws, V, E, n, m)) return;
    std::vector<int> indeg(n, 0);
    for (int u = 0; u < n; ++u) {
        int begin = std::max(0, V[u]);
        int end = std::min(m, V[u + 1]);
        for (int p = begin; p < end; ++p) {
            int v = E[p];
            if (v >= 0 && v < n && v != u) indeg[v] += 1;
        }
    }
    ws.cached_rev_V.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) ws.cached_rev_V[i + 1] = ws.cached_rev_V[i] + indeg[i];
    ws.cached_rev_E.assign(ws.cached_rev_V[n], 0);
    std::vector<int> cur = ws.cached_rev_V;
    for (int u = 0; u < n; ++u) {
        int begin = std::max(0, V[u]);
        int end = std::min(m, V[u + 1]);
        for (int p = begin; p < end; ++p) {
            int v = E[p];
            if (v >= 0 && v < n && v != u) {
                ws.cached_rev_E[cur[v]++] = u;
            }
        }
    }
    update_reverse_signature(ws, V, E, n, m);
}

static inline int grid_for(int n, int block_size, int max_blocks = 65535) {
    if (n <= 0) return 1;
    int g = (n + block_size - 1) / block_size;
    return std::max(1, std::min(g, max_blocks));
}

static int run_scc_bfs(
    const int* d_row_ptr,
    const int* d_col_idx,
    int n,
    int pivot,
    const int* d_valid,
    int* d_visited,
    int* d_frontier,
    int* d_next_frontier,
    int* d_next_size,
    int max_blocks,
    int visit_mark
) {
    const int bs = 256;
    cudaError_t ret = cudaMemcpy(d_visited + pivot, &visit_mark, sizeof(int), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) return EG_GPU_DEVICE_ERR;
    ret = cudaMemcpy(d_frontier, &pivot, sizeof(int), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) return EG_GPU_DEVICE_ERR;
    int frontier_size = 1;
    while (frontier_size > 0) {
        ret = cudaMemset(d_next_size, 0, sizeof(int));
        if (ret != cudaSuccess) return EG_GPU_DEVICE_ERR;
        int grid = std::min(max_blocks, std::max(1, (frontier_size + bs - 1) / bs));
        long long warp_threads = std::max(32LL, (long long)frontier_size * 32LL);
        grid = std::min(max_blocks, std::max(1, (int)((warp_threads + bs - 1) / bs)));
        k_scc_bfs_expand<<<grid, bs>>>(
            d_row_ptr, d_col_idx, d_visited, d_frontier,
            frontier_size, d_next_frontier, d_next_size, d_valid, visit_mark);
        ret = cudaGetLastError();
        if (ret != cudaSuccess) return EG_GPU_DEVICE_ERR;
        ret = cudaMemcpy(&frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);
        if (ret != cudaSuccess) return EG_GPU_DEVICE_ERR;
        std::swap(d_frontier, d_next_frontier);
    }
    return EG_GPU_SUCC;
}

static int run_active_singleton_trim(
    const int* d_row_ptr,
    const int* d_col_idx,
    int n,
    int* d_valid,
    int* d_comp,
    int* d_indeg,
    int* d_outdeg,
    int* d_next_cid,
    int* d_trim_count,
    int grid,
    int block,
    int max_iters
) {
    if (max_iters <= 0) return EG_GPU_SUCC;
    for (int iter = 0; iter < max_iters; ++iter) {
        cudaError_t ret = cudaMemset(d_indeg, 0, (size_t)n * sizeof(int));
        if (ret != cudaSuccess) return EG_GPU_DEVICE_ERR;
        ret = cudaMemset(d_outdeg, 0, (size_t)n * sizeof(int));
        if (ret != cudaSuccess) return EG_GPU_DEVICE_ERR;
        k_active_degrees<<<grid, block>>>(d_row_ptr, d_col_idx, d_valid, d_indeg, d_outdeg, n);
        ret = cudaGetLastError();
        if (ret != cudaSuccess) return EG_GPU_DEVICE_ERR;
        ret = cudaMemset(d_trim_count, 0, sizeof(int));
        if (ret != cudaSuccess) return EG_GPU_DEVICE_ERR;
        k_trim_singletons_assign<<<grid, block>>>(
            d_indeg, d_outdeg, d_valid, d_comp, n, d_next_cid, d_trim_count);
        ret = cudaGetLastError();
        if (ret != cudaSuccess) return EG_GPU_DEVICE_ERR;
        int trimmed = 0;
        ret = cudaMemcpy(&trimmed, d_trim_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (ret != cudaSuccess) return EG_GPU_DEVICE_ERR;
        if (trimmed <= 0) break;
    }
    return EG_GPU_SUCC;
}

} // namespace

int cuda_connected_components(
    const int* V,
    const int* E,
    int len_V,
    int len_E,
    std::vector<int>& labels,
    double* kernel_seconds
) {
    labels.clear();
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    if (len_V <= 0) return EG_GPU_SUCC;

    const int n = len_V;
    const int m = len_E;
    labels.resize(n);
    if (m <= 0) {
        for (int i = 0; i < n; ++i) labels[i] = i;
        return EG_GPU_SUCC;
    }
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    int* d_parent = nullptr;
    int* d_changed = nullptr;
    int* d_labels = nullptr;
    int status = EG_GPU_SUCC;
    const int bs_vertex = 256;
    const int bs_edge = 512;
    int gv = grid_for(n, bs_vertex);
    int ge = grid_for((int)std::min<long long>((long long)n * 32LL, 2147483647LL), bs_edge);
    int h_changed = 0;
    DeviceCsrView graph_view;

    auto fail = [&](cudaError_t err, int code) {
        if (err != cudaSuccess) status = code;
    };
    auto fail_if_status = [&](int rc) {
        if (rc != EG_GPU_SUCC) status = rc;
    };

    fail_if_status(acquire_device_csr(V, E, nullptr, n, m, false, &graph_view));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_parent.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_changed.ensure_bytes(sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_labels.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;

    d_row_ptr = graph_view.d_V;
    d_col_idx = graph_view.d_E;
    d_parent = g_conn_ws.d_parent.as<int>();
    d_changed = g_conn_ws.d_changed.as<int>();
    d_labels = g_conn_ws.d_labels.as<int>();

    if (!g_conn_ws.runtime_ready) {
        fail(cudaEventCreate(&g_conn_ws.ev_begin), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaEventCreate(&g_conn_ws.ev_end), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        g_conn_ws.runtime_ready = true;
    }

    fail(cudaEventRecord(g_conn_ws.ev_begin), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    k_init_parents<<<gv, bs_vertex>>>(d_parent, n);
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    fail(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    k_sample_hook<2><<<gv, bs_vertex>>>(d_row_ptr, d_col_idx, d_parent, n, d_changed);
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    k_compress<<<gv, bs_vertex>>>(d_parent, n);
    k_compress<<<gv, bs_vertex>>>(d_parent, n);
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    h_changed = 1;
    while (h_changed) {
        h_changed = 0;
        fail(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        k_hook_csr_edges<<<ge, bs_edge>>>(d_row_ptr, d_col_idx, n, d_parent, d_changed);
        fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        k_compress<<<gv, bs_vertex>>>(d_parent, n);
        k_compress<<<gv, bs_vertex>>>(d_parent, n);
        fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
    }

    k_compress<<<gv, bs_vertex>>>(d_parent, n);
    k_compress<<<gv, bs_vertex>>>(d_parent, n);
    k_compress<<<gv, bs_vertex>>>(d_parent, n);
    k_assign_roots<<<gv, bs_vertex>>>(d_parent, d_labels, n);
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    fail(cudaEventRecord(g_conn_ws.ev_end), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail(cudaEventSynchronize(g_conn_ws.ev_end), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    if (kernel_seconds != nullptr) {
        float ms = 0.0f;
        fail(cudaEventElapsedTime(&ms, g_conn_ws.ev_begin, g_conn_ws.ev_end), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        *kernel_seconds = (double)ms / 1000.0;
    }

    fail(cudaMemcpy(labels.data(), d_labels, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

cleanup:
    if (status != EG_GPU_SUCC) {
        if (g_conn_ws.ev_begin != nullptr) {
            cudaEventDestroy(g_conn_ws.ev_begin);
            g_conn_ws.ev_begin = nullptr;
        }
        if (g_conn_ws.ev_end != nullptr) {
            cudaEventDestroy(g_conn_ws.ev_end);
            g_conn_ws.ev_end = nullptr;
        }
        g_conn_ws.runtime_ready = false;
    }
    return status;
}

int cuda_strongly_connected_components(
    const int* V,
    const int* E,
    int len_V,
    int len_E,
    std::vector<int>& labels,
    double* kernel_seconds
) {
    labels.clear();
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    if (len_V <= 0) return EG_GPU_SUCC;
    labels.resize(len_V);
    if (len_E <= 0) {
        for (int i = 0; i < len_V; ++i) labels[i] = i;
        return EG_GPU_SUCC;
    }

    const int n = len_V;
    const int m = len_E;
    const int bs = 256;
    const int gv = grid_for(n, bs);
    int device = 0;
    int sms = 0;
    int max_frontier_blocks = 4096;
    (void)cudaGetDevice(&device);
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess && sms > 0) {
        max_frontier_blocks = sms * 32;
    }

    int status = EG_GPU_SUCC;
    DeviceCsrView graph_view;
    int* d_V = nullptr;
    int* d_E = nullptr;
    int* d_rev_V = nullptr;
    int* d_rev_E = nullptr;
    int* d_forward = nullptr;
    int* d_backward = nullptr;
    int* d_comp = nullptr;
    int* d_valid = nullptr;
    int* d_frontier = nullptr;
    int* d_next_frontier = nullptr;
    int* d_next_size = nullptr;
    int* d_indeg = nullptr;
    int* d_outdeg = nullptr;
    int* d_trim_count = nullptr;
    int* d_next_cid = nullptr;
    int* d_pivot = nullptr;
    unsigned long long* d_pivot_score = nullptr;
    const int* h_rev_V = nullptr;
    const int* h_rev_E = nullptr;
    bool reg_rev_V = false;
    bool reg_rev_E = false;
    bool rev_changed = !reverse_signature_matches(g_conn_ws, V, E, n, m);

    auto fail = [&](cudaError_t err, int code) {
        if (err != cudaSuccess) status = code;
    };
    auto fail_if_status = [&](int rc) {
        if (rc != EG_GPU_SUCC) status = rc;
    };

    build_reverse_csr_cached(g_conn_ws, V, E, n, m);
    fail_if_status(acquire_device_csr(V, E, nullptr, n, m, false, &graph_view));
    if (status != EG_GPU_SUCC) goto cleanup;

    fail_if_status(g_conn_ws.d_rev_V.ensure_bytes((size_t)(n + 1) * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_rev_E.ensure_bytes((size_t)g_conn_ws.cached_rev_E.size() * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_forward.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_backward.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_comp.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_valid.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_frontier.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_next_frontier.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_next_size.ensure_bytes(sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_indeg.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_outdeg.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_trim_count.ensure_bytes(sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_next_cid.ensure_bytes(sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_pivot.ensure_bytes(sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_conn_ws.d_pivot_score.ensure_bytes(sizeof(unsigned long long)));
    if (status != EG_GPU_SUCC) goto cleanup;

    d_V = graph_view.d_V;
    d_E = graph_view.d_E;
    d_rev_V = g_conn_ws.d_rev_V.as<int>();
    d_rev_E = g_conn_ws.d_rev_E.as<int>();
    d_forward = g_conn_ws.d_forward.as<int>();
    d_backward = g_conn_ws.d_backward.as<int>();
    d_comp = g_conn_ws.d_comp.as<int>();
    d_valid = g_conn_ws.d_valid.as<int>();
    d_frontier = g_conn_ws.d_frontier.as<int>();
    d_next_frontier = g_conn_ws.d_next_frontier.as<int>();
    d_next_size = g_conn_ws.d_next_size.as<int>();
    d_indeg = g_conn_ws.d_indeg.as<int>();
    d_outdeg = g_conn_ws.d_outdeg.as<int>();
    d_trim_count = g_conn_ws.d_trim_count.as<int>();
    d_next_cid = g_conn_ws.d_next_cid.as<int>();
    d_pivot = g_conn_ws.d_pivot.as<int>();
    d_pivot_score = g_conn_ws.d_pivot_score.as<unsigned long long>();

    if (rev_changed) {
        h_rev_V = prepare_h2d_source(g_conn_ws.cached_rev_V.data(), (size_t)(n + 1), g_conn_ws.h_rev_V_stage, &reg_rev_V);
        if (h_rev_V == nullptr) h_rev_V = g_conn_ws.cached_rev_V.data();
        fail(cudaMemcpy(d_rev_V, h_rev_V, (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        if (!g_conn_ws.cached_rev_E.empty()) {
            h_rev_E = prepare_h2d_source(g_conn_ws.cached_rev_E.data(), g_conn_ws.cached_rev_E.size(), g_conn_ws.h_rev_E_stage, &reg_rev_E);
            if (h_rev_E == nullptr) h_rev_E = g_conn_ws.cached_rev_E.data();
            fail(cudaMemcpy(d_rev_E, h_rev_E, g_conn_ws.cached_rev_E.size() * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
            if (status != EG_GPU_SUCC) goto cleanup;
        }
    }

    if (!g_conn_ws.runtime_ready) {
        fail(cudaEventCreate(&g_conn_ws.ev_begin), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaEventCreate(&g_conn_ws.ev_end), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        g_conn_ws.runtime_ready = true;
    }

    fail(cudaEventRecord(g_conn_ws.ev_begin), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    k_init_array_int<<<gv, bs>>>(d_comp, -1, n);
    k_mark_valid_nodes<<<gv, bs>>>(d_valid, 1, n);
    k_init_array_int<<<gv, bs>>>(d_forward, 0, n);
    k_init_array_int<<<gv, bs>>>(d_backward, 0, n);
    k_outdeg<<<gv, bs>>>(d_V, d_outdeg, n);
    k_outdeg<<<gv, bs>>>(d_rev_V, d_indeg, n);
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    {
        int zero = 0;
        fail(cudaMemcpy(d_next_cid, &zero, sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(d_trim_count, &zero, sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
    }
    k_trim_singletons_assign<<<gv, bs>>>(d_indeg, d_outdeg, d_valid, d_comp, n, d_next_cid, d_trim_count);
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    if (env_flag_enabled("EASYGRAPH_GPU_SCC_ACTIVE_TRIM", true)) {
        int trimmed = 0;
        fail(cudaMemcpy(&trimmed, d_trim_count, sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        if (trimmed > 0) {
            int max_trim_iters = env_int_value("EASYGRAPH_GPU_SCC_ACTIVE_TRIM_MAX_ITERS", 16);
            fail_if_status(run_active_singleton_trim(
                d_V, d_E, n, d_valid, d_comp, d_indeg, d_outdeg,
                d_next_cid, d_trim_count, gv, bs, max_trim_iters));
            if (status != EG_GPU_SUCC) goto cleanup;
        }
    }

    {
        int comp_id = 0;
        fail(cudaMemcpy(&comp_id, d_next_cid, sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        while (true) {
            int pivot = 0;
            if (env_flag_enabled("EASYGRAPH_GPU_SCC_DEGREE_PIVOT", true)) {
                unsigned long long zero_key = 0;
                unsigned long long best_key = 0;
                fail(cudaMemcpy(d_pivot_score, &zero_key, sizeof(unsigned long long), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
                if (status != EG_GPU_SUCC) goto cleanup;
                k_find_pivot_max_degree<<<gv, bs>>>(d_valid, d_indeg, d_outdeg, n, d_pivot_score);
                fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
                if (status != EG_GPU_SUCC) goto cleanup;
                fail(cudaMemcpy(&best_key, d_pivot_score, sizeof(unsigned long long), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
                if (status != EG_GPU_SUCC) goto cleanup;
                pivot = (best_key == 0)
                    ? n
                    : (int)(0xffffffffu - (unsigned int)(best_key & 0xffffffffull));
            } else {
                int inf = n;
                fail(cudaMemcpy(d_pivot, &inf, sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
                if (status != EG_GPU_SUCC) goto cleanup;
                k_find_pivot_min<<<gv, bs>>>(d_valid, n, d_pivot);
                fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
                if (status != EG_GPU_SUCC) goto cleanup;
                fail(cudaMemcpy(&pivot, d_pivot, sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
                if (status != EG_GPU_SUCC) goto cleanup;
            }
            if (pivot == n) break;

            int visit_mark = comp_id + 1;
            fail_if_status(run_scc_bfs(
                d_V, d_E, n, pivot, d_valid, d_forward,
                d_frontier, d_next_frontier, d_next_size, max_frontier_blocks, visit_mark));
            if (status != EG_GPU_SUCC) goto cleanup;

            fail_if_status(run_scc_bfs(
                d_rev_V, d_rev_E, n, pivot, d_valid, d_backward,
                d_frontier, d_next_frontier, d_next_size, max_frontier_blocks, visit_mark));
            if (status != EG_GPU_SUCC) goto cleanup;

            k_scc_intersect_assign<<<gv, bs>>>(
                d_forward, d_backward, d_valid, d_comp, n, comp_id, visit_mark);
            fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
            if (status != EG_GPU_SUCC) goto cleanup;
            ++comp_id;
            if (comp_id > n) {
                status = EG_GPU_DEVICE_ERR;
                goto cleanup;
            }
        }
    }

    fail(cudaEventRecord(g_conn_ws.ev_end), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail(cudaEventSynchronize(g_conn_ws.ev_end), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    if (kernel_seconds != nullptr) {
        float ms = 0.0f;
        fail(cudaEventElapsedTime(&ms, g_conn_ws.ev_begin, g_conn_ws.ev_end), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        *kernel_seconds = (double)ms / 1000.0;
    }

    fail(cudaMemcpy(labels.data(), d_comp, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

cleanup:
    release_h2d_source(h_rev_V, reg_rev_V);
    release_h2d_source(h_rev_E, reg_rev_E);
    if (status != EG_GPU_SUCC) {
        labels.clear();
        if (g_conn_ws.ev_begin != nullptr) {
            cudaEventDestroy(g_conn_ws.ev_begin);
            g_conn_ws.ev_begin = nullptr;
        }
        if (g_conn_ws.ev_end != nullptr) {
            cudaEventDestroy(g_conn_ws.ev_end);
            g_conn_ws.ev_end = nullptr;
        }
        g_conn_ws.runtime_ready = false;
    }
    return status;
}

} // namespace gpu_easygraph
