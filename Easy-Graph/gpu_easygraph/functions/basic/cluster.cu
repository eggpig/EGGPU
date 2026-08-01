#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "basic/cluster.cuh"
#include "buffer_cache.h"
#include "device_graph_cache.h"
#include "err.h"

namespace gpu_easygraph {

namespace {

using tri_t = unsigned long long;

struct ClusteringWorkspace {
    PersistentDeviceBuffer d_rpF;
    PersistentDeviceBuffer d_coF;
    PersistentDeviceBuffer d_srcF;
    PersistentDeviceBuffer d_deg;
    PersistentDeviceBuffer d_tri;
    PersistentDeviceBuffer d_cc;

    PersistentPinnedBuffer h_rpF;
    PersistentPinnedBuffer h_coF;
    PersistentPinnedBuffer h_srcF;
    PersistentPinnedBuffer h_deg;
    PersistentPinnedBuffer h_cc;

    std::vector<int> cached_deg;
    std::vector<int> cached_rpF;
    std::vector<int> cached_coF;
    std::vector<int> cached_srcF;
    int cached_mF = 0;

    const int* sig_V = nullptr;
    const int* sig_E = nullptr;
    int sig_n = -1;
    int sig_m = -1;
    int sig_v0 = 0;
    int sig_vn = 0;
    int sig_e0 = 0;
    int sig_em = 0;
    std::uint64_t sig_sample_hash = 0;
    std::uint64_t sig_graph_id = 0;
    int owner_device_id = -1;
    bool graph_cached = false;

    cudaEvent_t ev_begin = nullptr;
    cudaEvent_t ev_end = nullptr;
    bool runtime_ready = false;

    ~ClusteringWorkspace() {
        if (ev_begin != nullptr) cudaEventDestroy(ev_begin);
        if (ev_end != nullptr) cudaEventDestroy(ev_end);
    }
};

static thread_local ClusteringWorkspace g_cc_ws;

static bool graph_signature_matches(
    const ClusteringWorkspace& ws,
    const int* V,
    const int* E,
    int n,
    int m,
    std::uint64_t graph_id,
    int device_id
) {
    if (!ws.graph_cached) return false;
    if (ws.sig_graph_id != graph_id || ws.owner_device_id != device_id) return false;
    if (ws.sig_V != V || ws.sig_E != E) return false;
    if (ws.sig_n != n || ws.sig_m != m) return false;
    if (n <= 0) return true;
    if (ws.sig_v0 != V[0] || ws.sig_vn != V[n]) return false;
    if (m > 0 && (ws.sig_e0 != E[0] || ws.sig_em != E[m - 1])) return false;
    std::uint64_t h = 1469598103934665603ULL;
    auto mix = [&](int value) {
        h ^= (std::uint64_t)(std::uint32_t)value;
        h *= 1099511628211ULL;
    };
    const int v_step = std::max(1, n / 16);
    for (int i = 0; i <= n; i += v_step) mix(V[i]);
    if (m > 0) {
        const int e_step = std::max(1, m / 64);
        for (int i = 0; i < m; i += e_step) mix(E[i]);
        mix(E[m - 1]);
    }
    if (ws.sig_sample_hash != h) return false;
    return true;
}

static void update_graph_signature(
    ClusteringWorkspace& ws,
    const int* V,
    const int* E,
    int n,
    int m,
    std::uint64_t graph_id,
    int device_id
) {
    ws.sig_V = V;
    ws.sig_E = E;
    ws.sig_n = n;
    ws.sig_m = m;
    ws.sig_v0 = (n > 0) ? V[0] : 0;
    ws.sig_vn = (n > 0) ? V[n] : 0;
    ws.sig_e0 = (m > 0) ? E[0] : 0;
    ws.sig_em = (m > 0) ? E[m - 1] : 0;
    std::uint64_t h = 1469598103934665603ULL;
    auto mix = [&](int value) {
        h ^= (std::uint64_t)(std::uint32_t)value;
        h *= 1099511628211ULL;
    };
    const int v_step = std::max(1, n / 16);
    for (int i = 0; i <= n; i += v_step) mix(V[i]);
    if (m > 0) {
        const int e_step = std::max(1, m / 64);
        for (int i = 0; i < m; i += e_step) mix(E[i]);
        mix(E[m - 1]);
    }
    ws.sig_sample_hash = h;
    ws.sig_graph_id = graph_id;
    ws.owner_device_id = device_id;
    ws.graph_cached = true;
}

__device__ __forceinline__ void twop_intersect_emit_uvagg(
    const int* __restrict__ col,
    int as,
    int ae,
    int bs,
    int be,
    int u,
    int v,
    tri_t* __restrict__ tri
) {
    int i = as;
    int j = bs;
    int c_uv = 0;
    while (i < ae && j < be) {
        int x = col[i];
        int y = col[j];
        if (x == y) {
            atomicAdd(&tri[x], 1ULL);
            ++c_uv;
            ++i;
            ++j;
        } else if (x < y) {
            ++i;
        } else {
            ++j;
        }
    }
    if (c_uv > 0) {
        atomicAdd(&tri[u], (tri_t)c_uv);
        atomicAdd(&tri[v], (tri_t)c_uv);
    }
}

__global__ void triangles_forward_edge_kernel(
    const int* __restrict__ rpF,
    const int* __restrict__ coF,
    const int* __restrict__ srcF,
    tri_t* __restrict__ tri,
    int mF
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int e = tid; e < mF; e += stride) {
        int u = srcF[e];
        int v = coF[e];
        twop_intersect_emit_uvagg(coF, rpF[u], rpF[u + 1], rpF[v], rpF[v + 1], u, v, tri);
    }
}

__global__ void triangles_forward_vertex_kernel(
    const int* __restrict__ rpF,
    const int* __restrict__ coF,
    tri_t* __restrict__ tri,
    int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = tid >> 5;
    const int warp_count = (blockDim.x * gridDim.x) >> 5;
    if (warp_count <= 0) return;
    for (int u = warp_id; u < n; u += warp_count) {
        for (int edge = rpF[u] + lane; edge < rpF[u + 1]; edge += 32) {
            const int v = coF[edge];
            twop_intersect_emit_uvagg(
                coF,
                rpF[u],
                rpF[u + 1],
                rpF[v],
                rpF[v + 1],
                u,
                v,
                tri
            );
        }
    }
}

__global__ void cc_from_tri_kernel(
    const int* __restrict__ deg,
    const tri_t* __restrict__ tri,
    float* __restrict__ cc,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int u = tid; u < n; u += stride) {
        int d = deg[u];
        if (d < 2) {
            cc[u] = 0.0f;
            continue;
        }
        float denom = 0.5f * (float)d * (float)(d - 1);
        cc[u] = (float)tri[u] / denom;
    }
}

__device__ __forceinline__ bool row_contains_sorted(
    const int* __restrict__ rp,
    const int* __restrict__ co,
    int row,
    int target
) {
    int lo = rp[row];
    int hi = rp[row + 1] - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        int val = co[mid];
        if (val == target) return true;
        if (val < target) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return false;
}

__global__ void clustering_vertex_pairs_kernel(
    const int* __restrict__ rp,
    const int* __restrict__ co,
    const int* __restrict__ deg,
    float* __restrict__ cc,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int u = tid; u < n; u += stride) {
        int d = deg[u];
        if (d < 2) {
            cc[u] = 0.0f;
            continue;
        }
        unsigned long long triangles = 0ULL;
        int begin = rp[u];
        int end = rp[u + 1];
        for (int i = begin; i < end; ++i) {
            int v = co[i];
            for (int j = i + 1; j < end; ++j) {
                int w = co[j];
                if (row_contains_sorted(rp, co, v, w)) {
                    ++triangles;
                }
            }
        }
        float denom = 0.5f * (float)d * (float)(d - 1);
        cc[u] = (float)triangles / denom;
    }
}

static inline int grid_for(int n, int block_size) {
    if (n <= 0) return 1;
    int g = (n + block_size - 1) / block_size;
    return std::max(1, std::min(g, 65535));
}

static inline bool valid_pos(int x, int bound) {
    return (0 <= x) && (x < bound);
}

} // namespace

int cuda_clustering(
    const int* V,
    const int* E,
    int len_V,
    int len_E,
    std::vector<double>& CC,
    double* kernel_seconds
) {
    CC.clear();
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    if (len_V <= 0) return EG_GPU_SUCC;

    const int n = len_V;
    const int m = len_E;
    CC.assign(n, 0.0);
    if (m <= 0) return EG_GPU_SUCC;
    int device_id = -1;
    if (cudaGetDevice(&device_id) != cudaSuccess || device_id < 0) {
        return EG_GPU_DEVICE_ERR;
    }
    if (g_cc_ws.owner_device_id >= 0 && g_cc_ws.owner_device_id != device_id) {
        return EG_GPU_DEVICE_ERR;
    }
    g_cc_ws.owner_device_id = device_id;
    const std::uint64_t graph_id = active_device_graph_id();
    const bool graph_changed = !graph_signature_matches(
        g_cc_ws, V, E, n, m, graph_id, device_id);
    if (graph_changed) {
        g_cc_ws.cached_deg.assign(n, 0);
        for (int u = 0; u < n; ++u) {
            int begin = V[u];
            int end = V[u + 1];
            if (begin < 0) begin = 0;
            if (end > m) end = m;
            if (begin > end) begin = end;
            g_cc_ws.cached_deg[u] = end - begin;
        }

        g_cc_ws.cached_rpF.assign(n + 1, 0);
        for (int u = 0; u < n; ++u) {
            int begin = V[u];
            int end = V[u + 1];
            if (begin < 0) begin = 0;
            if (end > m) end = m;
            if (begin > end) begin = end;
                int du = g_cc_ws.cached_deg[u];
                int cnt = 0;
                for (int p = begin; p < end; ++p) {
                    int v = E[p];
                    if (!valid_pos(v, n) || v == u) continue;
                    int dv = g_cc_ws.cached_deg[v];
                    if (du > dv || (du == dv && u > v)) continue;
                    ++cnt;
                }
                g_cc_ws.cached_rpF[u + 1] = cnt;
        }
        for (int i = 1; i <= n; ++i) {
            g_cc_ws.cached_rpF[i] += g_cc_ws.cached_rpF[i - 1];
        }

        g_cc_ws.cached_mF = g_cc_ws.cached_rpF[n];
        g_cc_ws.cached_coF.assign(g_cc_ws.cached_mF, 0);
        g_cc_ws.cached_srcF.assign(g_cc_ws.cached_mF, 0);
        if (g_cc_ws.cached_mF > 0) {
            std::vector<int> cur = g_cc_ws.cached_rpF;
            for (int u = 0; u < n; ++u) {
                int begin = V[u];
                int end = V[u + 1];
                if (begin < 0) begin = 0;
                if (end > m) end = m;
                if (begin > end) begin = end;
                int du = g_cc_ws.cached_deg[u];
                for (int p = begin; p < end; ++p) {
                    int v = E[p];
                    if (!valid_pos(v, n) || v == u) continue;
                    int dv = g_cc_ws.cached_deg[v];
                    if (du > dv || (du == dv && u > v)) continue;
                    int idx = cur[u]++;
                    g_cc_ws.cached_coF[idx] = v;
                    g_cc_ws.cached_srcF[idx] = u;
                }
            }
            for (int u = 0; u < n; ++u) {
                int begin = g_cc_ws.cached_rpF[u];
                int end = g_cc_ws.cached_rpF[u + 1];
                if (begin < end) {
                    std::sort(g_cc_ws.cached_coF.begin() + begin, g_cc_ws.cached_coF.begin() + end);
                }
            }
        }
        update_graph_signature(g_cc_ws, V, E, n, m, graph_id, device_id);
    }

    const int mF = g_cc_ws.cached_mF;
    if (mF <= 0) return EG_GPU_SUCC;

    int* d_rpF = nullptr;
    int* d_coF = nullptr;
    int* d_srcF = nullptr;
    int* d_deg = nullptr;
    tri_t* d_tri = nullptr;
    float* d_cc = nullptr;
    float* h_cc = nullptr;
    const int* h_rpF = nullptr;
    const int* h_coF = nullptr;
    const int* h_srcF = nullptr;
    const int* h_deg = nullptr;
    bool reg_rpF = false;
    bool reg_coF = false;
    bool reg_srcF = false;
    bool reg_deg = false;
    int status = EG_GPU_SUCC;
    const int block = 256;
    int grid_edges = grid_for(mF, block);
    int grid_nodes = grid_for(n, block);

    auto fail = [&](cudaError_t err, int code) {
        if (err != cudaSuccess) status = code;
    };
    auto fail_if_status = [&](int rc) {
        if (rc != EG_GPU_SUCC) status = rc;
    };

    fail_if_status(g_cc_ws.d_rpF.ensure_bytes((size_t)(n + 1) * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_cc_ws.d_coF.ensure_bytes((size_t)mF * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_cc_ws.d_srcF.ensure_bytes((size_t)mF * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_cc_ws.d_deg.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_cc_ws.d_tri.ensure_bytes((size_t)n * sizeof(tri_t)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_cc_ws.d_cc.ensure_bytes((size_t)n * sizeof(float)));
    if (status != EG_GPU_SUCC) goto cleanup;

    d_rpF = g_cc_ws.d_rpF.as<int>();
    d_coF = g_cc_ws.d_coF.as<int>();
    d_srcF = g_cc_ws.d_srcF.as<int>();
    d_deg = g_cc_ws.d_deg.as<int>();
    d_tri = g_cc_ws.d_tri.as<tri_t>();
    d_cc = g_cc_ws.d_cc.as<float>();

    if (graph_changed) {
        h_rpF = prepare_h2d_source(g_cc_ws.cached_rpF.data(), (size_t)(n + 1), g_cc_ws.h_rpF, &reg_rpF);
        h_coF = prepare_h2d_source(g_cc_ws.cached_coF.data(), (size_t)mF, g_cc_ws.h_coF, &reg_coF);
        h_srcF = prepare_h2d_source(g_cc_ws.cached_srcF.data(), (size_t)mF, g_cc_ws.h_srcF, &reg_srcF);
        h_deg = prepare_h2d_source(g_cc_ws.cached_deg.data(), (size_t)n, g_cc_ws.h_deg, &reg_deg);
        if (h_rpF == nullptr) h_rpF = g_cc_ws.cached_rpF.data();
        if (h_coF == nullptr) h_coF = g_cc_ws.cached_coF.data();
        if (h_srcF == nullptr) h_srcF = g_cc_ws.cached_srcF.data();
        if (h_deg == nullptr) h_deg = g_cc_ws.cached_deg.data();

        fail(cudaMemcpy(d_rpF, h_rpF, (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(d_coF, h_coF, (size_t)mF * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(d_srcF, h_srcF, (size_t)mF * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(d_deg, h_deg, (size_t)n * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
    }
    fail(cudaMemset(d_tri, 0, (size_t)n * sizeof(tri_t)), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    if (!g_cc_ws.runtime_ready) {
        fail(cudaEventCreate(&g_cc_ws.ev_begin), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaEventCreate(&g_cc_ws.ev_end), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        g_cc_ws.runtime_ready = true;
    }

    fail(cudaEventRecord(g_cc_ws.ev_begin), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    triangles_forward_edge_kernel<<<grid_edges, block>>>(d_rpF, d_coF, d_srcF, d_tri, mF);
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    cc_from_tri_kernel<<<grid_nodes, block>>>(d_deg, d_tri, d_cc, n);
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail(cudaEventRecord(g_cc_ws.ev_end), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail(cudaEventSynchronize(g_cc_ws.ev_end), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    if (kernel_seconds != nullptr) {
        float ms = 0.0f;
        fail(cudaEventElapsedTime(&ms, g_cc_ws.ev_begin, g_cc_ws.ev_end), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        *kernel_seconds = (double)ms / 1000.0;
    }

    fail_if_status(g_cc_ws.h_cc.ensure_bytes((size_t)n * sizeof(float)));
    if (status != EG_GPU_SUCC) goto cleanup;
    h_cc = g_cc_ws.h_cc.as<float>();
    fail(cudaMemcpy(h_cc, d_cc, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    for (int i = 0; i < n; ++i) {
        CC[i] = (double)h_cc[i];
    }

cleanup:
    release_h2d_source(h_rpF, reg_rpF);
    release_h2d_source(h_coF, reg_coF);
    release_h2d_source(h_srcF, reg_srcF);
    release_h2d_source(h_deg, reg_deg);
    if (status != EG_GPU_SUCC) {
        // Runtime may be left in a bad state after CUDA runtime failures.
        if (g_cc_ws.ev_begin != nullptr) {
            cudaEventDestroy(g_cc_ws.ev_begin);
            g_cc_ws.ev_begin = nullptr;
        }
        if (g_cc_ws.ev_end != nullptr) {
            cudaEventDestroy(g_cc_ws.ev_end);
            g_cc_ws.ev_end = nullptr;
        }
        g_cc_ws.runtime_ready = false;
    }
    return status;
}

int cuda_clustering_forward(
    const int* forward_V,
    const int* forward_E,
    const int* degree,
    int len_V,
    int len_E,
    std::vector<double>& CC,
    double* kernel_seconds
) {
    CC.clear();
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    if (len_V <= 0) return EG_GPU_SUCC;
    if (forward_V == nullptr || degree == nullptr ||
        (len_E > 0 && forward_E == nullptr)) {
        return EG_GPU_DEVICE_ERR;
    }

    const int n = len_V;
    const int mF = len_E;
    CC.assign(n, 0.0);
    if (mF <= 0) return EG_GPU_SUCC;
    int device_id = -1;
    if (cudaGetDevice(&device_id) != cudaSuccess || device_id < 0) {
        return EG_GPU_DEVICE_ERR;
    }
    if (g_cc_ws.owner_device_id >= 0 && g_cc_ws.owner_device_id != device_id) {
        return EG_GPU_DEVICE_ERR;
    }
    g_cc_ws.owner_device_id = device_id;
    const std::uint64_t graph_id = active_device_graph_id();
    const bool graph_changed =
        !graph_signature_matches(
            g_cc_ws, forward_V, forward_E, n, mF, graph_id, device_id);
    if (graph_changed) {
        g_cc_ws.cached_deg.assign(degree, degree + n);
        g_cc_ws.cached_rpF.assign(forward_V, forward_V + n + 1);
        g_cc_ws.cached_coF.assign(forward_E, forward_E + mF);
        g_cc_ws.cached_srcF.clear();
        g_cc_ws.cached_mF = mF;
        update_graph_signature(
            g_cc_ws, forward_V, forward_E, n, mF, graph_id, device_id);
    }

    int status = EG_GPU_SUCC;
    int* d_rpF = nullptr;
    int* d_coF = nullptr;
    int* d_deg = nullptr;
    tri_t* d_tri = nullptr;
    float* d_cc = nullptr;
    float* h_cc = nullptr;
    const int* h_rpF = nullptr;
    const int* h_coF = nullptr;
    const int* h_deg = nullptr;
    bool reg_rpF = false;
    bool reg_coF = false;
    bool reg_deg = false;
    const int block = 256;
    const int grid_nodes = grid_for(n, block);

    auto fail = [&](cudaError_t err, int code) {
        if (err != cudaSuccess) status = code;
    };
    auto fail_if_status = [&](int rc) {
        if (rc != EG_GPU_SUCC) status = rc;
    };

    fail_if_status(g_cc_ws.d_rpF.ensure_bytes((size_t)(n + 1) * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_cc_ws.d_coF.ensure_bytes((size_t)mF * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_cc_ws.d_deg.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_cc_ws.d_tri.ensure_bytes((size_t)n * sizeof(tri_t)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_cc_ws.d_cc.ensure_bytes((size_t)n * sizeof(float)));
    if (status != EG_GPU_SUCC) goto cleanup;

    d_rpF = g_cc_ws.d_rpF.as<int>();
    d_coF = g_cc_ws.d_coF.as<int>();
    d_deg = g_cc_ws.d_deg.as<int>();
    d_tri = g_cc_ws.d_tri.as<tri_t>();
    d_cc = g_cc_ws.d_cc.as<float>();

    if (graph_changed) {
        h_rpF = prepare_h2d_source(
            g_cc_ws.cached_rpF.data(), (size_t)(n + 1), g_cc_ws.h_rpF, &reg_rpF);
        h_coF = prepare_h2d_source(
            g_cc_ws.cached_coF.data(), (size_t)mF, g_cc_ws.h_coF, &reg_coF);
        h_deg = prepare_h2d_source(
            g_cc_ws.cached_deg.data(), (size_t)n, g_cc_ws.h_deg, &reg_deg);
        if (h_rpF == nullptr) h_rpF = g_cc_ws.cached_rpF.data();
        if (h_coF == nullptr) h_coF = g_cc_ws.cached_coF.data();
        if (h_deg == nullptr) h_deg = g_cc_ws.cached_deg.data();
        fail(cudaMemcpy(
            d_rpF, h_rpF, (size_t)(n + 1) * sizeof(int),
            cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(
            d_coF, h_coF, (size_t)mF * sizeof(int),
            cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(
            d_deg, h_deg, (size_t)n * sizeof(int),
            cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
    }
    fail(cudaMemset(d_tri, 0, (size_t)n * sizeof(tri_t)), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    if (!g_cc_ws.runtime_ready) {
        fail(cudaEventCreate(&g_cc_ws.ev_begin), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaEventCreate(&g_cc_ws.ev_end), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        g_cc_ws.runtime_ready = true;
    }

    fail(cudaEventRecord(g_cc_ws.ev_begin), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    triangles_forward_vertex_kernel<<<grid_nodes, block>>>(
        d_rpF, d_coF, d_tri, n);
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    cc_from_tri_kernel<<<grid_nodes, block>>>(d_deg, d_tri, d_cc, n);
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail(cudaEventRecord(g_cc_ws.ev_end), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail(cudaEventSynchronize(g_cc_ws.ev_end), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    if (kernel_seconds != nullptr) {
        float ms = 0.0f;
        fail(cudaEventElapsedTime(
            &ms, g_cc_ws.ev_begin, g_cc_ws.ev_end), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        *kernel_seconds = (double)ms / 1000.0;
    }

    fail_if_status(g_cc_ws.h_cc.ensure_bytes((size_t)n * sizeof(float)));
    if (status != EG_GPU_SUCC) goto cleanup;
    h_cc = g_cc_ws.h_cc.as<float>();
    fail(cudaMemcpy(
        h_cc, d_cc, (size_t)n * sizeof(float),
        cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    for (int i = 0; i < n; ++i) CC[i] = (double)h_cc[i];

cleanup:
    release_h2d_source(h_rpF, reg_rpF);
    release_h2d_source(h_coF, reg_coF);
    release_h2d_source(h_deg, reg_deg);
    if (status != EG_GPU_SUCC) {
        if (g_cc_ws.ev_begin != nullptr) {
            cudaEventDestroy(g_cc_ws.ev_begin);
            g_cc_ws.ev_begin = nullptr;
        }
        if (g_cc_ws.ev_end != nullptr) {
            cudaEventDestroy(g_cc_ws.ev_end);
            g_cc_ws.ev_end = nullptr;
        }
        g_cc_ws.runtime_ready = false;
    }
    return status;
}

} // namespace gpu_easygraph
