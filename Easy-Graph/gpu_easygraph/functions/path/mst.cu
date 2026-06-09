#include <climits>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "path/mst.cuh"
#include "buffer_cache.h"
#include "err.h"

namespace gpu_easygraph {

namespace {

using ull = unsigned long long;
constexpr int kThreadsPerBlock = 512;

struct MstWorkspace {
    PersistentDeviceBuffer d_in_mst;
    PersistentDeviceBuffer d_parent;
    PersistentDeviceBuffer d_minv;
    PersistentDeviceBuffer d_wl1;
    PersistentDeviceBuffer d_wl2;
    PersistentDeviceBuffer d_wlsize;
    PersistentDeviceBuffer d_nindex;
    PersistentDeviceBuffer d_nlist;
    PersistentDeviceBuffer d_eweight;
    PersistentDeviceBuffer d_eweight_fp;
    PersistentDeviceBuffer d_out_src;
    PersistentDeviceBuffer d_out_dst;
    PersistentDeviceBuffer d_out_weight;
    PersistentDeviceBuffer d_out_count;
    PersistentDeviceBuffer d_out_overflow;

    PersistentPinnedBuffer h_nindex;
    PersistentPinnedBuffer h_nlist;
    PersistentPinnedBuffer h_eweight;
    PersistentPinnedBuffer h_eweight_fp;

    std::vector<int> cached_nindex;
    std::vector<int> cached_nlist;
    std::vector<int> cached_eweight;
    std::vector<double> cached_eweight_fp;
    int cached_threshold = INT_MAX;
    int cached_n = 0;
    int cached_csr_edges = 0;

    const int* sig_V = nullptr;
    const int* sig_E = nullptr;
    const double* sig_W = nullptr;
    int sig_n = -1;
    int sig_m = -1;
    int sig_v0 = 0;
    int sig_vn = 0;
    int sig_e0 = 0;
    int sig_em = 0;
    std::uint64_t sig_w0 = 0;
    std::uint64_t sig_wm = 0;
    bool graph_cached = false;

    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;
    bool runtime_ready = false;

    ~MstWorkspace() {
        if (ev0 != nullptr) cudaEventDestroy(ev0);
        if (ev1 != nullptr) cudaEventDestroy(ev1);
    }
};

static thread_local MstWorkspace g_mst_ws;

static inline std::uint64_t bitcast_u64(double x) {
    std::uint64_t u = 0;
    std::memcpy(&u, &x, sizeof(double));
    return u;
}

static bool graph_signature_matches(
    const MstWorkspace& ws,
    const int* V,
    const int* E,
    const double* W,
    int n,
    int m
) {
    if (!ws.graph_cached) return false;
    if (ws.sig_V != V || ws.sig_E != E || ws.sig_W != W) return false;
    if (ws.sig_n != n || ws.sig_m != m) return false;
    if (n <= 0) return true;
    if (ws.sig_v0 != V[0] || ws.sig_vn != V[n]) return false;
    if (m > 0) {
        if (ws.sig_e0 != E[0] || ws.sig_em != E[m - 1]) return false;
        if (W != nullptr) {
            if (ws.sig_w0 != bitcast_u64(W[0]) || ws.sig_wm != bitcast_u64(W[m - 1])) return false;
        } else if (ws.sig_w0 != 0 || ws.sig_wm != 0) {
            return false;
        }
    }
    return true;
}

static void update_graph_signature(
    MstWorkspace& ws,
    const int* V,
    const int* E,
    const double* W,
    int n,
    int m
) {
    ws.sig_V = V;
    ws.sig_E = E;
    ws.sig_W = W;
    ws.sig_n = n;
    ws.sig_m = m;
    ws.sig_v0 = (n > 0) ? V[0] : 0;
    ws.sig_vn = (n > 0) ? V[n] : 0;
    ws.sig_e0 = (m > 0) ? E[0] : 0;
    ws.sig_em = (m > 0) ? E[m - 1] : 0;
    ws.sig_w0 = (W != nullptr && m > 0) ? bitcast_u64(W[0]) : 0;
    ws.sig_wm = (W != nullptr && m > 0) ? bitcast_u64(W[m - 1]) : 0;
    ws.graph_cached = true;
}

struct UndirectedEdge {
    int u;
    int v;
    int w_int;
    double w_fp;
};

static inline unsigned hash32(unsigned v) {
    v = ((v >> 16) ^ v) * 0x45d9f3bU;
    v = ((v >> 16) ^ v) * 0x45d9f3bU;
    return (v >> 16) ^ v;
}

static inline int encode_weight_int(double w) {
    if (!std::isfinite(w)) return 0;
    if (w > (double)INT_MAX) return INT_MAX;
    if (w < (double)INT_MIN) return INT_MIN;
    return (int)std::llround(w);
}

static __device__ int find_root(int curr, const int* __restrict__ parent) {
    int next;
    while (curr != (next = parent[curr])) curr = next;
    return curr;
}

static __global__ void init_pm(int nodes, int* __restrict__ parent, ull* __restrict__ minv) {
    int v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v < nodes) {
        parent[v] = v;
        minv[v] = ULLONG_MAX;
    }
}

template <bool first_pass>
static __global__ void init_worklist(
    int4* __restrict__ wl2,
    int* __restrict__ wl2size,
    int nodes,
    const int* __restrict__ nindex,
    const int* __restrict__ nlist,
    const int* __restrict__ eweight,
    const int* __restrict__ parent,
    int threshold
) {
    const int v = threadIdx.x + blockIdx.x * blockDim.x;
    int beg = 0;
    int end = 0;
    int arep = 0;
    int deg = -1;
    if (v < nodes) {
        beg = nindex[v];
        end = nindex[v + 1];
        deg = end - beg;
        arep = first_pass ? v : find_root(v, parent);
        if (deg < 4) {
            for (int j = beg; j < end; ++j) {
                int n = nlist[j];
                if (n > v) {
                    int wei = eweight[j];
                    bool accept = first_pass ? (wei <= threshold) : (wei > threshold);
                    if (!accept) continue;
                    int brep = first_pass ? n : find_root(n, parent);
                    if (!first_pass && arep == brep) continue;
                    int k = atomicAdd(wl2size, 1);
                    wl2[k] = make_int4(arep, brep, wei, j);
                }
            }
        }
    }

    constexpr int ws = 32;
    int lane = threadIdx.x & (ws - 1);
    unsigned mask = __ballot_sync(0xffffffff, deg >= 4);
    while (mask) {
        int who = __ffs(mask) - 1;
        mask &= mask - 1;
        int wi = __shfl_sync(0xffffffff, v, who);
        int wbeg = __shfl_sync(0xffffffff, beg, who);
        int wend = __shfl_sync(0xffffffff, end, who);
        int warep = first_pass ? wi : __shfl_sync(0xffffffff, arep, who);
        for (int j = wbeg + lane; j < wend; j += ws) {
            int n = nlist[j];
            if (n > wi) {
                int wei = eweight[j];
                bool accept = first_pass ? (wei <= threshold) : (wei > threshold);
                if (!accept) continue;
                int brep = first_pass ? n : find_root(n, parent);
                if (!first_pass && warep == brep) continue;
                int k = atomicAdd(wl2size, 1);
                wl2[k] = make_int4(warep, brep, wei, j);
            }
        }
    }
}

static __global__ void kernel_filter_components(
    const int4* __restrict__ wl1,
    int wl1size,
    int4* __restrict__ wl2,
    int* __restrict__ wl2size,
    const int* __restrict__ parent,
    volatile ull* __restrict__ minv
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= wl1size) return;
    int4 el = wl1[idx];
    int a = find_root(el.x, parent);
    int b = find_root(el.y, parent);
    if (a != b) {
        el.x = a;
        el.y = b;
        int pos = atomicAdd(wl2size, 1);
        wl2[pos] = el;
        ull val = (((ull)(unsigned int)el.z) << 32) | (unsigned int)el.w;
        if (minv[a] > val) atomicMin((ull*)&minv[a], val);
        if (minv[b] > val) atomicMin((ull*)&minv[b], val);
    }
}

static __global__ void kernel_join_and_mark(
    const int4* __restrict__ wl,
    int wlsize,
    int* __restrict__ parent,
    ull* __restrict__ minv,
    bool* __restrict__ in_mst
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= wlsize) return;
    int4 el = wl[idx];
    ull val = (((ull)(unsigned int)el.z) << 32) | (unsigned int)el.w;
    if (val == minv[el.x] || val == minv[el.y]) {
        int a = el.x;
        int b = el.y;
        int m;
        do {
            m = max(a, b);
            a = min(a, b);
        } while ((b = atomicCAS(&parent[m], m, a)) != m);
        in_mst[el.w] = true;
    }
}

static __global__ void kernel_reset_minv(
    const int4* __restrict__ wl,
    int wlsize,
    volatile ull* __restrict__ minv
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= wlsize) return;
    int4 el = wl[idx];
    minv[el.x] = ULLONG_MAX;
    minv[el.y] = ULLONG_MAX;
}

static __global__ void kernel_collect_mst_edges(
    int nodes,
    const int* __restrict__ nindex,
    const int* __restrict__ nlist,
    const double* __restrict__ eweight_fp,
    const bool* __restrict__ in_mst,
    int* __restrict__ out_src,
    int* __restrict__ out_dst,
    double* __restrict__ out_weight,
    int* __restrict__ out_count,
    int* __restrict__ overflow,
    int max_out
) {
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= nodes) return;
    int begin = nindex[u];
    int end = nindex[u + 1];
    for (int p = begin; p < end; ++p) {
        int v = nlist[p];
        if (v > u && in_mst[p]) {
            int pos = atomicAdd(out_count, 1);
            if (pos < max_out) {
                out_src[pos] = u;
                out_dst[pos] = v;
                out_weight[pos] = eweight_fp[p];
            } else {
                atomicExch(overflow, 1);
            }
        }
    }
}

static inline int blocks_for(int n) {
    return (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
}

} // namespace


int cuda_mst(
    const int* V,
    const int* E,
    const double* W,
    int len_V,
    int len_E,
    std::vector<int>& mst_src,
    std::vector<int>& mst_dst,
    std::vector<double>& mst_weight,
    double* kernel_seconds
) {
    mst_src.clear();
    mst_dst.clear();
    mst_weight.clear();
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;

    const int n = len_V;
    const int m = len_E;
    if (n <= 0 || m <= 0) return EG_GPU_SUCC;
    const bool graph_changed = !graph_signature_matches(g_mst_ws, V, E, W, n, m);
    if (graph_changed) {
        std::vector<UndirectedEdge> edges;
        edges.reserve((size_t)m / 2 + 1);
        for (int u = 0; u < n; ++u) {
            int begin = V[u];
            int end = V[u + 1];
            if (begin < 0) begin = 0;
            if (end > m) end = m;
            if (begin > end) begin = end;
            for (int p = begin; p < end; ++p) {
                int v = E[p];
                if (v < 0 || v >= n || v == u) continue;
                int a = u;
                int b = v;
                if (a > b) std::swap(a, b);
                double wf = (W != nullptr) ? W[p] : 1.0;
                if (!std::isfinite(wf)) wf = 1.0;
                edges.push_back({a, b, encode_weight_int(wf), wf});
            }
        }
        if (edges.empty()) return EG_GPU_SUCC;

        std::sort(edges.begin(), edges.end(), [](const UndirectedEdge& x, const UndirectedEdge& y) {
            if (x.u != y.u) return x.u < y.u;
            if (x.v != y.v) return x.v < y.v;
            if (x.w_int != y.w_int) return x.w_int < y.w_int;
            return x.w_fp < y.w_fp;
        });

        std::vector<UndirectedEdge> unique_edges;
        unique_edges.reserve(edges.size());
        for (const auto& e : edges) {
            if (!unique_edges.empty() && unique_edges.back().u == e.u && unique_edges.back().v == e.v) {
                if (e.w_int < unique_edges.back().w_int ||
                    (e.w_int == unique_edges.back().w_int && e.w_fp < unique_edges.back().w_fp)) {
                    unique_edges.back().w_int = e.w_int;
                    unique_edges.back().w_fp = e.w_fp;
                }
            } else {
                unique_edges.push_back(e);
            }
        }
        edges.swap(unique_edges);

        std::vector<int> deg(n, 0);
        for (const auto& e : edges) {
            deg[e.u] += 1;
            deg[e.v] += 1;
        }
        g_mst_ws.cached_nindex.assign(n + 1, 0);
        for (int i = 1; i <= n; ++i) g_mst_ws.cached_nindex[i] = g_mst_ws.cached_nindex[i - 1] + deg[i - 1];
        g_mst_ws.cached_csr_edges = g_mst_ws.cached_nindex[n];
        if (g_mst_ws.cached_csr_edges <= 0) return EG_GPU_SUCC;

        g_mst_ws.cached_nlist.assign(g_mst_ws.cached_csr_edges, 0);
        g_mst_ws.cached_eweight.assign(g_mst_ws.cached_csr_edges, 0);
        g_mst_ws.cached_eweight_fp.assign(g_mst_ws.cached_csr_edges, 0.0);
        std::vector<int> cur = g_mst_ws.cached_nindex;
        for (const auto& e : edges) {
            int pu = cur[e.u]++;
            int pv = cur[e.v]++;
            g_mst_ws.cached_nlist[pu] = e.v;
            g_mst_ws.cached_nlist[pv] = e.u;
            g_mst_ws.cached_eweight[pu] = e.w_int;
            g_mst_ws.cached_eweight[pv] = e.w_int;
            g_mst_ws.cached_eweight_fp[pu] = e.w_fp;
            g_mst_ws.cached_eweight_fp[pv] = e.w_fp;
        }

        g_mst_ws.cached_threshold = INT_MAX;
        const int avg_deg_local = g_mst_ws.cached_csr_edges / std::max(1, n);
        if (avg_deg_local >= 4) {
            int sorted[20];
            int samples = std::min(g_mst_ws.cached_csr_edges, 20);
            for (int i = 0; i < samples; ++i) {
                sorted[i] = g_mst_ws.cached_eweight[hash32((unsigned)i) % g_mst_ws.cached_csr_edges];
            }
            std::sort(sorted, sorted + samples);
            int tindex = (int)(3.0 * n * samples / (double)std::max(1, g_mst_ws.cached_csr_edges));
            if (tindex >= samples) tindex = samples - 1;
            if (tindex < 0) tindex = 0;
            g_mst_ws.cached_threshold = sorted[tindex];
        }
        g_mst_ws.cached_n = n;
        update_graph_signature(g_mst_ws, V, E, W, n, m);
    }

    const int csr_edges = g_mst_ws.cached_csr_edges;
    if (csr_edges <= 0) return EG_GPU_SUCC;
    const int threshold = g_mst_ws.cached_threshold;
    const int avg_deg = csr_edges / std::max(1, n);

    bool* d_inMST = nullptr;
    int* d_parent = nullptr;
    ull* d_minv = nullptr;
    int4* d_wl1 = nullptr;
    int4* d_wl2 = nullptr;
    int* d_wlsize = nullptr;
    int* d_nindex = nullptr;
    int* d_nlist = nullptr;
    int* d_eweight = nullptr;
    double* d_eweight_fp = nullptr;
    int* d_out_src = nullptr;
    int* d_out_dst = nullptr;
    double* d_out_weight = nullptr;
    int* d_out_count = nullptr;
    int* d_out_overflow = nullptr;
    const int* h_nindex = nullptr;
    const int* h_nlist = nullptr;
    const int* h_eweight = nullptr;
    const double* h_eweight_fp = nullptr;
    bool reg_nindex = false;
    bool reg_nlist = false;
    bool reg_eweight = false;
    bool reg_eweight_fp = false;

    int status = EG_GPU_SUCC;
    int blocks_nodes = 1;
    int wlsize = 0;
    int h_out_count = 0;
    int h_out_overflow = 0;
    auto fail = [&](cudaError_t err, int code) {
        if (err != cudaSuccess) status = code;
    };
    auto fail_if_status = [&](int rc) {
        if (rc != EG_GPU_SUCC) status = rc;
    };

    fail_if_status(g_mst_ws.d_in_mst.ensure_bytes((size_t)csr_edges * sizeof(bool)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_parent.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_minv.ensure_bytes((size_t)n * sizeof(ull)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_wl1.ensure_bytes((size_t)csr_edges * sizeof(int4)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_wl2.ensure_bytes((size_t)csr_edges * sizeof(int4)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_wlsize.ensure_bytes(sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_nindex.ensure_bytes((size_t)(n + 1) * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_nlist.ensure_bytes((size_t)csr_edges * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_eweight.ensure_bytes((size_t)csr_edges * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_eweight_fp.ensure_bytes((size_t)csr_edges * sizeof(double)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_out_src.ensure_bytes((size_t)std::max(1, n) * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_out_dst.ensure_bytes((size_t)std::max(1, n) * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_out_weight.ensure_bytes((size_t)std::max(1, n) * sizeof(double)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_out_count.ensure_bytes(sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_mst_ws.d_out_overflow.ensure_bytes(sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;

    d_inMST = g_mst_ws.d_in_mst.as<bool>();
    d_parent = g_mst_ws.d_parent.as<int>();
    d_minv = g_mst_ws.d_minv.as<ull>();
    d_wl1 = g_mst_ws.d_wl1.as<int4>();
    d_wl2 = g_mst_ws.d_wl2.as<int4>();
    d_wlsize = g_mst_ws.d_wlsize.as<int>();
    d_nindex = g_mst_ws.d_nindex.as<int>();
    d_nlist = g_mst_ws.d_nlist.as<int>();
    d_eweight = g_mst_ws.d_eweight.as<int>();
    d_eweight_fp = g_mst_ws.d_eweight_fp.as<double>();
    d_out_src = g_mst_ws.d_out_src.as<int>();
    d_out_dst = g_mst_ws.d_out_dst.as<int>();
    d_out_weight = g_mst_ws.d_out_weight.as<double>();
    d_out_count = g_mst_ws.d_out_count.as<int>();
    d_out_overflow = g_mst_ws.d_out_overflow.as<int>();

    if (graph_changed) {
        h_nindex = prepare_h2d_source(g_mst_ws.cached_nindex.data(), (size_t)(n + 1), g_mst_ws.h_nindex, &reg_nindex);
        h_nlist = prepare_h2d_source(g_mst_ws.cached_nlist.data(), (size_t)csr_edges, g_mst_ws.h_nlist, &reg_nlist);
        h_eweight = prepare_h2d_source(g_mst_ws.cached_eweight.data(), (size_t)csr_edges, g_mst_ws.h_eweight, &reg_eweight);
        h_eweight_fp = prepare_h2d_source(g_mst_ws.cached_eweight_fp.data(), (size_t)csr_edges, g_mst_ws.h_eweight_fp, &reg_eweight_fp);
        if (h_nindex == nullptr) h_nindex = g_mst_ws.cached_nindex.data();
        if (h_nlist == nullptr) h_nlist = g_mst_ws.cached_nlist.data();
        if (h_eweight == nullptr) h_eweight = g_mst_ws.cached_eweight.data();
        if (h_eweight_fp == nullptr) h_eweight_fp = g_mst_ws.cached_eweight_fp.data();

        fail(cudaMemcpy(d_nindex, h_nindex, (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(d_nlist, h_nlist, (size_t)csr_edges * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(d_eweight, h_eweight, (size_t)csr_edges * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(d_eweight_fp, h_eweight_fp, (size_t)csr_edges * sizeof(double), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
    }

    fail(cudaMemset(d_inMST, 0, (size_t)csr_edges * sizeof(bool)), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail(cudaMemset(d_wlsize, 0, sizeof(int)), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    if (!g_mst_ws.runtime_ready) {
        fail(cudaEventCreate(&g_mst_ws.ev0), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaEventCreate(&g_mst_ws.ev1), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        g_mst_ws.runtime_ready = true;
    }
    fail(cudaEventRecord(g_mst_ws.ev0), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    blocks_nodes = blocks_for(n);
    init_pm<<<blocks_nodes, kThreadsPerBlock>>>(n, d_parent, d_minv);
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    init_worklist<true><<<blocks_nodes, kThreadsPerBlock>>>(
        d_wl1, d_wlsize, n, d_nindex, d_nlist, d_eweight, d_parent, threshold
    );
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    fail(cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    while (wlsize > 0) {
        fail(cudaMemset(d_wlsize, 0, sizeof(int)), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        int blocks_k1 = blocks_for(wlsize);
        kernel_filter_components<<<blocks_k1, kThreadsPerBlock>>>(d_wl1, wlsize, d_wl2, d_wlsize, d_parent, d_minv);
        fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        std::swap(d_wl1, d_wl2);
        fail(cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        if (wlsize > 0) {
            int blocks_kx = blocks_for(wlsize);
            kernel_join_and_mark<<<blocks_kx, kThreadsPerBlock>>>(d_wl1, wlsize, d_parent, d_minv, d_inMST);
            fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
            if (status != EG_GPU_SUCC) goto cleanup;
            kernel_reset_minv<<<blocks_kx, kThreadsPerBlock>>>(d_wl1, wlsize, d_minv);
            fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
            if (status != EG_GPU_SUCC) goto cleanup;
        }
    }

    if (avg_deg >= 4) {
        fail(cudaMemset(d_wlsize, 0, sizeof(int)), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        init_worklist<false><<<blocks_nodes, kThreadsPerBlock>>>(
            d_wl1, d_wlsize, n, d_nindex, d_nlist, d_eweight, d_parent, threshold
        );
        fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        while (wlsize > 0) {
            fail(cudaMemset(d_wlsize, 0, sizeof(int)), EG_GPU_DEVICE_ERR);
            if (status != EG_GPU_SUCC) goto cleanup;
            int blocks_k1 = blocks_for(wlsize);
            kernel_filter_components<<<blocks_k1, kThreadsPerBlock>>>(d_wl1, wlsize, d_wl2, d_wlsize, d_parent, d_minv);
            fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
            if (status != EG_GPU_SUCC) goto cleanup;
            std::swap(d_wl1, d_wl2);
            fail(cudaMemcpy(&wlsize, d_wlsize, sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
            if (status != EG_GPU_SUCC) goto cleanup;
            if (wlsize > 0) {
                int blocks_kx = blocks_for(wlsize);
                kernel_join_and_mark<<<blocks_kx, kThreadsPerBlock>>>(d_wl1, wlsize, d_parent, d_minv, d_inMST);
                fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
                if (status != EG_GPU_SUCC) goto cleanup;
                kernel_reset_minv<<<blocks_kx, kThreadsPerBlock>>>(d_wl1, wlsize, d_minv);
                fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
                if (status != EG_GPU_SUCC) goto cleanup;
            }
        }
    }

    fail(cudaEventRecord(g_mst_ws.ev1), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail(cudaEventSynchronize(g_mst_ws.ev1), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    if (kernel_seconds != nullptr) {
        float elapsed_ms = 0.0f;
        fail(cudaEventElapsedTime(&elapsed_ms, g_mst_ws.ev0, g_mst_ws.ev1), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        *kernel_seconds = (double)elapsed_ms * 1e-3;
    }

    fail(cudaMemset(d_out_count, 0, sizeof(int)), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail(cudaMemset(d_out_overflow, 0, sizeof(int)), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    kernel_collect_mst_edges<<<blocks_nodes, kThreadsPerBlock>>>(
        n,
        d_nindex,
        d_nlist,
        d_eweight_fp,
        d_inMST,
        d_out_src,
        d_out_dst,
        d_out_weight,
        d_out_count,
        d_out_overflow,
        std::max(1, n)
    );
    fail(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail(cudaMemcpy(&h_out_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail(cudaMemcpy(&h_out_overflow, d_out_overflow, sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    if (h_out_overflow || h_out_count < 0 || h_out_count > n) {
        status = EG_GPU_DEVICE_ERR;
        goto cleanup;
    }
    mst_src.resize(h_out_count);
    mst_dst.resize(h_out_count);
    mst_weight.resize(h_out_count);
    if (h_out_count > 0) {
        fail(cudaMemcpy(mst_src.data(), d_out_src, (size_t)h_out_count * sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(mst_dst.data(), d_out_dst, (size_t)h_out_count * sizeof(int), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail(cudaMemcpy(mst_weight.data(), d_out_weight, (size_t)h_out_count * sizeof(double), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
    }

cleanup:
    release_h2d_source(h_nindex, reg_nindex);
    release_h2d_source(h_nlist, reg_nlist);
    release_h2d_source(h_eweight, reg_eweight);
    release_h2d_source(h_eweight_fp, reg_eweight_fp);

    if (status != EG_GPU_SUCC) {
        if (g_mst_ws.ev0 != nullptr) {
            cudaEventDestroy(g_mst_ws.ev0);
            g_mst_ws.ev0 = nullptr;
        }
        if (g_mst_ws.ev1 != nullptr) {
            cudaEventDestroy(g_mst_ws.ev1);
            g_mst_ws.ev1 = nullptr;
        }
        g_mst_ws.runtime_ready = false;
        mst_src.clear();
        mst_dst.clear();
        mst_weight.clear();
        return status;
    }
    return EG_GPU_SUCC;
}

} // namespace gpu_easygraph
