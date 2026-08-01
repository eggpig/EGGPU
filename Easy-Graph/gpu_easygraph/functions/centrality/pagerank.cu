#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "centrality/pagerank.cuh"
#include "buffer_cache.h"
#include "device_graph_cache.h"
#include "err.h"

namespace gpu_easygraph {

namespace {

using pr_real = float;
constexpr cudaDataType CUDA_R_PR_REAL = CUDA_R_32F;

__device__ __forceinline__ double atomic_add_double_compat(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull =
        reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(val + __longlong_as_double(assumed))
        );
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

struct PagerankWorkspace {
    PersistentDeviceBuffer d_in_row;
    PersistentDeviceBuffer d_in_col;
    PersistentDeviceBuffer d_in_val;
    PersistentDeviceBuffer d_x;
    PersistentDeviceBuffer d_y;
    PersistentDeviceBuffer d_dmask;
    PersistentDeviceBuffer d_dang;
    PersistentDeviceBuffer d_sumY;
    PersistentDeviceBuffer d_l1diff;
    PersistentDeviceBuffer d_tmp_int;
    PersistentDeviceBuffer d_spmv_buf;

    PersistentPinnedBuffer h_in_row;
    PersistentPinnedBuffer h_in_col;
    PersistentPinnedBuffer h_in_val;
    PersistentPinnedBuffer h_dangling01;
    PersistentPinnedBuffer h_pr_raw;
    PersistentPinnedBuffer h_scalars;

    // Cached inbound graph tensors to reuse CPU preprocessing + H2D upload
    // across repeated calls on the same CSR graph.
    std::vector<int> cached_in_row;
    std::vector<int> cached_in_col;
    std::vector<pr_real> cached_in_val;
    std::vector<int> cached_dangling01;

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
    std::uint64_t sig_alpha = 0;
    std::uint64_t sig_graph_id = 0;
    int owner_device_id = -1;
    bool graph_cached = false;

    // Reusable CUDA runtime objects.
    cublasHandle_t h_bl = nullptr;
    cusparseHandle_t h_sp = nullptr;
    cudaStream_t s_spmv = nullptr;
    cudaStream_t s_aux = nullptr;
    cudaEvent_t ev_dang = nullptr;
    cudaEvent_t ev_begin = nullptr;
    cudaEvent_t ev_end = nullptr;
    bool runtime_ready = false;

    ~PagerankWorkspace() { release_runtime(); }

    void release_runtime() {
        if (h_sp != nullptr) {
            cusparseDestroy(h_sp);
            h_sp = nullptr;
        }
        if (h_bl != nullptr) {
            cublasDestroy(h_bl);
            h_bl = nullptr;
        }
        if (ev_dang != nullptr) {
            cudaEventDestroy(ev_dang);
            ev_dang = nullptr;
        }
        if (ev_begin != nullptr) {
            cudaEventDestroy(ev_begin);
            ev_begin = nullptr;
        }
        if (ev_end != nullptr) {
            cudaEventDestroy(ev_end);
            ev_end = nullptr;
        }
        if (s_spmv != nullptr) {
            cudaStreamDestroy(s_spmv);
            s_spmv = nullptr;
        }
        if (s_aux != nullptr) {
            cudaStreamDestroy(s_aux);
            s_aux = nullptr;
        }
        runtime_ready = false;
    }
};

static thread_local PagerankWorkspace g_pr_ws;

static inline std::uint64_t bitcast_u64(double x) {
    std::uint64_t u = 0;
    std::memcpy(&u, &x, sizeof(double));
    return u;
}

static bool graph_signature_matches(
    const PagerankWorkspace& ws,
    const int* V,
    const int* E,
    const double* W,
    int n,
    int m,
    double alpha,
    std::uint64_t graph_id,
    int device_id
) {
    if (!ws.graph_cached) return false;
    if (ws.sig_graph_id != graph_id || ws.owner_device_id != device_id) return false;
    if (ws.sig_V != V || ws.sig_E != E || ws.sig_W != W) return false;
    if (ws.sig_n != n || ws.sig_m != m) return false;
    if (ws.sig_alpha != bitcast_u64(alpha)) return false;
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
    PagerankWorkspace& ws,
    const int* V,
    const int* E,
    const double* W,
    int n,
    int m,
    double alpha,
    std::uint64_t graph_id,
    int device_id
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
    ws.sig_alpha = bitcast_u64(alpha);
    ws.sig_graph_id = graph_id;
    ws.owner_device_id = device_id;
    ws.graph_cached = true;
}

static int ensure_runtime(PagerankWorkspace& ws) {
    if (ws.runtime_ready) return EG_GPU_SUCC;

    if (cusparseCreate(&ws.h_sp) != CUSPARSE_STATUS_SUCCESS) goto fail;
    if (cublasCreate(&ws.h_bl) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (cudaStreamCreate(&ws.s_spmv) != cudaSuccess) goto fail;
    if (cudaStreamCreate(&ws.s_aux) != cudaSuccess) goto fail;
    if (cudaEventCreateWithFlags(&ws.ev_dang, cudaEventDisableTiming) != cudaSuccess) goto fail;
    if (cudaEventCreate(&ws.ev_begin) != cudaSuccess) goto fail;
    if (cudaEventCreate(&ws.ev_end) != cudaSuccess) goto fail;
    if (cusparseSetStream(ws.h_sp, ws.s_spmv) != CUSPARSE_STATUS_SUCCESS) goto fail;
    if (cublasSetStream(ws.h_bl, ws.s_aux) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (cublasSetPointerMode(ws.h_bl, CUBLAS_POINTER_MODE_DEVICE) != CUBLAS_STATUS_SUCCESS) goto fail;

    ws.runtime_ready = true;
    return EG_GPU_SUCC;
fail:
    ws.release_runtime();
    return EG_GPU_DEVICE_ERR;
}

__global__ void init_uniform(pr_real* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (n > 0) ? (pr_real)(1.0f / (pr_real)n) : (pr_real)0;
}

__global__ void int_to_float_mask(const int* in_mask, pr_real* out_mask, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out_mask[i] = in_mask[i] ? (pr_real)1.0f : (pr_real)0.0f;
}

__global__ void scale_vec(pr_real* x, int n, pr_real factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}

__global__ void add_base_and_reduce(
    pr_real* __restrict__ y,
    const pr_real* __restrict__ x,
    int n,
    pr_real c1,
    pr_real c2,
    const pr_real* __restrict__ d_dang,
    double* __restrict__ sum_y,
    double* __restrict__ l1diff
) {
    extern __shared__ double smem[];
    double* s_sum = smem;
    double* s_diff = smem + blockDim.x;

    __shared__ pr_real s_base;
    if (threadIdx.x == 0) s_base = c1 + c2 * (*d_dang);
    __syncthreads();

    const pr_real base = s_base;
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int i = blockIdx.x * blockDim.x + tid;

    double local_sum = 0.0;
    double local_diff = 0.0;
    for (; i < n; i += stride) {
        pr_real v = y[i] + base;
        y[i] = v;
        local_sum += fabs((double)v);
        local_diff += fabs((double)v - (double)x[i]);
    }

    s_sum[tid] = local_sum;
    s_diff[tid] = local_diff;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
            s_diff[tid] += s_diff[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomic_add_double_compat(sum_y, s_sum[0]);
        atomic_add_double_compat(l1diff, s_diff[0]);
    }
}

static inline bool valid_edge_endpoint(int x, int n) {
    return (0 <= x) && (x < n);
}

static void build_inbound_from_csr(
    const int* V,
    const int* E,
    const double* W,
    int n,
    int m,
    double alpha,
    std::vector<int>& in_row,
    std::vector<int>& in_col,
    std::vector<pr_real>& in_val,
    std::vector<int>& dangling01
) {
    std::vector<double> out_weight_sum(n, 0.0);
    std::vector<int> indeg(n, 0);

    for (int u = 0; u < n; ++u) {
        int begin = V[u];
        int end = V[u + 1];
        if (begin < 0) begin = 0;
        if (end > m) end = m;
        if (begin > end) begin = end;

        for (int p = begin; p < end; ++p) {
            const int v = E[p];
            if (!valid_edge_endpoint(v, n)) continue;
            double w = (W != nullptr) ? W[p] : 1.0;
            if (!std::isfinite(w) || w < 0.0) w = 0.0;
            out_weight_sum[u] += w;
            indeg[v] += 1;
        }
    }

    in_row.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) in_row[i + 1] = in_row[i] + indeg[i];
    in_col.assign(in_row[n], 0);
    in_val.assign(in_row[n], (pr_real)0);

    dangling01.assign(n, 0);
    for (int u = 0; u < n; ++u) {
        if (out_weight_sum[u] <= 0.0) dangling01[u] = 1;
    }

    std::vector<int> cur = in_row;
    for (int u = 0; u < n; ++u) {
        int begin = V[u];
        int end = V[u + 1];
        if (begin < 0) begin = 0;
        if (end > m) end = m;
        if (begin > end) begin = end;

        const double denom = out_weight_sum[u];
        for (int p = begin; p < end; ++p) {
            const int v = E[p];
            if (!valid_edge_endpoint(v, n)) continue;
            double w = (W != nullptr) ? W[p] : 1.0;
            if (!std::isfinite(w) || w < 0.0) w = 0.0;
            const int pos = cur[v]++;
            in_col[pos] = u;
            in_val[pos] = (denom > 0.0) ? (pr_real)(alpha * (w / denom)) : (pr_real)0.0f;
        }
    }
}

} // namespace


int cuda_pagerank(
    const int* V,
    const int* E,
    const double* W,
    int len_V,
    int len_E,
    double alpha,
    int max_iter_num,
    double threshold,
    std::vector<double>& PR,
    double* kernel_seconds
) {
    const int n = len_V;
    const int m = len_E;
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    PR.clear();
    if (n <= 0) return EG_GPU_SUCC;
    int device_id = -1;
    if (cudaGetDevice(&device_id) != cudaSuccess || device_id < 0) {
        return EG_GPU_DEVICE_ERR;
    }
    if (g_pr_ws.owner_device_id >= 0 && g_pr_ws.owner_device_id != device_id) {
        // Persistent buffers, streams, handles, and events are device-owned.
        // EGGPU binds one worker thread to one CUDA device while they live.
        return EG_GPU_DEVICE_ERR;
    }
    g_pr_ws.owner_device_id = device_id;
    const std::uint64_t graph_id = active_device_graph_id();
    const bool graph_changed = !graph_signature_matches(
        g_pr_ws, V, E, W, n, m, alpha, graph_id, device_id);

    int* d_in_row = nullptr;
    int* d_in_col = nullptr;
    pr_real* d_in_val = nullptr;
    pr_real* d_x = nullptr;
    pr_real* d_y = nullptr;
    pr_real* d_dmask = nullptr;
    pr_real* d_dang = nullptr;
    double* d_sumY = nullptr;
    double* d_l1diff = nullptr;
    int* d_tmp = nullptr;
    void* d_buf = nullptr;
    const int* h_in_row = nullptr;
    const int* h_in_col = nullptr;
    const pr_real* h_in_val = nullptr;
    const int* h_dangling = nullptr;
    double* h_scalars = nullptr;
    double* h_sum_y_p = nullptr;
    double* h_diff_p = nullptr;
    pr_real* pr_raw = nullptr;
    bool reg_in_row = false;
    bool reg_in_col = false;
    bool reg_in_val = false;
    bool reg_dangling = false;

    cusparseSpMatDescr_t A_in = nullptr;
    cusparseDnVecDescr_t vx = nullptr;
    cusparseDnVecDescr_t vy = nullptr;

    int status = EG_GPU_SUCC;
    cudaError_t cuda_ret = cudaSuccess;
    cublasStatus_t cublas_ret = CUBLAS_STATUS_SUCCESS;
    cusparseStatus_t cusparse_ret = CUSPARSE_STATUS_SUCCESS;
    int block = 256;
    int grid = 1;
    int blocks = 1;
    int64_t nnz = 0;
    size_t buf_size = 0;
    pr_real spmv_alpha = (pr_real)1.0f;
    pr_real zero = (pr_real)0.0f;
    pr_real c1 = (pr_real)0.0f;
    pr_real c2 = (pr_real)0.0f;
    double stop_eps = 0.0;
    double sum = 0.0;

    auto fail_if_cuda = [&](cudaError_t err, int fail_code) {
        cuda_ret = err;
        if (cuda_ret != cudaSuccess) status = fail_code;
    };
    auto fail_if_status = [&](int rc) {
        if (rc != EG_GPU_SUCC) status = rc;
    };
    auto fail_if_cusparse = [&](cusparseStatus_t s) {
        cusparse_ret = s;
        if (cusparse_ret != CUSPARSE_STATUS_SUCCESS) status = EG_GPU_DEVICE_ERR;
    };
    auto fail_if_cublas = [&](cublasStatus_t s) {
        cublas_ret = s;
        if (cublas_ret != CUBLAS_STATUS_SUCCESS) status = EG_GPU_DEVICE_ERR;
    };

    fail_if_status(ensure_runtime(g_pr_ws));
    if (status != EG_GPU_SUCC) goto cleanup;

    fail_if_status(g_pr_ws.d_x.ensure_bytes((size_t)n * sizeof(pr_real)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_pr_ws.d_y.ensure_bytes((size_t)n * sizeof(pr_real)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_pr_ws.d_dmask.ensure_bytes((size_t)n * sizeof(pr_real)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_pr_ws.d_dang.ensure_bytes(sizeof(pr_real)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_pr_ws.d_sumY.ensure_bytes(sizeof(double)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_pr_ws.d_l1diff.ensure_bytes(sizeof(double)));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_pr_ws.d_tmp_int.ensure_bytes((size_t)n * sizeof(int)));
    if (status != EG_GPU_SUCC) goto cleanup;

    d_in_row = g_pr_ws.d_in_row.as<int>();
    d_in_col = g_pr_ws.d_in_col.as<int>();
    d_in_val = g_pr_ws.d_in_val.as<pr_real>();
    d_x = g_pr_ws.d_x.as<pr_real>();
    d_y = g_pr_ws.d_y.as<pr_real>();
    d_dmask = g_pr_ws.d_dmask.as<pr_real>();
    d_dang = g_pr_ws.d_dang.as<pr_real>();
    d_sumY = g_pr_ws.d_sumY.as<double>();
    d_l1diff = g_pr_ws.d_l1diff.as<double>();
    d_tmp = g_pr_ws.d_tmp_int.as<int>();

    if (graph_changed) {
        build_inbound_from_csr(
            V,
            E,
            W,
            n,
            m,
            alpha,
            g_pr_ws.cached_in_row,
            g_pr_ws.cached_in_col,
            g_pr_ws.cached_in_val,
            g_pr_ws.cached_dangling01
        );

        fail_if_status(g_pr_ws.d_in_row.ensure_bytes((size_t)(n + 1) * sizeof(int)));
        if (status != EG_GPU_SUCC) goto cleanup;
        fail_if_status(g_pr_ws.d_in_col.ensure_bytes((size_t)g_pr_ws.cached_in_col.size() * sizeof(int)));
        if (status != EG_GPU_SUCC) goto cleanup;
        fail_if_status(g_pr_ws.d_in_val.ensure_bytes((size_t)g_pr_ws.cached_in_val.size() * sizeof(pr_real)));
        if (status != EG_GPU_SUCC) goto cleanup;

        d_in_row = g_pr_ws.d_in_row.as<int>();
        d_in_col = g_pr_ws.d_in_col.as<int>();
        d_in_val = g_pr_ws.d_in_val.as<pr_real>();

        h_in_row = prepare_h2d_source(g_pr_ws.cached_in_row.data(), (size_t)(n + 1), g_pr_ws.h_in_row, &reg_in_row);
        h_in_col = prepare_h2d_source(g_pr_ws.cached_in_col.data(), g_pr_ws.cached_in_col.size(), g_pr_ws.h_in_col, &reg_in_col);
        h_in_val = prepare_h2d_source(g_pr_ws.cached_in_val.data(), g_pr_ws.cached_in_val.size(), g_pr_ws.h_in_val, &reg_in_val);
        h_dangling = prepare_h2d_source(g_pr_ws.cached_dangling01.data(), (size_t)n, g_pr_ws.h_dangling01, &reg_dangling);
        if (h_in_row == nullptr) h_in_row = g_pr_ws.cached_in_row.data();
        if (h_in_col == nullptr) h_in_col = g_pr_ws.cached_in_col.data();
        if (h_in_val == nullptr) h_in_val = g_pr_ws.cached_in_val.data();
        if (h_dangling == nullptr) h_dangling = g_pr_ws.cached_dangling01.data();

        fail_if_cuda(cudaMemcpy(d_in_row, h_in_row, (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        if (!g_pr_ws.cached_in_col.empty()) {
            fail_if_cuda(cudaMemcpy(d_in_col, h_in_col, (size_t)g_pr_ws.cached_in_col.size() * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
            if (status != EG_GPU_SUCC) goto cleanup;
            fail_if_cuda(cudaMemcpy(d_in_val, h_in_val, (size_t)g_pr_ws.cached_in_val.size() * sizeof(pr_real), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
            if (status != EG_GPU_SUCC) goto cleanup;
        }

        grid = (n + block - 1) / block;
        fail_if_cuda(cudaMemcpy(d_tmp, h_dangling, (size_t)n * sizeof(int), cudaMemcpyHostToDevice), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        int_to_float_mask<<<grid, block>>>(d_tmp, d_dmask, n);
        fail_if_cuda(cudaGetLastError(), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;

        update_graph_signature(
            g_pr_ws, V, E, W, n, m, alpha, graph_id, device_id);
    } else {
        d_in_row = g_pr_ws.d_in_row.as<int>();
        d_in_col = g_pr_ws.d_in_col.as<int>();
        d_in_val = g_pr_ws.d_in_val.as<pr_real>();
    }

    grid = (n + block - 1) / block;
    init_uniform<<<grid, block>>>(d_x, n);
    fail_if_cuda(cudaGetLastError(), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    if (g_pr_ws.cached_in_col.empty()) {
        PR.assign(n, 1.0 / (double)n);
        if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
        goto cleanup;
    }

    nnz = (int64_t)g_pr_ws.cached_in_col.size();
    fail_if_cusparse(cusparseCreateCsr(
        &A_in,
        (int64_t)n,
        (int64_t)n,
        nnz,
        d_in_row,
        d_in_col,
        d_in_val,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_PR_REAL
    ));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_cusparse(cusparseCreateDnVec(&vx, n, d_x, CUDA_R_PR_REAL));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_cusparse(cusparseCreateDnVec(&vy, n, d_y, CUDA_R_PR_REAL));
    if (status != EG_GPU_SUCC) goto cleanup;

    fail_if_cusparse(cusparseSpMV_bufferSize(
        g_pr_ws.h_sp,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha,
        A_in,
        vx,
        &zero,
        vy,
        CUDA_R_PR_REAL,
        CUSPARSE_SPMV_CSR_ALG2,
        &buf_size
    ));
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_status(g_pr_ws.d_spmv_buf.ensure_bytes(buf_size));
    if (status != EG_GPU_SUCC) goto cleanup;
    d_buf = g_pr_ws.d_spmv_buf.data();

    fail_if_cuda(cudaEventRecord(g_pr_ws.ev_begin, g_pr_ws.s_spmv), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    c1 = (pr_real)(((double)1.0 - alpha) / (double)n);
    c2 = (pr_real)(alpha / (double)n);
    stop_eps = std::max(0.0, threshold) * (double)n;
    blocks = std::max(1, std::min(grid, 4096));
    fail_if_status(g_pr_ws.h_scalars.ensure_bytes(sizeof(double) * 2));
    if (status != EG_GPU_SUCC) goto cleanup;
    h_scalars = g_pr_ws.h_scalars.as<double>();
    h_sum_y_p = h_scalars;
    h_diff_p = h_scalars + 1;

    for (int iter = 0; iter < std::max(1, max_iter_num); ++iter) {
        fail_if_cusparse(cusparseSpMV(
            g_pr_ws.h_sp,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha,
            A_in,
            vx,
            &zero,
            vy,
            CUDA_R_PR_REAL,
            CUSPARSE_SPMV_CSR_ALG2,
            d_buf
        ));
        if (status != EG_GPU_SUCC) goto cleanup;

        fail_if_cublas(cublasSdot(
            g_pr_ws.h_bl,
            n,
            d_dmask,
            1,
            d_x,
            1,
            d_dang
        ));
        if (status != EG_GPU_SUCC) goto cleanup;
        fail_if_cuda(cudaEventRecord(g_pr_ws.ev_dang, g_pr_ws.s_aux), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;

        fail_if_cuda(cudaStreamWaitEvent(g_pr_ws.s_spmv, g_pr_ws.ev_dang, 0), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail_if_cuda(cudaMemsetAsync(d_sumY, 0, sizeof(double), g_pr_ws.s_spmv), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail_if_cuda(cudaMemsetAsync(d_l1diff, 0, sizeof(double), g_pr_ws.s_spmv), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;

        add_base_and_reduce<<<blocks, block, (size_t)(2 * block) * sizeof(double), g_pr_ws.s_spmv>>>(
            d_y,
            d_x,
            n,
            c1,
            c2,
            d_dang,
            d_sumY,
            d_l1diff
        );
        fail_if_cuda(cudaGetLastError(), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        fail_if_cuda(cudaStreamSynchronize(g_pr_ws.s_spmv), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;

        fail_if_cuda(cudaMemcpy(h_sum_y_p, d_sumY, sizeof(double), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        double h_sum_y = *h_sum_y_p;
        if (h_sum_y > 0.0) {
            pr_real inv = (pr_real)(1.0 / h_sum_y);
            scale_vec<<<grid, block, 0, g_pr_ws.s_spmv>>>(d_y, n, inv);
            fail_if_cuda(cudaGetLastError(), EG_GPU_DEVICE_ERR);
            if (status != EG_GPU_SUCC) goto cleanup;
            fail_if_cuda(cudaStreamSynchronize(g_pr_ws.s_spmv), EG_GPU_DEVICE_ERR);
            if (status != EG_GPU_SUCC) goto cleanup;
        }

        fail_if_cuda(cudaMemcpy(h_diff_p, d_l1diff, sizeof(double), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        double h_diff = *h_diff_p;

        std::swap(d_x, d_y);
        fail_if_cusparse(cusparseDnVecSetValues(vx, d_x));
        if (status != EG_GPU_SUCC) goto cleanup;
        fail_if_cusparse(cusparseDnVecSetValues(vy, d_y));
        if (status != EG_GPU_SUCC) goto cleanup;

        if (h_diff < stop_eps) break;
    }

    fail_if_cuda(cudaEventRecord(g_pr_ws.ev_end, g_pr_ws.s_spmv), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    fail_if_cuda(cudaEventSynchronize(g_pr_ws.ev_end), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;

    if (kernel_seconds != nullptr) {
        float elapsed_ms = 0.0f;
        fail_if_cuda(cudaEventElapsedTime(&elapsed_ms, g_pr_ws.ev_begin, g_pr_ws.ev_end), EG_GPU_DEVICE_ERR);
        if (status != EG_GPU_SUCC) goto cleanup;
        *kernel_seconds = (double)elapsed_ms * 1e-3;
    }

    fail_if_status(g_pr_ws.h_pr_raw.ensure_bytes((size_t)n * sizeof(pr_real)));
    if (status != EG_GPU_SUCC) goto cleanup;
    pr_raw = g_pr_ws.h_pr_raw.as<pr_real>();
    fail_if_cuda(cudaMemcpy(pr_raw, d_x, (size_t)n * sizeof(pr_real), cudaMemcpyDeviceToHost), EG_GPU_DEVICE_ERR);
    if (status != EG_GPU_SUCC) goto cleanup;
    PR.resize(n, 0.0);
    sum = 0.0;
    for (int i = 0; i < n; ++i) {
        PR[i] = (double)pr_raw[i];
        sum += PR[i];
    }
    if (sum > 0.0) {
        double inv = 1.0 / sum;
        for (double& x : PR) x *= inv;
    } else {
        const double uniform = 1.0 / (double)n;
        for (double& x : PR) x = uniform;
    }

cleanup:
    release_h2d_source(h_in_row, reg_in_row);
    release_h2d_source(h_in_col, reg_in_col);
    release_h2d_source(h_in_val, reg_in_val);
    release_h2d_source(h_dangling, reg_dangling);
    if (A_in != nullptr) cusparseDestroySpMat(A_in);
    if (vx != nullptr) cusparseDestroyDnVec(vx);
    if (vy != nullptr) cusparseDestroyDnVec(vy);

    if (status != EG_GPU_SUCC) {
        // Runtime objects may be in a partial/bad state after CUDA failures.
        g_pr_ws.release_runtime();
        PR.clear();
        return status;
    }
    return EG_GPU_SUCC;
}

} // namespace gpu_easygraph
