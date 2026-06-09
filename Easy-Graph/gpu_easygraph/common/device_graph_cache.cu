#include "device_graph_cache.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "buffer_cache.h"

namespace gpu_easygraph {

namespace {

static inline std::uint64_t bitcast_u64(double x) {
    std::uint64_t u = 0;
    std::memcpy(&u, &x, sizeof(double));
    return u;
}

struct HostGraphSignature {
    const int* V = nullptr;
    const int* E = nullptr;
    int len_V = -1;
    int len_E = -1;
    int v0 = 0;
    int vn = 0;
    int e0 = 0;
    int em = 0;
    bool ready = false;
};

struct HostWeightSignature {
    const double* W = nullptr;
    int len_E = -1;
    std::uint64_t w0 = 0;
    std::uint64_t wm = 0;
    bool ready = false;
};

struct SharedDeviceCsrCache {
    PersistentDeviceBuffer d_V;
    PersistentDeviceBuffer d_E;
    PersistentDeviceBuffer d_W;
    PersistentPinnedBuffer h_V_stage;
    PersistentPinnedBuffer h_E_stage;
    PersistentPinnedBuffer h_W_stage;
    HostGraphSignature graph_sig;
    HostWeightSignature weight_sig;
    HostCsrStats stats;
};

static thread_local SharedDeviceCsrCache g_csr_cache;

static bool device_csr_cache_disabled() {
    static bool initialized = false;
    static bool disabled = false;
    if (!initialized) {
        initialized = true;
        const char* v = std::getenv("EASYGRAPH_GPU_DISABLE_DEVICE_CSR_CACHE");
        if (v != nullptr) {
            disabled = (
                std::strcmp(v, "1") == 0 ||
                std::strcmp(v, "TRUE") == 0 ||
                std::strcmp(v, "true") == 0 ||
                std::strcmp(v, "ON") == 0 ||
                std::strcmp(v, "on") == 0 ||
                std::strcmp(v, "YES") == 0 ||
                std::strcmp(v, "yes") == 0
            );
        }
    }
    return disabled;
}

static bool graph_signature_matches(
    const HostGraphSignature& sig,
    const int* V,
    const int* E,
    int len_V,
    int len_E
) {
    if (!sig.ready) return false;
    if (sig.V != V || sig.E != E) return false;
    if (sig.len_V != len_V || sig.len_E != len_E) return false;
    if (len_V <= 0) return true;
    if (sig.v0 != V[0] || sig.vn != V[len_V]) return false;
    if (len_E > 0 && (sig.e0 != E[0] || sig.em != E[len_E - 1])) return false;
    return true;
}

static void update_graph_signature(
    HostGraphSignature& sig,
    const int* V,
    const int* E,
    int len_V,
    int len_E
) {
    sig.V = V;
    sig.E = E;
    sig.len_V = len_V;
    sig.len_E = len_E;
    sig.v0 = (len_V > 0) ? V[0] : 0;
    sig.vn = (len_V > 0) ? V[len_V] : 0;
    sig.e0 = (len_E > 0) ? E[0] : 0;
    sig.em = (len_E > 0) ? E[len_E - 1] : 0;
    sig.ready = true;
}

static bool weight_signature_matches(
    const HostWeightSignature& sig,
    const double* W,
    int len_E
) {
    if (!sig.ready) return false;
    if (sig.W != W || sig.len_E != len_E) return false;
    if (len_E <= 0) return true;
    return sig.w0 == bitcast_u64(W[0]) && sig.wm == bitcast_u64(W[len_E - 1]);
}

static void update_weight_signature(
    HostWeightSignature& sig,
    const double* W,
    int len_E
) {
    sig.W = W;
    sig.len_E = len_E;
    sig.w0 = (W != nullptr && len_E > 0) ? bitcast_u64(W[0]) : 0;
    sig.wm = (W != nullptr && len_E > 0) ? bitcast_u64(W[len_E - 1]) : 0;
    sig.ready = (W != nullptr);
}

} // namespace

int acquire_device_csr(
    const int* V,
    const int* E,
    const double* W,
    int len_V,
    int len_E,
    bool require_weights,
    DeviceCsrView* out
) {
    if (out == nullptr || V == nullptr || E == nullptr || len_V < 0 || len_E < 0) {
        return EG_GPU_DEVICE_ERR;
    }

    out->d_V = nullptr;
    out->d_E = nullptr;
    out->d_W = nullptr;
    out->stats = HostCsrStats();
    const bool disable_cache = device_csr_cache_disabled();
    out->structure_changed = disable_cache || !graph_signature_matches(g_csr_cache.graph_sig, V, E, len_V, len_E);
    out->weights_changed = require_weights
        ? (disable_cache || !weight_signature_matches(g_csr_cache.weight_sig, W, len_E))
        : false;

    int rc = g_csr_cache.d_V.ensure_bytes(sizeof(int) * (static_cast<std::size_t>(len_V) + 1));
    if (rc != EG_GPU_SUCC) return rc;
    rc = g_csr_cache.d_E.ensure_bytes(sizeof(int) * static_cast<std::size_t>(len_E));
    if (rc != EG_GPU_SUCC) return rc;
    if (require_weights) {
        if (W == nullptr && len_E > 0) return EG_GPU_DEVICE_ERR;
        rc = g_csr_cache.d_W.ensure_bytes(sizeof(double) * static_cast<std::size_t>(len_E));
        if (rc != EG_GPU_SUCC) return rc;
    }

    int* d_V = g_csr_cache.d_V.as<int>();
    int* d_E = g_csr_cache.d_E.as<int>();
    double* d_W = require_weights ? g_csr_cache.d_W.as<double>() : nullptr;

    if (out->structure_changed) {
        HostCsrStats new_stats = adaptive_policy_enabled()
            ? summarize_host_csr(V, len_V, len_E)
            : HostCsrStats();
        bool reg_V = false;
        bool reg_E = false;
        const int* h_V = prepare_h2d_source(V, static_cast<std::size_t>(len_V) + 1, g_csr_cache.h_V_stage, &reg_V);
        const int* h_E = prepare_h2d_source(E, static_cast<std::size_t>(len_E), g_csr_cache.h_E_stage, &reg_E);
        if (h_V == nullptr) h_V = V;
        if (h_E == nullptr) h_E = E;
        cudaError_t ret = cudaMemcpy(d_V, h_V, sizeof(int) * (static_cast<std::size_t>(len_V) + 1), cudaMemcpyHostToDevice);
        release_h2d_source(h_V, reg_V);
        if (ret != cudaSuccess) return (ret == cudaErrorMemoryAllocation) ? EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM : EG_GPU_DEVICE_ERR;
        ret = cudaMemcpy(d_E, h_E, sizeof(int) * static_cast<std::size_t>(len_E), cudaMemcpyHostToDevice);
        release_h2d_source(h_E, reg_E);
        if (ret != cudaSuccess) return (ret == cudaErrorMemoryAllocation) ? EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM : EG_GPU_DEVICE_ERR;
        if (!disable_cache) {
            update_graph_signature(g_csr_cache.graph_sig, V, E, len_V, len_E);
            g_csr_cache.stats = new_stats;
        }
        out->stats = new_stats;
    } else {
        out->stats = g_csr_cache.stats;
    }

    if (require_weights && out->weights_changed) {
        bool reg_W = false;
        const double* h_W = prepare_h2d_source(W, static_cast<std::size_t>(len_E), g_csr_cache.h_W_stage, &reg_W);
        if (h_W == nullptr) h_W = W;
        cudaError_t ret = cudaMemcpy(d_W, h_W, sizeof(double) * static_cast<std::size_t>(len_E), cudaMemcpyHostToDevice);
        release_h2d_source(h_W, reg_W);
        if (ret != cudaSuccess) return (ret == cudaErrorMemoryAllocation) ? EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM : EG_GPU_DEVICE_ERR;
        if (!disable_cache) {
            update_weight_signature(g_csr_cache.weight_sig, W, len_E);
        }
    }

    out->d_V = d_V;
    out->d_E = d_E;
    out->d_W = d_W;
    if (!out->stats.ready && adaptive_policy_enabled()) {
        out->stats = summarize_host_csr(V, len_V, len_E);
        if (!disable_cache) g_csr_cache.stats = out->stats;
    }
    return EG_GPU_SUCC;
}

void reset_device_csr_cache() {
    g_csr_cache.d_V.reset();
    g_csr_cache.d_E.reset();
    g_csr_cache.d_W.reset();
    g_csr_cache.h_V_stage.reset();
    g_csr_cache.h_E_stage.reset();
    g_csr_cache.h_W_stage.reset();
    g_csr_cache.graph_sig = HostGraphSignature();
    g_csr_cache.weight_sig = HostWeightSignature();
    g_csr_cache.stats = HostCsrStats();
}

} // namespace gpu_easygraph
