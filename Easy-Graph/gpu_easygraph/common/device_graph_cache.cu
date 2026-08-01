#include "device_graph_cache.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

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
    std::uint64_t graph_id = 0;
    const int* V = nullptr;
    const int* E = nullptr;
    int len_V = -1;
    int len_E = -1;
    int v0 = 0;
    int vm = 0;
    int vn = 0;
    int e0 = 0;
    int eq1 = 0;
    int em = 0;
    int eq3 = 0;
    int en = 0;
    bool ready = false;
};

struct HostWeightSignature {
    const double* W = nullptr;
    int len_E = -1;
    std::uint64_t w0 = 0;
    std::uint64_t wq1 = 0;
    std::uint64_t wm = 0;
    std::uint64_t wq3 = 0;
    std::uint64_t wn = 0;
    bool ready = false;
};

struct SharedDeviceCsrCache {
    int device_id = -1;
    PersistentDeviceBuffer d_V;
    PersistentDeviceBuffer d_E;
    PersistentDeviceBuffer d_W;
    HostGraphSignature graph_sig;
    HostWeightSignature weight_sig;
    HostCsrStats stats;
    std::uint64_t last_use = 0;

    explicit SharedDeviceCsrCache(int device) : device_id(device) {}

    ~SharedDeviceCsrCache() { reset(); }

    std::size_t device_bytes() const {
        return d_V.capacity_bytes() + d_E.capacity_bytes() + d_W.capacity_bytes();
    }

    void invalidate() {
        graph_sig = HostGraphSignature();
        weight_sig = HostWeightSignature();
        stats = HostCsrStats();
        last_use = 0;
    }

    void reset() {
        int previous_device = -1;
        const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        const bool switch_device = (
            device_id >= 0 && have_previous && previous_device != device_id &&
            cudaSetDevice(device_id) == cudaSuccess
        );
        d_V.reset();
        d_E.reset();
        d_W.reset();
        if (switch_device) (void)cudaSetDevice(previous_device);
        invalidate();
    }
};

struct ThreadDeviceCsrRegistry {
    std::vector<std::unique_ptr<SharedDeviceCsrCache>> entries;
    PersistentPinnedBuffer h_V_stage;
    PersistentPinnedBuffer h_E_stage;
    PersistentPinnedBuffer h_W_stage;
    DeviceCsrCacheStats counters;
    std::uint64_t logical_clock = 0;
};

static thread_local ThreadDeviceCsrRegistry g_csr_registry;
static thread_local std::uint64_t g_active_graph_id = 0;

static bool env_truthy(const char* value) {
    if (value == nullptr) return false;
    return (
        std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "TRUE") == 0 ||
        std::strcmp(value, "true") == 0 ||
        std::strcmp(value, "ON") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "YES") == 0 ||
        std::strcmp(value, "yes") == 0
    );
}

static bool device_csr_cache_disabled() {
    static const bool disabled = env_truthy(
        std::getenv("EASYGRAPH_GPU_DISABLE_DEVICE_CSR_CACHE")
    );
    return disabled;
}

static std::size_t max_entries_per_device() {
    static const std::size_t value = []() {
        const char* raw = std::getenv("EASYGRAPH_GPU_DEVICE_CSR_CACHE_MAX_ENTRIES");
        if (raw == nullptr) return static_cast<std::size_t>(4);
        const unsigned long long parsed = std::strtoull(raw, nullptr, 10);
        if (parsed == 0ULL) return static_cast<std::size_t>(1);
        return static_cast<std::size_t>(std::min<unsigned long long>(parsed, 16ULL));
    }();
    return value;
}

static std::size_t cached_device_bytes(int device_id) {
    std::size_t bytes = 0;
    for (const auto& entry : g_csr_registry.entries) {
        if (entry->device_id == device_id) bytes += entry->device_bytes();
    }
    return bytes;
}

static std::size_t cache_budget_bytes(int device_id) {
    const char* raw = std::getenv("EASYGRAPH_GPU_DEVICE_CSR_CACHE_MAX_BYTES");
    if (raw != nullptr) {
        const unsigned long long parsed = std::strtoull(raw, nullptr, 10);
        if (parsed > 0ULL) return static_cast<std::size_t>(parsed);
    }
    int previous_device = -1;
    const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
    const bool switch_device = (
        have_previous && previous_device != device_id &&
        cudaSetDevice(device_id) == cudaSuccess
    );
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (switch_device) (void)cudaSetDevice(previous_device);
    if (status != cudaSuccess || total_bytes == 0) {
        return static_cast<std::size_t>(2) << 30;
    }
    const std::size_t cache_owned = cached_device_bytes(device_id);
    const std::size_t available = free_bytes > std::numeric_limits<std::size_t>::max() - cache_owned
        ? std::numeric_limits<std::size_t>::max()
        : free_bytes + cache_owned;
    return std::max<std::size_t>(available / 4, static_cast<std::size_t>(256) << 20);
}

static bool graph_signature_matches(
    const HostGraphSignature& sig,
    std::uint64_t graph_id,
    const int* V,
    const int* E,
    int len_V,
    int len_E
) {
    if (!sig.ready) return false;
    if (sig.graph_id != graph_id) return false;
    if (sig.V != V || sig.E != E) return false;
    if (sig.len_V != len_V || sig.len_E != len_E) return false;
    if (len_V > 0) {
        if (sig.v0 != V[0] || sig.vm != V[len_V / 2] || sig.vn != V[len_V]) return false;
    }
    if (len_E > 0) {
        if (
            sig.e0 != E[0] || sig.eq1 != E[len_E / 4] ||
            sig.em != E[len_E / 2] ||
            sig.eq3 != E[(static_cast<std::size_t>(3) * len_E) / 4] ||
            sig.en != E[len_E - 1]
        ) return false;
    }
    return true;
}

static void update_graph_signature(
    HostGraphSignature& sig,
    std::uint64_t graph_id,
    const int* V,
    const int* E,
    int len_V,
    int len_E
) {
    sig.graph_id = graph_id;
    sig.V = V;
    sig.E = E;
    sig.len_V = len_V;
    sig.len_E = len_E;
    sig.v0 = (len_V > 0) ? V[0] : 0;
    sig.vm = (len_V > 0) ? V[len_V / 2] : 0;
    sig.vn = (len_V > 0) ? V[len_V] : 0;
    sig.e0 = (len_E > 0) ? E[0] : 0;
    sig.eq1 = (len_E > 0) ? E[len_E / 4] : 0;
    sig.em = (len_E > 0) ? E[len_E / 2] : 0;
    sig.eq3 = (len_E > 0)
        ? E[(static_cast<std::size_t>(3) * len_E) / 4]
        : 0;
    sig.en = (len_E > 0) ? E[len_E - 1] : 0;
    sig.ready = true;
}

static bool weight_signature_matches(
    const HostWeightSignature& sig,
    const double* W,
    int len_E
) {
    if (!sig.ready || sig.W != W || sig.len_E != len_E) return false;
    if (len_E <= 0) return true;
    return (
        sig.w0 == bitcast_u64(W[0]) &&
        sig.wq1 == bitcast_u64(W[len_E / 4]) &&
        sig.wm == bitcast_u64(W[len_E / 2]) &&
        sig.wq3 == bitcast_u64(
            W[(static_cast<std::size_t>(3) * len_E) / 4]
        ) &&
        sig.wn == bitcast_u64(W[len_E - 1])
    );
}

static void update_weight_signature(
    HostWeightSignature& sig,
    const double* W,
    int len_E
) {
    sig.W = W;
    sig.len_E = len_E;
    sig.w0 = (W != nullptr && len_E > 0) ? bitcast_u64(W[0]) : 0;
    sig.wq1 = (W != nullptr && len_E > 0) ? bitcast_u64(W[len_E / 4]) : 0;
    sig.wm = (W != nullptr && len_E > 0) ? bitcast_u64(W[len_E / 2]) : 0;
    sig.wq3 = (W != nullptr && len_E > 0)
        ? bitcast_u64(W[(static_cast<std::size_t>(3) * len_E) / 4])
        : 0;
    sig.wn = (W != nullptr && len_E > 0) ? bitcast_u64(W[len_E - 1]) : 0;
    sig.ready = (W != nullptr);
}

static std::size_t entry_count(int device_id) {
    return static_cast<std::size_t>(std::count_if(
        g_csr_registry.entries.begin(),
        g_csr_registry.entries.end(),
        [device_id](const auto& entry) { return entry->device_id == device_id; }
    ));
}

static bool evict_lru(int device_id, const SharedDeviceCsrCache* protected_entry = nullptr) {
    std::size_t victim = g_csr_registry.entries.size();
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < g_csr_registry.entries.size(); ++i) {
        const auto& entry = g_csr_registry.entries[i];
        if (entry->device_id != device_id || entry.get() == protected_entry) continue;
        if (entry->last_use < oldest) {
            oldest = entry->last_use;
            victim = i;
        }
    }
    if (victim == g_csr_registry.entries.size()) return false;
    g_csr_registry.entries.erase(g_csr_registry.entries.begin() + victim);
    ++g_csr_registry.counters.evictions;
    return true;
}

static SharedDeviceCsrCache* recycle_lru(int device_id) {
    SharedDeviceCsrCache* victim = nullptr;
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (auto& entry : g_csr_registry.entries) {
        if (entry->device_id != device_id) continue;
        if (entry->last_use < oldest) {
            oldest = entry->last_use;
            victim = entry.get();
        }
    }
    if (victim == nullptr) return nullptr;
    victim->invalidate();
    ++g_csr_registry.counters.evictions;
    return victim;
}

static void reserve_cache_budget(
    int device_id,
    std::size_t additional_bytes,
    const SharedDeviceCsrCache* protected_entry = nullptr
) {
    const std::size_t budget = cache_budget_bytes(device_id);
    while (
        cached_device_bytes(device_id) + additional_bytes > budget &&
        evict_lru(device_id, protected_entry)
    ) {}
}

static SharedDeviceCsrCache* create_cache_entry(int device_id, std::size_t required_bytes) {
    if (entry_count(device_id) >= max_entries_per_device()) {
        // Re-purpose the least-recently-used slot without releasing its CUDA
        // allocations. This preserves the original single-slot behavior and
        // avoids allocator synchronization whenever two graph views alternate.
        SharedDeviceCsrCache* recycled = recycle_lru(device_id);
        if (recycled != nullptr) return recycled;
    }
    reserve_cache_budget(device_id, required_bytes);
    auto entry = std::make_unique<SharedDeviceCsrCache>(device_id);
    SharedDeviceCsrCache* ptr = entry.get();
    g_csr_registry.entries.emplace_back(std::move(entry));
    return ptr;
}

static SharedDeviceCsrCache* find_cache_entry(
    int device_id,
    std::uint64_t graph_id,
    const int* V,
    const int* E,
    int len_V,
    int len_E
) {
    for (auto& entry : g_csr_registry.entries) {
        if (
            entry->device_id == device_id &&
            graph_signature_matches(entry->graph_sig, graph_id, V, E, len_V, len_E)
        ) return entry.get();
    }
    return nullptr;
}

static SharedDeviceCsrCache* scratch_entry(int device_id) {
    for (auto& entry : g_csr_registry.entries) {
        if (entry->device_id == device_id) return entry.get();
    }
    return create_cache_entry(device_id, 0);
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

    int device_id = -1;
    if (cudaGetDevice(&device_id) != cudaSuccess || device_id < 0) return EG_GPU_DEVICE_ERR;

    out->d_V = nullptr;
    out->d_E = nullptr;
    out->d_W = nullptr;
    out->stats = HostCsrStats();

    const bool disable_cache = device_csr_cache_disabled();
    SharedDeviceCsrCache* cache = disable_cache
        ? scratch_entry(device_id)
        : find_cache_entry(device_id, g_active_graph_id, V, E, len_V, len_E);
    const bool structure_hit = !disable_cache && cache != nullptr;
    if (structure_hit) {
        ++g_csr_registry.counters.structure_hits;
    } else {
        ++g_csr_registry.counters.structure_misses;
        if (!disable_cache) {
            const std::size_t required_bytes =
                sizeof(int) * (static_cast<std::size_t>(len_V) + 1) +
                sizeof(int) * static_cast<std::size_t>(len_E) +
                (require_weights ? sizeof(double) * static_cast<std::size_t>(len_E) : 0);
            cache = create_cache_entry(device_id, required_bytes);
        }
    }
    if (cache == nullptr) return EG_GPU_DEVICE_ERR;
    cache->last_use = ++g_csr_registry.logical_clock;

    out->structure_changed = !structure_hit;
    const bool weight_hit = (
        require_weights && structure_hit &&
        weight_signature_matches(cache->weight_sig, W, len_E)
    );
    out->weights_changed = require_weights && !weight_hit;
    if (require_weights) {
        if (weight_hit) ++g_csr_registry.counters.weight_hits;
        else ++g_csr_registry.counters.weight_misses;
    }

    const std::size_t v_bytes = sizeof(int) * (static_cast<std::size_t>(len_V) + 1);
    const std::size_t e_bytes = sizeof(int) * static_cast<std::size_t>(len_E);
    const std::size_t w_bytes = sizeof(double) * static_cast<std::size_t>(len_E);
    const std::size_t additional_bytes =
        (v_bytes > cache->d_V.capacity_bytes() ? v_bytes - cache->d_V.capacity_bytes() : 0) +
        (e_bytes > cache->d_E.capacity_bytes() ? e_bytes - cache->d_E.capacity_bytes() : 0) +
        (require_weights && w_bytes > cache->d_W.capacity_bytes()
            ? w_bytes - cache->d_W.capacity_bytes() : 0);
    // An exact hit with sufficient capacity performs no allocation.  Existing
    // buffers therefore certify this hot path without consulting global free
    // memory, even when the registry retains several graph views.  Admission
    // and capacity growth still enforce the budget before allocating.
    if (additional_bytes > 0) {
        reserve_cache_budget(device_id, additional_bytes, cache);
    }

    int rc = cache->d_V.ensure_bytes(v_bytes);
    if (rc != EG_GPU_SUCC) return rc;
    rc = cache->d_E.ensure_bytes(e_bytes);
    if (rc != EG_GPU_SUCC) return rc;
    if (require_weights) {
        if (W == nullptr && len_E > 0) return EG_GPU_DEVICE_ERR;
        rc = cache->d_W.ensure_bytes(w_bytes);
        if (rc != EG_GPU_SUCC) return rc;
    }

    int* d_V = cache->d_V.as<int>();
    int* d_E = cache->d_E.as<int>();
    double* d_W = require_weights ? cache->d_W.as<double>() : nullptr;

    if (out->structure_changed) {
        HostCsrStats new_stats = adaptive_policy_enabled()
            ? summarize_host_csr(V, len_V, len_E)
            : HostCsrStats();
        bool reg_V = false;
        bool reg_E = false;
        const int* h_V = prepare_h2d_source(
            V, static_cast<std::size_t>(len_V) + 1, g_csr_registry.h_V_stage, &reg_V
        );
        const int* h_E = prepare_h2d_source(
            E, static_cast<std::size_t>(len_E), g_csr_registry.h_E_stage, &reg_E
        );
        if (h_V == nullptr) h_V = V;
        if (h_E == nullptr) h_E = E;
        cudaError_t ret = cudaMemcpy(d_V, h_V, v_bytes, cudaMemcpyHostToDevice);
        release_h2d_source(h_V, reg_V);
        if (ret != cudaSuccess) {
            return (ret == cudaErrorMemoryAllocation)
                ? EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM : EG_GPU_DEVICE_ERR;
        }
        ret = cudaMemcpy(d_E, h_E, e_bytes, cudaMemcpyHostToDevice);
        release_h2d_source(h_E, reg_E);
        if (ret != cudaSuccess) {
            return (ret == cudaErrorMemoryAllocation)
                ? EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM : EG_GPU_DEVICE_ERR;
        }
        g_csr_registry.counters.structure_bytes_copied += v_bytes + e_bytes;
        if (!disable_cache) {
            update_graph_signature(
                cache->graph_sig, g_active_graph_id, V, E, len_V, len_E
            );
            cache->stats = new_stats;
        }
        out->stats = new_stats;
    } else {
        out->stats = cache->stats;
    }

    if (require_weights && out->weights_changed) {
        bool reg_W = false;
        const double* h_W = prepare_h2d_source(
            W, static_cast<std::size_t>(len_E), g_csr_registry.h_W_stage, &reg_W
        );
        if (h_W == nullptr) h_W = W;
        const cudaError_t ret = cudaMemcpy(d_W, h_W, w_bytes, cudaMemcpyHostToDevice);
        release_h2d_source(h_W, reg_W);
        if (ret != cudaSuccess) {
            return (ret == cudaErrorMemoryAllocation)
                ? EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM : EG_GPU_DEVICE_ERR;
        }
        g_csr_registry.counters.weight_bytes_copied += w_bytes;
        if (!disable_cache) update_weight_signature(cache->weight_sig, W, len_E);
    }

    out->d_V = d_V;
    out->d_E = d_E;
    out->d_W = d_W;
    if (!out->stats.ready && adaptive_policy_enabled()) {
        out->stats = summarize_host_csr(V, len_V, len_E);
        if (!disable_cache) cache->stats = out->stats;
    }
    return EG_GPU_SUCC;
}

void reset_device_csr_cache() {
    g_csr_registry.entries.clear();
    g_csr_registry.h_V_stage.reset();
    g_csr_registry.h_E_stage.reset();
    g_csr_registry.h_W_stage.reset();
    g_csr_registry.counters = DeviceCsrCacheStats();
    g_csr_registry.logical_clock = 0;
}

DeviceCsrCacheScope::DeviceCsrCacheScope(std::uint64_t graph_id)
    : previous_graph_id_(g_active_graph_id) {
    g_active_graph_id = graph_id;
}

DeviceCsrCacheScope::~DeviceCsrCacheScope() {
    g_active_graph_id = previous_graph_id_;
}

std::uint64_t active_device_graph_id() {
    return g_active_graph_id;
}

DeviceCsrCacheStats get_device_csr_cache_stats() {
    DeviceCsrCacheStats stats = g_csr_registry.counters;
    stats.active_entries = g_csr_registry.entries.size();
    stats.device_bytes = 0;
    for (const auto& entry : g_csr_registry.entries) {
        stats.device_bytes += entry->device_bytes();
    }
    return stats;
}

} // namespace gpu_easygraph
