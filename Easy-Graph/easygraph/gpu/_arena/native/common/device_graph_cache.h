#pragma once

#include <cstddef>
#include <cstdint>

#include "err.h"
#include "adaptive_transfer_policy.h"

namespace gpu_easygraph {

struct DeviceCsrView {
    int* d_V = nullptr;
    int* d_E = nullptr;
    double* d_W = nullptr;
    bool structure_changed = true;
    bool weights_changed = true;
    HostCsrStats stats;
};

struct DeviceCsrCacheStats {
    std::uint64_t structure_hits = 0;
    std::uint64_t structure_misses = 0;
    std::uint64_t weight_hits = 0;
    std::uint64_t weight_misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t structure_bytes_copied = 0;
    std::uint64_t weight_bytes_copied = 0;
    std::size_t active_entries = 0;
    std::size_t device_bytes = 0;
};

// Binds an immutable host CSR identity to all device-cache acquisitions made
// by one public graph-function call on the current thread.
class DeviceCsrCacheScope {
public:
    explicit DeviceCsrCacheScope(std::uint64_t graph_id);
    ~DeviceCsrCacheScope();

    DeviceCsrCacheScope(const DeviceCsrCacheScope&) = delete;
    DeviceCsrCacheScope& operator=(const DeviceCsrCacheScope&) = delete;

private:
    std::uint64_t previous_graph_id_;
};

// Returns the immutable host-CSR identity bound to the current public call.
// Specialized layouts use the same identity as the common CSR registry.
std::uint64_t active_device_graph_id();

int acquire_device_csr(
    const int* V,
    const int* E,
    const double* W,
    int len_V,
    int len_E,
    bool require_weights,
    DeviceCsrView* out
);

void reset_device_csr_cache();

DeviceCsrCacheStats get_device_csr_cache_stats();

} // namespace gpu_easygraph
