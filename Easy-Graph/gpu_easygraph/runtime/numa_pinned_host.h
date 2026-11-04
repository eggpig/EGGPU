#pragma once

#include "collective_transport.h"
#include "gpu_numa_cpu_affinity.h"

#include <sched.h>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

struct GpuLocalPinnedAllocationInfo {
    int logical_device = -1;
    int numa_node = -1;
    int allocation_cpu = -1;
    std::string pci_bus_id;
    std::vector<int> allowed_node_cpus;
};

inline std::runtime_error pinned_affinity_error(const char* operation) {
    return std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

// This explicit allocator affects only the calling thread during allocation.
// Runtime workers, CUDA device selection and global NUMA policies are unchanged.
// The caller retains its ordinary PinnedHostBuffer ownership and lease rules.
inline std::shared_ptr<PinnedHostBuffer> make_gpu_local_pinned_host_buffer(
    std::size_t bytes, int logical_device,
    GpuLocalPinnedAllocationInfo* information = nullptr) {
    const GpuNumaCpuLocation location = discover_gpu_numa_cpu_location(logical_device);
    cpu_set_t saved;
    if (sched_getaffinity(0, sizeof(saved), &saved) != 0)
        throw pinned_affinity_error("read allocation-thread CPU affinity");
    const cpu_set_t target = gpu_numa_allowed_cpu_mask(location, saved);
    GpuLocalPinnedAllocationInfo info;
    info.logical_device = logical_device; info.numa_node = location.numa_node;
    info.pci_bus_id = location.pci_bus_id;
    info.allowed_node_cpus = gpu_cpu_mask_values(target);
    if (sched_setaffinity(0, sizeof(target), &target) != 0) throw pinned_affinity_error("bind allocation thread to GPU NUMA node");
    try {
        info.allocation_cpu = sched_getcpu();
        auto buffer = std::make_shared<PinnedHostBuffer>(bytes);
        // Establish physical pages while this thread is on the target node.
        // This one-time preparation cost remains in cold graph-state timing.
        if (bytes) std::memset(buffer->data(), 0, bytes);
        if (sched_setaffinity(0, sizeof(saved), &saved) != 0) throw pinned_affinity_error("restore allocation-thread CPU affinity");
        if (information) *information = std::move(info);
        return buffer;
    } catch (...) {
        // A failed allocation must not leave a changed calling-thread mask.
        const int saved_errno = errno;
        const int restored = sched_setaffinity(0, sizeof(saved), &saved);
        if (restored != 0) throw pinned_affinity_error("restore allocation-thread CPU affinity after allocation failure");
        errno = saved_errno;
        throw;
    }
}

}  // namespace runtime
}  // namespace gpu_easygraph
