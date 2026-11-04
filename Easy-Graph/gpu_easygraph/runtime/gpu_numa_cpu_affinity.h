#pragma once

#include <cuda_runtime_api.h>
#include <sched.h>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu_easygraph { namespace runtime {

struct GpuNumaCpuLocation {
    int logical_device = -1, numa_node = -1;
    std::string pci_bus_id;
    std::vector<int> node_cpus;
};

inline std::runtime_error gpu_cpu_affinity_error(const char* operation) {
    return std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

// Extracted from the existing GPU-local pinned allocator. This discovers a
// CPU location; it does not allocate pages or change any NUMA policy.
inline GpuNumaCpuLocation discover_gpu_numa_cpu_location(int logical_device) {
    char raw_pci[32] = {};
    const cudaError_t status = cudaDeviceGetPCIBusId(raw_pci, sizeof(raw_pci), logical_device);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("GPU NUMA PCI query: ") + cudaGetErrorString(status));
    std::string pci(raw_pci);
    for (char& character : pci) character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    const auto colon = pci.find(':');
    if (colon == std::string::npos) throw std::runtime_error("invalid CUDA PCI bus ID");
    std::ostringstream normalized;
    normalized << std::hex << std::setfill('0') << std::setw(4)
               << std::stoul(pci.substr(0, colon), nullptr, 16) << pci.substr(colon);
    GpuNumaCpuLocation result;
    result.logical_device = logical_device; result.pci_bus_id = normalized.str();
    std::ifstream node_file("/sys/bus/pci/devices/" + result.pci_bus_id + "/numa_node");
    if (!(node_file >> result.numa_node) || result.numa_node < 0)
        throw std::runtime_error("GPU CPU placement requires a known PCI NUMA node: " + result.pci_bus_id);
    std::string cpu_list;
    std::ifstream cpu_file("/sys/devices/system/node/node" + std::to_string(result.numa_node) + "/cpulist");
    if (!(cpu_file >> cpu_list)) throw std::runtime_error("cannot read target NUMA node cpulist");
    std::istringstream ranges(cpu_list);
    std::string range;
    while (std::getline(ranges, range, ',')) {
        const auto dash = range.find('-');
        const int first = std::stoi(range.substr(0, dash));
        const int last = dash == std::string::npos ? first : std::stoi(range.substr(dash + 1));
        for (int cpu = first; cpu <= last; ++cpu)
            if (cpu >= 0 && cpu < CPU_SETSIZE) result.node_cpus.push_back(cpu);
    }
    return result;
}

inline cpu_set_t gpu_numa_allowed_cpu_mask(const GpuNumaCpuLocation& location, const cpu_set_t& allowed) {
    cpu_set_t target; CPU_ZERO(&target);
    for (int cpu : location.node_cpus)
        if (CPU_ISSET(cpu, &allowed)) CPU_SET(cpu, &target);
    if (CPU_COUNT(&target) == 0)
        throw std::runtime_error("GPU CPU placement has no allowed CPU on NUMA node " + std::to_string(location.numa_node));
    return target;
}

inline std::vector<int> gpu_cpu_mask_values(const cpu_set_t& mask) {
    std::vector<int> result;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &mask)) result.push_back(cpu);
    return result;
}

struct GpuWorkerCpuPlacementInfo {
    bool observed = false;
    int logical_device = -1, numa_node = -1;
    int cpu_before = -1, cpu_during = -1, cpu_after = -1;
    std::string pci_bus_id;
    std::vector<int> mask_before, mask_during, mask_after;
};

// Only the current thread is bound. Restore on normal completion and on a
// callback exception; cooperative callers must invoke this inside their
// barrier-cancellation wrapper. No allocator, prefaulting, or page migration.
template <typename Function>
void run_with_gpu_numa_cpu_location(const GpuNumaCpuLocation* location, bool observe,
    GpuWorkerCpuPlacementInfo* information, Function&& function) {
    if (!location && !observe) { function(); return; }
    cpu_set_t saved;
    if (sched_getaffinity(0, sizeof(saved), &saved) != 0)
        throw gpu_cpu_affinity_error("read worker CPU affinity");
    GpuWorkerCpuPlacementInfo info;
    info.observed = true; info.cpu_before = sched_getcpu();
    info.mask_before = gpu_cpu_mask_values(saved);
    cpu_set_t target = saved;
    if (location) {
        target = gpu_numa_allowed_cpu_mask(*location, saved);
        info.logical_device = location->logical_device; info.numa_node = location->numa_node;
        info.pci_bus_id = location->pci_bus_id;
    }
    bool changed = false;
    const auto restore = [&]() {
        if (!changed) return;
        if (sched_setaffinity(0, sizeof(saved), &saved) != 0)
            throw gpu_cpu_affinity_error("restore worker CPU affinity");
        changed = false;
    };
    try {
        if (location) {
            if (sched_setaffinity(0, sizeof(target), &target) != 0)
                throw gpu_cpu_affinity_error("bind worker to GPU NUMA CPUs");
            changed = true;
        }
        cpu_set_t actual;
        if (sched_getaffinity(0, sizeof(actual), &actual) != 0)
            throw gpu_cpu_affinity_error("read active worker CPU affinity");
        info.cpu_during = sched_getcpu(); info.mask_during = gpu_cpu_mask_values(actual);
        function();
        restore();
        if (sched_getaffinity(0, sizeof(actual), &actual) != 0)
            throw gpu_cpu_affinity_error("read restored worker CPU affinity");
        info.cpu_after = sched_getcpu(); info.mask_after = gpu_cpu_mask_values(actual);
        *information = std::move(info);
    } catch (...) {
        restore();
        throw;
    }
}

template <typename Function>
void run_with_gpu_worker_cpu_placement(int logical_device, bool gpu_numa, bool observe,
    GpuWorkerCpuPlacementInfo* information, Function&& function) {
    if (!gpu_numa) {
        run_with_gpu_numa_cpu_location(nullptr, observe, information, std::forward<Function>(function));
        if (information->observed) information->logical_device = logical_device;
        return;
    }
    // Fixed runtime worker threads have a stable CUDA ordinal/PCI mapping.
    // Cache immutable discovery only; re-read the actual thread mask per call.
    thread_local std::map<int, GpuNumaCpuLocation> locations;
    const GpuNumaCpuLocation* location = nullptr;
    if (gpu_numa) {
        auto found = locations.find(logical_device);
        if (found == locations.end())
            found = locations.emplace(logical_device, discover_gpu_numa_cpu_location(logical_device)).first;
        location = &found->second;
    }
    run_with_gpu_numa_cpu_location(location, observe, information, std::forward<Function>(function));
}

} }  // namespace gpu_easygraph::runtime
