#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <cstdint>

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

// Standalone equivalent of collective_transport.cpp; no native EGMGPU linkage.
class PinnedHostBuffer {
 public:
  explicit PinnedHostBuffer(std::size_t bytes) : data_(nullptr), size_(bytes) {
    if (bytes) {
      const auto status = cudaHostAlloc(&data_, bytes, cudaHostAllocPortable);
      if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaHostAlloc staging buffer: ") + cudaGetErrorString(status));
    }
  }
  ~PinnedHostBuffer() { if (data_) cudaFreeHost(data_); }
  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
  void* data() { return data_; }
  const void* data() const { return data_; }
  std::size_t size() const { return size_; }
 private:
  void* data_;
  std::size_t size_;
};

inline std::string json_quoted(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
    else out << c;
  }
  out << '"';
  return out.str();
}

// Locate the exact virtual-memory area containing the D2H destination and
// retain its corresponding numa_maps line, including observed N<node>=pages.
inline std::string pinned_numa_mapping(const void* pointer) {
  if (!pointer) return "empty allocation";
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  std::ifstream maps("/proc/self/maps");
  std::string line, base;
  while (std::getline(maps, line)) {
    const auto dash = line.find('-');
    const auto space = line.find(' ');
    if (dash == std::string::npos || space == std::string::npos) continue;
    const auto first = std::stoull(line.substr(0, dash), nullptr, 16);
    const auto last = std::stoull(line.substr(dash + 1, space - dash - 1), nullptr, 16);
    if (address >= first && address < last) { base = line.substr(0, dash); break; }
  }
  if (base.empty()) throw std::runtime_error("pinned address missing from /proc/self/maps");
  std::ifstream numa("/proc/self/numa_maps");
  while (std::getline(numa, line))
    if (line.compare(0, base.size() + 1, base + " ") == 0) return line;
  throw std::runtime_error("pinned mapping missing from /proc/self/numa_maps");
}

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
    char raw_pci[32] = {};
    const cudaError_t pci_status = cudaDeviceGetPCIBusId(raw_pci, sizeof(raw_pci), logical_device);
    if (pci_status != cudaSuccess) {
        throw std::runtime_error(std::string("GPU-local pinned PCI query: ") + cudaGetErrorString(pci_status));
    }
    std::string pci(raw_pci);
    for (char& character : pci) character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    const auto colon = pci.find(':');
    if (colon == std::string::npos) throw std::runtime_error("invalid CUDA PCI bus ID");
    std::ostringstream normalized;
    normalized << std::hex << std::setfill('0') << std::setw(4)
               << std::stoul(pci.substr(0, colon), nullptr, 16) << pci.substr(colon);
    pci = normalized.str();
    int node = -1;
    std::ifstream node_file("/sys/bus/pci/devices/" + pci + "/numa_node");
    if (!(node_file >> node) || node < 0) {
        throw std::runtime_error("GPU-local pinned allocation requires a known PCI NUMA node: " + pci);
    }
    std::string cpu_list;
    std::ifstream cpu_file("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    if (!(cpu_file >> cpu_list)) throw std::runtime_error("cannot read target NUMA node cpulist");
    cpu_set_t saved, target;
    if (sched_getaffinity(0, sizeof(saved), &saved) != 0) throw pinned_affinity_error("read allocation-thread CPU affinity");
    CPU_ZERO(&target);
    GpuLocalPinnedAllocationInfo info;
    info.logical_device = logical_device; info.numa_node = node; info.pci_bus_id = pci;
    std::istringstream ranges(cpu_list);
    std::string range;
    while (std::getline(ranges, range, ',')) {
        const auto dash = range.find('-');
        const int first = std::stoi(range.substr(0, dash));
        const int last = dash == std::string::npos ? first : std::stoi(range.substr(dash + 1));
        for (int cpu = first; cpu <= last; ++cpu) {
            if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &saved)) {
                CPU_SET(cpu, &target); info.allowed_node_cpus.push_back(cpu);
            }
        }
    }
    if (info.allowed_node_cpus.empty()) {
        throw std::runtime_error("GPU-local pinned allocation has no allowed CPU on NUMA node " + std::to_string(node));
    }
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
