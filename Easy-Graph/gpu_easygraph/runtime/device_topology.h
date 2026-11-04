#pragma once

#include "runtime_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

struct DeviceInfo {
    int device_id;
    std::string name;
    std::uint64_t total_global_memory;
    int compute_major;
    int compute_minor;
    int multiprocessor_count;
    int warp_size;
    std::string pci_bus_id;
    bool unified_addressing;
};

// DeviceTopology performs read-only CUDA runtime queries.  It deliberately
// does not enable peer access, create contexts, allocate memory, or change the
// caller's current CUDA device.
class DeviceTopology {
public:
    static DeviceTopology probe() noexcept;

    bool runtime_available() const;
    const std::string& diagnostic() const;
    const std::vector<DeviceInfo>& devices() const;
    const DeviceInfo& device(int device_id) const;
    bool has_device(int device_id) const;
    bool can_access_peer(int source_device, int target_device) const;
    bool is_fully_peer_connected(const DeviceGroup& group) const;
    bool nccl_compiled() const;

    CommunicationBackend resolve_backend(
        const DeviceGroup& group,
        CommunicationBackend requested) const;

private:
    bool runtime_available_;
    std::string diagnostic_;
    std::vector<DeviceInfo> devices_;
    std::vector<std::vector<bool> > peer_access_;
};

}  // namespace runtime
}  // namespace gpu_easygraph
