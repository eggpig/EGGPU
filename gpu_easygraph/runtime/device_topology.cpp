#include "device_topology.h"

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {
namespace {

std::string cuda_diagnostic(const char* operation, cudaError_t error) {
    std::ostringstream out;
    out << operation << " failed: " << cudaGetErrorName(error) << " ("
        << cudaGetErrorString(error) << ")";
    return out.str();
}

}  // namespace

DeviceTopology DeviceTopology::probe() noexcept {
    DeviceTopology topology;
    topology.runtime_available_ = false;

    int count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&count);
    if (count_status != cudaSuccess) {
        topology.diagnostic_ = cuda_diagnostic("cudaGetDeviceCount", count_status);
        return topology;
    }

    topology.runtime_available_ = true;
    topology.peer_access_.assign(
        static_cast<std::size_t>(count),
        std::vector<bool>(static_cast<std::size_t>(count), false));

    for (int device_id = 0; device_id < count; ++device_id) {
        cudaDeviceProp properties;
        const cudaError_t property_status =
            cudaGetDeviceProperties(&properties, device_id);
        if (property_status != cudaSuccess) {
            topology.runtime_available_ = false;
            topology.diagnostic_ =
                cuda_diagnostic("cudaGetDeviceProperties", property_status);
            topology.devices_.clear();
            topology.peer_access_.clear();
            return topology;
        }

        char pci_bus_id[32] = {0};
        const cudaError_t pci_status =
            cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_id);

        DeviceInfo info;
        info.device_id = device_id;
        info.name = properties.name;
        info.total_global_memory =
            static_cast<std::uint64_t>(properties.totalGlobalMem);
        info.compute_major = properties.major;
        info.compute_minor = properties.minor;
        info.multiprocessor_count = properties.multiProcessorCount;
        info.warp_size = properties.warpSize;
        info.pci_bus_id = pci_status == cudaSuccess ? pci_bus_id : "unavailable";
        info.unified_addressing = properties.unifiedAddressing != 0;
        topology.devices_.push_back(info);
    }

    for (int source = 0; source < count; ++source) {
        topology.peer_access_[static_cast<std::size_t>(source)]
                             [static_cast<std::size_t>(source)] = true;
        for (int target = 0; target < count; ++target) {
            if (source == target) {
                continue;
            }
            int accessible = 0;
            const cudaError_t peer_status =
                cudaDeviceCanAccessPeer(&accessible, source, target);
            if (peer_status == cudaSuccess) {
                topology.peer_access_[static_cast<std::size_t>(source)]
                                     [static_cast<std::size_t>(target)] =
                    accessible != 0;
            }
        }
    }

    if (count == 0) {
        topology.diagnostic_ = "CUDA runtime is available but reports no devices";
    }
    return topology;
}

bool DeviceTopology::runtime_available() const { return runtime_available_; }
const std::string& DeviceTopology::diagnostic() const { return diagnostic_; }
const std::vector<DeviceInfo>& DeviceTopology::devices() const { return devices_; }

bool DeviceTopology::has_device(int device_id) const {
    return device_id >= 0 &&
           static_cast<std::size_t>(device_id) < devices_.size() &&
           devices_[static_cast<std::size_t>(device_id)].device_id == device_id;
}

const DeviceInfo& DeviceTopology::device(int device_id) const {
    if (!has_device(device_id)) {
        throw std::out_of_range("CUDA device is not present in the probed topology");
    }
    return devices_[static_cast<std::size_t>(device_id)];
}

bool DeviceTopology::can_access_peer(int source_device, int target_device) const {
    if (!has_device(source_device) || !has_device(target_device)) {
        return false;
    }
    return peer_access_[static_cast<std::size_t>(source_device)]
                       [static_cast<std::size_t>(target_device)];
}

bool DeviceTopology::is_fully_peer_connected(const DeviceGroup& group) const {
    const std::vector<int>& ids = group.device_ids();
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (!has_device(ids[i])) {
            return false;
        }
        for (std::size_t j = 0; j < ids.size(); ++j) {
            if (i != j && !can_access_peer(ids[i], ids[j])) {
                return false;
            }
        }
    }
    return true;
}

bool DeviceTopology::nccl_compiled() const {
#ifdef EGMGPU_HAS_NCCL
    return true;
#else
    return false;
#endif
}

CommunicationBackend DeviceTopology::resolve_backend(
    const DeviceGroup& group,
    CommunicationBackend requested) const {
    if (group.size() == 1) {
        return CommunicationBackend::None;
    }
    if (requested == CommunicationBackend::None) {
        throw std::invalid_argument(
            "communication backend 'none' is invalid for a multi-device group");
    }
    if (requested == CommunicationBackend::Nccl) {
        if (!nccl_compiled()) {
            throw std::runtime_error(
                "NCCL was requested but this EGMGPU build has no NCCL support");
        }
        return requested;
    }
    if (requested == CommunicationBackend::PeerToPeer) {
        if (!is_fully_peer_connected(group)) {
            throw std::runtime_error(
                "peer-to-peer communication was requested for a device group "
                "without full directed P2P reachability");
        }
        return requested;
    }
    if (requested == CommunicationBackend::HostStaging) {
        return requested;
    }
    if (requested != CommunicationBackend::Auto) {
        throw std::invalid_argument("unsupported EGMGPU communication backend");
    }
    if (nccl_compiled()) {
        return CommunicationBackend::Nccl;
    }
    if (is_fully_peer_connected(group)) {
        return CommunicationBackend::PeerToPeer;
    }
    return CommunicationBackend::HostStaging;
}

}  // namespace runtime
}  // namespace gpu_easygraph
