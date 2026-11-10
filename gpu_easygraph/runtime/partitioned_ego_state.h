#pragma once

#include "partitioned_device_state.h"
#include "partitioned_ego_plan.h"
#include "runtime_types.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

struct EgoPartitionDeviceUpload {
    int partition_id;
    int device_id;
    std::uint64_t graph_identity;
    std::uint64_t generation;
    std::uint64_t ego_view_signature;
    std::vector<int> device_group;
    std::string semantic_key;
    const EgoHostPartition* host;

    EgoPartitionDeviceUpload();
};

struct EgoPartitionDeviceView {
    int partition_id;
    int device_id;
    std::size_t owned_vertex_count;
    std::size_t incidence_count;
    const int* offsets;
    const std::uint8_t* mutual;
    const std::uint8_t* neighbor_max_scale;
    const std::uint64_t* common_product;
    const double* indirect;
    const int* degree;
    const int* sum_scale;
};

class ResidentEgoPartitionState {
public:
    static std::shared_ptr<ResidentEgoPartitionState> upload(
        const EgoPartitionDeviceUpload& upload);

    const StateKey& state_key() const;
    const std::string& semantic_key() const;
    const EgoPartitionDeviceView& view() const;
    std::size_t resident_bytes() const;

private:
    explicit ResidentEgoPartitionState(const EgoPartitionDeviceUpload& upload);
    void initialize(const EgoPartitionDeviceUpload& upload);

    StateKey state_key_;
    std::string semantic_key_;
    EgoPartitionDeviceView view_;
    std::shared_ptr<DeviceAllocation> offsets_;
    std::shared_ptr<DeviceAllocation> mutual_;
    std::shared_ptr<DeviceAllocation> neighbor_max_scale_;
    std::shared_ptr<DeviceAllocation> common_product_;
    std::shared_ptr<DeviceAllocation> indirect_;
    std::shared_ptr<DeviceAllocation> degree_;
    std::shared_ptr<DeviceAllocation> sum_scale_;
    std::size_t resident_bytes_;
};

// Mutable call buffers are deliberately separate from the immutable ego
// state.  A fixed worker executes at most one task at a time, so capacity can
// be retained safely without exposing it to another CUDA device.
class EgoPartitionWorkspace {
public:
    explicit EgoPartitionWorkspace(int device_id);
    void ensure(std::size_t selected_count);
    // Batched reductions retain one selection and independently grow outputs.
    void ensure(std::size_t selected_count, std::size_t result_count);
    int* selected_nodes() const;
    double* result() const;
    std::size_t capacity_bytes() const;

private:
    int device_id_;
    std::size_t selected_capacity_;
    std::size_t result_capacity_;
    std::shared_ptr<DeviceAllocation> selected_nodes_;
    std::shared_ptr<DeviceAllocation> result_;
};

class PartitionedEgoStateRegistry {
public:
    std::shared_ptr<ResidentEgoPartitionState> acquire(
        const EgoPartitionDeviceUpload& upload,
        bool* cache_hit = nullptr);
    std::shared_ptr<EgoPartitionWorkspace> workspace(
        const std::shared_ptr<ResidentEgoPartitionState>& state);
    void clear();
    std::size_t resident_state_count() const;
    std::size_t workspace_count() const;

private:
    struct RegistryKey {
        StateKey state;
        int partition_id;

        RegistryKey(const StateKey& state_value, int partition_value);
        bool operator==(const RegistryKey& other) const;
    };
    struct RegistryKeyHash {
        std::size_t operator()(const RegistryKey& key) const;
    };

    void touch_and_trim(const RegistryKey& key);

    std::unordered_map<RegistryKey,
        std::shared_ptr<ResidentEgoPartitionState>, RegistryKeyHash> states_;
    std::unordered_map<RegistryKey,
        std::shared_ptr<EgoPartitionWorkspace>, RegistryKeyHash> workspaces_;
    std::deque<RegistryKey> lru_;
};

PartitionedEgoStateRegistry& partitioned_ego_state_registry();

}  // namespace runtime
}  // namespace gpu_easygraph
