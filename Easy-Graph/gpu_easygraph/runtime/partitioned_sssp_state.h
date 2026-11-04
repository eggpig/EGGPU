#pragma once
#include "managed_state_inventory.h"

#include "partitioned_device_state.h"
#include "partitioned_sssp_types.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits>

namespace gpu_easygraph {
namespace runtime {

struct SsspPartitionUpload {
    int partition_id;
    int device_id;
    std::uint64_t graph_identity;
    std::string graph_identity_text;
    std::uint64_t generation;
    std::vector<int> device_group;
    std::string view_signature;
    std::size_t owned_vertex_count;
    std::vector<int> offsets;
    std::vector<int> indices;
    std::vector<double> weights;
    std::vector<std::uint8_t> weights_u8;
    SsspWeightEncoding weight_encoding = SsspWeightEncoding::F64;
    SsspWireEncoding wire_encoding = SsspWireEncoding::F64;
    double wire_distance_bound = std::numeric_limits<double>::infinity();
    std::vector<int> owners;
    std::string offsets_sha256;
    std::string indices_sha256;
    std::string weights_sha256;
    std::string owners_sha256;
    std::string topology_compatibility;
    std::string weight_compatibility;
    std::shared_ptr<const ExactWeightPatch> native_weight_patch;
    std::string ownership_compatibility;
    // local halo ID -> flat outgoing route slot; owned IDs use -1.
    std::vector<int> halo_route_slots;
    std::vector<int> route_destination_ranks;
    std::vector<int> route_owner_local_ids;
    std::vector<std::size_t> outgoing_route_offsets;
    std::vector<std::size_t> incoming_route_offsets;

    SsspPartitionUpload();
};

struct SsspPartitionDeviceView {
    int partition_id;
    int device_id;
    std::size_t owned_vertex_count;
    std::size_t local_vertex_count;
    std::size_t adjacency_count;
    const int* offsets;
    const int* indices;
    const void* weights;
    SsspWeightEncoding weight_encoding;
    SsspWireEncoding wire_encoding;
    double wire_distance_bound;
    std::size_t weight_bytes;
    const int* owners;
    const int* halo_route_slots;
    const int* route_destination_ranks;
    const int* route_owner_local_ids;
    std::vector<std::size_t> outgoing_route_offsets;
    std::vector<std::size_t> incoming_route_offsets;
};

class ResidentSsspPartitionView {
public:
    static std::shared_ptr<ResidentSsspPartitionView> upload(
        const SsspPartitionUpload& upload);

    const StateKey& state_key() const;
    const std::string& graph_identity_text() const;
    const std::string& view_signature_text() const;
    const SsspPartitionDeviceView& view() const;
    std::size_t resident_bytes() const;
    const SharedArrayCallStats& shared_array_stats() const;
    std::size_t shared_array_count() const;
    bool cacheable() const;
    const std::string& managed_owner_id() const { return managed_owner_id_; }
    void append_managed_allocations(ManagedStateInventory& inventory,
                                    const ManagedOwnerRecord& owner) const;

private:
    explicit ResidentSsspPartitionView(const SsspPartitionUpload& upload);
    void initialize(const SsspPartitionUpload& upload);

    std::string managed_owner_id_;
    StateKey state_key_;
    std::string graph_identity_text_;
    std::string view_signature_text_;
    SsspPartitionDeviceView view_;
    std::shared_ptr<SharedDeviceArray> offsets_;
    std::shared_ptr<SharedDeviceArray> indices_;
    std::shared_ptr<SharedDeviceArray> weights_;
    std::shared_ptr<SharedDeviceArray> owners_;
    std::shared_ptr<DeviceAllocation> halo_route_slots_;
    std::shared_ptr<DeviceAllocation> route_destination_ranks_;
    std::shared_ptr<DeviceAllocation> route_owner_local_ids_;
    std::size_t resident_bytes_;
    SharedArrayCallStats shared_array_stats_;
    std::size_t shared_array_count_;
};

class SsspPartitionWorkspace {
public:
    explicit SsspPartitionWorkspace(int device_id);

    void ensure(const SsspPartitionDeviceView& view);
    double* distances() const;
    void ensure_single_owner_scratch(std::size_t vertices);
    void* single_owner_scratch() const;
    std::size_t single_owner_scratch_capacity() const;
    int* settled() const;
    int* current_frontier() const;
    int* next_frontier() const;
    int* r_flags() const;
    int* next_flags() const;
    int* frontier_count() const;
    std::uint64_t* minimum_bucket() const;
    double* remote_candidates() const;
    void* send_values(std::size_t destination_rank) const;
    int* send_counts() const;
    void* receive_values(std::size_t source_rank) const;
    void advance_frontier();
    std::size_t capacity_bytes() const;
    const std::string& managed_owner_id() const { return managed_owner_id_; }
    void append_managed_allocations(ManagedStateInventory& inventory,
                                    const ManagedOwnerRecord& owner) const;

private:
    std::string managed_owner_id_;
    int device_id_;
    SsspWireEncoding wire_encoding_ = SsspWireEncoding::F64;
    std::vector<std::size_t> outgoing_route_offsets_;
    std::vector<std::size_t> incoming_route_offsets_;
    std::shared_ptr<DeviceAllocation> single_owner_scratch_;
    std::shared_ptr<DeviceAllocation> distances_;
    std::shared_ptr<DeviceAllocation> settled_;
    std::shared_ptr<DeviceAllocation> current_frontier_;
    std::shared_ptr<DeviceAllocation> next_frontier_;
    std::shared_ptr<DeviceAllocation> r_flags_;
    std::shared_ptr<DeviceAllocation> next_flags_;
    std::shared_ptr<DeviceAllocation> frontier_count_;
    std::shared_ptr<DeviceAllocation> minimum_bucket_;
    std::shared_ptr<DeviceAllocation> remote_candidates_;
    std::shared_ptr<DeviceAllocation> send_values_;
    std::shared_ptr<DeviceAllocation> send_counts_;
    std::shared_ptr<DeviceAllocation> receive_values_;
};

class PartitionedSsspStateRegistry {
public:
    std::shared_ptr<ResidentSsspPartitionView> acquire(
        const SsspPartitionUpload& upload,
        bool* cache_hit = nullptr,
        SharedArrayCallStats* shared_stats = nullptr);
    std::shared_ptr<SsspPartitionWorkspace> workspace(
        const std::shared_ptr<ResidentSsspPartitionView>& view);
    void clear();
    std::size_t resident_view_count() const;
    std::size_t workspace_count() const;
    void append_managed_inventory(ManagedStateInventory& inventory) const;
    void managed_evict(const std::set<std::string>& requested,
                       bool release_workspaces, ManagedEvictRecord& receipt);

private:
    struct RegistryKey {
        StateKey state;
        int partition_id;
        bool operator==(const RegistryKey& other) const;
    };
    struct RegistryKeyHash {
        std::size_t operator()(const RegistryKey& key) const;
    };
    std::unordered_map<
        RegistryKey,
        std::shared_ptr<ResidentSsspPartitionView>,
        RegistryKeyHash> views_;
    std::unordered_map<
        RegistryKey,
        std::shared_ptr<SsspPartitionWorkspace>,
        RegistryKeyHash> workspaces_;
    std::shared_ptr<SsspPartitionWorkspace> pooled_workspace_;
    std::uint64_t managed_pool_reuses_ = 0;
    std::uint64_t managed_pool_fallbacks_ = 0;
    std::deque<RegistryKey> lru_;
    void touch_and_trim(const RegistryKey& key);
};

PartitionedSsspStateRegistry& partitioned_sssp_state_registry();

}  // namespace runtime
}  // namespace gpu_easygraph
