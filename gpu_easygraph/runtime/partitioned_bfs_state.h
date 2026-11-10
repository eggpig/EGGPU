#pragma once
#include "managed_state_inventory.h"

#include "partitioned_device_state.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

struct BfsPartitionUpload {
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
    std::vector<int> owners;
    std::string offsets_sha256;
    std::string indices_sha256;
    std::string owners_sha256;
    std::string topology_compatibility;
    std::string ownership_compatibility;
    // local ID -> flat source-route slot; owned IDs use -1.
    std::vector<int> halo_route_slots;
    // Flat route metadata, grouped by destination rank.
    std::vector<int> route_destination_ranks;
    std::vector<int> route_owner_local_ids;
    std::vector<std::size_t> outgoing_route_offsets;
    // Receive storage is grouped by source rank.
    std::vector<std::size_t> incoming_route_offsets;
    // Bitmap-word layouts for dense owner-directed exchange.
    std::vector<std::size_t> outgoing_dense_word_offsets;
    std::vector<std::size_t> incoming_dense_word_offsets;

    BfsPartitionUpload();
};

struct BfsPartitionDeviceView {
    int partition_id;
    int device_id;
    std::size_t owned_vertex_count;
    std::size_t local_vertex_count;
    std::size_t adjacency_count;
    const int* offsets;
    const int* indices;
    const int* owners;
    const int* halo_route_slots;
    const int* route_destination_ranks;
    const int* route_owner_local_ids;
    const std::size_t* outgoing_dense_word_offsets_device;
    const std::size_t* outgoing_route_offsets_device;
    std::vector<std::size_t> outgoing_route_offsets;
    std::vector<std::size_t> incoming_route_offsets;
    std::vector<std::size_t> outgoing_dense_word_offsets;
    std::vector<std::size_t> incoming_dense_word_offsets;
};

class ResidentBfsPartitionView {
public:
    static std::shared_ptr<ResidentBfsPartitionView> upload(
        const BfsPartitionUpload& upload);

    const StateKey& state_key() const;
    const std::string& graph_identity_text() const;
    const std::string& view_signature_text() const;
    const BfsPartitionDeviceView& view() const;
    std::size_t resident_bytes() const;
    const SharedArrayCallStats& shared_array_stats() const;
    std::size_t shared_array_count() const;
    bool cacheable() const;
    const std::string& managed_owner_id() const { return managed_owner_id_; }
    void append_managed_allocations(ManagedStateInventory& inventory,
                                    const ManagedOwnerRecord& owner) const;

private:
    explicit ResidentBfsPartitionView(const BfsPartitionUpload& upload);
    void initialize(const BfsPartitionUpload& upload);

    std::string managed_owner_id_;
    StateKey state_key_;
    std::string graph_identity_text_;
    std::string view_signature_text_;
    BfsPartitionDeviceView view_;
    std::shared_ptr<SharedDeviceArray> offsets_;
    std::shared_ptr<SharedDeviceArray> indices_;
    std::shared_ptr<SharedDeviceArray> owners_;
    std::shared_ptr<DeviceAllocation> halo_route_slots_;
    std::shared_ptr<DeviceAllocation> route_destination_ranks_;
    std::shared_ptr<DeviceAllocation> route_owner_local_ids_;
    std::shared_ptr<DeviceAllocation> outgoing_dense_word_offsets_device_;
    std::shared_ptr<DeviceAllocation> outgoing_route_offsets_device_;
    std::size_t resident_bytes_;
    SharedArrayCallStats shared_array_stats_;
    std::size_t shared_array_count_;
};

class BfsPartitionWorkspace {
public:
    explicit BfsPartitionWorkspace(int device_id);

    void ensure(const BfsPartitionDeviceView& view);
    void ensure_high_degree(const BfsPartitionDeviceView& view);
    int* high_degree_rows() const;
    int* high_degree_control() const;
    int high_degree_grid_limit() const;
    std::size_t high_degree_capacity_bytes() const;
    std::int64_t* distances() const;
    int* current_frontier() const;
    int* next_frontier() const;
    int* next_count() const;
    int* send_flags() const;
    int* send_values(std::size_t destination_rank) const;
    int* send_counts() const;
    int* receive_values(std::size_t source_rank) const;
    std::uint32_t* current_bitmap() const;
    std::uint32_t* next_bitmap() const;
    std::uint32_t* dense_send_bitmap(std::size_t destination_rank) const;
    std::uint32_t* dense_receive_bitmap(std::size_t source_rank) const;
    std::size_t dense_send_word_count(std::size_t destination_rank) const;
    std::size_t dense_receive_word_count(std::size_t source_rank) const;
    void advance_frontier();
    std::size_t capacity_bytes() const;
    const std::string& managed_owner_id() const { return managed_owner_id_; }
    void append_managed_allocations(ManagedStateInventory& inventory,
                                    const ManagedOwnerRecord& owner) const;

private:
    std::string managed_owner_id_;
    int device_id_;
    std::vector<std::size_t> outgoing_route_offsets_;
    std::vector<std::size_t> incoming_route_offsets_;
    std::vector<std::size_t> outgoing_dense_word_offsets_;
    std::vector<std::size_t> incoming_dense_word_offsets_;
    std::shared_ptr<DeviceAllocation> distances_;
    std::shared_ptr<DeviceAllocation> current_frontier_;
    std::shared_ptr<DeviceAllocation> next_frontier_;
    std::shared_ptr<DeviceAllocation> next_count_;
    std::shared_ptr<DeviceAllocation> send_flags_;
    std::shared_ptr<DeviceAllocation> send_values_;
    std::shared_ptr<DeviceAllocation> send_counts_;
    std::shared_ptr<DeviceAllocation> receive_values_;
    std::shared_ptr<DeviceAllocation> current_bitmap_;
    std::shared_ptr<DeviceAllocation> next_bitmap_;
    std::shared_ptr<DeviceAllocation> dense_send_bitmap_;
    std::shared_ptr<DeviceAllocation> dense_receive_bitmap_;
    std::shared_ptr<DeviceAllocation> high_degree_rows_;
    std::shared_ptr<DeviceAllocation> high_degree_control_;
    int high_degree_grid_limit_ = 0;
};

class PartitionedBfsStateRegistry {
public:
    std::shared_ptr<ResidentBfsPartitionView> acquire(
        const BfsPartitionUpload& upload,
        bool* cache_hit = nullptr,
        SharedArrayCallStats* shared_stats = nullptr);
    std::shared_ptr<BfsPartitionWorkspace> workspace(
        const std::shared_ptr<ResidentBfsPartitionView>& view);
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
        std::shared_ptr<ResidentBfsPartitionView>,
        RegistryKeyHash> views_;
    std::unordered_map<
        RegistryKey,
        std::shared_ptr<BfsPartitionWorkspace>,
        RegistryKeyHash> workspaces_;
    std::shared_ptr<BfsPartitionWorkspace> pooled_workspace_;
    std::uint64_t managed_pool_reuses_ = 0;
    std::uint64_t managed_pool_fallbacks_ = 0;
    std::deque<RegistryKey> lru_;
    void touch_and_trim(const RegistryKey& key);
};

// Thread-local by construction: each fixed native worker owns one device and
// never accesses another worker's view or workspace registry.
PartitionedBfsStateRegistry& partitioned_bfs_state_registry();

}  // namespace runtime
}  // namespace gpu_easygraph
