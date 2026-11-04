#pragma once
#include "managed_state_inventory.h"

#include "partitioned_device_state.h"
#include "numa_pinned_host.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

struct WccPartitionUpload {
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
    std::vector<std::uint64_t> local_to_global;
    std::vector<int> owners;
    bool forest_contraction;
    bool boundary_csr;
    std::size_t boundary_edge_count;
    std::vector<int> boundary_sources;
    std::vector<int> boundary_neighbors;
    std::string offsets_sha256;
    std::string indices_sha256;
    std::string local_to_global_sha256;
    std::string owners_sha256;
    std::string topology_compatibility;
    std::string ownership_compatibility;
    std::vector<std::vector<int> > outgoing_owner_local_ids;
    std::vector<std::vector<int> > incoming_halo_local_ids;

    WccPartitionUpload();
};

struct WccPartitionDeviceView {
    int partition_id;
    int device_id;
    std::size_t owned_vertex_count;
    std::size_t local_vertex_count;
    std::size_t adjacency_count;
    const int* offsets;
    const int* indices;
    const std::uint64_t* local_to_global;
    const int* owners;
    bool forest_contraction;
    bool boundary_csr;
    std::size_t boundary_edge_count;
    const int* boundary_sources;
    const int* boundary_neighbors;
    std::vector<const int*> outgoing_owner_local_ids;
    std::vector<const int*> incoming_halo_local_ids;
    std::vector<std::size_t> outgoing_counts;
    std::vector<std::size_t> incoming_counts;
};

class ResidentWccPartitionView {
public:
    static std::shared_ptr<ResidentWccPartitionView> upload(
        const WccPartitionUpload& upload);

    const StateKey& state_key() const;
    const std::string& graph_identity_text() const { return graph_identity_text_; }
    const std::string& view_signature_text() const;
    const WccPartitionDeviceView& view() const;
    std::size_t resident_bytes() const;
    const SharedArrayCallStats& shared_array_stats() const;
    std::size_t shared_array_count() const;
    bool cacheable() const;
    const std::string& managed_owner_id() const { return managed_owner_id_; }
    void append_managed_allocations(ManagedStateInventory& inventory,
                                    const ManagedOwnerRecord& owner) const;

private:
    explicit ResidentWccPartitionView(const WccPartitionUpload& upload);
    void initialize(const WccPartitionUpload& upload);

    std::string managed_owner_id_;
    StateKey state_key_;
    std::string graph_identity_text_;
    std::string view_signature_text_;
    WccPartitionDeviceView view_;
    std::shared_ptr<SharedDeviceArray> offsets_;
    std::shared_ptr<SharedDeviceArray> indices_;
    std::shared_ptr<SharedDeviceArray> local_to_global_;
    std::shared_ptr<SharedDeviceArray> owners_;
    std::shared_ptr<DeviceAllocation> boundary_sources_;
    std::shared_ptr<DeviceAllocation> boundary_neighbors_;
    std::vector<std::shared_ptr<DeviceAllocation> > outgoing_indices_;
    std::vector<std::shared_ptr<DeviceAllocation> > incoming_indices_;
    std::size_t resident_bytes_;
    SharedArrayCallStats shared_array_stats_;
    std::size_t shared_array_count_;
};

class WccPartitionWorkspace {
public:
    explicit WccPartitionWorkspace(int device_id);

    void ensure(const WccPartitionDeviceView& view);
    std::uint64_t* labels() const;
    int* changed() const;
    int* local_parents() const;
    std::uint64_t* send_buffer(std::size_t destination_rank) const;
    std::uint64_t* receive_buffer(std::size_t source_rank) const;
    std::size_t capacity_bytes() const;
    const std::string& managed_owner_id() const { return managed_owner_id_; }
    void append_managed_allocations(ManagedStateInventory& inventory,
                                    const ManagedOwnerRecord& owner) const;
    bool ensure_host_output(std::size_t count, const std::string& placement);
    std::shared_ptr<PinnedHostBuffer> host_output() const;
    const GpuLocalPinnedAllocationInfo& host_output_info() const;

private:
    std::string managed_owner_id_;
    int device_id_;
    std::shared_ptr<DeviceAllocation> labels_;
    std::shared_ptr<DeviceAllocation> changed_;
    std::shared_ptr<DeviceAllocation> local_parents_;
    std::vector<std::shared_ptr<DeviceAllocation> > send_buffers_;
    std::vector<std::shared_ptr<DeviceAllocation> > receive_buffers_;
    std::shared_ptr<PinnedHostBuffer> host_output_;
    std::string host_output_placement_;
    GpuLocalPinnedAllocationInfo host_output_info_;
};

class PartitionedWccStateRegistry {
public:
    std::shared_ptr<ResidentWccPartitionView> acquire(
        const WccPartitionUpload& upload,
        bool* cache_hit = nullptr,
        SharedArrayCallStats* shared_stats = nullptr);
    std::shared_ptr<WccPartitionWorkspace> workspace(
        const std::shared_ptr<ResidentWccPartitionView>& view);
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
        std::shared_ptr<ResidentWccPartitionView>,
        RegistryKeyHash> views_;
    std::unordered_map<
        RegistryKey,
        std::shared_ptr<WccPartitionWorkspace>,
        RegistryKeyHash> workspaces_;
    std::shared_ptr<WccPartitionWorkspace> pooled_workspace_;
    std::uint64_t managed_pool_reuses_ = 0;
    std::uint64_t managed_pool_fallbacks_ = 0;
    std::deque<RegistryKey> lru_;
    void touch_and_trim(const RegistryKey& key);
};

PartitionedWccStateRegistry& partitioned_wcc_state_registry();

}  // namespace runtime
}  // namespace gpu_easygraph
