#pragma once
#include "managed_state_inventory.h"

#include "runtime_types.h"
#include "shared_device_array_catalog.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

// Per-worker diagnostics for one state-acquisition scope. GPU event spans
// bracket the existing pageable cudaMemcpy calls; they can include staging
// and host submission gaps and are not isolated DMA-engine busy time.
struct DeviceStateProfile {
    bool enabled = false;
    double malloc_wall_seconds = 0.0;
    double h2d_wall_seconds = 0.0;
    double h2d_device_seconds = 0.0;
    std::uint64_t malloc_bytes = 0;
    std::uint64_t malloc_count = 0;
    std::uint64_t h2d_bytes = 0;
    std::uint64_t h2d_count = 0;
};

// Begin/end must run on the same fixed CUDA worker. Empty/disabled scopes
// never create CUDA events or initialize a CUDA context.
void begin_device_state_profile(bool enabled);
DeviceStateProfile end_device_state_profile();
std::map<std::string,std::uint64_t> dynamic_weight_counters();

struct PartitionDeviceUpload {
    int partition_id;
    int device_id;
    std::uint64_t graph_identity;
    std::string graph_identity_text;
    std::uint64_t generation;
    std::vector<int> device_group;
    std::string view_signature;
    std::string orientation;
    std::size_t owned_vertex_count;
    // This upload/view type is used by PageRank. Other graph algorithms own
    // separate state types and retain their required metadata arrays.
    std::string device_metadata_policy;  // full (default) or pull_only
    std::size_t local_vertex_count;      // explicit when metadata is omitted
    std::vector<int> offsets;
    std::vector<int> indices;
    std::string offsets_sha256;
    std::string indices_sha256;
    std::string topology_compatibility;
    std::vector<double> weights;
    std::string weights_sha256;
    std::string weight_compatibility;
    std::vector<std::uint64_t> local_to_global;
    std::vector<int> owners;
    std::vector<double> owned_out_weight;
    std::vector<int> pagerank_heavy_rows;
    // Destination halo entries grouped by source owner. Immutable for this view.
    std::vector<int> halo_source_local;
    std::vector<int> halo_destination_local;
    std::vector<std::size_t> halo_owner_offsets;
    std::string local_to_global_sha256;
    std::string owners_sha256;
    std::string ownership_compatibility;

    PartitionDeviceUpload();
};

struct PartitionDeviceView {
    int partition_id;
    int device_id;
    std::size_t owned_vertex_count;
    std::size_t local_vertex_count;
    std::size_t adjacency_count;
    const int* offsets;
    const int* indices;
    const double* weights;
    const std::uint64_t* local_to_global;
    const int* owners;
    const double* owned_out_weight;
    const int* pagerank_heavy_rows;
    std::size_t pagerank_heavy_row_count;
    const int* halo_source_local;
    const int* halo_destination_local;
};

class DeviceAllocation {
public:
    DeviceAllocation(int device_id, std::size_t bytes);
    ~DeviceAllocation();

    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    void* data() const;
    std::size_t bytes() const;
    int device_id() const;
    std::uint64_t allocation_generation() const { return allocation_generation_; }

private:
    std::uint64_t allocation_generation_ = 0;
    int device_id_;
    void* pointer_;
    std::size_t bytes_;
};

// Internal upload of an immutable, already verified host array. The digest
// belongs to the exact source bytes: native manifest loading verifies imported
// arrays, and host-plan construction computes identities for transformed ones.
// The caller keeps the source alive until the synchronous upload returns.
struct ExactWeightPatch {
    std::string base_sha256;
    std::string target_sha256;
    std::vector<int> positions;
    std::vector<double> values;
};

std::shared_ptr<SharedDeviceArray> upload_verified_shared_device_array(
    int device_id,
    SharedArrayKind kind,
    const std::string& verified_sha256,
    const std::string& compatibility,
    const void* source,
    std::size_t bytes,
    const char* operation,
    SharedArrayAcquireResult* acquisition = nullptr,
    const ExactWeightPatch* native_patch = nullptr);

struct RetainedSparseHaloMaps {
    const int* source = nullptr;
    const int* packed = nullptr;
    bool cache_hit = false;
    std::size_t bytes = 0, new_bytes = 0, h2d_bytes = 0;
};

class ResidentPartitionView {
public:
    static std::shared_ptr<ResidentPartitionView> upload(
        const PartitionDeviceUpload& upload);
    ~ResidentPartitionView();

    ResidentPartitionView(const ResidentPartitionView&) = delete;
    ResidentPartitionView& operator=(const ResidentPartitionView&) = delete;

    const StateKey& state_key() const;
    const std::string& orientation() const;
    const std::string& graph_identity_text() const;
    const std::string& view_signature_text() const;
    const std::string& device_metadata_policy() const;
    const PartitionDeviceView& view() const;
    std::size_t resident_bytes() const;
    // Fixed-worker only. The caller must keep this immutable view leased until
    // all kernels finish. A different full snapshot/partition plan is rejected.
    RetainedSparseHaloMaps ensure_sparse_halo_maps(
        const std::string& plan_key, const std::vector<int>& source,
        const std::vector<int>& packed, cudaStream_t stream);
    const SharedArrayCallStats& shared_array_stats() const;
    std::size_t shared_array_count() const;
    bool cacheable() const;
    const std::string& managed_owner_id() const { return managed_owner_id_; }
    void append_managed_allocations(ManagedStateInventory& inventory,
                                    const ManagedOwnerRecord& owner) const;

private:
    explicit ResidentPartitionView(const PartitionDeviceUpload& upload);
    void initialize(const PartitionDeviceUpload& upload);

    std::string managed_owner_id_;
    StateKey state_key_;
    std::string orientation_;
    std::string graph_identity_text_;
    std::string view_signature_text_;
    std::string device_metadata_policy_;
    PartitionDeviceView view_;
    std::shared_ptr<DeviceAllocation> offsets_;
    std::shared_ptr<DeviceAllocation> indices_;
    std::shared_ptr<SharedDeviceArray> shared_offsets_;
    std::shared_ptr<SharedDeviceArray> shared_indices_;
    std::shared_ptr<SharedDeviceArray> weights_;
    std::shared_ptr<SharedDeviceArray> local_to_global_;
    std::shared_ptr<SharedDeviceArray> owners_;
    std::shared_ptr<DeviceAllocation> owned_out_weight_;
    std::shared_ptr<DeviceAllocation> pagerank_heavy_rows_;
    std::shared_ptr<DeviceAllocation> halo_source_local_;
    std::shared_ptr<DeviceAllocation> halo_destination_local_;
    std::size_t resident_bytes_;
    std::string sparse_halo_plan_key_;
    std::size_t sparse_halo_count_ = 0;
    std::shared_ptr<DeviceAllocation> sparse_halo_source_, sparse_halo_packed_;
    SharedArrayCallStats shared_array_stats_;
    std::size_t shared_array_count_;
};

class PartitionWorkspace {
public:
    explicit PartitionWorkspace(int device_id);

    void ensure(std::size_t local_vertex_count, std::size_t owned_vertex_count);
    int device_id() const;
    double* current_rank() const;
    double* next_rank() const;
    double* local_contribution() const;
    double* scalars() const;
    void advance_rank();
    std::size_t capacity_bytes() const;
    // Mutable send values use this workspace's existing exclusive pool lease.
    std::size_t ensure_sparse_send_capacity(std::size_t count);
    double* sparse_send() const;
    std::size_t sparse_send_capacity_bytes() const;
    const std::string& managed_owner_id() const { return managed_owner_id_; }
    void append_managed_allocations(ManagedStateInventory& inventory,
                                    const ManagedOwnerRecord& owner) const;

private:
    std::string managed_owner_id_;
    int device_id_;
    std::shared_ptr<DeviceAllocation> current_rank_;
    std::shared_ptr<DeviceAllocation> next_rank_;
    std::shared_ptr<DeviceAllocation> local_contribution_;
    std::shared_ptr<DeviceAllocation> scalars_;
    std::shared_ptr<DeviceAllocation> sparse_send_;
};

class PartitionedDeviceStateRegistry {
public:
    // Fixed-worker, read-only lease of existing immutable views; never uploads.
    std::vector<std::shared_ptr<ResidentPartitionView> > hold_graph_views(
        const std::string& graph_id, std::uint64_t generation) const;
    std::shared_ptr<ResidentPartitionView> acquire(
        const PartitionDeviceUpload& upload,
        bool* cache_hit = nullptr,
        SharedArrayCallStats* shared_stats = nullptr);
    std::shared_ptr<PartitionWorkspace> workspace(
        const std::shared_ptr<ResidentPartitionView>& view);
    bool is_pooled_workspace(const std::shared_ptr<PartitionWorkspace>& value) const { return value==pooled_workspace_; }
    void evict(const StateKey& key, int partition_id);
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
        std::shared_ptr<ResidentPartitionView>,
        RegistryKeyHash> views_;
    std::unordered_map<
        RegistryKey,
        std::shared_ptr<PartitionWorkspace>,
        RegistryKeyHash> workspaces_;
    std::shared_ptr<PartitionWorkspace> pooled_workspace_;
    std::uint64_t managed_pool_reuses_ = 0;
    std::uint64_t managed_pool_fallbacks_ = 0;
    std::deque<RegistryKey> lru_;
    void touch_and_trim(const RegistryKey& key);
};

// The registry is local to one fixed CUDA worker.  It retains views and
// workspaces across calls while ensuring allocation and destruction occur on
// the owning device thread.
PartitionedDeviceStateRegistry& partitioned_device_state_registry();

std::uint64_t stable_signature64(const std::string& value);

}  // namespace runtime
}  // namespace gpu_easygraph
