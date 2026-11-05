#include "partitioned_wcc_state.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorName(status) +
        " (" + cudaGetErrorString(status) + ")");
}

void check_current_device(int expected) {
    int current = -1;
    check_cuda(cudaGetDevice(&current), "cudaGetDevice for WCC state");
    if (current != expected) {
        throw std::runtime_error("WCC state must be used by its fixed worker");
    }
}

template <typename T>
std::shared_ptr<DeviceAllocation> upload_array(
    int device_id, const std::vector<T>& source, const char* operation) {
    if (source.empty()) return std::shared_ptr<DeviceAllocation>();
    const std::size_t bytes = source.size() * sizeof(T);
    std::shared_ptr<DeviceAllocation> result(
        new DeviceAllocation(device_id, bytes));
    check_cuda(
        cudaMemcpy(result->data(), &source[0], bytes, cudaMemcpyHostToDevice),
        operation);
    return result;
}

std::size_t bytes(const std::shared_ptr<DeviceAllocation>& allocation) {
    return allocation ? allocation->bytes() : 0U;
}

std::size_t bytes(const std::shared_ptr<SharedDeviceArray>& allocation) {
    return allocation ? allocation->bytes() : 0U;
}

void ensure_allocation(
    int device_id,
    std::size_t required_bytes,
    std::shared_ptr<DeviceAllocation>* allocation) {
    if (required_bytes == 0) {
        allocation->reset();
    } else if (!*allocation || (*allocation)->bytes() < required_bytes) {
        allocation->reset(new DeviceAllocation(device_id, required_bytes));
    }
}

}  // namespace

WccPartitionUpload::WccPartitionUpload()
    : partition_id(-1),
      device_id(-1),
      graph_identity(0),
      generation(0),
      owned_vertex_count(0),
      forest_contraction(false),
      boundary_csr(false),
      boundary_edge_count(0) {}

ResidentWccPartitionView::ResidentWccPartitionView(
    const WccPartitionUpload& upload)
    : managed_owner_id_(new_managed_owner_id("wcc", "view", upload.device_id)),
      state_key_(
          upload.graph_identity,
          upload.generation,
          stable_signature64(upload.view_signature),
          DeviceGroup(upload.device_group),
          GraphLayout::Partitioned),
      graph_identity_text_(upload.graph_identity_text),
      view_signature_text_(upload.view_signature),
      resident_bytes_(0),
      shared_array_count_(0U) {
    view_.partition_id = -1;
    view_.device_id = -1;
    view_.owned_vertex_count = 0;
    view_.local_vertex_count = 0;
    view_.adjacency_count = 0;
    view_.offsets = nullptr;
    view_.indices = nullptr;
    view_.local_to_global = nullptr;
    view_.owners = nullptr;
    view_.forest_contraction = upload.forest_contraction;
    view_.boundary_csr = upload.boundary_csr;
    view_.boundary_edge_count = 0;
    view_.boundary_sources = nullptr;
    view_.boundary_neighbors = nullptr;
}

std::shared_ptr<ResidentWccPartitionView> ResidentWccPartitionView::upload(
    const WccPartitionUpload& upload) {
    std::shared_ptr<ResidentWccPartitionView> result(
        new ResidentWccPartitionView(upload));
    result->initialize(upload);
    check_cuda(cudaStreamSynchronize(nullptr), "complete WCC view upload");
    return result;
}

void ResidentWccPartitionView::initialize(const WccPartitionUpload& upload) {
    const std::size_t local_count = upload.local_to_global.size();
    const std::size_t rank_count = upload.outgoing_owner_local_ids.size();
    if ((upload.wide_offsets ? !upload.offsets.empty() : !upload.offsets64.empty()) ||
        upload.partition_id < 0 || upload.device_id < 0 ||
        upload.view_signature.empty() || rank_count == 0 ||
        std::find(upload.device_group.begin(), upload.device_group.end(),
                  upload.device_id) == upload.device_group.end() ||
        upload.owned_vertex_count > local_count ||
        upload.owners.size() != local_count ||
        upload.offset_count() != upload.owned_vertex_count + 1U ||
        upload.offset_count() == 0 || upload.offset(0) != 0 ||
        upload.offset(upload.offset_count() - 1) < 0 ||
        static_cast<std::size_t>(upload.offset(upload.offset_count() - 1)) != upload.indices.size() ||
        upload.outgoing_owner_local_ids.size() != rank_count ||
        upload.incoming_halo_local_ids.size() != rank_count) {
        throw std::invalid_argument("invalid WCC partition upload");
    }
    if (upload.owned_vertex_count == 0 &&
        (local_count != 0 || !upload.indices.empty() ||
         (upload.offset_count() != 1U || upload.offset(0) != 0))) {
        throw std::invalid_argument("invalid empty WCC partition upload");
    }
    for (std::size_t edge = 0; edge < upload.indices.size(); ++edge) {
        if (upload.indices[edge] < 0 ||
            static_cast<std::size_t>(upload.indices[edge]) >= local_count) {
            throw std::invalid_argument("WCC partition index is out of range");
        }
    }
    for (std::size_t peer = 0; peer < rank_count; ++peer) {
        for (std::size_t index = 0;
             index < upload.outgoing_owner_local_ids[peer].size(); ++index) {
            const int local = upload.outgoing_owner_local_ids[peer][index];
            if (local < 0 || static_cast<std::size_t>(local) >=
                    upload.owned_vertex_count) {
                throw std::invalid_argument("WCC outgoing owner ID is invalid");
            }
        }
        for (std::size_t index = 0;
             index < upload.incoming_halo_local_ids[peer].size(); ++index) {
            const int local = upload.incoming_halo_local_ids[peer][index];
            if (local < 0 || static_cast<std::size_t>(local) <
                    upload.owned_vertex_count ||
                static_cast<std::size_t>(local) >= local_count) {
                throw std::invalid_argument("WCC incoming halo ID is invalid");
            }
        }
    }
    check_current_device(upload.device_id);
    // Empty partitions retain an empty state object and still participate in
    // the gang barrier; they never issue a zero-byte CUDA allocation.
    if (upload.owned_vertex_count != 0) {
        SharedArrayAcquireResult acquired;
        offsets_ = upload_verified_shared_device_array(
            upload.device_id, SharedArrayKind::Offsets,
            upload.offsets_sha256, upload.topology_compatibility,
            upload.offset_data(), upload.offset_bytes(),
            "upload shared WCC offsets", &acquired);
        shared_array_stats_.add(acquired);
        ++shared_array_count_;
        indices_ = upload_verified_shared_device_array(
            upload.device_id, SharedArrayKind::Indices,
            upload.indices_sha256, upload.topology_compatibility,
            upload.indices.empty() ? nullptr : upload.indices.data(),
            upload.indices.size() * sizeof(int),
            "upload shared WCC indices", &acquired);
        if (indices_) {
            shared_array_stats_.add(acquired);
            ++shared_array_count_;
        }
        local_to_global_ = upload_verified_shared_device_array(
            upload.device_id, SharedArrayKind::LocalToGlobal,
            upload.local_to_global_sha256, upload.ownership_compatibility,
            upload.local_to_global.data(),
            upload.local_to_global.size() * sizeof(std::uint64_t),
            "upload shared WCC local-to-global map", &acquired);
        shared_array_stats_.add(acquired);
        ++shared_array_count_;
        owners_ = upload_verified_shared_device_array(
            upload.device_id, SharedArrayKind::Owners,
            upload.owners_sha256, upload.ownership_compatibility,
            upload.owners.data(), upload.owners.size() * sizeof(int),
            "upload shared WCC owner map", &acquired);
        shared_array_stats_.add(acquired);
        ++shared_array_count_;
    }
    outgoing_indices_.resize(rank_count);
    incoming_indices_.resize(rank_count);
    view_.outgoing_owner_local_ids.resize(rank_count, nullptr);
    view_.incoming_halo_local_ids.resize(rank_count, nullptr);
    view_.outgoing_counts.resize(rank_count, 0U);
    view_.incoming_counts.resize(rank_count, 0U);
    for (std::size_t peer = 0; peer < rank_count; ++peer) {
        outgoing_indices_[peer] = upload_array(
            upload.device_id, upload.outgoing_owner_local_ids[peer],
            "upload WCC outgoing route");
        incoming_indices_[peer] = upload_array(
            upload.device_id, upload.incoming_halo_local_ids[peer],
            "upload WCC incoming route");
        view_.outgoing_owner_local_ids[peer] = outgoing_indices_[peer]
            ? static_cast<const int*>(outgoing_indices_[peer]->data())
            : nullptr;
        view_.incoming_halo_local_ids[peer] = incoming_indices_[peer]
            ? static_cast<const int*>(incoming_indices_[peer]->data())
            : nullptr;
        view_.outgoing_counts[peer] =
            upload.outgoing_owner_local_ids[peer].size();
        view_.incoming_counts[peer] =
            upload.incoming_halo_local_ids[peer].size();
        resident_bytes_ += bytes(outgoing_indices_[peer]) +
            bytes(incoming_indices_[peer]);
    }
    view_.partition_id = upload.partition_id;
    view_.device_id = upload.device_id;
    view_.owned_vertex_count = upload.owned_vertex_count;
    view_.local_vertex_count = local_count;
    view_.adjacency_count = upload.indices.size();
    view_.offsets = upload.wide_offsets
        ? gpu_easygraph::partitioned::WccOffsetPointer::from_wide(offsets_ ? static_cast<const std::int64_t*>(offsets_->data()) : nullptr)
        : gpu_easygraph::partitioned::WccOffsetPointer(offsets_ ? static_cast<const int*>(offsets_->data()) : nullptr);
    view_.indices = indices_ ? static_cast<const int*>(indices_->data()) : nullptr;
    view_.local_to_global = local_to_global_
        ? static_cast<const std::uint64_t*>(local_to_global_->data())
        : nullptr;
    view_.owners = owners_ ? static_cast<const int*>(owners_->data()) : nullptr;
    if (upload.forest_contraction) {
        view_.boundary_edge_count = upload.boundary_csr
            ? upload.boundary_edge_count : upload.boundary_sources.size();
    }
    if (upload.forest_contraction && !upload.boundary_csr) {
        boundary_sources_ = upload_array(upload.device_id,
            upload.boundary_sources, "upload WCC boundary sources");
        boundary_neighbors_ = upload_array(upload.device_id,
            upload.boundary_neighbors, "upload WCC boundary neighbors");
        view_.boundary_sources = boundary_sources_
            ? static_cast<const int*>(boundary_sources_->data()) : nullptr;
        view_.boundary_neighbors = boundary_neighbors_
            ? static_cast<const int*>(boundary_neighbors_->data()) : nullptr;
        resident_bytes_ += bytes(boundary_sources_) + bytes(boundary_neighbors_);
    }
    resident_bytes_ += bytes(offsets_) + bytes(indices_) +
        bytes(local_to_global_) + bytes(owners_);
}

const StateKey& ResidentWccPartitionView::state_key() const {
    return state_key_;
}
const std::string& ResidentWccPartitionView::view_signature_text() const {
    return view_signature_text_;
}
const WccPartitionDeviceView& ResidentWccPartitionView::view() const {
    return view_;
}
std::size_t ResidentWccPartitionView::resident_bytes() const {
    return resident_bytes_;
}
const SharedArrayCallStats& ResidentWccPartitionView::shared_array_stats() const {
    return shared_array_stats_;
}
std::size_t ResidentWccPartitionView::shared_array_count() const {
    return shared_array_count_;
}
bool ResidentWccPartitionView::cacheable() const {
    return shared_array_stats_.all_retained;
}

WccPartitionWorkspace::WccPartitionWorkspace(int device_id)
    : managed_owner_id_(new_managed_owner_id("wcc", "workspace", device_id)),
      device_id_(device_id) {
    if (device_id_ < 0) {
        throw std::invalid_argument("WCC workspace requires a device");
    }
}

void WccPartitionWorkspace::ensure(const WccPartitionDeviceView& view) {
    check_current_device(device_id_);
    ensure_allocation(
        device_id_, view.local_vertex_count * sizeof(std::uint64_t), &labels_);
    ensure_allocation(
        device_id_, view.owned_vertex_count ? sizeof(int) : 0U, &changed_);
    ensure_allocation(device_id_, view.forest_contraction
        ? view.owned_vertex_count * sizeof(int) : 0U, &local_parents_);
    send_buffers_.resize(view.outgoing_counts.size());
    receive_buffers_.resize(view.incoming_counts.size());
    for (std::size_t peer = 0; peer < view.outgoing_counts.size(); ++peer) {
        ensure_allocation(
            device_id_, view.outgoing_counts[peer] * sizeof(std::uint64_t),
            &send_buffers_[peer]);
        ensure_allocation(
            device_id_, view.incoming_counts[peer] * sizeof(std::uint64_t),
            &receive_buffers_[peer]);
    }
}

std::uint64_t* WccPartitionWorkspace::labels() const {
    return labels_ ? static_cast<std::uint64_t*>(labels_->data()) : nullptr;
}
int* WccPartitionWorkspace::changed() const {
    return changed_ ? static_cast<int*>(changed_->data()) : nullptr;
}
int* WccPartitionWorkspace::local_parents() const {
    return local_parents_ ? static_cast<int*>(local_parents_->data()) : nullptr;
}
std::uint64_t* WccPartitionWorkspace::send_buffer(
    std::size_t destination_rank) const {
    if (destination_rank >= send_buffers_.size()) {
        throw std::out_of_range("WCC send-buffer rank is out of range");
    }
    return send_buffers_[destination_rank]
        ? static_cast<std::uint64_t*>(send_buffers_[destination_rank]->data())
        : nullptr;
}
std::uint64_t* WccPartitionWorkspace::receive_buffer(
    std::size_t source_rank) const {
    if (source_rank >= receive_buffers_.size()) {
        throw std::out_of_range("WCC receive-buffer rank is out of range");
    }
    return receive_buffers_[source_rank]
        ? static_cast<std::uint64_t*>(receive_buffers_[source_rank]->data())
        : nullptr;
}
std::size_t WccPartitionWorkspace::capacity_bytes() const {
    std::size_t total = bytes(labels_) + bytes(changed_) + bytes(local_parents_);
    for (std::size_t index = 0; index < send_buffers_.size(); ++index) {
        total += bytes(send_buffers_[index]);
    }
    for (std::size_t index = 0; index < receive_buffers_.size(); ++index) {
        total += bytes(receive_buffers_[index]);
    }
    return total;
}

bool WccPartitionWorkspace::ensure_host_output(
    std::size_t count, const std::string& placement) {
    const std::size_t required = count * sizeof(std::uint64_t);
    if (!required) return false;
    if (host_output_ && host_output_->size() >= required &&
        host_output_placement_ == placement) return false;
    if (placement == "numa_local") {
        host_output_ = make_gpu_local_pinned_host_buffer(
            required, device_id_, &host_output_info_);
    } else if (placement == "caller") {
        host_output_.reset(new PinnedHostBuffer(required));
        host_output_info_ = GpuLocalPinnedAllocationInfo();
        host_output_info_.logical_device = device_id_;
        host_output_info_.allocation_cpu = sched_getcpu();
    } else {
        throw std::invalid_argument("WCC pinned placement must be caller or numa_local");
    }
    host_output_placement_ = placement;
    return true;
}

std::shared_ptr<PinnedHostBuffer> WccPartitionWorkspace::host_output() const {
    return host_output_;
}

const GpuLocalPinnedAllocationInfo& WccPartitionWorkspace::host_output_info() const {
    return host_output_info_;
}

bool PartitionedWccStateRegistry::RegistryKey::operator==(
    const RegistryKey& other) const {
    return partition_id == other.partition_id && state == other.state;
}

std::size_t PartitionedWccStateRegistry::RegistryKeyHash::operator()(
    const RegistryKey& key) const {
    std::size_t seed = StateKeyHash()(key.state);
    seed ^= std::hash<int>()(key.partition_id) +
        static_cast<std::size_t>(0x9e3779b9U) + (seed << 6U) + (seed >> 2U);
    return seed;
}

void PartitionedWccStateRegistry::touch_and_trim(const RegistryKey& key) {
    const std::size_t capacity = partition_state_cache_capacity();
    for (std::deque<RegistryKey>::iterator item = lru_.begin();
         item != lru_.end();) {
        if (*item == key) item = lru_.erase(item);
        else ++item;
    }
    lru_.push_back(key);
    while (lru_.size() > capacity) {
        const RegistryKey victim = lru_.front();
        lru_.pop_front();
        views_.erase(victim);
        workspaces_.erase(victim);
    }
}

std::shared_ptr<ResidentWccPartitionView>
PartitionedWccStateRegistry::acquire(
    const WccPartitionUpload& upload,
    bool* cache_hit,
    SharedArrayCallStats* shared_stats) {
    const RegistryKey key{
        StateKey(
            upload.graph_identity,
            upload.generation,
            stable_signature64(upload.view_signature),
            DeviceGroup(upload.device_group),
            GraphLayout::Partitioned),
        upload.partition_id};
    std::unordered_map<
        RegistryKey,
        std::shared_ptr<ResidentWccPartitionView>,
        RegistryKeyHash>::iterator found = views_.find(key);
    if (found != views_.end() &&
        found->second->view_signature_text() == upload.view_signature) {
        if (cache_hit) *cache_hit = true;
        if (shared_stats) {
            shared_stats->hits = found->second->shared_array_count();
            shared_stats->new_bytes = 0U;
            shared_stats->retained_bytes =
                shared_device_array_catalog().stats().retained_bytes;
            shared_stats->all_retained = true;
        }
        std::shared_ptr<ResidentWccPartitionView> result = found->second;
        touch_and_trim(key);
        return result;
    }
    if (found != views_.end()) {
        views_.erase(found);
        workspaces_.erase(key);
    }
    std::shared_ptr<ResidentWccPartitionView> built =
        ResidentWccPartitionView::upload(upload);
    for (std::unordered_map<
             RegistryKey,
             std::shared_ptr<ResidentWccPartitionView>,
             RegistryKeyHash>::iterator item = views_.begin();
         item != views_.end();) {
        const StateKey& state = item->first.state;
        if (item->first.partition_id == upload.partition_id &&
            state.graph_identity == upload.graph_identity &&
            state.device_group == key.state.device_group &&
            state.view_signature == key.state.view_signature &&
            state.generation != upload.generation) {
            workspaces_.erase(item->first);
            item = views_.erase(item);
        } else {
            ++item;
        }
    }
    if (built->cacheable()) {
        touch_and_trim(key);
        views_[key] = built;
    }
    if (cache_hit) *cache_hit = false;
    if (shared_stats) {
        *shared_stats = built->shared_array_stats();
        shared_stats->retained_bytes =
            shared_device_array_catalog().stats().retained_bytes;
    }
    return built;
}

std::shared_ptr<WccPartitionWorkspace> PartitionedWccStateRegistry::workspace(
    const std::shared_ptr<ResidentWccPartitionView>& view) {
    if (!view) throw std::invalid_argument("WCC workspace requires a view");
    if (managed_workspace_per_worker()) {
        if (!pooled_workspace_) {
            pooled_workspace_.reset(new WccPartitionWorkspace(view->view().device_id));
        } else if (pooled_workspace_.use_count() != 1) {
            // A caller may hold another partition on this worker (serial2), or
            // a prior lease may still be live. Never alias its mutable state.
            ++managed_pool_fallbacks_;
            std::shared_ptr<WccPartitionWorkspace> transient(new WccPartitionWorkspace(view->view().device_id));
            transient->ensure(view->view());
            return transient;
        } else {
            ++managed_pool_reuses_;
        }
        pooled_workspace_->ensure(view->view());
        return pooled_workspace_;
    }
    if (!view->cacheable()) {
        std::shared_ptr<WccPartitionWorkspace> transient(
            new WccPartitionWorkspace(view->view().device_id));
        transient->ensure(view->view());
        return transient;
    }
    const RegistryKey key{view->state_key(), view->view().partition_id};
    std::shared_ptr<WccPartitionWorkspace>& workspace = workspaces_[key];
    if (!workspace) {
        workspace.reset(new WccPartitionWorkspace(view->view().device_id));
    }
    workspace->ensure(view->view());
    return workspace;
}

void PartitionedWccStateRegistry::clear() {
    pooled_workspace_.reset();
    views_.clear();
    workspaces_.clear();
    lru_.clear();
}
std::size_t PartitionedWccStateRegistry::resident_view_count() const {
    return views_.size();
}
std::size_t PartitionedWccStateRegistry::workspace_count() const {
    return workspaces_.size() + (pooled_workspace_ ? 1U : 0U);
}

PartitionedWccStateRegistry& partitioned_wcc_state_registry() {
    static thread_local PartitionedWccStateRegistry registry;
    return registry;
}


void ResidentWccPartitionView::append_managed_allocations(
    ManagedStateInventory& inventory, const ManagedOwnerRecord& owner) const {
    inventory.owner(owner);
    inventory.add(offsets_, owner.owner_id, "wcc.offsets", true);
    inventory.add(indices_, owner.owner_id, "wcc.indices", true);
    inventory.add(local_to_global_, owner.owner_id, "wcc.local_to_global", true);
    inventory.add(owners_, owner.owner_id, "wcc.owners", true);
    inventory.add(boundary_sources_, owner.owner_id, "wcc.boundary_sources", false);
    inventory.add(boundary_neighbors_, owner.owner_id, "wcc.boundary_neighbors", false);
    for (std::size_t i=0; i<outgoing_indices_.size(); ++i)
        inventory.add(outgoing_indices_[i], owner.owner_id, ("wcc.outgoing_indices." + std::to_string(i)).c_str());
    for (std::size_t i=0; i<incoming_indices_.size(); ++i)
        inventory.add(incoming_indices_[i], owner.owner_id, ("wcc.incoming_indices." + std::to_string(i)).c_str());
}

void WccPartitionWorkspace::append_managed_allocations(
    ManagedStateInventory& inventory, const ManagedOwnerRecord& owner) const {
    inventory.owner(owner);
    inventory.add(labels_, owner.owner_id, "wcc.workspace.labels");
    inventory.add(changed_, owner.owner_id, "wcc.workspace.changed");
    inventory.add(local_parents_, owner.owner_id, "wcc.workspace.local_parents");
    for (std::size_t i=0; i<send_buffers_.size(); ++i)
        inventory.add(send_buffers_[i], owner.owner_id, ("wcc.workspace.send_buffers." + std::to_string(i)).c_str());
    for (std::size_t i=0; i<receive_buffers_.size(); ++i)
        inventory.add(receive_buffers_[i], owner.owner_id, ("wcc.workspace.receive_buffers." + std::to_string(i)).c_str());
    // host_output_ is pinned HOST storage, excluded from device allocation union.
}

void PartitionedWccStateRegistry::append_managed_inventory(ManagedStateInventory& inventory) const {
    for (const auto& entry : views_) {
        const auto& view = entry.second;
        ManagedOwnerRecord owner;
        owner.owner_id = view->managed_owner_id();
        owner.kind = "view"; owner.algorithm = "wcc";
        owner.device_id = inventory.device_id;
        owner.graph_id = view->graph_identity_text();
        owner.partition_id = entry.first.partition_id;
        owner.generation = entry.first.state.generation;
        owner.active_leases = view.use_count() - 1U;
        view->append_managed_allocations(inventory, owner);
    }
    for (const auto& entry : workspaces_) {
        const auto& workspace = entry.second;
        ManagedOwnerRecord owner;
        owner.owner_id = workspace->managed_owner_id();
        owner.kind = "workspace"; owner.algorithm = "wcc";
        owner.policy = "per_view"; owner.device_id = inventory.device_id;
        owner.partition_id = entry.first.partition_id;
        owner.generation = entry.first.state.generation;
        const auto view = views_.find(entry.first);
        if (view != views_.end()) {
            owner.graph_id = view->second->graph_identity_text();
            owner.parent_owner_id = view->second->managed_owner_id();
        }
        owner.active_leases = workspace.use_count() - 1U;
        workspace->append_managed_allocations(inventory, owner);
    }
    if (pooled_workspace_) {
        ManagedOwnerRecord owner;
        owner.owner_id = pooled_workspace_->managed_owner_id();
        owner.kind = "workspace"; owner.algorithm = "wcc";
        owner.policy = "per_worker"; owner.device_id = inventory.device_id;
        owner.active_leases = pooled_workspace_.use_count() - 1U;
        pooled_workspace_->append_managed_allocations(inventory, owner);
    }
    inventory.counters["wcc_pool_reuses"] = managed_pool_reuses_;
    inventory.counters["wcc_pool_active_fallbacks"] = managed_pool_fallbacks_;
}

void PartitionedWccStateRegistry::managed_evict(const std::set<std::string>& requested,
    bool release_workspaces, ManagedEvictRecord& receipt) {
    for (auto item = views_.begin(); item != views_.end();) {
        const std::string id = item->second->managed_owner_id();
        if (!requested.count(id)) { ++item; continue; }
        const auto workspace = workspaces_.find(item->first);
        if (item->second.use_count() != 1 ||
            (workspace != workspaces_.end() && workspace->second.use_count() != 1)) {
            receipt.skipped_active.push_back(id); ++item; continue;
        }
        if (workspace != workspaces_.end()) {
            receipt.released_workspaces.push_back(workspace->second->managed_owner_id());
            workspaces_.erase(workspace);
        }
        const RegistryKey key = item->first;
        for (auto lru = lru_.begin(); lru != lru_.end();) {
            if (*lru == key) lru = lru_.erase(lru); else ++lru;
        }
        receipt.removed.push_back(id);
        item = views_.erase(item);
    }
    for (auto item = workspaces_.begin(); item != workspaces_.end();) {
        const std::string id = item->second->managed_owner_id();
        if (!release_workspaces && !requested.count(id)) { ++item; continue; }
        if (item->second.use_count() != 1) {
            receipt.skipped_active.push_back(id); ++item; continue;
        }
        receipt.released_workspaces.push_back(id);
        item = workspaces_.erase(item);
    }
    if (pooled_workspace_ &&
        (release_workspaces || requested.count(pooled_workspace_->managed_owner_id()))) {
        if (pooled_workspace_.use_count() == 1) {
            receipt.released_workspaces.push_back(pooled_workspace_->managed_owner_id());
            pooled_workspace_.reset();
        } else {
            receipt.skipped_active.push_back(pooled_workspace_->managed_owner_id());
        }
    }
}

}  // namespace runtime
}  // namespace gpu_easygraph
