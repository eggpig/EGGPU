#include "partitioned_sssp_state.h"
#include "single_owner_sssp_contract.h"

#include <algorithm>
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
    check_cuda(cudaGetDevice(&current), "cudaGetDevice for SSSP state");
    if (current != expected) {
        throw std::runtime_error(
            "SSSP state must be accessed by its fixed device worker");
    }
}

template <typename T>
std::shared_ptr<DeviceAllocation> upload_array(
    int device_id, const std::vector<T>& source, const char* operation) {
    if (source.empty()) return std::shared_ptr<DeviceAllocation>();
    const std::size_t byte_count = source.size() * sizeof(T);
    std::shared_ptr<DeviceAllocation> result(
        new DeviceAllocation(device_id, byte_count));
    check_cuda(
        cudaMemcpy(
            result->data(), source.data(), byte_count, cudaMemcpyHostToDevice),
        operation);
    return result;
}

std::size_t allocation_bytes(
    const std::shared_ptr<DeviceAllocation>& allocation) {
    return allocation ? allocation->bytes() : 0U;
}

std::size_t allocation_bytes(
    const std::shared_ptr<SharedDeviceArray>& allocation) {
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

void validate_offsets(
    const std::vector<std::size_t>& offsets,
    std::size_t rank_count,
    std::size_t total,
    const char* context) {
    if (offsets.size() != rank_count + 1U || offsets.front() != 0U ||
        offsets.back() != total ||
        !std::is_sorted(offsets.begin(), offsets.end())) {
        throw std::invalid_argument(context);
    }
}

}  // namespace

SsspPartitionUpload::SsspPartitionUpload()
    : partition_id(-1),
      device_id(-1),
      graph_identity(0),
      generation(0),
      owned_vertex_count(0) {}

ResidentSsspPartitionView::ResidentSsspPartitionView(
    const SsspPartitionUpload& upload)
    : managed_owner_id_(new_managed_owner_id("sssp", "view", upload.device_id)),
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
    view_.weights = nullptr;
    view_.weight_encoding = SsspWeightEncoding::F64;
    view_.wire_encoding = SsspWireEncoding::F64;
    view_.wire_distance_bound = std::numeric_limits<double>::infinity();
    view_.weight_bytes = 0;
    view_.owners = nullptr;
    view_.halo_route_slots = nullptr;
    view_.route_destination_ranks = nullptr;
    view_.route_owner_local_ids = nullptr;
}

std::shared_ptr<ResidentSsspPartitionView>
ResidentSsspPartitionView::upload(const SsspPartitionUpload& upload) {
    std::shared_ptr<ResidentSsspPartitionView> result(
        new ResidentSsspPartitionView(upload));
    result->initialize(upload);
    check_cuda(cudaStreamSynchronize(nullptr), "complete SSSP view upload");
    return result;
}

void ResidentSsspPartitionView::initialize(
    const SsspPartitionUpload& upload) {
    const std::size_t local_count = upload.owners.size();
    if (upload.weight_encoding != SsspWeightEncoding::F64 &&
        upload.weight_encoding != SsspWeightEncoding::U8Exact) {
        throw std::invalid_argument("unknown resident SSSP weight encoding");
    }
    const bool compact_weights = upload.weight_encoding == SsspWeightEncoding::U8Exact;
    const std::size_t weight_count = compact_weights ? upload.weights_u8.size() : upload.weights.size();
    if ((compact_weights && !upload.weights.empty()) ||
        (!compact_weights && !upload.weights_u8.empty())) {
        throw std::invalid_argument("SSSP upload must hold exactly its selected weight encoding");
    }
    // Logical ownership/routing ranks are independent of the physical device
    // count when all partitions execute serially on one GPU.
    const std::size_t rank_count = upload.outgoing_route_offsets.empty()
        ? 0U : upload.outgoing_route_offsets.size() - 1U;
    const std::size_t route_total = upload.route_owner_local_ids.size();
    if (upload.partition_id < 0 || upload.device_id < 0 ||
        upload.graph_identity_text.empty() || upload.view_signature.empty() ||
        rank_count == 0 ||
        std::find(
            upload.device_group.begin(), upload.device_group.end(),
            upload.device_id) == upload.device_group.end() ||
        upload.owned_vertex_count > local_count ||
        upload.halo_route_slots.size() != local_count ||
        upload.offsets.size() != upload.owned_vertex_count + 1U ||
        upload.offsets.empty() || upload.offsets.front() != 0 ||
        upload.offsets.back() < 0 ||
        static_cast<std::size_t>(upload.offsets.back()) !=
            upload.indices.size() ||
        weight_count != upload.indices.size() ||
        upload.route_destination_ranks.size() != route_total) {
        throw std::invalid_argument("invalid SSSP partition upload");
    }
    if (upload.owned_vertex_count == 0 &&
        (local_count != 0 || !upload.indices.empty() ||
         weight_count != 0 || route_total != 0 ||
         upload.offsets.view() != std::vector<int>(1U, 0))) {
        throw std::invalid_argument("invalid empty SSSP partition upload");
    }
    validate_offsets(
        upload.outgoing_route_offsets, rank_count, route_total,
        "invalid SSSP outgoing route offsets");
    if (upload.incoming_route_offsets.size() != rank_count + 1U ||
        upload.incoming_route_offsets.front() != 0U ||
        !std::is_sorted(
            upload.incoming_route_offsets.begin(),
            upload.incoming_route_offsets.end())) {
        throw std::invalid_argument("invalid SSSP incoming route offsets");
    }
    for (std::size_t edge = 0; edge < upload.indices.size(); ++edge) {
        const int target = upload.indices[edge];
        if (target < 0 || static_cast<std::size_t>(target) >= local_count) {
            throw std::invalid_argument(
                "SSSP partition edge endpoint is out of range");
        }
    }
    for (std::size_t local = 0; local < local_count; ++local) {
        const int owner = upload.owners[local];
        const int slot = upload.halo_route_slots[local];
        if (owner < 0 || static_cast<std::size_t>(owner) >= rank_count) {
            throw std::invalid_argument("SSSP partition owner is out of range");
        }
        if (local < upload.owned_vertex_count) {
            if (owner != upload.partition_id || slot != -1) {
                throw std::invalid_argument(
                    "SSSP owned vertex has invalid route metadata");
            }
        } else if (owner == upload.partition_id || slot < 0 ||
                   static_cast<std::size_t>(slot) >= route_total ||
                   upload.route_destination_ranks[
                       static_cast<std::size_t>(slot)] != owner) {
            throw std::invalid_argument(
                "SSSP halo vertex has invalid route metadata");
        }
    }

    check_current_device(upload.device_id);
    SharedArrayAcquireResult acquired;
    offsets_ = upload_verified_shared_device_array(
        upload.device_id, SharedArrayKind::Offsets, upload.offsets_sha256,
        upload.topology_compatibility, upload.offsets.data(),
        upload.offsets.size() * sizeof(int), "upload shared SSSP offsets",
        &acquired);
    shared_array_stats_.add(acquired);
    ++shared_array_count_;
    indices_ = upload_verified_shared_device_array(
        upload.device_id, SharedArrayKind::Indices, upload.indices_sha256,
        upload.topology_compatibility,
        upload.indices.empty() ? nullptr : upload.indices.data(),
        upload.indices.size() * sizeof(int), "upload shared SSSP indices",
        &acquired);
    if (indices_) {
        shared_array_stats_.add(acquired);
        ++shared_array_count_;
    }
    const std::size_t weight_bytes = weight_count * (compact_weights ? sizeof(std::uint8_t) : sizeof(double));
    const void* weight_source = weight_count == 0 ? nullptr : compact_weights
        ? static_cast<const void*>(upload.weights_u8.data()) : static_cast<const void*>(upload.weights.data());
    weights_ = upload_verified_shared_device_array(
        upload.device_id, SharedArrayKind::Weights, upload.weights_sha256,
        upload.weight_compatibility,
        weight_source, weight_bytes, "upload shared SSSP weights",
        &acquired, upload.native_weight_patch.get());
    if (weights_) {
        shared_array_stats_.add(acquired);
        ++shared_array_count_;
    }
    owners_ = upload_verified_shared_device_array(
        upload.device_id, SharedArrayKind::Owners, upload.owners_sha256,
        upload.ownership_compatibility,
        upload.owners.empty() ? nullptr : upload.owners.data(),
        upload.owners.size() * sizeof(int), "upload shared SSSP owners",
        &acquired);
    if (owners_) {
        shared_array_stats_.add(acquired);
        ++shared_array_count_;
    }
    halo_route_slots_ = upload_array(
        upload.device_id, upload.halo_route_slots.view(),
        "upload SSSP halo route slots");
    route_destination_ranks_ = upload_array(
        upload.device_id, upload.route_destination_ranks.view(),
        "upload SSSP route destination ranks");
    route_owner_local_ids_ = upload_array(
        upload.device_id, upload.route_owner_local_ids.view(),
        "upload SSSP owner local IDs");

    view_.partition_id = upload.partition_id;
    view_.device_id = upload.device_id;
    view_.owned_vertex_count = upload.owned_vertex_count;
    view_.local_vertex_count = local_count;
    view_.adjacency_count = upload.indices.size();
    view_.offsets = offsets_ ? static_cast<const int*>(offsets_->data()) : nullptr;
    view_.indices = indices_ ? static_cast<const int*>(indices_->data()) : nullptr;
    view_.weights = weights_ ? weights_->data() : nullptr;
    view_.weight_encoding = upload.weight_encoding;
    view_.wire_encoding = upload.wire_encoding;
    view_.wire_distance_bound = upload.wire_distance_bound;
    view_.weight_bytes = weights_ ? weights_->bytes() : 0U;
    view_.owners = owners_ ? static_cast<const int*>(owners_->data()) : nullptr;
    view_.halo_route_slots = halo_route_slots_
        ? static_cast<const int*>(halo_route_slots_->data())
        : nullptr;
    view_.route_destination_ranks = route_destination_ranks_
        ? static_cast<const int*>(route_destination_ranks_->data())
        : nullptr;
    view_.route_owner_local_ids = route_owner_local_ids_
        ? static_cast<const int*>(route_owner_local_ids_->data())
        : nullptr;
    view_.outgoing_route_offsets = upload.outgoing_route_offsets;
    view_.incoming_route_offsets = upload.incoming_route_offsets;
    resident_bytes_ = allocation_bytes(offsets_) + allocation_bytes(indices_) +
        allocation_bytes(weights_) + allocation_bytes(owners_) +
        allocation_bytes(halo_route_slots_) +
        allocation_bytes(route_destination_ranks_) +
        allocation_bytes(route_owner_local_ids_);
}

const StateKey& ResidentSsspPartitionView::state_key() const {
    return state_key_;
}
const std::string& ResidentSsspPartitionView::graph_identity_text() const {
    return graph_identity_text_;
}
const std::string& ResidentSsspPartitionView::view_signature_text() const {
    return view_signature_text_;
}
const SsspPartitionDeviceView& ResidentSsspPartitionView::view() const {
    return view_;
}
std::size_t ResidentSsspPartitionView::resident_bytes() const {
    return resident_bytes_;
}
const SharedArrayCallStats& ResidentSsspPartitionView::shared_array_stats() const {
    return shared_array_stats_;
}
std::size_t ResidentSsspPartitionView::shared_array_count() const {
    return shared_array_count_;
}
bool ResidentSsspPartitionView::cacheable() const {
    // The complete view owns every array, including catalog-bypass arrays.
    // Catalog admission governs cross-view sharing, not this view's lifetime.
    // Retention/release remains governed by registry LRU and the byte controller.
    return true;
}

SsspPartitionWorkspace::SsspPartitionWorkspace(int device_id)
    : managed_owner_id_(new_managed_owner_id("sssp", "workspace", device_id)),
      device_id_(device_id) {
    if (device_id_ < 0) {
        throw std::invalid_argument("SSSP workspace requires a device");
    }
}

void SsspPartitionWorkspace::ensure(const SsspPartitionDeviceView& view) {
    check_current_device(device_id_);
    wire_encoding_ = view.wire_encoding;
    outgoing_route_offsets_ = view.outgoing_route_offsets;
    incoming_route_offsets_ = view.incoming_route_offsets;
    const std::size_t owned = view.owned_vertex_count;
    const std::size_t outgoing = outgoing_route_offsets_.back();
    const std::size_t incoming = incoming_route_offsets_.back();
    const std::size_t ranks = outgoing_route_offsets_.size() - 1U;
    ensure_allocation(device_id_, owned * sizeof(double), &distances_);
    ensure_allocation(device_id_, owned * sizeof(int), &settled_);
    ensure_allocation(device_id_, owned * sizeof(int), &current_frontier_);
    ensure_allocation(device_id_, owned * sizeof(int), &next_frontier_);
    ensure_allocation(device_id_, owned * sizeof(int), &r_flags_);
    ensure_allocation(device_id_, owned * sizeof(int), &next_flags_);
    // Scalars remain allocated for empty partitions so every worker can
    // participate in termination without issuing a zero-byte allocation.
    ensure_allocation(device_id_, sizeof(int), &frontier_count_);
    ensure_allocation(device_id_, sizeof(std::uint64_t), &minimum_bucket_);
    ensure_allocation(
        device_id_, outgoing * sizeof(double), &remote_candidates_);
    ensure_allocation(
        device_id_, outgoing * sssp_wire_bytes(wire_encoding_), &send_values_);
    ensure_allocation(device_id_, ranks * sizeof(int), &send_counts_);
    ensure_allocation(
        device_id_, incoming * sssp_wire_bytes(wire_encoding_), &receive_values_);
}

void SsspPartitionWorkspace::ensure_single_owner_scratch(std::size_t vertices) {
    check_current_device(device_id_);
    const auto bytes = single_owner_sssp_workspace_bytes(vertices);
    if (!bytes) throw std::invalid_argument("Single-owner SSSP scratch size outside supported ABI");
    ensure_allocation(device_id_, bytes, &single_owner_scratch_);
}
void* SsspPartitionWorkspace::single_owner_scratch() const {
    return single_owner_scratch_ ? single_owner_scratch_->data() : nullptr;
}
std::size_t SsspPartitionWorkspace::single_owner_scratch_capacity() const {
    return allocation_bytes(single_owner_scratch_);
}
double* SsspPartitionWorkspace::distances() const {
    return distances_ ? static_cast<double*>(distances_->data()) : nullptr;
}
int* SsspPartitionWorkspace::settled() const {
    return settled_ ? static_cast<int*>(settled_->data()) : nullptr;
}
int* SsspPartitionWorkspace::current_frontier() const {
    return current_frontier_
        ? static_cast<int*>(current_frontier_->data())
        : nullptr;
}
int* SsspPartitionWorkspace::next_frontier() const {
    return next_frontier_ ? static_cast<int*>(next_frontier_->data()) : nullptr;
}
int* SsspPartitionWorkspace::r_flags() const {
    return r_flags_ ? static_cast<int*>(r_flags_->data()) : nullptr;
}
int* SsspPartitionWorkspace::next_flags() const {
    return next_flags_ ? static_cast<int*>(next_flags_->data()) : nullptr;
}
int* SsspPartitionWorkspace::frontier_count() const {
    return static_cast<int*>(frontier_count_->data());
}
std::uint64_t* SsspPartitionWorkspace::minimum_bucket() const {
    return static_cast<std::uint64_t*>(minimum_bucket_->data());
}
double* SsspPartitionWorkspace::remote_candidates() const {
    return remote_candidates_
        ? static_cast<double*>(remote_candidates_->data())
        : nullptr;
}
void* SsspPartitionWorkspace::send_values(
    std::size_t destination_rank) const {
    if (destination_rank + 1U >= outgoing_route_offsets_.size()) {
        throw std::out_of_range("SSSP destination rank is out of range");
    }
    return send_values_
        ? static_cast<unsigned char*>(send_values_->data()) +
              outgoing_route_offsets_[destination_rank] * sssp_wire_bytes(wire_encoding_)
        : nullptr;
}
int* SsspPartitionWorkspace::send_counts() const {
    return static_cast<int*>(send_counts_->data());
}
void* SsspPartitionWorkspace::receive_values(
    std::size_t source_rank) const {
    if (source_rank + 1U >= incoming_route_offsets_.size()) {
        throw std::out_of_range("SSSP source rank is out of range");
    }
    return receive_values_
        ? static_cast<unsigned char*>(receive_values_->data()) +
              incoming_route_offsets_[source_rank] * sssp_wire_bytes(wire_encoding_)
        : nullptr;
}
void SsspPartitionWorkspace::advance_frontier() {
    current_frontier_.swap(next_frontier_);
}
std::size_t SsspPartitionWorkspace::capacity_bytes() const {
    return allocation_bytes(single_owner_scratch_) + allocation_bytes(distances_) + allocation_bytes(settled_) +
        allocation_bytes(current_frontier_) +
        allocation_bytes(next_frontier_) + allocation_bytes(r_flags_) +
        allocation_bytes(next_flags_) + allocation_bytes(frontier_count_) +
        allocation_bytes(minimum_bucket_) +
        allocation_bytes(remote_candidates_) + allocation_bytes(send_values_) +
        allocation_bytes(send_counts_) + allocation_bytes(receive_values_);
}

bool PartitionedSsspStateRegistry::RegistryKey::operator==(
    const RegistryKey& other) const {
    return partition_id == other.partition_id && state == other.state;
}
std::size_t PartitionedSsspStateRegistry::RegistryKeyHash::operator()(
    const RegistryKey& key) const {
    std::size_t seed = StateKeyHash()(key.state);
    seed ^= std::hash<int>()(key.partition_id) +
        static_cast<std::size_t>(0x9e3779b9U) + (seed << 6U) +
        (seed >> 2U);
    return seed;
}

void PartitionedSsspStateRegistry::touch_and_trim(const RegistryKey& key) {
    const std::size_t capacity = partition_state_cache_capacity();
    for (std::deque<RegistryKey>::iterator item = lru_.begin();
         item != lru_.end();) {
        if (*item == key) item = lru_.erase(item);
        else ++item;
    }
    lru_.push_back(key);
    std::size_t remaining = lru_.size();
    while (lru_.size() > capacity && remaining--) {
        const RegistryKey victim = lru_.front();
        lru_.pop_front();
        const auto view = views_.find(victim);
        const auto workspace = workspaces_.find(victim);
        if ((view != views_.end() && view->second.use_count() > 1U) ||
            (workspace != workspaces_.end() && workspace->second.use_count() > 1U)) {
            // A live readonly view remains inventoried until its lease ends.
            // Cache entry count is a soft bound while every victim is leased.
            lru_.push_back(victim);
            continue;
        }
        views_.erase(victim);
        workspaces_.erase(victim);
    }
}

std::vector<std::shared_ptr<ResidentSsspPartitionView> >
PartitionedSsspStateRegistry::hold_graph_views(
    const std::string& graph_id, std::uint64_t generation) const {
    std::vector<std::shared_ptr<ResidentSsspPartitionView> > held;
    for (const auto& item : views_) {
        if (item.second->graph_identity_text() == graph_id &&
            item.second->state_key().generation == generation)
            held.push_back(item.second);
    }
    return held;
}

std::shared_ptr<ResidentSsspPartitionView>
PartitionedSsspStateRegistry::acquire(
    const SsspPartitionUpload& upload,
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
        std::shared_ptr<ResidentSsspPartitionView>,
        RegistryKeyHash>::iterator found = views_.find(key);
    if (found != views_.end() &&
        found->second->graph_identity_text() == upload.graph_identity_text &&
        found->second->view_signature_text() == upload.view_signature) {
        if (cache_hit) *cache_hit = true;
        if (shared_stats) {
            shared_stats->hits = found->second->shared_array_count();
            shared_stats->new_bytes = 0U;
            shared_stats->retained_bytes =
                shared_device_array_catalog().stats().retained_bytes;
            shared_stats->all_retained = true;
        }
        std::shared_ptr<ResidentSsspPartitionView> result = found->second;
        touch_and_trim(key);
        return result;
    }
    if (found != views_.end()) {
        views_.erase(found);
        workspaces_.erase(key);
    }
    std::shared_ptr<ResidentSsspPartitionView> built =
        ResidentSsspPartitionView::upload(upload);
    for (std::unordered_map<
             RegistryKey,
             std::shared_ptr<ResidentSsspPartitionView>,
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

std::shared_ptr<SsspPartitionWorkspace>
PartitionedSsspStateRegistry::workspace(
    const std::shared_ptr<ResidentSsspPartitionView>& view) {
    if (!view) throw std::invalid_argument("SSSP workspace requires a view");
    if (managed_workspace_per_worker()) {
        if (!pooled_workspace_) {
            pooled_workspace_.reset(new SsspPartitionWorkspace(view->view().device_id));
        } else if (pooled_workspace_.use_count() != 1) {
            // A caller may hold another partition on this worker (serial2), or
            // a prior lease may still be live. Never alias its mutable state.
            ++managed_pool_fallbacks_;
            std::shared_ptr<SsspPartitionWorkspace> transient(new SsspPartitionWorkspace(view->view().device_id));
            transient->ensure(view->view());
            return transient;
        } else {
            ++managed_pool_reuses_;
        }
        pooled_workspace_->ensure(view->view());
        return pooled_workspace_;
    }
    if (!view->cacheable()) {
        std::shared_ptr<SsspPartitionWorkspace> transient(
            new SsspPartitionWorkspace(view->view().device_id));
        transient->ensure(view->view());
        return transient;
    }
    const RegistryKey key{view->state_key(), view->view().partition_id};
    std::shared_ptr<SsspPartitionWorkspace>& workspace = workspaces_[key];
    if (!workspace) {
        workspace.reset(new SsspPartitionWorkspace(view->view().device_id));
    }
    workspace->ensure(view->view());
    return workspace;
}
void PartitionedSsspStateRegistry::clear() {
    pooled_workspace_.reset();
    views_.clear();
    workspaces_.clear();
    lru_.clear();
}
std::size_t PartitionedSsspStateRegistry::resident_view_count() const {
    return views_.size();
}
std::size_t PartitionedSsspStateRegistry::workspace_count() const {
    return workspaces_.size() + (pooled_workspace_ ? 1U : 0U);
}
PartitionedSsspStateRegistry& partitioned_sssp_state_registry() {
    static thread_local PartitionedSsspStateRegistry registry;
    return registry;
}


void ResidentSsspPartitionView::append_managed_allocations(
    ManagedStateInventory& inventory, const ManagedOwnerRecord& owner) const {
    inventory.owner(owner);
    inventory.add(offsets_, owner.owner_id, "sssp.offsets", true);
    inventory.add(indices_, owner.owner_id, "sssp.indices", true);
    inventory.add(weights_, owner.owner_id, "sssp.weights", true);
    inventory.add(owners_, owner.owner_id, "sssp.owners", true);
    inventory.add(halo_route_slots_, owner.owner_id, "sssp.halo_route_slots", false);
    inventory.add(route_destination_ranks_, owner.owner_id, "sssp.route_destination_ranks", false);
    inventory.add(route_owner_local_ids_, owner.owner_id, "sssp.route_owner_local_ids", false);
}

void SsspPartitionWorkspace::append_managed_allocations(
    ManagedStateInventory& inventory, const ManagedOwnerRecord& owner) const {
    inventory.owner(owner);
    inventory.add(single_owner_scratch_, owner.owner_id, "sssp.workspace.single_owner_scratch");
    inventory.add(distances_, owner.owner_id, "sssp.workspace.distances");
    inventory.add(settled_, owner.owner_id, "sssp.workspace.settled");
    inventory.add(current_frontier_, owner.owner_id, "sssp.workspace.current_frontier");
    inventory.add(next_frontier_, owner.owner_id, "sssp.workspace.next_frontier");
    inventory.add(r_flags_, owner.owner_id, "sssp.workspace.r_flags");
    inventory.add(next_flags_, owner.owner_id, "sssp.workspace.next_flags");
    inventory.add(frontier_count_, owner.owner_id, "sssp.workspace.frontier_count");
    inventory.add(minimum_bucket_, owner.owner_id, "sssp.workspace.minimum_bucket");
    inventory.add(remote_candidates_, owner.owner_id, "sssp.workspace.remote_candidates");
    inventory.add(send_values_, owner.owner_id, "sssp.workspace.send_values");
    inventory.add(send_counts_, owner.owner_id, "sssp.workspace.send_counts");
    inventory.add(receive_values_, owner.owner_id, "sssp.workspace.receive_values");
}

void PartitionedSsspStateRegistry::append_managed_inventory(ManagedStateInventory& inventory) const {
    for (const auto& entry : views_) {
        const auto& view = entry.second;
        ManagedOwnerRecord owner;
        owner.owner_id = view->managed_owner_id();
        owner.kind = "view"; owner.algorithm = "sssp";
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
        owner.kind = "workspace"; owner.algorithm = "sssp";
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
        owner.kind = "workspace"; owner.algorithm = "sssp";
        owner.policy = "per_worker"; owner.device_id = inventory.device_id;
        owner.active_leases = pooled_workspace_.use_count() - 1U;
        pooled_workspace_->append_managed_allocations(inventory, owner);
    }
    inventory.counters["sssp_pool_reuses"] = managed_pool_reuses_;
    inventory.counters["sssp_pool_active_fallbacks"] = managed_pool_fallbacks_;
}

void PartitionedSsspStateRegistry::managed_evict(const std::set<std::string>& requested,
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
