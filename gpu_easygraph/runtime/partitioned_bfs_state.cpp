#include "partitioned_bfs_state.h"
#include "../functions/partitioned/bfs_partitioned.cuh"

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
    check_cuda(cudaGetDevice(&current), "cudaGetDevice for BFS state");
    if (current != expected) {
        throw std::runtime_error(
            "BFS state must be accessed by its fixed device worker");
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
        cudaMemcpy(result->data(), source.data(), bytes, cudaMemcpyHostToDevice),
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

void validate_route_offsets(
    const std::vector<std::size_t>& offsets,
    std::size_t rank_count,
    std::size_t expected_total,
    const char* context) {
    if (offsets.size() != rank_count + 1U || offsets.front() != 0U ||
        offsets.back() != expected_total ||
        !std::is_sorted(offsets.begin(), offsets.end())) {
        throw std::invalid_argument(context);
    }
}

}  // namespace

BfsPartitionUpload::BfsPartitionUpload()
    : partition_id(-1),
      device_id(-1),
      graph_identity(0),
      generation(0),
      owned_vertex_count(0) {}

ResidentBfsPartitionView::ResidentBfsPartitionView(
    const BfsPartitionUpload& upload)
    : managed_owner_id_(new_managed_owner_id("bfs", "view", upload.device_id)),
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
    view_.owners = nullptr;
    view_.halo_route_slots = nullptr;
    view_.route_destination_ranks = nullptr;
    view_.route_owner_local_ids = nullptr;
    view_.outgoing_dense_word_offsets_device = nullptr;
    view_.outgoing_route_offsets_device = nullptr;
}

std::shared_ptr<ResidentBfsPartitionView> ResidentBfsPartitionView::upload(
    const BfsPartitionUpload& upload) {
    std::shared_ptr<ResidentBfsPartitionView> result(
        new ResidentBfsPartitionView(upload));
    result->initialize(upload);
    // The cache miss acquires a ready view; warm cache hits do not fence.
    // All preceding pageable H2D uploads use the default stream.
    check_cuda(cudaStreamSynchronize(nullptr), "complete BFS view upload");
    return result;
}

void ResidentBfsPartitionView::initialize(
    const BfsPartitionUpload& upload) {
    const std::size_t local_count = upload.owners.size();
    const std::size_t rank_count = upload.device_group.size();
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
        upload.route_destination_ranks.size() != route_total) {
        throw std::invalid_argument("invalid BFS partition upload");
    }
    if (upload.owned_vertex_count == 0 &&
        (local_count != 0 || !upload.indices.empty() || route_total != 0 ||
         upload.offsets != std::vector<int>(1U, 0))) {
        throw std::invalid_argument("invalid empty BFS partition upload");
    }
    validate_route_offsets(
        upload.outgoing_route_offsets, rank_count, route_total,
        "invalid BFS outgoing route offsets");
    if (upload.incoming_route_offsets.size() != rank_count + 1U ||
        upload.incoming_route_offsets.front() != 0U ||
        !std::is_sorted(
            upload.incoming_route_offsets.begin(),
            upload.incoming_route_offsets.end())) {
        throw std::invalid_argument("invalid BFS incoming route offsets");
    }
    if (upload.outgoing_dense_word_offsets.size() != rank_count + 1U ||
        upload.incoming_dense_word_offsets.size() != rank_count + 1U ||
        upload.outgoing_dense_word_offsets.front() != 0U ||
        upload.incoming_dense_word_offsets.front() != 0U ||
        !std::is_sorted(
            upload.outgoing_dense_word_offsets.begin(),
            upload.outgoing_dense_word_offsets.end()) ||
        !std::is_sorted(
            upload.incoming_dense_word_offsets.begin(),
            upload.incoming_dense_word_offsets.end())) {
        throw std::invalid_argument("invalid BFS dense bitmap route offsets");
    }
    for (std::size_t edge = 0; edge < upload.indices.size(); ++edge) {
        const int target = upload.indices[edge];
        if (target < 0 || static_cast<std::size_t>(target) >= local_count) {
            throw std::invalid_argument(
                "BFS partition index is out of range");
        }
    }
    for (std::size_t local = 0; local < local_count; ++local) {
        const int owner = upload.owners[local];
        const int slot = upload.halo_route_slots[local];
        if (owner < 0 || static_cast<std::size_t>(owner) >= rank_count) {
            throw std::invalid_argument("BFS partition owner is out of range");
        }
        if (local < upload.owned_vertex_count) {
            if (owner != upload.partition_id || slot != -1) {
                throw std::invalid_argument(
                    "BFS owned vertex has invalid route metadata");
            }
        } else if (owner == upload.partition_id || slot < 0 ||
                   static_cast<std::size_t>(slot) >= route_total ||
                   upload.route_destination_ranks[
                       static_cast<std::size_t>(slot)] != owner) {
            throw std::invalid_argument(
                "BFS halo vertex has invalid route metadata");
        }
    }
    for (std::size_t slot = 0; slot < route_total; ++slot) {
        const int destination = upload.route_destination_ranks[slot];
        const int owner_local = upload.route_owner_local_ids[slot];
        if (destination < 0 ||
            static_cast<std::size_t>(destination) >= rank_count ||
            owner_local < 0) {
            throw std::invalid_argument("BFS route metadata is invalid");
        }
    }

    check_current_device(upload.device_id);
    SharedArrayAcquireResult acquired;
    offsets_ = upload_verified_shared_device_array(
        upload.device_id, SharedArrayKind::Offsets, upload.offsets_sha256,
        upload.topology_compatibility, upload.offsets.data(),
        upload.offsets.size() * sizeof(int), "upload shared BFS offsets",
        &acquired);
    shared_array_stats_.add(acquired);
    ++shared_array_count_;
    indices_ = upload_verified_shared_device_array(
        upload.device_id, SharedArrayKind::Indices, upload.indices_sha256,
        upload.topology_compatibility,
        upload.indices.empty() ? nullptr : upload.indices.data(),
        upload.indices.size() * sizeof(int), "upload shared BFS indices",
        &acquired);
    if (indices_) {
        shared_array_stats_.add(acquired);
        ++shared_array_count_;
    }
    owners_ = upload_verified_shared_device_array(
        upload.device_id, SharedArrayKind::Owners, upload.owners_sha256,
        upload.ownership_compatibility,
        upload.owners.empty() ? nullptr : upload.owners.data(),
        upload.owners.size() * sizeof(int), "upload shared BFS owners",
        &acquired);
    if (owners_) {
        shared_array_stats_.add(acquired);
        ++shared_array_count_;
    }
    halo_route_slots_ = upload_array(
        upload.device_id, upload.halo_route_slots,
        "upload BFS halo route slots");
    route_destination_ranks_ = upload_array(
        upload.device_id, upload.route_destination_ranks,
        "upload BFS route destination ranks");
    route_owner_local_ids_ = upload_array(
        upload.device_id, upload.route_owner_local_ids,
        "upload BFS route owner local IDs");
    outgoing_route_offsets_device_ = upload_array(
        upload.device_id, upload.outgoing_route_offsets,
        "upload BFS sparse route offsets");
    outgoing_dense_word_offsets_device_ = upload_array(
        upload.device_id, upload.outgoing_dense_word_offsets,
        "upload BFS dense route word offsets");

    view_.partition_id = upload.partition_id;
    view_.device_id = upload.device_id;
    view_.owned_vertex_count = upload.owned_vertex_count;
    view_.local_vertex_count = local_count;
    view_.adjacency_count = upload.indices.size();
    view_.offsets = offsets_
        ? static_cast<const int*>(offsets_->data())
        : nullptr;
    view_.indices = indices_
        ? static_cast<const int*>(indices_->data())
        : nullptr;
    view_.owners = owners_
        ? static_cast<const int*>(owners_->data())
        : nullptr;
    view_.halo_route_slots = halo_route_slots_
        ? static_cast<const int*>(halo_route_slots_->data())
        : nullptr;
    view_.route_destination_ranks = route_destination_ranks_
        ? static_cast<const int*>(route_destination_ranks_->data())
        : nullptr;
    view_.route_owner_local_ids = route_owner_local_ids_
        ? static_cast<const int*>(route_owner_local_ids_->data())
        : nullptr;
    view_.outgoing_dense_word_offsets_device =
        outgoing_dense_word_offsets_device_
        ? static_cast<const std::size_t*>(
            outgoing_dense_word_offsets_device_->data())
        : nullptr;
    view_.outgoing_route_offsets_device =
        static_cast<const std::size_t*>(outgoing_route_offsets_device_->data());
    view_.outgoing_route_offsets = upload.outgoing_route_offsets;
    view_.incoming_route_offsets = upload.incoming_route_offsets;
    view_.outgoing_dense_word_offsets =
        upload.outgoing_dense_word_offsets;
    view_.incoming_dense_word_offsets =
        upload.incoming_dense_word_offsets;
    resident_bytes_ = bytes(offsets_) + bytes(indices_) + bytes(owners_) +
        bytes(halo_route_slots_) + bytes(route_destination_ranks_) +
        bytes(route_owner_local_ids_) +
        bytes(outgoing_dense_word_offsets_device_) +
        bytes(outgoing_route_offsets_device_);
}

const StateKey& ResidentBfsPartitionView::state_key() const {
    return state_key_;
}

const std::string& ResidentBfsPartitionView::graph_identity_text() const {
    return graph_identity_text_;
}

const std::string& ResidentBfsPartitionView::view_signature_text() const {
    return view_signature_text_;
}

const BfsPartitionDeviceView& ResidentBfsPartitionView::view() const {
    return view_;
}

std::size_t ResidentBfsPartitionView::resident_bytes() const {
    return resident_bytes_;
}
const SharedArrayCallStats& ResidentBfsPartitionView::shared_array_stats() const {
    return shared_array_stats_;
}
std::size_t ResidentBfsPartitionView::shared_array_count() const {
    return shared_array_count_;
}
bool ResidentBfsPartitionView::cacheable() const {
    return shared_array_stats_.all_retained;
}

BfsPartitionWorkspace::BfsPartitionWorkspace(int device_id)
    : managed_owner_id_(new_managed_owner_id("bfs", "workspace", device_id)),
      device_id_(device_id) {
    if (device_id_ < 0) {
        throw std::invalid_argument("BFS workspace requires a device");
    }
}

void BfsPartitionWorkspace::ensure(const BfsPartitionDeviceView& view) {
    check_current_device(device_id_);
    outgoing_route_offsets_ = view.outgoing_route_offsets;
    incoming_route_offsets_ = view.incoming_route_offsets;
    outgoing_dense_word_offsets_ = view.outgoing_dense_word_offsets;
    incoming_dense_word_offsets_ = view.incoming_dense_word_offsets;
    const std::size_t owned = view.owned_vertex_count;
    const std::size_t outgoing = outgoing_route_offsets_.back();
    const std::size_t incoming = incoming_route_offsets_.back();
    const std::size_t ranks = outgoing_route_offsets_.size() - 1U;
    const std::size_t bitmap_words = owned / 32U +
        (owned % 32U == 0U ? 0U : 1U);
    ensure_allocation(
        device_id_, owned * sizeof(std::int64_t), &distances_);
    ensure_allocation(
        device_id_, owned * sizeof(int), &current_frontier_);
    ensure_allocation(
        device_id_, owned * sizeof(int), &next_frontier_);
    ensure_allocation(
        device_id_, owned ? sizeof(int) : 0U, &next_count_);
    ensure_allocation(
        device_id_, outgoing * sizeof(int), &send_flags_);
    ensure_allocation(
        device_id_, outgoing * sizeof(int), &send_values_);
    ensure_allocation(
        device_id_, ranks * sizeof(int), &send_counts_);
    ensure_allocation(
        device_id_, incoming * sizeof(int), &receive_values_);
    ensure_allocation(
        device_id_, bitmap_words * sizeof(std::uint32_t), &current_bitmap_);
    ensure_allocation(
        device_id_, bitmap_words * sizeof(std::uint32_t), &next_bitmap_);
    ensure_allocation(
        device_id_, outgoing_dense_word_offsets_.back() *
            sizeof(std::uint32_t),
        &dense_send_bitmap_);
    ensure_allocation(
        device_id_, incoming_dense_word_offsets_.back() *
            sizeof(std::uint32_t),
        &dense_receive_bitmap_);
}

std::int64_t* BfsPartitionWorkspace::distances() const {
    return distances_
        ? static_cast<std::int64_t*>(distances_->data())
        : nullptr;
}

void BfsPartitionWorkspace::ensure_high_degree(const BfsPartitionDeviceView& view) {
    const std::size_t capacity = std::min(view.owned_vertex_count,
        view.adjacency_count / partitioned::kBfsHighDegreeThreshold);
    if (!capacity) return;
    ensure_allocation(device_id_, capacity * sizeof(int), &high_degree_rows_);
    ensure_allocation(device_id_, 2U * sizeof(int), &high_degree_control_);
    if (!high_degree_grid_limit_) {
        int processors = 0;
        check_cuda(cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount,
            device_id_), "query BFS high-degree CTA grid limit");
        high_degree_grid_limit_ = 2 * processors;
    }
}

int* BfsPartitionWorkspace::high_degree_rows() const {
    return high_degree_rows_ ? static_cast<int*>(high_degree_rows_->data()) : nullptr;
}
int* BfsPartitionWorkspace::high_degree_control() const {
    return high_degree_control_ ? static_cast<int*>(high_degree_control_->data()) : nullptr;
}
int BfsPartitionWorkspace::high_degree_grid_limit() const { return high_degree_grid_limit_; }
std::size_t BfsPartitionWorkspace::high_degree_capacity_bytes() const {
    return bytes(high_degree_rows_) + bytes(high_degree_control_);
}

int* BfsPartitionWorkspace::current_frontier() const {
    return current_frontier_
        ? static_cast<int*>(current_frontier_->data())
        : nullptr;
}

int* BfsPartitionWorkspace::next_frontier() const {
    return next_frontier_
        ? static_cast<int*>(next_frontier_->data())
        : nullptr;
}

int* BfsPartitionWorkspace::next_count() const {
    return next_count_ ? static_cast<int*>(next_count_->data()) : nullptr;
}

int* BfsPartitionWorkspace::send_flags() const {
    return send_flags_ ? static_cast<int*>(send_flags_->data()) : nullptr;
}

int* BfsPartitionWorkspace::send_values(
    std::size_t destination_rank) const {
    if (destination_rank + 1U >= outgoing_route_offsets_.size()) {
        throw std::out_of_range("BFS destination rank is out of range");
    }
    const std::size_t begin = outgoing_route_offsets_[destination_rank];
    return send_values_
        ? static_cast<int*>(send_values_->data()) + begin
        : nullptr;
}

int* BfsPartitionWorkspace::send_counts() const {
    return send_counts_ ? static_cast<int*>(send_counts_->data()) : nullptr;
}

int* BfsPartitionWorkspace::receive_values(std::size_t source_rank) const {
    if (source_rank + 1U >= incoming_route_offsets_.size()) {
        throw std::out_of_range("BFS source rank is out of range");
    }
    const std::size_t begin = incoming_route_offsets_[source_rank];
    return receive_values_
        ? static_cast<int*>(receive_values_->data()) + begin
        : nullptr;
}

std::uint32_t* BfsPartitionWorkspace::current_bitmap() const {
    return current_bitmap_
        ? static_cast<std::uint32_t*>(current_bitmap_->data())
        : nullptr;
}

std::uint32_t* BfsPartitionWorkspace::next_bitmap() const {
    return next_bitmap_
        ? static_cast<std::uint32_t*>(next_bitmap_->data())
        : nullptr;
}

std::uint32_t* BfsPartitionWorkspace::dense_send_bitmap(
    std::size_t destination_rank) const {
    if (destination_rank + 1U >= outgoing_dense_word_offsets_.size()) {
        throw std::out_of_range("BFS dense destination rank is out of range");
    }
    const std::size_t begin =
        outgoing_dense_word_offsets_[destination_rank];
    return dense_send_bitmap_
        ? static_cast<std::uint32_t*>(dense_send_bitmap_->data()) + begin
        : nullptr;
}

std::uint32_t* BfsPartitionWorkspace::dense_receive_bitmap(
    std::size_t source_rank) const {
    if (source_rank + 1U >= incoming_dense_word_offsets_.size()) {
        throw std::out_of_range("BFS dense source rank is out of range");
    }
    const std::size_t begin = incoming_dense_word_offsets_[source_rank];
    return dense_receive_bitmap_
        ? static_cast<std::uint32_t*>(dense_receive_bitmap_->data()) + begin
        : nullptr;
}

std::size_t BfsPartitionWorkspace::dense_send_word_count(
    std::size_t destination_rank) const {
    if (destination_rank + 1U >= outgoing_dense_word_offsets_.size()) {
        throw std::out_of_range("BFS dense destination rank is out of range");
    }
    return outgoing_dense_word_offsets_[destination_rank + 1U] -
        outgoing_dense_word_offsets_[destination_rank];
}

std::size_t BfsPartitionWorkspace::dense_receive_word_count(
    std::size_t source_rank) const {
    if (source_rank + 1U >= incoming_dense_word_offsets_.size()) {
        throw std::out_of_range("BFS dense source rank is out of range");
    }
    return incoming_dense_word_offsets_[source_rank + 1U] -
        incoming_dense_word_offsets_[source_rank];
}

void BfsPartitionWorkspace::advance_frontier() {
    current_frontier_.swap(next_frontier_);
    current_bitmap_.swap(next_bitmap_);
}

std::size_t BfsPartitionWorkspace::capacity_bytes() const {
    return bytes(distances_) + bytes(current_frontier_) +
        bytes(next_frontier_) + bytes(next_count_) + bytes(send_flags_) +
        bytes(send_values_) + bytes(send_counts_) + bytes(receive_values_) +
        bytes(current_bitmap_) + bytes(next_bitmap_) +
        bytes(dense_send_bitmap_) + bytes(dense_receive_bitmap_) + high_degree_capacity_bytes();
}

bool PartitionedBfsStateRegistry::RegistryKey::operator==(
    const RegistryKey& other) const {
    return partition_id == other.partition_id && state == other.state;
}

std::size_t PartitionedBfsStateRegistry::RegistryKeyHash::operator()(
    const RegistryKey& key) const {
    std::size_t seed = StateKeyHash()(key.state);
    seed ^= std::hash<int>()(key.partition_id) +
        static_cast<std::size_t>(0x9e3779b9U) + (seed << 6U) +
        (seed >> 2U);
    return seed;
}

void PartitionedBfsStateRegistry::touch_and_trim(const RegistryKey& key) {
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

std::shared_ptr<ResidentBfsPartitionView>
PartitionedBfsStateRegistry::acquire(
    const BfsPartitionUpload& upload,
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
        std::shared_ptr<ResidentBfsPartitionView>,
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
        std::shared_ptr<ResidentBfsPartitionView> result = found->second;
        touch_and_trim(key);
        return result;
    }
    if (found != views_.end()) {
        views_.erase(found);
        workspaces_.erase(key);
    }
    std::shared_ptr<ResidentBfsPartitionView> built =
        ResidentBfsPartitionView::upload(upload);
    for (std::unordered_map<
             RegistryKey,
             std::shared_ptr<ResidentBfsPartitionView>,
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

std::shared_ptr<BfsPartitionWorkspace>
PartitionedBfsStateRegistry::workspace(
    const std::shared_ptr<ResidentBfsPartitionView>& view) {
    if (!view) throw std::invalid_argument("BFS workspace requires a view");
    if (managed_workspace_per_worker()) {
        if (!pooled_workspace_) {
            pooled_workspace_.reset(new BfsPartitionWorkspace(view->view().device_id));
        } else if (pooled_workspace_.use_count() != 1) {
            // A caller may hold another partition on this worker (serial2), or
            // a prior lease may still be live. Never alias its mutable state.
            ++managed_pool_fallbacks_;
            std::shared_ptr<BfsPartitionWorkspace> transient(new BfsPartitionWorkspace(view->view().device_id));
            transient->ensure(view->view());
            return transient;
        } else {
            ++managed_pool_reuses_;
        }
        pooled_workspace_->ensure(view->view());
        return pooled_workspace_;
    }
    if (!view->cacheable()) {
        std::shared_ptr<BfsPartitionWorkspace> transient(
            new BfsPartitionWorkspace(view->view().device_id));
        transient->ensure(view->view());
        return transient;
    }
    const RegistryKey key{view->state_key(), view->view().partition_id};
    std::shared_ptr<BfsPartitionWorkspace>& workspace = workspaces_[key];
    if (!workspace) {
        workspace.reset(new BfsPartitionWorkspace(view->view().device_id));
    }
    workspace->ensure(view->view());
    return workspace;
}

void PartitionedBfsStateRegistry::clear() {
    pooled_workspace_.reset();
    views_.clear();
    workspaces_.clear();
    lru_.clear();
}
std::size_t PartitionedBfsStateRegistry::resident_view_count() const {
    return views_.size();
}
std::size_t PartitionedBfsStateRegistry::workspace_count() const {
    return workspaces_.size() + (pooled_workspace_ ? 1U : 0U);
}

PartitionedBfsStateRegistry& partitioned_bfs_state_registry() {
    static thread_local PartitionedBfsStateRegistry registry;
    return registry;
}


void ResidentBfsPartitionView::append_managed_allocations(
    ManagedStateInventory& inventory, const ManagedOwnerRecord& owner) const {
    inventory.owner(owner);
    inventory.add(offsets_, owner.owner_id, "bfs.offsets", true);
    inventory.add(indices_, owner.owner_id, "bfs.indices", true);
    inventory.add(owners_, owner.owner_id, "bfs.owners", true);
    inventory.add(halo_route_slots_, owner.owner_id, "bfs.halo_route_slots", false);
    inventory.add(route_destination_ranks_, owner.owner_id, "bfs.route_destination_ranks", false);
    inventory.add(route_owner_local_ids_, owner.owner_id, "bfs.route_owner_local_ids", false);
    inventory.add(outgoing_dense_word_offsets_device_, owner.owner_id, "bfs.outgoing_dense_word_offsets_device", false);
    inventory.add(outgoing_route_offsets_device_, owner.owner_id, "bfs.outgoing_route_offsets_device", false);
}

void BfsPartitionWorkspace::append_managed_allocations(
    ManagedStateInventory& inventory, const ManagedOwnerRecord& owner) const {
    inventory.owner(owner);
    inventory.add(distances_, owner.owner_id, "bfs.workspace.distances");
    inventory.add(current_frontier_, owner.owner_id, "bfs.workspace.current_frontier");
    inventory.add(next_frontier_, owner.owner_id, "bfs.workspace.next_frontier");
    inventory.add(next_count_, owner.owner_id, "bfs.workspace.next_count");
    inventory.add(send_flags_, owner.owner_id, "bfs.workspace.send_flags");
    inventory.add(send_values_, owner.owner_id, "bfs.workspace.send_values");
    inventory.add(send_counts_, owner.owner_id, "bfs.workspace.send_counts");
    inventory.add(receive_values_, owner.owner_id, "bfs.workspace.receive_values");
    inventory.add(current_bitmap_, owner.owner_id, "bfs.workspace.current_bitmap");
    inventory.add(next_bitmap_, owner.owner_id, "bfs.workspace.next_bitmap");
    inventory.add(dense_send_bitmap_, owner.owner_id, "bfs.workspace.dense_send_bitmap");
    inventory.add(dense_receive_bitmap_, owner.owner_id, "bfs.workspace.dense_receive_bitmap");
    inventory.add(high_degree_rows_, owner.owner_id, "bfs.workspace.high_degree_rows");
    inventory.add(high_degree_control_, owner.owner_id, "bfs.workspace.high_degree_control");
}

void PartitionedBfsStateRegistry::append_managed_inventory(ManagedStateInventory& inventory) const {
    for (const auto& entry : views_) {
        const auto& view = entry.second;
        ManagedOwnerRecord owner;
        owner.owner_id = view->managed_owner_id();
        owner.kind = "view"; owner.algorithm = "bfs";
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
        owner.kind = "workspace"; owner.algorithm = "bfs";
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
        owner.kind = "workspace"; owner.algorithm = "bfs";
        owner.policy = "per_worker"; owner.device_id = inventory.device_id;
        owner.active_leases = pooled_workspace_.use_count() - 1U;
        pooled_workspace_->append_managed_allocations(inventory, owner);
    }
    inventory.counters["bfs_pool_reuses"] = managed_pool_reuses_;
    inventory.counters["bfs_pool_active_fallbacks"] = managed_pool_fallbacks_;
}

void PartitionedBfsStateRegistry::managed_evict(const std::set<std::string>& requested,
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
