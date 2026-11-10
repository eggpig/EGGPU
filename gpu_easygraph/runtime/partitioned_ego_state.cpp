#include "partitioned_ego_state.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
}

void check_current_device(int expected, const char* operation) {
    int current = -1;
    check_cuda(cudaGetDevice(&current), operation);
    if (current != expected) {
        throw std::runtime_error(
            "partitioned ego state must be accessed by its fixed worker");
    }
}

template <typename T>
std::shared_ptr<DeviceAllocation> copy_to_device(
    int device_id, const std::vector<T>& source, const char* operation) {
    if (source.empty()) return std::shared_ptr<DeviceAllocation>();
    std::shared_ptr<DeviceAllocation> allocation(new DeviceAllocation(
        device_id, source.size() * sizeof(T)));
    check_cuda(cudaMemcpy(
        allocation->data(), source.data(), source.size() * sizeof(T),
        cudaMemcpyHostToDevice), operation);
    return allocation;
}

std::size_t bytes(const std::shared_ptr<DeviceAllocation>& value) {
    return value ? value->bytes() : 0U;
}

void hash_combine(std::size_t* seed, std::size_t value) {
    *seed ^= value + static_cast<std::size_t>(0x9e3779b9U) +
        (*seed << 6U) + (*seed >> 2U);
}

}  // namespace

EgoPartitionDeviceUpload::EgoPartitionDeviceUpload()
    : partition_id(-1),
      device_id(-1),
      graph_identity(0U),
      generation(0U),
      ego_view_signature(0U),
      host(nullptr) {}

ResidentEgoPartitionState::ResidentEgoPartitionState(
    const EgoPartitionDeviceUpload& upload)
    : state_key_(
          upload.graph_identity,
          upload.generation,
          upload.ego_view_signature,
          DeviceGroup(upload.device_group),
          GraphLayout::Partitioned),
      semantic_key_(upload.semantic_key),
      resident_bytes_(0U) {
    std::memset(&view_, 0, sizeof(view_));
}

std::shared_ptr<ResidentEgoPartitionState>
ResidentEgoPartitionState::upload(const EgoPartitionDeviceUpload& upload) {
    std::shared_ptr<ResidentEgoPartitionState> state(
        new ResidentEgoPartitionState(upload));
    state->initialize(upload);
    return state;
}

void ResidentEgoPartitionState::initialize(
    const EgoPartitionDeviceUpload& upload) {
    if (upload.partition_id < 0 || upload.device_id < 0 ||
        upload.device_group.empty() || upload.semantic_key.empty() ||
        !upload.host || upload.host->partition_id != upload.partition_id ||
        upload.host->device_id != upload.device_id ||
        std::find(upload.device_group.begin(), upload.device_group.end(),
                  upload.device_id) == upload.device_group.end()) {
        throw std::invalid_argument("invalid partitioned ego device upload");
    }
    const EgoHostPartition& host = *upload.host;
    const std::size_t owned = host.owned_global_ids.size();
    if (host.offsets.size() != owned + 1U || host.offsets.empty() ||
        host.offsets.front() != 0 || host.offsets.back() < 0 ||
        host.degree.size() != owned || host.sum_scale.size() != owned) {
        throw std::invalid_argument("invalid partitioned ego owner arrays");
    }
    const std::size_t incidences = static_cast<std::size_t>(host.offsets.back());
    if (host.mutual.size() != incidences ||
        host.neighbor_max_scale.size() != incidences ||
        host.common_product.size() != incidences ||
        host.indirect.size() != incidences) {
        throw std::invalid_argument("invalid partitioned ego incidence arrays");
    }
    check_current_device(
        upload.device_id, "cudaGetDevice before partitioned ego upload");
    offsets_ = copy_to_device(
        upload.device_id, host.offsets, "copy ego offsets");
    mutual_ = copy_to_device(
        upload.device_id, host.mutual, "copy ego mutual values");
    neighbor_max_scale_ = copy_to_device(
        upload.device_id, host.neighbor_max_scale,
        "copy ego neighbor max scales");
    common_product_ = copy_to_device(
        upload.device_id, host.common_product, "copy ego common products");
    indirect_ = copy_to_device(
        upload.device_id, host.indirect, "copy ego indirect values");
    degree_ = copy_to_device(
        upload.device_id, host.degree, "copy ego degrees");
    sum_scale_ = copy_to_device(
        upload.device_id, host.sum_scale, "copy ego mutual scales");

    view_.partition_id = upload.partition_id;
    view_.device_id = upload.device_id;
    view_.owned_vertex_count = owned;
    view_.incidence_count = incidences;
    view_.offsets = offsets_ ? static_cast<int*>(offsets_->data()) : nullptr;
    view_.mutual = mutual_
        ? static_cast<std::uint8_t*>(mutual_->data()) : nullptr;
    view_.neighbor_max_scale = neighbor_max_scale_
        ? static_cast<std::uint8_t*>(neighbor_max_scale_->data()) : nullptr;
    view_.common_product = common_product_
        ? static_cast<std::uint64_t*>(common_product_->data()) : nullptr;
    view_.indirect = indirect_
        ? static_cast<double*>(indirect_->data()) : nullptr;
    view_.degree = degree_ ? static_cast<int*>(degree_->data()) : nullptr;
    view_.sum_scale = sum_scale_
        ? static_cast<int*>(sum_scale_->data()) : nullptr;
    resident_bytes_ = bytes(offsets_) + bytes(mutual_) +
        bytes(neighbor_max_scale_) + bytes(common_product_) +
        bytes(indirect_) + bytes(degree_) + bytes(sum_scale_);
}

const StateKey& ResidentEgoPartitionState::state_key() const {
    return state_key_;
}

const std::string& ResidentEgoPartitionState::semantic_key() const {
    return semantic_key_;
}

const EgoPartitionDeviceView& ResidentEgoPartitionState::view() const {
    return view_;
}

std::size_t ResidentEgoPartitionState::resident_bytes() const {
    return resident_bytes_;
}

EgoPartitionWorkspace::EgoPartitionWorkspace(int device_id)
    : device_id_(device_id), selected_capacity_(0U), result_capacity_(0U) {
    if (device_id_ < 0) {
        throw std::invalid_argument("ego workspace requires a CUDA device");
    }
}

void EgoPartitionWorkspace::ensure(std::size_t selected_count) {
    ensure(selected_count, selected_count);
}

void EgoPartitionWorkspace::ensure(
    std::size_t selected_count, std::size_t result_count) {
    check_current_device(device_id_, "cudaGetDevice before ego workspace growth");
    if (selected_count > selected_capacity_) {
        selected_nodes_.reset(new DeviceAllocation(
            device_id_, selected_count * sizeof(int)));
        selected_capacity_ = selected_count;
    }
    if (result_count > result_capacity_) {
        result_.reset(new DeviceAllocation(
            device_id_, result_count * sizeof(double)));
        result_capacity_ = result_count;
    }
}

int* EgoPartitionWorkspace::selected_nodes() const {
    return selected_nodes_ ? static_cast<int*>(selected_nodes_->data()) : nullptr;
}

double* EgoPartitionWorkspace::result() const {
    return result_ ? static_cast<double*>(result_->data()) : nullptr;
}

std::size_t EgoPartitionWorkspace::capacity_bytes() const {
    return bytes(selected_nodes_) + bytes(result_);
}

PartitionedEgoStateRegistry::RegistryKey::RegistryKey(
    const StateKey& state_value, int partition_value)
    : state(state_value), partition_id(partition_value) {}

bool PartitionedEgoStateRegistry::RegistryKey::operator==(
    const RegistryKey& other) const {
    return partition_id == other.partition_id && state == other.state;
}

std::size_t PartitionedEgoStateRegistry::RegistryKeyHash::operator()(
    const RegistryKey& key) const {
    std::size_t seed = StateKeyHash()(key.state);
    hash_combine(&seed, std::hash<int>()(key.partition_id));
    return seed;
}

void PartitionedEgoStateRegistry::touch_and_trim(const RegistryKey& key) {
    for (std::deque<RegistryKey>::iterator item = lru_.begin();
         item != lru_.end(); ++item) {
        if (*item == key) {
            lru_.erase(item);
            break;
        }
    }
    lru_.push_front(key);
    const std::size_t capacity = partition_state_cache_capacity();
    while (lru_.size() > capacity) {
        const RegistryKey victim = lru_.back();
        lru_.pop_back();
        states_.erase(victim);
        workspaces_.erase(victim);
    }
}

std::shared_ptr<ResidentEgoPartitionState>
PartitionedEgoStateRegistry::acquire(
    const EgoPartitionDeviceUpload& upload,
    bool* cache_hit) {
    const StateKey requested(
        upload.graph_identity,
        upload.generation,
        upload.ego_view_signature,
        DeviceGroup(upload.device_group),
        GraphLayout::Partitioned);
    const RegistryKey key(requested, upload.partition_id);
    std::unordered_map<RegistryKey,
        std::shared_ptr<ResidentEgoPartitionState>, RegistryKeyHash>::iterator
        found = states_.find(key);
    if (found != states_.end()) {
        if (found->second->semantic_key() != upload.semantic_key) {
            throw std::logic_error(
                "partitioned ego state hash collision changed semantic key");
        }
        if (cache_hit) *cache_hit = true;
        touch_and_trim(key);
        return found->second;
    }
    std::shared_ptr<ResidentEgoPartitionState> state =
        ResidentEgoPartitionState::upload(upload);
    states_.insert(std::make_pair(key, state));
    if (cache_hit) *cache_hit = false;
    touch_and_trim(key);
    return state;
}

std::shared_ptr<EgoPartitionWorkspace>
PartitionedEgoStateRegistry::workspace(
    const std::shared_ptr<ResidentEgoPartitionState>& state) {
    if (!state) throw std::invalid_argument("ego workspace requires state");
    const RegistryKey key(state->state_key(), state->view().partition_id);
    std::unordered_map<RegistryKey,
        std::shared_ptr<EgoPartitionWorkspace>, RegistryKeyHash>::iterator found =
        workspaces_.find(key);
    if (found != workspaces_.end()) {
        touch_and_trim(key);
        return found->second;
    }
    std::shared_ptr<EgoPartitionWorkspace> workspace(
        new EgoPartitionWorkspace(state->view().device_id));
    workspaces_.insert(std::make_pair(key, workspace));
    touch_and_trim(key);
    return workspace;
}

void PartitionedEgoStateRegistry::clear() {
    states_.clear();
    workspaces_.clear();
    lru_.clear();
}

std::size_t PartitionedEgoStateRegistry::resident_state_count() const {
    return states_.size();
}

std::size_t PartitionedEgoStateRegistry::workspace_count() const {
    return workspaces_.size();
}

PartitionedEgoStateRegistry& partitioned_ego_state_registry() {
    static thread_local PartitionedEgoStateRegistry registry;
    return registry;
}

}  // namespace runtime
}  // namespace gpu_easygraph
