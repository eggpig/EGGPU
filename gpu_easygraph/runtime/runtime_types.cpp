#include "runtime_types.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <stdexcept>
#include <utility>

namespace gpu_easygraph {
namespace runtime {
namespace {

std::string normalized_name(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(value[i]);
        if (ch == '-' || ch == '_' || std::isspace(ch)) {
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

void hash_combine(std::size_t* seed, std::size_t value) {
    *seed ^= value + static_cast<std::size_t>(0x9e3779b9U) + (*seed << 6U) +
             (*seed >> 2U);
}

}  // namespace

const char* communication_backend_name(CommunicationBackend backend) {
    switch (backend) {
        case CommunicationBackend::None:
            return "none";
        case CommunicationBackend::Auto:
            return "auto";
        case CommunicationBackend::Nccl:
            return "nccl";
        case CommunicationBackend::PeerToPeer:
            return "peer_to_peer";
        case CommunicationBackend::HostStaging:
            return "host_staging";
    }
    return "unknown";
}

CommunicationBackend parse_communication_backend(const std::string& value) {
    const std::string name = normalized_name(value);
    if (name == "none") {
        return CommunicationBackend::None;
    }
    if (name == "auto") {
        return CommunicationBackend::Auto;
    }
    if (name == "nccl") {
        return CommunicationBackend::Nccl;
    }
    if (name == "p2p" || name == "peertopeer") {
        return CommunicationBackend::PeerToPeer;
    }
    if (name == "host" || name == "hoststaging") {
        return CommunicationBackend::HostStaging;
    }
    throw std::invalid_argument("unknown EGMGPU communication backend: " + value);
}

const char* execution_mode_name(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::Auto:
            return "auto";
        case ExecutionMode::SingleDevice:
            return "single";
        case ExecutionMode::ReplicatedTaskParallel:
            return "replicated";
        case ExecutionMode::PartitionedIterative:
            return "partitioned";
        case ExecutionMode::RequestParallel:
            return "request_parallel";
    }
    return "unknown";
}

ExecutionMode parse_execution_mode(const std::string& value) {
    const std::string name = normalized_name(value);
    if (name == "auto") {
        return ExecutionMode::Auto;
    }
    if (name == "single" || name == "singledevice") {
        return ExecutionMode::SingleDevice;
    }
    if (name == "replicated" || name == "replicatedtaskparallel") {
        return ExecutionMode::ReplicatedTaskParallel;
    }
    if (name == "partitioned" || name == "partitionediterative") {
        return ExecutionMode::PartitionedIterative;
    }
    if (name == "request" || name == "requestparallel") {
        return ExecutionMode::RequestParallel;
    }
    throw std::invalid_argument("unknown EGMGPU execution mode: " + value);
}

const char* graph_layout_name(GraphLayout layout) {
    switch (layout) {
        case GraphLayout::SingleDevice:
            return "single";
        case GraphLayout::Replicated:
            return "replicated";
        case GraphLayout::Partitioned:
            return "partitioned";
    }
    return "unknown";
}

GraphSnapshot::GraphSnapshot(
    std::uint64_t graph_identity,
    std::uint64_t generation,
    std::uint64_t topology_and_weight_signature,
    std::uint64_t vertex_count,
    std::uint64_t edge_count,
    bool directed,
    bool weighted)
    : graph_identity_(graph_identity),
      generation_(generation),
      signature_(topology_and_weight_signature),
      vertex_count_(vertex_count),
      edge_count_(edge_count),
      directed_(directed),
      weighted_(weighted) {}

std::uint64_t GraphSnapshot::graph_identity() const { return graph_identity_; }
std::uint64_t GraphSnapshot::generation() const { return generation_; }
std::uint64_t GraphSnapshot::signature() const { return signature_; }
std::uint64_t GraphSnapshot::vertex_count() const { return vertex_count_; }
std::uint64_t GraphSnapshot::edge_count() const { return edge_count_; }
bool GraphSnapshot::directed() const { return directed_; }
bool GraphSnapshot::weighted() const { return weighted_; }

DeviceGroup::DeviceGroup(
    std::vector<int> device_ids,
    CommunicationBackend backend)
    : device_ids_(std::move(device_ids)), backend_(backend) {
    if (device_ids_.empty()) {
        throw std::invalid_argument("an EGMGPU device group cannot be empty");
    }
    for (std::size_t i = 0; i < device_ids_.size(); ++i) {
        if (device_ids_[i] < 0) {
            throw std::invalid_argument("EGMGPU device identifiers must be non-negative");
        }
    }
    std::sort(device_ids_.begin(), device_ids_.end());
    if (std::adjacent_find(device_ids_.begin(), device_ids_.end()) !=
        device_ids_.end()) {
        throw std::invalid_argument("an EGMGPU device group cannot contain duplicates");
    }
}

const std::vector<int>& DeviceGroup::device_ids() const { return device_ids_; }
CommunicationBackend DeviceGroup::backend() const { return backend_; }
std::size_t DeviceGroup::size() const { return device_ids_.size(); }
bool DeviceGroup::contains(int device_id) const {
    return std::binary_search(device_ids_.begin(), device_ids_.end(), device_id);
}

bool DeviceGroup::operator==(const DeviceGroup& other) const {
    return backend_ == other.backend_ && device_ids_ == other.device_ids_;
}

bool DeviceGroup::operator!=(const DeviceGroup& other) const {
    return !(*this == other);
}

StateKey::StateKey(
    std::uint64_t graph_identity_value,
    std::uint64_t generation_value,
    std::uint64_t view_signature_value,
    DeviceGroup device_group_value,
    GraphLayout layout_value)
    : graph_identity(graph_identity_value),
      generation(generation_value),
      view_signature(view_signature_value),
      device_group(std::move(device_group_value)),
      layout(layout_value) {}

bool StateKey::operator==(const StateKey& other) const {
    return graph_identity == other.graph_identity && generation == other.generation &&
           view_signature == other.view_signature &&
           device_group.device_ids() == other.device_group.device_ids() &&
           layout == other.layout;
}

bool StateKey::operator!=(const StateKey& other) const { return !(*this == other); }

std::size_t StateKeyHash::operator()(const StateKey& key) const {
    std::size_t seed = 0;
    hash_combine(&seed, std::hash<std::uint64_t>()(key.graph_identity));
    hash_combine(&seed, std::hash<std::uint64_t>()(key.generation));
    hash_combine(&seed, std::hash<std::uint64_t>()(key.view_signature));
    const std::vector<int>& devices = key.device_group.device_ids();
    for (std::size_t i = 0; i < devices.size(); ++i) {
        hash_combine(&seed, std::hash<int>()(devices[i]));
    }
    hash_combine(&seed, std::hash<int>()(static_cast<int>(key.layout)));
    return seed;
}

}  // namespace runtime
}  // namespace gpu_easygraph
