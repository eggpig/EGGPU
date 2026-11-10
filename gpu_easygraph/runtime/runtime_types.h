#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

enum class CommunicationBackend {
    None,
    Auto,
    Nccl,
    PeerToPeer,
    HostStaging,
};

enum class ExecutionMode {
    Auto,
    SingleDevice,
    ReplicatedTaskParallel,
    PartitionedIterative,
    RequestParallel,
};

enum class GraphLayout {
    SingleDevice,
    Replicated,
    Partitioned,
};

const char* communication_backend_name(CommunicationBackend backend);
CommunicationBackend parse_communication_backend(const std::string& value);
const char* execution_mode_name(ExecutionMode mode);
ExecutionMode parse_execution_mode(const std::string& value);
const char* graph_layout_name(GraphLayout layout);

// A GraphSnapshot contains only immutable native metadata.  Python graph
// objects and pybind handles must not cross the worker-thread boundary.
class GraphSnapshot {
public:
    GraphSnapshot(
        std::uint64_t graph_identity,
        std::uint64_t generation,
        std::uint64_t topology_and_weight_signature,
        std::uint64_t vertex_count,
        std::uint64_t edge_count,
        bool directed,
        bool weighted);

    std::uint64_t graph_identity() const;
    std::uint64_t generation() const;
    std::uint64_t signature() const;
    std::uint64_t vertex_count() const;
    std::uint64_t edge_count() const;
    bool directed() const;
    bool weighted() const;

private:
    std::uint64_t graph_identity_;
    std::uint64_t generation_;
    std::uint64_t signature_;
    std::uint64_t vertex_count_;
    std::uint64_t edge_count_;
    bool directed_;
    bool weighted_;
};

class DeviceGroup {
public:
    explicit DeviceGroup(
        std::vector<int> device_ids,
        CommunicationBackend backend = CommunicationBackend::Auto);

    const std::vector<int>& device_ids() const;
    CommunicationBackend backend() const;
    std::size_t size() const;
    bool contains(int device_id) const;

    bool operator==(const DeviceGroup& other) const;
    bool operator!=(const DeviceGroup& other) const;

private:
    std::vector<int> device_ids_;
    CommunicationBackend backend_;
};

struct StateKey {
    std::uint64_t graph_identity;
    std::uint64_t generation;
    std::uint64_t view_signature;
    DeviceGroup device_group;
    GraphLayout layout;

    StateKey(
        std::uint64_t graph_identity_value,
        std::uint64_t generation_value,
        std::uint64_t view_signature_value,
        DeviceGroup device_group_value,
        GraphLayout layout_value);

    bool operator==(const StateKey& other) const;
    bool operator!=(const StateKey& other) const;
};

struct StateKeyHash {
    std::size_t operator()(const StateKey& key) const;
};

}  // namespace runtime
}  // namespace gpu_easygraph
