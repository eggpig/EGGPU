#include "partitioned_host_plan_cache.h"

#include <functional>
#include <set>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {
namespace {

bool is_canonical_sha256(const std::string& value) {
    return value.size() == 64 &&
           value.find_first_not_of("0123456789abcdef") == std::string::npos;
}

void combine_hash(std::size_t* seed, std::size_t value) {
    // Boost-style combination is sufficient because full key equality is
    // always checked by unordered_map.
    *seed ^= value + static_cast<std::size_t>(0x9e3779b9U) +
             (*seed << 6) + (*seed >> 2);
}

}  // namespace

const char* partitioned_host_plan_algorithm_name(
    PartitionedHostPlanAlgorithm algorithm) {
    switch (algorithm) {
        case PartitionedHostPlanAlgorithm::Bfs:
            return "bfs";
        case PartitionedHostPlanAlgorithm::Wcc:
            return "wcc";
        case PartitionedHostPlanAlgorithm::Sssp:
            return "sssp";
        case PartitionedHostPlanAlgorithm::Ego:
            return "ego";
    }
    throw std::invalid_argument("unknown partitioned host-plan algorithm");
}

bool PartitionedHostPlanCacheKey::operator==(
    const PartitionedHostPlanCacheKey& other) const {
    return algorithm == other.algorithm && graph_id == other.graph_id &&
           generation == other.generation &&
           view_signature == other.view_signature &&
           topology_signature == other.topology_signature &&
           weighted == other.weighted &&
           weight_signature == other.weight_signature &&
           vertex_count == other.vertex_count &&
           adjacency_count == other.adjacency_count &&
           device_group == other.device_group &&
           plan_variant == other.plan_variant;
}

std::size_t PartitionedHostPlanCacheKeyHash::operator()(
    const PartitionedHostPlanCacheKey& key) const {
    std::size_t seed = 0;
    combine_hash(&seed, std::hash<int>()(static_cast<int>(key.algorithm)));
    combine_hash(&seed, std::hash<std::string>()(key.graph_id));
    combine_hash(&seed, std::hash<std::string>()(key.plan_variant));
    combine_hash(&seed, std::hash<std::uint64_t>()(key.generation));
    combine_hash(&seed, std::hash<std::string>()(key.view_signature));
    combine_hash(&seed, std::hash<std::string>()(key.topology_signature));
    combine_hash(&seed, std::hash<bool>()(key.weighted));
    combine_hash(&seed, std::hash<std::string>()(key.weight_signature));
    combine_hash(&seed, std::hash<std::uint64_t>()(key.vertex_count));
    combine_hash(&seed, std::hash<std::uint64_t>()(key.adjacency_count));
    for (std::size_t index = 0; index < key.device_group.size(); ++index) {
        combine_hash(&seed, std::hash<int>()(key.device_group[index]));
    }
    return seed;
}

PartitionedHostPlanCacheKey make_partitioned_host_plan_cache_key(
    PartitionedHostPlanAlgorithm algorithm,
    const std::string& graph_id,
    std::uint64_t generation,
    const std::string& view_signature,
    const std::string& topology_signature,
    bool weighted,
    const std::string& weight_signature,
    std::uint64_t vertex_count,
    std::uint64_t adjacency_count,
    const std::vector<int>& requested_device_group,
    const std::string& plan_variant) {
    // Validate the enum even though it is strongly typed; callers may obtain
    // it through an unchecked cast in native code.
    (void)partitioned_host_plan_algorithm_name(algorithm);
    if (graph_id.empty()) {
        throw std::invalid_argument(
            "partitioned host-plan cache requires a graph id");
    }
    if (!is_canonical_sha256(view_signature) ||
        !is_canonical_sha256(topology_signature)) {
        throw std::invalid_argument(
            "partitioned host-plan cache requires validated manifest "
            "view and topology signatures");
    }
    if ((weighted && !is_canonical_sha256(weight_signature)) ||
        (!weighted && !weight_signature.empty())) {
        throw std::invalid_argument(
            "partitioned host-plan cache weight signature is inconsistent "
            "with the validated graph view");
    }
    if (vertex_count == 0) {
        throw std::invalid_argument(
            "partitioned host-plan cache requires a nonempty graph");
    }
    if (requested_device_group.empty()) {
        throw std::invalid_argument(
            "partitioned host-plan cache requires a device group");
    }

    std::vector<int> canonical_devices(requested_device_group);
    for (std::size_t index = 0; index < canonical_devices.size(); ++index) {
        if (canonical_devices[index] < 0) {
            throw std::invalid_argument(
                "partitioned host-plan cache device IDs must be nonnegative");
        }
    }
    std::sort(canonical_devices.begin(), canonical_devices.end());
    if (std::adjacent_find(
            canonical_devices.begin(), canonical_devices.end()) !=
        canonical_devices.end()) {
        throw std::invalid_argument(
            "partitioned host-plan cache device IDs must be distinct");
    }

    PartitionedHostPlanCacheKey key;
    key.algorithm = algorithm;
    key.graph_id = graph_id;
    key.generation = generation;
    key.view_signature = view_signature;
    key.topology_signature = topology_signature;
    key.weighted = weighted;
    key.weight_signature = weight_signature;
    key.vertex_count = vertex_count;
    key.adjacency_count = adjacency_count;
    key.device_group = canonical_devices;
    key.plan_variant = plan_variant;
    return key;
}

}  // namespace runtime
}  // namespace gpu_easygraph
