#pragma once

#include "egmgpu_partition_loader.h"
#include "egmgpu_sha256.h"

#include <cstddef>
#include <string>
#include <vector>

namespace easygraph {
namespace egmgpu {

inline std::string partition_ownership_compatibility_for_arrays(
    const MultiPartitionGraph& graph,
    int partition_id,
    const std::vector<int>& devices,
    const std::string& local_to_global_sha256,
    const std::string& owners_sha256,
    std::size_t owned_vertex_count,
    std::size_t local_vertex_count) {
    std::string result = "ownership-v1|" + graph.ownership_encoding() +
        "|partition=" + std::to_string(partition_id) +
        "|owned=" + std::to_string(owned_vertex_count) +
        "|local=" + std::to_string(local_vertex_count) +
        "|local-to-global=" + local_to_global_sha256 +
        "|owners=" + owners_sha256 + "|devices=";
    for (std::size_t index = 0; index < devices.size(); ++index) {
        if (index) result += ",";
        result += std::to_string(devices[index]);
    }
    return result;
}

inline std::string partition_ownership_compatibility(
    const MultiPartitionGraph& graph,
    int partition_id,
    const std::vector<int>& devices) {
    const PartitionCSR& partition = graph.partitions().at(
        static_cast<std::size_t>(partition_id));
    return partition_ownership_compatibility_for_arrays(
        graph, partition_id, devices, partition.local_to_global_sha256,
        partition.owners_sha256,
        static_cast<std::size_t>(partition.owned_vertex_count),
        partition.local_to_global.size());
}

inline std::string partition_topology_compatibility(
    const MultiPartitionGraph& graph,
    int partition_id,
    const std::vector<int>& devices) {
    return partition_ownership_compatibility(graph, partition_id, devices) +
        "|orientation=" + graph.view_orientation() +
        "|topology=" + graph.topology_signature();
}

inline std::string partition_weight_compatibility(
    const MultiPartitionGraph& graph,
    int partition_id,
    const std::vector<int>& devices) {
    return partition_topology_compatibility(graph, partition_id, devices) +
        "|weight-key=" + graph.weight_key() +
        "|weight=" + graph.weight_signature();
}

inline std::string sha256_bytes(const void* data, std::size_t bytes) {
    Sha256 digest;
    if (bytes != 0U) {
        digest.update(static_cast<const unsigned char*>(data), bytes);
    }
    return digest.final_hex();
}

template <typename T>
std::string sha256_vector(const std::vector<T>& values) {
    return sha256_bytes(
        values.empty() ? nullptr : &values[0], values.size() * sizeof(T));
}

}  // namespace egmgpu
}  // namespace easygraph
