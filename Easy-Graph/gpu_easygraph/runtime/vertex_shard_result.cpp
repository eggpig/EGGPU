#include "vertex_shard_result.h"

#include <limits>
#include <stdexcept>

namespace gpu_easygraph {
namespace runtime {

std::vector<double> merge_ordered_vertex_shards(
    const std::vector<int>& expected_nodes,
    const std::vector<std::vector<int> >& shard_nodes,
    const std::vector<std::vector<double> >& shard_values) {
    if (shard_nodes.size() != shard_values.size()) {
        throw std::invalid_argument(
            "VertexShard node/value container counts differ");
    }
    std::vector<double> merged;
    merged.reserve(expected_nodes.size());
    std::size_t cursor = 0;
    for (std::size_t shard = 0; shard < shard_nodes.size(); ++shard) {
        if (shard_nodes[shard].size() != shard_values[shard].size()) {
            throw std::invalid_argument(
                "VertexShard node and value lengths differ");
        }
        for (std::size_t index = 0; index < shard_nodes[shard].size(); ++index) {
            if (cursor >= expected_nodes.size() ||
                shard_nodes[shard][index] != expected_nodes[cursor]) {
                throw std::invalid_argument(
                    "VertexShard nodes do not match canonical request order");
            }
            merged.push_back(shard_values[shard][index]);
            ++cursor;
        }
    }
    if (cursor != expected_nodes.size()) {
        throw std::invalid_argument(
            "VertexShard results do not cover every requested node");
    }
    return merged;
}

std::vector<double> normalize_effective_size_shard(
    const std::vector<int>& nodes,
    const std::vector<double>& effective_size,
    const std::vector<double>& degree) {
    if (nodes.size() != effective_size.size()) {
        throw std::invalid_argument(
            "effective-size result length differs from its node shard");
    }
    std::vector<double> efficiency(nodes.size());
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const int node = nodes[index];
        if (node < 0 || static_cast<std::size_t>(node) >= degree.size()) {
            throw std::out_of_range(
                "VertexShard node is outside the degree vector");
        }
        efficiency[index] = degree[static_cast<std::size_t>(node)] == 0.0
            ? std::numeric_limits<double>::quiet_NaN()
            : effective_size[index] /
                degree[static_cast<std::size_t>(node)];
    }
    return efficiency;
}

}  // namespace runtime
}  // namespace gpu_easygraph
