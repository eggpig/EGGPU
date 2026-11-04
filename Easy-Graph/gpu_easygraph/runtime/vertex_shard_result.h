#pragma once

#include <cstddef>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

// Merge compact VertexShard results in canonical shard order.  Both node IDs
// and values are checked so a missing, duplicated, or reordered shard cannot
// silently attach a value to the wrong Python key.
std::vector<double> merge_ordered_vertex_shards(
    const std::vector<int>& expected_nodes,
    const std::vector<std::vector<int> >& shard_nodes,
    const std::vector<std::vector<double> >& shard_values);

// Efficiency is defined from the already-computed effective-size shard.  The
// degree vector is indexed by the frozen CSR's internal vertex identifier.
std::vector<double> normalize_effective_size_shard(
    const std::vector<int>& nodes,
    const std::vector<double>& effective_size,
    const std::vector<double>& degree);

}  // namespace runtime
}  // namespace gpu_easygraph
