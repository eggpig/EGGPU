#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

// Minimal owner-row view copied from one validated partition manifest.  The
// first owned_global_ids.size() entries of local_to_global are the owned rows;
// the remaining entries are one-hop halo vertices referenced by indices.
struct EgoInputPartition {
    int partition_id;
    int device_id;
    std::vector<std::uint64_t> owned_global_ids;
    std::vector<int> offsets;
    std::vector<int> indices;
    std::vector<std::uint64_t> local_to_global;

    EgoInputPartition();
};

// Borrowed equivalent used by the native manifest loader.  It prevents a
// second aggregate copy of the already immutable host CSR during plan build;
// the caller must keep the source MultiPartitionGraph alive until build ends.
struct EgoInputPartitionView {
    int partition_id;
    int device_id;
    std::size_t owned_vertex_count;
    const std::vector<int>* offsets;
    const std::vector<int>* indices;
    const std::vector<std::uint64_t>* local_to_global;

    EgoInputPartitionView();
};

// Immutable per-owner incidence statistics.  An entry in [offsets[u],
// offsets[u+1]) represents one logical-undirected edge incident to owner-local
// vertex u.  These arrays are sufficient for all four structural-hole
// functions; no full graph is replicated on a device.
struct EgoHostPartition {
    int partition_id;
    int device_id;
    std::vector<std::uint64_t> owned_global_ids;
    std::vector<int> offsets;
    std::vector<std::uint8_t> mutual;
    std::vector<std::uint8_t> neighbor_max_scale;
    std::vector<std::uint64_t> common_product;
    std::vector<double> indirect;
    std::vector<int> degree;
    std::vector<int> sum_scale;
    std::uint64_t remote_incidence_count;
    std::uint64_t unique_remote_rows;
    std::uint64_t reused_remote_rows;
    std::uint64_t remote_row_adjacency_entries;

    EgoHostPartition();
    std::size_t resident_bytes() const;
};

enum class PartitionedEgoMetric {
    EffectiveSize,
    Efficiency,
    Constraint,
    Hierarchy,
};

struct PartitionedEgoPlan {
    std::string semantic_key;
    std::uint64_t vertex_count;
    std::uint64_t directed_entry_count;
    std::uint64_t logical_edge_count;
    std::uint64_t duplicate_directed_entries_removed;
    std::vector<int> owner_partition;
    std::vector<int> owner_local_id;
    std::vector<EgoHostPartition> partitions;

    PartitionedEgoPlan();
    std::size_t host_bytes() const;
};

// Builds exact unweighted directed structural-hole statistics from the
// collectively complete owner rows.  Remote rows are deduplicated per owner
// partition and every logical edge's two-hop intersection is evaluated once.
std::shared_ptr<const PartitionedEgoPlan> build_partitioned_ego_plan(
    const std::vector<EgoInputPartition>& partitions,
    std::uint64_t vertex_count,
    const std::string& semantic_key);

std::shared_ptr<const PartitionedEgoPlan>
build_partitioned_ego_plan_from_views(
    const std::vector<EgoInputPartitionView>& partitions,
    std::uint64_t vertex_count,
    const std::string& semantic_key);

// Independent host evaluator used by unit and qualification tests.
std::vector<double> evaluate_partitioned_ego_plan_cpu(
    const PartitionedEgoPlan& plan,
    PartitionedEgoMetric metric,
    const std::vector<std::uint64_t>& global_nodes);

}  // namespace runtime
}  // namespace gpu_easygraph
