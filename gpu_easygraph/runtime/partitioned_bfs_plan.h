#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

struct BfsPartitionUpload;

enum class BfsFrontierPolicy {
    Sparse,
    Dense,
    Adaptive,
};

enum class BfsFrontierKind {
    Sparse,
    Dense,
};

struct BfsFrontierMetrics {
    double active_vertex_fraction;
    double estimated_active_edge_fraction;
    double boundary_edge_fraction;

    BfsFrontierMetrics();
};

// Enter and exit thresholds are separate so an adaptive traversal does not
// oscillate when one level lies close to the crossover point.  Values are
// runtime calibration inputs; this type contains no graph-specific guess.
struct BfsFrontierThresholds {
    double dense_vertex_enter;
    double dense_vertex_exit;
    double dense_edge_enter;
    double dense_edge_exit;
    double dense_boundary_enter;

    BfsFrontierThresholds();
};

BfsFrontierPolicy parse_bfs_frontier_policy(const std::string& value);
const char* bfs_frontier_policy_name(BfsFrontierPolicy policy);
bool bfs_dense_frontier_available();
const char* bfs_frontier_kind_name(BfsFrontierKind kind);
void validate_bfs_frontier_thresholds(
    const BfsFrontierThresholds& thresholds);
BfsFrontierKind select_bfs_frontier_kind(
    BfsFrontierPolicy policy,
    const BfsFrontierMetrics& metrics,
    const BfsFrontierThresholds& thresholds,
    BfsFrontierKind previous);
std::size_t bfs_bitmap_word_count(std::size_t vertex_count);
std::vector<std::uint32_t> pack_bfs_owner_bitmap_reference(
    const std::vector<int>& owner_local_ids, std::size_t owned_vertex_count);
std::vector<int> unpack_bfs_owner_bitmap_reference(
    const std::vector<std::uint32_t>& bitmap,
    std::size_t owned_vertex_count);

struct BfsHostPartition {
    int partition_id;
    int device_id;
    std::size_t owned_vertex_count;
    std::vector<int> offsets;
    std::vector<int> indices;
    std::vector<std::uint64_t> local_to_global;
    std::vector<int> owners;
};

// A source partition sends compact owner-local IDs to the destination owner.
// Each source halo appears once in exactly one route, so setting its active
// flag deduplicates all frontier edges targeting the same remote vertex.
struct BfsOwnerRoute {
    int source_rank;
    int destination_rank;
    std::vector<int> source_halo_local_ids;
    std::vector<int> destination_owner_local_ids;

    std::size_t capacity() const;
};

struct PartitionedBfsPlan {
    std::uint64_t vertex_count;
    std::vector<BfsHostPartition> partitions;
    // Row-major [source_rank * partition_count + destination_rank].
    std::vector<BfsOwnerRoute> routes;
    std::vector<int> owner_by_global;
    std::uint64_t route_capacity;
    std::uint64_t boundary_adjacency_count;
    // Populated by the native graph binding once per immutable cached plan.
    // Active queries hold the plan, so cache eviction never invalidates a
    // descriptor while its device tasks are acquiring resident state.
    std::vector<std::shared_ptr<const BfsPartitionUpload> > prepared_uploads;

    const BfsOwnerRoute& route(
        std::size_t source_rank, std::size_t destination_rank) const;
    std::size_t owner_rank(std::uint64_t global_vertex) const;
    std::size_t owner_local(std::uint64_t global_vertex) const;
};

// Dense exchange sends one owner-local bitmap for every ordered pair of
// distinct ranks.  Offsets are expressed in uint32 bitmap words.
struct BfsDenseRouteLayout {
    std::vector<std::size_t> outgoing_word_offsets;
    std::vector<std::size_t> incoming_word_offsets;

    std::size_t outgoing_word_count() const;
    std::size_t incoming_word_count() const;
};

BfsDenseRouteLayout build_bfs_dense_route_layout(
    const PartitionedBfsPlan& plan, std::size_t rank);

// Validates local CSR and the exact global owner cover, including strict
// empty partitions, then derives owner-directed sparse-frontier routes.
PartitionedBfsPlan build_partitioned_bfs_plan(
    const std::vector<BfsHostPartition>& partitions,
    std::uint64_t vertex_count);

// Deterministic correctness oracle.  Unreachable vertices use -1, matching
// the native coordinator's internal dense representation.
std::vector<std::int64_t> partitioned_bfs_reference(
    const PartitionedBfsPlan& plan,
    std::uint64_t source);

// Deterministically assembles owner-prefix results in global vertex order.
// A wrong rank count or owner-vector length fails before a result is returned.
std::vector<std::int64_t> merge_partitioned_bfs_owner_distances(
    const PartitionedBfsPlan& plan,
    const std::vector<std::vector<std::int64_t> >& owned_distances,
    std::int64_t unvisited_value);

}  // namespace runtime
}  // namespace gpu_easygraph
