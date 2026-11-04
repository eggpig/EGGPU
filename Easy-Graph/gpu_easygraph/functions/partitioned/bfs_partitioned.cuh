#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace gpu_easygraph {
namespace partitioned {

static const std::size_t kBfsHighDegreeThreshold = 1024U;

const std::int64_t kBfsUnvisitedDistance = INT64_MAX;

// Initializes owner-only distances and both source-frontier encodings.  A
// rank that does not own the source receives source_local=-1.
void initialize_bfs_frontiers(
    std::int64_t* distances,
    int* current_frontier,
    std::uint32_t* current_bitmap,
    std::size_t owned_count,
    int source_local,
    cudaStream_t stream);

// Converts the current sparse list to an owner-local bitmap when an adaptive
// traversal switches to the dense path.
void build_bfs_frontier_bitmap(
    const int* current_frontier,
    std::size_t current_count,
    std::uint32_t* current_bitmap,
    std::size_t owned_count,
    cudaStream_t stream);

// Expands only owner-local frontier entries.  Local discoveries are claimed
// with atomicCAS; remote halo targets set one deduplicating route flag.
void expand_bfs_sparse(
    const int* offsets,
    const int* indices,
    const int* owners,
    const int* halo_route_slots,
    std::int64_t* distances,
    const int* current_frontier,
    std::size_t current_count,
    int* next_frontier,
    int* next_count,
    int* send_flags,
    std::size_t owned_count,
    int partition_id,
    std::int64_t next_distance,
    cudaStream_t stream,
    bool warp_edges = false,
    const int* route_destination_ranks = nullptr,
    const int* route_owner_local_ids = nullptr,
    const std::size_t* outgoing_route_offsets = nullptr,
    int* direct_send_values = nullptr,
    int* direct_send_counts = nullptr,
    int sparse_ctas_per_row = 0,
    int* high_degree_rows = nullptr,
    int* high_degree_control = nullptr,
    int high_degree_grid_limit = 0);

// Direction-optimizing dense pull. Each unvisited owner row checks its incoming
// neighbors against an immutable current frontier. The caller currently admits
// only a validated symmetric undirected CSR, where outgoing == incoming.
// remote_frontiers reuses dense-send storage indexed by owner word offsets.
void expand_bfs_pull(
    const int* offsets, const int* indices, const int* owners,
    const int* halo_route_slots, const int* route_destination_ranks,
    const int* route_owner_local_ids, const std::size_t* owner_word_offsets,
    std::int64_t* distances, const std::uint32_t* current_bitmap,
    const std::uint32_t* remote_frontiers, std::uint32_t* next_bitmap,
    int* next_frontier, int* next_count, std::size_t owned_count,
    int partition_id, std::int64_t next_distance, cudaStream_t stream,
    int pull_lanes = 1);

// Compacts one destination route.  Output contains destination owner-local
// IDs and is therefore ready for owner-directed P2P or host-staged transfer.
void compact_bfs_route(
    const int* flags,
    const int* owner_local_ids,
    int* compact_owner_local_ids,
    std::size_t route_capacity,
    int* compact_count,
    cudaStream_t stream);

// Applies remote owner-local candidates and appends only first discoveries to
// the next sparse frontier.  Multiple source ranks may safely target a vertex.
void apply_bfs_owner_candidates(
    const int* owner_local_ids,
    std::size_t count,
    std::int64_t* distances,
    int* next_frontier,
    int* next_count,
    std::size_t owned_count,
    std::int64_t next_distance,
    cudaStream_t stream);

// Same owner claim/append operator with its candidate count resident on the
// same device. The serial-partition executor can consume all queues without
// copying per-route counts to the host between expansion and owner apply.
void apply_bfs_owner_candidates_device_count(
    const int* owner_local_ids, const int* count,
    std::size_t route_capacity, std::int64_t* distances,
    int* next_frontier, int* next_count, std::size_t owned_count,
    std::int64_t next_distance, cudaStream_t stream);

// Scans the owner-local frontier bitmap.  Local discoveries populate both
// next-frontier encodings.  Remote discoveries set bits in destination-owner
// bitmaps using the exact route metadata.
void expand_bfs_dense(
    const int* offsets,
    const int* indices,
    const int* owners,
    const int* halo_route_slots,
    const int* route_destination_ranks,
    const int* route_owner_local_ids,
    const std::size_t* outgoing_dense_word_offsets,
    std::int64_t* distances,
    const std::uint32_t* current_bitmap,
    std::uint32_t* next_bitmap,
    int* next_frontier,
    int* next_count,
    std::uint32_t* dense_send_bitmap,
    std::size_t owned_count,
    int partition_id,
    std::int64_t next_distance,
    cudaStream_t stream,
    bool warp_edges = false,
    int* persistent_remote_seen = nullptr);

// Applies one received destination-owner bitmap.  atomicCAS preserves the
// same first-discovery semantics when several source ranks set the same bit.
void apply_bfs_owner_bitmap(
    const std::uint32_t* owner_bitmap,
    std::size_t bitmap_words,
    std::int64_t* distances,
    std::uint32_t* next_bitmap,
    int* next_frontier,
    int* next_count,
    std::size_t owned_count,
    std::int64_t next_distance,
    cudaStream_t stream);

}  // namespace partitioned
}  // namespace gpu_easygraph
