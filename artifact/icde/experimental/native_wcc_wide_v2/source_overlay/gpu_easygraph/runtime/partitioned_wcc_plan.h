#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

struct WccPartitionUpload;

// Host-only description consumed by both the native binding and the compact
// correctness test.  Local IDs [0, owned_vertex_count) are owner state; the
// remaining IDs are immutable halo identities.
struct WccHostPartition {
    int partition_id;
    int device_id;
    std::size_t owned_vertex_count;
    std::vector<int> offsets;
    std::vector<int> indices;
    std::vector<std::uint64_t> local_to_global;
    std::vector<int> owners;
    std::vector<std::int64_t> offsets64;
    bool wide_offsets = false;
    std::size_t offset_count() const { return wide_offsets ? offsets64.size() : offsets.size(); }
    std::int64_t offset(std::size_t row) const { return wide_offsets ? offsets64[row] : offsets[row]; }
};

// One owner-to-halo route.  Values packed from source_owned_local_ids on the
// source worker are scattered to destination_halo_local_ids on the
// destination worker in the same order.
struct WccHaloRoute {
    int source_rank;
    int destination_rank;
    std::vector<int> source_owned_local_ids;
    std::vector<int> destination_halo_local_ids;

    std::size_t size() const;
};

struct PartitionedWccPlan {
    std::uint64_t vertex_count;
    std::vector<WccHostPartition> partitions;
    // Row-major [source_rank * partition_count + destination_rank].
    std::vector<WccHaloRoute> routes;
    std::uint64_t halo_value_count;
    std::vector<std::shared_ptr<const WccPartitionUpload> > prepared_uploads;
    // Optional immutable host-output index, allocated only by tiled assembly.
    // [global ID] = (owner rank << 32) | owner-local ID. GPU state is unchanged.
    std::vector<std::uint64_t> output_source_for_global;
    double output_index_build_seconds = 0.0;

    const WccHaloRoute& route(
        std::size_t source_rank, std::size_t destination_rank) const;
};

void prepare_wcc_output_gather_index(PartitionedWccPlan& plan);

// Validates exact global ownership, strict empty partitions, local CSR, and
// owner metadata, then derives deduplicated owner-directed halo routes.
PartitionedWccPlan build_partitioned_wcc_plan(
    const std::vector<WccHostPartition>& partitions,
    std::uint64_t vertex_count);

// Deterministic host oracle used for small-graph correctness validation.  It
// treats every materialized arc as an undirected relation and returns the
// minimum global vertex ID of each component.
std::vector<std::uint64_t> partitioned_wcc_reference(
    const PartitionedWccPlan& plan);

}  // namespace runtime
}  // namespace gpu_easygraph
