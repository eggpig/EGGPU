#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpu_easygraph {
namespace runtime {

// Source-owned outgoing CSR accepted by the host preparation step.  The
// conversion below builds destination-owned pull views without replicating the
// complete graph on any device.
struct PageRankOutgoingPartition {
    int partition_id;
    int device_id;
    std::size_t owned_vertex_count;
    std::vector<int> outgoing_offsets;
    std::vector<int> outgoing_destinations;
    std::vector<double> outgoing_weights;
    std::vector<std::uint64_t> local_to_global;
    std::vector<int> owners;
};

struct PageRankHostPartition {
    int partition_id;
    int device_id;
    std::size_t owned_vertex_count;
    std::vector<int> incoming_offsets;
    std::vector<int> incoming_sources;
    // Empty means implicit unit weights; otherwise one value per stored arc.
    std::vector<double> incoming_weights;
    std::vector<std::uint64_t> local_to_global;
    std::vector<int> owners;
    std::vector<double> owned_out_weight;
};

struct PartitionedPageRankOptions {
    double alpha;
    double tolerance;
    int max_iterations;

    PartitionedPageRankOptions();
};

struct PartitionedPageRankResult {
    std::vector<double> values;
    int iterations;
    double residual;
    bool converged;
    std::uint64_t logical_halo_values;
};

void validate_page_rank_partitions(
    const std::vector<PageRankHostPartition>& partitions,
    std::uint64_t vertex_count);

// Independent whole-graph specialization. The general builder below remains unchanged.
struct WholeContiguousTransposeTimings {
    double metadata_seconds = 0.0;
    double degree_count_seconds = 0.0;
    double prefix_and_allocation_seconds = 0.0;
    double scatter_seconds = 0.0;
    double total_seconds = 0.0;
};
// Shape, identity ownership and monotone CSR metadata only. Edge validation and
// outgoing denominators belong to the selected structure builder. No CUDA dependency.
std::vector<PageRankHostPartition> make_whole_contiguous_page_rank_metadata(
    const std::vector<PageRankOutgoingPartition>& outgoing, std::uint64_t vertex_count);

std::vector<PageRankHostPartition> build_whole_contiguous_page_rank(
    const std::vector<PageRankOutgoingPartition>& outgoing,
    std::uint64_t vertex_count, bool materialize_unit_weights = true,
    WholeContiguousTransposeTimings* timings = nullptr);

std::vector<PageRankHostPartition> build_destination_owned_page_rank(
    const std::vector<PageRankOutgoingPartition>& outgoing,
    std::uint64_t vertex_count, bool materialize_unit_weights = true);

struct PackedRouteTransposeTimings {
    double metadata_seconds = 0.0;
    double route_preparation_seconds = 0.0;
    double degree_and_halo_count_seconds = 0.0;
    double prefix_halo_sort_allocation_seconds = 0.0;
    double scatter_seconds = 0.0;
    double inner_total_seconds = 0.0;
    std::uint64_t route_temporary_bytes = 0;
    std::uint64_t count_scratch_temporary_bytes = 0;
    std::uint64_t scatter_scratch_temporary_bytes = 0;
    std::uint64_t general_scatter_temporary_payload_bytes = 0;
    std::uint64_t packed_boundary_peak_temporary_payload_bytes = 0;
    std::uint64_t route_build_peak_temporary_payload_bytes = 0;
    std::uint64_t prefix_peak_temporary_payload_bytes = 0;
    std::uint64_t scatter_temporary_payload_bytes = 0;
    std::uint64_t temporary_payload_upper_bound_bytes = 0;
};

std::vector<PageRankHostPartition> build_destination_owned_page_rank_packed(
    const std::vector<PageRankOutgoingPartition>& outgoing,
    std::uint64_t vertex_count, bool materialize_unit_weights = true,
    PackedRouteTransposeTimings* timings = nullptr);

// For CSR whose graph contract is undirected with symmetric arc weights,
// outgoing rows already represent incoming rows. Move the compatible view
// and compute only denominators; do not transpose and rediscover its halo.
std::vector<PageRankHostPartition> reuse_symmetric_page_rank_view(
    std::vector<PageRankOutgoingPartition> outgoing,
    bool materialize_unit_weights = true);

PartitionedPageRankResult partitioned_page_rank_reference(
    const std::vector<PageRankHostPartition>& partitions,
    std::uint64_t vertex_count,
    const PartitionedPageRankOptions& options = PartitionedPageRankOptions());

}  // namespace runtime
}  // namespace gpu_easygraph
