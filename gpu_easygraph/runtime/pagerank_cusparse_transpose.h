#ifndef EGMGPU_PAGERANK_CUSPARSE_TRANSPOSE_H
#define EGMGPU_PAGERANK_CUSPARSE_TRANSPOSE_H
#include "partitioned_pagerank_reference.h"

namespace gpu_easygraph { namespace runtime {
struct PageRankCuSparseTransposeMetrics {
    double metadata_and_denominator_seconds = 0.0;
    double host_csc_allocation_seconds = 0.0;
    double context_handle_stream_seconds = 0.0;
    double device_array_allocation_seconds = 0.0;
    double h2d_wall_seconds = 0.0;
    double workspace_plan_allocation_seconds = 0.0;
    double conversion_synchronized_wall_seconds = 0.0;
    double d2h_wall_seconds = 0.0;
    double device_release_seconds = 0.0;
    double total_seconds = 0.0;
    std::uint64_t workspace_bytes = 0;
    std::uint64_t temporary_explicit_device_peak_bytes = 0;
    std::uint64_t h2d_bytes = 0;
    std::uint64_t d2h_bytes = 0;
    int logical_device = -1;
    int cusparse_version = 0;
    bool conversion_performed = false;
};
// Complete host CSR -> host CSC. Temporary device storage is released before
// return. No global CUDA setting or persistent graph/device cache is modified.
std::vector<PageRankHostPartition> build_whole_contiguous_page_rank_cusparse(
    const std::vector<PageRankOutgoingPartition>& outgoing,
    std::uint64_t vertex_count, bool materialize_unit_weights, int logical_device,
    PageRankCuSparseTransposeMetrics* metrics);
}}
#endif
