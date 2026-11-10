#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace gpu_easygraph {
namespace partitioned {

constexpr int kPageRankHeavyRowDegree = 32;

void initialize_owned_rank(
    double* rank, std::size_t owned_count, double value, cudaStream_t stream);

// One launch reads an owner's immutable contributions through enabled P2P
// access. The caller synchronizes all owners before and after the halo phase.
void gather_peer_page_rank_halo(
    const double* owner_contribution, const int* source_local,
    const int* destination_local, std::size_t count,
    double* contribution, cudaStream_t stream);

// Writes contribution[0:owned_count] and a single local dangling sum.
void prepare_page_rank_contribution(
    const double* rank,
    const double* out_weight,
    double* contribution,
    std::size_t owned_count,
    double* dangling_sum,
    cudaStream_t stream);

// Computes destination-owned pull rows and a local L1 residual.
// A null incoming_weights pointer denotes implicit unit weights. Both
// overloads dispatch to a specialization that omits weight loads/multiplies.
void pull_page_rank_rows(
    const int* incoming_offsets,
    const int* incoming_sources,
    const double* incoming_weights,
    const double* local_contribution,
    const double* current_rank,
    double* next_rank,
    std::size_t owned_count,
    double alpha,
    double base,
    double* residual,
    cudaStream_t stream);

void pull_page_rank_rows(
    const int* incoming_offsets, const int* incoming_sources,
    const double* incoming_weights, const double* local_contribution,
    const double* current_rank, double* next_rank, std::size_t owned_count,
    double alpha, double base, double* residual, cudaStream_t stream,
    const int* heavy_rows, std::size_t heavy_row_count);

}  // namespace partitioned
}  // namespace gpu_easygraph
