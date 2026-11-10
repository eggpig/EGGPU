#pragma once

#include <cuda_runtime_api.h>

#include "../../runtime/partitioned_sssp_types.h"
#include "sssp_frontier_epoch.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace gpu_easygraph {
namespace partitioned {

const std::uint64_t kSsspNoBucket = UINT64_MAX;

enum class SsspRelaxPolicy { Thread, Warp };
enum class SsspControlPolicy { Atomic, Warp };
enum class SsspAtomicMinPolicy { Cas, NativeU64 };
struct SsspKernelPolicy {
    SsspRelaxPolicy relax = SsspRelaxPolicy::Thread;
    SsspControlPolicy control = SsspControlPolicy::Atomic;
    SsspAtomicMinPolicy atomic_min = SsspAtomicMinPolicy::Cas;
};

void initialize_sssp_delta(
    double* distances,
    int* settled,
    int* r_flags,
    std::size_t owned_count,
    int source_local,
    cudaStream_t stream);

void select_sssp_bucket_frontier(
    const double* distances,
    const int* settled,
    int* frontier,
    int* frontier_count,
    std::size_t owned_count,
    std::uint64_t bucket,
    double delta,
    cudaStream_t stream,
    SsspControlPolicy policy = SsspControlPolicy::Atomic);

void reset_sssp_remote_candidates(
    double* candidates,
    std::size_t count,
    cudaStream_t stream);

// light=true processes weights <= delta and marks extracted vertices in R.
// light=false processes weights > delta from the compacted R frontier.
void relax_sssp_delta_edges(
    const int* offsets,
    const int* indices,
    const void* weights,
    const int* owners,
    const int* halo_route_slots,
    double* distances,
    const int* settled,
    const int* frontier,
    std::size_t frontier_count,
    int* next_flags,
    int* r_flags,
    double* remote_candidates,
    std::size_t owned_count,
    int partition_id,
    std::uint64_t current_bucket,
    double delta,
    bool light,
    cudaStream_t stream,
    SsspRelaxPolicy policy = SsspRelaxPolicy::Thread,
    SsspAtomicMinPolicy atomic_min_policy = SsspAtomicMinPolicy::Cas,
    gpu_easygraph::runtime::SsspWeightEncoding weight_encoding = gpu_easygraph::runtime::SsspWeightEncoding::F64,
    SsspNextFrontier next = SsspNextFrontier());

void compact_sssp_relaxation_route(
    const double* candidates,
    const int* owner_local_ids,
    void* compact,
    std::size_t capacity,
    int* compact_count,
    cudaStream_t stream,
    SsspControlPolicy policy = SsspControlPolicy::Atomic,
    gpu_easygraph::runtime::SsspWireEncoding wire_encoding = gpu_easygraph::runtime::SsspWireEncoding::F64,
    double maximum_distance = std::numeric_limits<double>::infinity());

void apply_sssp_owner_relaxations(
    const void* candidates,
    std::size_t count,
    double* distances,
    const int* settled,
    int* next_flags,
    std::size_t owned_count,
    std::uint64_t current_bucket,
    double delta,
    bool enqueue_current_bucket,
    cudaStream_t stream,
    SsspAtomicMinPolicy atomic_min_policy = SsspAtomicMinPolicy::Cas,
    gpu_easygraph::runtime::SsspWireEncoding wire_encoding = gpu_easygraph::runtime::SsspWireEncoding::F64,
    SsspNextFrontier next = SsspNextFrontier());

void compact_sssp_vertex_flags(
    const int* flags,
    int* frontier,
    std::size_t owned_count,
    int* frontier_count,
    cudaStream_t stream,
    SsspControlPolicy policy = SsspControlPolicy::Atomic);

void settle_sssp_r_vertices(
    const int* r_flags,
    int* settled,
    std::size_t owned_count,
    cudaStream_t stream);

void find_sssp_minimum_bucket(
    const double* distances,
    const int* settled,
    std::uint64_t* minimum_bucket,
    std::size_t owned_count,
    double delta,
    cudaStream_t stream,
    SsspControlPolicy policy = SsspControlPolicy::Atomic);

}  // namespace partitioned
}  // namespace gpu_easygraph
