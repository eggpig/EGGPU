#pragma once

#include <cuda_runtime_api.h>
#include "../../runtime/wcc_offset_pointer.h"

#include <cstddef>
#include <cstdint>

namespace gpu_easygraph {
namespace partitioned {

void initialize_wcc_labels(
    const std::uint64_t* local_to_global,
    std::uint64_t* labels,
    std::size_t local_count,
    cudaStream_t stream);

void clear_wcc_changed(int* changed, cudaStream_t stream);

// Per-query local forest: int32 owner-local parents, no global parent array.
void initialize_wcc_local_parents(int* parents, std::size_t owned_count,
                                cudaStream_t stream);
void union_wcc_local_edges(WccOffsetPointer offsets, const int* indices,
                          int* parents, std::size_t owned_count,
                          cudaStream_t stream, bool unique_edges = false);
void compress_wcc_local_parents(int* parents, std::size_t owned_count,
                               cudaStream_t stream);
void project_wcc_component_labels(const int* parents, std::uint64_t* labels,
                                 std::size_t owned_count, cudaStream_t stream);
void combine_wcc_boundary_components(
    const int* sources, const int* neighbors, std::size_t edge_count,
    const int* parents, std::uint64_t* labels, int* changed,
    cudaStream_t stream);

void combine_wcc_boundary_components_csr(
    WccOffsetPointer offsets, const int* indices, std::size_t owned_count,
    const int* parents, std::uint64_t* labels, int* changed,
    cudaStream_t stream);

// Hooks only edges whose endpoint is owned by this partition, then performs
// bounded path compression over owner-local labels.
void local_wcc_hook_compress(
    WccOffsetPointer offsets,
    const int* indices,
    const int* owners,
    const std::uint64_t* local_to_global,
    std::uint64_t* labels,
    std::size_t owned_count,
    int partition_id,
    int* changed,
    cudaStream_t stream);

// Applies owner-side min-combine from refreshed remote halo values, followed
// by the same owner-local path compression.
void boundary_wcc_hook_compress(
    WccOffsetPointer offsets,
    const int* indices,
    const int* owners,
    const std::uint64_t* local_to_global,
    std::uint64_t* labels,
    std::size_t owned_count,
    int partition_id,
    int* changed,
    cudaStream_t stream);

void pack_wcc_owner_labels(
    const std::uint64_t* labels,
    const int* owner_local_ids,
    std::uint64_t* packed,
    std::size_t count,
    cudaStream_t stream);

void scatter_wcc_halo_labels(
    const std::uint64_t* packed,
    const int* halo_local_ids,
    std::uint64_t* labels,
    std::size_t count,
    cudaStream_t stream);

}  // namespace partitioned
}  // namespace gpu_easygraph
