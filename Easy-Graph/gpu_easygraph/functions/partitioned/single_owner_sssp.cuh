#pragma once
#include <cuda_runtime_api.h>
#include "../../runtime/single_owner_sssp_contract.h"

namespace gpu_easygraph { namespace partitioned {
// All pointers belong to a live native worker view/workspace lease. No device
// allocation occurs here. CSR/weights were validated by the immutable producer.
// Distances are IEEE FP64, +infinity for unreachable/unrepresentable path sums.
// Completion is established before return, including the final checkpoint.
cudaError_t run_single_owner_sssp(const int* rows, const int* columns,
    const double* weights, int vertices, int edges, int source,
    void* scratch, std::size_t scratch_capacity, double* distances,
    cudaStream_t stream, runtime::SingleOwnerSsspStats* stats);
}}
