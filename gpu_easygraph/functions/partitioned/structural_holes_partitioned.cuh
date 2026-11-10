#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace gpu_easygraph {
namespace partitioned {

enum class EgoMetric {
    EffectiveSize = 0,
    Efficiency = 1,
    Constraint = 2,
    Hierarchy = 3,
};

// Reduces immutable owner-incidence statistics into one compact value per
// selected owner-local node.  selected_nodes and result are mutable call
// workspace; every other pointer belongs to the shared resident ego state.
void evaluate_partitioned_ego_metric(
    const int* offsets,
    const std::uint8_t* mutual,
    const std::uint8_t* neighbor_max_scale,
    const std::uint64_t* common_product,
    const double* indirect,
    const int* degree,
    const int* sum_scale,
    const int* selected_nodes,
    std::size_t selected_count,
    std::size_t owned_vertex_count,
    EgoMetric metric,
    double* result,
    cudaStream_t stream);

// A single incidence scan accumulates effective size, C=sum(c), and
// L=sum(c*log(c)). Output is metric-major [4][selected_count] in the order
// effective_size, efficiency, constraint, hierarchy, requiring one D2H copy.
void evaluate_partitioned_ego_metrics(
    const int* offsets,
    const std::uint8_t* mutual,
    const std::uint8_t* neighbor_max_scale,
    const std::uint64_t* common_product,
    const double* indirect,
    const int* degree,
    const int* sum_scale,
    const int* selected_nodes,
    std::size_t selected_count,
    std::size_t owned_vertex_count,
    double* result,
    cudaStream_t stream);

}  // namespace partitioned
}  // namespace gpu_easygraph
