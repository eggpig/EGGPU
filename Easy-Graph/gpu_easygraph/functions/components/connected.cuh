#pragma once

#include <vector>

namespace gpu_easygraph {

int cuda_connected_components(
    const int* V,
    const int* E,
    int len_V,
    int len_E,
    std::vector<int>& labels,
    double* kernel_seconds
);

int cuda_strongly_connected_components(
    const int* V,
    const int* E,
    int len_V,
    int len_E,
    std::vector<int>& labels,
    double* kernel_seconds
);

} // namespace gpu_easygraph
