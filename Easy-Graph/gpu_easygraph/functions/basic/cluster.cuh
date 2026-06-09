#pragma once

#include <vector>

namespace gpu_easygraph {

int cuda_clustering(
    const int* V,
    const int* E,
    int len_V,
    int len_E,
    std::vector<double>& CC,
    double* kernel_seconds
);

} // namespace gpu_easygraph

