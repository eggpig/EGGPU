#pragma once

#include <vector>

namespace gpu_easygraph {

int cuda_mst(
    const int* V,
    const int* E,
    const double* W,
    int len_V,
    int len_E,
    std::vector<int>& mst_src,
    std::vector<int>& mst_dst,
    std::vector<double>& mst_weight,
    double* kernel_seconds
);

} // namespace gpu_easygraph

