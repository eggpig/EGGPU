#pragma once

#include <vector>

namespace gpu_easygraph {

int cuda_pagerank(
    const int* V,
    const int* E,
    const double* W,
    int len_V,
    int len_E,
    double alpha,
    int max_iter_num,
    double threshold,
    std::vector<double>& PR,
    double* kernel_seconds
);

} // namespace gpu_easygraph

