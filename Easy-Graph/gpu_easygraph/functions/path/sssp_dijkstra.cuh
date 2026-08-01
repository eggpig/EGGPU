#pragma once

#include "common.h"

namespace gpu_easygraph {

int cuda_sssp_dijkstra(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const double* W,
    _IN_ const int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int target,
    _IN_ int warp_size,
    _OUT_ double* res,
    _OUT_ double* kernel_seconds = nullptr
);

int cuda_sssp_unweighted_bfs(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int target,
    _OUT_ double* res,
    _OUT_ double* kernel_seconds = nullptr
);

int cuda_sssp_bellman_ford(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const double* W,
    _IN_ const int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int target,
    _OUT_ double* res,
    _OUT_ double* kernel_seconds = nullptr
);

} // namespace gpu_easygraph
