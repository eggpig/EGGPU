#pragma once
#include <cuda_runtime_api.h>
namespace gpu_easygraph {
void launch_triangle_count(const int* offsets, const int* indices,
    const int* owner_sources, int edge_begin, int edge_count,
    unsigned long long* total, cudaStream_t stream);
}
