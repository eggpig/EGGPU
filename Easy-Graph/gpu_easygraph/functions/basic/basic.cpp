#include <vector>

#include "basic/cluster.cuh"
#include "common.h"

namespace gpu_easygraph {

int clustering(
    _IN_ const std::vector<int>& V,
    _IN_ const std::vector<int>& E,
    _IN_ bool directed,
    _OUT_ std::vector<double>& CC,
    _IN_ double* kernel_seconds
) {
    if (directed) {
        CC.clear();
        if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
        return EG_UNSUPPORTED_GRAPH;
    }

    int len_V = (int)V.size() - 1;
    int len_E = (int)E.size();
    if (len_V <= 0) {
        CC.clear();
        if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
        return EG_GPU_SUCC;
    }

    return cuda_clustering(
        V.data(),
        E.data(),
        len_V,
        len_E,
        CC,
        kernel_seconds
    );
}

} // namespace gpu_easygraph

