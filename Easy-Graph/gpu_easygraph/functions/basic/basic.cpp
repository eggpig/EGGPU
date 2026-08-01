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

int clustering_forward(
    _IN_ const std::vector<int>& forward_V,
    _IN_ const std::vector<int>& forward_E,
    _IN_ const std::vector<int>& degree,
    _OUT_ std::vector<double>& CC,
    _IN_ double* kernel_seconds
) {
    const int len_V = static_cast<int>(forward_V.size()) - 1;
    const int len_E = static_cast<int>(forward_E.size());
    if (len_V < 0 || static_cast<int>(degree.size()) != len_V) {
        CC.clear();
        if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
        return EG_GPU_DEVICE_ERR;
    }
    if (len_V == 0) {
        CC.clear();
        if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
        return EG_GPU_SUCC;
    }
    return cuda_clustering_forward(
        forward_V.data(),
        forward_E.data(),
        degree.data(),
        len_V,
        len_E,
        CC,
        kernel_seconds
    );
}

} // namespace gpu_easygraph
