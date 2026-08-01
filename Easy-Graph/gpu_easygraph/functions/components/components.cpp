#include <vector>

#include "components/connected.cuh"
#include "common.h"

namespace gpu_easygraph {

int connected_components(
    _IN_ const std::vector<int>& V,
    _IN_ const std::vector<int>& E,
    _IN_ bool directed,
    _OUT_ std::vector<int>& labels,
    _IN_ double* kernel_seconds
) {
    if (directed) {
        int len_V = (int)V.size() - 1;
        int len_E = (int)E.size();
        if (len_V <= 0) {
            labels.clear();
            if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
            return EG_GPU_SUCC;
        }
        return cuda_strongly_connected_components(
            V.data(),
            E.data(),
            len_V,
            len_E,
            labels,
            kernel_seconds
        );
    }

    int len_V = (int)V.size() - 1;
    int len_E = (int)E.size();
    if (len_V <= 0) {
        labels.clear();
        if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
        return EG_GPU_SUCC;
    }

    return cuda_connected_components(
        V.data(),
        E.data(),
        len_V,
        len_E,
        labels,
        kernel_seconds
    );
}

} // namespace gpu_easygraph
