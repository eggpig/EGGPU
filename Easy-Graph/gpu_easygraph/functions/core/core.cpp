#include <string>
#include <vector>

#include "core/k_core.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::vector;

int k_core(
    _IN_ const std::vector<int>& V,
    _IN_ const std::vector<int>& E,
    _OUT_ std::vector<int>& KC,
    _IN_ double* kernel_seconds
) {
    int len_V = V.size() - 1;
    int len_E = E.size();

    KC = vector<int>(len_V, 0);
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    int r = cuda_k_core(V.data(), E.data(), len_V, len_E, KC.data(), kernel_seconds);

    return r;
}

int k_core_into(
    _IN_ const std::vector<int>& V,
    _IN_ const std::vector<int>& E,
    _OUT_ int* KC,
    _IN_ double* kernel_seconds
) {
    const int len_V = static_cast<int>(V.size()) - 1;
    const int len_E = static_cast<int>(E.size());
    if (len_V < 0 || (len_V > 0 && KC == nullptr)) {
        return EG_GPU_DEVICE_ERR;
    }
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    return cuda_k_core(V.data(), E.data(), len_V, len_E, KC, kernel_seconds);
}

int k_core_split_into(
    _IN_ const std::vector<int>& lower_V,
    _IN_ const std::vector<int>& lower_E,
    _IN_ const std::vector<int>& upper_V,
    _IN_ const std::vector<int>& upper_E,
    _IN_ const std::vector<int>& degree,
    _OUT_ int* KC,
    _IN_ double* kernel_seconds
) {
    const int len_V = static_cast<int>(degree.size());
    if (static_cast<int>(lower_V.size()) != len_V + 1 ||
        static_cast<int>(upper_V.size()) != len_V + 1 ||
        (len_V > 0 && KC == nullptr)) {
        return EG_GPU_DEVICE_ERR;
    }
    if (kernel_seconds != nullptr) *kernel_seconds = 0.0;
    return cuda_k_core_split(
        lower_V.data(),
        lower_E.data(),
        static_cast<int>(lower_E.size()),
        upper_V.data(),
        upper_E.data(),
        static_cast<int>(upper_E.size()),
        degree.data(),
        len_V,
        KC,
        kernel_seconds
    );
}

} // namespace gpu_easygraph
