#pragma once

#include "err.h"
#include "adaptive_transfer_policy.h"

namespace gpu_easygraph {

struct DeviceCsrView {
    int* d_V = nullptr;
    int* d_E = nullptr;
    double* d_W = nullptr;
    bool structure_changed = true;
    bool weights_changed = true;
    HostCsrStats stats;
};

int acquire_device_csr(
    const int* V,
    const int* E,
    const double* W,
    int len_V,
    int len_E,
    bool require_weights,
    DeviceCsrView* out
);

void reset_device_csr_cache();

} // namespace gpu_easygraph
