#pragma once

#include <cstdio>

#include "err.h"

#define EG_DOUBLE_INF 1e100

#define EXIT_IF_CUDA_FAILED(condition)              \
        cuda_ret = condition;                       \
        if (cuda_ret != cudaSuccess) {              \
            std::fprintf(stderr, "EGGPU CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(static_cast<cudaError_t>(cuda_ret))); \
            goto exit;                              \
        }                                           \

#ifndef _IN_
#define _IN_
#endif

#ifndef _OUT_
#define _OUT_
#endif

#ifndef _BUFFER_
#define _BUFFER_
#endif
