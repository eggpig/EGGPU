#pragma once
#include <cstdint>
namespace gpu_easygraph {
namespace partitioned {
struct WccOffsetPointer {
    const int* narrow;
    const std::int64_t* wide;
    WccOffsetPointer(const int* value = nullptr) : narrow(value), wide(nullptr) {}
    static WccOffsetPointer from_wide(const std::int64_t* value) {
        WccOffsetPointer result; result.wide = value; return result;
    }
    explicit operator bool() const { return narrow != nullptr || wide != nullptr; }
};
}
}
