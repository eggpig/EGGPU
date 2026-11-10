#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace easygraph { namespace egmgpu {
// Host-only output conversion; no graph traversal or distance recomputation.
// The native caller owns the complete input until this function returns and
// does not publish writable aliases while the GIL is released.
inline pybind11::dict finalize_bfs_int64(const pybind11::array& distances,
                                       pybind11::ssize_t count) {
    namespace py = pybind11;
    if (count < 0 || static_cast<std::uint64_t>(count) > (std::uint64_t(1) << 53)
        || distances.ndim() != 1 || distances.shape(0) != count
        || !distances.dtype().is(py::dtype::of<std::int64_t>())
        || !(distances.flags() & py::array::c_style)
        || reinterpret_cast<std::uintptr_t>(distances.data()) % alignof(std::int64_t))
        throw std::invalid_argument("BFS finalizer requires aligned contiguous native int64 distances and a matching exact FP64 dense domain");
    py::array_t<double> values(count);
    const std::int64_t* input = static_cast<const std::int64_t*>(distances.data());
    double* output = values.mutable_data();
    std::uint64_t negative = 0, invalid = 0;
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < count; ++i) {
            const std::int64_t distance = input[i];
            if (distance < 0) {
                // Preserve np.where(x < 0, +inf, x), including INT64_MIN.
                output[i] = std::numeric_limits<double>::infinity();
                ++negative;
            } else if (static_cast<std::uint64_t>(distance) >= static_cast<std::uint64_t>(count)) {
                ++invalid;
            } else {
                output[i] = static_cast<double>(distance);
            }
        }
    }
    py::dict result;
    result["version"] = "native_int64_bfs_materialize_v1";
    result["certified"] = invalid == 0;
    result["checked_vertices"] = count;
    result["negative_vertices"] = negative;
    result["invalid_nonnegative_vertices"] = invalid;
    // Never expose a partially written candidate. The Python caller retains
    // the original conversion for out-of-domain values, including its rounding.
    if (invalid == 0) result["values"] = std::move(values);
    else result["values"] = py::none();
    return result;
}
}}  // namespace easygraph::egmgpu
