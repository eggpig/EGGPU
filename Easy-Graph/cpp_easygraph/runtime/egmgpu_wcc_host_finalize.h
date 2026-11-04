#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace easygraph { namespace egmgpu {
// This routine never computes components. It certifies an already complete
// label vector and materializes one fresh owning public-value candidate.
inline pybind11::dict finalize_wcc_uint64(const pybind11::array& labels,
                                         pybind11::ssize_t count) {
    namespace py = pybind11;
    if (count < 0 || static_cast<std::uint64_t>(count) > (std::uint64_t(1) << 53)
        || labels.ndim() != 1 || labels.shape(0) != count
        || !labels.dtype().is(py::dtype::of<std::uint64_t>())
        || !(labels.flags() & py::array::c_style)
        || reinterpret_cast<std::uintptr_t>(labels.data()) % alignof(std::uint64_t))
        throw std::invalid_argument("WCC finalizer requires aligned contiguous native uint64 labels in an exact FP64 dense domain");
    py::array_t<double> values(count);
    const std::uint64_t* input = static_cast<const std::uint64_t*>(labels.data());
    double* output = values.mutable_data();
    bool canonical = true;
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < count; ++i) {
            const std::uint64_t label = input[i];
            // Domain check precedes the only indirect read, including when a
            // preceding element already failed the canonical predicate.
            if (label >= static_cast<std::uint64_t>(count))
                throw std::invalid_argument("WCC component labels must be exact dense indices");
            if (label > static_cast<std::uint64_t>(i) || input[label] != label)
                canonical = false;
            output[i] = static_cast<double>(label);
        }
    }
    py::dict result;
    result["version"] = "native_uint64_canonical_materialize_v1";
    result["certified"] = canonical;
    result["checked_vertices"] = count;
    // A noncanonical candidate is never published by the Python integration.
    result["values"] = std::move(values);
    return result;
}
}}  // namespace easygraph::egmgpu
