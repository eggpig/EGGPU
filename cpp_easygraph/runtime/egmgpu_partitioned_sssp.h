#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_egmgpu_partitioned_sssp(py::module_& module);
