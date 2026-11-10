#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Bind the evidence-gated single-device dispatcher used by both SINGLE and
// REQUEST_PARALLEL.  The binding parses Python arguments on its controller
// thread, then submits exactly one native-only task to the fixed CUDA worker
// that owns the requested device.
void bind_egmgpu_single_dispatch(py::module_& module);
