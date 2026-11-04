#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Includes the four single-metric APIs and cpp_egmgpu_partitioned_ego_metrics.
// The fused entry returns values[metric][node]; diagnostic values_dense has
// shape (4, selected_nodes) in effective_size/efficiency/constraint/hierarchy order.
void bind_egmgpu_partitioned_structural_holes(py::module_& module);
