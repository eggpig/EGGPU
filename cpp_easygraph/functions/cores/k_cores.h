#pragma once

#include "../../common/common.h"

py::object core_decomposition(py::object G);
py::object cpu_core_decomposition(py::object G);
py::object gpu_core_decomposition(py::object G);
