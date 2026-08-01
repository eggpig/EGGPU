#pragma once

#include "../../common/common.h"

py::object clustering(py::object G, py::object nodes, py::object weight);
py::object _clustering_gpu_native(py::object G);
py::object _clustering_gpu_native_dense(py::object G);
