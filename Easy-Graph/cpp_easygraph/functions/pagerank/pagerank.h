#pragma once

#include "../../common/common.h"

py::object _pagerank(py::object G, double alpha, int max_iterator, double threshold, py::object weight);
py::object _pagerank_gpu_native(py::object G, double alpha, int max_iterator, double threshold, py::object weight);
py::object _pagerank_gpu_native_dense(py::object G, double alpha, int max_iterator, double threshold, py::object weight);
