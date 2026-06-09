#pragma once

#include "graph.h"
#include "directed_graph.h"
#include "operation.h"

py::object cpp_graph_from_easygraph(py::object G, bool directed);
