#pragma once

#include "graph.h"
#include "directed_graph.h"
#include "operation.h"

py::object cpp_graph_from_easygraph(py::object G, bool directed);
py::object cpp_graph_from_csr_files(
    const std::string& offsets_path,
    const std::string& indices_path,
    std::int64_t num_nodes,
    std::int64_t num_entries,
    bool directed,
    bool validate,
    const std::string& weights_path,
    const std::string& weight_key,
    const py::object& undirected_projection);
py::dict cpp_graph_undirected_projection_info(const Graph& graph);
