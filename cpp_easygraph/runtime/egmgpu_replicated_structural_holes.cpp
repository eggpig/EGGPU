#include "egmgpu_replicated_structural_holes.h"

#ifdef EASYGRAPH_ENABLE_GPU
#include "../classes/graph.h"

#include <gpu_easygraph.h>
#include <runtime/native_runtime.h>
#include <runtime/source_shard.h>
#include <runtime/vertex_shard_result.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#endif

namespace {

#ifdef EASYGRAPH_ENABLE_GPU
enum class StructuralMetric {
    EffectiveSize,
    Efficiency,
    Constraint,
    Hierarchy,
};

struct PreparedStructuralCall {
    std::shared_ptr<CSRGraph> graph;
    std::shared_ptr<UndirectedProjection> projection;
    std::vector<int> nodes;
    std::vector<int> devices;
};

std::vector<int> structural_requested_devices(const py::object& requested) {
    const gpu_easygraph::runtime::RuntimeStatus status =
        gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running) {
        throw std::runtime_error(
            "EGMGPU native runtime must be started before cooperative execution");
    }
    std::vector<int> devices = requested.is_none()
        ? status.device_ids
        : requested.cast<std::vector<int> >();
    if (devices.empty()) {
        throw std::invalid_argument(
            "replicated VertexShard execution requires at least one CUDA device");
    }
    std::sort(devices.begin(), devices.end());
    if (std::adjacent_find(devices.begin(), devices.end()) != devices.end()) {
        throw std::invalid_argument(
            "replicated VertexShard device identifiers must be unique");
    }
    return devices;
}

std::vector<int> structural_nodes(
    const py::object& nodes_object,
    std::size_t vertex_count) {
    std::vector<int> nodes;
    if (nodes_object.is_none()) {
        nodes.resize(vertex_count);
        for (std::size_t node = 0; node < vertex_count; ++node) {
            nodes[node] = static_cast<int>(node);
        }
        return nodes;
    }

    std::unordered_set<int> seen;
    for (auto item : nodes_object) {
        const std::int64_t value = py::cast<std::int64_t>(item);
        if (value < 0 || static_cast<std::uint64_t>(value) >= vertex_count) {
            throw py::key_error(
                "node is outside the contiguous bulk-CSR vertex domain");
        }
        const int node = static_cast<int>(value);
        if (seen.insert(node).second) nodes.push_back(node);
    }
    return nodes;
}

PreparedStructuralCall prepare_structural_call(
    py::object graph_object,
    py::object nodes_object,
    py::object weight_object,
    const py::object& requested_devices) {
    if (!weight_object.is_none()) {
        throw std::invalid_argument(
            "replicated structural-hole execution currently requires weight=None");
    }
    if (!graph_object.attr("is_directed")().cast<bool>()) {
        throw std::invalid_argument(
            "replicated structural-hole execution currently requires a directed "
            "graph with a validated undirected projection");
    }

    Graph& graph = graph_object.cast<Graph&>();
    PreparedStructuralCall prepared;
    prepared.graph = graph.gen_CSR();
    prepared.projection = prepared.graph->undirected_projection;
    const std::size_t vertex_count = prepared.graph->V.empty()
        ? 0U
        : prepared.graph->V.size() - 1U;
    if (!prepared.graph->contiguous_zero_based ||
        !prepared.graph->simple_sorted_topology ||
        prepared.projection == nullptr ||
        prepared.projection->lower_V.size() != prepared.graph->V.size() ||
        prepared.projection->upper_V.size() != prepared.graph->V.size() ||
        prepared.projection->forward_V.size() != prepared.graph->V.size() ||
        prepared.projection->degree.size() != vertex_count) {
        throw std::invalid_argument(
            "replicated structural-hole execution requires a validated "
            "contiguous bulk CSR and its logical-undirected projection");
    }
    prepared.nodes = structural_nodes(nodes_object, vertex_count);
    prepared.devices = structural_requested_devices(requested_devices);
    return prepared;
}

std::vector<int> copy_structural_shard(
    const std::vector<int>& nodes,
    const gpu_easygraph::runtime::ContiguousShard& shard) {
    return std::vector<int>(
        nodes.begin() + static_cast<std::ptrdiff_t>(shard.begin),
        nodes.begin() + static_cast<std::ptrdiff_t>(shard.end));
}

std::shared_ptr<std::vector<double> > directed_unweighted_degrees(
    const std::shared_ptr<CSRGraph>& graph) {
    const std::size_t vertex_count = graph->V.empty()
        ? 0U
        : graph->V.size() - 1U;
    const std::shared_ptr<std::vector<double> > degree(
        new std::vector<double>(vertex_count, 0.0));
    for (std::size_t node = 0; node < vertex_count; ++node) {
        (*degree)[node] += static_cast<double>(
            graph->V[node + 1] - graph->V[node]);
    }
    for (std::size_t edge = 0; edge < graph->E.size(); ++edge) {
        const int target = graph->E[edge];
        if (target < 0 || static_cast<std::size_t>(target) >= vertex_count) {
            throw std::runtime_error(
                "frozen CSR contains an endpoint outside its vertex domain");
        }
        (*degree)[static_cast<std::size_t>(target)] += 1.0;
    }
    return degree;
}

int run_structural_subset(
    StructuralMetric metric,
    const std::shared_ptr<CSRGraph>& graph,
    const std::shared_ptr<UndirectedProjection>& projection,
    const std::vector<int>& nodes,
    std::vector<double>& values,
    double* kernel_seconds) {
    if (metric == StructuralMetric::EffectiveSize ||
        metric == StructuralMetric::Efficiency) {
        return gpu_easygraph::effective_size_ego_edge_statistics_subset(
            graph->V,
            graph->E,
            projection->lower_V,
            projection->lower_E,
            projection->upper_V,
            projection->upper_E,
            projection->forward_V,
            projection->forward_E,
            projection->degree,
            graph->cache_id,
            nodes,
            values,
            kernel_seconds);
    }
    if (metric == StructuralMetric::Constraint) {
        return gpu_easygraph::constraint_ego_edge_statistics_subset(
            graph->V,
            graph->E,
            projection->lower_V,
            projection->lower_E,
            projection->upper_V,
            projection->upper_E,
            projection->forward_V,
            projection->forward_E,
            projection->degree,
            graph->cache_id,
            nodes,
            values,
            kernel_seconds);
    }
    return gpu_easygraph::hierarchy_ego_edge_statistics_subset(
        graph->V,
        graph->E,
        projection->lower_V,
        projection->lower_E,
        projection->upper_V,
        projection->upper_E,
        projection->forward_V,
        projection->forward_E,
        projection->degree,
        graph->cache_id,
        nodes,
        values,
        kernel_seconds);
}

py::dict structural_result(
    const std::vector<int>& nodes,
    const std::vector<double>& values,
    double kernel_seconds,
    const std::vector<double>& per_device_kernel_seconds,
    const std::vector<int>& devices_used,
    bool return_diagnostics,
    bool dense_only) {
    if (nodes.size() != values.size()) {
        throw std::logic_error(
            "merged structural-hole values do not match requested nodes");
    }
    // This is the only Python result materialization point.  Fixed GPU workers
    // produce compact native vectors and never access Python objects.
    py::dict out;
    if (!dense_only) {
        py::dict values_by_node;
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            values_by_node[py::int_(nodes[index])] = py::float_(values[index]);
        }
        out["values"] = std::move(values_by_node);
    }
    out["kernel_seconds"] = kernel_seconds;
    out["devices_used"] = devices_used;
    out["execution_mode"] = "replicated_vertex_shard";
    if (return_diagnostics || dense_only) {
        py::array_t<int> dense_nodes(nodes.size());
        py::array_t<double> dense_values(values.size());
        std::copy(nodes.begin(), nodes.end(), dense_nodes.mutable_data());
        std::copy(values.begin(), values.end(), dense_values.mutable_data());
        out["nodes_dense"] = std::move(dense_nodes);
        out["values_dense"] = std::move(dense_values);
    }
    if (return_diagnostics) {
        out["per_device_kernel_seconds"] = per_device_kernel_seconds;
    }
    return out;
}

py::dict replicated_structural_hole(
    StructuralMetric metric,
    py::object graph_object,
    py::object nodes_object,
    py::object weight_object,
    py::object requested_devices,
    bool return_diagnostics,
    const std::string& output_format) {
    if (output_format != "mapping" && output_format != "dense") {
        throw std::invalid_argument("output_format must be 'mapping' or 'dense'");
    }
    const bool dense_only = output_format == "dense";
    PreparedStructuralCall prepared = prepare_structural_call(
        graph_object, nodes_object, weight_object, requested_devices);
    if (prepared.nodes.empty()) {
        return structural_result(
            prepared.nodes,
            std::vector<double>(),
            0.0,
            std::vector<double>(),
            std::vector<int>(),
            return_diagnostics,
            dense_only);
    }

    const std::vector<gpu_easygraph::runtime::ContiguousShard> shards =
        gpu_easygraph::runtime::balanced_contiguous_shards(
            prepared.nodes.size(), prepared.devices.size());
    prepared.devices.resize(shards.size());
    std::vector<std::vector<int> > nodes_by_device;
    nodes_by_device.reserve(shards.size());
    for (std::size_t index = 0; index < shards.size(); ++index) {
        nodes_by_device.push_back(
            copy_structural_shard(prepared.nodes, shards[index]));
    }

    const std::shared_ptr<std::vector<std::vector<double> > > partials(
        new std::vector<std::vector<double> >(shards.size()));
    const std::shared_ptr<std::vector<double> > kernel_times(
        new std::vector<double>(shards.size(), 0.0));
    const std::shared_ptr<std::vector<double> > degree(
        new std::vector<double>());

    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.reserve(shards.size());
    for (std::size_t index = 0; index < shards.size(); ++index) {
        const std::shared_ptr<CSRGraph> graph = prepared.graph;
        const std::shared_ptr<UndirectedProjection> projection =
            prepared.projection;
        const std::vector<int> shard = nodes_by_device[index];
        tasks.push_back(
            [metric, graph, projection, shard, partials, kernel_times,
             degree, index](int) {
                gpu_easygraph::DeviceCsrCacheScope cache_scope(graph->cache_id);
                const int status = run_structural_subset(
                    metric,
                    graph,
                    projection,
                    shard,
                    (*partials)[index],
                    &(*kernel_times)[index]);
                if (status != gpu_easygraph::EG_GPU_SUCC) {
                    throw std::runtime_error(
                        gpu_easygraph::err_code_detail(status));
                }
                if (metric == StructuralMetric::Efficiency) {
                    (*partials)[index] =
                        gpu_easygraph::runtime::normalize_effective_size_shard(
                            shard, (*partials)[index], *degree);
                }
            });
    }

    {
        py::gil_scoped_release release;
        if (metric == StructuralMetric::Efficiency) {
            *degree = *directed_unweighted_degrees(prepared.graph);
        }
        gpu_easygraph::runtime::runtime_manager().execute_gang(
            prepared.devices, std::move(tasks));
    }

    const std::vector<double> merged =
        gpu_easygraph::runtime::merge_ordered_vertex_shards(
            prepared.nodes, nodes_by_device, *partials);
    const double kernel_seconds = *std::max_element(
        kernel_times->begin(), kernel_times->end());
    return structural_result(
        prepared.nodes,
        merged,
        kernel_seconds,
        *kernel_times,
        prepared.devices,
        return_diagnostics,
        dense_only);
}

py::dict replicated_effective_size(
    py::object graph,
    py::object nodes,
    py::object weight,
    py::object devices,
    bool return_diagnostics,
    const std::string& output_format) {
    return replicated_structural_hole(
        StructuralMetric::EffectiveSize, graph, nodes, weight, devices,
        return_diagnostics, output_format);
}

py::dict replicated_efficiency(
    py::object graph,
    py::object nodes,
    py::object weight,
    py::object devices,
    bool return_diagnostics,
    const std::string& output_format) {
    return replicated_structural_hole(
        StructuralMetric::Efficiency, graph, nodes, weight, devices,
        return_diagnostics, output_format);
}

py::dict replicated_constraint(
    py::object graph,
    py::object nodes,
    py::object weight,
    py::object devices,
    bool return_diagnostics,
    const std::string& output_format) {
    return replicated_structural_hole(
        StructuralMetric::Constraint, graph, nodes, weight, devices,
        return_diagnostics, output_format);
}

py::dict replicated_hierarchy(
    py::object graph,
    py::object nodes,
    py::object weight,
    py::object devices,
    bool return_diagnostics,
    const std::string& output_format) {
    return replicated_structural_hole(
        StructuralMetric::Hierarchy, graph, nodes, weight, devices,
        return_diagnostics, output_format);
}
#endif

}  // namespace

void bind_egmgpu_replicated_structural_holes(py::module_& module) {
#ifdef EASYGRAPH_ENABLE_GPU
    module.def(
        "cpp_egmgpu_replicated_effective_size",
        &replicated_effective_size,
        py::arg("G"),
        py::arg("nodes") = py::none(),
        py::arg("weight") = py::none(),
        py::arg("devices") = py::none(),
        py::arg("return_diagnostics") = false,
        py::kw_only(),
        py::arg("output_format") = "mapping");
    module.def(
        "cpp_egmgpu_replicated_efficiency",
        &replicated_efficiency,
        py::arg("G"),
        py::arg("nodes") = py::none(),
        py::arg("weight") = py::none(),
        py::arg("devices") = py::none(),
        py::arg("return_diagnostics") = false,
        py::kw_only(),
        py::arg("output_format") = "mapping");
    module.def(
        "cpp_egmgpu_replicated_constraint",
        &replicated_constraint,
        py::arg("G"),
        py::arg("nodes") = py::none(),
        py::arg("weight") = py::none(),
        py::arg("devices") = py::none(),
        py::arg("return_diagnostics") = false,
        py::kw_only(),
        py::arg("output_format") = "mapping");
    module.def(
        "cpp_egmgpu_replicated_hierarchy",
        &replicated_hierarchy,
        py::arg("G"),
        py::arg("nodes") = py::none(),
        py::arg("weight") = py::none(),
        py::arg("devices") = py::none(),
        py::arg("return_diagnostics") = false,
        py::kw_only(),
        py::arg("output_format") = "mapping");
#else
    (void)module;
#endif
}
