#include "egmgpu_replicated_centrality.h"

#ifdef EASYGRAPH_ENABLE_GPU
#include "../classes/graph.h"
#include "../common/utils.h"

#include <gpu_easygraph.h>
#include <runtime/native_runtime.h>
#include <runtime/result_merge.h>
#include <runtime/source_shard.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#endif

namespace {

#ifdef EASYGRAPH_ENABLE_GPU
struct PreparedCentralityCall {
    std::shared_ptr<CSRGraph> graph;
    std::shared_ptr<std::vector<double> > weights;
    std::vector<int> sources;
    std::vector<int> devices;
    bool unweighted;
    bool directed;
};

std::vector<int> parse_requested_devices(const py::object& requested) {
    gpu_easygraph::runtime::RuntimeStatus status =
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
            "replicated execution requires at least one CUDA device");
    }
    std::sort(devices.begin(), devices.end());
    if (std::adjacent_find(devices.begin(), devices.end()) != devices.end()) {
        throw std::invalid_argument(
            "replicated execution device identifiers must be unique");
    }
    return devices;
}

PreparedCentralityCall prepare_centrality_call(
    py::object graph_object,
    py::object sources_object,
    py::object weight_object,
    const py::object& requested_devices) {
    Graph& graph = graph_object.cast<Graph&>();
    PreparedCentralityCall prepared;
    prepared.unweighted = weight_object.is_none();
    if (prepared.unweighted) {
        prepared.graph = graph.gen_CSR();
    } else {
        const std::string weight_key = weight_to_string(weight_object);
        prepared.graph = graph.gen_CSR(weight_key);
        const std::unordered_map<
            std::string,
            std::shared_ptr<std::vector<double> > >::const_iterator weight_it =
            prepared.graph->W_map.find(weight_key);
        if (weight_it == prepared.graph->W_map.end()) {
            throw std::runtime_error(
                "weighted CSR preparation did not materialize the requested weights");
        }
        prepared.weights = weight_it->second;
    }
    prepared.sources = *graph.gen_CSR_sources(sources_object);
    prepared.devices = parse_requested_devices(requested_devices);
    prepared.directed = graph_object.attr("is_directed")().cast<bool>();
    return prepared;
}

std::vector<int> shard_sources(
    const std::vector<int>& sources,
    const gpu_easygraph::runtime::ContiguousShard& shard) {
    return std::vector<int>(
        sources.begin() + static_cast<std::ptrdiff_t>(shard.begin),
        sources.begin() + static_cast<std::ptrdiff_t>(shard.end));
}

py::dict dense_result(
    const std::vector<double>& values,
    double kernel_seconds,
    const std::vector<double>& per_device_kernel_seconds,
    const std::vector<int>& devices_used) {
    py::array_t<double> result(values.size());
    std::copy(values.begin(), values.end(), result.mutable_data());
    py::dict out;
    out["values_dense"] = result;
    out["kernel_seconds"] = kernel_seconds;
    out["per_device_kernel_seconds"] = per_device_kernel_seconds;
    out["devices_used"] = devices_used;
    return out;
}

py::object replicated_betweenness(
    py::object graph_object,
    py::object sources_object,
    py::object weight_object,
    bool normalized,
    bool endpoints,
    py::object requested_devices) {
    PreparedCentralityCall prepared = prepare_centrality_call(
        graph_object, sources_object, weight_object, requested_devices);
    const std::size_t vertex_count = prepared.graph->V.empty()
        ? 0U
        : prepared.graph->V.size() - 1U;
    std::vector<double> merged(vertex_count, 0.0);
    if (prepared.sources.empty()) {
        return dense_result(
            merged, 0.0, std::vector<double>(), std::vector<int>());
    }

    const std::vector<gpu_easygraph::runtime::ContiguousShard> shards =
        gpu_easygraph::runtime::balanced_contiguous_shards(
            prepared.sources.size(), prepared.devices.size());
    prepared.devices.resize(shards.size());
    std::vector<std::vector<int> > sources_by_device;
    sources_by_device.reserve(shards.size());
    for (std::size_t i = 0; i < shards.size(); ++i) {
        sources_by_device.push_back(shard_sources(prepared.sources, shards[i]));
    }

    const std::shared_ptr<std::vector<std::vector<double> > > partials(
        new std::vector<std::vector<double> >(shards.size()));
    const std::shared_ptr<std::vector<double> > kernel_times(
        new std::vector<double>(shards.size(), 0.0));
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.reserve(shards.size());
    for (std::size_t i = 0; i < shards.size(); ++i) {
        const std::shared_ptr<CSRGraph> graph = prepared.graph;
        const std::shared_ptr<std::vector<double> > weights = prepared.weights;
        const bool unweighted = prepared.unweighted;
        const bool directed = prepared.directed;
        const std::vector<int> shard = sources_by_device[i];
        tasks.push_back(
            [graph, weights, unweighted, directed, normalized, endpoints,
             shard, partials, kernel_times, i](int) {
                gpu_easygraph::DeviceCsrCacheScope cache_scope(graph->cache_id);
                const std::vector<double>& edge_weights =
                    unweighted ? graph->unweighted_W : *weights;
                const int status = gpu_easygraph::betweenness_centrality(
                    graph->V,
                    graph->E,
                    edge_weights,
                    shard,
                    directed,
                    normalized,
                    endpoints,
                    unweighted,
                    (*partials)[i],
                    &(*kernel_times)[i]);
                if (status != gpu_easygraph::EG_GPU_SUCC) {
                    throw std::runtime_error(gpu_easygraph::err_code_detail(status));
                }
            });
    }

    {
        py::gil_scoped_release release;
        gpu_easygraph::runtime::runtime_manager().execute_gang(
            prepared.devices, std::move(tasks));
    }

    // Device order is canonicalized before sharding.  Keeping that order here
    // makes the floating-point reduction deterministic across runs.  Each
    // low-level BC shard already applies the full-graph normalization scale,
    // so summing scaled partial contributions is algebraically equivalent to
    // scaling the complete source sum once.
    merged = gpu_easygraph::runtime::deterministic_dense_sum(
        *partials, vertex_count);
    const double kernel_seconds = *std::max_element(
        kernel_times->begin(), kernel_times->end());
    return dense_result(
        merged, kernel_seconds, *kernel_times, prepared.devices);
}

py::object replicated_closeness(
    py::object graph_object,
    py::object sources_object,
    py::object weight_object,
    py::object requested_devices) {
    PreparedCentralityCall prepared = prepare_centrality_call(
        graph_object, sources_object, weight_object, requested_devices);
    if (prepared.sources.empty()) {
        return dense_result(
            std::vector<double>(), 0.0, std::vector<double>(),
            std::vector<int>());
    }

    const std::vector<gpu_easygraph::runtime::ContiguousShard> shards =
        gpu_easygraph::runtime::balanced_contiguous_shards(
            prepared.sources.size(), prepared.devices.size());
    prepared.devices.resize(shards.size());
    std::vector<std::vector<int> > sources_by_device;
    sources_by_device.reserve(shards.size());
    for (std::size_t i = 0; i < shards.size(); ++i) {
        sources_by_device.push_back(shard_sources(prepared.sources, shards[i]));
    }

    const std::shared_ptr<std::vector<std::vector<double> > > partials(
        new std::vector<std::vector<double> >(shards.size()));
    const std::shared_ptr<std::vector<double> > kernel_times(
        new std::vector<double>(shards.size(), 0.0));
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.reserve(shards.size());
    for (std::size_t i = 0; i < shards.size(); ++i) {
        const std::shared_ptr<CSRGraph> graph = prepared.graph;
        const std::shared_ptr<std::vector<double> > weights = prepared.weights;
        const bool unweighted = prepared.unweighted;
        const std::vector<int> shard = sources_by_device[i];
        tasks.push_back(
            [graph, weights, unweighted, shard, partials, kernel_times, i](int) {
                gpu_easygraph::DeviceCsrCacheScope cache_scope(graph->cache_id);
                const std::vector<double>& edge_weights =
                    unweighted ? graph->unweighted_W : *weights;
                const int status = gpu_easygraph::closeness_centrality(
                    graph->V,
                    graph->E,
                    edge_weights,
                    shard,
                    unweighted,
                    (*partials)[i],
                    &(*kernel_times)[i]);
                if (status != gpu_easygraph::EG_GPU_SUCC) {
                    throw std::runtime_error(gpu_easygraph::err_code_detail(status));
                }
            });
    }

    {
        py::gil_scoped_release release;
        gpu_easygraph::runtime::runtime_manager().execute_gang(
            prepared.devices, std::move(tasks));
    }

    std::vector<double> merged = gpu_easygraph::runtime::concatenate_in_order(
        *partials, prepared.sources.size());
    const double kernel_seconds = *std::max_element(
        kernel_times->begin(), kernel_times->end());
    return dense_result(
        merged, kernel_seconds, *kernel_times, prepared.devices);
}
#endif

}  // namespace

void bind_egmgpu_replicated_centrality(py::module_& module) {
#ifdef EASYGRAPH_ENABLE_GPU
    module.def(
        "cpp_egmgpu_replicated_betweenness",
        &replicated_betweenness,
        py::arg("G"),
        py::arg("sources") = py::none(),
        py::arg("weight") = py::none(),
        py::arg("normalized") = true,
        py::arg("endpoints") = false,
        py::arg("devices") = py::none());
    module.def(
        "cpp_egmgpu_replicated_closeness",
        &replicated_closeness,
        py::arg("G"),
        py::arg("sources") = py::none(),
        py::arg("weight") = py::none(),
        py::arg("devices") = py::none());
#else
    (void)module;
#endif
}
