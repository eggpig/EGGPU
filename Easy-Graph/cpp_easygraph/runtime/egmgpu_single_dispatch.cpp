#include "egmgpu_single_dispatch.h"

#include <pybind11/numpy.h>

#ifdef EASYGRAPH_ENABLE_GPU
#include "../classes/graph.h"

#include <gpu_easygraph.h>
#include <runtime/native_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#endif

namespace {

#ifdef EASYGRAPH_ENABLE_GPU

enum class SingleOperation {
    PageRank,
    Mst,
    Lcc,
    Wcc,
    Scc,
    Bfs,
    Dijkstra,
    BellmanFord,
    Sssp,
    KCore,
    Betweenness,
    Closeness,
    EffectiveSize,
    Efficiency,
    Constraint,
    Hierarchy,
};

struct PreparedSingleCall {
    SingleOperation operation;
    std::shared_ptr<CSRGraph> graph;
    std::shared_ptr<std::vector<double> > weights;
    std::vector<int> sources;
    std::vector<int> nodes;
    int device = -1;
    int target = -1;
    bool directed = false;
    bool weighted = false;
    bool normalized = true;
    bool endpoints = false;
    bool scalar_nodes_result = false;
    double alpha = 0.85;
    int max_iter = 200;
    double tolerance = 1.0e-6;
};

struct SingleNativeResult {
    std::vector<double> values;
    std::vector<int> integer_values;
    std::vector<int> nodes;
    std::vector<int> mst_sources;
    std::vector<int> mst_targets;
    std::vector<double> mst_weights;
    double kernel_seconds = 0.0;
};

SingleOperation parse_operation(const std::string& name) {
    if (name == "pagerank") return SingleOperation::PageRank;
    if (name == "mst") return SingleOperation::Mst;
    if (name == "lcc") return SingleOperation::Lcc;
    if (name == "wcc") return SingleOperation::Wcc;
    if (name == "scc") return SingleOperation::Scc;
    if (name == "bfs") return SingleOperation::Bfs;
    if (name == "dijkstra") return SingleOperation::Dijkstra;
    if (name == "bellman_ford") return SingleOperation::BellmanFord;
    if (name == "sssp") return SingleOperation::Sssp;
    if (name == "kcore") return SingleOperation::KCore;
    if (name == "bc") return SingleOperation::Betweenness;
    if (name == "closeness") return SingleOperation::Closeness;
    if (name == "effective_size") return SingleOperation::EffectiveSize;
    if (name == "efficiency") return SingleOperation::Efficiency;
    if (name == "constraint") return SingleOperation::Constraint;
    if (name == "hierarchy") return SingleOperation::Hierarchy;
    throw std::invalid_argument("unknown EGMGPU single-device operation: " + name);
}

bool is_structural(SingleOperation operation) {
    return operation == SingleOperation::EffectiveSize ||
           operation == SingleOperation::Efficiency ||
           operation == SingleOperation::Constraint ||
           operation == SingleOperation::Hierarchy;
}

template <typename T>
T parameter_or(const py::dict& parameters, const char* name, const T& fallback) {
    const py::str key(name);
    if (!parameters.contains(key) || parameters[key].is_none()) return fallback;
    return parameters[key].cast<T>();
}

py::object parameter_object(const py::dict& parameters, const char* name) {
    const py::str key(name);
    return parameters.contains(key)
        ? py::reinterpret_borrow<py::object>(parameters[key])
        : py::none();
}

std::vector<int> all_nodes(std::size_t count) {
    std::vector<int> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = static_cast<int>(index);
    }
    return values;
}

std::vector<int> parse_nodes(
    const py::object& object,
    std::size_t vertex_count,
    bool* scalar) {
    if (scalar != nullptr) *scalar = false;
    if (object.is_none()) return all_nodes(vertex_count);

    std::vector<int> nodes;
    if (py::isinstance<py::int_>(object)) {
        const std::int64_t value = object.cast<std::int64_t>();
        if (value < 0 || static_cast<std::uint64_t>(value) >= vertex_count) {
            throw py::key_error("node is outside the contiguous bulk-CSR domain");
        }
        nodes.push_back(static_cast<int>(value));
        if (scalar != nullptr) *scalar = true;
        return nodes;
    }

    for (const py::handle item : object) {
        const std::int64_t value = py::cast<std::int64_t>(item);
        if (value < 0 || static_cast<std::uint64_t>(value) >= vertex_count) {
            throw py::key_error("node is outside the contiguous bulk-CSR domain");
        }
        nodes.push_back(static_cast<int>(value));
    }
    return nodes;
}

std::vector<int> parse_sources(
    const py::object& object,
    std::size_t vertex_count,
    bool default_all) {
    if (object.is_none()) {
        return default_all ? all_nodes(vertex_count) : std::vector<int>();
    }
    std::vector<int> sources;
    for (const py::handle item : object) {
        const std::int64_t value = py::cast<std::int64_t>(item);
        if (value < 0 || static_cast<std::uint64_t>(value) >= vertex_count) {
            throw py::key_error("source is outside the contiguous bulk-CSR domain");
        }
        sources.push_back(static_cast<int>(value));
    }
    return sources;
}

int parse_target(const py::object& object, std::size_t vertex_count) {
    if (object.is_none()) return -1;
    const std::int64_t value = object.cast<std::int64_t>();
    if (value < 0 || static_cast<std::uint64_t>(value) >= vertex_count) {
        throw py::key_error("target is outside the contiguous bulk-CSR domain");
    }
    return static_cast<int>(value);
}

void validate_fixed_worker(int device) {
    const gpu_easygraph::runtime::RuntimeStatus status =
        gpu_easygraph::runtime::runtime_manager().status();
    if (!status.running) {
        throw std::runtime_error(
            "EGMGPU native runtime must be started before single-device execution");
    }
    if (std::find(status.device_ids.begin(), status.device_ids.end(), device) ==
        status.device_ids.end()) {
        throw std::out_of_range(
            "requested device is outside the active EGMGPU native runtime");
    }
}

std::shared_ptr<std::vector<double> > parse_weights(
    const std::shared_ptr<CSRGraph>& graph,
    const py::object& weight_object,
    bool* weighted) {
    if (weighted == nullptr) throw std::invalid_argument("weighted output is null");
    *weighted = false;
    if (weight_object.is_none()) return std::shared_ptr<std::vector<double> >();
    const std::string key = weight_object.cast<std::string>();
    // A frozen generation can receive concurrent requests for distinct
    // weight attributes.  gen_CSR(weight) serializes map insertion; protect
    // this lookup as well so another controller cannot rehash W_map while the
    // shared_ptr is copied out.  CUDA execution starts only after this lock is
    // released.
    std::lock_guard<std::mutex> weight_guard(graph->weights_mutex);
    const std::unordered_map<
        std::string,
        std::shared_ptr<std::vector<double> > >::const_iterator found =
        graph->W_map.find(key);
    if (found == graph->W_map.end()) {
        // A bulk artifact without a weight vector represents the normal
        // EasyGraph default of unit weights for a missing edge attribute.
        if (graph->W_map.empty() && graph->implicit_unit_weights) {
            return std::shared_ptr<std::vector<double> >();
        }
        throw py::key_error("requested weight key is absent from the immutable CSR");
    }
    if (!found->second || found->second->size() != graph->E.size()) {
        throw std::runtime_error("immutable CSR weight vector has an invalid length");
    }
    *weighted = true;
    return found->second;
}

PreparedSingleCall prepare_call(
    py::object graph_object,
    const std::string& operation_name,
    const py::dict& parameters,
    int device) {
    validate_fixed_worker(device);
    Graph& graph = graph_object.cast<Graph&>();
    if (!graph.csr_graph) {
        throw std::invalid_argument(
            "single/request execution requires a prebuilt immutable native CSR");
    }
    PreparedSingleCall call;
    call.operation = parse_operation(operation_name);
    call.graph = graph.csr_graph;
    call.device = device;
    call.directed = graph_object.attr("is_directed")().cast<bool>();
    if (call.graph->V.size() !=
        static_cast<std::size_t>(call.graph->node_count) + 1U) {
        throw std::invalid_argument(
            "single/request execution requires a complete immutable native CSR");
    }
    const std::size_t vertex_count = static_cast<std::size_t>(call.graph->node_count);
    const py::object weight_object = parameter_object(parameters, "weight");
    if (!weight_object.is_none() && !call.graph->contiguous_zero_based) {
        // A mutable EasyGraph source is copied into a private native Graph at
        // submission.  Materialize the requested weight view from that copy
        // while the controller still owns the GIL; fixed workers subsequently
        // retain only CSR vectors and never touch Python objects.
        graph.gen_CSR(weight_object.cast<std::string>());
    }
    call.weights = parse_weights(
        call.graph, weight_object, &call.weighted);

    call.alpha = parameter_or<double>(parameters, "alpha", 0.85);
    call.max_iter = parameter_or<int>(parameters, "max_iter", 200);
    call.tolerance = parameter_or<double>(parameters, "tol", 1.0e-6);
    call.normalized = parameter_or<bool>(parameters, "normalized", true);
    call.endpoints = parameter_or<bool>(parameters, "endpoints", false);
    if (!(call.alpha >= 0.0 && call.alpha <= 1.0)) {
        throw std::invalid_argument("PageRank alpha must be in [0, 1]");
    }
    if (call.max_iter < 1 || call.tolerance < 0.0) {
        throw std::invalid_argument("PageRank iteration parameters are invalid");
    }

    switch (call.operation) {
        case SingleOperation::Bfs:
        case SingleOperation::Dijkstra: {
            const py::object source = parameter_object(parameters, "source");
            if (source.is_none()) throw std::invalid_argument("source is required");
            call.sources = parse_nodes(source, vertex_count, nullptr);
            call.target = parse_target(parameter_object(parameters, "target"), vertex_count);
            break;
        }
        case SingleOperation::BellmanFord:
        case SingleOperation::Sssp:
            call.sources = parse_sources(
                parameter_object(parameters, "sources"), vertex_count, false);
            call.target = parse_target(parameter_object(parameters, "target"), vertex_count);
            break;
        case SingleOperation::Betweenness:
        case SingleOperation::Closeness:
            call.sources = parse_sources(
                parameter_object(parameters, "sources"), vertex_count, true);
            break;
        case SingleOperation::Lcc:
        case SingleOperation::EffectiveSize:
        case SingleOperation::Efficiency:
        case SingleOperation::Constraint:
        case SingleOperation::Hierarchy:
            call.nodes = parse_nodes(
                parameter_object(parameters, "nodes"), vertex_count,
                &call.scalar_nodes_result);
            break;
        default:
            break;
    }

    if (call.operation == SingleOperation::Lcc && call.weighted) {
        throw std::invalid_argument(
            "the current native LCC ABI supports weight=None only");
    }
    if (call.operation == SingleOperation::Lcc && call.directed &&
        call.graph->undirected_projection == nullptr) {
        throw std::invalid_argument(
            "directed bulk LCC requires a validated logical-undirected projection");
    }
    if (call.operation == SingleOperation::Mst && call.directed) {
        throw std::invalid_argument("minimum_spanning_tree requires an undirected graph");
    }
    if (call.operation == SingleOperation::Mst &&
        parameter_or<bool>(parameters, "ignore_nan", false)) {
        throw std::invalid_argument(
            "native MST does not implement ignore_nan=True edge filtering");
    }
    if (call.operation == SingleOperation::Mst) {
        const std::string algorithm =
            parameter_or<std::string>(parameters, "algorithm", "kruskal");
        if (algorithm != "kruskal" && algorithm != "prim" && algorithm != "boruvka") {
            throw std::invalid_argument("unknown minimum-spanning-tree algorithm");
        }
    }
    if (is_structural(call.operation) && call.weighted &&
        !call.graph->simple_sorted_topology) {
        throw std::invalid_argument(
            "weighted structural-hole execution requires validated simple sorted CSR");
    }
    return call;
}

void ensure_reverse_csr(const std::shared_ptr<CSRGraph>& graph) {
    std::lock_guard<std::mutex> lock(graph->reverse_mutex);
    if (graph->reverse_ready) return;
    const int vertex_count = static_cast<int>(graph->node_count);
    graph->reverse_V.assign(static_cast<std::size_t>(vertex_count) + 1U, 0);
    graph->reverse_E.assign(graph->E.size(), 0);
    for (std::size_t edge = 0; edge < graph->E.size(); ++edge) {
        const int target = graph->E[edge];
        if (target < 0 || target >= vertex_count) {
            throw std::runtime_error("CSR endpoint is outside the vertex domain");
        }
        ++graph->reverse_V[static_cast<std::size_t>(target) + 1U];
    }
    for (int vertex = 0; vertex < vertex_count; ++vertex) {
        graph->reverse_V[static_cast<std::size_t>(vertex) + 1U] +=
            graph->reverse_V[static_cast<std::size_t>(vertex)];
    }
    std::vector<int> cursor = graph->reverse_V;
    for (int source = 0; source < vertex_count; ++source) {
        for (int edge = graph->V[static_cast<std::size_t>(source)];
             edge < graph->V[static_cast<std::size_t>(source) + 1U]; ++edge) {
            graph->reverse_E[static_cast<std::size_t>(
                cursor[static_cast<std::size_t>(graph->E[edge])]++)] = source;
        }
    }
    graph->reverse_ready = true;
}

const std::vector<double>& unit_weights_for(const std::shared_ptr<CSRGraph>& graph) {
    // Fixed workers retain this thread-local capacity across requests.  This
    // supplies the legacy weighted-only MST/Bellman-Ford ABI without adding a
    // per-call |E|-sized host allocation or mutating the immutable snapshot.
    static thread_local std::uint64_t cached_graph_id = 0;
    static thread_local std::vector<double> unit_weights;
    if (cached_graph_id != graph->cache_id || unit_weights.size() != graph->E.size()) {
        unit_weights.assign(graph->E.size(), 1.0);
        cached_graph_id = graph->cache_id;
    }
    return unit_weights;
}

const std::vector<double>& call_weights(
    const PreparedSingleCall& call,
    bool require_materialized_unit) {
    if (call.weighted) return *call.weights;
    if (require_materialized_unit) return unit_weights_for(call.graph);
    return call.graph->unweighted_W;
}

void throw_gpu_error(int status) {
    if (status != gpu_easygraph::EG_GPU_SUCC) {
        throw std::runtime_error(gpu_easygraph::err_code_detail(status));
    }
}

void run_structural(
    const PreparedSingleCall& call,
    SingleNativeResult* result) {
    const std::shared_ptr<CSRGraph>& graph = call.graph;
    const std::size_t vertex_count = static_cast<std::size_t>(graph->node_count);
    result->nodes = call.nodes;
    if (call.nodes.empty()) return;

    const std::shared_ptr<UndirectedProjection>& projection =
        graph->undirected_projection;
    const bool projection_path =
        call.directed && !call.weighted && graph->simple_sorted_topology &&
        projection != nullptr &&
        projection->lower_V.size() == graph->V.size() &&
        projection->upper_V.size() == graph->V.size() &&
        projection->forward_V.size() == graph->V.size() &&
        projection->degree.size() == vertex_count;
    int status = gpu_easygraph::EG_GPU_DEVICE_ERR;
    std::vector<double> values;
    if (projection_path) {
        if (call.operation == SingleOperation::EffectiveSize ||
            call.operation == SingleOperation::Efficiency) {
            status = gpu_easygraph::effective_size_ego_edge_statistics_subset(
                graph->V, graph->E,
                projection->lower_V, projection->lower_E,
                projection->upper_V, projection->upper_E,
                projection->forward_V, projection->forward_E,
                projection->degree, graph->cache_id, call.nodes, values,
                &result->kernel_seconds);
        } else if (call.operation == SingleOperation::Constraint) {
            status = gpu_easygraph::constraint_ego_edge_statistics_subset(
                graph->V, graph->E,
                projection->lower_V, projection->lower_E,
                projection->upper_V, projection->upper_E,
                projection->forward_V, projection->forward_E,
                projection->degree, graph->cache_id, call.nodes, values,
                &result->kernel_seconds);
        } else {
            status = gpu_easygraph::hierarchy_ego_edge_statistics_subset(
                graph->V, graph->E,
                projection->lower_V, projection->lower_E,
                projection->upper_V, projection->upper_E,
                projection->forward_V, projection->forward_E,
                projection->degree, graph->cache_id, call.nodes, values,
                &result->kernel_seconds);
        }
        throw_gpu_error(status);
    } else {
        if (call.directed) ensure_reverse_csr(graph);
        std::vector<int> row;
        const std::vector<int>* row_ptr = nullptr;
        const std::vector<int>* col_ptr = nullptr;
        static const std::vector<int> empty_int;
        if (call.weighted) {
            row.resize(graph->E.size());
            for (std::size_t source = 0; source < vertex_count; ++source) {
                for (int edge = graph->V[source]; edge < graph->V[source + 1U]; ++edge) {
                    row[static_cast<std::size_t>(edge)] = static_cast<int>(source);
                }
            }
            row_ptr = &row;
            col_ptr = &graph->E;
        } else {
            row_ptr = &empty_int;
            col_ptr = &empty_int;
        }
        std::vector<int> mask(vertex_count, 0);
        for (std::size_t index = 0; index < call.nodes.size(); ++index) {
            mask[static_cast<std::size_t>(call.nodes[index])] = 1;
        }
        std::vector<double> dense(vertex_count, 0.0);
        const std::vector<int>& incoming_v =
            call.directed ? graph->reverse_V : graph->V;
        const std::vector<int>& incoming_e =
            call.directed ? graph->reverse_E : graph->E;
        const std::vector<double>& weights = call_weights(call, false);
        if (call.operation == SingleOperation::EffectiveSize ||
            call.operation == SingleOperation::Efficiency) {
            status = gpu_easygraph::effective_size(
                graph->V, graph->E, incoming_v, incoming_e,
                *row_ptr, *col_ptr, static_cast<int>(vertex_count), weights,
                call.directed, mask, dense, &result->kernel_seconds);
        } else if (call.operation == SingleOperation::Constraint) {
            status = gpu_easygraph::constraint(
                graph->V, graph->E, incoming_v, incoming_e,
                *row_ptr, *col_ptr, static_cast<int>(vertex_count), weights,
                call.directed, mask, dense, &result->kernel_seconds);
        } else {
            status = gpu_easygraph::hierarchy(
                graph->V, graph->E, incoming_v, incoming_e,
                *row_ptr, *col_ptr, static_cast<int>(vertex_count), weights,
                call.directed, mask, dense, &result->kernel_seconds);
        }
        throw_gpu_error(status);
        values.reserve(call.nodes.size());
        for (std::size_t index = 0; index < call.nodes.size(); ++index) {
            values.push_back(dense[static_cast<std::size_t>(call.nodes[index])]);
        }
    }

    if (values.size() != call.nodes.size()) {
        throw std::runtime_error("structural-hole native result has invalid length");
    }
    if (call.operation == SingleOperation::Efficiency) {
        for (std::size_t index = 0; index < call.nodes.size(); ++index) {
            const int node = call.nodes[index];
            double degree = static_cast<double>(
                graph->V[static_cast<std::size_t>(node) + 1U] -
                graph->V[static_cast<std::size_t>(node)]);
            if (call.weighted) {
                degree = 0.0;
                for (int edge = graph->V[static_cast<std::size_t>(node)];
                     edge < graph->V[static_cast<std::size_t>(node) + 1U]; ++edge) {
                    degree += (*call.weights)[static_cast<std::size_t>(edge)];
                }
            }
            if (call.directed) {
                if (call.weighted) {
                    for (std::size_t source = 0; source < vertex_count; ++source) {
                        for (int edge = graph->V[source]; edge < graph->V[source + 1U]; ++edge) {
                            if (graph->E[static_cast<std::size_t>(edge)] == node) {
                                degree += (*call.weights)[static_cast<std::size_t>(edge)];
                            }
                        }
                    }
                } else {
                    ensure_reverse_csr(graph);
                    degree += static_cast<double>(
                        graph->reverse_V[static_cast<std::size_t>(node) + 1U] -
                        graph->reverse_V[static_cast<std::size_t>(node)]);
                }
            }
            values[index] = degree == 0.0
                ? std::numeric_limits<double>::quiet_NaN()
                : values[index] / degree;
        }
    }
    result->values.swap(values);
}

void execute_single_call(
    const PreparedSingleCall& call,
    SingleNativeResult* result) {
    gpu_easygraph::DeviceCsrCacheScope cache_scope(call.graph->cache_id);
    const std::shared_ptr<CSRGraph>& graph = call.graph;
    const std::size_t vertex_count = static_cast<std::size_t>(graph->node_count);
    int status = gpu_easygraph::EG_GPU_SUCC;
    if (vertex_count == 0U) {
        result->nodes = call.nodes;
        return;
    }
    if (call.sources.empty() &&
        (call.operation == SingleOperation::Bfs ||
         call.operation == SingleOperation::Dijkstra ||
         call.operation == SingleOperation::BellmanFord ||
         call.operation == SingleOperation::Sssp ||
         call.operation == SingleOperation::Betweenness ||
         call.operation == SingleOperation::Closeness)) {
        if (call.operation == SingleOperation::Betweenness) {
            result->values.assign(vertex_count, 0.0);
        }
        return;
    }
    switch (call.operation) {
        case SingleOperation::PageRank:
            status = gpu_easygraph::pagerank(
                graph->V, graph->E, call_weights(call, false), call.alpha,
                call.max_iter, call.tolerance, result->values,
                &result->kernel_seconds);
            break;
        case SingleOperation::Mst:
            if (graph->undirected_projection != nullptr) {
                const std::shared_ptr<UndirectedProjection>& projection =
                    graph->undirected_projection;
                const std::vector<double>& weights = call.weighted
                    ? projection->lower_W
                    : graph->unweighted_W;
                if (call.weighted &&
                    weights.size() != projection->lower_E.size()) {
                    throw std::runtime_error(
                        "MST projection lacks aligned requested weights");
                }
                status = gpu_easygraph::mst_single_incidence(
                    projection->lower_V, projection->lower_E, weights,
                    result->mst_sources, result->mst_targets,
                    result->mst_weights, &result->kernel_seconds);
            } else {
                status = gpu_easygraph::mst(
                    graph->V, graph->E, call_weights(call, !call.weighted),
                    result->mst_sources, result->mst_targets,
                    result->mst_weights, &result->kernel_seconds);
            }
            break;
        case SingleOperation::Lcc: {
            std::vector<double> dense;
            if (call.directed) {
                const std::shared_ptr<UndirectedProjection>& projection =
                    graph->undirected_projection;
                status = gpu_easygraph::clustering_forward(
                    projection->forward_V, projection->forward_E,
                    projection->degree, dense, &result->kernel_seconds);
            } else {
                status = gpu_easygraph::clustering(
                    graph->V, graph->E, false, dense,
                    &result->kernel_seconds);
            }
            throw_gpu_error(status);
            result->nodes = call.nodes;
            result->values.reserve(call.nodes.size());
            for (std::size_t index = 0; index < call.nodes.size(); ++index) {
                result->values.push_back(
                    dense[static_cast<std::size_t>(call.nodes[index])]);
            }
            return;
        }
        case SingleOperation::Wcc:
        case SingleOperation::Scc:
            status = gpu_easygraph::connected_components(
                graph->V, graph->E,
                call.operation == SingleOperation::Scc,
                result->integer_values, &result->kernel_seconds);
            break;
        case SingleOperation::Bfs:
            status = gpu_easygraph::sssp_unweighted_bfs(
                graph->V, graph->E, call.sources, call.target,
                result->values, &result->kernel_seconds);
            break;
        case SingleOperation::Dijkstra:
        case SingleOperation::Sssp:
            if (call.weighted) {
                status = gpu_easygraph::sssp_dijkstra(
                    graph->V, graph->E, *call.weights, call.sources,
                    call.target, result->values, &result->kernel_seconds);
            } else {
                status = gpu_easygraph::sssp_unweighted_bfs(
                    graph->V, graph->E, call.sources, call.target,
                    result->values, &result->kernel_seconds);
            }
            break;
        case SingleOperation::BellmanFord:
            status = gpu_easygraph::sssp_bellman_ford(
                graph->V, graph->E, call_weights(call, !call.weighted),
                call.sources, call.target, result->values,
                &result->kernel_seconds);
            break;
        case SingleOperation::KCore:
            result->integer_values.assign(vertex_count, 0);
            if (graph->undirected_projection != nullptr) {
                const std::shared_ptr<UndirectedProjection>& projection =
                    graph->undirected_projection;
                status = gpu_easygraph::k_core_split_into(
                    projection->lower_V, projection->lower_E,
                    projection->upper_V, projection->upper_E,
                    projection->degree, result->integer_values.data(),
                    &result->kernel_seconds);
            } else {
                status = gpu_easygraph::k_core_into(
                    graph->V, graph->E, result->integer_values.data(),
                    &result->kernel_seconds);
            }
            break;
        case SingleOperation::Betweenness:
            status = gpu_easygraph::betweenness_centrality(
                graph->V, graph->E, call_weights(call, false), call.sources,
                call.directed, call.normalized, call.endpoints,
                !call.weighted, result->values, &result->kernel_seconds);
            break;
        case SingleOperation::Closeness:
            status = gpu_easygraph::closeness_centrality(
                graph->V, graph->E, call_weights(call, false), call.sources,
                !call.weighted, result->values, &result->kernel_seconds);
            break;
        case SingleOperation::EffectiveSize:
        case SingleOperation::Efficiency:
        case SingleOperation::Constraint:
        case SingleOperation::Hierarchy:
            run_structural(call, result);
            return;
    }
    throw_gpu_error(status);
}

template <typename T>
py::array_t<T> copy_array(const std::vector<T>& values) {
    py::array_t<T> array(values.size());
    if (!values.empty()) {
        std::copy(values.begin(), values.end(), array.mutable_data());
    }
    return array;
}

py::dict materialize_raw_result(
    const std::string& operation,
    const PreparedSingleCall& call,
    const SingleNativeResult& result) {
    py::dict out;
    out["operation"] = operation;
    out["device"] = call.device;
    out["kernel_seconds"] = result.kernel_seconds;
    out["scalar_nodes_result"] = call.scalar_nodes_result;
    if (!result.values.empty() ||
        call.operation == SingleOperation::PageRank ||
        call.operation == SingleOperation::Bfs ||
        call.operation == SingleOperation::Dijkstra ||
        call.operation == SingleOperation::BellmanFord ||
        call.operation == SingleOperation::Sssp ||
        call.operation == SingleOperation::Betweenness ||
        call.operation == SingleOperation::Closeness ||
        call.operation == SingleOperation::Lcc ||
        is_structural(call.operation)) {
        out["values_dense"] = copy_array(result.values);
    }
    if (!result.nodes.empty() || call.operation == SingleOperation::Lcc ||
        is_structural(call.operation)) {
        out["nodes_dense"] = copy_array(result.nodes);
    }
    if (call.operation == SingleOperation::Wcc ||
        call.operation == SingleOperation::Scc ||
        call.operation == SingleOperation::KCore) {
        out["integer_values_dense"] = copy_array(result.integer_values);
    }
    if (call.operation == SingleOperation::Mst) {
        out["mst_sources"] = copy_array(result.mst_sources);
        out["mst_targets"] = copy_array(result.mst_targets);
        out["mst_weights"] = copy_array(result.mst_weights);
    }
    return out;
}

py::dict single_execute(
    py::object graph_object,
    const std::string& operation,
    const py::dict& parameters,
    int device) {
    // Everything above the release point runs under the controller's GIL.
    // PreparedSingleCall retains only shared native buffers and scalar/vector
    // values.  The worker lambda below neither captures nor accesses Python.
    const PreparedSingleCall prepared =
        prepare_call(graph_object, operation, parameters, device);
    const std::shared_ptr<SingleNativeResult> result(new SingleNativeResult());
    std::vector<gpu_easygraph::runtime::NativeRuntime::Task> tasks;
    tasks.push_back([prepared, result](int worker_device) {
        if (worker_device != prepared.device) {
            throw std::runtime_error(
                "native runtime dispatched single request to the wrong fixed worker");
        }
        execute_single_call(prepared, result.get());
    });
    {
        py::gil_scoped_release release;
        gpu_easygraph::runtime::runtime_manager().execute_gang(
            std::vector<int>(1, device), std::move(tasks));
    }
    // No result object is exposed until the native task has completed without
    // error.  This is the sole coordinator-side Python materialization point.
    return materialize_raw_result(operation, prepared, *result);
}

#endif

}  // namespace

void bind_egmgpu_single_dispatch(py::module_& module) {
#ifdef EASYGRAPH_ENABLE_GPU
    module.def(
        "cpp_egmgpu_single_execute",
        &single_execute,
        py::arg("G"),
        py::arg("operation"),
        py::arg("parameters"),
        py::arg("device"));
#else
    (void)module;
#endif
}
