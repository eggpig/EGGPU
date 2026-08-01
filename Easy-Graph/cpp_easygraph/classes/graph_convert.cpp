#include "directed_graph.h"
#include "graph.h"

#include "../common/utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace {

edge_attr_dict_factory attrs_from_python(py::handle obj) {
    edge_attr_dict_factory attrs;
    if (!py::isinstance<py::dict>(obj)) {
        return attrs;
    }
    py::dict dict = py::reinterpret_borrow<py::dict>(obj);
    for (auto item : dict) {
        try {
            py::object key = py::reinterpret_borrow<py::object>(item.first);
            attrs[weight_to_string(key)] = item.second.cast<weight_t>();
        } catch (const std::exception&) {
            // EasyGraph permits arbitrary metadata. GPU kernels only consume
            // numeric edge/node attributes, so nonnumeric metadata is ignored.
        }
    }
    return attrs;
}

node_t ensure_graph_node(Graph& out, py::handle node_obj, py::handle attr_obj) {
    py::object node = py::reinterpret_borrow<py::object>(node_obj);
    if (out.node_to_id.contains(node)) {
        return out.node_to_id[node].cast<node_t>();
    }
    node_t id = ++out.id;
    out.id_to_node[py::cast(id)] = node;
    out.node_to_id[node] = id;
    out.node[id] = attrs_from_python(attr_obj);
    out.adj[id] = adj_attr_dict_factory();
    return id;
}

node_t ensure_digraph_node(DiGraph& out, py::handle node_obj, py::handle attr_obj) {
    py::object node = py::reinterpret_borrow<py::object>(node_obj);
    if (out.node_to_id.contains(node)) {
        return out.node_to_id[node].cast<node_t>();
    }
    node_t id = ++out.id;
    out.id_to_node[py::cast(id)] = node;
    out.node_to_id[node] = id;
    out.node[id] = attrs_from_python(attr_obj);
    out.adj[id] = adj_attr_dict_factory();
    out.pred[id] = adj_attr_dict_factory();
    return id;
}

void copy_common_graph_attrs(Graph& out, py::object G) {
    try {
        out.graph.attr("update")(G.attr("graph"));
    } catch (const std::exception&) {
    }
    out.dirty_nodes = true;
    out.dirty_adj = true;
    out.linkgraph_dirty = true;
    out.csr_graph = nullptr;
    out.coo_graph = nullptr;
}

Graph build_graph(py::object G) {
    Graph out;
    copy_common_graph_attrs(out, G);

    py::dict nodes = py::reinterpret_borrow<py::dict>(G.attr("_node"));
    for (auto item : nodes) {
        ensure_graph_node(out, item.first, item.second);
    }

    py::dict adj = py::reinterpret_borrow<py::dict>(G.attr("_adj"));
    py::dict empty_attr;
    std::unordered_set<unsigned long long> seen_edges;
    seen_edges.reserve((size_t)py::len(adj) * 2);
    for (auto src_item : adj) {
        py::object u_node = py::reinterpret_borrow<py::object>(src_item.first);
        node_t u = ensure_graph_node(out, u_node, empty_attr);
        py::dict nbrs = py::reinterpret_borrow<py::dict>(src_item.second);
        for (auto dst_item : nbrs) {
            py::object v_node = py::reinterpret_borrow<py::object>(dst_item.first);
            node_t v = ensure_graph_node(out, v_node, empty_attr);
            if (u == v) {
                continue;
            }
            node_t a = std::min(u, v);
            node_t b = std::max(u, v);
            unsigned long long key = (static_cast<unsigned long long>(a) << 32)
                ^ static_cast<unsigned long long>(b);
            if (!seen_edges.insert(key).second) {
                continue;
            }
            edge_attr_dict_factory attrs = attrs_from_python(dst_item.second);
            out.adj[u][v] = attrs;
            out.adj[v][u] = attrs;
        }
    }
    return out;
}

DiGraph build_digraph(py::object G) {
    DiGraph out;
    copy_common_graph_attrs(out, G);

    py::dict nodes = py::reinterpret_borrow<py::dict>(G.attr("_node"));
    for (auto item : nodes) {
        ensure_digraph_node(out, item.first, item.second);
    }

    py::dict adj = py::reinterpret_borrow<py::dict>(G.attr("_adj"));
    py::dict empty_attr;
    for (auto src_item : adj) {
        py::object u_node = py::reinterpret_borrow<py::object>(src_item.first);
        node_t u = ensure_digraph_node(out, u_node, empty_attr);
        py::dict nbrs = py::reinterpret_borrow<py::dict>(src_item.second);
        for (auto dst_item : nbrs) {
            py::object v_node = py::reinterpret_borrow<py::object>(dst_item.first);
            node_t v = ensure_digraph_node(out, v_node, empty_attr);
            edge_attr_dict_factory attrs = attrs_from_python(dst_item.second);
            out.adj[u][v] = attrs;
            out.pred[v][u] = attrs;
        }
    }
    return out;
}

template <typename T>
void read_exact_vector(
    const std::string& path,
    std::vector<T>& out,
    std::size_t expected_items) {
    static_assert(std::is_trivially_copyable<T>::value, "raw CSR types must be POD");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw py::value_error("cannot open CSR file: " + path);
    }
    const std::streamoff size = input.tellg();
    const std::uint64_t expected_bytes =
        static_cast<std::uint64_t>(expected_items) * sizeof(T);
    if (size < 0 || static_cast<std::uint64_t>(size) != expected_bytes) {
        std::ostringstream oss;
        oss << "CSR file size mismatch for " << path << ": expected "
            << expected_bytes << " bytes, found " << size;
        throw py::value_error(oss.str());
    }
    out.resize(expected_items);
    input.seekg(0, std::ios::beg);
    constexpr std::size_t chunk_bytes = 1ULL << 30;
    std::size_t copied = 0;
    char* dst = reinterpret_cast<char*>(out.data());
    while (copied < expected_bytes) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(chunk_bytes, expected_bytes - copied));
        input.read(dst + copied, static_cast<std::streamsize>(count));
        if (input.gcount() != static_cast<std::streamsize>(count)) {
            throw py::value_error("short read while loading CSR file: " + path);
        }
        copied += count;
    }
}

std::string required_projection_path(
    const py::dict& spec,
    const char* key) {
    if (!spec.contains(key)) {
        throw py::value_error(
            std::string("undirected projection is missing ") + key);
    }
    const std::string value = py::cast<std::string>(spec[key]);
    if (value.empty()) {
        throw py::value_error(
            std::string("undirected projection path is empty: ") + key);
    }
    return value;
}

std::uint64_t projection_edge_key(int first, int second) {
    const int lower = std::min(first, second);
    const int upper = std::max(first, second);
    return (static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(lower))
            << 32) |
        static_cast<std::uint32_t>(upper);
}

std::uint64_t projection_hash(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

struct ProjectionFingerprint {
    std::uint64_t count = 0;
    std::uint64_t sum = 0;
    std::uint64_t xor_value = 0;

    void add(int first, int second) {
        const std::uint64_t hashed =
            projection_hash(projection_edge_key(first, second));
        ++count;
        sum += hashed;
        xor_value ^= hashed;
    }

    bool operator==(const ProjectionFingerprint& other) const {
        return count == other.count && sum == other.sum &&
            xor_value == other.xor_value;
    }
};

ProjectionFingerprint validate_projection_half(
    const char* name,
    const std::vector<int>& offsets,
    const std::vector<int>& indices,
    std::int64_t num_nodes,
    bool lower_half) {
    ProjectionFingerprint fingerprint;
    if (offsets.size() != static_cast<std::size_t>(num_nodes) + 1 ||
        offsets.empty() || offsets.front() != 0 ||
        offsets.back() != static_cast<int>(indices.size())) {
        throw py::value_error(
            std::string("invalid undirected projection ") + name +
            " offsets");
    }
    int previous_offset = 0;
    for (std::int64_t source = 0; source < num_nodes; ++source) {
        const int begin = offsets[static_cast<std::size_t>(source)];
        const int end = offsets[static_cast<std::size_t>(source) + 1];
        if (begin < previous_offset || end < begin ||
            end > static_cast<int>(indices.size())) {
            throw py::value_error(
                std::string("nonmonotonic undirected projection ") + name +
                " offsets");
        }
        int previous_target = -1;
        for (int position = begin; position < end; ++position) {
            const int target = indices[static_cast<std::size_t>(position)];
            const bool endpoint_order_valid =
                lower_half ? target > source : target < source;
            if (target < 0 || target >= num_nodes ||
                !endpoint_order_valid || target <= previous_target) {
                throw py::value_error(
                    std::string("invalid or duplicate edge in undirected ") +
                    "projection " + name);
            }
            previous_target = target;
            fingerprint.add(static_cast<int>(source), target);
        }
        previous_offset = end;
    }
    return fingerprint;
}

bool projection_rank_less(
    int source,
    int target,
    const std::vector<int>& degree) {
    const int source_degree = degree[static_cast<std::size_t>(source)];
    const int target_degree = degree[static_cast<std::size_t>(target)];
    return source_degree < target_degree ||
        (source_degree == target_degree && source < target);
}

ProjectionFingerprint validate_projection_forward(
    const std::vector<int>& offsets,
    const std::vector<int>& indices,
    const std::vector<int>& degree,
    std::int64_t num_nodes) {
    ProjectionFingerprint fingerprint;
    if (offsets.size() != static_cast<std::size_t>(num_nodes) + 1 ||
        offsets.empty() || offsets.front() != 0 ||
        offsets.back() != static_cast<int>(indices.size())) {
        throw py::value_error(
            "invalid undirected projection forward offsets");
    }
    for (std::int64_t source = 0; source < num_nodes; ++source) {
        const int begin = offsets[static_cast<std::size_t>(source)];
        const int end = offsets[static_cast<std::size_t>(source) + 1];
        if (begin < 0 || end < begin ||
            end > static_cast<int>(indices.size())) {
            throw py::value_error(
                "nonmonotonic undirected projection forward offsets");
        }
        int previous_target = -1;
        for (int position = begin; position < end; ++position) {
            const int target = indices[static_cast<std::size_t>(position)];
            if (target < 0 || target >= num_nodes ||
                target <= previous_target ||
                !projection_rank_less(
                    static_cast<int>(source), target, degree)) {
                throw py::value_error(
                    "invalid edge in undirected projection forward CSR");
            }
            previous_target = target;
            fingerprint.add(static_cast<int>(source), target);
        }
    }
    return fingerprint;
}

void load_undirected_projection(
    const py::object& projection_spec,
    const std::shared_ptr<CSRGraph>& csr,
    std::int64_t num_nodes,
    bool validate) {
    if (projection_spec.is_none()) {
        return;
    }
    if (!py::isinstance<py::dict>(projection_spec)) {
        throw py::type_error(
            "undirected_projection must be a dictionary or None");
    }
    const py::dict spec =
        py::reinterpret_borrow<py::dict>(projection_spec);
    if (!spec.contains("unique_edge_count")) {
        throw py::value_error(
            "undirected projection is missing unique_edge_count");
    }
    const std::int64_t unique_edge_count =
        py::cast<std::int64_t>(spec["unique_edge_count"]);
    if (unique_edge_count < 0 ||
        unique_edge_count >= std::numeric_limits<int>::max()) {
        throw py::value_error(
            "undirected projection unique_edge_count exceeds int32");
    }

    auto projection = std::make_shared<UndirectedProjection>();
    projection->unique_edge_count = unique_edge_count;
    read_exact_vector<int>(
        required_projection_path(spec, "lower_V_path"),
        projection->lower_V,
        static_cast<std::size_t>(num_nodes) + 1);
    read_exact_vector<int>(
        required_projection_path(spec, "lower_E_path"),
        projection->lower_E,
        static_cast<std::size_t>(unique_edge_count));
    read_exact_vector<int>(
        required_projection_path(spec, "upper_V_path"),
        projection->upper_V,
        static_cast<std::size_t>(num_nodes) + 1);
    read_exact_vector<int>(
        required_projection_path(spec, "upper_E_path"),
        projection->upper_E,
        static_cast<std::size_t>(unique_edge_count));
    read_exact_vector<int>(
        required_projection_path(spec, "degree_path"),
        projection->degree,
        static_cast<std::size_t>(num_nodes));
    read_exact_vector<int>(
        required_projection_path(spec, "forward_V_path"),
        projection->forward_V,
        static_cast<std::size_t>(num_nodes) + 1);
    read_exact_vector<int>(
        required_projection_path(spec, "forward_E_path"),
        projection->forward_E,
        static_cast<std::size_t>(unique_edge_count));
    if (spec.contains("weights_path")) {
        read_exact_vector<double>(
            required_projection_path(spec, "weights_path"),
            projection->lower_W,
            static_cast<std::size_t>(unique_edge_count));
        projection->weight_key = spec.contains("weight_key")
            ? py::cast<std::string>(spec["weight_key"])
            : std::string("weight");
    }

    const bool endpoint_offsets_valid =
        !projection->lower_V.empty() &&
        !projection->upper_V.empty() &&
        !projection->forward_V.empty() &&
        projection->lower_V.front() == 0 &&
        projection->upper_V.front() == 0 &&
        projection->forward_V.front() == 0 &&
        projection->lower_V.back() == unique_edge_count &&
        projection->upper_V.back() == unique_edge_count &&
        projection->forward_V.back() == unique_edge_count;
    if (!endpoint_offsets_valid) {
        throw py::value_error(
            "undirected projection offsets do not end at unique_edge_count");
    }

    if (validate) {
        const ProjectionFingerprint lower_fingerprint =
            validate_projection_half(
                "lower half",
                projection->lower_V,
                projection->lower_E,
                num_nodes,
                true);
        const ProjectionFingerprint upper_fingerprint =
            validate_projection_half(
                "upper half",
                projection->upper_V,
                projection->upper_E,
                num_nodes,
                false);
        std::int64_t degree_sum = 0;
        for (std::int64_t node = 0; node < num_nodes; ++node) {
            const int degree =
                projection->degree[static_cast<std::size_t>(node)];
            const std::int64_t expected_degree =
                static_cast<std::int64_t>(
                    projection->lower_V[
                        static_cast<std::size_t>(node) + 1]) -
                projection->lower_V[static_cast<std::size_t>(node)] +
                static_cast<std::int64_t>(
                    projection->upper_V[
                        static_cast<std::size_t>(node) + 1]) -
                projection->upper_V[static_cast<std::size_t>(node)];
            if (degree < 0 || degree != expected_degree) {
                throw py::value_error(
                    "undirected projection degree differs from half views");
            }
            degree_sum += degree;
        }
        if (degree_sum != 2 * unique_edge_count) {
            throw py::value_error(
                "undirected projection degree sum does not equal 2|E|");
        }
        const ProjectionFingerprint forward_fingerprint =
            validate_projection_forward(
                projection->forward_V,
                projection->forward_E,
                projection->degree,
                num_nodes);
        if (!(lower_fingerprint == upper_fingerprint) ||
            !(lower_fingerprint == forward_fingerprint) ||
            lower_fingerprint.count !=
                static_cast<std::uint64_t>(unique_edge_count)) {
            throw py::value_error(
                "undirected projection views contain different edge sets");
        }
    }
    csr->undirected_projection = projection;
}

std::shared_ptr<CSRGraph> load_csr_files(
    const std::string& offsets_path,
    const std::string& indices_path,
    const std::string& weights_path,
    const std::string& weight_key,
    std::int64_t num_nodes,
    std::int64_t num_entries,
    bool validate) {
    static_assert(sizeof(int) == 4, "EGGPU bulk CSR requires 32-bit int");
    if (num_nodes < 0 || num_nodes >= std::numeric_limits<int>::max()) {
        throw py::value_error("num_nodes exceeds the signed 32-bit EGGPU CSR limit");
    }
    if (num_entries < 0 || num_entries >= std::numeric_limits<int>::max()) {
        throw py::value_error("num_entries exceeds the signed 32-bit EGGPU CSR limit");
    }

    auto csr = std::make_shared<CSRGraph>();
    read_exact_vector<int>(
        offsets_path,
        csr->V,
        static_cast<std::size_t>(num_nodes) + 1);
    read_exact_vector<int>(
        indices_path,
        csr->E,
        static_cast<std::size_t>(num_entries));
    if (!weights_path.empty()) {
        if (weight_key.empty()) {
            throw py::value_error("weight_key must be non-empty when weights_path is provided");
        }
        auto weights = std::make_shared<std::vector<double>>();
        read_exact_vector<double>(
            weights_path,
            *weights,
            static_cast<std::size_t>(num_entries));
        for (std::size_t i = 0; i < weights->size(); ++i) {
            if (!std::isfinite((*weights)[i])) {
                throw py::value_error("bulk CSR weights must be finite");
            }
        }
        csr->W_map[weight_key] = weights;
    }

    if (csr->V.empty() || csr->V.front() != 0 || csr->V.back() != num_entries) {
        throw py::value_error("invalid CSR offsets: first offset must be zero and final offset must equal num_entries");
    }
    if (validate) {
        int previous = 0;
        for (std::size_t i = 0; i < csr->V.size(); ++i) {
            const int offset = csr->V[i];
            if (offset < previous || offset < 0 || offset > num_entries) {
                throw py::value_error("invalid CSR offsets: offsets must be monotonic and in range");
            }
            previous = offset;
        }
        bool simple_sorted = true;
        for (std::int64_t source = 0; source < num_nodes; ++source) {
            int previous_target = -1;
            for (int edge = csr->V[static_cast<std::size_t>(source)];
                 edge < csr->V[static_cast<std::size_t>(source) + 1];
                 ++edge) {
                const int dst = csr->E[static_cast<std::size_t>(edge)];
                if (dst < 0 || dst >= num_nodes) {
                    throw py::value_error(
                        "invalid CSR index: endpoint is outside "
                        "[0, num_nodes)");
                }
                if (dst == source || dst <= previous_target) {
                    simple_sorted = false;
                }
                previous_target = dst;
            }
        }
        csr->simple_sorted_topology = simple_sorted;
    }

    csr->contiguous_zero_based = true;
    csr->implicit_unit_weights = weights_path.empty();
    csr->node_count = num_nodes;
    return csr;
}

template <typename GraphType>
GraphType graph_with_bulk_csr(const std::shared_ptr<CSRGraph>& csr) {
    GraphType out;
    out.csr_graph = csr;
    out.id = static_cast<node_t>(csr->node_count);
    out.dirty_nodes = false;
    out.dirty_adj = false;
    out.linkgraph_dirty = true;
    return out;
}

}  // namespace

py::object cpp_graph_from_easygraph(py::object G, bool directed) {
    if (directed) {
        return py::cast(build_digraph(G));
    }
    return py::cast(build_graph(G));
}

py::object cpp_graph_from_csr_files(
    const std::string& offsets_path,
    const std::string& indices_path,
    std::int64_t num_nodes,
    std::int64_t num_entries,
    bool directed,
    bool validate,
    const std::string& weights_path,
    const std::string& weight_key,
    const py::object& undirected_projection) {
    auto csr = load_csr_files(
        offsets_path,
        indices_path,
        weights_path,
        weight_key,
        num_nodes,
        num_entries,
        validate);
    load_undirected_projection(
        undirected_projection, csr, num_nodes, validate);
    if (directed) {
        return py::cast(graph_with_bulk_csr<DiGraph>(csr));
    }
    return py::cast(graph_with_bulk_csr<Graph>(csr));
}

py::dict cpp_graph_undirected_projection_info(const Graph& graph) {
    py::dict info;
    const auto& projection =
        graph.csr_graph ? graph.csr_graph->undirected_projection : nullptr;
    info["loaded"] = py::bool_(projection != nullptr);
    if (!projection) {
        return info;
    }
    info["unique_edge_count"] =
        py::int_(projection->unique_edge_count);
    info["lower_V_size"] = py::int_(projection->lower_V.size());
    info["lower_E_size"] = py::int_(projection->lower_E.size());
    info["upper_V_size"] = py::int_(projection->upper_V.size());
    info["upper_E_size"] = py::int_(projection->upper_E.size());
    info["degree_size"] = py::int_(projection->degree.size());
    info["forward_V_size"] = py::int_(projection->forward_V.size());
    info["forward_E_size"] = py::int_(projection->forward_E.size());
    info["lower_W_size"] = py::int_(projection->lower_W.size());
    info["weight_key"] = py::str(projection->weight_key);
    return info;
}
