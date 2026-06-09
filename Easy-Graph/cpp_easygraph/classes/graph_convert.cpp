#include "directed_graph.h"
#include "graph.h"

#include "../common/utils.h"

#include <algorithm>
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

}  // namespace

py::object cpp_graph_from_easygraph(py::object G, bool directed) {
    if (directed) {
        return py::cast(build_digraph(G));
    }
    return py::cast(build_graph(G));
}
