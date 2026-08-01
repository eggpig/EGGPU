import os

_TRUE_VALUES = {"1", "TRUE", "ON", "YES"}


def _env_true(name):
    return os.environ.get(name, "").strip().upper() in _TRUE_VALUES


def gpu_runtime_enabled():
    return _env_true("EASYGRAPH_ENABLE_GPU")


def gpu_strict_errors():
    return _env_true("EASYGRAPH_GPU_STRICT_ERRORS")


def graph_nodes_list(G):
    nodes = G.nodes
    if hasattr(nodes, "keys"):
        return list(nodes.keys())
    return list(nodes)


def build_node_index(G):
    nodes = graph_nodes_list(G)
    return nodes, {node: idx for idx, node in enumerate(nodes)}


def indexed_edges(G, node_to_idx, undirected_projection=False, weight_key=None):
    rows = []
    if undirected_projection:
        seen = set()
        for u, v, data in G.edges:
            iu = node_to_idx[u]
            iv = node_to_idx[v]
            if iu == iv:
                continue
            if iu > iv:
                iu, iv = iv, iu
            key = (iu, iv)
            if key in seen:
                continue
            seen.add(key)
            if weight_key is None:
                rows.append((iu, iv))
            else:
                rows.append((iu, iv, float(data.get(weight_key, 1))))
        return rows

    for u, v, data in G.edges:
        iu = node_to_idx[u]
        iv = node_to_idx[v]
        if weight_key is None:
            rows.append((iu, iv))
        else:
            rows.append((iu, iv, float(data.get(weight_key, 1))))
    return rows
