import os

_MINE_BACKENDS = {"mine", "mine-bin", "eggpu", "native-mine", "auto", "default"}
_RAPIDS_BACKENDS = {"auto", "rapids"}
_SUPPORTED_BACKENDS = _MINE_BACKENDS | _RAPIDS_BACKENDS


def gpu_runtime_enabled():
    value = os.environ.get("EASYGRAPH_ENABLE_GPU", "")
    return value.strip().upper() in {"1", "TRUE", "ON", "YES"}


def gpu_backend_name():
    name = os.environ.get("EASYGRAPH_GPU_BACKEND", "mine").strip().lower()
    if gpu_runtime_enabled() and gpu_strict_errors() and name not in _SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(_SUPPORTED_BACKENDS))
        raise RuntimeError(
            f"Unsupported EASYGRAPH_GPU_BACKEND={name!r}. "
            f"Supported GPU backends are: {supported}."
        )
    return name


def gpu_strict_errors():
    value = os.environ.get("EASYGRAPH_GPU_STRICT_ERRORS", "")
    return value.strip().upper() in {"1", "TRUE", "ON", "YES"}


def rapids_backend_enabled():
    # Keep RAPIDS as an explicit fallback backend (`auto`/`rapids`) instead of
    # implicitly enabling it for our native backend aliases.
    return gpu_backend_name() in _RAPIDS_BACKENDS


def graph_nodes_list(G):
    nodes = G.nodes
    if hasattr(nodes, "keys"):
        return list(nodes.keys())
    return list(nodes)


def build_node_index(G):
    nodes = graph_nodes_list(G)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    return nodes, node_to_idx


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


def import_rapids():
    import cudf
    import cugraph

    return cudf, cugraph


def to_cudf_edgelist(cudf, rows, weighted=False):
    if weighted:
        return cudf.DataFrame(rows, columns=["src", "dst", "weight"])
    return cudf.DataFrame(rows, columns=["src", "dst"])


def make_cugraph_graph(cugraph, directed=False):
    try:
        if directed and hasattr(cugraph, "DiGraph"):
            return cugraph.DiGraph()
        return cugraph.Graph(directed=directed)
    except Exception:
        return cugraph.Graph(directed=directed)


def load_cugraph_edgelist(G, cudf, edge_df, weighted=False, num_nodes=0, renumber=False):
    kwargs = {"source": "src", "destination": "dst", "renumber": renumber}
    if weighted:
        kwargs["edge_attr"] = "weight"
    if num_nodes > 0:
        kwargs["vertices"] = cudf.DataFrame(
            {"vertex": cudf.Series(range(num_nodes), dtype="int32")}
        )
    try:
        G.from_cudf_edgelist(edge_df, **kwargs)
    except TypeError:
        kwargs.pop("vertices", None)
        G.from_cudf_edgelist(edge_df, **kwargs)


def component_sets_from_labels(result_df, nodes):
    pdf = result_df.to_pandas()
    label_col = (
        "labels"
        if "labels" in pdf.columns
        else ("label" if "label" in pdf.columns else "component")
    )
    groups = {}
    for vertex, label in zip(pdf["vertex"], pdf[label_col]):
        idx = int(vertex)
        if idx < 0 or idx >= len(nodes):
            continue
        groups.setdefault(int(label), set()).add(nodes[idx])
    return list(groups.values())
