from easygraph.utils.decorators import *
from easygraph.utils.gpu_runtime import *


__all__ = [
    "is_connected",
    "number_connected_components",
    "connected_components",
    "connected_components_directed",
    "connected_component_of_node",
]


@not_implemented_for("multigraph")
def is_connected(G):
    """Returns whether the graph is connected or not.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    Returns
    -------
    is_biconnected : boolean
        `True` if the graph is connected.

    Examples
    --------

    >>> is_connected(G)

    """
    assert len(G) != 0, "No node in the graph."
    arbitrary_node = next(iter(G))  # Pick an arbitrary node to run BFS
    return len(G) == sum(1 for node in _plain_bfs(G, arbitrary_node))


@not_implemented_for("multigraph")
def number_connected_components(G):
    """Returns the number of connected components.

    Parameters
    ----------
    G : easygraph.Graph

    Returns
    -------
    number_connected_components : int
        The number of connected components.

    Examples
    --------
    >>> number_connected_components(G)

    """
    return sum(1 for component in _generator_connected_components(G))


@not_implemented_for("multigraph")
@hybrid("cpp_connected_components_undirected")
def connected_components(G):
    """Returns a list of connected components, each of which denotes the edges set of a connected component.

    Parameters
    ----------
    G : easygraph.Graph
    Returns
    -------
    connected_components : list of list
        Each element list is the edges set of a connected component.

    Examples
    --------
    >>> connected_components(G)

    """
    gpu_components = _connected_components_gpu_runtime_dispatch(G)
    if gpu_components is not None:
        for comp in gpu_components:
            yield comp
        return

    seen = set()
    for v in G:
        if v not in seen:
            c = set(_plain_bfs(G, v))
            seen.update(c)
            yield c


@not_implemented_for("multigraph")
@hybrid("cpp_connected_components_directed")
def connected_components_directed(G):
    """Returns a list of connected components, each of which denotes the edges set of a connected component.

    Parameters
    ----------
    G :  easygraph.DiGraph
    Returns
    -------
    connected_components : list of list
        Each element list is the edges set of a connected component.

    Examples
    --------
    >>> connected_components(G)

    """
    seen = set()
    for v in G:
        if v not in seen:
            c = set(_plain_bfs(G, v))
            seen.update(c)
            yield c


def _generator_connected_components(G):
    seen = set()
    for v in G:
        if v not in seen:
            component = set(_plain_bfs(G, v))
            yield component
            seen.update(component)


@not_implemented_for("multigraph")
def connected_component_of_node(G, node):
    """Returns the connected component that *node* belongs to.

    Parameters
    ----------
    G : easygraph.Graph

    node : object
        The target node

    Returns
    -------
    connected_component_of_node : set
        The connected component that *node* belongs to.

    Examples
    --------
    Returns the connected component of one node `Jack`.

    >>> connected_component_of_node(G, node='Jack')

    """
    return set(_plain_bfs(G, node))


@hybrid("cpp_plain_bfs")
def _plain_bfs(G, source):
    """
    A fast BFS node generator
    """
    G_adj = G.adj
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                yield v
                seen.add(v)
                nextlevel.update(G_adj[v])


def _connected_components_gpu_runtime_dispatch(G):
    if not gpu_runtime_enabled():
        return None

    try:
        from easygraph.utils import gpu_mine_backend as mine_backend

        if mine_backend.mine_backend_enabled():
            return mine_backend.connected_components(G, directed=False)
    except Exception:
        if gpu_strict_errors():
            raise

    if not rapids_backend_enabled():
        return None
    if G.is_directed():
        return None

    try:
        nodes, node_to_idx = build_node_index(G)
        if not nodes:
            return []
        rows = indexed_edges(
            G,
            node_to_idx,
            undirected_projection=True,
            weight_key=None,
        )
        if not rows:
            return [{node} for node in nodes]

        cudf, cugraph = import_rapids()
        edge_df = to_cudf_edgelist(cudf, rows, weighted=False)
        cg = make_cugraph_graph(cugraph, directed=False)
        load_cugraph_edgelist(
            cg,
            cudf,
            edge_df,
            weighted=False,
            num_nodes=len(nodes),
            renumber=False,
        )
        out = cugraph.connected_components(cg)
        return component_sets_from_labels(out, nodes)
    except Exception:
        return None
