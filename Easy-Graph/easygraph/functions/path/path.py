from easygraph.utils import *
from easygraph.utils.decorators import *
from easygraph.utils.gpu_runtime import *


__all__ = [
    "BFS",
    "bfs",
    "Dijkstra",
    "Floyd",
    "Prim",
    "Kruskal",
    "Spfa",
    "single_source_bfs",
    "multi_source_bfs",
    "single_source_dijkstra",
    "multi_source_dijkstra",
    "BellmanFord",
    "single_source_bellman_ford",
    "multi_source_bellman_ford",
]


@not_implemented_for("multigraph")
def BFS(G, node, target=None):
    """Returns unweighted shortest path lengths from one source node."""
    return single_source_bfs(G, node, target=target)


@not_implemented_for("multigraph")
def bfs(G, source, target=None):
    """Alias for :func:`single_source_bfs`."""
    return single_source_bfs(G, source, target=target)


@hybrid("cpp_spfa")
def Spfa(G, node, weight="weight"):
    raise EasyGraphError("Please input GraphC or DiGraphC.")


@not_implemented_for("multigraph")
def Dijkstra(G, node, weight="weight"):
    """Returns the length of paths from the certain node to remaining nodes

    Parameters
    ----------
    G : graph
        weighted graph
    node : int

    Returns
    -------
    result_dict : dict
        the length of paths from the certain node to remaining nodes

    Examples
    --------
    Returns the length of paths from node 1 to remaining nodes

    >>> Dijkstra(G,node=1,weight="weight")

    """
    return single_source_dijkstra(G, node, weight=weight)


@not_implemented_for("multigraph")
def BellmanFord(G, node, weight="weight"):
    """Returns Bellman-Ford shortest path lengths from one source node."""
    return single_source_bellman_ford(G, node, weight=weight)


@not_implemented_for("multigraph")
@only_implemented_for_UnDirected_graph
@hybrid("cpp_Floyd")
def Floyd(G, weight="weight"):
    """Returns the length of paths from all nodes to remaining nodes

    Parameters
    ----------
    G : graph
        weighted graph

    Returns
    -------
    result_dict : dict
        the length of paths from all nodes to remaining nodes

    Examples
    --------
    Returns the length of paths from all nodes to remaining nodes

    >>> Floyd(G,weight="weight")

    """
    adj = G.adj.copy()
    result_dict = {}
    for i in G:
        result_dict[i] = {}
    for i in G:
        temp_key = adj[i].keys()
        for j in G:
            if j in temp_key:
                result_dict[i][j] = adj[i][j].get(weight, 1)
            else:
                result_dict[i][j] = float("inf")
            if i == j:
                result_dict[i][i] = 0
    for k in G:
        for i in G:
            for j in G:
                temp = result_dict[i][k] + result_dict[k][j]
                if result_dict[i][j] > temp:
                    result_dict[i][j] = temp
    return result_dict


@not_implemented_for("multigraph")
@only_implemented_for_UnDirected_graph
@hybrid("cpp_Prim")
def Prim(G, weight="weight"):
    """Returns the edges that make up the minimum spanning tree

    Parameters
    ----------
    G : graph
        weighted graph

    Returns
    -------
    result_dict : dict
        the edges that make up the minimum spanning tree

    Examples
    --------
    Returns the edges that make up the minimum spanning tree

    >>> Prim(G,weight="weight")

    """
    adj = G.adj.copy()
    result_dict = {}
    for i in G:
        result_dict[i] = {}
    selected = []
    candidate = []
    for i in G:
        if not selected:
            selected.append(i)
        else:
            candidate.append(i)
    while len(candidate):
        start = None
        end = None
        min_weight = float("inf")
        for i in selected:
            for j in candidate:
                if i in G and j in G[i] and adj[i][j].get(weight, 1) < min_weight:
                    start = i
                    end = j
                    min_weight = adj[i][j].get(weight, 1)
        if start != None and end != None:
            result_dict[start][end] = min_weight
            selected.append(end)
            candidate.remove(end)
        else:
            break
    return result_dict


@not_implemented_for("multigraph")
@only_implemented_for_UnDirected_graph
@hybrid("cpp_Kruskal")
def Kruskal(G, weight="weight"):
    """Returns the edges that make up the minimum spanning tree

    Parameters
    ----------
    G : graph
        weighted graph

    Returns
    -------
    result_dict : dict
        the edges that make up the minimum spanning tree

    Examples
    --------
    Returns the edges that make up the minimum spanning tree

    >>> Kruskal(G,weight="weight")

    """
    adj = G.adj.copy()
    result_dict = {}
    edge_list = []
    for i in G:
        result_dict[i] = {}
    for i in G:
        for j in G[i]:
            wt = adj[i][j].get(weight, 1)
            edge_list.append([i, j, wt])
    edge_list.sort(key=lambda a: a[2])
    group = [[i] for i in G]
    for edge in edge_list:
        for i in range(len(group)):
            if edge[0] in group[i]:
                m = i
            if edge[1] in group[i]:
                n = i
        if m != n:
            result_dict[edge[0]][edge[1]] = edge[2]
            group[m] = group[m] + group[n]
            group[n] = []
    return result_dict


@not_implemented_for("multigraph")
def single_source_bfs(G, source, target=None):
    """Return a source-to-node hop-distance mapping.

    GPU execution may return an immutable dense-backed mapping. Use
    ``dict(result)`` when an independently mutable dictionary is required.
    """
    gpu_result = _path_gpu_runtime_dispatch(
        "single_source_bfs",
        G,
        source=source,
        target=target,
    )
    if gpu_result is not None:
        return gpu_result
    nextlevel = {source: 0}
    return dict(_single_source_bfs(G.adj, nextlevel, target=target))


def _single_source_bfs(adj, firstlevel, target=None):
    seen = {}
    level = 0
    nextlevel = firstlevel

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            if v not in seen:
                seen[v] = level
                nextlevel.update(adj[v])
                yield (v, level)
                if v == target:
                    break
        level += 1
    del seen


@not_implemented_for("multigraph")
def multi_source_bfs(G, sources, target=None):
    """Return a source-to-distance-mapping result.

    GPU execution may return immutable dense-backed mappings.
    """
    gpu_result = _path_gpu_runtime_dispatch(
        "multi_source_bfs",
        G,
        sources=sources,
        target=target,
    )
    if gpu_result is not None:
        return gpu_result
    return {source: single_source_bfs(G, source, target=target) for source in sources}


@not_implemented_for("multigraph")
def single_source_dijkstra(G, source, weight="weight", target=None):
    """Return weighted distances from ``source`` as a node-value mapping.

    GPU execution may return an immutable dense-backed mapping. Use
    ``dict(result)`` when an independently mutable dictionary is required.
    """
    gpu_result = _path_gpu_runtime_dispatch(
        "single_source_dijkstra",
        G,
        source=source,
        weight=weight,
        target=target,
    )
    if gpu_result is not None:
        return gpu_result
    from heapq import heappop
    from heapq import heappush

    push = heappush
    pop = heappop
    adj = G.adj
    dist = {}
    seen = {}
    from itertools import count

    c = count()
    Q = []
    seen[source] = 0
    push(Q, (0, next(c), source))
    while Q:
        (d, _, v) = pop(Q)
        if v in dist:
            continue
        dist[v] = d
        if v == target:
            break
        for u in adj[v]:
            cost = adj[v][u].get(weight, 1)
            vu_dist = dist[v] + cost
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError("Contradictory paths found:", "negative weights?")
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(Q, (vu_dist, next(c), u))
            else:
                continue
    return dist


@not_implemented_for("multigraph")
@hybrid("cpp_dijkstra_multisource")
def multi_source_dijkstra(G, sources, weight="weight", target=None):
    """Return a source-to-distance-mapping result.

    GPU execution may return immutable dense-backed mappings.
    """
    gpu_result = _multi_source_dijkstra_gpu_runtime_dispatch(
        G,
        sources=sources,
        weight=weight,
        target=target,
    )
    if gpu_result is not None:
        return gpu_result
    return {
        source: single_source_dijkstra(G, source, weight, target) for source in sources
    }


@not_implemented_for("multigraph")
def single_source_bellman_ford(G, source, weight="weight", target=None):
    """Return Bellman--Ford distances as a node-value mapping.

    GPU execution may return an immutable dense-backed mapping.
    """
    gpu_result = _path_gpu_runtime_dispatch(
        "single_source_bellman_ford",
        G,
        source=source,
        weight=weight,
        target=target,
    )
    if gpu_result is not None:
        return gpu_result
    return _single_source_bellman_ford_cpu(G, source, weight=weight, target=target)


@not_implemented_for("multigraph")
def multi_source_bellman_ford(G, sources, weight="weight", target=None):
    """Return a source-to-Bellman--Ford-distance-mapping result.

    GPU execution may return immutable dense-backed mappings.
    """
    gpu_result = _path_gpu_runtime_dispatch(
        "multi_source_bellman_ford",
        G,
        sources=sources,
        weight=weight,
        target=target,
    )
    if gpu_result is not None:
        return gpu_result
    return {
        source: _single_source_bellman_ford_cpu(G, source, weight=weight, target=target)
        for source in sources
    }


def _single_source_bellman_ford_cpu(G, source, weight="weight", target=None):
    adj = G.adj
    if source not in adj:
        raise EasyGraphError("source node should exist in the graph")
    nodes = list(G)
    dist = {source: 0.0}
    for _ in range(max(0, len(nodes) - 1)):
        changed = False
        for u in nodes:
            du = dist.get(u)
            if du is None:
                continue
            for v in adj[u]:
                data = adj[u][v]
                cost = 1 if weight is None else data.get(weight, 1)
                nd = du + cost
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    changed = True
        if not changed:
            break

    for u in nodes:
        du = dist.get(u)
        if du is None:
            continue
        for v in adj[u]:
            data = adj[u][v]
            cost = 1 if weight is None else data.get(weight, 1)
            if du + cost < dist.get(v, float("inf")):
                raise EasyGraphError("Negative weight cycle detected")
    if target is not None and target in dist:
        return {target: dist[target]}
    return dist


def _multi_source_dijkstra_gpu_runtime_dispatch(G, sources, weight="weight", target=None):
    return _path_gpu_runtime_dispatch(
        "multi_source_dijkstra",
        G,
        sources=sources,
        weight=weight,
        target=target,
    )


def _path_gpu_runtime_dispatch(name, G, **kwargs):
    if not gpu_runtime_enabled():
        return None
    try:
        from easygraph.utils import gpu_eggpu_backend as eggpu_backend

        if not eggpu_backend.eggpu_backend_enabled():
            return None
        return getattr(eggpu_backend, name)(G, **kwargs)
    except Exception:
        if gpu_strict_errors() or getattr(G, "_eggpu_bulk_csr", False):
            raise
        return None
