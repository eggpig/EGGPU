from easygraph.functions.basic import *
from easygraph.functions.path import single_source_bfs
from easygraph.functions.path import single_source_dijkstra
from easygraph.utils import *
from easygraph.utils.gpu_runtime import *


__all__ = [
    "closeness_centrality",
]


def closeness_centrality_parallel(nodes, G, path_length):
    ret = []
    length = len(G)
    for node in nodes:
        x = path_length(G, node)
        dist = sum(x.values())
        cnt = len(x)
        if dist == 0:
            ret.append([node, 0])
        else:
            ret.append([node, (cnt - 1) * (cnt - 1) / (dist * (length - 1))])
    return ret


@not_implemented_for("multigraph")
@hybrid("cpp_closeness_centrality")
def closeness_centrality(G, weight=None, sources=None, n_workers=None):
    r"""
    Compute closeness centrality for nodes.

    .. math::

        C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    Notice that the closeness distance function computes the
    outcoming distance to `u` for directed graphs. To use
    incoming distance, act on `G.reverse()`.

    Parameters
    ----------
    G : graph
      A easygraph graph

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    sources : None or nodes list, optional (default=None)
      If None, all nodes are returned
      Otherwise,the set of source vertices to creturn.

    Returns
    -------
    scores : list of float
      Closeness scores ordered by ``sources`` when it is provided, otherwise
      by the graph's internal node order (``G.index2node``).
    """
    gpu_result = _closeness_centrality_gpu_runtime_dispatch(
        G,
        weight=weight,
        sources=sources,
        n_workers=n_workers,
    )
    if gpu_result is not None:
        return gpu_result

    closeness = dict()
    if sources is not None:
        output_nodes = list(sources)
    else:
        output_nodes = [G.index2node[i] for i in range(len(G))]
    nodes = list(output_nodes)
    length = len(G)
    import functools

    if weight is not None:
        path_length = functools.partial(single_source_dijkstra, weight=weight)
    else:
        path_length = functools.partial(single_source_bfs)

    if n_workers is not None:
        # use parallel version for large graph
        import random

        from functools import partial
        from multiprocessing import Pool

        random.shuffle(nodes)

        if len(nodes) > n_workers * 30000:
            nodes = split_len(nodes, step=30000)
        else:
            nodes = split(nodes, n_workers)
        local_function = partial(
            closeness_centrality_parallel, G=G, path_length=path_length
        )
        with Pool(n_workers) as p:
            ret = p.imap(local_function, nodes)
            res = [x for i in ret for x in i]
        closeness = dict(res)
    else:
        # use np-parallel version for small graph
        for node in nodes:
            x = path_length(G, node)
            dist = sum(x.values())
            cnt = len(x)
            if dist == 0:
                closeness[node] = 0
            else:
                closeness[node] = (cnt - 1) * (cnt - 1) / (dist * (length - 1))
    return [closeness[node] for node in output_nodes]


def _closeness_centrality_gpu_runtime_dispatch(G, weight=None, sources=None, n_workers=None):
    if not gpu_runtime_enabled():
        return None
    if n_workers is not None:
        if gpu_strict_errors():
            raise RuntimeError("EGGPU closeness does not support n_workers")
        return None
    try:
        from easygraph.utils import gpu_eggpu_backend as eggpu_backend

        if eggpu_backend.eggpu_backend_enabled():
            values = eggpu_backend.closeness_centrality(
                G,
                weight=weight,
                sources=sources,
            )
            if hasattr(values, "tolist"):
                values = values.tolist()
            if isinstance(values, dict):
                output_nodes = (
                    list(sources)
                    if sources is not None
                    else [G.index2node[i] for i in range(len(G))]
                )
                return [float(values[node]) for node in output_nodes]
            return [float(value) for value in values]
    except Exception:
        if gpu_strict_errors():
            raise
    return None
