import os

import easygraph as eg

from easygraph.utils import *
from easygraph.utils.gpu_runtime import *


__all__ = ["pagerank"]


@not_implemented_for("multigraph")
@hybrid("cpp_pagerank")
def pagerank(G, alpha=0.85, weight=None, max_iter=None, tol=None):
    """
    Returns the PageRank value of each node in G.

    Parameters
    ----------
    G : graph
        Undirected graph will be considered as directed graph with two directed edges for each undirected edge.

    alpha : float
        The damping factor. Default is 0.85

    weight : None or string, optional (default=None)
        If None, all edge weights are considered equal.
        Otherwise holds the name of the edge attribute used as weight.

    max_iter : int, optional
        Maximum iteration count for power iteration.
        If None, read from environment variable EASYGRAPH_CPU_PR_MAX_ITER (default 200).

    tol : float, optional
        Convergence tolerance for L1 residual.
        If None, read from environment variable EASYGRAPH_CPU_PR_TOL (default 1e-6).
    """
    if max_iter is None:
        max_iter_eff = max(1, int(os.environ.get("EASYGRAPH_CPU_PR_MAX_ITER", "200")))
    else:
        max_iter_eff = max(1, int(max_iter))
    if tol is None:
        tol_eff = max(0.0, float(os.environ.get("EASYGRAPH_CPU_PR_TOL", "1e-6")))
    else:
        tol_eff = max(0.0, float(tol))

    gpu_result = _pagerank_gpu_runtime_dispatch(
        G,
        alpha=alpha,
        weight=weight,
        max_iter=max_iter_eff,
        tol=tol_eff,
    )
    if gpu_result is not None:
        return gpu_result

    if len(G) == 0:
        return {}
    return _pagerank_power_iteration(
        G,
        alpha=alpha,
        weight=weight,
        max_iter=max_iter_eff,
        tol=tol_eff,
    )


def _pagerank_power_iteration(G, alpha=0.85, weight=None, max_iter=200, tol=1.0e-6):
    nodes, node_to_idx = build_node_index(G)
    n = len(nodes)
    if n == 0:
        return {}

    rows = indexed_edges(
        G,
        node_to_idx,
        undirected_projection=False,
        weight_key=weight,
    )
    if not G.is_directed():
        if weight is None:
            rows = [(u, v) for (u, v) in rows if u != v] + [(v, u) for (u, v) in rows if u != v]
        else:
            rows = [(u, v, w) for (u, v, w) in rows if u != v] + [(v, u, w) for (u, v, w) in rows if u != v]

    if not rows:
        uniform = 1.0 / n
        return {node: uniform for node in nodes}

    out_adj = [[] for _ in range(n)]
    out_sums = [0.0] * n
    if weight is None:
        for src, dst in rows:
            out_adj[src].append((dst, 1.0))
            out_sums[src] += 1.0
    else:
        for src, dst, w in rows:
            wf = float(w)
            out_adj[src].append((dst, wf))
            out_sums[src] += wf

    pr = [1.0 / n] * n
    new_pr = [0.0] * n

    for _ in range(max_iter):
        for i in range(n):
            new_pr[i] = 0.0

        dangling_sum = 0.0
        for i in range(n):
            if out_sums[i] <= 1.0e-15:
                dangling_sum += pr[i]
                continue
            base = alpha * pr[i] / out_sums[i]
            for dst, w in out_adj[i]:
                new_pr[dst] += base * w

        jump = (1.0 - alpha) / n + alpha * dangling_sum / n
        err = 0.0
        for i in range(n):
            val = new_pr[i] + jump
            err += abs(val - pr[i])
            pr[i] = val
        if err < tol * n:
            break

    norm = sum(pr)
    if norm <= 0.0:
        uniform = 1.0 / n
        return {node: uniform for node in nodes}
    inv = 1.0 / norm
    return {nodes[i]: float(pr[i] * inv) for i in range(n)}


def google_matrix(G, alpha, weight=None):
    import numpy as np

    M = eg.to_numpy_array(G, weight=weight).astype(float)
    N = len(G)
    if N == 0:
        return M

    # Get dangling nodes(nodes with no out link)
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]
    dangling_weights = np.repeat(1.0 / N, N)
    for node in dangling_nodes:
        M[node] = dangling_weights

    M /= M.sum(axis=1)[:, np.newaxis]

    return alpha * M + (1 - alpha) * np.repeat(1.0 / N, N)


def _pagerank_gpu_runtime_dispatch(G, alpha, weight, max_iter, tol):
    if not gpu_runtime_enabled():
        return None

    try:
        from easygraph.utils import gpu_mine_backend as mine_backend

        if mine_backend.mine_backend_enabled():
            return mine_backend.pagerank(
                G,
                alpha=alpha,
                max_iter=int(max_iter),
                eps=float(tol),
                weight=weight,
            )
    except Exception:
        if gpu_strict_errors():
            raise

    if not rapids_backend_enabled():
        return None

    # First try nx-cugraph (simpler semantics for isolated vertices).
    try:
        import networkx as nx
        import nx_cugraph  # noqa: F401

        g_nx = nx.DiGraph() if G.is_directed() else nx.Graph()
        g_nx.add_nodes_from(graph_nodes_list(G))
        if weight is None:
            g_nx.add_edges_from((u, v) for u, v, _ in G.edges)
            return nx.pagerank(g_nx, alpha=alpha, max_iter=int(max_iter), tol=float(tol), backend="cugraph")
        g_nx.add_edges_from((u, v, {weight: data.get(weight, 1)}) for u, v, data in G.edges)
        return nx.pagerank(
            g_nx,
            alpha=alpha,
            weight=weight,
            max_iter=int(max_iter),
            tol=float(tol),
            backend="cugraph",
        )
    except Exception:
        pass

    # Fallback to native cuGraph if nx-cugraph path is unavailable.
    try:
        nodes, node_to_idx = build_node_index(G)
        if not nodes:
            return {}
        rows = indexed_edges(
            G,
            node_to_idx,
            undirected_projection=not G.is_directed(),
            weight_key=weight,
        )
        if not rows:
            uniform = 1.0 / len(nodes)
            return {node: uniform for node in nodes}

        cudf, cugraph = import_rapids()
        edge_df = to_cudf_edgelist(cudf, rows, weighted=(weight is not None))
        cg = make_cugraph_graph(cugraph, directed=G.is_directed())
        load_cugraph_edgelist(
            cg,
            cudf,
            edge_df,
            weighted=(weight is not None),
            num_nodes=len(nodes),
            renumber=False,
        )
        try:
            pr = cugraph.pagerank(cg, alpha=alpha, max_iter=int(max_iter), tol=float(tol))
        except TypeError:
            pr = cugraph.pagerank(cg, alpha=alpha)
        pr_pdf = pr.to_pandas()
        score_col = "pagerank" if "pagerank" in pr_pdf.columns else pr_pdf.columns[-1]
        result = {node: 0.0 for node in nodes}
        for vertex, value in zip(pr_pdf["vertex"], pr_pdf[score_col]):
            idx = int(vertex)
            if 0 <= idx < len(nodes):
                result[nodes[idx]] = float(value)
        total = sum(result.values())
        if total > 0:
            inv_total = 1.0 / total
            for node in result:
                result[node] *= inv_total
        else:
            uniform = 1.0 / len(nodes)
            result = {node: uniform for node in nodes}
        return result
    except Exception:
        return None
