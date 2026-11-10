import easygraph as eg

from easygraph.utils import *
from easygraph.utils.gpu_runtime import *


__all__ = [
    "k_core",
]


from typing import TYPE_CHECKING
from typing import List
from typing import Union


if TYPE_CHECKING:
    from easygraph import Graph


@hybrid("cpp_k_core")
def k_core(G: "Graph") -> Union["Graph", List]:
    """Return unweighted core numbers in the graph's node-index order.

    Each value is the largest k for which its vertex belongs to a k-core.
    Edge weight attributes do not affect the degree used for decomposition.
    """
    gpu_result = _k_core_gpu_runtime_dispatch(G)
    if gpu_result is not None:
        return gpu_result

    # Initialize a dictionary to store the degrees of the nodes
    degrees = dict(G.degree(weight=None))
    # Sort nodes by degree.
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    # The initial guess for the core number of a node is its degree.
    core = degrees
    nbrs = {v: list(G.neighbors(v)) for v in G}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                if v not in nbrs[u]:
                    # Directed/non-symmetric neighbor lists can miss reverse edges.
                    # Skip this update to avoid invalid list.remove and preserve stability.
                    continue
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    ret = [0.0 for i in range(len(G))]
    for i in range(len(ret)):
        ret[i] = core[G.index2node[i]]
    return ret


def _k_core_gpu_runtime_dispatch(G):
    if not gpu_runtime_enabled():
        return None
    try:
        from easygraph.utils import gpu_eggpu_backend as eggpu_backend

        if eggpu_backend.eggpu_backend_enabled():
            return eggpu_backend.k_core(G)
    except Exception:
        if gpu_strict_errors() or getattr(G, "_eggpu_bulk_csr", False):
            raise
    return None
