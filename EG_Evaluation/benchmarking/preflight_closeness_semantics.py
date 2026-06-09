#!/usr/bin/env python3
"""CPU-only Closeness semantics preflight for benchmark baselines.

EasyGraph's directed closeness uses outgoing distances. NetworkX's directed
closeness is inward by default, so the benchmark uses a reverse graph view.
python-igraph supports outgoing mode, but its normalized result needs the
Wasserman-Faust disconnected-graph correction to match EasyGraph/NetworkX.
"""

from __future__ import annotations

import json
import math
import os
from typing import Iterable


def _set_gpu_disabled() -> None:
    os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"
    os.environ.pop("EASYGRAPH_GPU_BACKEND", None)


def _nodes_list(g) -> list:
    nodes = g.nodes
    if hasattr(nodes, "keys"):
        return list(nodes.keys())
    return list(nodes)


def _easygraph_values(eg, g, sources=None) -> list[float]:
    return [float(x) for x in eg.closeness_centrality(g, weight=None, sources=sources)]


def _cpp_values(g, sources=None) -> list[float]:
    import cpp_easygraph

    result = cpp_easygraph.cpp_closeness_centrality(
        g.cpp(),
        weight=None,
        cutoff=None,
        sources=sources,
    )
    if isinstance(result, dict):
        result = result.get("values_dense", [])
    return [float(x) for x in result]


def _networkx_values(nx, eg_g, nx_g) -> list[float]:
    close_graph = nx_g.reverse(copy=False) if nx_g.is_directed() else nx_g
    values = nx.closeness_centrality(close_graph, distance=None, wf_improved=True)
    return [float(values[node]) for node in _nodes_list(eg_g)]


def _networkx_subset_values(nx, eg_g, nx_g, sources) -> list[float]:
    close_graph = nx_g.reverse(copy=False) if nx_g.is_directed() else nx_g
    values = nx.closeness_centrality(close_graph, distance=None, wf_improved=True)
    return [float(values[node]) for node in sources]


def _igraph_values(ig, eg_g, edges, directed) -> list[float]:
    nodes = _nodes_list(eg_g)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    remapped_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]
    g = ig.Graph(n=len(nodes), edges=remapped_edges, directed=directed)
    mode = "OUT" if directed else "ALL"
    raw = g.closeness(vertices=None, mode=mode, cutoff=None, weights=None, normalized=True)
    reach = g.neighborhood_size(vertices=None, order=g.vcount(), mode=mode)
    denom = max(1, g.vcount() - 1)
    out = []
    for value, count in zip(raw, reach):
        try:
            x = float(value)
        except Exception:
            x = 0.0
        if not math.isfinite(x):
            x = 0.0
        x *= max(0, int(count) - 1) / float(denom)
        out.append(x)
    return out


def _igraph_subset_values(ig, eg_g, edges, directed, sources) -> list[float]:
    values = _igraph_values(ig, eg_g, edges, directed)
    node_to_pos = {node: idx for idx, node in enumerate(_nodes_list(eg_g))}
    return [values[node_to_pos[node]] for node in sources]


def _max_abs_diff(a: Iterable[float], b: Iterable[float]) -> float:
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def _case_specs():
    return [
        {
            "name": "directed_outward",
            "directed": True,
            "nodes": [0, 1, 2, 3, 4],
            "edges": [(0, 1), (1, 2), (3, 2), (2, 4), (4, 1)],
        },
        {
            "name": "undirected_disconnected",
            "directed": False,
            "nodes": [0, 1, 2, 3, 10, 11],
            "edges": [(0, 1), (1, 2), (2, 3), (10, 11)],
        },
    ]


def main() -> int:
    _set_gpu_disabled()

    import easygraph as eg
    import igraph as ig
    import networkx as nx

    rows = []
    for spec in _case_specs():
        graph_cls = eg.DiGraph if spec["directed"] else eg.Graph
        eg_g = graph_cls()
        eg_g.add_nodes_from(spec["nodes"])
        eg_g.add_edges_from(spec["edges"])

        nx_cls = nx.DiGraph if spec["directed"] else nx.Graph
        nx_g = nx_cls()
        nx_g.add_nodes_from(spec["nodes"])
        nx_g.add_edges_from(spec["edges"])

        ref = _easygraph_values(eg, eg_g)
        candidates = {
            "easygraph_cpp": _cpp_values(eg_g),
            "networkx": _networkx_values(nx, eg_g, nx_g),
            "igraph": _igraph_values(ig, eg_g, spec["edges"], spec["directed"]),
        }
        for baseline, values in candidates.items():
            max_diff = _max_abs_diff(ref, values)
            rows.append(
                {
                    "case": spec["name"],
                    "baseline": baseline,
                    "nodes": len(spec["nodes"]),
                    "edges": len(spec["edges"]),
                    "max_abs_diff": max_diff,
                    "status": "ok" if max_diff <= 1e-12 else "fail",
                }
            )

        subset = [spec["nodes"][0], spec["nodes"][-1]]
        ref_subset = _easygraph_values(eg, eg_g, sources=subset)
        subset_candidates = {
            "easygraph_cpp": _cpp_values(eg_g, sources=subset),
            "networkx": _networkx_subset_values(nx, eg_g, nx_g, subset),
            "igraph": _igraph_subset_values(ig, eg_g, spec["edges"], spec["directed"], subset),
        }
        for baseline, values in subset_candidates.items():
            max_diff = _max_abs_diff(ref_subset, values)
            rows.append(
                {
                    "case": spec["name"] + "_sources_subset",
                    "baseline": baseline,
                    "nodes": len(subset),
                    "edges": len(spec["edges"]),
                    "max_abs_diff": max_diff,
                    "status": "ok" if max_diff <= 1e-12 else "fail",
                }
            )

    for row in rows:
        print("RESULT_JSON " + json.dumps(row, sort_keys=True))
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
