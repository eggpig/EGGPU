#!/usr/bin/env python3
"""Deterministic, non-timing correctness smoke for the 16 EGGPU calls."""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections.abc import Mapping

import easygraph as eg


@contextlib.contextmanager
def gpu_enabled(enabled: bool):
    previous_enable = os.environ.get("EASYGRAPH_ENABLE_GPU")
    previous_strict = os.environ.get("EASYGRAPH_GPU_STRICT_ERRORS")
    if enabled:
        os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
        os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
    else:
        os.environ.pop("EASYGRAPH_ENABLE_GPU", None)
        os.environ.pop("EASYGRAPH_GPU_STRICT_ERRORS", None)
    try:
        yield
    finally:
        for name, value in (
            ("EASYGRAPH_ENABLE_GPU", previous_enable),
            ("EASYGRAPH_GPU_STRICT_ERRORS", previous_strict),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def directed_graph():
    graph = eg.DiGraph()
    graph.add_weighted_edges_from(
        [
            (0, 1, 1.0),
            (1, 2, 2.0),
            (2, 0, 0.5),
            (2, 3, 1.5),
            (3, 4, 1.0),
            (4, 3, 0.75),
            (1, 4, 4.0),
        ]
    )
    return graph


def undirected_graph():
    graph = eg.Graph()
    graph.add_weighted_edges_from(
        [
            (0, 1, 1.0),
            (1, 2, 2.0),
            (2, 0, 0.5),
            (2, 3, 1.5),
            (3, 4, 1.0),
            (4, 5, 2.5),
            (5, 3, 0.75),
            (1, 5, 4.0),
        ]
    )
    return graph


def unweighted_undirected_graph():
    graph = eg.Graph()
    graph.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 0),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 3),
            (1, 5),
        ]
    )
    return graph


def canonical(value):
    if isinstance(value, Mapping):
        return {key: canonical(item) for key, item in value.items()}
    if hasattr(value, "edges") and hasattr(value, "nodes"):
        edges = []
        for source, target, data in value.edges:
            weight = float(data.get("weight", 1.0))
            edge = (source, target) if value.is_directed() else tuple(sorted((source, target)))
            edges.append((edge[0], edge[1], weight))
        return sorted(edges)
    if isinstance(value, (set, frozenset)):
        return sorted(canonical(item) for item in value)
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (set, frozenset)) for item in value):
            return sorted(tuple(sorted(item)) for item in value)
        return [canonical(item) for item in value]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        materialized = list(value)
        if materialized and all(isinstance(item, (set, frozenset)) for item in materialized):
            return sorted(tuple(sorted(item)) for item in materialized)
        return [canonical(item) for item in materialized]
    return value


def assert_close(actual, expected, path="result"):
    actual = canonical(actual)
    expected = canonical(expected)
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{path}: key mismatch")
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}[{key!r}]")
        return
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: length {len(actual)} != {len(expected)}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_close(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(actual), float(expected), rel_tol=2.0e-4, abs_tol=2.0e-5):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def cases():
    return [
        ("PageRank", directed_graph, lambda g: eg.pagerank(g, alpha=0.75, weight="weight", max_iter=200, tol=1e-6)),
        ("BC", undirected_graph, lambda g: eg.betweenness_centrality(g, sources=[0, 2], weight="weight", normalized=False)),
        ("Closeness", directed_graph, lambda g: eg.closeness_centrality(g, weight="weight", sources=[0, 2])),
        ("LCC", undirected_graph, lambda g: eg.clustering(g)),
        ("WCC", directed_graph, lambda g: list(eg.weakly_connected_components(g))),
        ("SCC", directed_graph, lambda g: list(eg.strongly_connected_components(g))),
        ("KCore", unweighted_undirected_graph, lambda g: eg.k_core(g)),
        ("MST", undirected_graph, lambda g: eg.minimum_spanning_tree(g, weight="weight")),
        ("BFS", directed_graph, lambda g: eg.multi_source_bfs(g, [0, 3])),
        ("Dijkstra", directed_graph, lambda g: eg.single_source_dijkstra(g, 0, weight="weight")),
        ("BellmanFord", directed_graph, lambda g: eg.multi_source_bellman_ford(g, [0, 3], weight="weight")),
        ("SSSP", directed_graph, lambda g: eg.multi_source_dijkstra(g, [0, 3], weight="weight")),
        ("EffectiveSize", undirected_graph, lambda g: eg.effective_size(g, weight="weight")),
        ("Efficiency", undirected_graph, lambda g: eg.efficiency(g, weight="weight")),
        ("Constraint", undirected_graph, lambda g: eg.constraint(g, weight="weight")),
        ("Hierarchy", undirected_graph, lambda g: eg.hierarchy(g, weight="weight")),
    ]


def main():
    passed = []
    for name, graph_factory, function in cases():
        with gpu_enabled(False):
            expected = function(graph_factory())
        with gpu_enabled(True):
            actual = function(graph_factory())
        assert_close(actual, expected, name)
        passed.append(name)
        print(f"PASS {name}", flush=True)
    print(json.dumps({"status": "PASS", "functions": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
