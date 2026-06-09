#!/usr/bin/env python3
"""Preflight correctness check for EGGPU structural-hole scan-v kernels.

The check compares EGGPU against EasyGraph CPU semantics on:

1. a small directed graph covering ordinary directed structural-hole behavior;
2. a high-degree synthetic directed graph that triggers the neighbor-side scan
   branch added for skewed hub graphs such as wiki-Talk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Callable, Dict, Iterable, Tuple

import easygraph as eg
from easygraph.functions.structural_holes import (
    constraint,
    effective_size,
    efficiency,
    hierarchy,
)


MetricFn = Callable[[eg.Graph], Dict[object, float]]


def _small_directed_graph() -> eg.DiGraph:
    g = eg.DiGraph()
    g.add_edges_from(
        [
            (0, 1),
            (1, 0),
            (0, 2),
            (0, 3),
            (4, 0),
            (5, 0),
            (1, 2),
            (2, 1),
            (3, 2),
            (2, 4),
            (4, 5),
            (5, 3),
        ]
    )
    return g


def _high_degree_directed_graph(n_neighbors: int = 320) -> eg.DiGraph:
    g = eg.DiGraph()
    for i in range(1, n_neighbors + 1):
        g.add_edge(0, i)
        if i % 7 == 0:
            g.add_edge(i, 0)
        if i < n_neighbors:
            g.add_edge(i, i + 1)
        if i % 11 == 0 and i + 5 <= n_neighbors:
            g.add_edge(i, i + 5)
    for i in range(1, min(80, n_neighbors - 80) + 1):
        g.add_edge(i + 80, i)
    return g


def _set_gpu_enabled(enabled: bool) -> None:
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE" if enabled else "FALSE"
    if enabled:
        os.environ["EASYGRAPH_GPU_BACKEND"] = "mine"


def _compare(cpu: Dict[object, float], gpu: Dict[object, float], tol: float) -> Tuple[float, int]:
    max_diff = 0.0
    bad = 0
    for node, a in cpu.items():
        b = gpu[node]
        if math.isnan(float(a)) and math.isnan(float(b)):
            continue
        diff = abs(float(a) - float(b))
        max_diff = max(max_diff, diff)
        if diff > tol:
            bad += 1
    return max_diff, bad


def _run_case(name: str, graph: eg.DiGraph, tol: float) -> Iterable[dict]:
    metrics: Tuple[Tuple[str, MetricFn], ...] = (
        ("effective_size", effective_size),
        ("efficiency", efficiency),
        ("constraint", constraint),
        ("hierarchy", hierarchy),
    )
    for metric_name, fn in metrics:
        _set_gpu_enabled(False)
        cpu = fn(graph)
        _set_gpu_enabled(True)
        gpu = fn(graph)
        max_diff, bad = _compare(cpu, gpu, tol)
        yield {
            "case": name,
            "metric": metric_name,
            "nodes": len(graph),
            "edges": graph.size(),
            "max_abs_diff": max_diff,
            "bad_count": bad,
            "status": "ok" if bad == 0 else "fail",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--high-degree-neighbors", type=int, default=320)
    args = parser.parse_args()

    results = []
    results.extend(_run_case("small_directed", _small_directed_graph(), args.tol))
    results.extend(
        _run_case(
            "high_degree_directed",
            _high_degree_directed_graph(args.high_degree_neighbors),
            args.tol,
        )
    )

    for row in results:
        print("RESULT_JSON " + json.dumps(row, sort_keys=True))

    return 0 if all(row["status"] == "ok" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
