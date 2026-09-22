#!/usr/bin/env python3
"""Emit deterministic structural-hole results for extension A/B validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import easygraph as eg


FUNCTIONS = {
    "EffectiveSize": eg.effective_size,
    "Efficiency": eg.efficiency,
    "Constraint": eg.constraint,
    "Hierarchy": eg.hierarchy,
}


def build_graph(directed: bool):
    graph = eg.DiGraph() if directed else eg.Graph()
    nodes = [f"node-{index}" for index in range(9)]
    graph.add_nodes(nodes)
    edges = [
        (nodes[0], nodes[1]),
        (nodes[1], nodes[2]),
        (nodes[2], nodes[0]),
        (nodes[2], nodes[3]),
        (nodes[3], nodes[4]),
        (nodes[4], nodes[2]),
        (nodes[4], nodes[5]),
        (nodes[5], nodes[6]),
        (nodes[6], nodes[4]),
        (nodes[6], nodes[7]),
    ]
    if directed:
        edges.extend(
            [
                (nodes[1], nodes[0]),
                (nodes[3], nodes[2]),
                (nodes[7], nodes[6]),
            ]
        )
    graph.add_edges(edges)
    return graph, nodes


def encode(value):
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import cpp_easygraph

    payload = {
        "extension": str(Path(cpp_easygraph.__file__).resolve()),
        "cases": {},
    }
    for directed in (False, True):
        graph, nodes = build_graph(directed)
        case = {}
        for scope, selected in (
            ("all", None),
            ("subset", [nodes[0], nodes[2], nodes[4], nodes[7], nodes[8]]),
        ):
            results = {}
            for name, function in FUNCTIONS.items():
                values = function(graph, nodes=selected)
                keys = nodes if selected is None else selected
                results[name] = {key: encode(values[key]) for key in keys}
            case[scope] = results
        payload["cases"]["directed" if directed else "undirected"] = case

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
