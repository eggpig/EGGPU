#!/usr/bin/env python3
"""Emit deterministic low/high-degree structural-hole results for A/B checks."""

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


def build_graph():
    graph = eg.DiGraph()
    nodes = [f"node-{index}" for index in range(512)]
    graph.add_nodes(nodes)
    edges = []
    for target in range(1, 321):
        edges.append((nodes[0], nodes[target]))
        if target % 3 == 0:
            edges.append((nodes[target], nodes[0]))
    for target in range(2, 258):
        edges.append((nodes[1], nodes[target]))
        if target % 5 == 0:
            edges.append((nodes[target], nodes[1]))
    for source in range(2, 400):
        edges.append((nodes[source], nodes[2 + ((source + 1) % 398)]))
        if source % 7 == 0:
            edges.append((nodes[source], nodes[2 + ((source + 17) % 398)]))
    graph.add_edges(edges)
    return graph, nodes


def encode(value):
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import cpp_easygraph

    graph, nodes = build_graph()
    payload = {
        "extension": str(Path(cpp_easygraph.__file__).resolve()),
        "node_count": len(nodes),
        "edge_count": graph.size(),
        "cases": {},
    }
    scopes = {
        "all": None,
        "subset": [
            nodes[0],
            nodes[1],
            nodes[2],
            nodes[127],
            nodes[320],
            nodes[400],
            nodes[511],
        ],
    }
    for scope, selected in scopes.items():
        keys = nodes if selected is None else selected
        payload["cases"][scope] = {}
        for name, function in FUNCTIONS.items():
            values = function(graph, nodes=selected, weight=None)
            payload["cases"][scope][name] = {
                key: encode(values[key]) for key in keys
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
