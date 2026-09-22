#!/usr/bin/env python3
"""Validate single-source EGGPU Dijkstra on small boundary cases."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def canonical_distances(result):
    return {
        str(node): float(distance)
        for node, distance in sorted(dict(result).items(), key=lambda item: str(item[0]))
    }


def run_dijkstra(eg, graph, source, target, gpu_enabled):
    previous = os.environ.get("EASYGRAPH_ENABLE_GPU")
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE" if gpu_enabled else "FALSE"
    try:
        return canonical_distances(
            eg.single_source_dijkstra(
                graph,
                source,
                weight="weight",
                target=target,
            )
        )
    finally:
        if previous is None:
            os.environ.pop("EASYGRAPH_ENABLE_GPU", None)
        else:
            os.environ["EASYGRAPH_ENABLE_GPU"] = previous


def make_graph(eg, nodes, edges):
    graph = eg.DiGraph()
    graph.add_nodes(nodes)
    for source, target, weight in edges:
        graph.add_edge(source, target, weight=float(weight))
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--easygraph-repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime = Path(args.easygraph_repo).expanduser().resolve()
    sys.path.insert(0, str(runtime))
    importlib.util.find_spec("easygraph")

    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
    os.environ["EASYGRAPH_GPU_ADAPTIVE_HOST"] = "FALSE"
    os.environ["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"

    import cpp_easygraph
    import easygraph as eg

    cases = [
        {
            "name": "reachable_and_unreachable",
            "nodes": [0, 1, 2, 3, 4],
            "edges": [(0, 1, 2), (1, 2, 3), (0, 2, 10), (3, 4, 1)],
            "source": 0,
            "target": None,
        },
        {
            "name": "reachable_target",
            "nodes": [0, 1, 2, 3, 4],
            "edges": [(0, 1, 2), (1, 2, 3), (0, 2, 10), (3, 4, 1)],
            "source": 0,
            "target": 2,
        },
        {
            "name": "unreachable_target",
            "nodes": [0, 1, 2, 3, 4],
            "edges": [(0, 1, 2), (1, 2, 3), (0, 2, 10), (3, 4, 1)],
            "source": 0,
            "target": 4,
        },
        {
            "name": "edgeless_graph",
            "nodes": [0, 1, 2],
            "edges": [],
            "source": 1,
            "target": None,
        },
        {
            "name": "edgeless_unreachable_target",
            "nodes": [0, 1, 2],
            "edges": [],
            "source": 1,
            "target": 2,
        },
        {
            "name": "single_node_target",
            "nodes": [42],
            "edges": [],
            "source": 42,
            "target": 42,
        },
    ]

    results = []
    for case in cases:
        print(f"[case] {case['name']}", flush=True)
        gpu_graph = make_graph(eg, case["nodes"], case["edges"])
        cpu_graph = make_graph(eg, case["nodes"], case["edges"])
        gpu = run_dijkstra(
            eg,
            gpu_graph,
            case["source"],
            case["target"],
            gpu_enabled=True,
        )
        cpu = run_dijkstra(
            eg,
            cpu_graph,
            case["source"],
            case["target"],
            gpu_enabled=False,
        )
        passed = cpu == gpu
        results.append(
            {
                "name": case["name"],
                "source": case["source"],
                "target": case["target"],
                "cpu": cpu,
                "gpu": gpu,
                "pass": passed,
            }
        )
        if not passed:
            raise AssertionError(
                f"{case['name']}: CPU={cpu!r}, EGGPU={gpu!r}"
            )

    empty_graph = eg.DiGraph()
    empty_outcome = {}
    for backend, enabled in (("cpu", False), ("gpu", True)):
        try:
            value = run_dijkstra(
                eg,
                empty_graph,
                source=0,
                target=None,
                gpu_enabled=enabled,
            )
        except Exception as exc:
            empty_outcome[backend] = {
                "raised": True,
                "type": type(exc).__name__,
                "message": str(exc),
            }
        else:
            empty_outcome[backend] = {"raised": False, "value": value}
    if not empty_outcome["cpu"]["raised"] or not empty_outcome["gpu"]["raised"]:
        raise AssertionError(
            f"empty graph should reject a missing source: {empty_outcome!r}"
        )

    extension = Path(cpp_easygraph.__file__).resolve()
    payload = {
        "status": "pass",
        "runtime": str(runtime),
        "easygraph_origin": str(Path(eg.__file__).resolve()),
        "cpp_easygraph_origin": str(extension),
        "cpp_easygraph_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "cases": results,
        "empty_graph_missing_source": empty_outcome,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
