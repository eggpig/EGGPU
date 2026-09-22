#!/usr/bin/env python3
"""Cross-check all timed nx-cugraph adapters against NetworkX on one graph."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from library_baselines import (
    NX_CUGRAPH_DEVICE_BOUNDARIES,
    build_networkx,
    build_nxcugraph_native,
    configure_nx_cugraph_cuda_runtime,
    deterministic_weighted_edges,
    load_graph,
    nx_cugraph_call,
    nx_cugraph_expected_interval_count,
    pick_sources,
)
from nxcugraph_device_timer import NxCugraphDeviceTimer


FUNCTIONS = (
    "PageRank",
    "LCC",
    "WCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
)


def numeric_mapping_error(reference, observed):
    if set(reference) != set(observed):
        raise RuntimeError("mapping key set mismatch")
    return max(
        (
            abs(float(reference[key]) - float(observed[key]))
            for key in reference
        ),
        default=0.0,
    )


def canonical_components(components):
    return sorted(tuple(sorted(int(node) for node in component)) for component in components)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edge_path", type=Path)
    parser.add_argument(
        "--graph-type", choices=("directed", "undirected"), required=True
    )
    parser.add_argument("--source-count", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cuda_runtime = configure_nx_cugraph_cuda_runtime()
    import networkx as nx

    nx.config.fallback_to_nx = False
    nx.config.cache_converted_graphs = True
    views = load_graph(args.edge_path)
    n, directed_edges, undirected_edges = views["clean"]
    directed = args.graph_type == "directed"
    base_edges = directed_edges if directed else undirected_edges
    sources = pick_sources(n, args.source_count)
    records = []

    for function in FUNCTIONS:
        weighted = function in {"Dijkstra", "BellmanFord", "SSSP"}
        if function in {"LCC", "KCore"}:
            function_edges = undirected_edges
            function_directed = False
        else:
            function_edges = (
                deterministic_weighted_edges(n, base_edges)
                if weighted
                else base_edges
            )
            function_directed = directed
        cpu_graph = build_networkx(
            n, function_edges, function_directed, weighted=weighted
        )
        gpu_graph = build_nxcugraph_native(
            n, function_edges, function_directed, weighted=weighted
        )

        if function == "PageRank":
            cpu_call = lambda: nx.pagerank(
                cpu_graph, alpha=0.75, tol=1e-6, max_iter=200
            )
            gpu_call = lambda: nx_cugraph_call(
                nx.pagerank,
                gpu_graph,
                alpha=0.75,
                tol=1e-6,
                max_iter=200,
            )
            interval_count = 1
        elif function == "LCC":
            cpu_call = lambda: nx.clustering(cpu_graph)
            gpu_call = lambda: nx_cugraph_call(nx.clustering, gpu_graph)
            interval_count = 1
        elif function == "WCC":
            component_call = (
                nx.weakly_connected_components
                if function_directed
                else nx.connected_components
            )
            cpu_call = lambda: list(component_call(cpu_graph))
            gpu_call = lambda: list(
                nx_cugraph_call(component_call, gpu_graph)
            )
            interval_count = 1
        elif function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
            selected = sources[:1] if function == "Dijkstra" else sources

            def run_path(graph, backend):
                output = {}
                for source in selected:
                    kwargs = {"backend": backend} if backend else {}
                    if function == "BFS":
                        values = nx.single_source_shortest_path_length(
                            graph, source, **kwargs
                        )
                    elif function == "BellmanFord":
                        values = nx.single_source_bellman_ford_path_length(
                            graph, source, weight="weight", **kwargs
                        )
                    else:
                        values = nx.single_source_dijkstra_path_length(
                            graph, source=source, weight="weight", **kwargs
                        )
                    output[int(source)] = values
                return output

            cpu_call = lambda: run_path(cpu_graph, None)
            gpu_call = lambda: run_path(gpu_graph, "cugraph")
            interval_count = len(selected)
        else:
            cpu_call = lambda: nx.core_number(cpu_graph)
            gpu_call = lambda: nx_cugraph_call(nx.core_number, gpu_graph)
            interval_count = 1

        reference = cpu_call()
        with NxCugraphDeviceTimer() as timer:
            timer.reset()
            started = time.perf_counter()
            observed = timer.call(gpu_call)
            e2e_seconds = time.perf_counter() - started
            device_seconds = timer.validate_against_public_wall(e2e_seconds)
            provenance = timer.validate_provenance(
                expected_backends=NX_CUGRAPH_DEVICE_BOUNDARIES[function],
                expected_interval_count=nx_cugraph_expected_interval_count(
                    function, interval_count
                ),
            )

        if function == "WCC":
            error = (
                0.0
                if canonical_components(reference)
                == canonical_components(observed)
                else math.inf
            )
        elif function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
            error = max(
                numeric_mapping_error(reference[source], observed[source])
                for source in reference
            )
        else:
            error = numeric_mapping_error(reference, observed)
        tolerance = 5e-5 if function == "PageRank" else 1e-10
        status = "pass" if math.isfinite(error) and error <= tolerance else "fail"
        records.append(
            {
                "function": function,
                "status": status,
                "max_abs_error": error,
                "tolerance": tolerance,
                "e2e_seconds": e2e_seconds,
                "device_seconds": device_seconds,
                "device_not_above_e2e": device_seconds <= e2e_seconds,
                "timer_provenance": provenance,
            }
        )

    report = {
        "status": (
            "pass"
            if all(record["status"] == "pass" for record in records)
            else "fail"
        ),
        "edge_path": str(args.edge_path.resolve()),
        "graph_type": args.graph_type,
        "num_nodes": n,
        "cuda_runtime": cuda_runtime,
        "networkx_cache_converted_graphs": bool(
            nx.config.cache_converted_graphs
        ),
        "networkx_fallback_to_nx": bool(nx.config.fallback_to_nx),
        "validation_reference": "NetworkX CPU on the identical normalized graph",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
