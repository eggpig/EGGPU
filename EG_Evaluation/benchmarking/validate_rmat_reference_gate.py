#!/usr/bin/env python3
"""Validate a small controlled R-MAT CSR artifact against NetworkX."""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

import easygraph as eg
from easygraph.utils import gpu_eggpu_backend


def canonical_partition(components, size):
    labels = np.empty(size, dtype=np.int32)
    for component in components:
        representative = min(component)
        for node in component:
            labels[node] = representative
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-count", type=int, default=4)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(metadata["num_nodes"]) > 1_000_000:
        raise ValueError("CPU reference gate is intentionally restricted to small artifacts")
    offsets = np.memmap(
        manifest_path.parent / metadata["offsets_path"], dtype=np.int32, mode="r"
    )
    indices = np.memmap(
        manifest_path.parent / metadata["indices_path"], dtype=np.int32, mode="r"
    )
    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(range(int(metadata["num_nodes"])))
    for source in range(int(metadata["num_nodes"])):
        graph_nx.add_edges_from(
            (source, int(destination))
            for destination in indices[offsets[source] : offsets[source + 1]]
        )

    graph = eg.read_eggpu_csr(manifest_path, validate=True)
    pagerank_gpu = np.asarray(
        eg.pagerank(graph, alpha=0.75, max_iter=200, tol=1.0e-6, weight=None),
        dtype=np.float64,
    )
    pagerank_cpu_dict = nx.pagerank(
        graph_nx, alpha=0.75, max_iter=200, tol=1.0e-6, weight=None
    )
    pagerank_cpu = np.asarray(
        [pagerank_cpu_dict[node] for node in range(len(graph))], dtype=np.float64
    )
    pagerank_max_abs = float(np.max(np.abs(pagerank_gpu - pagerank_cpu)))

    wcc_result = gpu_eggpu_backend.connected_components(graph, directed=False)
    wcc_gpu_raw = np.asarray(wcc_result.labels_numpy(), dtype=np.int32)
    gpu_components = {}
    for node, label in enumerate(wcc_gpu_raw):
        gpu_components.setdefault(int(label), []).append(node)
    wcc_gpu = canonical_partition(gpu_components.values(), len(graph))
    wcc_cpu = canonical_partition(nx.weakly_connected_components(graph_nx), len(graph))

    sources = [
        int(value)
        for value in metadata["benchmark_sources_zero_based"][: args.source_count]
    ]
    bfs_gpu = np.asarray(eg.multi_source_bfs(graph, sources).to_numpy(), dtype=np.float64)
    bfs_cpu = np.full((len(sources), len(graph)), np.inf, dtype=np.float64)
    for row, source in enumerate(sources):
        for node, distance in nx.single_source_shortest_path_length(
            graph_nx, source
        ).items():
            bfs_cpu[row, node] = distance

    report = {
        "status": "pass",
        "manifest": str(manifest_path),
        "num_nodes": len(graph),
        "num_entries": int(metadata["num_entries"]),
        "checks": {
            "pagerank": {
                "status": "pass" if pagerank_max_abs <= 5.0e-5 else "fail",
                "maximum_absolute_error": pagerank_max_abs,
                "tolerance": 5.0e-5,
            },
            "wcc": {
                "status": "pass" if np.array_equal(wcc_gpu, wcc_cpu) else "fail",
                "component_count": int(np.unique(wcc_gpu).size),
            },
            "bfs": {
                "status": "pass" if np.array_equal(bfs_gpu, bfs_cpu) else "fail",
                "sources": sources,
            },
        },
    }
    failed = [name for name, check in report["checks"].items() if check["status"] != "pass"]
    if failed:
        report["status"] = "fail"
        report["failed_checks"] = failed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
