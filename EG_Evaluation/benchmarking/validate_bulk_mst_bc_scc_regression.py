#!/usr/bin/env python3
"""GPU regression gate for the large-CSR MST, BC, and SCC contracts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from array import array
from pathlib import Path

import numpy as np


EDGES = ((0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3))


def write_manifest(root: Path) -> Path:
    rows = [[] for _ in range(6)]
    for source, target in EDGES:
        rows[source].append(target)
        rows[target].append(source)
    offsets = [0]
    indices = []
    weights = []
    for source, row in enumerate(rows):
        for target in sorted(row):
            indices.append(target)
            weights.append(1.0 + ((source * target) % 6))
        offsets.append(len(indices))
    array("i", offsets).tofile((root / "tiny.offsets.i32").open("wb"))
    array("i", indices).tofile((root / "tiny.indices.i32").open("wb"))
    array("d", weights).tofile((root / "tiny.weights.f64").open("wb"))
    manifest = {
        "format": "eggpu-csr-v1",
        "generation": 1,
        "name": "tiny-bulk-regression",
        "directed": False,
        "num_nodes": 6,
        "num_edges": len(EDGES),
        "num_entries": len(indices),
        "node_labels": "zero_based_contiguous",
        "offset_dtype": "int32",
        "index_dtype": "int32",
        "offsets_path": "tiny.offsets.i32",
        "indices_path": "tiny.indices.i32",
        "weights_path": "tiny.weights.f64",
        "weight_dtype": "float64",
        "weight_key": "weight",
        "weight_semantics": "1 + (src * dst) % num_nodes",
    }
    path = root / "tiny.weighted.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def total_weight(graph) -> float:
    return float(sum(float(data["weight"]) for _, _, data in graph.edges))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    os.environ.setdefault("EASYGRAPH_ENABLE_GPU", "TRUE")
    os.environ.setdefault("EASYGRAPH_GPU_STRICT_ERRORS", "TRUE")
    os.environ.setdefault("EASYGRAPH_GPU_ADAPTIVE_HOST", "FALSE")
    os.environ.setdefault("EASYGRAPH_GPU_RESULT_CACHE", "FALSE")

    import easygraph as eg
    from run_eggpu_scaling import function_call

    with tempfile.TemporaryDirectory(prefix="eggpu_bulk_contract_") as temp:
        manifest = write_manifest(Path(temp))
        bulk = eg.read_eggpu_csr(manifest, validate=True)
        regular = eg.Graph()
        regular.add_nodes(range(6))
        for source, target in EDGES:
            regular.add_edge(
                source,
                target,
                weight=1.0 + ((source * target) % 6),
            )

        expected_bc = eg.betweenness_centrality(
            regular,
            sources=[0, 4],
            normalized=False,
            endpoints=False,
        )
        actual_bc = eg.betweenness_centrality(
            bulk,
            sources=[0, 4],
            normalized=False,
            endpoints=False,
        )
        bc_ok = bool(
            np.allclose(
                np.asarray(actual_bc, dtype=np.float64),
                np.asarray(expected_bc, dtype=np.float64),
                rtol=1.0e-6,
                atol=1.0e-8,
            )
        )

        expected_mst = eg.minimum_spanning_tree(regular, weight="weight")
        actual_mst = eg.minimum_spanning_tree(bulk, weight="weight")
        mst_ok = (
            len(actual_mst) == len(bulk)
            and list(actual_mst) == list(range(len(bulk)))
            and actual_mst.number_of_edges() == expected_mst.number_of_edges()
            and np.isclose(total_weight(actual_mst), total_weight(expected_mst))
        )

        components = list(function_call(bulk, "SCC", 2, 2, 2)())
        covered = sum(len(component) for component in components)
        scc_dispatch_ok = covered == len(bulk) and len(components) == 1

        payload = {
            "status": "pass" if bc_ok and mst_ok and scc_dispatch_ok else "fail",
            "unweighted_bulk_bc": {
                "status": "pass" if bc_ok else "fail",
                "expected": list(map(float, expected_bc)),
                "actual": list(map(float, actual_bc)),
            },
            "weighted_bulk_mst": {
                "status": "pass" if mst_ok else "fail",
                "expected_edges": expected_mst.number_of_edges(),
                "actual_edges": actual_mst.number_of_edges(),
                "expected_weight": total_weight(expected_mst),
                "actual_weight": total_weight(actual_mst),
            },
            "undirected_scc_dispatch": {
                "status": "pass" if scc_dispatch_ok else "fail",
                "components": len(components),
                "covered_nodes": covered,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
