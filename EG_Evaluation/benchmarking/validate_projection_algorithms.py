#!/usr/bin/env python3
"""Validate projection-backed LCC, k-core, and MST against NetworkX."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from array import array
from pathlib import Path

import networkx as nx
import numpy as np

import easygraph as eg
from easygraph.utils import gpu_eggpu_backend


ARCS = (
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
    (2, 3),
    (3, 3),
    (4, 3),
    (3, 4),
    (0, 1),
)


def write_source(root: Path) -> Path:
    rows = [[] for _ in range(5)]
    for source, target in ARCS:
        rows[source].append(target)
    offsets = [0]
    indices = []
    weights = []
    for source, row in enumerate(rows):
        indices.extend(row)
        weights.extend(
            1.0 + float((source * target) % 5) for target in row
        )
        offsets.append(len(indices))

    offsets_path = root / "tiny.offsets.i32"
    indices_path = root / "tiny.indices.i32"
    weights_path = root / "tiny.weights.f64"
    with offsets_path.open("wb") as handle:
        array("i", offsets).tofile(handle)
    with indices_path.open("wb") as handle:
        array("i", indices).tofile(handle)
    with weights_path.open("wb") as handle:
        array("d", weights).tofile(handle)
    manifest = root / "tiny.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "eggpu-csr-v1",
                "generation": 1,
                "name": "tiny-projection-validation",
                "directed": True,
                "num_nodes": 5,
                "num_edges": len(indices),
                "num_entries": len(indices),
                "node_labels": "zero_based_contiguous",
                "offset_dtype": "int32",
                "index_dtype": "int32",
                "weight_dtype": "float64",
                "weight_key": "weight",
                "weight_semantics": "1 + (src * dst) % num_nodes",
                "offsets_path": offsets_path.name,
                "indices_path": indices_path.name,
                "weights_path": weights_path.name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def reference_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from((source, target) for source, target in ARCS if source != target)
    return graph


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_projection_algorithms.py WORK_DIR")
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(exist_ok=True)
    source = write_source(root)
    output = root / "projection"
    work = root / "work"
    driver = Path(__file__).with_name(
        "prepare_logical_undirected_projection.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(driver),
            str(source),
            "--output-dir",
            str(output),
            "--work-dir",
            str(work),
            "--allowed-root",
            str(root),
            "--name",
            "tiny",
            "--threads",
            "2",
        ],
        check=True,
        env={
            **os.environ,
            "TMPDIR": str(root / "tmp"),
            "TMP": str(root / "tmp"),
            "TEMP": str(root / "tmp"),
        },
    )
    projection = output / "tiny.logical-undirected.json"
    projection_metadata = json.loads(projection.read_text(encoding="utf-8"))
    lower_v = np.fromfile(
        output / projection_metadata["lower_V_path"], dtype=np.int32
    )
    lower_e = np.fromfile(
        output / projection_metadata["lower_E_path"], dtype=np.int32
    )
    lower_w = np.empty(lower_e.size, dtype=np.float64)
    for source_node in range(5):
        begin = int(lower_v[source_node])
        end = int(lower_v[source_node + 1])
        lower_w[begin:end] = 1.0 + (
            (source_node * lower_e[begin:end]) % 5
        )
    lower_w_path = output / "tiny.logical-undirected.weights.f64"
    lower_w.tofile(lower_w_path)
    projection_metadata.update(
        {
            "weights_path": lower_w_path.name,
            "weight_dtype": "float64",
            "weight_key": "weight",
            "weight_semantics": "1 + (src * dst) % num_nodes",
        }
    )
    projection.write_text(
        json.dumps(projection_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    graph = eg.read_eggpu_csr(
        source,
        undirected_projection_path=projection,
        validate=True,
    )

    actual_lcc = np.asarray(
        [float(eg.clustering(graph)[node]) for node in range(5)],
        dtype=np.float64,
    )
    actual_kcore = np.asarray(eg.k_core(graph), dtype=np.int64)
    actual_mst = eg.minimum_spanning_tree(graph, weight="weight")
    expected = reference_graph()
    expected_lcc = np.asarray(
        [float(nx.clustering(expected, node)) for node in range(5)],
        dtype=np.float64,
    )
    expected_kcore_map = nx.core_number(expected)
    expected_kcore = np.asarray(
        [int(expected_kcore_map[node]) for node in range(5)],
        dtype=np.int64,
    )
    for source_node, target_node in expected.edges:
        expected[source_node][target_node]["weight"] = (
            1.0 + float((source_node * target_node) % 5)
        )
    expected_mst = nx.minimum_spanning_tree(
        expected, weight="weight", algorithm="kruskal"
    )
    actual_mst_edges = list(actual_mst.edges)
    actual_mst_weight = sum(
        float(attributes["weight"])
        for _, _, attributes in actual_mst_edges
    )
    expected_mst_weight = sum(
        float(attributes["weight"])
        for _, _, attributes in expected_mst.edges(data=True)
    )

    failures = []
    if not np.allclose(actual_lcc, expected_lcc, rtol=1.0e-7, atol=1.0e-7):
        failures.append("LCC differs from NetworkX undirected projection")
    if not np.array_equal(actual_kcore, expected_kcore):
        failures.append("KCore differs from NetworkX undirected projection")
    if len(actual_mst_edges) != expected_mst.number_of_edges():
        failures.append("MST edge count differs from NetworkX forest")
    if not np.isclose(
        actual_mst_weight, expected_mst_weight, rtol=0.0, atol=0.0
    ):
        failures.append("MST total weight differs from NetworkX forest")

    result = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "projection_info": graph.undirected_projection_info(),
        "lcc": {
            "actual": actual_lcc.tolist(),
            "expected": expected_lcc.tolist(),
            "kernel_seconds": gpu_eggpu_backend.get_last_kernel_time("lcc"),
        },
        "kcore": {
            "actual": actual_kcore.tolist(),
            "expected": expected_kcore.tolist(),
            "kernel_seconds": gpu_eggpu_backend.get_last_kernel_time("kcore"),
        },
        "mst": {
            "actual_edge_count": len(actual_mst_edges),
            "expected_edge_count": expected_mst.number_of_edges(),
            "actual_total_weight": actual_mst_weight,
            "expected_total_weight": expected_mst_weight,
            "kernel_seconds": gpu_eggpu_backend.get_last_kernel_time("mst"),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
