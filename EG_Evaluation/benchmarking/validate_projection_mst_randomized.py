#!/usr/bin/env python3
"""Randomized regression for projection-backed EGGPU minimum spanning forests."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from array import array
from pathlib import Path

import easygraph as eg
import networkx as nx
import numpy as np


def write_case(root: Path, case_id: int, rng: random.Random) -> tuple[Path, nx.Graph]:
    num_nodes = 8 + case_id * 3
    split = max(3, num_nodes // 2)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    groups = (range(0, split), range(split, num_nodes))
    for group in groups:
        group = list(group)
        for index in range(1, len(group)):
            source = group[index - 1]
            target = group[index]
            graph.add_edge(source, target)
        for source in group:
            for target in group:
                if source >= target or graph.has_edge(source, target):
                    continue
                if rng.random() < 0.28:
                    graph.add_edge(source, target)

    for edge_index, (source, target) in enumerate(sorted(graph.edges)):
        if case_id % 3 == 0:
            weight = float((edge_index % 4) - 2)
        elif case_id % 3 == 1:
            weight = rng.uniform(-3.0, 7.0) + edge_index * 1.0e-8
        else:
            weight = 1.0 + float((source * 17 + target * 13) % 11) / 10.0
        graph[source][target]["weight"] = weight

    rows: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for source, target, data in graph.edges(data=True):
        weight = float(data["weight"])
        rows[source].append((target, weight))
        rows[target].append((source, weight))
    first_source, first_target, first_data = next(iter(graph.edges(data=True)))
    rows[first_source].append((first_target, float(first_data["weight"])))

    offsets = [0]
    indices: list[int] = []
    weights: list[float] = []
    for row in rows:
        row.sort(key=lambda item: item[0])
        for target, weight in row:
            indices.append(target)
            weights.append(weight)
        offsets.append(len(indices))

    source_dir = root / f"case-{case_id}" / "source"
    source_dir.mkdir(parents=True)
    offsets_path = source_dir / "offsets.i32"
    indices_path = source_dir / "indices.i32"
    weights_path = source_dir / "weights.f64"
    with offsets_path.open("wb") as handle:
        array("i", offsets).tofile(handle)
    with indices_path.open("wb") as handle:
        array("i", indices).tofile(handle)
    with weights_path.open("wb") as handle:
        array("d", weights).tofile(handle)

    manifest = source_dir / "graph.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "eggpu-csr-v1",
                "generation": 1,
                "name": f"random-mst-{case_id}",
                "directed": True,
                "num_nodes": num_nodes,
                "num_edges": len(indices),
                "num_entries": len(indices),
                "node_labels": "zero_based_contiguous",
                "offset_dtype": "int32",
                "index_dtype": "int32",
                "weight_dtype": "float64",
                "weight_key": "weight",
                "offsets_path": offsets_path.name,
                "indices_path": indices_path.name,
                "weights_path": weights_path.name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, graph


def add_projection_weights(projection: Path, graph: nx.Graph) -> None:
    metadata = json.loads(projection.read_text(encoding="utf-8"))
    base = projection.parent
    offsets = np.fromfile(base / metadata["lower_V_path"], dtype=np.int32)
    targets = np.fromfile(base / metadata["lower_E_path"], dtype=np.int32)
    weights = np.empty(targets.size, dtype=np.float64)
    for source in range(graph.number_of_nodes()):
        begin = int(offsets[source])
        end = int(offsets[source + 1])
        for edge in range(begin, end):
            target = int(targets[edge])
            weights[edge] = float(graph[source][target]["weight"])
    weight_path = base / "projection.weights.f64"
    weights.tofile(weight_path)
    metadata.update(
        {
            "weights_path": weight_path.name,
            "weight_dtype": "float64",
            "weight_key": "weight",
        }
    )
    projection.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_projection_mst_randomized.py WORK_DIR")
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    driver = Path(__file__).with_name("prepare_logical_undirected_projection.py")
    rng = random.Random(20260727)
    results = []

    for case_id in range(9):
        source, reference = write_case(root, case_id, rng)
        case_root = source.parent.parent
        output = case_root / "projection"
        work = root / "projection-worker"
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
                f"case-{case_id}",
                "--threads",
                "2",
            ],
            check=True,
            env={
                **os.environ,
                "TMPDIR": str(root),
                "TMP": str(root),
                "TEMP": str(root),
            },
        )
        projection = output / f"case-{case_id}.logical-undirected.json"
        add_projection_weights(projection, reference)
        graph = eg.read_eggpu_csr(
            source,
            undirected_projection_path=projection,
            validate=True,
        )
        actual = eg.minimum_spanning_tree(graph, weight="weight")
        expected = nx.minimum_spanning_tree(
            reference, weight="weight", algorithm="kruskal"
        )
        actual_edges = list(actual.edges)
        actual_weight = sum(
            float(data["weight"]) for _, _, data in actual_edges
        )
        expected_weight = sum(
            float(data["weight"]) for _, _, data in expected.edges(data=True)
        )
        passed = (
            len(actual_edges) == expected.number_of_edges()
            and np.isclose(actual_weight, expected_weight, rtol=1.0e-12, atol=1.0e-12)
        )
        results.append(
            {
                "case": case_id,
                "nodes": reference.number_of_nodes(),
                "edges": reference.number_of_edges(),
                "components": nx.number_connected_components(reference),
                "actual_forest_edges": len(actual_edges),
                "expected_forest_edges": expected.number_of_edges(),
                "actual_total_weight": actual_weight,
                "expected_total_weight": expected_weight,
                "status": "pass" if passed else "fail",
            }
        )

    summary = {
        "status": "pass"
        if all(row["status"] == "pass" for row in results)
        else "fail",
        "cases": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
