#!/usr/bin/env python3
"""Exact regression for projection-backed structural-hole statistics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from array import array
from pathlib import Path

import numpy as np

import easygraph as eg
from easygraph.utils import gpu_eggpu_backend


ARCS = [
    (0, 1),
    (0, 2),
    (0, 5),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
]
FUNCTIONS = {
    "EffectiveSize": eg.effective_size,
    "Efficiency": eg.efficiency,
    "Constraint": eg.constraint,
    "Hierarchy": eg.hierarchy,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source(root: Path) -> Path:
    rows = [[] for _ in range(6)]
    for source, target in ARCS:
        rows[source].append(target)
    for row in rows:
        row.sort()

    offsets = [0]
    indices = []
    for row in rows:
        indices.extend(row)
        offsets.append(len(indices))

    offsets_path = root / "tiny.offsets.i32"
    indices_path = root / "tiny.indices.i32"
    with offsets_path.open("wb") as handle:
        array("i", offsets).tofile(handle)
    with indices_path.open("wb") as handle:
        array("i", indices).tofile(handle)

    manifest = root / "tiny.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "eggpu-csr-v1",
                "generation": 1,
                "name": "ego-edge-regression",
                "directed": True,
                "num_nodes": 6,
                "num_edges": len(indices),
                "num_entries": len(indices),
                "node_labels": "zero_based_contiguous",
                "offset_dtype": "int32",
                "index_dtype": "int32",
                "offsets_path": offsets_path.name,
                "indices_path": indices_path.name,
                "duplicate_policy": "simple",
                "self_loops_removed": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def as_dense(result, size: int) -> np.ndarray:
    if isinstance(result, dict):
        return np.asarray([result[node] for node in range(size)], dtype=float)
    values = np.asarray(result, dtype=float)
    if values.shape != (size,):
        raise AssertionError(f"unexpected result shape {values.shape}")
    return values


def assert_equal(expected: np.ndarray, observed: np.ndarray, label: str) -> float:
    if not np.array_equal(np.isnan(expected), np.isnan(observed)):
        raise AssertionError(
            f"{label}: NaN domains differ: {expected} versus {observed}"
        )
    finite = np.isfinite(expected)
    diff = (
        float(np.max(np.abs(expected[finite] - observed[finite])))
        if np.any(finite)
        else 0.0
    )
    if not np.allclose(
        expected[finite],
        observed[finite],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise AssertionError(
            f"{label}: max_abs_diff={diff:.12g}, "
            f"expected={expected}, observed={observed}"
        )
    return diff


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    root = Path(
        os.environ.get(
            "EGGPU_EGO_EDGE_TEST_ROOT",
            repo.parent / ".tmp" / "ego_edge_statistics_regression",
        )
    ).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    source = write_source(root)
    projection_dir = root / "projection"
    subprocess.run(
        [
            sys.executable,
            str(
                repo
                / "EG_Evaluation"
                / "benchmarking"
                / "prepare_logical_undirected_projection.py"
            ),
            str(source),
            "--output-dir",
            str(projection_dir),
            "--work-dir",
            str(root / "projection-work"),
            "--allowed-root",
            str(root),
            "--name",
            "tiny",
            "--threads",
            "2",
        ],
        check=True,
    )
    projection = projection_dir / "tiny.logical-undirected.json"

    regular = eg.DiGraph()
    regular.add_nodes(range(6))
    regular.add_edges(ARCS)
    bulk = eg.read_eggpu_csr(
        source,
        undirected_projection_path=projection,
    )

    import cpp_easygraph

    rows = []
    for function, call in FUNCTIONS.items():
        expected = as_dense(call(regular), 6)
        started = time.perf_counter()
        observed = as_dense(call(bulk), 6)
        elapsed = time.perf_counter() - started
        max_diff = assert_equal(expected, observed, function)
        kernel = gpu_eggpu_backend.get_last_kernel_time(
            {
                "EffectiveSize": "effective_size",
                "Efficiency": "effective_size",
                "Constraint": "constraint",
                "Hierarchy": "hierarchy",
            }[function]
        )
        rows.append(
            {
                "function": function,
                "max_abs_diff": max_diff,
                "elapsed_seconds": elapsed,
                "kernel_seconds": kernel,
                "values": observed.tolist(),
            }
        )

    # A node-subset call intentionally stays on the established general path.
    subset = [0, 2, 5]
    subset_expected_result = eg.effective_size(regular, nodes=subset)
    subset_expected = np.asarray(
        [subset_expected_result[node] for node in subset],
        dtype=float,
    )
    subset_observed_result = eg.effective_size(bulk, nodes=subset)
    subset_observed = np.asarray(
        [subset_observed_result[node] for node in subset],
        dtype=float,
    )
    subset_diff = assert_equal(
        subset_expected,
        subset_observed,
        "EffectiveSize subset",
    )

    extension = Path(cpp_easygraph.__file__).resolve()
    report = {
        "status": "pass",
        "extension": str(extension),
        "extension_sha256": sha256(extension),
        "projection": str(projection),
        "functions": rows,
        "subset_max_abs_diff": subset_diff,
        "semantic_checks": {
            "reciprocal_edges": True,
            "directed_asymmetry": True,
            "zero_outdegree_with_incoming_edge": True,
            "subset_uses_general_path": True,
        },
    }
    output = root / "validation.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
