#!/usr/bin/env python3
"""Qualify EasyGraph C++ structural-hole functions under aligned semantics.

The CPU-only C++ implementation accepts native integer identifiers as internal
graph IDs and returns arrays for some functions.  This runner applies only a
frontend adapter: it relabels the prepared graph to contiguous one-based
identifiers, invokes the unmodified C++ implementation, and reconstructs the
public ``dict[original_node, score]`` result.  Algorithm code is not patched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"

import easygraph as eg
import cpp_easygraph

from library_baselines import load_graph


FUNCTIONS = ("EffectiveSize", "Efficiency", "Constraint", "Hierarchy")
FUNCTION_CALLS = {
    "EffectiveSize": ("effective_size", "cpp_effective_size"),
    "Efficiency": ("efficiency", "cpp_efficiency"),
    "Constraint": ("constraint", "cpp_constraint"),
    "Hierarchy": ("hierarchy", "cpp_hierarchy"),
}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_graph(n: int, edges, directed: bool, *, one_based: bool):
    graph = eg.DiGraph() if directed else eg.Graph()
    offset = 1 if one_based else 0
    graph.add_nodes_from(range(offset, n + offset))
    graph.add_edges_from(
        (int(row.src) + offset, int(row.dst) + offset)
        for row in edges.itertuples(index=False)
    )
    return graph


def _as_mapping(value, node_order):
    if isinstance(value, dict):
        return {int(key): float(score) for key, score in value.items()}
    if hasattr(value, "tolist"):
        value = value.tolist()
    values = list(value)
    if len(values) != len(node_order):
        raise ValueError(
            f"result length {len(values)} differs from requested nodes "
            f"{len(node_order)}"
        )
    return {int(node): float(values[index]) for index, node in enumerate(node_order)}


def _compare(reference, candidate, *, rel_tol: float, abs_tol: float):
    if set(reference) != set(candidate):
        return False, {
            "missing_nodes": sorted(set(reference) - set(candidate))[:16],
            "extra_nodes": sorted(set(candidate) - set(reference))[:16],
        }
    mismatches = []
    max_abs = 0.0
    max_rel = 0.0
    for node in sorted(reference):
        expected = float(reference[node])
        actual = float(candidate[node])
        if math.isnan(expected) and math.isnan(actual):
            continue
        abs_error = abs(actual - expected)
        rel_error = abs_error / max(abs(expected), abs(actual), abs_tol)
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
        if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            if len(mismatches) < 16:
                mismatches.append(
                    {
                        "node": int(node),
                        "expected": expected,
                        "actual": actual,
                        "abs_error": abs_error,
                    }
                )
    return not mismatches, {
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "mismatches": mismatches,
    }


def _stats(values):
    values = [float(value) for value in values]
    return {
        "mean_seconds": statistics.mean(values),
        "best_seconds": min(values),
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples_seconds": values,
    }


def qualify_dataset(
    *,
    dataset: str,
    path: Path,
    directed: bool,
    repeat: int,
    rel_tol: float,
    abs_tol: float,
):
    views = load_graph(path)
    n, directed_edges, undirected_edges = views["clean"]
    edges = directed_edges if directed else undirected_edges

    python_graph = _build_graph(n, edges, directed, one_based=False)
    cpp_host_graph = _build_graph(n, edges, directed, one_based=True)
    cpp_graph = cpp_host_graph.cpp()
    cpp_nodes = list(range(1, n + 1))

    rows = []
    for function in FUNCTIONS:
        python_name, cpp_name = FUNCTION_CALLS[function]
        python_fn = getattr(eg, python_name)
        cpp_fn = getattr(cpp_easygraph, cpp_name)

        reference_started = time.perf_counter()
        reference = python_fn(python_graph, nodes=None, weight=None)
        reference_seconds = time.perf_counter() - reference_started
        reference = {int(node): float(value) for node, value in reference.items()}

        elapsed = []
        candidate = None
        return_type = ""
        for _ in range(repeat):
            started = time.perf_counter()
            raw = cpp_fn(cpp_graph, cpp_nodes, None, None)
            one_based_result = _as_mapping(raw, cpp_nodes)
            candidate = {
                int(one_based_node) - 1: score
                for one_based_node, score in one_based_result.items()
            }
            elapsed.append(time.perf_counter() - started)
            return_type = type(raw).__name__

        valid, detail = _compare(
            reference,
            candidate or {},
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        rows.append(
            {
                "dataset": dataset,
                "graph_type": "directed" if directed else "undirected",
                "nodes": n,
                "normalized_edges": len(edges),
                "function": function,
                "support": "P" if valid else "F",
                "validation_status": "pass" if valid else "fail",
                "adapter": (
                    "one-based contiguous relabel before native graph construction; "
                    "dict reconstruction and original-label restoration after call"
                ),
                "algorithm_modified": False,
                "native_return_type": return_type,
                "public_return_type": "dict",
                "reference_seconds": reference_seconds,
                **_stats(elapsed),
                "validation_detail": json.dumps(
                    detail, ensure_ascii=False, sort_keys=True
                ),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "ca-GrQc:undirected:datasets/undirected/ca-GrQc.txt",
            "p2p-Gnutella04:directed:datasets/directed/p2p-Gnutella04.txt",
        ],
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--rel-tol", type=float, default=1.0e-8)
    parser.add_argument("--abs-tol", type=float, default=1.0e-9)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.datasets:
        name, graph_type, raw_path = spec.split(":", 2)
        rows.extend(
            qualify_dataset(
                dataset=name,
                path=Path(raw_path).resolve(),
                directed=graph_type == "directed",
                repeat=max(1, args.repeat),
                rel_tol=args.rel_tol,
                abs_tol=args.abs_tol,
            )
        )

    csv_path = args.output_dir / "easygraph_cpp_structural_qualification.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    extension_path = Path(cpp_easygraph.__file__).resolve()
    summary = {
        "status": "pass"
        if rows and all(row["validation_status"] == "pass" for row in rows)
        else "fail",
        "rows": len(rows),
        "passed": sum(row["validation_status"] == "pass" for row in rows),
        "failed": sum(row["validation_status"] != "pass" for row in rows),
        "easygraph_path": str(Path(eg.__file__).resolve()),
        "cpp_extension_path": str(extension_path),
        "cpp_extension_sha256": _hash_file(extension_path),
        "algorithm_modified": False,
        "adapter_scope": "benchmark frontend and result container only",
        "csv": str(csv_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
