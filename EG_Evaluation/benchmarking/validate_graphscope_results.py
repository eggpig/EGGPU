#!/usr/bin/env python3
"""Validate GraphScope's 13-dataset outputs against aligned frozen references."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


FUNCTIONS = [
    "PageRank",
    "MST",
    "LCC",
    "WCC",
    "SCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
    "BC",
    "Closeness",
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
]

SUPPORT = {
    "PageRank": "P",
    "MST": "F",
    "LCC": "T",
    "WCC": "T",
    "SCC": "T",
    "BFS": "T",
    "Dijkstra": "P",
    "BellmanFord": "F",
    "SSSP": "T",
    "KCore": "P",
    "BC": "P",
    "Closeness": "P",
    "EffectiveSize": "F",
    "Efficiency": "F",
    "Constraint": "F",
    "Hierarchy": "F",
}

FIXED_13 = [
    "ca-HepTh",
    "LastFM",
    "p2p-Gnutella04",
    "ca-HepPh",
    "email-Enron",
    "ca-CondMat",
    "soc-Epinions1",
    "soc-Slashdot0811",
    "ER-100k",
    "web-NotreDame",
    "com-youtube",
    "com-Orkut",
    "GAP-twitter",
]

LARGE_DATASETS = {"com-Orkut", "GAP-twitter"}
DETAIL_PATTERN = re.compile(r"(?:^|,\s*)detail=([^,]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-dir", required=True, type=Path)
    parser.add_argument("--legacy-validation", required=True, type=Path)
    parser.add_argument("--large-reference-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
    result = {}
    for row in rows:
        if row.get("baseline") == "GraphScope" and row.get("metric") == "e2e":
            result[(row["dataset"], row["function"])] = row
    return result


def extract_detail(correctness: str) -> Path | None:
    match = DETAIL_PATTERN.search(correctness or "")
    if not match:
        return None
    path = Path(match.group(1))
    return path if path.is_file() else None


def reference_details(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], tuple[Path, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("dataset") not in FIXED_13:
            continue
        grouped.setdefault((row["dataset"], row["function"]), []).append(row)
    priority = ("networkx", "EGGPU", "easygraph-cpp", "igraph", "easygraph-cpu")
    result = {}
    for key, candidates in grouped.items():
        for baseline in priority:
            row = next(
                (
                    item
                    for item in candidates
                    if item.get("baseline") == baseline
                    and item.get("validation_status") in {"pass", "reference"}
                    and extract_detail(item.get("correctness", ""))
                ),
                None,
            )
            if row is not None:
                path = extract_detail(row["correctness"])
                assert path is not None
                result[key] = (path, baseline)
                break
    return result


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def canonical_partition(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels).reshape(-1)
    _, inverse = np.unique(values, return_inverse=True)
    return inverse


def compare_arrays(function: str, got: dict, reference: dict) -> tuple[bool, str]:
    got_values = np.asarray(got["values"])
    ref_values = np.asarray(reference["values"])
    got_sources = np.asarray(got.get("sources", []), dtype=np.int64)
    ref_sources = np.asarray(reference.get("sources", []), dtype=np.int64)

    if function == "Closeness" and got_sources.size and ref_values.ndim == 1:
        ref_values = ref_values[got_sources]
    if function in {"BFS", "Dijkstra", "SSSP"}:
        if not np.array_equal(got_sources, ref_sources[: got_sources.size]):
            return False, "selected source nodes differ from the aligned reference"
    if got_values.shape != ref_values.shape:
        return False, f"shape mismatch: got {got_values.shape}, expected {ref_values.shape}"

    if function in {"WCC", "SCC"}:
        passed = np.array_equal(
            canonical_partition(got_values), canonical_partition(ref_values)
        )
        return passed, "partition equivalent" if passed else "component partition differs"

    if function in {"BFS", "Dijkstra", "SSSP"}:
        finite_got = np.isfinite(got_values)
        finite_ref = np.isfinite(ref_values)
        if not np.array_equal(finite_got, finite_ref):
            return False, "reachable-node mask differs"
        passed = np.allclose(
            got_values[finite_got],
            ref_values[finite_ref],
            rtol=1.0e-8,
            atol=1.0e-8,
        )
        return passed, "all selected-source distances match" if passed else "distance values differ"

    if function == "KCore":
        passed = np.array_equal(got_values, ref_values)
        return passed, "core numbers match" if passed else "core numbers differ"

    tolerance = {
        "PageRank": (1.0e-4, 1.0e-8),
        "LCC": (1.0e-6, 1.0e-8),
        "BC": (1.0e-5, 1.0e-7),
        "Closeness": (1.0e-6, 1.0e-8),
    }.get(function, (1.0e-7, 1.0e-9))
    passed = np.allclose(
        got_values, ref_values, rtol=tolerance[0], atol=tolerance[1], equal_nan=True
    )
    if passed:
        max_error = float(np.max(np.abs(got_values - ref_values))) if got_values.size else 0.0
        return True, f"full aligned result matches; max_abs_error={max_error:.6g}"
    max_error = float(np.nanmax(np.abs(got_values - ref_values)))
    return False, f"numeric result differs; max_abs_error={max_error:.6g}"


def first_worker_payload(timing_dir: Path, dataset: str, function: str) -> dict | None:
    path = (
        timing_dir
        / "samples"
        / "timing"
        / dataset
        / function
        / "r1"
        / "worker_result.json"
    )
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def compare_large_summary(
    dataset: str, function: str, payload: dict, reference_path: Path
) -> tuple[str, str]:
    reference = json.loads(reference_path.read_text())
    if reference.get("status") != "ok":
        return "inconclusive", "frozen EGGPU reference is not successful"
    if (reference.get("result_validation") or {}).get("status") != "pass":
        return "inconclusive", "frozen EGGPU reference lacks a passing validation record"
    got = payload.get("result_summary") or {}
    expected = reference.get("result") or {}

    if list(got.get("shape") or []) != list(expected.get("shape") or []):
        return "fail", f"shape mismatch: got {got.get('shape')}, expected {expected.get('shape')}"

    if function in {"WCC", "SCC"}:
        got_digest = got.get("partition_sha256")
        expected_digest = expected.get("partition_sha256")
        if got_digest and expected_digest and got_digest == expected_digest:
            return "pass", "canonical component partition digest matches"
        return "fail", "canonical component partition digest differs"

    got_sample = np.asarray(got.get("validation_sample") or [], dtype=np.float64)
    expected_sample = np.asarray(expected.get("validation_sample") or [], dtype=np.float64)
    if got_sample.size and expected_sample.size:
        rtol, atol = {
            "PageRank": (1.0e-4, 1.0e-8),
            "LCC": (1.0e-6, 1.0e-8),
            "BC": (1.0e-5, 1.0e-7),
            "Closeness": (1.0e-6, 1.0e-8),
        }.get(function, (1.0e-7, 1.0e-9))
        if got_sample.shape != expected_sample.shape:
            return "fail", "deterministic validation-sample shape differs"
        if not np.allclose(got_sample, expected_sample, rtol=rtol, atol=atol):
            error = float(np.max(np.abs(got_sample - expected_sample)))
            return "fail", f"deterministic validation sample differs; max_abs_error={error:.6g}"

    checks = []
    for got_key, expected_key in (
        ("finite_count", "finite_count"),
        ("max", "maximum"),
        ("sum", "sum"),
    ):
        left = got.get(got_key)
        right = expected.get(expected_key)
        if left is None or right is None:
            continue
        if finite_number(left) and finite_number(right):
            if not math.isclose(float(left), float(right), rel_tol=1.0e-6, abs_tol=1.0e-6):
                return "fail", f"summary field {got_key} differs: {left} vs {right}"
            checks.append(got_key)
    if got_sample.size and expected_sample.size:
        checks.append("validation_sample")
    if checks:
        return "pass", "large-result evidence matches: " + ", ".join(checks)
    return "inconclusive", "no comparable per-dataset large-result evidence"


def validate_pair(
    timing_dir: Path,
    references: dict[tuple[str, str], tuple[Path, str]],
    large_root: Path,
    dataset: str,
    function: str,
) -> tuple[str, str, str]:
    payload = first_worker_payload(timing_dir, dataset, function)
    if payload is None or payload.get("status") != "ok":
        return "inconclusive", "worker result payload is unavailable", ""

    if dataset in LARGE_DATASETS:
        reference_path = large_root / f"{dataset}_{function}_timing.json"
        if not reference_path.is_file():
            return (
                "inconclusive",
                "no frozen per-dataset large-graph reference; qualification evidence only",
                "",
            )
        status, details = compare_large_summary(
            dataset, function, payload, reference_path
        )
        return status, details, str(reference_path)

    got_path = timing_dir / "details" / dataset / f"GraphScope_{function}.npz"
    reference_info = references.get((dataset, function))
    if not got_path.is_file():
        return "inconclusive", "GraphScope full-result detail is unavailable", ""
    if reference_info is None:
        return "inconclusive", "aligned full-result reference is unavailable", ""
    reference_path, reference_baseline = reference_info
    try:
        passed, details = compare_arrays(
            function, load_npz(got_path), load_npz(reference_path)
        )
    except Exception as error:
        return "inconclusive", f"validation exception: {type(error).__name__}: {error}", str(reference_path)
    return ("pass" if passed else "fail"), details, f"{reference_baseline}:{reference_path}"


def main() -> None:
    args = parse_args()
    timing_rows = read_csv(args.timing_dir / "results_long.csv")
    timing = aggregate_index(timing_rows)
    references = reference_details(read_csv(args.legacy_validation))
    rows = []

    for dataset in FIXED_13:
        for function in FUNCTIONS:
            support = SUPPORT[function]
            timing_row = timing.get((dataset, function), {})
            timing_status = timing_row.get("status", "missing")
            if support == "F":
                validation_status = "unsupported"
                details = "no aligned callable GraphScope implementation"
                reference = ""
            elif timing_status != "ok":
                validation_status = "not_run"
                details = timing_row.get("notes") or timing_row.get("skip_reason") or timing_status
                reference = ""
            else:
                validation_status, details, reference = validate_pair(
                    args.timing_dir,
                    references,
                    args.large_reference_dir,
                    dataset,
                    function,
                )
            rows.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "baseline": "GraphScope",
                    "support_label": support,
                    "timing_status": timing_status,
                    "validation_status": validation_status,
                    "details": details,
                    "reference": reference,
                    "qualification_scope": (
                        "directed+undirected weighted/disconnected unit graphs; "
                        "this row adds per-dataset result evidence when available"
                    ),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "graphscope_correctness_validation.csv", rows)
    counts = {}
    for row in rows:
        key = row["validation_status"]
        counts[key] = counts.get(key, 0) + 1
    report = [
        "# GraphScope 13-Dataset Correctness Audit",
        "",
        "GraphScope timings are admissible only when the native call completes and "
        "the aligned result passes this independent validation. Static support "
        "qualification alone is not used to turn an unvalidated timing into a paper result.",
        "",
        f"- Cells audited: {len(rows)}",
        "- Validation counts: "
        + ", ".join(f"`{key}`={value}" for key, value in sorted(counts.items())),
        "",
        "## Non-passing runnable cells",
        "",
        "| Dataset | Function | Timing | Validation | Reason |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        if row["support_label"] != "F" and row["validation_status"] != "pass":
            report.append(
                f"| {row['dataset']} | {row['function']} | {row['timing_status']} | "
                f"{row['validation_status']} | {row['details'].replace('|', '/')} |"
            )
    (args.output / "GRAPHSCOPE_13_CORRECTNESS_AUDIT.md").write_text(
        "\n".join(report) + "\n"
    )
    (args.output / "graphscope_correctness_summary.json").write_text(
        json.dumps(
            {
                "cells": len(rows),
                "validation_counts": counts,
                "support": SUPPORT,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(args.output / "graphscope_correctness_validation.csv")


if __name__ == "__main__":
    main()
