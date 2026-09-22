#!/usr/bin/env python3
"""Audit the targeted Gunrock semantic-adapter recovery experiment.

Execution failures and timeouts are recorded outcomes.  This audit treats only
missing cells, malformed rows, mixed baselines, and publishable rows without
the required sample/validation evidence as protocol defects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_DATASETS = (
    "ca-HepTh", "LastFM", "p2p-Gnutella04", "ca-HepPh", "email-Enron",
    "ca-CondMat", "soc-Epinions1", "soc-Slashdot0811", "ER-100k",
    "web-NotreDame", "com-youtube",
)
DEFAULT_FUNCTIONS = (
    "PageRank", "LCC", "BFS", "Dijkstra", "BellmanFord", "SSSP", "KCore", "BC"
)
LARGE_DATASETS = ("com-Orkut", "GAP-twitter")
DIAGNOSTIC_ONLY = {
    "BellmanFord": (
        "Gunrock has no Bellman-Ford application; its maintained SSSP executable "
        "is result-equivalent only for this benchmark's nonnegative weights"
    ),
}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_tokens(value, default):
    tokens = tuple(item.strip() for item in str(value).split(",") if item.strip())
    return tokens or tuple(default)


def parse_detail_path(correctness):
    match = re.search(r"(?:^|[,;]\s*)detail\s*=\s*([^,;]+)", str(correctness))
    if not match:
        return None
    path = Path(match.group(1).strip())
    return path if path.is_file() else None


def compare_detail_files(function, observed_path, reference_path):
    """Compare recovery output with the saved authoritative EGGPU vector."""

    if observed_path is None or reference_path is None:
        return None, "observed or reference detail file is missing"
    try:
        with np.load(observed_path, allow_pickle=False) as observed, np.load(
            reference_path, allow_pickle=False
        ) as reference:
            observed_kind = str(observed["kind"].tolist())
            reference_kind = str(reference["kind"].tolist())
            if observed_kind != reference_kind:
                return False, f"detail kind differs: {observed_kind} vs {reference_kind}"
            lhs = np.asarray(observed["values"])
            rhs = np.asarray(reference["values"])
            if lhs.shape != rhs.shape:
                return False, f"detail shape differs: {lhs.shape} vs {rhs.shape}"
            if "sources" in observed and "sources" in reference:
                if not np.array_equal(observed["sources"], reference["sources"]):
                    return False, "source list differs from the authoritative run"
            elif function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
                return False, "only one path detail artifact records sources"
            if function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
                lhs_finite = np.isfinite(lhs)
                rhs_finite = np.isfinite(rhs)
                if not np.array_equal(lhs_finite, rhs_finite):
                    return False, (
                        "reachable mask differs at "
                        f"{int(np.count_nonzero(lhs_finite != rhs_finite))} entries"
                    )
                if lhs_finite.any() and not np.allclose(
                    lhs[lhs_finite], rhs[rhs_finite], rtol=1.0e-6, atol=1.0e-5
                ):
                    maximum = float(
                        np.max(np.abs(lhs[lhs_finite] - rhs[rhs_finite]))
                    )
                    return False, f"distance values differ; max_abs_error={maximum:.6g}"
                return True, "full source-by-vertex distance output matches EGGPU"
            if function == "KCore":
                if not np.array_equal(lhs, rhs):
                    return False, (
                        "KCore vector differs at "
                        f"{int(np.count_nonzero(lhs != rhs))} vertices"
                    )
                return True, "full KCore vector matches EGGPU"
            rtol, atol = ((1.0e-4, 1.0e-8) if function == "PageRank" else (1.0e-6, 1.0e-8))
            if not np.allclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=True):
                finite = np.isfinite(lhs) & np.isfinite(rhs)
                maximum = float(np.max(np.abs(lhs[finite] - rhs[finite]))) if finite.any() else 0.0
                return False, f"full vector differs; max_abs_error={maximum:.6g}"
            return True, "full output vector matches EGGPU"
    except Exception as exc:
        return None, f"detail comparison failed: {exc}"


def external_reference_map(reference_result):
    if reference_result is None:
        return {}
    path = reference_result / "results_long.csv"
    if not path.is_file():
        return {}
    return {
        (row.get("dataset"), row.get("function")): row
        for row in read_csv(path)
        if row.get("baseline") == "EGGPU"
        and row.get("metric") == "e2e"
        and row.get("status") == "ok"
    }


def standard_cells(result: Path, datasets, functions, expected_repeat, reference_result=None):
    long_path = result / "results_long.csv"
    validation_path = result / "correctness_validation.csv"
    issues = []
    rows = read_csv(long_path) if long_path.is_file() else []
    validations = read_csv(validation_path) if validation_path.is_file() else []
    if not rows:
        return [], [f"missing or empty standard results: {long_path}"]
    baselines = {row.get("baseline", "") for row in rows}
    if baselines != {"Gunrock"}:
        issues.append(f"standard recovery contains unexpected baselines: {sorted(baselines)}")
    validation_map = {
        (row.get("dataset"), row.get("function"), row.get("baseline")):
        row.get("validation_status", "")
        for row in validations
    }
    references = external_reference_map(reference_result)
    index = {
        (row.get("dataset"), row.get("function"), row.get("metric")): row
        for row in rows
        if row.get("baseline") == "Gunrock"
    }
    cells = []
    for dataset in datasets:
        for function in functions:
            e2e = index.get((dataset, function, "e2e"))
            kernel = index.get((dataset, function, "kernel"))
            if e2e is None or kernel is None:
                issues.append(
                    f"missing standard timing row: {dataset}/{function}; "
                    f"e2e={e2e is not None}, kernel={kernel is not None}"
                )
                continue
            status = str(e2e.get("status", "unknown"))
            validation = validation_map.get((dataset, function, "Gunrock"), "")
            validation_note = ""
            if status == "ok" and not (
                function == "PageRank" and validation == "pass"
            ):
                reference_row = references.get((dataset, function))
                observed_detail = parse_detail_path(e2e.get("correctness", ""))
                reference_detail = parse_detail_path(
                    reference_row.get("correctness", "") if reference_row else ""
                )
                external_ok, validation_note = compare_detail_files(
                    function, observed_detail, reference_detail
                )
                if external_ok is True:
                    validation = "external_reference_pass"
                elif external_ok is False:
                    validation = "external_reference_fail"
            sample_count = int(float(e2e.get("sample_count") or 0))
            publishable = (
                status == "ok"
                and finite(e2e.get("mean_seconds", e2e.get("seconds")))
                and finite(kernel.get("mean_seconds", kernel.get("seconds")))
                and validation in {"pass", "reference", "external_reference_pass"}
            )
            publication_note = ""
            if publishable and function in DIAGNOSTIC_ONLY:
                publishable = False
                publication_note = DIAGNOSTIC_ONLY[function]
            if publishable and sample_count != expected_repeat:
                issues.append(
                    f"publishable standard row has {sample_count}/{expected_repeat} "
                    f"timing samples: {dataset}/{function}"
                )
                publishable = False
            cells.append({
                "scope": "cross_library_11",
                "dataset": dataset,
                "function": function,
                "baseline": "Gunrock",
                "execution_status": status,
                "validation_status": validation,
                "timing_samples": sample_count,
                "publishable": publishable,
                "reason": publication_note or validation_note or e2e.get("notes") or e2e.get("skip_reason") or "",
            })
    return cells, issues


def large_cells(result: Path, functions, expected_repeat):
    path = result / "gunrock_large_matrix.csv"
    issues = []
    rows = read_csv(path) if path.is_file() else []
    if not rows:
        return [], [f"missing or empty scale-anchor results: {path}"]
    index = {
        (row.get("dataset"), row.get("function")): row
        for row in rows
        if row.get("baseline") == "Gunrock"
    }
    cells = []
    for dataset in LARGE_DATASETS:
        for function in functions:
            row = index.get((dataset, function))
            if row is None:
                issues.append(f"missing scale-anchor row: {dataset}/{function}")
                continue
            status = str(row.get("status", "unknown"))
            validation = str(row.get("validation", ""))
            sample_count = int(float(row.get("timing_process_samples") or 0))
            publishable = (
                status == "ok"
                and validation == "pass"
                and finite(row.get("e2e_mean_seconds"))
                and finite(row.get("kernel_mean_seconds"))
            )
            publication_note = ""
            if publishable and function in DIAGNOSTIC_ONLY:
                publishable = False
                publication_note = DIAGNOSTIC_ONLY[function]
            if publishable and sample_count != expected_repeat:
                issues.append(
                    f"publishable scale row has {sample_count}/{expected_repeat} "
                    f"timing samples: {dataset}/{function}"
                )
                publishable = False
            cells.append({
                "scope": "scale_anchors",
                "dataset": dataset,
                "function": function,
                "baseline": "Gunrock",
                "execution_status": status,
                "validation_status": validation,
                "timing_samples": sample_count,
                "publishable": publishable,
                "reason": publication_note or row.get("validation_note") or row.get("reason") or "",
            })
    return cells, issues


def write_csv(path: Path, rows):
    fields = list(rows[0]) if rows else [
        "scope", "dataset", "function", "baseline", "execution_status",
        "validation_status", "timing_samples", "publishable", "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-result", type=Path, required=True)
    parser.add_argument("--large-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--functions", default=",".join(DEFAULT_FUNCTIONS))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--reference-result", type=Path)
    args = parser.parse_args()

    datasets = parse_tokens(args.datasets, DEFAULT_DATASETS)
    functions = parse_tokens(args.functions, DEFAULT_FUNCTIONS)
    standard, standard_issues = standard_cells(
        args.standard_result.resolve(),
        datasets,
        functions,
        args.repeat,
        args.reference_result.resolve() if args.reference_result else None,
    )
    large, large_issues = large_cells(
        args.large_result.resolve(), functions, args.repeat
    )
    cells = standard + large
    issues = standard_issues + large_issues
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "gpu_baseline_recovery_cell_status.csv", cells)

    status_counts = Counter(row["execution_status"] for row in cells)
    validation_counts = Counter(row["validation_status"] for row in cells)
    summary = {
        "standard_expected_cells": len(datasets) * len(functions),
        "standard_observed_cells": len(standard),
        "scale_expected_cells": len(LARGE_DATASETS) * len(functions),
        "scale_observed_cells": len(large),
        "publishable_cells": sum(bool(row["publishable"]) for row in cells),
        "execution_status_counts": dict(status_counts),
        "validation_status_counts": dict(validation_counts),
        "protocol_issues": issues,
        "audit_status": "pass" if not issues else "fail",
    }
    (output / "gpu_baseline_recovery_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# GPU Baseline Gap Recovery Audit", "",
        f"- Audit status: **{summary['audit_status']}**.",
        f"- Standard cells: {len(standard)}/{summary['standard_expected_cells']}.",
        f"- Scale-anchor cells: {len(large)}/{summary['scale_expected_cells']}.",
        f"- Correctness-qualified publishable cells: {summary['publishable_cells']}.",
        f"- Execution statuses: `{dict(status_counts)}`.",
        f"- Validation statuses: `{dict(validation_counts)}`.", "",
        "A timeout, representation limit, or failed semantic validation is a recorded "
        "experimental outcome and is not replaced by a fabricated value. Only rows with "
        "complete timing samples and correctness validation are eligible for paper tables.",
    ]
    if issues:
        lines.extend(["", "## Protocol Issues", ""] + [f"- {item}" for item in issues])
    (output / "GPU_BASELINE_RECOVERY_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
