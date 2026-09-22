#!/usr/bin/env python3
"""Audit the three timing boundaries required by the paper's main experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


VALIDATION_OK = {
    "pass",
    "reference",
    "external_reference_pass",
    "sampled_pass",
}
CPU_LIBRARIES = {"networkx", "easygraph-cpu", "easygraph-cpp", "igraph"}
GPU_LIBRARIES = {"EGGPU", "nx-cugraph"}
METRICS = {
    "construction": "build_paper_seconds",
    "processing": "kernel_paper_seconds",
    "end_to_end": "e2e_paper_seconds",
}


def finite_positive(value: object) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def audit_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    issues: list[dict] = []
    successful = [
        row
        for row in rows
        if row.get("execution_status") == "ok"
        and row.get("validation_status") in VALIDATION_OK
    ]

    for row in successful:
        key = {
            "dataset": row.get("dataset", ""),
            "function": row.get("function", ""),
            "baseline": row.get("baseline", ""),
        }
        values: dict[str, float] = {}
        for metric, column in METRICS.items():
            value = row.get(column)
            if not finite_positive(value):
                issues.append(
                    {
                        **key,
                        "issue": "missing_or_nonpositive_timing",
                        "metric": metric,
                        "value": value,
                    }
                )
            else:
                values[metric] = float(value)

        try:
            sample_count = int(float(row.get("sample_count", "")))
        except (TypeError, ValueError):
            sample_count = -1
        if sample_count != 5:
            issues.append(
                {
                    **key,
                    "issue": "sample_count_not_five",
                    "metric": "all",
                    "value": row.get("sample_count", ""),
                }
            )

        for metric, prefix in (
            ("construction", "build"),
            ("processing", "kernel"),
            ("end_to_end", "e2e"),
        ):
            if not str(row.get(f"{prefix}_estimator", "")).strip():
                issues.append(
                    {
                        **key,
                        "issue": "missing_estimator_provenance",
                        "metric": metric,
                        "value": "",
                    }
                )
            ratio = row.get(f"{prefix}_max_over_median")
            if finite_positive(ratio) and float(ratio) > 20.0:
                issues.append(
                    {
                        **key,
                        "issue": "extreme_sample_spread",
                        "metric": metric,
                        "value": ratio,
                    }
                )

        if {"processing", "end_to_end"} <= values.keys():
            processing = values["processing"]
            end_to_end = values["end_to_end"]
            baseline = row.get("baseline")
            if baseline in CPU_LIBRARIES and not close(processing, end_to_end):
                issues.append(
                    {
                        **key,
                        "issue": "cpu_processing_must_equal_public_call",
                        "metric": "processing",
                        "value": processing / end_to_end,
                    }
                )
            elif baseline == "GraphScope" and processing > end_to_end * (1.0 + 1e-9):
                issues.append(
                    {
                        **key,
                        "issue": "graphscope_app_exceeds_public_return",
                        "metric": "processing",
                        "value": processing / end_to_end,
                    }
                )
            elif baseline in GPU_LIBRARIES and processing > end_to_end * (1.0 + 1e-9):
                issues.append(
                    {
                        **key,
                        "issue": "gpu_device_interval_exceeds_public_call",
                        "metric": "processing",
                        "value": processing / end_to_end,
                    }
                )
            elif baseline == "Gunrock":
                if processing > end_to_end * (1.0 + 1e-9):
                    issues.append(
                        {
                            **key,
                            "issue": "gunrock_device_interval_exceeds_standalone_e2e",
                            "metric": "processing",
                            "value": processing / end_to_end,
                        }
                    )
                if (
                    "construction" in values
                    and values["construction"] > end_to_end * (1.0 + 1e-9)
                ):
                    issues.append(
                        {
                            **key,
                            "issue": "gunrock_construction_exceeds_standalone_e2e",
                            "metric": "construction",
                            "value": values["construction"] / end_to_end,
                        }
                    )

        if row.get("baseline") == "nx-cugraph":
            estimator = str(row.get("kernel_estimator", "")).lower()
            if "wall" in estimator or "surrogate" in estimator:
                issues.append(
                    {
                        **key,
                        "issue": "nxcugraph_processing_is_wall_surrogate",
                        "metric": "processing",
                        "value": row.get("kernel_estimator", ""),
                    }
                )
            if (
                {"processing", "end_to_end"} <= values.keys()
                and close(values["processing"], values["end_to_end"])
            ):
                issues.append(
                    {
                        **key,
                        "issue": "nxcugraph_processing_duplicates_e2e",
                        "metric": "processing",
                        "value": values["processing"],
                    }
                )

    summary = {
        "status": "pass" if not issues else "fail",
        "ledger_rows": len(rows),
        "successful_validated_cells": len(successful),
        "required_metrics": METRICS,
        "issue_count": len(issues),
    }
    return issues, summary


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_issues(path: Path, issues: list[dict]) -> None:
    fields = ["dataset", "function", "baseline", "issue", "metric", "value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issues)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Audit only the named baseline(s); repeat for multiple baselines.",
    )
    args = parser.parse_args()

    all_rows = read_rows(args.ledger.resolve())
    rows = (
        [
            row
            for row in all_rows
            if row.get("baseline", "") in set(args.baseline)
        ]
        if args.baseline
        else all_rows
    )
    issues, summary = audit_rows(rows)
    summary["input_ledger_rows"] = len(all_rows)
    summary["baseline_filter"] = sorted(set(args.baseline))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_issues(args.output_dir / "timing_triplet_issues.csv", issues)
    (args.output_dir / "timing_triplet_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
