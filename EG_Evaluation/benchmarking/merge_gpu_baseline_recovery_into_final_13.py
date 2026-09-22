#!/usr/bin/env python3
"""Merge correctness-qualified Gunrock recovery rows into a derived paper ledger.

The authoritative raw experiments remain immutable.  A recovered cell replaces the
old Gunrock cell only when the recovery audit marks it publishable.  Failed,
unsupported, and semantically mismatched attempts remain explicit ledger outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


GUNROCK_SUPPORT = {
    "PageRank": "P",
    "MST": "P",
    "LCC": "P",
    "WCC": "F",
    "SCC": "F",
    "BFS": "T",
    "Dijkstra": "P",
    "BellmanFord": "F",
    "SSSP": "T",
    "KCore": "T",
    "BC": "F",
    "Closeness": "F",
    "EffectiveSize": "F",
    "Efficiency": "F",
    "Constraint": "F",
    "Hierarchy": "F",
}

TIME_FIELDS = (
    "e2e_paper_seconds",
    "e2e_raw_mean_seconds",
    "e2e_std_seconds",
    "e2e_estimator",
    "kernel_paper_seconds",
    "kernel_raw_mean_seconds",
    "kernel_std_seconds",
    "kernel_estimator",
)
MEMORY_FIELDS = (
    "gpu_peak_mb_mean",
    "gpu_peak_mb_std",
    "memory_sample_count",
    "host_rss_peak_mb_mean",
    "host_rss_peak_mb_std",
    "memory_measurement_window",
)


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_measurements(ledger: pd.DataFrame, index: int) -> None:
    for field in TIME_FIELDS + MEMORY_FIELDS:
        ledger.at[index, field] = pd.NA
    for field in (
        "build_paper_seconds",
        "build_raw_mean_seconds",
        "build_std_seconds",
        "build_estimator",
    ):
        ledger.at[index, field] = pd.NA


def value(row: pd.Series, primary: str, fallback: str | None = None):
    candidate = row.get(primary)
    if finite(candidate):
        return float(candidate)
    if fallback is not None and finite(row.get(fallback)):
        return float(row[fallback])
    return pd.NA


def standard_metric(rows: pd.DataFrame, dataset: str, function: str, metric: str):
    part = rows[
        rows["dataset"].eq(dataset)
        & rows["function"].eq(function)
        & rows["baseline"].eq("Gunrock")
        & rows["metric"].eq(metric)
    ]
    if part.empty:
        return None
    return part.iloc[-1]


def write_standard_publishable(
    ledger: pd.DataFrame,
    index: int,
    rows: pd.DataFrame,
    dataset: str,
    function: str,
    audit_row: pd.Series,
) -> None:
    e2e = standard_metric(rows, dataset, function, "e2e")
    kernel = standard_metric(rows, dataset, function, "kernel")
    if e2e is None or kernel is None:
        raise ValueError(f"audit marked missing standard row publishable: {dataset}/{function}")
    clear_measurements(ledger, index)
    ledger.at[index, "execution_status"] = "ok"
    ledger.at[index, "failure_kind"] = ""
    ledger.at[index, "validation_status"] = audit_row.get("validation_status", "pass")
    ledger.at[index, "reason"] = (
        str(e2e.get("notes") or "")
        + "; correctness-qualified project-local Gunrock adapter; "
        + str(audit_row.get("reason") or "")
    ).strip("; ")
    ledger.at[index, "result_source"] = (
        "Gunrock semantic-adapter recovery: five-run timing, isolated three-run "
        "memory, and independent full-result validation"
    )
    ledger.at[index, "e2e_paper_seconds"] = value(e2e, "mean_seconds", "seconds")
    ledger.at[index, "e2e_raw_mean_seconds"] = value(e2e, "mean_seconds", "seconds")
    ledger.at[index, "e2e_std_seconds"] = value(e2e, "std_seconds")
    ledger.at[index, "e2e_estimator"] = "arithmetic_mean"
    ledger.at[index, "kernel_paper_seconds"] = value(kernel, "mean_seconds", "seconds")
    ledger.at[index, "kernel_raw_mean_seconds"] = value(kernel, "mean_seconds", "seconds")
    ledger.at[index, "kernel_std_seconds"] = value(kernel, "std_seconds")
    ledger.at[index, "kernel_estimator"] = "arithmetic_mean"
    ledger.at[index, "sample_count"] = int(float(e2e.get("sample_count") or 0))

    memory_mapping = {
        "memory_peak_gpu_proc_delta_mb": ("gpu_peak_mb_mean", "gpu_peak_mb_std"),
        "memory_peak_rss_delta_mb": ("host_rss_peak_mb_mean", "host_rss_peak_mb_std"),
    }
    for metric, (mean_field, std_field) in memory_mapping.items():
        memory = standard_metric(rows, dataset, function, metric)
        if memory is None or str(memory.get("status")) != "ok":
            continue
        ledger.at[index, mean_field] = value(memory, "mean_value", "value")
        ledger.at[index, std_field] = value(memory, "std_value")
        if finite(memory.get("sample_count")):
            ledger.at[index, "memory_sample_count"] = int(float(memory["sample_count"]))
    ledger.at[index, "memory_measurement_window"] = "isolated_worker_process_full_lifetime"


def write_large_publishable(
    ledger: pd.DataFrame,
    index: int,
    row: pd.Series,
    audit_row: pd.Series,
) -> None:
    clear_measurements(ledger, index)
    ledger.at[index, "execution_status"] = "ok"
    ledger.at[index, "failure_kind"] = ""
    ledger.at[index, "validation_status"] = audit_row.get("validation_status", "pass")
    ledger.at[index, "reason"] = (
        str(row.get("validation_note") or "")
        + "; correctness-qualified project-local Gunrock adapter"
    ).strip("; ")
    ledger.at[index, "result_source"] = (
        "Gunrock scale-anchor semantic-adapter recovery: five-run timing, "
        "isolated three-run memory, and independent result validation"
    )
    ledger.at[index, "e2e_paper_seconds"] = value(row, "e2e_mean_seconds")
    ledger.at[index, "e2e_raw_mean_seconds"] = value(row, "e2e_mean_seconds")
    ledger.at[index, "e2e_std_seconds"] = value(row, "e2e_stdev_seconds")
    ledger.at[index, "e2e_estimator"] = "arithmetic_mean"
    ledger.at[index, "kernel_paper_seconds"] = value(row, "kernel_mean_seconds")
    ledger.at[index, "kernel_raw_mean_seconds"] = value(row, "kernel_mean_seconds")
    ledger.at[index, "kernel_std_seconds"] = value(row, "kernel_stdev_seconds")
    ledger.at[index, "kernel_estimator"] = "arithmetic_mean"
    ledger.at[index, "gpu_peak_mb_mean"] = value(row, "gpu_process_peak_mb_mean")
    ledger.at[index, "gpu_peak_mb_std"] = value(row, "gpu_process_peak_mb_stdev")
    ledger.at[index, "host_rss_peak_mb_mean"] = value(row, "host_rss_peak_mb_mean")
    ledger.at[index, "host_rss_peak_mb_std"] = value(row, "host_rss_peak_mb_stdev")
    ledger.at[index, "memory_sample_count"] = (
        int(float(row["memory_samples"])) if finite(row.get("memory_samples")) else pd.NA
    )
    ledger.at[index, "memory_measurement_window"] = "isolated_worker_process_full_lifetime"
    ledger.at[index, "sample_count"] = (
        int(float(row["timing_process_samples"]))
        if finite(row.get("timing_process_samples"))
        else 0
    )


def mark_nonpublishable(
    ledger: pd.DataFrame,
    index: int,
    audit_row: pd.Series,
) -> None:
    clear_measurements(ledger, index)
    status = str(audit_row.get("execution_status") or "not_publishable")
    validation = str(audit_row.get("validation_status") or "not_publishable")
    ledger.at[index, "execution_status"] = status
    ledger.at[index, "failure_kind"] = (
        "semantic_mismatch" if "fail" in validation or "mismatch" in validation else status
    )
    ledger.at[index, "validation_status"] = validation
    ledger.at[index, "reason"] = str(audit_row.get("reason") or "recovery row is not publishable")
    ledger.at[index, "result_source"] = "Gunrock semantic-adapter recovery and audit"
    ledger.at[index, "sample_count"] = int(float(audit_row.get("timing_samples") or 0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ledger", type=Path, required=True)
    parser.add_argument("--standard-result", type=Path, required=True)
    parser.add_argument("--large-result", type=Path, required=True)
    parser.add_argument("--recovery-audit", type=Path, required=True)
    parser.add_argument("--output-ledger", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "base_ledger": args.base_ledger.resolve(),
        "standard_results": (args.standard_result.resolve() / "results_long.csv"),
        "large_results": (args.large_result.resolve() / "gunrock_large_matrix.csv"),
        "audit_cells": (
            args.recovery_audit.resolve() / "gpu_baseline_recovery_cell_status.csv"
        ),
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing merge input(s): " + ", ".join(missing))

    ledger = pd.read_csv(inputs["base_ledger"], low_memory=False)
    standard = pd.read_csv(inputs["standard_results"], low_memory=False)
    large = pd.read_csv(inputs["large_results"], low_memory=False)
    audit = pd.read_csv(inputs["audit_cells"], low_memory=False)
    for function, support in GUNROCK_SUPPORT.items():
        mask = ledger["baseline"].eq("Gunrock") & ledger["function"].eq(function)
        ledger.loc[mask, "support_class"] = support

    replaced = []
    rejected = []
    for _, audit_row in audit.iterrows():
        dataset = str(audit_row["dataset"])
        function = str(audit_row["function"])
        mask = (
            ledger["baseline"].eq("Gunrock")
            & ledger["dataset"].eq(dataset)
            & ledger["function"].eq(function)
        )
        matches = ledger.index[mask].tolist()
        if len(matches) != 1:
            raise ValueError(
                f"expected one Gunrock ledger row for {dataset}/{function}, found {len(matches)}"
            )
        index = matches[0]
        if not truthy(audit_row.get("publishable")):
            mark_nonpublishable(ledger, index, audit_row)
            rejected.append(f"{dataset}/{function}")
            continue
        if str(audit_row.get("scope")) == "cross_library_11":
            write_standard_publishable(ledger, index, standard, dataset, function, audit_row)
        elif str(audit_row.get("scope")) == "scale_anchors":
            part = large[
                large["baseline"].eq("Gunrock")
                & large["dataset"].eq(dataset)
                & large["function"].eq(function)
            ]
            if len(part) != 1:
                raise ValueError(
                    f"expected one scale row for {dataset}/{function}, found {len(part)}"
                )
            write_large_publishable(ledger, index, part.iloc[0], audit_row)
        else:
            raise ValueError(f"unknown recovery scope: {audit_row.get('scope')}")
        replaced.append(f"{dataset}/{function}")

    output = args.output_ledger.resolve()
    summary_path = args.output_summary.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output, index=False)
    summary = {
        "status": "complete",
        "policy": (
            "Only audit-qualified Gunrock cells replace the immutable base ledger; "
            "all failed attempts remain explicit nonpublishable outcomes."
        ),
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "output_ledger": str(output),
        "output_ledger_sha256": sha256(output),
        "replaced_publishable_cells": len(replaced),
        "rejected_nonpublishable_cells": len(rejected),
        "replaced_cells": replaced,
        "rejected_cells": rejected,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
