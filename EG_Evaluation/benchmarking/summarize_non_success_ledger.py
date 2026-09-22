#!/usr/bin/env python3
"""Emit a complete, paper-facing explanation for every non-success cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FIELDS = (
    "dataset",
    "function",
    "category",
    "baseline",
    "support_class",
    "execution_status",
    "failure_kind",
    "validation_status",
    "sample_count",
    "reason",
    "result_source",
)


def clean(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ledger = pd.read_csv(args.ledger.resolve(), low_memory=False)
    required = {"dataset", "function", "baseline", "execution_status"}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"ledger is missing required columns: {missing}")
    duplicate = ledger.duplicated(["dataset", "function", "baseline"], keep=False)
    if duplicate.any():
        examples = ledger.loc[duplicate, ["dataset", "function", "baseline"]].head(10)
        raise ValueError(f"duplicate experiment cells:\n{examples.to_string(index=False)}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    available_fields = [field for field in FIELDS if field in ledger.columns]
    failures = ledger[~ledger["execution_status"].astype(str).eq("ok")].copy()
    failures = failures[available_fields].sort_values(
        ["baseline", "dataset", "function"], kind="stable"
    )
    failures.to_csv(output / "all_experiment_non_success_ledger.csv", index=False)

    status_counts = (
        failures.groupby(["baseline", "execution_status"], dropna=False)
        .size()
        .reset_index(name="cells")
        .sort_values(["baseline", "execution_status"])
    )
    status_counts.to_csv(output / "non_success_counts_by_baseline.csv", index=False)
    function_counts = (
        failures.groupby(
            ["baseline", "function", "support_class", "execution_status"],
            dropna=False,
        )
        .size()
        .reset_index(name="cells")
        .sort_values(["baseline", "function", "execution_status"])
    )
    function_counts.to_csv(
        output / "non_success_counts_by_baseline_function.csv", index=False
    )
    recoverable_gpu = failures[
        failures["baseline"].isin(("nx-cugraph", "Gunrock"))
        & failures["support_class"].isin(("T", "P"))
    ].copy()
    recoverable_gpu.to_csv(
        output / "callable_gpu_baseline_non_success_cells.csv", index=False
    )
    summary = {
        "status": "complete",
        "total_cells": int(len(ledger)),
        "successful_cells": int(ledger["execution_status"].astype(str).eq("ok").sum()),
        "non_success_cells": int(len(failures)),
        "datasets": int(ledger["dataset"].nunique()),
        "functions": int(ledger["function"].nunique()),
        "baselines": int(ledger["baseline"].nunique()),
        "execution_status_counts": {
            clean(key): int(value)
            for key, value in ledger["execution_status"].value_counts(dropna=False).items()
        },
    }
    (output / "all_experiment_non_success_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Complete Non-Success Experiment Ledger",
        "",
        f"- Total matrix cells: **{summary['total_cells']}**.",
        f"- Correctness-qualified successful cells: **{summary['successful_cells']}**.",
        f"- Explicit non-success outcomes: **{summary['non_success_cells']}**.",
        "- A missing numeric value is never interpreted as an unattempted experiment; the status and reason below are authoritative.",
        "",
        "## Counts by baseline and outcome",
        "",
        "| Baseline | Outcome | Cells |",
        "| --- | --- | ---: |",
    ]
    for row in status_counts.itertuples(index=False):
        lines.append(f"| {clean(row.baseline)} | {clean(row.execution_status)} | {int(row.cells)} |")
    lines.extend([
        "",
        "## Cell-level reasons",
        "",
        "| Baseline | Dataset | Function | Support | Outcome | Validation | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for _, row in failures.iterrows():
        reason = clean(row.get("reason")) or clean(row.get("failure_kind")) or "No numeric result."
        reason = reason.replace("|", "\\|")
        lines.append(
            f"| {clean(row.get('baseline'))} | {clean(row.get('dataset'))} | "
            f"{clean(row.get('function'))} | {clean(row.get('support_class'))} | "
            f"{clean(row.get('execution_status'))} | {clean(row.get('validation_status'))} | "
            f"{reason} |"
        )
    (output / "ALL_EXPERIMENT_NON_SUCCESS_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
