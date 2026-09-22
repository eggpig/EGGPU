#!/usr/bin/env python3
"""Create one explicit non-success ledger for every paper experiment.

This script does not infer that an unmeasured cell is unsupported.  It keeps
unsupported APIs, timeouts, resource/representation limits, semantic
mismatches, execution errors, and genuinely missing experiments separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import generate_final_paper_bundle as base


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def text(row, *names) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).replace("\n", " ").strip()
    return ""


def add(rows, experiment, dataset="", baseline="", function="", measurement="",
        status="", reason="", source=""):
    rows.append(
        {
            "experiment": experiment,
            "dataset": dataset,
            "baseline": baseline,
            "function": function,
            "measurement": measurement,
            "status": status,
            "reason": reason,
            "source": source,
        }
    )


def main_matrix(rows, ledger_path: Path):
    data = read_csv(ledger_path)
    for _, row in data[~data["execution_status"].eq("ok")].iterrows():
        add(
            rows,
            "final_13_main_matrix",
            row.dataset,
            row.baseline,
            row.function,
            "E2E/kernel",
            row.execution_status,
            text(row, "reason", "failure_kind"),
            str(ledger_path),
        )


def first_use(rows, result_dir: Path):
    summary_path = result_dir / "first_use_vs_steady.csv"
    data = read_csv(summary_path)
    if data.empty:
        add(rows, "first_use_vs_steady", status="missing_experiment",
            reason="first_use_vs_steady.csv is absent or empty", source=str(summary_path))
        return
    assembly_manifest = result_dir / "ASSEMBLY_MANIFEST.json"
    dataset_stats_path = result_dir / "dataset_stats.json"
    if assembly_manifest.is_file():
        manifest = json.loads(assembly_manifest.read_text())
        datasets = [str(name) for name in manifest["datasets"]]
    elif dataset_stats_path.is_file():
        dataset_stats = json.loads(dataset_stats_path.read_text())
        datasets = [str(item["name"]) for item in dataset_stats]
    else:
        datasets = sorted(str(name) for name in data["dataset"].dropna().unique())
    measured = set(zip(data["dataset"], data["function"]))
    for dataset in datasets:
        for function in base.FUNCTION_ORDER:
            if (dataset, function) in measured:
                continue
            if function == "Closeness":
                reason = (
                    "The original first-use run omitted all-node exact Closeness on this "
                    "scale-guarded graph. A separately validated 16-target steady-state "
                    "Closeness result exists, but no matched 16-target first-use sample was run."
                )
            else:
                reason = "No matched first-use and steady-state row was produced."
            add(rows, "first_use_vs_steady", dataset, "EGGPU", function,
                "first-use E2E/kernel", "missing_experiment", reason, str(summary_path))


def ablation(rows, result_dir: Path):
    failed_path = result_dir / "ablation_failed_rows.csv"
    for _, row in read_csv(failed_path).iterrows():
        add(rows, "ablation", baseline="EGGPU", function=text(row, "function"),
            measurement=text(row, "experiment"), status="execution_error",
            reason=f"variant={text(row, 'variant')}; failed_rows={text(row, 'failed_rows')}",
            source=str(failed_path))
    timeout_path = result_dir / "ablation_timeout_rows.csv"
    for _, row in read_csv(timeout_path).iterrows():
        add(rows, "ablation", baseline="EGGPU", function=text(row, "function"),
            measurement=text(row, "experiment"), status="timeout",
            reason=f"variant={text(row, 'variant')}; timeout_rows={text(row, 'timeout_rows')}",
            source=str(timeout_path))


def cumulative(rows, result_dir: Path):
    failures_path = result_dir / "cumulative_workflow_failures.csv"
    failures = read_csv(failures_path)
    for _, row in failures.iterrows():
        add(rows, "cumulative_workflow", text(row, "dataset"), text(row, "baseline"),
            "WCC -> PageRank -> BFS", "fresh-process workflow",
            text(row, "failure_kind", "status") or "execution_error",
            text(row, "error"), str(failures_path))
    metadata_path = result_dir / "metadata.json"
    summary_path = result_dir / "cumulative_workflow_summary.csv"
    if not metadata_path.is_file() or not summary_path.is_file():
        add(rows, "cumulative_workflow", status="missing_experiment",
            reason="workflow metadata or summary is absent", source=str(result_dir))
        return
    metadata = json.loads(metadata_path.read_text())
    summary = read_csv(summary_path)
    repeat = int(metadata["repeat"])
    for dataset in metadata["datasets"]:
        for baseline in metadata["baselines"]:
            for position, function in enumerate(metadata["workflow"], start=1):
                hit = summary[
                    summary["dataset"].eq(dataset)
                    & summary["baseline"].eq(baseline)
                    & summary["call_position"].eq(position)
                    & summary["function"].eq(function)
                ]
                count = int(hit["sample_count"].iloc[0]) if not hit.empty else 0
                if count < repeat:
                    add(rows, "cumulative_workflow", dataset, baseline, function,
                        f"call {position}", "incomplete_samples",
                        f"Only {count}/{repeat} fresh-process samples completed.",
                        str(summary_path))


def generic_status_csv(rows, experiment, path: Path, baseline_default=""):
    data = read_csv(path)
    if data.empty:
        add(rows, experiment, baseline=baseline_default, status="missing_experiment",
            reason=f"Expected result CSV is absent or empty: {path}", source=str(path))
        return
    status_column = "status" if "status" in data else None
    if status_column is None:
        return
    for _, row in data[~data[status_column].eq("ok")].iterrows():
        status = text(row, "failure_kind", "status") or "execution_error"
        reason = text(row, "error", "reason", "skip_reason", "validation_note")
        add(
            rows,
            experiment,
            text(row, "dataset"),
            text(row, "baseline") or baseline_default,
            text(row, "function"),
            text(row, "measurement"),
            status,
            reason,
            str(path),
        )


def scaling(rows, core_dir: Path, followup_dir: Path, sygraph_dir: Path | None):
    sources = [
        ("scale_anchor_eggpu", core_dir / "eggpu_large_matrix" / "scaling_all.csv", "EGGPU"),
        ("scale_anchor_nx_cugraph", core_dir / "nxcugraph_large_matrix" / "nxcugraph_large_matrix.csv", "nx-cugraph"),
        ("scale_anchor_gunrock", core_dir / "gunrock_large_matrix" / "gunrock_large_matrix.csv", "Gunrock"),
        ("scale_anchor_cpu", core_dir / "cpu_large_matrix" / "cpu_large_matrix.csv", ""),
        ("rmat_eggpu", followup_dir / "eggpu_rmat_four_functions" / "scaling_all.csv", "EGGPU"),
        ("rmat_nx_cugraph", followup_dir / "nxcugraph_rmat_four_functions" / "nxcugraph_large_matrix.csv", "nx-cugraph"),
        ("rmat_igraph", followup_dir / "igraph_rmat_four_functions" / "cpu_large_matrix.csv", "igraph"),
    ]
    if sygraph_dir is not None:
        sources.append(
            ("sygraph_native_bfs", sygraph_dir / "sygraph_bfs.csv", "SYgraph")
        )
    for experiment, path, baseline in sources:
        generic_status_csv(rows, experiment, path, baseline)


def write_report(data: pd.DataFrame, output: Path):
    lines = [
        "# EGGPU All-Experiment Non-Success Ledger",
        "",
        "Only non-success outcomes are listed. A row means the result must not be "
        "silently treated as a valid timing. Unsupported APIs, timeouts, resource "
        "limits, semantic mismatches, execution errors, incomplete samples, and "
        "experiments never run remain distinct.",
        "",
        "## Summary",
        "",
        "| Experiment | Non-success rows | Status counts |",
        "| --- | ---: | --- |",
    ]
    for experiment, part in data.groupby("experiment", sort=True):
        counts = ", ".join(
            f"{key}={value}" for key, value in part["status"].value_counts().items()
        )
        lines.append(f"| {experiment} | {len(part)} | {counts} |")
    lines.extend(["", "## Exact outcomes", ""])
    for experiment, part in data.groupby("experiment", sort=True):
        lines.extend([f"### {experiment}", ""])
        if part.empty:
            lines.extend(["None.", ""])
            continue
        for _, row in part.sort_values(["dataset", "baseline", "function", "measurement"]).iterrows():
            identity = " / ".join(
                item for item in (row.dataset, row.baseline, row.function, row.measurement) if item
            )
            lines.append(f"- **{identity or 'experiment-level'}**: `{row.status}`. {row.reason}")
        lines.append("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-ledger", required=True, type=Path)
    parser.add_argument("--first-use", required=True, type=Path)
    parser.add_argument("--ablation", required=True, type=Path)
    parser.add_argument("--cumulative", required=True, type=Path)
    parser.add_argument("--core-result", required=True, type=Path)
    parser.add_argument("--followup-result", required=True, type=Path)
    parser.add_argument("--sygraph-result", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    main_matrix(rows, args.final_ledger.resolve())
    first_use(rows, args.first_use.resolve())
    ablation(rows, args.ablation.resolve())
    cumulative(rows, args.cumulative.resolve())
    scaling(
        rows,
        args.core_result.resolve(),
        args.followup_result.resolve(),
        args.sygraph_result.resolve() if args.sygraph_result else None,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(output / "all_experiment_non_success_ledger.csv", index=False)
    write_report(data, output / "ALL_EXPERIMENT_NON_SUCCESS_REPORT.md")
    summary = {
        "non_success_rows": len(data),
        "experiment_counts": data["experiment"].value_counts().to_dict(),
        "status_counts": data["status"].value_counts().to_dict(),
    }
    (output / "all_experiment_non_success_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
