#!/usr/bin/env python3
"""Summarize EGGPU ablation runs with a reproducible system-level protocol.

The ablation runner emits one CSV per dataset/experiment and, after a complete
run, an `ablation_all.csv`.  This script accepts either a finished or partial
ablation directory and writes compact CSV/Markdown summaries for:

1. Same-graph workflow reuse.
2. Return materialization cost.
3. CSR versus COO layout.

Only rows with `status == ok` and finite positive timings are used for ratio
statistics.  This keeps timeout/failed rows visible in separate tables without
turning missing data into artificial speedups.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_stats import _t_critical_95


WORKFLOW_VARIANTS = (
    "full",
    "no_graph_context",
    "no_cpp_graph_cache",
    "no_device_csr_cache",
    "no_adaptive_policy",
)


def geomean(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def median(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def max_value(values) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(arr))


def fmt_float(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.3f}"


def load_ablation_rows(ablation_dir: Path) -> pd.DataFrame:
    all_path = ablation_dir / "ablation_all.csv"
    if all_path.exists():
        return pd.read_csv(all_path)

    frames = []
    for path in sorted(ablation_dir.glob("*.csv")):
        if path.name == "ablation_all.csv":
            continue
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def ok_values(df: pd.DataFrame, experiment: str, metric: str) -> pd.DataFrame:
    sub = df[(df["experiment"] == experiment) & (df["metric"] == metric) & (df["status"] == "ok")].copy()
    if sub.empty:
        return sub
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub[sub["value"].notna() & np.isfinite(sub["value"]) & (sub["value"] > 0)]
    return sub


def summarize_workflow(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = ok_values(df, "workflow", "e2e")
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if "workflow_order_id" not in sub.columns:
        sub["workflow_order_id"] = "canonical"
    sub["workflow_order_id"] = sub["workflow_order_id"].fillna("canonical")

    per_repeat = (
        sub.groupby(["dataset", "workflow_order_id", "variant", "repeat"], dropna=False)["value"]
        .sum()
        .reset_index(name="seconds")
    )
    totals = (
        per_repeat.groupby(["dataset", "workflow_order_id", "variant"], dropna=False)["seconds"]
        .median()
        .reset_index()
    )
    totals = totals[totals["variant"].isin(WORKFLOW_VARIANTS)].copy()
    totals.to_numpy()
    pivot = totals.pivot_table(
        index=["dataset", "workflow_order_id"],
        columns="variant",
        values="seconds",
        aggfunc="median",
    )

    rows = []
    full = pivot.get("full")
    for variant in WORKFLOW_VARIANTS:
        if variant == "full":
            completed = int(full.dropna().shape[0]) if full is not None else 0
            rows.append(
                {
                    "variant": variant,
                    "datasets_compared": completed,
                    "geomean_slowdown": 1.0 if completed else float("nan"),
                    "median_slowdown": 1.0 if completed else float("nan"),
                    "max_slowdown": 1.0 if completed else float("nan"),
                }
            )
            continue
        if full is None or variant not in pivot:
            rows.append(
                {
                    "variant": variant,
                    "datasets_compared": 0,
                    "geomean_slowdown": float("nan"),
                    "median_slowdown": float("nan"),
                    "max_slowdown": float("nan"),
                }
            )
            continue
        pair = pivot[["full", variant]].dropna()
        pair = pair[(pair["full"] > 0) & (pair[variant] > 0)]
        slowdown = pair[variant] / pair["full"]
        rows.append(
            {
                "variant": variant,
                "datasets_compared": int(len(slowdown)),
                "geomean_slowdown": geomean(slowdown),
                "median_slowdown": median(slowdown),
                "max_slowdown": max_value(slowdown),
            }
        )
    comparisons = [
        ("total_reusable_graph_state", "no_graph_context", "full"),
        ("cpp_graph_cache", "no_cpp_graph_cache", "full"),
        ("graph_context_incremental", "no_graph_context", "no_cpp_graph_cache"),
        ("device_csr_cache", "no_device_csr_cache", "full"),
        ("adaptive_policy", "no_adaptive_policy", "full"),
    ]
    nested_rows = []
    for label, numerator, denominator in comparisons:
        if numerator not in pivot or denominator not in pivot:
            continue
        pair = pivot[[numerator, denominator]].dropna()
        pair = pair[(pair[numerator] > 0) & (pair[denominator] > 0)]
        ratios = pair[numerator] / pair[denominator]
        nested_rows.append(
            {
                "comparison": label,
                "numerator_variant": numerator,
                "denominator_variant": denominator,
                "dataset_order_pairs": int(len(ratios)),
                "geomean_slowdown": geomean(ratios),
                "median_slowdown": median(ratios),
                "max_slowdown": max_value(ratios),
            }
        )

    order_rows = []
    full_totals = totals[totals["variant"] == "full"].pivot_table(
        index="dataset",
        columns="workflow_order_id",
        values="seconds",
        aggfunc="median",
    )
    if "canonical" in full_totals:
        for order_id in sorted(col for col in full_totals.columns if col != "canonical"):
            pair = full_totals[["canonical", order_id]].dropna()
            pair = pair[(pair["canonical"] > 0) & (pair[order_id] > 0)]
            ratios = pair[order_id] / pair["canonical"]
            order_rows.append(
                {
                    "workflow_order_id": order_id,
                    "datasets_compared": int(len(ratios)),
                    "geomean_order_over_canonical": geomean(ratios),
                    "median_order_over_canonical": median(ratios),
                    "max_order_over_canonical": max_value(ratios),
                }
            )
    return pd.DataFrame(rows), totals, pd.DataFrame(nested_rows), pd.DataFrame(order_rows)


def summarize_return(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[(df["experiment"] == "return") & (df["status"] == "ok")].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub[sub["value"].notna() & np.isfinite(sub["value"]) & (sub["value"] > 0)]
    uses_paired_schema = (sub["metric"] == "standard_container_return_seconds").any()
    uses_new_schema = (sub["metric"] == "call_return_seconds").any()
    if uses_paired_schema:
        if "return_equivalent" in sub.columns:
            equivalent = sub["return_equivalent"].astype(str).str.lower()
            sub = sub[(equivalent == "true") | sub["return_equivalent"].isna()]
        if "materialization_claim_valid" in sub.columns:
            sub = sub[sub["materialization_claim_valid"].astype(str).str.lower() == "true"]
        call_metric = "call_return_seconds"
        traversed_metric = "standard_container_return_seconds"
    elif uses_new_schema:
        if "materialization_claim_valid" in sub.columns:
            sub = sub[sub["materialization_claim_valid"].astype(str).str.lower() == "true"]
        call_metric = "call_return_seconds"
        traversed_metric = "call_plus_forced_traversal_seconds"
    else:
        call_metric = "lazy_call_e2e"
        traversed_metric = "eager_equivalent_e2e"
    med = (
        sub[sub["metric"].isin([call_metric, traversed_metric])]
        .groupby(["dataset", "function", "metric"], dropna=False)["value"]
        .median()
        .reset_index()
    )
    if med.empty:
        return pd.DataFrame(), pd.DataFrame()
    pivot = med.pivot_table(index=["dataset", "function"], columns="metric", values="value", aggfunc="median")
    if call_metric not in pivot or traversed_metric not in pivot:
        return pd.DataFrame(), pd.DataFrame()
    pairs = pivot.dropna().reset_index()
    pairs = pairs[(pairs[call_metric] > 0) & (pairs[traversed_metric] > 0)].copy()
    pairs["standard_over_deferred"] = pairs[traversed_metric] / pairs[call_metric]
    summary = pd.DataFrame(
        [
            {
                "pairs": int(len(pairs)),
                "protocol": "paired_standard_container" if uses_paired_schema else "post_call_traversal_decomposition",
                "geomean_standard_over_deferred": geomean(pairs["standard_over_deferred"]),
                "median_standard_over_deferred": median(pairs["standard_over_deferred"]),
                "max_standard_over_deferred": max_value(pairs["standard_over_deferred"]),
            }
        ]
    )
    return summary, pairs


def summarize_layout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[(df["experiment"] == "layout") & (df["status"] == "ok")].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub[sub["value"].notna() & np.isfinite(sub["value"]) & (sub["value"] > 0)]
    med = (
        sub.groupby(["dataset", "function", "metric"], dropna=False)["value"]
        .median()
        .reset_index()
    )
    pivot = med.pivot_table(index="dataset", columns=["function", "metric"], values="value", aggfunc="median")

    rows = []
    for dataset in pivot.index:
        row = {"dataset": dataset}
        for function, metric, name in [
            ("COO", "host_storage_mb", "coo_host_mb"),
            ("CSR", "host_storage_mb", "csr_host_mb"),
            ("COO", "degree_seconds", "coo_degree_s"),
            ("CSR", "degree_seconds", "csr_degree_s"),
            ("COO-PageRank", "gpu_kernel_seconds", "coo_pr_kernel_s"),
            ("CSR-PageRank", "gpu_kernel_seconds", "csr_pr_kernel_s"),
        ]:
            try:
                row[name] = float(pivot.loc[dataset, (function, metric)])
            except Exception:
                row[name] = float("nan")
        row["storage_ratio_coo_over_csr"] = (
            row["coo_host_mb"] / row["csr_host_mb"]
            if row["csr_host_mb"] and math.isfinite(row["csr_host_mb"])
            else float("nan")
        )
        row["degree_speedup_csr_vs_coo"] = (
            row["coo_degree_s"] / row["csr_degree_s"]
            if row["csr_degree_s"] and math.isfinite(row["csr_degree_s"])
            else float("nan")
        )
        row["pagerank_speedup_csr_vs_coo"] = (
            row["coo_pr_kernel_s"] / row["csr_pr_kernel_s"]
            if row["csr_pr_kernel_s"] and math.isfinite(row["csr_pr_kernel_s"])
            else float("nan")
        )
        rows.append(row)
    per_dataset = pd.DataFrame(rows)
    summary_rows = []
    for metric, col in [
        ("host_storage_mb", "storage_ratio_coo_over_csr"),
        ("degree_seconds", "degree_speedup_csr_vs_coo"),
        ("pagerank_kernel_seconds", "pagerank_speedup_csr_vs_coo"),
    ]:
        vals = pd.to_numeric(per_dataset.get(col, pd.Series(dtype=float)), errors="coerce")
        vals = vals.dropna()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.empty:
            continue
        summary_rows.append(
            {
                "metric": metric,
                "datasets": int(len(vals)),
                "geomean_coo_over_csr": geomean(vals),
                "median_coo_over_csr": median(vals),
                "max_coo_over_csr": max_value(vals),
            }
        )
    return pd.DataFrame(summary_rows), per_dataset


def summarize_status(df: pd.DataFrame, status: str) -> pd.DataFrame:
    sub = df[df["status"] == status].copy()
    if sub.empty:
        return pd.DataFrame(columns=["experiment", "variant", "function", f"{status}_rows"])
    out = (
        sub.groupby(["experiment", "variant", "function"], dropna=False)
        .size()
        .reset_index(name=f"{status}_rows")
        .sort_values(["experiment", "variant", "function"])
    )
    return out


def summarize_repeat_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve all descriptive statistics needed by the paper estimator policy."""

    sub = df[df["status"] == "ok"].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub[sub["value"].notna() & np.isfinite(sub["value"])]
    group_fields = [
        field
        for field in (
            "experiment",
            "variant",
            "dataset",
            "graph_type",
            "workflow_order_id",
            "function",
            "metric",
            "unit",
        )
        if field in sub.columns
    ]
    rows = []
    for key, group in sub.groupby(group_fields, dropna=False):
        values = group["value"].to_numpy(dtype=float)
        count = int(values.size)
        mean = float(np.mean(values))
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        std = float(np.std(values, ddof=1)) if count >= 2 else float("nan")
        variance = std * std if math.isfinite(std) else float("nan")
        ci_half = (
            _t_critical_95(count - 1) * std / math.sqrt(count)
            if count >= 2 and math.isfinite(std)
            else float("nan")
        )
        relative_std = (
            abs(std / mean) * 100.0
            if math.isfinite(std) and mean != 0.0
            else float("nan")
        )
        row = dict(zip(group_fields, key if isinstance(key, tuple) else (key,)))
        row.update(
            {
                "sample_count": count,
                "mean": mean,
                "minimum": minimum,
                "maximum": maximum,
                "sample_std": std,
                "sample_variance": variance,
                "relative_std_percent": relative_std,
                "ci95_low": mean - ci_half if math.isfinite(ci_half) else float("nan"),
                "ci95_high": mean + ci_half if math.isfinite(ci_half) else float("nan"),
                "ci95_half_width": ci_half,
                "mean_plus_minus_sd": (
                    f"{mean:.6g} +/- {std:.3g}"
                    if math.isfinite(std)
                    else f"{mean:.6g}"
                ),
                "submission_estimator": "best_observed",
                "submission_value": minimum,
                "submission_estimator_definition": (
                    "minimum_of_five_independent_runs"
                ),
                "submission_repeat_contract_satisfied": count == 5,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_fields).reset_index(drop=True)


def write_markdown(
    out_path: Path,
    title: str,
    workflow: pd.DataFrame,
    workflow_nested: pd.DataFrame,
    workflow_orders: pd.DataFrame,
    return_summary: pd.DataFrame,
    layout: pd.DataFrame,
    timeout_rows: pd.DataFrame,
    failed_rows: pd.DataFrame,
    partial: bool,
) -> None:
    lines = [f"# {title}", ""]
    if partial:
        lines.extend(
            [
                "This summary was generated from a partial ablation directory.",
                "Rows that have not been produced yet are absent from ratio statistics.",
                "",
            ]
        )

    lines.extend(["## Workflow cache/policy slowdown", ""])
    if workflow.empty:
        lines.extend(["No workflow ratios available.", ""])
    else:
        lines.extend(
            [
                "| variant | datasets_compared | geomean_slowdown | median_slowdown | max_slowdown |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in workflow.itertuples(index=False):
            lines.append(
                f"| {row.variant} | {int(row.datasets_compared)} | {fmt_float(row.geomean_slowdown)} | "
                f"{fmt_float(row.median_slowdown)} | {fmt_float(row.max_slowdown)} |"
            )
        lines.append("")

    lines.extend(["## Controlled graph-state comparisons", ""])
    if workflow_nested.empty:
        lines.extend(["No nested graph-state ratios available.", ""])
    else:
        lines.extend(
            [
                "| comparison | numerator | denominator | pairs | geomean slowdown | median slowdown |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in workflow_nested.itertuples(index=False):
            lines.append(
                f"| {row.comparison} | {row.numerator_variant} | {row.denominator_variant} | "
                f"{int(row.dataset_order_pairs)} | {fmt_float(row.geomean_slowdown)} | "
                f"{fmt_float(row.median_slowdown)} |"
            )
        lines.append("")

    lines.extend(["## Workflow-order robustness", ""])
    if workflow_orders.empty:
        lines.extend(["No multi-order workflow ratios available.", ""])
    else:
        lines.extend(
            [
                "| order | datasets | geomean order/canonical | median order/canonical | max order/canonical |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in workflow_orders.itertuples(index=False):
            lines.append(
                f"| {row.workflow_order_id} | {int(row.datasets_compared)} | "
                f"{fmt_float(row.geomean_order_over_canonical)} | "
                f"{fmt_float(row.median_order_over_canonical)} | "
                f"{fmt_float(row.max_order_over_canonical)} |"
            )
        lines.append("")

    lines.extend(["## Return materialization cost", ""])
    if return_summary.empty:
        lines.extend(["No return-path ratios available.", ""])
    else:
        lines.extend(
            [
                "| protocol | pairs | geomean standard/deferred | median standard/deferred | max standard/deferred |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        row = return_summary.iloc[0]
        lines.append(
            f"| {row['protocol']} | {int(row['pairs'])} | {fmt_float(row['geomean_standard_over_deferred'])} | "
            f"{fmt_float(row['median_standard_over_deferred'])} | {fmt_float(row['max_standard_over_deferred'])} |"
        )
        lines.append("")

    lines.extend(["## CSR vs COO layout", ""])
    if layout.empty:
        lines.extend(["No layout ratios available.", ""])
    else:
        lines.extend(
            [
                "| metric | datasets | geomean_coo_over_csr | median_coo_over_csr | max_coo_over_csr |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in layout.itertuples(index=False):
            lines.append(
                f"| {row.metric} | {int(row.datasets)} | {fmt_float(row.geomean_coo_over_csr)} | "
                f"{fmt_float(row.median_coo_over_csr)} | {fmt_float(row.max_coo_over_csr)} |"
            )
        lines.append("")

    lines.extend(["## Timeout rows", ""])
    if timeout_rows.empty:
        lines.extend(["No timeout rows.", ""])
    else:
        lines.extend(["| experiment | variant | function | timeout_rows |", "| --- | --- | --- | ---: |"])
        for row in timeout_rows.itertuples(index=False):
            lines.append(f"| {row.experiment} | {row.variant} | {row.function} | {int(row.timeout_rows)} |")
        lines.append("")

    lines.extend(["## Failed rows", ""])
    if failed_rows.empty:
        lines.extend(["No failed rows.", ""])
    else:
        lines.extend(["| experiment | variant | function | failed_rows |", "| --- | --- | --- | ---: |"])
        for row in failed_rows.itertuples(index=False):
            lines.append(f"| {row.experiment} | {row.variant} | {row.function} | {int(row.failed_rows)} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--title", default="EGGPU Ablation System Summary")
    args = ap.parse_args()

    ablation_dir = args.ablation_dir.resolve()
    out_dir = (args.out_dir or ablation_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_ablation_rows(ablation_dir)
    if df.empty:
        raise SystemExit(f"No ablation CSV rows found in {ablation_dir}")

    workflow, workflow_totals, workflow_nested, workflow_orders = summarize_workflow(df)
    return_summary, return_pairs = summarize_return(df)
    layout, layout_per_dataset = summarize_layout(df)
    timeout_rows = summarize_status(df, "timeout")
    failed_rows = summarize_status(df, "failed")
    repeat_statistics = summarize_repeat_statistics(df)

    workflow.to_csv(out_dir / "ablation_workflow_slowdown_summary.csv", index=False)
    workflow_totals.to_csv(out_dir / "ablation_workflow_dataset_totals.csv", index=False)
    workflow_nested.to_csv(out_dir / "ablation_workflow_controlled_comparisons.csv", index=False)
    workflow_orders.to_csv(out_dir / "ablation_workflow_order_robustness.csv", index=False)
    return_summary.to_csv(out_dir / "ablation_return_summary.csv", index=False)
    return_pairs.to_csv(out_dir / "ablation_return_pairs.csv", index=False)
    layout.to_csv(out_dir / "ablation_layout_summary.csv", index=False)
    layout_per_dataset.to_csv(out_dir / "ablation_layout_per_dataset.csv", index=False)
    timeout_rows.to_csv(out_dir / "ablation_timeout_rows.csv", index=False)
    failed_rows.to_csv(out_dir / "ablation_failed_rows.csv", index=False)
    repeat_statistics.to_csv(
        out_dir / "ablation_repeat_statistics.csv", index=False
    )

    partial = not (ablation_dir / "ablation_all.csv").exists()
    write_markdown(
        out_dir / "ABLATION_SYSTEM_SUMMARY.md",
        args.title,
        workflow,
        workflow_nested,
        workflow_orders,
        return_summary,
        layout,
        timeout_rows,
        failed_rows,
        partial,
    )
    print(f"Wrote ablation summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
