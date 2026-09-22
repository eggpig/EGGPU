#!/usr/bin/env python3
"""Generate paper-facing tables and exact statistics for the final 13 graphs.

The input ledger is authoritative: every dataset/function/system cell has an
explicit execution outcome.  Only status=ok cells with a finite aligned metric
enter performance comparisons.  Timing estimators are carried by the ledger so
the same generator can build both the archived sensitivity view and the uniform
five-run arithmetic-mean view used by the VLDB paper.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import generate_final_13_dataset_story as story
import generate_final_paper_bundle as base


GPU_BASELINES = ("nx-cugraph", "Gunrock")
CPU_BASELINES = ("networkx", "easygraph-cpu", "easygraph-cpp", "igraph")
BASELINES = (*CPU_BASELINES, *GPU_BASELINES, "EGGPU")
HIGH_LEVEL_GPU_BASELINES = ("nx-cugraph",)
EXTERNAL_CLI_BASELINES = ("Gunrock",)
DEVICE_TIMER_BASELINES = ("Gunrock",)
REPRESENTATIVE_FUNCTIONS = (
    "PageRank", "BC", "WCC", "SCC", "KCore", "BFS", "SSSP", "Constraint"
)
STATUS_CODE = {
    "unsupported_api": "--",
    "unsupported_workload_semantics": "WS",
    "skipped": "--",
    "timeout": "TO",
    "load_timeout": "LTO",
    "oom": "OOM",
    "resource_limit": "OOM",
    "representation_limit": "RL",
    "graph_semantics_limit": "GL",
    "semantic_mismatch": "SM",
    "validation_inconclusive": "VI",
    "validation_missing": "VM",
    "failed": "ERR",
    "execution_error": "ERR",
    "missing_experiment": "MISS",
    "unknown": "MISS",
}


def configure_baselines(ledger):
    """Include optional qualified systems without changing the fixed paper order."""

    global GPU_BASELINES, BASELINES, EXTERNAL_CLI_BASELINES, DEVICE_TIMER_BASELINES
    present = set(ledger["baseline"].dropna().astype(str))
    optional_gpu = tuple(name for name in ("SYgraph",) if name in present)
    GPU_BASELINES = ("nx-cugraph", "Gunrock", *optional_gpu)
    EXTERNAL_CLI_BASELINES = ("Gunrock", *optional_gpu)
    DEVICE_TIMER_BASELINES = ("Gunrock", *optional_gpu)
    BASELINES = (*CPU_BASELINES, *GPU_BASELINES, "EGGPU")


def comparable_baselines(metric):
    """Return systems with a measurement boundary comparable for `metric`."""

    if metric == "e2e":
        return (*CPU_BASELINES, *HIGH_LEVEL_GPU_BASELINES)
    # The ledger keeps the historical ``kernel`` column name for artifact
    # compatibility.  In the paper this metric is processing time.  CPU
    # libraries contribute an existing-graph algorithm wall-time surrogate;
    # GPU systems contribute only when an actual device timer is available.
    # Strict nx-cugraph exposes only the enclosing NetworkX backend call, so it
    # is intentionally excluded from processing-time comparisons.
    return (*CPU_BASELINES, *DEVICE_TIMER_BASELINES)


def comparable_gpu_baselines(metric):
    if metric == "e2e":
        return HIGH_LEVEL_GPU_BASELINES
    return DEVICE_TIMER_BASELINES


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def geomean(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.exp(np.log(values).mean())) if len(values) else float("nan")


def latex(value):
    return base.latex_escape(value)


def fmt_seconds(value):
    if not finite(value):
        return "--"
    value = float(value)
    if value < 1e-4:
        return f"{value:.2e}"
    if value < 1e-2:
        return f"{value:.4f}"
    if value < 1:
        return f"{value:.3f}"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def metric_ok(row, metric):
    return str(row.get("execution_status")) == "ok" and finite(
        row.get(f"{metric}_paper_seconds")
    )


def value_with_sd(row, metric):
    if not metric_ok(row, metric):
        status = str(row.get("execution_status", "unknown"))
        kind = str(row.get("failure_kind", ""))
        return STATUS_CODE.get(status, STATUS_CODE.get(kind, "ERR"))
    value = fmt_seconds(row[f"{metric}_paper_seconds"])
    sd = row.get(f"{metric}_std_seconds")
    return f"{value} $\\pm$ {fmt_seconds(sd)}" if finite(sd) else value


def dataset_table(output):
    rows = []
    for name in story.FINAL_13:
        info = story.DATASET_STRUCTURE[name]
        rows.append({"dataset": name, **info, "role": story.DATASET_ROLE[name]})
    data = pd.DataFrame(rows)
    data.to_csv(output / "paper_table_datasets_13.csv", index=False)
    lines = [
        r"% Requires: booktabs, graphicx",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Normalized graph statistics for the final 13-dataset evaluation. Simple edges exclude self-loops and duplicates; CSR entries are the adjacency records scanned by the implementations.}",
        r"\label{tab:datasets-13}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrcl}",
        r"\toprule",
        r"Dataset & Nodes & Simple edges & CSR entries & Avg. degree & Max degree & Density & Loops removed & Dir. & Role \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        lines.append(
            f"{latex(row.dataset)} & {int(row.nodes):,} & {int(row.simple_edges):,} & "
            f"{int(row.csr_entries):,} & {float(row.avg_degree):.2f} & {int(row.max_degree):,} & "
            f"{float(row.density):.2e} & {int(row.self_loops):,} & "
            f"{'T' if row.directed else 'F'} & {latex(row.role)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    (output / "paper_table_datasets_13.tex").write_text("\n".join(lines), encoding="utf-8")
    return data


def pairwise_summary(ledger, metric):
    rows = []
    for dataset in story.FINAL_13:
        for function in base.FUNCTION_ORDER:
            cell = ledger[ledger.dataset.eq(dataset) & ledger.function.eq(function)]
            egg = cell[cell.baseline.eq("EGGPU")]
            if egg.empty or not metric_ok(egg.iloc[0], metric):
                rows.append(
                    {
                        "dataset": dataset,
                        "function": function,
                        "category": base.FUNCTION_CATEGORY[function],
                        "metric": metric,
                        "eggpu_status": egg.iloc[0].execution_status if not egg.empty else "missing_experiment",
                        "eggpu_seconds": np.nan,
                        "competitors": 0,
                        "best_competitor": "",
                        "best_competitor_seconds": np.nan,
                        "fastest_tied_or_only": False,
                        "only_validated": False,
                        "speedup": np.nan,
                    }
                )
                continue
            egg_row = egg.iloc[0]
            egg_value = float(egg_row[f"{metric}_paper_seconds"])
            competitors = cell[cell.baseline.isin(comparable_baselines(metric))].copy()
            competitors = competitors[
                competitors.apply(lambda row: metric_ok(row, metric), axis=1)
            ]
            if competitors.empty:
                best_name, best_value = "", np.nan
                won, only, speedup = True, True, np.nan
            else:
                key = f"{metric}_paper_seconds"
                best_index = pd.to_numeric(competitors[key], errors="coerce").idxmin()
                best = competitors.loc[best_index]
                best_name, best_value = str(best.baseline), float(best[key])
                won = egg_value <= best_value * (1.0 + 1e-9)
                only = False
                speedup = best_value / egg_value
            rows.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "category": base.FUNCTION_CATEGORY[function],
                    "metric": metric,
                    "eggpu_status": "ok",
                    "eggpu_seconds": egg_value,
                    "eggpu_std_seconds": egg_row.get(f"{metric}_std_seconds"),
                    "competitors": len(competitors),
                    "best_competitor": best_name,
                    "best_competitor_seconds": best_value,
                    "fastest_tied_or_only": won,
                    "only_validated": only,
                    "speedup": speedup,
                }
            )
    return pd.DataFrame(rows)


def pairwise_baseline_summary(ledger, metric):
    rows = []
    for baseline in BASELINES[:-1]:
        for function in base.FUNCTION_ORDER:
            incomparable = (
                metric == "e2e" and baseline in EXTERNAL_CLI_BASELINES
            ) or (
                metric == "kernel"
                and baseline not in (*CPU_BASELINES, *DEVICE_TIMER_BASELINES)
            )
            if incomparable:
                rows.append(
                    {
                        "baseline": baseline,
                        "function": function,
                        "category": base.FUNCTION_CATEGORY[function],
                        "metric": metric,
                        "common_pairs": 0,
                        "eggpu_wins": 0,
                        "geomean_speedup": np.nan,
                        "eggpu_geomean_seconds": np.nan,
                        "baseline_geomean_seconds": np.nan,
                    }
                )
                continue
            egg = ledger[
                ledger.baseline.eq("EGGPU") & ledger.function.eq(function)
            ][["dataset", f"{metric}_paper_seconds", "execution_status"]].rename(
                columns={f"{metric}_paper_seconds": "eggpu"}
            )
            other = ledger[
                ledger.baseline.eq(baseline) & ledger.function.eq(function)
            ][["dataset", f"{metric}_paper_seconds", "execution_status"]].rename(
                columns={f"{metric}_paper_seconds": "other", "execution_status": "other_status"}
            )
            paired = egg.merge(other, on="dataset", how="inner")
            paired = paired[
                paired.execution_status.eq("ok")
                & paired.other_status.eq("ok")
                & pd.to_numeric(paired.eggpu, errors="coerce").notna()
                & pd.to_numeric(paired.other, errors="coerce").notna()
            ].copy()
            ratios = paired.other.astype(float) / paired.eggpu.astype(float)
            rows.append(
                {
                    "baseline": baseline,
                    "function": function,
                    "category": base.FUNCTION_CATEGORY[function],
                    "metric": metric,
                    "common_pairs": len(paired),
                    "eggpu_wins": int((paired.eggpu <= paired.other * (1.0 + 1e-9)).sum()),
                    "geomean_speedup": geomean(ratios),
                    "eggpu_geomean_seconds": geomean(paired.eggpu),
                    "baseline_geomean_seconds": geomean(paired.other),
                }
            )
    return pd.DataFrame(rows)


def write_compact_speedup_table(summary, output):
    data = summary[summary.function.isin(REPRESENTATIVE_FUNCTIONS)].copy()
    data.to_csv(output / "paper_table_main_compact_pairwise_speedup.csv", index=False)
    lookup = data.set_index(["baseline", "function"])
    lines = [
        r"% Requires: booktabs, graphicx, xcolor(table)",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{EGGPU end-to-end speedup over each baseline on exactly matched, correctness-validated dataset--function pairs. Parentheses show EGGPU wins/common pairs; unsupported and failed cells are excluded rather than imputed.}",
        r"\label{tab:main-pairwise-speedup}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l*{8}{c}}",
        r"\toprule",
        "Baseline & " + " & ".join(latex(item) for item in REPRESENTATIVE_FUNCTIONS) + r" \\",
        r"\midrule",
    ]
    for baseline in BASELINES[:-1]:
        cells = []
        for function in REPRESENTATIVE_FUNCTIONS:
            row = lookup.loc[(baseline, function)]
            if int(row.common_pairs) == 0 or not finite(row.geomean_speedup):
                cells.append("--")
            else:
                cells.append(
                    f"{float(row.geomean_speedup):.2f}$\\times$ "
                    f"({int(row.eggpu_wins)}/{int(row.common_pairs)})"
                )
        lines.append(f"{latex(baseline)} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    (output / "paper_table_main_compact_pairwise_speedup.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_compact_best_competitor_table(pairwise, output):
    rows = []
    source = pairwise[
        pairwise["metric"].eq("e2e")
        & pairwise["function"].isin(REPRESENTATIVE_FUNCTIONS)
        & pairwise["eggpu_status"].eq("ok")
    ]
    for function in REPRESENTATIVE_FUNCTIONS:
        part = source[source["function"].eq(function)]
        competitive = part[part["competitors"].gt(0)].copy()
        rows.append(
            {
                "function": function,
                "category": base.FUNCTION_CATEGORY[function],
                "eggpu_successful_workloads": len(part),
                "only_validated_workloads": int(part["only_validated"].sum()),
                "common_pairs": len(competitive),
                "eggpu_wins": int(competitive["fastest_tied_or_only"].sum()),
                "eggpu_geomean_seconds": geomean(competitive["eggpu_seconds"]),
                "best_competitor_geomean_seconds": geomean(
                    competitive["best_competitor_seconds"]
                ),
                "geomean_speedup": geomean(competitive["speedup"]),
            }
        )
    data = pd.DataFrame(rows)
    data.to_csv(output / "paper_table_main_compact_best_competitor.csv", index=False)
    lines = [
        r"% Requires: booktabs",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Compact end-to-end comparison against the pairwise fastest correctness-validated competitor. Times are geometric means over exactly matched common workloads.}",
        r"\label{tab:main-compact-best}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Function & EGGPU (s) & Best comp. (s) & Speedup & Wins/common \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        if int(row.common_pairs) == 0:
            eggpu, competitor, speedup = "--", "--", "--"
        else:
            eggpu = fmt_seconds(row.eggpu_geomean_seconds)
            competitor = fmt_seconds(row.best_competitor_geomean_seconds)
            speedup = f"{float(row.geomean_speedup):.2f}$\\times$"
        lines.append(
            f"{latex(row.function)} & {eggpu} & {competitor} & {speedup} & "
            f"{int(row.eggpu_wins)}/{int(row.common_pairs)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output / "paper_table_main_compact_best_competitor.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return data


def write_build_dataset_table(ledger, output):
    """Report comparable graph preparation without conflating it with calls.

    The two scale anchors use their explicitly recorded bulk-CSR load/device
    preparation path.  Gunrock is left unavailable there because its CLI does
    not expose an equivalent separable graph-construction timer.
    """
    rows = []
    for dataset in story.FINAL_13:
        for baseline in BASELINES:
            part = ledger[
                ledger.dataset.eq(dataset) & ledger.baseline.eq(baseline)
            ].copy()
            values = pd.to_numeric(part.get("build_paper_seconds"), errors="coerce")
            values = values[np.isfinite(values) & (values > 0)]
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "function_cells": len(values),
                    "build_geomean_seconds": geomean(values),
                    "build_min_seconds": float(values.min()) if len(values) else np.nan,
                    "build_max_seconds": float(values.max()) if len(values) else np.nan,
                }
            )
    data = pd.DataFrame(rows)
    data.to_csv(output / "paper_table_build_13_by_dataset.csv", index=False)
    lookup = data.set_index(["dataset", "baseline"])
    lines = [
        r"% Requires: booktabs, graphicx",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Graph preparation time on the final 13 datasets. Each cell is the geometric mean in seconds over available function-specific preparation measurements; parentheses give the number of measured function cells. The scale-anchor EGGPU path loads normalized bulk CSR, nx-cugraph and qualified SYgraph cells prepare native device graphs, and CPU libraries construct their native sparse graphs. Gunrock does not expose a separable aligned preparation timer.}",
        r"\label{tab:build-13-by-dataset}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l*{7}{c}}",
        r"\toprule",
        "Dataset & " + " & ".join(latex(item) for item in BASELINES) + r" \\",
        r"\midrule",
    ]
    for dataset in story.FINAL_13:
        cells = []
        for baseline in BASELINES:
            row = lookup.loc[(dataset, baseline)]
            if not finite(row.build_geomean_seconds):
                cells.append("--")
            else:
                cells.append(
                    f"{fmt_seconds(row.build_geomean_seconds)} ({int(row.function_cells)})"
                )
        lines.append(f"{latex(dataset)} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    (output / "paper_table_build_13_by_dataset.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return data


def write_full_tables(ledger, metric, output):
    metric_rows = ledger.copy()
    metric_rows["display"] = metric_rows.apply(
        lambda row: value_with_sd(row, metric), axis=1
    )
    if metric == "e2e":
        metric_rows.loc[
            metric_rows.baseline.isin(EXTERNAL_CLI_BASELINES), "display"
        ] = "CLI"
    elif metric == "kernel":
        metric_rows.loc[
            ~metric_rows.baseline.isin((*CPU_BASELINES, *DEVICE_TIMER_BASELINES, "EGGPU")),
            "display",
        ] = "N/A"
    metric_rows.to_csv(output / f"paper_table_{metric}_13_full.csv", index=False)
    for family in base.CATEGORY_ORDER:
        functions = [
            function for function in base.FUNCTION_ORDER
            if base.FUNCTION_CATEGORY[function] == family
        ]
        # A 13-dataset x 7-system table is taller than a paper page even when
        # resized horizontally.  Emit three continuation floats so every row
        # remains readable and no appendix table is silently clipped.
        dataset_chunks = (story.FINAL_13[:5], story.FINAL_13[5:9], story.FINAL_13[9:])
        lines = [r"% Requires: booktabs, graphicx, xcolor(table)"]
        base_label = f"tab:{metric}-13-{family.lower().replace(' ', '-').replace('&', 'and')}"
        for part_index, datasets in enumerate(dataset_chunks, start=1):
            metric_label = "processing" if metric == "kernel" else metric
            caption = (
                f"{latex(family)} {metric_label.upper()} time on the final 13 datasets "
                f"(part {part_index}/3). Values are seconds $\\pm$ sample SD. "
                "TO/LTO/OOM/RL/GL/WS/SM/VI/VM/ERR denote call timeout, load timeout, "
                "memory limit, representation limit, graph-semantics limit, "
                "workload-semantics mismatch, semantic mismatch, inconclusive "
                "validation, missing validation, and execution error."
            )
            if metric == "e2e":
                caption += (
                    " CLI denotes an external-process latency whose boundary is not "
                    "comparable to a user-visible in-process function call."
                )
            elif metric == "kernel":
                caption += (
                    " EGGPU and Gunrock use device timers; CPU libraries use the "
                    "wall time of the algorithm on an existing graph. N/A denotes "
                    "a backend without a separable processing timer."
                )
            label = base_label if part_index == 1 else f"{base_label}-part-{part_index}"
            lines.extend(
                [
                    r"\begin{table*}[t]",
                    r"\centering",
                    f"\\caption{{{caption}}}",
                    f"\\label{{{label}}}",
                    r"\resizebox{\textwidth}{!}{%",
                    r"\begin{tabular}{ll" + "c" * len(functions) + "}",
                    r"\toprule",
                    "Dataset & System & " + " & ".join(latex(item) for item in functions) + r" \\",
                    r"\midrule",
                ]
            )
            for dataset in datasets:
                for baseline in BASELINES:
                    cells = []
                    for function in functions:
                        hit = metric_rows[
                            metric_rows.dataset.eq(dataset)
                            & metric_rows.baseline.eq(baseline)
                            & metric_rows.function.eq(function)
                        ]
                        cells.append(str(hit.display.iloc[0]) if not hit.empty else "MISS")
                    prefix = "\\rowcolor{blue!7} " if baseline == "EGGPU" else ""
                    lines.append(
                        prefix
                        + f"{latex(dataset) if baseline == BASELINES[0] else ''} & {latex(baseline)} & "
                        + " & ".join(cells)
                        + r" \\"
                    )
                lines.append(r"\addlinespace[1pt]")
            lines.extend(
                [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
            )
        stem = family.lower().replace(" ", "_").replace("&", "and")
        (output / f"paper_table_{metric}_13_{stem}.tex").write_text(
            "\n".join(lines), encoding="utf-8"
        )


def write_category_table(pairwise, output):
    valid = pairwise[pairwise.eggpu_status.eq("ok")]
    rows = []
    for metric in ("e2e", "kernel"):
        part_metric = valid[valid.metric.eq(metric)]
        for family in base.CATEGORY_ORDER:
            part = part_metric[part_metric.category.eq(family)]
            competitive = part[part.competitors.gt(0)]
            rows.append(
                {
                    "metric": metric,
                    "category": family,
                    "eggpu_successful_workloads": len(part),
                    "fastest_tied_or_only": int(part.fastest_tied_or_only.sum()),
                    "only_validated": int(part.only_validated.sum()),
                    "competitive_pairs": len(competitive),
                    "competitive_wins": int(competitive.fastest_tied_or_only.sum()),
                    "speedup_over_best_competitor": geomean(competitive.speedup),
                }
            )
    data = pd.DataFrame(rows)
    data.to_csv(output / "paper_table_category_13_summary.csv", index=False)
    lines = [
        r"% Requires: booktabs",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Final 13-dataset EGGPU coverage and pairwise speedup. Only correctness-validated common pairs contribute to speedup.}",
        r"\label{tab:category-13-summary}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Metric & Family & Fastest/valid & Only & Wins/common & Speedup \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        lines.append(
            f"{row.metric.upper()} & {latex(row.category)} & "
            f"{int(row.fastest_tied_or_only)}/{int(row.eggpu_successful_workloads)} & "
            f"{int(row.only_validated)} & {int(row.competitive_wins)}/{int(row.competitive_pairs)} & "
            f"{float(row.speedup_over_best_competitor):.2f}$\\times$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output / "paper_table_category_13_summary.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return data


def write_numeric_report(ledger, pairwise, pair_baseline, category, output):
    summary = {"datasets": 13, "functions": 16, "attempted_workloads": 208}
    lines = [
        "# EGGPU Final 13-Dataset Numerical Results",
        "",
        "Every number below is derived from the complete audited cell ledger. Error "
        "terms are sample standard deviations from the retained repeated runs.",
        "",
    ]
    for metric in ("e2e", "kernel"):
        data = pairwise[pairwise.metric.eq(metric)]
        valid = data[data.eggpu_status.eq("ok")]
        competitive = valid[valid.competitors.gt(0)]
        key = metric.lower()
        summary[f"{key}_eggpu_successful_workloads"] = len(valid)
        summary[f"{key}_fastest_tied_or_only"] = int(valid.fastest_tied_or_only.sum())
        summary[f"{key}_only_validated"] = int(valid.only_validated.sum())
        summary[f"{key}_competitive_pairs"] = len(competitive)
        summary[f"{key}_competitive_wins"] = int(competitive.fastest_tied_or_only.sum())
        summary[f"{key}_speedup_over_best_competitor"] = geomean(competitive.speedup)
        gpu_rows = []
        cpu_rows = []
        for _, group in ledger.groupby(["dataset", "function"]):
            egg = group[group.baseline.eq("EGGPU")]
            if egg.empty or not metric_ok(egg.iloc[0], metric):
                continue
            egg_value = float(egg.iloc[0][f"{metric}_paper_seconds"])
            for names, sink in (
                (comparable_gpu_baselines(metric), gpu_rows),
                (CPU_BASELINES, cpu_rows),
            ):
                candidates = group[group.baseline.isin(names)]
                candidates = candidates[candidates.apply(lambda row: metric_ok(row, metric), axis=1)]
                if not candidates.empty:
                    sink.append(float(candidates[f"{metric}_paper_seconds"].min()) / egg_value)
        summary[f"{key}_speedup_over_best_gpu"] = geomean(gpu_rows)
        summary[f"{key}_best_gpu_common_pairs"] = len(gpu_rows)
        summary[f"{key}_speedup_over_best_cpu"] = geomean(cpu_rows)
        summary[f"{key}_best_cpu_common_pairs"] = len(cpu_rows)
        metric_label = "PROCESSING" if metric == "kernel" else metric.upper()
        lines.extend(
            [
                f"## {metric_label}",
                "",
                f"- EGGPU successful workloads: **{len(valid)}/208**.",
                f"- Fastest, tied, or only validated: **{int(valid.fastest_tied_or_only.sum())}/{len(valid)}**.",
                f"- Competitive wins: **{int(competitive.fastest_tied_or_only.sum())}/{len(competitive)}**.",
                f"- Geometric-mean speedup over the pairwise best competitor: **{geomean(competitive.speedup):.2f}x**.",
                f"- Speedup over the pairwise best GPU baseline: **{geomean(gpu_rows):.2f}x** on {len(gpu_rows)} common pairs.",
                f"- Speedup over the pairwise best CPU baseline: **{geomean(cpu_rows):.2f}x** on {len(cpu_rows)} common pairs.",
                "",
            ]
        )
    statuses = ledger.execution_status.value_counts().to_dict()
    summary["cell_status_counts"] = statuses
    memory = ledger[
        ledger.baseline.eq("EGGPU") & pd.to_numeric(ledger.gpu_peak_mb_mean, errors="coerce").notna()
    ] if "gpu_peak_mb_mean" in ledger else pd.DataFrame()
    if not memory.empty:
        memory.to_csv(output / "eggpu_memory_13_details.csv", index=False)
        summary["eggpu_memory_cells"] = len(memory)
        summary["eggpu_max_gpu_peak_mb"] = float(memory.gpu_peak_mb_mean.max())
    lines.extend(
        [
            "## Cell outcomes",
            "",
            "```json",
            json.dumps(statuses, indent=2, sort_keys=True),
            "```",
            "",
            "The companion failure report lists every non-OK cell and its exact reason.",
            "",
        ]
    )
    (output / "FINAL_13_NUMERICAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "final_13_numeric_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    category.to_csv(output / "category_13_exact_summary.csv", index=False)
    pair_baseline.to_csv(output / "pairwise_baseline_13_exact_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(args.ledger.resolve(), low_memory=False)
    configure_baselines(ledger)
    dataset_table(output)
    pairwise = pd.concat(
        [pairwise_summary(ledger, "e2e"), pairwise_summary(ledger, "kernel")],
        ignore_index=True,
    )
    pairwise.to_csv(output / "final_13_pairwise_sota_details.csv", index=False)
    pair_baseline = pd.concat(
        [pairwise_baseline_summary(ledger, "e2e"), pairwise_baseline_summary(ledger, "kernel")],
        ignore_index=True,
    )
    write_build_dataset_table(ledger, output)
    write_compact_speedup_table(pair_baseline[pair_baseline.metric.eq("e2e")], output)
    write_compact_best_competitor_table(pairwise, output)
    write_full_tables(ledger, "e2e", output)
    write_full_tables(ledger, "kernel", output)
    category = write_category_table(pairwise, output)
    summary = write_numeric_report(ledger, pairwise, pair_baseline, category, output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
