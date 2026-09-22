#!/usr/bin/env python3
"""Generate paper-facing tables and exact statistics for the final 13 graphs.

The input ledger is authoritative: every dataset/function/system cell has an
explicit execution outcome. Only status=ok cells with a finite aligned metric
enter performance comparisons. Timing estimators are carried by the ledger and
paper-facing values are read from the corresponding estimator columns rather
than recomputed implicitly by this generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import generate_final_13_dataset_story as story
import generate_final_paper_bundle as base


GPU_BASELINES = ("nx-cugraph", "Gunrock")
CPU_BASELINES = (
    "networkx",
    "easygraph-cpu",
    "easygraph-cpp",
    "igraph",
    "GraphScope",
)
BASELINES = (*CPU_BASELINES, *GPU_BASELINES, "EGGPU")
HIGH_LEVEL_GPU_BASELINES = ("nx-cugraph",)
EXTERNAL_CLI_BASELINES = ("Gunrock",)
DEVICE_TIMER_BASELINES = ("nx-cugraph", "Gunrock")
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
RELATIVE_TIE_TOLERANCE = 5.0e-4


def classify_timing_outcome(eggpu_seconds, competitor_seconds):
    """Classify one common pair with a 0.05% relative tie tolerance."""

    left = float(eggpu_seconds)
    right = float(competitor_seconds)
    tolerance = RELATIVE_TIE_TOLERANCE * max(abs(left), abs(right), 1.0e-15)
    delta = left - right
    if delta < -tolerance:
        return "strict_win"
    if abs(delta) <= tolerance:
        return "tie"
    return "loss"


def configure_baselines(ledger):
    """Include optional qualified systems without changing the fixed paper order."""

    global GPU_BASELINES, BASELINES, EXTERNAL_CLI_BASELINES, DEVICE_TIMER_BASELINES
    GPU_BASELINES = ("nx-cugraph", "Gunrock")
    EXTERNAL_CLI_BASELINES = ("Gunrock",)
    DEVICE_TIMER_BASELINES = ("nx-cugraph", "Gunrock")
    BASELINES = (*CPU_BASELINES, *GPU_BASELINES, "EGGPU")


def comparable_baselines(metric):
    """Return systems with a measurement boundary comparable for `metric`."""

    if metric == "e2e":
        return (*CPU_BASELINES, *HIGH_LEVEL_GPU_BASELINES)
    # The ledger keeps the historical ``kernel`` column name for artifact
    # compatibility. Device-performance claims compare only systems with an
    # actual GPU execution interval. CPU and GraphScope wall intervals remain
    # visible in the descriptive processing panel but do not enter this SOTA
    # device comparison.
    return DEVICE_TIMER_BASELINES


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
                        "common_strict_win": False,
                        "common_tie": False,
                        "common_loss": False,
                        "pair_outcome": "eggpu_unavailable",
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
                outcome = "sole_validated"
            else:
                key = f"{metric}_paper_seconds"
                best_index = pd.to_numeric(competitors[key], errors="coerce").idxmin()
                best = competitors.loc[best_index]
                best_name, best_value = str(best.baseline), float(best[key])
                outcome = classify_timing_outcome(egg_value, best_value)
                won = outcome in {"strict_win", "tie"}
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
                    "common_strict_win": outcome == "strict_win",
                    "common_tie": outcome == "tie",
                    "common_loss": outcome == "loss",
                    "pair_outcome": outcome,
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
                        "eggpu_ties": 0,
                        "eggpu_losses": 0,
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
            outcomes = [
                classify_timing_outcome(eggpu, other)
                for eggpu, other in zip(paired.eggpu, paired.other)
            ]
            rows.append(
                {
                    "baseline": baseline,
                    "function": function,
                    "category": base.FUNCTION_CATEGORY[function],
                    "metric": metric,
                    "common_pairs": len(paired),
                    "eggpu_wins": outcomes.count("strict_win"),
                    "eggpu_ties": outcomes.count("tie"),
                    "eggpu_losses": outcomes.count("loss"),
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
        r"\caption{EGGPU public-return speedup over each baseline on exactly matched, correctness-validated dataset--function pairs. Parentheses show strict EGGPU wins/common pairs; a relative difference within 0.05\% is a tie. Unsupported and failed cells are excluded rather than imputed.}",
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
                "eggpu_wins": int(competitive["common_strict_win"].sum()),
                "eggpu_ties": int(competitive["common_tie"].sum()),
                "eggpu_losses": int(competitive["common_loss"].sum()),
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
        r"\caption{Compact public-return comparison against the pairwise fastest correctness-validated competitor. Times are geometric means over exactly matched common workloads; a relative difference within 0.05\% is a tie.}",
        r"\label{tab:main-compact-best}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Function & EGGPU (s) & Best comp. (s) & Speedup & W/T/common \\",
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
            f"{int(row.eggpu_wins)}/{int(row.eggpu_ties)}/{int(row.common_pairs)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output / "paper_table_main_compact_best_competitor.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return data


def write_build_dataset_table(ledger, output):
    """Report comparable graph preparation without conflating it with calls.

    The two scale anchors use their explicitly recorded bulk-CSR load/device
    preparation path.  Gunrock reports the input-loading and native graph
    construction interval measured inside its standalone execution boundary.
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
        r"\caption{Graph construction time on the final 13 datasets. Each cell is the geometric mean in seconds over available function-specific construction measurements; parentheses give the number of measured function cells. EGGPU loads the normalized bulk CSR representation on the two scale anchors; GraphScope and the CPU libraries construct their native host graphs, nx-cugraph constructs its converted GPU representation, and Gunrock reports input loading plus native graph construction within its standalone execution boundary.}",
        r"\label{tab:build-13-by-dataset}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l*{8}{c}}",
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
    if metric == "kernel":
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
            metric_label = (
                "processing" if metric == "kernel" else "public-return"
            )
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
                    " Gunrock values span input loading through complete host-result "
                    "availability in its standalone executable; they are shown but "
                    "excluded from library-level E2E SOTA comparisons."
                )
            elif metric == "kernel":
                caption += (
                    " EGGPU, strict nx-cugraph, and Gunrock use device timers; "
                    "CPU libraries use the wall time of the algorithm on an "
                    "existing graph. N/A denotes a backend without a separable "
                    "processing timer."
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


def write_category_table(ledger, pairwise, output):
    valid = pairwise[pairwise.eggpu_status.eq("ok")]
    rows = []
    for metric in ("e2e", "kernel"):
        part_metric = valid[valid.metric.eq(metric)]
        for family in (*base.CATEGORY_ORDER, "Overall"):
            part = (
                part_metric
                if family == "Overall"
                else part_metric[part_metric.category.eq(family)]
            )
            competitive = part[part.competitors.gt(0)]
            graphscope = ledger[
                ledger.baseline.eq("GraphScope")
                & (
                    True
                    if family == "Overall"
                    else ledger.function.map(base.FUNCTION_CATEGORY).eq(family)
                )
            ]
            graphscope_valid = int(
                graphscope.apply(lambda row: metric_ok(row, metric), axis=1).sum()
            )
            aggregate_speedup = geomean(competitive.speedup)
            if metric == "kernel" and family == "Overall":
                # Do not collapse heterogeneous device timers and CPU
                # existing-graph wall-time surrogates into one headline.
                aggregate_speedup = np.nan
            rows.append(
                {
                    "metric": metric,
                    "category": family,
                    "common_strict_wins": int(
                        competitive.common_strict_win.sum()
                    ),
                    "common_ties": int(competitive.common_tie.sum()),
                    "common_losses": int(competitive.common_loss.sum()),
                    "common_pairs": len(competitive),
                    "sole_validated": int(part.only_validated.sum()),
                    "graphscope_validated": graphscope_valid,
                    "speedup_over_best_competitor": aggregate_speedup,
                }
            )
    data = pd.DataFrame(rows)
    data.to_csv(output / "paper_table_category_13_summary.csv", index=False)
    payload = {
        "relative_tie_tolerance": RELATIVE_TIE_TOLERANCE,
        "tie_definition": (
            "absolute timing difference <= 0.05% of the larger pair timing"
        ),
        "speedup_definition": (
            "geometric mean of pairwise-fastest competitor time / EGGPU time "
            "over correctness-valid common pairs"
        ),
        "metrics": {},
    }
    def json_records(frame):
        records = frame.to_dict("records")
        for record in records:
            for key, value in tuple(record.items()):
                if isinstance(value, float) and not math.isfinite(value):
                    record[key] = None
        return records

    for metric in ("e2e", "kernel"):
        metric_rows = data[data.metric.eq(metric)]
        payload["metrics"][
            "public_return" if metric == "e2e" else "processing"
        ] = {
            "families": json_records(
                metric_rows[~metric_rows.category.eq("Overall")]
            ),
            "total": json_records(
                metric_rows[metric_rows.category.eq("Overall")]
            )[0],
        }
    (output / "paper_table_category_13_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        r"% Requires: booktabs",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Final 13-dataset outcomes by function family. W/T reports strict EGGPU wins and ties against the pairwise fastest competitor; a relative difference within 0.05\% is a tie. Only correctness-valid common pairs contribute to speedup. Sole valid reports workloads with no validated competitor and is never counted as a win. GraphScope valid reports GraphScope's validated cells. The overall processing speedup is omitted because it would combine GPU device timers with CPU existing-graph wall-time surrogates.}",
        r"\label{tab:category-13-summary}",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Metric & Family & Common W/T & Common pairs & Sole valid & GraphScope valid & Pairwise speedup \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        metric_label = "Public return" if row.metric == "e2e" else "Processing"
        speedup = (
            f"{float(row.speedup_over_best_competitor):.2f}$\\times$"
            if finite(row.speedup_over_best_competitor)
            else "--"
        )
        lines.append(
            f"{metric_label} & {latex(row.category)} & "
            f"{int(row.common_strict_wins)}/{int(row.common_ties)} & "
            f"{int(row.common_pairs)} & {int(row.sole_validated)} & "
            f"{int(row.graphscope_validated)} & {speedup} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (output / "paper_table_category_13_summary.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return data


def write_numeric_report(ledger, pairwise, pair_baseline, category, output):
    summary = {
        "datasets": 13,
        "functions": 16,
        "attempted_workloads": 208,
        "relative_tie_tolerance": RELATIVE_TIE_TOLERANCE,
        "tie_definition": (
            "absolute timing difference <= 0.05% of the larger pair timing"
        ),
    }
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
        tolerance = RELATIVE_TIE_TOLERANCE * np.maximum(
            np.maximum(
                pd.to_numeric(
                    competitive.eggpu_seconds, errors="coerce"
                ).abs(),
                pd.to_numeric(
                    competitive.best_competitor_seconds, errors="coerce"
                ).abs(),
            ),
            1.0e-15,
        )
        delta = (
            pd.to_numeric(competitive.eggpu_seconds, errors="coerce")
            - pd.to_numeric(
                competitive.best_competitor_seconds, errors="coerce"
            )
        )
        strict_wins = int((delta < -tolerance).sum())
        ties = int((delta.abs() <= tolerance).sum())
        losses = int((delta > tolerance).sum())
        key = metric.lower()
        summary[f"{key}_eggpu_successful_workloads"] = len(valid)
        summary[f"{key}_total_coverage_cells"] = len(valid)
        summary[f"{key}_only_validated"] = int(valid.only_validated.sum())
        summary[f"{key}_sole_validated_cells"] = int(valid.only_validated.sum())
        summary[f"{key}_competitive_pairs"] = len(competitive)
        summary[f"{key}_common_pair_strict_wins"] = strict_wins
        summary[f"{key}_common_pair_ties"] = ties
        summary[f"{key}_common_pair_losses"] = losses
        graphscope = ledger[ledger.baseline.eq("GraphScope")]
        summary[f"{key}_graphscope_validated_cells"] = int(
            graphscope.apply(lambda row: metric_ok(row, metric), axis=1).sum()
        )
        if metric == "e2e":
            summary[f"{key}_speedup_over_best_competitor"] = geomean(
                competitive.speedup
            )
        else:
            summary["kernel_cross_system_aggregate_reported"] = False
            summary["kernel_cross_system_aggregate_reason"] = (
                "An overall processing speedup would combine GPU device "
                "timers with CPU existing-graph wall-time surrogates."
            )
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
        if metric == "e2e":
            # Gunrock exposes a prepared-native processing interval, not a
            # comparable high-level library-call boundary.  The only strict
            # high-level GPU backend in the public-return candidate set is
            # nx-cugraph.
            summary["e2e_speedup_over_strict_nx_cugraph"] = geomean(gpu_rows)
            summary["e2e_strict_nx_cugraph_common_pairs"] = len(gpu_rows)
            summary["e2e_comparable_high_level_gpu_baselines"] = ["nx-cugraph"]
        else:
            summary["kernel_speedup_over_best_native_gpu"] = geomean(gpu_rows)
            summary["kernel_best_native_gpu_common_pairs"] = len(gpu_rows)
            summary["kernel_comparable_native_gpu_baselines"] = list(
                DEVICE_TIMER_BASELINES
            )
        summary[f"{key}_speedup_over_best_cpu"] = geomean(cpu_rows)
        summary[f"{key}_best_cpu_common_pairs"] = len(cpu_rows)
        metric_label = "PROCESSING" if metric == "kernel" else "PUBLIC RETURN"
        gpu_label = (
            "strict nx-cugraph"
            if metric == "e2e"
            else "the best comparable native GPU implementation"
        )
        lines.extend(
            [
                f"## {metric_label}",
                "",
                f"- EGGPU coverage: **{len(valid)}/208** successful workloads.",
                f"- Sole-validated coverage cells: **{int(valid.only_validated.sum())}**; these are excluded from performance wins.",
                f"- Common-pair outcomes at the 0.05% relative tie tolerance: **{strict_wins} strict wins / {ties} ties / {losses} losses** over {len(competitive)} pairs.",
            ]
        )
        if metric == "e2e":
            lines.append(
                "- Geometric-mean speedup over the pairwise best competitor: "
                f"**{geomean(competitive.speedup):.2f}x**."
            )
        else:
            lines.append(
                "- No overall cross-system processing speedup is reported: "
                "that aggregate would mix GPU device timers with CPU "
                "existing-graph wall-time surrogates."
            )
        lines.extend(
            [
                f"- Speedup over {gpu_label}: **{geomean(gpu_rows):.2f}x** on {len(gpu_rows)} common pairs.",
                f"- Speedup over the pairwise best CPU baseline: **{geomean(cpu_rows):.2f}x** on {len(cpu_rows)} common pairs.",
                "",
            ]
        )
    summary["family_outcomes"] = {}
    for metric in ("e2e", "kernel"):
        metric_rows = category[category.metric.eq(metric)]
        family_records = metric_rows[
            ~metric_rows.category.eq("Overall")
        ].to_dict("records")
        total_record = metric_rows[
            metric_rows.category.eq("Overall")
        ].iloc[0].to_dict()
        for record in (*family_records, total_record):
            for field, value in tuple(record.items()):
                if isinstance(value, float) and not math.isfinite(value):
                    record[field] = None
        summary["family_outcomes"][
            "public_return" if metric == "e2e" else "processing"
        ] = {
            "families": family_records,
            "total": total_record,
        }
    statuses = ledger.execution_status.value_counts().to_dict()
    summary["cell_status_counts"] = statuses
    memory = ledger[
        ledger.baseline.eq("EGGPU") & pd.to_numeric(ledger.gpu_peak_mb_mean, errors="coerce").notna()
    ] if "gpu_peak_mb_mean" in ledger else pd.DataFrame()
    if not memory.empty:
        memory.to_csv(output / "eggpu_memory_13_details.csv", index=False)
        summary["eggpu_memory_cells"] = len(memory)
        summary["eggpu_max_gpu_peak_mb"] = float(memory.gpu_peak_mb_mean.max())

    resource_rows = []
    if "host_rss_peak_mb_mean" in ledger:
        for baseline, group in ledger.groupby("baseline"):
            values = pd.to_numeric(
                group.host_rss_peak_mb_mean, errors="coerce"
            ).dropna()
            values = values[np.isfinite(values) & (values > 0)]
            if len(values):
                resource_rows.append(
                    {
                        "baseline": baseline,
                        "resource": "host_peak_rss_mb",
                        "cells": len(values),
                        "geometric_mean_mb": geomean(values),
                        "median_mb": float(np.median(values)),
                        "maximum_mb": float(np.max(values)),
                        "comparison_scope": (
                            "isolated benchmark process tree; compare only "
                            "within the host-memory resource domain"
                        ),
                    }
                )
    if "gpu_peak_mb_mean" in ledger:
        for baseline in ("EGGPU", "nx-cugraph", "Gunrock"):
            group = ledger[ledger.baseline.eq(baseline)]
            values = pd.to_numeric(
                group.gpu_peak_mb_mean, errors="coerce"
            ).dropna()
            values = values[np.isfinite(values) & (values > 0)]
            if len(values):
                resource_rows.append(
                    {
                        "baseline": baseline,
                        "resource": "process_gpu_peak_mb",
                        "cells": len(values),
                        "geometric_mean_mb": geomean(values),
                        "median_mb": float(np.median(values)),
                        "maximum_mb": float(np.max(values)),
                        "comparison_scope": (
                            "isolated benchmark process GPU memory; compare "
                            "only within the device-memory resource domain"
                        ),
                    }
                )
    if resource_rows:
        resource_summary = pd.DataFrame(resource_rows)
        resource_summary.to_csv(
            output / "memory_resource_domain_summary.csv", index=False
        )
        summary["memory_resource_rows"] = len(resource_summary)

    graphscope_memory = (
        ledger[
            ledger.baseline.eq("GraphScope")
            & pd.to_numeric(
                ledger.host_rss_peak_mb_mean, errors="coerce"
            ).notna()
        ].copy()
        if "host_rss_peak_mb_mean" in ledger
        else pd.DataFrame()
    )
    if not graphscope_memory.empty:
        graphscope_memory.to_csv(
            output / "graphscope_host_memory_13_details.csv", index=False
        )
        graphscope_values = pd.to_numeric(
            graphscope_memory.host_rss_peak_mb_mean, errors="coerce"
        ).dropna()
        summary["graphscope_host_memory_cells"] = len(graphscope_values)
        summary["graphscope_host_rss_geomean_mb"] = geomean(graphscope_values)
        summary["graphscope_host_rss_max_mb"] = float(graphscope_values.max())
        lines.extend(
            [
                "## GraphScope host memory",
                "",
                f"- Successful isolated memory cells: **{len(graphscope_values)}**.",
                (
                    "- Geometric-mean absolute process-tree peak RSS: "
                    f"**{geomean(graphscope_values):.2f} MB**."
                ),
                (
                    "- Maximum absolute process-tree peak RSS: "
                    f"**{float(graphscope_values.max()):.2f} MB**."
                ),
                "- GPU memory is N/A because GraphScope is evaluated as a CPU/distributed analytical-engine baseline.",
                "",
            ]
        )
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_baseline_versions(output, ledger, ledger_path):
    csv_path = output / "paper_table_baseline_versions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing copied baseline version table: {csv_path}")
    versions = pd.read_csv(csv_path, keep_default_na=False)
    versions = versions[
        ~versions["system"].astype(str).str.casefold().eq("sygraph")
        & ~versions["system"].astype(str).str.casefold().eq("graphscope")
    ].copy()
    eggpu = ledger[
        ledger["baseline"].astype(str).eq("EGGPU")
        & ledger["execution_status"].astype(str).eq("ok")
    ].copy()
    if len(eggpu) != 208:
        raise ValueError(f"expected 208 successful EGGPU rows, found {len(eggpu)}")

    def one_nonempty(column):
        values = {
            str(value)
            for value in eggpu[column]
            if str(value) and str(value).lower() != "nan"
        }
        if len(values) != 1:
            raise ValueError(f"EGGPU {column} has {len(values)} distinct values")
        return next(iter(values))

    candidate = one_nonempty("candidate_sha256")
    runtime = one_nonempty("runtime_python_snapshot_sha256")
    memory_candidate = one_nonempty("memory_candidate_sha256")
    memory_runtime = one_nonempty("memory_runtime_python_snapshot_sha256")
    if candidate != memory_candidate or runtime != memory_runtime:
        raise ValueError("EGGPU timing and memory runtime identities differ")
    if not pd.to_numeric(
        eggpu["memory_sample_count"], errors="raise"
    ).eq(3).all():
        raise ValueError("EGGPU memory evidence is not three-process complete")
    if not pd.to_numeric(eggpu["sample_count"], errors="raise").eq(5).all():
        raise ValueError("EGGPU timing evidence is not five-process complete")
    memory_sources = sorted(
        {
            Path(str(value)).name
            for value in eggpu["memory_result_source"]
            if str(value) and str(value).lower() != "nan"
        }
    )
    timing_sources = sorted(
        {
            Path(str(value)).name
            for value in eggpu["timing_result_source"]
            if str(value) and str(value).lower() != "nan"
        }
    )
    scope = (
        f"V15 timing and memory: candidate binary {candidate[:12]}; "
        f"frozen Python runtime {runtime[:12]}"
    )
    eggpu_mask = versions["system"].astype(str).eq("EGGPU")
    if int(eggpu_mask.sum()) != 1:
        raise ValueError("baseline-version table does not contain one EGGPU row")
    versions.loc[eggpu_mask, "version"] = (
        "EasyGraph 1.6 + V15 current-runtime candidate"
    )
    versions.loc[eggpu_mask, "commit"] = candidate
    versions.loc[eggpu_mask, "artifact_scope"] = scope
    for column, value in {
        "timing_candidate_binary_sha256": candidate,
        "memory_source_ledger_sha256": sha256(ledger_path),
        "memory_archived_source_snapshot_sha256": "",
        "memory_row_level_binary_sha256": memory_candidate,
        "timing_runtime_python_snapshot_sha256": runtime,
        "memory_runtime_python_snapshot_sha256": memory_runtime,
    }.items():
        if column not in versions:
            versions[column] = ""
        versions.loc[eggpu_mask, column] = value

    graphscope = pd.DataFrame(
        [
            {
                "system": "GraphScope",
                "role": "CPU/distributed graph analytical engine baseline",
                "version": "0.29.0 (PyPI)",
                "commit": "d8581c18ac92f7cf1f0ffdd2f4a739958cbdcd1c",
                "artifact_scope": (
                    "PyPI runtime; native GAE/FLASH applications only; "
                    "upstream source cross-check at the recorded commit"
                ),
            }
        ]
    )
    insert_at = next(
        (
            index
            for index, value in enumerate(versions["system"])
            if value == "nx-cugraph"
        ),
        len(versions),
    )
    versions = pd.concat(
        [versions.iloc[:insert_at], graphscope, versions.iloc[insert_at:]],
        ignore_index=True,
    )
    versions.to_csv(csv_path, index=False)

    json_path = output / "paper_baseline_versions.json"
    payload = json.loads(json_path.read_text()) if json_path.is_file() else {}
    payload.pop("sygraph_build", None)
    payload["eggpu_candidate_binary_sha256"] = candidate
    payload["eggpu_runtime_python_snapshot_sha256"] = runtime
    payload["eggpu_provenance"] = {
        "release_label": "V15",
        "timing": {
            "candidate_binary_sha256": candidate,
            "runtime_python_snapshot_sha256": runtime,
            "scope": "208 current-runtime timing cells",
            "sample_count_per_cell": 5,
            "display_estimator": "minimum_of_five",
            "reported_error": "sample_standard_deviation_ddof1",
            "source_labels": timing_sources,
        },
        "memory": {
            "candidate_binary_sha256": memory_candidate,
            "runtime_python_snapshot_sha256": memory_runtime,
            "scope": "208 current-runtime memory cells",
            "sample_count_per_cell": 3,
            "source_labels": memory_sources,
            "measured_with_timing_candidate": True,
        },
        "final_ledger_sha256": sha256(ledger_path),
    }
    systems = [
        item
        for item in payload.get("systems", [])
        if str(item.get("system", "")).casefold() not in {"sygraph", "graphscope"}
    ]
    eggpu_systems = [
        item for item in systems if str(item.get("system", "")) == "EGGPU"
    ]
    if len(eggpu_systems) != 1:
        raise ValueError("baseline-version JSON does not contain one EGGPU system")
    eggpu_system = eggpu_systems[0]
    eggpu_system.update(
        {
            "version": "EasyGraph 1.6 + V15 current-runtime candidate",
            "commit": candidate,
            "artifact_scope": scope,
            "timing_provenance": payload["eggpu_provenance"]["timing"],
            "memory_provenance": payload["eggpu_provenance"]["memory"],
        }
    )
    systems.insert(insert_at, graphscope.iloc[0].to_dict())
    payload["systems"] = systems
    payload["graphscope_runtime"] = {
        "distribution": "PyPI graphscope==0.29.0",
        "source_crosscheck_commit": (
            "d8581c18ac92f7cf1f0ffdd2f4a739958cbdcd1c"
        ),
        "execution": "one local hosts-mode analytical worker",
        "qualification": "native GAE/FLASH only; no NetworkX fallback",
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "% Requires: booktabs, tabularx",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Versions and artifacts of the evaluated systems.}",
        r"\label{tab:baseline-versions}",
        r"\small",
        r"\begin{tabularx}{\columnwidth}{l l X}",
        r"\toprule",
        r"System & Version & Evaluated artifact \\",
        r"\midrule",
    ]
    for row in versions.itertuples(index=False):
        commit = f" ({str(row.commit)[:10]})" if str(row.commit) else ""
        lines.append(
            f"{latex(row.system)} & {latex(row.version)}{latex(commit)} & "
            f"{latex(row.artifact_scope)} \\\\"
        )
    lines.extend(
        [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    )
    (output / "paper_table_baseline_versions.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(args.ledger.resolve(), low_memory=False)
    configure_baselines(ledger)
    write_baseline_versions(output, ledger, args.ledger.resolve(strict=True))
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
    category = write_category_table(ledger, pairwise, output)
    summary = write_numeric_report(ledger, pairwise, pair_baseline, category, output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
