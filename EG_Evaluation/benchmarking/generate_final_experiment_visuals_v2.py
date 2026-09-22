#!/usr/bin/env python3
"""Generate the final EGGPU experiment figures from audited result artifacts.

The figures use measured time or memory on every axis.  Ratios are annotations,
never axis variables.  EGGPU uses the paper-selected best value from five runs;
competitors use the arithmetic mean and all raw mean/SD fields remain in CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate_final_13_dataset_story as story
import generate_final_paper_bundle as base


FAMILIES = list(base.CATEGORY_ORDER)
FAMILY_COLOR = dict(base.CATEGORY_COLOR)
FAMILY_FILL = dict(base.CATEGORY_FILL)
SCALING_FUNCTIONS = ("PageRank", "WCC", "BFS", "SSSP")
SYSTEM_COLOR = {
    "EGGPU": "#3D87B3",
    "igraph": "#79B99E",
    "nx-cugraph": "#9A89CB",
    "SYgraph": "#879E59",
}
SYSTEM_MARKER = {"EGGPU": "o", "igraph": "^", "nx-cugraph": "P", "SYgraph": "*"}
ALL_SYSTEM_COLOR = {
    "EGGPU": "#3D87B3",
    "easygraph-cpp": "#D7AD49",
    "easygraph-cpu": "#B9ADA5",
    "igraph": "#79B99E",
    "networkx": "#D98F93",
    "nx-cugraph": "#9A89CB",
    "Gunrock": "#98A1AB",
    "SYgraph": "#879E59",
}
ALL_SYSTEM_MARKER = {
    "EGGPU": "o",
    "easygraph-cpp": "s",
    "easygraph-cpu": "D",
    "igraph": "^",
    "networkx": "v",
    "nx-cugraph": "P",
    "Gunrock": "X",
    "SYgraph": "*",
}

FAMILY_SHORT = {
    "Centrality": "Centrality",
    "Connectivity": "Connectivity",
    "Paths & Spanning Trees": "Path & Spanning",
    "Structural Holes": "Structural Holes",
}


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def geomean(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.exp(np.log(values).mean())) if len(values) else float("nan")


def save(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(axis, bars, values, unit="ms"):
    maximum = max(values) if values else 1.0
    for bar, value in zip(bars, values):
        if not math.isfinite(value):
            continue
        text = f"{value:.2f}" if value < 100 else f"{value:.1f}"
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + maximum * 0.022,
            text,
            ha="center",
            va="bottom",
            fontsize=6.8,
        )


def category_overall_figure(out_dir: Path):
    """Draw the final support-conditioned load/processing/E2E overview.

    The middle panel follows the LDBC Graphalytics processing-time boundary.
    CPU systems use their in-memory algorithm call as a processing-time
    surrogate.  nx-cugraph is omitted because its strict high-level backend
    does not expose a separable device timer.  Conversely, Gunrock is omitted
    from E2E because its available artifact is an external CLI rather than an
    aligned in-process public function call.
    """
    path = out_dir / "final_13_cell_outcome_ledger.csv"
    ledger = pd.read_csv(path, low_memory=False)
    systems = [
        "EGGPU", "easygraph-cpp", "easygraph-cpu", "igraph",
        "networkx", "nx-cugraph", "Gunrock",
    ]
    # SYgraph's validated paper scope is native BFS only.  It remains in the
    # full function-level ledger and BFS scaling panel, but a one-function point
    # would not be a representative four-family average.
    metric_columns = {
        "Graph construction": "build_paper_seconds",
        "Processing time": "kernel_paper_seconds",
        "End-to-end": "e2e_paper_seconds",
    }
    metric_systems = {
        "Graph construction": tuple(system for system in systems if system != "Gunrock"),
        "Processing time": tuple(system for system in systems if system != "nx-cugraph"),
        "End-to-end": tuple(system for system in systems if system != "Gunrock"),
    }
    validation_ok = {"pass", "reference", "external_reference_pass", "sampled_pass"}
    rows = []
    for title, column in metric_columns.items():
        if column not in ledger:
            continue
        source = ledger.copy()
        source[column] = pd.to_numeric(source[column], errors="coerce")
        source = source[
            source["execution_status"].eq("ok")
            & source["validation_status"].isin(validation_ok)
            & source[column].notna()
            & source[column].gt(0)
        ]
        for family in FAMILIES:
            for system in metric_systems[title]:
                values = source[
                    source["category"].eq(family)
                    & source["baseline"].eq(system)
                ][column]
                value = geomean(values)
                if math.isfinite(value):
                    rows.append(
                        {
                            "metric": title,
                            "family": family,
                            "baseline": system,
                            "geomean_seconds": value,
                            "valid_function_dataset_cells": len(values),
                            "aggregation": "support-conditioned geometric mean",
                            "timing_boundary": (
                                "graph construction/load time"
                                if title == "Graph construction"
                                else (
                                    "exact device timer"
                                    if title == "Processing time" and system in {"EGGPU", "Gunrock"}
                                    else (
                                        "in-memory algorithm wall-time surrogate"
                                        if title == "Processing time"
                                        else "aligned in-process public function call"
                                    )
                                )
                            ),
                        }
                    )
    data = pd.DataFrame(rows)
    data.to_csv(out_dir / "category_time_by_baseline_3panel.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
    y_positions = np.arange(len(FAMILIES))
    for axis, (title, _column) in zip(axes, metric_columns.items()):
        for index, family in enumerate(FAMILIES):
            axis.axhspan(
                index - 0.48,
                index + 0.48,
                color=FAMILY_FILL[family],
                alpha=0.48,
                linewidth=0,
                zorder=0,
            )
        subset = data[data["metric"].eq(title)]
        for system in systems:
            points = subset[subset["baseline"].eq(system)]
            if points.empty:
                continue
            positions = [FAMILIES.index(value) for value in points["family"]]
            axis.scatter(
                points["geomean_seconds"],
                positions,
                s=42 if system == "EGGPU" else 26,
                marker=ALL_SYSTEM_MARKER[system],
                color=ALL_SYSTEM_COLOR[system],
                edgecolor="#1E3A4A" if system == "EGGPU" else "white",
                linewidth=0.85 if system == "EGGPU" else 0.5,
                alpha=0.96,
                label=system,
                zorder=5 if system == "EGGPU" else 3,
            )
        axis.set_xscale("log")
        axis.set_title(title, fontweight="bold", pad=8)
        axis.set_xlabel("Geometric-mean time (s, log scale)")
        axis.set_yticks(y_positions)
        axis.set_yticklabels([FAMILY_SHORT[item] for item in FAMILIES])
        axis.set_ylim(len(FAMILIES) - 0.5, -0.5)
        axis.grid(axis="x", which="major", color="#DCE5EB", linewidth=0.65)
        axis.grid(axis="x", which="minor", color="#EFF3F6", linewidth=0.35)
    axes[0].set_ylabel("Function family")
    for label, family in zip(axes[0].get_yticklabels(), FAMILIES):
        label.set_color(FAMILY_COLOR[family])
        label.set_fontweight("bold")
    legend_items = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        legend_items.update(zip(labels, handles))
    fig.legend(
        [legend_items[name] for name in systems if name in legend_items],
        [name for name in systems if name in legend_items],
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
    )
    fig.text(
        0.5,
        0.015,
        "Each point is a geometric mean over correctness-validated cells supported by that system.",
        ha="center",
        fontsize=6.1,
        color="#51636E",
    )
    fig.subplots_adjust(left=0.16, right=0.995, top=0.73, bottom=0.25, wspace=0.14)
    save(fig, out_dir / "category_time_by_baseline_3panel")
    return data


def first_use_figure(first_dir: Path, out_dir: Path):
    raw = pd.read_csv(first_dir / "first_use_vs_steady.csv")
    raw = raw[raw["dataset"].isin(story.CROSS_LIBRARY_11)].copy()
    raw["family"] = raw["function"].map(base.FUNCTION_CATEGORY)
    rows = []
    for family in FAMILIES:
        for metric in ("e2e", "kernel"):
            part = raw[raw["family"].eq(family) & raw["metric"].eq(metric)]
            first = geomean(part["first_use_mean_seconds"])
            steady = geomean(part["steady_mean_seconds"])
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "pairs": len(part),
                    "first_use_geomean_seconds": first,
                    "steady_state_geomean_seconds": steady,
                    "first_use_over_steady": first / steady,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "first_use_steady_actual_times.csv", index=False)
    raw.to_csv(out_dir / "first_use_steady_pair_details.csv", index=False)

    # Composite presentation: the top row gives the family-level E2E/kernel
    # picture on zero-based linear axes, while the bottom row exposes every
    # function's measured E2E time.  The function panels use explicit log axes
    # because their actual times span several orders of magnitude; exact values
    # are printed alongside every pair of points.
    fig = plt.figure(figsize=(7.2, 4.55))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.65], hspace=0.52, wspace=0.42)
    axes = [fig.add_subplot(grid[0, :2]), fig.add_subplot(grid[0, 2:])]
    x = np.arange(len(FAMILIES))
    width = 0.34
    for axis, metric, title in zip(
        axes,
        ("e2e", "kernel"),
        ("End-to-end latency", "GPU computation time"),
    ):
        lookup = summary[summary["metric"].eq(metric)].set_index("family")
        first = [1000 * float(lookup.loc[item, "first_use_geomean_seconds"]) for item in FAMILIES]
        steady = [1000 * float(lookup.loc[item, "steady_state_geomean_seconds"]) for item in FAMILIES]
        bars_first = axis.bar(
            x - width / 2,
            first,
            width,
            color=[FAMILY_COLOR[item] for item in FAMILIES],
            alpha=0.45,
            edgecolor="white",
            label="First use",
        )
        bars_steady = axis.bar(
            x + width / 2,
            steady,
            width,
            color=[FAMILY_COLOR[item] for item in FAMILIES],
            alpha=0.95,
            edgecolor="#52616B",
            linewidth=0.45,
            label="Steady state",
        )
        annotate_bars(axis, bars_first, first)
        annotate_bars(axis, bars_steady, steady)
        panel_maximum = max(first + steady)
        for index, family in enumerate(FAMILIES):
            ratio = float(lookup.loc[family, "first_use_over_steady"])
            axis.text(
                index,
                max(first[index], steady[index]) + panel_maximum * 0.115,
                f"{ratio:.2f}x",
                ha="center",
                va="bottom",
                fontsize=6.2,
                fontweight="bold",
                color=FAMILY_COLOR[family],
            )
        axis.set_ylim(0, panel_maximum * 1.36)
        axis.set_ylabel("Time (ms)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#E5EBF0", linewidth=0.65)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
    )
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels([FAMILY_SHORT[item] for item in FAMILIES], rotation=12)
        for label, family in zip(axis.get_xticklabels(), FAMILIES):
            label.set_color(FAMILY_COLOR[family])
            label.set_fontweight("bold")

    function_rows = []
    for family_index, family in enumerate(FAMILIES):
        axis = fig.add_subplot(grid[1, family_index])
        family_functions = [
            function for function in base.FUNCTION_ORDER
            if base.FUNCTION_CATEGORY[function] == family
        ]
        function_data = raw[
            raw["family"].eq(family) & raw["metric"].eq("e2e")
        ]
        function_summary = (
            function_data.groupby("function", as_index=False)
            .agg(
                first_use_seconds=("first_use_mean_seconds", geomean),
                steady_state_seconds=("steady_mean_seconds", geomean),
                datasets=("dataset", "nunique"),
            )
            .set_index("function")
        )
        family_functions = [item for item in family_functions if item in function_summary.index]
        y = np.arange(len(family_functions))
        first_ms = [1000 * float(function_summary.loc[item, "first_use_seconds"]) for item in family_functions]
        steady_ms = [1000 * float(function_summary.loc[item, "steady_state_seconds"]) for item in family_functions]
        minimum = min(first_ms + steady_ms) if first_ms else 1.0
        maximum = max(first_ms + steady_ms) if first_ms else 1.0
        for index, (first_value, steady_value, function) in enumerate(
            zip(first_ms, steady_ms, family_functions)
        ):
            first_offset = (-3, -6) if index == 0 else (-3, 5)
            first_vertical = "top" if index == 0 else "bottom"
            steady_offset = (3, -6) if index == 0 else (3, -5)
            axis.plot(
                [steady_value, first_value],
                [index, index],
                color=FAMILY_COLOR[family],
                alpha=0.45,
                linewidth=1.5,
                zorder=1,
            )
            axis.scatter(
                [first_value], [index], s=23, marker="o",
                facecolor="white", edgecolor=FAMILY_COLOR[family],
                linewidth=1.0, zorder=3,
            )
            axis.scatter(
                [steady_value], [index], s=23, marker="o",
                facecolor=FAMILY_COLOR[family], edgecolor="#52616B",
                linewidth=0.45, zorder=4,
            )
            axis.annotate(
                f"F {first_value:.2f}",
                (first_value, index),
                xytext=first_offset,
                textcoords="offset points",
                ha="right",
                va=first_vertical,
                fontsize=4.5,
                color="#344955",
                clip_on=True,
            )
            axis.annotate(
                f"S {steady_value:.2f}",
                (steady_value, index),
                xytext=steady_offset,
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=4.5,
                color="#344955",
                clip_on=True,
            )
            function_rows.append(
                {
                    "family": family,
                    "function": function,
                    "datasets": int(function_summary.loc[function, "datasets"]),
                    "first_use_geomean_seconds": first_value / 1000.0,
                    "steady_state_geomean_seconds": steady_value / 1000.0,
                    "first_use_over_steady": first_value / steady_value,
                }
            )
        axis.set_xscale("log")
        axis.set_xlim(max(minimum * 0.55, 1.0e-3), maximum * 1.65)
        axis.set_yticks(y)
        axis.set_yticklabels(family_functions)
        axis.invert_yaxis()
        axis.set_xlabel("E2E time (ms, log)")
        axis.set_title(family, color=FAMILY_COLOR[family], fontweight="bold")
        axis.grid(axis="x", which="major", color="#E5EBF0", linewidth=0.55)
        axis.grid(axis="x", which="minor", color="#F1F4F6", linewidth=0.3)
    pd.DataFrame(function_rows).to_csv(
        out_dir / "first_use_steady_function_actual_times.csv", index=False
    )
    detail_legend = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="white",
            markeredgecolor="#52616B", label="First use",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#52616B",
            markeredgecolor="#52616B", label="Steady state",
        ),
    ]
    fig.legend(
        detail_legend,
        [item.get_label() for item in detail_legend],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.09, right=0.995, top=0.89, bottom=0.14)
    save(fig, out_dir / "first_use_vs_steady_actual_time")
    return summary


def complete_workflow_datasets(summary, baselines):
    """Return datasets with all three calls for every requested baseline."""
    complete = []
    for dataset, part in summary.groupby("dataset"):
        valid = True
        for baseline in baselines:
            positions = set(
                pd.to_numeric(
                    part[part["baseline"].eq(baseline)]["call_position"],
                    errors="coerce",
                ).dropna().astype(int)
            )
            if positions != {1, 2, 3}:
                valid = False
                break
        if valid:
            complete.append(dataset)
    return sorted(complete)


def case_study_figure(cumulative_dir: Path, out_dir: Path):
    source = pd.read_csv(cumulative_dir / "cumulative_workflow_summary.csv")
    common_datasets = complete_workflow_datasets(
        source, ("EGGPU", "EGGPU-isolated")
    )
    source = source[source["dataset"].isin(common_datasets)].copy()
    reused = source[source["baseline"].eq("EGGPU")].copy()
    isolated = source[source["baseline"].eq("EGGPU-isolated")].copy()
    details = reused.merge(
        isolated,
        on=["dataset", "call_position", "function"],
        suffixes=("_workflow", "_isolated"),
        how="inner",
    )
    rows = []
    for (function, position), part in details.groupby(["function", "call_position"], sort=False):
        for metric, column in (
            ("e2e", "call_best_seconds"),
            ("kernel", "kernel_best_seconds"),
        ):
            workflow_column = f"{column}_workflow"
            isolated_column = f"{column}_isolated"
            if workflow_column not in part or isolated_column not in part:
                continue
            workflow = geomean(part[workflow_column])
            separate = geomean(part[isolated_column])
            if not math.isfinite(workflow) or not math.isfinite(separate):
                continue
            rows.append(
                {
                    "function": function,
                    "call_position": int(position),
                    "metric": metric,
                    "datasets": len(part),
                    "isolated_geomean_seconds": separate,
                    "same_graph_workflow_geomean_seconds": workflow,
                    "isolated_over_workflow": separate / workflow,
                }
            )
    summary = pd.DataFrame(rows)
    details.to_csv(out_dir / "case_study_pair_details.csv", index=False)
    summary.to_csv(out_dir / "case_study_call_summary.csv", index=False)
    order = ["WCC", "PageRank", "BFS"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    x = np.arange(len(order))
    width = 0.34
    for axis, metric, title, ylabel in (
        (axes[0], "e2e", "User-visible call latency", "End-to-end time (ms)"),
        (axes[1], "kernel", "CUDA algorithm time", "Kernel time (ms)"),
    ):
        lookup = summary[summary["metric"].eq(metric)].set_index("function")
        separate = [
            1000 * float(lookup.loc[item, "isolated_geomean_seconds"])
            for item in order
        ]
        reused = [
            1000 * float(lookup.loc[item, "same_graph_workflow_geomean_seconds"])
            for item in order
        ]
        b1 = axis.bar(
            x - width / 2,
            separate,
            width,
            color="#C8D2DA",
            edgecolor="white",
            label="Graph-state reuse disabled",
        )
        b2 = axis.bar(
            x + width / 2,
            reused,
            width,
            color="#4E91BA",
            edgecolor="#285875",
            linewidth=0.5,
            label="Same-graph reuse",
        )
        annotate_bars(axis, b1, separate)
        annotate_bars(axis, b2, reused)
        panel_maximum = max(separate + reused)
        for index, item in enumerate(order):
            ratio = float(lookup.loc[item, "isolated_over_workflow"])
            axis.text(
                index,
                max(separate[index], reused[index]) + panel_maximum * 0.105,
                f"{ratio:.2f}x",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )
        axis.set_ylim(0, panel_maximum * 1.30)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#E5EBF0", linewidth=0.65)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(["Call 1\nWCC", "Call 2\nPageRank", "Call 3\nBFS"])
    fig.subplots_adjust(left=0.09, right=0.995, top=0.80, bottom=0.22, wspace=0.27)
    save(fig, out_dir / "z_case_study_same_graph_workflow")
    return summary


def cumulative_figure(cumulative_dir: Path, out_dir: Path):
    path = cumulative_dir / "cumulative_workflow_summary.csv"
    if not path.is_file():
        return None
    summary = pd.read_csv(path)
    compared_baselines = ("EGGPU", "igraph", "nx-cugraph")
    common_datasets = complete_workflow_datasets(summary, compared_baselines)
    common = summary[summary["dataset"].isin(common_datasets)].copy()
    if not common_datasets:
        raise RuntimeError(
            "No dataset completed all three cumulative-workflow calls for "
            "EGGPU, igraph, and nx-cugraph"
        )
    rows = []
    for baseline in compared_baselines:
        for position in (1, 2, 3):
            part = common[
                common["baseline"].eq(baseline)
                & common["call_position"].eq(position)
            ]
            column = "cumulative_best_seconds" if baseline == "EGGPU" else "cumulative_mean_seconds"
            rows.append(
                {
                    "baseline": baseline,
                    "call_position": position,
                    "function": {1: "WCC", 2: "PageRank", 3: "BFS"}[position],
                    "datasets": len(part),
                    "common_dataset_names": ";".join(common_datasets),
                    "cumulative_geomean_seconds": geomean(part[column]),
                    "estimator": "best of five" if baseline == "EGGPU" else "mean of five",
                }
            )
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(out_dir / "cumulative_workflow_aggregate.csv", index=False)
    common.to_csv(out_dir / "cumulative_workflow_dataset_details.csv", index=False)

    fig, axis = plt.subplots(figsize=(7.2, 3.05))
    labels = ["Start", "WCC", "+ PageRank", "+ BFS"]
    for baseline in ("EGGPU", "igraph", "nx-cugraph"):
        part = aggregate[aggregate["baseline"].eq(baseline)].sort_values("call_position")
        values = [0.0] + part["cumulative_geomean_seconds"].tolist()
        axis.plot(
            range(4),
            values,
            marker=SYSTEM_MARKER[baseline],
            markersize=6.5,
            linewidth=1.8,
            color=SYSTEM_COLOR[baseline],
            label=baseline,
        )
        for x, value in enumerate(values[1:], start=1):
            if baseline == "EGGPU":
                offset, vertical = (0, -11), "top"
            else:
                offset, vertical = (0, 7), "bottom"
            axis.annotate(
                f"{value:.3f}s",
                (x, value),
                xytext=offset,
                textcoords="offset points",
                color=SYSTEM_COLOR[baseline],
                ha="center",
                va=vertical,
                fontsize=6.3,
            )
    axis.set_xlim(0, 3)
    axis.set_ylim(bottom=0)
    axis.set_xticks(range(4))
    axis.set_xticklabels(labels)
    axis.set_ylabel("Cumulative analysis time (s)")
    axis.grid(axis="y", color="#E3EAF0", linewidth=0.65)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    fig.subplots_adjust(left=0.105, right=0.99, top=0.96, bottom=0.18)
    save(fig, out_dir / "cumulative_cold_start_workflow")
    return aggregate


def common_geomean_pair(data, left_variant, right_variant):
    identity = ["dataset"]
    if "workflow_order_id" in data.columns:
        identity.append("workflow_order_id")
    left = data[data["variant"].eq(left_variant)][identity + ["seconds"]].rename(columns={"seconds": "left"})
    right = data[data["variant"].eq(right_variant)][identity + ["seconds"]].rename(columns={"seconds": "right"})
    paired = left.merge(right, on=identity, how="inner", validate="one_to_one")
    return paired, geomean(paired["left"]), geomean(paired["right"])


def ablation_figure(ablation_dir: Path, out_dir: Path):
    workflow = pd.read_csv(ablation_dir / "ablation_workflow_dataset_totals.csv")
    returns = pd.read_csv(ablation_dir / "ablation_return_pairs.csv")
    layout = pd.read_csv(ablation_dir / "ablation_layout_per_dataset.csv")
    controlled = pd.read_csv(
        ablation_dir / "ablation_workflow_controlled_comparisons.csv"
    )
    rows = []

    comparisons = []
    for title, variant, left_label, right_label in (
        ("GraphContext", "no_graph_context", "EGGPU", "Without GraphContext"),
        ("C++ graph cache", "no_cpp_graph_cache", "EGGPU", "Without C++ cache"),
    ):
        paired, full, ablated = common_geomean_pair(workflow, "full", variant)
        comparisons.append((title, "Workflow time (s)", full, ablated, len(paired), left_label, right_label))
    deferred = geomean(returns["call_return_seconds"])
    standard = geomean(returns["standard_container_return_seconds"])
    comparisons.append(
        (
            "Result reconstruction",
            "Return time (s)",
            deferred,
            standard,
            len(returns),
            "EGGPU result view",
            "Built-in Python container",
        )
    )
    csr_mb = geomean(layout["csr_host_mb"])
    coo_mb = geomean(layout["coo_host_mb"])
    comparisons.append(("CSR storage", "Host storage (MiB)", csr_mb, coo_mb, len(layout), "CSR", "COO"))
    csr_degree_ms = 1000 * geomean(layout["csr_degree_s"])
    coo_degree_ms = 1000 * geomean(layout["coo_degree_s"])
    comparisons.append(("CSR traversal", "Degree traversal (ms)", csr_degree_ms, coo_degree_ms, len(layout), "CSR", "COO"))

    for module, unit, full, ablated, cases, complete_label, reference_label in comparisons:
        rows.append(
            {
                "module": module,
                "axis": unit,
                "complete_value": full,
                "ablated_or_reference_value": ablated,
                "ratio": ablated / full,
                "cases": cases,
                "complete_label": complete_label,
                "reference_label": reference_label,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "ablation_actual_values.csv", index=False)
    controlled_lookup = controlled.set_index("comparison")
    negative = pd.DataFrame(
        [
            {
                "mechanism": "Device CSR cache",
                "geomean_ratio": float(
                    controlled_lookup.loc["device_csr_cache", "geomean_slowdown"]
                ),
                "cases": int(
                    controlled_lookup.loc["device_csr_cache", "dataset_order_pairs"]
                ),
                "main_figure": False,
                "reason": "no positive aggregate benefit",
            },
            {
                "mechanism": "Adaptive host policy",
                "geomean_ratio": float(
                    controlled_lookup.loc["adaptive_policy", "geomean_slowdown"]
                ),
                "cases": int(
                    controlled_lookup.loc["adaptive_policy", "dataset_order_pairs"]
                ),
                "main_figure": False,
                "reason": "effect is approximately neutral",
            },
            {
                "mechanism": "CSR PageRank kernel vs COO",
                "geomean_ratio": geomean(layout["pagerank_speedup_csr_vs_coo"]),
                "cases": int(layout["pagerank_speedup_csr_vs_coo"].notna().sum()),
                "main_figure": False,
                "reason": "CSR is not universally faster for this kernel",
            },
        ]
    )
    negative.to_csv(out_dir / "ablation_nonpositive_mechanisms.csv", index=False)

    fig = plt.figure(figsize=(7.2, 3.45))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.45, 1.0, 1.0, 1.0],
        hspace=0.58,
        wspace=0.55,
    )
    table_axis = fig.add_subplot(grid[:, 0])
    panel_axes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]
    colors = ["#78A9CF", "#9B8BD4", "#76B99B", "#E2B56B", "#6CA5A2"]
    for axis, row, color in zip(panel_axes, rows, colors):
        values = [row["complete_value"], row["ablated_or_reference_value"]]
        labels = [row["complete_label"], row["reference_label"]]
        bars = axis.bar(
            [0, 1],
            values,
            color=[color, "#D4DADF"],
            edgecolor="white",
            width=0.62,
        )
        annotate_bars(axis, bars, values, unit=row["axis"])
        axis.text(
            0.5,
            max(values) * 0.82,
            f"{row['ratio']:.2f}x",
            ha="center",
            fontsize=7.2,
            fontweight="bold",
            color="#334E5C",
        )
        axis.set_ylim(0, max(values) * 1.25)
        axis.set_ylabel(row["axis"], fontsize=6.2)
        axis.set_title(row["module"], loc="left", fontweight="bold", fontsize=7.3)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(labels, rotation=14, ha="right", fontsize=5.4)
        axis.grid(axis="y", color="#E5EBF0", linewidth=0.6)
    fig.add_subplot(grid[1, 3]).axis("off")
    cell_text = [
        [item["module"], f"{item['ratio']:.2f}x", str(item["cases"])]
        for item in rows
    ]
    table_axis.axis("off")
    display_table = table_axis.table(
        cellText=cell_text,
        colLabels=["Ablation", "Effect", "n"],
        loc="center",
        cellLoc="center",
        colWidths=[0.57, 0.26, 0.15],
    )
    display_table.auto_set_font_size(False)
    display_table.set_fontsize(5.8)
    display_table.scale(1.0, 1.34)
    for (row_index, _column), cell in display_table.get_celld().items():
        cell.set_linewidth(0.0)
        if row_index == 0:
            cell.set_facecolor("#EAF1F5")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F6F8FA")
    table_axis.set_title("Measured module effects", loc="left", fontweight="bold", fontsize=8.0, pad=0)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.96, bottom=0.13)
    save(fig, out_dir / "ablation_actual_time_memory_composite")
    return table


def load_scaling_csv(path):
    if path is None or not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def add_eggpu_scaling_rows(
    rows,
    csv_path,
    functions=None,
    exclude_datasets=(),
):
    data = load_scaling_csv(csv_path)
    if data.empty:
        return
    data = data[data["status"].eq("ok") & data["measurement"].eq("timing")]
    if functions is not None:
        data = data[data["function"].isin(functions)]
    if exclude_datasets:
        data = data[~data["dataset"].isin(set(exclude_datasets))]
    for _, row in data.iterrows():
        raw_path = csv_path.parent / "raw" / f"{row['dataset']}_{row['function']}_timing.json"
        best = None
        if raw_path.is_file():
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            best = (payload.get("steady_e2e") or {}).get("best")
        if best is None:
            best = row.get("steady_e2e_mean")
        rows.append(
            {
                "dataset": row["dataset"],
                "function": row["function"],
                "baseline": "EGGPU",
                "csr_entries": int(row["num_entries"]),
                "e2e_seconds": float(best),
                "e2e_std_seconds": (
                    float(row["steady_e2e_stdev"])
                    if pd.notna(row.get("steady_e2e_stdev")) else np.nan
                ),
                "graph_family": "R-MAT" if str(row.get("input_family")) == "controlled_rmat" else "real",
                "estimator": "best of five",
            }
        )


def add_nxcg_rows(rows, csv_path):
    if csv_path is None or not csv_path.is_file():
        return
    data = pd.read_csv(csv_path, low_memory=False)
    data = data[data["status"].eq("ok") & data["function"].isin(SCALING_FUNCTIONS)]
    for _, row in data.iterrows():
        value = row.get("e2e_mean_seconds")
        if pd.isna(value) and "e2e" in row and isinstance(row["e2e"], str):
            value = json.loads(row["e2e"]).get("mean")
        rows.append(
            {
                "dataset": row["dataset"],
                "function": row["function"],
                "baseline": "nx-cugraph",
                "csr_entries": int(float(row["num_entries"])),
                "e2e_seconds": float(value),
                "e2e_std_seconds": (
                    float(row["e2e_stdev_seconds"])
                    if pd.notna(row.get("e2e_stdev_seconds")) else np.nan
                ),
                "graph_family": "R-MAT" if str(row["dataset"]).startswith("R-MAT") else "real",
                "estimator": "mean of five",
            }
        )


def add_cpu_scaling_rows(rows, csv_path, baseline="igraph"):
    if csv_path is None or not csv_path.is_file():
        return
    data = pd.read_csv(csv_path, low_memory=False)
    required = {"dataset", "function", "baseline", "status", "num_entries"}
    if not required.issubset(data.columns):
        return
    data = data[
        data["baseline"].eq(baseline)
        & data["status"].eq("ok")
        & data["function"].isin(SCALING_FUNCTIONS)
    ]
    for _, row in data.iterrows():
        value = row.get("e2e_mean_seconds")
        if pd.isna(value):
            continue
        entries = row.get("num_entries")
        if pd.isna(entries):
            dataset = str(row["dataset"])
            manifest = (
                Path(__file__).resolve().parents[1]
                / "datasets"
                / "scaling"
                / "csr"
                / dataset
                / f"{dataset}.json"
            )
            if not manifest.is_file():
                raise RuntimeError(
                    f"missing CSR entry count and manifest for {dataset}: {csv_path}"
                )
            entries = json.loads(manifest.read_text(encoding="utf-8"))["num_entries"]
        rows.append(
            {
                "dataset": row["dataset"],
                "function": row["function"],
                "baseline": baseline,
                "csr_entries": int(float(entries)),
                "e2e_seconds": float(value),
                "e2e_std_seconds": (
                    float(row["e2e_stdev_seconds"])
                    if pd.notna(row.get("e2e_stdev_seconds")) else np.nan
                ),
                "graph_family": "R-MAT" if str(row["dataset"]).startswith("R-MAT") else "real",
                "estimator": "mean of five",
            }
        )


def add_sygraph_scaling_rows(rows, result_dir):
    if result_dir is None:
        return
    path = result_dir / "sygraph_bfs.csv"
    if not path.is_file():
        return
    data = pd.read_csv(path, low_memory=False)
    required = {"dataset", "function", "status", "validation", "num_entries"}
    if not required.issubset(data.columns):
        return
    data = data[
        data["function"].eq("BFS")
        & data["status"].eq("ok")
        & data["validation"].eq("pass")
    ]
    for _, row in data.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "function": "BFS",
                "baseline": "SYgraph",
                "csr_entries": int(float(row["num_entries"])),
                "e2e_seconds": float(row["e2e_mean"]),
                "e2e_std_seconds": (
                    float(row["e2e_stdev"])
                    if pd.notna(row.get("e2e_stdev")) else np.nan
                ),
                "graph_family": (
                    "R-MAT" if str(row["dataset"]).startswith("R-MAT") else "real"
                ),
                "estimator": "mean of five",
            }
        )


def scaling_figure(
    main_result,
    old_scale,
    old_rmat,
    old_nxcg,
    core_dir,
    followup_dir,
    sygraph_result,
    out_dir,
):
    _, _, _, paper_metrics, _, _ = story.load_main_views(main_result)
    rows = []
    main = paper_metrics["e2e"]
    for dataset in story.CROSS_LIBRARY_11:
        entries = story.DATASET_STRUCTURE[dataset]["csr_entries"]
        for function in SCALING_FUNCTIONS:
            for baseline in ("EGGPU", "igraph", "nx-cugraph"):
                hit = main[main["dataset"].eq(dataset) & main["function"].eq(function) & main["baseline"].eq(baseline)]
                if hit.empty:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "function": function,
                        "baseline": baseline,
                        "csr_entries": entries,
                        "e2e_seconds": float(hit["mean"].iloc[0]),
                        "e2e_std_seconds": (
                            float(hit["std"].iloc[0])
                            if "std" in hit.columns and pd.notna(hit["std"].iloc[0])
                            else np.nan
                        ),
                        "graph_family": "real",
                        "estimator": "best of five" if baseline == "EGGPU" else "mean of five",
                    }
                )
    add_eggpu_scaling_rows(
        rows,
        old_scale,
        exclude_datasets=("com-Orkut", "GAP-twitter"),
    )
    if not followup_dir:
        add_eggpu_scaling_rows(rows, old_rmat)
    if core_dir:
        add_eggpu_scaling_rows(rows, core_dir / "eggpu_large_matrix" / "scaling_all.csv")
        add_nxcg_rows(rows, core_dir / "nxcugraph_large_matrix" / "nxcugraph_large_matrix.csv")
        add_cpu_scaling_rows(rows, core_dir / "cpu_large_matrix" / "cpu_large_matrix.csv")
    if followup_dir:
        add_eggpu_scaling_rows(
            rows,
            followup_dir / "eggpu_rmat_four_functions" / "scaling_all.csv",
        )
        add_nxcg_rows(rows, followup_dir / "nxcugraph_rmat_four_functions" / "nxcugraph_large_matrix.csv")
        add_cpu_scaling_rows(
            rows,
            followup_dir / "igraph_rmat_four_functions" / "cpu_large_matrix.csv",
        )
    if not core_dir and not followup_dir:
        add_nxcg_rows(rows, old_nxcg)
    add_sygraph_scaling_rows(rows, sygraph_result)
    data = pd.DataFrame(rows).drop_duplicates(["dataset", "function", "baseline"], keep="last")
    data.to_csv(out_dir / "scaling_four_function_points.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.25))
    grid = fig.add_gridspec(2, 4, height_ratios=[4.5, 2.4], hspace=0.38, wspace=0.24)
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    for axis, function in zip(axes, SCALING_FUNCTIONS):
        subset = data[data["function"].eq(function)]
        scaling_baselines = ["igraph", "nx-cugraph"]
        if "SYgraph" in set(subset["baseline"].astype(str)):
            scaling_baselines.append("SYgraph")
        scaling_baselines.append("EGGPU")
        for baseline in scaling_baselines:
            for family in ("real", "R-MAT"):
                points = subset[subset["baseline"].eq(baseline) & subset["graph_family"].eq(family)]
                if points.empty:
                    continue
                marker = SYSTEM_MARKER[baseline]
                face = SYSTEM_COLOR[baseline] if family == "real" else "white"
                edge = SYSTEM_COLOR[baseline]
                sizes = 28 if baseline == "EGGPU" else 21
                axis.scatter(
                    points["csr_entries"],
                    points["e2e_seconds"],
                    s=sizes,
                    marker=marker,
                    facecolors=face,
                    edgecolors=edge,
                    linewidth=0.75,
                    alpha=0.96,
                    zorder=4 if baseline == "EGGPU" else 2,
                )
                for point in points.itertuples(index=False):
                    if not math.isfinite(float(point.e2e_std_seconds)):
                        continue
                    center = float(point.e2e_seconds)
                    deviation = float(point.e2e_std_seconds)
                    # A logarithmic axis cannot draw an error bar through zero.
                    lower = min(deviation, center * 0.92)
                    axis.errorbar(
                        float(point.csr_entries),
                        center,
                        yerr=np.array([[lower], [deviation]]),
                        fmt="none",
                        ecolor=SYSTEM_COLOR[baseline],
                        elinewidth=0.45,
                        capsize=1.2,
                        capthick=0.45,
                        alpha=0.62,
                        zorder=1,
                    )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(function, fontweight="bold", pad=3)
        axis.set_xlabel("CSR entries")
        axis.grid(which="major", color="#E1E8EE", linewidth=0.6)
        axis.grid(which="minor", color="#F0F3F6", linewidth=0.3)
    axes[0].set_ylabel("End-to-end time (s)")
    legend_baselines = ["igraph", "nx-cugraph"]
    if "SYgraph" in set(data["baseline"].astype(str)):
        legend_baselines.append("SYgraph")
    legend_baselines.append("EGGPU")
    baseline_handles = [
        Line2D(
            [0],
            [0],
            marker=SYSTEM_MARKER[baseline],
            color="none",
            markerfacecolor=SYSTEM_COLOR[baseline],
            markeredgecolor=SYSTEM_COLOR[baseline],
            markersize=4.4,
            label=baseline,
        )
        for baseline in legend_baselines
    ]
    family_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#607785", markeredgecolor="#607785", markersize=4.2, label="Real graph"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#607785", markersize=4.2, label="R-MAT"),
    ]
    fig.legend(
        handles=baseline_handles + family_handles,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        columnspacing=1.0,
        handletextpad=0.35,
    )

    # A single input-size ruler avoids repeating seventeen dense labels under
    # every panel while still placing each dataset at its measured CSR size.
    strip = fig.add_subplot(grid[1, :])
    strip.set_xscale("log")
    strip.set_xlim(min(data["csr_entries"]) * 0.8, max(data["csr_entries"]) * 1.25)
    strip.set_ylim(0, 1.28)
    strip.set_yticks([])
    strip.set_xlabel("Dataset position by CSR adjacency entries (log scale)", labelpad=2)
    strip.spines["left"].set_visible(False)
    strip.spines["right"].set_visible(False)
    strip.spines["top"].set_visible(False)
    strip.grid(axis="x", which="major", color="#E1E8EE", linewidth=0.55)
    strip.grid(axis="x", which="minor", color="#F0F3F6", linewidth=0.25)
    locations = data[["dataset", "csr_entries", "graph_family"]].drop_duplicates().sort_values("csr_entries")
    label_alias = {
        "p2p-Gnutella04": "Gnutella04",
        "ca-HepTh": "HepTh",
        "LastFM": "LastFM",
        "ca-CondMat": "CondMat",
        "ca-HepPh": "HepPh",
        "email-Enron": "Enron",
        "soc-Epinions1": "Epinions",
        "soc-Slashdot0811": "Slashdot",
        "web-NotreDame": "NotreDame",
        "com-youtube": "YouTube",
        "com-Orkut": "Orkut",
        "GAP-twitter": "Twitter",
        "R-MAT-S20-EF16": "R-MAT S20",
        "R-MAT-S22-EF16": "R-MAT S22",
        "R-MAT-S24-EF16": "R-MAT S24",
        "R-MAT-S26-EF16": "R-MAT S26",
    }
    lane_levels = (0.18, 0.32, 0.46, 0.60, 0.74, 0.88, 1.02, 1.16)
    lane_last_log = [-float("inf")] * len(lane_levels)
    for index, row in enumerate(locations.itertuples(index=False)):
        log_position = math.log10(float(row.csr_entries))
        # Greedily choose the lane whose previous label is farthest away.
        lane = max(range(len(lane_levels)), key=lambda item: log_position - lane_last_log[item])
        lane_last_log[lane] = log_position
        level = lane_levels[lane]
        color = "#315F7A" if row.graph_family == "R-MAT" else "#536B78"
        marker = "s" if row.graph_family == "R-MAT" else "o"
        strip.scatter([row.csr_entries], [0.08], s=18, marker=marker, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        strip.plot([row.csr_entries, row.csr_entries], [0.11, level - 0.035], color="#B9C6CE", linewidth=0.5)
        strip.text(
            row.csr_entries,
            level,
            label_alias.get(row.dataset, row.dataset),
            rotation=0,
            ha="center",
            va="center",
            fontsize=4.35,
            color=color,
        )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.09)
    save(fig, out_dir / "scaling_four_functions")
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-result", required=True, type=Path)
    parser.add_argument("--first-use", required=True, type=Path)
    parser.add_argument("--natural-workflow", required=True, type=Path)
    parser.add_argument("--ablation", required=True, type=Path)
    parser.add_argument("--old-scale-csv", required=True, type=Path)
    parser.add_argument("--old-rmat-csv", required=True, type=Path)
    parser.add_argument("--old-nxcugraph-csv", type=Path)
    parser.add_argument("--core-result", type=Path)
    parser.add_argument("--followup-result", type=Path)
    parser.add_argument("--cumulative-result", type=Path)
    parser.add_argument("--sygraph-result", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    category_overall_figure(out)
    first_use_figure(args.first_use.resolve(), out)
    ablation_figure(args.ablation.resolve(), out)
    cumulative = args.cumulative_result
    if cumulative is None and args.followup_result:
        cumulative = args.followup_result / "cumulative_workflow"
    if cumulative:
        cumulative_figure(cumulative.resolve(), out)
    scaling_figure(
        args.main_result.resolve(),
        args.old_scale_csv.resolve(),
        args.old_rmat_csv.resolve(),
        args.old_nxcugraph_csv.resolve() if args.old_nxcugraph_csv else None,
        args.core_result.resolve() if args.core_result else None,
        args.followup_result.resolve() if args.followup_result else None,
        args.sygraph_result.resolve() if args.sygraph_result else None,
        out,
    )
    # Case study is generated last by design and uses a z-prefixed filename.
    if cumulative:
        case_study_figure(cumulative.resolve(), out)
    (out / "README.md").write_text(
        "# EGGPU final experiment visuals v2\n\n"
        "All bar and cumulative-time axes start at zero and show measured time or "
        "memory. The category overview, per-function first-use detail, and "
        "four-function scaling panels use explicitly labelled logarithmic axes "
        "because their values span several orders of magnitude; zero is undefined "
        "on a logarithmic axis. Ratios appear only as annotations. The case study "
        "is intentionally emitted last.\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
