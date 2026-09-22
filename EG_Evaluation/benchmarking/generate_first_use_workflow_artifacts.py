#!/usr/bin/env python3
"""Generate paper artifacts for First-use and same-graph workflow supplements."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CATEGORY_ORDER = [
    "Centrality",
    "Connectivity",
    "Paths & Spanning Trees",
    "Structural Holes",
]

FUNCTION_CATEGORY = {
    "PageRank": "Centrality",
    "BC": "Centrality",
    "Closeness": "Centrality",
    "LCC": "Connectivity",
    "WCC": "Connectivity",
    "SCC": "Connectivity",
    "KCore": "Connectivity",
    "BFS": "Paths & Spanning Trees",
    "Dijkstra": "Paths & Spanning Trees",
    "BellmanFord": "Paths & Spanning Trees",
    "SSSP": "Paths & Spanning Trees",
    "MST": "Paths & Spanning Trees",
    "EffectiveSize": "Structural Holes",
    "Efficiency": "Structural Holes",
    "Constraint": "Structural Holes",
    "Hierarchy": "Structural Holes",
}

CATEGORY_COLORS = {
    "Centrality": "#9B8AD8",
    "Connectivity": "#72A9D4",
    "Paths & Spanning Trees": "#76B79B",
    "Structural Holes": "#DDA064",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def geomean(values) -> float:
    data = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    data = data[np.isfinite(data) & (data > 0)]
    if not len(data):
        return float("nan")
    return float(np.exp(np.mean(np.log(data))))


def first_use_summary(
    first_use_dir: Path, out_dir: Path, datasets: list[str] | None = None
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(first_use_dir / "first_use_vs_steady.csv")
    if datasets is not None:
        data = data[data["dataset"].isin(datasets)].copy()
    data["category"] = data["function"].map(FUNCTION_CATEGORY)
    data["ratio"] = pd.to_numeric(data["first_use_over_steady"], errors="coerce")
    rows = []
    for metric in ["e2e", "kernel"]:
        for category in CATEGORY_ORDER:
            subset = data[data["metric"].eq(metric) & data["category"].eq(category)]
            rows.append(
                {
                    "metric": metric,
                    "category": category,
                    "pairs": int(subset["ratio"].notna().sum()),
                    "geomean_first_use_over_steady": geomean(subset["ratio"]),
                    "median_first_use_over_steady": float(subset["ratio"].median()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "first_use_vs_steady_by_family.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 3.45), constrained_layout=True)
    y = np.arange(len(CATEGORY_ORDER))
    width = 0.34
    maximum = 1.0
    for pos, metric in enumerate(["e2e", "kernel"]):
        subset = summary[summary["metric"].eq(metric)].set_index("category")
        values = [float(subset.loc[category, "geomean_first_use_over_steady"]) for category in CATEGORY_ORDER]
        maximum = max(maximum, *values)
        bars = ax.barh(
            y + (pos - 0.5) * width,
            values,
            width,
            label=metric.upper(),
            color=[CATEGORY_COLORS[category] for category in CATEGORY_ORDER],
            alpha=0.92 if metric == "e2e" else 0.48,
            edgecolor="#FFFFFF",
            hatch="" if metric == "e2e" else "///",
        )
        for bar, value in zip(bars, values):
            ax.text(
                value + maximum * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}x",
                ha="left",
                va="center",
                fontsize=7,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(["Centrality", "Connectivity", "Paths & Spanning Trees", "Structural Holes"])
    ax.invert_yaxis()
    ax.set_xlim(0, maximum * 1.18)
    ax.set_xlabel("First-use / steady-state")
    ax.set_title("First-use cost by function family", fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor="#8FA8BB", edgecolor="white", label="E2E"),
            Patch(
                facecolor="#8FA8BB",
                alpha=0.48,
                edgecolor="white",
                hatch="///",
                label="Kernel",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="lower right",
    )
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E8EDF2", linewidth=0.6)
    fig.savefig(out_dir / "first_use_vs_steady_by_family.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "first_use_vs_steady_by_family.pdf", bbox_inches="tight")
    return summary


def summarize_beneficiaries(beneficiaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [
        (function, metric, group)
        for (function, metric), group in beneficiaries.groupby(["function", "metric"])
    ]
    groups.extend(
        (
            "ALL_BENEFICIARIES",
            metric,
            beneficiaries[beneficiaries["metric"].eq(metric)],
        )
        for metric in ("e2e", "kernel")
    )
    for function, metric, group in groups:
        first_total = float(group["first_use_mean_seconds"].sum())
        natural_total = float(group["natural_mean_seconds"].sum())
        saved = first_total - natural_total
        rows.append(
            {
                "function": function,
                "metric": metric,
                "comparison_pairs": len(group),
                "datasets": int(group["dataset"].nunique()),
                "call_positions": ",".join(
                    str(int(value)) for value in sorted(group["call_position"].unique())
                ),
                "first_use_total_seconds": first_total,
                "natural_total_seconds": natural_total,
                "time_saved_total_seconds": saved,
                "time_saved_percent_of_first_use_total": (
                    100.0 * saved / first_total if first_total > 0 else float("nan")
                ),
                "mean_time_saved_seconds": float(group["time_saved_seconds"].mean()),
                "median_time_saved_seconds": float(group["time_saved_seconds"].median()),
                "geomean_first_use_over_natural": geomean(
                    group["first_use_over_natural"]
                ),
                "natural_faster_pairs": int(group["natural_is_faster"].sum()),
                "natural_not_faster_pairs": int((~group["natural_is_faster"].astype(bool)).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["function", "metric"]).reset_index(drop=True)


def workflow_summary(
    natural_dir: Path, out_dir: Path, datasets: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    whole = pd.read_csv(natural_dir / "natural_workflow_vs_isolated.csv")
    whole["isolated_over_natural"] = pd.to_numeric(
        whole["isolated_over_natural"], errors="coerce"
    )
    beneficiary_pairs = pd.read_csv(
        natural_dir / "natural_workflow_reuse_beneficiaries.csv"
    )
    if datasets is not None:
        whole = whole[whole["dataset"].isin(datasets)].copy()
        beneficiary_pairs = beneficiary_pairs[
            beneficiary_pairs["dataset"].isin(datasets)
        ].copy()
    beneficiaries = summarize_beneficiaries(beneficiary_pairs)
    beneficiaries["geomean_first_use_over_natural"] = pd.to_numeric(
        beneficiaries["geomean_first_use_over_natural"], errors="coerce"
    )
    whole.to_csv(out_dir / "natural_workflow_vs_isolated.csv", index=False)
    beneficiaries.to_csv(out_dir / "natural_workflow_reuse_summary.csv", index=False)

    e2e = whole[whole["metric"].eq("e2e")].dropna(subset=["isolated_over_natural"])
    e2e = e2e.sort_values("isolated_over_natural", ascending=True)
    calls = beneficiaries[
        beneficiaries["function"].isin(["PageRank", "BFS"])
        & beneficiaries["metric"].isin(["e2e", "kernel"])
    ].copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 5.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.35, 1, 1]},
    )
    y = np.arange(len(e2e))
    values = e2e["isolated_over_natural"].to_numpy(float)
    axes[0].barh(y, values, color="#82B1D4", edgecolor="white", height=0.72)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(e2e["dataset"])
    axes[0].set_xlim(0, max(1.0, float(values.max())) * 1.14)
    axes[0].set_xlabel("Matched isolated calls / same-graph workflow")
    axes[0].set_title("Complete three-function workflow", fontweight="bold")

    call_order = ["PageRank", "BFS"]
    y2 = np.arange(len(call_order))
    for axis, metric, color in zip(
        axes[1:], ["e2e", "kernel"], ["#6EA8D7", "#78B79D"]
    ):
        subset = calls[calls["metric"].eq(metric)].set_index("function")
        call_values = [
            float(subset.loc[function, "geomean_first_use_over_natural"])
            for function in call_order
        ]
        maximum = max(1.0, *call_values)
        bars = axis.barh(
            y2,
            call_values,
            0.56,
            color=color,
            edgecolor="white",
        )
        for bar, value in zip(bars, call_values):
            axis.text(
                value + maximum * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}x",
                ha="left",
                va="center",
                fontsize=7,
            )
        axis.set_yticks(y2)
        axis.set_yticklabels(["PageRank, call 2", "BFS, call 3"])
        axis.invert_yaxis()
        axis.set_xlim(0, maximum * 1.22)
        axis.set_xlabel("Matched first-use / reused call")
        axis.set_title(
            "Reuse-beneficiary E2E" if metric == "e2e" else "Reuse-beneficiary kernel",
            fontweight="bold",
        )
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E8EDF2", linewidth=0.6)
    fig.savefig(out_dir / "natural_workflow_reuse.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "natural_workflow_reuse.pdf", bbox_inches="tight")
    return whole, beneficiaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-use-dir", required=True, type=Path)
    parser.add_argument("--natural-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated ordered dataset whitelist.",
    )
    args = parser.parse_args()

    setup_style()
    first_use_dir = args.first_use_dir.resolve()
    natural_dir = args.natural_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = (
        [item.strip() for item in args.datasets.split(",") if item.strip()]
        if args.datasets.strip()
        else None
    )
    first = first_use_summary(first_use_dir, out_dir, datasets)
    whole, beneficiaries = workflow_summary(natural_dir, out_dir, datasets)

    lines = [
        "# First-use and Same-Graph Workflow Artifacts",
        "",
        "- Primary estimator: arithmetic mean over five independent processes.",
        "- Error statistic retained in source CSVs: sample standard deviation.",
        "- First-use runs use no warmup and start with no reusable graph state.",
        "- The natural workflow is WCC -> PageRank -> BFS on one unchanged graph object.",
        "- WCC pays initialization cost; PageRank and BFS are direct reuse beneficiaries.",
        "- Each isolated control is measured in the same run, on the same GPU, with the same graph semantics and parameters.",
        "- Case order is cyclically rotated across repeats to control temporal drift.",
        "- The isolated control sums the three matched First-use calls.",
        "",
        "## Files",
        "",
        "- `first_use_vs_steady_by_family.pdf`",
        "- `natural_workflow_reuse.pdf`",
        "- `first_use_vs_steady_by_family.csv`",
        "- `natural_workflow_vs_isolated.csv`",
        "- `natural_workflow_reuse_summary.csv`",
        "",
        f"First-use summary rows: {len(first)}.",
        f"Whole-workflow comparison rows: {len(whole)}.",
        f"Reuse-beneficiary summary rows: {len(beneficiaries)}.",
        f"Dataset filter: {', '.join(datasets) if datasets else 'all available datasets'}.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote supplement paper artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
