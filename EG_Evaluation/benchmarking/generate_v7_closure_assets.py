#!/usr/bin/env python3
"""Generate the final Intro scaling and evaluation-closure assets.

All values come from the frozen V7 ledger and its audited scaling/memory
derivatives.  This script performs no benchmark repair and no estimator
substitution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COLORS = {
    "EGGPU": "#3D87B3",
    "Gunrock": "#8D99A6",
    "igraph": "#75B798",
}
MARKERS = {"EGGPU": "o", "Gunrock": "X", "igraph": "^"}
VALIDATION_OK = {"pass", "reference", "external_reference_pass", "sampled_pass"}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


def geomean(values) -> float:
    data = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    data = data[np.isfinite(data) & (data > 0)]
    return float(np.exp(np.log(data).mean())) if len(data) else float("nan")


def generate_intro_scaling(source: Path, output: Path) -> None:
    real = pd.read_csv(source / "intro_scaling_real_pagerank.csv", low_memory=False)
    real = real[
        real["baseline"].isin(COLORS)
        & real["execution_status"].eq("ok")
        & real["validation_status"].isin(VALIDATION_OK)
    ].copy()
    real["latency_seconds"] = pd.to_numeric(real["e2e_paper_seconds"], errors="coerce")
    real["std_seconds"] = pd.to_numeric(real["e2e_std_seconds"], errors="coerce")
    real["series"] = "Real"
    real = real[
        np.isfinite(real["latency_seconds"])
        & real["latency_seconds"].gt(0)
        & pd.to_numeric(real["csr_entries"], errors="coerce").gt(0)
    ]

    controlled = pd.read_csv(source / "scaling_four_function_points.csv")
    controlled = controlled[
        controlled["function"].eq("PageRank")
        & controlled["graph_family"].eq("R-MAT")
        & controlled["baseline"].isin({"EGGPU", "igraph"})
    ].copy()
    controlled = controlled.rename(
        columns={"e2e_seconds": "latency_seconds", "e2e_std_seconds": "std_seconds"}
    )
    controlled["series"] = "R-MAT"

    columns = [
        "dataset",
        "baseline",
        "csr_entries",
        "latency_seconds",
        "std_seconds",
        "series",
        "estimator",
    ]
    points = pd.concat([real[columns], controlled[columns]], ignore_index=True)
    points.to_csv(output / "intro_pagerank_scaling_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.45, 2.30))
    for baseline in ("igraph", "Gunrock", "EGGPU"):
        part = points[(points["baseline"].eq(baseline)) & points["series"].eq("Real")]
        part = part.sort_values("csr_entries")
        ax.scatter(
            part["csr_entries"],
            part["latency_seconds"],
            s=25 if baseline == "EGGPU" else 20,
            marker=MARKERS[baseline],
            color=COLORS[baseline],
            edgecolor="#23526B" if baseline == "EGGPU" else "white",
            linewidth=0.65,
            zorder=4,
            label=baseline,
        )

    for baseline in ("igraph", "EGGPU"):
        part = points[(points["baseline"].eq(baseline)) & points["series"].eq("R-MAT")]
        part = part.sort_values("csr_entries")
        ax.plot(
            part["csr_entries"],
            part["latency_seconds"],
            color=COLORS[baseline],
            linestyle=(0, (3.0, 2.2)),
            linewidth=1.1,
            alpha=0.92,
            zorder=2,
        )
        ax.scatter(
            part["csr_entries"],
            part["latency_seconds"],
            s=27,
            marker=MARKERS[baseline],
            facecolor="white",
            edgecolor=COLORS[baseline],
            linewidth=1.05,
            zorder=5,
        )

    label_offsets = {
        ("EGGPU", "com-Orkut"): (5, 5, "left"),
        ("EGGPU", "GAP-twitter"): (-4, 6, "right"),
    }
    for (baseline, dataset), (dx, dy, ha) in label_offsets.items():
        hit = points[
            points["baseline"].eq(baseline)
            & points["dataset"].eq(dataset)
            & points["series"].eq("Real")
        ]
        if len(hit):
            row = hit.iloc[0]
            label = "Twitter" if dataset == "GAP-twitter" else "Orkut"
            ax.annotate(
                label,
                (row["csr_entries"], row["latency_seconds"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="bottom" if dy >= 0 else "top",
                fontsize=5.5,
                color=COLORS[baseline],
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Adjacency entries")
    ax.set_ylabel("Post-construction latency (s)")
    ax.grid(which="major", color="#DCE5EC", linewidth=0.55)
    ax.grid(which="minor", color="#EEF2F5", linewidth=0.25)
    ax.set_xlim(3.1e4, 2.4e9)
    ax.set_ylim(4e-4, 1.1e2)
    handles = [
        Line2D([], [], marker=MARKERS[name], linestyle="none", color=COLORS[name],
               markeredgecolor="#23526B" if name == "EGGPU" else "white",
               markersize=5.2, label=name)
        for name in ("EGGPU", "Gunrock", "igraph")
    ]
    handles.extend(
        [
            Line2D([], [], color="#667783", linewidth=1.0, linestyle="none",
                   marker="o", markersize=4.2, label="Real graph"),
            Line2D([], [], color="#667783", linewidth=1.0, linestyle=(0, (3, 2)),
                   marker="o", markerfacecolor="white", markersize=4.2,
                   label="R-MAT"),
        ]
    )
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper left",
              columnspacing=0.65, handletextpad=0.3, borderaxespad=0.2)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.20, top=0.97)
    save(fig, output / "intro_pagerank_scaling")


def generate_resource_table(source: Path, output: Path) -> None:
    memory = json.loads((source / "memory_analysis" / "memory_summary.json").read_text())
    gpu = {
        row["baseline"]: row
        for row in memory["pairwise_system_summaries"]
        if row["boundary"] == "GPU peak memory"
    }
    best = next(
        row for row in memory["best_competitor_summaries"]
        if row["boundary"] == "GPU peak memory"
    )
    numeric = json.loads((source / "final_13_numeric_summary.json").read_text())
    ledger = pd.read_csv(source / "final_13_cell_outcome_ledger.csv", low_memory=False)
    gap = ledger[
        ledger["dataset"].eq("GAP-twitter")
        & ledger["execution_status"].eq("ok")
        & ledger["validation_status"].isin(VALIDATION_OK)
    ]
    capacity = gap.groupby("baseline").size().to_dict()
    rows = [
        {
            "comparison": "vs. nx-cugraph",
            "common": gpu["nx-cugraph"]["common_pairs"],
            "lower": gpu["nx-cugraph"]["eggpu_lower_pairs"],
            "ratio": gpu["nx-cugraph"]["geomean_baseline_over_eggpu"],
        },
        {
            "comparison": "vs. Gunrock",
            "common": gpu["Gunrock"]["common_pairs"],
            "lower": gpu["Gunrock"]["eggpu_lower_pairs"],
            "ratio": gpu["Gunrock"]["geomean_baseline_over_eggpu"],
        },
        {
            "comparison": "vs. pairwise best GPU",
            "common": best["common_pairs"],
            "lower": best["eggpu_lower_pairs"],
            "ratio": best["geomean_best_competitor_over_eggpu"],
        },
    ]
    pd.DataFrame(rows).to_csv(output / "resource_efficiency_summary.csv", index=False)
    lines = [
        "% Generated from the frozen V7 memory audit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Device-memory efficiency and billion-edge capacity. Ratios above one indicate lower peak device memory for EGGPU on correctness-aligned common workloads.}",
        r"\label{tab:resource-efficiency}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"GPU comparison & Common & EGGPU lower & Memory ratio \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['comparison']} & {row['common']} & {row['lower']} & {row['ratio']:.2f}$\\times$ \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            rf"GAP-twitter validated functions & 16 EGGPU & {capacity.get('nx-cugraph', 0)} nx-cugraph & {capacity.get('GraphScope', 0)} GraphScope \\",
            rf"Maximum EGGPU device memory & \multicolumn{{3}}{{c}}{{{numeric['eggpu_max_gpu_peak_mb'] / 1024:.2f} GiB}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (output / "paper_table_resource_efficiency.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_main_summary_table(source: Path, output: Path) -> None:
    summary = pd.read_csv(source / "paper_table_category_13_summary.csv")
    summary["metric"] = summary["metric"].str.upper()
    summary["category"] = summary["category"].replace(
        {"Paths & Spanning Trees": "Path & Spanning"}
    )
    rows = summary[summary["metric"].eq("E2E")]
    lines = [
        "% Generated from the frozen V7 category summary.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{End-to-end coverage and pairwise speedup by function family. Speedup uses correctness-validated common workloads.}",
        r"\label{tab:overall-family-summary}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Family & Fastest/total & Wins/common & Speedup \\",
        r"\midrule",
    ]
    for row in rows.itertuples(index=False):
        family = str(row.category).replace("&", r"\&")
        fastest = f"{int(row.fastest_tied_or_only)}/{int(row.eggpu_successful_workloads)}"
        only = int(row.only_validated)
        wins = f"{int(row.competitive_wins)}/{int(row.competitive_pairs)}"
        speedup = float(row.speedup_over_best_competitor)
        lines.append(f"{family} & {fastest} ({only} only) & {wins} & {speedup:.2f}$\\times$ \\\\ ")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output / "paper_table_overall_family_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-assets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_assets.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup_style()
    generate_intro_scaling(source, output)
    generate_resource_table(source, output)
    generate_main_summary_table(source, output)
    manifest = {
        "source_assets": str(source),
        "frozen_ledger": str(source / "final_13_cell_outcome_ledger.csv"),
        "assets": sorted(path.name for path in output.iterdir()),
    }
    (output / "closure_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
