#!/usr/bin/env python3
"""Generate paper-style result visualizations from one full-eval directory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


FUNCTION_ORDER = [
    "PageRank",
    "BC",
    "Closeness",
    "LCC",
    "WCC",
    "SCC",
    "KCore",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "MST",
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
]

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

BASELINE_ORDER = [
    "EGGPU",
    "easygraph-cpu",
    "easygraph-cpp",
    "networkx",
    "igraph",
    "nx-cugraph",
    "Gunrock",
]

BASELINE_LABEL = {
    "EGGPU": "EGGPU",
    "easygraph-cpu": "EasyGraph CPU",
    "easygraph-cpp": "EasyGraph C++",
    "networkx": "NetworkX",
    "igraph": "igraph",
    "nx-cugraph": "nx-cugraph",
    "Gunrock": "Gunrock",
}

BASELINE_COLOR = {
    "EGGPU": "#4C9BD6",
    "easygraph-cpu": "#B9A394",
    "easygraph-cpp": "#D9A441",
    "networkx": "#B7797C",
    "igraph": "#69A88F",
    "nx-cugraph": "#8B79C9",
    "Gunrock": "#8A8F98",
}

FILTER_LABEL = {
    "full": "All",
    "nodes>=10000": "Nodes >= 10k",
    "gpu-friendly": "GPU-friendly",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def geomean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def read_dataset_order(result_dir: Path, fallback_df: pd.DataFrame) -> list[str]:
    stats_path = result_dir / "dataset_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        return [row["name"] for row in stats if "name" in row]
    return sorted(fallback_df["dataset"].dropna().unique().tolist())


def load_metric(result_dir: Path, metric: str) -> pd.DataFrame:
    path = result_dir / f"results_{metric}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    df["category"] = df["function"].map(FUNCTION_CATEGORY)
    return df[df["category"].notna()].copy()


def category_runtime(metric_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ok = metric_df[(metric_df["status"] == "ok") & metric_df["seconds"].notna()].copy()
    rows = []
    for category in CATEGORY_ORDER:
        for baseline in BASELINE_ORDER:
            sub = ok[(ok["category"] == category) & (ok["baseline"] == baseline)]
            if sub.empty:
                continue
            rows.append(
                {
                    "metric": metric,
                    "category": category,
                    "baseline": baseline,
                    "baseline_label": BASELINE_LABEL.get(baseline, baseline),
                    "ok_pairs": int(len(sub)),
                    "geomean_seconds": geomean(sub["seconds"]),
                    "mean_seconds": float(sub["seconds"].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_category_runtime(agg: pd.DataFrame, metric: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.35), constrained_layout=True)
    present = [b for b in BASELINE_ORDER if b in set(agg["baseline"])]
    x = np.arange(len(CATEGORY_ORDER), dtype=float)
    width = min(0.11, 0.78 / max(1, len(present)))

    for i, baseline in enumerate(present):
        vals = []
        for category in CATEGORY_ORDER:
            hit = agg[(agg["category"] == category) & (agg["baseline"] == baseline)]
            vals.append(float(hit["geomean_seconds"].iloc[0]) if len(hit) else np.nan)
        offset = (i - (len(present) - 1) / 2) * width
        edge = "#1F4E79" if baseline == "EGGPU" else "#F8FAFC"
        linewidth = 1.2 if baseline == "EGGPU" else 0.45
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=BASELINE_LABEL.get(baseline, baseline),
            color=BASELINE_COLOR.get(baseline, "#9CA3AF"),
            edgecolor=edge,
            linewidth=linewidth,
            alpha=0.96,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Geometric mean time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_ORDER)
    ax.set_title(f"{metric.upper()} Runtime by Function Category", fontweight="bold", pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="#D8DEE8", linewidth=0.7, alpha=0.75)
    ax.grid(axis="y", which="minor", color="#EEF1F5", linewidth=0.45, alpha=0.55)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False)
    ax.margins(x=0.035)

    stem = f"category_geomean_{metric}"
    fig.savefig(out_dir / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def speedup_matrix(metric_df: pd.DataFrame, dataset_order: list[str]) -> pd.DataFrame:
    ok = metric_df[(metric_df["status"] == "ok") & metric_df["seconds"].notna()].copy()
    matrix = pd.DataFrame(index=FUNCTION_ORDER, columns=dataset_order, dtype=float)
    rows = []
    for function in FUNCTION_ORDER:
        for dataset in dataset_order:
            pair = ok[(ok["dataset"] == dataset) & (ok["function"] == function)]
            eggpu = pair[pair["baseline"] == "EGGPU"]
            others = pair[pair["baseline"] != "EGGPU"]
            if eggpu.empty or others.empty:
                ratio = float("nan")
                best_baseline = ""
                best_seconds = float("nan")
                eggpu_seconds = float("nan") if eggpu.empty else float(eggpu["seconds"].iloc[0])
            else:
                eggpu_seconds = float(eggpu["seconds"].iloc[0])
                best = others.sort_values("seconds").iloc[0]
                best_seconds = float(best["seconds"])
                best_baseline = str(best["baseline"])
                ratio = best_seconds / eggpu_seconds if eggpu_seconds > 0 else float("nan")
            matrix.loc[function, dataset] = ratio
            rows.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "category": FUNCTION_CATEGORY.get(function, ""),
                    "eggpu_seconds": eggpu_seconds,
                    "best_non_eggpu_baseline": best_baseline,
                    "best_non_eggpu_seconds": best_seconds,
                    "speedup_vs_best_non_eggpu": ratio,
                }
            )
    return matrix, pd.DataFrame(rows)


def format_speedup(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    if value >= 100:
        return ">99x"
    if value >= 10:
        return f"{value:.0f}x"
    if value >= 1:
        return f"{value:.1f}x"
    if value >= 0.1:
        return f"{value:.2f}x"
    return "<0.1x"


def plot_speedup_heatmap(matrix: pd.DataFrame, metric: str, out_dir: Path) -> None:
    ratios = matrix.to_numpy(dtype=float)
    scores = np.full_like(ratios, np.nan, dtype=float)
    finite_mask = np.isfinite(ratios) & (ratios > 0)
    scores[finite_mask] = np.log2(ratios[finite_mask])
    cmap = LinearSegmentedColormap.from_list(
        "eggpu_speedup",
        ["#B85C5C", "#F5F1EA", "#4F9A84", "#246B8F"],
        N=256,
    )
    cmap.set_bad("#ECEFF3")
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=6.0)

    fig_width = max(10.8, 0.56 * len(matrix.columns) + 2.1)
    fig_height = max(6.2, 0.34 * len(matrix.index) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    im = ax.imshow(np.ma.masked_invalid(scores), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=34, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(f"EGGPU {metric.upper()} Speedup over Best Baseline", fontweight="bold", pad=8)

    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    for boundary in [3, 7, 12]:
        ax.axhline(boundary - 0.5, color="#2F3A46", linewidth=1.0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ratio = float(matrix.iloc[i, j]) if pd.notna(matrix.iloc[i, j]) else float("nan")
            score = math.log2(ratio) if math.isfinite(ratio) and ratio > 0 else 0.0
            color = "white" if score > 3.3 or score < -1.25 else "#172033"
            ax.text(
                j,
                i,
                format_speedup(ratio),
                ha="center",
                va="center",
                fontsize=6.3,
                color=color,
                fontweight="bold" if math.isfinite(ratio) and ratio >= 1 else "normal",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.018)
    tick_ratios = [0.25, 0.5, 1, 2, 4, 8, 16, 64]
    cbar.set_ticks([math.log2(v) for v in tick_ratios])
    cbar.set_ticklabels([f"{v:g}x" for v in tick_ratios])
    cbar.set_label("Speedup")

    stem = f"heatmap_{metric}_speedup"
    fig.savefig(out_dir / f"{stem}.png", dpi=280, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sota_overall(result_dir: Path, out_dir: Path) -> None:
    path = result_dir / "eggpu_final_sota_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["filter"].isin(FILTER_LABEL)].copy()
    df["filter_label"] = df["filter"].map(FILTER_LABEL)
    filter_order = ["full", "nodes>=10000", "gpu-friendly"]

    fig, ax = plt.subplots(figsize=(7.2, 4.15), constrained_layout=True)
    x = np.arange(len(filter_order), dtype=float)
    width = 0.32
    colors = {"e2e": "#6EA8D7", "kernel": "#73B79B"}
    for i, metric in enumerate(["e2e", "kernel"]):
        vals = []
        labels = []
        for flt in filter_order:
            hit = df[(df["filter"] == flt) & (df["metric"] == metric)]
            val = float(hit["sota_pct"].iloc[0]) if len(hit) else np.nan
            vals.append(val)
            if len(hit):
                labels.append(f"{int(hit['sota_pairs'].iloc[0])}/{int(hit['total_pairs'].iloc[0])}")
            else:
                labels.append("")
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=metric.upper(),
            color=colors[metric],
            edgecolor="#F8FAFC",
            linewidth=0.55,
        )
        for bar, val, pair_label in zip(bars, vals, labels):
            if not math.isfinite(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.2,
                f"{val:.1f}%\n{pair_label}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1F2937",
            )

    ax.set_ylim(0, 108)
    ax.set_ylabel("SOTA coverage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([FILTER_LABEL[f] for f in filter_order])
    ax.set_title("SOTA Coverage", fontweight="bold", pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.75)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    fig.savefig(out_dir / "sota_coverage_overall.png", dpi=260, bbox_inches="tight")
    fig.savefig(out_dir / "sota_coverage_overall.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sota_by_category(result_dir: Path, out_dir: Path) -> None:
    path = result_dir / "eggpu_final_sota_details.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["filter"].eq("full")].copy()
    if df.empty:
        return
    rename = {
        "Centrality/Core": "Centrality",
        "Connectivity/Traversal": "Connectivity",
        "Path/Spanning": "Paths & Spanning Trees",
        "Paths & Spanning Trees": "Paths & Spanning Trees",
        "Structural Holes": "Structural Holes",
    }
    df["category_label"] = df["category"].map(rename).fillna(df["category"])
    rows = []
    for metric in ["e2e", "kernel"]:
        subm = df[df["metric"] == metric]
        for category in CATEGORY_ORDER:
            sub = subm[subm["category_label"] == category]
            if sub.empty:
                continue
            sota = int(sub["is_sota"].astype(bool).sum())
            total = int(len(sub))
            rows.append(
                {
                    "metric": metric,
                    "category": category,
                    "sota_pairs": sota,
                    "total_pairs": total,
                    "sota_pct": 100.0 * sota / total if total else float("nan"),
                }
            )
    agg = pd.DataFrame(rows)
    agg.to_csv(out_dir / "sota_coverage_by_category.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
    x = np.arange(len(CATEGORY_ORDER), dtype=float)
    width = 0.32
    colors = {"e2e": "#6EA8D7", "kernel": "#73B79B"}
    for i, metric in enumerate(["e2e", "kernel"]):
        vals = []
        labels = []
        for category in CATEGORY_ORDER:
            hit = agg[(agg["category"] == category) & (agg["metric"] == metric)]
            val = float(hit["sota_pct"].iloc[0]) if len(hit) else np.nan
            vals.append(val)
            if len(hit):
                labels.append(f"{int(hit['sota_pairs'].iloc[0])}/{int(hit['total_pairs'].iloc[0])}")
            else:
                labels.append("")
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            label=metric.upper(),
            color=colors[metric],
            edgecolor="#F8FAFC",
            linewidth=0.55,
        )
        for bar, val, pair_label in zip(bars, vals, labels):
            if not math.isfinite(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.2,
                f"{val:.1f}%\n{pair_label}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1F2937",
            )

    ax.set_ylim(0, 108)
    ax.set_ylabel("SOTA coverage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_ORDER)
    ax.set_title("SOTA Coverage by Function Category", fontweight="bold", pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.75)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    fig.savefig(out_dir / "sota_coverage_by_category.png", dpi=260, bbox_inches="tight")
    fig.savefig(out_dir / "sota_coverage_by_category.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(result_dir: Path, out_dir: Path, dataset_order: list[str]) -> None:
    lines = [
        "# Result Showcase Visuals",
        "",
        f"Source result: `{result_dir}`",
        "",
        "Generated figures:",
        "",
        "- `category_geomean_build.png/.pdf`",
        "- `category_geomean_kernel.png/.pdf`",
        "- `category_geomean_e2e.png/.pdf`",
        "- `heatmap_e2e_speedup.png/.pdf`",
        "- `heatmap_kernel_speedup.png/.pdf`",
        "- `sota_coverage_overall.png/.pdf`",
        "- `sota_coverage_by_category.png/.pdf`",
        "",
        "Function categories:",
        "",
        "- Centrality: PageRank, BC, Closeness",
        "- Connectivity: LCC, WCC, SCC, KCore",
        "- Paths & Spanning Trees: BFS, Dijkstra, BellmanFord, SSSP, MST",
        "- Structural Holes: EffectiveSize, Efficiency, Constraint, Hierarchy",
        "",
        "Dataset order:",
        "",
        ", ".join(dataset_order),
        "",
    ]
    (out_dir / "SHOWCASE_VISUALS.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    setup_style()
    result_dir = args.result_dir.resolve()
    out_dir = (args.out_dir or (result_dir / "paper_showcase_610")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {metric: load_metric(result_dir, metric) for metric in ["build", "kernel", "e2e"]}
    dataset_order = read_dataset_order(result_dir, metrics["e2e"])

    all_aggs = []
    for metric, df in metrics.items():
        agg = category_runtime(df, metric)
        all_aggs.append(agg)
        agg.to_csv(out_dir / f"category_geomean_{metric}.csv", index=False)
        plot_category_runtime(agg, metric, out_dir)

    pd.concat(all_aggs, ignore_index=True).to_csv(out_dir / "category_geomean_all_metrics.csv", index=False)

    for metric in ["e2e", "kernel"]:
        matrix, detail = speedup_matrix(metrics[metric], dataset_order)
        matrix.to_csv(out_dir / f"heatmap_{metric}_speedup_matrix.csv")
        detail.to_csv(out_dir / f"heatmap_{metric}_speedup_detail.csv", index=False)
        plot_speedup_heatmap(matrix, metric, out_dir)

    plot_sota_overall(result_dir, out_dir)
    plot_sota_by_category(result_dir, out_dir)
    write_summary(result_dir, out_dir, dataset_order)
    print(f"Wrote showcase visuals to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
