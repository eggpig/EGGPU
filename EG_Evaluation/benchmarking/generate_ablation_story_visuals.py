#!/usr/bin/env python3
"""Generate paper-style ablation and drill-down visuals from existing results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CATEGORY_ORDER = ["Centrality", "Connectivity", "Paths & Spanning Trees", "Structural Holes"]
CATEGORY_COLOR = {
    "Centrality": "#4C9BD6",
    "Connectivity": "#91C7A7",
    "Paths & Spanning Trees": "#D9A441",
    "Structural Holes": "#B7797C",
}
CATEGORY_RENAME = {
    "Centrality/Core": "Centrality",
    "Connectivity/Traversal": "Connectivity",
    "Path/Spanning": "Paths & Spanning Trees",
    "Paths & Spanning Trees": "Paths & Spanning Trees",
    "Structural Holes": "Structural Holes",
}
FUNCTION_ORDER = {
    "Centrality": ["PageRank", "BC", "Closeness"],
    "Connectivity": ["LCC", "WCC", "SCC", "KCore"],
    "Paths & Spanning Trees": ["BFS", "Dijkstra", "BellmanFord", "SSSP", "MST"],
    "Structural Holes": ["EffectiveSize", "Efficiency", "Constraint", "Hierarchy"],
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


def geomean(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def load_speedups(result_dir: Path, metric: str = "e2e") -> pd.DataFrame:
    df = pd.read_csv(result_dir / "eggpu_final_sota_details.csv")
    df = df[(df["metric"] == metric) & (df["filter"] == "full")].copy()
    df["category_label"] = df["category"].map(CATEGORY_RENAME).fillna(df["category"])
    df["speedup"] = 1.0 / pd.to_numeric(df["ratio_to_best"], errors="coerce")
    df = df[df["category_label"].isin(CATEGORY_ORDER)]
    return df


def plot_category_drilldown(result_dir: Path, out_dir: Path) -> pd.DataFrame:
    df = load_speedups(result_dir, "e2e")
    rows = []
    for category in CATEGORY_ORDER:
        subc = df[df["category_label"] == category]
        rows.append(
            {
                "level": "category",
                "category": category,
                "function": "",
                "pairs": int(len(subc)),
                "geomean_speedup": geomean(subc["speedup"]),
                "sota_pct": 100.0 * float(subc["is_sota"].astype(bool).mean()) if len(subc) else float("nan"),
            }
        )
        for fn in FUNCTION_ORDER[category]:
            sub = subc[subc["function"] == fn]
            if sub.empty:
                continue
            rows.append(
                {
                    "level": "function",
                    "category": category,
                    "function": fn,
                    "pairs": int(len(sub)),
                    "geomean_speedup": geomean(sub["speedup"]),
                    "sota_pct": 100.0 * float(sub["is_sota"].astype(bool).mean()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "category_function_drilldown_e2e.csv", index=False)

    fig = plt.figure(figsize=(12.2, 6.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.35])
    ax_main = fig.add_subplot(gs[0, :])
    cat = summary[summary["level"] == "category"].copy()
    x = np.arange(len(cat), dtype=float)
    vals = cat["geomean_speedup"].to_numpy(dtype=float)
    bars = ax_main.bar(
        x,
        vals,
        width=0.62,
        color=[CATEGORY_COLOR[c] for c in cat["category"]],
        edgecolor="#F8FAFC",
        linewidth=0.7,
    )
    for bar, row in zip(bars, cat.itertuples()):
        ax_main.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.035,
            f"{row.geomean_speedup:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1F2937",
            fontweight="bold",
        )
    ax_main.set_ylim(0, max(vals) * 1.25)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(cat["category"])
    ax_main.set_ylabel("Best-normalized performance")
    ax_main.set_title("EGGPU Relative E2E Performance by Function Category", fontweight="bold", pad=8)
    ax_main.set_axisbelow(True)
    ax_main.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.75)

    for idx, category in enumerate(CATEGORY_ORDER):
        ax = fig.add_subplot(gs[1, idx])
        sub = summary[(summary["level"] == "function") & (summary["category"] == category)].copy()
        order = [f for f in FUNCTION_ORDER[category] if f in set(sub["function"])]
        sub["function"] = pd.Categorical(sub["function"], categories=order, ordered=True)
        sub = sub.sort_values("function")
        y = np.arange(len(sub), dtype=float)
        vals = sub["geomean_speedup"].to_numpy(dtype=float)
        ax.barh(
            y,
            vals,
            height=0.56,
            color=CATEGORY_COLOR[category],
            edgecolor="#F8FAFC",
            linewidth=0.6,
            alpha=0.96,
        )
        for yi, row in zip(y, sub.itertuples()):
            ax.text(
                row.geomean_speedup + max(vals) * 0.035,
                yi,
                f"{row.geomean_speedup:.2f}",
                va="center",
                ha="left",
                fontsize=7.6,
                color="#1F2937",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(sub["function"])
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.36 if len(vals) else 1.0)
        ax.set_title(category, fontweight="bold", color=CATEGORY_COLOR[category], pad=6)
        ax.set_xlabel("Best-normalized performance")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E1E6ED", linewidth=0.65, alpha=0.8)
    fig.savefig(out_dir / "category_function_drilldown_e2e.png", dpi=280, bbox_inches="tight")
    fig.savefig(out_dir / "category_function_drilldown_e2e.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def load_eggpu_times(result_dir: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(result_dir / f"results_{metric}.csv")
    df = df[(df["baseline"] == "EGGPU") & (df["status"] == "ok")].copy()
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    df = df[df["seconds"].notna()]
    function_to_category = {
        fn: category
        for category, funcs in FUNCTION_ORDER.items()
        for fn in funcs
    }
    df["category"] = df["function"].map(function_to_category)
    return df[df["category"].notna()].copy()


def plot_category_function_time_drilldown(result_dir: Path, out_dir: Path) -> pd.DataFrame:
    e2e = load_eggpu_times(result_dir, "e2e")
    kernel = load_eggpu_times(result_dir, "kernel")
    rows = []
    for metric, df in [("E2E", e2e), ("Kernel", kernel)]:
        for category in CATEGORY_ORDER:
            subc = df[df["category"] == category]
            rows.append(
                {
                    "level": "category",
                    "metric": metric,
                    "category": category,
                    "function": "",
                    "pairs": int(len(subc)),
                    "geomean_seconds": geomean(subc["seconds"]),
                }
            )
            for fn in FUNCTION_ORDER[category]:
                sub = subc[subc["function"] == fn]
                if sub.empty:
                    continue
                rows.append(
                    {
                        "level": "function",
                        "metric": metric,
                        "category": category,
                        "function": fn,
                        "pairs": int(len(sub)),
                        "geomean_seconds": geomean(sub["seconds"]),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "category_function_time_drilldown.csv", index=False)

    fig = plt.figure(figsize=(12.4, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.45])
    ax_main = fig.add_subplot(gs[0, :])
    cat = summary[summary["level"] == "category"].copy()
    e2e_cat = cat[cat["metric"] == "E2E"].set_index("category").reindex(CATEGORY_ORDER)
    ker_cat = cat[cat["metric"] == "Kernel"].set_index("category").reindex(CATEGORY_ORDER)
    x = np.arange(len(CATEGORY_ORDER), dtype=float)
    width = 0.34
    e2e_ms = e2e_cat["geomean_seconds"].to_numpy(dtype=float) * 1000.0
    ker_ms = ker_cat["geomean_seconds"].to_numpy(dtype=float) * 1000.0
    ax_main.bar(
        x - width / 2,
        e2e_ms,
        width=width,
        color=[CATEGORY_COLOR[c] for c in CATEGORY_ORDER],
        edgecolor="#F8FAFC",
        linewidth=0.65,
        alpha=0.62,
        label="E2E",
    )
    ax_main.bar(
        x + width / 2,
        ker_ms,
        width=width,
        color=[CATEGORY_COLOR[c] for c in CATEGORY_ORDER],
        edgecolor="#26323D",
        linewidth=0.8,
        alpha=0.96,
        label="Kernel",
    )
    for xi, e2ev, kerv in zip(x, e2e_ms, ker_ms):
        ax_main.text(xi - width / 2, e2ev * 1.13, f"{e2ev:.1f}", ha="center", va="bottom", fontsize=7.5, color="#1F2937")
        ax_main.text(xi + width / 2, kerv * 1.13, f"{kerv:.1f}", ha="center", va="bottom", fontsize=7.5, color="#1F2937", fontweight="bold")
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Geomean time (ms)")
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(CATEGORY_ORDER)
    ax_main.set_title("EGGPU Time by Function Category", fontweight="bold", pad=8)
    ax_main.set_axisbelow(True)
    ax_main.grid(axis="y", which="major", color="#D8DEE8", linewidth=0.7, alpha=0.75)
    ax_main.grid(axis="y", which="minor", color="#EEF1F5", linewidth=0.45, alpha=0.55)
    ax_main.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)

    for idx, category in enumerate(CATEGORY_ORDER):
        ax = fig.add_subplot(gs[1, idx])
        funcs = FUNCTION_ORDER[category]
        func_rows = summary[(summary["level"] == "function") & (summary["category"] == category)].copy()
        e2ef = func_rows[func_rows["metric"] == "E2E"].set_index("function").reindex(funcs)
        kerf = func_rows[func_rows["metric"] == "Kernel"].set_index("function").reindex(funcs)
        y = np.arange(len(funcs), dtype=float)
        height = 0.34
        e2ev = e2ef["geomean_seconds"].to_numpy(dtype=float) * 1000.0
        kerv = kerf["geomean_seconds"].to_numpy(dtype=float) * 1000.0
        ax.barh(
            y - height / 2,
            e2ev,
            height=height,
            color=CATEGORY_COLOR[category],
            edgecolor="#F8FAFC",
            linewidth=0.5,
            alpha=0.55,
        )
        ax.barh(
            y + height / 2,
            kerv,
            height=height,
            color=CATEGORY_COLOR[category],
            edgecolor="#26323D",
            linewidth=0.65,
            alpha=0.96,
        )
        maxv = np.nanmax(np.concatenate([e2ev, kerv]))
        for yi, ev, kv in zip(y, e2ev, kerv):
            if math.isfinite(ev):
                ax.text(ev * 1.07, yi - height / 2, f"{ev:.1f}", va="center", ha="left", fontsize=6.8, color="#1F2937")
            if math.isfinite(kv):
                ax.text(kv * 1.07, yi + height / 2, f"{kv:.1f}", va="center", ha="left", fontsize=6.8, color="#1F2937", fontweight="bold")
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels(funcs)
        ax.invert_yaxis()
        ax.set_xlim(max(0.05, np.nanmin(np.concatenate([e2ev, kerv])) * 0.7), maxv * 2.5)
        ax.set_xlabel("Time (ms)")
        ax.set_title(category, fontweight="bold", color=CATEGORY_COLOR[category], pad=6)
        ax.set_axisbelow(True)
        ax.grid(axis="x", which="major", color="#E1E6ED", linewidth=0.65, alpha=0.8)
        ax.grid(axis="x", which="minor", color="#EEF1F5", linewidth=0.35, alpha=0.55)
    fig.savefig(out_dir / "category_function_time_drilldown.png", dpi=280, bbox_inches="tight")
    fig.savefig(out_dir / "category_function_time_drilldown.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def collect_layout(ablation_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(ablation_dir.glob("layout_*.csv")):
        dataset = path.name[len("layout_") : -len(".csv")]
        df = pd.read_csv(path)
        def val(func: str, metric: str) -> float:
            hit = df[(df["function"] == func) & (df["metric"] == metric) & (df["status"] == "ok")]
            if hit.empty:
                return float("nan")
            return float(pd.to_numeric(hit["value"], errors="coerce").median())
        row = {
            "dataset": dataset,
            "coo_host_mb": val("COO", "host_storage_mb"),
            "csr_host_mb": val("CSR", "host_storage_mb"),
            "coo_degree_s": val("COO", "degree_seconds"),
            "csr_degree_s": val("CSR", "degree_seconds"),
            "coo_pr_kernel_s": val("COO-PageRank", "gpu_kernel_seconds"),
            "csr_pr_kernel_s": val("CSR-PageRank", "gpu_kernel_seconds"),
        }
        row["storage_ratio_coo_over_csr"] = row["coo_host_mb"] / row["csr_host_mb"] if row["csr_host_mb"] > 0 else float("nan")
        row["degree_speedup_csr_vs_coo"] = row["coo_degree_s"] / row["csr_degree_s"] if row["csr_degree_s"] > 0 else float("nan")
        row["pagerank_speedup_csr_vs_coo"] = row["coo_pr_kernel_s"] / row["csr_pr_kernel_s"] if row["csr_pr_kernel_s"] > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def collect_return(ablation_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(ablation_dir.glob("return_*.csv")):
        dataset = path.name[len("return_") : -len(".csv")]
        df = pd.read_csv(path)
        sub = df[df["status"] == "ok"].copy()
        if sub.empty:
            continue
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        uses_new_schema = (sub["metric"] == "call_return_seconds").any()
        if uses_new_schema:
            if "materialization_claim_valid" in sub.columns:
                sub = sub[sub["materialization_claim_valid"].astype(str).str.lower() == "true"]
            call_metric = "call_return_seconds"
            traversed_metric = "call_plus_forced_traversal_seconds"
        else:
            call_metric = "lazy_call_e2e"
            traversed_metric = "eager_equivalent_e2e"
        call = sub[sub["metric"] == call_metric].groupby("function")["value"].median()
        traversed = sub[sub["metric"] == traversed_metric].groupby("function")["value"].median()
        for fn in sorted(set(call.index) & set(traversed.index)):
            if call[fn] > 0 and traversed[fn] > 0:
                rows.append(
                    {
                        "dataset": dataset,
                        "function": fn,
                        "call_return_seconds": float(call[fn]),
                        "call_plus_forced_traversal_seconds": float(traversed[fn]),
                        "traversed_over_call": float(traversed[fn] / call[fn]),
                    }
                )
    return pd.DataFrame(rows)


def collect_workflow_layer(out_dir: Path, ablation_dir: Path) -> pd.DataFrame:
    existing = out_dir / "workflow_layer_slowdown.csv"
    if existing.exists():
        return pd.read_csv(existing)
    # Fallback if workflow visuals have not been generated.
    from generate_workflow_reuse_visuals import plot_layer_slowdown
    return plot_layer_slowdown(ablation_dir, out_dir)


def plot_ablation_overview(out_dir: Path, ablation_dir: Path) -> pd.DataFrame:
    layout = collect_layout(ablation_dir)
    ret = collect_return(ablation_dir)
    layer = collect_workflow_layer(out_dir, ablation_dir)
    layout.to_csv(out_dir / "ablation_layout_summary.csv", index=False)
    ret.to_csv(out_dir / "ablation_return_summary.csv", index=False)

    rows = [
        {
            "panel": "Cross-function reuse",
            "metric": "Disable C++ graph cache",
            "value": float(layer[layer["variant"] == "no_cpp_graph_cache"]["geomean_slowdown"].iloc[0]),
            "unit": "slowdown",
        },
        {
            "panel": "Cross-function reuse",
            "metric": "Disable GraphContext",
            "value": float(layer[layer["variant"] == "no_graph_context"]["geomean_slowdown"].iloc[0]),
            "unit": "slowdown",
        },
        {
            "panel": "CSR vs COO layout",
            "metric": "Degree traversal",
            "value": geomean(layout["degree_speedup_csr_vs_coo"]),
            "unit": "speedup",
        },
        {
            "panel": "CSR vs COO layout",
            "metric": "PageRank kernel",
            "value": geomean(layout["pagerank_speedup_csr_vs_coo"]),
            "unit": "speedup",
        },
        {
            "panel": "Result return path",
            "metric": "Forced result traversal",
            "value": geomean(ret["traversed_over_call"]),
            "unit": "slowdown",
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "ablation_story_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), constrained_layout=True)
    panels = [
        ("Cross-function reuse", "#B7797C", "Slowdown after disabling reuse"),
        ("CSR vs COO layout", "#D9A441", "CSR speedup over COO"),
        ("Result return path", "#4C9BD6", "Call plus traversal / call return"),
    ]
    for ax, (panel, color, ylabel) in zip(axes, panels):
        sub = summary[summary["panel"] == panel]
        x = np.arange(len(sub), dtype=float)
        vals = sub["value"].to_numpy(dtype=float)
        bar_width = 0.40 if len(sub) == 1 else 0.56
        ax.bar(x, vals, width=bar_width, color=color, edgecolor="#F8FAFC", linewidth=0.65)
        for xi, row in zip(x, sub.itertuples()):
            ax.text(
                xi,
                row.value + max(vals) * 0.045,
                f"{row.value:.1f}x",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#1F2937",
            )
        ax.set_ylim(0, max(vals) * 1.25)
        if len(sub) == 1:
            ax.set_xlim(-0.55, 0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["metric"], rotation=14, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(panel, fontweight="bold", pad=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.75)
    fig.savefig(out_dir / "ablation_story_overview.png", dpi=280, bbox_inches="tight")
    fig.savefig(out_dir / "ablation_story_overview.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def write_readme(out_dir: Path) -> None:
    lines = [
        "# Ablation story visuals",
        "",
        "- `ablation_story_overview.*`: compact summary of workflow reuse, CSR layout, and return-path ablations.",
        "- `category_function_drilldown_e2e.*`: one large category-level best-normalized E2E view plus four function-level drill-down panels.",
        "- `category_function_time_drilldown.*`: one large category-level EGGPU time view plus four function-level time drill-down panels.",
        "",
        "The category drill-down uses best-normalized EGGPU E2E performance.  A value of 1.0 means EGGPU is the fastest or tied for fastest for that function-dataset pair; smaller values mean EGGPU is slower than the best available baseline.",
        "",
    ]
    (out_dir / "ablation_story_README.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-result-dir", required=True, type=Path)
    parser.add_argument("--ablation-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    setup_style()
    result_dir = args.main_result_dir.resolve()
    ablation_dir = args.ablation_dir.resolve()
    out_dir = (args.out_dir or (result_dir / "paper_visuals_v4")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_category_drilldown(result_dir, out_dir)
    plot_category_function_time_drilldown(result_dir, out_dir)
    plot_ablation_overview(out_dir, ablation_dir)
    write_readme(out_dir)
    print(f"Wrote ablation story visuals to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
