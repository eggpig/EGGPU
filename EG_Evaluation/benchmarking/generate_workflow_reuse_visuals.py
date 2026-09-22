#!/usr/bin/env python3
"""Generate paper-style workflow reuse figures.

The figures compare isolated EGGPU calls from the full benchmark with
same-graph workflow calls from the EGGPU ablation run.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAIR_ORDER = [
    ("LCC", "WCC"),
    ("WCC", "SCC"),
    ("BFS", "Dijkstra"),
    ("Dijkstra", "BellmanFord"),
    ("BellmanFord", "SSSP"),
    ("EffectiveSize", "Efficiency"),
    ("Constraint", "Hierarchy"),
]

PAIR_LABEL = {
    ("LCC", "WCC"): "LCC + WCC",
    ("WCC", "SCC"): "WCC + SCC",
    ("BFS", "Dijkstra"): "BFS + Dijkstra",
    ("Dijkstra", "BellmanFord"): "Dijkstra + BF",
    ("BellmanFord", "SSSP"): "BF + SSSP",
    ("EffectiveSize", "Efficiency"): "EffSize + Eff",
    ("Constraint", "Hierarchy"): "Constraint + Hier",
}

EXAMPLE_CASES = [
    ("p2p-Gnutella04", ("LCC", "WCC")),
    ("wiki-Vote", ("WCC", "SCC")),
    ("ca-HepPh", ("BellmanFord", "SSSP")),
    ("wiki-Vote", ("EffectiveSize", "Efficiency")),
    ("soc-Epinions1", ("Constraint", "Hierarchy")),
]

VARIANT_LABEL = {
    "full": "Full reuse",
    "no_device_csr_cache": "No device CSR cache",
    "no_cpp_graph_cache": "No C++ graph cache",
    "no_graph_context": "No GraphContext",
}

VARIANT_ORDER = [
    "full",
    "no_device_csr_cache",
    "no_cpp_graph_cache",
    "no_graph_context",
]


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


def dataset_from_workflow_path(path: Path, variant: str) -> str:
    prefix = "workflow_"
    suffix = f"_{variant}.csv"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(name)
    return name[len(prefix) : -len(suffix)]


def load_isolated_e2e(main_result_dir: Path) -> dict[tuple[str, str], float]:
    path = main_result_dir / "results_e2e.csv"
    df = pd.read_csv(path)
    sub = df[(df["baseline"] == "EGGPU") & (df["status"] == "ok")].copy()
    sub["seconds"] = pd.to_numeric(sub["seconds"], errors="coerce")
    sub = sub[sub["seconds"].notna()]
    return {(str(r.dataset), str(r.function)): float(r.seconds) for r in sub.itertuples()}


def load_workflow_function_medians(ablation_dir: Path, variant: str = "full") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in sorted(ablation_dir.glob(f"workflow_*_{variant}.csv")):
        dataset = dataset_from_workflow_path(path, variant)
        df = pd.read_csv(path)
        sub = df[(df["status"] == "ok") & (df["metric"] == "e2e")].copy()
        if sub.empty:
            continue
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub[sub["value"].notna()]
        if sub.empty:
            continue
        med = sub.groupby("function")["value"].median()
        out[dataset] = {str(k): float(v) for k, v in med.items()}
    return out


def build_pair_rows(main_result_dir: Path, ablation_dir: Path) -> pd.DataFrame:
    isolated = load_isolated_e2e(main_result_dir)
    workflow = load_workflow_function_medians(ablation_dir, "full")
    rows = []
    for dataset, wf_times in workflow.items():
        for pair in PAIR_ORDER:
            f1, f2 = pair
            if (dataset, f1) not in isolated or (dataset, f2) not in isolated:
                continue
            if f1 not in wf_times or f2 not in wf_times:
                continue
            isolated_sum = isolated[(dataset, f1)] + isolated[(dataset, f2)]
            workflow_sum = wf_times[f1] + wf_times[f2]
            if isolated_sum <= 0 or workflow_sum <= 0:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "function_a": f1,
                    "function_b": f2,
                    "pair": PAIR_LABEL[pair],
                    "isolated_sum_seconds": isolated_sum,
                    "workflow_sum_seconds": workflow_sum,
                    "speedup": isolated_sum / workflow_sum,
                    "isolated_a_seconds": isolated[(dataset, f1)],
                    "isolated_b_seconds": isolated[(dataset, f2)],
                    "workflow_a_seconds": wf_times[f1],
                    "workflow_b_seconds": wf_times[f2],
                }
            )
    return pd.DataFrame(rows)


def plot_pair_geomean(pair_rows: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summary_rows = []
    for pair in [PAIR_LABEL[p] for p in PAIR_ORDER]:
        sub = pair_rows[pair_rows["pair"] == pair]
        if sub.empty:
            continue
        summary_rows.append(
            {
                "pair": pair,
                "datasets": int(len(sub)),
                "isolated_geomean_seconds": geomean(sub["isolated_sum_seconds"]),
                "workflow_geomean_seconds": geomean(sub["workflow_sum_seconds"]),
                "speedup_geomean": geomean(sub["speedup"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "workflow_pair_reuse_geomean.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.2, 4.35), constrained_layout=True)
    x = np.arange(len(summary), dtype=float)
    width = 0.34
    isolated_ms = summary["isolated_geomean_seconds"].to_numpy(dtype=float) * 1000.0
    workflow_ms = summary["workflow_geomean_seconds"].to_numpy(dtype=float) * 1000.0

    ax.bar(
        x - width / 2,
        isolated_ms,
        width=width,
        label="Separate calls",
        color="#B9C0CA",
        edgecolor="#F8FAFC",
        linewidth=0.55,
    )
    ax.bar(
        x + width / 2,
        workflow_ms,
        width=width,
        label="Same-graph workflow",
        color="#4C9BD6",
        edgecolor="#173B57",
        linewidth=1.0,
    )

    for i, row in enumerate(summary.itertuples()):
        y = max(isolated_ms[i], workflow_ms[i])
        ax.text(
            i,
            y * 1.18,
            f"{row.speedup_geomean:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1F2937",
            fontweight="bold",
        )

    ax.set_yscale("log")
    ax.set_ylabel("Geomean pair time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["pair"], rotation=22, ha="right")
    ax.set_title("Same-Graph Pair Reuse", fontweight="bold", pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="#D8DEE8", linewidth=0.7, alpha=0.75)
    ax.grid(axis="y", which="minor", color="#EEF1F5", linewidth=0.45, alpha=0.55)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    fig.savefig(out_dir / "workflow_pair_reuse_geomean.png", dpi=260, bbox_inches="tight")
    fig.savefig(out_dir / "workflow_pair_reuse_geomean.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_pair_examples(pair_rows: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for dataset, pair in EXAMPLE_CASES:
        label = PAIR_LABEL[pair]
        hit = pair_rows[(pair_rows["dataset"] == dataset) & (pair_rows["pair"] == label)]
        if hit.empty:
            continue
        rows.append(hit.iloc[0].to_dict())
    examples = pd.DataFrame(rows)
    examples.to_csv(out_dir / "workflow_pair_reuse_examples.csv", index=False)
    if examples.empty:
        return examples

    labels = [f"{r.dataset}\n{r.pair}" for r in examples.itertuples()]
    x = np.arange(len(examples), dtype=float)
    width = 0.16
    fig, ax = plt.subplots(figsize=(9.4, 4.65), constrained_layout=True)

    iso_a = examples["isolated_a_seconds"].to_numpy(dtype=float) * 1000.0
    iso_b = examples["isolated_b_seconds"].to_numpy(dtype=float) * 1000.0
    wf_a = examples["workflow_a_seconds"].to_numpy(dtype=float) * 1000.0
    wf_b = examples["workflow_b_seconds"].to_numpy(dtype=float) * 1000.0

    offsets = np.array([-1.8, -0.6, 0.6, 1.8]) * width
    ax.bar(
        x + offsets[0],
        iso_a,
        width=width,
        color="#B9C0CA",
        edgecolor="#F8FAFC",
        linewidth=0.55,
        label="Separate call A",
    )
    ax.bar(
        x + offsets[1],
        iso_b,
        width=width,
        color="#D7DCE3",
        edgecolor="#F8FAFC",
        linewidth=0.55,
        label="Separate call B",
    )
    ax.bar(
        x + offsets[2],
        wf_a,
        width=width,
        color="#4C9BD6",
        edgecolor="#173B57",
        linewidth=0.8,
        label="Workflow call A",
    )
    ax.bar(
        x + offsets[3],
        wf_b,
        width=width,
        color="#8BC3E6",
        edgecolor="#173B57",
        linewidth=0.8,
        label="Workflow call B",
    )

    iso_total = (examples["isolated_sum_seconds"].to_numpy(dtype=float) * 1000.0)
    wf_total = (examples["workflow_sum_seconds"].to_numpy(dtype=float) * 1000.0)
    for i, row in enumerate(examples.itertuples()):
        y = max(iso_total[i], wf_total[i])
        ax.text(
            i,
            y * 1.16,
            f"{row.speedup:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1F2937",
            fontweight="bold",
        )

    ax.set_yscale("log")
    ax.set_ylabel("Component time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Representative Same-Graph Pair Reuse", fontweight="bold", pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="#D8DEE8", linewidth=0.7, alpha=0.75)
    ax.grid(axis="y", which="minor", color="#EEF1F5", linewidth=0.45, alpha=0.55)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False)
    fig.savefig(out_dir / "workflow_pair_reuse_examples.png", dpi=260, bbox_inches="tight")
    fig.savefig(out_dir / "workflow_pair_reuse_examples.pdf", bbox_inches="tight")
    plt.close(fig)
    return examples


def workflow_totals_for_variant(ablation_dir: Path, variant: str) -> pd.DataFrame:
    rows = []
    for path in sorted(ablation_dir.glob(f"workflow_*_{variant}.csv")):
        dataset = dataset_from_workflow_path(path, variant)
        df = pd.read_csv(path)
        sub = df[(df["status"] == "ok") & (df["metric"] == "e2e")].copy()
        if sub.empty:
            rows.append({"dataset": dataset, "variant": variant, "status": "timeout", "seconds": math.nan})
            continue
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub[sub["value"].notna()]
        totals = sub.groupby("repeat")["value"].sum()
        if totals.empty:
            rows.append({"dataset": dataset, "variant": variant, "status": "missing", "seconds": math.nan})
        else:
            rows.append({"dataset": dataset, "variant": variant, "status": "ok", "seconds": float(totals.median())})
    return pd.DataFrame(rows)


def plot_layer_slowdown(ablation_dir: Path, out_dir: Path) -> pd.DataFrame:
    all_rows = []
    for variant in VARIANT_ORDER:
        all_rows.append(workflow_totals_for_variant(ablation_dir, variant))
    totals = pd.concat(all_rows, ignore_index=True)
    totals.to_csv(out_dir / "workflow_layer_totals.csv", index=False)

    pivot = totals.pivot(index="dataset", columns="variant", values="seconds")
    rows = []
    for variant in VARIANT_ORDER:
        if variant == "full":
            rows.append(
                {
                    "variant": variant,
                    "label": VARIANT_LABEL[variant],
                    "completed_datasets": int(pivot["full"].dropna().shape[0]),
                    "geomean_slowdown": 1.0,
                }
            )
            continue
        sub = pivot[["full", variant]].dropna()
        sub = sub[(sub["full"] > 0) & (sub[variant] > 0)]
        slowdown = sub[variant] / sub["full"]
        rows.append(
            {
                "variant": variant,
                "label": VARIANT_LABEL[variant],
                "completed_datasets": int(len(sub)),
                "geomean_slowdown": geomean(slowdown),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "workflow_layer_slowdown.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    x = np.arange(len(summary), dtype=float)
    vals = summary["geomean_slowdown"].to_numpy(dtype=float)
    colors = ["#4C9BD6", "#8BC3E6", "#D9A441", "#B7797C"]
    bars = ax.bar(
        x,
        vals,
        width=0.58,
        color=colors[: len(summary)],
        edgecolor="#F8FAFC",
        linewidth=0.65,
    )
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.035,
            f"{row.geomean_slowdown:.1f}x",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1F2937",
            fontweight="bold",
        )
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_ylabel("Workflow slowdown")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"], rotation=18, ha="right")
    ax.set_title("Workflow Reuse Layer Contribution", fontweight="bold", pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D8DEE8", linewidth=0.7, alpha=0.75)
    ax.grid(axis="x", visible=False)
    fig.savefig(out_dir / "workflow_layer_slowdown.png", dpi=260, bbox_inches="tight")
    fig.savefig(out_dir / "workflow_layer_slowdown.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def write_readme(out_dir: Path, main_result_dir: Path, ablation_dir: Path, pair_summary: pd.DataFrame, layer_summary: pd.DataFrame) -> None:
    lines = [
        "# Workflow reuse visuals",
        "",
        f"Full result: `{main_result_dir}`",
        f"Ablation result: `{ablation_dir}`",
        "",
        "Generated files:",
        "",
        "- `workflow_pair_reuse_geomean.png/.pdf`: separate calls versus same-graph workflow for representative function pairs.",
        "- `workflow_pair_reuse_examples.png/.pdf`: examples showing the two function-call components.",
        "- `workflow_layer_slowdown.png/.pdf`: workflow slowdown after disabling reuse layers.",
        "",
        "Pair geomean speedups:",
        "",
        "| Pair | Datasets | Speedup |",
        "|---|---:|---:|",
    ]
    for row in pair_summary.itertuples():
        lines.append(f"| {row.pair} | {int(row.datasets)} | {row.speedup_geomean:.2f}x |")
    lines.extend(["", "Layer slowdown:", "", "| Variant | Completed datasets | Slowdown |", "|---|---:|---:|"])
    for row in layer_summary.itertuples():
        lines.append(f"| {row.label} | {int(row.completed_datasets)} | {row.geomean_slowdown:.2f}x |")
    lines.extend(
        [
            "",
            "The pair comparison is conservative because isolated functions in the full benchmark already follow the two-call EGGPU warmup policy.  The layer figure is the direct ablation evidence for GraphContext and graph-cache reuse.",
            "",
        ]
    )
    (out_dir / "workflow_reuse_README.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-result-dir", required=True, type=Path)
    parser.add_argument("--ablation-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    setup_style()
    main_result_dir = args.main_result_dir.resolve()
    ablation_dir = args.ablation_dir.resolve()
    out_dir = (args.out_dir or (main_result_dir / "paper_visuals_v4")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = build_pair_rows(main_result_dir, ablation_dir)
    pair_rows.to_csv(out_dir / "workflow_pair_reuse_detail.csv", index=False)
    pair_summary = plot_pair_geomean(pair_rows, out_dir)
    plot_pair_examples(pair_rows, out_dir)
    layer_summary = plot_layer_slowdown(ablation_dir, out_dir)
    write_readme(out_dir, main_result_dir, ablation_dir, pair_summary, layer_summary)
    print(f"Wrote workflow reuse visuals to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
