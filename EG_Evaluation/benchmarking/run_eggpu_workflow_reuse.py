#!/usr/bin/env python3
"""Same-process cross-function reuse benchmark for EGGPU.

The full baseline runner intentionally isolates each library/function in a
separate process.  That is the right fairness protocol for library comparison,
but it under-represents the system benefit of EGGPU's reusable graph context.

This runner measures the complementary user workflow:

    build one EasyGraph object/view bundle
    prewarm GraphContext/C++ graph/device CSR caches
    run multiple EGGPU functions in the same Python process
    compare those timed calls with the isolated full-eval results

Only EGGPU is executed by this script.  Baseline numbers are read from an
existing full-eval directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from run_eggpu_ablations import DEFAULT_FUNCTIONS
from run_eggpu_ablations import run_workflow


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = [
    ("small", "undirected", "ca-GrQc", "datasets/undirected/ca-GrQc.txt"),
    ("small", "undirected", "ca-HepTh", "datasets/undirected/ca-HepTh.txt"),
    ("small", "undirected", "LastFM", "datasets/undirected/LastFM.txt"),
    ("small", "undirected", "pgp", "datasets/undirected/pgp.txt"),
    ("medium", "undirected", "ca-CondMat", "datasets/undirected/ca-CondMat.txt"),
    ("medium", "undirected", "ca-HepPh", "datasets/undirected/ca-HepPh.txt"),
    ("medium", "undirected", "email-Enron", "datasets/undirected/email-Enron.txt"),
    ("large", "undirected", "com-youtube", "datasets/undirected/com-youtube.ungraph.txt"),
    ("small", "directed", "p2p-Gnutella04", "datasets/directed/p2p-Gnutella04.txt"),
    ("small", "directed", "p2p-Gnutella08", "datasets/directed/p2p-Gnutella08.txt"),
    ("medium", "directed", "wiki-Vote", "datasets/directed/wiki-Vote.txt"),
    ("medium", "directed", "soc-Epinions1", "datasets/directed/soc-Epinions1.txt"),
    ("medium", "directed", "email-EuAll", "datasets/directed/email-EuAll.txt"),
    ("large", "directed", "soc-Slashdot0811", "datasets/directed/soc-Slashdot0811.txt"),
    ("large", "directed", "web-NotreDame", "datasets/directed/web-NotreDame.txt"),
    ("large", "directed", "ER-100k", "datasets/directed/ER-100k.txt"),
    ("large", "directed", "wiki-Talk", "datasets/directed/wiki-Talk.txt"),
]


def parse_csv_tokens(value):
    if not value or str(value).strip().lower() == "all":
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def select_datasets(value):
    tokens = {x.lower() for x in parse_csv_tokens(value)}
    if not tokens:
        return list(DEFAULT_DATASETS)
    out = []
    for size, graph_type, name, path in DEFAULT_DATASETS:
        keys = {size.lower(), graph_type.lower(), name.lower()}
        if keys & tokens:
            out.append((size, graph_type, name, path))
    if not out:
        raise SystemExit(f"No datasets matched --datasets={value!r}")
    return out


def select_functions(value):
    tokens = parse_csv_tokens(value)
    if not tokens:
        return list(DEFAULT_FUNCTIONS)
    allowed = set(DEFAULT_FUNCTIONS)
    bad = [x for x in tokens if x not in allowed]
    if bad:
        raise SystemExit(f"Invalid --functions entries {bad}; valid={sorted(allowed)}")
    return tokens


def geomean(values):
    arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    arr = arr[arr > 0]
    if arr.empty:
        return np.nan
    return float(np.exp(np.log(arr).mean()))


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(args, size, graph_type, name, edge_path, functions):
    workflow_args = SimpleNamespace(
        experiment="workflow",
        variant="full",
        edge_path=edge_path,
        dataset_name=name,
        graph_type=graph_type,
        functions=",".join(functions),
        repeat=args.repeat,
        warmup=args.warmup,
        gpu=args.gpu,
        sssp_sources=args.sssp_sources,
        bc_sources=args.bc_sources,
        layout_pr_iters=20,
        out="",
    )
    rows = run_workflow(workflow_args)
    for row in rows:
        row["dataset_size"] = size
        row["reuse_protocol"] = (
            "same-process same-EasyGraph-object/view-bundle; "
            f"GraphContext prewarm + {args.warmup} full workflow warmup cycle(s)"
        )
    return rows


def median_workflow(rows):
    df = pd.DataFrame(rows)
    ok = df[(df["status"] == "ok") & (df["function"] != "ALL") & (df["metric"].isin(["e2e", "kernel"]))]
    med = (
        ok.groupby(["dataset_size", "graph_type", "dataset", "function", "metric"], as_index=False)["value"]
        .median()
        .rename(columns={"value": "workflow_seconds"})
    )
    return med


def compare_with_full_eval(workflow_median, full_eval_dir):
    full_path = Path(full_eval_dir) / "results_long.csv"
    if not full_path.exists():
        raise SystemExit(f"Missing full-eval results_long.csv: {full_path}")
    full = pd.read_csv(full_path)
    ok = full[(full["status"] == "ok") & (full["metric"].isin(["e2e", "kernel"]))].copy()
    isolated = (
        ok[ok["baseline"] == "EGGPU"][
            ["dataset_size", "graph_type", "dataset", "function", "metric", "seconds"]
        ]
        .rename(columns={"seconds": "isolated_eggpu_seconds"})
    )
    best_other = (
        ok[ok["baseline"] != "EGGPU"]
        .groupby(["dataset_size", "graph_type", "dataset", "function", "metric"], as_index=False)["seconds"]
        .min()
        .rename(columns={"seconds": "best_other_seconds"})
    )
    cmp = workflow_median.merge(
        isolated,
        on=["dataset_size", "graph_type", "dataset", "function", "metric"],
        how="left",
    ).merge(
        best_other,
        on=["dataset_size", "graph_type", "dataset", "function", "metric"],
        how="left",
    )
    cmp["isolated_over_workflow"] = cmp["isolated_eggpu_seconds"] / cmp["workflow_seconds"]
    cmp["best_other_over_workflow"] = cmp["best_other_seconds"] / cmp["workflow_seconds"]
    cmp["workflow_faster_than_isolated"] = cmp["isolated_over_workflow"] > 1.0
    cmp["workflow_sota_vs_best_other"] = cmp["best_other_over_workflow"] > 1.0
    return cmp


def savefig(fig, out_dir, stem):
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_heatmap(cmp, out_dir, metric, value_col, title, stem):
    funcs = list(DEFAULT_FUNCTIONS)
    datasets = [d[2] for d in DEFAULT_DATASETS if d[2] in set(cmp["dataset"])]
    mat = (
        cmp[cmp["metric"] == metric]
        .pivot_table(index="function", columns="dataset", values=value_col, aggfunc="median")
        .reindex(index=funcs, columns=datasets)
    )
    values = mat.values[np.isfinite(mat.values)]
    vmax = max(2.5, float(np.nanpercentile(values, 90))) if values.size else 2.5
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(
        mat.values,
        aspect="auto",
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=0.25, vcenter=1.0, vmax=vmax),
    )
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            value = mat.values[r, c]
            if np.isfinite(value):
                ax.text(c, r, f"{value:.2f}x", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("ratio")
    savefig(fig, out_dir, stem)


def write_analysis(cmp, out_dir, full_eval_dir, args):
    lines = []
    lines.append("# EGGPU Cross-Function Reuse Benchmark\n\n")
    lines.append("## Protocol\n\n")
    lines.append("- Only EGGPU is executed in this benchmark.\n")
    lines.append("- Each dataset is built once in one Python process as an EasyGraph graph/view bundle.\n")
    lines.append("- GraphContext/C++ graph/device CSR paths are prewarmed before measurement.\n")
    lines.append(f"- The full workflow is warmed up `{args.warmup}` cycle(s) before timed calls.\n")
    lines.append("- Timed functions then run on the same graph objects, so all measured calls can reuse cached graph context.\n")
    lines.append("- Isolated EGGPU and best non-EGGPU numbers are read from the full-eval directory, not rerun here.\n\n")
    lines.append(f"Full-eval comparison source: `{Path(full_eval_dir)}`\n\n")

    for metric in ("e2e", "kernel"):
        sub = cmp[cmp["metric"] == metric]
        lines.append(f"## {metric.upper()} summary\n\n")
        lines.append(
            f"- Workflow faster than isolated EGGPU on "
            f"**{int(sub['workflow_faster_than_isolated'].sum())}/{len(sub)}** pairs; "
            f"geomean isolated/workflow = **{geomean(sub['isolated_over_workflow']):.2f}x**.\n"
        )
        lines.append(
            f"- Workflow SOTA vs best non-EGGPU on "
            f"**{int(sub['workflow_sota_vs_best_other'].sum())}/{len(sub)}** pairs; "
            f"geomean best-other/workflow = **{geomean(sub['best_other_over_workflow']):.2f}x**.\n\n"
        )
        func = (
            sub.groupby("function")
            .agg(
                n=("function", "size"),
                workflow_vs_isolated_wins=("workflow_faster_than_isolated", "sum"),
                workflow_sota_wins=("workflow_sota_vs_best_other", "sum"),
                isolated_over_workflow_geomean=("isolated_over_workflow", geomean),
                best_other_over_workflow_geomean=("best_other_over_workflow", geomean),
                workflow_seconds_median=("workflow_seconds", "median"),
            )
            .reindex(DEFAULT_FUNCTIONS)
            .reset_index()
        )
        lines.append(func.to_string(index=False))
        lines.append("\n\n")

    lines.append("## Generated artifacts\n\n")
    lines.append("- `workflow_reuse_long.csv`\n")
    lines.append("- `workflow_reuse_median.csv`\n")
    lines.append("- `workflow_reuse_compare.csv`\n")
    lines.append("- `workflow_reuse_vs_isolated_e2e.png/.pdf`\n")
    lines.append("- `workflow_reuse_vs_best_other_e2e.png/.pdf`\n")
    lines.append("- `workflow_reuse_vs_isolated_kernel.png/.pdf`\n")
    lines.append("- `workflow_reuse_vs_best_other_kernel.png/.pdf`\n")
    (out_dir / "workflow_reuse_analysis.md").write_text("".join(lines), encoding="utf-8")


def write_plots(cmp, out_dir):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 170,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for metric in ("e2e", "kernel"):
        plot_heatmap(
            cmp,
            out_dir,
            metric,
            "isolated_over_workflow",
            f"EGGPU workflow reuse vs isolated EGGPU {metric.upper()} (>1 means reuse faster)",
            f"workflow_reuse_vs_isolated_{metric}",
        )
        plot_heatmap(
            cmp,
            out_dir,
            metric,
            "best_other_over_workflow",
            f"EGGPU workflow reuse vs best non-EGGPU {metric.upper()} (>1 means EGGPU SOTA)",
            f"workflow_reuse_vs_best_other_{metric}",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--functions", default="all")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--sssp-sources", type=int, default=8)
    ap.add_argument("--bc-sources", type=int, default=16)
    ap.add_argument("--full-eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["EGGPU_MONITOR_GPU_INDEX"] = str(args.gpu)
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    os.environ["EASYGRAPH_GPU_BACKEND"] = "mine"
    os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "TRUE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = select_datasets(args.datasets)
    functions = select_functions(args.functions)

    all_rows = []
    for size, graph_type, name, edge_path in datasets:
        print(f"=== workflow reuse dataset {name} ({graph_type}, {size}) ===", flush=True)
        rows = run_dataset(args, size, graph_type, name, edge_path, functions)
        all_rows.extend(rows)
        write_rows(all_rows, out_dir / "workflow_reuse_long.csv")

    write_rows(all_rows, out_dir / "workflow_reuse_long.csv")
    med = median_workflow(all_rows)
    med.to_csv(out_dir / "workflow_reuse_median.csv", index=False)
    cmp = compare_with_full_eval(med, args.full_eval_dir)
    cmp.to_csv(out_dir / "workflow_reuse_compare.csv", index=False)
    write_plots(cmp, out_dir)
    write_analysis(cmp, out_dir, args.full_eval_dir, args)
    print(f"Done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
