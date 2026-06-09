#!/usr/bin/env python3
"""Generate paper-ready tables and ablation figures from benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path


BASELINE_ORDER = [
    "networkx",
    "easygraph-cpu",
    "easygraph-cpp",
    "igraph",
    "nx-cugraph",
    "Gunrock",
    "EGGPU",
]

BASELINE_LABEL = {
    "networkx": "NetworkX",
    "easygraph-cpu": "EasyGraph-Py",
    "easygraph-cpp": "EasyGraph-C++",
    "igraph": "igraph",
    "nx-cugraph": "nx-cugraph",
    "Gunrock": "Gunrock",
    "EGGPU": "EGGPU",
}

FUNCTION_ORDER = [
    "PageRank",
    "MST",
    "LCC",
    "WCC",
    "SCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
    "BC",
    "Closeness",
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
]
SCALE_ORDER = {"small": 0, "medium": 1, "large": 2}


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt_int(value: int | float | str) -> str:
    return f"{int(value):,}"


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def fmt_time(value: float | None, status: str | None = None) -> str:
    if status == "timeout":
        return "TO"
    if value is None or math.isnan(value):
        return "--"
    if value == 0:
        return "0"
    if value < 1e-3:
        return f"{value:.2e}"
    if value < 0.01:
        return f"{value:.4f}"
    if value < 1:
        return f"{value:.3f}"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def read_metric_csv(path: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    rows: dict[tuple[str, str, str], dict[str, object]] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as fp:
        for row in csv.DictReader(fp):
            key = (row["dataset"], row["function"], row["baseline"])
            seconds = None
            if row.get("seconds"):
                try:
                    seconds = float(row["seconds"])
                except ValueError:
                    seconds = None
            rows[key] = {
                "seconds": seconds,
                "status": row.get("status", ""),
                "dataset_size": row.get("dataset_size", ""),
                "graph_type": row.get("graph_type", ""),
            }
    return rows


def sorted_cases(metric_rows: dict[tuple[str, str, str], dict[str, object]], stats: list[dict[str, object]]):
    ds_meta = {d["name"]: d for d in stats}
    cases = sorted(
        {(dataset, function) for dataset, function, _ in metric_rows.keys()},
        key=lambda x: (
            FUNCTION_ORDER.index(x[1]) if x[1] in FUNCTION_ORDER else 999,
            SCALE_ORDER.get(str(ds_meta.get(x[0], {}).get("size", "")), 999),
            str(ds_meta.get(x[0], {}).get("graph_type", "")),
            x[0],
        ),
    )
    return cases


def ranked_values(metric_rows: dict[tuple[str, str, str], dict[str, object]], dataset: str, function: str):
    vals = []
    for baseline in BASELINE_ORDER:
        item = metric_rows.get((dataset, function, baseline))
        if not item:
            continue
        if item.get("status") == "ok" and item.get("seconds") is not None:
            vals.append((baseline, float(item["seconds"])))
    vals.sort(key=lambda x: x[1])
    best = vals[0][0] if vals else None
    second = vals[1][0] if len(vals) > 1 else None
    return best, second


def emit_metric_table(
    metric_rows: dict[tuple[str, str, str], dict[str, object]],
    stats: list[dict[str, object]],
    out_tex: Path,
    out_csv: Path,
    caption: str,
    label: str,
    fallback_rows: dict[tuple[str, str, str], dict[str, object]] | None = None,
    na_baselines: set[str] | None = None,
):
    cases = sorted_cases(metric_rows, stats)
    header = ["Algorithm", "Dataset"] + [BASELINE_LABEL[b] for b in BASELINE_ORDER]
    with out_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for dataset, function in cases:
            row = [function, dataset]
            for baseline in BASELINE_ORDER:
                item = metric_rows.get((dataset, function, baseline))
                if not item:
                    fallback = (fallback_rows or {}).get((dataset, function, baseline))
                    if baseline in (na_baselines or set()):
                        row.append("N/A")
                    elif fallback and fallback.get("status") == "timeout":
                        row.append("timeout")
                    else:
                        row.append("N/A")
                else:
                    row.append(item.get("seconds") if item.get("status") == "ok" else item.get("status"))
            writer.writerow(row)

    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{longtable}, \usepackage[table]{xcolor}",
        r"\definecolor{EGGPUBlue}{RGB}{225,241,255}",
        r"\begin{scriptsize}",
        r"\setlength{\tabcolsep}{2.2pt}",
        rf"\begin{{longtable}}{{ll{''.join(['r' for _ in BASELINE_ORDER])}}}",
        rf"\caption{{{latex_escape(caption)}}}\label{{{label}}}\\",
        r"\toprule",
        " & ".join(latex_escape(x) for x in header) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(latex_escape(x) for x in header) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for dataset, function in cases:
        best, second = ranked_values(metric_rows, dataset, function)
        cells = [latex_escape(function), latex_escape(dataset)]
        for baseline in BASELINE_ORDER:
            item = metric_rows.get((dataset, function, baseline))
            if item:
                text = fmt_time(item.get("seconds"), str(item.get("status")))
            else:
                fallback = (fallback_rows or {}).get((dataset, function, baseline))
                if baseline in (na_baselines or set()):
                    text = "N/A"
                elif fallback and fallback.get("status") == "timeout":
                    text = "TO"
                else:
                    text = "N/A"
            if baseline == best:
                text = rf"\textbf{{{text}}}"
            elif baseline == second:
                text = rf"\underline{{{text}}}"
            if baseline == "EGGPU":
                text = rf"\cellcolor{{EGGPUBlue}}{text}"
            cells.append(text)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\end{scriptsize}", ""])
    out_tex.write_text("\n".join(lines))


def read_edges_for_max_degree(root: Path, rel_path: str) -> int:
    path = root / rel_path
    degrees: defaultdict[int, int] = defaultdict(int)
    seen: set[int] = set()
    with path.open(errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            key = (a << 32) | b
            if key in seen:
                continue
            seen.add(key)
            degrees[a] += 1
            degrees[b] += 1
    return max(degrees.values()) if degrees else 0


def emit_dataset_table(stats: list[dict[str, object]], root: Path, out_tex: Path, out_csv: Path):
    rows = []
    for item in sorted(stats, key=lambda d: (SCALE_ORDER.get(str(d.get("size")), 999), str(d.get("graph_type")), str(d.get("name")))):
        n = int(item["nodes_raw"])
        e = int(item["edge_rows_no_selfloops"] if item["graph_type"] == "directed" else item["edges_undirected_unique"])
        avg = (2.0 * e / n) if n else 0.0
        density = (e / (n * (n - 1))) if item["graph_type"] == "directed" and n > 1 else ((2.0 * e) / (n * (n - 1)) if n > 1 else 0.0)
        dmax = read_edges_for_max_degree(root, str(item["path"]))
        rows.append(
            {
                "Dataset": item["name"],
                "|V|": n,
                "|E|": e,
                "d_avg": avg,
                "Density": density,
                "d_max": dmax,
                "Self-loops": int(item.get("selfloops", 0)),
                "Directed": "True" if item["graph_type"] == "directed" else "False",
                "Scale": item.get("size", ""),
            }
        )
    with out_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Attributes of the datasets}",
        r"\label{tab:dataset-attributes}",
        r"\begin{tabular}{lrrrrrrll}",
        r"\toprule",
        r"Dataset & $|V|$ & $|E|$ & $d_{\mathrm{avg}}$ & Density & $d_{\max}$ & Self-loops & Directed & Scale \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(r["Dataset"]),
                    fmt_int(r["|V|"]),
                    fmt_int(r["|E|"]),
                    fmt_float(float(r["d_avg"]), 2),
                    f"{float(r['Density']):.2e}",
                    fmt_int(r["d_max"]),
                    fmt_int(r["Self-loops"]),
                    latex_escape(r["Directed"]),
                    latex_escape(r["Scale"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    out_tex.write_text("\n".join(lines))


def use_paper_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


ABLATION_COLORS = {
    "blue": "#8FB3D9",
    "teal": "#8EC9C1",
    "sand": "#F2C48D",
    "green": "#A7D59B",
    "lavender": "#B8ADD9",
}


def style_ablation_axis(ax, *, log=False):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E9EEF4", linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", length=2.5, width=0.55, color="#7A8793")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B7C1CC")
        ax.spines[side].set_linewidth(0.6)
    if log:
        ax.set_yscale("log")


def read_ablation_metric(path: Path, metric: str) -> float | None:
    if not path.exists():
        return None
    total = 0.0
    found = False
    with path.open(newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("metric") == metric and row.get("status") == "ok":
                total += float(row["value"])
                found = True
    return total if found else None


def emit_workflow_ablation_figure(abl_dir: Path, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    records = []
    for full_path in sorted(abl_dir.glob("workflow_*_full.csv")):
        dataset = full_path.name[len("workflow_") : -len("_full.csv")]
        full = read_ablation_metric(full_path, "e2e")
        no_ctx = read_ablation_metric(abl_dir / f"workflow_{dataset}_no_graph_context.csv", "e2e")
        no_cpp = read_ablation_metric(abl_dir / f"workflow_{dataset}_no_cpp_graph_cache.csv", "e2e")
        no_dev = read_ablation_metric(abl_dir / f"workflow_{dataset}_no_device_csr_cache.csv", "e2e")
        if not full or not no_ctx or not no_cpp:
            continue
        records.append((dataset, no_ctx / full, no_cpp / full, (no_dev / full) if no_dev else None))
    if not records:
        return
    records.sort(key=lambda x: x[1], reverse=True)
    labels = [r[0] for r in records]
    x = np.arange(len(records))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.1, 2.6))
    bar_kw = {"edgecolor": "#FFFFFF", "linewidth": 0.45, "zorder": 3}
    ax.bar(x - width, [r[1] for r in records], width, label="No GraphContext", color=ABLATION_COLORS["blue"], **bar_kw)
    ax.bar(x, [r[2] for r in records], width, label="No C++ graph cache", color=ABLATION_COLORS["teal"], **bar_kw)
    vals = [r[3] if r[3] is not None else np.nan for r in records]
    ax.bar(x + width, vals, width, label="No device CSR cache", color=ABLATION_COLORS["sand"], **bar_kw)
    style_ablation_axis(ax, log=True)
    ax.axhline(1.0, color="#73808C", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.set_ylabel("Slowdown over EGGPU")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right")
    ax.legend(ncol=3, frameon=False, loc="upper right", handlelength=1.1, columnspacing=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"ablation_workflow_reuse.{ext}")
    plt.close(fig)


def collect_return_ratios(abl_dir: Path):
    values: defaultdict[str, list[float]] = defaultdict(list)
    for path in abl_dir.glob("return_*.csv"):
        tmp: dict[tuple[str, str, str], float] = {}
        with path.open(newline="") as fp:
            for row in csv.DictReader(fp):
                if row.get("status") != "ok":
                    continue
                if row.get("metric") in {"lazy_call_e2e", "eager_equivalent_e2e"}:
                    tmp[(row["dataset"], row["function"], row["metric"])] = float(row["value"])
        for dataset, function, metric in list(tmp.keys()):
            if metric != "lazy_call_e2e":
                continue
            lazy = tmp[(dataset, function, "lazy_call_e2e")]
            eager = tmp.get((dataset, function, "eager_equivalent_e2e"))
            if lazy and eager:
                values[function].append(eager / lazy)
    return values


def emit_return_ablation_figure(abl_dir: Path, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    values = collect_return_ratios(abl_dir)
    funcs = [f for f in ["PageRank", "MST", "LCC", "SSSP"] if values.get(f)]
    if not funcs:
        return
    means = [float(np.exp(np.mean(np.log(values[f])))) for f in funcs]
    fig, ax = plt.subplots(figsize=(3.6, 2.25))
    ax.bar(
        funcs,
        means,
        color=[ABLATION_COLORS["blue"], ABLATION_COLORS["teal"], ABLATION_COLORS["sand"], ABLATION_COLORS["green"]][: len(funcs)],
        width=0.62,
        edgecolor="#FFFFFF",
        linewidth=0.5,
        zorder=3,
    )
    style_ablation_axis(ax)
    ax.axhline(1.0, color="#73808C", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.set_ylabel("Eager return / lazy return")
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=7, color="#334155")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"ablation_return_path.{ext}")
    plt.close(fig)


def emit_layout_ablation_figure(abl_dir: Path, out_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    records = []
    for path in sorted(abl_dir.glob("layout_*.csv")):
        dataset = path.name[len("layout_") : -4]
        vals: dict[tuple[str, str], float] = {}
        with path.open(newline="") as fp:
            for row in csv.DictReader(fp):
                if row.get("status") != "ok":
                    continue
                if row.get("function") in {"COO", "CSR"} and row.get("metric") in {"host_storage_mb", "degree_seconds"}:
                    vals[(row["function"], row["metric"])] = float(row["value"])
        coo_mem = vals.get(("COO", "host_storage_mb"))
        csr_mem = vals.get(("CSR", "host_storage_mb"))
        coo_deg = vals.get(("COO", "degree_seconds"))
        csr_deg = vals.get(("CSR", "degree_seconds"))
        if coo_mem and csr_mem and coo_deg and csr_deg:
            records.append((dataset, coo_mem / csr_mem, coo_deg / csr_deg))
    if not records:
        return
    records.sort(key=lambda x: x[1], reverse=True)
    labels = [r[0] for r in records]
    x = np.arange(len(records))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), sharex=True)
    bar_kw = {"edgecolor": "#FFFFFF", "linewidth": 0.45, "zorder": 3}
    axes[0].bar(x, [r[1] for r in records], color=ABLATION_COLORS["blue"], width=0.72, **bar_kw)
    style_ablation_axis(axes[0])
    axes[0].axhline(1.0, color="#73808C", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    axes[0].set_ylabel("COO / CSR storage")
    axes[1].bar(x, [r[2] for r in records], color=ABLATION_COLORS["teal"], width=0.72, **bar_kw)
    style_ablation_axis(axes[1])
    axes[1].axhline(1.0, color="#73808C", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    axes[1].set_ylabel("COO / CSR degree time")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=38, ha="right")
    fig.tight_layout(w_pad=1.3)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"ablation_csr_layout.{ext}")
    plt.close(fig)


def geomean(values: list[float]) -> float | None:
    vals = [v for v in values if v and v > 0 and math.isfinite(v)]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def emit_ablation_summary_table(abl_dir: Path, out_dir: Path):
    workflow_ratios: dict[str, list[float]] = {
        "No GraphContext": [],
        "No C++ graph cache": [],
        "No device CSR cache": [],
        "No adaptive policy": [],
    }
    for full_path in sorted(abl_dir.glob("workflow_*_full.csv")):
        dataset = full_path.name[len("workflow_") : -len("_full.csv")]
        full = read_ablation_metric(full_path, "e2e")
        if not full:
            continue
        mapping = {
            "No GraphContext": abl_dir / f"workflow_{dataset}_no_graph_context.csv",
            "No C++ graph cache": abl_dir / f"workflow_{dataset}_no_cpp_graph_cache.csv",
            "No device CSR cache": abl_dir / f"workflow_{dataset}_no_device_csr_cache.csv",
            "No adaptive policy": abl_dir / f"workflow_{dataset}_no_adaptive_policy.csv",
        }
        for label, path in mapping.items():
            value = read_ablation_metric(path, "e2e")
            if value:
                workflow_ratios[label].append(value / full)

    return_ratios = collect_return_ratios(abl_dir)
    ret_values = []
    for values in return_ratios.values():
        ret_values.extend(values)

    storage_values = []
    degree_values = []
    for path in sorted(abl_dir.glob("layout_*.csv")):
        vals: dict[tuple[str, str], float] = {}
        with path.open(newline="") as fp:
            for row in csv.DictReader(fp):
                if row.get("status") != "ok":
                    continue
                if row.get("function") in {"COO", "CSR"} and row.get("metric") in {"host_storage_mb", "degree_seconds"}:
                    vals[(row["function"], row["metric"])] = float(row["value"])
        coo_mem = vals.get(("COO", "host_storage_mb"))
        csr_mem = vals.get(("CSR", "host_storage_mb"))
        coo_deg = vals.get(("COO", "degree_seconds"))
        csr_deg = vals.get(("CSR", "degree_seconds"))
        if coo_mem and csr_mem:
            storage_values.append(coo_mem / csr_mem)
        if coo_deg and csr_deg:
            degree_values.append(coo_deg / csr_deg)

    rows = []
    for label, values in workflow_ratios.items():
        gm = geomean(values)
        if gm is not None:
            rows.append((label, "Workflow E2E slowdown", gm, len(values)))
    gm = geomean(ret_values)
    if gm is not None:
        rows.append(("Lazy return path", "Eager/lazy E2E ratio", gm, len(ret_values)))
    gm = geomean(storage_values)
    if gm is not None:
        rows.append(("CSR layout", "COO/CSR host storage", gm, len(storage_values)))
    gm = geomean(degree_values)
    if gm is not None:
        rows.append(("CSR layout", "COO/CSR degree time", gm, len(degree_values)))

    with (out_dir / "ablation_summary.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Module", "Metric", "Geomean", "Cases"])
        for row in rows:
            writer.writerow(row)

    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablation summary}",
        r"\label{tab:ablation-summary}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Module & Metric & Geomean & Cases \\",
        r"\midrule",
    ]
    for module, metric, value, cases in rows:
        lines.append(
            f"{latex_escape(module)} & {latex_escape(metric)} & {value:.2f}x & {cases} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out_dir / "ablation_summary.tex").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--ablation-dir")
    parser.add_argument("--build-results", help="Optional standalone build_times.csv to use for the build table.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else result_dir / "paper_artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = json.loads((result_dir / "dataset_stats.json").read_text())
    emit_dataset_table(stats, repo_root, out_dir / "paper_table_datasets.tex", out_dir / "paper_table_datasets.csv")

    metric_specs = [
        ("e2e", "End-to-end runtime comparison", "tab:e2e-runtime"),
        ("kernel", "Kernel and algorithm runtime comparison", "tab:kernel-runtime"),
        ("build", "Graph construction time comparison", "tab:graph-load-time"),
    ]
    for metric, caption, label in metric_specs:
        if metric == "build" and args.build_results:
            rows = read_metric_csv(Path(args.build_results).resolve())
        else:
            rows = read_metric_csv(result_dir / f"results_{metric}.csv")
        fallback = read_metric_csv(result_dir / "results_e2e.csv") if metric == "build" else None
        emit_metric_table(
            rows,
            stats,
            out_dir / f"paper_table_{metric}.tex",
            out_dir / f"paper_table_{metric}.csv",
            caption,
            label,
            fallback_rows=fallback,
            na_baselines={"Gunrock"} if metric == "build" else None,
        )

    if args.ablation_dir:
        ablation_dir = Path(args.ablation_dir).resolve()
        if ablation_dir.exists():
            use_paper_style()
            emit_workflow_ablation_figure(ablation_dir, out_dir)
            emit_return_ablation_figure(ablation_dir, out_dir)
            emit_layout_ablation_figure(ablation_dir, out_dir)
            emit_ablation_summary_table(ablation_dir, out_dir)

    readme = [
        "# Paper Artifacts",
        "",
        "- `paper_table_datasets.tex`: dataset attributes.",
        "- `paper_table_e2e.tex`: end-to-end runtime table.",
        "- `paper_table_kernel.tex`: kernel/algorithm runtime table.",
        "- `paper_table_build.tex`: graph construction/load table.",
        "- `ablation_workflow_reuse.pdf`: cross-function reuse ablation.",
        "- `ablation_return_path.pdf`: lazy return-path ablation.",
        "- `ablation_csr_layout.pdf`: CSR layout ablation.",
        "- `ablation_summary.tex`: compact ablation summary table.",
        "",
        "LaTeX tables use `booktabs`, `longtable`, and `xcolor`.",
        "EGGPU is always the last column and is shaded light blue.",
        "The best runtime in each row is bold, and the second best is underlined.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme))
    print(f"Wrote paper artifacts to {out_dir}")


if __name__ == "__main__":
    main()
