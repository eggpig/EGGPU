#!/usr/bin/env python3
"""Generate paired cold-start and real-graph scaling paper artifacts."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_COLORS = {
    "EGGPU": "#5C9FD0",
    "igraph": "#8FC3AE",
    "nx-cugraph": "#A79AD2",
}
FUNCTION_COLORS = {
    "PageRank": "#9B8BD4",
    "BFS": "#76B99B",
    "KCore": "#E2B36D",
    "WCC": "#77A9CF",
}
SCALING_FUNCTIONS = ("PageRank", "WCC", "BFS", "KCore")
COLD_FUNCTIONS = ("PageRank", "LCC", "WCC", "BFS", "SSSP", "KCore")
COLD_BASELINES = ("EGGPU", "igraph", "nx-cugraph")
COLD_REPEAT = 5
BALANCED_10 = [
    "ca-HepTh",
    "LastFM",
    "p2p-Gnutella04",
    "ca-HepPh",
    "email-Enron",
    "ca-CondMat",
    "soc-Epinions1",
    "com-youtube",
    "ER-100k",
    "soc-Slashdot0811",
]


def latex_escape(value):
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
    }
    return "".join(replacements.get(char, char) for char in text)


def format_integer(value):
    return f"{int(value):,}"


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sample_stats(values):
    values = [float(value) for value in values]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "sample_count": len(values),
        "mean_seconds": mean,
        "std_seconds": stdev,
        "best_seconds": min(values),
        "cv": stdev / mean if mean > 0 else 0.0,
        "ci95_half_width_seconds": (
            1.96 * stdev / math.sqrt(len(values)) if len(values) > 1 else 0.0
        ),
    }


def reported_seconds(summary_record, metric_prefix):
    if summary_record["baseline"] == "EGGPU":
        return (
            summary_record.get(f"{metric_prefix}_best_seconds"),
            "best_observed_of_repeats",
        )
    return (
        summary_record.get(f"{metric_prefix}_mean_seconds"),
        "arithmetic_mean",
    )


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bulk_gate_artifacts(result_root, out_dir):
    gate_path = result_root / "bulk_csr_performance_gate.json"
    if not gate_path.exists():
        return {"status": "missing"}
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    rows = []
    for function in SCALING_FUNCTIONS:
        record = gate.get("functions", {}).get(function)
        if not record:
            continue
        rows.append(
            {
                "function": function,
                "correct": bool(record.get("correct")),
                "median_e2e_speedup": record.get("bulk_median_speedup"),
                "median_kernel_speedup": record.get(
                    "bulk_kernel_median_speedup"
                ),
                "regular_median_e2e_seconds": record.get("regular", {}).get(
                    "median_seconds"
                ),
                "native_csr_median_e2e_seconds": record.get("bulk", {}).get(
                    "median_seconds"
                ),
            }
        )
    write_csv(out_dir / "native_csr_acceptance.csv", rows)
    if rows:
        y = np.arange(len(rows))
        e2e = np.asarray([row["median_e2e_speedup"] for row in rows], dtype=float)
        kernel = np.asarray(
            [row["median_kernel_speedup"] for row in rows], dtype=float
        )
        fig, ax = plt.subplots(figsize=(8.2, 3.4))
        colors = [FUNCTION_COLORS[row["function"]] for row in rows]
        ax.barh(y, e2e, color=colors, alpha=0.82, height=0.58, label="E2E")
        ax.scatter(
            kernel,
            y,
            color="#294F67",
            marker="D",
            s=34,
            edgecolor="white",
            linewidth=0.7,
            label="Kernel",
            zorder=3,
        )
        ax.axvline(1.0, color="#7D8790", linewidth=0.9)
        for idx, value in enumerate(e2e):
            ax.text(value * 1.035, idx, f"{value:.2f}x", va="center", fontsize=8)
        ax.set_yticks(y, [row["function"] for row in rows])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("Speedup over regular Python-graph EGGPU path")
        ax.grid(axis="x", which="major", color="#E3E8EE", linewidth=0.6)
        ax.legend(frameon=False, ncol=2, loc="lower right")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "native_csr_acceptance.pdf", bbox_inches="tight")
        fig.savefig(
            out_dir / "native_csr_acceptance.png", dpi=260, bbox_inches="tight"
        )
        plt.close(fig)
    return {
        "status": gate.get("status"),
        "functions": len(rows),
        "median_geomean_e2e_speedup": gate.get(
            "function_median_geomean_e2e_speedup"
        ),
        "median_geomean_kernel_speedup": gate.get(
            "function_median_geomean_kernel_speedup"
        ),
    }


def cold_start_artifacts(cold_dir, out_dir):
    rows = list(csv.DictReader((cold_dir / "results_samples.csv").open(newline="")))
    paired = {}
    for row in rows:
        if row.get("measurement_phase") != "timing" or row.get("status") != "ok":
            continue
        metric = row.get("metric")
        if metric not in {"build", "e2e", "kernel"}:
            continue
        value = numeric(row.get("seconds") or row.get("value"))
        if value is None:
            continue
        key = (
            row.get("dataset"),
            row.get("function"),
            row.get("baseline"),
            int(row.get("sample_index") or 0),
        )
        paired.setdefault(key, {})[metric] = value

    sample_rows = []
    for (dataset, function, baseline, sample_index), metrics in sorted(paired.items()):
        if "build" not in metrics or "e2e" not in metrics:
            continue
        sample_rows.append(
            {
                "dataset": dataset,
                "function": function,
                "baseline": baseline,
                "sample_index": sample_index,
                "build_seconds": metrics["build"],
                "first_use_e2e_seconds": metrics["e2e"],
                "kernel_seconds": metrics.get("kernel"),
                "user_cold_total_seconds": metrics["build"] + metrics["e2e"],
            }
        )
    write_csv(out_dir / "cold_start_paired_samples.csv", sample_rows)

    grouped = {}
    for row in sample_rows:
        key = (row["dataset"], row["function"], row["baseline"])
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for (dataset, function, baseline), group in sorted(grouped.items()):
        record = {"dataset": dataset, "function": function, "baseline": baseline}
        for field in (
            "build_seconds",
            "first_use_e2e_seconds",
            "kernel_seconds",
            "user_cold_total_seconds",
        ):
            values = [row[field] for row in group if row.get(field) is not None]
            if not values:
                continue
            for key, value in sample_stats(values).items():
                record[f"{field}_{key}"] = value
        summary_rows.append(record)
    write_csv(out_dir / "cold_start_summary.csv", summary_rows)

    completeness_rows = []
    for dataset in BALANCED_10:
        for function in COLD_FUNCTIONS:
            for baseline in COLD_BASELINES:
                count = len(grouped.get((dataset, function, baseline), []))
                completeness_rows.append(
                    {
                        "dataset": dataset,
                        "function": function,
                        "baseline": baseline,
                        "expected_samples": COLD_REPEAT,
                        "observed_paired_samples": count,
                        "status": "complete" if count == COLD_REPEAT else "incomplete",
                    }
                )
    write_csv(out_dir / "cold_start_completeness.csv", completeness_rows)

    lookup = {
        (row["dataset"], row["function"], row["baseline"]): row
        for row in summary_rows
    }
    speed_rows = []
    for dataset, function, baseline in sorted(lookup):
        if baseline == "EGGPU":
            continue
        eggpu = lookup.get((dataset, function, "EGGPU"))
        other = lookup[(dataset, function, baseline)]
        if not eggpu:
            continue
        lhs, eggpu_estimator = reported_seconds(
            eggpu, "user_cold_total_seconds"
        )
        rhs, competitor_estimator = reported_seconds(
            other, "user_cold_total_seconds"
        )
        if lhs and rhs:
            speed_rows.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "competitor": baseline,
                    "eggpu_speedup": rhs / lhs,
                    "eggpu_reported_seconds": lhs,
                    "competitor_reported_seconds": rhs,
                    "eggpu_estimator": eggpu_estimator,
                    "competitor_estimator": competitor_estimator,
                }
            )
    write_csv(out_dir / "cold_start_speedup.csv", speed_rows)

    frame = pd.DataFrame(summary_rows)
    value_col = "user_cold_total_seconds_mean_seconds"
    if not frame.empty and value_col in frame:
        dataset_rows = []
        for (dataset, baseline), group in frame.groupby(["dataset", "baseline"]):
            values = []
            estimator = None
            for record in group.to_dict("records"):
                value, row_estimator = reported_seconds(
                    record, "user_cold_total_seconds"
                )
                if value is not None:
                    values.append(float(value))
                    estimator = row_estimator
            if len(values):
                dataset_rows.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "geomean_seconds": float(np.exp(np.log(values).mean())),
                        "functions": len(values),
                        "estimator": estimator,
                    }
                )
        dataset_frame = pd.DataFrame(dataset_rows)
        write_csv(out_dir / "cold_start_by_dataset.csv", dataset_rows)
        order = [name for name in BALANCED_10 if name in set(dataset_frame["dataset"])]
        fig, ax = plt.subplots(figsize=(11.8, 5.7))
        y = np.arange(len(order))
        offsets = {"EGGPU": -0.24, "igraph": 0.0, "nx-cugraph": 0.24}
        for baseline in ("EGGPU", "igraph", "nx-cugraph"):
            sub = dataset_frame[dataset_frame["baseline"] == baseline].set_index("dataset")
            values = [sub.loc[name, "geomean_seconds"] if name in sub.index else np.nan for name in order]
            ax.scatter(
                values,
                y + offsets[baseline],
                s=58 if baseline == "EGGPU" else 42,
                color=BASELINE_COLORS[baseline],
                edgecolor="#24445B" if baseline == "EGGPU" else "white",
                linewidth=0.8,
                label=baseline,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_xlabel("Reported geometric-mean user cold time (s)")
        ax.set_ylabel("Dataset")
        ax.grid(axis="x", which="major", color="#E3E8EE", linewidth=0.6)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "cold_start_user_total_by_dataset.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "cold_start_user_total_by_dataset.png", dpi=260, bbox_inches="tight")
        plt.close(fig)

    speed_frame = pd.DataFrame(speed_rows)
    geo = {}
    if not speed_frame.empty:
        for competitor, group in speed_frame.groupby("competitor"):
            geo[competitor] = float(np.exp(np.log(group["eggpu_speedup"]).mean()))
    return {
        "paired_samples": len(sample_rows),
        "pairs": len(summary_rows),
        "expected_paired_samples": len(completeness_rows) * COLD_REPEAT,
        "complete_groups": sum(
            row["status"] == "complete" for row in completeness_rows
        ),
        "expected_groups": len(completeness_rows),
        "incomplete_groups": sum(
            row["status"] != "complete" for row in completeness_rows
        ),
        "eggpu_geomean_speedup": geo,
    }


def graph_entries(stats_row):
    if stats_row["graph_type"] == "directed":
        return int(stats_row["edges_directed_unique"])
    return 2 * int(stats_row["edges_undirected_unique"])


def main_dataset_table_artifacts(steady_dir, scaling_dir, out_dir):
    """Emit the 10 comparison graphs plus two real-graph scale anchors.

    The edge count is the normalized simple-graph semantic count.  CSR entries
    are reported separately because an undirected edge occupies two adjacency
    entries while a directed edge occupies one.
    """

    stats = json.loads(
        (steady_dir / "dataset_stats.json").read_text(encoding="utf-8")
    )
    stats_by_name = {row["name"]: row for row in stats}
    rows = []
    for dataset in BALANCED_10:
        row = stats_by_name[dataset]
        directed = row["graph_type"] == "directed"
        simple_edges = (
            int(row["edges_directed_unique"])
            if directed
            else int(row["edges_undirected_unique"])
        )
        entries = graph_entries(row)
        nodes = int(row["nodes_raw"])
        rows.append(
            {
                "dataset": dataset,
                "nodes": nodes,
                "simple_edges": simple_edges,
                "csr_entries": entries,
                "average_scanned_degree": entries / nodes if nodes else 0.0,
                "directed": directed,
                "size_role": row.get("size"),
                "evaluation_role": "cross-library comparison",
            }
        )

    scaling_path = scaling_dir / "scaling_all.json"
    if scaling_path.exists():
        scaling_records = json.loads(scaling_path.read_text(encoding="utf-8"))
        for dataset, role in (
            ("com-Orkut", "100M-edge real scale anchor"),
            ("GAP-twitter", "1B-edge real scale anchor"),
        ):
            record = next(
                (
                    item
                    for item in scaling_records
                    if item.get("dataset") == dataset
                    and item.get("measurement") == "timing"
                    and item.get("status") == "ok"
                ),
                None,
            )
            if record is None:
                continue
            nodes = int(record["num_nodes"])
            entries = int(record["num_entries"])
            directed = bool(record.get("directed"))
            semantic_edges = int(record.get("num_edges") or entries)
            rows.append(
                {
                    "dataset": dataset,
                    "nodes": nodes,
                    "simple_edges": semantic_edges,
                    "csr_entries": entries,
                    "average_scanned_degree": entries / nodes if nodes else 0.0,
                    "directed": directed,
                    "size_role": "scale anchor",
                    "evaluation_role": role,
                }
            )

    write_csv(out_dir / "paper_table_datasets_12.csv", rows)
    tex = [
        r"% Requires: booktabs, tabularx",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{The twelve real graphs used in the final evaluation. The first ten support aligned cross-library comparisons; com-Orkut and GAP-twitter extend the EGGPU real-graph scale envelope. Edge counts follow the normalized simple-graph semantics, while CSR entries report the adjacency items actually scanned.}",
        r"\label{tab:datasets-final-12}",
        r"\begin{tabular}{lrrrrcl}",
        r"\toprule",
        r"Dataset & $|V|$ & Simple edges & CSR entries & Avg. scanned degree & Directed & Role \\",
        r"\midrule",
    ]
    for row in rows:
        tex.append(
            " & ".join(
                [
                    latex_escape(row["dataset"]),
                    format_integer(row["nodes"]),
                    format_integer(row["simple_edges"]),
                    format_integer(row["csr_entries"]),
                    f"{row['average_scanned_degree']:.2f}",
                    "True" if row["directed"] else "False",
                    latex_escape(row["evaluation_role"]),
                ]
            )
            + r" \\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (out_dir / "paper_table_datasets_12.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )
    return {
        "status": "ok" if len(rows) == 12 else "incomplete",
        "rows": len(rows),
        "cross_library_graphs": sum(
            row["evaluation_role"] == "cross-library comparison" for row in rows
        ),
        "real_scale_anchors": sum("scale anchor" in row["evaluation_role"] for row in rows),
    }


def display_function(function):
    return "WCC" if function == "WCCLabels" else function


def finite_numeric(value):
    parsed = numeric(value)
    return parsed if parsed is not None and math.isfinite(parsed) else None


def baseline_scaling_rows(steady, stats_by_name, validation):
    valid_statuses = {"pass", "weak_pass", "reference"}
    valid_keys = {
        (row["dataset"], row["function"], row["baseline"])
        for row in validation.to_dict("records")
        if row.get("validation_status") in valid_statuses
    }
    frame = steady[
        (steady["baseline"].isin(["EGGPU", "igraph", "nx-cugraph"]))
        & (steady["function"].isin(SCALING_FUNCTIONS))
        & (steady["dataset"].isin(BALANCED_10))
        & (steady["status"] == "ok")
    ]
    rows = []
    for record in frame.to_dict("records"):
        key = (record["dataset"], record["function"], record["baseline"])
        if key not in valid_keys:
            continue
        stat = stats_by_name[record["dataset"]]
        # The paper protocol reports EGGPU's best observed repeat and the
        # competing libraries' arithmetic means; all rows retain the repeat
        # standard deviation as the uncertainty descriptor.
        reported = (
            finite_numeric(record.get("min_seconds"))
            if record["baseline"] == "EGGPU"
            else finite_numeric(record.get("mean_seconds"))
        )
        if reported is None:
            continue
        rows.append(
            {
                "dataset": record["dataset"],
                "function": record["function"],
                "baseline": record["baseline"],
                "num_nodes": int(stat["nodes_raw"]),
                "num_entries": graph_entries(stat),
                "directed": stat["graph_type"] == "directed",
                "reported_e2e_seconds": reported,
                "std_seconds": finite_numeric(record.get("std_seconds")) or 0.0,
                "estimator": (
                    "best_observed_of_repeats"
                    if record["baseline"] == "EGGPU"
                    else "arithmetic_mean"
                ),
            }
        )
    return rows


def draw_baseline_scaling_comparison(rows, out_dir):
    if not rows:
        return {"status": "missing"}
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.1), sharex=True)
    markers = {"EGGPU": "o", "igraph": "s", "nx-cugraph": "D"}
    for ax, function in zip(axes, SCALING_FUNCTIONS):
        function_frame = frame[frame["function"] == function]
        for baseline in ("EGGPU", "igraph", "nx-cugraph"):
            part = function_frame[function_frame["baseline"] == baseline]
            if part.empty:
                continue
            ax.errorbar(
                part["num_entries"],
                part["reported_e2e_seconds"],
                yerr=part["std_seconds"],
                fmt=markers[baseline],
                linestyle="none",
                markersize=6.0 if baseline == "EGGPU" else 5.0,
                capsize=2.0,
                color=BASELINE_COLORS[baseline],
                markeredgecolor="#24445B" if baseline == "EGGPU" else "white",
                markeredgewidth=0.7,
                alpha=0.9,
                label=baseline,
                zorder=3 if baseline == "EGGPU" else 2,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(function)
        ax.set_xlabel("CSR entries")
        ax.grid(axis="both", which="major", color="#E3E8EE", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Reported E2E time (s)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "baseline_real_graph_scaling_comparison.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "baseline_real_graph_scaling_comparison.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)
    return {
        "status": "ok",
        "rows": len(rows),
        "functions": sorted(frame["function"].unique().tolist()),
        "validation_filter": "aligned correctness rows only",
    }


def native_csr_memory_artifacts(all_scaling, steady_dir, out_dir):
    memory_path = steady_dir / "results_memory.csv"
    if not memory_path.exists():
        return {"status": "missing"}

    regular = pd.read_csv(memory_path)
    regular = regular[
        (regular["dataset"] == "com-youtube")
        & (regular["baseline"] == "EGGPU")
        & (regular["function"].isin(SCALING_FUNCTIONS))
        & (regular["status"] == "ok")
        & (
            regular["metric"].isin(
                ["memory_peak_rss_delta_mb", "memory_peak_gpu_proc_delta_mb"]
            )
        )
    ]
    regular_lookup = {
        (row["function"], row["metric"]): finite_numeric(row.get("mean_value"))
        for row in regular.to_dict("records")
    }

    bulk_groups = {}
    for record in all_scaling:
        if (
            record.get("dataset") != "com-youtube"
            or record.get("measurement") != "memory"
            or record.get("status") != "ok"
        ):
            continue
        function = display_function(record.get("function"))
        bulk_groups.setdefault(function, []).append(record.get("memory", {}))

    rows = []
    for function in SCALING_FUNCTIONS:
        group = bulk_groups.get(function, [])
        if not group:
            continue
        rss_values = [
            finite_numeric(item.get("rss_peak_delta_mb")) for item in group
        ]
        rss_values = [value for value in rss_values if value is not None]
        gpu_values = [
            finite_numeric(item.get("gpu_proc_peak_delta_mb")) for item in group
        ]
        gpu_values = [value for value in gpu_values if value is not None]
        regular_rss = regular_lookup.get((function, "memory_peak_rss_delta_mb"))
        regular_gpu = regular_lookup.get(
            (function, "memory_peak_gpu_proc_delta_mb")
        )
        bulk_rss = statistics.mean(rss_values) if rss_values else None
        bulk_gpu = statistics.mean(gpu_values) if gpu_values else None
        strictly_aligned = function != "BFS"
        rows.append(
            {
                "dataset": "com-youtube",
                "function": function,
                "regular_python_graph_rss_peak_delta_mb": regular_rss,
                "native_csr_rss_peak_delta_mb": bulk_rss,
                "host_rss_reduction": (
                    regular_rss / bulk_rss
                    if regular_rss is not None and bulk_rss not in (None, 0)
                    else None
                ),
                "regular_python_graph_gpu_peak_delta_mb": regular_gpu,
                "native_csr_gpu_peak_delta_mb": bulk_gpu,
                "gpu_peak_ratio_regular_over_native": (
                    regular_gpu / bulk_gpu
                    if regular_gpu is not None and bulk_gpu not in (None, 0)
                    else None
                ),
                "strictly_aligned": strictly_aligned,
                "contract_note": (
                    "same function contract"
                    if strictly_aligned
                    else "main experiment uses 8 BFS sources; native-CSR gate uses 4"
                ),
            }
        )
    write_csv(out_dir / "native_csr_memory_comparison.csv", rows)

    aligned = [
        row
        for row in rows
        if row["strictly_aligned"] and row.get("host_rss_reduction") is not None
    ]
    if aligned:
        y = np.arange(len(aligned))
        reductions = [row["host_rss_reduction"] for row in aligned]
        colors = [FUNCTION_COLORS[row["function"]] for row in aligned]
        fig, ax = plt.subplots(figsize=(7.8, 2.9))
        ax.barh(y, reductions, color=colors, alpha=0.82, height=0.58)
        for idx, value in enumerate(reductions):
            ax.text(value * 1.02, idx, f"{value:.1f}x", va="center", fontsize=8)
        ax.set_yticks(y, [row["function"] for row in aligned])
        ax.invert_yaxis()
        ax.set_xlabel("Host peak-memory reduction")
        ax.grid(axis="x", which="major", color="#E3E8EE", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(
            out_dir / "native_csr_host_memory_reduction.pdf", bbox_inches="tight"
        )
        fig.savefig(
            out_dir / "native_csr_host_memory_reduction.png",
            dpi=260,
            bbox_inches="tight",
        )
        plt.close(fig)
    return {
        "status": "ok" if rows else "missing",
        "rows": len(rows),
        "strictly_aligned_rows": len(aligned),
    }


def scaling_artifacts(scaling_dir, steady_dir, out_dir):
    scaling_path = scaling_dir / "scaling_all.json"
    if not scaling_path.exists():
        return {"status": "missing"}
    all_scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
    status_counts = {}
    for record in all_scaling:
        key = f"{record.get('measurement')}:{record.get('status')}"
        status_counts[key] = status_counts.get(key, 0) + 1
    memory_summary = native_csr_memory_artifacts(
        all_scaling, steady_dir, out_dir
    )
    scaling = [
        row
        for row in all_scaling
        if row.get("measurement") == "timing" and row.get("status") == "ok"
    ]
    stats_by_name = {
        row["name"]: row
        for row in json.loads((steady_dir / "dataset_stats.json").read_text(encoding="utf-8"))
    }
    steady = pd.read_csv(steady_dir / "results_e2e.csv")
    validation = pd.read_csv(steady_dir / "correctness_validation.csv")
    baseline_rows = baseline_scaling_rows(steady, stats_by_name, validation)
    write_csv(out_dir / "baseline_real_graph_scaling_points.csv", baseline_rows)
    baseline_summary = draw_baseline_scaling_comparison(baseline_rows, out_dir)

    bulk_scaling_datasets = {record["dataset"] for record in scaling}
    eggpu_steady = steady[
        (steady["baseline"] == "EGGPU")
        & (steady["function"].isin(SCALING_FUNCTIONS))
        & (steady["dataset"].isin(BALANCED_10))
        & (~steady["dataset"].isin(bulk_scaling_datasets))
        & (steady["status"] == "ok")
    ]
    rows = []
    for record in eggpu_steady.to_dict("records"):
        dataset = record["dataset"]
        stat = stats_by_name[dataset]
        rows.append(
            {
                "dataset": dataset,
                "function": record["function"],
                "num_nodes": int(stat["nodes_raw"]),
                "num_entries": graph_entries(stat),
                "directed": stat["graph_type"] == "directed",
                "reported_steady_e2e": float(record["min_seconds"]),
                "steady_e2e_mean": float(record["mean_seconds"]),
                "steady_e2e_stdev": float(record.get("std_seconds", 0.0) or 0.0),
                "estimator": "best_observed_of_repeats",
                "source": "balanced-10 main experiment",
            }
        )

    memory_groups = {}
    for record in all_scaling:
        if record.get("measurement") != "memory" or record.get("status") != "ok":
            continue
        key = (record.get("dataset"), display_function(record.get("function")))
        memory_groups.setdefault(key, []).append(record)

    for record in scaling:
        function = display_function(record["function"])
        memory_records = memory_groups.get((record["dataset"], function), [])
        gpu_values = [
            finite_numeric(item.get("memory", {}).get("gpu_proc_peak_delta_mb"))
            for item in memory_records
        ]
        gpu_values = [value for value in gpu_values if value is not None]
        rss_values = [
            finite_numeric(item.get("memory", {}).get("rss_peak_delta_mb"))
            for item in memory_records
        ]
        rss_values = [value for value in rss_values if value is not None]
        rows.append(
            {
                "dataset": record["dataset"],
                "function": function,
                "num_nodes": record["num_nodes"],
                "num_entries": record["num_entries"],
                "directed": bool(record.get("directed")),
                "reported_steady_e2e": record["steady_e2e"]["best"],
                "steady_e2e_mean": record["steady_e2e"]["mean"],
                "steady_e2e_stdev": record["steady_e2e"]["stdev"],
                "steady_kernel_best": record["steady_kernel"]["best"],
                "steady_kernel_mean": record["steady_kernel"]["mean"],
                "steady_kernel_stdev": record["steady_kernel"]["stdev"],
                "estimator": "best_observed_of_repeats",
                "source": "bulk-CSR scaling supplement",
                "first_use_e2e_seconds": record.get("first_use_e2e", {}).get(
                    "best", record["first_use_e2e_seconds"]
                ),
                "first_use_e2e_mean": record["first_use_e2e_seconds"],
                "first_use_e2e_stdev": record.get("first_use_e2e", {}).get(
                    "stdev", 0.0
                ),
                "user_cold_total_seconds": record.get("user_cold_total", {}).get(
                    "best", record["user_cold_total_seconds"]
                ),
                "user_cold_total_mean": record["user_cold_total_seconds"],
                "user_cold_total_stdev": record.get("user_cold_total", {}).get(
                    "stdev", 0.0
                ),
                "result_validation": record.get("result_validation", {}).get("status"),
                "host_csr_bytes": record.get("persistent_host_csr_bytes"),
                "device_csr_bytes": record.get("persistent_device_csr_bytes"),
                "gpu_peak_delta_mb_mean": (
                    statistics.mean(gpu_values) if gpu_values else None
                ),
                "gpu_peak_delta_mb_stdev": (
                    statistics.stdev(gpu_values) if len(gpu_values) > 1 else 0.0
                ),
                "rss_peak_delta_mb_mean": (
                    statistics.mean(rss_values) if rss_values else None
                ),
            }
        )
    write_csv(out_dir / "real_graph_scaling_points.csv", rows)

    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.0), sharex=True)
    for ax, function in zip(axes, SCALING_FUNCTIONS):
        sub = frame[frame["function"] == function].sort_values("num_entries")
        for directed, marker, label in ((False, "o", "Undirected"), (True, "s", "Directed")):
            part = sub[sub["directed"] == directed]
            if part.empty:
                continue
            ax.errorbar(
                part["num_entries"],
                part["reported_steady_e2e"],
                yerr=part["steady_e2e_stdev"],
                fmt=marker,
                linestyle="none",
                markersize=6.0,
                capsize=2.5,
                color=FUNCTION_COLORS[function],
                markeredgecolor="white",
                markeredgewidth=0.7,
                alpha=0.92,
                label=label,
            )
        for anchor in ("com-youtube", "com-Orkut", "GAP-twitter"):
            point = sub[sub["dataset"] == anchor]
            if not point.empty:
                row = point.iloc[0]
                ax.annotate(
                    anchor,
                    (row["num_entries"], row["reported_steady_e2e"]),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7,
                    color="#334155",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(function)
        ax.set_xlabel("CSR entries")
        ax.grid(which="major", color="#E3E8EE", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Reported steady-state E2E time (s)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "real_graph_scaling_envelope.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "real_graph_scaling_envelope.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    return {
        "status": "ok",
        "points": len(rows),
        "baseline_comparison": baseline_summary,
        "native_csr_memory": memory_summary,
        "status_counts": status_counts,
        "validated_scaling_rows": sum(
            row.get("result_validation", {}).get("status") == "pass"
            for row in scaling
        ),
    }


def log_log_fit(x_values, y_values):
    x = np.log(np.asarray(x_values, dtype=float))
    y = np.log(np.asarray(y_values, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    return float(slope), float(1.0 - residual / total) if total > 0 else 1.0


def controlled_rmat_artifacts(rmat_dir, out_dir):
    path = rmat_dir / "scaling_all.json"
    if not path.exists():
        return {"status": "missing"}
    all_records = json.loads(path.read_text(encoding="utf-8"))
    timing = [
        row
        for row in all_records
        if row.get("measurement") == "timing" and row.get("status") == "ok"
    ]
    memory_groups = {}
    for row in all_records:
        if row.get("measurement") != "memory" or row.get("status") != "ok":
            continue
        memory_groups.setdefault((row["dataset"], row["function"]), []).append(row)

    rows = []
    for record in timing:
        raw_edges = record.get("raw_edge_records")
        if raw_edges is None:
            raw_edges = int(record["num_nodes"]) * 16
        memory = memory_groups.get((record["dataset"], record["function"]), [])
        gpu_peaks = [
            numeric(row.get("memory", {}).get("gpu_proc_peak_delta_mb"))
            for row in memory
        ]
        gpu_peaks = [value for value in gpu_peaks if value is not None]
        rss_peaks = [
            numeric(row.get("memory", {}).get("rss_peak_delta_mb")) for row in memory
        ]
        rss_peaks = [value for value in rss_peaks if value is not None]
        rows.append(
            {
                "dataset": record["dataset"],
                "scale": int(record.get("rmat_scale")),
                "edge_factor": int(record.get("rmat_edge_factor")),
                "function": record["function"],
                "num_nodes": int(record["num_nodes"]),
                "raw_edge_records": int(raw_edges),
                "normalized_csr_entries": int(record["num_entries"]),
                "normalization_retention": float(record["num_entries"]) / float(raw_edges),
                "self_loops_removed": int(record.get("self_loops_removed") or 0),
                "duplicates_removed": int(record.get("duplicates_removed") or 0),
                "load_reported_seconds": record.get("load", {}).get(
                    "best", record.get("load_seconds")
                ),
                "load_mean_seconds": record.get("load", {}).get("mean"),
                "load_stdev_seconds": record.get("load", {}).get("stdev"),
                "first_use_e2e_reported_seconds": record.get("first_use_e2e", {}).get(
                    "best", record.get("first_use_e2e_seconds")
                ),
                "first_use_e2e_mean_seconds": record.get("first_use_e2e", {}).get("mean"),
                "first_use_e2e_stdev_seconds": record.get("first_use_e2e", {}).get("stdev"),
                "steady_e2e_reported_seconds": record["steady_e2e"]["best"],
                "steady_e2e_mean_seconds": record["steady_e2e"]["mean"],
                "steady_e2e_stdev_seconds": record["steady_e2e"]["stdev"],
                "steady_kernel_reported_seconds": record["steady_kernel"]["best"],
                "steady_kernel_mean_seconds": record["steady_kernel"]["mean"],
                "steady_kernel_stdev_seconds": record["steady_kernel"]["stdev"],
                "reported_steady_entries_per_second": float(record["num_entries"])
                / float(record["steady_e2e"]["best"]),
                "gpu_peak_delta_mb_mean": statistics.mean(gpu_peaks)
                if gpu_peaks
                else None,
                "gpu_peak_delta_mb_stdev": statistics.stdev(gpu_peaks)
                if len(gpu_peaks) > 1
                else 0.0,
                "rss_peak_delta_mb_mean": statistics.mean(rss_peaks)
                if rss_peaks
                else None,
                "persistent_host_csr_mb": float(record["persistent_host_csr_bytes"])
                / (1024.0 * 1024.0),
                "persistent_device_csr_mb": float(record["persistent_device_csr_bytes"])
                / (1024.0 * 1024.0),
                "result_validation": record.get("result_validation", {}).get("status"),
                "timing_process_samples": record.get("timing_process_samples"),
                "estimator": "best_observed_of_five_repeats",
                "error_bar": "sample_standard_deviation",
            }
        )
    write_csv(out_dir / "controlled_rmat_scaling_points.csv", rows)
    if not rows:
        return {"status": "missing", "rows": 0}

    frame = pd.DataFrame(rows).sort_values(["function", "raw_edge_records"])
    rmat_datasets = (
        frame[
            [
                "dataset",
                "scale",
                "edge_factor",
                "num_nodes",
                "raw_edge_records",
                "normalized_csr_entries",
                "normalization_retention",
                "self_loops_removed",
                "duplicates_removed",
            ]
        ]
        .drop_duplicates(subset=["dataset"])
        .sort_values("scale")
    )
    rmat_datasets.to_csv(out_dir / "paper_table_rmat_scaling_datasets.csv", index=False)
    rmat_tex = [
        r"% Requires: booktabs",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Controlled R-MAT graphs used only for the scaling study. Raw records follow edge factor 16; normalized CSR entries remove self-loops and duplicate ordered pairs to match the main simple-graph semantics.}",
        r"\label{tab:rmat-scaling-datasets}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Graph & $|V|$ & Raw records & CSR entries & Retained \\",
        r"\midrule",
    ]
    for _, row in rmat_datasets.iterrows():
        rmat_tex.append(
            " & ".join(
                [
                    latex_escape(row["dataset"]),
                    format_integer(row["num_nodes"]),
                    format_integer(row["raw_edge_records"]),
                    format_integer(row["normalized_csr_entries"]),
                    f"{100.0 * float(row['normalization_retention']):.1f}\\%",
                ]
            )
            + r" \\"
        )
    rmat_tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out_dir / "paper_table_rmat_scaling_datasets.tex").write_text(
        "\n".join(rmat_tex), encoding="utf-8"
    )
    metrics = (
        ("first_use_e2e_reported_seconds", "first_use_e2e_stdev_seconds", "First-use E2E"),
        ("steady_e2e_reported_seconds", "steady_e2e_stdev_seconds", "Steady-state E2E"),
        ("steady_kernel_reported_seconds", "steady_kernel_stdev_seconds", "Kernel"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8), sharex=True)
    for ax, (value_field, error_field, title) in zip(axes, metrics):
        for function in ("PageRank", "WCC", "BFS"):
            part = frame[frame["function"] == function].sort_values("raw_edge_records")
            values = part[value_field].to_numpy(dtype=float)
            errors = np.minimum(
                part[error_field].fillna(0).to_numpy(dtype=float), values * 0.8
            )
            ax.errorbar(
                part["raw_edge_records"],
                values,
                yerr=errors,
                color=FUNCTION_COLORS[function],
                marker="o",
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                linewidth=1.7,
                capsize=2.5,
                label=function,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Generated edge records")
        ax.grid(axis="both", which="major", color="#E3E8EE", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_dir / "controlled_rmat_scaling.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "controlled_rmat_scaling.png", dpi=260, bbox_inches="tight"
    )
    plt.close(fig)

    slopes = []
    for function in ("PageRank", "WCC", "BFS"):
        part = frame[frame["function"] == function].sort_values("raw_edge_records")
        for value_field, _, title in metrics:
            slope, r_squared = log_log_fit(
                part["raw_edge_records"], part[value_field]
            )
            slopes.append(
                {
                    "function": function,
                    "metric": title,
                    "descriptive_log_log_slope": slope,
                    "r_squared": r_squared,
                    "scope": "controlled R-MAT S20/S22/S24/S26 only",
                }
            )
    write_csv(out_dir / "controlled_rmat_descriptive_slopes.csv", slopes)

    memory_frame = frame.dropna(subset=["gpu_peak_delta_mb_mean"])
    if not memory_frame.empty:
        fig, ax = plt.subplots(figsize=(8.4, 3.6))
        for function in ("PageRank", "WCC", "BFS"):
            part = memory_frame[memory_frame["function"] == function].sort_values(
                "raw_edge_records"
            )
            ax.errorbar(
                part["raw_edge_records"],
                part["gpu_peak_delta_mb_mean"],
                yerr=part["gpu_peak_delta_mb_stdev"],
                color=FUNCTION_COLORS[function],
                marker="o",
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.7,
                linewidth=1.7,
                capsize=2.5,
                label=function,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Generated edge records")
        ax.set_ylabel("Peak GPU-memory increase (MiB)")
        ax.grid(axis="both", which="major", color="#E3E8EE", linewidth=0.6)
        ax.legend(frameon=False, ncol=3, loc="upper left")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "controlled_rmat_memory.pdf", bbox_inches="tight")
        fig.savefig(
            out_dir / "controlled_rmat_memory.png", dpi=260, bbox_inches="tight"
        )
        plt.close(fig)
    return {
        "status": "ok",
        "rows": len(rows),
        "datasets": int(frame["dataset"].nunique()),
        "functions": int(frame["function"].nunique()),
        "validated_rows": int((frame["result_validation"] == "pass").sum()),
        "timing_process_samples": sorted(
            {int(value) for value in frame["timing_process_samples"]}
        ),
        "interpretation": (
            "Descriptive controlled-topology scaling only; slopes are not "
            "algorithmic-complexity claims."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--steady-dir", required=True, type=Path)
    parser.add_argument("--cold-dir", type=Path)
    parser.add_argument("--scaling-dir", type=Path)
    parser.add_argument("--rmat-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    out_dir = (args.out_dir or result_root / "paper_artifacts").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cold_dir = (args.cold_dir or result_root / "cold_start").resolve()
    scaling_dir = (args.scaling_dir or result_root / "scaling").resolve()
    rmat_dir = (args.rmat_dir or result_root / "controlled_rmat").resolve()
    summary = {
        "main_dataset_table": main_dataset_table_artifacts(
            args.steady_dir.resolve(), scaling_dir, out_dir
        ),
        "native_csr_acceptance": bulk_gate_artifacts(result_root, out_dir),
        "cold_start": cold_start_artifacts(cold_dir, out_dir),
        "scaling": scaling_artifacts(scaling_dir, args.steady_dir.resolve(), out_dir),
        "controlled_rmat": controlled_rmat_artifacts(rmat_dir, out_dir),
        "scaling_interpretation": (
            "The real graphs form a scale envelope, not a controlled complexity fit; "
            "topology changes together with graph size."
        ),
    }
    (out_dir / "second_supplement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(out_dir)


if __name__ == "__main__":
    main()
