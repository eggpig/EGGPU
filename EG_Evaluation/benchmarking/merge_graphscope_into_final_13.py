#!/usr/bin/env python3
"""Merge qualified GraphScope outcomes into the authoritative final-13 ledger."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FUNCTIONS = [
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

FIXED_13 = [
    "ca-HepTh",
    "LastFM",
    "p2p-Gnutella04",
    "ca-HepPh",
    "email-Enron",
    "ca-CondMat",
    "soc-Epinions1",
    "soc-Slashdot0811",
    "ER-100k",
    "web-NotreDame",
    "com-youtube",
    "com-Orkut",
    "GAP-twitter",
]

CATEGORY = {
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

SUPPORT = {
    "PageRank": ("P", "native PageRank; undirected input requires a bidirected projection"),
    "MST": ("F", "FLASH returns only total MSF weight, not the required forest edge set"),
    "LCC": ("T", "native clustering on the common simple undirected projection"),
    "WCC": ("T", "native weakly connected components"),
    "SCC": ("T", "native FLASH SCC; connected-component equivalent for undirected input"),
    "BFS": ("T", "native FLASH BFS over the aligned source set"),
    "Dijkstra": ("P", "native nonnegative-weight SSSP; no separately named Dijkstra app"),
    "BellmanFord": ("F", "no callable aligned GAE/FLASH Bellman-Ford app"),
    "SSSP": ("T", "native FLASH SSSP over the aligned source set"),
    "KCore": ("P", "native FLASH k-core requires a single-orientation undirected projection"),
    "BC": ("P", "BC-16 composes 16 native source-BC calls"),
    "Closeness": ("P", "exact native all-node closeness; directed outgoing mode requires reversal"),
    "EffectiveSize": ("F", "no aligned callable implementation"),
    "Efficiency": ("F", "no aligned callable implementation"),
    "Constraint": ("F", "no aligned callable implementation"),
    "Hierarchy": ("F", "no aligned callable implementation"),
}

FIELDS = [
    "dataset",
    "function",
    "category",
    "baseline",
    "support_class",
    "execution_status",
    "failure_kind",
    "reason",
    "e2e_paper_seconds",
    "e2e_raw_mean_seconds",
    "e2e_std_seconds",
    "e2e_estimator",
    "validation_status",
    "result_source",
    "sample_count",
    "build_paper_seconds",
    "build_raw_mean_seconds",
    "build_std_seconds",
    "build_estimator",
    "kernel_paper_seconds",
    "kernel_raw_mean_seconds",
    "kernel_std_seconds",
    "kernel_estimator",
    "gpu_peak_mb_mean",
    "gpu_peak_mb_std",
    "memory_sample_count",
    "host_rss_peak_mb_mean",
    "host_rss_peak_mb_std",
    "memory_measurement_window",
    "excluded_observed_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ledger", required=True, type=Path)
    parser.add_argument("--timing-dir", required=True, type=Path)
    parser.add_argument("--memory-dir", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_index(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_csv(path)
    return {
        (row["dataset"], row["function"], row["metric"]): row
        for row in rows
        if row.get("baseline") == "GraphScope"
    }


def float_text(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{numeric:.12g}" if math.isfinite(numeric) else ""


def metric_values(index: dict, dataset: str, function: str, metric: str):
    row = index.get((dataset, function, metric), {})
    if row.get("status") != "ok" or row.get("publishable", "").lower() != "true":
        return "", "", "", "", row
    mean = float_text(row.get("mean_seconds") or row.get("mean_value"))
    std = float_text(row.get("std_seconds") or row.get("std_value"))
    count = row.get("n_valid", "")
    return mean, mean, std, count, row


def classify_failure(timing_row: dict[str, str]) -> tuple[str, str]:
    status = timing_row.get("status", "missing")
    notes = timing_row.get("notes") or timing_row.get("skip_reason") or status
    lowered = notes.lower()
    if status == "timeout" or "timeout" in lowered:
        return "timeout", notes
    if (
        "view is unavailable" in lowered
        or "projection is unavailable" in lowered
        or "dataset-specific aligned simple undirected projection is unavailable"
        in lowered
    ):
        return "graph_semantics_limit", notes
    if status == "unsupported":
        return "unsupported_api", notes
    if "memory" in lowered or "bad_alloc" in lowered or "oom" in lowered:
        return "oom", notes
    if status == "missing":
        return "missing_experiment", notes
    return "execution_error", notes


def main() -> None:
    args = parse_args()
    old_rows = [
        row
        for row in read_csv(args.base_ledger)
        if row.get("baseline") not in {"GraphScope", "SYgraph"}
    ]
    timing = aggregate_index(args.timing_dir / "results_long.csv")
    memory = aggregate_index(args.memory_dir / "results_long.csv")
    validation = {
        (row["dataset"], row["function"]): row
        for row in read_csv(args.validation_csv)
    }
    graphscope_rows = []

    for dataset in FIXED_13:
        for function in FUNCTIONS:
            support, support_reason = SUPPORT[function]
            validation_row = validation.get((dataset, function), {})
            timing_row = timing.get((dataset, function, "e2e"), {})
            reason = f"GraphScope 0.29.0; support={support}; {support_reason}"
            failure_kind = ""
            validation_status = validation_row.get("validation_status", "missing")
            execution_status = "ok"

            if support == "F":
                execution_status = "unsupported_api"
                failure_kind = "unsupported_api"
                validation_status = "unsupported"
            elif timing_row.get("status") != "ok" or timing_row.get(
                "publishable", ""
            ).lower() != "true":
                execution_status, failure_reason = classify_failure(timing_row)
                failure_kind = execution_status
                reason += "; " + failure_reason
            elif validation_status != "pass":
                execution_status = (
                    "semantic_mismatch"
                    if validation_status == "fail"
                    else "validation_inconclusive"
                )
                failure_kind = execution_status
                reason += "; " + validation_row.get("details", validation_status)
            else:
                reason += "; independent validation: " + validation_row.get(
                    "details", "pass"
                )

            e2e = metric_values(timing, dataset, function, "e2e")
            build = metric_values(timing, dataset, function, "build")
            kernel = metric_values(timing, dataset, function, "kernel")
            rss = metric_values(memory, dataset, function, "memory_peak_rss_mb")
            if execution_status != "ok":
                e2e = ("", "", "", "", e2e[4])
                build = ("", "", "", "", build[4])
                kernel = ("", "", "", "", kernel[4])

            graphscope_rows.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "category": CATEGORY[function],
                    "baseline": "GraphScope",
                    "support_class": support,
                    "execution_status": execution_status,
                    "failure_kind": failure_kind,
                    "reason": reason,
                    "e2e_paper_seconds": e2e[0],
                    "e2e_raw_mean_seconds": e2e[1],
                    "e2e_std_seconds": e2e[2],
                    "e2e_estimator": "arithmetic_mean" if e2e[0] else "",
                    "validation_status": validation_status,
                    "result_source": str(args.timing_dir),
                    "sample_count": e2e[3],
                    "build_paper_seconds": build[0],
                    "build_raw_mean_seconds": build[1],
                    "build_std_seconds": build[2],
                    "build_estimator": "arithmetic_mean" if build[0] else "",
                    "kernel_paper_seconds": kernel[0],
                    "kernel_raw_mean_seconds": kernel[1],
                    "kernel_std_seconds": kernel[2],
                    "kernel_estimator": "arithmetic_mean" if kernel[0] else "",
                    "gpu_peak_mb_mean": "",
                    "gpu_peak_mb_std": "",
                    "memory_sample_count": rss[3],
                    "host_rss_peak_mb_mean": rss[0],
                    "host_rss_peak_mb_std": rss[2],
                    "memory_measurement_window": (
                        "isolated_worker_process_full_lifetime" if rss[0] else ""
                    ),
                    "excluded_observed_status": "",
                }
            )

    rows = old_rows + graphscope_rows
    dataset_order = {name: index for index, name in enumerate(FIXED_13)}
    function_order = {name: index for index, name in enumerate(FUNCTIONS)}
    baseline_order = {
        name: index
        for index, name in enumerate(
            [
                "networkx",
                "easygraph-cpu",
                "easygraph-cpp",
                "igraph",
                "GraphScope",
                "nx-cugraph",
                "Gunrock",
                "EGGPU",
            ]
        )
    }
    rows.sort(
        key=lambda row: (
            dataset_order.get(row["dataset"], 999),
            function_order.get(row["function"], 999),
            baseline_order.get(row["baseline"], 999),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_dir / "final_13_cell_outcome_ledger.csv"
    write_csv(ledger_path, rows)

    counts = {}
    for row in graphscope_rows:
        counts[row["execution_status"]] = counts.get(row["execution_status"], 0) + 1
    audit = {
        "datasets": len(FIXED_13),
        "functions": len(FUNCTIONS),
        "baselines": sorted({row["baseline"] for row in rows}),
        "expected_cells": 13 * 16 * 8,
        "actual_cells": len(rows),
        "graphscope_status_counts": counts,
        "graphscope_support_function_counts": {
            label: sum(support == label for support, _ in SUPPORT.values())
            for label in ("T", "P", "F")
        },
        "graphscope_support_cell_counts": {
            label: sum(row["support_class"] == label for row in graphscope_rows)
            for label in ("T", "P", "F")
        },
        "sygraph_rows_removed": sum(
            row.get("baseline") == "SYgraph" for row in read_csv(args.base_ledger)
        ),
    }
    (args.output_dir / "FINAL_13_GRAPHSCOPE_MERGE_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )
    if audit["actual_cells"] != audit["expected_cells"]:
        raise SystemExit(
            f"ledger cardinality mismatch: {audit['actual_cells']} != {audit['expected_cells']}"
        )
    print(ledger_path)


if __name__ == "__main__":
    main()
