#!/usr/bin/env python3
"""Derive paper-facing performance and resource-efficiency metrics.

The source of truth remains results_long.csv.  This file only derives rates,
host-side overhead, speedups, and normalized memory footprints so plots do not
need to repeat metric-join logic or silently change the benchmark protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CPU_BASELINES = {"easygraph-cpu", "easygraph-cpp", "networkx", "igraph"}
GPU_BASELINES = {"nx-cugraph", "Gunrock"}


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    rows = read_csv(result_dir / "results_long.csv")
    stats = json.loads((result_dir / "dataset_stats.json").read_text())
    stats_by_dataset = {row["name"]: row for row in stats}

    grouped = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("dataset", ""), row.get("function", ""), row.get("baseline", ""))
        grouped.setdefault(key, {})[row.get("metric", "")] = row

    derived = []
    for (dataset, function, baseline), metrics in sorted(grouped.items()):
        stat = stats_by_dataset.get(dataset, {})
        graph_type = str(stat.get("graph_type", ""))
        if graph_type == "directed":
            input_edges = int(stat.get("edges_directed_unique") or stat.get("edge_rows_no_selfloops") or 0)
        else:
            input_edges = int(stat.get("edges_undirected_unique") or 0)
        nodes = int(stat.get("nodes_raw") or 0)

        build = number((metrics.get("build") or {}).get("value"))
        e2e = number((metrics.get("e2e") or {}).get("value"))
        kernel = number((metrics.get("kernel") or {}).get("value"))
        gpu_peak = number((metrics.get("memory_peak_gpu_proc_mb") or {}).get("value"))
        rss_peak = number((metrics.get("memory_peak_rss_mb") or {}).get("value"))
        memory_probe_calls = number((metrics.get("memory_probe_calls") or {}).get("value"))
        memory_probe_window = number((metrics.get("memory_probe_window_seconds") or {}).get("value"))
        memory_monitor_samples = number((metrics.get("memory_monitor_gpu_proc_samples") or {}).get("value"))
        memory_monitor_poll_ms = number((metrics.get("memory_monitor_poll_ms") or {}).get("value"))
        e2e_std = number((metrics.get("e2e") or {}).get("std_value"))
        e2e_rsd = number((metrics.get("e2e") or {}).get("relative_std_percent"))
        e2e_ci = number((metrics.get("e2e") or {}).get("relative_ci95_half_width_percent"))

        host_overhead = None
        kernel_share = None
        if baseline in {"EGGPU", "Gunrock"} and e2e is not None and kernel is not None:
            host_overhead = max(0.0, e2e - kernel)
            kernel_share = safe_ratio(kernel, e2e)

        derived.append(
            {
                "dataset": dataset,
                "graph_type": graph_type,
                "function": function,
                "baseline": baseline,
                "nodes": nodes,
                "normalized_input_edges": input_edges,
                "build_mean_seconds": build,
                "e2e_mean_seconds": e2e,
                "kernel_mean_seconds": kernel,
                "e2e_std_seconds": e2e_std,
                "e2e_relative_std_percent": e2e_rsd,
                "e2e_relative_ci95_half_width_percent": e2e_ci,
                "host_overhead_seconds": host_overhead,
                "kernel_share_percent": None if kernel_share is None else 100.0 * kernel_share,
                "e2e_normalized_input_medges_per_second": safe_ratio(input_edges / 1.0e6, e2e),
                "kernel_normalized_input_medges_per_second": safe_ratio(input_edges / 1.0e6, kernel),
                "gpu_process_tree_peak_mib": gpu_peak,
                "cpu_process_tree_peak_rss_mib": rss_peak,
                "memory_probe_calls": memory_probe_calls,
                "memory_probe_window_seconds": memory_probe_window,
                "memory_monitor_gpu_process_samples": memory_monitor_samples,
                "memory_monitor_poll_ms": memory_monitor_poll_ms,
                "gpu_peak_bytes_per_node": safe_ratio(None if gpu_peak is None else gpu_peak * 1024.0 * 1024.0, nodes),
                "gpu_peak_bytes_per_normalized_input_edge": safe_ratio(None if gpu_peak is None else gpu_peak * 1024.0 * 1024.0, input_edges),
                "timing_measurement_phase": (metrics.get("e2e") or {}).get("measurement_phase", ""),
                "memory_measurement_phase": (metrics.get("memory_peak_gpu_proc_mb") or {}).get("measurement_phase", ""),
            }
        )

    write_csv(result_dir / "performance_efficiency_metrics.csv", derived)

    by_pair = {}
    for row in derived:
        e2e = number(row.get("e2e_mean_seconds"))
        if e2e is None or e2e <= 0:
            continue
        by_pair.setdefault((row["dataset"], row["function"]), {})[row["baseline"]] = e2e

    eggpu_rows = []
    for (dataset, function), values in sorted(by_pair.items()):
        eggpu = values.get("EGGPU")
        if eggpu is None:
            continue
        cpu = [value for baseline, value in values.items() if baseline in CPU_BASELINES]
        gpu = [value for baseline, value in values.items() if baseline in GPU_BASELINES]
        other = [value for baseline, value in values.items() if baseline != "EGGPU"]
        best_cpu = min(cpu) if cpu else None
        best_gpu = min(gpu) if gpu else None
        best_other = min(other) if other else None
        eggpu_rows.append(
            {
                "dataset": dataset,
                "function": function,
                "eggpu_e2e_mean_seconds": eggpu,
                "speedup_vs_best_cpu": safe_ratio(best_cpu, eggpu),
                "speedup_vs_best_gpu_baseline": safe_ratio(best_gpu, eggpu),
                "speedup_vs_best_other": safe_ratio(best_other, eggpu),
                "is_fastest_overall": str(best_other is None or eggpu <= best_other).lower(),
            }
        )
    write_csv(result_dir / "eggpu_pair_efficiency_summary.csv", eggpu_rows)
    print(result_dir / "performance_efficiency_metrics.csv")
    print(result_dir / "eggpu_pair_efficiency_summary.csv")


if __name__ == "__main__":
    main()
