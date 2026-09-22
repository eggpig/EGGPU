#!/usr/bin/env python3
"""Derive auditable EGGPU paper evidence from the frozen result pack.

This script is intentionally read-only with respect to source benchmark data. It
normalizes the validation filter used by the final summarizer, recomputes
pairwise timing and memory statistics, and reconstructs processed dataset
statistics from the exact edge-list files consumed by the benchmark runner.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "EG_Evaluation"
DEFAULT_PACK = EVAL / "benchmarking/results/paper_final_result_pack_20260626"
DEFAULT_OUT = REPO / "writing/evidence_20260710"

PROTOCOLS = {
    "A100_raw": "a100_raw_reference",
    "A100_paper_final": "a100_paper_final/paper_final_protocol",
    "RTX3080Ti": "rtx3080ti_final",
}

FAMILY = {
    "PageRank": "Centrality",
    "BC": "Centrality",
    "Closeness": "Centrality",
    "LCC": "Connectivity",
    "WCC": "Connectivity",
    "SCC": "Connectivity",
    "KCore": "Connectivity",
    "MST": "Path and Spanning",
    "BFS": "Path and Spanning",
    "Dijkstra": "Path and Spanning",
    "BellmanFord": "Path and Spanning",
    "SSSP": "Path and Spanning",
    "EffectiveSize": "Structural Holes",
    "Efficiency": "Structural Holes",
    "Constraint": "Structural Holes",
    "Hierarchy": "Structural Holes",
}

WEIGHTED_TASKS = {"MST", "Dijkstra", "BellmanFord", "SSSP"}

CPU_BASELINES = {"igraph", "networkx", "easygraph-cpu", "easygraph-cpp"}
GPU_BASELINES = {"nx-cugraph", "Gunrock"}
BASELINE_VALID = {"pass", "weak_pass", "reference"}
EGGPU_VALID = BASELINE_VALID | {"sampled_pass", "inconclusive_self_reference"}
TIE_REL_TOL = 0.0005


def geomean(values) -> float | None:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return None
    return float(np.exp(np.log(arr).mean()))


def numeric_summary(values) -> dict:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "geomean": None, "median": None, "min": None, "max": None}
    positive = arr[arr > 0]
    return {
        "count": int(arr.size),
        "geomean": geomean(positive),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def validation_map(result_dir: Path) -> pd.DataFrame:
    path = result_dir / "correctness_validation.csv"
    data = pd.read_csv(path)
    return data[["dataset", "function", "baseline", "validation_status"]].drop_duplicates()


def load_metric(result_dir: Path, metric: str, *, validated_only: bool = True) -> pd.DataFrame:
    data = pd.read_csv(result_dir / f"results_{metric}.csv")
    data["value"] = pd.to_numeric(data["seconds"], errors="coerce")
    data["family"] = data["function"].map(FAMILY)
    if not validated_only:
        return data
    merged = data.merge(validation_map(result_dir), on=["dataset", "function", "baseline"], how="left")
    valid_status = np.where(
        merged["baseline"].eq("EGGPU"),
        merged["validation_status"].isin(EGGPU_VALID),
        merged["validation_status"].isin(BASELINE_VALID),
    )
    return merged[valid_status | ~merged["status"].eq("ok")].copy()


def valid_timing_rows(result_dir: Path, metric: str) -> pd.DataFrame:
    data = load_metric(result_dir, metric)
    return data[data["status"].eq("ok") & data["value"].notna() & data["value"].gt(0)].copy()


def coverage_rows(label: str, result_dir: Path, metric: str) -> tuple[list[dict], list[dict]]:
    raw = load_metric(result_dir, metric)
    valid = raw[raw["status"].eq("ok") & raw["value"].notna() & raw["value"].gt(0)].copy()
    target = raw[raw["baseline"].eq("EGGPU")][
        ["dataset_size", "graph_type", "dataset", "function", "family", "status", "value"]
    ].drop_duplicates(["dataset", "function"], keep="first")
    details = []
    for row in target.itertuples(index=False):
        pair = valid[(valid["dataset"] == row.dataset) & (valid["function"] == row.function)]
        egg = pair[pair["baseline"].eq("EGGPU")]
        if egg.empty or pair.empty:
            details.append({
                "protocol": label,
                "metric": metric,
                "dataset_size": row.dataset_size,
                "graph_type": row.graph_type,
                "dataset": row.dataset,
                "function": row.function,
                "family": row.family,
                "eggpu_seconds": None,
                "best_baseline": None,
                "best_seconds": None,
                "ratio": None,
                "is_sota": False,
            })
            continue
        egg_s = float(egg["value"].min())
        best = pair.sort_values("value").iloc[0]
        ratio = egg_s / float(best["value"])
        details.append({
            "protocol": label,
            "metric": metric,
            "dataset_size": row.dataset_size,
            "graph_type": row.graph_type,
            "dataset": row.dataset,
            "function": row.function,
            "family": row.family,
            "eggpu_seconds": egg_s,
            "best_baseline": best["baseline"],
            "best_seconds": float(best["value"]),
            "ratio": ratio,
            "is_sota": bool(ratio <= 1.0 + TIE_REL_TOL),
        })

    detail_df = pd.DataFrame(details)
    summaries = []
    group_specs = [
        ("full", ["protocol", "metric"]),
        ("family", ["protocol", "metric", "family"]),
        ("dataset_size", ["protocol", "metric", "dataset_size"]),
        ("graph_type", ["protocol", "metric", "graph_type"]),
    ]
    for view, columns in group_specs:
        for keys, group in detail_df.groupby(columns, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            out = dict(zip(columns, keys))
            wins = int(group["is_sota"].sum())
            total = int(len(group))
            out.update({
                "view": view,
                "sota_pairs": wins,
                "total_pairs": total,
                "coverage_pct": (100.0 * wins / total) if total else None,
            })
            summaries.append(out)
    return summaries, details


def pairwise_speedups(label: str, result_dir: Path, metric: str) -> list[dict]:
    valid = valid_timing_rows(result_dir, metric)
    egg = valid[valid["baseline"].eq("EGGPU")][
        ["dataset", "function", "family", "dataset_size", "graph_type", "value"]
    ].rename(columns={"value": "eggpu"})
    rows = []
    for baseline in sorted(set(valid["baseline"]) - {"EGGPU"}):
        base = valid[valid["baseline"].eq(baseline)][["dataset", "function", "value"]].rename(
            columns={"value": "baseline_seconds"}
        )
        merged = egg.merge(base, on=["dataset", "function"], how="inner")
        if merged.empty:
            continue
        merged["speedup"] = merged["baseline_seconds"] / merged["eggpu"]
        summary = numeric_summary(merged["speedup"])
        rows.append({
            "protocol": label,
            "metric": metric,
            "comparison": baseline,
            "comparison_kind": "GPU" if baseline in GPU_BASELINES else "CPU",
            "common_pairs": int(len(merged)),
            "speedup_geomean": summary["geomean"],
            "speedup_median": summary["median"],
            "speedup_min": summary["min"],
            "speedup_max": summary["max"],
            "eggpu_fast_or_tie": int((merged["speedup"] >= 1.0 / (1.0 + TIE_REL_TOL)).sum()),
            "eggpu_slower": int((merged["speedup"] < 1.0 / (1.0 + TIE_REL_TOL)).sum()),
        })

    for comparison, baseline_set in (("best_CPU", CPU_BASELINES), ("best_GPU_available", GPU_BASELINES)):
        subset = valid[valid["baseline"].isin(baseline_set)]
        if subset.empty:
            continue
        best = subset.groupby(["dataset", "function"], as_index=False)["value"].min().rename(
            columns={"value": "baseline_seconds"}
        )
        merged = egg.merge(best, on=["dataset", "function"], how="inner")
        merged["speedup"] = merged["baseline_seconds"] / merged["eggpu"]
        summary = numeric_summary(merged["speedup"])
        rows.append({
            "protocol": label,
            "metric": metric,
            "comparison": comparison,
            "comparison_kind": "GPU" if "GPU" in comparison else "CPU",
            "common_pairs": int(len(merged)),
            "speedup_geomean": summary["geomean"],
            "speedup_median": summary["median"],
            "speedup_min": summary["min"],
            "speedup_max": summary["max"],
            "eggpu_fast_or_tie": int((merged["speedup"] >= 1.0 / (1.0 + TIE_REL_TOL)).sum()),
            "eggpu_slower": int((merged["speedup"] < 1.0 / (1.0 + TIE_REL_TOL)).sum()),
        })
    return rows


def grouped_pairwise_speedups(label: str, result_dir: Path, metric: str) -> list[dict]:
    """Common-pair speedups split by function family and benchmark weight policy."""
    valid = valid_timing_rows(result_dir, metric)
    valid["weight_policy"] = np.where(
        valid["function"].isin(WEIGHTED_TASKS), "weighted", "unweighted"
    )
    egg = valid[valid["baseline"].eq("EGGPU")][
        ["dataset", "function", "family", "weight_policy", "value"]
    ].rename(columns={"value": "eggpu"})
    rows = []
    for baseline in sorted(set(valid["baseline"]) - {"EGGPU"}):
        base = valid[valid["baseline"].eq(baseline)][
            ["dataset", "function", "value"]
        ].rename(columns={"value": "baseline_seconds"})
        merged = egg.merge(base, on=["dataset", "function"], how="inner")
        merged["speedup"] = merged["baseline_seconds"] / merged["eggpu"]
        for dimension in ("family", "weight_policy"):
            for group_name, group in merged.groupby(dimension, sort=True):
                summary = numeric_summary(group["speedup"])
                rows.append({
                    "protocol": label,
                    "metric": metric,
                    "baseline": baseline,
                    "dimension": dimension,
                    "group": group_name,
                    "common_pairs": int(len(group)),
                    "speedup_geomean": summary["geomean"],
                    "speedup_median": summary["median"],
                    "speedup_min": summary["min"],
                    "speedup_max": summary["max"],
                })
    return rows


def memory_evidence(label: str, result_dir: Path) -> tuple[list[dict], list[dict]]:
    data = pd.read_csv(result_dir / "results_memory.csv")
    data["value"] = pd.to_numeric(data["seconds"], errors="coerce")
    data["family"] = data["function"].map(FAMILY)
    valid = validation_map(result_dir)
    data = data.merge(valid, on=["dataset", "function", "baseline"], how="left")
    allowed = np.where(
        data["baseline"].eq("EGGPU"),
        data["validation_status"].isin(EGGPU_VALID),
        data["validation_status"].isin(BASELINE_VALID),
    )
    data = data[allowed & data["status"].eq("ok") & data["value"].notna()].copy()
    summaries = []
    maxima = []
    metrics = [
        "memory_peak_gpu_proc_mb",
        "memory_peak_gpu_proc_delta_mb",
        "memory_peak_gpu_mb",
        "memory_peak_rss_mb",
    ]
    for metric in metrics:
        metric_rows = data[data["metric"].eq(metric)]
        for baseline, group in metric_rows.groupby("baseline", sort=True):
            summary = numeric_summary(group["value"])
            summaries.append({
                "protocol": label,
                "metric": metric,
                "baseline": baseline,
                **summary,
                "zero_count": int(group["value"].eq(0).sum()),
            })
            if not group.empty:
                hit = group.sort_values("value", ascending=False).iloc[0]
                maxima.append({
                    "protocol": label,
                    "metric": metric,
                    "baseline": baseline,
                    "dataset": hit["dataset"],
                    "function": hit["function"],
                    "value": float(hit["value"]),
                })

    peak = data[data["metric"].eq("memory_peak_gpu_proc_mb")]
    egg = peak[peak["baseline"].eq("EGGPU")][["dataset", "function", "value"]].rename(
        columns={"value": "eggpu_mb"}
    )
    for baseline in sorted(set(peak["baseline"]) & GPU_BASELINES):
        base = peak[peak["baseline"].eq(baseline)][["dataset", "function", "value"]].rename(
            columns={"value": "baseline_mb"}
        )
        merged = egg.merge(base, on=["dataset", "function"], how="inner")
        if merged.empty:
            continue
        merged = merged[(merged["eggpu_mb"] > 0) & (merged["baseline_mb"] > 0)].copy()
        merged["baseline_over_eggpu"] = merged["baseline_mb"] / merged["eggpu_mb"]
        summary = numeric_summary(merged["baseline_over_eggpu"])
        summaries.append({
            "protocol": label,
            "metric": "memory_peak_gpu_proc_mb_ratio",
            "baseline": baseline,
            **summary,
            "zero_count": 0,
        })
    return summaries, maxima


def memory_by_family(label: str, result_dir: Path) -> list[dict]:
    data = pd.read_csv(result_dir / "results_memory.csv")
    data["value"] = pd.to_numeric(data["seconds"], errors="coerce")
    data["family"] = data["function"].map(FAMILY)
    validation = validation_map(result_dir)
    data = data.merge(validation, on=["dataset", "function", "baseline"], how="left")
    allowed = np.where(
        data["baseline"].eq("EGGPU"),
        data["validation_status"].isin(EGGPU_VALID),
        data["validation_status"].isin(BASELINE_VALID),
    )
    data = data[
        allowed
        & data["status"].eq("ok")
        & data["value"].notna()
        & data["metric"].eq("memory_peak_gpu_proc_mb")
    ].copy()
    rows = []
    for (baseline, family), group in data.groupby(["baseline", "family"], sort=True):
        summary = numeric_summary(group["value"])
        rows.append({
            "protocol": label,
            "baseline": baseline,
            "family": family,
            **summary,
        })
    return rows


def e2e_status_counts(label: str, result_dir: Path) -> list[dict]:
    data = pd.read_csv(result_dir / "results_e2e.csv")
    rows = []
    for (baseline, status), group in data.groupby(["baseline", "status"], dropna=False):
        rows.append({"protocol": label, "baseline": baseline, "status": status, "rows": int(len(group))})
    return rows


def failure_reason_counts(label: str, result_dir: Path) -> list[dict]:
    data = pd.read_csv(result_dir / "results_e2e.csv")
    notes = data["notes"].fillna("").astype(str).str.lower()
    data = data.copy()
    data["reason"] = np.select(
        [
            data["status"].eq("timeout"),
            notes.str.contains(r"out of memory|\boom\b|allocation failed", regex=True),
            data["status"].eq("skipped"),
            data["status"].eq("failed"),
        ],
        ["timeout", "oom_or_allocation", "unsupported_or_unavailable", "other_failure"],
        default="ok",
    )
    return [
        {
            "protocol": label,
            "baseline": baseline,
            "reason": reason,
            "rows": int(len(group)),
        }
        for (baseline, reason), group in data.groupby(["baseline", "reason"], sort=True)
    ]


def cross_device_consistency(protocol_dirs: dict[str, Path]) -> list[dict]:
    a100 = valid_timing_rows(protocol_dirs["A100_raw"], "e2e")
    rtx = valid_timing_rows(protocol_dirs["RTX3080Ti"], "e2e")
    rows = []
    for metric in ("e2e", "kernel"):
        a100 = valid_timing_rows(protocol_dirs["A100_raw"], metric)
        rtx = valid_timing_rows(protocol_dirs["RTX3080Ti"], metric)
        left = a100[a100["baseline"].eq("EGGPU")][
            ["dataset", "function", "family", "value"]
        ].rename(columns={"value": "a100_seconds"})
        right = rtx[rtx["baseline"].eq("EGGPU")][
            ["dataset", "function", "value"]
        ].rename(columns={"value": "rtx_seconds"})
        merged = left.merge(right, on=["dataset", "function"], how="inner")
        for family, group in [("ALL", merged), *list(merged.groupby("family", sort=True))]:
            if len(group) < 2:
                continue
            log_a = np.log(group["a100_seconds"].to_numpy(dtype=float))
            log_r = np.log(group["rtx_seconds"].to_numpy(dtype=float))
            pearson = float(np.corrcoef(log_a, log_r)[0, 1])
            rank_a = pd.Series(log_a).rank(method="average").to_numpy()
            rank_r = pd.Series(log_r).rank(method="average").to_numpy()
            spearman = float(np.corrcoef(rank_a, rank_r)[0, 1])
            ratio = group["rtx_seconds"] / group["a100_seconds"]
            summary = numeric_summary(ratio)
            rows.append({
                "metric": metric,
                "family": family,
                "common_pairs": int(len(group)),
                "log_time_pearson": pearson,
                "rank_spearman": spearman,
                "rtx_over_a100_geomean": summary["geomean"],
                "rtx_over_a100_median": summary["median"],
                "rtx_over_a100_min": summary["min"],
                "rtx_over_a100_max": summary["max"],
            })
    return rows


def load_manifest() -> dict[str, dict]:
    path = EVAL / "datasets/MANIFEST_20260609.tsv"
    if not path.exists():
        return {}
    data = pd.read_csv(path, sep="\t")
    return {str(row.path): row._asdict() for row in data.itertuples(index=False)}


def process_dataset_stats(result_dir: Path) -> list[dict]:
    metadata = json.loads((result_dir / "dataset_stats.json").read_text())
    manifest = load_manifest()
    rows = []
    for item in metadata:
        rel = str(item["path"])
        path = EVAL / rel
        raw_edges = []
        raw_nodes = set()
        with path.open() as handle:
            for line in handle:
                text = line.strip()
                if not text or text[0] in "#%/c":
                    continue
                parts = text.split()
                if len(parts) < 2:
                    continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                raw_nodes.add(u)
                raw_nodes.add(v)
                raw_edges.append((u, v))
        clean_edges_raw = [(u, v) for u, v in raw_edges if u != v]
        clean_nodes = sorted({x for edge in clean_edges_raw for x in edge})
        node_index = {node: idx for idx, node in enumerate(clean_nodes)}
        directed_unique = {(node_index[u], node_index[v]) for u, v in clean_edges_raw}
        undirected_unique = {(min(u, v), max(u, v)) for u, v in directed_unique}
        n = len(clean_nodes)
        graph_type = str(item["graph_type"])
        if graph_type == "directed":
            processed_edges = directed_unique
            out_degree = np.zeros(n, dtype=np.int64)
            in_degree = np.zeros(n, dtype=np.int64)
            for u, v in processed_edges:
                out_degree[u] += 1
                in_degree[v] += 1
            total_degree = out_degree + in_degree
            m = len(processed_edges)
            avg_degree = (m / n) if n else 0.0
            avg_total_degree = (2.0 * m / n) if n else 0.0
            max_degree = int(total_degree.max()) if n else 0
            max_out_degree = int(out_degree.max()) if n else 0
            max_in_degree = int(in_degree.max()) if n else 0
            density = (m / (n * (n - 1))) if n > 1 else 0.0
        else:
            processed_edges = undirected_unique
            degree = np.zeros(n, dtype=np.int64)
            for u, v in processed_edges:
                degree[u] += 1
                degree[v] += 1
            m = len(processed_edges)
            avg_degree = (2.0 * m / n) if n else 0.0
            avg_total_degree = avg_degree
            max_degree = int(degree.max()) if n else 0
            max_out_degree = max_degree
            max_in_degree = max_degree
            density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
        manifest_key = rel.removeprefix("datasets/")
        source = manifest.get(manifest_key, {})
        rows.append({
            "dataset": item["name"],
            "size": item["size"],
            "graph_type": graph_type,
            "raw_nodes": len(raw_nodes),
            "raw_edge_rows": len(raw_edges),
            "raw_self_loops": sum(1 for u, v in raw_edges if u == v),
            "processed_clean_nodes": n,
            "processed_all_vertices": len(raw_nodes),
            "processed_edges": m,
            "processed_directed_unique_edges": len(directed_unique),
            "processed_undirected_unique_edges": len(undirected_unique),
            "average_degree": avg_degree,
            "average_total_degree": avg_total_degree,
            "max_degree": max_degree,
            "max_out_degree": max_out_degree,
            "max_in_degree": max_in_degree,
            "density": density,
            "source_file": rel,
            "source_hint": source.get("source_hint", ""),
            "sha256": source.get("sha256", ""),
            "weight_policy": "deterministic synthetic 1+(src*dst mod |V|) for weighted tasks; otherwise unit",
            "self_loop_policy": "removed before every algorithm",
            "duplicate_policy": "ordered duplicates removed; undirected projections additionally canonicalize unordered pairs",
        })
    return rows


def density_coverage(detail_rows: list[dict], dataset_rows: list[dict]) -> list[dict]:
    """Coverage by dataset-density tertile, with cut points derived from 17 inputs."""
    details = pd.DataFrame(detail_rows)
    datasets = pd.DataFrame(dataset_rows)[["dataset", "density"]].copy()
    q1, q2 = datasets["density"].quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    datasets["density_group"] = np.select(
        [datasets["density"] <= q1, datasets["density"] <= q2],
        ["low", "medium"],
        default="high",
    )
    merged = details.merge(datasets, on="dataset", how="left")
    rows = []
    for (protocol, metric, group_name), group in merged.groupby(
        ["protocol", "metric", "density_group"], sort=True
    ):
        wins = int(group["is_sota"].sum())
        total = int(len(group))
        rows.append({
            "protocol": protocol,
            "metric": metric,
            "view": "density_tertile",
            "density_group": group_name,
            "density_q1": float(q1),
            "density_q2": float(q2),
            "sota_pairs": wins,
            "total_pairs": total,
            "coverage_pct": 100.0 * wins / total if total else None,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    pack = args.pack.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    coverage = []
    detail = []
    speedups = []
    grouped_speedups = []
    memory = []
    memory_maxima = []
    memory_family = []
    statuses = []
    failure_reasons = []
    metadata = {}
    protocol_dirs = {}
    for label, rel in PROTOCOLS.items():
        result_dir = pack / rel
        protocol_dirs[label] = result_dir
        metadata[label] = json.loads((result_dir / "run_metadata.json").read_text())
        for metric in ("e2e", "kernel"):
            summary_rows, detail_rows = coverage_rows(label, result_dir, metric)
            coverage.extend(summary_rows)
            detail.extend(detail_rows)
            speedups.extend(pairwise_speedups(label, result_dir, metric))
            grouped_speedups.extend(grouped_pairwise_speedups(label, result_dir, metric))
        mem_rows, max_rows = memory_evidence(label, result_dir)
        memory.extend(mem_rows)
        memory_maxima.extend(max_rows)
        memory_family.extend(memory_by_family(label, result_dir))
        statuses.extend(e2e_status_counts(label, result_dir))
        failure_reasons.extend(failure_reason_counts(label, result_dir))

    dataset_rows = process_dataset_stats(pack / PROTOCOLS["A100_paper_final"])
    density_rows = density_coverage(detail, dataset_rows)
    coverage.extend(density_rows)
    consistency = cross_device_consistency(protocol_dirs)
    non_sota = [row for row in detail if not row["is_sota"]]

    pd.DataFrame(coverage).to_csv(out / "coverage_recomputed.csv", index=False)
    pd.DataFrame(detail).to_csv(out / "pair_details_recomputed.csv", index=False)
    pd.DataFrame(non_sota).to_csv(out / "non_sota_recomputed.csv", index=False)
    pd.DataFrame(speedups).to_csv(out / "common_pair_speedups.csv", index=False)
    pd.DataFrame(grouped_speedups).to_csv(out / "grouped_common_pair_speedups.csv", index=False)
    pd.DataFrame(memory).to_csv(out / "memory_summary.csv", index=False)
    pd.DataFrame(memory_maxima).to_csv(out / "memory_maxima.csv", index=False)
    pd.DataFrame(memory_family).to_csv(out / "memory_by_family.csv", index=False)
    pd.DataFrame(statuses).to_csv(out / "e2e_status_counts.csv", index=False)
    pd.DataFrame(failure_reasons).to_csv(out / "e2e_failure_reason_counts.csv", index=False)
    pd.DataFrame(consistency).to_csv(out / "cross_device_consistency.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(out / "processed_dataset_statistics.csv", index=False)

    payload = {
        "schema_version": 1,
        "tie_relative_tolerance": TIE_REL_TOL,
        "sources": {label: str((pack / rel).relative_to(REPO)) for label, rel in PROTOCOLS.items()},
        "formulae": {
            "speedup": "baseline_seconds / eggpu_seconds",
            "geomean": "exp(mean(log(x_i))) over positive finite common-pair ratios",
            "sota": "eggpu_seconds / minimum_validated_seconds <= 1 + 0.0005",
            "directed_average_degree": "processed directed edge count / processed clean vertex count",
            "directed_average_total_degree": "2 * processed directed edge count / processed clean vertex count",
            "undirected_average_degree": "2 * processed undirected edge count / processed clean vertex count",
        },
        "coverage": coverage,
        "common_pair_speedups": speedups,
        "grouped_common_pair_speedups": grouped_speedups,
        "memory_summary": memory,
        "memory_maxima": memory_maxima,
        "memory_by_family": memory_family,
        "e2e_status_counts": statuses,
        "e2e_failure_reason_counts": failure_reasons,
        "cross_device_consistency": consistency,
        "datasets": dataset_rows,
        "run_metadata": metadata,
    }
    (out / "complete_evidence_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
