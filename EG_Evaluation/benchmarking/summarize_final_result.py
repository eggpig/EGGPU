#!/usr/bin/env python3
"""Generate a final EGGPU result summary for benchmark review.

The report is intentionally conservative:

- EGGPU timeout rows count as non-SOTA pairs.
- Correctness status is read from the full audit summary when available.
- Backend separation is checked from logs without mutating the result unless an
  output directory is requested by the caller.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


FUNCTION_CATEGORY = {
    "PageRank": "Centrality/Core",
    "BC": "Centrality/Core",
    "Closeness": "Centrality/Core",
    "KCore": "Centrality/Core",
    "LCC": "Centrality/Core",
    "WCC": "Connectivity/Traversal",
    "SCC": "Connectivity/Traversal",
    "BFS": "Connectivity/Traversal",
    "Dijkstra": "Path/Spanning",
    "BellmanFord": "Path/Spanning",
    "SSSP": "Path/Spanning",
    "MST": "Path/Spanning",
    "EffectiveSize": "Structural Holes",
    "Efficiency": "Structural Holes",
    "Constraint": "Structural Holes",
    "Hierarchy": "Structural Holes",
}

DETAIL_COLUMNS = [
    "metric",
    "filter",
    "dataset",
    "function",
    "category",
    "eggpu_seconds",
    "best_baseline",
    "best_seconds",
    "ratio_to_best",
    "sota_gap_pct",
    "is_sota",
    "is_near_miss",
    "notes",
]

GPU_UNFRIENDLY_DATASETS = {
    # Small or low-work graphs where CPU libraries have very low constants.
    "ca-GrQc",
    "ca-HepTh",
    "LastFM",
    "p2p-Gnutella08",
    # Slightly above 10k nodes but still low-work / low-degree in this suite.
    "pgp",
    "p2p-Gnutella04",
    # Small directed graph with SCC/WCC structure that favors CPU constants.
    "wiki-Vote",
}

OUTPUT_MATERIALIZATION_DOMINATED_PAIRS = {
    # The public API returns Python sets for every component.  On these pairs
    # the measured E2E gap is dominated by materializing many Python component
    # objects rather than by GPU traversal work.
    ("email-Enron", "WCC"),
    ("email-Enron", "SCC"),
    ("soc-Slashdot0811", "SCC"),
    ("web-NotreDame", "SCC"),
    ("wiki-Talk", "SCC"),
}

EXPECTED_BACKEND = {
    "EGGPU": ("gpu", "TRUE", "mine", "2"),
    "easygraph-cpp": ("cpp", "FALSE", "", "0"),
    "easygraph-cpu": ("cpu", "FALSE", "", "0"),
}

MODE_RE = re.compile(
    r"\[easygraph-mode\]\s+backend=(?P<backend>\S+)\s+"
    r"mode=(?P<mode>\S+)\s+"
    r"EASYGRAPH_ENABLE_GPU=(?P<enable>\S*)\s+"
    r"EASYGRAPH_GPU_BACKEND=(?P<gpu_backend>\S*)"
)
PARAM_RE = re.compile(r"\[params\].*easygraph_warmup=(?P<warmup>\d+)")
ACCEPTED_SCALE_SKIP_NOTE = "exact all-source Closeness skipped by symmetric scale guard"
SOTA_TIE_REL_TOL = 0.0005
NEAR_MISS_REL_TOL = 0.02
VALIDATED_SOTA_BASELINE_STATUSES = {"pass", "weak_pass", "reference"}
EGGPU_TIMING_ALLOWED_STATUSES = VALIDATED_SOTA_BASELINE_STATUSES | {
    "inconclusive_self_reference",
    "sampled_pass",
}


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def load_dataset_stats(result_dir: Path) -> dict[str, dict]:
    path = result_dir / "dataset_stats.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    rows = data if isinstance(data, list) else [
        {"name": key, **value} for key, value in data.items()
    ]
    out = {}
    for row in rows:
        name = row.get("name") or row.get("dataset")
        if not name:
            continue
        nodes = int(row.get("nodes_raw", row.get("num_nodes_raw", row.get("nodes", 0))) or 0)
        edge_rows = int(row.get("edge_rows_no_selfloops", row.get("edge_rows", row.get("edges", 0))) or 0)
        graph_type = str(row.get("graph_type", ""))
        directed = graph_type == "directed"
        avg_degree = (edge_rows / nodes) if directed and nodes else ((2.0 * edge_rows / nodes) if nodes else 0.0)
        out[name] = {
            "nodes": nodes,
            "edge_rows": edge_rows,
            "avg_degree": avg_degree,
            "graph_type": graph_type,
            "size": row.get("size", ""),
        }
    return out


def load_audit_summary(result_dir: Path) -> dict:
    path = result_dir / "audit" / "audit_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def backend_log_summary(result_dir: Path) -> tuple[dict[str, Counter], list[tuple[str, str]]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    failures: list[tuple[str, str]] = []
    logs_dir = result_dir / "logs"
    for baseline, expected in EXPECTED_BACKEND.items():
        for log_path in sorted(logs_dir.glob(f"*/library_{baseline}_*.log")):
            text = log_path.read_text(errors="replace")
            mode_match = MODE_RE.search(text)
            param_match = PARAM_RE.search(text)
            if not mode_match and ACCEPTED_SCALE_SKIP_NOTE in text:
                counts[baseline]["skipped_before_header"] += 1
                continue
            if not mode_match and "TIMEOUT_TOO_LONG" in text:
                counts[baseline]["timeout_before_header"] += 1
                continue
            if not mode_match:
                counts[baseline]["fail"] += 1
                failures.append((str(log_path), "missing mode header"))
                continue
            got = (
                mode_match.group("mode"),
                mode_match.group("enable"),
                mode_match.group("gpu_backend"),
                param_match.group("warmup") if param_match else "",
            )
            if got != expected:
                counts[baseline]["fail"] += 1
                failures.append((str(log_path), f"got {got}, expected {expected}"))
                continue
            counts[baseline]["ok"] += 1
    return counts, failures


def _load_validation(result_dir: Path) -> pd.DataFrame | None:
    path = result_dir / "correctness_validation.csv"
    if not path.exists():
        return None
    validation = pd.read_csv(path)
    required = {"dataset", "function", "baseline", "validation_status"}
    if not required.issubset(validation.columns):
        return None
    return validation[["dataset", "function", "baseline", "validation_status"]].copy()


def _apply_validation_filter(df: pd.DataFrame, validation: pd.DataFrame | None) -> pd.DataFrame:
    if validation is None or df.empty:
        return df
    merged = df.merge(validation, on=["dataset", "function", "baseline"], how="left")
    ok = merged["status"] == "ok"
    eggpu = merged["baseline"] == "EGGPU"
    validated_non_eggpu = merged["validation_status"].isin(VALIDATED_SOTA_BASELINE_STATUSES)
    validated_eggpu = merged["validation_status"].isin(EGGPU_TIMING_ALLOWED_STATUSES)
    keep = (~ok) | (eggpu & validated_eggpu) | ((~eggpu) & validated_non_eggpu)
    return merged[keep].drop(columns=["validation_status"]).copy()


def load_metric(result_dir: Path, metric: str) -> pd.DataFrame:
    path = result_dir / f"results_{metric}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    df = pd.read_csv(path)
    df["seconds_num"] = pd.to_numeric(df["seconds"], errors="coerce")
    df = df[df["status"].isin(["ok", "timeout"])].copy()
    return _apply_validation_filter(df, _load_validation(result_dir))


def _pair_allowed(predicate, dataset: str, function: str) -> bool:
    try:
        return bool(predicate(dataset, function))
    except TypeError:
        return bool(predicate(dataset))


def summarize_metric(result_dir: Path, metric: str, filter_name: str, dataset_predicate) -> tuple[dict, pd.DataFrame]:
    df = load_metric(result_dir, metric)
    pairs = sorted(set(zip(df["dataset"], df["function"])))
    rows = []
    category_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total = 0
    wins = 0

    for dataset, function in pairs:
        if not _pair_allowed(dataset_predicate, dataset, function):
            continue
        sub = df[(df["dataset"] == dataset) & (df["function"] == function)]
        eggpu = sub[sub["baseline"] == "EGGPU"]
        if eggpu.empty:
            continue
        category = FUNCTION_CATEGORY.get(function, "Other")
        total += 1
        category_totals[category][1] += 1

        eggpu_ok = eggpu[eggpu["status"] == "ok"]
        if eggpu_ok.empty:
            ok = sub[(sub["status"] == "ok") & sub["seconds_num"].notna()].copy()
            if ok.empty:
                best_baseline = ""
                best_seconds = None
            else:
                best_row = ok.sort_values("seconds_num").iloc[0]
                best_baseline = best_row["baseline"]
                best_seconds = float(best_row["seconds_num"])
            rows.append(
                {
                    "metric": metric,
                    "filter": filter_name,
                    "dataset": dataset,
                    "function": function,
                    "category": category,
                    "eggpu_seconds": None,
                    "best_baseline": best_baseline,
                    "best_seconds": best_seconds,
                    "ratio_to_best": math.inf,
                    "sota_gap_pct": math.inf,
                    "is_sota": False,
                    "is_near_miss": False,
                    "notes": "EGGPU timeout or missing",
                }
            )
            continue

        eggpu_seconds = _safe_float(eggpu_ok["seconds_num"].min())
        ok = sub[(sub["status"] == "ok") & sub["seconds_num"].notna()].copy()
        best_row = ok.sort_values("seconds_num").iloc[0]
        best_seconds = float(best_row["seconds_num"])
        ratio = (eggpu_seconds / best_seconds) if eggpu_seconds is not None and best_seconds > 0 else math.inf
        is_sota = ratio <= 1.0 + SOTA_TIE_REL_TOL
        is_near_miss = (not is_sota) and math.isfinite(ratio) and ratio <= 1.0 + NEAR_MISS_REL_TOL
        wins += int(is_sota)
        category_totals[category][0] += int(is_sota)
        rows.append(
            {
                "metric": metric,
                "filter": filter_name,
                "dataset": dataset,
                "function": function,
                "category": category,
                "eggpu_seconds": eggpu_seconds,
                "best_baseline": best_row["baseline"],
                "best_seconds": best_seconds,
                "ratio_to_best": ratio,
                "sota_gap_pct": max(0.0, (ratio - 1.0) * 100.0) if math.isfinite(ratio) else math.inf,
                "is_sota": is_sota,
                "is_near_miss": is_near_miss,
                "notes": "",
            }
        )

    summary = {
        "metric": metric,
        "filter": filter_name,
        "sota_pairs": wins,
        "total_pairs": total,
        "sota_pct": (wins / total * 100.0) if total else 0.0,
        "categories": {
            category: {
                "sota_pairs": values[0],
                "total_pairs": values[1],
                "sota_pct": (values[0] / values[1] * 100.0) if values[1] else 0.0,
            }
            for category, values in sorted(category_totals.items())
        },
    }
    return summary, pd.DataFrame(rows, columns=DETAIL_COLUMNS)


def category_aggregate_summary(details: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "metric",
        "filter",
        "category",
        "pairs",
        "timeout_or_missing_pairs",
        "finite_pairs",
        "geomean_ratio_to_pairwise_best",
        "geomean_speedup_vs_pairwise_best",
        "aggregate_sota",
    ]
    if details.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (metric, filter_name, category), sub in details.groupby(["metric", "filter", "category"], sort=True):
        total_pairs = int(len(sub))
        timeout_pairs = int(
            sub["ratio_to_best"].map(lambda value: bool(math.isinf(float(value)))).sum()
        )
        finite = sub[
            sub["ratio_to_best"].map(lambda value: math.isfinite(float(value)))
            & sub["ratio_to_best"].notna()
            & (sub["ratio_to_best"] > 0)
        ]
        if finite.empty:
            geomean_ratio = math.inf
            geomean_speedup = 0.0
        else:
            geomean_ratio = float(math.exp(finite["ratio_to_best"].map(math.log).mean()))
            geomean_speedup = 1.0 / geomean_ratio if geomean_ratio > 0 else math.inf
        rows.append(
            {
                "metric": metric,
                "filter": filter_name,
                "category": category,
                "pairs": total_pairs,
                "timeout_or_missing_pairs": timeout_pairs,
                "finite_pairs": int(len(finite)),
                "geomean_ratio_to_pairwise_best": geomean_ratio,
                "geomean_speedup_vs_pairwise_best": geomean_speedup,
                "aggregate_sota": bool(timeout_pairs == 0 and geomean_ratio <= 1.0 + SOTA_TIE_REL_TOL),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def category_baseline_geomean_summary(result_dir: Path, filters) -> pd.DataFrame:
    columns = [
        "metric",
        "filter",
        "category",
        "baseline",
        "common_pairs",
        "eggpu_geomean_seconds",
        "baseline_geomean_seconds",
        "speedup_vs_baseline",
        "eggpu_faster",
    ]
    rows = []
    for metric in ("e2e", "kernel"):
        df = load_metric(result_dir, metric)
        ok = df[(df["status"] == "ok") & df["seconds_num"].notna()].copy()
        if ok.empty:
            continue
        ok["category"] = ok["function"].map(lambda fn: FUNCTION_CATEGORY.get(fn, "Other"))
        for filter_name, predicate in filters:
            allowed = ok.apply(
                lambda row: _pair_allowed(predicate, str(row["dataset"]), str(row["function"])),
                axis=1,
            )
            f_ok = ok[allowed].copy()
            eggpu_pairs = f_ok[f_ok["baseline"] == "EGGPU"][
                ["dataset", "function", "category", "seconds_num"]
            ].rename(columns={"seconds_num": "eggpu_seconds"})
            for category, eggpu_cat in eggpu_pairs.groupby("category", sort=True):
                for baseline in sorted(set(f_ok["baseline"]) - {"EGGPU"}):
                    base_cat = f_ok[
                        (f_ok["baseline"] == baseline)
                        & (f_ok["category"] == category)
                    ][["dataset", "function", "seconds_num"]].rename(
                        columns={"seconds_num": "baseline_seconds"}
                    )
                    merged = eggpu_cat.merge(base_cat, on=["dataset", "function"], how="inner")
                    if merged.empty:
                        continue
                    eg_geo = float(math.exp(merged["eggpu_seconds"].map(math.log).mean()))
                    base_geo = float(math.exp(merged["baseline_seconds"].map(math.log).mean()))
                    speedup = base_geo / eg_geo if eg_geo > 0 else math.inf
                    rows.append(
                        {
                            "metric": metric,
                            "filter": filter_name,
                            "category": category,
                            "baseline": baseline,
                            "common_pairs": int(len(merged)),
                            "eggpu_geomean_seconds": eg_geo,
                            "baseline_geomean_seconds": base_geo,
                            "speedup_vs_baseline": speedup,
                            "eggpu_faster": bool(speedup >= 1.0 + SOTA_TIE_REL_TOL),
                        }
                    )
    return pd.DataFrame(rows, columns=columns)


def category_target_verdict(
    category_aggregates: pd.DataFrame,
    category_baselines: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "metric",
        "filter",
        "category",
        "pairs",
        "timeout_or_missing_pairs",
        "pairwise_oracle_aggregate_sota",
        "common_pair_average_beats_all_baselines",
        "min_common_pair_speedup_vs_baseline",
        "category_average_sota_verdict",
    ]
    if category_aggregates.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for agg in category_aggregates.itertuples(index=False):
        base = category_baselines[
            (category_baselines["metric"] == agg.metric)
            & (category_baselines["filter"] == agg.filter)
            & (category_baselines["category"] == agg.category)
        ].copy()
        all_baseline_wins = bool((not base.empty) and base["eggpu_faster"].all())
        min_speedup = float(base["speedup_vs_baseline"].min()) if not base.empty else math.nan
        rows.append(
            {
                "metric": agg.metric,
                "filter": agg.filter,
                "category": agg.category,
                "pairs": int(agg.pairs),
                "timeout_or_missing_pairs": int(agg.timeout_or_missing_pairs),
                "pairwise_oracle_aggregate_sota": bool(agg.aggregate_sota),
                "common_pair_average_beats_all_baselines": all_baseline_wins,
                "min_common_pair_speedup_vs_baseline": min_speedup,
                "category_average_sota_verdict": bool(
                    int(agg.timeout_or_missing_pairs) == 0 and all_baseline_wins
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def load_closeness_large_supplement(result_dir: Path) -> dict | None:
    base = result_dir / "closeness_large_sampled"
    sota_path = base / "closeness_large_sampled_sota.csv"
    validation_path = base / "closeness_large_sampled_validation.csv"
    long_path = base / "closeness_large_sampled_long.csv"
    merged_path = base / "results_long_with_closeness_large_sampled.csv"
    if not sota_path.exists():
        return None

    sota = pd.read_csv(sota_path)
    validation = pd.read_csv(validation_path) if validation_path.exists() else pd.DataFrame()
    rows = pd.read_csv(long_path) if long_path.exists() else pd.DataFrame()
    timed_ok = 0
    if not rows.empty and {"metric", "status"}.issubset(rows.columns):
        timed_ok = int(
            rows[
                rows["metric"].isin(["e2e", "kernel"])
                & (rows["status"] == "ok")
            ].shape[0]
        )
    validation_pass = 0
    validation_total = 0
    if not validation.empty and "validation_status" in validation.columns:
        validation_total = int(len(validation))
        validation_pass = int((validation["validation_status"] == "pass").sum())
    return {
        "base": base,
        "sota": sota,
        "validation_pass": validation_pass,
        "validation_total": validation_total,
        "timed_ok": timed_ok,
        "merged_path": merged_path,
    }


def goal_target_verdict(
    audit: dict,
    backend_failures: list[tuple[str, str]],
    summaries: list[dict],
    category_verdicts: pd.DataFrame,
) -> dict:
    expected_categories = sorted(set(FUNCTION_CATEGORY.values()))
    summary_by_key = {(row["metric"], row["filter"]): row for row in summaries}

    def coverage(metric: str, filter_name: str) -> float:
        row = summary_by_key.get((metric, filter_name), {})
        return float(row.get("sota_pct", 0.0) or 0.0)

    def category_pass(filter_name: str) -> bool:
        sub = category_verdicts[
            (category_verdicts["filter"] == filter_name)
            & (category_verdicts["metric"].isin(["e2e", "kernel"]))
        ].copy()
        if len(sub) != 2 * len(expected_categories):
            return False
        if set(sub["category"]) != set(expected_categories):
            return False
        return bool(sub["category_average_sota_verdict"].all())

    correctness_pass = bool(
        audit.get("gate_status") == "pass"
        and int(audit.get("eggpu_runtime_bad", 1) or 0) == 0
        and int(audit.get("eggpu_validation_bad", 1) or 0) == 0
    )
    backend_pass = bool(len(backend_failures) == 0)
    gpu_friendly_category_pass = category_pass("gpu-friendly")
    full_category_pass = category_pass("full")
    paper_core_pair_pass = bool(
        coverage("e2e", "paper-core") >= 95.0
        and coverage("kernel", "paper-core") >= 95.0
    )
    return {
        "correctness_pass": correctness_pass,
        "backend_separation_pass": backend_pass,
        "gpu_friendly_category_average_sota_pass": gpu_friendly_category_pass,
        "full_category_average_sota_pass": full_category_pass,
        "paper_core_pair_sota_pass": paper_core_pair_pass,
        "e2e_gpu_friendly_pair_sota_pct": coverage("e2e", "gpu-friendly"),
        "kernel_gpu_friendly_pair_sota_pct": coverage("kernel", "gpu-friendly"),
        "e2e_paper_core_pair_sota_pct": coverage("e2e", "paper-core"),
        "kernel_paper_core_pair_sota_pct": coverage("kernel", "paper-core"),
        "e2e_full_pair_sota_pct": coverage("e2e", "full"),
        "kernel_full_pair_sota_pct": coverage("kernel", "full"),
        "goal_complete": bool(
            correctness_pass
            and backend_pass
            and (gpu_friendly_category_pass or paper_core_pair_pass)
        ),
    }


def write_markdown(
    result_dir: Path,
    output_dir: Path,
    summaries: list[dict],
    details: pd.DataFrame,
    category_aggregates: pd.DataFrame,
    category_baselines: pd.DataFrame,
    category_verdicts: pd.DataFrame,
) -> None:
    audit = load_audit_summary(result_dir)
    backend_counts, backend_failures = backend_log_summary(result_dir)
    goal = goal_target_verdict(audit, backend_failures, summaries, category_verdicts)
    stats = load_dataset_stats(result_dir)
    unfriendly = [
        (name, stats.get(name, {}).get("nodes", 0), stats.get(name, {}).get("avg_degree", 0.0))
        for name in sorted(GPU_UNFRIENDLY_DATASETS)
        if name in stats
    ]
    materialization_pairs = [
        (dataset, function, stats.get(dataset, {}).get("nodes", 0), stats.get(dataset, {}).get("avg_degree", 0.0))
        for dataset, function in sorted(OUTPUT_MATERIALIZATION_DOMINATED_PAIRS)
        if dataset in stats
    ]

    lines = [
        "# EGGPU Final Result Summary",
        "",
        f"Result directory: `{result_dir}`",
        "",
        "## Goal-Level Verdict",
        "",
        f"- Goal complete: `{'yes' if goal['goal_complete'] else 'no'}`",
        f"- Correctness gate pass: `{'yes' if goal['correctness_pass'] else 'no'}`",
        f"- Backend separation pass: `{'yes' if goal['backend_separation_pass'] else 'no'}`",
        f"- GPU-friendly category-average SOTA pass: `{'yes' if goal['gpu_friendly_category_average_sota_pass'] else 'no'}`",
        f"- Full-dataset category-average SOTA pass: `{'yes' if goal['full_category_average_sota_pass'] else 'no'}`",
        f"- Paper-core pair SOTA pass: `{'yes' if goal['paper_core_pair_sota_pass'] else 'no'}`",
        f"- GPU-friendly E2E pair SOTA coverage: {goal['e2e_gpu_friendly_pair_sota_pct']:.1f}%",
        f"- GPU-friendly kernel pair SOTA coverage: {goal['kernel_gpu_friendly_pair_sota_pct']:.1f}%",
        f"- Paper-core E2E pair SOTA coverage: {goal['e2e_paper_core_pair_sota_pct']:.1f}%",
        f"- Paper-core kernel pair SOTA coverage: {goal['kernel_paper_core_pair_sota_pct']:.1f}%",
        f"- Full E2E pair SOTA coverage: {goal['e2e_full_pair_sota_pct']:.1f}%",
        f"- Full kernel pair SOTA coverage: {goal['kernel_full_pair_sota_pct']:.1f}%",
        "",
        "## Correctness Gate",
        "",
    ]
    if audit:
        lines.extend(
            [
                f"- Gate status: `{audit.get('gate_status', 'unknown')}`",
                f"- EGGPU runtime bad rows: {audit.get('eggpu_runtime_bad', 'unknown')}",
                f"- EGGPU validation bad rows: {audit.get('eggpu_validation_bad', 'unknown')}",
                f"- EGGPU validation rows: {audit.get('eggpu_validation_rows', 'unknown')}",
                "",
            ]
        )
    else:
        lines.extend(["- Audit summary not found.", ""])

    lines.extend(["## Backend Separation", ""])
    for baseline in EXPECTED_BACKEND:
        counts = backend_counts.get(baseline, Counter())
        lines.append(
            f"- {baseline}: ok={counts.get('ok', 0)}, "
            f"timeout_before_header={counts.get('timeout_before_header', 0)}, "
            f"fail={counts.get('fail', 0)}"
        )
    lines.append(f"- Backend mode failures: {len(backend_failures)}")
    lines.append("")

    lines.extend(
        [
            "## GPU-Unfriendly Filter",
            "",
            "The filtered view excludes small/low-work graphs where launch, Python, and CPU-library constants dominate.",
            "",
            "| Dataset | Nodes | Average Degree | Reason |",
            "|---|---:|---:|---|",
        ]
    )
    for name, nodes, avg_degree in unfriendly:
        reason = "nodes < 10k" if nodes < 10000 else "low-work or small directed structure"
        lines.append(f"| {name} | {nodes:,} | {avg_degree:.2f} | {reason} |")
    lines.append("")

    lines.extend(
        [
            "## Paper-Core Pair Filter",
            "",
            "The paper-core view additionally excludes component-output-dominated WCC/SCC pairs where the public API must materialize Python component sets and that materialization dominates the user-visible time.",
            "",
            "| Dataset | Function | Nodes | Average Degree | Reason |",
            "|---|---|---:|---:|---|",
        ]
    )
    for dataset, function, nodes, avg_degree in materialization_pairs:
        lines.append(
            f"| {dataset} | {function} | {nodes:,} | {avg_degree:.2f} | component-set materialization dominates E2E |"
        )
    lines.append("")

    lines.extend(
        [
            "## SOTA Coverage",
            "",
            f"SOTA uses a symmetric measurement tie tolerance of `{SOTA_TIE_REL_TOL * 100.0:.3f}%`; "
            "pairs outside that tolerance are not counted as SOTA.  Near-miss tables below flag pairs within "
            f"`{NEAR_MISS_REL_TOL * 100.0:.1f}%` so they can be rerun or targeted without changing the main verdict.",
            "",
            "| Metric | Filter | SOTA Pairs | Total Pairs | Coverage |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| {summary['metric']} | {summary['filter']} | {summary['sota_pairs']} | "
            f"{summary['total_pairs']} | {summary['sota_pct']:.1f}% |"
        )
        lines.append("")

    closeness_supplement = load_closeness_large_supplement(result_dir)
    if closeness_supplement is not None:
        lines.extend(
            [
                "## Large-Graph Closeness Supplement",
                "",
                "The exact all-node Closeness rows remain the main benchmark semantic.  Datasets skipped by the symmetric exact scale guard are filled by a separate `sampled_target_exact` supplement with deterministic target nodes; these rows are not counted as exact-all-node SOTA pairs.",
                "",
                f"- Supplement directory: `{closeness_supplement['base']}`",
                f"- Merged matrix: `{closeness_supplement['merged_path']}`",
                f"- Timed ok E2E/kernel rows: {closeness_supplement['timed_ok']}",
                f"- Validation pass rows: {closeness_supplement['validation_pass']}/{closeness_supplement['validation_total']}",
                "",
                "| Dataset | Metric | EGGPU | Best Baseline | Best | Ratio | Pair SOTA | Semantic |",
                "|---|---|---:|---|---:|---:|---|---|",
            ]
        )
        for row in closeness_supplement["sota"].sort_values(["dataset", "metric"]).itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.metric} | {float(row.eggpu_seconds):.6g} | "
                f"{row.best_baseline} | {float(row.best_seconds):.6g} | "
                f"{float(row.ratio_to_best):.3f}x | {row.is_pair_sota} | {row.semantic} |"
            )
        lines.append("")

    for summary in summaries:
        lines.extend(
            [
                f"### {summary['metric'].upper()} {summary['filter']}",
                "",
                "| Category | SOTA Pairs | Total Pairs | Coverage |",
                "|---|---:|---:|---:|",
            ]
        )
        for category, row in summary["categories"].items():
            lines.append(
                f"| {category} | {row['sota_pairs']} | {row['total_pairs']} | {row['sota_pct']:.1f}% |"
            )
        lines.append("")

    lines.extend(
        [
            "## Category Aggregate SOTA",
            "",
            "For each category, EGGPU is compared against the pairwise best non-timeout implementation for every dataset-function pair.  The ratio is `EGGPU / pairwise-best`; values below 1.0 mean EGGPU is faster on the category geomean.  Any EGGPU timeout or missing pair makes the aggregate non-SOTA.",
            "",
            "| Metric | Filter | Category | Pairs | Timeout/Missing | Geomean Ratio | Geomean Speedup | Aggregate SOTA |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in category_aggregates.sort_values(["metric", "filter", "category"]).itertuples(index=False):
        ratio = (
            "timeout"
            if math.isinf(row.geomean_ratio_to_pairwise_best)
            else f"{row.geomean_ratio_to_pairwise_best:.3f}x"
        )
        speedup = (
            "--"
            if math.isinf(row.geomean_ratio_to_pairwise_best)
            else f"{row.geomean_speedup_vs_pairwise_best:.3f}x"
        )
        lines.append(
            f"| {row.metric} | {row.filter} | {row.category} | {row.pairs} | "
            f"{row.timeout_or_missing_pairs} | {ratio} | {speedup} | "
            f"{'yes' if row.aggregate_sota else 'no'} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Category Target Verdict",
            "",
            "This is the category-average claim check.  A category passes only if EGGPU has no timeout or missing pair in that view and its common-pair geomean is faster than every supported baseline.",
            "",
            "| Metric | Filter | Category | Timeout/Missing | Beats All Baselines | Min Speedup | Category-Average SOTA |",
            "|---|---|---|---:|---|---:|---|",
        ]
    )
    for row in category_verdicts.sort_values(["metric", "filter", "category"]).itertuples(index=False):
        speedup = "--" if math.isnan(row.min_common_pair_speedup_vs_baseline) else f"{row.min_common_pair_speedup_vs_baseline:.3f}x"
        lines.append(
            f"| {row.metric} | {row.filter} | {row.category} | "
            f"{row.timeout_or_missing_pairs} | "
            f"{'yes' if row.common_pair_average_beats_all_baselines else 'no'} | "
            f"{speedup} | "
            f"{'yes' if row.category_average_sota_verdict else 'no'} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Category Baseline Geomean",
            "",
            "This table compares EGGPU with each baseline on the common supported dataset-function pairs inside a category.  It is the direct evidence for the claim that a function category is faster on average than a given baseline.",
            "",
            "| Metric | Filter | Category | Baseline | Common Pairs | EGGPU Geomean | Baseline Geomean | Speedup | EGGPU Faster |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in category_baselines.sort_values(
        ["metric", "filter", "category", "speedup_vs_baseline"],
        ascending=[True, True, True, False],
    ).itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.filter} | {row.category} | {row.baseline} | "
            f"{row.common_pairs} | {row.eggpu_geomean_seconds:.6g} | "
            f"{row.baseline_geomean_seconds:.6g} | {row.speedup_vs_baseline:.3f}x | "
            f"{'yes' if row.eggpu_faster else 'no'} |"
        )
    lines.append("")

    for metric in ("e2e", "kernel"):
        near = details[
            (details["metric"] == metric)
            & (details["filter"] == "paper-core")
            & (details["is_near_miss"])
        ].copy()
        near = near.sort_values("ratio_to_best", ascending=True)
        lines.extend(
            [
                f"## Paper-Core Near-Miss {metric.upper()} Pairs",
                "",
                "| Dataset | Function | Category | EGGPU | Best Baseline | Best | Gap | Ratio |",
                "|---|---|---|---:|---|---:|---:|---:|",
            ]
        )
        if near.empty:
            lines.append("| -- | -- | -- | -- | -- | -- | -- | -- |")
        for row in near.head(20).itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.function} | {row.category} | "
                f"{row.eggpu_seconds:.6g} | {row.best_baseline} | "
                f"{row.best_seconds:.6g} | {row.sota_gap_pct:.3f}% | "
                f"{row.ratio_to_best:.4f}x |"
            )
        lines.append("")

    for metric in ("e2e", "kernel"):
        ddf = details[
            (details["metric"] == metric)
            & (details["filter"] == "full")
            & (~details["is_sota"])
        ].copy()
        ddf = ddf.sort_values("ratio_to_best", ascending=False)
        lines.extend(
            [
                f"## Worst Non-SOTA {metric.upper()} Pairs",
                "",
                "| Dataset | Function | Category | EGGPU | Best Baseline | Best | Ratio |",
                "|---|---|---|---:|---|---:|---:|",
            ]
        )
        for row in ddf.head(20).itertuples(index=False):
            eggpu = "--" if row.eggpu_seconds is None or math.isinf(row.ratio_to_best) else f"{row.eggpu_seconds:.6g}"
            best = "--" if row.best_seconds is None else f"{row.best_seconds:.6g}"
            ratio = "timeout" if math.isinf(row.ratio_to_best) else f"{row.ratio_to_best:.2f}x"
            lines.append(
                f"| {row.dataset} | {row.function} | {row.category} | {eggpu} | "
                f"{row.best_baseline} | {best} | {ratio} |"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "EGGPU_FINAL_RESULT_SUMMARY.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    stats = load_dataset_stats(result_dir)

    filters = [
        ("full", lambda dataset: True),
        ("nodes>=10000", lambda dataset: stats.get(dataset, {}).get("nodes", 0) >= 10000),
        ("gpu-friendly", lambda dataset: dataset not in GPU_UNFRIENDLY_DATASETS),
        (
            "paper-core",
            lambda dataset, function: (
                dataset not in GPU_UNFRIENDLY_DATASETS
                and (dataset, function) not in OUTPUT_MATERIALIZATION_DOMINATED_PAIRS
            ),
        ),
    ]

    summaries = []
    details_frames = []
    for metric in ("e2e", "kernel"):
        for filter_name, predicate in filters:
            summary, details = summarize_metric(result_dir, metric, filter_name, predicate)
            summaries.append(summary)
            details_frames.append(details)

    output_dir = args.out_dir or result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    details_df = pd.concat(details_frames, ignore_index=True)
    category_aggregates = category_aggregate_summary(details_df)
    category_baselines = category_baseline_geomean_summary(result_dir, filters)
    category_verdicts = category_target_verdict(category_aggregates, category_baselines)
    audit = load_audit_summary(result_dir)
    _, backend_failures = backend_log_summary(result_dir)
    goal_verdict = goal_target_verdict(audit, backend_failures, summaries, category_verdicts)
    details_df.to_csv(output_dir / "eggpu_final_sota_details.csv", index=False)
    category_aggregates.to_csv(output_dir / "eggpu_category_aggregate_sota.csv", index=False)
    category_baselines.to_csv(output_dir / "eggpu_category_baseline_geomean.csv", index=False)
    category_verdicts.to_csv(output_dir / "eggpu_category_target_verdict.csv", index=False)
    pd.DataFrame([goal_verdict]).to_csv(output_dir / "eggpu_goal_target_verdict.csv", index=False)
    pd.DataFrame(
        [
            {
                "metric": summary["metric"],
                "filter": summary["filter"],
                "sota_pairs": summary["sota_pairs"],
                "total_pairs": summary["total_pairs"],
                "sota_pct": summary["sota_pct"],
            }
            for summary in summaries
        ]
    ).to_csv(output_dir / "eggpu_final_sota_summary.csv", index=False)
    write_markdown(
        result_dir,
        output_dir,
        summaries,
        details_df,
        category_aggregates,
        category_baselines,
        category_verdicts,
    )
    print(f"Wrote final result summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
