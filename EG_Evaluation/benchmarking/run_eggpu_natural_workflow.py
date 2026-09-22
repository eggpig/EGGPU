#!/usr/bin/env python3
"""Natural same-graph workflow case study for EGGPU.

The canonical workflow uses three algorithms selected from the LDBC
Graphalytics workload: WCC, PageRank, and BFS. Graphalytics motivates the
algorithm set; the sequential order is defined by this EGGPU case study.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmark_stats import aggregate_sample_rows
from gpu_device_profile import collect_gpu_device_profile, collect_host_profile
from gpu_visibility_marker import GpuVisibilityMarker
from library_baselines import build_easygraph, load_graph, pick_sources, summarize_sssp_result
from measurement_schema import describe_metric, write_measurement_schema
from run_full_baselines import (
    DEFAULT_DATASETS,
    check_eggpu_child_gpu_idle,
    collect_implementation_source_snapshot,
    collect_source_snapshot,
    cpp_easygraph_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ("WCC", "PageRank", "BFS")
KERNEL_KEYS = {"WCC": "cc", "PageRank": "pagerank", "BFS": "bfs"}
RESULT_PREFIX = "WORKFLOW_RESULT_JSON "


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parsed_workflow_rows(log_path: Path) -> list[dict[str, object]]:
    if not log_path.exists():
        return []
    parsed = []
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith(RESULT_PREFIX):
            row = json.loads(line[len(RESULT_PREFIX) :])
            row["log"] = str(log_path)
            parsed.append(row)
    return parsed


def cpp_artifact_digests(metadata_or_artifacts) -> list[str]:
    if isinstance(metadata_or_artifacts, dict) and "build_artifacts" in metadata_or_artifacts:
        artifacts = (metadata_or_artifacts.get("build_artifacts") or {}).get(
            "cpp_easygraph", []
        )
    else:
        artifacts = metadata_or_artifacts or []
    return sorted(
        str(item.get("sha256", ""))
        for item in artifacts
        if isinstance(item, dict) and item.get("sha256")
    )


def verify_first_use_implementation(first_use_dir: Path):
    metadata_path = first_use_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"missing First-use metadata: {metadata_path}")
    first_metadata = json.loads(metadata_path.read_text())
    first_snapshot = (first_metadata.get("source_snapshot") or {}).get("digest", "")
    current_snapshot = collect_source_snapshot()
    current_digest = current_snapshot.get("digest", "")
    first_impl = first_metadata.get("implementation_source_snapshot") or {}
    current_impl = collect_implementation_source_snapshot()
    first_impl_digest = first_impl.get("digest", "")
    current_impl_digest = current_impl.get("digest", "")
    if first_impl_digest:
        if not current_impl_digest or first_impl_digest != current_impl_digest:
            raise SystemExit(
                "First-use/natural-workflow implementation source mismatch: "
                f"first_use={first_impl_digest or 'missing'} "
                f"current={current_impl_digest or 'missing'}"
            )
    elif not first_snapshot or first_snapshot != current_digest:
        raise SystemExit(
            "First-use/natural-workflow source snapshot mismatch: "
            f"first_use={first_snapshot or 'missing'} current={current_digest or 'missing'}"
        )
    first_artifacts = cpp_artifact_digests(first_metadata)
    current_artifact_rows = cpp_easygraph_artifacts()
    current_artifacts = cpp_artifact_digests(current_artifact_rows)
    if not first_artifacts or first_artifacts != current_artifacts:
        raise SystemExit(
            "First-use/natural-workflow cpp_easygraph artifact mismatch: "
            f"first_use={first_artifacts} current={current_artifacts}"
        )
    verification = {
        "status": "pass",
        "mode": (
            "implementation_source_and_binary_match"
            if first_impl_digest
            else "full_source_and_binary_match"
        ),
        "first_use_full_source_snapshot": first_snapshot,
        "current_full_source_snapshot": current_digest,
        "first_use_implementation_source_snapshot": first_impl,
        "current_implementation_source_snapshot": current_impl,
        "cpp_easygraph_sha256": current_artifacts,
    }
    (first_use_dir / "natural_workflow_implementation_compatibility.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n"
    )
    return first_metadata, current_snapshot, current_artifact_rows


def parse_datasets(token_string: str):
    tokens = {
        token.strip().lower()
        for token in str(token_string).split(",")
        if token.strip()
    }
    if not tokens or "all" in tokens:
        return list(DEFAULT_DATASETS)
    selected = [
        item
        for item in DEFAULT_DATASETS
        if {item[0].lower(), item[1].lower(), item[2].lower()} & tokens
    ]
    if not selected:
        raise SystemExit(f"no datasets matched --datasets={token_string!r}")
    return selected


def numeric(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def plus_minus(mean, std) -> str:
    mean_value = numeric(mean)
    std_value = numeric(std)
    if mean_value is None:
        return ""
    if std_value is None:
        return f"{mean_value:.6g}"
    return f"{mean_value:.6g} +/- {std_value:.3g}"


def graph_state(G) -> dict[str, object]:
    cache = getattr(G, "cache", {})
    prepared = cache.get("__eggpu_prepared_graph__") if isinstance(cache, dict) else None
    cpp_entries = 0
    if isinstance(cache, dict):
        cpp_entries = sum(
            1
            for key in cache
            if str(key).startswith("__eggpu_cpp_graph__:")
        )
    return {
        "graph_context_present": isinstance(prepared, dict),
        "graph_context_id": prepared.get("ctx_id", "") if isinstance(prepared, dict) else "",
        "graph_context_reuse_hits": prepared.get("reuse_hits", 0) if isinstance(prepared, dict) else 0,
        "directed_view_ready": bool(prepared.get("directed_ready")) if isinstance(prepared, dict) else False,
        "undirected_view_ready": bool(prepared.get("undirected_ready")) if isinstance(prepared, dict) else False,
        "cpp_graph_cache_entries": cpp_entries,
    }


def result_summary(function: str, result) -> str:
    if function == "WCC":
        return f"components={len(result)},covered_nodes={sum(len(part) for part in result)}"
    if function == "PageRank":
        values = result.values() if hasattr(result, "values") else result
        return f"nodes={len(result)},sum={sum(float(value) for value in values):.12g}"
    reachable, checksum = summarize_sssp_result(result)
    return f"sources={len(result)},reachable={reachable},checksum={checksum:.12g}"


def sample_row(
    args,
    function: str,
    metric: str,
    value: float,
    *,
    position: int,
    correctness: str = "",
    state_before: dict[str, object] | None = None,
    state_after: dict[str, object] | None = None,
    baseline: str = "EGGPU-natural-workflow",
    execution_protocol: str = "natural-workflow",
    semantic: str = "natural_same_graph_workflow",
    notes: str = (
        "Graphalytics-inspired WCC->PageRank->BFS case study; "
        "fresh process and one Python graph; no artificial prewarm"
    ),
) -> dict[str, object]:
    descriptor = describe_metric(metric, baseline="EGGPU")
    row: dict[str, object] = {
        "dataset_size": args.dataset_size,
        "graph_type": args.graph_type,
        "dataset": args.dataset_name,
        "function": function,
        "baseline": baseline,
        "metric": metric,
        "seconds": value,
        "value": value,
        **descriptor,
        "status": "ok",
        "correctness": correctness,
        "log": "",
        "notes": notes,
        "semantic": semantic,
        "skip_reason": "",
        "estimator_kind": "independent_process_samples",
        "sample_sources": args.bfs_sources if function == "BFS" else "",
        "source_policy": "deterministic_evenly_spaced" if function == "BFS" else "",
        "source_seed": "",
        "source_nodes_sha": "",
        "sample_index": str(args.sample_index),
        "sample_count": str(args.repeat),
        "measurement_phase": "timing",
        "execution_protocol": execution_protocol,
        "workflow_order_id": "wcc_pagerank_bfs",
        "call_position": position,
        "python_graph_object_id": "one-object-within-sample",
        "execution_order_slot": getattr(args, "schedule_slot", ""),
    }
    for prefix, state in (("before", state_before), ("after", state_after)):
        for key, state_value in (state or {}).items():
            row[f"state_{prefix}_{key}"] = state_value
    return row


def execute_function(args, graph, function: str, eg, backend):
    """Execute one workflow function with an identical timed-call boundary."""

    kernel_key = KERNEL_KEYS[function]
    backend.set_last_kernel_time(kernel_key, None)
    state_before = graph_state(graph)
    sources = pick_sources(graph.number_of_nodes(), args.bfs_sources)
    start = time.perf_counter()
    if function == "WCC":
        if args.graph_type == "directed":
            result = list(eg.weakly_connected_components(graph))
        else:
            result = list(eg.connected_components(graph))
    elif function == "PageRank":
        result = eg.pagerank(
            graph,
            alpha=args.pr_alpha,
            max_iter=args.pr_max_iter,
            tol=args.pr_eps,
        )
    else:
        result = eg.multi_source_bfs(graph, sources, target=None)
    e2e_seconds = time.perf_counter() - start
    kernel_seconds = backend.get_last_kernel_time(kernel_key)
    if kernel_seconds is None:
        raise RuntimeError(f"missing CUDA-event kernel timing for {function}")
    kernel_seconds = float(kernel_seconds)
    if not math.isfinite(kernel_seconds) or kernel_seconds < 0:
        raise RuntimeError(
            f"invalid CUDA-event kernel timing for {function}: {kernel_seconds}"
        )
    return (
        result,
        e2e_seconds,
        kernel_seconds,
        state_before,
        graph_state(graph),
    )


def child_main(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["EGGPU_MONITOR_GPU_INDEX"] = str(args.gpu)
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"
    os.environ["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
    os.environ["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
    os.environ["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
    child_protocol = (
        "isolated-first-use-control"
        if args.child_mode == "isolated"
        else "natural-workflow"
    )
    os.environ["EGGPU_EXECUTION_PROTOCOL"] = child_protocol
    os.environ["EGGPU_MEASUREMENT_MODE"] = "timing"

    import easygraph as eg
    from easygraph.utils import gpu_eggpu_backend as backend

    views = load_graph(args.edge_path)
    n, directed_edges, undirected_edges = views["all_vertices"]
    edges = directed_edges if args.graph_type == "directed" else undirected_edges
    build_start = time.perf_counter()
    graph = build_easygraph(n, edges, args.graph_type == "directed")
    build_seconds = time.perf_counter() - build_start
    before_any = graph_state(graph)
    if args.child_mode == "isolated":
        functions = (args.isolated_function,)
        baseline = "EGGPU-isolated-first-use"
        semantic = "matched_isolated_first_use_control"
        notes = (
            "matched isolated control for same-graph workflow; fresh process, "
            "same graph representation and API semantics, no artificial prewarm"
        )
    else:
        functions = WORKFLOW
        baseline = "EGGPU-natural-workflow"
        semantic = "natural_same_graph_workflow"
        notes = (
            "Graphalytics-inspired WCC->PageRank->BFS case study; "
            "fresh process and one Python graph; no artificial prewarm"
        )
    rows = [
        sample_row(
            args,
            "GRAPH",
            "build",
            build_seconds,
            position=0,
            correctness=f"nodes={n},edges={len(edges)}",
            state_before=before_any,
            state_after=graph_state(graph),
            baseline=baseline,
            execution_protocol=child_protocol,
            semantic=semantic,
            notes=notes,
        )
    ]
    total_e2e = 0.0
    total_kernel = 0.0
    for function in functions:
        position = WORKFLOW.index(function) + 1
        result, e2e_seconds, kernel_seconds, state_before, state_after = (
            execute_function(args, graph, function, eg, backend)
        )
        correctness = result_summary(function, result)
        rows.append(
            sample_row(
                args,
                function,
                "e2e",
                e2e_seconds,
                position=position,
                correctness=correctness,
                state_before=state_before,
                state_after=state_after,
                baseline=baseline,
                execution_protocol=child_protocol,
                semantic=semantic,
                notes=notes,
            )
        )
        rows.append(
            sample_row(
                args,
                function,
                "kernel",
                kernel_seconds,
                position=position,
                correctness=correctness,
                state_before=state_before,
                state_after=state_after,
                baseline=baseline,
                execution_protocol=child_protocol,
                semantic=semantic,
                notes=notes,
            )
        )
        total_e2e += e2e_seconds
        total_kernel += kernel_seconds
    if args.child_mode == "workflow":
        final_state = graph_state(graph)
        rows.append(
            sample_row(
                args,
                "ALL",
                "e2e",
                total_e2e,
                position=len(WORKFLOW) + 1,
                correctness="all_calls_completed",
                state_before=before_any,
                state_after=final_state,
            )
        )
        rows.append(
            sample_row(
                args,
                "ALL",
                "kernel",
                total_kernel,
                position=len(WORKFLOW) + 1,
                correctness="all_calls_completed",
                state_before=before_any,
                state_after=final_state,
            )
        )
    for row in rows:
        print(RESULT_PREFIX + json.dumps(row, sort_keys=True), flush=True)


def isolated_sum_samples(
    source: list[dict[str, object]], repeat: int
) -> list[dict[str, object]]:
    """Sum matched, same-run isolated controls by dataset/sample/metric."""

    grouped: dict[tuple[str, str, str], list[float]] = {}
    metadata: dict[str, dict[str, object]] = {}
    for row in source:
        if (
            row.get("baseline") != "EGGPU-isolated-first-use"
            or row.get("function") not in WORKFLOW
        ):
            continue
        if row.get("metric") not in {"e2e", "kernel"} or row.get("status") != "ok":
            continue
        key = (row.get("dataset", ""), row.get("sample_index", ""), row.get("metric", ""))
        value = numeric(row.get("value") or row.get("seconds"))
        if value is None:
            continue
        grouped.setdefault(key, []).append(value)
        metadata[row.get("dataset", "")] = row
    rows = []
    for (dataset, sample_index, metric), values in sorted(grouped.items()):
        if len(values) != len(WORKFLOW):
            raise SystemExit(
                f"incomplete isolated First-use workflow control: dataset={dataset} "
                f"sample={sample_index} metric={metric} functions={len(values)}/{len(WORKFLOW)}"
            )
        source_row = metadata[dataset]
        descriptor = describe_metric(metric, baseline="EGGPU")
        total = sum(values)
        rows.append(
            {
                "dataset_size": source_row.get("dataset_size", ""),
                "graph_type": source_row.get("graph_type", ""),
                "dataset": dataset,
                "function": "ALL",
                "baseline": "EGGPU-isolated-first-use-sum",
                "metric": metric,
                "seconds": total,
                "value": total,
                **descriptor,
                "status": "ok",
                "correctness": "sum_of_three_validated_function_calls",
                "log": "",
                "notes": (
                    "sum of three matched isolated First-use controls collected "
                    "on the same GPU and in the same experiment batch"
                ),
                "semantic": "matched_isolated_first_use_sum",
                "skip_reason": "",
                "estimator_kind": "independent_process_samples",
                "sample_sources": "",
                "source_policy": "",
                "source_seed": "",
                "source_nodes_sha": "",
                "sample_index": sample_index,
                "sample_count": str(repeat),
                "measurement_phase": "timing",
            }
        )
    return rows


def compare_aggregates(aggregates: list[dict[str, str]]) -> list[dict[str, object]]:
    indexed = {
        (row.get("dataset", ""), row.get("baseline", ""), row.get("metric", "")): row
        for row in aggregates
        if row.get("function") == "ALL" and row.get("status") == "ok"
    }
    rows = []
    datasets = sorted({key[0] for key in indexed})
    for dataset in datasets:
        for metric in ("e2e", "kernel"):
            natural = indexed.get((dataset, "EGGPU-natural-workflow", metric))
            isolated = indexed.get((dataset, "EGGPU-isolated-first-use-sum", metric))
            if natural is None or isolated is None:
                continue
            natural_mean = numeric(natural.get("mean_value"))
            isolated_mean = numeric(isolated.get("mean_value"))
            rows.append(
                {
                    "dataset": dataset,
                    "graph_type": natural.get("graph_type", ""),
                    "dataset_size": natural.get("dataset_size", ""),
                    "metric": metric,
                    "natural_mean_seconds": natural_mean,
                    "natural_std_seconds": numeric(natural.get("std_value")),
                    "natural_mean_plus_minus_sd": plus_minus(
                        natural_mean, natural.get("std_value")
                    ),
                    "natural_relative_std_percent": natural.get("relative_std_percent", ""),
                    "isolated_mean_seconds": isolated_mean,
                    "isolated_std_seconds": numeric(isolated.get("std_value")),
                    "isolated_mean_plus_minus_sd": plus_minus(
                        isolated_mean, isolated.get("std_value")
                    ),
                    "isolated_relative_std_percent": isolated.get("relative_std_percent", ""),
                    "isolated_over_natural": (
                        isolated_mean / natural_mean
                        if isolated_mean is not None and natural_mean not in (None, 0.0)
                        else None
                    ),
                    "sample_count": natural.get("sample_count", ""),
                }
            )
    return rows


def compare_per_call(
    natural_aggregates: list[dict[str, str]],
    isolated_aggregates: list[dict[str, str]],
) -> list[dict[str, object]]:
    first_index = {
        (row.get("dataset", ""), row.get("function", ""), row.get("metric", "")): row
        for row in isolated_aggregates
        if row.get("baseline") == "EGGPU-isolated-first-use"
        and row.get("function") in WORKFLOW
        and row.get("metric") in {"e2e", "kernel"}
        and row.get("status") == "ok"
    }
    rows = []
    for natural in natural_aggregates:
        if natural.get("baseline") != "EGGPU-natural-workflow":
            continue
        function = natural.get("function", "")
        metric = natural.get("metric", "")
        if function not in WORKFLOW or metric not in {"e2e", "kernel"}:
            continue
        first_row = first_index.get((natural.get("dataset", ""), function, metric))
        if first_row is None:
            continue
        natural_mean = numeric(natural.get("mean_value"))
        natural_std = numeric(natural.get("std_value"))
        first_mean = numeric(first_row.get("mean_value"))
        first_std = numeric(first_row.get("std_value"))
        position = int(natural.get("call_position") or WORKFLOW.index(function) + 1)
        time_saved = (
            first_mean - natural_mean
            if first_mean is not None and natural_mean is not None
            else None
        )
        rows.append(
            {
                "dataset": natural.get("dataset", ""),
                "graph_type": natural.get("graph_type", ""),
                "dataset_size": natural.get("dataset_size", ""),
                "function": function,
                "call_position": position,
                "comparison_role": (
                    "initialization_payer" if position == 1 else "reuse_beneficiary"
                ),
                "reuse_beneficiary": position in {2, 3},
                "metric": metric,
                "natural_mean_seconds": natural_mean,
                "natural_std_seconds": natural_std,
                "natural_mean_plus_minus_sd": plus_minus(natural_mean, natural_std),
                "first_use_mean_seconds": first_mean,
                "first_use_std_seconds": first_std,
                "first_use_mean_plus_minus_sd": plus_minus(first_mean, first_std),
                "first_use_over_natural": (
                    first_mean / natural_mean
                    if first_mean is not None and natural_mean not in (None, 0.0)
                    else None
                ),
                "natural_over_first_use": (
                    natural_mean / first_mean
                    if natural_mean is not None and first_mean not in (None, 0.0)
                    else None
                ),
                "time_saved_seconds": time_saved,
                "time_saved_percent_of_first_use": (
                    100.0 * time_saved / first_mean
                    if time_saved is not None and first_mean not in (None, 0.0)
                    else None
                ),
                "natural_is_faster": (
                    time_saved > 0.0 if time_saved is not None else None
                ),
                "sample_count": natural.get("sample_count", ""),
            }
        )
    return rows


def geomean(values) -> float | None:
    valid = [float(value) for value in values if value is not None and float(value) > 0]
    if not valid:
        return None
    return math.exp(sum(math.log(value) for value in valid) / len(valid))


def summarize_reuse_beneficiaries(
    per_call_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    beneficiaries = [
        row for row in per_call_rows if bool(row.get("reuse_beneficiary"))
    ]
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in beneficiaries:
        groups.setdefault((str(row.get("function", "")), str(row.get("metric", ""))), []).append(row)
        groups.setdefault(("ALL_BENEFICIARIES", str(row.get("metric", ""))), []).append(row)

    summaries: list[dict[str, object]] = []
    for (function, metric), rows in sorted(groups.items()):
        first_values = [numeric(row.get("first_use_mean_seconds")) for row in rows]
        natural_values = [numeric(row.get("natural_mean_seconds")) for row in rows]
        savings = [numeric(row.get("time_saved_seconds")) for row in rows]
        ratios = [numeric(row.get("first_use_over_natural")) for row in rows]
        valid_first = [value for value in first_values if value is not None]
        valid_natural = [value for value in natural_values if value is not None]
        valid_savings = [value for value in savings if value is not None]
        first_total = sum(valid_first)
        natural_total = sum(valid_natural)
        saved_total = first_total - natural_total
        summaries.append(
            {
                "function": function,
                "metric": metric,
                "comparison_pairs": len(rows),
                "datasets": len({str(row.get("dataset", "")) for row in rows}),
                "call_positions": ",".join(
                    sorted({str(row.get("call_position", "")) for row in rows})
                ),
                "first_use_total_seconds": first_total,
                "natural_total_seconds": natural_total,
                "time_saved_total_seconds": saved_total,
                "time_saved_percent_of_first_use_total": (
                    100.0 * saved_total / first_total if first_total > 0.0 else None
                ),
                "mean_time_saved_seconds": (
                    statistics.fmean(valid_savings) if valid_savings else None
                ),
                "median_time_saved_seconds": (
                    statistics.median(valid_savings) if valid_savings else None
                ),
                "geomean_first_use_over_natural": geomean(ratios),
                "natural_faster_pairs": sum(
                    1 for row in rows if row.get("natural_is_faster") is True
                ),
                "natural_not_faster_pairs": sum(
                    1 for row in rows if row.get("natural_is_faster") is False
                ),
            }
        )
    return summaries


def write_report(
    out_dir: Path,
    comparison: list[dict[str, object]],
    beneficiary_summary: list[dict[str, object]],
    repeat: int,
) -> None:
    lines = [
        "# EGGPU Natural Same-Graph Workflow\n\n",
        "## Selection basis\n\n",
        "- WCC, PageRank, and BFS are three core algorithms in the LDBC Graphalytics workload.\n",
        "- LDBC source: https://ldbcouncil.org/benchmarks/graphalytics/algorithms\n",
        "- The exact WCC -> PageRank -> BFS order is an EGGPU case-study design, not an LDBC-prescribed pipeline.\n\n",
        "## Protocol\n\n",
        "- Each workflow repeat starts a fresh Python process and builds one EasyGraph object.\n",
        "- No GraphContext prewarm and no function warmup are performed.\n",
        "- The first call creates reusable graph state; later calls use the same graph object.\n",
        "- Every workflow repeat is paired with three isolated First-use controls collected in the same run on the same GPU.\n",
        "- Each isolated control uses the same graph direction, API, parameters, source policy, timed-call boundary, and result semantics as its workflow counterpart.\n",
        "- WCC at position 1 is reported as the initialization payer. PageRank at position 2 and BFS at position 3 are the direct reuse beneficiaries.\n",
        "- Every beneficiary is compared with the same function's isolated First-use mean; absolute seconds, percentage saved, ratio, and sample standard deviation are retained.\n",
        f"- Values are arithmetic mean +/- sample standard deviation over {repeat} independent processes.\n\n",
        "## Summary\n\n",
    ]
    for metric in ("e2e", "kernel"):
        subset = [row for row in comparison if row["metric"] == metric]
        ratio = geomean(row["isolated_over_natural"] for row in subset)
        lines.append(
            f"- {metric.upper()}: {len(subset)} datasets; geometric mean isolated/natural = "
            + (f"**{ratio:.3f}x**.\n" if ratio is not None else "unavailable.\n")
        )
    lines.append("\n## Calls benefiting from prior graph state\n\n")
    for row in beneficiary_summary:
        if row["function"] != "ALL_BENEFICIARIES":
            continue
        ratio = numeric(row.get("geomean_first_use_over_natural"))
        saved = numeric(row.get("time_saved_total_seconds"))
        percent = numeric(row.get("time_saved_percent_of_first_use_total"))
        lines.append(
            f"- {str(row['metric']).upper()}: positions 2-3 across "
            f"{row['datasets']} datasets; geometric mean isolated First-use/natural = "
            + (f"**{ratio:.3f}x**" if ratio is not None else "unavailable")
            + (f", aggregate time saved = **{saved:.6g}s ({percent:.2f}%)**.\n" if saved is not None and percent is not None else ".\n")
        )
    lines.extend(
        [
            "\n## Artifacts\n\n",
            "- `natural_workflow_samples.csv` and `natural_workflow_aggregates.csv`\n",
            "- `isolated_first_use_samples.csv`\n",
            "- `isolated_first_use_sum_samples.csv`\n",
            "- `natural_workflow_vs_isolated.csv`\n",
            "- `natural_workflow_per_call_vs_first_use.csv`\n",
            "- `natural_workflow_reuse_beneficiaries.csv`\n",
            "- `natural_workflow_reuse_beneficiary_summary.csv`\n",
            "- `natural_workflow_metadata.json`\n",
        ]
    )
    (out_dir / "NATURAL_WORKFLOW_RESULT.md").write_text("".join(lines))


def parent_main(args) -> None:
    if args.repeat < 2:
        raise SystemExit("natural workflow requires at least two repeats for dispersion")
    first_use_dir = Path(args.first_use_dir).resolve()
    if not (first_use_dir / "results_samples.csv").exists():
        raise SystemExit(f"missing First-use samples: {first_use_dir}")
    first_metadata, current_snapshot, current_artifact_rows = (
        verify_first_use_implementation(first_use_dir)
    )
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume_existing:
        raise SystemExit(f"refusing to overwrite non-empty workflow directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    logs.mkdir(exist_ok=True)
    write_measurement_schema(out_dir / "measurement_schema.json")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["EGGPU_MONITOR_GPU_INDEX"] = str(args.gpu)
    gpu_device_profile = collect_gpu_device_profile(args.gpu, os.environ)
    host_profile = collect_host_profile()
    marker = GpuVisibilityMarker(args.gpu, "natural workflow case study").start()
    if marker.started:
        os.environ["EGGPU_EXTERNAL_VISIBILITY_MARKER"] = "TRUE"

    datasets = parse_datasets(args.datasets)
    samples: list[dict[str, object]] = []
    total = len(datasets) * args.repeat * (1 + len(WORKFLOW))
    current = 0

    def run_case(
        *,
        size: str,
        graph_type: str,
        name: str,
        edge_path: str,
        sample_index: int,
        child_mode: str,
        isolated_function: str = "",
        schedule_slot: int = 0,
    ) -> list[dict[str, object]]:
        nonlocal current
        current += 1
        idle_ok, idle_note = check_eggpu_child_gpu_idle(os.environ)
        if not idle_ok:
            raise SystemExit(idle_note)
        label = (
            "workflow"
            if child_mode == "workflow"
            else f"isolated-{isolated_function}"
        )
        print(
            f"[progress] natural-workflow {current}/{total} "
            f"case={label} dataset={name} sample={sample_index}/{args.repeat}",
            flush=True,
        )
        case_dir = logs / child_mode
        case_dir.mkdir(parents=True, exist_ok=True)
        suffix = "workflow" if child_mode == "workflow" else isolated_function.lower()
        log_path = case_dir / f"{name}_{suffix}_r{sample_index}.log"
        expected_rows = 9 if child_mode == "workflow" else 3
        if args.resume_existing:
            parsed = parsed_workflow_rows(log_path)
            if len(parsed) == expected_rows:
                print(
                    f"[resume] accepted complete {label} sample: "
                    f"dataset={name} sample={sample_index}/{args.repeat}",
                    flush=True,
                )
                return parsed
            if parsed:
                print(
                    f"[resume] rerunning incomplete {label} sample with "
                    f"{len(parsed)}/{expected_rows} rows: dataset={name} "
                    f"sample={sample_index}",
                    flush=True,
                )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--child-mode",
            child_mode,
            "--gpu",
            str(args.gpu),
            "--dataset-size",
            size,
            "--graph-type",
            graph_type,
            "--dataset-name",
            name,
            "--edge-path",
            edge_path,
            "--sample-index",
            str(sample_index),
            "--schedule-slot",
            str(schedule_slot),
            "--repeat",
            str(args.repeat),
            "--bfs-sources",
            str(args.bfs_sources),
            "--pr-alpha",
            str(args.pr_alpha),
            "--pr-eps",
            str(args.pr_eps),
            "--pr-max-iter",
            str(args.pr_max_iter),
        ]
        if child_mode == "isolated":
            command.extend(["--isolated-function", isolated_function])
        env = dict(os.environ)
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            log_path.write_text(
                partial + f"\nTIMEOUT: {label} exceeded {args.timeout}s\n"
            )
            raise SystemExit(
                f"natural workflow timeout: case={label} dataset={name} "
                f"sample={sample_index}; see {log_path}"
            ) from exc
        log_path.write_text(completed.stdout)
        if completed.returncode != 0:
            raise SystemExit(
                f"natural workflow failed: case={label} dataset={name} "
                f"sample={sample_index} exit={completed.returncode}; see {log_path}"
            )
        parsed = parsed_workflow_rows(log_path)
        if len(parsed) != expected_rows:
            raise SystemExit(
                f"{label} emitted {len(parsed)}/{expected_rows} rows for "
                f"{name} sample {sample_index}"
            )
        if args.cooldown > 0:
            time.sleep(args.cooldown)
        return parsed

    for size, graph_type, name, edge_path in datasets:
        for sample_index in range(1, args.repeat + 1):
            cases = [("workflow", "")] + [
                ("isolated", function) for function in WORKFLOW
            ]
            shift = (sample_index - 1) % len(cases)
            cases = cases[shift:] + cases[:shift]
            for schedule_slot, (child_mode, function) in enumerate(cases, start=1):
                samples.extend(
                    run_case(
                        size=size,
                        graph_type=graph_type,
                        name=name,
                        edge_path=edge_path,
                        sample_index=sample_index,
                        child_mode=child_mode,
                        isolated_function=function,
                        schedule_slot=schedule_slot,
                    )
                )

    call_aggregates = aggregate_sample_rows(samples, expected_samples=args.repeat)
    isolated_samples = isolated_sum_samples(samples, args.repeat)
    isolated_sum_aggregates = aggregate_sample_rows(
        isolated_samples, expected_samples=args.repeat
    )
    combined_aggregates = call_aggregates + isolated_sum_aggregates
    comparison = compare_aggregates(combined_aggregates)
    expected_comparisons = len(datasets) * 2
    if len(comparison) != expected_comparisons:
        raise SystemExit(
            f"natural workflow comparison coverage {len(comparison)}/{expected_comparisons}"
        )
    per_call = compare_per_call(call_aggregates, call_aggregates)
    expected_per_call = len(datasets) * len(WORKFLOW) * 2
    if len(per_call) != expected_per_call:
        raise SystemExit(
            f"natural workflow per-call coverage {len(per_call)}/{expected_per_call}"
        )
    beneficiaries = [row for row in per_call if row["reuse_beneficiary"]]
    expected_beneficiaries = len(datasets) * 2 * 2
    if len(beneficiaries) != expected_beneficiaries:
        raise SystemExit(
            "natural workflow reuse-beneficiary coverage "
            f"{len(beneficiaries)}/{expected_beneficiaries}"
        )
    beneficiary_summary = summarize_reuse_beneficiaries(per_call)

    write_csv(out_dir / "natural_workflow_samples.csv", samples)
    write_csv(
        out_dir / "isolated_first_use_samples.csv",
        [
            row
            for row in samples
            if row.get("baseline") == "EGGPU-isolated-first-use"
        ],
    )
    write_csv(out_dir / "natural_workflow_aggregates.csv", combined_aggregates)
    write_csv(out_dir / "isolated_first_use_sum_samples.csv", isolated_samples)
    write_csv(out_dir / "natural_workflow_vs_isolated.csv", comparison)
    write_csv(
        out_dir / "natural_workflow_per_call_vs_first_use.csv",
        per_call,
    )
    write_csv(out_dir / "natural_workflow_reuse_beneficiaries.csv", beneficiaries)
    write_csv(
        out_dir / "natural_workflow_reuse_beneficiary_summary.csv",
        beneficiary_summary,
    )
    metadata = {
        "schema_version": 2,
        "protocol": "natural_same_graph_workflow_matched_controls_v2",
        "workflow": list(WORKFLOW),
        "selection_basis": "LDBC Graphalytics core algorithms",
        "selection_source": "https://ldbcouncil.org/benchmarks/graphalytics/algorithms",
        "order_claim": "EGGPU case-study order; not prescribed by LDBC",
        "repeat": args.repeat,
        "dispersion": "sample_standard_deviation",
        "confidence_interval": "two-sided_95_percent_student_t",
        "datasets": [item[2] for item in datasets],
        "first_use_reference": str(first_use_dir),
        "isolated_control": (
            "three same-run matched independent-process controls per dataset/sample"
        ),
        "execution_order_control": (
            "deterministic cyclic rotation of workflow, isolated WCC, isolated "
            "PageRank, and isolated BFS within each repeat"
        ),
        "gpu_device_profile": gpu_device_profile,
        "host_profile": host_profile,
        "comparison_contract": {
            "position_1": "initialization_payer_and_diagnostic",
            "positions_2_3": "reuse_beneficiaries_compared_with_same_function_isolated_first_use",
            "reported_statistics": [
                "arithmetic_mean",
                "sample_standard_deviation",
                "absolute_time_saved_seconds",
                "time_saved_percent",
                "first_use_over_natural_ratio",
            ],
        },
        "source_snapshot": current_snapshot,
        "build_artifacts": {"cpp_easygraph": current_artifact_rows},
    }
    (out_dir / "natural_workflow_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    write_report(out_dir, comparison, beneficiary_summary, args.repeat)
    marker.stop()
    print(f"Done: {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument(
        "--child-mode", choices=("workflow", "isolated"), default="workflow"
    )
    parser.add_argument("--isolated-function", choices=WORKFLOW, default=None)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--first-use-dir")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--bfs-sources", type=int, default=8)
    parser.add_argument("--pr-alpha", type=float, default=0.75)
    parser.add_argument("--pr-eps", type=float, default=1.0e-6)
    parser.add_argument("--pr-max-iter", type=int, default=200)
    parser.add_argument("--dataset-size", default="")
    parser.add_argument("--graph-type", choices=["directed", "undirected"])
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--edge-path", default="")
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--schedule-slot", type=int, default=0)
    args = parser.parse_args()
    if args.child:
        required = (args.graph_type, args.dataset_name, args.edge_path)
        if not all(required):
            raise SystemExit("child mode requires graph type, dataset name, and edge path")
        if args.child_mode == "isolated" and not args.isolated_function:
            raise SystemExit("isolated child mode requires --isolated-function")
        child_main(args)
    else:
        if not args.out_dir or not args.first_use_dir:
            raise SystemExit("parent mode requires --out-dir and --first-use-dir")
        parent_main(args)


if __name__ == "__main__":
    main()
