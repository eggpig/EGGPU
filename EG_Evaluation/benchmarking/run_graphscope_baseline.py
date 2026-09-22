#!/usr/bin/env python3
"""Run the GraphScope baseline with EGGPU's split timing/memory protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


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

SUPPORT = {
    "PageRank": (
        "P",
        "native GAE PageRank; undirected inputs use a bidirected projection",
    ),
    "MST": (
        "F",
        "native FLASH MSF returns only total weight, not the required forest edge set",
    ),
    "LCC": ("T", "native GAE clustering on the common simple undirected projection"),
    "WCC": ("T", "native GAE weakly connected components"),
    "SCC": ("T", "native FLASH SCC; undirected inputs use connected components"),
    "BFS": ("T", "native FLASH BFS repeated over the benchmark source set"),
    "Dijkstra": (
        "P",
        "native FLASH nonnegative-weight SSSP; no separately named Dijkstra app",
    ),
    "BellmanFord": (
        "F",
        "no aligned callable GAE/FLASH Bellman-Ford app in GraphScope 0.29.0",
    ),
    "SSSP": ("T", "native FLASH SSSP repeated over the benchmark source set"),
    "KCore": (
        "P",
        "native FLASH k-core requires each logical undirected edge stored once",
    ),
    "BC": (
        "P",
        "specified-source BC is composed from native FLASH source-BC calls",
    ),
    "Closeness": (
        "P",
        "native exact all-node GAE closeness; directed outgoing semantics use a reversed view",
    ),
    "EffectiveSize": (
        "F",
        "no aligned callable GAE/GraphScope-NX/FLASH implementation",
    ),
    "Efficiency": (
        "F",
        "no aligned callable GAE/GraphScope-NX/FLASH implementation",
    ),
    "Constraint": (
        "F",
        "no aligned callable GAE/GraphScope-NX/FLASH implementation",
    ),
    "Hierarchy": (
        "F",
        "no aligned callable GAE/GraphScope-NX/FLASH implementation",
    ),
}

DATASET_SIZE = {
    "ca-HepTh": "small",
    "LastFM": "small",
    "p2p-Gnutella04": "small",
    "ca-HepPh": "small",
    "email-Enron": "medium",
    "ca-CondMat": "medium",
    "soc-Epinions1": "medium",
    "soc-Slashdot0811": "large",
    "ER-100k": "large",
    "web-NotreDame": "large",
    "com-youtube": "large",
    "com-Orkut": "very-large",
    "GAP-twitter": "billion-edge",
}

DATASET_ORDER = [
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

STATIC_NO_TIMING = {
    "MST": (
        "unsupported",
        "GraphScope returns total MSF weight only; an aligned MST/forest edge result cannot be reconstructed",
    ),
    "BellmanFord": ("unsupported", SUPPORT["BellmanFord"][1]),
    "EffectiveSize": ("unsupported", SUPPORT["EffectiveSize"][1]),
    "Efficiency": ("unsupported", SUPPORT["Efficiency"][1]),
    "Constraint": ("unsupported", SUPPORT["Constraint"][1]),
    "Hierarchy": ("unsupported", SUPPORT["Hierarchy"][1]),
}

DEFAULT_BENCHMARKING_DIR = Path(
    "/home/dataset-assist-0/einwang/workspace/haorandu/"
    "EGGPU_Paper_Repo/EG_Evaluation/benchmarking"
)
DEFAULT_GRAPHSCOPE_PYTHON = Path(
    "/home/dataset-assist-0/einwang/workspace/haorandu/"
    ".envs/graphscope-0.29.0/bin/python"
)
DEFAULT_WORKER = Path(__file__).with_name("graphscope_baseline_worker.py")
DEFAULT_EGGPU_LIB = Path(
    "/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/lib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--graphscope-python", type=Path, default=DEFAULT_GRAPHSCOPE_PYTHON)
    parser.add_argument("--benchmarking-dir", type=Path, default=DEFAULT_BENCHMARKING_DIR)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--functions", default="all")
    parser.add_argument("--phase", choices=("timing", "memory", "both"), default="both")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--memory-poll-ms", type=float, default=2.0)
    parser.add_argument(
        "--cleanup-grace-seconds",
        type=float,
        default=1.0,
        help=(
            "Grace period after the worker atomically writes its result before "
            "terminating GraphScope services. This occurs after all reported "
            "timing intervals and therefore does not alter measured values."
        ),
    )
    parser.add_argument(
        "--eligibility-results",
        type=Path,
        help=(
            "For a memory-only run, measure only dataset/function pairs whose "
            "aggregate E2E timing row is successful in this CSV."
        ),
    )
    parser.add_argument("--pr-alpha", type=float, default=0.75)
    parser.add_argument("--pr-tol", type=float, default=1.0e-6)
    parser.add_argument("--pr-max-iter", type=int, default=200)
    parser.add_argument("--sssp-sources", type=int, default=8)
    parser.add_argument("--bc-sources", type=int, default=16)
    parser.add_argument("--closeness-sources", type=int, default=16)
    parser.add_argument(
        "--closeness-sampled-datasets",
        default=(
            "ER-100k,soc-Slashdot0811,web-NotreDame,com-youtube,"
            "com-Orkut,GAP-twitter"
        ),
    )
    parser.add_argument("--detail-node-limit", type=int, default=5_000_000)
    return parser.parse_args()


def csv_selection(raw: str, allowed: list[str]) -> list[str]:
    if raw.strip().lower() == "all":
        return list(allowed)
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise SystemExit(f"unknown selection: {', '.join(unknown)}")
    return selected


def load_manifests(root: Path, selected: str) -> list[tuple[Path, dict]]:
    candidates = []
    for path in sorted(root.glob("*/manifest.json")):
        metadata = json.loads(path.read_text())
        candidates.append((path, metadata))
    by_name = {metadata["name"]: (path, metadata) for path, metadata in candidates}
    available = [name for name in DATASET_ORDER if name in by_name]
    available.extend(sorted(set(by_name) - set(available)))
    names = csv_selection(selected, available)
    return [by_name[name] for name in names]


def load_eligible_pairs(path: Path | None) -> set[tuple[str, str]] | None:
    if path is None:
        return None
    eligible: set[tuple[str, str]] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("baseline") == "GraphScope"
                and row.get("metric") == "e2e"
                and row.get("status") == "ok"
                and (row.get("seconds") or row.get("value"))
            ):
                eligible.add((row["dataset"], row["function"]))
    if not eligible:
        raise SystemExit(f"no successful GraphScope E2E pairs found in {path}")
    return eligible


def process_tree_rss(root_pid: int) -> int:
    try:
        root = psutil.Process(root_pid)
    except (psutil.Error, ProcessLookupError):
        return 0
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        pass
    total = 0
    for process in processes:
        try:
            total += int(process.memory_info().rss)
        except psutil.Error:
            pass
    return total


def snapshot_descendants(root_pid: int) -> dict[int, psutil.Process]:
    """Return currently visible descendants keyed by PID."""

    try:
        root = psutil.Process(root_pid)
        return {child.pid: child for child in root.children(recursive=True)}
    except psutil.Error:
        return {}


def terminate_process_tree(
    process: subprocess.Popen,
    observed_descendants: dict[int, psutil.Process] | None = None,
) -> None:
    """Terminate GraphScope services even when they create new process groups."""

    descendants = dict(observed_descendants or {})
    descendants.update(snapshot_descendants(process.pid))
    descendant_processes = list(descendants.values())

    # Snapshot descendants before terminating the worker. GraphScope's
    # coordinator starts services in their own sessions; after the worker
    # exits those processes are reparented and can no longer be discovered
    # through the worker PID.
    for descendant in reversed(descendant_processes):
        try:
            descendant.terminate()
        except psutil.Error:
            pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    _, alive = psutil.wait_procs(descendant_processes, timeout=5)
    for descendant in alive:
        try:
            descendant.kill()
        except psutil.Error:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=5)

    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_child(
    command: list[str],
    log_path: Path,
    result_path: Path,
    environment: dict[str, str],
    timeout: float,
    monitor_memory: bool,
    poll_ms: float,
    cleanup_grace_seconds: float,
) -> tuple[str, dict | None, dict]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()

    started = time.perf_counter()
    with log_path.open("w") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
            start_new_session=True,
        )
        deadline = started + timeout
        result_seen_at = None
        observed_descendants: dict[int, psutil.Process] = {}
        start_rss = None
        peak_rss = 0
        samples = 0
        status = "failed"
        while True:
            now = time.perf_counter()
            observed_descendants.update(snapshot_descendants(process.pid))
            if monitor_memory:
                rss = process_tree_rss(process.pid)
                samples += 1
                if start_rss is None:
                    start_rss = rss
                peak_rss = max(peak_rss, rss)

            if result_path.is_file() and result_seen_at is None:
                result_seen_at = now
            return_code = process.poll()
            if return_code is not None:
                status = "exited"
                break
            if result_seen_at is None and now >= deadline:
                status = "timeout"
                log.write(f"\nTIMEOUT after {timeout:.3f} seconds\n")
                log.flush()
                terminate_process_tree(process, observed_descendants)
                break
            if (
                result_seen_at is not None
                and now - result_seen_at >= cleanup_grace_seconds
            ):
                status = "cleanup_timeout"
                log.write(
                    "\nWorker result completed; forced GraphScope cleanup after "
                    f"{cleanup_grace_seconds:.3f} seconds\n"
                )
                log.flush()
                terminate_process_tree(process, observed_descendants)
                break
            time.sleep(max(0.002, poll_ms / 1000.0 if monitor_memory else 0.05))

        # A worker can exit after an engine-side failure while its coordinator,
        # Vineyard, or GRAPE services survive as reparented processes. Clean
        # every descendant observed during the run on all exit paths.
        if observed_descendants:
            terminate_process_tree(process, observed_descendants)

        if monitor_memory:
            rss = process_tree_rss(process.pid)
            samples += 1
            peak_rss = max(peak_rss, rss)

    elapsed = time.perf_counter() - started
    payload = None
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text())
        except Exception:
            payload = None
    memory = {
        "rss_mb": peak_rss / (1024.0 * 1024.0) if peak_rss else None,
        "rss_start_mb": start_rss / (1024.0 * 1024.0)
        if start_rss is not None
        else None,
        "rss_peak_delta_mb": max(0, peak_rss - (start_rss or 0))
        / (1024.0 * 1024.0)
        if peak_rss
        else None,
        "monitor_rss_samples": samples,
        "monitor_poll_ms": poll_ms,
        "monitor_window_seconds": elapsed,
    }
    if payload and payload.get("status") == "ok":
        status = "ok"
    elif status == "exited":
        status = "failed"
    return status, payload, memory


def descriptor_modules(benchmarking_dir: Path):
    sys.path.insert(0, str(benchmarking_dir))
    from benchmark_stats import aggregate_sample_rows
    from measurement_schema import describe_metric, schema_document

    return aggregate_sample_rows, describe_metric, schema_document


def pick_sources(
    n: int, k: int, preferred: list[int] | None = None
) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if preferred is not None:
        normalized = [int(source) for source in preferred]
        if len(normalized) < min(k, n):
            raise ValueError(
                "benchmark_sources_zero_based does not contain enough sources"
            )
        selected = normalized[: min(k, n)]
        if len(selected) != len(set(selected)):
            raise ValueError("benchmark source selection contains duplicates")
        if any(source < 0 or source >= n for source in selected):
            raise ValueError("benchmark source is outside the graph node domain")
        return selected
    if k >= n:
        return list(range(n))
    step = max(1, n // k)
    sources = list(range(0, n, step))[:k]
    if len(sources) < k:
        seen = set(sources)
        candidate = n - 1
        while len(sources) < k and candidate >= 0:
            if candidate not in seen:
                sources.append(candidate)
                seen.add(candidate)
            candidate -= 1
    return sources


def source_metadata(
    function: str, metadata: dict, args: argparse.Namespace
) -> dict:
    n = int(metadata["num_nodes"])
    dataset = str(metadata["name"])
    source_count = 0
    semantic = ""
    estimator_kind = ""
    if function in {"BFS", "SSSP"}:
        source_count = args.sssp_sources
        semantic = "exact_selected_sources"
        estimator_kind = "exact"
    elif function == "Dijkstra":
        source_count = 1
        semantic = "exact_selected_sources_nonnegative_weights"
        estimator_kind = "exact"
    elif function == "BC":
        source_count = args.bc_sources
        semantic = "exact_specified_source_subset_unnormalized"
        estimator_kind = "exact"
    elif function == "Closeness":
        sampled = {
            item.strip()
            for item in args.closeness_sampled_datasets.split(",")
            if item.strip()
        }
        source_count = args.closeness_sources if dataset in sampled else 0
        semantic = "exact_selected_vertices" if source_count else "exact_all_node"
        estimator_kind = "exact"
    preferred = metadata.get("benchmark_sources_zero_based")
    sources = pick_sources(n, source_count, preferred) if source_count else []
    source_sha = ""
    if sources:
        import hashlib

        source_sha = hashlib.sha256(
            ",".join(str(node) for node in sources).encode("ascii")
        ).hexdigest()[:16]
    return {
        "semantic": semantic,
        "estimator_kind": estimator_kind,
        "sample_sources": str(source_count) if source_count else "",
        "source_policy": (
            "manifest_benchmark_sources_zero_based"
            if source_count and preferred is not None
            else "deterministic_evenly_spaced"
            if source_count
            else ""
        ),
        "source_seed": "none" if source_count else "",
        "source_nodes_sha": source_sha,
    }


def make_row(
    describe_metric,
    metadata: dict,
    function: str,
    metric: str,
    value,
    status: str,
    log: Path,
    notes: str,
    correctness: str,
    sample_index: int,
    sample_count: int,
    phase: str,
    args: argparse.Namespace,
) -> dict:
    descriptor = describe_metric(metric, baseline="GraphScope")
    if phase == "memory":
        descriptor["measurement_window"] = "isolated_memory_subprocess"
    formatted = "" if value is None else f"{float(value):.12g}"
    return {
        "dataset_size": DATASET_SIZE.get(metadata["name"], ""),
        "graph_type": metadata["graph_type"],
        "dataset": metadata["name"],
        "function": function,
        "baseline": "GraphScope",
        "metric": metric,
        "seconds": formatted,
        "value": formatted,
        **descriptor,
        "status": status,
        "correctness": correctness,
        "log": str(log),
        "notes": notes,
        **source_metadata(function, metadata, args),
        "skip_reason": notes if status != "ok" else "",
        "sample_index": str(sample_index),
        "sample_count": str(sample_count),
        "measurement_phase": phase,
        "external_cli_wall_seconds": "",
        "support_label": SUPPORT[function][0],
    }


def append_static_rows(
    rows: list[dict],
    describe_metric,
    metadata: dict,
    function: str,
    phase: str,
    args: argparse.Namespace,
) -> None:
    status, reason = STATIC_NO_TIMING[function]
    metrics = (
        ("build", "e2e", "kernel")
        if phase == "timing"
        else ("memory_peak_rss_mb",)
    )
    expected = args.repeat if phase == "timing" else args.memory_repeat
    log = args.output / "logs" / metadata["name"] / f"graphscope_{function.lower()}_{phase}.log"
    for metric in metrics:
        rows.append(
            make_row(
                describe_metric,
                metadata,
                function,
                metric,
                None,
                "unsupported",
                log,
                f"{status}: {reason}",
                "",
                1,
                expected,
                phase,
                args,
            )
        )


def payload_correctness(payload: dict) -> str:
    summary = payload.get("result_summary") or {}
    ordered = (
        ("nodes", "nodes"),
        ("vertices", "vertices"),
        ("sources", "source_count"),
        ("reachable", "reachable"),
        ("checksum", "checksum"),
        ("components", "components"),
        ("partition_sha256", "partition_sha256"),
        ("finite", "finite_count"),
        ("sum", "sum"),
        ("mean", "mean"),
        ("max", "max"),
        ("detail_sha", "detail_sha"),
    )
    fields = [
        f"{label}={summary[key]}"
        for label, key in ordered
        if key in summary and summary[key] is not None
    ]
    fields.append(f"shape={summary.get('shape')}")
    if payload.get("detail_path"):
        fields.append(f"detail={payload['detail_path']}")
    return ", ".join(fields)


def run_phase(
    phase: str,
    manifests: list[tuple[Path, dict]],
    functions: list[str],
    rows: list[dict],
    describe_metric,
    args: argparse.Namespace,
    eligible_pairs: set[tuple[str, str]] | None = None,
) -> None:
    repeat = args.repeat if phase == "timing" else args.memory_repeat
    monitor_memory = phase == "memory"
    total = len(manifests) * len(functions) * repeat
    completed = 0
    for manifest_path, metadata in manifests:
        for function in functions:
            pair = (metadata["name"], function)
            if phase == "memory" and eligible_pairs is not None and pair not in eligible_pairs:
                completed += repeat
                print(
                    f"[progress] {phase} {completed}/{total} "
                    f"{metadata['name']} {function}: skipped (timing not successful)",
                    flush=True,
                )
                continue
            if function in STATIC_NO_TIMING:
                append_static_rows(
                    rows, describe_metric, metadata, function, phase, args
                )
                write_csv(args.output / "results_samples.partial.csv", rows)
                completed += repeat
                print(
                    f"[progress] {phase} {completed}/{total} "
                    f"{metadata['name']} {function}: static {STATIC_NO_TIMING[function][0]}",
                    flush=True,
                )
                continue
            if function in {"LCC", "KCore"} and not metadata.get("undirected"):
                reason = (
                    "dataset-specific aligned simple undirected projection is unavailable; "
                    "the function is supported by GraphScope but this input cannot be "
                    "evaluated under the common graph semantics"
                )
                previous = STATIC_NO_TIMING.get(function)
                STATIC_NO_TIMING[function] = ("dataset_view_unavailable", reason)
                append_static_rows(
                    rows, describe_metric, metadata, function, phase, args
                )
                if previous is None:
                    STATIC_NO_TIMING.pop(function, None)
                else:
                    STATIC_NO_TIMING[function] = previous
                write_csv(args.output / "results_samples.partial.csv", rows)
                completed += repeat
                continue

            for sample_index in range(1, repeat + 1):
                completed += 1
                sample_dir = (
                    args.output
                    / "samples"
                    / phase
                    / metadata["name"]
                    / function
                    / f"r{sample_index}"
                )
                result_path = sample_dir / "worker_result.json"
                log_path = sample_dir / "worker.log"
                detail_path = None
                if (
                    phase == "timing"
                    and sample_index == 1
                    and int(metadata["num_nodes"]) <= args.detail_node_limit
                ):
                    detail_path = (
                        args.output
                        / "details"
                        / metadata["name"]
                        / f"GraphScope_{function}.npz"
                    )
                command = [
                    str(args.graphscope_python),
                    str(args.worker),
                    "--manifest",
                    str(manifest_path),
                    "--function",
                    function,
                    "--output",
                    str(result_path),
                    "--pr-alpha",
                    str(args.pr_alpha),
                    "--pr-tol",
                    str(args.pr_tol),
                    "--pr-max-iter",
                    str(args.pr_max_iter),
                    "--sssp-sources",
                    str(args.sssp_sources),
                    "--bc-sources",
                    str(args.bc_sources),
                ]
                sampled_closeness = {
                    item.strip()
                    for item in args.closeness_sampled_datasets.split(",")
                    if item.strip()
                }
                closeness_sources = (
                    args.closeness_sources
                    if metadata["name"] in sampled_closeness
                    else 0
                )
                command.extend(["--closeness-sources", str(closeness_sources)])
                if phase == "timing" and sample_index == 1:
                    command.append("--validation-sample")
                if detail_path is not None:
                    command.extend(["--detail-output", str(detail_path)])
                environment = dict(os.environ)
                old_ld = environment.get("LD_LIBRARY_PATH", "")
                environment["LD_LIBRARY_PATH"] = (
                    str(DEFAULT_EGGPU_LIB)
                    if not old_ld
                    else f"{DEFAULT_EGGPU_LIB}:{old_ld}"
                )
                status, payload, memory = run_child(
                    command,
                    log_path,
                    result_path,
                    environment,
                    args.timeout,
                    monitor_memory,
                    args.memory_poll_ms,
                    args.cleanup_grace_seconds,
                )
                support_label, support_note = SUPPORT[function]
                notes = (
                    f"GraphScope 0.29.0; support={support_label}; {support_note}; "
                    "normalized input preprocessing excluded; session startup excluded "
                    "from build; no fallback"
                )
                correctness = payload_correctness(payload) if payload else ""
                if status == "ok" and payload:
                    if phase == "timing":
                        for metric, value in (
                            ("build", payload["build_seconds"]),
                            ("e2e", payload["e2e_seconds"]),
                            ("kernel", payload["algorithm_seconds"]),
                        ):
                            rows.append(
                                make_row(
                                    describe_metric,
                                    metadata,
                                    function,
                                    metric,
                                    value,
                                    "ok",
                                    log_path,
                                    notes
                                    + f"; graph_view={payload.get('graph_view')}; "
                                    f"materialization_seconds={payload.get('materialization_seconds')}",
                                    correctness,
                                    sample_index,
                                    repeat,
                                    phase,
                                    args,
                                )
                            )
                    else:
                        for metric, value in (
                            ("memory_peak_rss_mb", memory["rss_mb"]),
                            ("memory_start_rss_mb", memory["rss_start_mb"]),
                            ("memory_peak_rss_delta_mb", memory["rss_peak_delta_mb"]),
                            ("memory_monitor_rss_samples", memory["monitor_rss_samples"]),
                            ("memory_monitor_poll_ms", memory["monitor_poll_ms"]),
                            ("memory_monitor_window_seconds", memory["monitor_window_seconds"]),
                        ):
                            rows.append(
                                make_row(
                                    describe_metric,
                                    metadata,
                                    function,
                                    metric,
                                    value,
                                    "ok",
                                    log_path,
                                    notes + "; isolated complete process-tree RSS",
                                    correctness,
                                    sample_index,
                                    repeat,
                                    phase,
                                    args,
                                )
                            )
                else:
                    failure_note = (
                        f"{notes}; run_status={status}; "
                        f"worker_error={(payload or {}).get('error', 'no result payload')}"
                    )
                    metrics = (
                        ("build", "e2e", "kernel")
                        if phase == "timing"
                        else ("memory_peak_rss_mb",)
                    )
                    row_status = "timeout" if status == "timeout" else "failed"
                    for metric in metrics:
                        rows.append(
                            make_row(
                                describe_metric,
                                metadata,
                                function,
                                metric,
                                None,
                                row_status,
                                log_path,
                                failure_note,
                                correctness,
                                sample_index,
                                repeat,
                                phase,
                                args,
                            )
                        )

                print(
                    f"[progress] {phase} {completed}/{total} "
                    f"{metadata['name']} {function} r{sample_index}/{repeat}: {status}",
                    flush=True,
                )
                write_csv(args.output / "results_samples.partial.csv", rows)
                if status in {"timeout", "failed"}:
                    # Deterministic failures and 100-second timeouts are explicit
                    # outcomes; repeating them wastes compute and cannot produce
                    # a publishable five-sample aggregate.
                    completed += repeat - sample_index
                    break


def fieldnames(rows: list[dict]) -> list[str]:
    ordered = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_value(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> None:
    args = parse_args()
    aggregate_sample_rows, describe_metric, schema_document = descriptor_modules(
        args.benchmarking_dir
    )
    manifests = load_manifests(args.manifest_root, args.datasets)
    functions = csv_selection(args.functions, FUNCTIONS)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.eligibility_results and args.phase != "memory":
        raise SystemExit("--eligibility-results is valid only with --phase memory")
    eligible_pairs = load_eligible_pairs(args.eligibility_results)
    phases = ("timing", "memory") if args.phase == "both" else (args.phase,)
    sample_rows: list[dict] = []
    started = datetime.now().isoformat(timespec="seconds")
    for phase in phases:
        run_phase(
            phase,
            manifests,
            functions,
            sample_rows,
            describe_metric,
            args,
            eligible_pairs,
        )

    write_csv(args.output / "results_samples.csv", sample_rows)
    timing_rows = [row for row in sample_rows if row["measurement_phase"] == "timing"]
    memory_rows = [row for row in sample_rows if row["measurement_phase"] == "memory"]
    aggregate_rows = []
    if timing_rows:
        aggregate_rows.extend(
            aggregate_sample_rows(timing_rows, expected_samples=args.repeat)
        )
    if memory_rows:
        aggregate_rows.extend(
            aggregate_sample_rows(memory_rows, expected_samples=args.memory_repeat)
        )
    write_csv(args.output / "results_long.csv", aggregate_rows)
    for metric, filename in (
        ("build", "results_build.csv"),
        ("kernel", "results_kernel.csv"),
        ("e2e", "results_e2e.csv"),
    ):
        write_csv(
            args.output / filename,
            [row for row in aggregate_rows if row["metric"] == metric],
        )
    write_csv(
        args.output / "results_memory.csv",
        [row for row in aggregate_rows if row["metric"].startswith("memory_")],
    )

    schema = schema_document()
    schema["graphscope_extension"] = {
        "build": "GraphScope-native graph loading and projection",
        "kernel": "native GAE/FLASH application wall-time surrogate",
        "e2e": "native application plus context transfer and aligned host result reconstruction",
        "memory": "absolute peak RSS of the isolated Python/GraphScope/Vineyard process tree",
        "excluded": "offline normalized CSV generation and GraphScope session startup",
    }
    (args.output / "measurement_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True)
    )

    graphscope_repo = Path(
        "/home/dataset-assist-0/einwang/workspace/haorandu/GraphScope_baseline"
    )
    versions = {
        "GraphScope": {
            "runtime_version": "0.29.0",
            "runtime_distribution": "PyPI graphscope==0.29.0",
            "source_crosscheck_commit": git_value(
                graphscope_repo, "rev-parse", "HEAD"
            ),
            "source_crosscheck_dirty": bool(
                git_value(graphscope_repo, "status", "--short")
            ),
            "python": str(args.graphscope_python),
            "qualification": "native GAE/FLASH only; no NetworkX fallback",
            "support": {function: SUPPORT[function][0] for function in FUNCTIONS},
        }
    }
    (args.output / "baseline_versions.json").write_text(
        json.dumps(versions, indent=2, sort_keys=True)
    )
    metadata = {
        "started_at": started,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": [metadata["name"] for _, metadata in manifests],
        "functions": functions,
        "timing_repeat": args.repeat,
        "memory_repeat": args.memory_repeat,
        "memory_eligibility_results": (
            str(args.eligibility_results) if args.eligibility_results else None
        ),
        "timeout_seconds": args.timeout,
        "post_result_cleanup_grace_seconds": args.cleanup_grace_seconds,
        "timing_estimator": "arithmetic_mean",
        "memory_estimator": "arithmetic_mean",
        "graphscope_session_workers": 1,
        "preprocessing_in_timed_build": False,
        "command": sys.argv,
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
