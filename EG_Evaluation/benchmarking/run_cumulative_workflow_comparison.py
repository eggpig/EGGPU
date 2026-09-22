#!/usr/bin/env python3
"""Measure cumulative same-graph analysis from the first function call.

Each sample is a fresh process.  Host graph construction is measured but kept
outside the cumulative x-axis.  The cumulative curve therefore starts at the
first user-visible analysis call and includes every conversion, transfer,
kernel, synchronization, and usable public result produced by that call.
Benchmark-side validation starts only after the public-call timer stops.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from library_baselines import (
    build_easygraph,
    build_igraph,
    build_networkx,
    deterministic_weighted_edges,
    load_graph,
    nx_cugraph_call,
    pick_sources,
)
from run_full_baselines import DEFAULT_DATASETS
from easygraph_runtime_provenance import (
    collect_loaded_runtime_provenance,
    collect_relevant_environment,
    collect_runtime_repository_provenance,
    install_runtime_import_root,
)


DEFAULT_WORKFLOW = ("WCC", "PageRank", "BFS", "SSSP", "Closeness")
SUPPORTED_WORKFLOW_FUNCTIONS = ("WCC", "PageRank", "BFS", "SSSP", "Closeness")
EGGPU_KERNEL_KEYS = {
    "WCC": "cc",
    "PageRank": "pagerank",
    "BFS": "bfs",
    "SSSP": "sssp",
    "Closeness": "closeness",
}
BASELINES = ("EGGPU", "EGGPU-isolated", "igraph", "nx-cugraph")
BASELINE_UNSUPPORTED_FUNCTIONS = {
    "nx-cugraph": {
        "Closeness": (
            "strict NetworkX backend='cugraph' does not expose an aligned "
            "closeness_centrality implementation"
        ),
    },
}
DEFAULT_SELECTED = (
    "ca-HepTh",
    "LastFM",
    "p2p-Gnutella04",
    "ca-HepPh",
    "email-Enron",
    "ca-CondMat",
    "ER-100k",
    "soc-Slashdot0811",
)
PREFIX = "CUMULATIVE_RESULT_JSON "
SANITIZE_ENV_VARS = (
    "CC",
    "CXX",
    "GCC",
    "GXX",
    "CFLAGS",
    "CPPFLAGS",
    "CXXFLAGS",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "CPATH",
    "LIBRARY_PATH",
)
EGGPU_CUDA_ROOT = Path(
    os.environ.get(
        "EGGPU_CUDA_ROOT",
        "/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang",
    )
)
NXCUGRAPH_CUDA_ROOT = Path(
    os.environ.get(
        "NXCUGRAPH_CUDA_ROOT",
        "/home/dataset-assist-0/einwang/conda_cache/conda_env/tongyideepresearch",
    )
)
def stats(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "best": min(values),
        "count": len(values),
        "samples": values,
    }


def supported_workflow_prefix(baseline, workflow):
    """Return the longest natively supported prefix and its unsupported tail."""

    unsupported_contracts = BASELINE_UNSUPPORTED_FUNCTIONS.get(baseline, {})
    prefix = []
    unsupported = []
    blocked_by = None
    for position, function in enumerate(workflow, start=1):
        reason = unsupported_contracts.get(function)
        if blocked_by is not None:
            unsupported.append(
                {
                    "call_position": position,
                    "function": function,
                    "reason": f"not executed after unsupported step {blocked_by}",
                }
            )
        elif reason is not None:
            blocked_by = function
            unsupported.append(
                {
                    "call_position": position,
                    "function": function,
                    "reason": reason,
                }
            )
        else:
            prefix.append(function)
    return tuple(prefix), tuple(unsupported)


def graph_spec(name):
    for size, graph_type, dataset, path in DEFAULT_DATASETS:
        if dataset == name:
            return size, graph_type, dataset, path
    raise KeyError(name)


def _cuda_library_path(root):
    return ":".join(
        str(path)
        for path in (root / "lib", root / "targets" / "x86_64-linux" / "lib")
        if path.is_dir()
    )


def _pin_cuda_environment(root):
    root = Path(root).resolve()
    os.environ.update(
        {
            "CUDA_PATH": str(root),
            "CUDA_HOME": str(root),
            "CUPY_CUDA_PATH": str(root),
            "CUDAToolkit_ROOT": str(root),
            "CONDA_PREFIX": str(root),
            "LD_LIBRARY_PATH": _cuda_library_path(root),
        }
    )


def backend_subprocess_env(baseline, gpu, easygraph_repo):
    env = os.environ.copy()
    for name in SANITIZE_ENV_VARS:
        env.pop(name, None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EGGPU_MONITOR_GPU_INDEX"] = str(gpu)
    root = NXCUGRAPH_CUDA_ROOT if baseline == "nx-cugraph" else EGGPU_CUDA_ROOT
    root = root.resolve()
    env.update(
        {
            "CUDA_PATH": str(root),
            "CUDA_HOME": str(root),
            "CUPY_CUDA_PATH": str(root),
            "CUDAToolkit_ROOT": str(root),
            "CONDA_PREFIX": str(root),
            "LD_LIBRARY_PATH": _cuda_library_path(root),
        }
    )
    if baseline.startswith("EGGPU"):
        old_pythonpath = env.get("PYTHONPATH", "")
        runtime_root = str(Path(easygraph_repo).expanduser().resolve())
        env["PYTHONPATH"] = runtime_root + (
            os.pathsep + old_pythonpath if old_pythonpath else ""
        )
    return env


def configure_strict_gpu(gpu, baseline="EGGPU", isolated=False):
    cuda_root = NXCUGRAPH_CUDA_ROOT if baseline == "nx-cugraph" else EGGPU_CUDA_ROOT
    _pin_cuda_environment(cuda_root)
    if baseline == "nx-cugraph":
        nvrtc = cuda_root / "lib" / "libnvrtc.so.13"
        if not nvrtc.is_file():
            raise RuntimeError(f"nx-cugraph CUDA 13 NVRTC is missing: {nvrtc}")
        # CuPy may otherwise retain a CUDA 12 NVRTC already visible to the
        # Python environment while compiling against CUDA 13 headers.
        ctypes.CDLL(str(nvrtc), mode=ctypes.RTLD_GLOBAL)
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "EGGPU_MONITOR_GPU_INDEX": str(gpu),
            "EASYGRAPH_ENABLE_GPU": "TRUE",
            "EASYGRAPH_GPU_STRICT_ERRORS": "TRUE",
            "EASYGRAPH_GPU_RESULT_CACHE": "FALSE",
            "EASYGRAPH_GPU_ADAPTIVE_HOST": "FALSE",
            "EGGPU_EXECUTION_PROTOCOL": "first-use",
            "EGGPU_MEASUREMENT_MODE": "timing",
            "NETWORKX_FALLBACK_TO_NX": "FALSE",
            "NX_CUGRAPH_AUTOCONFIG": "TRUE",
        }
    )
    if isolated:
        # Preserve the same process, graph object, call order, kernels, and
        # return semantics while forcing every call to rebuild the reusable
        # graph state. Function-local CUDA kernels and workspaces are unchanged.
        os.environ.update(
            {
                "EASYGRAPH_GPU_DISABLE_GRAPH_CONTEXT_CACHE": "TRUE",
                "EASYGRAPH_GPU_DISABLE_CPP_GRAPH_CACHE": "TRUE",
                "EASYGRAPH_GPU_DISABLE_DEVICE_CSR_CACHE": "TRUE",
            }
        )
    else:
        for name in (
            "EASYGRAPH_GPU_DISABLE_GRAPH_CONTEXT_CACHE",
            "EASYGRAPH_GPU_DISABLE_CPP_GRAPH_CACHE",
            "EASYGRAPH_GPU_DISABLE_DEVICE_CSR_CACHE",
        ):
            os.environ.pop(name, None)


def build_graph(baseline, n, edges, directed, weighted=False):
    if baseline.startswith("EGGPU"):
        return build_easygraph(n, edges, directed, weighted=weighted)
    if baseline == "igraph":
        return build_igraph(n, edges, directed, weighted=weighted)
    if baseline == "nx-cugraph":
        return build_networkx(n, edges, directed, weighted=weighted)
    raise ValueError(baseline)


def invoke_public_call(baseline, graph, directed, function, sources):
    """Invoke one public workflow operation and return its usable result.

    Contract-required materialization belongs here.  In particular, the WCC
    workflow contract is a concrete component collection, while EasyGraph and
    NetworkX expose component iterators.  Materializing that iterator once is
    therefore part of the public operation.  Secondary scans, copies, sums,
    digests, and invariant checks belong in ``validate_public_result``.
    """

    if baseline.startswith("EGGPU"):
        import easygraph as eg

        if function == "WCC":
            return list(
                eg.weakly_connected_components(graph)
                if directed
                else eg.connected_components(graph)
            )
        if function == "PageRank":
            return eg.pagerank(
                graph,
                alpha=0.75,
                max_iter=200,
                tol=1.0e-6,
                weight=None,
            )
        if function == "BFS":
            return eg.multi_source_bfs(graph, sources, target=None)
        if function == "SSSP":
            return eg.multi_source_dijkstra(
                graph,
                sources,
                weight="weight",
                target=None,
            )
        if function == "Closeness":
            return eg.closeness_centrality(
                graph,
                weight=None,
                sources=None,
            )
        raise ValueError(f"unsupported workflow function: {function}")

    if baseline == "igraph":
        if function == "WCC":
            return graph.connected_components(
                mode="weak" if directed else "strong"
            )
        if function == "PageRank":
            return graph.pagerank(
                directed=directed, damping=0.75, weights=None
            )
        if function == "BFS":
            return graph.distances(
                source=sources,
                mode="out" if directed else "all",
            )
        if function == "SSSP":
            return graph.distances(
                source=sources,
                weights="weight",
                mode="out" if directed else "all",
                algorithm="dijkstra",
            )
        if function == "Closeness":
            return graph.closeness(
                vertices=None,
                mode="OUT" if directed else "ALL",
                weights=None,
                normalized=True,
            )
        raise ValueError(f"unsupported workflow function: {function}")

    if baseline == "nx-cugraph":
        import networkx as nx

        if function == "WCC":
            component_fn = nx.weakly_connected_components if directed else nx.connected_components
            return list(nx_cugraph_call(component_fn, graph))
        if function == "PageRank":
            return nx_cugraph_call(
                nx.pagerank,
                graph,
                alpha=0.75,
                max_iter=200,
                tol=1.0e-6,
                weight=None,
            )
        if function == "BFS":
            return {
                int(source): nx_cugraph_call(
                    nx.single_source_shortest_path_length,
                    graph,
                    int(source),
                )
                for source in sources
            }
        if function == "SSSP":
            return {
                int(source): nx_cugraph_call(
                    nx.single_source_dijkstra_path_length,
                    graph,
                    int(source),
                    weight="weight",
                )
                for source in sources
            }
        if function == "Closeness":
            raise RuntimeError(
                "strict nx-cugraph does not expose an aligned closeness_centrality implementation"
            )
        raise ValueError(f"unsupported workflow function: {function}")

    raise ValueError(baseline)


def validate_public_result(
    baseline, graph, directed, function, sources, result
):
    """Validate a completed public result outside the public-call timer."""

    if baseline.startswith("EGGPU") or baseline == "nx-cugraph":
        node_count = len(graph)
    elif baseline == "igraph":
        node_count = graph.vcount()
    else:
        raise ValueError(baseline)

    if function == "WCC":
        covered_nodes = sum(len(component) for component in result)
        valid = covered_nodes == node_count
        detail = {
            "components": len(result),
            "covered_nodes": covered_nodes,
        }
    elif function == "PageRank":
        values = (
            list(result.values()) if hasattr(result, "values") else list(result)
        )
        score_sum = float(sum(values))
        valid = len(values) == node_count and math.isclose(
            score_sum, 1.0, rel_tol=5.0e-5, abs_tol=5.0e-5
        )
        detail = {"result_size": len(values), "score_sum": score_sum}
    elif function == "BFS":
        valid = len(result) == len(sources)
        if baseline == "nx-cugraph":
            reachable_counts = [len(result[int(source)]) for source in sources]
            valid = valid and all(count > 0 for count in reachable_counts)
            detail = {
                "sources": sources,
                "reachable_counts": reachable_counts,
            }
        elif baseline == "igraph":
            valid = valid and all(len(row) == node_count for row in result)
            detail = {"sources": sources, "source_results": len(result)}
        else:
            detail = {"sources": sources, "source_results": len(result)}
    elif function == "SSSP":
        valid = len(result) == len(sources)
        if baseline == "nx-cugraph":
            reachable_counts = [len(result[int(source)]) for source in sources]
            valid = valid and all(count > 0 for count in reachable_counts)
            detail = {
                "sources": sources,
                "reachable_counts": reachable_counts,
                "weights": "deterministic nonnegative integer weights",
            }
        elif baseline == "igraph":
            valid = valid and all(len(row) == node_count for row in result)
            detail = {
                "sources": sources,
                "source_results": len(result),
                "weights": "deterministic nonnegative integer weights",
            }
        else:
            detail = {
                "sources": sources,
                "source_results": len(result),
                "weights": "deterministic nonnegative integer weights",
            }
    elif function == "Closeness":
        if baseline == "igraph":
            # This semantic adapter and its additional reachability query are
            # benchmark-side work.  They validate alignment with the workflow
            # contract but must not inflate the timed igraph API call.
            mode = "OUT" if directed else "ALL"
            reachable = graph.neighborhood_size(
                vertices=None,
                order=max(0, node_count),
                mode=mode,
            )
            denominator = max(1, node_count - 1)
            values = []
            for value, count in zip(result, reachable):
                number = float(value)
                if not math.isfinite(number):
                    number = 0.0
                values.append(
                    number * max(0, int(count) - 1) / denominator
                )
        else:
            values = [float(value) for value in result]
        valid = len(values) == node_count and all(
            math.isfinite(value) and value >= 0.0 for value in values
        )
        detail = {
            "result_size": len(values),
            "score_sum": float(sum(values)),
            "semantics": (
                "exact all-node outgoing closeness with disconnected-graph "
                "correction"
            ),
        }
    else:
        raise ValueError(f"unsupported workflow function: {function}")

    if not valid:
        raise RuntimeError(f"{baseline}/{function} result invariant failed: {detail}")
    return detail


def last_kernel_seconds(baseline, function):
    """Read the library-reported device interval after the public timer."""

    if not baseline.startswith("EGGPU"):
        return None
    from easygraph.utils import gpu_eggpu_backend

    return float(
        gpu_eggpu_backend.get_last_kernel_time(EGGPU_KERNEL_KEYS[function])
    )


def run_call_with_timeout(timeout_seconds, baseline, graph, directed, function, sources):
    """Time only one usable-result public operation under its call timeout."""

    def handle_timeout(_signum, _frame):
        raise TimeoutError(
            f"{baseline}/{function} exceeded the {timeout_seconds:.3f}s call limit"
        )

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        started = time.perf_counter()
        result = invoke_public_call(
            baseline, graph, directed, function, sources
        )
        elapsed = time.perf_counter() - started
        return result, elapsed
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def execute_workflow_call(
    timeout_seconds, baseline, graph, directed, function, sources
):
    """Execute one call, then read kernel metadata and validate off-timer."""

    result, elapsed = run_call_with_timeout(
        timeout_seconds, baseline, graph, directed, function, sources
    )
    kernel_seconds = last_kernel_seconds(baseline, function)
    detail = validate_public_result(
        baseline, graph, directed, function, sources, result
    )
    return detail, kernel_seconds, elapsed


def child(args):
    install_runtime_import_root(args.easygraph_repo)
    configure_strict_gpu(
        args.gpu,
        baseline=args.baseline,
        isolated=args.baseline == "EGGPU-isolated",
    )
    size, graph_type, dataset, relative_path = graph_spec(args.dataset)
    path = Path(args.root) / relative_path
    views = load_graph(path)
    n, directed_edges, undirected_edges = views["all_vertices"]
    directed = graph_type == "directed"
    edges = directed_edges if directed else undirected_edges
    workflow = tuple(args.workflow)
    weighted = "SSSP" in workflow
    if weighted:
        edges = deterministic_weighted_edges(n, edges)
    build_started = time.perf_counter()
    graph = build_graph(args.baseline, n, edges, directed, weighted=weighted)
    build_seconds = time.perf_counter() - build_started
    runtime_provenance = (
        collect_loaded_runtime_provenance(args.easygraph_repo)
        if args.baseline.startswith("EGGPU")
        else None
    )
    sources = pick_sources(n, args.sources)
    cumulative = 0.0
    rows = []
    for position, function in enumerate(workflow, start=1):
        detail, kernel_seconds, elapsed = execute_workflow_call(
            args.timeout,
            args.baseline,
            graph,
            directed,
            function,
            sources,
        )
        if elapsed > args.timeout:
            raise TimeoutError(
                f"{args.baseline}/{dataset}/{function} took {elapsed:.3f}s, "
                f"above the {args.timeout:.3f}s call limit"
            )
        cumulative += elapsed
        rows.append(
            {
                "status": "ok",
                "dataset": dataset,
                "dataset_size": size,
                "graph_type": graph_type,
                "baseline": args.baseline,
                "sample_index": args.sample_index,
                "call_position": position,
                "function": function,
                "build_seconds": build_seconds,
                "call_seconds": elapsed,
                "kernel_seconds": kernel_seconds,
                "cumulative_seconds": cumulative,
                "result_validation": "pass",
                "result_detail": detail,
                "workflow": list(workflow),
                "cumulative_scope": (
                    "analysis calls only; host graph construction is reported "
                    "separately and excluded from the curve; benchmark-side "
                    "validation is outside every call timer"
                ),
                "timer_boundary": (
                    "public operation invocation through its usable result "
                    "return; contract-required lazy-result materialization is "
                    "included; benchmark validation is excluded"
                ),
                "validation_outside_timer": True,
                "runtime_provenance": runtime_provenance,
            }
        )
    print(
        PREFIX
        + json.dumps(
            {
                "argv": list(sys.argv),
                "rows": rows,
                "runtime_provenance": runtime_provenance,
                "runtime_environment": collect_relevant_environment(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_child(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(PREFIX):
            payload = json.loads(line[len(PREFIX) :])
            if isinstance(payload, list):
                return {"rows": payload}
            if not isinstance(payload, dict) or not isinstance(
                payload.get("rows"), list
            ):
                raise RuntimeError("cumulative child emitted malformed result payload")
            return payload
    raise RuntimeError("cumulative child emitted no result payload")


def write_csv(path, rows):
    fields = []
    flat = []
    for source in rows:
        row = {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in source.items()
        }
        flat.append(row)
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def coordinator(args):
    requested_runtime = collect_runtime_repository_provenance(args.easygraph_repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    rows = []
    failures = []
    unsupported_rows = []
    terminal_failures = {}
    observed_eggpu_runtimes = {}
    for dataset in args.datasets:
        for baseline in args.baselines:
            _prefix, unsupported = supported_workflow_prefix(
                baseline, args.workflow
            )
            for item in unsupported:
                unsupported_rows.append(
                    {
                        "status": "unsupported",
                        "dataset": dataset,
                        "baseline": baseline,
                        **item,
                    }
                )
    tasks = [
        (dataset, baseline, sample)
        for dataset in args.datasets
        for baseline in args.baselines
        for sample in range(1, args.repeat + 1)
    ]
    for index, (dataset, baseline, sample) in enumerate(tasks, start=1):
        raw_path = raw_dir / f"{dataset}_{baseline}_{sample}.json"
        if args.resume and raw_path.is_file():
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            observed_native_sha = (
                (payload.get("runtime_provenance") or {}).get("native_sha256", "")
            )
            recorded_request = payload.get("requested_runtime_provenance") or {}
            observed_python_digest = (
                (recorded_request.get("runtime_python_snapshot") or {}).get(
                    "digest", ""
                )
            )
            runtime_matches = (
                not baseline.startswith("EGGPU")
                or (
                    observed_native_sha == requested_runtime["native_sha256"]
                    and observed_python_digest
                    == requested_runtime["runtime_python_snapshot"]["digest"]
                )
            )
            if runtime_matches:
                print(
                    f"[cumulative {index}/{len(tasks)}] reused "
                    f"{dataset}/{baseline}/{sample}"
                )
            else:
                payload = None
                print(
                    f"[cumulative {index}/{len(tasks)}] rejected stale runtime "
                    f"{dataset}/{baseline}/{sample}",
                    flush=True,
                )
        else:
            payload = None
        if payload is None and (dataset, baseline) in terminal_failures:
            source = terminal_failures[(dataset, baseline)]
            payload = {
                "status": "failed",
                "failure_kind": source["failure_kind"],
                "dataset": dataset,
                "baseline": baseline,
                "sample_index": sample,
                "execution_attempted": False,
                "derived_from_sample": source.get("sample_index"),
                "error": (
                    "not repeated after a terminal failure for the same "
                    f"dataset/baseline cell: {source.get('error', '')}"
                ),
            }
            raw_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif payload is None:
            effective_workflow, _unsupported = supported_workflow_prefix(
                baseline, args.workflow
            )
            if not effective_workflow:
                raise RuntimeError(
                    f"{baseline} supports no prefix of the requested workflow"
                )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--root",
                str(args.root),
                "--easygraph-repo",
                str(args.easygraph_repo),
                "--dataset",
                dataset,
                "--baseline",
                baseline,
                "--sample-index",
                str(sample),
                "--gpu",
                str(args.gpu),
                "--sources",
                str(args.sources),
                "--timeout",
                str(args.timeout),
                "--workflow",
                *effective_workflow,
            ]
            env = backend_subprocess_env(
                baseline, args.gpu, args.easygraph_repo
            )
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    timeout=(
                        args.load_timeout
                        + len(effective_workflow) * args.timeout
                        + 30.0
                    ),
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr[-4000:] or completed.stdout[-4000:])
                payload = {
                    "status": "ok",
                    **parse_child(completed.stdout),
                    "requested_runtime_provenance": requested_runtime,
                }
            except subprocess.TimeoutExpired as exc:
                payload = {
                    "status": "failed",
                    "failure_kind": "timeout",
                    "dataset": dataset,
                    "baseline": baseline,
                    "sample_index": sample,
                    "error": str(exc),
                }
            except Exception as exc:
                lowered = str(exc).lower()
                if "timeout" in lowered or "timed out" in lowered or "call limit" in lowered:
                    failure_kind = "timeout"
                elif "out of memory" in lowered:
                    failure_kind = "oom"
                else:
                    failure_kind = "execution_error"
                payload = {
                    "status": "failed",
                    "failure_kind": failure_kind,
                    "dataset": dataset,
                    "baseline": baseline,
                    "sample_index": sample,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if payload.get("status") == "ok":
            rows.extend(payload["rows"])
            runtime = payload.get("runtime_provenance")
            if baseline.startswith("EGGPU") and runtime:
                observed_eggpu_runtimes[
                    json.dumps(runtime, sort_keys=True)
                ] = runtime
        else:
            failures.append(payload)
            if payload.get("failure_kind") in {"timeout", "oom", "resource_limit"}:
                terminal_failures.setdefault((dataset, baseline), payload)
        print(
            f"[cumulative {index}/{len(tasks)}] {dataset}/{baseline}/{sample}: "
            f"{payload.get('status')}",
            flush=True,
        )

    write_csv(args.output_dir / "cumulative_workflow_samples.csv", rows)
    write_csv(args.output_dir / "cumulative_workflow_failures.csv", failures)
    write_csv(
        args.output_dir / "cumulative_workflow_unsupported.csv",
        unsupported_rows,
    )
    summary = []
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["baseline"], row["call_position"], row["function"])
        grouped.setdefault(key, []).append(row)
    for (dataset, baseline, position, function), group in sorted(grouped.items()):
        call = stats([row["call_seconds"] for row in group])
        cumulative = stats([row["cumulative_seconds"] for row in group])
        item = {
            "dataset": dataset,
            "baseline": baseline,
            "call_position": position,
            "function": function,
            "sample_count": len(group),
            "call_mean_seconds": call["mean"],
            "call_stdev_seconds": call["stdev"],
            "call_best_seconds": call["best"],
            "cumulative_mean_seconds": cumulative["mean"],
            "cumulative_stdev_seconds": cumulative["stdev"],
            "cumulative_best_seconds": cumulative["best"],
            "build_mean_seconds": statistics.mean(row["build_seconds"] for row in group),
        }
        kernel_values = [
            float(row["kernel_seconds"])
            for row in group
            if row.get("kernel_seconds") is not None
        ]
        if kernel_values:
            kernel = stats(kernel_values)
            item.update(
                {
                    "kernel_mean_seconds": kernel["mean"],
                    "kernel_stdev_seconds": kernel["stdev"],
                    "kernel_best_seconds": kernel["best"],
                }
            )
        summary.append(item)
    write_csv(args.output_dir / "cumulative_workflow_summary.csv", summary)
    all_function_semantics = {
        "WCC": "weak components for directed graphs; connected components for undirected graphs",
        "PageRank": "alpha=0.75, tolerance=1e-6, maximum 200 iterations, unweighted",
        "BFS": f"{args.sources} deterministic sources, all reachable distances",
        "SSSP": f"{args.sources} deterministic sources, nonnegative deterministic integer weights",
        "Closeness": "exact all-node outgoing distance with Wasserman-Faust disconnected-graph correction",
    }
    if "Closeness" in args.workflow:
        dataset_eligibility = (
            "The default candidates use exact all-node Closeness. Datasets whose "
            "main-matrix Closeness protocol is sampled-target exact are excluded. "
            "A candidate that exceeds the per-call limit is retained as an explicit "
            "timeout and excluded from matched aggregation."
        )
    else:
        dataset_eligibility = (
            "Datasets are supplied explicitly. A dataset enters matched aggregation "
            "only when every requested baseline completes every workflow function in "
            "all requested fresh-process repetitions."
        )
    metadata = {
        "protocol": "fresh_process_cumulative_same_graph_workflow_v2",
        "argv": list(sys.argv),
        "easygraph_repo": str(Path(args.easygraph_repo).expanduser().resolve()),
        "runtime_provenance": requested_runtime,
        "observed_eggpu_runtimes": list(observed_eggpu_runtimes.values()),
        "environment": {
            "coordinator": collect_relevant_environment(),
            "EGGPU_child": collect_relevant_environment(
                backend_subprocess_env(
                    "EGGPU", args.gpu, args.easygraph_repo
                )
            ),
        },
        "workflow": list(args.workflow),
        "datasets": args.datasets,
        "baselines": args.baselines,
        "repeat": args.repeat,
        "sources": args.sources,
        "function_semantics": {
            function: all_function_semantics[function]
            for function in args.workflow
        },
        "dataset_eligibility": dataset_eligibility,
        "per_call_timeout_seconds": args.timeout,
        "failures": len(failures),
        "unsupported_calls": unsupported_rows,
        "cuda_roots": {
            "EGGPU": str(EGGPU_CUDA_ROOT.resolve()),
            "nx-cugraph": str(NXCUGRAPH_CUDA_ROOT.resolve()),
        },
        "eggpu_isolated_control": (
            "same process, graph object, functions, and outputs as EGGPU; "
            "GraphContext, C++ graph-container, and device-CSR caching are "
            "disabled so each call rebuilds cross-call graph state; "
            "function-local kernels and workspaces are unchanged"
        ),
        "kernel_measurement": (
            "EGGPU and EGGPU-isolated retain the CUDA algorithm timer for each "
            "call; CPU and nx-cugraph rows do not claim a comparable kernel timer"
        ),
        "public_call_measurement": (
            "each call timer starts immediately before the public operation "
            "and stops when its usable result returns; one contract-required "
            "materialization of lazy component iterators is included, while "
            "benchmark-side scans, copies, sums, semantic-alignment checks, "
            "and invariant validation run after the timer stops"
        ),
        "validation_outside_timer": True,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--easygraph-repo",
        type=Path,
        required=True,
        help="Frozen Easy-Graph runtime containing easygraph/ and cpp_easygraph*.so.",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_SELECTED))
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--sources", type=int, default=8)
    parser.add_argument(
        "--workflow",
        nargs="+",
        choices=SUPPORTED_WORKFLOW_FUNCTIONS,
        default=list(DEFAULT_WORKFLOW),
        help="Ordered same-graph analysis calls.",
    )
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--load-timeout", type=float, default=300.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="rerun samples even when their raw result files already exist",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--dataset")
    parser.add_argument("--baseline", choices=BASELINES)
    parser.add_argument("--sample-index", type=int, default=1)
    args = parser.parse_args()
    if args.child:
        if not args.dataset or not args.baseline:
            parser.error("--child requires --dataset and --baseline")
    elif args.output_dir is None:
        parser.error("coordinator requires --output-dir")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.child:
        child(parsed)
    else:
        raise SystemExit(coordinator(parsed))
