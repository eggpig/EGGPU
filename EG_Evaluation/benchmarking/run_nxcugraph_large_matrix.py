#!/usr/bin/env python3
"""Run the strict nx-cugraph portion of the 13-dataset large-graph matrix.

The input is the same normalized host CSR used by EGGPU.  Graph preparation
is recorded separately.  Timings include strict NetworkX ``backend='cugraph'``
dispatch and Python-compatible result construction; CPU and native-cuGraph
fallbacks are never used.  Every one of the 16 paper functions receives an
explicit outcome, including unsupported and representation-limited entries.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from child_process_memory_monitor import ChildProcessMemoryMonitor
from nxcugraph_device_timer import NxCugraphDeviceTimer


ALL_FUNCTIONS = (
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
)
ALIGNED_CALLABLES = {
    "PageRank",
    "LCC",
    "WCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
}
WEIGHTED_FUNCTIONS = {"Dijkstra", "BellmanFord", "SSSP"}
PROJECTED_FUNCTIONS = {"LCC", "KCore"}
EXPECTED_DEVICE_BOUNDARIES = {
    "PageRank": ("pagerank",),
    "LCC": ("triangle_count",),
    "WCC": ("weakly_connected_components",),
    "BFS": ("bfs",),
    "Dijkstra": ("sssp",),
    "BellmanFord": ("sssp",),
    "SSSP": ("sssp",),
    "KCore": ("core_number",),
}
UNSUPPORTED_REASONS = {
    "MST": "nx-cugraph exposes no aligned minimum-spanning-forest backend",
    "SCC": "nx-cugraph exposes no strongly_connected_components backend",
    "BC": (
        "ordinary all-source BC exists, but the paper workload is exact "
        "16-source betweenness_centrality_subset, which the cugraph backend "
        "does not implement"
    ),
    "Closeness": "nx-cugraph exposes no closeness_centrality backend",
    "EffectiveSize": "nx-cugraph exposes no Burt effective-size backend",
    "Efficiency": "nx-cugraph exposes no Burt efficiency backend",
    "Constraint": "nx-cugraph exposes no Burt constraint backend",
    "Hierarchy": "nx-cugraph exposes no Burt hierarchy backend",
}


def sample_stats(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "best": min(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": values,
        "count": len(values),
    }


def manifest_for_function(path: Path, function: str) -> Path:
    if function not in WEIGHTED_FUNCTIONS:
        return path
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("weights_path"):
        return path
    sibling = path.with_name(f"{path.stem}.weighted.json")
    return sibling if sibling.is_file() else path


def applicability(metadata: dict, function: str):
    if function in UNSUPPORTED_REASONS:
        kind = "semantic_mismatch" if function == "BC" else "unsupported_api"
        return False, kind, UNSUPPORTED_REASONS[function]
    if function in PROJECTED_FUNCTIONS and bool(metadata.get("directed")):
        projected_entries = 2 * int(metadata.get("num_entries", 0))
        return (
            False,
            "representation_limit",
            f"{function} requires the common undirected projection; it may "
            f"contain {projected_entries:,} adjacency entries and exceeds the "
            "current signed-int32 graph ABI",
        )
    if function in WEIGHTED_FUNCTIONS and not metadata.get("weights_path"):
        return (
            False,
            "missing_weight_artifact",
            "aligned deterministic weights are absent; generate the weighted "
            "sibling manifest before running this function",
        )
    if function not in ALIGNED_CALLABLES:
        return False, "unsupported_api", "no aligned strict nx-cugraph call"
    return True, "", ""


def configure_cuda_runtime():
    candidates = []
    configured = os.environ.get("NX_CUGRAPH_NVRTC_PATH")
    if configured:
        candidates.append(Path(configured))
    candidates.append(
        Path(
            "/home/dataset-assist-0/einwang/conda_cache/conda_env/"
            "tongyideepresearch/lib/libnvrtc.so.13"
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            prefix = candidate.parents[1]
            os.environ.setdefault("CUDA_PATH", str(prefix))
            return str(candidate)
    return "dynamic-loader-default"


def configure_networkx_backend_contract():
    import networkx as nx

    nx.config.fallback_to_nx = False
    nx.config.cache_converted_graphs = True
    contract = {
        "networkx_fallback_to_nx": bool(nx.config.fallback_to_nx),
        "networkx_cache_converted_graphs": bool(
            nx.config.cache_converted_graphs
        ),
    }
    if (
        contract["networkx_fallback_to_nx"]
        or not contract["networkx_cache_converted_graphs"]
    ):
        raise RuntimeError(
            "strict nx-cugraph protocol requires CPU fallback disabled and "
            "NetworkX converted-graph caching enabled"
        )
    return contract


def load_device_graph(manifest_path: Path):
    nvrtc = configure_cuda_runtime()
    import cupy as cp
    import nx_cugraph as nxcg

    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    started = time.perf_counter()
    host_offsets = np.memmap(
        root / metadata["offsets_path"],
        dtype=np.int32,
        mode="r",
        shape=(int(metadata["num_nodes"]) + 1,),
    )
    host_indices = np.memmap(
        root / metadata["indices_path"],
        dtype=np.int32,
        mode="r",
        shape=(int(metadata["num_entries"]),),
    )
    offsets = cp.asarray(host_offsets)
    destinations = cp.asarray(host_indices)
    sources = cp.empty(int(metadata["num_entries"]), dtype=cp.int32)
    fill_sources = cp.RawKernel(
        r'''
        extern "C" __global__ void csr_to_src(
            const int *offsets, int *sources, int num_nodes) {
          int node = blockDim.x * blockIdx.x + threadIdx.x;
          if (node < num_nodes) {
            for (int edge = offsets[node]; edge < offsets[node + 1]; ++edge) {
              sources[edge] = node;
            }
          }
        }
        ''',
        "csr_to_src",
    )
    blocks = (int(metadata["num_nodes"]) + 255) // 256
    fill_sources((blocks,), (256,), (offsets, sources, int(metadata["num_nodes"])))
    edge_values = None
    host_weights = None
    if metadata.get("weights_path"):
        host_weights = np.memmap(
            root / metadata["weights_path"],
            dtype=np.float64,
            mode="r",
            shape=(int(metadata["num_entries"]),),
        )
        edge_values = {"weight": cp.asarray(host_weights)}
    graph_class = nxcg.CudaDiGraph if metadata["directed"] else nxcg.CudaGraph
    graph = graph_class.from_coo(
        int(metadata["num_nodes"]),
        sources,
        destinations,
        edge_values=edge_values,
        use_compat_graph=False,
    )
    cp.cuda.runtime.deviceSynchronize()
    preparation_seconds = time.perf_counter() - started
    return metadata, graph, preparation_seconds, nvrtc


def select_sources(metadata: dict, count: int):
    recorded = metadata.get("benchmark_sources_zero_based")
    if isinstance(recorded, list) and len(recorded) >= count:
        return [int(value) for value in recorded[:count]]
    n = int(metadata["num_nodes"])
    if n <= 0:
        return []
    return sorted({min(n - 1, index * n // max(1, count)) for index in range(count)})


def strict_public_call(graph, metadata: dict, function: str, source_count: int):
    """Return the complete strict public-API result without benchmark checks."""

    import networkx as nx

    if function == "PageRank":
        result = pagerank_public_call(graph)
    elif function == "LCC":
        result = nx.clustering(graph, backend="cugraph")
    elif function == "WCC":
        component_call = (
            nx.weakly_connected_components
            if metadata["directed"]
            else nx.connected_components
        )
        result = list(component_call(graph, backend="cugraph"))
    elif function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        count = 1 if function == "Dijkstra" else source_count
        sources = select_sources(metadata, count)
        result = {}
        for source in sources:
            if function == "BFS":
                values = nx.single_source_shortest_path_length(
                    graph, source, backend="cugraph"
                )
            elif function == "BellmanFord":
                values = nx.single_source_bellman_ford_path_length(
                    graph, source, weight="weight", backend="cugraph"
                )
            else:
                values = nx.single_source_dijkstra_path_length(
                    graph, source=source, weight="weight", backend="cugraph"
                )
            result[int(source)] = values
    elif function == "KCore":
        result = nx.core_number(graph, backend="cugraph")
    else:
        raise NotImplementedError(function)
    return result


def validate_strict_result(result, metadata: dict, function: str, source_count: int):
    """Validate a complete result after both paper timers have stopped."""

    n = int(metadata["num_nodes"])
    detail = {}
    if function == "PageRank":
        values = list(result.values())
        detail = {
            "result_size": len(result),
            "score_sum": float(sum(values)),
            "minimum": float(min(values)) if values else 0.0,
        }
        valid = (
            len(result) == n
            and detail["minimum"] >= -1.0e-12
            and math.isclose(detail["score_sum"], 1.0, rel_tol=5.0e-5, abs_tol=5.0e-5)
        )
    elif function == "LCC":
        values = list(result.values())
        detail = {
            "result_size": len(result),
            "minimum": float(min(values)) if values else 0.0,
            "maximum": float(max(values)) if values else 0.0,
            "sum": float(sum(values)),
        }
        valid = len(result) == n and detail["minimum"] >= 0 and detail["maximum"] <= 1.0 + 1e-9
    elif function == "WCC":
        covered = int(sum(len(component) for component in result))
        detail = {"component_count": len(result), "covered_nodes": covered}
        valid = covered == n
    elif function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        count = 1 if function == "Dijkstra" else source_count
        sources = select_sources(metadata, count)
        reachable = [len(values) for values in result.values()]
        checksums = [float(sum(values.values())) for values in result.values()]
        detail = {
            "sources": sources,
            "reachable_counts": reachable,
            "distance_checksums": checksums,
        }
        valid = len(result) == len(sources) and all(value > 0 for value in reachable)
    elif function == "KCore":
        values = list(result.values())
        detail = {
            "result_size": len(result),
            "minimum": int(min(values)) if values else 0,
            "maximum": int(max(values)) if values else 0,
            "sum": int(sum(values)),
        }
        valid = (
            len(result) == n
            and detail["minimum"] >= 0
            and detail["maximum"] <= int(metadata.get("max_degree", n))
        )
    else:
        raise NotImplementedError(function)
    if not valid:
        raise RuntimeError(f"result validation failed: {detail}")
    return detail


def pagerank_public_call(graph):
    """Invoke the strict public nx-cugraph PageRank API."""

    import networkx as nx

    if nx.config.fallback_to_nx:
        raise RuntimeError("NetworkX CPU fallback must remain disabled")
    return nx.pagerank(
        graph,
        alpha=0.75,
        tol=1.0e-6,
        max_iter=200,
        backend="cugraph",
    )


def validate_pagerank_result(result, expected_size: int):
    """Validate a returned public result after the benchmark timer stops."""

    values = list(result.values())
    detail = {
        "result_size": len(result),
        "score_sum": float(sum(values)),
        "minimum": float(min(values)) if values else 0.0,
        "maximum": float(max(values)) if values else 0.0,
    }
    valid = (
        len(result) == expected_size
        and detail["minimum"] >= -1.0e-12
        and math.isclose(
            detail["score_sum"], 1.0, rel_tol=5.0e-5, abs_tol=5.0e-5
        )
    )
    if not valid:
        raise RuntimeError(f"result validation failed: {detail}")
    return detail


def call_with_public_return_boundary(
    graph, metadata, function, source_count, device_timer
):
    """Measure public return and backend device intervals on the same call."""

    device_timer.reset()
    started = time.perf_counter()
    result = device_timer.call(
        lambda: strict_public_call(graph, metadata, function, source_count)
    )
    elapsed = time.perf_counter() - started
    kernel_seconds = device_timer.validate_against_public_wall(elapsed)
    expected_count = (
        1
        if function in {"PageRank", "LCC", "WCC", "Dijkstra", "KCore"}
        else int(source_count)
    )
    timer_provenance = device_timer.validate_provenance(
        expected_backends=EXPECTED_DEVICE_BOUNDARIES[function],
        expected_interval_count=expected_count,
    )
    detail = validate_strict_result(
        result, metadata, function, source_count
    )
    return result, detail, elapsed, kernel_seconds, timer_provenance


def worker(args) -> int:
    record = {}
    try:
        metadata, graph, preparation_seconds, nvrtc = load_device_graph(args.manifest)
        backend_contract = configure_networkx_backend_contract()
        with NxCugraphDeviceTimer() as device_timer:
            warmup_details = []
            for warmup_index in range(args.warmup_calls):
                (
                    warmup_result,
                    warmup_detail,
                    warmup_seconds,
                    warmup_kernel_seconds,
                    warmup_timer_provenance,
                ) = (
                    call_with_public_return_boundary(
                        graph,
                        metadata,
                        args.function,
                        args.source_count,
                        device_timer,
                    )
                )
                warmup_details.append(
                    {
                        "call_position": warmup_index + 1,
                        "seconds": warmup_seconds,
                        "device_seconds": warmup_kernel_seconds,
                        "device_timer": warmup_timer_provenance,
                        "validation": warmup_detail,
                    }
                )
                del warmup_result
            if args.measurement == "memory":
                (
                    result,
                    detail,
                    elapsed,
                    kernel_seconds,
                    timer_provenance,
                ) = call_with_public_return_boundary(
                    graph,
                    metadata,
                    args.function,
                    args.source_count,
                    device_timer,
                )
                if elapsed > args.timeout:
                    raise TimeoutError(
                        f"strict public API call took {elapsed:.3f}s"
                    )
                record = {
                    "status": "ok",
                    "measurement": "memory",
                    "dataset": metadata["name"],
                    "function": args.function,
                    "baseline": "nx-cugraph",
                    "graph_prepare_seconds": preparation_seconds,
                    "e2e_seconds": elapsed,
                    "kernel_seconds": kernel_seconds,
                    "device_timer": timer_provenance,
                    "validation": "pass",
                    "validation_details": detail,
                    "warmup_calls": args.warmup_calls,
                    "warmup_details": warmup_details,
                    "measured_call_position": args.warmup_calls + 1,
                    "prepared_native_graph": True,
                    "public_return_boundary": True,
                    "validation_outside_timer": True,
                    **backend_contract,
                }
                del result
            else:
                samples = []
                kernel_samples = []
                details = []
                timer_provenance_rows = []
                for _ in range(args.repeat):
                    gc.collect()
                    (
                        result,
                        detail,
                        elapsed,
                        kernel_seconds,
                        timer_provenance,
                    ) = call_with_public_return_boundary(
                        graph,
                        metadata,
                        args.function,
                        args.source_count,
                        device_timer,
                    )
                    if elapsed > args.timeout:
                        raise TimeoutError(
                            f"strict public API call took {elapsed:.3f}s"
                        )
                    samples.append(elapsed)
                    kernel_samples.append(kernel_seconds)
                    details.append(detail)
                    timer_provenance_rows.append(timer_provenance)
                    del result
                record = {
                    "status": "ok",
                    "measurement": "timing",
                    "dataset": metadata["name"],
                    "function": args.function,
                    "baseline": "nx-cugraph",
                    "input_path": "normalized_host_csr_to_device_coo",
                    "num_nodes": int(metadata["num_nodes"]),
                    "num_entries": int(metadata["num_entries"]),
                    "directed": bool(metadata["directed"]),
                    "graph_prepare_seconds": preparation_seconds,
                    "e2e": sample_stats(samples),
                    "e2e_best_seconds": min(samples),
                    "e2e_mean_seconds": statistics.mean(samples),
                    "e2e_stdev_seconds": (
                        statistics.stdev(samples) if len(samples) > 1 else 0.0
                    ),
                    "kernel": sample_stats(kernel_samples),
                    "kernel_best_seconds": min(kernel_samples),
                    "kernel_mean_seconds": statistics.mean(kernel_samples),
                    "kernel_stdev_seconds": (
                        statistics.stdev(kernel_samples)
                        if len(kernel_samples) > 1
                        else 0.0
                    ),
                    "device_timer_records": timer_provenance_rows,
                    "timing_samples": len(samples),
                    "validation": "pass",
                    "validation_details": details,
                    "nvrtc_source": nvrtc,
                    "warmup_calls": args.warmup_calls,
                    "warmup_details": warmup_details,
                    "measured_call_position": args.warmup_calls + 1,
                    "timer_boundary": (
                        "strict NetworkX public invocation through complete "
                        "result return; benchmark validation and any "
                        "post-return synchronization excluded"
                    ),
                    "kernel_timer_boundary": (
                        "CUDA events at each invoked pylibcugraph algorithm "
                        "entry on an explicit RAFT-bound stream; graph "
                        "conversion and Python result materialization excluded"
                    ),
                    "validation_outside_timer": True,
                    "prepared_native_graph": True,
                    "public_return_boundary": True,
                    **backend_contract,
                }
    except Exception as exc:
        lowered = str(exc).lower()
        kind = "oom" if "out of memory" in lowered or "memoryerror" in lowered else "execution_error"
        if isinstance(exc, TimeoutError):
            kind = "timeout"
        record = {
            "status": "failed",
            "failure_kind": kind,
            "measurement": args.measurement,
            "dataset": args.manifest.stem.replace(".weighted", ""),
            "function": args.function,
            "baseline": "nx-cugraph",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        record.setdefault(
            "host_rss_peak_mb",
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        )
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if record.get("status") == "ok" else 2


def collect_gpu_exclusive_snapshot(physical_gpu: int):
    """Capture one auditable physical-GPU occupancy snapshot."""

    gpu_command = [
        "nvidia-smi",
        f"--id={int(physical_gpu)}",
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    gpu_query = subprocess.run(
        gpu_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if gpu_query.returncode != 0:
        raise RuntimeError(
            "nvidia-smi GPU snapshot failed: "
            + (gpu_query.stderr.strip() or gpu_query.stdout.strip())
        )
    gpu_lines = [line.strip() for line in gpu_query.stdout.splitlines() if line.strip()]
    if len(gpu_lines) != 1:
        raise RuntimeError(
            f"expected one GPU snapshot row for physical GPU {physical_gpu}, "
            f"received {len(gpu_lines)}"
        )
    gpu_fields = [part.strip() for part in gpu_lines[0].split(",")]
    if len(gpu_fields) != 4:
        raise RuntimeError(f"unexpected nvidia-smi GPU snapshot: {gpu_lines[0]!r}")
    gpu_index, gpu_uuid, memory_used_mb, utilization_percent = gpu_fields

    process_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    process_query = subprocess.run(
        process_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process_query.returncode != 0:
        raise RuntimeError(
            "nvidia-smi compute-process snapshot failed: "
            + (process_query.stderr.strip() or process_query.stdout.strip())
        )
    compute_processes = []
    for line in process_query.stdout.splitlines():
        fields = [part.strip() for part in line.split(",", 3)]
        if len(fields) != 4 or fields[0] != gpu_uuid:
            continue
        compute_processes.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_gpu_memory_mb": float(fields[3]),
            }
        )

    memory_used_mb_value = float(memory_used_mb)
    utilization_percent_value = float(utilization_percent)
    max_memory_mb = 1024.0
    max_utilization_percent = 5.0
    exclusive = (
        not compute_processes
        and memory_used_mb_value <= max_memory_mb
        and utilization_percent_value <= max_utilization_percent
    )
    return {
        "captured_at_epoch": time.time(),
        "physical_gpu": int(physical_gpu),
        "reported_gpu_index": int(gpu_index),
        "gpu_uuid": gpu_uuid,
        "memory_used_mb": memory_used_mb_value,
        "utilization_percent": utilization_percent_value,
        "compute_processes": compute_processes,
        "exclusive": exclusive,
        "thresholds": {
            "max_memory_used_mb": max_memory_mb,
            "max_utilization_percent": max_utilization_percent,
            "max_foreign_compute_processes": 0,
        },
        "commands": {
            "gpu": gpu_command,
            "compute_processes": process_command,
        },
    }


def wait_for_gpu_exclusive_snapshot(
    physical_gpu: int,
    *,
    attempts: int = 10,
    retry_seconds: float = 0.5,
):
    """Wait out stale CUDA accounting, but never ignore a live process."""

    snapshot = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        snapshot = collect_gpu_exclusive_snapshot(physical_gpu)
        snapshot["attempt"] = attempt
        if snapshot["exclusive"]:
            return snapshot
        if snapshot["compute_processes"] or attempt >= attempts:
            return snapshot
        time.sleep(max(0.0, float(retry_seconds)))
    return snapshot


def safe_gpu_exclusive_snapshot(physical_gpu: int):
    try:
        return wait_for_gpu_exclusive_snapshot(physical_gpu)
    except Exception as exc:
        return {
            "captured_at_epoch": time.time(),
            "physical_gpu": int(physical_gpu),
            "exclusive": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_worker(
    args,
    manifest: Path,
    function: str,
    measurement: str,
    *,
    worker_repeat: int | None = None,
):
    preflight_snapshot = safe_gpu_exclusive_snapshot(args.gpu)
    if not preflight_snapshot["exclusive"]:
        return {
            "status": "failed",
            "failure_kind": "gpu_busy_before_worker",
            "measurement": measurement,
            "function": function,
            "baseline": "nx-cugraph",
            "gpu_exclusive_snapshot_before_worker": preflight_snapshot,
            "error": (
                "physical GPU failed the exclusive preflight; worker was not "
                "started"
            ),
        }
    repeat = args.repeat if worker_repeat is None else int(worker_repeat)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--manifest",
        str(manifest),
        "--function",
        function,
        "--measurement",
        measurement,
        "--repeat",
        str(repeat),
        "--source-count",
        str(args.source_count),
        "--timeout",
        str(args.timeout),
        "--warmup-calls",
        str(args.warmup_calls),
        "--physical-gpu",
        str(args.gpu),
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "CUPY_CACHE_DIR": str((args.out_dir / "cupy_cache").resolve()),
            "MALLOC_ARENA_MAX": "2",
        }
    )
    measured_calls = repeat if measurement == "timing" else 1
    process_timeout = (
        args.load_timeout
        + args.timeout * (args.warmup_calls + measured_calls)
        + 30.0
    )
    monitor = None
    monitor_result = None
    try:
        if measurement == "memory":
            process = subprocess.Popen(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # The monitor lives in the driver process, which is waiting in
            # communicate() and therefore cannot be starved by a native call
            # retaining the worker's Python GIL.
            monitor = ChildProcessMemoryMonitor(
                process.pid,
                physical_gpu=args.gpu,
                interval_seconds=0.002,
            ).start()
            try:
                stdout, stderr = process.communicate(timeout=process_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise
            finally:
                monitor_result = monitor.stop()
            completed = subprocess.CompletedProcess(
                command,
                process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        else:
            completed = subprocess.run(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=process_timeout,
                check=False,
            )
        postflight_snapshot = safe_gpu_exclusive_snapshot(args.gpu)
        try:
            record = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception:
            lowered = completed.stderr.lower()
            kind = "oom" if "out of memory" in lowered else "execution_error"
            record = {
                "status": "failed",
                "failure_kind": kind,
                "measurement": measurement,
                "function": function,
                "baseline": "nx-cugraph",
                "error": completed.stderr[-4000:],
                "returncode": completed.returncode,
            }
        record["gpu_exclusive_snapshot_before_worker"] = preflight_snapshot
        record["gpu_exclusive_snapshot_after_worker"] = postflight_snapshot
        record["gpu_exclusive_preflight_verified"] = bool(
            preflight_snapshot["exclusive"]
        )
        record["gpu_exclusive_postflight_verified"] = bool(
            postflight_snapshot["exclusive"]
        )
        if not postflight_snapshot["exclusive"]:
            record["status"] = "failed"
            record["failure_kind"] = "gpu_busy_after_worker"
            record["error"] = (
                "physical GPU failed the exclusive postflight; sample is "
                "excluded"
            )
        if monitor_result is not None:
            record["gpu_process_peak_mb"] = monitor_result["gpu_proc_peak_mb"]
            record["host_rss_peak_mb"] = monitor_result["rss_mb"]
            record["memory_monitor_samples"] = monitor_result[
                "monitor_gpu_proc_samples"
            ]
            record["memory_monitor_rss_samples"] = monitor_result[
                "monitor_rss_samples"
            ]
            record["memory_monitor_interval_ms"] = monitor_result[
                "monitor_poll_ms"
            ]
            record["memory_monitor_origin"] = monitor_result[
                "memory_monitor_origin"
            ]
            record["memory_measurement_window"] = (
                "isolated_worker_process_full_lifetime"
            )
        record["stderr_tail"] = completed.stderr[-1200:]
        return record
    except subprocess.TimeoutExpired as exc:
        postflight_snapshot = safe_gpu_exclusive_snapshot(args.gpu)
        record = {
            "status": "failed",
            "failure_kind": "timeout",
            "measurement": measurement,
            "function": function,
            "baseline": "nx-cugraph",
                "error": (
                    "worker exceeded graph-preparation plus one-call timeout "
                    f"budget: {exc}"
                ),
            "gpu_exclusive_snapshot_before_worker": preflight_snapshot,
            "gpu_exclusive_snapshot_after_worker": postflight_snapshot,
            "gpu_exclusive_preflight_verified": bool(
                preflight_snapshot["exclusive"]
            ),
            "gpu_exclusive_postflight_verified": bool(
                postflight_snapshot.get("exclusive")
            ),
        }
        if monitor_result is not None:
            record["gpu_process_peak_mb"] = monitor_result["gpu_proc_peak_mb"]
            record["host_rss_peak_mb"] = monitor_result["rss_mb"]
            record["memory_monitor_samples"] = monitor_result[
                "monitor_gpu_proc_samples"
            ]
            record["memory_monitor_rss_samples"] = monitor_result[
                "monitor_rss_samples"
            ]
            record["memory_monitor_interval_ms"] = monitor_result[
                "monitor_poll_ms"
            ]
            record["memory_monitor_origin"] = monitor_result[
                "memory_monitor_origin"
            ]
            record["memory_measurement_window"] = (
                "isolated_worker_process_full_lifetime"
            )
        return record


def aggregate_timing_rows(rows):
    """Aggregate five fresh-process calls without hiding partial failures."""

    successful = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        kinds = sorted({row.get("failure_kind", "execution_error") for row in failed})
        return {
            "status": "failed",
            "failure_kind": "+".join(kinds),
            "baseline": "nx-cugraph",
            "successful_timing_processes": len(successful),
            "failed_timing_processes": len(failed),
            "error": " | ".join(str(row.get("error", "unknown failure")) for row in failed),
            "timing_process_records": rows,
        }
    samples = []
    kernel_samples = []
    prepare_samples = []
    for row in successful:
        values = (row.get("e2e") or {}).get("samples") or []
        kernel_values = (row.get("kernel") or {}).get("samples") or []
        if len(values) != 1:
            raise RuntimeError(
                "fresh-process nx-cugraph timing worker must return one sample"
            )
        if len(kernel_values) != 1:
            raise RuntimeError(
                "fresh-process nx-cugraph timing worker must return one "
                "device-interval sample"
            )
        if not row.get("validation_outside_timer"):
            raise RuntimeError(
                "fresh-process nx-cugraph timing worker included validation "
                "inside the paper timer"
            )
        provenance_rows = row.get("device_timer_records") or []
        if len(provenance_rows) != 1:
            raise RuntimeError(
                "fresh-process nx-cugraph timing worker has ambiguous "
                "device-timer provenance"
            )
        provenance = provenance_rows[0]
        if (
            provenance.get("timer")
            != "cuda_events_at_pylibcugraph_backend_boundary"
            or not provenance.get("backend_functions")
            or int(provenance.get("backend_interval_count", 0)) <= 0
            or int(provenance.get("resource_handle_factory_calls", 0)) <= 0
        ):
            raise RuntimeError(
                "fresh-process nx-cugraph timing worker failed the "
                "device-timer provenance gate"
            )
        samples.append(float(values[0]))
        kernel_samples.append(float(kernel_values[0]))
        prepare_samples.append(float(row["graph_prepare_seconds"]))
        if kernel_samples[-1] <= 0.0 or kernel_samples[-1] > samples[-1]:
            raise RuntimeError(
                "fresh-process nx-cugraph device interval is not in "
                "(0, public E2E]"
            )
    output = dict(successful[-1])
    output.update(
        {
            "status": "ok",
            "timing_process_samples": len(successful),
            "e2e": sample_stats(samples),
            "e2e_best_seconds": min(samples),
            "e2e_mean_seconds": statistics.mean(samples),
            "e2e_stdev_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "kernel": sample_stats(kernel_samples),
            "kernel_best_seconds": min(kernel_samples),
            "kernel_mean_seconds": statistics.mean(kernel_samples),
            "kernel_stdev_seconds": (
                statistics.stdev(kernel_samples)
                if len(kernel_samples) > 1
                else 0.0
            ),
            "graph_prepare": sample_stats(prepare_samples),
            "graph_prepare_seconds": min(prepare_samples),
            "graph_prepare_best_seconds": min(prepare_samples),
            "graph_prepare_mean_seconds": statistics.mean(prepare_samples),
            "graph_prepare_stdev_seconds": (
                statistics.stdev(prepare_samples) if len(prepare_samples) > 1 else 0.0
            ),
            "timing_samples": len(samples),
            "timing_process_records": rows,
        }
    )
    return output


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def command_output(command):
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def collect_run_manifest(args):
    source_root = Path(__file__).resolve().parents[1]
    timer_path = Path(__file__).with_name("nxcugraph_device_timer.py")
    manifests = []
    for path in args.manifests:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        manifests.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "name": metadata.get("name"),
                "num_nodes": metadata.get("num_nodes"),
                "num_entries": metadata.get("num_entries"),
                "directed": metadata.get("directed"),
                "csr_artifacts": metadata.get("csr_artifacts", {}),
                "weighted_sibling": str(
                    path.with_name(f"{path.stem}.weighted.json").resolve()
                )
                if path.with_name(f"{path.stem}.weighted.json").is_file()
                else "",
            }
        )
    gpu_query = command_output(
        [
            "nvidia-smi",
            f"--id={args.gpu}",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    git_commit = command_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"]
    )
    git_status = command_output(
        [
            "git",
            "-C",
            str(source_root),
            "status",
            "--short",
            "--",
            "benchmarking/nxcugraph_device_timer.py",
            "benchmarking/run_nxcugraph_large_matrix.py",
            "benchmarking/library_baselines.py",
            "benchmarking/measurement_schema.py",
        ]
    )
    return {
        "schema_version": 1,
        "status": "running",
        "created_at_epoch": time.time(),
        "argv": sys.argv,
        "protocol": {
            "construction": (
                "normalized host CSR through H2D COO materialization and "
                "fully constructed native nx-cugraph graph"
            ),
            "warmup_calls": int(args.warmup_calls),
            "timing_processes": int(args.repeat),
            "fresh_process_per_timing_sample": True,
            "e2e": (
                "strict NetworkX public call entry through complete result "
                "return; validation excluded"
            ),
            "processing": (
                "sum of CUDA-event intervals at actual pylibcugraph algorithm "
                "entries on one explicit RAFT-bound stream"
            ),
            "display_estimator": "minimum_of_five",
        },
        "gpu": {
            "physical_index": int(args.gpu),
            "nvidia_smi": gpu_query,
        },
        "software": {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "networkx": package_version("networkx"),
            "nx-cugraph": package_version("nx-cugraph"),
            "pylibcugraph": package_version("pylibcugraph"),
            "pylibraft": package_version("pylibraft"),
            "cupy": package_version("cupy"),
        },
        "source": {
            "git_commit": git_commit,
            "relevant_git_status": git_status,
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "timer": str(timer_path.resolve()),
            "timer_sha256": sha256_file(timer_path),
        },
        "inputs": manifests,
        "functions": list(args.functions),
    }


def flatten(record):
    output = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            output[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            output[key] = value
    return output


def write_summary(out_dir: Path, records):
    import csv

    records = sorted(records, key=lambda item: (item["dataset"], item["function"]))
    (out_dir / "nxcugraph_large_matrix.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = [flatten(record) for record in records]
    fields = sorted({key for row in rows for key in row})
    with (out_dir / "nxcugraph_large_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def driver(args) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = args.out_dir / "raw"
    raw.mkdir(exist_ok=True)
    run_manifest_path = args.out_dir / "run_manifest.json"
    run_manifest = collect_run_manifest(args)
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    records = []
    total = len(args.manifests) * len(args.functions)
    progress = 0
    for plain_manifest in args.manifests:
        plain_metadata = json.loads(plain_manifest.read_text(encoding="utf-8"))
        dataset = plain_metadata["name"]
        for function in args.functions:
            progress += 1
            final_path = args.out_dir / f"{dataset}_{function}.json"
            if args.resume and final_path.is_file():
                existing = json.loads(final_path.read_text(encoding="utf-8"))
                if existing.get("dataset") == dataset and existing.get("function") == function:
                    records.append(existing)
                    print(f"[nx-cugraph {progress}/{total}] reused {dataset}/{function}")
                    continue
            manifest = manifest_for_function(plain_manifest, function)
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            allowed, kind, reason = applicability(metadata, function)
            if not allowed:
                record = {
                    "status": "skipped",
                    "failure_kind": kind,
                    "dataset": dataset,
                    "function": function,
                    "baseline": "nx-cugraph",
                    "num_nodes": int(plain_metadata["num_nodes"]),
                    "num_entries": int(plain_metadata["num_entries"]),
                    "directed": bool(plain_metadata["directed"]),
                    "reason": reason,
                }
            else:
                timing_rows = []
                for timing_index in range(args.repeat):
                    timing = run_worker(
                        args,
                        manifest.resolve(),
                        function,
                        "timing",
                        worker_repeat=1,
                    )
                    timing.setdefault("dataset", dataset)
                    timing.setdefault("num_nodes", int(plain_metadata["num_nodes"]))
                    timing.setdefault("num_entries", int(plain_metadata["num_entries"]))
                    timing.setdefault("directed", bool(plain_metadata["directed"]))
                    timing["timing_process_index"] = timing_index + 1
                    timing_rows.append(timing)
                    (raw / f"{dataset}_{function}_timing_{timing_index + 1}.json").write_text(
                        json.dumps(timing, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    if timing.get("status") != "ok":
                        break
                record = aggregate_timing_rows(timing_rows)
                record.setdefault("dataset", dataset)
                record.setdefault("function", function)
                record.setdefault("num_nodes", int(plain_metadata["num_nodes"]))
                record.setdefault("num_entries", int(plain_metadata["num_entries"]))
                record.setdefault("directed", bool(plain_metadata["directed"]))
                (raw / f"{dataset}_{function}_timing.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                if record.get("status") == "ok":
                    memory_rows = []
                    for index in range(args.memory_repeat):
                        memory = run_worker(args, manifest.resolve(), function, "memory")
                        memory.setdefault("dataset", dataset)
                        memory_rows.append(memory)
                        (raw / f"{dataset}_{function}_memory_{index + 1}.json").write_text(
                            json.dumps(memory, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8",
                        )
                    successful = [row for row in memory_rows if row.get("status") == "ok"]
                    if successful:
                        record["memory_samples"] = len(successful)
                        record["gpu_process_peak_mb_mean"] = statistics.mean(
                            float(row.get("gpu_process_peak_mb", 0.0)) for row in successful
                        )
                        record["gpu_process_peak_mb_stdev"] = (
                            statistics.stdev(
                                float(row.get("gpu_process_peak_mb", 0.0)) for row in successful
                            )
                            if len(successful) > 1
                            else 0.0
                        )
                        record["host_rss_peak_mb_mean"] = statistics.mean(
                            float(row.get("host_rss_peak_mb", 0.0)) for row in successful
                        )
                    if len(successful) != args.memory_repeat:
                        record["memory_status"] = "incomplete"
                        record["memory_failure_details"] = [
                            row for row in memory_rows if row.get("status") != "ok"
                        ]
                    else:
                        record["memory_status"] = "ok"
            final_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            records.append(record)
            print(
                f"[nx-cugraph {progress}/{total}] {dataset}/{function}: "
                f"{record.get('status')} {record.get('failure_kind', '')}",
                flush=True,
            )
    write_summary(args.out_dir, records)
    run_manifest["status"] = "complete"
    run_manifest["completed_at_epoch"] = time.time()
    run_manifest["record_count"] = len(records)
    run_manifest["status_counts"] = {
        status: sum(record.get("status") == status for record in records)
        for status in sorted({record.get("status", "") for record in records})
    }
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifests", type=Path, nargs="+")
    parser.add_argument("--function", choices=ALL_FUNCTIONS)
    parser.add_argument("--functions", nargs="+", choices=ALL_FUNCTIONS, default=list(ALL_FUNCTIONS))
    parser.add_argument("--measurement", choices=("timing", "memory"), default="timing")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-repeat", type=int, default=3)
    parser.add_argument("--source-count", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--warmup-calls", type=int, default=3)
    parser.add_argument("--load-timeout", type=float, default=900.0)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.worker:
        if args.manifest is None or args.function is None:
            parser.error("--worker requires --manifest and --function")
    elif not args.manifests or args.out_dir is None:
        parser.error("driver requires --manifests and --out-dir")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(worker(parsed) if parsed.worker else driver(parsed))
