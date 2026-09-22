#!/usr/bin/env python3
"""Run EGGPU first-use, steady-state, and memory scaling experiments."""

import argparse
import atexit
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import select
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path


os.environ.setdefault("EASYGRAPH_ENABLE_GPU", "TRUE")
os.environ.setdefault("EASYGRAPH_GPU_RESULT_CACHE", "FALSE")
os.environ.setdefault("EASYGRAPH_GPU_STRICT_ERRORS", "TRUE")
os.environ.setdefault("EASYGRAPH_GPU_ADAPTIVE_HOST", "FALSE")
os.environ.setdefault("EGGPU_ALLOW_CUDA_SYNC", "TRUE")

import numpy as np

from child_process_memory_monitor import ChildProcessMemoryMonitor
from easygraph_runtime_provenance import (
    collect_loaded_runtime_provenance,
    collect_relevant_environment,
    collect_runtime_repository_provenance,
    install_runtime_import_root,
)
from library_baselines import PeakMemoryMonitor
from stable_timing_protocol import (
    DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    DEFAULT_MEDIAN_OVER_MIN_LIMIT,
    EXPECTED_TIMING_SAMPLES,
    PAPER_ESTIMATOR,
    apply_controlled_thread_environment,
    collect_execution_placement,
    controlled_protocol_metadata,
    summarize_five_samples,
    validate_controlled_execution,
)


SCALING_FUNCTIONS = ("PageRank", "WCC", "BFS", "KCore")
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
FUNCTIONS = ALL_FUNCTIONS
WEIGHTED_FUNCTIONS = {"MST", "Dijkstra", "BellmanFord", "SSSP"}
PROJECTED_FUNCTIONS = {"MST", "LCC", "KCore"}
COMPONENT_FUNCTIONS = {"WCC", "SCC"}
FLOAT_VECTOR_FUNCTIONS = {
    "PageRank",
    "LCC",
    "BC",
    "Closeness",
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
}
# Repeated PageRank vectors may differ slightly because sparse reductions are
# not required to be bit deterministic.  These tolerances match the aligned
# cross-library vector comparison used by the main correctness audit.
PAGERANK_REPEAT_RTOL = 1.0e-4
PAGERANK_REPEAT_ATOL = 1.0e-8
PAGERANK_DIGEST_SAMPLES = 4096


eg = None
gpu_eggpu_backend = None
_LOADED_RUNTIME_PROVENANCE = None
_REPOSITORY_RUNTIME_PROVENANCE = None


def _same_resolved_path(lhs, rhs):
    try:
        return Path(lhs or ".").resolve() == Path(rhs or ".").resolve()
    except OSError:
        return False


def runtime_subprocess_environment(easygraph_repo, environment=None):
    """Return an environment whose first Python import root is the frozen repo."""

    runtime_root = Path(easygraph_repo).expanduser().resolve(strict=True)
    env = dict(os.environ if environment is None else environment)
    retained = [
        entry
        for entry in env.get("PYTHONPATH", "").split(os.pathsep)
        if entry and not _same_resolved_path(entry, runtime_root)
    ]
    env["PYTHONPATH"] = os.pathsep.join([str(runtime_root), *retained])
    return env


def configure_runtime_import_root(easygraph_repo):
    """Pin both in-process and child-process imports before loading EasyGraph."""

    runtime_root = install_runtime_import_root(easygraph_repo)
    os.environ.update(runtime_subprocess_environment(runtime_root))
    return runtime_root


def _runtime_identity(provenance):
    snapshot = (provenance or {}).get("runtime_python_snapshot") or {}
    return {
        "native_sha256": (provenance or {}).get("native_sha256", ""),
        "runtime_python_digest": snapshot.get("digest", ""),
        "resolved_root": (provenance or {}).get("resolved_root", ""),
    }


def _require_runtime_identity(provenance, label):
    identity = _runtime_identity(provenance)
    if not identity["native_sha256"] or not identity["runtime_python_digest"]:
        raise RuntimeError(
            f"{label} is missing the native binary SHA or runtime Python digest"
        )
    return identity


def assert_runtime_matches(expected, observed_repository, observed_loaded):
    """Reject a worker that imported a different Python tree or native module."""

    expected_identity = _require_runtime_identity(expected, "requested runtime")
    observed_identity = _require_runtime_identity(
        observed_repository, "worker repository runtime"
    )
    if observed_identity != expected_identity:
        raise RuntimeError(
            "worker repository runtime differs from coordinator snapshot: "
            f"expected={expected_identity} observed={observed_identity}"
        )
    loaded_sha = (observed_loaded or {}).get("native_sha256", "")
    loaded_root = (observed_loaded or {}).get("resolved_root", "")
    if loaded_sha != expected_identity["native_sha256"]:
        raise RuntimeError(
            "loaded cpp_easygraph binary differs from coordinator snapshot: "
            f"expected={expected_identity['native_sha256']} observed={loaded_sha or 'missing'}"
        )
    if not _same_resolved_path(loaded_root, expected_identity["resolved_root"]):
        raise RuntimeError(
            "loaded EasyGraph runtime root differs from coordinator snapshot: "
            f"expected={expected_identity['resolved_root']} observed={loaded_root or 'missing'}"
        )


def load_easygraph_runtime(easygraph_repo):
    """Import the frozen EasyGraph runtime after pinning its absolute root."""

    global eg
    global gpu_eggpu_backend
    global _LOADED_RUNTIME_PROVENANCE
    global _REPOSITORY_RUNTIME_PROVENANCE

    configure_runtime_import_root(easygraph_repo)
    if eg is None:
        eg = importlib.import_module("easygraph")
        importlib.import_module("cpp_easygraph")
        gpu_eggpu_backend = importlib.import_module(
            "easygraph.utils.gpu_eggpu_backend"
        )
    _REPOSITORY_RUNTIME_PROVENANCE = collect_runtime_repository_provenance(
        easygraph_repo
    )
    _LOADED_RUNTIME_PROVENANCE = collect_loaded_runtime_provenance(
        easygraph_repo
    )
    assert_runtime_matches(
        _REPOSITORY_RUNTIME_PROVENANCE,
        _REPOSITORY_RUNTIME_PROVENANCE,
        _LOADED_RUNTIME_PROVENANCE,
    )
    return _REPOSITORY_RUNTIME_PROVENANCE, _LOADED_RUNTIME_PROVENANCE


def runtime_record_envelope(
    easygraph_repo,
    requested_runtime,
    *,
    loaded_runtime=None,
    argv=None,
    environment=None,
):
    """Attach the frozen runtime identity needed for audit and safe resume."""

    identity = _require_runtime_identity(requested_runtime, "requested runtime")
    requested_root = Path(easygraph_repo).expanduser().resolve(strict=True)
    if not _same_resolved_path(requested_root, identity["resolved_root"]):
        raise RuntimeError(
            "--easygraph-repo differs from the recorded runtime snapshot: "
            f"argument={requested_root} snapshot={identity['resolved_root']}"
        )
    return {
        "argv": list(sys.argv if argv is None else argv),
        "easygraph_repo": identity["resolved_root"],
        "runtime_identity": identity,
        "requested_runtime_provenance": requested_runtime,
        "runtime_provenance": loaded_runtime,
        "runtime_environment": collect_relevant_environment(
            os.environ if environment is None else environment
        ),
    }


def record_runtime_matches(record, requested_runtime):
    """Return whether a raw record belongs to exactly the requested runtime."""

    try:
        expected = _require_runtime_identity(
            requested_runtime, "requested runtime"
        )
    except RuntimeError:
        return False
    recorded_request = _runtime_identity(
        record.get("requested_runtime_provenance")
    )
    recorded = record.get("runtime_identity") or recorded_request
    if recorded != expected or recorded_request != expected:
        return False
    if record.get("status") == "ok":
        loaded = record.get("runtime_provenance") or {}
        if loaded.get("native_sha256", "") != expected["native_sha256"]:
            return False
        if not _same_resolved_path(
            loaded.get("resolved_root", ""), expected["resolved_root"]
        ):
            return False
    return True


def watchdog_main(parent_pid: int) -> int:
    """Kill a worker when an armed native call exceeds its deadline.

    A separate process is required because a long-running CUDA/C++ extension
    may retain the Python GIL.  The watchdog is started before the measured
    calls and controlled over stdin, so process-startup time is never included
    in E2E timing.
    """

    deadline = None
    marker = None
    limit_seconds = None
    while True:
        try:
            os.kill(parent_pid, 0)
        except OSError:
            return 0
        wait = 0.10
        if deadline is not None:
            wait = max(0.0, min(wait, deadline - time.monotonic()))
        ready, _, _ = select.select([sys.stdin], [], [], wait)
        if ready:
            line = sys.stdin.readline()
            if not line:
                return 0
            command = json.loads(line)
            operation = command.get("op")
            if operation == "stop":
                return 0
            if operation == "disarm":
                deadline = None
                marker = None
                limit_seconds = None
                continue
            if operation == "arm":
                deadline = float(command["deadline"])
                marker = Path(command["marker"])
                limit_seconds = float(command["limit_seconds"])
                continue
            raise RuntimeError(f"unknown watchdog operation: {operation!r}")
        if deadline is not None and time.monotonic() >= deadline:
            payload = {
                "status": "timeout",
                "failure_kind": "timeout",
                "parent_pid": parent_pid,
                "deadline_monotonic": deadline,
                "limit_seconds": limit_seconds,
                "recorded_at_unix": time.time(),
            }
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.kill(parent_pid, signal.SIGKILL)
            return 124


class CallWatchdog:
    def __init__(
        self,
        marker_path: Path,
        limit_seconds: float,
        easygraph_repo: Path,
    ):
        self.marker_path = marker_path
        self.limit_seconds = float(limit_seconds)
        environment = runtime_subprocess_environment(easygraph_repo)
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--watchdog",
                "--watchdog-pid",
                str(os.getpid()),
                "--easygraph-repo",
                str(Path(easygraph_repo).expanduser().resolve()),
            ],
            stdin=subprocess.PIPE,
            text=True,
            env=environment,
        )
        atexit.register(self.close)

    def _send(self, payload):
        if self.process.poll() is not None or self.process.stdin is None:
            raise RuntimeError("per-call watchdog exited unexpectedly")
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def arm(self, limit_seconds=None):
        limit_seconds = (
            self.limit_seconds if limit_seconds is None else float(limit_seconds)
        )
        self.marker_path.unlink(missing_ok=True)
        self._send(
            {
                "op": "arm",
                "deadline": time.monotonic() + limit_seconds,
                "limit_seconds": limit_seconds,
                "marker": str(self.marker_path),
            }
        )

    def disarm(self):
        self._send({"op": "disarm"})

    def close(self):
        if getattr(self, "process", None) is None:
            return
        if self.process.poll() is None:
            try:
                self._send({"op": "stop"})
                self.process.wait(timeout=2.0)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.process = None


def function_applicability(metadata, function):
    """Return whether a function preserves the declared bulk-graph semantics."""

    if (
        function in PROJECTED_FUNCTIONS
        and bool(metadata.get("directed"))
        and not metadata.get("undirected_projection_manifest")
    ):
        entries = int(metadata.get("num_entries", 0))
        projected_upper_bound = 2 * entries
        return False, (
            f"{function} requires the common undirected-projection semantics; "
            "the artifact stores only directed outgoing CSR and an explicit "
            f"projection can require up to {projected_upper_bound:,} entries, "
            "which exceeds the current signed-int32 CSR ABI"
        )
    if function in WEIGHTED_FUNCTIONS and not metadata.get("weights_path"):
        return False, (
            f"{function} requires explicit deterministic edge weights; run "
            "prepare_bulk_csr_weights.py and use the generated weighted manifest"
        )
    # WCC is valid on a directed CSR: its kernel unions both endpoints of
    # every stored edge, so edge orientation does not alter the partition.
    return True, ""


def manifest_for_function(manifest: Path, function: str) -> Path:
    if function not in WEIGHTED_FUNCTIONS:
        return manifest
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("weights_path"):
        return manifest
    weighted = manifest.with_name(f"{manifest.stem}.weighted.json")
    return weighted if weighted.is_file() else manifest


def stats(values):
    values = [float(value) for value in values]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "stdev": stdev,
        "best": min(values),
        "cv": stdev / mean if mean > 0 else 0.0,
        "ci95_half_width": 1.96 * stdev / math.sqrt(len(values)) if len(values) > 1 else 0.0,
        "samples": values,
    }


def result_digests_equivalent(function, lhs, rhs):
    """Compare repeated results without requiring floating-point bit identity."""

    if function == "MST":
        return (
            lhs.get("node_count") == rhs.get("node_count")
            and lhs.get("edge_count") == rhs.get("edge_count")
            and math.isclose(
                float(lhs.get("total_weight", float("nan"))),
                float(rhs.get("total_weight", float("nan"))),
                rel_tol=PAGERANK_REPEAT_RTOL,
                abs_tol=PAGERANK_REPEAT_ATOL,
            )
        )
    if function in COMPONENT_FUNCTIONS:
        return lhs.get("partition_sha256") == rhs.get("partition_sha256")
    if function not in FLOAT_VECTOR_FUNCTIONS:
        key = "sha256"
        return lhs.get(key) == rhs.get(key)
    if lhs.get("shape") != rhs.get("shape"):
        return False
    # Exact zero counts are not a stable semantic invariant for floating
    # reductions: a different legal atomic-add order can move a value by one
    # ulp across zero while preserving every public tolerance contract.
    for key in ("finite_count", "nan_count", "inf_count"):
        if lhs.get(key) != rhs.get(key):
            return False
    lhs_sample = np.asarray(lhs.get("validation_sample", []), dtype=np.float64)
    rhs_sample = np.asarray(rhs.get("validation_sample", []), dtype=np.float64)
    if lhs_sample.size or rhs_sample.size:
        if lhs_sample.shape != rhs_sample.shape or lhs_sample.size == 0:
            return False
        if not np.allclose(
            lhs_sample,
            rhs_sample,
            rtol=PAGERANK_REPEAT_RTOL,
            atol=PAGERANK_REPEAT_ATOL,
            equal_nan=True,
        ):
            return False
    for key in ("sum", "minimum", "maximum"):
        if not math.isclose(
            float(lhs[key]),
            float(rhs[key]),
            rel_tol=PAGERANK_REPEAT_RTOL,
            abs_tol=PAGERANK_REPEAT_ATOL,
        ):
            return False
    return True


def result_digest_difference(lhs, rhs):
    """Return bounded diagnostics for a repeated floating-result mismatch."""

    summary = {
        key: {"first": lhs.get(key), "steady": rhs.get(key)}
        for key in (
            "shape",
            "finite_count",
            "nan_count",
            "inf_count",
            "zero_count",
            "sum",
            "minimum",
            "maximum",
        )
        if lhs.get(key) != rhs.get(key)
    }
    first_sample = np.asarray(lhs.get("validation_sample", []), dtype=np.float64)
    steady_sample = np.asarray(rhs.get("validation_sample", []), dtype=np.float64)
    if first_sample.shape == steady_sample.shape and first_sample.size:
        finite = np.isfinite(first_sample) & np.isfinite(steady_sample)
        if finite.any():
            absolute = np.abs(first_sample[finite] - steady_sample[finite])
            scale = np.maximum(
                np.maximum(np.abs(first_sample[finite]), np.abs(steady_sample[finite])),
                1.0,
            )
            summary["validation_sample"] = {
                "max_abs_diff": float(absolute.max()),
                "max_rel_diff": float((absolute / scale).max()),
            }
    return summary


def aggregate_timing_records(sample_records):
    """Aggregate fresh-process timing records without losing paired samples."""

    if not sample_records:
        raise ValueError("no timing records to aggregate")
    first = sample_records[0]
    identity = (first["dataset"], first["function"], first["num_nodes"], first["num_entries"])
    runtime_identity = first.get("runtime_identity")
    if not runtime_identity:
        raise RuntimeError("timing sample is missing frozen runtime identity")
    for record in sample_records:
        if (
            record["dataset"],
            record["function"],
            record["num_nodes"],
            record["num_entries"],
        ) != identity:
            raise RuntimeError("timing sample identity mismatch")
        if record.get("runtime_identity") != runtime_identity:
            raise RuntimeError("timing samples use different frozen runtimes")
        if record.get("result_validation", {}).get("status") != "pass":
            raise RuntimeError("a timing sample did not pass result validation")
        for protocol_key in (
            "first_use_calls",
            "additional_warmup_calls",
            "preceding_public_calls",
            "measured_call_position",
            "measured_calls_per_process",
            "timer_boundary",
            "validation_outside_timer",
        ):
            if record.get(protocol_key) != first.get(protocol_key):
                raise RuntimeError(
                    f"timing samples disagree on protocol field {protocol_key}"
                )
        if not result_digests_equivalent(
            record["function"], record.get("result", {}), first.get("result", {})
        ):
            raise RuntimeError("fresh-process timing result digests differ")

    load_values = [record["load_seconds"] for record in sample_records]
    first_e2e_values = [record["first_use_e2e_seconds"] for record in sample_records]
    first_kernel_values = [record["first_use_kernel_seconds"] for record in sample_records]
    user_cold_values = [record["user_cold_total_seconds"] for record in sample_records]
    steady_e2e_values = [
        value
        for record in sample_records
        for value in record["steady_e2e"]["samples"]
    ]
    steady_kernel_values = [
        value
        for record in sample_records
        for value in record["steady_kernel"]["samples"]
    ]

    aggregated = dict(first)
    aggregated["timing_process_samples"] = len(sample_records)
    aggregated["load"] = stats(load_values)
    aggregated["load_seconds"] = aggregated["load"]["mean"]
    aggregated["first_use_e2e"] = stats(first_e2e_values)
    aggregated["first_use_e2e_seconds"] = aggregated["first_use_e2e"]["mean"]
    aggregated["first_use_kernel"] = stats(first_kernel_values)
    aggregated["first_use_kernel_seconds"] = aggregated["first_use_kernel"]["mean"]
    aggregated["user_cold_total"] = stats(user_cold_values)
    aggregated["user_cold_total_seconds"] = aggregated["user_cold_total"]["mean"]
    aggregated["steady_e2e"] = stats(steady_e2e_values)
    aggregated["steady_kernel"] = stats(steady_kernel_values)

    e2e_mean = aggregated["steady_e2e"]["mean"]
    kernel_mean = aggregated["steady_kernel"]["mean"]
    first_e2e_mean = aggregated["first_use_e2e"]["mean"]
    first_kernel_mean = aggregated["first_use_kernel"]["mean"]
    aggregated["efficiency"] = {
        "input_csr_entries_per_second_e2e": (
            float(first["num_entries"]) / e2e_mean if e2e_mean > 0 else None
        ),
        "input_csr_entries_per_second_kernel": (
            float(first["num_entries"]) / kernel_mean if kernel_mean > 0 else None
        ),
        "steady_non_kernel_seconds": max(0.0, e2e_mean - kernel_mean),
        "first_use_non_kernel_seconds": max(0.0, first_e2e_mean - first_kernel_mean),
        "steady_non_kernel_fraction": (
            max(0.0, e2e_mean - kernel_mean) / e2e_mean
            if e2e_mean > 0
            else None
        ),
    }
    return aggregated


def component_labels(result, num_nodes):
    if hasattr(result, "labels_numpy"):
        return np.asarray(result.labels_numpy(), dtype=np.int32)
    if hasattr(result, "partition_labels_numpy"):
        return np.asarray(result.partition_labels_numpy(), dtype=np.int32)
    if (
        isinstance(result, (list, tuple))
        and result
        and hasattr(result[0], "partition_labels_numpy")
    ):
        labels = np.asarray(
            result[0].partition_labels_numpy(), dtype=np.int32
        ).reshape(-1)
        if len(labels) != int(num_nodes):
            raise RuntimeError(
                "component partition label count does not match graph node count"
            )
        return labels
    labels = np.full(int(num_nodes), -1, dtype=np.int32)
    for component_id, component in enumerate(result):
        for node in component:
            node_id = int(node)
            if node_id < 0 or node_id >= num_nodes:
                raise RuntimeError("component result contains an out-of-range node")
            if labels[node_id] != -1:
                raise RuntimeError("component result contains a node more than once")
            labels[node_id] = component_id
    if num_nodes and np.any(labels < 0):
        raise RuntimeError("component result does not cover every graph node")
    return labels


def output_array(result, function, num_nodes=None):
    if function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        return np.asarray(result.to_numpy(), dtype=np.float64)
    if function in COMPONENT_FUNCTIONS:
        return component_labels(result, int(num_nodes))
    if hasattr(result, "to_numpy"):
        return np.asarray(result.to_numpy())
    if hasattr(result, "values") and callable(result.values):
        return np.asarray(list(result.values()))
    return np.asarray(result)


def component_partition_digest(labels):
    """Hash a partition independently of implementation-specific label ids."""

    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    n = len(labels)
    if n == 0:
        return hashlib.sha256(b"").hexdigest(), 0
    minimum = int(labels.min())
    maximum = int(labels.max())
    if minimum < 0 or maximum >= n:
        return None, None

    sentinel = np.iinfo(np.int32).max
    representatives = np.full(n, sentinel, dtype=np.int32)
    chunk = 4_000_000
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        node_ids = np.arange(start, end, dtype=np.int32)
        np.minimum.at(representatives, labels[start:end], node_ids)

    digest = hashlib.sha256()
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        canonical = np.ascontiguousarray(representatives[labels[start:end]])
        digest.update(canonical.view(np.uint8))
    component_count = int(np.count_nonzero(representatives != sentinel))
    return digest.hexdigest(), component_count


def mst_digest(result):
    cache = getattr(result, "cache", {})
    lazy_arrays = cache.get("_lazy_edge_arrays") if isinstance(cache, dict) else None
    if lazy_arrays is not None and len(lazy_arrays) >= 3:
        weights = np.asarray(lazy_arrays[2], dtype=np.float64).reshape(-1)
        return {
            "kind": "spanning_forest",
            "node_count": int(len(result)),
            "edge_count": int(len(weights)),
            "total_weight": float(weights.sum(dtype=np.float64)),
            "finite_weights": bool(np.isfinite(weights).all()),
            "shape": [int(len(weights)), 3],
            "dtype": "edge-list",
            "nbytes": int(weights.nbytes),
        }
    edge_view = result.edges
    edges = edge_view() if callable(edge_view) else edge_view
    edge_count = 0
    total_weight = 0.0
    finite = True
    for edge in edges:
        if len(edge) < 2:
            continue
        data = edge[2] if len(edge) >= 3 and isinstance(edge[2], dict) else {}
        weight = float(data.get("weight", 1.0))
        edge_count += 1
        total_weight += weight
        finite = finite and math.isfinite(weight)
    return {
        "kind": "spanning_forest",
        "node_count": int(len(result)),
        "edge_count": edge_count,
        "total_weight": total_weight,
        "finite_weights": finite,
        "shape": [edge_count, 3],
        "dtype": "edge-list",
        "nbytes": None,
    }


def result_digest(result, function, num_nodes=None):
    if function == "MST":
        return mst_digest(result)
    array = np.ascontiguousarray(output_array(result, function, num_nodes=num_nodes))
    raw = array.view(np.uint8)
    digest = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "finite_count": int(np.isfinite(array).sum()) if array.dtype.kind == "f" else None,
        "sum": float(np.nansum(array, dtype=np.float64)),
        "zero_count": int(np.count_nonzero(array == 0)),
    }
    if array.size:
        if array.dtype.kind == "f":
            finite = np.isfinite(array)
            digest["nan_count"] = int(np.isnan(array).sum())
            digest["inf_count"] = int(np.isinf(array).sum())
            if finite.any():
                digest["minimum"] = float(array[finite].min())
                digest["maximum"] = float(array[finite].max())
            else:
                digest["minimum"] = None
                digest["maximum"] = None
        else:
            digest["minimum"] = int(array.min())
            digest["maximum"] = int(array.max())
            # Exact component/cardinality counts are useful on the 100M-edge
            # anchor but would add a large sort/allocation on the billion-edge
            # anchor. Keep that diagnostic outside measured time and bounded.
            digest["unique_count"] = (
                int(np.unique(array).size) if array.size <= 5_000_000 else None
            )
    if function in COMPONENT_FUNCTIONS:
        partition_sha256, component_count = component_partition_digest(array)
        digest["partition_sha256"] = partition_sha256
        digest["component_count"] = component_count
    elif function in FLOAT_VECTOR_FUNCTIONS and array.size:
        sample_count = min(PAGERANK_DIGEST_SAMPLES, int(array.size))
        sample_indices = np.linspace(
            0, int(array.size) - 1, num=sample_count, dtype=np.int64
        )
        digest["validation_sample"] = np.asarray(
            array.reshape(-1)[sample_indices], dtype=np.float64
        ).tolist()
    return digest


def validate_result(
    graph,
    function,
    result,
    digest,
    source_count,
    bc_source_count,
    closeness_source_count,
    structural_node_count=0,
):
    n = int(len(graph))
    shape = tuple(digest["shape"])
    failures = []
    if function == "MST":
        if digest.get("node_count") != n:
            failures.append(f"expected {n} returned nodes, got {digest.get('node_count')}")
        if digest.get("edge_count", n) > max(0, n - 1):
            failures.append("spanning-forest result contains too many edges")
        if not digest.get("finite_weights"):
            failures.append("spanning-forest result contains non-finite weights")
    elif function == "PageRank":
        if shape != (n,):
            failures.append(f"expected shape {(n,)}, got {shape}")
        if digest["finite_count"] != n:
            failures.append("PageRank contains non-finite scores")
        if digest.get("minimum") is not None and digest["minimum"] < -1.0e-12:
            failures.append("PageRank contains negative scores")
        if not math.isclose(digest["sum"], 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6):
            failures.append(f"PageRank score sum is {digest['sum']}, expected 1")
    elif function in COMPONENT_FUNCTIONS:
        if shape != (n,):
            failures.append(f"expected shape {(n,)}, got {shape}")
        if digest.get("minimum", 0) < 0 or digest.get("maximum", 0) >= n:
            failures.append("component labels are outside the node-id domain")
        if not digest.get("partition_sha256") or digest.get("component_count") is None:
            failures.append("component partition fingerprint could not be constructed")
    elif function in {"BFS", "BellmanFord", "SSSP"}:
        expected_sources = len(select_sources(graph, source_count))
        if shape != (expected_sources, n):
            failures.append(f"expected shape {(expected_sources, n)}, got {shape}")
        if digest.get("minimum") is not None and digest["minimum"] < 0:
            failures.append(f"{function} contains negative finite distances")
    elif function == "Dijkstra":
        if shape != (1, n):
            failures.append(f"expected shape {(1, n)}, got {shape}")
    elif function == "KCore":
        if shape != (n,):
            failures.append(f"expected shape {(n,)}, got {shape}")
        if digest.get("minimum", 0) < 0:
            failures.append("KCore contains a negative core number")
        max_degree = graph.metadata.get("max_degree")
        if max_degree is not None and digest.get("maximum", 0) > int(max_degree):
            failures.append("KCore exceeds the manifest maximum degree")
    elif function == "Closeness":
        expected = len(select_sources(graph, closeness_source_count))
        if shape not in {(expected,), (expected, n)}:
            failures.append(f"expected {expected} sampled closeness values, got shape {shape}")
        if digest.get("minimum") is not None and digest["minimum"] < -1.0e-12:
            failures.append("Closeness contains negative values")
    elif function == "BC":
        if shape != (n,):
            failures.append(f"expected shape {(n,)}, got {shape}")
        if digest.get("minimum") is not None and digest["minimum"] < -1.0e-9:
            failures.append("BC contains negative values")
    elif function == "LCC":
        if shape != (n,):
            failures.append(f"expected shape {(n,)}, got {shape}")
        if digest.get("minimum") is not None and digest["minimum"] < -1.0e-12:
            failures.append("LCC contains negative values")
        if digest.get("maximum") is not None and digest["maximum"] > 1.0 + 1.0e-9:
            failures.append("LCC exceeds one")
    elif function in {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}:
        expected = (
            len(select_sources(graph, structural_node_count))
            if structural_node_count > 0
            else n
        )
        if shape != (expected,):
            failures.append(f"expected shape {(expected,)}, got {shape}")
    return {"status": "pass" if not failures else "fail", "failures": failures}


def select_sources(graph, count):
    recorded = graph.metadata.get("benchmark_sources_zero_based")
    if isinstance(recorded, list) and len(recorded) >= count:
        return [int(source) for source in recorded[:count]]
    n = len(graph)
    if n <= 0:
        return []
    return sorted({min(n - 1, (idx * n) // max(1, count)) for idx in range(count)})


def function_call(
    graph,
    function,
    source_count,
    bc_source_count,
    closeness_source_count,
    structural_node_count=0,
):
    if function == "PageRank":
        return lambda: eg.pagerank(
            graph, alpha=0.75, max_iter=200, tol=1.0e-6, weight=None
        )
    if function == "MST":
        return lambda: eg.minimum_spanning_tree(graph, weight="weight")
    if function == "LCC":
        return lambda: eg.clustering(graph)
    if function == "WCC":
        if graph.is_directed():
            return lambda: eg.weakly_connected_components(graph)
        return lambda: eg.connected_components(graph)
    if function == "SCC":
        if not graph.is_directed():
            return lambda: eg.connected_components(graph)
        return lambda: eg.strongly_connected_components(graph)
    if function == "BFS":
        sources = select_sources(graph, source_count)
        return lambda: eg.multi_source_bfs(graph, sources)
    if function == "Dijkstra":
        sources = select_sources(graph, 1)
        return lambda: eg.multi_source_dijkstra(graph, sources, weight="weight")
    if function == "BellmanFord":
        sources = select_sources(graph, source_count)
        return lambda: eg.multi_source_bellman_ford(graph, sources, weight="weight")
    if function == "SSSP":
        sources = select_sources(graph, source_count)
        return lambda: eg.multi_source_dijkstra(graph, sources, weight="weight")
    if function == "KCore":
        if (
            graph.is_directed()
            and getattr(graph, "undirected_projection", None) is None
        ):
            raise NotImplementedError(
                "KCore requires an explicit undirected-projection bulk artifact"
            )
        return lambda: eg.k_core(graph)
    if function == "BC":
        sources = select_sources(graph, bc_source_count)
        return lambda: eg.betweenness_centrality(
            graph,
            weight=None,
            sources=sources,
            normalized=False,
            endpoints=False,
        )
    if function == "Closeness":
        sources = select_sources(graph, closeness_source_count)
        return lambda: eg.closeness_centrality(
            graph,
            weight=None,
            sources=sources,
        )
    if function in {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}:
        selected = (
            select_sources(graph, structural_node_count)
            if structural_node_count > 0
            else None
        )
        if function == "EffectiveSize":
            return lambda: eg.effective_size(graph, nodes=selected, weight=None)
        if function == "Efficiency":
            return lambda: eg.efficiency(graph, nodes=selected, weight=None)
        if function == "Constraint":
            return lambda: eg.constraint(graph, nodes=selected, weight=None)
        return lambda: eg.hierarchy(graph, nodes=selected, weight=None)
    raise ValueError(function)


def kernel_key(function, graph=None):
    if function == "SCC" and graph is not None and not graph.is_directed():
        return "cc"
    return {
        "PageRank": "pagerank",
        "MST": "mst",
        "LCC": "lcc",
        "WCC": "cc",
        "SCC": "scc",
        "BFS": "bfs",
        "Dijkstra": "dijkstra",
        "BellmanFord": "bellman_ford",
        "SSSP": "sssp",
        "KCore": "kcore",
        "BC": "bc",
        "Closeness": "closeness",
        "EffectiveSize": "effective_size",
        "Efficiency": "efficiency",
        "Constraint": "constraint",
        "Hierarchy": "hierarchy",
    }[function]


def required_device_bytes(graph, function, source_count, bc_source_count, closeness_source_count):
    n = int(len(graph))
    m = int(graph.num_entries)
    base_csr = 4 * (n + 1 + m)
    if function == "PageRank":
        # Device CSR + inbound CSR/values + vectors + cuSPARSE margin.
        return base_csr + 8 * m + 64 * n + (1 << 30)
    if function == "WCC":
        return base_csr + 28 * n + (512 << 20)
    if function == "BFS":
        return base_csr + 8 * n * max(1, source_count) + 32 * n + (512 << 20)
    if function == "KCore":
        return base_csr + 48 * n + (512 << 20)
    if function == "SCC":
        return 2 * base_csr + 48 * n + (1 << 30)
    if function == "LCC":
        return base_csr + 48 * n + (1 << 30)
    if function == "MST":
        return base_csr + 8 * m + 64 * n + (1 << 30)
    if function in {"Dijkstra", "BellmanFord", "SSSP"}:
        count = 1 if function == "Dijkstra" else max(1, source_count)
        return base_csr + 8 * n * count + 40 * n + (1 << 30)
    if function == "BC":
        return base_csr + 24 * n * max(1, bc_source_count) + (1 << 30)
    if function == "Closeness":
        return base_csr + 8 * n * max(1, closeness_source_count) + (1 << 30)
    if function in {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}:
        # Structural-hole kernels consume CSR plus COO row/column views.
        return 4 * (n + 1) + 12 * m + 64 * n + (1 << 30)
    return base_csr


def available_device_bytes():
    """Query free memory without creating a CUDA context in this process."""

    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        command = [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]
        if visible:
            command.insert(1, f"--id={visible}")
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        if completed.returncode != 0:
            return None
        free_mib = int(completed.stdout.strip().splitlines()[0])
        return free_mib * 1024 * 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def timed(callable_obj, limit_seconds=None, watchdog=None):
    gc.collect()
    if watchdog is not None:
        watchdog.arm(limit_seconds)
    started = time.perf_counter()
    try:
        value = callable_obj()
        elapsed = time.perf_counter() - started
    finally:
        if watchdog is not None:
            watchdog.disarm()
    if limit_seconds is not None and elapsed > float(limit_seconds):
        raise TimeoutError(
            f"single function call took {elapsed:.3f}s, exceeding the "
            f"{float(limit_seconds):.3f}s protocol limit"
        )
    return value, elapsed


def worker(args):
    requested_runtime, loaded_runtime = load_easygraph_runtime(
        args.easygraph_repo
    )
    worker_identity = _require_runtime_identity(
        requested_runtime, "worker runtime"
    )
    if (
        args.expected_native_sha256 != worker_identity["native_sha256"]
        or args.expected_runtime_python_digest
        != worker_identity["runtime_python_digest"]
    ):
        raise RuntimeError(
            "worker runtime differs from coordinator expectation before "
            "measurement: "
            f"expected_native={args.expected_native_sha256} "
            f"observed_native={worker_identity['native_sha256']} "
            f"expected_python={args.expected_runtime_python_digest} "
            f"observed_python={worker_identity['runtime_python_digest']}"
        )
    worker_execution_placement = {}
    if os.environ.get("EGGPU_STABLE_TIMING_PROTOCOL", "").upper() == "TRUE":
        worker_execution_placement = collect_execution_placement(os.environ)
        validate_controlled_execution(
            worker_execution_placement,
            expected_cpu_affinity=os.environ.get(
                "EGGPU_EXPECTED_CPU_AFFINITY", ""
            ),
            expected_numa_nodes=os.environ.get(
                "EGGPU_EXPECTED_NUMA_NODES", ""
            ),
        )
    started = time.perf_counter()
    graph = eg.read_eggpu_csr(args.manifest, validate=args.validate)
    load_seconds = time.perf_counter() - started
    required = required_device_bytes(
        graph,
        args.function,
        args.source_count,
        args.bc_source_count,
        args.closeness_source_count,
    )
    host_csr_bytes = int(graph.memory_estimate()["host_total_bytes"])
    device_csr_bytes = 4 * (int(len(graph)) + 1 + int(graph.num_entries))
    edge_counts = graph.metadata.get("edge_counts", {})
    rmat = graph.metadata.get("rmat", {})
    input_provenance = {
        "input_family": "controlled_rmat" if rmat else "real_graph",
        "raw_edge_records": edge_counts.get("raw_edge_records"),
        "self_loops_removed": edge_counts.get("self_loops_removed"),
        "duplicates_removed": edge_counts.get("duplicates_removed"),
        "rmat_scale": rmat.get("scale"),
        "rmat_edge_factor": rmat.get("edge_factor"),
    }
    available = available_device_bytes()
    if available is not None and required > int(available * 0.90):
        raise RuntimeError(
            f"memory planner rejected {args.function}: estimated {required} bytes, "
            f"available {available} bytes"
        )
    call = function_call(
        graph,
        args.function,
        args.source_count,
        args.bc_source_count,
        args.closeness_source_count,
        args.structural_node_count,
    )
    structural_query_nodes = (
        select_sources(graph, args.structural_node_count)
        if args.function in {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}
        and args.structural_node_count > 0
        else []
    )
    structural_query_digest = (
        hashlib.sha256(
            np.asarray(structural_query_nodes, dtype=np.int64).tobytes()
        ).hexdigest()
        if structural_query_nodes
        else None
    )
    watchdog = None
    if args.hard_call_timeout:
        watchdog = CallWatchdog(
            Path(str(args.worker_output) + ".timeout.json"),
            args.per_call_timeout,
            args.easygraph_repo,
        )

    if args.measurement == "memory":
        monitor = PeakMemoryMonitor(poll_seconds=args.memory_poll_ms / 1000.0).start()
        value, e2e_seconds = timed(
            call,
            args.first_use_call_timeout,
            watchdog=watchdog,
        )
        memory = monitor.stop()
        record = {
            "status": "ok",
            "measurement": "memory",
            "dataset": graph.name,
            "function": args.function,
            "num_nodes": len(graph),
            "num_edges": graph.num_edges,
            "num_entries": graph.num_entries,
            "directed": graph.is_directed(),
            "load_seconds": load_seconds,
            "e2e_seconds": e2e_seconds,
            "kernel_seconds": gpu_eggpu_backend.get_last_kernel_time(
                kernel_key(args.function, graph)
            ),
            "memory": memory,
            "result": result_digest(value, args.function, num_nodes=len(graph)),
            "estimated_required_device_bytes": required,
            "persistent_host_csr_bytes": host_csr_bytes,
            "persistent_device_csr_bytes": device_csr_bytes,
            "available_device_bytes_before_call": available,
            "first_use_call_timeout_seconds": args.first_use_call_timeout,
            "steady_call_timeout_seconds": args.per_call_timeout,
            "structural_query_node_count": len(structural_query_nodes),
            "structural_query_node_sha256": structural_query_digest,
            **input_provenance,
        }
        record["result_validation"] = validate_result(
            graph,
            args.function,
            value,
            record["result"],
            args.source_count,
            args.bc_source_count,
            args.closeness_source_count,
            args.structural_node_count,
        )
        if record["result_validation"]["status"] != "pass":
            raise RuntimeError(
                f"result invariant validation failed: {record['result_validation']['failures']}"
            )
    else:
        first_value, first_e2e = timed(
            call,
            args.first_use_call_timeout,
            watchdog=watchdog,
        )
        first_kernel = gpu_eggpu_backend.get_last_kernel_time(
            kernel_key(args.function, graph)
        )
        first_digest = result_digest(first_value, args.function, num_nodes=len(graph))
        del first_value
        for _ in range(args.warmup):
            warm, _ = timed(call, args.per_call_timeout, watchdog=watchdog)
            del warm
        e2e_samples = []
        kernel_samples = []
        final_value = None
        for _ in range(args.repeat):
            final_value, elapsed = timed(call, args.per_call_timeout, watchdog=watchdog)
            e2e_samples.append(elapsed)
            kernel_samples.append(
                float(
                    gpu_eggpu_backend.get_last_kernel_time(
                        kernel_key(args.function, graph)
                    )
                )
            )
            if len(e2e_samples) < args.repeat:
                del final_value
                final_value = None
        final_digest = result_digest(final_value, args.function, num_nodes=len(graph))
        if not result_digests_equivalent(args.function, first_digest, final_digest):
            difference = result_digest_difference(first_digest, final_digest)
            raise RuntimeError(
                "first-use and steady-state result digests differ: "
                + json.dumps(difference, sort_keys=True)
            )
        record = {
            "status": "ok",
            "measurement": "timing",
            "dataset": graph.name,
            "function": args.function,
            "num_nodes": len(graph),
            "num_edges": graph.num_edges,
            "num_entries": graph.num_entries,
            "directed": graph.is_directed(),
            "load_seconds": load_seconds,
            "first_use_e2e_seconds": first_e2e,
            "first_use_kernel_seconds": first_kernel,
            "user_cold_total_seconds": load_seconds + first_e2e,
            "steady_e2e": stats(e2e_samples),
            "steady_kernel": stats(kernel_samples),
            "result": final_digest,
            "estimated_required_device_bytes": required,
            "persistent_host_csr_bytes": host_csr_bytes,
            "persistent_device_csr_bytes": device_csr_bytes,
            "available_device_bytes_before_call": available,
            "first_use_call_timeout_seconds": args.first_use_call_timeout,
            "steady_call_timeout_seconds": args.per_call_timeout,
            "first_use_calls": 1,
            "additional_warmup_calls": args.warmup,
            "preceding_public_calls": 1 + args.warmup,
            "measured_call_position": 2 + args.warmup,
            "measured_calls_per_process": args.repeat,
            "timer_boundary": (
                "EGGPU public invocation through complete result return; "
                "benchmark validation excluded"
            ),
            "validation_outside_timer": True,
            "structural_query_node_count": len(structural_query_nodes),
            "structural_query_node_sha256": structural_query_digest,
            **input_provenance,
        }
        e2e_mean = record["steady_e2e"]["mean"]
        kernel_mean = record["steady_kernel"]["mean"]
        record["efficiency"] = {
            # This is an input-size-normalized rate, not an algorithmic edge-
            # visit rate.  Iterative and frontier algorithms may inspect an
            # input entry zero, one, or many times during a call.
            "input_csr_entries_per_second_e2e": (
                float(graph.num_entries) / e2e_mean if e2e_mean > 0 else None
            ),
            "input_csr_entries_per_second_kernel": (
                float(graph.num_entries) / kernel_mean if kernel_mean > 0 else None
            ),
            "steady_non_kernel_seconds": max(0.0, e2e_mean - kernel_mean),
            "first_use_non_kernel_seconds": max(0.0, first_e2e - first_kernel),
            "steady_non_kernel_fraction": (
                max(0.0, e2e_mean - kernel_mean) / e2e_mean
                if e2e_mean > 0
                else None
            ),
        }
        record["result_validation"] = validate_result(
            graph,
            args.function,
            final_value,
            final_digest,
            args.source_count,
            args.bc_source_count,
            args.closeness_source_count,
            args.structural_node_count,
        )
        if record["result_validation"]["status"] != "pass":
            raise RuntimeError(
                f"result invariant validation failed: {record['result_validation']['failures']}"
            )
    record.update(
        runtime_record_envelope(
            args.easygraph_repo,
            requested_runtime,
            loaded_runtime=loaded_runtime,
        )
    )
    if worker_execution_placement:
        record["worker_execution_placement"] = worker_execution_placement
    if watchdog is not None:
        watchdog.close()
    Path(args.worker_output).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_child(
    base_args,
    manifest,
    function,
    measurement,
    output,
    timeout,
    worker_repeat=None,
):
    requested_runtime = getattr(
        base_args, "requested_runtime_provenance", None
    )
    if requested_runtime is None:
        requested_runtime = collect_runtime_repository_provenance(
            base_args.easygraph_repo
        )
    requested_identity = _require_runtime_identity(
        requested_runtime, "coordinator runtime"
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--easygraph-repo",
        str(Path(base_args.easygraph_repo).expanduser().resolve()),
        "--expected-native-sha256",
        requested_identity["native_sha256"],
        "--expected-runtime-python-digest",
        requested_identity["runtime_python_digest"],
        "--manifest",
        str(manifest),
        "--function",
        function,
        "--measurement",
        measurement,
        "--worker-output",
        str(output),
        "--repeat",
        str(base_args.repeat if worker_repeat is None else worker_repeat),
        "--warmup",
        str(base_args.warmup),
        "--source-count",
        str(base_args.source_count),
        "--bc-source-count",
        str(base_args.bc_source_count),
        "--closeness-source-count",
        str(base_args.closeness_source_count),
        "--structural-node-count",
        str(base_args.structural_node_count),
        "--memory-poll-ms",
        str(base_args.memory_poll_ms),
        "--per-call-timeout",
        str(base_args.timeout),
        "--first-use-call-timeout",
        str(base_args.first_use_call_timeout),
    ]
    if base_args.validate:
        command.append("--validate")
    if base_args.hard_call_timeout:
        command.append("--hard-call-timeout")
    steady_calls = 0 if measurement == "memory" else (
        int(base_args.warmup)
        + int(base_args.repeat if worker_repeat is None else worker_repeat)
    )
    process_timeout = (
        float(base_args.load_timeout)
        + float(base_args.first_use_call_timeout)
        + float(timeout) * steady_calls
    )
    environment = runtime_subprocess_environment(base_args.easygraph_repo)
    parent_memory = None
    if measurement == "memory":
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        physical_gpu = int(
            os.environ.get(
                "EGGPU_MONITOR_GPU_INDEX",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
            )
        )
        monitor = ChildProcessMemoryMonitor(
            process.pid,
            physical_gpu=physical_gpu,
            interval_seconds=base_args.memory_poll_ms / 1000.0,
        ).start()
        try:
            stdout, _ = process.communicate(timeout=process_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        finally:
            parent_memory = monitor.stop()
        completed = subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout=stdout,
            stderr=None,
        )
    else:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=process_timeout,
            check=False,
            env=environment,
        )
    if completed.returncode != 0:
        timeout_marker = Path(str(output) + ".timeout.json")
        if timeout_marker.is_file():
            payload = json.loads(timeout_marker.read_text(encoding="utf-8"))
            raise TimeoutError(
                f"hard per-call timeout after "
                f"{float(payload.get('limit_seconds', base_args.timeout)):.3f}s for "
                f"{manifest}/{function}/{measurement}; watchdog={payload}"
            )
        return_signal = None
        if completed.returncode < 0:
            try:
                return_signal = signal.Signals(-completed.returncode).name
            except ValueError:
                return_signal = f"signal_{-completed.returncode}"
        failure_payload = {
            **runtime_record_envelope(
                base_args.easygraph_repo,
                requested_runtime,
                argv=command,
                environment=environment,
            ),
            "command": command,
            "function": function,
            "manifest": str(manifest),
            "measurement": measurement,
            "returncode": completed.returncode,
            "signal": return_signal,
            "stdout_tail": (completed.stdout or "")[-12000:],
        }
        Path(str(output) + ".failure.json").write_text(
            json.dumps(failure_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"scaling worker failed for {manifest}/{function}/{measurement}; "
            f"returncode={completed.returncode}, signal={return_signal}:\n"
            f"{(completed.stdout or '')[-12000:]}"
        )
    record = json.loads(output.read_text(encoding="utf-8"))
    worker_repository = record.get("requested_runtime_provenance") or {}
    worker_loaded = record.get("runtime_provenance") or {}
    assert_runtime_matches(
        requested_runtime, worker_repository, worker_loaded
    )
    worker_stdout = (completed.stdout or "").strip()
    if worker_stdout:
        record["worker_stdout_tail"] = worker_stdout[-12000:]
    if parent_memory is not None:
        # Preserve the call-window probe for diagnosis, but publish the
        # coordinator-side full worker-process peak as the authoritative
        # memory value. It includes graph loading, first-use setup, the call,
        # and result reconstruction, matching the split-memory main protocol.
        record["worker_call_window_memory"] = record.get("memory")
        parent_memory["measurement_window"] = (
            "isolated_worker_process_full_lifetime"
        )
        record["memory"] = parent_memory
        output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return record


TERMINAL_FAILURE_KINDS = {
    "timeout",
    "oom",
    "resource_limit",
    "representation_limit",
    "unsupported",
    "missing_weight_artifact",
}


def terminal_failure(record):
    if not isinstance(record, dict):
        return False
    kinds = str(record.get("failure_kind", record.get("status", ""))).split("+")
    return any(kind in TERMINAL_FAILURE_KINDS for kind in kinds)


def reusable_record(
    path,
    dataset,
    function,
    measurement,
    requested_runtime,
    *,
    reuse_terminal_failures=False,
):
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not record_runtime_matches(record, requested_runtime):
        return None
    if (
        record.get("dataset") == dataset
        and record.get("function") == function
        and record.get("measurement") == measurement
    ):
        if record.get("status") != "ok":
            return record if reuse_terminal_failures and terminal_failure(record) else None
        if measurement == "memory":
            memory = record.get("memory") or {}
            if memory.get("memory_monitor_origin") != (
                "coordinator_process_child_tree"
            ):
                return None
            try:
                rss_samples = int(memory.get("monitor_rss_samples") or 0)
                gpu_samples = int(memory.get("monitor_gpu_proc_samples") or 0)
                rss_peak = float(memory.get("rss_mb") or 0.0)
                gpu_peak = float(memory.get("gpu_proc_peak_mb") or 0.0)
            except (TypeError, ValueError):
                return None
            if rss_samples < 1 or gpu_samples < 1 or rss_peak <= 0 or gpu_peak <= 0:
                return None
        return record
    return None


def classify_failure(error):
    text = str(error)
    lowered = text.lower()
    if (
        isinstance(error, TimeoutError)
        or
        isinstance(error, subprocess.TimeoutExpired)
        or "timed out" in lowered
        or "timeout" in lowered
        or "timeouterror" in lowered
        or "protocol limit" in lowered
    ):
        return "timeout"
    if "out of memory" in lowered or "cuda_error_out_of_memory" in lowered:
        return "oom"
    if "memory planner rejected" in lowered:
        return "resource_limit"
    if "not implemented" in lowered or "notimplementederror" in lowered:
        return "unsupported"
    if "validation failed" in lowered or "result invariant" in lowered:
        return "validation_failed"
    return "execution_error"


def skip_kind(function, metadata, reason):
    if function in PROJECTED_FUNCTIONS and bool(metadata.get("directed")):
        return "representation_limit"
    if function in WEIGHTED_FUNCTIONS and not metadata.get("weights_path"):
        return "missing_weight_artifact"
    return "unsupported"


def coordinator(args):
    requested_runtime = collect_runtime_repository_provenance(
        args.easygraph_repo
    )
    args.requested_runtime_provenance = requested_runtime
    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    coordinator_envelope = runtime_record_envelope(
        args.easygraph_repo,
        requested_runtime,
    )

    def attach_coordinator_runtime(record):
        record.update(coordinator_envelope)
        return record

    worker_environment = runtime_subprocess_environment(args.easygraph_repo)
    run_metadata = {
        "protocol": "eggpu_scaling_frozen_runtime_v10",
        "argv": list(sys.argv),
        "easygraph_repo": requested_runtime["resolved_root"],
        "repository_runtime_provenance": requested_runtime,
        "runtime_python_digest": requested_runtime[
            "runtime_python_snapshot"
        ]["digest"],
        "python_origins": {
            name: artifact["module_origin_resolved"]
            for name, artifact in requested_runtime["modules"].items()
        },
        "native_binary_sha256": requested_runtime["native_sha256"],
        "controlled_environment": {
            "coordinator": collect_relevant_environment(),
            "worker": collect_relevant_environment(worker_environment),
        },
        "timing_semantics": (
            "Runtime fingerprinting and origin validation complete before graph "
            "loading and before every measured user call; call timers are unchanged."
        ),
        "requested_functions": args.functions,
        "requested_manifests": list(args.manifests),
        "timing_processes": args.timing_processes,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "memory_repeat": args.memory_repeat,
        "stable_timing_protocol": getattr(
            args, "stable_timing_protocol_metadata", {}
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    functions = (
        list(ALL_FUNCTIONS)
        if args.functions.strip().lower() == "all"
        else [item for item in args.functions.split(",") if item]
    )
    unknown = sorted(set(functions) - set(FUNCTIONS))
    if unknown:
        raise ValueError(f"unknown scaling functions: {unknown}")
    records = []
    retry_cells = set()
    for value in args.retry_cells.split(","):
        value = value.strip()
        if not value:
            continue
        if ":" not in value:
            raise ValueError(
                f"invalid --retry-cells entry {value!r}; expected DATASET:FUNCTION"
            )
        dataset_name, function_name = value.split(":", 1)
        if function_name not in FUNCTIONS:
            raise ValueError(f"unknown function in --retry-cells: {function_name}")
        retry_cells.add((dataset_name, function_name))
    timing_processes = args.timing_processes if args.timing_processes > 0 else args.repeat
    timing_worker_repeat = args.repeat if args.timing_processes > 0 else 1
    total = len(args.manifests) * len(functions) * (timing_processes + args.memory_repeat)
    progress = 0
    for manifest_value in args.manifests:
        manifest = Path(manifest_value).resolve()
        for function in functions:
            function_manifest = manifest_for_function(manifest, function)
            metadata = json.loads(function_manifest.read_text(encoding="utf-8"))
            dataset = metadata["name"]
            force_retry_cell = (dataset, function) in retry_cells
            applicable, skip_reason = function_applicability(metadata, function)
            if not applicable:
                measurements = [
                    f"timing-{index + 1}" for index in range(timing_processes)
                ] + [
                    f"memory-{index + 1}" for index in range(args.memory_repeat)
                ]
                for measurement in measurements:
                    progress += 1
                    print(
                        f"[scaling {progress}/{total}] {dataset}/{function}/{measurement} "
                        f"skipped: {skip_reason}",
                        flush=True,
                    )
                    record = {
                        "status": "skipped",
                        "measurement": (
                            "memory" if measurement.startswith("memory-") else "timing"
                        ),
                        "measurement_instance": measurement,
                        "dataset": dataset,
                        "function": function,
                        "num_nodes": metadata.get("num_nodes"),
                        "num_edges": metadata.get("num_edges"),
                        "num_entries": metadata.get("num_entries"),
                        "directed": bool(metadata.get("directed")),
                        "failure_kind": skip_kind(function, metadata, skip_reason),
                        "skip_reason": skip_reason,
                    }
                    attach_coordinator_runtime(record)
                    skip_path = raw_dir / f"{dataset}_{function}_{measurement}.json"
                    skip_path.write_text(
                        json.dumps(record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                records.append(
                    attach_coordinator_runtime({
                        "status": "skipped",
                        "measurement": "timing",
                        "dataset": dataset,
                        "function": function,
                        "num_nodes": metadata.get("num_nodes"),
                        "num_edges": metadata.get("num_edges"),
                        "num_entries": metadata.get("num_entries"),
                        "directed": bool(metadata.get("directed")),
                        "failure_kind": skip_kind(function, metadata, skip_reason),
                        "skip_reason": skip_reason,
                    })
                )
                if args.memory_repeat:
                    records.append(
                        attach_coordinator_runtime({
                            "status": "skipped",
                            "measurement": "memory",
                            "dataset": dataset,
                            "function": function,
                            "num_nodes": metadata.get("num_nodes"),
                            "num_edges": metadata.get("num_edges"),
                            "num_entries": metadata.get("num_entries"),
                            "directed": bool(metadata.get("directed")),
                            "failure_kind": skip_kind(function, metadata, skip_reason),
                            "skip_reason": skip_reason,
                        })
                    )
                continue
            timing_samples = []
            timing_errors = []
            terminal_timing_record = None
            for timing_index in range(timing_processes):
                propagated_terminal = False
                progress += 1
                print(
                    f"[scaling {progress}/{total}] {dataset}/{function}/"
                    f"timing-{timing_index + 1}",
                    flush=True,
                )
                timing_path = raw_dir / (
                    f"{dataset}_{function}_timing_{timing_index + 1}.json"
                )
                if terminal_timing_record is not None:
                    propagated_terminal = True
                    timing_record = {
                        "status": "skipped",
                        "failure_kind": "not_run_after_terminal_failure",
                        "root_failure_kind": terminal_timing_record.get(
                            "failure_kind", terminal_timing_record.get("status")
                        ),
                        "measurement": "timing",
                        "measurement_instance": f"timing-{timing_index + 1}",
                        "dataset": dataset,
                        "function": function,
                        "num_nodes": metadata.get("num_nodes"),
                        "num_edges": metadata.get("num_edges"),
                        "num_entries": metadata.get("num_entries"),
                        "directed": bool(metadata.get("directed")),
                        "skip_reason": (
                            "A prior fresh-process timing sample reached a terminal "
                            "outcome under the same protocol."
                        ),
                    }
                    attach_coordinator_runtime(timing_record)
                    timing_path.write_text(
                        json.dumps(timing_record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                else:
                    timing_record = (
                        reusable_record(
                            timing_path,
                            dataset,
                            function,
                            "timing",
                            requested_runtime,
                            reuse_terminal_failures=not force_retry_cell,
                        )
                        if args.resume
                        else None
                    )
                if timing_record is not None:
                    action = (
                        "propagated terminal outcome to"
                        if propagated_terminal
                        else "reused"
                    )
                    print(f"[scaling] {action} {timing_path}", flush=True)
                elif terminal_timing_record is None:
                    try:
                        timing_record = run_child(
                            args,
                            function_manifest,
                            function,
                            "timing",
                            timing_path,
                            args.timeout,
                            worker_repeat=timing_worker_repeat,
                        )
                    except Exception as exc:
                        timing_record = {
                            "status": classify_failure(exc),
                            "failure_kind": classify_failure(exc),
                            "measurement": "timing",
                            "measurement_instance": f"timing-{timing_index + 1}",
                            "dataset": dataset,
                            "function": function,
                            "num_nodes": metadata.get("num_nodes"),
                            "num_edges": metadata.get("num_edges"),
                            "num_entries": metadata.get("num_entries"),
                            "directed": bool(metadata.get("directed")),
                            "error": str(exc),
                        }
                        attach_coordinator_runtime(timing_record)
                        timing_path.write_text(
                            json.dumps(timing_record, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                        print(f"[scaling] failed: {exc}", file=sys.stderr, flush=True)
                if timing_record.get("status") == "ok":
                    timing_samples.append(timing_record)
                else:
                    timing_errors.append(timing_record)
                    if terminal_failure(timing_record):
                        terminal_timing_record = timing_record
            if timing_errors:
                kinds = sorted(
                    {
                        record.get("failure_kind", record.get("status", "execution_error"))
                        for record in timing_errors
                        if record.get("failure_kind") != "not_run_after_terminal_failure"
                    }
                )
                if not kinds:
                    kinds = ["execution_error"]
                timing_record = {
                    "status": "failed",
                    "failure_kind": "+".join(kinds),
                    "measurement": "timing",
                    "dataset": dataset,
                    "function": function,
                    "num_nodes": metadata.get("num_nodes"),
                    "num_edges": metadata.get("num_edges"),
                    "num_entries": metadata.get("num_entries"),
                    "directed": bool(metadata.get("directed")),
                    "successful_timing_processes": len(timing_samples),
                    "failed_timing_processes": len(timing_errors),
                    "error": " | ".join(
                        record.get("error", "unknown failure")
                        for record in timing_errors
                    ),
                }
                attach_coordinator_runtime(timing_record)
            else:
                try:
                    timing_record = aggregate_timing_records(timing_samples)
                    if getattr(
                        args, "eggpu_stable_timing_protocol", False
                    ):
                        timing_stability = summarize_five_samples(
                            timing_record["steady_e2e"]["samples"],
                            max_over_median_limit=(
                                args.stability_max_over_median_limit
                            ),
                            median_over_min_limit=(
                                args.stability_median_over_min_limit
                            ),
                        )
                        timing_record["timing_stability"] = timing_stability
                        timing_record["paper_estimator"] = PAPER_ESTIMATOR
                        timing_record["submission_e2e_seconds"] = (
                            timing_stability["submission_seconds"]
                        )
                except Exception as exc:
                    timing_record = {
                        "status": "failed",
                        "measurement": "timing",
                        "dataset": dataset,
                        "function": function,
                        "num_nodes": metadata.get("num_nodes"),
                        "num_edges": metadata.get("num_edges"),
                        "num_entries": metadata.get("num_entries"),
                        "directed": bool(metadata.get("directed")),
                        "error": f"timing aggregation failed: {exc}",
                    }
                    attach_coordinator_runtime(timing_record)
            aggregate_path = raw_dir / f"{dataset}_{function}_timing.json"
            aggregate_path.write_text(
                json.dumps(timing_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            records.append(timing_record)
            for memory_index in range(args.memory_repeat):
                timing_unavailable = False
                progress += 1
                print(
                    f"[scaling {progress}/{total}] {dataset}/{function}/memory-{memory_index + 1}",
                    flush=True,
                )
                memory_path = raw_dir / f"{dataset}_{function}_memory_{memory_index + 1}.json"
                if timing_record.get("status") != "ok":
                    timing_unavailable = True
                    memory_record = {
                        "status": "skipped",
                        "failure_kind": "timing_unavailable",
                        "root_failure_kind": timing_record.get(
                            "failure_kind", timing_record.get("status")
                        ),
                        "measurement": "memory",
                        "measurement_instance": f"memory-{memory_index + 1}",
                        "dataset": dataset,
                        "function": function,
                        "num_nodes": metadata.get("num_nodes"),
                        "num_edges": metadata.get("num_edges"),
                        "num_entries": metadata.get("num_entries"),
                        "directed": bool(metadata.get("directed")),
                        "skip_reason": (
                            "Memory measurement is not run because the timing contract "
                            "did not produce a complete valid sample set."
                        ),
                    }
                    attach_coordinator_runtime(memory_record)
                    memory_path.write_text(
                        json.dumps(memory_record, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                else:
                    memory_record = (
                        reusable_record(
                            memory_path,
                            dataset,
                            function,
                            "memory",
                            requested_runtime,
                            reuse_terminal_failures=not force_retry_cell,
                        )
                        if args.resume
                        else None
                    )
                if memory_record is not None:
                    action = "skipped after failed timing" if timing_unavailable else "reused"
                    print(f"[scaling] {action} {memory_path}", flush=True)
                elif timing_record.get("status") == "ok":
                    try:
                        memory_record = run_child(
                            args, function_manifest, function, "memory", memory_path, args.timeout
                        )
                    except Exception as exc:
                        memory_record = {
                            "status": classify_failure(exc),
                            "failure_kind": classify_failure(exc),
                            "measurement": "memory",
                            "measurement_instance": f"memory-{memory_index + 1}",
                            "dataset": dataset,
                            "function": function,
                            "num_nodes": metadata.get("num_nodes"),
                            "num_edges": metadata.get("num_edges"),
                            "num_entries": metadata.get("num_entries"),
                            "directed": bool(metadata.get("directed")),
                            "error": str(exc),
                        }
                        attach_coordinator_runtime(memory_record)
                        memory_path.write_text(
                            json.dumps(memory_record, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                        print(f"[scaling] failed: {exc}", file=sys.stderr, flush=True)
                records.append(memory_record)

    (output_dir / "scaling_all.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "status",
        "failure_kind",
        "skip_reason",
        "dataset",
        "function",
        "measurement",
        "num_nodes",
        "num_edges",
        "num_entries",
        "directed",
        "input_family",
        "raw_edge_records",
        "self_loops_removed",
        "duplicates_removed",
        "rmat_scale",
        "rmat_edge_factor",
        "structural_query_node_count",
        "structural_query_node_sha256",
        "timing_process_samples",
        "first_use_calls",
        "additional_warmup_calls",
        "preceding_public_calls",
        "measured_call_position",
        "measured_calls_per_process",
        "timer_boundary",
        "validation_outside_timer",
        "first_use_call_timeout_seconds",
        "steady_call_timeout_seconds",
        "load_seconds",
        "load_stdev_seconds",
        "first_use_e2e_seconds",
        "first_use_e2e_stdev_seconds",
        "first_use_kernel_seconds",
        "first_use_kernel_stdev_seconds",
        "user_cold_total_seconds",
        "user_cold_total_stdev_seconds",
        "steady_e2e_mean",
        "steady_e2e_stdev",
        "steady_e2e_samples",
        "steady_e2e_minimum",
        "steady_e2e_median",
        "steady_e2e_maximum",
        "steady_e2e_cv",
        "max_over_median",
        "median_over_min",
        "variance_policy",
        "paper_estimator",
        "batch_acceptance_status",
        "submission_e2e_seconds",
        "steady_kernel_mean",
        "steady_kernel_stdev",
        "e2e_seconds",
        "kernel_seconds",
        "rss_peak_mb",
        "rss_peak_delta_mb",
        "gpu_proc_peak_delta_mb",
        "gpu_proc_peak_mb",
        "result_minimum",
        "result_maximum",
        "result_finite_count",
        "result_unique_count",
        "result_component_count",
        "result_partition_sha256",
        "result_output_bytes",
        "result_validation",
        "persistent_host_csr_bytes",
        "persistent_device_csr_bytes",
        "estimated_required_device_bytes",
        "input_csr_entries_per_second_e2e",
        "input_csr_entries_per_second_kernel",
        "steady_non_kernel_seconds",
        "first_use_non_kernel_seconds",
        "steady_non_kernel_fraction",
        "error",
    ]
    with (output_dir / "scaling_all.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            memory = record.get("memory", {})
            writer.writerow(
                {
                    "status": record.get("status"),
                    "failure_kind": record.get("failure_kind"),
                    "skip_reason": record.get("skip_reason"),
                    "dataset": record.get("dataset"),
                    "function": record.get("function"),
                    "measurement": record.get("measurement"),
                    "num_nodes": record.get("num_nodes"),
                    "num_edges": record.get("num_edges"),
                    "num_entries": record.get("num_entries"),
                    "directed": record.get("directed"),
                    "input_family": record.get("input_family"),
                    "raw_edge_records": record.get("raw_edge_records"),
                    "self_loops_removed": record.get("self_loops_removed"),
                    "duplicates_removed": record.get("duplicates_removed"),
                    "rmat_scale": record.get("rmat_scale"),
                    "rmat_edge_factor": record.get("rmat_edge_factor"),
                    "structural_query_node_count": record.get(
                        "structural_query_node_count"
                    ),
                    "structural_query_node_sha256": record.get(
                        "structural_query_node_sha256"
                    ),
                    "timing_process_samples": record.get("timing_process_samples"),
                    "first_use_calls": record.get("first_use_calls"),
                    "additional_warmup_calls": record.get(
                        "additional_warmup_calls"
                    ),
                    "preceding_public_calls": record.get(
                        "preceding_public_calls"
                    ),
                    "measured_call_position": record.get(
                        "measured_call_position"
                    ),
                    "measured_calls_per_process": record.get(
                        "measured_calls_per_process"
                    ),
                    "timer_boundary": record.get("timer_boundary"),
                    "validation_outside_timer": record.get(
                        "validation_outside_timer"
                    ),
                    "first_use_call_timeout_seconds": record.get(
                        "first_use_call_timeout_seconds"
                    ),
                    "steady_call_timeout_seconds": record.get(
                        "steady_call_timeout_seconds"
                    ),
                    "load_seconds": record.get("load_seconds"),
                    "load_stdev_seconds": record.get("load", {}).get("stdev"),
                    "first_use_e2e_seconds": record.get("first_use_e2e_seconds"),
                    "first_use_e2e_stdev_seconds": record.get("first_use_e2e", {}).get("stdev"),
                    "first_use_kernel_seconds": record.get("first_use_kernel_seconds"),
                    "first_use_kernel_stdev_seconds": record.get("first_use_kernel", {}).get("stdev"),
                    "user_cold_total_seconds": record.get("user_cold_total_seconds"),
                    "user_cold_total_stdev_seconds": record.get("user_cold_total", {}).get("stdev"),
                    "steady_e2e_mean": record.get("steady_e2e", {}).get("mean"),
                    "steady_e2e_stdev": record.get("steady_e2e", {}).get("stdev"),
                    "steady_e2e_samples": json.dumps(
                        record.get("steady_e2e", {}).get("samples", []),
                        separators=(",", ":"),
                    ),
                    "steady_e2e_minimum": record.get(
                        "steady_e2e", {}
                    ).get("best"),
                    "steady_e2e_median": (
                        record.get("timing_stability", {}).get(
                            "median_seconds"
                        )
                    ),
                    "steady_e2e_maximum": (
                        record.get("timing_stability", {}).get(
                            "maximum_seconds"
                        )
                    ),
                    "steady_e2e_cv": (
                        record.get("timing_stability", {}).get(
                            "coefficient_of_variation"
                        )
                    ),
                    "max_over_median": (
                        record.get("timing_stability", {}).get(
                            "max_over_median"
                        )
                    ),
                    "median_over_min": (
                        record.get("timing_stability", {}).get(
                            "median_over_min"
                        )
                    ),
                    "variance_policy": (
                        record.get("timing_stability", {}).get(
                            "variance_policy"
                        )
                    ),
                    "paper_estimator": record.get("paper_estimator"),
                    "batch_acceptance_status": (
                        record.get("timing_stability", {}).get(
                            "batch_acceptance_status"
                        )
                    ),
                    "submission_e2e_seconds": record.get(
                        "submission_e2e_seconds"
                    ),
                    "steady_kernel_mean": record.get("steady_kernel", {}).get("mean"),
                    "steady_kernel_stdev": record.get("steady_kernel", {}).get("stdev"),
                    "e2e_seconds": record.get("e2e_seconds"),
                    "kernel_seconds": record.get("kernel_seconds"),
                    "rss_peak_mb": memory.get("rss_mb"),
                    "rss_peak_delta_mb": memory.get("rss_peak_delta_mb"),
                    "gpu_proc_peak_delta_mb": memory.get("gpu_proc_peak_delta_mb"),
                    "gpu_proc_peak_mb": memory.get("gpu_proc_peak_mb"),
                    "result_minimum": record.get("result", {}).get("minimum"),
                    "result_maximum": record.get("result", {}).get("maximum"),
                    "result_finite_count": record.get("result", {}).get("finite_count"),
                    "result_unique_count": record.get("result", {}).get("unique_count"),
                    "result_component_count": record.get("result", {}).get("component_count"),
                    "result_partition_sha256": record.get("result", {}).get("partition_sha256"),
                    "result_output_bytes": record.get("result", {}).get("nbytes"),
                    "result_validation": record.get("result_validation", {}).get("status"),
                    "persistent_host_csr_bytes": record.get("persistent_host_csr_bytes"),
                    "persistent_device_csr_bytes": record.get("persistent_device_csr_bytes"),
                    "estimated_required_device_bytes": record.get("estimated_required_device_bytes"),
                    "input_csr_entries_per_second_e2e": record.get("efficiency", {}).get(
                        "input_csr_entries_per_second_e2e"
                    ),
                    "input_csr_entries_per_second_kernel": record.get("efficiency", {}).get(
                        "input_csr_entries_per_second_kernel"
                    ),
                    "steady_non_kernel_seconds": record.get("efficiency", {}).get("steady_non_kernel_seconds"),
                    "first_use_non_kernel_seconds": record.get("efficiency", {}).get("first_use_non_kernel_seconds"),
                    "steady_non_kernel_fraction": record.get("efficiency", {}).get(
                        "steady_non_kernel_fraction"
                    ),
                    "error": record.get("error"),
                }
            )
    failed = [
        record
        for record in records
        if record.get("status") not in {"ok", "skipped"}
    ]
    rejected_timing_batches = [
        record
        for record in records
        if record.get("measurement") == "timing"
        and record.get("timing_stability", {}).get("stability_status")
        == "fail"
    ]
    print(f"Done: {output_dir}", flush=True)
    if rejected_timing_batches and getattr(
        args, "eggpu_stable_timing_protocol", False
    ):
        print(
            "Scaling rejected "
            f"{len(rejected_timing_batches)} complete timing batch(es) under "
            "the catastrophic-outlier guard",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if failed and not args.continue_on_failure:
        print(f"Scaling completed with {len(failed)} failed task(s)", file=sys.stderr)
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--easygraph-repo",
        type=Path,
        required=True,
        help=(
            "Frozen Easy-Graph runtime containing easygraph/ and the "
            "cpp_easygraph native extension."
        ),
    )
    parser.add_argument("--manifests", nargs="+")
    parser.add_argument("--functions", default=",".join(SCALING_FUNCTIONS))
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of independent fresh-process timing samples.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--timing-processes",
        type=int,
        default=0,
        help=(
            "Fresh timing processes. Zero preserves the original scaling protocol "
            "(repeat processes with one measured call each); a positive value runs "
            "--repeat measured calls inside each process. Use 1 for the main table."
        ),
    )
    parser.add_argument("--memory-repeat", type=int, default=3)
    parser.add_argument("--source-count", type=int, default=4)
    parser.add_argument("--bc-source-count", type=int, default=16)
    parser.add_argument("--closeness-source-count", type=int, default=16)
    parser.add_argument(
        "--structural-node-count",
        type=int,
        default=0,
        help=(
            "Evaluate structural-hole functions on this deterministic node subset. "
            "Zero preserves the full-node public call."
        ),
    )
    parser.add_argument("--memory-poll-ms", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--load-timeout", type=float, default=900.0)
    parser.add_argument("--per-call-timeout", type=float)
    parser.add_argument(
        "--first-use-call-timeout",
        type=float,
        help=(
            "Timeout for the first call in a fresh process, including reusable "
            "graph-state preparation. Warmup and steady calls retain --timeout. "
            "Defaults to --timeout."
        ),
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--eggpu-stable-timing-protocol",
        action="store_true",
        help=(
            "Require one controlled five-sample timing batch per cell, record "
            "CPU-affinity/NUMA/thread provenance, and reject the complete batch "
            "when the catastrophic-outlier guard fails."
        ),
    )
    parser.add_argument(
        "--expected-cpu-affinity",
        default="",
        help="Optional exact taskset CPU list to verify before measurement.",
    )
    parser.add_argument(
        "--expected-numa-nodes",
        default="",
        help=(
            "Optional exact memory-node list verified through numactl or the "
            "launcher's libnuma policy before measurement."
        ),
    )
    parser.add_argument(
        "--stability-max-over-median-limit",
        type=float,
        default=DEFAULT_MAX_OVER_MEDIAN_LIMIT,
        help="Complete-raw5 max/median guard (formal default: 5.0).",
    )
    parser.add_argument(
        "--stability-median-over-min-limit",
        type=float,
        default=DEFAULT_MEDIAN_OVER_MIN_LIMIT,
        help=(
            "Complete-raw5 median/min guard (calibrated formal default: 3.0)."
        ),
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument(
        "--retry-cells",
        default="",
        help=(
            "Comma-separated DATASET:FUNCTION cells whose existing failed samples "
            "must be recomputed after a code fix. Terminal failures in other cells "
            "are preserved as protocol outcomes."
        ),
    )
    parser.add_argument(
        "--hard-call-timeout",
        action="store_true",
        help=(
            "Use an out-of-process watchdog to enforce the per-call timeout "
            "even while a native CUDA extension is executing."
        ),
    )
    parser.add_argument("--watchdog", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--watchdog-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--expected-native-sha256",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-runtime-python-digest",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--manifest")
    parser.add_argument("--function", choices=FUNCTIONS)
    parser.add_argument("--measurement", choices=["timing", "memory"])
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if args.watchdog:
        if args.watchdog_pid is None:
            parser.error("--watchdog requires --watchdog-pid")
        return args
    if args.per_call_timeout is None:
        args.per_call_timeout = args.timeout
    if args.first_use_call_timeout is None:
        args.first_use_call_timeout = args.per_call_timeout
    if args.worker:
        required = [
            args.manifest,
            args.function,
            args.measurement,
            args.worker_output,
            args.expected_native_sha256,
            args.expected_runtime_python_digest,
        ]
    else:
        required = [args.manifests, args.output_dir]
    if any(value is None for value in required):
        parser.error("missing required worker or coordinator arguments")
    return args


def configure_stable_timing_protocol(args):
    """Apply controlled launcher state before importing or spawning workers."""

    args.stable_timing_protocol_metadata = {}
    if not args.eggpu_stable_timing_protocol:
        return
    if args.repeat != EXPECTED_TIMING_SAMPLES:
        raise SystemExit(
            "--eggpu-stable-timing-protocol requires "
            f"--repeat {EXPECTED_TIMING_SAMPLES}"
        )
    if args.timing_processes != 0:
        raise SystemExit(
            "--eggpu-stable-timing-protocol requires --timing-processes 0 "
            "(five independent fresh worker processes)"
        )
    if args.memory_repeat != 0:
        raise SystemExit(
            "--eggpu-stable-timing-protocol requires --memory-repeat 0; "
            "run memory measurements as a separate pass"
        )
    if args.resume:
        raise SystemExit(
            "--eggpu-stable-timing-protocol requires --no-resume so a batch "
            "cannot mix samples collected under different placement states"
        )
    if args.stability_max_over_median_limit < 1:
        raise SystemExit("--stability-max-over-median-limit must be at least 1")
    if args.stability_median_over_min_limit < 1:
        raise SystemExit("--stability-median-over-min-limit must be at least 1")
    apply_controlled_thread_environment(os.environ)
    os.environ["EGGPU_STABLE_TIMING_PROTOCOL"] = "TRUE"
    os.environ["EGGPU_EXPECTED_CPU_AFFINITY"] = args.expected_cpu_affinity
    os.environ["EGGPU_EXPECTED_NUMA_NODES"] = args.expected_numa_nodes
    placement = collect_execution_placement(os.environ)
    try:
        validate_controlled_execution(
            placement,
            expected_cpu_affinity=args.expected_cpu_affinity,
            expected_numa_nodes=args.expected_numa_nodes,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.stable_timing_protocol_metadata = controlled_protocol_metadata(
        placement=placement,
        expected_cpu_affinity=args.expected_cpu_affinity,
        expected_numa_nodes=args.expected_numa_nodes,
        max_over_median_limit=args.stability_max_over_median_limit,
        median_over_min_limit=args.stability_median_over_min_limit,
    )


if __name__ == "__main__":
    parsed = parse_args()
    configure_stable_timing_protocol(parsed)
    configure_runtime_import_root(parsed.easygraph_repo)
    if parsed.watchdog:
        raise SystemExit(watchdog_main(parsed.watchdog_pid))
    if parsed.worker:
        worker(parsed)
    else:
        coordinator(parsed)
