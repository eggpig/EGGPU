#!/usr/bin/env python3
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import threading
import time

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from baseline_versions import collect_python_baseline_versions
from child_process_memory_monitor import process_tree_pids_and_rss, resolve_host_pid
from measurement_schema import describe_metric
from nxcugraph_device_timer import NxCugraphDeviceTimer

try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
except Exception:
    pynvml = None

_NVML_INITIALIZED = False
TRUE_VALUES = {"1", "TRUE", "ON", "YES"}
FUNCTION_ORDER = (
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
PATH_SOURCE_FUNCTIONS = {"BFS", "Dijkstra", "BellmanFord", "SSSP"}
STRUCTURAL_HOLE_FUNCTIONS = {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}
NX_CUGRAPH_DEVICE_BOUNDARIES = {
    "PageRank": ("pagerank",),
    "LCC": ("triangle_count",),
    "WCC": ("weakly_connected_components",),
    "BFS": ("bfs",),
    "Dijkstra": ("sssp",),
    "BellmanFord": ("sssp",),
    "SSSP": ("sssp",),
    "KCore": ("core_number",),
}
LEGACY_FUNCTION_ALIASES = {"CC": ("WCC", "SCC")}
STRICT_VALIDATION = os.environ.get("EGGPU_STRICT_VALIDATION", "").strip().upper() in TRUE_VALUES
DETAIL_DIR = os.environ.get("EGGPU_VALIDATION_DETAIL_DIR", "").strip()
MEASUREMENT_MODE = os.environ.get("EGGPU_MEASUREMENT_MODE", "combined").strip().lower()
if MEASUREMENT_MODE not in {"timing", "memory", "combined"}:
    MEASUREMENT_MODE = "combined"


def records_timing_metrics():
    return MEASUREMENT_MODE in {"timing", "combined"}


def records_memory_metrics():
    return MEASUREMENT_MODE in {"memory", "combined"}


def validate_pagerank_result(ranks, n):
    """Fail closed when a PageRank public result violates its core contract."""

    if len(ranks) != n:
        raise RuntimeError(
            f"PageRank result cardinality differs: {len(ranks)} != {n}"
        )
    if isinstance(ranks, Mapping):
        values = [float(value) for value in ranks.values()]
    else:
        try:
            values = [float(value) for value in ranks]
        except TypeError as exc:
            raise TypeError(
                f"unsupported pagerank result type: {type(ranks).__name__}"
            ) from exc
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("PageRank result contains a non-finite value")
    if any(value < -1.0e-12 for value in values):
        raise RuntimeError("PageRank result contains a negative score")
    rank_sum = math.fsum(values)
    if not math.isclose(rank_sum, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-6):
        raise RuntimeError(f"PageRank result does not sum to one: {rank_sum}")
    return rank_sum


def internal_memory_monitor_enabled():
    return os.environ.get("EGGPU_INTERNAL_MEMORY_MONITOR", "TRUE").strip().upper() in TRUE_VALUES


def _ensure_nvml():
    global _NVML_INITIALIZED
    if pynvml is None:
        return False
    if _NVML_INITIALIZED:
        return True
    try:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True
        return True
    except Exception:
        return False


def _resolve_monitor_gpu_index():
    env_idx = os.environ.get("EGGPU_MONITOR_GPU_INDEX", "").strip()
    if env_idx.isdigit():
        return int(env_idx)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        tok = cvd.split(",")[0].strip()
        if tok.isdigit():
            return int(tok)
    return 0


def _nvml_compute_processes(handle):
    if pynvml is None:
        return []
    fn_names = (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    )
    for name in fn_names:
        fn = getattr(pynvml, name, None)
        if fn is None:
            continue
        try:
            procs = fn(handle)
            return procs if procs is not None else []
        except Exception:
            continue
    return []


def _safe_used_gpu_memory_bytes(proc_info):
    try:
        used = int(getattr(proc_info, "usedGpuMemory", 0))
    except Exception:
        return 0
    # Some NVML versions may return "not available" as a huge sentinel value.
    if used < 0 or used >= (1 << 62):
        return 0
    return used


def visibility_marker_adjust_mb():
    if os.environ.get("EGGPU_GPU_VISIBILITY_MARKER", "").strip().upper() not in TRUE_VALUES:
        return 0.0
    raw = os.environ.get("EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB", "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, value)


def visibility_marker_local_allocation_mb():
    """Return the marker allocation only when this process owns it."""

    if os.environ.get("EGGPU_GPU_VISIBILITY_MARKER", "").strip().upper() not in TRUE_VALUES:
        return 0.0
    try:
        owner_pid = int(os.environ.get("EGGPU_GPU_VISIBILITY_MARKER_OWNER_PID", "-1"))
    except ValueError:
        return 0.0
    if owner_pid != os.getpid():
        return 0.0
    try:
        return max(
            0.0,
            float(os.environ.get("EGGPU_GPU_VISIBILITY_MARKER_ALLOCATED_MB", "0")),
        )
    except ValueError:
        return 0.0


def load_graph(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#%/c":
                continue
            p = s.split()
            if len(p) < 2:
                continue
            try:
                u, v = int(p[0]), int(p[1])
            except ValueError:
                continue
            rows.append((u, v))
    if not rows:
        raise SystemExit(f"empty/invalid graph: {path}")

    raw = pd.DataFrame(rows, columns=["src", "dst"])

    uniq = pd.Index(
        pd.Categorical(pd.concat([raw["src"], raw["dst"]], ignore_index=True)).categories
    )
    remap = {int(node): index for index, node in enumerate(uniq)}
    directed = raw.copy()
    directed["src"] = directed["src"].map(remap).astype("int32")
    directed["dst"] = directed["dst"].map(remap).astype("int32")
    directed = (
        directed[directed["src"] != directed["dst"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    lo = directed[["src", "dst"]].min(axis=1)
    hi = directed[["src", "dst"]].max(axis=1)
    undirected = (
        pd.DataFrame({"src": lo, "dst": hi}).drop_duplicates().reset_index(drop=True)
    )
    n = int(uniq.size)
    undirected["weight"] = (
        1
        + (undirected["src"].astype("int64") * undirected["dst"].astype("int64"))
        % max(1, n)
    ).astype("int32")

    normalized = (n, directed, undirected)
    return {
        "clean": normalized,
        "all_vertices": normalized,
    }


def sync_gpu():
    allow = os.environ.get("EGGPU_ALLOW_CUDA_SYNC", "").strip().upper() in {
        "1",
        "TRUE",
        "ON",
        "YES",
    }
    if not allow and os.environ.get("EASYGRAPH_ENABLE_GPU", "").strip().upper() not in {
        "1",
        "TRUE",
        "ON",
        "YES",
    }:
        return
    try:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def barrier(cooldown):
    sync_gpu()
    gc.collect()
    if cooldown > 0:
        time.sleep(cooldown)
    sync_gpu()


class PeakMemoryMonitor:
    def __init__(self, poll_seconds=0.01):
        self.poll_seconds = max(0.002, float(poll_seconds))
        self._stop = threading.Event()
        self._thread = None
        self._proc = None
        self._gpu_handle = None
        self._gpu_index = None
        self._started_at = None
        self.start_rss_bytes = None
        self.peak_rss_bytes = 0
        self.num_rss_samples = 0
        self.peak_gpu_bytes = 0
        self.sum_gpu_bytes = 0
        self.num_gpu_samples = 0
        self.start_gpu_bytes = None
        self.peak_gpu_delta_bytes = 0
        self.sum_gpu_delta_bytes = 0
        self.num_gpu_delta_samples = 0
        self.peak_gpu_proc_bytes = 0
        self.sum_gpu_proc_bytes = 0
        self.num_gpu_proc_samples = 0
        self.start_gpu_proc_bytes = None
        self.peak_gpu_proc_delta_bytes = 0
        self.sum_gpu_proc_delta_bytes = 0
        self.num_gpu_proc_delta_samples = 0
        self._root_pid = resolve_host_pid(os.getpid())
        if psutil is not None:
            try:
                self._proc = psutil.Process(self._root_pid)
                self.start_rss_bytes = int(self._proc.memory_info().rss)
                self.peak_rss_bytes = self.start_rss_bytes
            except Exception:
                self._proc = None
        if self.start_rss_bytes is None:
            _pids, rss = process_tree_pids_and_rss(self._root_pid)
            self.start_rss_bytes = int(rss)
            self.peak_rss_bytes = self.start_rss_bytes
        if _ensure_nvml():
            try:
                self._gpu_index = _resolve_monitor_gpu_index()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            except Exception:
                self._gpu_handle = None
                self._gpu_index = None

    def _process_tree_pids(self):
        pids, _rss = process_tree_pids_and_rss(self._root_pid)
        return pids

    def _sample_once(self):
        _pids, rss = process_tree_pids_and_rss(self._root_pid)
        self.num_rss_samples += 1
        if self.start_rss_bytes is None:
            self.start_rss_bytes = rss
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        if self._gpu_handle is not None:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                used = int(info.used)
                if self.start_gpu_bytes is None:
                    self.start_gpu_bytes = used
                if used > self.peak_gpu_bytes:
                    self.peak_gpu_bytes = used
                self.sum_gpu_bytes += used
                self.num_gpu_samples += 1
                delta = used - self.start_gpu_bytes
                if delta < 0:
                    delta = 0
                if delta > self.peak_gpu_delta_bytes:
                    self.peak_gpu_delta_bytes = delta
                self.sum_gpu_delta_bytes += delta
                self.num_gpu_delta_samples += 1
            except Exception:
                pass
            try:
                pids = self._process_tree_pids()
                proc_used = 0
                for pinfo in _nvml_compute_processes(self._gpu_handle):
                    try:
                        pid = int(getattr(pinfo, "pid", -1))
                    except Exception:
                        pid = -1
                    if pid in pids:
                        proc_used += _safe_used_gpu_memory_bytes(pinfo)
                if self.start_gpu_proc_bytes is None:
                    self.start_gpu_proc_bytes = proc_used
                if proc_used > self.peak_gpu_proc_bytes:
                    self.peak_gpu_proc_bytes = proc_used
                self.sum_gpu_proc_bytes += proc_used
                self.num_gpu_proc_samples += 1
                proc_delta = proc_used - self.start_gpu_proc_bytes
                if proc_delta < 0:
                    proc_delta = 0
                if proc_delta > self.peak_gpu_proc_delta_bytes:
                    self.peak_gpu_proc_delta_bytes = proc_delta
                self.sum_gpu_proc_delta_bytes += proc_delta
                self.num_gpu_proc_delta_samples += 1
            except Exception:
                pass

    def _run(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.poll_seconds)

    def start(self):
        self._started_at = time.perf_counter()
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.05, 2 * self.poll_seconds))
        self._sample_once()
        monitor_window_seconds = (
            time.perf_counter() - self._started_at if self._started_at is not None else None
        )
        rss_start_mb = None
        rss_peak_mb = None
        rss_peak_delta_mb = None
        gpu_peak_mb = None
        gpu_avg_mb = None
        gpu_start_mb = None
        gpu_peak_delta_mb = None
        gpu_avg_delta_mb = None
        gpu_proc_peak_mb = None
        gpu_proc_avg_mb = None
        gpu_proc_start_mb = None
        gpu_proc_peak_delta_mb = None
        gpu_proc_avg_delta_mb = None
        if self.start_rss_bytes is not None:
            rss_start_mb = self.start_rss_bytes / (1024.0 * 1024.0)
        if self.peak_rss_bytes > 0:
            rss_peak_mb = self.peak_rss_bytes / (1024.0 * 1024.0)
        if self.start_rss_bytes is not None and self.peak_rss_bytes > 0:
            rss_peak_delta_mb = max(0, self.peak_rss_bytes - self.start_rss_bytes) / (
                1024.0 * 1024.0
            )
        marker_adjust_mb = visibility_marker_adjust_mb()
        raw_gpu_start_mb = None
        raw_gpu_proc_start_mb = None
        if self.num_gpu_samples > 0:
            gpu_peak_mb = self.peak_gpu_bytes / (1024.0 * 1024.0)
            gpu_avg_mb = (self.sum_gpu_bytes / self.num_gpu_samples) / (1024.0 * 1024.0)
            if self.start_gpu_bytes is not None:
                raw_gpu_start_mb = self.start_gpu_bytes / (1024.0 * 1024.0)
                gpu_start_mb = raw_gpu_start_mb
            if marker_adjust_mb > 0:
                gpu_peak_mb = max(0.0, gpu_peak_mb - marker_adjust_mb)
                gpu_avg_mb = max(0.0, gpu_avg_mb - marker_adjust_mb)
                if gpu_start_mb is not None:
                    gpu_start_mb = max(0.0, gpu_start_mb - marker_adjust_mb)
        if self.num_gpu_delta_samples > 0:
            gpu_peak_delta_mb = self.peak_gpu_delta_bytes / (1024.0 * 1024.0)
            gpu_avg_delta_mb = (self.sum_gpu_delta_bytes / self.num_gpu_delta_samples) / (1024.0 * 1024.0)
        if self.num_gpu_proc_samples > 0:
            gpu_proc_peak_mb = self.peak_gpu_proc_bytes / (1024.0 * 1024.0)
            gpu_proc_avg_mb = (self.sum_gpu_proc_bytes / self.num_gpu_proc_samples) / (1024.0 * 1024.0)
            if self.start_gpu_proc_bytes is not None:
                raw_gpu_proc_start_mb = self.start_gpu_proc_bytes / (1024.0 * 1024.0)
                gpu_proc_start_mb = raw_gpu_proc_start_mb
            local_marker_mb = visibility_marker_local_allocation_mb()
            if local_marker_mb > 0:
                gpu_proc_peak_mb = max(0.0, gpu_proc_peak_mb - local_marker_mb)
                gpu_proc_avg_mb = max(0.0, gpu_proc_avg_mb - local_marker_mb)
                if gpu_proc_start_mb is not None:
                    gpu_proc_start_mb = max(0.0, gpu_proc_start_mb - local_marker_mb)
        if self.num_gpu_proc_delta_samples > 0:
            gpu_proc_peak_delta_mb = self.peak_gpu_proc_delta_bytes / (1024.0 * 1024.0)
            gpu_proc_avg_delta_mb = (self.sum_gpu_proc_delta_bytes / self.num_gpu_proc_delta_samples) / (1024.0 * 1024.0)
        gpu_non_process_start_mb = None
        if raw_gpu_start_mb is not None and raw_gpu_proc_start_mb is not None:
            gpu_non_process_start_mb = max(
                0.0, raw_gpu_start_mb - raw_gpu_proc_start_mb
            )
            if visibility_marker_local_allocation_mb() <= 0:
                gpu_non_process_start_mb = max(
                    0.0, gpu_non_process_start_mb - marker_adjust_mb
                )
        return {
            "rss_mb": rss_peak_mb,
            "rss_start_mb": rss_start_mb,
            "rss_peak_delta_mb": rss_peak_delta_mb,
            "gpu_peak_mb": gpu_peak_mb,
            "gpu_avg_mb": gpu_avg_mb,
            "gpu_start_mb": gpu_start_mb,
            "gpu_peak_delta_mb": gpu_peak_delta_mb,
            "gpu_avg_delta_mb": gpu_avg_delta_mb,
            "gpu_proc_peak_mb": gpu_proc_peak_mb,
            "gpu_proc_avg_mb": gpu_proc_avg_mb,
            "gpu_proc_start_mb": gpu_proc_start_mb,
            "gpu_proc_peak_delta_mb": gpu_proc_peak_delta_mb,
            "gpu_proc_avg_delta_mb": gpu_proc_avg_delta_mb,
            "gpu_non_process_start_mb": gpu_non_process_start_mb,
            "monitor_rss_samples": self.num_rss_samples,
            "monitor_gpu_samples": self.num_gpu_samples,
            "monitor_gpu_proc_samples": self.num_gpu_proc_samples,
            "monitor_poll_ms": self.poll_seconds * 1000.0,
            "monitor_window_seconds": monitor_window_seconds,
            "gpu_index": self._gpu_index,
        }


def timed_algorithm(callable_obj, sync_after=True):
    poll_ms = max(2.0, float(os.environ.get("EGGPU_MEMORY_POLL_MS", "2")))
    memory_probe_enabled = records_memory_metrics()
    monitor = (
        PeakMemoryMonitor(poll_seconds=poll_ms / 1000.0).start()
        if memory_probe_enabled and internal_memory_monitor_enabled()
        else None
    )
    minimum_probe_seconds = max(
        0.0, float(os.environ.get("EGGPU_MEMORY_PROBE_MIN_SECONDS", "0.05"))
    )
    maximum_probe_calls = max(
        1, int(os.environ.get("EGGPU_MEMORY_PROBE_MAX_CALLS", "128"))
    )
    t0 = time.perf_counter()
    probe_calls = 0
    while True:
        out = callable_obj()
        probe_calls += 1
        if sync_after:
            sync_gpu()
        if not memory_probe_enabled:
            break
        if time.perf_counter() - t0 >= minimum_probe_seconds:
            break
        if probe_calls >= maximum_probe_calls:
            break
    elapsed = time.perf_counter() - t0
    mem = monitor.stop() if monitor is not None else ({} if memory_probe_enabled else None)
    if mem is not None:
        mem["probe_calls"] = probe_calls
        mem["probe_window_seconds"] = elapsed
    return out, elapsed, mem


def emit(
    backend,
    function,
    metric,
    seconds,
    status="ok",
    correctness="",
    notes="",
    provenance=None,
):
    if backend == "EGGPU":
        protocol = os.environ.get("EGGPU_EXECUTION_PROTOCOL", "steady-state")
        protocol_note = f"eggpu_execution_protocol={protocol}"
        notes = f"{notes}; {protocol_note}" if notes else protocol_note
    descriptor = describe_metric(metric, baseline=backend)
    row = {
        "backend": backend,
        "function": function,
        "metric": metric,
        "seconds": None if seconds is None else float(seconds),
        "value": None if seconds is None else float(seconds),
        **descriptor,
        "status": status,
        "correctness": correctness,
        "notes": notes,
    }
    if provenance is not None:
        row["timing_provenance"] = provenance
    print("RESULT_JSON " + json.dumps(row, sort_keys=True), flush=True)


def emit_metrics(
    backend,
    function,
    build_s,
    algo_s,
    kernel_s,
    status="ok",
    correctness="",
    notes="",
    memory=None,
    provenance=None,
):
    if records_timing_metrics():
        emit(
            backend,
            function,
            "build",
            build_s,
            status=status,
            correctness=correctness,
            notes=notes,
            provenance=provenance,
        )
        emit(
            backend,
            function,
            "e2e",
            algo_s,
            status=status,
            correctness=correctness,
            notes=notes,
            provenance=provenance,
        )
        emit(
            backend,
            function,
            "kernel",
            kernel_s,
            status=status,
            correctness=correctness,
            notes=notes,
            provenance=provenance,
        )
    emitted_memory = False
    if records_memory_metrics() and isinstance(memory, dict):
        rss_mb = memory.get("rss_mb")
        if rss_mb is not None:
            emitted_memory = True
            emit(
                backend,
                function,
                "memory_peak_rss_mb",
                rss_mb,
                status=status,
                correctness=correctness,
                notes=(notes + "; peak RSS during algorithm window" if notes else "peak RSS during algorithm window"),
            )
        gpu_peak_mb = memory.get("gpu_peak_mb")
        if gpu_peak_mb is not None:
            emitted_memory = True
            emit(
                backend,
                function,
                "memory_peak_gpu_mb",
                gpu_peak_mb,
                status=status,
                correctness=correctness,
                notes=(notes + "; peak GPU memory during algorithm window" if notes else "peak GPU memory during algorithm window"),
            )
        gpu_avg_mb = memory.get("gpu_avg_mb")
        if gpu_avg_mb is not None:
            emitted_memory = True
            emit(
                backend,
                function,
                "memory_avg_gpu_mb",
                gpu_avg_mb,
                status=status,
                correctness=correctness,
                notes=(notes + "; average GPU memory during algorithm window" if notes else "average GPU memory during algorithm window"),
            )
        gpu_peak_delta_mb = memory.get("gpu_peak_delta_mb")
        if gpu_peak_delta_mb is not None:
            emitted_memory = True
            emit(
                backend,
                function,
                "memory_peak_gpu_delta_mb",
                gpu_peak_delta_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; peak GPU memory delta from algorithm-window start"
                    if notes
                    else "peak GPU memory delta from algorithm-window start"
                ),
            )
        gpu_avg_delta_mb = memory.get("gpu_avg_delta_mb")
        if gpu_avg_delta_mb is not None:
            emitted_memory = True
            emit(
                backend,
                function,
                "memory_avg_gpu_delta_mb",
                gpu_avg_delta_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; average GPU memory delta from algorithm-window start"
                    if notes
                    else "average GPU memory delta from algorithm-window start"
                ),
            )
        gpu_proc_peak_mb = memory.get("gpu_proc_peak_mb")
        if gpu_proc_peak_mb is not None:
            emitted_memory = True
            emit(
                backend,
                function,
                "memory_peak_gpu_proc_mb",
                gpu_proc_peak_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; peak GPU memory of benchmark process tree during algorithm window"
                    if notes
                    else "peak GPU memory of benchmark process tree during algorithm window"
                ),
            )
        gpu_proc_avg_mb = memory.get("gpu_proc_avg_mb")
        if gpu_proc_avg_mb is not None:
            emitted_memory = True
            emit(
                backend,
                function,
                "memory_avg_gpu_proc_mb",
                gpu_proc_avg_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; average GPU memory of benchmark process tree during algorithm window"
                    if notes
                    else "average GPU memory of benchmark process tree during algorithm window"
                ),
            )
        gpu_proc_peak_delta_mb = memory.get("gpu_proc_peak_delta_mb")
        if gpu_proc_peak_delta_mb is not None:
            emitted_memory = True
            emit(
                backend,
                function,
                "memory_peak_gpu_proc_delta_mb",
                gpu_proc_peak_delta_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; peak GPU memory delta of benchmark process tree from algorithm-window start"
                    if notes
                    else "peak GPU memory delta of benchmark process tree from algorithm-window start"
                ),
            )
        gpu_proc_avg_delta_mb = memory.get("gpu_proc_avg_delta_mb")
        if gpu_proc_avg_delta_mb is not None:
            emitted_memory = True
            emit(
                backend,
                function,
                "memory_avg_gpu_proc_delta_mb",
                gpu_proc_avg_delta_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; average GPU memory delta of benchmark process tree from algorithm-window start"
                    if notes
                    else "average GPU memory delta of benchmark process tree from algorithm-window start"
                ),
            )
        supplemental_memory_metrics = (
            ("rss_start_mb", "memory_start_rss_mb", "process-tree RSS at algorithm-window start"),
            ("rss_peak_delta_mb", "memory_peak_rss_delta_mb", "process-tree peak RSS delta from algorithm-window start"),
            ("gpu_start_mb", "memory_start_gpu_mb", "whole-device GPU memory at algorithm-window start; diagnostic only"),
            ("gpu_proc_start_mb", "memory_start_gpu_proc_mb", "benchmark process-tree GPU memory at algorithm-window start"),
            ("gpu_non_process_start_mb", "memory_start_gpu_non_process_mb", "derived non-benchmark GPU memory at algorithm-window start"),
            ("monitor_rss_samples", "memory_monitor_rss_samples", "RSS sample count in algorithm window"),
            ("monitor_gpu_samples", "memory_monitor_gpu_samples", "whole-device NVML sample count in algorithm window"),
            ("monitor_gpu_proc_samples", "memory_monitor_gpu_proc_samples", "process-tree NVML sample count in algorithm window"),
            ("monitor_poll_ms", "memory_monitor_poll_ms", "configured memory-monitor polling interval"),
            ("monitor_window_seconds", "memory_monitor_window_seconds", "observed memory-monitor window"),
            ("probe_calls", "memory_probe_calls", "algorithm calls executed only in the independent memory-probe window"),
            ("probe_window_seconds", "memory_probe_window_seconds", "algorithm-call span of the independent memory-probe window"),
        )
        for key, metric, metric_note in supplemental_memory_metrics:
            value = memory.get(key)
            if value is None:
                continue
            emitted_memory = True
            emit(
                backend,
                function,
                metric,
                value,
                status=status,
                correctness=correctness,
                notes=(notes + "; " + metric_note if notes else metric_note),
            )
    if records_memory_metrics() and not emitted_memory and status != "ok":
        emit(
            backend,
            function,
            "memory_peak_rss_mb",
            None,
            status=status,
            correctness=correctness,
            notes=notes or "memory measurement unavailable",
        )


def emit_skip(backend, function, notes):
    emit_metrics(backend, function, None, None, None, status="skipped", notes=notes)


def emit_exception(backend, function, exc):
    notes = str(exc)
    status = "skipped" if "not implemented by 'cugraph' backend" in notes else "failed"
    emit_metrics(backend, function, None, None, None, status=status, notes=notes)


def is_nx_cugraph_not_implemented(exc):
    return "not implemented by 'cugraph' backend" in str(exc)


def concise_error(exc, limit=220):
    text = str(exc).strip()
    head = text.splitlines()[0] if text else ""
    return head if len(head) <= limit else (head[: limit - 3] + "...")


def warmup_call(callable_obj, warmup):
    for _ in range(max(0, int(warmup))):
        try:
            callable_obj()
        except Exception:
            break


def pick_sources(n, k):
    if n <= 0:
        return []
    k = max(1, int(k))
    if k >= n:
        return list(range(n))
    step = max(1, n // k)
    out = list(range(0, n, step))[:k]
    if len(out) < k:
        tail = n - 1
        while len(out) < k and tail >= 0:
            if tail not in out:
                out.append(tail)
            tail -= 1
    return out


def pick_optional_sources(n, k):
    k = int(k or 0)
    if k <= 0:
        return None
    return pick_sources(n, k)


def call_with_supported_kwargs(func, kwargs):
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return func(**kwargs), kwargs
    filtered = {k: v for k, v in kwargs.items() if k in params}
    if "G" in filtered:
        g = filtered["G"]
        rest = {k: v for k, v in filtered.items() if k != "G"}
        return func(g, **rest), filtered
    return func(**filtered), filtered


def build_igraph(n, edges, directed, weighted=False):
    import igraph as ig

    pairs = list(edges[["src", "dst"]].itertuples(index=False, name=None))
    g = ig.Graph(n=n, edges=[(int(u), int(v)) for u, v in pairs], directed=directed)
    if weighted:
        g.es["weight"] = [int(w) for w in edges["weight"].tolist()]
    return g


def build_networkx(n, edges, directed, weighted=False):
    import networkx as nx

    g = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(range(n))
    if weighted:
        for u, v, w in edges[["src", "dst", "weight"]].itertuples(index=False, name=None):
            g.add_edge(int(u), int(v), weight=int(w))
    else:
        g.add_edges_from((int(u), int(v)) for u, v in edges[["src", "dst"]].itertuples(index=False, name=None))
    return g


def build_nxcugraph_native(n, edges, directed, weighted=False):
    """Build the callable native device graph used by nx-cugraph.

    ``views`` stores each undirected edge once.  nx-cugraph's COO graph
    representation stores adjacency entries, so an undirected input is
    expanded in both directions before construction.
    """

    import cupy as cp
    import nx_cugraph as nxcg
    import numpy as np

    src_host = edges["src"].to_numpy(dtype=np.int32, copy=False)
    dst_host = edges["dst"].to_numpy(dtype=np.int32, copy=False)
    if directed:
        src = cp.asarray(src_host)
        dst = cp.asarray(dst_host)
    else:
        src = cp.asarray(np.concatenate((src_host, dst_host)))
        dst = cp.asarray(np.concatenate((dst_host, src_host)))
    edge_values = None
    if weighted:
        weight_host = edges["weight"].to_numpy(dtype=np.float64, copy=False)
        if not directed:
            weight_host = np.concatenate((weight_host, weight_host))
        edge_values = {"weight": cp.asarray(weight_host)}
    graph_class = nxcg.CudaDiGraph if directed else nxcg.CudaGraph
    graph = graph_class.from_coo(
        int(n),
        src,
        dst,
        edge_values=edge_values,
        use_compat_graph=False,
    )
    cp.cuda.runtime.deviceSynchronize()
    return graph


def build_easygraph(n, edges, directed, weighted=False):
    import easygraph as eg

    g = eg.DiGraph() if directed else eg.Graph()
    g.add_nodes_from(range(n))
    if weighted:
        for u, v, w in edges[["src", "dst", "weight"]].itertuples(index=False, name=None):
            g.add_edge(int(u), int(v), weight=int(w))
    else:
        g.add_edges_from((int(u), int(v)) for u, v in edges[["src", "dst"]].itertuples(index=False, name=None))
    return g


def component_semantics(func, graph_type, views):
    """Return a normalized component benchmark plan for WCC/SCC."""
    if func not in {"WCC", "SCC"}:
        raise ValueError(f"not a component function: {func}")
    if func == "SCC" and graph_type == "directed":
        n, directed_edges, _ = views["all_vertices"]
        return {
            "n": n,
            "edges": directed_edges,
            "build_directed": True,
            "igraph_mode": "strong",
            "nx_kind": "scc",
            "eg_kind": "scc",
            "cugraph_directed": True,
            "note": "SCC semantics",
            "kernel_key": "scc",
        }

    n, _, undirected_edges = views["all_vertices"]
    note = "WCC semantics; undirected projection" if graph_type == "directed" else "WCC semantics"
    if func == "SCC" and graph_type != "directed":
        note = "undirected graph: SCC equals WCC"
    return {
        "n": n,
        "edges": undirected_edges,
        "build_directed": False,
        "igraph_mode": "weak",
        "nx_kind": "wcc",
        "eg_kind": "wcc",
        "cugraph_directed": False,
        "note": note,
        "kernel_key": "cc",
    }


def easygraph_edges(graph):
    edges = graph.edges
    return edges() if callable(edges) else edges


def deterministic_weighted_edges(n, edges):
    if len(edges) == 0:
        out = edges[["src", "dst"]].copy()
        out["weight"] = pd.Series(dtype="int32")
        return out
    out = edges[["src", "dst"]].copy()
    mod = max(1, int(n))
    out["weight"] = (
        1
        + (
            out["src"].astype("int64") * out["dst"].astype("int64")
        ) % mod
    ).astype("int32")
    return out


def path_benchmark_plan(func, graph_type, views, source_count):
    n, directed_edges, undirected_edges = views["clean"]
    base_edges = directed_edges if graph_type == "directed" else undirected_edges
    weighted = func in {"Dijkstra", "BellmanFord", "SSSP"}
    edges = deterministic_weighted_edges(n, base_edges) if weighted else base_edges
    requested_sources = 1 if func == "Dijkstra" else source_count
    sources = pick_sources(n, requested_sources)
    if func == "BFS":
        note = f"unweighted shortest paths; sources={len(sources)}"
        detail_name = "BFS"
        kernel_key = "bfs"
    elif func == "BellmanFord":
        note = f"Bellman-Ford shortest paths on deterministic nonnegative weights; sources={len(sources)}"
        detail_name = "BellmanFord"
        kernel_key = "bellman_ford"
    elif func == "Dijkstra":
        note = "single-source Dijkstra on deterministic nonnegative weights; sources=1"
        detail_name = "Dijkstra"
        kernel_key = "dijkstra"
    else:
        note = f"batched SSSP on deterministic nonnegative weights; sources={len(sources)}"
        detail_name = "SSSP"
        kernel_key = "sssp"
    return {
        "n": n,
        "edges": edges,
        "weighted": weighted,
        "directed": graph_type == "directed",
        "sources": sources,
        "note": note,
        "detail_name": detail_name,
        "kernel_key": kernel_key,
    }


def structural_hole_plan(func, graph_type, views):
    n, directed_edges, undirected_edges = views["clean"]
    base_edges = directed_edges if graph_type == "directed" else undirected_edges
    key = {
        "EffectiveSize": "effective_size",
        "Efficiency": "efficiency",
        "Constraint": "constraint",
        "Hierarchy": "hierarchy",
    }[func]
    return {
        "n": n,
        "edges": base_edges,
        "directed": graph_type == "directed",
        "kernel_key": key,
        "note": "Burt structural-hole metric; unweighted graph",
    }


def summarize_numeric_mapping(values):
    if hasattr(values, "tolist") and not isinstance(values, Mapping):
        values = values.tolist()
    if isinstance(values, Mapping):
        seq = list(values.values())
    elif isinstance(values, (list, tuple)):
        seq = list(values)
    else:
        seq = []
    clean = []
    for v in seq:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            clean.append(x)
    total = float(sum(clean)) if clean else 0.0
    mean = total / len(clean) if clean else 0.0
    return len(seq), total, mean


def summarize_sssp_result(result):
    reachable = 0
    checksum = 0.0
    if isinstance(result, Mapping) and "values" in result:
        result = result.get("values")
    if isinstance(result, Mapping):
        if result and all(isinstance(v, Mapping) for v in result.values()):
            for dist_map in result.values():
                for dv in dist_map.values():
                    try:
                        x = float(dv)
                    except Exception:
                        continue
                    if math.isfinite(x) and abs(x) < 1.0e30:
                        reachable += 1
                        checksum += x
            return reachable, checksum
        for dv in result.values():
            try:
                x = float(dv)
            except Exception:
                continue
            if math.isfinite(x) and abs(x) < 1.0e30:
                reachable += 1
                checksum += x
        return reachable, checksum
    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, (list, tuple)):
        for row in result:
            if hasattr(row, "tolist"):
                row = row.tolist()
            if isinstance(row, (list, tuple)):
                for dv in row:
                    try:
                        x = float(dv)
                    except Exception:
                        continue
                    if math.isfinite(x) and abs(x) < 1.0e30:
                        reachable += 1
                        checksum += x
            else:
                try:
                    x = float(row)
                except Exception:
                    continue
                if math.isfinite(x) and abs(x) < 1.0e30:
                    reachable += 1
                    checksum += x
    return reachable, checksum


def _detail_path(backend, function, suffix="npz"):
    if not STRICT_VALIDATION or not DETAIL_DIR:
        return None
    safe_backend = str(backend).replace("/", "_")
    safe_function = str(function).replace("/", "_")
    out_dir = Path(DETAIL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{safe_backend}_{safe_function}.{suffix}"


def _array_digest(arr):
    import numpy as np

    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(tuple(arr.shape)).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _numeric_vector_from_mapping(values, n, dtype="float64", default=0.0):
    import numpy as np

    arr = np.full(int(n), default, dtype=dtype)
    if hasattr(values, "tolist") and not isinstance(values, Mapping):
        values = values.tolist()
    if isinstance(values, Mapping):
        for k, v in values.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if 0 <= idx < n:
                arr[idx] = v
    elif isinstance(values, (list, tuple)):
        upto = min(int(n), len(values))
        if upto:
            arr[:upto] = values[:upto]
    return arr


def write_vector_detail(backend, function, values, n, dtype="float64", default=0.0):
    import numpy as np

    path = _detail_path(backend, function)
    if path is None:
        return ""
    arr = _numeric_vector_from_mapping(values, n, dtype=dtype, default=default)
    np.savez_compressed(path, kind="vector", values=arr)
    return f", detail={path}, detail_kind=vector, detail_sha={_array_digest(arr)}"


def write_source_vector_detail(backend, function, sources, values, n, dtype="float64"):
    import numpy as np

    path = _detail_path(backend, function)
    if path is None:
        return ""
    src = np.asarray([int(s) for s in sources], dtype=np.int64)
    vals = np.asarray([float(v) for v in values], dtype=dtype)
    np.savez_compressed(path, kind="source_vector", sources=src, values=vals, graph_n=int(n))
    digest_arr = np.column_stack((src, vals.astype(np.float64, copy=False))) if len(src) else vals
    return f", detail={path}, detail_kind=source_vector, detail_sha={_array_digest(digest_arr)}"


def write_cc_detail(backend, function, components, n):
    import numpy as np

    path = _detail_path(backend, function)
    if path is None:
        return ""
    labels = np.full(int(n), -1, dtype=np.int64)
    normalized_components = []
    for comp in components:
        vals = []
        try:
            iterator = list(comp)
        except Exception:
            iterator = []
        for node in iterator:
            try:
                idx = int(node)
            except Exception:
                continue
            if 0 <= idx < n:
                vals.append(idx)
        if vals:
            normalized_components.append(sorted(set(vals)))
    normalized_components.sort(key=lambda xs: (xs[0], len(xs)))
    for label, comp in enumerate(normalized_components):
        for idx in comp:
            labels[idx] = label
    next_label = len(normalized_components)
    for idx in range(int(n)):
        if labels[idx] < 0:
            labels[idx] = next_label
            next_label += 1
    np.savez_compressed(path, kind="cc_labels", values=labels)
    return f", detail={path}, detail_kind=cc_labels, detail_sha={_array_digest(labels)}"


def write_sssp_detail(backend, function, result, sources, n):
    import numpy as np

    path = _detail_path(backend, function)
    if path is None:
        return ""
    source_list = [int(s) for s in sources]
    arr = np.full((len(source_list), int(n)), np.inf, dtype=np.float64)
    source_pos = {s: i for i, s in enumerate(source_list)}

    def put(row, node, value):
        try:
            idx = int(node)
            val = float(value)
        except Exception:
            return
        if 0 <= idx < n and math.isfinite(val) and abs(val) < 1.0e30:
            arr[row, idx] = val

    if isinstance(result, Mapping) and "values" in result:
        result = result.get("values")

    if isinstance(result, Mapping):
        if result and all(isinstance(v, Mapping) for v in result.values()):
            for source, dist_map in result.items():
                try:
                    row = source_pos[int(source)]
                except Exception:
                    continue
                for node, value in dist_map.items():
                    put(row, node, value)
        else:
            row = 0
            for node, value in result.items():
                put(row, node, value)
    else:
        rows = result.tolist() if hasattr(result, "tolist") else list(result)
        for row, dist_row in enumerate(rows[: len(source_list)]):
            vals = dist_row.tolist() if hasattr(dist_row, "tolist") else list(dist_row)
            for idx, value in enumerate(vals[: int(n)]):
                put(row, idx, value)
    finite = np.isfinite(arr)
    digest_arr = np.where(finite, arr, -1.0)
    np.savez_compressed(path, kind="sssp", sources=np.asarray(source_list, dtype=np.int64), values=arr)
    return f", detail={path}, detail_kind=sssp, detail_sha={_array_digest(digest_arr)}"


def nx_cugraph_call(func, *args, **kwargs):
    """Return from the strict public call without extending its boundary."""

    kwargs["backend"] = "cugraph"
    return func(*args, **kwargs)


def configure_nx_cugraph_cuda_runtime():
    """Pin NVRTC and its headers before importing RAPIDS/CuPy modules."""

    import ctypes

    candidates = []
    configured = os.environ.get("NX_CUGRAPH_NVRTC_PATH", "").strip()
    if configured:
        candidates.append(Path(configured))
    candidates.append(
        Path(
            "/home/dataset-assist-0/einwang/conda_cache/conda_env/"
            "tongyideepresearch/lib/libnvrtc.so.13"
        )
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        root = str(candidate.parents[1])
        for key in (
            "EGGPU_CUDA_ROOT",
            "CUDA_PATH",
            "CUDA_HOME",
            "CUPY_CUDA_PATH",
            "CUDAToolkit_ROOT",
            "CONDA_PREFIX",
        ):
            os.environ[key] = root
        return {
            "nvrtc": str(candidate.resolve()),
            "cuda_root": root,
        }
    return {
        "nvrtc": "dynamic-loader-default",
        "cuda_root": os.environ.get("CUDA_PATH", ""),
    }


def nx_cugraph_expected_interval_count(function, source_count):
    if function in {"PageRank", "LCC", "WCC", "Dijkstra", "KCore"}:
        return 1
    return int(source_count)


def warm_nx_cugraph(callable_obj, device_timer, function, warmup, source_count):
    details = []
    for call_position in range(1, max(0, int(warmup)) + 1):
        device_timer.reset()
        started = time.perf_counter()
        result = device_timer.call(callable_obj)
        wall_seconds = time.perf_counter() - started
        device_seconds = device_timer.validate_against_public_wall(wall_seconds)
        provenance = device_timer.validate_provenance(
            expected_backends=NX_CUGRAPH_DEVICE_BOUNDARIES[function],
            expected_interval_count=nx_cugraph_expected_interval_count(
                function, source_count
            ),
        )
        details.append(
            {
                "call_position": call_position,
                "e2e_seconds": wall_seconds,
                "device_seconds": device_seconds,
                "device_timer": provenance,
            }
        )
        del result
    return details


def timed_nx_cugraph_algorithm(
    callable_obj,
    device_timer,
    function,
    source_count,
):
    """Measure one strict public call and its actual backend device interval."""

    device_timer.reset()
    result, wall_seconds, memory = timed_algorithm(
        lambda: device_timer.call(callable_obj),
        sync_after=False,
    )
    device_seconds = device_timer.validate_against_public_wall(wall_seconds)
    probe_calls = (
        int(memory.get("probe_calls", 1))
        if isinstance(memory, dict)
        else 1
    )
    timer_provenance = device_timer.validate_provenance(
        expected_backends=NX_CUGRAPH_DEVICE_BOUNDARIES[function],
        expected_interval_count=(
            nx_cugraph_expected_interval_count(function, source_count)
            * probe_calls
        ),
    )
    return result, wall_seconds, device_seconds, memory, {
        "timer": timer_provenance,
        "prepared_native_graph": True,
        "validation_outside_timer": True,
        "public_return_boundary": True,
    }


def bench_igraph(views, graph_type, skip_cpu, functions, pr_alpha, cooldown, warmup, sssp_sources, bc_sources, closeness_sources):
    backend = "igraph"
    if skip_cpu:
        for func in functions:
            emit_skip(backend, func, "CPU baseline skipped by size threshold")
        return

    try:
        import igraph as ig  # noqa: F401
    except Exception as e:
        for func in functions:
            emit_skip(backend, func, f"igraph import failed: {e}")
        return

    for func in functions:
        try:
            if func == "PageRank":
                n, directed_edges, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_igraph(n, directed_edges if graph_type == "directed" else undirected_edges, graph_type == "directed")
                t1 = time.perf_counter()
                warmup_call(lambda: g.pagerank(directed=(graph_type == "directed"), damping=pr_alpha), warmup)
                barrier(cooldown)
                ranks, algo_s, mem = timed_algorithm(lambda: g.pagerank(directed=(graph_type == "directed"), damping=pr_alpha), sync_after=False)
                emit_metrics(
                    backend,
                    "PageRank",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"sum={sum(ranks):.9g}" + write_vector_detail(backend, "PageRank", ranks, n),
                    notes=f"alpha={pr_alpha}; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "MST":
                n, _, undirected_edges = views["all_vertices"]
                t0 = time.perf_counter()
                g = build_igraph(n, undirected_edges, False, weighted=True)
                t1 = time.perf_counter()
                warmup_call(lambda: g.spanning_tree(weights=g.es["weight"], return_tree=True), warmup)
                barrier(cooldown)
                tree, algo_s, mem = timed_algorithm(lambda: g.spanning_tree(weights=g.es["weight"], return_tree=True), sync_after=False)
                weight = int(sum(tree.es["weight"])) if tree.ecount() else 0
                emit_metrics(
                    backend,
                    "MST",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"weight={weight}",
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "LCC":
                n, _, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_igraph(n, undirected_edges, False)
                t1 = time.perf_counter()
                warmup_call(lambda: g.transitivity_local_undirected(mode="zero"), warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: g.transitivity_local_undirected(mode="zero"), sync_after=False)
                finite = [v for v in vals if v is not None and not math.isnan(float(v))]
                corr = f"vertices={len(vals)}, mean={sum(finite)/len(finite):.9g}" if finite else f"vertices={len(vals)}"
                corr += write_vector_detail(backend, "LCC", vals, n)
                emit_metrics(
                    backend,
                    "LCC",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=corr,
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "Closeness":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                t0 = time.perf_counter()
                g = build_igraph(n, base_edges, graph_type == "directed")
                t1 = time.perf_counter()
                mode = "OUT" if graph_type == "directed" else "ALL"
                source_nodes = pick_optional_sources(n, closeness_sources)

                def run_closeness():
                    vertices = None if source_nodes is None else source_nodes
                    vals = g.closeness(vertices=vertices, mode=mode, weights=None, normalized=True)
                    reachable = g.neighborhood_size(vertices=vertices, order=max(0, n), mode=mode)
                    out = []
                    denom = max(1, n - 1)
                    for idx, v in enumerate(vals):
                        try:
                            x = float(v)
                        except Exception:
                            x = 0.0
                        if not math.isfinite(x):
                            x = 0.0
                        try:
                            wf = max(0, int(reachable[idx]) - 1) / float(denom)
                        except Exception:
                            wf = 0.0
                        x *= wf
                        out.append(x)
                    return out

                warmup_call(run_closeness, warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(run_closeness, sync_after=False)
                count, total, mean = summarize_numeric_mapping(vals)
                if source_nodes is None:
                    detail = write_vector_detail(backend, "Closeness", vals, n)
                    corr_prefix = f"nodes={count}"
                    note_prefix = "outgoing distance for directed graphs"
                else:
                    detail = write_source_vector_detail(backend, "Closeness", source_nodes, vals, n)
                    corr_prefix = f"sources={len(source_nodes)}, graph_nodes={n}"
                    note_prefix = (
                        "sampled-target exact closeness; deterministic evenly spaced sources; "
                        f"sources={len(source_nodes)}; outgoing distance for directed graphs"
                    )
                emit_metrics(
                    backend,
                    "Closeness",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"{corr_prefix}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=note_prefix
                    + "; igraph closeness mode="
                    + mode
                    + "; Wasserman-Faust disconnected correction applied; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in {"WCC", "SCC"}:
                plan = component_semantics(func, graph_type, views)
                n = plan["n"]
                t0 = time.perf_counter()
                g = build_igraph(n, plan["edges"], plan["build_directed"])
                t1 = time.perf_counter()
                warmup_call(lambda: g.connected_components(mode=plan["igraph_mode"]), warmup)
                barrier(cooldown)
                comps, algo_s, mem = timed_algorithm(
                    lambda: g.connected_components(mode=plan["igraph_mode"]),
                    sync_after=False,
                )
                sizes = comps.sizes()
                cc_detail = write_cc_detail(backend, func, comps, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"components={len(sizes)}" + cc_detail,
                    notes=plan["note"] + "; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in PATH_SOURCE_FUNCTIONS:
                plan = path_benchmark_plan(func, graph_type, views, sssp_sources)
                n = plan["n"]
                source_nodes = plan["sources"]
                if not source_nodes:
                    emit_skip(backend, func, "empty source set")
                    continue
                t0 = time.perf_counter()
                g = build_igraph(n, plan["edges"], plan["directed"], weighted=plan["weighted"])
                t1 = time.perf_counter()
                mode = "OUT" if graph_type == "directed" else "ALL"
                weights = "weight" if plan["weighted"] else None
                algorithm = "bellman_ford" if func == "BellmanFord" else ("dijkstra" if plan["weighted"] else "auto")
                warmup_call(lambda: g.distances(source=source_nodes, weights=weights, mode=mode, algorithm=algorithm), warmup)
                barrier(cooldown)
                dists, algo_s, mem = timed_algorithm(
                    lambda: g.distances(source=source_nodes, weights=weights, mode=mode, algorithm=algorithm),
                    sync_after=False,
                )
                reachable, checksum = summarize_sssp_result(dists)
                sssp_detail = write_sssp_detail(backend, func, dists, source_nodes, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"sources={len(source_nodes)}, reachable={reachable}, checksum={checksum:.9g}" + sssp_detail,
                    notes=plan["note"] + f"; igraph algorithm={algorithm}; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in STRUCTURAL_HOLE_FUNCTIONS:
                if func != "Constraint":
                    emit_skip(backend, func, "igraph has no native Burt effective-size/efficiency/hierarchy API aligned with EasyGraph")
                    continue
                plan = structural_hole_plan(func, graph_type, views)
                n = plan["n"]
                t0 = time.perf_counter()
                g = build_igraph(n, plan["edges"], plan["directed"])
                t1 = time.perf_counter()
                warmup_call(lambda: g.constraint(weights=None), warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: g.constraint(weights=None), sync_after=False)
                count, total, mean = summarize_numeric_mapping(vals)
                detail = write_vector_detail(backend, func, vals, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={count}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=plan["note"] + "; igraph Graph.constraint; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "KCore":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = undirected_edges
                t0 = time.perf_counter()
                g = build_igraph(n, base_edges, False)
                t1 = time.perf_counter()
                mode = "all"
                warmup_call(lambda: g.coreness(mode=mode), warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: g.coreness(mode=mode), sync_after=False)
                s = float(sum(vals)) if vals else 0.0
                mx = int(max(vals)) if vals else 0
                detail = write_vector_detail(backend, "KCore", vals, n, dtype="int64")
                emit_metrics(
                    backend,
                    "KCore",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={len(vals)}, sum={s:.9g}, max={mx}" + detail,
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "BC":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                source_nodes = pick_sources(n, bc_sources)
                if not source_nodes:
                    emit_skip(backend, "BC", "empty source set")
                    continue
                t0 = time.perf_counter()
                g = build_igraph(n, base_edges, graph_type == "directed")
                t1 = time.perf_counter()
                all_nodes = list(range(n))
                directed_flag = graph_type == "directed"
                warmup_call(
                    lambda: g.betweenness(
                        vertices=None,
                        directed=directed_flag,
                        weights=None,
                        sources=source_nodes,
                        targets=all_nodes,
                    ),
                    warmup,
                )
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(
                    lambda: g.betweenness(
                        vertices=None,
                        directed=directed_flag,
                        weights=None,
                        sources=source_nodes,
                        targets=all_nodes,
                    ),
                    sync_after=False,
                )
                total = float(sum(vals)) if vals else 0.0
                detail = write_vector_detail(backend, "BC", vals, n)
                emit_metrics(
                    backend,
                    "BC",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={len(vals)}, sum={total:.9g}" + detail,
                    notes=f"source-sampled exact mode; sources={len(source_nodes)}; normalized=False; cpu backend kernel=algorithm",
                    memory=mem,
                )
        except Exception as e:
            emit_exception(backend, func, e)


def bench_networkx(views, graph_type, skip_cpu, functions, pr_alpha, pr_max_iter, pr_tol, cooldown, warmup, sssp_sources, bc_sources, closeness_sources):
    backend = "networkx"
    if skip_cpu:
        for func in functions:
            emit_skip(backend, func, "CPU baseline skipped by explicit --skip-cpu")
        return

    try:
        import networkx as nx
    except Exception as e:
        for func in functions:
            emit_skip(backend, func, f"networkx import failed: {e}")
        return

    for func in functions:
        try:
            if func == "PageRank":
                n, directed_edges, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_networkx(n, directed_edges if graph_type == "directed" else undirected_edges, graph_type == "directed")
                t1 = time.perf_counter()
                warmup_call(lambda: nx.pagerank(g, alpha=pr_alpha, max_iter=pr_max_iter, tol=pr_tol), warmup)
                barrier(cooldown)
                ranks, algo_s, mem = timed_algorithm(
                    lambda: nx.pagerank(g, alpha=pr_alpha, max_iter=pr_max_iter, tol=pr_tol),
                    sync_after=False,
                )
                emit_metrics(
                    backend,
                    "PageRank",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"sum={sum(ranks.values()):.9g}" + write_vector_detail(backend, "PageRank", ranks, n),
                    notes=f"alpha={pr_alpha}, max_iter={pr_max_iter}, tol={pr_tol}; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "MST":
                n, _, undirected_edges = views["all_vertices"]
                t0 = time.perf_counter()
                g = build_networkx(n, undirected_edges, False, weighted=True)
                t1 = time.perf_counter()
                warmup_call(lambda: nx.minimum_spanning_tree(g, weight="weight"), warmup)
                barrier(cooldown)
                tree, algo_s, mem = timed_algorithm(lambda: nx.minimum_spanning_tree(g, weight="weight"), sync_after=False)
                weight = int(sum(data.get("weight", 1) for _, _, data in tree.edges(data=True)))
                emit_metrics(
                    backend,
                    "MST",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"weight={weight}",
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "LCC":
                n, _, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_networkx(n, undirected_edges, False)
                t1 = time.perf_counter()
                warmup_call(lambda: nx.clustering(g), warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: nx.clustering(g), sync_after=False)
                mean = sum(vals.values()) / len(vals) if vals else 0.0
                detail = write_vector_detail(backend, "LCC", vals, n)
                emit_metrics(
                    backend,
                    "LCC",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"vertices={len(vals)}, mean={mean:.9g}" + detail,
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "Closeness":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                t0 = time.perf_counter()
                g = build_networkx(n, base_edges, graph_type == "directed")
                close_graph = g.reverse(copy=False) if graph_type == "directed" else g
                t1 = time.perf_counter()
                source_nodes = pick_optional_sources(n, closeness_sources)
                if source_nodes is None:
                    run_closeness = lambda: nx.closeness_centrality(
                        close_graph,
                        distance=None,
                        wf_improved=True,
                    )
                else:
                    run_closeness = lambda: [
                        nx.closeness_centrality(
                            close_graph,
                            u=int(source),
                            distance=None,
                            wf_improved=True,
                        )
                        for source in source_nodes
                    ]
                warmup_call(run_closeness, warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(run_closeness, sync_after=False)
                count, total, mean = summarize_numeric_mapping(vals)
                if source_nodes is None:
                    detail = write_vector_detail(backend, "Closeness", vals, n)
                    corr_prefix = f"nodes={count}"
                    note = "outgoing distance for directed graphs"
                else:
                    detail = write_source_vector_detail(backend, "Closeness", source_nodes, vals, n)
                    corr_prefix = f"sources={len(source_nodes)}, graph_nodes={n}"
                    note = (
                        "sampled-target exact closeness; deterministic evenly spaced sources; "
                        f"sources={len(source_nodes)}; outgoing distance for directed graphs"
                    )
                if graph_type == "directed":
                    note += "; networkx uses reverse graph view because its directed closeness is inward by default"
                emit_metrics(
                    backend,
                    "Closeness",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"{corr_prefix}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=note + "; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in {"WCC", "SCC"}:
                plan = component_semantics(func, graph_type, views)
                n = plan["n"]
                t0 = time.perf_counter()
                g = build_networkx(n, plan["edges"], plan["build_directed"])
                t1 = time.perf_counter()
                if plan["nx_kind"] == "scc":
                    run_components = lambda: list(nx.strongly_connected_components(g))
                else:
                    run_components = lambda: list(nx.connected_components(g))
                warmup_call(run_components, warmup)
                barrier(cooldown)
                comps, algo_s, mem = timed_algorithm(run_components, sync_after=False)
                detail = write_cc_detail(backend, func, comps, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"components={len(comps)}" + detail,
                    notes=plan["note"] + "; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in PATH_SOURCE_FUNCTIONS:
                plan = path_benchmark_plan(func, graph_type, views, sssp_sources)
                n = plan["n"]
                source_nodes = plan["sources"]
                if not source_nodes:
                    emit_skip(backend, func, "empty source set")
                    continue
                t0 = time.perf_counter()
                g = build_networkx(n, plan["edges"], plan["directed"], weighted=plan["weighted"])
                t1 = time.perf_counter()
                if func == "BFS":
                    nx_path_fn = lambda s: nx.single_source_shortest_path_length(g, s)
                    algo_name = "single_source_shortest_path_length"
                elif func == "BellmanFord":
                    nx_path_fn = lambda s: nx.single_source_bellman_ford_path_length(g, s, weight="weight")
                    algo_name = "single_source_bellman_ford_path_length"
                else:
                    nx_path_fn = lambda s: nx.single_source_dijkstra_path_length(g, s, weight="weight")
                    algo_name = "single_source_dijkstra_path_length"
                warmup_call(
                    lambda: {
                        s: nx_path_fn(s)
                        for s in source_nodes
                    },
                    warmup,
                )
                barrier(cooldown)
                dists, algo_s, mem = timed_algorithm(
                    lambda: {
                        s: nx_path_fn(s)
                        for s in source_nodes
                    },
                    sync_after=False,
                )
                reachable, checksum = summarize_sssp_result(dists)
                detail = write_sssp_detail(backend, func, dists, source_nodes, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"sources={len(source_nodes)}, reachable={reachable}, checksum={checksum:.9g}" + detail,
                    notes=plan["note"] + f"; networkx algorithm={algo_name}; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in STRUCTURAL_HOLE_FUNCTIONS:
                if func in {"Efficiency", "Hierarchy"}:
                    metric_name = "efficiency" if func == "Efficiency" else "hierarchy"
                    emit_skip(
                        backend,
                        func,
                        "networkx has no native Burt "
                        f"{metric_name} metric API; runner-side derivation is not counted "
                        "as baseline function support",
                    )
                    continue
                plan = structural_hole_plan(func, graph_type, views)
                n = plan["n"]
                t0 = time.perf_counter()
                g = build_networkx(n, plan["edges"], plan["directed"])
                t1 = time.perf_counter()

                if func == "EffectiveSize":
                    run_structural = lambda: nx.effective_size(g, weight=None)
                    note = plan["note"] + "; networkx structuralholes.effective_size"
                else:
                    run_structural = lambda: nx.constraint(g, weight=None)
                    note = plan["note"] + "; networkx structuralholes.constraint"

                warmup_call(run_structural, warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(run_structural, sync_after=False)
                count, total, mean = summarize_numeric_mapping(vals)
                detail = write_vector_detail(backend, func, vals, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={count}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=note + "; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "KCore":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = undirected_edges
                t0 = time.perf_counter()
                g = build_networkx(n, base_edges, False)
                t1 = time.perf_counter()
                warmup_call(lambda: nx.core_number(g), warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: nx.core_number(g), sync_after=False)
                total = float(sum(vals.values())) if vals else 0.0
                max_core = int(max(vals.values())) if vals else 0
                detail = write_vector_detail(backend, "KCore", vals, n, dtype="int64")
                emit_metrics(
                    backend,
                    "KCore",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={len(vals)}, sum={total:.9g}, max={max_core}" + detail,
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "BC":
                from networkx.algorithms.centrality import betweenness_centrality_subset

                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                source_nodes = pick_sources(n, bc_sources)
                if not source_nodes:
                    emit_skip(backend, "BC", "empty source set")
                    continue
                t0 = time.perf_counter()
                g = build_networkx(n, base_edges, graph_type == "directed")
                t1 = time.perf_counter()
                targets = list(g.nodes())
                warmup_call(
                    lambda: betweenness_centrality_subset(
                        g,
                        sources=source_nodes,
                        targets=targets,
                        normalized=False,
                        weight=None,
                    ),
                    warmup,
                )
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(
                    lambda: betweenness_centrality_subset(
                        g,
                        sources=source_nodes,
                        targets=targets,
                        normalized=False,
                        weight=None,
                    ),
                    sync_after=False,
                )
                total = float(sum(vals.values())) if vals else 0.0
                detail = write_vector_detail(backend, "BC", vals, n)
                emit_metrics(
                    backend,
                    "BC",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={len(vals)}, sum={total:.9g}" + detail,
                    notes=f"source-sampled subset mode; sources={len(source_nodes)}; normalized=False; cpu backend kernel=algorithm",
                    memory=mem,
                )
        except Exception as e:
            emit_exception(backend, func, e)


def bench_easygraph_mode(
    views,
    graph_type,
    skip_cpu,
    functions,
    pr_alpha,
    pr_max_iter,
    pr_tol,
    warmup,
    easygraph_warmup,
    sssp_sources,
    bc_sources,
    closeness_sources,
    cooldown,
    mode,
    execution_protocol="steady-state",
):
    if mode == "gpu":
        backend = "EGGPU"
        if execution_protocol not in {"steady-state", "first-use"}:
            raise ValueError(f"unknown EGGPU execution protocol: {execution_protocol}")
        if execution_protocol == "first-use" and (
            int(warmup) != 0 or int(easygraph_warmup) != 0
        ):
            raise ValueError(
                "first-use protocol requires --warmup 0 and --easygraph-warmup 0"
            )
        os.environ["EGGPU_EXECUTION_PROTOCOL"] = execution_protocol
        os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
        os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "TRUE"
        os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
        os.environ["EASYGRAPH_GPU_PR_MAX_ITER"] = str(pr_max_iter)
        os.environ["EASYGRAPH_GPU_PR_EPS"] = str(pr_tol)
        os.environ["EASYGRAPH_CPU_PR_MAX_ITER"] = str(pr_max_iter)
        os.environ["EASYGRAPH_CPU_PR_TOL"] = str(pr_tol)
        # Keep timed runs unbiased by backend result cache hits.
        os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
    elif mode == "cpu":
        backend = "easygraph-cpu"
        os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"
        os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "FALSE"
        os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_CPU_PR_MAX_ITER"] = str(pr_max_iter)
        os.environ["EASYGRAPH_CPU_PR_TOL"] = str(pr_tol)
    elif mode == "cpp":
        backend = "easygraph-cpp"
        os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"
        os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "FALSE"
        os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_CPU_PR_MAX_ITER"] = str(pr_max_iter)
        os.environ["EASYGRAPH_CPU_PR_TOL"] = str(pr_tol)
    else:
        raise ValueError(f"unknown easygraph mode: {mode}")

    if skip_cpu and mode in ("cpu", "cpp"):
        for func in functions:
            emit_skip(backend, func, "CPU baseline skipped by explicit --skip-cpu")
        return

    print(
        f"[easygraph-mode] backend={backend} mode={mode} "
        f"EASYGRAPH_ENABLE_GPU={os.environ.get('EASYGRAPH_ENABLE_GPU', '')}",
        flush=True,
    )

    try:
        import easygraph as eg
    except Exception as e:
        for func in functions:
            emit_skip(backend, func, f"easygraph import failed: {e}")
        return

    eggpu_backend = None
    if mode == "gpu":
        try:
            from easygraph.utils import gpu_eggpu_backend as eggpu_backend_module

            eggpu_backend = eggpu_backend_module
        except Exception:
            eggpu_backend = None

    def to_mode_graph(n, edges, directed, weighted=False):
        g = build_easygraph(n, edges, directed, weighted=weighted)
        if mode == "cpp":
            return g.cpp()
        if (
            mode == "gpu"
            and execution_protocol == "steady-state"
            and eggpu_backend is not None
        ):
            try:
                eggpu_backend._graph_context(g, prewarm_cpp=True)
            except Exception as exc:
                if os.environ.get("EASYGRAPH_GPU_STRICT_ERRORS", "").strip().upper() in {
                    "1",
                    "TRUE",
                    "ON",
                    "YES",
                }:
                    raise RuntimeError("EGGPU graph-context prewarm failed") from exc
                pass
        return g

    if mode == "gpu" and execution_protocol == "steady-state":
        effective_warmup = max(0, int(max(int(warmup), int(easygraph_warmup))))
    elif mode == "gpu":
        effective_warmup = 0
    else:
        effective_warmup = max(0, int(warmup))
    # EGGPU kernels synchronize/copy required outputs before returning. An
    # extra Python-side CUDA sync is not part of the user-facing call path and
    # over-penalizes small-graph e2e latency.
    sync_after_eg_call = False

    def easygraph_warmup(callable_obj):
        for _ in range(effective_warmup):
            try:
                callable_obj()
            except Exception:
                if mode == "gpu":
                    raise
                break

    def kernel_or_algo(kernel_key, algo_seconds):
        if mode != "gpu" or eggpu_backend is None:
            return algo_seconds
        try:
            k = eggpu_backend.get_last_kernel_time(kernel_key)
        except Exception:
            k = None
        return algo_seconds if k is None else float(k)

    def reset_kernel_key(kernel_key):
        if mode != "gpu" or eggpu_backend is None:
            return
        try:
            eggpu_backend.set_last_kernel_time(kernel_key, None)
        except Exception:
            pass

    def require_kernel_time(kernel_key):
        if eggpu_backend is None:
            raise RuntimeError("EGGPU kernel timer backend is unavailable")
        kernel_value = eggpu_backend.get_last_kernel_time(kernel_key)
        if kernel_value is None:
            raise RuntimeError(f"missing EGGPU kernel timing for {kernel_key}")
        if not math.isfinite(kernel_value):
            raise RuntimeError(
                f"non-finite EGGPU kernel timing for {kernel_key}: {kernel_value}"
            )
        if kernel_value <= 0:
            raise RuntimeError(
                f"non-positive EGGPU kernel timing for {kernel_key}: {kernel_value}"
            )
        return float(kernel_value)

    for func in functions:
        try:
            if func == "PageRank":
                n, directed_edges, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = to_mode_graph(n, directed_edges if graph_type == "directed" else undirected_edges, graph_type == "directed")
                t1 = time.perf_counter()
                pagerank_signature = inspect.signature(eg.pagerank)
                required_parameters = {"G", "alpha", "max_iter", "tol", "weight"}
                missing_parameters = sorted(
                    required_parameters - set(pagerank_signature.parameters)
                )
                if missing_parameters:
                    raise TypeError(
                        "EasyGraph PageRank public signature lacks required "
                        f"parameters: {missing_parameters}"
                    )

                def invoke_pagerank_direct():
                    return eg.pagerank(
                        g,
                        alpha=pr_alpha,
                        max_iter=pr_max_iter,
                        tol=pr_tol,
                        weight=None,
                    )

                easygraph_warmup(invoke_pagerank_direct)
                barrier(cooldown)
                reset_kernel_key("pagerank")
                ranks, algo_s, mem = timed_algorithm(
                    invoke_pagerank_direct,
                    sync_after=sync_after_eg_call,
                )
                rank_sum = validate_pagerank_result(ranks, n)
                if mode == "gpu":
                    kernel_s = require_kernel_time("pagerank")
                    if kernel_s >= algo_s:
                        raise RuntimeError(
                            "PageRank device interval must be shorter than "
                            f"the public-call interval: {kernel_s} >= {algo_s}"
                        )
                else:
                    kernel_s = algo_s
                note = f"alpha={pr_alpha}, max_iter={pr_max_iter}, tol={pr_tol}"
                if isinstance(ranks, (list, tuple)):
                    note += "; pagerank returned dense vector"
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                detail = write_vector_detail(backend, "PageRank", ranks, n)
                emit_metrics(
                    backend,
                    "PageRank",
                    t1 - t0,
                    algo_s,
                    kernel_s,
                    correctness=f"sum={rank_sum:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func == "MST":
                reset_kernel_key("mst")
                n, _, undirected_edges = views["all_vertices"]
                t0 = time.perf_counter()
                g = to_mode_graph(n, undirected_edges, False, weighted=True)
                t1 = time.perf_counter()
                easygraph_warmup(lambda: eg.minimum_spanning_tree(g, weight="weight"))
                barrier(cooldown)
                tree, algo_s, mem = timed_algorithm(
                    lambda: eg.minimum_spanning_tree(g, weight="weight"),
                    sync_after=sync_after_eg_call,
                )
                weight = int(sum(data.get("weight", 1) for _, _, data in easygraph_edges(tree)))
                note = "undirected projection"
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                emit_metrics(
                    backend,
                    "MST",
                    t1 - t0,
                    algo_s,
                    kernel_or_algo("mst", algo_s),
                    correctness=f"weight={weight}",
                    notes=note,
                    memory=mem,
                )

            elif func == "LCC":
                reset_kernel_key("lcc")
                n, _, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = to_mode_graph(n, undirected_edges, False)
                t1 = time.perf_counter()
                easygraph_warmup(lambda: eg.clustering(g))
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: eg.clustering(g), sync_after=sync_after_eg_call)
                mean = sum(vals.values()) / len(vals) if vals else 0.0
                note = "undirected projection"
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                detail = write_vector_detail(backend, "LCC", vals, n)
                emit_metrics(
                    backend,
                    "LCC",
                    t1 - t0,
                    algo_s,
                    kernel_or_algo("lcc", algo_s),
                    correctness=f"vertices={len(vals)}, mean={mean:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func == "Closeness":
                reset_kernel_key("closeness")
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                t0 = time.perf_counter()
                g = to_mode_graph(n, base_edges, graph_type == "directed")
                t1 = time.perf_counter()
                source_nodes = pick_optional_sources(n, closeness_sources)
                run_closeness = lambda: eg.closeness_centrality(
                    g,
                    weight=None,
                    sources=source_nodes,
                )
                easygraph_warmup(run_closeness)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(run_closeness, sync_after=sync_after_eg_call)
                count, total, mean = summarize_numeric_mapping(vals)
                if source_nodes is None:
                    detail = write_vector_detail(backend, "Closeness", vals, n)
                    corr_prefix = f"nodes={count}"
                    note = "outgoing distance for directed graphs"
                else:
                    detail = write_source_vector_detail(backend, "Closeness", source_nodes, vals, n)
                    corr_prefix = f"sources={len(source_nodes)}, graph_nodes={n}"
                    note = (
                        "sampled-target exact closeness; deterministic evenly spaced sources; "
                        f"sources={len(source_nodes)}; outgoing distance for directed graphs"
                    )
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                else:
                    note += "; unweighted GPU path uses BFS kernel"
                emit_metrics(
                    backend,
                    "Closeness",
                    t1 - t0,
                    algo_s,
                    kernel_or_algo("closeness", algo_s),
                    correctness=f"{corr_prefix}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func in {"WCC", "SCC"}:
                plan = component_semantics(func, graph_type, views)
                reset_kernel_key(plan["kernel_key"])
                n = plan["n"]
                t0 = time.perf_counter()
                g = to_mode_graph(n, plan["edges"], plan["build_directed"])
                t1 = time.perf_counter()
                if plan["eg_kind"] == "scc":
                    run_components = lambda: list(eg.strongly_connected_components(g))
                else:
                    run_components = lambda: list(eg.connected_components(g))
                easygraph_warmup(run_components)
                barrier(cooldown)
                comps, algo_s, mem = timed_algorithm(run_components, sync_after=sync_after_eg_call)
                note = plan["note"]
                kernel_key = plan["kernel_key"]
                if mode != "gpu":
                    note = (note + "; " if note else "") + "cpu backend kernel=algorithm"
                detail = write_cc_detail(backend, func, comps, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    kernel_or_algo(kernel_key, algo_s),
                    correctness=f"components={len(comps)}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func in PATH_SOURCE_FUNCTIONS:
                plan = path_benchmark_plan(func, graph_type, views, sssp_sources)
                reset_kernel_key(plan["kernel_key"])
                if func in {"Dijkstra", "BFS"}:
                    reset_kernel_key("sssp")
                n = plan["n"]
                source_nodes = plan["sources"]
                if not source_nodes:
                    emit_skip(backend, func, "empty source set")
                    continue
                if mode == "cpp" and func == "BellmanFord":
                    emit_skip(backend, func, "easygraph-cpp has no Bellman-Ford binding; CPU/GPU EasyGraph paths are benchmarked")
                    continue
                t0 = time.perf_counter()
                g = to_mode_graph(n, plan["edges"], plan["directed"], weighted=plan["weighted"])
                t1 = time.perf_counter()

                if func == "BFS":
                    if mode == "cpp":
                        run_paths = lambda: eg.multi_source_dijkstra(g, source_nodes, weight=None, target=None)
                    else:
                        run_paths = lambda: eg.multi_source_bfs(g, source_nodes, target=None)
                    note = plan["note"]
                    if mode == "gpu":
                        note += "; unweighted GPU path uses BFS kernel"
                elif func == "BellmanFord":
                    run_paths = lambda: eg.multi_source_bellman_ford(g, source_nodes, weight="weight", target=None)
                    note = plan["note"]
                else:
                    run_paths = lambda: eg.multi_source_dijkstra(g, source_nodes, weight="weight", target=None)
                    note = plan["note"]

                easygraph_warmup(run_paths)
                barrier(cooldown)
                dists, algo_s, mem = timed_algorithm(
                    run_paths,
                    sync_after=sync_after_eg_call,
                )
                reachable, checksum = summarize_sssp_result(dists)
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                detail = write_sssp_detail(backend, func, dists, source_nodes, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    kernel_or_algo(plan["kernel_key"], algo_s),
                    correctness=f"sources={len(source_nodes)}, reachable={reachable}, checksum={checksum:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func in STRUCTURAL_HOLE_FUNCTIONS:
                if mode == "cpp":
                    emit_skip(
                        backend,
                        func,
                        "GPU-enabled cpp_easygraph structural-hole bindings route to CUDA at compile time; skipped to keep CPU C++ baseline isolated",
                    )
                    continue
                plan = structural_hole_plan(func, graph_type, views)
                reset_kernel_key(plan["kernel_key"])
                n = plan["n"]
                t0 = time.perf_counter()
                g = to_mode_graph(n, plan["edges"], plan["directed"])
                t1 = time.perf_counter()
                fn = {
                    "EffectiveSize": eg.effective_size,
                    "Efficiency": eg.efficiency,
                    "Constraint": eg.constraint,
                    "Hierarchy": eg.hierarchy,
                }[func]
                run_structural = lambda: fn(g, weight=None)
                easygraph_warmup(run_structural)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(run_structural, sync_after=sync_after_eg_call)
                count, total, mean = summarize_numeric_mapping(vals)
                note = plan["note"]
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                detail = write_vector_detail(backend, func, vals, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    kernel_or_algo(plan["kernel_key"], algo_s),
                    correctness=f"nodes={count}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func == "KCore":
                reset_kernel_key("kcore")
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = undirected_edges
                t0 = time.perf_counter()
                g = to_mode_graph(n, base_edges, False)
                t1 = time.perf_counter()
                kcore_note_suffix = ""

                def run_kcore_once():
                    return eg.k_core(g)

                easygraph_warmup(run_kcore_once)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(
                    run_kcore_once,
                    sync_after=sync_after_eg_call,
                )
                if hasattr(vals, "tolist"):
                    vals = vals.tolist()
                if isinstance(vals, (list, tuple)) and len(vals) == n + 1:
                    vals = list(vals)[1:]
                if isinstance(vals, Mapping):
                    seq = list(vals.values())
                elif isinstance(vals, (list, tuple)):
                    seq = list(vals)
                else:
                    seq = []
                total = float(sum(seq)) if seq else 0.0
                max_core = int(max(seq)) if seq else 0
                note = "undirected projection"
                if mode != "gpu":
                    note = (note + "; " if note else "") + "cpu backend kernel=algorithm"
                detail = write_vector_detail(backend, "KCore", vals, n, dtype="int64")
                emit_metrics(
                    backend,
                    "KCore",
                    t1 - t0,
                    algo_s,
                    kernel_or_algo("kcore", algo_s),
                    correctness=f"nodes={len(seq)}, sum={total:.9g}, max={max_core}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func == "BC":
                reset_kernel_key("bc")
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                source_nodes = pick_sources(n, bc_sources)
                if not source_nodes:
                    emit_skip(backend, "BC", "empty source set")
                    continue
                t0 = time.perf_counter()
                g = to_mode_graph(n, base_edges, graph_type == "directed")
                t1 = time.perf_counter()
                easygraph_warmup(
                    lambda: eg.betweenness_centrality(
                        g,
                        weight=None,
                        sources=source_nodes,
                        normalized=False,
                        endpoints=False,
                    )
                )
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(
                    lambda: eg.betweenness_centrality(
                        g,
                        weight=None,
                        sources=source_nodes,
                        normalized=False,
                        endpoints=False,
                    ),
                    sync_after=sync_after_eg_call,
                )
                if hasattr(vals, "tolist"):
                    vals = vals.tolist()
                if isinstance(vals, dict):
                    seq = list(vals.values())
                elif isinstance(vals, (list, tuple)):
                    seq = list(vals)
                else:
                    seq = []
                total = float(sum(seq)) if seq else 0.0
                note = f"source-sampled; sources={len(source_nodes)}; normalized=False"
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                else:
                    note += "; unweighted GPU path uses BFS-Brandes kernel"
                detail = write_vector_detail(backend, "BC", vals, n)
                emit_metrics(
                    backend,
                    "BC",
                    t1 - t0,
                    algo_s,
                    kernel_or_algo("bc", algo_s),
                    correctness=f"nodes={len(seq)}, sum={total:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )
        except Exception as e:
            emit_exception(backend, func, e)


def bench_nx_cugraph(views, graph_type, functions, pr_alpha, pr_max_iter, pr_tol, cooldown, warmup, sssp_sources, bc_sources, closeness_sources):
    backend = "nx-cugraph"
    os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "TRUE"
    if all(func in STRUCTURAL_HOLE_FUNCTIONS for func in functions):
        for func in functions:
            emit_skip(backend, func, "nx-cugraph has no native Burt structural-hole metric API")
        return
    cuda_runtime = configure_nx_cugraph_cuda_runtime()
    try:
        import nx_cugraph  # noqa: F401
        import networkx as nx
    except Exception as e:
        for func in functions:
            emit_skip(backend, func, f"nx-cugraph import failed: {e}")
        return
    nx.config.fallback_to_nx = False
    nx.config.cache_converted_graphs = True
    if nx.config.fallback_to_nx or not nx.config.cache_converted_graphs:
        raise RuntimeError(
            "strict nx-cugraph protocol requires CPU fallback disabled and "
            "NetworkX converted-graph caching enabled"
        )

    def run_strict(callable_obj, function, source_count=1):
        with NxCugraphDeviceTimer() as device_timer:
            warmup_details = warm_nx_cugraph(
                callable_obj,
                device_timer,
                function,
                warmup,
                source_count,
            )
            barrier(cooldown)
            result = timed_nx_cugraph_algorithm(
                callable_obj,
                device_timer,
                function,
                source_count,
            )
        output, wall_s, device_s, memory, provenance = result
        provenance.update(
            {
                "warmup_calls": int(warmup),
                "warmup_details": warmup_details,
                "networkx_cache_converted_graphs": bool(
                    nx.config.cache_converted_graphs
                ),
                "networkx_fallback_to_nx": bool(nx.config.fallback_to_nx),
                "cuda_runtime": cuda_runtime,
            }
        )
        return output, wall_s, device_s, memory, provenance

    for func in functions:
        try:
            if func == "PageRank":
                n, directed_edges, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_nxcugraph_native(
                    n,
                    directed_edges
                    if graph_type == "directed"
                    else undirected_edges,
                    graph_type == "directed",
                )
                t1 = time.perf_counter()
                ranks, algo_s, kernel_s, mem, provenance = run_strict(
                    lambda: nx_cugraph_call(nx.pagerank, g, alpha=pr_alpha, max_iter=pr_max_iter, tol=pr_tol),
                    "PageRank",
                )
                emit_metrics(
                    backend,
                    "PageRank",
                    t1 - t0,
                    algo_s,
                    kernel_s,
                    correctness=f"sum={sum(ranks.values()):.9g}" + write_vector_detail(backend, "PageRank", ranks, n),
                    notes=(
                        f"alpha={pr_alpha}, max_iter={pr_max_iter}, "
                        f"tol={pr_tol}; prepared native nx-cugraph graph; "
                        "processing uses backend CUDA-event intervals"
                    ),
                    memory=mem,
                    provenance=provenance,
                )

            elif func == "MST":
                emit_skip(
                    backend,
                    "MST",
                    (
                        "nx-cugraph exposes no aligned "
                        "minimum-spanning-forest backend"
                    ),
                )

            elif func == "LCC":
                n, _, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_nxcugraph_native(n, undirected_edges, False)
                t1 = time.perf_counter()
                vals, algo_s, kernel_s, mem, provenance = run_strict(
                    lambda: nx_cugraph_call(nx.clustering, g), "LCC"
                )
                note = (
                    "undirected projection; prepared native nx-cugraph "
                    "graph; processing uses backend CUDA-event intervals"
                )
                mean = sum(vals.values()) / len(vals) if vals else 0.0
                detail = write_vector_detail(backend, "LCC", vals, n)
                emit_metrics(
                    backend,
                    "LCC",
                    t1 - t0,
                    algo_s,
                    kernel_s,
                    correctness=f"vertices={len(vals)}, mean={mean:.9g}" + detail,
                    notes=note,
                    memory=mem,
                    provenance=provenance,
                )

            elif func == "Closeness":
                emit_skip(
                    backend,
                    "Closeness",
                    "nx-cugraph supported-algorithm list does not include closeness_centrality; "
                    "skipped to avoid measuring an unsupported backend or CPU fallback",
                )

            elif func in {"WCC", "SCC"}:
                if func == "SCC":
                    emit_skip(
                        backend,
                        "SCC",
                        (
                            "nx-cugraph exposes no aligned "
                            "strongly_connected_components backend"
                        ),
                    )
                    continue
                plan = component_semantics(func, graph_type, views)
                n = plan["n"]
                t0 = time.perf_counter()
                g = build_nxcugraph_native(
                    n, plan["edges"], plan["build_directed"]
                )
                t1 = time.perf_counter()
                nx_component_call = lambda: list(
                    nx_cugraph_call(nx.connected_components, g)
                )
                comps, algo_s, kernel_s, mem, provenance = run_strict(
                    nx_component_call, "WCC"
                )
                comp_count = len(comps)
                note = (
                    plan["note"]
                    + "; prepared native nx-cugraph graph; processing uses "
                    "backend CUDA-event intervals"
                )
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    kernel_s,
                    correctness=f"components={comp_count}",
                    notes=note,
                    memory=mem,
                    provenance=provenance,
                )

            elif func in PATH_SOURCE_FUNCTIONS:
                plan = path_benchmark_plan(func, graph_type, views, sssp_sources)
                n = plan["n"]
                source_nodes = plan["sources"]
                if not source_nodes:
                    emit_skip(backend, func, "empty source set")
                    continue
                t0 = time.perf_counter()
                g = build_nxcugraph_native(
                    n,
                    plan["edges"],
                    plan["directed"],
                    weighted=plan["weighted"],
                )
                t1 = time.perf_counter()

                def run_paths():
                    if func == "BFS":
                        out = {
                            s: nx_cugraph_call(
                                nx.single_source_shortest_path_length,
                                g,
                                s,
                            )
                            for s in source_nodes
                        }
                        note = (
                            plan["note"]
                            + "; nx-cugraph "
                            "single_source_shortest_path_length"
                        )
                        return out, note
                    if func == "BellmanFord":
                        out = {
                            s: nx_cugraph_call(
                                nx.single_source_bellman_ford_path_length,
                                g,
                                s,
                                weight="weight",
                            )
                            for s in source_nodes
                        }
                        note = (
                            plan["note"]
                            + "; nx-cugraph "
                            "single_source_bellman_ford_path_length"
                        )
                        return out, note
                    out = {
                        s: nx_cugraph_call(
                            nx.single_source_dijkstra_path_length,
                            g,
                            source=s,
                            weight="weight",
                        )
                        for s in source_nodes
                    }
                    note = (
                        plan["note"]
                        + "; nx-cugraph "
                        "single_source_dijkstra_path_length"
                    )
                    return out, note

                (
                    (dists, note),
                    algo_s,
                    kernel_s,
                    mem,
                    provenance,
                ) = run_strict(
                    run_paths,
                    func,
                    source_count=len(source_nodes),
                )
                reachable, checksum = summarize_sssp_result(dists)
                detail = write_sssp_detail(backend, func, dists, source_nodes, n)
                note += (
                    "; prepared native nx-cugraph graph; processing uses "
                    "backend CUDA-event intervals"
                )
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    kernel_s,
                    correctness=f"sources={len(source_nodes)}, reachable={reachable}, checksum={checksum:.9g}" + detail,
                    notes=note,
                    memory=mem,
                    provenance=provenance,
                )

            elif func in STRUCTURAL_HOLE_FUNCTIONS:
                emit_skip(backend, func, "nx-cugraph has no native Burt structural-hole metric API")

            elif func == "KCore":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = undirected_edges
                t0 = time.perf_counter()
                g = build_nxcugraph_native(n, base_edges, False)
                t1 = time.perf_counter()

                def run_kcore():
                    return nx_cugraph_call(nx.core_number, g)

                vals, algo_s, kernel_s, mem, provenance = run_strict(
                    run_kcore, "KCore"
                )
                note = (
                    "undirected projection; prepared native nx-cugraph "
                    "graph; processing uses backend CUDA-event intervals"
                )
                total = float(sum(vals.values())) if isinstance(vals, dict) and vals else 0.0
                max_core = int(max(vals.values())) if isinstance(vals, dict) and vals else 0
                count = len(vals) if isinstance(vals, dict) else 0
                detail = write_vector_detail(backend, "KCore", vals, n, dtype="int64") if isinstance(vals, dict) else ""
                emit_metrics(
                    backend,
                    "KCore",
                    t1 - t0,
                    algo_s,
                    kernel_s,
                    correctness=f"nodes={count}, sum={total:.9g}, max={max_core}" + detail,
                    notes=note,
                    memory=mem,
                    provenance=provenance,
                )

            elif func == "BC":
                emit_skip(
                    backend,
                    "BC",
                    (
                        "ordinary all-source BC exists, but the paper workload "
                        "is exact 16-source betweenness_centrality_subset, "
                        "which the cugraph backend does not implement"
                    ),
                )
        except Exception as e:
            if is_nx_cugraph_not_implemented(e):
                emit_skip(backend, func, f"not implemented by nx-cugraph: {e}")
            else:
                emit_exception(backend, func, e)


def main():
    global MEASUREMENT_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("edge_path")
    ap.add_argument("graph_type", choices=["directed", "undirected"])
    ap.add_argument(
        "--backend",
        choices=["all", "igraph", "networkx", "EGGPU", "easygraph-cpu", "easygraph-cpp", "nx-cugraph"],
        default="all",
    )
    ap.add_argument("--function", choices=["all", *FUNCTION_ORDER, *LEGACY_FUNCTION_ALIASES], default="all")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--pr-alpha", type=float, default=0.75)
    ap.add_argument("--pr-tol", type=float, default=1e-6)
    ap.add_argument("--pr-max-iter", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=0, help="Legacy EGGPU warmup input.")
    ap.add_argument("--easygraph-warmup", type=int, default=2)
    ap.add_argument(
        "--nx-cugraph-warmup",
        type=int,
        default=3,
        help=(
            "Untimed strict public calls on the prepared native nx-cugraph "
            "graph before the measured call."
        ),
    )
    ap.add_argument(
        "--eggpu-execution-protocol",
        choices=["steady-state", "first-use"],
        default="steady-state",
        help=(
            "steady-state prebuilds reusable EGGPU graph state and applies the configured "
            "untimed warmups; first-use performs neither and measures the first supported "
            "function call on a newly built Python graph."
        ),
    )
    ap.add_argument("--sssp-sources", type=int, default=8, help="Number of deterministic sources for SSSP.")
    ap.add_argument("--bc-sources", type=int, default=16, help="Number of deterministic sources for BC source-sampled mode.")
    ap.add_argument(
        "--closeness-sources",
        type=int,
        default=0,
        help=(
            "If >0, run Closeness in sampled-target exact mode on this many deterministic nodes. "
            "Default 0 keeps exact all-node Closeness semantics."
        ),
    )
    ap.add_argument("--cooldown", type=float, default=0.2, help="Sleep seconds between build/algorithm phases for run isolation.")
    ap.add_argument(
        "--measurement-mode",
        choices=["timing", "memory", "combined"],
        default=os.environ.get("EGGPU_MEASUREMENT_MODE", "combined").strip().lower(),
        help="timing disables memory sampling; memory emits only memory metrics; combined is legacy behavior.",
    )
    args = ap.parse_args()
    MEASUREMENT_MODE = args.measurement_mode
    os.environ["EGGPU_MEASUREMENT_MODE"] = MEASUREMENT_MODE

    print(
        "BASELINE_VERSION_JSON "
        + json.dumps(collect_python_baseline_versions(), sort_keys=True),
        flush=True,
    )

    if args.function == "all":
        functions = list(FUNCTION_ORDER)
    elif args.function in LEGACY_FUNCTION_ALIASES:
        functions = list(LEGACY_FUNCTION_ALIASES[args.function])
    else:
        functions = [args.function]

    views = load_graph(args.edge_path)
    n_clean, directed_clean, undirected_clean = views["clean"]
    n_all, directed_all, undirected_all = views["all_vertices"]
    print(
        f"[data] clean_nodes={n_clean} all_nodes={n_all} directed_edges={len(directed_clean)} undirected_unique={len(undirected_clean)} graph_type={args.graph_type}",
        flush=True,
    )
    print(
        "[data] self-loop rows are removed during benchmark graph construction; raw files are left unchanged.",
        flush=True,
    )
    print(
        f"[params] pagerank alpha={args.pr_alpha} tol={args.pr_tol} max_iter={args.pr_max_iter}",
        flush=True,
    )
    print(
        f"[params] easygraph_series_warmup={args.warmup} easygraph_warmup={args.easygraph_warmup} "
        f"nx_cugraph_warmup={args.nx_cugraph_warmup} "
        f"sssp_sources={args.sssp_sources} bc_sources={args.bc_sources} closeness_sources={args.closeness_sources} "
        f"measurement_mode={MEASUREMENT_MODE} eggpu_execution_protocol={args.eggpu_execution_protocol}",
        flush=True,
    )
    print(
        "[note] timings exclude raw edge-list parsing and import time; timing and memory metrics are emitted according to measurement_mode.",
        flush=True,
    )

    if args.backend in ("all", "igraph"):
        bench_igraph(
            views,
            args.graph_type,
            args.skip_cpu,
            functions,
            args.pr_alpha,
            args.cooldown,
            0,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
        )
    if args.backend in ("all", "networkx"):
        bench_networkx(
            views,
            args.graph_type,
            args.skip_cpu,
            functions,
            args.pr_alpha,
            args.pr_max_iter,
            args.pr_tol,
            args.cooldown,
            0,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
        )
    if args.backend in ("all", "EGGPU"):
        bench_easygraph_mode(
            views,
            args.graph_type,
            args.skip_cpu,
            functions,
            args.pr_alpha,
            args.pr_max_iter,
            args.pr_tol,
            args.warmup,
            args.easygraph_warmup,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
            args.cooldown,
            mode="gpu",
            execution_protocol=args.eggpu_execution_protocol,
        )
    if args.backend in ("all", "easygraph-cpu"):
        bench_easygraph_mode(
            views,
            args.graph_type,
            args.skip_cpu,
            functions,
            args.pr_alpha,
            args.pr_max_iter,
            args.pr_tol,
            args.warmup,
            args.easygraph_warmup,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
            args.cooldown,
            mode="cpu",
        )
    if args.backend in ("all", "easygraph-cpp"):
        bench_easygraph_mode(
            views,
            args.graph_type,
            args.skip_cpu,
            functions,
            args.pr_alpha,
            args.pr_max_iter,
            args.pr_tol,
            args.warmup,
            args.easygraph_warmup,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
            args.cooldown,
            mode="cpp",
        )
    if args.backend in ("all", "nx-cugraph"):
        bench_nx_cugraph(
            views,
            args.graph_type,
            functions,
            args.pr_alpha,
            args.pr_max_iter,
            args.pr_tol,
            args.cooldown,
            args.nx_cugraph_warmup,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
        )


if __name__ == "__main__":
    main()
