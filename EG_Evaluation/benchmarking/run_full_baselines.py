#!/usr/bin/env python3
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import platform
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
except Exception:
    pynvml = None

from gpu_visibility_marker import GpuVisibilityMarker
from gpu_device_profile import collect_gpu_device_profile, collect_host_profile
from benchmark_stats import aggregate_sample_rows
from baseline_versions import collect_python_baseline_versions
from gunrock_timing_protocol import (
    GunrockTimingProtocolError,
    parse_strict_gunrock_timing,
)
from measurement_schema import describe_metric, write_measurement_schema
from stable_timing_protocol import (
    DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    DEFAULT_MEDIAN_OVER_MIN_LIMIT,
    EXPECTED_TIMING_SAMPLES,
    apply_controlled_thread_environment,
    collect_execution_placement,
    controlled_protocol_metadata,
    validate_controlled_execution,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
GUNROCK_ARTIFACT_MANIFEST = ROOT / "benchmarking" / "gunrock_artifact_manifest.json"
CONDA_EXE = os.environ.get("CONDA_EXE") or os.environ.get("_CONDA_EXE") or shutil.which("conda") or "conda"
DIRECT_CHILD_PYTHON = (
    os.environ.get("EGGPU_CHILD_PYTHON")
    or os.environ.get("COMMON_PY")
    or sys.executable
)


@contextmanager
def temporary_import_root(path):
    """Resolve provenance through the same import root used by child runs."""
    root = str(Path(path).resolve())
    sys.path.insert(0, root)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        try:
            sys.path.remove(root)
        except ValueError:
            pass
        importlib.invalidate_caches()


def _default_local_cuda_root():
    explicit_root = os.environ.get("EGGPU_CUDA_ROOT", "").strip()
    if explicit_root:
        path = Path(explicit_root).expanduser()
        if path.exists():
            return path

    for key in ("CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT"):
        value = os.environ.get(key, "").strip()
        if value:
            path = Path(value).expanduser()
            if path.exists():
                return path

    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        path = Path(conda_prefix).expanduser()
        # In the paper environment, the Python env is named EGGPU while the
        # compatible local CUDA toolkit lives in a sibling sglang env.  CuPy and
        # nx-cugraph JIT compilation can otherwise pick EGGPU/targets headers
        # and fail on CUDA 13.x-only headers.  This only affects inferred roots;
        # an explicit EGGPU_CUDA_ROOT above still wins.
        if path.name == "EGGPU":
            sibling = path.parent / "sglang"
            if (sibling / "bin" / "nvcc").exists():
                return sibling
        if path.exists():
            return path
    return None


DEFAULT_LOCAL_CUDA_ROOT = _default_local_cuda_root()
TRUE_VALUES = {"1", "TRUE", "ON", "YES"}
SANITIZE_ENV_VARS = (
    "CC",
    "CXX",
    "GCC",
    "GXX",
    "CUDAHOSTCXX",
    "CFLAGS",
    "DEBUG_CFLAGS",
    "CPPFLAGS",
    "DEBUG_CPPFLAGS",
    "CXXFLAGS",
    "DEBUG_CXXFLAGS",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "CPATH",
    "LIBRARY_PATH",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
)
DEFAULT_DATASETS = [
    ("small", "undirected", "ca-GrQc", "datasets/undirected/ca-GrQc.txt"),
    ("small", "undirected", "ca-HepTh", "datasets/undirected/ca-HepTh.txt"),
    ("small", "undirected", "LastFM", "datasets/undirected/LastFM.txt"),
    ("small", "undirected", "pgp", "datasets/undirected/pgp.txt"),
    ("medium", "undirected", "ca-CondMat", "datasets/undirected/ca-CondMat.txt"),
    ("medium", "undirected", "ca-HepPh", "datasets/undirected/ca-HepPh.txt"),
    ("medium", "undirected", "email-Enron", "datasets/undirected/email-Enron.txt"),
    ("large", "undirected", "com-youtube", "datasets/undirected/com-youtube.ungraph.txt"),
    ("small", "directed", "p2p-Gnutella04", "datasets/directed/p2p-Gnutella04.txt"),
    ("small", "directed", "p2p-Gnutella08", "datasets/directed/p2p-Gnutella08.txt"),
    ("medium", "directed", "wiki-Vote", "datasets/directed/wiki-Vote.txt"),
    ("medium", "directed", "soc-Epinions1", "datasets/directed/soc-Epinions1.txt"),
    ("medium", "directed", "email-EuAll", "datasets/directed/email-EuAll.txt"),
    ("large", "directed", "soc-Slashdot0811", "datasets/directed/soc-Slashdot0811.txt"),
    ("large", "directed", "web-NotreDame", "datasets/directed/web-NotreDame.txt"),
    ("large", "directed", "ER-100k", "datasets/directed/ER-100k.txt"),
    ("large", "directed", "wiki-Talk", "datasets/directed/wiki-Talk.txt"),
]
DEFAULT_FUNCTIONS = (
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
AVAILABLE_BASELINES = (
    "igraph",
    "networkx",
    "EGGPU",
    "easygraph-cpu",
    "easygraph-cpp",
    "nx-cugraph",
    "Gunrock",
)
LEGACY_FUNCTION_ALIASES = {"CC": ("WCC", "SCC")}
PER_FUNCTION_TIMEOUT_SECONDS = 100
RUN_GUNROCK_BASELINE = True
_NVML_INITIALIZED = False

# Gunrock's maintained v2.x tree does not ship every application that existed
# in the legacy v1.x tree.  Record the application lineage per executable so a
# result cannot silently present a mixed local build as one monolithic version.
GUNROCK_APPLICATION_PROVENANCE = {
    "pr": {
        "upstream_release_lineage": "v2.2.0",
        "application_generation": "maintained-v2",
        "benchmark_support": "direct-parameter-aligned",
        "build_cuda_toolkit_version": "13.2",
        "build_cuda_architecture": "80",
    },
    "mst": {
        "upstream_release_lineage": "v2.2.0",
        "application_generation": "maintained-v2",
        "benchmark_support": "connected-input-only",
        "build_cuda_toolkit_version": "13.2",
        "build_cuda_architecture": "80",
    },
    "bfs": {
        "upstream_release_lineage": "v2.2.0",
        "application_generation": "maintained-v2",
        "benchmark_support": "direct",
        "build_cuda_toolkit_version": "13.2",
        "build_cuda_architecture": "80",
    },
    "sssp": {
        "upstream_release_lineage": "v2.2.0",
        "application_generation": "maintained-v2",
        "benchmark_support": "direct-weighted-sssp",
        "build_cuda_toolkit_version": "13.2",
        "build_cuda_architecture": "80",
    },
    "kcore": {
        "upstream_release_lineage": "v2.2.0",
        "application_generation": "maintained-v2",
        "benchmark_support": "direct-undirected-projection",
        "build_cuda_toolkit_version": "13.2",
        "build_cuda_architecture": "80",
    },
    "bc": {
        "upstream_release_lineage": "v2.2.0",
        "application_generation": "maintained-v2",
        "benchmark_support": "direct",
        "build_cuda_toolkit_version": "13.2",
        "build_cuda_architecture": "80",
    },
    "lcc": {
        "upstream_release_lineage": "v1.x release-79-g386d04450",
        "application_generation": "legacy-v1",
        "benchmark_support": "direct-legacy-executable",
        "build_cuda_toolkit_version": "12.8",
        "build_cuda_architecture": "80",
    },
}


class ProgressReporter:
    def __init__(self, label, total):
        self.label = str(label)
        self.total = max(0, int(total))
        self.current = 0
        self.started = time.perf_counter()

    def tick(self, message):
        self.current += 1
        pct = (100.0 * self.current / self.total) if self.total > 0 else 100.0
        elapsed = time.perf_counter() - self.started
        print(
            f"[progress] {self.label} {self.current}/{self.total} "
            f"({pct:.1f}%, elapsed={elapsed:.0f}s) {message}",
            flush=True,
        )


def parse_csv_tokens(value):
    return [x.strip() for x in str(value).split(",") if x.strip()]


def timeout_too_long_note(timeout_seconds):
    return f"TIMEOUT_TOO_LONG: exceeded per-function limit ({int(timeout_seconds)}s)"


def timeout_after_results_note(timeout_seconds):
    return (
        "SUBPROCESS_TIMEOUT_AFTER_RESULT_JSON: subprocess exceeded "
        f"{int(timeout_seconds)}s after emitting parsed RESULT_JSON rows; "
        "preserved emitted metric"
    )


def should_continue_after_isolated_timeout(successful_e2e_seconds, timeout_seconds):
    """Continue repeats only when a timeout contradicts at least two fast calls."""
    valid = [
        float(value)
        for value in successful_e2e_seconds
        if value is not None and math.isfinite(float(value)) and float(value) >= 0.0
    ]
    if len(valid) < 2:
        return False
    return max(valid) <= float(timeout_seconds) / 10.0


def timeout_seconds_from_notes(notes):
    m = re.search(r"TIMEOUT_TOO_LONG: exceeded per-function limit \((\d+)s\)", str(notes))
    return float(m.group(1)) if m else float(PER_FUNCTION_TIMEOUT_SECONDS)


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


def _resolve_monitor_gpu_index(env):
    env_idx = env.get("EGGPU_MONITOR_GPU_INDEX", "").strip()
    if env_idx.isdigit():
        return int(env_idx)
    cvd = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        tok = cvd.split(",")[0].strip()
        if tok.isdigit():
            return int(tok)
    return 0


def _nvml_compute_processes(handle):
    if pynvml is None:
        return []
    for name in (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    ):
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
    if used < 0 or used >= (1 << 62):
        return 0
    return used


def _pid_is_alive(pid):
    if pid <= 0:
        return False
    if psutil is not None:
        return bool(psutil.pid_exists(pid))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_busy_override_enabled(env):
    return env.get("EGGPU_ALLOW_BUSY_GPU", env.get("ALLOW_BUSY_GPU", "")).strip().upper() in TRUE_VALUES


def visibility_marker_adjust_mb(env):
    if env.get("EGGPU_GPU_VISIBILITY_MARKER", "").strip().upper() not in TRUE_VALUES:
        return 0.0
    raw = env.get("EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB", "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, value)


def check_eggpu_child_gpu_idle(env):
    if gpu_busy_override_enabled(env):
        return True, "busy-GPU guard explicitly disabled; debug-only"
    if not _ensure_nvml():
        return False, "gpu_busy_before_eggpu_child: NVML unavailable; refusing to launch EGGPU timing row"
    max_mem_mb = float(env.get("EGGPU_IDLE_MAX_MEMORY_MB", "1024"))
    max_util = float(env.get("EGGPU_IDLE_MAX_UTILIZATION", "5"))
    retry_attempts = max(1, int(env.get("EGGPU_IDLE_RETRY_ATTEMPTS", "6")))
    retry_sleep_s = max(0.0, float(env.get("EGGPU_IDLE_RETRY_SLEEP_S", "0.5")))
    last_sample = None
    for attempt in range(1, retry_attempts + 1):
        try:
            gpu_index = _resolve_monitor_gpu_index(env)
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            raw_used_mb = int(info.used) / (1024.0 * 1024.0)
            marker_adjust_mb = visibility_marker_adjust_mb(env)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = int(getattr(util, "gpu", 0))
            procs = []
            allowed_process_memory_mb = 0.0
            marker_process_memory_mb = 0.0
            own_pid = os.getpid()
            allowed_pids = {own_pid}
            marker_owner_pid = -1
            if env.get("EGGPU_EXTERNAL_VISIBILITY_MARKER", "").strip().upper() in TRUE_VALUES:
                try:
                    marker_owner_pid = int(
                        env.get("EGGPU_GPU_VISIBILITY_MARKER_OWNER_PID", "-1")
                    )
                    allowed_pids.add(marker_owner_pid)
                except ValueError:
                    pass
            for proc_info in _nvml_compute_processes(handle):
                try:
                    pid = int(getattr(proc_info, "pid", -1))
                except Exception:
                    pid = -1
                mem_mb = _safe_used_gpu_memory_bytes(proc_info) / (1024.0 * 1024.0)
                if pid > 0 and pid in allowed_pids:
                    allowed_process_memory_mb += mem_mb
                    if pid == marker_owner_pid:
                        marker_process_memory_mb += mem_mb
                elif pid > 0:
                    # NVML can retain a just-exited CUDA context for a short
                    # interval.  Treat only live foreign PIDs as blockers;
                    # residual memory and utilization are still checked below.
                    if not _pid_is_alive(pid):
                        continue
                    pname = ""
                    try:
                        pname = psutil.Process(pid).name() if psutil is not None else ""
                    except Exception:
                        pname = ""
                    procs.append(f"pid={pid},name={pname},mem_mb={mem_mb:.1f}")
            marker_unattributed_adjust_mb = max(
                0.0, marker_adjust_mb - marker_process_memory_mb
            )
            used_mb = max(
                0.0,
                raw_used_mb
                - allowed_process_memory_mb
                - marker_unattributed_adjust_mb,
            )
        except Exception as exc:
            return False, f"gpu_busy_before_eggpu_child: unable to query GPU idleness: {exc}"

        last_sample = (
            gpu_index,
            used_mb,
            raw_used_mb,
            marker_adjust_mb,
            allowed_process_memory_mb,
            gpu_util,
            procs,
            attempt,
        )
        if not procs and used_mb <= max_mem_mb and gpu_util <= max_util:
            retry_note = "" if attempt == 1 else f", idle_retry_attempt={attempt}"
            return (
                True,
                "gpu_idle_before_eggpu_child: "
                f"gpu={gpu_index}, memory_mb={used_mb:.1f}, raw_memory_mb={raw_used_mb:.1f}, "
                f"allowed_process_memory_mb={allowed_process_memory_mb:.1f}, "
                f"visibility_marker_adjust_mb={marker_adjust_mb:.1f}, utilization={gpu_util}%"
                f"{retry_note}",
            )
        # NVML utilization and context memory can briefly lag after the previous
        # isolated child exits.  Retry only when no foreign compute PID exists;
        # a real process remains an immediate hard blocker.
        if procs or attempt >= retry_attempts:
            break
        if retry_sleep_s:
            time.sleep(retry_sleep_s)

    (
        gpu_index,
        used_mb,
        raw_used_mb,
        marker_adjust_mb,
        allowed_process_memory_mb,
        gpu_util,
        procs,
        attempt,
    ) = last_sample
    if procs or used_mb > max_mem_mb or gpu_util > max_util:
        proc_note = "; ".join(procs) if procs else "none"
        return (
            False,
            "gpu_busy_before_eggpu_child: "
            f"gpu={gpu_index}, memory_mb={used_mb:.1f}, raw_memory_mb={raw_used_mb:.1f}, "
            f"allowed_process_memory_mb={allowed_process_memory_mb:.1f}, "
            f"visibility_marker_adjust_mb={marker_adjust_mb:.1f}, utilization={gpu_util}%, "
            f"threshold_memory_mb={max_mem_mb:.1f}, threshold_utilization={max_util:.1f}%, "
            f"compute_processes=[{proc_note}], idle_retry_attempts={attempt}",
        )


def run_cmd(cmd, out_path, env, timeout=None, cooldown=0.0):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_idle_note = ""
    if env.get("EGGPU_REQUIRE_GPU_IDLE", "").strip().upper() in TRUE_VALUES:
        idle_ok, idle_note = check_eggpu_child_gpu_idle(env)
        if not idle_ok:
            out_path.write_text(idle_note + "\n")
            raise SystemExit(idle_note)
        preflight_idle_note = idle_note
    memory_monitor_enabled = (
        env.get("EGGPU_MEASUREMENT_MODE", "combined").strip().lower() != "timing"
    )
    monitor_poll_ms = max(
        2.0, float(env.get("EGGPU_MEMORY_POLL_MS", "2"))
    ) if memory_monitor_enabled else 50.0
    t0 = time.perf_counter()
    with out_path.open("w") as f:
        f.write("$ " + " ".join(str(x) for x in cmd) + "\n\n")
        if preflight_idle_note:
            f.write(preflight_idle_note + "\n\n")
        f.flush()
        p = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        start_rss = None
        peak_rss = 0
        rss_samples = 0
        pinfo = None
        if memory_monitor_enabled and psutil is not None:
            try:
                pinfo = psutil.Process(p.pid)
            except Exception:
                pinfo = None
        gpu_handle = None
        if memory_monitor_enabled and _ensure_nvml():
            try:
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(_resolve_monitor_gpu_index(env))
            except Exception:
                gpu_handle = None
        memory = {
            "rss_mb": None,
            "rss_start_mb": None,
            "rss_peak_delta_mb": None,
            "gpu_peak_mb": None,
            "gpu_avg_mb": None,
            "gpu_start_mb": None,
            "gpu_peak_delta_mb": None,
            "gpu_avg_delta_mb": None,
            "gpu_proc_peak_mb": None,
            "gpu_proc_avg_mb": None,
            "gpu_proc_start_mb": None,
            "gpu_proc_peak_delta_mb": None,
            "gpu_proc_avg_delta_mb": None,
            "gpu_non_process_start_mb": None,
            "monitor_rss_samples": 0,
            "monitor_gpu_samples": 0,
            "monitor_gpu_proc_samples": 0,
            "monitor_poll_ms": monitor_poll_ms,
            "monitor_window_seconds": None,
            "gpu_exclusive_preflight_verified": bool(preflight_idle_note),
            "gpu_exclusive_snapshot_before_worker": preflight_idle_note or None,
            "gpu_exclusive_postflight_verified": False,
            "gpu_exclusive_snapshot_after_worker": None,
        }
        monitor_started_at = time.perf_counter() if memory_monitor_enabled else None
        start_gpu_bytes = None
        gpu_peak_bytes = 0
        gpu_sum_bytes = 0
        gpu_samples = 0
        gpu_peak_delta_bytes = 0
        gpu_delta_sum_bytes = 0
        gpu_delta_samples = 0
        start_gpu_proc_bytes = None
        gpu_proc_peak_bytes = 0
        gpu_proc_sum_bytes = 0
        gpu_proc_samples = 0
        gpu_proc_peak_delta_bytes = 0
        gpu_proc_delta_sum_bytes = 0
        gpu_proc_delta_samples = 0

        def sample_peak_rss():
            nonlocal start_rss, peak_rss, rss_samples
            if pinfo is None:
                return
            rss = 0
            procs = [pinfo]
            try:
                procs.extend(pinfo.children(recursive=True))
            except Exception:
                pass
            for proc in procs:
                try:
                    rss += int(proc.memory_info().rss)
                except Exception:
                    pass
            rss_samples += 1
            if start_rss is None:
                start_rss = rss
            if rss > peak_rss:
                peak_rss = rss

        def process_tree_pids():
            if pinfo is None:
                return {p.pid}
            pids = {p.pid}
            try:
                pids.add(int(pinfo.pid))
            except Exception:
                pass
            try:
                for child in pinfo.children(recursive=True):
                    try:
                        pids.add(int(child.pid))
                    except Exception:
                        pass
            except Exception:
                pass
            return pids

        def sample_gpu_memory():
            nonlocal start_gpu_bytes, gpu_peak_bytes, gpu_sum_bytes, gpu_samples
            nonlocal gpu_peak_delta_bytes, gpu_delta_sum_bytes, gpu_delta_samples
            nonlocal start_gpu_proc_bytes, gpu_proc_peak_bytes, gpu_proc_sum_bytes, gpu_proc_samples
            nonlocal gpu_proc_peak_delta_bytes, gpu_proc_delta_sum_bytes, gpu_proc_delta_samples
            if gpu_handle is None:
                return
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                used = int(info.used)
                if start_gpu_bytes is None:
                    start_gpu_bytes = used
                gpu_peak_bytes = max(gpu_peak_bytes, used)
                gpu_sum_bytes += used
                gpu_samples += 1
                delta = max(0, used - start_gpu_bytes)
                gpu_peak_delta_bytes = max(gpu_peak_delta_bytes, delta)
                gpu_delta_sum_bytes += delta
                gpu_delta_samples += 1
            except Exception:
                pass
            try:
                pids = process_tree_pids()
                proc_used = 0
                for proc_info in _nvml_compute_processes(gpu_handle):
                    try:
                        pid = int(getattr(proc_info, "pid", -1))
                    except Exception:
                        pid = -1
                    if pid in pids:
                        proc_used += _safe_used_gpu_memory_bytes(proc_info)
                if start_gpu_proc_bytes is None:
                    start_gpu_proc_bytes = proc_used
                gpu_proc_peak_bytes = max(gpu_proc_peak_bytes, proc_used)
                gpu_proc_sum_bytes += proc_used
                gpu_proc_samples += 1
                proc_delta = max(0, proc_used - start_gpu_proc_bytes)
                gpu_proc_peak_delta_bytes = max(gpu_proc_peak_delta_bytes, proc_delta)
                gpu_proc_delta_sum_bytes += proc_delta
                gpu_proc_delta_samples += 1
            except Exception:
                pass

        def sample_memory():
            sample_peak_rss()
            sample_gpu_memory()

        deadline = None if timeout is None else (time.perf_counter() + float(timeout))
        timed_out = False
        while True:
            if memory_monitor_enabled:
                sample_memory()
            rc = p.poll()
            if rc is not None:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                f.write(f"\nTIMEOUT after {timeout} seconds\n")
                f.write(timeout_too_long_note(timeout) + "\n")
                f.flush()
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=5)
                except Exception:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait(timeout=5)
                    except Exception:
                        pass
                break
            time.sleep(monitor_poll_ms / 1000.0 if memory_monitor_enabled else 0.05)

        if memory_monitor_enabled:
            sample_memory()
        elapsed = time.perf_counter() - t0
        if cooldown and cooldown > 0:
            time.sleep(float(cooldown))
        peak_rss_mb = (peak_rss / (1024.0 * 1024.0)) if peak_rss > 0 else None
        memory["rss_mb"] = peak_rss_mb
        memory["rss_start_mb"] = (
            start_rss / (1024.0 * 1024.0) if start_rss is not None else None
        )
        memory["rss_peak_delta_mb"] = (
            max(0, peak_rss - start_rss) / (1024.0 * 1024.0)
            if start_rss is not None and peak_rss > 0
            else None
        )
        marker_adjust_mb = visibility_marker_adjust_mb(env)
        if gpu_samples > 0:
            memory["gpu_peak_mb"] = gpu_peak_bytes / (1024.0 * 1024.0)
            memory["gpu_avg_mb"] = (gpu_sum_bytes / gpu_samples) / (1024.0 * 1024.0)
            memory["gpu_start_mb"] = (
                start_gpu_bytes / (1024.0 * 1024.0)
                if start_gpu_bytes is not None
                else None
            )
            if marker_adjust_mb > 0:
                memory["gpu_peak_mb"] = max(0.0, memory["gpu_peak_mb"] - marker_adjust_mb)
                memory["gpu_avg_mb"] = max(0.0, memory["gpu_avg_mb"] - marker_adjust_mb)
                if memory["gpu_start_mb"] is not None:
                    memory["gpu_start_mb"] = max(
                        0.0, memory["gpu_start_mb"] - marker_adjust_mb
                    )
        if gpu_delta_samples > 0:
            memory["gpu_peak_delta_mb"] = gpu_peak_delta_bytes / (1024.0 * 1024.0)
            memory["gpu_avg_delta_mb"] = (gpu_delta_sum_bytes / gpu_delta_samples) / (1024.0 * 1024.0)
        if gpu_proc_samples > 0:
            memory["gpu_proc_peak_mb"] = gpu_proc_peak_bytes / (1024.0 * 1024.0)
            memory["gpu_proc_avg_mb"] = (gpu_proc_sum_bytes / gpu_proc_samples) / (1024.0 * 1024.0)
            memory["gpu_proc_start_mb"] = (
                start_gpu_proc_bytes / (1024.0 * 1024.0)
                if start_gpu_proc_bytes is not None
                else None
            )
        if gpu_proc_delta_samples > 0:
            memory["gpu_proc_peak_delta_mb"] = gpu_proc_peak_delta_bytes / (1024.0 * 1024.0)
            memory["gpu_proc_avg_delta_mb"] = (gpu_proc_delta_sum_bytes / gpu_proc_delta_samples) / (1024.0 * 1024.0)
        if memory["gpu_start_mb"] is not None and memory["gpu_proc_start_mb"] is not None:
            memory["gpu_non_process_start_mb"] = max(
                0.0, memory["gpu_start_mb"] - memory["gpu_proc_start_mb"]
            )
        memory["monitor_rss_samples"] = rss_samples
        memory["monitor_gpu_samples"] = gpu_samples
        memory["monitor_gpu_proc_samples"] = gpu_proc_samples
        if monitor_started_at is not None:
            memory["monitor_window_seconds"] = time.perf_counter() - monitor_started_at
        if env.get("EGGPU_REQUIRE_GPU_IDLE", "").strip().upper() in TRUE_VALUES:
            idle_ok, idle_note = check_eggpu_child_gpu_idle(env)
            memory["gpu_exclusive_postflight_verified"] = bool(idle_ok)
            memory["gpu_exclusive_snapshot_after_worker"] = idle_note
            f.write("\n" + idle_note + "\n")
            f.flush()
            if not idle_ok:
                raise SystemExit(idle_note)
        if timed_out:
            return 124, elapsed, memory
        return p.returncode, elapsed, memory


def sanitized_subprocess_env(base_env):
    env = dict(base_env)
    for key in SANITIZE_ENV_VARS:
        env.pop(key, None)
    return env


def local_cuda_root():
    for key in ("EGGPU_CUDA_ROOT", "CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT"):
        value = os.environ.get(key, "").strip()
        if value and Path(value).exists():
            return Path(value)
    if DEFAULT_LOCAL_CUDA_ROOT is not None and DEFAULT_LOCAL_CUDA_ROOT.exists():
        return DEFAULT_LOCAL_CUDA_ROOT
    return None


def cpp_easygraph_artifacts():
    repo = WORKSPACE_ROOT / "Easy-Graph"
    artifacts = []
    for pattern in ("cpp_easygraph*.so", "build/lib*/cpp_easygraph*.so"):
        for path in sorted(repo.glob(pattern)):
            try:
                st = path.stat()
            except OSError:
                continue
            artifacts.append(
                {
                    "path": str(path.resolve()),
                    "relative_path": str(path.relative_to(WORKSPACE_ROOT)),
                    "size_bytes": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return artifacts


def active_cpp_easygraph_artifact():
    """Fingerprint the extension resolved by this benchmark environment."""

    try:
        spec = importlib.util.find_spec("cpp_easygraph")
    except (ImportError, AttributeError, ValueError):
        spec = None
    origin = getattr(spec, "origin", "") if spec is not None else ""
    if not origin:
        return {}
    path = Path(origin)
    if not path.is_file():
        return {"path": str(path), "sha256": "", "size_bytes": 0}
    try:
        payload = path.read_bytes()
        stat = path.stat()
    except OSError:
        return {"path": str(path), "sha256": "", "size_bytes": 0}
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _iter_source_candidates(root, allowed_suffixes, ignored_parts):
    """Yield source files while pruning generated trees before traversal."""

    if not root.exists():
        return
    for directory, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [name for name in dirnames if name not in ignored_parts]
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            if filename == "CMakeLists.txt" or path.suffix.lower() in allowed_suffixes:
                yield path


def collect_source_snapshot():
    """Fingerprint the executable benchmark sources without requiring Git.

    Shared experiment hosts do not always install the ``git`` executable even
    when the synchronized workspace contains ``.git`` metadata.  A content
    digest is therefore the portable provenance anchor.  Generated results,
    datasets, build trees, and binaries are deliberately excluded; compiled
    artifacts are fingerprinted separately by ``cpp_easygraph_artifacts`` and
    ``gunrock_executable_artifacts``.
    """

    source_roots = (
        WORKSPACE_ROOT / "Easy-Graph" / "easygraph",
        WORKSPACE_ROOT / "Easy-Graph" / "cpp_easygraph",
        WORKSPACE_ROOT / "Easy-Graph" / "gpu_easygraph",
        WORKSPACE_ROOT / "EG_Evaluation" / "benchmarking",
        WORKSPACE_ROOT / "scripts",
    )
    top_level_files = (
        WORKSPACE_ROOT / "Easy-Graph" / "setup.py",
        WORKSPACE_ROOT / "Easy-Graph" / "CMakeLists.txt",
        WORKSPACE_ROOT / "EG_Evaluation" / "run_main_and_ablation.sh",
        WORKSPACE_ROOT / "EG_Evaluation" / "run_final_gpu5_gate_and_full.sh",
        WORKSPACE_ROOT / "EG_Evaluation" / "run_complete_paper_experiments.sh",
    )
    allowed_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".json",
        ".py",
        ".sh",
        ".txt",
    }
    ignored_parts = {
        ".git",
        "__pycache__",
        "build",
        "datasets",
        "results",
    }

    candidates = set(top_level_files)
    for root in source_roots:
        candidates.update(
            _iter_source_candidates(root, allowed_suffixes, ignored_parts)
        )

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(candidates, key=lambda p: str(p)):
        if not path.exists() or not path.is_file():
            continue
        try:
            relative = path.resolve().relative_to(WORKSPACE_ROOT.resolve())
            payload = path.read_bytes()
        except (OSError, ValueError):
            continue
        relative_bytes = str(relative).encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        file_count += 1
        total_bytes += len(payload)

    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest() if file_count else "",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "scope": [
            "Easy-Graph/easygraph",
            "Easy-Graph/cpp_easygraph",
            "Easy-Graph/gpu_easygraph",
            "Easy-Graph/setup.py",
            "EG_Evaluation/benchmarking",
            "EG_Evaluation/run_main_and_ablation.sh",
            "EG_Evaluation/run_final_gpu5_gate_and_full.sh",
            "scripts",
        ],
    }


def collect_implementation_source_snapshot():
    """Fingerprint only sources that can change EGGPU execution semantics."""

    source_roots = (
        WORKSPACE_ROOT / "Easy-Graph" / "easygraph",
        WORKSPACE_ROOT / "Easy-Graph" / "cpp_easygraph",
        WORKSPACE_ROOT / "Easy-Graph" / "gpu_easygraph",
    )
    top_level_files = (
        WORKSPACE_ROOT / "Easy-Graph" / "setup.py",
        WORKSPACE_ROOT / "Easy-Graph" / "CMakeLists.txt",
    )
    allowed_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".json",
        ".py",
        ".txt",
    }
    ignored_parts = {".git", "__pycache__", "build", "datasets", "results"}

    candidates = set(top_level_files)
    for root in source_roots:
        candidates.update(
            _iter_source_candidates(root, allowed_suffixes, ignored_parts)
        )

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    latest_mtime_epoch = 0.0
    for path in sorted(candidates, key=lambda p: str(p)):
        if not path.exists() or not path.is_file():
            continue
        try:
            relative = path.resolve().relative_to(WORKSPACE_ROOT.resolve())
            payload = path.read_bytes()
            latest_mtime_epoch = max(latest_mtime_epoch, path.stat().st_mtime)
        except (OSError, ValueError):
            continue
        relative_bytes = str(relative).encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        file_count += 1
        total_bytes += len(payload)

    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest() if file_count else "",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "latest_mtime_epoch": latest_mtime_epoch or None,
        "latest_mtime": (
            datetime.fromtimestamp(latest_mtime_epoch).isoformat(timespec="seconds")
            if latest_mtime_epoch
            else ""
        ),
        "scope": [
            "Easy-Graph/easygraph",
            "Easy-Graph/cpp_easygraph",
            "Easy-Graph/gpu_easygraph",
            "Easy-Graph/setup.py",
            "Easy-Graph/CMakeLists.txt",
        ],
    }


def collect_runtime_python_snapshot(easygraph_repo):
    """Fingerprint the Python package loaded beside the candidate extension.

    The compiled extension hash does not freeze Python dispatch and result
    reconstruction.  Record those files independently so a paper-facing run
    cannot silently follow a mutable ``easygraph`` symlink.
    """

    repo_root = Path(easygraph_repo).expanduser().resolve()
    package_path = Path(easygraph_repo).expanduser() / "easygraph"
    package_resolved = package_path.resolve()
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    if package_resolved.is_dir():
        for directory, dirnames, filenames in os.walk(package_resolved, topdown=True):
            dirnames[:] = [
                name for name in dirnames
                if name not in {".git", "__pycache__", "build", "results"}
            ]
            base = Path(directory)
            for filename in sorted(filenames):
                path = base / filename
                if path.suffix.lower() not in {".py", ".json", ".txt"}:
                    continue
                try:
                    relative = path.relative_to(package_resolved)
                    payload = path.read_bytes()
                except (OSError, ValueError):
                    continue
                relative_bytes = str(relative).encode("utf-8")
                digest.update(len(relative_bytes).to_bytes(8, "big"))
                digest.update(relative_bytes)
                digest.update(len(payload).to_bytes(8, "big"))
                digest.update(payload)
                file_count += 1
                total_bytes += len(payload)
    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest() if file_count else "",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "runtime_root": str(repo_root),
        "package_path": str(package_path.absolute()),
        "package_resolved_path": str(package_resolved),
        "package_is_symlink": package_path.is_symlink(),
        "package_symlink_target": (
            os.readlink(package_path) if package_path.is_symlink() else ""
        ),
        "scope": ["easygraph/**/*.py", "easygraph/**/*.json", "easygraph/**/*.txt"],
    }


def collect_run_metadata(args, out_dir, datasets, selected_functions, env):
    cuda_root = local_cuda_root()
    with temporary_import_root(args.easygraph_repo):
        baseline_versions = collect_python_baseline_versions()
        active_cpp_artifact = active_cpp_easygraph_artifact()
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "EGGPU_MONITOR_GPU_INDEX",
        "EGGPU_CUDA_ROOT",
        "CUDA_PATH",
        "CUDA_HOME",
        "CUPY_CUDA_PATH",
        "CUDAToolkit_ROOT",
        "CONDA_PREFIX",
        "EASYGRAPH_ENABLE_GPU",
        "EASYGRAPH_GPU_STRICT_ERRORS",
        "EGGPU_GPU_VISIBILITY_MARKER",
        "EGGPU_GPU_VISIBILITY_MARKER_MB",
        "EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB",
        "EGGPU_GPU_VISIBILITY_MARKER_OWNER_PID",
        "EGGPU_GPU_VISIBILITY_MARKER_ALLOCATED_MB",
        "EGGPU_EXTERNAL_VISIBILITY_MARKER",
        "EGGPU_MEASUREMENT_MODE",
        "EGGPU_EXECUTION_PROTOCOL",
        "EGGPU_STABLE_TIMING_PROTOCOL",
        "EGGPU_EXPECTED_CPU_AFFINITY",
        "EGGPU_EXPECTED_NUMA_NODES",
        "EGGPU_AFFINITY_LAUNCHER",
        "EGGPU_AFFINITY_LAUNCH_RECORD",
        "EGGPU_NUMA_BINDING_METHOD",
        "EGGPU_INTERNAL_MEMORY_MONITOR",
        "EGGPU_MEMORY_POLL_MS",
        "EGGPU_MEMORY_PROBE_MIN_SECONDS",
        "EGGPU_MEMORY_PROBE_MAX_CALLS",
        "EASYGRAPH_GPU_ADAPTIVE_POLICY",
        "EASYGRAPH_GPU_DEVICE_CSR_CACHE_MAX_ENTRIES",
        "EASYGRAPH_GPU_DEVICE_CSR_CACHE_MAX_BYTES",
        "EASYGRAPH_GPU_COMPONENT_DENSE_RETURN",
        "EASYGRAPH_GPU_SCC_ACTIVE_TRIM",
        "EASYGRAPH_GPU_SCC_ACTIVE_TRIM_MAX_ITERS",
        "EASYGRAPH_GPU_SCC_DEGREE_PIVOT",
        "EASYGRAPH_GPU_SCC_HOST_ENABLE",
        "EASYGRAPH_GPU_SCC_PROFILE",
        "EASYGRAPH_GPU_KCORE_HOST_ENABLE",
        "EASYGRAPH_GPU_SSSP_HOST_ENABLE",
        "EASYGRAPH_GPU_SSSP_FRONTIER",
        "EASYGRAPH_GPU_SSSP_FRONTIER_MIN_EDGES",
        "EASYGRAPH_GPU_SSSP_FRONTIER_MIN_VERTICES",
        "EASYGRAPH_GPU_SSSP_FRONTIER_MAX_AVG_DEGREE",
        "EASYGRAPH_GPU_SSSP_FRONTIER_MAX_DEGREE",
        "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_AVG_DEGREE",
        "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE",
        "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS",
        "EASYGRAPH_GPU_BC_WARP_SIZE",
        "EASYGRAPH_GPU_BC_UNWEIGHTED_BFS",
        "EASYGRAPH_GPU_BC_MAX_CONCURRENT_SOURCES",
        "EASYGRAPH_GPU_BC_WORKSPACE_FRACTION",
        "EASYGRAPH_GPU_CLOSENESS_UNWEIGHTED_BFS",
        "EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION",
        "EGGPU_CLOSENESS_EXACT_MAX_NODES",
        "EGGPU_CLOSENESS_EXACT_MAX_WORK",
        "EGGPU_USE_CONDA_RUN",
        "EGGPU_CHILD_PYTHON",
        "COMMON_PY",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_PROC_BIND",
        "PYTHONHASHSEED",
        "MALLOC_ARENA_MAX",
    ]
    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_dir": str(Path(out_dir).resolve()),
        "workspace_root": str(WORKSPACE_ROOT.resolve()),
        "easygraph_repo": str(Path(args.easygraph_repo).resolve()),
        "argv": list(sys.argv),
        "python": {
            "executable": sys.executable,
            "direct_child_python": DIRECT_CHILD_PYTHON,
            "version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
        },
        "source_snapshot": collect_source_snapshot(),
        "implementation_source_snapshot": collect_implementation_source_snapshot(),
        "runtime_python_snapshot": collect_runtime_python_snapshot(args.easygraph_repo),
        "cuda": {
            "local_cuda_root": str(cuda_root) if cuda_root is not None else "",
            "nvcc": str(cuda_root / "bin" / "nvcc") if cuda_root is not None else "",
        },
        "gpu_device_profile": collect_gpu_device_profile(args.gpu, env),
        "host_profile": collect_host_profile(),
        "baseline_versions": baseline_versions,
        "build_artifacts": {
            "cpp_easygraph": cpp_easygraph_artifacts(),
            "active_cpp_easygraph": active_cpp_artifact,
            "gunrock_executables": gunrock_executable_artifacts(),
        },
        "benchmark_args": {
            "gpu": args.gpu,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "easygraph_warmup": args.easygraph_warmup,
            "nx_cugraph_warmup": args.nx_cugraph_warmup,
            "library_timeout": args.library_timeout,
            "inter_run_cooldown": args.inter_run_cooldown,
            "pr_alpha": args.pr_alpha,
            "pr_eps": args.pr_eps,
            "pr_max_iter": args.pr_max_iter,
            "sssp_sources": args.sssp_sources,
            "bc_sources": args.bc_sources,
            "closeness_sources": args.closeness_sources,
            "measurement_mode": args.measurement_mode,
            "eggpu_execution_protocol": args.eggpu_execution_protocol,
            "eggpu_stable_timing_protocol": bool(
                getattr(args, "eggpu_stable_timing_protocol", False)
            ),
            "baselines": list(args.selected_baselines),
            "datasets": [name for _, _, name, _ in datasets],
            "functions": list(selected_functions),
        },
        "environment": {key: env.get(key, os.environ.get(key, "")) for key in env_keys},
        "stable_timing_protocol": getattr(
            args, "stable_timing_protocol_metadata", {}
        ),
    }


def write_run_metadata(out_dir, metadata):
    (Path(out_dir) / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def write_baseline_version_manifest(out_dir, metadata):
    manifest = {
        "paper_repo_source_snapshot": metadata.get("source_snapshot", {}),
        "python_baselines": metadata.get("baseline_versions", {}),
        "gunrock_executables": (metadata.get("build_artifacts") or {}).get(
            "gunrock_executables", {}
        ),
    }
    (Path(out_dir) / "baseline_versions.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def conda_run_prefix():
    use_conda_run = os.environ.get("EGGPU_USE_CONDA_RUN", "").strip().upper() in {
        "1",
        "TRUE",
        "YES",
        "ON",
    }
    if not use_conda_run:
        return [DIRECT_CHILD_PYTHON]

    cmd = [CONDA_EXE, "run", "-n", "EGGPU", "env"]
    for key in SANITIZE_ENV_VARS:
        cmd.extend(["-u", key])
    cuda_root = local_cuda_root()
    if cuda_root is not None:
        # CuPy installed from conda picks CUDA headers from $CONDA_PREFIX/targets
        # for NVRTC/JIT, ignoring CUDA_PATH for that include-dir decision.  Set
        # these inside the `conda run ... env` command so CUDA/JIT baselines use
        # the same local toolkit as EGGPU without touching global CUDA.
        root = str(cuda_root)
        cmd.extend(
            [
                f"EGGPU_CUDA_ROOT={root}",
                f"CUDA_PATH={root}",
                f"CUDA_HOME={root}",
                f"CUPY_CUDA_PATH={root}",
                f"CUDAToolkit_ROOT={root}",
                f"CONDA_PREFIX={root}",
            ]
        )
    return cmd


def conda_python_cmd(script, *script_args):
    use_conda_run = os.environ.get("EGGPU_USE_CONDA_RUN", "").strip().upper() in {
        "1",
        "TRUE",
        "YES",
        "ON",
    }
    if use_conda_run:
        return [*conda_run_prefix(), "python", str(script), *[str(x) for x in script_args]]
    return [DIRECT_CHILD_PYTHON, str(script), *[str(x) for x in script_args]]


def read_text(path):
    try:
        return Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return ""


def first_float(pattern, text):
    m = re.search(pattern, text, re.MULTILINE)
    return float(m.group(1)) if m else None


def first_int(pattern, text):
    m = re.search(pattern, text, re.MULTILINE)
    return int(m.group(1).replace(",", "")) if m else None


def deterministic_sources(n, k):
    n = max(0, int(n))
    k = max(0, int(k))
    if n <= 0 or k <= 0:
        return []
    step = max(1, n // k)
    out = list(range(0, n, step))[:k]
    if len(out) < k:
        seen = set(out)
        tail = n - 1
        while len(out) < k and tail >= 0:
            if tail not in seen:
                out.append(tail)
                seen.add(tail)
            tail -= 1
    return out


def matrix_market_n(path):
    with Path(path).open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                return int(parts[0])
    raise RuntimeError(f"failed to parse MatrixMarket dimensions: {path}")


def dataset_stats(path):
    nodes = set()
    rows = 0
    loops = 0
    canon = set()
    directed_unique = set()
    with (ROOT / path).open() as f:
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
            rows += 1
            nodes.add(u)
            nodes.add(v)
            if u == v:
                loops += 1
                continue
            directed_unique.add((u, v))
            if u > v:
                u, v = v, u
            canon.add((u, v))
    return {
        "nodes_raw": len(nodes),
        "edge_rows": rows,
        "edge_rows_no_selfloops": rows - loops,
        "edges_directed_unique": len(directed_unique),
        "selfloops": loops,
        "edges_undirected_unique": len(canon),
        "bytes": (ROOT / path).stat().st_size,
    }


def add_row(rows, dataset_size, graph_type, dataset_name, function, baseline, metric, seconds,
            status, log, correctness="", notes="", extra=None):
    if "TIMEOUT_TOO_LONG" in str(notes):
        status = "timeout"
        if seconds is None and metric in ("e2e", "kernel"):
            seconds = timeout_seconds_from_notes(notes)
    structured = {
        "semantic": "exact_all_node" if function == "Closeness" else "",
        "skip_reason": "",
        "estimator_kind": "",
        "sample_sources": "",
        "source_policy": "",
        "source_seed": "",
        "source_nodes_sha": "",
        "sample_index": "",
        "sample_count": "",
        "measurement_phase": os.environ.get("EGGPU_MEASUREMENT_MODE", "combined"),
        "external_cli_wall_seconds": "",
    }
    structured.update(extra or {})
    descriptor = describe_metric(metric, baseline=baseline)
    formatted_value = "" if seconds is None else f"{seconds:.9g}"
    rows.append({
        "dataset_size": dataset_size,
        "graph_type": graph_type,
        "dataset": dataset_name,
        "function": function,
        "baseline": baseline,
        "metric": metric,
        "seconds": formatted_value,
        "value": formatted_value,
        **descriptor,
        "status": status,
        "correctness": correctness,
        "log": str(log),
        "notes": notes,
        **structured,
    })


def mark_sample_rows(rows, start_index, sample_index, sample_count):
    for row in rows[start_index:]:
        row["sample_index"] = str(sample_index)
        row["sample_count"] = str(sample_count)


def run_repeated_gunrock(
    rows,
    repeat,
    ds_dir,
    function,
    progress,
    run_once,
):
    if not RUN_GUNROCK_BASELINE:
        return
    repeat = max(1, int(repeat))
    for sample_index in range(1, repeat + 1):
        if progress is not None:
            progress("Gunrock", function, sample_index, repeat)
        sample_dir = ds_dir / "gunrock_samples" / f"{function.lower()}_r{sample_index}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        start_index = len(rows)
        run_once(sample_dir)
        mark_sample_rows(rows, start_index, sample_index, repeat)
        statuses = {str(row.get("status", "failed")) for row in rows[start_index:]}
        if statuses and statuses <= {"skipped", "unsupported"}:
            for omitted in range(sample_index + 1, repeat + 1):
                if progress is not None:
                    progress("Gunrock", function, omitted, repeat)
            break
        if "timeout" in statuses:
            for omitted in range(sample_index + 1, repeat + 1):
                if progress is not None:
                    progress("Gunrock", function, omitted, repeat)
            break


def add_memory_metric_rows(
    rows,
    dataset_size,
    graph_type,
    dataset_name,
    function,
    baseline,
    log,
    peak_rss_mb,
    status="ok",
    correctness="",
    notes="",
    measurement_window=None,
    extra=None,
):
    if isinstance(peak_rss_mb, dict):
        memory = peak_rss_mb
    else:
        memory = {"rss_mb": peak_rss_mb}

    def add_mem(metric, value, metric_note):
        if value is None:
            return
        row_extra = dict(extra or {})
        if measurement_window:
            row_extra["measurement_window"] = measurement_window
        add_row(
            rows,
            dataset_size,
            graph_type,
            dataset_name,
            function,
            baseline,
            metric,
            value,
            status,
            log,
            correctness=correctness,
            notes=(notes + "; " + metric_note if notes else metric_note),
            extra=row_extra or None,
        )

    add_mem("memory_peak_rss_mb", memory.get("rss_mb"), "process peak RSS memory (MB)")
    add_mem("memory_start_rss_mb", memory.get("rss_start_mb"), "process-tree RSS at subprocess-window start (MB)")
    add_mem(
        "memory_peak_rss_delta_mb",
        memory.get("rss_peak_delta_mb"),
        "process-tree peak RSS delta from subprocess-window start (MB)",
    )
    add_mem("memory_peak_gpu_mb", memory.get("gpu_peak_mb"), "device peak GPU memory during subprocess window (MB)")
    add_mem("memory_avg_gpu_mb", memory.get("gpu_avg_mb"), "device average GPU memory during subprocess window (MB)")
    add_mem(
        "memory_start_gpu_mb",
        memory.get("gpu_start_mb"),
        "whole-device GPU memory at subprocess-window start (MB; diagnostic only)",
    )
    add_mem(
        "memory_peak_gpu_delta_mb",
        memory.get("gpu_peak_delta_mb"),
        "device peak GPU memory delta from subprocess-start baseline (MB)",
    )
    add_mem(
        "memory_avg_gpu_delta_mb",
        memory.get("gpu_avg_delta_mb"),
        "device average GPU memory delta from subprocess-start baseline (MB)",
    )
    add_mem(
        "memory_peak_gpu_proc_mb",
        memory.get("gpu_proc_peak_mb"),
        "benchmark process-tree peak GPU memory during subprocess window (MB)",
    )
    add_mem(
        "memory_avg_gpu_proc_mb",
        memory.get("gpu_proc_avg_mb"),
        "benchmark process-tree average GPU memory during subprocess window (MB)",
    )
    add_mem(
        "memory_start_gpu_proc_mb",
        memory.get("gpu_proc_start_mb"),
        "benchmark process-tree GPU memory at subprocess-window start (MB)",
    )
    add_mem(
        "memory_peak_gpu_proc_delta_mb",
        memory.get("gpu_proc_peak_delta_mb"),
        "benchmark process-tree peak GPU memory delta from subprocess-start baseline (MB)",
    )
    add_mem(
        "memory_avg_gpu_proc_delta_mb",
        memory.get("gpu_proc_avg_delta_mb"),
        "benchmark process-tree average GPU memory delta from subprocess-start baseline (MB)",
    )
    add_mem(
        "memory_start_gpu_non_process_mb",
        memory.get("gpu_non_process_start_mb"),
        "derived non-benchmark GPU memory at subprocess-window start (MB)",
    )
    add_mem(
        "memory_monitor_rss_samples",
        memory.get("monitor_rss_samples"),
        "RSS sample count in subprocess window",
    )
    add_mem(
        "memory_monitor_gpu_samples",
        memory.get("monitor_gpu_samples"),
        "whole-device NVML sample count in subprocess window",
    )
    add_mem(
        "memory_monitor_gpu_proc_samples",
        memory.get("monitor_gpu_proc_samples"),
        "process-tree NVML sample count in subprocess window",
    )
    add_mem(
        "memory_monitor_poll_ms",
        memory.get("monitor_poll_ms"),
        "configured memory-monitor polling interval (ms)",
    )
    add_mem(
        "memory_monitor_window_seconds",
        memory.get("monitor_window_seconds"),
        "observed memory-monitor window (s)",
    )
    if all(v is None for v in memory.values()):
        return


def combine_memory_metrics(metrics):
    metrics = [m for m in metrics if isinstance(m, dict)]
    if not metrics:
        return None
    out = {}
    for key in (
        "rss_mb",
        "rss_peak_delta_mb",
        "gpu_peak_mb",
        "gpu_peak_delta_mb",
        "gpu_proc_peak_mb",
        "gpu_proc_peak_delta_mb",
    ):
        vals = [m.get(key) for m in metrics if m.get(key) is not None]
        out[key] = max(vals) if vals else None
    for key in (
        "rss_start_mb",
        "gpu_avg_mb",
        "gpu_start_mb",
        "gpu_avg_delta_mb",
        "gpu_proc_avg_mb",
        "gpu_proc_start_mb",
        "gpu_proc_avg_delta_mb",
        "gpu_non_process_start_mb",
    ):
        vals = [m.get(key) for m in metrics if m.get(key) is not None]
        out[key] = (sum(vals) / len(vals)) if vals else None
    for key in (
        "monitor_rss_samples",
        "monitor_gpu_samples",
        "monitor_gpu_proc_samples",
    ):
        vals = [m.get(key) for m in metrics if m.get(key) is not None]
        out[key] = sum(vals) if vals else None
    poll_values = [m.get("monitor_poll_ms") for m in metrics if m.get("monitor_poll_ms") is not None]
    out["monitor_poll_ms"] = max(poll_values) if poll_values else None
    windows = [m.get("monitor_window_seconds") for m in metrics if m.get("monitor_window_seconds") is not None]
    out["monitor_window_seconds"] = sum(windows) if windows else None
    return out


def path_for_cmd(path):
    path = Path(path)
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def gunrock_bin_candidates():
    env_paths = os.environ.get("EG_GUNROCK_BIN_PATHS", "").strip()
    env_override = os.environ.get("EG_GUNROCK_BIN") or os.environ.get("GUNROCK_BIN")
    candidates = []
    if env_paths:
        candidates.extend(
            Path(value).expanduser()
            for value in env_paths.split(os.pathsep)
            if value.strip()
        )
    if env_override:
        candidates.append(Path(env_override).expanduser())
    local_candidates = [
        ROOT / "gunrock_latest" / "build_cuda132_a100_migrated" / "bin",
        ROOT / "gunrock_latest" / "build_cuda132_a100_migrated",
        ROOT / "gunrock_latest" / "build_cuda131_a100" / "bin",
        ROOT / "gunrock_legacy_master" / "build_lcc_result_cuda128_clean" / "bin",
        ROOT / "gunrock_legacy_master" / "build_overlay_cuda128_cc_lcc" / "bin",
        ROOT / "gunrock_legacy_master" / "build_cc_only_cuda128" / "bin",
        ROOT / "gunrock_legacy_v12_clean" / "build_overlay_cuda128_cc_lcc" / "bin",
        ROOT / "gunrock_legacy_v12_clean" / "build_cc_only_cuda128" / "bin",
        ROOT / "gunrock_legacy_master" / "build_legacy_a100_cc_lcc_overlay" / "bin",
        ROOT / "gunrock_legacy_master" / "build_legacy_a100_cc_lcc" / "bin",
        ROOT / "gunrock_latest" / "build" / "bin",
        ROOT / "gunrock_latest" / "build_cuda131_a100",
        ROOT / "gunrock_legacy_master" / "build_overlay_cuda128_cc_lcc",
        ROOT / "gunrock_legacy_v12_clean" / "build_overlay_cuda128_cc_lcc",
        ROOT / "build_cuda124" / "bin",
    ]
    legacy_eval_root = ROOT.parent.parent / "EG_Evaluation"
    sibling_candidates = [
        legacy_eval_root / "gunrock_latest" / "build_cuda132_a100_migrated" / "bin",
        legacy_eval_root / "gunrock_latest" / "build_cuda132_a100_migrated",
        legacy_eval_root / "gunrock_legacy_master" / "build_lcc_result_cuda128_clean" / "bin",
        legacy_eval_root / "gunrock_legacy_master" / "build_overlay_cuda128_cc_lcc" / "bin",
        legacy_eval_root / "gunrock_legacy_master" / "build_overlay_cuda128_cc_lcc",
        legacy_eval_root / "gunrock_latest" / "build" / "bin",
    ]
    seen = set()
    ordered = []
    for candidate in [*candidates, *local_candidates, *sibling_candidates]:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def find_gunrock_exe(exe_name):
    for candidate in gunrock_bin_candidates():
        if candidate.is_file() and candidate.name == exe_name and os.access(candidate, os.X_OK):
            return candidate
        exe = candidate / exe_name
        if exe.exists() and os.access(exe, os.X_OK):
            return exe
        # Some Gunrock builds place executables under nested per-app dirs.
        if candidate.exists() and candidate.is_dir():
            for sub in candidate.rglob(exe_name):
                if sub.is_file() and os.access(sub, os.X_OK):
                    return sub
    return None


def _gunrock_manifest_provenance(executable, binary_sha256, executable_path):
    try:
        manifest_bytes = GUNROCK_ARTIFACT_MANIFEST.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"unable to read {GUNROCK_ARTIFACT_MANIFEST}: {exc}"

    if manifest.get("schema_version") != 1:
        return {}, "unsupported Gunrock artifact manifest schema"
    executable_entry = manifest.get("executables", {}).get(executable)
    if not isinstance(executable_entry, dict):
        return {}, f"manifest has no executable entry for {executable}"
    expected_sha256 = str(executable_entry.get("sha256", ""))
    if not expected_sha256 or expected_sha256 != binary_sha256:
        return {}, (
            f"binary SHA-256 mismatch for {executable}: "
            f"expected={expected_sha256 or '<missing>'}, actual={binary_sha256}"
        )

    source_id = str(executable_entry.get("source", ""))
    source = manifest.get("sources", {}).get(source_id)
    if not isinstance(source, dict):
        return {}, f"manifest has no source entry for {source_id or '<missing>'}"
    required = ("source_remote", "source_commit", "source_version", "source_diff_sha256")
    missing = [field for field in required if not source.get(field)]
    if missing:
        return {}, f"manifest source {source_id} is missing {','.join(missing)}"

    source_root = None
    directory_name = str(source.get("source_directory_name", ""))
    if directory_name:
        for candidate in (executable_path.parent, *executable_path.parents):
            if candidate.name == directory_name:
                source_root = candidate
                break
    provenance = {
        key: source.get(key)
        for key in (
            "source_remote",
            "source_commit",
            "source_version",
            "source_tracked_dirty",
            "source_diff_sha256",
            "source_modified_files",
            "source_modified_files_latest_mtime_epoch",
        )
    }
    relevant_files = executable_entry.get("source_relevant_modified_files")
    relevant_mtime = executable_entry.get(
        "source_relevant_files_latest_mtime_epoch"
    )
    relevant_hashes = executable_entry.get("source_relevant_file_sha256")
    if isinstance(relevant_files, list):
        provenance["source_relevant_modified_files"] = relevant_files
        provenance["source_modified_files"] = relevant_files
    if relevant_mtime is not None:
        provenance["source_relevant_files_latest_mtime_epoch"] = relevant_mtime
        provenance["source_modified_files_latest_mtime_epoch"] = relevant_mtime
    if isinstance(relevant_hashes, dict):
        provenance["source_relevant_file_sha256"] = relevant_hashes
    if source_root is not None:
        provenance["source_root"] = str(source_root.resolve())
    provenance.update(
        {
            "source_provenance_origin": "pinned-binary-manifest",
            "source_manifest_path": str(GUNROCK_ARTIFACT_MANIFEST.resolve()),
            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_manifest_source_id": source_id,
            "source_manifest_binary_sha256_match": True,
            "source_manifest_source_match": True,
            "source_provenance_errors": [],
        }
    )
    return provenance, ""


def gunrock_executable_artifacts(executable_names=("pr", "mst", "lcc", "bfs", "sssp", "kcore", "bc")):
    artifacts = {}
    for name in executable_names:
        path = find_gunrock_exe(name)
        if path is None:
            artifacts[name] = {"status": "missing"}
            continue
        resolved = path.resolve()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = resolved.stat()
        binary_sha256 = digest.hexdigest()
        manifest_provenance, manifest_error = _gunrock_manifest_provenance(
            name, binary_sha256, resolved
        )
        if manifest_provenance:
            source_provenance = dict(manifest_provenance)
            source_provenance["runtime_git_validation"] = False
            source_provenance["provenance_contract"] = (
                "binary_sha256_bound_pinned_manifest"
            )
        else:
            source_provenance = {
                "source_provenance_origin": "unavailable",
                "runtime_git_validation": False,
                "provenance_contract": "binary_sha256_bound_pinned_manifest",
            }
            source_provenance["source_manifest_binary_sha256_match"] = False
            source_provenance["source_manifest_source_match"] = False
            source_provenance["source_provenance_errors"] = [
                {
                    "operation": "artifact-manifest",
                    "returncode": 1,
                    "error": manifest_error,
                }
            ]
        changed_source_mtime = source_provenance.get(
            "source_modified_files_latest_mtime_epoch"
        )
        artifacts[name] = {
            "status": "available",
            "path": str(resolved),
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "sha256": binary_sha256,
            **GUNROCK_APPLICATION_PROVENANCE.get(name, {}),
            **source_provenance,
            "binary_not_older_than_latest_tracked_source_change": (
                stat.st_mtime >= changed_source_mtime
                if changed_source_mtime is not None
                else None
            ),
        }
    return artifacts


def gunrock_search_note():
    return "searched " + ", ".join(str(path_for_cmd(p)) for p in gunrock_bin_candidates())


def parse_pagerank(text):
    mine_kernel = first_float(r"\[kernel\]\s+mine=([0-9.eE+-]+)s", text)
    cugraph_kernel = first_float(r"\[kernel\]\s+mine=[0-9.eE+-]+s\s+cuGraph=([0-9.eE+-]+)s", text)
    mine_e2e = first_float(r"\[e2e\s+\]\s+mine=([0-9.eE+-]+)s", text)
    cugraph_e2e = first_float(r"\[e2e\s+\]\s+mine=[0-9.eE+-]+s\s+cuGraph=([0-9.eE+-]+)s", text)
    mae = first_float(r"MAE=([0-9.eE+-]+)", text)
    return mine_kernel, cugraph_kernel, mine_e2e, cugraph_e2e, mae


def parse_mst(text):
    mine = first_float(r"B \(FULL.*?\):\s+([0-9.eE+-]+) s", text)
    cugraph = first_float(r"cuGraph \(FULL.*?\):\s+([0-9.eE+-]+) s", text)
    igraph = first_float(r"igraph:\s+([0-9.eE+-]+) s", text)
    ok = "Weights equal (B vs cuGraph)? True" in text
    return mine, cugraph, igraph, ok


def parse_lcc(text):
    mine_kernel = first_float(r"\[mine\s+\].*?kernel=([0-9.eE+-]+)s", text)
    mine_e2e = first_float(r"\[mine\s+\].*?e2e=([0-9.eE+-]+)s", text)
    cg_nr_kernel = first_float(r"\[cuGraph no-renum\].*?kernel=([0-9.eE+-]+)s", text)
    cg_nr_e2e = first_float(r"\[cuGraph no-renum\].*?e2e=([0-9.eE+-]+)s", text)
    cg_r_kernel = first_float(r"\[cuGraph\s+renum\s+\].*?kernel=([0-9.eE+-]+)s", text)
    cg_r_e2e = first_float(r"\[cuGraph\s+renum\s+\].*?e2e=([0-9.eE+-]+)s", text)
    mae = first_float(r"MAE=([0-9.eE+-]+)", text)
    return mine_kernel, mine_e2e, cg_nr_kernel, cg_nr_e2e, cg_r_kernel, cg_r_e2e, mae


def parse_cc(text):
    mine = first_float(r"Your GPU:\s+([0-9.eE+-]+) s", text)
    cugraph = first_float(r"cuGraph:\s+([0-9.eE+-]+) s", text)
    ok = "Size multisets equal? True" in text
    comps_mine = first_int(r"Components: yours=([0-9,]+)", text)
    comps_cugraph = first_int(r"Components: yours=[0-9,]+,\s+cuGraph=([0-9,]+)", text)
    return mine, cugraph, ok, comps_mine, comps_cugraph


def parse_gunrock_elapsed_ms(text):
    patterns = (
        r"GPU Total Elapsed Time\s*:\s*([0-9.eE+-]+)\s*\(ms\)",
        r"GPU Elapsed Time\s*:\s*([0-9.eE+-]+)\s*\(ms\)",
        r"avg\.\s+elapsed:\s*([0-9.eE+-]+)\s*ms",
        r"Run\s+\d+\s+elapsed:\s*([0-9.eE+-]+)\s*ms",
    )
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1)) / 1000.0
    return None


def parse_gunrock_aligned_e2e_ms(text):
    value = first_float(
        r"Aligned E2E Time\s*:\s*([0-9.eE+-]+)\s*\(ms\)",
        text,
    )
    return None if value is None else value / 1000.0


def parse_gunrock_construction_ms(text):
    value = first_float(
        r"Aligned Construction Time\s*:\s*([0-9.eE+-]+)\s*\(ms\)",
        text,
    )
    return None if value is None else value / 1000.0


def gunrock_strict_timing_or_none(text):
    try:
        return parse_strict_gunrock_timing(text), ""
    except GunrockTimingProtocolError as exc:
        return None, str(exc)


def gunrock_native_status(returncode, text):
    if returncode == 124:
        return "timeout"
    timing, _ = gunrock_strict_timing_or_none(text)
    return "ok" if returncode == 0 and timing is not None else "failed"


def gunrock_e2e_measurement(text, process_elapsed):
    timing, error = gunrock_strict_timing_or_none(text)
    if timing is not None:
        return (
            timing["e2e_seconds"],
            "standalone E2E is measured from MatrixMarket loading through "
            "host CSR/device graph materialization, device processing, and "
            "complete device-to-host result collection; validation, printing, "
            "and result-file I/O are excluded",
            {
                "measurement_scope": timing["measurement_scope"],
                "measurement_window": timing["e2e_window"],
                "timer_kind": "steady_clock_wall",
                "external_cli_wall_seconds": f"{float(process_elapsed):.9g}",
                "timing_protocol_version": timing["protocol_version"],
                "external_cli_wall_used": "false",
            },
        )
    return (
        None,
        "strict standalone E2E unavailable: "
        f"{error}; external CLI wall time is diagnostic only and is never substituted",
        {
            "measurement_scope": "standalone_gpu_function",
            "measurement_window": "unavailable",
            "external_cli_wall_seconds": f"{float(process_elapsed):.9g}",
            "external_cli_wall_used": "false",
            "timing_protocol_error": error,
        },
    )


def parse_gunrock_mst_weight(text):
    return first_float(r"GPU MST Weight:\s*([0-9.eE+-]+)", text)


def parse_gunrock_error_count(text):
    m = re.search(r"Number of errors\s*:\s*([0-9]+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([0-9]+)\s+errors occurred\b", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def add_gunrock_build_not_applicable(rows, size, graph_type, name, function, log, notes):
    timing, error = gunrock_strict_timing_or_none(read_text(log))
    if timing is not None:
        add_row(
            rows,
            size,
            graph_type,
            name,
            function,
            "Gunrock",
            "build",
            timing["construction_seconds"],
            "ok",
            log,
            notes=(
                notes
                + "; construction spans MatrixMarket load through host CSR "
                "and device graph materialization"
            ),
            extra={
                "measurement_scope": timing["measurement_scope"],
                "measurement_window": timing["construction_window"],
                "timer_kind": "steady_clock_wall",
                "timing_protocol_version": timing["protocol_version"],
                "external_cli_wall_used": "false",
            },
        )
        return
    add_row(
        rows,
        size,
        graph_type,
        name,
        function,
        "Gunrock",
        "build",
        None,
        "skipped",
        log,
        notes=(
            notes
            + "; strict construction metric unavailable: "
            + error
            + "; external CLI wall time is not substituted"
        ),
    )


def parse_library_results(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith("RESULT_JSON "):
            continue
        try:
            rows.append(json.loads(line[len("RESULT_JSON "):]))
        except json.JSONDecodeError:
            continue
    return rows


def write_matrix_market(
    edge_path,
    out_path,
    directed,
    weighted=False,
    return_metadata=False,
):
    """Write a 1-based MatrixMarket coordinate file for Gunrock examples.

    PageRank receives directed edges for directed datasets and a symmetrized
    edge list for undirected datasets. MST always receives an undirected
    projection, symmetrized explicitly, with the same deterministic weights used
    by the MST benchmark.
    """
    raw = []
    ids = set()
    with (ROOT / edge_path).open() as f:
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
            ids.add(u)
            ids.add(v)
            if u != v:
                raw.append((u, v))
    if not ids or not raw:
        raise RuntimeError(f"empty graph for MatrixMarket conversion: {edge_path}")

    sorted_ids = sorted(ids)
    remap = {v: i for i, v in enumerate(sorted_ids)}
    n = len(sorted_ids)

    metadata = {"nodes": n}
    if directed and not weighted:
        edges = sorted(set((remap[u], remap[v]) for u, v in raw if remap[u] != remap[v]))
        header = "%%MatrixMarket matrix coordinate real general\n"
    elif weighted:
        edges = []
        canon = set()
        for u0, v0 in raw:
            u = remap[u0]
            v = remap[v0]
            if u == v:
                continue
            if u > v:
                u, v = v, u
            canon.add((u, v))
        for u, v in sorted(canon):
            w = 1 + ((u * v) % n)
            edges.append((u, v, w))
        parent = list(range(n))

        def find_root(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for u, v, _ in edges:
            ru = find_root(u)
            rv = find_root(v)
            if ru != rv:
                parent[rv] = ru
        metadata.update(
            {
                "undirected_edges": len(edges),
                "components": len({find_root(node) for node in range(n)}),
            }
        )
        header = "%%MatrixMarket matrix coordinate real symmetric\n"
    else:
        # Undirected PageRank uses a general matrix with both directions
        # materialized. Gunrock MST instead requires a symmetric matrix header,
        # handled by the weighted branch above.
        canon = set()
        for u0, v0 in raw:
            u = remap[u0]
            v = remap[v0]
            if u == v:
                continue
            if u > v:
                u, v = v, u
            canon.add((u, v))
        edges = []
        for u, v in sorted(canon):
            edges.append((u, v))
            edges.append((v, u))
        header = "%%MatrixMarket matrix coordinate real general\n"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        out.write(header)
        out.write(f"{n} {n} {len(edges)}\n")
        if directed and not weighted:
            for u, v in edges:
                out.write(f"{u + 1} {v + 1} 1\n")
        elif weighted:
            for u, v, w in edges:
                out.write(f"{u + 1} {v + 1} {w}\n")
        else:
            for u, v in edges:
                out.write(f"{u + 1} {v + 1} 1\n")
    return (out_path, metadata) if return_metadata else out_path


def write_sssp_weighted_matrix_market(edge_path, out_path, directed):
    """Write deterministic weighted MatrixMarket for SSSP alignment.

    Directed datasets keep directed edges (general matrix). Undirected datasets
    are emitted as symmetric matrices with one canonical edge per pair.
    """
    raw = []
    ids = set()
    with (ROOT / edge_path).open() as f:
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
            ids.add(u)
            ids.add(v)
            if u != v:
                raw.append((u, v))
    if not ids or not raw:
        raise RuntimeError(f"empty graph for weighted MatrixMarket conversion: {edge_path}")

    sorted_ids = sorted(ids)
    remap = {v: i for i, v in enumerate(sorted_ids)}
    n = len(sorted_ids)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        if directed:
            edges = sorted(set((remap[u], remap[v]) for u, v in raw if remap[u] != remap[v]))
            out.write("%%MatrixMarket matrix coordinate real general\n")
            out.write(f"{n} {n} {len(edges)}\n")
            for u, v in edges:
                w = 1 + ((u * v) % max(1, n))
                out.write(f"{u + 1} {v + 1} {w}\n")
        else:
            canon = set()
            for u0, v0 in raw:
                u = remap[u0]
                v = remap[v0]
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                canon.add((u, v))
            edges = sorted(canon)
            out.write("%%MatrixMarket matrix coordinate real symmetric\n")
            out.write(f"{n} {n} {len(edges)}\n")
            for u, v in edges:
                w = 1 + ((u * v) % max(1, n))
                out.write(f"{u + 1} {v + 1} {w}\n")
    return out_path


def write_plot_and_tables(out_dir, rows, datasets, gpu_label):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    df = pd.DataFrame(rows)
    comparison_df = df
    validation_path = Path(out_dir) / "correctness_validation.csv"
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        valid_statuses = {"pass", "sampled_pass", "weak_pass", "reference"}
        valid = validation[validation["validation_status"].isin(valid_statuses)][
            ["dataset", "function", "baseline"]
        ].drop_duplicates()
        comparison_df = df.merge(
            valid,
            on=["dataset", "function", "baseline"],
            how="inner",
        )
    runtime_status = {"ok", "timeout"}
    runtime = comparison_df[
        (comparison_df["status"].isin(runtime_status))
        & (comparison_df["seconds"] != "")
    ]
    runtime = runtime.copy()
    runtime["seconds"] = runtime["seconds"].astype(float)
    runtime["is_timeout"] = runtime["status"].astype(str).eq("timeout") | runtime["notes"].astype(str).str.contains("TIMEOUT_TOO_LONG", na=False)
    ok = runtime[runtime["status"] == "ok"].copy()
    build = runtime[runtime["metric"] == "build"].copy()
    kernel = runtime[runtime["metric"] == "kernel"].copy()
    e2e = runtime[runtime["metric"] == "e2e"].copy()
    memory = ok[ok["metric"].astype(str).str.startswith("memory")].copy()

    colors = {
        "EGGPU": "#1f77b4",
        "easygraph-cpu": "#d62728",
        "easygraph-cpp": "#ff7f0e",
        "igraph": "#2ca02c",
        "networkx": "#8c564b",
        "nx-cugraph": "#9467bd",
        "Gunrock": "#7f7f7f",
    }
    hatches = {
        "EGGPU": "",
        "easygraph-cpu": "///",
        "easygraph-cpp": "\\\\\\",
        "igraph": "...",
        "networkx": "xx",
        "nx-cugraph": "++",
        "Gunrock": "--",
    }

    def grouped_plot(metric_df, metric_name, filename, y_label="Seconds (log)", log_scale=True):
        funcs = [f for f in DEFAULT_FUNCTIONS if f in set(metric_df["function"])]
        if not funcs:
            return
        dataset_order = [d[2] for d in datasets]
        ncols = min(4, max(1, len(funcs)))
        nrows = int(math.ceil(len(funcs) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.9 * ncols, max(2.9, 2.65 * nrows)),
            constrained_layout=True,
        )
        axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
        for ax, func in zip(axes_flat, funcs):
            sub = metric_df[metric_df["function"] == func]
            bases = [b for b in ["EGGPU", "easygraph-cpu", "easygraph-cpp", "igraph", "networkx", "nx-cugraph", "Gunrock"]
                     if b in set(sub["baseline"])]
            x = list(range(len(dataset_order)))
            width = 0.75 / max(1, len(bases))
            for i, base in enumerate(bases):
                vals = []
                timeout_positions = []
                for ds in dataset_order:
                    hit = sub[(sub["dataset"] == ds) & (sub["baseline"] == base)]
                    if len(hit):
                        vals.append(float(hit["seconds"].iloc[0]))
                        timeout_positions.append(bool(hit["is_timeout"].iloc[0]) if "is_timeout" in hit.columns else False)
                    else:
                        vals.append(math.nan)
                        timeout_positions.append(False)
                offset = (i - (len(bases) - 1) / 2) * width
                xpos = [v + offset for v in x]
                ax.bar(xpos, vals, width=width, label=base,
                       color=colors.get(base, None), edgecolor="#111827", linewidth=0.25,
                       hatch=hatches.get(base, ""))
                for px, val, is_to in zip(xpos, vals, timeout_positions):
                    if is_to and not math.isnan(val):
                        ax.text(px, val, "TO", ha="center", va="bottom",
                                fontsize=7, rotation=90, fontweight="bold", color="#991b1b")
            ax.set_title(func, fontweight="bold", pad=4)
            ax.set_xticks(x, dataset_order, rotation=28, ha="right")
            if log_scale:
                ax.set_yscale("log")
            ax.set_ylabel(y_label)
            ax.grid(axis="y", which="both")
            ax.grid(axis="x", visible=False)
            ax.margins(x=0.02)
        for ax in axes_flat[len(funcs):]:
            ax.set_visible(False)
        handles, labels = axes_flat[0].get_legend_handles_labels()
        for ax in axes_flat[1:]:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        dedup = dict(zip(labels, handles))
        fig.legend(
            dedup.values(),
            dedup.keys(),
            loc="upper center",
            ncol=min(7, max(1, len(dedup))),
            frameon=False,
            bbox_to_anchor=(0.5, 1.04),
        )
        fig.suptitle(f"{metric_name} on GPU {gpu_label}", fontsize=13, fontweight="bold", y=1.08)
        fig.savefig(out_dir / filename, dpi=220, bbox_inches="tight")
        fig.savefig(out_dir / filename.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)

    grouped_plot(build, "Graph Build", "runtime_build.png")
    grouped_plot(kernel, "Kernel", "runtime_kernel.png")
    grouped_plot(e2e, "End-to-end Time", "runtime_e2e.png")

    # Memory plot using best-available comparable metric.
    memory_metric_priority = [
        "memory_peak_gpu_proc_mb",
        "memory_peak_gpu_mb",
        "memory_peak_gpu_proc_delta_mb",
        "memory_peak_gpu_delta_mb",
        "memory_peak_rss_mb",
    ]
    mem_metric = next((m for m in memory_metric_priority if m in set(memory["metric"])), None)
    if mem_metric is not None:
        mem_sub = memory[memory["metric"] == mem_metric].copy()
        grouped_plot(
            mem_sub,
            f"{mem_metric} (MB)",
            "runtime_memory.png",
            y_label="Memory (MB)",
            log_scale=False,
        )

    # Speedup heatmap: best comparable baseline / EGGPU for e2e.
    speed_rows = []
    for _, r in e2e[e2e["baseline"] == "EGGPU"].iterrows():
        sub = e2e[(e2e["dataset"] == r["dataset"]) & (e2e["function"] == r["function"]) &
                  (e2e["baseline"] != "EGGPU")]
        if len(sub):
            best = sub.sort_values("seconds").iloc[0]
            speed_rows.append({
                "dataset": r["dataset"],
                "function": r["function"],
                "speedup_vs_best_baseline": float(best["seconds"]) / float(r["seconds"]),
                "best_baseline": best["baseline"],
            })
    speed = pd.DataFrame(speed_rows)
    if len(speed):
        pivot = speed.pivot(index="function", columns="dataset", values="speedup_vs_best_baseline")
        func_order = [f for f in DEFAULT_FUNCTIONS if f in set(speed["function"])]
        pivot = pivot.reindex(index=func_order, columns=[d[2] for d in datasets])
        fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
        vals = pivot.values.astype(float)
        finite = vals[~pd.isna(vals)]
        vmax = max(2.0, float(pd.Series(finite).quantile(0.95)) if finite.size else 2.0)
        norm = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
        im = ax.imshow(vals, cmap=mpl.colormaps["RdYlGn"], norm=norm, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=28, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not math.isnan(val):
                    label = f"{val:.1f}x" if val < 100 else ">99x"
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        color="#111827",
                        fontweight="bold",
                        fontsize=8,
                    )
        ax.set_title("EGGPU E2E Speedup vs Best Available Baseline", fontweight="bold")
        fig.colorbar(im, ax=ax, label="speedup")
        fig.savefig(out_dir / "speedup_heatmap.png", dpi=220, bbox_inches="tight")
        fig.savefig(out_dir / "speedup_heatmap.pdf", bbox_inches="tight")
        plt.close(fig)

    # LaTeX main table: e2e seconds, EGGPU plus baselines that produced values.
    baseline_order = ["EGGPU", "easygraph-cpu", "easygraph-cpp", "igraph", "networkx", "nx-cugraph", "Gunrock"]
    present_baselines = [b for b in baseline_order if b in set(e2e["baseline"])]
    main = e2e[e2e["baseline"].isin(present_baselines)].copy()
    main["time"] = main.apply(
        lambda r: f">{float(r['seconds']):.0f}s" if bool(r.get("is_timeout", False)) else f"{float(r['seconds']):.4g}",
        axis=1,
    )
    main["key"] = main["function"] + " / " + main["dataset"]
    table = main.pivot_table(index=["function", "graph_type", "dataset"], columns="baseline", values="time", aggfunc="first")
    table = table.reindex(columns=present_baselines)
    table = table.fillna("--")
    latex = table.to_latex(
        escape=False,
        caption=f"End-to-end runtime comparison on A100 GPU {gpu_label}. Times are seconds; >{PER_FUNCTION_TIMEOUT_SECONDS}s marks timeout; -- means the baseline was unavailable or skipped.",
        label="tab:eggpu-full-baseline",
    )
    (out_dir / "main_table.tex").write_text(latex)

    df.to_csv(out_dir / "results_long.csv", index=False)
    build.to_csv(out_dir / "results_build.csv", index=False)
    kernel.to_csv(out_dir / "results_kernel.csv", index=False)
    e2e.to_csv(out_dir / "results_e2e.csv", index=False)
    memory.to_csv(out_dir / "results_memory.csv", index=False)


def _plot_worker(out_dir, rows, datasets, gpu_label):
    write_plot_and_tables(out_dir, rows, datasets, gpu_label)


def write_plot_and_tables_isolated(out_dir, rows, datasets, gpu_label):
    """Run optional plotting outside the benchmark process.

    Matplotlib/PDF generation is not part of the measured benchmark path.  If
    the renderer is killed by memory pressure or an optional dependency issue,
    preserve the completed timing/correctness artifacts and record a plot error
    instead of losing final metadata.
    """
    if os.environ.get("EGGPU_SKIP_PLOTS", "").strip().upper() in {"1", "TRUE", "YES", "ON"}:
        print("[info] plot/table generation skipped because EGGPU_SKIP_PLOTS=TRUE", flush=True)
        return None

    timeout_s = float(os.environ.get("EGGPU_PLOT_TIMEOUT", "120"))
    proc = mp.Process(target=_plot_worker, args=(out_dir, rows, datasets, gpu_label))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return f"plot/table generation exceeded {timeout_s:g}s and was terminated"
    if proc.exitcode != 0:
        return f"plot/table generation process exited with code {proc.exitcode}"
    return None


def validated_comparison_keys(out_dir):
    path = Path(out_dir) / "correctness_validation.csv"
    if not path.exists():
        return None
    accepted = {"pass", "sampled_pass", "weak_pass", "reference"}
    keys = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("validation_status") in accepted:
                keys.add((row.get("dataset"), row.get("function"), row.get("baseline")))
    return keys


def write_metric_csvs_no_pandas(out_dir, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    eligible = validated_comparison_keys(out_dir)
    ok_rows = [
        r
        for r in rows
        if r.get("status") in ("ok", "timeout") and str(r.get("seconds", "")).strip() != ""
        and (
            r.get("status") == "timeout"
            or eligible is None
            or (r.get("dataset"), r.get("function"), r.get("baseline")) in eligible
        )
    ]
    for metric in ("build", "kernel", "e2e"):
        path = out_dir / f"results_{metric}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in ok_rows:
                if r.get("metric") == metric:
                    writer.writerow(r)
    mem_path = out_dir / "results_memory.csv"
    with mem_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in ok_rows:
            metric = str(r.get("metric", ""))
            if metric.startswith("memory"):
                writer.writerow(r)


def maybe_run_gunrock_pr(
    rows,
    size,
    graph_type,
    name,
    path,
    ds_dir,
    env,
    repeat,
    cooldown,
    requested_alpha,
    requested_tol,
):
    exe = find_gunrock_exe("pr")
    log = ds_dir / "gunrock_pr.log"
    if exe is None:
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "build", None, "skipped", log,
                notes=f"Gunrock pr not found; {gunrock_search_note()}")
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "kernel", None, "skipped", log,
                notes=f"Gunrock pr not found; {gunrock_search_note()}")
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "e2e", None, "skipped", log,
                notes=f"Gunrock pr not found; {gunrock_search_note()}")
        return
    try:
        mtx = ds_dir / "gunrock_pr.mtx"
        write_matrix_market(
            path,
            mtx,
            directed=(graph_type == "directed"),
            weighted=False,
        )
        cmd = [
            str(path_for_cmd(exe)),
            "-m",
            str(path_for_cmd(mtx)),
            "-n",
            str(repeat),
            "--alpha",
            f"{float(requested_alpha):.12g}",
            "--tol",
            f"{float(requested_tol):.12g}",
        ]
        rc, elapsed, peak_rss_mb = run_cmd(cmd, log, gunrock_runtime_env(exe, env), timeout=PER_FUNCTION_TIMEOUT_SECONDS, cooldown=cooldown)
        txt = read_text(log)
        sec = parse_gunrock_elapsed_ms(txt)
        e2e_sec, e2e_note, e2e_extra = gunrock_e2e_measurement(txt, elapsed)
        err_cnt = parse_gunrock_error_count(txt)
        status = gunrock_native_status(rc, txt)
        note = (
            "kernel from Gunrock's reported device timer; "
            f"{e2e_note}; "
            f"PageRank CLI adapter forwards alpha={requested_alpha:g}, tol={requested_tol:g} "
            "to Gunrock's native PageRank parameter object"
        )
        if err_cnt:
            note += f"; internal validation reported {err_cnt} mismatches"
        corr = ""
        if rc == 124:
            status = "timeout"
            sec = None
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        if status == "ok":
            detail_suffix, detail_note = ensure_gunrock_pagerank_detail(
                exe,
                mtx,
                ds_dir,
                env,
                requested_alpha,
                requested_tol,
                cooldown,
            )
            corr = detail_suffix
            if detail_note:
                note += "; " + detail_note
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "PageRank", log, note)
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(
            rows,
            size,
            graph_type,
            name,
            "PageRank",
            "Gunrock",
            "e2e",
            e2e_sec,
            status,
            log,
            corr,
            notes=note,
            extra=e2e_extra,
        )
        add_memory_metric_rows(rows, size, graph_type, name, "PageRank", "Gunrock", log, peak_rss_mb, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock PR setup failed: {e}\n")
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "e2e", None, "failed", log, notes=str(e))


def _gunrock_dataset_detail_dir(sample_dir):
    sample_dir = Path(sample_dir)
    if sample_dir.parent.name == "gunrock_samples":
        return sample_dir.parent.parent / "details"
    return sample_dir / "details"


def ensure_gunrock_pagerank_detail(
    exe,
    mtx,
    sample_dir,
    env,
    alpha,
    tolerance,
    cooldown,
):
    """Create one unmeasured full-vector PageRank artifact per dataset."""

    import numpy as np

    detail_dir = _gunrock_dataset_detail_dir(sample_dir)
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_path = detail_dir / "Gunrock_PageRank.npz"
    raw_path = detail_dir / "Gunrock_PageRank.float32"
    validation_log = detail_dir / "Gunrock_PageRank_validation.log"

    if not detail_path.exists():
        validation_cmd = [
            str(path_for_cmd(exe)),
            "-m",
            str(path_for_cmd(mtx)),
            "-n",
            "1",
            "--alpha",
            f"{float(alpha):.12g}",
            "--tol",
            f"{float(tolerance):.12g}",
            "--result_file",
            str(path_for_cmd(raw_path)),
        ]
        rc, _, _ = run_cmd(
            validation_cmd,
            validation_log,
            gunrock_runtime_env(exe, env),
            timeout=PER_FUNCTION_TIMEOUT_SECONDS,
            cooldown=cooldown,
        )
        if rc != 0 or not raw_path.exists():
            return "", "full-vector validation probe failed; strict validation remains inconclusive"
        values = np.fromfile(raw_path, dtype=np.float32)
        expected_vertices = matrix_market_n(mtx)
        if values.shape != (expected_vertices,):
            return (
                "",
                "full-vector validation probe returned "
                f"{values.size} values for {expected_vertices} vertices",
            )
        np.savez_compressed(detail_path, kind="vector", values=values)
        raw_path.unlink(missing_ok=True)

    with np.load(detail_path, allow_pickle=False) as detail:
        values = np.ascontiguousarray(detail["values"])
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(tuple(values.shape)).encode())
    digest.update(values.tobytes())
    return (
        f"detail={detail_path.resolve()}, detail_kind=vector, "
        f"detail_sha={digest.hexdigest()[:16]}",
        "full PageRank vector is exported by a separate unmeasured validation probe",
    )


def maybe_run_gunrock_mst(rows, size, graph_type, name, path, ds_dir, env, cooldown):
    exe = find_gunrock_exe("mst")
    log = ds_dir / "gunrock_mst.log"
    if exe is None:
        add_row(rows, size, graph_type, name, "MST", "Gunrock", "build", None, "skipped", log,
                notes=f"Gunrock mst not found; {gunrock_search_note()}")
        add_row(rows, size, graph_type, name, "MST", "Gunrock", "kernel", None, "skipped", log,
                notes=f"Gunrock mst not found; {gunrock_search_note()}")
        add_row(rows, size, graph_type, name, "MST", "Gunrock", "e2e", None, "skipped", log,
                notes=f"Gunrock mst not found; {gunrock_search_note()}")
        return
    try:
        mtx = ds_dir / "gunrock_mst.mtx"
        mtx, graph_meta = write_matrix_market(
            path,
            mtx,
            directed=False,
            weighted=True,
            return_metadata=True,
        )
        if graph_meta.get("components", 1) != 1:
            note = (
                "Gunrock MST executable requires a connected input, while the benchmark requires "
                "a full minimum spanning forest; "
                f"components={graph_meta.get('components')}, nodes={graph_meta.get('nodes')}. "
                "No largest-component fallback is timed."
            )
            log.write_text(note + "\n")
            add_gunrock_build_not_applicable(rows, size, graph_type, name, "MST", log, note)
            add_row(rows, size, graph_type, name, "MST", "Gunrock", "kernel", None, "skipped", log, notes=note)
            add_row(rows, size, graph_type, name, "MST", "Gunrock", "e2e", None, "skipped", log, notes=note)
            return
        cmd = [str(path_for_cmd(exe)), "-m", str(path_for_cmd(mtx))]
        rc, elapsed, peak_rss_mb = run_cmd(cmd, log, gunrock_runtime_env(exe, env), timeout=PER_FUNCTION_TIMEOUT_SECONDS, cooldown=cooldown)
        txt = read_text(log)
        sec = parse_gunrock_elapsed_ms(txt)
        e2e_sec, e2e_note, e2e_extra = gunrock_e2e_measurement(txt, elapsed)
        weight = parse_gunrock_mst_weight(txt)
        err_cnt = parse_gunrock_error_count(txt)
        status = gunrock_native_status(rc, txt)
        note = (
            "kernel from Gunrock's reported device timer; "
            f"{e2e_note}"
        )
        lower = txt.lower()
        connected_precondition_failure = (
            "connected graph" in lower
            or "input graph must be connected" in lower
            or "super vertices not decremented" in lower
        )
        if connected_precondition_failure:
            status = "skipped"
            note = (
                "Gunrock MST failed due graph precondition/implementation limits "
                "(connected-graph style semantics or super-vertex constraints); "
                "this benchmark uses spanning-forest semantics on disconnected inputs."
            )
        elif "input matrix must be symmetric" in lower:
            note += "; gunrock rejected generated matrix as non-symmetric"
        if rc == 124 and status != "ok":
            status = "timeout"
            sec = None
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        if err_cnt:
            note += f"; internal validation reported {err_cnt} mismatches"
        corr = "" if weight is None else f"weight={weight:.0f}"
        if err_cnt is not None:
            corr = (corr + "; " if corr else "") + f"validation_errors={err_cnt}"
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "MST", log, note)
        add_row(rows, size, graph_type, name, "MST", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(
            rows,
            size,
            graph_type,
            name,
            "MST",
            "Gunrock",
            "e2e",
            e2e_sec,
            status,
            log,
            corr,
            notes=note,
            extra=e2e_extra,
        )
        add_memory_metric_rows(rows, size, graph_type, name, "MST", "Gunrock", log, peak_rss_mb, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock MST setup failed: {e}\n")
        add_row(rows, size, graph_type, name, "MST", "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, "MST", "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, "MST", "Gunrock", "e2e", None, "failed", log, notes=str(e))


def maybe_run_gunrock_lcc(rows, size, graph_type, name, path, ds_dir, env, cooldown):
    exe = find_gunrock_exe("lcc")
    log = ds_dir / "gunrock_lcc.log"
    if exe is None:
        add_unavailable_gunrock(rows, size, graph_type, name, "LCC", ds_dir)
        return
    try:
        mtx = ds_dir / "gunrock_lcc.mtx"
        write_matrix_market(path, mtx, directed=False, weighted=False)
        cmd = [
            str(path_for_cmd(exe)),
            "market",
            str(path_for_cmd(mtx)),
            "--undirected=true",
            "--sort-csr=true",
            "--validation=none",
            "--quick=true",
        ]
        rc, elapsed, peak_rss_mb = run_cmd(cmd, log, gunrock_runtime_env(exe, env), timeout=PER_FUNCTION_TIMEOUT_SECONDS, cooldown=cooldown)
        txt = read_text(log)
        sec = parse_gunrock_elapsed_ms(txt)
        e2e_sec, e2e_note, e2e_extra = gunrock_e2e_measurement(txt, elapsed)
        err_cnt = parse_gunrock_error_count(txt)
        status = gunrock_native_status(rc, txt)
        note = (
            "kernel from Gunrock's reported device timer; "
            f"{e2e_note}; pre-generation of the aligned MatrixMarket file is "
            "outside the measured invocation, while MatrixMarket loading is included; "
            "run with --undirected=true --sort-csr=true --validation=none --quick=true"
        )
        if "libgunrock_utils.so" in txt:
            note += "; failed to load libgunrock_utils.so (runtime LD_LIBRARY_PATH issue)"
        if err_cnt:
            note += f"; internal validation reported {err_cnt} mismatches"
        corr = f"validation_errors={err_cnt}" if err_cnt is not None else ""
        if rc == 124:
            status = "timeout"
            sec = None
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        if status == "ok":
            detail_suffix, detail_note = ensure_gunrock_lcc_detail(
                exe, mtx, ds_dir, env, cooldown
            )
            corr = ", ".join(value for value in (corr, detail_suffix) if value)
            if detail_note:
                note += "; " + detail_note
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "LCC", log, note)
        add_row(rows, size, graph_type, name, "LCC", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(
            rows,
            size,
            graph_type,
            name,
            "LCC",
            "Gunrock",
            "e2e",
            e2e_sec,
            status,
            log,
            corr,
            notes=note,
            extra=e2e_extra,
        )
        add_memory_metric_rows(rows, size, graph_type, name, "LCC", "Gunrock", log, peak_rss_mb, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock LCC setup failed: {e}\n")
        add_row(rows, size, graph_type, name, "LCC", "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, "LCC", "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, "LCC", "Gunrock", "e2e", None, "failed", log, notes=str(e))


def ensure_gunrock_lcc_detail(exe, mtx, sample_dir, env, cooldown):
    """Create one unmeasured full-vector LCC artifact per dataset."""

    import numpy as np

    detail_dir = _gunrock_dataset_detail_dir(sample_dir)
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_path = detail_dir / "Gunrock_LCC.npz"
    raw_path = detail_dir / "Gunrock_LCC.float64"
    validation_log = detail_dir / "Gunrock_LCC_validation.log"

    if not detail_path.exists():
        validation_cmd = [
            str(path_for_cmd(exe)),
            "market",
            str(path_for_cmd(mtx)),
            "--undirected=true",
            "--sort-csr=true",
            "--validation=none",
            "--quick=true",
        ]
        validation_env = gunrock_runtime_env(exe, env)
        validation_env["EG_GUNROCK_LCC_RESULT_FILE"] = str(
            path_for_cmd(raw_path)
        )
        rc, _, _ = run_cmd(
            validation_cmd,
            validation_log,
            validation_env,
            timeout=PER_FUNCTION_TIMEOUT_SECONDS,
            cooldown=cooldown,
        )
        if rc != 0 or not raw_path.exists():
            return "", "full-vector validation probe failed; strict validation remains inconclusive"
        values = np.fromfile(raw_path, dtype=np.float64)
        expected_vertices = matrix_market_n(mtx)
        if values.shape != (expected_vertices,):
            return (
                "",
                "full-vector validation probe returned "
                f"{values.size} values for {expected_vertices} vertices",
            )
        np.savez_compressed(detail_path, kind="vector", values=values)
        raw_path.unlink(missing_ok=True)

    with np.load(detail_path, allow_pickle=False) as detail:
        values = np.ascontiguousarray(detail["values"])
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(tuple(values.shape)).encode())
    digest.update(values.tobytes())
    return (
        f"vertices={values.size}, mean={float(np.mean(values)):.17g}, "
        f"detail={detail_path.resolve()}, detail_kind=vector, "
        f"detail_sha={digest.hexdigest()[:16]}",
        "full LCC vector is exported by a separate unmeasured validation probe",
    )


def _gunrock_array_digest(values):
    """Return the detail digest used by the cross-library validator."""

    import numpy as np

    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(tuple(values.shape)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()[:16]


def ensure_gunrock_path_detail(
    exe,
    mtx,
    sample_dir,
    env,
    cooldown,
    sources,
    function,
):
    """Validate and export BFS/SSSP-family outputs outside the timed call."""

    import numpy as np

    if function not in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        raise ValueError(f"unsupported Gunrock path detail: {function}")
    source_array = np.asarray([int(source) for source in sources], dtype=np.int64)
    detail_dir = _gunrock_dataset_detail_dir(sample_dir)
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_path = detail_dir / f"Gunrock_{function}.npz"
    expected_vertices = matrix_market_n(mtx)

    if not detail_path.exists():
        values = np.full(
            (len(source_array), expected_vertices), np.inf, dtype=np.float64
        )
        for source_index, source in enumerate(source_array):
            raw_dtype = np.int32 if function == "BFS" else np.float32
            raw_path = detail_dir / (
                f"Gunrock_{function}_source_{int(source)}.{raw_dtype.__name__}"
            )
            validation_log = detail_dir / (
                f"Gunrock_{function}_source_{int(source)}_validation.log"
            )
            validation_cmd = [
                str(path_for_cmd(exe)),
                "-m",
                str(path_for_cmd(mtx)),
                "-s",
                str(int(source)),
                "--validate",
                "--result_file",
                str(path_for_cmd(raw_path)),
            ]
            rc, _, _ = run_cmd(
                validation_cmd,
                validation_log,
                gunrock_runtime_env(exe, env),
                timeout=PER_FUNCTION_TIMEOUT_SECONDS,
                cooldown=cooldown,
            )
            validation_errors = parse_gunrock_error_count(read_text(validation_log))
            if rc != 0 or validation_errors != 0 or not raw_path.exists():
                raw_path.unlink(missing_ok=True)
                return (
                    "",
                    "separate full-result validation failed for "
                    f"source={int(source)} (returncode={rc}, "
                    f"validation_errors={validation_errors})",
                )
            raw_values = np.fromfile(raw_path, dtype=raw_dtype)
            raw_path.unlink(missing_ok=True)
            if raw_values.shape != (expected_vertices,):
                return (
                    "",
                    "separate full-result validation returned "
                    f"{raw_values.size} values for {expected_vertices} vertices "
                    f"at source={int(source)}",
                )
            row = raw_values.astype(np.float64)
            if function == "BFS":
                row[raw_values == np.iinfo(np.int32).max] = np.inf
            else:
                row[np.abs(row) >= 1.0e30] = np.inf
            values[source_index] = row
        np.savez_compressed(
            detail_path,
            kind="sssp",
            sources=source_array,
            values=values,
        )

    with np.load(detail_path, allow_pickle=False) as detail:
        values = np.ascontiguousarray(detail["values"])
        saved_sources = np.ascontiguousarray(detail["sources"])
    if not np.array_equal(saved_sources, source_array):
        return "", "cached path detail uses a different source set"
    finite = np.isfinite(values)
    digest_values = np.where(finite, values, -1.0)
    return (
        f"sources={len(source_array)}, reachable={int(finite.sum())}, "
        f"detail={detail_path.resolve()}, detail_kind=sssp, "
        f"detail_sha={_gunrock_array_digest(digest_values)}",
        "CPU validation and full-result export run in separate unmeasured "
        "processes; every source reported zero validation errors",
    )


def ensure_gunrock_kcore_detail(exe, mtx, sample_dir, env, cooldown):
    """Validate and export the full KCore vector outside the timed call."""

    import numpy as np

    detail_dir = _gunrock_dataset_detail_dir(sample_dir)
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_path = detail_dir / "Gunrock_KCore.npz"
    raw_path = detail_dir / "Gunrock_KCore.int32"
    validation_log = detail_dir / "Gunrock_KCore_validation.log"

    if not detail_path.exists():
        validation_cmd = [
            str(path_for_cmd(exe)),
            str(path_for_cmd(mtx)),
            "--result_file",
            str(path_for_cmd(raw_path)),
        ]
        rc, _, _ = run_cmd(
            validation_cmd,
            validation_log,
            gunrock_runtime_env(exe, env),
            timeout=PER_FUNCTION_TIMEOUT_SECONDS,
            cooldown=cooldown,
        )
        validation_errors = parse_gunrock_error_count(read_text(validation_log))
        if rc != 0 or validation_errors != 0 or not raw_path.exists():
            raw_path.unlink(missing_ok=True)
            return (
                "",
                "separate full KCore validation failed "
                f"(returncode={rc}, validation_errors={validation_errors})",
            )
        values = np.fromfile(raw_path, dtype=np.int32)
        raw_path.unlink(missing_ok=True)
        expected_vertices = matrix_market_n(mtx)
        if values.shape != (expected_vertices,):
            return (
                "",
                "separate full KCore validation returned "
                f"{values.size} values for {expected_vertices} vertices",
            )
        np.savez_compressed(detail_path, kind="vector", values=values)

    with np.load(detail_path, allow_pickle=False) as detail:
        values = np.ascontiguousarray(detail["values"])
    return (
        f"nodes={values.size}, sum={int(np.sum(values, dtype=np.int64))}, "
        f"detail={detail_path.resolve()}, detail_kind=vector, "
        f"detail_sha={_gunrock_array_digest(values)}",
        "CPU validation and full-result export run in a separate unmeasured "
        "process and report zero validation errors",
    )


def maybe_run_gunrock_cc(rows, size, graph_type, name, path, ds_dir, env, cooldown, function="WCC"):
    log = ds_dir / f"gunrock_{function.lower()}.log"
    if function == "SCC" and graph_type == "directed":
        for metric in ("build", "kernel", "e2e"):
            add_row(
                rows,
                size,
                graph_type,
                name,
                function,
                "Gunrock",
                metric,
                None,
                "skipped",
                log,
                notes="Gunrock legacy cc exposes WCC/undirected connected-components semantics, not directed SCC.",
            )
        return
    exe = find_gunrock_exe("cc")
    if exe is None:
        add_unavailable_gunrock(rows, size, graph_type, name, function, ds_dir)
        return
    try:
        mtx = ds_dir / f"gunrock_{function.lower()}.mtx"
        write_matrix_market(path, mtx, directed=False, weighted=False)
        cmd = [
            str(path_for_cmd(exe)),
            "market",
            str(path_for_cmd(mtx)),
            "--undirected=true",
            "--sort-csr=true",
            "--validation=none",
            "--quick=true",
        ]
        rc, elapsed, peak_rss_mb = run_cmd(cmd, log, gunrock_runtime_env(exe, env), timeout=PER_FUNCTION_TIMEOUT_SECONDS, cooldown=cooldown)
        txt = read_text(log)
        sec = parse_gunrock_elapsed_ms(txt)
        err_cnt = parse_gunrock_error_count(txt)
        status = gunrock_native_status(rc, txt)
        semantic = "WCC semantics; undirected projection"
        if function == "SCC":
            semantic = "undirected graph: SCC equals WCC"
        note = (
            f"{semantic}; kernel from Gunrock output; a strict native aligned E2E label "
            "is required and external process wall time is never substituted; aligned "
            "MatrixMarket file pre-generation is outside the measured invocation; "
            "run with --undirected=true --sort-csr=true --validation=none --quick=true"
        )
        if "libgunrock_utils.so" in txt:
            note += "; failed to load libgunrock_utils.so (runtime LD_LIBRARY_PATH issue)"
        if err_cnt:
            note += f"; internal validation reported {err_cnt} mismatches"
        corr = f"validation_errors={err_cnt}" if err_cnt is not None else ""
        if rc == 124:
            status = "failed"
            sec = None
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        add_gunrock_build_not_applicable(rows, size, graph_type, name, function, log, note)
        add_row(rows, size, graph_type, name, function, "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(rows, size, graph_type, name, function, "Gunrock", "e2e", elapsed, status, log, corr, notes=note)
        add_memory_metric_rows(rows, size, graph_type, name, function, "Gunrock", log, peak_rss_mb, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock {function} setup failed: {e}\n")
        add_row(rows, size, graph_type, name, function, "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, function, "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, function, "Gunrock", "e2e", None, "failed", log, notes=str(e))


def maybe_run_gunrock_sssp(
    rows,
    size,
    graph_type,
    name,
    path,
    ds_dir,
    env,
    cooldown,
    sssp_sources,
    function="SSSP",
):
    if function not in {"Dijkstra", "BellmanFord", "SSSP"}:
        raise ValueError(f"unsupported Gunrock SSSP semantic adapter: {function}")
    exe = find_gunrock_exe("sssp")
    log = ds_dir / f"gunrock_{function.lower()}.log"
    if exe is None:
        add_unavailable_gunrock(rows, size, graph_type, name, function, ds_dir)
        return
    try:
        mtx = ds_dir / f"gunrock_{function.lower()}.mtx"
        write_sssp_weighted_matrix_market(path, mtx, directed=(graph_type == "directed"))
        n = matrix_market_n(mtx)
        source_count = 1 if function == "Dijkstra" else sssp_sources
        sources = deterministic_sources(n, source_count)
        if not sources:
            sources = [0]
        source_arg = ",".join(str(int(source)) for source in sources)
        cmd = [
            str(path_for_cmd(exe)),
            "-m",
            str(path_for_cmd(mtx)),
            "-s",
            source_arg,
        ]
        rc, elapsed, memory = run_cmd(
            cmd,
            log,
            gunrock_runtime_env(exe, env),
            timeout=PER_FUNCTION_TIMEOUT_SECONDS,
            cooldown=cooldown,
        )
        txt = read_text(log)
        sec = parse_gunrock_elapsed_ms(txt)
        e2e_sec, e2e_note, e2e_extra = gunrock_e2e_measurement(txt, elapsed)
        timed_out = rc == 124
        status = (
            "timeout"
            if timed_out
            else gunrock_native_status(rc, txt)
        )
        if function == "Dijkstra":
            note = (
                "single-source Dijkstra semantics on deterministic nonnegative weights; "
                "Gunrock's maintained weighted SSSP executable is the implementation alias; "
                "sources=1; aligned MatrixMarket file pre-generation is outside the "
                "measured invocation, while file loading is included; CPU validation excluded "
                f"from the timing window; {e2e_note}"
            )
        elif function == "BellmanFord":
            note = (
                "Bellman-Ford return semantics on the benchmark's deterministic nonnegative "
                "weights; Gunrock's maintained weighted SSSP executable is a conditional "
                "implementation alias and does not establish negative-edge support; "
                f"sources={len(sources)}; one process loads the graph once; "
                "reported kernel time is summed over sources; "
                "aligned MatrixMarket file pre-generation and CPU validation are outside "
                "the timing window, while MatrixMarket loading is included; "
                f"{e2e_note}"
            )
        else:
            note = (
                "weighted deterministic edges; one Gunrock process receives the "
                "complete source list and loads the graph once; reported kernel "
                "time is summed over sources; aligned MatrixMarket file pre-generation "
                "and CPU validation are outside the timing window, while MatrixMarket "
                f"loading is included; {e2e_note}"
            )
        corr = f"sources={len(sources)}"
        if timed_out:
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        if status == "ok":
            detail_suffix, detail_note = ensure_gunrock_path_detail(
                exe,
                mtx,
                ds_dir,
                env,
                cooldown,
                sources,
                function,
            )
            corr = ", ".join(value for value in (corr, detail_suffix) if value)
            if detail_note:
                note += "; " + detail_note
        add_gunrock_build_not_applicable(rows, size, graph_type, name, function, log, note)
        add_row(rows, size, graph_type, name, function, "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(
            rows,
            size,
            graph_type,
            name,
            function,
            "Gunrock",
            "e2e",
            e2e_sec,
            status,
            log,
            corr,
            notes=note,
            extra=e2e_extra,
        )
        add_memory_metric_rows(rows, size, graph_type, name, function, "Gunrock", log, memory, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock {function} setup failed: {e}\n")
        add_row(rows, size, graph_type, name, function, "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, function, "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, function, "Gunrock", "e2e", None, "failed", log, notes=str(e))


def maybe_run_gunrock_bfs(rows, size, graph_type, name, path, ds_dir, env, cooldown, sssp_sources):
    exe = find_gunrock_exe("bfs")
    log = ds_dir / "gunrock_bfs.log"
    if exe is None:
        add_unavailable_gunrock(rows, size, graph_type, name, "BFS", ds_dir)
        return
    try:
        mtx = ds_dir / "gunrock_bfs.mtx"
        write_matrix_market(path, mtx, directed=(graph_type == "directed"), weighted=False)
        n = matrix_market_n(mtx)
        sources = deterministic_sources(n, sssp_sources)
        if not sources:
            sources = [0]
        source_arg = ",".join(str(int(source)) for source in sources)
        cmd = [
            str(path_for_cmd(exe)),
            "-m",
            str(path_for_cmd(mtx)),
            "-s",
            source_arg,
        ]
        rc, elapsed, memory = run_cmd(
            cmd,
            log,
            gunrock_runtime_env(exe, env),
            timeout=PER_FUNCTION_TIMEOUT_SECONDS,
            cooldown=cooldown,
        )
        txt = read_text(log)
        sec = parse_gunrock_elapsed_ms(txt)
        e2e_sec, e2e_note, e2e_extra = gunrock_e2e_measurement(txt, elapsed)
        timed_out = rc == 124
        status = (
            "timeout"
            if timed_out
            else gunrock_native_status(rc, txt)
        )
        note = (
            "unweighted shortest paths; one Gunrock process receives the complete "
            "source list and loads the graph once; reported kernel time is summed "
            "over sources; aligned MatrixMarket file pre-generation and CPU validation "
            "are outside the timing window, while MatrixMarket loading is included; "
            f"{e2e_note}"
        )
        corr = f"sources={len(sources)}"
        if timed_out:
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        if status == "ok":
            detail_suffix, detail_note = ensure_gunrock_path_detail(
                exe,
                mtx,
                ds_dir,
                env,
                cooldown,
                sources,
                "BFS",
            )
            corr = ", ".join(value for value in (corr, detail_suffix) if value)
            if detail_note:
                note += "; " + detail_note
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "BFS", log, note)
        add_row(rows, size, graph_type, name, "BFS", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(
            rows,
            size,
            graph_type,
            name,
            "BFS",
            "Gunrock",
            "e2e",
            e2e_sec,
            status,
            log,
            corr,
            notes=note,
            extra=e2e_extra,
        )
        add_memory_metric_rows(
            rows,
            size,
            graph_type,
            name,
            "BFS",
            "Gunrock",
            log,
            memory,
            status=status,
            notes=note,
        )
    except Exception as e:
        log.write_text(f"Gunrock BFS setup failed: {e}\n")
        add_row(
            rows,
            size,
            graph_type,
            name,
            "BFS",
            "Gunrock",
            "build",
            None,
            "skipped",
            log,
            notes=str(e),
        )
        add_row(
            rows,
            size,
            graph_type,
            name,
            "BFS",
            "Gunrock",
            "kernel",
            None,
            "failed",
            log,
            notes=str(e),
        )
        add_row(
            rows,
            size,
            graph_type,
            name,
            "BFS",
            "Gunrock",
            "e2e",
            None,
            "failed",
            log,
            notes=str(e),
        )


def maybe_run_gunrock_kcore(rows, size, graph_type, name, path, ds_dir, env, cooldown):
    exe = find_gunrock_exe("kcore")
    log = ds_dir / "gunrock_kcore.log"
    if exe is None:
        add_unavailable_gunrock(rows, size, graph_type, name, "KCore", ds_dir)
        return
    try:
        mtx = ds_dir / "gunrock_kcore.mtx"
        # The benchmark defines KCore on the simple undirected projection for
        # every library.  Passing a directed MatrixMarket file made Gunrock
        # validate a different directed-degree problem on directed datasets.
        write_matrix_market(path, mtx, directed=False, weighted=False)
        cmd = [
            str(path_for_cmd(exe)),
            str(path_for_cmd(mtx)),
            "--no-validate",
        ]
        rc, elapsed, peak_rss_mb = run_cmd(cmd, log, gunrock_runtime_env(exe, env), timeout=PER_FUNCTION_TIMEOUT_SECONDS, cooldown=cooldown)
        txt = read_text(log)
        sec = parse_gunrock_elapsed_ms(txt)
        e2e_sec, e2e_note, e2e_extra = gunrock_e2e_measurement(txt, elapsed)
        status = gunrock_native_status(rc, txt)
        note = (
            "undirected projection; kernel from Gunrock's reported device timer; "
            f"{e2e_note}; CPU validation excluded from the measured invocation"
        )
        corr = ""
        if rc == 124:
            status = "failed"
            sec = None
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        if status == "ok":
            detail_suffix, detail_note = ensure_gunrock_kcore_detail(
                exe, mtx, ds_dir, env, cooldown
            )
            corr = detail_suffix
            if detail_note:
                note += "; " + detail_note
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "KCore", log, note)
        add_row(rows, size, graph_type, name, "KCore", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(
            rows,
            size,
            graph_type,
            name,
            "KCore",
            "Gunrock",
            "e2e",
            e2e_sec,
            status,
            log,
            corr,
            notes=note,
            extra=e2e_extra,
        )
        add_memory_metric_rows(rows, size, graph_type, name, "KCore", "Gunrock", log, peak_rss_mb, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock KCore setup failed: {e}\n")
        add_row(rows, size, graph_type, name, "KCore", "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, "KCore", "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, "KCore", "Gunrock", "e2e", None, "failed", log, notes=str(e))


def maybe_run_gunrock_bc(rows, size, graph_type, name, path, ds_dir, env, cooldown, bc_sources):
    exe = find_gunrock_exe("bc")
    log = ds_dir / "gunrock_bc.log"
    if exe is None:
        add_unavailable_gunrock(rows, size, graph_type, name, "BC", ds_dir)
        return
    try:
        mtx = ds_dir / "gunrock_bc.mtx"
        write_matrix_market(path, mtx, directed=(graph_type == "directed"), weighted=False)
        n = matrix_market_n(mtx)
        sources = deterministic_sources(n, bc_sources)
        if not sources:
            sources = [0]
        source_arg = ",".join(str(int(source)) for source in sources)
        cmd = [
            str(path_for_cmd(exe)),
            "-m",
            str(path_for_cmd(mtx)),
            "-s",
            source_arg,
        ]
        rc, elapsed, memory = run_cmd(
            cmd,
            log,
            gunrock_runtime_env(exe, env),
            timeout=PER_FUNCTION_TIMEOUT_SECONDS,
            cooldown=cooldown,
        )
        txt = read_text(log)
        sec = parse_gunrock_elapsed_ms(txt)
        e2e_sec, e2e_note, e2e_extra = gunrock_e2e_measurement(txt, elapsed)
        timed_out = rc == 124
        status = "timeout" if timed_out else gunrock_native_status(rc, txt)
        note = (
            f"exact specified-source mode; sources={len(sources)}; one Gunrock process "
            "loads the graph once and invokes the maintained single-source BC kernel "
            "for every source; reported kernel time is the sum over sources; "
            "aligned MatrixMarket file pre-generation is outside the measured invocation, "
            f"while MatrixMarket loading is included; {e2e_note}"
        )
        corr = f"sources={len(sources)}"
        if timed_out:
            sec = None
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        if status == "ok":
            detail_suffix, detail_note = ensure_gunrock_bc_detail(
                exe,
                mtx,
                ds_dir,
                env,
                sources,
                graph_type == "directed",
                cooldown,
            )
            corr = ", ".join(value for value in (corr, detail_suffix) if value)
            if detail_note:
                note += "; " + detail_note
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "BC", log, note)
        add_row(rows, size, graph_type, name, "BC", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(
            rows,
            size,
            graph_type,
            name,
            "BC",
            "Gunrock",
            "e2e",
            e2e_sec,
            status,
            log,
            corr,
            notes=note,
            extra=e2e_extra,
        )
        add_memory_metric_rows(rows, size, graph_type, name, "BC", "Gunrock", log, memory, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock BC setup failed: {e}\n")
        add_row(rows, size, graph_type, name, "BC", "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, "BC", "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, "BC", "Gunrock", "e2e", None, "failed", log, notes=str(e))


def ensure_gunrock_bc_detail(
    exe,
    mtx,
    sample_dir,
    env,
    sources,
    directed,
    cooldown,
):
    """Export and normalize the full specified-source BC vector outside timing."""

    import numpy as np

    detail_dir = _gunrock_dataset_detail_dir(sample_dir)
    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_path = detail_dir / "Gunrock_BC.npz"
    raw_path = detail_dir / "Gunrock_BC.float32"
    validation_log = detail_dir / "Gunrock_BC_validation.log"
    source_array = np.asarray([int(source) for source in sources], dtype=np.int64)

    if not detail_path.exists():
        validation_cmd = [
            str(path_for_cmd(exe)),
            "-m",
            str(path_for_cmd(mtx)),
            "-s",
            ",".join(str(int(source)) for source in sources),
            "--result_file",
            str(path_for_cmd(raw_path)),
        ]
        rc, _, _ = run_cmd(
            validation_cmd,
            validation_log,
            gunrock_runtime_env(exe, env),
            timeout=PER_FUNCTION_TIMEOUT_SECONDS,
            cooldown=cooldown,
        )
        if rc != 0 or not raw_path.exists():
            return "", "full-vector validation probe failed; strict validation remains inconclusive"
        values = np.fromfile(raw_path, dtype=np.float32).astype(np.float64)
        expected_vertices = matrix_market_n(mtx)
        if values.shape != (expected_vertices,):
            return (
                "",
                "full-vector validation probe returned "
                f"{values.size} values for {expected_vertices} vertices",
            )
        # The maintained Gunrock BC kernel applies the undirected 1/2 path
        # convention internally. NetworkX subset BC applies no such factor on
        # directed graphs, so restore the directed public-result convention.
        if directed:
            values *= 2.0
        np.savez_compressed(
            detail_path,
            kind="vector",
            values=values,
            sources=source_array,
        )
        raw_path.unlink(missing_ok=True)

    with np.load(detail_path, allow_pickle=False) as detail:
        values = np.ascontiguousarray(detail["values"])
        saved_sources = np.ascontiguousarray(detail["sources"])
    if not np.array_equal(saved_sources, source_array):
        return "", "cached BC detail uses a different source set"
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(tuple(values.shape)).encode())
    digest.update(values.tobytes())
    return (
        f"nodes={values.size}, sum={float(np.sum(values)):.17g}, "
        f"detail={detail_path.resolve()}, detail_kind=vector, "
        f"detail_sha={digest.hexdigest()[:16]}",
        "full specified-source BC vector is exported by a separate unmeasured "
        "validation probe; directed output scaling is aligned to NetworkX subset semantics",
    )


def add_unavailable_gunrock(rows, size, graph_type, name, function, ds_dir):
    log = ds_dir / f"gunrock_{function.lower()}_unavailable.log"
    log.write_text(f"Gunrock {function} baseline unavailable: no matching executable. {gunrock_search_note()}.\n")
    status = "skipped"
    note = "no matching Gunrock executable in current build"
    if function in {"WCC", "SCC", "CC"}:
        note += "; modern Gunrock has no aligned executable; legacy CC requires an unmaintained API/toolchain branch"
    for metric in ("build", "kernel", "e2e"):
        add_row(rows, size, graph_type, name, function, "Gunrock", metric, None, status, log,
                notes=note)


def _closeness_exact_limits():
    raw_nodes = os.environ.get("EGGPU_CLOSENESS_EXACT_MAX_NODES", "").strip().upper()
    raw_work = os.environ.get("EGGPU_CLOSENESS_EXACT_MAX_WORK", "").strip().upper()
    if raw_nodes in {"", "AUTO"}:
        max_nodes = 1000000
    else:
        try:
            max_nodes = int(raw_nodes)
        except Exception:
            max_nodes = 1000000
    if raw_work in {"", "AUTO"}:
        max_work = 50_000_000_000
    else:
        try:
            max_work = int(raw_work)
        except Exception:
            max_work = 50_000_000_000
    return max_nodes, max_work


def exact_closeness_too_large(dataset_stat):
    max_nodes, max_work = _closeness_exact_limits()
    try:
        nodes = int(dataset_stat.get("nodes_raw", 0))
    except Exception:
        nodes = 0
    try:
        edges = int(dataset_stat.get("edge_rows_no_selfloops", 0))
    except Exception:
        edges = 0
    work = nodes * edges
    return (max_nodes > 0 and nodes > max_nodes) or (max_work > 0 and work > max_work)


def add_exact_closeness_scale_skip(rows, size, graph_type, name, baseline, ds_dir, dataset_stat):
    log = ds_dir / f"library_{baseline}_closeness.log"
    nodes = int(dataset_stat.get("nodes_raw", 0) or 0)
    edges = int(dataset_stat.get("edge_rows_no_selfloops", 0) or 0)
    work = nodes * edges
    max_nodes, max_work = _closeness_exact_limits()
    note = (
        "exact all-source Closeness skipped by symmetric scale guard: "
        f"nodes={nodes:,}, edge_rows_no_selfloops={edges:,}, work_estimate=nodes*edges={work:,}; "
        f"limits: EGGPU_CLOSENESS_EXACT_MAX_NODES={max_nodes}, "
        f"EGGPU_CLOSENESS_EXACT_MAX_WORK={max_work}; "
        "all exact CPU/GPU backends use the same predeclared skip rule; "
        "large-graph Closeness is reported in the separate sampled-target supplement"
    )
    log.write_text(note + "\n")
    for metric in ("build", "e2e", "kernel"):
        add_row(
            rows,
            size,
            graph_type,
            name,
            "Closeness",
            baseline,
            metric,
            None,
            "skipped",
            log,
            notes=note,
            extra={
                "semantic": "exact_all_node",
                "skip_reason": "exact_scale_guard",
            },
        )


def collect_gunrock_lib_dirs(exe_path):
    exe = Path(exe_path).resolve()
    dirs = []
    seen = set()

    def add_dir(p):
        p = Path(p).resolve()
        key = str(p)
        if key in seen or not p.exists() or not p.is_dir():
            return
        seen.add(key)
        dirs.append(key)

    anchors = [exe.parent.parent]
    anchors += list(exe.parents[:6])
    for anchor in anchors:
        for rel in ("lib", "lib64", "build/lib", "build/lib64"):
            add_dir(anchor / rel)
        for p in anchor.glob("build*/lib*"):
            add_dir(p)

    # Include sibling build trees under the same gunrock root.
    gunrock_root = None
    for p in exe.parents:
        if p.name.startswith("gunrock_"):
            gunrock_root = p
            break
    if gunrock_root is not None:
        for p in gunrock_root.glob("build*/lib*"):
            add_dir(p)
        for p in gunrock_root.rglob("libgunrock_utils.so"):
            add_dir(p.parent)
    return dirs


def gunrock_runtime_env(exe_path, base_env):
    env = dict(base_env)
    env["EGGPU_REQUIRE_GPU_IDLE"] = "TRUE"
    env["EGGPU_GUNROCK_ALIGNED_E2E"] = "TRUE"
    lib_dirs = collect_gunrock_lib_dirs(exe_path)
    if not lib_dirs:
        return env
    old_ld = env.get("LD_LIBRARY_PATH", "")
    merged = ":".join(lib_dirs + ([old_ld] if old_ld else []))
    env["LD_LIBRARY_PATH"] = merged
    return env


def run_library_baselines(
    rows,
    size,
    graph_type,
    name,
    path,
    ds_dir,
    env,
    skip_cpu,
    timeout,
    repeat,
    pr_alpha,
    pr_tol,
    pr_max_iter,
    easygraph_repo,
    warmup,
    easygraph_warmup,
    nx_cugraph_warmup,
    sssp_sources,
    bc_sources,
    closeness_sources,
    inter_run_cooldown,
    selected_functions,
    selected_baselines,
    eggpu_execution_protocol,
    dataset_stat,
    progress=None,
):
    env_lib = dict(env)
    if easygraph_repo:
        repo_path = str(Path(easygraph_repo).resolve())
        old_pp = env_lib.get("PYTHONPATH", "")
        env_lib["PYTHONPATH"] = repo_path if not old_pp else (repo_path + ":" + old_pp)
    # Do not leave GPU enabled for every library subprocess.  Each baseline gets
    # an explicit setting below so the EasyGraph CPU/C++ baselines cannot be
    # contaminated by the integrated EGGPU path.
    env_lib["EASYGRAPH_GPU_PR_MAX_ITER"] = str(pr_max_iter)
    env_lib["EASYGRAPH_GPU_PR_EPS"] = str(pr_tol)
    env_lib["EGGPU_STRICT_VALIDATION"] = "TRUE"

    for base in (
        item
        for item in AVAILABLE_BASELINES
        if item != "Gunrock" and item in selected_baselines
    ):
        for func in selected_functions:
            if (
                func == "Closeness"
                and int(closeness_sources) <= 0
                and exact_closeness_too_large(dataset_stat)
            ):
                if progress is not None:
                    for omitted in range(1, max(1, int(repeat)) + 1):
                        progress(base, func, omitted, repeat)
                start_index = len(rows)
                add_exact_closeness_scale_skip(rows, size, graph_type, name, base, ds_dir, dataset_stat)
                mark_sample_rows(rows, start_index, 1, repeat)
                continue
            structured_extra = {}
            if func == "Closeness" and int(closeness_sources) > 0:
                source_nodes = deterministic_sources(
                    int(dataset_stat.get("nodes_raw", 0) or 0),
                    int(closeness_sources),
                )
                source_payload = ",".join(str(node) for node in source_nodes).encode("ascii")
                structured_extra = {
                    "semantic": "sampled_target_exact",
                    "estimator_kind": "exact_selected_vertices",
                    "sample_sources": str(closeness_sources),
                    "source_policy": "deterministic_evenly_spaced",
                    "source_seed": "none",
                    "source_nodes_sha": hashlib.sha256(source_payload).hexdigest()[:16],
                }
            successful_e2e_samples = []
            for sample_index in range(1, max(1, int(repeat)) + 1):
                if progress is not None:
                    progress(base, func, sample_index, repeat)
                log = ds_dir / f"library_{base}_{func.lower()}_r{sample_index}.log"
                env_one = dict(env_lib)
                env_one["EGGPU_VALIDATION_DETAIL_DIR"] = str((ds_dir / "details").resolve())
                if base == "EGGPU":
                    env_one["EASYGRAPH_ENABLE_GPU"] = "TRUE"
                    env_one["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
                    env_one["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
                    env_one["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"
                    env_one["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
                    env_one["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
                    env_one["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
                    env_one["EGGPU_REQUIRE_GPU_IDLE"] = "TRUE"
                elif base == "nx-cugraph":
                    env_one["EASYGRAPH_ENABLE_GPU"] = "FALSE"
                    env_one["EASYGRAPH_GPU_STRICT_ERRORS"] = "FALSE"
                    env_one["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
                    env_one["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
                    env_one["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
                    env_one["EGGPU_REQUIRE_GPU_IDLE"] = "TRUE"
                else:
                    env_one["EASYGRAPH_ENABLE_GPU"] = "FALSE"
                    env_one["EASYGRAPH_GPU_STRICT_ERRORS"] = "FALSE"
                    env_one["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
                    env_one["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
                    env_one["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
                base_warmup = warmup if base == "EGGPU" else 0
                base_easygraph_warmup = easygraph_warmup if base == "EGGPU" else 0
                cmd = conda_python_cmd(
                    "benchmarking/library_baselines.py",
                    path,
                    graph_type,
                    "--backend",
                    base,
                    "--function",
                    func,
                    "--pr-alpha",
                    str(pr_alpha),
                    "--pr-tol",
                    str(pr_tol),
                    "--pr-max-iter",
                    str(pr_max_iter),
                    "--warmup",
                    str(base_warmup),
                    "--easygraph-warmup",
                    str(base_easygraph_warmup),
                    "--nx-cugraph-warmup",
                    str(nx_cugraph_warmup if base == "nx-cugraph" else 0),
                    "--eggpu-execution-protocol",
                    str(eggpu_execution_protocol),
                    "--sssp-sources",
                    str(sssp_sources),
                    "--bc-sources",
                    str(bc_sources),
                    "--closeness-sources",
                    str(closeness_sources),
                    "--cooldown",
                    str(inter_run_cooldown),
                    "--measurement-mode",
                    env_one.get("EGGPU_MEASUREMENT_MODE", "combined"),
                )
                if skip_cpu and base in ("igraph", "networkx"):
                    cmd.append("--skip-cpu")
                start_index = len(rows)
                if base == "EGGPU":
                    idle_ok, idle_note = check_eggpu_child_gpu_idle(env_one)
                    if not idle_ok:
                        log.write_text(idle_note + "\n")
                        raise SystemExit(idle_note)
                rc, _, process_memory = run_cmd(
                    cmd,
                    log,
                    env_one,
                    timeout=timeout,
                    cooldown=inter_run_cooldown,
                )
                txt = read_text(log)
                results = parse_library_results(txt)
                if not results:
                    note = f"{base}/{func} baseline script produced no RESULT_JSON rows"
                    if rc == 124:
                        note = f"{base}/{func} baseline timed out; {timeout_too_long_note(timeout)}"
                    for metric in (
                        "build",
                        "e2e",
                        "kernel",
                        "memory_peak_rss_mb",
                        "memory_peak_gpu_mb",
                        "memory_avg_gpu_mb",
                        "memory_peak_gpu_delta_mb",
                        "memory_avg_gpu_delta_mb",
                        "memory_peak_gpu_proc_mb",
                        "memory_avg_gpu_proc_mb",
                        "memory_peak_gpu_proc_delta_mb",
                        "memory_avg_gpu_proc_delta_mb",
                    ):
                        add_row(rows, size, graph_type, name, func, base, metric, None, "failed", log, notes=note)
                    mark_sample_rows(rows, start_index, sample_index, repeat)
                    if rc == 124:
                        if should_continue_after_isolated_timeout(
                            successful_e2e_samples, timeout
                        ):
                            continue
                        if progress is not None:
                            for omitted in range(sample_index + 1, max(1, int(repeat)) + 1):
                                progress(base, func, omitted, repeat)
                        break
                    continue
                for result in results:
                    status = result.get("status", "failed")
                    notes = result.get("notes", "")
                    seconds = result.get("seconds")
                    if rc == 124:
                        if status == "ok" and seconds is not None:
                            notes = (notes + "; " if notes else "") + timeout_after_results_note(timeout)
                        else:
                            status = "failed"
                            seconds = None
                            notes = (notes + "; " if notes else "") + timeout_too_long_note(timeout)
                    elif rc != 0 and status == "ok":
                        status = "failed"
                    result_extra = dict(structured_extra)
                    for key in (
                        "unit",
                        "metric_family",
                        "measurement_scope",
                        "timer_kind",
                        "measurement_window",
                        "timing_provenance",
                    ):
                        if result.get(key) not in (None, ""):
                            result_extra[key] = result[key]
                    for key in (
                        "gpu_exclusive_preflight_verified",
                        "gpu_exclusive_snapshot_before_worker",
                        "gpu_exclusive_postflight_verified",
                        "gpu_exclusive_snapshot_after_worker",
                    ):
                        if process_memory.get(key) not in (None, ""):
                            result_extra[key] = process_memory[key]
                    add_row(
                        rows,
                        size,
                        graph_type,
                        name,
                        result.get("function", ""),
                        result.get("backend", ""),
                        result.get("metric", "e2e"),
                        seconds,
                        status,
                        log,
                        result.get("correctness", ""),
                        notes,
                        extra=result_extra or None,
                    )
                if env_one.get("EGGPU_MEASUREMENT_MODE", "combined") == "memory":
                    result_statuses = [str(result.get("status", "failed")) for result in results]
                    if any(status == "ok" for status in result_statuses):
                        memory_status = "ok"
                    elif result_statuses and set(result_statuses) <= {"skipped", "unsupported"}:
                        memory_status = "unsupported" if "unsupported" in result_statuses else "skipped"
                    else:
                        memory_status = "failed"
                    correctness = next(
                        (str(result.get("correctness", "")) for result in results if result.get("correctness")),
                        "",
                    )
                    add_memory_metric_rows(
                        rows,
                        size,
                        graph_type,
                        name,
                        func,
                        base,
                        log,
                        process_memory,
                        status=memory_status,
                        correctness=correctness,
                        notes=(
                            "independent parent-process monitor over the complete isolated benchmark subprocess; "
                            "includes graph loading/build, untimed warmup, algorithm call(s), and result emission"
                        ),
                        measurement_window="isolated_memory_subprocess",
                        extra=structured_extra,
                    )
                mark_sample_rows(rows, start_index, sample_index, repeat)
                successful_e2e_samples.extend(
                    result.get("seconds")
                    for result in results
                    if (
                        result.get("metric") == "e2e"
                        and result.get("status", "failed") == "ok"
                        and result.get("seconds") is not None
                    )
                )
                result_statuses = {str(result.get("status", "failed")) for result in results}
                if result_statuses and result_statuses <= {"skipped", "unsupported"}:
                    if progress is not None:
                        for omitted in range(sample_index + 1, max(1, int(repeat)) + 1):
                            progress(base, func, omitted, repeat)
                    break
                if rc == 124:
                    if should_continue_after_isolated_timeout(
                        successful_e2e_samples, timeout
                    ):
                        continue
                    if progress is not None:
                        for omitted in range(sample_index + 1, max(1, int(repeat)) + 1):
                            progress(base, func, omitted, repeat)
                    break


def run_integrated_eggpu_stability_audit(
    args,
    out_dir,
    *,
    expected_cells,
    environment,
):
    """Run the paper gate after raw samples and frozen provenance are durable."""

    audit_csv = Path(out_dir) / "eggpu_timing_stability.csv"
    audit_json = Path(out_dir) / "eggpu_timing_stability.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("audit_eggpu_timing_stability.py")),
        "--main-result-dir",
        str(Path(out_dir).resolve()),
        "--metric",
        "e2e",
        "--expected-samples",
        str(EXPECTED_TIMING_SAMPLES),
        "--expected-cells",
        str(int(expected_cells)),
        "--max-over-median-limit",
        str(args.stability_max_over_median_limit),
        "--median-over-min-limit",
        str(args.stability_median_over_min_limit),
        "--output-csv",
        str(audit_csv.resolve()),
        "--output-json",
        str(audit_json.resolve()),
    ]
    audit_log = Path(out_dir) / "eggpu_timing_stability.log"
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parent),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
        audit_output = completed.stdout or ""
        audit_returncode = completed.returncode
        invocation_failure = ""
    except subprocess.TimeoutExpired as exc:
        audit_output = exc.stdout or ""
        if isinstance(audit_output, bytes):
            audit_output = audit_output.decode("utf-8", errors="replace")
        invocation_failure = (
            "integrated stability audit exceeded the 120-second timeout"
        )
        audit_output += f"\n[fail] {invocation_failure}\n"
        audit_returncode = 124
    except OSError as exc:
        invocation_failure = f"integrated stability audit could not start: {exc}"
        audit_output = f"[fail] {invocation_failure}\n"
        audit_returncode = 126
    audit_log.write_text(audit_output, encoding="utf-8")
    summary = {}
    if not invocation_failure:
        if audit_json.is_file():
            try:
                summary = json.loads(audit_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                invocation_failure = (
                    f"integrated stability audit JSON is invalid: {exc}"
                )
                audit_returncode = audit_returncode or 1
        else:
            invocation_failure = (
                "integrated stability audit did not produce its JSON result"
            )
            audit_returncode = audit_returncode or 1
    if invocation_failure:
        summary = {
            "status": "fail",
            "failure_reasons": [invocation_failure],
            "batch_acceptance": "rejected_entire_batch",
        }
    summary["audit_command"] = command
    summary["audit_returncode"] = audit_returncode
    summary["audit_log"] = str(audit_log.resolve())
    return summary


def main():
    global PER_FUNCTION_TIMEOUT_SECONDS, RUN_GUNROCK_BASELINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="7")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--repeat", type=int, default=5, help="Independent measured subprocesses per supported baseline/function/dataset pair.")
    ap.add_argument("--warmup", type=int, default=0, help="Legacy EGGPU warmup input; the effective EGGPU warmup is max(this, --easygraph-warmup).")
    ap.add_argument(
        "--skip-large-cpu",
        action="store_true",
        help="Explicitly skip CPU-only library baselines on large graphs. Default is to run every requested baseline on every dataset.",
    )
    ap.add_argument(
        "--library-timeout",
        type=int,
        default=PER_FUNCTION_TIMEOUT_SECONDS,
        help="Per backend/function timeout in seconds for every baseline, including Gunrock subprocesses.",
    )
    ap.add_argument("--pr-alpha", type=float, default=0.75, help="PageRank damping factor for EGGPU and compatible baselines.")
    ap.add_argument("--pr-eps", type=float, default=1e-6, help="PageRank convergence tolerance for EGGPU + compatible baselines.")
    ap.add_argument("--pr-max-iter", type=int, default=200, help="PageRank max iterations for EGGPU + compatible baselines.")
    ap.add_argument(
        "--easygraph-repo",
        default=str(WORKSPACE_ROOT / "Easy-Graph"),
        help="Path to local Easy-Graph repo to prepend into PYTHONPATH for library baselines.",
    )
    ap.add_argument("--easygraph-warmup", type=int, default=2, help="Untimed warmup calls before each independent EGGPU sample; excluded from timing.")
    ap.add_argument(
        "--nx-cugraph-warmup",
        type=int,
        default=3,
        help=(
            "Untimed strict public calls on a prepared native nx-cugraph "
            "graph before every fresh-process sample."
        ),
    )
    ap.add_argument("--sssp-sources", type=int, default=8, help="Number of deterministic sources for SSSP benchmarks.")
    ap.add_argument("--bc-sources", type=int, default=16, help="Number of deterministic sources for BC source-sampled benchmarks.")
    ap.add_argument(
        "--closeness-sources",
        type=int,
        default=0,
        help=(
            "If greater than zero, run Closeness for this many deterministic target vertices. "
            "The default zero retains exact all-node semantics and its symmetric scale guard."
        ),
    )
    ap.add_argument(
        "--inter-run-cooldown",
        type=float,
        default=0.2,
        help="Cooldown seconds between subprocess runs to reduce cross-run interference.",
    )
    ap.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset filters (name/size/type). Example: ca-GrQc,wiki-Vote or small,directed. Default: all.",
    )
    ap.add_argument(
        "--functions",
        default="all",
        help="Comma-separated functions from DEFAULT_FUNCTIONS; CC expands to WCC,SCC. Default: all.",
    )
    ap.add_argument(
        "--baselines",
        default="all",
        help=(
            "Comma-separated baselines. Choices: "
            + ", ".join(AVAILABLE_BASELINES)
            + ". Default: all."
        ),
    )
    ap.add_argument(
        "--eggpu-execution-protocol",
        choices=["steady-state", "first-use"],
        default="steady-state",
        help=(
            "steady-state prebuilds reusable EGGPU graph state and applies EGGPU warmups; "
            "first-use measures the first supported function call with neither step."
        ),
    )
    ap.add_argument(
        "--measurement-mode",
        choices=["timing", "memory", "combined"],
        default="combined",
        help="timing disables all memory sampling; memory emits only memory rows; combined is legacy behavior.",
    )
    ap.add_argument(
        "--eggpu-stable-timing-protocol",
        action="store_true",
        help=(
            "Run one controlled EGGPU-only five-sample timing batch, record "
            "CPU-affinity/NUMA/thread provenance, and reject the whole batch when "
            "the catastrophic-outlier guard fails."
        ),
    )
    ap.add_argument(
        "--expected-cpu-affinity",
        default="",
        help=(
            "Optional exact taskset CPU list (for example 32-39,96-103). "
            "The controlled protocol fails before measurement if it differs "
            "from sched_getaffinity(0)."
        ),
    )
    ap.add_argument(
        "--expected-numa-nodes",
        default="",
        help=(
            "Optional exact memory-node list. The controlled protocol fails "
            "before measurement unless numactl or the verified libnuma policy "
            "confirms it."
        ),
    )
    ap.add_argument(
        "--stability-max-over-median-limit",
        type=float,
        default=DEFAULT_MAX_OVER_MEDIAN_LIMIT,
        help="Complete-raw5 max/median guard (formal default: 5.0).",
    )
    ap.add_argument(
        "--stability-median-over-min-limit",
        type=float,
        default=DEFAULT_MEDIAN_OVER_MIN_LIMIT,
        help=(
            "Complete-raw5 median/min guard (calibrated formal default: 3.0)."
        ),
    )
    args = ap.parse_args()
    args.stable_timing_protocol_metadata = {}
    if args.eggpu_stable_timing_protocol:
        if args.repeat != EXPECTED_TIMING_SAMPLES:
            ap.error(
                "--eggpu-stable-timing-protocol requires "
                f"--repeat {EXPECTED_TIMING_SAMPLES}"
            )
        if args.measurement_mode != "timing":
            ap.error(
                "--eggpu-stable-timing-protocol requires "
                "--measurement-mode timing"
            )
        if args.stability_max_over_median_limit < 1:
            ap.error("--stability-max-over-median-limit must be at least 1")
        if args.stability_median_over_min_limit < 1:
            ap.error("--stability-median-over-min-limit must be at least 1")
        apply_controlled_thread_environment(os.environ)
        os.environ["EGGPU_STABLE_TIMING_PROTOCOL"] = "TRUE"
        os.environ["EGGPU_EXPECTED_CPU_AFFINITY"] = (
            args.expected_cpu_affinity
        )
        os.environ["EGGPU_EXPECTED_NUMA_NODES"] = args.expected_numa_nodes
        placement = collect_execution_placement(os.environ)
        try:
            validate_controlled_execution(
                placement,
                expected_cpu_affinity=args.expected_cpu_affinity,
                expected_numa_nodes=args.expected_numa_nodes,
            )
        except ValueError as exc:
            ap.error(str(exc))
        args.stable_timing_protocol_metadata = controlled_protocol_metadata(
            placement=placement,
            expected_cpu_affinity=args.expected_cpu_affinity,
            expected_numa_nodes=args.expected_numa_nodes,
            max_over_median_limit=args.stability_max_over_median_limit,
            median_over_min_limit=args.stability_median_over_min_limit,
        )
    PER_FUNCTION_TIMEOUT_SECONDS = int(args.library_timeout)
    os.environ["EGGPU_MEASUREMENT_MODE"] = args.measurement_mode
    os.environ["EGGPU_EXECUTION_PROTOCOL"] = args.eggpu_execution_protocol

    # Optional idle CUDA context so nvitop shows the long-lived benchmark driver
    # even while CPU-only baselines are running.  It does not launch kernels.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("EGGPU_MONITOR_GPU_INDEX", str(args.gpu))
    if os.environ.get("EGGPU_EXTERNAL_VISIBILITY_MARKER", "").strip().upper() in TRUE_VALUES:
        visibility_marker = None
    else:
        visibility_marker = GpuVisibilityMarker(args.gpu, "run_full_baselines.py").start()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else ROOT / "benchmarking" / "results" / f"{ts}_full_baseline_gpu{args.gpu}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_measurement_schema(out_dir / "measurement_schema.json")
    logs = out_dir / "logs"
    logs.mkdir(exist_ok=True)

    env = sanitized_subprocess_env(os.environ)
    env["EGGPU_MEASUREMENT_MODE"] = args.measurement_mode
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    local_cuda_libs = []
    cuda_root = local_cuda_root()
    if cuda_root is not None:
        cuda_root_str = str(cuda_root)
        env["EGGPU_CUDA_ROOT"] = cuda_root_str
        env["CUDA_PATH"] = cuda_root_str
        env["CUDA_HOME"] = cuda_root_str
        env["CUPY_CUDA_PATH"] = cuda_root_str
        env["CUDAToolkit_ROOT"] = cuda_root_str
        env["CONDA_PREFIX"] = cuda_root_str
        local_cuda_libs.extend([
            str(cuda_root / "lib"),
            str(cuda_root / "targets" / "x86_64-linux" / "lib"),
        ])
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([p for p in local_cuda_libs if Path(p).exists()] + ([existing_ld] if existing_ld else []))

    fn_tokens = parse_csv_tokens(args.functions)
    if not fn_tokens or any(x.lower() == "all" for x in fn_tokens):
        selected_functions = list(DEFAULT_FUNCTIONS)
    else:
        selected_functions = []
        for tok in fn_tokens:
            alias = LEGACY_FUNCTION_ALIASES.get(tok.upper())
            if alias is not None:
                for f in alias:
                    if f not in selected_functions:
                        selected_functions.append(f)
                continue
            hit = next((f for f in DEFAULT_FUNCTIONS if f.lower() == tok.lower()), None)
            if hit is None:
                raise SystemExit(
                    f"Unknown function token: {tok}. Choose from {', '.join(DEFAULT_FUNCTIONS)}, CC, or all."
                )
            if hit not in selected_functions:
                selected_functions.append(hit)

    baseline_tokens = parse_csv_tokens(args.baselines)
    if not baseline_tokens or any(x.lower() == "all" for x in baseline_tokens):
        selected_baselines = list(AVAILABLE_BASELINES)
    else:
        selected_baselines = []
        by_lower = {name.lower(): name for name in AVAILABLE_BASELINES}
        for token in baseline_tokens:
            hit = by_lower.get(token.lower())
            if hit is None:
                raise SystemExit(
                    f"Unknown baseline token: {token}. Choose from "
                    + ", ".join(AVAILABLE_BASELINES)
                    + ", or all."
                )
            if hit not in selected_baselines:
                selected_baselines.append(hit)
    if (
        "EGGPU" in selected_baselines
        and args.eggpu_execution_protocol == "first-use"
        and (int(args.warmup) != 0 or int(args.easygraph_warmup) != 0)
    ):
        raise SystemExit(
            "--eggpu-execution-protocol first-use requires --warmup 0 "
            "and --easygraph-warmup 0"
        )
    args.selected_baselines = selected_baselines
    if (
        args.eggpu_stable_timing_protocol
        and selected_baselines != ["EGGPU"]
    ):
        raise SystemExit(
            "--eggpu-stable-timing-protocol requires "
            "--baselines EGGPU"
        )
    RUN_GUNROCK_BASELINE = "Gunrock" in selected_baselines

    ds_tokens = {x.lower() for x in parse_csv_tokens(args.datasets)}
    if not ds_tokens or "all" in ds_tokens:
        datasets = list(DEFAULT_DATASETS)
    else:
        datasets = []
        for size, graph_type, name, path in DEFAULT_DATASETS:
            keys = {size.lower(), graph_type.lower(), name.lower()}
            if keys & ds_tokens:
                datasets.append((size, graph_type, name, path))
        if not datasets:
            raise SystemExit(f"No datasets matched --datasets={args.datasets}")

    run_metadata = collect_run_metadata(args, out_dir, datasets, selected_functions, env)
    write_run_metadata(out_dir, run_metadata)
    write_baseline_version_manifest(out_dir, run_metadata)

    rows = []
    notes = []
    stats = []
    total_main_tasks = (
        len(datasets)
        * len(selected_functions)
        * len(selected_baselines)
        * max(1, args.repeat)
    )
    progress = ProgressReporter("main", total_main_tasks)

    for dataset_index, (size, graph_type, name, path) in enumerate(datasets, start=1):
        st = dataset_stats(path)
        stats.append({"size": size, "graph_type": graph_type, "name": name, "path": path, **st})
        ds_dir = logs / name
        ds_dir.mkdir(exist_ok=True)
        print(
            f"\n=== Dataset {dataset_index}/{len(datasets)}: {name} ({graph_type}, {size}) ===",
            flush=True,
        )

        def progress_step(backend, function, sample_index=1, sample_count=1):
            progress.tick(
                f"dataset {dataset_index}/{len(datasets)} {name}: "
                f"{backend}/{function} sample {sample_index}/{sample_count}"
            )

        # Gunrock baselines (when executables are available and semantically applicable).
        if graph_type == "directed":
            notes.append(f"{name} MST uses undirected projection of the directed edge list.")
            notes.append(f"{name} LCC uses undirected projection of the directed edge list.")
        if st["selfloops"] > 0:
            notes.append(
                f"{name} has {st['selfloops']:,} raw self-loop edge row(s); "
                "benchmark graph construction and Gunrock MatrixMarket conversion remove self-loops."
            )
        if "PageRank" in selected_functions:
            run_repeated_gunrock(
                rows,
                args.repeat,
                ds_dir,
                "PageRank",
                progress_step,
                lambda sample_dir: maybe_run_gunrock_pr(
                    rows,
                    size,
                    graph_type,
                    name,
                    path,
                    sample_dir,
                    env,
                    1,
                    args.inter_run_cooldown,
                    args.pr_alpha,
                    args.pr_eps,
                ),
            )
        if "MST" in selected_functions:
            run_repeated_gunrock(
                rows,
                args.repeat,
                ds_dir,
                "MST",
                progress_step,
                lambda sample_dir: maybe_run_gunrock_mst(
                    rows, size, graph_type, name, path, sample_dir, env, args.inter_run_cooldown
                ),
            )
        if "LCC" in selected_functions:
            run_repeated_gunrock(
                rows,
                args.repeat,
                ds_dir,
                "LCC",
                progress_step,
                lambda sample_dir: maybe_run_gunrock_lcc(
                    rows, size, graph_type, name, path, sample_dir, env, args.inter_run_cooldown
                ),
            )
        for component_func in ("WCC", "SCC"):
            if component_func in selected_functions:
                run_repeated_gunrock(
                    rows,
                    args.repeat,
                    ds_dir,
                    component_func,
                    progress_step,
                    lambda sample_dir, component_func=component_func: add_unavailable_gunrock(
                        rows, size, graph_type, name, component_func, sample_dir
                    ),
                )
        if "BFS" in selected_functions:
            run_repeated_gunrock(
                rows,
                args.repeat,
                ds_dir,
                "BFS",
                progress_step,
                lambda sample_dir: maybe_run_gunrock_bfs(
                    rows, size, graph_type, name, path, sample_dir, env, args.inter_run_cooldown, args.sssp_sources
                ),
            )
        if "Dijkstra" in selected_functions:
            run_repeated_gunrock(
                rows,
                args.repeat,
                ds_dir,
                "Dijkstra",
                progress_step,
                lambda sample_dir: maybe_run_gunrock_sssp(
                    rows,
                    size,
                    graph_type,
                    name,
                    path,
                    sample_dir,
                    env,
                    args.inter_run_cooldown,
                    1,
                    function="Dijkstra",
                ),
            )
        if "BellmanFord" in selected_functions:
            run_repeated_gunrock(
                rows,
                args.repeat,
                ds_dir,
                "BellmanFord",
                progress_step,
                lambda sample_dir: maybe_run_gunrock_sssp(
                    rows,
                    size,
                    graph_type,
                    name,
                    path,
                    sample_dir,
                    env,
                    args.inter_run_cooldown,
                    args.sssp_sources,
                    function="BellmanFord",
                ),
            )
        for structural_func in ("EffectiveSize", "Efficiency", "Constraint", "Hierarchy"):
            if structural_func in selected_functions:
                run_repeated_gunrock(
                    rows,
                    args.repeat,
                    ds_dir,
                    structural_func,
                    progress_step,
                    lambda sample_dir, structural_func=structural_func: add_unavailable_gunrock(
                        rows, size, graph_type, name, structural_func, sample_dir
                    ),
                )
        if "SSSP" in selected_functions:
            run_repeated_gunrock(
                rows,
                args.repeat,
                ds_dir,
                "SSSP",
                progress_step,
                lambda sample_dir: maybe_run_gunrock_sssp(
                    rows, size, graph_type, name, path, sample_dir, env, args.inter_run_cooldown, args.sssp_sources
                ),
            )
        if "KCore" in selected_functions:
            run_repeated_gunrock(
                rows,
                args.repeat,
                ds_dir,
                "KCore",
                progress_step,
                lambda sample_dir: maybe_run_gunrock_kcore(
                    rows, size, graph_type, name, path, sample_dir, env, args.inter_run_cooldown
                ),
            )
        if "BC" in selected_functions:
            run_repeated_gunrock(
                rows,
                args.repeat,
                ds_dir,
                "BC",
                progress_step,
                lambda sample_dir: maybe_run_gunrock_bc(
                    rows, size, graph_type, name, path, sample_dir, env, args.inter_run_cooldown, args.bc_sources
                ),
            )
        if "Closeness" in selected_functions:
            run_repeated_gunrock(
                rows,
                args.repeat,
                ds_dir,
                "Closeness",
                progress_step,
                lambda sample_dir: add_unavailable_gunrock(
                    rows, size, graph_type, name, "Closeness", sample_dir
                ),
            )

        skip_cpu = args.skip_large_cpu and st["edges_undirected_unique"] > 300000
        if skip_cpu:
            notes.append(f"{name} CPU library baselines skipped for all functions: graph has {st['edges_undirected_unique']:,} unique undirected edges.")
        run_library_baselines(
            rows,
            size,
            graph_type,
            name,
            path,
            ds_dir,
            env,
            skip_cpu,
            args.library_timeout,
            args.repeat,
            args.pr_alpha,
            args.pr_eps,
            args.pr_max_iter,
            args.easygraph_repo,
            args.warmup,
            args.easygraph_warmup,
            args.nx_cugraph_warmup,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
            args.inter_run_cooldown,
            selected_functions,
            selected_baselines,
            args.eggpu_execution_protocol,
            st,
            progress=progress_step,
        )

    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "notes.txt").write_text("\n".join(notes) + ("\n" if notes else ""))
    if args.measurement_mode == "timing":
        rows = [row for row in rows if not str(row.get("metric", "")).startswith("memory_")]
    elif args.measurement_mode == "memory":
        rows = [row for row in rows if str(row.get("metric", "")).startswith("memory_")]
    for row in rows:
        row["measurement_phase"] = args.measurement_mode
    sample_rows = rows
    if not sample_rows:
        empty_fields = [
            "dataset_size",
            "graph_type",
            "dataset",
            "function",
            "baseline",
            "metric",
            "seconds",
            "value",
            "unit",
            "metric_family",
            "measurement_scope",
            "timer_kind",
            "measurement_window",
            "status",
            "correctness",
            "log",
            "notes",
            "semantic",
            "skip_reason",
            "estimator_kind",
            "sample_sources",
            "source_policy",
            "source_seed",
            "source_nodes_sha",
            "sample_index",
            "sample_count",
            "measurement_phase",
        ]
        for filename in (
            "results_samples.csv",
            "results_long.csv",
            "results_build.csv",
            "results_kernel.csv",
            "results_e2e.csv",
            "results_memory.csv",
        ):
            with (out_dir / filename).open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=empty_fields).writeheader()
        run_metadata["completed_at"] = datetime.now().isoformat(timespec="seconds")
        run_metadata["artifacts"] = {
            "results_long_rows": 0,
            "validation_error": "",
            "plot_error": "",
            "files": [
                "measurement_schema.json",
                "dataset_stats.json",
                "notes.txt",
                "results_samples.csv",
                "results_long.csv",
                "results_build.csv",
                "results_kernel.csv",
                "results_e2e.csv",
                "results_memory.csv",
                "baseline_versions.json",
            ],
        }
        write_run_metadata(out_dir, run_metadata)
        write_baseline_version_manifest(out_dir, run_metadata)
        if args.eggpu_stable_timing_protocol:
            stability_summary = run_integrated_eggpu_stability_audit(
                args,
                out_dir,
                expected_cells=len(datasets) * len(selected_functions),
                environment=env,
            )
            run_metadata["stable_timing_protocol"]["audit"] = stability_summary
            run_metadata["artifacts"]["files"].extend(
                [
                    "eggpu_timing_stability.csv",
                    "eggpu_timing_stability.json",
                    "eggpu_timing_stability.log",
                ]
            )
            write_run_metadata(out_dir, run_metadata)
            write_baseline_version_manifest(out_dir, run_metadata)
        print(f"\nDone with no rows after measurement-mode filtering: {out_dir}", flush=True)
        if args.eggpu_stable_timing_protocol:
            raise SystemExit(2)
        return
    with (out_dir / "results_samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    rows = aggregate_sample_rows(sample_rows, expected_samples=args.repeat)
    with (out_dir / "results_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_metric_csvs_no_pandas(out_dir, rows)

    validation_error = None
    try:
        try:
            from benchmarking.validate_correctness import write_validation_outputs
        except ModuleNotFoundError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from validate_correctness import write_validation_outputs

        write_validation_outputs(out_dir, rows)
        # Keep raw samples/aggregates intact, while ensuring the convenience
        # metric tables used for comparisons contain only validated rows.
        write_metric_csvs_no_pandas(out_dir, rows)
    except Exception as e:
        validation_error = f"correctness validation generation failed: {e}"
        print(f"[warn] {validation_error}", flush=True)

    plot_error = None
    try:
        plot_error = write_plot_and_tables_isolated(out_dir, rows, datasets, args.gpu)
        if plot_error:
            print(f"[warn] {plot_error}", flush=True)
    except ModuleNotFoundError as e:
        plot_error = f"missing optional plotting dependency: {e}"
        print(f"[warn] {plot_error}", flush=True)
    except Exception as e:
        plot_error = f"plot/table generation failed: {e}"
        print(f"[warn] {plot_error}", flush=True)

    summary = [
        f"# Full baseline run on GPU {args.gpu}",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Datasets:",
    ]
    for st in stats:
        summary.append(
            f"- {st['name']} ({st['graph_type']}, {st['size']}): nodes(raw)={st['nodes_raw']:,}, "
            f"edge rows={st['edge_rows']:,}, non-self edge rows={st['edge_rows_no_selfloops']:,}, "
            f"selfloops={st['selfloops']:,}, undirected unique={st['edges_undirected_unique']:,}"
        )
    summary += [
        "",
        "Artifacts:",
        "- `measurement_schema.json` defines units, timer provenance, memory attribution, and aggregation semantics.",
        "- `results_samples.csv` preserves independent measurements; `results_long.csv` contains registered aggregate statistics.",
        "- `results_build.csv`, `results_kernel.csv`, `results_e2e.csv`, and `results_memory.csv` contain validated comparison-eligible rows (plus explicit timeout rows); failed, inconclusive, and semantic-mismatch measurements remain in the raw files.",
        "- `correctness_validation.csv`, `correctness_validation.md`",
        "- `runtime_build.png/.pdf`, `runtime_kernel.png/.pdf`, `runtime_e2e.png/.pdf`, `runtime_memory.png/.pdf`, `speedup_heatmap.png/.pdf`",
        "- `main_table.tex`",
        "- per-run logs under `logs/`",
        "",
        "Notes:",
        "- All benchmarks were run serially with `CUDA_VISIBLE_DEVICES` fixed to the requested GPU.",
        f"- Added inter-run cooldown: {args.inter_run_cooldown}s.",
        (
            f"- Repeat policy: {args.repeat} independent subprocess samples per "
            "EGGPU pair; the submission estimator is the minimum of the same "
            "complete batch, subject to the catastrophic-outlier guard. "
            "Mean, sample SD, and CV remain recorded diagnostics."
            if args.eggpu_stable_timing_protocol
            else f"- Repeat policy: {args.repeat} independent subprocess sample(s) "
            "per supported pair; the generic aggregate is the arithmetic mean "
            "and incomplete groups are not publishable."
        ),
        f"- EGGPU execution protocol: `{args.eggpu_execution_protocol}`.",
        (
            f"- Warmup policy: EGGPU only ({max(args.warmup, args.easygraph_warmup)} untimed calls before every sample); all other baselines run without warmup."
            if args.eggpu_execution_protocol == "steady-state"
            else "- Warmup policy: first-use EGGPU uses zero graph-state prewarm and zero function warmup; all other selected baselines also run without warmup."
        ),
        "- Child benchmark processes explicitly unset CFLAGS/CPPFLAGS/CXXFLAGS/CPATH/LIBRARY_PATH and pin CUDA_PATH/CUDA_HOME/CUPY_CUDA_PATH/CONDA_PREFIX to the selected local CUDA toolkit, avoiding host-toolchain and conda-header contamination of CUDA JIT/compilation paths. The default runner uses the selected EGGPU Python directly; `conda run` is only used when `EGGPU_USE_CONDA_RUN=TRUE` is explicitly set.",
        f"- PageRank hyperparameters were aligned where supported: alpha={args.pr_alpha}, tol/eps={args.pr_eps}, max_iter={args.pr_max_iter}.",
        f"- SSSP uses deterministic weighted edges with {args.sssp_sources} sampled sources; BC uses source-sampled mode with {args.bc_sources} sampled sources (normalized=False).",
        (
            f"- Closeness uses sampled-target exact semantics on {args.closeness_sources} deterministic vertices."
            if args.closeness_sources > 0
            else "- Closeness uses exact all-node semantics subject to the symmetric predeclared scale guard."
        ),
        f"- Selected datasets: {', '.join(d[2] for d in datasets)}.",
        f"- Selected functions: {', '.join(selected_functions)}.",
        f"- Selected baselines: {', '.join(selected_baselines)}.",
        "- EGGPU is enabled only by `EASYGRAPH_ENABLE_GPU=TRUE`; strict experiment mode rejects any C++/CPU fallback.",
        "- Timing schema uses three metrics only: `build` (baseline-native graph construction), `kernel` (device kernel time where exposed; otherwise algorithm wall-time surrogate), and `e2e` (user-visible function call wall-time, including per-call prep/transfer/sync). In-process library baselines exclude import/file parsing from `build`; Gunrock's standalone boundary explicitly begins immediately before MatrixMarket loading.",
        (
            "- Under steady-state EGGPU, `build` is graph preparation: Python graph construction plus GraphContext and C++ graph-container prebuild."
            if args.eggpu_execution_protocol == "steady-state"
            else "- Under first-use EGGPU, `build` contains only Python graph construction; GraphContext/C++/CSR/device initialization remains inside the first measured E2E call."
        ),
        "- Function-specific preparation like CSR conversion remains in `e2e` because it is paid at function-call time, not at baseline graph-object construction time.",
        "- CPU backends do not expose a kernel concept, so CPU `kernel` equals algorithm wall-time by definition.",
        "- Gunrock reports all three native phases: `build` spans MatrixMarket loading through host CSR and device-graph materialization; `kernel` is the executable's native synchronized device interval; `e2e` spans MatrixMarket loading through complete host-result availability. Validation, result printing, result-file I/O, and external process-launch wall time are excluded.",
        f"- Per-function timeout is {PER_FUNCTION_TIMEOUT_SECONDS}s for all baselines; timed-out runs are marked with `TIMEOUT_TOO_LONG` in logs/notes.",
        "- Directed PageRank preserves edge direction. WCC ignores edge direction, SCC preserves directed reachability; on undirected graphs SCC=WCC. MST and LCC use an undirected projection on directed datasets.",
        "- Raw self-loop rows are preserved in source files for provenance but removed during benchmark graph construction and Gunrock MatrixMarket conversion for all libraries/functions.",
        "- Exact all-source Closeness uses a symmetric predeclared scale guard: "
        f"EGGPU_CLOSENESS_EXACT_MAX_NODES={env.get('EGGPU_CLOSENESS_EXACT_MAX_NODES', os.environ.get('EGGPU_CLOSENESS_EXACT_MAX_NODES', '1000000'))}, "
        f"EGGPU_CLOSENESS_EXACT_MAX_WORK={env.get('EGGPU_CLOSENESS_EXACT_MAX_WORK', os.environ.get('EGGPU_CLOSENESS_EXACT_MAX_WORK', '50000000000'))}. "
        "Guarded rows remain `semantic=exact_all_node` with `skip_reason=exact_scale_guard`; large-graph evidence is reported separately as `sampled_target_exact`.",
        "- nx-cugraph is measured only through NetworkX backend dispatch; native cuGraph and NetworkX CPU fallback are forbidden in that baseline.",
        "- Gunrock baselines are discovered per executable (`pr`, `mst`, `lcc`, `bfs`, `sssp`, `kcore`, `bc`) across configured build directories; each executable's path and SHA-256 are recorded. WCC/SCC and Closeness have no aligned executable in the preferred modern artifact and are marked unavailable; legacy CC is not treated as modern Gunrock support.",
        "- Gunrock MST is comparable only on connected inputs. Disconnected inputs are skipped because the benchmark requires a full minimum spanning forest; no largest-component fallback is timed.",
        "- The primary comparable memory metrics are process-tree absolute peaks: `memory_peak_rss_mb` for CPU memory and `memory_peak_gpu_proc_mb` for GPU memory. Start values, deltas, sample counts, polling interval, and monitor window are also recorded. Whole-device GPU values are diagnostic only because they may include unrelated processes.",
        f"- GPU visibility marker: enabled={env.get('EGGPU_GPU_VISIBILITY_MARKER', '') or 'FALSE'}, marker_mb={env.get('EGGPU_GPU_VISIBILITY_MARKER_MB', '0')}, whole_device_adjust_mb={env.get('EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB', env.get('EGGPU_GPU_VISIBILITY_MARKER_MB', '0'))}. Whole-device absolute metrics subtract the measured marker footprint. Process-tree absolute metrics subtract only the fixed allocation when the measured process owns the marker; delta metrics are unadjusted.",
        f"- `igraph`, `networkx`, `EGGPU`, `easygraph-cpu`, `easygraph-cpp`, and `nx-cugraph` baselines are run through `benchmarking/library_baselines.py` with a per backend/function timeout of {args.library_timeout} seconds.",
        "",
        "Reproducibility:",
        f"- Executed source snapshot (SHA-256): `{run_metadata['source_snapshot']['digest']}` over {run_metadata['source_snapshot']['file_count']} source files.",
        "- Run metadata: `run_metadata.json`.",
    ]
    if plot_error:
        summary.append(f"- Plot/table generation skipped: {plot_error}.")
    if validation_error:
        summary.append(f"- Correctness validation skipped: {validation_error}.")
    if notes:
        summary += [""] + [f"- {n}" for n in notes]
    (out_dir / "summary.md").write_text("\n".join(summary) + "\n")
    run_metadata["completed_at"] = datetime.now().isoformat(timespec="seconds")
    run_metadata["artifacts"] = {
        "results_long_rows": len(rows),
        "validation_error": validation_error or "",
        "plot_error": plot_error or "",
        "files": [
            "measurement_schema.json",
            "dataset_stats.json",
            "notes.txt",
            "results_samples.csv",
            "results_long.csv",
            "results_build.csv",
            "results_kernel.csv",
            "results_e2e.csv",
            "results_memory.csv",
            "correctness_validation.csv",
            "correctness_validation.md",
            "summary.md",
            "baseline_versions.json",
        ],
    }
    with temporary_import_root(args.easygraph_repo):
        final_active_cpp_artifact = active_cpp_easygraph_artifact()
    run_metadata["build_artifacts"] = {
        "cpp_easygraph": cpp_easygraph_artifacts(),
        "active_cpp_easygraph": final_active_cpp_artifact,
        "gunrock_executables": gunrock_executable_artifacts(),
    }
    write_run_metadata(out_dir, run_metadata)
    write_baseline_version_manifest(out_dir, run_metadata)
    stability_summary = {}
    if args.eggpu_stable_timing_protocol:
        stability_summary = run_integrated_eggpu_stability_audit(
            args,
            out_dir,
            expected_cells=len(datasets) * len(selected_functions),
            environment=env,
        )
        run_metadata["stable_timing_protocol"]["audit"] = stability_summary
        run_metadata["artifacts"]["files"].extend(
            [
                "eggpu_timing_stability.csv",
                "eggpu_timing_stability.json",
                "eggpu_timing_stability.log",
            ]
        )
        write_run_metadata(out_dir, run_metadata)
        write_baseline_version_manifest(out_dir, run_metadata)
    print(f"\nDone: {out_dir}", flush=True)
    if (
        args.eggpu_stable_timing_protocol
        and (
            stability_summary.get("status") != "pass"
            or stability_summary.get("audit_returncode") != 0
        )
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
