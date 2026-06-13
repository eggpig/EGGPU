#!/usr/bin/env python3
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import platform
import re
import signal
from collections import defaultdict, deque
import shutil
import subprocess
import sys
import tempfile
import time
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
from gpu_device_profile import collect_gpu_device_profile


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
CONDA_EXE = os.environ.get("CONDA_EXE") or os.environ.get("_CONDA_EXE") or shutil.which("conda") or "conda"
DIRECT_CHILD_PYTHON = (
    os.environ.get("EGGPU_CHILD_PYTHON")
    or os.environ.get("COMMON_PY")
    or sys.executable
)


def _default_local_cuda_root():
    for key in ("EGGPU_CUDA_ROOT", "CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT", "CONDA_PREFIX"):
        value = os.environ.get(key, "").strip()
        if value:
            path = Path(value).expanduser()
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
    "CFLAGS",
    "CPPFLAGS",
    "CXXFLAGS",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "CPATH",
    "LIBRARY_PATH",
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
LEGACY_FUNCTION_ALIASES = {"CC": ("WCC", "SCC")}
PER_FUNCTION_TIMEOUT_SECONDS = 100
_NVML_INITIALIZED = False


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
            used_mb = max(0.0, raw_used_mb - marker_adjust_mb)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = int(getattr(util, "gpu", 0))
            procs = []
            own_pid = os.getpid()
            for proc_info in _nvml_compute_processes(handle):
                try:
                    pid = int(getattr(proc_info, "pid", -1))
                except Exception:
                    pid = -1
                if pid > 0 and pid != own_pid:
                    pname = ""
                    try:
                        pname = psutil.Process(pid).name() if psutil is not None else ""
                    except Exception:
                        pname = ""
                    mem_mb = _safe_used_gpu_memory_bytes(proc_info) / (1024.0 * 1024.0)
                    procs.append(f"pid={pid},name={pname},mem_mb={mem_mb:.1f}")
        except Exception as exc:
            return False, f"gpu_busy_before_eggpu_child: unable to query GPU idleness: {exc}"

        last_sample = (gpu_index, used_mb, raw_used_mb, marker_adjust_mb, gpu_util, procs, attempt)
        if not procs and used_mb <= max_mem_mb and gpu_util <= max_util:
            retry_note = "" if attempt == 1 else f", idle_retry_attempt={attempt}"
            return (
                True,
                "gpu_idle_before_eggpu_child: "
                f"gpu={gpu_index}, memory_mb={used_mb:.1f}, raw_memory_mb={raw_used_mb:.1f}, "
                f"visibility_marker_adjust_mb={marker_adjust_mb:.1f}, utilization={gpu_util}%"
                f"{retry_note}",
            )
        # NVML utilization can briefly lag after the previous child process exits.
        # Only retry this benign residual-utilization case; real processes or
        # high memory remain hard blockers for comparable timing rows.
        if procs or used_mb > max_mem_mb or attempt >= retry_attempts:
            break
        if retry_sleep_s:
            time.sleep(retry_sleep_s)

    gpu_index, used_mb, raw_used_mb, marker_adjust_mb, gpu_util, procs, attempt = last_sample
    if procs or used_mb > max_mem_mb or gpu_util > max_util:
        proc_note = "; ".join(procs) if procs else "none"
        return (
            False,
            "gpu_busy_before_eggpu_child: "
            f"gpu={gpu_index}, memory_mb={used_mb:.1f}, raw_memory_mb={raw_used_mb:.1f}, "
            f"visibility_marker_adjust_mb={marker_adjust_mb:.1f}, utilization={gpu_util}%, "
            f"threshold_memory_mb={max_mem_mb:.1f}, threshold_utilization={max_util:.1f}%, "
            f"compute_processes=[{proc_note}], idle_retry_attempts={attempt}",
        )


def run_cmd(cmd, out_path, env, timeout=None, cooldown=0.0):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with out_path.open("w") as f:
        f.write("$ " + " ".join(str(x) for x in cmd) + "\n\n")
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
        peak_rss = 0
        pinfo = None
        if psutil is not None:
            try:
                pinfo = psutil.Process(p.pid)
            except Exception:
                pinfo = None
        gpu_handle = None
        if _ensure_nvml():
            try:
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(_resolve_monitor_gpu_index(env))
            except Exception:
                gpu_handle = None
        memory = {
            "rss_mb": None,
            "gpu_peak_mb": None,
            "gpu_avg_mb": None,
            "gpu_peak_delta_mb": None,
            "gpu_avg_delta_mb": None,
            "gpu_proc_peak_mb": None,
            "gpu_proc_avg_mb": None,
            "gpu_proc_peak_delta_mb": None,
            "gpu_proc_avg_delta_mb": None,
        }
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
            nonlocal peak_rss
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
            time.sleep(0.05)

        sample_memory()
        elapsed = time.perf_counter() - t0
        if cooldown and cooldown > 0:
            time.sleep(float(cooldown))
        peak_rss_mb = (peak_rss / (1024.0 * 1024.0)) if peak_rss > 0 else None
        memory["rss_mb"] = peak_rss_mb
        if gpu_samples > 0:
            memory["gpu_peak_mb"] = gpu_peak_bytes / (1024.0 * 1024.0)
            memory["gpu_avg_mb"] = (gpu_sum_bytes / gpu_samples) / (1024.0 * 1024.0)
            marker_adjust_mb = visibility_marker_adjust_mb(env)
            if marker_adjust_mb > 0:
                memory["gpu_peak_mb"] = max(0.0, memory["gpu_peak_mb"] - marker_adjust_mb)
                memory["gpu_avg_mb"] = max(0.0, memory["gpu_avg_mb"] - marker_adjust_mb)
        if gpu_delta_samples > 0:
            memory["gpu_peak_delta_mb"] = gpu_peak_delta_bytes / (1024.0 * 1024.0)
            memory["gpu_avg_delta_mb"] = (gpu_delta_sum_bytes / gpu_delta_samples) / (1024.0 * 1024.0)
        if gpu_proc_samples > 0:
            memory["gpu_proc_peak_mb"] = gpu_proc_peak_bytes / (1024.0 * 1024.0)
            memory["gpu_proc_avg_mb"] = (gpu_proc_sum_bytes / gpu_proc_samples) / (1024.0 * 1024.0)
        if gpu_proc_delta_samples > 0:
            memory["gpu_proc_peak_delta_mb"] = gpu_proc_peak_delta_bytes / (1024.0 * 1024.0)
            memory["gpu_proc_avg_delta_mb"] = (gpu_proc_delta_sum_bytes / gpu_proc_delta_samples) / (1024.0 * 1024.0)
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


def _git_stdout(args, cwd=WORKSPACE_ROOT):
    if shutil.which("git") is None:
        return "", "git executable not found", 127
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        return "", repr(exc), 1
    return proc.stdout.strip(), proc.stderr.strip(), proc.returncode


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
                }
            )
    return artifacts


def collect_run_metadata(args, out_dir, datasets, selected_functions, env):
    commit, commit_err, commit_rc = _git_stdout(["rev-parse", "HEAD"])
    branch, branch_err, branch_rc = _git_stdout(["rev-parse", "--abbrev-ref", "HEAD"])
    status_short, status_err, status_rc = _git_stdout(["status", "--short"])
    diff_stat, diff_stat_err, diff_stat_rc = _git_stdout(["diff", "--stat"])
    cuda_root = local_cuda_root()
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
        "EASYGRAPH_GPU_BACKEND",
        "EASYGRAPH_GPU_STRICT_ERRORS",
        "EGGPU_GPU_VISIBILITY_MARKER",
        "EGGPU_GPU_VISIBILITY_MARKER_MB",
        "EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB",
        "EASYGRAPH_GPU_ADAPTIVE_POLICY",
        "EASYGRAPH_GPU_COMPONENT_DENSE_RETURN",
        "EASYGRAPH_GPU_SCC_ACTIVE_TRIM",
        "EASYGRAPH_GPU_SCC_ACTIVE_TRIM_MAX_ITERS",
        "EASYGRAPH_GPU_SCC_DEGREE_PIVOT",
        "EASYGRAPH_GPU_SCC_HOST_ENABLE",
        "EASYGRAPH_GPU_KCORE_HOST_ENABLE",
        "EASYGRAPH_GPU_SSSP_HOST_ENABLE",
        "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_AVG_DEGREE",
        "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE",
        "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS",
        "EASYGRAPH_GPU_BC_WARP_SIZE",
        "EASYGRAPH_GPU_BC_UNWEIGHTED_BFS",
        "EASYGRAPH_GPU_CLOSENESS_UNWEIGHTED_BFS",
        "EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION",
        "EGGPU_CLOSENESS_EXACT_MAX_NODES",
        "EGGPU_CLOSENESS_EXACT_MAX_WORK",
        "EGGPU_USE_CONDA_RUN",
        "EGGPU_CHILD_PYTHON",
        "COMMON_PY",
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
        "git": {
            "commit": commit if commit_rc == 0 else "",
            "branch": branch if branch_rc == 0 else "",
            "dirty": bool(status_short),
            "status_short": status_short,
            "diff_stat": diff_stat,
            "errors": {
                "commit": commit_err if commit_rc else "",
                "branch": branch_err if branch_rc else "",
                "status": status_err if status_rc else "",
                "diff_stat": diff_stat_err if diff_stat_rc else "",
            },
        },
        "cuda": {
            "local_cuda_root": str(cuda_root) if cuda_root is not None else "",
            "nvcc": str(cuda_root / "bin" / "nvcc") if cuda_root is not None else "",
        },
        "gpu_device_profile": collect_gpu_device_profile(args.gpu, env),
        "build_artifacts": {
            "cpp_easygraph": cpp_easygraph_artifacts(),
        },
        "benchmark_args": {
            "gpu": args.gpu,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "easygraph_warmup": args.easygraph_warmup,
            "library_timeout": args.library_timeout,
            "inter_run_cooldown": args.inter_run_cooldown,
            "pr_alpha": args.pr_alpha,
            "pr_eps": args.pr_eps,
            "pr_max_iter": args.pr_max_iter,
            "sssp_sources": args.sssp_sources,
            "bc_sources": args.bc_sources,
            "datasets": [name for _, _, name, _ in datasets],
            "functions": list(selected_functions),
            "easygraph_gpu_backend": args.easygraph_gpu_backend,
        },
        "environment": {key: env.get(key, os.environ.get(key, "")) for key in env_keys},
    }


def write_run_metadata(out_dir, metadata):
    (Path(out_dir) / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
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
            if u > v:
                u, v = v, u
            canon.add((u, v))
    return {
        "nodes_raw": len(nodes),
        "edge_rows": rows,
        "edge_rows_no_selfloops": rows - loops,
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
    }
    structured.update(extra or {})
    rows.append({
        "dataset_size": dataset_size,
        "graph_type": graph_type,
        "dataset": dataset_name,
        "function": function,
        "baseline": baseline,
        "metric": metric,
        "seconds": "" if seconds is None else f"{seconds:.9g}",
        "status": status,
        "correctness": correctness,
        "log": str(log),
        "notes": notes,
        **structured,
    })


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
):
    if isinstance(peak_rss_mb, dict):
        memory = peak_rss_mb
    else:
        memory = {"rss_mb": peak_rss_mb}

    def add_mem(metric, value, metric_note):
        if value is None:
            return
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
        )

    add_mem("memory_peak_rss_mb", memory.get("rss_mb"), "process peak RSS memory (MB)")
    add_mem("memory_peak_gpu_mb", memory.get("gpu_peak_mb"), "device peak GPU memory during subprocess window (MB)")
    add_mem("memory_avg_gpu_mb", memory.get("gpu_avg_mb"), "device average GPU memory during subprocess window (MB)")
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
        "memory_peak_gpu_proc_delta_mb",
        memory.get("gpu_proc_peak_delta_mb"),
        "benchmark process-tree peak GPU memory delta from subprocess-start baseline (MB)",
    )
    add_mem(
        "memory_avg_gpu_proc_delta_mb",
        memory.get("gpu_proc_avg_delta_mb"),
        "benchmark process-tree average GPU memory delta from subprocess-start baseline (MB)",
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
        "gpu_peak_mb",
        "gpu_peak_delta_mb",
        "gpu_proc_peak_mb",
        "gpu_proc_peak_delta_mb",
    ):
        vals = [m.get(key) for m in metrics if m.get(key) is not None]
        out[key] = max(vals) if vals else None
    for key in (
        "gpu_avg_mb",
        "gpu_avg_delta_mb",
        "gpu_proc_avg_mb",
        "gpu_proc_avg_delta_mb",
    ):
        vals = [m.get(key) for m in metrics if m.get(key) is not None]
        out[key] = (sum(vals) / len(vals)) if vals else None
    return out


def path_for_cmd(path):
    path = Path(path)
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def gunrock_bin_candidates():
    env_override = os.environ.get("EG_GUNROCK_BIN") or os.environ.get("GUNROCK_BIN")
    candidates = []
    if env_override:
        candidates.append(Path(env_override).expanduser())
    candidates += [
        ROOT / "gunrock_latest" / "build_cuda132_a100_migrated" / "bin",
        ROOT / "gunrock_latest" / "build_cuda132_a100_migrated",
        ROOT / "gunrock_latest" / "build_cuda131_a100" / "bin",
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
    return candidates


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
        r"GPU Elapsed Time\s*:\s*([0-9.eE+-]+)\s*\(ms\)",
        r"avg\.\s+elapsed:\s*([0-9.eE+-]+)\s*ms",
        r"Run\s+\d+\s+elapsed:\s*([0-9.eE+-]+)\s*ms",
    )
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1)) / 1000.0
    return None


def parse_gunrock_mst_weight(text):
    return first_float(r"GPU MST Weight:\s*([0-9.eE+-]+)", text)


def parse_gunrock_error_count(text):
    m = re.search(r"Number of errors\s*:\s*([0-9]+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([1-9][0-9]*)\s+errors occurred\b", text, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def add_gunrock_build_not_applicable(rows, size, graph_type, name, function, log, notes):
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
            + "; build metric not comparable for Gunrock CLI binaries "
            "(graph loading happens inside external executable and is not isolated as baseline-side graph construction)"
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


def write_matrix_market(edge_path, out_path, directed, weighted=False):
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
    return out_path


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


def write_mst_largest_component_matrix_market(edge_path, out_path):
    """Write a weighted symmetric MatrixMarket file for the largest connected component.

    This is a fallback for Gunrock MST when the full graph triggers connected-graph
    constraints. We keep deterministic weights aligned with the main MST benchmark.
    """
    nodes = set()
    canon = set()
    adj = defaultdict(set)
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
            if u == v:
                continue
            nodes.add(u)
            nodes.add(v)
            if u > v:
                u, v = v, u
            canon.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
    if not canon:
        raise RuntimeError(f"empty graph for largest-component MST conversion: {edge_path}")

    # Largest connected component on undirected projection.
    seen = set()
    best = []
    for s in nodes:
        if s in seen:
            continue
        q = deque([s])
        seen.add(s)
        comp = []
        while q:
            x = q.popleft()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        if len(comp) > len(best):
            best = comp
    if not best:
        raise RuntimeError(f"failed to find connected component in: {edge_path}")

    keep = set(best)
    remap_nodes = sorted(keep)
    remap = {v: i for i, v in enumerate(remap_nodes)}
    comp_edges = []
    for u, v in sorted(canon):
        if u in keep and v in keep:
            comp_edges.append((remap[u], remap[v]))
    if not comp_edges:
        raise RuntimeError(f"largest component has no edges for: {edge_path}")

    n = len(remap_nodes)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        out.write("%%MatrixMarket matrix coordinate real symmetric\n")
        out.write(f"{n} {n} {len(comp_edges)}\n")
        for u, v in comp_edges:
            w = 1 + ((u * v) % n)
            out.write(f"{u + 1} {v + 1} {w}\n")
    return out_path, {"nodes": n, "edges": len(comp_edges)}


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
    runtime_status = {"ok", "timeout"}
    runtime = df[(df["status"].isin(runtime_status)) & (df["seconds"] != "")]
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
    grouped_plot(e2e, "Algorithm (Build Excluded)", "runtime_e2e.png")

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
        return "plot/table generation skipped because EGGPU_SKIP_PLOTS=TRUE"

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


def write_metric_csvs_no_pandas(out_dir, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    ok_rows = [
        r
        for r in rows
        if r.get("status") in ("ok", "timeout") and str(r.get("seconds", "")).strip() != ""
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


def maybe_run_gunrock_pr(rows, size, graph_type, name, path, ds_dir, env, repeat, cooldown):
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
        write_matrix_market(path, mtx, directed=False, weighted=False)
        cmd = [str(path_for_cmd(exe)), "-m", str(path_for_cmd(mtx)), "-n", str(repeat)]
        rc, elapsed, peak_rss_mb = run_cmd(cmd, log, gunrock_runtime_env(exe, env), timeout=PER_FUNCTION_TIMEOUT_SECONDS, cooldown=cooldown)
        txt = read_text(log)
        sec = parse_gunrock_elapsed_ms(txt)
        err_cnt = parse_gunrock_error_count(txt)
        status = "ok" if rc == 0 and sec is not None else "failed"
        note = "kernel from Gunrock output; e2e from process wall-time; MatrixMarket conversion excluded"
        if err_cnt:
            note += f"; internal validation reported {err_cnt} mismatches"
        corr = f"validation_errors={err_cnt}" if err_cnt else ""
        if rc == 124:
            status = "failed"
            sec = None
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "PageRank", log, note)
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "e2e", elapsed, status, log, corr, notes=note)
        add_memory_metric_rows(rows, size, graph_type, name, "PageRank", "Gunrock", log, peak_rss_mb, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock PR setup failed: {e}\n")
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, "PageRank", "Gunrock", "e2e", None, "failed", log, notes=str(e))


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
        write_matrix_market(path, mtx, directed=False, weighted=True)
        cmd = [str(path_for_cmd(exe)), "-m", str(path_for_cmd(mtx))]
        rc, elapsed, peak_rss_mb = run_cmd(cmd, log, gunrock_runtime_env(exe, env), timeout=PER_FUNCTION_TIMEOUT_SECONDS, cooldown=cooldown)
        txt = read_text(log)
        sec = parse_gunrock_elapsed_ms(txt)
        weight = parse_gunrock_mst_weight(txt)
        err_cnt = parse_gunrock_error_count(txt)
        status = "ok" if rc == 0 and sec is not None else "failed"
        note = "Gunrock reports kernel time only; MatrixMarket conversion time excluded"
        lower = txt.lower()
        needs_connected_retry = (
            "connected graph" in lower
            or "input graph must be connected" in lower
            or "super vertices not decremented" in lower
        )
        if needs_connected_retry:
            status = "skipped"
            note = (
                "Gunrock MST failed due graph precondition/implementation limits "
                "(connected-graph style semantics or super-vertex constraints); "
                "this benchmark uses spanning-forest semantics on disconnected inputs."
            )
        elif "input matrix must be symmetric" in lower:
            note += "; gunrock rejected generated matrix as non-symmetric"
        if needs_connected_retry:
            retry_log = ds_dir / "gunrock_mst_largest_component.log"
            try:
                retry_mtx = ds_dir / "gunrock_mst_largest_component.mtx"
                retry_mtx, comp_meta = write_mst_largest_component_matrix_market(path, retry_mtx)
                retry_cmd = [str(path_for_cmd(exe)), "-m", str(path_for_cmd(retry_mtx))]
                rc2, elapsed2, peak_rss_mb2 = run_cmd(
                    retry_cmd,
                    retry_log,
                    gunrock_runtime_env(exe, env),
                    timeout=PER_FUNCTION_TIMEOUT_SECONDS,
                    cooldown=cooldown,
                )
                txt2 = read_text(retry_log)
                sec2 = parse_gunrock_elapsed_ms(txt2)
                weight2 = parse_gunrock_mst_weight(txt2)
                err_cnt2 = parse_gunrock_error_count(txt2)
                if rc2 == 0 and sec2 is not None:
                    status = "ok"
                    sec = sec2
                    weight = weight2
                    err_cnt = err_cnt2
                    elapsed = elapsed2
                    log = retry_log
                    peak_rss_mb = peak_rss_mb2
                    note = (
                        "Gunrock MST retried on largest connected component due full-graph disconnected semantics; "
                        f"component_nodes={comp_meta['nodes']}, component_edges={comp_meta['edges']}. "
                        "Time is kernel-only; compare carefully against full-graph spanning-forest baselines."
                    )
                else:
                    note += "; fallback retry on largest connected component failed"
                    if rc2 == 124:
                        note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
            except Exception as retry_exc:
                note += f"; fallback retry on largest connected component failed: {retry_exc}"
        if rc == 124 and status != "ok":
            status = "failed"
            sec = None
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        if err_cnt:
            note += f"; internal validation reported {err_cnt} mismatches"
        corr = "" if weight is None else f"weight={weight:.0f}"
        if err_cnt:
            corr = (corr + "; " if corr else "") + f"validation_errors={err_cnt}"
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "MST", log, note)
        add_row(rows, size, graph_type, name, "MST", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(rows, size, graph_type, name, "MST", "Gunrock", "e2e", elapsed, status, log, corr, notes=note)
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
        err_cnt = parse_gunrock_error_count(txt)
        status = "ok" if rc == 0 and sec is not None else "failed"
        note = (
            "Gunrock reports kernel time only; MatrixMarket conversion time excluded; "
            "run with --undirected=true --sort-csr=true --validation=none --quick=true"
        )
        if "libgunrock_utils.so" in txt:
            note += "; failed to load libgunrock_utils.so (runtime LD_LIBRARY_PATH issue)"
        if err_cnt:
            note += f"; internal validation reported {err_cnt} mismatches"
        corr = f"validation_errors={err_cnt}" if err_cnt else ""
        if rc == 124:
            status = "failed"
            sec = None
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "LCC", log, note)
        add_row(rows, size, graph_type, name, "LCC", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(rows, size, graph_type, name, "LCC", "Gunrock", "e2e", elapsed, status, log, corr, notes=note)
        add_memory_metric_rows(rows, size, graph_type, name, "LCC", "Gunrock", log, peak_rss_mb, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock LCC setup failed: {e}\n")
        add_row(rows, size, graph_type, name, "LCC", "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, "LCC", "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, "LCC", "Gunrock", "e2e", None, "failed", log, notes=str(e))


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
        status = "ok" if rc == 0 and sec is not None else "failed"
        semantic = "WCC semantics; undirected projection"
        if function == "SCC":
            semantic = "undirected graph: SCC equals WCC"
        note = (
            f"{semantic}; kernel from Gunrock output; e2e from process wall-time; MatrixMarket conversion excluded; "
            "run with --undirected=true --sort-csr=true --validation=none --quick=true"
        )
        if "libgunrock_utils.so" in txt:
            note += "; failed to load libgunrock_utils.so (runtime LD_LIBRARY_PATH issue)"
        if err_cnt:
            note += f"; internal validation reported {err_cnt} mismatches"
        corr = f"validation_errors={err_cnt}" if err_cnt else ""
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


def maybe_run_gunrock_sssp(rows, size, graph_type, name, path, ds_dir, env, cooldown, sssp_sources):
    exe = find_gunrock_exe("sssp")
    log = ds_dir / "gunrock_sssp.log"
    if exe is None:
        add_unavailable_gunrock(rows, size, graph_type, name, "SSSP", ds_dir)
        return
    try:
        mtx = ds_dir / "gunrock_sssp.mtx"
        write_sssp_weighted_matrix_market(path, mtx, directed=(graph_type == "directed"))
        n = matrix_market_n(mtx)
        sources = deterministic_sources(n, sssp_sources)
        if not sources:
            sources = [0]
        total_sec = 0.0
        total_elapsed = 0.0
        total_err_cnt = 0
        any_failed = False
        memories = []
        logs = []
        for source in sources:
            src_log = ds_dir / f"gunrock_sssp_source_{int(source)}.log"
            cmd = [str(path_for_cmd(exe)), "-m", str(path_for_cmd(mtx)), "-s", str(int(source)), "--validate"]
            rc, elapsed, mem = run_cmd(
                cmd,
                src_log,
                gunrock_runtime_env(exe, env),
                timeout=PER_FUNCTION_TIMEOUT_SECONDS,
                cooldown=cooldown,
            )
            txt = read_text(src_log)
            sec_i = parse_gunrock_elapsed_ms(txt)
            err_i = parse_gunrock_error_count(txt)
            logs.append(src_log)
            memories.append(mem)
            total_elapsed += elapsed
            total_err_cnt += int(err_i or 0)
            if rc == 0 and sec_i is not None and err_i == 0:
                total_sec += float(sec_i)
            else:
                any_failed = True
                if rc == 124:
                    src_log.write_text(read_text(src_log) + "\n" + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS) + "\n")
        log.write_text(
            "Gunrock SSSP was run once per source because the CLI accepts a single source.\n"
            + "\n".join(str(path_for_cmd(p)) for p in logs)
            + "\n"
        )
        sec = None if any_failed else total_sec
        elapsed = total_elapsed
        memory = combine_memory_metrics(memories)
        err_cnt = total_err_cnt
        status = "ok" if not any_failed and sec is not None and err_cnt == 0 else "failed"
        note = (
            "weighted deterministic edges; Gunrock CLI run once per source; "
            "kernel/e2e are summed over sources; MatrixMarket conversion excluded"
        )
        corr = f"sources={len(sources)}, validation_errors={err_cnt}"
        if err_cnt:
            note += f"; internal validation reported {err_cnt} mismatches"
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "SSSP", log, note)
        add_row(rows, size, graph_type, name, "SSSP", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(rows, size, graph_type, name, "SSSP", "Gunrock", "e2e", elapsed, status, log, corr, notes=note)
        add_memory_metric_rows(rows, size, graph_type, name, "SSSP", "Gunrock", log, memory, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock SSSP setup failed: {e}\n")
        add_row(rows, size, graph_type, name, "SSSP", "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, "SSSP", "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, "SSSP", "Gunrock", "e2e", None, "failed", log, notes=str(e))


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
        total_sec = 0.0
        total_elapsed = 0.0
        total_err_cnt = 0
        any_failed = False
        memories = []
        logs = []
        for source in sources:
            src_log = ds_dir / f"gunrock_bfs_source_{int(source)}.log"
            cmd = [str(path_for_cmd(exe)), "-m", str(path_for_cmd(mtx)), "-s", str(int(source)), "--validate"]
            rc, elapsed, mem = run_cmd(
                cmd,
                src_log,
                gunrock_runtime_env(exe, env),
                timeout=PER_FUNCTION_TIMEOUT_SECONDS,
                cooldown=cooldown,
            )
            txt = read_text(src_log)
            sec_i = parse_gunrock_elapsed_ms(txt)
            err_i = parse_gunrock_error_count(txt)
            logs.append(src_log)
            memories.append(mem)
            total_elapsed += elapsed
            total_err_cnt += int(err_i or 0)
            if rc == 0 and sec_i is not None and err_i == 0:
                total_sec += float(sec_i)
            else:
                any_failed = True
                if rc == 124:
                    src_log.write_text(read_text(src_log) + "\n" + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS) + "\n")
        log.write_text(
            "Gunrock BFS was run once per source because the CLI accepts a single source.\n"
            + "\n".join(str(path_for_cmd(p)) for p in logs)
            + "\n"
        )
        sec = None if any_failed else total_sec
        elapsed = total_elapsed
        memory = combine_memory_metrics(memories)
        err_cnt = total_err_cnt
        status = "ok" if not any_failed and sec is not None and err_cnt == 0 else "failed"
        note = (
            "unweighted shortest paths; Gunrock CLI run once per source; "
            "kernel/e2e are summed over sources; MatrixMarket conversion excluded"
        )
        corr = f"sources={len(sources)}, validation_errors={err_cnt}"
        if err_cnt:
            note += f"; internal validation reported {err_cnt} mismatches"
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "BFS", log, note)
        add_row(rows, size, graph_type, name, "BFS", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(rows, size, graph_type, name, "BFS", "Gunrock", "e2e", elapsed, status, log, corr, notes=note)
        add_memory_metric_rows(rows, size, graph_type, name, "BFS", "Gunrock", log, memory, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock BFS setup failed: {e}\n")
        add_row(rows, size, graph_type, name, "BFS", "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, "BFS", "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, "BFS", "Gunrock", "e2e", None, "failed", log, notes=str(e))


def maybe_run_gunrock_kcore(rows, size, graph_type, name, path, ds_dir, env, cooldown):
    exe = find_gunrock_exe("kcore")
    log = ds_dir / "gunrock_kcore.log"
    if exe is None:
        add_unavailable_gunrock(rows, size, graph_type, name, "KCore", ds_dir)
        return
    try:
        mtx = ds_dir / "gunrock_kcore.mtx"
        write_matrix_market(path, mtx, directed=(graph_type == "directed"), weighted=False)
        cmd = [str(path_for_cmd(exe)), str(path_for_cmd(mtx))]
        rc, elapsed, peak_rss_mb = run_cmd(cmd, log, gunrock_runtime_env(exe, env), timeout=PER_FUNCTION_TIMEOUT_SECONDS, cooldown=cooldown)
        txt = read_text(log)
        sec = parse_gunrock_elapsed_ms(txt)
        err_cnt = parse_gunrock_error_count(txt)
        status = "ok" if rc == 0 and sec is not None and err_cnt == 0 else "failed"
        note = (
            "undirected projection; kernel time from Gunrock output; e2e from process wall-time "
            "(MatrixMarket conversion excluded)"
        )
        corr = ""
        if err_cnt:
            corr = f"validation_errors={err_cnt}"
            note += f"; internal validation reported {err_cnt} mismatches"
        if rc == 124:
            status = "failed"
            sec = None
            note += "; " + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS)
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "KCore", log, note)
        add_row(rows, size, graph_type, name, "KCore", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(rows, size, graph_type, name, "KCore", "Gunrock", "e2e", elapsed, status, log, corr, notes=note)
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
        total_sec = 0.0
        total_elapsed = 0.0
        any_failed = False
        memories = []
        logs = []
        for source in sources:
            src_log = ds_dir / f"gunrock_bc_source_{int(source)}.log"
            cmd = [str(path_for_cmd(exe)), "-m", str(path_for_cmd(mtx)), "-s", str(int(source))]
            rc, elapsed, mem = run_cmd(
                cmd,
                src_log,
                gunrock_runtime_env(exe, env),
                timeout=PER_FUNCTION_TIMEOUT_SECONDS,
                cooldown=cooldown,
            )
            txt = read_text(src_log)
            sec_i = parse_gunrock_elapsed_ms(txt)
            logs.append(src_log)
            memories.append(mem)
            total_elapsed += elapsed
            if rc == 0 and sec_i is not None:
                total_sec += float(sec_i)
            else:
                any_failed = True
                if rc == 124:
                    src_log.write_text(read_text(src_log) + "\n" + timeout_too_long_note(PER_FUNCTION_TIMEOUT_SECONDS) + "\n")
        log.write_text(
            "Gunrock BC was run once per source because the CLI accepts a single source.\n"
            + "\n".join(str(path_for_cmd(p)) for p in logs)
            + "\n"
        )
        sec = None if any_failed else total_sec
        elapsed = total_elapsed
        memory = combine_memory_metrics(memories)
        status = "ok" if not any_failed and sec is not None else "failed"
        note = (
            f"source-sampled mode; sources={len(sources)}; Gunrock CLI run once per source; "
            "kernel/e2e are summed over sources; MatrixMarket conversion excluded"
        )
        corr = f"sources={len(sources)}"
        add_gunrock_build_not_applicable(rows, size, graph_type, name, "BC", log, note)
        add_row(rows, size, graph_type, name, "BC", "Gunrock", "kernel", sec, status, log, corr, notes=note)
        add_row(rows, size, graph_type, name, "BC", "Gunrock", "e2e", elapsed, status, log, corr, notes=note)
        add_memory_metric_rows(rows, size, graph_type, name, "BC", "Gunrock", log, memory, status=status, notes=note)
    except Exception as e:
        log.write_text(f"Gunrock BC setup failed: {e}\n")
        add_row(rows, size, graph_type, name, "BC", "Gunrock", "build", None, "skipped", log, notes=str(e))
        add_row(rows, size, graph_type, name, "BC", "Gunrock", "kernel", None, "failed", log, notes=str(e))
        add_row(rows, size, graph_type, name, "BC", "Gunrock", "e2e", None, "failed", log, notes=str(e))


def add_unavailable_gunrock(rows, size, graph_type, name, function, ds_dir):
    log = ds_dir / f"gunrock_{function.lower()}_unavailable.log"
    log.write_text(f"Gunrock {function} baseline unavailable: no matching executable. {gunrock_search_note()}.\n")
    status = "skipped"
    note = "no matching Gunrock executable in current build"
    if function in {"WCC", "SCC", "CC"}:
        note += "; legacy Gunrock executable name is cc"
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
    pr_alpha,
    pr_tol,
    pr_max_iter,
    easygraph_repo,
    easygraph_gpu_backend,
    warmup,
    easygraph_warmup,
    sssp_sources,
    bc_sources,
    inter_run_cooldown,
    selected_functions,
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

    for base in ("igraph", "networkx", "EGGPU", "easygraph-cpu", "easygraph-cpp", "nx-cugraph"):
        for func in selected_functions:
            if progress is not None:
                progress(base, func)
            log = ds_dir / f"library_{base}_{func.lower()}.log"
            if func == "Closeness" and exact_closeness_too_large(dataset_stat):
                add_exact_closeness_scale_skip(rows, size, graph_type, name, base, ds_dir, dataset_stat)
                continue
            env_one = dict(env_lib)
            env_one["EGGPU_VALIDATION_DETAIL_DIR"] = str((ds_dir / "details").resolve())
            if base == "EGGPU":
                env_one["EASYGRAPH_ENABLE_GPU"] = "TRUE"
                env_one["EASYGRAPH_GPU_BACKEND"] = "mine"
                env_one["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
                env_one["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
                env_one["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"
                env_one["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
                env_one["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
                env_one["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
            else:
                env_one["EASYGRAPH_ENABLE_GPU"] = "FALSE"
                env_one.pop("EASYGRAPH_GPU_BACKEND", None)
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
                "--easygraph-gpu-backend",
                str(easygraph_gpu_backend),
                "--easygraph-warmup",
                str(base_easygraph_warmup),
                "--sssp-sources",
                str(sssp_sources),
                "--bc-sources",
                str(bc_sources),
                "--cooldown",
                str(inter_run_cooldown),
            )
            if skip_cpu and base in ("igraph", "networkx"):
                cmd.append("--skip-cpu")
            if base == "EGGPU":
                idle_ok, idle_note = check_eggpu_child_gpu_idle(env_one)
                if not idle_ok:
                    log.write_text(idle_note + "\n")
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
                        add_row(rows, size, graph_type, name, func, base, metric, None, "failed", log, notes=idle_note)
                    continue
            rc, _, _ = run_cmd(cmd, log, env_one, timeout=timeout, cooldown=inter_run_cooldown)
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
                continue
            for r in results:
                status = r.get("status", "failed")
                notes = r.get("notes", "")
                seconds = r.get("seconds")
                if rc == 124:
                    status = "failed"
                    seconds = None
                    notes = (notes + "; " if notes else "") + timeout_too_long_note(timeout)
                elif rc != 0 and status == "ok":
                    status = "failed"
                add_row(
                    rows,
                    size,
                    graph_type,
                    name,
                    r.get("function", ""),
                    r.get("backend", ""),
                    r.get("metric", "e2e"),
                    seconds,
                    status,
                    log,
                    r.get("correctness", ""),
                    notes,
                )


def main():
    global PER_FUNCTION_TIMEOUT_SECONDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="7")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="Untimed warmup calls for the EGGPU baseline only.")
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
    ap.add_argument("--pr-alpha", type=float, default=0.75, help="PageRank damping factor for Mine + library baselines.")
    ap.add_argument("--pr-eps", type=float, default=1e-6, help="PageRank convergence tolerance for EGGPU + compatible baselines.")
    ap.add_argument("--pr-max-iter", type=int, default=200, help="PageRank max iterations for EGGPU + compatible baselines.")
    ap.add_argument(
        "--easygraph-repo",
        default=str(WORKSPACE_ROOT / "Easy-Graph"),
        help="Path to local Easy-Graph repo to prepend into PYTHONPATH for library baselines.",
    )
    ap.add_argument(
        "--easygraph-gpu-backend",
        default="mine",
        help="GPU backend for EGGPU baseline. Defaults to mine and is force-set to mine in runner for consistency.",
    )
    ap.add_argument("--easygraph-warmup", type=int, default=1, help="Additional untimed warmup calls for the EGGPU baseline only.")
    ap.add_argument("--sssp-sources", type=int, default=8, help="Number of deterministic sources for SSSP benchmarks.")
    ap.add_argument("--bc-sources", type=int, default=16, help="Number of deterministic sources for BC source-sampled benchmarks.")
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
    args = ap.parse_args()
    PER_FUNCTION_TIMEOUT_SECONDS = int(args.library_timeout)

    # Optional idle CUDA context so nvitop shows the long-lived benchmark driver
    # even while CPU-only baselines are running.  It does not launch kernels.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("EGGPU_MONITOR_GPU_INDEX", str(args.gpu))
    visibility_marker = GpuVisibilityMarker(args.gpu, "run_full_baselines.py").start()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else ROOT / "benchmarking" / "results" / f"{ts}_full_baseline_gpu{args.gpu}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    logs.mkdir(exist_ok=True)

    env = sanitized_subprocess_env(os.environ)
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

    rows = []
    notes = []
    stats = []
    total_main_tasks = len(datasets) * len(selected_functions) * 7
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

        def progress_step(backend, function):
            progress.tick(f"dataset {dataset_index}/{len(datasets)} {name}: {backend}/{function}")

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
            progress_step("Gunrock", "PageRank")
            maybe_run_gunrock_pr(rows, size, graph_type, name, path, ds_dir, env, args.repeat, args.inter_run_cooldown)
        if "MST" in selected_functions:
            progress_step("Gunrock", "MST")
            maybe_run_gunrock_mst(rows, size, graph_type, name, path, ds_dir, env, args.inter_run_cooldown)
        if "LCC" in selected_functions:
            progress_step("Gunrock", "LCC")
            maybe_run_gunrock_lcc(rows, size, graph_type, name, path, ds_dir, env, args.inter_run_cooldown)
        if "WCC" in selected_functions:
            progress_step("Gunrock", "WCC")
            maybe_run_gunrock_cc(rows, size, graph_type, name, path, ds_dir, env, args.inter_run_cooldown, function="WCC")
        if "SCC" in selected_functions:
            progress_step("Gunrock", "SCC")
            maybe_run_gunrock_cc(rows, size, graph_type, name, path, ds_dir, env, args.inter_run_cooldown, function="SCC")
        if "BFS" in selected_functions:
            progress_step("Gunrock", "BFS")
            maybe_run_gunrock_bfs(rows, size, graph_type, name, path, ds_dir, env, args.inter_run_cooldown, args.sssp_sources)
        for path_func in ("Dijkstra", "BellmanFord"):
            if path_func in selected_functions:
                progress_step("Gunrock", path_func)
                add_unavailable_gunrock(rows, size, graph_type, name, path_func, ds_dir)
        for structural_func in ("EffectiveSize", "Efficiency", "Constraint", "Hierarchy"):
            if structural_func in selected_functions:
                progress_step("Gunrock", structural_func)
                add_unavailable_gunrock(rows, size, graph_type, name, structural_func, ds_dir)
        if "SSSP" in selected_functions:
            progress_step("Gunrock", "SSSP")
            maybe_run_gunrock_sssp(
                rows, size, graph_type, name, path, ds_dir, env, args.inter_run_cooldown, args.sssp_sources
            )
        if "KCore" in selected_functions:
            progress_step("Gunrock", "KCore")
            maybe_run_gunrock_kcore(rows, size, graph_type, name, path, ds_dir, env, args.inter_run_cooldown)
        if "BC" in selected_functions:
            progress_step("Gunrock", "BC")
            maybe_run_gunrock_bc(
                rows, size, graph_type, name, path, ds_dir, env, args.inter_run_cooldown, args.bc_sources
            )
        if "Closeness" in selected_functions:
            progress_step("Gunrock", "Closeness")
            add_unavailable_gunrock(rows, size, graph_type, name, "Closeness", ds_dir)

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
            args.pr_alpha,
            args.pr_eps,
            args.pr_max_iter,
            args.easygraph_repo,
            args.easygraph_gpu_backend,
            args.warmup,
            args.easygraph_warmup,
            args.sssp_sources,
            args.bc_sources,
            args.inter_run_cooldown,
            selected_functions,
            st,
            progress=progress_step,
        )

    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "notes.txt").write_text("\n".join(notes) + ("\n" if notes else ""))
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
        "- `results_long.csv`, `results_build.csv`, `results_kernel.csv`, `results_e2e.csv`, `results_memory.csv`",
        "- `correctness_validation.csv`, `correctness_validation.md`",
        "- `runtime_build.png/.pdf`, `runtime_kernel.png/.pdf`, `runtime_e2e.png/.pdf`, `runtime_memory.png/.pdf`, `speedup_heatmap.png/.pdf`",
        "- `main_table.tex`",
        "- per-run logs under `logs/`",
        "",
        "Notes:",
        "- All benchmarks were run serially with `CUDA_VISIBLE_DEVICES` fixed to the requested GPU.",
        f"- Added inter-run cooldown: {args.inter_run_cooldown}s.",
        f"- Warmup policy: EGGPU only ({args.warmup} untimed call(s), merged with EGGPU-specific warmup={args.easygraph_warmup}); all other baselines run without extra warmup.",
        "- Child benchmark processes explicitly unset CFLAGS/CPPFLAGS/CXXFLAGS/CPATH/LIBRARY_PATH and pin CUDA_PATH/CUDA_HOME/CUPY_CUDA_PATH/CONDA_PREFIX to the selected local CUDA toolkit, avoiding host-toolchain and conda-header contamination of CUDA JIT/compilation paths. The default runner uses the selected EGGPU Python directly; `conda run` is only used when `EGGPU_USE_CONDA_RUN=TRUE` is explicitly set.",
        f"- PageRank hyperparameters were aligned where supported: alpha={args.pr_alpha}, tol/eps={args.pr_eps}, max_iter={args.pr_max_iter}.",
        f"- SSSP uses deterministic weighted edges with {args.sssp_sources} sampled sources; BC uses source-sampled mode with {args.bc_sources} sampled sources (normalized=False).",
        f"- Selected datasets: {', '.join(d[2] for d in datasets)}.",
        f"- Selected functions: {', '.join(selected_functions)}.",
        "- EGGPU baseline is the integrated EasyGraph GPU path and is force-pinned to `EASYGRAPH_GPU_BACKEND=mine` in runner.",
        "- Timing schema uses three metrics only: `build` (baseline-native graph construction, import/file parse excluded), `kernel` (device kernel time where exposed; otherwise algorithm wall-time surrogate), and `e2e` (user-visible function call wall-time, including per-call prep/transfer/sync).",
        "- Function-specific preparation like CSR conversion remains in `e2e` because it is paid at function-call time, not at baseline graph-object construction time.",
        "- CPU backends do not expose a kernel concept, so CPU `kernel` equals algorithm wall-time by definition.",
        "- Gunrock `build` is marked non-comparable/skipped: graph loading happens inside external CLI executables and cannot be isolated as in-process baseline graph construction.",
        f"- Per-function timeout is {PER_FUNCTION_TIMEOUT_SECONDS}s for all baselines; timed-out runs are marked with `TIMEOUT_TOO_LONG` in logs/notes.",
        "- Directed PageRank preserves edge direction. WCC ignores edge direction, SCC preserves directed reachability; on undirected graphs SCC=WCC. MST and LCC use an undirected projection on directed datasets.",
        "- Raw self-loop rows are preserved in source files for provenance but removed during benchmark graph construction and Gunrock MatrixMarket conversion for all libraries/functions.",
        "- Exact all-source Closeness uses a symmetric predeclared scale guard: "
        f"EGGPU_CLOSENESS_EXACT_MAX_NODES={env.get('EGGPU_CLOSENESS_EXACT_MAX_NODES', os.environ.get('EGGPU_CLOSENESS_EXACT_MAX_NODES', '1000000'))}, "
        f"EGGPU_CLOSENESS_EXACT_MAX_WORK={env.get('EGGPU_CLOSENESS_EXACT_MAX_WORK', os.environ.get('EGGPU_CLOSENESS_EXACT_MAX_WORK', '50000000000'))}. "
        "Guarded rows remain `semantic=exact_all_node` with `skip_reason=exact_scale_guard`; large-graph evidence is reported separately as `sampled_target_exact`.",
        "- Native cuGraph timings are not emitted as standalone baseline rows; nx-cugraph may fallback to native cuGraph for unsupported functions and logs that fallback.",
        "- Gunrock baselines are discovered per executable (`pr`, `mst`, `lcc`, `cc`, `sssp`, `kcore`, `bc`) across configured build directories; each algorithm may come from a different build tree. Closeness has no aligned Gunrock executable in this benchmark and is marked unavailable.",
        "- Gunrock MST retries on the largest connected component when full-graph input triggers connected-graph/super-vertex constraints; such rows are explicitly annotated as component-level timings.",
        "- Memory metrics include `memory_peak_rss_mb`; when NVML is available, both device-level (`memory_peak_gpu_mb`/`memory_avg_gpu_mb` + delta) and process-tree GPU memory (`memory_peak_gpu_proc_mb`/`memory_avg_gpu_proc_mb` + delta) are collected.",
        f"- GPU visibility marker: enabled={env.get('EGGPU_GPU_VISIBILITY_MARKER', '') or 'FALSE'}, marker_mb={env.get('EGGPU_GPU_VISIBILITY_MARKER_MB', '0')}, whole_device_adjust_mb={env.get('EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB', env.get('EGGPU_GPU_VISIBILITY_MARKER_MB', '0'))}.  The fixed adjustment applies only to whole-device absolute memory metrics; process-tree and delta memory metrics are unadjusted.",
        f"- `igraph`, `networkx`, `EGGPU`, `easygraph-cpu`, `easygraph-cpp`, and `nx-cugraph` baselines are run through `benchmarking/library_baselines.py` with a per backend/function timeout of {args.library_timeout} seconds.",
        "",
        "Reproducibility:",
        f"- Git commit: `{run_metadata['git']['commit'] or 'unknown'}`.",
        f"- Git dirty: `{str(run_metadata['git']['dirty']).lower()}`.",
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
            "dataset_stats.json",
            "notes.txt",
            "results_long.csv",
            "results_build.csv",
            "results_kernel.csv",
            "results_e2e.csv",
            "results_memory.csv",
            "correctness_validation.csv",
            "correctness_validation.md",
            "summary.md",
        ],
    }
    run_metadata["build_artifacts"] = {
        "cpp_easygraph": cpp_easygraph_artifacts(),
    }
    write_run_metadata(out_dir, run_metadata)
    print(f"\nDone: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
