#!/usr/bin/env python3
"""Parent-side RSS and GPU-memory sampling for isolated benchmark workers."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

try:
    import psutil
except Exception:  # pragma: no cover - optional outside benchmark environments
    psutil = None

try:
    import pynvml
except Exception:  # pragma: no cover - optional outside GPU environments
    pynvml = None


def _parse_proc_status(path: Path):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    fields = {}
    for line in lines:
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value.strip()
    return fields


def resolve_host_pid(pid: int) -> int:
    """Map a PID-namespace-local PID to the host PID exposed by /proc/NVML."""

    pid = int(pid)
    if pid == os.getpid():
        fields = _parse_proc_status(Path("/proc/self/status"))
        try:
            return int(fields.get("Pid", pid))
        except ValueError:
            return pid

    direct = Path("/proc") / str(pid)
    if direct.exists():
        return pid

    try:
        namespace_inode = os.stat("/proc/self/ns/pid").st_ino
    except OSError:
        namespace_inode = None

    try:
        entries = os.scandir("/proc")
    except OSError:
        return pid
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            candidate = Path(entry.path)
            fields = _parse_proc_status(candidate / "status")
            namespace_pids = fields.get("NSpid", "").split()
            if not namespace_pids or namespace_pids[-1] != str(pid):
                continue
            if namespace_inode is not None:
                try:
                    if os.stat(candidate / "ns/pid").st_ino != namespace_inode:
                        continue
                except OSError:
                    continue
            try:
                return int(fields.get("Pid", entry.name))
            except ValueError:
                continue
    return pid


def _proc_descendants(root_pid: int):
    pending = [int(root_pid)]
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        children_path = Path("/proc") / str(current) / "task" / str(current) / "children"
        try:
            children = [int(value) for value in children_path.read_text().split()]
        except (OSError, ValueError):
            children = []
        pending.extend(children)
    return seen


def _proc_rss_bytes(pid: int) -> int:
    fields = _parse_proc_status(Path("/proc") / str(pid) / "status")
    raw = fields.get("VmRSS", "0 kB").split()
    try:
        return int(raw[0]) * 1024
    except (IndexError, ValueError):
        return 0


def process_tree_pids_and_rss(pid: int):
    """Return host PIDs and aggregate RSS for one namespace-aware process tree."""

    host_pid = resolve_host_pid(pid)
    if psutil is not None:
        try:
            root = psutil.Process(host_pid)
            processes = [root]
            try:
                processes.extend(root.children(recursive=True))
            except Exception:
                pass
            pids = set()
            rss = 0
            for process in processes:
                try:
                    pids.add(int(process.pid))
                    rss += int(process.memory_info().rss)
                except Exception:
                    pass
            if pids:
                return pids, rss
        except Exception:
            pass

    pids = _proc_descendants(host_pid)
    return pids or {host_pid}, sum(_proc_rss_bytes(item) for item in pids)


class ChildProcessMemoryMonitor:
    """Sample one subprocess tree without depending on the worker's GIL.

    The monitor is intentionally owned by the benchmark coordinator. Native
    CUDA/C++ calls may retain the worker's Python GIL, but cannot starve this
    thread because it runs in a different process.
    """

    def __init__(self, pid: int, physical_gpu: int, interval_seconds: float = 0.002):
        self.pid = int(pid)
        self.host_pid = resolve_host_pid(self.pid)
        self.physical_gpu = int(physical_gpu)
        self.gpu_monitor_enabled = self.physical_gpu >= 0
        self.interval_seconds = max(0.002, float(interval_seconds))
        self.sample_count = 0
        self.rss_sample_count = 0
        self.gpu_sample_count = 0
        self.rss_start_bytes = None
        self.rss_peak_bytes = 0
        self.gpu_start_bytes = None
        self.gpu_peak_bytes = 0
        self._started_at = None
        self._stop = threading.Event()
        # NVML can block for tens of milliseconds on a busy multi-GPU host.
        # Keep it off the RSS sampler so GPU accounting cannot reduce host
        # memory sampling density.
        self._rss_thread = threading.Thread(target=self._run_rss, daemon=True)
        self._gpu_thread = (
            threading.Thread(target=self._run_gpu, daemon=True)
            if self.gpu_monitor_enabled
            else None
        )

    def _tree_pids_and_rss(self):
        return process_tree_pids_and_rss(self.host_pid)

    @staticmethod
    def _used_bytes(process_info):
        value = getattr(process_info, "usedGpuMemory", 0)
        try:
            value = int(value)
        except Exception:
            return 0
        # NVML uses an all-bits-set sentinel when memory accounting is absent.
        return 0 if value < 0 or value >= (1 << 63) else value

    def _sample_rss(self):
        _pids, rss = self._tree_pids_and_rss()
        if self.rss_start_bytes is None:
            self.rss_start_bytes = rss
        self.rss_peak_bytes = max(self.rss_peak_bytes, rss)
        self.rss_sample_count += 1
        # Retain the original field as an alias for host samples.
        self.sample_count = self.rss_sample_count

    def _sample_gpu(self, handle):
        pids, _rss = self._tree_pids_and_rss()
        gpu_bytes = 0
        if handle is not None:
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                gpu_bytes = sum(
                    self._used_bytes(item)
                    for item in processes
                    if int(getattr(item, "pid", -1)) in pids
                )
            except Exception:
                gpu_bytes = 0
        if self.gpu_start_bytes is None:
            self.gpu_start_bytes = gpu_bytes
        self.gpu_peak_bytes = max(self.gpu_peak_bytes, gpu_bytes)
        self.gpu_sample_count += 1

    def _run_rss(self):
        while not self._stop.is_set():
            self._sample_rss()
            self._stop.wait(self.interval_seconds)

    def _run_gpu(self):
        handle = None
        nvml_ready = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.physical_gpu)
                nvml_ready = True
            except Exception:
                handle = None
        try:
            while not self._stop.is_set():
                self._sample_gpu(handle)
                self._stop.wait(self.interval_seconds)
        finally:
            if nvml_ready:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    def start(self):
        self._started_at = time.perf_counter()
        self._rss_thread.start()
        if self._gpu_thread is not None:
            self._gpu_thread.start()
        return self

    def stop(self):
        self._stop.set()
        timeout = max(2.0, 4 * self.interval_seconds)
        self._rss_thread.join(timeout=timeout)
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=timeout)
        elapsed = (
            time.perf_counter() - self._started_at
            if self._started_at is not None
            else 0.0
        )
        mib = 1024.0 * 1024.0
        rss_start = float(self.rss_start_bytes or 0) / mib
        gpu_start = (
            float(self.gpu_start_bytes or 0) / mib
            if self.gpu_monitor_enabled
            else None
        )
        rss_peak = float(self.rss_peak_bytes) / mib
        gpu_peak = (
            float(self.gpu_peak_bytes) / mib
            if self.gpu_monitor_enabled
            else None
        )
        return {
            "rss_mb": rss_peak,
            "rss_start_mb": rss_start,
            "rss_peak_delta_mb": max(0.0, rss_peak - rss_start),
            "gpu_proc_peak_mb": gpu_peak,
            "gpu_proc_start_mb": gpu_start,
            "gpu_proc_peak_delta_mb": (
                max(0.0, gpu_peak - gpu_start)
                if gpu_peak is not None and gpu_start is not None
                else None
            ),
            "monitor_rss_samples": int(self.rss_sample_count),
            "monitor_gpu_proc_samples": int(self.gpu_sample_count),
            "monitor_poll_ms": self.interval_seconds * 1000.0,
            "monitor_window_seconds": elapsed,
            "gpu_index": self.physical_gpu if self.gpu_monitor_enabled else None,
            "memory_monitor_origin": "coordinator_process_child_tree",
        }
