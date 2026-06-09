#!/usr/bin/env python3
"""Keep a benchmark process visible in nvitop without launching work kernels.

When EGGPU benchmarks spend a long time inside CPU-only baselines, nvitop can
look idle and other users may start jobs on the same card.  This helper creates
one CUDA context in the long-lived benchmark driver process.  It does not launch
any kernel and does not allocate benchmark buffers; it is only a visibility
marker.  Set EGGPU_GPU_VISIBILITY_MARKER_MB to reserve a small, fixed amount of
device memory for a more obvious nvidia-smi/nvtop footprint.
"""

import ctypes
import ctypes.util
import math
import os
import sys
from pathlib import Path


TRUE_VALUES = {"1", "TRUE", "ON", "YES"}


def marker_enabled():
    return os.environ.get("EGGPU_GPU_VISIBILITY_MARKER", "").strip().upper() in TRUE_VALUES


def marker_memory_mb():
    raw = os.environ.get("EGGPU_GPU_VISIBILITY_MARKER_MB", "0").strip()
    if not raw:
        return 0
    try:
        value = int(float(raw))
    except ValueError:
        return 0
    return max(0, value)


def _marker_adjust_raw():
    return os.environ.get("EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB", "").strip()


def _auto_marker_adjust_requested():
    raw = _marker_adjust_raw()
    return not raw or raw.upper() == "AUTO"


def _nvml_device_used_mb(gpu_index):
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used) / (1024.0 * 1024.0)
    except Exception:
        return None


def _format_mb(value):
    if value is None:
        return ""
    if math.isclose(value, round(value), abs_tol=0.001):
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _candidate_cudart_paths():
    seen = set()
    roots = [
        os.environ.get("EGGPU_CUDA_ROOT"),
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDA_HOME"),
        os.environ.get("CONDA_PREFIX"),
    ]
    for root in roots:
        if not root:
            continue
        for rel in ("lib/libcudart.so", "targets/x86_64-linux/lib/libcudart.so"):
            path = str(Path(root) / rel)
            if path not in seen:
                seen.add(path)
                yield path
    found = ctypes.util.find_library("cudart")
    if found and found not in seen:
        seen.add(found)
        yield found
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        if name not in seen:
            seen.add(name)
            yield name


def _load_cudart():
    errors = []
    for candidate in _candidate_cudart_paths():
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise OSError("; ".join(errors[-3:]) if errors else "libcudart not found")


class GpuVisibilityMarker:
    def __init__(self, gpu_index, label="EGGPU benchmark"):
        self.gpu_index = str(gpu_index)
        self.label = label
        self.started = False
        self.error = None
        self.lib = None
        self.loaded_from = None
        self.ptr = ctypes.c_void_p()
        self.allocated_mb = 0
        self.allocated_bytes = 0

    def start(self):
        if not marker_enabled():
            return self
        auto_adjust = _auto_marker_adjust_requested()
        before_used_mb = _nvml_device_used_mb(self.gpu_index) if auto_adjust else None
        try:
            # Make direct invocations behave like the benchmark runner: expose
            # only the requested physical GPU, then use logical CUDA device 0.
            if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
                os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_index
            self.lib, self.loaded_from = _load_cudart()
            self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
            self.lib.cudaSetDevice.restype = ctypes.c_int
            self.lib.cudaFree.argtypes = [ctypes.c_void_p]
            self.lib.cudaFree.restype = ctypes.c_int
            self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            self.lib.cudaMalloc.restype = ctypes.c_int

            err = int(self.lib.cudaSetDevice(0))
            if err != 0:
                raise RuntimeError(f"cudaSetDevice(0) failed with cudaError_t={err}")
            err = int(self.lib.cudaFree(ctypes.c_void_p(0)))
            if err != 0:
                raise RuntimeError(f"cudaFree(0) failed with cudaError_t={err}")
            requested_mb = marker_memory_mb()
            if requested_mb > 0:
                requested_bytes = requested_mb * 1024 * 1024
                ptr = ctypes.c_void_p()
                err = int(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(requested_bytes)))
                if err != 0:
                    raise RuntimeError(
                        f"cudaMalloc({requested_mb} MiB marker) failed with cudaError_t={err}"
                    )
                self.ptr = ptr
                self.allocated_mb = requested_mb
                self.allocated_bytes = requested_bytes
            marker_adjust_mb = None
            if auto_adjust:
                after_used_mb = _nvml_device_used_mb(self.gpu_index)
                if before_used_mb is not None and after_used_mb is not None:
                    marker_adjust_mb = max(0.0, after_used_mb - before_used_mb)
                if marker_adjust_mb is None or marker_adjust_mb <= 0:
                    marker_adjust_mb = float(self.allocated_mb)
                os.environ["EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB"] = _format_mb(marker_adjust_mb)
            else:
                marker_adjust_mb = None
            self.started = True
            print(
                f"[EGGPU marker] {self.label} is visible on CUDA_VISIBLE_DEVICES="
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')} via {self.loaded_from}; "
                f"marker_mb={self.allocated_mb}, "
                f"whole_device_adjust_mb={os.environ.get('EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB', '0')}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            self.error = str(exc)
            print(f"[EGGPU marker] disabled: {self.error}", file=sys.stderr, flush=True)
        return self

    def stop(self):
        if self.lib is not None and self.ptr is not None and self.ptr.value:
            try:
                self.lib.cudaFree(self.ptr)
            except Exception:
                pass
        self.ptr = ctypes.c_void_p()
        self.allocated_mb = 0
        self.allocated_bytes = 0
        return self

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
