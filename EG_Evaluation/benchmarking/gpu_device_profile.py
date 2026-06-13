#!/usr/bin/env python3
"""GPU device profile collection for reproducible EGGPU experiments.

The benchmark uses CUDA_VISIBLE_DEVICES for logical CUDA routing, while NVML
memory accounting uses the physical monitor index.  This helper records both
views once per run so cross-GPU comparisons have an auditable hardware profile
without touching the timed per-function path.
"""

from __future__ import annotations

import os
import subprocess

try:
    import pynvml
except Exception:  # pragma: no cover - optional dependency on CPU-only hosts.
    pynvml = None


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _to_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _resolve_monitor_gpu_index(env):
    env = env or os.environ
    raw = str(env.get("EGGPU_MONITOR_GPU_INDEX", "")).strip()
    if raw.isdigit():
        return int(raw)
    cvd = str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if cvd:
        token = cvd.split(",")[0].strip()
        if token.isdigit():
            return int(token)
    return 0


def _memory_total_mb(info):
    try:
        return int(getattr(info, "total")) / (1024.0 * 1024.0)
    except Exception:
        return None


def _nvml_device(index):
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    device = {"index": int(index)}
    getters = (
        ("name", "nvmlDeviceGetName"),
        ("uuid", "nvmlDeviceGetUUID"),
        ("pci_bus_id", "nvmlDeviceGetPciInfo"),
    )
    for field, getter_name in getters:
        getter = getattr(pynvml, getter_name, None)
        if getter is None:
            continue
        try:
            value = getter(handle)
            if field == "pci_bus_id":
                value = getattr(value, "busId", "")
            device[field] = str(_decode(value))
        except Exception:
            pass
    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        device["compute_capability"] = f"{int(major)}.{int(minor)}"
        device["cuda_architecture"] = f"{int(major)}{int(minor)}"
    except Exception:
        pass
    getter = getattr(pynvml, "nvmlDeviceGetMultiProcessorCount", None)
    if getter is not None:
        try:
            device["multiprocessor_count"] = int(getter(handle))
        except Exception:
            pass
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mb = _memory_total_mb(info)
        if total_mb is not None:
            device["memory_total_mb"] = total_mb
    except Exception:
        pass
    return device


def _collect_with_nvml(errors):
    if pynvml is None:
        errors.append("pynvml unavailable")
        return None
    try:
        pynvml.nvmlInit()
        count = int(pynvml.nvmlDeviceGetCount())
        try:
            driver = str(_decode(pynvml.nvmlSystemGetDriverVersion()))
        except Exception:
            driver = ""
        devices = []
        for index in range(count):
            try:
                devices.append(_nvml_device(index))
            except Exception as exc:
                errors.append(f"nvml device {index}: {type(exc).__name__}: {exc}")
        return {"driver_version": driver, "devices": devices}
    except Exception as exc:
        errors.append(f"nvml: {type(exc).__name__}: {exc}")
        return None


def _collect_with_nvidia_smi(errors):
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        errors.append(f"nvidia-smi: {type(exc).__name__}: {exc}")
        return None
    if proc.returncode != 0:
        errors.append(f"nvidia-smi rc={proc.returncode}: {(proc.stderr or proc.stdout).strip()[-300:]}")
        return None
    devices = []
    driver = ""
    for raw in proc.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 6:
            continue
        cap = parts[5].replace(" ", "")
        device = {
            "index": _to_int(parts[0], -1),
            "name": parts[1],
            "uuid": parts[2],
            "driver_version": parts[3],
            "memory_total_mb": float(parts[4]) if parts[4] else None,
            "compute_capability": cap,
            "cuda_architecture": cap.replace(".", ""),
        }
        driver = parts[3] or driver
        devices.append(device)
    return {"driver_version": driver, "devices": devices}


def collect_gpu_device_profile(gpu_index=None, env=None):
    env = env or os.environ
    resolved = int(gpu_index) if str(gpu_index).strip().isdigit() else _resolve_monitor_gpu_index(env)
    errors = []
    profile = {
        "schema_version": 1,
        "requested_gpu_index": gpu_index,
        "cuda_visible_devices": str(env.get("CUDA_VISIBLE_DEVICES", "")),
        "monitor_gpu_index": str(env.get("EGGPU_MONITOR_GPU_INDEX", "")),
        "resolved_monitor_gpu_index": resolved,
        "source": "",
        "driver_version": "",
        "devices": [],
        "selected_device": {},
        "errors": errors,
    }
    collected = _collect_with_nvml(errors)
    if collected is not None and collected.get("devices"):
        profile.update(collected)
        profile["source"] = "nvml"
    else:
        collected = _collect_with_nvidia_smi(errors)
        if collected is not None and collected.get("devices"):
            profile.update(collected)
            profile["source"] = "nvidia-smi"
    for device in profile.get("devices", []):
        if _to_int(device.get("index"), -1) == resolved:
            profile["selected_device"] = dict(device)
            break
    return profile
