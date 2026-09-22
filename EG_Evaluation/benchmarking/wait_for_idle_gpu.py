#!/usr/bin/env python3
"""Wait until one physical GPU is stably idle without touching any process."""

from __future__ import annotations

import argparse
import time

import pynvml


def compute_processes(handle):
    for name in (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    ):
        fn = getattr(pynvml, name, None)
        if fn is None:
            continue
        try:
            return fn(handle) or []
        except Exception:
            continue
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-memory-mib", type=float, default=1024.0)
    parser.add_argument("--max-utilization", type=float, default=5.0)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    stable = 0
    while stable < max(1, args.stable_checks):
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024.0 * 1024.0)
        utilization = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        processes = compute_processes(handle)
        process_rows = []
        for process in processes:
            pid = int(getattr(process, "pid", -1))
            used = int(getattr(process, "usedGpuMemory", 0))
            used_mib = 0.0 if used < 0 or used >= (1 << 62) else used / (1024.0 * 1024.0)
            process_rows.append(f"{pid}:{used_mib:.0f}MiB")
        idle = (
            not process_rows
            and memory <= args.max_memory_mib
            and utilization <= args.max_utilization
        )
        stable = stable + 1 if idle else 0
        print(
            f"[idle-wait] gpu={args.gpu} memory={memory:.0f}MiB util={utilization:.0f}% "
            f"compute=[{','.join(process_rows) or 'none'}] stable={stable}/{args.stable_checks}",
            flush=True,
        )
        if stable < max(1, args.stable_checks):
            time.sleep(max(1.0, args.poll_seconds))
    print(f"[idle-wait] GPU {args.gpu} is stably idle", flush=True)


if __name__ == "__main__":
    main()
