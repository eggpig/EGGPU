"""Existing process/context observer; terminates only its owned process group."""
from pathlib import Path
import os,time,signal,subprocess

def wait_native(child, contract, expected):
    import psutil
    import pynvml as nvml
    nvml.nvmlInit()
    handles = {nvml.nvmlDeviceGetUUID(nvml.nvmlDeviceGetHandleByIndex(i)): nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())}
    observed = {u: {} for u in expected}; peak = 0; started = time.monotonic()
    try:
        while True:
            try:
                processes = [psutil.Process(child.pid), *psutil.Process(child.pid).children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            own = {x.pid for x in processes}; rss = 0
            for proc in processes:
                try:
                    rss += proc.memory_info().rss
                    line = next((x for x in Path(f"/proc/{proc.pid}/status").read_text().splitlines() if x.startswith("NSpid:")), None)
                    if line: own.update(map(int, line.split()[1:]))
                except (OSError, psutil.NoSuchProcess):
                    pass
            peak = max(peak, rss)
            if peak > contract["rss_cap_bytes"]:
                raise RuntimeError("Owned process-tree host RSS exceeded frozen NR cap")
            for uuid, handle in handles.items():
                for proc in nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if proc.pid in own:
                        if uuid not in observed:
                            raise ValueError("Native process opened an unauthorized GPU")
                        observed[uuid][str(proc.pid)] = int(proc.usedGpuMemory)
            if child.poll() is not None: break
            if time.monotonic()-started > contract["timeout_seconds"]:
                raise TimeoutError("Owned NR native process deadline")
            time.sleep(.05)
    finally:
        nvml.nvmlShutdown()
    if not all(observed.values()):
        raise ValueError("Missing observed native context on selected GPU")
    return dict(contexts=observed, all_selected_observed=True, max_process_tree_rss_bytes=peak,
                seconds=time.monotonic()-started, namespace_pid_mapping=True)

def stop(child):
    if child is None or child.poll() is not None: return
    os.killpg(child.pid, signal.SIGTERM)
    try: child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL); child.wait()
