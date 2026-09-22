#!/usr/bin/env python3
"""Qualify nx-cugraph on EGGPU's real CSR scale anchors.

This is an appendix-scale qualification, not a replacement for the main
Python-graph benchmark.  nx-cugraph receives a favorable device COO graph
constructed from the same normalized CSR, so graph preparation is recorded
separately from the public NetworkX API call.  Successful calls still include
nx-cugraph's Python-compatible result materialization.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


FUNCTIONS = ("PageRank", "WCC", "BFS", "KCore")
NVRTC13 = Path(
    "/home/dataset-assist-0/einwang/conda_cache/conda_env/"
    "tongyideepresearch/lib/libnvrtc.so.13"
)
CUDA13_PREFIX = NVRTC13.parents[1]


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


class ProcessMemoryMonitor:
    def __init__(self, physical_gpu: int, interval: float = 0.005):
        self.physical_gpu = physical_gpu
        self.interval = interval
        self.pid = os.getpid()
        self.gpu_peak_bytes = 0
        self.rss_peak_bytes = 0
        self.sample_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.physical_gpu)
        except Exception:
            pynvml = None
            handle = None
        try:
            while not self._stop.is_set():
                self.sample_count += 1
                try:
                    status = Path(f"/proc/{self.pid}/status").read_text()
                    for line in status.splitlines():
                        if line.startswith("VmRSS:"):
                            self.rss_peak_bytes = max(
                                self.rss_peak_bytes, int(line.split()[1]) * 1024
                            )
                            break
                except Exception:
                    pass
                if handle is not None:
                    try:
                        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                            if proc.pid == self.pid:
                                self.gpu_peak_bytes = max(
                                    self.gpu_peak_bytes, int(proc.usedGpuMemory)
                                )
                                break
                    except Exception:
                        pass
                self._stop.wait(self.interval)
        finally:
            if pynvml is not None:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass


def load_device_graph(manifest_path: Path):
    ctypes.CDLL(str(NVRTC13), mode=ctypes.RTLD_GLOBAL)
    os.environ["CONDA_PREFIX"] = str(CUDA13_PREFIX)
    os.environ["CUDA_PATH"] = str(CUDA13_PREFIX)

    import cupy as cp
    import numpy as np
    import nx_cugraph as nxcg

    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    offsets_path = root / metadata["offsets_path"]
    indices_path = root / metadata["indices_path"]
    started = time.perf_counter()
    host_offsets = np.memmap(
        offsets_path,
        dtype=np.int32,
        mode="r",
        shape=(int(metadata["num_nodes"]) + 1,),
    )
    host_indices = np.memmap(
        indices_path,
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
    cp.cuda.runtime.deviceSynchronize()
    graph_class = nxcg.CudaDiGraph if metadata["directed"] else nxcg.CudaGraph
    graph = graph_class.from_coo(
        int(metadata["num_nodes"]),
        sources,
        destinations,
        use_compat_graph=False,
    )
    cp.cuda.runtime.deviceSynchronize()
    return metadata, graph, time.perf_counter() - started


def select_sources(metadata: dict, count: int = 4) -> list[int]:
    recorded = metadata.get("benchmark_sources_zero_based")
    if isinstance(recorded, list) and len(recorded) >= count:
        return [int(item) for item in recorded[:count]]
    n = int(metadata["num_nodes"])
    return sorted({min(n - 1, index * n // count) for index in range(count)})


def execute(graph, metadata: dict, function: str):
    import cupy as cp
    import networkx as nx

    if function == "PageRank":
        result = nx.pagerank(
            graph,
            alpha=0.75,
            tol=1.0e-6,
            max_iter=200,
            backend="cugraph",
        )
        cp.cuda.runtime.deviceSynchronize()
        score_sum = float(sum(result.values()))
        validation = len(result) == int(metadata["num_nodes"]) and math.isclose(
            score_sum, 1.0, rel_tol=5.0e-5, abs_tol=5.0e-5
        )
        detail = {"result_size": len(result), "score_sum": score_sum}
    elif function == "WCC":
        component_fn = (
            nx.weakly_connected_components
            if metadata["directed"]
            else nx.connected_components
        )
        result = list(component_fn(graph, backend="cugraph"))
        cp.cuda.runtime.deviceSynchronize()
        covered = int(sum(len(component) for component in result))
        validation = covered == int(metadata["num_nodes"])
        detail = {"component_count": len(result), "covered_nodes": covered}
    elif function == "BFS":
        sizes = []
        for source in select_sources(metadata):
            result = nx.single_source_shortest_path_length(
                graph, source, backend="cugraph"
            )
            sizes.append(len(result))
            del result
        cp.cuda.runtime.deviceSynchronize()
        validation = len(sizes) == 4 and all(size > 0 for size in sizes)
        detail = {"sources": select_sources(metadata), "reachable_counts": sizes}
        result = None
    elif function == "KCore":
        if metadata["directed"]:
            raise NotImplementedError(
                "KCore requires an explicit undirected projection under the common contract"
            )
        result = nx.core_number(graph, backend="cugraph")
        cp.cuda.runtime.deviceSynchronize()
        values = result.values()
        minimum = int(min(values)) if result else 0
        maximum = int(max(result.values())) if result else 0
        validation = (
            len(result) == int(metadata["num_nodes"])
            and minimum >= 0
            and maximum <= int(metadata["max_degree"])
        )
        detail = {
            "result_size": len(result),
            "minimum": minimum,
            "maximum": maximum,
        }
    else:
        raise ValueError(function)
    return result, validation, detail


def worker(args) -> int:
    monitor = ProcessMemoryMonitor(args.physical_gpu)
    monitor.start()
    try:
        metadata, graph, preparation_seconds = load_device_graph(args.manifest)
        samples = []
        validation_details = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            result, valid, detail = execute(graph, metadata, args.function)
            elapsed = time.perf_counter() - started
            if elapsed > args.timeout:
                raise TimeoutError(
                    f"public API call exceeded {args.timeout}s: {elapsed:.3f}s"
                )
            if not valid:
                raise RuntimeError(f"result validation failed: {detail}")
            samples.append(elapsed)
            validation_details.append(detail)
            del result
            gc.collect()
        record = {
            "status": "ok",
            "dataset": metadata["name"],
            "function": args.function,
            "baseline": "nx-cugraph",
            "input_path": "normalized_host_csr_to_device_coo",
            "num_nodes": int(metadata["num_nodes"]),
            "num_entries": int(metadata["num_entries"]),
            "directed": bool(metadata["directed"]),
            "graph_prepare_seconds": preparation_seconds,
            "e2e_samples_seconds": samples,
            "e2e_best_seconds": min(samples),
            "e2e_mean_seconds": statistics.mean(samples),
            "e2e_stdev_seconds": sample_std(samples),
            "timing_samples": len(samples),
            "validation": "pass",
            "validation_details": validation_details,
        }
    except NotImplementedError as exc:
        record = {
            "status": "not_applicable",
            "dataset": args.manifest.stem,
            "function": args.function,
            "baseline": "nx-cugraph",
            "error": str(exc),
        }
    except Exception as exc:
        record = {
            "status": "failed",
            "dataset": args.manifest.stem,
            "function": args.function,
            "baseline": "nx-cugraph",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        monitor.stop()
    record["gpu_process_peak_mb"] = monitor.gpu_peak_bytes / (1024 * 1024)
    record["host_rss_peak_mb"] = max(
        monitor.rss_peak_bytes,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    ) / (1024 * 1024)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if record["status"] in {"ok", "not_applicable"} else 2


def classify_failure(stderr: str, returncode: int) -> str:
    lowered = stderr.lower()
    if "out of memory" in lowered or "memoryerror" in lowered:
        return "oom"
    if returncode == 124 or "timed out" in lowered:
        return "timeout"
    return "failed"


def write_summary(out_dir: Path) -> list[dict]:
    records = []
    for path in sorted(out_dir.glob("*.json")):
        if path.name == "nxcugraph_scale_qualification.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(record, dict)
            and record.get("baseline") == "nx-cugraph"
            and record.get("dataset")
            and record.get("function")
        ):
            records.append(record)
    records.sort(key=lambda item: (str(item["dataset"]), str(item["function"])))
    (out_dir / "nxcugraph_scale_qualification.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    import pandas as pd

    flat = [
        {key: value for key, value in record.items() if not isinstance(value, (list, dict))}
        for record in records
    ]
    pd.DataFrame(flat).to_csv(
        out_dir / "nxcugraph_scale_qualification.csv", index=False
    )
    return records


def driver(args) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for manifest in args.manifests:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        for function in args.functions:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--manifest",
                str(manifest.resolve()),
                "--function",
                function,
                "--repeat",
                str(args.repeat),
                "--timeout",
                str(args.timeout),
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
            completed = subprocess.run(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(300.0, args.timeout * args.repeat + 120.0),
                check=False,
            )
            try:
                record = json.loads(completed.stdout.strip().splitlines()[-1])
            except Exception:
                record = {
                    "status": classify_failure(completed.stderr, completed.returncode),
                    "dataset": metadata["name"],
                    "function": function,
                    "baseline": "nx-cugraph",
                    "error": completed.stderr[-4000:],
                    "returncode": completed.returncode,
                }
            record["stderr_tail"] = completed.stderr[-1000:]
            (args.out_dir / f"{metadata['name']}_{function}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"{metadata['name']}/{function}: {record['status']}",
                flush=True,
            )
    write_summary(args.out_dir)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifests", type=Path, nargs="+")
    parser.add_argument("--function", choices=FUNCTIONS)
    parser.add_argument("--functions", nargs="+", choices=FUNCTIONS, default=list(FUNCTIONS))
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--physical-gpu", type=int, default=7)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.worker and (args.manifest is None or args.function is None):
        parser.error("--worker requires --manifest and --function")
    if not args.worker and not args.summarize_only and (not args.manifests or args.out_dir is None):
        parser.error("driver requires --manifests and --out-dir")
    if args.summarize_only and args.out_dir is None:
        parser.error("--summarize-only requires --out-dir")
    return args


def main() -> int:
    args = parse_args()
    if args.summarize_only:
        write_summary(args.out_dir)
        return 0
    return worker(args) if args.worker else driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
