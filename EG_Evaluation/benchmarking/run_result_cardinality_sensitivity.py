#!/usr/bin/env python3
"""Measure EGGPU result-reconstruction cost as component count grows.

The controlled inputs keep the node count and stored CSR entries fixed.  Only
the number of connected components changes.  ``compact_labels`` stops after
the dense host label array is available; ``public_sets`` additionally consumes
the public EasyGraph component iterator into Python sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from child_process_memory_monitor import ChildProcessMemoryMonitor


DEFAULT_NODE_COUNT = 3 * (1 << 18)
DEFAULT_COMPONENT_COUNTS = (1, 64, 4096, 65536, 262144)


def sample_stats(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "best": min(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def active_extension_fingerprint():
    spec = importlib.util.find_spec("cpp_easygraph")
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    if origin is None or not origin.is_file():
        return {"path": str(origin or ""), "sha256": "", "size_bytes": 0}
    return {
        "path": str(origin),
        "sha256": sha256_file(origin),
        "size_bytes": origin.stat().st_size,
    }


def build_cycle_partition(root, node_count, component_count):
    if node_count % component_count:
        raise ValueError("node count must be divisible by component count")
    component_size = node_count // component_count
    if component_size < 3:
        raise ValueError("every undirected cycle must contain at least three nodes")

    graph_dir = Path(root) / f"components_{component_count}"
    graph_dir.mkdir(parents=True, exist_ok=True)
    offsets_path = graph_dir / "offsets.i32"
    indices_path = graph_dir / "indices.i32"
    manifest_path = graph_dir / "manifest.json"

    offsets = np.memmap(
        offsets_path,
        mode="w+",
        dtype=np.int32,
        shape=(node_count + 1,),
    )
    offsets[:] = np.arange(node_count + 1, dtype=np.int64) * 2
    offsets.flush()
    del offsets

    nodes = np.arange(node_count, dtype=np.int64)
    starts = (nodes // component_size) * component_size
    local = nodes - starts
    indices = np.memmap(
        indices_path,
        mode="w+",
        dtype=np.int32,
        shape=(2 * node_count,),
    )
    indices[0::2] = starts + ((local - 1) % component_size)
    indices[1::2] = starts + ((local + 1) % component_size)
    indices.flush()
    del indices, nodes, starts, local

    manifest = {
        "format": "eggpu-csr-v1",
        "generation": 1,
        "name": f"fixed-csr-components-{component_count}",
        "source": "controlled-cycle-partition",
        "directed": False,
        "node_labels": "zero_based_contiguous",
        "offset_dtype": "int32",
        "index_dtype": "int32",
        "offsets_path": offsets_path.name,
        "indices_path": indices_path.name,
        "num_nodes": node_count,
        "num_edges": node_count,
        "num_entries": 2 * node_count,
        "max_degree": 2,
        "nonzero_degree_nodes": node_count,
        "edge_counts": {
            "raw_edge_records": node_count,
            "unique_undirected_edges": node_count,
            "csr_entries": 2 * node_count,
            "self_loops_removed": 0,
            "duplicates_removed": 0,
        },
        "controlled_partition": {
            "component_count": component_count,
            "component_size": component_size,
            "topology": "disjoint undirected cycles",
        },
        "csr_artifacts": {
            "offsets": {
                "path": offsets_path.name,
                "bytes": offsets_path.stat().st_size,
                "sha256": sha256_file(offsets_path),
            },
            "indices": {
                "path": indices_path.name,
                "bytes": indices_path.stat().st_size,
                "sha256": sha256_file(indices_path),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def consume_result(backend, graph, mode):
    result = backend.connected_components(graph, directed=False)
    if mode == "compact_labels":
        if not hasattr(result, "labels_numpy"):
            raise RuntimeError("compact component-label view is unavailable")
        labels = np.asarray(result.labels_numpy(copy=False), dtype=np.int32)
        return result, labels
    if mode == "public_sets":
        components = list(result)
        return components, None
    raise ValueError(mode)


def worker(args):
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    os.environ["EASYGRAPH_GPU_STRICT"] = "TRUE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"

    import easygraph as eg
    from easygraph.utils import gpu_eggpu_backend as backend

    graph = eg.read_eggpu_csr(args.manifest, validate=True)

    def call():
        return consume_result(backend, graph, args.mode)

    for _ in range(args.warmup):
        value, labels = call()
        del value, labels

    e2e_samples = []
    kernel_samples = []
    final_value = None
    final_labels = None
    call_count = 1 if args.measurement == "memory" else args.repeat
    for _ in range(call_count):
        started = time.perf_counter()
        final_value, final_labels = call()
        e2e_samples.append(time.perf_counter() - started)
        kernel_samples.append(float(backend.get_last_kernel_time("cc") or 0.0))
        if len(e2e_samples) < call_count:
            del final_value, final_labels

    expected = int(graph.metadata["controlled_partition"]["component_count"])
    if args.mode == "compact_labels":
        observed = int(np.unique(final_labels).size)
        result_bytes = int(final_labels.nbytes)
        materialized_nodes = 0
    else:
        observed = len(final_value)
        materialized_nodes = sum(len(component) for component in final_value)
        result_bytes = None
        if materialized_nodes != len(graph):
            raise RuntimeError(
                f"public component sets cover {materialized_nodes} nodes, "
                f"expected {len(graph)}"
            )
    if observed != expected:
        raise RuntimeError(f"observed {observed} components, expected {expected}")

    payload = {
        "status": "ok",
        "measurement": args.measurement,
        "mode": args.mode,
        "cpp_easygraph": active_extension_fingerprint(),
        "node_count": len(graph),
        "csr_entries": graph.num_entries,
        "component_count": expected,
        "component_size": len(graph) // expected,
        "e2e": sample_stats(e2e_samples),
        "kernel": sample_stats(kernel_samples),
        "result_bytes": result_bytes,
        "materialized_nodes": materialized_nodes,
        "public_contract": (
            "complete list of Python sets"
            if args.mode == "public_sets"
            else "internal dense int32 component labels"
        ),
    }
    Path(args.worker_output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_worker(args, manifest, mode, measurement, index):
    output = (
        Path(args.output)
        / "raw"
        / f"components_{manifest.parent.name.split('_')[-1]}_{mode}_{measurement}_{index}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--manifest",
        str(manifest),
        "--mode",
        mode,
        "--measurement",
        measurement,
        "--worker-output",
        str(output),
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup if measurement == "timing" else 0),
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "EGGPU_MONITOR_GPU_INDEX": str(args.gpu),
            "EASYGRAPH_ENABLE_GPU": "TRUE",
            "EASYGRAPH_GPU_STRICT": "TRUE",
            "EASYGRAPH_GPU_RESULT_CACHE": "FALSE",
        }
    )
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    monitor = None
    if measurement == "memory":
        monitor = ChildProcessMemoryMonitor(
            process.pid,
            physical_gpu=args.gpu,
            interval_seconds=args.memory_poll_ms / 1000.0,
        ).start()
    try:
        stdout, _ = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, _ = process.communicate()
        raise TimeoutError(f"worker timeout: {' '.join(command)}\n{stdout[-4000:]}")
    memory = monitor.stop() if monitor is not None else None
    if process.returncode != 0:
        raise RuntimeError(
            f"worker failed with exit code {process.returncode}: "
            f"{' '.join(command)}\n{stdout[-8000:]}"
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if memory is not None:
        memory["measurement_window"] = "isolated_worker_process_full_lifetime"
        payload["memory"] = memory
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def aggregate(args, records):
    rows = []
    grouped = {}
    for record in records:
        key = (record["component_count"], record["mode"])
        grouped.setdefault(key, []).append(record)
    for (component_count, mode), group in sorted(grouped.items()):
        timing = [item for item in group if item["measurement"] == "timing"]
        memory = [item for item in group if item["measurement"] == "memory"]
        if len(timing) != 1:
            raise RuntimeError(f"missing timing record for {component_count}/{mode}")
        timing = timing[0]
        rss = [item["memory"]["rss_mb"] for item in memory]
        gpu = [item["memory"]["gpu_proc_peak_mb"] for item in memory]
        rows.append(
            {
                "component_count": component_count,
                "component_size": timing["component_size"],
                "node_count": timing["node_count"],
                "csr_entries": timing["csr_entries"],
                "mode": mode,
                "e2e_best_seconds": timing["e2e"]["best"],
                "e2e_mean_seconds": timing["e2e"]["mean"],
                "e2e_stdev_seconds": timing["e2e"]["stdev"],
                "kernel_mean_seconds": timing["kernel"]["mean"],
                "kernel_stdev_seconds": timing["kernel"]["stdev"],
                "rss_peak_mean_mb": statistics.fmean(rss) if rss else None,
                "rss_peak_stdev_mb": statistics.stdev(rss) if len(rss) > 1 else 0.0,
                "gpu_peak_mean_mb": statistics.fmean(gpu) if gpu else None,
                "gpu_peak_stdev_mb": statistics.stdev(gpu) if len(gpu) > 1 else 0.0,
                "timing_repeat": args.repeat,
                "memory_repeat": args.memory_repeat,
            }
        )

    compact = {
        row["component_count"]: row
        for row in rows
        if row["mode"] == "compact_labels"
    }
    for row in rows:
        reference = compact[row["component_count"]]
        row["e2e_over_compact_mean"] = (
            row["e2e_mean_seconds"] / reference["e2e_mean_seconds"]
        )
        row["e2e_over_compact_best"] = (
            row["e2e_best_seconds"] / reference["e2e_best_seconds"]
        )

    csv_path = Path(args.output) / "result_cardinality_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (Path(args.output) / "protocol.json").write_text(
        json.dumps(
            {
                "purpose": "isolate output reconstruction from fixed graph and kernel input",
                "graph_family": "disjoint undirected cycles",
                "node_count": args.node_count,
                "csr_entries": 2 * args.node_count,
                "component_counts": list(args.component_counts),
                "timing_repeat": args.repeat,
                "memory_repeat": args.memory_repeat,
                "warmup": args.warmup,
                "timing_center_values": {
                    "raw": "arithmetic mean with sample standard deviation",
                    "paper_eggpu": "best observed value over five calls",
                },
                "memory_center_value": "arithmetic mean with sample standard deviation",
                "public_semantics": "all component sets are completely consumed",
                "compact_semantics": "dense int32 labels are available on host; no Python sets constructed",
                "cpp_easygraph": active_extension_fingerprint(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--node-count", type=int, default=DEFAULT_NODE_COUNT)
    parser.add_argument(
        "--component-counts",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        default=DEFAULT_COMPONENT_COUNTS,
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--memory-poll-ms", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", choices=("compact_labels", "public_sets"))
    parser.add_argument("--measurement", choices=("timing", "memory"))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker:
        required = (args.manifest, args.mode, args.measurement, args.worker_output)
        if any(item is None for item in required):
            parser.error("worker mode requires manifest, mode, measurement, and output")
    elif args.output is None:
        parser.error("--output is required")
    return args


def main():
    args = parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    graph_root = args.output / "controlled_graphs"
    manifests = [
        build_cycle_partition(graph_root, args.node_count, count)
        for count in args.component_counts
    ]
    records = []
    for manifest in manifests:
        for mode in ("compact_labels", "public_sets"):
            records.append(run_worker(args, manifest, mode, "timing", 0))
            for index in range(1, args.memory_repeat + 1):
                records.append(run_worker(args, manifest, mode, "memory", index))
    aggregate(args, records)
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
