#!/usr/bin/env python3
"""Compare the bulk CSR path against the existing EasyGraph object path."""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path


os.environ.setdefault("EASYGRAPH_GPU_RESULT_CACHE", "FALSE")
os.environ.setdefault("EASYGRAPH_GPU_STRICT_ERRORS", "TRUE")
os.environ.setdefault("EASYGRAPH_GPU_ADAPTIVE_HOST", "FALSE")
os.environ.setdefault("EGGPU_ALLOW_CUDA_SYNC", "TRUE")

import numpy as np

import easygraph as eg
from easygraph.utils import gpu_eggpu_backend

from library_baselines import build_easygraph
from library_baselines import load_graph
from library_baselines import sync_gpu


def timed(callable_obj, kernel_key):
    sync_gpu()
    started = time.perf_counter()
    value = callable_obj()
    sync_gpu()
    elapsed = time.perf_counter() - started
    kernel_seconds = gpu_eggpu_backend.get_last_kernel_time(kernel_key)
    return value, elapsed, float(kernel_seconds)


def samples(callable_obj, kernel_key, warmup, repeat):
    value = None
    for _ in range(warmup):
        value, _, _ = timed(callable_obj, kernel_key)
    durations = []
    kernel_durations = []
    for _ in range(repeat):
        value, elapsed, kernel_seconds = timed(callable_obj, kernel_key)
        durations.append(elapsed)
        kernel_durations.append(kernel_seconds)
    return value, durations, kernel_durations


def paired_samples(regular_call, bulk_call, kernel_key, warmup, repeat):
    """Use AB/BA hot-cache blocks so order drift cannot favor one path."""

    regular_value = bulk_value = None
    regular_durations = []
    bulk_durations = []
    regular_kernel_durations = []
    bulk_kernel_durations = []
    for order in (
        (("regular", regular_call), ("bulk", bulk_call)),
        (("bulk", bulk_call), ("regular", regular_call)),
    ):
        for name, callable_obj in order:
            gc.collect()
            for _ in range(warmup):
                value, _, _ = timed(callable_obj, kernel_key)
                if name == "regular":
                    regular_value = value
                else:
                    bulk_value = value
            for _ in range(repeat):
                value, elapsed, kernel_seconds = timed(callable_obj, kernel_key)
                if name == "regular":
                    regular_value = value
                    regular_durations.append(elapsed)
                    regular_kernel_durations.append(kernel_seconds)
                else:
                    bulk_value = value
                    bulk_durations.append(elapsed)
                    bulk_kernel_durations.append(kernel_seconds)
    return (
        regular_value,
        regular_durations,
        regular_kernel_durations,
        bulk_value,
        bulk_durations,
        bulk_kernel_durations,
    )


def dense_array(result, dtype=None):
    if hasattr(result, "to_numpy"):
        return np.asarray(result.to_numpy(), dtype=dtype)
    if isinstance(result, dict):
        return np.asarray([result[node] for node in range(len(result))], dtype=dtype)
    return np.asarray(result, dtype=dtype)


def component_size_signature(components):
    if hasattr(components, "labels_numpy"):
        _, counts = np.unique(
            np.asarray(components.labels_numpy(), dtype=np.int32),
            return_counts=True,
        )
        return np.sort(counts)
    if isinstance(components, dict):
        _, counts = np.unique(
            dense_array(components, dtype=np.int32), return_counts=True
        )
        return np.sort(counts)
    return np.sort(np.fromiter((len(component) for component in components), dtype=np.int64))


def component_partition_vector(components, node_count):
    """Canonicalize a partition without forcing deferred GPU results into sets."""

    if hasattr(components, "labels_numpy"):
        labels = np.asarray(components.labels_numpy(), dtype=np.int64).reshape(-1)
        if len(labels) != node_count:
            raise ValueError("component label count does not match node count")
        _, inverse = np.unique(labels, return_inverse=True)
        representatives = np.full(int(inverse.max()) + 1, node_count, dtype=np.int64)
        np.minimum.at(representatives, inverse, np.arange(node_count, dtype=np.int64))
        return representatives[inverse]
    if isinstance(components, dict):
        labels = dense_array(components, dtype=np.int64)
        _, inverse = np.unique(labels, return_inverse=True)
        representatives = np.full(int(inverse.max()) + 1, node_count, dtype=np.int64)
        np.minimum.at(representatives, inverse, np.arange(node_count, dtype=np.int64))
        return representatives[inverse]

    partition = np.full(node_count, -1, dtype=np.int64)
    for component in components:
        members = np.fromiter(component, dtype=np.int64)
        if len(members):
            partition[members] = int(members.min())
    return partition


def summary(values):
    return {
        "samples_seconds": values,
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "best_seconds": min(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-path", required=True)
    parser.add_argument("--bulk-manifest", required=True)
    parser.add_argument("--graph-type", choices=["directed", "undirected"], required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--minimum-geomean-speedup", type=float, default=1.02)
    parser.add_argument("--output")
    args = parser.parse_args()

    started = time.perf_counter()
    views = load_graph(args.edge_path)
    raw_load_seconds = time.perf_counter() - started
    n, directed_edges, undirected_edges = views["all_vertices"]
    directed = args.graph_type == "directed"
    edges = directed_edges if directed else undirected_edges
    started = time.perf_counter()
    regular = build_easygraph(n, edges, directed, weighted=False)
    python_graph_build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    bulk = eg.read_eggpu_csr(args.bulk_manifest, validate=True)
    bulk_load_seconds = time.perf_counter() - started
    if len(regular) != len(bulk):
        raise RuntimeError("regular and bulk graph node counts differ")

    source = 0
    calls = {
        "PageRank": (
            "pagerank",
            lambda graph: eg.pagerank(
                graph, alpha=0.75, max_iter=200, tol=1.0e-6
            ),
        ),
        "WCC": (
            "cc",
            lambda graph: gpu_eggpu_backend.connected_components(
                graph, directed=False
            ),
        ),
        "BFS": ("bfs", lambda graph: eg.multi_source_bfs(graph, [source])),
        "KCore": ("kcore", lambda graph: eg.k_core(graph)),
    }
    rows = {}
    failures = []
    for name, (kernel_key, function) in calls.items():
        (
            regular_result,
            regular_samples,
            regular_kernel_samples,
            bulk_result,
            bulk_samples,
            bulk_kernel_samples,
        ) = paired_samples(
            lambda: function(regular),
            lambda: function(bulk),
            kernel_key,
            args.warmup,
            args.repeat,
        )
        if name == "PageRank":
            correct = np.allclose(
                dense_array(regular_result, np.float64),
                dense_array(bulk_result, np.float64),
                rtol=2.0e-5,
                atol=2.0e-7,
            )
        elif name == "WCC":
            correct = np.array_equal(
                component_partition_vector(regular_result, len(bulk)),
                component_partition_vector(bulk_result, len(bulk)),
            )
        elif name == "BFS":
            correct = np.allclose(
                regular_result[source].to_numpy(),
                bulk_result[source].to_numpy(),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        else:
            correct = np.array_equal(
                dense_array(regular_result, np.int32),
                dense_array(bulk_result, np.int32),
            )
        regular_stats = summary(regular_samples)
        bulk_stats = summary(bulk_samples)
        regular_kernel_stats = summary(regular_kernel_samples)
        bulk_kernel_stats = summary(bulk_kernel_samples)
        speedup = regular_stats["mean_seconds"] / bulk_stats["mean_seconds"]
        median_speedup = (
            regular_stats["median_seconds"] / bulk_stats["median_seconds"]
        )
        kernel_median_speedup = (
            regular_kernel_stats["median_seconds"]
            / bulk_kernel_stats["median_seconds"]
        )
        rows[name] = {
            "correct": bool(correct),
            "regular": regular_stats,
            "bulk": bulk_stats,
            "regular_kernel": regular_kernel_stats,
            "bulk_kernel": bulk_kernel_stats,
            "bulk_speedup": speedup,
            "bulk_median_speedup": median_speedup,
            "bulk_kernel_median_speedup": kernel_median_speedup,
            "bulk_median_positive_or_neutral_5pct": bool(
                median_speedup >= 1.0 / 1.05
            ),
        }
        if not correct:
            failures.append(f"{name}: result mismatch")
        if median_speedup < 1.0 / 1.05:
            failures.append(
                f"{name}: bulk path median regressed by more than 5%"
            )
        if kernel_median_speedup < 1.0 / 1.05:
            failures.append(
                f"{name}: bulk path kernel median regressed by more than 5%"
            )

    mean_geomean_speedup = statistics.geometric_mean(
        row["bulk_speedup"] for row in rows.values()
    )
    median_geomean_speedup = statistics.geometric_mean(
        row["bulk_median_speedup"] for row in rows.values()
    )
    kernel_median_geomean_speedup = statistics.geometric_mean(
        row["bulk_kernel_median_speedup"] for row in rows.values()
    )
    if median_geomean_speedup < args.minimum_geomean_speedup:
        failures.append(
            "bulk path geometric-mean median E2E speedup "
            f"{median_geomean_speedup:.4f}x is below required "
            f"{args.minimum_geomean_speedup:.4f}x"
        )

    result = {
        "status": "pass" if not failures else "fail",
        "edge_path": str(Path(args.edge_path).resolve()),
        "bulk_manifest": str(Path(args.bulk_manifest).resolve()),
        "num_nodes": len(bulk),
        "num_edges": bulk.num_edges,
        "num_entries": bulk.num_entries,
        "regular_raw_load_seconds": raw_load_seconds,
        "regular_python_graph_build_seconds": python_graph_build_seconds,
        "bulk_native_load_seconds": bulk_load_seconds,
        "reusable_artifact_load_speedup_vs_raw_text_plus_python_graph": (
            (raw_load_seconds + python_graph_build_seconds) / bulk_load_seconds
        ),
        "function_mean_geomean_e2e_speedup": mean_geomean_speedup,
        "function_median_geomean_e2e_speedup": median_geomean_speedup,
        "function_median_geomean_kernel_speedup": kernel_median_geomean_speedup,
        "minimum_required_geomean_e2e_speedup": args.minimum_geomean_speedup,
        "functions": rows,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
