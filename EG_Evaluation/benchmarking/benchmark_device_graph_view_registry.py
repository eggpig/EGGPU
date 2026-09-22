#!/usr/bin/env python3
"""Measure alternating-view reuse in the shared EGGPU device CSR registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

from easygraph_runtime_provenance import (
    collect_loaded_runtime_provenance,
    collect_relevant_environment,
    collect_runtime_repository_provenance,
    install_runtime_import_root,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--easygraph-repo",
        type=Path,
        required=True,
        help="Frozen Easy-Graph runtime containing easygraph/ and cpp_easygraph*.so.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--preheat-sequences", type=int, default=4)
    parser.add_argument("--max-entries", type=int, choices=range(1, 17), required=True)
    parser.add_argument("--call-path", choices=("public", "dense"), default="public")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_directed_graph(path):
    import easygraph as eg

    edges = set()
    nodes = set()
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line[0] in "#%/c":
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                u = int(fields[0])
                v = int(fields[1])
            except ValueError:
                continue
            if u == v:
                continue
            edges.add((u, v))
            nodes.add(u)
            nodes.add(v)
    ordered = sorted(nodes)
    remap = {node: index for index, node in enumerate(ordered)}
    graph = eg.DiGraph()
    graph.add_nodes_from(range(len(ordered)))
    modulus = max(1, len(ordered))
    for u, v in sorted(edges):
        src = remap[u]
        dst = remap[v]
        graph.add_edge(src, dst, weight=1 + (src * dst) % modulus)
    return graph


def synchronize():
    try:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def timed_call(callable_):
    synchronize()
    start = time.perf_counter()
    result = callable_()
    if (
        hasattr(result, "__iter__")
        and not isinstance(result, (dict, list, tuple))
        and not hasattr(result, "shape")
        and not hasattr(result, "to_numpy")
    ):
        result = list(result)
    synchronize()
    return time.perf_counter() - start, result


def result_signature(name, result):
    import numpy as np

    if name == "WCC":
        if isinstance(result, np.ndarray):
            labels = np.asarray(result, dtype=np.int64)
            return {
                "components": int(np.unique(labels).size),
                "nodes": int(labels.size),
                "sha256": hashlib.sha256(labels.tobytes(order="C")).hexdigest(),
            }
        canonical = tuple(sorted(tuple(sorted(component)) for component in result))
        payload = repr(canonical).encode("utf-8")
        return {
            "components": len(canonical),
            "nodes": sum(len(component) for component in canonical),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    if isinstance(result, np.ndarray):
        values = np.asarray(result, dtype=np.float64)
    elif hasattr(result, "to_numpy"):
        values = np.asarray(result.to_numpy(copy=True), dtype=np.float64)
    else:
        rows = []
        for source in sorted(result):
            row = result[source]
            if hasattr(row, "to_numpy"):
                rows.append(np.asarray(row.to_numpy(copy=True), dtype=np.float64))
            else:
                rows.append(np.asarray([value for _, value in sorted(row.items())], dtype=np.float64))
        values = np.asarray(rows, dtype=np.float64)
    canonical = np.nan_to_num(values, nan=9.87654321e299, posinf=8.76543210e299, neginf=-8.76543210e299)
    return {
        "shape": list(values.shape),
        "finite": int(np.count_nonzero(np.isfinite(values))),
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def main():
    args = parse_args()
    requested_runtime = collect_runtime_repository_provenance(args.easygraph_repo)
    install_runtime_import_root(args.easygraph_repo)
    os.environ["EASYGRAPH_GPU_DEVICE_CSR_CACHE_MAX_ENTRIES"] = str(args.max_entries)
    os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"

    import cpp_easygraph
    import easygraph as eg
    loaded_runtime = collect_loaded_runtime_provenance(args.easygraph_repo)
    if loaded_runtime["native_sha256"] != requested_runtime["native_sha256"]:
        raise RuntimeError(
            "loaded cpp_easygraph does not match the requested runtime artifact"
        )

    graph = load_directed_graph(args.dataset.resolve())
    source = next(iter(graph))

    if args.call_path == "dense":
        directed_cpp = cpp_easygraph.cpp_graph_from_easygraph(graph, directed=True)
        undirected_cpp = cpp_easygraph.cpp_graph_from_easygraph(graph, directed=False)

        def wcc():
            out = cpp_easygraph.cpp_gpu_connected_component_labels_dense(
                undirected_cpp, directed=False
            )
            return out["labels_dense"]

        def sssp():
            out = cpp_easygraph.cpp_gpu_dijkstra_multisource_dense(
                directed_cpp, sources=[source], weight="weight", target=None
            )
            return out["values_dense"]
    else:
        def wcc():
            return list(eg.weakly_connected_components(graph))

        def sssp():
            return eg.multi_source_dijkstra(graph, sources=[source], weight="weight")

    # Initialize Python/C++ graph state and function-specific workspaces. Extra
    # sequences bring the GPU to a stable clock before the device-view registry
    # alone is reset for the measured cold-registry workflow.
    _, warm_wcc = timed_call(wcc)
    _, warm_sssp = timed_call(sssp)
    reference_signatures = {
        "WCC": result_signature("WCC", warm_wcc),
        "SSSP": result_signature("SSSP", warm_sssp),
    }
    for _ in range(args.preheat_sequences):
        for name, callback in (("WCC", wcc), ("SSSP", sssp), ("WCC", wcc)):
            _, result = timed_call(callback)
            if result_signature(name, result) != reference_signatures[name]:
                raise RuntimeError(f"{name} changed during preheating")
    cpp_easygraph.cpp_gpu_reset_device_csr_cache()

    samples = []
    signatures = []
    for _ in range(args.repeat):
        call_samples = []
        for name, callback in (("WCC", wcc), ("SSSP", sssp), ("WCC", wcc)):
            seconds, result = timed_call(callback)
            call_samples.append({"function": name, "seconds": seconds})
            signature = result_signature(name, result)
            signatures.append((name, signature))
            if signature != reference_signatures[name]:
                raise RuntimeError(f"{name} result changed in measured sequence")
        samples.append({
            "calls": call_samples,
            "sequence_seconds": sum(item["seconds"] for item in call_samples),
        })

    if len(warm_wcc) == 0 or len(warm_sssp) == 0:
        raise RuntimeError("warmup produced an empty result")
    sequence_times = [sample["sequence_seconds"] for sample in samples]
    payload = {
        "argv": list(sys.argv),
        "easygraph_repo": str(Path(args.easygraph_repo).expanduser().resolve()),
        "runtime_provenance": requested_runtime,
        "loaded_runtime_provenance": loaded_runtime,
        "environment": collect_relevant_environment(),
        "dataset": str(args.dataset.resolve()),
        "nodes": len(graph),
        "directed_edges": graph.number_of_edges(),
        "sequence": ["WCC", "SSSP", "WCC"],
        "call_path": args.call_path,
        "max_entries": args.max_entries,
        "repeat": args.repeat,
        "preheat_sequences": args.preheat_sequences,
        "result_signatures": reference_signatures,
        "sequence_mean_seconds": statistics.fmean(sequence_times),
        "sequence_stdev_seconds": statistics.stdev(sequence_times) if len(sequence_times) > 1 else 0.0,
        "sequence_min_seconds": min(sequence_times),
        "sequence_max_seconds": max(sequence_times),
        "samples": samples,
        "cache_stats": dict(cpp_easygraph.cpp_gpu_device_csr_cache_stats()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
