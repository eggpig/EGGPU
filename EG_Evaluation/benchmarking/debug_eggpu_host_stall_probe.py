#!/usr/bin/env python3
"""Diagnose host-side EGGPU latency stalls without changing benchmark code."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def _delta_usage(before, after):
    return {
        "cpu_seconds": (after.ru_utime + after.ru_stime)
        - (before.ru_utime + before.ru_stime),
        "voluntary_context_switches": after.ru_nvcsw - before.ru_nvcsw,
        "involuntary_context_switches": after.ru_nivcsw - before.ru_nivcsw,
        "major_faults": after.ru_majflt - before.ru_majflt,
        "minor_faults": after.ru_minflt - before.ru_minflt,
    }


def _worker(args):
    repo = str(Path(args.easygraph_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"
    os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "TRUE"
    os.environ.setdefault("EASYGRAPH_GPU_ADAPTIVE_POLICY", "TRUE")
    os.environ.setdefault("EASYGRAPH_GPU_BC_WARP_SIZE", "AUTO")
    os.environ.setdefault("EASYGRAPH_GPU_SCC_HOST_ENABLE", "FALSE")
    os.environ.setdefault("EASYGRAPH_GPU_KCORE_HOST_ENABLE", "FALSE")
    os.environ.setdefault("EASYGRAPH_GPU_SSSP_HOST_ENABLE", "FALSE")

    from library_baselines import barrier, build_easygraph, load_graph, pick_sources

    import easygraph as eg
    from easygraph.utils import gpu_eggpu_backend as backend

    views = load_graph(args.dataset)
    n, directed_edges, undirected_edges = views["clean"]
    edges = directed_edges if args.graph_type == "directed" else undirected_edges
    graph = build_easygraph(n, edges, args.graph_type == "directed")
    backend._graph_context(graph, prewarm_cpp=True)
    sources = pick_sources(n, args.sources)

    def public_call():
        return eg.betweenness_centrality(
            graph,
            weight=None,
            sources=sources,
            normalized=False,
            endpoints=False,
        )

    for _ in range(args.warmup):
        public_call()
    barrier(args.cooldown)

    before = resource.getrusage(resource.RUSAGE_SELF)
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    result = public_call()
    wall_seconds = time.perf_counter() - wall0
    cpu_seconds = time.process_time() - cpu0
    after = resource.getrusage(resource.RUSAGE_SELF)

    usage = _delta_usage(before, after)
    usage["cpu_seconds_process_time"] = cpu_seconds
    usage.update(
        {
            "wall_seconds": wall_seconds,
            "wall_minus_cpu_seconds": wall_seconds - cpu_seconds,
            "kernel_seconds": backend.get_last_kernel_time("bc"),
            "nodes": len(result),
            "sum": float(sum(result)),
            "pid": os.getpid(),
        }
    )
    print(json.dumps(usage, sort_keys=True), flush=True)


def _parent(args):
    rows = []
    script = str(Path(__file__).resolve())
    for index in range(1, args.repetitions + 1):
        command = [
            sys.executable,
            script,
            "--worker",
            "--dataset",
            args.dataset,
            "--graph-type",
            args.graph_type,
            "--easygraph-repo",
            args.easygraph_repo,
            "--gpu",
            str(args.gpu),
            "--sources",
            str(args.sources),
            "--warmup",
            str(args.warmup),
            "--cooldown",
            str(args.cooldown),
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        row = json.loads(completed.stdout.strip().splitlines()[-1])
        row["sample_index"] = index
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    wall = [row["wall_seconds"] for row in rows]
    cpu = [row["cpu_seconds_process_time"] for row in rows]
    summary = {
        "samples": len(rows),
        "wall_min_seconds": min(wall),
        "wall_max_seconds": max(wall),
        "wall_mean_seconds": sum(wall) / len(wall),
        "cpu_min_seconds": min(cpu),
        "cpu_max_seconds": max(cpu),
        "cpu_mean_seconds": sum(cpu) / len(cpu),
        "stalls_over_20ms": sum(value > 0.020 for value in wall),
        "stalls_over_100ms": sum(value > 0.100 for value in wall),
    }
    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--graph-type", choices=("directed", "undirected"), required=True)
    parser.add_argument("--easygraph-repo", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sources", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=0.2)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
