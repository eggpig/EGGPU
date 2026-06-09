#!/usr/bin/env python3
"""EGGPU same-graph reuse ablation.

This is intentionally separate from the full baseline runner: the full runner
keeps per-function comparisons isolated, while this script measures the user
workflow where one EasyGraph graph object is reused across multiple GPU calls.
"""

import argparse
import csv
import os
import time

os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
os.environ["EASYGRAPH_GPU_BACKEND"] = "mine"
os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "TRUE"
os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
os.environ["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"

from library_baselines import build_easygraph
from library_baselines import deterministic_weighted_edges
from library_baselines import load_graph
from library_baselines import pick_sources


def sync_gpu():
    try:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def kernel_time(key, fallback):
    try:
        from easygraph.utils import gpu_mine_backend as mine_backend

        v = mine_backend.get_last_kernel_time(key)
        return float(v) if v is not None else fallback
    except Exception:
        return fallback


def reset_kernel(key):
    try:
        from easygraph.utils import gpu_mine_backend as mine_backend

        mine_backend.set_last_kernel_time(key, None)
    except Exception:
        pass


def run_one(name, key, fn):
    reset_kernel(key)
    sync_gpu()
    t0 = time.perf_counter()
    result = fn()
    sync_gpu()
    e2e = time.perf_counter() - t0
    k = kernel_time(key, e2e)
    try:
        n = len(result)
    except Exception:
        n = 0
    return {"function": name, "e2e": e2e, "kernel": k, "result_len": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edge_path")
    ap.add_argument("graph_type", choices=["directed", "undirected"])
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--sssp-sources", type=int, default=8)
    ap.add_argument("--bc-sources", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import easygraph as eg

    views = load_graph(args.edge_path)
    n, directed_edges, undirected_edges = views["clean"]
    base_edges = directed_edges if args.graph_type == "directed" else undirected_edges
    weighted_edges = deterministic_weighted_edges(n, base_edges)
    graph = build_easygraph(n, weighted_edges, args.graph_type == "directed", weighted=True)

    undirected_graph = graph
    if args.graph_type == "directed":
        _, _, undirected_projection = views["clean"]
        undirected_weighted = deterministic_weighted_edges(n, undirected_projection)
        undirected_graph = build_easygraph(n, undirected_weighted, False, weighted=True)

    sssp_sources = pick_sources(n, args.sssp_sources)
    bc_sources = pick_sources(n, args.bc_sources)
    rows = []
    for pass_idx in range(max(1, args.passes)):
        jobs = [
            ("PageRank", "pagerank", lambda: eg.pagerank(graph, alpha=0.75, max_iter=200, tol=1e-6, weight=None)),
            ("MST", "mst", lambda: eg.minimum_spanning_tree(undirected_graph, weight="weight")),
            ("LCC", "lcc", lambda: eg.clustering(undirected_graph)),
            ("WCC", "cc", lambda: list(eg.connected_components(undirected_graph))),
            (
                "SCC",
                "scc" if args.graph_type == "directed" else "cc",
                lambda: list(eg.strongly_connected_components(graph))
                if args.graph_type == "directed"
                else list(eg.connected_components(undirected_graph)),
            ),
            ("SSSP", "sssp", lambda: eg.multi_source_dijkstra(graph, sssp_sources, weight="weight")),
            ("KCore", "kcore", lambda: eg.k_core(undirected_graph)),
            (
                "BC",
                "bc",
                lambda: eg.betweenness_centrality(
                    graph,
                    weight=None,
                    sources=bc_sources,
                    normalized=False,
                ),
            ),
        ]
        for name, key, fn in jobs:
            try:
                row = run_one(name, key, fn)
                row.update({"pass": pass_idx, "status": "ok"})
            except Exception as e:
                row = {
                    "pass": pass_idx,
                    "function": name,
                    "e2e": "",
                    "kernel": "",
                    "result_len": 0,
                    "status": f"failed: {type(e).__name__}: {e}",
                }
            rows.append(row)
            print(row, flush=True)

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["pass", "function", "status", "e2e", "kernel", "result_len"],
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
