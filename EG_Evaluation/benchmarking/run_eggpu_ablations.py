#!/usr/bin/env python3
"""Focused EGGPU ablations.

This script is intentionally separate from the full baseline runner.  The full
runner isolates libraries/functions for fairness; these ablations isolate
specific EGGPU design choices:

1. workflow: same EasyGraph object, multiple GPU functions, with warmup.
2. return: lazy/dense return latency versus forced materialization.
3. layout: CSR versus COO preparation/copy and a representative COO PageRank.
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from library_baselines import PeakMemoryMonitor
from library_baselines import build_easygraph
from library_baselines import deterministic_weighted_edges
from library_baselines import load_graph
from library_baselines import pick_sources
from library_baselines import sync_gpu
from gpu_visibility_marker import GpuVisibilityMarker


ROOT = Path(__file__).resolve().parents[1]
_GPU_VISIBILITY_MARKER = None
DEFAULT_FUNCTIONS = (
    "PageRank",
    "MST",
    "LCC",
    "WCC",
    "SCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
    "BC",
    "Closeness",
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
)
RETURN_FUNCTIONS = (
    "PageRank",
    "MST",
    "LCC",
    "WCC",
    "SCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
    "BC",
    "Closeness",
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
)


def configure_eggpu_env(args):
    global _GPU_VISIBILITY_MARKER
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    os.environ["EASYGRAPH_GPU_BACKEND"] = "mine"
    os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "TRUE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"
    os.environ.setdefault("EASYGRAPH_GPU_ADAPTIVE_POLICY", "TRUE")
    os.environ.setdefault("EASYGRAPH_GPU_COMPONENT_DENSE_RETURN", "FALSE")
    os.environ.setdefault("EASYGRAPH_GPU_SCC_ACTIVE_TRIM", "TRUE")
    os.environ.setdefault("EASYGRAPH_GPU_SCC_ACTIVE_TRIM_MAX_ITERS", "16")
    os.environ.setdefault("EASYGRAPH_GPU_SCC_DEGREE_PIVOT", "TRUE")
    os.environ.setdefault("EASYGRAPH_GPU_SCC_HOST_ENABLE", "FALSE")
    os.environ.setdefault("EASYGRAPH_GPU_KCORE_HOST_ENABLE", "FALSE")
    os.environ.setdefault("EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_AVG_DEGREE", "10")
    os.environ.setdefault("EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE", "AUTO")
    os.environ.setdefault("EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS", "1024")
    os.environ.setdefault("EASYGRAPH_GPU_BC_WARP_SIZE", "AUTO")
    os.environ.setdefault("EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION", "AUTO")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["EGGPU_MONITOR_GPU_INDEX"] = str(args.gpu)
    cuda_root = (
        os.environ.get("EGGPU_CUDA_ROOT")
        or os.environ.get("CUDA_PATH")
        or os.environ.get("CUDA_HOME")
        or os.environ.get("CUDAToolkit_ROOT")
        or os.environ.get("CONDA_PREFIX")
    )
    if cuda_root:
        # CuPy uses these to discover CUDA/NVCC.  This is process-local and
        # avoids relying on a global /usr/local/cuda installation.
        os.environ.setdefault("CUDA_PATH", cuda_root)
        os.environ.setdefault("CONDA_PREFIX", cuda_root)
    if args.variant == "no_graph_context":
        os.environ["EASYGRAPH_GPU_DISABLE_GRAPH_CONTEXT_CACHE"] = "TRUE"
        os.environ["EASYGRAPH_GPU_DISABLE_CPP_GRAPH_CACHE"] = "TRUE"
    if args.variant == "no_cpp_graph_cache":
        os.environ["EASYGRAPH_GPU_DISABLE_CPP_GRAPH_CACHE"] = "TRUE"
    if args.variant == "no_device_csr_cache":
        os.environ["EASYGRAPH_GPU_DISABLE_DEVICE_CSR_CACHE"] = "TRUE"
    if args.variant == "adaptive_policy":
        os.environ["EASYGRAPH_GPU_ADAPTIVE_POLICY"] = "TRUE"
        os.environ["EASYGRAPH_GPU_COMPONENT_DENSE_RETURN"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
    if args.variant == "no_adaptive_policy":
        os.environ["EASYGRAPH_GPU_ADAPTIVE_POLICY"] = "FALSE"
        os.environ["EASYGRAPH_GPU_COMPONENT_DENSE_RETURN"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
    if _GPU_VISIBILITY_MARKER is None:
        _GPU_VISIBILITY_MARKER = GpuVisibilityMarker(args.gpu, "run_eggpu_ablations.py").start()


def kernel_time(key, fallback=None):
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


def timed(callable_obj, sync_after=False):
    monitor = PeakMemoryMonitor().start()
    t0 = time.perf_counter()
    result = callable_obj()
    if sync_after:
        sync_gpu()
    elapsed = time.perf_counter() - t0
    memory = monitor.stop()
    return result, elapsed, memory


def row_common(args, experiment, function, metric, value, status="ok", notes="", extra=None):
    row = {
        "experiment": experiment,
        "variant": args.variant,
        "dataset": args.dataset_name,
        "graph_type": args.graph_type,
        "function": function,
        "metric": metric,
        "value": "" if value is None else float(value),
        "status": status,
        "notes": notes,
    }
    if extra:
        row.update(extra)
    return row


def graph_views(args):
    views = load_graph(str(ROOT / args.edge_path))
    return views


def make_graph_bundle(args, views):
    n_clean, directed_edges, undirected_edges = views["clean"]
    n_all, _, undirected_all = views["all_vertices"]
    directed = args.graph_type == "directed"
    base_edges = directed_edges if directed else undirected_edges
    weighted_base = deterministic_weighted_edges(n_clean, base_edges)
    weighted_undirected_clean = deterministic_weighted_edges(n_clean, undirected_edges)
    weighted_undirected_all = deterministic_weighted_edges(n_all, undirected_all)

    t0 = time.perf_counter()
    graph = build_easygraph(n_clean, weighted_base, directed, weighted=True)
    undirected_graph = build_easygraph(n_clean, weighted_undirected_clean, False, weighted=True)
    mst_graph = build_easygraph(n_all, weighted_undirected_all, False, weighted=True)
    build_seconds = time.perf_counter() - t0

    return {
        "n": n_clean,
        "n_all": n_all,
        "directed": directed,
        "graph": graph,
        "undirected_graph": undirected_graph,
        "mst_graph": mst_graph,
        "build_seconds": build_seconds,
        "sssp_sources": pick_sources(n_clean, args.sssp_sources),
        "bc_sources": pick_sources(n_clean, args.bc_sources),
    }


def prewarm_context(bundle):
    try:
        from easygraph.utils import gpu_mine_backend as mine_backend

        mine_backend._graph_context(bundle["graph"], prewarm_cpp=True)
        mine_backend._graph_context(bundle["undirected_graph"], prewarm_cpp=True)
        mine_backend._graph_context(bundle["mst_graph"], prewarm_cpp=True)
    except Exception:
        pass


def call_function(name, bundle):
    import easygraph as eg

    if name == "PageRank":
        return "pagerank", eg.pagerank(
            bundle["graph"], alpha=0.75, max_iter=200, tol=1.0e-6, weight=None
        )
    if name == "MST":
        return "mst", eg.minimum_spanning_tree(bundle["mst_graph"], weight="weight")
    if name == "LCC":
        return "lcc", eg.clustering(bundle["undirected_graph"])
    if name == "WCC":
        return "cc", list(eg.connected_components(bundle["undirected_graph"]))
    if name == "SCC":
        if bundle["directed"]:
            return "scc", list(eg.strongly_connected_components(bundle["graph"]))
        return "cc", list(eg.connected_components(bundle["undirected_graph"]))
    if name == "BFS":
        return "bfs", eg.multi_source_bfs(
            bundle["graph"], bundle["sssp_sources"], target=None
        )
    if name == "Dijkstra":
        return "dijkstra", eg.multi_source_dijkstra(
            bundle["graph"], bundle["sssp_sources"], weight="weight", target=None
        )
    if name == "BellmanFord":
        return "bellman_ford", eg.multi_source_bellman_ford(
            bundle["graph"], bundle["sssp_sources"], weight="weight", target=None
        )
    if name == "SSSP":
        return "sssp", eg.multi_source_dijkstra(
            bundle["graph"], bundle["sssp_sources"], weight="weight", target=None
        )
    if name == "KCore":
        return "kcore", eg.k_core(bundle["undirected_graph"])
    if name == "BC":
        return "bc", eg.betweenness_centrality(
            bundle["graph"],
            weight=None,
            sources=bundle["bc_sources"],
            normalized=False,
            endpoints=False,
        )
    if name == "Closeness":
        return "closeness", eg.closeness_centrality(
            bundle["graph"],
            weight=None,
            sources=None,
        )
    if name == "EffectiveSize":
        return "effective_size", eg.effective_size(bundle["graph"], weight=None)
    if name == "Efficiency":
        return "efficiency", eg.efficiency(bundle["graph"], weight=None)
    if name == "Constraint":
        return "constraint", eg.constraint(bundle["graph"], weight=None)
    if name == "Hierarchy":
        return "hierarchy", eg.hierarchy(bundle["graph"], weight=None)
    raise ValueError(f"unknown function: {name}")


def result_size(result):
    try:
        return len(result)
    except Exception:
        return 0


def materialize_result(function, result):
    t0 = time.perf_counter()
    checksum = 0.0
    count = 0
    if function in {
        "PageRank",
        "LCC",
        "KCore",
        "BC",
        "Closeness",
        "EffectiveSize",
        "Efficiency",
        "Constraint",
        "Hierarchy",
    }:
        if hasattr(result, "items"):
            for _, v in result.items():
                try:
                    checksum += float(v)
                    count += 1
                except Exception:
                    pass
        elif isinstance(result, (list, tuple)):
            for v in result:
                try:
                    checksum += float(v)
                    count += 1
                except Exception:
                    pass
    elif function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        if hasattr(result, "items"):
            for _, dist_map in result.items():
                if hasattr(dist_map, "items"):
                    for _, v in dist_map.items():
                        try:
                            x = float(v)
                        except Exception:
                            continue
                        if math.isfinite(x) and abs(x) < 1.0e30:
                            checksum += x
                            count += 1
        elif isinstance(result, (list, tuple)):
            for row in result:
                for v in row:
                    try:
                        x = float(v)
                    except Exception:
                        continue
                    if math.isfinite(x) and abs(x) < 1.0e30:
                        checksum += x
                        count += 1
    elif function == "MST":
        edges = list(result.edges)
        count = len(edges)
        for edge in edges:
            if len(edge) >= 3 and isinstance(edge[2], dict):
                checksum += float(edge[2].get("weight", 1.0))
        _ = result.adj
    else:
        if isinstance(result, (list, tuple)):
            count = sum(len(x) for x in result if hasattr(x, "__len__"))
    return time.perf_counter() - t0, count, checksum


def run_workflow(args):
    configure_eggpu_env(args)
    views = graph_views(args)
    bundle = make_graph_bundle(args, views)
    functions = parse_functions(args.functions, DEFAULT_FUNCTIONS)
    prewarm_context(bundle)
    for _ in range(max(0, args.warmup)):
        for fn in functions:
            try:
                call_function(fn, bundle)
            except Exception:
                pass
    sync_gpu()

    rows = [
        row_common(
            args,
            "workflow",
            "ALL",
            "build_graph_bundle",
            bundle["build_seconds"],
            notes="EasyGraph graph construction for directed/undirected/MST views; not included in per-function e2e",
        )
    ]
    for rep in range(args.repeat):
        for fn in functions:
            try:
                key_guess = {
                    "PageRank": "pagerank",
                    "MST": "mst",
                    "LCC": "lcc",
                    "WCC": "cc",
                    "SCC": "scc" if bundle["directed"] else "cc",
                    "BFS": "bfs",
                    "Dijkstra": "dijkstra",
                    "BellmanFord": "bellman_ford",
                    "SSSP": "sssp",
                    "KCore": "kcore",
                    "BC": "bc",
                    "Closeness": "closeness",
                    "EffectiveSize": "effective_size",
                    "Efficiency": "efficiency",
                    "Constraint": "constraint",
                    "Hierarchy": "hierarchy",
                }[fn]
                reset_kernel(key_guess)
                result, e2e, mem = timed(lambda fn=fn: call_function(fn, bundle), sync_after=False)
                kernel_key, value = result
                kernel = kernel_time(kernel_key, e2e)
                rows.append(row_common(args, "workflow", fn, "e2e", e2e, extra={"repeat": rep, "result_len": result_size(value)}))
                rows.append(row_common(args, "workflow", fn, "kernel", kernel, extra={"repeat": rep, "result_len": result_size(value)}))
                for mkey, mval in mem.items():
                    if mval is not None and mkey != "gpu_index":
                        rows.append(row_common(args, "workflow", fn, f"memory_{mkey}", mval, extra={"repeat": rep}))
            except Exception as exc:
                rows.append(row_common(args, "workflow", fn, "e2e", None, status="failed", notes=f"{type(exc).__name__}: {exc}", extra={"repeat": rep}))
    return rows


def run_return(args):
    configure_eggpu_env(args)
    views = graph_views(args)
    bundle = make_graph_bundle(args, views)
    functions = parse_functions(args.functions, RETURN_FUNCTIONS)
    prewarm_context(bundle)
    for _ in range(max(0, args.warmup)):
        for fn in functions:
            try:
                call_function(fn, bundle)
            except Exception:
                pass
    sync_gpu()

    rows = []
    for rep in range(args.repeat):
        for fn in functions:
            try:
                reset_key = {
                    "PageRank": "pagerank",
                    "MST": "mst",
                    "LCC": "lcc",
                    "WCC": "cc",
                    "SCC": "scc" if bundle["directed"] else "cc",
                    "BFS": "bfs",
                    "Dijkstra": "dijkstra",
                    "BellmanFord": "bellman_ford",
                    "SSSP": "sssp",
                    "KCore": "kcore",
                    "BC": "bc",
                    "Closeness": "closeness",
                    "EffectiveSize": "effective_size",
                    "Efficiency": "efficiency",
                    "Constraint": "constraint",
                    "Hierarchy": "hierarchy",
                }.get(fn, "")
                if reset_key:
                    reset_kernel(reset_key)
                result_pair, call_s, mem = timed(lambda fn=fn: call_function(fn, bundle), sync_after=False)
                kernel_key, result = result_pair
                kernel = kernel_time(kernel_key, call_s)
                mat_s, mat_count, checksum = materialize_result(fn, result)
                rows.append(row_common(args, "return", fn, "lazy_call_e2e", call_s, extra={"repeat": rep, "materialized_count": mat_count, "checksum": checksum}))
                rows.append(row_common(args, "return", fn, "kernel", kernel, extra={"repeat": rep}))
                rows.append(row_common(args, "return", fn, "materialize_extra", mat_s, extra={"repeat": rep, "materialized_count": mat_count, "checksum": checksum}))
                rows.append(row_common(args, "return", fn, "eager_equivalent_e2e", call_s + mat_s, notes="lazy call plus forced Python materialization", extra={"repeat": rep, "materialized_count": mat_count, "checksum": checksum}))
                for mkey, mval in mem.items():
                    if mval is not None and mkey != "gpu_index":
                        rows.append(row_common(args, "return", fn, f"memory_{mkey}", mval, extra={"repeat": rep}))
            except Exception as exc:
                rows.append(row_common(args, "return", fn, "lazy_call_e2e", None, status="failed", notes=f"{type(exc).__name__}: {exc}", extra={"repeat": rep}))
    return rows


def build_layout_arrays(args, views):
    directed = args.graph_type == "directed"
    n, directed_edges, undirected_edges = views["clean"]
    edges = directed_edges if directed else undirected_edges
    src = edges["src"].to_numpy(dtype=np.int32, copy=True)
    dst = edges["dst"].to_numpy(dtype=np.int32, copy=True)
    return n, src, dst


def csr_from_coo(n, src, dst):
    order = np.lexsort((dst, src))
    src_s = src[order].astype(np.int32, copy=False)
    dst_s = dst[order].astype(np.int32, copy=False)
    rowptr = np.zeros(int(n) + 1, dtype=np.int32)
    np.add.at(rowptr, src_s.astype(np.int64) + 1, 1)
    np.cumsum(rowptr, out=rowptr)
    return rowptr, dst_s


def run_layout(args):
    configure_eggpu_env(args)
    views = graph_views(args)
    rows = []
    for rep in range(args.repeat):
        t0 = time.perf_counter()
        n, src, dst = build_layout_arrays(args, views)
        coo_build = time.perf_counter() - t0
        t1 = time.perf_counter()
        rowptr, col = csr_from_coo(n, src, dst)
        csr_build = time.perf_counter() - t1
        m = int(len(src))
        coo_bytes = src.nbytes + dst.nbytes
        csr_bytes = rowptr.nbytes + col.nbytes
        rows.append(row_common(args, "layout", "COO", "build_seconds", coo_build, notes="COO edge-list arrays src,dst", extra={"repeat": rep, "nodes": n, "edges": m, "bytes": coo_bytes}))
        rows.append(row_common(args, "layout", "CSR", "build_seconds", csr_build, notes="sort COO and build rowptr/col", extra={"repeat": rep, "nodes": n, "edges": m, "bytes": csr_bytes}))
        rows.append(row_common(args, "layout", "COO", "host_storage_mb", coo_bytes / (1024.0 * 1024.0), extra={"repeat": rep, "nodes": n, "edges": m}))
        rows.append(row_common(args, "layout", "CSR", "host_storage_mb", csr_bytes / (1024.0 * 1024.0), extra={"repeat": rep, "nodes": n, "edges": m}))

        t2 = time.perf_counter()
        deg_coo = np.bincount(src, minlength=n)
        coo_deg = time.perf_counter() - t2
        t3 = time.perf_counter()
        deg_csr = rowptr[1:] - rowptr[:-1]
        csr_deg = time.perf_counter() - t3
        rows.append(row_common(args, "layout", "COO", "degree_seconds", coo_deg, notes="degree by COO bincount", extra={"repeat": rep, "checksum": int(deg_coo.sum())}))
        rows.append(row_common(args, "layout", "CSR", "degree_seconds", csr_deg, notes="degree by CSR rowptr diff", extra={"repeat": rep, "checksum": int(deg_csr.sum())}))

        rows.extend(run_optional_coo_pagerank(args, n, src, dst, rep))
        rows.extend(run_csr_pagerank(args, n, src, dst, rep))
    return rows


def run_optional_coo_pagerank(args, n, src, dst, rep):
    rows = []
    try:
        import cupy as cp
    except Exception as exc:
        rows.append(row_common(args, "layout", "COO-PageRank", "gpu_kernel_seconds", None, status="skipped", notes=f"cupy unavailable: {exc}", extra={"repeat": rep}))
        return rows

    alpha = 0.75
    iters = max(1, int(args.layout_pr_iters))
    try:
        sync_gpu()
        t_copy0 = time.perf_counter()
        d_src = cp.asarray(src)
        d_dst = cp.asarray(dst)
        cp.cuda.runtime.deviceSynchronize()
        copy_s = time.perf_counter() - t_copy0
        outdeg = cp.bincount(d_src, minlength=int(n)).astype(cp.float64)
        rank = cp.full(int(n), 1.0 / max(1, int(n)), dtype=cp.float64)
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        for _ in range(iters):
            safe_outdeg = cp.maximum(outdeg[d_src], 1.0)
            contrib = rank[d_src] / safe_outdeg
            nxt = cp.zeros(int(n), dtype=cp.float64)
            cp.add.at(nxt, d_dst, contrib)
            dangling = rank[outdeg == 0].sum()
            rank = (1.0 - alpha) / max(1, int(n)) + alpha * (nxt + dangling / max(1, int(n)))
        stop.record()
        stop.synchronize()
        kernel_s = float(cp.cuda.get_elapsed_time(start, stop)) / 1000.0
        checksum = float(rank.sum().get())
        rows.append(row_common(args, "layout", "COO-PageRank", "h2d_seconds", copy_s, notes="COO src,dst copy only", extra={"repeat": rep, "nodes": int(n), "edges": int(len(src))}))
        rows.append(row_common(args, "layout", "COO-PageRank", "gpu_kernel_seconds", kernel_s, notes=f"CuPy COO scatter PageRank, fixed {iters} iterations", extra={"repeat": rep, "nodes": int(n), "edges": int(len(src)), "checksum": checksum}))
    except Exception as exc:
        rows.append(row_common(args, "layout", "COO-PageRank", "gpu_kernel_seconds", None, status="failed", notes=f"{type(exc).__name__}: {exc}", extra={"repeat": rep}))
    return rows


def run_csr_pagerank(args, n, src, dst, rep):
    rows = []
    try:
        import pandas as pd
        import easygraph as eg
    except Exception as exc:
        rows.append(row_common(args, "layout", "CSR-PageRank", "e2e_seconds", None, status="failed", notes=f"import failed: {exc}", extra={"repeat": rep}))
        return rows

    try:
        edges = pd.DataFrame({"src": src, "dst": dst})
        directed = args.graph_type == "directed"
        t0 = time.perf_counter()
        graph = build_easygraph(n, edges, directed, weighted=False)
        build_s = time.perf_counter() - t0
        try:
            from easygraph.utils import gpu_mine_backend as mine_backend

            mine_backend._graph_context(graph, prewarm_cpp=True)
            mine_backend.set_last_kernel_time("pagerank", None)
        except Exception:
            pass
        # Fixed-iteration setting makes the comparison closer to the COO
        # scatter microbenchmark.  The algorithms are not claimed to be
        # identical; this is a layout-oriented representative primitive.
        _, e2e_s, mem = timed(
            lambda: eg.pagerank(
                graph,
                alpha=0.75,
                max_iter=max(1, int(args.layout_pr_iters)),
                tol=0.0,
                weight=None,
            ),
            sync_after=False,
        )
        k = kernel_time("pagerank", e2e_s)
        rows.append(row_common(args, "layout", "CSR-PageRank", "build_graph_seconds", build_s, notes="EasyGraph graph build for CSR PageRank representative", extra={"repeat": rep, "nodes": int(n), "edges": int(len(src))}))
        rows.append(row_common(args, "layout", "CSR-PageRank", "e2e_seconds", e2e_s, notes=f"EGGPU CSR PageRank, max_iter={args.layout_pr_iters}, tol=0", extra={"repeat": rep, "nodes": int(n), "edges": int(len(src))}))
        rows.append(row_common(args, "layout", "CSR-PageRank", "gpu_kernel_seconds", k, notes=f"EGGPU CSR PageRank kernel, max_iter={args.layout_pr_iters}, tol=0", extra={"repeat": rep, "nodes": int(n), "edges": int(len(src))}))
        for mkey, mval in mem.items():
            if mval is not None and mkey != "gpu_index":
                rows.append(row_common(args, "layout", "CSR-PageRank", f"memory_{mkey}", mval, extra={"repeat": rep}))
    except Exception as exc:
        rows.append(row_common(args, "layout", "CSR-PageRank", "e2e_seconds", None, status="failed", notes=f"{type(exc).__name__}: {exc}", extra={"repeat": rep}))
    return rows


def parse_functions(value, default):
    if not value or str(value).lower() == "all":
        return list(default)
    out = []
    allowed = set(default)
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"function {token!r} is not valid for this experiment; valid={sorted(allowed)}")
        out.append(token)
    return out or list(default)


def write_rows(rows, out_path):
    if not out_path:
        for row in rows:
            print("RESULT_JSON " + json.dumps(row, sort_keys=True), flush=True)
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["workflow", "return", "layout"], required=True)
    ap.add_argument("--variant", default="full", choices=["full", "no_graph_context", "no_cpp_graph_cache", "no_device_csr_cache", "adaptive_policy", "no_adaptive_policy"])
    ap.add_argument("--edge-path", required=True, help="Dataset path relative to EG_Evaluation root or absolute path.")
    ap.add_argument("--dataset-name", default="")
    ap.add_argument("--graph-type", choices=["directed", "undirected"], required=True)
    ap.add_argument("--functions", default="all")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sssp-sources", type=int, default=8)
    ap.add_argument("--bc-sources", type=int, default=16)
    ap.add_argument("--layout-pr-iters", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.dataset_name:
        args.dataset_name = Path(args.edge_path).stem
    if not Path(args.edge_path).is_absolute():
        args.edge_path = str(Path(args.edge_path))

    if args.experiment == "workflow":
        rows = run_workflow(args)
    elif args.experiment == "return":
        rows = run_return(args)
    elif args.experiment == "layout":
        rows = run_layout(args)
    else:
        raise ValueError(args.experiment)
    write_rows(rows, args.out)


if __name__ == "__main__":
    main()
