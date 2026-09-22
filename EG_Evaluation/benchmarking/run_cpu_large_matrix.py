#!/usr/bin/env python3
"""Measure CPU-library functions on normalized bulk-CSR scale anchors.

The ordinary benchmark loader intentionally favors small and medium text edge
lists.  Expanding a billion-edge CSR into Python tuples and a pandas frame is
not a faithful large-graph qualification.  This runner imports the normalized
CSR through each library's sparse-matrix entry point, qualifies graph loading
once, and then uses fresh processes for every measured sample.  Unsupported
APIs, projection limits, load failures, call failures, and timeouts remain
distinct outcomes.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from child_process_memory_monitor import ChildProcessMemoryMonitor


FUNCTIONS = (
    "PageRank", "MST", "LCC", "WCC", "SCC", "BFS", "Dijkstra",
    "BellmanFord", "SSSP", "KCore", "BC", "Closeness", "EffectiveSize",
    "Efficiency", "Constraint", "Hierarchy",
)
BASELINES = ("igraph", "networkx", "easygraph-cpu", "easygraph-cpp")
PROJECTED = {"MST", "LCC", "KCore"}
WEIGHTED = {"MST", "Dijkstra", "BellmanFord", "SSSP"}
INT32_MAX = (1 << 31) - 1
EASYGRAPH_REPO = Path(__file__).resolve().parents[2] / "Easy-Graph"
_CALL_TIMEOUT_SECONDS = 100.0

SUPPORT = {
    "igraph": {
        "PageRank": "P", "MST": "T", "LCC": "T", "WCC": "T",
        "SCC": "T", "BFS": "T", "Dijkstra": "T", "BellmanFord": "T",
        "SSSP": "T", "KCore": "T", "BC": "T", "Closeness": "P",
        "EffectiveSize": "F", "Efficiency": "F", "Constraint": "P",
        "Hierarchy": "F",
    },
    "networkx": {
        "PageRank": "T", "MST": "T", "LCC": "T", "WCC": "T",
        "SCC": "T", "BFS": "T", "Dijkstra": "T", "BellmanFord": "T",
        "SSSP": "T", "KCore": "T", "BC": "T", "Closeness": "T",
        "EffectiveSize": "T", "Efficiency": "F", "Constraint": "T",
        "Hierarchy": "F",
    },
    "easygraph-cpu": {function: "T" for function in FUNCTIONS},
    "easygraph-cpp": {
        "PageRank": "T", "MST": "T", "LCC": "T", "WCC": "T",
        "SCC": "T", "BFS": "T", "Dijkstra": "T", "BellmanFord": "F",
        "SSSP": "T", "KCore": "T", "BC": "T", "Closeness": "T",
        "EffectiveSize": "F", "Efficiency": "F", "Constraint": "F",
        "Hierarchy": "F",
    },
}


class FunctionTimeout(RuntimeError):
    pass


def sample_stats(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "best": min(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": values,
        "count": len(values),
    }


def weighted_manifest(path: Path, function: str) -> Path:
    if function not in WEIGHTED:
        return path
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("weights_path"):
        return path
    sibling = path.with_name(f"{path.stem}.weighted.json")
    return sibling if sibling.is_file() else path


def applicability(metadata, baseline, function):
    support = SUPPORT[baseline][function]
    if support == "F":
        return False, "unsupported_api", (
            f"{baseline} has no aligned callable {function} implementation under "
            "the audited support contract"
        )
    if function in PROJECTED and metadata.get("directed"):
        projected = 2 * int(metadata.get("num_entries", 0))
        if projected > INT32_MAX:
            return False, "representation_limit", (
                f"{function} requires an undirected projection with up to "
                f"{projected:,} CSR entries, beyond the signed-int32 sparse ABI"
            )
    if function in WEIGHTED and not metadata.get("weights_path"):
        return False, "missing_weight_artifact", (
            "the deterministic weighted sibling manifest is absent"
        )
    return True, "", ""


def set_memory_limit(limit_gb):
    if limit_gb <= 0:
        return
    limit = int(limit_gb * (1024 ** 3))
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def load_sparse(manifest_path: Path, project_undirected: bool):
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    n = int(metadata["num_nodes"])
    m = int(metadata["num_entries"])
    offsets = np.memmap(
        root / metadata["offsets_path"], dtype=np.int32, mode="r", shape=(n + 1,)
    )
    indices = np.memmap(
        root / metadata["indices_path"], dtype=np.int32, mode="r", shape=(m,)
    )
    if metadata.get("weights_path"):
        values = np.memmap(
            root / metadata["weights_path"], dtype=np.float64, mode="r", shape=(m,)
        )
    else:
        values = np.ones(m, dtype=np.uint8)
    matrix = sp.csr_matrix((values, indices, offsets), shape=(n, n), copy=False)
    if project_undirected and metadata.get("directed"):
        matrix = matrix.maximum(matrix.transpose()).tocsr()
        matrix.sort_indices()
    return metadata, matrix


def build_igraph_from_normalized_csr(manifest_path: Path, metadata: dict):
    """Build igraph without python-igraph's sparse-adjacency tuple expansion.

    In python-igraph 1.0.0, ``Graph.Adjacency(scipy_csr)`` converts the matrix
    to COO and then materializes one Python tuple per edge.  The normalized
    scaling manifests already contain a simple, loop-free CSR, so construct a
    contiguous int32 edge buffer instead.  ``Graph`` passes NumPy arrays to the
    C layer through a memoryview and therefore avoids the Python-object list.
    """

    import igraph as ig

    root = manifest_path.parent
    n = int(metadata["num_nodes"])
    m = int(metadata["num_entries"])
    offsets = np.memmap(
        root / metadata["offsets_path"], dtype=np.int32, mode="r", shape=(n + 1,)
    )
    indices = np.memmap(
        root / metadata["indices_path"], dtype=np.int32, mode="r", shape=(m,)
    )
    edges = np.empty((m, 2), dtype=np.int32)
    edges[:, 1] = indices
    vertex_chunk = 1 << 18
    for first_vertex in range(0, n, vertex_chunk):
        last_vertex = min(n, first_vertex + vertex_chunk)
        first_edge = int(offsets[first_vertex])
        last_edge = int(offsets[last_vertex])
        degrees = np.diff(offsets[first_vertex : last_vertex + 1]).astype(
            np.int64, copy=False
        )
        edges[first_edge:last_edge, 0] = np.repeat(
            np.arange(first_vertex, last_vertex, dtype=np.int32), degrees
        )
    graph = ig.Graph(
        n=n,
        edges=edges,
        directed=bool(metadata["directed"]),
    )
    del edges
    gc.collect()
    return graph


def build_graph(baseline, manifest_path, function):
    project = function in PROJECTED
    manifest_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        baseline == "igraph"
        and not project
        and not manifest_metadata.get("weights_path")
    ):
        started = time.perf_counter()
        graph = build_igraph_from_normalized_csr(manifest_path, manifest_metadata)
        elapsed = time.perf_counter() - started
        return (
            manifest_metadata,
            graph,
            elapsed,
            bool(manifest_metadata["directed"]),
            "normalized_csr_to_contiguous_int32_edge_buffer",
        )

    metadata, matrix = load_sparse(manifest_path, project)
    directed = bool(metadata["directed"]) and not project
    started = time.perf_counter()
    if baseline == "igraph":
        import igraph as ig

        mode = "directed" if directed else "undirected"
        if metadata.get("weights_path"):
            graph = ig.Graph.Weighted_Adjacency(
                matrix, mode=mode, attr="weight", loops="ignore"
            )
        else:
            graph = ig.Graph.Adjacency(matrix, mode=mode, loops="ignore")
    elif baseline == "networkx":
        import networkx as nx

        create_using = nx.DiGraph if directed else nx.Graph
        graph = nx.from_scipy_sparse_array(
            matrix, create_using=create_using, edge_attribute="weight"
        )
    else:
        os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"
        os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "FALSE"
        import easygraph as eg

        create_using = eg.DiGraph() if directed else eg.Graph()
        graph = eg.from_scipy_sparse_matrix(matrix, create_using=create_using)
        if baseline == "easygraph-cpp":
            graph = graph.cpp()
    elapsed = time.perf_counter() - started
    return metadata, graph, elapsed, directed, "scipy_sparse_matrix_entry_point"


def sources(metadata, count):
    recorded = metadata.get("benchmark_sources_zero_based")
    if isinstance(recorded, list) and len(recorded) >= count:
        return [int(value) for value in recorded[:count]]
    n = int(metadata["num_nodes"])
    return sorted({min(n - 1, index * n // max(1, count)) for index in range(count)})


def validate_vector(values, expected, *, unit_interval=False, sum_one=False):
    array = np.asarray(values, dtype=np.float64)
    valid = array.size == expected and bool(np.isfinite(array).all())
    if unit_interval and array.size:
        valid = valid and float(array.min()) >= -1e-10 and float(array.max()) <= 1 + 1e-9
    total = float(array.sum()) if array.size else 0.0
    if sum_one:
        valid = valid and math.isclose(total, 1.0, rel_tol=5e-5, abs_tol=5e-5)
    return valid, {
        "result_size": int(array.size),
        "sum": total,
        "minimum": float(array.min()) if array.size else 0.0,
        "maximum": float(array.max()) if array.size else 0.0,
    }


def igraph_call(graph, metadata, function, source_count, bc_count, close_count):
    n = int(metadata["num_nodes"])
    directed = graph.is_directed()
    mode = "out" if directed else "all"
    if function == "PageRank":
        result = graph.pagerank(damping=0.75, directed=directed)
        valid, detail = validate_vector(result, n, sum_one=True)
    elif function == "MST":
        result = graph.spanning_tree(weights="weight", return_tree=True)
        detail = {
            "edge_count": result.ecount(),
            "weight": float(sum(result.es["weight"])) if result.ecount() else 0.0,
        }
        valid = result.vcount() == n and result.ecount() <= max(0, n - 1)
    elif function == "LCC":
        result = graph.transitivity_local_undirected(mode="zero")
        valid, detail = validate_vector(result, n, unit_interval=True)
    elif function in {"WCC", "SCC"}:
        component_mode = "strong" if function == "SCC" and directed else "weak"
        result = graph.connected_components(mode=component_mode)
        covered = int(sum(len(component) for component in result))
        detail = {"component_count": len(result), "covered_nodes": covered}
        valid = covered == n
    elif function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        count = 1 if function == "Dijkstra" else source_count
        selected = sources(metadata, count)
        algorithm = "bellman_ford" if function == "BellmanFord" else "auto"
        weights = None if function == "BFS" else "weight"
        result = graph.distances(
            source=selected, mode=mode, weights=weights, algorithm=algorithm
        )
        shape = (len(result), len(result[0]) if result else 0)
        detail = {"sources": selected, "shape": list(shape)}
        valid = shape == (len(selected), n)
    elif function == "KCore":
        result = graph.coreness(mode="all")
        valid, detail = validate_vector(result, n)
        detail["maximum_core"] = int(max(result)) if result else 0
    elif function == "BC":
        selected = sources(metadata, bc_count)
        result = graph.betweenness(
            directed=directed, weights=None, sources=selected, targets=None
        )
        valid, detail = validate_vector(result, n)
        detail["sources"] = selected
    elif function == "Closeness":
        selected = sources(metadata, close_count)
        result = graph.closeness(vertices=selected, mode=mode, weights=None, normalized=True)
        valid, detail = validate_vector(result, len(selected))
        detail["sources"] = selected
    elif function == "Constraint":
        result = graph.constraint(vertices=None, weights=None)
        valid, detail = validate_vector(result, n)
    else:
        raise NotImplementedError(function)
    if not valid:
        raise RuntimeError(f"result invariant failed: {detail}")
    return result, detail


def networkx_call(graph, metadata, function, source_count, bc_count, close_count):
    import networkx as nx

    n = int(metadata["num_nodes"])
    directed = graph.is_directed()
    if function == "PageRank":
        result = nx.pagerank(graph, alpha=0.75, tol=1e-6, max_iter=200)
        valid, detail = validate_vector(list(result.values()), n, sum_one=True)
    elif function == "MST":
        result = nx.minimum_spanning_tree(graph, weight="weight")
        detail = {
            "edge_count": result.number_of_edges(),
            "weight": float(sum(data.get("weight", 1) for _, _, data in result.edges(data=True))),
        }
        valid = result.number_of_nodes() == n and result.number_of_edges() <= max(0, n - 1)
    elif function == "LCC":
        result = nx.clustering(graph)
        valid, detail = validate_vector(list(result.values()), n, unit_interval=True)
    elif function in {"WCC", "SCC"}:
        if function == "SCC" and directed:
            result = list(nx.strongly_connected_components(graph))
        elif directed:
            result = list(nx.weakly_connected_components(graph))
        else:
            result = list(nx.connected_components(graph))
        covered = int(sum(len(component) for component in result))
        detail = {"component_count": len(result), "covered_nodes": covered}
        valid = covered == n
    elif function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        count = 1 if function == "Dijkstra" else source_count
        selected = sources(metadata, count)
        result = []
        for source in selected:
            if function == "BFS":
                result.append(nx.single_source_shortest_path_length(graph, source))
            elif function == "BellmanFord":
                result.append(nx.single_source_bellman_ford_path_length(graph, source, weight="weight"))
            else:
                result.append(nx.single_source_dijkstra_path_length(graph, source, weight="weight"))
        detail = {"sources": selected, "reachable_counts": [len(item) for item in result]}
        valid = len(result) == len(selected) and all(len(item) > 0 for item in result)
    elif function == "KCore":
        result = nx.core_number(graph)
        valid, detail = validate_vector(list(result.values()), n)
    elif function == "BC":
        selected = sources(metadata, bc_count)
        result = nx.betweenness_centrality_subset(
            graph, sources=selected, targets=list(graph.nodes()), normalized=False, weight=None
        )
        valid, detail = validate_vector(list(result.values()), n)
        detail["sources"] = selected
    elif function == "Closeness":
        selected = sources(metadata, close_count)
        distance_graph = graph.reverse(copy=False) if directed else graph
        result = {
            node: nx.closeness_centrality(distance_graph, u=node, distance=None, wf_improved=True)
            for node in selected
        }
        valid, detail = validate_vector(list(result.values()), len(selected))
        detail["sources"] = selected
    elif function == "EffectiveSize":
        result = nx.effective_size(graph, weight=None)
        valid, detail = validate_vector(list(result.values()), n)
    elif function == "Constraint":
        result = nx.constraint(graph, weight=None)
        valid, detail = validate_vector(list(result.values()), n)
    else:
        raise NotImplementedError(function)
    if not valid:
        raise RuntimeError(f"result invariant failed: {detail}")
    return result, detail


def easygraph_call(graph, metadata, function, source_count, bc_count, close_count):
    import easygraph as eg

    n = int(metadata["num_nodes"])
    selected = sources(metadata, source_count)
    if function == "PageRank":
        result = eg.pagerank(graph, alpha=0.75, max_iter=200, tol=1e-6)
        values = list(result.values()) if isinstance(result, dict) else list(result)
        valid, detail = validate_vector(values, n, sum_one=True)
    elif function == "MST":
        result = eg.minimum_spanning_tree(graph, weight="weight")
        edges = list(result.edges)
        detail = {"edge_count": len(edges)}
        valid = result.number_of_nodes() == n and len(edges) <= max(0, n - 1)
    elif function == "LCC":
        result = eg.clustering(graph)
        valid, detail = validate_vector(list(result.values()), n, unit_interval=True)
    elif function in {"WCC", "SCC"}:
        if function == "SCC" and bool(metadata["directed"]):
            result = list(eg.strongly_connected_components(graph))
        else:
            result = list(eg.connected_components(graph))
        covered = int(sum(len(component) for component in result))
        detail = {"component_count": len(result), "covered_nodes": covered}
        valid = covered == n
    elif function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        count = 1 if function == "Dijkstra" else source_count
        selected = sources(metadata, count)
        if function == "BFS":
            result = eg.multi_source_bfs(graph, selected, target=None)
        elif function == "BellmanFord":
            result = eg.multi_source_bellman_ford(graph, selected, weight="weight", target=None)
        else:
            result = eg.multi_source_dijkstra(graph, selected, weight="weight", target=None)
        detail = {"sources": selected, "result_type": type(result).__name__}
        valid = result is not None
    elif function == "KCore":
        result = eg.k_core(graph)
        values = list(result.values()) if isinstance(result, dict) else list(result)
        if len(values) == n + 1:
            values = values[1:]
        valid, detail = validate_vector(values, n)
    elif function == "BC":
        selected = sources(metadata, bc_count)
        result = eg.betweenness_centrality(
            graph, weight=None, sources=selected, normalized=False, endpoints=False
        )
        values = list(result.values()) if isinstance(result, dict) else list(result)
        valid, detail = validate_vector(values, n)
        detail["sources"] = selected
    elif function == "Closeness":
        selected = sources(metadata, close_count)
        result = eg.closeness_centrality(graph, weight=None, sources=selected)
        values = list(result.values()) if isinstance(result, dict) else list(result)
        valid, detail = validate_vector(values, len(selected))
        detail["sources"] = selected
    elif function in {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}:
        call = {
            "EffectiveSize": eg.effective_size,
            "Efficiency": eg.efficiency,
            "Constraint": eg.constraint,
            "Hierarchy": eg.hierarchy,
        }[function]
        result = call(graph, weight=None)
        values = list(result.values()) if isinstance(result, dict) else list(result)
        valid, detail = validate_vector(values, n)
    else:
        raise NotImplementedError(function)
    if not valid:
        raise RuntimeError(f"result invariant failed: {detail}")
    return result, detail


def call_function(baseline, graph, metadata, function, args):
    if baseline == "igraph":
        return igraph_call(
            graph, metadata, function, args.source_count,
            args.bc_source_count, args.closeness_source_count,
        )
    if baseline == "networkx":
        return networkx_call(
            graph, metadata, function, args.source_count,
            args.bc_source_count, args.closeness_source_count,
        )
    return easygraph_call(
        graph, metadata, function, args.source_count,
        args.bc_source_count, args.closeness_source_count,
    )


def call_with_public_return_boundary(baseline, graph, metadata, function, args):
    """Time one public call and perform benchmark-only validation afterwards."""

    if baseline == "igraph" and function == "PageRank":
        started = time.perf_counter()
        result = graph.pagerank(
            damping=0.75,
            directed=graph.is_directed(),
        )
        elapsed = time.perf_counter() - started
        valid, detail = validate_vector(
            result, int(metadata["num_nodes"]), sum_one=True
        )
        if not valid:
            raise RuntimeError(f"result invariant failed: {detail}")
        return result, detail, elapsed

    started = time.perf_counter()
    result, detail = call_function(
        baseline, graph, metadata, function, args
    )
    return result, detail, time.perf_counter() - started


def timeout_handler(_signum, _frame):
    raise FunctionTimeout(
        f"high-level function call exceeded its {_CALL_TIMEOUT_SECONDS:g}-second budget"
    )


def worker(args):
    global _CALL_TIMEOUT_SECONDS
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    load_start = os.getloadavg()
    worker_wall_start = time.perf_counter()
    record = {
        "dataset": args.manifest.stem.replace(".weighted", ""),
        "baseline": args.baseline,
        "function": args.function or "__build__",
        "measurement": args.measurement,
    }
    try:
        set_memory_limit(args.worker_memory_limit_gb)
        function = args.function or "PageRank"
        metadata, graph, build_seconds, directed, construction_path = build_graph(
            args.baseline, args.manifest, function
        )
        record.update(
            {
                "dataset": metadata["name"],
                "num_nodes": int(metadata["num_nodes"]),
                "num_entries": int(metadata["num_entries"]),
                "directed": bool(metadata["directed"]),
                "effective_directed": directed,
                "graph_prepare_seconds": build_seconds,
                "graph_construction_path": construction_path,
            }
        )
        if args.measurement == "build":
            record.update({"status": "ok", "validation": "pass"})
        else:
            _CALL_TIMEOUT_SECONDS = float(args.timeout)
            signal.signal(signal.SIGALRM, timeout_handler)
            try:
                warmup_details = []
                for warmup_index in range(args.warmup_calls):
                    signal.setitimer(signal.ITIMER_REAL, args.timeout)
                    warmup_result, warmup_detail, warmup_seconds = (
                        call_with_public_return_boundary(
                            args.baseline,
                            graph,
                            metadata,
                            args.function,
                            args,
                        )
                    )
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    warmup_details.append(
                        {
                            "call_position": warmup_index + 1,
                            "seconds": warmup_seconds,
                            "validation": warmup_detail,
                        }
                    )
                    del warmup_result
                signal.setitimer(signal.ITIMER_REAL, args.timeout)
                result, detail, elapsed = call_with_public_return_boundary(
                    args.baseline,
                    graph,
                    metadata,
                    args.function,
                    args,
                )
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            record.update(
                {
                    "status": "ok",
                    "e2e_seconds": elapsed,
                    "kernel_seconds": elapsed,
                    "validation": "pass",
                    "validation_details": detail,
                    "warmup_calls": args.warmup_calls,
                    "warmup_details": warmup_details,
                    "measured_call_position": args.warmup_calls + 1,
                    "timer_boundary": (
                        "public invocation through complete public result "
                        "return; benchmark validation excluded for igraph PageRank"
                    ),
                    "validation_outside_timer": (
                        args.baseline == "igraph"
                        and args.function == "PageRank"
                    ),
                }
            )
            del result
        del graph
        gc.collect()
    except Exception as exc:
        lowered = str(exc).lower()
        kind = "execution_error"
        if isinstance(exc, (FunctionTimeout, TimeoutError)):
            kind = "timeout"
        elif isinstance(exc, MemoryError) or "out of memory" in lowered or "cannot allocate" in lowered:
            kind = "resource_limit"
        elif "result invariant failed" in lowered:
            kind = "semantic_mismatch"
        record.update(
            {
                "status": "failed",
                "failure_kind": kind,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        usage_end = resource.getrusage(resource.RUSAGE_SELF)
        load_end = os.getloadavg()
        record.update(
            {
                "host_rss_peak_mb_worker": usage_end.ru_maxrss / 1024.0,
                "worker_wall_seconds": time.perf_counter() - worker_wall_start,
                "process_user_cpu_seconds": usage_end.ru_utime - usage_start.ru_utime,
                "process_system_cpu_seconds": usage_end.ru_stime - usage_start.ru_stime,
                "minor_page_faults": usage_end.ru_minflt - usage_start.ru_minflt,
                "major_page_faults": usage_end.ru_majflt - usage_start.ru_majflt,
                "voluntary_context_switches": usage_end.ru_nvcsw - usage_start.ru_nvcsw,
                "involuntary_context_switches": usage_end.ru_nivcsw - usage_start.ru_nivcsw,
                "host_loadavg_1m_start": load_start[0],
                "host_loadavg_1m_end": load_end[0],
                "host_loadavg_5m_start": load_start[1],
                "host_loadavg_5m_end": load_end[1],
                "logical_cpu_count": os.cpu_count(),
            }
        )
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if record.get("status") == "ok" else 2


def run_worker(args, manifest, baseline, function, measurement):
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--manifest", str(manifest), "--baseline", baseline,
        "--measurement", measurement, "--timeout", str(args.timeout),
        "--source-count", str(args.source_count),
        "--bc-source-count", str(args.bc_source_count),
        "--closeness-source-count", str(args.closeness_source_count),
        "--worker-memory-limit-gb", str(args.worker_memory_limit_gb),
        "--warmup-calls", str(args.warmup_calls),
        "--physical-gpu", "-1",
    ]
    if function:
        command.extend(["--function", function])
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env.update(
        {
            "EASYGRAPH_ENABLE_GPU": "FALSE",
            "EASYGRAPH_GPU_STRICT_ERRORS": "FALSE",
            "MALLOC_ARENA_MAX": "2",
            "PYTHONPATH": str(EASYGRAPH_REPO)
            + (os.pathsep + existing_pythonpath if existing_pythonpath else ""),
        }
    )
    call_budget = (
        args.timeout * (args.warmup_calls + 1)
        if measurement != "build"
        else 0
    )
    budget = args.load_timeout + call_budget + 45.0
    monitor_result = None
    try:
        process = subprocess.Popen(
            command, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        monitor = None
        if measurement == "memory" or measurement == "build":
            monitor = ChildProcessMemoryMonitor(
                process.pid, physical_gpu=-1, interval_seconds=0.01
            ).start()
        try:
            stdout, stderr = process.communicate(timeout=budget)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise
        finally:
            if monitor is not None:
                monitor_result = monitor.stop()
        try:
            record = json.loads(stdout.strip().splitlines()[-1])
        except Exception:
            record = {
                "status": "failed", "failure_kind": "execution_error",
                "dataset": manifest.stem.replace(".weighted", ""),
                "baseline": baseline, "function": function or "__build__",
                "measurement": measurement, "error": stderr[-4000:],
                "returncode": process.returncode,
            }
        record["stderr_tail"] = stderr[-1200:]
    except subprocess.TimeoutExpired as exc:
        record = {
            "status": "failed",
            "failure_kind": "load_timeout" if measurement == "build" else "timeout",
            "dataset": manifest.stem.replace(".weighted", ""),
            "baseline": baseline,
            "function": function or "__build__",
            "measurement": measurement,
            "error": f"worker exceeded graph-load plus call budget: {exc}",
        }
    if monitor_result is not None:
        record.update(
            {
                "host_rss_peak_mb": monitor_result["rss_mb"],
                "gpu_process_peak_mb": monitor_result["gpu_proc_peak_mb"],
                "memory_monitor_rss_samples": monitor_result["monitor_rss_samples"],
                "memory_monitor_gpu_samples": monitor_result["monitor_gpu_proc_samples"],
                "memory_monitor_origin": monitor_result["memory_monitor_origin"],
                "memory_measurement_window": "isolated_worker_process_full_lifetime",
                "gpu_memory_applicability": "not_applicable_cpu_only_baseline",
            }
        )
    return record


def aggregate(records, baseline, dataset, function):
    successful = [row for row in records if row.get("status") == "ok"]
    failed = [row for row in records if row.get("status") != "ok"]
    base = {
        "dataset": dataset, "baseline": baseline, "function": function,
        "support_class": SUPPORT[baseline][function],
        "timing_process_records": records,
    }
    if failed:
        base.update(
            {
                "status": "failed",
                "failure_kind": "+".join(sorted({row.get("failure_kind", "execution_error") for row in failed})),
                "successful_timing_processes": len(successful),
                "failed_timing_processes": len(failed),
                "error": " | ".join(str(row.get("error", "")) for row in failed),
            }
        )
        return base
    values = [float(row["e2e_seconds"]) for row in successful]
    builds = [float(row["graph_prepare_seconds"]) for row in successful]
    base.update(
        {
            "status": "ok", "validation": "pass",
            "e2e_mean_seconds": statistics.mean(values),
            "e2e_stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
            "e2e_best_seconds": min(values), "timing_samples": len(values),
            "graph_prepare_mean_seconds": statistics.mean(builds),
            "graph_prepare_stdev_seconds": statistics.stdev(builds) if len(builds) > 1 else 0.0,
            "validation_details": [row.get("validation_details") for row in successful],
        }
    )
    return base


def flatten(record):
    return {
        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (dict, list)) else value
        for key, value in record.items()
    }


def write_outputs(out_dir, records, qualifications):
    import csv

    (out_dir / "cpu_large_matrix.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "build_qualifications.json").write_text(
        json.dumps(qualifications, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = [flatten(record) for record in records]
    fields = sorted({key for row in rows for key in row})
    with (out_dir / "cpu_large_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def driver(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = args.out_dir / "raw"
    raw.mkdir(exist_ok=True)
    records = []
    qualifications = []
    total = len(args.manifests) * len(args.baselines) * len(args.functions)
    progress = 0
    for plain_manifest in args.manifests:
        plain = json.loads(plain_manifest.read_text(encoding="utf-8"))
        dataset = plain["name"]
        for baseline in args.baselines:
            qualification_path = args.out_dir / f"{dataset}_{baseline}_build.json"
            if args.resume and qualification_path.is_file():
                qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
            else:
                qualification = run_worker(
                    args, plain_manifest.resolve(), baseline, None, "build"
                )
                qualification_path.write_text(
                    json.dumps(qualification, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            qualifications.append(qualification)
            build_ok = qualification.get("status") == "ok"
            for function in args.functions:
                progress += 1
                final_path = args.out_dir / f"{dataset}_{baseline}_{function}.json"
                if args.resume and final_path.is_file():
                    record = json.loads(final_path.read_text(encoding="utf-8"))
                    records.append(record)
                    print(f"[cpu-large {progress}/{total}] reused {dataset}/{baseline}/{function}")
                    continue
                manifest = weighted_manifest(plain_manifest, function)
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                allowed, kind, reason = applicability(metadata, baseline, function)
                if not allowed:
                    record = {
                        "status": "skipped", "failure_kind": kind,
                        "reason": reason, "dataset": dataset, "baseline": baseline,
                        "function": function, "support_class": SUPPORT[baseline][function],
                        "num_nodes": int(plain["num_nodes"]),
                        "num_entries": int(plain["num_entries"]),
                    }
                elif not build_ok:
                    record = {
                        "status": "failed",
                        "failure_kind": qualification.get("failure_kind", "load_failure"),
                        "reason": (
                            "baseline graph construction did not qualify; one isolated "
                            "qualification is shared across this dataset's function cells"
                        ),
                        "error": qualification.get("error", ""),
                        "dataset": dataset, "baseline": baseline, "function": function,
                        "support_class": SUPPORT[baseline][function],
                        "num_nodes": int(plain["num_nodes"]),
                        "num_entries": int(plain["num_entries"]),
                    }
                else:
                    timing_rows = []
                    for index in range(args.repeat):
                        row = run_worker(args, manifest.resolve(), baseline, function, "timing")
                        row["timing_process_index"] = index + 1
                        timing_rows.append(row)
                        (raw / f"{dataset}_{baseline}_{function}_timing_{index + 1}.json").write_text(
                            json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                        )
                        if row.get("status") != "ok":
                            break
                    record = aggregate(timing_rows, baseline, dataset, function)
                    if record.get("status") == "ok":
                        memory_rows = []
                        for index in range(args.memory_repeat):
                            row = run_worker(args, manifest.resolve(), baseline, function, "memory")
                            memory_rows.append(row)
                            (raw / f"{dataset}_{baseline}_{function}_memory_{index + 1}.json").write_text(
                                json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                            )
                        good = [row for row in memory_rows if row.get("status") == "ok"]
                        record["memory_status"] = "ok" if len(good) == args.memory_repeat else "incomplete"
                        record["memory_samples"] = len(good)
                        if good:
                            rss = [float(row.get("host_rss_peak_mb", 0.0)) for row in good]
                            record["host_rss_peak_mb_mean"] = statistics.mean(rss)
                            record["host_rss_peak_mb_stdev"] = statistics.stdev(rss) if len(rss) > 1 else 0.0
                        record["memory_process_records"] = memory_rows
                final_path.write_text(
                    json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                records.append(record)
                print(
                    f"[cpu-large {progress}/{total}] {dataset}/{baseline}/{function}: "
                    f"{record.get('status')} {record.get('failure_kind', '')}", flush=True
                )
    write_outputs(args.out_dir, records, qualifications)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifests", type=Path, nargs="+")
    parser.add_argument("--baseline", choices=BASELINES)
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--function", choices=FUNCTIONS)
    parser.add_argument("--functions", nargs="+", choices=FUNCTIONS, default=list(FUNCTIONS))
    parser.add_argument("--measurement", choices=("build", "timing", "memory"), default="timing")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-repeat", type=int, default=3)
    parser.add_argument("--source-count", type=int, default=8)
    parser.add_argument("--bc-source-count", type=int, default=16)
    parser.add_argument("--closeness-source-count", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--load-timeout", type=float, default=900.0)
    parser.add_argument("--worker-memory-limit-gb", type=float, default=128.0)
    parser.add_argument("--warmup-calls", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.worker:
        if args.manifest is None or args.baseline is None:
            parser.error("--worker requires --manifest and --baseline")
        if args.measurement != "build" and args.function is None:
            parser.error("timing/memory workers require --function")
    elif not args.manifests or args.out_dir is None:
        parser.error("driver requires --manifests and --out-dir")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(worker(parsed) if parsed.worker else driver(parsed))
