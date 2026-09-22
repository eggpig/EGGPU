#!/usr/bin/env python3
"""Run one isolated GraphScope baseline function on one prepared dataset.

The worker deliberately separates:

* session startup: runtime setup, outside the reported build metric;
* build: GraphScope-native graph loading and projection;
* algorithm: native GAE/FLASH application wall time;
* E2E: application calls plus host transfer and aligned result reconstruction.

No NetworkX fallback or runner-side graph algorithm is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np

import graphscope
from graphscope import flash
from graphscope.framework.app import AppAssets
from graphscope.framework.app import project_to_simple
from graphscope.framework.loader import Loader


SUPPORTED_FUNCTIONS = {
    "PageRank",
    "LCC",
    "WCC",
    "SCC",
    "BFS",
    "Dijkstra",
    "SSSP",
    "KCore",
    "BC",
    "Closeness",
}


@project_to_simple
def exact_closeness_application(graph, wf_improved=True):
    """Invoke the standard GAE app after GraphScope's native simple projection."""

    return AppAssets(algo="closeness_centrality", context="vertex_data")(
        graph, wf_improved
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--function", choices=sorted(SUPPORTED_FUNCTIONS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detail-output", type=Path)
    parser.add_argument("--pr-alpha", type=float, default=0.75)
    parser.add_argument("--pr-tol", type=float, default=1.0e-6)
    parser.add_argument("--pr-max-iter", type=int, default=200)
    parser.add_argument("--sssp-sources", type=int, default=8)
    parser.add_argument("--bc-sources", type=int, default=16)
    parser.add_argument("--closeness-sources", type=int, default=0)
    parser.add_argument("--validation-sample", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def pick_sources(
    n: int, k: int, preferred: list[int] | None = None
) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if preferred is not None:
        normalized = [int(source) for source in preferred]
        if len(normalized) < min(k, n):
            raise ValueError(
                "benchmark_sources_zero_based does not contain enough sources"
            )
        selected = normalized[: min(k, n)]
        if len(selected) != len(set(selected)):
            raise ValueError("benchmark source selection contains duplicates")
        if any(source < 0 or source >= n for source in selected):
            raise ValueError("benchmark source is outside the graph node domain")
        return selected
    if k >= n:
        return list(range(n))
    step = max(1, n // k)
    result = list(range(0, n, step))[:k]
    if len(result) < k:
        seen = set(result)
        candidate = n - 1
        while len(result) < k and candidate >= 0:
            if candidate not in seen:
                result.append(candidate)
                seen.add(candidate)
            candidate -= 1
    return result


def graph_plan(metadata: dict, function: str) -> tuple[Path, bool, bool, str]:
    """Return edge path, directed flag, weighted flag, and view description."""

    graph_type = metadata["graph_type"]
    directed_input = graph_type == "directed"
    weighted = function in {"Dijkstra", "SSSP"}

    if function == "PageRank":
        key = "edges" if directed_input else "bidirected"
        return required_path(metadata, key), True, False, key

    if function == "LCC":
        return required_path(metadata, "undirected"), False, False, "simple_undirected"

    if function == "WCC":
        if directed_input and metadata.get("undirected") is None:
            # WCC is defined on the underlying undirected relation. Loading a
            # directed edge list into an undirected GraphScope graph preserves
            # reachability even when reciprocal arcs occur.
            return required_path(metadata, "edges"), False, False, "weak_projection"
        key = "undirected" if metadata.get("undirected") else "edges"
        return required_path(metadata, key), False, False, "simple_undirected"

    if function == "SCC":
        if directed_input:
            return required_path(metadata, "edges"), True, False, "directed"
        return required_path(metadata, "undirected"), False, False, "simple_undirected"

    if function == "KCore":
        # The evaluated FLASH implementation traverses both incoming and
        # outgoing adjacency. Store each logical undirected edge once.
        return required_path(metadata, "undirected"), True, False, "single_orientation"

    if function == "Closeness" and directed_input:
        return required_path(metadata, "reverse"), True, False, "reversed_directed"

    if directed_input:
        return required_path(metadata, "edges"), True, weighted, "directed"
    return required_path(metadata, "undirected"), False, weighted, "simple_undirected"


def required_path(metadata: dict, key: str) -> Path:
    value = metadata.get(key)
    if not value:
        raise RuntimeError(f"required prepared graph view is unavailable: {key}")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"prepared graph view does not exist: {path}")
    return path


def build_graph(session, metadata: dict, function: str):
    edge_path, directed, weighted, view_name = graph_plan(metadata, function)
    graph = session.g(generate_eid=False, retain_oid=True, directed=directed)
    graph = graph.add_vertices(
        Loader(metadata["vertices"], delimiter=",", header_row=True),
        label="v",
        vid_field="id",
    )
    edge_properties = [("weight", "double")] if weighted else []
    graph = graph.add_edges(
        Loader(str(edge_path), delimiter=",", header_row=True),
        label="e",
        properties=edge_properties,
        src_label="v",
        dst_label="v",
        src_field="src",
        dst_field="dst",
    )
    projected_edges = {"e": ["weight"] if weighted else []}
    graph = graph.project(vertices={"v": []}, edges=projected_edges)
    return graph, view_name, str(edge_path)


def context_vector(context, n: int, *, dtype=np.float64) -> np.ndarray:
    frame = context.to_dataframe({"node": "v.id", "value": "r"})
    nodes = frame["node"].to_numpy(dtype=np.int64, copy=False)
    values = frame["value"].to_numpy(dtype=dtype, copy=False)
    result = np.zeros(n, dtype=dtype)
    result[nodes] = values
    return result


def context_distance(context, n: int) -> np.ndarray:
    values = context_vector(context, n)
    invalid = (~np.isfinite(values)) | (values < 0.0) | (np.abs(values) >= 1.0e30)
    values[invalid] = np.inf
    return values


def run_context(application, *args, **kwargs):
    started = time.perf_counter()
    context = application(*args, **kwargs)
    return context, time.perf_counter() - started


def project_once(graph, edge_property: str | None = None):
    """Project a property graph once before repeated native application calls."""

    started = time.perf_counter()
    projected = graph._project_to_simple(e_prop=edge_property)
    projected._base_graph = graph
    return projected, time.perf_counter() - started


def run_function(graph, metadata: dict, args: argparse.Namespace):
    function = args.function
    n = int(metadata["num_nodes"])
    graph_type = metadata["graph_type"]
    algorithm_seconds = 0.0
    started_e2e = time.perf_counter()
    sources: list[int] = []
    preferred_sources = metadata.get("benchmark_sources_zero_based")

    if function == "PageRank":
        context, elapsed = run_context(
            graphscope.pagerank_nx,
            graph,
            args.pr_alpha,
            args.pr_max_iter,
            args.pr_tol,
        )
        algorithm_seconds += elapsed
        result = context_vector(context, n)
        semantic = "exact_all_node"

    elif function == "LCC":
        context, elapsed = run_context(graphscope.clustering, graph)
        algorithm_seconds += elapsed
        result = context_vector(context, n)
        semantic = "exact_all_node_common_undirected_projection"

    elif function == "WCC":
        context, elapsed = run_context(graphscope.wcc_projected, graph)
        algorithm_seconds += elapsed
        result = context_vector(context, n, dtype=np.int64)
        semantic = "exact_component_labels"

    elif function == "SCC":
        application = flash.scc if graph_type == "directed" else graphscope.wcc_projected
        context, elapsed = run_context(application, graph)
        algorithm_seconds += elapsed
        result = context_vector(context, n, dtype=np.int64)
        semantic = "exact_component_labels"

    elif function == "BFS":
        sources = pick_sources(n, args.sssp_sources, preferred_sources)
        application = flash.bfs if graph_type == "directed" else flash.bfs_undirected
        application_graph, elapsed = project_once(graph)
        algorithm_seconds += elapsed
        rows = []
        for source in sources:
            context, elapsed = run_context(
                application, application_graph, source=source
            )
            algorithm_seconds += elapsed
            rows.append(context_distance(context, n))
        result = np.stack(rows, axis=0)
        semantic = "exact_selected_sources"

    elif function in {"Dijkstra", "SSSP"}:
        source_count = 1 if function == "Dijkstra" else args.sssp_sources
        sources = pick_sources(n, source_count, preferred_sources)
        application = flash.sssp if graph_type == "directed" else flash.sssp_undirected
        application_graph = graph
        if len(sources) > 1:
            application_graph, elapsed = project_once(graph, "weight")
            algorithm_seconds += elapsed
        rows = []
        for source in sources:
            context, elapsed = run_context(
                application, application_graph, source=source
            )
            algorithm_seconds += elapsed
            rows.append(context_distance(context, n))
        result = np.stack(rows, axis=0)
        semantic = "exact_selected_sources_nonnegative_weights"

    elif function == "KCore":
        context, elapsed = run_context(flash.kcore_decomposition, graph)
        algorithm_seconds += elapsed
        result = context_vector(context, n, dtype=np.int64)
        semantic = "exact_all_node_common_undirected_projection"

    elif function == "BC":
        sources = pick_sources(n, args.bc_sources, preferred_sources)
        application_graph, elapsed = project_once(graph)
        algorithm_seconds += elapsed
        result = np.zeros(n, dtype=np.float64)
        for source in sources:
            context, elapsed = run_context(
                flash.betweenness_centrality, application_graph, source=source
            )
            algorithm_seconds += elapsed
            contribution = context_vector(context, n)
            contribution[source] = 0.0
            result += contribution
        if graph_type == "undirected":
            result *= 0.5
        semantic = "exact_specified_source_subset_unnormalized"

    elif function == "Closeness":
        context, elapsed = run_context(exact_closeness_application, graph, True)
        algorithm_seconds += elapsed
        all_values = context_vector(context, n)
        sources = pick_sources(n, args.closeness_sources, preferred_sources)
        if sources:
            result = all_values[np.asarray(sources, dtype=np.int64)]
            semantic = "exact_selected_vertices"
        else:
            result = all_values
            semantic = "exact_all_node"

    else:  # pragma: no cover - argparse enforces this.
        raise RuntimeError(f"unhandled function: {function}")

    e2e_seconds = time.perf_counter() - started_e2e
    return result, algorithm_seconds, e2e_seconds, semantic, sources


def array_digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(tuple(values.shape)).encode())
    if np.issubdtype(values.dtype, np.floating):
        stable = np.where(np.isfinite(values), values, -1.0)
        digest.update(np.ascontiguousarray(stable).tobytes())
    else:
        digest.update(values.tobytes())
    return digest.hexdigest()[:16]


def component_partition_digest(labels: np.ndarray) -> tuple[str | None, int | None]:
    """Hash a partition independently of implementation-specific label IDs."""

    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    n = len(labels)
    if n == 0:
        return hashlib.sha256(b"").hexdigest(), 0
    minimum = int(labels.min())
    maximum = int(labels.max())
    if minimum < 0 or maximum >= n:
        return None, None

    sentinel = np.iinfo(np.int32).max
    representatives = np.full(n, sentinel, dtype=np.int32)
    chunk = 4_000_000
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        node_ids = np.arange(start, end, dtype=np.int32)
        np.minimum.at(representatives, labels[start:end], node_ids)

    digest = hashlib.sha256()
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        canonical = np.ascontiguousarray(representatives[labels[start:end]])
        digest.update(canonical.view(np.uint8))
    return digest.hexdigest(), int(np.count_nonzero(representatives != sentinel))


def result_summary(
    function: str,
    values: np.ndarray,
    sources: list[int],
    include_validation_sample: bool,
) -> dict:
    flat = np.asarray(values).reshape(-1)
    if np.issubdtype(flat.dtype, np.floating):
        finite = np.isfinite(flat) & (np.abs(flat) < 1.0e30)
        finite_values = flat[finite]
    else:
        finite = np.ones(flat.shape, dtype=bool)
        finite_values = flat
    summary = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "detail_sha": array_digest(values),
        "finite_count": int(finite.sum()),
        "sum": float(finite_values.sum(dtype=np.float64)) if finite_values.size else 0.0,
        "mean": float(finite_values.mean(dtype=np.float64)) if finite_values.size else 0.0,
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
        "sources": sources,
        "source_count": len(sources),
    }
    if function == "LCC":
        summary["vertices"] = int(values.size)
    if function in {"WCC", "SCC"}:
        partition_sha, component_count = component_partition_digest(values)
        summary["components"] = component_count
        summary["partition_sha256"] = partition_sha
    if function in {"BFS", "Dijkstra", "SSSP"}:
        summary["reachable"] = int(finite.sum())
        summary["checksum"] = (
            float(finite_values.sum(dtype=np.float64)) if finite_values.size else 0.0
        )
    if function in {"KCore", "BC", "Closeness"}:
        summary["nodes"] = int(values.size)
    if (
        include_validation_sample
        and function
        in {
            "PageRank",
            "LCC",
            "BFS",
            "Dijkstra",
            "SSSP",
            "KCore",
            "BC",
            "Closeness",
        }
        and values.size
    ):
        sample_count = min(4096, int(values.size))
        sample_indices = np.linspace(
            0, int(values.size) - 1, num=sample_count, dtype=np.int64
        )
        summary["validation_sample_indices"] = sample_indices.tolist()
        summary["validation_sample"] = np.asarray(
            values.reshape(-1)[sample_indices], dtype=np.float64
        ).tolist()
    return summary


def write_detail(path: Path | None, function: str, values: np.ndarray, sources: list[int]):
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = "vector"
    payload = {"values": values}
    if function in {"WCC", "SCC"}:
        kind = "cc_labels"
    elif function in {"BFS", "Dijkstra", "SSSP"}:
        kind = "sssp"
        payload["sources"] = np.asarray(sources, dtype=np.int64)
    elif function == "Closeness" and sources:
        kind = "source_vector"
        payload["sources"] = np.asarray(sources, dtype=np.int64)
    np.savez_compressed(path, kind=kind, **payload)
    return str(path)


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.manifest.read_text())
    payload = {
        "status": "failed",
        "function": args.function,
        "dataset": metadata.get("name", args.manifest.parent.name),
        "manifest": str(args.manifest),
        "graphscope_version": graphscope.__version__,
    }

    graphscope.set_option(show_log=False)
    graphscope.set_option(log_level="ERROR")
    session = None
    try:
        setup_started = time.perf_counter()
        session = graphscope.session(cluster_type="hosts", num_workers=1)
        session_setup_seconds = time.perf_counter() - setup_started

        build_started = time.perf_counter()
        graph, view_name, edge_path = build_graph(session, metadata, args.function)
        build_seconds = time.perf_counter() - build_started

        values, algorithm_seconds, e2e_seconds, semantic, sources = run_function(
            graph, metadata, args
        )
        summary = result_summary(
            args.function, values, sources, args.validation_sample
        )
        detail_path = write_detail(args.detail_output, args.function, values, sources)

        payload.update(
            {
                "status": "ok",
                "session_setup_seconds": session_setup_seconds,
                "build_seconds": build_seconds,
                "algorithm_seconds": algorithm_seconds,
                "e2e_seconds": e2e_seconds,
                "materialization_seconds": max(0.0, e2e_seconds - algorithm_seconds),
                "semantic": semantic,
                "graph_view": view_name,
                "edge_path": edge_path,
                "num_nodes": int(metadata["num_nodes"]),
                "result_summary": summary,
                "detail_path": detail_path,
            }
        )
        # Emit before session cleanup. This preserves a valid measurement if a
        # GraphScope service is slow to terminate and the parent enforces a
        # cleanup grace timeout.
        atomic_json(args.output, payload)
        print("RESULT_JSON " + json.dumps(payload, sort_keys=True), flush=True)
    except Exception as error:
        payload.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(args.output, payload)
        print("RESULT_JSON " + json.dumps(payload, sort_keys=True), flush=True)
        raise
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
