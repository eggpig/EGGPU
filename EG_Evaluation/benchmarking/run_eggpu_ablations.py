#!/usr/bin/env python3
"""Focused EGGPU ablations.

This script is intentionally separate from the full baseline runner.  The full
runner isolates libraries/functions for fairness; these ablations isolate
specific EGGPU design choices:

1. workflow: same EasyGraph object, multiple GPU functions, with warmup.
2. return: function-return latency versus an explicit traversal of the returned
   object, with the concrete return representation recorded per function.
3. layout: CSR versus COO preparation/copy and a representative COO PageRank.
"""

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from library_baselines import PeakMemoryMonitor
from library_baselines import build_easygraph
from library_baselines import deterministic_weighted_edges
from library_baselines import load_graph
from library_baselines import pick_sources
from library_baselines import sync_gpu
from easygraph_runtime_provenance import (
    collect_loaded_runtime_provenance,
    collect_relevant_environment,
    collect_runtime_repository_provenance,
    install_runtime_import_root,
)
from gpu_device_profile import collect_gpu_device_profile, collect_host_profile
from measurement_schema import describe_metric, schema_document, write_measurement_schema
from run_full_baselines import check_eggpu_child_gpu_idle, collect_source_snapshot


ROOT = Path(__file__).resolve().parents[1]
_COO_PAGERANK_KERNELS = None
MEASUREMENT_MODE = os.environ.get("EGGPU_MEASUREMENT_MODE", "combined").strip().lower()
RUNTIME_CONTRACT_VERSION = 1
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


def runtime_identity(provenance):
    snapshot = (provenance or {}).get("runtime_python_snapshot") or {}
    modules = (provenance or {}).get("modules") or {}
    return {
        "resolved_root": str((provenance or {}).get("resolved_root", "")),
        "native_sha256": str((provenance or {}).get("native_sha256", "")),
        "runtime_python_digest": str(snapshot.get("digest", "")),
        "python_origins": {
            name: str((modules.get(name) or {}).get("module_origin_resolved", ""))
            for name in ("easygraph", "cpp_easygraph")
        },
    }


def _require_complete_runtime_identity(identity, label):
    missing = [
        key
        for key in ("resolved_root", "native_sha256", "runtime_python_digest")
        if not identity.get(key)
    ]
    missing.extend(
        f"python_origins.{name}"
        for name, value in (identity.get("python_origins") or {}).items()
        if not value
    )
    if missing:
        raise RuntimeError(
            f"{label} is missing frozen runtime identity fields: {missing}"
        )


def runtime_subprocess_environment(easygraph_repo, environment=None):
    """Return a child environment with the frozen runtime first in PYTHONPATH."""

    source = os.environ if environment is None else environment
    env = dict(source)
    runtime_root = Path(easygraph_repo).expanduser().resolve(strict=True)
    retained = []
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            if Path(entry).expanduser().resolve() == runtime_root:
                continue
        except OSError:
            pass
        retained.append(entry)
    env["PYTHONPATH"] = os.pathsep.join([str(runtime_root), *retained])
    return env


def _assert_runtime_matches(expected, observed, label):
    expected_identity = runtime_identity(expected)
    observed_identity = runtime_identity(observed)
    _require_complete_runtime_identity(expected_identity, "requested runtime")
    _require_complete_runtime_identity(observed_identity, label)
    mismatches = {
        key: (expected_identity.get(key), observed_identity.get(key))
        for key in ("resolved_root", "native_sha256", "runtime_python_digest")
        if expected_identity.get(key) != observed_identity.get(key)
    }
    if expected_identity["python_origins"] != observed_identity["python_origins"]:
        mismatches["python_origins"] = (
            expected_identity["python_origins"],
            observed_identity["python_origins"],
        )
    if mismatches:
        raise RuntimeError(
            f"{label} does not match the requested frozen runtime: {mismatches}"
        )


def load_and_validate_runtime(args):
    """Pin imports, load EasyGraph, and reject any Python/native identity drift."""

    requested = collect_runtime_repository_provenance(args.easygraph_repo)
    requested_identity = runtime_identity(requested)
    _require_complete_runtime_identity(requested_identity, "requested runtime")
    if not args.expected_native_sha256 or not args.expected_runtime_python_digest:
        raise RuntimeError(
            "ablation worker requires coordinator-provided native and Python "
            "runtime digests"
        )
    if args.expected_native_sha256 != requested_identity["native_sha256"]:
        raise RuntimeError(
            "requested cpp_easygraph binary changed between coordinator and "
            f"worker: expected={args.expected_native_sha256} "
            f"observed={requested_identity['native_sha256']}"
        )
    if (
        args.expected_runtime_python_digest
        != requested_identity["runtime_python_digest"]
    ):
        raise RuntimeError(
            "requested EasyGraph Python runtime changed between coordinator "
            f"and worker: expected={args.expected_runtime_python_digest} "
            f"observed={requested_identity['runtime_python_digest']}"
        )

    runtime_root = install_runtime_import_root(args.easygraph_repo)
    pinned_environment = runtime_subprocess_environment(
        runtime_root, os.environ
    )
    os.environ["PYTHONPATH"] = pinned_environment["PYTHONPATH"]
    importlib.import_module("easygraph")
    importlib.import_module("cpp_easygraph")
    loaded = collect_loaded_runtime_provenance(args.easygraph_repo)
    loaded_with_snapshot = {
        **loaded,
        "runtime_python_snapshot": requested["runtime_python_snapshot"],
    }
    _assert_runtime_matches(requested, loaded_with_snapshot, "loaded runtime")
    for module_name in ("easygraph", "cpp_easygraph"):
        requested_artifact = requested["modules"][module_name]
        loaded_artifact = loaded["modules"][module_name]
        if requested_artifact.get("sha256") != loaded_artifact.get("sha256"):
            raise RuntimeError(
                f"loaded {module_name} artifact differs from --easygraph-repo: "
                f"expected={requested_artifact.get('sha256')} "
                f"observed={loaded_artifact.get('sha256')}"
            )
    return requested, loaded


def _resolved_edge_path(args):
    path = Path(args.edge_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def benchmark_contract(args):
    return {
        "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
        "experiment": args.experiment,
        "variant": args.variant,
        "edge_path": str(_resolved_edge_path(args)),
        "dataset_name": args.dataset_name,
        "graph_type": args.graph_type,
        "functions": args.functions,
        "workflow_order_id": args.workflow_order_id,
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "measurement_mode": args.measurement_mode,
        "gpu": int(args.gpu),
        "sssp_sources": int(args.sssp_sources),
        "bc_sources": int(args.bc_sources),
        "closeness_sources": int(args.closeness_sources),
        "layout_pr_iters": int(args.layout_pr_iters),
    }


def estimator_policy(args):
    return {
        "raw_sample_contract": (
            "retain every independent repetition; do not discard samples "
            "after observing latency"
        ),
        "descriptive_statistics": [
            "arithmetic_mean",
            "sample_standard_deviation",
            "minimum",
        ],
        "submission_estimator": "best_observed",
        "submission_estimator_definition": "minimum_of_five_independent_runs",
        "configured_repeat": int(args.repeat),
        "submission_repeat_contract_satisfied": int(args.repeat) == 5,
        "variance_requirement": (
            "report sample standard deviation over all five raw runs and "
            "audit unexpectedly high relative variance"
        ),
    }


def _resume_paths(args):
    if not args.out:
        return None, None
    output = Path(args.out)
    return output, output.with_suffix(".metadata.json")


def reusable_output(args, requested_runtime):
    """Return True only for a complete output with the same runtime and protocol.

    A pre-existing output from another binary, Python tree, dataset artifact, or
    benchmark contract is rejected instead of being silently mixed.
    """

    output, metadata_path = _resume_paths(args)
    if output is None:
        return False
    output_exists = output.is_file()
    metadata_exists = metadata_path.is_file()
    if not output_exists and not metadata_exists:
        return False
    if output_exists != metadata_exists:
        raise RuntimeError(
            "refusing incomplete ablation resume pair: "
            f"output={output_exists} metadata={metadata_exists}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"refusing unreadable ablation resume metadata: {metadata_path}"
        ) from exc
    recorded_runtime = metadata.get("runtime_provenance") or {}
    _assert_runtime_matches(
        requested_runtime, recorded_runtime, "resumed runtime"
    )
    loaded_runtime = metadata.get("loaded_runtime_provenance") or {}
    loaded_with_snapshot = {
        **loaded_runtime,
        "runtime_python_snapshot": (
            recorded_runtime.get("runtime_python_snapshot") or {}
        ),
    }
    _assert_runtime_matches(
        requested_runtime, loaded_with_snapshot, "resumed loaded runtime"
    )
    expected_contract = benchmark_contract(args)
    observed_contract = metadata.get("benchmark_contract") or {}
    if observed_contract != expected_contract:
        raise RuntimeError(
            "refusing ablation resume with a different benchmark contract: "
            f"expected={expected_contract} observed={observed_contract}"
        )
    expected_dataset = _dataset_artifact(args)
    observed_dataset = metadata.get("dataset_artifact") or {}
    if (
        observed_dataset.get("sha256") != expected_dataset.get("sha256")
        or observed_dataset.get("path") != expected_dataset.get("path")
    ):
        raise RuntimeError(
            "refusing ablation resume with a different dataset artifact: "
            f"expected={expected_dataset.get('sha256')} "
            f"observed={observed_dataset.get('sha256')}"
        )
    if metadata.get("run_status") != "complete":
        return False
    if output.stat().st_size <= 0:
        return False
    return True


def worker_command(args, requested_runtime):
    identity = runtime_identity(requested_runtime)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--easygraph-repo",
        str(Path(args.easygraph_repo).expanduser().resolve()),
        "--expected-native-sha256",
        identity["native_sha256"],
        "--expected-runtime-python-digest",
        identity["runtime_python_digest"],
        "--experiment",
        args.experiment,
        "--variant",
        args.variant,
        "--edge-path",
        str(args.edge_path),
        "--dataset-name",
        args.dataset_name,
        "--graph-type",
        args.graph_type,
        "--functions",
        args.functions,
        "--workflow-order-id",
        args.workflow_order_id,
        "--repeat",
        str(args.repeat),
        "--warmup",
        str(args.warmup),
        "--gpu",
        str(args.gpu),
        "--sssp-sources",
        str(args.sssp_sources),
        "--bc-sources",
        str(args.bc_sources),
        "--closeness-sources",
        str(args.closeness_sources),
        "--layout-pr-iters",
        str(args.layout_pr_iters),
        "--measurement-mode",
        args.measurement_mode,
    ]
    if args.out:
        command.extend(["--out", str(args.out)])
    return command


def configure_eggpu_env(args):
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
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
    # Ablations execute CUDA in this process, so the measured process becomes
    # visible in nvitop by itself.  A same-process auxiliary allocation changes
    # the CUDA context under measurement and is therefore explicitly forbidden.
    os.environ["EGGPU_GPU_VISIBILITY_MARKER"] = "FALSE"
    os.environ["EGGPU_GPU_VISIBILITY_MARKER_MB"] = "0"
    os.environ["EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB"] = "0"
    os.environ.pop("EGGPU_GPU_VISIBILITY_MARKER_OWNER_PID", None)
    os.environ.pop("EGGPU_GPU_VISIBILITY_MARKER_ALLOCATED_MB", None)
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
def require_idle_gpu(context):
    ok, note = check_eggpu_child_gpu_idle(os.environ)
    if not ok:
        raise SystemExit(f"{context}: {note}")


def kernel_time(key):
    from easygraph.utils import gpu_eggpu_backend as eggpu_backend

    value = eggpu_backend.get_last_kernel_time(key)
    if value is None:
        raise RuntimeError(f"missing CUDA-event kernel timing for {key}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"invalid CUDA-event kernel timing for {key}: {value}")
    return value


def reset_kernel(key):
    from easygraph.utils import gpu_eggpu_backend as eggpu_backend

    eggpu_backend.set_last_kernel_time(key, None)


def timed(callable_obj, sync_after=False):
    monitor = PeakMemoryMonitor().start() if MEASUREMENT_MODE in {"memory", "combined"} else None
    t0 = time.perf_counter()
    result = callable_obj()
    if sync_after:
        sync_gpu()
    elapsed = time.perf_counter() - t0
    memory = monitor.stop() if monitor is not None else {}
    return result, elapsed, memory


def row_common(args, experiment, function, metric, value, status="ok", notes="", extra=None):
    descriptor = describe_metric(metric, baseline="EGGPU")
    row = {
        "experiment": experiment,
        "variant": args.variant,
        "workflow_order_id": getattr(args, "workflow_order_id", "canonical"),
        "measurement_phase": MEASUREMENT_MODE,
        "dataset": args.dataset_name,
        "graph_type": args.graph_type,
        "function": function,
        "metric": metric,
        "value": "" if value is None else float(value),
        **descriptor,
        "status": status,
        "notes": notes,
        "graph_nodes": getattr(args, "graph_nodes", ""),
        "graph_nodes_including_isolates": getattr(
            args, "graph_nodes_including_isolates", ""
        ),
        "graph_edges": getattr(args, "graph_edges", ""),
        "directed_edge_rows": getattr(args, "directed_edge_rows", ""),
        "undirected_edge_rows": getattr(args, "undirected_edge_rows", ""),
    }
    if extra:
        row.update(extra)
    return row


def graph_views(args):
    views = load_graph(str(ROOT / args.edge_path))
    return views


def integer_env(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw or raw.upper() == "AUTO":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def make_graph_bundle(args, views):
    n_clean, directed_edges, undirected_edges = views["clean"]
    n_all, _, undirected_all = views["all_vertices"]
    directed = args.graph_type == "directed"
    base_edges = directed_edges if directed else undirected_edges
    args.graph_nodes = int(n_clean)
    args.graph_nodes_including_isolates = int(n_all)
    args.graph_edges = int(len(base_edges))
    args.directed_edge_rows = int(len(directed_edges))
    args.undirected_edge_rows = int(len(undirected_edges))
    weighted_base = deterministic_weighted_edges(n_clean, base_edges)
    weighted_undirected_clean = deterministic_weighted_edges(n_clean, undirected_edges)
    weighted_undirected_all = deterministic_weighted_edges(n_all, undirected_all)
    exact_max_nodes = integer_env("EGGPU_CLOSENESS_EXACT_MAX_NODES", 1_000_000)
    exact_max_work = integer_env("EGGPU_CLOSENESS_EXACT_MAX_WORK", 50_000_000_000)
    closeness_work = int(n_clean) * int(len(base_edges))
    closeness_guarded = (
        (exact_max_nodes > 0 and int(n_clean) > exact_max_nodes)
        or (exact_max_work > 0 and closeness_work > exact_max_work)
    )
    closeness_sources = (
        pick_sources(n_clean, args.closeness_sources) if closeness_guarded else None
    )

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
        "closeness_sources": closeness_sources,
        "closeness_semantic": (
            "sampled_target_exact" if closeness_sources is not None else "exact_all_node"
        ),
        "closeness_work_estimate": closeness_work,
    }


def prewarm_context(bundle):
    from easygraph.utils import gpu_eggpu_backend as eggpu_backend

    eggpu_backend._graph_context(bundle["graph"], prewarm_cpp=True)
    eggpu_backend._graph_context(bundle["undirected_graph"], prewarm_cpp=True)
    eggpu_backend._graph_context(bundle["mst_graph"], prewarm_cpp=True)


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
        source = bundle["sssp_sources"][:1]
        return "dijkstra", eg.multi_source_dijkstra(
            bundle["graph"], source, weight="weight", target=None
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
            sources=bundle["closeness_sources"],
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


def function_semantics(function, bundle):
    if function == "Dijkstra":
        return {"semantic": "single_source_weighted", "source_count": 1}
    if function in {"BFS", "BellmanFord", "SSSP"}:
        return {
            "semantic": "multi_source_unweighted" if function == "BFS" else "multi_source_weighted",
            "source_count": len(bundle["sssp_sources"]),
        }
    if function == "BC":
        return {"semantic": "specified_source_subset", "source_count": len(bundle["bc_sources"])}
    if function == "Closeness":
        sources = bundle["closeness_sources"]
        return {
            "semantic": bundle["closeness_semantic"],
            "source_count": bundle["n"] if sources is None else len(sources),
            "work_estimate": bundle["closeness_work_estimate"],
        }
    return {"semantic": "aligned_main_benchmark"}


def result_representation(result):
    class_name = type(result).__name__
    deferred_classes = {
        "_DenseValueDict": "on_demand_dense_mapping",
        "_DenseDistanceDict": "on_demand_distance_mapping",
        "_DenseMultiSourceSSSPDict": "on_demand_batched_distance_mapping",
    }
    if class_name in deferred_classes:
        return deferred_classes[class_name], True, class_name
    if isinstance(result, dict):
        return "materialized_mapping", False, class_name
    if isinstance(result, (list, tuple)):
        return "materialized_sequence", False, class_name
    if isinstance(result, set):
        return "materialized_set", False, class_name
    if isinstance(result, np.ndarray):
        return "materialized_array", False, class_name
    if hasattr(result, "edges") and hasattr(result, "adj"):
        return "materialized_graph_object", False, class_name
    return "other_python_object", False, class_name


SCALAR_VALUE_FUNCTIONS = {
    "PageRank",
    "LCC",
    "KCore",
    "BC",
    "Closeness",
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
}


def _numeric_values(function, result):
    if function in SCALAR_VALUE_FUNCTIONS:
        if hasattr(result, "items"):
            yield from (value for _, value in result.items())
        elif isinstance(result, np.ndarray):
            yield from np.asarray(result).reshape(-1)
        elif isinstance(result, (list, tuple)):
            yield from result
        return
    if function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        if hasattr(result, "items"):
            outer = (distances for _, distances in result.items())
        else:
            outer = result if isinstance(result, (list, tuple)) else ()
        for distances in outer:
            if hasattr(distances, "items"):
                yield from (value for _, value in distances.items())
            elif isinstance(distances, (list, tuple, np.ndarray)):
                yield from distances
        return
    if function == "MST" and hasattr(result, "edges"):
        for edge in result.edges:
            if len(edge) >= 3 and isinstance(edge[2], dict):
                yield edge[2].get("weight", 1.0)


def numeric_result_profile(function, result):
    profile = {
        "numeric_value_count": 0,
        "finite_value_count": 0,
        "nan_value_count": 0,
        "positive_inf_value_count": 0,
        "negative_inf_value_count": 0,
        "finite_checksum": 0.0,
    }
    for value in _numeric_values(function, result):
        try:
            numeric = float(value)
        except Exception:
            continue
        profile["numeric_value_count"] += 1
        if math.isnan(numeric):
            profile["nan_value_count"] += 1
        elif numeric == math.inf:
            profile["positive_inf_value_count"] += 1
        elif numeric == -math.inf:
            profile["negative_inf_value_count"] += 1
        elif math.isfinite(numeric):
            profile["finite_value_count"] += 1
            profile["finite_checksum"] += numeric
    return profile


def compare_result_containers(left, right, *, rel_tol=1.0e-8, abs_tol=1.0e-8):
    """Compare API return values recursively with IEEE non-finite semantics."""

    stats = {
        "equivalent": True,
        "value_pairs_compared": 0,
        "mismatch_count": 0,
        "missing_key_count": 0,
        "nan_pair_count": 0,
        "infinity_pair_count": 0,
        "max_finite_abs_diff": 0.0,
    }

    def mismatch(count=1):
        stats["mismatch_count"] += int(count)
        stats["equivalent"] = False

    def walk(lhs, rhs):
        lhs_mapping = hasattr(lhs, "items")
        rhs_mapping = hasattr(rhs, "items")
        if lhs_mapping or rhs_mapping:
            if not (lhs_mapping and rhs_mapping):
                mismatch()
                return
            lhs_map = dict(lhs.items())
            rhs_map = dict(rhs.items())
            lhs_keys = set(lhs_map)
            rhs_keys = set(rhs_map)
            missing = len(lhs_keys ^ rhs_keys)
            if missing:
                stats["missing_key_count"] += missing
                mismatch(missing)
            for key in lhs_keys & rhs_keys:
                walk(lhs_map[key], rhs_map[key])
            return

        if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray):
            lhs_values = np.asarray(lhs).reshape(-1)
            rhs_values = np.asarray(rhs).reshape(-1)
            if lhs_values.shape != rhs_values.shape:
                mismatch(abs(lhs_values.size - rhs_values.size) or 1)
                return
            for lhs_value, rhs_value in zip(lhs_values, rhs_values):
                walk(lhs_value, rhs_value)
            return

        if isinstance(lhs, (list, tuple)) or isinstance(rhs, (list, tuple)):
            if not (isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple))):
                mismatch()
                return
            if len(lhs) != len(rhs):
                mismatch(abs(len(lhs) - len(rhs)) or 1)
            for lhs_value, rhs_value in zip(lhs, rhs):
                walk(lhs_value, rhs_value)
            return

        stats["value_pairs_compared"] += 1
        try:
            lhs_number = float(lhs)
            rhs_number = float(rhs)
        except Exception:
            try:
                equal = bool(lhs == rhs)
            except Exception:
                equal = False
            if not equal:
                mismatch()
            return

        if math.isnan(lhs_number) or math.isnan(rhs_number):
            if math.isnan(lhs_number) and math.isnan(rhs_number):
                stats["nan_pair_count"] += 1
            else:
                mismatch()
            return
        if math.isinf(lhs_number) or math.isinf(rhs_number):
            if lhs_number == rhs_number:
                stats["infinity_pair_count"] += 1
            else:
                mismatch()
            return
        difference = abs(lhs_number - rhs_number)
        stats["max_finite_abs_diff"] = max(
            stats["max_finite_abs_diff"], difference
        )
        if not math.isclose(
            lhs_number, rhs_number, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            mismatch()

    walk(left, right)
    return stats


def materialize_result(function, result):
    t0 = time.perf_counter()
    checksum = 0.0
    count = 0
    if function in SCALAR_VALUE_FUNCTIONS:
        if hasattr(result, "items"):
            for _, v in result.items():
                try:
                    numeric = float(v)
                    count += 1
                    if math.isfinite(numeric):
                        checksum += numeric
                except Exception:
                    pass
        elif isinstance(result, (list, tuple)):
            for v in result:
                try:
                    numeric = float(v)
                    count += 1
                    if math.isfinite(numeric):
                        checksum += numeric
                except Exception:
                    pass
        elif isinstance(result, np.ndarray):
            values = np.asarray(result).reshape(-1)
            finite = values[np.isfinite(values)]
            count = int(values.size)
            checksum = float(finite.astype(np.float64, copy=False).sum())
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


def standard_python_container(function, result):
    """Force an API-equivalent built-in Python container for return A/B tests."""

    if function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        return {
            source: dict(distances.items())
            for source, distances in result.items()
        }
    if hasattr(result, "items"):
        return dict(result.items())
    return result


def run_workflow(args):
    require_idle_gpu("workflow start")
    configure_eggpu_env(args)
    views = graph_views(args)
    bundle = make_graph_bundle(args, views)
    functions = parse_functions(args.functions, DEFAULT_FUNCTIONS)
    prewarm_context(bundle)
    for _ in range(max(0, args.warmup)):
        for fn in functions:
            call_function(fn, bundle)
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
                require_idle_gpu(f"workflow {args.dataset_name}/{fn}/repeat={rep}")
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
                kernel = kernel_time(kernel_key)
                semantic = function_semantics(fn, bundle)
                row_extra = {"repeat": rep, "result_len": result_size(value), **semantic}
                rows.append(row_common(args, "workflow", fn, "e2e", e2e, extra=row_extra))
                rows.append(row_common(args, "workflow", fn, "kernel", kernel, extra=row_extra))
                for mkey, mval in mem.items():
                    if mval is not None and mkey != "gpu_index":
                        rows.append(row_common(args, "workflow", fn, f"memory_{mkey}", mval, extra={"repeat": rep, **semantic}))
            except Exception as exc:
                rows.append(row_common(args, "workflow", fn, "e2e", None, status="failed", notes=f"{type(exc).__name__}: {exc}", extra={"repeat": rep}))
    return rows


def run_return(args):
    require_idle_gpu("return start")
    configure_eggpu_env(args)
    views = graph_views(args)
    bundle = make_graph_bundle(args, views)
    functions = parse_functions(args.functions, RETURN_FUNCTIONS)
    prewarm_context(bundle)
    on_demand_by_function = {}
    for _ in range(max(0, args.warmup)):
        for fn in functions:
            _, warm_result = call_function(fn, bundle)
            _, on_demand, _ = result_representation(warm_result)
            on_demand_by_function[fn] = on_demand
            if on_demand:
                standard_python_container(fn, warm_result)
    sync_gpu()

    rows = []
    for rep in range(args.repeat):
        for fn in functions:
            try:
                require_idle_gpu(f"return {args.dataset_name}/{fn}/repeat={rep}")
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
                def run_optimized():
                    if reset_key:
                        reset_kernel(reset_key)
                    result_pair, call_seconds, memory = timed(
                        lambda fn=fn: call_function(fn, bundle), sync_after=False
                    )
                    kernel_key, value = result_pair
                    return value, call_seconds, kernel_time(kernel_key), memory

                def run_standard():
                    if reset_key:
                        reset_kernel(reset_key)

                    def call_and_convert():
                        kernel_key, value = call_function(fn, bundle)
                        return kernel_key, standard_python_container(fn, value)

                    result_pair, call_seconds, memory = timed(
                        call_and_convert, sync_after=False
                    )
                    kernel_key, value = result_pair
                    return value, call_seconds, kernel_time(kernel_key), memory

                standard_measurement = None
                # Alternate pair order across repeats to avoid assigning all
                # allocator/cache drift to one side of the comparison.
                if on_demand_by_function.get(fn, False) and rep % 2 == 1:
                    standard_measurement = run_standard()
                    result, call_s, kernel, mem = run_optimized()
                    pair_order = "standard_then_deferred"
                else:
                    result, call_s, kernel, mem = run_optimized()
                    pair_order = "deferred_then_standard"
                    if on_demand_by_function.get(fn, False):
                        standard_measurement = run_standard()

                mat_s, mat_count, checksum = materialize_result(fn, result)
                numeric_profile = numeric_result_profile(fn, result)
                representation, on_demand, class_name = result_representation(result)
                semantic = function_semantics(fn, bundle)
                row_extra = {
                    "repeat": rep,
                    "materialized_count": mat_count,
                    "checksum": checksum,
                    "return_representation": representation,
                    "return_class": class_name,
                    "on_demand_materialization": str(on_demand).lower(),
                    "materialization_claim_valid": str(on_demand).lower(),
                    "deferred_python_pairs": mat_count if on_demand else 0,
                    "paired_order": pair_order,
                    **numeric_profile,
                    **semantic,
                }
                rows.append(row_common(args, "return", fn, "call_return_seconds", call_s, extra=row_extra))
                rows.append(row_common(args, "return", fn, "kernel", kernel, extra=row_extra))
                rows.append(row_common(args, "return", fn, "forced_result_traversal_seconds", mat_s, extra=row_extra))
                rows.append(
                    row_common(
                        args,
                        "return",
                        fn,
                        "call_plus_forced_traversal_seconds",
                        call_s + mat_s,
                        notes=(
                            "function call followed by explicit traversal of the returned object; "
                            "this is an on-demand-materialization comparison only when materialization_claim_valid=true"
                        ),
                        extra=row_extra,
                    )
                )
                if standard_measurement is not None:
                    standard_result, standard_s, standard_kernel, standard_mem = standard_measurement
                    _, standard_count, standard_checksum = materialize_result(
                        fn, standard_result
                    )
                    standard_profile = numeric_result_profile(fn, standard_result)
                    equivalence = compare_result_containers(result, standard_result)
                    equivalent = bool(
                        standard_count == mat_count and equivalence["equivalent"]
                    )
                    strict_extra = {
                        **row_extra,
                        "standard_container_class": type(standard_result).__name__,
                        "standard_materialized_count": standard_count,
                        "standard_checksum": standard_checksum,
                        **{
                            f"standard_{key}": value
                            for key, value in standard_profile.items()
                        },
                        **{
                            f"equivalence_{key}": value
                            for key, value in equivalence.items()
                            if key != "equivalent"
                        },
                        "return_equivalent": str(equivalent).lower(),
                    }
                    strict_status = "ok" if equivalent else "failed"
                    strict_note = (
                        "same warmed graph and function; deferred result adapter is replaced "
                        "by an API-equivalent built-in Python container within the measured call"
                    )
                    rows.append(
                        row_common(
                            args,
                            "return",
                            fn,
                            "standard_container_return_seconds",
                            standard_s if equivalent else None,
                            status=strict_status,
                            notes=strict_note,
                            extra=strict_extra,
                        )
                    )
                    rows.append(
                        row_common(
                            args,
                            "return",
                            fn,
                            "standard_container_kernel_seconds",
                            standard_kernel if equivalent else None,
                            status=strict_status,
                            notes=strict_note,
                            extra=strict_extra,
                        )
                    )
                    if equivalent and call_s > 0:
                        rows.append(
                            row_common(
                                args,
                                "return",
                                fn,
                                "standard_over_deferred_return_ratio",
                                standard_s / call_s,
                                notes=strict_note,
                                extra=strict_extra,
                            )
                        )
                for mkey, mval in mem.items():
                    if mval is not None and mkey != "gpu_index":
                        rows.append(row_common(args, "return", fn, f"memory_{mkey}", mval, extra=row_extra))
                if standard_measurement is not None:
                    for mkey, mval in standard_measurement[3].items():
                        if mval is not None and mkey != "gpu_index":
                            rows.append(
                                row_common(
                                    args,
                                    "return",
                                    fn,
                                    f"memory_standard_{mkey}",
                                    mval,
                                    extra=row_extra,
                                )
                            )
            except Exception as exc:
                rows.append(row_common(args, "return", fn, "call_return_seconds", None, status="failed", notes=f"{type(exc).__name__}: {exc}", extra={"repeat": rep, **function_semantics(fn, bundle)}))
    return rows


def build_layout_arrays(args, views):
    directed = args.graph_type == "directed"
    n, directed_edges, undirected_edges = views["clean"]
    edges = directed_edges if directed else undirected_edges
    src_base = edges["src"].to_numpy(dtype=np.int32, copy=True)
    dst_base = edges["dst"].to_numpy(dtype=np.int32, copy=True)
    if directed:
        src, dst = src_base, dst_base
    else:
        # EasyGraph's undirected CSR stores both adjacency directions.  The COO
        # alternative must carry the same edge slots or PageRank/degree values
        # are not semantically comparable.
        src = np.concatenate((src_base, dst_base))
        dst = np.concatenate((dst_base, src_base))
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
    require_idle_gpu("layout start")
    configure_eggpu_env(args)
    views = graph_views(args)
    n_clean, directed_edges, undirected_edges = views["clean"]
    n_all = views["all_vertices"][0]
    args.graph_nodes = int(n_clean)
    args.graph_nodes_including_isolates = int(n_all)
    args.graph_edges = int(
        len(directed_edges if args.graph_type == "directed" else undirected_edges)
    )
    args.directed_edge_rows = int(len(directed_edges))
    args.undirected_edge_rows = int(len(undirected_edges))
    rows = []

    # Layout is an EGGPU ablation as well: warm both GPU representations the
    # configured number of times, then discard those rows.  This excludes CUDA
    # context/JIT/allocator startup consistently from the measured repeats.
    if args.warmup > 0:
        n_warm, src_warm, dst_warm = build_layout_arrays(args, views)
        rowptr_warm, _ = csr_from_coo(n_warm, src_warm, dst_warm)
        _ = np.bincount(src_warm, minlength=n_warm)
        _ = rowptr_warm[1:] - rowptr_warm[:-1]
        for warmup_index in range(args.warmup):
            warmup_rows = []
            warmup_rows.extend(
                run_optional_coo_pagerank(
                    args, n_warm, src_warm, dst_warm, -(warmup_index + 1)
                )
            )
            warmup_rows.extend(
                run_csr_pagerank(
                    args, n_warm, src_warm, dst_warm, -(warmup_index + 1)
                )
            )
            failures = [row for row in warmup_rows if row.get("status") == "failed"]
            if failures:
                details = "; ".join(
                    f"{row.get('function')}/{row.get('metric')}: {row.get('notes')}"
                    for row in failures
                )
                raise RuntimeError(f"layout warmup failed: {details}")
        sync_gpu()

    for rep in range(args.repeat):
        require_idle_gpu(f"layout {args.dataset_name}/repeat={rep}")
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

        coo_pr_rows = run_optional_coo_pagerank(args, n, src, dst, rep)
        csr_pr_rows = run_csr_pagerank(args, n, src, dst, rep)
        rows.extend(coo_pr_rows)
        rows.extend(csr_pr_rows)

        coo_kernel = next(
            (
                row
                for row in coo_pr_rows
                if row.get("metric") == "gpu_kernel_seconds"
                and row.get("status") == "ok"
            ),
            None,
        )
        csr_kernel = next(
            (
                row
                for row in csr_pr_rows
                if row.get("metric") == "gpu_kernel_seconds"
                and row.get("status") == "ok"
            ),
            None,
        )
        if coo_kernel is not None and csr_kernel is not None:
            relative_errors = []
            for field in ("checksum", "rank_l2", "weighted_checksum"):
                left = float(coo_kernel[field])
                right = float(csr_kernel[field])
                relative_errors.append(abs(left - right) / max(abs(left), abs(right), 1e-15))
            max_relative_error = max(relative_errors)
            validation_status = "ok" if max_relative_error <= 1e-7 else "failed"
            rows.append(
                row_common(
                    args,
                    "layout",
                    "COO-vs-CSR-PageRank",
                    "layout_validation_relative_error",
                    max_relative_error,
                    status=validation_status,
                    notes="fixed-iteration PageRank equivalence over sum, L2 norm, and weighted checksum",
                    extra={"repeat": rep, "tolerance": 1e-7},
                )
            )
    return rows


def _coo_pagerank_kernels(cp):
    """Build the explicit COO PageRank kernels once per process.

    CuPy's generic ``add.at`` path instantiates CCCL templates at runtime and
    can pick up an NVRTC/header combination unrelated to the EGGPU build.  The
    ablation needs a stable representation comparison, so use the direct COO
    atomic-scatter kernels that the experiment intends to measure.
    """

    global _COO_PAGERANK_KERNELS
    if _COO_PAGERANK_KERNELS is not None:
        return _COO_PAGERANK_KERNELS
    code = r"""
    extern "C" __global__ void init_rank(double* rank, int n, double value) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) rank[i] = value;
    }
    extern "C" __global__ void scatter_edges(
        const int* src, const int* dst, const int* outdeg,
        const double* rank, double* next, int m) {
        int e = blockDim.x * blockIdx.x + threadIdx.x;
        if (e < m) {
            int s = src[e];
            int degree = outdeg[s];
            if (degree > 0) atomicAdd(next + dst[e], rank[s] / (double)degree);
        }
    }
    extern "C" __global__ void sum_dangling(
        const double* rank, const int* outdeg, double* total, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n && outdeg[i] == 0) atomicAdd(total, rank[i]);
    }
    extern "C" __global__ void update_rank(
        double* rank, const double* next, const double* dangling,
        int n, double alpha) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) {
            double inv_n = 1.0 / (double)n;
            rank[i] = (1.0 - alpha) * inv_n
                    + alpha * (next[i] + dangling[0] * inv_n);
        }
    }
    """
    module = cp.RawModule(
        code=code,
        options=("--std=c++11",),
        name_expressions=("init_rank", "scatter_edges", "sum_dangling", "update_rank"),
    )
    _COO_PAGERANK_KERNELS = tuple(
        module.get_function(name)
        for name in ("init_rank", "scatter_edges", "sum_dangling", "update_rank")
    )
    return _COO_PAGERANK_KERNELS


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
        init_rank, scatter_edges, sum_dangling, update_rank = _coo_pagerank_kernels(cp)
        outdeg_host = np.bincount(src, minlength=int(n)).astype(np.int32, copy=False)
        sync_gpu()
        t_copy0 = time.perf_counter()
        d_src = cp.asarray(src)
        d_dst = cp.asarray(dst)
        d_outdeg = cp.asarray(outdeg_host)
        cp.cuda.runtime.deviceSynchronize()
        copy_s = time.perf_counter() - t_copy0
        rank = cp.empty(int(n), dtype=cp.float64)
        nxt = cp.empty(int(n), dtype=cp.float64)
        dangling = cp.empty(1, dtype=cp.float64)
        block = 256
        node_grid = ((int(n) + block - 1) // block,)
        edge_grid = ((int(len(src)) + block - 1) // block,)
        stream = cp.cuda.get_current_stream()
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        init_rank(node_grid, (block,), (rank, int(n), 1.0 / max(1, int(n))))
        for _ in range(iters):
            cp.cuda.runtime.memsetAsync(nxt.data.ptr, 0, nxt.nbytes, stream.ptr)
            cp.cuda.runtime.memsetAsync(dangling.data.ptr, 0, dangling.nbytes, stream.ptr)
            scatter_edges(
                edge_grid,
                (block,),
                (d_src, d_dst, d_outdeg, rank, nxt, int(len(src))),
            )
            sum_dangling(node_grid, (block,), (rank, d_outdeg, dangling, int(n)))
            update_rank(node_grid, (block,), (rank, nxt, dangling, int(n), alpha))
        stop.record()
        stop.synchronize()
        kernel_s = float(cp.cuda.get_elapsed_time(start, stop)) / 1000.0
        rank_host = cp.asnumpy(rank).astype(np.float64, copy=False)
        checksum = float(rank_host.sum(dtype=np.float64))
        rank_l2 = float(np.linalg.norm(rank_host))
        weighted_checksum = float(
            np.dot(rank_host, (np.arange(int(n), dtype=np.float64) % 1009.0) + 1.0)
        )
        rows.append(row_common(args, "layout", "COO-PageRank", "h2d_seconds", copy_s, notes="COO src,dst,outdegree copy", extra={"repeat": rep, "nodes": int(n), "edges": int(len(src))}))
        rows.append(row_common(args, "layout", "COO-PageRank", "gpu_kernel_seconds", kernel_s, notes=f"explicit CUDA COO atomic-scatter PageRank, fixed {iters} iterations", extra={"repeat": rep, "nodes": int(n), "edges": int(len(src)), "checksum": checksum, "rank_l2": rank_l2, "weighted_checksum": weighted_checksum}))
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
            from easygraph.utils import gpu_eggpu_backend as eggpu_backend

            eggpu_backend._graph_context(graph, prewarm_cpp=True)
            eggpu_backend.set_last_kernel_time("pagerank", None)
        except Exception:
            pass
        # Fixed-iteration setting makes the comparison closer to the COO
        # scatter microbenchmark.  The algorithms are not claimed to be
        # identical; this is a layout-oriented representative primitive.
        result, e2e_s, mem = timed(
            lambda: eg.pagerank(
                graph,
                alpha=0.75,
                max_iter=max(1, int(args.layout_pr_iters)),
                tol=0.0,
                weight=None,
            ),
            sync_after=False,
        )
        k = kernel_time("pagerank")
        rank_host = np.full(int(n), np.nan, dtype=np.float64)
        for node, value in result.items():
            rank_host[int(node)] = float(value)
        if not np.isfinite(rank_host).all():
            raise RuntimeError("CSR PageRank result does not contain one finite value per node")
        checksum = float(rank_host.sum(dtype=np.float64))
        rank_l2 = float(np.linalg.norm(rank_host))
        weighted_checksum = float(
            np.dot(rank_host, (np.arange(int(n), dtype=np.float64) % 1009.0) + 1.0)
        )
        rows.append(row_common(args, "layout", "CSR-PageRank", "build_graph_seconds", build_s, notes="EasyGraph graph build for CSR PageRank representative", extra={"repeat": rep, "nodes": int(n), "edges": int(len(src))}))
        correctness = {"checksum": checksum, "rank_l2": rank_l2, "weighted_checksum": weighted_checksum}
        rows.append(row_common(args, "layout", "CSR-PageRank", "e2e_seconds", e2e_s, notes=f"EGGPU CSR PageRank, max_iter={args.layout_pr_iters}, tol=0", extra={"repeat": rep, "nodes": int(n), "edges": int(len(src)), **correctness}))
        rows.append(row_common(args, "layout", "CSR-PageRank", "gpu_kernel_seconds", k, notes=f"EGGPU CSR PageRank kernel, max_iter={args.layout_pr_iters}, tol=0", extra={"repeat": rep, "nodes": int(n), "edges": int(len(src)), **correctness}))
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    print(f"Wrote {path}", flush=True)


def _dataset_artifact(args):
    path = Path(args.edge_path)
    if not path.is_absolute():
        path = ROOT / path
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": int(stat.st_size),
        "graph_nodes": getattr(args, "graph_nodes", None),
        "graph_nodes_including_isolates": getattr(
            args, "graph_nodes_including_isolates", None
        ),
        "graph_edges": getattr(args, "graph_edges", None),
        "directed_edge_rows": getattr(args, "directed_edge_rows", None),
        "undirected_edge_rows": getattr(args, "undirected_edge_rows", None),
        "graph_type": args.graph_type,
    }


def _source_state(cache_dir):
    """Use an executed-source digest; repository metadata is intentionally irrelevant."""

    cache_path = Path(cache_dir) / "ablation_source_snapshot.json"
    try:
        cached = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        cached = None
    if (
        isinstance(cached, dict)
        and cached.get("algorithm") == "sha256"
        and len(str(cached.get("digest", ""))) == 64
        and int(cached.get("file_count", 0) or 0) > 0
    ):
        return cached

    snapshot = {
        "provenance": "content-addressed-executed-source-tree",
        "git_required": False,
        **collect_source_snapshot(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, cache_path)
    return snapshot


def write_metadata(
    args,
    out_path,
    *,
    run_started_at,
    run_completed_at,
    experiment_wall_seconds,
    gpu_device_profile,
    host_profile,
    requested_runtime,
    loaded_runtime,
    run_status,
    failure_row_count,
):
    if not out_path:
        return
    path = Path(out_path).with_suffix(".metadata.json")
    schema_path = write_measurement_schema(path.parent / "measurement_schema.json")
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "EGGPU_MONITOR_GPU_INDEX",
        "EGGPU_CUDA_ROOT",
        "EGGPU_GPU_VISIBILITY_MARKER",
        "EGGPU_GPU_VISIBILITY_MARKER_MB",
        "EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB",
        "EGGPU_GPU_VISIBILITY_MARKER_OWNER_PID",
        "EGGPU_GPU_VISIBILITY_MARKER_ALLOCATED_MB",
        "CUDA_PATH",
        "CONDA_PREFIX",
        "EASYGRAPH_ENABLE_GPU",
        "EASYGRAPH_GPU_STRICT_ERRORS",
        "EASYGRAPH_GPU_ADAPTIVE_POLICY",
        "EASYGRAPH_GPU_COMPONENT_DENSE_RETURN",
        "EASYGRAPH_GPU_SCC_ACTIVE_TRIM_MAX_ITERS",
        "EASYGRAPH_GPU_SCC_HOST_ENABLE",
        "EASYGRAPH_GPU_KCORE_HOST_ENABLE",
        "EASYGRAPH_GPU_SSSP_HOST_ENABLE",
        "EASYGRAPH_GPU_BC_WARP_SIZE",
        "EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION",
    ]
    identity = runtime_identity(requested_runtime)
    metadata = {
        "schema_version": 4,
        "created_at": run_started_at,
        "run_started_at": run_started_at,
        "run_completed_at": run_completed_at,
        "run_status": run_status,
        "failure_row_count": int(failure_row_count),
        "experiment_wall_seconds": float(experiment_wall_seconds),
        "argv": list(sys.argv),
        "easygraph_repo": identity["resolved_root"],
        "runtime_provenance": requested_runtime,
        "loaded_runtime_provenance": loaded_runtime,
        "runtime_identity": identity,
        "native_binary_sha256": identity["native_sha256"],
        "runtime_python_digest": identity["runtime_python_digest"],
        "python_origins": identity["python_origins"],
        "python": {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
        },
        "source_state": _source_state(path.parent),
        "dataset_artifact": _dataset_artifact(args),
        "benchmark_args": benchmark_contract(args),
        "benchmark_contract": benchmark_contract(args),
        "estimator_policy": estimator_policy(args),
        "environment": {key: os.environ.get(key, "") for key in env_keys},
        "controlled_environment": collect_relevant_environment(),
        "gpu_device_profile": gpu_device_profile,
        "host_profile": host_profile,
        "measurement_schema_path": str(schema_path),
        "measurement_schema": schema_document(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    print(f"Wrote {path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--easygraph-repo",
        type=Path,
        required=True,
        help="Frozen EasyGraph runtime containing easygraph/ and cpp_easygraph*.so.",
    )
    parser.add_argument(
        "--experiment", choices=["workflow", "return", "layout"], required=True
    )
    parser.add_argument(
        "--variant",
        default="full",
        choices=[
            "full",
            "no_graph_context",
            "no_cpp_graph_cache",
            "no_device_csr_cache",
            "adaptive_policy",
            "no_adaptive_policy",
        ],
    )
    parser.add_argument(
        "--edge-path",
        required=True,
        help="Dataset path relative to EG_Evaluation root or absolute path.",
    )
    parser.add_argument("--dataset-name", default="")
    parser.add_argument(
        "--graph-type", choices=["directed", "undirected"], required=True
    )
    parser.add_argument("--functions", default="all")
    parser.add_argument(
        "--workflow-order-id",
        default="canonical",
        help="Stable identifier for the selected same-graph function order.",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sssp-sources", type=int, default=8)
    parser.add_argument("--bc-sources", type=int, default=16)
    parser.add_argument(
        "--closeness-sources",
        type=int,
        default=16,
        help="Deterministic targets used only when exact all-node Closeness exceeds the main benchmark scale guard.",
    )
    parser.add_argument("--layout-pr-iters", type=int, default=20)
    parser.add_argument(
        "--measurement-mode",
        choices=["timing", "memory", "combined"],
        default=os.environ.get("EGGPU_MEASUREMENT_MODE", "combined").strip().lower(),
    )
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Run again even when a complete same-runtime output exists.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--expected-native-sha256", default="", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-runtime-python-digest", default="", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if not args.dataset_name:
        args.dataset_name = Path(args.edge_path).stem
    if not Path(args.edge_path).is_absolute():
        args.edge_path = str(Path(args.edge_path))
    return args


def run_worker(args):
    global MEASUREMENT_MODE
    MEASUREMENT_MODE = args.measurement_mode
    os.environ["EGGPU_MEASUREMENT_MODE"] = MEASUREMENT_MODE
    requested_runtime, loaded_runtime = load_and_validate_runtime(args)

    run_started_at = datetime.now().isoformat(timespec="seconds")
    run_t0 = time.perf_counter()
    gpu_device_profile = collect_gpu_device_profile(args.gpu, os.environ)
    host_profile = collect_host_profile()

    if args.experiment == "workflow":
        rows = run_workflow(args)
    elif args.experiment == "return":
        rows = run_return(args)
    elif args.experiment == "layout":
        rows = run_layout(args)
    else:
        raise ValueError(args.experiment)
    if MEASUREMENT_MODE == "timing":
        rows = [
            row
            for row in rows
            if not str(row.get("metric", "")).startswith("memory_")
        ]
    elif MEASUREMENT_MODE == "memory":
        rows = [
            row
            for row in rows
            if str(row.get("metric", "")).startswith("memory_")
        ]
    for row in rows:
        row["measurement_phase"] = MEASUREMENT_MODE
    failures = [row for row in rows if row.get("status") == "failed"]
    write_rows(rows, args.out)
    run_completed_at = datetime.now().isoformat(timespec="seconds")
    write_metadata(
        args,
        args.out,
        run_started_at=run_started_at,
        run_completed_at=run_completed_at,
        experiment_wall_seconds=time.perf_counter() - run_t0,
        gpu_device_profile=gpu_device_profile,
        host_profile=host_profile,
        requested_runtime=requested_runtime,
        loaded_runtime=loaded_runtime,
        run_status="failed" if failures else "complete",
        failure_row_count=len(failures),
    )
    if failures:
        examples = "; ".join(
            f"{row.get('function')}/{row.get('metric')}: {row.get('notes')}"
            for row in failures[:5]
        )
        raise SystemExit(
            f"ablation produced {len(failures)} failed row(s); examples: {examples}"
        )
    return 0


def coordinator(args):
    requested_runtime = collect_runtime_repository_provenance(args.easygraph_repo)
    if args.resume and reusable_output(args, requested_runtime):
        print(
            "Reusing complete same-runtime ablation output: "
            f"{Path(args.out).resolve()}",
            flush=True,
        )
        return 0
    command = worker_command(args, requested_runtime)
    environment = runtime_subprocess_environment(
        args.easygraph_repo, os.environ
    )
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    if args.out and not reusable_output(args, requested_runtime):
        raise RuntimeError(
            "ablation worker exited successfully without a complete, "
            "same-runtime output"
        )
    return 0


def main():
    args = parse_args()
    if args.worker:
        return run_worker(args)
    return coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
