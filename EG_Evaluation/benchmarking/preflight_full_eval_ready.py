#!/usr/bin/env python3
"""Lightweight preflight checks before a final EGGPU full benchmark.

This script intentionally does not run the full benchmark.  It verifies that
the benchmark process will see the workspace EasyGraph tree, the in-place
cpp_easygraph extension, and the expected EGGPU warmup/audit configuration.
It also runs the structural-hole scan-v correctness preflight.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
EASYGRAPH_REPO = WORKSPACE / "Easy-Graph"
RUN_SCRIPT = ROOT / "run_main_and_ablation.sh"
BUILD_SCRIPT = WORKSPACE / "scripts" / "build_eggpu.sh"
FULL_BASELINE_RUNNER = ROOT / "benchmarking" / "run_full_baselines.py"
ABLATION_RUNNER = ROOT / "benchmarking" / "run_eggpu_ablations.py"
GPU_DEVICE_PROFILE = ROOT / "benchmarking" / "gpu_device_profile.py"
FULL_AUDIT = ROOT / "benchmarking" / "audit_full_result.py"
BACKEND_SEPARATION_AUDIT = ROOT / "benchmarking" / "audit_backend_separation.py"
STRUCTURAL_PREFLIGHT = ROOT / "benchmarking" / "preflight_structural_scanv.py"
CLOSENESS_PREFLIGHT = ROOT / "benchmarking" / "preflight_closeness_semantics.py"
CLOSENESS_CUDA = EASYGRAPH_REPO / "gpu_easygraph" / "functions" / "centrality" / "closeness_centrality.cu"
BC_CUDA = EASYGRAPH_REPO / "gpu_easygraph" / "functions" / "centrality" / "betweenness_centrality.cu"

COMPILED_SOURCE_FILES = (
    EASYGRAPH_REPO / "cpp_easygraph" / "cpp_easygraph.cpp",
    EASYGRAPH_REPO / "cpp_easygraph" / "functions" / "centrality" / "closeness.cpp",
    EASYGRAPH_REPO / "cpp_easygraph" / "functions" / "centrality" / "betweenness.cpp",
    EASYGRAPH_REPO / "cpp_easygraph" / "functions" / "centrality" / "centrality.h",
    EASYGRAPH_REPO / "gpu_easygraph" / "gpu_easygraph.h",
    EASYGRAPH_REPO / "gpu_easygraph" / "functions" / "centrality" / "centrality.cpp",
    EASYGRAPH_REPO / "gpu_easygraph" / "functions" / "centrality" / "closeness_centrality.cu",
    EASYGRAPH_REPO / "gpu_easygraph" / "functions" / "centrality" / "closeness_centrality.cuh",
    EASYGRAPH_REPO / "gpu_easygraph" / "functions" / "centrality" / "betweenness_centrality.cu",
    EASYGRAPH_REPO / "gpu_easygraph" / "functions" / "centrality" / "betweenness_centrality.cuh",
)

EXPECTED_FUNCTIONS = (
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

PUBLIC_GPU_DISPATCH_CONTRACTS = (
    (
        EASYGRAPH_REPO / "easygraph" / "functions" / "basic" / "cluster.py",
        "_clustering_gpu_runtime_dispatch",
    ),
    (
        EASYGRAPH_REPO / "easygraph" / "functions" / "centrality" / "betweenness.py",
        "_betweenness_centrality_gpu_runtime_dispatch",
    ),
    (
        EASYGRAPH_REPO / "easygraph" / "functions" / "centrality" / "closeness.py",
        "_closeness_centrality_gpu_runtime_dispatch",
    ),
    (
        EASYGRAPH_REPO / "easygraph" / "functions" / "centrality" / "pagerank.py",
        "_pagerank_gpu_runtime_dispatch",
    ),
    (
        EASYGRAPH_REPO / "easygraph" / "functions" / "components" / "connected.py",
        "_connected_components_gpu_runtime_dispatch",
    ),
    (
        EASYGRAPH_REPO / "easygraph" / "functions" / "components" / "strongly_connected.py",
        "_strongly_connected_components_gpu_runtime_dispatch",
    ),
    (
        EASYGRAPH_REPO / "easygraph" / "functions" / "components" / "weakly_connected.py",
        "_weakly_connected_components_gpu_runtime_dispatch",
    ),
    (
        EASYGRAPH_REPO / "easygraph" / "functions" / "core" / "k_core.py",
        "_k_core_gpu_runtime_dispatch",
    ),
    (
        EASYGRAPH_REPO / "easygraph" / "functions" / "path" / "mst.py",
        "_minimum_spanning_tree_gpu_runtime_dispatch",
    ),
    (
        EASYGRAPH_REPO / "easygraph" / "functions" / "path" / "path.py",
        "_path_gpu_runtime_dispatch",
    ),
    (
        EASYGRAPH_REPO / "easygraph" / "functions" / "structural_holes" / "evaluation.py",
        "_structural_holes_gpu_runtime_dispatch",
    ),
)


def emit(name: str, status: str, **kwargs) -> None:
    row = {"check": name, "status": status}
    row.update(kwargs)
    print("RESULT_JSON " + json.dumps(row, sort_keys=True))


def check_import_paths() -> bool:
    try:
        import cpp_easygraph
        import easygraph
    except Exception as exc:
        emit("import_paths", "fail", error=repr(exc))
        return False

    eg_path = Path(easygraph.__file__).resolve()
    cpp_path = Path(cpp_easygraph.__file__).resolve()
    ok = str(eg_path).startswith(str(EASYGRAPH_REPO.resolve())) and cpp_path.parent == EASYGRAPH_REPO.resolve()
    emit(
        "import_paths",
        "ok" if ok else "fail",
        easygraph_file=str(eg_path),
        cpp_easygraph_file=str(cpp_path),
    )
    return ok


def check_compiled_extension_freshness() -> bool:
    try:
        import cpp_easygraph
    except Exception as exc:
        emit("compiled_extension_freshness", "fail", error=repr(exc))
        return False

    cpp_file = getattr(cpp_easygraph, "__file__", None)
    if not cpp_file:
        emit(
            "compiled_extension_freshness",
            "fail",
            note="cpp_easygraph has no compiled extension file; rebuild EGGPU in this checkout",
        )
        return False

    so_path = Path(cpp_file).resolve()
    missing = [str(path) for path in COMPILED_SOURCE_FILES if not path.exists()]
    if missing:
        emit("compiled_extension_freshness", "fail", missing=missing)
        return False

    so_mtime = so_path.stat().st_mtime
    newest_source = max(COMPILED_SOURCE_FILES, key=lambda path: path.stat().st_mtime)
    newest_source_mtime = newest_source.stat().st_mtime
    ok = so_mtime + 1.0 >= newest_source_mtime
    emit(
        "compiled_extension_freshness",
        "ok" if ok else "fail",
        cpp_easygraph_file=str(so_path),
        cpp_easygraph_mtime=so_mtime,
        newest_source=str(newest_source),
        newest_source_mtime=newest_source_mtime,
        note=(
            "compiled extension is newer than tracked C++/CUDA inputs"
            if ok
            else "compiled extension is older than at least one tracked C++/CUDA input; rebuild EGGPU"
        ),
    )
    return ok


def check_run_script() -> bool:
    text = RUN_SCRIPT.read_text()
    required = {
        "easygraph_warmup_2": "--easygraph-warmup 2",
        "direct_child_python_env": 'EGGPU_CHILD_PYTHON="${COMMON_PY}"',
        "conda_run_disabled_by_default": "EGGPU_USE_CONDA_RUN=FALSE",
        "pythonpath_repo": 'PYTHONPATH="${EG_REPO}:${PYTHONPATH:-}"',
        "audit_full_result": "benchmarking/audit_full_result.py",
        "audit_backend_separation": "benchmarking/audit_backend_separation.py",
        "pair_sota_summary": "benchmarking/summarize_non_sota_pairs.py",
        "final_result_summary": "benchmarking/summarize_final_result.py",
        "preflight_default_enabled": 'RUN_PREFLIGHT="${RUN_PREFLIGHT:-TRUE}"',
        "preflight_runner": "benchmarking/preflight_full_eval_ready.py",
        "preflight_log": ".preflight.log",
        "main_eval_subshell": "( run_main_eval )",
        "ablation_subshell": "( run_ablations )",
        "visibility_marker_default_false": 'EGGPU_GPU_VISIBILITY_MARKER="${EGGPU_GPU_VISIBILITY_MARKER:-FALSE}"',
        "visibility_marker_enabled_mb_default": 'EGGPU_GPU_VISIBILITY_MARKER_MB="${EGGPU_GPU_VISIBILITY_MARKER_MB:-256}"',
        "visibility_marker_adjust_auto_default": 'EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB="${EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB:-AUTO}"',
        "visibility_marker_mb_propagated": 'EGGPU_GPU_VISIBILITY_MARKER_MB="${EGGPU_GPU_VISIBILITY_MARKER_MB}"',
        "visibility_marker_adjust_propagated": 'EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB="${EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB}"',
        "kcore_min_max_degree_auto": 'EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE="${EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE:-AUTO}"',
        "functions_all": "--functions all",
        "datasets_all": "--datasets all",
    }
    missing = [name for name, needle in required.items() if needle not in text]
    emit("run_script_config", "ok" if not missing else "fail", missing=missing)
    return not missing


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def _flatten_category_functions(categories) -> list[str]:
    out = []
    for values in categories.values():
        out.extend(list(values))
    return out


def check_function_registry_consistency() -> bool:
    expected = list(EXPECTED_FUNCTIONS)
    exact_checks = {
        "run_full_baselines.DEFAULT_FUNCTIONS": list(
            _literal_assignment(FULL_BASELINE_RUNNER, "DEFAULT_FUNCTIONS")
        ),
        "library_baselines.FUNCTION_ORDER": list(
            _literal_assignment(ROOT / "benchmarking" / "library_baselines.py", "FUNCTION_ORDER")
        ),
        "run_eggpu_ablations.DEFAULT_FUNCTIONS": list(
            _literal_assignment(ROOT / "benchmarking" / "run_eggpu_ablations.py", "DEFAULT_FUNCTIONS")
        ),
        "run_eggpu_ablations.RETURN_FUNCTIONS": list(
            _literal_assignment(ROOT / "benchmarking" / "run_eggpu_ablations.py", "RETURN_FUNCTIONS")
        ),
        "generate_paper_artifacts.FUNCTION_ORDER": list(
            _literal_assignment(ROOT / "benchmarking" / "generate_paper_artifacts.py", "FUNCTION_ORDER")
        ),
        "audit_full_result.LEGACY_EXPECTED_FUNCTIONS": list(
            _literal_assignment(FULL_AUDIT, "LEGACY_EXPECTED_FUNCTIONS")
        ),
    }
    set_checks = {
        "summarize_category_estimate.CATEGORIES": _flatten_category_functions(
            _literal_assignment(ROOT / "benchmarking" / "summarize_category_estimate.py", "CATEGORIES")
        ),
        "summarize_final_result.FUNCTION_CATEGORY": list(
            _literal_assignment(ROOT / "benchmarking" / "summarize_final_result.py", "FUNCTION_CATEGORY").keys()
        ),
    }
    mismatches = []
    for label, actual in exact_checks.items():
        if actual != expected:
            mismatches.append(
                {
                    "registry": label,
                    "missing": [x for x in expected if x not in actual],
                    "extra": [x for x in actual if x not in expected],
                    "order_matches": actual == expected,
                }
            )
    expected_set = set(expected)
    for label, actual in set_checks.items():
        actual_set = set(actual)
        if actual_set != expected_set:
            mismatches.append(
                {
                    "registry": label,
                    "missing": [x for x in expected if x not in actual_set],
                    "extra": [x for x in actual if x not in expected_set],
                    "order_matches": None,
                }
            )
    emit(
        "function_registry_consistency",
        "ok" if not mismatches else "fail",
        expected=expected,
        mismatches=mismatches,
    )
    return not mismatches


def check_child_python_wrapper() -> bool:
    text = FULL_BASELINE_RUNNER.read_text()
    required = {
        "direct_child_python_declared": "DIRECT_CHILD_PYTHON",
        "conda_run_opt_in": "EGGPU_USE_CONDA_RUN",
        "direct_child_command": "return [DIRECT_CHILD_PYTHON, str(script)",
        "conda_run_python_command": 'return [*conda_run_prefix(), "python", str(script)',
        "direct_cuda_home_pin": 'env["CUDA_HOME"] = cuda_root_str',
        "direct_cupy_cuda_pin": 'env["CUPY_CUDA_PATH"] = cuda_root_str',
        "direct_conda_prefix_pin": 'env["CONDA_PREFIX"] = cuda_root_str',
        "run_metadata_json": "run_metadata.json",
        "collect_run_metadata": "collect_run_metadata",
        "write_run_metadata": "write_run_metadata",
        "isolated_plot_generation": "write_plot_and_tables_isolated",
        "plot_generation_timeout": "EGGPU_PLOT_TIMEOUT",
    }
    missing = [name for name, needle in required.items() if needle not in text]
    emit("child_python_wrapper", "ok" if not missing else "fail", missing=missing)
    return not missing


def check_gpu_routing_contract() -> bool:
    """Verify physical GPU monitoring and logical CUDA device usage stay split.

    The official runner exposes a requested physical GPU through
    CUDA_VISIBLE_DEVICES, then all CUDA runtime calls inside that restricted
    process must use logical device 0.  NVML memory accounting still needs the
    physical index, carried by EGGPU_MONITOR_GPU_INDEX.
    """
    run_text = RUN_SCRIPT.read_text()
    full_text = FULL_BASELINE_RUNNER.read_text()
    lib_text = (ROOT / "benchmarking" / "library_baselines.py").read_text()
    abl_text = (ROOT / "benchmarking" / "run_eggpu_ablations.py").read_text()
    marker_text = (ROOT / "benchmarking" / "gpu_visibility_marker.py").read_text()

    required = {
        "main_sets_physical_visible_gpu": 'CUDA_VISIBLE_DEVICES="${MAIN_GPU}"' in run_text,
        "main_sets_physical_monitor_gpu": 'EGGPU_MONITOR_GPU_INDEX="${MAIN_GPU}"' in run_text,
        "ablation_sets_physical_visible_gpu": 'CUDA_VISIBLE_DEVICES="${ABL_GPU}"' in run_text,
        "ablation_sets_physical_monitor_gpu": 'EGGPU_MONITOR_GPU_INDEX="${ABL_GPU}"' in run_text,
        "marker_uses_logical_zero": "cudaSetDevice(0)" in marker_text,
        "marker_documents_logical_zero": "logical CUDA device 0" in marker_text,
        "marker_allocates_fixed_memory": "cudaMalloc" in marker_text and "EGGPU_GPU_VISIBILITY_MARKER_MB" in marker_text,
        "full_runner_resolves_monitor_index": "EGGPU_MONITOR_GPU_INDEX" in full_text
        and "nvmlDeviceGetHandleByIndex(_resolve_monitor_gpu_index(env))" in full_text,
        "full_runner_adjusts_visibility_marker_memory": "visibility_marker_adjust_mb(env)" in full_text
        and "EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB" in full_text,
        "library_runner_resolves_monitor_index": "EGGPU_MONITOR_GPU_INDEX" in lib_text
        and "nvmlDeviceGetHandleByIndex(self._gpu_index)" in lib_text,
        "library_runner_adjusts_visibility_marker_memory": "visibility_marker_adjust_mb()" in lib_text
        and "EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB" in lib_text,
        "ablation_sets_monitor_index": 'os.environ["EGGPU_MONITOR_GPU_INDEX"] = str(args.gpu)' in abl_text,
    }
    missing = [name for name, ok in required.items() if not ok]
    emit("gpu_routing_contract", "ok" if not missing else "fail", missing=missing)
    return not missing


def check_build_script() -> bool:
    text = BUILD_SCRIPT.read_text()
    required = {
        "local_cuda_root_required": "EGGPU_CUDA_ROOT",
        "nvcc_under_cuda_root": '${CUDA_ROOT}/bin/nvcc',
        "ninja_default": "CMAKE_GENERATOR=Ninja",
        "future_mtime_scan": "future_mtimes",
        "clock_skew_guard": "EGGPU_ALLOW_CLOCK_SKEW",
        "safe_build_root": "Refusing to clean unexpected build path",
        "openmp_gomp_pin": "OpenMP_gomp_LIBRARY",
        "cuda_architecture_pin": "CMAKE_CUDA_ARCHITECTURES",
        "cuda_architecture_auto_detect": "detect_cuda_architectures",
        "cuda_architecture_dry_run": "EGGPU_BUILD_DRY_RUN",
        "sanitize_compile_env": "-u CFLAGS -u CPPFLAGS -u CXXFLAGS",
    }
    missing = [name for name, needle in required.items() if needle not in text]
    emit("build_script_safety", "ok" if not missing else "fail", missing=missing)
    return not missing


def check_closeness_cuda_launch_contract() -> bool:
    text = CLOSENESS_CUDA.read_text()
    centrality_text = (EASYGRAPH_REPO / "gpu_easygraph" / "functions" / "centrality" / "centrality.cpp").read_text()
    required = {
        "wrapper_output_matches_source_count": "CC = vector<double>(sources.size())" in centrality_text,
        "wrapper_passes_unweighted_flag": "unweighted, CC.data(), kernel_seconds" in centrality_text,
        "cuda_uses_device_csr_cache": "acquire_device_csr(V, E, W, len_V, len_E, needs_weights" in text,
        "cuda_unweighted_bfs_default_with_regression_switch": "EASYGRAPH_GPU_CLOSENESS_UNWEIGHTED_BFS" in text
        and "bfs_env == nullptr || env_truthy(bfs_env)" in text
        and "!env_falsey(bfs_env)" in text
        and "const bool needs_weights = !use_bfs_kernel" in text,
        "cuda_keeps_bfs_path_explicit": "d_bfs_cc" in text
        and "if (use_bfs_kernel)" in text,
        "dijkstra_grid_capped_by_sources": "dijkstra_grid_size > len_sources" in text
        and "dijkstra_grid_size = len_sources" in text,
        "bfs_grid_capped_by_sources": "bfs_grid_size > len_sources" in text
        and "bfs_grid_size = len_sources" in text,
        "workspace_uses_dijkstra_grid": "sizeof(int) * dijkstra_grid_size * len_V" in text
        and "sizeof(double) * dijkstra_grid_size * len_V" in text,
        "device_output_uses_source_count": "sizeof(double) * len_sources" in text,
        "min_edge_launch_uses_min_edge_config": (
            "d_calc_min_edge<<<min_edge_grid_size, min_edge_block_size>>>" in text
        ),
        "dijkstra_launch_uses_dijkstra_config": (
            "d_dijkstra_cc<<<dijkstra_grid_size, dijkstra_block_size>>>" in text
        ),
        "kernel_timer_excludes_output_copy": text.find("cudaEventRecord(runtime.stop_event)") < text.find("cudaMemcpy(CC, d_CC"),
    }
    missing = [name for name, ok in required.items() if not ok]
    emit("closeness_cuda_launch_contract", "ok" if not missing else "fail", missing=missing)
    return not missing


def _function_source(path: Path, function_name: str) -> str:
    text = path.read_text()
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise KeyError(f"{function_name} not found in {path}")


def _has_strict_raise_contract(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            continue
        if isinstance(node.type, ast.Name) and node.type.id != "Exception":
            continue
        body = node.body
        for idx, stmt in enumerate(body):
            if not isinstance(stmt, ast.If):
                continue
            test = ast.unparse(stmt.test)
            strict_test = "gpu_strict_errors()" in test or "strict_errors" in test
            raises = any(isinstance(inner, ast.Raise) for inner in ast.walk(stmt))
            if strict_test and raises:
                return True
    return False


def check_public_gpu_dispatch_strict_errors() -> bool:
    """Prevent paper runs from silently timing a CPU fallback as EGGPU.

    Public EasyGraph functions are allowed to return `None` when GPU dispatch is
    disabled or when the selected backend is not our native backend.  Once a
    native EGGPU dispatch path has thrown, however, benchmark mode sets
    `EASYGRAPH_GPU_STRICT_ERRORS=TRUE`; the exception must be re-raised so the
    row is marked failed/timeout instead of falling through to a CPU path.
    """
    issues = []
    for path, function_name in PUBLIC_GPU_DISPATCH_CONTRACTS:
        try:
            source = _function_source(path, function_name)
        except Exception as exc:
            issues.append(
                {
                    "file": str(path.relative_to(WORKSPACE)),
                    "function": function_name,
                    "issue": "missing dispatch function",
                    "detail": repr(exc),
                }
            )
            continue

        required = {
            "checks_runtime_enabled": "gpu_runtime_enabled()" in source,
            "uses_native_backend": "mine_backend" in source,
            "catches_gpu_exception": "except Exception" in source,
            "reraises_under_strict": _has_strict_raise_contract(source),
        }
        missing = [name for name, ok in required.items() if not ok]
        if missing:
            issues.append(
                {
                    "file": str(path.relative_to(WORKSPACE)),
                    "function": function_name,
                    "missing": missing,
                }
            )

    run_text = RUN_SCRIPT.read_text()
    if "EASYGRAPH_GPU_STRICT_ERRORS=TRUE" not in run_text:
        issues.append(
            {
                "file": str(RUN_SCRIPT.relative_to(WORKSPACE)),
                "function": "official runner",
                "missing": ["sets EASYGRAPH_GPU_STRICT_ERRORS=TRUE"],
            }
        )

    emit(
        "public_gpu_dispatch_strict_errors",
        "ok" if not issues else "fail",
        checked=[
            {
                "file": str(path.relative_to(WORKSPACE)),
                "function": function_name,
            }
            for path, function_name in PUBLIC_GPU_DISPATCH_CONTRACTS
        ],
        issues=issues,
    )
    return not issues


def check_closeness_baseline_semantics_contract() -> bool:
    text = (ROOT / "benchmarking" / "library_baselines.py").read_text()
    semantic_text = CLOSENESS_PREFLIGHT.read_text()
    required = {
        "networkx_reverse_directed_view": "g.reverse(copy=False)" in text
        and "networkx uses reverse graph view" in text,
        "networkx_wf_improved": "wf_improved=True" in text,
        "igraph_outward_mode": 'mode = "OUT" if graph_type == "directed" else "ALL"' in text,
        "igraph_wf_correction": "g.neighborhood_size" in text
        and "Wasserman-Faust disconnected correction applied" in text,
        "nx_cugraph_closeness_skipped": "does not include closeness_centrality" in text
        and "avoid measuring an unsupported backend or CPU fallback" in text,
        "semantic_preflight_networkx_reverse": "nx_g.reverse(copy=False)" in semantic_text,
        "semantic_preflight_igraph_wf": "g.neighborhood_size" in semantic_text
        and "(count) - 1" in semantic_text,
    }
    missing = [name for name, ok in required.items() if not ok]
    emit(
        "closeness_baseline_semantics_contract",
        "ok" if not missing else "fail",
        missing=missing,
    )
    return not missing


def check_unweighted_centrality_bfs_contract() -> bool:
    bc_text = BC_CUDA.read_text()
    centrality_text = (EASYGRAPH_REPO / "gpu_easygraph" / "functions" / "centrality" / "centrality.cpp").read_text()
    pybind_text = (EASYGRAPH_REPO / "cpp_easygraph" / "functions" / "centrality" / "betweenness.cpp").read_text()
    lib_text = (ROOT / "benchmarking" / "library_baselines.py").read_text()
    required = {
        "bc_has_bfs_brandes_kernel": "d_bfs_bc" in bc_text
        and "EASYGRAPH_GPU_BC_UNWEIGHTED_BFS" in bc_text,
        "bc_default_bfs_has_regression_switch": "bfs_env == nullptr || env_truthy(bfs_env)" in bc_text
        and "!env_falsey(bfs_env)" in bc_text,
        "bc_wrapper_passes_unweighted_flag": "bool unweighted" in centrality_text
        and "endpoints, unweighted" in centrality_text,
        "bc_pybind_uses_weight_none": "weight.is_none(), BC, &kernel_seconds" in pybind_text,
        "runner_notes_unweighted_bfs": "unweighted GPU path uses BFS-Brandes kernel" in lib_text
        and "unweighted GPU path uses BFS kernel" in lib_text,
    }
    missing = [name for name, ok in required.items() if not ok]
    emit(
        "unweighted_centrality_bfs_contract",
        "ok" if not missing else "fail",
        missing=missing,
    )
    return not missing


def check_backend_separation_static_contract() -> bool:
    """Ensure EasyGraph CPU/C++ baselines cannot inherit the EGGPU path.

    The result-level backend separation audit verifies completed logs.  This
    static gate catches accidental runner edits before a long benchmark starts:
    EGGPU must be the only EasyGraph mode with GPU enabled and warmup/prewarm,
    while CPU/C++ baselines must run as separate child processes with the GPU
    path explicitly disabled.
    """
    lib_text = (ROOT / "benchmarking" / "library_baselines.py").read_text()
    full_text = FULL_BASELINE_RUNNER.read_text()
    audit_text = BACKEND_SEPARATION_AUDIT.read_text()
    run_text = RUN_SCRIPT.read_text()

    required = {
        "main_runs_backend_audit": "benchmarking/audit_backend_separation.py" in run_text,
        "library_prints_mode_header": "[easygraph-mode]" in lib_text
        and "EASYGRAPH_ENABLE_GPU={os.environ.get('EASYGRAPH_ENABLE_GPU', '')}" in lib_text
        and "EASYGRAPH_GPU_BACKEND={os.environ.get('EASYGRAPH_GPU_BACKEND', '')}" in lib_text,
        "library_gpu_mode_enables_only_eggpu": 'backend = "EGGPU"' in lib_text
        and 'os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"' in lib_text
        and 'os.environ["EASYGRAPH_GPU_BACKEND"] = "mine"' in lib_text
        and 'os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"' in lib_text
        and 'os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "TRUE"' in lib_text,
        "library_gpu_mode_disables_result_cache": 'os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"' in lib_text
        and 'os.environ["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"' in lib_text,
        "library_gpu_mode_disables_host_policy": 'os.environ["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"' in lib_text
        and 'os.environ["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"' in lib_text
        and 'os.environ["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"' in lib_text,
        "library_cpu_cpp_modes_disable_gpu": 'backend = "easygraph-cpu"' in lib_text
        and 'backend = "easygraph-cpp"' in lib_text
        and lib_text.count('os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"') >= 2
        and lib_text.count('os.environ.pop("EASYGRAPH_GPU_BACKEND", None)') >= 2
        and lib_text.count('os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "FALSE"') >= 2,
        "library_cpp_mode_uses_cpp_graph": 'if mode == "cpp":' in lib_text
        and "return g.cpp()" in lib_text,
        "library_gpu_prewarm_is_strict": "EGGPU graph-context prewarm failed" in lib_text
        and "EASYGRAPH_GPU_STRICT_ERRORS" in lib_text,
        "library_gpu_warmup_is_strict": "def easygraph_warmup" in lib_text
        and 'if mode == "gpu":' in lib_text
        and "raise" in lib_text,
        "library_no_pagerank_alpha_fallback": "pr_alpha_candidates" not in lib_text
        and "fallback from" not in lib_text,
        "library_no_kcore_retry_fallback": "directed->undirected fallback" not in lib_text,
        "library_cpp_structural_hole_isolation": (
            "GPU-enabled cpp_easygraph structural-hole bindings route to CUDA at compile time; "
            "skipped to keep CPU C++ baseline isolated"
        )
        in lib_text,
        "full_runner_enables_gpu_only_for_eggpu": 'if base == "EGGPU":' in full_text
        and 'env_one["EASYGRAPH_ENABLE_GPU"] = "TRUE"' in full_text
        and 'env_one["EASYGRAPH_GPU_BACKEND"] = "mine"' in full_text
        and 'env_one["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"' in full_text
        and 'env_one["EASYGRAPH_ENABLE_GPU"] = "FALSE"' in full_text
        and 'env_one.pop("EASYGRAPH_GPU_BACKEND", None)' in full_text,
        "full_runner_disables_host_policy_for_eggpu": 'env_one["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"' in full_text
        and 'env_one["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"' in full_text
        and 'env_one["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"' in full_text,
        "full_runner_checks_idle_before_each_eggpu_child": "check_eggpu_child_gpu_idle" in full_text
        and "gpu_busy_before_eggpu_child" in full_text
        and 'if base == "EGGPU":' in full_text,
        "full_runner_warmup_only_for_eggpu": 'base_warmup = warmup if base == "EGGPU" else 0' in full_text
        and 'base_easygraph_warmup = easygraph_warmup if base == "EGGPU" else 0' in full_text,
        "backend_audit_expected_modes": '"EGGPU":' in audit_text
        and '"mode": "gpu"' in audit_text
        and '"easygraph-cpp":' in audit_text
        and '"mode": "cpp"' in audit_text
        and '"easygraph-cpu":' in audit_text
        and '"mode": "cpu"' in audit_text,
        "backend_audit_checks_gpu_flags": "EASYGRAPH_ENABLE_GPU" in audit_text
        and "EASYGRAPH_GPU_BACKEND" in audit_text
        and "easygraph_warmup" in audit_text,
        "backend_audit_accepts_timeout_before_entry_only_as_timeout": "timeout_before_backend_entry" in audit_text
        and "TIMEOUT_TOO_LONG" in audit_text,
        "backend_audit_checks_structural_skip_note": (
            "skipped to keep CPU C++ baseline isolated" in audit_text
        ),
    }
    missing = [name for name, ok in required.items() if not ok]
    emit(
        "backend_separation_static_contract",
        "ok" if not missing else "fail",
        missing=missing,
    )
    return not missing


def check_sota_summary_validation_filter_contract() -> bool:
    final_text = (ROOT / "benchmarking" / "summarize_final_result.py").read_text()
    pair_text = (ROOT / "benchmarking" / "summarize_non_sota_pairs.py").read_text()
    required = {
        "final_reads_correctness_validation": "correctness_validation.csv" in final_text
        and "validation_status" in final_text
        and "VALIDATED_SOTA_BASELINE_STATUSES" in final_text,
        "final_excludes_non_eggpu_validation_failures": "validated_non_eggpu" in final_text
        and "inconclusive_self_reference" in final_text
        and "sampled_pass" in final_text,
        "pair_reads_correctness_validation": "correctness_validation.csv" in pair_text
        and "validation_status" in pair_text
        and "VALIDATED_SOTA_BASELINE_STATUSES" in pair_text,
        "pair_excludes_non_eggpu_validation_failures": "validated_non_eggpu" in pair_text
        and "inconclusive_self_reference" in pair_text
        and "sampled_pass" in pair_text,
    }
    missing = [name for name, ok in required.items() if not ok]
    emit(
        "sota_summary_validation_filter_contract",
        "ok" if not missing else "fail",
        missing=missing,
    )
    return not missing


def check_validation_reference_oracle_contract() -> bool:
    validation_text = (ROOT / "benchmarking" / "validate_correctness.py").read_text()
    required = {
        "has_unsafe_reference_map": "UNSAFE_REFERENCE_BY_FUNCTION" in validation_text,
        "bc_excludes_nx_cugraph_oracle": '"BC": {"nx-cugraph"}' in validation_text
        or "'BC': {'nx-cugraph'}" in validation_text,
        "pick_reference_skips_unsafe": "if backend in unsafe:" in validation_text
        and "continue" in validation_text,
        "validation_doc_mentions_unsafe_reference": "Function-specific unsafe references are skipped" in validation_text,
    }
    missing = [name for name, ok in required.items() if not ok]
    emit(
        "validation_reference_oracle_contract",
        "ok" if not missing else "fail",
        missing=missing,
    )
    return not missing


def check_gpu_device_profile_metadata_contract() -> bool:
    profile_text = GPU_DEVICE_PROFILE.read_text()
    full_text = FULL_BASELINE_RUNNER.read_text()
    ablation_text = ABLATION_RUNNER.read_text()
    required = {
        "profile_module_collects_selected_device": "def collect_gpu_device_profile" in profile_text
        and "selected_device" in profile_text
        and "resolved_monitor_gpu_index" in profile_text,
        "profile_records_compute_capability": "compute_capability" in profile_text
        and "cuda_architecture" in profile_text,
        "profile_records_memory": "memory_total_mb" in profile_text,
        "profile_falls_back_to_nvidia_smi": "def _collect_with_nvidia_smi" in profile_text,
        "full_metadata_records_gpu_profile": "collect_gpu_device_profile(args.gpu, env)" in full_text
        and '"gpu_device_profile"' in full_text,
        "ablation_metadata_records_gpu_profile": "collect_gpu_device_profile(args.gpu, os.environ)" in ablation_text
        and '"gpu_device_profile"' in ablation_text
        and "metadata.json" in ablation_text,
    }
    missing = [name for name, ok in required.items() if not ok]
    emit(
        "gpu_device_profile_metadata_contract",
        "ok" if not missing else "fail",
        missing=missing,
    )
    return not missing


def check_audit_metadata_gate() -> bool:
    with tempfile.TemporaryDirectory(prefix="eggpu_preflight_audit_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "results_long.csv").write_text(
            "\n".join(
                [
                    "dataset_size,graph_type,dataset,function,baseline,metric,seconds,status,correctness,log,notes",
                    "small,undirected,toy,PageRank,EGGPU,e2e,0.1,ok,,,",
                    "",
                ]
            )
        )
        (tmp_path / "correctness_validation.csv").write_text(
            "\n".join(
                [
                    "dataset,function,baseline,validation_status,details",
                    "toy,PageRank,EGGPU,pass,",
                    "",
                ]
            )
        )
        proc = subprocess.run(
            [sys.executable, str(FULL_AUDIT), str(tmp_path)],
            cwd=str(WORKSPACE),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        issues_path = tmp_path / "audit" / "metadata_issues.csv"
        issues_text = issues_path.read_text(errors="replace") if issues_path.exists() else ""
        ok = (
            proc.returncode == 2
            and issues_path.exists()
            and "run_metadata.json" in issues_text
        )
        emit(
            "audit_metadata_gate",
            "ok" if ok else "fail",
            returncode=proc.returncode,
            has_metadata_issues_csv=issues_path.exists(),
        )
        return ok


def check_audit_validation_gate() -> bool:
    with tempfile.TemporaryDirectory(prefix="eggpu_preflight_validation_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "results_long.csv").write_text(
            "\n".join(
                [
                    "dataset_size,graph_type,dataset,function,baseline,metric,seconds,status,correctness,log,notes",
                    "small,undirected,toy,PageRank,EGGPU,e2e,0.1,ok,sum=1.0,,",
                    "",
                ]
            )
        )
        (tmp_path / "run_metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "completed_at": "2026-06-02T00:00:00",
                    "git": {"commit": "preflight", "dirty": False},
                    "python": {"executable": sys.executable, "direct_child_python": sys.executable},
                    "cuda": {"local_cuda_root": "/tmp", "nvcc": "/tmp/nvcc"},
                    "build_artifacts": {"cpp_easygraph": [{"path": "preflight.so"}]},
                    "artifacts": {
                        "results_long_rows": 1,
                        "validation_error": "",
                        "plot_error": "",
                        "files": ["results_long.csv"],
                    },
                    "benchmark_args": {
                        "datasets": ["toy"],
                        "functions": ["PageRank"],
                        "warmup": 0,
                        "easygraph_warmup": 2,
                    },
                    "environment": {
                        "EASYGRAPH_ENABLE_GPU": "TRUE",
                        "EASYGRAPH_GPU_BACKEND": "mine",
                        "EGGPU_USE_CONDA_RUN": "FALSE",
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
        proc = subprocess.run(
            [sys.executable, str(FULL_AUDIT), str(tmp_path)],
            cwd=str(WORKSPACE),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        issues_path = tmp_path / "audit" / "validation_issues.csv"
        issues_text = issues_path.read_text(errors="replace") if issues_path.exists() else ""
        ok = (
            proc.returncode == 2
            and issues_path.exists()
            and "correctness_validation.csv missing or empty" in issues_text
            and "missing EGGPU pass validation" in issues_text
        )
        emit(
            "audit_validation_gate",
            "ok" if ok else "fail",
            returncode=proc.returncode,
            has_validation_issues_csv=issues_path.exists(),
        )
        return ok


def check_audit_validation_error_gate() -> bool:
    with tempfile.TemporaryDirectory(prefix="eggpu_preflight_validation_error_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "results_long.csv").write_text(
            "\n".join(
                [
                    "dataset_size,graph_type,dataset,function,baseline,metric,seconds,status,correctness,log,notes",
                    "small,undirected,toy,PageRank,EGGPU,e2e,0.1,ok,sum=1.0,,",
                    "",
                ]
            )
        )
        (tmp_path / "correctness_validation.csv").write_text(
            "\n".join(
                [
                    "dataset,function,baseline,validation_status,details",
                    "toy,PageRank,EGGPU,pass,",
                    "",
                ]
            )
        )
        (tmp_path / "run_metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "completed_at": "2026-06-02T00:00:00",
                    "git": {"commit": "preflight", "dirty": False},
                    "python": {"executable": sys.executable, "direct_child_python": sys.executable},
                    "cuda": {"local_cuda_root": "/tmp", "nvcc": "/tmp/nvcc"},
                    "build_artifacts": {"cpp_easygraph": [{"path": "preflight.so"}]},
                    "artifacts": {
                        "results_long_rows": 1,
                        "validation_error": "synthetic validation failure",
                        "plot_error": "",
                        "files": ["results_long.csv", "correctness_validation.csv"],
                    },
                    "benchmark_args": {
                        "datasets": ["toy"],
                        "functions": ["PageRank"],
                        "warmup": 0,
                        "easygraph_warmup": 2,
                    },
                    "environment": {
                        "EASYGRAPH_ENABLE_GPU": "TRUE",
                        "EASYGRAPH_GPU_BACKEND": "mine",
                        "EGGPU_USE_CONDA_RUN": "FALSE",
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
        proc = subprocess.run(
            [sys.executable, str(FULL_AUDIT), str(tmp_path)],
            cwd=str(WORKSPACE),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        issues_path = tmp_path / "audit" / "metadata_issues.csv"
        issues_text = issues_path.read_text(errors="replace") if issues_path.exists() else ""
        ok = (
            proc.returncode == 2
            and issues_path.exists()
            and "artifacts.validation_error" in issues_text
            and "synthetic validation failure" in issues_text
        )
        emit(
            "audit_validation_error_gate",
            "ok" if ok else "fail",
            returncode=proc.returncode,
            has_metadata_issues_csv=issues_path.exists(),
        )
        return ok


def run_closeness_preflight() -> bool:
    env = dict(os.environ)
    env["EASYGRAPH_ENABLE_GPU"] = "FALSE"
    env["PYTHONPATH"] = str(EASYGRAPH_REPO) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [sys.executable, str(CLOSENESS_PREFLIGHT)],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="")
    emit(
        "closeness_semantics_preflight",
        "ok" if proc.returncode == 0 else "fail",
        returncode=proc.returncode,
    )
    return proc.returncode == 0


def check_timeout_recommendation() -> bool:
    text = RUN_SCRIPT.read_text()
    has_env_timeout = 'LIBRARY_TIMEOUT="${LIBRARY_TIMEOUT:-100}"' in text
    emit(
        "timeout_recommendation",
        "ok" if has_env_timeout else "fail",
        recommendation="Current fast final-check default is LIBRARY_TIMEOUT=100; raise it only for a paper rerun where timeout coverage matters more than wall time.",
    )
    return has_env_timeout


def check_outer_shell_hygiene() -> bool:
    """Warn about startup files that can leak machine env into paper logs.

    The benchmark child processes sanitize their own environment, but the outer
    shell can still print/export service variables before the runner starts.
    This check catches the observed shared-server hazard where `.bashrc` reads
    `/proc/1/environ` before returning for non-interactive shells.

    This is intentionally a warning by default.  User dotfiles live outside the
    EGGPU workspace and must not block or be modified by the benchmark workflow.
    Set EGGPU_STRICT_OUTER_SHELL_HYGIENE=1 only for a deliberately strict local
    reproducibility audit.
    """
    strict = _truthy_env("EGGPU_STRICT_OUTER_SHELL_HYGIENE")
    bashrc = Path.home() / ".bashrc"
    if not bashrc.exists():
        emit("outer_shell_hygiene", "ok", bashrc=str(bashrc), note="no .bashrc found")
        return True
    try:
        text = bashrc.read_text(errors="replace")
    except Exception as exc:
        emit(
            "outer_shell_hygiene",
            "fail" if strict else "warn",
            bashrc=str(bashrc),
            error=repr(exc),
            strict=strict,
        )
        return not strict

    active_text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    danger_pos = active_text.find("/proc/1/environ")
    guard_positions = [
        pos for marker in ("case $- in", 'case "$-" in', "return;;")
        if (pos := active_text.find(marker)) >= 0
    ]
    guard_pos = min(guard_positions) if guard_positions else -1
    bad = danger_pos >= 0 and (guard_pos < 0 or danger_pos < guard_pos)
    status = "fail" if (bad and strict) else ("warn" if bad else "ok")
    emit(
        "outer_shell_hygiene",
        status,
        bashrc=str(bashrc),
        reads_proc1_environ=danger_pos >= 0,
        interactive_guard_before_read=guard_pos >= 0 and guard_pos < danger_pos,
        strict=strict,
        note=(
            "outer shell may import /proc/1/environ; benchmark children are still sanitized"
            if bad
            else "shell startup does not expose /proc/1/environ before the non-interactive guard"
        ),
    )
    return not (bad and strict)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().upper() in {"1", "TRUE", "YES", "ON"}


def _parse_nvidia_smi_number(value: str) -> int | None:
    digits = []
    for ch in str(value):
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return None
    try:
        return int("".join(digits))
    except ValueError:
        return None


def check_gpu_preflight_idle_guard() -> bool:
    """Avoid running the small GPU preflight on an occupied shared GPU.

    The official entry script already checks GPU idleness before calling this
    preflight.  This local guard protects direct/manual preflight invocations,
    where touching an already-busy GPU can produce opaque CUDA allocation errors
    and may disturb another user's long-running job.
    """
    if _truthy_env("EGGPU_ALLOW_BUSY_GPU") or _truthy_env("ALLOW_BUSY_GPU"):
        emit(
            "gpu_preflight_idle_guard",
            "ok",
            override=True,
            note="busy-GPU guard explicitly disabled; debug-only",
        )
        return True

    gpu = (
        os.environ.get("EGGPU_MONITOR_GPU_INDEX")
        or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
        or "0"
    ).strip()
    if not gpu:
        gpu = "0"

    max_mem_mb = int(os.environ.get("EGGPU_IDLE_MAX_MEMORY_MB", "1024"))
    max_util = int(os.environ.get("EGGPU_IDLE_MAX_UTILIZATION", "5"))
    smi_env = dict(os.environ)
    # The benchmark process pins LD_LIBRARY_PATH to the user-space CUDA root.
    # nvidia-smi is a driver tool and can fail if it loads incompatible conda
    # libraries first, so query it with the runtime library path removed.
    smi_env.pop("LD_LIBRARY_PATH", None)
    proc = None
    errors = []
    for fmt in ("csv,noheader,nounits", "csv,noheader"):
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    f"--format={fmt}",
                ],
                env=smi_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=10,
            )
        except Exception as exc:
            errors.append(f"{fmt}: {exc!r}")
            continue
        if proc.returncode == 0:
            break
        errors.append(f"{fmt}: {proc.stdout.strip()[-300:]}")

    if proc is None:
        emit(
            "gpu_preflight_idle_guard",
            "fail",
            gpu=gpu,
            errors=errors,
            note="unable to check GPU idleness before GPU preflight",
        )
        return False

    if proc.returncode != 0:
        emit(
            "gpu_preflight_idle_guard",
            "fail",
            gpu=gpu,
            output=proc.stdout.strip()[-500:],
            errors=errors,
            note="nvidia-smi query failed before GPU preflight",
        )
        return False

    found = False
    observed_mem = observed_util = None
    for raw in proc.stdout.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 3 or parts[0] != gpu:
            continue
        found = True
        observed_mem = _parse_nvidia_smi_number(parts[1])
        observed_util = _parse_nvidia_smi_number(parts[2])
        break

    if not found or observed_mem is None or observed_util is None:
        emit(
            "gpu_preflight_idle_guard",
            "fail",
            gpu=gpu,
            output=proc.stdout.strip()[-500:],
            note="target GPU not found or nvidia-smi output was not parseable",
        )
        return False

    ok = observed_mem <= max_mem_mb and observed_util <= max_util
    emit(
        "gpu_preflight_idle_guard",
        "ok" if ok else "fail",
        gpu=gpu,
        memory_mb=observed_mem,
        utilization_percent=observed_util,
        max_memory_mb=max_mem_mb,
        max_utilization_percent=max_util,
        note=(
            "GPU is idle enough for structural GPU preflight"
            if ok
            else "GPU is busy; skip direct preflight or pick an idle GPU"
        ),
    )
    return ok


def run_structural_preflight() -> bool:
    env = dict(os.environ)
    env["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    env["EASYGRAPH_GPU_BACKEND"] = "mine"
    env["PYTHONPATH"] = str(EASYGRAPH_REPO) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cuda_root_raw = (
        env.get("EGGPU_CUDA_ROOT")
        or env.get("CUDA_PATH")
        or env.get("CUDA_HOME")
        or env.get("CUDAToolkit_ROOT")
        or env.get("CONDA_PREFIX")
        or ""
    )
    cuda_root = Path(cuda_root_raw).expanduser() if cuda_root_raw else None
    ld_parts = []
    if cuda_root is not None:
        for candidate in (cuda_root / "lib", cuda_root / "targets/x86_64-linux/lib"):
            if candidate.exists():
                ld_parts.append(str(candidate))
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    if ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    proc = subprocess.run(
        [sys.executable, str(STRUCTURAL_PREFLIGHT)],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="")
    emit(
        "structural_scanv_preflight",
        "ok" if proc.returncode == 0 else "fail",
        returncode=proc.returncode,
    )
    return proc.returncode == 0


def main() -> int:
    checks = [
        check_import_paths(),
        check_compiled_extension_freshness(),
        check_run_script(),
        check_function_registry_consistency(),
        check_child_python_wrapper(),
        check_gpu_routing_contract(),
        check_build_script(),
        check_closeness_cuda_launch_contract(),
        check_unweighted_centrality_bfs_contract(),
        check_public_gpu_dispatch_strict_errors(),
        check_closeness_baseline_semantics_contract(),
        check_backend_separation_static_contract(),
        check_sota_summary_validation_filter_contract(),
        check_validation_reference_oracle_contract(),
        check_gpu_device_profile_metadata_contract(),
        check_audit_metadata_gate(),
        check_audit_validation_gate(),
        check_audit_validation_error_gate(),
        check_timeout_recommendation(),
        check_outer_shell_hygiene(),
        run_closeness_preflight(),
    ]
    gpu_idle = check_gpu_preflight_idle_guard()
    checks.append(gpu_idle)
    checks.append(run_structural_preflight() if gpu_idle else False)
    ok = all(checks)
    emit("full_eval_preflight", "ok" if ok else "fail")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
