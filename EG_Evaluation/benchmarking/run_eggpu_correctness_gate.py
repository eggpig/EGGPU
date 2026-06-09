#!/usr/bin/env python3
"""Focused EGGPU correctness gate.

This runner is intentionally narrower than run_full_baselines.py.  It compares
EGGPU against EasyGraph CPU on selected small/medium datasets and writes the
same validation artifacts as the full benchmark.  The goal is to catch semantic
or return-path regressions quickly without waiting for slow optional baselines
such as NetworkX structural holes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

from run_full_baselines import (
    DEFAULT_DATASETS,
    DEFAULT_FUNCTIONS,
    DEFAULT_LOCAL_CUDA_ROOT,
    LEGACY_FUNCTION_ALIASES,
    ROOT,
    conda_python_cmd,
    parse_csv_tokens,
    parse_library_results,
    read_text,
    run_cmd,
    sanitized_subprocess_env,
)
from validate_correctness import write_validation_outputs


RESULT_FIELDS = (
    "dataset_size",
    "graph_type",
    "dataset",
    "function",
    "baseline",
    "metric",
    "seconds",
    "status",
    "log",
    "correctness",
    "notes",
)


def _select_datasets(tokens):
    wanted = {x.lower() for x in parse_csv_tokens(tokens)}
    if "none" in wanted:
        return []
    if not wanted or "default" in wanted:
        wanted = {"ca-grqc", "p2p-gnutella08"}
    if "all" in wanted:
        return list(DEFAULT_DATASETS)
    out = []
    for item in DEFAULT_DATASETS:
        size, graph_type, name, _ = item
        keys = {size.lower(), graph_type.lower(), name.lower()}
        if keys & wanted:
            out.append(item)
    if not out:
        raise SystemExit(f"No datasets matched --datasets={tokens!r}")
    return out


def _write_synthetic_datasets(out_dir):
    """Create small adversarial graphs for correctness regression testing."""
    root = out_dir / "synthetic_inputs"
    root.mkdir(parents=True, exist_ok=True)

    # High-degree undirected ego graph: stresses hierarchy/effective-size
    # neighborhood reductions and BC back-propagation on many same-level nodes.
    high_degree = root / "synthetic_high_degree_undirected.txt"
    with high_degree.open("w") as f:
        for v in range(1, 73):
            f.write(f"0 {v}\n")
        for v in range(1, 72):
            f.write(f"{v} {v + 1}\n")
        for v in range(1, 73, 3):
            w = 1 + (v + 9) % 72
            if v != w:
                f.write(f"{v} {w}\n")
        # one disconnected triangle to exercise component and MST forest paths
        f.write("100 101\n101 102\n102 100\n")

    # Directed asymmetric graph: node 0 has incoming edges but no outgoing edge,
    # which is the structural-hole NaN corner case fixed in the CUDA path.
    directed_asym = root / "synthetic_directed_asymmetry.txt"
    with directed_asym.open("w") as f:
        for u in range(1, 48):
            f.write(f"{u} 0\n")
        for u in range(1, 47):
            f.write(f"{u} {u + 1}\n")
        for u in range(1, 48, 4):
            f.write(f"{u} {1 + (u * 7) % 47}\n")
        f.write("80 81\n81 82\n82 80\n")

    return [
        ("synthetic", "undirected", "synthetic-high-degree", str(high_degree)),
        ("synthetic", "directed", "synthetic-directed-asym", str(directed_asym)),
    ]


def _select_functions(tokens):
    wanted = parse_csv_tokens(tokens)
    if not wanted or any(x.lower() == "all" for x in wanted):
        return list(DEFAULT_FUNCTIONS)
    out = []
    for tok in wanted:
        alias = LEGACY_FUNCTION_ALIASES.get(tok.upper())
        if alias is not None:
            for name in alias:
                if name not in out:
                    out.append(name)
            continue
        hit = next((f for f in DEFAULT_FUNCTIONS if f.lower() == tok.lower()), None)
        if hit is None:
            raise SystemExit(
                f"Unknown function token: {tok}. Choose from {', '.join(DEFAULT_FUNCTIONS)}, CC, or all."
            )
        if hit not in out:
            out.append(hit)
    return out


def _base_env(gpu, easygraph_repo):
    env = sanitized_subprocess_env(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["EGGPU_MONITOR_GPU_INDEX"] = str(gpu)
    env["EGGPU_STRICT_VALIDATION"] = "TRUE"
    env["EGGPU_ALLOW_CUDA_SYNC"] = "TRUE"
    env["EASYGRAPH_GPU_BACKEND"] = "mine"
    env["EASYGRAPH_GPU_ADAPTIVE_POLICY"] = env.get("EASYGRAPH_GPU_ADAPTIVE_POLICY", "TRUE")
    env["EASYGRAPH_GPU_COMPONENT_DENSE_RETURN"] = env.get(
        "EASYGRAPH_GPU_COMPONENT_DENSE_RETURN", "FALSE"
    )
    env["EASYGRAPH_GPU_SCC_ACTIVE_TRIM"] = env.get("EASYGRAPH_GPU_SCC_ACTIVE_TRIM", "TRUE")
    env["EASYGRAPH_GPU_SCC_ACTIVE_TRIM_MAX_ITERS"] = env.get(
        "EASYGRAPH_GPU_SCC_ACTIVE_TRIM_MAX_ITERS", "16"
    )
    env["EASYGRAPH_GPU_SCC_DEGREE_PIVOT"] = env.get("EASYGRAPH_GPU_SCC_DEGREE_PIVOT", "TRUE")
    env["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = env.get("EASYGRAPH_GPU_SCC_HOST_ENABLE", "FALSE")
    env["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = env.get("EASYGRAPH_GPU_KCORE_HOST_ENABLE", "FALSE")
    env["EASYGRAPH_GPU_KCORE_HOST_MAX_EDGE_SLOTS"] = env.get(
        "EASYGRAPH_GPU_KCORE_HOST_MAX_EDGE_SLOTS", "0"
    )
    env["EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_AVG_DEGREE"] = env.get(
        "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_AVG_DEGREE", "10"
    )
    env["EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE"] = env.get(
        "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE", "AUTO"
    )
    env["EASYGRAPH_GPU_SSSP_FRONTIER"] = env.get("EASYGRAPH_GPU_SSSP_FRONTIER", "TRUE")
    env["EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION"] = env.get(
        "EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION", "TRUE"
    )

    cuda_root = os.environ.get("EGGPU_CUDA_ROOT", "").strip()
    if not cuda_root and DEFAULT_LOCAL_CUDA_ROOT is not None and DEFAULT_LOCAL_CUDA_ROOT.exists():
        cuda_root = str(DEFAULT_LOCAL_CUDA_ROOT)
    if cuda_root:
        env["EGGPU_CUDA_ROOT"] = cuda_root
        env["CUDA_PATH"] = cuda_root
        env["CUDA_HOME"] = cuda_root
        env["CUPY_CUDA_PATH"] = cuda_root
        env["CUDAToolkit_ROOT"] = cuda_root
        lib_paths = [
            str(Path(cuda_root) / "lib"),
            str(Path(cuda_root) / "targets" / "x86_64-linux" / "lib"),
        ]
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([p for p in lib_paths if Path(p).exists()] + ([old_ld] if old_ld else []))

    if easygraph_repo:
        old_pp = env.get("PYTHONPATH", "")
        repo = str(Path(easygraph_repo).resolve())
        env["PYTHONPATH"] = repo if not old_pp else repo + ":" + old_pp
    return env


def _add_row(rows, size, graph_type, dataset, result, log_path, rc, timeout):
    status = result.get("status", "failed")
    notes = result.get("notes", "")
    seconds = result.get("seconds")
    if rc == 124:
        status = "failed"
        seconds = None
        notes = (notes + "; " if notes else "") + f"TIMEOUT_TOO_LONG: exceeded correctness gate timeout ({timeout}s)"
    elif rc != 0 and status == "ok":
        status = "failed"
        notes = (notes + "; " if notes else "") + f"subprocess exit code {rc}"
    rows.append(
        {
            "dataset_size": size,
            "graph_type": graph_type,
            "dataset": dataset,
            "function": result.get("function", ""),
            "baseline": result.get("backend", ""),
            "metric": result.get("metric", "e2e"),
            "seconds": "" if seconds is None else float(seconds),
            "status": status,
            "log": str(log_path),
            "correctness": result.get("correctness", ""),
            "notes": notes,
        }
    )


def _run_one(
    rows,
    base_env,
    out_dir,
    size,
    graph_type,
    dataset,
    edge_path,
    backend,
    function,
    timeout,
    pr_alpha,
    pr_tol,
    pr_max_iter,
    easygraph_warmup,
    sssp_sources,
    bc_sources,
):
    log_dir = out_dir / "logs" / dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{backend}_{function}.log"
    env = dict(base_env)
    env["EGGPU_VALIDATION_DETAIL_DIR"] = str((out_dir / "details" / dataset).resolve())
    if backend == "EGGPU":
        env["EASYGRAPH_ENABLE_GPU"] = "TRUE"
        env["EASYGRAPH_GPU_BACKEND"] = "mine"
        backend_arg = "mine"
        warmup = 0
        eg_warmup = easygraph_warmup
    else:
        env["EASYGRAPH_ENABLE_GPU"] = "FALSE"
        env["EASYGRAPH_GPU_BACKEND"] = ""
        backend_arg = ""
        warmup = 0
        eg_warmup = 0
    cmd = conda_python_cmd(
        "benchmarking/library_baselines.py",
        edge_path,
        graph_type,
        "--backend",
        backend,
        "--function",
        function,
        "--pr-alpha",
        str(pr_alpha),
        "--pr-tol",
        str(pr_tol),
        "--pr-max-iter",
        str(pr_max_iter),
        "--warmup",
        str(warmup),
        "--easygraph-warmup",
        str(eg_warmup),
        "--easygraph-gpu-backend",
        backend_arg,
        "--sssp-sources",
        str(sssp_sources),
        "--bc-sources",
        str(bc_sources),
        "--cooldown",
        "0",
    )
    rc, _, _ = run_cmd(cmd, log_path, env, timeout=timeout, cooldown=0)
    results = parse_library_results(read_text(log_path))
    if not results:
        for metric in ("build", "e2e", "kernel"):
            rows.append(
                {
                    "dataset_size": size,
                    "graph_type": graph_type,
                    "dataset": dataset,
                    "function": function,
                    "baseline": backend,
                    "metric": metric,
                    "seconds": "",
                    "status": "failed",
                    "log": str(log_path),
                    "correctness": "",
                    "notes": f"no RESULT_JSON rows; exit={rc}",
                }
            )
        return
    for result in results:
        _add_row(rows, size, graph_type, dataset, result, log_path, rc, timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--datasets", default="default")
    ap.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Also run small adversarial synthetic graphs for high-degree and directed asymmetry corner cases.",
    )
    ap.add_argument("--functions", default="all")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--pr-alpha", type=float, default=0.75)
    ap.add_argument("--pr-tol", type=float, default=1e-6)
    ap.add_argument("--pr-max-iter", type=int, default=200)
    ap.add_argument("--easygraph-repo", default=str(ROOT.parent / "Easy-Graph"))
    ap.add_argument("--easygraph-warmup", type=int, default=1)
    ap.add_argument("--sssp-sources", type=int, default=4)
    ap.add_argument("--bc-sources", type=int, default=4)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else ROOT / "benchmarking" / "results" / f"eggpu_correctness_gate_gpu{args.gpu}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = _select_datasets(args.datasets)
    if args.include_synthetic:
        datasets.extend(_write_synthetic_datasets(out_dir))
    functions = _select_functions(args.functions)
    base_env = _base_env(args.gpu, args.easygraph_repo)
    rows = []

    for size, graph_type, dataset, edge_path in datasets:
        print(f"=== correctness dataset {dataset} ({graph_type}, {size}) ===", flush=True)
        for function in functions:
            # EasyGraph CPU is the semantic reference for every EGGPU public API.
            for backend in ("easygraph-cpu", "EGGPU"):
                _run_one(
                    rows,
                    base_env,
                    out_dir,
                    size,
                    graph_type,
                    dataset,
                    edge_path,
                    backend,
                    function,
                    args.timeout,
                    args.pr_alpha,
                    args.pr_tol,
                    args.pr_max_iter,
                    args.easygraph_warmup,
                    args.sssp_sources,
                    args.bc_sources,
                )

    with (out_dir / "results_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    validation = write_validation_outputs(out_dir, rows)
    counts = {}
    for row in validation:
        counts[row["validation_status"]] = counts.get(row["validation_status"], 0) + 1
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "datasets": [d[2] for d in datasets],
                "functions": functions,
                "validation_counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Done: {out_dir}")
    print(f"Validation counts: {counts}")


if __name__ == "__main__":
    main()
