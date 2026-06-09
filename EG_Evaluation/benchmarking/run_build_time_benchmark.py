#!/usr/bin/env python3
"""Standalone graph-construction benchmark.

This script measures build/load time independently from algorithm execution.
It intentionally excludes Python import time and raw edge-list parsing time.
For each dataset/function/baseline case, timing starts immediately before the
baseline-native graph/container construction and stops once that object is ready
for the corresponding algorithm call.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.library_baselines import (  # noqa: E402
    build_easygraph,
    build_igraph,
    build_networkx,
    component_semantics,
    deterministic_weighted_edges,
    load_graph,
)
from benchmarking.run_full_baselines import (  # noqa: E402
    DEFAULT_DATASETS,
    DEFAULT_FUNCTIONS,
    LEGACY_FUNCTION_ALIASES,
    parse_csv_tokens,
)


BASELINES = ("networkx", "easygraph-cpu", "easygraph-cpp", "igraph", "nx-cugraph", "EGGPU")


def selected_datasets(tokens: str):
    ds_tokens = {x.lower() for x in parse_csv_tokens(tokens)}
    if not ds_tokens or "all" in ds_tokens:
        return list(DEFAULT_DATASETS)
    out = []
    for size, graph_type, name, path in DEFAULT_DATASETS:
        if name.lower() in ds_tokens or size.lower() in ds_tokens or graph_type.lower() in ds_tokens:
            out.append((size, graph_type, name, path))
    if not out:
        raise SystemExit(f"No datasets matched --datasets={tokens}")
    return out


def selected_functions(tokens: str):
    fn_tokens = parse_csv_tokens(tokens)
    if not fn_tokens or any(x.lower() == "all" for x in fn_tokens):
        return list(DEFAULT_FUNCTIONS)
    out = []
    for token in fn_tokens:
        key = token.strip()
        if key in LEGACY_FUNCTION_ALIASES:
            out.extend(LEGACY_FUNCTION_ALIASES[key])
        elif key in DEFAULT_FUNCTIONS:
            out.append(key)
        else:
            raise SystemExit(f"Unknown function: {token}")
    dedup = []
    seen = set()
    for fn in out:
        if fn not in seen:
            seen.add(fn)
            dedup.append(fn)
    return dedup


def read_coverage(result_dir: str | None):
    if not result_dir:
        return None
    path = Path(result_dir) / "results_e2e.csv"
    if not path.exists():
        raise SystemExit(f"coverage result missing: {path}")
    coverage = set()
    with path.open(newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("status") in {"ok", "timeout"}:
                coverage.add((row["dataset"], row["function"], row["baseline"]))
    return coverage


def build_spec(function: str, graph_type: str, views):
    if function == "PageRank":
        n, directed_edges, undirected_edges = views["clean"]
        directed = graph_type == "directed"
        edges = directed_edges if directed else undirected_edges
        return {
            "spec": "clean-directed" if directed else "clean-undirected",
            "n": n,
            "edges": edges,
            "directed": directed,
            "weighted": False,
            "notes": "PageRank graph view",
        }
    if function == "MST":
        n, _, undirected_edges = views["all_vertices"]
        return {
            "spec": "all-undirected-weighted",
            "n": n,
            "edges": undirected_edges,
            "directed": False,
            "weighted": True,
            "notes": "MST undirected weighted view",
        }
    if function == "LCC":
        n, _, undirected_edges = views["clean"]
        return {
            "spec": "clean-undirected",
            "n": n,
            "edges": undirected_edges,
            "directed": False,
            "weighted": False,
            "notes": "LCC undirected projection",
        }
    if function in {"WCC", "SCC"}:
        plan = component_semantics(function, graph_type, views)
        return {
            "spec": "all-directed" if plan["build_directed"] else "all-undirected",
            "n": plan["n"],
            "edges": plan["edges"],
            "directed": bool(plan["build_directed"]),
            "weighted": False,
            "notes": plan["note"],
        }
    if function in {"SSSP", "Dijkstra", "BellmanFord"}:
        n, directed_edges, undirected_edges = views["clean"]
        directed = graph_type == "directed"
        base_edges = directed_edges if directed else undirected_edges
        return {
            "spec": "clean-directed-weighted" if directed else "clean-undirected-weighted",
            "n": n,
            "edges": deterministic_weighted_edges(n, base_edges),
            "directed": directed,
            "weighted": True,
            "notes": f"{function} weighted deterministic view",
        }
    if function == "BFS":
        n, directed_edges, undirected_edges = views["clean"]
        directed = graph_type == "directed"
        base_edges = directed_edges if directed else undirected_edges
        return {
            "spec": "clean-directed-unweighted" if directed else "clean-undirected-unweighted",
            "n": n,
            "edges": base_edges,
            "directed": directed,
            "weighted": False,
            "notes": "BFS unweighted shortest-path view",
        }
    if function == "KCore":
        n, _, undirected_edges = views["clean"]
        return {
            "spec": "clean-undirected",
            "n": n,
            "edges": undirected_edges,
            "directed": False,
            "weighted": False,
            "notes": "KCore undirected projection",
        }
    if function in {"BC", "Closeness"}:
        n, directed_edges, undirected_edges = views["clean"]
        directed = graph_type == "directed"
        edges = directed_edges if directed else undirected_edges
        return {
            "spec": "clean-directed" if directed else "clean-undirected",
            "n": n,
            "edges": edges,
            "directed": directed,
            "weighted": False,
            "notes": f"{function} graph view",
        }
    if function in {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}:
        n, directed_edges, undirected_edges = views["clean"]
        directed = graph_type == "directed"
        edges = directed_edges if directed else undirected_edges
        return {
            "spec": "clean-directed" if directed else "clean-undirected",
            "n": n,
            "edges": edges,
            "directed": directed,
            "weighted": False,
            "notes": "Burt structural-hole graph view",
        }
    raise ValueError(f"unknown function: {function}")


def build_once(baseline: str, spec):
    n = spec["n"]
    edges = spec["edges"]
    directed = spec["directed"]
    weighted = spec["weighted"]

    if baseline == "networkx":
        return build_networkx(n, edges, directed, weighted=weighted)
    if baseline == "igraph":
        return build_igraph(n, edges, directed, weighted=weighted)
    if baseline == "nx-cugraph":
        # nx-cugraph is benchmarked through the NetworkX dispatch API, so the
        # user-visible graph construction object is still a NetworkX graph.
        return build_networkx(n, edges, directed, weighted=weighted)
    if baseline == "easygraph-cpu":
        os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"
        return build_easygraph(n, edges, directed, weighted=weighted)
    if baseline == "easygraph-cpp":
        os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"
        g = build_easygraph(n, edges, directed, weighted=weighted)
        return g.cpp()
    if baseline == "EGGPU":
        # Fair build/load metric: EGGPU users still construct a normal
        # EasyGraph graph object. GPU GraphContext/C++ cache/device CSR
        # preparation is paid by function calls and is measured in E2E or
        # workflow-reuse ablations, not in graph construction time.
        os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
        os.environ["EASYGRAPH_GPU_BACKEND"] = "mine"
        return build_easygraph(n, edges, directed, weighted=weighted)
    raise ValueError(f"unknown baseline: {baseline}")


def preimport_baselines(baselines):
    """Exclude library import time from all build measurements."""
    if "networkx" in baselines or "nx-cugraph" in baselines:
        import networkx  # noqa: F401
    if "igraph" in baselines:
        import igraph  # noqa: F401
    if any(b in baselines for b in ("easygraph-cpu", "easygraph-cpp", "EGGPU")):
        import easygraph  # noqa: F401
    if "nx-cugraph" in baselines:
        try:
            import nx_cugraph  # noqa: F401
        except Exception:
            # Build-time for nx-cugraph is the NetworkX graph object used by
            # the dispatch API; lack of nx-cugraph import should not block that
            # construction measurement.
            pass


def timed_build(baseline: str, spec, repeat: int, warmup: int):
    for _ in range(max(0, int(warmup))):
        obj = build_once(baseline, spec)
        del obj
        gc.collect()
    values = []
    for _ in range(max(1, int(repeat))):
        gc.collect()
        t0 = time.perf_counter()
        obj = build_once(baseline, spec)
        elapsed = time.perf_counter() - t0
        values.append(elapsed)
        del obj
        gc.collect()
    return values


def summarize(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "median": statistics.median(vals) if vals else None,
        "mean": statistics.mean(vals) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--functions", default="all")
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--coverage-result-dir", default="")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    datasets = selected_datasets(args.datasets)
    functions = selected_functions(args.functions)
    baselines = [b for b in parse_csv_tokens(args.baselines) if b]
    unknown = [b for b in baselines if b not in BASELINES]
    if unknown:
        raise SystemExit(f"Unknown baselines: {unknown}")
    coverage = read_coverage(args.coverage_result_dir or None)
    preimport_baselines(baselines)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_rows = []
    agg_rows = []

    print("[note] imports and raw edge-list parsing are excluded from all build timings.", flush=True)
    print("[note] Gunrock is intentionally excluded: CLI graph loading cannot be isolated.", flush=True)
    print(f"[params] repeat={args.repeat} warmup={args.warmup}", flush=True)

    measured_cache = {}
    for size, graph_type, name, rel_path in datasets:
        print(f"=== build-time dataset {name} ({graph_type}, {size}) ===", flush=True)
        views = load_graph(str(ROOT / rel_path))
        for function in functions:
            spec = build_spec(function, graph_type, views)
            for baseline in baselines:
                if coverage is not None and (name, function, baseline) not in coverage:
                    continue
                cache_key = (
                    name,
                    function,
                    baseline,
                    spec["spec"],
                    int(spec["n"]),
                    bool(spec["directed"]),
                    bool(spec["weighted"]),
                    len(spec["edges"]),
                )
                # Keep the result per function, but do not rebuild if two
                # functions share the same graph view for the same baseline.
                reusable_key = (
                    name,
                    baseline,
                    spec["spec"],
                    int(spec["n"]),
                    bool(spec["directed"]),
                    bool(spec["weighted"]),
                    len(spec["edges"]),
                )
                if reusable_key in measured_cache:
                    values = measured_cache[reusable_key]
                    status = "ok"
                    note = spec["notes"] + "; reused identical build-view measurement"
                else:
                    try:
                        values = timed_build(baseline, spec, args.repeat, args.warmup)
                        measured_cache[reusable_key] = values
                        status = "ok"
                        note = spec["notes"]
                    except Exception as exc:
                        values = []
                        status = "failed"
                        note = spec["notes"] + f"; build failed: {type(exc).__name__}: {exc}"
                for i, value in enumerate(values):
                    long_rows.append(
                        {
                            "dataset_size": size,
                            "graph_type": graph_type,
                            "dataset": name,
                            "function": function,
                            "baseline": baseline,
                            "metric": "build",
                            "repeat": i,
                            "seconds": value,
                            "status": status,
                            "notes": note,
                            "nodes": spec["n"],
                            "edges": len(spec["edges"]),
                            "directed": spec["directed"],
                            "weighted": spec["weighted"],
                            "view": spec["spec"],
                        }
                    )
                summary = summarize(values)
                agg_rows.append(
                    {
                        "dataset_size": size,
                        "graph_type": graph_type,
                        "dataset": name,
                        "function": function,
                        "baseline": baseline,
                        "metric": "build",
                        "seconds": summary["median"] if status == "ok" else "",
                        "status": status,
                        "notes": note + "; standalone build-only median",
                        "mean_seconds": summary["mean"] if status == "ok" else "",
                        "min_seconds": summary["min"] if status == "ok" else "",
                        "max_seconds": summary["max"] if status == "ok" else "",
                        "std_seconds": summary["std"] if status == "ok" else "",
                        "nodes": spec["n"],
                        "edges": len(spec["edges"]),
                        "directed": spec["directed"],
                        "weighted": spec["weighted"],
                        "view": spec["spec"],
                    }
                )

    long_fields = [
        "dataset_size",
        "graph_type",
        "dataset",
        "function",
        "baseline",
        "metric",
        "repeat",
        "seconds",
        "status",
        "notes",
        "nodes",
        "edges",
        "directed",
        "weighted",
        "view",
    ]
    agg_fields = [
        "dataset_size",
        "graph_type",
        "dataset",
        "function",
        "baseline",
        "metric",
        "seconds",
        "status",
        "notes",
        "mean_seconds",
        "min_seconds",
        "max_seconds",
        "std_seconds",
        "nodes",
        "edges",
        "directed",
        "weighted",
        "view",
    ]
    with (out_dir / "build_times_long.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=long_fields)
        writer.writeheader()
        writer.writerows(long_rows)
    with (out_dir / "build_times.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(agg_rows)
    notes = [
        "# Standalone Build-Time Benchmark",
        "",
        "Timing starts after imports and raw edge-list parsing.",
        "Each row measures baseline-native graph/container construction only.",
        "EGGPU build includes only the normal EasyGraph graph-object construction path used by the Python API.",
        "EGGPU GraphContext, C++ graph-cache, device CSR, CUDA context, and warmup costs are intentionally excluded from build time and measured in E2E or workflow-reuse experiments.",
        "EasyGraph-C++ build includes EasyGraph graph construction plus `.cpp()` conversion.",
        "nx-cugraph build uses NetworkX graph construction because the benchmark invokes nx-cugraph through the NetworkX dispatch API.",
        "Gunrock is excluded because graph loading happens inside external CLI binaries and cannot be isolated comparably.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(notes))
    print(f"Done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
