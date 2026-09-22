#!/usr/bin/env python3
"""Verify that the nvitop visibility allocation does not slow EGGPU calls."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from gpu_visibility_marker import GpuVisibilityMarker


ROOT = Path(__file__).resolve().parents[1]


def paired_ratios(rows, trials):
    paired = []
    for pair_index in range(trials):
        pair = rows[pair_index * 2 : pair_index * 2 + 2]
        by_mode = {row["marker"]: row for row in pair}
        if len(pair) != 2 or set(by_mode) != {"off", "on"}:
            raise ValueError(f"invalid marker A/B pair at index {pair_index}: {pair}")
        paired.append(
            {
                "pair_index": pair_index + 1,
                "order": "->".join(row["marker"] for row in pair),
                "e2e_on_over_off": (
                    by_mode["on"]["e2e_median_seconds"]
                    / by_mode["off"]["e2e_median_seconds"]
                ),
                "kernel_on_over_off": (
                    by_mode["on"]["kernel_median_seconds"]
                    / by_mode["off"]["kernel_median_seconds"]
                ),
            }
        )
    return paired


def worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["EGGPU_MONITOR_GPU_INDEX"] = str(args.gpu)
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"
    import easygraph as eg
    from library_baselines import build_easygraph, load_graph
    from easygraph.utils import gpu_eggpu_backend

    views = load_graph(str(ROOT / args.edge_path))
    n, _, undirected = views["clean"]
    graph = build_easygraph(n, undirected, False, weighted=False)
    gpu_eggpu_backend._graph_context(graph, prewarm_cpp=True)

    def call():
        return eg.pagerank(
            graph,
            alpha=0.75,
            max_iter=200,
            tol=1.0e-6,
            weight=None,
        )

    for _ in range(args.warmup):
        call()
    e2e = []
    kernel = []
    for _ in range(args.calls):
        gpu_eggpu_backend.set_last_kernel_time("pagerank", None)
        started = time.perf_counter()
        result = call()
        e2e.append(time.perf_counter() - started)
        value = gpu_eggpu_backend.get_last_kernel_time("pagerank")
        if value is None:
            raise SystemExit("missing PageRank CUDA-event timing")
        kernel.append(float(value))
        if len(result) != n:
            raise SystemExit(f"invalid PageRank result length {len(result)} != {n}")
    payload = {
        "marker": args.marker,
        "marker_started": os.environ.get(
            "EGGPU_NEUTRALITY_PARENT_MARKER_STARTED", "FALSE"
        ).strip().upper() in {"1", "TRUE", "ON", "YES"},
        "marker_error": os.environ.get(
            "EGGPU_NEUTRALITY_PARENT_MARKER_ERROR", ""
        ) or None,
        "marker_process_scope": "parent_process",
        "e2e_median_seconds": statistics.median(e2e),
        "kernel_median_seconds": statistics.median(kernel),
        "e2e_samples": len(e2e),
        "kernel_samples": len(kernel),
    }
    print("MARKER_RESULT " + json.dumps(payload, sort_keys=True), flush=True)


def trial(args):
    """Model the production topology: parent marker, isolated timed child."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["EGGPU_MONITOR_GPU_INDEX"] = str(args.gpu)
    os.environ["EGGPU_GPU_VISIBILITY_MARKER"] = (
        "TRUE" if args.marker == "on" else "FALSE"
    )
    os.environ["EGGPU_GPU_VISIBILITY_MARKER_MB"] = str(
        args.marker_mb if args.marker == "on" else 0
    )
    os.environ["EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB"] = "AUTO"
    marker = GpuVisibilityMarker(
        args.gpu, f"neutrality parent marker={args.marker}"
    ).start()
    if args.marker == "on" and not marker.started:
        raise SystemExit(f"visibility marker failed to start: {marker.error}")

    os.environ["EGGPU_NEUTRALITY_PARENT_MARKER_STARTED"] = (
        "TRUE" if marker.started else "FALSE"
    )
    os.environ["EGGPU_NEUTRALITY_PARENT_MARKER_ERROR"] = marker.error or ""
    if marker.started:
        os.environ["EGGPU_EXTERNAL_VISIBILITY_MARKER"] = "TRUE"
    else:
        os.environ.pop("EGGPU_EXTERNAL_VISIBILITY_MARKER", None)

    command = [
        sys.executable,
        __file__,
        "--worker",
        "--gpu",
        str(args.gpu),
        "--edge-path",
        args.edge_path,
        "--marker",
        args.marker,
        "--marker-mb",
        str(args.marker_mb),
        "--warmup",
        str(args.warmup),
        "--calls",
        str(args.calls),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(os.environ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(completed.stdout, end="", flush=True)
        if completed.returncode != 0:
            raise SystemExit(
                f"isolated benchmark worker failed with exit code {completed.returncode}"
            )
    finally:
        marker.stop()


def controller(args):
    rows = []
    # Alternate AB/BA pair order so process startup, clock drift, and thermal
    # drift cannot be systematically attributed to the marker-on condition.
    pair_orders = [
        (["off", "on"] if trial % 2 == 0 else ["on", "off"])
        for trial in range(args.trials)
    ]
    sequence = [mode for pair in pair_orders for mode in pair]
    for index, marker in enumerate(sequence, start=1):
        command = [
            sys.executable,
            __file__,
            "--trial",
            "--gpu",
            str(args.gpu),
            "--edge-path",
            args.edge_path,
            "--marker",
            marker,
            "--marker-mb",
            str(args.marker_mb),
            "--warmup",
            str(args.warmup),
            "--calls",
            str(args.calls),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(os.environ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(f"[marker-check] trial {index}/{len(sequence)} marker={marker}", flush=True)
        print(completed.stdout, end="", flush=True)
        if completed.returncode != 0:
            raise SystemExit(f"marker {marker} worker failed with exit code {completed.returncode}")
        payload = None
        for line in completed.stdout.splitlines():
            if line.startswith("MARKER_RESULT "):
                payload = json.loads(line.split(" ", 1)[1])
        if payload is None:
            raise SystemExit("worker emitted no MARKER_RESULT")
        if marker == "on" and not payload.get("marker_started"):
            raise SystemExit(f"visibility marker failed to start: {payload.get('marker_error')}")
        rows.append(payload)

    off = [row for row in rows if row["marker"] == "off"]
    on = [row for row in rows if row["marker"] == "on"]
    try:
        paired = paired_ratios(rows, args.trials)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    summary = {
        "gpu": args.gpu,
        "trials_per_mode": args.trials,
        "calls_per_trial": args.calls,
        "marker_mb": args.marker_mb,
        "marker_topology": "parent_process_marker_with_isolated_timed_child",
        "off_e2e_median_seconds": statistics.median(row["e2e_median_seconds"] for row in off),
        "on_e2e_median_seconds": statistics.median(row["e2e_median_seconds"] for row in on),
        "off_kernel_median_seconds": statistics.median(row["kernel_median_seconds"] for row in off),
        "on_kernel_median_seconds": statistics.median(row["kernel_median_seconds"] for row in on),
        "max_allowed_ratio": args.max_ratio,
        "aggregation": "median_of_paired_on_over_off_ratios",
        "pair_order_policy": "alternating_AB_BA",
        "paired_ratios": paired,
        "rows": rows,
    }
    summary["unpaired_e2e_on_over_off_diagnostic"] = (
        summary["on_e2e_median_seconds"] / summary["off_e2e_median_seconds"]
    )
    summary["unpaired_kernel_on_over_off_diagnostic"] = (
        summary["on_kernel_median_seconds"] / summary["off_kernel_median_seconds"]
    )
    summary["e2e_on_over_off"] = statistics.median(
        pair["e2e_on_over_off"] for pair in paired
    )
    summary["kernel_on_over_off"] = statistics.median(
        pair["kernel_on_over_off"] for pair in paired
    )
    summary["passed"] = (
        summary["e2e_on_over_off"] <= args.max_ratio
        and summary["kernel_on_over_off"] <= args.max_ratio
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not summary["passed"]:
        raise SystemExit("visibility marker exceeds the configured timing-neutrality threshold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--edge-path", default="datasets/undirected/ca-GrQc.txt")
    parser.add_argument("--marker", choices=["on", "off"], default="off")
    parser.add_argument("--marker-mb", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-ratio", type=float, default=1.05)
    parser.add_argument("--out", default="benchmarking/results/visibility_marker_neutrality.json")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    elif args.trial:
        trial(args)
    else:
        controller(args)


if __name__ == "__main__":
    main()
