#!/usr/bin/env python3
"""Audit an order-balanced device graph-view registry experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max4-a", type=Path, required=True)
    parser.add_argument("--max1", type=Path, required=True)
    parser.add_argument("--max4-b", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-order-drift", type=float, default=0.25)
    parser.add_argument("--min-robust-speedup", type=float, default=1.05)
    parser.add_argument("--min-transfer-reduction", type=float, default=2.0)
    return parser.parse_args()


def load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = [float(row["sequence_seconds"]) for row in payload.get("samples", [])]
    if not samples:
        raise ValueError(f"{path}: no measured samples")
    payload["_samples"] = samples
    return payload


def trimmed_mean(values, fraction=0.1):
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    if trim == 0 or 2 * trim >= len(ordered):
        return statistics.fmean(ordered)
    return statistics.fmean(ordered[trim:-trim])


def summarize(payload):
    values = payload["_samples"]
    cache = payload.get("cache_stats", {})
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "sample_stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median_seconds": statistics.median(values),
        "trimmed_mean_seconds": trimmed_mean(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "h2d_bytes": int(cache.get("structure_bytes_copied", 0))
        + int(cache.get("weight_bytes_copied", 0)),
        "evictions": int(cache.get("evictions", 0)),
        "active_entries": int(cache.get("active_entries", 0)),
        "retained_device_bytes": int(cache.get("device_bytes", 0)),
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else math.inf


def main():
    args = parse_args()
    a = load(args.max4_a)
    one = load(args.max1)
    b = load(args.max4_b)

    issues = []
    hard_issues = []
    for field in ("dataset", "call_path", "sequence", "repeat", "preheat_sequences"):
        values = [a.get(field), one.get(field), b.get(field)]
        if values[0] != values[1] or values[0] != values[2]:
            hard_issues.append(f"metadata mismatch for {field}: {values}")
    if [a.get("max_entries"), one.get("max_entries"), b.get("max_entries")] != [4, 1, 4]:
        hard_issues.append("expected max-entry order 4,1,4")
    if a.get("result_signatures") != one.get("result_signatures"):
        hard_issues.append("result signatures differ between max4-A and max1")
    if a.get("result_signatures") != b.get("result_signatures"):
        hard_issues.append("result signatures differ between max4-A and max4-B")

    sa = summarize(a)
    s1 = summarize(one)
    sb = summarize(b)
    pooled4 = a["_samples"] + b["_samples"]
    pooled4_summary = {
        "count": len(pooled4),
        "mean_seconds": statistics.fmean(pooled4),
        "sample_stdev_seconds": statistics.stdev(pooled4) if len(pooled4) > 1 else 0.0,
        "median_seconds": statistics.median(pooled4),
        "trimmed_mean_seconds": trimmed_mean(pooled4),
    }

    order_drift = abs(sa["trimmed_mean_seconds"] - sb["trimmed_mean_seconds"]) / min(
        sa["trimmed_mean_seconds"], sb["trimmed_mean_seconds"]
    )
    speedups = {
        "mean": ratio(s1["mean_seconds"], pooled4_summary["mean_seconds"]),
        "median": ratio(s1["median_seconds"], pooled4_summary["median_seconds"]),
        "trimmed_mean": ratio(
            s1["trimmed_mean_seconds"], pooled4_summary["trimmed_mean_seconds"]
        ),
    }
    max4_h2d = max(sa["h2d_bytes"], sb["h2d_bytes"])
    transfer_reduction = ratio(s1["h2d_bytes"], max4_h2d)

    if order_drift > args.max_order_drift:
        issues.append(
            f"max4 A/B trimmed-mean drift {order_drift:.3f} exceeds "
            f"{args.max_order_drift:.3f}"
        )
    if min(speedups["median"], speedups["trimmed_mean"]) < args.min_robust_speedup:
        issues.append(
            "robust speedup is below threshold: "
            f"median={speedups['median']:.3f}, trimmed={speedups['trimmed_mean']:.3f}"
        )
    if transfer_reduction < args.min_transfer_reduction:
        issues.append(
            f"H2D reduction {transfer_reduction:.3f} is below "
            f"{args.min_transfer_reduction:.3f}"
        )

    if hard_issues:
        status = "fail"
    elif issues:
        status = "inconclusive"
    else:
        status = "pass"

    result = {
        "status": status,
        "dataset": a.get("dataset"),
        "call_path": a.get("call_path"),
        "sequence": a.get("sequence"),
        "max4_a": sa,
        "max1": s1,
        "max4_b": sb,
        "max4_pooled": pooled4_summary,
        "order_drift_fraction": order_drift,
        "speedup_max4_over_max1": speedups,
        "h2d_reduction_max4_over_max1": transfer_reduction,
        "thresholds": {
            "max_order_drift": args.max_order_drift,
            "min_robust_speedup": args.min_robust_speedup,
            "min_transfer_reduction": args.min_transfer_reduction,
        },
        "hard_issues": hard_issues,
        "issues": issues,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
