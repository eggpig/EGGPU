#!/usr/bin/env python3
"""Recompute paper speedup claims from the authoritative cell ledger."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


CPU_BASELINES = ("networkx", "easygraph-cpu", "easygraph-cpp", "igraph")
GPU_BASELINES = ("nx-cugraph", "Gunrock")


def finite_positive(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def geomean(values):
    values = [float(value) for value in values if finite_positive(value)]
    return math.exp(sum(math.log(value) for value in values) / len(values)) if values else math.nan


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_timeout(row):
    status = (row.get("execution_status") or "").strip().lower()
    failure_kind = (row.get("failure_kind") or "").strip().lower()
    reason = (row.get("reason") or "").strip().lower()
    return (
        status == "timeout"
        or failure_kind == "timeout"
        or "timed out" in reason
        or "timeout after" in reason
        or "exceeded per-function limit" in reason
        or "exceeded its 100-second budget" in reason
    )


def metric_value(row, metric, timeout_seconds=None, value_mode="paper"):
    suffix = "paper" if value_mode == "paper" else "raw_mean"
    field = f"{metric}_{suffix}_seconds"
    if row.get("execution_status") == "ok" and finite_positive(row.get(field)):
        return float(row[field]), "measured"
    if (
        metric == "e2e"
        and timeout_seconds is not None
        and row.get("support_class") in {"T", "P"}
        and is_timeout(row)
    ):
        return float(timeout_seconds), "timeout_cap"
    return None, None


def best(group, names, metric, timeout_seconds=None, value_mode="paper"):
    candidates = []
    for row in group:
        if row.get("baseline") not in names:
            continue
        value, source = metric_value(
            row,
            metric,
            timeout_seconds=timeout_seconds,
            value_mode=value_mode,
        )
        if value is not None:
            candidates.append((value, row["baseline"], source))
    if not candidates:
        return None, [], []
    value = min(item[0] for item in candidates)
    winners = sorted(
        name
        for candidate, name, _ in candidates
        if math.isclose(candidate, value, rel_tol=1e-12, abs_tol=0.0)
    )
    sources = sorted(
        source
        for candidate, _, source in candidates
        if math.isclose(candidate, value, rel_tol=1e-12, abs_tol=0.0)
    )
    return value, winners, sources


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ["dataset", "function"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(rows, metric, timeout_seconds=None, value_mode="paper"):
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("dataset", ""), row.get("function", ""))].append(row)

    records = []
    winner_counts = Counter()
    ratios_external = []
    ratios_cpu = []
    ratios_gpu = []
    intersection = []
    timeout_winner_counts = Counter()
    eggpu_timeout_capped = 0
    for (dataset, function), group in sorted(groups.items()):
        eggpu_value, _, eggpu_sources = best(
            group,
            ("EGGPU",),
            metric,
            timeout_seconds=timeout_seconds,
            value_mode=value_mode,
        )
        if eggpu_value is None:
            continue
        if "timeout_cap" in eggpu_sources:
            eggpu_timeout_capped += 1
        external_value, external_names, external_sources = best(
            group,
            (*CPU_BASELINES, *GPU_BASELINES),
            metric,
            timeout_seconds=timeout_seconds,
            value_mode=value_mode,
        )
        cpu_value, cpu_names, cpu_sources = best(
            group,
            CPU_BASELINES,
            metric,
            timeout_seconds=timeout_seconds,
            value_mode=value_mode,
        )
        gpu_value, gpu_names, gpu_sources = best(
            group,
            GPU_BASELINES,
            metric,
            timeout_seconds=timeout_seconds,
            value_mode=value_mode,
        )
        record = {
            "dataset": dataset,
            "function": function,
            "eggpu_seconds": eggpu_value,
            "best_external_seconds": external_value if external_value is not None else "",
            "best_external": "+".join(external_names),
            "best_external_value_source": "+".join(external_sources),
            "best_cpu_seconds": cpu_value if cpu_value is not None else "",
            "best_cpu": "+".join(cpu_names),
            "best_cpu_value_source": "+".join(cpu_sources),
            "best_gpu_seconds": gpu_value if gpu_value is not None else "",
            "best_gpu": "+".join(gpu_names),
            "best_gpu_value_source": "+".join(gpu_sources),
            "eggpu_value_source": "+".join(eggpu_sources),
            "speedup_external_over_eggpu": external_value / eggpu_value if external_value is not None else "",
            "speedup_cpu_over_eggpu": cpu_value / eggpu_value if cpu_value is not None else "",
            "speedup_gpu_over_eggpu": gpu_value / eggpu_value if gpu_value is not None else "",
        }
        records.append(record)
        if external_value is not None:
            ratios_external.append(external_value / eggpu_value)
            for name in external_names:
                winner_counts[name] += 1
            if "timeout_cap" in external_sources:
                timeout_winner_counts["external"] += 1
        if cpu_value is not None:
            ratios_cpu.append(cpu_value / eggpu_value)
            if "timeout_cap" in cpu_sources:
                timeout_winner_counts["cpu"] += 1
        if gpu_value is not None:
            ratios_gpu.append(gpu_value / eggpu_value)
            if "timeout_cap" in gpu_sources:
                timeout_winner_counts["gpu"] += 1
        if cpu_value is not None and gpu_value is not None:
            intersection.append({
                **record,
                "gpu_over_cpu": gpu_value / cpu_value,
                "cpu_faster_than_gpu": cpu_value < gpu_value,
                "gpu_faster_than_cpu": gpu_value < cpu_value,
            })

    summary = {
        "metric": metric,
        "value_mode": value_mode,
        "timeout_seconds": timeout_seconds,
        "timeout_policy": "completed_only" if timeout_seconds is None else "right_censored_at_timeout",
        "eggpu_timeout_capped_workloads": eggpu_timeout_capped,
        "best_external_common_workloads": len(ratios_external),
        "speedup_over_pairwise_best_external": geomean(ratios_external),
        "best_cpu_common_workloads": len(ratios_cpu),
        "speedup_over_pairwise_best_cpu": geomean(ratios_cpu),
        "best_gpu_common_workloads": len(ratios_gpu),
        "speedup_over_pairwise_best_gpu": geomean(ratios_gpu),
        "best_external_identity_counts": dict(sorted(winner_counts.items())),
        "timeout_capped_winner_counts": dict(sorted(timeout_winner_counts.items())),
        "strict_cpu_gpu_intersection_workloads": len(intersection),
        "strict_intersection_best_gpu_over_best_cpu": geomean(row["gpu_over_cpu"] for row in intersection),
        "strict_intersection_eggpu_speedup_over_best_cpu": geomean(row["speedup_cpu_over_eggpu"] for row in intersection),
        "strict_intersection_eggpu_speedup_over_best_gpu": geomean(row["speedup_gpu_over_eggpu"] for row in intersection),
        "strict_intersection_cpu_faster_count": sum(bool(row["cpu_faster_than_gpu"]) for row in intersection),
        "strict_intersection_gpu_faster_count": sum(bool(row["gpu_faster_than_cpu"]) for row in intersection),
    }
    return records, intersection, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=100.0,
        help="E2E right-censoring threshold used for the separate timeout-capped audit.",
    )
    parser.add_argument(
        "--value-mode",
        choices=("paper", "raw-mean"),
        default="paper",
        help="Use declared paper values or the arithmetic mean retained for every system.",
    )
    args = parser.parse_args()

    rows = read_rows(args.ledger.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    value_mode = args.value_mode.replace("-", "_")
    output_tag = "" if value_mode == "paper" else "_uniform_raw_mean"
    summaries = {}
    for metric in ("e2e", "kernel"):
        records, intersection, summary = audit(rows, metric, value_mode=value_mode)
        summaries[metric] = summary
        write_csv(output / f"{metric}{output_tag}_pairwise_speedup_audit.csv", records)
        write_csv(output / f"{metric}{output_tag}_strict_cpu_gpu_intersection.csv", intersection)

    (output / f"paper_speedup_claim_audit{output_tag}.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    capped_records, capped_intersection, capped_summary = audit(
        rows,
        "e2e",
        timeout_seconds=args.timeout_seconds,
        value_mode=value_mode,
    )
    timeout_tag = f"{args.timeout_seconds:g}s"
    write_csv(
        output / f"e2e{output_tag}_timeout_capped_{timeout_tag}_pairwise_speedup_audit.csv",
        capped_records,
    )
    write_csv(
        output / f"e2e{output_tag}_timeout_capped_{timeout_tag}_strict_cpu_gpu_intersection.csv",
        capped_intersection,
    )
    (output / f"paper_speedup_claim_audit{output_tag}_timeout_capped_{timeout_tag}.json").write_text(
        json.dumps(capped_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    e2e = summaries["e2e"]
    lines = [
        "# Paper Speedup Claim Audit",
        "",
        f"- Value mode: **{value_mode}**.",
        f"- Pairwise best external E2E speedup: **{e2e['speedup_over_pairwise_best_external']:.6f}x** "
        f"over **{e2e['best_external_common_workloads']}** workloads.",
        f"- Pairwise best GPU E2E speedup: **{e2e['speedup_over_pairwise_best_gpu']:.6f}x** "
        f"over **{e2e['best_gpu_common_workloads']}** workloads.",
        f"- Pairwise best CPU E2E speedup: **{e2e['speedup_over_pairwise_best_cpu']:.6f}x** "
        f"over **{e2e['best_cpu_common_workloads']}** workloads.",
        f"- Best-external identities: `{e2e['best_external_identity_counts']}`.",
        "",
        "## Strict Shared-Workload Check",
        "",
        f"On the **{e2e['strict_cpu_gpu_intersection_workloads']}** workloads where EGGPU, "
        "a CPU baseline, and a GPU baseline all have validated E2E values:",
        f"- the pairwise best GPU baseline is **{e2e['strict_intersection_best_gpu_over_best_cpu']:.6f}x** "
        "slower than the pairwise best CPU baseline;",
        f"- the CPU baseline is faster on **{e2e['strict_intersection_cpu_faster_count']}** workloads, "
        f"while the GPU baseline is faster on **{e2e['strict_intersection_gpu_faster_count']}**;",
        f"- EGGPU is **{e2e['strict_intersection_eggpu_speedup_over_best_cpu']:.6f}x** faster than "
        "the best CPU baseline and **"
        f"{e2e['strict_intersection_eggpu_speedup_over_best_gpu']:.6f}x** faster than the best GPU baseline.",
        "",
        "The two headline speedups use explicitly stated valid-workload sets and therefore "
        "must not be interpreted as measurements over one identical denominator set.",
        "",
        f"## Timeout-Capped Sensitivity ({args.timeout_seconds:g}s)",
        "",
        "This sensitivity analysis assigns the timeout threshold to every semantically "
        "supported E2E call that reached the watchdog, including EGGPU timeouts. The value "
        "is a conservative lower bound for a timed-out implementation, not a measured runtime.",
        f"- Pairwise best external E2E speedup: **{capped_summary['speedup_over_pairwise_best_external']:.6f}x** "
        f"over **{capped_summary['best_external_common_workloads']}** workloads.",
        f"- Pairwise best CPU E2E speedup: **{capped_summary['speedup_over_pairwise_best_cpu']:.6f}x** "
        f"over **{capped_summary['best_cpu_common_workloads']}** workloads.",
        f"- Pairwise best GPU E2E speedup: **{capped_summary['speedup_over_pairwise_best_gpu']:.6f}x** "
        f"over **{capped_summary['best_gpu_common_workloads']}** workloads.",
        f"- EGGPU timeout-capped workloads: **{capped_summary['eggpu_timeout_capped_workloads']}**.",
        f"- Timeout-capped winner counts: `{capped_summary['timeout_capped_winner_counts']}`.",
    ]
    (output / f"PAPER_SPEEDUP_CLAIM_AUDIT{output_tag.upper()}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
