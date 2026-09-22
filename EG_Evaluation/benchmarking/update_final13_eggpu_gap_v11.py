#!/usr/bin/env python3
"""Replace the eight stale GAP-twitter EGGPU cells with qualified v11 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


REPLACEMENTS = {
    "MST": "projection",
    "LCC": "projection",
    "KCore": "projection",
    "SCC": "scc",
    "EffectiveSize": "effective_size",
    "Efficiency": "efficiency",
    "Constraint": "constraint",
    "Hierarchy": "hierarchy",
}

REASONS = {
    "MST": (
        "generation-2 logical undirected projection stores each unique edge once; "
        "the returned minimum spanning forest passed exact external weight validation"
    ),
    "LCC": (
        "degree-oriented logical undirected projection avoids materializing a duplicated "
        "symmetric CSR while preserving exact neighbor-intersection semantics"
    ),
    "KCore": (
        "single-copy logical undirected projection removes the prior signed-int32 "
        "symmetric-CSR limit and preserves peeling semantics"
    ),
    "SCC": (
        "all-node directed SCC completes after reusable state preparation; the steady "
        "five-call window remains within the 100-second main-call budget"
    ),
    "EffectiveSize": (
        "shared degree-oriented ego-edge statistics replace repeated all-node ego "
        "neighborhood intersections and preserve the public result semantics"
    ),
    "Efficiency": (
        "reuses the shared Effective Size statistics before applying degree normalization"
    ),
    "Constraint": (
        "shared per-edge direct and indirect constraint terms replace repeated all-node "
        "ego neighborhood intersections"
    ),
    "Hierarchy": (
        "reuses shared per-edge constraint terms and applies the hierarchy reduction "
        "without reconstructing ego neighborhoods"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ledger", required=True, type=Path)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--scc", required=True, type=Path)
    parser.add_argument("--effective-size", required=True, type=Path)
    parser.add_argument("--efficiency", required=True, type=Path)
    parser.add_argument("--constraint", required=True, type=Path)
    parser.add_argument("--hierarchy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric field: {value!r}")
    return result


def qualified_record(source: Path, function: str) -> tuple[dict[str, object], dict]:
    summary_path = source / "scaling_all.csv"
    rows = read_csv(summary_path)
    timing = [
        row for row in rows
        if row.get("dataset") == "GAP-twitter"
        and row.get("function") == function
        and row.get("measurement") == "timing"
    ]
    memory = [
        row for row in rows
        if row.get("dataset") == "GAP-twitter"
        and row.get("function") == function
        and row.get("measurement") == "memory"
        and row.get("status") == "ok"
    ]
    if len(timing) != 1:
        raise ValueError(f"{function}: timing rows={len(timing)}, expected=1")
    timing_row = timing[0]
    if timing_row.get("status") != "ok":
        raise ValueError(f"{function}: status={timing_row.get('status')}")
    if timing_row.get("result_validation") != "pass":
        raise ValueError(
            f"{function}: validation={timing_row.get('result_validation')}"
        )
    if int(number(timing_row.get("timing_process_samples", 0))) != 1:
        raise ValueError(
            f"{function}: timing_process_samples="
            f"{timing_row.get('timing_process_samples')}"
        )
    if len(memory) != 3:
        raise ValueError(f"{function}: memory rows={len(memory)}, expected=3")
    if any(row.get("result_validation") != "pass" for row in memory):
        raise ValueError(f"{function}: a memory repetition failed validation")

    aggregate_path = source / "raw" / f"GAP-twitter_{function}_timing.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    e2e = aggregate["steady_e2e"]
    kernel = aggregate["steady_kernel"]
    load = aggregate["load"]
    if len(e2e.get("samples", [])) != 5 or len(kernel.get("samples", [])) != 5:
        raise ValueError(
            f"{function}: expected five internal timing samples, got "
            f"e2e={len(e2e.get('samples', []))}, "
            f"kernel={len(kernel.get('samples', []))}"
        )

    gpu_values = [
        number(row.get("gpu_proc_peak_mb") or row.get("gpu_proc_peak_delta_mb"))
        for row in memory
    ]
    rss_values = [
        number(row.get("rss_peak_mb") or row.get("rss_peak_delta_mb"))
        for row in memory
    ]
    update = {
        "support_class": "T",
        "execution_status": "ok",
        "failure_kind": "",
        "reason": REASONS[function],
        "e2e_paper_seconds": number(e2e["best"]),
        "e2e_raw_mean_seconds": number(e2e["mean"]),
        "e2e_std_seconds": number(e2e["stdev"]),
        "e2e_estimator": "best_observed_of_five_in_raw_aggregate",
        "validation_status": "pass",
        "result_source": str(source.resolve()),
        "sample_count": 5,
        "build_paper_seconds": number(load["best"]),
        "build_raw_mean_seconds": number(load["mean"]),
        "build_std_seconds": number(load["stdev"]),
        "build_estimator": "minimum_of_five_bulk_csr_load",
        "kernel_paper_seconds": number(kernel["best"]),
        "kernel_raw_mean_seconds": number(kernel["mean"]),
        "kernel_std_seconds": number(kernel["stdev"]),
        "kernel_estimator": "best_observed_of_five_in_raw_aggregate",
        "gpu_peak_mb_mean": statistics.mean(gpu_values),
        "gpu_peak_mb_std": statistics.stdev(gpu_values),
        "memory_sample_count": 3,
        "host_rss_peak_mb_mean": statistics.mean(rss_values),
        "host_rss_peak_mb_std": statistics.stdev(rss_values),
        "memory_measurement_window": "isolated_worker_process_full_lifetime",
        "excluded_observed_status": "",
    }
    provenance = {
        "function": function,
        "source_directory": str(source.resolve()),
        "summary_csv": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
        "timing_aggregate": str(aggregate_path.resolve()),
        "timing_aggregate_sha256": sha256(aggregate_path),
        "steady_e2e_samples": e2e["samples"],
        "steady_kernel_samples": kernel["samples"],
        "gpu_memory_samples_mib": gpu_values,
        "host_rss_samples_mib": rss_values,
    }
    return update, provenance


def main() -> None:
    args = parse_args()
    source_by_role = {
        "projection": args.projection,
        "scc": args.scc,
        "effective_size": args.effective_size,
        "efficiency": args.efficiency,
        "constraint": args.constraint,
        "hierarchy": args.hierarchy,
    }
    with args.input_ledger.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    qualified = {}
    provenance = []
    for function, role in REPLACEMENTS.items():
        update, evidence = qualified_record(source_by_role[role], function)
        qualified[function] = update
        provenance.append(evidence)

    replaced = []
    for row in rows:
        if row.get("category") == "Paths & Spanning Trees":
            row["category"] = "Path & Spanning"
        if (
            row.get("dataset") == "GAP-twitter"
            and row.get("baseline") == "EGGPU"
            and row.get("function") in qualified
        ):
            row.update({key: str(value) for key, value in qualified[row["function"]].items()})
            replaced.append(row["function"])

    if set(replaced) != set(REPLACEMENTS) or len(replaced) != len(REPLACEMENTS):
        raise ValueError(f"replacement set mismatch: {replaced}")
    if len(rows) != 13 * 16 * 8:
        raise ValueError(f"ledger cardinality={len(rows)}, expected={13 * 16 * 8}")
    eggpu = [row for row in rows if row.get("baseline") == "EGGPU"]
    ok_count = sum(row.get("execution_status") == "ok" for row in eggpu)
    if len(eggpu) != 208 or ok_count != 208:
        raise ValueError(f"EGGPU matrix is not complete: rows={len(eggpu)}, ok={ok_count}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "final_13_cell_outcome_ledger.csv"
    write_csv(output, rows, fields)
    audit = {
        "status": "pass",
        "input_ledger": str(args.input_ledger.resolve()),
        "input_ledger_sha256": sha256(args.input_ledger),
        "output_ledger": str(output.resolve()),
        "output_ledger_sha256": sha256(output),
        "ledger_cells": len(rows),
        "eggpu_cells": len(eggpu),
        "eggpu_ok_cells": ok_count,
        "replaced_functions": sorted(replaced),
        "provenance": provenance,
    }
    (args.output_dir / "EGGPU_GAP_V11_LEDGER_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
