#!/usr/bin/env python3
"""Create a new final ledger with the strict nx-cugraph 99-cell overlay.

The source ledger is immutable: the output path must be different.  The script
verifies the strict audit, all five raw samples, per-cell timing provenance,
validation placement, and an objective extreme-variance gate before replacing
the 99 previously successful nx-cugraph timing cells in memory.  It then writes
the new ledger and its manifest with atomic filesystem replacements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path


FUNCTIONS = (
    "PageRank",
    "LCC",
    "WCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
)
MODERATE_DATASETS = (
    "ca-HepTh",
    "LastFM",
    "p2p-Gnutella04",
    "ca-HepPh",
    "email-Enron",
    "ca-CondMat",
    "soc-Epinions1",
    "soc-Slashdot0811",
    "ER-100k",
    "web-NotreDame",
    "com-youtube",
)
EXPECTED_KEYS = frozenset(
    {(dataset, function) for dataset in MODERATE_DATASETS for function in FUNCTIONS}
    | {("com-Orkut", function) for function in FUNCTIONS}
    | {
        ("GAP-twitter", "PageRank"),
        ("GAP-twitter", "BFS"),
        ("GAP-twitter", "Dijkstra"),
    }
)
METRIC_COLUMNS = {
    "construction": "build",
    "e2e": "e2e",
    "processing": "kernel",
}
EXTREME_MAX_OVER_MEDIAN = 20.0


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def parse_json(value, label):
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: invalid JSON") from exc


def finite_positive(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label}: expected a finite positive number")
    return number


def sample_stats(values):
    values = [float(value) for value in values]
    median = statistics.median(values)
    mean = statistics.mean(values)
    sample_std = statistics.stdev(values)
    return {
        "minimum": min(values),
        "mean": mean,
        "sample_std": sample_std,
        "median": median,
        "maximum": max(values),
        "coefficient_of_variation": sample_std / mean if mean else 0.0,
        "max_over_median": max(values) / median,
        "median_over_minimum": median / min(values),
    }


def validate_provenance(row, key):
    label = f"{key[0]}/{key[1]}"
    if row.get("display_estimator") != "minimum_of_five":
        raise ValueError(f"{label}: display estimator is not minimum_of_five")
    if row.get("validation_status") != "pass":
        raise ValueError(f"{label}: validation status is not pass")
    if row.get("timer_kind") != "cuda_event_at_pylibcugraph_backend":
        raise ValueError(f"{label}: processing is not a backend device interval")
    if row.get("measurement_window") != "device_execution":
        raise ValueError(f"{label}: processing window is not device_execution")
    provenance = parse_json(row.get("timing_provenance"), f"{label}/provenance")
    required_true = (
        "validation_outside_timer",
        "public_return_boundary",
        "prepared_native_graph",
        "networkx_cache_converted_graphs",
        "gpu_exclusive_pre_and_post_snapshots",
    )
    for field in required_true:
        if provenance.get(field) is not True:
            raise ValueError(f"{label}: provenance field {field} is not true")
    if provenance.get("networkx_fallback_to_nx") is not False:
        raise ValueError(f"{label}: CPU fallback is not disabled")
    if int(provenance.get("warmup_calls", -1)) != 3:
        raise ValueError(f"{label}: warmup count is not three")
    if int(provenance.get("fresh_process_samples", 0)) != 5:
        raise ValueError(f"{label}: fresh process count is not five")
    samples = provenance.get("samples") or []
    if len(samples) != 5:
        raise ValueError(f"{label}: expected five raw provenance records")
    for index, sample in enumerate(samples, 1):
        if int(sample.get("sample_index", -1)) != index:
            raise ValueError(f"{label}: provenance sample ordering is incomplete")
        before = sample.get("gpu_exclusive_snapshot_before_worker")
        after = sample.get("gpu_exclusive_snapshot_after_worker")
        if not before or not after:
            raise ValueError(f"{label}/sample{index}: missing GPU snapshots")
        if isinstance(before, dict):
            if before.get("exclusive") is not True or before.get("compute_processes"):
                raise ValueError(f"{label}/sample{index}: non-exclusive preflight")
            if after.get("exclusive") is not True or after.get("compute_processes"):
                raise ValueError(f"{label}/sample{index}: non-exclusive postflight")
        else:
            if "gpu_idle_before_eggpu_child:" not in str(before):
                raise ValueError(f"{label}/sample{index}: invalid preflight snapshot")
            if "gpu_idle_before_eggpu_child:" not in str(after):
                raise ValueError(f"{label}/sample{index}: invalid postflight snapshot")
    return provenance


def validate_strict_rows(rows):
    by_key = {}
    metric_samples = {}
    metric_stats = {}
    provenance = {}
    for row in rows:
        if row.get("baseline") != "nx-cugraph":
            raise ValueError("strict overlay contains a non-nx-cugraph baseline")
        key = (row.get("dataset"), row.get("function"))
        if key in by_key:
            raise ValueError(f"duplicate strict overlay key: {key}")
        by_key[key] = row
        provenance[key] = validate_provenance(row, key)
        metric_samples[key] = {}
        metric_stats[key] = {}
        for strict_name in METRIC_COLUMNS:
            label = f"{key[0]}/{key[1]}/{strict_name}"
            samples = parse_json(row.get(f"{strict_name}_samples"), label)
            if len(samples) != 5:
                raise ValueError(f"{label}: expected five samples")
            values = [
                finite_positive(value, f"{label}/sample{index}")
                for index, value in enumerate(samples, 1)
            ]
            reported_min = finite_positive(
                row.get(f"{strict_name}_min_seconds"), f"{label}/min"
            )
            reported_mean = finite_positive(
                row.get(f"{strict_name}_mean_seconds"), f"{label}/mean"
            )
            reported_std = float(row.get(f"{strict_name}_sample_std_seconds"))
            computed = sample_stats(values)
            if not math.isclose(
                reported_min, computed["minimum"], rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(f"{label}: reported minimum does not match samples")
            if not math.isclose(
                reported_mean, computed["mean"], rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(f"{label}: reported mean does not match samples")
            if not math.isclose(
                reported_std, computed["sample_std"], rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(
                    f"{label}: reported sample standard deviation does not match"
                )
            metric_samples[key][strict_name] = values
            metric_stats[key][strict_name] = computed
        for index, (device, e2e) in enumerate(
            zip(metric_samples[key]["processing"], metric_samples[key]["e2e"]), 1
        ):
            if device > e2e:
                raise ValueError(
                    f"{key[0]}/{key[1]}/sample{index}: processing exceeds E2E"
                )
            if math.isclose(device, e2e, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"{key[0]}/{key[1]}/sample{index}: wall time copied to processing"
                )
    if set(by_key) != EXPECTED_KEYS:
        raise ValueError(
            "strict overlay key set mismatch: "
            f"missing={sorted(EXPECTED_KEYS - set(by_key))}, "
            f"extra={sorted(set(by_key) - EXPECTED_KEYS)}"
        )
    return by_key, metric_samples, metric_stats, provenance


def apply_overlay(ledger_rows, strict_rows, strict_source):
    by_key, _, metric_stats, _ = validate_strict_rows(strict_rows)
    ledger_success_keys = {
        (row.get("dataset"), row.get("function"))
        for row in ledger_rows
        if row.get("baseline") == "nx-cugraph"
        and row.get("execution_status") == "ok"
    }
    if ledger_success_keys != EXPECTED_KEYS:
        raise ValueError(
            "source ledger nx-cugraph success set is not the expected 99 cells"
        )

    output = []
    replaced = []
    extreme = []
    timing_note = (
        "strict native nx-cugraph graph; cache_converted_graphs=True; "
        "fallback_to_nx=False; three untimed warmups; five fresh-process "
        "samples; processing uses CUDA events at the pylibcugraph backend; "
        "validation is outside all timers"
    )
    variance_policy = (
        "five_fresh_processes; display=minimum; report=mean+sample-SD; "
        f"extreme_gate=max/median<={EXTREME_MAX_OVER_MEDIAN:g}"
    )
    for original in ledger_rows:
        row = dict(original)
        key = (row.get("dataset"), row.get("function"))
        if row.get("baseline") == "nx-cugraph" and key in by_key:
            strict = by_key[key]
            row["reason"] = timing_note
            row["execution_status"] = "ok"
            row["failure_kind"] = ""
            row["validation_status"] = "pass"
            row["result_source"] = (
                "strict nx-cugraph 99-cell rerun; minimum of five"
            )
            row["timing_result_source"] = str(strict_source)
            row["sample_count"] = "5"
            for strict_name, ledger_prefix in METRIC_COLUMNS.items():
                values = metric_stats[key][strict_name]
                row[f"{ledger_prefix}_paper_seconds"] = repr(values["minimum"])
                row[f"{ledger_prefix}_raw_mean_seconds"] = repr(values["mean"])
                row[f"{ledger_prefix}_std_seconds"] = repr(values["sample_std"])
                row[f"{ledger_prefix}_estimator"] = "minimum_of_five"
                row[f"{ledger_prefix}_raw_min_seconds"] = repr(values["minimum"])
                row[f"{ledger_prefix}_raw_median_seconds"] = repr(values["median"])
                row[f"{ledger_prefix}_raw_max_seconds"] = repr(values["maximum"])
                row[f"{ledger_prefix}_coefficient_of_variation"] = repr(
                    values["coefficient_of_variation"]
                )
                row[f"{ledger_prefix}_max_over_median"] = repr(
                    values["max_over_median"]
                )
                row[f"{ledger_prefix}_median_over_minimum"] = repr(
                    values["median_over_minimum"]
                )
                row[f"{ledger_prefix}_variance_policy"] = variance_policy
                stable = values["max_over_median"] <= EXTREME_MAX_OVER_MEDIAN
                row[f"{ledger_prefix}_stability_status"] = (
                    "pass" if stable else "extreme_variance"
                )
                if not stable:
                    extreme.append(
                        {
                            "dataset": key[0],
                            "function": key[1],
                            "metric": ledger_prefix,
                            "max_over_median": values["max_over_median"],
                        }
                    )
            replaced.append(key)
        output.append(row)

    if len(replaced) != 99 or set(replaced) != EXPECTED_KEYS:
        raise ValueError(f"replaced {len(replaced)} cells instead of exactly 99")
    if extreme:
        raise ValueError(
            "strict overlay failed the extreme-variance gate: "
            + json.dumps(extreme, ensure_ascii=False, sort_keys=True)
        )
    return output, {
        "replaced_cell_count": len(replaced),
        "replaced_keys": [
            {"dataset": dataset, "function": function}
            for dataset, function in sorted(replaced)
        ],
        "extreme_variance_threshold_max_over_median": EXTREME_MAX_OVER_MEDIAN,
        "extreme_variance_cells": extreme,
    }


def atomic_write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--strict-csv", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--out-ledger", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    out_manifest = args.out_manifest or args.out_ledger.with_suffix(
        args.out_ledger.suffix + ".manifest.json"
    )
    protected_inputs = {
        args.ledger.resolve(),
        args.strict_csv.resolve(),
        args.audit_manifest.resolve(),
    }
    if args.out_ledger.resolve() in protected_inputs:
        raise SystemExit("refusing to overwrite any source artifact")
    if out_manifest.resolve() in protected_inputs | {args.out_ledger.resolve()}:
        raise SystemExit("manifest output collides with another artifact")
    source_ledger_sha = sha256_file(args.ledger)
    audit = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    if audit.get("status") != "pass":
        raise SystemExit("strict 99-cell audit manifest is not passing")
    if int(audit.get("observed_success_cells", 0)) != 99:
        raise SystemExit("strict audit manifest does not contain 99 cells")
    expected_sample_records = len(EXPECTED_KEYS) * 5
    if int(audit.get("sample_row_count", 0)) != expected_sample_records:
        raise SystemExit("strict audit manifest does not contain 495 samples")
    samples_csv = Path(str(audit.get("samples_csv") or ""))
    if not samples_csv.is_file():
        raise SystemExit("strict audit raw sample CSV is missing")
    samples_sha = sha256_file(samples_csv)
    if audit.get("samples_csv_sha256") != samples_sha:
        raise SystemExit(
            "strict raw sample CSV SHA256 does not match the audit manifest"
        )
    if (
        args.out_ledger.resolve() == samples_csv.resolve()
        or out_manifest.resolve() == samples_csv.resolve()
    ):
        raise SystemExit("output collides with the strict raw sample artifact")
    strict_sha = sha256_file(args.strict_csv)
    if audit.get("merged_csv_sha256") != strict_sha:
        raise SystemExit("strict CSV SHA256 does not match the audit manifest")

    ledger_rows, ledger_fields = read_csv(args.ledger)
    strict_rows, _ = read_csv(args.strict_csv)
    output_rows, overlay_report = apply_overlay(
        ledger_rows, strict_rows, args.strict_csv.resolve()
    )
    memory_fields = [
        field
        for field in ledger_fields
        if (
            "memory" in field.lower()
            or field.startswith("gpu_peak_")
            or field.startswith("host_rss_")
        )
    ]
    memory_columns_preserved = all(
        all(before.get(field) == after.get(field) for field in memory_fields)
        for before, after in zip(ledger_rows, output_rows)
    )
    non_target_rows_preserved = all(
        before == after
        for before, after in zip(ledger_rows, output_rows)
        if not (
            before.get("baseline") == "nx-cugraph"
            and (before.get("dataset"), before.get("function")) in EXPECTED_KEYS
        )
    )
    if not memory_columns_preserved:
        raise SystemExit("overlay unexpectedly changed a memory column")
    if not non_target_rows_preserved:
        raise SystemExit("overlay unexpectedly changed a non-target row")
    atomic_write_csv(args.out_ledger, output_rows, ledger_fields)
    if sha256_file(args.ledger) != source_ledger_sha:
        raise SystemExit("source ledger changed during overlay creation")
    manifest = {
        "status": "pass",
        "source_ledger": str(args.ledger.resolve()),
        "source_ledger_sha256": source_ledger_sha,
        "source_ledger_preserved": True,
        "strict_csv": str(args.strict_csv.resolve()),
        "strict_csv_sha256": strict_sha,
        "strict_samples_csv": str(samples_csv.resolve()),
        "strict_samples_csv_sha256": samples_sha,
        "strict_sample_row_count": expected_sample_records,
        "strict_audit_manifest": str(args.audit_manifest.resolve()),
        "strict_audit_manifest_sha256": sha256_file(args.audit_manifest),
        "output_ledger": str(args.out_ledger.resolve()),
        "output_ledger_sha256": sha256_file(args.out_ledger),
        "atomic_write": True,
        "timing_and_provenance_columns_replaced_only": True,
        "non_target_rows_preserved": non_target_rows_preserved,
        "memory_columns_preserved": memory_columns_preserved,
        "preserved_memory_columns": memory_fields,
        "display_estimator": "minimum_of_five",
        "three_segment_timing_estimators": {
            "construction": "minimum_of_five",
            "processing": "minimum_of_five",
            "e2e": "minimum_of_five",
        },
        "fresh_process_samples_per_cell": 5,
        "fresh_process_sample_records": len(EXPECTED_KEYS) * 5,
        "reported_statistics": ["minimum", "mean", "sample_standard_deviation"],
        "validation_outside_timer": True,
        "networkx_cache_converted_graphs": True,
        "networkx_fallback_to_nx": False,
        "processing_boundary": "pylibcugraph_backend_device_interval",
        "device_timer_provenance": (
            "CUDA events at the actual pylibcugraph backend boundary on an "
            "explicit RAFT-bound stream"
        ),
        "gpu_exclusive_pre_and_post_snapshots_per_worker": True,
        **overlay_report,
    }
    atomic_write_json(out_manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
