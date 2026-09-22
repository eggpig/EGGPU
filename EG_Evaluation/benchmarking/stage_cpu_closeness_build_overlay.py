#!/usr/bin/env python3
"""Recover sampled-Closeness CPU construction timing from original logs.

The authoritative sampled-Closeness CSVs intentionally retained only kernel
and E2E rows, although every referenced invocation log also contains its
native ``baseline_graph_construction`` RESULT_JSON record.  This script
strictly validates those same-invocation records and publishes a new staged
ledger.  It never mutates the input ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


DATASETS = {
    "ER-100k",
    "com-youtube",
    "soc-Slashdot0811",
    "web-NotreDame",
}
BASELINES = {"networkx", "easygraph-cpu", "easygraph-cpp", "igraph"}
FUNCTION = "Closeness"
EXPECTED_CELLS = {
    (dataset, FUNCTION, baseline)
    for dataset in DATASETS
    for baseline in BASELINES
}
EXPECTED_SAMPLE_INDICES = {1, 2, 3, 4, 5}
VALIDATION_FIELDS = {
    "validation_status": "pass",
    "semantic": "sampled_target_exact",
    "estimator_kind": "exact_selected_vertices",
    "sample_sources": "16",
    "source_policy": "deterministic_evenly_spaced",
}


class EvidenceError(ValueError):
    """Raised before publication when source evidence is incomplete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def positive(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{context}: nonnumeric value {value!r}") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise EvidenceError(f"{context}: expected positive finite seconds")
    return result


def close(lhs: float, rhs: float) -> bool:
    return math.isclose(lhs, rhs, rel_tol=1.0e-9, abs_tol=1.0e-12)


def cell_key(row: dict) -> tuple[str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("function", ""),
        row.get("baseline", ""),
    )


def load_metric_samples(
    path: Path, expected_metric: str
) -> dict[tuple[str, str, str], dict[int, dict]]:
    _, rows = read_csv(path)
    grouped: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        key = cell_key(row)
        if key not in EXPECTED_CELLS:
            continue
        if (
            row.get("metric") != expected_metric
            or row.get("status") != "ok"
            or row.get("semantic") != "sampled_target_exact"
            or row.get("estimator_kind") != "exact_selected_vertices"
            or row.get("sample_sources") != "16"
        ):
            raise EvidenceError(f"{path.name} {key}: invalid metric/status semantics")
        index = int(row.get("sample_index", ""))
        if index in grouped[key]:
            raise EvidenceError(f"{path.name} {key}: duplicate sample {index}")
        positive(row.get("seconds"), f"{path.name} {key} sample {index}")
        grouped[key][index] = row
    if set(grouped) != EXPECTED_CELLS:
        raise EvidenceError(
            f"{path.name}: target cells differ; missing="
            f"{sorted(EXPECTED_CELLS - set(grouped))}"
        )
    for key, samples in grouped.items():
        if set(samples) != EXPECTED_SAMPLE_INDICES:
            raise EvidenceError(
                f"{path.name} {key}: sample indices={sorted(samples)}"
            )
    return grouped


def load_validation(path: Path) -> dict[tuple[str, str, str], dict[int, dict]]:
    _, rows = read_csv(path)
    grouped: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        key = cell_key(row)
        if key not in EXPECTED_CELLS:
            continue
        for field, expected in VALIDATION_FIELDS.items():
            if row.get(field) != expected:
                raise EvidenceError(
                    f"validation {key}: {field}={row.get(field)!r}"
                )
        index = int(row.get("sample_index", ""))
        if index in grouped[key]:
            raise EvidenceError(f"validation {key}: duplicate sample {index}")
        grouped[key][index] = row
    if set(grouped) != EXPECTED_CELLS:
        raise EvidenceError("validation does not cover exactly 16 target cells")
    for key, samples in grouped.items():
        if set(samples) != EXPECTED_SAMPLE_INDICES:
            raise EvidenceError(
                f"validation {key}: sample indices={sorted(samples)}"
            )
    return grouped


def parse_invocation_log(path: Path, baseline: str) -> dict[str, dict]:
    if not path.is_file():
        raise EvidenceError(f"referenced invocation log is absent: {path}")
    selected: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("RESULT_JSON "):
                continue
            payload = json.loads(line[len("RESULT_JSON ") :])
            if (
                payload.get("backend") == baseline
                and payload.get("function") == FUNCTION
                and payload.get("metric") in {"build", "kernel", "e2e"}
            ):
                selected[payload["metric"]].append(payload)
    if set(selected) != {"build", "kernel", "e2e"} or any(
        len(values) != 1 for values in selected.values()
    ):
        raise EvidenceError(
            f"{path}: expected exactly one build/kernel/e2e RESULT_JSON"
        )
    result = {metric: values[0] for metric, values in selected.items()}
    for metric, payload in result.items():
        if payload.get("status") != "ok":
            raise EvidenceError(f"{path}: {metric} status is not ok")
        positive(payload.get("seconds"), f"{path} {metric}")
    if (
        result["build"].get("measurement_scope")
        != "baseline_graph_construction"
        or result["build"].get("measurement_window") != "graph_construction"
        or result["build"].get("timer_kind") != "perf_counter_wall"
    ):
        raise EvidenceError(f"{path}: construction boundary/provenance mismatch")
    if result["kernel"].get("measurement_scope") != "algorithm_compute":
        raise EvidenceError(f"{path}: CPU processing boundary mismatch")
    if result["e2e"].get("measurement_scope") != "user_visible_function_call":
        raise EvidenceError(f"{path}: CPU E2E boundary mismatch")
    if not close(
        float(result["kernel"]["seconds"]), float(result["e2e"]["seconds"])
    ):
        raise EvidenceError(f"{path}: CPU processing does not equal public call")
    return result


def summarize(values: list[float]) -> dict[str, float]:
    if len(values) != 5 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise EvidenceError("summary requires five positive finite samples")
    mean = statistics.mean(values)
    median = statistics.median(values)
    sample_sd = statistics.stdev(values)
    return {
        "mean": mean,
        "sample_sd": sample_sd,
        "minimum": min(values),
        "median": median,
        "maximum": max(values),
        "cv": sample_sd / mean,
        "max_over_median": max(values) / median,
        "median_over_minimum": median / min(values),
    }


def format_float(value: float) -> str:
    return f"{value:.12g}"


def build_evidence(
    evidence_dir: Path,
) -> tuple[list[dict], list[dict], list[dict], dict[str, str]]:
    e2e_path = evidence_dir / "closeness_large_sampled_e2e.csv"
    kernel_path = evidence_dir / "closeness_large_sampled_kernel.csv"
    validation_path = evidence_dir / "closeness_large_sampled_validation.csv"
    for path in (e2e_path, kernel_path, validation_path):
        if not path.is_file():
            raise EvidenceError(f"required evidence is absent: {path}")
    e2e = load_metric_samples(e2e_path, "e2e")
    kernel = load_metric_samples(kernel_path, "kernel")
    validation = load_validation(validation_path)
    aggregate_rows = []
    raw_rows = []
    cell_records = []
    for key in sorted(EXPECTED_CELLS):
        dataset, function, baseline = key
        builds = []
        e2e_values = []
        kernel_values = []
        logs = []
        source_nodes_sha = set()
        for index in range(1, 6):
            e2e_row = e2e[key][index]
            kernel_row = kernel[key][index]
            validation_row = validation[key][index]
            e2e_log = Path(e2e_row.get("log", "")).resolve()
            kernel_log = Path(kernel_row.get("log", "")).resolve()
            if e2e_log != kernel_log:
                raise EvidenceError(f"{key} sample {index}: E2E/kernel logs differ")
            invocation = parse_invocation_log(e2e_log, baseline)
            source_e2e = positive(
                e2e_row.get("seconds"), f"{key} sample {index} E2E CSV"
            )
            source_kernel = positive(
                kernel_row.get("seconds"), f"{key} sample {index} kernel CSV"
            )
            if not close(source_e2e, float(invocation["e2e"]["seconds"])):
                raise EvidenceError(f"{key} sample {index}: E2E CSV/log mismatch")
            if not close(source_kernel, float(invocation["kernel"]["seconds"])):
                raise EvidenceError(f"{key} sample {index}: kernel CSV/log mismatch")
            if not close(source_e2e, source_kernel):
                raise EvidenceError(
                    f"{key} sample {index}: CPU processing != public call"
                )
            build_seconds = float(invocation["build"]["seconds"])
            builds.append(build_seconds)
            e2e_values.append(source_e2e)
            kernel_values.append(source_kernel)
            logs.append(str(e2e_log))
            source_nodes_sha.add(validation_row["source_nodes_sha"])
            raw_rows.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "baseline": baseline,
                    "sample_index": index,
                    "construction_seconds": format_float(build_seconds),
                    "processing_seconds": format_float(source_kernel),
                    "e2e_seconds": format_float(source_e2e),
                    "construction_measurement_scope": (
                        invocation["build"]["measurement_scope"]
                    ),
                    "construction_measurement_window": (
                        invocation["build"]["measurement_window"]
                    ),
                    "construction_timer_kind": invocation["build"]["timer_kind"],
                    "processing_equals_e2e": "true",
                    "validation_status": "pass",
                    "semantic": "sampled_target_exact",
                    "source_nodes_sha": validation_row["source_nodes_sha"],
                    "log": str(e2e_log),
                    "log_sha256": sha256_file(e2e_log),
                }
            )
        if len(source_nodes_sha) != 1:
            raise EvidenceError(f"{key}: source-node identity changed across samples")
        build_summary = summarize(builds)
        e2e_summary = summarize(e2e_values)
        kernel_summary = summarize(kernel_values)
        aggregate_rows.append(
            {
                "dataset": dataset,
                "function": function,
                "baseline": baseline,
                "construction_paper_seconds": format_float(
                    build_summary["mean"]
                ),
                "construction_raw_mean_seconds": format_float(
                    build_summary["mean"]
                ),
                "construction_sample_sd_seconds": format_float(
                    build_summary["sample_sd"]
                ),
                "construction_min_seconds": format_float(
                    build_summary["minimum"]
                ),
                "construction_median_seconds": format_float(
                    build_summary["median"]
                ),
                "construction_max_seconds": format_float(
                    build_summary["maximum"]
                ),
                "construction_cv": format_float(build_summary["cv"]),
                "construction_max_over_median": format_float(
                    build_summary["max_over_median"]
                ),
                "construction_median_over_minimum": format_float(
                    build_summary["median_over_minimum"]
                ),
                "construction_estimator": "arithmetic_mean",
                "dispersion_estimator": "sample_standard_deviation_n_minus_1",
                "sample_count": "5",
                "processing_paper_seconds": format_float(kernel_summary["mean"]),
                "e2e_paper_seconds": format_float(e2e_summary["mean"]),
                "processing_equals_e2e": "true",
                "validation_status": "pass",
                "semantic": "sampled_target_exact",
                "source_nodes_sha": next(iter(source_nodes_sha)),
                "construction_samples_json": json.dumps(
                    builds, separators=(",", ":")
                ),
                "sample_logs_json": json.dumps(logs, separators=(",", ":")),
            }
        )
        cell_records.append(
            {
                "dataset": dataset,
                "function": function,
                "baseline": baseline,
                "construction_samples": builds,
                "construction_summary": build_summary,
                "processing_samples": kernel_values,
                "e2e_samples": e2e_values,
                "processing_equals_e2e": True,
                "logs": [
                    {"path": path, "sha256": sha256_file(Path(path))}
                    for path in logs
                ],
                "source_nodes_sha": next(iter(source_nodes_sha)),
                "validation": "five_of_five_pass",
            }
        )
    evidence_hashes = {
        str(e2e_path.resolve()): sha256_file(e2e_path),
        str(kernel_path.resolve()): sha256_file(kernel_path),
        str(validation_path.resolve()): sha256_file(validation_path),
    }
    return aggregate_rows, raw_rows, cell_records, evidence_hashes


def overlay_ledger(
    base_path: Path,
    aggregate_rows: list[dict],
    manifest_future_path: Path,
    evidence_dir: Path,
) -> tuple[list[str], list[dict]]:
    fields, rows = read_csv(base_path)
    aggregate = {cell_key(row): row for row in aggregate_rows}
    if set(aggregate) != EXPECTED_CELLS:
        raise EvidenceError("aggregate overlay does not contain exactly 16 cells")
    indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if cell_key(row) in EXPECTED_CELLS:
            indices[cell_key(row)].append(index)
    if set(indices) != EXPECTED_CELLS or any(
        len(value) != 1 for value in indices.values()
    ):
        raise EvidenceError("base ledger does not contain each target exactly once")
    result = [dict(row) for row in rows]
    for key, positions in indices.items():
        index = positions[0]
        old = rows[index]
        source = aggregate[key]
        if (
            old.get("execution_status") != "ok"
            or old.get("validation_status") != "pass"
            or int(float(old.get("sample_count", ""))) != 5
        ):
            raise EvidenceError(f"{key}: base ledger cell is not five-run validated")
        if old.get("build_paper_seconds", "") not in {"", None}:
            raise EvidenceError(f"{key}: construction is already populated")
        for prefix, source_field in (
            ("kernel", "processing_paper_seconds"),
            ("e2e", "e2e_paper_seconds"),
        ):
            observed = positive(
                old.get(f"{prefix}_paper_seconds"), f"{key} base {prefix}"
            )
            expected = float(source[source_field])
            if not close(observed, expected):
                raise EvidenceError(
                    f"{key}: base {prefix}={observed} differs from evidence "
                    f"mean={expected}"
                )
        if not close(
            float(old["kernel_paper_seconds"]), float(old["e2e_paper_seconds"])
        ):
            raise EvidenceError(f"{key}: base CPU processing != public call")
        staged = dict(old)
        staged["build_paper_seconds"] = source["construction_paper_seconds"]
        staged["build_raw_mean_seconds"] = source[
            "construction_raw_mean_seconds"
        ]
        staged["build_std_seconds"] = source[
            "construction_sample_sd_seconds"
        ]
        staged["build_estimator"] = source["construction_estimator"]
        staged["build_raw_min_seconds"] = source["construction_min_seconds"]
        staged["build_raw_median_seconds"] = source[
            "construction_median_seconds"
        ]
        staged["build_raw_max_seconds"] = source["construction_max_seconds"]
        staged["build_coefficient_of_variation"] = source["construction_cv"]
        staged["build_max_over_median"] = source[
            "construction_max_over_median"
        ]
        staged["build_median_over_minimum"] = source[
            "construction_median_over_minimum"
        ]
        staged["build_variance_policy"] = source["dispersion_estimator"]
        staged["build_stability_status"] = "pass"
        staged["result_source"] = (
            "authoritative sampled-Closeness five-run evidence; construction "
            "recovered from the same invocation logs"
        )
        staged["timing_result_source"] = str(evidence_dir.resolve())
        staged["reason"] = (
            "CPU construction is native baseline_graph_construction from the "
            "same five validated sampled-target Closeness invocations; CPU "
            "processing equals public-call E2E by the declared timing rule"
        )
        staged["timing_overlay_manifest"] = str(manifest_future_path)
        staged["timing_protocol_version"] = (
            "cpu_same_invocation_closeness_triplet_v1"
        )
        staged["timing_external_cli_wall_used"] = "false"
        result[index] = staged

    # The overlay is deliberately limited to the 16 target rows.
    for index, (old, new) in enumerate(zip(rows, result)):
        if cell_key(old) not in EXPECTED_CELLS and old != new:
            raise EvidenceError(f"non-target ledger row {index} was modified")
    return fields, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ledger", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    base_path = args.base_ledger.resolve(strict=True)
    evidence_dir = args.evidence_dir.resolve(strict=True)
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite staged output: {out_dir}")
    if out_dir == base_path.parent:
        raise EvidenceError("output directory collides with the input ledger")

    aggregate_rows, raw_rows, cell_records, evidence_hashes = build_evidence(
        evidence_dir
    )
    stage = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent)
    )
    try:
        ledger_name = "final_13_cell_outcome_ledger.csv"
        overlay_name = "cpu_closeness_build_overlay.csv"
        raw_name = "cpu_closeness_build_raw_samples.csv"
        manifest_name = "CPU_CLOSENESS_BUILD_OVERLAY_MANIFEST_20260730.json"
        future_manifest = out_dir / manifest_name
        fields, output_rows = overlay_ledger(
            base_path, aggregate_rows, future_manifest, evidence_dir
        )
        extras = sorted({field for row in output_rows for field in row} - set(fields))
        output_fields = fields + extras
        ledger_path = stage / ledger_name
        overlay_path = stage / overlay_name
        raw_path = stage / raw_name
        manifest_path = stage / manifest_name
        write_csv(ledger_path, output_fields, output_rows)
        write_csv(overlay_path, list(aggregate_rows[0]), aggregate_rows)
        write_csv(raw_path, list(raw_rows[0]), raw_rows)
        manifest = {
            "schema_version": "cpu_closeness_build_overlay_v1",
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "status": "pass",
            "base_ledger": str(base_path),
            "base_ledger_sha256": sha256_file(base_path),
            "output_ledger": str(out_dir / ledger_name),
            "output_ledger_sha256": sha256_file(ledger_path),
            "base_rows": len(output_rows),
            "output_rows": len(output_rows),
            "target_cell_count": len(EXPECTED_CELLS),
            "raw_sample_count": len(raw_rows),
            "non_target_rows_unchanged": len(output_rows) - len(EXPECTED_CELLS),
            "target_cells": [
                "\t".join(key) for key in sorted(EXPECTED_CELLS)
            ],
            "overlay_csv": str(out_dir / overlay_name),
            "overlay_csv_sha256": sha256_file(overlay_path),
            "raw_samples_csv": str(out_dir / raw_name),
            "raw_samples_csv_sha256": sha256_file(raw_path),
            "evidence_files": evidence_hashes,
            "construction_contract": {
                "source": "same invocation RESULT_JSON build record",
                "measurement_scope": "baseline_graph_construction",
                "measurement_window": "graph_construction",
                "timer_kind": "perf_counter_wall",
                "display_estimator": "arithmetic_mean_of_five",
                "dispersion_estimator": "sample_standard_deviation_n_minus_1",
            },
            "cpu_processing_contract": (
                "processing equals the public function-call E2E interval; "
                "equality is verified per source invocation"
            ),
            "inference_policy": (
                "No timing value is inferred: construction, processing, and "
                "E2E are parsed from the same original five invocation logs."
            ),
            "publication_policy": (
                "V11 is immutable; publish only this new V12 staged directory."
            ),
            "cells": cell_records,
        }
        write_json(manifest_path, manifest)
        os.replace(stage, out_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
