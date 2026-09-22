#!/usr/bin/env python3
"""Audit and merge the strict 99-cell nx-cugraph timing replacement."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
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
MODERATE_SHARDS = (
    "g0_small_a",
    "g1_small_b",
    "g2_medium_a",
    "g4_medium_b",
    "g5_large_a",
    "g6_youtube",
)
LARGE_FUNCTIONS = {
    "com-Orkut": FUNCTIONS,
    "GAP-twitter": ("PageRank", "BFS", "Dijkstra"),
}
BACKENDS = {
    "PageRank": ("pagerank",),
    "LCC": ("triangle_count",),
    "WCC": ("weakly_connected_components",),
    "BFS": ("bfs",),
    "Dijkstra": ("sssp",),
    "BellmanFord": ("sssp",),
    "SSSP": ("sssp",),
    "KCore": ("core_number",),
}
STABILITY_MAX_OVER_MEDIAN = 20.0


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_contract_reference(path: Path, needle: str, raw_field: str):
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        line = next(
            index
            for index, text in enumerate(lines, 1)
            if needle in text
        )
    except StopIteration as exc:
        raise RuntimeError(
            f"source contract needle not found in {path}: {needle!r}"
        ) from exc
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "line": line,
        "source_expression": needle,
        "raw_field": raw_field,
    }


def parse_provenance(value):
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def is_positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def is_true(value):
    if value is True:
        return True
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def expected_count(function):
    if function in {"PageRank", "LCC", "WCC", "Dijkstra", "KCore"}:
        return 1
    return 8


def check_provenance(provenance, function, errors, label):
    timer = provenance.get("timer") or {}
    if timer.get("timer") != "cuda_events_at_pylibcugraph_backend_boundary":
        errors.append(f"{label}: wrong timer boundary")
    if set(timer.get("backend_functions") or []) != set(BACKENDS[function]):
        errors.append(f"{label}: wrong backend functions")
    if int(timer.get("backend_interval_count", 0)) != expected_count(function):
        errors.append(f"{label}: wrong backend interval count")
    if int(timer.get("resource_handle_factory_calls", 0)) <= 0:
        errors.append(f"{label}: explicit-stream ResourceHandle was not observed")
    if provenance.get("prepared_native_graph") is not True:
        errors.append(f"{label}: graph is not marked prepared-native")
    if provenance.get("validation_outside_timer") is not True:
        errors.append(f"{label}: validation is not outside timer")
    if provenance.get("public_return_boundary") is not True:
        errors.append(f"{label}: E2E is not the public-return boundary")
    if provenance.get("networkx_fallback_to_nx") is not False:
        errors.append(f"{label}: CPU fallback is not disabled")
    if provenance.get("networkx_cache_converted_graphs") is not True:
        errors.append(f"{label}: NetworkX converted-graph cache is not enabled")
    if int(provenance.get("warmup_calls", -1)) != 3:
        errors.append(f"{label}: warmup count is not three")
    warmups = provenance.get("warmup_details") or []
    if len(warmups) != 3:
        errors.append(f"{label}: expected three warmup records")
    for index, warmup in enumerate(warmups, 1):
        e2e = warmup.get("e2e_seconds")
        device = warmup.get("device_seconds")
        if (
            not is_positive(e2e)
            or not is_positive(device)
            or float(device) > float(e2e)
        ):
                errors.append(f"{label}: invalid warmup {index} timing")


def check_moderate_gpu_snapshot(row, errors, label):
    if not is_true(row.get("gpu_exclusive_preflight_verified")):
        errors.append(f"{label}: missing verified GPU preflight")
    if not is_true(row.get("gpu_exclusive_postflight_verified")):
        errors.append(f"{label}: missing verified GPU postflight")
    before = str(row.get("gpu_exclusive_snapshot_before_worker") or "")
    after = str(row.get("gpu_exclusive_snapshot_after_worker") or "")
    if "gpu_idle_before_eggpu_child:" not in before:
        errors.append(f"{label}: missing GPU preflight snapshot")
    if "gpu_idle_before_eggpu_child:" not in after:
        errors.append(f"{label}: missing GPU postflight snapshot")


def check_large_gpu_snapshot(process_record, errors, label):
    before = process_record.get("gpu_exclusive_snapshot_before_worker") or {}
    after = process_record.get("gpu_exclusive_snapshot_after_worker") or {}
    if (
        not is_true(process_record.get("gpu_exclusive_preflight_verified"))
        or before.get("exclusive") is not True
        or before.get("compute_processes")
    ):
        errors.append(f"{label}: GPU preflight was not exclusive")
    if (
        not is_true(process_record.get("gpu_exclusive_postflight_verified"))
        or after.get("exclusive") is not True
        or after.get("compute_processes")
    ):
        errors.append(f"{label}: GPU postflight was not exclusive")


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stats(values):
    values = [float(value) for value in values]
    median = statistics.median(values)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "best": min(values),
        "mean": mean,
        "stdev": stdev,
        "median": median,
        "maximum": max(values),
        "coefficient_of_variation": stdev / mean if mean else 0.0,
        "max_over_median": max(values) / median,
        "median_over_minimum": median / min(values),
        "samples": values,
        "count": len(values),
    }


def timing_rows_for_cell(rows, key):
    return [
        row
        for row in rows
        if row.get("baseline") == "nx-cugraph"
        and (row.get("dataset"), row.get("function")) == key
        and row.get("metric") in {"build", "e2e", "kernel"}
    ]


def extreme_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("metric")].append(float(row["seconds"]))
    output = {}
    for metric in ("build", "e2e", "kernel"):
        values = grouped.get(metric, [])
        if len(values) != 5:
            continue
        ratio = max(values) / statistics.median(values)
        if ratio > STABILITY_MAX_OVER_MEDIAN:
            output[metric] = ratio
    return output


def audit_moderate(root: Path, errors, evidence, override_roots=()):
    rows = []
    for shard in MODERATE_SHARDS:
        shard_root = root / shard
        sample_path = shard_root / "results_samples.csv"
        metadata_path = shard_root / "run_metadata.json"
        if not sample_path.is_file():
            errors.append(f"missing moderate shard samples: {sample_path}")
            continue
        if not metadata_path.is_file():
            errors.append(f"missing moderate shard metadata: {metadata_path}")
            continue
        shard_rows = load_csv(sample_path)
        rows.extend(shard_rows)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        args = metadata.get("benchmark_args") or {}
        if int(args.get("repeat", 0)) != 5:
            errors.append(f"{shard}: repeat is not five")
        if int(args.get("nx_cugraph_warmup", -1)) != 3:
            errors.append(f"{shard}: nx-cugraph warmup is not three")
        versions = metadata.get("baseline_versions") or {}
        if not versions.get("nx-cugraph"):
            errors.append(f"{shard}: missing nx-cugraph version")
        gpu_profile = metadata.get("gpu_device_profile") or {}
        selected_device = gpu_profile.get("selected_device") or {}
        evidence.append(
            {
                "path": str(sample_path.resolve()),
                "sha256": sha256_file(sample_path),
                "rows": len(shard_rows),
                "gpu": selected_device.get(
                    "index",
                    gpu_profile.get("resolved_monitor_gpu_index"),
                ),
                "gpu_uuid": selected_device.get("uuid"),
                "gpu_name": selected_device.get("name"),
                "nx_cugraph": versions.get("nx-cugraph"),
                "networkx": versions.get("networkx"),
            }
        )

    for override_root in override_roots:
        sample_path = override_root / "results_samples.csv"
        metadata_path = override_root / "run_metadata.json"
        if not sample_path.is_file() or not metadata_path.is_file():
            errors.append(
                f"missing moderate override samples or metadata: "
                f"{override_root}"
            )
            continue
        override_rows = load_csv(sample_path)
        override_timing_rows = [
            row
            for row in override_rows
            if row.get("baseline") == "nx-cugraph"
            and row.get("metric") in {"build", "e2e", "kernel"}
        ]
        override_keys = {
            (row.get("dataset"), row.get("function"))
            for row in override_timing_rows
        }
        if len(override_keys) != 1:
            errors.append(
                f"{override_root}: a stability override must contain exactly "
                f"one complete cell, got {sorted(override_keys)}"
            )
            continue
        key = next(iter(override_keys))
        original_rows = timing_rows_for_cell(rows, key)
        original_extreme = extreme_metrics(original_rows)
        if not original_extreme:
            errors.append(
                f"{override_root}: original {key[0]}/{key[1]} did not fail "
                "the predeclared extreme-variance gate"
            )
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_args = metadata.get("benchmark_args") or {}
        if int(run_args.get("repeat", 0)) != 5:
            errors.append(f"{override_root}: override repeat is not five")
        if int(run_args.get("nx_cugraph_warmup", -1)) != 3:
            errors.append(
                f"{override_root}: override nx-cugraph warmup is not three"
            )
        rows = [
            row
            for row in rows
            if not (
                row.get("baseline") == "nx-cugraph"
                and (row.get("dataset"), row.get("function")) == key
            )
        ]
        rows.extend(override_rows)
        evidence.append(
            {
                "role": "full_cell_stability_override",
                "path": str(sample_path.resolve()),
                "sha256": sha256_file(sample_path),
                "rows": len(override_rows),
                "cell": {"dataset": key[0], "function": key[1]},
                "excluded_original_extreme_metrics": original_extreme,
                "replacement_policy": (
                    "discard all five original samples and rerun all five; "
                    "never replace an individual sample"
                ),
            }
        )

    selected = [
        row
        for row in rows
        if row.get("baseline") == "nx-cugraph"
        and row.get("metric") in {"build", "e2e", "kernel"}
    ]
    grouped = defaultdict(list)
    for row in selected:
        grouped[
            (
                row.get("dataset"),
                row.get("function"),
                row.get("metric"),
            )
        ].append(row)

    expected_cells = {
        (dataset, function)
        for dataset in MODERATE_DATASETS
        for function in FUNCTIONS
    }
    observed_cells = {(key[0], key[1]) for key in grouped}
    if observed_cells != expected_cells:
        errors.append(
            "moderate cell set mismatch: "
            f"missing={sorted(expected_cells - observed_cells)}, "
            f"extra={sorted(observed_cells - expected_cells)}"
        )

    output = []
    for dataset, function in sorted(expected_cells):
        metric_samples = {}
        cell_provenance = []
        for metric in ("build", "e2e", "kernel"):
            metric_rows = sorted(
                grouped.get((dataset, function, metric), []),
                key=lambda row: int(row.get("sample_index", 0)),
            )
            if len(metric_rows) != 5:
                errors.append(
                    f"{dataset}/{function}/{metric}: expected five rows, "
                    f"got {len(metric_rows)}"
                )
                continue
            indices = sorted(int(row["sample_index"]) for row in metric_rows)
            if indices != [1, 2, 3, 4, 5]:
                errors.append(
                    f"{dataset}/{function}/{metric}: bad sample indices"
                )
            if any(row.get("status") != "ok" for row in metric_rows):
                errors.append(f"{dataset}/{function}/{metric}: non-ok status")
            values = [float(row["seconds"]) for row in metric_rows]
            if not all(is_positive(value) for value in values):
                errors.append(
                    f"{dataset}/{function}/{metric}: non-positive value"
                )
            metric_samples[metric] = values
            if metric == "kernel":
                for row in metric_rows:
                    if (
                        row.get("timer_kind")
                        != "cuda_event_at_pylibcugraph_backend"
                        or row.get("measurement_window") != "device_execution"
                    ):
                        errors.append(
                            f"{dataset}/{function}: wrong processing schema"
                        )
                    check_provenance(
                        parse_provenance(row.get("timing_provenance")),
                        function,
                        errors,
                        (
                            f"{dataset}/{function}/"
                            f"sample{row.get('sample_index')}"
                        ),
                    )
                    check_moderate_gpu_snapshot(
                        row,
                        errors,
                        (
                            f"{dataset}/{function}/"
                            f"sample{row.get('sample_index')}"
                        ),
                    )
                    cell_provenance.append(
                        {
                            "sample_index": int(row["sample_index"]),
                            "timing_provenance": parse_provenance(
                                row.get("timing_provenance")
                            ),
                            "gpu_exclusive_snapshot_before_worker": row.get(
                                "gpu_exclusive_snapshot_before_worker"
                            ),
                            "gpu_exclusive_snapshot_after_worker": row.get(
                                "gpu_exclusive_snapshot_after_worker"
                            ),
                        }
                    )
        if set(metric_samples) != {"build", "e2e", "kernel"}:
            continue
        for index, (e2e, kernel) in enumerate(
            zip(metric_samples["e2e"], metric_samples["kernel"]), 1
        ):
            if kernel > e2e:
                errors.append(
                    f"{dataset}/{function}/sample{index}: processing > E2E"
                )
            if math.isclose(kernel, e2e, rel_tol=0.0, abs_tol=1e-12):
                errors.append(
                    f"{dataset}/{function}/sample{index}: wall copied to kernel"
                )
        output.append(
            {
                "dataset": dataset,
                "function": function,
                "baseline": "nx-cugraph",
                "construction": stats(metric_samples["build"]),
                "e2e": stats(metric_samples["e2e"]),
                "processing": stats(metric_samples["kernel"]),
                "source": "moderate_run_full_native_graph",
                "validation_status": "pass",
                "timing_provenance": {
                    "timer_kind": "cuda_event_at_pylibcugraph_backend",
                    "measurement_window": "device_execution",
                    "validation_outside_timer": True,
                    "public_return_boundary": True,
                    "prepared_native_graph": True,
                    "networkx_fallback_to_nx": False,
                    "networkx_cache_converted_graphs": True,
                    "warmup_calls": 3,
                    "fresh_process_samples": 5,
                    "gpu_exclusive_pre_and_post_snapshots": True,
                    "samples": cell_provenance,
                },
            }
        )
    return output


def audit_large(path: Path, dataset: str, errors, evidence):
    summary_path = path / "nxcugraph_large_matrix.json"
    manifest_path = path / "run_manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        errors.append(f"{dataset}: missing large summary or run manifest")
        return []
    records = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        errors.append(f"{dataset}: run manifest is not complete")
    protocol = manifest.get("protocol") or {}
    if int(protocol.get("warmup_calls", -1)) != 3:
        errors.append(f"{dataset}: large warmup count is not three")
    if int(protocol.get("timing_processes", 0)) != 5:
        errors.append(f"{dataset}: large timing process count is not five")
    if protocol.get("fresh_process_per_timing_sample") is not True:
        errors.append(f"{dataset}: large samples are not fresh-process")
    evidence.append(
        {
            "path": str(summary_path.resolve()),
            "sha256": sha256_file(summary_path),
            "rows": len(records),
            "run_manifest": str(manifest_path.resolve()),
            "run_manifest_sha256": sha256_file(manifest_path),
            "gpu": manifest.get("gpu"),
            "software": manifest.get("software"),
            "source": manifest.get("source"),
        }
    )

    by_function = {
        record.get("function"): record
        for record in records
        if record.get("dataset") == dataset
    }
    expected = set(LARGE_FUNCTIONS[dataset])
    if set(by_function) != expected:
        errors.append(
            f"{dataset}: large function set mismatch "
            f"missing={sorted(expected-set(by_function))}, "
            f"extra={sorted(set(by_function)-expected)}"
        )
    output = []
    for function in sorted(expected):
        record = by_function.get(function)
        if not record:
            continue
        label = f"{dataset}/{function}"
        if record.get("status") != "ok":
            errors.append(f"{label}: non-ok large status")
            continue
        if int(record.get("timing_process_samples", 0)) != 5:
            errors.append(f"{label}: expected five fresh processes")
        e2e = (record.get("e2e") or {}).get("samples") or []
        kernel = (record.get("kernel") or {}).get("samples") or []
        build = (record.get("graph_prepare") or {}).get("samples") or []
        if not (len(e2e) == len(kernel) == len(build) == 5):
            errors.append(f"{label}: incomplete large metric samples")
            continue
        cell_provenance = []
        for index, process_record in enumerate(
            record.get("timing_process_records") or [], 1
        ):
            check_large_gpu_snapshot(
                process_record, errors, f"{label}/sample{index}"
            )
            if process_record.get("validation_outside_timer") is not True:
                errors.append(f"{label}/sample{index}: validation inside timer")
            provenance_rows = process_record.get("device_timer_records") or []
            if len(provenance_rows) != 1:
                errors.append(f"{label}/sample{index}: ambiguous provenance")
                continue
            timer = provenance_rows[0]
            wrapped = {
                "timer": timer,
                "prepared_native_graph": process_record.get(
                    "prepared_native_graph"
                ),
                "validation_outside_timer": process_record.get(
                    "validation_outside_timer"
                ),
                "public_return_boundary": process_record.get(
                    "public_return_boundary"
                ),
                "networkx_fallback_to_nx": process_record.get(
                    "networkx_fallback_to_nx"
                ),
                "networkx_cache_converted_graphs": process_record.get(
                    "networkx_cache_converted_graphs"
                ),
                "warmup_calls": int(process_record.get("warmup_calls", -1)),
                "warmup_details": [
                    {
                        "e2e_seconds": detail.get("seconds"),
                        "device_seconds": detail.get("device_seconds"),
                    }
                    for detail in process_record.get("warmup_details") or []
                ],
            }
            check_provenance(
                wrapped, function, errors, f"{label}/sample{index}"
            )
            cell_provenance.append(
                {
                    "sample_index": index,
                    "timing_provenance": wrapped,
                    "gpu_exclusive_snapshot_before_worker": process_record.get(
                        "gpu_exclusive_snapshot_before_worker"
                    ),
                    "gpu_exclusive_snapshot_after_worker": process_record.get(
                        "gpu_exclusive_snapshot_after_worker"
                    ),
                }
            )
        for index, (wall, device) in enumerate(zip(e2e, kernel), 1):
            if not is_positive(wall) or not is_positive(device):
                errors.append(f"{label}/sample{index}: non-positive timing")
            if float(device) > float(wall):
                errors.append(f"{label}/sample{index}: processing > E2E")
            if math.isclose(
                float(device), float(wall), rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(f"{label}/sample{index}: wall copied to kernel")
        output.append(
            {
                "dataset": dataset,
                "function": function,
                "baseline": "nx-cugraph",
                "construction": stats(build),
                "e2e": stats(e2e),
                "processing": stats(kernel),
                "source": "large_normalized_csr_native_graph",
                "validation_status": "pass",
                "timing_provenance": {
                    "timer_kind": "cuda_event_at_pylibcugraph_backend",
                    "measurement_window": "device_execution",
                    "validation_outside_timer": True,
                    "public_return_boundary": True,
                    "prepared_native_graph": True,
                    "networkx_fallback_to_nx": False,
                    "networkx_cache_converted_graphs": True,
                    "warmup_calls": 3,
                    "fresh_process_samples": 5,
                    "gpu_exclusive_pre_and_post_snapshots": True,
                    "samples": cell_provenance,
                },
            }
        )
    return output


def audit_semantic_validation(path: Path, errors, evidence):
    if not path.is_file():
        errors.append(f"missing semantic validation report: {path}")
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        errors.append("nx-cugraph semantic validation did not pass")
    if report.get("networkx_cache_converted_graphs") is not True:
        errors.append("semantic validation did not enable converted-graph cache")
    if report.get("networkx_fallback_to_nx") is not False:
        errors.append("semantic validation did not disable CPU fallback")
    records = {
        record.get("function"): record
        for record in report.get("records") or []
    }
    if set(records) != set(FUNCTIONS):
        errors.append(
            "semantic validation function set mismatch: "
            f"missing={sorted(set(FUNCTIONS) - set(records))}, "
            f"extra={sorted(set(records) - set(FUNCTIONS))}"
        )
    for function, record in records.items():
        label = f"semantic/{function}"
        if record.get("status") != "pass":
            errors.append(f"{label}: CPU reference mismatch")
        if record.get("device_not_above_e2e") is not True:
            errors.append(f"{label}: device interval exceeds E2E")
        timer = record.get("timer_provenance") or {}
        if set(timer.get("backend_functions") or []) != set(BACKENDS[function]):
            errors.append(f"{label}: wrong backend timer provenance")
        if int(timer.get("backend_interval_count", 0)) <= 0:
            errors.append(f"{label}: missing backend device interval")
        if int(timer.get("resource_handle_factory_calls", 0)) <= 0:
            errors.append(f"{label}: missing explicit-stream ResourceHandle")
    evidence.append(
        {
            "role": "cpu_semantic_cross_validation",
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "status": report.get("status"),
            "graph_type": report.get("graph_type"),
            "functions": sorted(records),
            "reference": report.get("validation_reference"),
            "networkx_cache_converted_graphs": report.get(
                "networkx_cache_converted_graphs"
            ),
            "networkx_fallback_to_nx": report.get(
                "networkx_fallback_to_nx"
            ),
        }
    )
    return report.get("graph_type")


def flatten_result(row):
    output = {
        "dataset": row["dataset"],
        "function": row["function"],
        "baseline": row["baseline"],
        "source": row["source"],
        "display_estimator": "minimum_of_five",
        "validation_status": row["validation_status"],
        "timer_kind": "cuda_event_at_pylibcugraph_backend",
        "measurement_window": "device_execution",
        "timing_provenance": json.dumps(
            row["timing_provenance"], ensure_ascii=False, sort_keys=True
        ),
    }
    for metric in ("construction", "processing", "e2e"):
        values = row[metric]
        output[f"{metric}_best_seconds"] = values["best"]
        output[f"{metric}_min_seconds"] = values["best"]
        output[f"{metric}_mean_seconds"] = values["mean"]
        output[f"{metric}_stdev_seconds"] = values["stdev"]
        output[f"{metric}_sample_std_seconds"] = values["stdev"]
        output[f"{metric}_median_seconds"] = values["median"]
        output[f"{metric}_max_seconds"] = values["maximum"]
        output[f"{metric}_coefficient_of_variation"] = values[
            "coefficient_of_variation"
        ]
        output[f"{metric}_max_over_median"] = values["max_over_median"]
        output[f"{metric}_median_over_minimum"] = values[
            "median_over_minimum"
        ]
        output[f"{metric}_samples"] = json.dumps(values["samples"])
        output[f"{metric}_sample_count"] = values["count"]
    return output


def flatten_sample_results(rows):
    output = []
    for row in rows:
        provenance_samples = {
            int(sample["sample_index"]): sample
            for sample in row["timing_provenance"]["samples"]
        }
        for index in range(1, 6):
            provenance = provenance_samples[index]
            output.append(
                {
                    "dataset": row["dataset"],
                    "function": row["function"],
                    "baseline": row["baseline"],
                    "sample_index": index,
                    "construction_seconds": row["construction"]["samples"][
                        index - 1
                    ],
                    "e2e_seconds": row["e2e"]["samples"][index - 1],
                    "processing_seconds": row["processing"]["samples"][
                        index - 1
                    ],
                    "status": "ok",
                    "validation_status": row["validation_status"],
                    "validation_outside_timer": True,
                    "timer_kind": "cuda_event_at_pylibcugraph_backend",
                    "measurement_window": "device_execution",
                    "gpu_exclusive_snapshot_before_worker": json.dumps(
                        provenance[
                            "gpu_exclusive_snapshot_before_worker"
                        ],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if isinstance(
                        provenance[
                            "gpu_exclusive_snapshot_before_worker"
                        ],
                        dict,
                    )
                    else provenance[
                        "gpu_exclusive_snapshot_before_worker"
                    ],
                    "gpu_exclusive_snapshot_after_worker": json.dumps(
                        provenance[
                            "gpu_exclusive_snapshot_after_worker"
                        ],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if isinstance(
                        provenance[
                            "gpu_exclusive_snapshot_after_worker"
                        ],
                        dict,
                    )
                    else provenance[
                        "gpu_exclusive_snapshot_after_worker"
                    ],
                    "timing_provenance": json.dumps(
                        provenance["timing_provenance"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "source": row["source"],
                }
            )
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moderate-root", type=Path, required=True)
    parser.add_argument("--orkut-root", type=Path, required=True)
    parser.add_argument("--twitter-root", type=Path, required=True)
    parser.add_argument(
        "--moderate-override",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--semantic-validation",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--excluded-run",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    errors = []
    evidence = []
    merged = audit_moderate(
        args.moderate_root,
        errors,
        evidence,
        override_roots=args.moderate_override,
    )
    merged.extend(audit_large(args.orkut_root, "com-Orkut", errors, evidence))
    merged.extend(
        audit_large(args.twitter_root, "GAP-twitter", errors, evidence)
    )
    semantic_graph_types = {
        graph_type
        for graph_type in (
            audit_semantic_validation(path, errors, evidence)
            for path in args.semantic_validation
        )
        if graph_type
    }
    if semantic_graph_types != {"directed", "undirected"}:
        errors.append(
            "semantic validation must cover directed and undirected graphs; "
            f"observed={sorted(semantic_graph_types)}"
        )
    merged = sorted(merged, key=lambda row: (row["dataset"], row["function"]))
    expected_cells = 88 + 8 + 3
    if len(merged) != expected_cells:
        errors.append(
            f"merged success cell count is {len(merged)}, expected {expected_cells}"
        )
    for row in merged:
        for metric in ("construction", "e2e", "processing"):
            ratio = float(row[metric]["max_over_median"])
            if ratio > STABILITY_MAX_OVER_MEDIAN:
                errors.append(
                    f"{row['dataset']}/{row['function']}/{metric}: "
                    f"extreme max/median={ratio:.6g} exceeds "
                    f"{STABILITY_MAX_OVER_MEDIAN:g}"
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "nxcugraph_strict_main99.csv"
    rows = [flatten_result(row) for row in merged]
    fields = list(rows[0]) if rows else ["dataset", "function", "baseline"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    sample_csv_path = args.out_dir / "nxcugraph_strict_main99_samples.csv"
    sample_rows = flatten_sample_results(merged)
    if len(sample_rows) != expected_cells * 5:
        errors.append(
            f"long-form sample row count is {len(sample_rows)}, "
            f"expected {expected_cells * 5}"
        )
    sample_fields = list(sample_rows[0]) if sample_rows else [
        "dataset",
        "function",
        "baseline",
        "sample_index",
    ]
    with sample_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(sample_rows)
    resolved_inputs = {
        "moderate_root": str(args.moderate_root.resolve()),
        "com_orkut_root": str(args.orkut_root.resolve()),
        "gap_twitter_root": str(args.twitter_root.resolve()),
        "moderate_overrides": [
            str(path.resolve()) for path in args.moderate_override
        ],
        "semantic_validation": [
            str(path.resolve()) for path in args.semantic_validation
        ],
    }
    included_paths = {
        args.moderate_root.resolve(),
        args.orkut_root.resolve(),
        args.twitter_root.resolve(),
        *(path.resolve() for path in args.moderate_override),
        *(path.resolve() for path in args.semantic_validation),
    }
    excluded_runs = []
    for path in args.excluded_run:
        resolved = path.resolve()
        if resolved in included_paths:
            errors.append(f"excluded run is also an audit input: {resolved}")
            continue
        run_log = resolved / "run.log"
        excluded_runs.append(
            {
                "path": str(resolved),
                "exists": resolved.exists(),
                "reason_from_directory_name": resolved.name,
                "run_log": str(run_log) if run_log.is_file() else "",
                "run_log_sha256": (
                    sha256_file(run_log) if run_log.is_file() else ""
                ),
            }
        )
    report = {
        "status": "pass" if not errors else "fail",
        "expected_success_cells": expected_cells,
        "observed_success_cells": len(merged),
        "errors": errors,
        "evidence": evidence,
        "merged_csv": str(csv_path.resolve()),
        "merged_csv_sha256": sha256_file(csv_path),
        "samples_csv": str(sample_csv_path.resolve()),
        "samples_csv_sha256": sha256_file(sample_csv_path),
        "sample_row_count": len(sample_rows),
        "resolved_inputs": resolved_inputs,
        "explicitly_excluded_runs": excluded_runs,
        "protocol": {
            "construction": "prepared native nx-cugraph graph",
            "processing": "actual pylibcugraph device interval",
            "e2e": "strict public NetworkX call through result return",
            "warmup_calls": 3,
            "fresh_process_samples": 5,
            "display_estimator": "minimum_of_five",
            "three_segment_timing_estimators": {
                "construction": "minimum_of_five",
                "processing": "minimum_of_five",
                "e2e": "minimum_of_five",
            },
            "sample_standard_deviation": True,
            "standard_deviation_ddof": 1,
            "validation_outside_timer": True,
            "networkx_cache_converted_graphs": True,
            "cpu_fallback": False,
            "networkx_fallback_to_nx": False,
            "wall_time_copied_to_processing": False,
            "gpu_exclusive_pre_and_post_snapshots_per_worker": True,
            "stability_max_over_median": STABILITY_MAX_OVER_MEDIAN,
            "stability_override_policy": (
                "only a full five-sample cell whose original max/median "
                "exceeded the predeclared threshold may be rerun; all five "
                "original samples are excluded together"
            ),
            "semantic_validation": (
                "all eight timed adapters compared with NetworkX CPU on "
                "identical normalized directed and undirected graphs"
            ),
        },
    }
    benchmarking_root = Path(__file__).resolve().parent
    library_source = benchmarking_root / "library_baselines.py"
    full_source = benchmarking_root / "run_full_baselines.py"
    large_source = benchmarking_root / "run_nxcugraph_large_matrix.py"
    timer_source = benchmarking_root / "nxcugraph_device_timer.py"
    report["raw_source_contract"] = {
        "cache_converted_graphs_enabled": source_contract_reference(
            library_source,
            "nx.config.cache_converted_graphs = True",
            "timing_provenance.networkx_cache_converted_graphs",
        ),
        "cpu_fallback_disabled": source_contract_reference(
            library_source,
            "nx.config.fallback_to_nx = False",
            "timing_provenance.networkx_fallback_to_nx",
        ),
        "runtime_config_provenance": source_contract_reference(
            library_source,
            '"networkx_cache_converted_graphs": bool(',
            "timing_provenance",
        ),
        "raw_provenance_persisted": source_contract_reference(
            full_source,
            '"timing_provenance",',
            "results_samples.csv.timing_provenance",
        ),
        "moderate_gpu_preflight_snapshot_persisted": source_contract_reference(
            full_source,
            '"gpu_exclusive_snapshot_before_worker",',
            "results_samples.csv.gpu_exclusive_snapshot_before_worker",
        ),
        "moderate_gpu_postflight_snapshot_persisted": source_contract_reference(
            full_source,
            '"gpu_exclusive_snapshot_after_worker",',
            "results_samples.csv.gpu_exclusive_snapshot_after_worker",
        ),
        "large_gpu_snapshot_capture": source_contract_reference(
            large_source,
            "def collect_gpu_exclusive_snapshot(",
            "timing_process_records[].gpu_exclusive_snapshot_before_worker",
        ),
        "large_cache_converted_graphs_enabled": source_contract_reference(
            large_source,
            "nx.config.cache_converted_graphs = True",
            (
                "timing_process_records[]."
                "networkx_cache_converted_graphs"
            ),
        ),
        "large_cpu_fallback_disabled": source_contract_reference(
            large_source,
            "nx.config.fallback_to_nx = False",
            "timing_process_records[].networkx_fallback_to_nx",
        ),
        "large_native_graph_raw_provenance": source_contract_reference(
            large_source,
            '"prepared_native_graph": True,',
            "timing_process_records[].prepared_native_graph",
        ),
        "backend_cuda_event_start": source_contract_reference(
            timer_source,
            "start = self.cp.cuda.Event()",
            "timing_provenance.timer",
        ),
        "backend_cuda_event_elapsed_time": source_contract_reference(
            timer_source,
            "self.cp.cuda.get_elapsed_time(start, stop)",
            "processing_seconds",
        ),
        "exact_backend_provenance_validation": source_contract_reference(
            timer_source,
            "def validate_provenance(",
            (
                "timing_provenance.timer.backend_functions + "
                "backend_interval_count + resource_handle_factory_calls"
            ),
        ),
    }
    audit_path = args.out_dir / "audit.json"
    audit_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path = args.out_dir / "AUDIT.md"
    md_path.write_text(
        "\n".join(
            [
                "# Strict nx-cugraph 99-cell audit",
                "",
                f"- Status: **{report['status']}**",
                f"- Expected successful cells: {expected_cells}",
                f"- Observed successful cells: {len(merged)}",
                f"- Fresh-process sample records: {len(sample_rows)}",
                "- Timing samples: five fresh processes per cell",
                "- Construction estimator: minimum of five "
                "(mean and sample SD retained)",
                "- Processing estimator: minimum of five "
                "(mean and sample SD retained)",
                "- E2E estimator: minimum of five "
                "(mean and sample SD retained)",
                "- Processing boundary: CUDA events at actual "
                "pylibcugraph algorithm entries",
                "- E2E boundary: strict NetworkX public call through "
                "complete result return",
                "- Validation: outside both timers",
                "- NetworkX converted-graph cache: enabled",
                "- CPU fallback: disabled",
                f"- Aggregated CSV SHA256: {report['merged_csv_sha256']}",
                f"- Raw sample CSV SHA256: {report['samples_csv_sha256']}",
                "",
                "## Errors",
                "",
                *(
                    [f"- {error}" for error in errors]
                    if errors
                    else ["- None"]
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
