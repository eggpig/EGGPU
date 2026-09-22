#!/usr/bin/env python3
"""Gate and package the strict Gunrock 90-cell formal measurement.

This script consumes two fresh formal batches:

* 83 successful main-matrix cells from ``run_full_baselines.py``;
* 7 successful com-Orkut cells from ``run_gunrock_large_matrix.py``.

It refuses partial, contaminated, wall-time-derived, or non-native evidence.
Only after every gate passes does it emit a paper-overlay CSV.  The displayed
``seconds`` value is the minimum of five fresh processes, while the mean and
sample standard deviation remain explicit columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from gunrock_timing_protocol import (
    PROTOCOL_VERSION,
    GunrockTimingProtocolError,
    parse_strict_gunrock_timing,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("build", "kernel", "e2e")
CORE_FUNCTIONS = (
    "PageRank",
    "LCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
)
MAIN_DATASETS = (
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
MST_DATASETS = (
    "LastFM",
    "p2p-Gnutella04",
    "soc-Slashdot0811",
    "ER-100k",
    "web-NotreDame",
    "com-youtube",
)
APP_BY_FUNCTION = {
    "PageRank": "pr",
    "MST": "mst",
    "LCC": "lcc",
    "BFS": "bfs",
    "Dijkstra": "sssp",
    "BellmanFord": "sssp",
    "SSSP": "sssp",
    "KCore": "kcore",
}
SECONDS_BY_METRIC = {
    "build": "construction_seconds",
    "kernel": "processing_seconds",
    "e2e": "e2e_seconds",
}
WINDOW_BY_METRIC = {
    "build": "matrix_market_load_to_device_graph",
    "kernel": "native_device_interval",
    "e2e": "matrix_market_load_to_complete_host_result",
}
TIMER_BY_METRIC = {
    "build": "steady_clock_wall",
    "kernel": "gunrock_native_device_interval",
    "e2e": "steady_clock_wall",
}
CATASTROPHIC_MAX_MIN_RATIO = 20.0
CATASTROPHIC_ABSOLUTE_SPAN_SECONDS = 0.1
MST_WEIGHT_REL_TOL = 1.0e-4
MST_WEIGHT_ABS_TOL = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_csv(path: Path, rows: list[dict], preferred_fields: tuple[str, ...]) -> None:
    extras = sorted({key for row in rows for key in row} - set(preferred_fields))
    fields = list(preferred_fields) + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def expected_cells() -> set[tuple[str, str]]:
    cells = {
        (dataset, function)
        for dataset in MAIN_DATASETS
        for function in CORE_FUNCTIONS
    }
    cells.update((dataset, "MST") for dataset in MST_DATASETS)
    cells.update(("com-Orkut", function) for function in CORE_FUNCTIONS)
    assert len(cells) == 90
    return cells


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_main_correctness_reference(
    formal_manifest: dict, issues: list[str]
) -> dict[tuple[str, str], dict]:
    reference = formal_manifest.get("main_correctness_reference", {})
    long_path = resolve_recorded_path(reference.get("results_long", "")).resolve()
    validation_path = resolve_recorded_path(
        reference.get("validation_csv", "")
    ).resolve()
    for path, expected_sha, label in (
        (long_path, reference.get("results_long_sha256"), "results_long"),
        (
            validation_path,
            reference.get("validation_csv_sha256"),
            "validation_csv",
        ),
    ):
        if not path.is_file():
            issues.append(f"main correctness reference {label} is absent: {path}")
        elif sha256_file(path) != expected_sha:
            issues.append(
                f"main correctness reference {label} SHA-256 differs from manifest"
            )
    if not long_path.is_file() or not validation_path.is_file():
        return {}
    validation = {
        (row.get("dataset", ""), row.get("function", "")): row
        for row in read_csv(validation_path)
        if row.get("baseline") == "Gunrock"
    }
    result = {}
    for row in read_csv(long_path):
        if (
            row.get("baseline") != "Gunrock"
            or row.get("metric") != "e2e"
            or row.get("status") != "ok"
        ):
            continue
        cell = (row.get("dataset", ""), row.get("function", ""))
        validation_row = validation.get(cell, {})
        result[cell] = {
            "validation_status": validation_row.get("validation_status", ""),
            "validation_details": validation_row.get("details", ""),
            "correctness": row.get("correctness", ""),
            "reference": validation_row.get("reference", ""),
            "reference_correctness": validation_row.get(
                "reference_correctness", ""
            ),
            "reference_results_long": str(long_path),
            "reference_validation_csv": str(validation_path),
        }
    return result


def parse_positive(value, context: str, issues: list[str]) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        issues.append(f"{context}: missing or nonnumeric seconds={value!r}")
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        issues.append(f"{context}: seconds must be finite and positive, got {parsed!r}")
        return None
    return parsed


def is_false(value) -> bool:
    return value in (None, "", False, 0, "0", "false", "False", "FALSE")


def parse_log(
    raw_path: str,
    allowed_root: Path,
    excluded_roots: tuple[Path, ...],
    cache: dict[Path, dict | None],
    issues: list[str],
) -> tuple[Path, dict | None]:
    if not raw_path:
        issues.append("timing row has no native log path")
        return Path(), None
    path = resolve_recorded_path(raw_path).resolve()
    allowed = allowed_root.resolve()
    if not path.is_relative_to(allowed):
        issues.append(f"log escapes included batch {allowed}: {path}")
    for excluded in excluded_roots:
        if path.is_relative_to(excluded.resolve()):
            issues.append(f"log resolves inside excluded batch: {path}")
    if path in cache:
        return path, cache[path]
    if not path.is_file():
        issues.append(f"native timing log is absent: {path}")
        cache[path] = None
        return path, None
    try:
        timing = parse_strict_gunrock_timing(
            path.read_text(encoding="utf-8", errors="replace")
        )
    except GunrockTimingProtocolError as exc:
        issues.append(f"strict native timing rejected {path}: {exc}")
        timing = None
    cache[path] = timing
    return path, timing


def executable_evidence(
    raw: dict,
    app: str,
    artifact_manifest: dict,
    context: str,
    issues: list[str],
) -> dict:
    evidence = raw.get(app) if isinstance(raw, dict) else None
    pinned = artifact_manifest.get("executables", {}).get(app, {})
    if not isinstance(evidence, dict):
        issues.append(f"{context}: executable evidence for {app!r} is absent")
        return {}
    if evidence.get("status") not in (None, "available"):
        issues.append(f"{context}: executable {app} status={evidence.get('status')!r}")
    actual_sha = evidence.get("sha256")
    pinned_sha = pinned.get("sha256")
    if not actual_sha or actual_sha != pinned_sha:
        issues.append(
            f"{context}: executable {app} SHA mismatch "
            f"(observed={actual_sha!r}, pinned={pinned_sha!r})"
        )
    if evidence.get("source_manifest_binary_sha256_match") is not True:
        issues.append(f"{context}: executable {app} is not manifest-bound")
    if evidence.get("binary_not_older_than_latest_tracked_source_change") is not True:
        issues.append(f"{context}: executable {app} predates a relevant source change")
    binary = Path(evidence.get("path", ""))
    if not binary.is_file():
        issues.append(f"{context}: executable path is absent: {binary}")
    elif actual_sha and sha256_file(binary) != actual_sha:
        issues.append(f"{context}: executable changed after measurement: {binary}")
    return {
        "app": app,
        "sha256": actual_sha or "",
        "path": str(binary),
        "source_commit": evidence.get("source_commit", ""),
        "source_diff_sha256": evidence.get("source_diff_sha256", ""),
        "source_manifest_sha256": evidence.get("source_manifest_sha256", ""),
    }


def metric_statistics(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    variance = statistics.variance(values)
    return {
        "samples": values,
        "count": len(values),
        "min": min(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "variance": variance,
        "median": statistics.median(values),
        "max": max(values),
        "cv": sample_sd / mean if mean else None,
        "max_min_ratio": max(values) / min(values),
        "absolute_span_seconds": max(values) - min(values),
    }


def package_cell(
    dataset: str,
    function: str,
    dataset_size: str,
    graph_type: str,
    samples: dict[str, list[dict]],
    validation: str,
    validation_note: str,
    source_batch: str,
    executable: dict,
) -> tuple[dict, list[dict], list[dict]]:
    cell = {
        "dataset_size": dataset_size,
        "graph_type": graph_type,
        "dataset": dataset,
        "function": function,
        "validation": validation,
        "validation_note": validation_note,
        "source_batch": source_batch,
        "executable": executable,
        "metrics": {},
    }
    aggregate_rows = []
    raw_rows = []
    for metric in METRICS:
        ordered = sorted(samples[metric], key=lambda row: int(row["sample_index"]))
        values = [float(row["seconds"]) for row in ordered]
        summary = metric_statistics(values)
        cell["metrics"][metric] = summary
        first = ordered[0]
        logs = [row["log"] for row in ordered]
        aggregate_rows.append(
            {
                "dataset_size": dataset_size,
                "graph_type": graph_type,
                "dataset": dataset,
                "function": function,
                "baseline": "Gunrock",
                "metric": metric,
                "seconds": f"{summary['min']:.12g}",
                "value": f"{summary['min']:.12g}",
                "unit": "s",
                "metric_family": "time",
                "measurement_scope": (
                    "algorithm_compute"
                    if metric == "kernel"
                    else "standalone_gpu_function"
                ),
                "timer_kind": TIMER_BY_METRIC[metric],
                "measurement_window": WINDOW_BY_METRIC[metric],
                "status": "ok",
                "correctness": validation_note,
                "validation": validation,
                "log": logs[0],
                "notes": (
                    "strict native Gunrock three-phase timing; five fresh "
                    "subprocesses; paper-facing seconds is min-of-five; mean "
                    "and sample standard deviation are retained; aligned "
                    "MatrixMarket file pre-generation, validation, printing, "
                    "result-file I/O, and external CLI launch wall time are "
                    "outside the reported standalone interval"
                ),
                "estimator_kind": "minimum_of_five_fresh_processes",
                "sample_index": "",
                "sample_count": "5",
                "measurement_phase": "timing",
                "timing_protocol_version": PROTOCOL_VERSION,
                "external_cli_wall_used": "false",
                "aggregation": "minimum_for_paper_display",
                "n_total": "5",
                "n_valid": "5",
                "publishable": "true",
                "mean_seconds": f"{summary['mean']:.12g}",
                "std_seconds": f"{summary['sample_sd']:.12g}",
                "sample_sd_seconds": f"{summary['sample_sd']:.12g}",
                "variance_seconds2": f"{summary['variance']:.12g}",
                "cv": f"{summary['cv']:.12g}" if summary["cv"] is not None else "",
                "median_seconds": f"{summary['median']:.12g}",
                "min_seconds": f"{summary['min']:.12g}",
                "max_seconds": f"{summary['max']:.12g}",
                "relative_std_percent": (
                    f"{100.0 * summary['cv']:.12g}"
                    if summary["cv"] is not None
                    else ""
                ),
                "max_min_ratio": f"{summary['max_min_ratio']:.12g}",
                "sample_seconds_json": json.dumps(values, separators=(",", ":")),
                "sample_logs_json": json.dumps(logs, separators=(",", ":")),
                "source_batch": source_batch,
                "executable_app": executable.get("app", ""),
                "executable_sha256": executable.get("sha256", ""),
                "source_commit": executable.get("source_commit", ""),
                "source_diff_sha256": executable.get("source_diff_sha256", ""),
                "source_manifest_sha256": executable.get(
                    "source_manifest_sha256", ""
                ),
                "boundary_contract": (
                    "construction=MatrixMarket-load-to-device-graph;"
                    "processing=native-device-interval;"
                    "e2e=MatrixMarket-load-to-complete-host-result"
                ),
            }
        )
        for row in ordered:
            raw_rows.append(
                {
                    "dataset_size": dataset_size,
                    "graph_type": graph_type,
                    "dataset": dataset,
                    "function": function,
                    "baseline": "Gunrock",
                    "metric": metric,
                    "sample_index": row["sample_index"],
                    "sample_count": "5",
                    "seconds": f"{float(row['seconds']):.12g}",
                    "status": "ok",
                    "validation": validation,
                    "measurement_scope": (
                        "algorithm_compute"
                        if metric == "kernel"
                        else "standalone_gpu_function"
                    ),
                    "timer_kind": TIMER_BY_METRIC[metric],
                    "measurement_window": WINDOW_BY_METRIC[metric],
                    "timing_protocol_version": PROTOCOL_VERSION,
                    "external_cli_wall_used": "false",
                    "log": row["log"],
                    "source_batch": source_batch,
                    "executable_app": executable.get("app", ""),
                    "executable_sha256": executable.get("sha256", ""),
                }
            )
    return cell, aggregate_rows, raw_rows


def audit_main(
    main_dir: Path,
    excluded_roots: tuple[Path, ...],
    artifact_manifest: dict,
    correctness_reference: dict[tuple[str, str], dict],
    issues: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    samples_path = main_dir / "results_samples.csv"
    long_path = main_dir / "results_long.csv"
    metadata_path = main_dir / "run_metadata.json"
    for required in (samples_path, long_path, metadata_path):
        if not required.is_file():
            issues.append(f"main83 required artifact is absent: {required}")
    if issues and not samples_path.is_file():
        return [], [], []

    rows = [
        row
        for row in read_csv(samples_path)
        if row.get("baseline") == "Gunrock"
        and row.get("measurement_phase") == "timing"
        and row.get("metric") in METRICS
    ]
    main_expected = expected_cells() - {
        ("com-Orkut", function) for function in CORE_FUNCTIONS
    }
    observed_ok = {
        (row.get("dataset", ""), row.get("function", ""))
        for row in rows
        if row.get("status") == "ok"
    }
    unexpected = observed_ok - main_expected
    if unexpected:
        issues.append(f"main83 contains unexpected successful cells: {sorted(unexpected)}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset"), row.get("function"), row.get("metric"))].append(
            row
        )
    logs = {}
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    args = metadata.get("benchmark_args", {})
    if int(args.get("repeat", -1)) != 5:
        issues.append(f"main83 metadata repeat must be 5, got {args.get('repeat')!r}")
    if str(args.get("gpu", "")) != "3":
        issues.append(f"main83 metadata physical GPU must be 3, got {args.get('gpu')!r}")
    if args.get("baselines") != ["Gunrock"]:
        issues.append(f"main83 metadata must select only Gunrock: {args.get('baselines')!r}")
    artifacts = metadata.get("build_artifacts", {}).get("gunrock_executables", {})

    packaged_cells = []
    aggregate_rows = []
    raw_rows = []
    for dataset, function in sorted(main_expected):
        cell_samples = {}
        dataset_size = ""
        graph_type = ""
        validation_notes = []
        sample_logs = defaultdict(dict)
        for metric in METRICS:
            metric_rows = grouped.get((dataset, function, metric), [])
            if len(metric_rows) != 5:
                issues.append(
                    f"main83 {dataset}/{function}/{metric}: expected 5 rows, "
                    f"observed {len(metric_rows)}"
                )
            indices = []
            accepted = []
            for row in metric_rows:
                context = f"main83 {dataset}/{function}/{metric}"
                try:
                    sample_index = int(row.get("sample_index", ""))
                    indices.append(sample_index)
                except (TypeError, ValueError):
                    issues.append(
                        f"{context}: invalid sample_index={row.get('sample_index')!r}"
                    )
                    continue
                if row.get("status") != "ok":
                    issues.append(
                        f"{context} sample {sample_index}: status={row.get('status')!r}"
                    )
                if str(row.get("sample_count", "")) != "5":
                    issues.append(
                        f"{context} sample {sample_index}: sample_count="
                        f"{row.get('sample_count')!r}, expected '5'"
                    )
                seconds = parse_positive(
                    row.get("seconds"), f"{context} sample {sample_index}", issues
                )
                if not is_false(row.get("external_cli_wall_used")):
                    issues.append(
                        f"{context} sample {sample_index}: external CLI wall was used"
                    )
                log_path, timing = parse_log(
                    row.get("log", ""),
                    main_dir,
                    excluded_roots,
                    logs,
                    issues,
                )
                if timing is not None and seconds is not None:
                    native = float(timing[SECONDS_BY_METRIC[metric]])
                    if not math.isclose(seconds, native, rel_tol=1.0e-8, abs_tol=1.0e-10):
                        issues.append(
                            f"{context} sample {sample_index}: CSV={seconds:.12g}, "
                            f"native log={native:.12g}"
                        )
                if metric in {"build", "e2e"}:
                    if row.get("timing_protocol_version") != PROTOCOL_VERSION:
                        issues.append(
                            f"{context} sample {sample_index}: protocol="
                            f"{row.get('timing_protocol_version')!r}"
                        )
                if seconds is not None:
                    accepted.append(
                        {
                            "sample_index": sample_index,
                            "seconds": seconds,
                            "log": str(log_path),
                        }
                    )
                    sample_logs[sample_index][metric] = str(log_path)
                dataset_size = row.get("dataset_size", dataset_size)
                graph_type = row.get("graph_type", graph_type)
                if metric == "e2e" and row.get("correctness"):
                    validation_notes.append(row["correctness"])
            if sorted(indices) != [1, 2, 3, 4, 5]:
                issues.append(
                    f"main83 {dataset}/{function}/{metric}: sample indices={sorted(indices)}"
                )
            cell_samples[metric] = accepted
        for sample_index in range(1, 6):
            if len(set(sample_logs[sample_index].values())) != 1:
                issues.append(
                    f"main83 {dataset}/{function} sample {sample_index}: "
                    "three metric rows do not share one native invocation log"
                )
        if not validation_notes:
            issues.append(f"main83 {dataset}/{function}: correctness evidence is absent")
        reference = correctness_reference.get((dataset, function), {})
        if reference.get("validation_status") != "pass":
            issues.append(
                f"main83 {dataset}/{function}: pinned cross-baseline validation "
                f"status={reference.get('validation_status')!r}, expected 'pass'"
            )
        if function == "MST":
            independent_weights = re.findall(
                r"\bweight=([0-9.eE+-]+)",
                reference.get("reference_correctness", ""),
            )
            pinned_gunrock_weights = re.findall(
                r"\bweight=([0-9.eE+-]+)", reference.get("correctness", "")
            )
            observed_weight_fields = [
                re.findall(r"\bweight=([0-9.eE+-]+)", note)
                for note in validation_notes
            ]
            if (
                reference.get("reference") != "networkx"
                or len(independent_weights) != 1
                or len(pinned_gunrock_weights) != 1
                or len(observed_weight_fields) != 5
                or any(len(values) != 1 for values in observed_weight_fields)
            ):
                issues.append(
                    f"main83 {dataset}/{function}: incomplete independent "
                    "NetworkX MST-weight evidence"
                )
            else:
                independent_weight = float(independent_weights[0])
                compared_weights = [
                    ("pinned Gunrock", float(pinned_gunrock_weights[0]))
                ] + [
                    (f"fresh sample {index}", float(values[0]))
                    for index, values in enumerate(observed_weight_fields, 1)
                ]
                for label, observed_weight in compared_weights:
                    if not math.isclose(
                        observed_weight,
                        independent_weight,
                        rel_tol=MST_WEIGHT_REL_TOL,
                        abs_tol=MST_WEIGHT_ABS_TOL,
                    ):
                        issues.append(
                            f"main83 {dataset}/{function}: {label} weight "
                            f"{observed_weight:.12g} differs from independent "
                            f"NetworkX weight {independent_weight:.12g} beyond "
                            f"rel_tol={MST_WEIGHT_REL_TOL:g}, "
                            f"abs_tol={MST_WEIGHT_ABS_TOL:g}"
                        )
        else:
            reference_digests = set(
                re.findall(
                    r"\bdetail_sha=([0-9a-fA-F]+)",
                    reference.get("correctness", ""),
                )
            )
            observed_digests = {
                value
                for note in validation_notes
                for value in re.findall(r"\bdetail_sha=([0-9a-fA-F]+)", note)
            }
            # PageRank is an iterative floating-point solve.  Different valid
            # runs need not be byte-identical, so its fresh fixed-point gate
            # below replaces exact digest continuity.  The deterministic
            # integer/path/LCC outputs must remain byte-identical to the
            # pinned cross-baseline-validated result.
            if function != "PageRank" and (
                len(reference_digests) != 1
                or observed_digests != reference_digests
            ):
                issues.append(
                    f"main83 {dataset}/{function}: full-result digest continuity "
                    f"failed (new={sorted(observed_digests)}, "
                    f"reference={sorted(reference_digests)})"
                )
            if function == "PageRank" and len(observed_digests) != 1:
                issues.append(
                    f"main83 {dataset}/PageRank: exactly one fresh full-vector "
                    "digest is required"
                )
            detail_paths = {
                resolve_recorded_path(value).resolve()
                for note in validation_notes
                for value in re.findall(r"\bdetail=([^,\s]+)", note)
            }
            if len(detail_paths) != 1 or any(
                not path.is_file() or not path.is_relative_to(main_dir.resolve())
                for path in detail_paths
            ):
                issues.append(
                    f"main83 {dataset}/{function}: one in-batch full-result detail "
                    f"artifact is required, observed={sorted(map(str, detail_paths))}"
                )
        executable = executable_evidence(
            artifacts,
            APP_BY_FUNCTION[function],
            artifact_manifest,
            f"main83 {dataset}/{function}",
            issues,
        )
        if all(len(cell_samples[metric]) == 5 for metric in METRICS):
            cell, aggregate, raw = package_cell(
                dataset,
                function,
                dataset_size,
                graph_type,
                cell_samples,
                "pass",
                (
                    "fresh native/full-result validation plus pinned cross-baseline "
                    "validation continuity: "
                    + " | ".join(sorted(set(validation_notes)))
                ),
                str(main_dir.relative_to(ROOT)),
                executable,
            )
            packaged_cells.append(cell)
            aggregate_rows.extend(aggregate)
            raw_rows.extend(raw)

    if long_path.is_file():
        long_rows = [
            row
            for row in read_csv(long_path)
            if row.get("baseline") == "Gunrock" and row.get("metric") in METRICS
        ]
        long_grouped = defaultdict(list)
        for row in long_rows:
            long_grouped[
                (row.get("dataset"), row.get("function"), row.get("metric"))
            ].append(row)
        raw_lookup = {
            (row["dataset"], row["function"], row["metric"]): row
            for row in aggregate_rows
        }
        for dataset, function in sorted(main_expected):
            for metric in METRICS:
                rows_for_key = long_grouped.get((dataset, function, metric), [])
                if len(rows_for_key) != 1:
                    issues.append(
                        f"main83 aggregate {dataset}/{function}/{metric}: "
                        f"expected 1 row, observed {len(rows_for_key)}"
                    )
                    continue
                row = rows_for_key[0]
                if row.get("status") != "ok":
                    issues.append(
                        f"main83 aggregate {dataset}/{function}/{metric}: "
                        f"status={row.get('status')!r}"
                    )
                if str(row.get("n_total", "")) != "5" or str(
                    row.get("n_valid", "")
                ) != "5":
                    issues.append(
                        f"main83 aggregate {dataset}/{function}/{metric}: "
                        f"n_total/n_valid={row.get('n_total')}/{row.get('n_valid')}"
                    )
                packaged = raw_lookup.get((dataset, function, metric))
                if packaged:
                    observed_min = parse_positive(
                        row.get("min_seconds"),
                        f"main83 aggregate {dataset}/{function}/{metric} min",
                        issues,
                    )
                    expected_min = float(packaged["min_seconds"])
                    if observed_min is not None and not math.isclose(
                        observed_min, expected_min, rel_tol=1.0e-8, abs_tol=1.0e-10
                    ):
                        issues.append(
                            f"main83 aggregate {dataset}/{function}/{metric}: "
                            f"min={observed_min:.12g}, raw min={expected_min:.12g}"
                        )
    fresh_validation_path = main_dir / "correctness_validation.csv"
    if not fresh_validation_path.is_file():
        issues.append(
            f"main83 fresh correctness summary is absent: {fresh_validation_path}"
        )
    else:
        fresh_validation = {
            (row.get("dataset", ""), row.get("function", "")): row
            for row in read_csv(fresh_validation_path)
            if row.get("baseline") == "Gunrock"
        }
        for dataset, function in sorted(main_expected):
            status = fresh_validation.get((dataset, function), {}).get(
                "validation_status", ""
            )
            if function == "PageRank" and status != "pass":
                issues.append(
                    f"main83 {dataset}/PageRank: fresh fixed-point validation "
                    f"status={status!r}, expected 'pass'"
                )
            elif status not in {"pass", "inconclusive"}:
                issues.append(
                    f"main83 {dataset}/{function}: fresh validation "
                    f"status={status!r}"
                )
    return packaged_cells, aggregate_rows, raw_rows


def audit_orkut(
    orkut_dir: Path,
    excluded_roots: tuple[Path, ...],
    artifact_manifest: dict,
    issues: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    logs = {}
    packaged_cells = []
    aggregate_rows = []
    raw_rows = []
    expected_files = {
        orkut_dir / f"com-Orkut_{function}.json" for function in CORE_FUNCTIONS
    }
    observed_files = set(orkut_dir.glob("com-Orkut_*.json"))
    if observed_files != expected_files:
        issues.append(
            "Orkut per-cell JSON set differs from the seven expected files: "
            f"missing={sorted(str(path) for path in expected_files - observed_files)}, "
            f"extra={sorted(str(path) for path in observed_files - expected_files)}"
        )
    artifact_records = {}
    artifacts_path = orkut_dir / "gunrock_executable_artifacts.json"
    if artifacts_path.is_file():
        artifact_records = json.loads(artifacts_path.read_text(encoding="utf-8"))
    else:
        issues.append(f"Orkut executable artifact manifest is absent: {artifacts_path}")

    for function in CORE_FUNCTIONS:
        path = orkut_dir / f"com-Orkut_{function}.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        context = f"Orkut com-Orkut/{function}"
        if record.get("status") != "ok":
            issues.append(f"{context}: status={record.get('status')!r}")
        if record.get("validation") != "pass":
            issues.append(f"{context}: validation={record.get('validation')!r}")
        probe = record.get("external_validation_probe")
        if not isinstance(probe, dict) or probe.get("status") != "pass":
            issues.append(f"{context}: separate full-result validation did not pass")
        timing_records = record.get("timing_process_records", [])
        if int(record.get("timing_process_samples", -1)) != 5:
            issues.append(
                f"{context}: timing_process_samples="
                f"{record.get('timing_process_samples')!r}, expected 5"
            )
        if len(timing_records) != 5:
            issues.append(
                f"{context}: expected 5 timing process records, observed "
                f"{len(timing_records)}"
            )
        cell_samples = {metric: [] for metric in METRICS}
        indices = []
        for sample in timing_records:
            try:
                sample_index = int(sample.get("sample_index", ""))
                indices.append(sample_index)
            except (TypeError, ValueError):
                issues.append(
                    f"{context}: invalid sample_index={sample.get('sample_index')!r}"
                )
                continue
            if sample.get("measurement") != "timing":
                issues.append(
                    f"{context} sample {sample_index}: measurement="
                    f"{sample.get('measurement')!r}"
                )
            if sample.get("status") != "ok":
                issues.append(
                    f"{context} sample {sample_index}: status={sample.get('status')!r}"
                )
            if not is_false(sample.get("external_cli_wall_used")):
                issues.append(
                    f"{context} sample {sample_index}: external CLI wall was used"
                )
            invocations = sample.get("invocations", [])
            if len(invocations) != 1:
                issues.append(
                    f"{context} sample {sample_index}: expected one fresh native "
                    f"process, observed {len(invocations)}"
                )
                continue
            invocation = invocations[0]
            log_path, timing = parse_log(
                invocation.get("log", ""),
                orkut_dir,
                excluded_roots,
                logs,
                issues,
            )
            for metric in METRICS:
                seconds = parse_positive(
                    sample.get(
                        {
                            "build": "build_seconds",
                            "kernel": "kernel_seconds",
                            "e2e": "e2e_seconds",
                        }[metric]
                    ),
                    f"{context}/{metric} sample {sample_index}",
                    issues,
                )
                if timing is not None and seconds is not None:
                    native = float(timing[SECONDS_BY_METRIC[metric]])
                    if not math.isclose(
                        seconds, native, rel_tol=1.0e-8, abs_tol=1.0e-10
                    ):
                        issues.append(
                            f"{context}/{metric} sample {sample_index}: "
                            f"JSON={seconds:.12g}, native log={native:.12g}"
                        )
                if seconds is not None:
                    cell_samples[metric].append(
                        {
                            "sample_index": sample_index,
                            "seconds": seconds,
                            "log": str(log_path),
                        }
                    )
        if sorted(indices) != [1, 2, 3, 4, 5]:
            issues.append(f"{context}: sample indices={sorted(indices)}")
        app = APP_BY_FUNCTION[function]
        evidence_source = artifact_records
        if app not in evidence_source and isinstance(record.get("executable"), dict):
            evidence_source = {app: record["executable"]}
        executable = executable_evidence(
            evidence_source, app, artifact_manifest, context, issues
        )
        if all(len(cell_samples[metric]) == 5 for metric in METRICS):
            cell, aggregate, raw = package_cell(
                "com-Orkut",
                function,
                "large",
                "undirected",
                cell_samples,
                "pass",
                str(record.get("validation_note", "full-result validation passed")),
                str(orkut_dir.relative_to(ROOT)),
                executable,
            )
            packaged_cells.append(cell)
            aggregate_rows.extend(aggregate)
            raw_rows.extend(raw)
    aggregate_json = orkut_dir / "gunrock_large_matrix.json"
    aggregate_csv = orkut_dir / "gunrock_large_matrix.csv"
    if not aggregate_json.is_file() or not aggregate_csv.is_file():
        issues.append("Orkut aggregate JSON/CSV artifacts are incomplete")
    else:
        records = json.loads(aggregate_json.read_text(encoding="utf-8"))
        keys = {(row.get("dataset"), row.get("function")) for row in records}
        expected = {("com-Orkut", function) for function in CORE_FUNCTIONS}
        if keys != expected:
            issues.append(
                f"Orkut aggregate key set mismatch: missing={sorted(expected - keys)}, "
                f"extra={sorted(keys - expected)}"
            )
    return packaged_cells, aggregate_rows, raw_rows


def markdown_report(report: dict) -> str:
    lines = [
        "# Gunrock strict 90-cell gate",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Expected/packaged cells: {report['expected_cells']}/{report['packaged_cells']}",
        f"- Expected/packaged fresh processes: {report['expected_process_samples']}/{report['packaged_process_samples']}",
        f"- Expected/packaged phase samples: {report['expected_metric_samples']}/{report['packaged_metric_samples']}",
        "- Display estimator: minimum of five fresh processes",
        "- Dispersion: sample standard deviation (n-1)",
        "- External process wall substitution: forbidden",
        "",
        "## Included batches",
        "",
    ]
    lines.extend(f"- `{path}`" for path in report["included_batches"])
    lines.extend(["", "## Excluded batches", ""])
    lines.extend(
        f"- `{item['path']}` — {item['reason']}"
        for item in report["excluded_batches"]
    )
    lines.extend(["", "## Gate issues", ""])
    if report["issues"]:
        lines.extend(f"- {issue}" for issue in report["issues"])
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def driver(args) -> int:
    main_dir = args.main_dir.resolve()
    orkut_dir = args.orkut_dir.resolve()
    formal_manifest = json.loads(
        args.formal_manifest.read_text(encoding="utf-8")
    )
    artifact_manifest = json.loads(
        args.artifact_manifest.read_text(encoding="utf-8")
    )
    issues = []
    correctness_reference = load_main_correctness_reference(
        formal_manifest, issues
    )
    excluded_entries = formal_manifest.get("excluded_batches", [])
    excluded_roots = tuple(
        resolve_recorded_path(item["path"]).resolve() for item in excluded_entries
    )
    if not excluded_entries:
        issues.append("formal manifest has no explicit excluded-batch policy")
    for item in excluded_entries:
        if item.get("status") != "excluded_entire_batch" or item.get(
            "must_not_merge"
        ) is not True:
            issues.append(f"excluded batch policy is not strict: {item!r}")
        evidence = item.get("exclusion_evidence", {})
        if evidence.get("surviving_matching_processes") != []:
            issues.append("excluded GPU4 batch still has a matching process")
        if evidence.get("completed_cell_json") != []:
            issues.append("excluded GPU4 batch contains a completed cell JSON")
        if not evidence.get("verified_at"):
            issues.append("excluded GPU4 batch lacks timestamped exclusion evidence")
    included_entries = formal_manifest.get("included_batches", [])
    included_manifest_paths = {
        resolve_recorded_path(item["path"]).resolve()
        for item in included_entries
    }
    if {main_dir, orkut_dir} != included_manifest_paths:
        issues.append(
            "CLI included batches do not exactly match the formal manifest: "
            f"cli={sorted(map(str, (main_dir, orkut_dir)))}, "
            f"manifest={sorted(map(str, included_manifest_paths))}"
        )
    if main_dir in excluded_roots or orkut_dir in excluded_roots:
        issues.append("an included batch is also marked excluded")
    if sum(int(item.get("expected_success_cells", 0)) for item in included_entries) != 90:
        issues.append("included-batch expected success-cell counts do not sum to 90")
    for item in included_entries:
        if item.get("status") != "complete":
            issues.append(
                f"included batch is not marked complete: "
                f"{item.get('path')} status={item.get('status')!r}"
            )
        if int(item.get("physical_gpu", -1)) == 7:
            before = item.get("exclusive_prelaunch_nvidia_smi", {})
            after = item.get("exclusive_postrun_nvidia_smi", {})
            if before.get("compute_processes") != []:
                issues.append("GPU7 prelaunch snapshot contains a compute process")
            if after.get("compute_processes") != []:
                issues.append("GPU7 postrun snapshot contains a compute process")
            if not before.get("captured_at") or not after.get("captured_at"):
                issues.append("GPU7 exclusive before/after snapshots are incomplete")

    main_cells, main_aggregate, main_raw = audit_main(
        main_dir,
        excluded_roots,
        artifact_manifest,
        correctness_reference,
        issues,
    )
    orkut_cells, orkut_aggregate, orkut_raw = audit_orkut(
        orkut_dir, excluded_roots, artifact_manifest, issues
    )
    cells = main_cells + orkut_cells
    aggregate_rows = main_aggregate + orkut_aggregate
    raw_rows = main_raw + orkut_raw
    observed_cells = {(row["dataset"], row["function"]) for row in cells}
    expected = expected_cells()
    if observed_cells != expected:
        issues.append(
            f"packaged cell set mismatch: missing={sorted(expected - observed_cells)}, "
            f"extra={sorted(observed_cells - expected)}"
        )
    if len(cells) != 90:
        issues.append(f"expected exactly 90 packaged cells, observed {len(cells)}")
    if len(aggregate_rows) != 270:
        issues.append(
            f"expected exactly 270 aggregate metric rows, observed {len(aggregate_rows)}"
        )
    if len(raw_rows) != 1350:
        issues.append(
            f"expected exactly 1,350 phase-sample rows, observed {len(raw_rows)}"
        )
    for cell in cells:
        for metric, summary in cell["metrics"].items():
            if (
                summary["max_min_ratio"] > CATASTROPHIC_MAX_MIN_RATIO
                and summary["absolute_span_seconds"]
                > CATASTROPHIC_ABSOLUTE_SPAN_SECONDS
            ):
                issues.append(
                    f"{cell['dataset']}/{cell['function']}/{metric}: "
                    f"max/min={summary['max_min_ratio']:.6g} exceeds the "
                    f"catastrophic-outlier limit {CATASTROPHIC_MAX_MIN_RATIO:g} "
                    f"with span={summary['absolute_span_seconds']:.6g}s"
                )

    report = {
        "schema_version": "gunrock_strict_90_gate_v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "pass" if not issues else "fail",
        "expected_cells": 90,
        "packaged_cells": len(cells),
        "expected_process_samples": 450,
        "packaged_process_samples": len(raw_rows) // 3,
        "expected_metric_samples": 1350,
        "packaged_metric_samples": len(raw_rows),
        "included_batches": [str(main_dir), str(orkut_dir)],
        "excluded_batches": excluded_entries,
        "timing_protocol_version": PROTOCOL_VERSION,
        "display_estimator": "minimum_of_five_fresh_processes",
        "dispersion_estimator": "sample_standard_deviation_n_minus_1",
        "external_cli_wall_allowed": False,
        "mst_independent_reference": "networkx",
        "mst_weight_relative_tolerance": MST_WEIGHT_REL_TOL,
        "mst_weight_absolute_tolerance": MST_WEIGHT_ABS_TOL,
        "catastrophic_max_min_ratio_limit": CATASTROPHIC_MAX_MIN_RATIO,
        "catastrophic_absolute_span_seconds": (
            CATASTROPHIC_ABSOLUTE_SPAN_SECONDS
        ),
        "issues": issues,
        "cells": cells,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gate_json = args.out_dir / "gunrock_strict_90_gate.json"
    gate_md = args.out_dir / "gunrock_strict_90_gate.md"
    atomic_text(gate_json, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    atomic_text(gate_md, markdown_report(report))
    if issues:
        return 1

    aggregate_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["function"],
            METRICS.index(row["metric"]),
        )
    )
    raw_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["function"],
            int(row["sample_index"]),
            METRICS.index(row["metric"]),
        )
    )
    overlay_path = args.out_dir / "gunrock_strict_overlay_results_long.csv"
    raw_path = args.out_dir / "gunrock_strict_overlay_samples.csv"
    preferred_overlay = (
        "dataset_size",
        "graph_type",
        "dataset",
        "function",
        "baseline",
        "metric",
        "seconds",
        "value",
        "unit",
        "metric_family",
        "measurement_scope",
        "timer_kind",
        "measurement_window",
        "status",
        "correctness",
        "validation",
        "log",
        "notes",
        "estimator_kind",
        "sample_index",
        "sample_count",
        "measurement_phase",
        "timing_protocol_version",
        "external_cli_wall_used",
        "aggregation",
        "n_total",
        "n_valid",
        "publishable",
        "mean_seconds",
        "std_seconds",
        "sample_sd_seconds",
        "variance_seconds2",
        "cv",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "relative_std_percent",
        "max_min_ratio",
        "sample_seconds_json",
        "sample_logs_json",
        "source_batch",
        "executable_app",
        "executable_sha256",
        "source_commit",
        "source_diff_sha256",
        "source_manifest_sha256",
        "boundary_contract",
    )
    preferred_raw = (
        "dataset_size",
        "graph_type",
        "dataset",
        "function",
        "baseline",
        "metric",
        "sample_index",
        "sample_count",
        "seconds",
        "status",
        "validation",
        "measurement_scope",
        "timer_kind",
        "measurement_window",
        "timing_protocol_version",
        "external_cli_wall_used",
        "log",
        "source_batch",
        "executable_app",
        "executable_sha256",
    )
    atomic_csv(overlay_path, aggregate_rows, preferred_overlay)
    atomic_csv(raw_path, raw_rows, preferred_raw)
    wide_rows = []
    wide_prefix = {
        "build": "construction",
        "kernel": "device_processing",
        "e2e": "standalone_e2e",
    }
    for cell in sorted(cells, key=lambda item: (item["dataset"], item["function"])):
        row = {
            "dataset_size": cell["dataset_size"],
            "graph_type": cell["graph_type"],
            "dataset": cell["dataset"],
            "function": cell["function"],
            "baseline": "Gunrock",
            "status": "ok",
            "validation": cell["validation"],
            "validation_note": cell["validation_note"],
            "fresh_process_samples": "5",
            "display_estimator": "minimum_of_five_fresh_processes",
            "dispersion_estimator": "sample_standard_deviation_n_minus_1",
            "timing_protocol_version": PROTOCOL_VERSION,
            "external_cli_wall_used": "false",
            "source_batch": cell["source_batch"],
            "executable_app": cell["executable"].get("app", ""),
            "executable_sha256": cell["executable"].get("sha256", ""),
            "source_commit": cell["executable"].get("source_commit", ""),
            "source_diff_sha256": cell["executable"].get(
                "source_diff_sha256", ""
            ),
        }
        for metric, prefix in wide_prefix.items():
            summary = cell["metrics"][metric]
            row[f"{prefix}_min_seconds"] = f"{summary['min']:.12g}"
            row[f"{prefix}_mean_seconds"] = f"{summary['mean']:.12g}"
            row[f"{prefix}_sample_sd_seconds"] = (
                f"{summary['sample_sd']:.12g}"
            )
            row[f"{prefix}_max_min_ratio"] = (
                f"{summary['max_min_ratio']:.12g}"
            )
            row[f"{prefix}_samples_json"] = json.dumps(
                summary["samples"], separators=(",", ":")
            )
        wide_rows.append(row)
    wide_path = args.out_dir / "gunrock_strict_90_cells.csv"
    preferred_wide = (
        "dataset_size",
        "graph_type",
        "dataset",
        "function",
        "baseline",
        "status",
        "validation",
        "validation_note",
        "fresh_process_samples",
        "display_estimator",
        "dispersion_estimator",
        "timing_protocol_version",
        "external_cli_wall_used",
        "construction_min_seconds",
        "construction_mean_seconds",
        "construction_sample_sd_seconds",
        "construction_samples_json",
        "construction_max_min_ratio",
        "device_processing_min_seconds",
        "device_processing_mean_seconds",
        "device_processing_sample_sd_seconds",
        "device_processing_samples_json",
        "device_processing_max_min_ratio",
        "standalone_e2e_min_seconds",
        "standalone_e2e_mean_seconds",
        "standalone_e2e_sample_sd_seconds",
        "standalone_e2e_samples_json",
        "standalone_e2e_max_min_ratio",
        "source_batch",
        "executable_app",
        "executable_sha256",
        "source_commit",
        "source_diff_sha256",
    )
    atomic_csv(wide_path, wide_rows, preferred_wide)
    replacement_keys = sorted(
        f"{row['dataset']}\t{row['function']}\tGunrock\t{row['metric']}"
        for row in aggregate_rows
    )
    replacement_key_sha = hashlib.sha256(
        ("\n".join(replacement_keys) + "\n").encode("utf-8")
    ).hexdigest()
    success_cell_keys = sorted(
        f"{dataset}\t{function}" for dataset, function in expected_cells()
    )
    success_cell_key_sha = hashlib.sha256(
        ("\n".join(success_cell_keys) + "\n").encode("utf-8")
    ).hexdigest()
    used_apps = sorted(set(APP_BY_FUNCTION.values()))
    overlay_manifest = {
        "schema_version": "gunrock_strict_overlay_v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "gate_status": "pass",
        "gate_json": str(gate_json),
        "gate_json_sha256": sha256_file(gate_json),
        "formal_run_manifest": str(args.formal_manifest.resolve()),
        "formal_run_manifest_sha256": sha256_file(args.formal_manifest.resolve()),
        "gunrock_artifact_manifest": str(args.artifact_manifest.resolve()),
        "gunrock_artifact_manifest_sha256": sha256_file(
            args.artifact_manifest.resolve()
        ),
        "overlay_csv": str(overlay_path),
        "overlay_csv_sha256": sha256_file(overlay_path),
        "raw_samples_csv": str(raw_path),
        "raw_samples_csv_sha256": sha256_file(raw_path),
        "wide_90_cell_csv": str(wide_path),
        "wide_90_cell_csv_sha256": sha256_file(wide_path),
        "replacement_key_fields": ["dataset", "function", "baseline", "metric"],
        "replacement_key_count": len(replacement_keys),
        "replacement_key_sha256": replacement_key_sha,
        "success_cell_count": 90,
        "success_cell_key_sha256": success_cell_key_sha,
        "metrics_per_cell": 3,
        "fresh_process_samples_per_cell": 5,
        "phase_sample_count": 1350,
        "display_estimator": "minimum_of_five_fresh_processes",
        "dispersion_estimator": "sample_standard_deviation_n_minus_1",
        "timing_protocol_version": PROTOCOL_VERSION,
        "external_cli_wall_allowed": False,
        "mst_independent_reference": "networkx",
        "mst_weight_relative_tolerance": MST_WEIGHT_REL_TOL,
        "mst_weight_absolute_tolerance": MST_WEIGHT_ABS_TOL,
        "catastrophic_max_min_ratio_limit": CATASTROPHIC_MAX_MIN_RATIO,
        "catastrophic_absolute_span_seconds": (
            CATASTROPHIC_ABSOLUTE_SPAN_SECONDS
        ),
        "included_batches": [str(main_dir), str(orkut_dir)],
        "included_batch_records": formal_manifest.get("included_batches", []),
        "excluded_batches": excluded_entries,
        "executable_provenance": {
            app: artifact_manifest.get("executables", {}).get(app, {})
            for app in used_apps
        },
        "main_correctness_reference": formal_manifest.get(
            "main_correctness_reference", {}
        ),
        "validation": "all 90 cells passed their native or separate full-result gate",
        "immutable_source_policy": (
            "overlay into a new output file; never modify the V10/base ledger in place"
        ),
    }
    manifest_path = args.out_dir / "gunrock_strict_overlay_manifest.json"
    atomic_text(
        manifest_path,
        json.dumps(overlay_manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-dir",
        type=Path,
        default=ROOT
        / "benchmarking/results/gunrock_strict_main83_gpu3_20260730",
    )
    parser.add_argument(
        "--orkut-dir",
        type=Path,
        default=ROOT
        / "benchmarking/results/gunrock_strict_orkut7_gpu7_20260730",
    )
    parser.add_argument(
        "--formal-manifest",
        type=Path,
        default=ROOT / "benchmarking/gunrock_strict_formal_manifest_20260730.json",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=ROOT / "benchmarking/gunrock_artifact_manifest.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "benchmarking/results/gunrock_strict_formal_audit_20260730",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(driver(parse_args()))
