#!/usr/bin/env python3
"""Audit a full EGGPU benchmark result directory.

The full benchmark intentionally produces a lot of rows: timing, memory,
correctness details, supported/unsupported baselines, and timeout markers.  This
script condenses the result into a gate-oriented report:

- EGGPU correctness failures are hard blockers.
- EGGPU runtime failures/timeouts are hard blockers.
- non-EGGPU correctness failures are reported separately as baseline semantic or
  implementation issues.
- missing baseline coverage is summarized by function.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LEGACY_EXPECTED_FUNCTIONS = [
    "PageRank",
    "MST",
    "LCC",
    "WCC",
    "SCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
    "BC",
    "Closeness",
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
]

EXPECTED_BASELINES = [
    "networkx",
    "easygraph-cpu",
    "easygraph-cpp",
    "igraph",
    "nx-cugraph",
    "Gunrock",
    "EGGPU",
]

PASSLIKE = {"pass", "weak_pass", "reference"}
EGGPU_PASSLIKE = {"pass", "sampled_pass"}
HARD_RUNTIME_BAD = {"failed", "timeout", "incomplete"}


def is_accepted_eggpu_skip(row):
    return (
        row.get("baseline") == "EGGPU"
        and row.get("function") == "Closeness"
        and row.get("status") == "skipped"
        and (
            row.get("skip_reason") == "exact_scale_guard"
            or "exact all-source Closeness skipped by symmetric scale guard" in row.get("notes", "")
        )
    )


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_read_error": repr(exc)}


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _status_counts(rows, key):
    c = Counter()
    for r in rows:
        c[r.get(key, "")] += 1
    return dict(sorted(c.items()))


def _dataset_count(rows):
    return len({r.get("dataset") for r in rows if r.get("dataset")})


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value, default=None):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _expected_non_process_gpu_baseline_mb(metadata):
    """Return the idle device-resident baseline observed at run start.

    NVML's device total includes driver/runtime allocations that are not owned
    by any compute process.  The visibility marker is the sole expected compute
    process at this point, so subtracting its process-attributed memory leaves
    the device-resident baseline that should not be reported as contamination.
    """

    profile = (metadata or {}).get("gpu_device_profile") or {}
    selected = profile.get("selected_device") or {}
    used = _safe_float(selected.get("memory_used_mb_at_run_start"))
    process_used = _safe_float(selected.get("compute_process_memory_mb_at_run_start"))
    process_count = _safe_int(selected.get("compute_process_count_at_run_start"), default=-1)
    utilization = _safe_float(selected.get("gpu_utilization_percent_at_run_start"))
    if (
        used is None
        or process_used is None
        or process_count < 0
        or process_count > 1
        or utilization is None
        or utilization > 0.0
    ):
        return None
    return max(0.0, used - process_used)


def audit_memory_rows(long_rows, metadata=None):
    """Flag memory rows whose whole-device value is likely externally polluted.

    `memory_peak_gpu_mb` is intentionally a whole-device NVML reading, so it can
    include unrelated processes.  The comparable benchmark metric is
    `memory_peak_gpu_proc_mb` when NVML can attribute memory to the benchmark
    process tree.  These rows are warnings, not gate blockers.
    """
    by_key = defaultdict(dict)
    for r in long_rows:
        metric = r.get("metric", "")
        if not metric.startswith("memory_"):
            continue
        key = (
            r.get("dataset", ""),
            r.get("function", ""),
            r.get("baseline", ""),
            r.get("status", ""),
        )
        by_key[key][metric] = r

    issues = []
    expected_non_process = _expected_non_process_gpu_baseline_mb(metadata)
    for (dataset, function, baseline, status), metrics in sorted(by_key.items()):
        peak_row = metrics.get("memory_peak_gpu_mb")
        proc_row = metrics.get("memory_peak_gpu_proc_mb")
        non_process_start_row = metrics.get("memory_start_gpu_non_process_mb")

        # The visibility marker is subtracted before this metric is emitted.
        # Compare the remainder with the device-resident baseline captured at
        # run start; only growth beyond that baseline indicates contamination.
        non_process_start = None
        if non_process_start_row is not None:
            non_process_start = _safe_float(
                non_process_start_row.get("value", non_process_start_row.get("seconds"))
            )
        contamination_floor = (expected_non_process or 0.0) + 256.0
        if non_process_start is not None and non_process_start > contamination_floor:
            issues.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "baseline": baseline,
                    "status": status,
                    "issue": "non_benchmark_gpu_memory_present_at_measurement_start",
                    "memory_peak_gpu_mb": "",
                    "memory_peak_gpu_proc_mb": "",
                    "non_process_start_mb": f"{non_process_start:.6g}",
                    "expected_non_process_baseline_mb": (
                        "" if expected_non_process is None else f"{expected_non_process:.6g}"
                    ),
                    "excess_device_over_proc_mb": f"{non_process_start:.6g}",
                    "severity": "warning",
                    "note": "non-process GPU memory exceeded the run-start device baseline by more than 256 MiB; process-tree memory remains the comparable metric",
                }
            )

        if peak_row is None or proc_row is None:
            continue
        peak = _safe_float(peak_row.get("value", peak_row.get("seconds")))
        proc = _safe_float(proc_row.get("value", proc_row.get("seconds")))
        if peak is None or proc is None:
            continue
        excess = peak - proc
        if baseline in {"easygraph-cpu", "easygraph-cpp", "networkx", "igraph"} and proc <= 1.0 and peak > 4096.0:
            issues.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "baseline": baseline,
                    "status": status,
                    "issue": "cpu_baseline_whole_device_gpu_memory_high",
                    "memory_peak_gpu_mb": f"{peak:.6g}",
                    "memory_peak_gpu_proc_mb": f"{proc:.6g}",
                    "non_process_start_mb": "",
                    "expected_non_process_baseline_mb": (
                        "" if expected_non_process is None else f"{expected_non_process:.6g}"
                    ),
                    "excess_device_over_proc_mb": f"{excess:.6g}",
                    "severity": "warning",
                    "note": "whole-device GPU memory likely includes unrelated or persistent GPU processes; use process-tree GPU memory for paper comparisons",
                }
            )
        elif excess > 4096.0:
            issues.append(
                {
                    "dataset": dataset,
                    "function": function,
                    "baseline": baseline,
                    "status": status,
                    "issue": "whole_device_gpu_memory_exceeds_process_tree_by_over_4gb",
                    "memory_peak_gpu_mb": f"{peak:.6g}",
                    "memory_peak_gpu_proc_mb": f"{proc:.6g}",
                    "non_process_start_mb": "",
                    "expected_non_process_baseline_mb": (
                        "" if expected_non_process is None else f"{expected_non_process:.6g}"
                    ),
                    "excess_device_over_proc_mb": f"{excess:.6g}",
                    "severity": "warning",
                    "note": "device-level GPU memory is not process-isolated; inspect process-tree memory and GPU-idle evidence before using whole-device memory",
                }
            )
    return issues


def audit_measurement_schema(result_dir: Path, long_rows):
    """Validate units, timer provenance, and primary EGGPU memory coverage."""

    issues = []

    def add(issue, severity="hard", row=None, details=""):
        row = row or {}
        issues.append(
            {
                "dataset": row.get("dataset", ""),
                "function": row.get("function", ""),
                "baseline": row.get("baseline", ""),
                "metric": row.get("metric", ""),
                "issue": issue,
                "severity": severity,
                "details": details,
            }
        )

    schema = read_json(result_dir / "measurement_schema.json")
    if schema is None:
        add("measurement_schema.json missing")
    elif schema.get("_read_error"):
        add("measurement_schema.json invalid", details=str(schema.get("_read_error")))
    elif _safe_int(schema.get("schema_version"), 0) < 2:
        add(
            "measurement schema version too old",
            details=f"schema_version={schema.get('schema_version')}",
        )

    metadata = read_json(result_dir / "run_metadata.json") or {}
    protocol = metadata.get("measurement_protocol") or {}
    split_protocol = protocol.get("protocol") == "split_timing_memory_v1"
    if split_protocol and _safe_int((schema or {}).get("schema_version"), 0) < 4:
        add("split measurement run requires schema version 4 or newer")
    if split_protocol and protocol.get("source_snapshot_match") is not True:
        implementation_match = (
            protocol.get("implementation_source_snapshot_match") is True
        )
        artifact_match = protocol.get("cpp_easygraph_artifact_match") is True
        contract_match = protocol.get("benchmark_contract_match") is True
        if implementation_match and artifact_match and contract_match:
            add(
                "broad repository source snapshot differs between timing and memory passes",
                severity="warning",
                details=(
                    "EGGPU implementation snapshot, loaded cpp_easygraph artifact, "
                    "and benchmark contract match; drift is outside the measured "
                    "implementation and contract"
                ),
            )
        else:
            add("timing and memory passes lack a verified matching source snapshot")
    if split_protocol and protocol.get("cpp_easygraph_artifact_match") is not True:
        add("timing and memory passes lack a verified matching cpp_easygraph artifact")

    required_fields = ["value", "unit", "metric_family", "measurement_scope", "timer_kind"]
    if _safe_int((schema or {}).get("schema_version"), 0) >= 3:
        required_fields.append("measurement_window")
    for row in long_rows:
        metric = str(row.get("metric", ""))
        phase = str(row.get("measurement_phase", ""))
        if split_protocol:
            if metric in {"build", "e2e", "kernel"} and phase != "timing":
                add(
                    "paper timing metric did not originate from timing pass",
                    row=row,
                    details=f"measurement_phase={phase}",
                )
            if metric.startswith("memory_") and phase != "memory":
                add(
                    "memory metric did not originate from memory pass",
                    row=row,
                    details=f"measurement_phase={phase}",
                )
        if row.get("status") != "ok" or str(row.get("seconds", "")).strip() == "":
            continue
        missing = [field for field in required_fields if str(row.get(field, "")).strip() == ""]
        if missing:
            add("numeric row missing measurement provenance", row=row, details=",".join(missing))
            continue
        seconds = _safe_float(row.get("seconds"))
        value = _safe_float(row.get("value"))
        tolerance = max(1e-12, abs(seconds or 0.0) * 1e-9)
        if seconds is None or value is None or abs(seconds - value) > tolerance:
            add(
                "legacy seconds alias differs from canonical value",
                row=row,
                details=f"seconds={row.get('seconds')}, value={row.get('value')}",
            )
        if row.get("metric") == "kernel" and row.get("baseline") == "EGGPU":
            if row.get("timer_kind") != "cuda_event":
                add("EGGPU kernel row is not CUDA-event timed", row=row)

    ok_eggpu_pairs = {
        (row.get("dataset"), row.get("function"))
        for row in long_rows
        if row.get("baseline") == "EGGPU"
        and row.get("metric") == "e2e"
        and row.get("status") == "ok"
    }
    proc_peak_pairs = {
        (row.get("dataset"), row.get("function"))
        for row in long_rows
        if row.get("baseline") == "EGGPU"
        and row.get("metric") == "memory_peak_gpu_proc_mb"
        and row.get("status") == "ok"
        and _safe_float(row.get("value", row.get("seconds"))) is not None
    }
    for dataset, function in sorted(ok_eggpu_pairs - proc_peak_pairs):
        add(
            "missing primary process-tree GPU memory peak",
            row={"dataset": dataset, "function": function, "baseline": "EGGPU"},
        )

    if split_protocol:
        probe_pairs = {
            (row.get("dataset"), row.get("function"))
            for row in long_rows
            if row.get("baseline") == "EGGPU"
            and row.get("metric") == "memory_probe_calls"
            and row.get("status") == "ok"
            and (_safe_float(row.get("value", row.get("seconds")), 0.0) or 0.0) >= 1.0
        }
        for dataset, function in sorted(ok_eggpu_pairs - probe_pairs):
            add(
                "missing independent memory-probe call count",
                row={"dataset": dataset, "function": function, "baseline": "EGGPU"},
            )

    for row in long_rows:
        if (
            row.get("baseline") == "EGGPU"
            and row.get("metric") in {
                "memory_monitor_gpu_samples",
                "memory_monitor_gpu_proc_samples",
            }
            and row.get("status") == "ok"
        ):
            samples = _safe_float(row.get("value", row.get("seconds")), 0.0)
            if samples < 2:
                add(
                    "memory monitor has fewer than two samples",
                    severity="warning",
                    row=row,
                    details=f"mean_samples={samples}",
                )
    return issues


def audit_sample_contract(result_dir: Path, long_rows, expected_samples: int):
    sample_rows = read_csv(result_dir / "results_samples.csv")
    if not sample_rows:
        return [
            {
                "dataset": "",
                "function": "",
                "baseline": "",
                "metric": "",
                "issue": "results_samples.csv missing or empty",
                "severity": "hard",
                "details": "independent raw timing samples are required",
            }
        ]

    key_fields = ("dataset", "function", "baseline", "metric")
    grouped = defaultdict(list)
    for row in sample_rows:
        grouped[tuple(row.get(field, "") for field in key_fields)].append(row)

    issues = []
    for aggregate in long_rows:
        key = tuple(aggregate.get(field, "") for field in key_fields)
        samples = grouped.get(key, [])
        status = aggregate.get("status", "")
        severity = "hard" if aggregate.get("baseline") == "EGGPU" else "warning"
        row_expected_samples = _safe_int(
            aggregate.get("sample_count"), expected_samples
        )
        if row_expected_samples < 1:
            row_expected_samples = expected_samples
        if status == "incomplete":
            issues.append(
                {
                    **{field: aggregate.get(field, "") for field in key_fields},
                    "issue": "incomplete_measured_group",
                    "severity": severity,
                    "details": f"n_total={aggregate.get('n_total', '')}, n_valid={aggregate.get('n_valid', '')}, expected={row_expected_samples}",
                }
            )
            continue
        if status != "ok":
            continue

        contract_errors = []
        if aggregate.get("aggregation") != "arithmetic_mean":
            contract_errors.append("aggregation is not arithmetic_mean")
        if _safe_int(aggregate.get("n_total"), -1) != row_expected_samples:
            contract_errors.append(f"n_total={aggregate.get('n_total', '')}")
        if _safe_int(aggregate.get("n_valid"), -1) != row_expected_samples:
            contract_errors.append(f"n_valid={aggregate.get('n_valid', '')}")
        if str(aggregate.get("publishable", "")).lower() != "true":
            contract_errors.append("publishable is not true")
        if len(samples) != row_expected_samples:
            contract_errors.append(f"raw sample rows={len(samples)}")
        indices = sorted(_safe_int(row.get("sample_index"), -1) for row in samples)
        if indices != list(range(1, row_expected_samples + 1)):
            contract_errors.append(f"sample indices={indices}")
        if any(row.get("status") != "ok" for row in samples):
            contract_errors.append("at least one raw sample is non-ok")
        values = [_safe_float(row.get("seconds")) for row in samples]
        if values and all(value is not None for value in values):
            raw_mean = sum(values) / len(values)
            aggregate_mean = _safe_float(aggregate.get("seconds"))
            tolerance = max(1e-12, abs(raw_mean) * 1e-9)
            if aggregate_mean is None or abs(aggregate_mean - raw_mean) > tolerance:
                contract_errors.append(
                    f"aggregate mean={aggregate_mean} does not match raw arithmetic mean={raw_mean}"
                )
        required_stat_fields = ["mean_seconds"]
        if row_expected_samples >= 2:
            required_stat_fields.extend(["std_seconds", "variance_seconds2"])
        for field in required_stat_fields:
            if aggregate.get(field, "") == "":
                contract_errors.append(f"missing {field}")
        for field in ("correctness_variant_count", "correctness_consistency"):
            if aggregate.get(field, "") == "":
                contract_errors.append(f"missing {field}")
        if contract_errors:
            issues.append(
                {
                    **{field: aggregate.get(field, "") for field in key_fields},
                    "issue": "invalid_repeat_statistics_contract",
                    "severity": "hard",
                    "details": "; ".join(contract_errors),
                }
            )
    return issues


def audit_metadata(
    result_dir: Path,
    datasets: list[str],
    functions: list[str],
    expected_repeat: int = 5,
):
    metadata_path = result_dir / "run_metadata.json"
    metadata = read_json(metadata_path)
    issues = []

    def issue(field, expected, actual, severity="hard"):
        issues.append(
            {
                "field": field,
                "expected": str(expected),
                "actual": str(actual),
                "severity": severity,
            }
        )

    if metadata is None:
        issue("run_metadata.json", "present", "missing")
        return metadata, issues
    if metadata.get("_read_error"):
        issue("run_metadata.json", "valid JSON", metadata.get("_read_error"))
        return metadata, issues

    if metadata.get("schema_version") != 1:
        issue("schema_version", 1, metadata.get("schema_version"))
    if not metadata.get("completed_at"):
        issue("completed_at", "non-empty", metadata.get("completed_at"))

    source_snapshot = metadata.get("source_snapshot") or {}
    snapshot_digest = str(source_snapshot.get("digest", ""))
    snapshot_valid = (
        source_snapshot.get("algorithm") == "sha256"
        and re.fullmatch(r"[0-9a-f]{64}", snapshot_digest) is not None
        and _safe_int(source_snapshot.get("file_count"), 0) > 0
    )
    if not snapshot_valid:
        issue(
            "source_snapshot",
            "non-empty SHA-256 snapshot of the executed source tree",
            source_snapshot,
        )

    py_meta = metadata.get("python") or {}
    if not py_meta.get("executable"):
        issue("python.executable", "non-empty", py_meta.get("executable"))
    if not py_meta.get("direct_child_python"):
        issue("python.direct_child_python", "non-empty", py_meta.get("direct_child_python"))

    baseline_versions = metadata.get("baseline_versions") or {}
    evaluated = baseline_versions.get("evaluated_baselines") or []
    if "cugraph" in evaluated:
        issue("baseline_versions.evaluated_baselines", "native cugraph excluded", evaluated)
    for baseline in ("easygraph", "cpp_easygraph", "networkx", "igraph", "nx-cugraph"):
        version = (baseline_versions.get(baseline) or {}).get("version", "")
        if not version or version == "not-installed":
            issue(f"baseline_versions.{baseline}.version", "installed version", version)
        module_origin = (baseline_versions.get(baseline) or {}).get("module_origin", "")
        if not module_origin or module_origin == "not-found":
            issue(f"baseline_versions.{baseline}.module_origin", "loaded module path", module_origin)
    cugraph_role = (
        ((baseline_versions.get("runtime_dependencies") or {}).get("cugraph") or {}).get("role", "")
    )
    if cugraph_role != "nx-cugraph runtime dependency; not an evaluated baseline":
        issue(
            "baseline_versions.runtime_dependencies.cugraph.role",
            "nx-cugraph runtime dependency; not an evaluated baseline",
            cugraph_role,
        )

    cuda_meta = metadata.get("cuda") or {}
    if not cuda_meta.get("local_cuda_root"):
        issue("cuda.local_cuda_root", "non-empty", cuda_meta.get("local_cuda_root"))
    if not cuda_meta.get("nvcc"):
        issue("cuda.nvcc", "non-empty", cuda_meta.get("nvcc"))

    build_artifacts = metadata.get("build_artifacts") or {}
    artifacts = build_artifacts.get("cpp_easygraph") or []
    if not artifacts:
        issue("build_artifacts.cpp_easygraph", "at least one shared library", "empty")
    for artifact_index, artifact in enumerate(artifacts):
        if not artifact.get("path") or not artifact.get("sha256"):
            issue(
                f"build_artifacts.cpp_easygraph[{artifact_index}]",
                "path and sha256",
                artifact,
            )
    cpp_origin = (baseline_versions.get("cpp_easygraph") or {}).get("module_origin", "")
    artifact_paths = {
        str(Path(item.get("path", "")).resolve())
        for item in artifacts
        if item.get("path")
    }
    active_cpp_easygraph = build_artifacts.get("active_cpp_easygraph") or {}
    if active_cpp_easygraph.get("path"):
        artifact_paths.add(str(Path(active_cpp_easygraph["path"]).resolve()))
    if cpp_origin and str(Path(cpp_origin).resolve()) not in artifact_paths:
        issue(
            "baseline_versions.cpp_easygraph.module_origin",
            "one of build_artifacts.cpp_easygraph or active_cpp_easygraph paths",
            cpp_origin,
        )
    gunrock_artifacts = (metadata.get("build_artifacts") or {}).get("gunrock_executables") or {}
    for executable in ("pr", "mst", "lcc", "bfs", "sssp", "kcore", "bc"):
        entry = gunrock_artifacts.get(executable) or {}
        if entry.get("status") != "available" or not entry.get("path") or not entry.get("sha256"):
            issue(
                f"build_artifacts.gunrock_executables.{executable}",
                "available with path and sha256",
                entry or "missing",
            )
            continue
        for field in (
            "upstream_release_lineage",
            "application_generation",
            "benchmark_support",
            "source_remote",
            "source_commit",
            "source_version",
            "source_diff_sha256",
            "build_cuda_toolkit_version",
            "build_cuda_architecture",
        ):
            if not entry.get(field):
                issue(
                    f"build_artifacts.gunrock_executables.{executable}.{field}",
                    "non-empty provenance field",
                    entry.get(field),
                )
    run_artifacts = metadata.get("artifacts") or {}
    if not run_artifacts:
        issue("artifacts", "present after completed run", "missing")
    if run_artifacts.get("validation_error"):
        issue("artifacts.validation_error", "empty", run_artifacts.get("validation_error"))
    if not (result_dir / "baseline_versions.json").exists():
        issue("baseline_versions.json", "present", "missing")
    if not (result_dir / "measurement_schema.json").exists():
        issue("measurement_schema.json", "present", "missing")

    bench_args = metadata.get("benchmark_args") or {}
    meta_datasets = bench_args.get("datasets") or []
    meta_functions = bench_args.get("functions") or []
    if sorted(meta_datasets) != sorted(datasets):
        issue("benchmark_args.datasets", sorted(datasets), sorted(meta_datasets))
    if sorted(meta_functions) != sorted(functions):
        issue("benchmark_args.functions", sorted(functions), sorted(meta_functions))
    if _safe_int(bench_args.get("warmup"), default=-1) != 0:
        issue("benchmark_args.warmup", 0, bench_args.get("warmup"))
    if _safe_int(bench_args.get("easygraph_warmup"), default=-1) != 2:
        issue("benchmark_args.easygraph_warmup", 2, bench_args.get("easygraph_warmup"))
    if _safe_int(bench_args.get("repeat"), default=-1) != expected_repeat:
        issue("benchmark_args.repeat", expected_repeat, bench_args.get("repeat"))

    env_meta = metadata.get("environment") or {}
    if env_meta.get("EASYGRAPH_ENABLE_GPU") != "TRUE":
        issue("environment.EASYGRAPH_ENABLE_GPU", "TRUE", env_meta.get("EASYGRAPH_ENABLE_GPU"))
    if env_meta.get("EGGPU_USE_CONDA_RUN") != "FALSE":
        issue("environment.EGGPU_USE_CONDA_RUN", "FALSE", env_meta.get("EGGPU_USE_CONDA_RUN"))

    return metadata, issues


def audit(result_dir: Path, expected_repeat: int = 5):
    long_rows = read_csv(result_dir / "results_long.csv")
    validation_rows = read_csv(result_dir / "correctness_validation.csv")
    if not long_rows:
        raise SystemExit(f"missing or empty results_long.csv in {result_dir}")
    validation_issues = []

    e2e_rows = [r for r in long_rows if r.get("metric") == "e2e"]
    eggpu_e2e = [r for r in e2e_rows if r.get("baseline") == "EGGPU"]
    eggpu_runtime_bad = [
        r
        for r in eggpu_e2e
        if not is_accepted_eggpu_skip(r)
        and (r.get("status") in HARD_RUNTIME_BAD or not r.get("seconds"))
    ]

    eggpu_validation = [r for r in validation_rows if r.get("baseline") == "EGGPU"]
    eggpu_validation_bad = [
        r for r in eggpu_validation if r.get("validation_status") not in EGGPU_PASSLIKE
    ]
    baseline_validation_bad = [
        r
        for r in validation_rows
        if r.get("baseline") != "EGGPU"
        and r.get("validation_status") not in PASSLIKE
        and r.get("validation_status") != "semantic_mismatch"
    ]

    coverage = defaultdict(lambda: defaultdict(Counter))
    for r in e2e_rows:
        coverage[r.get("function")][r.get("baseline")][r.get("status")] += 1

    missing_rows = []
    datasets = sorted({r.get("dataset") for r in e2e_rows if r.get("dataset")})

    metadata = read_json(result_dir / "run_metadata.json")
    metadata_functions = []
    if isinstance(metadata, dict) and not metadata.get("_read_error"):
        metadata_functions = (metadata.get("benchmark_args") or {}).get("functions") or []
    observed_functions = sorted({r.get("function") for r in e2e_rows if r.get("function")})
    expected_functions = sorted(set(metadata_functions or observed_functions or LEGACY_EXPECTED_FUNCTIONS))

    for function in expected_functions:
        for baseline in EXPECTED_BASELINES:
            have = [r for r in e2e_rows if r.get("function") == function and r.get("baseline") == baseline]
            ok_count = sum(1 for r in have if r.get("status") == "ok")
            skipped_count = sum(1 for r in have if r.get("status") == "skipped")
            accepted_skipped_count = sum(1 for r in have if is_accepted_eggpu_skip(r))
            failed_count = sum(1 for r in have if r.get("status") in HARD_RUNTIME_BAD)
            if not have:
                missing_rows.append(
                    {
                        "function": function,
                        "baseline": baseline,
                        "issue": "no e2e rows",
                        "ok": 0,
                        "skipped": 0,
                        "failed": 0,
                    }
                )
            elif baseline == "EGGPU" and ok_count + accepted_skipped_count != len(datasets):
                missing_rows.append(
                    {
                        "function": function,
                        "baseline": baseline,
                        "issue": "EGGPU not ok or accepted-skipped on every dataset",
                        "ok": ok_count,
                        "skipped": skipped_count,
                        "failed": failed_count,
                    }
                )

    metadata, metadata_issues = audit_metadata(
        result_dir,
        datasets,
        observed_functions,
        expected_repeat=expected_repeat,
    )
    expected_samples = _safe_int(
        (((metadata or {}).get("benchmark_args") or {}).get("repeat")),
        5,
    )
    sample_issues = audit_sample_contract(result_dir, long_rows, expected_samples)
    hard_sample_issues = [row for row in sample_issues if row.get("severity") == "hard"]
    hard_metadata_issues = [r for r in metadata_issues if r.get("severity") != "warning"]
    eggpu_coverage_issues = [r for r in missing_rows if r.get("baseline") == "EGGPU"]
    expected_eggpu_validation = {
        (r.get("dataset"), r.get("function"))
        for r in eggpu_e2e
        if r.get("status") == "ok"
    }
    observed_eggpu_validation = {
        (r.get("dataset"), r.get("function"))
        for r in eggpu_validation
        if r.get("validation_status") in EGGPU_PASSLIKE
    }
    missing_eggpu_validation = sorted(expected_eggpu_validation - observed_eggpu_validation)
    if not validation_rows and expected_eggpu_validation:
        validation_issues.append(
            {
                "dataset": "",
                "function": "",
                "issue": "correctness_validation.csv missing or empty",
            }
        )
    for dataset, function in missing_eggpu_validation:
        validation_issues.append(
            {
                "dataset": dataset,
                "function": function,
                "issue": "missing EGGPU pass validation for ok e2e row",
            }
        )
    memory_issues = audit_memory_rows(long_rows, metadata=metadata)
    measurement_issues = audit_measurement_schema(result_dir, long_rows)
    hard_measurement_issues = [
        row for row in measurement_issues if row.get("severity") == "hard"
    ]

    out_dir = result_dir / "audit"
    out_dir.mkdir(exist_ok=True)

    write_csv(
        out_dir / "eggpu_runtime_bad.csv",
        eggpu_runtime_bad,
        list(long_rows[0].keys()) if long_rows else [],
    )
    if validation_rows:
        write_csv(
            out_dir / "eggpu_validation_bad.csv",
            eggpu_validation_bad,
            list(validation_rows[0].keys()),
        )
        write_csv(
            out_dir / "baseline_validation_bad.csv",
            baseline_validation_bad,
            list(validation_rows[0].keys()),
        )
    write_csv(
        out_dir / "coverage_issues.csv",
        missing_rows,
        ["function", "baseline", "issue", "ok", "skipped", "failed"],
    )
    write_csv(
        out_dir / "metadata_issues.csv",
        metadata_issues,
        ["field", "expected", "actual", "severity"],
    )
    write_csv(
        out_dir / "validation_issues.csv",
        validation_issues,
        ["dataset", "function", "issue"],
    )
    write_csv(
        out_dir / "memory_issues.csv",
        memory_issues,
        [
            "dataset",
            "function",
            "baseline",
            "status",
            "issue",
            "memory_peak_gpu_mb",
            "memory_peak_gpu_proc_mb",
            "non_process_start_mb",
            "expected_non_process_baseline_mb",
            "excess_device_over_proc_mb",
            "severity",
            "note",
        ],
    )
    write_csv(
        out_dir / "sample_contract_issues.csv",
        sample_issues,
        ["dataset", "function", "baseline", "metric", "issue", "severity", "details"],
    )
    write_csv(
        out_dir / "measurement_schema_issues.csv",
        measurement_issues,
        ["dataset", "function", "baseline", "metric", "issue", "severity", "details"],
    )

    summary = {
        "result_dir": str(result_dir),
        "datasets": _dataset_count(e2e_rows),
        "functions_expected": expected_functions,
        "functions_seen": observed_functions,
        "baselines_seen": sorted({r.get("baseline") for r in e2e_rows if r.get("baseline")}),
        "e2e_status_counts": _status_counts(e2e_rows, "status"),
        "validation_status_counts": _status_counts(validation_rows, "validation_status"),
        "eggpu_e2e_rows": len(eggpu_e2e),
        "eggpu_runtime_bad": len(eggpu_runtime_bad),
        "eggpu_validation_rows": len(eggpu_validation),
        "eggpu_validation_bad": len(eggpu_validation_bad),
        "eggpu_validation_missing": len(validation_issues),
        "baseline_validation_bad": len(baseline_validation_bad),
        "coverage_issues": len(missing_rows),
        "eggpu_coverage_issues": len(eggpu_coverage_issues),
        "metadata_issues": len(metadata_issues),
        "metadata_hard_issues": len(hard_metadata_issues),
        "sample_contract_issues": len(sample_issues),
        "sample_contract_hard_issues": len(hard_sample_issues),
        "memory_issues": len(memory_issues),
        "measurement_schema_issues": len(measurement_issues),
        "measurement_schema_hard_issues": len(hard_measurement_issues),
        "metadata_source_snapshot": ((metadata or {}).get("source_snapshot") or {}).get("digest", ""),
        "gate_status": (
            "pass"
            if not eggpu_runtime_bad
            and not eggpu_validation_bad
            and not validation_issues
            and not eggpu_coverage_issues
            and not hard_metadata_issues
            and not hard_sample_issues
            and not hard_measurement_issues
            else "fail"
        ),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    md = [
        "# Full Benchmark Audit",
        "",
        f"Result directory: `{result_dir}`",
        "",
        "## Gate Status",
        "",
        f"**{summary['gate_status'].upper()}**",
        "",
        "A pass means EGGPU has no runtime failure/timeout, no non-pass correctness validation row, and valid reproducibility metadata in this result.",
        "",
        "## Summary",
        "",
        f"- datasets: {summary['datasets']}",
        f"- EGGPU e2e rows: {summary['eggpu_e2e_rows']}",
        f"- EGGPU runtime bad rows: {summary['eggpu_runtime_bad']}",
        f"- EGGPU validation rows: {summary['eggpu_validation_rows']}",
        f"- EGGPU validation bad rows: {summary['eggpu_validation_bad']}",
        f"- EGGPU missing validation rows: {summary['eggpu_validation_missing']}",
        f"- non-EGGPU validation bad rows: {summary['baseline_validation_bad']}",
        f"- coverage issues: {summary['coverage_issues']}",
        f"- EGGPU coverage issues: {summary['eggpu_coverage_issues']}",
        f"- metadata hard issues: {summary['metadata_hard_issues']}",
        f"- repeat/sample contract hard issues: {summary['sample_contract_hard_issues']}",
        f"- memory contamination warnings: {summary['memory_issues']}",
        f"- measurement schema hard issues: {summary['measurement_schema_hard_issues']}",
        f"- executed source snapshot: `{summary['metadata_source_snapshot'] or 'missing'}`",
        "",
        "## Status Counts",
        "",
        "E2E rows:",
    ]
    for k, v in summary["e2e_status_counts"].items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("Validation rows:")
    for k, v in summary["validation_status_counts"].items():
        md.append(f"- {k}: {v}")

    if eggpu_validation_bad:
        md.extend(["", "## EGGPU Correctness Problems", ""])
        for r in eggpu_validation_bad[:100]:
            md.append(
                f"- {r.get('dataset')} / {r.get('function')}: "
                f"{r.get('validation_status')} ({r.get('details')})"
            )
    if validation_issues:
        md.extend(["", "## EGGPU Validation Coverage Problems", ""])
        for r in validation_issues[:100]:
            if r.get("dataset") or r.get("function"):
                md.append(
                    f"- {r.get('dataset')} / {r.get('function')}: {r.get('issue')}"
                )
            else:
                md.append(f"- {r.get('issue')}")
        if len(validation_issues) > 100:
            md.append(f"- ... {len(validation_issues) - 100} more rows omitted")
    if eggpu_runtime_bad:
        md.extend(["", "## EGGPU Runtime Problems", ""])
        for r in eggpu_runtime_bad[:100]:
            md.append(
                f"- {r.get('dataset')} / {r.get('function')}: "
                f"{r.get('status')} ({r.get('notes')})"
            )
    if baseline_validation_bad:
        md.extend(["", "## Non-EGGPU Baseline Correctness Problems", ""])
        for r in baseline_validation_bad[:100]:
            md.append(
                f"- {r.get('dataset')} / {r.get('function')} / {r.get('baseline')}: "
                f"{r.get('validation_status')} ({r.get('details')})"
            )
        if len(baseline_validation_bad) > 100:
            md.append(f"- ... {len(baseline_validation_bad) - 100} more rows omitted")
    if metadata_issues:
        md.extend(["", "## Metadata Problems", ""])
        for r in metadata_issues[:100]:
            md.append(
                f"- {r.get('severity')} `{r.get('field')}`: expected {r.get('expected')}, got {r.get('actual')}"
            )
        if len(metadata_issues) > 100:
            md.append(f"- ... {len(metadata_issues) - 100} more rows omitted")
    if memory_issues:
        md.extend(["", "## Memory Contamination Warnings", ""])
        md.append(
            "`memory_peak_gpu_mb` is a whole-device NVML value and can include unrelated GPU processes. "
            "Use `memory_peak_gpu_proc_mb` / `memory_avg_gpu_proc_mb` for paper comparisons when available."
        )
        md.append("")
        for r in memory_issues[:100]:
            prefix = (
                f"- {r.get('dataset')} / {r.get('function')} / {r.get('baseline')}: "
                f"{r.get('issue')}"
            )
            if r.get("non_process_start_mb"):
                md.append(
                    f"{prefix} (non-process memory at call start="
                    f"{r.get('non_process_start_mb')} MB)"
                )
            else:
                md.append(
                    f"{prefix} (device={r.get('memory_peak_gpu_mb')} MB, "
                    f"proc={r.get('memory_peak_gpu_proc_mb')} MB)"
                )
        if len(memory_issues) > 100:
            md.append(f"- ... {len(memory_issues) - 100} more rows omitted")
    if sample_issues:
        md.extend(["", "## Repeat and Sample Contract", ""])
        for row in sample_issues[:100]:
            md.append(
                f"- {row.get('severity')} {row.get('dataset')} / {row.get('function')} / "
                f"{row.get('baseline')} / {row.get('metric')}: {row.get('issue')} "
                f"({row.get('details')})"
            )
        if len(sample_issues) > 100:
            md.append(f"- ... {len(sample_issues) - 100} more rows omitted")
    if measurement_issues:
        md.extend(["", "## Measurement Schema and Sampling", ""])
        for row in measurement_issues[:100]:
            md.append(
                f"- {row.get('severity')} {row.get('dataset')} / {row.get('function')} / "
                f"{row.get('baseline')} / {row.get('metric')}: {row.get('issue')} "
                f"({row.get('details')})"
            )
        if len(measurement_issues) > 100:
            md.append(f"- ... {len(measurement_issues) - 100} more rows omitted")

    md.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `audit/audit_summary.json`",
            "- `audit/eggpu_runtime_bad.csv`",
            "- `audit/eggpu_validation_bad.csv`",
            "- `audit/baseline_validation_bad.csv`",
            "- `audit/coverage_issues.csv`",
            "- `audit/metadata_issues.csv`",
            "- `audit/validation_issues.csv`",
            "- `audit/memory_issues.csv`",
            "- `audit/sample_contract_issues.csv`",
            "",
        ]
    )
    (out_dir / "AUDIT.md").write_text("\n".join(md))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--expected-repeat", type=int, default=5)
    args = ap.parse_args()
    summary = audit(Path(args.result_dir).resolve(), expected_repeat=args.expected_repeat)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["gate_status"] == "pass" else 2)


if __name__ == "__main__":
    main()
