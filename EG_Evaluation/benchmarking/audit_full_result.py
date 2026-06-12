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
HARD_RUNTIME_BAD = {"failed", "timeout"}


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


def audit_memory_rows(long_rows):
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
    for (dataset, function, baseline, status), metrics in sorted(by_key.items()):
        peak_row = metrics.get("memory_peak_gpu_mb")
        proc_row = metrics.get("memory_peak_gpu_proc_mb")
        if peak_row is None or proc_row is None:
            continue
        peak = _safe_float(peak_row.get("seconds"))
        proc = _safe_float(proc_row.get("seconds"))
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
                    "excess_device_over_proc_mb": f"{excess:.6g}",
                    "severity": "warning",
                    "note": "device-level GPU memory is not process-isolated; inspect process-tree memory and GPU-idle evidence before using whole-device memory",
                }
            )
    return issues


def audit_metadata(result_dir: Path, datasets: list[str], functions: list[str]):
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

    git_meta = metadata.get("git") or {}
    if not git_meta.get("commit"):
        issue("git.commit", "non-empty", git_meta.get("commit"))

    py_meta = metadata.get("python") or {}
    if not py_meta.get("executable"):
        issue("python.executable", "non-empty", py_meta.get("executable"))
    if not py_meta.get("direct_child_python"):
        issue("python.direct_child_python", "non-empty", py_meta.get("direct_child_python"))

    cuda_meta = metadata.get("cuda") or {}
    if not cuda_meta.get("local_cuda_root"):
        issue("cuda.local_cuda_root", "non-empty", cuda_meta.get("local_cuda_root"))
    if not cuda_meta.get("nvcc"):
        issue("cuda.nvcc", "non-empty", cuda_meta.get("nvcc"))

    artifacts = (metadata.get("build_artifacts") or {}).get("cpp_easygraph") or []
    if not artifacts:
        issue("build_artifacts.cpp_easygraph", "at least one shared library", "empty")
    run_artifacts = metadata.get("artifacts") or {}
    if not run_artifacts:
        issue("artifacts", "present after completed run", "missing")
    if run_artifacts.get("validation_error"):
        issue("artifacts.validation_error", "empty", run_artifacts.get("validation_error"))

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

    env_meta = metadata.get("environment") or {}
    if env_meta.get("EASYGRAPH_ENABLE_GPU") != "TRUE":
        issue("environment.EASYGRAPH_ENABLE_GPU", "TRUE", env_meta.get("EASYGRAPH_ENABLE_GPU"))
    if env_meta.get("EASYGRAPH_GPU_BACKEND") != "mine":
        issue("environment.EASYGRAPH_GPU_BACKEND", "mine", env_meta.get("EASYGRAPH_GPU_BACKEND"))
    if env_meta.get("EGGPU_USE_CONDA_RUN") != "FALSE":
        issue("environment.EGGPU_USE_CONDA_RUN", "FALSE", env_meta.get("EGGPU_USE_CONDA_RUN"))

    if git_meta.get("dirty"):
        issue("git.dirty", "false for final camera-ready artifact", True, severity="warning")

    return metadata, issues


def audit(result_dir: Path):
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
    )
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
    memory_issues = audit_memory_rows(long_rows)

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
            "excess_device_over_proc_mb",
            "severity",
            "note",
        ],
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
        "memory_issues": len(memory_issues),
        "metadata_git_commit": ((metadata or {}).get("git") or {}).get("commit", ""),
        "metadata_git_dirty": ((metadata or {}).get("git") or {}).get("dirty", ""),
        "gate_status": (
            "pass"
            if not eggpu_runtime_bad
            and not eggpu_validation_bad
            and not validation_issues
            and not eggpu_coverage_issues
            and not hard_metadata_issues
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
        f"- memory contamination warnings: {summary['memory_issues']}",
        f"- metadata git commit: `{summary['metadata_git_commit'] or 'unknown'}`",
        f"- metadata git dirty: `{summary['metadata_git_dirty']}`",
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
            md.append(
                f"- {r.get('dataset')} / {r.get('function')} / {r.get('baseline')}: "
                f"{r.get('issue')} (device={r.get('memory_peak_gpu_mb')} MB, "
                f"proc={r.get('memory_peak_gpu_proc_mb')} MB)"
            )
        if len(memory_issues) > 100:
            md.append(f"- ... {len(memory_issues) - 100} more rows omitted")

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
            "",
        ]
    )
    (out_dir / "AUDIT.md").write_text("\n".join(md))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    args = ap.parse_args()
    summary = audit(Path(args.result_dir).resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["gate_status"] == "pass" else 2)


if __name__ == "__main__":
    main()
