#!/usr/bin/env python3
"""Audit the final EGGPU paper supplements as one evidence bundle."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


WORKFLOW = ("WCC", "PageRank", "BFS")
PRIMARY_MEMORY_METRIC = "memory_peak_gpu_proc_mb"
CLOSENESS_RUN_BASELINES = {
    "igraph",
    "networkx",
    "EGGPU",
    "easygraph-cpu",
    "easygraph-cpp",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def record(checks, name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})


def artifact_digests(metadata: dict) -> list[str]:
    return sorted(
        item.get("sha256", "")
        for item in ((metadata.get("build_artifacts") or {}).get("cpp_easygraph") or [])
        if item.get("sha256")
    )


def hardware_signature(metadata: dict) -> tuple[str, str, str, str]:
    profile = metadata.get("gpu_device_profile") or {}
    device = profile.get("selected_device") or {}
    host = metadata.get("host_profile") or {}
    return (
        str(device.get("name", "")),
        str(device.get("compute_capability", "")),
        str(profile.get("driver_version", "")),
        str(host.get("cpu_model", "")),
    )


def publishable_eggpu_pairs(result: Path) -> set[tuple[str, str]]:
    return {
        (row.get("dataset", ""), row.get("function", ""))
        for row in read_csv(result / "results_long.csv")
        if row.get("baseline") == "EGGPU"
        and row.get("metric") == "e2e"
        and row.get("status") == "ok"
        and row.get("publishable", "true").lower() == "true"
    }


def audit_main(checks, result: Path) -> None:
    audit = read_json(result / "audit" / "audit_summary.json")
    record(checks, "main_audit", audit.get("gate_status") == "pass", f"gate_status={audit.get('gate_status')}")
    rows = read_csv(result / "results_long.csv")
    pairs = publishable_eggpu_pairs(result)
    record(checks, "main_eggpu_pair_coverage", len(pairs) == 266, f"ok_pairs={len(pairs)}/266")
    metadata = read_json(result / "run_metadata.json")
    args = metadata.get("benchmark_args") or {}
    protocol = metadata.get("measurement_protocol") or {}
    record(
        checks,
        "main_repeat_protocol",
        int(args.get("repeat", 0)) == 5
        and int(args.get("memory_repeat", 0)) == 3
        and protocol.get("protocol") == "split_timing_memory_v1",
        (
            f"repeat={args.get('repeat')} memory_repeat={args.get('memory_repeat')} "
            f"protocol={protocol.get('protocol')}"
        ),
    )


def audit_ablation(checks, result: Path) -> None:
    rows = read_csv(result / "ablation_all.csv")
    counts = Counter(row.get("status", "") for row in rows)
    record(checks, "ablation_no_failed_rows", counts.get("failed", 0) == 0, f"status_counts={dict(counts)}")
    patterns = {
        "workflow_full": "workflow_*_canonical_full.csv",
        "workflow_no_cpp_cache": "workflow_*_canonical_no_cpp_graph_cache.csv",
        "workflow_no_context": "workflow_*_canonical_no_graph_context.csv",
        "return_materialization": "return_*.csv",
        "graph_layout": "layout_*.csv",
    }
    missing = [name for name, pattern in patterns.items() if not any(result.glob(pattern))]
    record(checks, "ablation_core_modules", not missing, f"missing={missing}")


def audit_first_use(checks, result: Path, main_result: Path) -> None:
    expected = publishable_eggpu_pairs(main_result)
    rows = read_csv(result / "results_long.csv")
    aligned = defaultdict(set)
    incomplete = set()
    for row in rows:
        if row.get("baseline") != "EGGPU" or row.get("metric") not in {"e2e", "kernel"}:
            continue
        pair = (row.get("dataset", ""), row.get("function", ""))
        if row.get("status") == "ok" and str(row.get("publishable", "")).lower() == "true":
            aligned[pair].add(row["metric"])
        elif row.get("status") not in {"skipped", "unsupported"}:
            incomplete.add(pair)
    complete = {pair for pair, metrics in aligned.items() if metrics == {"e2e", "kernel"}}
    record(
        checks,
        "first_use_pair_coverage",
        complete == expected,
        (
            f"complete={len(complete & expected)}/{len(expected)} "
            f"missing={sorted(expected - complete)} incomplete={sorted(incomplete)}"
        ),
    )
    memory = {
        (row.get("dataset", ""), row.get("function", ""))
        for row in rows
        if row.get("baseline") == "EGGPU"
        and row.get("metric") == PRIMARY_MEMORY_METRIC
        and row.get("status") == "ok"
        and row.get("publishable", "true").lower() == "true"
    }
    record(
        checks,
        "first_use_primary_memory_coverage",
        memory >= expected,
        f"complete={len(memory & expected)}/{len(expected)} missing={sorted(expected - memory)}",
    )
    compatibility = read_json(result / "implementation_compatibility.json")
    record(
        checks,
        "first_use_implementation_match",
        compatibility.get("status") == "pass",
        f"status={compatibility.get('status')}",
    )
    first_metadata = read_json(result / "run_metadata.json")
    main_metadata = read_json(main_result / "run_metadata.json")
    record(
        checks,
        "first_use_binary_match",
        bool(artifact_digests(first_metadata))
        and artifact_digests(first_metadata) == artifact_digests(main_metadata),
        (
            f"first={artifact_digests(first_metadata)} "
            f"main={artifact_digests(main_metadata)}"
        ),
    )
    record(
        checks,
        "first_use_platform_match",
        hardware_signature(first_metadata) == hardware_signature(main_metadata),
        (
            f"first={hardware_signature(first_metadata)} "
            f"main={hardware_signature(main_metadata)}"
        ),
    )


def audit_workflow(checks, result: Path, first_use_result: Path) -> None:
    metadata = read_json(result / "natural_workflow_metadata.json")
    protocol = metadata.get("protocol")
    record(
        checks,
        "workflow_matched_control_protocol",
        protocol == "natural_same_graph_workflow_matched_controls_v2",
        f"protocol={protocol}",
    )
    rows = read_csv(result / "natural_workflow_samples.csv")
    datasets = metadata.get("datasets", [])
    repeat = int(metadata.get("repeat", 0))
    expected = len(datasets) * repeat * 18
    record(checks, "workflow_sample_coverage", len(rows) == expected, f"rows={len(rows)}/{expected}")
    bad = [row for row in rows if row.get("status") != "ok"]
    record(checks, "workflow_all_rows_ok", not bad, f"bad_rows={len(bad)}")

    by_key = defaultdict(dict)
    state_bad = []
    for row in rows:
        if row.get("metric") != "e2e" or row.get("function") not in WORKFLOW:
            continue
        key = (row.get("dataset"), row.get("sample_index"), row.get("function"))
        by_key[key][row.get("baseline")] = row
        if row.get("baseline") == "EGGPU-natural-workflow":
            position = int(row.get("call_position", 0))
            before = str(row.get("state_before_graph_context_present", "")).lower() == "true"
            after = str(row.get("state_after_graph_context_present", "")).lower() == "true"
            if position == 1 and (before or not after):
                state_bad.append(key)
            if position in {2, 3} and (not before or not after):
                state_bad.append(key)
    record(checks, "workflow_state_transition", not state_bad, f"bad={state_bad[:8]}")

    mismatches = []
    for key, baselines in by_key.items():
        natural = baselines.get("EGGPU-natural-workflow")
        isolated = baselines.get("EGGPU-isolated-first-use")
        if natural is None or isolated is None:
            mismatches.append((*key, "missing_control"))
        elif natural.get("correctness") != isolated.get("correctness"):
            mismatches.append((*key, "correctness_mismatch"))
    expected_pairs = len(datasets) * repeat * len(WORKFLOW)
    record(
        checks,
        "workflow_control_correctness",
        len(by_key) == expected_pairs and not mismatches,
        f"pairs={len(by_key)}/{expected_pairs} mismatches={mismatches[:8]}",
    )
    first_metadata = read_json(first_use_result / "run_metadata.json")
    record(
        checks,
        "workflow_binary_match",
        bool(artifact_digests(metadata))
        and artifact_digests(metadata) == artifact_digests(first_metadata),
        (
            f"workflow={artifact_digests(metadata)} "
            f"first_use={artifact_digests(first_metadata)}"
        ),
    )
    record(
        checks,
        "workflow_platform_match",
        hardware_signature(metadata) == hardware_signature(first_metadata),
        (
            f"workflow={hardware_signature(metadata)} "
            f"first_use={hardware_signature(first_metadata)}"
        ),
    )
    device = (metadata.get("gpu_device_profile") or {}).get("selected_device") or {}
    external_processes = int(device.get("compute_process_count_at_run_start", 0) or 0)
    record(
        checks,
        "workflow_uncontended_start",
        external_processes == 0,
        f"compute_process_count_at_run_start={external_processes}",
    )


def audit_closeness(checks, result: Path, main_result: Path) -> None:
    metadata = read_json(result / "closeness_large_sampled_metadata.json")
    datasets = metadata.get("datasets", [])
    repeat = int(metadata.get("repeat", 0))
    expected_datasets = {
        row.get("dataset", "")
        for row in read_csv(main_result / "results_long.csv")
        if row.get("baseline") == "EGGPU"
        and row.get("function") == "Closeness"
        and row.get("metric") == "e2e"
        and row.get("status") == "skipped"
        and (
            row.get("skip_reason") == "exact_scale_guard"
            or "exact all-source Closeness skipped" in row.get("notes", "")
        )
    }
    record(
        checks,
        "closeness_dataset_coverage",
        set(datasets) == expected_datasets and repeat == 5,
        f"datasets={datasets} expected={sorted(expected_datasets)} repeat={repeat}",
    )
    summary = read_csv(result / "closeness_large_sampled_summary.csv")
    eggpu = [
        row for row in summary
        if row.get("baseline") == "EGGPU" and row.get("metric") in {"build", "e2e", "kernel"}
    ]
    bad = [
        row for row in eggpu
        if row.get("status") != "ok" or int(float(row.get("sample_count", 0) or 0)) != repeat
    ]
    record(
        checks,
        "closeness_eggpu_repeat_coverage",
        len(eggpu) == len(datasets) * 3 and not bad,
        f"rows={len(eggpu)}/{len(datasets) * 3} bad={[(r.get('dataset'), r.get('metric'), r.get('status')) for r in bad]}",
    )
    validation = read_csv(result / "closeness_large_sampled_validation.csv")
    eggpu_validation = [row for row in validation if row.get("baseline") == "EGGPU"]
    validation_bad = [
        row for row in eggpu_validation
        if row.get("validation_status") != "pass" or row.get("reference") == "EGGPU"
    ]
    record(
        checks,
        "closeness_eggpu_correctness",
        len(eggpu_validation) == len(datasets) * repeat and not validation_bad,
        (
            f"validated={len(eggpu_validation)}/{len(datasets) * repeat} "
            f"bad_or_self_referenced={len(validation_bad)}"
        ),
    )
    raw_rows = read_csv(result / "closeness_large_sampled_long.csv")
    successful = {
        (row.get("dataset", ""), row.get("baseline", ""), row.get("sample_index", ""))
        for row in raw_rows
        if row.get("baseline") in CLOSENESS_RUN_BASELINES
        and row.get("metric") == "e2e"
        and row.get("status") == "ok"
    }
    validated_pass = {
        (row.get("dataset", ""), row.get("baseline", ""), row.get("sample_index", ""))
        for row in validation
        if row.get("baseline") in CLOSENESS_RUN_BASELINES
        and row.get("validation_status") == "pass"
    }
    summary_bad = [
        row for row in summary
        if row.get("baseline") in CLOSENESS_RUN_BASELINES
        and row.get("metric") in {"e2e", "kernel"}
        and row.get("status") == "ok"
        and row.get("validation_status") != "pass"
    ]
    record(
        checks,
        "closeness_all_ranked_baselines_correct",
        successful == validated_pass and not summary_bad,
        (
            f"successful_samples={len(successful)} validated_pass={len(validated_pass)} "
            f"missing_validation={sorted(successful - validated_pass)[:8]} "
            f"orphan_validation={sorted(validated_pass - successful)[:8]} "
            f"ranked_summary_bad={len(summary_bad)}"
        ),
    )
    main_metadata = read_json(main_result / "run_metadata.json")
    record(
        checks,
        "closeness_binary_match",
        bool(artifact_digests(metadata))
        and artifact_digests(metadata) == artifact_digests(main_metadata),
        (
            f"closeness={artifact_digests(metadata)} "
            f"main={artifact_digests(main_metadata)}"
        ),
    )
    record(
        checks,
        "closeness_platform_match",
        hardware_signature(metadata) == hardware_signature(main_metadata),
        (
            f"closeness={hardware_signature(metadata)} "
            f"main={hardware_signature(main_metadata)}"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--ablation", required=True, type=Path)
    parser.add_argument("--first-use", required=True, type=Path)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--closeness", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    checks = []
    main_result = args.main.resolve()
    first_use_result = args.first_use.resolve()
    audits = (
        ("main_bundle_readable", audit_main, (main_result,)),
        ("ablation_bundle_readable", audit_ablation, (args.ablation.resolve(),)),
        (
            "first_use_bundle_readable",
            audit_first_use,
            (first_use_result, main_result),
        ),
        (
            "workflow_bundle_readable",
            audit_workflow,
            (args.workflow.resolve(), first_use_result),
        ),
        (
            "closeness_bundle_readable",
            audit_closeness,
            (args.closeness.resolve(), main_result),
        ),
    )
    for check_name, function, paths in audits:
        try:
            function(checks, *paths)
        except Exception as exc:
            record(
                checks,
                check_name,
                False,
                f"{type(exc).__name__}: {exc}; paths={[str(path) for path in paths]}",
            )
    failed = [item for item in checks if item["status"] != "pass"]
    payload = {
        "schema_version": 2,
        "gate_status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": [item["check"] for item in failed],
    }
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "supplement_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# EGGPU Supplement Experiment Audit",
        "",
        f"Overall status: **{payload['gate_status'].upper()}**.",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for item in checks:
        detail = item["detail"].replace("|", "\\|")
        lines.append(f"| {item['check']} | {item['status']} | {detail} |")
    (out_dir / "SUPPLEMENT_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
