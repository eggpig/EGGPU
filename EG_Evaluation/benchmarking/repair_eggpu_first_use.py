#!/usr/bin/env python3
"""Repair only incomplete EGGPU First-use pairs without rewriting raw evidence.

The July paper protocol keeps the original First-use result immutable.  This
script derives the expected pair set from the audited steady-state run, reruns
only incomplete timing or primary-memory pairs, and writes a compact merged
evidence directory with an explicit repair manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from benchmark_stats import aggregate_sample_rows
from measurement_schema import write_measurement_schema
from run_full_baselines import write_metric_csvs_no_pandas
from validate_correctness import write_validation_outputs


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MEMORY_METRIC = "memory_peak_gpu_proc_mb"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pair(row: dict[str, str]) -> tuple[str, str]:
    return row.get("dataset", ""), row.get("function", "")


def publishable(row: dict[str, str]) -> bool:
    return (
        row.get("baseline") == "EGGPU"
        and row.get("status") == "ok"
        and row.get("publishable", "true").lower() == "true"
    )


def expected_pairs(main_rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {
        pair(row)
        for row in main_rows
        if row.get("metric") == "e2e" and publishable(row)
    }


def repair_plan(
    main_rows: list[dict[str, str]], first_rows: list[dict[str, str]]
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    expected = expected_pairs(main_rows)
    complete_metrics: dict[tuple[str, str], set[str]] = {}
    memory_complete: set[tuple[str, str]] = set()
    for row in first_rows:
        key = pair(row)
        if key not in expected or not publishable(row):
            continue
        if row.get("metric") in {"e2e", "kernel"}:
            complete_metrics.setdefault(key, set()).add(row["metric"])
        if row.get("metric") == PRIMARY_MEMORY_METRIC:
            memory_complete.add(key)
    timing_missing = {
        key for key in expected if complete_metrics.get(key, set()) != {"e2e", "kernel"}
    }
    memory_missing = expected - memory_complete
    return timing_missing, memory_missing


def artifact_digests(metadata: dict) -> list[str]:
    return sorted(
        item.get("sha256", "")
        for item in ((metadata.get("build_artifacts") or {}).get("cpp_easygraph") or [])
        if item.get("sha256")
    )


def implementation_digest(metadata: dict) -> str:
    return (metadata.get("implementation_source_snapshot") or {}).get("digest", "")


def selected_device(metadata: dict) -> dict:
    return (metadata.get("gpu_device_profile") or {}).get("selected_device") or {}


def hardware_signature(metadata: dict) -> tuple[str, str, str, str]:
    device = selected_device(metadata)
    gpu_profile = metadata.get("gpu_device_profile") or {}
    host = metadata.get("host_profile") or {}
    return (
        str(device.get("name", "")),
        str(device.get("compute_capability", "")),
        str(gpu_profile.get("driver_version", "")),
        str(host.get("cpu_model", "")),
    )


def run_pair(args, key: tuple[str, str], repair_root: Path) -> Path:
    dataset, function = key
    safe_name = f"{dataset}__{function}".replace("/", "_")
    result = repair_root / safe_name
    command = [
        sys.executable,
        str(ROOT / "benchmarking" / "run_split_full_baselines.py"),
        "--gpu",
        str(args.gpu),
        "--out-dir",
        str(result),
        "--repeat",
        str(args.repeat),
        "--memory-repeat",
        str(args.memory_repeat),
        "--warmup",
        "0",
        "--easygraph-warmup",
        "0",
        "--library-timeout",
        str(args.library_timeout),
        "--inter-run-cooldown",
        str(args.inter_run_cooldown),
        "--pr-alpha",
        str(args.pr_alpha),
        "--pr-eps",
        str(args.pr_eps),
        "--pr-max-iter",
        str(args.pr_max_iter),
        "--easygraph-repo",
        str(args.easygraph_repo),
        "--sssp-sources",
        str(args.sssp_sources),
        "--bc-sources",
        str(args.bc_sources),
        "--datasets",
        dataset,
        "--functions",
        function,
        "--baselines",
        "EGGPU",
        "--eggpu-execution-protocol",
        "first-use",
    ]
    env = dict(os.environ)
    env["EGGPU_EXECUTION_PROTOCOL"] = "first-use"
    env["EGGPU_SKIP_PLOTS"] = "TRUE"
    print(f"[first-use-repair] dataset={dataset} function={function}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"repair run failed for {dataset}/{function}: exit={completed.returncode}; "
            f"partial evidence={result}"
        )
    return result


def phase(row: dict[str, str]) -> str:
    value = row.get("measurement_phase", "")
    if value:
        return value
    return "memory" if row.get("metric", "").startswith("memory_") else "timing"


def merge_samples(
    base_rows: list[dict[str, str]],
    replacements: dict[tuple[str, str], list[dict[str, str]]],
    timing_missing: set[tuple[str, str]],
    memory_missing: set[tuple[str, str]],
) -> list[dict[str, str]]:
    merged = []
    for row in base_rows:
        key = pair(row)
        replace = (
            phase(row) == "timing" and key in timing_missing
        ) or (
            phase(row) == "memory" and key in memory_missing
        )
        if row.get("baseline") == "EGGPU" and replace:
            continue
        merged.append(dict(row))
    for key in sorted(replacements):
        for row in replacements[key]:
            include = (
                phase(row) == "timing" and key in timing_missing
            ) or (
                phase(row) == "memory" and key in memory_missing
            )
            if row.get("baseline") == "EGGPU" and include:
                merged.append(dict(row))
    return merged


def validate_replacement(
    rows: list[dict[str, str]],
    key: tuple[str, str],
    *,
    require_timing: bool,
    require_memory: bool,
    repeat: int,
    memory_repeat: int,
) -> None:
    subset = [row for row in rows if row.get("baseline") == "EGGPU" and pair(row) == key]
    if require_timing:
        for metric in ("e2e", "kernel"):
            valid = [
                row for row in subset
                if row.get("metric") == metric and phase(row) == "timing" and row.get("status") == "ok"
            ]
            if len(valid) != repeat:
                raise SystemExit(
                    f"repair remains incomplete for {key[0]}/{key[1]} {metric}: "
                    f"valid_samples={len(valid)}/{repeat}"
                )
    if require_memory:
        valid = [
            row for row in subset
            if row.get("metric") == PRIMARY_MEMORY_METRIC
            and phase(row) == "memory"
            and row.get("status") == "ok"
        ]
        if len(valid) != memory_repeat:
            raise SystemExit(
                f"repair remains incomplete for {key[0]}/{key[1]} memory: "
                f"valid_samples={len(valid)}/{memory_repeat}"
            )


def copy_supporting_files(base_dir: Path, out_dir: Path) -> None:
    for name in (
        "dataset_stats.json",
        "baseline_versions.json",
        "measurement_schema.json",
        "split_measurement_protocol.json",
    ):
        source = base_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
    if not (out_dir / "measurement_schema.json").exists():
        write_measurement_schema(out_dir / "measurement_schema.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--steady-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-repeat", type=int, default=1)
    parser.add_argument("--library-timeout", type=int, default=100)
    parser.add_argument("--inter-run-cooldown", type=float, default=1.0)
    parser.add_argument("--easygraph-repo", required=True, type=Path)
    parser.add_argument("--sssp-sources", type=int, default=8)
    parser.add_argument("--bc-sources", type=int, default=16)
    parser.add_argument("--pr-alpha", type=float, default=0.75)
    parser.add_argument("--pr-eps", type=float, default=1.0e-6)
    parser.add_argument("--pr-max-iter", type=int, default=200)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact repair plan without creating files or using a GPU.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    steady_dir = args.steady_dir.resolve()
    out_dir = args.out_dir.resolve()
    base_samples = read_csv(base_dir / "results_samples.csv")
    base_long = read_csv(base_dir / "results_long.csv")
    main_long = read_csv(steady_dir / "results_long.csv")
    timing_missing, memory_missing = repair_plan(main_long, base_long)
    repair_pairs = sorted(timing_missing | memory_missing)
    print(
        f"[first-use-repair] timing_missing={len(timing_missing)} "
        f"memory_missing={len(memory_missing)} unique_pairs={len(repair_pairs)}",
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "timing_missing_pairs": [list(key) for key in sorted(timing_missing)],
                    "memory_missing_pairs": [list(key) for key in sorted(memory_missing)],
                    "unique_repair_pairs": [list(key) for key in repair_pairs],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty repair directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    repair_root = out_dir / "repair_runs"
    repair_root.mkdir()

    base_metadata = json.loads((base_dir / "run_metadata.json").read_text())
    base_artifacts = artifact_digests(base_metadata)
    base_implementation = implementation_digest(base_metadata)
    base_hardware = hardware_signature(base_metadata)
    if not base_artifacts or not base_implementation:
        raise SystemExit("base First-use metadata lacks binary/source provenance")

    replacements: dict[tuple[str, str], list[dict[str, str]]] = {}
    repair_sources = []
    for key in repair_pairs:
        result = run_pair(args, key, repair_root)
        rows = read_csv(result / "results_samples.csv")
        validate_replacement(
            rows,
            key,
            require_timing=key in timing_missing,
            require_memory=key in memory_missing,
            repeat=args.repeat,
            memory_repeat=args.memory_repeat,
        )
        metadata = json.loads((result / "run_metadata.json").read_text())
        if artifact_digests(metadata) != base_artifacts:
            raise SystemExit(f"compiled artifact mismatch in repair result: {result}")
        if implementation_digest(metadata) != base_implementation:
            raise SystemExit(f"implementation source mismatch in repair result: {result}")
        if hardware_signature(metadata) != base_hardware:
            raise SystemExit(
                "repair hardware class differs from the base First-use run: "
                f"base={base_hardware} repair={hardware_signature(metadata)} result={result}"
            )
        replacements[key] = rows
        repair_sources.append(
            {
                "dataset": key[0],
                "function": key[1],
                "result_dir": str(result),
                "timing_replaced": key in timing_missing,
                "memory_replaced": key in memory_missing,
                "source_snapshot": metadata.get("source_snapshot"),
                "implementation_source_snapshot": metadata.get(
                    "implementation_source_snapshot"
                ),
                "gpu_device_profile": metadata.get("gpu_device_profile"),
                "host_profile": metadata.get("host_profile"),
            }
        )

    merged_samples = merge_samples(
        base_samples, replacements, timing_missing, memory_missing
    )
    timing_samples = [row for row in merged_samples if phase(row) == "timing"]
    memory_samples = [row for row in merged_samples if phase(row) == "memory"]
    merged_long = aggregate_sample_rows(timing_samples, args.repeat)
    merged_long.extend(aggregate_sample_rows(memory_samples, args.memory_repeat))

    write_csv(out_dir / "results_samples.csv", merged_samples)
    write_csv(out_dir / "results_long.csv", merged_long)
    copy_supporting_files(base_dir, out_dir)
    write_validation_outputs(out_dir, merged_long)
    write_metric_csvs_no_pandas(out_dir, merged_long)

    metadata = dict(base_metadata)
    metadata["result_dir"] = str(out_dir)
    metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
    metadata["repair_protocol"] = {
        "protocol": "targeted_first_use_pair_replacement_v1",
        "base_result": str(base_dir),
        "steady_state_expected_pair_source": str(steady_dir),
        "timing_missing_pairs": [list(key) for key in sorted(timing_missing)],
        "memory_missing_pairs": [list(key) for key in sorted(memory_missing)],
        "replacement_rule": (
            "replace the entire measurement phase for an incomplete pair; "
            "never mix partial old and new samples within one aggregate"
        ),
        "repeat": args.repeat,
        "memory_repeat": args.memory_repeat,
        "hardware_signature": list(base_hardware),
        "repair_sources": repair_sources,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    manifest = metadata["repair_protocol"] | {
        "expected_pairs": len(expected_pairs(main_long)),
        "merged_sample_rows": len(merged_samples),
        "merged_aggregate_rows": len(merged_long),
    }
    (out_dir / "first_use_repair_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "notes.txt").write_text(
        "This directory is a compact, provenance-preserving merge of the original "
        "First-use evidence and targeted replacement runs. The original directory "
        "is unchanged. See first_use_repair_manifest.json.\n"
    )

    derive = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarking" / "derive_performance_efficiency_metrics.py"),
            str(out_dir),
        ],
        cwd=ROOT,
        check=False,
    )
    if derive.returncode != 0:
        raise SystemExit(f"derived metric generation failed: exit={derive.returncode}")

    postprocess = [
        sys.executable,
        str(ROOT / "benchmarking" / "run_eggpu_first_use.py"),
        "--gpu",
        str(args.gpu),
        "--steady-dir",
        str(steady_dir),
        "--out-dir",
        str(out_dir),
        "--repeat",
        str(args.repeat),
        "--memory-repeat",
        str(args.memory_repeat),
        "--library-timeout",
        str(args.library_timeout),
        "--inter-run-cooldown",
        str(args.inter_run_cooldown),
        "--easygraph-repo",
        str(args.easygraph_repo),
        "--sssp-sources",
        str(args.sssp_sources),
        "--bc-sources",
        str(args.bc_sources),
        "--pr-alpha",
        str(args.pr_alpha),
        "--pr-eps",
        str(args.pr_eps),
        "--pr-max-iter",
        str(args.pr_max_iter),
        "--reuse-existing",
    ]
    completed = subprocess.run(postprocess, cwd=ROOT, env=dict(os.environ), check=False)
    if completed.returncode != 0:
        raise SystemExit(f"First-use comparison postprocessing failed: exit={completed.returncode}")
    print(f"Repaired First-use evidence: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
