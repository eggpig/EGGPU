#!/usr/bin/env python3
"""Run uncontended timing and memory passes, then merge their evidence.

Timing samples never start psutil/NVML sampling threads.  Memory samples run in
separate subprocesses and cannot contribute build/e2e/kernel values to the
paper-facing tables.  Raw rows and per-pass artifacts remain available under
``measurement_passes`` for provenance.
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
from gpu_visibility_marker import GpuVisibilityMarker
from measurement_schema import write_measurement_schema
from run_full_baselines import (
    DEFAULT_DATASETS,
    write_metric_csvs_no_pandas,
    write_plot_and_tables_isolated,
)
from validate_correctness import write_validation_outputs


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pass_command(args, *, mode, repeat, out_dir):
    command = [
        sys.executable,
        str(ROOT / "benchmarking" / "run_full_baselines.py"),
        "--gpu",
        str(args.gpu),
        "--out-dir",
        str(out_dir),
        "--repeat",
        str(repeat),
        "--warmup",
        str(args.warmup),
        "--easygraph-warmup",
        str(args.easygraph_warmup),
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
        "--closeness-sources",
        str(args.closeness_sources),
        "--datasets",
        str(args.datasets),
        "--functions",
        str(args.functions),
        "--baselines",
        str(args.baselines),
        "--eggpu-execution-protocol",
        str(args.eggpu_execution_protocol),
        "--measurement-mode",
        mode,
    ]
    if args.skip_large_cpu:
        command.append("--skip-large-cpu")
    return command


def run_pass(args, *, mode, repeat, out_dir):
    env = dict(os.environ)
    env["EGGPU_MEASUREMENT_MODE"] = mode
    env["EGGPU_INTERNAL_MEMORY_MONITOR"] = "FALSE" if mode == "memory" else "TRUE"
    env["EGGPU_SKIP_PLOTS"] = "TRUE"
    print(
        f"[split-measurement] starting {mode} pass: repeat={repeat}, out={out_dir}",
        flush=True,
    )
    completed = subprocess.run(
        pass_command(args, mode=mode, repeat=repeat, out_dir=out_dir),
        cwd=ROOT,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{mode} pass failed with exit code {completed.returncode}; partial output: {out_dir}"
        )
    print(f"[split-measurement] completed {mode} pass", flush=True)


def copy_timing_artifacts(timing_dir, out_dir):
    excluded = {
        "results_samples.csv",
        "results_long.csv",
        "results_build.csv",
        "results_e2e.csv",
        "results_kernel.csv",
        "results_memory.csv",
        "run_metadata.json",
        "measurement_schema.json",
        "logs",
    }
    for source in timing_dir.iterdir():
        if source.name in excluded:
            continue
        target = out_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def selected_datasets(token_string):
    tokens = {token.strip().lower() for token in str(token_string).split(",") if token.strip()}
    if not tokens or "all" in tokens:
        return list(DEFAULT_DATASETS)
    return [
        item
        for item in DEFAULT_DATASETS
        if {item[0].lower(), item[1].lower(), item[2].lower()} & tokens
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--easygraph-warmup", type=int, default=2)
    parser.add_argument("--skip-large-cpu", action="store_true")
    parser.add_argument("--library-timeout", type=int, default=100)
    parser.add_argument("--inter-run-cooldown", type=float, default=1.0)
    parser.add_argument("--pr-alpha", type=float, default=0.75)
    parser.add_argument("--pr-eps", type=float, default=1.0e-6)
    parser.add_argument("--pr-max-iter", type=int, default=200)
    parser.add_argument("--easygraph-repo", required=True)
    parser.add_argument("--sssp-sources", type=int, default=8)
    parser.add_argument("--bc-sources", type=int, default=16)
    parser.add_argument("--closeness-sources", type=int, default=0)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--functions", default="all")
    parser.add_argument("--baselines", default="all")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge already completed timing and memory passes without rerunning them.",
    )
    parser.add_argument(
        "--allow-unbuilt-implementation-source-drift",
        action="store_true",
        help=(
            "Allow an implementation-source digest change only when merging "
            "existing passes whose loaded cpp_easygraph binary and benchmark "
            "contract are identical. The exception is recorded in provenance."
        ),
    )
    parser.add_argument(
        "--eggpu-execution-protocol",
        choices=["steady-state", "first-use"],
        default="steady-state",
    )
    args = parser.parse_args()

    if args.repeat < 1 or args.memory_repeat < 1:
        raise SystemExit("--repeat and --memory-repeat must both be positive")

    out_dir = Path(args.out_dir).resolve()
    pass_root = out_dir / "measurement_passes"
    timing_dir = pass_root / "timing"
    memory_dir = pass_root / "memory"
    timing_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    if not args.merge_existing:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.setdefault("EGGPU_MONITOR_GPU_INDEX", str(args.gpu))
        visibility_marker = GpuVisibilityMarker(
            args.gpu, "split timing/memory benchmark"
        ).start()
        if visibility_marker.started:
            os.environ["EGGPU_EXTERNAL_VISIBILITY_MARKER"] = "TRUE"

        run_pass(args, mode="timing", repeat=args.repeat, out_dir=timing_dir)
        run_pass(args, mode="memory", repeat=args.memory_repeat, out_dir=memory_dir)
    else:
        required = (
            timing_dir / "results_samples.csv",
            timing_dir / "run_metadata.json",
            memory_dir / "results_samples.csv",
            memory_dir / "run_metadata.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SystemExit(
                "cannot merge existing split passes; missing: " + ", ".join(missing)
            )

    timing_samples = read_csv(timing_dir / "results_samples.csv")
    memory_samples = read_csv(memory_dir / "results_samples.csv")
    if any(str(row.get("metric", "")).startswith("memory_") for row in timing_samples):
        raise SystemExit("timing pass unexpectedly contains memory metrics")
    if any(not str(row.get("metric", "")).startswith("memory_") for row in memory_samples):
        raise SystemExit("memory pass unexpectedly contains timing metrics")
    for row in timing_samples:
        row["measurement_phase"] = "timing"
    for row in memory_samples:
        row["measurement_phase"] = "memory"

    timing_long = aggregate_sample_rows(timing_samples, expected_samples=args.repeat)
    memory_long = aggregate_sample_rows(memory_samples, expected_samples=args.memory_repeat)
    merged_samples = timing_samples + memory_samples
    merged_long = timing_long + memory_long

    copy_timing_artifacts(timing_dir, out_dir)
    final_logs = out_dir / "logs"
    expected_logs_target = Path("measurement_passes") / "timing" / "logs"
    if final_logs.is_symlink():
        if Path(os.readlink(final_logs)) != expected_logs_target:
            raise SystemExit(f"unexpected final log symlink target: {final_logs}")
    elif final_logs.exists():
        raise SystemExit(f"refusing to replace existing final log path: {final_logs}")
    else:
        final_logs.symlink_to(expected_logs_target, target_is_directory=True)
    shutil.copy2(timing_dir / "dataset_stats.json", out_dir / "dataset_stats.json")
    write_csv(out_dir / "results_samples.csv", merged_samples)
    write_csv(out_dir / "results_long.csv", merged_long)
    write_measurement_schema(out_dir / "measurement_schema.json")
    write_validation_outputs(out_dir, merged_long)
    write_metric_csvs_no_pandas(out_dir, merged_long)

    timing_metadata = json.loads((timing_dir / "run_metadata.json").read_text())
    memory_metadata = json.loads((memory_dir / "run_metadata.json").read_text())
    timing_snapshot = (timing_metadata.get("source_snapshot") or {}).get("digest", "")
    memory_snapshot = (memory_metadata.get("source_snapshot") or {}).get("digest", "")
    timing_implementation = (
        timing_metadata.get("implementation_source_snapshot") or {}
    ).get("digest", "")
    memory_implementation = (
        memory_metadata.get("implementation_source_snapshot") or {}
    ).get("digest", "")
    implementation_source_match = bool(
        timing_implementation
        and timing_implementation == memory_implementation
    )
    source_snapshot_match = bool(
        timing_snapshot and timing_snapshot == memory_snapshot
    )

    ignored_contract_keys = {"measurement_mode", "repeat"}
    timing_contract = {
        key: value
        for key, value in (timing_metadata.get("benchmark_args") or {}).items()
        if key not in ignored_contract_keys
    }
    memory_contract = {
        key: value
        for key, value in (memory_metadata.get("benchmark_args") or {}).items()
        if key not in ignored_contract_keys
    }
    if timing_contract != memory_contract:
        raise SystemExit(
            "timing/memory benchmark contract mismatch after excluding "
            "measurement mode and repeat count"
        )

    def extension_digests(metadata):
        active = (
            (metadata.get("build_artifacts") or {}).get("active_cpp_easygraph")
            or {}
        ).get("sha256", "")
        if active:
            return [active]
        return sorted(
            item.get("sha256", "")
            for item in ((metadata.get("build_artifacts") or {}).get("cpp_easygraph") or [])
            if item.get("sha256")
        )

    timing_extensions = extension_digests(timing_metadata)
    memory_extensions = extension_digests(memory_metadata)
    if not timing_extensions or timing_extensions != memory_extensions:
        raise SystemExit(
            "timing/memory cpp_easygraph artifact mismatch: "
            f"timing={timing_extensions} memory={memory_extensions}"
        )
    if (
        not implementation_source_match
        and not args.allow_unbuilt_implementation_source_drift
    ):
        raise SystemExit(
            "timing/memory EGGPU implementation snapshot mismatch: "
            f"timing={timing_implementation or 'missing'} "
            f"memory={memory_implementation or 'missing'}; rerun both passes or "
            "use the explicit merge exception only after verifying that the "
            "identical extension binary was loaded"
        )

    metadata = timing_metadata
    metadata["result_dir"] = str(out_dir)
    metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
    metadata.setdefault("benchmark_args", {})["repeat"] = args.repeat
    metadata["benchmark_args"]["memory_repeat"] = args.memory_repeat
    metadata["benchmark_args"]["measurement_mode"] = "split_timing_memory"
    metadata["benchmark_args"]["eggpu_execution_protocol"] = (
        args.eggpu_execution_protocol
    )
    metadata["benchmark_args"]["baselines_filter"] = args.baselines
    metadata["measurement_protocol"] = {
        "protocol": "split_timing_memory_v1",
        "timing_pass": {
            "repeat": args.repeat,
            "memory_sampling_enabled": False,
            "paper_time_source": True,
            "artifact_dir": str(timing_dir),
        },
        "memory_pass": {
            "repeat": args.memory_repeat,
            "memory_sampling_enabled": True,
            "paper_time_source": False,
            "artifact_dir": str(memory_dir),
        },
        "merge_key": ["dataset", "function", "baseline", "metric", "semantic"],
        "primary_gpu_memory_metric": "memory_peak_gpu_proc_mb",
        "primary_cpu_memory_metric": "memory_peak_rss_mb",
        "source_snapshot_match": source_snapshot_match,
        "implementation_source_snapshot_match": implementation_source_match,
        "unbuilt_implementation_source_drift_accepted": bool(
            not implementation_source_match
            and args.allow_unbuilt_implementation_source_drift
        ),
        "benchmark_contract_match": True,
        "cpp_easygraph_artifact_match": True,
        "eggpu_execution_protocol": args.eggpu_execution_protocol,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    protocol = metadata["measurement_protocol"]
    (out_dir / "split_measurement_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    notes_path = out_dir / "notes.txt"
    prior_notes = notes_path.read_text() if notes_path.exists() else ""
    notes_path.write_text(
        prior_notes
        + ("\n" if prior_notes and not prior_notes.endswith("\n\n") else "")
        + "Timing and memory were collected in independent subprocess passes. "
        + "Only timing-pass build/e2e/kernel rows are eligible for performance tables. "
        + (
            "The broad repository source snapshot changed between passes, but the "
            "EGGPU implementation snapshot, loaded extension, and benchmark contract "
            "match. "
            if not source_snapshot_match and implementation_source_match
            else ""
        )
        + (
            "Implementation source files changed between passes without rebuilding "
            "the identical loaded cpp_easygraph binary; this explicit merge exception "
            "is recorded and all affected functions must be refreshed after rebuild. "
            if not implementation_source_match
            and args.allow_unbuilt_implementation_source_drift
            else ""
        )
        + f"EGGPU execution protocol: {args.eggpu_execution_protocol}.\n"
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
        raise SystemExit(f"derived metric generation failed with exit code {derive.returncode}")

    plot_error = write_plot_and_tables_isolated(
        out_dir,
        merged_long,
        selected_datasets(args.datasets),
        str(args.gpu),
    )
    if plot_error:
        (out_dir / "plot_error.txt").write_text(plot_error + "\n")
        raise SystemExit(plot_error)

    print(f"Done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
