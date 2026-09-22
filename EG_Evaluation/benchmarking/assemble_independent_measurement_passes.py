#!/usr/bin/env python3
"""Assemble independently executed timing and memory benchmark passes.

This helper is intended for targeted baseline recovery runs that were launched
as separate processes rather than through ``run_split_full_baselines.py``.  It
keeps timing and memory samples disjoint, verifies that both passes cover the
same baseline/dataset/function cells under the same benchmark contract, and
records every accepted provenance exception explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from measurement_schema import write_measurement_schema
from run_full_baselines import (
    aggregate_sample_rows,
    write_metric_csvs_no_pandas,
)
from validate_correctness import write_validation_outputs


CELL_FIELDS = ("dataset", "function", "baseline")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cells(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {
        tuple(str(row.get(field, "")).strip() for field in CELL_FIELDS)
        for row in rows
        if all(str(row.get(field, "")).strip() for field in CELL_FIELDS)
    }


def benchmark_contract(metadata: dict[str, object]) -> dict[str, object]:
    ignored = {"measurement_mode", "repeat"}
    return {
        key: value
        for key, value in (metadata.get("benchmark_args") or {}).items()
        if key not in ignored
    }


def baseline_versions(
    version_data: dict[str, object], baselines: set[str]
) -> dict[str, object]:
    python_versions = version_data.get("python_baselines") or {}
    aliases = {
        "networkx": "networkx",
        "igraph": "igraph",
        "nx-cugraph": "nx-cugraph",
        "easygraph-cpu": "easygraph",
        "easygraph-cpp": "cpp_easygraph",
        "EGGPU": "cpp_easygraph",
    }
    selected: dict[str, object] = {}
    for baseline in sorted(baselines):
        if baseline == "Gunrock":
            selected[baseline] = version_data.get("gunrock_executables") or {}
            continue
        key = aliases.get(baseline)
        if key:
            selected[baseline] = python_versions.get(key)
    return selected


def require_files(root: Path) -> None:
    required = (
        "results_samples.csv",
        "run_metadata.json",
        "baseline_versions.json",
        "dataset_stats.json",
    )
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("missing pass artifact(s): " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timing-repeat", required=True, type=int)
    parser.add_argument("--memory-repeat", required=True, type=int)
    parser.add_argument(
        "--allow-unrelated-source-drift",
        action="store_true",
        help=(
            "Accept an overall repository snapshot mismatch only when the "
            "evaluated baseline package versions and benchmark contract match."
        ),
    )
    args = parser.parse_args()

    timing = args.timing.resolve()
    memory = args.memory.resolve()
    output = args.output.resolve()
    require_files(timing)
    require_files(memory)

    timing_samples = read_csv(timing / "results_samples.csv")
    memory_samples = read_csv(memory / "results_samples.csv")
    if not timing_samples or not memory_samples:
        raise ValueError("timing and memory passes must both contain samples")
    if any(str(row.get("metric", "")).startswith("memory_") for row in timing_samples):
        raise ValueError("timing pass unexpectedly contains memory metrics")
    if any(
        not str(row.get("metric", "")).startswith("memory_")
        for row in memory_samples
    ):
        raise ValueError("memory pass unexpectedly contains timing metrics")

    timing_cells = cells(timing_samples)
    memory_cells = cells(memory_samples)
    if timing_cells != memory_cells:
        missing_timing = sorted(memory_cells - timing_cells)
        missing_memory = sorted(timing_cells - memory_cells)
        raise ValueError(
            "timing/memory cell mismatch: "
            f"missing_timing={missing_timing[:8]} "
            f"missing_memory={missing_memory[:8]}"
        )
    baselines = {cell[2] for cell in timing_cells}

    timing_metadata = json.loads((timing / "run_metadata.json").read_text())
    memory_metadata = json.loads((memory / "run_metadata.json").read_text())
    if benchmark_contract(timing_metadata) != benchmark_contract(memory_metadata):
        raise ValueError("timing/memory benchmark contracts differ")

    timing_versions = json.loads((timing / "baseline_versions.json").read_text())
    memory_versions = json.loads((memory / "baseline_versions.json").read_text())
    evaluated_timing_versions = baseline_versions(timing_versions, baselines)
    evaluated_memory_versions = baseline_versions(memory_versions, baselines)
    if evaluated_timing_versions != evaluated_memory_versions:
        raise ValueError(
            "evaluated baseline versions differ between timing and memory passes"
        )

    timing_snapshot = (
        timing_versions.get("paper_repo_source_snapshot") or {}
    ).get("digest", "")
    memory_snapshot = (
        memory_versions.get("paper_repo_source_snapshot") or {}
    ).get("digest", "")
    source_snapshot_match = bool(
        timing_snapshot and timing_snapshot == memory_snapshot
    )
    if not source_snapshot_match and not args.allow_unrelated_source_drift:
        raise ValueError(
            "repository source snapshots differ; rerun both passes or use the "
            "explicit exception after confirming the evaluated baseline "
            "package and benchmark contract are unchanged"
        )

    for row in timing_samples:
        row["measurement_phase"] = "timing"
    for row in memory_samples:
        row["measurement_phase"] = "memory"
    merged_samples = timing_samples + memory_samples
    merged_long = aggregate_sample_rows(timing_samples, args.timing_repeat)
    merged_long += aggregate_sample_rows(memory_samples, args.memory_repeat)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "results_samples.csv", merged_samples)
    write_csv(output / "results_long.csv", merged_long)
    write_measurement_schema(output / "measurement_schema.json")
    write_validation_outputs(output, merged_long)
    write_metric_csvs_no_pandas(output, merged_long)
    shutil.copy2(timing / "dataset_stats.json", output / "dataset_stats.json")
    shutil.copy2(memory / "baseline_versions.json", output / "baseline_versions.json")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_kind": "assembled_independent_timing_memory_passes",
        "timing": {
            "path": str(timing),
            "repeat": args.timing_repeat,
            "results_samples_sha256": sha256(timing / "results_samples.csv"),
            "run_metadata_sha256": sha256(timing / "run_metadata.json"),
        },
        "memory": {
            "path": str(memory),
            "repeat": args.memory_repeat,
            "results_samples_sha256": sha256(memory / "results_samples.csv"),
            "run_metadata_sha256": sha256(memory / "run_metadata.json"),
        },
        "baselines": sorted(baselines),
        "cells": len(timing_cells),
        "measurement_protocol": {
            "timing_memory_isolated": True,
            "timing_estimator": "arithmetic mean with sample standard deviation",
            "memory_estimator": "arithmetic mean with sample standard deviation",
            "primary_gpu_memory_metric": "memory_peak_gpu_proc_mb",
            "primary_host_memory_metric": "memory_peak_rss_mb",
        },
        "contract_checks": {
            "cell_sets_match": True,
            "benchmark_contract_match": True,
            "evaluated_baseline_versions_match": True,
            "source_snapshot_match": source_snapshot_match,
            "unrelated_source_drift_explicitly_accepted": bool(
                not source_snapshot_match and args.allow_unrelated_source_drift
            ),
        },
        "source_drift_scope": (
            "The repository snapshot changed between passes, but the evaluated "
            "baseline package metadata and all benchmark arguments other than "
            "measurement mode/repeat are identical."
            if not source_snapshot_match
            else "none"
        ),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
