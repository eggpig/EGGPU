#!/usr/bin/env python3
"""Reconcile independently rerun GraphScope pairs without hiding failures."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from benchmark_stats import aggregate_sample_rows


TIMING_METRICS = {"build", "e2e", "kernel"}
MEMORY_REQUIRED_METRIC = "memory_peak_rss_mb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--repair-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("timing", "memory"))
    parser.add_argument("--expected-samples", required=True, type=int)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fieldnames(rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames(rows), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def pair(row: dict[str, str]) -> tuple[str, str]:
    return row["dataset"], row["function"]


def publishable_pairs(
    rows: list[dict[str, str]], phase: str
) -> set[tuple[str, str]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("baseline") != "GraphScope":
            continue
        grouped.setdefault(pair(row), {})[row["metric"]] = row

    accepted: set[tuple[str, str]] = set()
    for key, metrics in grouped.items():
        required = (
            TIMING_METRICS if phase == "timing" else {MEMORY_REQUIRED_METRIC}
        )
        if not required.issubset(metrics):
            continue
        if all(
            metrics[metric].get("status") == "ok"
            and metrics[metric].get("publishable", "").lower() == "true"
            for metric in required
        ):
            accepted.add(key)
    return accepted


def copy_pair_artifacts(
    repair_dir: Path,
    output_dir: Path,
    phase: str,
    pairs: set[tuple[str, str]],
) -> None:
    for dataset, function in pairs:
        source = repair_dir / "samples" / phase / dataset / function
        target = output_dir / "samples" / phase / dataset / function
        if target.exists():
            shutil.rmtree(target)
        if source.is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)

        if phase == "timing":
            detail = repair_dir / "details" / dataset / f"GraphScope_{function}.npz"
            if detail.is_file():
                target_detail = (
                    output_dir / "details" / dataset / f"GraphScope_{function}.npz"
                )
                target_detail.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(detail, target_detail)


def main() -> None:
    args = parse_args()
    if args.output_dir.resolve() in {
        args.base_dir.resolve(),
        args.repair_dir.resolve(),
    }:
        raise SystemExit("output-dir must differ from base-dir and repair-dir")

    base_samples_path = args.base_dir / "results_samples.csv"
    repair_samples_path = args.repair_dir / "results_samples.csv"
    base_long_path = args.base_dir / "results_long.csv"
    repair_long_path = args.repair_dir / "results_long.csv"
    for path in (
        base_samples_path,
        repair_samples_path,
        base_long_path,
        repair_long_path,
    ):
        if not path.is_file():
            raise SystemExit(f"required artifact is missing: {path}")

    repair_long = read_csv(repair_long_path)
    accepted = publishable_pairs(repair_long, args.phase)
    if not accepted:
        raise SystemExit("repair artifact has no complete publishable pair")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.base_dir, args.output_dir)

    base_samples = read_csv(base_samples_path)
    repair_samples = read_csv(repair_samples_path)
    superseded = [row for row in base_samples if pair(row) in accepted]
    retained = [row for row in base_samples if pair(row) not in accepted]
    replacements = [row for row in repair_samples if pair(row) in accepted]
    combined = retained + replacements
    combined.sort(
        key=lambda row: (
            row.get("dataset", ""),
            row.get("function", ""),
            row.get("metric", ""),
            int(row.get("sample_index") or 0),
        )
    )

    write_csv(args.output_dir / "results_samples.csv", combined)
    write_csv(args.output_dir / "superseded_failed_samples.csv", superseded)
    aggregates = aggregate_sample_rows(
        combined, expected_samples=args.expected_samples
    )
    write_csv(args.output_dir / "results_long.csv", aggregates)
    for metric, filename in (
        ("build", "results_build.csv"),
        ("kernel", "results_kernel.csv"),
        ("e2e", "results_e2e.csv"),
    ):
        write_csv(
            args.output_dir / filename,
            [row for row in aggregates if row.get("metric") == metric],
        )
    write_csv(
        args.output_dir / "results_memory.csv",
        [
            row
            for row in aggregates
            if str(row.get("metric", "")).startswith("memory_")
        ],
    )
    copy_pair_artifacts(
        args.repair_dir, args.output_dir, args.phase, accepted
    )

    repair_record_dir = args.output_dir / "repair_provenance"
    repair_record_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "run_metadata.json",
        "baseline_versions.json",
        "measurement_schema.json",
    ):
        source = args.repair_dir / name
        if source.is_file():
            shutil.copy2(source, repair_record_dir / name)

    payload = {
        "phase": args.phase,
        "expected_samples": args.expected_samples,
        "base_dir": str(args.base_dir.resolve()),
        "repair_dir": str(args.repair_dir.resolve()),
        "accepted_repair_pairs": [
            {"dataset": dataset, "function": function}
            for dataset, function in sorted(accepted)
        ],
        "superseded_sample_rows": len(superseded),
        "replacement_sample_rows": len(replacements),
        "policy": (
            "A repair replaces a pair only when every required aggregate metric "
            "is status=ok and publishable=true. Original rows remain in "
            "superseded_failed_samples.csv."
        ),
        "command": sys.argv,
    }
    (args.output_dir / "REPAIR_RECONCILIATION.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
