#!/usr/bin/env python3
"""Assemble a matched cold-start study from already audited fresh-process runs."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


DATASETS = (
    "ca-HepTh",
    "LastFM",
    "p2p-Gnutella04",
    "ca-HepPh",
    "email-Enron",
    "ca-CondMat",
    "soc-Epinions1",
    "com-youtube",
    "ER-100k",
    "soc-Slashdot0811",
)
FUNCTIONS = ("PageRank", "LCC", "WCC", "BFS", "SSSP", "KCore")
BASELINES = ("EGGPU", "igraph", "nx-cugraph")
METRICS = ("build", "e2e", "kernel")
REPEAT = 5


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timing_metadata(result_dir):
    path = result_dir / "measurement_passes" / "timing" / "run_metadata.json"
    if not path.is_file():
        merged_path = result_dir / "run_metadata.json"
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
        artifact_dir = (
            (merged.get("measurement_protocol") or {})
            .get("timing_pass", {})
            .get("artifact_dir")
        )
        if not artifact_dir:
            raise RuntimeError(
                f"cannot resolve timing-pass metadata from {result_dir}"
            )
        path = Path(artifact_dir).resolve() / "run_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def require_protocol(main_dir, first_use_dir):
    main, main_path = timing_metadata(main_dir)
    first, first_path = timing_metadata(first_use_dir)
    main_args = main.get("benchmark_args") or {}
    first_args = first.get("benchmark_args") or {}
    failures = []
    expected = (
        ("main repeat", main_args.get("repeat"), REPEAT),
        ("first-use repeat", first_args.get("repeat"), REPEAT),
        ("main non-EGGPU warmup", main_args.get("warmup"), 0),
        ("first-use warmup", first_args.get("warmup"), 0),
        ("main SSSP sources", main_args.get("sssp_sources"), 8),
        ("first-use SSSP sources", first_args.get("sssp_sources"), 8),
        ("first-use execution protocol", first_args.get("eggpu_execution_protocol"), "first-use"),
    )
    for label, actual, wanted in expected:
        if actual != wanted:
            failures.append(f"{label}: expected {wanted!r}, found {actual!r}")
    compatibility_path = first_use_dir / "implementation_compatibility.json"
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    if compatibility.get("status") != "pass":
        failures.append("first-use implementation compatibility did not pass")
    if failures:
        raise RuntimeError("cold-start provenance check failed: " + "; ".join(failures))
    return {
        "main_timing_metadata": str(main_path.resolve()),
        "first_use_timing_metadata": str(first_path.resolve()),
        "implementation_compatibility": compatibility,
        "main_benchmark_args": main_args,
        "first_use_benchmark_args": first_args,
    }


def selected_sample_rows(path, baselines):
    rows, fields = read_rows(path)
    selected = []
    for row in rows:
        if row.get("measurement_phase") != "timing":
            continue
        if row.get("dataset") not in DATASETS:
            continue
        if row.get("function") not in FUNCTIONS:
            continue
        if row.get("baseline") not in baselines:
            continue
        if row.get("metric") not in METRICS:
            continue
        if row.get("status") != "ok":
            raise RuntimeError(
                "non-ok cold timing row: "
                f"{row.get('dataset')}/{row.get('function')}/"
                f"{row.get('baseline')}/{row.get('metric')}={row.get('status')}"
            )
        selected.append(row)
    return selected, fields


def validate_samples(rows):
    counts = Counter()
    sources = defaultdict(set)
    for row in rows:
        key = (
            row["dataset"],
            row["function"],
            row["baseline"],
            row["metric"],
        )
        counts[key] += 1
        source_sha = (row.get("source_nodes_sha") or "").strip()
        if source_sha:
            sources[(row["dataset"], row["function"])].add(source_sha)
    expected = {
        (dataset, function, baseline, metric)
        for dataset in DATASETS
        for function in FUNCTIONS
        for baseline in BASELINES
        for metric in METRICS
    }
    missing = sorted(expected - set(counts))
    extra = sorted(set(counts) - expected)
    wrong = sorted((key, value) for key, value in counts.items() if value != REPEAT)
    source_mismatch = sorted(key for key, values in sources.items() if len(values) > 1)
    if missing or extra or wrong or source_mismatch:
        raise RuntimeError(
            "cold-start sample contract failed: "
            f"missing={missing[:5]}, extra={extra[:5]}, wrong={wrong[:5]}, "
            f"source_mismatch={source_mismatch[:5]}"
        )


def selected_correctness_rows(path):
    rows, fields = read_rows(path)
    selected = [
        row
        for row in rows
        if row.get("dataset") in DATASETS
        and row.get("function") in FUNCTIONS
        and row.get("baseline") in BASELINES
    ]
    expected = len(DATASETS) * len(FUNCTIONS) * len(BASELINES)
    valid = {"pass", "reference"}
    bad = [
        row
        for row in selected
        if row.get("validation_status") not in valid
    ]
    if len(selected) != expected or bad:
        raise RuntimeError(
            f"cold-start correctness contract failed: rows={len(selected)}/{expected}, "
            f"bad={[(r.get('dataset'), r.get('function'), r.get('baseline'), r.get('validation_status')) for r in bad[:5]]}"
        )
    return selected, fields


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", required=True, type=Path)
    parser.add_argument("--first-use-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    main_dir = args.main_dir.resolve()
    first_use_dir = args.first_use_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = require_protocol(main_dir, first_use_dir)
    first_rows, first_fields = selected_sample_rows(
        first_use_dir / "results_samples.csv", {"EGGPU"}
    )
    competitor_rows, competitor_fields = selected_sample_rows(
        main_dir / "results_samples.csv", {"igraph", "nx-cugraph"}
    )
    if first_fields != competitor_fields:
        raise RuntimeError("main and first-use sample schemas differ")
    sample_rows = first_rows + competitor_rows
    validate_samples(sample_rows)
    sample_rows.sort(
        key=lambda row: (
            DATASETS.index(row["dataset"]),
            FUNCTIONS.index(row["function"]),
            BASELINES.index(row["baseline"]),
            int(row.get("sample_index") or 0),
            METRICS.index(row["metric"]),
        )
    )
    write_csv(output_dir / "results_samples.csv", first_fields, sample_rows)

    correctness_rows, correctness_fields = selected_correctness_rows(
        main_dir / "correctness_validation.csv"
    )
    correctness_rows.sort(
        key=lambda row: (
            DATASETS.index(row["dataset"]),
            FUNCTIONS.index(row["function"]),
            BASELINES.index(row["baseline"]),
        )
    )
    write_csv(
        output_dir / "correctness_validation.csv",
        correctness_fields,
        correctness_rows,
    )

    source_files = {
        "main_results_samples": main_dir / "results_samples.csv",
        "first_use_results_samples": first_use_dir / "results_samples.csv",
        "main_correctness_validation": main_dir / "correctness_validation.csv",
    }
    metadata = {
        "protocol": "audited_existing_fresh_process_cold_start_v1",
        "datasets": list(DATASETS),
        "functions": list(FUNCTIONS),
        "baselines": list(BASELINES),
        "timing_repeat": REPEAT,
        "measurement_scope": "timing_only",
        "user_cold_total": "paired graph build plus first supported function call",
        "eggpu_state": "first-use with no reusable graph state and no warmup",
        "competitor_state": "unwarmed fresh-process samples from the main run",
        "path_source_count": 8,
        "memory_policy": (
            "Cold-start memory is not remeasured; steady-state memory is taken "
            "from the split main experiment and large-graph memory from the "
            "independent scaling memory pass."
        ),
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in source_files.items()
        },
        **provenance,
    }
    (output_dir / "cold_start_reuse_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "sample_rows": len(sample_rows),
                "correctness_rows": len(correctness_rows),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
