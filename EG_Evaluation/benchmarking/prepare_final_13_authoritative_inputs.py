#!/usr/bin/env python3
"""Create immutable, provenance-preserving inputs from validated current runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_stats import aggregate_sample_rows
from run_full_baselines import write_metric_csvs_no_pandas


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"authoritative output is immutable and already exists: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def detail_sha(text: str) -> str:
    match = re.search(r"detail_sha=([^,; ]+)", str(text or ""))
    return match.group(1) if match else ""


def detail_path(text: str) -> Path:
    match = re.search(r"detail=([^,;]+)", str(text or ""))
    if not match:
        raise ValueError("correctness record has no detail path")
    return Path(match.group(1))


def row_value(row: dict) -> float:
    value = row.get("value", row.get("seconds"))
    return float(value)


def sample_group(rows: list[dict], dataset: str, function: str, metric: str) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("dataset") == dataset
        and row.get("function") == function
        and row.get("baseline") == "EGGPU"
        and row.get("metric") == metric
        and row.get("status") == "ok"
    ]


def prepare_main(main: Path, retest: Path, output: Path) -> list[dict]:
    create_fresh_output(output)
    for filename in (
        "baseline_versions.json",
        "dataset_stats.json",
        "measurement_schema.json",
        "notes.txt",
        "run_metadata.json",
    ):
        source = main / filename
        if source.is_file():
            shutil.copy2(source, output / filename)

    original = read_rows(main / "results_samples.csv")
    targeted = read_rows(retest / "results_samples.csv")
    selected = list(original)
    decisions = []
    keys = sorted(
        {
            (row["dataset"], row["function"], row["metric"])
            for row in targeted
            if row.get("baseline") == "EGGPU" and row.get("metric") in {"e2e", "kernel"}
        }
    )
    for dataset, function, metric in keys:
        old_group = sample_group(original, dataset, function, metric)
        new_group = sample_group(targeted, dataset, function, metric)
        if len(old_group) != 5 or len(new_group) != 5:
            raise RuntimeError(
                f"targeted retest requires two five-sample groups: "
                f"{dataset}/{function}/{metric}, old={len(old_group)}, new={len(new_group)}"
            )
        old_hashes = {detail_sha(row.get("correctness", "")) for row in old_group}
        new_hashes = {detail_sha(row.get("correctness", "")) for row in new_group}
        if not (old_hashes & new_hashes) or "" in old_hashes | new_hashes:
            raise RuntimeError(
                f"correctness detail hash mismatch: {dataset}/{function}/{metric}: "
                f"old={sorted(old_hashes)}, new={sorted(new_hashes)}"
            )
        old_best = min(row_value(row) for row in old_group)
        new_best = min(row_value(row) for row in new_group)
        selected = [
            row
            for row in selected
            if not (
                row.get("dataset") == dataset
                and row.get("function") == function
                and row.get("baseline") == "EGGPU"
                and row.get("metric") == metric
            )
        ]
        selected.extend(dict(row) for row in new_group)
        decisions.append(
            {
                "dataset": dataset,
                "function": function,
                "metric": metric,
                "old_best_seconds": old_best,
                "new_best_seconds": new_best,
                "selected_source": "current_qualified_retest",
                "replacement_decision": "replace_historical_current_system_row",
                "correctness_hash": next(iter(old_hashes & new_hashes)),
            }
        )

    timing = [row for row in selected if not str(row.get("metric", "")).startswith("memory_")]
    memory = [row for row in selected if str(row.get("metric", "")).startswith("memory_")]
    aggregated = aggregate_sample_rows(timing, expected_samples=5)
    aggregated.extend(aggregate_sample_rows(memory, expected_samples=3))
    write_rows(output / "results_samples.csv", selected)
    write_rows(output / "results_long.csv", aggregated)
    # Every targeted group is admitted only after its complete-result detail
    # SHA matches the original group.  Re-reading all 17 x 16 validation
    # vectors here is therefore redundant and can take several minutes.
    for filename in ("correctness_validation.csv", "correctness_validation.md"):
        source = main / filename
        if source.is_file():
            shutil.copy2(source, output / filename)
    write_metric_csvs_no_pandas(output, aggregated)
    pd.DataFrame(decisions).to_csv(output / "TARGETED_EGGPU_SELECTION_DECISIONS.csv", index=False)

    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    metadata["authoritative_view"] = {
        "original_main": str(main.resolve()),
        "targeted_retest": str(retest.resolve()),
        "selection_rule": (
            "A correctness-qualified current EGGPU five-sample group replaces the "
            "historical EGGPU group regardless of which version is faster. The "
            "selected group's sample standard deviation is retained. Baselines are "
            "unchanged."
        ),
        "policy_id": "eggpu-vldb-final-evidence-v1",
        "source_sha256": {
            "original_results_samples": sha256(main / "results_samples.csv"),
            "current_results_samples": sha256(retest / "results_samples.csv"),
        },
        "decisions": decisions,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return decisions


def exact_closeness_validation(
    current_rows: list[dict], reference_rows: list[dict], dataset: str
) -> dict:
    gpu = next(
        row for row in current_rows
        if row.get("dataset") == dataset and row.get("metric") == "e2e"
    )
    cpu = next(
        row for row in reference_rows
        if row.get("dataset") == dataset and row.get("metric") == "e2e"
    )
    gpu_data = np.load(detail_path(gpu.get("correctness", "")), allow_pickle=False)
    cpu_data = np.load(detail_path(cpu.get("correctness", "")), allow_pickle=False)
    gpu_values = np.asarray(gpu_data["values"], dtype=np.float64).reshape(-1)
    cpu_values = np.asarray(cpu_data["values"], dtype=np.float64).reshape(-1)
    gpu_sources = np.asarray(gpu_data["sources"]).reshape(-1)
    cpu_sources = np.asarray(cpu_data["sources"]).reshape(-1)
    if not np.array_equal(gpu_sources, cpu_sources):
        raise RuntimeError(f"Closeness source mismatch for {dataset}")
    if gpu_values.shape != cpu_values.shape:
        raise RuntimeError(f"Closeness result-shape mismatch for {dataset}")
    difference = np.abs(gpu_values - cpu_values)
    max_abs = float(difference.max(initial=0.0))
    denominator = np.maximum(np.abs(cpu_values), 1.0)
    max_rel = float((difference / denominator).max(initial=0.0))
    if not np.allclose(gpu_values, cpu_values, rtol=1e-7, atol=1e-10):
        raise RuntimeError(
            f"Closeness external validation failed for {dataset}: "
            f"max_abs={max_abs}, max_rel={max_rel}"
        )
    return {
        "dataset": dataset,
        "sample_sources": len(gpu_sources),
        "sources_equal": True,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "validation_status": "pass",
        "reference": "easygraph-cpu",
    }


def prepare_closeness(
    old: Path, current: Path, reference: Path, output: Path
) -> list[dict]:
    create_fresh_output(output)
    old_build_path = old / "closeness_large_sampled_build.csv"
    if not old_build_path.is_file():
        raise FileNotFoundError(
            "historical Closeness construction evidence is required: "
            f"{old_build_path}"
        )
    # The current-code retest targets the public-call and device intervals.  Its
    # correctness qualification does not supersede the separately measured
    # graph-construction samples, so preserve the historical build ledger
    # verbatim in the immutable authoritative input.
    shutil.copy2(old_build_path, output / old_build_path.name)
    current_rows = read_rows(current / "measurement_passes" / "timing" / "results_samples.csv")
    reference_rows = read_rows(reference / "results_samples.csv")
    datasets = sorted({row["dataset"] for row in current_rows if row.get("metric") == "e2e"})
    exact = [exact_closeness_validation(current_rows, reference_rows, dataset) for dataset in datasets]
    exact_by_dataset = {row["dataset"]: row for row in exact}
    decisions = []

    for metric in ("e2e", "kernel"):
        old_path = old / f"closeness_large_sampled_{metric}.csv"
        old_rows = read_rows(old_path)
        columns = list(old_rows[0])
        final_rows = list(old_rows)
        for dataset in datasets:
            old_group = [
                row for row in old_rows
                if row.get("dataset") == dataset and row.get("baseline") == "EGGPU"
            ]
            new_group = [
                row for row in current_rows
                if row.get("dataset") == dataset
                and row.get("baseline") == "EGGPU"
                and row.get("metric") == metric
                and row.get("status") == "ok"
            ]
            if len(old_group) != 5 or len(new_group) != 5:
                raise RuntimeError(
                    f"Closeness requires old/new five-sample groups: "
                    f"{dataset}/{metric}, old={len(old_group)}, new={len(new_group)}"
                )
            old_best = min(float(row["seconds"]) for row in old_group)
            new_best = min(row_value(row) for row in new_group)
            final_rows = [
                row for row in final_rows
                if not (row.get("dataset") == dataset and row.get("baseline") == "EGGPU")
            ]
            template = dict(old_group[0])
            for row in new_group:
                converted = {column: template.get(column, "") for column in columns}
                for key in (
                    "dataset_size", "graph_type", "dataset", "function", "baseline",
                    "metric", "status", "correctness", "log", "notes", "semantic",
                    "skip_reason", "estimator_kind", "sample_sources", "source_policy",
                    "source_seed", "source_nodes_sha", "sample_index",
                ):
                    if key in converted:
                        converted[key] = row.get(key, converted[key])
                converted["seconds"] = str(row_value(row))
                converted["is_timeout"] = "False"
                converted["exact_matrix_inclusion"] = "False"
                converted["is_supplement"] = "True"
                converted["supplement_reason"] = "current_code_external_exact_validation"
                converted["notes"] = (
                    str(converted.get("notes", ""))
                    + "; externally exact-matched to easygraph-cpu on identical sources"
                )
                final_rows.append(converted)
            decisions.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "old_best_seconds": old_best,
                    "new_best_seconds": new_best,
                    "selected_source": "current_qualified_retest",
                    "replacement_decision": "replace_historical_current_system_row",
                    **exact_by_dataset[dataset],
                }
            )
        write_rows(output / old_path.name, final_rows)

    validation = read_rows(old / "closeness_large_sampled_validation.csv")
    if not validation:
        raise RuntimeError("historical Closeness validation table is empty")
    validation_template = dict(validation[0])
    replaced = {
        row["dataset"]
        for row in decisions
        if row["metric"] == "e2e"
    }
    validation = [
        row for row in validation
        if not (row.get("baseline") == "EGGPU" and row.get("dataset") in replaced)
    ]
    template = validation_template
    for dataset in sorted(replaced):
        source_row = next(
            row for row in current_rows
            if row.get("dataset") == dataset and row.get("metric") == "e2e"
        )
        for sample_index in range(1, 6):
            row = {key: "" for key in template}
            row.update(
                {
                    "dataset": dataset,
                    "function": "Closeness",
                    "baseline": "EGGPU",
                    "sample_index": sample_index,
                    "reference": "easygraph-cpu",
                    "validation_status": "pass",
                    "max_abs": exact_by_dataset[dataset]["max_abs"],
                    "max_rel": exact_by_dataset[dataset]["max_rel"],
                    "semantic": source_row.get("semantic", "sampled_target_exact"),
                    "estimator_kind": source_row.get("estimator_kind", "exact_selected_vertices"),
                    "sample_sources": source_row.get("sample_sources", "16"),
                    "source_policy": source_row.get("source_policy", "deterministic_evenly_spaced"),
                    "source_seed": source_row.get("source_seed", "none"),
                    "source_nodes_sha": source_row.get("source_nodes_sha", ""),
                }
            )
            validation.append(row)
    write_rows(output / "closeness_large_sampled_validation.csv", validation)
    pd.DataFrame(exact).to_csv(output / "CURRENT_CLOSENESS_EXTERNAL_VALIDATION.csv", index=False)
    pd.DataFrame(decisions).to_csv(output / "CURRENT_CLOSENESS_SELECTION_DECISIONS.csv", index=False)
    (output / "AUTHORITATIVE_INPUT_MANIFEST.json").write_text(
        json.dumps(
            {
                "policy_id": "eggpu-vldb-final-evidence-v1",
                "selection_rule": (
                    "Every externally validated current EGGPU five-sample group "
                    "replaces its historical EGGPU E2E/device group; baseline "
                    "groups and the independently measured construction ledger "
                    "remain unchanged."
                ),
                "sources": {
                    "historical_closeness": str(old.resolve()),
                    "current_closeness": str(current.resolve()),
                    "external_reference": str(reference.resolve()),
                },
                "source_sha256": {
                    "historical_build": sha256(old_build_path),
                    "historical_e2e": sha256(old / "closeness_large_sampled_e2e.csv"),
                    "historical_kernel": sha256(old / "closeness_large_sampled_kernel.csv"),
                    "current_results_samples": sha256(
                        current / "measurement_passes" / "timing" / "results_samples.csv"
                    ),
                    "reference_results_samples": sha256(reference / "results_samples.csv"),
                },
                "decisions": decisions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--targeted-retest", required=True, type=Path)
    parser.add_argument("--main-output", required=True, type=Path)
    parser.add_argument("--old-closeness", required=True, type=Path)
    parser.add_argument("--current-closeness", required=True, type=Path)
    parser.add_argument("--closeness-reference", required=True, type=Path)
    parser.add_argument("--closeness-output", required=True, type=Path)
    args = parser.parse_args()

    main_decisions = prepare_main(
        args.main.resolve(), args.targeted_retest.resolve(), args.main_output.resolve()
    )
    closeness_decisions = prepare_closeness(
        args.old_closeness.resolve(),
        args.current_closeness.resolve(),
        args.closeness_reference.resolve(),
        args.closeness_output.resolve(),
    )
    print(
        json.dumps(
            {
                "main_decisions": len(main_decisions),
                "main_replacements": sum(
                    row["selected_source"] == "current_qualified_retest"
                    for row in main_decisions
                ),
                "closeness_decisions": len(closeness_decisions),
                "closeness_replacements": sum(
                    row["selected_source"] == "current_qualified_retest"
                    for row in closeness_decisions
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
