#!/usr/bin/env python3
"""Replace historical EGGPU rows with one latest-source measurement run.

This is deliberately different from the earlier targeted-retention helper:
no historical EGGPU minimum is retained.  Every EGGPU timing and memory sample
comes from ``--latest-eggpu``; competing systems remain from ``--baseline-main``.
Correctness is regenerated after merging so current EGGPU result details are
validated against the existing independent baseline results.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from measurement_schema import write_measurement_schema
from run_full_baselines import write_metric_csvs_no_pandas
from validate_correctness import write_validation_outputs


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


def selected_datasets(latest_rows: list[dict[str, str]]) -> list[str]:
    names: list[str] = []
    for row in latest_rows:
        name = str(row.get("dataset", "")).strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise ValueError("latest EGGPU result contains no dataset rows")
    return names


def assert_latest_contract(
    samples: list[dict[str, str]],
    datasets: list[str],
) -> None:
    baselines = {row.get("baseline") for row in samples}
    if baselines != {"EGGPU"}:
        raise ValueError(
            "latest result must contain EGGPU only; observed baselines="
            + ",".join(sorted(str(item) for item in baselines))
        )
    phases = {str(row.get("measurement_phase", "")) for row in samples}
    if not {"timing", "memory"}.issubset(phases):
        raise ValueError(
            "latest result must contain isolated timing and memory phases; "
            f"observed={sorted(phases)}"
        )
    if set(selected_datasets(samples)) != set(datasets):
        raise ValueError("latest sample and aggregate dataset sets differ")


def merge_rows(
    baseline_rows: list[dict[str, str]],
    latest_rows: list[dict[str, str]],
    datasets: set[str],
) -> list[dict[str, str]]:
    retained = [
        row
        for row in baseline_rows
        if row.get("dataset") in datasets and row.get("baseline") != "EGGPU"
    ]
    current = [
        row
        for row in latest_rows
        if row.get("dataset") in datasets and row.get("baseline") == "EGGPU"
    ]
    return retained + current


def merge_version_manifests(
    baseline_path: Path,
    latest_path: Path,
) -> dict[str, object]:
    """Preserve baseline provenance while binding the current EGGPU binary."""

    baseline = (
        json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_path.is_file()
        else {}
    )
    latest = (
        json.loads(latest_path.read_text(encoding="utf-8"))
        if latest_path.is_file()
        else {}
    )
    merged = copy.deepcopy(baseline)
    merged["version_merge_policy"] = (
        "The retained baseline manifest remains authoritative for all non-EGGPU "
        "rows. The latest EGGPU runtime and source snapshot are recorded in "
        "dedicated fields and do not overwrite baseline runtime identities."
    )
    merged["retained_baseline_manifest"] = {
        "path": str(baseline_path),
        "sha256": sha256(baseline_path) if baseline_path.is_file() else None,
    }
    merged["latest_eggpu_manifest"] = {
        "path": str(latest_path),
        "sha256": sha256(latest_path) if latest_path.is_file() else None,
    }
    if "paper_repo_source_snapshot" in baseline:
        merged["retained_baseline_source_snapshot"] = baseline[
            "paper_repo_source_snapshot"
        ]
    if "paper_repo_source_snapshot" in latest:
        merged["latest_eggpu_source_snapshot"] = latest[
            "paper_repo_source_snapshot"
        ]
    latest_python = latest.get("python_baselines", {})
    merged_python = merged.setdefault("python_baselines", {})
    if "cpp_easygraph" in latest_python:
        merged_python["eggpu_cpp_easygraph"] = latest_python["cpp_easygraph"]
    if "easygraph" in latest_python:
        merged_python["eggpu_easygraph"] = latest_python["easygraph"]
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-main", type=Path, required=True)
    parser.add_argument("--latest-eggpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_main = args.baseline_main.resolve()
    latest_eggpu = args.latest_eggpu.resolve()
    output = args.output.resolve()
    required = (
        baseline_main / "results_samples.csv",
        baseline_main / "results_long.csv",
        latest_eggpu / "results_samples.csv",
        latest_eggpu / "results_long.csv",
        latest_eggpu / "run_metadata.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing result input(s): " + ", ".join(missing))

    latest_samples = read_csv(latest_eggpu / "results_samples.csv")
    latest_long = read_csv(latest_eggpu / "results_long.csv")
    datasets = selected_datasets(latest_long)
    assert_latest_contract(latest_samples, datasets)
    dataset_set = set(datasets)

    merged_samples = merge_rows(
        read_csv(baseline_main / "results_samples.csv"),
        latest_samples,
        dataset_set,
    )
    merged_long = merge_rows(
        read_csv(baseline_main / "results_long.csv"),
        latest_long,
        dataset_set,
    )
    if not merged_samples or not merged_long:
        raise RuntimeError("merge produced an empty result")

    if output.exists():
        raise FileExistsError(
            f"refusing to modify existing authoritative output: {output}"
        )
    output.mkdir(parents=True)
    write_csv(output / "results_samples.csv", merged_samples)
    write_csv(output / "results_long.csv", merged_long)
    write_measurement_schema(output / "measurement_schema.json")

    stats_path = baseline_main / "dataset_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        filtered = [row for row in stats if row.get("name") in dataset_set]
        (output / "dataset_stats.json").write_text(
            json.dumps(filtered, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # Regenerate current-code correctness before exposing comparison tables.
    write_validation_outputs(output, merged_long)
    write_metric_csvs_no_pandas(output, merged_long)

    latest_versions = latest_eggpu / "baseline_versions.json"
    baseline_versions = baseline_main / "baseline_versions.json"
    versions = merge_version_manifests(baseline_versions, latest_versions)
    if versions:
        (output / "baseline_versions.json").write_text(
            json.dumps(versions, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    baseline_metadata = baseline_main / "run_metadata.json"
    latest_metadata = latest_eggpu / "run_metadata.json"
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_kind": "latest_architecture_eggpu_with_retained_baselines",
        "datasets": datasets,
        "eggpu_policy": (
            "all EGGPU raw rows come exclusively from latest_eggpu; "
            "paper timing uses the minimum of five samples downstream; "
            "memory uses three isolated samples and arithmetic mean"
        ),
        "baseline_policy": (
            "non-EGGPU raw rows are retained from baseline_main; "
            "paper timing uses the arithmetic mean of five samples"
        ),
        "correctness_policy": (
            "correctness_validation.csv was regenerated from the merged "
            "current EGGPU and independent baseline result details"
        ),
        "inputs": {
            "baseline_main": str(baseline_main),
            "latest_eggpu": str(latest_eggpu),
            "baseline_results_samples_sha256": sha256(
                baseline_main / "results_samples.csv"
            ),
            "latest_results_samples_sha256": sha256(
                latest_eggpu / "results_samples.csv"
            ),
            "baseline_run_metadata_sha256": (
                sha256(baseline_metadata) if baseline_metadata.is_file() else None
            ),
            "latest_run_metadata_sha256": sha256(latest_metadata),
        },
        "row_counts": {
            "samples": len(merged_samples),
            "aggregates": len(merged_long),
            "eggpu_samples": sum(
                row.get("baseline") == "EGGPU" for row in merged_samples
            ),
            "baseline_samples": sum(
                row.get("baseline") != "EGGPU" for row in merged_samples
            ),
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "LATEST_EGGPU_MERGE.md").write_text(
        "# Latest EGGPU evidence merge\n\n"
        f"- Current EGGPU source: `{latest_eggpu}`\n"
        f"- Retained baseline source: `{baseline_main}`\n"
        f"- Datasets: {', '.join(datasets)}\n"
        "- No historical EGGPU timing or memory row is retained.\n"
        "- Correctness and comparison-eligible metric tables were regenerated.\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
