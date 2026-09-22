#!/usr/bin/env python3
"""Audit the frozen V10 workflow run and regenerate paper-facing assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import pandas as pd

import generate_chapter4_evaluation_assets_graphscope as chapter4
from generate_converged_workflow_figure_20260729 import generate_figure


DATASETS = tuple(chapter4.WORKFLOW_DATASETS)
BASELINES = ("EGGPU", "EGGPU-isolated")
FUNCTIONS = ("WCC", "PageRank", "BFS", "SSSP", "Closeness")
RAW_NAME = re.compile(r"(.+)_(EGGPU(?:-isolated)?)_([1-5])\.json$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def runtime_identity(
    provenance: dict,
    candidate_sha256: str,
    runtime_sha256: str,
    label: str,
) -> None:
    if provenance.get("native_sha256") != candidate_sha256:
        raise ValueError(f"{label}: native candidate SHA differs")
    snapshot = provenance.get("runtime_python_snapshot") or {}
    if snapshot:
        if snapshot.get("digest") != runtime_sha256:
            raise ValueError(f"{label}: Python runtime SHA differs")
        if snapshot.get("package_is_symlink") is not False:
            raise ValueError(f"{label}: Python package is not frozen")


def validate_and_copy_raw(
    workflow: Path,
    output: Path,
    candidate_sha256: str,
    runtime_sha256: str,
) -> list[dict[str, object]]:
    raw_paths = sorted((workflow / "raw").glob("*.json"))
    expected_keys = {
        (dataset, baseline, sample)
        for dataset in DATASETS
        for baseline in BASELINES
        for sample in range(1, 6)
    }
    observed_keys = set()
    inventory = []
    destination_dir = output / "source_protocol_evidence" / "workflow_raw"
    destination_dir.mkdir(parents=True, exist_ok=False)
    for path in raw_paths:
        match = RAW_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected workflow raw filename: {path.name}")
        key = (match.group(1), match.group(2), int(match.group(3)))
        if key not in expected_keys or key in observed_keys:
            raise ValueError(f"unexpected or duplicate workflow raw key: {key}")
        observed_keys.add(key)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            raise ValueError(f"{path}: raw sample status is not ok")
        requested = payload.get("requested_runtime_provenance") or {}
        runtime_identity(
            requested,
            candidate_sha256,
            runtime_sha256,
            str(path),
        )
        observed_runtime = payload.get("runtime_provenance") or {}
        if observed_runtime.get("native_sha256") != candidate_sha256:
            raise ValueError(f"{path}: observed native candidate SHA differs")
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != 5:
            raise ValueError(f"{path}: expected exactly five workflow calls")
        for position, (row, function) in enumerate(
            zip(rows, FUNCTIONS, strict=True),
            start=1,
        ):
            if (
                row.get("status") != "ok"
                or row.get("result_validation") != "pass"
                or row.get("dataset") != key[0]
                or row.get("baseline") != key[1]
                or int(row.get("sample_index", -1)) != key[2]
                or int(row.get("call_position", -1)) != position
                or row.get("function") != function
                or row.get("workflow") != list(FUNCTIONS)
            ):
                raise ValueError(f"{path}: workflow call {position} differs")
            row_runtime = row.get("runtime_provenance") or {}
            if row_runtime.get("native_sha256") != candidate_sha256:
                raise ValueError(f"{path}: row runtime candidate SHA differs")
            for field in ("call_seconds", "cumulative_seconds", "kernel_seconds"):
                value = float(row[field])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{path}: invalid {field}")
        destination = destination_dir / path.name
        shutil.copy2(path, destination)
        inventory.append(
            {
                "dataset": key[0],
                "baseline": key[1],
                "sample_index": key[2],
                "copied_path": str(destination.relative_to(output)),
                "sha256": sha256(destination),
                "calls": 5,
            }
        )
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        raise ValueError(
            f"workflow raw sample files={len(observed_keys)}/60; missing={missing}"
        )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    args = parser.parse_args()

    workflow = args.workflow_result.resolve()
    output = args.output_dir.resolve()
    metadata_path = workflow / "metadata.json"
    summary_path = workflow / "cumulative_workflow_summary.csv"
    samples_path = workflow / "cumulative_workflow_samples.csv"
    failures_path = workflow / "cumulative_workflow_failures.csv"
    unsupported_path = workflow / "cumulative_workflow_unsupported.csv"
    for path in (
        metadata_path,
        summary_path,
        samples_path,
        failures_path,
        unsupported_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"V10 workflow result is incomplete: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "protocol": "fresh_process_cumulative_same_graph_workflow_v2",
        "datasets": list(DATASETS),
        "baselines": list(BASELINES),
        "repeat": 5,
        "workflow": list(FUNCTIONS),
        "failures": 0,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"workflow metadata {key}={metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    runtime_identity(
        metadata.get("runtime_provenance") or {},
        args.candidate_sha256,
        args.runtime_sha256,
        "workflow metadata",
    )
    observed_runtimes = metadata.get("observed_eggpu_runtimes")
    if not isinstance(observed_runtimes, list) or len(observed_runtimes) != 1:
        raise ValueError("workflow metadata does not contain one observed runtime")
    if observed_runtimes[0].get("native_sha256") != args.candidate_sha256:
        raise ValueError("workflow observed runtime candidate SHA differs")
    if read_csv_rows(failures_path) or read_csv_rows(unsupported_path):
        raise ValueError("workflow contains failures or unsupported calls")

    raw_inventory = validate_and_copy_raw(
        workflow,
        output,
        args.candidate_sha256,
        args.runtime_sha256,
    )
    summary = pd.read_csv(summary_path)
    samples = pd.read_csv(samples_path)
    expected_summary_keys = {
        (dataset, baseline, position, function)
        for dataset in DATASETS
        for baseline in BASELINES
        for position, function in enumerate(FUNCTIONS, start=1)
    }
    summary_keys = set(
        zip(
            summary["dataset"],
            summary["baseline"],
            summary["call_position"].astype(int),
            summary["function"],
        )
    )
    if (
        len(summary) != 60
        or summary_keys != expected_summary_keys
        or not pd.to_numeric(summary["sample_count"]).eq(5).all()
    ):
        raise ValueError("workflow summary is not 6 datasets x 2 variants x 5 calls")
    if len(samples) != 300:
        raise ValueError("workflow samples are not 60 sample files x 5 calls")
    sample_counts = samples.groupby(
        ["dataset", "baseline", "call_position", "function"]
    ).size()
    if len(sample_counts) != 60 or not sample_counts.eq(5).all():
        raise ValueError("workflow sample groups are not complete raw5 batches")
    if not samples["status"].eq("ok").all() or not samples[
        "result_validation"
    ].eq("pass").all():
        raise ValueError("workflow sample validation is not uniformly pass")

    summary.to_csv(output / "cumulative_workflow_dataset_details.csv", index=False)
    common_names = ";".join(DATASETS)
    cumulative_rows = []
    self_rows = []
    for baseline in BASELINES:
        for position, function in enumerate(FUNCTIONS, start=1):
            part = summary[
                summary["baseline"].eq(baseline)
                & summary["call_position"].eq(position)
            ]
            cumulative_rows.append(
                {
                    "baseline": baseline,
                    "call_position": position,
                    "function": function,
                    "datasets": len(part),
                    "common_dataset_names": common_names,
                    "cumulative_geomean_seconds": chapter4.geomean(
                        part["cumulative_mean_seconds"]
                    ),
                    "estimator": "arithmetic mean of five",
                }
            )
    cumulative = pd.DataFrame(cumulative_rows)
    cumulative_path = (
        output / "workflow_five_call_reuse_cumulative_arithmetic_mean.csv"
    )
    cumulative.to_csv(cumulative_path, index=False)
    cumulative.to_csv(output / "cumulative_workflow_aggregate.csv", index=False)

    reused = summary[summary["baseline"].eq("EGGPU")]
    isolated = summary[summary["baseline"].eq("EGGPU-isolated")]
    paired = reused.merge(
        isolated,
        on=("dataset", "call_position", "function"),
        suffixes=("_workflow", "_isolated"),
        validate="one_to_one",
    )
    paired.to_csv(output / "case_study_pair_details.csv", index=False)
    for position, function in enumerate(FUNCTIONS, start=1):
        part = paired[
            paired["call_position"].eq(position)
            & paired["function"].eq(function)
        ]
        for metric, field in (
            ("e2e", "call_mean_seconds"),
            ("kernel", "kernel_mean_seconds"),
        ):
            reuse = chapter4.geomean(part[f"{field}_workflow"])
            rebuild = chapter4.geomean(part[f"{field}_isolated"])
            self_rows.append(
                {
                    "call_position": position,
                    "function": function,
                    "metric": metric,
                    "datasets": len(part),
                    "reuse_seconds": reuse,
                    "isolated_seconds": rebuild,
                    "isolated_over_reuse": rebuild / reuse,
                    "estimator": "arithmetic mean of five",
                }
            )
    self_control = pd.DataFrame(self_rows)
    self_path = output / "workflow_five_call_self_control_arithmetic_mean.csv"
    self_control.to_csv(self_path, index=False)
    self_control.rename(
        columns={
            "reuse_seconds": "same_graph_workflow_geomean_seconds",
            "isolated_seconds": "isolated_geomean_seconds",
            "isolated_over_reuse": "isolated_over_workflow",
        }
    ).to_csv(output / "case_study_call_summary.csv", index=False)

    figure_path = output / "workflow_state_reuse.pdf"
    generate_figure(self_path, cumulative_path, figure_path)
    copied_metadata = output / "source_protocol_evidence" / "workflow_metadata.json"
    shutil.copy2(metadata_path, copied_metadata)
    final_position = len(FUNCTIONS)
    final_values = cumulative[cumulative["call_position"].eq(final_position)].set_index(
        "baseline"
    )["cumulative_geomean_seconds"]
    first_reuse = float(
        self_control[
            self_control["metric"].eq("e2e")
            & self_control["call_position"].eq(1)
        ]["reuse_seconds"].iloc[0]
    )
    final_reuse = float(final_values["EGGPU"])
    final_isolated = float(final_values["EGGPU-isolated"])
    provenance = {
        "status": "pass_v10_candidate_bound",
        "paper_role": (
            "retained-state versus rebuild-per-call causal self-control"
        ),
        "bound_to_v10_candidate_binary": True,
        "bound_to_v10_runtime_snapshot": True,
        "candidate_binary_sha256": args.candidate_sha256,
        "runtime_python_snapshot_sha256": args.runtime_sha256,
        "source_result_directory": str(workflow),
        "source_metadata_sha256": sha256(metadata_path),
        "source_summary_sha256": sha256(summary_path),
        "source_samples_sha256": sha256(samples_path),
        "raw_sample_files": 60,
        "calls_per_sample_file": 5,
        "summary_rows": 60,
        "sample_rows": 300,
        "datasets": list(DATASETS),
        "baselines": list(BASELINES),
        "workflow": list(FUNCTIONS),
        "statistics": {
            "within_dataset_estimator": (
                "arithmetic mean of five independent processes"
            ),
            "across_dataset_estimator": (
                "geometric mean over six fixed common datasets"
            ),
        },
        "headline": {
            "final_retained_seconds": final_reuse,
            "final_rebuild_seconds": final_isolated,
            "final_rebuild_over_retained": final_isolated / final_reuse,
            "first_call_share_of_retained_percent": (
                first_reuse / final_reuse * 100.0
            ),
        },
        "raw_evidence": raw_inventory,
        "artifacts": {
            self_path.name: sha256(self_path),
            cumulative_path.name: sha256(cumulative_path),
            figure_path.name: sha256(figure_path),
            figure_path.with_suffix(".png").name: sha256(
                figure_path.with_suffix(".png")
            ),
        },
    }
    provenance_path = output / "workflow_state_reuse_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "workflow_state_reuse_summary.json").write_text(
        json.dumps(provenance["headline"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
