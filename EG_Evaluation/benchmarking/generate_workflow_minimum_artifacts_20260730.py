#!/usr/bin/env python3
"""Generate controlled workflow artifacts with min5 centers and five-run SDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_BASELINES = ("EGGPU", "EGGPU-isolated")
EXPECTED_FUNCTIONS = ("WCC", "PageRank", "BFS", "SSSP", "Closeness")
EXPECTED_SAMPLES = {1, 2, 3, 4, 5}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geomean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(float)
    if len(numeric) == 0 or np.any(~np.isfinite(numeric)) or np.any(numeric <= 0):
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.log(numeric).mean()))


def aggregate_sample_sd(
    data: pd.DataFrame,
    *,
    baseline: str,
    position: int,
    function: str,
    value_column: str,
    dataset_names: list[str],
) -> float:
    """SD across five repetition-index geometric means on the fixed graphs."""

    part = data[
        data["baseline"].eq(baseline)
        & pd.to_numeric(data["call_position"], errors="raise").eq(position)
        & data["function"].eq(function)
    ]
    aggregate_samples = []
    expected_datasets = set(dataset_names)
    for sample_index in sorted(EXPECTED_SAMPLES):
        sample = part[
            pd.to_numeric(part["sample_index"], errors="raise").eq(sample_index)
        ]
        if len(sample) != len(dataset_names) or set(sample["dataset"]) != expected_datasets:
            raise ValueError(
                f"{baseline}/{function}/sample-{sample_index}: incomplete "
                "workflow dataset intersection"
            )
        aggregate_samples.append(geomean(sample[value_column]))
    return float(np.std(aggregate_samples, ddof=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = args.samples.resolve(strict=True)
    data = pd.read_csv(source, low_memory=False)
    required = {
        "dataset",
        "baseline",
        "sample_index",
        "call_position",
        "function",
        "call_seconds",
        "cumulative_seconds",
        "kernel_seconds",
        "status",
        "result_validation",
    }
    if not required.issubset(data.columns):
        raise ValueError(f"workflow samples lack {sorted(required - set(data.columns))}")
    data = data[
        data["baseline"].isin(EXPECTED_BASELINES)
        & data["function"].isin(EXPECTED_FUNCTIONS)
    ].copy()
    if not data["status"].astype(str).eq("ok").all():
        raise ValueError("workflow evidence contains a non-successful sample")
    if set(data["baseline"]) != set(EXPECTED_BASELINES):
        raise ValueError("workflow evidence lacks a controlled baseline")
    if set(data["function"]) != set(EXPECTED_FUNCTIONS):
        raise ValueError("workflow evidence lacks a required function")

    dataset_names = list(dict.fromkeys(data["dataset"].tolist()))
    grouped = data.groupby(
        ["dataset", "baseline", "call_position", "function"], sort=False
    )
    detail_rows = []
    for (dataset, baseline, position, function), group in grouped:
        indices = set(pd.to_numeric(group["sample_index"], errors="raise").astype(int))
        if indices != EXPECTED_SAMPLES or len(group) != 5:
            raise ValueError(
                f"{dataset}/{baseline}/{function}: sample indices={sorted(indices)}"
            )
        detail_rows.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "call_position": int(position),
                "function": function,
                "sample_count": 5,
                "call_min_seconds": float(group["call_seconds"].min()),
                "call_mean_seconds": float(group["call_seconds"].mean()),
                "call_sample_sd_seconds": float(group["call_seconds"].std(ddof=1)),
                "cumulative_min_seconds": float(group["cumulative_seconds"].min()),
                "cumulative_mean_seconds": float(group["cumulative_seconds"].mean()),
                "cumulative_sample_sd_seconds": float(
                    group["cumulative_seconds"].std(ddof=1)
                ),
                "kernel_min_seconds": float(group["kernel_seconds"].min()),
                "kernel_mean_seconds": float(group["kernel_seconds"].mean()),
                "kernel_sample_sd_seconds": float(group["kernel_seconds"].std(ddof=1)),
            }
        )
    details = pd.DataFrame(detail_rows).sort_values(
        ["baseline", "call_position", "dataset"]
    )
    expected_groups = (
        len(dataset_names) * len(EXPECTED_BASELINES) * len(EXPECTED_FUNCTIONS)
    )
    if len(details) != expected_groups:
        raise ValueError(f"workflow groups={len(details)}, expected={expected_groups}")

    cumulative_rows = []
    for baseline in EXPECTED_BASELINES:
        for position, function in enumerate(EXPECTED_FUNCTIONS, start=1):
            part = details[
                details["baseline"].eq(baseline)
                & details["call_position"].eq(position)
                & details["function"].eq(function)
            ]
            if len(part) != len(dataset_names):
                raise ValueError(f"{baseline}/{function}: incomplete dataset intersection")
            cumulative_rows.append(
                {
                    "baseline": baseline,
                    "call_position": position,
                    "function": function,
                    "datasets": len(part),
                    "common_dataset_names": ";".join(dataset_names),
                    "cumulative_geomean_seconds": geomean(
                        part["cumulative_min_seconds"]
                    ),
                    "cumulative_sample_sd_seconds": aggregate_sample_sd(
                        data,
                        baseline=baseline,
                        position=position,
                        function=function,
                        value_column="cumulative_seconds",
                        dataset_names=dataset_names,
                    ),
                    "aggregate_sample_count": 5,
                    "estimator": "minimum of five fresh processes",
                    "reported_error": "sample standard deviation (ddof=1)",
                }
            )
    cumulative = pd.DataFrame(cumulative_rows)

    self_control_rows = []
    for position, function in enumerate(EXPECTED_FUNCTIONS, start=1):
        for metric, field, raw_field in (
            ("e2e", "call_min_seconds", "call_seconds"),
            ("kernel", "kernel_min_seconds", "kernel_seconds"),
        ):
            reuse = details[
                details["baseline"].eq("EGGPU")
                & details["call_position"].eq(position)
            ]
            isolated = details[
                details["baseline"].eq("EGGPU-isolated")
                & details["call_position"].eq(position)
            ]
            reuse_seconds = geomean(reuse[field])
            isolated_seconds = geomean(isolated[field])
            self_control_rows.append(
                {
                    "call_position": position,
                    "function": function,
                    "metric": metric,
                    "datasets": len(dataset_names),
                    "reuse_seconds": reuse_seconds,
                    "isolated_seconds": isolated_seconds,
                    "reuse_sample_sd_seconds": aggregate_sample_sd(
                        data,
                        baseline="EGGPU",
                        position=position,
                        function=function,
                        value_column=raw_field,
                        dataset_names=dataset_names,
                    ),
                    "isolated_sample_sd_seconds": aggregate_sample_sd(
                        data,
                        baseline="EGGPU-isolated",
                        position=position,
                        function=function,
                        value_column=raw_field,
                        dataset_names=dataset_names,
                    ),
                    "aggregate_sample_count": 5,
                    "isolated_over_reuse": isolated_seconds / reuse_seconds,
                    "estimator": "minimum of five fresh processes",
                    "reported_error": "sample standard deviation (ddof=1)",
                }
            )
    self_control = pd.DataFrame(self_control_rows)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "workflow_five_call_raw_samples.csv"
    details_path = output / "workflow_five_call_minimum_dataset_details.csv"
    cumulative_path = output / "workflow_five_call_reuse_cumulative_minimum.csv"
    control_path = output / "workflow_five_call_self_control_minimum.csv"
    data.to_csv(raw_path, index=False)
    details.to_csv(details_path, index=False)
    cumulative.to_csv(cumulative_path, index=False)
    self_control.to_csv(control_path, index=False)

    final_reuse = cumulative[
        cumulative["baseline"].eq("EGGPU")
        & cumulative["call_position"].eq(len(EXPECTED_FUNCTIONS))
    ].iloc[0]
    final_isolated = cumulative[
        cumulative["baseline"].eq("EGGPU-isolated")
        & cumulative["call_position"].eq(len(EXPECTED_FUNCTIONS))
    ].iloc[0]
    kernel = self_control[self_control["metric"].eq("kernel")]
    headline = {
        "retained_cumulative_seconds": float(
            final_reuse["cumulative_geomean_seconds"]
        ),
        "retained_cumulative_sample_sd_seconds": float(
            final_reuse["cumulative_sample_sd_seconds"]
        ),
        "rebuild_cumulative_seconds": float(
            final_isolated["cumulative_geomean_seconds"]
        ),
        "rebuild_cumulative_sample_sd_seconds": float(
            final_isolated["cumulative_sample_sd_seconds"]
        ),
        "rebuild_over_retained": float(
            final_isolated["cumulative_geomean_seconds"]
            / final_reuse["cumulative_geomean_seconds"]
        ),
        "kernel_ratio_min": float(kernel["isolated_over_reuse"].min()),
        "kernel_ratio_max": float(kernel["isolated_over_reuse"].max()),
    }
    manifest = {
        "schema_version": "controlled_workflow_minimum_of_five_with_sd_v2",
        "status": "pass",
        "source": str(source),
        "source_sha256": sha256(source),
        "datasets": dataset_names,
        "baselines": list(EXPECTED_BASELINES),
        "workflow": list(EXPECTED_FUNCTIONS),
        "samples_per_dataset_baseline": 5,
        "estimator": "minimum_of_five_fresh_processes",
        "reported_error": "sample_standard_deviation_ddof1_same_five_processes",
        "aggregation": (
            "minimum per dataset/baseline/call boundary, followed by a "
            "geometric mean over the fixed six-dataset complete intersection; "
            "error is the sample SD across the five repetition-index geometric "
            "means on that same intersection"
        ),
        "headline": headline,
        "artifacts": {
            path.name: sha256(path)
            for path in (raw_path, details_path, cumulative_path, control_path)
        },
    }
    manifest_path = output / "WORKFLOW_MINIMUM_OF_FIVE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(headline, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
