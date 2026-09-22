#!/usr/bin/env python3
"""Rebuild the final-13 ledger from complete five-run timing evidence.

The V13 release preserved each source system's historical estimator.  That
made the paper-facing comparison asymmetric: EGGPU, strict nx-cugraph, and
Gunrock used the minimum of five, while CPU libraries and GraphScope used the
arithmetic mean of five. This script reconstructs every successful
build/processing/E2E cell from its five raw timing observations and emits a
portable filtered raw-sample table. It supports both the historical uniform
minimum policy and the final paper policy in which EGGPU uses the minimum of
five while every external baseline uses the arithmetic mean of five.

No run is discarded and no stability threshold selects a batch.  Arithmetic
means, sample standard deviations, medians, maxima, and coefficients of
variation remain in the ledger as descriptive evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

import stage_cpu_closeness_build_overlay as closeness_build


METRICS = ("build", "kernel", "e2e")
CPU_BASELINES = {"networkx", "easygraph-cpu", "easygraph-cpp", "igraph"}
EXPECTED_SAMPLES = {1, 2, 3, 4, 5}
UNIFORM_MINIMUM_POLICY = "uniform-minimum"
MIXED_ESTIMATOR_POLICY = "eggpu-minimum-baseline-mean"


class EvidenceError(ValueError):
    """Raised when a displayed timing cannot be reconstructed from raw data."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def close(lhs: object, rhs: object) -> bool:
    return finite(lhs) and finite(rhs) and math.isclose(
        float(lhs), float(rhs), rel_tol=2.0e-8, abs_tol=2.0e-11
    )


def cell_metric_key(
    dataset: object, function: object, baseline: object, metric: object
) -> tuple[str, str, str, str]:
    return tuple(map(str, (dataset, function, baseline, metric)))


def add_sample(
    samples: dict[tuple[str, str, str, str], dict[int, dict]],
    *,
    dataset: object,
    function: object,
    baseline: object,
    metric: object,
    sample_index: object,
    seconds: object,
    source: Path | str,
    source_sha256: str,
    replace: bool = False,
) -> None:
    key = cell_metric_key(dataset, function, baseline, metric)
    try:
        index = int(float(sample_index))
        value = float(seconds)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{key}: invalid sample {sample_index!r}/{seconds!r}") from exc
    if index not in EXPECTED_SAMPLES:
        raise EvidenceError(f"{key}: unexpected sample index {index}")
    if not math.isfinite(value) or value <= 0:
        raise EvidenceError(f"{key}: nonpositive or nonfinite sample {value}")
    if replace:
        samples[key] = {}
    bucket = samples[key]
    if index in bucket:
        previous = bucket[index]
        if not close(previous["seconds"], value):
            raise EvidenceError(
                f"{key}: duplicate sample {index} differs: "
                f"{previous['seconds']} versus {value}"
            )
        return
    bucket[index] = {
        "dataset": key[0],
        "function": key[1],
        "baseline": key[2],
        "metric": key[3],
        "sample_index": index,
        "seconds": value,
        "source": str(source),
        "source_sha256": source_sha256,
    }


def add_long_samples(
    samples: dict,
    path: Path,
    *,
    baselines: set[str] | None = None,
    replace_keys: set[tuple[str, str, str, str]] | None = None,
) -> None:
    path = path.resolve(strict=True)
    data = pd.read_csv(path, low_memory=False)
    required = {
        "dataset",
        "function",
        "baseline",
        "metric",
        "sample_index",
        "seconds",
        "status",
    }
    if not required.issubset(data.columns):
        raise EvidenceError(f"{path}: missing fields {sorted(required - set(data.columns))}")
    digest = sha256(path)
    data = data[
        data["status"].astype(str).str.lower().eq("ok")
        & data["metric"].isin(METRICS)
    ]
    if baselines is not None:
        data = data[data["baseline"].isin(baselines)]
    replaced: set[tuple[str, str, str, str]] = set()
    for row in data.itertuples(index=False):
        key = cell_metric_key(row.dataset, row.function, row.baseline, row.metric)
        replace = (
            replace_keys is not None
            and key in replace_keys
            and key not in replaced
        )
        add_sample(
            samples,
            dataset=row.dataset,
            function=row.function,
            baseline=row.baseline,
            metric=row.metric,
            sample_index=row.sample_index,
            seconds=row.seconds,
            source=path,
            source_sha256=digest,
            replace=replace,
        )
        if replace:
            replaced.add(key)


def add_cpu_closeness_samples(samples: dict, evidence_dir: Path) -> None:
    aggregate, raw_rows, _cells, _hashes = closeness_build.build_evidence(
        evidence_dir.resolve(strict=True)
    )
    del aggregate
    for row in raw_rows:
        source = Path(row["log"]).resolve(strict=True)
        digest = row["log_sha256"]
        for metric, field in (
            ("build", "construction_seconds"),
            ("kernel", "processing_seconds"),
            ("e2e", "e2e_seconds"),
        ):
            add_sample(
                samples,
                dataset=row["dataset"],
                function=row["function"],
                baseline=row["baseline"],
                metric=metric,
                sample_index=row["sample_index"],
                seconds=row[field],
                source=source,
                source_sha256=digest,
                replace=False,
            )


def add_cpu_large_samples(samples: dict, path: Path) -> None:
    path = path.resolve(strict=True)
    data = pd.read_csv(path, low_memory=False)
    digest = sha256(path)
    for row in data.itertuples(index=False):
        if str(row.status) != "ok" or str(row.baseline) not in CPU_BASELINES:
            continue
        try:
            records = json.loads(row.timing_process_records)
        except (TypeError, json.JSONDecodeError) as exc:
            raise EvidenceError(
                f"{row.dataset}/{row.function}/{row.baseline}: invalid timing_process_records"
            ) from exc
        for record in records:
            if record.get("status") != "ok":
                continue
            index = record.get("timing_process_index")
            for metric, field in (
                ("build", "graph_prepare_seconds"),
                ("kernel", "kernel_seconds"),
                ("e2e", "e2e_seconds"),
            ):
                add_sample(
                    samples,
                    dataset=row.dataset,
                    function=row.function,
                    baseline=row.baseline,
                    metric=metric,
                    sample_index=index,
                    seconds=record.get(field),
                    source=path,
                    source_sha256=digest,
                )


def add_nxcugraph_samples(samples: dict, path: Path) -> None:
    path = path.resolve(strict=True)
    data = pd.read_csv(path, low_memory=False)
    digest = sha256(path)
    required = {
        "dataset",
        "function",
        "baseline",
        "sample_index",
        "construction_seconds",
        "processing_seconds",
        "e2e_seconds",
        "status",
        "validation_status",
    }
    if not required.issubset(data.columns):
        raise EvidenceError(f"{path}: incomplete strict nx-cugraph sample schema")
    data = data[
        data["status"].astype(str).eq("ok")
        & data["validation_status"].astype(str).eq("pass")
    ]
    for row in data.itertuples(index=False):
        for metric, field in (
            ("build", "construction_seconds"),
            ("kernel", "processing_seconds"),
            ("e2e", "e2e_seconds"),
        ):
            add_sample(
                samples,
                dataset=row.dataset,
                function=row.function,
                baseline=row.baseline,
                metric=metric,
                sample_index=row.sample_index,
                seconds=getattr(row, field),
                source=path,
                source_sha256=digest,
            )


def add_eggpu_samples(samples: dict, ledger: pd.DataFrame) -> None:
    rows = ledger[
        ledger["baseline"].eq("EGGPU")
        & ledger["execution_status"].eq("ok")
    ]
    loaded_long_sources: set[Path] = set()
    for row in rows.itertuples(index=False):
        source = Path(row.timing_result_source).resolve(strict=True)
        long_path = source / "results_samples.csv"
        if long_path.is_file():
            if long_path not in loaded_long_sources:
                add_long_samples(samples, long_path, baselines={"EGGPU"})
                loaded_long_sources.add(long_path)
            continue

        # The two bulk-CSR scale anchors store one JSON document per fresh
        # process rather than the standard long-form sample table.
        for sample_index in sorted(EXPECTED_SAMPLES):
            path = (
                source
                / "raw"
                / f"{row.dataset}_{row.function}_timing_{sample_index}.json"
            ).resolve(strict=True)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "ok":
                raise EvidenceError(f"{path}: EGGPU anchor sample is not successful")
            e2e_samples = (payload.get("steady_e2e") or {}).get("samples") or []
            kernel_samples = (payload.get("steady_kernel") or {}).get("samples") or []
            if len(e2e_samples) != 1 or len(kernel_samples) != 1:
                raise EvidenceError(f"{path}: expected one steady call per process")
            digest = sha256(path)
            for metric, value in (
                ("build", payload.get("load_seconds")),
                ("kernel", kernel_samples[0]),
                ("e2e", e2e_samples[0]),
            ):
                add_sample(
                    samples,
                    dataset=row.dataset,
                    function=row.function,
                    baseline="EGGPU",
                    metric=metric,
                    sample_index=sample_index,
                    seconds=value,
                    source=path,
                    source_sha256=digest,
                )


def summarize(bucket: dict[int, dict]) -> dict[str, float]:
    if set(bucket) != EXPECTED_SAMPLES:
        raise EvidenceError(f"sample indices are {sorted(bucket)}, expected 1..5")
    values = [float(bucket[index]["seconds"]) for index in sorted(bucket)]
    mean = statistics.mean(values)
    median = statistics.median(values)
    sd = statistics.stdev(values)
    minimum = min(values)
    return {
        "minimum": minimum,
        "mean": mean,
        "sd": sd,
        "median": median,
        "maximum": max(values),
        "cv": sd / mean,
        "max_over_median": max(values) / median,
        "median_over_minimum": median / minimum,
    }


def update_ledger(
    ledger: pd.DataFrame,
    samples: dict,
    estimator_policy: str,
) -> tuple[pd.DataFrame, list[dict]]:
    output = ledger.copy()
    audit_rows: list[dict] = []
    for index, row in output.iterrows():
        if str(row.get("execution_status")) != "ok":
            continue
        for metric in METRICS:
            paper_field = f"{metric}_paper_seconds"
            if not finite(row.get(paper_field)):
                raise EvidenceError(
                    f"{row.dataset}/{row.function}/{row.baseline}: "
                    f"successful cell lacks {metric}"
                )
            key = cell_metric_key(
                row.dataset, row.function, row.baseline, metric
            )
            if key not in samples:
                raise EvidenceError(f"{key}: no raw sample evidence")
            summary = summarize(samples[key])
            old_paper = float(row[paper_field])
            old_estimator = str(row.get(f"{metric}_estimator", ""))
            if old_estimator.startswith("minimum") or old_estimator.startswith(
                "best_observed"
            ):
                if not close(old_paper, summary["minimum"]):
                    raise EvidenceError(
                        f"{key}: frozen minimum {old_paper} != raw {summary['minimum']}"
                    )
            else:
                expected_mean = row.get(f"{metric}_raw_mean_seconds")
                if not close(expected_mean, summary["mean"]):
                    raise EvidenceError(
                        f"{key}: frozen mean {expected_mean} != raw {summary['mean']}"
                    )
            use_minimum = (
                estimator_policy == UNIFORM_MINIMUM_POLICY
                or str(row.baseline) == "EGGPU"
            )
            displayed = (
                summary["minimum"] if use_minimum else summary["mean"]
            )
            estimator = (
                "minimum_of_five"
                if use_minimum
                else "arithmetic_mean_of_five"
            )
            output.at[index, paper_field] = displayed
            output.at[index, f"{metric}_raw_mean_seconds"] = summary["mean"]
            output.at[index, f"{metric}_std_seconds"] = summary["sd"]
            output.at[index, f"{metric}_estimator"] = estimator
            output.at[index, f"{metric}_raw_min_seconds"] = summary["minimum"]
            output.at[index, f"{metric}_raw_median_seconds"] = summary["median"]
            output.at[index, f"{metric}_raw_max_seconds"] = summary["maximum"]
            output.at[
                index, f"{metric}_coefficient_of_variation"
            ] = summary["cv"]
            output.at[index, f"{metric}_max_over_median"] = summary[
                "max_over_median"
            ]
            output.at[index, f"{metric}_median_over_minimum"] = summary[
                "median_over_minimum"
            ]
            output.at[index, f"{metric}_variance_policy"] = (
                "descriptive_only_no_posthoc_exclusion"
            )
            output.at[index, f"{metric}_stability_status"] = "reported"
            output.at[index, "sample_count"] = 5
            audit_rows.append(
                {
                    "dataset": row.dataset,
                    "function": row.function,
                    "baseline": row.baseline,
                    "metric": metric,
                    "old_paper_seconds": old_paper,
                    "old_estimator": old_estimator,
                    "new_paper_seconds": displayed,
                    "new_estimator": estimator,
                    "raw_mean_seconds": summary["mean"],
                    "raw_min_seconds": summary["minimum"],
                    "sample_sd_seconds": summary["sd"],
                    "raw_median_seconds": summary["median"],
                    "raw_max_seconds": summary["maximum"],
                    "sample_count": 5,
                    "action": (
                        f"unchanged_{estimator}"
                        if close(old_paper, displayed)
                        else f"changed_to_{estimator}"
                    ),
                }
            )
        if estimator_policy == MIXED_ESTIMATOR_POLICY:
            baseline = str(row.baseline)
            if baseline == "nx-cugraph":
                source = str(output.at[index, "result_source"])
                output.at[index, "result_source"] = source.replace(
                    "minimum of five",
                    "arithmetic mean of five fresh processes",
                )
            elif baseline == "Gunrock":
                reason = str(output.at[index, "reason"])
                output.at[index, "reason"] = reason.replace(
                    "minimum of five fresh processes",
                    "arithmetic mean of five fresh processes",
                )
    expected_metrics = int(output["execution_status"].eq("ok").sum()) * len(METRICS)
    if len(audit_rows) != expected_metrics:
        raise EvidenceError(
            f"updated {len(audit_rows)} metrics, expected {expected_metrics}"
        )
    return output, audit_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ledger", required=True, type=Path)
    parser.add_argument("--main-samples", required=True, type=Path)
    parser.add_argument("--closeness-evidence", required=True, type=Path)
    parser.add_argument("--graphscope-samples", required=True, type=Path)
    parser.add_argument("--cpu-large-matrix", required=True, type=Path)
    parser.add_argument("--nxcugraph-samples", required=True, type=Path)
    parser.add_argument("--gunrock-samples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--estimator-policy",
        choices=(UNIFORM_MINIMUM_POLICY, MIXED_ESTIMATOR_POLICY),
        default=UNIFORM_MINIMUM_POLICY,
        help=(
            "uniform-minimum preserves the prior release policy; "
            "eggpu-minimum-baseline-mean uses the final paper policy."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_ledger = args.base_ledger.resolve(strict=True)
    ledger = pd.read_csv(base_ledger, low_memory=False)
    samples: dict[tuple[str, str, str, str], dict[int, dict]] = defaultdict(dict)

    add_long_samples(
        samples, args.main_samples, baselines=CPU_BASELINES
    )
    add_cpu_closeness_samples(samples, args.closeness_evidence)
    add_cpu_large_samples(samples, args.cpu_large_matrix)
    add_long_samples(
        samples, args.graphscope_samples, baselines={"GraphScope"}
    )
    add_eggpu_samples(samples, ledger)
    add_nxcugraph_samples(samples, args.nxcugraph_samples)
    add_long_samples(samples, args.gunrock_samples, baselines={"Gunrock"})

    updated, audit_rows = update_ledger(
        ledger, samples, args.estimator_policy
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "final_13_cell_outcome_ledger.csv"
    mixed_policy = args.estimator_policy == MIXED_ESTIMATOR_POLICY
    raw_path = output / (
        "five_run_raw_samples.csv"
        if mixed_policy
        else "uniform_minimum_raw_samples.csv"
    )
    audit_path = output / (
        "mixed_estimator_audit.csv"
        if mixed_policy
        else "uniform_minimum_overlay_audit.csv"
    )
    updated.to_csv(ledger_path, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    needed_keys = {
        cell_metric_key(row.dataset, row.function, row.baseline, metric)
        for row in updated[updated["execution_status"].eq("ok")].itertuples()
        for metric in METRICS
    }
    raw_rows = [
        samples[key][index]
        for key in sorted(needed_keys)
        for index in sorted(samples[key])
    ]
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)

    changed = sum(
        not str(row["action"]).startswith("unchanged_")
        for row in audit_rows
    )
    unchanged = len(audit_rows) - changed
    if mixed_policy:
        schema_version = "eggpu_minimum_baseline_mean_five_run_v1"
        display_estimator = "mixed_by_baseline_class"
        manifest_name = "MIXED_ESTIMATOR_FIVE_RUN_MANIFEST.json"
        estimator_by_baseline_class = {
            "EGGPU": "minimum_of_five",
            "external_baselines": "arithmetic_mean_of_five",
        }
        selection_policy = (
            "All five correctness-qualified observations are retained. "
            "EGGPU displays the minimum of five and every external baseline "
            "displays the arithmetic mean of five. The sample standard "
            "deviation of the same five observations is the reported error; "
            "no observed-data threshold accepts or rejects a batch."
        )
    else:
        schema_version = "uniform_minimum_of_five_v1"
        display_estimator = "minimum_of_five"
        manifest_name = "UNIFORM_MINIMUM_OF_FIVE_MANIFEST.json"
        estimator_by_baseline_class = {
            "EGGPU": "minimum_of_five",
            "external_baselines": "minimum_of_five",
        }
        selection_policy = (
            "All five correctness-qualified observations are retained. "
            "The minimum is displayed uniformly; no batch is accepted or "
            "rejected using an observed-data stability threshold."
        )
    manifest = {
        "schema_version": schema_version,
        "status": "pass",
        "display_estimator": display_estimator,
        "display_estimator_by_baseline_class": estimator_by_baseline_class,
        "reported_error": "sample_standard_deviation_ddof1",
        "sample_count_per_displayed_metric": 5,
        "successful_cells": int(updated["execution_status"].eq("ok").sum()),
        "displayed_metrics": len(audit_rows),
        "portable_raw_rows": len(raw_rows),
        "changed_display_metrics": changed,
        "unchanged_display_metrics": unchanged,
        "selection_policy": selection_policy,
        "base_ledger": str(base_ledger),
        "base_ledger_sha256": sha256(base_ledger),
        "output_ledger": str(ledger_path),
        "output_ledger_sha256": sha256(ledger_path),
        "portable_raw_samples": str(raw_path),
        "portable_raw_samples_sha256": sha256(raw_path),
        "overlay_audit": str(audit_path),
        "overlay_audit_sha256": sha256(audit_path),
    }
    manifest_path = output / manifest_name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
