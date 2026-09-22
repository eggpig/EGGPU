#!/usr/bin/env python3
"""Build a reviewer-facing final-13 bundle with frozen per-system estimators.

The script never mutates raw experiments or archived paper assets.  It copies
the correctness-qualified outcome ledger without rewriting paper estimators,
regenerates numerical tables, and records byte-level provenance for all
inputs.  EGGPU uses the stability-gated minimum of five while retaining its
arithmetic mean, sample standard deviation, extrema, and CV.  External systems
retain the estimator frozen in the source ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import pandas as pd

import generate_final_13_complete_assets_graphscope as complete
import generate_chapter4_evaluation_assets_graphscope as chapter4_visuals
import update_final13_eggpu_uniform_20260729 as overlay_tools
from stable_timing_protocol import (
    DEFAULT_MAX_OVER_MEDIAN_LIMIT as EGGPU_MAX_OVER_MEDIAN_LIMIT,
    DEFAULT_MEDIAN_OVER_MIN_LIMIT as EGGPU_MEDIAN_OVER_MIN_LIMIT,
    FORMAL_BATCH_SELECTION_POLICY as EGGPU_BATCH_SELECTION_POLICY,
    PAPER_ESTIMATOR as EGGPU_TIMING_ESTIMATOR,
    VARIANCE_POLICY as EGGPU_VARIANCE_POLICY,
    StableTimingProtocolError,
    gate_calibration_payload,
    validate_gate_calibration,
)


METRICS = ("build", "kernel", "e2e")
# Backward-compatible read-only module alias for existing callers/tests.  New
# writes use gate_calibration_payload() so nested evidence cannot be mutated.
EGGPU_GATE_CALIBRATION = gate_calibration_payload()
MEMORY_COLUMNS = (
    "gpu_peak_mb_mean",
    "gpu_peak_mb_std",
    "memory_sample_count",
    "host_rss_peak_mb_mean",
    "host_rss_peak_mb_std",
    "memory_measurement_window",
)
LEGACY_HOT_HIT_PATCH_SCOPE = (
    "Capacity-certified single-entry cache hits bypass the device-memory "
    "budget query. Admission, growth, and multi-entry accesses retain the "
    "archived release behavior."
)
UNIFIED_HOT_HIT_PATCH_SCOPE = (
    "Capacity-certified exact cache hits allocate no memory and bypass device "
    "budget queries, including multi-entry registries. Admission and growth "
    "still enforce the configured budget and LRU eviction."
)
SUPPLEMENT_FILES = (
    "paper_baseline_versions.json",
    "paper_table_baseline_versions.csv",
    "paper_table_baseline_versions.tex",
    "FINAL_13_FAILURE_AND_COMPLETENESS_REPORT.md",
    "ALL_EXPERIMENT_NON_SUCCESS_REPORT.md",
    "all_experiment_non_success_ledger.csv",
    "all_experiment_non_success_summary.json",
    "scaling_four_function_points.csv",
    "scaling_four_functions.pdf",
    "scaling_four_functions.png",
    "intro_rmat_pagerank_uniform_mean.csv",
    "first_use_steady_actual_times.csv",
    "first_use_steady_function_actual_times.csv",
    "first_use_steady_pair_details.csv",
    "ablation_actual_values.csv",
    "ablation_nonpositive_mechanisms.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def make_uniform_ledger(source: Path, output: Path) -> pd.DataFrame:
    ledger = pd.read_csv(source, low_memory=False)
    audit_rows = []
    for metric in METRICS:
        paper = f"{metric}_paper_seconds"
        raw_mean = f"{metric}_raw_mean_seconds"
        estimator = f"{metric}_estimator"
        if paper not in ledger or raw_mean not in ledger:
            raise ValueError(f"ledger is missing {paper} or {raw_mean}")
        for _index, row in ledger.iterrows():
            if (
                str(row.get("execution_status")) != "ok"
                or not finite(row.get(paper))
            ):
                continue
            audit_rows.append(
                {
                    "dataset": row["dataset"],
                    "function": row["function"],
                    "baseline": row["baseline"],
                    "metric": metric,
                    "paper_value_seconds": row[paper],
                    "raw_mean_seconds": row.get(raw_mean),
                    "preserved_estimator": row.get(estimator),
                    "action": "preserved_source_estimator",
                }
            )
    ledger.to_csv(output / "final_13_cell_outcome_ledger.csv", index=False)
    pd.DataFrame(
        audit_rows,
        columns=[
            "dataset",
            "function",
            "baseline",
            "metric",
            "paper_value_seconds",
            "raw_mean_seconds",
            "preserved_estimator",
            "action",
        ],
    ).to_csv(
        output / "uniform_estimator_replacements.csv", index=False
    )
    return ledger


def regenerate_tables(ledger: pd.DataFrame, output: Path) -> dict[str, object]:
    complete.configure_baselines(ledger)
    complete.dataset_table(output)
    pairwise = pd.concat(
        [
            complete.pairwise_summary(ledger, "e2e"),
            complete.pairwise_summary(ledger, "kernel"),
        ],
        ignore_index=True,
    )
    pairwise.to_csv(output / "final_13_pairwise_sota_details.csv", index=False)
    pair_baseline = pd.concat(
        [
            complete.pairwise_baseline_summary(ledger, "e2e"),
            complete.pairwise_baseline_summary(ledger, "kernel"),
        ],
        ignore_index=True,
    )
    complete.write_build_dataset_table(ledger, output)
    complete.write_compact_speedup_table(
        pair_baseline[pair_baseline.metric.eq("e2e")], output
    )
    complete.write_compact_best_competitor_table(pairwise, output)
    complete.write_full_tables(ledger, "e2e", output)
    complete.write_full_tables(ledger, "kernel", output)
    category = complete.write_category_table(ledger, pairwise, output)
    return complete.write_numeric_report(
        ledger, pairwise, pair_baseline, category, output
    )


def write_historical_table(
    comparison: Path,
    output: Path,
    ledger: pd.DataFrame | None = None,
) -> dict[str, float]:
    data = pd.read_csv(comparison)
    required = {
        "function",
        "legacy_mean_seconds",
        "current_mean_seconds",
        "current_over_legacy_speedup",
        "legacy_validation_status",
        "current_validation_status",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"historical comparison is missing {sorted(missing)}")
    if ledger is not None:
        current = ledger[
            ledger["baseline"].eq("EGGPU")
            & ledger["execution_status"].eq("ok")
        ][
            [
                "dataset",
                "function",
                "e2e_paper_seconds",
                "e2e_raw_mean_seconds",
                "e2e_std_seconds",
                "e2e_estimator",
                "validation_status",
            ]
        ].copy()
        if current.duplicated(["dataset", "function"]).any():
            raise ValueError("candidate ledger has duplicate historical workload keys")
        current = current.rename(
            columns={
                "e2e_raw_mean_seconds": "_candidate_current_mean",
                "e2e_paper_seconds": "_candidate_current_paper",
                "e2e_std_seconds": "_candidate_current_std",
                "e2e_estimator": "_candidate_current_estimator",
                "validation_status": "_candidate_validation",
            }
        )
        data = data.merge(
            current,
            on=["dataset", "function"],
            how="left",
            validate="many_to_one",
        )
        if data["_candidate_current_mean"].isna().any():
            missing_keys = data.loc[
                data["_candidate_current_mean"].isna(), ["dataset", "function"]
            ].to_dict("records")
            raise ValueError(
                f"candidate ledger lacks historical comparison keys: {missing_keys}"
            )
        candidate_mean = pd.to_numeric(
            data["_candidate_current_mean"], errors="raise"
        )
        candidate_paper = pd.to_numeric(
            data["_candidate_current_paper"], errors="raise"
        )
        legacy_mean = pd.to_numeric(data["legacy_mean_seconds"], errors="raise")
        data["current_mean_seconds"] = candidate_mean
        data["current_min_seconds"] = candidate_paper
        data["current_paper_seconds"] = candidate_paper
        data["current_raw_mean_seconds"] = candidate_mean
        data["current_sample_std_seconds"] = pd.to_numeric(
            data["_candidate_current_std"], errors="raise"
        )
        data["current_over_legacy_speedup"] = legacy_mean / candidate_paper
        data["current_estimator"] = data["_candidate_current_estimator"]
        data["current_validation_status"] = data["_candidate_validation"].replace(
            {"sampled_pass": "pass"}
        )
        data = data.drop(
            columns=[
                "_candidate_current_mean",
                "_candidate_current_paper",
                "_candidate_current_std",
                "_candidate_current_estimator",
                "_candidate_validation",
            ]
        )
    valid = data[
        data["legacy_validation_status"].eq("pass")
        & data["current_validation_status"].eq("pass")
    ].copy()
    valid.to_csv(output / "historical_eggpu_pair_details.csv", index=False)

    def geomean(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce")
        values = values[values.notna() & values.gt(0)]
        return float(values.prod() ** (1.0 / len(values)))

    rows = []
    for function in ("BC", "SSSP", "KCore"):
        part = valid[valid["function"].eq(function)]
        rows.append(
            {
                "function": function,
                "datasets": len(part),
                "legacy_geomean_seconds": geomean(part["legacy_mean_seconds"]),
                "current_geomean_seconds": geomean(part["current_paper_seconds"]),
                "speedup": geomean(part["current_over_legacy_speedup"]),
            }
        )
    rows.append(
        {
            "function": "Overall",
            "datasets": len(valid),
            "legacy_geomean_seconds": geomean(valid["legacy_mean_seconds"]),
            "current_geomean_seconds": geomean(valid["current_paper_seconds"]),
            "speedup": geomean(valid["current_over_legacy_speedup"]),
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "historical_eggpu_summary.csv", index=False)
    lines = [
        r"% Generated by prepare_vldb_uniform_assets.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Evolution from the 2024 EGGPU prototype to the current system on correctness-aligned common workloads. The prototype contains BC, SSSP, and KCore only.}",
        r"\label{tab:historical-eggpu}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Function & Pairs & EGGPU-2024 (s) & Current (s) & Speedup \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.function} & {row.datasets} & {row.legacy_geomean_seconds:.4f} & "
            f"{row.current_geomean_seconds:.4f} & {row.speedup:.2f}$\\times$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output / "paper_table_historical_eggpu.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    overall = summary[summary["function"].eq("Overall")].iloc[0]
    return {
        "validated_pairs": int(len(valid)),
        "overall_speedup": float(overall["speedup"]),
    }


def write_gap_timing_table(
    ledger: pd.DataFrame,
    output: Path,
    expected_sha256: str,
    release_label: str,
) -> dict[str, object]:
    """Emit the 16-cell GAP-twitter timing table and a machine-readable gate."""

    gap = ledger[
        ledger.dataset.eq("GAP-twitter") & ledger.baseline.eq("EGGPU")
    ].copy()
    order = {name: index for index, name in enumerate(complete.base.FUNCTION_ORDER)}
    gap["_order"] = gap.function.map(order)
    gap = gap.sort_values("_order").drop(columns="_order")
    if len(gap) != 16 or gap.function.nunique() != 16:
        raise ValueError(
            f"GAP-twitter {release_label} table does not have 16 unique functions"
        )
    if not gap.execution_status.eq("ok").all():
        raise ValueError(f"GAP-twitter {release_label} table contains a non-OK cell")
    if not gap.validation_status.isin({"pass", "sampled_pass"}).all():
        raise ValueError(
            f"GAP-twitter {release_label} table contains an unvalidated cell"
        )
    if not pd.to_numeric(gap.sample_count, errors="coerce").eq(5).all():
        raise ValueError(
            f"GAP-twitter {release_label} table does not have five samples per cell"
        )
    if not gap.candidate_sha256.astype(str).eq(expected_sha256).all():
        raise ValueError(
            f"GAP-twitter {release_label} table is not uniformly SHA-bound"
        )
    for metric in METRICS:
        if not gap[f"{metric}_estimator"].astype(str).eq(
            EGGPU_TIMING_ESTIMATOR
        ).all():
            raise ValueError(
                f"GAP-twitter {metric} estimator is not uniformly "
                f"{EGGPU_TIMING_ESTIMATOR}"
            )
    if not gap["e2e_stability_status"].astype(str).eq("pass").all():
        raise ValueError(
            f"GAP-twitter {release_label} table contains an unstable E2E batch"
        )

    columns = [
        "function",
        "category",
        "build_paper_seconds",
        "build_raw_mean_seconds",
        "build_std_seconds",
        "build_coefficient_of_variation",
        "kernel_paper_seconds",
        "kernel_raw_mean_seconds",
        "kernel_std_seconds",
        "kernel_coefficient_of_variation",
        "e2e_paper_seconds",
        "e2e_raw_mean_seconds",
        "e2e_std_seconds",
        "e2e_coefficient_of_variation",
        "e2e_max_over_median",
        "e2e_median_over_minimum",
        "e2e_stability_status",
        "sample_count",
        "validation_status",
        "candidate_sha256",
        "result_source",
    ]
    table = gap[columns].rename(
        columns={
            "kernel_paper_seconds": "processing_min_seconds",
            "kernel_raw_mean_seconds": "processing_mean_seconds",
            "kernel_std_seconds": "processing_sample_std_seconds",
            "kernel_coefficient_of_variation": "processing_cv",
            "e2e_paper_seconds": "public_return_min_seconds",
            "e2e_raw_mean_seconds": "public_return_mean_seconds",
            "e2e_std_seconds": "public_return_sample_std_seconds",
            "e2e_coefficient_of_variation": "public_return_cv",
            "build_paper_seconds": "build_min_seconds",
            "build_raw_mean_seconds": "build_mean_seconds",
            "build_std_seconds": "build_sample_std_seconds",
            "build_coefficient_of_variation": "build_cv",
        }
    )
    release_slug = re.sub(r"[^a-z0-9]+", "_", release_label.lower()).strip("_")
    if not release_slug:
        raise ValueError("release label does not contain a usable file-name token")
    csv_path = output / f"gap_twitter_{release_slug}_timing_table.csv"
    table.to_csv(csv_path, index=False)

    lines = [
        r"% Requires: booktabs",
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{EGGPU on GAP-twitter under the "
            + release_label
            + r" timing protocol. Every entry reports the minimum of five "
            r"fresh processes with the same batch's sample standard deviation; "
            r"the arithmetic mean and CV remain in the evidence CSV. Each "
            r"process measures the third call after a first-use call and one "
            r"additional untimed warmup. Public return is the reused-state "
            r"in-process function-call boundary.}"
        ),
        rf"\label{{tab:gap-twitter-{release_slug}}}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Function & \multicolumn{2}{c}{Build (s)} & \multicolumn{2}{c}{Processing (s)} & \multicolumn{2}{c}{Public return (s)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & Min & SD & Min & SD & Min & SD \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{complete.latex(row.function)} & "
            f"{float(row.build_min_seconds):.4f} & "
            f"{float(row.build_sample_std_seconds):.4f} & "
            f"{float(row.processing_min_seconds):.4f} & "
            f"{float(row.processing_sample_std_seconds):.4f} & "
            f"{float(row.public_return_min_seconds):.4f} & "
            f"{float(row.public_return_sample_std_seconds):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    tex_path = output / f"paper_table_gap_twitter_{release_slug}_timing.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "status": "pass",
        "release_label": release_label,
        "dataset": "GAP-twitter",
        "function_cells": 16,
        "sample_count_per_cell": 5,
        "timing_estimator": EGGPU_TIMING_ESTIMATOR,
        "error_term": "sample_standard_deviation",
        "variance_policy": EGGPU_VARIANCE_POLICY,
        "max_over_median_limit": EGGPU_MAX_OVER_MEDIAN_LIMIT,
        "median_over_min_limit": EGGPU_MEDIAN_OVER_MIN_LIMIT,
        "measured_call_ordinal": 3,
        "candidate_binary_sha256": expected_sha256,
        "public_return_boundary": "reused-state in-process public function call",
        "csv_sha256": sha256(csv_path),
        "tex_sha256": sha256(tex_path),
    }
    json_path = output / f"gap_twitter_{release_slug}_timing_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["json_sha256"] = sha256(json_path)
    return summary


def write_igraph_crossover(
    ledger: pd.DataFrame, output: Path
) -> dict[str, object]:
    """Write common-pair EGGPU/igraph crossovers by CSR scale and function."""

    validation_ok = {
        "pass",
        "sampled_pass",
        "reference",
        "external_reference_pass",
    }
    buckets = (
        ("<1e5", 0, 100_000),
        ("1e5--1e6", 100_000, 1_000_000),
        (">=1e6", 1_000_000, math.inf),
    )
    metric_columns = {
        "e2e_public_call": "e2e_paper_seconds",
        "processing": "kernel_paper_seconds",
    }
    boundary = {
        "e2e_public_call": (
            "EGGPU reused-state public call versus igraph in-memory public call"
        ),
        "processing": (
            "EGGPU device processing versus igraph in-memory algorithm wall-time "
            "surrogate"
        ),
    }
    cell_rows = []
    for dataset in complete.story.FINAL_13:
        structure = complete.story.DATASET_STRUCTURE[dataset]
        entries = int(structure["csr_entries"])
        size_bucket = next(
            name for name, lower, upper in buckets if lower <= entries < upper
        )
        for function in complete.base.FUNCTION_ORDER:
            pair = ledger[
                ledger["dataset"].eq(dataset)
                & ledger["function"].eq(function)
                & ledger["baseline"].isin({"EGGPU", "igraph"})
            ]
            for metric, column in metric_columns.items():
                values = {}
                for system in ("EGGPU", "igraph"):
                    row = pair[pair["baseline"].eq(system)]
                    if len(row) != 1:
                        continue
                    item = row.iloc[0]
                    value = pd.to_numeric(
                        pd.Series([item.get(column)]), errors="coerce"
                    ).iloc[0]
                    if (
                        item.get("execution_status") != "ok"
                        or item.get("validation_status") not in validation_ok
                        or not math.isfinite(value)
                        or value <= 0
                    ):
                        continue
                    values[system] = float(value)
                if set(values) != {"EGGPU", "igraph"}:
                    continue
                speedup = values["igraph"] / values["EGGPU"]
                tolerance = complete.RELATIVE_TIE_TOLERANCE * max(
                    abs(values["igraph"]), abs(values["EGGPU"]), 1.0e-15
                )
                delta = values["EGGPU"] - values["igraph"]
                outcome = (
                    "strict_win"
                    if delta < -tolerance
                    else ("tie" if abs(delta) <= tolerance else "loss")
                )
                cell_rows.append(
                    {
                        "metric": metric,
                        "timing_boundary": boundary[metric],
                        "size_bucket": size_bucket,
                        "dataset": dataset,
                        "nodes": int(structure["nodes"]),
                        "csr_entries": entries,
                        "function": function,
                        "eggpu_seconds": values["EGGPU"],
                        "igraph_seconds": values["igraph"],
                        "igraph_over_eggpu_speedup": speedup,
                        "outcome": outcome,
                    }
                )
    cells = pd.DataFrame(cell_rows)
    cells.to_csv(output / "eggpu_vs_igraph_crossover_cells.csv", index=False)

    aggregate_rows = []
    functions = (*complete.base.FUNCTION_ORDER, "ALL")
    bucket_names = tuple(name for name, _lower, _upper in buckets)
    for metric in metric_columns:
        metric_cells = cells[cells["metric"].eq(metric)]
        for size_bucket in (*bucket_names, "ALL"):
            scaled = (
                metric_cells
                if size_bucket == "ALL"
                else metric_cells[metric_cells["size_bucket"].eq(size_bucket)]
            )
            for function in functions:
                part = (
                    scaled
                    if function == "ALL"
                    else scaled[scaled["function"].eq(function)]
                )
                speeds = pd.to_numeric(
                    part["igraph_over_eggpu_speedup"], errors="coerce"
                )
                speeds = speeds[speeds.notna() & speeds.gt(0)]
                aggregate_rows.append(
                    {
                        "metric": metric,
                        "timing_boundary": boundary[metric],
                        "size_bucket": size_bucket,
                        "function": function,
                        "common_pairs": len(part),
                        "strict_wins": int(part["outcome"].eq("strict_win").sum()),
                        "ties": int(part["outcome"].eq("tie").sum()),
                        "losses": int(part["outcome"].eq("loss").sum()),
                        "igraph_over_eggpu_geomean": (
                            float(
                                math.exp(
                                    sum(math.log(value) for value in speeds)
                                    / len(speeds)
                                )
                            )
                            if len(speeds)
                            else math.nan
                        ),
                        "crossover_by_geomean": (
                            bool(len(speeds) and math.exp(
                                sum(math.log(value) for value in speeds)
                                / len(speeds)
                            ) > 1.0)
                        ),
                    }
                )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        output / "eggpu_vs_igraph_crossover_by_scale_function.csv",
        index=False,
    )

    summary: dict[str, object] = {
        "speedup_definition": "igraph_seconds / eggpu_seconds",
        "relative_tie_tolerance": complete.RELATIVE_TIE_TOLERANCE,
        "tie_definition": (
            "absolute timing difference <= 0.05% of the larger pair timing"
        ),
        "size_bucket_definition": (
            "normalized CSR adjacency entries: <1e5, [1e5,1e6), >=1e6"
        ),
        "sole_validated_cells_included": False,
        "metrics": {},
    }
    for metric in metric_columns:
        metric_summary = {}
        for size_bucket in (*bucket_names, "ALL"):
            row = aggregate[
                aggregate["metric"].eq(metric)
                & aggregate["size_bucket"].eq(size_bucket)
                & aggregate["function"].eq("ALL")
            ].iloc[0]
            metric_summary[size_bucket] = {
                "common_pairs": int(row["common_pairs"]),
                "strict_wins": int(row["strict_wins"]),
                "ties": int(row["ties"]),
                "losses": int(row["losses"]),
                "igraph_over_eggpu_geomean": (
                    float(row["igraph_over_eggpu_geomean"])
                    if math.isfinite(row["igraph_over_eggpu_geomean"])
                    else None
                ),
            }
        summary["metrics"][metric] = metric_summary
    (output / "eggpu_vs_igraph_crossover_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def copy_supplements(source: Path, output: Path) -> list[str]:
    copied = []
    for name in SUPPLEMENT_FILES:
        path = source / name
        if not path.is_file():
            continue
        shutil.copy2(path, output / name)
        copied.append(name)
    return copied


def normalize_candidate_sha256(value: str) -> str:
    candidate = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", candidate):
        raise ValueError(f"invalid EGGPU candidate SHA-256: {value!r}")
    return candidate


def audit_candidate_ledger(
    ledger: pd.DataFrame,
    expected_sha256: str,
    *,
    require_runtime_provenance: bool = False,
) -> dict[str, object]:
    """Require one uniform, SHA-bound EGGPU timing cell per workload."""

    required = {
        "dataset",
        "function",
        "baseline",
        "execution_status",
        "validation_status",
        "sample_count",
        "candidate_sha256",
    }
    for metric in METRICS:
        required.update(
            {
                f"{metric}_paper_seconds",
                f"{metric}_raw_min_seconds",
                f"{metric}_raw_median_seconds",
                f"{metric}_raw_mean_seconds",
                f"{metric}_std_seconds",
                f"{metric}_raw_max_seconds",
                f"{metric}_coefficient_of_variation",
                f"{metric}_max_over_median",
                f"{metric}_median_over_minimum",
                f"{metric}_variance_policy",
                f"{metric}_stability_status",
                f"{metric}_estimator",
            }
        )
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(
            f"candidate ledger is missing required fields: {sorted(missing)}"
        )

    eggpu = ledger[ledger["baseline"].eq("EGGPU")].copy()
    if len(eggpu) != 208:
        raise ValueError(f"candidate ledger has {len(eggpu)} EGGPU cells, expected 208")
    if eggpu.duplicated(["dataset", "function"]).any():
        raise ValueError("candidate ledger has duplicate EGGPU dataset/function cells")
    if not eggpu["execution_status"].eq("ok").all():
        raise ValueError("candidate ledger contains a non-OK EGGPU cell")
    if not eggpu["validation_status"].isin({"pass", "sampled_pass"}).all():
        raise ValueError("candidate ledger contains an unvalidated EGGPU cell")
    samples = pd.to_numeric(eggpu["sample_count"], errors="coerce")
    if not samples.eq(5).all():
        raise ValueError("candidate ledger does not use five samples for every EGGPU cell")
    observed_sha = sorted(
        {
            str(value).strip().lower()
            for value in eggpu["candidate_sha256"]
            if str(value).strip()
        }
    )
    if observed_sha != [expected_sha256]:
        raise ValueError(
            f"candidate ledger SHA values are {observed_sha}, expected {[expected_sha256]}"
        )
    runtime_sha = ""
    if require_runtime_provenance:
        runtime_column = "runtime_python_snapshot_sha256"
        if runtime_column not in eggpu:
            raise ValueError(
                f"candidate ledger is missing required field: {runtime_column}"
            )
        observed_runtime = sorted(
            {
                normalize_candidate_sha256(value)
                for value in eggpu[runtime_column]
                if str(value).strip()
            }
        )
        if len(observed_runtime) != 1:
            raise ValueError(
                "candidate ledger does not use one frozen Python runtime: "
                f"{observed_runtime}"
            )
        runtime_sha = observed_runtime[0]

    for metric in METRICS:
        paper = pd.to_numeric(
            eggpu[f"{metric}_paper_seconds"], errors="coerce"
        )
        raw_min = pd.to_numeric(
            eggpu[f"{metric}_raw_min_seconds"], errors="coerce"
        )
        raw_median = pd.to_numeric(
            eggpu[f"{metric}_raw_median_seconds"], errors="coerce"
        )
        raw_mean = pd.to_numeric(
            eggpu[f"{metric}_raw_mean_seconds"], errors="coerce"
        )
        stdev = pd.to_numeric(eggpu[f"{metric}_std_seconds"], errors="coerce")
        raw_max = pd.to_numeric(
            eggpu[f"{metric}_raw_max_seconds"], errors="coerce"
        )
        cv = pd.to_numeric(
            eggpu[f"{metric}_coefficient_of_variation"], errors="coerce"
        )
        max_over_median = pd.to_numeric(
            eggpu[f"{metric}_max_over_median"], errors="coerce"
        )
        median_over_minimum = pd.to_numeric(
            eggpu[f"{metric}_median_over_minimum"], errors="coerce"
        )
        estimator = eggpu[f"{metric}_estimator"].astype(str)
        if not estimator.eq(EGGPU_TIMING_ESTIMATOR).all():
            raise ValueError(
                f"{metric}: a candidate estimator is not "
                f"{EGGPU_TIMING_ESTIMATOR}"
            )
        numeric_columns = {
            "paper minimum": paper,
            "raw minimum": raw_min,
            "raw median": raw_median,
            "raw mean": raw_mean,
            "raw maximum": raw_max,
            "CV": cv,
            "max/median": max_over_median,
            "median/min": median_over_minimum,
        }
        for label, values in numeric_columns.items():
            if not values.map(math.isfinite).all():
                raise ValueError(
                    f"{metric}: a candidate {label} is non-finite"
                )
        if not stdev.map(math.isfinite).all() or not stdev.ge(0).all():
            raise ValueError(f"{metric}: a candidate sample deviation is invalid")
        if not (
            raw_min.le(raw_median)
            & raw_median.le(raw_max)
            & raw_min.le(raw_mean)
            & raw_mean.le(raw_max)
        ).all():
            raise ValueError(f"{metric}: raw timing order statistics are invalid")
        aligned = [
            math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15)
            for left, right in zip(paper, raw_min)
        ]
        if not all(aligned):
            raise ValueError(
                f"{metric}: paper timing differs from its five-run minimum"
            )
        expected_cv = [
            deviation / mean if mean > 0 else 0.0
            for deviation, mean in zip(stdev, raw_mean)
        ]
        if not all(
            math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-15)
            for left, right in zip(cv, expected_cv)
        ):
            raise ValueError(f"{metric}: candidate CV does not match mean/SD")
        expected_max_over_median = [
            maximum / median if median > 0 else 1.0
            for maximum, median in zip(raw_max, raw_median)
        ]
        expected_median_over_minimum = [
            median / minimum if minimum > 0 else 1.0
            for median, minimum in zip(raw_median, raw_min)
        ]
        if not all(
            math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-15)
            for left, right in zip(
                max_over_median, expected_max_over_median
            )
        ):
            raise ValueError(f"{metric}: max/median ratio is inconsistent")
        if not all(
            math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-15)
            for left, right in zip(
                median_over_minimum, expected_median_over_minimum
            )
        ):
            raise ValueError(f"{metric}: median/min ratio is inconsistent")
        if not eggpu[f"{metric}_variance_policy"].astype(str).eq(
            EGGPU_VARIANCE_POLICY
        ).all():
            raise ValueError(f"{metric}: variance policy differs")
        if not eggpu[f"{metric}_stability_status"].astype(str).isin(
            {"pass", "fail"}
        ).all():
            raise ValueError(f"{metric}: stability status is invalid")
        if metric == "e2e":
            if not eggpu[f"{metric}_stability_status"].astype(str).eq(
                "pass"
            ).all():
                raise ValueError("e2e: a candidate batch failed stability")
            if not max_over_median.le(EGGPU_MAX_OVER_MEDIAN_LIMIT).all():
                raise ValueError("e2e: max/median exceeds the stability limit")
            if not median_over_minimum.le(
                EGGPU_MEDIAN_OVER_MIN_LIMIT
            ).all():
                raise ValueError("e2e: median/min exceeds the stability limit")

    graphscope_cells = int(ledger["baseline"].eq("GraphScope").sum())
    if graphscope_cells != 208:
        raise ValueError(
            f"candidate ledger has {graphscope_cells} GraphScope cells, expected 208"
        )
    return {
        "scope": "final-13 EGGPU timing ledger",
        "eggpu_cells": len(eggpu),
        "unique_eggpu_workloads": int(
            eggpu[["dataset", "function"]].drop_duplicates().shape[0]
        ),
        "sample_count_per_cell": 5,
        "timing_estimator": EGGPU_TIMING_ESTIMATOR,
        "variance_policy": EGGPU_VARIANCE_POLICY,
        "max_over_median_limit": EGGPU_MAX_OVER_MEDIAN_LIMIT,
        "median_over_min_limit": EGGPU_MEDIAN_OVER_MIN_LIMIT,
        "candidate_sha256": expected_sha256,
        "runtime_python_snapshot_sha256": runtime_sha or None,
        "graphscope_cells": graphscope_cells,
    }


def audit_frozen_memory_provenance(
    ledger: pd.DataFrame, supplements: Path
) -> dict[str, object]:
    """Prove that legacy mode retained, rather than reattributed, memory rows."""

    archived_path = supplements / "final_13_cell_outcome_ledger.csv"
    versions_path = supplements / "paper_baseline_versions.json"
    if not archived_path.is_file() or not versions_path.is_file():
        raise FileNotFoundError(
            "the supplement bundle lacks its ledger or baseline provenance"
        )
    archived = pd.read_csv(archived_path, low_memory=False)
    keys = ["dataset", "function", "baseline"]
    missing = set((*keys, *MEMORY_COLUMNS)) - set(archived.columns)
    missing.update(set((*keys, *MEMORY_COLUMNS)) - set(ledger.columns))
    if missing:
        raise ValueError(
            f"memory provenance audit is missing fields: {sorted(missing)}"
        )
    current = ledger[ledger.baseline.eq("EGGPU")][
        [*keys, *MEMORY_COLUMNS]
    ].sort_values(keys).reset_index(drop=True)
    frozen = archived[archived.baseline.eq("EGGPU")][
        [*keys, *MEMORY_COLUMNS]
    ].sort_values(keys).reset_index(drop=True)
    if len(current) != 208 or len(frozen) != 208:
        raise ValueError(
            f"memory provenance expects 208 EGGPU rows, got {len(current)} and "
            f"{len(frozen)}"
        )
    if not current[keys].equals(frozen[keys]):
        raise ValueError(
            "candidate and archived memory ledgers have different workload keys"
        )
    changed = []
    for column in MEMORY_COLUMNS:
        left = current[column]
        right = frozen[column]
        equal = left.eq(right) | (left.isna() & right.isna())
        if not bool(equal.all()):
            changed.append(column)
    if changed:
        raise ValueError(
            f"legacy mode changed frozen EGGPU memory columns: {sorted(changed)}"
        )

    versions = json.loads(versions_path.read_text(encoding="utf-8"))
    archived_eggpu = [
        item
        for item in versions.get("systems", [])
        if str(item.get("system")) == "EGGPU"
    ]
    archived_source_digest = (
        str(archived_eggpu[0].get("commit", "")).strip()
        if len(archived_eggpu) == 1
        else ""
    )
    if not re.fullmatch(r"[0-9a-fA-F]{64}", archived_source_digest):
        archived_source_digest = "indeterminate"

    source_counts = (
        archived[archived.baseline.eq("EGGPU")]["result_source"]
        .fillna("<missing>")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    v7_path = (
        supplements.parent
        / "EGGPU_FINAL_EXPERIMENT_ASSETS_13_V7_FINAL_20260728"
        / "final_13_cell_outcome_ledger.csv"
    )
    v7_lineage: dict[str, object] = {
        "status": "indeterminate",
        "reason": "the archived V7 ledger is unavailable",
    }
    if v7_path.is_file():
        v7 = pd.read_csv(v7_path, low_memory=False)
        v7_rows = v7[v7.baseline.eq("EGGPU")][
            [*keys, *MEMORY_COLUMNS]
        ].sort_values(keys).reset_index(drop=True)
        if len(v7_rows) != 208 or not v7_rows[keys].equals(frozen[keys]):
            raise ValueError("the archived V7 memory ledger has a different key set")
        mismatched = []
        for column in MEMORY_COLUMNS:
            left = frozen[column]
            right = v7_rows[column]
            if pd.api.types.is_numeric_dtype(left):
                equal = [
                    (
                        pd.isna(a)
                        and pd.isna(b)
                    )
                    or (
                        not pd.isna(a)
                        and not pd.isna(b)
                        and math.isclose(
                            float(a),
                            float(b),
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-12,
                        )
                    )
                    for a, b in zip(left, right)
                ]
                aligned = all(equal)
            else:
                aligned = bool(
                    (left.eq(right) | (left.isna() & right.isna())).all()
                )
            if not aligned:
                mismatched.append(column)
        if mismatched:
            raise ValueError(
                f"V8 and V7 memory lineage differs in {sorted(mismatched)}"
            )
        v7_lineage = {
            "status": "pass_numeric_equivalence",
            "source_ledger": str(v7_path.resolve()),
            "source_ledger_sha256": sha256(v7_path),
            "preserved_rows": 208,
            "comparison_tolerance": "1e-12 relative/absolute for numeric CSV fields",
        }
    return {
        "status": "pass_frozen_rows_byte_value_aligned",
        "scope": "208 EGGPU memory rows; timing fields are outside this binding",
        "row_count": 208,
        "preserved_columns": list(MEMORY_COLUMNS),
        "immediate_source_bundle": str(supplements.resolve()),
        "immediate_source_ledger": str(archived_path.resolve()),
        "immediate_source_ledger_sha256": sha256(archived_path),
        "archived_paper_source_snapshot_sha256": archived_source_digest,
        "archived_digest_kind": (
            "source-tree snapshot digest recorded by the archived V8 baseline "
            "metadata; not a compiled-binary digest"
            if archived_source_digest != "indeterminate"
            else "indeterminate"
        ),
        "row_level_compiled_binary_sha256": "indeterminate",
        "measured_with_timing_candidate": False,
        "archived_v7_lineage": v7_lineage,
        "source_label_counts": {
            str(key): int(value) for key, value in source_counts.items()
        },
        "lineage_note": (
            "The V8 uniform ledger is the immediate source, and its 208 EGGPU "
            "memory rows are numerically equivalent to the archived V7 ledger "
            "when that ledger is available. Row labels reference archived "
            "latest-v5 and July 27--28 measurements; those records do not carry "
            "a row-level compiled-binary SHA-256. The timing candidate SHA must "
            "therefore not be attributed to these memory rows."
        ),
    }


def audit_unified_memory_provenance(
    ledger: pd.DataFrame,
    source_ledger: Path,
    overlay_audit: Path,
    expected_sha256: str,
    expected_runtime_sha256: str,
) -> dict[str, object]:
    """Bind all 208 memory cells to the same candidate and frozen runtime."""

    keys = ["dataset", "function", "baseline"]
    provenance_columns = (
        "memory_result_source",
        "memory_candidate_sha256",
        "memory_runtime_python_snapshot_sha256",
    )
    missing = set((*keys, *MEMORY_COLUMNS, *provenance_columns)) - set(
        ledger.columns
    )
    if missing:
        raise ValueError(
            f"unified memory audit is missing fields: {sorted(missing)}"
        )
    eggpu = ledger[ledger.baseline.eq("EGGPU")].copy()
    if len(eggpu) != 208 or eggpu.duplicated(["dataset", "function"]).any():
        raise ValueError(
            "unified memory audit requires 208 unique EGGPU workload rows"
        )
    counts = pd.to_numeric(eggpu["memory_sample_count"], errors="coerce")
    if not counts.eq(3).all():
        raise ValueError(
            "unified memory audit requires three samples for every EGGPU cell"
        )
    for mean_column in ("gpu_peak_mb_mean", "host_rss_peak_mb_mean"):
        values = pd.to_numeric(eggpu[mean_column], errors="coerce")
        if not values.map(math.isfinite).all() or not values.gt(0).all():
            raise ValueError(
                f"unified memory audit has an invalid {mean_column}"
            )
    for std_column in ("gpu_peak_mb_std", "host_rss_peak_mb_std"):
        values = pd.to_numeric(eggpu[std_column], errors="coerce")
        if not values.map(math.isfinite).all() or not values.ge(0).all():
            raise ValueError(
                f"unified memory audit has an invalid {std_column}"
            )
    valid_windows = {
        "isolated_memory_subprocess",
        "isolated_worker_process_full_lifetime",
    }
    observed_windows = set(eggpu["memory_measurement_window"].astype(str))
    if not observed_windows or not observed_windows.issubset(valid_windows):
        raise ValueError(
            "unified memory audit contains an invalid measurement window: "
            f"{sorted(observed_windows)}"
        )
    anchor_datasets = {"com-Orkut", "GAP-twitter"}
    regular_window_ok = eggpu.loc[
        ~eggpu["dataset"].isin(anchor_datasets), "memory_measurement_window"
    ].astype(str).eq("isolated_memory_subprocess")
    anchor_window_ok = eggpu.loc[
        eggpu["dataset"].isin(anchor_datasets), "memory_measurement_window"
    ].astype(str).eq("isolated_worker_process_full_lifetime")
    if not regular_window_ok.all() or not anchor_window_ok.all():
        raise ValueError(
            "unified memory rows do not preserve regular versus anchor "
            "measurement-window provenance"
        )
    if not eggpu["memory_result_source"].astype(str).str.strip().ne("").all():
        raise ValueError("a unified memory cell lacks its result source")
    if not eggpu["memory_candidate_sha256"].astype(str).eq(
        expected_sha256
    ).all():
        raise ValueError("a unified memory cell uses a different candidate SHA")
    if not eggpu["memory_runtime_python_snapshot_sha256"].astype(str).eq(
        expected_runtime_sha256
    ).all():
        raise ValueError(
            "a unified memory cell uses a different Python runtime snapshot"
        )

    payload = json.loads(overlay_audit.read_text(encoding="utf-8"))
    if payload.get("candidate_mode") != "unified-timing-memory":
        raise ValueError("overlay audit is not a unified timing-memory run")
    if int(payload.get("memory_replacement_count", -1)) != 208:
        raise ValueError("overlay audit does not contain 208 memory replacements")
    if int(payload.get("expected_memory_replacements", -1)) != 208:
        raise ValueError("overlay audit did not require 208 memory replacements")
    if payload.get("candidate_sha256") != [expected_sha256]:
        raise ValueError("overlay memory candidate SHA does not match")
    if payload.get("runtime_python_snapshot_sha256") != [
        expected_runtime_sha256
    ]:
        raise ValueError("overlay memory runtime snapshot does not match")
    if payload.get("output_ledger_sha256") != sha256(source_ledger):
        raise ValueError("overlay audit output hash does not match the source ledger")
    if payload.get("rejected_candidates"):
        raise ValueError("overlay audit contains an active rejected candidate")

    cells_path = overlay_audit.with_suffix(".memory_cells.csv")
    cells = pd.read_csv(cells_path, keep_default_na=False)
    if len(cells) != 208 or cells.duplicated(["dataset", "function"]).any():
        raise ValueError(
            "memory overlay comparison does not contain 208 unique workloads"
        )
    if set(cells["candidate_sha256"].astype(str)) != {expected_sha256}:
        raise ValueError("memory overlay comparison uses a different candidate SHA")
    if set(cells["runtime_python_snapshot_sha256"].astype(str)) != {
        expected_runtime_sha256
    }:
        raise ValueError(
            "memory overlay comparison uses a different Python runtime snapshot"
        )

    memory_sources = [
        source
        for source in payload.get("sources", [])
        if source.get("measurement_kind") == "memory"
    ]
    if not memory_sources:
        raise ValueError("overlay audit contains no memory sources")
    for source in memory_sources:
        if source.get("candidate_sha256") != expected_sha256:
            raise ValueError("a memory source uses a different candidate SHA")
        if source.get("runtime_python_sha256") != expected_runtime_sha256:
            raise ValueError("a memory source uses a different runtime snapshot")
        if source.get("runtime_package_is_symlink") is not False:
            raise ValueError("a memory source does not prove a non-symlink runtime")

    source_counts = (
        eggpu["memory_result_source"].astype(str).value_counts().to_dict()
    )
    return {
        "status": "pass_unified_candidate",
        "scope": (
            "208 EGGPU memory rows measured with the timing candidate and "
            "frozen Python runtime"
        ),
        "row_count": 208,
        "sample_count_per_cell": 3,
        "preserved_columns": list(MEMORY_COLUMNS),
        "candidate_binary_sha256": expected_sha256,
        "runtime_python_snapshot_sha256": expected_runtime_sha256,
        "row_level_compiled_binary_sha256": expected_sha256,
        "measured_with_timing_candidate": True,
        "source_ledger": str(source_ledger.resolve()),
        "source_ledger_sha256": sha256(source_ledger),
        "overlay_audit": str(overlay_audit.resolve()),
        "overlay_audit_sha256": sha256(overlay_audit),
        "memory_comparison_csv": str(cells_path.resolve()),
        "memory_comparison_csv_sha256": sha256(cells_path),
        "measurement_windows": sorted(observed_windows),
        "source_label_counts": {
            str(key): int(value) for key, value in source_counts.items()
        },
        "lineage_note": (
            "Every published memory mean and sample deviation is recomputed "
            "from three independent isolated-process samples recorded by the "
            "same compiled extension and frozen Python runtime as the timing "
            "matrix."
        ),
    }


def bind_candidate_version(
    output: Path,
    expected_sha256: str,
    memory_provenance: dict[str, object],
    *,
    release_label: str,
    hot_hit_patch_scope: str,
) -> dict[str, str]:
    """Record release-specific timing and memory provenance."""

    csv_path = output / "paper_table_baseline_versions.csv"
    json_path = output / "paper_baseline_versions.json"
    if not csv_path.is_file() or not json_path.is_file():
        raise FileNotFoundError("baseline version supplements are incomplete")

    versions = pd.read_csv(csv_path, keep_default_na=False)
    eggpu_rows = versions["system"].astype(str).eq("EGGPU")
    if int(eggpu_rows.sum()) != 1:
        raise ValueError("baseline version table must contain exactly one EGGPU row")
    unified_memory = bool(
        memory_provenance.get("measured_with_timing_candidate")
    )
    runtime_sha = str(
        memory_provenance.get("runtime_python_snapshot_sha256", "")
    )
    if unified_memory:
        version = f"EasyGraph 1.6 + {release_label} unified candidate"
        artifact_scope = (
            f"{release_label} timing and memory: candidate binary "
            f"{expected_sha256[:12]}; frozen Python runtime "
            f"{runtime_sha[:12]}"
        )
        memory_source_ledger_sha = str(
            memory_provenance["source_ledger_sha256"]
        )
        memory_source_snapshot_sha = runtime_sha
        memory_row_binary_sha = expected_sha256
    else:
        version = (
            f"EasyGraph 1.6 + {release_label} timing candidate; archived memory"
        )
        artifact_scope = (
            f"{release_label} public-return/build/processing timings: candidate "
            f"binary {expected_sha256[:12]}; memory: frozen ledger "
            f"{str(memory_provenance['immediate_source_ledger_sha256'])[:12]}, "
            "row-level binary indeterminate"
        )
        memory_source_ledger_sha = str(
            memory_provenance["immediate_source_ledger_sha256"]
        )
        memory_source_snapshot_sha = str(
            memory_provenance["archived_paper_source_snapshot_sha256"]
        )
        memory_row_binary_sha = "indeterminate"

    versions.loc[eggpu_rows, "commit"] = expected_sha256
    versions.loc[eggpu_rows, "version"] = version
    versions.loc[eggpu_rows, "artifact_scope"] = artifact_scope
    extra_columns = {
        "timing_candidate_binary_sha256": expected_sha256,
        "timing_runtime_python_snapshot_sha256": runtime_sha,
        "memory_source_ledger_sha256": memory_source_ledger_sha,
        "memory_archived_source_snapshot_sha256": memory_source_snapshot_sha,
        "memory_row_level_binary_sha256": memory_row_binary_sha,
        "memory_runtime_python_snapshot_sha256": runtime_sha,
    }
    for column, value in extra_columns.items():
        if column not in versions:
            versions[column] = ""
        versions.loc[eggpu_rows, column] = value
    versions.to_csv(csv_path, index=False)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    matched = 0
    for system in payload.get("systems", []):
        if str(system.get("system")) != "EGGPU":
            continue
        matched += 1
        system["commit"] = expected_sha256
        system["version"] = version
        system["artifact_scope"] = artifact_scope
        system["timing_provenance"] = {
            "candidate_binary_sha256": expected_sha256,
            "runtime_python_snapshot_sha256": runtime_sha or None,
            "scope": "208 final-13 build, processing, and public-return timing cells",
        }
        system["memory_provenance"] = memory_provenance
    if matched != 1:
        raise ValueError("baseline version JSON must contain exactly one EGGPU record")
    payload["eggpu_candidate_binary_sha256"] = expected_sha256
    payload["eggpu_runtime_python_snapshot_sha256"] = runtime_sha or None
    payload["eggpu_provenance"] = {
        "timing": {
            "candidate_binary_sha256": expected_sha256,
            "runtime_python_snapshot_sha256": runtime_sha or None,
            "scope": "208 final-13 build, processing, and public-return timing cells",
        },
        "memory": memory_provenance,
        "hot_hit_patch_scope": hot_hit_patch_scope,
        "release_label": release_label,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Rebuild the TeX table from the just-bound CSV/JSON while preserving the
    # GraphScope row supplied by the GraphScope-aware generator.
    complete.write_baseline_versions(output)
    return {
        "csv": sha256(csv_path),
        "json": sha256(json_path),
        "tex": sha256(output / "paper_table_baseline_versions.tex"),
    }


def copy_and_audit_protocol_evidence(
    audit_path: Path,
    source_ledger: Path,
    output: Path,
    expected_sha256: str,
    *,
    expected_runtime_sha256: str = "",
    unified_memory: bool = False,
    release_label: str = "V9",
) -> dict[str, object]:
    """Bind the overlay audit and source protocols into the final bundle."""

    audit_path = audit_path.resolve()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise ValueError("overlay audit status is not pass")
    timing_replacements = int(
        payload.get(
            "timing_replacement_count",
            payload.get("replacement_count", -1),
        )
    )
    expected_timing_replacements = int(
        payload.get(
            "expected_timing_replacements",
            payload.get("expected_replacements", -1),
        )
    )
    if timing_replacements != 208:
        raise ValueError("overlay audit does not contain exactly 208 replacements")
    if expected_timing_replacements != 208:
        raise ValueError("overlay audit was not run with --expected-replacements 208")
    if payload.get("candidate_sha256") != [expected_sha256]:
        raise ValueError(
            f"overlay audit candidate SHA does not match the {release_label} candidate"
        )
    if unified_memory:
        if payload.get("candidate_mode") != "unified-timing-memory":
            raise ValueError("overlay audit is not in unified timing-memory mode")
        if int(payload.get("memory_replacement_count", -1)) != 208:
            raise ValueError("overlay audit does not contain 208 memory replacements")
        if int(payload.get("expected_memory_replacements", -1)) != 208:
            raise ValueError("overlay audit did not require 208 memory replacements")
        if payload.get("runtime_python_snapshot_sha256") != [
            expected_runtime_sha256
        ]:
            raise ValueError("overlay audit runtime snapshot does not match")
    if payload.get("output_ledger_sha256") != sha256(source_ledger):
        raise ValueError("overlay audit output hash does not match the asset ledger")
    if payload.get("rejected_candidates"):
        raise ValueError("overlay audit contains an unsuperseded rejected candidate")

    cells_path = audit_path.with_suffix(".cells.csv")
    cells = pd.read_csv(cells_path, keep_default_na=False)
    if len(cells) != 208 or cells.duplicated(["dataset", "function"]).any():
        raise ValueError("overlay comparison does not contain 208 unique final cells")
    if sorted(set(cells["candidate_sha256"])) != [expected_sha256]:
        raise ValueError("overlay comparison contains a different candidate SHA")
    if unified_memory:
        if "runtime_python_snapshot_sha256" not in cells:
            raise ValueError(
                "overlay timing comparison lacks runtime snapshot provenance"
            )
        if set(cells["runtime_python_snapshot_sha256"].astype(str)) != {
            expected_runtime_sha256
        }:
            raise ValueError(
                "overlay timing comparison uses a different runtime snapshot"
            )

    evidence_dir = output / "source_protocol_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    evidence = []
    source_kind_counts: dict[str, int] = {}
    for index, source in enumerate(payload.get("sources", []), start=1):
        kind = str(source.get("source_kind", ""))
        measurement_kind = str(
            source.get("measurement_kind", "timing")
        )
        count_key = f"{measurement_kind}_{kind}"
        protocol = source.get("measurement_protocol", {})
        if kind == "main" and measurement_kind == "timing":
            required_protocol = {
                "repeat": 5,
                "warmup": 2,
                "easygraph_warmup": 2,
                "eggpu_execution_protocol": "steady-state",
                "measurement_mode": "timing",
                "measured_call_ordinal": 3,
                "aggregation": "arithmetic_mean",
                "paper_estimator": EGGPU_TIMING_ESTIMATOR,
                "independent_process_samples": 5,
                "stability_metric": "e2e",
                "variance_policy": EGGPU_VARIANCE_POLICY,
                "max_over_median_limit": EGGPU_MAX_OVER_MEDIAN_LIMIT,
                "median_over_min_limit": EGGPU_MEDIAN_OVER_MIN_LIMIT,
            }
            source_path = Path(str(source.get("run_metadata_path", "")))
            expected_source_hash = source.get("run_metadata_sha256")
        elif kind == "anchor" and measurement_kind == "timing":
            required_protocol = {
                "aggregate": "arithmetic_mean",
                "repeat": 5,
                "timing_processes": 0,
                "first_use_calls_per_process": 1,
                "extra_untimed_warmups_per_process": 1,
                "measured_call_ordinal": 3,
                "measured_calls_per_process": 1,
                "candidate_sha256": expected_sha256,
                "paper_estimator": EGGPU_TIMING_ESTIMATOR,
                "stability_metric": "e2e",
                "variance_policy": EGGPU_VARIANCE_POLICY,
                "max_over_median_limit": EGGPU_MAX_OVER_MEDIAN_LIMIT,
                "median_over_min_limit": EGGPU_MEDIAN_OVER_MIN_LIMIT,
            }
            if unified_memory:
                required_protocol.update(
                    {
                        "warmup": 1,
                        "memory_repeat": 3,
                        "runtime_python_snapshot_sha256": (
                            expected_runtime_sha256
                        ),
                    }
                )
            source_path = Path(
                str(
                    source.get("anchor_metadata_path")
                    or source.get("protocol_path", "")
                )
            )
            expected_source_hash = (
                source.get("anchor_metadata_sha256")
                or source.get("protocol_json_sha256")
            )
        elif kind == "main" and measurement_kind == "memory":
            required_protocol = {
                "repeat": 3,
                "measurement_mode": "memory",
                "aggregation": "arithmetic_mean",
                "independent_process_samples": 3,
                "measurement_window": "isolated_memory_subprocess",
            }
            source_path = Path(str(source.get("run_metadata_path", "")))
            expected_source_hash = source.get("run_metadata_sha256")
        elif kind == "anchor" and measurement_kind == "memory":
            required_protocol = {
                "memory_repeat": 3,
                "memory_measurement_window": (
                    "isolated_worker_process_full_lifetime"
                ),
                "independent_process_samples": 3,
                "candidate_sha256": expected_sha256,
            }
            if unified_memory:
                required_protocol.update(
                    {
                        "repeat": 5,
                        "timing_processes": 0,
                        "warmup": 1,
                        "runtime_python_snapshot_sha256": (
                            expected_runtime_sha256
                        ),
                    }
                )
            source_path = Path(
                str(
                    source.get("anchor_metadata_path")
                    or source.get("protocol_path", "")
                )
            )
            expected_source_hash = (
                source.get("anchor_metadata_sha256")
                or source.get("protocol_json_sha256")
            )
        else:
            raise ValueError(
                "unknown overlay source kind/measurement: "
                f"{kind!r}/{measurement_kind!r}"
            )
        source_kind_counts[count_key] = source_kind_counts.get(count_key, 0) + 1
        if source.get("candidate_sha256") != expected_sha256:
            raise ValueError(
                f"{source.get('result_dir')}: candidate SHA differs"
            )
        if unified_memory:
            if source.get("runtime_python_sha256") != expected_runtime_sha256:
                raise ValueError(
                    f"{source.get('result_dir')}: runtime snapshot differs"
                )
            if source.get("runtime_package_is_symlink") is not False:
                raise ValueError(
                    f"{source.get('result_dir')}: runtime is not proven frozen"
                )
        for key, expected in required_protocol.items():
            if protocol.get(key) != expected:
                raise ValueError(
                    f"{source.get('result_dir')}: protocol {key}="
                    f"{protocol.get(key)!r}, expected {expected!r}"
                )
        if not source_path.is_file():
            raise FileNotFoundError(f"missing protocol evidence: {source_path}")
        observed_source_hash = sha256(source_path)
        if observed_source_hash != expected_source_hash:
            raise ValueError(f"protocol evidence hash changed: {source_path}")
        destination = evidence_dir / (
            f"{index:02d}_{measurement_kind}_{kind}_"
            f"{Path(str(source['result_dir'])).name}_"
            f"{source_path.name}"
        )
        shutil.copy2(source_path, destination)
        evidence.append(
            {
                "source_kind": kind,
                "measurement_kind": measurement_kind,
                "result_dir": source["result_dir"],
                "copied_path": str(destination.relative_to(output)),
                "sha256": observed_source_hash,
                "anchor_metadata_schema": source.get(
                    "anchor_metadata_schema"
                ),
                "measurement_protocol": protocol,
            }
        )
    if not source_kind_counts.get("timing_main") or not source_kind_counts.get(
        "timing_anchor"
    ):
        raise ValueError(
            f"timing source classes are incomplete: {source_kind_counts}"
        )
    if unified_memory and (
        not source_kind_counts.get("memory_main")
        or not source_kind_counts.get("memory_anchor")
    ):
        raise ValueError(
            f"memory source classes are incomplete: {source_kind_counts}"
        )

    stability_evidence = None
    if unified_memory:
        stability = payload.get("timing_stability_audit")
        if not isinstance(stability, dict):
            raise ValueError("overlay audit lacks timing stability evidence")
        required_stability = {
            "status": "pass",
            "variance_policy": EGGPU_VARIANCE_POLICY,
            "paper_estimator": EGGPU_TIMING_ESTIMATOR,
            "batch_selection_policy": EGGPU_BATCH_SELECTION_POLICY,
            "audited_cells": 208,
            "candidate_sha256": expected_sha256,
            "runtime_python_snapshot_sha256": expected_runtime_sha256,
            "max_over_median_limit": EGGPU_MAX_OVER_MEDIAN_LIMIT,
            "median_over_min_limit": EGGPU_MEDIAN_OVER_MIN_LIMIT,
        }
        for key, expected in required_stability.items():
            if stability.get(key) != expected:
                raise ValueError(
                    f"timing stability {key}={stability.get(key)!r}, "
                    f"expected {expected!r}"
                )
        try:
            validate_gate_calibration(
                stability.get("gate_calibration"),
                repo_root=Path(__file__).resolve().parents[1],
            )
        except (StableTimingProtocolError, TypeError, ValueError) as exc:
            raise ValueError(
                f"timing stability gate calibration differs: {exc}"
            ) from exc
        stability_json = Path(str(stability.get("audit_path", "")))
        stability_rows = Path(str(stability.get("rows_csv_path", "")))
        if not stability_json.is_file() or not stability_rows.is_file():
            raise FileNotFoundError(
                "timing stability JSON/CSV evidence is missing"
            )
        if sha256(stability_json) != stability.get("audit_sha256"):
            raise ValueError("timing stability JSON hash changed")
        if sha256(stability_rows) != stability.get("rows_csv_sha256"):
            raise ValueError("timing stability CSV hash changed")
        stability_payload = json.loads(
            stability_json.read_text(encoding="utf-8")
        )
        for key, expected in required_stability.items():
            if stability_payload.get(key) != expected:
                raise ValueError(
                    f"timing stability JSON {key} differs from overlay evidence"
                )
        try:
            validate_gate_calibration(
                stability_payload.get("gate_calibration"),
                repo_root=Path(__file__).resolve().parents[1],
            )
        except (StableTimingProtocolError, TypeError, ValueError) as exc:
            raise ValueError(
                f"timing stability JSON gate calibration differs: {exc}"
            ) from exc
        overrides = stability.get("batch_overrides_applied")
        if not isinstance(overrides, list):
            raise ValueError("timing stability replacement records are missing")
        if (
            stability.get("batch_overrides_applied_sha256")
            != canonical_json_sha256(overrides)
            or stability_payload.get("batch_overrides_applied") != overrides
        ):
            raise ValueError("timing stability replacement records differ")
        manifest_path_text = str(
            stability.get("override_manifest_path", "")
        )
        manifest_sha256 = str(
            stability.get("override_manifest_sha256", "")
        )
        if overrides or manifest_path_text or manifest_sha256:
            raise ValueError(
                "formal timing stability evidence must use one unique "
                "original raw5 per key and forbids replacements"
            )

        raw_evidence = stability_payload.get("input_evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise ValueError("timing stability raw evidence inventory is missing")
        if (
            stability_payload.get("input_evidence_sha256")
            != canonical_json_sha256(raw_evidence)
        ):
            raise ValueError("timing stability raw evidence digest differs")
        copied_raw_inventory = []
        raw_dir = evidence_dir / "timing_stability_raw"
        raw_dir.mkdir()
        for index, record in enumerate(raw_evidence, 1):
            raw_path = Path(str(record.get("evidence_path", "")))
            expected_raw_sha = str(record.get("evidence_sha256", ""))
            if not raw_path.is_file():
                raise FileNotFoundError(
                    f"timing stability raw evidence is missing: {raw_path}"
                )
            observed_raw_sha = sha256(raw_path)
            if observed_raw_sha != expected_raw_sha:
                raise ValueError(
                    f"timing stability raw evidence hash changed: {raw_path}"
                )
            destination = raw_dir / (
                f"{index:03d}_{observed_raw_sha[:12]}_{raw_path.name}"
            )
            shutil.copy2(raw_path, destination)
            copied_raw_inventory.append(
                {
                    "source_kind": record.get("source_kind"),
                    "result_source": record.get("result_source"),
                    "copied_path": str(destination.relative_to(output)),
                    "sha256": observed_raw_sha,
                }
            )
        copied_stability_json = (
            evidence_dir / "timing_stability_audit.json"
        )
        copied_stability_rows = (
            evidence_dir / "timing_stability_cells.csv"
        )
        shutil.copy2(stability_json, copied_stability_json)
        shutil.copy2(stability_rows, copied_stability_rows)
        stability_evidence = {
            **required_stability,
            "gate_calibration": gate_calibration_payload(),
            "audit_path": str(
                copied_stability_json.relative_to(output)
            ),
            "audit_sha256": sha256(copied_stability_json),
            "rows_csv_path": str(
                copied_stability_rows.relative_to(output)
            ),
            "rows_csv_sha256": sha256(copied_stability_rows),
            "override_manifest_path": "",
            "override_manifest_sha256": "",
            "batch_overrides_applied": overrides,
            "batch_overrides_applied_sha256": (
                canonical_json_sha256(overrides)
            ),
            "raw_evidence": copied_raw_inventory,
            "raw_evidence_count": len(copied_raw_inventory),
        }

    copied_audit = output / audit_path.name
    copied_cells = output / cells_path.name
    shutil.copy2(audit_path, copied_audit)
    shutil.copy2(cells_path, copied_cells)
    copied_memory_cells = None
    if unified_memory:
        memory_cells_path = audit_path.with_suffix(".memory_cells.csv")
        copied_memory_cells = output / memory_cells_path.name
        shutil.copy2(memory_cells_path, copied_memory_cells)
    return {
        "overlay_audit": copied_audit.name,
        "overlay_audit_sha256": sha256(copied_audit),
        "overlay_cells": copied_cells.name,
        "overlay_cells_sha256": sha256(copied_cells),
        "overlay_memory_cells": (
            copied_memory_cells.name if copied_memory_cells is not None else None
        ),
        "overlay_memory_cells_sha256": (
            sha256(copied_memory_cells)
            if copied_memory_cells is not None
            else None
        ),
        "replacement_count": 208,
        "memory_replacement_count": 208 if unified_memory else 0,
        "unique_final_cells": 208,
        "superseded_rejection_count": int(
            payload.get("superseded_rejection_count", 0)
        ),
        "source_kind_counts": source_kind_counts,
        "sources": evidence,
        "timing_stability_audit": stability_evidence,
    }


def write_canonical_timing_evidence(
    ledger: pd.DataFrame,
    overlay_audit: Path,
    output: Path,
    expected_sha256: str,
    expected_runtime_sha256: str,
) -> dict[str, object]:
    """Materialize all V10 EGGPU E2E/kernel raw5 values inside the bundle.

    The overlay comparison stores the derived statistics used by the paper,
    while the source result directories retain the actual process samples.
    This step reparses those source directories with the same strict parser as
    the ledger overlay, binds each workload to the overlay-selected source,
    and emits a canonical long-form file that an offline reviewer can use to
    recompute every reported minimum, mean, sample deviation, and guard ratio.
    """

    overlay_audit = overlay_audit.resolve()
    payload = json.loads(overlay_audit.read_text(encoding="utf-8"))
    timing_sources = [
        source
        for source in payload.get("sources", [])
        if str(source.get("measurement_kind", "timing")) == "timing"
    ]
    if not timing_sources:
        raise ValueError("overlay audit contains no timing sources")

    candidates: list[overlay_tools.Candidate] = []
    rejected_candidates: list[dict[str, object]] = []
    source_inventory = []
    for source in timing_sources:
        kind = str(source.get("source_kind", ""))
        result_dir = Path(str(source.get("result_dir", ""))).resolve()
        if kind == "main":
            accepted, rejected, provenance = overlay_tools.main_candidates(
                result_dir,
                expected_sha256,
                strict_provenance=True,
                max_over_median_limit=EGGPU_MAX_OVER_MEDIAN_LIMIT,
                median_over_min_limit=EGGPU_MEDIAN_OVER_MIN_LIMIT,
            )
        elif kind == "anchor":
            accepted, rejected, provenance = overlay_tools.anchor_candidates(
                result_dir,
                expected_sha256,
                strict_provenance=True,
                max_over_median_limit=EGGPU_MAX_OVER_MEDIAN_LIMIT,
                median_over_min_limit=EGGPU_MEDIAN_OVER_MIN_LIMIT,
            )
        else:
            raise ValueError(f"unknown timing source kind: {kind!r}")
        if provenance.get("candidate_sha256") != expected_sha256:
            raise ValueError(f"{result_dir}: timing candidate SHA changed")
        if provenance.get("runtime_python_sha256") != expected_runtime_sha256:
            raise ValueError(f"{result_dir}: timing runtime SHA changed")
        if provenance.get("runtime_package_is_symlink") is not False:
            raise ValueError(f"{result_dir}: timing runtime is not frozen")
        candidates.extend(accepted)
        rejected_candidates.extend(rejected)
        source_inventory.append(
            {
                "source_kind": kind,
                "result_dir": str(result_dir),
                "candidate_sha256": provenance["candidate_sha256"],
                "runtime_python_snapshot_sha256": provenance[
                    "runtime_python_sha256"
                ],
                "cells": len(accepted),
                "rejected_cells": len(rejected),
            }
        )

    by_key: dict[tuple[str, str], overlay_tools.Candidate] = {}
    for candidate in candidates:
        if candidate.key in by_key:
            raise ValueError(
                f"canonical timing evidence has duplicate cell {candidate.key}"
            )
        by_key[candidate.key] = candidate
    if len(by_key) != 208:
        raise ValueError(
            f"canonical timing evidence has {len(by_key)} cells, expected 208"
        )
    unresolved_rejections = [
        record
        for record in rejected_candidates
        if (
            str(record.get("dataset", "")),
            str(record.get("function", "")),
        )
        not in by_key
    ]
    if unresolved_rejections:
        raise ValueError(
            "canonical timing evidence has rejected workload keys without one "
            f"complete selected batch: {unresolved_rejections[:3]}"
        )

    eggpu = ledger[
        ledger["baseline"].eq("EGGPU")
        & ledger["execution_status"].eq("ok")
    ].copy()
    if len(eggpu) != 208 or eggpu.duplicated(
        ["dataset", "function"]
    ).any():
        raise ValueError(
            "canonical timing evidence requires 208 unique successful EGGPU rows"
        )

    sample_rows: list[dict[str, object]] = []
    for ledger_row in eggpu.to_dict("records"):
        key = (str(ledger_row["dataset"]), str(ledger_row["function"]))
        candidate = by_key.get(key)
        if candidate is None:
            raise ValueError(f"canonical timing evidence lacks ledger cell {key}")
        if candidate.candidate_sha256 != expected_sha256:
            raise ValueError(f"{key}: canonical timing candidate SHA differs")
        if candidate.runtime_python_sha256 != expected_runtime_sha256:
            raise ValueError(f"{key}: canonical timing runtime SHA differs")
        selected_source = str(
            ledger_row.get("timing_result_source")
            or ledger_row.get("result_source")
            or ""
        )
        if Path(selected_source).resolve() != Path(
            candidate.result_source
        ).resolve():
            raise ValueError(
                f"{key}: ledger source {selected_source!r} differs from "
                f"canonical source {candidate.result_source!r}"
            )

        for metric in ("e2e", "kernel"):
            stats = candidate.metrics[metric]
            expected_fields = {
                f"{metric}_paper_seconds": stats.minimum,
                f"{metric}_raw_min_seconds": stats.minimum,
                f"{metric}_raw_median_seconds": stats.median,
                f"{metric}_raw_mean_seconds": stats.mean,
                f"{metric}_std_seconds": stats.stdev,
                f"{metric}_raw_max_seconds": stats.maximum,
                f"{metric}_coefficient_of_variation": (
                    stats.coefficient_of_variation
                ),
                f"{metric}_max_over_median": stats.max_over_median,
                f"{metric}_median_over_minimum": (
                    stats.median_over_minimum
                ),
            }
            for field, expected in expected_fields.items():
                observed = float(ledger_row[field])
                if not math.isclose(
                    observed,
                    expected,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-15,
                ):
                    raise ValueError(
                        f"{key}/{metric}: ledger {field}={observed} differs "
                        f"from raw5-derived {expected}"
                    )
            if str(ledger_row[f"{metric}_estimator"]) != EGGPU_TIMING_ESTIMATOR:
                raise ValueError(f"{key}/{metric}: estimator is not minimum-of-five")
            for sample_index, seconds in enumerate(stats.samples, start=1):
                sample_rows.append(
                    {
                        "dataset": key[0],
                        "function": key[1],
                        "baseline": "EGGPU",
                        "metric": metric,
                        "sample_index": sample_index,
                        "seconds": seconds,
                        "status": "ok",
                        "candidate_sha256": expected_sha256,
                        "runtime_python_snapshot_sha256": (
                            expected_runtime_sha256
                        ),
                        "source_kind": candidate.source_kind,
                        "result_source": candidate.result_source,
                    }
                )

    samples_path = output / "final_13_eggpu_timing_samples.csv"
    pd.DataFrame(sample_rows).sort_values(
        ["dataset", "function", "metric", "sample_index"]
    ).to_csv(samples_path, index=False)
    if len(sample_rows) != 208 * 2 * 5:
        raise ValueError(
            f"canonical timing evidence has {len(sample_rows)} rows, "
            f"expected {208 * 2 * 5}"
        )

    policy = {
        "status": "pass",
        "variance_policy": EGGPU_VARIANCE_POLICY,
        "eggpu_center": EGGPU_TIMING_ESTIMATOR,
        "raw_sample_count": 5,
        "acceptance_metrics": ["e2e"],
        "retained_raw_metrics": ["e2e", "kernel"],
        "dispersion_statistics": [
            "raw_mean",
            "sample_standard_deviation",
            "coefficient_of_variation",
        ],
        "sample_std_and_cv_role": "reported_diagnostics_not_acceptance_gate",
        "max_over_median_limit": EGGPU_MAX_OVER_MEDIAN_LIMIT,
        "median_over_min_limit": EGGPU_MEDIAN_OVER_MIN_LIMIT,
        "batch_selection_policy": EGGPU_BATCH_SELECTION_POLICY,
        "gate_calibration": gate_calibration_payload(),
        "candidate_binary_sha256": expected_sha256,
        "runtime_python_snapshot_sha256": expected_runtime_sha256,
        "timing_cells": 208,
        "raw_sample_groups": 208 * 2,
        "raw_sample_rows": 208 * 2 * 5,
        "raw_samples_artifact": samples_path.name,
        "raw_samples_sha256": sha256(samples_path),
        "source_overlay_audit": str(overlay_audit),
        "source_overlay_audit_sha256": sha256(overlay_audit),
        "source_inventory": source_inventory,
        "superseded_incomplete_candidates": rejected_candidates,
    }
    policy_path = output / "eggpu_timing_stability_policy.json"
    policy_path.write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "pass",
        "raw_samples": samples_path.name,
        "raw_samples_sha256": sha256(samples_path),
        "policy": policy_path.name,
        "policy_sha256": sha256(policy_path),
        "timing_cells": 208,
        "metrics_per_cell": 2,
        "samples_per_metric": 5,
        "raw_sample_groups": 416,
        "raw_sample_rows": 2080,
        "acceptance_metrics": ["e2e"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-assets", required=True, type=Path)
    parser.add_argument("--supplement-assets", required=True, type=Path)
    parser.add_argument("--historical-comparison", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--overlay-audit", type=Path)
    parser.add_argument(
        "--memory-provenance-mode",
        choices=("frozen-supplement", "unified-candidate"),
        default="frozen-supplement",
        help=(
            "Legacy mode verifies untouched archived memory columns. Unified "
            "mode requires 208 three-sample memory replacements from the same "
            "candidate/runtime as timing."
        ),
    )
    parser.add_argument(
        "--release-label",
        default="V9",
        help="Release token used in generated artifact names and descriptions.",
    )
    parser.add_argument(
        "--hot-hit-patch-scope",
        help="Override the release-specific cache fast-path provenance text.",
    )
    args = parser.parse_args()

    source = args.source_assets.resolve()
    supplements = args.supplement_assets.resolve()
    historical = args.historical_comparison.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_ledger = source / "final_13_cell_outcome_ledger.csv"
    if not source_ledger.is_file() or not historical.is_file():
        raise FileNotFoundError("uniform bundle inputs are incomplete")
    unified_memory = args.memory_provenance_mode == "unified-candidate"
    if unified_memory and (
        not args.candidate_sha256 or args.overlay_audit is None
    ):
        raise ValueError(
            "unified-candidate memory requires --candidate-sha256 and "
            "--overlay-audit"
        )
    release_label = str(args.release_label).strip()
    if not release_label:
        raise ValueError("--release-label must not be empty")
    hot_hit_patch_scope = (
        args.hot_hit_patch_scope
        or (
            UNIFIED_HOT_HIT_PATCH_SCOPE
            if unified_memory
            else LEGACY_HOT_HIT_PATCH_SCOPE
        )
    )

    ledger = make_uniform_ledger(source_ledger, output)
    candidate_audit = None
    version_artifacts = None
    protocol_evidence = None
    canonical_timing_evidence = None
    gap_artifacts = None
    candidate_sha256 = None
    runtime_python_sha256 = ""
    if args.candidate_sha256:
        candidate_sha256 = normalize_candidate_sha256(args.candidate_sha256)
        candidate_audit = audit_candidate_ledger(
            ledger,
            candidate_sha256,
            require_runtime_provenance=unified_memory,
        )
        runtime_python_sha256 = str(
            candidate_audit.get("runtime_python_snapshot_sha256") or ""
        )
    elif args.overlay_audit is not None:
        raise ValueError("--candidate-sha256 is required with --overlay-audit")

    if unified_memory:
        memory_provenance = audit_unified_memory_provenance(
            ledger,
            source_ledger,
            args.overlay_audit.resolve(),
            candidate_sha256,
            runtime_python_sha256,
        )
    else:
        memory_provenance = audit_frozen_memory_provenance(ledger, supplements)

    copied = copy_supplements(supplements, output)
    if candidate_sha256 is not None:
        version_artifacts = bind_candidate_version(
            output,
            candidate_sha256,
            memory_provenance,
            release_label=release_label,
            hot_hit_patch_scope=hot_hit_patch_scope,
        )
        if args.overlay_audit is None:
            raise ValueError("--overlay-audit is required with --candidate-sha256")
        protocol_evidence = copy_and_audit_protocol_evidence(
            args.overlay_audit,
            source_ledger,
            output,
            candidate_sha256,
            expected_runtime_sha256=runtime_python_sha256,
            unified_memory=unified_memory,
            release_label=release_label,
        )
        if unified_memory:
            canonical_timing_evidence = write_canonical_timing_evidence(
                ledger,
                args.overlay_audit.resolve(),
                output,
                candidate_sha256,
                runtime_python_sha256,
            )
        gap_artifacts = write_gap_timing_table(
            ledger,
            output,
            candidate_sha256,
            release_label,
        )
    numeric = regenerate_tables(ledger, output)
    igraph_crossover = write_igraph_crossover(ledger, output)
    numeric["igraph_crossover"] = igraph_crossover
    (output / "final_13_numeric_summary.json").write_text(
        json.dumps(numeric, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    chapter4_visuals.setup_style()
    chapter4_visuals.scaling_figure(output, output)
    chapter4_visuals.category_overview_figure(
        output,
        output,
        timing_ledger=output / "final_13_cell_outcome_ledger.csv",
    )
    regenerated_visuals = {
        name: sha256(output / name)
        for name in (
            "scaling_four_functions.pdf",
            "scaling_four_functions.png",
            "scaling_four_functions.metadata.json",
            "category_time_by_baseline_3panel.pdf",
            "category_time_by_baseline_3panel.png",
            "category_time_by_baseline_3panel.metadata.json",
        )
    }
    historical_summary = write_historical_table(historical, output, ledger=ledger)
    final_gate_summary = None
    if candidate_audit is not None:
        final_gate_summary = {
            "status": "pass",
            "release_label": release_label,
            "candidate_binary_sha256": candidate_sha256,
            "runtime_python_snapshot_sha256": (
                runtime_python_sha256 or None
            ),
            "timing_cells": int(candidate_audit["eggpu_cells"]),
            "unique_timing_workloads": int(
                candidate_audit["unique_eggpu_workloads"]
            ),
            "sample_count_per_timing_cell": int(
                candidate_audit["sample_count_per_cell"]
            ),
            "timing_estimator": candidate_audit["timing_estimator"],
            "timing_variance_policy": candidate_audit[
                "variance_policy"
            ],
            "timing_stability_audit": protocol_evidence[
                "timing_stability_audit"
            ],
            "canonical_timing_evidence": canonical_timing_evidence,
            "graphscope_ledger_cells": int(candidate_audit["graphscope_cells"]),
            "overlay_replacements": int(
                protocol_evidence["replacement_count"]
            ),
            "overlay_unique_final_cells": int(
                protocol_evidence["unique_final_cells"]
            ),
            "active_rejections": 0,
            "superseded_rejections": int(
                protocol_evidence["superseded_rejection_count"]
            ),
            "protocol_source_counts": protocol_evidence[
                "source_kind_counts"
            ],
            "gap_twitter": gap_artifacts,
            "memory_provenance_status": memory_provenance["status"],
            "memory_source_ledger_sha256": memory_provenance.get(
                "source_ledger_sha256",
                memory_provenance.get("immediate_source_ledger_sha256"),
            ),
            "memory_archived_v7_lineage": memory_provenance.get(
                "archived_v7_lineage"
            ),
            "memory_row_level_compiled_binary_sha256": (
                memory_provenance.get(
                    "row_level_compiled_binary_sha256", "indeterminate"
                )
            ),
            "memory_runtime_python_snapshot_sha256": (
                memory_provenance.get("runtime_python_snapshot_sha256")
            ),
            "memory_attributed_to_timing_candidate": bool(
                memory_provenance.get("measured_with_timing_candidate")
            ),
            "memory_sample_count_per_cell": memory_provenance.get(
                "sample_count_per_cell"
            ),
            "relative_tie_tolerance": complete.RELATIVE_TIE_TOLERANCE,
            "processing_overall_cross_system_speedup_reported": False,
            "public_return_endpoint": (
                "reused-state in-process public function call"
            ),
        }
        release_slug = re.sub(
            r"[^A-Za-z0-9]+", "_", release_label
        ).strip("_").upper()
        gate_path = output / f"{release_slug}_FINAL_GATE_SUMMARY.json"
        gate_path.write_text(
            json.dumps(final_gate_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        final_gate_summary["artifact"] = gate_path.name
        final_gate_summary["artifact_sha256"] = sha256(gate_path)

    manifest = {
        "status": "complete",
        "release_label": release_label,
        "memory_provenance_mode": args.memory_provenance_mode,
        "timing_estimator": (
            "EGGPU reports the stability-gated minimum of five independent "
            "processes and retains the same batch's arithmetic mean, sample "
            "standard deviation, extrema, and CV; external systems retain "
            "their frozen source-ledger estimators"
        ),
        "eggpu_variance_policy": EGGPU_VARIANCE_POLICY,
        "source_assets": str(source),
        "source_ledger_sha256": sha256(source_ledger),
        "supplement_assets": str(supplements),
        "copied_supplements": copied,
        "historical_comparison": str(historical),
        "historical_comparison_sha256": sha256(historical),
        "numeric_summary": numeric,
        "historical_summary": historical_summary,
        "igraph_crossover": igraph_crossover,
        "gap_twitter": gap_artifacts,
        "final_gate_summary": final_gate_summary,
        "regenerated_visuals": regenerated_visuals,
        "candidate_audit": candidate_audit,
        "eggpu_provenance": {
            "timing": (
                {
                    "candidate_binary_sha256": candidate_sha256,
                    "runtime_python_snapshot_sha256": (
                        runtime_python_sha256 or None
                    ),
                    "scope": (
                        "208 final-13 build, processing, and public-return "
                        "timing cells"
                    ),
                }
                if candidate_audit is not None
                else None
            ),
            "memory": memory_provenance,
            "hot_hit_patch_scope": hot_hit_patch_scope,
        },
        "candidate_version_artifacts": version_artifacts,
        "protocol_evidence": protocol_evidence,
        "canonical_timing_evidence": canonical_timing_evidence,
        "candidate_binding_scope": (
            (
                "The candidate binary SHA and frozen Python runtime digest bind "
                "all 208 final-13 EGGPU timing cells and all 208 three-sample "
                "memory cells, including regular workloads and scale anchors. "
                "Copied scaling, workflow, and ablation supplements retain their "
                "own provenance."
                if unified_memory
                else (
                    "The candidate binary SHA binds the 208 final-13 EGGPU "
                    "timing cells and copied timing protocols. Memory rows are "
                    "retained from the supplement ledger under their archived "
                    "provenance. Copied scaling, workflow, and ablation "
                    "supplements retain their own provenance."
                )
            )
            if candidate_audit is not None
            else None
        ),
    }
    (output / "VLDB_UNIFORM_ASSET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
