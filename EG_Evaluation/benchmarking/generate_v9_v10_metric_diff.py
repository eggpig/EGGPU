#!/usr/bin/env python3
"""Generate an auditable V9-to-V10 metric and cell-level comparison.

The report describes what changed between two frozen bundles.  It deliberately
does not attribute the delta to a single optimization because V10 changes both
the compiled candidate and the EGGPU paper estimator (V9 arithmetic mean versus
V10 stability-gated minimum of one complete five-run batch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


KEY_METRICS = (
    ("e2e_eggpu_successful_workloads", "higher"),
    ("e2e_competitive_pairs", "higher"),
    ("e2e_common_pair_strict_wins", "higher"),
    ("e2e_common_pair_ties", "neutral"),
    ("e2e_common_pair_losses", "lower"),
    ("e2e_sole_validated_cells", "neutral"),
    ("e2e_graphscope_validated_cells", "neutral"),
    ("e2e_speedup_over_best_competitor", "higher"),
    ("e2e_speedup_over_best_cpu", "higher"),
    ("e2e_speedup_over_strict_nx_cugraph", "higher"),
    ("e2e_strict_nx_cugraph_common_pairs", "neutral"),
    ("kernel_eggpu_successful_workloads", "higher"),
    ("kernel_competitive_pairs", "higher"),
    ("kernel_common_pair_strict_wins", "higher"),
    ("kernel_common_pair_ties", "neutral"),
    ("kernel_common_pair_losses", "lower"),
    ("kernel_sole_validated_cells", "neutral"),
    ("kernel_graphscope_validated_cells", "neutral"),
    ("kernel_speedup_over_best_native_gpu", "higher"),
    ("kernel_best_native_gpu_common_pairs", "neutral"),
    ("eggpu_memory_cells", "higher"),
    ("eggpu_max_gpu_peak_mb", "lower"),
    ("graphscope_host_memory_cells", "neutral"),
    ("graphscope_host_rss_max_mb", "lower"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geomean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def load_bundle(root: Path, expected_release: str) -> tuple[pd.DataFrame, dict, dict]:
    root = root.resolve()
    ledger_path = root / "final_13_cell_outcome_ledger.csv"
    numeric_path = root / "final_13_numeric_summary.json"
    manifest_path = root / "VLDB_UNIFORM_ASSET_MANIFEST.json"
    for path in (ledger_path, numeric_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"comparison input is missing: {path}")
    ledger = pd.read_csv(ledger_path, low_memory=False)
    numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_release = str(manifest.get("release_label", "")).upper()
    if observed_release != expected_release.upper():
        # The archived V9 manifest predates the release_label field.  Bind
        # that one legacy schema through its signed-off V9 gate rather than
        # inferring the release from a directory name.
        legacy_gate_path = root / "V9_FINAL_GATE_SUMMARY.json"
        legacy_gate = (
            json.loads(legacy_gate_path.read_text(encoding="utf-8"))
            if legacy_gate_path.is_file()
            else {}
        )
        manifest_candidate = (
            (manifest.get("candidate_audit") or {}).get("candidate_sha256")
        )
        legacy_v9_bound = (
            expected_release.upper() == "V9"
            and not observed_release
            and legacy_gate.get("status") == "pass"
            and legacy_gate.get("candidate_binary_sha256")
            == manifest_candidate
            and int(legacy_gate.get("timing_cells", -1)) == 208
        )
        if not legacy_v9_bound:
            raise ValueError(
                f"{manifest_path}: release_label="
                f"{manifest.get('release_label')!r}, expected "
                f"{expected_release!r}, and no matching legacy gate exists"
            )
    return ledger, numeric, manifest


def eggpu_rows(ledger: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {
        "dataset",
        "function",
        "baseline",
        "execution_status",
        "validation_status",
        "e2e_paper_seconds",
        "kernel_paper_seconds",
        "build_paper_seconds",
        "e2e_raw_mean_seconds",
        "kernel_raw_mean_seconds",
        "build_raw_mean_seconds",
        "e2e_std_seconds",
        "kernel_std_seconds",
        "build_std_seconds",
        "e2e_estimator",
        "kernel_estimator",
        "build_estimator",
        "gpu_peak_mb_mean",
        "host_rss_peak_mb_mean",
    }
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"{label} ledger lacks fields {sorted(missing)}")
    rows = ledger[ledger["baseline"].eq("EGGPU")].copy()
    if len(rows) != 208 or rows.duplicated(["dataset", "function"]).any():
        raise ValueError(f"{label} does not contain 208 unique EGGPU cells")
    if not rows["execution_status"].eq("ok").all():
        raise ValueError(f"{label} contains a non-successful EGGPU cell")
    return rows.sort_values(["dataset", "function"]).reset_index(drop=True)


def build_cell_timing_diff(v9: pd.DataFrame, v10: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "function"]
    left = v9.set_index(keys)
    right = v10.set_index(keys)
    if set(left.index) != set(right.index):
        raise ValueError("V9 and V10 timing ledgers have different workload keys")
    rows = []
    for key in sorted(left.index):
        old = left.loc[key]
        new = right.loc[key]
        for metric in ("build", "kernel", "e2e"):
            old_seconds = finite_number(
                old[f"{metric}_paper_seconds"], f"V9 {key}/{metric}"
            )
            new_seconds = finite_number(
                new[f"{metric}_paper_seconds"], f"V10 {key}/{metric}"
            )
            if old_seconds <= 0 or new_seconds <= 0:
                raise ValueError(f"{key}/{metric}: paper timing is non-positive")
            rows.append(
                {
                    "dataset": key[0],
                    "function": key[1],
                    "metric": metric,
                    "v9_estimator": old[f"{metric}_estimator"],
                    "v10_estimator": new[f"{metric}_estimator"],
                    "v9_paper_seconds": old_seconds,
                    "v10_paper_seconds": new_seconds,
                    "v9_over_v10": old_seconds / new_seconds,
                    "v10_minus_v9_seconds": new_seconds - old_seconds,
                    "v10_relative_change_percent": (
                        (new_seconds / old_seconds - 1.0) * 100.0
                    ),
                    "v9_raw_mean_seconds": finite_number(
                        old[f"{metric}_raw_mean_seconds"],
                        f"V9 raw mean {key}/{metric}",
                    ),
                    "v10_raw_mean_seconds": finite_number(
                        new[f"{metric}_raw_mean_seconds"],
                        f"V10 raw mean {key}/{metric}",
                    ),
                    "v9_sample_std_seconds": finite_number(
                        old[f"{metric}_std_seconds"],
                        f"V9 SD {key}/{metric}",
                    ),
                    "v10_sample_std_seconds": finite_number(
                        new[f"{metric}_std_seconds"],
                        f"V10 SD {key}/{metric}",
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_memory_diff(v9: pd.DataFrame, v10: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "function"]
    left = v9.set_index(keys)
    right = v10.set_index(keys)
    rows = []
    for key in sorted(left.index):
        old = left.loc[key]
        new = right.loc[key]
        row: dict[str, Any] = {"dataset": key[0], "function": key[1]}
        for resource, field in (
            ("gpu_peak_mb", "gpu_peak_mb_mean"),
            ("host_rss_peak_mb", "host_rss_peak_mb_mean"),
        ):
            old_value = finite_number(old[field], f"V9 {key}/{field}")
            new_value = finite_number(new[field], f"V10 {key}/{field}")
            if old_value <= 0 or new_value <= 0:
                raise ValueError(f"{key}/{field}: memory value is non-positive")
            row[f"v9_{resource}"] = old_value
            row[f"v10_{resource}"] = new_value
            row[f"v10_over_v9_{resource}"] = new_value / old_value
        for name in (
            "memory_measurement_window",
            "memory_result_source",
            "memory_candidate_sha256",
            "memory_runtime_python_snapshot_sha256",
        ):
            row[f"v9_{name}"] = old.get(name, "")
            row[f"v10_{name}"] = new.get(name, "")
        rows.append(row)
    return pd.DataFrame(rows)


def scalar_delta(old: Any, new: Any) -> tuple[Any, Any, Any]:
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        old_number = finite_number(old, "V9 scalar")
        new_number = finite_number(new, "V10 scalar")
        absolute = new_number - old_number
        ratio = new_number / old_number if old_number != 0 else None
        return absolute, ratio, (
            (ratio - 1.0) * 100.0 if ratio is not None else None
        )
    return None, None, None


def add_metric_row(
    rows: list[dict[str, Any]],
    metric: str,
    old: Any,
    new: Any,
    preferred_direction: str,
    source: str,
) -> None:
    absolute, ratio, percent = scalar_delta(old, new)
    rows.append(
        {
            "metric": metric,
            "v9_value": old,
            "v10_value": new,
            "absolute_delta_v10_minus_v9": absolute,
            "v10_over_v9": ratio,
            "v10_relative_change_percent": percent,
            "preferred_direction": preferred_direction,
            "source": source,
        }
    )


def build_key_metrics(
    v9_numeric: dict,
    v10_numeric: dict,
    timing: pd.DataFrame,
    memory: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, direction in KEY_METRICS:
        if metric not in v9_numeric or metric not in v10_numeric:
            raise ValueError(f"numeric summary lacks comparison metric {metric}")
        add_metric_row(
            rows,
            metric,
            v9_numeric[metric],
            v10_numeric[metric],
            direction,
            "final_13_numeric_summary.json",
        )

    timing_summary: dict[str, Any] = {}
    for metric in ("build", "kernel", "e2e"):
        part = timing[timing["metric"].eq(metric)]
        ratios = part["v9_over_v10"].astype(float)
        tolerance = 1.0e-12
        summary = {
            "cells": len(part),
            "v9_over_v10_geomean": geomean(ratios.tolist()),
            "v9_over_v10_median": float(ratios.median()),
            "v9_over_v10_minimum": float(ratios.min()),
            "v9_over_v10_maximum": float(ratios.max()),
            "v10_lower_seconds_cells": int((ratios > 1.0 + tolerance).sum()),
            "equal_seconds_cells": int(
                ((ratios - 1.0).abs() <= tolerance).sum()
            ),
            "v10_higher_seconds_cells": int((ratios < 1.0 - tolerance).sum()),
        }
        timing_summary[metric] = summary
        add_metric_row(
            rows,
            f"{metric}_paper_seconds_v9_over_v10_geomean",
            1.0,
            summary["v9_over_v10_geomean"],
            "higher",
            "V9_TO_V10_CELL_TIMING_DIFF.csv",
        )
        for count_name in (
            "v10_lower_seconds_cells",
            "equal_seconds_cells",
            "v10_higher_seconds_cells",
        ):
            add_metric_row(
                rows,
                f"{metric}_{count_name}",
                None,
                summary[count_name],
                "neutral",
                "V9_TO_V10_CELL_TIMING_DIFF.csv",
            )

    memory_summary: dict[str, Any] = {}
    for resource in ("gpu_peak_mb", "host_rss_peak_mb"):
        old_values = memory[f"v9_{resource}"].astype(float).tolist()
        new_values = memory[f"v10_{resource}"].astype(float).tolist()
        ratios = memory[f"v10_over_v9_{resource}"].astype(float).tolist()
        summary = {
            "cells": len(ratios),
            "v9_geomean_mb": geomean(old_values),
            "v10_geomean_mb": geomean(new_values),
            "v10_over_v9_geomean": geomean(ratios),
            "v9_maximum_mb": max(old_values),
            "v10_maximum_mb": max(new_values),
        }
        memory_summary[resource] = summary
        add_metric_row(
            rows,
            f"eggpu_{resource}_geomean_mb",
            summary["v9_geomean_mb"],
            summary["v10_geomean_mb"],
            "lower",
            "V9_TO_V10_MEMORY_DIFF.csv",
        )
        add_metric_row(
            rows,
            f"eggpu_{resource}_maximum_mb",
            summary["v9_maximum_mb"],
            summary["v10_maximum_mb"],
            "lower",
            "V9_TO_V10_MEMORY_DIFF.csv",
        )
    return pd.DataFrame(rows), timing_summary, memory_summary


def markdown_report(
    summary: dict[str, Any],
    key_metrics: pd.DataFrame,
) -> str:
    lines = [
        "# V9 → V10 Key-Metric Diff",
        "",
        "Status: **PASS**",
        "",
        (
            "This is a release-to-release comparison, not a single-factor "
            "ablation. V10 changes the compiled candidate and reports EGGPU as "
            "the stability-gated minimum of one complete five-run batch; V9 "
            "used its archived estimator. External-baseline values and "
            "estimators remain frozen."
        ),
        "",
        "## Timing-cell summary",
        "",
        "| Metric | Cells | V9/V10 geomean | V10 lower | Equal | V10 higher |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in ("build", "kernel", "e2e"):
        item = summary["timing_cell_summary"][metric]
        lines.append(
            f"| {metric} | {item['cells']} | "
            f"{item['v9_over_v10_geomean']:.6g}× | "
            f"{item['v10_lower_seconds_cells']} | "
            f"{item['equal_seconds_cells']} | "
            f"{item['v10_higher_seconds_cells']} |"
        )
    lines.extend(
        [
            "",
            "## Headline metrics",
            "",
            "| Metric | V9 | V10 | Δ (V10−V9) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    headline = key_metrics[
        key_metrics["source"].eq("final_13_numeric_summary.json")
    ]
    for row in headline.to_dict("records"):
        old = row["v9_value"]
        new = row["v10_value"]
        delta = row["absolute_delta_v10_minus_v9"]
        lines.append(f"| {row['metric']} | {old} | {new} | {delta} |")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- V9 ledger SHA-256: `{summary['inputs']['v9_ledger_sha256']}`",
            f"- V10 ledger SHA-256: `{summary['inputs']['v10_ledger_sha256']}`",
            (
                "- V10 candidate SHA-256: "
                f"`{summary['v10_candidate_binary_sha256']}`"
            ),
            (
                "- V10 runtime snapshot SHA-256: "
                f"`{summary['v10_runtime_python_snapshot_sha256']}`"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v9-assets", required=True, type=Path)
    parser.add_argument("--v10-assets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    args = parser.parse_args()

    v9_root = args.v9_assets.resolve()
    v10_root = args.v10_assets.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    v9_ledger, v9_numeric, _v9_manifest = load_bundle(v9_root, "V9")
    v10_ledger, v10_numeric, v10_manifest = load_bundle(v10_root, "V10")
    v9_eggpu = eggpu_rows(v9_ledger, "V9")
    v10_eggpu = eggpu_rows(v10_ledger, "V10")

    candidate_values = sorted(
        set(v10_eggpu["candidate_sha256"].dropna().astype(str))
    )
    if candidate_values != [args.candidate_sha256]:
        raise ValueError(
            f"V10 candidate values are {candidate_values}, expected "
            f"{[args.candidate_sha256]}"
        )
    runtime_values = sorted(
        set(
            v10_eggpu["runtime_python_snapshot_sha256"]
            .dropna()
            .astype(str)
        )
    )
    if len(runtime_values) != 1:
        raise ValueError(f"V10 runtime identities are {runtime_values}")
    if v10_manifest.get("memory_provenance_mode") != "unified-candidate":
        raise ValueError("V10 bundle is not in unified-candidate memory mode")

    timing = build_cell_timing_diff(v9_eggpu, v10_eggpu)
    memory = build_memory_diff(v9_eggpu, v10_eggpu)
    key_metrics, timing_summary, memory_summary = build_key_metrics(
        v9_numeric,
        v10_numeric,
        timing,
        memory,
    )

    timing_path = output / "V9_TO_V10_CELL_TIMING_DIFF.csv"
    memory_path = output / "V9_TO_V10_MEMORY_DIFF.csv"
    key_path = output / "V9_TO_V10_KEY_METRIC_DIFF.csv"
    timing.to_csv(timing_path, index=False)
    memory.to_csv(memory_path, index=False)
    key_metrics.to_csv(key_path, index=False)

    summary = {
        "status": "pass",
        "comparison": "V9_UNIFORM_to_V10_UNIFORM",
        "attribution_scope": (
            "release-to-release delta; not attributable to one optimization "
            "because candidate code and the EGGPU paper estimator both change"
        ),
        "external_baseline_policy": (
            "external values and estimators remain frozen in each source ledger"
        ),
        "inputs": {
            "v9_assets": str(v9_root),
            "v10_assets": str(v10_root),
            "v9_ledger_sha256": sha256(
                v9_root / "final_13_cell_outcome_ledger.csv"
            ),
            "v10_ledger_sha256": sha256(
                v10_root / "final_13_cell_outcome_ledger.csv"
            ),
            "v9_numeric_summary_sha256": sha256(
                v9_root / "final_13_numeric_summary.json"
            ),
            "v10_numeric_summary_sha256": sha256(
                v10_root / "final_13_numeric_summary.json"
            ),
        },
        "v10_candidate_binary_sha256": args.candidate_sha256,
        "v10_runtime_python_snapshot_sha256": runtime_values[0],
        "timing_cell_summary": timing_summary,
        "memory_summary": memory_summary,
        "artifacts": {
            timing_path.name: sha256(timing_path),
            memory_path.name: sha256(memory_path),
            key_path.name: sha256(key_path),
        },
    }
    markdown_path = output / "V9_TO_V10_KEY_METRIC_DIFF.md"
    markdown_path.write_text(
        markdown_report(summary, key_metrics),
        encoding="utf-8",
    )
    summary["artifacts"][markdown_path.name] = sha256(markdown_path)
    json_path = output / "V9_TO_V10_KEY_METRIC_DIFF.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
