#!/usr/bin/env python3
"""Derive paper-sync facts from one verified V15 asset directory.

This program never carries benchmark results as constants.  It validates the
release attestation and the hash-bound headline inputs, then derives a compact
JSON document from the published CSV/JSON evidence.  ``--allow-fixture-v14``
is a deliberately non-release escape hatch for regression tests against the
frozen V14 uniform-minimum fixture.  Release mode requires the mixed estimator:
EGGPU minimum-of-five, external-baseline mean-of-five, and five-run sample SD.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "eggpu_v15_paper_sync_facts_v2_mixed_estimator"
VERIFICATION_NAME = "V15_FINAL_VERIFICATION.json"
HEADLINE_CSV = "paper_table_headline_results.csv"
HEADLINE_MANIFEST = "paper_table_headline_results_manifest.json"
FAMILY_SUMMARY = "paper_table_category_13_summary.csv"
PAIRWISE_SOTA = "final_13_pairwise_sota_details.csv"
PAIRWISE_BASELINE = "pairwise_baseline_13_exact_summary.csv"
LEDGER = "final_13_cell_outcome_ledger.csv"
FIGURE3_METADATA = "category_time_by_baseline_3panel.metadata.json"
FIGURE3_DATA = "category_time_by_baseline_3panel.csv"
V14_ESTIMATOR_MANIFEST = "UNIFORM_MINIMUM_OF_FIVE_MANIFEST.json"
V14_RAW_SAMPLES = "uniform_minimum_raw_samples.csv"
V15_ESTIMATOR_MANIFEST = "MIXED_ESTIMATOR_FIVE_RUN_MANIFEST.json"
V15_RAW_SAMPLES = "five_run_raw_samples.csv"
V15_ESTIMATOR_AUDIT = "mixed_estimator_audit.csv"
V15_AUDIT_DIR = "v15_unified_overlay_audit"
V15_OVERLAY_LEDGER = "eggpu_v15_unified_overlay_ledger.csv"
V14_HEADLINE_SCHEMA = "eggpu_headline_results_v1"
V15_HEADLINE_SCHEMA = "eggpu_headline_results_v2_mixed_estimator"
V15_ESTIMATOR_MANIFEST_SCHEMA = (
    "eggpu_minimum_baseline_mean_five_run_v1"
)
V15_DISPLAY_ESTIMATOR = "mixed_by_baseline_class"
V15_ESTIMATOR_BY_BASELINE_CLASS = {
    "EGGPU": "minimum_of_five",
    "external_baselines": "arithmetic_mean_of_five",
}
V15_ESTIMATOR_POLICY = {
    "EGGPU": "minimum_of_five",
    "baselines": "arithmetic_mean_of_five",
    "error_bar": "sample_standard_deviation_ddof1",
    "samples": 5,
}
V15_ERROR_BAR = "sample_standard_deviation_ddof1"

HEADLINE_IDS = (
    "library_public_call_e2e",
    "native_gpu_processing",
    "strict_nxcugraph_public_call_e2e",
)
FAMILIES = (
    "Centrality",
    "Connectivity",
    "Paths & Spanning Trees",
    "Structural Holes",
)
BASELINES = ("igraph", "GraphScope", "Gunrock")
METRICS = ("e2e", "kernel")
FIGURE3_TITLES = ("Graph construction", "Processing time", "End-to-end")
ARCHIVED_SINGLE_SOURCE_SYSTEMS = ("EGGPU-2024",)
RELATIVE_TIE_TOLERANCE = 5.0e-4
EXPECTED_NATIVE_SHA256 = (
    "d2a93a3da0ffd0c554c9dab52c9d05d8c1f5f5b4904bcfd19fd95b695bb06eaf"
)
EXPECTED_RUNTIME_SHA256 = (
    "1be28aeb48374c65642b26f7233a7764f66fb3fb8003e6434e04744e08b643fb"
)
EXPECTED_V14_BASE_LEDGER_SHA256 = (
    "307722ed0ae82e5b7f43500d27e9847df215b4d85fc2093105f75ab675a293a1"
)
EXPECTED_V14_BASE_DIRECTORY = (
    "EGGPU_FINAL_EXPERIMENT_ASSETS_14_UNIFORM_MIN_20260730"
)
PAGERANK_PROTOCOL = "eggpu_pagerank_direct_public_call_fail_closed_v1"
PAGERANK_ADOPTION_POLICY = (
    "protocol_validity_only_unconditional_no_relative_timing_selection"
)
PAGERANK_DIRECT_ANCHORS = ("com-Orkut", "GAP-twitter")
EXPECTED_SAMPLES_PER_METRIC = 5
EXPECTED_DATASETS = 13
EXPECTED_FUNCTIONS = 16
EXPECTED_SYSTEM_NAMES = (
    "networkx",
    "easygraph-cpu",
    "easygraph-cpp",
    "igraph",
    "GraphScope",
    "nx-cugraph",
    "Gunrock",
    "EGGPU",
)
EXPECTED_WORKLOADS = EXPECTED_DATASETS * EXPECTED_FUNCTIONS
EXPECTED_LEDGER_ROWS = EXPECTED_WORKLOADS * len(EXPECTED_SYSTEM_NAMES)
EXPECTED_SUCCESSFUL_CELLS = 1040
EXPECTED_DISPLAYED_METRICS = EXPECTED_SUCCESSFUL_CELLS * 3
EXPECTED_RAW_ROWS = EXPECTED_DISPLAYED_METRICS * EXPECTED_SAMPLES_PER_METRIC
EXPECTED_EGGPU_CELLS = EXPECTED_WORKLOADS
EXPECTED_NON_EGGPU_ROWS = EXPECTED_LEDGER_ROWS - EXPECTED_EGGPU_CELLS
EXPECTED_V15_DIRECTORY_PREFIX = "EGGPU_FINAL_EXPERIMENT_ASSETS_15_"
V15_ATTESTED_ASSETS = (
    "FINAL_13_NUMERICAL_RESULTS.md",
    "final_13_numeric_summary.json",
    PAIRWISE_SOTA,
    PAIRWISE_BASELINE,
    "paper_table_main_compact_best_competitor.tex",
    "paper_table_main_compact_pairwise_speedup.tex",
    HEADLINE_CSV,
    "paper_table_headline_results.tex",
    "paper_table_headline_results_README.md",
    HEADLINE_MANIFEST,
    "paper_table_baseline_versions.csv",
    "paper_table_baseline_versions.tex",
    "paper_baseline_versions.json",
    "equal_mean_estimator_sensitivity.csv",
    "EQUAL_MEAN_ESTIMATOR_SENSITIVITY.json",
    FIGURE3_DATA,
    FIGURE3_METADATA,
    "category_time_by_baseline_3panel.pdf",
    "category_time_by_baseline_3panel.png",
)
VALIDATION_OK = {
    "pass",
    "reference",
    "external_reference_pass",
    "sampled_pass",
}


class FactError(RuntimeError):
    """Raised when an input is not release-safe or internally consistent."""


def fail(message: str) -> None:
    raise FactError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{label} is not a lowercase SHA-256 digest")
    return value


def require_file(asset_dir: Path, name: str) -> Path:
    path = asset_dir / name
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing or empty required asset: {name}")
    return path


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON {path.name}: {error}")
    if not isinstance(value, dict):
        fail(f"{path.name} must contain a JSON object")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                fail(f"{path.name} has no CSV header")
            return [dict(row) for row in reader]
    except OSError as error:
        fail(f"cannot read CSV {path.name}: {error}")


def require_columns(
    rows: Sequence[Mapping[str, str]], names: Iterable[str], source: str
) -> None:
    if not rows:
        fail(f"{source} has no data rows")
    missing = sorted(set(names) - set(rows[0]))
    if missing:
        fail(f"{source} is missing required columns: {missing}")


def parse_int(value: object, label: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        fail(f"{label} is not an integer: {value!r}")
    if not math.isfinite(number) or not number.is_integer():
        fail(f"{label} is not an integer: {value!r}")
    return int(number)


def parse_float(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nullable: bool = False,
) -> float | None:
    if nullable and (value is None or str(value).strip() == ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        fail(f"{label} is not numeric: {value!r}")
    if not math.isfinite(number):
        fail(f"{label} is not finite: {value!r}")
    if positive and number <= 0.0:
        fail(f"{label} must be positive: {value!r}")
    return number


def parse_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    fail(f"{label} is not boolean: {value!r}")


def require_close(left: object, right: object, label: str) -> None:
    left_number = parse_float(left, f"{label} left")
    right_number = parse_float(right, f"{label} right")
    assert left_number is not None and right_number is not None
    if not math.isclose(
        left_number, right_number, rel_tol=1.0e-11, abs_tol=1.0e-14
    ):
        fail(f"{label} differs: {left_number!r} != {right_number!r}")


def require_measurement_close(
    left: object, right: object, label: str
) -> None:
    """Compare values reconstructed from portable five-run CSV evidence."""
    left_number = parse_float(left, f"{label} left")
    right_number = parse_float(right, f"{label} right")
    assert left_number is not None and right_number is not None
    if not math.isclose(
        left_number, right_number, rel_tol=2.0e-8, abs_tol=2.0e-11
    ):
        fail(f"{label} differs: {left_number!r} != {right_number!r}")


def numeric_equivalent(left: object, right: object) -> bool:
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_number)
        and math.isfinite(right_number)
        and math.isclose(
            left_number,
            right_number,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
    )


def weighted_geomean(
    rows: Sequence[Mapping[str, str]], column: str, source: str
) -> float | None:
    weighted_logs: list[float] = []
    weights: list[int] = []
    for row in rows:
        weight = parse_int(row["common_pairs"], f"{source}.common_pairs")
        if weight == 0:
            continue
        if weight < 0:
            fail(f"{source}.common_pairs cannot be negative")
        value = parse_float(row[column], f"{source}.{column}", positive=True)
        assert value is not None
        weighted_logs.append(weight * math.log(value))
        weights.append(weight)
    if not weights:
        return None
    return math.exp(math.fsum(weighted_logs) / sum(weights))


def geomean_numbers(values: Sequence[float], label: str) -> float | None:
    if not values:
        return None
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        fail(f"{label} requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def safe_relative_path(root: Path, name: object, label: str) -> Path:
    if not isinstance(name, str) or not name:
        fail(f"{label} contains an invalid path key: {name!r}")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        fail(f"{label} path must remain relative to the asset directory: {name}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        fail(f"{label} path escapes the asset directory: {name}")
    return candidate


def validate_hash_map(
    root: Path, value: object, label: str
) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        fail(f"{label} must be a non-empty hash map")
    validated: dict[str, str] = {}
    for name, expected in sorted(value.items()):
        path = safe_relative_path(root, name, label)
        if not path.is_file():
            fail(f"{label} references a missing file: {name}")
        expected = require_sha256(expected, f"{label}.{name}")
        actual = sha256_file(path)
        if actual != expected:
            fail(f"{label} SHA-256 differs for {name}: {actual} != {expected}")
        validated[str(name)] = actual
    return validated


def typed_headline(row: Mapping[str, str]) -> dict[str, object]:
    integer_fields = (
        "eggpu_validated_workloads",
        "common_pairs",
        "eggpu_strict_wins",
        "eggpu_ties",
        "eggpu_losses",
        "eggpu_only_vs_set",
    )
    result: dict[str, object] = {
        name: str(row[name])
        for name in (
            "comparison_id",
            "display_label",
            "metric",
            "comparison_set",
            "timing_boundary",
            "display_speedup",
            "display_estimator",
        )
    }
    result.update(
        {
            name: parse_int(row[name], f"{HEADLINE_CSV}.{name}")
            for name in integer_fields
        }
    )
    result["geomean_speedup"] = parse_float(
        row["geomean_speedup"],
        f"{HEADLINE_CSV}.geomean_speedup",
        positive=True,
    )
    result["tie_tolerance_relative"] = parse_float(
        row["tie_tolerance_relative"],
        f"{HEADLINE_CSV}.tie_tolerance_relative",
        positive=True,
    )
    if (
        int(result["eggpu_strict_wins"])
        + int(result["eggpu_ties"])
        + int(result["eggpu_losses"])
        != int(result["common_pairs"])
    ):
        fail(
            f"{HEADLINE_CSV}.{result['comparison_id']}: "
            "wins + ties + losses != common pairs"
        )
    if (
        int(result["common_pairs"]) + int(result["eggpu_only_vs_set"])
        != int(result["eggpu_validated_workloads"])
    ):
        fail(
            f"{HEADLINE_CSV}.{result['comparison_id']}: "
            "common + EGGPU-only != validated workloads"
        )
    return result


def validate_headlines(
    asset_dir: Path,
    *,
    release_mode: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    csv_path = require_file(asset_dir, HEADLINE_CSV)
    manifest_path = require_file(asset_dir, HEADLINE_MANIFEST)
    manifest = read_json(manifest_path)
    expected_schema = (
        V15_HEADLINE_SCHEMA if release_mode else V14_HEADLINE_SCHEMA
    )
    if (
        manifest.get("status") != "pass"
        or manifest.get("schema_version") != expected_schema
    ):
        fail("headline manifest status/schema gate did not pass")
    if manifest.get("source_asset_directory_name") != asset_dir.name:
        fail("headline manifest source directory differs from --asset-dir")
    matrix = manifest.get("matrix_contract")
    if not isinstance(matrix, dict) or not (
        parse_int(matrix.get("datasets"), "headline matrix datasets")
        == EXPECTED_DATASETS
        and parse_int(matrix.get("functions"), "headline matrix functions")
        == EXPECTED_FUNCTIONS
        and parse_int(matrix.get("systems"), "headline matrix systems")
        == len(EXPECTED_SYSTEM_NAMES)
        and parse_int(matrix.get("workloads"), "headline matrix workloads")
        == EXPECTED_WORKLOADS
        and parse_int(matrix.get("ledger_rows"), "headline matrix ledger_rows")
        == EXPECTED_LEDGER_ROWS
        and matrix.get("system_names") == list(EXPECTED_SYSTEM_NAMES)
    ):
        fail("headline manifest fixed matrix contract differs")
    estimator = manifest.get("estimator_contract")
    if not isinstance(estimator, dict) or not (
        parse_int(
            estimator.get("retained_samples_per_displayed_metric"),
            "headline estimator retained sample count",
        )
        == EXPECTED_SAMPLES_PER_METRIC
        and parse_int(
            estimator.get("successful_cells"),
            "headline estimator successful_cells",
        )
        == EXPECTED_SUCCESSFUL_CELLS
        and parse_int(
            estimator.get("displayed_metrics"),
            "headline estimator displayed_metrics",
        )
        == EXPECTED_DISPLAYED_METRICS
        and parse_int(
            estimator.get("portable_raw_rows"),
            "headline estimator portable_raw_rows",
        )
        == EXPECTED_RAW_ROWS
    ):
        fail("headline manifest fixed estimator counts differ")
    if release_mode:
        if estimator.get("display_estimator") != V15_DISPLAY_ESTIMATOR:
            fail("headline manifest mixed display estimator differs")
        if estimator.get("policy") != V15_ESTIMATOR_POLICY:
            fail("headline manifest mixed estimator policy differs")
    elif estimator.get("display_estimator") != "minimum_of_five":
        fail("V14 fixture headline display estimator differs")
    gates = manifest.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        fail("headline manifest contains a failed semantic gate")
    input_hashes = manifest.get("input_sha256")
    output_hashes = manifest.get("output_sha256")
    expected_input_names = {
        LEDGER,
        PAIRWISE_SOTA,
        PAIRWISE_BASELINE,
        "final_13_numeric_summary.json",
        (
            V15_ESTIMATOR_MANIFEST
            if release_mode
            else V14_ESTIMATOR_MANIFEST
        ),
    }
    expected_output_names = {
        HEADLINE_CSV,
        "paper_table_headline_results.tex",
        "paper_table_headline_results_README.md",
    }
    if (
        not isinstance(input_hashes, dict)
        or set(input_hashes) != expected_input_names
        or not isinstance(output_hashes, dict)
        or set(output_hashes) != expected_output_names
    ):
        fail("headline manifest exact input/output filename contract differs")
    validate_hash_map(asset_dir, input_hashes, "headline inputs")
    validate_hash_map(asset_dir, output_hashes, "headline outputs")

    rows = read_csv(csv_path)
    require_columns(
        rows,
        (
            "comparison_id",
            "display_label",
            "metric",
            "comparison_set",
            "timing_boundary",
            "eggpu_validated_workloads",
            "common_pairs",
            "eggpu_strict_wins",
            "eggpu_ties",
            "eggpu_losses",
            "eggpu_only_vs_set",
            "geomean_speedup",
            "display_speedup",
            "display_estimator",
            "tie_tolerance_relative",
        ),
        HEADLINE_CSV,
    )
    by_id = {str(row["comparison_id"]): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(HEADLINE_IDS):
        fail(
            f"{HEADLINE_CSV} must contain exactly the three headline groups: "
            f"{list(HEADLINE_IDS)}"
        )
    typed = [typed_headline(by_id[name]) for name in HEADLINE_IDS]
    expected_row_estimator = (
        V15_DISPLAY_ESTIMATOR if release_mode else "minimum_of_five"
    )
    if any(
        row["display_estimator"] != expected_row_estimator for row in typed
    ):
        fail(f"{HEADLINE_CSV} display estimator differs")
    if any(
        row["eggpu_validated_workloads"] != EXPECTED_WORKLOADS for row in typed
    ):
        fail("headline EGGPU workload denominator differs")

    manifest_rows = manifest.get("rows")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != len(typed):
        fail("headline manifest rows differ from the headline CSV")
    manifest_by_id = {
        str(row.get("comparison_id")): row
        for row in manifest_rows
        if isinstance(row, dict)
    }
    if set(manifest_by_id) != set(HEADLINE_IDS):
        fail("headline manifest comparison IDs differ")
    for row in typed:
        attested = manifest_by_id[str(row["comparison_id"])]
        for key, actual in row.items():
            expected = attested.get(key)
            if isinstance(actual, (int, float)) and not isinstance(actual, bool):
                require_close(actual, expected, f"headline row {row['comparison_id']}.{key}")
            elif actual != expected:
                fail(
                    f"headline row {row['comparison_id']}.{key}: "
                    f"{actual!r} != {expected!r}"
                )
    return typed, manifest


def summary_values(row: Mapping[str, str], source: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for name in (
        "common_strict_wins",
        "common_ties",
        "common_losses",
        "common_pairs",
        "sole_validated",
        "graphscope_validated",
    ):
        values[name] = parse_int(row[name], f"{source}.{name}")
    common_pairs = int(values["common_pairs"])
    if (
        int(values["common_strict_wins"])
        + int(values["common_ties"])
        + int(values["common_losses"])
        != common_pairs
    ):
        fail(f"{source}: wins + ties + losses != common_pairs")
    speedup = parse_float(
        row["speedup_over_best_competitor"],
        f"{source}.speedup_over_best_competitor",
        positive=True,
        nullable=True,
    )
    if common_pairs == 0 and speedup is not None:
        fail(f"{source}: zero common pairs cannot have a speedup")
    if common_pairs > 0 and speedup is None and row["category"] != "Overall":
        fail(f"{source}: nonzero family common pairs require a speedup")
    values["geomean_speedup_over_best_competitor"] = speedup
    return values


def derive_family_summary(
    asset_dir: Path, headlines: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows = read_csv(require_file(asset_dir, FAMILY_SUMMARY))
    require_columns(
        rows,
        (
            "metric",
            "category",
            "common_strict_wins",
            "common_ties",
            "common_losses",
            "common_pairs",
            "sole_validated",
            "graphscope_validated",
            "speedup_over_best_competitor",
        ),
        FAMILY_SUMMARY,
    )
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row["metric"]), str(row["category"]))
        if key in by_key:
            fail(f"{FAMILY_SUMMARY} has a duplicate row: {key}")
        by_key[key] = row
    expected = {
        (metric, category)
        for metric in METRICS
        for category in (*FAMILIES, "Overall")
    }
    if set(by_key) != expected:
        fail(f"{FAMILY_SUMMARY} metric/family matrix differs")

    typed: dict[tuple[str, str], dict[str, object]] = {
        key: summary_values(row, f"{FAMILY_SUMMARY}{key}")
        for key, row in by_key.items()
    }

    pairwise = read_csv(require_file(asset_dir, PAIRWISE_SOTA))
    require_columns(
        pairwise,
        (
            "dataset",
            "function",
            "category",
            "metric",
            "eggpu_status",
            "eggpu_seconds",
            "competitors",
            "best_competitor",
            "best_competitor_seconds",
            "fastest_tied_or_only",
            "only_validated",
            "common_strict_win",
            "common_tie",
            "common_loss",
            "pair_outcome",
            "speedup",
        ),
        PAIRWISE_SOTA,
    )
    pairwise_keys: set[tuple[str, str, str]] = set()
    for row in pairwise:
        key = (str(row["dataset"]), str(row["function"]), str(row["metric"]))
        if key in pairwise_keys:
            fail(f"{PAIRWISE_SOTA} has a duplicate key: {key}")
        pairwise_keys.add(key)
        if row["metric"] not in METRICS or row["category"] not in FAMILIES:
            fail(f"{PAIRWISE_SOTA} contains an unexpected metric/family: {key}")
        if row["eggpu_status"] != "ok":
            fail(f"{PAIRWISE_SOTA} EGGPU row is not validated: {key}")

    ledger_rows = read_csv(require_file(asset_dir, LEDGER))
    ledger = ledger_index(ledger_rows)
    ledger_datasets = {str(row["dataset"]) for row in ledger_rows}
    ledger_functions = {str(row["function"]) for row in ledger_rows}
    expected_pairwise_keys = {
        (dataset, function, metric)
        for dataset in ledger_datasets
        for function in ledger_functions
        for metric in METRICS
    }
    if pairwise_keys != expected_pairwise_keys:
        fail(f"{PAIRWISE_SOTA} does not cover the exact workload/metric matrix")
    comparable_systems = {
        "e2e": (
            "networkx",
            "easygraph-cpu",
            "easygraph-cpp",
            "igraph",
            "GraphScope",
            "nx-cugraph",
        ),
        "kernel": ("nx-cugraph", "Gunrock"),
    }
    for row in pairwise:
        dataset = str(row["dataset"])
        function = str(row["function"])
        metric = str(row["metric"])
        key = (dataset, function, metric)
        eggpu_row = ledger.get((dataset, function, "EGGPU"))
        if eggpu_row is None:
            fail(f"{LEDGER} lacks EGGPU row for {key}")
        eggpu_seconds = parse_float(
            eggpu_row[f"{metric}_paper_seconds"],
            f"{LEDGER}.{key}.EGGPU",
            positive=True,
        )
        declared_eggpu_seconds = parse_float(
            row["eggpu_seconds"],
            f"{PAIRWISE_SOTA}.{key}.eggpu_seconds",
            positive=True,
        )
        assert eggpu_seconds is not None and declared_eggpu_seconds is not None
        require_close(
            declared_eggpu_seconds,
            eggpu_seconds,
            f"{PAIRWISE_SOTA}.{key}.ledger EGGPU seconds",
        )
        candidates: list[tuple[str, float]] = []
        for baseline in comparable_systems[metric]:
            candidate = ledger.get((dataset, function, baseline))
            if (
                candidate is None
                or candidate["execution_status"] != "ok"
                or candidate["validation_status"] not in VALIDATION_OK
            ):
                continue
            seconds = parse_float(
                candidate[f"{metric}_paper_seconds"],
                f"{LEDGER}.{key}.{baseline}",
                positive=True,
                nullable=True,
            )
            if seconds is not None:
                candidates.append((baseline, seconds))
        if parse_int(
            row["competitors"], f"{PAIRWISE_SOTA}.{key}.competitors"
        ) != len(candidates):
            fail(f"{PAIRWISE_SOTA}.{key} competitor count differs from ledger")
        only = parse_bool(
            row["only_validated"], f"{PAIRWISE_SOTA}.{key}.only_validated"
        )
        fastest = parse_bool(
            row["fastest_tied_or_only"],
            f"{PAIRWISE_SOTA}.{key}.fastest_tied_or_only",
        )
        flags = {
            "strict_win": parse_bool(
                row["common_strict_win"],
                f"{PAIRWISE_SOTA}.{key}.common_strict_win",
            ),
            "tie": parse_bool(
                row["common_tie"], f"{PAIRWISE_SOTA}.{key}.common_tie"
            ),
            "loss": parse_bool(
                row["common_loss"], f"{PAIRWISE_SOTA}.{key}.common_loss"
            ),
        }
        if not candidates:
            if not (
                only
                and fastest
                and row["pair_outcome"] == "sole_validated"
                and not any(flags.values())
                and not str(row["best_competitor"]).strip()
                and not str(row["best_competitor_seconds"]).strip()
                and not str(row["speedup"]).strip()
            ):
                fail(f"{PAIRWISE_SOTA}.{key} sole-validated row differs")
            continue
        best_seconds = min(seconds for _, seconds in candidates)
        best_names = {
            baseline
            for baseline, seconds in candidates
            if math.isclose(
                seconds,
                best_seconds,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        }
        declared_best = str(row["best_competitor"])
        if (
            declared_best not in best_names
            or declared_best in ARCHIVED_SINGLE_SOURCE_SYSTEMS
        ):
            fail(f"{PAIRWISE_SOTA}.{key} best competitor differs from ledger")
        declared_best_seconds = parse_float(
            row["best_competitor_seconds"],
            f"{PAIRWISE_SOTA}.{key}.best_competitor_seconds",
            positive=True,
        )
        speedup = parse_float(
            row["speedup"], f"{PAIRWISE_SOTA}.{key}.speedup", positive=True
        )
        assert declared_best_seconds is not None and speedup is not None
        require_close(
            declared_best_seconds,
            best_seconds,
            f"{PAIRWISE_SOTA}.{key}.ledger best seconds",
        )
        tolerance = RELATIVE_TIE_TOLERANCE * max(
            abs(eggpu_seconds), abs(best_seconds), 1.0e-15
        )
        difference = eggpu_seconds - best_seconds
        outcome = (
            "strict_win"
            if difference < -tolerance
            else ("tie" if abs(difference) <= tolerance else "loss")
        )
        if (
            only
            or row["pair_outcome"] != outcome
            or sum(flags.values()) != 1
            or flags[outcome] is not True
            or fastest != (outcome in {"strict_win", "tie"})
        ):
            fail(f"{PAIRWISE_SOTA}.{key} outcome differs from ledger")
        require_close(
            speedup,
            best_seconds / eggpu_seconds,
            f"{PAIRWISE_SOTA}.{key}.ledger speedup",
        )
    derived: dict[tuple[str, str], dict[str, object]] = {}
    for metric in METRICS:
        for family in FAMILIES:
            selected = [
                row
                for row in pairwise
                if row["metric"] == metric and row["category"] == family
            ]
            if not selected:
                fail(f"{PAIRWISE_SOTA} has no rows for {metric}/{family}")
            counts = {
                "common_strict_wins": 0,
                "common_ties": 0,
                "common_losses": 0,
                "common_pairs": 0,
                "sole_validated": 0,
                "graphscope_validated": 0,
            }
            speedups: list[float] = []
            for row in selected:
                key = (
                    str(row["dataset"]),
                    str(row["function"]),
                    str(row["metric"]),
                )
                flags = {
                    "common_strict_wins": parse_bool(
                        row["common_strict_win"],
                        f"{PAIRWISE_SOTA}{key}.common_strict_win",
                    ),
                    "common_ties": parse_bool(
                        row["common_tie"],
                        f"{PAIRWISE_SOTA}{key}.common_tie",
                    ),
                    "common_losses": parse_bool(
                        row["common_loss"],
                        f"{PAIRWISE_SOTA}{key}.common_loss",
                    ),
                }
                only = parse_bool(
                    row["only_validated"],
                    f"{PAIRWISE_SOTA}{key}.only_validated",
                )
                common_flag_count = sum(flags.values())
                if common_flag_count + int(only) != 1:
                    fail(
                        f"{PAIRWISE_SOTA}{key}: workload must be exactly one "
                        "of strict-win/tie/loss/sole-validated"
                    )
                if only:
                    counts["sole_validated"] += 1
                else:
                    counts["common_pairs"] += 1
                    for name, enabled in flags.items():
                        counts[name] += int(enabled)
                    speedup = parse_float(
                        row["speedup"],
                        f"{PAIRWISE_SOTA}{key}.speedup",
                        positive=True,
                    )
                    assert speedup is not None
                    speedups.append(speedup)

                graphscope = ledger.get(
                    (str(row["dataset"]), str(row["function"]), "GraphScope")
                )
                if graphscope is None:
                    fail(
                        f"{LEDGER} lacks GraphScope row for "
                        f"{row['dataset']}/{row['function']}"
                    )
                graphscope_seconds = parse_float(
                    graphscope[f"{metric}_paper_seconds"],
                    (
                        f"{LEDGER}.{row['dataset']}.{row['function']}."
                        f"GraphScope.{metric}"
                    ),
                    positive=True,
                    nullable=True,
                )
                if (
                    graphscope["execution_status"] == "ok"
                    and graphscope["validation_status"] in VALIDATION_OK
                    and graphscope_seconds is not None
                ):
                    counts["graphscope_validated"] += 1
            derived[(metric, family)] = {
                **counts,
                "geomean_speedup_over_best_competitor": geomean_numbers(
                    speedups, f"{PAIRWISE_SOTA}.{metric}.{family}.speedups"
                ),
            }

    for metric in METRICS:
        family_rows = [derived[(metric, family)] for family in FAMILIES]
        all_speedups = [
            parse_float(
                row["speedup"],
                f"{PAIRWISE_SOTA}.{metric}.overall.speedup",
                positive=True,
            )
            for row in pairwise
            if row["metric"] == metric
            and not parse_bool(
                row["only_validated"],
                f"{PAIRWISE_SOTA}.{metric}.overall.only_validated",
            )
        ]
        derived[(metric, "Overall")] = {
            name: sum(int(row[name]) for row in family_rows)
            for name in (
                "common_strict_wins",
                "common_ties",
                "common_losses",
                "common_pairs",
                "sole_validated",
                "graphscope_validated",
            )
        }
        derived[(metric, "Overall")][
            "geomean_speedup_over_best_competitor"
        ] = (
            geomean_numbers(
                [value for value in all_speedups if value is not None],
                f"{PAIRWISE_SOTA}.{metric}.overall.speedups",
            )
            if metric == "e2e"
            else None
        )

    for key, observed in typed.items():
        expected_values = derived[key]
        for name, expected_value in expected_values.items():
            observed_value = observed[name]
            if isinstance(expected_value, float):
                require_close(
                    observed_value,
                    expected_value,
                    f"{FAMILY_SUMMARY}{key}.{name}",
                )
            elif observed_value != expected_value:
                fail(
                    f"{FAMILY_SUMMARY}{key}.{name}: "
                    f"{observed_value!r} != pairwise/ledger-derived "
                    f"{expected_value!r}"
                )

    headline_by_id = {
        str(row["comparison_id"]): row for row in headlines
    }
    for metric, headline_id in (
        ("e2e", "library_public_call_e2e"),
        ("kernel", "native_gpu_processing"),
    ):
        overall = typed[(metric, "Overall")]
        headline = headline_by_id[headline_id]
        for summary_name, headline_name in (
            ("common_strict_wins", "eggpu_strict_wins"),
            ("common_ties", "eggpu_ties"),
            ("common_losses", "eggpu_losses"),
            ("common_pairs", "common_pairs"),
            ("sole_validated", "eggpu_only_vs_set"),
        ):
            if overall[summary_name] != headline[headline_name]:
                fail(
                    f"{metric} family Overall.{summary_name} differs from "
                    f"headline {headline_id}.{headline_name}"
                )

    families: list[dict[str, object]] = []
    for family in FAMILIES:
        families.append(
            {
                "family": family,
                "e2e": typed[("e2e", family)],
                "device": {
                    "source_metric": "kernel",
                    **typed[("kernel", family)],
                },
            }
        )
    overall_by_metric = {
        metric: typed[(metric, "Overall")] for metric in METRICS
    }
    return families, overall_by_metric


def ledger_index(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str, str], Mapping[str, str]]:
    require_columns(
        rows,
        (
            "dataset",
            "function",
            "baseline",
            "support_class",
            "execution_status",
            "validation_status",
            "build_paper_seconds",
            "build_std_seconds",
            "kernel_paper_seconds",
            "kernel_std_seconds",
            "e2e_paper_seconds",
            "e2e_std_seconds",
        ),
        LEDGER,
    )
    result: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["function"]),
            str(row["baseline"]),
        )
        if key in result:
            fail(f"{LEDGER} has a duplicate key: {key}")
        result[key] = row
    return result


def derive_e2e_losses(
    asset_dir: Path,
    ledger: Mapping[tuple[str, str, str], Mapping[str, str]],
    expected_loss_count: int,
) -> list[dict[str, object]]:
    rows = read_csv(require_file(asset_dir, PAIRWISE_SOTA))
    require_columns(
        rows,
        (
            "dataset",
            "function",
            "category",
            "metric",
            "eggpu_status",
            "eggpu_seconds",
            "best_competitor",
            "best_competitor_seconds",
            "common_loss",
            "pair_outcome",
            "speedup",
        ),
        PAIRWISE_SOTA,
    )
    keys: set[tuple[str, str, str]] = set()
    losses: list[Mapping[str, str]] = []
    for row in rows:
        key = (str(row["dataset"]), str(row["function"]), str(row["metric"]))
        if key in keys:
            fail(f"{PAIRWISE_SOTA} has a duplicate key: {key}")
        keys.add(key)
        common_loss = parse_bool(
            row["common_loss"], f"{PAIRWISE_SOTA}{key}.common_loss"
        )
        if row["metric"] == "e2e" and common_loss:
            if row["pair_outcome"] != "loss" or row["eggpu_status"] != "ok":
                fail(f"{PAIRWISE_SOTA}{key}: inconsistent E2E loss status")
            losses.append(row)
    if len(losses) != expected_loss_count:
        fail(
            f"{PAIRWISE_SOTA}: derived {len(losses)} E2E losses, "
            f"family summary attests {expected_loss_count}"
        )

    result: list[dict[str, object]] = []
    for row in sorted(losses, key=lambda item: (item["dataset"], item["function"])):
        dataset = str(row["dataset"])
        function = str(row["function"])
        eggpu_e2e = parse_float(
            row["eggpu_seconds"],
            f"{PAIRWISE_SOTA}.{dataset}.{function}.eggpu_seconds",
            positive=True,
        )
        competitor_e2e = parse_float(
            row["best_competitor_seconds"],
            f"{PAIRWISE_SOTA}.{dataset}.{function}.best_competitor_seconds",
            positive=True,
        )
        speedup = parse_float(
            row["speedup"],
            f"{PAIRWISE_SOTA}.{dataset}.{function}.speedup",
            positive=True,
        )
        assert eggpu_e2e is not None
        assert competitor_e2e is not None
        assert speedup is not None
        require_close(
            speedup,
            competitor_e2e / eggpu_e2e,
            f"{PAIRWISE_SOTA}.{dataset}.{function}.speedup formula",
        )
        eggpu_row = ledger.get((dataset, function, "EGGPU"))
        if eggpu_row is None:
            fail(f"{LEDGER} lacks EGGPU row for E2E loss {dataset}/{function}")
        if (
            eggpu_row["execution_status"] != "ok"
            or eggpu_row["validation_status"] not in VALIDATION_OK
        ):
            fail(f"EGGPU ledger row is not correctness-valid: {dataset}/{function}")
        ledger_e2e = parse_float(
            eggpu_row["e2e_paper_seconds"],
            f"{LEDGER}.{dataset}.{function}.e2e_paper_seconds",
            positive=True,
        )
        kernel = parse_float(
            eggpu_row["kernel_paper_seconds"],
            f"{LEDGER}.{dataset}.{function}.kernel_paper_seconds",
            positive=True,
        )
        eggpu_e2e_std = parse_float(
            eggpu_row["e2e_std_seconds"],
            f"{LEDGER}.{dataset}.{function}.e2e_std_seconds",
        )
        eggpu_kernel_std = parse_float(
            eggpu_row["kernel_std_seconds"],
            f"{LEDGER}.{dataset}.{function}.kernel_std_seconds",
        )
        competitor = str(row["best_competitor"])
        competitor_row = ledger.get((dataset, function, competitor))
        if (
            competitor_row is None
            or competitor_row["execution_status"] != "ok"
            or competitor_row["validation_status"] not in VALIDATION_OK
        ):
            fail(
                "best-competitor ledger row is not correctness-valid: "
                f"{dataset}/{function}/{competitor}"
            )
        competitor_std = parse_float(
            competitor_row["e2e_std_seconds"],
            (
                f"{LEDGER}.{dataset}.{function}.{competitor}."
                "e2e_std_seconds"
            ),
        )
        assert ledger_e2e is not None and kernel is not None
        assert eggpu_e2e_std is not None and eggpu_kernel_std is not None
        assert competitor_std is not None
        competitor_ledger_e2e = parse_float(
            competitor_row["e2e_paper_seconds"],
            (
                f"{LEDGER}.{dataset}.{function}.{competitor}."
                "e2e_paper_seconds"
            ),
            positive=True,
        )
        assert competitor_ledger_e2e is not None
        require_close(
            eggpu_e2e,
            ledger_e2e,
            f"pairwise/ledger EGGPU E2E {dataset}/{function}",
        )
        require_close(
            competitor_e2e,
            competitor_ledger_e2e,
            (
                "pairwise/ledger best-competitor E2E "
                f"{dataset}/{function}/{competitor}"
            ),
        )
        result.append(
            {
                "dataset": dataset,
                "function": function,
                "family": str(row["category"]),
                "best_competitor": competitor,
                "best_competitor_e2e_seconds": competitor_e2e,
                "best_competitor_e2e_sample_std_seconds": competitor_std,
                "eggpu_e2e_seconds": ledger_e2e,
                "eggpu_e2e_sample_std_seconds": eggpu_e2e_std,
                "eggpu_kernel_seconds": kernel,
                "eggpu_kernel_sample_std_seconds": eggpu_kernel_std,
                "speedup": speedup,
                "inverse_speedup": 1.0 / speedup,
                "eggpu_kernel_over_e2e": kernel / ledger_e2e,
            }
        )
    return result


def derive_baseline_summaries(
    asset_dir: Path, expected_functions: set[str]
) -> list[dict[str, object]]:
    rows = read_csv(require_file(asset_dir, PAIRWISE_BASELINE))
    require_columns(
        rows,
        (
            "baseline",
            "function",
            "metric",
            "common_pairs",
            "eggpu_wins",
            "eggpu_ties",
            "eggpu_losses",
            "geomean_speedup",
            "eggpu_geomean_seconds",
            "baseline_geomean_seconds",
        ),
        PAIRWISE_BASELINE,
    )
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row["baseline"]), str(row["function"]), str(row["metric"]))
        if key in keys:
            fail(f"{PAIRWISE_BASELINE} has a duplicate key: {key}")
        keys.add(key)
    expected_baselines = set(EXPECTED_SYSTEM_NAMES) - {"EGGPU"}
    expected_keys = {
        (baseline, function, metric)
        for baseline in expected_baselines
        for function in expected_functions
        for metric in METRICS
    }
    if keys != expected_keys:
        fail(
            f"{PAIRWISE_BASELINE} does not cover the exact current-system "
            "baseline/function/metric matrix"
        )
    ledger_rows = read_csv(require_file(asset_dir, LEDGER))
    ledger = ledger_index(ledger_rows)
    datasets = sorted({str(row["dataset"]) for row in ledger_rows})

    summaries: list[dict[str, object]] = []
    for baseline in BASELINES:
        metric_summaries: dict[str, object] = {}
        for metric in METRICS:
            selected = [
                row
                for row in rows
                if row["baseline"] == baseline and row["metric"] == metric
            ]
            if {str(row["function"]) for row in selected} != expected_functions:
                fail(
                    f"{PAIRWISE_BASELINE}: {baseline}/{metric} does not cover "
                    "the complete function set"
                )
            for row in selected:
                source = (
                    f"{PAIRWISE_BASELINE}.{baseline}.{metric}."
                    f"{row['function']}"
                )
                common_pairs = parse_int(row["common_pairs"], f"{source}.common")
                wins = parse_int(row["eggpu_wins"], f"{source}.wins")
                ties = parse_int(row["eggpu_ties"], f"{source}.ties")
                losses = parse_int(row["eggpu_losses"], f"{source}.losses")
                if wins + ties + losses != common_pairs:
                    fail(f"{source}: wins + ties + losses != common pairs")
                derived_pairs: list[tuple[float, float]] = []
                if not (baseline == "Gunrock" and metric == "e2e"):
                    for dataset in datasets:
                        eggpu_row = ledger.get((dataset, row["function"], "EGGPU"))
                        baseline_row = ledger.get(
                            (dataset, row["function"], baseline)
                        )
                        if eggpu_row is None or baseline_row is None:
                            fail(
                                f"{LEDGER} lacks pair for "
                                f"{dataset}/{row['function']}/{baseline}"
                            )
                        pair_seconds: list[float] = []
                        for system, ledger_row in (
                            ("EGGPU", eggpu_row),
                            (baseline, baseline_row),
                        ):
                            if (
                                ledger_row["execution_status"] != "ok"
                                or ledger_row["validation_status"]
                                not in VALIDATION_OK
                            ):
                                break
                            seconds = parse_float(
                                ledger_row[f"{metric}_paper_seconds"],
                                (
                                    f"{LEDGER}.{dataset}.{row['function']}."
                                    f"{system}.{metric}"
                                ),
                                positive=True,
                                nullable=True,
                            )
                            if seconds is None:
                                break
                            pair_seconds.append(seconds)
                        if len(pair_seconds) == 2:
                            derived_pairs.append(
                                (pair_seconds[0], pair_seconds[1])
                            )
                derived_outcomes = {"wins": 0, "ties": 0, "losses": 0}
                for eggpu_seconds, baseline_seconds in derived_pairs:
                    tolerance = RELATIVE_TIE_TOLERANCE * max(
                        abs(eggpu_seconds),
                        abs(baseline_seconds),
                        1.0e-15,
                    )
                    difference = eggpu_seconds - baseline_seconds
                    if difference < -tolerance:
                        derived_outcomes["wins"] += 1
                    elif abs(difference) <= tolerance:
                        derived_outcomes["ties"] += 1
                    else:
                        derived_outcomes["losses"] += 1
                if (
                    common_pairs != len(derived_pairs)
                    or wins != derived_outcomes["wins"]
                    or ties != derived_outcomes["ties"]
                    or losses != derived_outcomes["losses"]
                ):
                    fail(f"{source}: summary W/T/L differs from ledger pairs")
                derived_speedup = geomean_numbers(
                    [
                        baseline_seconds / eggpu_seconds
                        for eggpu_seconds, baseline_seconds in derived_pairs
                    ],
                    f"{source}.ledger speedups",
                )
                derived_eggpu_seconds = geomean_numbers(
                    [eggpu_seconds for eggpu_seconds, _ in derived_pairs],
                    f"{source}.ledger EGGPU seconds",
                )
                derived_baseline_seconds = geomean_numbers(
                    [baseline_seconds for _, baseline_seconds in derived_pairs],
                    f"{source}.ledger baseline seconds",
                )
                if not derived_pairs:
                    for column in (
                        "geomean_speedup",
                        "eggpu_geomean_seconds",
                        "baseline_geomean_seconds",
                    ):
                        if str(row[column]).strip():
                            fail(f"{source}: zero pairs have nonempty {column}")
                    continue
                observed_speedup = parse_float(
                    row["geomean_speedup"],
                    f"{source}.geomean_speedup",
                    positive=True,
                )
                observed_eggpu_seconds = parse_float(
                    row["eggpu_geomean_seconds"],
                    f"{source}.eggpu_geomean_seconds",
                    positive=True,
                )
                observed_baseline_seconds = parse_float(
                    row["baseline_geomean_seconds"],
                    f"{source}.baseline_geomean_seconds",
                    positive=True,
                )
                assert observed_speedup is not None
                assert observed_eggpu_seconds is not None
                assert observed_baseline_seconds is not None
                assert derived_speedup is not None
                assert derived_eggpu_seconds is not None
                assert derived_baseline_seconds is not None
                require_close(
                    observed_speedup,
                    derived_speedup,
                    f"{source}.ledger-derived speedup",
                )
                require_close(
                    observed_eggpu_seconds,
                    derived_eggpu_seconds,
                    f"{source}.ledger-derived EGGPU seconds",
                )
                require_close(
                    observed_baseline_seconds,
                    derived_baseline_seconds,
                    f"{source}.ledger-derived baseline seconds",
                )
            totals = {
                name: sum(
                    parse_int(
                        row[column],
                        f"{PAIRWISE_BASELINE}.{baseline}.{metric}.{column}",
                    )
                    for row in selected
                )
                for name, column in (
                    ("common_pairs", "common_pairs"),
                    ("eggpu_wins", "eggpu_wins"),
                    ("eggpu_ties", "eggpu_ties"),
                    ("eggpu_losses", "eggpu_losses"),
                )
            }
            if (
                totals["eggpu_wins"]
                + totals["eggpu_ties"]
                + totals["eggpu_losses"]
                != totals["common_pairs"]
            ):
                fail(
                    f"{PAIRWISE_BASELINE}: {baseline}/{metric} W/T/L totals "
                    "do not equal common pairs"
                )
            common = [row for row in selected if int(float(row["common_pairs"])) > 0]
            speedup = weighted_geomean(
                selected,
                "geomean_speedup",
                f"{PAIRWISE_BASELINE}.{baseline}.{metric}",
            )
            eggpu_seconds = weighted_geomean(
                selected,
                "eggpu_geomean_seconds",
                f"{PAIRWISE_BASELINE}.{baseline}.{metric}",
            )
            baseline_seconds = weighted_geomean(
                selected,
                "baseline_geomean_seconds",
                f"{PAIRWISE_BASELINE}.{baseline}.{metric}",
            )
            if speedup is not None:
                assert eggpu_seconds is not None and baseline_seconds is not None
                require_close(
                    speedup,
                    baseline_seconds / eggpu_seconds,
                    f"{baseline}/{metric} aggregate speedup",
                )
            metric_summaries[metric] = {
                **totals,
                "functions_with_common_pairs": len(common),
                "geomean_speedup": speedup,
                "eggpu_geomean_seconds": eggpu_seconds,
                "baseline_geomean_seconds": baseline_seconds,
            }
        summaries.append({"baseline": baseline, "metrics": metric_summaries})
    return summaries


def derive_gap_twitter_bfs(
    ledger_rows: Sequence[Mapping[str, str]], system_order: Sequence[str]
) -> dict[str, object]:
    selected = [
        row
        for row in ledger_rows
        if row["dataset"] == "GAP-twitter" and row["function"] == "BFS"
    ]
    by_system = {str(row["baseline"]): row for row in selected}
    if len(by_system) != len(selected) or set(by_system) != set(system_order):
        fail(f"{LEDGER}: GAP-twitter/BFS does not cover the complete system set")
    systems: list[dict[str, object]] = []
    validated_triplets = 0
    for baseline in system_order:
        row = by_system[baseline]
        record: dict[str, object] = {
            "baseline": baseline,
            "support_class": str(row["support_class"]),
            "execution_status": str(row["execution_status"]),
            "validation_status": str(row["validation_status"]),
            "triplet": None,
        }
        if row["execution_status"] == "ok":
            if row["validation_status"] not in VALIDATION_OK:
                fail(f"GAP-twitter/BFS {baseline} is successful but unvalidated")
            build = parse_float(
                row["build_paper_seconds"],
                f"{LEDGER}.GAP-twitter.BFS.{baseline}.build",
                positive=True,
            )
            kernel = parse_float(
                row["kernel_paper_seconds"],
                f"{LEDGER}.GAP-twitter.BFS.{baseline}.kernel",
                positive=True,
            )
            e2e = parse_float(
                row["e2e_paper_seconds"],
                f"{LEDGER}.GAP-twitter.BFS.{baseline}.e2e",
                positive=True,
            )
            build_std = parse_float(
                row["build_std_seconds"],
                f"{LEDGER}.GAP-twitter.BFS.{baseline}.build_std",
            )
            kernel_std = parse_float(
                row["kernel_std_seconds"],
                f"{LEDGER}.GAP-twitter.BFS.{baseline}.kernel_std",
            )
            e2e_std = parse_float(
                row["e2e_std_seconds"],
                f"{LEDGER}.GAP-twitter.BFS.{baseline}.e2e_std",
            )
            record["triplet"] = {
                "graph_construction_seconds": build,
                "graph_construction_sample_std_seconds": build_std,
                "processing_seconds": kernel,
                "processing_sample_std_seconds": kernel_std,
                "end_to_end_seconds": e2e,
                "end_to_end_sample_std_seconds": e2e_std,
            }
            validated_triplets += 1
        systems.append(record)
    if validated_triplets == 0:
        fail(f"{LEDGER}: GAP-twitter/BFS has no correctness-valid timing triplet")
    return {
        "dataset": "GAP-twitter",
        "function": "BFS",
        "correctness_valid_triplet_count": validated_triplets,
        "systems": systems,
    }


def derive_figure3(
    asset_dir: Path, *, release_mode: bool
) -> dict[str, object]:
    metadata_path = require_file(asset_dir, FIGURE3_METADATA)
    metadata = read_json(metadata_path)
    if metadata.get("status") != "complete":
        fail(f"{FIGURE3_METADATA} status is not complete")
    for key in ("interpretation", "ranking_source", "boundary_annotations"):
        if not metadata.get(key):
            fail(f"{FIGURE3_METADATA} lacks {key}")
    rows = read_csv(require_file(asset_dir, FIGURE3_DATA))
    required_columns = ["metric", "baseline"]
    if release_mode:
        required_columns.extend(
            ("sample_std_seconds", "aggregate_sample_count")
        )
    require_columns(rows, required_columns, FIGURE3_DATA)
    observed_titles = tuple(dict.fromkeys(str(row["metric"]) for row in rows))
    if observed_titles != FIGURE3_TITLES:
        fail(
            f"Figure 3 panel titles differ: {observed_titles!r} != "
            f"{FIGURE3_TITLES!r}"
        )
    if "panel_titles" in metadata:
        panel_titles = metadata["panel_titles"]
        if panel_titles != list(FIGURE3_TITLES):
            fail(f"{FIGURE3_METADATA}.panel_titles differs from the fixed titles")
    figure_systems = {str(row["baseline"]) for row in rows}
    archived_series = sorted(
        figure_systems.intersection(ARCHIVED_SINGLE_SOURCE_SYSTEMS)
    )
    if release_mode:
        if archived_series:
            fail(
                f"{FIGURE3_DATA} mixes archived single-source systems into "
                "the current five-run panel"
            )
        error_bars = metadata.get("error_bars")
        point_estimator = metadata.get("point_estimator")
        if not (
            isinstance(error_bars, dict)
            and error_bars.get("statistic") == V15_ERROR_BAR
            and parse_int(
                error_bars.get("samples"),
                f"{FIGURE3_METADATA}.error_bars.samples",
            )
            == EXPECTED_SAMPLES_PER_METRIC
            and str(error_bars.get("aggregation", "")).strip()
            and isinstance(point_estimator, dict)
            and point_estimator.get("current_EGGPU")
            == "minimum_of_five"
            and point_estimator.get("external_baselines")
            == "arithmetic_mean_of_five"
            and point_estimator.get("EGGPU-2024_starred_archive")
            == "single_source_historical"
        ):
            fail(f"{FIGURE3_METADATA} mixed estimator/error-bar contract differs")
        for index, row in enumerate(rows):
            sample_sd = parse_float(
                row["sample_std_seconds"],
                f"{FIGURE3_DATA}.row-{index}.sample_std_seconds",
            )
            if sample_sd is None or sample_sd < 0.0:
                fail(f"{FIGURE3_DATA}.row-{index} sample SD is invalid")
            if (
                parse_int(
                    row["aggregate_sample_count"],
                    f"{FIGURE3_DATA}.row-{index}.aggregate_sample_count",
                )
                != EXPECTED_SAMPLES_PER_METRIC
            ):
                fail(f"{FIGURE3_DATA}.row-{index} is not a five-run aggregate")
    return {
        "metadata_status": "complete",
        "panel_titles": list(FIGURE3_TITLES),
        "panel_titles_source": (
            f"{FIGURE3_DATA} metric order, under the complete "
            f"{FIGURE3_METADATA} gate"
        ),
        "metadata_sha256": sha256_file(metadata_path),
        "data_sha256": sha256_file(asset_dir / FIGURE3_DATA),
        "interpretation": metadata["interpretation"],
        "ranking_source": metadata["ranking_source"],
        "boundary_annotations": metadata["boundary_annotations"],
        "archived_single_source_series": archived_series,
        "error_bar_contract": (
            metadata.get("error_bars") if release_mode else None
        ),
        "point_estimator_contract": (
            metadata.get("point_estimator") if release_mode else None
        ),
        "archived_series_policy": (
            "EGGPU-2024, when present, is a labeled archived single-source "
            "reference and is excluded from the current five-run mixed-"
            "estimator ledger and all current W/T/L or speedup derivations."
        ),
    }


def derive_counts(
    ledger_rows: Sequence[Mapping[str, str]],
    asset_dir: Path,
    raw_samples_name: str,
) -> dict[str, object]:
    successful = [row for row in ledger_rows if row["execution_status"] == "ok"]
    displayed_metrics = 0
    for row in successful:
        for metric in ("build", "kernel", "e2e"):
            value = parse_float(
                row[f"{metric}_paper_seconds"],
                f"{LEDGER}.{metric}_paper_seconds",
                positive=True,
            )
            if value is not None:
                displayed_metrics += 1
    raw_rows = read_csv(require_file(asset_dir, raw_samples_name))
    eggpu_cells = sum(row["baseline"] == "EGGPU" for row in ledger_rows)
    return {
        "ledger_rows": len(ledger_rows),
        "successful_cells": len(successful),
        "displayed_metrics": displayed_metrics,
        "portable_raw_rows": len(raw_rows),
        "eggpu_cells": eggpu_cells,
        "non_eggpu_rows": len(ledger_rows) - eggpu_cells,
    }


def validate_mixed_estimator_evidence(
    asset_dir: Path,
    ledger_rows: Sequence[Mapping[str, str]],
    counts: Mapping[str, object],
) -> dict[str, object]:
    """Reconstruct every V15 displayed value and error from five raw runs."""
    manifest_path = require_file(asset_dir, V15_ESTIMATOR_MANIFEST)
    raw_path = require_file(asset_dir, V15_RAW_SAMPLES)
    audit_path = require_file(asset_dir, V15_ESTIMATOR_AUDIT)
    manifest = read_json(manifest_path)
    expected_manifest_fields: dict[str, object] = {
        "schema_version": V15_ESTIMATOR_MANIFEST_SCHEMA,
        "status": "pass",
        "display_estimator": V15_DISPLAY_ESTIMATOR,
        "display_estimator_by_baseline_class": (
            V15_ESTIMATOR_BY_BASELINE_CLASS
        ),
        "reported_error": V15_ERROR_BAR,
        "sample_count_per_displayed_metric": EXPECTED_SAMPLES_PER_METRIC,
        "successful_cells": counts["successful_cells"],
        "displayed_metrics": counts["displayed_metrics"],
        "portable_raw_rows": counts["portable_raw_rows"],
    }
    for field, expected in expected_manifest_fields.items():
        if manifest.get(field) != expected:
            fail(
                f"{V15_ESTIMATOR_MANIFEST}.{field}: "
                f"{manifest.get(field)!r} != {expected!r}"
            )
    for field, expected_name in (
        ("portable_raw_samples", V15_RAW_SAMPLES),
        ("overlay_audit", V15_ESTIMATOR_AUDIT),
        ("output_ledger", LEDGER),
        ("base_ledger", V15_OVERLAY_LEDGER),
    ):
        value = manifest.get(field)
        if not isinstance(value, str) or Path(value).name != expected_name:
            fail(
                f"{V15_ESTIMATOR_MANIFEST}.{field} does not identify "
                f"{expected_name}"
            )
    base_ledger_value = Path(str(manifest["base_ledger"]))
    if base_ledger_value.parent.name != V15_AUDIT_DIR:
        fail(
            f"{V15_ESTIMATOR_MANIFEST}.base_ledger does not identify the "
            "V15 unified overlay"
        )
    require_sha256(
        manifest.get("base_ledger_sha256"),
        f"{V15_ESTIMATOR_MANIFEST}.base_ledger_sha256",
    )
    for field, path in (
        ("output_ledger_sha256", asset_dir / LEDGER),
        ("portable_raw_samples_sha256", raw_path),
        ("overlay_audit_sha256", audit_path),
    ):
        expected = require_sha256(
            manifest.get(field), f"{V15_ESTIMATOR_MANIFEST}.{field}"
        )
        actual = sha256_file(path)
        if actual != expected:
            fail(
                f"{V15_ESTIMATOR_MANIFEST}.{field} differs: "
                f"{actual} != {expected}"
            )
    changed = parse_int(
        manifest.get("changed_display_metrics"),
        f"{V15_ESTIMATOR_MANIFEST}.changed_display_metrics",
    )
    unchanged = parse_int(
        manifest.get("unchanged_display_metrics"),
        f"{V15_ESTIMATOR_MANIFEST}.unchanged_display_metrics",
    )
    if changed < 0 or unchanged < 0 or changed + unchanged != counts[
        "displayed_metrics"
    ]:
        fail(f"{V15_ESTIMATOR_MANIFEST} change counts differ")
    if not str(manifest.get("selection_policy", "")).strip():
        fail(f"{V15_ESTIMATOR_MANIFEST}.selection_policy is empty")

    required_ledger_columns = ["sample_count"]
    for metric in ("build", "kernel", "e2e"):
        required_ledger_columns.extend(
            (
                f"{metric}_paper_seconds",
                f"{metric}_raw_mean_seconds",
                f"{metric}_std_seconds",
                f"{metric}_estimator",
                f"{metric}_raw_min_seconds",
                f"{metric}_raw_median_seconds",
                f"{metric}_raw_max_seconds",
                f"{metric}_coefficient_of_variation",
                f"{metric}_max_over_median",
                f"{metric}_median_over_minimum",
                f"{metric}_variance_policy",
                f"{metric}_stability_status",
            )
        )
    require_columns(
        ledger_rows,
        required_ledger_columns,
        LEDGER,
    )
    ledger = ledger_index(ledger_rows)
    successful = {
        key: row
        for key, row in ledger.items()
        if row["execution_status"] == "ok"
    }
    expected_raw_keys = {
        (dataset, function, baseline, metric)
        for dataset, function, baseline in successful
        for metric in ("build", "kernel", "e2e")
    }

    raw_rows = read_csv(raw_path)
    require_columns(
        raw_rows,
        (
            "dataset",
            "function",
            "baseline",
            "metric",
            "sample_index",
            "seconds",
        ),
        V15_RAW_SAMPLES,
    )
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, str]]] = {}
    for row in raw_rows:
        key = (
            str(row["dataset"]),
            str(row["function"]),
            str(row["baseline"]),
            str(row["metric"]),
        )
        grouped.setdefault(key, []).append(row)
    if set(grouped) != expected_raw_keys:
        fail(f"{V15_RAW_SAMPLES} keys differ from successful ledger metrics")

    reconstructed: dict[
        tuple[str, str, str, str], dict[str, float | str]
    ] = {}
    for key in sorted(expected_raw_keys):
        samples = grouped[key]
        indices = [
            parse_int(
                row["sample_index"],
                f"{V15_RAW_SAMPLES}.{key}.sample_index",
            )
            for row in samples
        ]
        if sorted(indices) != list(
            range(1, EXPECTED_SAMPLES_PER_METRIC + 1)
        ):
            fail(f"{V15_RAW_SAMPLES}.{key} sample indices differ")
        values_by_index: dict[int, float] = {}
        for index, row in zip(indices, samples):
            if index in values_by_index:
                fail(f"{V15_RAW_SAMPLES}.{key} has duplicate sample {index}")
            value = parse_float(
                row["seconds"],
                f"{V15_RAW_SAMPLES}.{key}.seconds",
                positive=True,
            )
            assert value is not None
            values_by_index[index] = value
        values = [
            values_by_index[index]
            for index in range(1, EXPECTED_SAMPLES_PER_METRIC + 1)
        ]
        mean = statistics.mean(values)
        sample_sd = statistics.stdev(values)
        minimum = min(values)
        median = statistics.median(values)
        maximum = max(values)
        dataset, function, baseline, metric = key
        ledger_row = successful[(dataset, function, baseline)]
        expected_estimator = (
            "minimum_of_five"
            if baseline == "EGGPU"
            else "arithmetic_mean_of_five"
        )
        displayed = minimum if baseline == "EGGPU" else mean
        if ledger_row[f"{metric}_estimator"] != expected_estimator:
            fail(f"{LEDGER}.{key} estimator policy differs")
        if (
            parse_int(
                ledger_row["sample_count"], f"{LEDGER}.{key}.sample_count"
            )
            != EXPECTED_SAMPLES_PER_METRIC
        ):
            fail(f"{LEDGER}.{key} does not retain five samples")
        for field, expected in (
            (f"{metric}_paper_seconds", displayed),
            (f"{metric}_raw_mean_seconds", mean),
            (f"{metric}_std_seconds", sample_sd),
            (f"{metric}_raw_min_seconds", minimum),
            (f"{metric}_raw_median_seconds", median),
            (f"{metric}_raw_max_seconds", maximum),
            (f"{metric}_coefficient_of_variation", sample_sd / mean),
            (f"{metric}_max_over_median", maximum / median),
            (f"{metric}_median_over_minimum", median / minimum),
        ):
            require_measurement_close(
                ledger_row[field],
                expected,
                f"{LEDGER}.{key}.{field}",
            )
        if (
            ledger_row[f"{metric}_variance_policy"]
            != "descriptive_only_no_posthoc_exclusion"
            or ledger_row[f"{metric}_stability_status"] != "reported"
        ):
            fail(f"{LEDGER}.{key} dispersion-reporting contract differs")
        reconstructed[key] = {
            "displayed": displayed,
            "estimator": expected_estimator,
            "mean": mean,
            "sample_sd": sample_sd,
            "minimum": minimum,
            "median": median,
            "maximum": maximum,
        }

    audit_rows = read_csv(audit_path)
    require_columns(
        audit_rows,
        (
            "dataset",
            "function",
            "baseline",
            "metric",
            "new_paper_seconds",
            "new_estimator",
            "raw_mean_seconds",
            "raw_min_seconds",
            "sample_sd_seconds",
            "raw_median_seconds",
            "raw_max_seconds",
            "sample_count",
            "action",
        ),
        V15_ESTIMATOR_AUDIT,
    )
    audit_by_key: dict[
        tuple[str, str, str, str], Mapping[str, str]
    ] = {}
    for row in audit_rows:
        key = (
            str(row["dataset"]),
            str(row["function"]),
            str(row["baseline"]),
            str(row["metric"]),
        )
        if key in audit_by_key:
            fail(f"{V15_ESTIMATOR_AUDIT} has duplicate key {key}")
        audit_by_key[key] = row
    if set(audit_by_key) != expected_raw_keys:
        fail(f"{V15_ESTIMATOR_AUDIT} keys differ from raw evidence")
    derived_changed = 0
    derived_unchanged = 0
    for key, expected in reconstructed.items():
        row = audit_by_key[key]
        if row["new_estimator"] != expected["estimator"]:
            fail(f"{V15_ESTIMATOR_AUDIT}.{key} estimator differs")
        if (
            parse_int(
                row["sample_count"],
                f"{V15_ESTIMATOR_AUDIT}.{key}.sample_count",
            )
            != EXPECTED_SAMPLES_PER_METRIC
        ):
            fail(f"{V15_ESTIMATOR_AUDIT}.{key} sample count differs")
        old_paper = parse_float(
            row["old_paper_seconds"],
            f"{V15_ESTIMATOR_AUDIT}.{key}.old_paper_seconds",
            positive=True,
        )
        assert old_paper is not None
        unchanged_point = math.isclose(
            old_paper,
            float(expected["displayed"]),
            rel_tol=2.0e-8,
            abs_tol=2.0e-11,
        )
        expected_action = (
            f"unchanged_{expected['estimator']}"
            if unchanged_point
            else f"changed_to_{expected['estimator']}"
        )
        if row["action"] != expected_action:
            fail(f"{V15_ESTIMATOR_AUDIT}.{key} action differs")
        derived_unchanged += int(unchanged_point)
        derived_changed += int(not unchanged_point)
        for field, reconstructed_name in (
            ("new_paper_seconds", "displayed"),
            ("raw_mean_seconds", "mean"),
            ("raw_min_seconds", "minimum"),
            ("sample_sd_seconds", "sample_sd"),
            ("raw_median_seconds", "median"),
            ("raw_max_seconds", "maximum"),
        ):
            require_measurement_close(
                row[field],
                expected[reconstructed_name],
                f"{V15_ESTIMATOR_AUDIT}.{key}.{field}",
            )
    if derived_changed != changed or derived_unchanged != unchanged:
        fail(
            f"{V15_ESTIMATOR_MANIFEST} change counts differ from "
            f"{V15_ESTIMATOR_AUDIT}"
        )

    return {
        "schema_version": V15_ESTIMATOR_MANIFEST_SCHEMA,
        "display_estimator": V15_DISPLAY_ESTIMATOR,
        "policy": dict(V15_ESTIMATOR_POLICY),
        "display_estimator_by_baseline_class": dict(
            V15_ESTIMATOR_BY_BASELINE_CLASS
        ),
        "error_bar": V15_ERROR_BAR,
        "samples_per_displayed_metric": EXPECTED_SAMPLES_PER_METRIC,
        "manifest_sha256": sha256_file(manifest_path),
        "raw_samples_sha256": sha256_file(raw_path),
        "audit_sha256": sha256_file(audit_path),
        "reconstructed_displayed_metrics": len(reconstructed),
    }


def require_attested_hash(
    asset_dir: Path,
    verification: Mapping[str, object],
    field: str,
    relative_name: str,
) -> str:
    expected = require_sha256(
        verification.get(field), f"{VERIFICATION_NAME}.{field}"
    )
    path = require_file(asset_dir, relative_name)
    actual = sha256_file(path)
    if actual != expected:
        fail(
            f"{VERIFICATION_NAME}.{field} differs for {relative_name}: "
            f"{actual} != {expected}"
        )
    return actual


def validate_release_verification(
    asset_dir: Path,
    verification: Mapping[str, object],
    actual_counts: Mapping[str, object],
    headline_manifest: Mapping[str, object],
    externally_anchored_sha256: str,
) -> dict[str, object]:
    if verification.get("status") != "pass":
        fail(f"{VERIFICATION_NAME} status gate did not pass")
    if not asset_dir.name.startswith(EXPECTED_V15_DIRECTORY_PREFIX):
        fail("release asset directory is not a V15 directory")
    if (asset_dir / "V15_BUILD_INCOMPLETE").exists():
        fail("V15_BUILD_INCOMPLETE is present")
    candidate_sha256 = require_sha256(
        verification.get("candidate_sha256"),
        f"{VERIFICATION_NAME}.candidate_sha256",
    )
    runtime_sha256 = require_sha256(
        verification.get("runtime_python_snapshot_sha256"),
        f"{VERIFICATION_NAME}.runtime_python_snapshot_sha256",
    )
    if (
        candidate_sha256 != EXPECTED_NATIVE_SHA256
        or runtime_sha256 != EXPECTED_RUNTIME_SHA256
    ):
        fail("V15 final verification does not identify the frozen V15 runtime")
    for key, actual in actual_counts.items():
        if parse_int(verification.get(key), f"{VERIFICATION_NAME}.{key}") != actual:
            fail(
                f"{VERIFICATION_NAME}.{key}: "
                f"{verification.get(key)!r} != derived {actual!r}"
            )
    if (
        verification.get("non_eggpu_evidence_fields_unchanged") is not True
        or verification.get("non_eggpu_display_fields_recomputed") is not True
        or parse_int(
            verification.get("non_eggpu_display_field_changes"),
            f"{VERIFICATION_NAME}.non_eggpu_display_field_changes",
        )
        <= 0
    ):
        fail(f"{VERIFICATION_NAME} non-EGGPU recomputation gate differs")
    if verification.get("estimator_policy") != (
        "eggpu-minimum-baseline-mean"
    ):
        fail(f"{VERIFICATION_NAME}.estimator_policy differs")
    if verification.get("display_estimator") != V15_DISPLAY_ESTIMATOR:
        fail(f"{VERIFICATION_NAME}.display_estimator differs")
    if (
        verification.get("display_estimator_by_baseline_class")
        != V15_ESTIMATOR_BY_BASELINE_CLASS
    ):
        fail(
            f"{VERIFICATION_NAME}.display_estimator_by_baseline_class differs"
        )
    if verification.get("reported_error") != V15_ERROR_BAR:
        fail(f"{VERIFICATION_NAME}.reported_error differs")
    retained_samples = parse_int(
        verification.get("samples_per_displayed_metric"),
        f"{VERIFICATION_NAME}.samples_per_displayed_metric",
    )
    if retained_samples != EXPECTED_SAMPLES_PER_METRIC:
        fail(
            f"{VERIFICATION_NAME}.samples_per_displayed_metric must be "
            f"{EXPECTED_SAMPLES_PER_METRIC}"
        )
    estimator_contract = headline_manifest.get("estimator_contract")
    if not isinstance(estimator_contract, dict):
        fail("headline manifest lacks estimator_contract")
    if (
        estimator_contract.get("display_estimator")
        != verification.get("display_estimator")
        or estimator_contract.get("policy") != V15_ESTIMATOR_POLICY
        or parse_int(
            estimator_contract.get("retained_samples_per_displayed_metric"),
            "headline estimator retained_samples_per_displayed_metric",
        )
        != retained_samples
    ):
        fail("V15/headline estimator contracts differ")
    for field in ("successful_cells", "displayed_metrics", "portable_raw_rows"):
        if parse_int(
            estimator_contract.get(field), f"headline estimator {field}"
        ) != actual_counts[field]:
            fail(f"headline estimator {field} differs from derived count")

    direct_hashes = {
        field: require_attested_hash(asset_dir, verification, field, name)
        for field, name in (
            ("ledger_sha256", LEDGER),
            ("raw_samples_sha256", V15_RAW_SAMPLES),
            ("mixed_manifest_sha256", V15_ESTIMATOR_MANIFEST),
            ("mixed_estimator_audit_sha256", V15_ESTIMATOR_AUDIT),
            (
                "stability_audit_sha256",
                f"{V15_AUDIT_DIR}/eggpu_timing_stability.json",
            ),
            (
                "overlay_audit_sha256",
                f"{V15_AUDIT_DIR}/EGGPU_V15_UNIFIED_OVERLAY_AUDIT.json",
            ),
        )
    }
    asset_hashes = validate_hash_map(
        asset_dir, verification.get("asset_sha256"), "V15 asset hashes"
    )
    if set(asset_hashes) != set(V15_ATTESTED_ASSETS):
        fail(
            "V15 asset hash manifest differs from the fixed 19-asset contract"
        )

    correction = verification.get("pagerank_protocol_correction")
    if (
        verification.get("pagerank_protocol_corrected") is not True
        or not isinstance(correction, dict)
    ):
        fail("PageRank protocol correction gate did not pass")
    correction_hashes = {
        field: require_attested_hash(asset_dir, correction, field, name)
        for field, name in (
            (
                "correction_audit_sha256",
                (
                    f"{V15_AUDIT_DIR}/"
                    "V15_PAGERANK_PROTOCOL_CORRECTION_AUDIT.json"
                ),
            ),
            (
                "action_ledger_sha256",
                f"{V15_AUDIT_DIR}/pagerank_protocol_action_ledger.csv",
            ),
            (
                "non_pagerank_cell_hashes_sha256",
                f"{V15_AUDIT_DIR}/non_pagerank_cell_hashes.csv",
            ),
            (
                "main_timing_dirs_sha256",
                f"{V15_AUDIT_DIR}/main_timing_dirs.txt",
            ),
            (
                "anchor_timing_dirs_sha256",
                f"{V15_AUDIT_DIR}/anchor_timing_dirs.txt",
            ),
        )
    }
    correction_counts = {
        key: parse_int(correction.get(key), f"pagerank correction.{key}")
        for key in (
            "replace_protocol_invalid",
            "retain_equivalent_direct_protocol",
            "non_pagerank_cells_unchanged",
        )
    }
    if any(value <= 0 for value in correction_counts.values()):
        fail("PageRank correction counts must be positive")
    matrix_contract = headline_manifest.get("matrix_contract")
    if not isinstance(matrix_contract, dict):
        fail("headline manifest lacks matrix_contract")
    datasets = parse_int(
        matrix_contract.get("datasets"), "headline matrix datasets"
    )
    if (
        correction_counts["replace_protocol_invalid"]
        + correction_counts["retain_equivalent_direct_protocol"]
        != datasets
    ):
        fail("PageRank replacement + anchor counts differ from dataset count")
    if (
        correction_counts["non_pagerank_cells_unchanged"]
        != int(actual_counts["eggpu_cells"]) - datasets
    ):
        fail("non-PageRank identity count differs from the EGGPU matrix")
    if correction.get("protocol") != PAGERANK_PROTOCOL:
        fail("PageRank correction summary protocol differs")

    ledger_rows = read_csv(require_file(asset_dir, LEDGER))
    require_columns(
        ledger_rows,
        (
            "dataset",
            "function",
            "baseline",
            "execution_status",
            "candidate_sha256",
            "runtime_python_snapshot_sha256",
            "memory_candidate_sha256",
            "memory_runtime_python_snapshot_sha256",
        ),
        LEDGER,
    )
    eggpu_rows = [row for row in ledger_rows if row["baseline"] == "EGGPU"]
    if (
        len(eggpu_rows) != actual_counts["eggpu_cells"]
        or any(row["execution_status"] != "ok" for row in eggpu_rows)
    ):
        fail("V15 EGGPU ledger cell status/count gate differs")
    for field, expected in (
        ("candidate_sha256", candidate_sha256),
        ("runtime_python_snapshot_sha256", runtime_sha256),
        ("memory_candidate_sha256", candidate_sha256),
        ("memory_runtime_python_snapshot_sha256", runtime_sha256),
    ):
        if {row[field] for row in eggpu_rows} != {expected}:
            fail(f"V15 EGGPU ledger {field} identity differs")

    stability = read_json(
        require_file(
            asset_dir, f"{V15_AUDIT_DIR}/eggpu_timing_stability.json"
        )
    )
    if not (
        stability.get("status") == "pass"
        and parse_int(
            stability.get("audited_cells"), "stability audited_cells"
        )
        == actual_counts["eggpu_cells"]
        and parse_int(
            stability.get("unique_workload_keys"),
            "stability unique_workload_keys",
        )
        == actual_counts["eggpu_cells"]
        and parse_int(
            stability.get("expected_samples_per_cell"),
            "stability expected_samples_per_cell",
        )
        == retained_samples
        and stability.get("candidate_sha256") == candidate_sha256
        and stability.get("runtime_python_snapshot_sha256") == runtime_sha256
        and stability.get("runtime_package_is_symlink") is False
    ):
        fail("V15 stability identity/count gate differs")

    overlay = read_json(
        require_file(
            asset_dir,
            f"{V15_AUDIT_DIR}/EGGPU_V15_UNIFIED_OVERLAY_AUDIT.json",
        )
    )
    if not (
        overlay.get("status") == "pass"
        and overlay.get("candidate_mode") == "unified-timing-memory"
        and parse_int(
            overlay.get("timing_replacement_count"),
            "overlay timing_replacement_count",
        )
        == actual_counts["eggpu_cells"]
        and parse_int(
            overlay.get("memory_replacement_count"),
            "overlay memory_replacement_count",
        )
        == actual_counts["eggpu_cells"]
        and overlay.get("candidate_sha256") == [candidate_sha256]
        and overlay.get("runtime_python_snapshot_sha256") == [runtime_sha256]
    ):
        fail("V15 unified-overlay identity/count gate differs")
    overlay_ledger_path = require_file(
        asset_dir, f"{V15_AUDIT_DIR}/{V15_OVERLAY_LEDGER}"
    )
    overlay_ledger_sha256 = sha256_file(overlay_ledger_path)
    mixed_manifest = read_json(
        require_file(asset_dir, V15_ESTIMATOR_MANIFEST)
    )
    if not (
        overlay.get("base_ledger_sha256")
        == EXPECTED_V14_BASE_LEDGER_SHA256
        and isinstance(overlay.get("base_ledger"), str)
        and Path(str(overlay["base_ledger"])).name == LEDGER
        and Path(str(overlay["base_ledger"])).parent.name
        == EXPECTED_V14_BASE_DIRECTORY
        and overlay.get("output_ledger_sha256")
        == overlay_ledger_sha256
        and isinstance(overlay.get("output_ledger"), str)
        and Path(str(overlay["output_ledger"])).name
        == V15_OVERLAY_LEDGER
        and mixed_manifest.get("base_ledger_sha256")
        == overlay_ledger_sha256
    ):
        fail("V14 -> V15 overlay -> mixed-estimator provenance chain differs")

    overlay_rows = read_csv(overlay_ledger_path)
    if list(overlay_rows[0]) != list(ledger_rows[0]):
        fail("non-EGGPU ledger columns changed after the V15 overlay")
    overlay_non_eggpu = {
        (row["dataset"], row["function"], row["baseline"]): row
        for row in overlay_rows
        if row["baseline"] != "EGGPU"
    }
    final_non_eggpu = {
        (row["dataset"], row["function"], row["baseline"]): row
        for row in ledger_rows
        if row["baseline"] != "EGGPU"
    }
    if (
        len(overlay_non_eggpu) != EXPECTED_NON_EGGPU_ROWS
        or len(final_non_eggpu) != EXPECTED_NON_EGGPU_ROWS
        or set(overlay_non_eggpu) != set(final_non_eggpu)
    ):
        fail("non-EGGPU ledger key set/count changed after the V15 overlay")
    display_derived_fields = {
        f"{metric}_{suffix}"
        for metric in ("build", "kernel", "e2e")
        for suffix in (
            "paper_seconds",
            "raw_mean_seconds",
            "std_seconds",
            "estimator",
            "raw_min_seconds",
            "raw_median_seconds",
            "raw_max_seconds",
            "coefficient_of_variation",
            "max_over_median",
            "median_over_minimum",
            "variance_policy",
            "stability_status",
        )
    }
    display_derived_fields.add("sample_count")
    display_field_changes = 0
    provenance_text_changes = 0
    for key in sorted(final_non_eggpu):
        before = overlay_non_eggpu[key]
        after = final_non_eggpu[key]
        for field in before:
            if (
                before[field] == after[field]
                or numeric_equivalent(before[field], after[field])
            ):
                continue
            if (
                before["execution_status"] == "ok"
                and after["execution_status"] == "ok"
                and field in display_derived_fields
            ):
                display_field_changes += 1
                continue
            if (
                before["baseline"] == "nx-cugraph"
                and field == "result_source"
                and before[field]
                == "strict nx-cugraph 99-cell rerun; minimum of five"
                and after[field]
                == (
                    "strict nx-cugraph 99-cell rerun; arithmetic mean of five "
                    "fresh processes"
                )
            ):
                provenance_text_changes += 1
                continue
            if (
                before["baseline"] == "Gunrock"
                and field == "reason"
                and before[field]
                == (
                    "strict native Gunrock three-phase timing; minimum of "
                    "five fresh processes; full-result validation passed; "
                    "external CLI wall substitution forbidden"
                )
                and after[field]
                == (
                    "strict native Gunrock three-phase timing; arithmetic "
                    "mean of five fresh processes; full-result validation "
                    "passed; external CLI wall substitution forbidden"
                )
            ):
                provenance_text_changes += 1
                continue
            fail(
                "non-EGGPU source/provenance field changed after overlay: "
                f"{key}.{field}"
            )
    if (
        display_field_changes <= 0
        or display_field_changes
        != parse_int(
            verification.get("non_eggpu_display_field_changes"),
            f"{VERIFICATION_NAME}.non_eggpu_display_field_changes",
        )
    ):
        fail("non-EGGPU display-field recomputation count differs")
    provenance_normalized = (
        verification.get("external_estimator_provenance_text_normalized") is True
    )
    declared_provenance_changes = parse_int(
        verification.get("external_estimator_provenance_text_changes", 0),
        f"{VERIFICATION_NAME}.external_estimator_provenance_text_changes",
    )
    if provenance_normalized:
        if provenance_text_changes != declared_provenance_changes or (
            provenance_text_changes != 189
        ):
            fail("external estimator-provenance normalization count differs")
    elif provenance_text_changes != 0 or declared_provenance_changes != 0:
        fail("undeclared external estimator-provenance normalization")

    correction_audit = read_json(
        require_file(
            asset_dir,
            (
                f"{V15_AUDIT_DIR}/"
                "V15_PAGERANK_PROTOCOL_CORRECTION_AUDIT.json"
            ),
        )
    )
    if not (
        correction_audit.get("status") == "pass"
        and correction_audit.get("schema_version") == 1
        and correction_audit.get("protocol") == PAGERANK_PROTOCOL
        and correction_audit.get("candidate_sha256") == candidate_sha256
        and correction_audit.get("runtime_python_snapshot_sha256")
        == runtime_sha256
        and correction_audit.get("adoption_policy")
        == PAGERANK_ADOPTION_POLICY
        and correction_audit.get(
            "relative_performance_considered_for_adoption"
        )
        is False
        and correction_audit.get("relative_speed_comparison_performed") is False
    ):
        fail("PageRank correction audit identity/protocol gate differs")
    audit_actions = correction_audit.get("actions")
    if not isinstance(audit_actions, dict) or (
        parse_int(
            audit_actions.get("replace_protocol_invalid"),
            "correction audit replacement count",
        )
        != correction_counts["replace_protocol_invalid"]
        or parse_int(
            audit_actions.get("retain_equivalent_direct_protocol"),
            "correction audit anchor count",
        )
        != correction_counts["retain_equivalent_direct_protocol"]
        or parse_int(
            audit_actions.get("total_pagerank_cells"),
            "correction audit PageRank count",
        )
        != datasets
    ):
        fail("PageRank correction audit action counts differ")
    audit_matrix = correction_audit.get("matrix")
    if not isinstance(audit_matrix, dict) or (
        parse_int(audit_matrix.get("datasets"), "correction matrix datasets")
        != datasets
        or parse_int(
            audit_matrix.get("eggpu_cells"), "correction matrix EGGPU cells"
        )
        != actual_counts["eggpu_cells"]
        or parse_int(
            audit_matrix.get("pagerank_cells"),
            "correction matrix PageRank cells",
        )
        != datasets
        or parse_int(
            audit_matrix.get("non_pagerank_cells"),
            "correction matrix non-PageRank cells",
        )
        != correction_counts["non_pagerank_cells_unchanged"]
    ):
        fail("PageRank correction audit matrix differs")
    measurement = correction_audit.get("measurement_contract")
    if not isinstance(measurement, dict) or not (
        parse_int(
            measurement.get("samples_per_metric"),
            "correction samples_per_metric",
        )
        == retained_samples
        and measurement.get("metrics") == ["build", "e2e", "kernel"]
        and parse_int(
            measurement.get("timing_rows_per_cell"),
            "correction timing_rows_per_cell",
        )
        == retained_samples * len(measurement["metrics"])
        and measurement.get("validation_outside_timer") is True
        and measurement.get("missing_kernel_time") == "fail_closed"
    ):
        fail("PageRank correction measurement contract differs")
    audit_non_pr = correction_audit.get("non_pagerank_identity")
    if not isinstance(audit_non_pr, dict) or not (
        audit_non_pr.get("status") == "pass"
        and parse_int(
            audit_non_pr.get("normalized_cell_count"),
            "correction non-PageRank normalized_cell_count",
        )
        == correction_counts["non_pagerank_cells_unchanged"]
        and audit_non_pr.get("before_sha256")
        == audit_non_pr.get("after_sha256")
        and audit_non_pr.get("all_cell_hashes_equal") is True
    ):
        fail("PageRank correction non-PageRank identity gate differs")
    require_sha256(
        audit_non_pr.get("before_sha256"),
        "correction non-PageRank aggregate SHA-256",
    )

    action_rows = read_csv(
        require_file(
            asset_dir,
            f"{V15_AUDIT_DIR}/pagerank_protocol_action_ledger.csv",
        )
    )
    require_columns(
        action_rows,
        (
            "dataset",
            "function",
            "action",
            "protocol",
            "samples_per_metric",
            "timing_rows_per_cell",
            "validation_outside_timer",
            "relative_performance_considered_for_adoption",
        ),
        "pagerank_protocol_action_ledger.csv",
    )
    ledger_datasets = {row["dataset"] for row in eggpu_rows}
    if (
        len(action_rows) != datasets
        or {row["dataset"] for row in action_rows} != ledger_datasets
        or len({row["dataset"] for row in action_rows}) != len(action_rows)
    ):
        fail("PageRank action ledger dataset set/count differs")
    replacement_datasets = {
        row["dataset"]
        for row in action_rows
        if row["action"] == "replace_protocol_invalid"
    }
    anchor_datasets = {
        row["dataset"]
        for row in action_rows
        if row["action"] == "retain_equivalent_direct_protocol"
    }
    if (
        anchor_datasets != set(PAGERANK_DIRECT_ANCHORS)
        or replacement_datasets
        != ledger_datasets - set(PAGERANK_DIRECT_ANCHORS)
        or len(replacement_datasets)
        != correction_counts["replace_protocol_invalid"]
    ):
        fail("PageRank action ledger replacement/anchor sets differ")
    for row in action_rows:
        if not (
            row["function"] == "PageRank"
            and parse_int(
                row["samples_per_metric"], "action samples_per_metric"
            )
            == retained_samples
            and parse_int(
                row["timing_rows_per_cell"], "action timing_rows_per_cell"
            )
            == retained_samples * 3
            and parse_bool(
                row["validation_outside_timer"],
                "action validation_outside_timer",
            )
            and not parse_bool(
                row["relative_performance_considered_for_adoption"],
                "action relative_performance_considered_for_adoption",
            )
        ):
            fail("PageRank action ledger row contract differs")
        if (
            row["action"] == "replace_protocol_invalid"
            and row["protocol"] != PAGERANK_PROTOCOL
        ):
            fail("PageRank replacement row protocol differs")

    non_pr_rows = read_csv(
        require_file(
            asset_dir,
            f"{V15_AUDIT_DIR}/non_pagerank_cell_hashes.csv",
        )
    )
    require_columns(
        non_pr_rows,
        ("dataset", "function", "before_sha256", "after_sha256"),
        "non_pagerank_cell_hashes.csv",
    )
    expected_non_pr_keys = {
        (row["dataset"], row["function"])
        for row in eggpu_rows
        if row["function"] != "PageRank"
    }
    observed_non_pr_keys = {
        (row["dataset"], row["function"]) for row in non_pr_rows
    }
    if (
        len(non_pr_rows) != correction_counts["non_pagerank_cells_unchanged"]
        or len(observed_non_pr_keys) != len(non_pr_rows)
        or observed_non_pr_keys != expected_non_pr_keys
    ):
        fail("non-PageRank cell-hash ledger key set/count differs")
    for row in non_pr_rows:
        before = require_sha256(
            row["before_sha256"], "non-PageRank before SHA-256"
        )
        after = require_sha256(
            row["after_sha256"], "non-PageRank after SHA-256"
        )
        if before != after:
            fail("a non-PageRank cell hash changed")

    audit_output_hashes = correction_audit.get("output_file_sha256")
    if not isinstance(audit_output_hashes, dict):
        fail("PageRank correction audit output hash manifest is absent")
    for name in (
        "pagerank_protocol_action_ledger.csv",
        "non_pagerank_cell_hashes.csv",
        "main_timing_dirs.txt",
        "anchor_timing_dirs.txt",
    ):
        expected = require_sha256(
            audit_output_hashes.get(name), f"correction output hash {name}"
        )
        actual = sha256_file(
            require_file(asset_dir, f"{V15_AUDIT_DIR}/{name}")
        )
        if actual != expected:
            fail(f"PageRank correction audit output hash differs for {name}")

    headline = verification.get("headline_results")
    if not isinstance(headline, dict):
        fail(f"{VERIFICATION_NAME}.headline_results is absent")
    if (
        headline.get("status") != "pass"
        or headline.get("schema_version") != V15_HEADLINE_SCHEMA
        or headline.get("all_semantic_gates_pass") is not True
    ):
        fail("V15 headline-results gate did not pass")
    manifest_sha = sha256_file(asset_dir / HEADLINE_MANIFEST)
    if headline.get("manifest_sha256") != manifest_sha:
        fail("V15 headline-results manifest SHA-256 differs")
    if headline.get("input_sha256") != headline_manifest.get("input_sha256"):
        fail("V15/headline-manifest input hashes differ")
    if headline.get("output_sha256") != headline_manifest.get("output_sha256"):
        fail("V15/headline-manifest output hashes differ")

    identity = read_json(
        require_file(
            asset_dir, f"{V15_AUDIT_DIR}/V15_RUNTIME_IDENTITY.json"
        )
    )
    if (
        identity.get("status") != "pass"
        or identity.get("runtime_package_is_symlink") is not False
        or identity.get("candidate_sha256") != candidate_sha256
        or identity.get("runtime_python_snapshot_sha256") != runtime_sha256
    ):
        fail("V15 runtime identity does not match final verification")

    return {
        "mode": "v15_release",
        "release_gate_pass": True,
        "externally_anchored": True,
        "verification_sha256": sha256_file(asset_dir / VERIFICATION_NAME),
        "expected_verification_sha256": externally_anchored_sha256,
        "candidate_sha256": candidate_sha256,
        "runtime_python_snapshot_sha256": runtime_sha256,
        "counts": dict(actual_counts),
        "estimator_contract": {
            "display_estimator": V15_DISPLAY_ESTIMATOR,
            "policy": dict(V15_ESTIMATOR_POLICY),
            "display_estimator_by_baseline_class": dict(
                V15_ESTIMATOR_BY_BASELINE_CLASS
            ),
            "error_bar": V15_ERROR_BAR,
        },
        "samples_per_displayed_metric": retained_samples,
        "non_eggpu_evidence_fields_unchanged": True,
        "non_eggpu_display_fields_recomputed": True,
        "non_eggpu_display_field_changes": parse_int(
            verification.get("non_eggpu_display_field_changes"),
            f"{VERIFICATION_NAME}.non_eggpu_display_field_changes",
        ),
        "direct_hashes": direct_hashes,
        "asset_sha256": asset_hashes,
        "pagerank_protocol_correction": {
            "protocol": correction.get("protocol"),
            "counts": correction_counts,
            "hashes": correction_hashes,
        },
        "headline_results": {
            "manifest_sha256": manifest_sha,
            "input_sha256": headline_manifest.get("input_sha256"),
            "output_sha256": headline_manifest.get("output_sha256"),
            "all_semantic_gates_pass": True,
        },
    }


def validate_fixture_v14(
    asset_dir: Path,
    actual_counts: Mapping[str, object],
    headline_manifest: Mapping[str, object],
) -> dict[str, object]:
    if "ASSETS_14" not in asset_dir.name:
        fail(
            "--allow-fixture-v14 is test-only and requires an ASSETS_14 "
            "fixture directory"
        )
    uniform_path = require_file(asset_dir, V14_ESTIMATOR_MANIFEST)
    uniform = read_json(uniform_path)
    if uniform.get("status") != "pass":
        fail("V14 fixture uniform-minimum manifest status is not pass")
    if uniform.get("output_ledger_sha256") != sha256_file(asset_dir / LEDGER):
        fail("V14 fixture uniform-minimum manifest does not hash the ledger")
    return {
        "mode": "v14_fixture_only",
        "release_gate_pass": False,
        "fixture_only": True,
        "counts": dict(actual_counts),
        "estimator_contract": {
            "display_estimator": "minimum_of_five",
            "fixture_only": True,
        },
        "uniform_manifest_sha256": sha256_file(uniform_path),
        "headline_manifest_sha256": sha256_file(asset_dir / HEADLINE_MANIFEST),
        "headline_input_sha256": headline_manifest.get("input_sha256"),
        "headline_output_sha256": headline_manifest.get("output_sha256"),
    }


def derive(
    asset_dir: Path,
    allow_fixture_v14: bool,
    expected_verification_sha256: str | None,
) -> dict[str, object]:
    asset_dir = asset_dir.expanduser().resolve(strict=True)
    if not asset_dir.is_dir():
        fail(f"--asset-dir is not a directory: {asset_dir}")
    verification_path = asset_dir / VERIFICATION_NAME
    anchored_verification_sha256: str | None = None
    if verification_path.is_file():
        if expected_verification_sha256 is None:
            fail(
                "--expected-verification-sha256 is required for a V15 "
                "release; an in-directory verification file cannot "
                "self-anchor"
            )
        anchored_verification_sha256 = require_sha256(
            expected_verification_sha256,
            "--expected-verification-sha256",
        )
        actual_verification_sha256 = sha256_file(verification_path)
        if actual_verification_sha256 != anchored_verification_sha256:
            fail(
                "external V15 verification SHA-256 differs: "
                f"{actual_verification_sha256} != "
                f"{anchored_verification_sha256}"
            )
    elif expected_verification_sha256 is not None:
        fail(
            "--expected-verification-sha256 was provided but "
            f"{VERIFICATION_NAME} is absent"
        )

    release_mode = verification_path.is_file()
    headlines, headline_manifest = validate_headlines(
        asset_dir, release_mode=release_mode
    )
    ledger_rows = read_csv(require_file(asset_dir, LEDGER))
    ledger = ledger_index(ledger_rows)
    datasets = {str(row["dataset"]) for row in ledger_rows}
    functions = {str(row["function"]) for row in ledger_rows}
    systems = {str(row["baseline"]) for row in ledger_rows}
    archived_in_current_ledger = systems.intersection(
        ARCHIVED_SINGLE_SOURCE_SYSTEMS
    )
    if archived_in_current_ledger:
        fail(
            "archived single-source systems must not appear in the current "
            "five-run ledger: "
            f"{sorted(archived_in_current_ledger)}"
        )
    expected_keys = {
        (dataset, function, system)
        for dataset in datasets
        for function in functions
        for system in EXPECTED_SYSTEM_NAMES
    }
    if (
        len(datasets) != EXPECTED_DATASETS
        or len(functions) != EXPECTED_FUNCTIONS
        or systems != set(EXPECTED_SYSTEM_NAMES)
        or len(ledger_rows) != EXPECTED_LEDGER_ROWS
        or set(ledger) != expected_keys
    ):
        fail(f"{LEDGER} fixed 13x16x8 matrix contract differs")
    eggpu_rows = [
        row for row in ledger_rows if str(row["baseline"]) == "EGGPU"
    ]
    if (
        len(eggpu_rows) != EXPECTED_EGGPU_CELLS
        or any(
            row["execution_status"] != "ok"
            or row["validation_status"] not in VALIDATION_OK
            for row in eggpu_rows
        )
    ):
        fail("EGGPU must be correctness-valid on all 208 workloads")

    family_summary, family_overall = derive_family_summary(
        asset_dir, headlines
    )
    e2e_losses = derive_e2e_losses(
        asset_dir,
        ledger,
        int(family_overall["e2e"]["common_losses"]),
    )
    baseline_summaries = derive_baseline_summaries(asset_dir, functions)
    matrix = headline_manifest.get("matrix_contract")
    if not isinstance(matrix, dict) or not isinstance(matrix.get("system_names"), list):
        fail("headline manifest lacks matrix_contract.system_names")
    system_order = [str(value) for value in matrix["system_names"]]
    gap_twitter_bfs = derive_gap_twitter_bfs(ledger_rows, system_order)
    figure3 = derive_figure3(asset_dir, release_mode=release_mode)
    raw_samples_name = (
        V15_RAW_SAMPLES if release_mode else V14_RAW_SAMPLES
    )
    counts = derive_counts(ledger_rows, asset_dir, raw_samples_name)
    expected_counts = {
        "ledger_rows": EXPECTED_LEDGER_ROWS,
        "successful_cells": EXPECTED_SUCCESSFUL_CELLS,
        "displayed_metrics": EXPECTED_DISPLAYED_METRICS,
        "portable_raw_rows": EXPECTED_RAW_ROWS,
        "eggpu_cells": EXPECTED_EGGPU_CELLS,
        "non_eggpu_rows": EXPECTED_NON_EGGPU_ROWS,
    }
    if counts != expected_counts:
        fail(f"fixed V15 count contract differs: {counts} != {expected_counts}")

    estimator_evidence: dict[str, object]
    if release_mode:
        estimator_evidence = validate_mixed_estimator_evidence(
            asset_dir, ledger_rows, counts
        )
        assert anchored_verification_sha256 is not None
        verification = validate_release_verification(
            asset_dir,
            read_json(verification_path),
            counts,
            headline_manifest,
            anchored_verification_sha256,
        )
        status = "pass"
        release_ready = True
    else:
        if not allow_fixture_v14:
            fail(
                f"{VERIFICATION_NAME} is required; "
                "--allow-fixture-v14 is only for the V14 test fixture"
            )
        verification = validate_fixture_v14(
            asset_dir, counts, headline_manifest
        )
        estimator_evidence = {
            "schema_version": "uniform_minimum_of_five_v1",
            "display_estimator": "minimum_of_five",
            "fixture_only": True,
            "manifest_sha256": sha256_file(
                asset_dir / V14_ESTIMATOR_MANIFEST
            ),
            "raw_samples_sha256": sha256_file(
                asset_dir / V14_RAW_SAMPLES
            ),
        }
        status = "pass_fixture_only"
        release_ready = False

    estimator_source_names = (
        (
            V15_ESTIMATOR_MANIFEST,
            V15_RAW_SAMPLES,
            V15_ESTIMATOR_AUDIT,
        )
        if release_mode
        else (V14_ESTIMATOR_MANIFEST, V14_RAW_SAMPLES)
    )
    source_hashes = {
        name: sha256_file(require_file(asset_dir, name))
        for name in (
            HEADLINE_CSV,
            HEADLINE_MANIFEST,
            FAMILY_SUMMARY,
            PAIRWISE_SOTA,
            PAIRWISE_BASELINE,
            LEDGER,
            FIGURE3_METADATA,
            FIGURE3_DATA,
            *estimator_source_names,
        )
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "release_ready": release_ready,
        "asset_directory": str(asset_dir),
        "source_sha256": source_hashes,
        "verification": verification,
        "estimator_evidence": estimator_evidence,
        "estimator_scope": {
            "current_five_run_systems": list(EXPECTED_SYSTEM_NAMES),
            "excluded_archived_single_source_systems": list(
                ARCHIVED_SINGLE_SOURCE_SYSTEMS
            ),
            "archived_policy": (
                "Archived single-source timings are reference-only and are "
                "excluded from current five-run estimators, error bars, "
                "W/T/L counts, and speedups."
            ),
        },
        "headline_groups": headlines,
        "family_summary": family_summary,
        "e2e_losses": e2e_losses,
        "baseline_summaries": baseline_summaries,
        "gap_twitter_bfs": gap_twitter_bfs,
        "figure3": figure3,
    }


def write_atomic(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive hash-gated V15 facts for paper synchronization."
    )
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; the same JSON is always emitted to stdout.",
    )
    parser.add_argument(
        "--expected-verification-sha256",
        help=(
            "Trusted SHA-256 of V15_FINAL_VERIFICATION.json. Required for "
            "release mode so the in-directory attestation is externally "
            "anchored."
        ),
    )
    parser.add_argument(
        "--allow-fixture-v14",
        action="store_true",
        help=(
            "TEST ONLY: allow an ASSETS_14 fixture without "
            "V15_FINAL_VERIFICATION.json; never marks output release-ready."
        ),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        facts = derive(
            arguments.asset_dir,
            arguments.allow_fixture_v14,
            arguments.expected_verification_sha256,
        )
        text = json.dumps(facts, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if arguments.output is not None:
            write_atomic(arguments.output, text)
        print(text, end="")
    except (FactError, FileNotFoundError, NotADirectoryError) as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
