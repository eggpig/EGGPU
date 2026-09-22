#!/usr/bin/env python3
"""Audit submission-facing EGGPU claims against a frozen V10 asset directory.

The audit is intentionally CPU-only and read-only.  It checks four contracts:

1. repeated numerical claims in the paper agree with their source assets;
2. every required main or supplement asset exists and has the expected schema;
3. EGGPU timing cells use the minimum of five independent raw samples while
   retaining the arithmetic mean, sample standard deviation, and CV for
   dispersion auditing.  Catastrophic batches are rejected when
   ``max/median > 5`` or ``median/min > 3``; and
4. submission-facing TeX and asset manifests contain no superseded wording.

The canonical raw-sample file is ``final_13_eggpu_timing_samples.csv`` with
columns ``dataset,function,baseline,metric,sample_index,seconds,status``.
``eggpu_five_run_timing_samples.csv`` and ``eggpu_timing_samples.csv`` are
accepted aliases.  ``eggpu_timing_stability_policy.json`` records the
``catastrophic_outlier_guard`` policy, its two ratio limits, the five-sample
contract, minimum-of-five center, and retained dispersion statistics.  Missing
supplement, raw-sample, or policy assets are reported as ``missing`` and make
the command fail; they never become skipped passes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable


PASS = "pass"
FAIL = "fail"
MISSING = "missing"
INVALID = "invalid"
NOT_PRESENT = "not_present"

DEFAULT_MAX_OVER_MEDIAN_LIMIT = 5.0
DEFAULT_MEDIAN_OVER_MIN_LIMIT = 3.0
STRUCTURAL_FUNCTIONS = {
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
}
NUMBER = r"([0-9]+(?:\.[0-9]+)?)"
INTEGER_OR_ONE = r"([0-9]+|one)"
TIMES = r"\s*(?:\$\s*\\times\s*\$|\\times)"

STALE_PATTERNS = (
    ("best_observed", re.compile(r"\bbest[\s-]+observed\b", re.IGNORECASE)),
    ("best_of_five", re.compile(r"\bbest[\s-]+of[\s-]+five\b", re.IGNORECASE)),
    ("frozen_v8", re.compile(r"\bfrozen\s+v8\b", re.IGNORECASE)),
    ("v9_timing", re.compile(r"\bv9\s+timing\b", re.IGNORECASE)),
)
ARITHMETIC_MEAN_PATTERN = re.compile(
    r"\barithmetic(?:[\s_-]+)means?\b", re.IGNORECASE
)
MINIMUM_OF_FIVE_PATTERN = re.compile(
    r"\bminimum(?:[\s_-]+)(?:of|among|across)(?:[\s_-]+the)?"
    r"(?:[\s_-]+)(?:five|5)\b",
    re.IGNORECASE,
)


class AuditInputError(ValueError):
    """Raised when an existing asset cannot support its declared claim."""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _as_float(value: Any, label: str) -> float:
    if not _finite(value):
        raise AuditInputError(f"{label} is not finite: {value!r}")
    return float(value)


def _as_int(value: Any, label: str) -> int:
    number = _as_float(value, label)
    if not number.is_integer():
        raise AuditInputError(f"{label} is not an integer: {value!r}")
    return int(number)


def _normalise_estimator(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _validation_passes(value: Any) -> bool:
    normalised = _normalise_estimator(value)
    return normalised in {"pass", "ok", "validated"} or normalised.endswith("_pass")


def _is_minimum_of_five(value: Any) -> bool:
    normalised = _normalise_estimator(value)
    return (
        ("minimum" in normalised or re.search(r"(?:^|_)min(?:_|$)", normalised))
        and ("five" in normalised or re.search(r"(?:^|_)5(?:_|$)", normalised))
    )


def _estimator_family(value: Any) -> str:
    if _is_minimum_of_five(value):
        return "minimum_of_five"
    normalised = _normalise_estimator(value)
    if "arithmetic" in normalised and "mean" in normalised:
        return "arithmetic_mean"
    return normalised or "undeclared"


def _display(value: float, decimals: int) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}"


def _round_outward(value: float, decimals: int, upper: bool) -> float:
    quantum = Decimal(1).scaleb(-decimals)
    mode = ROUND_CEILING if upper else ROUND_FLOOR
    return float(Decimal(str(value)).quantize(quantum, rounding=mode))


def _geomean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values or not all(_positive(value) for value in values):
        raise AuditInputError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_columns(
    rows: list[dict[str, str]], columns: Iterable[str], label: str
) -> None:
    if not rows:
        raise AuditInputError(f"{label} contains no data rows")
    missing = set(columns) - set(rows[0])
    if missing:
        raise AuditInputError(f"{label} is missing columns {sorted(missing)}")


class AssetRegistry:
    """Resolve required assets while preserving missing/invalid state."""

    def __init__(self, root: Path):
        self.root = root
        self.results: dict[str, dict[str, Any]] = {}

    def require(
        self,
        name: str,
        candidates: Iterable[str],
        *,
        glob_pattern: str | None = None,
    ) -> Path | None:
        candidates = tuple(candidates)
        matches = [self.root / item for item in candidates if (self.root / item).is_file()]
        if not matches and glob_pattern:
            matches = sorted(path for path in self.root.glob(glob_pattern) if path.is_file())
        if not matches:
            self.results[name] = {
                "status": MISSING,
                "expected": list(candidates)
                + ([f"glob:{glob_pattern}"] if glob_pattern else []),
            }
            return None
        if len(matches) > 1:
            self.results[name] = {
                "status": INVALID,
                "detail": "ambiguous asset candidates",
                "paths": [str(path) for path in matches],
            }
            return None
        path = matches[0]
        self.results[name] = {"status": PASS, "path": str(path)}
        return path

    def invalidate(self, name: str, detail: str) -> None:
        current = self.results.setdefault(name, {"status": INVALID})
        current["status"] = INVALID
        current.setdefault("issues", []).append(detail)

    def annotate(self, name: str, **values: Any) -> None:
        self.results.setdefault(name, {"status": PASS}).update(values)


def _load_csv_asset(
    registry: AssetRegistry,
    name: str,
    candidates: Iterable[str],
    columns: Iterable[str],
    *,
    glob_pattern: str | None = None,
) -> list[dict[str, str]] | None:
    path = registry.require(name, candidates, glob_pattern=glob_pattern)
    if path is None:
        return None
    try:
        rows = _read_csv(path)
        _require_columns(rows, columns, name)
        registry.annotate(name, rows=len(rows))
        return rows
    except (OSError, UnicodeError, csv.Error, AuditInputError) as error:
        registry.invalidate(name, str(error))
        return None


def _load_json_asset(
    registry: AssetRegistry,
    name: str,
    candidates: Iterable[str],
    *,
    glob_pattern: str | None = None,
) -> Any | None:
    path = registry.require(name, candidates, glob_pattern=glob_pattern)
    if path is None:
        return None
    try:
        value = _read_json(path)
        registry.annotate(name, json_type=type(value).__name__)
        return value
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        registry.invalidate(name, str(error))
        return None


def _headline_evidence(
    summary: dict[str, Any], registry: AssetRegistry
) -> dict[str, float]:
    required = {
        "total": "e2e_eggpu_successful_workloads",
        "wins": "e2e_common_pair_strict_wins",
        "ties": "e2e_common_pair_ties",
        "losses": "e2e_common_pair_losses",
        "common": "e2e_competitive_pairs",
        "sole": "e2e_sole_validated_cells",
        "external_speedup": "e2e_speedup_over_best_competitor",
        "nx_speedup": "e2e_speedup_over_strict_nx_cugraph",
        "nx_pairs": "e2e_strict_nx_cugraph_common_pairs",
        "gunrock_speedup": "kernel_speedup_over_best_native_gpu",
        "gunrock_pairs": "kernel_best_native_gpu_common_pairs",
    }
    missing = [field for field in required.values() if field not in summary]
    if missing:
        raise AuditInputError(f"numeric summary is missing keys {missing}")
    values = {name: float(summary[field]) for name, field in required.items()}
    for name in ("total", "wins", "ties", "losses", "common", "sole", "nx_pairs", "gunrock_pairs"):
        values[name] = float(_as_int(values[name], name))
    if int(values["wins"] + values["ties"] + values["losses"]) != int(values["common"]):
        raise AuditInputError("headline win/tie/loss counts do not sum to common pairs")
    if int(values["common"] + values["sole"]) != int(values["total"]):
        raise AuditInputError("headline common plus sole-valid counts do not equal EGGPU total")
    if not all(
        _positive(values[name])
        for name in ("external_speedup", "nx_speedup", "gunrock_speedup")
    ):
        raise AuditInputError("headline speedups must be positive and finite")
    registry.annotate(
        "numeric_summary",
        headline={
            key: int(value) if float(value).is_integer() else value
            for key, value in values.items()
        },
    )
    return values


def _igraph_evidence(
    rows: list[dict[str, str]], registry: AssetRegistry
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["metric"] == "e2e_public_call" and row["function"] == "ALL"
    ]
    by_bucket = {row["size_bucket"]: row for row in selected}
    required_buckets = ("<1e5", "1e5--1e6", ">=1e6", "ALL")
    missing = [bucket for bucket in required_buckets if bucket not in by_bucket]
    if missing:
        raise AuditInputError(f"igraph crossover lacks ALL rows for buckets {missing}")
    parsed: dict[str, dict[str, float]] = {}
    for bucket in required_buckets:
        row = by_bucket[bucket]
        parsed[bucket] = {
            "common": float(_as_int(row["common_pairs"], f"{bucket}.common_pairs")),
            "wins": float(_as_int(row["strict_wins"], f"{bucket}.strict_wins")),
            "ties": float(_as_int(row["ties"], f"{bucket}.ties")),
            "losses": float(_as_int(row["losses"], f"{bucket}.losses")),
            "speedup": _as_float(
                row["igraph_over_eggpu_geomean"], f"{bucket}.speedup"
            ),
        }
        item = parsed[bucket]
        if int(item["wins"] + item["ties"] + item["losses"]) != int(item["common"]):
            raise AuditInputError(f"{bucket} igraph outcomes do not sum to common pairs")
    registry.annotate("igraph_crossover", selected_rows=len(parsed))
    return {"overall": parsed["ALL"], "buckets": parsed}


def _intro_evidence(
    rows: list[dict[str, str]], registry: AssetRegistry
) -> dict[str, Any]:
    by_key = {(row["system"], _as_int(row["scale"], "intro.scale")): row for row in rows}
    required = [
        ("EGGPU", 20),
        ("EGGPU", 22),
        ("EGGPU", 24),
        ("EGGPU", 26),
        ("igraph", 20),
        ("igraph", 22),
        ("igraph", 24),
        ("igraph", 26),
    ]
    missing = [key for key in required if key not in by_key]
    if missing:
        raise AuditInputError(f"Intro R-MAT asset lacks rows {missing}")

    for scale in (20, 22, 24, 26):
        eggpu = by_key[("EGGPU", scale)]
        if eggpu["status"].strip().lower() != "ok":
            raise AuditInputError(f"EGGPU S{scale} is not successful")
        if not str(eggpu.get("estimator", "")).strip():
            registry.invalidate(
                "intro_rmat", f"EGGPU S{scale} has no declared estimator"
            )
    for scale in (20, 22, 24):
        if by_key[("igraph", scale)]["status"].strip().lower() != "ok":
            raise AuditInputError(f"igraph S{scale} is not successful")
    if by_key[("igraph", 26)]["status"].strip().lower() == "ok":
        raise AuditInputError("Intro text expects the declared igraph S26 resource limit")
    registry.annotate(
        "intro_rmat",
        estimator_scope="standalone_intro_rmat_protocol",
        eggpu_estimators=sorted(
            {
                by_key[("EGGPU", scale)].get("estimator", "")
                for scale in (20, 22, 24, 26)
            }
        ),
    )

    speedups = []
    for scale in (20, 22, 24):
        eggpu = _as_float(
            by_key[("EGGPU", scale)]["pagerank_public_call_seconds"],
            f"EGGPU S{scale}",
        )
        igraph = _as_float(
            by_key[("igraph", scale)]["pagerank_public_call_seconds"],
            f"igraph S{scale}",
        )
        speedups.append(igraph / eggpu)
    return {
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
        "eggpu_first": _as_float(
            by_key[("EGGPU", 20)]["pagerank_public_call_seconds"], "EGGPU S20"
        ),
        "eggpu_last": _as_float(
            by_key[("EGGPU", 26)]["pagerank_public_call_seconds"], "EGGPU S26"
        ),
        "igraph_first": _as_float(
            by_key[("igraph", 20)]["pagerank_public_call_seconds"], "igraph S20"
        ),
        "igraph_last": _as_float(
            by_key[("igraph", 24)]["pagerank_public_call_seconds"], "igraph S24"
        ),
    }


def _memory_evidence(
    ledger: list[dict[str, str]],
    domain_rows: list[dict[str, str]],
    registry: AssetRegistry,
) -> dict[str, Any]:
    valid: dict[tuple[str, str, str], float] = {}
    for row in ledger:
        if row["baseline"] not in {"EGGPU", "nx-cugraph", "Gunrock"}:
            continue
        if row["execution_status"].strip().lower() != "ok":
            continue
        if not _validation_passes(row["validation_status"]):
            continue
        if not _positive(row.get("gpu_peak_mb_mean")):
            continue
        key = (row["dataset"], row["function"], row["baseline"])
        if key in valid:
            raise AuditInputError(f"duplicate memory ledger cell {key}")
        valid[key] = float(row["gpu_peak_mb_mean"])

    eggpu_keys = {
        (dataset, function)
        for dataset, function, baseline in valid
        if baseline == "EGGPU"
    }

    def compare(baseline: str) -> dict[str, float]:
        ratios = []
        for dataset, function in eggpu_keys:
            eggpu = valid.get((dataset, function, "EGGPU"))
            other = valid.get((dataset, function, baseline))
            if eggpu is not None and other is not None:
                ratios.append(other / eggpu)
        if not ratios:
            raise AuditInputError(f"no common EGGPU/{baseline} memory cells")
        return {
            "common": float(len(ratios)),
            "lower": float(sum(ratio > 1.0 for ratio in ratios)),
            "ratio": _geomean(ratios),
        }

    best_ratios = []
    for dataset, function in eggpu_keys:
        candidates = [
            valid[(dataset, function, baseline)]
            for baseline in ("nx-cugraph", "Gunrock")
            if (dataset, function, baseline) in valid
        ]
        if candidates:
            best_ratios.append(
                min(candidates) / valid[(dataset, function, "EGGPU")]
            )
    if not best_ratios:
        raise AuditInputError("no common best-GPU memory cells")
    max_mb = max(
        value
        for (dataset, function, baseline), value in valid.items()
        if baseline == "EGGPU"
    )
    evidence = {
        "nx": compare("nx-cugraph"),
        "gunrock": compare("Gunrock"),
        "best": {
            "common": float(len(best_ratios)),
            "lower": float(sum(ratio > 1.0 for ratio in best_ratios)),
            "ratio": _geomean(best_ratios),
        },
        "max_gib": max_mb / 1024.0,
        "eggpu_cells": float(len(eggpu_keys)),
    }

    domain = [
        row
        for row in domain_rows
        if row["baseline"] == "EGGPU"
        and row["resource"] == "process_gpu_peak_mb"
    ]
    if len(domain) != 1:
        registry.invalidate(
            "memory_domain_summary",
            f"expected one EGGPU process_gpu_peak_mb row, found {len(domain)}",
        )
    else:
        domain_max = _as_float(domain[0]["maximum_mb"], "memory domain maximum")
        domain_cells = _as_int(domain[0]["cells"], "memory domain cells")
        if not math.isclose(domain_max, max_mb, rel_tol=1e-9, abs_tol=1e-9):
            registry.invalidate(
                "memory_domain_summary",
                f"maximum_mb={domain_max} differs from ledger maximum {max_mb}",
            )
        if domain_cells != len(eggpu_keys):
            registry.invalidate(
                "memory_domain_summary",
                f"cells={domain_cells} differs from ledger EGGPU cells "
                f"{len(eggpu_keys)}",
            )
    return evidence


def _audit_ledger_minimum_declarations(
    ledger: list[dict[str, str]],
    registry: AssetRegistry,
    expected_successful_workloads: int | None,
) -> None:
    rows = [
        row
        for row in ledger
        if row["baseline"] == "EGGPU"
        and row["execution_status"].strip().lower() == "ok"
    ]
    if expected_successful_workloads is not None and len(rows) != expected_successful_workloads:
        registry.invalidate(
            "cell_ledger",
            f"successful EGGPU rows={len(rows)} but numeric summary declares "
            f"{expected_successful_workloads}",
        )
    bad_counts = [
        f"{row['dataset']}/{row['function']}={row['sample_count']}"
        for row in rows
        if _as_int(row["sample_count"], "ledger sample_count") != 5
    ]
    if bad_counts:
        registry.invalidate(
            "cell_ledger",
            f"{len(bad_counts)} EGGPU rows do not declare five timing samples; "
            f"examples={bad_counts[:5]}",
        )
    for metric in ("e2e", "kernel"):
        bad = [
            f"{row['dataset']}/{row['function']}:{row.get(f'{metric}_estimator')}"
            for row in rows
            if not _is_minimum_of_five(row.get(f"{metric}_estimator"))
        ]
        if bad:
            registry.invalidate(
                "cell_ledger",
                f"{len(bad)} EGGPU {metric} rows are not minimum-of-five; "
                f"examples={bad[:5]}",
            )


def _audit_variance_policy(
    policy: dict[str, Any],
    registry: AssetRegistry,
    max_over_median_limit: float,
    median_over_min_limit: float,
) -> set[str]:
    expected_strings = {
        "variance_policy": "catastrophic_outlier_guard",
        "eggpu_center": "minimum_of_five",
    }
    for key, expected in expected_strings.items():
        observed = _normalise_estimator(policy.get(key))
        if observed != expected:
            registry.invalidate(
                "timing_stability_policy",
                f"{key}={policy.get(key)!r}, expected {expected!r}",
            )
    try:
        sample_count = _as_int(
            policy.get("raw_sample_count"), "policy.raw_sample_count"
        )
    except AuditInputError as error:
        registry.invalidate("timing_stability_policy", str(error))
    else:
        if sample_count != 5:
            registry.invalidate(
                "timing_stability_policy",
                f"raw_sample_count={sample_count}, expected 5",
            )

    for key, expected in (
        ("max_over_median_limit", max_over_median_limit),
        ("median_over_min_limit", median_over_min_limit),
    ):
        try:
            observed = _as_float(policy.get(key), f"policy.{key}")
        except AuditInputError as error:
            registry.invalidate("timing_stability_policy", str(error))
            continue
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            registry.invalidate(
                "timing_stability_policy",
                f"{key}={observed}, active audit requires {expected}",
            )

    dispersion = policy.get("dispersion_statistics")
    if not isinstance(dispersion, list):
        registry.invalidate(
            "timing_stability_policy",
            "dispersion_statistics must list raw_mean, sample_standard_deviation, "
            "and coefficient_of_variation",
        )
        return {"e2e", "kernel"}
    normalised = {_normalise_estimator(value) for value in dispersion}
    required = {
        "raw_mean",
        "sample_standard_deviation",
        "coefficient_of_variation",
    }
    if not required.issubset(normalised):
        registry.invalidate(
            "timing_stability_policy",
            f"dispersion_statistics lacks {sorted(required - normalised)}",
        )
    declared_acceptance = policy.get("acceptance_metrics")
    if declared_acceptance is None:
        return {"e2e", "kernel"}
    if not isinstance(declared_acceptance, list):
        registry.invalidate(
            "timing_stability_policy",
            "acceptance_metrics must be a non-empty list drawn from e2e/kernel",
        )
        return {"e2e", "kernel"}
    acceptance_metrics = {
        _normalise_estimator(value) for value in declared_acceptance
    }
    if (
        not acceptance_metrics
        or not acceptance_metrics.issubset({"e2e", "kernel"})
        or "e2e" not in acceptance_metrics
    ):
        registry.invalidate(
            "timing_stability_policy",
            "acceptance_metrics must contain e2e and may additionally contain kernel",
        )
        return {"e2e", "kernel"}
    return acceptance_metrics


def _gap_evidence(
    rows: list[dict[str, str]],
    summary: dict[str, Any] | None,
    registry: AssetRegistry,
) -> dict[str, Any]:
    valid = {
        row["function"]: row
        for row in rows
        if _validation_passes(row["validation_status"])
    }
    required = {"SCC", "MST", "LCC", "KCore"} | STRUCTURAL_FUNCTIONS
    missing = sorted(required - set(valid))
    if missing:
        raise AuditInputError(f"GAP timing table lacks validated rows {missing}")

    estimator = None
    if summary:
        estimator = summary.get("timing_estimator")
    if estimator is None and rows:
        estimator = rows[0].get("estimator")
    if not _is_minimum_of_five(estimator):
        registry.invalidate(
            "gap_timing_summary",
            f"EGGPU GAP timing estimator is not minimum-of-five: {estimator!r}",
        )

    value_field = None
    for candidate in (
        "public_return_paper_seconds",
        "public_return_min_seconds",
        "public_return_seconds",
    ):
        if candidate in rows[0]:
            value_field = candidate
            break
    if value_field is None:
        registry.invalidate(
            "gap_timing_table",
            "minimum-of-five public-return field is missing; "
            "mean-only fields cannot support the final claim",
        )
        # Retain the old field only to expose textual mismatches in the same run.
        value_field = "public_return_mean_seconds"
        if value_field not in rows[0]:
            raise AuditInputError("GAP timing table has no public-return value field")

    values = {
        function: _as_float(row[value_field], f"GAP {function}")
        for function, row in valid.items()
    }
    structural = [values[function] for function in STRUCTURAL_FUNCTIONS]
    return {
        "validated": float(len(valid)),
        "SCC": values["SCC"],
        "MST": values["MST"],
        "LCC": values["LCC"],
        "KCore": values["KCore"],
        "structural_min": min(structural),
        "structural_max": max(structural),
    }


def _first_use_evidence(
    summary_rows: list[dict[str, str]],
    detail_rows: list[dict[str, str]],
    registry: AssetRegistry,
) -> dict[str, Any]:
    e2e = [row for row in summary_rows if row["metric"] == "e2e"]
    if len(e2e) != 4:
        raise AuditInputError(f"first-use summary expected four E2E families, got {len(e2e)}")
    detail_e2e = [row for row in detail_rows if row["metric"] == "e2e"]
    protocols = {
        "minimum_of_five": (
            ("first_use_paper_seconds", "first_use_min_seconds"),
            ("steady_paper_seconds", "steady_min_seconds"),
        ),
        "arithmetic_mean": (
            ("first_use_mean_seconds",),
            ("steady_mean_seconds",),
        ),
    }

    def fields_for(estimator: str) -> tuple[str, str] | None:
        first_candidates, steady_candidates = protocols[estimator]
        first = next(
            (field for field in first_candidates if field in detail_rows[0]),
            None,
        )
        steady = next(
            (field for field in steady_candidates if field in detail_rows[0]),
            None,
        )
        return (first, steady) if first and steady else None

    def matches(estimator: str) -> bool:
        fields = fields_for(estimator)
        if fields is None:
            return False
        first_field, steady_field = fields
        for row in e2e:
            family_details = [
                detail for detail in detail_e2e if detail["family"] == row["family"]
            ]
            if not family_details:
                return False
            first = _geomean(
                _as_float(item[first_field], first_field) for item in family_details
            )
            steady = _geomean(
                _as_float(item[steady_field], steady_field) for item in family_details
            )
            summary_first = _as_float(
                row["first_use_geomean_seconds"], "first-use summary first"
            )
            summary_steady = _as_float(
                row["steady_state_geomean_seconds"], "first-use summary steady"
            )
            claimed = _as_float(row["first_use_over_steady"], "first-use ratio")
            if not (
                math.isclose(summary_first, first, rel_tol=1e-8, abs_tol=1e-10)
                and math.isclose(
                    summary_steady, steady, rel_tol=1e-8, abs_tol=1e-10
                )
                and math.isclose(
                    claimed, first / steady, rel_tol=1e-8, abs_tol=1e-10
                )
            ):
                return False
        return True

    declared = {
        _estimator_family(row.get("estimator"))
        for row in e2e
        if row.get("estimator")
    }
    if len(declared) > 1:
        registry.invalidate(
            "first_use_summary",
            f"mixed within-dataset estimators across families: {sorted(declared)}",
        )
    if declared:
        estimator = next(iter(declared))
        if estimator not in protocols or not matches(estimator):
            registry.invalidate(
                "first_use_summary",
                f"summary does not match declared {estimator} detail fields",
            )
        provenance = "declared"
    else:
        matched = [name for name in protocols if matches(name)]
        if len(matched) != 1:
            registry.invalidate(
                "first_use_summary",
                f"could not uniquely infer estimator from detail fields: {matched}",
            )
            estimator = matched[0] if matched else "unresolved"
        else:
            estimator = matched[0]
        provenance = "inferred_from_summary_and_pair_details"
    registry.annotate(
        "first_use_summary",
        estimator_scope="standalone_first_use_protocol",
        declared_estimator=estimator,
        estimator_provenance=provenance,
    )
    ratios = [
        _as_float(row["first_use_over_steady"], f"{row['family']} ratio")
        for row in e2e
    ]
    return {"min": min(ratios), "max": max(ratios)}


def _ablation_evidence(
    rows: list[dict[str, str]], registry: AssetRegistry
) -> dict[str, float]:
    expected_modules = {
        "GraphContext",
        "C++ graph cache",
        "Result reconstruction",
        "CSR storage",
        "CSR traversal",
    }
    by_module = {row["module"]: row for row in rows}
    missing = sorted(expected_modules - set(by_module))
    if missing:
        raise AuditInputError(f"ablation table lacks modules {missing}")
    registry.annotate(
        "ablation",
        estimator_scope="standalone_ablation_protocol",
        declared_estimators=sorted(
            {
                _estimator_family(row.get("estimator"))
                for row in rows
                if row.get("estimator")
            }
        ),
    )
    result = {}
    for module in expected_modules:
        row = by_module[module]
        complete = _as_float(row["complete_value"], f"{module}.complete")
        reference = _as_float(
            row["ablated_or_reference_value"], f"{module}.reference"
        )
        ratio = _as_float(row["ratio"], f"{module}.ratio")
        if not math.isclose(ratio, reference / complete, rel_tol=1e-8, abs_tol=1e-10):
            registry.invalidate(
                "ablation",
                f"{module} ratio {ratio} differs from reference/complete "
                f"{reference / complete}",
            )
        result[module] = ratio
    return result


def _workflow_evidence(
    call_rows: list[dict[str, str]],
    cumulative_rows: list[dict[str, str]],
    provenance: dict[str, Any] | None,
    registry: AssetRegistry,
) -> dict[str, Any]:
    cumulative_estimators = sorted(
        {_estimator_family(row["estimator"]) for row in cumulative_rows}
    )
    call_estimators = sorted(
        {
            _estimator_family(row.get("estimator"))
            for row in call_rows
            if row.get("estimator")
        }
    )
    registry.annotate(
        "workflow_calls",
        estimator_scope="standalone_workflow_protocol",
        declared_estimators=call_estimators or cumulative_estimators,
    )
    registry.annotate(
        "workflow_cumulative",
        estimator_scope="standalone_workflow_protocol",
        declared_estimators=cumulative_estimators,
    )

    e2e = sorted(
        (row for row in call_rows if row["metric"] == "e2e"),
        key=lambda row: _as_int(row["call_position"], "workflow position"),
    )
    kernel = [row for row in call_rows if row["metric"] == "kernel"]
    if len(e2e) != 5 or len(kernel) != 5:
        raise AuditInputError(
            f"workflow expected five E2E and five kernel rows, got "
            f"{len(e2e)} and {len(kernel)}"
        )
    for row in call_rows:
        reuse = _as_float(row["reuse_seconds"], "workflow reuse")
        isolated = _as_float(row["isolated_seconds"], "workflow isolated")
        ratio = _as_float(row["isolated_over_reuse"], "workflow ratio")
        if not math.isclose(ratio, isolated / reuse, rel_tol=1e-8, abs_tol=1e-10):
            registry.invalidate(
                "workflow_calls",
                f"{row['function']}/{row['metric']} ratio does not equal "
                "isolated/reuse",
            )

    last_position = max(
        _as_int(row["call_position"], "workflow cumulative position")
        for row in cumulative_rows
    )
    final = {
        row["baseline"]: _as_float(
            row["cumulative_geomean_seconds"], "workflow cumulative"
        )
        for row in cumulative_rows
        if _as_int(row["call_position"], "workflow cumulative position")
        == last_position
    }
    if not {"EGGPU", "EGGPU-isolated"}.issubset(final):
        raise AuditInputError("workflow cumulative table lacks both final variants")

    if provenance:
        estimator = (
            provenance.get("statistics", {}).get("within_dataset_estimator")
            if isinstance(provenance.get("statistics"), dict)
            else None
        )
        registry.annotate(
            "workflow_provenance",
            estimator_scope="standalone_workflow_protocol",
            declared_estimator=_estimator_family(estimator),
            explicit_binary_bindings={
                key: value
                for key, value in provenance.items()
                if key.startswith("bound_to_")
            },
        )

    e2e_ratios = [
        _as_float(row["isolated_over_reuse"], "workflow E2E ratio") for row in e2e
    ]
    kernel_ratios = [
        _as_float(row["isolated_over_reuse"], "workflow kernel ratio")
        for row in kernel
    ]
    retained = final["EGGPU"]
    isolated = final["EGGPU-isolated"]
    return {
        "ratios": e2e_ratios,
        "kernel_min": float(_display(min(kernel_ratios), 2)),
        "kernel_max": _round_outward(max(kernel_ratios), 2, upper=True),
        "retained": retained,
        "isolated": isolated,
        "cumulative_ratio": isolated / retained,
        "first_share_percent": (
            _as_float(e2e[0]["reuse_seconds"], "workflow first call")
            / retained
            * 100.0
        ),
    }


def _audit_raw_eggpu_samples(
    ledger: list[dict[str, str]],
    samples: list[dict[str, str]],
    registry: AssetRegistry,
    max_over_median_limit: float,
    median_over_min_limit: float,
    acceptance_metrics: set[str] | None = None,
) -> dict[str, Any]:
    acceptance_metrics = acceptance_metrics or {"e2e", "kernel"}
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in samples:
        baseline = row.get("baseline", "EGGPU")
        if baseline != "EGGPU":
            continue
        if row.get("status", "ok").strip().lower() != "ok":
            continue
        grouped[(row["dataset"], row["function"], row["metric"])].append(row)

    issues: list[str] = []
    audited = 0
    dispersion: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in ledger:
        if row["baseline"] != "EGGPU" or row["execution_status"].lower() != "ok":
            continue
        if _as_int(row["sample_count"], "ledger sample_count") != 5:
            issues.append(
                f"{row['dataset']}/{row['function']}: ledger sample_count is "
                f"{row['sample_count']}, expected 5"
            )
        for metric in ("e2e", "kernel"):
            key = (row["dataset"], row["function"], metric)
            group = grouped.get(key, [])
            indices = []
            values = []
            for sample in group:
                indices.append(_as_int(sample["sample_index"], f"{key}.sample_index"))
                value = sample.get("seconds", sample.get("value"))
                values.append(_as_float(value, f"{key}.seconds"))
            if len(group) != 5 or sorted(indices) != [1, 2, 3, 4, 5]:
                issues.append(
                    f"{row['dataset']}/{row['function']}/{metric}: expected five "
                    f"raw samples with indices 1..5, got count={len(group)}, "
                    f"indices={sorted(indices)}"
                )
                continue
            audited += 1
            sample_mean = statistics.fmean(values)
            sample_std = statistics.stdev(values)
            sample_min = min(values)
            sample_median = statistics.median(values)
            sample_max = max(values)
            cv = sample_std / sample_mean if sample_mean > 0 else math.inf
            if sample_min < 0:
                issues.append(
                    f"{row['dataset']}/{row['function']}/{metric}: "
                    "negative timing sample"
                )
            paper = _as_float(
                row[f"{metric}_paper_seconds"], f"{key}.paper_seconds"
            )
            raw_mean = _as_float(
                row[f"{metric}_raw_mean_seconds"], f"{key}.raw_mean_seconds"
            )
            reported_std = _as_float(
                row[f"{metric}_std_seconds"], f"{key}.std_seconds"
            )
            estimator = row.get(f"{metric}_estimator")
            if not _is_minimum_of_five(estimator):
                issues.append(
                    f"{row['dataset']}/{row['function']}/{metric}: estimator "
                    f"{estimator!r} is not minimum-of-five"
                )
            for label, observed, expected in (
                ("paper_seconds", paper, sample_min),
                ("raw_mean_seconds", raw_mean, sample_mean),
                ("std_seconds", reported_std, sample_std),
            ):
                if not math.isclose(
                    observed, expected, rel_tol=1e-8, abs_tol=1e-12
                ):
                    issues.append(
                        f"{row['dataset']}/{row['function']}/{metric}: {label}="
                        f"{observed} differs from raw-derived {expected}"
                    )
            if sample_max == 0:
                max_over_median = 1.0
                median_over_min = 1.0
            else:
                max_over_median = (
                    sample_max / sample_median
                    if sample_median > 0
                    else math.inf
                )
                median_over_min = (
                    sample_median / sample_min if sample_min > 0 else math.inf
                )
            record = {
                "dataset": row["dataset"],
                "function": row["function"],
                "metric": metric,
                "raw_mean_seconds": sample_mean,
                "sample_std_seconds": sample_std,
                "cv": cv,
                "minimum_seconds": sample_min,
                "median_seconds": sample_median,
                "maximum_seconds": sample_max,
                "max_over_median": max_over_median,
                "median_over_min": median_over_min,
                "acceptance_gated": metric in acceptance_metrics,
            }
            dispersion.append(record)
            if (
                metric in acceptance_metrics
                and (
                    max_over_median > max_over_median_limit
                    or median_over_min > median_over_min_limit
                )
            ):
                rejected.append(record)
    if issues:
        for issue in issues:
            registry.invalidate("eggpu_raw_timing_samples", issue)
    if rejected:
        registry.invalidate(
            "eggpu_raw_timing_samples",
            f"{len(rejected)} EGGPU timing groups fail catastrophic-outlier "
            f"guard max/median<={max_over_median_limit:g} and "
            f"median/min<={median_over_min_limit:g}",
        )
    registry.annotate(
        "eggpu_raw_timing_samples",
        audited_groups=audited,
        variance_policy="catastrophic_outlier_guard",
        max_over_median_limit=max_over_median_limit,
        median_over_min_limit=median_over_min_limit,
        rejected_groups=rejected,
    )
    return {
        "audited_groups": audited,
        "dispersion_groups": dispersion,
        "rejected_groups": rejected,
        "formula_issues": issues,
        "acceptance_metrics": sorted(acceptance_metrics),
    }


def _parse_observed(value: str) -> float:
    return 1.0 if value.strip().lower() == "one" else float(value)


def _claim_rule(
    name: str,
    pattern: str,
    expected: Iterable[float],
    decimals: Iterable[int],
    source: str,
    *,
    required: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "pattern": re.compile(pattern, re.IGNORECASE | re.DOTALL),
        "expected": tuple(float(value) for value in expected),
        "decimals": tuple(decimals),
        "source": source,
        "required": required,
    }


def _build_claim_rules(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    headline = evidence.get("headline")
    if headline:
        rules.extend(
            [
                _claim_rule(
                    "headline_total_workloads",
                    r"(?:completes(?:\s+all)?|evaluate|covering(?:\s+all)?)\s+"
                    + r"(\d+)(?:\s+A100)?\s+(?:function--dataset\s+)?workloads",
                    [headline["total"]],
                    [0],
                    "final_13_numeric_summary.json",
                    required=True,
                ),
                _claim_rule(
                    "headline_fastest_or_tied",
                    r"fastest\s+or\s+tied\s+on\s+(\d+)\s+of\s+(\d+)",
                    [headline["wins"] + headline["ties"], headline["common"]],
                    [0, 0],
                    "final_13_numeric_summary.json",
                    required=True,
                ),
                _claim_rule(
                    "headline_sole_validated",
                    r"sole\s+validated\s+implementation\s+on\s+(\d+)"
                    r"\s+(?:more|additional)",
                    [headline["sole"]],
                    [0],
                    "final_13_numeric_summary.json",
                    required=True,
                ),
                _claim_rule(
                    "headline_best_external_speedup",
                    NUMBER
                    + TIMES
                    + r"\s+over\s+the\s+pairwise\s+best\s+"
                    r"(?:comparable\s+)?external",
                    [headline["external_speedup"]],
                    [2],
                    "final_13_numeric_summary.json",
                    required=True,
                ),
                _claim_rule(
                    "headline_nx_cugraph_speedup",
                    NUMBER + TIMES + r"\s+over\s+strict\s+nx-cugraph",
                    [headline["nx_speedup"]],
                    [2],
                    "final_13_numeric_summary.json",
                    required=True,
                ),
                _claim_rule(
                    "headline_nx_cugraph_pairs",
                    r"strict\s+nx-cugraph\s+on\s+(\d+)\s+calls",
                    [headline["nx_pairs"]],
                    [0],
                    "final_13_numeric_summary.json",
                ),
                _claim_rule(
                    "headline_gunrock_speedup",
                    NUMBER + TIMES + r"\s+faster\s+than\s+Gunrock",
                    [headline["gunrock_speedup"]],
                    [2],
                    "final_13_numeric_summary.json",
                    required=True,
                ),
                _claim_rule(
                    "headline_gunrock_pairs",
                    r"over\s+(\d+)\s+prepared-native\s+workloads",
                    [headline["gunrock_pairs"]],
                    [0],
                    "final_13_numeric_summary.json",
                ),
            ]
        )

    igraph = evidence.get("igraph")
    if igraph:
        overall = igraph["overall"]
        buckets = igraph["buckets"]
        rules.extend(
            [
                _claim_rule(
                    "igraph_outcomes",
                    r"Against\s+igraph.{0,120}?EGGPU\s+wins\s+(\d+),"
                    r"\s+ties\s+"
                    + INTEGER_OR_ONE
                    + r",\s+and\s+loses\s+(\d+)\s+of\s+(\d+)",
                    [
                        overall["wins"],
                        overall["ties"],
                        overall["losses"],
                        overall["common"],
                    ],
                    [0, 0, 0, 0],
                    "eggpu_vs_igraph_crossover_by_scale_function.csv",
                ),
                _claim_rule(
                    "igraph_size_bucket_speedups",
                    r"geometric-mean\s+speedup\s+grows\s+from\s+"
                    + NUMBER
                    + TIMES
                    + r"\s+below.{0,100}?\s+to\s+"
                    + NUMBER
                    + TIMES
                    + r"\s+at.{0,100}?\s+and\s+"
                    + NUMBER
                    + TIMES
                    + r"\s+above",
                    [
                        buckets["<1e5"]["speedup"],
                        buckets["1e5--1e6"]["speedup"],
                        buckets[">=1e6"]["speedup"],
                    ],
                    [2, 2, 2],
                    "eggpu_vs_igraph_crossover_by_scale_function.csv",
                ),
            ]
        )

    intro = evidence.get("intro")
    if intro:
        rules.extend(
            [
                _claim_rule(
                    "intro_rmat_speedup_range",
                    r"common\s+S20--S24\s+points.{0,180}?"
                    + NUMBER
                    + TIMES
                    + r"\s*--\s*"
                    + NUMBER
                    + TIMES
                    + r"\s+faster",
                    [intro["speedup_min"], intro["speedup_max"]],
                    [1, 1],
                    "intro_rmat_pagerank_uniform_mean.csv",
                ),
                _claim_rule(
                    "intro_eggpu_absolute_times",
                    r"EGGPU\s+grows\s+from\s+"
                    + NUMBER
                    + r"\s+seconds.{0,150}?\s+to\s+"
                    + NUMBER
                    + r"\s+seconds",
                    [intro["eggpu_first"], intro["eggpu_last"]],
                    [3, 3],
                    "intro_rmat_pagerank_uniform_mean.csv",
                ),
                _claim_rule(
                    "intro_igraph_absolute_times",
                    r"Igraph\s+grows\s+from\s+"
                    + NUMBER
                    + r"\s+seconds.{0,150}?\s+to\s+"
                    + NUMBER
                    + r"\s+seconds",
                    [intro["igraph_first"], intro["igraph_last"]],
                    [2, 2],
                    "intro_rmat_pagerank_uniform_mean.csv",
                ),
            ]
        )

    first_use = evidence.get("first_use")
    if first_use:
        rules.append(
            _claim_rule(
                "first_use_range",
                r"Across\s+the\s+four\s+families,\s+first\s+use\s+is\s+"
                + NUMBER
                + TIMES
                + r"\s*--\s*"
                + NUMBER
                + TIMES
                + r"\s+slower",
                [first_use["min"], first_use["max"]],
                [1, 1],
                "first_use_steady_actual_times.csv",
            )
        )

    ablation = evidence.get("ablation")
    if ablation:
        module_patterns = (
            (
                "ablation_graph_context",
                r"Removing\s+GraphContext.{0,180}?\s+by\s+" + NUMBER + TIMES,
                "GraphContext",
            ),
            (
                "ablation_cpp_cache",
                r"disabling\s+only\s+the\s+C\+\+\s+graph\s+cache"
                r".{0,180}?\s+by\s+"
                + NUMBER
                + TIMES,
                "C++ graph cache",
            ),
            (
                "ablation_result",
                r"Eager\s+Python\s+containers\s+raise\s+return\s+time\s+by\s+"
                + NUMBER
                + TIMES,
                "Result reconstruction",
            ),
            (
                "ablation_csr_storage",
                r"CSR\s+requires\s+" + NUMBER + TIMES + r"\s+less\s+host\s+storage",
                "CSR storage",
            ),
            (
                "ablation_csr_traversal",
                r"makes\s+the\s+tested\s+degree\s+traversal\s+"
                + NUMBER
                + TIMES
                + r"\s+faster",
                "CSR traversal",
            ),
        )
        rules.extend(
            _claim_rule(
                name,
                pattern,
                [ablation[module]],
                [2],
                "ablation_actual_values.csv",
            )
            for name, pattern, module in module_patterns
        )

    memory = evidence.get("memory")
    if memory:
        rules.extend(
            [
                _claim_rule(
                    "memory_nx_cugraph",
                    r"less\s+peak\s+device\s+memory\s+in\s+(\d+)\s+of\s+(\d+)"
                    r"\s+comparisons\s+with\s+nx-cugraph",
                    [memory["nx"]["lower"], memory["nx"]["common"]],
                    [0, 0],
                    "final_13_cell_outcome_ledger.csv",
                ),
                _claim_rule(
                    "memory_gunrock",
                    r"and\s+(\d+)\s+of\s+(\d+)\s+comparisons\s+with\s+Gunrock",
                    [memory["gunrock"]["lower"], memory["gunrock"]["common"]],
                    [0, 0],
                    "final_13_cell_outcome_ledger.csv",
                ),
                _claim_rule(
                    "memory_pairwise_best",
                    r"lower\s+on\s+(\d+)\s+of\s+(\d+)\s+workloads\s+with\s+a\s+"
                    r"geometric-mean\s+advantage\s+of\s+"
                    + NUMBER
                    + TIMES,
                    [
                        memory["best"]["lower"],
                        memory["best"]["common"],
                        memory["best"]["ratio"],
                    ],
                    [0, 0, 2],
                    "final_13_cell_outcome_ledger.csv",
                ),
                _claim_rule(
                    "memory_maximum_gib",
                    r"(?:maximum\s+device\s+footprint\s+of|"
                    r"Maximum\s+EGGPU\s+device\s+memory.{0,100}?)\s*"
                    + NUMBER
                    + r"\s*~?\s*GiB",
                    [memory["max_gib"]],
                    [2],
                    "memory_resource_domain_summary.csv",
                ),
            ]
        )

    gap = evidence.get("gap")
    if gap:
        rules.extend(
            [
                _claim_rule(
                    "gap_validated_functions",
                    r"(?:completes|validates)\s+all\s+(\d+)\s+"
                    r"(?:function\s+)?contracts\s+on\s+(?:this\s+graph|GAP-twitter)",
                    [gap["validated"]],
                    [0],
                    "gap_twitter timing table",
                ),
                *[
                    _claim_rule(
                        f"gap_{function.lower()}_seconds",
                        rf"\b{function}\s+in\s+" + NUMBER + r"\s*~?\s*s",
                        [gap[function]],
                        [2],
                        "gap_twitter timing table",
                    )
                    for function in ("SCC", "MST", "LCC", "KCore")
                ],
                _claim_rule(
                    "gap_structural_range",
                    r"four\s+structural-hole\s+measures\s+in\s+"
                    + NUMBER
                    + r"\s*--\s*"
                    + NUMBER
                    + r"\s*~?\s*s",
                    [gap["structural_min"], gap["structural_max"]],
                    [2, 2],
                    "gap_twitter timing table",
                ),
            ]
        )

    workflow = evidence.get("workflow")
    if workflow:
        ratios = workflow["ratios"]
        rules.extend(
            [
                _claim_rule(
                    "workflow_first_ratio",
                    r"Because\s+WCC\s+is\s+the\s+first\s+call,\s+its\s+ratio\s+"
                    r"is\s+only\s+"
                    + NUMBER
                    + TIMES,
                    [ratios[0]],
                    [2],
                    "workflow_five_call_self_control",
                ),
                _claim_rule(
                    "workflow_later_ratios",
                    r"later\s+calls\s+benefit\s+by\s+"
                    + NUMBER
                    + TIMES
                    + r",\s*"
                    + NUMBER
                    + TIMES
                    + r",\s*"
                    + NUMBER
                    + TIMES
                    + r",\s*and\s+"
                    + NUMBER
                    + TIMES,
                    ratios[1:],
                    [2, 2, 2, 2],
                    "workflow_five_call_self_control",
                ),
                _claim_rule(
                    "workflow_kernel_range",
                    r"kernel\s+ratios\s+remain\s+between\s+"
                    + NUMBER
                    + TIMES
                    + r"\s+and\s+"
                    + NUMBER
                    + TIMES,
                    [workflow["kernel_min"], workflow["kernel_max"]],
                    [2, 2],
                    "workflow_five_call_self_control",
                ),
                _claim_rule(
                    "workflow_cumulative",
                    r"reduces\s+latency\s+from\s+"
                    + NUMBER
                    + r"\s*~?\s*s\s+to\s+"
                    + NUMBER
                    + r"\s*~?\s*s,\s+a\s+"
                    + NUMBER
                    + TIMES
                    + r"\s+gain",
                    [
                        workflow["isolated"],
                        workflow["retained"],
                        workflow["cumulative_ratio"],
                    ],
                    [2, 2, 2],
                    "workflow_five_call_reuse_cumulative",
                ),
                _claim_rule(
                    "workflow_first_call_share",
                    r"first\s+call\s+accounts\s+for\s+about\s+(\d+)\s*\\%",
                    [workflow["first_share_percent"]],
                    [0],
                    "workflow_five_call_self_control",
                ),
            ]
        )
    return rules


def _paper_tex_files(paper_dir: Path) -> list[Path]:
    return sorted(path for path in paper_dir.rglob("*.tex") if path.is_file())


def _evaluate_claims(
    paper_dir: Path, rules: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    texts: list[tuple[Path, str]] = []
    read_errors: list[dict[str, str]] = []
    for path in _paper_tex_files(paper_dir):
        try:
            texts.append((path, path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError) as error:
            read_errors.append({"path": str(path), "error": str(error)})

    results = []
    for rule in rules:
        occurrences = []
        for path, text in texts:
            for match in rule["pattern"].finditer(text):
                observed = [_parse_observed(value) for value in match.groups()]
                expected = rule["expected"]
                decimals = rule["decimals"]
                matches = len(observed) == len(expected) and all(
                    _display(got, places) == _display(want, places)
                    for got, want, places in zip(observed, expected, decimals)
                )
                occurrences.append(
                    {
                        "path": str(path),
                        "line": text.count("\n", 0, match.start()) + 1,
                        "observed": observed,
                        "matches": matches,
                        "excerpt": " ".join(match.group(0).split())[:320],
                    }
                )
        if not occurrences:
            status = FAIL if rule["required"] else NOT_PRESENT
        else:
            status = PASS if all(item["matches"] for item in occurrences) else FAIL
        results.append(
            {
                "claim": rule["name"],
                "status": status,
                "required": rule["required"],
                "source": rule["source"],
                "expected": [
                    _display(value, places)
                    for value, places in zip(rule["expected"], rule["decimals"])
                ],
                "occurrences": occurrences,
            }
        )
    return results, read_errors


def _metadata_scalars(
    value: Any, key_path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = key_path + (str(key),)
            if isinstance(child, (dict, list)):
                yield from _metadata_scalars(child, path)
            else:
                yield path, child
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = key_path + (str(index),)
            if isinstance(child, (dict, list)):
                yield from _metadata_scalars(child, path)
            else:
                yield path, child


def _scan_submission_text(
    paper_dir: Path | None,
    assets_dir: Path,
    expected_max_over_median_limit: float,
    expected_median_over_min_limit: float,
    *,
    require_paper_declaration: bool = True,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, Any]],
]:
    files = set(_paper_tex_files(paper_dir)) if paper_dir is not None else set()
    asset_json_files = sorted(
        path for path in assets_dir.rglob("*.json") if path.is_file()
    )
    scan_roots = (assets_dir,) if paper_dir is None else (paper_dir, assets_dir)
    for root in scan_roots:
        files.update(
            path
            for path in root.rglob("*.json")
            if path.is_file() and "manifest" in path.name.lower()
        )
    findings = []
    errors = []
    timing_wording_classification = []
    paper_declares_main_matrix_minimum = False
    for path in sorted(files):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            errors.append({"path": str(path), "error": str(error)})
            continue
        for line_number, line in enumerate(lines, 1):
            for name, pattern in STALE_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "kind": "stale_wording",
                            "pattern": name,
                            "path": str(path),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )
            context = " ".join(
                lines[max(0, line_number - 3) : min(len(lines), line_number + 2)]
            ).lower()
            prefix_context = " ".join(
                lines[max(0, line_number - 8) : line_number]
            ).lower()
            minimum_in_supplement = (
                "r-mat" in prefix_context
                or "intro_rmat" in prefix_context
                or "workflow" in prefix_context
                or any(
                    marker in path.name.lower()
                    for marker in ("first_call", "first_use", "ablation")
                )
            )
            if (
                path.suffix.lower() == ".tex"
                and MINIMUM_OF_FIVE_PATTERN.search(line)
                and (
                    not minimum_in_supplement
                    or "main matrix" in line.lower()
                    or "main-matrix" in line.lower()
                )
            ):
                paper_declares_main_matrix_minimum = True
            universal_timing_statement = bool(
                re.search(
                    r"\b(?:each|every|all)\b.{0,80}\b(?:point|timing|call|run)"
                    r"|\bpoints?\s+are\b",
                    line,
                    re.IGNORECASE,
                )
            )
            global_estimator_statement = "ledger" in line.lower()
            line_lower = line.lower()
            explicit_competitor_only = (
                "eggpu" not in line_lower
                and any(
                    name in line_lower
                    for name in (
                        "igraph",
                        "networkx",
                        "nx-cugraph",
                        "gunrock",
                        "graphscope",
                        "easygraph cpu",
                        "easygraph c++",
                    )
                )
                and not universal_timing_statement
                and not global_estimator_statement
            )
            if path.suffix.lower() == ".tex" and ARITHMETIC_MEAN_PATTERN.search(line):
                standalone_supplement = (
                    "r-mat" in prefix_context
                    or "intro_rmat" in prefix_context
                    or "workflow" in prefix_context
                    or any(
                        marker in path.name.lower()
                        for marker in ("first_call", "first_use", "ablation")
                    )
                )
                explicit_main_matrix = (
                    "main matrix" in line_lower
                    or "main-matrix" in line_lower
                )
                explicit_estimator_sensitivity = (
                    "estimator sensitivity" in prefix_context
                    and "replacing every eggpu minimum" in prefix_context
                    and "same five raw samples" in context
                )
                if explicit_estimator_sensitivity:
                    classification = "eggpu_mean_counterfactual_allowed"
                elif explicit_competitor_only:
                    classification = "competitor_frozen_protocol_allowed"
                elif explicit_main_matrix:
                    classification = "main_matrix_eggpu_protocol_mismatch"
                elif standalone_supplement:
                    classification = "standalone_supplement_protocol_allowed"
                elif (
                    "eggpu" in context
                    or universal_timing_statement
                    or global_estimator_statement
                ):
                    classification = "main_matrix_eggpu_protocol_mismatch"
                else:
                    classification = "unscoped_arithmetic_mean_requires_review"
                record = {
                    "path": str(path),
                    "line": line_number,
                    "text": line.strip(),
                    "classification": classification,
                }
                timing_wording_classification.append(record)
                if classification in {
                    "main_matrix_eggpu_protocol_mismatch",
                    "unscoped_arithmetic_mean_requires_review",
                }:
                    findings.append(
                        {
                            "kind": "eggpu_protocol_mismatch",
                            "pattern": "arithmetic_mean_in_submission_tex",
                            **record,
                            "detail": (
                                "The main matrix must distinguish EGGPU's "
                                "minimum-of-five center from frozen competitor "
                                "means; standalone supplements retain their own "
                                "declared estimators."
                            ),
                        }
                    )
    for path in asset_json_files:
        try:
            metadata = _read_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for key_path, observed in _metadata_scalars(metadata):
            key_text = ".".join(key_path)
            key_lower = _normalise_estimator(key_text)
            competitor_only = (
                "eggpu" not in key_lower
                and any(
                    name in key_lower
                    for name in (
                        "igraph",
                        "networkx",
                        "nx_cugraph",
                        "gunrock",
                        "graphscope",
                        "easygraph_cpu",
                        "easygraph_cpp",
                    )
                )
            )
            if competitor_only:
                continue
            old_cv_key = (
                "cv" in key_lower
                and any(
                    token in key_lower
                    for token in ("threshold", "limit", "allowed", "max_cv")
                )
            )
            old_std_threshold_key = (
                (
                    "std" in key_lower
                    or bool(re.search(r"(?:^|_)sd(?:_|$)", key_lower))
                )
                and any(
                    token in key_lower
                    for token in ("threshold", "allowed", "limit", "gate_status")
                )
            )
            old_floor_key = (
                "floor" in key_lower
                and "second" in key_lower
                and (
                    "std" in key_lower
                    or "stability" in key_lower
                    or bool(re.search(r"(?:^|_)sd(?:_|$)", key_lower))
                )
            )
            if old_cv_key or old_std_threshold_key or old_floor_key:
                findings.append(
                    {
                        "kind": "eggpu_protocol_mismatch",
                        "pattern": "obsolete_sd_hard_gate_metadata",
                        "path": str(path),
                        "line": 0,
                        "text": key_text,
                        "observed": observed,
                        "detail": (
                            "SD/CV is retained for dispersion reporting but no "
                            "longer acts as the acceptance gate."
                        ),
                    }
                )
                continue
            ratio_limits = {
                "max_over_median_limit": expected_max_over_median_limit,
                "median_over_min_limit": expected_median_over_min_limit,
            }
            for suffix, expected in ratio_limits.items():
                if key_lower.endswith(suffix) and (
                    not _finite(observed)
                    or not math.isclose(
                        float(observed),
                        expected,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    findings.append(
                        {
                            "kind": "eggpu_protocol_mismatch",
                            "pattern": suffix,
                            "path": str(path),
                            "line": 0,
                            "text": key_text,
                            "observed": observed,
                            "expected": expected,
                            "detail": (
                                "Asset metadata ratio limit differs from the "
                                "active catastrophic-outlier guard."
                            ),
                        }
                    )
    if require_paper_declaration and not paper_declares_main_matrix_minimum:
        findings.append(
            {
                "kind": "eggpu_protocol_mismatch",
                "pattern": "missing_minimum_of_five_declaration",
                "path": str(paper_dir or "<paper-dir-not-supplied>"),
                "line": 0,
                "text": "",
                "detail": (
                    "Submission TeX does not explicitly declare EGGPU's "
                    "minimum-of-five center."
                ),
            }
        )
    return findings, errors, timing_wording_classification


def audit(
    paper_dir: Path | None,
    assets_dir: Path,
    *,
    max_over_median_limit: float = DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    median_over_min_limit: float = DEFAULT_MEDIAN_OVER_MIN_LIMIT,
    assets_only: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serialisable audit report without changing either input.

    ``assets_only`` is the pre-publication phase used before the paper tree is
    synchronized to a new release.  It runs every asset, raw-sample, estimator,
    memory, and protocol check, while deliberately deferring only assertions
    about prose that does not yet consume the release.
    """
    if not assets_only:
        if paper_dir is None:
            raise AuditInputError("paper directory is required outside assets-only mode")
        paper_dir = paper_dir.resolve()
    assets_dir = assets_dir.resolve()
    if not assets_only and (paper_dir is None or not paper_dir.is_dir()):
        raise AuditInputError(f"paper directory does not exist: {paper_dir}")
    if not assets_dir.is_dir():
        raise AuditInputError(f"assets directory does not exist: {assets_dir}")
    if max_over_median_limit < 1.0:
        raise AuditInputError("--max-over-median-limit must be at least 1")
    if median_over_min_limit < 1.0:
        raise AuditInputError("--median-over-min-limit must be at least 1")

    registry = AssetRegistry(assets_dir)
    evidence: dict[str, Any] = {}

    numeric = _load_json_asset(
        registry, "numeric_summary", ("final_13_numeric_summary.json",)
    )
    if isinstance(numeric, dict):
        try:
            evidence["headline"] = _headline_evidence(numeric, registry)
        except AuditInputError as error:
            registry.invalidate("numeric_summary", str(error))
    elif numeric is not None:
        registry.invalidate("numeric_summary", "expected a JSON object")

    crossover = _load_csv_asset(
        registry,
        "igraph_crossover",
        ("eggpu_vs_igraph_crossover_by_scale_function.csv",),
        (
            "metric",
            "size_bucket",
            "function",
            "common_pairs",
            "strict_wins",
            "ties",
            "losses",
            "igraph_over_eggpu_geomean",
        ),
    )
    if crossover:
        try:
            evidence["igraph"] = _igraph_evidence(crossover, registry)
        except AuditInputError as error:
            registry.invalidate("igraph_crossover", str(error))

    intro = _load_csv_asset(
        registry,
        "intro_rmat",
        ("intro_rmat_pagerank_uniform_mean.csv", "intro_rmat_pagerank.csv"),
        (
            "system",
            "scale",
            "pagerank_public_call_seconds",
            "status",
            "estimator",
        ),
    )
    if intro:
        try:
            evidence["intro"] = _intro_evidence(intro, registry)
        except AuditInputError as error:
            registry.invalidate("intro_rmat", str(error))

    ledger = _load_csv_asset(
        registry,
        "cell_ledger",
        ("final_13_cell_outcome_ledger.csv",),
        (
            "dataset",
            "function",
            "baseline",
            "execution_status",
            "validation_status",
            "sample_count",
            "e2e_paper_seconds",
            "e2e_raw_mean_seconds",
            "e2e_std_seconds",
            "e2e_estimator",
            "kernel_paper_seconds",
            "kernel_raw_mean_seconds",
            "kernel_std_seconds",
            "kernel_estimator",
            "gpu_peak_mb_mean",
        ),
    )
    if ledger:
        headline_total = (
            int(evidence["headline"]["total"])
            if "headline" in evidence
            else None
        )
        try:
            _audit_ledger_minimum_declarations(
                ledger, registry, headline_total
            )
        except AuditInputError as error:
            registry.invalidate("cell_ledger", str(error))
    memory_domain = _load_csv_asset(
        registry,
        "memory_domain_summary",
        ("memory_resource_domain_summary.csv",),
        ("baseline", "resource", "cells", "maximum_mb"),
    )
    if ledger and memory_domain:
        try:
            evidence["memory"] = _memory_evidence(
                ledger, memory_domain, registry
            )
        except AuditInputError as error:
            registry.invalidate("cell_ledger", f"memory audit: {error}")

    raw_samples = _load_csv_asset(
        registry,
        "eggpu_raw_timing_samples",
        (
            "final_13_eggpu_timing_samples.csv",
            "eggpu_five_run_timing_samples.csv",
            "eggpu_timing_samples.csv",
        ),
        ("dataset", "function", "metric", "sample_index", "seconds"),
    )
    variance_policy = _load_json_asset(
        registry,
        "timing_stability_policy",
        ("eggpu_timing_stability_policy.json",),
    )
    acceptance_metrics = {"e2e", "kernel"}
    if isinstance(variance_policy, dict):
        acceptance_metrics = _audit_variance_policy(
            variance_policy,
            registry,
            max_over_median_limit,
            median_over_min_limit,
        )
    elif variance_policy is not None:
        registry.invalidate(
            "timing_stability_policy", "expected a JSON object"
        )
    raw_sample_audit = None
    if ledger and raw_samples:
        try:
            raw_sample_audit = _audit_raw_eggpu_samples(
                ledger,
                raw_samples,
                registry,
                max_over_median_limit,
                median_over_min_limit,
                acceptance_metrics,
            )
        except AuditInputError as error:
            registry.invalidate("eggpu_raw_timing_samples", str(error))

    gap_rows = _load_csv_asset(
        registry,
        "gap_timing_table",
        ("gap_twitter_V10_timing_table.csv", "gap_twitter_v10_timing_table.csv"),
        ("function", "validation_status"),
        glob_pattern="gap_twitter*_timing_table.csv",
    )
    gap_summary = _load_json_asset(
        registry,
        "gap_timing_summary",
        ("gap_twitter_V10_timing_summary.json", "gap_twitter_v10_timing_summary.json"),
        glob_pattern="gap_twitter*_timing_summary.json",
    )
    if gap_summary is not None and not isinstance(gap_summary, dict):
        registry.invalidate("gap_timing_summary", "expected a JSON object")
    if gap_rows:
        try:
            evidence["gap"] = _gap_evidence(
                gap_rows,
                gap_summary if isinstance(gap_summary, dict) else None,
                registry,
            )
        except AuditInputError as error:
            registry.invalidate("gap_timing_table", str(error))

    first_use = _load_csv_asset(
        registry,
        "first_use_summary",
        ("first_use_steady_actual_times.csv",),
        (
            "family",
            "metric",
            "first_use_geomean_seconds",
            "steady_state_geomean_seconds",
            "first_use_over_steady",
        ),
    )
    first_details = _load_csv_asset(
        registry,
        "first_use_details",
        ("first_use_steady_pair_details.csv",),
        ("family", "metric"),
    )
    if first_use and first_details:
        try:
            evidence["first_use"] = _first_use_evidence(
                first_use, first_details, registry
            )
        except AuditInputError as error:
            registry.invalidate("first_use_summary", str(error))

    ablation = _load_csv_asset(
        registry,
        "ablation",
        ("ablation_actual_values.csv",),
        (
            "module",
            "axis",
            "complete_value",
            "ablated_or_reference_value",
            "ratio",
        ),
    )
    if ablation:
        try:
            evidence["ablation"] = _ablation_evidence(ablation, registry)
        except AuditInputError as error:
            registry.invalidate("ablation", str(error))

    workflow_calls = _load_csv_asset(
        registry,
        "workflow_calls",
        ("workflow_five_call_self_control_minimum_of_five.csv",),
        (
            "call_position",
            "function",
            "metric",
            "reuse_seconds",
            "isolated_seconds",
            "isolated_over_reuse",
        ),
        glob_pattern="workflow_five_call_self_control*.csv",
    )
    workflow_cumulative = _load_csv_asset(
        registry,
        "workflow_cumulative",
        ("workflow_five_call_reuse_cumulative_minimum_of_five.csv",),
        (
            "baseline",
            "call_position",
            "function",
            "cumulative_geomean_seconds",
            "estimator",
        ),
        glob_pattern="workflow_five_call_reuse_cumulative*.csv",
    )
    workflow_provenance = _load_json_asset(
        registry,
        "workflow_provenance",
        ("workflow_state_reuse_provenance.json",),
    )
    if workflow_provenance is not None and not isinstance(
        workflow_provenance, dict
    ):
        registry.invalidate("workflow_provenance", "expected a JSON object")
    if workflow_calls and workflow_cumulative:
        try:
            evidence["workflow"] = _workflow_evidence(
                workflow_calls,
                workflow_cumulative,
                (
                    workflow_provenance
                    if isinstance(workflow_provenance, dict)
                    else None
                ),
                registry,
            )
        except AuditInputError as error:
            registry.invalidate("workflow_calls", str(error))

    if assets_only:
        claim_checks: list[dict[str, Any]] = []
        claim_read_errors: list[dict[str, str]] = []
    else:
        assert paper_dir is not None
        claim_rules = _build_claim_rules(evidence)
        claim_checks, claim_read_errors = _evaluate_claims(
            paper_dir, claim_rules
        )
    (
        stale_findings,
        scan_read_errors,
        timing_wording_classification,
    ) = _scan_submission_text(
        None if assets_only else paper_dir,
        assets_dir,
        max_over_median_limit,
        median_over_min_limit,
        require_paper_declaration=not assets_only,
    )

    asset_failures = [
        name
        for name, result in registry.results.items()
        if result["status"] in {MISSING, INVALID}
    ]
    claim_failures = [
        check["claim"] for check in claim_checks if check["status"] == FAIL
    ]
    read_errors = claim_read_errors + scan_read_errors
    passed = not asset_failures and not claim_failures and not stale_findings and not read_errors
    return {
        "status": PASS if passed else FAIL,
        "audit_scope": "assets_only" if assets_only else "paper_and_assets",
        "paper_sync_pending": bool(assets_only),
        "paper_dir": None if assets_only else str(paper_dir),
        "assets_dir": str(assets_dir),
        "protocol": {
            "main_matrix_eggpu_center": "minimum_of_five_independent_runs",
            "standalone_supplement_centers": "as_declared_by_each_asset",
            "dispersion_statistics": [
                "raw_mean",
                "sample_standard_deviation",
                "coefficient_of_variation",
            ],
            "variance_policy": "catastrophic_outlier_guard",
            "max_over_median_limit": max_over_median_limit,
            "median_over_min_limit": median_over_min_limit,
            "competitor_estimators": "as_declared_by_frozen_assets",
        },
        "assets": registry.results,
        "missing_assets": sorted(
            name
            for name, result in registry.results.items()
            if result["status"] == MISSING
        ),
        "invalid_assets": sorted(
            name
            for name, result in registry.results.items()
            if result["status"] == INVALID
        ),
        "raw_sample_audit": raw_sample_audit,
        "claim_checks": claim_checks,
        "stale_or_protocol_wording": stale_findings,
        "timing_wording_classification": timing_wording_classification,
        "read_errors": read_errors,
        "summary": {
            "asset_failures": len(asset_failures),
            "claim_failures": len(claim_failures),
            "stale_or_protocol_wording": len(stale_findings),
            "read_errors": len(read_errors),
        },
    }


def _print_human_summary(report: dict[str, Any], stream: Any) -> None:
    summary = report["summary"]
    print(
        f"V10 paper-claim audit: {report['status'].upper()} "
        f"(asset_failures={summary['asset_failures']}, "
        f"claim_failures={summary['claim_failures']}, "
        f"wording_findings={summary['stale_or_protocol_wording']}, "
        f"read_errors={summary['read_errors']})",
        file=stream,
    )
    for name in report["missing_assets"]:
        print(f"MISSING asset: {name}", file=stream)
    for name in report["invalid_assets"]:
        print(f"INVALID asset: {name}", file=stream)
    for check in report["claim_checks"]:
        if check["status"] == FAIL:
            print(
                f"CLAIM {check['claim']}: expected {check['expected']}",
                file=stream,
            )
    for finding in report["stale_or_protocol_wording"]:
        print(
            f"WORDING {finding['pattern']}: "
            f"{finding['path']}:{finding['line']}",
            file=stream,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit V10 paper claims and minimum-of-five evidence."
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        help="Submission tree. Required unless --assets-only is selected.",
    )
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument(
        "--assets-only",
        action="store_true",
        help=(
            "Run the strict numerical/provenance asset gate before the paper "
            "tree is synchronized; prose claim checks remain pending."
        ),
    )
    parser.add_argument(
        "--max-over-median-limit",
        type=float,
        default=DEFAULT_MAX_OVER_MEDIAN_LIMIT,
        help="Reject a five-run group above this max/median ratio (default: 5).",
    )
    parser.add_argument(
        "--median-over-min-limit",
        type=float,
        default=DEFAULT_MEDIAN_OVER_MIN_LIMIT,
        help="Reject a five-run group above this median/min ratio (default: 3).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the complete JSON report; stdout remains concise.",
    )
    args = parser.parse_args(argv)
    if not args.assets_only and args.paper_dir is None:
        parser.error("--paper-dir is required unless --assets-only is selected")
    try:
        report = audit(
            args.paper_dir,
            args.assets_dir,
            max_over_median_limit=args.max_over_median_limit,
            median_over_min_limit=args.median_over_min_limit,
            assets_only=args.assets_only,
        )
    except AuditInputError as error:
        print(f"V10 paper-claim audit input error: {error}", file=sys.stderr)
        return 2

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _print_human_summary(report, sys.stdout)
    return 0 if report["status"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
