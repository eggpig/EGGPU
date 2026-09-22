#!/usr/bin/env python3
"""Audit five-sample EGGPU timing batches before paper aggregation.

The submission-facing EGGPU center value may be the minimum of five, but a
minimum is accepted only when the complete five-sample batch is present and
passes an estimator-independent catastrophic-outlier gate.  The default gate
is:

    maximum / median <= 5.0 and median / minimum <= 3.0

For a nonzero batch, a zero minimum fails.  Mean, sample standard deviation,
and CV are recorded as diagnostics but do not decide acceptance.  This keeps
the acceptance decision independent of how fast the minimum is while allowing
ordinary run-to-run variance.  The median/min threshold is calibrated from two
independent com-youtube/Constraint batches: both reproduce an approximately
2.26x E2E split while the rerun's device kernel remains within approximately
120--129 ms.  The split is therefore attributed to repeatable host-return and
materialization bimodality rather than pathological device-time jitter.
Regular final-13 runs are read from ``results_samples.csv``.  Scale-anchor
runs are read from the five-worker aggregate ``*_timing.json`` records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

from stable_timing_protocol import (
    DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    DEFAULT_MEDIAN_OVER_MIN_LIMIT,
    EXPECTED_TIMING_SAMPLES,
    FORMAL_BATCH_SELECTION_POLICY,
    GATE_CALIBRATION,
    PAPER_ESTIMATOR,
    VARIANCE_POLICY,
    StableTimingProtocolError,
    summarize_five_samples,
    validate_gate_calibration,
)
from update_final13_eggpu_uniform_20260729 import (
    GateError,
    candidate_sha_from_result_dir,
    runtime_python_snapshot_from_result_dir,
)

METRIC_TO_ANCHOR_KEY = {
    "build": "load",
    "e2e": "steady_e2e",
    "kernel": "steady_kernel",
}
NUMBERED_TIMING_RE = re.compile(r"_timing_\d+\.json$")
DEFAULT_VARIANCE_POLICY = VARIANCE_POLICY
REPLACEMENT_SELECTION_BASIS = (
    "failed_catastrophic_outlier_batch_remeasurement"
)


class StabilityError(ValueError):
    """An input batch cannot be audited without guessing."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StabilityError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise StabilityError(f"{label} is not finite: {value!r}")
    return result


def integer(value: object, label: str) -> int:
    result = finite_number(value, label)
    if int(result) != result:
        raise StabilityError(f"{label} is not an integer: {value!r}")
    return int(result)


def summarize_batch(
    *,
    dataset: str,
    function: str,
    metric: str,
    source_kind: str,
    result_source: Path,
    evidence_path: Path,
    values: Iterable[float],
    expected_samples: int,
    max_over_median_limit: float,
    median_over_min_limit: float,
) -> dict[str, object]:
    samples = [finite_number(value, "sample") for value in values]
    if len(samples) != expected_samples:
        raise StabilityError(
            f"{dataset}/{function}/{metric}: {len(samples)} samples, "
            f"expected {expected_samples}"
        )
    if any(value < 0 for value in samples):
        raise StabilityError(f"{dataset}/{function}/{metric}: negative sample")
    try:
        protocol = summarize_five_samples(
            samples,
            max_over_median_limit=max_over_median_limit,
            median_over_min_limit=median_over_min_limit,
        )
    except StableTimingProtocolError as exc:
        raise StabilityError(
            f"{dataset}/{function}/{metric}: {exc}"
        ) from exc
    mean = protocol["arithmetic_mean_seconds"]
    stdev = protocol["sample_std_seconds"]
    minimum = protocol["minimum_seconds"]
    maximum = protocol["maximum_seconds"]
    median = protocol["median_seconds"]
    cv = protocol["coefficient_of_variation"]
    max_over_median = protocol["max_over_median"]
    median_over_minimum = protocol["median_over_min"]
    stable = protocol["stability_status"] == "pass"
    raw5 = json.dumps(samples, separators=(",", ":"))
    return {
        "dataset": dataset,
        "function": function,
        "metric": metric,
        "source_kind": source_kind,
        "result_source": str(result_source.resolve()),
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": sha256(evidence_path),
        "sample_count": len(samples),
        "raw5_seconds": raw5,
        "samples_seconds": raw5,
        "minimum_seconds": minimum,
        "arithmetic_mean_seconds": mean,
        "median_seconds": median,
        "maximum_seconds": maximum,
        "sample_std_seconds": stdev,
        "coefficient_of_variation": cv,
        "max_over_median": max_over_median,
        "median_over_minimum": median_over_minimum,
        "max_over_min": (
            maximum / minimum
            if minimum > 0
            else (1.0 if maximum == 0 else math.inf)
        ),
        "variance_policy": DEFAULT_VARIANCE_POLICY,
        "paper_estimator": PAPER_ESTIMATOR,
        "sample_std_and_cv_role": (
            "reported_diagnostics_not_acceptance_gate"
        ),
        "max_over_median_limit": max_over_median_limit,
        "median_over_min_limit": median_over_min_limit,
        "stability_status": "pass" if stable else "fail",
        "batch_acceptance_status": protocol["batch_acceptance_status"],
        "submission_seconds": (
            minimum if stable else ""
        ),
        "failure_reasons": json.dumps(
            protocol["failure_reasons"], separators=(",", ":")
        ),
    }


def main_rows(
    result_dir: Path,
    metric: str,
    expected_samples: int,
    max_over_median_limit: float,
    median_over_min_limit: float,
) -> list[dict[str, object]]:
    result_dir = result_dir.resolve()
    evidence = result_dir / "results_samples.csv"
    if not evidence.is_file():
        phased = (
            result_dir
            / "measurement_passes"
            / "timing"
            / "results_samples.csv"
        )
        if phased.is_file():
            evidence = phased
        else:
            raise StabilityError(f"{result_dir}: missing results_samples.csv")
    groups: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in read_csv(evidence):
        if row.get("baseline") != "EGGPU" or row.get("metric") != metric:
            continue
        if row.get("status") != "ok":
            # Large exact-Closeness cells may be deliberately skipped by the
            # scale guard and supplied by a separate bounded-source timing
            # directory.  A skipped/error row is never counted as a sample;
            # completeness is enforced after all input directories are merged.
            continue
        phase = str(row.get("measurement_phase", "")).strip()
        if phase not in {"", "timing"}:
            continue
        key = (str(row.get("dataset", "")), str(row.get("function", "")))
        sample_index = integer(row.get("sample_index"), "sample_index")
        value = finite_number(
            row.get("value", row.get("seconds")),
            f"{key}/{metric}.value",
        )
        groups.setdefault(key, []).append((sample_index, value))
    if not groups:
        raise StabilityError(f"{result_dir}: no EGGPU {metric} timing samples")
    output = []
    for (dataset, function), indexed in sorted(groups.items()):
        indices = sorted(index for index, _value in indexed)
        expected_indices = list(range(1, expected_samples + 1))
        if indices != expected_indices:
            raise StabilityError(
                f"{dataset}/{function}/{metric}: sample indices {indices}, "
                f"expected {expected_indices}"
            )
        values = [value for _index, value in sorted(indexed)]
        output.append(
            summarize_batch(
                dataset=dataset,
                function=function,
                metric=metric,
                source_kind="main",
                result_source=result_dir,
                evidence_path=evidence,
                values=values,
                expected_samples=expected_samples,
                max_over_median_limit=max_over_median_limit,
                median_over_min_limit=median_over_min_limit,
            )
        )
    return output


def anchor_rows(
    result_dir: Path,
    metric: str,
    expected_samples: int,
    max_over_median_limit: float,
    median_over_min_limit: float,
) -> list[dict[str, object]]:
    result_dir = result_dir.resolve()
    raw_dir = result_dir / "raw"
    if not raw_dir.is_dir():
        raise StabilityError(f"{result_dir}: missing raw directory")
    aggregate_key = METRIC_TO_ANCHOR_KEY[metric]
    output = []
    for evidence in sorted(raw_dir.glob("*_timing.json")):
        if NUMBERED_TIMING_RE.search(evidence.name):
            continue
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        dataset = str(payload.get("dataset", ""))
        function = str(payload.get("function", ""))
        aggregate = payload.get(aggregate_key)
        if not dataset or not function or not isinstance(aggregate, dict):
            continue
        samples = aggregate.get("samples")
        if not isinstance(samples, list):
            raise StabilityError(
                f"{evidence}: {aggregate_key}.samples is missing"
            )
        output.append(
            summarize_batch(
                dataset=dataset,
                function=function,
                metric=metric,
                source_kind="anchor",
                result_source=result_dir,
                evidence_path=evidence,
                values=samples,
                expected_samples=expected_samples,
                max_over_median_limit=max_over_median_limit,
                median_over_min_limit=median_over_min_limit,
            )
        )
    if not output:
        raise StabilityError(
            f"{result_dir}: no complete anchor {metric} timing aggregates"
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def workload_key(row: dict[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("function", "")),
        str(row.get("metric", "")),
    )


def load_replacement_manifest(
    path: Path | None,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Load an explicit failed-batch replacement map from JSON or CSV."""

    if path is None:
        return [], {
            "path": "",
            "sha256": "",
            "selection_basis": REPLACEMENT_SELECTION_BASIS,
        }
    resolved = path.resolve(strict=True)
    if resolved.suffix.lower() == ".csv":
        entries = read_csv(resolved)
    else:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            entries = payload.get("replacements")
        else:
            entries = None
        if not isinstance(entries, list):
            raise StabilityError(
                f"{resolved}: expected a JSON list or an object with "
                "`replacements`"
            )
    normalized = []
    for index, raw in enumerate(entries, 1):
        if not isinstance(raw, dict):
            raise StabilityError(
                f"{resolved}: replacement entry {index} is not an object"
            )
        entry = {
            str(key): str(value or "").strip()
            for key, value in raw.items()
        }
        required = (
            "dataset",
            "function",
            "metric",
            "original_result_source",
            "replacement_result_source",
            "reason",
            "selection_basis",
        )
        missing = [field for field in required if not entry.get(field)]
        if missing:
            raise StabilityError(
                f"{resolved}: replacement entry {index} is missing {missing}"
            )
        if entry["selection_basis"] != REPLACEMENT_SELECTION_BASIS:
            raise StabilityError(
                f"{resolved}: replacement entry {index} selection_basis must "
                f"be {REPLACEMENT_SELECTION_BASIS!r}; performance-based batch "
                "selection is forbidden"
            )
        entry["original_result_source"] = str(
            Path(entry["original_result_source"]).resolve()
        )
        entry["replacement_result_source"] = str(
            Path(entry["replacement_result_source"]).resolve()
        )
        if (
            entry["original_result_source"]
            == entry["replacement_result_source"]
        ):
            raise StabilityError(
                f"{resolved}: replacement entry {index} names the same "
                "original and replacement source"
            )
        normalized.append(entry)
    return normalized, {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "selection_basis": REPLACEMENT_SELECTION_BASIS,
    }


def apply_explicit_replacements(
    rows: list[dict[str, object]],
    entries: list[dict[str, str]],
    manifest_identity: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Resolve duplicates only through a declared failed-to-passing rerun."""

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(workload_key(row), []).append(row)
    entry_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for entry in entries:
        key = (entry["dataset"], entry["function"], entry["metric"])
        if key in entry_by_key:
            raise StabilityError(
                f"replacement manifest repeats workload key {key}"
            )
        entry_by_key[key] = entry

    selected: list[dict[str, object]] = []
    replacements: list[dict[str, object]] = []
    for key in sorted(groups):
        candidates = groups[key]
        entry = entry_by_key.get(key)
        selection_fields = {
            "batch_selection_status": "unique_input_batch",
            "replacement_selection_basis": "",
            "replacement_reason": "",
            "original_result_source": "",
            "original_evidence_path": "",
            "original_evidence_sha256": "",
            "original_stability_status": "",
            "replacement_manifest_path": "",
            "replacement_manifest_sha256": "",
        }
        if len(candidates) == 1:
            if entry is not None:
                raise StabilityError(
                    f"replacement manifest names non-duplicate key {key}"
                )
            chosen = dict(candidates[0])
            chosen.update(selection_fields)
            selected.append(chosen)
            continue
        if len(candidates) != 2:
            raise StabilityError(
                f"workload key {key} has {len(candidates)} batches; explicit "
                "replacement supports exactly one original and one rerun"
            )
        if entry is None:
            raise StabilityError(
                f"duplicate workload key {key} requires "
                "--replacement-manifest; implicit or fastest-batch selection "
                "is forbidden"
            )
        by_source = {
            str(candidate["result_source"]): candidate
            for candidate in candidates
        }
        if len(by_source) != len(candidates):
            raise StabilityError(
                f"workload key {key} repeats a result source"
            )
        original = by_source.get(entry["original_result_source"])
        replacement = by_source.get(entry["replacement_result_source"])
        if original is None or replacement is None:
            raise StabilityError(
                f"replacement manifest sources for {key} do not exactly match "
                f"the duplicate inputs; observed={sorted(by_source)}"
            )
        if original["stability_status"] != "fail":
            raise StabilityError(
                f"replacement original for {key} is not a failed "
                "catastrophic-outlier batch; speed-based replacement is forbidden"
            )
        if replacement["stability_status"] != "pass":
            raise StabilityError(
                f"replacement rerun for {key} does not pass the same guard"
            )
        for field, candidate in (
            ("original_evidence_sha256", original),
            ("replacement_evidence_sha256", replacement),
        ):
            expected_sha = entry.get(field, "")
            if expected_sha and expected_sha.lower() != str(
                candidate["evidence_sha256"]
            ).lower():
                raise StabilityError(
                    f"replacement manifest {field} mismatch for {key}"
                )
        chosen = dict(replacement)
        chosen.update(
            {
                **selection_fields,
                "batch_selection_status": (
                    "explicit_failed_batch_replacement"
                ),
                "replacement_selection_basis": entry["selection_basis"],
                "replacement_reason": entry["reason"],
                "original_result_source": original["result_source"],
                "original_evidence_path": original["evidence_path"],
                "original_evidence_sha256": original["evidence_sha256"],
                "original_stability_status": original["stability_status"],
                "replacement_manifest_path": manifest_identity["path"],
                "replacement_manifest_sha256": manifest_identity["sha256"],
            }
        )
        selected.append(chosen)
        replacements.append(
            {
                "dataset": key[0],
                "function": key[1],
                "metric": key[2],
                "selection_basis": entry["selection_basis"],
                "reason": entry["reason"],
                "original_result_source": original["result_source"],
                "original_evidence_path": original["evidence_path"],
                "original_evidence_sha256": original["evidence_sha256"],
                "original_stability_status": original["stability_status"],
                "original_failure_reasons": json.loads(
                    str(original["failure_reasons"])
                ),
                "replacement_result_source": replacement["result_source"],
                "replacement_evidence_path": replacement["evidence_path"],
                "replacement_evidence_sha256": replacement[
                    "evidence_sha256"
                ],
                "replacement_stability_status": replacement[
                    "stability_status"
                ],
            }
        )
    unused = sorted(set(entry_by_key) - set(groups))
    if unused:
        raise StabilityError(
            f"replacement manifest contains keys absent from inputs: {unused}"
        )
    return selected, replacements


def evidence_inventory(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Bind every input batch, including a superseded failed batch."""

    by_path: dict[str, dict[str, object]] = {}
    for row in rows:
        evidence_path = str(row["evidence_path"])
        evidence = {
            "source_kind": row["source_kind"],
            "result_source": row["result_source"],
            "evidence_path": evidence_path,
            "evidence_sha256": row["evidence_sha256"],
        }
        prior = by_path.get(evidence_path)
        if prior is not None and prior != evidence:
            raise StabilityError(
                f"conflicting stability evidence identity: {evidence_path}"
            )
        by_path[evidence_path] = evidence
    return [by_path[path] for path in sorted(by_path)]


def result_identity(result_dir: Path) -> dict[str, object]:
    try:
        candidate_sha, _candidate_source = candidate_sha_from_result_dir(
            result_dir, None
        )
        runtime_sha, _runtime_source, package_is_symlink = (
            runtime_python_snapshot_from_result_dir(
                result_dir,
                required=True,
            )
        )
    except (OSError, json.JSONDecodeError, GateError) as exc:
        raise StabilityError(
            f"{result_dir}: cannot bind timing stability to a frozen runtime: "
            f"{exc}"
        ) from exc
    if package_is_symlink is not False:
        raise StabilityError(
            f"{result_dir}: timing stability runtime is not proven non-symlinked"
        )
    return {
        "result_dir": str(result_dir.resolve()),
        "candidate_sha256": candidate_sha,
        "runtime_python_snapshot_sha256": runtime_sha,
        "runtime_package_is_symlink": False,
    }


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-result-dir",
        action="append",
        default=[],
        type=Path,
    )
    parser.add_argument(
        "--anchor-result-dir",
        action="append",
        default=[],
        type=Path,
    )
    parser.add_argument(
        "--metric",
        choices=tuple(METRIC_TO_ANCHOR_KEY),
        default="e2e",
    )
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=EXPECTED_TIMING_SAMPLES,
    )
    parser.add_argument(
        "--max-over-median-limit",
        type=float,
        default=DEFAULT_MAX_OVER_MEDIAN_LIMIT,
        help="Catastrophic high-outlier ratio limit (formal default: 5.0).",
    )
    parser.add_argument(
        "--median-over-min-limit",
        type=float,
        default=DEFAULT_MEDIAN_OVER_MIN_LIMIT,
        help=(
            "Catastrophic low-outlier ratio limit (calibrated formal "
            "default: 3.0)."
        ),
    )
    parser.add_argument("--expected-cells", type=int)
    parser.add_argument(
        "--replacement-manifest",
        type=Path,
        help=(
            "Diagnostic-only JSON/CSV mapping from a failed catastrophic-"
            "outlier batch to one explicit full rerun. The formal V10 gate "
            "omits this option and requires one unique original batch per key."
        ),
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write the audit but exit zero when unstable cells exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_samples != EXPECTED_TIMING_SAMPLES:
        raise SystemExit(
            f"--expected-samples must be exactly {EXPECTED_TIMING_SAMPLES}"
        )
    if args.max_over_median_limit < 1:
        raise SystemExit("--max-over-median-limit must be at least 1")
    if args.median_over_min_limit < 1:
        raise SystemExit("--median-over-min-limit must be at least 1")
    if not args.main_result_dir and not args.anchor_result_dir:
        raise SystemExit("provide at least one result directory")
    try:
        gate_calibration = validate_gate_calibration(
            GATE_CALIBRATION,
            repo_root=Path(__file__).resolve().parents[1],
        )
    except (OSError, StableTimingProtocolError) as exc:
        raise SystemExit(
            f"publication gate calibration validation failed: {exc}"
        ) from exc

    rows: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    try:
        for result_dir in args.main_result_dir:
            identities.append(result_identity(result_dir.resolve()))
            rows.extend(
                main_rows(
                    result_dir,
                    args.metric,
                    args.expected_samples,
                    args.max_over_median_limit,
                    args.median_over_min_limit,
                )
            )
        for result_dir in args.anchor_result_dir:
            identities.append(result_identity(result_dir.resolve()))
            rows.extend(
                anchor_rows(
                    result_dir,
                    args.metric,
                    args.expected_samples,
                    args.max_over_median_limit,
                    args.median_over_min_limit,
                )
            )
    except (OSError, json.JSONDecodeError, StabilityError) as exc:
        raise SystemExit(str(exc)) from exc

    all_input_rows = rows
    try:
        replacement_entries, replacement_manifest = (
            load_replacement_manifest(args.replacement_manifest)
        )
        rows, batch_overrides_applied = apply_explicit_replacements(
            all_input_rows,
            replacement_entries,
            replacement_manifest,
        )
        input_evidence = evidence_inventory(all_input_rows)
    except (OSError, json.JSONDecodeError, StabilityError) as exc:
        raise SystemExit(str(exc)) from exc
    keys = [workload_key(row) for row in rows]
    if args.expected_cells is not None and len(rows) != args.expected_cells:
        raise SystemExit(
            f"audited {len(rows)} cells, expected {args.expected_cells}"
        )

    rows.sort(key=lambda row: (row["dataset"], row["function"], row["metric"]))
    candidate_shas = sorted(
        {str(identity["candidate_sha256"]) for identity in identities}
    )
    runtime_shas = sorted(
        {
            str(identity["runtime_python_snapshot_sha256"])
            for identity in identities
        }
    )
    if len(candidate_shas) != 1 or len(runtime_shas) != 1:
        raise SystemExit(
            "stability inputs do not use one candidate/runtime identity: "
            f"candidate={candidate_shas}, runtime={runtime_shas}"
        )
    failures = [
        {
            "dataset": row["dataset"],
            "function": row["function"],
            "sample_std_seconds": row["sample_std_seconds"],
            "coefficient_of_variation": row["coefficient_of_variation"],
            "max_over_median": row["max_over_median"],
            "median_over_minimum": row["median_over_minimum"],
            "failure_reasons": json.loads(row["failure_reasons"]),
            "result_source": row["result_source"],
        }
        for row in rows
        if row["stability_status"] != "pass"
    ]
    output_csv = args.output_csv.resolve()
    write_csv(output_csv, rows)
    summary = {
        "status": "pass" if not failures else "fail",
        "metric": args.metric,
        "paper_estimator": PAPER_ESTIMATOR,
        "acceptance_estimator_independent": True,
        "variance_policy": DEFAULT_VARIANCE_POLICY,
        "sample_std_and_cv_role": (
            "reported_diagnostics_not_acceptance_gate"
        ),
        "failed_batch_policy": (
            "reject_entire_batch_do_not_select_across_batches"
        ),
        "batch_selection_policy": (
            FORMAL_BATCH_SELECTION_POLICY
            if not replacement_manifest["path"]
            and not batch_overrides_applied
            else "unique_batch_or_explicit_failed_batch_replacement"
        ),
        "max_over_median_limit": args.max_over_median_limit,
        "median_over_min_limit": args.median_over_min_limit,
        "gate_calibration": gate_calibration,
        "expected_samples_per_cell": args.expected_samples,
        "audited_cells": len(rows),
        "unique_workload_keys": len(set(keys)),
        "stable_cells": len(rows) - len(failures),
        "unstable_cells": len(failures),
        "failures": failures,
        "main_result_dirs": [
            str(path.resolve()) for path in args.main_result_dir
        ],
        "anchor_result_dirs": [
            str(path.resolve()) for path in args.anchor_result_dir
        ],
        "candidate_sha256": candidate_shas[0],
        "runtime_python_snapshot_sha256": runtime_shas[0],
        "runtime_package_is_symlink": False,
        "result_dir_identities": identities,
        "input_evidence": input_evidence,
        "input_evidence_sha256": canonical_sha256(input_evidence),
        "override_manifest_path": replacement_manifest["path"],
        "override_manifest_sha256": replacement_manifest["sha256"],
        "batch_overrides_applied": batch_overrides_applied,
        "batch_overrides_applied_sha256": canonical_sha256(
            batch_overrides_applied
        ),
        "audited_keys_sha256": canonical_sha256(sorted(keys)),
        "rows_csv_path": str(output_csv),
        "rows_csv_sha256": sha256(output_csv),
    }
    args.output_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    if failures and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
