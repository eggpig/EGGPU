#!/usr/bin/env python3
"""Atomically overlay gated strict Gunrock rows into a new result ledger.

The base ledger is immutable: input and output paths must differ.  The script
verifies the overlay manifest, raw five-sample evidence, timing protocol,
estimator, validation status, included/excluded batch policy, and file hashes
before replacing matching Gunrock keys in a newly written CSV.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from gunrock_timing_protocol import PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parents[1]
KEY_FIELDS = ("dataset", "function", "baseline", "metric")
CELL_KEY_FIELDS = ("dataset", "function", "baseline")
METRICS = {"build", "kernel", "e2e"}


class OverlayValidationError(ValueError):
    """Raised before any output mutation when the overlay is not admissible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def resolve_recorded_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root_candidate = (ROOT / path).resolve()
    manifest_candidate = (manifest_path.parent / path).resolve()
    if root_candidate.exists() or not manifest_candidate.exists():
        return root_candidate
    return manifest_candidate


def false_value(value) -> bool:
    return value in (None, "", False, 0, "0", "false", "False", "FALSE")


def positive_float(value, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OverlayValidationError(f"{context}: invalid number {value!r}") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise OverlayValidationError(
            f"{context}: expected a finite positive number, got {result!r}"
        )
    return result


def key(row: dict) -> tuple[str, str, str, str]:
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)


def cell_key(row: dict) -> tuple[str, str, str]:
    return tuple(str(row.get(field, "")) for field in CELL_KEY_FIELDS)


def key_digest(keys) -> str:
    lines = ["\t".join(item) for item in sorted(keys)]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def validate_sources(
    row: dict,
    included: tuple[Path, ...],
    excluded: tuple[Path, ...],
    manifest_path: Path,
) -> None:
    source = resolve_recorded_path(row.get("source_batch", ""), manifest_path)
    if not any(source == root or source.is_relative_to(root) for root in included):
        raise OverlayValidationError(
            f"{key(row)}: source batch is not included by manifest: {source}"
        )
    if any(source == root or source.is_relative_to(root) for root in excluded):
        raise OverlayValidationError(
            f"{key(row)}: source batch is explicitly excluded: {source}"
        )


def validate_overlay(
    overlay_path: Path,
    manifest_path: Path,
    manifest: dict,
) -> tuple[list[str], list[dict], list[dict]]:
    if manifest.get("gate_status") != "pass":
        raise OverlayValidationError("overlay manifest gate_status is not 'pass'")
    if manifest.get("timing_protocol_version") != PROTOCOL_VERSION:
        raise OverlayValidationError(
            f"unexpected timing protocol: {manifest.get('timing_protocol_version')!r}"
        )
    if manifest.get("external_cli_wall_allowed") is not False:
        raise OverlayValidationError(
            "overlay manifest does not forbid external CLI wall substitution"
        )
    evidence_payloads = {}
    for field, sha_field in (
        ("gate_json", "gate_json_sha256"),
        ("formal_run_manifest", "formal_run_manifest_sha256"),
        ("gunrock_artifact_manifest", "gunrock_artifact_manifest_sha256"),
    ):
        evidence_path = resolve_recorded_path(
            manifest.get(field, ""), manifest_path
        )
        if not evidence_path.is_file():
            raise OverlayValidationError(
                f"required manifest evidence is absent: {field}={evidence_path}"
            )
        if sha256_file(evidence_path) != manifest.get(sha_field):
            raise OverlayValidationError(
                f"{field} SHA-256 does not match the overlay manifest"
            )
        evidence_payloads[field] = json.loads(
            evidence_path.read_text(encoding="utf-8")
        )
    gate = evidence_payloads["gate_json"]
    if (
        gate.get("status") != "pass"
        or int(gate.get("packaged_cells", -1))
        != int(manifest.get("success_cell_count", -2))
        or int(gate.get("packaged_metric_samples", -1))
        != int(manifest.get("phase_sample_count", -2))
    ):
        raise OverlayValidationError("pinned gate JSON is not a complete pass")
    expected_overlay_sha = manifest.get("overlay_csv_sha256")
    ratio_limit = float(manifest.get("catastrophic_max_min_ratio_limit", 0.0))
    absolute_span_limit = float(
        manifest.get("catastrophic_absolute_span_seconds", 0.0)
    )
    if ratio_limit <= 1.0:
        raise OverlayValidationError(
            "manifest catastrophic max/min ratio limit is absent or invalid"
        )
    if absolute_span_limit <= 0.0:
        raise OverlayValidationError(
            "manifest catastrophic absolute-span limit is absent or invalid"
        )
    if sha256_file(overlay_path) != expected_overlay_sha:
        raise OverlayValidationError("overlay CSV SHA-256 does not match manifest")
    recorded_overlay = resolve_recorded_path(
        manifest.get("overlay_csv", ""), manifest_path
    )
    if recorded_overlay != overlay_path.resolve():
        raise OverlayValidationError(
            f"provided overlay differs from manifest path: {overlay_path} != "
            f"{recorded_overlay}"
        )
    raw_path = resolve_recorded_path(
        manifest.get("raw_samples_csv", ""), manifest_path
    )
    if not raw_path.is_file():
        raise OverlayValidationError(f"raw five-sample evidence is absent: {raw_path}")
    if sha256_file(raw_path) != manifest.get("raw_samples_csv_sha256"):
        raise OverlayValidationError("raw sample CSV SHA-256 does not match manifest")

    included = tuple(
        resolve_recorded_path(value, manifest_path)
        for value in manifest.get("included_batches", [])
    )
    excluded_entries = manifest.get("excluded_batches", [])
    excluded = tuple(
        resolve_recorded_path(item.get("path", ""), manifest_path)
        for item in excluded_entries
    )
    if not included:
        raise OverlayValidationError("overlay manifest has no included batches")
    if not excluded_entries:
        raise OverlayValidationError("overlay manifest has no excluded-batch policy")
    for item in excluded_entries:
        if item.get("status") != "excluded_entire_batch" or item.get(
            "must_not_merge"
        ) is not True:
            raise OverlayValidationError(
                f"excluded batch is not marked must-not-merge: {item!r}"
            )

    fields, rows = read_csv(overlay_path)
    expected_rows = int(manifest.get("replacement_key_count", -1))
    expected_cells = int(manifest.get("success_cell_count", -1))
    if len(rows) != expected_rows:
        raise OverlayValidationError(
            f"overlay row count={len(rows)}, expected={expected_rows}"
        )
    keys = [key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise OverlayValidationError("overlay contains duplicate replacement keys")
    if key_digest(keys) != manifest.get("replacement_key_sha256"):
        raise OverlayValidationError("replacement-key digest does not match manifest")
    aggregate_rows_by_key = {key(row): row for row in rows}
    grouped = defaultdict(list)
    aggregate_samples = {}
    for row in rows:
        row_key = key(row)
        cell = row_key[:2]
        grouped[cell].append(row)
        if row.get("baseline") != "Gunrock":
            raise OverlayValidationError(f"{row_key}: baseline must be Gunrock")
        if row.get("metric") not in METRICS:
            raise OverlayValidationError(f"{row_key}: unexpected metric")
        if row.get("status") != "ok" or row.get("validation") != "pass":
            raise OverlayValidationError(
                f"{row_key}: status/validation must both pass"
            )
        if row.get("estimator_kind") != "minimum_of_five_fresh_processes":
            raise OverlayValidationError(f"{row_key}: invalid display estimator")
        if row.get("aggregation") != "minimum_for_paper_display":
            raise OverlayValidationError(f"{row_key}: invalid aggregation policy")
        for field in ("sample_count", "n_total", "n_valid"):
            if str(row.get(field, "")) != "5":
                raise OverlayValidationError(
                    f"{row_key}: {field}={row.get(field)!r}, expected '5'"
                )
        if row.get("timing_protocol_version") != PROTOCOL_VERSION:
            raise OverlayValidationError(f"{row_key}: timing protocol mismatch")
        if not false_value(row.get("external_cli_wall_used")):
            raise OverlayValidationError(f"{row_key}: external CLI wall was used")
        app = row.get("executable_app", "")
        executable_sha = row.get("executable_sha256", "")
        pinned_sha = (
            evidence_payloads["gunrock_artifact_manifest"]
            .get("executables", {})
            .get(app, {})
            .get("sha256")
        )
        if not app or not executable_sha or executable_sha != pinned_sha:
            raise OverlayValidationError(
                f"{row_key}: executable provenance is absent or not pinned"
            )
        validate_sources(row, included, excluded, manifest_path)
        try:
            samples = [float(value) for value in json.loads(row["sample_seconds_json"])]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OverlayValidationError(
                f"{row_key}: invalid sample_seconds_json"
            ) from exc
        if len(samples) != 5 or any(
            not math.isfinite(value) or value <= 0.0 for value in samples
        ):
            raise OverlayValidationError(
                f"{row_key}: expected five finite positive raw samples"
            )
        if (
            max(samples) / min(samples) > ratio_limit
            and max(samples) - min(samples) > absolute_span_limit
        ):
            raise OverlayValidationError(
                f"{row_key}: catastrophic max/min timing ratio exceeds "
                f"{ratio_limit:g}"
            )
        observed_min = positive_float(row.get("seconds"), f"{row_key} seconds")
        reported_min = positive_float(
            row.get("min_seconds"), f"{row_key} min_seconds"
        )
        reported_mean = positive_float(
            row.get("mean_seconds"), f"{row_key} mean_seconds"
        )
        reported_sd = float(row.get("sample_sd_seconds", "nan"))
        expected_min = min(samples)
        expected_mean = statistics.mean(samples)
        expected_sd = statistics.stdev(samples)
        for name, observed, expected in (
            ("seconds", observed_min, expected_min),
            ("min_seconds", reported_min, expected_min),
            ("mean_seconds", reported_mean, expected_mean),
            ("sample_sd_seconds", reported_sd, expected_sd),
        ):
            if not math.isfinite(observed) or not math.isclose(
                observed, expected, rel_tol=1.0e-9, abs_tol=1.0e-12
            ):
                raise OverlayValidationError(
                    f"{row_key}: {name}={observed!r}, recomputed={expected!r}"
                )
        aggregate_samples[row_key] = samples
    if len(grouped) != expected_cells:
        raise OverlayValidationError(
            f"overlay success cells={len(grouped)}, expected={expected_cells}"
        )
    for cell, cell_rows in grouped.items():
        if {row["metric"] for row in cell_rows} != METRICS or len(cell_rows) != 3:
            raise OverlayValidationError(
                f"{cell}: expected exactly build/kernel/e2e rows"
            )

    _, raw_rows = read_csv(raw_path)
    expected_raw = expected_rows * 5
    if len(raw_rows) != expected_raw:
        raise OverlayValidationError(
            f"raw phase samples={len(raw_rows)}, expected={expected_raw}"
        )
    raw_grouped = defaultdict(list)
    for row in raw_rows:
        row_key = key(row)
        if row_key not in aggregate_samples:
            raise OverlayValidationError(
                f"raw sample references a non-overlay key: {row_key}"
            )
        if row.get("status") != "ok" or row.get("validation") != "pass":
            raise OverlayValidationError(f"{row_key}: raw status/validation failed")
        if str(row.get("sample_count", "")) != "5":
            raise OverlayValidationError(f"{row_key}: raw sample_count is not five")
        if row.get("timing_protocol_version") != PROTOCOL_VERSION:
            raise OverlayValidationError(f"{row_key}: raw protocol mismatch")
        if not false_value(row.get("external_cli_wall_used")):
            raise OverlayValidationError(f"{row_key}: raw external wall was used")
        if (
            row.get("executable_app", "") == ""
            or row.get("executable_sha256", "")
            != aggregate_rows_by_key[row_key].get("executable_sha256", "")
        ):
            raise OverlayValidationError(
                f"{row_key}: raw executable provenance differs from aggregate"
            )
        validate_sources(row, included, excluded, manifest_path)
        index = int(row.get("sample_index", ""))
        seconds = positive_float(row.get("seconds"), f"{row_key} raw sample")
        raw_grouped[row_key].append((index, seconds))
    for row_key, samples in aggregate_samples.items():
        observed = sorted(raw_grouped.get(row_key, []))
        if [index for index, _ in observed] != [1, 2, 3, 4, 5]:
            raise OverlayValidationError(
                f"{row_key}: raw sample indices are not exactly 1..5"
            )
        values = [value for _, value in observed]
        for lhs, rhs in zip(values, samples):
            if not math.isclose(lhs, rhs, rel_tol=1.0e-9, abs_tol=1.0e-12):
                raise OverlayValidationError(
                    f"{row_key}: aggregate/raw sample arrays differ"
                )
    return fields, rows, raw_rows


def atomic_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def final13_overlay_rows(
    base_fields: list[str],
    base_rows: list[dict],
    overlay_rows: list[dict],
    overlay_path: Path,
    manifest_path: Path,
) -> tuple[list[dict], int]:
    required_fields = {
        "dataset",
        "function",
        "baseline",
        "execution_status",
        "validation_status",
        "sample_count",
        "build_paper_seconds",
        "kernel_paper_seconds",
        "e2e_paper_seconds",
    }
    missing_fields = required_fields - set(base_fields)
    if missing_fields:
        raise OverlayValidationError(
            "final13 ledger is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    grouped: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for row in overlay_rows:
        metric = row["metric"]
        grouped[cell_key(row)][metric] = row
    target_cells = set(grouped)
    if len(target_cells) * 3 != len(overlay_rows):
        raise OverlayValidationError(
            "final13 conversion requires exactly three metric rows per cell"
        )

    target_indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(base_rows):
        row_cell = cell_key(row)
        if row_cell in target_cells:
            target_indices[row_cell].append(index)
    missing_cells = sorted(
        row_cell for row_cell in target_cells if not target_indices[row_cell]
    )
    duplicate_cells = sorted(
        row_cell
        for row_cell, indices in target_indices.items()
        if len(indices) != 1
    )
    if missing_cells or duplicate_cells:
        raise OverlayValidationError(
            "final13 target-cell cardinality mismatch: "
            f"missing={missing_cells}, duplicate={duplicate_cells}"
        )

    result = [dict(row) for row in base_rows]
    for row_cell, metric_rows in grouped.items():
        if set(metric_rows) != METRICS:
            raise OverlayValidationError(
                f"{row_cell}: final13 conversion lacks build/kernel/e2e"
            )
        index = target_indices[row_cell][0]
        staged = dict(result[index])
        staged["execution_status"] = "ok"
        staged["failure_kind"] = ""
        staged["validation_status"] = "pass"
        staged["sample_count"] = "5"
        staged["result_source"] = str(overlay_path)
        staged["timing_result_source"] = str(overlay_path)
        staged["reason"] = (
            "strict native Gunrock three-phase timing; minimum of five fresh "
            "processes; full-result validation passed; external CLI wall "
            "substitution forbidden"
        )
        for metric, prefix in (
            ("build", "build"),
            ("kernel", "kernel"),
            ("e2e", "e2e"),
        ):
            aggregate = metric_rows[metric]
            samples = [
                float(value)
                for value in json.loads(aggregate["sample_seconds_json"])
            ]
            sample_min = min(samples)
            sample_median = statistics.median(samples)
            sample_max = max(samples)
            sample_mean = statistics.mean(samples)
            sample_sd = statistics.stdev(samples)
            staged[f"{prefix}_paper_seconds"] = f"{sample_min:.12g}"
            staged[f"{prefix}_raw_mean_seconds"] = f"{sample_mean:.12g}"
            staged[f"{prefix}_std_seconds"] = f"{sample_sd:.12g}"
            staged[f"{prefix}_estimator"] = (
                "minimum_of_five_fresh_processes"
            )
            staged[f"{prefix}_raw_min_seconds"] = f"{sample_min:.12g}"
            staged[f"{prefix}_raw_median_seconds"] = (
                f"{sample_median:.12g}"
            )
            staged[f"{prefix}_raw_max_seconds"] = f"{sample_max:.12g}"
            staged[f"{prefix}_coefficient_of_variation"] = (
                f"{sample_sd / sample_mean:.12g}"
            )
            staged[f"{prefix}_max_over_median"] = (
                f"{sample_max / sample_median:.12g}"
            )
            staged[f"{prefix}_median_over_minimum"] = (
                f"{sample_median / sample_min:.12g}"
            )
            staged[f"{prefix}_variance_policy"] = (
                "sample_standard_deviation_n_minus_1"
            )
            staged[f"{prefix}_stability_status"] = "pass"
        staged["timing_overlay_manifest"] = str(manifest_path)
        staged["timing_protocol_version"] = PROTOCOL_VERSION
        staged["timing_external_cli_wall_used"] = "false"
        result[index] = staged
    return result, len(target_cells)


def driver(args) -> int:
    base = args.base_ledger.resolve()
    overlay = args.overlay_csv.resolve()
    manifest_path = args.overlay_manifest.resolve()
    output = args.output.resolve()
    receipt_path = (
        args.receipt.resolve()
        if args.receipt is not None
        else output.with_name(output.name + ".gunrock_overlay_receipt.json")
    )
    if base == output:
        raise OverlayValidationError(
            "in-place ledger mutation is forbidden; choose a new --output path"
        )
    if output in {overlay, manifest_path}:
        raise OverlayValidationError("output would overwrite overlay evidence")
    if receipt_path in {base, overlay, manifest_path, output}:
        raise OverlayValidationError("receipt path collides with an evidence/data file")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overlay_fields, overlay_rows, raw_rows = validate_overlay(
        overlay, manifest_path, manifest
    )
    base_fields, base_rows = read_csv(base)
    requested_format = getattr(args, "ledger_format", "auto")
    if requested_format == "auto":
        ledger_format = "long" if "metric" in base_fields else "final13"
    else:
        ledger_format = requested_format
    replacement_keys = {key(row) for row in overlay_rows}
    if ledger_format == "long":
        retained = [row for row in base_rows if key(row) not in replacement_keys]
        removed = len(base_rows) - len(retained)
        output_fields = base_fields + [
            field for field in overlay_fields if field not in set(base_fields)
        ]
        merged = retained + overlay_rows
        inserted = len(overlay_rows)
        replaced_cells = len({cell_key(row) for row in overlay_rows})
    elif ledger_format == "final13":
        merged, replaced_cells = final13_overlay_rows(
            base_fields,
            base_rows,
            overlay_rows,
            overlay,
            manifest_path,
        )
        preferred_new_fields = [
            field
            for field in (
                "timing_overlay_manifest",
                "timing_protocol_version",
                "timing_external_cli_wall_used",
            )
            if field not in set(base_fields)
        ]
        known_fields = set(base_fields) | set(preferred_new_fields)
        extra_fields = sorted(
            {field for row in merged for field in row} - known_fields
        )
        output_fields = base_fields + preferred_new_fields + extra_fields
        removed = replaced_cells
        inserted = replaced_cells
    else:
        raise OverlayValidationError(
            f"unsupported ledger format: {ledger_format!r}"
        )
    atomic_csv(output, output_fields, merged)
    receipt = {
        "schema_version": "gunrock_strict_overlay_receipt_v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "base_ledger": str(base),
        "base_ledger_sha256": sha256_file(base),
        "overlay_manifest": str(manifest_path),
        "overlay_manifest_sha256": sha256_file(manifest_path),
        "overlay_csv": str(overlay),
        "overlay_csv_sha256": sha256_file(overlay),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "base_rows": len(base_rows),
        "ledger_format": ledger_format,
        "removed_matching_rows": removed,
        "inserted_overlay_rows": inserted,
        "replaced_success_cells": replaced_cells,
        "overlay_metric_rows": len(overlay_rows),
        "output_rows": len(merged),
        "replacement_key_count": len(replacement_keys),
        "raw_phase_samples_verified": len(raw_rows),
        "in_place_mutation": False,
        "status": "pass",
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ledger", type=Path, required=True)
    parser.add_argument("--overlay-csv", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--ledger-format",
        choices=("auto", "long", "final13"),
        default="auto",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(driver(parse_args()))
