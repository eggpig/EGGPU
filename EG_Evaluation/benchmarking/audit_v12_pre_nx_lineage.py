#!/usr/bin/env python3
"""Audit the immutable V10 -> Gunrock V11 -> CPU V12 ledger lineage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


KEY_FIELDS = ("dataset", "function", "baseline")
EXPECTED_LEDGER_ROWS = 1664
EXPECTED_GUNROCK_CELLS = 90
EXPECTED_CPU_CELLS = 16


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


def key(row: dict) -> tuple[str, str, str]:
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)


def indexed_ledger(
    path: Path, label: str, issues: list[str]
) -> tuple[list[str], list[dict], dict[tuple[str, str, str], dict]]:
    if not path.is_file():
        issues.append(f"{label}: ledger is absent: {path}")
        return [], [], {}
    fields, rows = read_csv(path)
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    duplicates = sorted(item for item, values in grouped.items() if len(values) != 1)
    if len(rows) != EXPECTED_LEDGER_ROWS:
        issues.append(
            f"{label}: expected {EXPECTED_LEDGER_ROWS} rows, observed {len(rows)}"
        )
    if duplicates:
        issues.append(f"{label}: duplicate dataset/function/baseline keys={duplicates}")
    if len(grouped) != len(rows):
        issues.append(
            f"{label}: unique key count={len(grouped)}, row count={len(rows)}"
        )
    return fields, rows, {
        item: values[0] for item, values in grouped.items() if len(values) == 1
    }


def canonical_non_target_digest(
    rows: dict[tuple[str, str, str], dict],
    selected_keys: set[tuple[str, str, str]],
    fields: list[str],
) -> str:
    payload = []
    for item in sorted(selected_keys):
        row = rows[item]
        payload.append(
            {
                "key": list(item),
                "fields": [str(row.get(field, "")) for field in fields],
            }
        )
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_transition(
    before_label: str,
    before_fields: list[str],
    before: dict[tuple[str, str, str], dict],
    after_label: str,
    after_fields: list[str],
    after: dict[tuple[str, str, str], dict],
    expected_targets: set[tuple[str, str, str]],
    issues: list[str],
) -> dict:
    before_keys = set(before)
    after_keys = set(after)
    if before_keys != after_keys:
        issues.append(
            f"{before_label}->{after_label}: key set changed; "
            f"missing={sorted(before_keys - after_keys)}, "
            f"extra={sorted(after_keys - before_keys)}"
        )
    common_keys = before_keys & after_keys
    fields = list(dict.fromkeys(before_fields + after_fields))
    changed = {
        item
        for item in common_keys
        if any(
            str(before[item].get(field, ""))
            != str(after[item].get(field, ""))
            for field in fields
        )
    }
    if changed != expected_targets:
        issues.append(
            f"{before_label}->{after_label}: changed-cell set mismatch; "
            f"missing={sorted(expected_targets - changed)}, "
            f"extra={sorted(changed - expected_targets)}"
        )
    non_targets = common_keys - expected_targets
    differing_non_targets = []
    for item in sorted(non_targets):
        if any(
            str(before[item].get(field, ""))
            != str(after[item].get(field, ""))
            for field in fields
        ):
            differing_non_targets.append(item)
    if differing_non_targets:
        issues.append(
            f"{before_label}->{after_label}: non-target fields changed for "
            f"{differing_non_targets}"
        )
    before_digest = canonical_non_target_digest(before, non_targets, fields)
    after_digest = canonical_non_target_digest(after, non_targets, fields)
    if before_digest != after_digest:
        issues.append(
            f"{before_label}->{after_label}: canonical non-target byte digest differs"
        )
    return {
        "before": before_label,
        "after": after_label,
        "row_key_set_preserved": before_keys == after_keys,
        "expected_target_count": len(expected_targets),
        "observed_changed_cell_count": len(changed),
        "changed_cell_set_exact": changed == expected_targets,
        "non_target_cell_count": len(non_targets),
        "non_target_field_differences": len(differing_non_targets),
        "canonical_field_order": fields,
        "before_non_target_canonical_sha256": before_digest,
        "after_non_target_canonical_sha256": after_digest,
        "canonical_byte_identity": before_digest == after_digest,
        "changed_cells": ["\t".join(item) for item in sorted(changed)],
    }


def resolve_recorded(value: object, owner: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (owner.parent / path).resolve()


def verify_path_sha(
    payload: dict,
    owner: Path,
    path_field: str,
    sha_field: str,
    label: str,
    issues: list[str],
) -> dict:
    path = resolve_recorded(payload.get(path_field, ""), owner)
    exists = path.is_file()
    observed_sha = sha256_file(path) if exists else ""
    expected_sha = str(payload.get(sha_field, ""))
    matches = exists and observed_sha == expected_sha
    if not matches:
        issues.append(
            f"{label}: {path_field}/{sha_field} mismatch; path={path}, "
            f"exists={exists}, expected={expected_sha}, observed={observed_sha}"
        )
    return {
        "label": label,
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected_sha,
        "observed_sha256": observed_sha,
        "matches": matches,
    }


def positive(value: object) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed > 0.0


def expected_gunrock_targets(
    overlay_manifest_path: Path,
    overlay_manifest: dict,
    issues: list[str],
) -> set[tuple[str, str, str]]:
    overlay_path = resolve_recorded(
        overlay_manifest.get("overlay_csv", ""), overlay_manifest_path
    )
    if not overlay_path.is_file():
        issues.append(f"Gunrock overlay CSV is absent: {overlay_path}")
        return set()
    _, rows = read_csv(overlay_path)
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        item = key(row)
        grouped.setdefault(item, set()).add(row.get("metric", ""))
    bad = sorted(item for item, metrics in grouped.items() if metrics != {"build", "kernel", "e2e"})
    if bad:
        issues.append(f"Gunrock overlay cells without exact triplets={bad}")
    targets = set(grouped)
    if len(targets) != EXPECTED_GUNROCK_CELLS:
        issues.append(
            f"Gunrock overlay target cells={len(targets)}, "
            f"expected={EXPECTED_GUNROCK_CELLS}"
        )
    if any(item[2] != "Gunrock" for item in targets):
        issues.append("Gunrock overlay target set contains another baseline")
    return targets


def expected_cpu_targets(cpu_manifest: dict, issues: list[str]) -> set[tuple[str, str, str]]:
    targets = set()
    for value in cpu_manifest.get("target_cells", []):
        parts = tuple(str(value).split("\t"))
        if len(parts) != 3:
            issues.append(f"CPU manifest target key is malformed: {value!r}")
            continue
        targets.add(parts)
    if len(targets) != EXPECTED_CPU_CELLS:
        issues.append(
            f"CPU overlay target cells={len(targets)}, expected={EXPECTED_CPU_CELLS}"
        )
    allowed_baselines = {"networkx", "easygraph-cpu", "easygraph-cpp", "igraph"}
    if any(
        dataset not in {
            "ER-100k",
            "com-youtube",
            "soc-Slashdot0811",
            "web-NotreDame",
        }
        or function != "Closeness"
        or baseline not in allowed_baselines
        for dataset, function, baseline in targets
    ):
        issues.append("CPU manifest target set is outside the declared 16 cells")
    return targets


def atomic_text(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def markdown(report: dict) -> str:
    lines = [
        "# V12 pre-nx lineage audit",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- V10/V11/V12 rows: {report['ledger_rows']['V10']}/"
        f"{report['ledger_rows']['V11']}/{report['ledger_rows']['V12']}",
        f"- Unique keys: {report['unique_key_counts']['V10']}/"
        f"{report['unique_key_counts']['V11']}/"
        f"{report['unique_key_counts']['V12']}",
        "- Key: `dataset/function/baseline`",
        "",
        "## Lineage transitions",
        "",
    ]
    for transition in report["transitions"]:
        lines.extend(
            [
                f"### {transition['before']} → {transition['after']}",
                "",
                f"- Expected/observed changed cells: "
                f"{transition['expected_target_count']}/"
                f"{transition['observed_changed_cell_count']}",
                f"- Changed-cell set exact: "
                f"{transition['changed_cell_set_exact']}",
                f"- Non-target cells: {transition['non_target_cell_count']}",
                f"- Non-target field differences: "
                f"{transition['non_target_field_differences']}",
                f"- Canonical byte identity: "
                f"{transition['canonical_byte_identity']}",
                f"- Before digest: "
                f"`{transition['before_non_target_canonical_sha256']}`",
                f"- After digest: "
                f"`{transition['after_non_target_canonical_sha256']}`",
                "",
            ]
        )
    lines.extend(["## Receipt and manifest references", ""])
    for item in report["sha_references"]:
        lines.append(
            f"- {'PASS' if item['matches'] else 'FAIL'} `{item['path']}` — "
            f"`{item['observed_sha256']}`"
        )
    lines.extend(["", "## Audit issues", ""])
    if report["issues"]:
        lines.extend(f"- {issue}" for issue in report["issues"])
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10-ledger", required=True, type=Path)
    parser.add_argument("--v11-ledger", required=True, type=Path)
    parser.add_argument("--v11-receipt", required=True, type=Path)
    parser.add_argument("--v12-ledger", required=True, type=Path)
    parser.add_argument("--cpu-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    paths = {
        "V10": args.v10_ledger.resolve(),
        "V11": args.v11_ledger.resolve(),
        "V12": args.v12_ledger.resolve(),
    }
    receipt_path = args.v11_receipt.resolve()
    cpu_manifest_path = args.cpu_manifest.resolve()
    issues: list[str] = []
    ledgers = {}
    for label, path in paths.items():
        fields, rows, indexed = indexed_ledger(path, label, issues)
        ledgers[label] = {"fields": fields, "rows": rows, "indexed": indexed}
    key_sets = {label: set(value["indexed"]) for label, value in ledgers.items()}
    if not (key_sets["V10"] == key_sets["V11"] == key_sets["V12"]):
        issues.append("V10/V11/V12 dataset/function/baseline key sets differ")

    if not receipt_path.is_file():
        issues.append(f"V11 receipt is absent: {receipt_path}")
        receipt = {}
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    overlay_manifest_path = resolve_recorded(
        receipt.get("overlay_manifest", ""), receipt_path
    )
    if not overlay_manifest_path.is_file():
        issues.append(f"Gunrock overlay manifest is absent: {overlay_manifest_path}")
        overlay_manifest = {}
    else:
        overlay_manifest = json.loads(
            overlay_manifest_path.read_text(encoding="utf-8")
        )
    if not cpu_manifest_path.is_file():
        issues.append(f"CPU overlay manifest is absent: {cpu_manifest_path}")
        cpu_manifest = {}
    else:
        cpu_manifest = json.loads(cpu_manifest_path.read_text(encoding="utf-8"))

    gunrock_targets = expected_gunrock_targets(
        overlay_manifest_path, overlay_manifest, issues
    )
    cpu_targets = expected_cpu_targets(cpu_manifest, issues)
    transitions = [
        compare_transition(
            "V10",
            ledgers["V10"]["fields"],
            ledgers["V10"]["indexed"],
            "V11",
            ledgers["V11"]["fields"],
            ledgers["V11"]["indexed"],
            gunrock_targets,
            issues,
        ),
        compare_transition(
            "V11",
            ledgers["V11"]["fields"],
            ledgers["V11"]["indexed"],
            "V12",
            ledgers["V12"]["fields"],
            ledgers["V12"]["indexed"],
            cpu_targets,
            issues,
        ),
    ]

    # Check the timing contracts in the published target rows.
    for item in sorted(gunrock_targets):
        row = ledgers["V11"]["indexed"].get(item, {})
        if not (
            row.get("execution_status") == "ok"
            and row.get("validation_status") == "pass"
            and int(float(row.get("sample_count", "0"))) == 5
            and all(
                positive(row.get(field))
                for field in (
                    "build_paper_seconds",
                    "kernel_paper_seconds",
                    "e2e_paper_seconds",
                )
            )
            and row.get("timing_external_cli_wall_used") == "false"
        ):
            issues.append(f"V11 Gunrock target contract failed: {item}")
    for item in sorted(cpu_targets):
        row = ledgers["V12"]["indexed"].get(item, {})
        try:
            processing_equals_e2e = math.isclose(
                float(row.get("kernel_paper_seconds", "")),
                float(row.get("e2e_paper_seconds", "")),
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
        except (TypeError, ValueError):
            processing_equals_e2e = False
        if not (
            row.get("execution_status") == "ok"
            and row.get("validation_status") == "pass"
            and int(float(row.get("sample_count", "0"))) == 5
            and all(
                positive(row.get(field))
                for field in (
                    "build_paper_seconds",
                    "kernel_paper_seconds",
                    "e2e_paper_seconds",
                )
            )
            and row.get("build_estimator") == "arithmetic_mean"
            and processing_equals_e2e
            and row.get("timing_external_cli_wall_used") == "false"
        ):
            issues.append(f"V12 CPU target contract failed: {item}")

    sha_references = []
    if receipt:
        if receipt.get("status") != "pass":
            issues.append("V11 receipt status is not pass")
        sha_references.extend(
            [
                verify_path_sha(
                    receipt,
                    receipt_path,
                    "base_ledger",
                    "base_ledger_sha256",
                    "V11 receipt -> V10",
                    issues,
                ),
                verify_path_sha(
                    receipt,
                    receipt_path,
                    "output",
                    "output_sha256",
                    "V11 receipt -> V11",
                    issues,
                ),
                verify_path_sha(
                    receipt,
                    receipt_path,
                    "overlay_manifest",
                    "overlay_manifest_sha256",
                    "V11 receipt -> Gunrock manifest",
                    issues,
                ),
                verify_path_sha(
                    receipt,
                    receipt_path,
                    "overlay_csv",
                    "overlay_csv_sha256",
                    "V11 receipt -> Gunrock overlay",
                    issues,
                ),
            ]
        )
        expected_receipt_counts = {
            "base_rows": EXPECTED_LEDGER_ROWS,
            "output_rows": EXPECTED_LEDGER_ROWS,
            "replaced_success_cells": EXPECTED_GUNROCK_CELLS,
            "overlay_metric_rows": EXPECTED_GUNROCK_CELLS * 3,
            "raw_phase_samples_verified": EXPECTED_GUNROCK_CELLS * 3 * 5,
        }
        for field, expected in expected_receipt_counts.items():
            if int(receipt.get(field, -1)) != expected:
                issues.append(
                    f"V11 receipt {field}={receipt.get(field)!r}, expected={expected}"
                )
    if overlay_manifest:
        if overlay_manifest.get("gate_status") != "pass":
            issues.append("Gunrock overlay manifest gate_status is not pass")
        for path_field, sha_field, label in (
            ("gate_json", "gate_json_sha256", "Gunrock strict gate"),
            (
                "formal_run_manifest",
                "formal_run_manifest_sha256",
                "Gunrock formal run manifest",
            ),
            (
                "gunrock_artifact_manifest",
                "gunrock_artifact_manifest_sha256",
                "Gunrock artifact manifest",
            ),
            ("overlay_csv", "overlay_csv_sha256", "Gunrock overlay CSV"),
            (
                "raw_samples_csv",
                "raw_samples_csv_sha256",
                "Gunrock raw sample CSV",
            ),
            (
                "wide_90_cell_csv",
                "wide_90_cell_csv_sha256",
                "Gunrock wide 90-cell CSV",
            ),
        ):
            sha_references.append(
                verify_path_sha(
                    overlay_manifest,
                    overlay_manifest_path,
                    path_field,
                    sha_field,
                    label,
                    issues,
                )
            )
        gate_path = resolve_recorded(
            overlay_manifest.get("gate_json", ""), overlay_manifest_path
        )
        if gate_path.is_file():
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            if not (
                gate.get("status") == "pass"
                and int(gate.get("packaged_cells", -1)) == 90
                and int(gate.get("packaged_process_samples", -1)) == 450
                and int(gate.get("packaged_metric_samples", -1)) == 1350
                and gate.get("issues") == []
            ):
                issues.append("Gunrock strict gate counts/status are inconsistent")
    if cpu_manifest:
        if cpu_manifest.get("status") != "pass":
            issues.append("CPU overlay manifest status is not pass")
        for path_field, sha_field, label in (
            ("base_ledger", "base_ledger_sha256", "CPU manifest -> V11"),
            ("output_ledger", "output_ledger_sha256", "CPU manifest -> V12"),
            ("overlay_csv", "overlay_csv_sha256", "CPU overlay CSV"),
            ("raw_samples_csv", "raw_samples_csv_sha256", "CPU raw sample CSV"),
        ):
            sha_references.append(
                verify_path_sha(
                    cpu_manifest,
                    cpu_manifest_path,
                    path_field,
                    sha_field,
                    label,
                    issues,
                )
            )
        for recorded_path, expected_sha in cpu_manifest.get(
            "evidence_files", {}
        ).items():
            evidence_path = Path(recorded_path).resolve()
            exists = evidence_path.is_file()
            observed_sha = sha256_file(evidence_path) if exists else ""
            matches = exists and observed_sha == expected_sha
            if not matches:
                issues.append(
                    f"CPU evidence SHA mismatch: {evidence_path}; "
                    f"expected={expected_sha}, observed={observed_sha}"
                )
            sha_references.append(
                {
                    "label": "CPU source evidence",
                    "path": str(evidence_path),
                    "exists": exists,
                    "expected_sha256": expected_sha,
                    "observed_sha256": observed_sha,
                    "matches": matches,
                }
            )
        expected_cpu_counts = {
            "base_rows": EXPECTED_LEDGER_ROWS,
            "output_rows": EXPECTED_LEDGER_ROWS,
            "target_cell_count": EXPECTED_CPU_CELLS,
            "raw_sample_count": EXPECTED_CPU_CELLS * 5,
            "non_target_rows_unchanged": EXPECTED_LEDGER_ROWS - EXPECTED_CPU_CELLS,
        }
        for field, expected in expected_cpu_counts.items():
            if int(cpu_manifest.get(field, -1)) != expected:
                issues.append(
                    f"CPU manifest {field}={cpu_manifest.get(field)!r}, "
                    f"expected={expected}"
                )

    report = {
        "schema_version": "v12_pre_nx_lineage_audit_v1",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "pass" if not issues else "fail",
        "ledger_paths": {label: str(path) for label, path in paths.items()},
        "ledger_sha256": {
            label: sha256_file(path) if path.is_file() else ""
            for label, path in paths.items()
        },
        "ledger_rows": {
            label: len(value["rows"]) for label, value in ledgers.items()
        },
        "unique_key_counts": {
            label: len(value["indexed"]) for label, value in ledgers.items()
        },
        "key_fields": list(KEY_FIELDS),
        "key_sets_identical": (
            key_sets["V10"] == key_sets["V11"] == key_sets["V12"]
        ),
        "gunrock_target_count": len(gunrock_targets),
        "cpu_target_count": len(cpu_targets),
        "transitions": transitions,
        "v11_receipt": str(receipt_path),
        "v11_receipt_sha256": (
            sha256_file(receipt_path) if receipt_path.is_file() else ""
        ),
        "gunrock_overlay_manifest": str(overlay_manifest_path),
        "gunrock_overlay_manifest_sha256": (
            sha256_file(overlay_manifest_path)
            if overlay_manifest_path.is_file()
            else ""
        ),
        "cpu_overlay_manifest": str(cpu_manifest_path),
        "cpu_overlay_manifest_sha256": (
            sha256_file(cpu_manifest_path)
            if cpu_manifest_path.is_file()
            else ""
        ),
        "sha_reference_count": len(sha_references),
        "sha_references_all_match": all(
            item["matches"] for item in sha_references
        ),
        "sha_references": sha_references,
        "issues": issues,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "V12_PRE_NX_LINEAGE_AUDIT_20260730.json"
    md_path = args.out_dir / "V12_PRE_NX_LINEAGE_AUDIT_20260730.md"
    atomic_text(
        json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    atomic_text(md_path, markdown(report))
    print(json.dumps({
        "status": report["status"],
        "issues": len(issues),
        "ledger_rows": report["ledger_rows"],
        "unique_key_counts": report["unique_key_counts"],
        "gunrock_target_count": len(gunrock_targets),
        "cpu_target_count": len(cpu_targets),
        "sha_reference_count": len(sha_references),
        "sha_references_all_match": report["sha_references_all_match"],
    }, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
