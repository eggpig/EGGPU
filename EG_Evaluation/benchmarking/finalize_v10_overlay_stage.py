#!/usr/bin/env python3
"""Relocate hash-bound V10 overlay metadata before atomic publication.

The stability and overlay writers record absolute output paths.  A release is
first built in a same-filesystem staging directory, so those paths must be
rewritten to the future immutable release directory before the directory is
atomically renamed.  This helper updates only output-local paths, recomputes
the affected stability JSON hash, and revalidates every referenced file hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def require_local_path(
    observed: object,
    expected: Path,
    field: str,
) -> None:
    try:
        actual = Path(str(observed)).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{field} is not a usable path") from exc
    if actual != expected.resolve():
        raise ValueError(
            f"{field}={actual} does not match staged file {expected.resolve()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--final-dir", required=True, type=Path)
    parser.add_argument(
        "--overlay-audit-name",
        default="EGGPU_V10_OVERLAY_AUDIT_20260729.json",
    )
    parser.add_argument(
        "--stability-json-name",
        default="eggpu_timing_stability.json",
    )
    parser.add_argument(
        "--stability-csv-name",
        default="eggpu_timing_stability.csv",
    )
    parser.add_argument(
        "--ledger-name",
        default="final_13_cell_outcome_ledger.csv",
    )
    args = parser.parse_args()

    stage = args.stage_dir.resolve(strict=True)
    final = args.final_dir.resolve()
    if final.exists():
        raise FileExistsError(f"final release directory already exists: {final}")
    if stage.parent != final.parent:
        raise ValueError(
            "stage and final directories must share a parent for atomic rename"
        )

    stability_json = stage / args.stability_json_name
    stability_csv = stage / args.stability_csv_name
    overlay_json = stage / args.overlay_audit_name
    overlay_cells = overlay_json.with_suffix(".cells.csv")
    overlay_memory_cells = overlay_json.with_suffix(".memory_cells.csv")
    ledger = stage / args.ledger_name
    for path in (
        stability_json,
        stability_csv,
        overlay_json,
        overlay_cells,
        overlay_memory_cells,
        ledger,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"staged release file is missing: {path}")

    stability = json.loads(stability_json.read_text(encoding="utf-8"))
    original_stability_sha = sha256(stability_json)
    require_local_path(
        stability.get("rows_csv_path"),
        stability_csv,
        "stability.rows_csv_path",
    )
    if stability.get("rows_csv_sha256") != sha256(stability_csv):
        raise ValueError("stability CSV hash differs before relocation")
    stability["rows_csv_path"] = str(final / stability_csv.name)
    write_json_atomic(stability_json, stability)
    relocated_stability_sha = sha256(stability_json)

    overlay = json.loads(overlay_json.read_text(encoding="utf-8"))
    if overlay.get("output_ledger_sha256") != sha256(ledger):
        raise ValueError("overlay ledger hash differs before relocation")
    if overlay.get("comparison_csv_sha256") != sha256(overlay_cells):
        raise ValueError("overlay timing-cell hash differs before relocation")
    if overlay.get("memory_comparison_csv_sha256") != sha256(
        overlay_memory_cells
    ):
        raise ValueError("overlay memory-cell hash differs before relocation")
    stability_record = overlay.get("timing_stability_audit")
    if not isinstance(stability_record, dict):
        raise ValueError("overlay lacks timing stability evidence")
    if stability_record.get("audit_sha256") != original_stability_sha:
        raise ValueError("overlay stability JSON hash differs before relocation")
    if stability_record.get("rows_csv_sha256") != sha256(stability_csv):
        raise ValueError("overlay stability CSV hash differs before relocation")
    require_local_path(
        stability_record.get("audit_path"),
        stability_json,
        "overlay.timing_stability_audit.audit_path",
    )
    require_local_path(
        stability_record.get("rows_csv_path"),
        stability_csv,
        "overlay.timing_stability_audit.rows_csv_path",
    )
    stability_record["audit_path"] = str(final / stability_json.name)
    stability_record["audit_sha256"] = relocated_stability_sha
    stability_record["rows_csv_path"] = str(final / stability_csv.name)
    stability_record["rows_csv_sha256"] = sha256(stability_csv)
    overlay["output_ledger"] = str(final / ledger.name)
    overlay["comparison_csv"] = str(final / overlay_cells.name)
    overlay["memory_comparison_csv"] = str(
        final / overlay_memory_cells.name
    )
    write_json_atomic(overlay_json, overlay)

    verified = json.loads(overlay_json.read_text(encoding="utf-8"))
    verified_stability = verified["timing_stability_audit"]
    if (
        verified_stability["audit_sha256"] != sha256(stability_json)
        or verified_stability["rows_csv_sha256"] != sha256(stability_csv)
        or verified["output_ledger_sha256"] != sha256(ledger)
    ):
        raise ValueError("relocated V10 evidence failed its final hash check")

    print(
        json.dumps(
            {
                "status": "pass",
                "stage_dir": str(stage),
                "future_final_dir": str(final),
                "stability_json_sha256": sha256(stability_json),
                "stability_csv_sha256": sha256(stability_csv),
                "overlay_audit_sha256": sha256(overlay_json),
                "ledger_sha256": sha256(ledger),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
