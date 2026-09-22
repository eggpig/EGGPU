#!/usr/bin/env python3
"""Fail closed unless a staged V10 timing-memory overlay is publishable."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


EXPECTED_CELLS = 208
EXPECTED_LEDGER_CELLS = 13 * 16 * 8
EXPECTED_ESTIMATOR = "minimum_of_five"
EXPECTED_VARIANCE_POLICY = "catastrophic_outlier_guard"
EXPECTED_BATCH_SELECTION = (
    "unique_complete_batch_per_key_no_cross_batch_selection"
)
EXPECTED_MAX_OVER_MEDIAN = 5.0
EXPECTED_MEDIAN_OVER_MIN = 3.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_sha(value: str, label: str) -> str:
    normalized = str(value).strip().lower()
    require(
        bool(re.fullmatch(r"[0-9a-f]{64}", normalized)),
        f"{label} is not a SHA-256 digest",
    )
    return normalized


def require_close(actual: object, expected: float, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    require(
        math.isfinite(value)
        and math.isclose(
            value,
            expected,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ),
        f"{label}={value} differs from {expected}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability-json", required=True, type=Path)
    parser.add_argument("--overlay-json", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    args = parser.parse_args()

    candidate_sha = require_sha(args.candidate_sha256, "candidate SHA")
    runtime_sha = require_sha(args.runtime_sha256, "runtime SHA")
    stability_path = args.stability_json.resolve(strict=True)
    overlay_path = args.overlay_json.resolve(strict=True)
    ledger_path = args.ledger.resolve(strict=True)
    stability_rows_path = stability_path.with_suffix(".csv")
    timing_cells_path = overlay_path.with_suffix(".cells.csv")
    memory_cells_path = overlay_path.with_suffix(".memory_cells.csv")
    for path in (
        stability_rows_path,
        timing_cells_path,
        memory_cells_path,
    ):
        require(path.is_file(), f"required gate file is missing: {path}")

    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    expected_stability = {
        "status": "pass",
        "metric": "e2e",
        "paper_estimator": EXPECTED_ESTIMATOR,
        "acceptance_estimator_independent": True,
        "variance_policy": EXPECTED_VARIANCE_POLICY,
        "sample_std_and_cv_role": (
            "reported_diagnostics_not_acceptance_gate"
        ),
        "failed_batch_policy": (
            "reject_entire_batch_do_not_select_across_batches"
        ),
        "batch_selection_policy": EXPECTED_BATCH_SELECTION,
        "expected_samples_per_cell": 5,
        "audited_cells": EXPECTED_CELLS,
        "unique_workload_keys": EXPECTED_CELLS,
        "stable_cells": EXPECTED_CELLS,
        "unstable_cells": 0,
        "failures": [],
        "candidate_sha256": candidate_sha,
        "runtime_python_snapshot_sha256": runtime_sha,
        "runtime_package_is_symlink": False,
        "override_manifest_path": "",
        "override_manifest_sha256": "",
        "batch_overrides_applied": [],
    }
    for field, expected in expected_stability.items():
        require(
            stability.get(field) == expected,
            f"stability {field}={stability.get(field)!r}, "
            f"expected {expected!r}",
        )
    require_close(
        stability.get("max_over_median_limit"),
        EXPECTED_MAX_OVER_MEDIAN,
        "stability max/median limit",
    )
    require_close(
        stability.get("median_over_min_limit"),
        EXPECTED_MEDIAN_OVER_MIN,
        "stability median/min limit",
    )
    require(
        stability.get("rows_csv_sha256") == sha256(stability_rows_path),
        "stability CSV hash differs",
    )
    require(
        Path(str(stability.get("rows_csv_path", ""))).resolve()
        == stability_rows_path,
        "stability CSV path differs",
    )
    stability_rows = load_csv(stability_rows_path)
    stability_keys = [
        (row["dataset"], row["function"], row["metric"])
        for row in stability_rows
    ]
    require(
        len(stability_rows) == EXPECTED_CELLS
        and len(set(stability_keys)) == EXPECTED_CELLS,
        "stability CSV does not contain 208 unique keys",
    )
    for row in stability_rows:
        key = (row["dataset"], row["function"])
        require(
            row.get("stability_status") == "pass"
            and row.get("batch_acceptance_status")
            == "accepted_complete_batch"
            and row.get("batch_selection_status") == "unique_input_batch",
            f"{key}: timing batch is not one accepted unique batch",
        )
        require(
            row.get("paper_estimator") == EXPECTED_ESTIMATOR,
            f"{key}: paper estimator differs",
        )
        require_close(
            row.get("submission_seconds"),
            float(row["minimum_seconds"]),
            f"{key}: submission minimum",
        )
        require_close(
            row.get("median_over_min_limit"),
            EXPECTED_MEDIAN_OVER_MIN,
            f"{key}: median/min limit",
        )

    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    expected_overlay = {
        "status": "pass",
        "candidate_mode": "unified-timing-memory",
        "paper_timing_estimator": EXPECTED_ESTIMATOR,
        "timing_variance_policy": EXPECTED_VARIANCE_POLICY,
        "stability_acceptance_uses_minimum_value": False,
        "ledger_cells": EXPECTED_LEDGER_CELLS,
        "eggpu_cells": EXPECTED_CELLS,
        "eggpu_ok_cells": EXPECTED_CELLS,
        "accepted_candidate_cells": EXPECTED_CELLS,
        "timing_accepted_candidate_cells": EXPECTED_CELLS,
        "timing_replacement_count": EXPECTED_CELLS,
        "expected_timing_replacements": EXPECTED_CELLS,
        "memory_accepted_candidate_cells": EXPECTED_CELLS,
        "memory_replacement_count": EXPECTED_CELLS,
        "expected_memory_replacements": EXPECTED_CELLS,
        "candidate_sha256": [candidate_sha],
        "runtime_python_snapshot_sha256": [runtime_sha],
        "rejected_candidates": [],
    }
    for field, expected in expected_overlay.items():
        require(
            overlay.get(field) == expected,
            f"overlay {field}={overlay.get(field)!r}, expected {expected!r}",
        )
    require_close(
        overlay.get("stability_max_over_median_limit"),
        EXPECTED_MAX_OVER_MEDIAN,
        "overlay max/median limit",
    )
    require_close(
        overlay.get("stability_median_over_min_limit"),
        EXPECTED_MEDIAN_OVER_MIN,
        "overlay median/min limit",
    )
    require(
        overlay.get("output_ledger_sha256") == sha256(ledger_path),
        "overlay ledger hash differs",
    )
    require(
        overlay.get("comparison_csv_sha256") == sha256(timing_cells_path),
        "overlay timing-cell hash differs",
    )
    require(
        overlay.get("memory_comparison_csv_sha256")
        == sha256(memory_cells_path),
        "overlay memory-cell hash differs",
    )
    stability_record = overlay.get("timing_stability_audit")
    require(
        isinstance(stability_record, dict)
        and stability_record.get("status") == "pass"
        and stability_record.get("audited_cells") == EXPECTED_CELLS
        and stability_record.get("batch_overrides_applied") == []
        and stability_record.get("audit_sha256") == sha256(stability_path)
        and stability_record.get("rows_csv_sha256")
        == sha256(stability_rows_path),
        "overlay timing stability binding differs",
    )

    timing_cells = load_csv(timing_cells_path)
    memory_cells = load_csv(memory_cells_path)
    for label, rows in (
        ("timing", timing_cells),
        ("memory", memory_cells),
    ):
        keys = [(row["dataset"], row["function"]) for row in rows]
        require(
            len(rows) == EXPECTED_CELLS
            and len(set(keys)) == EXPECTED_CELLS,
            f"overlay {label} comparison lacks 208 unique cells",
        )
        require(
            {row["candidate_sha256"] for row in rows} == {candidate_sha}
            and {
                row["runtime_python_snapshot_sha256"] for row in rows
            }
            == {runtime_sha},
            f"overlay {label} cells use a different candidate/runtime",
        )
    for row in timing_cells:
        key = (row["dataset"], row["function"])
        require(
            row.get("new_e2e_estimator") == EXPECTED_ESTIMATOR
            and row.get("new_e2e_stability_status") == "pass",
            f"{key}: final timing cell lacks minimum/stability evidence",
        )
        require_close(
            row.get("new_e2e_paper_seconds"),
            float(row["new_e2e_raw_min_seconds"]),
            f"{key}: final paper minimum",
        )

    print(
        json.dumps(
            {
                "status": "pass",
                "candidate_sha256": candidate_sha,
                "runtime_python_snapshot_sha256": runtime_sha,
                "timing_cells": len(timing_cells),
                "memory_cells": len(memory_cells),
                "stability_cells": len(stability_rows),
                "ledger_sha256": sha256(ledger_path),
                "overlay_audit_sha256": sha256(overlay_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
