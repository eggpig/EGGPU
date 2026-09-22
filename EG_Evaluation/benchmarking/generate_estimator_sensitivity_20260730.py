#!/usr/bin/env python3
"""Derive an equal-estimator headline sensitivity check from the V15 ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import generate_headline_results_20260730 as headline


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    ledger_path = args.ledger.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(ledger_path, low_memory=False)
    datasets, functions = headline.validate_ledger(ledger)

    equal = ledger.copy()
    eggpu = equal["baseline"].astype(str).eq(headline.EGGPU)
    successful = equal["execution_status"].astype(str).eq("ok")
    for metric in ("build", "kernel", "e2e"):
        source = f"{metric}_raw_mean_seconds"
        if source not in equal:
            raise headline.AuditError(f"ledger lacks {source}")
        values = pd.to_numeric(equal.loc[eggpu & successful, source], errors="raise")
        if not values.map(headline.finite_positive).all():
            raise headline.AuditError(f"EGGPU {source} contains an invalid value")
        equal.loc[eggpu & successful, f"{metric}_paper_seconds"] = values
        equal.loc[eggpu & successful, f"{metric}_estimator"] = (
            "arithmetic_mean_of_five"
        )

    index = headline.ledger_index(equal)
    library_comparisons = headline.derive_comparisons(
        index,
        datasets,
        functions,
        "e2e",
        headline.LIBRARY_E2E_SYSTEMS,
    )
    native_comparisons = headline.derive_comparisons(
        index,
        datasets,
        functions,
        "kernel",
        headline.NATIVE_GPU_SYSTEMS,
    )
    strict_nx_comparisons = headline.derive_comparisons(
        index,
        datasets,
        functions,
        "e2e",
        (headline.STRICT_NXCUGRAPH,),
    )
    rows = [
        headline.summarize(
            "library_public_call_e2e",
            "Library public-call E2E",
            "e2e",
            "pairwise best correctness-qualified library baseline",
            "public function-call E2E",
            library_comparisons,
            "arithmetic_mean_of_five_for_all_systems",
        ),
        headline.summarize(
            "native_gpu_processing",
            "Native-GPU processing",
            "kernel",
            "pairwise best correctness-qualified native GPU implementation",
            "measured device interval",
            native_comparisons,
            "arithmetic_mean_of_five_for_all_systems",
        ),
        headline.summarize(
            "strict_nxcugraph_public_call_e2e",
            "Strict nx-cugraph public-call E2E",
            "e2e",
            "strict nx-cugraph only",
            "public function-call E2E",
            strict_nx_comparisons,
            "arithmetic_mean_of_five_for_all_systems",
        ),
    ]

    csv_path = output / "equal_mean_estimator_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    expected = {
        "library_public_call_e2e": (167, 161, 0, 6, 8.36),
        "native_gpu_processing": (105, 99, 0, 6, 7.49),
        "strict_nxcugraph_public_call_e2e": (99, 99, 0, 0, 42.05),
    }
    for row in rows:
        observed = (
            row.common_pairs,
            row.eggpu_strict_wins,
            row.eggpu_ties,
            row.eggpu_losses,
            round(row.geomean_speedup, 2),
        )
        if observed != expected[row.comparison_id]:
            raise headline.AuditError(
                f"{row.comparison_id} sensitivity result {observed} "
                f"!= {expected[row.comparison_id]}"
            )

    manifest = {
        "schema_version": "eggpu_equal_mean_estimator_sensitivity_v1",
        "status": "pass",
        "purpose": (
            "Sensitivity analysis only; the declared primary policy remains "
            "EGGPU minimum-of-five and external-baseline arithmetic-mean-of-five."
        ),
        "source_ledger": ledger_path.name,
        "source_ledger_sha256": sha256(ledger_path),
        "sample_count_per_system_cell": 5,
        "sensitivity_estimator": "arithmetic_mean_of_five_for_all_systems",
        "reported_error_in_primary_assets": (
            "sample_standard_deviation_ddof1_same_five"
        ),
        "results": [
            {
                "comparison_id": row.comparison_id,
                "common_pairs": row.common_pairs,
                "eggpu_strict_wins": row.eggpu_strict_wins,
                "eggpu_ties": row.eggpu_ties,
                "eggpu_losses": row.eggpu_losses,
                "eggpu_only_vs_set": row.eggpu_only_vs_set,
                "geomean_speedup": row.geomean_speedup,
            }
            for row in rows
        ],
        "output_sha256": {csv_path.name: sha256(csv_path)},
    }
    manifest_path = output / "EQUAL_MEAN_ESTIMATOR_SENSITIVITY.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
