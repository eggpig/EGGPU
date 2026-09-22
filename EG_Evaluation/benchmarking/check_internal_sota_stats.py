#!/usr/bin/env python3
"""Check internal paper SOTA statistics from a full benchmark result directory."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


CONSTRAINT_EXPECTED_FLIPS = {"ca-CondMat", "ca-HepTh", "ca-GrQc", "pgp"}
SUMMARY_REQUIRED = {
    "Correctness gate pass": "yes",
    "Backend separation pass": "yes",
    "EGGPU runtime bad rows": "0",
    "EGGPU validation bad rows": "0",
}


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _load_pair_rows(result_dir: Path) -> list[dict[str, str]]:
    path = result_dir / "eggpu_pair_sota_details.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing pair SOTA details: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _normalize_summary_value(value: str) -> str:
    return value.strip().strip("`").strip().lower()


def _load_summary_values(result_dir: Path) -> dict[str, str]:
    path = result_dir / "EGGPU_FINAL_RESULT_SUMMARY.md"
    if not path.exists():
        raise FileNotFoundError(f"missing final summary: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        values[key.strip()] = _normalize_summary_value(value)
    return values


def _print_summary_gate_checks(result_dir: Path) -> bool:
    values = _load_summary_values(result_dir)
    ok = True
    print("Final-summary gates:")
    for key, expected in SUMMARY_REQUIRED.items():
        actual = values.get(key)
        passed = actual == expected
        ok = ok and passed
        print(
            f"  {key}: {actual if actual is not None else 'missing'} "
            f"(expected {expected}) {'PASS' if passed else 'FAIL'}"
        )
    return ok


def _coverage(rows: list[dict[str, str]], metric: str) -> tuple[int, int, float]:
    metric_rows = [row for row in rows if row.get("metric") == metric]
    total = len(metric_rows)
    sota = sum(1 for row in metric_rows if _as_bool(row.get("is_pair_sota", "")))
    coverage = (sota / total) if total else 0.0
    return sota, total, coverage


def _print_constraint_rows(rows: list[dict[str, str]]) -> bool:
    e2e_rows = [
        row
        for row in rows
        if row.get("metric") == "e2e"
        and row.get("function") == "Constraint"
        and row.get("dataset") in CONSTRAINT_EXPECTED_FLIPS
    ]
    by_dataset = {row.get("dataset"): row for row in e2e_rows}
    ok = True
    print("Constraint rows expected to flip after 2026-06-11 return-path patch:")
    for dataset in sorted(CONSTRAINT_EXPECTED_FLIPS):
        row = by_dataset.get(dataset)
        if row is None:
            print(f"  {dataset}: missing")
            ok = False
            continue
        is_sota = _as_bool(row.get("is_pair_sota", ""))
        ok = ok and is_sota
        print(
            "  {dataset}: {status} "
            "EGGPU={eggpu_seconds}s best={best_baseline}:{best_baseline_seconds}s "
            "ratio={ratio_to_best}".format(
                dataset=dataset,
                status="SOTA" if is_sota else "NON-SOTA",
                eggpu_seconds=row.get("eggpu_seconds", "?"),
                best_baseline=row.get("best_baseline", "?"),
                best_baseline_seconds=row.get("best_baseline_seconds", "?"),
                ratio_to_best=row.get("ratio_to_best", "?"),
            )
        )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--min-full-e2e", type=float, default=0.95)
    parser.add_argument("--min-full-kernel", type=float, default=0.95)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    rows = _load_pair_rows(result_dir)

    checks: list[tuple[str, bool]] = []
    summary_gates_ok = _print_summary_gate_checks(result_dir)
    checks.append(("final_summary_gates", summary_gates_ok))

    for metric, threshold in (
        ("e2e", args.min_full_e2e),
        ("kernel", args.min_full_kernel),
    ):
        sota, total, coverage = _coverage(rows, metric)
        passed = coverage >= threshold
        checks.append((metric, passed))
        print(
            f"full {metric}: {sota}/{total} = {coverage * 100:.2f}% "
            f"(internal >= {threshold * 100:.2f}%) {'PASS' if passed else 'FAIL'}"
        )

    constraint_ok = _print_constraint_rows(rows)
    checks.append(("constraint_expected_flips", constraint_ok))

    failed = [name for name, ok in checks if not ok]
    if failed:
        print("internal paper SOTA check: FAIL -> " + ", ".join(failed))
        return 1
    print("internal paper SOTA check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
