#!/usr/bin/env python3
"""Replace selected baseline cells and regenerate correctness-qualified tables.

Each replacement directory declares its scope through the baseline, dataset,
function, and metric keys present in ``results_long.csv``. Raw samples for
those same cells are replaced as one unit; all unrelated rows remain from the
base result. This keeps targeted recovery runs auditable without mixing old
and new samples inside one aggregate cell.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from measurement_schema import write_measurement_schema
from run_full_baselines import write_metric_csvs_no_pandas
from validate_correctness import write_validation_outputs


CELL_FIELDS = ("dataset", "function", "baseline")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(str(row.get(field, "")) for field in CELL_FIELDS)


def replacement_scope(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    scope = {
        cell_key(row)
        for row in rows
        if all(str(row.get(field, "")).strip() for field in CELL_FIELDS)
    }
    if not scope:
        raise ValueError("replacement contains no baseline/dataset/function cells")
    return scope


def merge_rows(
    base_rows: list[dict[str, str]],
    replacement_rows: list[dict[str, str]],
    scope: set[tuple[str, str, str]],
) -> list[dict[str, str]]:
    retained = [row for row in base_rows if cell_key(row) not in scope]
    current = [row for row in replacement_rows if cell_key(row) in scope]
    return retained + current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--replacement",
        type=Path,
        action="append",
        required=True,
        help="Targeted result directory; may be supplied more than once.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = args.base.resolve()
    replacements = [path.resolve() for path in args.replacement]
    output = args.output.resolve()
    required = [base / "results_long.csv", base / "results_samples.csv"]
    for replacement in replacements:
        required.extend(
            [replacement / "results_long.csv", replacement / "results_samples.csv"]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing result input(s): " + ", ".join(missing))

    merged_long = read_csv(base / "results_long.csv")
    merged_samples = read_csv(base / "results_samples.csv")
    provenance = []
    replaced_cells: set[tuple[str, str, str]] = set()
    for replacement in replacements:
        replacement_long = read_csv(replacement / "results_long.csv")
        replacement_samples = read_csv(replacement / "results_samples.csv")
        scope = replacement_scope(replacement_long)
        overlap = replaced_cells.intersection(scope)
        if overlap:
            raise ValueError(
                "replacement scopes overlap: "
                + ", ".join("/".join(key) for key in sorted(overlap))
            )
        merged_long = merge_rows(merged_long, replacement_long, scope)
        merged_samples = merge_rows(merged_samples, replacement_samples, scope)
        replaced_cells.update(scope)
        provenance.append(
            {
                "path": str(replacement),
                "results_long_sha256": sha256(replacement / "results_long.csv"),
                "results_samples_sha256": sha256(
                    replacement / "results_samples.csv"
                ),
                "cells": [
                    {"dataset": key[0], "function": key[1], "baseline": key[2]}
                    for key in sorted(scope)
                ],
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "results_long.csv", merged_long)
    write_csv(output / "results_samples.csv", merged_samples)
    write_measurement_schema(output / "measurement_schema.json")
    write_validation_outputs(output, merged_long)
    write_metric_csvs_no_pandas(output, merged_long)

    for name in ("dataset_stats.json", "baseline_versions.json"):
        source = base / name
        if source.is_file():
            shutil.copy2(source, output / name)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_kind": "targeted_latest_baseline_replacement",
        "base": {
            "path": str(base),
            "results_long_sha256": sha256(base / "results_long.csv"),
            "results_samples_sha256": sha256(base / "results_samples.csv"),
        },
        "replacements": provenance,
        "replacement_policy": (
            "replace every metric and raw sample for each declared "
            "baseline/dataset/function cell; retain all unrelated base rows"
        ),
        "correctness_policy": (
            "regenerate correctness after merging so targeted baselines are "
            "compared with independent references from the base result"
        ),
        "row_counts": {
            "results_long": len(merged_long),
            "results_samples": len(merged_samples),
            "replaced_cells": len(replaced_cells),
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
