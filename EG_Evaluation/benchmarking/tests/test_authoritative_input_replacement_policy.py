from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "prepare_final_13_authoritative_inputs.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("authoritative_inputs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class AuthoritativeInputReplacementPolicyTests(unittest.TestCase):
    def test_current_validated_closeness_replaces_faster_historical_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / "old"
            current = root / "current"
            reference = root / "reference"
            output = root / "output"
            old.mkdir()
            reference.mkdir()

            old_rows = [
                {
                    "dataset": "g",
                    "function": "Closeness",
                    "baseline": "EGGPU",
                    "metric": metric,
                    "seconds": 1.0 + sample_index / 100,
                    "status": "ok",
                    "sample_index": sample_index,
                }
                for metric in ("build", "e2e", "kernel")
                for sample_index in range(1, 6)
            ]
            for metric in ("build", "e2e", "kernel"):
                write_csv(
                    old / f"closeness_large_sampled_{metric}.csv",
                    [row for row in old_rows if row["metric"] == metric],
                )
            write_csv(
                old / "closeness_large_sampled_validation.csv",
                [
                    {
                        "dataset": "g",
                        "function": "Closeness",
                        "baseline": "EGGPU",
                        "validation_status": "pass",
                    }
                ],
            )

            current_detail = root / "current_detail.npz"
            reference_detail = root / "reference_detail.npz"
            np.savez(current_detail, values=np.array([0.5]), sources=np.array([7]))
            np.savez(reference_detail, values=np.array([0.5]), sources=np.array([7]))
            current_rows = [
                {
                    "dataset": "g",
                    "function": "Closeness",
                    "baseline": "EGGPU",
                    "metric": metric,
                    "seconds": 10.0 + sample_index,
                    "value": 10.0 + sample_index,
                    "status": "ok",
                    "sample_index": sample_index,
                    "correctness": f"detail={current_detail}",
                }
                for metric in ("e2e", "kernel")
                for sample_index in range(1, 6)
            ]
            write_csv(
                current / "measurement_passes" / "timing" / "results_samples.csv",
                current_rows,
            )
            write_csv(
                reference / "results_samples.csv",
                [
                    {
                        "dataset": "g",
                        "function": "Closeness",
                        "baseline": "easygraph-cpu",
                        "metric": "e2e",
                        "status": "ok",
                        "correctness": f"detail={reference_detail}",
                    }
                ],
            )

            decisions = MODULE.prepare_closeness(old, current, reference, output)
            selected = MODULE.read_rows(output / "closeness_large_sampled_e2e.csv")
            preserved_build = MODULE.read_rows(
                output / "closeness_large_sampled_build.csv"
            )

            self.assertEqual({float(row["seconds"]) for row in selected}, {11, 12, 13, 14, 15})
            self.assertEqual(
                {float(row["seconds"]) for row in preserved_build},
                {1.01, 1.02, 1.03, 1.04, 1.05},
            )
            self.assertTrue(
                all(
                    row["selected_source"] == "current_qualified_retest"
                    for row in decisions
                )
            )
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                MODULE.prepare_closeness(old, current, reference, output)


if __name__ == "__main__":
    unittest.main()
