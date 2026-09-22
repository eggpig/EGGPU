import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARKING = Path(__file__).resolve().parents[1]
if str(BENCHMARKING) not in sys.path:
    sys.path.insert(0, str(BENCHMARKING))

import audit_eggpu_timing_stability as audit


def write_samples(result_dir, values):
    rows = []
    for index, value in enumerate(values, 1):
        rows.append(
            {
                "dataset": "toy",
                "function": "Efficiency",
                "baseline": "EGGPU",
                "metric": "e2e",
                "value": value,
                "seconds": value,
                "status": "ok",
                "sample_index": index,
                "measurement_phase": "timing",
            }
        )
    path = result_dir / "results_samples.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class EggpuTimingStabilityAuditTests(unittest.TestCase):
    def audit_rows(self, values):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        result_dir = Path(temporary.name)
        write_samples(result_dir, values)
        return audit.main_rows(
            result_dir,
            "e2e",
            5,
            5.0,
            3.0,
        )

    def test_raw5_min_mean_sample_sd_and_cv_are_preserved(self):
        row = self.audit_rows((1.0, 1.2, 1.4, 1.6, 1.8))[0]

        self.assertEqual(json.loads(row["raw5_seconds"]), [1, 1.2, 1.4, 1.6, 1.8])
        self.assertEqual(row["minimum_seconds"], 1.0)
        self.assertEqual(row["arithmetic_mean_seconds"], 1.4)
        self.assertGreater(row["sample_std_seconds"], 0.0)
        self.assertGreater(row["coefficient_of_variation"], 0.10)
        self.assertEqual(row["stability_status"], "pass")
        self.assertEqual(row["submission_seconds"], 1.0)
        self.assertEqual(
            row["batch_acceptance_status"], "accepted_complete_batch"
        )

    def test_huge_high_value_has_no_submission_value(self):
        row = self.audit_rows((1.0, 1.1, 1.2, 1.3, 100.0))[0]

        self.assertEqual(row["stability_status"], "fail")
        self.assertEqual(row["submission_seconds"], "")
        self.assertEqual(
            row["batch_acceptance_status"], "rejected_entire_batch"
        )
        self.assertIn(
            "max_over_median_exceeds_limit",
            json.loads(row["failure_reasons"]),
        )

    def test_unique_low_value_has_no_submission_value(self):
        row = self.audit_rows((0.1, 1.0, 1.1, 1.2, 1.3))[0]

        self.assertEqual(row["stability_status"], "fail")
        self.assertEqual(row["submission_seconds"], "")
        self.assertIn(
            "median_over_min_exceeds_limit",
            json.loads(row["failure_reasons"]),
        )

    def test_duplicate_or_missing_sample_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = Path(temporary)
            write_samples(result_dir, (1.0, 1.1, 1.2, 1.3))
            with self.assertRaisesRegex(
                audit.StabilityError, "sample indices"
            ):
                audit.main_rows(
                    result_dir,
                    "e2e",
                    5,
                    5.0,
                    3.0,
                )

    def replacement_row(self, root, name, values):
        result_dir = root / name
        result_dir.mkdir()
        evidence = result_dir / "results_samples.csv"
        evidence.write_text(name, encoding="utf-8")
        return audit.summarize_batch(
            dataset="soc-Slashdot0811",
            function="EffectiveSize",
            metric="e2e",
            source_kind="main",
            result_source=result_dir,
            evidence_path=evidence,
            values=values,
            expected_samples=5,
            max_over_median_limit=5.0,
            median_over_min_limit=3.0,
        )

    def test_duplicate_key_requires_explicit_failed_batch_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.replacement_row(
                root, "original", (0.2, 0.21, 0.22, 0.23, 9.0)
            )
            rerun = self.replacement_row(
                root, "rerun", (0.21, 0.23, 0.30, 0.40, 0.41)
            )

            with self.assertRaisesRegex(
                audit.StabilityError, "implicit or fastest-batch"
            ):
                audit.apply_explicit_replacements(
                    [original, rerun],
                    [],
                    {"path": "", "sha256": ""},
                )

    def test_explicit_replacement_records_both_evidence_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.replacement_row(
                root, "original", (0.2, 0.21, 0.22, 0.23, 9.0)
            )
            rerun = self.replacement_row(
                root, "rerun", (0.21, 0.23, 0.30, 0.40, 0.41)
            )
            entry = {
                "dataset": "soc-Slashdot0811",
                "function": "EffectiveSize",
                "metric": "e2e",
                "original_result_source": original["result_source"],
                "replacement_result_source": rerun["result_source"],
                "reason": "original batch failed the declared outlier guard",
                "selection_basis": audit.REPLACEMENT_SELECTION_BASIS,
            }
            selected, overrides = audit.apply_explicit_replacements(
                [original, rerun],
                [entry],
                {"path": "/tmp/override.json", "sha256": "a" * 64},
            )

            self.assertEqual(len(selected), 1)
            self.assertEqual(
                selected[0]["result_source"], rerun["result_source"]
            )
            self.assertEqual(
                selected[0]["batch_selection_status"],
                "explicit_failed_batch_replacement",
            )
            self.assertEqual(
                selected[0]["original_evidence_sha256"],
                original["evidence_sha256"],
            )
            self.assertEqual(len(overrides), 1)
            self.assertEqual(
                overrides[0]["replacement_evidence_sha256"],
                rerun["evidence_sha256"],
            )

    def test_replacement_of_passing_batch_is_rejected_as_speed_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.replacement_row(
                root, "first", (1.0, 1.1, 1.2, 1.3, 1.4)
            )
            second = self.replacement_row(
                root, "second", (0.8, 0.9, 1.0, 1.1, 1.2)
            )
            entry = {
                "dataset": "soc-Slashdot0811",
                "function": "EffectiveSize",
                "metric": "e2e",
                "original_result_source": first["result_source"],
                "replacement_result_source": second["result_source"],
                "reason": "second happens to be faster",
                "selection_basis": audit.REPLACEMENT_SELECTION_BASIS,
            }

            with self.assertRaisesRegex(
                audit.StabilityError, "speed-based replacement"
            ):
                audit.apply_explicit_replacements(
                    [first, second],
                    [entry],
                    {"path": "/tmp/override.json", "sha256": "a" * 64},
                )

    def test_manifest_rejects_performance_selection_basis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "override.json"
            manifest.write_text(
                json.dumps(
                    {
                        "replacements": [
                            {
                                "dataset": "toy",
                                "function": "Efficiency",
                                "metric": "e2e",
                                "original_result_source": "/tmp/a",
                                "replacement_result_source": "/tmp/b",
                                "reason": "pick faster batch",
                                "selection_basis": "minimum_across_batches",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                audit.StabilityError, "performance-based"
            ):
                audit.load_replacement_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
