import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "audit_full_result.py"
SPEC = importlib.util.spec_from_file_location("eggpu_test_full_audit", MODULE_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditSampleContractTests(unittest.TestCase):
    def test_complete_five_sample_mean_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            samples = []
            for index in range(1, 6):
                samples.append(
                    {
                        "dataset": "toy",
                        "function": "PageRank",
                        "baseline": "EGGPU",
                        "metric": "e2e",
                        "seconds": str(index),
                        "status": "ok",
                        "sample_index": str(index),
                        "sample_count": "5",
                    }
                )
            self._write_samples(result_dir, samples)
            aggregate = {
                **samples[0],
                "seconds": "3",
                "status": "ok",
                "sample_index": "",
                "aggregation": "arithmetic_mean",
                "n_total": "5",
                "n_valid": "5",
                "publishable": "true",
                "mean_seconds": "3",
                "std_seconds": "1.58113883",
                "variance_seconds2": "2.5",
                "correctness_variant_count": "1",
                "correctness_consistency": "identical",
            }
            self.assertEqual([], AUDIT.audit_sample_contract(result_dir, [aggregate], 5))

    def test_missing_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            samples = [
                {
                    "dataset": "toy",
                    "function": "PageRank",
                    "baseline": "EGGPU",
                    "metric": "e2e",
                    "seconds": "1",
                    "status": "ok",
                    "sample_index": str(index),
                    "sample_count": "5",
                }
                for index in range(1, 5)
            ]
            self._write_samples(result_dir, samples)
            aggregate = {
                **samples[0],
                "status": "incomplete",
                "sample_index": "",
                "aggregation": "arithmetic_mean",
                "n_total": "4",
                "n_valid": "4",
                "publishable": "false",
            }
            issues = AUDIT.audit_sample_contract(result_dir, [aggregate], 5)
            self.assertTrue(any(row["issue"] == "incomplete_measured_group" for row in issues))

    def test_one_sample_memory_pass_uses_its_own_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            sample = {
                "dataset": "toy",
                "function": "PageRank",
                "baseline": "EGGPU",
                "metric": "memory_peak_gpu_proc_mb",
                "seconds": "128",
                "status": "ok",
                "sample_index": "1",
                "sample_count": "1",
                "measurement_phase": "memory",
            }
            self._write_samples(result_dir, [sample])
            aggregate = {
                **sample,
                "sample_index": "",
                "aggregation": "arithmetic_mean",
                "n_total": "1",
                "n_valid": "1",
                "publishable": "true",
                "mean_seconds": "128",
                "std_seconds": "",
                "variance_seconds2": "",
                "correctness_variant_count": "0",
                "correctness_consistency": "not_reported",
            }
            self.assertEqual([], AUDIT.audit_sample_contract(result_dir, [aggregate], 5))

    @staticmethod
    def _write_samples(result_dir, rows):
        with (result_dir / "results_samples.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
