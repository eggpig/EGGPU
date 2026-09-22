import tempfile
import unittest
from pathlib import Path

import pandas as pd

from benchmarking import summarize_final_result as summary


class FinalResultSotaSemanticsTests(unittest.TestCase):
    def _result_dir(self, include_competitor: bool) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        rows = [
            {
                "dataset": "g",
                "function": "Hierarchy",
                "baseline": "EGGPU",
                "status": "ok",
                "seconds": 1.0,
            }
        ]
        validation = [
            {
                "dataset": "g",
                "function": "Hierarchy",
                "baseline": "EGGPU",
                "validation_status": "sampled_pass",
            }
        ]
        if include_competitor:
            rows.append(
                {
                    "dataset": "g",
                    "function": "Hierarchy",
                    "baseline": "networkx",
                    "status": "ok",
                    "seconds": 2.0,
                }
            )
            validation.append(
                {
                    "dataset": "g",
                    "function": "Hierarchy",
                    "baseline": "networkx",
                    "validation_status": "pass",
                }
            )
        pd.DataFrame(rows).to_csv(root / "results_e2e.csv", index=False)
        pd.DataFrame(validation).to_csv(root / "correctness_validation.csv", index=False)
        return root

    def test_unique_coverage_is_not_counted_as_sota(self):
        result, details = summary.summarize_metric(
            self._result_dir(False), "e2e", "full", lambda _: True
        )
        self.assertEqual(result["all_eggpu_pairs"], 1)
        self.assertEqual(result["coverage_only_pairs"], 1)
        self.assertEqual(result["total_pairs"], 0)
        self.assertEqual(result["sota_pairs"], 0)
        self.assertTrue(bool(details.iloc[0]["coverage_only"]))
        self.assertFalse(bool(details.iloc[0]["is_sota"]))

    def test_validated_competitor_creates_comparable_pair(self):
        result, details = summary.summarize_metric(
            self._result_dir(True), "e2e", "full", lambda _: True
        )
        self.assertEqual(result["coverage_only_pairs"], 0)
        self.assertEqual(result["total_pairs"], 1)
        self.assertEqual(result["sota_pairs"], 1)
        self.assertTrue(bool(details.iloc[0]["has_comparator"]))


if __name__ == "__main__":
    unittest.main()
