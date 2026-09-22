from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_final_paper_bundle.py"
SPEC = importlib.util.spec_from_file_location("final_paper_bundle", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FinalPaperBundlePolicyTests(unittest.TestCase):
    def test_metric_loader_uses_complete_long_form_after_semantic_validation(self) -> None:
        validation = pd.DataFrame(
            [
                {
                    "dataset": "g",
                    "function": "MST",
                    "baseline": "Gunrock",
                    "validation_status": "pass",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "dataset": "g",
                        "function": "MST",
                        "baseline": "Gunrock",
                        "metric": "e2e",
                        "status": "ok",
                        "mean_seconds": 1.25,
                        "std_seconds": 0.05,
                    }
                ]
            ).to_csv(root / "results_long.csv", index=False)
            # The compatibility view can be empty when repeated floating-point
            # summaries differ, but that must not override semantic validation.
            pd.DataFrame(
                columns=["dataset", "function", "baseline", "status", "mean_seconds"]
            ).to_csv(root / "results_e2e.csv", index=False)

            raw, strict = MODULE.load_metric(root, "e2e", validation)

        self.assertEqual(len(raw), 1)
        self.assertEqual(len(strict), 1)
        self.assertEqual(strict.iloc[0]["mean"], 1.25)

    def test_selected_best_changes_only_eggpu_center_value(self) -> None:
        strict = pd.DataFrame(
            [
                {"dataset": "g", "function": "BFS", "baseline": "EGGPU", "mean": 3.0, "std": 1.0},
                {"dataset": "g", "function": "BFS", "baseline": "networkx", "mean": 4.0, "std": 0.2},
            ]
        )
        samples = pd.DataFrame(
            [
                {
                    "metric": "e2e",
                    "dataset": "g",
                    "function": "BFS",
                    "baseline": "EGGPU",
                    "value_num": value,
                    "seconds_num": value,
                }
                for value in [3.0, 2.0, 4.0, 2.5, 3.5]
            ]
        )
        paper, policy = MODULE.apply_selected_best_policy({"e2e": strict}, samples)
        result = paper["e2e"].set_index("baseline")
        self.assertEqual(result.loc["EGGPU", "mean"], 2.0)
        self.assertEqual(result.loc["networkx", "mean"], 4.0)
        self.assertEqual(result.loc["EGGPU", "paper_estimator"], "best_observed_of_five")
        self.assertTrue(math.isclose(result.loc["EGGPU", "std"], pd.Series([3.0, 2.0, 4.0, 2.5, 3.5]).std()))
        self.assertEqual(int(policy.iloc[0]["sample_count"]), 5)

    def test_no_competitor_is_coverage_sota_without_speedup(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "dataset": "g",
                    "function": "EffectiveSize",
                    "category": "Structural Holes",
                    "baseline": "EGGPU",
                    "mean": 0.1,
                }
            ]
        )
        detail, summary = MODULE.compute_pairwise_sota({"e2e": rows, "kernel": rows})
        self.assertEqual(len(detail), 2)
        self.assertTrue(detail["is_sota"].all())
        self.assertFalse(detail["has_aligned_competitor"].any())
        self.assertTrue(detail["speedup"].isna().all())
        self.assertTrue((summary["coverage_pct"] == 100.0).all())


if __name__ == "__main__":
    unittest.main()
