import sys
import unittest
from pathlib import Path


BENCHMARKING = Path(__file__).resolve().parents[1]
if str(BENCHMARKING) not in sys.path:
    sys.path.insert(0, str(BENCHMARKING))

from generate_best_observed_sensitivity import build_sensitivity_rows


def rows(baseline, values, dataset="toy", function="PageRank"):
    return [
        {
            "dataset": dataset,
            "function": function,
            "metric": "e2e",
            "baseline": baseline,
            "status": "ok",
            "seconds": str(value),
            "sample_index": str(index),
        }
        for index, value in enumerate(values, 1)
    ]


class TestBestObservedSensitivity(unittest.TestCase):
    def test_near_miss_can_show_transparent_best_observed_flip(self):
        samples = rows("EGGPU", [1.04, 1.01, 0.99, 1.03, 1.02])
        samples += rows("networkx", [1.00] * 5)
        result = build_sensitivity_rows(samples, near_miss_fraction=0.05)

        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertAlmostEqual(row["eggpu_mean_seconds"], 1.018)
        self.assertAlmostEqual(row["eggpu_best_seconds"], 0.99)
        self.assertEqual(row["best_baseline"], "networkx")
        self.assertTrue(row["best_observed_flips_lead"])

    def test_more_than_five_percent_gap_is_not_selected(self):
        samples = rows("EGGPU", [1.10] * 5)
        samples += rows("networkx", [1.00] * 5)
        self.assertEqual(build_sensitivity_rows(samples, 0.05), [])

    def test_existing_eggpu_winner_is_not_a_near_miss(self):
        samples = rows("EGGPU", [0.90] * 5)
        samples += rows("networkx", [1.00] * 5)
        self.assertEqual(build_sensitivity_rows(samples, 0.05), [])


if __name__ == "__main__":
    unittest.main()
