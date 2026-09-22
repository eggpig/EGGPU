import math
import sys
import unittest
from pathlib import Path


BENCHMARKING = Path(__file__).resolve().parents[1]
if str(BENCHMARKING) not in sys.path:
    sys.path.insert(0, str(BENCHMARKING))

from benchmark_stats import aggregate_sample_rows


def sample(index, seconds=None, status="ok", baseline="EGGPU", metric="e2e"):
    return {
        "dataset_size": "small",
        "graph_type": "undirected",
        "dataset": "toy",
        "function": "PageRank",
        "baseline": baseline,
        "metric": metric,
        "seconds": "" if seconds is None else str(seconds),
        "status": status,
        "correctness": "pass" if status == "ok" else "",
        "log": f"sample-{index}.log",
        "notes": "",
        "semantic": "",
        "skip_reason": "",
        "estimator_kind": "",
        "sample_sources": "",
        "source_policy": "",
        "source_seed": "",
        "source_nodes_sha": "",
        "sample_index": str(index),
        "sample_count": "5",
    }


class TestBenchmarkStats(unittest.TestCase):
    def test_five_valid_samples_use_arithmetic_mean_and_sample_variance(self):
        rows = [sample(i, value) for i, value in enumerate((1, 2, 3, 4, 5), 1)]
        result = aggregate_sample_rows(rows, expected_samples=5)

        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["n_valid"], "5")
        self.assertEqual(float(row["seconds"]), 3.0)
        self.assertEqual(float(row["mean_seconds"]), 3.0)
        self.assertEqual(float(row["median_seconds"]), 3.0)
        self.assertEqual(float(row["min_seconds"]), 1.0)
        self.assertEqual(float(row["max_seconds"]), 5.0)
        self.assertEqual(float(row["variance_seconds2"]), 2.5)
        self.assertAlmostEqual(float(row["std_seconds"]), math.sqrt(2.5))
        self.assertEqual(float(row["mean_value"]), 3.0)
        self.assertEqual(float(row["variance_value2"]), 2.5)
        self.assertAlmostEqual(
            float(row["relative_std_percent"]), math.sqrt(2.5) / 3.0 * 100.0
        )
        self.assertGreater(float(row["ci95_half_width_value"]), 0.0)
        self.assertEqual(row["stability_class"], "variable_gt_10pct_rsd")

    def test_value_column_is_canonical_and_units_are_preserved(self):
        rows = [sample(i, value) for i, value in enumerate((10, 10, 10, 10, 10), 1)]
        for row in rows:
            row["seconds"] = "legacy-do-not-read"
            row["value"] = "10"
            row["unit"] = "MiB"
            row["metric_family"] = "memory"
        result = aggregate_sample_rows(rows, expected_samples=5)[0]

        self.assertEqual(float(result["value"]), 10.0)
        self.assertEqual(result["unit"], "MiB")
        self.assertEqual(result["metric_family"], "memory")
        self.assertEqual(float(result["relative_std_percent"]), 0.0)
        self.assertEqual(result["stability_class"], "stable_le_5pct_rsd")

    def test_missing_valid_sample_is_incomplete_and_not_publishable(self):
        rows = [sample(i, value) for i, value in enumerate((1, 2, 3, 4), 1)]
        rows.append(sample(5, status="timeout"))
        row = aggregate_sample_rows(rows, expected_samples=5)[0]

        self.assertEqual(row["status"], "incomplete")
        self.assertEqual(row["n_valid"], "4")
        self.assertEqual(row["n_total"], "5")
        self.assertEqual(row["publishable"], "false")

    def test_static_unsupported_entry_does_not_require_five_probes(self):
        row = aggregate_sample_rows(
            [sample(1, status="unsupported")], expected_samples=5
        )[0]

        self.assertEqual(row["status"], "unsupported")
        self.assertEqual(row["n_total"], "1")
        self.assertEqual(row["publishable"], "false")

    def test_grouping_keeps_metrics_and_baselines_separate(self):
        rows = []
        for index in range(1, 6):
            rows.append(sample(index, index, baseline="EGGPU", metric="e2e"))
            rows.append(sample(index, index / 10, baseline="EGGPU", metric="kernel"))
            rows.append(sample(index, index * 2, baseline="networkx", metric="e2e"))
        result = aggregate_sample_rows(rows, expected_samples=5)

        self.assertEqual(len(result), 3)
        keys = {(r["baseline"], r["metric"]) for r in result}
        self.assertEqual(
            keys,
            {("EGGPU", "e2e"), ("EGGPU", "kernel"), ("networkx", "e2e")},
        )

    def test_grouping_keeps_memory_measurement_windows_separate(self):
        rows = []
        for index in range(1, 4):
            algorithm = sample(index, 400 + index, metric="memory_peak_gpu_proc_mb")
            algorithm.update(
                {
                    "unit": "MiB",
                    "metric_family": "memory",
                    "measurement_scope": "benchmark_process_tree_gpu_memory",
                    "timer_kind": "nvml_process_tree_sampling",
                    "measurement_window": "algorithm_call",
                    "sample_count": "3",
                }
            )
            subprocess = dict(algorithm)
            subprocess["seconds"] = str(500 + index)
            subprocess["value"] = str(500 + index)
            subprocess["measurement_window"] = "isolated_memory_subprocess"
            rows.extend((algorithm, subprocess))

        result = aggregate_sample_rows(rows, expected_samples=3)

        self.assertEqual(len(result), 2)
        self.assertEqual(
            {row["measurement_window"] for row in result},
            {"algorithm_call", "isolated_memory_subprocess"},
        )
        self.assertTrue(all(row["status"] == "ok" for row in result))
        self.assertTrue(all(row["n_total"] == "3" for row in result))

    def test_correctness_detail_is_preserved_for_validation(self):
        rows = [sample(i, i) for i in range(1, 6)]
        for index, row in enumerate(rows, 1):
            row["correctness"] = f"sum=1, detail=/tmp/vector.npz, detail_sha=sha{index}"
        result = aggregate_sample_rows(rows, expected_samples=5)[0]

        self.assertEqual(
            result["correctness"],
            "sum=1, detail=/tmp/vector.npz, detail_sha=sha5",
        )
        self.assertEqual(result["correctness_variant_count"], "5")
        self.assertEqual(result["correctness_consistency"], "varied_across_samples")

    def test_empty_correctness_is_not_invented_as_pass(self):
        rows = [sample(i, i) for i in range(1, 6)]
        for row in rows:
            row["correctness"] = ""
        result = aggregate_sample_rows(rows, expected_samples=5)[0]

        self.assertEqual(result["correctness"], "")
        self.assertEqual(result["correctness_consistency"], "not_reported")


if __name__ == "__main__":
    unittest.main()
