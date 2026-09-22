import importlib.util
import json
import statistics
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_final_13_failure_ledger.py"
SPEC = importlib.util.spec_from_file_location("final_13_failure_ledger", MODULE_PATH)
LEDGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEDGER)


class Final13FailureLedgerContractTests(unittest.TestCase):
    def test_nxcugraph_scc_is_unsupported_in_strict_dispatch(self):
        self.assertEqual(LEDGER.SUPPORT["nx-cugraph"]["SCC"], LEDGER.F)

    def test_gunrock_dijkstra_is_output_equivalent_sssp_alias(self):
        self.assertEqual(LEDGER.SUPPORT["Gunrock"]["Dijkstra"], LEDGER.T)

    def test_sygraph_failed_validation_is_not_support(self):
        self.assertEqual(
            {LEDGER.SUPPORT["SYgraph"][function] for function in LEDGER.FUNCTIONS},
            {LEDGER.F},
        )

    def test_common_unsupported_messages_are_classified(self):
        examples = (
            "no matching Gunrock executable",
            "backend has no native implementation",
            "artifact does not include this function",
            "runner would route to CUDA fallback",
        )
        for message in examples:
            with self.subTest(message=message):
                self.assertEqual(
                    LEDGER.classify_non_success_text(message),
                    "unsupported_api",
                )

    def test_f_support_cannot_retain_numeric_performance(self):
        row = {
            "support_class": LEDGER.F,
            "execution_status": "ok",
            "reason": "historical runner-side derivation",
            "e2e_paper_seconds": 1.0,
            "kernel_paper_seconds": 0.5,
            "gpu_peak_mb_mean": 12.0,
            "build_paper_seconds": 3.0,
        }
        result = LEDGER.enforce_support_contract(row)
        self.assertEqual(result["execution_status"], "unsupported_api")
        self.assertEqual(result["excluded_observed_status"], "ok")
        self.assertIsNone(result["e2e_paper_seconds"])
        self.assertIsNone(result["kernel_paper_seconds"])
        self.assertIsNone(result["gpu_peak_mb_mean"])
        self.assertEqual(result["build_paper_seconds"], 3.0)

    def test_t_or_p_support_is_not_rewritten(self):
        row = {
            "support_class": LEDGER.P,
            "execution_status": "ok",
            "e2e_paper_seconds": 1.0,
        }
        self.assertIs(LEDGER.enforce_support_contract(row), row)
        self.assertEqual(row["e2e_paper_seconds"], 1.0)

    def test_main_memory_prefers_absolute_process_peaks(self):
        rows = []
        for metric, value in (
            ("memory_peak_gpu_proc_mb", 500.0),
            ("memory_peak_gpu_proc_delta_mb", 450.0),
            ("memory_peak_rss_mb", 1200.0),
            ("memory_peak_rss_delta_mb", 1100.0),
        ):
            rows.append(
                {
                    "dataset": "g",
                    "function": "PageRank",
                    "baseline": "EGGPU",
                    "status": "ok",
                    "metric": metric,
                    "mean_value": value,
                    "std_value": 2.0,
                    "sample_count": 3,
                }
            )
        result = LEDGER.main_memory_record(
            pd.DataFrame(rows), "g", "PageRank", "EGGPU"
        )
        self.assertEqual(result["gpu_peak_mb_mean"], 500.0)
        self.assertEqual(result["host_rss_peak_mb_mean"], 1200.0)
        self.assertEqual(result["memory_sample_count"], 3)

    def test_scaling_csv_exports_absolute_rss_peak(self):
        source = (
            Path(__file__).resolve().parents[1] / "run_eggpu_scaling.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"rss_peak_mb"', source)
        self.assertIn('"rss_peak_mb": memory.get("rss_mb")', source)

    def test_large_eggpu_build_uses_best_of_five(self):
        row = {
            "dataset": "g",
            "function": "PageRank",
            "measurement": "timing",
            "status": "ok",
            "timing_process_samples": 5,
            "result_validation": "pass",
            "load_seconds": 2.0,
            "load_stdev_seconds": 0.2,
            "steady_e2e_mean": 1.0,
            "steady_e2e_stdev": 0.1,
            "steady_kernel_mean": 0.5,
            "steady_kernel_stdev": 0.05,
        }
        with tempfile.TemporaryDirectory() as temporary:
            core = Path(temporary)
            raw = core / "eggpu_large_matrix" / "raw"
            raw.mkdir(parents=True)
            (raw / "g_PageRank_timing.json").write_text(
                json.dumps(
                    {
                        "load": {"best": 1.5},
                        "steady_e2e": {"best": 0.8},
                        "steady_kernel": {"best": 0.4},
                    }
                ),
                encoding="utf-8",
            )
            result = LEDGER.large_eggpu_record(
                pd.DataFrame([row]), core, "g", "PageRank"
            )
        self.assertEqual(result["build_paper_seconds"], 1.5)
        self.assertEqual(result["build_raw_mean_seconds"], 2.0)
        self.assertEqual(result["build_estimator"], "minimum_of_five_bulk_csr_load")

    def test_closeness_supplement_loads_and_aggregates_all_build_samples(self):
        datasets = (
            "ER-100k",
            "com-youtube",
            "soc-Slashdot0811",
            "web-NotreDame",
        )
        baselines = ("easygraph-cpp", "easygraph-cpu", "igraph", "networkx")
        build_rows = []
        algorithm_rows = {"e2e": [], "kernel": []}
        expected_build_values = {}
        for dataset_index, dataset in enumerate(datasets):
            for baseline_index, baseline in enumerate(baselines):
                values = [
                    10.0 * dataset_index + baseline_index + sample_index / 10.0
                    for sample_index in range(1, 6)
                ]
                expected_build_values[(dataset, baseline)] = values
                for sample_index, seconds in enumerate(values, start=1):
                    common = {
                        "dataset": dataset,
                        "function": "Closeness",
                        "baseline": baseline,
                        "status": "ok",
                        "sample_index": sample_index,
                        "is_supplement": True,
                    }
                    build_rows.append(
                        {**common, "metric": "build", "seconds": seconds}
                    )
                    for metric in algorithm_rows:
                        algorithm_rows[metric].append(
                            {**common, "metric": metric, "seconds": seconds / 2.0}
                        )

        sentinel = {
            "dataset": "sentinel-graph",
            "function": "PageRank",
            "baseline": "networkx",
            "metric": "e2e",
            "status": "ok",
            "seconds": 123.456,
            "sample_index": 1,
            "is_supplement": False,
        }
        superseded_build_rows = [
            {
                "dataset": dataset,
                "function": "Closeness",
                "baseline": baseline,
                "metric": "build",
                "status": "ok",
                "seconds": -1.0,
                "sample_index": 1,
                "is_supplement": True,
            }
            for dataset in datasets
            for baseline in baselines
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            main_result = root / "main"
            closeness_result = root / "closeness"
            main_result.mkdir()
            closeness_result.mkdir()
            pd.DataFrame([sentinel, *superseded_build_rows]).to_csv(
                main_result / "results_samples.csv", index=False
            )
            pd.DataFrame(build_rows).to_csv(
                closeness_result / "closeness_large_sampled_build.csv",
                index=False,
            )
            for metric, rows in algorithm_rows.items():
                pd.DataFrame(rows).to_csv(
                    closeness_result / f"closeness_large_sampled_{metric}.csv",
                    index=False,
                )

            samples, validation = LEDGER.load_main_outcome_evidence(
                main_result, closeness_result
            )

        supplement_build = samples[
            samples["metric"].eq("build")
            & samples["function"].eq("Closeness")
        ]
        self.assertEqual(len(supplement_build), 80)
        group_sizes = supplement_build.groupby(["dataset", "baseline"]).size()
        self.assertEqual(len(group_sizes), 16)
        self.assertTrue(group_sizes.eq(5).all())
        self.assertFalse(supplement_build["seconds"].eq(-1.0).any())

        preserved = samples[
            samples["dataset"].eq(sentinel["dataset"])
            & samples["function"].eq(sentinel["function"])
            & samples["baseline"].eq(sentinel["baseline"])
        ]
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved.iloc[0]["seconds"], sentinel["seconds"])
        self.assertEqual(preserved.iloc[0]["metric"], sentinel["metric"])

        for (dataset, baseline), values in expected_build_values.items():
            with self.subTest(dataset=dataset, baseline=baseline):
                outcome = LEDGER.main_outcome_record(
                    samples,
                    validation,
                    dataset,
                    "Closeness",
                    baseline,
                )
                self.assertEqual(
                    outcome["build_paper_seconds"], statistics.mean(values)
                )
                self.assertEqual(
                    outcome["build_raw_mean_seconds"], statistics.mean(values)
                )
                self.assertEqual(
                    outcome["build_std_seconds"], statistics.stdev(values)
                )
                self.assertEqual(
                    outcome["build_estimator"],
                    "arithmetic_mean_of_raw_build_samples",
                )


if __name__ == "__main__":
    unittest.main()
