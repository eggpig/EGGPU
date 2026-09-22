import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


BENCHMARKING = Path(__file__).resolve().parents[1]
EVALUATION = BENCHMARKING.parent
SPEC = importlib.util.spec_from_file_location(
    "eggpu_second_supplement_artifacts",
    BENCHMARKING / "generate_second_supplement_artifacts.py",
)
ARTIFACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARTIFACTS)
SCALING_SPEC = importlib.util.spec_from_file_location(
    "eggpu_scaling_protocol",
    BENCHMARKING / "run_eggpu_scaling.py",
)
SCALING = importlib.util.module_from_spec(SCALING_SPEC)
SCALING_SPEC.loader.exec_module(SCALING)


class SecondSupplementProtocolTests(unittest.TestCase):
    def test_large_matrix_function_scope_is_explicit(self):
        self.assertEqual(len(SCALING.ALL_FUNCTIONS), 16)
        self.assertEqual(
            SCALING.SCALING_FUNCTIONS,
            ("PageRank", "WCC", "BFS", "KCore"),
        )
        self.assertEqual(
            SCALING.PROJECTED_FUNCTIONS,
            {"MST", "LCC", "KCore"},
        )

    def test_directed_projection_limit_is_not_silently_skipped(self):
        metadata = {"directed": True, "num_entries": 1_468_364_884}
        for function in SCALING.PROJECTED_FUNCTIONS:
            applicable, reason = SCALING.function_applicability(metadata, function)
            self.assertFalse(applicable)
            self.assertIn("undirected-projection", reason)
            self.assertEqual(
                SCALING.skip_kind(function, metadata, reason),
                "representation_limit",
            )

    def test_weighted_functions_resolve_sibling_manifest(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            plain = root / "toy.json"
            weighted = root / "toy.weighted.json"
            plain.write_text(json.dumps({"name": "toy"}))
            weighted.write_text(
                json.dumps({"name": "toy", "weights_path": "toy.weights.f64"})
            )
            self.assertEqual(
                SCALING.manifest_for_function(plain, "Dijkstra"), weighted
            )
            self.assertEqual(
                SCALING.manifest_for_function(plain, "PageRank"), plain
            )

    def test_pagerank_repeat_validation_is_numeric_not_bitwise(self):
        common = {
            "shape": [4],
            "finite_count": 4,
            "nan_count": 0,
            "inf_count": 0,
            "zero_count": 0,
            "sum": 1.0,
            "minimum": 0.1,
            "maximum": 0.4,
        }
        first = {
            **common,
            "sha256": "a" * 64,
            "validation_sample": [0.1, 0.2, 0.3, 0.4],
        }
        rounded = {
            **common,
            "sha256": "b" * 64,
            "validation_sample": [0.100000001, 0.199999999, 0.3, 0.4],
        }
        self.assertTrue(
            SCALING.result_digests_equivalent("PageRank", first, rounded)
        )

    def test_discrete_repeat_validation_remains_exact(self):
        first = {"partition_sha256": "a" * 64}
        changed = {"partition_sha256": "b" * 64}
        self.assertFalse(SCALING.result_digests_equivalent("WCC", first, changed))

    def test_balanced_ten_covers_direction_and_three_size_bands(self):
        protocol = json.loads(
            (BENCHMARKING / "second_supplement_protocol_20260715.json").read_text()
        )
        balanced = protocol["balanced_10"]
        self.assertEqual(len(balanced), 10)
        self.assertEqual({row["size"] for row in balanced}, {"small", "medium", "large"})
        self.assertEqual(
            {row["graph_type"] for row in balanced}, {"directed", "undirected"}
        )

    def test_protocol_matches_real_graph_scaling_scope(self):
        protocol = json.loads(
            (BENCHMARKING / "second_supplement_protocol_20260715.json").read_text()
        )
        self.assertEqual(protocol["controlled_rmat"]["status"], "enabled_scaling_only")
        self.assertEqual(protocol["controlled_rmat"]["scales"], [20, 22, 24, 26])
        self.assertEqual(protocol["controlled_rmat"]["edge_factor"], 16)
        self.assertEqual(
            protocol["controlled_rmat"]["functions"], ["PageRank", "WCC", "BFS"]
        )
        self.assertEqual(
            protocol["main_12"]["cross_library_graphs"]
            + protocol["main_12"]["real_scale_anchors"],
            12,
        )
        self.assertEqual(
            protocol["scaling"]["function_applicability"]["KCore"],
            "undirected_projection_only",
        )
        self.assertEqual(protocol["scaling"]["native_csr_bridge"], "com-youtube")
        self.assertEqual(
            protocol["statistics"]["primary_estimator"]["EGGPU"],
            "best_observed_of_five_repeats",
        )
        self.assertGreaterEqual(
            protocol["native_csr_acceptance"][
                "minimum_median_geomean_e2e_speedup"
            ],
            1.02,
        )
        self.assertEqual(
            tuple(protocol["cold_start"]["common_functions"]),
            ARTIFACTS.COLD_FUNCTIONS,
        )
        self.assertEqual(
            tuple(protocol["cold_start"]["baselines"]),
            ARTIFACTS.COLD_BASELINES,
        )
        self.assertEqual(
            protocol["cold_start"]["timing_repeat"], ARTIFACTS.COLD_REPEAT
        )

    def test_baseline_scaling_estimator_is_locked(self):
        frame = pd.DataFrame(
            [
                {
                    "dataset": "toy",
                    "function": "WCC",
                    "baseline": "EGGPU",
                    "status": "ok",
                    "mean_seconds": 2.0,
                    "min_seconds": 1.5,
                    "std_seconds": 0.2,
                },
                {
                    "dataset": "toy",
                    "function": "WCC",
                    "baseline": "igraph",
                    "status": "ok",
                    "mean_seconds": 3.0,
                    "min_seconds": 2.5,
                    "std_seconds": 0.3,
                },
            ]
        )
        old_balanced = ARTIFACTS.BALANCED_10
        ARTIFACTS.BALANCED_10 = ["toy"]
        try:
            rows = ARTIFACTS.baseline_scaling_rows(
                frame,
                {
                    "toy": {
                        "nodes_raw": 10,
                        "graph_type": "undirected",
                        "edges_undirected_unique": 20,
                        "edges_directed_unique": 40,
                    }
                },
                pd.DataFrame(
                    [
                        {
                            "dataset": "toy",
                            "function": "WCC",
                            "baseline": baseline,
                            "validation_status": "pass",
                        }
                        for baseline in ("EGGPU", "igraph")
                    ]
                ),
            )
        finally:
            ARTIFACTS.BALANCED_10 = old_balanced
        by_baseline = {row["baseline"]: row for row in rows}
        self.assertEqual(by_baseline["EGGPU"]["reported_e2e_seconds"], 1.5)
        self.assertEqual(by_baseline["igraph"]["reported_e2e_seconds"], 3.0)

    def test_cold_start_estimator_is_locked(self):
        eggpu, eggpu_estimator = ARTIFACTS.reported_seconds(
            {
                "baseline": "EGGPU",
                "user_cold_total_seconds_best_seconds": 1.5,
                "user_cold_total_seconds_mean_seconds": 2.0,
            },
            "user_cold_total_seconds",
        )
        igraph, igraph_estimator = ARTIFACTS.reported_seconds(
            {
                "baseline": "igraph",
                "user_cold_total_seconds_best_seconds": 2.5,
                "user_cold_total_seconds_mean_seconds": 3.0,
            },
            "user_cold_total_seconds",
        )
        self.assertEqual(eggpu, 1.5)
        self.assertEqual(eggpu_estimator, "best_observed_of_repeats")
        self.assertEqual(igraph, 3.0)
        self.assertEqual(igraph_estimator, "arithmetic_mean")

    def test_scaling_function_name_is_paper_facing(self):
        self.assertEqual(ARTIFACTS.display_function("WCCLabels"), "WCC")
        self.assertEqual(ARTIFACTS.display_function("PageRank"), "PageRank")

    def test_formal_runner_enforces_positive_bulk_csr_gate(self):
        runner = (EVALUATION / "run_second_supplement.sh").read_text()
        self.assertIn("run_bulk_csr_performance_gate.py", runner)
        self.assertIn("--minimum-geomean-speedup", runner)
        self.assertIn("datasets/scaling/csr/com-youtube/com-youtube.json", runner)
        self.assertIn("audit_second_supplement.py", runner)

    def test_native_csr_memory_comparison_excludes_misaligned_bfs_from_plot(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            steady = root / "steady"
            output = root / "output"
            steady.mkdir()
            output.mkdir()
            pd.DataFrame(
                [
                    {
                        "dataset": "com-youtube",
                        "baseline": "EGGPU",
                        "function": "PageRank",
                        "metric": "memory_peak_rss_delta_mb",
                        "status": "ok",
                        "mean_value": 3000.0,
                    },
                    {
                        "dataset": "com-youtube",
                        "baseline": "EGGPU",
                        "function": "PageRank",
                        "metric": "memory_peak_gpu_proc_delta_mb",
                        "status": "ok",
                        "mean_value": 500.0,
                    },
                    {
                        "dataset": "com-youtube",
                        "baseline": "EGGPU",
                        "function": "BFS",
                        "metric": "memory_peak_rss_delta_mb",
                        "status": "ok",
                        "mean_value": 3600.0,
                    },
                ]
            ).to_csv(steady / "results_memory.csv", index=False)
            scaling = [
                {
                    "dataset": "com-youtube",
                    "function": "PageRank",
                    "measurement": "memory",
                    "status": "ok",
                    "memory": {
                        "rss_peak_delta_mb": 200.0,
                        "gpu_proc_peak_delta_mb": 500.0,
                    },
                },
                {
                    "dataset": "com-youtube",
                    "function": "BFS",
                    "measurement": "memory",
                    "status": "ok",
                    "memory": {"rss_peak_delta_mb": 180.0},
                },
            ]
            summary = ARTIFACTS.native_csr_memory_artifacts(
                scaling, steady, output
            )
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["strictly_aligned_rows"], 1)
            rows = pd.read_csv(output / "native_csr_memory_comparison.csv")
            pagerank = rows[rows["function"] == "PageRank"].iloc[0]
            self.assertAlmostEqual(pagerank["host_rss_reduction"], 15.0)
            bfs = rows[rows["function"] == "BFS"].iloc[0]
            self.assertFalse(bool(bfs["strictly_aligned"]))

    def test_controlled_rmat_artifacts_keep_the_synthetic_series_separate(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            scaling = root / "controlled_rmat"
            output = root / "paper"
            scaling.mkdir()
            output.mkdir()
            records = []
            for scale in (20, 22, 24, 26):
                nodes = 1 << scale
                raw_edges = nodes * 16
                entries = int(raw_edges * 0.8)
                for function_index, function in enumerate(("PageRank", "WCC", "BFS"), 1):
                    value_seconds = function_index * nodes / float(1 << 20)
                    records.append(
                        {
                            "status": "ok",
                            "measurement": "timing",
                            "dataset": f"R-MAT-S{scale}-EF16",
                            "function": function,
                            "num_nodes": nodes,
                            "num_entries": entries,
                            "raw_edge_records": raw_edges,
                            "self_loops_removed": 1,
                            "duplicates_removed": raw_edges - entries - 1,
                            "rmat_scale": scale,
                            "rmat_edge_factor": 16,
                            "load_seconds": value_seconds / 4,
                            "load": {"best": value_seconds / 5, "mean": value_seconds / 4, "stdev": 0.01},
                            "first_use_e2e_seconds": value_seconds * 1.2,
                            "first_use_e2e": {"best": value_seconds, "mean": value_seconds * 1.2, "stdev": 0.02},
                            "steady_e2e": {"best": value_seconds / 2, "mean": value_seconds * 0.6, "stdev": 0.01},
                            "steady_kernel": {"best": value_seconds / 3, "mean": value_seconds * 0.4, "stdev": 0.01},
                            "persistent_host_csr_bytes": (nodes + 1 + entries) * 4,
                            "persistent_device_csr_bytes": (nodes + 1 + entries) * 4,
                            "result_validation": {"status": "pass"},
                            "timing_process_samples": 5,
                        }
                    )
                    records.append(
                        {
                            "status": "ok",
                            "measurement": "memory",
                            "dataset": f"R-MAT-S{scale}-EF16",
                            "function": function,
                            "memory": {
                                "gpu_proc_peak_delta_mb": value_seconds * 100,
                                "rss_peak_delta_mb": value_seconds * 10,
                            },
                        }
                    )
            (scaling / "scaling_all.json").write_text(json.dumps(records))
            summary = ARTIFACTS.controlled_rmat_artifacts(scaling, output)
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["rows"], 12)
            self.assertTrue((output / "controlled_rmat_scaling.pdf").exists())
            slopes = pd.read_csv(output / "controlled_rmat_descriptive_slopes.csv")
            self.assertEqual(len(slopes), 9)


if __name__ == "__main__":
    unittest.main()
