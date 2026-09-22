import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import psutil


BENCHMARKING = Path(__file__).resolve().parents[1]
REPOSITORY = BENCHMARKING.parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = load_module(
    "graphscope_runner_contract",
    BENCHMARKING / "run_graphscope_baseline.py",
)
PREPARER = load_module(
    "graphscope_preparer_contract",
    BENCHMARKING / "prepare_graphscope_13.py",
)
MERGER = load_module(
    "graphscope_merger_contract",
    BENCHMARKING / "merge_graphscope_into_final_13.py",
)


class GraphScopeBaselineContractTests(unittest.TestCase):
    def test_manifest_sources_override_evenly_spaced_sources(self):
        preferred = [7, 3, 9, 1]
        self.assertEqual(RUNNER.pick_sources(10, 1, preferred), [7])
        self.assertEqual(RUNNER.pick_sources(10, 3, preferred), [7, 3, 9])
        with self.assertRaises(ValueError):
            RUNNER.pick_sources(10, 5, preferred)

    def test_source_metadata_records_manifest_policy_and_hash(self):
        args = type(
            "Args",
            (),
            {
                "sssp_sources": 8,
                "bc_sources": 16,
                "closeness_sources": 16,
                "closeness_sampled_datasets": "GAP-twitter",
            },
        )()
        metadata = {
            "name": "GAP-twitter",
            "num_nodes": 61_578_415,
            "benchmark_sources_zero_based": [
                12_441_072,
                54_488_257,
                25_451_915,
                57_714_473,
                14_839_494,
                32_081_104,
                52_957_357,
                50_444_380,
                49_590_701,
                20_127_816,
                34_939_333,
                48_251_001,
                19_524_253,
                43_676_726,
                33_055_508,
                15_244_687,
            ],
        }
        dijkstra = RUNNER.source_metadata("Dijkstra", metadata, args)
        self.assertEqual(
            dijkstra["source_policy"], "manifest_benchmark_sources_zero_based"
        )
        expected_sha = __import__("hashlib").sha256(
            b"12441072"
        ).hexdigest()[:16]
        self.assertEqual(dijkstra["source_nodes_sha"], expected_sha)
        self.assertEqual(dijkstra["sample_sources"], "1")

    def test_preparer_propagates_frozen_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            source_manifest = temporary_path / "source.json"
            prepared_manifest = temporary_path / "prepared.json"
            source_manifest.write_text(
                json.dumps({"benchmark_sources_zero_based": [5, 2, 8]})
            )
            prepared_manifest.write_text("{}")
            old = PREPARER.CANONICAL_SOURCE_MANIFESTS.get("synthetic")
            PREPARER.CANONICAL_SOURCE_MANIFESTS["synthetic"] = source_manifest
            try:
                metadata = {
                    "name": "synthetic",
                    "num_nodes": 10,
                }
                result = PREPARER.normalize_manifest_paths(
                    prepared_manifest, metadata
                )
            finally:
                if old is None:
                    PREPARER.CANONICAL_SOURCE_MANIFESTS.pop("synthetic", None)
                else:
                    PREPARER.CANONICAL_SOURCE_MANIFESTS["synthetic"] = old
            self.assertEqual(
                result["benchmark_sources_zero_based"], [5, 2, 8]
            )
            persisted = json.loads(prepared_manifest.read_text())
            self.assertEqual(
                persisted["benchmark_sources_zero_based"], [5, 2, 8]
            )

    def test_worker_applies_manifest_sources_to_all_source_driven_functions(self):
        source = (
            BENCHMARKING / "graphscope_baseline_worker.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'preferred_sources = metadata.get("benchmark_sources_zero_based")',
            source,
        )
        self.assertEqual(
            source.count("pick_sources(n, args.sssp_sources, preferred_sources)"),
            1,
        )
        self.assertIn(
            "pick_sources(n, source_count, preferred_sources)", source
        )
        self.assertIn(
            "pick_sources(n, args.bc_sources, preferred_sources)", source
        )
        self.assertIn(
            "pick_sources(n, args.closeness_sources, preferred_sources)", source
        )
        self.assertIn("def project_once(", source)
        self.assertIn('project_once(graph, "weight")', source)
        self.assertIn(
            "flash.betweenness_centrality, application_graph", source
        )

    def test_support_matrix_is_complete_and_stable(self):
        self.assertEqual(RUNNER.FUNCTIONS, MERGER.FUNCTIONS)
        self.assertEqual(set(RUNNER.FUNCTIONS), set(RUNNER.SUPPORT))
        self.assertEqual(
            {function: label for function, (label, _) in RUNNER.SUPPORT.items()},
            {function: label for function, (label, _) in MERGER.SUPPORT.items()},
        )
        counts = {
            label: sum(
                support == label for support, _ in RUNNER.SUPPORT.values()
            )
            for label in ("T", "P", "F")
        }
        self.assertEqual(counts, {"T": 5, "P": 5, "F": 6})

    def test_unsupported_functions_are_never_timed(self):
        expected = {
            function
            for function, (support, _) in RUNNER.SUPPORT.items()
            if support == "F"
        }
        self.assertEqual(set(RUNNER.STATIC_NO_TIMING), expected)

    def test_worker_uses_native_graphscope_apps_without_fallback(self):
        source = (
            BENCHMARKING / "graphscope_baseline_worker.py"
        ).read_text(encoding="utf-8")
        required_native_calls = (
            "graphscope.pagerank_nx",
            "graphscope.clustering",
            "graphscope.wcc_projected",
            "flash.scc",
            "flash.bfs",
            "flash.sssp",
            "flash.kcore_decomposition",
            "flash.betweenness_centrality",
            'algo="closeness_centrality"',
        )
        for call in required_native_calls:
            self.assertIn(call, source)
        self.assertNotIn("import networkx", source)
        self.assertNotIn("nx.", source)
        self.assertNotIn("cugraph", source.lower())

    def test_measurement_protocol_matches_split_main_experiment(self):
        source = (
            BENCHMARKING / "run_graphscope_baseline.py"
        ).read_text(encoding="utf-8")
        self.assertIn('choices=("timing", "memory", "both")', source)
        self.assertIn('default=5', source)
        self.assertIn('default=3', source)
        self.assertIn('"isolated_memory_subprocess"', source)
        self.assertIn('"timing_estimator": "arithmetic_mean"', source)
        self.assertIn('"memory_estimator": "arithmetic_mean"', source)

    def test_dataset_order_is_fixed_from_small_to_billion_edge(self):
        self.assertEqual(RUNNER.DATASET_ORDER[0], "ca-HepTh")
        self.assertEqual(RUNNER.DATASET_ORDER[-1], "GAP-twitter")
        self.assertEqual(len(RUNNER.DATASET_ORDER), 13)

    def test_timeout_cleanup_terminates_detached_graphscope_like_child(self):
        script = (
            "import os, subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', "
            "\"import os,time; os.setsid(); time.sleep(60)\"])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert process.stdout is not None
        child_pid = int(process.stdout.readline().strip())
        time.sleep(0.1)
        RUNNER.terminate_process_tree(process)
        process.stdout.close()
        self.assertIsNotNone(process.poll())
        self.assertFalse(psutil.pid_exists(child_pid))

    def test_failed_worker_cleanup_terminates_observed_detached_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            child_pid_path = temporary_path / "child.pid"
            script = (
                "import os, pathlib, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', "
                "\"import os,time; os.setsid(); time.sleep(60)\"])\n"
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
                "time.sleep(0.25)\n"
                "raise RuntimeError('synthetic worker failure')\n"
            )
            status, payload, _ = RUNNER.run_child(
                [sys.executable, "-c", script],
                temporary_path / "worker.log",
                temporary_path / "worker_result.json",
                dict(os.environ),
                timeout=5,
                monitor_memory=False,
                poll_ms=10,
            )
            child_pid = int(child_pid_path.read_text())
            self.assertEqual(status, "failed")
            self.assertIsNone(payload)
            self.assertFalse(psutil.pid_exists(child_pid))

    def test_current_decision_record_matches_runtime_contract(self):
        decision_path = (
            BENCHMARKING
            / "final_function_support_decisions_20260727_graphscope.json"
        )
        decisions = json.loads(decision_path.read_text(encoding="utf-8"))
        graphscope = decisions["baselines"]["GraphScope"]
        expected = {
            function: support
            for function, (support, _) in RUNNER.SUPPORT.items()
        }
        self.assertEqual(graphscope["statuses"], expected)
        self.assertEqual(graphscope["version"], "0.29.0")

    def test_paper_support_row_and_sygraph_removal(self):
        # The canonical support data is versioned; manuscript exports are local.
        with (BENCHMARKING.parent / "paper_table_function_support.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            baselines = {row["Baseline"] for row in csv.DictReader(stream)}
        self.assertIn("GraphScope", baselines)
        self.assertNotIn("SYgraph", baselines)


if __name__ == "__main__":
    unittest.main()
