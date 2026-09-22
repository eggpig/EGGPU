import importlib.util
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


BENCHMARKING = Path(__file__).resolve().parents[1]
if str(BENCHMARKING) not in sys.path:
    sys.path.insert(0, str(BENCHMARKING))

import library_baselines
from library_baselines import PeakMemoryMonitor
from measurement_schema import describe_metric, schema_document


ABLATION_SPEC = importlib.util.spec_from_file_location(
    "eggpu_test_ablation_runner", BENCHMARKING / "run_eggpu_ablations.py"
)
ABLATION = importlib.util.module_from_spec(ABLATION_SPEC)
ABLATION_SPEC.loader.exec_module(ABLATION)

WORKFLOW_SPEC = importlib.util.spec_from_file_location(
    "eggpu_test_workflow_runner", BENCHMARKING / "run_eggpu_workflow_reuse.py"
)
WORKFLOW = importlib.util.module_from_spec(WORKFLOW_SPEC)
WORKFLOW_SPEC.loader.exec_module(WORKFLOW)


class MeasurementSchemaTests(unittest.TestCase):
    def test_timing_mode_does_not_construct_memory_monitor(self):
        prior = library_baselines.MEASUREMENT_MODE
        library_baselines.MEASUREMENT_MODE = "timing"
        try:
            with mock.patch.object(
                library_baselines,
                "PeakMemoryMonitor",
                side_effect=AssertionError("monitor must not be constructed"),
            ):
                value, elapsed, memory = library_baselines.timed_algorithm(
                    lambda: 7, sync_after=False
                )
            self.assertEqual(value, 7)
            self.assertGreaterEqual(elapsed, 0.0)
            self.assertIsNone(memory)
        finally:
            library_baselines.MEASUREMENT_MODE = prior

    def test_primary_timer_and_memory_provenance_are_explicit(self):
        self.assertEqual(describe_metric("kernel", "EGGPU")["timer_kind"], "cuda_event")
        self.assertEqual(
            describe_metric("kernel", "networkx")["timer_kind"],
            "algorithm_wall_time_surrogate",
        )
        gunrock_e2e = describe_metric("e2e", "Gunrock")
        self.assertEqual(gunrock_e2e["measurement_scope"], "external_cli_invocation")
        self.assertEqual(gunrock_e2e["measurement_window"], "external_cli_process")
        self.assertNotEqual(
            gunrock_e2e["measurement_scope"],
            describe_metric("e2e", "igraph")["measurement_scope"],
        )
        proc = describe_metric("memory_peak_gpu_proc_mb", "EGGPU")
        self.assertEqual(proc["unit"], "MiB")
        self.assertEqual(proc["measurement_scope"], "benchmark_process_tree_gpu_memory")
        samples = describe_metric("memory_monitor_gpu_samples", "EGGPU")
        self.assertEqual(samples["unit"], "count")
        self.assertEqual(samples["measurement_window"], "algorithm_call")
        validation = describe_metric("layout_validation_relative_error", "EGGPU")
        self.assertEqual(validation["unit"], "ratio")
        self.assertEqual(validation["metric_family"], "correctness")
        self.assertEqual(schema_document()["primary_metrics"]["gpu_memory"], "memory_peak_gpu_proc_mb")
        self.assertIn("split_measurement_protocol", schema_document())

    def test_memory_monitor_reports_attribution_and_sampling_quality(self):
        monitor = PeakMemoryMonitor(poll_seconds=0.002).start()
        time.sleep(0.006)
        result = monitor.stop()
        for field in (
            "rss_mb",
            "rss_start_mb",
            "rss_peak_delta_mb",
            "monitor_rss_samples",
            "monitor_poll_ms",
            "monitor_window_seconds",
        ):
            self.assertIn(field, result)
            self.assertIsNotNone(result[field])
        self.assertGreaterEqual(result["monitor_rss_samples"], 2)
        self.assertGreater(result["monitor_window_seconds"], 0.0)


class AblationSemanticTests(unittest.TestCase):
    def test_workflow_wrapper_supplies_current_ablation_contract(self):
        captured = {}

        def fake_run_workflow(args):
            captured.update(vars(args))
            return []

        args = types.SimpleNamespace(
            repeat=5,
            warmup=2,
            gpu=0,
            sssp_sources=8,
            bc_sources=16,
            closeness_sources=16,
        )
        with mock.patch.object(
            WORKFLOW.ablation_runner, "run_workflow", side_effect=fake_run_workflow
        ):
            WORKFLOW.run_dataset(
                args, "small", "undirected", "g", "g.txt", ["PageRank"]
            )

        self.assertEqual(captured["closeness_sources"], 16)
        self.assertEqual(captured["workflow_order_id"], "canonical")
        self.assertEqual(captured["measurement_mode"], "timing")

    def test_dijkstra_uses_one_source_while_sssp_uses_all_sources(self):
        calls = []

        def multi_source_dijkstra(graph, sources, **kwargs):
            calls.append(list(sources))
            return {source: {} for source in sources}

        fake_easygraph = types.SimpleNamespace(multi_source_dijkstra=multi_source_dijkstra)
        bundle = {"graph": object(), "sssp_sources": [0, 4, 8]}
        with mock.patch.dict(sys.modules, {"easygraph": fake_easygraph}):
            ABLATION.call_function("Dijkstra", bundle)
            ABLATION.call_function("SSSP", bundle)

        self.assertEqual(calls[0], [0])
        self.assertEqual(calls[1], [0, 4, 8])

    def test_return_claim_is_only_valid_for_on_demand_views(self):
        dense_type = type("_DenseValueDict", (), {})
        representation, on_demand, _ = ABLATION.result_representation(dense_type())
        self.assertEqual(representation, "on_demand_dense_mapping")
        self.assertTrue(on_demand)

        representation, on_demand, _ = ABLATION.result_representation([1.0, 2.0])
        self.assertEqual(representation, "materialized_sequence")
        self.assertFalse(on_demand)

    def test_numpy_return_is_materialized_and_counted(self):
        result = np.asarray([1, 2, 3], dtype=np.int32)
        representation, on_demand, _ = ABLATION.result_representation(result)
        elapsed, count, checksum = ABLATION.materialize_result("KCore", result)

        self.assertEqual(representation, "materialized_array")
        self.assertFalse(on_demand)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(count, 3)
        self.assertEqual(checksum, 6.0)

    def test_standard_container_for_batched_distances_is_builtin_and_equivalent(self):
        result = {
            "s0": {"u": 1.0, "v": 2.0},
            "s1": {"u": 3.0},
        }
        converted = ABLATION.standard_python_container("SSSP", result)
        self.assertIs(type(converted), dict)
        self.assertTrue(all(type(value) is dict for value in converted.values()))
        _, before_count, before_checksum = ABLATION.materialize_result("SSSP", result)
        _, after_count, after_checksum = ABLATION.materialize_result("SSSP", converted)
        self.assertEqual(before_count, after_count)
        self.assertEqual(before_checksum, after_checksum)

    def test_return_equivalence_treats_matching_nan_as_equal(self):
        deferred = {"finite": 2.5, "undefined": float("nan")}
        standard = dict(deferred)

        _, count, checksum = ABLATION.materialize_result(
            "EffectiveSize", deferred
        )
        profile = ABLATION.numeric_result_profile("EffectiveSize", deferred)
        comparison = ABLATION.compare_result_containers(deferred, standard)

        self.assertEqual(count, 2)
        self.assertEqual(checksum, 2.5)
        self.assertEqual(profile["finite_value_count"], 1)
        self.assertEqual(profile["nan_value_count"], 1)
        self.assertTrue(comparison["equivalent"])
        self.assertEqual(comparison["nan_pair_count"], 1)
        self.assertEqual(comparison["mismatch_count"], 0)

    def test_return_equivalence_rejects_missing_keys_and_infinity_sign(self):
        comparison = ABLATION.compare_result_containers(
            {"a": float("inf"), "b": 1.0},
            {"a": float("-inf"), "c": 1.0},
        )

        self.assertFalse(comparison["equivalent"])
        self.assertEqual(comparison["missing_key_count"], 2)
        self.assertGreaterEqual(comparison["mismatch_count"], 3)

    def test_return_equivalence_recurses_through_distance_mappings(self):
        deferred = {
            "s0": {"u": 1.0, "v": float("inf")},
            "s1": {"u": float("nan")},
        }
        standard = {
            "s1": {"u": float("nan")},
            "s0": {"v": float("inf"), "u": 1.0 + 1.0e-12},
        }
        comparison = ABLATION.compare_result_containers(deferred, standard)

        self.assertTrue(comparison["equivalent"])
        self.assertEqual(comparison["nan_pair_count"], 1)
        self.assertEqual(comparison["infinity_pair_count"], 1)

    def test_undirected_layout_has_two_adjacency_slots_per_edge(self):
        class Column:
            def __init__(self, values):
                self.values = values

            def to_numpy(self, dtype=None, copy=False):
                return np.asarray(self.values, dtype=dtype).copy() if copy else np.asarray(self.values, dtype=dtype)

        undirected = {"src": Column([0, 1]), "dst": Column([1, 2])}
        directed = {"src": Column([0, 1]), "dst": Column([1, 2])}
        views = {"clean": (3, directed, undirected)}
        args = types.SimpleNamespace(graph_type="undirected")

        n, src, dst = ABLATION.build_layout_arrays(args, views)

        self.assertEqual(n, 3)
        self.assertEqual(list(zip(src.tolist(), dst.tolist())), [(0, 1), (1, 2), (1, 0), (2, 1)])


if __name__ == "__main__":
    unittest.main()
