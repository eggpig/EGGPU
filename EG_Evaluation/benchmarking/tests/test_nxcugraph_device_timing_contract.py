import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


BENCHMARKING = Path(__file__).resolve().parents[1]
if str(BENCHMARKING) not in sys.path:
    sys.path.insert(0, str(BENCHMARKING))

import measurement_schema
import audit_nxcugraph_strict_main99 as strict_audit
import run_nxcugraph_large_matrix as matrix_runner
from nxcugraph_device_timer import DeviceInterval, NxCugraphDeviceTimer


class _HandleFactory:
    def __init__(self, calls):
        self.calls = calls


def timing_row(e2e, kernel, build):
    return {
        "status": "ok",
        "baseline": "nx-cugraph",
        "dataset": "tiny",
        "function": "PageRank",
        "graph_prepare_seconds": build,
        "e2e": {"samples": [e2e]},
        "kernel": {"samples": [kernel]},
        "validation_outside_timer": True,
        "device_timer_records": [
            {
                "timer": "cuda_events_at_pylibcugraph_backend_boundary",
                "backend_functions": ["pagerank"],
                "backend_interval_count": 1,
                "resource_handle_factory_calls": 2,
            }
        ],
    }


class NxCugraphDeviceTimingContractTest(unittest.TestCase):
    def test_metric_schema_declares_real_device_timer(self):
        row = measurement_schema.describe_metric(
            "kernel", baseline="nx-cugraph"
        )
        self.assertEqual(
            row["timer_kind"], "cuda_event_at_pylibcugraph_backend"
        )
        self.assertEqual(row["measurement_window"], "device_execution")

    def test_backend_and_handle_provenance_are_strict(self):
        timer = object.__new__(NxCugraphDeviceTimer)
        timer._intervals = [DeviceInterval("pagerank", 0.002)]
        timer._handle_factory = _HandleFactory(calls=2)
        timer._interval_resource_handle_start = 0
        timer.validate_against_public_wall(0.01)
        provenance = timer.validate_provenance(
            expected_backends=("pagerank",),
            expected_interval_count=1,
        )
        self.assertEqual(provenance["backend_functions"], ["pagerank"])
        self.assertEqual(provenance["resource_handle_factory_calls"], 2)

        with self.assertRaisesRegex(RuntimeError, "unexpected.*backend"):
            timer.validate_provenance(
                expected_backends=("sssp",),
                expected_interval_count=1,
            )

    def test_five_fresh_processes_aggregate_device_samples(self):
        rows = [
            timing_row(0.010 + index * 0.001, 0.002 + index * 0.0001, 0.5 + index)
            for index in range(5)
        ]
        result = matrix_runner.aggregate_timing_rows(rows)
        self.assertEqual(result["timing_process_samples"], 5)
        self.assertEqual(result["e2e"]["count"], 5)
        self.assertEqual(result["kernel"]["count"], 5)
        self.assertAlmostEqual(result["kernel_best_seconds"], 0.002)
        self.assertAlmostEqual(result["graph_prepare_seconds"], 0.5)
        self.assertNotEqual(
            result["kernel"]["samples"], result["e2e"]["samples"]
        )

    def test_kernel_above_public_e2e_is_rejected(self):
        rows = [timing_row(0.01, 0.02, 0.5) for _ in range(5)]
        with self.assertRaisesRegex(RuntimeError, "not in"):
            matrix_runner.aggregate_timing_rows(rows)

    def test_public_return_boundary_has_no_post_return_device_sync(self):
        source = inspect.getsource(
            matrix_runner.call_with_public_return_boundary
        )
        self.assertNotIn("deviceSynchronize", source)
        self.assertLess(
            source.index("strict_public_call"),
            source.index("time.perf_counter() - started"),
        )
        self.assertLess(
            source.index("time.perf_counter() - started"),
            source.index("validate_strict_result"),
        )

    def test_production_runner_uses_native_graph_and_three_warmups(self):
        library_source = (
            BENCHMARKING / "library_baselines.py"
        ).read_text(encoding="utf-8")
        full_source = (
            BENCHMARKING / "run_full_baselines.py"
        ).read_text(encoding="utf-8")
        self.assertIn("build_nxcugraph_native(", library_source)
        self.assertIn("--nx-cugraph-warmup", library_source)
        self.assertIn("default=3", library_source)
        self.assertIn("--nx-cugraph-warmup", full_source)
        self.assertNotIn(
            "kernel uses algorithm timer (backend does not expose kernel)",
            library_source,
        )

    def test_large_worker_records_exclusive_gpu_snapshots(self):
        stale = {
            "exclusive": False,
            "compute_processes": [],
            "memory_used_mb": 8.0,
            "utilization_percent": 10.0,
        }
        idle = {
            "exclusive": True,
            "compute_processes": [],
            "memory_used_mb": 4.0,
            "utilization_percent": 0.0,
        }
        with mock.patch.object(
            matrix_runner,
            "collect_gpu_exclusive_snapshot",
            side_effect=[stale, idle],
        ), mock.patch.object(matrix_runner.time, "sleep"):
            snapshot = matrix_runner.wait_for_gpu_exclusive_snapshot(
                0, attempts=2
            )
        self.assertTrue(snapshot["exclusive"])
        self.assertEqual(snapshot["attempt"], 2)

        source = inspect.getsource(matrix_runner.run_worker)
        self.assertIn("gpu_exclusive_snapshot_before_worker", source)
        self.assertIn("gpu_exclusive_snapshot_after_worker", source)

    def test_moderate_worker_persists_idle_gate_snapshots(self):
        full_source = (
            BENCHMARKING / "run_full_baselines.py"
        ).read_text(encoding="utf-8")
        self.assertIn("gpu_exclusive_snapshot_before_worker", full_source)
        self.assertIn("gpu_exclusive_snapshot_after_worker", full_source)
        self.assertIn("f.write(preflight_idle_note", full_source)

    def test_long_form_sample_export_preserves_aligned_triplets(self):
        provenance_samples = [
            {
                "sample_index": index,
                "timing_provenance": {"sample": index},
                "gpu_exclusive_snapshot_before_worker": "before",
                "gpu_exclusive_snapshot_after_worker": "after",
            }
            for index in range(1, 6)
        ]
        rows = strict_audit.flatten_sample_results(
            [
                {
                    "dataset": "tiny",
                    "function": "PageRank",
                    "baseline": "nx-cugraph",
                    "source": "test",
                    "validation_status": "pass",
                    "construction": {"samples": [1, 2, 3, 4, 5]},
                    "e2e": {"samples": [6, 7, 8, 9, 10]},
                    "processing": {"samples": [0.1, 0.2, 0.3, 0.4, 0.5]},
                    "timing_provenance": {"samples": provenance_samples},
                }
            ]
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[2]["sample_index"], 3)
        self.assertEqual(rows[2]["construction_seconds"], 3)
        self.assertEqual(rows[2]["e2e_seconds"], 8)
        self.assertEqual(rows[2]["processing_seconds"], 0.3)

    def test_large_audit_reads_backend_contract_from_raw_worker(self):
        worker_source = inspect.getsource(matrix_runner.worker)
        self.assertIn("configure_networkx_backend_contract()", worker_source)
        self.assertIn('"prepared_native_graph": True', worker_source)
        contract_source = inspect.getsource(
            matrix_runner.configure_networkx_backend_contract
        )
        self.assertIn("nx.config.cache_converted_graphs = True", contract_source)
        self.assertIn("nx.config.fallback_to_nx = False", contract_source)

        audit_source = inspect.getsource(strict_audit.audit_large)
        self.assertIn(
            'process_record.get(\n                    "networkx_cache_converted_graphs"',
            audit_source,
        )
        self.assertIn(
            'process_record.get(\n                    "networkx_fallback_to_nx"',
            audit_source,
        )

    def test_extreme_variance_gate_is_full_cell_based(self):
        rows = []
        for metric in ("build", "e2e", "kernel"):
            values = [1.0, 1.0, 1.0, 1.0, 1.0]
            if metric == "e2e":
                values[-1] = 25.0
            rows.extend(
                {
                    "metric": metric,
                    "seconds": value,
                }
                for value in values
            )
        self.assertEqual(
            strict_audit.extreme_metrics(rows),
            {"e2e": 25.0},
        )

    def test_semantic_validation_gate_covers_all_eight_adapters(self):
        records = []
        for function in strict_audit.FUNCTIONS:
            records.append(
                {
                    "function": function,
                    "status": "pass",
                    "device_not_above_e2e": True,
                    "timer_provenance": {
                        "backend_functions": list(
                            strict_audit.BACKENDS[function]
                        ),
                        "backend_interval_count": 1,
                        "resource_handle_factory_calls": 2,
                    },
                }
            )
        report = {
            "status": "pass",
            "networkx_cache_converted_graphs": True,
            "networkx_fallback_to_nx": False,
            "validation_reference": "NetworkX CPU",
            "records": records,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            errors = []
            evidence = []
            strict_audit.audit_semantic_validation(
                path, errors, evidence
            )
        self.assertEqual(errors, [])
        self.assertEqual(evidence[0]["role"], "cpu_semantic_cross_validation")


if __name__ == "__main__":
    unittest.main()
