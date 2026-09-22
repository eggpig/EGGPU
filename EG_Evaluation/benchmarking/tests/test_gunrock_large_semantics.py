import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


BENCHMARKING = Path(__file__).resolve().parents[1]
if str(BENCHMARKING) not in sys.path:
    sys.path.insert(0, str(BENCHMARKING))
MODULE_PATH = BENCHMARKING / "run_gunrock_large_matrix.py"
SPEC = importlib.util.spec_from_file_location("eggpu_test_gunrock_large", MODULE_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
from pagerank_residual_validation import validate_pagerank_fixed_point


class GunrockLargeSemanticsTests(unittest.TestCase):
    def test_pagerank_fixed_point_validation_uses_graph_equation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.asarray([0, 1, 2, 3], dtype=np.int32).tofile(root / "offsets.i32")
            np.asarray([1, 2, 0], dtype=np.int32).tofile(root / "indices.i32")
            np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float32).tofile(
                root / "pagerank.f32"
            )
            manifest = {
                "num_nodes": 3,
                "num_entries": 3,
                "offsets_path": "offsets.i32",
                "indices_path": "indices.i32",
            }
            (root / "graph.json").write_text(json.dumps(manifest), encoding="utf-8")
            evidence = validate_pagerank_fixed_point(
                root / "pagerank.f32",
                root / "graph.json",
                alpha=0.75,
                chunk_entries=2,
            )
            self.assertEqual(evidence["status"], "pass")
            self.assertLess(evidence["mean_residual"], 1.0e-7)

    def test_total_multi_source_kernel_time_has_precedence(self):
        output = (
            "GPU Elapsed Time : 1.5 (ms)\n"
            "GPU Total Elapsed Time : 7.25 (ms)\n"
        )
        self.assertAlmostEqual(RUNNER.shared.parse_gunrock_elapsed_ms(output), 0.00725)

    def test_pagerank_command_forwards_paper_parameters(self):
        command = RUNNER.command_for(
            Path("/tmp/pr"),
            "PageRank",
            Path("/tmp/graph.mtx"),
            pagerank_alpha=0.75,
            pagerank_tolerance=1.0e-6,
            result_file=Path("/tmp/result.raw"),
        )
        self.assertEqual(command[command.index("--alpha") + 1], "0.75")
        self.assertEqual(command[command.index("--tol") + 1], "1e-06")
        self.assertEqual(command[command.index("--result_file") + 1], "/tmp/result.raw")

    def test_dijkstra_is_an_explicit_single_source_sssp_alias(self):
        self.assertEqual(RUNNER.APP["Dijkstra"], "sssp")
        self.assertEqual(RUNNER.SUPPORT["Dijkstra"], "T")
        command = RUNNER.command_for(
            Path("/tmp/sssp"), "Dijkstra", Path("/tmp/graph.mtx"), source=7
        )
        self.assertEqual(command, [
            "/tmp/sssp", "-m", "/tmp/graph.mtx", "-s", "7"
        ])

    def test_bellman_ford_is_a_conditional_nonnegative_sssp_alias(self):
        self.assertEqual(RUNNER.SUPPORT["BellmanFord"], "P")
        self.assertEqual(RUNNER.APP["BellmanFord"], "sssp")
        self.assertNotIn("BellmanFord", RUNNER.UNSUPPORTED)
        self.assertIn("BellmanFord", RUNNER.MULTISOURCE)
        command = RUNNER.command_for(
            Path("/tmp/sssp"), "BellmanFord", Path("/tmp/graph.mtx"), source=7
        )
        self.assertEqual(command, [
            "/tmp/sssp", "-m", "/tmp/graph.mtx", "-s", "7"
        ])

    def test_bellman_ford_validation_note_does_not_claim_negative_edges(self):
        status, failure_kind, note = RUNNER.classify(
            "BellmanFord",
            [{
                "timed_out": False,
                "returncode": 0,
                "output": (
                    "GPU Elapsed Time : 1.0 (ms)\n"
                    "Aligned E2E Time : 1.5 (ms)\n"
                ),
            }],
            None,
        )
        self.assertEqual(status, "ok")
        self.assertEqual(failure_kind, "")
        self.assertIn("conditional implementation alias", note)
        self.assertIn("does not establish negative-edge support", note)

    def test_validation_command_is_explicit_and_outside_timing(self):
        command = RUNNER.command_for(
            Path("/tmp/sssp"),
            "SSSP",
            Path("/tmp/graph.mtx"),
            source=7,
            validate=True,
            result_file=Path("/tmp/sssp.raw"),
        )
        self.assertIn("--validate", command)
        self.assertEqual(command[command.index("--result_file") + 1], "/tmp/sssp.raw")

    def test_kcore_timing_disables_cpu_validation(self):
        command = RUNNER.command_for(
            Path("/tmp/kcore"), "KCore", Path("/tmp/graph.mtx")
        )
        self.assertEqual(command, ["/tmp/kcore", "/tmp/graph.mtx", "--no-validate"])

    def test_bc_uses_one_multi_source_process_and_optional_result_export(self):
        command = RUNNER.command_for(
            Path("/tmp/bc"),
            "BC",
            Path("/tmp/graph.mtx"),
            source=[0, 7, 11],
            result_file=Path("/tmp/bc.raw"),
        )
        self.assertEqual(command, [
            "/tmp/bc", "-m", "/tmp/graph.mtx", "-s", "0,7,11",
            "--result_file", "/tmp/bc.raw",
        ])
        status, failure_kind, note = RUNNER.classify(
            "BC",
            [{
                "timed_out": False,
                "returncode": 0,
                "output": (
                    "GPU Elapsed Time : 2.0 (ms)\n"
                    "Aligned E2E Time : 2.5 (ms)\n"
                ),
            }],
            None,
        )
        self.assertEqual((status, failure_kind), ("ok", ""))
        self.assertIn("complete specified source list", note)

    def test_external_cli_wall_time_is_not_accepted_as_aligned_e2e(self):
        status, failure_kind, note = RUNNER.classify(
            "BFS",
            [{
                "timed_out": False,
                "returncode": 0,
                "output": "GPU Elapsed Time : 1.0 (ms)\n",
            }],
            None,
        )
        self.assertEqual(status, "failed")
        self.assertEqual(failure_kind, "missing_aligned_e2e_timing")
        self.assertIn("external CLI wall time is not", note)


if __name__ == "__main__":
    unittest.main()
