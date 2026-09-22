import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_full_baselines.py"
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("eggpu_test_gunrock_adapters", MODULE_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class GunrockSemanticAdapterTests(unittest.TestCase):
    def test_pagerank_forwards_the_aligned_parameters(self):
        commands = []

        def fake_run_cmd(command, *_args, **_kwargs):
            commands.append([str(token) for token in command])
            return 0, 0.01, {}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "graph.txt"
            graph.write_text("0 1\n1 0\n")
            log_dir = root / "logs"
            log_dir.mkdir()
            with mock.patch.object(RUNNER, "find_gunrock_exe", return_value=Path("/tmp/pr")), mock.patch.object(
                RUNNER, "write_matrix_market"
            ), mock.patch.object(RUNNER, "run_cmd", side_effect=fake_run_cmd), mock.patch.object(
                RUNNER, "read_text", return_value="GPU Elapsed Time : 1.0 (ms)"
            ), mock.patch.object(
                RUNNER,
                "ensure_gunrock_pagerank_detail",
                return_value=("detail=/tmp/pr.npz, detail_kind=vector", "validated"),
            ):
                rows = []
                RUNNER.maybe_run_gunrock_pr(
                    rows,
                    "small",
                    "directed",
                    "toy",
                    graph,
                    log_dir,
                    {},
                    1,
                    0.0,
                    0.75,
                    1.0e-6,
                )

        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[command.index("--alpha") + 1], "0.75")
        self.assertEqual(command[command.index("--tol") + 1], "1e-06")
        self.assertFalse(any("parameter contract is not aligned" in row["notes"] for row in rows))

    def test_dijkstra_uses_one_source_and_the_maintained_sssp_executable(self):
        commands = []

        def fake_run_cmd(command, *_args, **_kwargs):
            commands.append([str(token) for token in command])
            return 0, 0.01, {}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "graph.txt"
            graph.write_text("0 1\n1 2\n")
            log_dir = root / "logs"
            log_dir.mkdir()
            with mock.patch.object(RUNNER, "find_gunrock_exe", return_value=Path("/tmp/sssp")), mock.patch.object(
                RUNNER, "write_sssp_weighted_matrix_market"
            ), mock.patch.object(RUNNER, "matrix_market_n", return_value=3), mock.patch.object(
                RUNNER, "run_cmd", side_effect=fake_run_cmd
            ), mock.patch.object(
                RUNNER, "read_text", return_value="GPU Elapsed Time : 1.0 (ms)"
            ), mock.patch.object(
                RUNNER,
                "ensure_gunrock_path_detail",
                return_value=(
                    "detail=/tmp/dijkstra.npz, detail_kind=sssp",
                    "separate unmeasured validation",
                ),
            ):
                rows = []
                RUNNER.maybe_run_gunrock_sssp(
                    rows,
                    "small",
                    "directed",
                    "toy",
                    graph,
                    log_dir,
                    {},
                    0.0,
                    8,
                    function="Dijkstra",
                )

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][0], "/tmp/sssp")
        self.assertNotIn("--validate", commands[0])
        e2e = next(row for row in rows if row["metric"] == "e2e")
        self.assertEqual(e2e["function"], "Dijkstra")
        self.assertIn("sources=1", e2e["correctness"])
        self.assertIn("detail_kind=sssp", e2e["correctness"])
        self.assertIn("implementation alias", e2e["notes"])
        self.assertIn("CPU validation excluded", e2e["notes"])
        self.assertTrue(all(row["function"] == "Dijkstra" for row in rows))

    def test_bellman_ford_alias_is_explicitly_limited_to_nonnegative_weights(self):
        commands = []

        def fake_run_cmd(command, *_args, **_kwargs):
            commands.append([str(token) for token in command])
            return 0, 0.01, {}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "graph.txt"
            graph.write_text("0 1\n1 2\n")
            log_dir = root / "logs"
            log_dir.mkdir()
            with mock.patch.object(RUNNER, "find_gunrock_exe", return_value=Path("/tmp/sssp")), mock.patch.object(
                RUNNER, "write_sssp_weighted_matrix_market"
            ), mock.patch.object(RUNNER, "matrix_market_n", return_value=3), mock.patch.object(
                RUNNER, "deterministic_sources", return_value=[0, 1]
            ), mock.patch.object(RUNNER, "run_cmd", side_effect=fake_run_cmd), mock.patch.object(
                RUNNER, "read_text", return_value="GPU Elapsed Time : 1.0 (ms)"
            ), mock.patch.object(
                RUNNER,
                "ensure_gunrock_path_detail",
                return_value=(
                    "detail=/tmp/bellmanford.npz, detail_kind=sssp",
                    "separate unmeasured validation",
                ),
            ):
                rows = []
                RUNNER.maybe_run_gunrock_sssp(
                    rows, "small", "directed", "toy", graph, log_dir, {}, 0.0, 8,
                    function="BellmanFord",
                )

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][commands[0].index("-s") + 1], "0,1")
        self.assertNotIn("--validate", commands[0])
        e2e = next(row for row in rows if row["metric"] == "e2e")
        self.assertIn("conditional implementation alias", e2e["notes"])
        self.assertIn("does not establish negative-edge support", e2e["notes"])
        self.assertTrue(all(row["function"] == "BellmanFord" for row in rows))

    def test_kcore_excludes_cpu_validation_from_timing(self):
        commands = []

        def fake_run_cmd(command, *_args, **_kwargs):
            commands.append([str(token) for token in command])
            return 0, 0.01, {}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "graph.txt"
            graph.write_text("0 1\n1 2\n")
            log_dir = root / "logs"
            log_dir.mkdir()
            with mock.patch.object(
                RUNNER, "find_gunrock_exe", return_value=Path("/tmp/kcore")
            ), mock.patch.object(
                RUNNER, "write_matrix_market"
            ), mock.patch.object(
                RUNNER, "run_cmd", side_effect=fake_run_cmd
            ), mock.patch.object(
                RUNNER, "read_text", return_value="GPU Elapsed Time : 1.0 (ms)"
            ), mock.patch.object(
                RUNNER,
                "ensure_gunrock_kcore_detail",
                return_value=(
                    "detail=/tmp/kcore.npz, detail_kind=vector",
                    "separate unmeasured validation",
                ),
            ) as detail_probe:
                rows = []
                RUNNER.maybe_run_gunrock_kcore(
                    rows, "small", "directed", "toy", graph, log_dir, {}, 0.0
                )

        self.assertEqual(len(commands), 1)
        self.assertIn("--no-validate", commands[0])
        detail_probe.assert_called_once()
        e2e = next(row for row in rows if row["metric"] == "e2e")
        self.assertIn("detail_kind=vector", e2e["correctness"])
        self.assertIn("CPU validation excluded", e2e["notes"])

    def test_lcc_uses_an_unmeasured_full_vector_validation_probe(self):
        commands = []

        def fake_run_cmd(command, *_args, **_kwargs):
            commands.append([str(token) for token in command])
            return 0, 0.01, {}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "graph.txt"
            graph.write_text("0 1\n1 2\n2 0\n")
            log_dir = root / "logs"
            log_dir.mkdir()
            with mock.patch.object(RUNNER, "find_gunrock_exe", return_value=Path("/tmp/lcc")), mock.patch.object(
                RUNNER, "write_matrix_market"
            ), mock.patch.object(RUNNER, "run_cmd", side_effect=fake_run_cmd), mock.patch.object(
                RUNNER, "read_text", return_value="GPU Elapsed Time : 1.0 (ms)"
            ), mock.patch.object(
                RUNNER,
                "ensure_gunrock_lcc_detail",
                return_value=(
                    "vertices=3, mean=1, detail=/tmp/lcc.npz, detail_kind=vector",
                    "full LCC vector is exported by a separate unmeasured validation probe",
                ),
            ) as detail_probe:
                rows = []
                RUNNER.maybe_run_gunrock_lcc(
                    rows,
                    "small",
                    "undirected",
                    "toy",
                    graph,
                    log_dir,
                    {},
                    0.0,
                )

        self.assertEqual(len(commands), 1)
        detail_probe.assert_called_once()
        e2e = next(row for row in rows if row["metric"] == "e2e")
        self.assertIn("vertices=3", e2e["correctness"])
        self.assertIn("detail_kind=vector", e2e["correctness"])
        self.assertIn("separate unmeasured", e2e["notes"])

    def test_bc_uses_one_process_for_the_exact_source_list(self):
        commands = []

        def fake_run_cmd(command, *_args, **_kwargs):
            commands.append([str(token) for token in command])
            return 0, 0.02, {}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "graph.txt"
            graph.write_text("0 1\n1 2\n2 0\n")
            log_dir = root / "logs"
            log_dir.mkdir()
            with mock.patch.object(RUNNER, "find_gunrock_exe", return_value=Path("/tmp/bc")), mock.patch.object(
                RUNNER, "write_matrix_market"
            ), mock.patch.object(RUNNER, "matrix_market_n", return_value=3), mock.patch.object(
                RUNNER, "deterministic_sources", return_value=[0, 2]
            ), mock.patch.object(RUNNER, "run_cmd", side_effect=fake_run_cmd), mock.patch.object(
                RUNNER, "read_text", return_value="GPU Elapsed Time : 2.0 (ms)"
            ), mock.patch.object(
                RUNNER,
                "ensure_gunrock_bc_detail",
                return_value=(
                    "nodes=3, sum=1, detail=/tmp/bc.npz, detail_kind=vector",
                    "full specified-source vector validated",
                ),
            ) as detail_probe:
                rows = []
                RUNNER.maybe_run_gunrock_bc(
                    rows, "small", "directed", "toy", graph, log_dir, {}, 0.0, 2
                )

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][commands[0].index("-s") + 1], "0,2")
        detail_probe.assert_called_once()
        e2e = next(row for row in rows if row["metric"] == "e2e")
        self.assertIn("one Gunrock process", e2e["notes"])
        self.assertIn("detail_kind=vector", e2e["correctness"])


if __name__ == "__main__":
    unittest.main()
