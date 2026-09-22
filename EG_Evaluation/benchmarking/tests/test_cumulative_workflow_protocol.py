import os
import sys
import tempfile
import time
import unittest
import importlib.machinery
import hashlib
import json
import types
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import run_cumulative_workflow_comparison as cumulative


class CumulativeWorkflowProtocolTests(unittest.TestCase):
    def test_easygraph_repo_is_required(self):
        with mock.patch.object(
            sys,
            "argv",
            ["run_cumulative_workflow_comparison.py", "--output-dir", "/tmp/out"],
        ):
            with self.assertRaises(SystemExit):
                cumulative.parse_args()

    def test_isolated_control_disables_all_cross_call_graph_state(self):
        names = (
            "EASYGRAPH_GPU_DISABLE_GRAPH_CONTEXT_CACHE",
            "EASYGRAPH_GPU_DISABLE_CPP_GRAPH_CACHE",
            "EASYGRAPH_GPU_DISABLE_DEVICE_CSR_CACHE",
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in names:
                os.environ.pop(name, None)
            cumulative.configure_strict_gpu(7, baseline="EGGPU", isolated=True)
            self.assertTrue(all(os.environ[name] == "TRUE" for name in names))

            cumulative.configure_strict_gpu(7, baseline="EGGPU", isolated=False)
            self.assertTrue(all(name not in os.environ for name in names))

    def test_default_workflow_reaches_distance_centrality(self):
        self.assertEqual(
            cumulative.DEFAULT_WORKFLOW,
            ("WCC", "PageRank", "BFS", "SSSP", "Closeness"),
        )
        self.assertIn("Closeness", cumulative.SUPPORTED_WORKFLOW_FUNCTIONS)
        self.assertEqual(cumulative.EGGPU_KERNEL_KEYS["Closeness"], "closeness")

    def test_nx_cugraph_retains_supported_prefix_without_fallback(self):
        prefix, unsupported = cumulative.supported_workflow_prefix(
            "nx-cugraph", cumulative.DEFAULT_WORKFLOW
        )
        self.assertEqual(prefix, ("WCC", "PageRank", "BFS", "SSSP"))
        self.assertEqual(len(unsupported), 1)
        self.assertEqual(unsupported[0]["call_position"], 5)
        self.assertEqual(unsupported[0]["function"], "Closeness")

        full_prefix, full_unsupported = cumulative.supported_workflow_prefix(
            "EGGPU", cumulative.DEFAULT_WORKFLOW
        )
        self.assertEqual(full_prefix, cumulative.DEFAULT_WORKFLOW)
        self.assertEqual(full_unsupported, ())

    def test_backend_environment_uses_separate_cuda_roots(self):
        runtime = Path("/frozen/easygraph-runtime")
        eggpu = cumulative.backend_subprocess_env("EGGPU", 7, runtime)
        nxcg = cumulative.backend_subprocess_env("nx-cugraph", 7, runtime)
        self.assertNotEqual(eggpu["CUDA_PATH"], nxcg["CUDA_PATH"])
        self.assertEqual(eggpu["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(nxcg["CUDA_VISIBLE_DEVICES"], "7")
        self.assertNotIn("CFLAGS", nxcg)

    def test_eggpu_child_process_prioritizes_requested_frozen_runtime(self):
        runtime = Path("/frozen/easygraph-runtime")
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}):
            eggpu = cumulative.backend_subprocess_env("EGGPU", 7, runtime)
            nxcg = cumulative.backend_subprocess_env(
                "nx-cugraph", 7, runtime
            )

        python_paths = eggpu["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(Path(python_paths[0]), runtime)
        self.assertIn("/existing/path", python_paths)
        self.assertEqual(nxcg["PYTHONPATH"], "/existing/path")

    def test_runtime_provenance_hashes_python_tree_and_native_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            package = runtime / "easygraph"
            package.mkdir()
            (package / "__init__.py").write_text("VERSION = 'frozen'\n")
            native = runtime / (
                "cpp_easygraph" + importlib.machinery.EXTENSION_SUFFIXES[0]
            )
            native.write_bytes(b"candidate-native-binary")

            provenance = (
                cumulative.collect_runtime_repository_provenance(runtime)
            )

        self.assertFalse(
            provenance["runtime_python_snapshot"]["package_is_symlink"]
        )
        self.assertEqual(
            provenance["native_sha256"],
            hashlib.sha256(b"candidate-native-binary").hexdigest(),
        )
        self.assertTrue(
            provenance["modules"]["easygraph"]["module_origin_resolved"].endswith(
                "easygraph/__init__.py"
            )
        )

    def test_child_payload_retains_runtime_and_argv(self):
        payload = {
            "argv": ["runner.py", "--child"],
            "rows": [{"status": "ok"}],
            "runtime_provenance": {"native_sha256": "a" * 64},
            "runtime_environment": {"PYTHONPATH": "/frozen/runtime"},
        }
        stdout = cumulative.PREFIX + json.dumps(payload)
        self.assertEqual(cumulative.parse_child(stdout), payload)

    def test_runtime_rejects_easygraph_symlink_to_active_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            active = root / "active"
            runtime.mkdir()
            package = active / "easygraph"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("VERSION = 'mutable'\n")
            (runtime / "easygraph").symlink_to(package, target_is_directory=True)
            native = runtime / (
                "cpp_easygraph" + importlib.machinery.EXTENSION_SUFFIXES[0]
            )
            native.write_bytes(b"candidate-native-binary")

            with self.assertRaisesRegex(
                RuntimeError, "easygraph resolved outside --easygraph-repo"
            ):
                cumulative.collect_runtime_repository_provenance(runtime)

    def test_per_call_timeout_interrupts_a_long_baseline_call(self):
        def slow_call(*_args, **_kwargs):
            time.sleep(0.2)
            return {}

        with mock.patch.object(
            cumulative, "invoke_public_call", side_effect=slow_call
        ):
            with self.assertRaises(TimeoutError):
                cumulative.run_call_with_timeout(
                    0.02, "igraph", object(), False, "Closeness", []
                )

    def test_slow_validation_runs_but_is_excluded_from_call_seconds(self):
        validation_observed = []

        def fast_public_call(*_args, **_kwargs):
            return {"complete": "public result"}

        def slow_validation(*_args, **_kwargs):
            validation_observed.append(True)
            time.sleep(0.08)
            return {"result_validation": "pass"}

        with mock.patch.object(
            cumulative,
            "invoke_public_call",
            side_effect=fast_public_call,
        ), mock.patch.object(
            cumulative,
            "validate_public_result",
            side_effect=slow_validation,
        ):
            wall_started = time.perf_counter()
            detail, kernel_seconds, call_seconds = (
                cumulative.execute_workflow_call(
                    1.0, "igraph", object(), False, "PageRank", []
                )
            )
            wall_seconds = time.perf_counter() - wall_started

        self.assertEqual(validation_observed, [True])
        self.assertEqual(detail, {"result_validation": "pass"})
        self.assertIsNone(kernel_seconds)
        self.assertLess(call_seconds, 0.04)
        self.assertGreaterEqual(wall_seconds, 0.07)

    def test_wcc_contract_materialization_remains_in_public_timer(self):
        fake_networkx = types.SimpleNamespace(
            weakly_connected_components=object(),
            connected_components=object(),
        )

        def lazy_components():
            time.sleep(0.06)
            yield {0, 1}

        with mock.patch.dict(
            sys.modules, {"networkx": fake_networkx}
        ), mock.patch.object(
            cumulative,
            "nx_cugraph_call",
            return_value=lazy_components(),
        ):
            result, call_seconds = cumulative.run_call_with_timeout(
                1.0, "nx-cugraph", object(), True, "WCC", []
            )

        self.assertEqual(result, [{0, 1}])
        self.assertGreaterEqual(call_seconds, 0.05)

    def test_default_candidates_use_exact_all_node_closeness_protocol(self):
        self.assertNotIn("soc-Epinions1", cumulative.DEFAULT_SELECTED)
        self.assertNotIn("com-youtube", cumulative.DEFAULT_SELECTED)
        self.assertIn("ER-100k", cumulative.DEFAULT_SELECTED)


if __name__ == "__main__":
    unittest.main()
