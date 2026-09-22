import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "Easy-Graph"))
sys.path.insert(0, str(HERE.parents[1]))

import run_eggpu_scaling as scaling


def runtime_provenance(root, native_sha="a" * 64, python_digest="b" * 64):
    root = str(Path(root).resolve())
    return {
        "requested_root": root,
        "resolved_root": root,
        "native_sha256": native_sha,
        "runtime_python_snapshot": {
            "algorithm": "sha256",
            "digest": python_digest,
            "runtime_root": root,
        },
        "modules": {
            "easygraph": {
                "module_origin_resolved": str(Path(root) / "easygraph" / "__init__.py"),
            },
            "cpp_easygraph": {
                "module_origin_resolved": str(Path(root) / "cpp_easygraph.so"),
                "sha256": native_sha,
            },
        },
    }


def runtime_record(root, provenance, **fields):
    record = {
        **fields,
        **scaling.runtime_record_envelope(
            root,
            provenance,
            loaded_runtime={
                "resolved_root": str(Path(root).resolve()),
                "native_sha256": provenance["native_sha256"],
                "modules": provenance["modules"],
            },
            argv=["run_eggpu_scaling.py", "--worker"],
            environment={"PYTHONPATH": str(Path(root).resolve())},
        ),
    }
    return record


class ScalingResumeProtocolTests(unittest.TestCase):
    def test_easygraph_repo_is_required(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "run_eggpu_scaling.py",
                "--manifests",
                "/tmp/G.json",
                "--output-dir",
                "/tmp/out",
            ],
        ):
            with self.assertRaises(SystemExit):
                scaling.parse_args()

    def test_runtime_root_is_first_in_sys_path_and_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            runtime.mkdir()
            existing = str(Path(directory) / "existing")
            with mock.patch.object(sys, "path", [existing, str(runtime)]):
                with mock.patch.dict(
                    os.environ,
                    {"PYTHONPATH": os.pathsep.join([existing, str(runtime)])},
                    clear=False,
                ):
                    resolved = scaling.configure_runtime_import_root(runtime)
                    self.assertEqual(Path(sys.path[0]), runtime.resolve())
                    python_paths = os.environ["PYTHONPATH"].split(os.pathsep)
                    self.assertEqual(Path(python_paths[0]), runtime.resolve())
                    self.assertEqual(python_paths.count(str(runtime.resolve())), 1)
                    self.assertEqual(resolved, runtime.resolve())

    def test_runtime_is_pinned_before_easygraph_import_and_origins_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            runtime.mkdir()
            provenance = runtime_provenance(runtime)
            loaded = {
                "resolved_root": str(runtime.resolve()),
                "native_sha256": provenance["native_sha256"],
                "modules": provenance["modules"],
            }
            order = []

            def configure(path):
                self.assertEqual(Path(path), runtime)
                order.append("configure")
                return runtime.resolve()

            def import_module(name):
                order.append(f"import:{name}")
                return mock.Mock()

            with mock.patch.object(scaling, "eg", None):
                with mock.patch.object(scaling, "gpu_eggpu_backend", None):
                    with mock.patch.object(
                        scaling,
                        "configure_runtime_import_root",
                        side_effect=configure,
                    ):
                        with mock.patch.object(
                            scaling.importlib,
                            "import_module",
                            side_effect=import_module,
                        ):
                            with mock.patch.object(
                                scaling,
                                "collect_runtime_repository_provenance",
                                return_value=provenance,
                            ):
                                with mock.patch.object(
                                    scaling,
                                    "collect_loaded_runtime_provenance",
                                    return_value=loaded,
                                ):
                                    scaling.load_easygraph_runtime(runtime)

            self.assertEqual(order[0], "configure")
            self.assertEqual(order[1], "import:easygraph")
            self.assertIn("import:cpp_easygraph", order)
            self.assertIn(
                "import:easygraph.utils.gpu_eggpu_backend", order
            )

    def test_timed_arms_watchdog_with_phase_specific_limit(self):
        watchdog = mock.Mock()

        value, elapsed = scaling.timed(lambda: 7, 321.0, watchdog=watchdog)

        self.assertEqual(value, 7)
        self.assertGreaterEqual(elapsed, 0.0)
        watchdog.arm.assert_called_once_with(321.0)
        watchdog.disarm.assert_called_once_with()

    def test_watchdog_command_inherits_frozen_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            runtime.mkdir()
            process = mock.Mock()
            process.poll.return_value = 0
            with mock.patch.object(
                scaling.subprocess, "Popen", return_value=process
            ) as popen:
                watchdog = scaling.CallWatchdog(
                    Path(directory) / "timeout.json", 10.0, runtime
                )
                command = popen.call_args.args[0]
                kwargs = popen.call_args.kwargs
                easygraph_index = command.index("--easygraph-repo")
                self.assertEqual(
                    Path(command[easygraph_index + 1]), runtime.resolve()
                )
                self.assertEqual(
                    Path(kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0]),
                    runtime.resolve(),
                )
                watchdog.close()

    def test_scc_kernel_timer_matches_graph_semantics(self):
        undirected = SimpleNamespace(is_directed=lambda: False)
        directed = SimpleNamespace(is_directed=lambda: True)

        self.assertEqual(scaling.kernel_key("SCC", undirected), "cc")
        self.assertEqual(scaling.kernel_key("SCC", directed), "scc")

    def test_hard_watchdog_timeout_is_classified_as_timeout(self):
        error = TimeoutError("hard per-call timeout after 100.000s")
        self.assertEqual(scaling.classify_failure(error), "timeout")

    def test_memory_resume_requires_nonzero_coordinator_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            runtime.mkdir()
            provenance = runtime_provenance(runtime)
            path = Path(directory) / "G_BFS_memory_1.json"
            record = runtime_record(
                runtime,
                provenance,
                status="ok",
                measurement="memory",
                dataset="G",
                function="BFS",
                memory={
                    "memory_monitor_origin": "coordinator_process_child_tree",
                    "monitor_rss_samples": 12,
                    "monitor_gpu_proc_samples": 0,
                    "rss_mb": 256.0,
                    "gpu_proc_peak_mb": 0.0,
                },
            )
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertIsNone(
                scaling.reusable_record(
                    path, "G", "BFS", "memory", provenance
                )
            )

            record["memory"]["monitor_gpu_proc_samples"] = 8
            record["memory"]["gpu_proc_peak_mb"] = 64.0
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(
                scaling.reusable_record(
                    path, "G", "BFS", "memory", provenance
                ),
                record,
            )

    def test_resume_rejects_native_or_python_runtime_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            runtime.mkdir()
            expected = runtime_provenance(runtime)
            path = Path(directory) / "G_BFS_timing_1.json"
            record = runtime_record(
                runtime,
                expected,
                status="ok",
                measurement="timing",
                dataset="G",
                function="BFS",
            )
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(
                scaling.reusable_record(
                    path, "G", "BFS", "timing", expected
                ),
                record,
            )

            changed_native = runtime_provenance(runtime, native_sha="c" * 64)
            self.assertIsNone(
                scaling.reusable_record(
                    path, "G", "BFS", "timing", changed_native
                )
            )
            changed_python = runtime_provenance(
                runtime, python_digest="d" * 64
            )
            self.assertIsNone(
                scaling.reusable_record(
                    path, "G", "BFS", "timing", changed_python
                )
            )

    def test_worker_command_and_environment_pin_requested_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            provenance = runtime_provenance(runtime)
            output = root / "worker.json"
            args = SimpleNamespace(
                easygraph_repo=runtime,
                requested_runtime_provenance=provenance,
                repeat=1,
                warmup=0,
                source_count=1,
                bc_source_count=1,
                closeness_source_count=1,
                structural_node_count=0,
                memory_poll_ms=2.0,
                timeout=10.0,
                first_use_call_timeout=10.0,
                load_timeout=10.0,
                validate=False,
                hard_call_timeout=False,
            )

            def fake_run(command, **kwargs):
                easygraph_index = command.index("--easygraph-repo")
                self.assertEqual(
                    Path(command[easygraph_index + 1]), runtime.resolve()
                )
                native_index = command.index("--expected-native-sha256")
                python_index = command.index(
                    "--expected-runtime-python-digest"
                )
                self.assertEqual(
                    command[native_index + 1], provenance["native_sha256"]
                )
                self.assertEqual(
                    command[python_index + 1],
                    provenance["runtime_python_snapshot"]["digest"],
                )
                self.assertEqual(
                    Path(kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0]),
                    runtime.resolve(),
                )
                output.write_text(
                    json.dumps(
                        runtime_record(
                            runtime,
                            provenance,
                            status="ok",
                            measurement="timing",
                            dataset="G",
                            function="BFS",
                        )
                    ),
                    encoding="utf-8",
                )
                return scaling.subprocess.CompletedProcess(
                    command, 0, stdout=""
                )

            with mock.patch.object(
                scaling.subprocess, "run", side_effect=fake_run
            ):
                record = scaling.run_child(
                    args,
                    root / "G.json",
                    "BFS",
                    "timing",
                    output,
                    timeout=10.0,
                )
            self.assertEqual(record["runtime_identity"], scaling._runtime_identity(provenance))

    def test_terminal_timeout_is_reused_and_stops_followup_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            provenance = runtime_provenance(runtime)
            manifest = root / "G.json"
            manifest.write_text(
                json.dumps(
                    {
                        "name": "G",
                        "num_nodes": 10,
                        "num_edges": 20,
                        "num_entries": 20,
                        "directed": True,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "out"
            raw = output / "raw"
            raw.mkdir(parents=True)
            (raw / "G_SCC_timing_1.json").write_text(
                json.dumps(
                    runtime_record(
                        runtime,
                        provenance,
                        status="timeout",
                        failure_kind="timeout",
                        measurement="timing",
                        dataset="G",
                        function="SCC",
                        error="protocol timeout",
                    )
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                easygraph_repo=runtime,
                output_dir=str(output),
                functions="SCC",
                retry_cells="",
                timing_processes=0,
                repeat=5,
                warmup=2,
                memory_repeat=3,
                manifests=[str(manifest)],
                resume=True,
                continue_on_failure=True,
            )
            with mock.patch.object(
                scaling,
                "collect_runtime_repository_provenance",
                return_value=provenance,
            ):
                with mock.patch.object(
                    scaling,
                    "run_child",
                    side_effect=AssertionError(
                        "terminal cell must not launch a worker"
                    ),
                ):
                    scaling.coordinator(args)

            timing_two = json.loads(
                (raw / "G_SCC_timing_2.json").read_text(encoding="utf-8")
            )
            memory_one = json.loads(
                (raw / "G_SCC_memory_1.json").read_text(encoding="utf-8")
            )
            aggregate = json.loads(
                (raw / "G_SCC_timing.json").read_text(encoding="utf-8")
            )
            run_metadata = json.loads(
                (output / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                timing_two["failure_kind"], "not_run_after_terminal_failure"
            )
            self.assertEqual(timing_two["root_failure_kind"], "timeout")
            self.assertEqual(memory_one["failure_kind"], "timing_unavailable")
            self.assertEqual(aggregate["failure_kind"], "timeout")
            self.assertEqual(
                run_metadata["native_binary_sha256"],
                provenance["native_sha256"],
            )
            self.assertEqual(
                run_metadata["runtime_python_digest"],
                provenance["runtime_python_snapshot"]["digest"],
            )
            self.assertEqual(
                run_metadata["python_origins"]["easygraph"],
                provenance["modules"]["easygraph"][
                    "module_origin_resolved"
                ],
            )
            self.assertIn(
                "worker", run_metadata["controlled_environment"]
            )


if __name__ == "__main__":
    unittest.main()
