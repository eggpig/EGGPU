import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve()
BENCHMARKING = HERE.parents[1]
sys.path.insert(0, str(BENCHMARKING))

import run_eggpu_ablations as ablation
import summarize_ablation_system as ablation_summary


def runtime_provenance(root, native_sha="a" * 64, python_digest="b" * 64):
    root = Path(root).resolve()
    return {
        "requested_root": str(root),
        "resolved_root": str(root),
        "native_sha256": native_sha,
        "runtime_python_snapshot": {
            "algorithm": "sha256",
            "digest": python_digest,
            "runtime_root": str(root),
        },
        "modules": {
            "easygraph": {
                "module_origin_resolved": str(root / "easygraph" / "__init__.py"),
                "sha256": "e" * 64,
            },
            "cpp_easygraph": {
                "module_origin_resolved": str(root / "cpp_easygraph.so"),
                "sha256": native_sha,
            },
        },
    }


def loaded_runtime(root, native_sha="a" * 64):
    requested = runtime_provenance(root, native_sha=native_sha)
    return {
        "resolved_root": requested["resolved_root"],
        "native_sha256": native_sha,
        "modules": requested["modules"],
    }


def benchmark_args(root, dataset):
    return SimpleNamespace(
        easygraph_repo=Path(root),
        expected_native_sha256="a" * 64,
        expected_runtime_python_digest="b" * 64,
        experiment="workflow",
        variant="full",
        edge_path=str(dataset),
        dataset_name="toy",
        graph_type="undirected",
        functions="all",
        workflow_order_id="canonical",
        repeat=5,
        warmup=2,
        gpu=0,
        sssp_sources=8,
        bc_sources=16,
        closeness_sources=16,
        layout_pr_iters=20,
        measurement_mode="timing",
        out="",
        resume=True,
        worker=False,
    )


class AblationRuntimeProvenanceTests(unittest.TestCase):
    def test_easygraph_repo_is_required(self):
        argv = [
            "run_eggpu_ablations.py",
            "--experiment",
            "workflow",
            "--edge-path",
            "/tmp/toy.txt",
            "--graph-type",
            "undirected",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                ablation.parse_args()

    def test_child_environment_and_command_pin_frozen_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            dataset = root / "toy.txt"
            dataset.write_text("0 1\n", encoding="utf-8")
            args = benchmark_args(runtime, dataset)
            provenance = runtime_provenance(runtime)
            existing = str(root / "other")
            environment = {
                "PYTHONPATH": os.pathsep.join(
                    [existing, str(runtime), str(runtime)]
                )
            }

            child_environment = ablation.runtime_subprocess_environment(
                runtime, environment
            )
            python_paths = child_environment["PYTHONPATH"].split(os.pathsep)
            self.assertEqual(Path(python_paths[0]), runtime.resolve())
            self.assertEqual(python_paths.count(str(runtime.resolve())), 1)

            command = ablation.worker_command(args, provenance)
            repo_index = command.index("--easygraph-repo")
            native_index = command.index("--expected-native-sha256")
            python_index = command.index(
                "--expected-runtime-python-digest"
            )
            self.assertEqual(Path(command[repo_index + 1]), runtime.resolve())
            self.assertEqual(command[native_index + 1], "a" * 64)
            self.assertEqual(command[python_index + 1], "b" * 64)
            self.assertIn("--worker", command)

    def test_worker_rejects_coordinator_or_loaded_runtime_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            dataset = root / "toy.txt"
            dataset.write_text("0 1\n", encoding="utf-8")
            args = benchmark_args(runtime, dataset)
            requested = runtime_provenance(runtime)
            loaded = loaded_runtime(runtime)

            with mock.patch.object(
                ablation,
                "collect_runtime_repository_provenance",
                return_value=requested,
            ), mock.patch.object(
                ablation,
                "install_runtime_import_root",
                return_value=runtime.resolve(),
            ), mock.patch.object(
                ablation.importlib, "import_module", return_value=mock.Mock()
            ), mock.patch.object(
                ablation,
                "collect_loaded_runtime_provenance",
                return_value=loaded,
            ):
                observed_requested, observed_loaded = (
                    ablation.load_and_validate_runtime(args)
                )
            self.assertEqual(observed_requested, requested)
            self.assertEqual(observed_loaded, loaded)

            args.expected_runtime_python_digest = "c" * 64
            with mock.patch.object(
                ablation,
                "collect_runtime_repository_provenance",
                return_value=requested,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Python runtime changed"
                ):
                    ablation.load_and_validate_runtime(args)

            args.expected_runtime_python_digest = "b" * 64
            drifted_loaded = loaded_runtime(runtime, native_sha="d" * 64)
            with mock.patch.object(
                ablation,
                "collect_runtime_repository_provenance",
                return_value=requested,
            ), mock.patch.object(
                ablation,
                "install_runtime_import_root",
                return_value=runtime.resolve(),
            ), mock.patch.object(
                ablation.importlib, "import_module", return_value=mock.Mock()
            ), mock.patch.object(
                ablation,
                "collect_loaded_runtime_provenance",
                return_value=drifted_loaded,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "loaded runtime"
                ):
                    ablation.load_and_validate_runtime(args)

    def test_resume_reuses_exact_identity_and_rejects_mixed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir()
            dataset = root / "toy.txt"
            dataset.write_text("0 1\n", encoding="utf-8")
            output = root / "workflow_toy_canonical_full.csv"
            output.write_text("status\nok\n", encoding="utf-8")
            args = benchmark_args(runtime, dataset)
            args.out = str(output)
            requested = runtime_provenance(runtime)
            metadata = {
                "run_status": "complete",
                "runtime_provenance": requested,
                "loaded_runtime_provenance": loaded_runtime(runtime),
                "benchmark_contract": ablation.benchmark_contract(args),
                "dataset_artifact": ablation._dataset_artifact(args),
            }
            output.with_suffix(".metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            self.assertTrue(ablation.reusable_output(args, requested))
            with self.assertRaisesRegex(RuntimeError, "resumed runtime"):
                ablation.reusable_output(
                    args,
                    runtime_provenance(runtime, native_sha="c" * 64),
                )
            with self.assertRaisesRegex(RuntimeError, "resumed runtime"):
                ablation.reusable_output(
                    args,
                    runtime_provenance(runtime, python_digest="d" * 64),
                )

    def test_submission_estimator_keeps_all_raw_samples_and_dispersion(self):
        args = SimpleNamespace(repeat=5)
        policy = ablation.estimator_policy(args)
        self.assertEqual(policy["submission_estimator"], "best_observed")
        self.assertEqual(
            policy["submission_estimator_definition"],
            "minimum_of_five_independent_runs",
        )
        self.assertTrue(policy["submission_repeat_contract_satisfied"])
        self.assertIn("arithmetic_mean", policy["descriptive_statistics"])
        self.assertIn(
            "sample_standard_deviation", policy["descriptive_statistics"]
        )
        self.assertIn("minimum", policy["descriptive_statistics"])

    def test_repeat_summary_retains_mean_std_and_minimum_of_five(self):
        rows = []
        for repeat, value in enumerate((1.0, 0.9, 1.1, 0.95, 1.05)):
            rows.append(
                {
                    "experiment": "workflow",
                    "variant": "full",
                    "dataset": "toy",
                    "graph_type": "undirected",
                    "workflow_order_id": "canonical",
                    "function": "PageRank",
                    "metric": "e2e",
                    "unit": "s",
                    "status": "ok",
                    "repeat": repeat,
                    "value": value,
                }
            )
        frame = ablation_summary.pd.DataFrame(rows)
        summary = ablation_summary.summarize_repeat_statistics(frame).iloc[0]
        self.assertEqual(summary["sample_count"], 5)
        self.assertAlmostEqual(summary["mean"], 1.0)
        self.assertGreater(summary["sample_std"], 0.0)
        self.assertAlmostEqual(summary["minimum"], 0.9)
        self.assertAlmostEqual(summary["submission_value"], 0.9)
        self.assertEqual(summary["submission_estimator"], "best_observed")
        self.assertTrue(summary["submission_repeat_contract_satisfied"])


if __name__ == "__main__":
    unittest.main()
