import csv
import importlib.machinery
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


BENCHMARKING = Path(__file__).resolve().parents[1]
if str(BENCHMARKING) not in sys.path:
    sys.path.insert(0, str(BENCHMARKING))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKING / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIRST = load_module("eggpu_test_first_use", "run_eggpu_first_use.py")
FIRST_REPAIR = load_module(
    "eggpu_test_first_use_repair", "repair_eggpu_first_use.py"
)
NATURAL = load_module("eggpu_test_natural_workflow", "run_eggpu_natural_workflow.py")
CLOSENESS = load_module(
    "eggpu_test_closeness_supplement", "run_closeness_large_supplement.py"
)
ABLATION = load_module("eggpu_test_ablations", "run_eggpu_ablations.py")
FULL = load_module("eggpu_test_full_runner", "run_full_baselines.py")
MARKER = load_module(
    "eggpu_test_marker_neutrality", "verify_visibility_marker_neutrality.py"
)


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_frozen_runtime(root):
    runtime = root / "frozen-runtime"
    package = runtime / "easygraph"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VERSION = 'v10-test'\n")
    native = runtime / (
        "cpp_easygraph" + importlib.machinery.EXTENSION_SUFFIXES[0]
    )
    native.write_bytes(b"v10-native-extension")
    return runtime


def v10_metadata(runtime, implementation_digest="c" * 64):
    provenance = FIRST.collect_runtime_repository_provenance(runtime)
    easygraph = provenance["modules"]["easygraph"]
    native = provenance["modules"]["cpp_easygraph"]
    return {
        "easygraph_repo": str(runtime.resolve()),
        "source_snapshot": {"digest": "a" * 64},
        "implementation_source_snapshot": {
            "algorithm": "sha256",
            "digest": implementation_digest,
        },
        "runtime_python_snapshot": provenance["runtime_python_snapshot"],
        # The discovery list is deliberately not the runtime identity.
        "build_artifacts": {
            "cpp_easygraph": [{"sha256": "f" * 64}],
            "active_cpp_easygraph": {
                "path": native["module_origin_resolved"],
                "sha256": native["sha256"],
            },
        },
        "baseline_versions": {
            "easygraph": {
                "module_origin": easygraph["module_origin_resolved"],
            },
            "cpp_easygraph": {
                "module_origin": native["module_origin_resolved"],
                "sha256": native["sha256"],
            },
        },
    }, provenance


class ExecutionProtocolTests(unittest.TestCase):
    def test_protocol_flags_reach_every_runner_layer(self):
        library = (BENCHMARKING / "library_baselines.py").read_text()
        full = (BENCHMARKING / "run_full_baselines.py").read_text()
        split = (BENCHMARKING / "run_split_full_baselines.py").read_text()
        self.assertIn("--eggpu-execution-protocol", library)
        self.assertIn("execution_protocol == \"steady-state\"", library)
        self.assertIn("first-use protocol requires", library)
        self.assertIn("--baselines", full)
        self.assertIn("--eggpu-execution-protocol", full)
        self.assertIn("--eggpu-execution-protocol", split)

    def test_supplement_driver_repairs_and_audits_the_complete_bundle(self):
        source = (BENCHMARKING.parent / "run_missing_paper_supplements.sh").read_text()
        self.assertIn("repair_eggpu_first_use.py", source)
        self.assertIn("run_eggpu_natural_workflow.py", source)
        self.assertIn("run_closeness_large_supplement.py", source)
        self.assertIn("audit_paper_supplements.py", source)
        self.assertIn("FIRST_EVIDENCE_OUT", source)

    def test_first_use_comparison_preserves_mean_sd_and_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steady = root / "steady"
            first = root / "first"
            steady.mkdir()
            first.mkdir()
            base = {
                "dataset_size": "small",
                "graph_type": "undirected",
                "dataset": "toy",
                "function": "PageRank",
                "baseline": "EGGPU",
                "metric": "e2e",
                "status": "ok",
                "publishable": "true",
                "sample_count": "5",
                "relative_std_percent": "10",
                "ci95_half_width_value": "0.1",
            }
            write_csv(
                steady / "results_long.csv",
                [{**base, "seconds": "1", "mean_seconds": "1", "std_seconds": "0.1"}],
            )
            write_csv(
                first / "results_long.csv",
                [{**base, "seconds": "2", "mean_seconds": "2", "std_seconds": "0.2"}],
            )
            rows = FIRST.comparison_rows(steady, first)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["first_use_over_steady"], 2.0)
            self.assertEqual(rows[0]["steady_mean_plus_minus_sd"], "1 +/- 0.1")
            self.assertEqual(rows[0]["first_use_mean_plus_minus_sd"], "2 +/- 0.2")

    def test_first_use_repair_plan_uses_main_publishable_pairs(self):
        def row(dataset, function, metric, status="ok", publishable="true"):
            return {
                "dataset": dataset,
                "function": function,
                "baseline": "EGGPU",
                "metric": metric,
                "status": status,
                "publishable": publishable,
            }

        main = [
            row("a", "PageRank", "e2e"),
            row("b", "Closeness", "e2e", "skipped", "false"),
        ]
        first = [
            row("a", "PageRank", "e2e"),
            row("a", "PageRank", "kernel", "incomplete", "false"),
        ]
        timing, memory = FIRST_REPAIR.repair_plan(main, first)
        self.assertEqual(timing, {("a", "PageRank")})
        self.assertEqual(memory, {("a", "PageRank")})

    def test_first_use_repair_replaces_whole_incomplete_phase_only(self):
        base = [
            {
                "dataset": "a",
                "function": "BC",
                "baseline": "EGGPU",
                "metric": "e2e",
                "measurement_phase": "timing",
                "sample_index": "1",
                "status": "ok",
            },
            {
                "dataset": "a",
                "function": "BC",
                "baseline": "EGGPU",
                "metric": "memory_peak_gpu_proc_mb",
                "measurement_phase": "memory",
                "sample_index": "1",
                "status": "timeout",
            },
        ]
        replacement = [
            {**base[0], "seconds": "9"},
            {**base[1], "status": "ok", "seconds": "100"},
        ]
        merged = FIRST_REPAIR.merge_samples(
            base,
            {("a", "BC"): replacement},
            timing_missing=set(),
            memory_missing={("a", "BC")},
        )
        timing = [row for row in merged if row["measurement_phase"] == "timing"]
        memory = [row for row in merged if row["measurement_phase"] == "memory"]
        self.assertEqual(len(timing), 1)
        self.assertNotIn("seconds", timing[0])
        self.assertEqual(memory[0]["seconds"], "100")

    def test_first_use_requires_complete_v10_frozen_runtime_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = make_frozen_runtime(root)
            metadata, provenance = v10_metadata(runtime)
            for name in ("steady", "first"):
                directory = root / name
                directory.mkdir()
                (directory / "run_metadata.json").write_text(json.dumps(metadata))
            snapshot, artifacts = FIRST.verify_same_implementation(
                root / "steady",
                root / "first",
                expected_runtime_root=runtime,
                requested_runtime_provenance=provenance,
            )
            self.assertEqual(snapshot, "a" * 64)
            self.assertEqual(artifacts, [provenance["native_sha256"]])
            verification = json.loads(
                (root / "first" / "implementation_compatibility.json").read_text()
            )
            self.assertEqual(
                verification["mode"], "v10_frozen_runtime_identity_match"
            )

    def test_first_use_rejects_each_v10_identity_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = make_frozen_runtime(root)
            base, _provenance = v10_metadata(runtime)
            steady = root / "steady"
            first = root / "first"
            steady.mkdir()
            first.mkdir()
            (steady / "run_metadata.json").write_text(json.dumps(base))

            mutations = {
                "implementation source": lambda data: data[
                    "implementation_source_snapshot"
                ].update(digest="d" * 64),
                "runtime Python": lambda data: data[
                    "runtime_python_snapshot"
                ].update(digest="e" * 64),
                "package symlink": lambda data: data[
                    "runtime_python_snapshot"
                ].update(package_is_symlink=True),
                "active native": lambda data: (
                    data["build_artifacts"]["active_cpp_easygraph"].update(
                        sha256="1" * 64
                    ),
                    data["baseline_versions"]["cpp_easygraph"].update(
                        sha256="1" * 64
                    ),
                ),
                "module origin": lambda data: (
                    data["build_artifacts"]["active_cpp_easygraph"].update(
                        path=str(runtime / "alternate_cpp_easygraph.so")
                    ),
                    data["baseline_versions"]["cpp_easygraph"].update(
                        module_origin=str(runtime / "alternate_cpp_easygraph.so")
                    ),
                ),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    candidate = json.loads(json.dumps(base))
                    mutate(candidate)
                    (first / "run_metadata.json").write_text(
                        json.dumps(candidate)
                    )
                    with self.assertRaises(SystemExit):
                        FIRST.verify_same_implementation(steady, first)

    def test_first_use_ignores_cpp_discovery_list_when_active_native_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = make_frozen_runtime(root)
            steady_metadata, _ = v10_metadata(runtime)
            first_metadata = json.loads(json.dumps(steady_metadata))
            first_metadata["build_artifacts"]["active_cpp_easygraph"][
                "sha256"
            ] = "1" * 64
            first_metadata["baseline_versions"]["cpp_easygraph"][
                "sha256"
            ] = "1" * 64
            self.assertEqual(
                steady_metadata["build_artifacts"]["cpp_easygraph"],
                first_metadata["build_artifacts"]["cpp_easygraph"],
            )
            for name, metadata in (
                ("steady", steady_metadata),
                ("first", first_metadata),
            ):
                directory = root / name
                directory.mkdir()
                (directory / "run_metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(
                SystemExit, "frozen runtime identity mismatch"
            ):
                FIRST.verify_same_implementation(
                    root / "steady", root / "first"
                )

    def test_first_use_installs_absolute_runtime_before_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = make_frozen_runtime(root)
            original_path = list(sys.path)
            try:
                resolved, provenance = FIRST.prepare_frozen_runtime(runtime)
                self.assertEqual(Path(sys.path[0]), runtime.resolve())
                self.assertEqual(resolved, runtime.resolve())
                self.assertFalse(
                    provenance["runtime_python_snapshot"]["package_is_symlink"]
                )
                environment = {
                    "PYTHONPATH": os.pathsep.join(
                        ["/existing/runtime", str(runtime)]
                    )
                }
                FIRST.prepend_runtime_pythonpath(environment, resolved)
                self.assertEqual(
                    Path(environment["PYTHONPATH"].split(os.pathsep)[0]),
                    runtime.resolve(),
                )
                self.assertEqual(
                    environment["PYTHONPATH"].split(os.pathsep).count(
                        str(runtime.resolve())
                    ),
                    1,
                )
            finally:
                sys.path[:] = original_path

            with self.assertRaisesRegex(
                SystemExit, "must be an absolute frozen runtime path"
            ):
                FIRST.prepare_frozen_runtime(Path("relative-runtime"))

    def test_first_use_rejects_symlinked_python_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "frozen-runtime"
            real_package = runtime / "real-easygraph"
            real_package.mkdir(parents=True)
            (real_package / "__init__.py").write_text("VERSION = 'mutable'\n")
            (runtime / "easygraph").symlink_to(
                real_package, target_is_directory=True
            )
            native = runtime / (
                "cpp_easygraph" + importlib.machinery.EXTENSION_SUFFIXES[0]
            )
            native.write_bytes(b"v10-native-extension")
            original_path = list(sys.path)
            try:
                with self.assertRaisesRegex(
                    SystemExit, "package_is_symlink must be false"
                ):
                    FIRST.prepare_frozen_runtime(runtime)
            finally:
                sys.path[:] = original_path

    def test_first_use_keeps_zero_warmup_and_first_use_timing_protocol(self):
        source = (BENCHMARKING / "run_eggpu_first_use.py").read_text()
        self.assertIn('"--warmup",\n        "0"', source)
        self.assertIn('"--easygraph-warmup",\n        "0"', source)
        self.assertIn(
            '"--eggpu-execution-protocol",\n        "first-use"', source
        )
        self.assertIn(
            "Every cell is the arithmetic mean of independent subprocess samples",
            source,
        )

    def test_first_use_can_reuse_validated_unwarmed_competitor_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steady = root / "steady"
            first = root / "first"
            steady.mkdir()
            first.mkdir()
            base = {
                "dataset_size": "small",
                "graph_type": "undirected",
                "dataset": "toy",
                "function": "PageRank",
                "metric": "e2e",
                "status": "ok",
            }
            write_csv(
                steady / "results_e2e.csv",
                [
                    {**base, "baseline": "networkx", "seconds": "3"},
                    {**base, "baseline": "igraph", "seconds": "1.5"},
                    {**base, "baseline": "EGGPU", "seconds": "1"},
                ],
            )
            write_csv(
                first / "results_long.csv",
                [
                    {
                        **base,
                        "baseline": "EGGPU",
                        "publishable": "true",
                        "seconds": "2",
                        "mean_seconds": "2",
                        "std_seconds": "0.2",
                    }
                ],
            )
            rows = FIRST.compare_with_unwarmed_baselines(steady, first)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["best_unwarmed_baseline"], "igraph")
            self.assertEqual(rows[0]["best_baseline_over_eggpu_first_use"], 0.75)
            self.assertFalse(rows[0]["eggpu_first_use_is_fastest"])

    def test_natural_workflow_row_records_call_position_and_state(self):
        args = SimpleNamespace(
            dataset_size="small",
            graph_type="directed",
            dataset_name="toy",
            bfs_sources=8,
            sample_index=1,
            repeat=5,
        )
        row = NATURAL.sample_row(
            args,
            "PageRank",
            "e2e",
            0.5,
            position=2,
            state_before={"graph_context_present": True},
            state_after={"graph_context_reuse_hits": 2},
        )
        self.assertEqual(row["execution_protocol"], "natural-workflow")
        self.assertEqual(row["workflow_order_id"], "wcc_pagerank_bfs")
        self.assertEqual(row["call_position"], 2)
        self.assertTrue(row["state_before_graph_context_present"])
        self.assertEqual(row["state_after_graph_context_reuse_hits"], 2)

    def test_natural_workflow_is_not_mislabeled_as_ldbc_prescribed_order(self):
        source = (BENCHMARKING / "run_eggpu_natural_workflow.py").read_text()
        self.assertIn("order is an EGGPU case-study design", source)
        self.assertEqual(NATURAL.WORKFLOW, ("WCC", "PageRank", "BFS"))

    def test_natural_workflow_verifies_first_use_source_and_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = {
                "source_snapshot": {"algorithm": "sha256", "digest": "a" * 64},
                "implementation_source_snapshot": {
                    "algorithm": "sha256",
                    "digest": "c" * 64,
                },
                "build_artifacts": {
                    "cpp_easygraph": [{"sha256": "b" * 64}]
                },
            }
            (root / "run_metadata.json").write_text(json.dumps(metadata))
            with mock.patch.object(
                NATURAL,
                "collect_source_snapshot",
                return_value={"algorithm": "sha256", "digest": "a" * 64},
            ), mock.patch.object(
                NATURAL,
                "collect_implementation_source_snapshot",
                return_value={"algorithm": "sha256", "digest": "c" * 64},
            ), mock.patch.object(
                NATURAL,
                "cpp_easygraph_artifacts",
                return_value=[{"sha256": "b" * 64}],
            ):
                first, snapshot, artifacts = NATURAL.verify_first_use_implementation(root)
            self.assertEqual(first, metadata)
            self.assertEqual(snapshot["digest"], "a" * 64)
            self.assertEqual(artifacts, [{"sha256": "b" * 64}])

    def test_natural_workflow_records_position_two_and_three_savings(self):
        control_rows = []
        natural_rows = []
        values = {
            "WCC": (3.0, 2.0, 1),
            "PageRank": (2.0, 1.0, 2),
            "BFS": (4.0, 1.0, 3),
        }
        for function, (first_mean, natural_mean, position) in values.items():
            control_rows.append(
                {
                    "dataset": "toy",
                    "function": function,
                    "baseline": "EGGPU-isolated-first-use",
                    "metric": "e2e",
                    "status": "ok",
                    "mean_value": first_mean,
                    "std_value": 0.2,
                }
            )
            natural_rows.append(
                {
                    "dataset": "toy",
                    "dataset_size": "small",
                    "graph_type": "directed",
                    "function": function,
                    "baseline": "EGGPU-natural-workflow",
                    "metric": "e2e",
                    "status": "ok",
                    "mean_value": natural_mean,
                    "std_value": 0.1,
                    "sample_count": 5,
                    "call_position": position,
                }
            )
        per_call = NATURAL.compare_per_call(natural_rows, control_rows)
        beneficiaries = [row for row in per_call if row["reuse_beneficiary"]]
        self.assertEqual([row["call_position"] for row in beneficiaries], [2, 3])
        self.assertEqual([row["time_saved_seconds"] for row in beneficiaries], [1.0, 3.0])
        summary = NATURAL.summarize_reuse_beneficiaries(per_call)
        overall = next(
            row
            for row in summary
            if row["function"] == "ALL_BENEFICIARIES" and row["metric"] == "e2e"
        )
        self.assertEqual(overall["first_use_total_seconds"], 6.0)
        self.assertEqual(overall["natural_total_seconds"], 2.0)
        self.assertEqual(overall["time_saved_total_seconds"], 4.0)
        self.assertAlmostEqual(overall["geomean_first_use_over_natural"], 8.0 ** 0.5)

    def test_natural_workflow_control_is_same_run_and_same_semantics(self):
        source = (BENCHMARKING / "run_eggpu_natural_workflow.py").read_text()
        self.assertIn("matched_isolated_first_use_control", source)
        self.assertIn("same graph representation and API semantics", source)
        self.assertIn("isolated_first_use_samples.csv", source)

    def test_closeness_supplement_validates_every_repeat(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for sample_index in (1, 2):
                reference = root / f"reference_{sample_index}.npz"
                eggpu = root / f"eggpu_{sample_index}.npz"
                np.savez(reference, sources=np.array([0, 2]), values=np.array([1.0, 0.5]))
                np.savez(eggpu, sources=np.array([0, 2]), values=np.array([1.0, 0.5]))
                for baseline, path in (("networkx", reference), ("EGGPU", eggpu)):
                    rows.append(
                        {
                            "dataset": "toy",
                            "function": "Closeness",
                            "baseline": baseline,
                            "metric": "e2e",
                            "status": "ok",
                            "sample_index": sample_index,
                            "correctness": f"detail={path}, detail_kind=source_vector",
                        }
                    )
            validated = CLOSENESS.validate(rows)
            self.assertEqual(len(validated), 4)
            self.assertEqual({row["sample_index"] for row in validated}, {"1", "2"})
            self.assertEqual(
                {row["validation_status"] for row in validated}, {"pass"}
            )

    def test_closeness_ranking_excludes_incompletely_validated_baseline(self):
        aggregates = [
            {
                "dataset": "toy",
                "baseline": "EGGPU",
                "metric": "e2e",
                "status": "ok",
                "seconds": "1.0",
            },
            {
                "dataset": "toy",
                "baseline": "networkx",
                "metric": "e2e",
                "status": "ok",
                "seconds": "0.5",
            },
        ]
        validation = [
            {
                "dataset": "toy",
                "baseline": "EGGPU",
                "validation_status": "pass",
            },
            {
                "dataset": "toy",
                "baseline": "networkx",
                "validation_status": "fail",
            },
        ]
        filtered = CLOSENESS.apply_validation_contract(
            aggregates, validation, expected_repeat=1
        )
        networkx = next(row for row in filtered if row["baseline"] == "networkx")
        self.assertEqual(networkx["status"], "validation_failed")
        sota = CLOSENESS.summarize_sota(filtered)
        self.assertEqual(sota[0]["best_baseline"], "EGGPU")

    def test_ablation_source_provenance_never_executes_git(self):
        snapshot = {
            "algorithm": "sha256",
            "digest": "a" * 64,
            "file_count": 7,
            "total_bytes": 1234,
            "scope": ["EG_Evaluation/benchmarking"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                ABLATION, "collect_source_snapshot", return_value=snapshot
            ) as collect:
                first = ABLATION._source_state(Path(tmp))
                second = ABLATION._source_state(Path(tmp))
            collect.assert_called_once_with()
            self.assertEqual(first, second)
            self.assertFalse(first["git_required"])
            self.assertEqual(first["digest"], "a" * 64)
            self.assertTrue((Path(tmp) / "ablation_source_snapshot.json").exists())
        source = (BENCHMARKING / "run_eggpu_ablations.py").read_text()
        self.assertNotIn("rev-parse", source)

    def test_formal_runtime_has_no_git_cli_dependency(self):
        paths = [
            BENCHMARKING / "run_full_baselines.py",
            BENCHMARKING / "run_eggpu_ablations.py",
            BENCHMARKING / "preflight_full_eval_ready.py",
            BENCHMARKING / "audit_full_result.py",
            BENCHMARKING.parent / "run_final_gpu5_gate_and_full.sh",
            BENCHMARKING.parent / "run_complete_paper_experiments.sh",
        ]
        for path in paths:
            source = path.read_text()
            self.assertNotIn("run_git(", source, str(path))
            self.assertNotIn('subprocess.run(["git"', source, str(path))
            self.assertNotIn("command -v git", source, str(path))

    def test_marker_gate_uses_paired_ratios_under_temporal_drift(self):
        # Reproduce the July-12 trace: marginal medians are misleading because
        # the final off/on pair both ran slower, while paired A/B ratios are stable.
        rows = [
            {"marker": "off", "e2e_median_seconds": 0.0005758506, "kernel_median_seconds": 0.0004925120},
            {"marker": "on", "e2e_median_seconds": 0.0006571610, "kernel_median_seconds": 0.0005561920},
            {"marker": "off", "e2e_median_seconds": 0.0005765004, "kernel_median_seconds": 0.0004948480},
            {"marker": "on", "e2e_median_seconds": 0.0005876659, "kernel_median_seconds": 0.0005015040},
            {"marker": "off", "e2e_median_seconds": 0.0006322956, "kernel_median_seconds": 0.0005374240},
            {"marker": "on", "e2e_median_seconds": 0.0006323354, "kernel_median_seconds": 0.0005433280},
        ]
        paired = MARKER.paired_ratios(rows, 3)
        self.assertLess(
            statistics.median(row["e2e_on_over_off"] for row in paired), 1.05
        )
        self.assertLess(
            statistics.median(row["kernel_on_over_off"] for row in paired), 1.05
        )

    def test_marker_neutrality_matches_production_parent_child_topology(self):
        source = (BENCHMARKING / "verify_visibility_marker_neutrality.py").read_text()
        self.assertIn("def trial(args):", source)
        self.assertIn('"--trial"', source)
        self.assertIn('"--worker"', source)
        self.assertIn("EGGPU_NEUTRALITY_PARENT_MARKER_STARTED", source)
        self.assertIn(
            "parent_process_marker_with_isolated_timed_child", source
        )

    def test_in_process_ablation_forbids_auxiliary_marker(self):
        source = (BENCHMARKING / "run_eggpu_ablations.py").read_text()
        self.assertNotIn("GpuVisibilityMarker", source)
        self.assertIn(
            'os.environ["EGGPU_GPU_VISIBILITY_MARKER"] = "FALSE"', source
        )

    def test_idle_guard_retries_driver_memory_after_child_exit(self):
        mib = 1024 * 1024
        fake_nvml = SimpleNamespace(
            nvmlDeviceGetHandleByIndex=lambda index: object(),
            nvmlDeviceGetMemoryInfo=mock.Mock(
                side_effect=[
                    SimpleNamespace(used=1500 * mib),
                    SimpleNamespace(used=4 * mib),
                ]
            ),
            nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=0),
        )
        env = {
            "EGGPU_MONITOR_GPU_INDEX": "5",
            "EGGPU_IDLE_RETRY_ATTEMPTS": "2",
            "EGGPU_IDLE_RETRY_SLEEP_S": "0",
        }
        with mock.patch.object(FULL, "_ensure_nvml", return_value=True), mock.patch.object(
            FULL, "pynvml", fake_nvml
        ), mock.patch.object(FULL, "_nvml_compute_processes", return_value=[]):
            ok, note = FULL.check_eggpu_child_gpu_idle(env)
        self.assertTrue(ok, note)
        self.assertIn("idle_retry_attempt=2", note)

    def test_idle_guard_subtracts_current_benchmark_process_memory(self):
        mib = 1024 * 1024
        own_process = SimpleNamespace(pid=os.getpid(), usedGpuMemory=1500 * mib)
        fake_nvml = SimpleNamespace(
            nvmlDeviceGetHandleByIndex=lambda index: object(),
            nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=1504 * mib),
            nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=0),
        )
        env = {"EGGPU_MONITOR_GPU_INDEX": "5"}
        with mock.patch.object(FULL, "_ensure_nvml", return_value=True), mock.patch.object(
            FULL, "pynvml", fake_nvml
        ), mock.patch.object(
            FULL, "_nvml_compute_processes", return_value=[own_process]
        ):
            ok, note = FULL.check_eggpu_child_gpu_idle(env)
        self.assertTrue(ok, note)
        self.assertIn("allowed_process_memory_mb=1500.0", note)

    def test_idle_guard_ignores_stale_nvml_process_entry(self):
        mib = 1024 * 1024
        stale_process = SimpleNamespace(pid=987654321, usedGpuMemory=78 * mib)
        fake_nvml = SimpleNamespace(
            nvmlDeviceGetHandleByIndex=lambda index: object(),
            nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=854 * mib),
            nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=0),
        )
        env = {"EGGPU_MONITOR_GPU_INDEX": "5"}
        with mock.patch.object(FULL, "_ensure_nvml", return_value=True), mock.patch.object(
            FULL, "pynvml", fake_nvml
        ), mock.patch.object(
            FULL, "_nvml_compute_processes", return_value=[stale_process]
        ), mock.patch.object(FULL, "_pid_is_alive", return_value=False):
            ok, note = FULL.check_eggpu_child_gpu_idle(env)
        self.assertTrue(ok, note)
        self.assertNotIn("compute_processes", note)

    def test_complete_wrapper_does_not_inject_cuda_libs_into_child_bash(self):
        source = (BENCHMARKING.parent / "run_complete_paper_experiments.sh").read_text()
        self.assertIn("-u LD_LIBRARY_PATH", source)
        self.assertIn('LD_LIBRARY_PATH="${CUDA_LD_PATH}"', source)


if __name__ == "__main__":
    unittest.main()
