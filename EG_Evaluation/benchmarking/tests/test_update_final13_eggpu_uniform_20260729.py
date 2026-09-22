import csv
import importlib.util
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "update_final13_eggpu_uniform_20260729.py"
)
SPEC = importlib.util.spec_from_file_location("eggpu_uniform_overlay", MODULE_PATH)
OVERLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OVERLAY
SPEC.loader.exec_module(OVERLAY)


SHA = "a" * 64
RUNTIME_SHA = "b" * 64
FUNCTIONS = (
    "PageRank",
    "MST",
    "LCC",
    "WCC",
    "SCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
    "BC",
    "Closeness",
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
)
DATASETS = tuple(f"g{index}" for index in range(13))
BASELINES = (
    "networkx",
    "igraph",
    "easygraph-cpu",
    "easygraph-cpp",
    "EGGPU",
    "nx-cugraph",
    "Gunrock",
    "GraphScope",
)


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def base_ledger():
    rows = []
    for dataset in DATASETS:
        for function in FUNCTIONS:
            for baseline in BASELINES:
                rows.append(
                    {
                        "dataset": dataset,
                        "function": function,
                        "category": "C",
                        "baseline": baseline,
                        "support_class": "T",
                        "execution_status": "ok",
                        "failure_kind": "",
                        "reason": "",
                        "e2e_paper_seconds": "9",
                        "e2e_raw_mean_seconds": "9",
                        "e2e_std_seconds": "1",
                        "e2e_estimator": "arithmetic_mean",
                        "validation_status": "pass",
                        "result_source": "old",
                        "sample_count": "5",
                        "build_paper_seconds": "8",
                        "build_raw_mean_seconds": "8",
                        "build_std_seconds": "1",
                        "build_estimator": "arithmetic_mean",
                        "kernel_paper_seconds": "7",
                        "kernel_raw_mean_seconds": "7",
                        "kernel_std_seconds": "1",
                        "kernel_estimator": "arithmetic_mean",
                        "gpu_peak_mb_mean": "123",
                        "gpu_peak_mb_std": "4",
                        "memory_sample_count": "3",
                        "host_rss_peak_mb_mean": "456",
                        "host_rss_peak_mb_std": "5",
                        "memory_measurement_window": "preserve-me",
                        "excluded_observed_status": "",
                    }
                )
    return rows


def metadata(path):
    (path / "run_metadata.json").write_text(
        json.dumps(
            {
                "baseline_versions": {"cpp_easygraph": {"sha256": SHA}},
                "runtime_python_snapshot": {
                    "digest": RUNTIME_SHA,
                    "package_is_symlink": False,
                },
            }
        ),
        encoding="utf-8",
    )


def scaling_metadata(path):
    runtime_root = str(path.resolve())
    package_path = str(path.resolve() / "easygraph")
    easygraph_origin = str(path.resolve() / "easygraph" / "__init__.py")
    cpp_origin = str(
        path.resolve()
        / "cpp_easygraph.cpython-310-x86_64-linux-gnu.so"
    )
    (path / "run_metadata.json").write_text(
        json.dumps(
            {
                "protocol": "eggpu_scaling_frozen_runtime_v10",
                "argv": [
                    "run_eggpu_scaling.py",
                    "--easygraph-repo",
                    runtime_root,
                ],
                "easygraph_repo": runtime_root,
                "repository_runtime_provenance": {
                    "requested_root": runtime_root,
                    "resolved_root": runtime_root,
                    "runtime_python_snapshot": {
                        "algorithm": "sha256",
                        "digest": RUNTIME_SHA,
                        "file_count": 316,
                        "total_bytes": 123456,
                        "runtime_root": runtime_root,
                        "package_path": package_path,
                        "package_resolved_path": package_path,
                        "package_is_symlink": False,
                        "package_symlink_target": "",
                        "scope": [
                            "easygraph/**/*.py",
                            "easygraph/**/*.json",
                            "easygraph/**/*.txt",
                        ],
                    },
                    "modules": {
                        "easygraph": {
                            "module_origin": easygraph_origin,
                            "module_origin_resolved": easygraph_origin,
                            "relative_to_runtime": "easygraph/__init__.py",
                            "size_bytes": 100,
                            "sha256": "d" * 64,
                            "is_native_extension": False,
                        },
                        "cpp_easygraph": {
                            "module_origin": cpp_origin,
                            "module_origin_resolved": cpp_origin,
                            "relative_to_runtime": (
                                "cpp_easygraph.cpython-310-"
                                "x86_64-linux-gnu.so"
                            ),
                            "size_bytes": 1000,
                            "sha256": SHA,
                            "is_native_extension": True,
                        },
                    },
                    "native_sha256": SHA,
                },
                "runtime_python_digest": RUNTIME_SHA,
                "python_origins": {
                    "easygraph": easygraph_origin,
                    "cpp_easygraph": cpp_origin,
                },
                "native_binary_sha256": SHA,
                "controlled_environment": {
                    "coordinator": {},
                    "worker": {},
                },
                "timing_semantics": (
                    "Runtime fingerprinting and origin validation complete "
                    "before graph loading."
                ),
                "requested_functions": "PageRank",
                "requested_manifests": ["/datasets/g0.json"],
                "timing_processes": 0,
                "repeat": 5,
                "warmup": 1,
                "memory_repeat": 3,
            }
        ),
        encoding="utf-8",
    )


def scaling_raw_runtime(path):
    run_metadata = json.loads(
        (path / "run_metadata.json").read_text(encoding="utf-8")
    )
    requested = run_metadata["repository_runtime_provenance"]
    return {
        "runtime_identity": {
            "native_sha256": SHA,
            "runtime_python_digest": RUNTIME_SHA,
            "resolved_root": requested["resolved_root"],
        },
        "requested_runtime_provenance": requested,
        "runtime_provenance": {
            "resolved_root": requested["resolved_root"],
            "modules": requested["modules"],
            "native_sha256": SHA,
        },
    }


def timing_stats(minimum=1.0):
    return OVERLAY.summarize_timing_samples(
        [
            minimum,
            minimum * 1.05,
            minimum * 1.10,
            minimum * 1.15,
            minimum * 1.20,
        ],
        "fixture",
    )


class UniformOverlayTests(unittest.TestCase):
    def test_catastrophic_outlier_guard_accepts_jitter_and_rejects_outliers(self):
        ordinary_jitter = OVERLAY.summarize_timing_samples(
            [1.0, 1.4, 1.2, 1.3, 1.1],
            "ordinary-jitter",
        )
        self.assertEqual(ordinary_jitter.stability_status, "pass")
        self.assertGreater(ordinary_jitter.coefficient_of_variation, 0)

        huge_high_outlier = OVERLAY.summarize_timing_samples(
            [0.01, 0.011, 0.012, 0.013, 0.20],
            "high-outlier",
        )
        self.assertEqual(huge_high_outlier.stability_status, "fail")
        self.assertGreater(huge_high_outlier.max_over_median, 5.0)

        anomalously_low_minimum = OVERLAY.summarize_timing_samples(
            [0.001, 0.010, 0.011, 0.012, 0.013],
            "low-minimum",
        )
        self.assertEqual(anomalously_low_minimum.stability_status, "fail")
        self.assertGreater(anomalously_low_minimum.median_over_minimum, 3.0)

        zero_minimum = OVERLAY.summarize_timing_samples(
            [0.0, 0.010, 0.011, 0.012, 0.013],
            "zero-minimum",
        )
        self.assertEqual(zero_minimum.stability_status, "fail")
        self.assertEqual(zero_minimum.median_over_minimum, float("inf"))

    def test_overlay_updates_uniform_timing_and_preserves_memory(self):
        rows = base_ledger()
        fields = list(rows[0])
        candidate = OVERLAY.Candidate(
            dataset="g0",
            function="PageRank",
            source_kind="main",
            result_source="/new",
            candidate_sha256=SHA,
            metrics={
                "build": timing_stats(3.0),
                "e2e": timing_stats(2.0),
                "kernel": timing_stats(1.0),
            },
        )
        out_fields, out, comparisons, rejected = OVERLAY.overlay(
            fields, rows, [candidate], 1
        )
        row = next(
            item
            for item in out
            if item["dataset"] == "g0"
            and item["function"] == "PageRank"
            and item["baseline"] == "EGGPU"
        )
        self.assertIn("candidate_sha256", out_fields)
        self.assertEqual(row["e2e_paper_seconds"], "2.0")
        self.assertEqual(row["e2e_raw_mean_seconds"], "2.2")
        self.assertEqual(row["e2e_raw_min_seconds"], "2.0")
        self.assertEqual(row["e2e_raw_max_seconds"], "2.4")
        self.assertEqual(row["e2e_stability_status"], "pass")
        self.assertEqual(row["e2e_estimator"], "minimum_of_five")
        self.assertEqual(row["candidate_sha256"], SHA)
        self.assertEqual(row["gpu_peak_mb_mean"], "123")
        self.assertEqual(row["memory_measurement_window"], "preserve-me")
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(rejected, [])

    def test_duplicate_candidate_key_is_rejected(self):
        rows = base_ledger()
        candidate = OVERLAY.Candidate(
            dataset="g0",
            function="PageRank",
            source_kind="main",
            result_source="/new",
            candidate_sha256=SHA,
            metrics={metric: timing_stats() for metric in OVERLAY.METRICS},
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            OVERLAY.overlay(list(rows[0]), rows, [candidate, candidate], None)

    def test_sampled_pass_semantic_gate_is_preserved(self):
        rows = base_ledger()
        target = next(
            row
            for row in rows
            if row["dataset"] == "g0"
            and row["function"] == "PageRank"
            and row["baseline"] == "EGGPU"
        )
        target["validation_status"] = "sampled_pass"
        candidate = OVERLAY.Candidate(
            dataset="g0",
            function="PageRank",
            source_kind="main",
            result_source="/new",
            candidate_sha256=SHA,
            metrics={metric: timing_stats() for metric in OVERLAY.METRICS},
        )
        _fields, output, comparisons, rejected = OVERLAY.overlay(
            list(rows[0]), rows, [candidate], 1
        )
        replaced = next(
            row
            for row in output
            if row["dataset"] == "g0"
            and row["function"] == "PageRank"
            and row["baseline"] == "EGGPU"
        )
        self.assertEqual(replaced["validation_status"], "sampled_pass")
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(rejected, [])

    def test_main_parser_requires_five_consistent_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata(root)
            samples = []
            aggregates = []
            for metric_index, metric in enumerate(OVERLAY.METRICS, start=1):
                values = [metric_index + index / 10 for index in range(5)]
                for index, value in enumerate(values, start=1):
                    samples.append(
                        {
                            "dataset": "g0",
                            "function": "PageRank",
                            "baseline": "EGGPU",
                            "metric": metric,
                            "value": value,
                            "seconds": value,
                            "unit": "s",
                            "status": "ok",
                            "correctness": "digest",
                            "sample_index": index,
                            "sample_count": 5,
                            "measurement_phase": "timing",
                        }
                    )
                aggregates.append(
                    {
                        "dataset": "g0",
                        "function": "PageRank",
                        "baseline": "EGGPU",
                        "metric": metric,
                        "status": "ok",
                        "sample_count": 5,
                        "n_total": 5,
                        "n_valid": 5,
                        "publishable": "true",
                        "aggregation": "arithmetic_mean",
                        "mean_seconds": statistics.mean(values),
                        "std_seconds": statistics.stdev(values),
                        "min_seconds": min(values),
                        "max_seconds": max(values),
                        "cv": (
                            statistics.stdev(values)
                            / statistics.mean(values)
                        ),
                    }
                )
            write_csv(root / "results_samples.csv", samples)
            write_csv(root / "results_long.csv", aggregates)
            candidates, rejected, provenance = OVERLAY.main_candidates(
                root, None, strict_provenance=True
            )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(rejected, [])
            self.assertEqual(candidates[0].candidate_sha256, SHA)
            self.assertEqual(candidates[0].metrics["build"].minimum, 1.0)
            self.assertEqual(candidates[0].metrics["build"].maximum, 1.4)
            self.assertEqual(provenance["source_kind"], "main")

            aggregates[0]["min_seconds"] = 99
            write_csv(root / "results_long.csv", aggregates)
            candidates, rejected, _ = OVERLAY.main_candidates(
                root, None, strict_provenance=True
            )
            self.assertEqual(candidates, [])
            self.assertIn("minimum", rejected[0]["reason"])
            aggregates[0]["min_seconds"] = 1.0
            write_csv(root / "results_long.csv", aggregates)

            samples.pop()
            write_csv(root / "results_samples.csv", samples)
            candidates, rejected, _ = OVERLAY.main_candidates(
                root, None, strict_provenance=True
            )
            self.assertEqual(candidates, [])
            self.assertIn("4 raw samples", rejected[0]["reason"])

    def test_stability_audit_is_hash_bound_and_must_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples_path = root / "results_samples.csv"
            samples_path.write_text("raw-five-evidence\n", encoding="utf-8")
            timing = timing_stats()
            candidates = [
                OVERLAY.Candidate(
                    dataset=dataset,
                    function=function,
                    source_kind="main",
                    result_source=str(root),
                    candidate_sha256=SHA,
                    runtime_python_sha256=RUNTIME_SHA,
                    metrics={
                        metric: timing
                        for metric in OVERLAY.METRICS
                    },
                )
                for dataset in DATASETS
                for function in FUNCTIONS
            ]
            provenance = [
                {
                    "source_kind": "main",
                    "measurement_kind": "timing",
                    "result_dir": str(root),
                    "candidate_sha256": SHA,
                    "runtime_python_sha256": RUNTIME_SHA,
                    "runtime_package_is_symlink": False,
                    "results_samples_path": str(samples_path),
                    "results_samples_sha256": OVERLAY.sha256(samples_path),
                }
            ]
            evidence = OVERLAY.expected_stability_evidence(provenance)
            rows = []
            for candidate in candidates:
                stats = candidate.metrics["e2e"]
                raw5 = json.dumps(
                    list(stats.samples),
                    separators=(",", ":"),
                )
                rows.append(
                    {
                        "dataset": candidate.dataset,
                        "function": candidate.function,
                        "metric": "e2e",
                        "source_kind": "main",
                        "result_source": str(root.resolve()),
                        "evidence_path": str(samples_path.resolve()),
                        "evidence_sha256": OVERLAY.sha256(samples_path),
                        "sample_count": 5,
                        "raw5_seconds": raw5,
                        "samples_seconds": raw5,
                        "minimum_seconds": stats.minimum,
                        "arithmetic_mean_seconds": stats.mean,
                        "median_seconds": stats.median,
                        "maximum_seconds": stats.maximum,
                        "sample_std_seconds": stats.stdev,
                        "coefficient_of_variation": (
                            stats.coefficient_of_variation
                        ),
                        "max_over_min": (
                            stats.maximum / stats.minimum
                        ),
                        "max_over_median": stats.max_over_median,
                        "median_over_minimum": (
                            stats.median_over_minimum
                        ),
                        "variance_policy": (
                            OVERLAY.DEFAULT_VARIANCE_POLICY
                        ),
                        "paper_estimator": (
                            OVERLAY.PAPER_TIMING_ESTIMATOR
                        ),
                        "sample_std_and_cv_role": (
                            "reported_diagnostics_not_acceptance_gate"
                        ),
                        "max_over_median_limit": 5.0,
                        "median_over_min_limit": 3.0,
                        "stability_status": "pass",
                        "batch_acceptance_status": (
                            "accepted_complete_batch"
                        ),
                        "submission_seconds": stats.minimum,
                        "failure_reasons": "[]",
                        "batch_selection_status": "unique_input_batch",
                        "replacement_selection_basis": "",
                        "replacement_reason": "",
                        "original_result_source": "",
                        "original_evidence_path": "",
                        "original_evidence_sha256": "",
                        "original_stability_status": "",
                        "replacement_manifest_path": "",
                        "replacement_manifest_sha256": "",
                    }
                )
            rows_path = root / "stability.csv"
            write_csv(rows_path, rows)
            keys = sorted(
                (candidate.dataset, candidate.function, "e2e")
                for candidate in candidates
            )
            audit = {
                "status": "pass",
                "metric": "e2e",
                "paper_estimator": OVERLAY.PAPER_TIMING_ESTIMATOR,
                "acceptance_estimator_independent": True,
                "variance_policy": OVERLAY.DEFAULT_VARIANCE_POLICY,
                "sample_std_and_cv_role": (
                    "reported_diagnostics_not_acceptance_gate"
                ),
                "failed_batch_policy": (
                    "reject_entire_batch_do_not_select_across_batches"
                ),
                "batch_selection_policy": (
                    OVERLAY.BATCH_SELECTION_POLICY
                ),
                "gate_calibration": OVERLAY.GATE_CALIBRATION,
                "max_over_median_limit": 5.0,
                "median_over_min_limit": 3.0,
                "expected_samples_per_cell": 5,
                "audited_cells": 208,
                "unique_workload_keys": 208,
                "stable_cells": 208,
                "unstable_cells": 0,
                "failures": [],
                "main_result_dirs": [str(root.resolve())],
                "anchor_result_dirs": [],
                "candidate_sha256": SHA,
                "runtime_python_snapshot_sha256": RUNTIME_SHA,
                "runtime_package_is_symlink": False,
                "result_dir_identities": [
                    {
                        "result_dir": str(root.resolve()),
                        "candidate_sha256": SHA,
                        "runtime_python_snapshot_sha256": RUNTIME_SHA,
                        "runtime_package_is_symlink": False,
                    }
                ],
                "input_evidence": evidence,
                "input_evidence_sha256": (
                    OVERLAY.canonical_json_sha256(evidence)
                ),
                "override_manifest_path": "",
                "override_manifest_sha256": "",
                "batch_overrides_applied": [],
                "batch_overrides_applied_sha256": (
                    OVERLAY.canonical_json_sha256([])
                ),
                "audited_keys_sha256": (
                    OVERLAY.canonical_json_sha256(keys)
                ),
                "rows_csv_path": str(rows_path.resolve()),
                "rows_csv_sha256": OVERLAY.sha256(rows_path),
            }
            audit_path = root / "stability.json"
            audit_path.write_text(
                json.dumps(audit),
                encoding="utf-8",
            )

            accepted, selected = OVERLAY.validate_timing_stability_audit(
                audit_path,
                candidates,
                provenance,
                max_over_median_limit=5.0,
                median_over_min_limit=3.0,
            )
            self.assertEqual(accepted["audited_cells"], 208)
            self.assertEqual(len(selected), 208)
            calibration_without_hash_evidence = json.loads(
                json.dumps(audit)
            )
            calibration_without_hash_evidence["gate_calibration"].pop(
                "calibration_evidence"
            )
            audit_path.write_text(
                json.dumps(calibration_without_hash_evidence),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                OVERLAY.GateError,
                "gate calibration evidence differs",
            ):
                OVERLAY.validate_timing_stability_audit(
                    audit_path,
                    candidates,
                    provenance,
                    max_over_median_limit=5.0,
                    median_over_min_limit=3.0,
                )
            audit_path.write_text(
                json.dumps(audit),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                OVERLAY.GateError,
                "one unique original complete batch",
            ):
                OVERLAY.validate_timing_stability_audit(
                    audit_path,
                    [*candidates, candidates[0]],
                    provenance,
                    max_over_median_limit=5.0,
                    median_over_min_limit=3.0,
                )

            audit["status"] = "fail"
            audit_path.write_text(
                json.dumps(audit),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                OVERLAY.GateError,
                "status=.*expected",
            ):
                OVERLAY.validate_timing_stability_audit(
                    audit_path,
                    candidates,
                    provenance,
                    max_over_median_limit=5.0,
                    median_over_min_limit=3.0,
                )

            audit["status"] = "pass"
            audit["candidate_sha256"] = "c" * 64
            audit_path.write_text(
                json.dumps(audit),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                OVERLAY.GateError,
                "candidate SHA differs",
            ):
                OVERLAY.validate_timing_stability_audit(
                    audit_path,
                    candidates,
                    provenance,
                    max_over_median_limit=5.0,
                    median_over_min_limit=3.0,
                )

    def _legacy_explicit_replacement_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_dir = root / "original"
            replacement_dir = root / "replacement"
            original_dir.mkdir()
            replacement_dir.mkdir()
            original_evidence = original_dir / "results_samples.csv"
            replacement_evidence = replacement_dir / "results_samples.csv"
            original_evidence.write_text("original\n", encoding="utf-8")
            replacement_evidence.write_text("replacement\n", encoding="utf-8")
            original_stats = OVERLAY.summarize_timing_samples(
                [0.20, 0.21, 0.22, 0.23, 9.0],
                "original",
            )
            replacement_stats = OVERLAY.summarize_timing_samples(
                [0.21, 0.23, 0.30, 0.40, 0.41],
                "replacement",
            )
            self.assertEqual(original_stats.stability_status, "fail")
            self.assertEqual(replacement_stats.stability_status, "pass")
            candidates = [
                OVERLAY.Candidate(
                    dataset="soc-Slashdot0811",
                    function="EffectiveSize",
                    source_kind="main",
                    result_source=str(result_dir),
                    candidate_sha256=SHA,
                    runtime_python_sha256=RUNTIME_SHA,
                    metrics={
                        metric: stats for metric in OVERLAY.METRICS
                    },
                )
                for result_dir, stats in (
                    (original_dir, original_stats),
                    (replacement_dir, replacement_stats),
                )
            ]
            provenance = [
                {
                    "source_kind": "main",
                    "measurement_kind": "timing",
                    "result_dir": str(result_dir),
                    "candidate_sha256": SHA,
                    "runtime_python_sha256": RUNTIME_SHA,
                    "runtime_package_is_symlink": False,
                    "results_samples_path": str(evidence),
                    "results_samples_sha256": OVERLAY.sha256(evidence),
                }
                for result_dir, evidence in (
                    (original_dir, original_evidence),
                    (replacement_dir, replacement_evidence),
                )
            ]
            manifest = root / "replacement_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "replacements": [
                            {
                                "dataset": "soc-Slashdot0811",
                                "function": "EffectiveSize",
                                "metric": "e2e",
                                "original_result_source": str(
                                    original_dir.resolve()
                                ),
                                "replacement_result_source": str(
                                    replacement_dir.resolve()
                                ),
                                "reason": (
                                    "original complete batch failed the "
                                    "declared catastrophic-outlier guard"
                                ),
                                "selection_basis": (
                                    OVERLAY.REPLACEMENT_SELECTION_BASIS
                                ),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            reason = (
                "original complete batch failed the declared "
                "catastrophic-outlier guard"
            )
            override = {
                "dataset": "soc-Slashdot0811",
                "function": "EffectiveSize",
                "metric": "e2e",
                "selection_basis": OVERLAY.REPLACEMENT_SELECTION_BASIS,
                "reason": reason,
                "original_result_source": str(original_dir.resolve()),
                "original_evidence_path": str(original_evidence.resolve()),
                "original_evidence_sha256": OVERLAY.sha256(
                    original_evidence
                ),
                "original_stability_status": "fail",
                "original_failure_reasons": [
                    "max_over_median_exceeds_limit"
                ],
                "replacement_result_source": str(
                    replacement_dir.resolve()
                ),
                "replacement_evidence_path": str(
                    replacement_evidence.resolve()
                ),
                "replacement_evidence_sha256": OVERLAY.sha256(
                    replacement_evidence
                ),
                "replacement_stability_status": "pass",
            }
            stats = replacement_stats
            raw5 = json.dumps(
                list(stats.samples),
                separators=(",", ":"),
            )
            rows = [
                {
                    "dataset": "soc-Slashdot0811",
                    "function": "EffectiveSize",
                    "metric": "e2e",
                    "source_kind": "main",
                    "result_source": str(replacement_dir.resolve()),
                    "evidence_path": str(replacement_evidence.resolve()),
                    "evidence_sha256": OVERLAY.sha256(
                        replacement_evidence
                    ),
                    "sample_count": 5,
                    "raw5_seconds": raw5,
                    "samples_seconds": raw5,
                    "minimum_seconds": stats.minimum,
                    "arithmetic_mean_seconds": stats.mean,
                    "median_seconds": stats.median,
                    "maximum_seconds": stats.maximum,
                    "sample_std_seconds": stats.stdev,
                    "coefficient_of_variation": (
                        stats.coefficient_of_variation
                    ),
                    "max_over_min": stats.maximum / stats.minimum,
                    "max_over_median": stats.max_over_median,
                    "median_over_minimum": stats.median_over_minimum,
                    "variance_policy": OVERLAY.DEFAULT_VARIANCE_POLICY,
                    "paper_estimator": OVERLAY.PAPER_TIMING_ESTIMATOR,
                    "sample_std_and_cv_role": (
                        "reported_diagnostics_not_acceptance_gate"
                    ),
                    "max_over_median_limit": 5.0,
                    "median_over_min_limit": 2.0,
                    "stability_status": "pass",
                    "batch_acceptance_status": "accepted_complete_batch",
                    "submission_seconds": stats.minimum,
                    "failure_reasons": "[]",
                    "batch_selection_status": (
                        "explicit_failed_batch_replacement"
                    ),
                    "replacement_selection_basis": (
                        OVERLAY.REPLACEMENT_SELECTION_BASIS
                    ),
                    "replacement_reason": reason,
                    "original_result_source": str(original_dir.resolve()),
                    "original_evidence_path": str(
                        original_evidence.resolve()
                    ),
                    "original_evidence_sha256": OVERLAY.sha256(
                        original_evidence
                    ),
                    "original_stability_status": "fail",
                    "replacement_manifest_path": str(manifest.resolve()),
                    "replacement_manifest_sha256": OVERLAY.sha256(manifest),
                }
            ]
            rows_path = root / "stability.csv"
            write_csv(rows_path, rows)
            evidence = OVERLAY.expected_stability_evidence(provenance)
            identities = [
                {
                    "result_dir": str(result_dir.resolve()),
                    "candidate_sha256": SHA,
                    "runtime_python_snapshot_sha256": RUNTIME_SHA,
                    "runtime_package_is_symlink": False,
                }
                for result_dir in (original_dir, replacement_dir)
            ]
            audit = {
                "status": "pass",
                "metric": "e2e",
                "paper_estimator": OVERLAY.PAPER_TIMING_ESTIMATOR,
                "acceptance_estimator_independent": True,
                "variance_policy": OVERLAY.DEFAULT_VARIANCE_POLICY,
                "sample_std_and_cv_role": (
                    "reported_diagnostics_not_acceptance_gate"
                ),
                "failed_batch_policy": (
                    "reject_entire_batch_do_not_select_across_batches"
                ),
                "batch_selection_policy": OVERLAY.BATCH_SELECTION_POLICY,
                "max_over_median_limit": 5.0,
                "median_over_min_limit": 2.0,
                "expected_samples_per_cell": 5,
                "audited_cells": 1,
                "unique_workload_keys": 1,
                "stable_cells": 1,
                "unstable_cells": 0,
                "failures": [],
                "main_result_dirs": [
                    str(original_dir.resolve()),
                    str(replacement_dir.resolve()),
                ],
                "anchor_result_dirs": [],
                "candidate_sha256": SHA,
                "runtime_python_snapshot_sha256": RUNTIME_SHA,
                "runtime_package_is_symlink": False,
                "result_dir_identities": identities,
                "input_evidence": evidence,
                "input_evidence_sha256": (
                    OVERLAY.canonical_json_sha256(evidence)
                ),
                "override_manifest_path": str(manifest.resolve()),
                "override_manifest_sha256": OVERLAY.sha256(manifest),
                "batch_overrides_applied": [override],
                "batch_overrides_applied_sha256": (
                    OVERLAY.canonical_json_sha256([override])
                ),
                "audited_keys_sha256": OVERLAY.canonical_json_sha256(
                    [
                        (
                            "soc-Slashdot0811",
                            "EffectiveSize",
                            "e2e",
                        )
                    ]
                ),
                "rows_csv_path": str(rows_path.resolve()),
                "rows_csv_sha256": OVERLAY.sha256(rows_path),
            }
            audit_path = root / "stability.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            with mock.patch.object(
                OVERLAY,
                "EXPECTED_EGGPU_CELLS",
                1,
            ):
                record, selected = (
                    OVERLAY.validate_timing_stability_audit(
                        audit_path,
                        candidates,
                        provenance,
                        max_over_median_limit=5.0,
                        median_over_min_limit=2.0,
                    )
                )
            self.assertEqual(len(selected), 1)
            self.assertEqual(
                selected[0].result_source,
                str(replacement_dir),
            )
            self.assertEqual(len(record["batch_overrides_applied"]), 1)

    def test_main_memory_parser_requires_three_isolated_process_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata(root)
            gpu = [100.0, 110.0, 120.0]
            host = [1000.0, 1010.0, 1020.0]
            samples = []
            aggregates = []
            for metric, values in (
                ("memory_peak_gpu_proc_mb", gpu),
                ("memory_peak_rss_mb", host),
            ):
                for index, value in enumerate(values, start=1):
                    samples.append(
                        {
                            "dataset": "g0",
                            "function": "PageRank",
                            "baseline": "EGGPU",
                            "metric": metric,
                            "value": value,
                            "seconds": value,
                            "unit": "MiB",
                            "status": "ok",
                            "correctness": "digest",
                            "sample_index": index,
                            "sample_count": 3,
                            "measurement_phase": "memory",
                            "measurement_window": "isolated_memory_subprocess",
                        }
                    )
                aggregates.append(
                    {
                        "dataset": "g0",
                        "function": "PageRank",
                        "baseline": "EGGPU",
                        "metric": metric,
                        "value": statistics.mean(values),
                        "unit": "MiB",
                        "status": "ok",
                        "sample_count": 3,
                        "n_total": 3,
                        "n_valid": 3,
                        "publishable": "true",
                        "aggregation": "arithmetic_mean",
                        "mean_value": statistics.mean(values),
                        "std_value": statistics.stdev(values),
                        "measurement_phase": "memory",
                        "measurement_window": "isolated_memory_subprocess",
                    }
                )
            write_csv(root / "results_samples.csv", samples)
            write_csv(root / "results_long.csv", aggregates)
            run_metadata = json.loads(
                (root / "run_metadata.json").read_text(encoding="utf-8")
            )
            run_metadata["benchmark_args"] = {
                "repeat": 3,
                "warmup": 0,
                "easygraph_warmup": 2,
                "eggpu_execution_protocol": "steady-state",
                "measurement_mode": "memory",
            }
            (root / "run_metadata.json").write_text(
                json.dumps(run_metadata), encoding="utf-8"
            )

            candidates, rejected, provenance = OVERLAY.main_memory_candidates(
                root, None, strict_provenance=True
            )
            self.assertEqual(rejected, [])
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].gpu_peak_mb, (110.0, 10.0))
            self.assertEqual(candidates[0].host_rss_peak_mb, (1010.0, 10.0))
            self.assertEqual(candidates[0].runtime_python_sha256, RUNTIME_SHA)
            self.assertEqual(
                provenance["measurement_protocol"]["measurement_mode"],
                "memory",
            )

            samples[-1]["measurement_window"] = "algorithm_call"
            write_csv(root / "results_samples.csv", samples)
            candidates, rejected, _ = OVERLAY.main_memory_candidates(
                root, None, strict_provenance=True
            )
            self.assertEqual(candidates, [])
            self.assertIn("2 raw samples", rejected[0]["reason"])

    def test_memory_overlay_replaces_memory_and_records_row_identity(self):
        rows = base_ledger()
        candidate = OVERLAY.MemoryCandidate(
            dataset="g0",
            function="PageRank",
            source_kind="main",
            result_source="/memory",
            candidate_sha256=SHA,
            runtime_python_sha256=RUNTIME_SHA,
            gpu_peak_mb=(111.0, 1.5),
            host_rss_peak_mb=(222.0, 2.5),
            measurement_window="isolated_memory_subprocess",
        )
        fields, output, comparisons, rejected = OVERLAY.overlay_memory(
            list(rows[0]), rows, [candidate], 1
        )
        row = next(
            item
            for item in output
            if item["dataset"] == "g0"
            and item["function"] == "PageRank"
            and item["baseline"] == "EGGPU"
        )
        self.assertIn("memory_candidate_sha256", fields)
        self.assertEqual(row["gpu_peak_mb_mean"], "111.0")
        self.assertEqual(row["host_rss_peak_mb_std"], "2.5")
        self.assertEqual(row["memory_sample_count"], "3")
        self.assertEqual(row["memory_result_source"], "/memory")
        self.assertEqual(row["memory_candidate_sha256"], SHA)
        self.assertEqual(
            row["memory_runtime_python_snapshot_sha256"], RUNTIME_SHA
        )
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(rejected, [])

    def test_unified_identity_requires_208_matching_timing_and_memory_cells(self):
        timing = []
        memory = []
        for dataset in DATASETS:
            for function in FUNCTIONS:
                timing.append(
                    OVERLAY.Candidate(
                        dataset=dataset,
                        function=function,
                        source_kind="main",
                        result_source="/timing",
                        candidate_sha256=SHA,
                        metrics={
                            metric: timing_stats()
                            for metric in OVERLAY.METRICS
                        },
                        runtime_python_sha256=RUNTIME_SHA,
                    )
                )
                memory.append(
                    OVERLAY.MemoryCandidate(
                        dataset=dataset,
                        function=function,
                        source_kind="main",
                        result_source="/memory",
                        candidate_sha256=SHA,
                        runtime_python_sha256=RUNTIME_SHA,
                        gpu_peak_mb=(10.0, 1.0),
                        host_rss_peak_mb=(20.0, 2.0),
                        measurement_window="isolated_memory_subprocess",
                    )
                )
        identity = OVERLAY.validate_unified_candidate_identity(timing, memory)
        self.assertEqual(identity["candidate_sha256"], SHA)
        self.assertEqual(
            identity["runtime_python_snapshot_sha256"], RUNTIME_SHA
        )

        memory[-1] = OVERLAY.MemoryCandidate(
            **{
                **memory[-1].__dict__,
                "runtime_python_sha256": "c" * 64,
            }
        )
        with self.assertRaisesRegex(ValueError, "one common|multiple|differs"):
            OVERLAY.validate_unified_candidate_identity(timing, memory)

    def test_unified_provenance_rejects_symlinked_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata(root)
            payload = json.loads(
                (root / "run_metadata.json").read_text(encoding="utf-8")
            )
            payload["runtime_python_snapshot"]["package_is_symlink"] = True
            (root / "run_metadata.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(OVERLAY.GateError, "symlinked"):
                OVERLAY.runtime_python_snapshot_from_result_dir(
                    root, required=True
                )

    def test_anchor_parser_requires_five_fresh_single_measure_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            scaling_metadata(root)
            load = [2.0, 2.1, 2.2, 2.3, 2.4]
            e2e = [1.0, 1.1, 1.2, 1.3, 1.4]
            kernel = [0.5, 0.6, 0.7, 0.8, 0.9]
            first = [4.0, 4.1, 4.2, 4.3, 4.4]

            def stats(values):
                return {
                    "samples": values,
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values),
                }

            aggregate = {
                "status": "ok",
                "dataset": "g0",
                "function": "PageRank",
                "timing_process_samples": 5,
                "result_validation": {"status": "pass", "failures": []},
                "result": {"sha256": "result"},
                "load": stats(load),
                "first_use_e2e": stats(first),
                "steady_e2e": stats(e2e),
                "steady_kernel": stats(kernel),
                **scaling_raw_runtime(root),
            }
            (raw / "g0_PageRank_timing.json").write_text(
                json.dumps(aggregate), encoding="utf-8"
            )
            for index in range(1, 6):
                worker = {
                    "status": "ok",
                    "result_validation": {"status": "pass", "failures": []},
                    "result": {"sha256": "result"},
                    "steady_e2e": {"samples": [e2e[index - 1]]},
                    "steady_kernel": {"samples": [kernel[index - 1]]},
                    **scaling_raw_runtime(root),
                }
                (raw / f"g0_PageRank_timing_{index}.json").write_text(
                    json.dumps(worker), encoding="utf-8"
                )
            write_csv(
                root / "scaling_all.csv",
                [
                    {
                        "dataset": "g0",
                        "function": "PageRank",
                        "measurement": "timing",
                        "status": "ok",
                        "result_validation": "pass",
                        "timing_process_samples": 5,
                        "load_seconds": statistics.mean(load),
                        "load_stdev_seconds": statistics.stdev(load),
                        "steady_e2e_mean": statistics.mean(e2e),
                        "steady_e2e_stdev": statistics.stdev(e2e),
                        "steady_kernel_mean": statistics.mean(kernel),
                        "steady_kernel_stdev": statistics.stdev(kernel),
                    }
                ],
            )
            candidates, rejected, provenance = OVERLAY.anchor_candidates(
                root, None, strict_provenance=True
            )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(rejected, [])
            self.assertEqual(
                provenance["anchor_metadata_schema"], "run_metadata_v10"
            )
            self.assertTrue(
                provenance["anchor_metadata_path"].endswith(
                    "run_metadata.json"
                )
            )
            self.assertEqual(provenance["protocol_path"], "")
            self.assertEqual(
                provenance["measurement_protocol"]["warmup"], 1
            )
            self.assertEqual(
                provenance["measurement_protocol"]["memory_repeat"], 3
            )
            self.assertEqual(
                provenance["measurement_protocol"][
                    "runtime_python_snapshot_sha256"
                ],
                RUNTIME_SHA,
            )

            run_metadata_path = root / "run_metadata.json"
            run_metadata = json.loads(
                run_metadata_path.read_text(encoding="utf-8")
            )
            run_metadata["warmup"] = 2
            run_metadata_path.write_text(
                json.dumps(run_metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(OVERLAY.GateError, "warmup=2"):
                OVERLAY.anchor_candidates(
                    root, None, strict_provenance=True
                )
            run_metadata["warmup"] = 1
            run_metadata["python_origins"]["easygraph"] = (
                "/outside/easygraph/__init__.py"
            )
            run_metadata_path.write_text(
                json.dumps(run_metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(OVERLAY.GateError, "outside"):
                OVERLAY.anchor_candidates(
                    root, None, strict_provenance=True
                )
            run_metadata["python_origins"]["easygraph"] = (
                run_metadata["repository_runtime_provenance"]["modules"][
                    "easygraph"
                ]["module_origin_resolved"]
            )
            run_metadata_path.write_text(
                json.dumps(run_metadata), encoding="utf-8"
            )

            worker_path = raw / "g0_PageRank_timing_5.json"
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            worker["runtime_identity"]["runtime_python_digest"] = "c" * 64
            worker_path.write_text(json.dumps(worker), encoding="utf-8")
            candidates, rejected, _ = OVERLAY.anchor_candidates(
                root, None, strict_provenance=True
            )
            self.assertEqual(candidates, [])
            self.assertIn(
                "raw runtime Python digest differs",
                rejected[0]["reason"],
            )

            worker["runtime_identity"][
                "runtime_python_digest"
            ] = RUNTIME_SHA
            worker["steady_e2e"]["samples"].append(99)
            worker_path.write_text(json.dumps(worker), encoding="utf-8")
            candidates, rejected, _ = OVERLAY.anchor_candidates(
                root, None, strict_provenance=True
            )
            self.assertEqual(candidates, [])
            self.assertIn("expected 1", rejected[0]["reason"])

    def test_legacy_anchor_without_protocol_remains_nonstrict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata(root)
            candidate_sha, _source = OVERLAY.candidate_sha_from_result_dir(
                root, None
            )
            runtime_sha, _source, _symlink = (
                OVERLAY.runtime_python_snapshot_from_result_dir(
                    root, required=False
                )
            )
            evidence = OVERLAY.anchor_provenance_evidence(
                root,
                candidate_sha,
                runtime_sha,
                strict_provenance=False,
                measurement_kind="timing",
            )
            self.assertEqual(
                evidence["anchor_metadata_schema"], "unrecorded_legacy"
            )
            self.assertIn(
                "does not encode the warmup",
                evidence["protocol_evidence"],
            )

    def test_anchor_memory_parser_binds_three_worker_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            metadata(root)
            (root / "protocol.json").write_text(
                json.dumps(
                    {
                        "aggregate": "arithmetic_mean",
                        "candidate_sha256": SHA,
                        "repeat": 5,
                        "timing_processes": 0,
                        "warmup": 1,
                        "memory_repeat": 3,
                        "first_use_calls_per_process": 1,
                        "measured_call_ordinal": 3,
                        "measured_calls_per_process": 1,
                        "runtime_python_snapshot": {
                            "digest": RUNTIME_SHA,
                            "package_is_symlink": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            gpu = [300.0, 310.0, 320.0]
            host = [1300.0, 1310.0, 1320.0]
            rows = []
            for index, (gpu_value, host_value) in enumerate(
                zip(gpu, host), start=1
            ):
                rows.append(
                    {
                        "status": "ok",
                        "dataset": "g0",
                        "function": "PageRank",
                        "measurement": "memory",
                        "gpu_proc_peak_mb": gpu_value,
                        "gpu_proc_peak_delta_mb": gpu_value,
                        "rss_peak_mb": host_value,
                        "rss_peak_delta_mb": host_value - 5,
                        "result_validation": "pass",
                    }
                )
                worker = {
                    "status": "ok",
                    "dataset": "g0",
                    "function": "PageRank",
                    "measurement": "memory",
                    "result_validation": {"status": "pass", "failures": []},
                    "result": {"sha256": "result"},
                    "memory": {
                        "gpu_proc_peak_mb": gpu_value,
                        "gpu_proc_peak_delta_mb": gpu_value,
                        "rss_mb": host_value,
                        "rss_peak_delta_mb": host_value - 5,
                        "memory_monitor_origin": (
                            "coordinator_process_child_tree"
                        ),
                        "measurement_window": (
                            "isolated_worker_process_full_lifetime"
                        ),
                        "monitor_rss_samples": 4,
                        "monitor_gpu_proc_samples": 4,
                    },
                }
                (raw / f"g0_PageRank_memory_{index}.json").write_text(
                    json.dumps(worker), encoding="utf-8"
                )
            write_csv(root / "scaling_all.csv", rows)

            candidates, rejected, provenance = (
                OVERLAY.anchor_memory_candidates(
                    root, None, strict_provenance=True
                )
            )
            self.assertEqual(rejected, [])
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].gpu_peak_mb, (310.0, 10.0))
            self.assertEqual(candidates[0].host_rss_peak_mb, (1310.0, 10.0))
            self.assertEqual(
                provenance["measurement_protocol"]["memory_repeat"], 3
            )
            self.assertEqual(
                provenance["anchor_metadata_schema"],
                "protocol_json_legacy",
            )
            self.assertEqual(
                provenance["measurement_protocol"]["warmup"], 1
            )
            self.assertEqual(
                provenance["measurement_protocol"][
                    "extra_untimed_warmups_per_process"
                ],
                1,
            )

            worker_path = raw / "g0_PageRank_memory_3.json"
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            worker["memory"]["monitor_gpu_proc_samples"] = 0
            worker_path.write_text(json.dumps(worker), encoding="utf-8")
            candidates, rejected, _ = OVERLAY.anchor_memory_candidates(
                root, None, strict_provenance=True
            )
            self.assertEqual(candidates, [])
            self.assertIn("monitor_gpu_proc_samples is zero", rejected[0]["reason"])

    def test_invalid_ledger_cardinality_is_rejected(self):
        rows = base_ledger()
        with self.assertRaisesRegex(ValueError, "expected 1664"):
            OVERLAY.validate_ledger(list(rows[0]), rows[:-1])


if __name__ == "__main__":
    unittest.main()
