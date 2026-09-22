import importlib.util
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))
MODULE_PATH = BENCHMARK_DIR / "prepare_vldb_uniform_assets.py"
SPEC = importlib.util.spec_from_file_location("prepare_uniform_assets", MODULE_PATH)
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


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


def unified_ledger() -> pd.DataFrame:
    timing_samples = [1.0, 1.05, 1.10, 1.15, 1.20]
    timing_min = min(timing_samples)
    timing_median = statistics.median(timing_samples)
    timing_mean = statistics.mean(timing_samples)
    timing_std = statistics.stdev(timing_samples)
    timing_max = max(timing_samples)
    timing_cv = timing_std / timing_mean
    rows = []
    for dataset in DATASETS:
        for function in FUNCTIONS:
            for baseline in ("EGGPU", "GraphScope"):
                row = {
                    "dataset": dataset,
                    "function": function,
                    "baseline": baseline,
                    "execution_status": "ok",
                    "validation_status": "pass",
                    "sample_count": 5,
                    "candidate_sha256": SHA if baseline == "EGGPU" else "",
                    "runtime_python_snapshot_sha256": (
                        RUNTIME_SHA if baseline == "EGGPU" else ""
                    ),
                    "gpu_peak_mb_mean": 100.0,
                    "gpu_peak_mb_std": 1.0,
                    "memory_sample_count": 3,
                    "host_rss_peak_mb_mean": 200.0,
                    "host_rss_peak_mb_std": 2.0,
                    "memory_measurement_window": (
                        "isolated_memory_subprocess"
                    ),
                    "memory_result_source": "/memory/main",
                    "memory_candidate_sha256": (
                        SHA if baseline == "EGGPU" else ""
                    ),
                    "memory_runtime_python_snapshot_sha256": (
                        RUNTIME_SHA if baseline == "EGGPU" else ""
                    ),
                }
                for metric in PREPARE.METRICS:
                    if baseline == "EGGPU":
                        row[f"{metric}_paper_seconds"] = timing_min
                        row[f"{metric}_estimator"] = (
                            PREPARE.EGGPU_TIMING_ESTIMATOR
                        )
                    else:
                        row[f"{metric}_paper_seconds"] = timing_mean
                        row[f"{metric}_estimator"] = "arithmetic_mean"
                    row[f"{metric}_raw_min_seconds"] = timing_min
                    row[f"{metric}_raw_median_seconds"] = timing_median
                    row[f"{metric}_raw_mean_seconds"] = timing_mean
                    row[f"{metric}_std_seconds"] = timing_std
                    row[f"{metric}_raw_max_seconds"] = timing_max
                    row[f"{metric}_coefficient_of_variation"] = timing_cv
                    row[f"{metric}_max_over_median"] = (
                        timing_max / timing_median
                    )
                    row[f"{metric}_median_over_minimum"] = (
                        timing_median / timing_min
                    )
                    row[f"{metric}_variance_policy"] = (
                        PREPARE.EGGPU_VARIANCE_POLICY
                    )
                    row[f"{metric}_stability_status"] = "pass"
                rows.append(row)
    return pd.DataFrame(rows)


class PrepareUniformAssetsTests(unittest.TestCase):
    def test_canonical_timing_evidence_materializes_all_raw5_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_dir = root / "timing_main"
            result_dir.mkdir()
            overlay_path = root / "overlay.json"
            overlay_path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "source_kind": "main",
                                "measurement_kind": "timing",
                                "result_dir": str(result_dir),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            output.mkdir()
            ledger = unified_ledger()
            eggpu_rows = ledger["baseline"].eq("EGGPU")
            ledger.loc[eggpu_rows, "timing_result_source"] = str(result_dir)
            samples = [1.0, 1.05, 1.10, 1.15, 1.20]
            candidates = []
            for dataset in DATASETS:
                for function in FUNCTIONS:
                    metrics = {
                        metric: PREPARE.overlay_tools.summarize_timing_samples(
                            samples,
                            metric,
                            max_over_median_limit=5.0,
                            median_over_min_limit=3.0,
                        )
                        for metric in ("build", "e2e", "kernel")
                    }
                    candidates.append(
                        PREPARE.overlay_tools.Candidate(
                            dataset=dataset,
                            function=function,
                            source_kind="main",
                            result_source=str(result_dir.resolve()),
                            candidate_sha256=SHA,
                            metrics=metrics,
                            runtime_python_sha256=RUNTIME_SHA,
                        )
                    )
            provenance = {
                "candidate_sha256": SHA,
                "runtime_python_sha256": RUNTIME_SHA,
                "runtime_package_is_symlink": False,
            }
            superseded_incomplete = [
                {
                    "dataset": DATASETS[0],
                    "function": FUNCTIONS[0],
                    "source_kind": "main",
                    "result_source": str(result_dir.resolve()),
                    "reason": "build: 1 raw samples, expected 5",
                }
            ]
            with mock.patch.object(
                PREPARE.overlay_tools,
                "main_candidates",
                return_value=(
                    candidates,
                    superseded_incomplete,
                    provenance,
                ),
            ):
                result = PREPARE.write_canonical_timing_evidence(
                    ledger,
                    overlay_path,
                    output,
                    SHA,
                    RUNTIME_SHA,
                )

            raw = pd.read_csv(output / result["raw_samples"])
            policy = json.loads(
                (output / result["policy"]).read_text(encoding="utf-8")
            )
            self.assertEqual(len(raw), 2080)
            self.assertEqual(
                raw.groupby(["dataset", "function", "metric"]).size().unique().tolist(),
                [5],
            )
            self.assertEqual(policy["acceptance_metrics"], ["e2e"])
            self.assertEqual(policy["raw_sample_groups"], 416)
            self.assertEqual(
                policy["superseded_incomplete_candidates"],
                superseded_incomplete,
            )
            self.assertEqual(
                policy["raw_samples_sha256"],
                PREPARE.sha256(output / result["raw_samples"]),
            )

    def test_unified_memory_audit_binds_all_208_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = unified_ledger()
            source = root / "final_13_cell_outcome_ledger.csv"
            ledger.to_csv(source, index=False)
            audit_path = root / "overlay.json"
            memory_cells_path = root / "overlay.memory_cells.csv"
            pd.DataFrame(
                [
                    {
                        "dataset": dataset,
                        "function": function,
                        "candidate_sha256": SHA,
                        "runtime_python_snapshot_sha256": RUNTIME_SHA,
                    }
                    for dataset in DATASETS
                    for function in FUNCTIONS
                ]
            ).to_csv(memory_cells_path, index=False)
            audit = {
                "candidate_mode": "unified-timing-memory",
                "memory_replacement_count": 208,
                "expected_memory_replacements": 208,
                "candidate_sha256": [SHA],
                "runtime_python_snapshot_sha256": [RUNTIME_SHA],
                "output_ledger_sha256": PREPARE.sha256(source),
                "rejected_candidates": [],
                "sources": [
                    {
                        "source_kind": "main",
                        "measurement_kind": "memory",
                        "candidate_sha256": SHA,
                        "runtime_python_sha256": RUNTIME_SHA,
                        "runtime_package_is_symlink": False,
                    }
                ],
            }
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            result = PREPARE.audit_unified_memory_provenance(
                ledger,
                source,
                audit_path,
                SHA,
                RUNTIME_SHA,
            )
            self.assertEqual(result["status"], "pass_unified_candidate")
            self.assertEqual(result["row_count"], 208)
            self.assertEqual(result["sample_count_per_cell"], 3)
            self.assertTrue(result["measured_with_timing_candidate"])
            self.assertEqual(result["row_level_compiled_binary_sha256"], SHA)

    def test_unified_memory_audit_rejects_one_runtime_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = unified_ledger()
            target = ledger[
                ledger["baseline"].eq("EGGPU")
            ].index[-1]
            ledger.at[
                target, "memory_runtime_python_snapshot_sha256"
            ] = "c" * 64
            source = root / "final_13_cell_outcome_ledger.csv"
            ledger.to_csv(source, index=False)
            audit_path = root / "overlay.json"
            audit_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different Python runtime"):
                PREPARE.audit_unified_memory_provenance(
                    ledger,
                    source,
                    audit_path,
                    SHA,
                    RUNTIME_SHA,
                )

    def test_candidate_audit_requires_one_runtime_in_unified_mode(self):
        ledger = unified_ledger()
        result = PREPARE.audit_candidate_ledger(
            ledger,
            SHA,
            require_runtime_provenance=True,
        )
        self.assertEqual(
            result["timing_estimator"],
            PREPARE.EGGPU_TIMING_ESTIMATOR,
        )
        self.assertEqual(result["runtime_python_snapshot_sha256"], RUNTIME_SHA)
        target = ledger[ledger["baseline"].eq("EGGPU")].index[-1]
        ledger.at[target, "runtime_python_snapshot_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "one frozen Python runtime"):
            PREPARE.audit_candidate_ledger(
                ledger,
                SHA,
                require_runtime_provenance=True,
            )

    def test_candidate_audit_rejects_wrong_five_run_minimum(self):
        ledger = unified_ledger()
        target = ledger[ledger["baseline"].eq("EGGPU")].index[0]
        ledger.at[target, "e2e_paper_seconds"] = 0.99
        with self.assertRaisesRegex(ValueError, "five-run minimum"):
            PREPARE.audit_candidate_ledger(
                ledger,
                SHA,
                require_runtime_provenance=True,
            )

    def test_uniform_ledger_preserves_each_system_estimator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            ledger = unified_ledger()
            graphscope = ledger[
                ledger["baseline"].eq("GraphScope")
            ].index[0]
            ledger.at[graphscope, "e2e_paper_seconds"] = 7.0
            ledger.at[graphscope, "e2e_raw_mean_seconds"] = 8.0
            ledger.at[
                graphscope, "e2e_estimator"
            ] = "frozen_external_median"
            source = root / "source.csv"
            ledger.to_csv(source, index=False)

            prepared = PREPARE.make_uniform_ledger(source, output)

            preserved_graphscope = prepared.loc[graphscope]
            self.assertEqual(
                preserved_graphscope["e2e_estimator"],
                "frozen_external_median",
            )
            self.assertEqual(
                float(preserved_graphscope["e2e_paper_seconds"]),
                7.0,
            )
            eggpu = prepared[prepared["baseline"].eq("EGGPU")].iloc[0]
            self.assertEqual(
                eggpu["e2e_estimator"],
                PREPARE.EGGPU_TIMING_ESTIMATOR,
            )
            self.assertEqual(
                float(eggpu["e2e_paper_seconds"]),
                float(eggpu["e2e_raw_min_seconds"]),
            )

    def test_protocol_copy_accepts_explicit_timing_and_memory_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            source_ledger = root / "ledger.csv"
            source_ledger.write_text("ledger\n", encoding="utf-8")
            audit_path = root / "overlay.json"
            timing_cells = root / "overlay.cells.csv"
            memory_cells = root / "overlay.memory_cells.csv"
            cell_rows = [
                {
                    "dataset": dataset,
                    "function": function,
                    "candidate_sha256": SHA,
                    "runtime_python_snapshot_sha256": RUNTIME_SHA,
                }
                for dataset in DATASETS
                for function in FUNCTIONS
            ]
            pd.DataFrame(cell_rows).to_csv(timing_cells, index=False)
            pd.DataFrame(cell_rows).to_csv(memory_cells, index=False)
            stability_json = root / "timing_stability.json"
            stability_rows = root / "timing_stability.csv"
            stability_raw = root / "timing_raw.csv"
            stability_raw.write_text("raw5\n", encoding="utf-8")
            raw_inventory = [
                {
                    "source_kind": "main",
                    "result_source": str(root),
                    "evidence_path": str(stability_raw),
                    "evidence_sha256": PREPARE.sha256(stability_raw),
                }
            ]
            stability_json.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "variance_policy": (
                            PREPARE.EGGPU_VARIANCE_POLICY
                        ),
                        "paper_estimator": (
                            PREPARE.EGGPU_TIMING_ESTIMATOR
                        ),
                        "batch_selection_policy": (
                            PREPARE.EGGPU_BATCH_SELECTION_POLICY
                        ),
                        "gate_calibration": (
                            PREPARE.EGGPU_GATE_CALIBRATION
                        ),
                        "audited_cells": 208,
                        "candidate_sha256": SHA,
                        "runtime_python_snapshot_sha256": RUNTIME_SHA,
                        "max_over_median_limit": (
                            PREPARE.EGGPU_MAX_OVER_MEDIAN_LIMIT
                        ),
                        "median_over_min_limit": (
                            PREPARE.EGGPU_MEDIAN_OVER_MIN_LIMIT
                        ),
                        "batch_overrides_applied": [],
                        "input_evidence": raw_inventory,
                        "input_evidence_sha256": (
                            PREPARE.canonical_json_sha256(raw_inventory)
                        ),
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(cell_rows).to_csv(stability_rows, index=False)

            sources = []

            def add_source(
                kind,
                measurement,
                protocol,
                *,
                use_anchor_metadata=False,
            ):
                source_dir = root / f"{measurement}_{kind}"
                source_dir.mkdir()
                evidence_name = (
                    "run_metadata.json"
                    if kind == "main" or use_anchor_metadata
                    else "protocol.json"
                )
                evidence = source_dir / evidence_name
                evidence.write_text(
                    json.dumps({"kind": kind, "measurement": measurement}),
                    encoding="utf-8",
                )
                record = {
                    "source_kind": kind,
                    "measurement_kind": measurement,
                    "result_dir": str(source_dir),
                    "candidate_sha256": SHA,
                    "runtime_python_sha256": RUNTIME_SHA,
                    "runtime_package_is_symlink": False,
                    "measurement_protocol": protocol,
                }
                if kind == "main":
                    record["run_metadata_path"] = str(evidence)
                    record["run_metadata_sha256"] = PREPARE.sha256(evidence)
                elif use_anchor_metadata:
                    record["anchor_metadata_path"] = str(evidence)
                    record["anchor_metadata_sha256"] = PREPARE.sha256(
                        evidence
                    )
                    record["anchor_metadata_schema"] = "run_metadata_v10"
                else:
                    record["protocol_path"] = str(evidence)
                    record["protocol_json_sha256"] = PREPARE.sha256(evidence)
                sources.append(record)

            add_source(
                "main",
                "timing",
                {
                    "repeat": 5,
                    "warmup": 2,
                    "easygraph_warmup": 2,
                    "eggpu_execution_protocol": "steady-state",
                    "measurement_mode": "timing",
                    "measured_call_ordinal": 3,
                    "aggregation": "arithmetic_mean",
                    "paper_estimator": PREPARE.EGGPU_TIMING_ESTIMATOR,
                    "independent_process_samples": 5,
                    "stability_metric": "e2e",
                    "variance_policy": PREPARE.EGGPU_VARIANCE_POLICY,
                    "max_over_median_limit": (
                        PREPARE.EGGPU_MAX_OVER_MEDIAN_LIMIT
                    ),
                    "median_over_min_limit": (
                        PREPARE.EGGPU_MEDIAN_OVER_MIN_LIMIT
                    ),
                },
            )
            add_source(
                "anchor",
                "timing",
                {
                    "aggregate": "arithmetic_mean",
                    "repeat": 5,
                    "timing_processes": 0,
                    "warmup": 1,
                    "memory_repeat": 3,
                    "first_use_calls_per_process": 1,
                    "extra_untimed_warmups_per_process": 1,
                    "measured_call_ordinal": 3,
                    "measured_calls_per_process": 1,
                    "candidate_sha256": SHA,
                    "runtime_python_snapshot_sha256": RUNTIME_SHA,
                    "paper_estimator": PREPARE.EGGPU_TIMING_ESTIMATOR,
                    "stability_metric": "e2e",
                    "variance_policy": PREPARE.EGGPU_VARIANCE_POLICY,
                    "max_over_median_limit": (
                        PREPARE.EGGPU_MAX_OVER_MEDIAN_LIMIT
                    ),
                    "median_over_min_limit": (
                        PREPARE.EGGPU_MEDIAN_OVER_MIN_LIMIT
                    ),
                },
                use_anchor_metadata=True,
            )
            add_source(
                "main",
                "memory",
                {
                    "repeat": 3,
                    "measurement_mode": "memory",
                    "aggregation": "arithmetic_mean",
                    "independent_process_samples": 3,
                    "measurement_window": "isolated_memory_subprocess",
                },
            )
            add_source(
                "anchor",
                "memory",
                {
                    "memory_repeat": 3,
                    "repeat": 5,
                    "timing_processes": 0,
                    "warmup": 1,
                    "memory_measurement_window": (
                        "isolated_worker_process_full_lifetime"
                    ),
                    "independent_process_samples": 3,
                    "candidate_sha256": SHA,
                    "runtime_python_snapshot_sha256": RUNTIME_SHA,
                },
            )
            audit_path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "candidate_mode": "unified-timing-memory",
                        "timing_replacement_count": 208,
                        "expected_timing_replacements": 208,
                        "memory_replacement_count": 208,
                        "expected_memory_replacements": 208,
                        "candidate_sha256": [SHA],
                        "runtime_python_snapshot_sha256": [RUNTIME_SHA],
                        "output_ledger_sha256": PREPARE.sha256(source_ledger),
                        "rejected_candidates": [],
                        "superseded_rejection_count": 0,
                        "sources": sources,
                        "timing_stability_audit": {
                            "status": "pass",
                            "variance_policy": (
                                PREPARE.EGGPU_VARIANCE_POLICY
                            ),
                            "paper_estimator": (
                                PREPARE.EGGPU_TIMING_ESTIMATOR
                            ),
                            "batch_selection_policy": (
                                PREPARE.EGGPU_BATCH_SELECTION_POLICY
                            ),
                            "gate_calibration": (
                                PREPARE.EGGPU_GATE_CALIBRATION
                            ),
                            "audited_cells": 208,
                            "candidate_sha256": SHA,
                            "runtime_python_snapshot_sha256": RUNTIME_SHA,
                            "max_over_median_limit": (
                                PREPARE.EGGPU_MAX_OVER_MEDIAN_LIMIT
                            ),
                            "median_over_min_limit": (
                                PREPARE.EGGPU_MEDIAN_OVER_MIN_LIMIT
                            ),
                            "audit_path": str(stability_json),
                            "audit_sha256": PREPARE.sha256(stability_json),
                            "rows_csv_path": str(stability_rows),
                            "rows_csv_sha256": PREPARE.sha256(
                                stability_rows
                            ),
                            "override_manifest_path": "",
                            "override_manifest_sha256": "",
                            "batch_overrides_applied": [],
                            "batch_overrides_applied_sha256": (
                                PREPARE.canonical_json_sha256([])
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = PREPARE.copy_and_audit_protocol_evidence(
                audit_path,
                source_ledger,
                output,
                SHA,
                expected_runtime_sha256=RUNTIME_SHA,
                unified_memory=True,
                release_label="V10",
            )
            self.assertEqual(result["replacement_count"], 208)
            self.assertEqual(result["memory_replacement_count"], 208)
            self.assertEqual(
                result["source_kind_counts"],
                {
                    "timing_main": 1,
                    "timing_anchor": 1,
                    "memory_main": 1,
                    "memory_anchor": 1,
                },
            )
            self.assertEqual(len(result["sources"]), 4)
            self.assertEqual(
                result["timing_stability_audit"]["paper_estimator"],
                PREPARE.EGGPU_TIMING_ESTIMATOR,
            )
            self.assertEqual(
                result["timing_stability_audit"]["raw_evidence_count"],
                1,
            )
            self.assertEqual(
                len(
                    result["timing_stability_audit"][
                        "batch_overrides_applied"
                    ]
                ),
                0,
            )
            self.assertEqual(
                result["timing_stability_audit"][
                    "override_manifest_path"
                ],
                "",
            )
            timing_anchor = next(
                source
                for source in result["sources"]
                if source["source_kind"] == "anchor"
                and source["measurement_kind"] == "timing"
            )
            self.assertEqual(
                timing_anchor["anchor_metadata_schema"],
                "run_metadata_v10",
            )


if __name__ == "__main__":
    unittest.main()
