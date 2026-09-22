from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARKING = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = BENCHMARKING.parent
WRITING_ROOT = EVALUATION_ROOT.parent / "writing"
V14 = WRITING_ROOT / "EGGPU_FINAL_EXPERIMENT_ASSETS_14_UNIFORM_MIN_20260730"
DERIVER = BENCHMARKING / "derive_v15_paper_sync_facts_20260730.py"
HEADLINE_GENERATOR = BENCHMARKING / "generate_headline_results_20260730.py"
MIXED_MANIFEST = "MIXED_ESTIMATOR_FIVE_RUN_MANIFEST.json"
MIXED_RAW = "five_run_raw_samples.csv"
MIXED_AUDIT = "mixed_estimator_audit.csv"
MIXED_HEADLINE_SCHEMA = "eggpu_headline_results_v2_mixed_estimator"

FIXTURE_INPUTS = (
    "FINAL_13_NUMERICAL_RESULTS.md",
    "final_13_cell_outcome_ledger.csv",
    "final_13_pairwise_sota_details.csv",
    "pairwise_baseline_13_exact_summary.csv",
    "final_13_numeric_summary.json",
    "paper_table_main_compact_best_competitor.tex",
    "paper_table_main_compact_pairwise_speedup.tex",
    "UNIFORM_MINIMUM_OF_FIVE_MANIFEST.json",
    "uniform_minimum_raw_samples.csv",
    "paper_table_category_13_summary.csv",
    "category_time_by_baseline_3panel.csv",
    "category_time_by_baseline_3panel.metadata.json",
    "category_time_by_baseline_3panel.pdf",
    "category_time_by_baseline_3panel.png",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retarget_headline_manifest(asset_dir: Path) -> Path:
    path = asset_dir / "paper_table_headline_results_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_asset_directory_name"] = asset_dir.name
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def weighted_geomean(rows: list[dict[str, str]], column: str) -> float | None:
    weighted = []
    denominator = 0
    for row in rows:
        common = int(row["common_pairs"])
        if common:
            weighted.append(common * math.log(float(row[column])))
            denominator += common
    if not denominator:
        return None
    return math.exp(math.fsum(weighted) / denominator)


class V15PaperSyncFactsTest(unittest.TestCase):
    temporary: tempfile.TemporaryDirectory[str]
    fixture: Path
    facts: dict

    @classmethod
    def setUpClass(cls) -> None:
        if not V14.is_dir():
            raise AssertionError(f"required V14 fixture is absent: {V14}")
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="eggpu_paper_sync_facts_test_"
        )
        cls.fixture = (
            Path(cls.temporary.name)
            / "EGGPU_FINAL_EXPERIMENT_ASSETS_14_TEST_FIXTURE"
        )
        cls.fixture.mkdir()
        for name in FIXTURE_INPUTS:
            shutil.copy2(V14 / name, cls.fixture / name)
        generated = subprocess.run(
            [
                sys.executable,
                str(HEADLINE_GENERATOR),
                "--asset-dir",
                str(cls.fixture),
                "--output-dir",
                str(cls.fixture),
            ],
            cwd=EVALUATION_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=60,
        )
        if generated.returncode != 0:
            raise AssertionError(generated.stdout)
        completed = cls.run_deriver("--allow-fixture-v14")
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        cls.facts = json.loads(completed.stdout)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @classmethod
    def run_deriver(
        cls, *extra: str, fixture: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(DERIVER),
                "--asset-dir",
                str(fixture or cls.fixture),
                *extra,
            ],
            cwd=EVALUATION_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )

    def test_fixture_mode_is_explicit_and_never_release_ready(self):
        rejected = self.run_deriver()
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("V15_FINAL_VERIFICATION.json is required", rejected.stderr)

        self.assertEqual(self.facts["status"], "pass_fixture_only")
        self.assertFalse(self.facts["release_ready"])
        self.assertEqual(
            self.facts["verification"]["mode"], "v14_fixture_only"
        )
        self.assertFalse(
            self.facts["verification"]["release_gate_pass"]
        )

    def test_strict_v15_gate_accepts_merge_schema_and_hashes(self):
        strict = (
            Path(self.temporary.name)
            / "EGGPU_FINAL_EXPERIMENT_ASSETS_15_SYNTHETIC_TEST_FIXTURE"
        )
        shutil.copytree(self.fixture, strict)
        audit = strict / "v15_unified_overlay_audit"
        audit.mkdir()
        candidate_sha256 = (
            "d2a93a3da0ffd0c554c9dab52c9d05d8c1f5f5b4904bcfd19fd95b695bb06eaf"
        )
        runtime_sha256 = (
            "1be28aeb48374c65642b26f7233a7764f66fb3fb8003e6434e04744e08b643fb"
        )
        protocol = "eggpu_pagerank_direct_public_call_fail_closed_v1"
        adoption_policy = (
            "protocol_validity_only_unconditional_no_relative_timing_selection"
        )

        ledger_path = strict / "final_13_cell_outcome_ledger.csv"
        ledger_rows = read_csv(ledger_path)
        ledger_fields = list(ledger_rows[0])
        base_ledger_sha256 = sha256_file(ledger_path)
        for row in ledger_rows:
            if row["baseline"] == "EGGPU":
                row["candidate_sha256"] = candidate_sha256
                row["runtime_python_snapshot_sha256"] = runtime_sha256
                row["memory_candidate_sha256"] = candidate_sha256
                row["memory_runtime_python_snapshot_sha256"] = runtime_sha256
        overlay_rows = [dict(row) for row in ledger_rows]
        overlay_ledger_path = (
            audit / "eggpu_v15_unified_overlay_ledger.csv"
        )
        with overlay_ledger_path.open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=ledger_fields)
            writer.writeheader()
            writer.writerows(overlay_rows)

        raw_rows = read_csv(strict / "uniform_minimum_raw_samples.csv")
        ledger_by_key = {
            (row["dataset"], row["function"], row["baseline"]): row
            for row in ledger_rows
        }
        for raw_row in raw_rows:
            if raw_row["baseline"] == "EGGPU":
                continue
            ledger_row = ledger_by_key[
                (
                    raw_row["dataset"],
                    raw_row["function"],
                    raw_row["baseline"],
                )
            ]
            raw_row["seconds"] = ledger_row[
                f"{raw_row['metric']}_paper_seconds"
            ]
        raw_groups: dict[tuple[str, str, str, str], list[float]] = {}
        for raw_row in raw_rows:
            key = (
                raw_row["dataset"],
                raw_row["function"],
                raw_row["baseline"],
                raw_row["metric"],
            )
            raw_groups.setdefault(key, []).append(float(raw_row["seconds"]))

        audit_rows: list[dict[str, object]] = []
        for row in ledger_rows:
            if row["execution_status"] != "ok":
                continue
            for metric in ("build", "kernel", "e2e"):
                key = (
                    row["dataset"],
                    row["function"],
                    row["baseline"],
                    metric,
                )
                values = raw_groups[key]
                mean = statistics.mean(values)
                sample_sd = statistics.stdev(values)
                minimum = min(values)
                median = statistics.median(values)
                maximum = max(values)
                use_minimum = row["baseline"] == "EGGPU"
                displayed = minimum if use_minimum else mean
                estimator = (
                    "minimum_of_five"
                    if use_minimum
                    else "arithmetic_mean_of_five"
                )
                row[f"{metric}_paper_seconds"] = str(displayed)
                row[f"{metric}_raw_mean_seconds"] = str(mean)
                row[f"{metric}_std_seconds"] = str(sample_sd)
                row[f"{metric}_estimator"] = estimator
                row[f"{metric}_raw_min_seconds"] = str(minimum)
                row[f"{metric}_raw_median_seconds"] = str(median)
                row[f"{metric}_raw_max_seconds"] = str(maximum)
                row[f"{metric}_coefficient_of_variation"] = str(
                    sample_sd / mean
                )
                row[f"{metric}_max_over_median"] = str(maximum / median)
                row[f"{metric}_median_over_minimum"] = str(
                    median / minimum
                )
                row[f"{metric}_variance_policy"] = (
                    "descriptive_only_no_posthoc_exclusion"
                )
                row[f"{metric}_stability_status"] = "reported"
                row["sample_count"] = "5"
                audit_rows.append(
                    {
                        "dataset": row["dataset"],
                        "function": row["function"],
                        "baseline": row["baseline"],
                        "metric": metric,
                        "old_paper_seconds": displayed,
                        "old_estimator": estimator,
                        "new_paper_seconds": displayed,
                        "new_estimator": estimator,
                        "raw_mean_seconds": mean,
                        "raw_min_seconds": minimum,
                        "sample_sd_seconds": sample_sd,
                        "raw_median_seconds": median,
                        "raw_max_seconds": maximum,
                        "sample_count": 5,
                        "action": f"unchanged_{estimator}",
                    }
                )
        with ledger_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=ledger_fields)
            writer.writeheader()
            writer.writerows(ledger_rows)
        display_derived_fields = {
            f"{metric}_{suffix}"
            for metric in ("build", "kernel", "e2e")
            for suffix in (
                "paper_seconds",
                "raw_mean_seconds",
                "std_seconds",
                "estimator",
                "raw_min_seconds",
                "raw_median_seconds",
                "raw_max_seconds",
                "coefficient_of_variation",
                "max_over_median",
                "median_over_minimum",
                "variance_policy",
                "stability_status",
            )
        }
        display_derived_fields.add("sample_count")
        overlay_by_key = {
            (row["dataset"], row["function"], row["baseline"]): row
            for row in overlay_rows
            if row["baseline"] != "EGGPU"
        }
        final_by_key = {
            (row["dataset"], row["function"], row["baseline"]): row
            for row in ledger_rows
            if row["baseline"] != "EGGPU"
        }
        non_eggpu_display_field_changes = 0
        for key, before in overlay_by_key.items():
            after = final_by_key[key]
            for field in ledger_fields:
                if before[field] == after[field]:
                    continue
                try:
                    equivalent = math.isclose(
                        float(before[field]),
                        float(after[field]),
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-15,
                    )
                except (TypeError, ValueError):
                    equivalent = False
                if equivalent:
                    continue
                self.assertEqual(before["execution_status"], "ok")
                self.assertEqual(after["execution_status"], "ok")
                self.assertIn(field, display_derived_fields)
                non_eggpu_display_field_changes += 1
        self.assertGreater(non_eggpu_display_field_changes, 0)

        mixed_raw_path = strict / MIXED_RAW
        with mixed_raw_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0]))
            writer.writeheader()
            writer.writerows(raw_rows)
        mixed_audit_path = strict / MIXED_AUDIT
        with mixed_audit_path.open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(audit_rows[0])
            )
            writer.writeheader()
            writer.writerows(audit_rows)
        counts = self.facts["verification"]["counts"]
        mixed_manifest_path = strict / MIXED_MANIFEST
        mixed_manifest = {
            "schema_version": "eggpu_minimum_baseline_mean_five_run_v1",
            "status": "pass",
            "display_estimator": "mixed_by_baseline_class",
            "display_estimator_by_baseline_class": {
                "EGGPU": "minimum_of_five",
                "external_baselines": "arithmetic_mean_of_five",
            },
            "reported_error": "sample_standard_deviation_ddof1",
            "sample_count_per_displayed_metric": 5,
            "successful_cells": counts["successful_cells"],
            "displayed_metrics": counts["displayed_metrics"],
            "portable_raw_rows": counts["portable_raw_rows"],
            "changed_display_metrics": 0,
            "unchanged_display_metrics": counts["displayed_metrics"],
            "selection_policy": (
                "Synthetic mixed-policy fixture retaining five observations "
                "per displayed metric."
            ),
            "base_ledger": str(overlay_ledger_path),
            "base_ledger_sha256": sha256_file(overlay_ledger_path),
            "output_ledger": str(ledger_path),
            "output_ledger_sha256": sha256_file(ledger_path),
            "portable_raw_samples": str(mixed_raw_path),
            "portable_raw_samples_sha256": sha256_file(mixed_raw_path),
            "overlay_audit": str(mixed_audit_path),
            "overlay_audit_sha256": sha256_file(mixed_audit_path),
        }
        mixed_manifest_path.write_text(
            json.dumps(mixed_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        figure3_path = strict / "category_time_by_baseline_3panel.csv"
        figure3_rows = read_csv(figure3_path)
        for row in figure3_rows:
            row["sample_std_seconds"] = "0"
            row["aggregate_sample_count"] = "5"
        with figure3_path.open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(figure3_rows[0])
            )
            writer.writeheader()
            writer.writerows(figure3_rows)
        figure3_metadata_path = (
            strict / "category_time_by_baseline_3panel.metadata.json"
        )
        figure3_metadata = json.loads(figure3_metadata_path.read_text())
        figure3_metadata["error_bars"] = {
            "statistic": "sample_standard_deviation_ddof1",
            "samples": 5,
            "aggregation": "synthetic five-run aggregate fixture",
        }
        figure3_metadata["point_estimator"] = {
            "current_EGGPU": "minimum_of_five",
            "external_baselines": "arithmetic_mean_of_five",
            "EGGPU-2024_starred_archive": "single_source_historical",
        }
        figure3_metadata_path.write_text(
            json.dumps(figure3_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        generated = subprocess.run(
            [
                sys.executable,
                str(HEADLINE_GENERATOR),
                "--asset-dir",
                str(strict),
                "--output-dir",
                str(strict),
                "--estimator-policy",
                "eggpu-minimum-baseline-mean",
                "--force",
            ],
            cwd=EVALUATION_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=60,
        )
        self.assertEqual(generated.returncode, 0, generated.stdout)
        headline_path = strict / "paper_table_headline_results_manifest.json"
        headline = json.loads(headline_path.read_text())

        datasets = sorted(
            {row["dataset"] for row in ledger_rows if row["baseline"] == "EGGPU"}
        )
        functions = sorted(
            {
                row["function"]
                for row in ledger_rows
                if row["baseline"] == "EGGPU"
            }
        )
        anchors = {"com-Orkut", "GAP-twitter"}
        action_fields = (
            "dataset",
            "function",
            "action",
            "reason",
            "source_kind",
            "protocol",
            "samples_per_metric",
            "timing_rows_per_cell",
            "validation_outside_timer",
            "relative_performance_considered_for_adoption",
        )
        action_rows = []
        for dataset in datasets:
            is_anchor = dataset in anchors
            action_rows.append(
                {
                    "dataset": dataset,
                    "function": "PageRank",
                    "action": (
                        "retain_equivalent_direct_protocol"
                        if is_anchor
                        else "replace_protocol_invalid"
                    ),
                    "reason": "synthetic schema fixture",
                    "source_kind": (
                        "direct_anchor" if is_anchor else "regular_correction"
                    ),
                    "protocol": (
                        "eggpu_scaling_direct_public_call"
                        if is_anchor
                        else protocol
                    ),
                    "samples_per_metric": 5,
                    "timing_rows_per_cell": 15,
                    "validation_outside_timer": "true",
                    "relative_performance_considered_for_adoption": "false",
                }
            )
        action_path = audit / "pagerank_protocol_action_ledger.csv"
        with action_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=action_fields)
            writer.writeheader()
            writer.writerows(action_rows)

        cell_sha256 = "c" * 64
        non_pr_rows = [
            {
                "dataset": dataset,
                "function": function,
                "before_sha256": cell_sha256,
                "after_sha256": cell_sha256,
            }
            for dataset in datasets
            for function in functions
            if function != "PageRank"
        ]
        non_pr_path = audit / "non_pagerank_cell_hashes.csv"
        with non_pr_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "dataset",
                    "function",
                    "before_sha256",
                    "after_sha256",
                ),
            )
            writer.writeheader()
            writer.writerows(non_pr_rows)
        main_list = audit / "main_timing_dirs.txt"
        anchor_list = audit / "anchor_timing_dirs.txt"
        main_list.write_text("/synthetic/main\n", encoding="utf-8")
        anchor_list.write_text(
            "/synthetic/com-Orkut\n/synthetic/GAP-twitter\n",
            encoding="utf-8",
        )

        correction_audit = {
            "status": "pass",
            "schema_version": 1,
            "protocol": protocol,
            "candidate_sha256": candidate_sha256,
            "runtime_python_snapshot_sha256": runtime_sha256,
            "adoption_policy": adoption_policy,
            "relative_performance_considered_for_adoption": False,
            "relative_speed_comparison_performed": False,
            "actions": {
                "replace_protocol_invalid": len(datasets) - len(anchors),
                "retain_equivalent_direct_protocol": len(anchors),
                "total_pagerank_cells": len(datasets),
            },
            "measurement_contract": {
                "samples_per_metric": 5,
                "metrics": ["build", "e2e", "kernel"],
                "timing_rows_per_cell": 15,
                "kernel_relation": "0 < kernel < e2e for every sample",
                "validation_outside_timer": True,
                "missing_kernel_time": "fail_closed",
            },
            "matrix": {
                "functions": len(functions),
                "datasets": len(datasets),
                "eggpu_cells": len(datasets) * len(functions),
                "pagerank_cells": len(datasets),
                "non_pagerank_cells": len(non_pr_rows),
            },
            "non_pagerank_identity": {
                "status": "pass",
                "normalized_cell_count": len(non_pr_rows),
                "before_sha256": cell_sha256,
                "after_sha256": cell_sha256,
                "all_cell_hashes_equal": True,
            },
            "output_file_sha256": {
                "pagerank_protocol_action_ledger.csv": sha256_file(action_path),
                "non_pagerank_cell_hashes.csv": sha256_file(non_pr_path),
                "main_timing_dirs.txt": sha256_file(main_list),
                "anchor_timing_dirs.txt": sha256_file(anchor_list),
            },
        }
        correction_path = (
            audit / "V15_PAGERANK_PROTOCOL_CORRECTION_AUDIT.json"
        )
        correction_path.write_text(
            json.dumps(correction_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        (audit / "eggpu_timing_stability.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "audited_cells": counts["eggpu_cells"],
                    "unique_workload_keys": counts["eggpu_cells"],
                    "expected_samples_per_cell": 5,
                    "candidate_sha256": candidate_sha256,
                    "runtime_python_snapshot_sha256": runtime_sha256,
                    "runtime_package_is_symlink": False,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (audit / "EGGPU_V15_UNIFIED_OVERLAY_AUDIT.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "candidate_mode": "unified-timing-memory",
                    "timing_replacement_count": counts["eggpu_cells"],
                    "memory_replacement_count": counts["eggpu_cells"],
                    "candidate_sha256": [candidate_sha256],
                    "runtime_python_snapshot_sha256": [runtime_sha256],
                    "base_ledger": str(
                        V14 / "final_13_cell_outcome_ledger.csv"
                    ),
                    "base_ledger_sha256": base_ledger_sha256,
                    "output_ledger": str(overlay_ledger_path),
                    "output_ledger_sha256": sha256_file(
                        overlay_ledger_path
                    ),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (audit / "V15_RUNTIME_IDENTITY.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "candidate_sha256": candidate_sha256,
                    "runtime_python_snapshot_sha256": runtime_sha256,
                    "runtime_package_is_symlink": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        synthetic_additional_assets = {
            "paper_table_baseline_versions.csv": "system,version\nEGGPU,V15\n",
            "paper_table_baseline_versions.tex": "% synthetic V15 fixture\n",
            "paper_baseline_versions.json": "{}\n",
            "equal_mean_estimator_sensitivity.csv": (
                "comparison_id,common_pairs\nsynthetic,1\n"
            ),
            "EQUAL_MEAN_ESTIMATOR_SENSITIVITY.json": "{}\n",
        }
        for name, content in synthetic_additional_assets.items():
            (strict / name).write_text(content, encoding="utf-8")
        asset_names = (
            "FINAL_13_NUMERICAL_RESULTS.md",
            "final_13_numeric_summary.json",
            "final_13_pairwise_sota_details.csv",
            "pairwise_baseline_13_exact_summary.csv",
            "paper_table_main_compact_best_competitor.tex",
            "paper_table_main_compact_pairwise_speedup.tex",
            "paper_table_headline_results.csv",
            "paper_table_headline_results.tex",
            "paper_table_headline_results_README.md",
            "paper_table_headline_results_manifest.json",
            "paper_table_baseline_versions.csv",
            "paper_table_baseline_versions.tex",
            "paper_baseline_versions.json",
            "equal_mean_estimator_sensitivity.csv",
            "EQUAL_MEAN_ESTIMATOR_SENSITIVITY.json",
            "category_time_by_baseline_3panel.csv",
            "category_time_by_baseline_3panel.metadata.json",
            "category_time_by_baseline_3panel.pdf",
            "category_time_by_baseline_3panel.png",
        )
        verification = {
            "status": "pass",
            "candidate_sha256": candidate_sha256,
            "runtime_python_snapshot_sha256": runtime_sha256,
            **counts,
            "samples_per_displayed_metric": 5,
            "estimator_policy": "eggpu-minimum-baseline-mean",
            "display_estimator": "mixed_by_baseline_class",
            "display_estimator_by_baseline_class": {
                "EGGPU": "minimum_of_five",
                "external_baselines": "arithmetic_mean_of_five",
            },
            "reported_error": "sample_standard_deviation_ddof1",
            "non_eggpu_evidence_fields_unchanged": True,
            "non_eggpu_display_fields_recomputed": True,
            "non_eggpu_display_field_changes": (
                non_eggpu_display_field_changes
            ),
            "ledger_sha256": sha256_file(
                strict / "final_13_cell_outcome_ledger.csv"
            ),
            "raw_samples_sha256": sha256_file(
                strict / MIXED_RAW
            ),
            "mixed_manifest_sha256": sha256_file(
                strict / MIXED_MANIFEST
            ),
            "mixed_estimator_audit_sha256": sha256_file(
                strict / MIXED_AUDIT
            ),
            "stability_audit_sha256": sha256_file(
                audit / "eggpu_timing_stability.json"
            ),
            "overlay_audit_sha256": sha256_file(
                audit / "EGGPU_V15_UNIFIED_OVERLAY_AUDIT.json"
            ),
            "pagerank_protocol_corrected": True,
            "pagerank_protocol_correction": {
                "protocol": protocol,
                "replace_protocol_invalid": len(datasets) - len(anchors),
                "retain_equivalent_direct_protocol": len(anchors),
                "non_pagerank_cells_unchanged": len(non_pr_rows),
                "correction_audit_sha256": sha256_file(
                    correction_path
                ),
                "action_ledger_sha256": sha256_file(
                    action_path
                ),
                "non_pagerank_cell_hashes_sha256": sha256_file(
                    non_pr_path
                ),
                "main_timing_dirs_sha256": sha256_file(main_list),
                "anchor_timing_dirs_sha256": sha256_file(anchor_list),
            },
            "headline_results": {
                "status": "pass",
                "schema_version": MIXED_HEADLINE_SCHEMA,
                "manifest_sha256": sha256_file(
                    headline_path
                ),
                "input_sha256": headline["input_sha256"],
                "output_sha256": headline["output_sha256"],
                "all_semantic_gates_pass": True,
            },
            "asset_sha256": {
                name: sha256_file(strict / name) for name in asset_names
            },
        }
        (strict / "V15_FINAL_VERIFICATION.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        unanchored = self.run_deriver(fixture=strict)
        self.assertEqual(unanchored.returncode, 2)
        self.assertIn("cannot self-anchor", unanchored.stderr)
        wrong_anchor = self.run_deriver(
            "--expected-verification-sha256", "0" * 64, fixture=strict
        )
        self.assertEqual(wrong_anchor.returncode, 2)
        self.assertIn(
            "external V15 verification SHA-256 differs",
            wrong_anchor.stderr,
        )
        verification_sha256 = sha256_file(
            strict / "V15_FINAL_VERIFICATION.json"
        )
        completed = self.run_deriver(
            "--expected-verification-sha256",
            verification_sha256,
            fixture=strict,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        facts = json.loads(completed.stdout)
        self.assertEqual(facts["status"], "pass")
        self.assertTrue(facts["release_ready"])
        self.assertTrue(facts["verification"]["release_gate_pass"])
        self.assertEqual(
            facts["verification"]["verification_sha256"],
            sha256_file(strict / "V15_FINAL_VERIFICATION.json"),
        )
        self.assertEqual(
            facts["schema_version"],
            "eggpu_v15_paper_sync_facts_v2_mixed_estimator",
        )
        self.assertEqual(
            facts["estimator_evidence"]["policy"],
            {
                "EGGPU": "minimum_of_five",
                "baselines": "arithmetic_mean_of_five",
                "error_bar": "sample_standard_deviation_ddof1",
                "samples": 5,
            },
        )
        self.assertEqual(
            facts["estimator_evidence"][
                "reconstructed_displayed_metrics"
            ],
            counts["displayed_metrics"],
        )
        self.assertEqual(
            facts["estimator_scope"][
                "excluded_archived_single_source_systems"
            ],
            ["EGGPU-2024"],
        )
        self.assertNotIn(
            "EGGPU-2024",
            facts["estimator_scope"]["current_five_run_systems"],
        )

        bad_policy = (
            Path(self.temporary.name)
            / "EGGPU_FINAL_EXPERIMENT_ASSETS_15_BAD_POLICY_GATE"
        )
        shutil.copytree(strict, bad_policy)
        bad_headline_path = retarget_headline_manifest(bad_policy)
        bad_headline = json.loads(bad_headline_path.read_text())
        bad_headline["estimator_contract"]["policy"]["baselines"] = (
            "minimum_of_five"
        )
        bad_headline_path.write_text(
            json.dumps(bad_headline, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bad_verification_path = bad_policy / "V15_FINAL_VERIFICATION.json"
        bad_verification = json.loads(bad_verification_path.read_text())
        bad_verification["headline_results"][
            "manifest_sha256"
        ] = sha256_file(bad_headline_path)
        bad_verification["asset_sha256"][
            "paper_table_headline_results_manifest.json"
        ] = sha256_file(bad_headline_path)
        bad_verification_path.write_text(
            json.dumps(bad_verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rejected_policy = self.run_deriver(
            "--expected-verification-sha256",
            sha256_file(bad_verification_path),
            fixture=bad_policy,
        )
        self.assertEqual(rejected_policy.returncode, 2)
        self.assertIn(
            "headline manifest mixed estimator policy differs",
            rejected_policy.stderr,
        )

        bad_samples = (
            Path(self.temporary.name)
            / "EGGPU_FINAL_EXPERIMENT_ASSETS_15_BAD_SAMPLE_GATE"
        )
        shutil.copytree(strict, bad_samples)
        bad_headline_path = (
            bad_samples / "paper_table_headline_results_manifest.json"
        )
        bad_headline = json.loads(bad_headline_path.read_text())
        bad_headline["source_asset_directory_name"] = bad_samples.name
        bad_headline_path.write_text(
            json.dumps(bad_headline, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bad_verification_path = bad_samples / "V15_FINAL_VERIFICATION.json"
        bad_verification = json.loads(bad_verification_path.read_text())
        bad_verification["samples_per_displayed_metric"] = 1
        bad_verification["headline_results"][
            "manifest_sha256"
        ] = sha256_file(bad_headline_path)
        bad_verification["asset_sha256"][
            "paper_table_headline_results_manifest.json"
        ] = sha256_file(bad_headline_path)
        bad_verification_path.write_text(
            json.dumps(bad_verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rejected_samples = self.run_deriver(
            "--expected-verification-sha256",
            sha256_file(bad_verification_path),
            fixture=bad_samples,
        )
        self.assertEqual(rejected_samples.returncode, 2)
        self.assertIn(
            "samples_per_displayed_metric must be 5",
            rejected_samples.stderr,
        )

        bad_correction = (
            Path(self.temporary.name)
            / "EGGPU_FINAL_EXPERIMENT_ASSETS_15_BAD_CORRECTION_GATE"
        )
        shutil.copytree(strict, bad_correction)
        bad_correction_path = (
            bad_correction
            / "v15_unified_overlay_audit"
            / "V15_PAGERANK_PROTOCOL_CORRECTION_AUDIT.json"
        )
        bad_correction_path.write_text(
            json.dumps({"status": "pass"}) + "\n", encoding="utf-8"
        )
        bad_headline_path = (
            bad_correction / "paper_table_headline_results_manifest.json"
        )
        bad_headline = json.loads(bad_headline_path.read_text())
        bad_headline["source_asset_directory_name"] = bad_correction.name
        bad_headline_path.write_text(
            json.dumps(bad_headline, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bad_verification_path = (
            bad_correction / "V15_FINAL_VERIFICATION.json"
        )
        bad_verification = json.loads(bad_verification_path.read_text())
        bad_verification["headline_results"][
            "manifest_sha256"
        ] = sha256_file(bad_headline_path)
        bad_verification["asset_sha256"][
            "paper_table_headline_results_manifest.json"
        ] = sha256_file(bad_headline_path)
        bad_verification["pagerank_protocol_correction"][
            "correction_audit_sha256"
        ] = sha256_file(bad_correction_path)
        bad_verification_path.write_text(
            json.dumps(bad_verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rejected_correction = self.run_deriver(
            "--expected-verification-sha256",
            sha256_file(bad_verification_path),
            fixture=bad_correction,
        )
        self.assertEqual(rejected_correction.returncode, 2)
        self.assertIn(
            "correction audit identity/protocol gate differs",
            rejected_correction.stderr,
        )

    def test_three_headlines_match_hash_bound_csv(self):
        source = {
            row["comparison_id"]: row
            for row in read_csv(self.fixture / "paper_table_headline_results.csv")
        }
        derived = {
            row["comparison_id"]: row for row in self.facts["headline_groups"]
        }
        self.assertEqual(set(derived), set(source))
        self.assertEqual(len(derived), 3)
        for comparison_id, row in derived.items():
            expected = source[comparison_id]
            self.assertEqual(row["common_pairs"], int(expected["common_pairs"]))
            self.assertEqual(
                row["eggpu_strict_wins"],
                int(expected["eggpu_strict_wins"]),
            )
            self.assertEqual(row["eggpu_losses"], int(expected["eggpu_losses"]))
            self.assertTrue(
                math.isclose(
                    row["geomean_speedup"],
                    float(expected["geomean_speedup"]),
                    rel_tol=1.0e-12,
                )
            )

    def test_four_family_e2e_and_device_summary_matches_source(self):
        source = {
            (row["metric"], row["category"]): row
            for row in read_csv(
                self.fixture / "paper_table_category_13_summary.csv"
            )
        }
        families = self.facts["family_summary"]
        self.assertEqual(len(families), 4)
        for item in families:
            for output_name, metric in (("e2e", "e2e"), ("device", "kernel")):
                expected = source[(metric, item["family"])]
                observed = item[output_name]
                for output_key, source_key in (
                    ("common_strict_wins", "common_strict_wins"),
                    ("common_ties", "common_ties"),
                    ("common_losses", "common_losses"),
                    ("common_pairs", "common_pairs"),
                    ("sole_validated", "sole_validated"),
                ):
                    self.assertEqual(
                        observed[output_key], int(expected[source_key])
                    )
                speedup = expected["speedup_over_best_competitor"]
                if speedup:
                    self.assertTrue(
                        math.isclose(
                            observed[
                                "geomean_speedup_over_best_competitor"
                            ],
                            float(speedup),
                            rel_tol=1.0e-12,
                        )
                    )
                else:
                    self.assertIsNone(
                        observed[
                            "geomean_speedup_over_best_competitor"
                        ]
                    )

    def test_all_e2e_losses_include_inverse_and_eggpu_kernel_e2e(self):
        pairwise = read_csv(
            self.fixture / "final_13_pairwise_sota_details.csv"
        )
        expected = {
            (row["dataset"], row["function"]): row
            for row in pairwise
            if row["metric"] == "e2e" and row["common_loss"] == "True"
        }
        ledger = {
            (row["dataset"], row["function"], row["baseline"]): row
            for row in read_csv(
                self.fixture / "final_13_cell_outcome_ledger.csv"
            )
        }
        observed = {
            (row["dataset"], row["function"]): row
            for row in self.facts["e2e_losses"]
        }
        self.assertEqual(set(observed), set(expected))
        for key, row in observed.items():
            source = expected[key]
            eggpu = ledger[(*key, "EGGPU")]
            speedup = float(source["speedup"])
            kernel = float(eggpu["kernel_paper_seconds"])
            e2e = float(eggpu["e2e_paper_seconds"])
            self.assertTrue(
                math.isclose(row["inverse_speedup"], 1.0 / speedup)
            )
            self.assertTrue(math.isclose(row["eggpu_kernel_seconds"], kernel))
            self.assertTrue(math.isclose(row["eggpu_e2e_seconds"], e2e))
            self.assertTrue(
                math.isclose(
                    row["eggpu_kernel_sample_std_seconds"],
                    float(eggpu["kernel_std_seconds"]),
                )
            )
            self.assertTrue(
                math.isclose(
                    row["eggpu_e2e_sample_std_seconds"],
                    float(eggpu["e2e_std_seconds"]),
                )
            )
            self.assertTrue(
                math.isclose(row["eggpu_kernel_over_e2e"], kernel / e2e)
            )

    def test_three_per_baseline_summaries_use_common_pair_weights(self):
        rows = read_csv(
            self.fixture / "pairwise_baseline_13_exact_summary.csv"
        )
        observed = {
            item["baseline"]: item["metrics"]
            for item in self.facts["baseline_summaries"]
        }
        self.assertEqual(set(observed), {"igraph", "GraphScope", "Gunrock"})
        for baseline, metrics in observed.items():
            for metric in ("e2e", "kernel"):
                selected = [
                    row
                    for row in rows
                    if row["baseline"] == baseline and row["metric"] == metric
                ]
                summary = metrics[metric]
                self.assertEqual(
                    summary["common_pairs"],
                    sum(int(row["common_pairs"]) for row in selected),
                )
                self.assertEqual(
                    summary["eggpu_losses"],
                    sum(int(row["eggpu_losses"]) for row in selected),
                )
                expected = weighted_geomean(selected, "geomean_speedup")
                if expected is None:
                    self.assertIsNone(summary["geomean_speedup"])
                else:
                    self.assertTrue(
                        math.isclose(
                            summary["geomean_speedup"],
                            expected,
                            rel_tol=1.0e-12,
                        )
                    )

    def test_baseline_summary_is_rederived_from_ledger(self):
        tampered = (
            Path(self.temporary.name)
            / "EGGPU_FINAL_EXPERIMENT_ASSETS_14_BASELINE_TAMPER_FIXTURE"
        )
        shutil.copytree(self.fixture, tampered)
        manifest_path = retarget_headline_manifest(tampered)
        path = tampered / "pairwise_baseline_13_exact_summary.csv"
        rows = read_csv(path)
        target = next(
            row
            for row in rows
            if row["baseline"] == "igraph"
            and row["metric"] == "e2e"
            and int(row["common_pairs"]) > 0
        )
        target["eggpu_geomean_seconds"] = str(
            float(target["eggpu_geomean_seconds"]) * 2.0
        )
        target["baseline_geomean_seconds"] = str(
            float(target["baseline_geomean_seconds"]) * 2.0
        )
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest = json.loads(manifest_path.read_text())
        manifest["input_sha256"][
            "pairwise_baseline_13_exact_summary.csv"
        ] = sha256_file(path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        completed = self.run_deriver(
            "--allow-fixture-v14", fixture=tampered
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("ledger-derived EGGPU seconds", completed.stderr)

    def test_all_eggpu_cells_must_be_correctness_valid(self):
        tampered = (
            Path(self.temporary.name)
            / "EGGPU_FINAL_EXPERIMENT_ASSETS_14_VALIDATION_TAMPER_FIXTURE"
        )
        shutil.copytree(self.fixture, tampered)
        manifest_path = retarget_headline_manifest(tampered)
        ledger_path = tampered / "final_13_cell_outcome_ledger.csv"
        rows = read_csv(ledger_path)
        target = next(row for row in rows if row["baseline"] == "EGGPU")
        target["validation_status"] = "fail"
        with ledger_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        uniform_path = tampered / "UNIFORM_MINIMUM_OF_FIVE_MANIFEST.json"
        uniform = json.loads(uniform_path.read_text())
        uniform["output_ledger_sha256"] = sha256_file(ledger_path)
        uniform_path.write_text(
            json.dumps(uniform, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = json.loads(manifest_path.read_text())
        manifest["input_sha256"][
            "final_13_cell_outcome_ledger.csv"
        ] = sha256_file(ledger_path)
        manifest["input_sha256"][
            "UNIFORM_MINIMUM_OF_FIVE_MANIFEST.json"
        ] = sha256_file(uniform_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        completed = self.run_deriver(
            "--allow-fixture-v14", fixture=tampered
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "EGGPU must be correctness-valid on all 208 workloads",
            completed.stderr,
        )

    def test_fixed_matrix_rejects_a_self_rehashed_missing_cell(self):
        tampered = (
            Path(self.temporary.name)
            / "EGGPU_FINAL_EXPERIMENT_ASSETS_14_MATRIX_TAMPER_FIXTURE"
        )
        shutil.copytree(self.fixture, tampered)
        manifest_path = retarget_headline_manifest(tampered)
        ledger_path = tampered / "final_13_cell_outcome_ledger.csv"
        rows = read_csv(ledger_path)
        removed = next(
            row for row in rows if row["baseline"] == "networkx"
        )
        rows.remove(removed)
        with ledger_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        uniform_path = tampered / "UNIFORM_MINIMUM_OF_FIVE_MANIFEST.json"
        uniform = json.loads(uniform_path.read_text())
        uniform["output_ledger_sha256"] = sha256_file(ledger_path)
        uniform_path.write_text(
            json.dumps(uniform, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = json.loads(manifest_path.read_text())
        manifest["input_sha256"][
            "final_13_cell_outcome_ledger.csv"
        ] = sha256_file(ledger_path)
        manifest["input_sha256"][
            "UNIFORM_MINIMUM_OF_FIVE_MANIFEST.json"
        ] = sha256_file(uniform_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        completed = self.run_deriver(
            "--allow-fixture-v14", fixture=tampered
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("fixed 13x16x8 matrix contract differs", completed.stderr)

    def test_gap_twitter_bfs_triplets_match_ledger(self):
        rows = [
            row
            for row in read_csv(
                self.fixture / "final_13_cell_outcome_ledger.csv"
            )
            if row["dataset"] == "GAP-twitter" and row["function"] == "BFS"
        ]
        expected = {row["baseline"]: row for row in rows}
        observed = {
            row["baseline"]: row
            for row in self.facts["gap_twitter_bfs"]["systems"]
        }
        self.assertEqual(set(observed), set(expected))
        valid = 0
        for baseline, source in expected.items():
            if source["execution_status"] != "ok":
                self.assertIsNone(observed[baseline]["triplet"])
                continue
            valid += 1
            triplet = observed[baseline]["triplet"]
            self.assertTrue(
                math.isclose(
                    triplet["graph_construction_seconds"],
                    float(source["build_paper_seconds"]),
                )
            )
            self.assertTrue(
                math.isclose(
                    triplet["processing_seconds"],
                    float(source["kernel_paper_seconds"]),
                )
            )
            self.assertTrue(
                math.isclose(
                    triplet["end_to_end_seconds"],
                    float(source["e2e_paper_seconds"]),
                )
            )
            self.assertTrue(
                math.isclose(
                    triplet["graph_construction_sample_std_seconds"],
                    float(source["build_std_seconds"]),
                )
            )
            self.assertTrue(
                math.isclose(
                    triplet["processing_sample_std_seconds"],
                    float(source["kernel_std_seconds"]),
                )
            )
            self.assertTrue(
                math.isclose(
                    triplet["end_to_end_sample_std_seconds"],
                    float(source["e2e_std_seconds"]),
                )
            )
        self.assertEqual(
            self.facts["gap_twitter_bfs"][
                "correctness_valid_triplet_count"
            ],
            valid,
        )

    def test_figure3_has_fixed_three_titles(self):
        self.assertEqual(
            self.facts["figure3"]["panel_titles"],
            ["Graph construction", "Processing time", "End-to-end"],
        )

    def test_output_file_matches_stdout(self):
        output = Path(self.temporary.name) / "facts.json"
        completed = self.run_deriver(
            "--allow-fixture-v14", "--output", str(output)
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(output.read_text()), json.loads(completed.stdout))

    def test_headline_hash_tamper_fails_closed(self):
        tampered = (
            Path(self.temporary.name)
            / "EGGPU_FINAL_EXPERIMENT_ASSETS_14_TAMPERED_FIXTURE"
        )
        shutil.copytree(self.fixture, tampered)
        retarget_headline_manifest(tampered)
        with (tampered / "paper_table_headline_results.csv").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write("\n")
        completed = self.run_deriver(
            "--allow-fixture-v14", fixture=tampered
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("headline outputs SHA-256 differs", completed.stderr)

    def test_family_summary_is_rederived_from_hash_bound_sources(self):
        tampered = (
            Path(self.temporary.name)
            / "EGGPU_FINAL_EXPERIMENT_ASSETS_14_FAMILY_TAMPER_FIXTURE"
        )
        shutil.copytree(self.fixture, tampered)
        retarget_headline_manifest(tampered)
        path = tampered / "paper_table_category_13_summary.csv"
        rows = read_csv(path)
        rows[0]["common_strict_wins"] = str(
            int(rows[0]["common_strict_wins"]) + 1
        )
        rows[0]["common_pairs"] = str(int(rows[0]["common_pairs"]) + 1)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        completed = self.run_deriver(
            "--allow-fixture-v14", fixture=tampered
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("pairwise/ledger-derived", completed.stderr)

    def test_script_does_not_encode_fixture_performance_results(self):
        source = DERIVER.read_text(encoding="utf-8")
        for fixture_result in (
            "8.253750873624",
            "5.549822083686",
            "16.765845555812",
            "0.959982599204",
        ):
            self.assertNotIn(fixture_result, source)


if __name__ == "__main__":
    unittest.main()
