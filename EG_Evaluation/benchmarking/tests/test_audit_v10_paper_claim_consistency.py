import csv
import importlib.util
import json
import math
import statistics
import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARKING = Path(__file__).resolve().parents[1]
MODULE_PATH = BENCHMARKING / "audit_v10_paper_claim_consistency.py"
SPEC = importlib.util.spec_from_file_location("audit_v10_claims", MODULE_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


MIN_FIVE = "minimum of five independent runs"


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class V10ClaimFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.paper = self.root / "paper"
        self.assets = self.root / "assets"
        self.paper.mkdir()
        self.assets.mkdir()
        self._write_paper()
        self._write_headline()
        self._write_igraph()
        self._write_intro()
        self._write_ledger_and_samples()
        self._write_variance_policy()
        self._write_gap()
        self._write_first_use()
        self._write_ablation()
        self._write_workflow()

    def _write_paper(self):
        (self.paper / "main.tex").write_text(
            r"""
EGGPU completes all 1 workloads. It is fastest or tied on 1 of 1
correctness-aligned common workloads and is the sole validated implementation
on 0 additional workloads. Its speedup is 2.00$\times$ over the pairwise best
comparable external library call and 2.00$\times$ over strict nx-cugraph.
At the matched device boundary, it is 3.00$\times$ faster than Gunrock.
The EGGPU center is the minimum of five independent runs.

R-MAT standalone supplement. Points are arithmetic means of five runs.
The workflow supplement reports an arithmetic mean of five runs.
igraph reports an arithmetic mean of five runs under its frozen protocol.
""",
            encoding="utf-8",
        )

    def _write_headline(self):
        summary = {
            "e2e_eggpu_successful_workloads": 1,
            "e2e_common_pair_strict_wins": 1,
            "e2e_common_pair_ties": 0,
            "e2e_common_pair_losses": 0,
            "e2e_competitive_pairs": 1,
            "e2e_sole_validated_cells": 0,
            "e2e_speedup_over_best_competitor": 2.0,
            "e2e_speedup_over_strict_nx_cugraph": 2.0,
            "e2e_strict_nx_cugraph_common_pairs": 1,
            "kernel_speedup_over_best_native_gpu": 3.0,
            "kernel_best_native_gpu_common_pairs": 1,
        }
        (self.assets / "final_13_numeric_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    def _write_igraph(self):
        rows = []
        for bucket, speedup in (
            ("<1e5", 1.5),
            ("1e5--1e6", 2.0),
            (">=1e6", 2.5),
            ("ALL", 2.0),
        ):
            rows.append(
                {
                    "metric": "e2e_public_call",
                    "timing_boundary": "test",
                    "size_bucket": bucket,
                    "function": "ALL",
                    "common_pairs": 1,
                    "strict_wins": 1,
                    "ties": 0,
                    "losses": 0,
                    "igraph_over_eggpu_geomean": speedup,
                }
            )
        write_csv(
            self.assets / "eggpu_vs_igraph_crossover_by_scale_function.csv", rows
        )

    def _write_intro(self):
        rows = []
        for scale, eggpu, igraph in (
            (20, 1.0, 2.0),
            (22, 2.0, 5.0),
            (24, 4.0, 12.0),
            (26, 8.0, None),
        ):
            rows.append(
                {
                    "system": "EGGPU",
                    "scale": scale,
                    "vertices": 2**scale,
                    "adjacency_entries": 16 * 2**scale,
                    "pagerank_public_call_seconds": eggpu,
                    "sample_standard_deviation_seconds": 0.01,
                    "status": "ok",
                    "estimator": "arithmetic mean of five isolated processes",
                }
            )
            rows.append(
                {
                    "system": "igraph",
                    "scale": scale,
                    "vertices": 2**scale,
                    "adjacency_entries": 16 * 2**scale,
                    "pagerank_public_call_seconds": "" if igraph is None else igraph,
                    "sample_standard_deviation_seconds": "",
                    "status": "resource_limit" if igraph is None else "ok",
                    "estimator": (
                        "not applicable"
                        if igraph is None
                        else "arithmetic mean of five isolated processes"
                    ),
                }
            )
        write_csv(self.assets / "intro_rmat_pagerank_uniform_mean.csv", rows)

    def _write_ledger_and_samples(self):
        e2e = [1.0, 1.01, 1.02, 1.03, 1.04]
        kernel = [0.5, 0.501, 0.502, 0.503, 0.504]
        self.raw_values = {"e2e": e2e, "kernel": kernel}
        rows = []
        for baseline, memory in (("EGGPU", 100.0), ("nx-cugraph", 120.0), ("Gunrock", 110.0)):
            row = {
                "dataset": "g",
                "function": "PageRank",
                "baseline": baseline,
                "execution_status": "ok",
                "validation_status": "pass",
                "sample_count": 5,
                "gpu_peak_mb_mean": memory,
            }
            for metric, values in self.raw_values.items():
                row[f"{metric}_paper_seconds"] = min(values)
                row[f"{metric}_raw_mean_seconds"] = statistics.fmean(values)
                row[f"{metric}_std_seconds"] = statistics.stdev(values)
                row[f"{metric}_estimator"] = (
                    MIN_FIVE if baseline == "EGGPU" else "frozen baseline protocol"
                )
            rows.append(row)
        self.ledger_path = self.assets / "final_13_cell_outcome_ledger.csv"
        write_csv(self.ledger_path, rows)

        samples = []
        for metric, values in self.raw_values.items():
            for index, value in enumerate(values, 1):
                samples.append(
                    {
                        "dataset": "g",
                        "function": "PageRank",
                        "baseline": "EGGPU",
                        "metric": metric,
                        "sample_index": index,
                        "seconds": value,
                        "status": "ok",
                    }
                )
        self.samples_path = self.assets / "final_13_eggpu_timing_samples.csv"
        write_csv(self.samples_path, samples)
        write_csv(
            self.assets / "memory_resource_domain_summary.csv",
            [
                {
                    "baseline": "EGGPU",
                    "resource": "process_gpu_peak_mb",
                    "cells": 1,
                    "geometric_mean_mb": 100,
                    "median_mb": 100,
                    "maximum_mb": 100,
                    "comparison_scope": "test",
                }
            ],
        )

    def _write_variance_policy(self):
        (self.assets / "eggpu_timing_stability_policy.json").write_text(
            json.dumps(
                {
                    "variance_policy": "catastrophic_outlier_guard",
                    "eggpu_center": "minimum_of_five",
                    "raw_sample_count": 5,
                    "dispersion_statistics": [
                        "raw_mean",
                        "sample_standard_deviation",
                        "coefficient_of_variation",
                    ],
                    "max_over_median_limit": 5.0,
                    "median_over_min_limit": 3.0,
                }
            ),
            encoding="utf-8",
        )

    def _write_gap(self):
        functions = (
            "SCC",
            "MST",
            "LCC",
            "KCore",
            "EffectiveSize",
            "Efficiency",
            "Constraint",
            "Hierarchy",
        )
        rows = [
            {
                "function": function,
                "validation_status": "pass",
                "public_return_paper_seconds": index / 10,
                "public_return_sample_std_seconds": 0.001,
                "sample_count": 5,
                "estimator": MIN_FIVE,
            }
            for index, function in enumerate(functions, 1)
        ]
        write_csv(self.assets / "gap_twitter_V10_timing_table.csv", rows)
        (self.assets / "gap_twitter_V10_timing_summary.json").write_text(
            json.dumps({"status": "pass", "timing_estimator": MIN_FIVE}),
            encoding="utf-8",
        )

    def _write_first_use(self):
        families = (
            "Centrality",
            "Connectivity",
            "Paths & Spanning Trees",
            "Structural Holes",
        )
        summary = []
        details = []
        for family in families:
            summary.append(
                {
                    "family": family,
                    "metric": "e2e",
                    "pairs": 1,
                    "first_use_geomean_seconds": 2,
                    "steady_state_geomean_seconds": 1,
                    "first_use_over_steady": 2,
                    "estimator": "arithmetic mean of five isolated processes",
                }
            )
            details.append(
                {
                    "dataset": "g",
                    "function": "PageRank",
                    "metric": "e2e",
                    "family": family,
                    "first_use_paper_seconds": 2,
                    "steady_paper_seconds": 1,
                    "first_use_mean_seconds": 2,
                    "steady_mean_seconds": 1,
                    "sample_count": 5,
                    "estimator": "arithmetic mean of five isolated processes",
                }
            )
        write_csv(self.assets / "first_use_steady_actual_times.csv", summary)
        write_csv(self.assets / "first_use_steady_pair_details.csv", details)

    def _write_ablation(self):
        modules = (
            ("GraphContext", "Workflow time (s)", 2.0),
            ("C++ graph cache", "Workflow time (s)", 3.0),
            ("Result reconstruction", "Return time (s)", 4.0),
            ("CSR storage", "Host storage (MiB)", 5.0),
            ("CSR traversal", "Degree traversal (ms)", 6.0),
        )
        write_csv(
            self.assets / "ablation_actual_values.csv",
            [
                {
                    "module": module,
                    "axis": axis,
                    "complete_value": 1,
                    "ablated_or_reference_value": ratio,
                    "ratio": ratio,
                    "cases": 1,
                    "complete_label": "complete",
                    "reference_label": "reference",
                    "estimator": "arithmetic mean of five isolated processes",
                }
                for module, axis, ratio in modules
            ],
        )

    def _write_workflow(self):
        functions = ("WCC", "PageRank", "BFS", "SSSP", "Closeness")
        calls = []
        cumulative = []
        retained_total = 0.0
        isolated_total = 0.0
        for position, function in enumerate(functions, 1):
            for metric in ("e2e", "kernel"):
                reuse = 1.0 if metric == "e2e" else 0.5
                isolated = reuse * (position + 1)
                calls.append(
                    {
                        "call_position": position,
                        "function": function,
                        "metric": metric,
                        "datasets": 1,
                        "reuse_seconds": reuse,
                        "isolated_seconds": isolated,
                        "isolated_over_reuse": isolated / reuse,
                        "estimator": "arithmetic mean of five isolated processes",
                    }
                )
            retained_total += 1.0
            isolated_total += position + 1
            for baseline, value in (
                ("EGGPU", retained_total),
                ("EGGPU-isolated", isolated_total),
            ):
                cumulative.append(
                    {
                        "baseline": baseline,
                        "call_position": position,
                        "function": function,
                        "datasets": 1,
                        "common_dataset_names": "g",
                        "cumulative_geomean_seconds": value,
                        "estimator": "arithmetic mean of five isolated processes",
                    }
                )
        write_csv(
            self.assets / "workflow_five_call_self_control_minimum_of_five.csv",
            calls,
        )
        write_csv(
            self.assets
            / "workflow_five_call_reuse_cumulative_minimum_of_five.csv",
            cumulative,
        )
        (self.assets / "workflow_state_reuse_provenance.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "bound_to_v10_candidate_binary": False,
                    "statistics": {
                        "within_dataset_estimator": (
                            "arithmetic mean of five isolated processes"
                        )
                    },
                }
            ),
            encoding="utf-8",
        )


class V10PaperClaimConsistencyTests(unittest.TestCase):
    def test_assets_only_phase_defers_unsynchronized_paper_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = V10ClaimFixture(temporary)
            (fixture.paper / "main.tex").write_text(
                "This intentionally stale paper has no V10 numerical claims.\n",
                encoding="utf-8",
            )
            report = AUDIT.audit(
                None,
                fixture.assets,
                assets_only=True,
            )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["audit_scope"], "assets_only")
        self.assertTrue(report["paper_sync_pending"])
        self.assertIsNone(report["paper_dir"])
        self.assertEqual(report["claim_checks"], [])
        self.assertEqual(report["raw_sample_audit"]["audited_groups"], 2)

    def test_consistent_minimum_of_five_bundle_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = V10ClaimFixture(temporary)
            report = AUDIT.audit(fixture.paper, fixture.assets)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["missing_assets"], [])
        self.assertEqual(report["invalid_assets"], [])
        self.assertEqual(report["raw_sample_audit"]["audited_groups"], 2)
        self.assertEqual(report["stale_or_protocol_wording"], [])
        classifications = {
            item["classification"]
            for item in report["timing_wording_classification"]
        }
        self.assertIn("standalone_supplement_protocol_allowed", classifications)
        self.assertIn("competitor_frozen_protocol_allowed", classifications)

    def test_repeated_headline_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = V10ClaimFixture(temporary)
            with (fixture.paper / "main.tex").open("a", encoding="utf-8") as handle:
                handle.write(
                    r"""
The geometric-mean speedup is 9.99$\times$ over the pairwise best comparable
external library call.
"""
                )
            report = AUDIT.audit(fixture.paper, fixture.assets)

        check = next(
            item
            for item in report["claim_checks"]
            if item["claim"] == "headline_best_external_speedup"
        )
        self.assertEqual(report["status"], "fail")
        self.assertEqual(check["status"], "fail")
        self.assertEqual(len(check["occurrences"]), 2)
        self.assertFalse(check["occurrences"][-1]["matches"])

    def test_missing_supplement_is_explicit_and_non_passing(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = V10ClaimFixture(temporary)
            (
                fixture.assets
                / "workflow_five_call_self_control_minimum_of_five.csv"
            ).unlink()
            report = AUDIT.audit(fixture.paper, fixture.assets)

        self.assertEqual(report["status"], "fail")
        self.assertIn("workflow_calls", report["missing_assets"])
        self.assertEqual(report["assets"]["workflow_calls"]["status"], "missing")

    def test_incomplete_raw_samples_and_arithmetic_mean_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = V10ClaimFixture(temporary)
            rows = read_csv(fixture.samples_path)
            write_csv(
                fixture.samples_path,
                [
                    row
                    for row in rows
                    if not (row["metric"] == "e2e" and row["sample_index"] == "5")
                ],
            )
            ledger = read_csv(fixture.ledger_path)
            ledger[0]["e2e_estimator"] = "arithmetic_mean"
            write_csv(fixture.ledger_path, ledger)
            report = AUDIT.audit(fixture.paper, fixture.assets)

        issues = report["assets"]["eggpu_raw_timing_samples"]["issues"]
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("expected five raw samples" in issue for issue in issues))

    def test_stale_wording_flags_best_of_five_but_allows_minimum_of_five(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = V10ClaimFixture(temporary)
            with (fixture.paper / "main.tex").open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n% best of five\n"
                    "Every EGGPU main-matrix point is an arithmetic mean.\n"
                    "The declared center is the minimum of five runs.\n"
                )
            report = AUDIT.audit(fixture.paper, fixture.assets)

        patterns = {
            item["pattern"] for item in report["stale_or_protocol_wording"]
        }
        self.assertIn("best_of_five", patterns)
        self.assertIn("arithmetic_mean_in_submission_tex", patterns)
        self.assertNotIn("minimum_of_five", patterns)

    def test_explicit_eggpu_mean_counterfactual_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = V10ClaimFixture(temporary)
            with (fixture.paper / "main.tex").open("a", encoding="utf-8") as handle:
                handle.write(
                    "\nAs an estimator sensitivity check, replacing every EGGPU "
                    "minimum with the\narithmetic mean of its same five raw samples "
                    "still preserves the conclusion.\n"
                )
            report = AUDIT.audit(fixture.paper, fixture.assets)

        classifications = {
            item["classification"]
            for item in report["timing_wording_classification"]
        }
        patterns = {
            item["pattern"] for item in report["stale_or_protocol_wording"]
        }
        self.assertIn("eggpu_mean_counterfactual_allowed", classifications)
        self.assertNotIn("arithmetic_mean_in_submission_tex", patterns)

    def test_catastrophic_outlier_guard_is_configurable_and_not_an_sd_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = V10ClaimFixture(temporary)
            rows = read_csv(fixture.samples_path)
            varied = [1.0, 1.1, 1.2, 1.3, 5.9]
            for row in rows:
                if row["metric"] == "e2e":
                    row["seconds"] = varied[int(row["sample_index"]) - 1]
            write_csv(fixture.samples_path, rows)
            ledger = read_csv(fixture.ledger_path)
            ledger[0]["e2e_paper_seconds"] = min(varied)
            ledger[0]["e2e_raw_mean_seconds"] = statistics.fmean(varied)
            ledger[0]["e2e_std_seconds"] = statistics.stdev(varied)
            write_csv(fixture.ledger_path, ledger)

            default_report = AUDIT.audit(fixture.paper, fixture.assets)
            policy_path = fixture.assets / "eggpu_timing_stability_policy.json"
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            policy["max_over_median_limit"] = 4.0
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            strict_report = AUDIT.audit(
                fixture.paper,
                fixture.assets,
                max_over_median_limit=4.0,
                median_over_min_limit=2.0,
            )

        self.assertEqual(default_report["status"], "pass")
        e2e_dispersion = next(
            item
            for item in default_report["raw_sample_audit"]["dispersion_groups"]
            if item["metric"] == "e2e"
        )
        self.assertGreater(e2e_dispersion["cv"], 0.5)
        self.assertEqual(strict_report["status"], "fail")
        self.assertEqual(
            len(strict_report["raw_sample_audit"]["rejected_groups"]), 1
        )

    def test_default_catastrophic_outlier_guard_rejects_pathological_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = V10ClaimFixture(temporary)
            rows = read_csv(fixture.samples_path)
            pathological = [1.0, 1.1, 1.2, 1.3, 6.1]
            for row in rows:
                if row["metric"] == "e2e":
                    row["seconds"] = pathological[int(row["sample_index"]) - 1]
            write_csv(fixture.samples_path, rows)
            ledger = read_csv(fixture.ledger_path)
            ledger[0]["e2e_paper_seconds"] = min(pathological)
            ledger[0]["e2e_raw_mean_seconds"] = statistics.fmean(pathological)
            ledger[0]["e2e_std_seconds"] = statistics.stdev(pathological)
            write_csv(fixture.ledger_path, ledger)

            report = AUDIT.audit(fixture.paper, fixture.assets)

        self.assertEqual(report["status"], "fail")
        rejected = report["raw_sample_audit"]["rejected_groups"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["metric"], "e2e")
        self.assertGreater(rejected[0]["max_over_median"], 5.0)
        self.assertLessEqual(rejected[0]["median_over_min"], 3.0)

    def test_obsolete_sd_floor_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = V10ClaimFixture(temporary)
            (fixture.assets / "V10_PROTOCOL_MANIFEST.json").write_text(
                json.dumps({"sample_std_floor_seconds": 0.001}),
                encoding="utf-8",
            )
            report = AUDIT.audit(fixture.paper, fixture.assets)

        findings = [
            item
            for item in report["stale_or_protocol_wording"]
            if item["pattern"] == "obsolete_sd_hard_gate_metadata"
        ]
        self.assertEqual(report["status"], "fail")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["observed"], 0.001)


if __name__ == "__main__":
    unittest.main()
