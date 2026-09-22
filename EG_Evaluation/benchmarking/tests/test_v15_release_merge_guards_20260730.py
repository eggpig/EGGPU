from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


BENCHMARKING = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = BENCHMARKING.parent
MERGE = BENCHMARKING / "run_v15_current_runtime_merge_audit_assets_20260730.sh"
ASSEMBLY = BENCHMARKING / "run_v15_pagerank_protocol_assembly_20260730.sh"


class CanonicalMergeGuardTest(unittest.TestCase):
    def run_merge(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(MERGE), *arguments],
            cwd=EVALUATION_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=10,
        )

    def test_rejects_custom_main_timing_before_any_build(self):
        completed = self.run_merge("--main-timing-dir", "/tmp/not-authoritative")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--main-timing-dir is disabled", completed.stdout)

    def test_rejects_custom_anchor_before_any_build(self):
        completed = self.run_merge("--anchor-dir", "/tmp/not-authoritative")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--anchor-dir is disabled", completed.stdout)

    def test_release_consumes_both_hash_bound_lists(self):
        source = MERGE.read_text(encoding="utf-8")
        self.assertIn(
            '<"${pagerank_correction_root}/main_timing_dirs.txt"', source
        )
        self.assertIn(
            '<"${pagerank_correction_root}/anchor_timing_dirs.txt"', source
        )
        self.assertNotIn(
            "final_v15_current_recovery_ca_hepth_gpu0_20260730\"\n"
            "    \"${result_root}",
            source,
        )

    def test_release_generates_and_gates_headline_quartet(self):
        source = MERGE.read_text(encoding="utf-8")
        self.assertIn("generate_headline_results_20260730.py", source)
        for name in (
            "paper_table_headline_results.csv",
            "paper_table_headline_results.tex",
            "paper_table_headline_results_README.md",
            "paper_table_headline_results_manifest.json",
            "pairwise_baseline_13_exact_summary.csv",
        ):
            self.assertIn(name, source)

    def test_release_uses_mixed_five_run_estimator_contract(self):
        source = MERGE.read_text(encoding="utf-8")
        self.assertIn(
            "EGGPU_FINAL_EXPERIMENT_ASSETS_15_CURRENT_RUNTIME_"
            "MIXED_ESTIMATOR_20260730",
            source,
        )
        self.assertEqual(
            source.count("--estimator-policy eggpu-minimum-baseline-mean"),
            2,
        )
        for contract in (
            "five_run_raw_samples.csv",
            "MIXED_ESTIMATOR_FIVE_RUN_MANIFEST.json",
            "mixed_estimator_audit.csv",
            '"estimator_policy": "eggpu-minimum-baseline-mean"',
            '"display_estimator": "mixed_by_baseline_class"',
            '"external_baselines": "arithmetic_mean_of_five"',
            '"reported_error": "sample_standard_deviation_ddof1"',
            '"mixed_manifest_sha256"',
            '"mixed_estimator_audit_sha256"',
            "eggpu_headline_results_v2_mixed_estimator",
        ):
            self.assertIn(contract, source)
        for retired_contract in (
            "EGGPU_FINAL_EXPERIMENT_ASSETS_15_CURRENT_RUNTIME_"
            "UNIFORM_MIN_20260730",
            "uniform_minimum_raw_samples.csv",
            "UNIFORM_MINIMUM_OF_FIVE_MANIFEST.json",
            '"uniform_manifest_sha256"',
            "eggpu_headline_results_v1",
        ):
            self.assertNotIn(retired_contract, source)

    def test_release_figure_uses_same_five_run_error_evidence(self):
        source = MERGE.read_text(encoding="utf-8")
        self.assertIn(
            'raw_samples=output / "five_run_raw_samples.csv"',
            source,
        )
        figure_source = (
            BENCHMARKING / "generate_chapter4_evaluation_assets_graphscope.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '(("Processing time",), "Processing time")',
            figure_source,
        )
        self.assertIn("np.std(aggregate_samples, ddof=1)", figure_source)

    def test_embedded_verifier_preserves_full_matrix_and_protocol_gates(self):
        source = MERGE.read_text(encoding="utf-8")
        for contract in (
            "EXPECTED_LEDGER_ROWS = 1664",
            "EXPECTED_SUCCESSFUL_CELLS = 1040",
            "EXPECTED_DISPLAYED_METRICS = 3120",
            "EXPECTED_RAW_ROWS = 15600",
            "EXPECTED_EGGPU_CELLS = 208",
            "EXPECTED_PAGERANK_REPLACEMENTS = 11",
            "EXPECTED_PAGERANK_ANCHORS = 2",
            "EXPECTED_NON_PAGERANK_CELLS = 195",
            '"minimum_of_five"',
            '"arithmetic_mean_of_five"',
            "values.std(ddof=1)",
            "non_eggpu_evidence_fields_unchanged",
        ):
            self.assertIn(contract, source)

    def test_assembly_wrapper_requires_both_lists(self):
        source = ASSEMBLY.read_text(encoding="utf-8")
        self.assertIn(
            "for canonical_list in main_timing_dirs.txt anchor_timing_dirs.txt",
            source,
        )


if __name__ == "__main__":
    unittest.main()
