import importlib.util
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


BENCHMARKING = Path(__file__).resolve().parents[1]
if str(BENCHMARKING) not in sys.path:
    sys.path.insert(0, str(BENCHMARKING))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKING / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module("eggpu_final_gate_runner", "run_full_baselines.py")
AUDIT = load_module("eggpu_final_gate_audit", "audit_full_result.py")
VALIDATE = load_module("eggpu_final_gate_validate", "validate_correctness.py")
CORRECTNESS_GATE = load_module(
    "eggpu_final_correctness_gate", "run_eggpu_correctness_gate.py"
)
CLOSENESS_SUPPLEMENT = load_module(
    "eggpu_closeness_supplement", "run_closeness_large_supplement.py"
)


class FinalGateSemanticsTests(unittest.TestCase):
    def test_closeness_timeout_output_accepts_bytes(self):
        self.assertEqual(
            CLOSENESS_SUPPLEMENT.decode_subprocess_output(b"partial\xffoutput"),
            "partial\ufffdoutput",
        )

    def test_closeness_supplement_aggregates_mean_and_sample_sd(self):
        rows = [
            {
                "dataset": "toy",
                "baseline": "EGGPU",
                "metric": "e2e",
                "seconds": value,
                "status": "ok",
                "notes": "",
            }
            for value in (1.0, 2.0, 3.0, 4.0, 5.0)
        ]
        aggregate = CLOSENESS_SUPPLEMENT.aggregate_samples(rows, 5)[0]
        self.assertEqual(aggregate["status"], "ok")
        self.assertEqual(aggregate["sample_count"], 5)
        self.assertAlmostEqual(aggregate["mean_seconds"], 3.0)
        self.assertAlmostEqual(aggregate["std_seconds"], 2.5 ** 0.5)

    def test_structural_oracle_retains_vertices_seen_only_in_self_loops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "self_loop_vertex.txt"
            graph.write_text("0 1\n2 2\n")
            VALIDATE._STRUCTURAL_GRAPH_CACHE.clear()
            with mock.patch.object(VALIDATE, "ROOT", root), mock.patch.dict(
                VALIDATE.DATASET_PATHS,
                {"unit-self-loop": "self_loop_vertex.txt"},
                clear=False,
            ):
                normalized = VALIDATE._normalized_clean_edges("unit-self-loop")
            VALIDATE._STRUCTURAL_GRAPH_CACHE.clear()

        self.assertEqual(normalized["n"], 3)
        self.assertEqual(normalized["directed"], [(0, 1)])

    def test_correctness_gate_csv_accepts_all_metric_provenance_fields(self):
        for metric in ("build", "e2e", "kernel", "memory_peak_gpu_proc_mb"):
            descriptor = CORRECTNESS_GATE.describe_metric(metric, "EGGPU")
            self.assertTrue(
                set(descriptor).issubset(CORRECTNESS_GATE.RESULT_FIELDS),
                f"{metric} fields missing from correctness CSV: "
                f"{sorted(set(descriptor) - set(CORRECTNESS_GATE.RESULT_FIELDS))}",
            )

    def test_correctness_gate_requires_runtime_and_external_validation(self):
        datasets = [("small", "undirected", "g", "g.txt")]
        functions = ["PageRank", "BFS"]
        rows = [
            {
                "dataset": "g",
                "function": function,
                "baseline": "EGGPU",
                "metric": "e2e",
                "status": "ok",
            }
            for function in functions
        ]
        validation = [
            {
                "dataset": "g",
                "function": "PageRank",
                "baseline": "EGGPU",
                "validation_status": "pass",
            },
            {
                "dataset": "g",
                "function": "BFS",
                "baseline": "EGGPU",
                "validation_status": "fail",
            },
        ]

        issues = CORRECTNESS_GATE._eggpu_gate_issues(
            rows, validation, datasets, functions
        )

        self.assertEqual(issues, ["missing passing EGGPU validation: g/BFS"])

    def test_source_snapshot_is_content_addressed_without_git(self):
        snapshot = RUNNER.collect_source_snapshot()
        self.assertEqual(snapshot["algorithm"], "sha256")
        self.assertRegex(snapshot["digest"], r"^[0-9a-f]{64}$")
        self.assertGreater(snapshot["file_count"], 0)
        self.assertGreater(snapshot["total_bytes"], 0)

    def test_weighted_matrix_metadata_detects_disconnected_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "graph.txt"
            source.write_text("0 1\n2 3\n")
            output = root / "graph.mtx"
            returned, metadata = RUNNER.write_matrix_market(
                source,
                output,
                directed=False,
                weighted=True,
                return_metadata=True,
            )
            self.assertEqual(returned, output)
            self.assertEqual(metadata["nodes"], 4)
            self.assertEqual(metadata["components"], 2)

    def test_gunrock_validation_parser_distinguishes_zero_from_missing(self):
        self.assertEqual(
            RUNNER.parse_gunrock_error_count("Number of errors : 0"),
            0,
        )
        self.assertIsNone(
            RUNNER.parse_gunrock_error_count("GPU Elapsed Time : 1.0 (ms)")
        )

    def test_gunrock_aligned_e2e_parser_and_provenance(self):
        output = (
            "GPU Total Elapsed Time : 7.25 (ms)\n"
            "Aligned E2E Time : 9.500000 (ms)\n"
        )
        self.assertAlmostEqual(
            RUNNER.parse_gunrock_aligned_e2e_ms(output),
            0.0095,
        )
        seconds, note, extra = RUNNER.gunrock_e2e_measurement(output, 3.0)
        self.assertAlmostEqual(seconds, 0.0095)
        self.assertIn("complete device-to-host result", note)
        self.assertEqual(
            extra["measurement_window"],
            "prepared_graph_to_complete_host_result",
        )

    def test_gunrock_e2e_fallback_remains_explicit(self):
        seconds, note, extra = RUNNER.gunrock_e2e_measurement(
            "GPU Elapsed Time : 7.25 (ms)",
            3.0,
        )
        self.assertEqual(seconds, 3.0)
        self.assertIn("falls back", note)
        self.assertIsNone(extra)

    def test_gunrock_runtime_enables_aligned_e2e_adapter(self):
        env = RUNNER.gunrock_runtime_env(Path("/tmp/gunrock/bin/bfs"), {})
        self.assertEqual(env["EGGPU_GUNROCK_ALIGNED_E2E"], "TRUE")

    def test_idle_device_resident_memory_is_not_external_contamination(self):
        metadata = {
            "gpu_device_profile": {
                "selected_device": {
                    "memory_used_mb_at_run_start": 1449.0625,
                    "compute_process_memory_mb_at_run_start": 672.0,
                    "compute_process_count_at_run_start": 1,
                    "gpu_utilization_percent_at_run_start": 0,
                }
            }
        }
        rows = [
            {
                "dataset": "g",
                "function": "PageRank",
                "baseline": "EGGPU",
                "status": "ok",
                "metric": "memory_start_gpu_non_process_mb",
                "value": "768.25",
            }
        ]
        self.assertEqual(AUDIT.audit_memory_rows(rows, metadata), [])

        rows[0]["value"] = "1200"
        issues = AUDIT.audit_memory_rows(rows, metadata)
        self.assertEqual(len(issues), 1)
        self.assertEqual(
            issues[0]["issue"],
            "non_benchmark_gpu_memory_present_at_measurement_start",
        )

    def test_pagerank_accepts_aligned_residual_despite_different_stopping_api(self):
        import numpy as np

        reference = {"baseline": "networkx"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "toy.txt"
            graph.write_text("0 1\n1 0\n")
            detail = root / "pagerank.npz"
            np.savez(detail, kind=np.asarray("vector"), values=np.asarray([0.5, 0.5]))
            VALIDATE._STRUCTURAL_GRAPH_CACHE.clear()
            VALIDATE._PAGERANK_TRANSITION_CACHE.clear()
            with mock.patch.object(VALIDATE, "ROOT", root), mock.patch.dict(
                VALIDATE.DATASET_PATHS, {"toy-pr": "toy.txt"}, clear=False
            ):
                status, details = VALIDATE._validate_one(
                    {
                        "dataset": "toy-pr",
                        "graph_type": "directed",
                        "function": "PageRank",
                        "baseline": "igraph",
                        "notes": "alpha=0.75",
                        "correctness": f"sum=1, detail={detail}, detail_kind=vector",
                    },
                    reference,
                    {"sum": 1.0},
                )
            VALIDATE._STRUCTURAL_GRAPH_CACHE.clear()
            VALIDATE._PAGERANK_TRANSITION_CACHE.clear()
        self.assertEqual(status, "pass")
        self.assertIn("fixed-point residual passes", details)

    def test_gunrock_pagerank_parameter_mismatch_remains_excluded(self):
        reference = {"baseline": "networkx"}
        status, _ = VALIDATE._validate_one(
            {
                "function": "PageRank",
                "baseline": "Gunrock",
                "notes": "parameter contract is not aligned",
                "correctness": "",
            },
            reference,
            {"sum": 1.0},
        )
        self.assertEqual(status, "semantic_mismatch")

    def test_mst_float_reduction_uses_declared_relative_tolerance(self):
        status, details = VALIDATE._validate_one(
            {
                "function": "MST",
                "baseline": "Gunrock",
                "notes": "connected input",
                "correctness": "weight=504180473856",
            },
            {"baseline": "networkx"},
            {"weight": 504223349391},
        )
        self.assertEqual(status, "pass")
        self.assertIn("summary fields match", details)

    def test_directed_igraph_constraint_is_a_documented_semantic_mismatch(self):
        status, details = VALIDATE._validate_one(
            {
                "function": "Constraint",
                "baseline": "igraph",
                "graph_type": "directed",
                "correctness": "nodes=3,sum=1",
            },
            {"baseline": "easygraph-cpu"},
            {"nodes": 3, "sum": 1.0},
        )

        self.assertEqual(status, "semantic_mismatch")
        self.assertIn("detail file missing", details)

    def test_directed_igraph_constraint_defined_domain_is_partial_support(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lhs = root / "igraph.npz"
            rhs = root / "easygraph.npz"
            np.savez(lhs, kind=np.asarray("vector"), values=np.asarray([0.5, 0.25]))
            np.savez(rhs, kind=np.asarray("vector"), values=np.asarray([0.5, np.nan]))
            status, details = VALIDATE._validate_one(
                {
                    "function": "Constraint",
                    "baseline": "igraph",
                    "graph_type": "directed",
                    "correctness": f"nodes=2, detail={lhs}, detail_kind=vector",
                },
                {"baseline": "easygraph-cpu"},
                {"nodes": 2, "detail": str(rhs), "detail_kind": "vector"},
            )
        self.assertEqual(status, "partial_pass")
        self.assertIn("undefined-vertex convention differs", details)

    def test_detail_comparison_reports_nonfinite_mask_mismatch(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lhs = root / "lhs.npz"
            rhs = root / "rhs.npz"
            np.savez(lhs, kind=np.asarray("vector"), values=np.asarray([1.0, np.nan]))
            np.savez(rhs, kind=np.asarray("vector"), values=np.asarray([1.0, 2.0]))

            ok, details = VALIDATE._compare_details(
                "Constraint",
                {"detail": str(lhs)},
                {"detail": str(rhs)},
            )

        self.assertFalse(ok)
        self.assertIn("non-finite masks differ", details)
        self.assertIn("nan=1", details)

    def test_smoke_audit_declares_its_repeat_contract(self):
        gate_source = (BENCHMARKING.parent / "run_final_gpu5_gate_and_full.sh").read_text()
        self.assertIn('audit_full_result.py "${SMOKE_MAIN}" --expected-repeat 1', gate_source)
        self.assertIn('FINAL_SMOKE_TIMEOUT="${FINAL_SMOKE_TIMEOUT:-20}"', gate_source)
        self.assertIn('--library-timeout "${FINAL_SMOKE_TIMEOUT}"', gate_source)
        self.assertIn("--functions all", gate_source)
        self.assertIn("FINAL_GATE_ONLY", gate_source)

        split_source = (BENCHMARKING / "run_split_full_baselines.py").read_text()
        self.assertIn("source_snapshot_match", split_source)
        self.assertIn(
            "timing/memory EGGPU implementation snapshot mismatch",
            split_source,
        )
        self.assertIn("timing/memory cpp_easygraph artifact mismatch", split_source)

        runner_source = (BENCHMARKING / "run_full_baselines.py").read_text()
        # The maintained Gunrock adapters now pass the complete source set to
        # one process for weighted paths, BFS, and BC.  Each invocation therefore
        # receives the normal per-function timeout instead of implementing an
        # obsolete per-source process loop and timeout debit.
        self.assertGreaterEqual(
            runner_source.count(
                'source_arg = ",".join(str(int(source)) for source in sources)'
            ),
            3,
        )
        self.assertGreaterEqual(
            runner_source.count("one Gunrock process"),
            3,
        )

    def test_marker_failure_disables_optional_marker_instead_of_stopping_run(self):
        gate_source = (BENCHMARKING.parent / "run_final_gpu5_gate_and_full.sh").read_text()
        complete_source = (
            BENCHMARKING.parent / "run_complete_paper_experiments.sh"
        ).read_text()
        self.assertIn("MARKER_CHECK_RC=$?", gate_source)
        self.assertIn('MARKER_ENABLED="FALSE"', gate_source)
        self.assertIn("disabling it and continuing", gate_source)
        self.assertIn("FINAL_MARKER_DECISION_FILE", complete_source)
        self.assertIn("experiment continues without it", complete_source)

    def test_metric_tables_exclude_failed_correctness_rows_but_keep_raw_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "correctness_validation.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("dataset", "function", "baseline", "validation_status"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "g",
                        "function": "MST",
                        "baseline": "Gunrock",
                        "validation_status": "fail",
                    }
                )
                writer.writerow(
                    {
                        "dataset": "g",
                        "function": "MST",
                        "baseline": "EGGPU",
                        "validation_status": "pass",
                    }
                )
            rows = [
                {
                    "dataset": "g",
                    "function": "MST",
                    "baseline": baseline,
                    "metric": "e2e",
                    "seconds": value,
                    "status": "ok",
                }
                for baseline, value in (("Gunrock", "1"), ("EGGPU", "2"))
            ]
            RUNNER.write_metric_csvs_no_pandas(root, rows)
            with (root / "results_e2e.csv").open(newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual([row["baseline"] for row in written], ["EGGPU"])

    def test_metric_csv_writer_accepts_heterogeneous_provenance_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "dataset": "g",
                    "function": "BFS",
                    "baseline": "EGGPU",
                    "metric": "e2e",
                    "seconds": "1",
                    "status": "ok",
                },
                {
                    "dataset": "g",
                    "function": "BFS",
                    "baseline": "Gunrock",
                    "metric": "e2e",
                    "seconds": "2",
                    "status": "ok",
                    "external_cli_wall_seconds": "3",
                },
            ]
            RUNNER.write_metric_csvs_no_pandas(root, rows)
            with (root / "results_e2e.csv").open(newline="") as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(len(written), 2)
            self.assertIn("external_cli_wall_seconds", written[0])
            self.assertEqual(written[1]["external_cli_wall_seconds"], "3")


if __name__ == "__main__":
    unittest.main()
