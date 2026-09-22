from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import textwrap
import unittest
from pathlib import Path


BENCHMARKING = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKING / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ASSEMBLER = load_module(
    "v15_pr_assembler",
    "assemble_v15_pagerank_protocol_correction_20260730.py",
)
VERIFIER = load_module(
    "v15_pr_verifier",
    "verify_v15_pagerank_worker_protocol_20260730.py",
)


SAMPLE_FIELDS = [
    "dataset_size",
    "graph_type",
    "dataset",
    "function",
    "baseline",
    "metric",
    "seconds",
    "value",
    "unit",
    "metric_family",
    "measurement_scope",
    "timer_kind",
    "measurement_window",
    "status",
    "correctness",
    "log",
    "notes",
    "sample_index",
    "sample_count",
    "measurement_phase",
]
LONG_FIELDS = [
    *SAMPLE_FIELDS,
    "n_total",
    "n_valid",
    "publishable",
    "aggregation",
    "mean_seconds",
    "std_seconds",
    "min_seconds",
    "max_seconds",
    "cv",
]
VALIDATION_FIELDS = [
    "dataset_size",
    "graph_type",
    "dataset",
    "function",
    "baseline",
    "reference",
    "validation_status",
    "details",
    "correctness",
    "reference_correctness",
]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metadata(functions: list[str]) -> dict:
    return {
        "schema_version": 1,
        "benchmark_args": {
            "baselines": ["EGGPU"],
            "functions": functions,
            "repeat": 5,
            "warmup": 2,
            "easygraph_warmup": 2,
            "pr_alpha": 0.75,
            "pr_eps": 1.0e-6,
            "pr_max_iter": 200,
            "eggpu_execution_protocol": "steady-state",
            "measurement_mode": "timing",
        },
        "repository_runtime_provenance": {
            "native_sha256": ASSEMBLER.EXPECTED_NATIVE_SHA256,
            "runtime_python_snapshot": {
                "digest": ASSEMBLER.EXPECTED_RUNTIME_SHA256,
                "package_is_symlink": False,
            },
        },
        "runtime_python_snapshot": {
            "digest": ASSEMBLER.EXPECTED_RUNTIME_SHA256,
            "package_is_symlink": False,
        },
    }


def attestation() -> dict:
    return {
        "status": "pass",
        "protocol": ASSEMBLER.PROTOCOL_NAME,
        "source_file": "/synthetic/library_baselines.py",
        "source_sha256": "a" * 64,
        "signature_preflight": {
            "status": "pass",
            "call": "inspect.signature(eg.pagerank)",
            "outside_timer": True,
        },
        "timed_public_call": {
            "status": "pass",
            "call": "eg.pagerank",
            "graph_argument": "g",
            "keyword_bindings": {
                "alpha": "pr_alpha",
                "max_iter": "pr_max_iter",
                "tol": "pr_tol",
            },
            "weight": None,
            "return_only_callable": True,
            "generic_adapter_absent": True,
        },
        "result_validation": {
            "status": "pass",
            "call": "validate_pagerank_result",
            "outside_timer": True,
            "checks": ["shape", "finite", "nonnegative", "unit_sum"],
        },
        "kernel_timing": {
            "status": "pass",
            "call": "require_kernel_time",
            "missing_value_policy": "fail_closed",
            "algorithm_time_fallback_absent": True,
        },
    }


def sample_row(
    dataset: str,
    function: str,
    metric: str,
    index: int,
    seconds: float,
) -> dict:
    contracts = {
        "build": (
            "baseline_graph_construction",
            "perf_counter_wall",
            "graph_construction",
        ),
        "e2e": (
            "user_visible_function_call",
            "perf_counter_wall",
            "python_function_call",
        ),
        "kernel": ("algorithm_compute", "cuda_event", "device_execution"),
    }
    scope, timer, window = contracts[metric]
    correctness = "sum=1, detail=/synthetic/vector.npz"
    if function != "PageRank":
        correctness = f"nodes=7, checksum={index}"
    return {
        "dataset_size": "medium",
        "graph_type": "undirected",
        "dataset": dataset,
        "function": function,
        "baseline": "EGGPU",
        "metric": metric,
        "seconds": f"{seconds:.9f}",
        "value": f"{seconds:.9f}",
        "unit": "s",
        "metric_family": "time",
        "measurement_scope": scope,
        "timer_kind": timer,
        "measurement_window": window,
        "status": "ok",
        "correctness": correctness,
        "log": f"/synthetic/{dataset}_{function}_{index}.log",
        "notes": "synthetic",
        "sample_index": str(index),
        "sample_count": "5",
        "measurement_phase": "timing",
    }


def aggregate_row(
    dataset: str,
    function: str,
    metric: str,
    samples: list[dict],
) -> dict:
    values = [float(row["seconds"]) for row in samples]
    base = dict(samples[0])
    base["sample_index"] = ""
    base.update(
        {
            "n_total": "5",
            "n_valid": "5",
            "publishable": "true",
            "aggregation": "arithmetic_mean",
            "mean_seconds": str(sum(values) / len(values)),
            "std_seconds": "0",
            "min_seconds": str(min(values)),
            "max_seconds": str(max(values)),
            "cv": "0",
        }
    )
    return base


def validation_row(dataset: str, function: str) -> dict:
    status = "pass" if function == "PageRank" else "inconclusive_self_reference"
    return {
        "dataset_size": "medium",
        "graph_type": "undirected",
        "dataset": dataset,
        "function": function,
        "baseline": "EGGPU",
        "reference": "EGGPU",
        "validation_status": status,
        "details": "synthetic invariant validation",
        "correctness": "sum=1",
        "reference_correctness": "sum=1",
    }


def create_regular_result(
    root: Path,
    datasets: tuple[str, ...],
    functions: tuple[str, ...],
    *,
    corrected: bool,
    invalid_kernel: bool = False,
) -> None:
    root.mkdir(parents=True)
    pagerank_details: dict[str, str] = {}
    if corrected:
        import numpy as np

        for dataset in datasets:
            detail_dir = root / "logs" / dataset / "details"
            detail_dir.mkdir(parents=True)
            detail_path = detail_dir / "EGGPU_PageRank.npz"
            values = np.asarray([0.25, 0.75], dtype=np.float64)
            np.savez_compressed(detail_path, kind="vector", values=values)
            pagerank_details[dataset] = (
                f"sum=1, detail={detail_path.resolve()}, detail_kind=vector, "
                f"detail_sha={ASSEMBLER.array_digest(values)}"
            )
    samples = []
    aggregates = []
    validation = []
    for dataset in datasets:
        for function in functions:
            per_metric = {}
            for metric in ASSEMBLER.METRICS:
                metric_rows = []
                for index in range(1, 6):
                    if corrected and function == "PageRank":
                        values = {"build": 4.0, "e2e": 9.0, "kernel": 8.0}
                        if invalid_kernel and metric == "kernel":
                            values["kernel"] = 10.0
                    else:
                        values = {"build": 2.0, "e2e": 1.0, "kernel": 0.5}
                    row = sample_row(
                        dataset,
                        function,
                        metric,
                        index,
                        values[metric] + index * 1.0e-4,
                    )
                    if corrected and function == "PageRank":
                        row["correctness"] = pagerank_details[dataset]
                    samples.append(row)
                    metric_rows.append(row)
                per_metric[metric] = metric_rows
                aggregates.append(
                    aggregate_row(dataset, function, metric, metric_rows)
                )
            validation.append(validation_row(dataset, function))

    write_csv(root / "results_samples.csv", SAMPLE_FIELDS, samples)
    write_csv(root / "results_long.csv", LONG_FIELDS, aggregates)
    for metric in ASSEMBLER.METRICS:
        write_csv(
            root / f"results_{metric}.csv",
            LONG_FIELDS,
            [row for row in aggregates if row["metric"] == metric],
        )
    write_csv(
        root / "correctness_validation.csv",
        VALIDATION_FIELDS,
        validation,
    )
    (root / "run_metadata.json").write_text(
        json.dumps(metadata(list(functions)), indent=2) + "\n",
        encoding="utf-8",
    )
    if corrected:
        (root / "pagerank_protocol_attestation.json").write_text(
            json.dumps(attestation(), indent=2) + "\n",
            encoding="utf-8",
        )


def create_anchor(
    root: Path,
    dataset: str,
    *,
    validation_outside_timer: bool = True,
) -> None:
    root.mkdir(parents=True)
    raw = root / "raw"
    raw.mkdir()
    summary_rows = []
    for function in ASSEMBLER.FUNCTIONS:
        summary_rows.append(
            {
                "status": "ok",
                "dataset": dataset,
                "function": function,
                "measurement": "timing",
                "result_validation": "pass",
            }
        )
        payload = {
            "status": "ok",
            "dataset": dataset,
            "function": function,
            "measurement": "timing",
            "timing_process_samples": 5,
            "load": {"samples": [2.0 + i * 0.01 for i in range(5)]},
            "steady_e2e": {
                "samples": [1.0 + i * 0.01 for i in range(5)]
            },
            "steady_kernel": {
                "samples": [0.5 + i * 0.005 for i in range(5)]
            },
            "result_validation": {"status": "pass", "failures": []},
            "timer_boundary": (
                "EGGPU public invocation through complete result return; "
                "benchmark validation excluded"
            ),
            "validation_outside_timer": validation_outside_timer,
        }
        (raw / f"{dataset}_{function}_timing.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    write_csv(
        root / "scaling_all.csv",
        ["status", "dataset", "function", "measurement", "result_validation"],
        summary_rows,
    )
    (root / "run_metadata.json").write_text(
        json.dumps(metadata(list(ASSEMBLER.FUNCTIONS)), indent=2) + "\n",
        encoding="utf-8",
    )


def strict_worker_source(
    *,
    direct_return: str = (
        "return eg.pagerank(g, alpha=pr_alpha, max_iter=pr_max_iter, "
        "tol=pr_tol, weight=None)"
    ),
    extra_timed_statement: str = "",
    validator_body: str | None = None,
    kernel_body: str | None = None,
) -> str:
    validator_body = validator_body or """
values = list(ranks.values()) if hasattr(ranks, "values") else list(ranks)
if len(values) != n:
    raise RuntimeError("shape")
if not all(math.isfinite(value) for value in values):
    raise RuntimeError("finite")
if any(value < -1e-12 for value in values):
    raise RuntimeError("negative")
total = math.fsum(values)
if not math.isclose(total, 1.0):
    raise RuntimeError("sum")
return total
"""
    kernel_body = kernel_body or """
kernel_value = eggpu_backend.get_last_kernel_time(kernel_key)
if kernel_value is None:
    raise RuntimeError("missing")
if not math.isfinite(kernel_value):
    raise RuntimeError("nonfinite")
if kernel_value <= 0:
    raise RuntimeError("nonpositive")
return float(kernel_value)
"""

    def indent(body: str, spaces: int) -> str:
        return textwrap.indent(textwrap.dedent(body).strip(), " " * spaces)

    timed_lines = []
    if extra_timed_statement:
        timed_lines.append(extra_timed_statement)
    timed_lines.append(direct_return)
    timed_body = "\n".join(f"                {line}" for line in timed_lines)
    return (
        "import inspect\n"
        "import math\n"
        "def validate_pagerank_result(ranks, n):\n"
        f"{indent(validator_body, 4)}\n"
        "def require_kernel_time(kernel_key):\n"
        f"{indent(kernel_body, 4)}\n"
        "def bench_easygraph_mode(functions, eg, g, n, timed_algorithm, "
        "pr_alpha, pr_max_iter, pr_tol):\n"
        "    signature = inspect.signature(eg.pagerank)\n"
        "    for func in functions:\n"
        '        if func == "PageRank":\n'
        "            def invoke_pagerank_direct():\n"
        f"{timed_body}\n"
        "            ranks, algo_s, mem = timed_algorithm(invoke_pagerank_direct)\n"
        "            validate_pagerank_result(ranks, n)\n"
        '            kernel_s = require_kernel_time("pagerank")\n'
        '            emit_metrics("EGGPU", "PageRank", 0.1, algo_s, kernel_s)\n'
    )


class ProtocolVerifierTest(unittest.TestCase):
    def assert_protocol_rejected(self, source_text: str, pattern: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "worker.py"
            source.write_text(source_text, encoding="utf-8")
            with self.assertRaisesRegex(VERIFIER.ProtocolError, pattern):
                VERIFIER.attest(source)

    def test_accepts_direct_fail_closed_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "worker.py"
            source.write_text(
                strict_worker_source(),
                encoding="utf-8",
            )
            result = VERIFIER.attest(source)
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["result_validation"]["outside_timer"])

    def test_rejects_generic_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "worker.py"
            source.write_text(
                """
import inspect
def bench_easygraph_mode(functions, eg, g, timed_algorithm):
    signature = inspect.signature(eg.pagerank)
    for func in functions:
        if func == "PageRank":
            def invoke():
                return call_with_supported_kwargs(eg.pagerank, {"G": g})
            ranks, algo_s, mem = timed_algorithm(invoke)
            validate_pagerank_result(ranks)
            kernel_s = require_kernel_time("pagerank")
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                VERIFIER.ProtocolError, "call_with_supported_kwargs"
            ):
                VERIFIER.attest(source)

    def test_rejects_wrong_direct_argument_bindings(self):
        self.assert_protocol_rejected(
            strict_worker_source(
                direct_return=(
                    "return eg.pagerank(other_graph, alpha=0.75, "
                    "max_iter=pr_max_iter, tol=pr_tol, weight=None)"
                )
            ),
            "positional graph argument g",
        )
        self.assert_protocol_rejected(
            strict_worker_source(
                direct_return=(
                    "return eg.pagerank(g, alpha=0.75, "
                    "max_iter=pr_max_iter, tol=pr_tol, weight=None)"
                )
            ),
            "alpha must use pr_alpha",
        )

    def test_rejects_noop_validator(self):
        self.assert_protocol_rejected(
            strict_worker_source(
                validator_body="""
return None
"""
            ),
            "not fail-closed",
        )

    def test_rejects_kernel_fallback(self):
        self.assert_protocol_rejected(
            strict_worker_source(
                kernel_body="""
kernel_value = eggpu_backend.get_last_kernel_time(kernel_key)
if kernel_value is None:
    return algo_seconds
return float(kernel_value)
"""
            ),
            "require_kernel_time",
        )

    def test_rejects_extra_timed_work(self):
        self.assert_protocol_rejected(
            strict_worker_source(extra_timed_statement="marker = 1"),
            "body must be exactly one direct return",
        )


class CorrectionAssemblerTest(unittest.TestCase):
    def make_inputs(self, root: Path, *, invalid_kernel: bool = False):
        original = root / "original"
        correction = root / "correction"
        orkut = root / "orkut"
        gap = root / "gap"
        create_regular_result(
            original,
            ASSEMBLER.REGULAR_REPLACEMENTS,
            ASSEMBLER.FUNCTIONS,
            corrected=False,
        )
        create_regular_result(
            correction,
            ASSEMBLER.REGULAR_REPLACEMENTS,
            ("PageRank",),
            corrected=True,
            invalid_kernel=invalid_kernel,
        )
        create_anchor(orkut, "com-Orkut")
        create_anchor(gap, "GAP-twitter")
        return original, correction, orkut, gap

    def test_partial_batch_validation_does_not_require_anchors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            create_regular_result(
                first,
                ("LastFM", "web-NotreDame"),
                ("PageRank",),
                corrected=True,
            )
            create_regular_result(
                second,
                ("com-youtube",),
                ("PageRank",),
                corrected=True,
            )
            audit = ASSEMBLER.validate_correction_batch([first, second])
            self.assertEqual(audit["status"], "pass")
            self.assertEqual(audit["dataset_count"], 3)
            self.assertFalse(audit["complete_11_dataset_set"])
            self.assertIn("ca-HepTh", audit["remaining_datasets"])

    def test_unconditional_slow_replacement_and_195_cell_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, correction, orkut, gap = self.make_inputs(root)
            output = root / "assembled"
            audit = ASSEMBLER.assemble(
                original_dirs=[original],
                correction_dirs=[correction],
                supplemental_dirs=[],
                anchor_dirs=[orkut, gap],
                output_root=output,
            )
            self.assertEqual(audit["actions"]["replace_protocol_invalid"], 11)
            self.assertEqual(
                audit["actions"]["retain_equivalent_direct_protocol"], 2
            )
            self.assertEqual(
                audit["non_pagerank_identity"]["normalized_cell_count"], 195
            )
            self.assertEqual(
                audit["non_pagerank_identity"]["before_sha256"],
                audit["non_pagerank_identity"]["after_sha256"],
            )
            combined = next((output / "combined_main_timing").iterdir())
            _fields, rows = ASSEMBLER.read_csv(
                combined / "results_samples.csv"
            )
            corrected_e2e = [
                float(row["seconds"])
                for row in rows
                if row["dataset"] == "ca-HepTh"
                and row["function"] == "PageRank"
                and row["metric"] == "e2e"
            ]
            self.assertEqual(len(corrected_e2e), 5)
            self.assertGreater(min(corrected_e2e), 8.0)
            verified = ASSEMBLER.verify_existing(output)
            self.assertEqual(verified["status"], "pass")

    def test_rejects_kernel_not_below_e2e(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, correction, orkut, gap = self.make_inputs(
                root, invalid_kernel=True
            )
            with self.assertRaisesRegex(
                ASSEMBLER.GateError, "0 < kernel < e2e"
            ):
                ASSEMBLER.assemble(
                    original_dirs=[original],
                    correction_dirs=[correction],
                    supplemental_dirs=[],
                    anchor_dirs=[orkut, gap],
                    output_root=root / "assembled",
                )

    def test_rejects_nonpass_pagerank_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            correction = root / "correction"
            create_regular_result(
                correction,
                ("ca-HepTh",),
                ("PageRank",),
                corrected=True,
            )
            fields, rows = ASSEMBLER.read_csv(
                correction / "correctness_validation.csv"
            )
            rows[0]["validation_status"] = "inconclusive_self_reference"
            ASSEMBLER.write_csv(
                correction / "correctness_validation.csv", fields, rows
            )
            with self.assertRaisesRegex(
                ASSEMBLER.GateError, "validation status must be 'pass'"
            ):
                ASSEMBLER.validate_correction_batch([correction])

    def test_rejects_seconds_value_mismatch_and_detail_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            correction = root / "correction"
            create_regular_result(
                correction,
                ("ca-HepTh",),
                ("PageRank",),
                corrected=True,
            )
            fields, rows = ASSEMBLER.read_csv(
                correction / "results_samples.csv"
            )
            rows[0]["value"] = str(float(rows[0]["seconds"]) * 2.0)
            ASSEMBLER.write_csv(correction / "results_samples.csv", fields, rows)
            with self.assertRaisesRegex(
                ASSEMBLER.GateError, "seconds/value differ"
            ):
                ASSEMBLER.validate_correction_batch([correction])

        with tempfile.TemporaryDirectory() as temporary:
            import numpy as np

            root = Path(temporary)
            correction = root / "correction"
            create_regular_result(
                correction,
                ("ca-HepTh",),
                ("PageRank",),
                corrected=True,
            )
            detail = (
                correction
                / "logs"
                / "ca-HepTh"
                / "details"
                / "EGGPU_PageRank.npz"
            )
            np.savez_compressed(
                detail,
                kind="vector",
                values=np.asarray([0.4, 0.4], dtype=np.float64),
            )
            with self.assertRaisesRegex(
                ASSEMBLER.GateError, "detail sum|detail digest"
            ):
                ASSEMBLER.validate_correction_batch([correction])

    def test_rejects_anchor_validation_inside_timer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original"
            correction = root / "correction"
            orkut = root / "orkut"
            gap = root / "gap"
            create_regular_result(
                original,
                ASSEMBLER.REGULAR_REPLACEMENTS,
                ASSEMBLER.FUNCTIONS,
                corrected=False,
            )
            create_regular_result(
                correction,
                ASSEMBLER.REGULAR_REPLACEMENTS,
                ("PageRank",),
                corrected=True,
            )
            create_anchor(
                orkut,
                "com-Orkut",
                validation_outside_timer=False,
            )
            create_anchor(gap, "GAP-twitter")
            with self.assertRaisesRegex(
                ASSEMBLER.GateError, "validation is not outside timer"
            ):
                ASSEMBLER.assemble(
                    original_dirs=[original],
                    correction_dirs=[correction],
                    supplemental_dirs=[],
                    anchor_dirs=[orkut, gap],
                    output_root=root / "assembled",
                )

    def test_verify_detects_combined_file_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, correction, orkut, gap = self.make_inputs(root)
            output = root / "assembled"
            ASSEMBLER.assemble(
                original_dirs=[original],
                correction_dirs=[correction],
                supplemental_dirs=[],
                anchor_dirs=[orkut, gap],
                output_root=output,
            )
            combined = next((output / "combined_main_timing").iterdir())
            with (combined / "results_samples.csv").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("\n")
            with self.assertRaisesRegex(
                ASSEMBLER.GateError, "generated correction evidence changed"
            ):
                ASSEMBLER.verify_existing(output)

    def test_verify_rejects_semantically_changed_path_lists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, correction, orkut, gap = self.make_inputs(root)
            output = root / "assembled"
            ASSEMBLER.assemble(
                original_dirs=[original],
                correction_dirs=[correction],
                supplemental_dirs=[],
                anchor_dirs=[orkut, gap],
                output_root=output,
            )
            audit_path = (
                output / "V15_PAGERANK_PROTOCOL_CORRECTION_AUDIT.json"
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            main_list = output / "main_timing_dirs.txt"
            main_list.write_text(f"{orkut.resolve()}\n", encoding="utf-8")
            audit["output_file_sha256"][
                "main_timing_dirs.txt"
            ] = ASSEMBLER.sha256(main_list)
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ASSEMBLER.GateError,
                "main timing directory list differs semantically",
            ):
                ASSEMBLER.verify_existing(output)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, correction, orkut, gap = self.make_inputs(root)
            output = root / "assembled"
            ASSEMBLER.assemble(
                original_dirs=[original],
                correction_dirs=[correction],
                supplemental_dirs=[],
                anchor_dirs=[orkut, gap],
                output_root=output,
            )
            audit_path = (
                output / "V15_PAGERANK_PROTOCOL_CORRECTION_AUDIT.json"
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            anchor_list = output / "anchor_timing_dirs.txt"
            anchor_list.write_text(
                f"{orkut.resolve()}\n{gap.resolve()}\n", encoding="utf-8"
            )
            audit["output_file_sha256"][
                "anchor_timing_dirs.txt"
            ] = ASSEMBLER.sha256(anchor_list)
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ASSEMBLER.GateError,
                "anchor timing directory list differs semantically",
            ):
                ASSEMBLER.verify_existing(output)


if __name__ == "__main__":
    unittest.main()
