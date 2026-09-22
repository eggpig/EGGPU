import csv
import io
import json
import statistics
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import audit_v15_stage4_outputs_20260730 as audit


NATIVE_SHA = "a" * 64
PYTHON_SHA = "b" * 64


def runtime_provenance():
    return {
        "native_sha256": NATIVE_SHA,
        "resolved_root": "/frozen/runtime",
        "modules": {
            "easygraph": {
                "sha256": "d" * 64,
                "relative_to_runtime": "easygraph/__init__.py",
            },
            "cpp_easygraph": {
                "sha256": NATIVE_SHA,
                "relative_to_runtime": "cpp_easygraph.so",
            },
        },
        "runtime_python_snapshot": {"digest": PYTHON_SHA},
    }


def loaded_runtime_provenance():
    provenance = runtime_provenance()
    provenance.pop("runtime_python_snapshot")
    return provenance


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_intro(root):
    root.mkdir(parents=True)
    (root / "run_metadata.json").write_text(
        json.dumps(
            {
                "native_binary_sha256": NATIVE_SHA,
                "runtime_python_digest": PYTHON_SHA,
                "repeat": 5,
                "warmup": 2,
                "timing_processes": 0,
                "memory_repeat": 0,
                "requested_functions": "PageRank",
                "repository_runtime_provenance": runtime_provenance(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    json_rows = []
    raw = root / "raw"
    raw.mkdir()
    for dataset, (scale, nodes) in audit.INTRO_DATASETS.items():
        samples = [0.01 * scale + index * 0.001 for index in range(1, 6)]
        rows.append(
            {
                "dataset": dataset,
                "status": "ok",
                "measurement": "timing",
                "function": "PageRank",
                "num_nodes": nodes,
                "directed": True,
                "rmat_scale": scale,
                "rmat_edge_factor": 16,
                "timing_process_samples": 5,
                "first_use_calls": 1,
                "additional_warmup_calls": 2,
                "preceding_public_calls": 3,
                "measured_call_position": 4,
                "measured_calls_per_process": 1,
                "paper_estimator": "minimum_of_five",
                "result_validation": "pass",
                "validation_outside_timer": True,
                "timer_boundary": audit.INTRO_TIMER_BOUNDARY,
                "steady_e2e_samples": json.dumps(samples),
                "steady_e2e_mean": statistics.mean(samples),
                "steady_e2e_stdev": statistics.stdev(samples),
                "steady_e2e_minimum": min(samples),
                "submission_e2e_seconds": min(samples),
            }
        )
        json_rows.append(
            {
                "dataset": dataset,
                "status": "ok",
                "measurement": "timing",
                "function": "PageRank",
                "preceding_public_calls": 3,
                "measured_call_position": 4,
                "validation_outside_timer": True,
                "steady_e2e": {"samples": samples},
            }
        )
        for index, seconds in enumerate(samples, start=1):
            payload = {
                "dataset": dataset,
                "status": "ok",
                "measurement": "timing",
                "function": "PageRank",
                "timing_process_index": index,
                "preceding_public_calls": 3,
                "measured_call_position": 4,
                "measured_calls_per_process": 1,
                "timer_boundary": audit.INTRO_TIMER_BOUNDARY,
                "validation_outside_timer": True,
                "result_validation": {"status": "pass"},
                "runtime_provenance": loaded_runtime_provenance(),
                "steady_e2e": {"samples": [seconds]},
            }
            path = raw / f"{dataset}_PageRank_timing_{index}.json"
            path.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    write_csv(root / "scaling_all.csv", rows)
    (root / "scaling_all.json").write_text(
        json.dumps(json_rows, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_workflow(root):
    root.mkdir(parents=True)
    provenance = runtime_provenance()
    loaded_provenance = loaded_runtime_provenance()
    metadata = {
        "protocol": "fresh_process_cumulative_same_graph_workflow_v2",
        "datasets": list(audit.WORKFLOW_DATASETS),
        "baselines": list(audit.WORKFLOW_BASELINES),
        "workflow": [function for _, function in audit.WORKFLOW_CALLS],
        "repeat": 5,
        "sources": 8,
        "failures": 0,
        "unsupported_calls": [],
        "validation_outside_timer": True,
        "runtime_provenance": provenance,
        "observed_eggpu_runtimes": [loaded_provenance],
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    raw_dir = root / "raw"
    raw_dir.mkdir()
    for dataset_index, dataset in enumerate(audit.WORKFLOW_DATASETS, start=1):
        for baseline_index, baseline in enumerate(
            audit.WORKFLOW_BASELINES, start=1
        ):
            for sample in range(1, 6):
                raw_rows = []
                cumulative = 0.0
                for position, function in audit.WORKFLOW_CALLS:
                    seconds = (
                        dataset_index * 0.001
                        + baseline_index * 0.002
                        + sample * 0.003
                        + position * 0.01
                    )
                    cumulative += seconds
                    row = {
                        "dataset": dataset,
                        "baseline": baseline,
                        "sample_index": sample,
                        "call_position": position,
                        "function": function,
                        "status": "ok",
                        "result_validation": "pass",
                        "validation_outside_timer": True,
                        "timer_boundary": audit.WORKFLOW_TIMER_BOUNDARY,
                        "call_seconds": seconds,
                        "cumulative_seconds": cumulative,
                        "runtime_provenance": loaded_provenance,
                    }
                    raw_rows.append(row)
                    flattened = dict(row)
                    flattened["runtime_provenance"] = json.dumps(
                        loaded_provenance, sort_keys=True
                    )
                    rows.append(flattened)
                raw_payload = {
                    "rows": raw_rows,
                    "runtime_provenance": loaded_provenance,
                }
                raw_path = (
                    raw_dir / f"{dataset}_{baseline}_{sample}.json"
                )
                raw_path.write_text(
                    json.dumps(raw_payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    write_csv(root / "cumulative_workflow_samples.csv", rows)
    empty_fields = ["dataset", "baseline", "sample_index", "error"]
    write_csv(
        root / "cumulative_workflow_failures.csv", [], empty_fields
    )
    write_csv(
        root / "cumulative_workflow_unsupported.csv", [], empty_fields
    )


class V15Stage4AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.intro = self.root / "intro"
        self.workflow = self.root / "workflow"
        build_intro(self.intro)
        build_workflow(self.workflow)

    def tearDown(self):
        self.temporary.cleanup()

    def run_audit(self):
        return audit.audit_stage4(
            intro_result_dir=self.intro,
            workflow_result_dir=self.workflow,
            expected_native_sha256=NATIVE_SHA,
            expected_runtime_python_sha256=PYTHON_SHA,
        )

    def test_complete_fixture_passes_deterministically(self):
        first = self.run_audit()
        second = self.run_audit()
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "pass")
        self.assertEqual(first["intro"]["raw_sample_files"], 20)
        self.assertEqual(first["workflow"]["rows"], 300)
        self.assertEqual(first["workflow"]["raw_process_files"], 60)
        self.assertEqual(
            len(first["combined_input_manifest_sha256"]), 64
        )

    def test_validation_inside_timer_is_rejected(self):
        with (
            self.workflow / "cumulative_workflow_samples.csv"
        ).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["validation_outside_timer"] = "False"
        write_csv(
            self.workflow / "cumulative_workflow_samples.csv", rows
        )
        with self.assertRaisesRegex(
            audit.AuditFailure, "validation was inside timer"
        ):
            self.run_audit()

    def test_mixed_runtime_is_rejected(self):
        raw_path = next((self.intro / "raw").glob("*_timing_1.json"))
        payload = json.loads(raw_path.read_text())
        payload["runtime_provenance"]["native_sha256"] = "c" * 64
        raw_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            audit.AuditFailure, "native SHA mismatch"
        ):
            self.run_audit()

    def test_cli_writes_the_same_pass_json_that_it_prints(self):
        output = self.root / "audit.json"
        argv = [
            "audit_v15_stage4_outputs_20260730.py",
            "--intro-result-dir",
            str(self.intro),
            "--workflow-result-dir",
            str(self.workflow),
            "--expected-native-sha256",
            NATIVE_SHA,
            "--expected-runtime-python-sha256",
            PYTHON_SHA,
            "--output-json",
            str(output),
        ]
        stdout = io.StringIO()
        with mock.patch.object(sys, "argv", argv), redirect_stdout(stdout):
            return_code = audit.main()
        self.assertEqual(return_code, 0)
        self.assertEqual(stdout.getvalue(), output.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(stdout.getvalue())["status"], "pass")


if __name__ == "__main__":
    unittest.main()
