import json
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

import apply_nxcugraph_strict_main99_overlay as overlay


def strict_row(dataset, function):
    samples = {
        "construction": [1.0, 1.1, 1.2, 1.3, 1.4],
        "e2e": [0.10, 0.11, 0.12, 0.13, 0.14],
        "processing": [0.01, 0.011, 0.012, 0.013, 0.014],
    }
    provenance = {
        "validation_outside_timer": True,
        "public_return_boundary": True,
        "prepared_native_graph": True,
        "networkx_cache_converted_graphs": True,
        "networkx_fallback_to_nx": False,
        "gpu_exclusive_pre_and_post_snapshots": True,
        "warmup_calls": 3,
        "fresh_process_samples": 5,
        "samples": [
            {
                "sample_index": index,
                "gpu_exclusive_snapshot_before_worker": (
                    "gpu_idle_before_eggpu_child: gpu=0, memory_mb=4.0"
                ),
                "gpu_exclusive_snapshot_after_worker": (
                    "gpu_idle_before_eggpu_child: gpu=0, memory_mb=4.0"
                ),
            }
            for index in range(1, 6)
        ],
    }
    row = {
        "dataset": dataset,
        "function": function,
        "baseline": "nx-cugraph",
        "display_estimator": "minimum_of_five",
        "validation_status": "pass",
        "timer_kind": "cuda_event_at_pylibcugraph_backend",
        "measurement_window": "device_execution",
        "timing_provenance": json.dumps(provenance),
    }
    for metric, values in samples.items():
        row[f"{metric}_samples"] = json.dumps(values)
        row[f"{metric}_min_seconds"] = min(values)
        row[f"{metric}_mean_seconds"] = statistics.mean(values)
        row[f"{metric}_sample_std_seconds"] = statistics.stdev(values)
    return row


def strict_rows():
    return [
        strict_row(dataset, function)
        for dataset, function in sorted(overlay.EXPECTED_KEYS)
    ]


def ledger_rows():
    rows = []
    for dataset, function in sorted(overlay.EXPECTED_KEYS):
        rows.append(
            {
                "dataset": dataset,
                "function": function,
                "baseline": "nx-cugraph",
                "execution_status": "ok",
                "memory_result_source": "preserve-me",
                "gpu_peak_mb_mean": "321",
            }
        )
    rows.append(
        {
            "dataset": "ca-HepTh",
            "function": "PageRank",
            "baseline": "EGGPU",
            "execution_status": "ok",
            "e2e_paper_seconds": "unchanged",
        }
    )
    return rows


class StrictOverlayTest(unittest.TestCase):
    def test_exact_99_overlay_preserves_memory_and_other_baselines(self):
        source = ledger_rows()
        output, report = overlay.apply_overlay(
            source, strict_rows(), Path("/tmp/strict.csv")
        )
        self.assertEqual(report["replaced_cell_count"], 99)
        updated = next(
            row
            for row in output
            if row["baseline"] == "nx-cugraph"
            and row["dataset"] == "ca-HepTh"
            and row["function"] == "PageRank"
        )
        self.assertEqual(updated["e2e_estimator"], "minimum_of_five")
        self.assertEqual(float(updated["e2e_paper_seconds"]), 0.10)
        self.assertAlmostEqual(
            float(updated["e2e_std_seconds"]),
            statistics.stdev([0.10, 0.11, 0.12, 0.13, 0.14]),
        )
        self.assertEqual(updated["memory_result_source"], "preserve-me")
        self.assertEqual(updated["gpu_peak_mb_mean"], "321")
        other = next(row for row in output if row["baseline"] == "EGGPU")
        self.assertEqual(other["e2e_paper_seconds"], "unchanged")
        self.assertNotIn("e2e_paper_seconds", source[0])

    def test_wall_time_cannot_be_copied_to_processing(self):
        rows = strict_rows()
        first = rows[0]
        first["processing_samples"] = first["e2e_samples"]
        first["processing_min_seconds"] = first["e2e_min_seconds"]
        first["processing_mean_seconds"] = first["e2e_mean_seconds"]
        first["processing_sample_std_seconds"] = first[
            "e2e_sample_std_seconds"
        ]
        with self.assertRaisesRegex(ValueError, "wall time copied"):
            overlay.apply_overlay(
                ledger_rows(), rows, Path("/tmp/strict.csv")
            )

    def test_atomic_csv_writer_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlay.csv"
            overlay.atomic_write_csv(
                path,
                [{"dataset": "tiny", "function": "PageRank"}],
                ["dataset", "function"],
            )
            self.assertTrue(path.is_file())
            self.assertEqual(
                [candidate.name for candidate in path.parent.iterdir()],
                ["overlay.csv"],
            )

    def test_cli_refuses_to_overwrite_any_input_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                ledger=root / "v10.csv",
                strict_csv=root / "strict.csv",
                audit_manifest=root / "audit.json",
                out_ledger=root / "strict.csv",
                out_manifest=root / "manifest.json",
            )
            with mock.patch.object(overlay, "parse_args", return_value=args):
                with self.assertRaisesRegex(SystemExit, "source artifact"):
                    overlay.main()


if __name__ == "__main__":
    unittest.main()
