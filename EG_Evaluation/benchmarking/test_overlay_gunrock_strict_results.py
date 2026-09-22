#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import statistics
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import overlay_gunrock_strict_results as overlay
from gunrock_timing_protocol import PROTOCOL_VERSION


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class GunrockStrictOverlayTest(unittest.TestCase):
    def fixture(
        self,
        root: Path,
        excluded_source: bool = False,
        catastrophic: bool = False,
    ):
        batches = root / "batches"
        included = batches / ("gpu4" if excluded_source else "gpu7")
        excluded = batches / "gpu4"
        included.mkdir(parents=True)
        excluded.mkdir(parents=True, exist_ok=True)
        overlay_path = root / "overlay.csv"
        raw_path = root / "raw.csv"
        base_path = root / "base.csv"
        output_path = root / "output.csv"
        metrics = {
            "build": (
                [0.001, 1.0, 1.1, 1.2, 1.3]
                if catastrophic
                else [1.0, 1.1, 1.2, 1.3, 1.4]
            ),
            "kernel": [0.10, 0.11, 0.12, 0.13, 0.14],
            "e2e": [2.0, 2.1, 2.2, 2.3, 2.4],
        }
        aggregate = []
        raw = []
        for metric, values in metrics.items():
            aggregate.append(
                {
                    "dataset": "graph",
                    "function": "PageRank",
                    "baseline": "Gunrock",
                    "metric": metric,
                    "seconds": str(min(values)),
                    "min_seconds": str(min(values)),
                    "mean_seconds": str(statistics.mean(values)),
                    "sample_sd_seconds": str(statistics.stdev(values)),
                    "sample_seconds_json": json.dumps(values),
                    "status": "ok",
                    "validation": "pass",
                    "estimator_kind": "minimum_of_five_fresh_processes",
                    "aggregation": "minimum_for_paper_display",
                    "sample_count": "5",
                    "n_total": "5",
                    "n_valid": "5",
                    "timing_protocol_version": PROTOCOL_VERSION,
                    "external_cli_wall_used": "false",
                    "source_batch": str(included),
                    "executable_app": "pr",
                    "executable_sha256": "pinned-pr-sha",
                }
            )
            for index, value in enumerate(values, start=1):
                raw.append(
                    {
                        "dataset": "graph",
                        "function": "PageRank",
                        "baseline": "Gunrock",
                        "metric": metric,
                        "sample_index": str(index),
                        "sample_count": "5",
                        "seconds": str(value),
                        "status": "ok",
                        "validation": "pass",
                        "timing_protocol_version": PROTOCOL_VERSION,
                        "external_cli_wall_used": "false",
                        "source_batch": str(included),
                        "executable_app": "pr",
                        "executable_sha256": "pinned-pr-sha",
                    }
                )
        write_csv(overlay_path, aggregate)
        write_csv(raw_path, raw)
        base_rows = [
            {
                "dataset": "graph",
                "function": "PageRank",
                "baseline": "Gunrock",
                "metric": metric,
                "seconds": "99",
            }
            for metric in ("build", "kernel", "e2e")
        ]
        base_rows.extend(
            [
                {
                    "dataset": "graph",
                    "function": "PageRank",
                    "baseline": "EGGPU",
                    "metric": "e2e",
                    "seconds": "0.5",
                },
                {
                    "dataset": "other",
                    "function": "BFS",
                    "baseline": "Gunrock",
                    "metric": "e2e",
                    "seconds": "7",
                },
            ]
        )
        write_csv(base_path, base_rows)
        keys = [overlay.key(row) for row in aggregate]
        gate_path = root / "gate.json"
        gate_path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "packaged_cells": 1,
                    "packaged_metric_samples": 15,
                }
            ),
            encoding="utf-8",
        )
        formal_path = root / "formal.json"
        formal_path.write_text(
            json.dumps({"schema_version": "test"}), encoding="utf-8"
        )
        artifact_path = root / "artifacts.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "executables": {
                        "pr": {
                            "sha256": "pinned-pr-sha",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        manifest = {
            "gate_status": "pass",
            "timing_protocol_version": PROTOCOL_VERSION,
            "external_cli_wall_allowed": False,
            "catastrophic_max_min_ratio_limit": 20.0,
            "catastrophic_absolute_span_seconds": 0.1,
            "overlay_csv": str(overlay_path),
            "overlay_csv_sha256": overlay.sha256_file(overlay_path),
            "raw_samples_csv": str(raw_path),
            "raw_samples_csv_sha256": overlay.sha256_file(raw_path),
            "gate_json": str(gate_path),
            "gate_json_sha256": overlay.sha256_file(gate_path),
            "formal_run_manifest": str(formal_path),
            "formal_run_manifest_sha256": overlay.sha256_file(formal_path),
            "gunrock_artifact_manifest": str(artifact_path),
            "gunrock_artifact_manifest_sha256": overlay.sha256_file(
                artifact_path
            ),
            "replacement_key_count": 3,
            "replacement_key_sha256": overlay.key_digest(keys),
            "success_cell_count": 1,
            "phase_sample_count": 15,
            "included_batches": [str(batches)],
            "excluded_batches": [
                {
                    "path": str(excluded),
                    "status": "excluded_entire_batch",
                    "must_not_merge": True,
                    "reason": "test conflict",
                }
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        args = SimpleNamespace(
            base_ledger=base_path,
            overlay_csv=overlay_path,
            overlay_manifest=manifest_path,
            output=output_path,
            receipt=None,
            ledger_format="auto",
        )
        return args

    def test_atomic_overlay_replaces_only_matching_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            self.assertEqual(overlay.driver(args), 0)
            _, rows = overlay.read_csv(args.output)
            self.assertEqual(len(rows), 5)
            selected = {
                overlay.key(row): row
                for row in rows
                if row["dataset"] == "graph"
                and row["baseline"] == "Gunrock"
            }
            self.assertEqual(len(selected), 3)
            self.assertEqual(float(selected[("graph", "PageRank", "Gunrock", "build")]["seconds"]), 1.0)
            self.assertTrue(
                any(
                    row["dataset"] == "graph" and row["baseline"] == "EGGPU"
                    for row in rows
                )
            )
            receipt = args.output.with_name(
                args.output.name + ".gunrock_overlay_receipt.json"
            )
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["removed_matching_rows"], 3)
            self.assertEqual(payload["inserted_overlay_rows"], 3)
            self.assertFalse(payload["in_place_mutation"])

    def test_in_place_base_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args.output = args.base_ledger
            with self.assertRaises(overlay.OverlayValidationError):
                overlay.driver(args)

    def test_final13_overlay_replaces_one_cell_and_preserves_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            write_csv(
                args.base_ledger,
                [
                    {
                        "dataset": "graph",
                        "function": "PageRank",
                        "baseline": "Gunrock",
                        "execution_status": "ok",
                        "validation_status": "pass",
                        "sample_count": "5",
                        "build_paper_seconds": "",
                        "kernel_paper_seconds": "99",
                        "e2e_paper_seconds": "100",
                        "gpu_peak_mb_mean": "123",
                    },
                    {
                        "dataset": "graph",
                        "function": "PageRank",
                        "baseline": "EGGPU",
                        "execution_status": "ok",
                        "validation_status": "pass",
                        "sample_count": "5",
                        "build_paper_seconds": "0.5",
                        "kernel_paper_seconds": "0.1",
                        "e2e_paper_seconds": "0.8",
                        "gpu_peak_mb_mean": "456",
                    },
                ],
            )
            args.ledger_format = "final13"
            self.assertEqual(overlay.driver(args), 0)
            fields, rows = overlay.read_csv(args.output)
            self.assertIn("timing_overlay_manifest", fields)
            gunrock = next(row for row in rows if row["baseline"] == "Gunrock")
            eggpu = next(row for row in rows if row["baseline"] == "EGGPU")
            self.assertEqual(float(gunrock["build_paper_seconds"]), 1.0)
            self.assertEqual(float(gunrock["kernel_paper_seconds"]), 0.1)
            self.assertEqual(float(gunrock["e2e_paper_seconds"]), 2.0)
            self.assertEqual(gunrock["build_estimator"], "minimum_of_five_fresh_processes")
            self.assertEqual(gunrock["gpu_peak_mb_mean"], "123")
            self.assertEqual(eggpu["e2e_paper_seconds"], "0.8")
            receipt = json.loads(
                args.output.with_name(
                    args.output.name + ".gunrock_overlay_receipt.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["ledger_format"], "final13")
            self.assertEqual(receipt["replaced_success_cells"], 1)
            self.assertEqual(receipt["output_rows"], 2)

    def test_tampered_overlay_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            with args.overlay_csv.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaises(overlay.OverlayValidationError):
                overlay.driver(args)
            self.assertFalse(args.output.exists())

    def test_explicitly_excluded_source_batch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), excluded_source=True)
            with self.assertRaisesRegex(
                overlay.OverlayValidationError, "explicitly excluded"
            ):
                overlay.driver(args)
            self.assertFalse(args.output.exists())

    def test_catastrophic_timing_spread_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory), catastrophic=True)
            with self.assertRaisesRegex(
                overlay.OverlayValidationError, "catastrophic"
            ):
                overlay.driver(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
