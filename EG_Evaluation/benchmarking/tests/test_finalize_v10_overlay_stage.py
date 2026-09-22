import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "finalize_v10_overlay_stage.py"
)
SPEC = importlib.util.spec_from_file_location(
    "finalize_v10_overlay_stage",
    MODULE_PATH,
)
FINALIZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FINALIZE
SPEC.loader.exec_module(FINALIZE)


class FinalizeV10OverlayStageTests(unittest.TestCase):
    def test_relocates_paths_and_rebinds_hashes_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            stage = parent / ".stage"
            final = parent / "final"
            stage.mkdir()
            stability_csv = stage / "eggpu_timing_stability.csv"
            stability_csv.write_text("dataset,function\n", encoding="utf-8")
            stability_json = stage / "eggpu_timing_stability.json"
            stability_json.write_text(
                json.dumps(
                    {
                        "rows_csv_path": str(stability_csv),
                        "rows_csv_sha256": FINALIZE.sha256(stability_csv),
                    }
                ),
                encoding="utf-8",
            )
            ledger = stage / "final_13_cell_outcome_ledger.csv"
            ledger.write_text("ledger\n", encoding="utf-8")
            overlay_json = (
                stage / "EGGPU_V10_OVERLAY_AUDIT_20260729.json"
            )
            timing_cells = overlay_json.with_suffix(".cells.csv")
            memory_cells = overlay_json.with_suffix(".memory_cells.csv")
            timing_cells.write_text("timing\n", encoding="utf-8")
            memory_cells.write_text("memory\n", encoding="utf-8")
            overlay_json.write_text(
                json.dumps(
                    {
                        "output_ledger": str(ledger),
                        "output_ledger_sha256": FINALIZE.sha256(ledger),
                        "comparison_csv": str(timing_cells),
                        "comparison_csv_sha256": FINALIZE.sha256(
                            timing_cells
                        ),
                        "memory_comparison_csv": str(memory_cells),
                        "memory_comparison_csv_sha256": FINALIZE.sha256(
                            memory_cells
                        ),
                        "timing_stability_audit": {
                            "audit_path": str(stability_json),
                            "audit_sha256": FINALIZE.sha256(
                                stability_json
                            ),
                            "rows_csv_path": str(stability_csv),
                            "rows_csv_sha256": FINALIZE.sha256(
                                stability_csv
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                str(MODULE_PATH),
                "--stage-dir",
                str(stage),
                "--final-dir",
                str(final),
            ]
            with mock.patch.object(sys, "argv", argv):
                FINALIZE.main()

            stability = json.loads(
                stability_json.read_text(encoding="utf-8")
            )
            overlay = json.loads(overlay_json.read_text(encoding="utf-8"))
            self.assertEqual(
                stability["rows_csv_path"],
                str(final / stability_csv.name),
            )
            self.assertEqual(
                overlay["output_ledger"],
                str(final / ledger.name),
            )
            self.assertEqual(
                overlay["timing_stability_audit"]["audit_sha256"],
                FINALIZE.sha256(stability_json),
            )
            self.assertEqual(
                overlay["timing_stability_audit"]["rows_csv_path"],
                str(final / stability_csv.name),
            )

    def test_rejects_cross_filesystem_style_stage_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            stage = parent / "staging" / "run"
            stage.mkdir(parents=True)
            final = parent / "release"
            argv = [
                str(MODULE_PATH),
                "--stage-dir",
                str(stage),
                "--final-dir",
                str(final),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "share a parent"):
                    FINALIZE.main()


if __name__ == "__main__":
    unittest.main()
