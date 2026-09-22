from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "audit_complete_timing_triplets.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("timing_triplets", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(baseline: str, build: float, processing: float, e2e: float) -> dict:
    return {
        "dataset": "g",
        "function": "PageRank",
        "baseline": baseline,
        "execution_status": "ok",
        "validation_status": "pass",
        "sample_count": "5",
        "build_paper_seconds": str(build),
        "kernel_paper_seconds": str(processing),
        "e2e_paper_seconds": str(e2e),
        "build_estimator": "minimum_of_five",
        "kernel_estimator": "backend_device_interval_minimum_of_five",
        "e2e_estimator": "minimum_of_five",
    }


class CompleteTimingTripletTests(unittest.TestCase):
    def test_valid_system_specific_boundaries_pass(self) -> None:
        rows = [
            row("networkx", 1.0, 2.0, 2.0),
            row("GraphScope", 1.0, 1.5, 2.0),
            row("EGGPU", 1.0, 0.4, 0.8),
            row("nx-cugraph", 1.0, 0.5, 1.0),
            row("Gunrock", 1.0, 0.5, 2.0),
        ]
        issues, summary = MODULE.audit_rows(rows)
        self.assertEqual(issues, [])
        self.assertEqual(summary["status"], "pass")

    def test_missing_and_surrogate_nxcugraph_timings_fail(self) -> None:
        bad = row("nx-cugraph", 1.0, 2.0, 2.0)
        bad["build_paper_seconds"] = ""
        bad["kernel_estimator"] = "public_call_wall_surrogate"
        issues, summary = MODULE.audit_rows([bad])
        kinds = {issue["issue"] for issue in issues}
        self.assertIn("missing_or_nonpositive_timing", kinds)
        self.assertIn("nxcugraph_processing_is_wall_surrogate", kinds)
        self.assertIn("nxcugraph_processing_duplicates_e2e", kinds)
        self.assertEqual(summary["status"], "fail")

    def test_cpu_processing_and_gunrock_nesting_are_enforced(self) -> None:
        cpu = row("igraph", 1.0, 1.0, 2.0)
        gunrock = row("Gunrock", 3.0, 1.0, 2.0)
        issues, _ = MODULE.audit_rows([cpu, gunrock])
        kinds = {issue["issue"] for issue in issues}
        self.assertIn("cpu_processing_must_equal_public_call", kinds)
        self.assertIn("gunrock_construction_exceeds_standalone_e2e", kinds)


if __name__ == "__main__":
    unittest.main()
