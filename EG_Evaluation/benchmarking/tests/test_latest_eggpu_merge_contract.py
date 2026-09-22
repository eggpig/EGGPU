import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "merge_latest_eggpu_main_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("merge_latest_eggpu", MODULE_PATH)
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


class LatestEggpuMergeContractTests(unittest.TestCase):
    def test_old_eggpu_rows_are_never_retained(self):
        baseline = [
            {"dataset": "g", "baseline": "EGGPU", "value": "old"},
            {"dataset": "g", "baseline": "igraph", "value": "base"},
            {"dataset": "other", "baseline": "igraph", "value": "drop"},
        ]
        latest = [{"dataset": "g", "baseline": "EGGPU", "value": "new"}]
        rows = MERGE.merge_rows(baseline, latest, {"g"})
        self.assertEqual(
            rows,
            [
                {"dataset": "g", "baseline": "igraph", "value": "base"},
                {"dataset": "g", "baseline": "EGGPU", "value": "new"},
            ],
        )

    def test_latest_contract_requires_split_measurement(self):
        rows = [
            {
                "dataset": "g",
                "baseline": "EGGPU",
                "measurement_phase": "timing",
            }
        ]
        with self.assertRaisesRegex(ValueError, "timing and memory"):
            MERGE.assert_latest_contract(rows, ["g"])

    def test_version_merge_keeps_baseline_and_binds_latest_eggpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = root / "baseline.json"
            latest = root / "latest.json"
            baseline.write_text(
                json.dumps(
                    {
                        "paper_repo_source_snapshot": {"digest": "old"},
                        "python_baselines": {
                            "cpp_easygraph": {"module_origin": "baseline.so"},
                            "igraph": {"version": "1"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            latest.write_text(
                json.dumps(
                    {
                        "paper_repo_source_snapshot": {"digest": "new"},
                        "python_baselines": {
                            "cpp_easygraph": {
                                "module_origin": "eggpu.so",
                                "sha256": "current",
                            },
                            "easygraph": {"version": "current"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            merged = MERGE.merge_version_manifests(baseline, latest)
        self.assertEqual(
            merged["python_baselines"]["cpp_easygraph"]["module_origin"],
            "baseline.so",
        )
        self.assertEqual(
            merged["python_baselines"]["eggpu_cpp_easygraph"]["module_origin"],
            "eggpu.so",
        )
        self.assertEqual(merged["latest_eggpu_source_snapshot"]["digest"], "new")
        self.assertEqual(
            merged["retained_baseline_source_snapshot"]["digest"], "old"
        )


if __name__ == "__main__":
    unittest.main()
