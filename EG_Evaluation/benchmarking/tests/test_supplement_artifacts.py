import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


BENCHMARKING = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "eggpu_test_supplement_artifacts",
    BENCHMARKING / "generate_first_use_workflow_artifacts.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SupplementArtifactTests(unittest.TestCase):
    def test_generates_first_use_and_workflow_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            natural = root / "natural"
            out = root / "out"
            first.mkdir()
            natural.mkdir()
            functions = ["PageRank", "LCC", "BFS", "EffectiveSize"]
            rows = []
            for metric in ("e2e", "kernel"):
                for function in functions:
                    rows.append(
                        {
                            "dataset": "toy",
                            "function": function,
                            "metric": metric,
                            "first_use_over_steady": 2.0,
                        }
                    )
            pd.DataFrame(rows).to_csv(first / "first_use_vs_steady.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "toy",
                        "metric": metric,
                        "isolated_over_natural": 1.5,
                    }
                    for metric in ("e2e", "kernel")
                ]
            ).to_csv(natural / "natural_workflow_vs_isolated.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset": "toy",
                        "function": function,
                        "metric": metric,
                        "call_position": 2 if function == "PageRank" else 3,
                        "first_use_mean_seconds": 1.4,
                        "natural_mean_seconds": 1.0,
                        "time_saved_seconds": 0.4,
                        "first_use_over_natural": 1.4,
                        "natural_is_faster": True,
                    }
                    for function in ("PageRank", "BFS")
                    for metric in ("e2e", "kernel")
                ]
            ).to_csv(
                natural / "natural_workflow_reuse_beneficiaries.csv",
                index=False,
            )

            MODULE.setup_style()
            MODULE.first_use_summary(first, out)
            MODULE.workflow_summary(natural, out)

            self.assertTrue((out / "first_use_vs_steady_by_family.pdf").exists())
            self.assertTrue((out / "natural_workflow_reuse.pdf").exists())


if __name__ == "__main__":
    unittest.main()
