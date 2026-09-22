import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "library_baselines.py"
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("eggpu_test_library_baselines", MODULE_PATH)
BASELINES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINES)


def views(n=16):
    directed = pd.DataFrame(
        {"src": list(range(n - 1)), "dst": list(range(1, n))}
    ).astype("int32")
    undirected = directed.copy()
    return {
        "clean": (n, directed, undirected),
        "all_vertices": (n, directed, undirected),
    }


class TestPathBenchmarkSemantics(unittest.TestCase):
    def test_dijkstra_is_single_source(self):
        plan = BASELINES.path_benchmark_plan(
            "Dijkstra", "directed", views(), source_count=8
        )
        self.assertEqual(len(plan["sources"]), 1)
        self.assertIn("single-source", plan["note"])

    def test_sssp_is_batched_multi_source(self):
        plan = BASELINES.path_benchmark_plan(
            "SSSP", "directed", views(), source_count=8
        )
        self.assertEqual(len(plan["sources"]), 8)
        self.assertIn("batched SSSP", plan["note"])


if __name__ == "__main__":
    unittest.main()
