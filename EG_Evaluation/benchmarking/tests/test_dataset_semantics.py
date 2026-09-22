import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "library_baselines.py"
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("eggpu_test_dataset_semantics", MODULE_PATH)
BASELINES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINES)


class TestDatasetSemantics(unittest.TestCase):
    def test_simple_graph_removes_loops_and_duplicates_but_retains_vertices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.txt"
            path.write_text("1 1\n1 2\n2 1\n1 2\n3 3\n")
            views = BASELINES.load_graph(path)

        n, directed, undirected = views["clean"]
        self.assertEqual(n, 3)
        self.assertEqual(len(directed), 2)
        self.assertEqual(len(undirected), 1)
        self.assertEqual(views["clean"][0], views["all_vertices"][0])
        self.assertEqual(len(views["clean"][1]), len(views["all_vertices"][1]))


if __name__ == "__main__":
    unittest.main()
