import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "baseline_versions.py"
SPEC = importlib.util.spec_from_file_location("eggpu_test_baseline_versions", MODULE_PATH)
VERSIONS = importlib.util.module_from_spec(SPEC)


class BaselineVersionProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SPEC.loader.exec_module(VERSIONS)

    def test_python_baselines_are_named_separately(self):
        versions = VERSIONS.collect_python_baseline_versions()
        self.assertIn("networkx", versions)
        self.assertIn("igraph", versions)
        self.assertIn("nx-cugraph", versions)
        self.assertIn("easygraph", versions)
        self.assertIn("cpp_easygraph", versions)
        self.assertNotIn("cugraph", versions["evaluated_baselines"])
        for name in ("easygraph", "networkx", "igraph", "nx-cugraph"):
            self.assertIn("module_origin", versions[name])
        self.assertTrue(versions["cpp_easygraph"]["module_origin"].endswith(".so"))

    def test_cugraph_is_recorded_only_as_nx_cugraph_dependency(self):
        versions = VERSIONS.collect_python_baseline_versions()
        cugraph = versions["runtime_dependencies"]["cugraph"]
        self.assertEqual(cugraph["role"], "nx-cugraph runtime dependency; not an evaluated baseline")


if __name__ == "__main__":
    unittest.main()
