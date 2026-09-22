import ast
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "library_baselines.py"


class TestNxCuGraphIsolation(unittest.TestCase):
    def test_runner_does_not_import_native_cugraph_or_cudf(self):
        tree = ast.parse(SOURCE_PATH.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("cugraph", imported)
        self.assertNotIn("cudf", imported)

    def test_native_cugraph_helpers_are_absent(self):
        tree = ast.parse(SOURCE_PATH.read_text())
        function_names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertFalse(
            {name for name in function_names if name.startswith("cugraph_")},
            function_names,
        )

    def test_nx_cugraph_calls_force_networkx_backend_dispatch(self):
        source = SOURCE_PATH.read_text()
        self.assertIn('kwargs["backend"] = "cugraph"', source)
        self.assertNotIn("native cuGraph fallback", source)
        self.assertNotIn("native cuGraph path", source)


if __name__ == "__main__":
    unittest.main()
