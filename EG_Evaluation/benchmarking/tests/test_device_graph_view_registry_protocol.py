import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve()
MODULE_PATH = HERE.parents[1] / "benchmark_device_graph_view_registry.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("device_registry_benchmark", MODULE_PATH)
registry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registry)


class DeviceGraphViewRegistryProtocolTests(unittest.TestCase):
    def test_easygraph_repo_is_required(self):
        argv = [
            str(MODULE_PATH),
            "--dataset",
            "/tmp/graph.txt",
            "--max-entries",
            "2",
            "--output",
            "/tmp/result.json",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                registry.parse_args()

    def test_parse_args_retains_explicit_frozen_runtime(self):
        argv = [
            str(MODULE_PATH),
            "--easygraph-repo",
            "/tmp/frozen-runtime",
            "--dataset",
            "/tmp/graph.txt",
            "--max-entries",
            "2",
            "--output",
            "/tmp/result.json",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = registry.parse_args()
        self.assertEqual(args.easygraph_repo, Path("/tmp/frozen-runtime"))


if __name__ == "__main__":
    unittest.main()
