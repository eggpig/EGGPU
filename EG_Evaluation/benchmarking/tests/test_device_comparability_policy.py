import importlib.util
import sys
import unittest
from pathlib import Path


BENCHMARKING = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    BENCHMARKING / "generate_final_13_complete_assets_graphscope.py"
)
sys.path.insert(0, str(BENCHMARKING))
SPEC = importlib.util.spec_from_file_location(
    "generate_complete_assets_graphscope", MODULE_PATH
)
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


class DeviceComparabilityPolicyTest(unittest.TestCase):
    def setUp(self):
        GENERATOR.configure_baselines(None)

    def test_public_return_uses_libraries_not_standalone_cli(self):
        candidates = GENERATOR.comparable_baselines("e2e")
        self.assertIn("networkx", candidates)
        self.assertIn("igraph", candidates)
        self.assertIn("nx-cugraph", candidates)
        self.assertNotIn("Gunrock", candidates)

    def test_device_comparison_excludes_cpu_wall_surrogates(self):
        candidates = GENERATOR.comparable_baselines("kernel")
        self.assertEqual(candidates, ("nx-cugraph", "Gunrock"))
        for baseline in GENERATOR.CPU_BASELINES:
            self.assertNotIn(baseline, candidates)
        self.assertNotIn("GraphScope", candidates)

    def test_gpu_comparison_uses_same_native_device_set(self):
        self.assertEqual(
            GENERATOR.comparable_gpu_baselines("kernel"),
            ("nx-cugraph", "Gunrock"),
        )


if __name__ == "__main__":
    unittest.main()
