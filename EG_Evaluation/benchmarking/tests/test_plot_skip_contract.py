import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


BENCHMARKING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARKING))
SPEC = importlib.util.spec_from_file_location(
    "eggpu_test_plot_skip_contract",
    BENCHMARKING / "run_full_baselines.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PlotSkipContractTests(unittest.TestCase):
    def test_explicit_plot_skip_is_success(self):
        with mock.patch.dict(os.environ, {"EGGPU_SKIP_PLOTS": "TRUE"}):
            result = MODULE.write_plot_and_tables_isolated(
                Path("unused"), [], [], "0"
            )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
