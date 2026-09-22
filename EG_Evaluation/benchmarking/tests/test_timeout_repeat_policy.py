import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_full_baselines.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "run_full_baselines_timeout_policy", MODULE_PATH
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TimeoutRepeatPolicyTests(unittest.TestCase):
    def test_first_timeout_remains_terminal(self):
        self.assertFalse(RUNNER.should_continue_after_isolated_timeout([], 100))

    def test_one_fast_sample_is_not_enough_to_override_timeout(self):
        self.assertFalse(RUNNER.should_continue_after_isolated_timeout([0.01], 100))

    def test_isolated_timeout_after_two_fast_samples_keeps_repeats_running(self):
        self.assertTrue(
            RUNNER.should_continue_after_isolated_timeout([0.0098, 0.0104], 100)
        )

    def test_calls_near_the_protocol_limit_remain_terminal(self):
        self.assertFalse(
            RUNNER.should_continue_after_isolated_timeout([18.0, 19.0], 100)
        )


if __name__ == "__main__":
    unittest.main()
