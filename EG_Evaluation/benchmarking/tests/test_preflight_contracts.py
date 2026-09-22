import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "preflight_full_eval_ready.py"
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("eggpu_test_preflight", MODULE_PATH)
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class PreflightContractTests(unittest.TestCase):
    def test_required_gunrock_executable_set_is_explicit(self):
        self.assertEqual(
            PREFLIGHT.REQUIRED_GUNROCK_EXECUTABLES,
            ("pr", "mst", "lcc", "bfs", "sssp", "kcore", "bc"),
        )

    def test_gunrock_preflight_fails_when_required_binary_is_missing(self):
        fake = {name: Path(f"/tmp/{name}") for name in PREFLIGHT.REQUIRED_GUNROCK_EXECUTABLES}
        fake["lcc"] = None
        with patch.object(PREFLIGHT, "_discover_gunrock_executables", return_value=fake), patch.object(
            PREFLIGHT, "emit"
        ):
            self.assertFalse(PREFLIGHT.check_gunrock_executables())

    def test_gunrock_runtime_linkage_is_checked(self):
        fake = {
            name: Path("/bin/true")
            for name in PREFLIGHT.REQUIRED_GUNROCK_EXECUTABLES
        }
        with patch.object(PREFLIGHT, "_discover_gunrock_executables", return_value=fake), patch.object(
            PREFLIGHT, "emit"
        ):
            self.assertTrue(PREFLIGHT.check_gunrock_runtime_linkage())

    def test_repeat_statistics_static_contract_is_enforced(self):
        with patch.object(PREFLIGHT, "emit"):
            self.assertTrue(PREFLIGHT.check_repeat_statistics_static_contract())

    def test_gpu_runtime_uses_only_the_canonical_enable_switch(self):
        with patch.object(PREFLIGHT, "emit"):
            self.assertTrue(PREFLIGHT.check_gpu_runtime_enable_contract())

    def test_gunrock_version_provenance_is_complete(self):
        with patch.object(PREFLIGHT, "emit"):
            self.assertTrue(PREFLIGHT.check_gunrock_provenance_static_contract())


if __name__ == "__main__":
    unittest.main()
