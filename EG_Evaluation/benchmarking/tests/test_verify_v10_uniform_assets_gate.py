import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = BENCHMARK_DIR / "verify_v10_uniform_assets_gate.py"
SPEC = importlib.util.spec_from_file_location("verify_v10_uniform", MODULE_PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY
SPEC.loader.exec_module(VERIFY)


class VerifyV10UniformAssetsGateTests(unittest.TestCase):
    def test_published_root_path_is_allowed_after_atomic_rename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "manifest.json").write_text(
                f'{{"asset_root": "{root}"}}\n',
                encoding="utf-8",
            )
            VERIFY.assert_no_staging_paths(root, root)

    def test_stage_root_path_is_rejected_before_atomic_rename(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            stage = parent / ".EGGPU_V10_UNIFORM.stage.ABC123"
            final = parent / "V10_UNIFORM"
            stage.mkdir()
            (stage / "manifest.json").write_text(
                f'{{"asset_root": "{stage}"}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "staging-local path"):
                VERIFY.assert_no_staging_paths(stage, final)

    def test_stage_token_is_always_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "manifest.json").write_text(
                '{"old": "/tmp/.EGGPU_V10_UNIFORM.stage.DEAD/file"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "staging-local path"):
                VERIFY.assert_no_staging_paths(root, root)


if __name__ == "__main__":
    unittest.main()
