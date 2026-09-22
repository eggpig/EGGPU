import importlib.util
import hashlib
import os
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_full_baselines.py"
if str(MODULE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("eggpu_test_full_runner", MODULE_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class EnvGuard:
    def __init__(self, *names):
        self.names = names
        self.old = {}

    def __enter__(self):
        self.old = {name: os.environ.get(name) for name in self.names}
        for name in self.names:
            os.environ.pop(name, None)
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, value in self.old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestGunrockDiscovery(unittest.TestCase):
    def test_colon_separated_paths_find_per_algorithm_executable(self):
        with tempfile.TemporaryDirectory() as tmp, EnvGuard(
            "EG_GUNROCK_BIN_PATHS", "EG_GUNROCK_BIN", "GUNROCK_BIN"
        ):
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            exe = second / "sssp"
            exe.write_text("#!/bin/sh\nexit 0\n")
            exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
            os.environ["EG_GUNROCK_BIN_PATHS"] = os.pathsep.join((str(first), str(second)))

            candidates = RUNNER.gunrock_bin_candidates()
            self.assertEqual(candidates[:2], [first, second])
            self.assertEqual(RUNNER.find_gunrock_exe("sssp"), exe)

    def test_missing_executable_is_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp, EnvGuard(
            "EG_GUNROCK_BIN_PATHS", "EG_GUNROCK_BIN", "GUNROCK_BIN"
        ):
            os.environ["EG_GUNROCK_BIN_PATHS"] = tmp
            self.assertIsNone(RUNNER.find_gunrock_exe("does-not-exist"))

    def test_executable_artifacts_record_path_and_digest(self):
        with tempfile.TemporaryDirectory() as tmp, EnvGuard(
            "EG_GUNROCK_BIN_PATHS", "EG_GUNROCK_BIN", "GUNROCK_BIN"
        ):
            root = Path(tmp)
            exe = root / "pr"
            payload = b"#!/bin/sh\nexit 0\n"
            exe.write_bytes(payload)
            exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
            os.environ["EG_GUNROCK_BIN_PATHS"] = str(root)

            artifacts = RUNNER.gunrock_executable_artifacts(("pr", "cc"))

            self.assertEqual(artifacts["pr"]["path"], str(exe.resolve()))
            self.assertEqual(
                artifacts["pr"]["sha256"], hashlib.sha256(payload).hexdigest()
            )
            self.assertEqual(artifacts["pr"]["upstream_release_lineage"], "v2.2.0")
            self.assertEqual(artifacts["pr"]["application_generation"], "maintained-v2")
            self.assertEqual(artifacts["pr"]["build_cuda_toolkit_version"], "13.2")
            self.assertEqual(artifacts["pr"]["build_cuda_architecture"], "80")
            self.assertEqual(artifacts["cc"], {"status": "missing"})

    def test_every_required_executable_has_an_explicit_release_lineage(self):
        self.assertEqual(
            set(RUNNER.GUNROCK_APPLICATION_PROVENANCE),
            {"pr", "mst", "lcc", "bfs", "sssp", "kcore", "bc"},
        )
        for entry in RUNNER.GUNROCK_APPLICATION_PROVENANCE.values():
            self.assertTrue(entry["upstream_release_lineage"].startswith("v"))
            self.assertIn(entry["application_generation"], {"maintained-v2", "legacy-v1"})
            self.assertTrue(entry["build_cuda_toolkit_version"])
            self.assertEqual(entry["build_cuda_architecture"], "80")

    def test_pinned_manifest_provenance_works_without_git(self):
        with tempfile.TemporaryDirectory() as tmp, EnvGuard(
            "EG_GUNROCK_BIN_PATHS", "EG_GUNROCK_BIN", "GUNROCK_BIN"
        ):
            root = Path(tmp)
            source_root = root / "gunrock_latest"
            binary_root = source_root / "bin"
            binary_root.mkdir(parents=True)
            executable = binary_root / "pr"
            payload = b"#!/bin/sh\nexit 0\n"
            executable.write_bytes(payload)
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            digest = hashlib.sha256(payload).hexdigest()
            manifest_path = root / "gunrock_artifact_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sources": {
                            "maintained-v2": {
                                "source_directory_name": "gunrock_latest",
                                "source_remote": "https://example.invalid/gunrock.git",
                                "source_commit": "a" * 40,
                                "source_version": "v2.2.0-test",
                                "source_tracked_dirty": True,
                                "source_diff_sha256": "b" * 64,
                                "source_modified_files": ["source.hxx"],
                                "source_modified_files_latest_mtime_epoch": 1.0,
                            }
                        },
                        "executables": {
                            "pr": {
                                "source": "maintained-v2",
                                "sha256": digest,
                                "source_relevant_modified_files": ["pr.cu"],
                                "source_relevant_files_latest_mtime_epoch": 2.0,
                            }
                        },
                    }
                )
            )
            os.environ["EG_GUNROCK_BIN_PATHS"] = str(binary_root)
            with mock.patch.object(RUNNER, "GUNROCK_ARTIFACT_MANIFEST", manifest_path):
                artifact = RUNNER.gunrock_executable_artifacts(("pr",))["pr"]

            self.assertEqual(artifact["source_provenance_origin"], "pinned-binary-manifest")
            self.assertTrue(artifact["source_manifest_binary_sha256_match"])
            self.assertEqual(artifact["source_commit"], "a" * 40)
            self.assertEqual(artifact["source_modified_files"], ["pr.cu"])
            self.assertEqual(
                artifact["source_modified_files_latest_mtime_epoch"], 2.0
            )
            self.assertEqual(artifact["source_provenance_errors"], [])
            self.assertFalse(artifact["runtime_git_validation"])

    def test_runtime_provenance_never_requires_git(self):
        source = MODULE_PATH.read_text()
        self.assertNotIn("def _git_source_provenance", source)
        self.assertNotIn('["git",', source)

    def test_pinned_manifest_rejects_a_different_binary(self):
        provenance, error = RUNNER._gunrock_manifest_provenance(
            "pr", "0" * 64, Path("/tmp/gunrock_latest/bin/pr")
        )
        self.assertEqual(provenance, {})
        self.assertIn("binary SHA-256 mismatch", error)

    def test_current_runner_does_not_claim_legacy_cc_as_wcc_or_scc(self):
        source = MODULE_PATH.read_text()
        main_body = source[source.index("def main():") :]
        self.assertNotIn("maybe_run_gunrock_cc(", main_body)
        self.assertIn('add_unavailable_gunrock(\n                        rows, size, graph_type, name, component_func', main_body)


if __name__ == "__main__":
    unittest.main()
