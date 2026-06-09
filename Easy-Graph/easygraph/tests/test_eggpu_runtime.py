import importlib.util
import os
from pathlib import Path
import unittest


def _load_gpu_runtime_module():
    path = Path(__file__).resolve().parents[1] / "utils" / "gpu_runtime.py"
    spec = importlib.util.spec_from_file_location("eggpu_test_gpu_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gpu_runtime = _load_gpu_runtime_module()


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


class TestEGGPURuntime(unittest.TestCase):
    ENV_KEYS = (
        "EASYGRAPH_ENABLE_GPU",
        "EASYGRAPH_GPU_BACKEND",
        "EASYGRAPH_GPU_STRICT_ERRORS",
    )

    def test_gpu_runtime_disabled_by_default(self):
        with EnvGuard(*self.ENV_KEYS):
            self.assertFalse(gpu_runtime.gpu_runtime_enabled())
            self.assertEqual(gpu_runtime.gpu_backend_name(), "mine")

    def test_strict_unknown_backend_fails_fast(self):
        with EnvGuard(*self.ENV_KEYS):
            os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
            os.environ["EASYGRAPH_GPU_BACKEND"] = "typo"
            os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
            with self.assertRaisesRegex(RuntimeError, "Unsupported EASYGRAPH_GPU_BACKEND"):
                gpu_runtime.gpu_backend_name()

    def test_non_strict_unknown_backend_remains_cpu_compatible(self):
        with EnvGuard(*self.ENV_KEYS):
            os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
            os.environ["EASYGRAPH_GPU_BACKEND"] = "typo"
            os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "FALSE"
            self.assertEqual(gpu_runtime.gpu_backend_name(), "typo")

    def test_gpu_disabled_ignores_backend_typo(self):
        with EnvGuard(*self.ENV_KEYS):
            os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"
            os.environ["EASYGRAPH_GPU_BACKEND"] = "typo"
            os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
            self.assertFalse(gpu_runtime.gpu_runtime_enabled())
            self.assertEqual(gpu_runtime.gpu_backend_name(), "typo")


if __name__ == "__main__":
    unittest.main()
