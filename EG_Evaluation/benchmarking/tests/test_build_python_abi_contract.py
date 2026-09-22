import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SETUP = ROOT / "Easy-Graph" / "setup.py"


class BuildPythonAbiContractTests(unittest.TestCase):
    def test_cmake_is_pinned_to_the_invoking_python_for_old_and_new_pybind(self):
        source = SETUP.read_text()
        self.assertIn('f"-DPYTHON_EXECUTABLE={sys.executable}"', source)
        self.assertIn('f"-DPython_EXECUTABLE={sys.executable}"', source)
        self.assertIn('f"-DPython3_EXECUTABLE={sys.executable}"', source)
        self.assertIn('"-DPYBIND11_FINDPYTHON=ON"', source)


if __name__ == "__main__":
    unittest.main()
