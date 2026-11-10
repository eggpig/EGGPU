"""Build the native Python extension with CMake."""
import os
from pathlib import Path
import shlex
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        source = Path(__file__).resolve().parent
        destination = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build = Path(self.build_temp).resolve() / ext.name
        build.mkdir(parents=True, exist_ok=True)
        gpu = os.environ.get("EASYGRAPH_ENABLE_GPU", "ON").upper()
        args = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={destination}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DPYBIND11_FINDPYTHON=ON",
            f"-DEASYGRAPH_ENABLE_GPU={gpu}",
        ]
        args.extend(shlex.split(os.environ.get("CMAKE_ARGS", "")))
        subprocess.run(["cmake", "-S", str(source), "-B", str(build), *args], check=True)
        subprocess.run(["cmake", "--build", str(build), "--config", "Release"], check=True)


setup(ext_modules=[Extension("cpp_easygraph", sources=[])],
      cmdclass={"build_ext": CMakeBuild})
