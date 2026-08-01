#!/usr/bin/env python
import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EASYGRAPH_ROOT = ROOT / "Easy-Graph"
if EASYGRAPH_ROOT.exists():
    sys.path.insert(0, str(EASYGRAPH_ROOT))


def run(cmd):
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return None, "not found"
    output = (proc.stdout or proc.stderr or "").strip()
    return proc.returncode, output


def detect_cuda_architectures():
    rc, output = run(
        [
            "nvidia-smi",
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc is None or rc != 0:
        return None
    archs = []
    for raw in output.splitlines():
        cap = raw.strip().replace(" ", "")
        digits = cap.replace(".", "")
        if digits.isdigit() and digits not in archs:
            archs.append(digits)
    return ";".join(archs) if archs else None


def env_cuda_root():
    for name in ("EGGPU_CUDA_ROOT", "CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT", "CONDA_PREFIX"):
        value = os.environ.get(name)
        if value:
            path = Path(value).expanduser()
            return name, path
    return None, None


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_status(module):
    try:
        imported = importlib.import_module(module)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(imported, "__version__", None) or package_version(module)
    return True, version or "imported"


def print_kv(label, value):
    print(f"{label}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Check whether the local machine can build and run the EGGPU artifact."
    )
    parser.add_argument("--strict", action="store_true", help="exit nonzero on missing core requirements")
    parser.add_argument(
        "--require-extension",
        action="store_true",
        help="treat a missing compiled cpp_easygraph extension as an error",
    )
    parser.add_argument("--expect-rapids", action="store_true", help="treat RAPIDS baseline packages as required")
    args = parser.parse_args()

    errors = []
    warnings = []

    print("== System ==")
    print_kv("python", sys.version.split()[0])
    print_kv("executable", sys.executable)
    print_kv("platform", platform.platform())
    print_kv("repo", ROOT)

    print("\n== CUDA Toolkit ==")
    root_env, cuda_root = env_cuda_root()
    print_kv("cuda root env", root_env or "unset")
    print_kv("cuda root", cuda_root or "unset")
    if cuda_root is None:
        errors.append("No CUDA root found. Set EGGPU_CUDA_ROOT or activate the EGGPU conda env.")
    else:
        nvcc = cuda_root / "bin" / "nvcc"
        print_kv("nvcc path", nvcc)
        if not nvcc.exists():
            errors.append(f"nvcc not found at {nvcc}")
        else:
            _, output = run([str(nvcc), "--version"])
            print(output.splitlines()[-1] if output else "nvcc version: unknown")

    print("\n== NVIDIA Driver/GPU ==")
    rc, output = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if rc is None or rc != 0:
        warnings.append("nvidia-smi is unavailable or failed; GPU runtime cannot be verified here.")
        print(output)
    else:
        print(output)
    detected_archs = detect_cuda_architectures()
    configured_archs = os.environ.get("EGGPU_CUDA_ARCHITECTURES") or os.environ.get(
        "CMAKE_CUDA_ARCHITECTURES"
    )
    print_kv("auto CUDA architectures", detected_archs or "unavailable")
    print_kv("configured CUDA architectures", configured_archs or "AUTO")
    if configured_archs and configured_archs.strip().upper() != "AUTO" and detected_archs:
        configured_set = {item.strip() for item in configured_archs.split(";") if item.strip()}
        detected_set = {item.strip() for item in detected_archs.split(";") if item.strip()}
        if detected_set and configured_set and not (detected_set & configured_set):
            warnings.append(
                "Configured EGGPU_CUDA_ARCHITECTURES does not include the detected GPU architecture."
            )

    print("\n== EasyGraph/EGGPU ==")
    for module in ("easygraph", "cpp_easygraph"):
        ok, detail = import_status(module)
        print_kv(module, detail)
        if not ok and module == "easygraph":
            errors.append(f"Cannot import {module}: {detail}")
        elif not ok and module == "cpp_easygraph":
            message = f"Cannot import cpp_easygraph yet: {detail}"
            if args.require_extension:
                errors.append(message)
            else:
                warnings.append(message)

    try:
        from easygraph.utils import gpu_runtime

        old_env = {
            key: os.environ.get(key)
            for key in ("EASYGRAPH_ENABLE_GPU", "EASYGRAPH_GPU_STRICT_ERRORS")
        }
        os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
        os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
        try:
            if not gpu_runtime.gpu_runtime_enabled():
                errors.append("EASYGRAPH_ENABLE_GPU=TRUE did not enable EGGPU dispatch.")
            if not gpu_runtime.gpu_strict_errors():
                errors.append("EASYGRAPH_GPU_STRICT_ERRORS=TRUE did not enable strict errors.")
            print_kv("EGGPU runtime contract", "enabled with strict errors")
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    except Exception as exc:
        warnings.append(f"Could not run EGGPU runtime contract check: {type(exc).__name__}: {exc}")

    print("\n== Python Packages ==")
    required = ("numpy", "pandas", "scipy", "networkx", "psutil", "matplotlib")
    optional = ("igraph", "pynvml")
    rapids = ("cupy", "cudf", "cugraph", "nx_cugraph")
    for name in required + optional + rapids:
        ok, detail = import_status(name)
        print_kv(name, detail)
        if not ok and name in required:
            errors.append(f"Missing required package {name}: {detail}")
        if not ok and name in rapids and args.expect_rapids:
            errors.append(f"Missing RAPIDS baseline package {name}: {detail}")

    if warnings:
        print("\n== Warnings ==")
        for item in warnings:
            print(f"- {item}")
    if errors:
        print("\n== Errors ==")
        for item in errors:
            print(f"- {item}")

    if args.strict and errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
