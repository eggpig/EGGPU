#!/usr/bin/env python3
"""Run public import, capability, CSR and numerical CPU contracts without CUDA.

Requires NumPy, SciPy and pytest in the caller's selected environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--native-dso", type=Path, help="Optional explicitly built extension for the compiled CPU subset")
    args = parser.parse_args()
    if args.work.exists():
        raise ValueError("Choose an absent CPU-test output directory")
    args.work.mkdir(parents=True)
    source = Path(__file__).resolve().parents[3] / "Easy-Graph"
    sys.path.insert(0, str(source))
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')
    native = args.native_dso.resolve() if args.native_dso else None
    if native is not None:
        if not native.is_file() or native.name.split(".")[0] != "cpp_easygraph":
            raise ValueError("Explicit built cpp_easygraph extension required")
        sys.path.insert(0, str(native.parent))
    import pytest
    tests = [source / 'tests/gpu' / name for name in (
        'test_capability_dispatch.py', 'test_arena_numerical_contract.py',
        'test_public_csr_input.py', 'test_public_lazy_import.py', 'test_arena_cold_import.py',
        'test_wcc_chunked_host.py', 'test_arena_output_ownership.py',
        'test_unified_contract.py', 'test_static_native_encoding.py')]
    compiled = ["test_pr_self_loop_and_explicit_native_options", "test_upstream_cpu_dijkstra_precision_and_new_functions"]
    deselect = [] if native else ["-k", " and ".join("not " + name for name in compiled)]
    status = pytest.main(["-q", "-p", "no:cacheprovider", '--confcutdir=' + str(source/'tests/gpu'),
                         "--basetemp", str(args.work / "temporary"), *deselect, *map(str, tests)])
    receipt = dict(status="PASS_CPU_SUITE" if status == 0 else "FAIL_CPU_SUITE", gpu_executed=False,
                   native_extension=None if native is None else dict(bytes=native.stat().st_size, sha256=hashlib.sha256(native.read_bytes()).hexdigest()),
                   compiled_subset="PENDING_EXPLICIT_BUILD" if native is None else "EXECUTED_CPU_ONLY",
                   excluded_compiled_tests=compiled if native is None else [],
                   tests={str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tests},
                   scope="Normal public EasyGraph imports and actual host/dispatch code; no CUDA/DSO qualification.")
    (args.work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
