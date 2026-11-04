#!/usr/bin/env python3
"""Run the CPU capability/numerical-contract suite without loading CUDA.

A package namespace skips unrelated EasyGraph root-package optional imports.
The actual easygraph.gpu modules and the published tests are imported normally.
Requires NumPy and pytest in the caller's selected environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if args.work.exists():
        raise ValueError("Choose an absent CPU-test output directory")
    args.work.mkdir(parents=True)
    source = Path(__file__).resolve().parents[3] / "Easy-Graph"
    package = types.ModuleType("easygraph")
    package.__path__ = [str(source / "easygraph")]
    sys.modules["easygraph"] = package
    import pytest
    tests = [source / "tests/gpu/test_capability_dispatch.py", source / "tests/gpu/test_arena_numerical_contract.py"]
    status = pytest.main(["-q", "-p", "no:cacheprovider", "--basetemp", str(args.work / "temporary"), *map(str, tests)])
    receipt = dict(status="PASS_CPU_SUITE" if status == 0 else "FAIL_CPU_SUITE", gpu_executed=False,
                   tests={str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tests},
                   scope="Actual GPU Python modules under an EasyGraph package namespace; no CUDA/DSO qualification.")
    (args.work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
