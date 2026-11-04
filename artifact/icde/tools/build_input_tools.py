#!/usr/bin/env python3
"""Plan or compile the published CPU input tools into a new external directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--cxx", default="c++")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = root / "artifact/icde/inputs"
    output = args.build_dir.resolve()
    if output.exists() or output == root or root in output.parents:
        raise ValueError("Choose an absent build directory outside the checkout")
    commands = [
        [args.cxx, "-O3", "-std=c++17", str(source / "edge_list_to_csr.cpp"), "-o", str(output / "edge_list_to_csr")],
        [args.cxx, "-O3", "-std=c++17", "-fopenmp", str(source / "weights.cpp"), "-lcrypto", "-o", str(output / "weights")],
    ]
    receipt = dict(status="PLAN_ONLY_NOT_BUILT", commands=commands, gpu_executed=False,
                   dependencies=["C++17 compiler", "OpenMP", "OpenSSL development headers and libcrypto"],
                   source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.cpp")})
    if args.execute:
        output.mkdir(parents=True)
        try:
            for command in commands:
                subprocess.run(command, check=True)
            receipt["status"] = "PASS_CPU_TOOL_BUILD_TESTS_REQUIRED"
        except BaseException:
            receipt["status"] = "FAIL_CPU_TOOL_BUILD"
            raise
        finally:
            (output / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
