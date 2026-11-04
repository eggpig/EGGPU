#!/usr/bin/env python3
"""Inspect reproduction status and forward explicit build/qualification commands.

There is deliberately no `run-all`: E1-E6 campaign runners are not yet published.
GPU qualification is a separate explicit command, never a consequence of status.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
TARGETS = {
    "verify": "artifact/icde/tools/verify.py",
    "test-cpu": "artifact/icde/tools/test_cpu.py",
    "build": "artifact/icde/tools/build.py",
    "build-input-tools": "artifact/icde/tools/build_input_tools.py",
    "test-input-tools": "artifact/icde/inputs/test_input_tools.py",
    "test-state-controls": "artifact/icde/tests/state_cpu_tests.py",
    "fetch-inputs": "artifact/icde/inputs/fetch_inputs.py",
    "prepare-inputs": "artifact/icde/inputs/prepare_inputs.py",
    "qualify-sg": "Easy-Graph/tests/gpu/qualify_capability_route.py",
    "run-native-example": "artifact/icde/tools/run_native.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", *TARGETS])
    args, rest = parser.parse_known_args()
    if args.command == "status":
        if rest:
            parser.error("status takes no additional arguments")
        print((ROOT / "artifact/icde/campaign_status.json").read_text())
        return
    raise SystemExit(subprocess.call([sys.executable, "-B", str(ROOT / TARGETS[args.command]), *rest]))


if __name__ == "__main__":
    main()
