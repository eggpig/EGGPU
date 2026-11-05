#!/usr/bin/env python3
"""Inspect reproduction status and forward explicit build/qualification commands.

There is no `run-all`: inputs, baselines and separate qualification gates remain explicit.
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
    "build-input-tools": "artifact/icde/inputs/setup_tools.py",
    "test-input-tools": "artifact/icde/tools/test_inputs_cpu.py",
    "preflight-inputs": "artifact/icde/inputs/preflight.py",
    "test-state-controls": "artifact/icde/campaign/state_cpu_tests.py",
    "test-campaign-cpu": "artifact/icde/tools/test_campaign_cpu.py",
    "capture-local-setup": "artifact/icde/tools/capture_setup.py",
    "fetch-inputs": "artifact/icde/inputs/fetch_inputs.py",
    "prepare-inputs": "artifact/icde/inputs/prepare_inputs.py",
    "qualify-sg": "Easy-Graph/tests/gpu/qualify_public_csr.py",
    "qualify-legacy-capability": "Easy-Graph/tests/gpu/qualify_capability_route.py",
    "run-native-example": "artifact/icde/tools/run_native.py",
    "run-sg": "artifact/icde/campaign/run_sg.py",
    "sg-worker": "artifact/icde/campaign/sg_worker.py",
    "run-native": "artifact/icde/campaign/run_native.py",
    "native-worker": "artifact/icde/campaign/native_worker.py",
    "state-inputs": "artifact/icde/campaign/state_inputs.py",
    "state-worker": "artifact/icde/campaign/state_worker.py",
    "run-e6": "artifact/icde/campaign/run_e6.py",
    "e6-worker": "artifact/icde/campaign/e6_worker.py",
    "run-sequence16": "artifact/icde/campaign/run_sequence16.py",
    "sequence16-worker": "artifact/icde/campaign/sequence16_worker.py",
    "run-gum": "artifact/icde/campaign/run_gum.py",
    "gum-worker": "artifact/icde/campaign/gum_worker.py",
    "run-cugraph-mg": "artifact/icde/campaign/run_cugraph_mg.py",
    "cugraph-mg-worker": "artifact/icde/campaign/cugraph_mg_worker.py",
    "run-sg-authors": "artifact/icde/campaign/baselines/run_sg_authors.py",
    "run-c2graph-p": "artifact/icde/campaign/c2p_v1/run_c2p.py",
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
