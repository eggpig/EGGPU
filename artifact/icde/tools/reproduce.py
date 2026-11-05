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
    "test-protocols-cpu": "artifact/icde/tools/test_protocols_cpu.py",
    "paper-current-grid": "artifact/icde/paper_results_current/prepare_selection.py",
    "paper-current-render": "artifact/icde/paper_results_current/render.py",
    "formal-incremental": "artifact/icde/campaign/formal_incremental_admission_v1/admit_incremental.py",
    "e4s-incremental": "artifact/icde/campaign/formal_incremental_e4s_v1/admit_e4s_incremental.py",
    "state-qualification-envelope": "artifact/icde/campaign/state_completion_queue_guard_v1/derive.py",
    "arena-diagnostic-plan": "artifact/icde/campaign/arena_diagnostic_v1/prepare.py",
    "wcc-v3-plan": "artifact/icde/experimental/native_wcc_wide_v3/prepare_qualification.py",
    "test-followup-cpu": "artifact/icde/tools/test_followup_cpu.py",
    "state-protocol-prepare": "artifact/icde/campaign/state_protocol_completion_v1/prepare.py",
    "state-protocol-test": "artifact/icde/campaign/state_protocol_completion_v1/run_cpu_tests.py",
    "e6-protocol-worker": "artifact/icde/campaign/e6_protocol_completion_v2/e6_worker.py",
    "e6-protocol-plan": "artifact/icde/campaign/e6_protocol_completion_v2/build_plans.py",
    "e4s-archive": "artifact/icde/campaign/e4s_recoverable_outputs_v1/archive_outputs.py",
    "e4s-export-check": "artifact/icde/campaign/e4s_recoverable_outputs_v1/export_eligible.py",
    "formal-disposition-inspect": "artifact/icde/campaign/formal_admission_disposition_v1/inspect_ledger.py",
    "external-formal-watch": "artifact/icde/campaign/watch_external_formals_v4.py",
    "state-deployment-template": "artifact/icde/tools/state_deployment_template.py",
    "paper-e2-e6": "artifact/icde/paper_results/render.py",
    "capacity-results": "artifact/icde/paper_results/capacity.py",
    "test-projection-cpu": "artifact/icde/tools/test_projection_cpu.py",
    "test-native-arrays-cpu": "artifact/icde/tools/test_native_array_cpu.py",
    "run-e4s": "artifact/icde/campaign/e4s/run_e4s.py",
    "state-plan": "artifact/icde/campaign/state_final_plan/state_plan.py",
    "state-grid": "artifact/icde/campaign/state_grid_prepare_v3/prepare_state_grid.py",
    "hotspot-inputs": "artifact/icde/campaign/state_partition_hotspot/prepare_hotspot_inputs.py",
    "pr-control-plan": "artifact/icde/campaign/e3_unweighted_control/pr_plan.py",
    "run-queue": "artifact/icde/campaign/run_campaign_queue_v8.py",
    "gum-fp32-worker": "artifact/icde/campaign/gum_fp32_author/gum_worker.py",
    "paper-results": "artifact/icde/presentation/paper_results.py",
    "protocol-grid": "artifact/icde/tools/protocol_grid.py",
    "bind-e6-references": "artifact/icde/campaign/e6_memory/bind_references.py",
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
    "run-sg": "artifact/icde/campaign/sg_current/run_sg.py",
    "sg-worker": "artifact/icde/campaign/sg_current/sg_worker.py",
    "run-native": "artifact/icde/campaign/native_load_p1/run_native.py",
    "native-worker": "artifact/icde/campaign/native_load_p1/native_worker.py",
    "state-inputs": "artifact/icde/campaign/state_inputs.py",
    "state-worker": "artifact/icde/campaign/state_worker.py",
    "run-e6": "artifact/icde/campaign/e6_memory/run_e6.py",
    "e6-worker": "artifact/icde/campaign/e6_memory/e6_worker.py",
    "run-sequence16": "artifact/icde/campaign/sg_current/run_sequence16.py",
    "sequence16-worker": "artifact/icde/campaign/sg_current/sequence16_worker.py",
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
