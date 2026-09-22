#!/usr/bin/env python3
"""Independent final gate for the staged V10_UNIFORM asset bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_no_staging_paths(root: Path, published: Path) -> None:
    """Reject temporary-stage references while allowing the published root."""
    root = root.resolve()
    published = published.resolve()
    forbidden_paths = [str(root)] if root != published else []
    forbidden_tokens = [".EGGPU_V10_UNIFORM.stage."]
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".csv",
            ".json",
            ".md",
            ".tex",
            ".txt",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        for forbidden in (*forbidden_paths, *forbidden_tokens):
            if forbidden in text:
                raise ValueError(
                    f"staging-local path remains in {path}: {forbidden}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--published-dir", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    args = parser.parse_args()

    root = args.assets_dir.resolve()
    published = args.published_dir.resolve()
    release_path = root / "V10_UNIFORM_RELEASE_MANIFEST.json"
    checksums_path = root / "SHA256SUMS.txt"
    release = load_json(release_path)
    if (
        release.get("status") != "asset_gate_pass_paper_sync_pending"
        or release.get("paper_sync_pending") is not True
        or release.get("asset_root") != str(published)
        or release.get("candidate_binary_sha256") != args.candidate_sha256
        or release.get("runtime_python_snapshot_sha256") != args.runtime_sha256
        or release.get("memory_provenance_mode") != "unified-candidate"
        or release.get("timing_estimator") != "minimum_of_five"
    ):
        raise ValueError("release manifest identity/publication state differs")

    excluded = {
        "V10_UNIFORM_RELEASE_MANIFEST.json",
        "SHA256SUMS.txt",
    }
    actual_content = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )
    records = release.get("files")
    if not isinstance(records, list):
        raise ValueError("release manifest file inventory is missing")
    recorded_content = [str(record["file"]) for record in records]
    if recorded_content != actual_content or len(records) != int(
        release.get("content_file_count", -1)
    ):
        raise ValueError("release manifest does not cover every content file")
    for record in records:
        path = root / record["file"]
        if path.stat().st_size != int(record["bytes"]) or sha256(path) != record[
            "sha256"
        ]:
            raise ValueError(f"release inventory hash differs: {record['file']}")

    checksum_records = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ./", 1)
        checksum_records[relative] = digest
    expected_checksum_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksums_path
    )
    if sorted(checksum_records) != expected_checksum_files:
        raise ValueError("SHA256SUMS does not cover every non-self artifact")
    for relative, digest in checksum_records.items():
        if sha256(root / relative) != digest:
            raise ValueError(f"SHA256SUMS mismatch: {relative}")

    audit_path = root / "V10_ASSET_ONLY_AUDIT.json"
    audit = load_json(audit_path)
    if (
        audit.get("status") != "pass"
        or audit.get("audit_scope") != "assets_only"
        or audit.get("paper_sync_pending") is not True
        or audit.get("assets_dir") != str(published)
        or audit.get("missing_assets")
        or audit.get("invalid_assets")
    ):
        raise ValueError("assets-only audit is not a clean deferred-paper pass")
    if release["assets_only_audit"]["sha256"] != sha256(audit_path):
        raise ValueError("release manifest asset-audit hash differs")

    ledger = pd.read_csv(root / "final_13_cell_outcome_ledger.csv", low_memory=False)
    eggpu = ledger[ledger["baseline"].eq("EGGPU")]
    if len(eggpu) != 208 or eggpu.duplicated(["dataset", "function"]).any():
        raise ValueError("V10 ledger does not contain 208 unique EGGPU cells")
    if not eggpu["execution_status"].eq("ok").all():
        raise ValueError("V10 ledger contains a non-OK EGGPU cell")
    if set(eggpu["candidate_sha256"].astype(str)) != {args.candidate_sha256}:
        raise ValueError("V10 ledger candidate SHA differs")
    if set(eggpu["runtime_python_snapshot_sha256"].astype(str)) != {
        args.runtime_sha256
    }:
        raise ValueError("V10 ledger runtime SHA differs")
    for metric in ("build", "kernel", "e2e"):
        if not eggpu[f"{metric}_estimator"].eq("minimum_of_five").all():
            raise ValueError(f"V10 {metric} estimator is not minimum-of-five")
        if not pd.to_numeric(eggpu["sample_count"]).eq(5).all():
            raise ValueError("V10 timing sample count differs")
    if not pd.to_numeric(eggpu["memory_sample_count"]).eq(3).all():
        raise ValueError("V10 memory sample count differs")
    if set(eggpu["memory_candidate_sha256"].astype(str)) != {
        args.candidate_sha256
    }:
        raise ValueError("V10 memory candidate SHA differs")

    raw = pd.read_csv(root / "final_13_eggpu_timing_samples.csv")
    raw_groups = raw.groupby(["dataset", "function", "metric"]).size()
    if len(raw) != 2080 or len(raw_groups) != 416 or not raw_groups.eq(5).all():
        raise ValueError("canonical main-matrix raw5 evidence is incomplete")
    policy = load_json(root / "eggpu_timing_stability_policy.json")
    if (
        policy.get("eggpu_center") != "minimum_of_five"
        or policy.get("acceptance_metrics") != ["e2e"]
        or int(policy.get("raw_sample_groups", -1)) != 416
    ):
        raise ValueError("main-matrix timing policy differs")

    workflow = load_json(root / "workflow_state_reuse_provenance.json")
    if (
        workflow.get("status") != "pass_v10_candidate_bound"
        or workflow.get("bound_to_v10_candidate_binary") is not True
        or workflow.get("candidate_binary_sha256") != args.candidate_sha256
        or workflow.get("runtime_python_snapshot_sha256") != args.runtime_sha256
        or int(workflow.get("raw_sample_files", -1)) != 60
        or int(workflow.get("calls_per_sample_file", -1)) != 5
        or int(workflow.get("summary_rows", -1)) != 60
        or int(workflow.get("sample_rows", -1)) != 300
        or workflow.get("statistics", {}).get("within_dataset_estimator")
        != "arithmetic mean of five independent processes"
    ):
        raise ValueError("V10 workflow provenance/shape differs")
    workflow_calls = pd.read_csv(
        root / "workflow_five_call_self_control_arithmetic_mean.csv"
    )
    workflow_cumulative = pd.read_csv(
        root / "workflow_five_call_reuse_cumulative_arithmetic_mean.csv"
    )
    if (
        len(workflow_calls) != 10
        or len(workflow_cumulative) != 10
        or set(workflow_cumulative["estimator"]) != {"arithmetic mean of five"}
        or "minimum" in " ".join(
            workflow_calls.get("estimator", pd.Series(dtype=str)).astype(str)
        ).lower()
    ):
        raise ValueError("workflow paper assets do not retain arithmetic-mean protocol")

    diff = load_json(root / "V9_TO_V10_KEY_METRIC_DIFF.json")
    if (
        diff.get("status") != "pass"
        or diff.get("inputs", {}).get("v10_assets") != str(published)
        or diff.get("v10_candidate_binary_sha256") != args.candidate_sha256
    ):
        raise ValueError("V9-to-V10 metric diff provenance differs")

    assert_no_staging_paths(root, published)
    print(
        json.dumps(
            {
                "status": "pass",
                "timing_cells": 208,
                "memory_cells": 208,
                "timing_raw_groups": 416,
                "workflow_sample_files": 60,
                "workflow_summary_rows": 60,
                "workflow_estimator": "arithmetic_mean_of_five",
                "content_files": len(records),
                "checksum_files": len(checksum_records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
