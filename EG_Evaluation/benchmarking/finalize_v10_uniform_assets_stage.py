#!/usr/bin/env python3
"""Initialize and seal a staged V10_UNIFORM asset release.

The helper never renames or deletes directories.  The shell runner owns the
single atomic rename after this helper has produced and verified relocation-
safe manifests and a conventional SHA256SUMS inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RELEASE_MANIFEST = "V10_UNIFORM_RELEASE_MANIFEST.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
ASSET_AUDIT_FILE = "V10_ASSET_ONLY_AUDIT.json"
DIFF_JSON = "V9_TO_V10_KEY_METRIC_DIFF.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def relocate_value(value: Any, stage: Path, final: Path) -> Any:
    """Relocate staging-root path strings without altering other evidence."""
    if isinstance(value, dict):
        return {
            key: relocate_value(child, stage, final)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [relocate_value(child, stage, final) for child in value]
    if isinstance(value, str):
        return value.replace(str(stage.resolve()), str(final.resolve()))
    return value


def relative_files(root: Path) -> list[Path]:
    return sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
    )


def assert_stage_layout(stage: Path, final: Path) -> None:
    stage = stage.resolve()
    final = final.resolve()
    if not stage.is_dir():
        raise ValueError(f"staging directory does not exist: {stage}")
    if final.exists():
        raise ValueError(f"final publication target already exists: {final}")
    if stage.parent != final.parent:
        raise ValueError("staging and final directories must share one parent")
    if stage == final:
        raise ValueError("staging and final directories must differ")


def publication_record(
    final: Path,
    repo_root: Path,
    *,
    audit_status: str,
    audit_sha256: str | None = None,
) -> dict[str, Any]:
    audit_script = (
        repo_root
        / "EG_Evaluation"
        / "benchmarking"
        / "audit_v10_paper_claim_consistency.py"
    )
    command = (
        f"python3 {audit_script} --paper-dir <PAPER_DIR> "
        f"--assets-dir {final} --max-over-median-limit 5 "
        "--median-over-min-limit 3 --json-output "
        "<PAPER_RELEASE_EVIDENCE_DIR>/V10_PAPER_CLAIM_AUDIT.json"
    )
    return {
        "status": (
            "asset_gate_pass_paper_sync_pending"
            if audit_status == "pass"
            else "assets_prepared_paper_sync_pending"
        ),
        "paper_sync_pending": True,
        "paper_release_evidence_status": "not_generated",
        "assets_only_audit_status": audit_status,
        "assets_only_audit_artifact": ASSET_AUDIT_FILE,
        "assets_only_audit_sha256": audit_sha256,
        "atomic_publish_target": str(final),
        "post_sync_claim_audit_command": command,
        "post_sync_release_evidence_contract": (
            "Root synchronizes the paper after this asset gate, then stores the "
            "full paper-and-assets claim audit JSON and compiled PDF SHA-256 in "
            "a separate paper release-evidence directory."
        ),
    }


def assert_no_stage_path(root: Path, stage: Path) -> None:
    needle = str(stage.resolve())
    offenders = []
    text_suffixes = {
        ".csv",
        ".json",
        ".md",
        ".tex",
        ".txt",
        ".tsv",
        ".yaml",
        ".yml",
    }
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        if needle in text:
            offenders.append(str(path.relative_to(root)))
    if offenders:
        raise ValueError(
            "staging-local absolute paths remain in publication assets: "
            f"{offenders[:20]}"
        )


def validate_release_identity(
    manifest: dict[str, Any],
    candidate_sha256: str,
    runtime_sha256: str,
) -> None:
    if manifest.get("status") != "complete":
        raise ValueError("uniform asset manifest is not complete")
    if str(manifest.get("release_label", "")).upper() != "V10":
        raise ValueError("uniform asset manifest is not release V10")
    if manifest.get("memory_provenance_mode") != "unified-candidate":
        raise ValueError("uniform asset manifest is not unified-candidate")
    candidate = manifest.get("candidate_audit") or {}
    if candidate.get("candidate_sha256") != candidate_sha256:
        raise ValueError("uniform manifest candidate SHA differs")
    if candidate.get("runtime_python_snapshot_sha256") != runtime_sha256:
        raise ValueError("uniform manifest runtime SHA differs")
    if candidate.get("timing_estimator") != "minimum_of_five":
        raise ValueError("uniform manifest timing estimator differs")
    memory = (manifest.get("eggpu_provenance") or {}).get("memory") or {}
    if memory.get("status") != "pass_unified_candidate":
        raise ValueError("uniform manifest memory provenance did not pass")
    if memory.get("candidate_binary_sha256") != candidate_sha256:
        raise ValueError("uniform manifest memory candidate SHA differs")
    if memory.get("runtime_python_snapshot_sha256") != runtime_sha256:
        raise ValueError("uniform manifest memory runtime SHA differs")
    canonical = manifest.get("canonical_timing_evidence") or {}
    if (
        canonical.get("status") != "pass"
        or int(canonical.get("raw_sample_groups", -1)) != 416
        or int(canonical.get("raw_sample_rows", -1)) != 2080
    ):
        raise ValueError("uniform manifest lacks complete canonical raw5 evidence")


def initialize(
    stage: Path,
    final: Path,
    repo_root: Path,
    candidate_sha256: str,
    runtime_sha256: str,
) -> None:
    assert_stage_layout(stage, final)
    manifest_path = stage / "VLDB_UNIFORM_ASSET_MANIFEST.json"
    diff_path = stage / DIFF_JSON
    if not manifest_path.is_file() or not diff_path.is_file():
        raise FileNotFoundError("staged uniform manifest or V9-to-V10 diff is missing")
    manifest = read_json(manifest_path)
    validate_release_identity(manifest, candidate_sha256, runtime_sha256)

    diff = read_json(diff_path)
    if diff.get("status") != "pass":
        raise ValueError("V9-to-V10 diff status is not pass")
    if diff.get("v10_candidate_binary_sha256") != candidate_sha256:
        raise ValueError("V9-to-V10 diff candidate SHA differs")
    if diff.get("v10_runtime_python_snapshot_sha256") != runtime_sha256:
        raise ValueError("V9-to-V10 diff runtime SHA differs")
    diff.setdefault("inputs", {})["v10_assets"] = str(final)
    write_json(diff_path, diff)

    manifest["paper_sync_pending"] = True
    manifest["release_publication"] = publication_record(
        final,
        repo_root,
        audit_status="pending",
    )
    manifest["v9_to_v10_diff"] = {
        "status": "pass",
        "json": DIFF_JSON,
        "json_sha256": sha256(diff_path),
        "key_metric_csv": "V9_TO_V10_KEY_METRIC_DIFF.csv",
        "cell_timing_csv": "V9_TO_V10_CELL_TIMING_DIFF.csv",
        "memory_csv": "V9_TO_V10_MEMORY_DIFF.csv",
        "markdown": "V9_TO_V10_KEY_METRIC_DIFF.md",
    }
    write_json(manifest_path, manifest)
    assert_no_stage_path(stage, stage)


def validate_asset_audit(audit: dict[str, Any]) -> None:
    if audit.get("status") != "pass":
        raise ValueError("V10 assets-only audit status is not pass")
    if audit.get("audit_scope") != "assets_only":
        raise ValueError("V10 audit was not run in assets-only mode")
    if audit.get("paper_sync_pending") is not True:
        raise ValueError("V10 assets-only audit does not defer paper sync")
    if audit.get("claim_checks") != []:
        raise ValueError("assets-only audit unexpectedly evaluated paper claims")
    if audit.get("missing_assets") or audit.get("invalid_assets"):
        raise ValueError("assets-only audit has missing or invalid assets")
    raw = audit.get("raw_sample_audit") or {}
    if int(raw.get("audited_groups", -1)) != 416:
        raise ValueError("assets-only audit did not verify 416 timing groups")
    if raw.get("rejected_groups") or raw.get("formula_issues"):
        raise ValueError("assets-only audit rejected raw5 timing evidence")
    if raw.get("acceptance_metrics") != ["e2e"]:
        raise ValueError("assets-only audit acceptance metric differs")
    protocol = audit.get("protocol") or {}
    if (
        float(protocol.get("max_over_median_limit", -1)) != 5.0
        or float(protocol.get("median_over_min_limit", -1)) != 3.0
        or protocol.get("variance_policy") != "catastrophic_outlier_guard"
    ):
        raise ValueError("assets-only audit stability protocol differs")


def seal(
    stage: Path,
    final: Path,
    repo_root: Path,
    candidate_sha256: str,
    runtime_sha256: str,
    audit_path: Path,
) -> None:
    assert_stage_layout(stage, final)
    audit_path = audit_path.resolve()
    if audit_path.parent != stage.resolve() or audit_path.name != ASSET_AUDIT_FILE:
        raise ValueError("assets-only audit must be the canonical staging artifact")
    audit = read_json(audit_path)
    validate_asset_audit(audit)
    audit = relocate_value(audit, stage, final)
    audit["assets_dir"] = str(final)
    write_json(audit_path, audit)
    audit_hash = sha256(audit_path)

    manifest_path = stage / "VLDB_UNIFORM_ASSET_MANIFEST.json"
    manifest = read_json(manifest_path)
    validate_release_identity(manifest, candidate_sha256, runtime_sha256)
    manifest["paper_sync_pending"] = True
    manifest["release_publication"] = publication_record(
        final,
        repo_root,
        audit_status="pass",
        audit_sha256=audit_hash,
    )
    write_json(manifest_path, manifest)
    assert_no_stage_path(stage, stage)

    excluded = {RELEASE_MANIFEST, CHECKSUM_FILE}
    inventory = []
    for relative in relative_files(stage):
        if relative.as_posix() in excluded:
            continue
        path = stage / relative
        inventory.append(
            {
                "file": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    release_manifest = {
        "status": "asset_gate_pass_paper_sync_pending",
        "release_label": "V10",
        "asset_root": str(final),
        "paper_sync_pending": True,
        "candidate_binary_sha256": candidate_sha256,
        "runtime_python_snapshot_sha256": runtime_sha256,
        "memory_provenance_mode": "unified-candidate",
        "timing_estimator": "minimum_of_five",
        "variance_policy": "catastrophic_outlier_guard",
        "max_over_median_limit": 5.0,
        "median_over_min_limit": 3.0,
        "assets_only_audit": {
            "artifact": ASSET_AUDIT_FILE,
            "sha256": audit_hash,
            "status": "pass",
        },
        "paper_release_evidence_status": "not_generated",
        "post_sync_claim_audit_command": manifest["release_publication"][
            "post_sync_claim_audit_command"
        ],
        "manifest_inventory_policy": (
            f"files lists every regular artifact except {RELEASE_MANIFEST} "
            f"and {CHECKSUM_FILE}; {CHECKSUM_FILE} hashes every regular "
            f"artifact including {RELEASE_MANIFEST} and excludes only itself"
        ),
        "content_file_count": len(inventory),
        "files": inventory,
    }
    release_path = stage / RELEASE_MANIFEST
    write_json(release_path, release_manifest)

    checksum_path = stage / CHECKSUM_FILE
    lines = []
    for relative in relative_files(stage):
        if relative.as_posix() == CHECKSUM_FILE:
            continue
        lines.append(f"{sha256(stage / relative)}  ./{relative.as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert_no_stage_path(stage, stage)

    # The shell runner uses os.replace/mv only after this same-filesystem gate.
    if os.stat(stage).st_dev != os.stat(stage.parent).st_dev:
        raise ValueError("staging directory is not on its publication filesystem")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("initialize", "seal"))
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--final-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--asset-audit", type=Path)
    args = parser.parse_args()

    stage = args.stage_dir.resolve()
    final = args.final_dir.resolve()
    repo_root = args.repo_root.resolve()
    if args.phase == "initialize":
        initialize(
            stage,
            final,
            repo_root,
            args.candidate_sha256,
            args.runtime_sha256,
        )
    else:
        if args.asset_audit is None:
            parser.error("seal requires --asset-audit")
        seal(
            stage,
            final,
            repo_root,
            args.candidate_sha256,
            args.runtime_sha256,
            args.asset_audit,
        )
    print(final)


if __name__ == "__main__":
    main()
