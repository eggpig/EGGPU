#!/usr/bin/env python3
"""Build and finalize an EGGPU logical undirected projection artifact.

The C++ worker performs the large-array transformation.  This driver parses
the source CSR manifest, pins every writable path to an explicitly allowed
shared-filesystem root, compiles the worker without using /tmp, and records
content-addressed provenance for the input and output artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path


FORMAT = "eggpu-logical-undirected-projection-v1"
SOURCE_FORMAT = "eggpu-csr-v1"
ARTIFACT_PATH_FIELDS = {
    "lower_V": "lower_V_path",
    "lower_E": "lower_E_path",
    "upper_V": "upper_V_path",
    "upper_E": "upper_E_path",
    "degree": "degree_path",
    "forward_V": "forward_V_path",
    "forward_E": "forward_E_path",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_from(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def require_under(root: Path, value: Path, label: str) -> Path:
    resolved = value.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside {root}: {resolved}") from exc
    return resolved


def portable_path(path: Path, base: Path) -> str:
    return os.path.relpath(str(path.resolve()), start=str(base.resolve()))


def run_checked(command, *, env=None, capture=False):
    return subprocess.run(
        [str(value) for value in command],
        check=True,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def expected_checksum(metadata, role):
    artifacts = metadata.get("csr_artifacts", {})
    record = artifacts.get(role)
    if isinstance(record, dict):
        return record.get("sha256")
    return None


def verify_source_manifest(manifest_path: Path):
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("format") != SOURCE_FORMAT:
        raise ValueError(
            f"source format must be {SOURCE_FORMAT!r}, found "
            f"{metadata.get('format')!r}"
        )
    if not bool(metadata.get("directed")):
        raise ValueError("logical projection preprocessing requires a directed CSR")
    if metadata.get("offset_dtype") != "int32":
        raise ValueError("source offsets must use int32")
    if metadata.get("index_dtype") != "int32":
        raise ValueError("source indices must use int32")
    if metadata.get("node_labels") != "zero_based_contiguous":
        raise ValueError("source node labels must be zero-based and contiguous")

    num_nodes = int(metadata["num_nodes"])
    num_entries = int(metadata["num_entries"])
    int32_limit = (1 << 31) - 1
    if not 0 <= num_nodes < int32_limit:
        raise ValueError("source num_nodes exceeds the signed int32 CSR ABI")
    if not 0 <= num_entries < int32_limit:
        raise ValueError("source num_entries exceeds the signed int32 CSR ABI")

    base = manifest_path.parent
    offsets_path = resolve_from(base, metadata["offsets_path"])
    indices_path = resolve_from(base, metadata["indices_path"])
    if offsets_path.stat().st_size != (num_nodes + 1) * 4:
        raise ValueError("source offset file size does not match num_nodes")
    if indices_path.stat().st_size != num_entries * 4:
        raise ValueError("source index file size does not match num_entries")
    return metadata, offsets_path, indices_path, num_nodes, num_entries


def compile_worker(source: Path, build_dir: Path, compiler: str, env):
    source_digest = sha256_file(source)
    compiler_version = run_checked(
        [compiler, "--version"], env=env, capture=True
    ).stdout.splitlines()[0]
    binary = build_dir / f"prepare_logical_undirected_projection-{source_digest[:16]}"
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fopenmp",
        "-pipe",
        str(source),
        "-o",
        str(binary),
    ]
    if not binary.exists():
        run_checked(command, env=env)
    return binary.resolve(), {
        "compiler": compiler,
        "compiler_version": compiler_version,
        "build_command": command,
        "source_path": str(source),
        "source_sha256": source_digest,
        "binary_sha256": sha256_file(binary),
    }


def artifact_record(path: Path, base: Path):
    return {
        "path": portable_path(path, base),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--name")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--binary")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc)
    allowed_root = Path(args.allowed_root).expanduser().resolve()
    if not allowed_root.is_dir():
        raise ValueError(f"--allowed-root is not a directory: {allowed_root}")
    output_dir = require_under(
        allowed_root, Path(args.output_dir), "--output-dir"
    )
    work_dir = require_under(allowed_root, Path(args.work_dir), "--work-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.source_manifest).expanduser().resolve()
    (
        source_metadata,
        offsets_path,
        indices_path,
        num_nodes,
        num_entries,
    ) = verify_source_manifest(manifest_path)
    name = args.name or f"{source_metadata.get('name', manifest_path.stem)}-projection"

    script_path = Path(__file__).resolve()
    worker_source = script_path.with_suffix(".cpp")
    if not worker_source.is_file():
        raise FileNotFoundError(worker_source)

    process_work = work_dir / f"driver-{os.getpid()}"
    compiler_tmp = process_work / "compiler-tmp"
    runtime_tmp = process_work / "runtime-tmp"
    build_dir = work_dir / "bin"
    for path in (compiler_tmp, runtime_tmp, build_dir):
        path.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    child_env["TMPDIR"] = str(compiler_tmp)
    child_env["TMP"] = str(compiler_tmp)
    child_env["TEMP"] = str(compiler_tmp)

    try:
        if args.binary:
            binary = Path(args.binary).expanduser().resolve()
            if not binary.is_file():
                raise FileNotFoundError(binary)
            compiler_record = {
                "compiler": None,
                "compiler_version": None,
                "build_command": None,
                "source_path": str(worker_source),
                "source_sha256": sha256_file(worker_source),
                "binary_sha256": sha256_file(binary),
            }
        else:
            binary, compiler_record = compile_worker(
                worker_source, build_dir, args.compiler, child_env
            )

        child_env["TMPDIR"] = str(runtime_tmp)
        child_env["TMP"] = str(runtime_tmp)
        child_env["TEMP"] = str(runtime_tmp)
        command = [
            binary,
            "--offsets",
            offsets_path,
            "--indices",
            indices_path,
            "--num-nodes",
            num_nodes,
            "--num-entries",
            num_entries,
            "--output-dir",
            output_dir,
            "--work-dir",
            runtime_tmp,
            "--allowed-root",
            allowed_root,
            "--name",
            name,
            "--threads",
            args.threads,
        ]
        if args.force:
            command.append("--force")
        run_checked(command, env=child_env)

        result_path = output_dir / f"{name}.logical-undirected.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("format") != FORMAT:
            raise ValueError("C++ worker wrote an unexpected projection format")
        if int(result["num_nodes"]) != num_nodes:
            raise ValueError("projection node count differs from source CSR")
        unique_edge_count = int(result["unique_edge_count"])
        if not 0 <= unique_edge_count < (1 << 31) - 1:
            raise ValueError("projection unique edge count exceeds int32")
        if (
            int(result["non_self_loop_arcs"])
            - int(result["duplicate_or_reciprocal_arcs_removed"])
            != unique_edge_count
        ):
            raise ValueError("projection edge accounting is inconsistent")

        output_artifacts = {}
        for role, field in ARTIFACT_PATH_FIELDS.items():
            path = resolve_from(output_dir, result[field])
            output_artifacts[role] = artifact_record(path, output_dir)

        expected_sizes = {
            "lower_V": (num_nodes + 1) * 4,
            "lower_E": unique_edge_count * 4,
            "upper_V": (num_nodes + 1) * 4,
            "upper_E": unique_edge_count * 4,
            "degree": num_nodes * 4,
            "forward_V": (num_nodes + 1) * 4,
            "forward_E": unique_edge_count * 4,
        }
        for role, expected_bytes in expected_sizes.items():
            found_bytes = int(output_artifacts[role]["bytes"])
            if found_bytes != expected_bytes:
                raise ValueError(
                    f"{role} size mismatch: expected {expected_bytes}, "
                    f"found {found_bytes}"
                )

        source_offsets_sha256 = sha256_file(offsets_path)
        source_indices_sha256 = sha256_file(indices_path)
        for role, actual in (
            ("offsets", source_offsets_sha256),
            ("indices", source_indices_sha256),
        ):
            expected = expected_checksum(source_metadata, role)
            if expected and expected != actual:
                raise ValueError(
                    f"source {role} checksum differs from the source manifest"
                )

        completed_at = datetime.now(timezone.utc)
        result["artifacts"] = output_artifacts
        result["source_graph"] = {
            "manifest_path": portable_path(manifest_path, output_dir),
            "manifest_sha256": sha256_file(manifest_path),
            "format": source_metadata["format"],
            "generation": int(source_metadata.get("generation", 0)),
            "name": source_metadata.get("name", manifest_path.stem),
            "directed": bool(source_metadata["directed"]),
            "num_nodes": num_nodes,
            "num_entries": num_entries,
            "offsets": {
                "path": portable_path(offsets_path, output_dir),
                "bytes": offsets_path.stat().st_size,
                "sha256": source_offsets_sha256,
            },
            "indices": {
                "path": portable_path(indices_path, output_dir),
                "bytes": indices_path.stat().st_size,
                "sha256": source_indices_sha256,
            },
        }
        result["provenance"] = {
            "created_at_utc": completed_at.isoformat(),
            "started_at_utc": started_at.isoformat(),
            "elapsed_seconds": (completed_at - started_at).total_seconds(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "driver_path": str(script_path),
            "driver_sha256": sha256_file(script_path),
            "worker": compiler_record,
            "command": [str(value) for value in command],
            "allowed_root": str(allowed_root),
            "work_dir": str(work_dir),
            "temporary_directory_policy": (
                "compiler and runtime temporary files are restricted to "
                "the recorded shared-filesystem work_dir"
            ),
        }

        incomplete = result_path.with_name(
            f".{result_path.name}.incomplete-{os.getpid()}"
        )
        incomplete.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        incomplete.replace(result_path)
        print(result_path)
    finally:
        shutil.rmtree(process_work, ignore_errors=True)


if __name__ == "__main__":
    main()
