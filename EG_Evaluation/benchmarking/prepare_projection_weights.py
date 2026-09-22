#!/usr/bin/env python3
"""Attach deterministic weights to a single-incidence projection artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_FORMAT = "eggpu-logical-undirected-projection-v1"
EXPECTED_WEIGHT_SEMANTICS = "1 + (src * dst) % num_nodes"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_under(root: Path, path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside {root}: {resolved}") from exc
    return resolved


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("projection_manifest")
    parser.add_argument("--allowed-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    root = Path(args.allowed_root).expanduser().resolve()
    manifest_path = require_under(
        root, Path(args.projection_manifest), "projection manifest"
    )
    work_dir = require_under(root, Path(args.work_dir), "work directory")
    work_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("format") != EXPECTED_FORMAT:
        raise ValueError("unexpected projection format")

    source_record = metadata.get("source_graph", {})
    source_manifest = resolve(
        manifest_path.parent, source_record["manifest_path"]
    )
    source_metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source_metadata.get("weight_semantics") != EXPECTED_WEIGHT_SEMANTICS:
        raise ValueError(
            "projection weight generation only supports the recorded "
            f"{EXPECTED_WEIGHT_SEMANTICS!r} benchmark semantics"
        )
    if source_metadata.get("weight_dtype") != "float64":
        raise ValueError("source benchmark weights must use float64")

    num_nodes = int(metadata["num_nodes"])
    num_edges = int(metadata["unique_edge_count"])
    lower_v = resolve(manifest_path.parent, metadata["lower_V_path"])
    lower_e = resolve(manifest_path.parent, metadata["lower_E_path"])
    output = manifest_path.with_name(
        f"{metadata['name']}.logical-undirected.weights.f64"
    )
    if output.exists() and not args.force:
        raise FileExistsError(
            f"{output} exists; pass --force only after validating provenance"
        )

    source = Path(__file__).with_suffix(".cpp").resolve()
    source_sha = sha256_file(source)
    binary = work_dir / f"prepare_projection_weights-{source_sha[:16]}"
    compiler_env = os.environ.copy()
    compiler_tmp = work_dir / "compiler-tmp"
    compiler_tmp.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TMP", "TEMP"):
        compiler_env[key] = str(compiler_tmp)
    compile_command = [
        args.compiler,
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
        subprocess.run(compile_command, check=True, env=compiler_env)
    compiler_version = subprocess.run(
        [args.compiler, "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        env=compiler_env,
    ).stdout.splitlines()[0]

    run_command = [
        str(binary),
        "--offsets",
        str(lower_v),
        "--indices",
        str(lower_e),
        "--num-nodes",
        str(num_nodes),
        "--num-edges",
        str(num_edges),
        "--output",
        str(output),
        "--threads",
        str(args.threads),
    ]
    subprocess.run(run_command, check=True, env=compiler_env)
    expected_bytes = num_edges * 8
    if output.stat().st_size != expected_bytes:
        raise ValueError("projection weight file has the wrong size")

    completed = datetime.now(timezone.utc)
    metadata["generation"] = int(metadata.get("generation", 1)) + 1
    metadata["weights_path"] = os.path.relpath(output, manifest_path.parent)
    metadata["weight_dtype"] = "float64"
    metadata["weight_key"] = str(source_metadata.get("weight_key", "weight"))
    metadata["weight_semantics"] = EXPECTED_WEIGHT_SEMANTICS
    metadata["weight_artifact"] = {
        "path": metadata["weights_path"],
        "bytes": expected_bytes,
        "sha256": sha256_file(output),
    }
    metadata["weight_provenance"] = {
        "created_at_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "source_weight_manifest": os.path.relpath(
            source_manifest, manifest_path.parent
        ),
        "source_weight_manifest_sha256": sha256_file(source_manifest),
        "generator_source": str(source),
        "generator_source_sha256": source_sha,
        "generator_binary_sha256": sha256_file(binary),
        "compiler": compiler_version,
        "compile_command": compile_command,
        "run_command": run_command,
        "temporary_directory_policy": (
            "all compiler temporary files are restricted to the recorded "
            "shared-filesystem work directory"
        ),
    }
    incomplete = manifest_path.with_name(
        f".{manifest_path.name}.incomplete-{os.getpid()}"
    )
    incomplete.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
