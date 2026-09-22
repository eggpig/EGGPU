#!/usr/bin/env python3
"""Attach reproducibility hashes to a controlled R-MAT CSR manifest."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative(path, base):
    return os.path.relpath(str(path.resolve()), start=str(base.resolve()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--generator-source", required=True, type=Path)
    parser.add_argument("--generator-binary", required=True, type=Path)
    parser.add_argument("--compile-command", required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    base = manifest_path.parent
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "eggpu-csr-v1" or "rmat" not in metadata:
        raise ValueError(f"not a controlled R-MAT EGGPU CSR manifest: {manifest_path}")

    artifacts = {}
    for key, field in (("offsets", "offsets_path"), ("indices", "indices_path")):
        path = (base / metadata[field]).resolve()
        artifacts[key] = {
            "path": relative(path, base),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    source = args.generator_source.resolve()
    binary = args.generator_binary.resolve()
    metadata["csr_artifacts"] = artifacts
    metadata["generator_provenance"] = {
        "implementation": "EGGPU controlled R-MAT CSR generator v1",
        "source_path": relative(source, base),
        "source_sha256": sha256(source),
        "binary_path": relative(binary, base),
        "binary_sha256": sha256(binary),
        "compile_command": args.compile_command,
        "reference_specification": "https://graph500.org/?page_id=12",
        "qualification": (
            "Uses the Graph500 initiator and edge factor with a deterministic "
            "counter-based RNG and bijective label scrambling; it is distribution-"
            "compatible but not bit-identical to the Graph500 reference generator."
        ),
    }
    metadata["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
