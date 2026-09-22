#!/usr/bin/env python3
"""Prepare exact normalized, deterministically weighted Gunrock inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "tools" / "csr_to_matrix_market.cpp"
BINARY = HERE / "tools" / "csr_to_matrix_market"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compile_converter():
    command = [
        os.environ.get("CXX", "/usr/bin/g++"), "-O3", "-DNDEBUG",
        "-std=c++17", str(SOURCE), "-o", str(BINARY),
    ]
    subprocess.run(command, check=True)


def prepare(manifest_path: Path, output_dir: Path):
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not metadata.get("weights_path"):
        sibling = manifest_path.with_name(f"{manifest_path.stem}.weighted.json")
        if not sibling.is_file():
            raise RuntimeError(f"weighted sibling is absent for {manifest_path}")
        manifest_path = sibling
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    output = output_dir / f"{metadata['name']}.aligned-weighted.mtx"
    sidecar = output.with_suffix(".metadata.json")
    converter_digest = sha256(BINARY)
    expected = {
        "dataset": metadata["name"],
        "nodes": int(metadata["num_nodes"]),
        "csr_entries": int(metadata["num_entries"]),
        "matrix_entries": int(metadata["num_entries"] if metadata["directed"] else metadata["num_entries"] // 2),
        "directed": bool(metadata["directed"]),
        "weighted_manifest": str(manifest_path.resolve()),
        "converter_sha256": converter_digest,
    }
    if output.is_file() and sidecar.is_file():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if all(previous.get(key) == value for key, value in expected.items()):
            print(f"reused {output} ({output.stat().st_size} bytes)", flush=True)
            return
    partial = output.with_suffix(".mtx.partial")
    partial.unlink(missing_ok=True)
    started = time.perf_counter()
    command = [
        str(BINARY),
        "--offsets", str((root / metadata["offsets_path"]).resolve()),
        "--indices", str((root / metadata["indices_path"]).resolve()),
        "--weights", str((root / metadata["weights_path"]).resolve()),
        "--output", str(partial),
        "--nodes", str(metadata["num_nodes"]),
        "--entries", str(metadata["num_entries"]),
        "--directed", "1" if metadata["directed"] else "0",
    ]
    subprocess.run(command, check=True)
    partial.replace(output)
    result = {
        **expected,
        "output": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "conversion_seconds": time.perf_counter() - started,
        "normalization": metadata.get("normalization"),
        "weight_semantics": "1 + (zero_based_src * zero_based_dst) % num_nodes",
    }
    sidecar.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compile_converter()
    for manifest in args.manifests:
        prepare(manifest.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
