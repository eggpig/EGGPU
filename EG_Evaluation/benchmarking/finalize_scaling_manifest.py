#!/usr/bin/env python3
"""Attach checksums and normalized edge-count provenance to an EGGPU CSR manifest."""

import argparse
import hashlib
import json
import os
from datetime import datetime
from datetime import timezone
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


def resolve(base, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def portable_path(path, base):
    """Store artifact paths relative to the manifest for relocatable bundles."""

    return os.path.relpath(str(path.resolve()), start=str(base.resolve()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--compressed-source")
    parser.add_argument("--plain-source")
    parser.add_argument("--download-url", required=True)
    parser.add_argument("--catalog-url", required=True)
    parser.add_argument("--raw-edge-records", type=int, required=True)
    parser.add_argument("--simple-directed-edges", type=int)
    parser.add_argument("--unique-undirected-edges", type=int)
    parser.add_argument(
        "--benchmark-sources-one-based",
        default="",
        help="Comma-separated official source ids; converted to zero-based ids in the manifest.",
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    offsets = resolve(base, metadata["offsets_path"])
    indices = resolve(base, metadata["indices_path"])
    artifacts = {
        "offsets": {
            "path": portable_path(offsets, base),
            "bytes": offsets.stat().st_size,
            "sha256": sha256(offsets),
        },
        "indices": {
            "path": portable_path(indices, base),
            "bytes": indices.stat().st_size,
            "sha256": sha256(indices),
        },
    }
    if metadata.get("original_labels_path"):
        labels = resolve(base, metadata["original_labels_path"])
        artifacts["original_labels"] = {
            "path": portable_path(labels, base),
            "bytes": labels.stat().st_size,
            "sha256": sha256(labels),
        }
    sources = {}
    for key, value in (
        ("compressed", args.compressed_source),
        ("plain", args.plain_source),
    ):
        if value:
            path = Path(value).resolve()
            sources[key] = {
                "path": portable_path(path, base),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    metadata["catalog_url"] = args.catalog_url
    metadata["download_url"] = args.download_url
    metadata["source_artifacts"] = sources
    metadata["csr_artifacts"] = artifacts
    metadata["edge_counts"] = {
        "raw_edge_records": args.raw_edge_records,
        "simple_directed_edges": args.simple_directed_edges,
        "unique_undirected_edges": args.unique_undirected_edges,
        "csr_entries": int(metadata["num_entries"]),
        "self_loops_removed": int(metadata.get("self_loops_removed", 0)),
        "duplicates_removed": int(metadata.get("duplicates_removed", 0)),
    }
    metadata["normalization"] = (
        "Original download retained; algorithm input removes self-loops and "
        "uses a simple-graph CSR with zero-based contiguous internal labels."
    )
    if args.benchmark_sources_one_based:
        one_based = [
            int(value.strip())
            for value in args.benchmark_sources_one_based.split(",")
            if value.strip()
        ]
        if any(value < 1 or value > int(metadata["num_nodes"]) for value in one_based):
            raise ValueError("benchmark source is outside the one-based node domain")
        metadata["benchmark_sources_one_based"] = one_based
        metadata["benchmark_sources_zero_based"] = [value - 1 for value in one_based]
    metadata["notes"] = args.notes
    metadata["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
