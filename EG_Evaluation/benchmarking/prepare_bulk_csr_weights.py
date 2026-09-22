#!/usr/bin/env python3
"""Create explicit deterministic weights for an existing EGGPU bulk CSR.

The main benchmark assigns every normalized edge ``(u, v)`` the weight
``1 + (u * v) % |V|``.  This tool reproduces that contract in CSR order
without materializing Python edge objects.  It writes a sibling weighted
manifest and leaves the original topology-only artifact unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def resolved(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def prepare(manifest_path: Path, edge_chunk: int, force: bool) -> Path:
    manifest_path = manifest_path.expanduser().resolve()
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "eggpu-csr-v1":
        raise ValueError(f"unsupported manifest format: {manifest_path}")
    n = int(metadata["num_nodes"])
    m = int(metadata["num_entries"])
    root = manifest_path.parent
    offsets_path = resolved(root, metadata["offsets_path"])
    indices_path = resolved(root, metadata["indices_path"])
    weights_path = root / f"{manifest_path.stem}.weights.f64"
    output_manifest = root / f"{manifest_path.stem}.weighted.json"
    expected_bytes = m * np.dtype(np.float64).itemsize

    if (
        not force
        and weights_path.is_file()
        and weights_path.stat().st_size == expected_bytes
        and output_manifest.is_file()
    ):
        print(f"[weights] reuse {output_manifest}", flush=True)
        return output_manifest

    offsets = np.memmap(offsets_path, mode="r", dtype=np.int32, shape=(n + 1,))
    indices = np.memmap(indices_path, mode="r", dtype=np.int32, shape=(m,))
    temporary = weights_path.with_suffix(weights_path.suffix + f".tmp.{os.getpid()}")
    started = time.perf_counter()
    out = np.memmap(temporary, mode="w+", dtype=np.float64, shape=(m,))

    node_start = 0
    while node_start < n:
        edge_start = int(offsets[node_start])
        target_edge = min(m, edge_start + edge_chunk)
        node_end = int(np.searchsorted(offsets, target_edge, side="right") - 1)
        node_end = min(n, max(node_start + 1, node_end))
        edge_end = int(offsets[node_end])
        degrees = np.diff(np.asarray(offsets[node_start : node_end + 1], dtype=np.int64))
        sources = np.repeat(
            np.arange(node_start, node_end, dtype=np.int64), degrees
        )
        destinations = np.asarray(indices[edge_start:edge_end], dtype=np.int64)
        if len(sources) != len(destinations):
            raise RuntimeError("CSR row expansion length mismatch")
        out[edge_start:edge_end] = (
            1 + (sources * destinations) % max(1, n)
        ).astype(np.float64, copy=False)
        node_start = node_end
        print(
            f"[weights] {metadata.get('name', manifest_path.stem)} "
            f"{edge_end:,}/{m:,} entries ({100.0 * edge_end / max(1, m):.1f}%)",
            flush=True,
        )

    out.flush()
    del out
    if temporary.stat().st_size != expected_bytes:
        raise RuntimeError("generated weight file has the wrong size")
    os.replace(temporary, weights_path)

    weighted = dict(metadata)
    weighted.update(
        {
            "weights_path": weights_path.name,
            "weight_dtype": "float64",
            "weight_key": "weight",
            "weight_semantics": "1 + (src * dst) % num_nodes",
            "weight_generation_seconds": time.perf_counter() - started,
            "parent_manifest": manifest_path.name,
        }
    )
    temp_manifest = output_manifest.with_suffix(
        output_manifest.suffix + f".tmp.{os.getpid()}"
    )
    temp_manifest.write_text(
        json.dumps(weighted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temp_manifest, output_manifest)
    print(f"[weights] wrote {output_manifest}", flush=True)
    return output_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--edge-chunk", type=int, default=8_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.edge_chunk <= 0:
        parser.error("--edge-chunk must be positive")
    outputs = [prepare(path, args.edge_chunk, args.force) for path in args.manifests]
    print(json.dumps({"status": "ok", "weighted_manifests": [str(p) for p in outputs]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
