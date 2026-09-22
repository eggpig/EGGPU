#!/usr/bin/env python3
"""Create exact BFS reference digests from accepted EGGPU experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


MAIN_DATASETS = (
    "ca-HepTh", "LastFM", "p2p-Gnutella04", "ca-HepPh", "email-Enron",
    "ca-CondMat", "soc-Epinions1", "soc-Slashdot0811", "ER-100k",
    "web-NotreDame", "com-youtube",
)

RMAT_DATASETS = (
    "R-MAT-S20-EF16",
    "R-MAT-S22-EF16",
    "R-MAT-S24-EF16",
    "R-MAT-S26-EF16",
)


def main_reference(main_result: Path, dataset: str):
    path = (
        main_result
        / "measurement_passes"
        / "timing"
        / "logs"
        / dataset
        / "details"
        / "EGGPU_BFS.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        sources = np.asarray(archive["sources"], dtype=np.int64)
        values = np.ascontiguousarray(archive["values"], dtype=np.float64)
    return {
        "artifact": str(path),
        "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "shape": list(values.shape),
        "sources": [int(value) for value in sources],
    }


def scale_reference(result_dir: Path, dataset: str, manifest: Path, source_count: int):
    aggregate = json.loads(
        (result_dir / "raw" / f"{dataset}_BFS_timing.json").read_text()
    )
    metadata = json.loads(manifest.read_text())
    recorded = metadata.get("benchmark_sources_zero_based")
    if isinstance(recorded, list) and len(recorded) >= source_count:
        sources = [int(value) for value in recorded[:source_count]]
    else:
        n = int(metadata["num_nodes"])
        sources = sorted(
            {min(n - 1, index * n // max(1, source_count)) for index in range(source_count)}
        )
    result = aggregate["result"]
    return {
        "artifact": str(result_dir / "raw" / f"{dataset}_BFS_timing.json"),
        "sha256": result["sha256"],
        "shape": [int(value) for value in result["shape"]],
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-result", required=True, type=Path)
    parser.add_argument("--core-result", required=True, type=Path)
    parser.add_argument("--com-orkut-manifest", required=True, type=Path)
    parser.add_argument("--gap-twitter-manifest", required=True, type=Path)
    parser.add_argument("--rmat-result", type=Path)
    parser.add_argument("--rmat-manifest", action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-count", type=int, default=8)
    args = parser.parse_args()

    references = {
        dataset: main_reference(args.main_result.resolve(), dataset)
        for dataset in MAIN_DATASETS
    }
    references["com-Orkut"] = scale_reference(
        args.core_result.resolve() / "eggpu_large_matrix",
        "com-Orkut",
        args.com_orkut_manifest.resolve(),
        args.source_count,
    )
    references["GAP-twitter"] = scale_reference(
        args.core_result.resolve() / "eggpu_large_matrix",
        "GAP-twitter",
        args.gap_twitter_manifest.resolve(),
        args.source_count,
    )
    if args.rmat_manifest:
        if args.rmat_result is None:
            parser.error("--rmat-result is required with --rmat-manifest")
        rmat_result = args.rmat_result.resolve()
        for manifest in args.rmat_manifest:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            dataset = str(metadata["name"])
            if dataset not in RMAT_DATASETS:
                parser.error(f"unexpected R-MAT dataset: {dataset}")
            references[dataset] = scale_reference(
                rmat_result,
                dataset,
                manifest.resolve(),
                args.source_count,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(references, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} with {len(references)} exact BFS references")


if __name__ == "__main__":
    main()
