#!/usr/bin/env python3
"""Perform one unmeasured structural validation pass over bulk CSR artifacts."""

import argparse
import json
from pathlib import Path

import easygraph as eg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    args = parser.parse_args()
    for value in args.manifests:
        path = Path(value).resolve()
        graph = eg.read_eggpu_csr(path, validate=True)
        metadata = graph.metadata
        expected_offset_bytes = (len(graph) + 1) * 4
        expected_index_bytes = int(graph.num_entries) * 4
        offsets = path.parent / metadata["offsets_path"]
        indices = path.parent / metadata["indices_path"]
        if offsets.stat().st_size != expected_offset_bytes:
            raise RuntimeError(f"offset byte count mismatch: {offsets}")
        if indices.stat().st_size != expected_index_bytes:
            raise RuntimeError(f"index byte count mismatch: {indices}")
        weights_value = metadata.get("weights_path")
        if weights_value:
            weights = path.parent / weights_value
            expected_weight_bytes = int(graph.num_entries) * 8
            if metadata.get("weight_dtype") != "float64":
                raise RuntimeError(f"unsupported weight dtype in {path}")
            if weights.stat().st_size != expected_weight_bytes:
                raise RuntimeError(f"weight byte count mismatch: {weights}")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "dataset": graph.name,
                    "num_nodes": len(graph),
                    "num_edges": graph.num_edges,
                    "num_entries": graph.num_entries,
                    "directed": graph.is_directed(),
                    "explicit_weights": bool(weights_value),
                    "weight_key": metadata.get("weight_key") if weights_value else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
