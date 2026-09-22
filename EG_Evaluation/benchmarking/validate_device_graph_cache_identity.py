#!/usr/bin/env python3
"""Validate device CSR reuse and invalidation across a graph mutation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EASYGRAPH_ROOT = ROOT / "Easy-Graph"
if str(EASYGRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(EASYGRAPH_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def canonical_components(components):
    return sorted(sorted(int(node) for node in component) for component in components)


def cache_stats(cpp_easygraph):
    return {
        key: int(value)
        for key, value in dict(cpp_easygraph.cpp_gpu_device_csr_cache_stats()).items()
    }


def main():
    args = parse_args()
    os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
    os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
    os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
    os.environ["EASYGRAPH_GPU_DEVICE_CSR_CACHE_MAX_ENTRIES"] = "4"

    import cpp_easygraph
    import easygraph as eg

    graph = eg.DiGraph()
    graph.add_edges_from([(0, 1), (2, 3)])
    cpp_easygraph.cpp_gpu_reset_device_csr_cache()

    before = canonical_components(eg.weakly_connected_components(graph))
    stats_after_first = cache_stats(cpp_easygraph)
    repeated = canonical_components(eg.weakly_connected_components(graph))
    stats_after_repeat = cache_stats(cpp_easygraph)

    graph.add_edge(1, 2)
    after_mutation = canonical_components(eg.weakly_connected_components(graph))
    stats_after_mutation = cache_stats(cpp_easygraph)
    repeated_after_mutation = canonical_components(eg.weakly_connected_components(graph))
    stats_after_final_repeat = cache_stats(cpp_easygraph)

    if before != [[0, 1], [2, 3]] or repeated != before:
        raise RuntimeError(f"unexpected pre-mutation components: {before!r}")
    if after_mutation != [[0, 1, 2, 3]] or repeated_after_mutation != after_mutation:
        raise RuntimeError(f"stale or incorrect post-mutation components: {after_mutation!r}")
    if stats_after_repeat["structure_hits"] <= stats_after_first["structure_hits"]:
        raise RuntimeError("unchanged graph did not hit the device CSR cache")
    if stats_after_mutation["structure_misses"] <= stats_after_repeat["structure_misses"]:
        raise RuntimeError("mutated graph did not acquire a new device CSR representation")
    if stats_after_final_repeat["structure_hits"] <= stats_after_mutation["structure_hits"]:
        raise RuntimeError("rebuilt graph representation was not reusable")

    payload = {
        "status": "pass",
        "components_before": before,
        "components_after_mutation": after_mutation,
        "stats_after_first": stats_after_first,
        "stats_after_repeat": stats_after_repeat,
        "stats_after_mutation": stats_after_mutation,
        "stats_after_final_repeat": stats_after_final_repeat,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
