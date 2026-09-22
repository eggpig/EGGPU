#!/usr/bin/env python3
"""Prepare the fixed 13-dataset GraphScope input bundle outside timed regions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


EVAL_ROOT = Path(
    "/home/dataset-assist-0/einwang/workspace/haorandu/"
    "EGGPU_Paper_Repo/EG_Evaluation"
)

CANONICAL_SOURCE_MANIFESTS = {
    "com-youtube": EVAL_ROOT
    / "datasets/scaling/csr/com-youtube/com-youtube.json",
    "com-Orkut": EVAL_ROOT
    / "datasets/scaling/csr/com-Orkut/com-Orkut.json",
    "GAP-twitter": EVAL_ROOT
    / "datasets/scaling/csr/GAP-twitter/GAP-twitter.json",
}

DATASETS = [
    ("ca-HepTh", "undirected", "datasets/undirected/ca-HepTh.txt", None, False),
    ("LastFM", "undirected", "datasets/undirected/LastFM.txt", None, False),
    (
        "p2p-Gnutella04",
        "directed",
        "datasets/directed/p2p-Gnutella04.txt",
        None,
        False,
    ),
    ("ca-HepPh", "undirected", "datasets/undirected/ca-HepPh.txt", None, False),
    ("email-Enron", "undirected", "datasets/undirected/email-Enron.txt", None, False),
    ("ca-CondMat", "undirected", "datasets/undirected/ca-CondMat.txt", None, False),
    (
        "soc-Epinions1",
        "directed",
        "datasets/directed/soc-Epinions1.txt",
        None,
        False,
    ),
    (
        "soc-Slashdot0811",
        "directed",
        "datasets/directed/soc-Slashdot0811.txt",
        None,
        False,
    ),
    ("ER-100k", "directed", "datasets/directed/ER-100k.txt", None, False),
    (
        "web-NotreDame",
        "directed",
        "datasets/directed/web-NotreDame.txt",
        None,
        False,
    ),
    (
        "com-youtube",
        "undirected",
        "datasets/undirected/com-youtube.ungraph.txt",
        None,
        False,
    ),
    (
        "com-Orkut",
        "undirected",
        "datasets/scaling/downloads/com-orkut.ungraph.txt",
        3_072_441,
        "sparse",
    ),
    (
        "GAP-twitter",
        "directed",
        "datasets/scaling/downloads/GAP-twitter/GAP-twitter.mtx",
        61_578_415,
        "matrix-market",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=Path(__file__).with_name("tools") / "prepare_graphscope_dataset",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def benchmark_sources(name: str, num_nodes: int) -> tuple[list[int] | None, Path | None]:
    """Load the frozen source sequence used by the large-graph benchmarks."""

    source_manifest = CANONICAL_SOURCE_MANIFESTS.get(name)
    if source_manifest is None or not source_manifest.is_file():
        return None, None
    source_metadata = json.loads(source_manifest.read_text())
    recorded = source_metadata.get("benchmark_sources_zero_based")
    if recorded is None:
        return None, source_manifest
    if not isinstance(recorded, list):
        raise ValueError(
            f"{source_manifest}: benchmark_sources_zero_based must be a list"
        )
    sources = [int(value) for value in recorded]
    if len(sources) != len(set(sources)):
        raise ValueError(f"{source_manifest}: benchmark sources contain duplicates")
    if any(source < 0 or source >= num_nodes for source in sources):
        raise ValueError(f"{source_manifest}: benchmark source is outside node domain")
    return sources, source_manifest


def normalize_manifest_paths(manifest: Path, metadata: dict) -> dict:
    """Normalize paths and propagate any frozen benchmark source sequence."""

    changed = False
    for key in ("vertices", "edges", "reverse", "undirected", "bidirected", "source"):
        value = metadata.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = (EVAL_ROOT / path).resolve()
            metadata[key] = str(path)
            changed = True
    sources, source_manifest = benchmark_sources(
        str(metadata["name"]), int(metadata["num_nodes"])
    )
    if sources is not None and metadata.get("benchmark_sources_zero_based") != sources:
        metadata["benchmark_sources_zero_based"] = sources
        metadata["benchmark_sources_manifest"] = str(source_manifest.resolve())
        changed = True
    if changed:
        manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    selected = (
        {name for name, *_ in DATASETS}
        if args.datasets == "all"
        else {item.strip() for item in args.datasets.split(",") if item.strip()}
    )
    known = {name for name, *_ in DATASETS}
    unknown = sorted(selected - known)
    if unknown:
        raise SystemExit(f"unknown datasets: {', '.join(unknown)}")

    bundle = []
    for name, graph_type, relative_path, num_nodes, input_mode in DATASETS:
        if name not in selected:
            continue
        source = EVAL_ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        output = args.output / name
        manifest = output / "manifest.json"
        if manifest.is_file() and not args.force:
            metadata = json.loads(manifest.read_text())
            if (
                metadata.get("source") == str(source)
                and int(metadata.get("source_size_bytes", -1)) == source.stat().st_size
            ):
                metadata = normalize_manifest_paths(manifest, metadata)
                print(f"[reuse] {name}: {manifest}", flush=True)
                bundle.append(metadata)
                continue

        command = [
            str(args.preprocessor),
            "--input",
            str(source),
            "--output",
            str(output),
            "--name",
            name,
        ]
        if graph_type == "directed":
            command.append("--directed")
        if num_nodes is not None and input_mode != "sparse":
            command.extend(
                ["--contiguous-one-based", "--num-nodes", str(num_nodes)]
            )
        if input_mode == "sparse":
            command.extend(["--sparse-one-based", "--num-nodes", str(num_nodes)])
        if input_mode == "matrix-market":
            command.extend(["--matrix-market", "--skip-undirected"])
        print("[prepare] " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
        metadata = json.loads(manifest.read_text())
        metadata = normalize_manifest_paths(manifest, metadata)
        bundle.append(metadata)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "format": "graphscope-normalized-csv-bundle-v1",
                "preprocessing_in_timed_build": False,
                "datasets": bundle,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(args.output / "bundle_manifest.json", flush=True)


if __name__ == "__main__":
    main()
