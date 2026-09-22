#!/usr/bin/env python3
"""Prepare normalized main-dataset CSR inputs for the strict SYgraph BFS audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from library_baselines import load_graph


DATASETS = {
    "ca-HepTh": ("datasets/undirected/ca-HepTh.txt", False),
    "LastFM": ("datasets/undirected/LastFM.txt", False),
    "p2p-Gnutella04": ("datasets/directed/p2p-Gnutella04.txt", True),
    "ca-HepPh": ("datasets/undirected/ca-HepPh.txt", False),
    "email-Enron": ("datasets/undirected/email-Enron.txt", False),
    "ca-CondMat": ("datasets/undirected/ca-CondMat.txt", False),
    "soc-Epinions1": ("datasets/directed/soc-Epinions1.txt", True),
    "soc-Slashdot0811": ("datasets/directed/soc-Slashdot0811.txt", True),
    "ER-100k": ("datasets/directed/ER-100k.txt", True),
    "web-NotreDame": ("datasets/directed/web-NotreDame.txt", True),
    "com-youtube": ("datasets/undirected/com-youtube.ungraph.txt", False),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(__file__).resolve().parent / "tools" / "edge_list_to_eggpu_csr",
    )
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    eval_root = args.eval_root.resolve()
    output_root = args.output_root.resolve()
    selected = args.dataset or list(DATASETS)
    output_root.mkdir(parents=True, exist_ok=True)
    for name in selected:
        relative_path, directed = DATASETS[name]
        source = eval_root / relative_path
        destination = output_root / name
        manifest = destination / f"{name}.json"
        if manifest.is_file() and not args.force:
            print(f"[SYgraph CSR] {name}: reuse {manifest}", flush=True)
            continue
        views = load_graph(source)
        n, directed_edges, undirected_edges = views["all_vertices"]
        edges = directed_edges if directed else undirected_edges
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination, prefix=f".{name}.", suffix=".edges", delete=False
        ) as handle:
            temporary = Path(handle.name)
            for u, v in edges[["src", "dst"]].itertuples(index=False, name=None):
                handle.write(f"{int(u)} {int(v)}\n")
        command = [
            str(args.converter.resolve()),
            "--input", str(temporary),
            "--output-dir", str(destination),
            "--name", name,
            "--format", "edge-list",
            "--source-url", f"local-main-benchmark:{relative_path}",
            "--num-nodes", str(int(n)),
            "--expected-edges", str(len(edges)),
        ]
        command.append("--directed" if directed else "--mirror-undirected")
        try:
            subprocess.run(command, check=True)
        finally:
            temporary.unlink(missing_ok=True)
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        metadata.update(
            {
                "source_file": relative_path,
                "normalization": (
                    "Identical to the main benchmark: sorted contiguous node mapping; "
                    "self-loops and duplicate edges removed; undirected pairs canonicalized."
                ),
                "sygraph_qualification_only": True,
            }
        )
        manifest.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"[SYgraph CSR] {name}: nodes={n:,} simple_edges={len(edges):,}",
            flush=True,
        )


if __name__ == "__main__":
    main()
