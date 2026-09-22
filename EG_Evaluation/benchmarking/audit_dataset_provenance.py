#!/usr/bin/env python3
"""Audit raw dataset files against pinned public metadata and simple-graph input."""

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATASETS = (
    ("ca-GrQc", "undirected", "datasets/undirected/ca-GrQc.txt", 5242, 14496, "https://snap.stanford.edu/data/ca-GrQc.html"),
    ("ca-HepTh", "undirected", "datasets/undirected/ca-HepTh.txt", 9877, 25998, "https://snap.stanford.edu/data/ca-HepTh.html"),
    ("LastFM", "undirected", "datasets/undirected/LastFM.txt", 7624, 27806, "https://snap.stanford.edu/data/feather-lastfm-social.html"),
    ("pgp", "undirected", "datasets/undirected/pgp.txt", 10680, 24316, "https://sparse.tamu.edu/Arenas/PGPgiantcompo"),
    ("ca-CondMat", "undirected", "datasets/undirected/ca-CondMat.txt", 23133, 93497, "https://snap.stanford.edu/data/ca-CondMat.html"),
    ("ca-HepPh", "undirected", "datasets/undirected/ca-HepPh.txt", 12008, 118521, "https://snap.stanford.edu/data/ca-HepPh.html"),
    ("email-Enron", "undirected", "datasets/undirected/email-Enron.txt", 36692, 183831, "https://snap.stanford.edu/data/email-Enron.html"),
    ("com-youtube", "undirected", "datasets/undirected/com-youtube.ungraph.txt", 1134890, 2987624, "https://snap.stanford.edu/data/com-Youtube.html"),
    ("p2p-Gnutella04", "directed", "datasets/directed/p2p-Gnutella04.txt", 10876, 39994, "https://snap.stanford.edu/data/p2p-Gnutella04.html"),
    ("p2p-Gnutella08", "directed", "datasets/directed/p2p-Gnutella08.txt", 6301, 20777, "https://snap.stanford.edu/data/p2p-Gnutella08.html"),
    ("wiki-Vote", "directed", "datasets/directed/wiki-Vote.txt", 7115, 103689, "https://snap.stanford.edu/data/wiki-Vote.html"),
    ("soc-Epinions1", "directed", "datasets/directed/soc-Epinions1.txt", 75879, 508837, "https://snap.stanford.edu/data/soc-Epinions1.html"),
    ("email-EuAll", "directed", "datasets/directed/email-EuAll.txt", 265214, 420045, "https://snap.stanford.edu/data/email-EuAll.html"),
    ("soc-Slashdot0811", "directed", "datasets/directed/soc-Slashdot0811.txt", 77360, 905468, "https://snap.stanford.edu/data/soc-Slashdot0811.html"),
    ("web-NotreDame", "directed", "datasets/directed/web-NotreDame.txt", 325729, 1497134, "https://snap.stanford.edu/data/web-NotreDame.html"),
    ("ER-100k", "directed", "datasets/directed/ER-100k.txt", 100000, 1000000, "project-generated synthetic graph; pinned by SHA-256"),
    ("wiki-Talk", "directed", "datasets/directed/wiki-Talk.txt", 2394385, 5021410, "https://snap.stanford.edu/data/wiki-Talk.html"),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_edge_list(path):
    nodes = set()
    directed = set()
    undirected = set()
    raw_rows = 0
    self_loops = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text or text[0] in "#%/c":
                continue
            fields = text.split()
            if len(fields) < 2:
                continue
            try:
                source = int(fields[0])
                target = int(fields[1])
            except ValueError:
                continue
            raw_rows += 1
            nodes.add(source)
            nodes.add(target)
            if source == target:
                self_loops += 1
                continue
            directed.add((source, target))
            undirected.add((source, target) if source < target else (target, source))
    return {
        "raw_nodes": len(nodes),
        "raw_edge_rows": raw_rows,
        "raw_self_loops": self_loops,
        "simple_directed_edges": len(directed),
        "simple_undirected_edges": len(undirected),
        "duplicate_directed_rows": raw_rows - self_loops - len(directed),
        "undirected_projection_collapses": raw_rows - self_loops - len(undirected),
    }


def audit_rows():
    rows = []
    for name, graph_type, relative, official_nodes, official_edges, source in DATASETS:
        path = ROOT / relative
        stats = scan_edge_list(path)
        processed_edges = (
            stats["simple_directed_edges"]
            if graph_type == "directed"
            else stats["simple_undirected_edges"]
        )
        raw_graph_edges = processed_edges + stats["raw_self_loops"]
        rows.append(
            {
                "dataset": name,
                "graph_type": graph_type,
                "path": relative,
                "sha256": sha256(path),
                "official_nodes": official_nodes,
                "official_edges": official_edges,
                **stats,
                "raw_graph_edges": raw_graph_edges,
                "processed_nodes": stats["raw_nodes"],
                "processed_edges": processed_edges,
                "node_count_matches_official": stats["raw_nodes"] == official_nodes,
                "raw_graph_edge_count_matches_official": raw_graph_edges == official_edges,
                "processed_edge_count_matches_official": processed_edges == official_edges,
                "processed_edge_delta_from_official": processed_edges - official_edges,
                "input_policy": "retain vertices; remove self-loops; deduplicate directed arcs or undirected endpoint pairs",
                "source": source,
            }
        )
    return rows


def write_outputs(rows, output_prefix):
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# EGGPU Dataset Provenance and Input Semantics",
        "",
        "All raw files remain unchanged. Every baseline receives the same simple-graph input: all observed vertices are retained, self-loop edges are removed, and duplicate arcs/endpoint pairs are deduplicated.",
        "",
        "| Dataset | Type | Official | Raw rows | Loops | Processed | Raw match | Source |",
        "|---|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        match = "T" if row["raw_graph_edge_count_matches_official"] else "F"
        lines.append(
            f"| {row['dataset']} | {row['graph_type']} | {row['official_nodes']:,}/{row['official_edges']:,} "
            f"| {row['raw_edge_rows']:,} | {row['raw_self_loops']:,} "
            f"| {row['processed_nodes']:,}/{row['processed_edges']:,} | {match} | {row['source']} |"
        )
    lines.extend(
        [
            "",
            "The official SNAP edge count includes self-loops for the datasets where loops occur. Undirected SNAP files may also store each non-loop edge in both directions. `Raw match` therefore compares the official graph-level count with unique non-loop edges plus self-loops, not with physical text rows.",
            "",
            "The benchmark intentionally removes self-loop edges under its cross-library simple-graph protocol. Consequently, a processed edge count can differ from the official raw count by exactly the number of unique self-loops without indicating corrupted data.",
            "",
            "PGP is pinned to Arenas/PGPgiantcompo. SuiteSparse reports 48,632 symmetric nonzeros, corresponding to 24,316 unique undirected edges.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        default=str(ROOT / "datasets" / "DATASET_PROVENANCE_20260710"),
    )
    args = parser.parse_args()
    paths = write_outputs(audit_rows(), Path(args.output_prefix))
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
