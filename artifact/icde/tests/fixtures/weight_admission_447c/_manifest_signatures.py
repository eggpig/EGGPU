"""Exact original manifest hash/sign functions; no experimental builders."""
from pathlib import Path
import hashlib,json

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def sign(document):
    graph, view = document["graph"], document["view"]
    parts = []
    for part in document["partitions"]:
        row = {key: part[key] for key in
               ("partition_id", "owned_vertex_count", "halo_vertex_count",
                "adjacency_count", "cut_edge_count", "ownership")}
        row["files"] = {key: {k: spec[k] for k in ("dtype", "count", "sha256")}
                        for key, spec in part["files"].items() if key != "weights"}
        parts.append(row)
    topology = canonical_digest(dict(
        format=document["format"], version=document["version"], directed=graph["directed"],
        view={key: view[key] for key in ("orientation", "weight_key")},
        vertex_count=graph["vertex_count"], edge_count=graph["edge_count"],
        adjacency_count=graph["adjacency_count"],
        ownership_encoding=document["partitioning"]["ownership_encoding"], partitions=parts))
    weight = canonical_digest(dict(weights=[
        dict(partition_id=part["partition_id"],
             array={key: part["files"]["weights"][key] for key in ("dtype", "count", "sha256")})
        for part in document["partitions"]])) if graph["weighted"] else None
    signature = canonical_digest(dict(orientation=view["orientation"], weight_key=view["weight_key"],
                                      topology_sha256=topology, weight_sha256=weight))
    return topology, weight, signature
