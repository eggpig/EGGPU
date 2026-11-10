"""Create the partitioned CSR input used by native GPU sessions."""
import hashlib
import json
import operator
from pathlib import Path
import uuid

import numpy as np

from ._manifest_signatures import sign


def write_partitioned_graph(vertex_count, edges, directory, *, weights=None,
                            devices=(0,), graph_id=None, directed=True):
    """Write outgoing CSR partitions and return their manifest path.

    Vertices are numbered from 0 to ``vertex_count - 1``. ``edges`` is an
    integer array of shape (m, 2); with ``directed=False``, supply both directions.
    Optional weights are finite nonnegative floats, one per arc. Duplicate
    arcs are rejected. Each GPU receives a consecutive, nonempty vertex range.
    The destination directory must not already exist.
    """
    if isinstance(vertex_count, (bool, np.bool_)):
        raise ValueError("vertex_count must be an integer")
    n = operator.index(vertex_count)
    if not 0 < n <= np.iinfo(np.int32).max:
        raise ValueError("vertex_count must fit a positive int32")
    selected = tuple(operator.index(d) for d in devices)
    if (not selected or len(set(selected)) != len(selected)
            or any(d < 0 for d in selected) or len(selected) > n):
        raise ValueError("devices must be distinct nonnegative IDs, at most vertex_count")
    arcs = np.asarray(edges)
    if arcs.size == 0:
        arcs = np.empty((0, 2), dtype=np.int32)
    if arcs.ndim != 2 or arcs.shape[1] != 2 or arcs.dtype.kind not in "iu":
        raise ValueError("edges must be an integer array with shape (m, 2)")
    if len(arcs) > np.iinfo(np.int32).max or np.any(arcs < 0) or np.any(arcs >= n):
        raise ValueError("Edge indices or adjacency count exceed the int32 domain")
    weighted = weights is not None
    values = np.ones(len(arcs)) if weights is None else np.asarray(weights, dtype=np.float64)
    if values.shape != (len(arcs),) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Supply one finite nonnegative weight per arc")
    order = np.lexsort((arcs[:, 1], arcs[:, 0]))
    arcs, values = arcs[order].astype(np.int32), values[order]
    if np.any(np.all(arcs[1:] == arcs[:-1], axis=1)):
        raise ValueError("Duplicate directed arcs")
    if not directed:
        reverse_order = np.lexsort((arcs[:, 0], arcs[:, 1]))
        if not np.array_equal(arcs, arcs[reverse_order, ::-1]):
            raise ValueError("Undirected inputs must supply both arc directions")
        if weighted and not np.array_equal(values, values[reverse_order]):
            raise ValueError("Undirected arc pairs must have matching weights")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cuts = np.linspace(0, n, len(selected) + 1, dtype=np.int64)
    offsets = np.r_[0, np.cumsum(np.bincount(arcs[:, 0], minlength=n))]
    parts = []
    for rank, (first, last) in enumerate(zip(cuts[:-1], cuts[1:])):
        a, b = int(offsets[first]), int(offsets[last])
        targets = arcs[a:b, 1]
        outside = (targets < first) | (targets >= last)
        halo = np.unique(targets[outside])
        mapping = np.r_[np.arange(first, last, dtype="<u8"), halo.astype("<u8")]
        local = (targets - first).astype("<i4")
        local[outside] = last - first + np.searchsorted(halo, targets[outside])
        arrays = {
            "offsets": (offsets[first:last + 1] - a).astype("<i4"),
            "indices": local,
            "local_to_global": mapping,
            "owners": np.searchsorted(cuts[1:], mapping, side="right").astype("<i4"),
        }
        if weighted:
            arrays["weights"] = values[a:b].astype("<f8")
        files = {}
        for name, array in arrays.items():
            path = directory / f"p{rank}.{name}.bin"
            array.tofile(path)
            files[name] = dict(path=path.name, dtype=str(array.dtype), count=len(array),
                               byte_size=array.nbytes,
                               sha256=hashlib.sha256(memoryview(array)).hexdigest())
        parts.append(dict(partition_id=rank, device_id=selected[rank],
                          owned_vertex_count=int(last - first), halo_vertex_count=len(halo),
                          adjacency_count=b - a, cut_edge_count=int(outside.sum()),
                          ownership=dict(encoding="range", vertex_begin=int(first), vertex_end=int(last)),
                          files=files))
    document = dict(format="egmgpu-partitioned-csr", version=1,
                    graph=dict(id=graph_id or str(uuid.uuid4()), generation=1,
                               vertex_count=n, adjacency_count=len(arcs), edge_count=len(arcs),
                               directed=bool(directed), weighted=weighted),
                    indexing=dict(endianness="little", global_vertex_dtype="uint64",
                                  index_dtype="int32", offset_dtype="int32"),
                    partitioning=dict(ownership_encoding="range"),
                    view=dict(orientation="outgoing" if directed else "undirected",
                              weight_key="weight" if weighted else None),
                    partitions=parts)
    topology, weight, signature = sign(document)
    document["graph"]["topology_sha256"] = topology
    if weighted:
        document["graph"]["weight_sha256"] = weight
    document["view"]["signature"] = signature
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(document, indent=2) + "\n")
    return manifest
