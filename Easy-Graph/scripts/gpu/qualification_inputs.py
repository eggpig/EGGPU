"""Small deterministic qualification inputs and independent CPU references.

Extracted from the archived portable protocol; no CUDA initialization.
"""
from pathlib import Path
import hashlib,heapq,json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")

def canonical(n, edges, weights):
    n = int(n)
    edges = np.asarray(edges, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)
    if edges.size == 0:
        edges = edges.reshape(0, 2)
    if not 0 < n <= np.iinfo(np.int32).max or edges.shape != (len(weights), 2):
        raise ValueError("Require positive int32 vertex domain and one weight per directed arc")
    if not (np.all(edges >= 0) and np.all(edges < n) and np.all(np.isfinite(weights)) and np.all(weights >= 0)):
        raise ValueError("Edges outside domain or nonfinite/negative weights")
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, weights = edges[order], weights[order]
    if np.any(np.all(edges[1:] == edges[:-1], axis=1)):
        raise ValueError("Duplicate directed arcs")
    return dict(n=n, edges=edges, weights=weights)

def heap_sssp(raw):
    """Separate heap implementation checks explicit zero-weight arcs too."""
    n, edges, weights = raw["n"], raw["edges"], raw["weights"]
    offsets = np.r_[0, np.cumsum(np.bincount(edges[:, 0], minlength=n))]
    distances = np.full(n, np.inf)
    distances[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        value, u = heapq.heappop(queue)
        if value != distances[u]:
            continue
        for position in range(int(offsets[u]), int(offsets[u + 1])):
            v = int(edges[position, 1])
            candidate = value + float(weights[position])
            if candidate < distances[v]:
                distances[v] = candidate
                heapq.heappush(queue, (candidate, v))
    return distances

def reference(raw):
    n, edges, weights = raw["n"], raw["edges"], raw["weights"]
    matrix = csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n))
    output = {"sssp": dijkstra(matrix, directed=True, indices=0)}
    iterations, residuals = {}, {}
    for algorithm, values in [("weighted_pr", weights), ("pr", np.ones(len(weights)))]:
        strength = np.bincount(edges[:, 0], weights=values, minlength=n)
        normalized = np.divide(values, strength[edges[:, 0]], out=np.zeros(len(values)), where=strength[edges[:, 0]] > 0)
        transition = csr_matrix((normalized, (edges[:, 1], edges[:, 0])), shape=(n, n))
        rank = np.full(n, 1 / n, dtype=np.float64)
        for count in range(10000):
            nxt = .15 / n + .85 * (transition @ rank + rank[strength == 0].sum() / n)
            if np.abs(nxt - rank).sum() < 1e-13:
                rank = nxt
                break
            rank = nxt
        else:
            raise RuntimeError("Independent reference did not converge")
        residual = float(np.abs(rank - (.15 / n + .85 * (transition @ rank + rank[strength == 0].sum() / n))).sum())
        if residual >= 1e-12 or abs(rank.sum() - 1) >= 1e-10:
            raise AssertionError("Independent PageRank reference failed residual contract")
        output[algorithm] = rank
        iterations[algorithm], residuals[algorithm] = count + 1, residual
    return output, dict(iterations=iterations, residual_l1=residuals,
        method="SciPy directed Dijkstra; independent normalized CSC power iteration to L1 change<1e-13; all original vertices")

def manifest(raw, directory, name, p, *, weighted=True, reverse_rows=False, cuts=None):
    """Serialize range-partition CSR; cost is measurable inside a worker."""
    from easygraph.gpu._manifest_signatures import sign
    directory = Path(directory)
    n, edges, weights = raw["n"], raw["edges"], raw["weights"]
    m = len(edges)
    if m > np.iinfo(np.int32).max:
        raise ValueError("This canonical serializer uses int32 offsets")
    if cuts is None:
        cuts = np.linspace(0, n, p + 1, dtype=int).tolist()
    if len(cuts) != p + 1 or cuts[0] != 0 or cuts[-1] != n or not all(a < b for a, b in zip(cuts, cuts[1:])):
        raise ValueError("Require a nonempty owned vertex range per rank")
    directory.mkdir(parents=True, exist_ok=False)
    offsets = np.r_[0, np.cumsum(np.bincount(edges[:, 0], minlength=n))].astype("<i4")
    order = np.arange(m, dtype=np.int32)
    if reverse_rows:
        for a, b in zip(offsets[:-1], offsets[1:]):
            order[a:b] = order[a:b][::-1]
    targets = edges[order, 1]
    parts, position_maps, total_cut = [], {}, 0
    for rank, (first, last) in enumerate(zip(cuts, cuts[1:])):
        a, b = int(offsets[first]), int(offsets[last])
        target = targets[a:b]
        halo = np.unique(target[(target < first) | (target >= last)])
        mapping = np.r_[np.arange(first, last, dtype="<u8"), halo.astype("<u8")]
        inverse = np.full(n, -1, dtype="<i4")
        inverse[mapping] = np.arange(len(mapping), dtype="<i4")
        arrays = dict(offsets=(offsets[first:last + 1] - a).astype("<i4"), indices=inverse[target], local_to_global=mapping,
                      owners=np.searchsorted(cuts[1:], mapping, side="right").astype("<i4"))
        if weighted:
            arrays["weights"] = weights[order[a:b]].astype("<f8")
        files = {}
        for key, array in arrays.items():
            file = directory / f"p{rank}.{key}.bin"
            array.tofile(file)
            files[key] = dict(path=file.name, dtype="uint64" if key == "local_to_global" else "float64" if key == "weights" else "int32",
                              count=len(array), byte_size=array.nbytes, sha256=sha(file))
        cut = int(np.count_nonzero((target < first) | (target >= last)))
        total_cut += cut
        parts.append(dict(partition_id=rank, device_id=rank, owned_vertex_count=last - first, halo_vertex_count=len(halo), adjacency_count=b - a, cut_edge_count=cut,
                          ownership=dict(encoding="range", vertex_begin=first, vertex_end=last), files=files))
        for position, eid in enumerate(order[a:b]):
            position_maps[int(eid)] = (rank, position)
    numeric_key = "controlled_positive_integer" if np.all(weights > 0) and np.all(weights == np.floor(weights)) else "controlled_nonnegative_f64"
    document = dict(format="egmgpu-partitioned-csr", version=1,
        graph=dict(id="eggpu-qualification-" + name, generation=1, vertex_count=n, adjacency_count=m, edge_count=m, directed=True, weighted=weighted),
        indexing=dict(endianness="little", global_vertex_dtype="uint64", index_dtype="int32", offset_dtype="int32"),
        partitioning=dict(ownership_encoding="range"), view=dict(orientation="outgoing", weight_key=numeric_key if weighted else None), partitions=parts)
    top, weight, sig = sign(document)
    document["graph"]["topology_sha256"] = top
    if weighted:
        document["graph"]["weight_sha256"] = weight
    document["view"]["signature"] = sig
    path = directory / "manifest.json"
    write_new(path, document)
    return dict(manifest=str(path), sha256=sha(path), cut_fraction=total_cut / max(m, 1), partitions=p), position_maps

def check_vector(actual, expected, algorithm):
    actual = np.asarray(actual)
    if actual.shape != expected.shape or actual.ndim != 1:
        raise AssertionError("Incomplete output shape")
    if algorithm in ("weighted_pr", "pr"):
        error = float(np.abs(actual - expected).sum())
        if not np.all(np.isfinite(actual)) or error >= 1e-8:
            raise AssertionError((algorithm, error))
        return error
    np.testing.assert_array_equal(actual, expected)
    return 0.0
