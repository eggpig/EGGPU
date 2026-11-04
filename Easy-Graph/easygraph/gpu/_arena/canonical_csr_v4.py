"""Generic canonical-COO bridge; no baseline graph/kernel changes.

Original vertex IDs are mapped with a contiguous-range or bounded lookup-table
path when possible, otherwise searchsorted. Sorted/unique edges are verified,
so canonical input never needs to be sorted again. All validation is timed.
"""
import time
import numpy as np

LOOKUP_MAX_BYTES = 64 * 1024 * 1024
LOOKUP_MAX_SPAN_RATIO = 16

def prepare(nodes, edges, weights, backend):
    begin = time.perf_counter()
    stages = {}
    assert nodes.ndim == 1 and edges.ndim == 2 and edges.shape[1] == 2
    assert weights.shape == (len(edges),) and len(nodes) > 0
    assert np.all(nodes[1:] > nodes[:-1])
    assert len(nodes) < 2**32 and len(edges) < 2**32
    stages['validate_nodes_seconds'] = time.perf_counter() - begin
    tick = time.perf_counter()
    lower, upper = int(nodes[0]), int(nodes[-1])
    assert len(edges) == 0 or (int(edges.min()) >= lower and int(edges.max()) <= upper)
    span = upper - lower + 1
    if span == len(nodes):
        dense = edges - lower
        mode = 'contiguous_vertex_ids'
        lookup_bytes = 0
    elif span <= LOOKUP_MAX_SPAN_RATIO * len(nodes) and span * 8 <= LOOKUP_MAX_BYTES:
        lookup = np.full(span, -1, np.int64)
        lookup[nodes - lower] = np.arange(len(nodes), dtype=np.int64)
        dense = lookup[edges - lower]
        assert np.all(dense >= 0), 'unknown vertex ID'
        mode = 'bounded_vertex_lookup'
        lookup_bytes = lookup.nbytes
        del lookup
    else:
        dense = np.searchsorted(nodes, edges)
        assert np.array_equal(nodes[dense], edges)
        mode = 'searchsorted_vertex_ids'
        lookup_bytes = 0
    stages['map_vertex_ids_seconds'] = time.perf_counter() - tick
    tick = time.perf_counter()
    sorted_unique = bool(np.all((dense[1:, 0] > dense[:-1, 0]) |
        ((dense[1:, 0] == dense[:-1, 0]) & (dense[1:, 1] > dense[:-1, 1]))))
    stages['validate_edge_order_seconds'] = time.perf_counter() - tick
    tick = time.perf_counter()
    if not sorted_unique:
        order = np.lexsort((dense[:, 1], dense[:, 0]))
        dense, weights = dense[order], weights[order]
        assert np.all((dense[1:, 0] != dense[:-1, 0]) |
                      (dense[1:, 1] != dense[:-1, 1])), 'duplicate COO must be normalized under an explicit rule'
    stages['sort_seconds'] = time.perf_counter() - tick
    tick = time.perf_counter()
    offsets = np.empty(len(nodes) + 1, np.uint32)
    offsets[0] = 0
    offsets[1:] = np.cumsum(np.bincount(dense[:, 0], minlength=len(nodes)))
    indices = dense[:, 1].astype(np.uint32)
    arrays = dict(offsets=offsets, indices=indices, weights=np.asarray(weights, dtype=np.float64))
    if backend in ('cggraph', 'c2graph'):
        assert np.all(np.isfinite(weights)) and np.all(weights >= 0) and np.all(weights < 2**32)
        wu = weights.astype(np.uint32)
        assert np.array_equal(weights, wu), 'author integral-weight contract'
        arrays['weights_u32'] = wu
    stages['build_csr_arrays_seconds'] = time.perf_counter() - tick
    elapsed = time.perf_counter() - begin
    return arrays, elapsed, dict(mapping=mode, lookup_temporary_bytes=lookup_bytes,
        input_sorted_unique=sorted_unique, sorting_performed=not sorted_unique,
        stages=stages, total_seconds=elapsed,
        contract='already-loaded canonical host COO with original vertex IDs -> validated native CSR; every mapping/validation/conversion is measured')
