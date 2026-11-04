"""Owned canonical COO -> CSR without a materialized dense endpoint table.

The fast path validates original labels, edge order and weights while copying
into fresh CSR arrays. Noncanonical edge order uses the existing sorting path.
No public caller array is adopted, mutated, or certified without validation.
"""
from time import perf_counter

import numpy as np
from numba import njit

from easygraph.gpu._arena.input_hotpath.owned_csr_v6 import _ProducedCSR, _PRODUCER, _produce as fallback


@njit(cache=False)
def _scan(nodes, edges, weights, contiguous, lookup, lower):
    n, m = len(nodes), len(edges)
    row = np.zeros(n + 1, np.int32)
    col = np.empty(m, np.int32)
    values = np.empty(m, np.float64)
    previous_source, previous_destination = -1, -1
    for i in range(m):
        source_label, destination_label = edges[i, 0], edges[i, 1]
        if source_label < nodes[0] or source_label > nodes[-1] or destination_label < nodes[0] or destination_label > nodes[-1]:
            return 2, row, col, values
        if contiguous:
            source, destination = source_label - lower, destination_label - lower
        else:
            source, destination = lookup[source_label - lower], lookup[destination_label - lower]
            if source < 0 or destination < 0:
                return 2, row, col, values
        if source < previous_source or (source == previous_source and destination <= previous_destination):
            return 1, row, col, values
        value = weights[i]
        if not np.isfinite(value) or value < 0:
            return 3, row, col, values
        row[source + 1] += 1
        col[i], values[i] = destination, value
        previous_source, previous_destination = source, destination
    for i in range(n):
        row[i + 1] += row[i]
    return 0, row, col, values


def _produce(nodes, edges, weights):
    started = perf_counter()
    nodes, edges, weights = np.asarray(nodes), np.asarray(edges), np.asarray(weights)
    if nodes.ndim != 1 or nodes.dtype.kind not in 'iu':
        raise ValueError('Integer node labels required')
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu':
        raise ValueError('Integer endpoint pairs required')
    for array in (nodes, edges):
        if array.dtype.kind == 'u' and array.size and int(array.max()) >= 1 << 63:
            raise ValueError('Label exceeds int64 domain')
    if len(nodes) >= 1 << 31 or len(edges) >= 1 << 31:
        raise ValueError('Signed int32 CSR capacity required')
    if np.iscomplexobj(weights):
        raise ValueError('Real weights required')
    weights = np.asarray(weights, np.float64)
    if weights.ndim != 1 or len(weights) != len(edges):
        raise ValueError('One weight per edge required')
    # The existing fallback also defines the empty-domain and wide sparse-label
    # semantics. Its entire cost remains inside this producer interval.
    if not len(nodes) or edges.dtype.kind == 'u':
        answer = fallback(nodes, edges, weights)
        answer[2]['producer_total_seconds'] = perf_counter() - started
        return answer
    if np.any(nodes[1:] <= nodes[:-1]):
        raise ValueError('Sorted unique node labels required')
    owned_nodes = nodes.astype(np.int64, copy=True)
    lower, upper = int(owned_nodes[0]), int(owned_nodes[-1])
    span = upper - lower + 1
    contiguous = span == len(nodes)
    if not contiguous and (span > 16 * len(nodes) or span * 8 > 64 << 20):
        answer = fallback(nodes, edges, weights)
        answer[2]['producer_total_seconds'] = perf_counter() - started
        return answer
    lookup = np.empty(0, np.int64) if contiguous else np.full(span, -1, np.int64)
    if not contiguous:
        lookup[owned_nodes - lower] = np.arange(len(nodes), dtype=np.int64)
    validation_setup = perf_counter() - started
    scan_started = perf_counter()
    code, row, col, values = _scan(owned_nodes, edges, weights, contiguous, lookup, lower)
    scan_seconds = perf_counter() - scan_started
    if code == 1:
        # Unsorted inputs are accepted under the same old explicit duplicate
        # rule, and no partial scan cost is removed from the total.
        answer = fallback(nodes, edges, weights)
        answer[2].update(fused_fallback_scan_seconds=scan_seconds,
                         producer_total_seconds=perf_counter() - started)
        return answer
    if code == 2:
        raise ValueError('Unknown vertex ID')
    if code == 3:
        raise ValueError('Finite nonnegative weights required')
    for array in (owned_nodes, row, col, values):
        array.flags.writeable = False
    certificate = _ProducedCSR(_PRODUCER, (owned_nodes, row, col, values))
    profile = dict(mapping='fused_contiguous' if contiguous else 'fused_bounded_lookup',
                   input_sorted_unique=True, sorting_performed=False,
                   validation_setup_seconds=validation_setup,
                   fused_validation_copy_build_seconds=scan_seconds,
                   lookup_temporary_bytes=lookup.nbytes,
                   dense_endpoint_temporary_bytes=0,
                   producer_total_seconds=perf_counter() - started,
                   transfer_payload_bytes=row.nbytes + col.nbytes,
                   contract='Original caller COO to newly owned validated CSR; validation, copying and fallback are timed.')
    return owned_nodes, certificate, profile


def prepare_owned(data):
    prepared, profiles = {}, {}
    for name, item in data.items():
        nodes, certificate, profile = _produce(item['nodes'], item['edges'], item['weights'])
        prepared[name] = dict(nodes=nodes, edges=certificate, weights=None)
        profiles[name] = profile
    return prepared, profiles
