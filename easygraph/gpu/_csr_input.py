"""Explicit canonical outgoing-CSR admission to the existing owned host store.

Columns are positions in the supplied sorted original-label node array. Rows
describe exactly the directed arcs stored by the caller. An undirected graph
must supply both orientations; neither admission nor updates synthesize them.
No caller arrays are adopted, and no private producer capability is minted.
"""
from collections.abc import Mapping
from time import perf_counter

import numpy as np

_CHUNK = 1 << 20


def prepare_csr(graphs):
    """Validate canonical layout; OwnedCSRStore performs the actual owned copies.

Reject noncanonical rows rather than invoking the legacy COO sorting fallback.
The bounded sortedness check never allocates the m-by-2 endpoint table.
"""
    from ._arena.input_hotpath.csr_session import CSRInput
    if not isinstance(graphs, Mapping) or not graphs:
        raise ValueError('CSR input requires a nonempty graph mapping')
    prepared, profiles = {}, {}
    for name, item in graphs.items():
        started = perf_counter()
        if not isinstance(item, Mapping) or not {'nodes', 'offsets', 'indices'} <= set(item):
            raise ValueError('CSR graph requires nodes, offsets and indices')
        if set(item) - {'nodes', 'offsets', 'indices', 'weights'}:
            raise ValueError('CSR input stores explicit directed arcs; allowed fields are nodes, offsets, indices, weights')
        nodes, row, col = (np.asarray(item[key]) for key in ('nodes', 'offsets', 'indices'))
        if nodes.ndim != 1 or nodes.dtype.kind not in 'iu':
            raise ValueError('CSR nodes must be a sorted unique integer-label array')
        if nodes.dtype.kind == 'u' and nodes.size and int(nodes.max()) >= 1 << 63:
            raise ValueError('Node label exceeds signed int64')
        if len(nodes) >= 1 << 31 or np.any(nodes[1:] <= nodes[:-1]):
            raise ValueError('CSR nodes must be sorted unique signed-int64 labels within int32 capacity')
        if row.ndim != 1 or col.ndim != 1 or row.dtype.kind not in 'iu' or col.dtype.kind not in 'iu':
            raise ValueError('CSR offsets and indices must be one-dimensional integer arrays')
        n, m = len(nodes), len(col)
        if (m >= 1 << 31 or len(row) != n + 1 or int(row[0]) != 0
                or int(row[-1]) != m or np.any(row[1:] < row[:-1])):
            raise ValueError('Invalid CSR offset geometry')
        if m and (int(col.min()) < 0 or int(col.max()) >= n):
            raise ValueError('CSR indices must be node positions, not original labels')
        # Check adjacent columns in bounded chunks. Row boundaries are exempt;
        # repeated offsets (empty rows/isolates) remain part of the node domain.
        for first in range(1, m, _CHUNK):
            end = min(m, first + _CHUNK)
            invalid = col[first:end] <= col[first-1:end-1]
            lo, hi = np.searchsorted(row, np.asarray((first, end), dtype=row.dtype), side='left')
            boundaries = row[lo:hi].astype(np.int64, copy=False)
            invalid[boundaries - first] = False
            if invalid.any():
                raise ValueError('CSR rows must have sorted unique destination positions')
        weights = item.get('weights')
        if weights is not None:
            values = np.asarray(weights)
            if (values.ndim != 1 or len(values) != m or values.dtype.kind not in 'iuf'
                    or not np.all(np.isfinite(values)) or np.any(values < 0)):
                raise ValueError('CSR requires one finite nonnegative real weight per stored arc')
        prepared[name] = dict(nodes=nodes, edges=CSRInput(row, col), weights=weights)
        profiles[name] = dict(input_format='csr', caller_arrays_adopted=False,
            validation_seconds=perf_counter()-started, dense_endpoint_temporary_bytes=0,
            canonical_dense_endpoint_bytes_avoided=16*m,
            ownership='existing OwnedCSRStore validates and copies nodes/offsets/indices/weights during registration',
            adjacency='explicit outgoing arcs; no symmetrization; updates use the existing directed-arc journal')
    return prepared, profiles
