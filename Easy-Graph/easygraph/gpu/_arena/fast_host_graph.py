"""CPU-only HostGraph factory for immutable, normalized source-major COO input.

This module does not change the frozen GPU runtime. Its subclass inherits the
existing update, reference, and repair semantics. Construction validates the
common input and performs one linear CSR -> CSC transpose carrying edge IDs,
instead of sorting incoming edges and sorting them back to outgoing order.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix

try:
    from .gpu_runtime import HostGraph
except ImportError:
    from easygraph.gpu._arena.gpu_runtime import HostGraph


class IdentityNodeMap(Mapping):
    """Exact identity labels without allocating n Python dict entries."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __iter__(self):
        return iter(range(self.n))

    def __getitem__(self, node):
        # Equal integer-valued labels follow Python dict equality (e.g. 1.0).
        try:
            value = int(node)
            valid = node == value and 0 <= value < self.n
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise KeyError(node)
        return value


def input_sha256(n, src, dst, weight, source=0):
    """Same byte contract as baselines/runner.py; no sorting is performed."""
    digest = hashlib.sha256()
    for a in (np.asarray([n, source], np.int64), np.asarray(src, np.int32),
              np.asarray(dst, np.int32), np.asarray(weight, np.float64)):
        digest.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    return digest.hexdigest()


@dataclass(frozen=True)
class CommonCOO:
    """Input boundary, not a prepared graph. Loading/hash are reported separately.

    Arrays are source-major and immutable. Each build still pays validation,
    CSR offsets, transpose/index maps, mutable weight copies, and label mapping.
    File decompression belongs to common input loading, outside construction.
    """

    n: int
    src: np.ndarray
    dst: np.ndarray
    attributes: dict
    original_nodes: tuple | None
    source: int
    actual_input_sha256: str
    input_file_sha256: str
    load_seconds: float

    def build(self, name='graph', original_labels=True, profile=False):
        return FastHostGraph(name, self.n, self.src, self.dst, self.attributes,
                             nodes=self.original_nodes if original_labels else None,
                             profile=profile)


def load_common_npz(path):
    """Load the exact NPZ consumed by the modern baseline runners."""
    start = time.perf_counter()
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        n = int(data['num_nodes'])
        src = np.ascontiguousarray(data['src'], dtype=np.int32)
        dst = np.ascontiguousarray(data['dst'], dtype=np.int32)
        weight = np.ascontiguousarray(data['weight'], dtype=np.float64)
        source = int(data['source']) if 'source' in data else 0
        original = tuple(map(int, data['original_node_ids'])) if 'original_node_ids' in data else None
    for a in (src, dst, weight):
        a.flags.writeable = False
    return CommonCOO(n, src, dst, {'cost': weight}, original, source,
                     input_sha256(n, src, dst, weight, source),
                     hashlib.sha256(path.read_bytes()).hexdigest(),
                     time.perf_counter() - start)


class FastHostGraph(HostGraph):
    """HostGraph-compatible eager bidirectional representation from common COO.

    Parallel edges are rejected, exactly as HostGraph does. No sum/min reduction
    is silently introduced. Arbitrary input order must be normalized at the
    declared common input boundary; unsorted data is rejected here.
    """

    def __init__(self, name, n, src, dst, attributes, nodes=None, profile=False):
        started = last = time.perf_counter()
        phases = {}

        def mark(name):
            nonlocal last
            now = time.perf_counter()
            if profile:
                phases[name] = now - last
            last = now

        self.name, self.n = name, int(n)
        if self.n != n or not 0 <= self.n < 2**31:
            raise ValueError('node domain must fit signed int32')
        if nodes is None:
            self.nodes = range(self.n)
            self.node_to_id = IdentityNodeMap(self.n)
        else:
            self.nodes = list(nodes)
            self.node_to_id = {node: i for i, node in enumerate(self.nodes)}
            if len(self.nodes) != self.n or len(self.node_to_id) != self.n:
                raise ValueError('node labels must uniquely cover every vertex')
        mark('node_mapping')

        s, d = np.asarray(src), np.asarray(dst)
        if s.ndim != 1 or s.shape != d.shape or not np.issubdtype(s.dtype, np.integer) or not np.issubdtype(d.dtype, np.integer):
            raise ValueError('one-dimensional integer endpoint arrays required')
        m = s.size
        if m >= 2**31 or (m and (min(s.min(), d.min()) < 0 or max(s.max(), d.max()) >= self.n)):
            raise ValueError('edge domain must fit signed int32')
        s = np.ascontiguousarray(s, dtype=np.int32)
        d = np.ascontiguousarray(d, dtype=np.int32)
        if m > 1:
            same = s[1:] == s[:-1]
            if np.any(s[1:] < s[:-1]) or np.any(same & (d[1:] < d[:-1])):
                raise ValueError('common COO input must be sorted by (source, target)')
            if np.any(same & (d[1:] == d[:-1])):
                raise ValueError('simple graph input must contain no duplicate edges')
        values = {}
        for attr, a in attributes.items():
            a = np.asarray(a, dtype=np.float64)
            if a.shape != s.shape or not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError('weights must be finite nonnegative edge arrays')
            values[attr] = a
        mark('validate_common_input')

        self.out_row = np.empty(self.n + 1, np.int32)
        self.out_row[0] = 0
        np.cumsum(np.bincount(s, minlength=self.n), out=self.out_row[1:])
        edge_ids = np.arange(m, dtype=np.int32)
        source_csr = csr_matrix((edge_ids, d, self.out_row), shape=(self.n, self.n), copy=False)
        # Rows and columns are already canonical, so no hidden sorting/dedup.
        source_csr.has_sorted_indices = True
        source_csr.has_canonical_format = True
        mark('outgoing_offsets')
        incoming = source_csr.tocsc(copy=False)
        self.row, self.col = incoming.indptr, incoming.indices
        self.in_to_out = incoming.data  # incoming edge position -> original COO position
        self.out_order = np.empty(m, dtype=np.int32)
        self.out_order[self.in_to_out] = edge_ids
        self.target = np.ascontiguousarray(d[self.in_to_out])
        mark('transpose_and_index_maps')
        self.attributes = {attr: np.ascontiguousarray(a[self.in_to_out]) for attr, a in values.items()}
        mark('mutable_attribute_copies')

        self.generation = 0
        self.versions = defaultdict(int)
        self.arrays = {}
        self.changes = defaultdict(list)
        self.change_floors = defaultdict(int)
        self.view_builds = self.repaired_values = 0
        self.host_prepare_seconds = 0.0
        self.construction_seconds = time.perf_counter() - started
        self.construction_phases = phases
        self.input_order = 'source-major COO (source,target), duplicate-free'
        # Like HostGraph, out_col/out_weight remain lazy and are charged to get().
        # Incoming CSR is intentionally eager for compatibility with the frozen
        # session manager's shape and host-byte accessors.


def native_outgoing_preparation(n, src, dst, attributes, nodes=None):
    """Separate cost probe, not a drop-in GraphSession graph.

    Builds only the outgoing representation required by original BFS/SSSP.
    Correctness/profile code charges all validation, offsets, weights, and label
    mapping. It demonstrates avoidable incoming representation work. Integrating
    this option requires explicit lazy incoming semantics in GraphSession first.
    """
    n = int(n)
    s, d = np.asarray(src), np.asarray(dst)
    if s.ndim != 1 or s.shape != d.shape or s.dtype != np.int32 or d.dtype != np.int32:
        raise ValueError('common COO requires int32 endpoint arrays')
    if n < 0 or n >= 2**31 or len(s) >= 2**31 or (s.size and (min(s.min(), d.min()) < 0 or max(s.max(), d.max()) >= n)):
        raise ValueError('invalid vertex/edge domain')
    if len(s) > 1:
        same = s[1:] == s[:-1]
        if np.any(s[1:] < s[:-1]) or np.any(same & (d[1:] <= d[:-1])):
            raise ValueError('sorted simple COO required')
    a = {}
    for k, v in attributes.items():
        v = np.asarray(v, np.float64)
        if v.shape != s.shape or not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError('nonnegative finite weights required')
        a[k] = v.copy()
    row = np.empty(n + 1, np.int32); row[0] = 0
    np.cumsum(np.bincount(s, minlength=n), out=row[1:])
    labels = range(n) if nodes is None else list(nodes)
    mapping = IdentityNodeMap(n) if nodes is None else {v: i for i, v in enumerate(labels)}
    if len(labels) != n or len(mapping) != n:
        raise ValueError('node labels must uniquely cover every vertex')
    return {'row': row, 'col': d.copy(), 'attributes': a, 'nodes': labels, 'node_to_id': mapping}
