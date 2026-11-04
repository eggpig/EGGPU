"""Typed logical graph authority, preserving R4's committed journal semantics.

CSR is a derived execution layout. The authority keeps sorted endpoint/value
arrays and an immutable net overlay, avoiding a Python edge dictionary on every
materialization. Public full-edge reads still materialize their complete result.
"""
from collections.abc import Mapping
from types import MappingProxyType
import sys
import time
import numpy as np

from easygraph.gpu._arena.round4_switch.journal import (
    JournalStore, FlushResult, Snapshot, _changes, _integer, default_materializer)

KEY_DTYPE = np.dtype([('src', '<i8'), ('dst', '<i8')])


class ArrayEdgeMap(Mapping):
    def __init__(self, edges, values):
        self.pairs = np.ascontiguousarray(edges, dtype=np.int64).copy()
        self.values_array = np.ascontiguousarray(values, dtype=np.float64).copy()
        self.pairs.flags.writeable = False
        self.values_array.flags.writeable = False
        self._keys = self.pairs.view(KEY_DTYPE).reshape(-1)

    def position(self, key):
        # Follow integer-tuple dictionary equality without truncating floats or
        # wrapping unsigned labels. Public mutation validation remains stricter.
        hash(key)
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(key)
        try:
            u, v = map(int, key)
            if (u, v) != key or not -(1 << 63) <= u < (1 << 63) or not -(1 << 63) <= v < (1 << 63):
                raise KeyError(key)
        except (TypeError, ValueError, OverflowError) as error:
            raise KeyError(key) from error
        item = np.array((u, v), dtype=KEY_DTYPE)
        pos = int(np.searchsorted(self._keys, item))
        if pos == len(self._keys) or tuple(self.pairs[pos]) != (u, v):
            raise KeyError(key)
        return pos

    def __getitem__(self, key):
        return float(self.values_array[self.position(key)])

    def __iter__(self):
        return ((int(u), int(v)) for u, v in self.pairs)

    def __len__(self):
        return len(self.values_array)

    def to_dict(self):
        return {(int(u), int(v)): float(w) for (u, v), w in zip(self.pairs, self.values_array)}

    def to_arrays(self):
        return self.pairs, self.values_array


class OverlayEdgeMap(Mapping):
    def __init__(self, base, changes):
        self.base = base.base if isinstance(base, OverlayEdgeMap) else base
        net = dict(base.net) if isinstance(base, OverlayEdgeMap) else {}
        for change in changes:
            key = (change.src, change.dst)
            if self.base.get(key) == change.final_weight:
                net.pop(key, None)
            else:
                net[key] = change.final_weight
        self._net = net
        self.net = MappingProxyType(net)
        self.count = len(self.base) + sum(
            int(value is not None) - int(key in self.base) for key, value in net.items())

    def __getitem__(self, key):
        if key in self.net:
            value = self.net[key]
            if value is None:
                raise KeyError(key)
            return value
        return self.base[key]

    def __iter__(self):
        for key in self.base:
            if self.net.get(key, True) is not None:
                yield key
        for key, value in self.net.items():
            if value is not None and key not in self.base:
                yield key

    def __len__(self):
        return self.count

    def to_dict(self):
        result = self.base.to_dict()
        for key, value in self.net.items():
            if value is None:
                result.pop(key, None)
            else:
                result[key] = value
        return result

    def to_arrays(self):
        pairs, original = self.base.to_arrays()
        values = original.copy()
        keep = np.ones(len(pairs), dtype=bool)
        appended = []
        for key, value in self.net.items():
            try:
                pos = self.base.position(key)
            except KeyError:
                if value is not None:
                    appended.append((*key, value))
            else:
                if value is None:
                    keep[pos] = False
                else:
                    values[pos] = value
        pairs, values = pairs[keep], values[keep]
        if appended:
            pairs = np.concatenate((pairs, np.asarray([row[:2] for row in appended], np.int64)))
            values = np.concatenate((values, np.asarray([row[2] for row in appended], np.float64)))
            order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            pairs, values = pairs[order], values[order]
        return pairs, values


def array_materializer(ref, nodes, edges, previous, changes, version):
    if previous is not None and previous.ref == ref and all(c.final_weight is not None for c in changes):
        return default_materializer(ref, nodes, edges, previous, changes, version)
    from easygraph.gpu._arena.round2_topology.gpu_adapter import PreparedState
    pairs, values = edges.to_arrays()
    state = PreparedState(np.asarray(nodes, np.int64), pairs, values)
    state.sums = np.asarray(state.sums, dtype=np.float64)
    return state


class ArrayJournalStore(JournalStore):
    def __init__(self, journal_budget_bytes, materializer=array_materializer):
        super().__init__(journal_budget_bytes, materializer)

    @staticmethod
    def _input(nodes, edges, weights):
        nodes = tuple(_integer(n) for n in nodes)
        if tuple(sorted(set(nodes))) != nodes:
            raise ValueError('Sorted unique node labels required')
        node_array = np.asarray(nodes, np.int64)
        if not isinstance(edges, np.ndarray):
            old_nodes, allowed, base = JournalStore._input(nodes, edges, weights)
            pairs = np.asarray(sorted(base), np.int64).reshape(-1, 2)
            return old_nodes, allowed, ArrayEdgeMap(pairs, [base[tuple(p)] for p in pairs])
        raw = edges
        if raw.ndim and len(raw) == 0:
            raw = np.empty((0, 2), np.int64)
        if raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype.kind not in 'iu':
            # Reuse exact scalar validation for heterogeneous Python inputs.
            old_nodes, allowed, base = JournalStore._input(nodes, raw.tolist(), weights)
            pairs = np.asarray(sorted(base), np.int64).reshape(-1, 2)
            return old_nodes, allowed, ArrayEdgeMap(pairs, [base[tuple(p)] for p in pairs])
        if raw.dtype.kind == 'u' and raw.size and int(raw.max()) >= (1 << 63):
            raise ValueError('Node label exceeds int64')
        pairs = raw.astype(np.int64, copy=True)
        mapped = np.searchsorted(node_array, pairs)
        if pairs.size and (not len(nodes) or np.any(mapped == len(nodes)) or np.any(node_array[mapped] != pairs)):
            raise ValueError('Endpoint outside generation node domain')
        if weights is None:
            values = np.ones(len(pairs), np.float64)
        else:
            original = weights if isinstance(weights, np.ndarray) else list(weights)
            # Original scalar contract accepts values convertible with float().
            if np.iscomplexobj(original):
                raise ValueError('Finite nonnegative weight required')
            try:
                values = np.asarray(original, np.float64)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError('Finite nonnegative weight required') from error
        if values.ndim != 1 or len(values) != len(pairs):
            raise ValueError('Initial edge/weight length mismatch')
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError('Finite nonnegative weight required')
        if len(pairs):
            order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            pairs, values = pairs[order], values[order]
            first = np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1)]
            if not first.all():
                grouped = np.zeros(int(first.sum()), np.float64)
                with np.errstate(over='ignore'):
                    np.add.at(grouped, np.cumsum(first) - 1, values)
                if not np.all(np.isfinite(grouped)):
                    raise ValueError('Finite nonnegative weight required')
                pairs, values = pairs[first], grouped
        return nodes, frozenset(nodes), ArrayEdgeMap(pairs, values)

    def _logical(self, record, journal=None):
        changes = _changes(journal)
        return OverlayEdgeMap(record.base_edges, changes) if changes else record.base_edges

    def edges(self, ref):
        with self._lock:
            record = self._record(ref)
            # Match the original full, owned mapping result and its lifetime.
            return MappingProxyType(self._logical(record, self._journals.get(ref)).to_dict())

    def _build(self, ref, nodes, edges, previous, changes, version):
        started = time.perf_counter()
        self._counters['materializer_attempts'] += 1
        try:
            state = self.materializer(ref, nodes, edges, previous, changes, version)
        finally:
            self._counters['materializer_seconds'] += time.perf_counter() - started
        return FlushResult(ref, previous, Snapshot(ref, version, state), changes,
                           previous is not None and state is previous.host_state)

    def authority_bytes(self):
        seen = set()
        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            result = sys.getsizeof(value)
            if isinstance(value, np.ndarray):
                if value.base is not None:
                    result += size(value.base)
            elif isinstance(value, ArrayEdgeMap):
                result += sys.getsizeof(value.__dict__) + size(value.pairs) + size(value.values_array) + size(value._keys)
            elif isinstance(value, OverlayEdgeMap):
                result += sys.getsizeof(value.__dict__) + size(value.base) + sys.getsizeof(value.net) + size(value._net)
            elif isinstance(value, Mapping):
                result += sum(size(k) + size(v) for k, v in value.items())
            elif isinstance(value, tuple):
                result += sum(map(size, value))
            return result
        return sum(size(record.base_edges) for record in self._records.values())

    def stats(self):
        return dict(super().stats(), authority_storage_bytes=self.authority_bytes(),
                    authority_format='sorted endpoint/value arrays plus immutable coalesced overlay')
