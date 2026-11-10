"""Single-pass typed initial-input preparation for the frozen R8 authority.

Input validation, stable duplicate accumulation and dense endpoint mapping are
paid inside register(). A certificate belongs to this registration frame and
this authority object only. Its mapping is consumed by the first CSR build;
deferred registration and exceptions discard it before returning to the caller.
Updates, recovery policy, retained index budgets and general Python inputs keep
the existing R8 implementations.
"""
from collections import Counter
from dataclasses import dataclass
from time import perf_counter
import weakref
import numpy as np

from easygraph.gpu._arena.round8_host.host_authority import Store
from easygraph.gpu._arena.round7_recovery.host_authority_v2 import IndexedEdgeMap
from easygraph.gpu._arena.round2_topology.gpu_adapter import PreparedState
from easygraph.gpu._arena.round2_topology.state import Direction
from easygraph.gpu._arena.retained_node_map import ImmutableNodeDomain


@dataclass
class ColdPreparation:
    authority: object
    labels: tuple
    nodes: np.ndarray
    mapped: np.ndarray
    consumed: bool = False

    def matches(self, labels, authority, previous, changes, version):
        return (not self.consumed and self.labels is labels and
                self.authority() is authority and previous is None and
                not changes and version == 0)

    @property
    def payload_bytes(self):
        return sum(x.nbytes for x in (self.nodes, self.mapped) if x is not None)

    @property
    def mapping_bytes(self):
        return 0 if self.mapped is None else self.mapped.nbytes

    def release(self):
        self.nodes = self.mapped = None
        self.labels = ()


def prepared_cold_state(preparation, edges):
    """Build the same R7 cold arrays using a checked dense endpoint mapping."""
    started = perf_counter()
    nodes, mapped = preparation.nodes, preparation.mapped
    src, dst = mapped[:, 0], mapped[:, 1]
    weights = edges.values_array
    n, m = len(nodes), len(mapped)
    out_count = np.bincount(src, minlength=n)
    in_count = np.bincount(dst, minlength=n)
    out_row = np.r_[0, np.cumsum(out_count)].astype(np.int32)
    in_row = np.r_[0, np.cumsum(in_count)].astype(np.int32)
    incoming_order = np.argsort(dst, kind='stable').astype(np.int32)
    state = PreparedState.__new__(PreparedState)
    state.nodes = nodes
    state.domain = ImmutableNodeDomain(preparation.labels,
                                      {label: i for i, label in enumerate(preparation.labels)})
    state.m = m
    state.capacity = max(1, 2*m)
    state.values = np.empty(state.capacity, np.float64)
    state.values[:m] = weights
    state.outgoing = Direction(out_row, dst.copy(), np.arange(m, dtype=np.int32))
    state.incoming = Direction(in_row, src[incoming_order], incoming_order)
    state.in_weight = weights[incoming_order].copy()
    state.degree = out_count.astype(np.float64)
    state.sums = np.bincount(src, weights=weights, minlength=n).astype(np.float64)
    state.graph = None
    state.version = 0
    state.readers = 0
    state.initialization_seconds = perf_counter()-started
    return state


class ColdStore(Store):
    def __init__(self, *args, **kwargs):
        self.cold_count = Counter()
        self._cold_frames = []
        self.cold_last_identity = None
        super().__init__(*args, **kwargs)

    def _cold_observe(self):
        live = sum(p.payload_bytes for p in self._cold_frames if p is not None)
        mapping = sum(p.mapping_bytes for p in self._cold_frames if p is not None)
        self.cold_count['preparation_payload_live_bytes'] = live
        self.cold_count['auxiliary_mapping_live_bytes'] = mapping
        self.cold_count['preparation_payload_peak_bytes'] = max(
            live, self.cold_count['preparation_payload_peak_bytes'])
        self.cold_count['auxiliary_mapping_peak_bytes'] = max(
            mapping, self.cold_count['auxiliary_mapping_peak_bytes'])

    def _register(self, graph_id, nodes, edges, weights, materialize):
        started = perf_counter()
        self._cold_frames.append(None)
        try:
            result = super()._register(graph_id, nodes, edges, weights, materialize)
            self.cold_count['successful_registrations'] += 1
            return result
        finally:
            preparation = self._cold_frames.pop()
            if preparation is not None:
                if not preparation.consumed:
                    self.cold_count['discarded_unconsumed_preparations'] += 1
                preparation.release()
            self._cold_observe()
            self.cold_count['raw_registration_seconds'] += perf_counter()-started

    def _input(self, nodes, edges, weights):
        typed = (self._cold_frames and isinstance(nodes, np.ndarray) and
                 nodes.ndim == 1 and nodes.dtype.kind in 'iu' and
                 isinstance(edges, np.ndarray) and edges.ndim == 2 and
                 edges.shape[1] == 2 and edges.dtype.kind in 'iu')
        if not typed:
            self.cold_count['fallback_input_calls'] += 1
            return super()._input(nodes, edges, weights)
        started = perf_counter()
        try:
            if ((nodes.dtype.kind == 'u' and nodes.size and int(nodes.max()) >= 1 << 63) or
                (edges.dtype.kind == 'u' and edges.size and int(edges.max()) >= 1 << 63)):
                raise ValueError('Node label exceeds int64')
            node_array = nodes.astype(np.int64, copy=True)
            if len(node_array) > 1 and not np.all(node_array[1:] > node_array[:-1]):
                raise ValueError('Sorted unique node labels required')
            if len(node_array) >= 1 << 31:
                raise ValueError('Node positions must fit signed int32')
            labels = tuple(node_array.tolist())
            pairs = edges.astype(np.int64, copy=True)
            if weights is None:
                values = np.ones(len(pairs), np.float64)
            else:
                original = weights if isinstance(weights, np.ndarray) else list(weights)
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
                # lexsort is stable: each duplicate group retains the input's
                # floating-addition order, including zero-weight edge presence.
                order = np.lexsort((pairs[:, 1], pairs[:, 0]))
                pairs, values = pairs[order], values[order]
                first = np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1)]
                if not first.all():
                    grouped = np.zeros(int(first.sum()), np.float64)
                    with np.errstate(over='ignore'):
                        np.add.at(grouped, np.cumsum(first)-1, values)
                    if not np.all(np.isfinite(grouped)):
                        raise ValueError('Finite nonnegative weight required')
                    pairs, values = pairs[first], grouped
            else:
                # The authority must never retain an alias of a caller weight
                # array, including a zero-length one.
                values = values.copy()
            if len(pairs) >= 1 << 31:
                raise ValueError('Edge positions must fit signed int32')
            mapped = np.searchsorted(node_array, pairs)
            if pairs.size and (not len(node_array) or np.any(mapped >= len(node_array)) or
                               np.any(node_array[mapped] != pairs)):
                raise ValueError('Endpoint outside generation node domain')
            mapped = mapped.astype(np.int32)
            mapped.flags.writeable = False
            base = IndexedEdgeMap(pairs, values)
            preparation = ColdPreparation(weakref.ref(base), labels, node_array, mapped)
            self._cold_frames[-1] = preparation
            self.cold_count['typed_input_calls'] += 1
            self.cold_count['endpoint_mapping_passes'] += 1
            self.cold_count['raw_edges_validated'] += len(edges)
            self.cold_count['canonical_edges_mapped'] += len(pairs)
            self._cold_observe()
            return labels, frozenset(labels), base
        finally:
            self.cold_count['typed_input_seconds'] += perf_counter()-started

    def _materialize(self, ref, nodes, edges, previous, changes, version):
        preparation = self._cold_frames[-1] if self._cold_frames else None
        if preparation is None or not preparation.matches(nodes, edges, previous, changes, version):
            self.cold_count['fallback_materializer_calls'] += 1
            return super()._materialize(ref, nodes, edges, previous, changes, version)
        self._remove_certificate(ref)
        started = perf_counter()
        try:
            state = prepared_cold_state(preparation, edges)
            state._r8_authority = weakref.ref(edges)
            self.cold_last_identity = dict(authority=id(edges), labels=id(nodes),
                prepared_nodes=id(preparation.nodes), published_nodes=id(state.nodes),
                shared_domain_labels=state.domain.nodes is nodes, consumed_once=True)
            preparation.consumed = True
            self.cold_count['consumed_preparations'] += 1
            return state
        finally:
            preparation.release()
            self._cold_observe()
            self.cold_count['prepared_materializer_seconds'] += perf_counter()-started

    def stats(self):
        return dict(super().stats(), cold_preparation=dict(self.cold_count,
            last_identity=self.cold_last_identity,
            scope='Preparation payload counts owned nodes plus the temporary dense endpoint mapping; '
                  'mapping is released before register returns. CSR/authority arrays, construction '
                  'temporaries, Python metadata and RSS are separate. All work is inside raw_registration_seconds.'))
