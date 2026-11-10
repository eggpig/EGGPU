"""V2: explicit scratch release even while a caller retains an error traceback.

Sorting/mapping semantics and budgets are unchanged from qualified v1. Helpers
clear their owned scratch references in finally, and the identity mapping path
joins the explicit construction-payload ledger. Input/certificate ownership is
still governed by the frozen ColdStore registration frame.
"""
from time import perf_counter
import weakref
import numpy as np
from easygraph.gpu._arena.round8_cold.authority import ColdPreparation
from easygraph.gpu._arena.round8_cold_fast.authority import lexicographic_unique
from easygraph.gpu._arena.round7_recovery.host_authority_v2 import IndexedEdgeMap
from easygraph.gpu._arena.round8_bounded_input.authority import BoundedInputStore

class BoundedInputStoreV2(BoundedInputStore):

    def _sort_input(self, pairs):
        keys = columns = order = None
        try:
            if self.packed_input_sort and pairs.size and (int(pairs.min()) >= 0) and (int(pairs.max()) < 1 << 32):
                keys = pairs[:, 0].astype(np.uint64)
                self._temporary('sort', keys)
                keys <<= np.uint64(32)
                columns = pairs[:, 1].astype(np.uint64)
                self._temporary('sort', keys, columns)
                keys |= columns
                del columns
                order = np.argsort(keys, kind='stable')
                self._temporary('sort', keys, order)
                self.cold_count['packed_stable_sort_calls'] += 1
                del keys
                self._temporary('sort', order)
                return order
            self.cold_count['lexicographic_fallback_sort_calls'] += 1
            order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            self._temporary('sort', order)
            return order
        finally:
            keys = columns = order = None
            self._temporary('sort')

    def _map_input(self, nodes, pairs):
        table = mapped = positions = values = result = None
        try:
            if not len(pairs):
                self.cold_count['empty_endpoint_mapping_calls'] += 1
                return np.empty((0, 2), np.int32)
            if len(nodes):
                lo, hi = (int(nodes[0]), int(nodes[-1]))
                span = hi - lo + 1
                eligible = 0 <= lo <= hi < 1 << 32 and span <= 8 * len(nodes)
                if eligible and span * 4 <= self.input_lut_budget_bytes:
                    if pairs.size and (int(pairs.min()) < lo or int(pairs.max()) > hi):
                        raise ValueError('Endpoint outside generation node domain')
                    table = np.full(span, -1, np.int32)
                    self.cold_count['lookup_table_live_bytes'] = table.nbytes
                    self.cold_count['lookup_table_peak_bytes'] = max(table.nbytes, self.cold_count['lookup_table_peak_bytes'])
                    self._temporary('mapping', table)
                    positions = nodes - lo
                    values = np.arange(len(nodes), dtype=np.int32)
                    self._temporary('mapping', table, positions, values)
                    table[positions] = values
                    del positions, values
                    mapped = np.empty(pairs.shape, np.int32)
                    self._temporary('mapping', table, mapped)
                    for column in (0, 1):
                        positions = pairs[:, column] - lo
                        self._temporary('mapping', table, mapped, positions)
                        values = table[positions]
                        self._temporary('mapping', table, mapped, positions, values)
                        if np.any(values < 0):
                            raise ValueError('Endpoint outside generation node domain')
                        mapped[:, column] = values
                        del positions, values
                    del table
                    self.cold_count['lookup_table_live_bytes'] = 0
                    self._temporary('mapping')
                    self.cold_count['bounded_lookup_mapping_calls'] += 1
                    return mapped
                self.cold_count['lookup_budget_or_span_fallbacks'] += 1
            mapped = np.searchsorted(nodes, pairs)
            self._temporary('mapping', mapped)
            if pairs.size and (not len(nodes) or np.any(mapped >= len(nodes)) or np.any(nodes[mapped] != pairs)):
                raise ValueError('Endpoint outside generation node domain')
            result = mapped.astype(np.int32)
            self._temporary('mapping', mapped, result)
            self.cold_count['endpoint_searchsorted_calls'] += 1
            self._temporary('mapping')
            return result
        finally:
            table = mapped = positions = values = result = None
            self.cold_count['lookup_table_live_bytes'] = 0
            self._temporary('mapping')

    def _input(self, nodes, edges, weights):
        typed = self._cold_frames and isinstance(nodes, np.ndarray) and (nodes.ndim == 1) and (nodes.dtype.kind in 'iu') and isinstance(edges, np.ndarray) and (edges.ndim == 2) and (edges.shape[1] == 2) and (edges.dtype.kind in 'iu')
        if not typed:
            return super()._input(nodes, edges, weights)
        started = perf_counter()
        try:
            if nodes.dtype.kind == 'u' and nodes.size and (int(nodes.max()) >= 1 << 63) or (edges.dtype.kind == 'u' and edges.size and (int(edges.max()) >= 1 << 63)):
                raise ValueError('Node label exceeds int64')
            node_array = nodes.astype(np.int64, copy=True)
            proof_started = perf_counter()
            if len(node_array) > 1 and (not np.all(node_array[1:] > node_array[:-1])):
                raise ValueError('Sorted unique node labels required')
            if len(node_array) >= 1 << 31:
                raise ValueError('Node positions must fit signed int32')
            identity = not len(node_array) or (node_array[0] == 0 and node_array[-1] == len(node_array) - 1)
            self.cold_count['node_property_validation_seconds'] += perf_counter() - proof_started
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
            proof_started = perf_counter()
            canonical = lexicographic_unique(pairs)
            self.cold_count['edge_property_validation_seconds'] += perf_counter() - proof_started
            if canonical:
                values = values.copy()
                self.cold_count['verified_canonical_inputs'] += 1
            elif len(pairs):
                sort_started = perf_counter()
                order = self._sort_input(pairs)
                pairs, values = (pairs[order], values[order])
                del order
                self._temporary('sort')
                first = np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1)]
                if not first.all():
                    grouped = np.zeros(int(first.sum()), np.float64)
                    with np.errstate(over='ignore'):
                        np.add.at(grouped, np.cumsum(first) - 1, values)
                    if not np.all(np.isfinite(grouped)):
                        raise ValueError('Finite nonnegative weight required')
                    pairs, values = (pairs[first], grouped)
                self.cold_count['stable_sort_calls'] += 1
                self.cold_count['stable_sort_seconds'] += perf_counter() - sort_started
            if len(pairs) >= 1 << 31:
                raise ValueError('Edge positions must fit signed int32')
            mapping_started = perf_counter()
            if identity:
                if pairs.size and (np.any(pairs < 0) or np.any(pairs >= len(node_array))):
                    raise ValueError('Endpoint outside generation node domain')
                mapped = pairs.astype(np.int32)
                self._temporary('mapping', mapped)
                self.cold_count['verified_identity_domains'] += 1
            else:
                mapped = self._map_input(node_array, pairs)
            self.cold_count['endpoint_mapping_seconds'] += perf_counter() - mapping_started
            mapped.flags.writeable = False
            base = IndexedEdgeMap(pairs, values)
            self._cold_frames[-1] = ColdPreparation(weakref.ref(base), labels, node_array, mapped)
            self.cold_count['typed_input_calls'] += 1
            self.cold_count['endpoint_mapping_passes'] += 1
            self.cold_count['raw_edges_validated'] += len(edges)
            self.cold_count['canonical_edges_mapped'] += len(pairs)
            self._cold_observe()
            return (labels, frozenset(labels), base)
        finally:
            self._temporary('input')
            self.cold_count['lookup_table_live_bytes'] = 0
            self.cold_count['typed_input_seconds'] += perf_counter() - started
