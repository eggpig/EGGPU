"""Reuse verified input ordering and identity labels during cold preparation.

The fast paths are input properties, not graph identities. They are checked
inside the same timed register call. Generic inputs retain stable duplicate
summation and signed-label mapping. The qualified ColdPreparation certificate
and its one-call lifetime are unchanged.
"""
from time import perf_counter
import weakref
import numpy as np
from easygraph.gpu._arena.round8_cold.authority import ColdStore, ColdPreparation
from easygraph.gpu._arena.round7_recovery.host_authority_v2 import IndexedEdgeMap


def lexicographic_unique(pairs):
    """Linear proof, with an early prefix rejection for unsorted streams."""
    if len(pairs) < 2:
        return True
    def valid(left, right):
        return np.all((right[:, 0] > left[:, 0]) |
                      ((right[:, 0] == left[:, 0]) & (right[:, 1] > left[:, 1])))
    prefix = min(len(pairs), 65)
    if not valid(pairs[:prefix-1], pairs[1:prefix]):
        return False
    return prefix == len(pairs) or bool(valid(pairs[prefix-1:-1], pairs[prefix:]))


class FastColdStore(ColdStore):
    def _input(self, nodes, edges, weights):
        typed = (self._cold_frames and isinstance(nodes, np.ndarray) and nodes.ndim == 1 and
                 nodes.dtype.kind in 'iu' and isinstance(edges, np.ndarray) and
                 edges.ndim == 2 and edges.shape[1] == 2 and edges.dtype.kind in 'iu')
        if not typed:
            return super()._input(nodes, edges, weights)
        started = perf_counter()
        try:
            if ((nodes.dtype.kind == 'u' and nodes.size and int(nodes.max()) >= 1 << 63) or
                (edges.dtype.kind == 'u' and edges.size and int(edges.max()) >= 1 << 63)):
                raise ValueError('Node label exceeds int64')
            node_array = nodes.astype(np.int64, copy=True)
            proof_started = perf_counter()
            if len(node_array) > 1 and not np.all(node_array[1:] > node_array[:-1]):
                raise ValueError('Sorted unique node labels required')
            if len(node_array) >= 1 << 31:
                raise ValueError('Node positions must fit signed int32')
            # n strictly increasing integers with endpoints 0 and n-1 are
            # exactly the dense identity domain; no second arange array needed.
            identity = (not len(node_array) or
                        (node_array[0] == 0 and node_array[-1] == len(node_array)-1))
            self.cold_count['node_property_validation_seconds'] += perf_counter()-proof_started
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
            self.cold_count['edge_property_validation_seconds'] += perf_counter()-proof_started
            if canonical:
                # Sorted input is still caller-owned; the authority takes its
                # own weight array instead of relying on sort's advanced copy.
                values = values.copy()
                self.cold_count['verified_canonical_inputs'] += 1
            elif len(pairs):
                sort_started = perf_counter()
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
                self.cold_count['stable_sort_calls'] += 1
                self.cold_count['stable_sort_seconds'] += perf_counter()-sort_started
            if len(pairs) >= 1 << 31:
                raise ValueError('Edge positions must fit signed int32')
            mapping_started = perf_counter()
            if identity:
                if pairs.size and (np.any(pairs < 0) or np.any(pairs >= len(node_array))):
                    raise ValueError('Endpoint outside generation node domain')
                mapped = pairs.astype(np.int32)
                self.cold_count['verified_identity_domains'] += 1
            else:
                mapped = np.searchsorted(node_array, pairs)
                if pairs.size and (not len(node_array) or np.any(mapped >= len(node_array)) or
                                   np.any(node_array[mapped] != pairs)):
                    raise ValueError('Endpoint outside generation node domain')
                mapped = mapped.astype(np.int32)
                self.cold_count['endpoint_searchsorted_calls'] += 1
            self.cold_count['endpoint_mapping_seconds'] += perf_counter()-mapping_started
            mapped.flags.writeable = False
            base = IndexedEdgeMap(pairs, values)
            self._cold_frames[-1] = ColdPreparation(weakref.ref(base), labels, node_array, mapped)
            self.cold_count['typed_input_calls'] += 1
            self.cold_count['endpoint_mapping_passes'] += 1
            self.cold_count['raw_edges_validated'] += len(edges)
            self.cold_count['canonical_edges_mapped'] += len(pairs)
            self._cold_observe()
            return labels, frozenset(labels), base
        finally:
            self.cold_count['typed_input_seconds'] += perf_counter()-started
