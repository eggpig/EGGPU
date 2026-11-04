"""CSR authority with compact node domains and demand-built host views.

Canonical outgoing CSR is a complete immutable initial authority. Original-ID
edge pairs are expanded only when the inherited mutation machinery needs them.
Incoming adjacency, reordered weights and source sums are constructed on their
first request. Updates complete the old host snapshot before invoking the
unchanged versioned transition machinery. No caller buffers are adopted.
"""
from collections import defaultdict
from collections.abc import Mapping, Sequence, Set
from functools import partial
from time import perf_counter
import sys
import weakref

import numpy as np
from scipy.sparse import csr_matrix

from easygraph.gpu._arena.input_hotpath.csr_session import CSRInput
from easygraph.gpu._arena.input_hotpath.csr_session_v3 import LinearCSRStore, LinearCSRSession, CSRPreparation
from easygraph.gpu._arena.round2_topology.gpu_adapter import PreparedState, GraphAdapter, ArrayShape
from easygraph.gpu._arena.round2_topology.state import Direction, unique_array_bytes
from easygraph.gpu._arena.round7_recovery.host_authority_v2 import IndexedEdgeMap, encode_keys
from easygraph.gpu._arena.round5_input.array_journal_v2 import ArrayEdgeMap, OverlayEdgeMap
from easygraph.gpu._arena.retained_node_map import ImmutableNodeDomain
from easygraph.gpu._arena.round8_startup.session import StartupTransferKernels
from easygraph.gpu._arena.round8_lazy import session as layout


class SortedLabels(Sequence):
    """Immutable sorted int64 sequence, without Python objects per node."""
    def __init__(self, owned):
        assert owned.ndim == 1 and owned.dtype == np.int64 and not owned.flags.writeable
        self.array = owned

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        value = self.array[index]
        return tuple(map(int, value)) if isinstance(value, np.ndarray) else int(value)

    def __iter__(self):
        return map(int, self.array)

    def __array__(self, dtype=None, copy=None):
        if copy is True:
            return np.array(self.array, dtype=dtype, copy=True)
        if copy is False and dtype is not None and np.dtype(dtype) != self.array.dtype:
            raise ValueError('Unable to avoid a copy for the requested dtype')
        return np.asarray(self.array, dtype=dtype)


class SortedNodeMapping(Mapping):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return iter(self.labels)

    def __getitem__(self, key):
        hash(key)
        try:
            integer = int(key)
            if integer != key or not -(1 << 63) <= integer < (1 << 63):
                raise KeyError(key)
        except (TypeError, ValueError, OverflowError) as error:
            raise KeyError(key) from error
        index = int(np.searchsorted(self.labels.array, integer))
        if index == len(self.labels) or self.labels.array[index] != integer:
            raise KeyError(key)
        return index


class SortedNodeSet(Set):
    def __init__(self, mapping):
        self.mapping = mapping

    def __len__(self):
        return len(self.mapping)

    def __iter__(self):
        return iter(self.mapping)

    def __contains__(self, key):
        return key in self.mapping


class CSRIndexedAuthority(IndexedEdgeMap):
    """Mapping-compatible authority whose owned physical source is CSR."""
    def __init__(self, nodes, row, col, values):
        self.nodes, self.row, self.col, self.values_array = nodes, row, col, values
        for array in (nodes, row, col, values):
            array.flags.writeable = False
        self.packed = not len(nodes) or (int(nodes[0]) >= 0 and int(nodes[-1]) < 1 << 32)
        self._expanded_pairs = None
        self._expanded_keys = None
        self.expansion_seconds = 0.

    @property
    def pairs(self):
        if self._expanded_pairs is None:
            started = perf_counter()
            src = np.repeat(np.arange(len(self.nodes), dtype=np.int32), np.diff(self.row))
            pairs = np.empty((len(self.col), 2), np.int64)
            pairs[:, 0] = self.nodes[src]
            pairs[:, 1] = self.nodes[self.col]
            pairs.flags.writeable = False
            self._expanded_pairs = pairs
            self.expansion_seconds += perf_counter() - started
        return self._expanded_pairs

    @property
    def _keys(self):
        if self._expanded_keys is None:
            pairs = self.pairs
            started = perf_counter()
            keys = encode_keys(pairs, self.packed)
            keys.flags.writeable = False
            self._expanded_keys = keys
            self.expansion_seconds += perf_counter() - started
        return self._expanded_keys

    def position(self, key):
        # Scalar reads need no full pair/key expansion. The inherited batch
        # lookup and update-fold path can request complete arrays when useful.
        hash(key)
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(key)
        try:
            u, v = map(int, key)
            if (u, v) != key or not all(-(1 << 63) <= x < 1 << 63 for x in (u, v)):
                raise KeyError(key)
        except (TypeError, ValueError, OverflowError) as error:
            raise KeyError(key) from error
        dense = np.searchsorted(self.nodes, (u, v))
        if np.any(dense == len(self.nodes)) or not np.array_equal(self.nodes[dense], (u, v)):
            raise KeyError(key)
        start, end = map(int, self.row[dense[0]:dense[0] + 2])
        location = start + int(np.searchsorted(self.col[start:end], dense[1]))
        if location == end or self.col[location] != dense[1]:
            raise KeyError(key)
        return location

    def retained_arrays(self):
        return [a for a in (self.nodes, self.row, self.col, self.values_array,
                            self._expanded_pairs, self._expanded_keys) if a is not None]


class DemandState(PreparedState):
    def __getattr__(self, name):
        started = perf_counter()
        if name == 'incoming':
            outgoing = self.outgoing
            transposed = csr_matrix((outgoing.ids, outgoing.col, outgoing.row),
                                    shape=(len(self.nodes), len(self.nodes)), copy=False).tocsc()
            value = Direction(transposed.indptr, transposed.indices, transposed.data)
        elif name == 'in_weight':
            incoming_ids = self.incoming.ids
            started = perf_counter()
            value = self.values[incoming_ids]
        elif name == 'degree':
            value = np.diff(self.outgoing.row).astype(np.float64)
        elif name == 'sums':
            sources = np.repeat(np.arange(len(self.nodes), dtype=np.int32), np.diff(self.outgoing.row))
            value = np.bincount(sources, weights=self.values[self.outgoing.ids], minlength=len(self.nodes)).astype(np.float64)
        else:
            raise AttributeError(name)
        self.__dict__[name] = value
        self.host_view_seconds[name] = self.host_view_seconds.get(name, 0.) + perf_counter() - started
        return value

    def complete_for_update(self):
        # PreparedState copies the instance dictionary during a transition.
        # Make that copy fully self-contained before entering existing code.
        for name in ('incoming', 'in_weight', 'degree', 'sums'):
            getattr(self, name)

    def arrays(self):
        arrays = [self.nodes, self.values, self.outgoing.row, self.outgoing.col, self.outgoing.ids]
        for name in ('degree', 'sums', 'in_weight'):
            if name in self.__dict__:
                arrays.append(self.__dict__[name])
        if 'incoming' in self.__dict__:
            incoming = self.__dict__['incoming']
            arrays.extend((incoming.row, incoming.col, incoming.ids))
        return arrays


class DemandPreparation(CSRPreparation):
    @property
    def payload_bytes(self):
        return 0 if self.state is None else self.state.owned_array_bytes()


class DemandCSRStore(LinearCSRStore):
    def _input(self, nodes, edges, weights):
        if not isinstance(edges, CSRInput):
            return super()._input(nodes, edges, weights)
        started = perf_counter()
        try:
            raw_nodes = np.asarray(nodes)
            row, col = np.asarray(edges.row), np.asarray(edges.col)
            if raw_nodes.ndim != 1 or raw_nodes.dtype.kind not in 'iu':
                raise ValueError('Sorted integer node labels required')
            if raw_nodes.dtype.kind == 'u' and raw_nodes.size and int(raw_nodes.max()) >= 1 << 63:
                raise ValueError('Node label exceeds int64')
            owned_nodes = raw_nodes.astype(np.int64, copy=True)
            n, m = len(owned_nodes), len(col)
            if n >= 1 << 31 or np.any(owned_nodes[1:] <= owned_nodes[:-1]):
                raise ValueError('Sorted unique int64 node labels required')
            if row.ndim != 1 or col.ndim != 1 or row.dtype.kind not in 'iu' or col.dtype.kind not in 'iu':
                raise ValueError('Integer CSR row/column arrays required')
            if m >= 1 << 31 or len(row) != n + 1 or int(row[0]) != 0 or int(row[-1]) != m or np.any(row[1:] < row[:-1]):
                raise ValueError('Invalid CSR row offsets')
            if m and (int(col.min()) < 0 or int(col.max()) >= n):
                raise ValueError('CSR destination outside node domain')
            if weights is not None and np.iscomplexobj(weights):
                raise ValueError('Real weights required')
            values = np.ones(m, np.float64) if weights is None else np.asarray(weights, np.float64).copy()
            if values.ndim != 1 or len(values) != m or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError('Finite nonnegative edge weights required')
            row, col = row.astype(np.int32, copy=True), col.astype(np.int32, copy=True)
            invalid = col[1:] <= col[:-1]
            boundaries = row[1:-1]
            boundaries = boundaries[(boundaries > 0) & (boundaries < m)]
            invalid[boundaries - 1] = False
            if invalid.any():
                return super()._input(raw_nodes, edges, weights)
            base = CSRIndexedAuthority(owned_nodes, row, col, values)
            labels = SortedLabels(owned_nodes)
            mapping = SortedNodeMapping(labels)
            domain = ImmutableNodeDomain.__new__(ImmutableNodeDomain)
            domain.nodes, domain.n, domain.node_to_id = labels, n, mapping
            domain.source_mapping_identity = id(mapping)
            domain.registration_seconds = 0.
            state = DemandState.__new__(DemandState)
            state.nodes, state.domain = owned_nodes, domain
            state.m, state.capacity = m, m
            state.values = values
            state.outgoing = Direction(row, col, np.arange(m, dtype=np.int32))
            state.graph, state.version, state.readers = None, 0, 0
            state.host_view_seconds = {}
            state.initialization_seconds = perf_counter() - started
            self._cold_frames[-1] = DemandPreparation(weakref.ref(base), labels, owned_nodes, state)
            self.cold_count['csr_direct_admissions'] += 1
            self.cold_count['csr_outgoing_rebuilds_avoided'] += 1
            self.cold_count['csr_initial_authority_pair_bytes_avoided'] += m * 16
            self._cold_observe()
            return labels, SortedNodeSet(mapping), base
        finally:
            self.cold_count['csr_demand_input_seconds'] += perf_counter() - started

    def _materialize(self, ref, nodes, edges, previous, changes, version):
        if previous is not None and isinstance(previous.host_state, DemandState):
            previous.host_state.complete_for_update()
        return super()._materialize(ref, nodes, edges, previous, changes, version)

    def authority_bytes(self):
        # Inspect existing physical buffers only. A memory observation must not
        # build an otherwise unused endpoint array or a search index.
        seen = set()
        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            result = sys.getsizeof(value)
            if isinstance(value, np.ndarray):
                if value.base is not None:
                    result += size(value.base)
            elif isinstance(value, CSRIndexedAuthority):
                result += sys.getsizeof(value.__dict__) + sum(size(a) for a in value.retained_arrays())
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

    def physical_arrays(self):
        arrays = []
        for record in self._records.values():
            base = record.base_edges
            if isinstance(base, CSRIndexedAuthority):
                arrays.extend(base.retained_arrays())
            elif isinstance(base, ArrayEdgeMap):
                arrays.extend((base.pairs, base.values_array, base._keys))
            if record.snapshot is not None:
                arrays.extend(record.snapshot.host_state.arrays())
        return arrays

    def stats(self):
        result = super().stats()
        result['authority_format'] = 'Canonical CSR authority, expanding original-label endpoint/key arrays on mutation demand; updated generations inherit sorted endpoint authority'
        result['host_authority_state_union_payload_bytes'] = unique_array_bytes(self.physical_arrays())
        result['host_authority_state_union_scope'] = 'Physical NumPy allocation union of current authority and snapshots; shared buffers counted once. Python metadata and external historical snapshots are separate.'
        return result


class DemandGraphAdapter(GraphAdapter):
    def __init__(self, state, name='graph', generation=0, attribute_version=0):
        self.r2_state, self.name, self.n = state, name, len(state.nodes)
        self.nodes, self.node_to_id = state.domain.nodes, state.domain.node_to_id
        self.row, self.col, self.out_row = ArrayShape(self.n + 1), ArrayShape(state.m), state.outgoing.row
        self.attributes = {'message_count': None}
        self.generation = generation
        self.versions = defaultdict(int, {'message_count': attribute_version})
        self.arrays, self.changes, self.change_floors = {}, defaultdict(list), defaultdict(int)
        self._registered_keys, self._active_key_callback = set(), None
        self.host_prepare_seconds = 0.
        self.view_builds = self.repaired_values = self.sum_repair_rows = self.sum_repair_scanned_edges = self.norm_closure_entries = 0
        self.input_graph_class = 'Canonical CSR authority with demand-built host views'

    def get(self, key):
        started = perf_counter()
        self.register_keys([key])
        kind, attr = key[1:]
        if attr not in (None, 'message_count'):
            raise KeyError(attr)
        state = self.r2_state
        if kind == 'row':
            result = state.incoming.row
            self.row = result
        elif kind == 'col':
            result = state.incoming.col
            self.col = result
        elif kind == 'out_row':
            result = state.outgoing.row
        elif kind == 'out_col':
            result = state.outgoing.col
        elif kind == 'weight':
            result = state.in_weight
            self.attributes['message_count'] = result
        elif kind == 'source_sum':
            result = state.degree if attr is None else state.sums
        elif kind == 'out_weight':
            # This adapter is constructed only for the initial CSR whose IDs
            # are contiguous and in outgoing order. Updates create the normal
            # frozen GraphAdapter and its general gather path.
            if isinstance(state, DemandState):
                result = state.values[:state.m]
            else:
                return super().get(key)
        else:
            raise KeyError(key)
        self.host_prepare_seconds += perf_counter() - started
        return result


class DemandCSRSession(LinearCSRSession):
    def __init__(self, inputs, budget_bytes, journal_budget_bytes, strategy, historical,
                 host_index_budget_bytes=0, host_producer='shared', packet_cost='online',
                 capture=False, preselected=None, *, lazy_module=True, change_budget_bytes=8 << 20,
                 input_lut_budget_bytes=8 << 20, packed_input_sort=True):
        if not isinstance(lazy_module, bool):
            raise ValueError('lazy_module must be bool')
        if len({item['name'] for item in inputs}) != len(inputs):
            raise ValueError('Graph identifiers must be unique')
        self.strategy, self.batch_requests, self.batch_index, self.batch_keys = 'ordinary', None, 0, ()
        self.recovery_plans, self.profile, self.historical_profile, self.mode = [], None, historical, 'buffer_scoped'
        factory = partial(DemandCSRStore, change_budget_bytes=change_budget_bytes,
                          input_lut_budget_bytes=input_lut_budget_bytes, packed_input_sort=packed_input_sort)
        self.journal = factory(journal_budget_bytes, host_index_budget_bytes, host_producer)
        self.refs, self.transfer_kernels, self.last_graph, self.residence, self.logical_tick = {}, None, None, {}, 0
        graphs = []
        for item in inputs:
            ref = self.journal.register(item['name'], item['nodes'], item['edges'], item.get('weights'))
            self.refs[item['name']] = ref
            state = self.journal.snapshot(ref).host_state
            graphs.append(DemandGraphAdapter(state, name=item['name']) if isinstance(state, DemandState)
                          else GraphAdapter(state, name=item['name']))
        # Keep the frozen device arena/workspace policy for this isolated host
        # change. The caller's registered budget factory continues to apply.
        layout.initialize_representation(self, graphs, budget_bytes, lazy_module)
        ledger = layout.EpochModel(budget_bytes, 'shared_component_LRU')
        ledger.__dict__.update(self.model.__dict__)
        ledger.epoch_owner = weakref.ref(self)
        self.model = ledger
        self.authority, self.packet_engine, self.packet_cost = 'r8', 'vector', packet_cost
        self.packet_price, self.packet_features, self.packet_prediction = layout.PacketPrice(), None, None
        self.strategy, self.capture_problems, self.problem_records = strategy, capture, []
        self.preselected, self.planner_context = {} if preselected is None else preselected, None
        self.lazy_module, self.fast_input = lazy_module, True
        self.input_lut_budget_bytes, self.packed_input_sort = input_lut_budget_bytes, packed_input_sort
        self.transfer_kernels = StartupTransferKernels('nvrtc', 'demand')

    def stats(self):
        result = super().stats()
        arrays = self.journal.physical_arrays()
        arrays.extend(array for graph in self.graphs.values() for array in graph.arrays.values())
        result['host_authority_state_view_union_payload_bytes'] = unique_array_bytes(arrays)
        result['host_union_scope'] = 'Current authority, host snapshots and graph view arrays, allocation identity deduplicated; metadata, journal and externally retained historical snapshots separately scoped.'
        return result
