"""Private producer ownership transfer with the final demand-state semantics.

Ordinary caller-owned CSRInput remains copied/validated by DemandCSRStore. The
private producer alone can hand over its newly allocated, validated immutable
CSR arrays once. Every producer copy and validation is charged by the replay.
"""
from functools import partial
from pathlib import Path
from time import perf_counter
import sys
import weakref
import numpy as np

from easygraph.gpu._arena.canonical_csr_v4 import prepare
from easygraph.gpu._arena.input_hotpath.csr_session_v4 import (
    DemandCSRStore, DemandCSRSession, DemandPreparation, DemandState,
    CSRIndexedAuthority, SortedLabels, SortedNodeMapping, SortedNodeSet,
    ImmutableNodeDomain, Direction, DemandGraphAdapter, GraphAdapter,
    StartupTransferKernels)
from easygraph.gpu._arena.round8_lazy import session as layout
from easygraph.gpu._arena.weighted_host_v1 import SegmentedDemandState

_PRODUCER = object()


class _ProducedCSR:
    """Internal single-consumer capability, never accepted from public arrays."""
    __slots__ = ('_arrays',)

    def __init__(self, seal, arrays):
        if seal is not _PRODUCER:
            raise TypeError('Only the private producer can issue this capability')
        self._arrays = arrays

    def _consume(self):
        if self._arrays is None:
            raise RuntimeError('CSR ownership has already been transferred')
        arrays = self._arrays
        self._arrays = None
        return arrays


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
    owned_nodes = nodes.astype(np.int64, copy=True)
    owned_weights = np.asarray(weights, np.float64).copy()
    if owned_weights.ndim != 1 or len(owned_weights) != len(edges):
        raise ValueError('One weight per edge required')
    if not np.isfinite(owned_weights).all() or np.any(owned_weights < 0):
        raise ValueError('Finite nonnegative weights required')
    ownership_prefix = perf_counter() - started
    if not len(nodes):
        if len(edges):
            raise ValueError('Edges outside empty node domain')
        row, col = np.zeros(1, np.int32), np.empty(0, np.int32)
        values = owned_weights
        profile = dict(mapping='empty_domain', input_sorted_unique=True,
                       sorting_performed=False, stages={}, total_seconds=0.)
    else:
        arrays, _, profile = prepare(owned_nodes, edges, owned_weights, 'eggpu')
        row, col = arrays['offsets'].view(np.int32), arrays['indices'].view(np.int32)
        values = arrays['weights']
        # Mark both owning arrays and their dtype views readonly.
        arrays['offsets'].flags.writeable = False
        arrays['indices'].flags.writeable = False
    for array in (owned_nodes, owned_weights, row, col, values):
        array.flags.writeable = False
    certificate = _ProducedCSR(_PRODUCER, (owned_nodes, row, col, values))
    profile = dict(profile, ownership_validation_copy_seconds=ownership_prefix,
                   producer_total_seconds=perf_counter() - started,
                   transfer_payload_bytes=row.nbytes + col.nbytes,
                   contract='Private newly produced canonical CSR ownership; caller nodes/weights copied, endpoints validated, every cost charged.')
    return owned_nodes, certificate, profile


def prepare_owned(data):
    """Common loaded COO -> private prepared state; all costs belong to caller."""
    prepared, profiles = {}, {}
    for name, item in data.items():
        nodes, certificate, profile = _produce(item['nodes'], item['edges'], item['weights'])
        prepared[name] = dict(nodes=nodes, edges=certificate, weights=None)
        profiles[name] = profile
    return prepared, profiles


class OwnedCSRStore(DemandCSRStore):
    def _input(self, nodes, edges, weights):
        if not isinstance(edges, _ProducedCSR):
            result = super()._input(nodes, edges, weights)
            frame = self._cold_frames[-1]
            if isinstance(frame, DemandPreparation) and type(frame.state) is DemandState:
                frame.state.__class__ = SegmentedDemandState
            return result
        if weights is not None:
            raise ValueError('A private CSR capability carries its own weights')
        started = perf_counter()
        try:
            owned_nodes, row, col, values = edges._consume()
            if nodes is not None and nodes is not owned_nodes:
                raise ValueError('Private producer node identity mismatch')
            n, m = len(owned_nodes), len(col)
            base = CSRIndexedAuthority(owned_nodes, row, col, values)
            labels = SortedLabels(owned_nodes)
            mapping = SortedNodeMapping(labels)
            domain = ImmutableNodeDomain.__new__(ImmutableNodeDomain)
            domain.nodes, domain.n, domain.node_to_id = labels, n, mapping
            domain.source_mapping_identity = id(mapping)
            domain.registration_seconds = 0.
            state = SegmentedDemandState.__new__(SegmentedDemandState)
            state.nodes, state.domain = owned_nodes, domain
            state.m, state.capacity, state.values = m, m, values
            state.outgoing = Direction(row, col, np.arange(m, dtype=np.int32))
            state.graph, state.version, state.readers = None, 0, 0
            state.host_view_seconds = {}
            state.initialization_seconds = perf_counter() - started
            self._cold_frames[-1] = DemandPreparation(weakref.ref(base), labels, owned_nodes, state)
            self.cold_count['private_owned_csr_admissions'] += 1
            self.cold_count['private_owned_csr_transferred_payload_bytes'] += row.nbytes + col.nbytes
            self.cold_count['csr_outgoing_rebuilds_avoided'] += 1
            self.cold_count['csr_initial_authority_pair_bytes_avoided'] += m * 16
            self._cold_observe()
            return labels, SortedNodeSet(mapping), base
        finally:
            self.cold_count['private_owned_csr_admission_seconds'] += perf_counter() - started


class OwnedCSRSession(DemandCSRSession):
    """Final session constructor with only its host-store factory substituted."""
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
        factory = partial(OwnedCSRStore, change_budget_bytes=change_budget_bytes,
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


from easygraph.gpu._arena.trace_backends import EGGPUAdapter


class OwnedCandidateAdapter(EGGPUAdapter):
    def __init__(self, data, *, budget_bytes, **kwargs):
        from easygraph.gpu._arena.kernel_sssp_v4 import SparseCooperativeSSSP
        from easygraph.gpu._arena.public_session_release import PublicExecutor
        from easygraph.gpu._arena.connectivity import connectivity_session, ConnectivityExecutor
        from easygraph.gpu._arena.bulk_assign import assignment_session
        from easygraph.gpu._arena.kernel_pr_v4 import HybridRowsExecutor
        self.nodes = {k: v['nodes'] for k, v in data.items()}
        cls = assignment_session(connectivity_session(OwnedCSRSession))
        inputs = [dict(name=k, nodes=v['nodes'], edges=v['edges'], weights=v['weights']) for k, v in data.items()]
        self.session = cls(inputs, budget_bytes, 4 << 20, 'ordinary', None, host_producer='shared',
                           host_index_budget_bytes=0, change_budget_bytes=8 << 20)
        self.session.executor = ConnectivityExecutor(HybridRowsExecutor(PublicExecutor(self.session.executor.base)))
        self.session.native_sssp = SparseCooperativeSSSP(self.session.native_sssp)
        self.candidate, self.last = True, {}
