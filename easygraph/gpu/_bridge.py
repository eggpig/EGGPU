"""Common complete-host-output contract; no hidden algorithm fallback.

Existing optimized runtimes are injected/imported from read-only frozen sources.
The recovery planner is a separate research prototype, not yet this executor's
controller. Times include common result ownership/normalization. Single-card
dynamic state and native partitioned state retain their actual capabilities.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
import operator
import os
import threading
import copy
import hashlib
import json
import uuid
import weakref
import numpy as np

from .contracts import (NodeDomain, GraphVersion, GraphDescriptor, CapabilityDescriptor,
                          HostOutputState, StateSnapshot, QueryContract, query_contract, managed_inventory)
from ._arena_output import _download_request, _consume_download

_native_runtime_lock = threading.RLock()
_native_runtime_owner = None
# Internal capability: only the fresh native WCC finalizer may transfer ownership.
_NATIVE_WCC_OWNED_OUTPUT = object()
# Independently minted capability for this query's BFS where/finalizer output.
_NATIVE_BFS_OWNED_OUTPUT = object()
# SSSP native workers fill a new NumPy-owned allocation for this request.
_NATIVE_SSSP_OWNED_OUTPUT = object()
# PR canonical scatter fills a fresh request-private NumPy allocation.
_NATIVE_PR_OWNED_OUTPUT = object()
# Arena WCC canonicalization creates a fresh, unaliased host allocation.
_ARENA_WCC_OWNED_OUTPUT = object()


def source_parameter(parameters):
    if set(parameters) != {'source'} or isinstance(parameters.get('source'), (bool, np.bool_)):
        raise ValueError('Require exactly one integer source parameter')
    try:
        return operator.index(parameters['source'])
    except TypeError as exc:
        raise ValueError('Source must be an integer label; no truncation is allowed') from exc


@dataclass(frozen=True)
class QueryResult:
    graph_id: str
    algorithm: str
    values: np.ndarray
    seconds: float
    backend_metrics: dict
    graph: GraphDescriptor | None = None
    node_ids: np.ndarray | None = None
    output_owner_id: str | None = None
    contract: QueryContract | None = None


def owning_result(graph_id, algorithm, values, begin, metrics, *, graph=None, node_ids=None, owner_id=None, contract=None, _private_owned=None, _arena_scope=None):
    # A common, independent immutable host value. The copy is explicitly charged.
    ownership_key = algorithm if algorithm in ('bfs', 'sssp') else 'pr' if algorithm in ('pr', 'weighted_pr') else None
    ownership_begin = perf_counter() if ownership_key is not None else None
    adopted = False
    arena_adopted = False
    if _private_owned is not None:
        if not (((_private_owned is _NATIVE_WCC_OWNED_OUTPUT or _private_owned is _ARENA_WCC_OWNED_OUTPUT) and algorithm == 'wcc')
                or (_private_owned is _NATIVE_BFS_OWNED_OUTPUT and algorithm == 'bfs')
                or (_private_owned is _NATIVE_SSSP_OWNED_OUTPUT and algorithm == 'sssp')
                or (_private_owned is _NATIVE_PR_OWNED_OUTPUT and algorithm in ('pr', 'weighted_pr'))):
            raise RuntimeError('Unrecognized internal output ownership transfer')
        # Internal request-private producer only: the caller
        # created this private array during this query and has never published
        # it or its writable aliases.
        if (type(values) is not np.ndarray or values.dtype != np.dtype(np.float64)
                or values.ndim != 1 or not values.flags.owndata
                or not values.flags.c_contiguous or values.base is not None):
            raise RuntimeError('Private producer did not return an owning float64 array')
        result = values
        adopted = True
    elif _arena_scope is not None and _consume_download(_arena_scope, graph_id, algorithm, values):
        result = values
        adopted = arena_adopted = True
    else:
        result = np.array(values, dtype=np.float64, copy=True)
    if ownership_begin is not None:
        ownership_seconds = perf_counter() - ownership_begin
        metrics[ownership_key+"_public_output_ownership"] = dict(
            route='adopt_arena_download' if arena_adopted else 'adopt_private_numpy' if adopted else 'copy_to_public_numpy',
            bridge_copy_bytes=0 if adopted else int(result.nbytes),
            bridge_copy_bytes_avoided=int(result.nbytes) if adopted else 0,
            ownership_seconds=ownership_seconds,
            bridge_copy_seconds=0.0 if adopted else ownership_seconds,
            accounting_scope="Actual host copy/adoption boundary; bytes are payload, not peak memory")
        publication_begin = perf_counter()
    if graph is not None and result.shape != (graph.node_domain.count,):
        raise RuntimeError('Backend did not return one complete value per registered node')
    result.flags.writeable = False
    answer = QueryResult(graph_id, algorithm, result, perf_counter() - begin, metrics,
                         graph, node_ids, owner_id, contract)
    if ownership_begin is not None:
        metrics[ownership_key+"_public_output_ownership"]["publication_seconds"] = perf_counter() - publication_begin
    return answer


class _Bridge:
    def _initialize_bridge(self, backend):
        self.backend_name = backend
        self.session_id = f'{backend}:{uuid.uuid4().hex}'
        self._domains = {name: NodeDomain.from_nodes(nodes) for name, nodes in self.nodes.items()}
        for nodes in self.nodes.values():
            nodes.flags.writeable = False
        self._outputs = {}
        self._output_sequence = 0

    def _own_result(self, graph_id, algorithm, values, begin, metrics, parameters, *, contract=None, _private_owned=None, _arena_scope=None):
        self._output_sequence += 1
        owner_id = f'{self.session_id}/output/{self._output_sequence}'
        result = owning_result(graph_id, algorithm, values, begin, metrics,
                               graph=self.describe(graph_id), node_ids=self.nodes[graph_id], owner_id=owner_id,
                               contract=contract or query_contract(algorithm, parameters, len(self.nodes[graph_id])),
                               _private_owned=_private_owned, _arena_scope=_arena_scope)
        # A result can be dropped without another snapshot/query. Do not keep dead
        # owner descriptors indefinitely in a long-lived service. The callback
        # references the session weakly and serializes with snapshots/queries.
        ownership_key = algorithm if algorithm in ('bfs', 'sssp') else 'pr' if algorithm in ('pr', 'weighted_pr') else None
        registry_begin = perf_counter() if ownership_key is not None else None
        owner_ref = weakref.ref(self)
        def released(_reference):
            owner = owner_ref()
            if owner is not None:
                with owner._lock:
                    owner._outputs.pop(owner_id, None)
        self._outputs[owner_id] = (weakref.ref(result.values, released), result.graph, algorithm)
        if registry_begin is not None:
            metrics[ownership_key+"_public_output_ownership"]["registry_seconds"] = perf_counter() - registry_begin
        return result

    def _live_outputs(self):
        live, expired = [], []
        for name, (reference, graph, algorithm) in self._outputs.items():
            array = reference()
            if array is None:
                expired.append(name)
            else:
                live.append(HostOutputState(name, graph, algorithm, int(array.nbytes)))
        for name in expired:
            del self._outputs[name]
        return tuple(live)


class ManagedSession(_Bridge):
    capabilities = frozenset({'bfs', 'sssp', 'pr', 'weighted_pr', 'updates', 'managed_budget'})

    def __init__(self, graphs, *, budget_bytes, pr_tolerance_l1=1e-6,
                  pr_max_iter=10000, input_format='coo'):
        if isinstance(pr_tolerance_l1, (bool, np.bool_)):
            raise ValueError('PageRank L1 tolerance must be finite and positive')
        tolerance = float(pr_tolerance_l1)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('PageRank L1 tolerance must be finite and positive')
        if isinstance(pr_max_iter, (bool, np.bool_)):
            raise ValueError('PageRank iteration limit must be a positive integer')
        try:
            iterations = operator.index(pr_max_iter)
        except TypeError as exc:
            raise ValueError('PageRank iteration limit must be a positive integer') from exc
        if iterations <= 0:
            raise ValueError('PageRank iteration limit must be a positive integer')
        self.pr_tolerance_l1, self.pr_max_iter = tolerance, iterations
        if input_format not in ('coo', 'csr'):
            raise ValueError('Arena input_format must be coo or csr')
        if input_format == 'csr':
            from ._csr_input import prepare_csr as prepare_owned
        else:
            from easygraph.gpu._arena.input_hotpath.fused_producer_v8 import prepare_owned
        from easygraph.gpu._arena.input_hotpath.confirm_candidate_v1 import ConfirmCandidateAdapter
        begin = perf_counter()
        prepared, self.producer_profiles = prepare_owned(graphs)
        self.backend = ConfirmCandidateAdapter(prepared, budget_bytes=budget_bytes)
        self.backend.pr_tolerance_l1 = tolerance
        self.backend.pr_max_iter = iterations
        if input_format == 'csr':
            # Copy the already validated owned node snapshot, not a caller array
            # retained by the small CSRInput envelope. Queries must remain bound
            # to registration even after the caller mutates its original labels.
            self.nodes = {name: np.array(self.backend.session.journal.snapshot(
                self.backend.session.refs[name]).host_state.nodes, dtype=np.int64, copy=True)
                for name in prepared}
            self.backend.nodes = self.nodes
        else:
            self.nodes = {name: np.array(data['nodes'], dtype=np.int64, copy=True) for name, data in prepared.items()}
        self._lock = self.backend.session.lock
        self._initialize_bridge('managed')
        self.initialization_seconds = perf_counter() - begin
        self.closed = False

    def query(self, graph_id, algorithm, parameters=None):
        begin = perf_counter()
        with self._lock:
            return self._query(graph_id, algorithm, parameters, begin)

    def _query(self, graph_id, algorithm, parameters=None, begin=None):
        begin = perf_counter() if begin is None else begin
        if self.closed:
            raise RuntimeError('Session closed')
        if algorithm not in {'bfs', 'sssp', 'pr', 'weighted_pr'}:
            raise NotImplementedError(algorithm)
        parameters = dict(parameters or {})
        if algorithm in {'bfs', 'sssp'}:
            parameters['source'] = source_parameter(parameters)
            # Registration owns a sorted unique int64 domain. Binary lookup
            # avoids scanning every vertex on each BFS/SSSP request.
            nodes = self.nodes[graph_id]
            source = parameters['source']
            if (not len(nodes) or source < int(nodes[0]) or source > int(nodes[-1])
                    or nodes[np.searchsorted(nodes, source)] != source):
                raise ValueError('Source absent from the registered graph')
        # Numerical options belong to the session; reject silent per-call overrides.
        if algorithm in {'pr', 'weighted_pr'} and parameters:
            raise ValueError('Set PageRank tolerance and iteration limit when creating the arena session')
        with _download_request(graph_id, algorithm) as scope:
            values = self.backend.query(graph_id, algorithm, parameters)
            contract = query_contract(algorithm, parameters, len(self.nodes[graph_id]))
            if algorithm in {'pr', 'weighted_pr'}:
                count = len(self.nodes[graph_id])
                contract = replace(contract, max_iterations=self.pr_max_iter,
                                   tolerance_per_vertex=self.pr_tolerance_l1 / count if count else None)
            return self._own_result(graph_id, algorithm, values, begin, copy.deepcopy(self.backend.last), parameters,
                                    contract=contract, _arena_scope=scope)

    def update(self, graph_id, batch):
        with self._lock:
            if self.closed:
                raise RuntimeError('Session closed')
            return self.backend.update(graph_id, batch)

    def stats(self):
        with self._lock:
            if self.closed:
                raise RuntimeError('Session closed')
            return self.backend.stats()

    def describe(self, graph_id):
        with self._lock:
            if self.closed:
                raise RuntimeError('Session closed')
            engine = self.backend.session
            record = engine.journal._records[graph_id]
            graph = engine.graphs[graph_id]
            version = GraphVersion(int(record.ref.generation), int(record.version),
                                   int(graph.generation), int(graph.versions['message_count']),
                                   'backend.session.journal._records[gid].ref.generation/.version; graphs[gid].generation/.versions')
            return GraphDescriptor(graph_id, f'{self.session_id}/graph/{graph_id}', 'managed', version,
                                   self._domains[graph_id], None, None,
                                   {'semantic_note': 'Directed adjacency and message_count projection; input symmetry is not inferred',
                                    'node_source': 'owned sorted registration labels',
                                    'cross_backend_identity_verified': False})

    def capability_descriptor(self):
        # Device identity comes from an existing arena array; do not query another GPU.
        engine = self.backend.session
        devices = tuple(sorted({int(slot.array.device.id) for slot in engine.arena.slots.values()}))
        return CapabilityDescriptor('managed', ('bfs', 'sssp', 'pr', 'weighted_pr'), devices,
                                    'fixed-node-domain edge set/insert/delete via existing journal',
                                    'sorted unique signed-int64 original labels', 'owning full float64 host array + node_ids',
                                    'complete arena extents and ledger owners at request boundaries', False,
                                    ('automatic_prepare', 'automatic_repair', 'automatic_evict'),
                                    'Capabilities describe implementation, not a claim that every combination was GPU-qualified',
                                    ('No node-domain replacement API in this wrapper', 'No cross-backend physical ownership',
                                     'CUDA context/modules and external allocations excluded from arena accounting'))

    def snapshot(self):
        with self._lock:
            if self.closed:
                raise RuntimeError('Session closed')
            allocations, components, consistency = managed_inventory(self.backend.session, self.session_id)
            return StateSnapshot('eggpu.state_snapshot.v1', self.session_id, 'managed',
                                 tuple(self.describe(g) for g in self.nodes), self.capability_descriptor(),
                                 allocations, components, self._live_outputs(), copy.deepcopy(self.stats()), True,
                                 'Disjoint rounded-capacity arena extents including reserved workspace; not total CUDA/NVML memory',
                                 consistency, ('Build-stage transient peaks and route costs require separate measurement',
                                               'Live outputs are host allocations and are not device pins'))

    def close(self):
        with self._lock:
            if not self.closed:
                self.backend.close()
                self.closed = True


class NativePartitionSession(_Bridge):
    capabilities = frozenset({'bfs', 'pr', 'immutable_partitions', 'multiple_devices'})

    def __init__(self, cpp, manifests, *, devices=(0,)):
        global _native_runtime_owner
        self.cpp, self.devices = cpp, list(devices)
        self._lock = _native_runtime_lock
        self.graphs, self.metadata = {}, {}
        self.nodes, self._manifest_documents, self._manifest_paths = {}, {}, {}
        self.closed = True
        if not manifests or not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError('Require nonempty manifests and unique devices')
        if any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in self.devices):
            raise ValueError('Devices must be nonnegative integer logical device IDs')
        begin = perf_counter()
        with _native_runtime_lock:
            # The DSO runtime is process-global. Attaching then shutting down would
            # invalidate another session; this adapter deliberately has one owner.
            if _native_runtime_owner is not None or cpp.cpp_egmgpu_runtime_status()['running']:
                raise RuntimeError('Native runtime already owned; explicitly close its owner before starting another session')
            for name, path in manifests.items():
                document = json.loads(Path(path).read_text())
                metadata = document['graph']
                self._validate_manifest_document(name, path, document)
                if 'partitions' in document and sorted(p['device_id'] for p in document['partitions']) != sorted(self.devices):
                    raise ValueError('Manifest device placement does not match requested logical devices')
                self.metadata[name] = metadata
                self._manifest_documents[name], self._manifest_paths[name] = document, str(Path(path).resolve())
                self.nodes[name] = np.arange(metadata['vertex_count'], dtype=np.int64)
            self._initialize_bridge('native')
            cpp.cpp_egmgpu_runtime_start(devices=self.devices, backend=os.environ.get('EGGPU_COMM_BACKEND', 'peer_to_peer'))
            self.closed = False
            _native_runtime_owner = self
            try:
                for name, path in manifests.items():
                    self.graphs[name] = cpp.cpp_egmgpu_load_partition_manifest(str(path), import_policy='structural')
            except BaseException:
                self.close()
                raise
        self.initialization_seconds = perf_counter() - begin

    def _validate_manifest_document(self, name, path, document):
        if document['graph'].get('weighted', False):
            raise NotImplementedError('Common native adapter is qualified for unweighted manifests only')

    def query(self, graph_id, algorithm, parameters=None):
        begin = perf_counter()
        with _native_runtime_lock:
            return self._query(graph_id, algorithm, parameters, begin)

    def _query(self, graph_id, algorithm, parameters=None, begin=None):
        begin = perf_counter() if begin is None else begin
        if self.closed:
            raise RuntimeError('Session closed')
        parameters = dict(parameters or {})
        graph = self.graphs[graph_id]
        if algorithm == 'bfs':
            source = source_parameter(parameters)
            parameters['source'] = source
            if not 0 <= source < self.metadata[graph_id]['vertex_count']:
                raise ValueError('Source outside the manifest global vertex domain')
            raw = self.cpp.cpp_egmgpu_partitioned_bfs(
                graph, source, devices=self.devices,
                dense_output=True, frontier_policy='adaptive')
            values = np.asarray(raw['values_dense'], dtype=np.float64)
            values = np.where(values < 0, np.inf, values)
        elif algorithm == 'pr':
            if parameters:
                raise ValueError('Common PR is fixed at alpha=.85, max_iter=10000, tol=1e-6/n')
            n = self.metadata[graph_id]['vertex_count']
            raw = self.cpp.cpp_egmgpu_partitioned_pagerank(graph, .85, 10000, 1e-6 / n, self.devices)
            if not raw['converged']:
                raise RuntimeError('Native PageRank did not converge')
            values = raw['values_dense']
        else:
            raise NotImplementedError(f'Not qualified through this adapter: {algorithm}')
        metrics = {key: value for key, value in raw.items() if key not in {'values', 'values_dense', 'labels_dense'}}
        return self._own_result(graph_id, algorithm, values, begin, copy.deepcopy(metrics), parameters)

    def describe(self, graph_id):
        with _native_runtime_lock:
            if self.closed:
                raise RuntimeError('Session closed')
            metadata = self.metadata[graph_id]
            document = self._manifest_documents[graph_id]
            generation = metadata.get('generation')
            if document.get('snapshot_storage') == 'native_memory':
                version = GraphVersion(None if generation is None else int(generation), None, None, None,
                                       'immutable native_memory snapshot generation')
                return GraphDescriptor(graph_id, f'{self.session_id}/graph/{graph_id}', 'native', version,
                    self._domains[graph_id], metadata.get('directed'), metadata.get('weighted'),
                    {'snapshot_storage': 'native_memory', 'manifest_path': None,
                     'snapshot_document_sha256': hashlib.sha256(json.dumps(document,sort_keys=True).encode()).hexdigest(),
                     'native_graph_id': metadata.get('id'), 'topology_signature': metadata.get('topology_sha256'),
                     'signature_scope': 'partition layout dependent; immutable native arrays derived from a verified base',
                     'cross_backend_identity_verified': False})
            version = GraphVersion(None if generation is None else int(generation), None,
                                   None, None, 'immutable manifest.graph.generation; no dynamic native revision API')
            return GraphDescriptor(graph_id, f'{self.session_id}/graph/{graph_id}', 'native', version,
                                   self._domains[graph_id], metadata.get('directed'), metadata.get('weighted'),
                                   {'manifest_path': self._manifest_paths[graph_id],
                                    'manifest_document_sha256': hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(),
                                    'manifest_graph_id': metadata.get('id'),
                                    'manifest_topology_signature': metadata.get('topology_sha256'),
                                    'signature_scope': 'partition layout dependent; not a cross-partition canonical graph hash',
                                    'cross_backend_identity_verified': False})

    def capability_descriptor(self):
        return CapabilityDescriptor('native', ('bfs', 'pr'), tuple(self.devices), 'immutable; updates explicitly rejected',
                                    'dense global integer IDs 0..n-1', 'owning full float64 host array + node_ids',
                                    'aggregate native cache counters only; allocation owner identities unavailable', False,
                                    ('automatic_cache_reuse',),
                                    'Per-cohort GPU qualification is recorded separately from available DSO entrypoints',
                                    ('No dynamic version-safe updates', 'No fine-grained retain/evict/materialize wrapper',
                                     'No allocation-level shared-owner/refcount export', 'No measured recovery route stage peaks',
                                     'SSSP/TC/WCC/other native entrypoints are not yet exposed by this common wrapper'))

    def snapshot(self):
        with _native_runtime_lock:
            if self.closed:
                raise RuntimeError('Session closed')
            counters = copy.deepcopy(self.stats())
            return StateSnapshot('eggpu.state_snapshot.v1', self.session_id, 'native',
                                 tuple(self.describe(g) for g in self.graphs), self.capability_descriptor(),
                                 (), (), self._live_outputs(), counters, False,
                                 'Native per-device cache entries/shared catalog counters; not total device allocation union',
                                 {'consistent': None, 'reason': 'No allocation inventory in frozen Python native API'},
                                 ('Aggregate bytes cannot be used as independent additive allocations',
                                  'Exact recovery planner ingestion is intentionally rejected'))

    def update(self, graph_id, batch):
        raise NotImplementedError('Frozen native manifests are immutable; no version-safe dynamic update adapter is implemented')

    def stats(self):
        with _native_runtime_lock:
            if self.closed:
                raise RuntimeError('Session closed')
            return self.cpp.cpp_egmgpu_partitioned_state_stats(devices=self.devices)

    def close(self):
        global _native_runtime_owner
        with _native_runtime_lock:
            if not self.closed:
                if _native_runtime_owner is not self:
                    raise RuntimeError('Native runtime ownership mismatch')
                # On shutdown failure retain ownership, allowing explicit cleanup.
                self.cpp.cpp_egmgpu_runtime_shutdown(drain=True)
                self.closed = True
                _native_runtime_owner = None
