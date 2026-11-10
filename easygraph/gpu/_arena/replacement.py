"""R9 replacement: prepare once, admit against both worlds, then publish.

The frozen R8 TypedSession and all its computational kernels are inherited.
Only replacement reuses the already qualified cold-input production chain.
No global monkeypatch, old-generation reuse, or uncharged prepared input.
"""
import time
from types import SimpleNamespace
import numpy as np
from easygraph.gpu._arena.session_manager import request_boundary
from easygraph.gpu._arena.factorized_session import portfolio_request_bytes, portfolio_workspace_bytes
from easygraph.gpu._arena.round2_topology.gpu_adapter import GraphAdapter, ArrayShape
from easygraph.gpu._arena.round4_switch.journal import GraphRef, _Record
from easygraph.gpu._arena.round8_columns.session import TypedSession


def logical_edge_count(store, ref):
    """Exact support size from coalesced base/final presence, without folding."""
    record = store._record(ref)
    count = len(record.base_edges)
    journal = store._journals.get(ref)
    if journal is not None:
        flags = journal.data['flags']
        count += int(np.count_nonzero(flags & 2)) - int(np.count_nonzero(flags & 1))
    if count < 0:
        raise AssertionError('Negative logical support size')
    return count


def prepare_replacement(store, ref, nodes, edges, weights):
    """Build a private complete record; all public authority remains untouched.

Caller holds the store lock. This uses the same _input/_build and temporary
ColdPreparation ownership as ColdStore._register, splitting its final publish.
"""
    store._record(ref)
    newref = GraphRef(ref.graph_id, store._generations[ref.graph_id] + 1)
    started = time.perf_counter()
    store._cold_frames.append(None)
    try:
        labels, allowed, base = store._input(nodes, edges, weights)
        snapshot = store._build(newref, labels, base, None, (), 0).new_snapshot
        return _Record(newref, labels, allowed, base, 0, 0, snapshot)
    finally:
        preparation = store._cold_frames.pop()
        if preparation is not None:
            if not preparation.consumed:
                store.cold_count['discarded_unconsumed_preparations'] += 1
            preparation.release()
        store._cold_observe()
        store.cold_count['raw_registration_seconds'] += time.perf_counter() - started


class ReplacementSession(TypedSession):
    @request_boundary
    def replace(self, gid, nodes, edges, weights=None, *, fail_after=None):
        started = time.perf_counter()
        if fail_after not in (None, 'after_workspace'):
            raise ValueError('Unknown replacement failure point')
        if self.batch_requests is not None:
            raise RuntimeError('Replacement commits between analysis epochs')
        if gid not in self.refs:
            raise KeyError(gid)
        if any(a.pins for a in self.model.allocations.values()):
            raise RuntimeError('Replacement waits for current readers')
        store = self.journal
        with store._lock:
            oldref = self.refs[gid]
            old_graph = self.graphs[gid]
            candidate_record = prepare_replacement(store, oldref, nodes, edges, weights)
            graph = GraphAdapter(candidate_record.snapshot.host_state, name=gid,
                                 generation=old_graph.generation + 1,
                                 attribute_version=old_graph.versions['message_count'] + 1)
            needed = portfolio_workspace_bytes(max(self.executor.nmax, graph.n), self.kernel_backend)
            rollback = [SimpleNamespace(n=g.n, row=ArrayShape(g.n+1),
                        col=ArrayShape(logical_edge_count(store, self.refs[name])),
                        attributes={'message_count':None}) for name,g in self.graphs.items()]
            largest = max(portfolio_request_bytes(graph), *(portfolio_request_bytes(g) for g in rollback))
            if needed + largest > self.arena.budget:
                raise MemoryError('Replacement and rollback requests exceed arena budget')
            # Allocate the prospective journal container before touching publication.
            remaining = {key:value for key,value in store._journals.items() if key != oldref}
            if graph.n > self.executor.nmax:
                self._grow_workspace(graph.n, needed)
            if fail_after == 'after_workspace':
                raise RuntimeError('Injected after charged workspace growth')
            # Publication point: both a successful target request and every old
            # logical rollback request fit the possibly expanded workspace.
            newref = candidate_record.ref
            store._records[gid] = candidate_record
            store._generations[gid] = newref.generation
            store._journals = remaining
            store._counters['replacements'] += 1
            store.cold_count['successful_registrations'] += 1
            store._remove_certificate(oldref)
            store.change_cache.prune(store._journals)
            store._observe_owners()
            # Same recovery publication order as the frozen session: refs are
            # visible before retirement so _sync_snapshot can finish adoption.
            self.refs[gid] = newref
            self._retire_graph(gid)
            self.graphs[gid] = graph
            self._bind_active_keys(graph)
            for suffix in ('topology', 'mapping', 'all', 'attr:message_count'):
                self.model.update(gid + ':' + suffix)
            self._log_compaction_pending = True
            self.counters['graph_replacements'] += 1
            self.counters['replacement_prepared_once'] += 1
            self.counters['replacement_shape_only_rollbacks'] += len(rollback)
            return dict(seconds=time.perf_counter()-started, generation=newref.generation,
                        preparation='single private cold production, admitted then published')
