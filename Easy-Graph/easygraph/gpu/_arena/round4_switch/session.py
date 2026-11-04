"""Shared-arena graph switching with observable mutations and deferred layouts.

Every control uses the same host journal, logical update semantics, native
algorithms, component LRU, and actual memory arena. Scheduling never sees a
future call. Graph identity and replacement generation qualify all history.
"""
from dataclasses import replace
import time
import numpy as np
import cupy as cp

from easygraph.gpu._arena.factorized_session import RepresentationSession, portfolio_request_bytes, portfolio_workspace_bytes
from easygraph.gpu._arena.gpu_runtime import Arena
from easygraph.gpu._arena.session_manager import request_boundary
from easygraph.gpu._arena.round2_topology.gpu_adapter import GraphAdapter, PreparedState, ArrayShape
from easygraph.gpu._arena.round2_topology.state import Delta, unique_array_bytes
from easygraph.gpu._arena.round3_transfer.packet import make_packet
from easygraph.gpu._arena.round3_transfer.arena_growth import grow_many, growth_bytes
from easygraph.gpu._arena.round3_transfer.kernels import TransferKernels
from easygraph.gpu._arena.round4_switch.journal import JournalStore, Mutation

MODES = ('eager', 'buffer_refresh', 'buffer_advance', 'buffer_scoped')


class SwitchingSession(RepresentationSession):
    def __init__(self, inputs, budget_bytes, journal_budget_bytes, mode):
        if mode not in MODES:
            raise ValueError(mode)
        if len({item['name'] for item in inputs}) != len(inputs):
            raise ValueError('Graph identifiers must be unique')
        self.mode = mode
        self.journal = JournalStore(journal_budget_bytes)
        self.refs = {}
        self.transfer_kernels = None
        self.last_graph = None
        self.residence = {}
        self.logical_tick = 0
        graphs = []
        for item in inputs:
            ref = self.journal.register(item['name'], item['nodes'], item['edges'], item.get('weights'))
            self.refs[item['name']] = ref
            snap = self.journal.snapshot(ref)
            graphs.append(GraphAdapter(snap.host_state, name=item['name']))
        super().__init__(graphs, budget_bytes, costs=None, representation='factorized',
                         sum_builder='host', selection='greedy', kernel_backend='native_paths')

    def _graph_components(self, gid):
        return [name for name in self.model.components if self.semantic[name][0] == gid]

    def _retire_graph(self, gid):
        for name in self._graph_components(gid):
            self._drop(name)

    def _sync_snapshot(self, gid):
        """Recover a committed host flush after interrupted device adoption."""
        snapshot = self.journal.snapshot(self.refs[gid])
        old = self.graphs[gid]
        if snapshot is None or snapshot.host_state is old.r2_state:
            return
        self._retire_graph(gid)
        graph = GraphAdapter(snapshot.host_state, name=gid, generation=old.generation+1,
                             attribute_version=old.versions['message_count']+1)
        self.graphs[gid] = graph; self._bind_active_keys(graph)
        for suffix in ('topology', 'mapping', 'all', 'attr:message_count'):
            self.model.update(gid+':'+suffix)
        self._log_compaction_pending = True
        self.counters['committed_snapshot_recoveries'] += 1

    def _drop(self, name):
        pending = self.residence.pop(name, None)
        if pending is not None:
            self.counters['maintained_then_evicted_before_use'] += 1
            self.counters['unconsumed_maintenance_residence_ticks'] += self.logical_tick - pending[1]
        return super()._drop(name)

    def _reconcile_residence(self):
        # The common planner can retire allocations without invoking _drop.
        for name in list(self.residence):
            if (name not in self.model.components or
                self.model.components[name].allocation_id != self.residence[name][0]):
                self.counters['maintained_then_evicted_before_use'] += 1
                self.counters['unconsumed_maintenance_residence_ticks'] += self.logical_tick - self.residence.pop(name)[1]

    def _advance(self, gid, old_state, new_state, changes, names):
        """One shared packet and growth operation for this graph's survivors."""
        if not names or not changes:
            return {}, {}
        if any(c.final_weight is None for c in changes):
            self.counters['deletion_refreshes'] += 1
            return {}, {'path': 'deletion_refresh'}
        nodes = new_state.nodes
        src = np.searchsorted(nodes, [c.src for c in changes]).astype(np.int32)
        dst = np.searchsorted(nodes, [c.dst for c in changes]).astype(np.int32)
        delta = Delta(src, dst, np.arange(len(changes), dtype=np.int32),
                      np.asarray([c.final_weight for c in changes], np.float64),
                      np.asarray([c.base_weight is None for c in changes], bool), np.unique(src))
        started = time.perf_counter()
        packet = make_packet(old_state.outgoing, old_state.incoming, new_state, delta,
                             {key[1:] for key in names.values()})
        shapes = {self.model.components[name].allocation_id:
                  ((new_state.m,), self.arena.slots[self.model.components[name].allocation_id].array.dtype)
                  for name, key in names.items() if key[1] in ('col', 'out_col', 'weight')}
        reserve = Arena.capacity((packet.nbytes,), np.uint8) if packet.nbytes else 0
        needed = growth_bytes(self.arena, shapes)
        detail = {'packet_bytes': packet.nbytes, 'packet_seconds': time.perf_counter()-started,
                  'migration_d2d_bytes': 0, 'growth_d2d_bytes': 0, 'path': 'incremental'}
        if self.arena.live + needed + reserve > self.arena.budget:
            self.counters['incremental_capacity_fallbacks'] += 1
            return {}, dict(detail, path='capacity_refresh')
        if self.transfer_kernels is None:
            self.transfer_kernels = TransferKernels()
        log_key = 'r4:transfer_packet'
        try:
            grow = grow_many(self.arena, shapes, reserve_bytes=reserve)
            detail['growth_d2d_bytes'] = grow['packed_growth_d2d_bytes']
            fields = {}
            if packet.nbytes:
                try:
                    array = self.arena.allocate(log_key, (packet.nbytes,), np.uint8)
                except MemoryError:
                    self.arena.compact(staging_key='@native_bfs_output')
                    array = self.arena.allocate(log_key, (packet.nbytes,), np.uint8)
                array.set(packet.data)
                self.counters['h2d_bytes'] += packet.nbytes
                self.counters['maintenance_packet_h2d_bytes'] += packet.nbytes
                slot = self.arena.slots[log_key]
                fields = {key: cp.ndarray(info['shape'], dtype=info['dtype'],
                           memptr=cp.cuda.memory.MemoryPointer(self.arena.memory, slot.offset+info['offset']))
                          for key, info in packet.fields.items()}
            for name, key in names.items():
                kind, attr = key[1:]
                array = self.arena.slots[self.model.components[name].allocation_id].array
                direction = 'out' if kind.startswith('out_') else 'in'
                desc = {p: fields[direction+'_'+p] for p in ('before', 'values', 'rows')
                        if direction+'_'+p in fields}
                if np.any(delta.new):
                    if kind in ('col', 'out_col', 'weight'):
                        detail['migration_d2d_bytes'] += self.transfer_kernels.insert(array, old_state.m, desc, self.arena)
                    elif kind in ('row', 'out_row'):
                        self.transfer_kernels.row(array, desc)
                if kind == 'weight' and 'weight_positions' in fields:
                    self.transfer_kernels.patch(array, fields['weight_positions'], fields['weight_values'])
                elif kind == 'source_sum':
                    prefix = 'degree' if attr is None else 'sum'
                    if prefix+'_positions' in fields:
                        self.transfer_kernels.patch(array, fields[prefix+'_positions'], fields[prefix+'_values'])
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            # Logical mutations were already committed. Retire affected device
            # state; a retry reconstructs the current authoritative host graph.
            self._retire_graph(gid)
            self.counters['failed_device_maintenance_flushes'] += 1
            raise
        finally:
            if log_key in self.arena.slots:
                self.arena.release(log_key)
        self.counters['maintenance_packets'] += bool(packet.nbytes)
        self.counters['migration_d2d_bytes'] += detail['migration_d2d_bytes']
        self.counters['growth_d2d_bytes'] += detail['growth_d2d_bytes']
        return names, detail

    def _adopt(self, flush, allow_advance):
        gid = flush.ref.graph_id
        old = self.graphs[gid]
        state = flush.new_snapshot.host_state
        changes = flush.changes
        topology = any(c.base_weight is None or c.final_weight is None for c in changes)
        allowed = {'row', 'col', 'out_row', 'out_col', 'weight', 'source_sum'}
        resident = {name: self.semantic[name] for name in self._graph_components(gid)
                    if self.model.valid(name) and self.semantic[name][1] in allowed
                    and self.model.allocations[self.model.components[name].allocation_id].owners == {name}
                    and (topology or self.semantic[name][1:] in
                         (('weight','message_count'), ('source_sum','message_count')))}
        graph = GraphAdapter(state, name=gid, generation=old.generation+int(topology),
                             attribute_version=old.versions['message_count']+int(bool(changes)))
        # The host snapshot is the committed authority, including when a later
        # optional device migration fails. Publish dependency versions afterward.
        advanced = {}; detail = {'path': 'refresh'}
        try:
            if allow_advance:
                advanced, detail = self._advance(gid, old.r2_state, state, changes, resident)
        finally:
            self.graphs[gid] = graph
            self._bind_active_keys(graph)
            if topology:
                self.model.update(gid+':topology')
            if changes:
                self.model.update(gid+':attr:message_count')
            self._log_compaction_pending = True
        for name, key in advanced.items():
            component = self.model.components[name]
            allocation = self.model.allocations[component.allocation_id]
            slot = self.arena.slots[component.allocation_id]
            allocation.capacity_bytes = slot.capacity
            component.spec = replace(component.spec, capacity_bytes=slot.capacity)
            component.stamp = self.model.stamp(component.spec.dependencies)
            self.device_versions[name] = graph.version(key)
            self.residence.setdefault(name, (component.allocation_id, self.logical_tick))
            self.counters['maintained_components'] += 1
        self.model.revision += 1
        self.model.assert_invariants()
        if self.model.live_bytes != self.arena.live:
            raise AssertionError('Multigraph physical ledger mismatch')
        self.counters['host_materializations'] += 1
        return detail

    @request_boundary
    def mutate(self, gid, operations, *, fail_after=None):
        started = time.perf_counter(); self.logical_tick += 1
        ref = self.refs[gid]
        self._sync_snapshot(gid)
        if fail_after == 'before_commit':
            raise RuntimeError('Injected before logical mutation commit')
        if fail_after is not None:
            raise ValueError('Unknown mutation failure point')
        receipt = self.journal.commit(ref, operations)
        for flush in receipt.auto_flushed:
            self._retire_graph(flush.ref.graph_id)
            self._adopt(flush, False)
            self.counters['host_pressure_flushes'] += 1
        detail = None
        if self.mode == 'eager' and self.journal.peek_pending(ref):
            detail = self._adopt(self.journal.flush(ref), True)
        self.counters['logical_operations'] += len(operations)
        self.counters['mutation_batches'] += 1
        self._reconcile_residence()
        return {'seconds': time.perf_counter()-started, 'logical_version': receipt.version,
                'maintenance': detail, 'auto_flushed': [x.ref.graph_id for x in receipt.auto_flushed]}

    @request_boundary
    def query(self, gid, function, attr=None, source=0, result='array', selected=None):
        started = time.perf_counter(); self.logical_tick += 1
        if self.last_graph is not None and self.last_graph != gid:
            self.counters['graph_switches'] += 1
        self.last_graph = gid
        ref = self.refs[gid]; changes = self.journal.peek_pending(ref)
        self._sync_snapshot(gid)
        topology = any(c.base_weight is None or c.final_weight is None for c in changes)
        needs_values = function == 'sssp' or (function == 'pr' and attr is not None)
        needed = bool(changes) and (self.mode != 'buffer_scoped' or topology or needs_values)
        detail = None
        if needed:
            detail = self._adopt(self.journal.flush(ref), self.mode != 'buffer_refresh')
        elif changes:
            self.counters['dependency_scoped_deferred_queries'] += 1
        graph = self.graphs[gid]
        if self.executor.workspace_bytes + portfolio_request_bytes(graph) > self.arena.budget:
            raise MemoryError('Current graph request exceeds shared arena budget')
        with graph.r2_state.lease():
            answer = super().query(gid, function, attr, source, result, selected)
        self._reconcile_residence()
        from easygraph.gpu._arena.session_manager import token
        for key in self._requirements(graph, function, attr):
            name = token(key)
            if name in self.residence:
                self.counters['maintained_then_consumed'] += 1
                self.counters['maintenance_to_consumption_ticks'] += self.logical_tick - self.residence.pop(name)[1]
        self.last.update(seconds=time.perf_counter()-started, deferred_host_materialization=needed,
                         journal_maintenance=detail, graph_epoch=ref.generation)
        return answer

    @request_boundary
    def suspend(self, gid):
        """Explicit pause preserves graph state, journal, and LRU eligibility."""
        if gid not in self.refs: raise KeyError(gid)
        self.counters['suspends'] += 1
        return {'graph': gid, 'retained_components': len(self._graph_components(gid)),
                'pending_unique_edges': len(self.journal.peek_pending(self.refs[gid]))}

    @request_boundary
    def evict_device(self, gid):
        """Retire a graph's GPU representations, preserving committed edits."""
        self._retire_graph(gid)
        self.counters['explicit_graph_device_evictions'] += 1

    @request_boundary
    def logical_edges(self, gid):
        """Public host reads observe committed edits even before CSR materialization."""
        return self.journal.edges(self.refs[gid])

    @request_boundary
    def update(self, graph_name, attr, positions, values):
        """Legacy positional updates refer to current logical incoming order."""
        if attr != 'message_count': raise KeyError(attr)
        positions = np.asarray(positions)
        values = np.asarray(values)
        if positions.ndim != 1 or positions.dtype.kind not in 'iu' or values.shape != positions.shape:
            raise ValueError('One integer incoming-edge position per weight is required')
        table = self.journal.edges(self.refs[graph_name])
        pairs = sorted(table, key=lambda edge:(edge[1], edge[0]))
        if np.any(positions < 0) or np.any(positions >= len(pairs)):
            raise ValueError('Position outside current logical incoming edge order')
        return self.mutate(graph_name, [Mutation('set', *pairs[int(p)], float(v))
                                       for p,v in zip(positions,values)])

    @request_boundary
    def replace_graph(self, graph):
        """Route the inherited graph-object API through the same authority."""
        nodes = np.asarray(graph.nodes)
        if nodes.ndim != 1 or nodes.dtype.kind not in 'iu':
            raise ValueError('Replacement graph must use integer node labels')
        nodes = np.asarray(nodes, np.int64)
        if hasattr(graph, 'r2_state'):
            state = graph.r2_state
            src = np.repeat(np.arange(graph.n), np.diff(state.outgoing.row))
            edges = np.column_stack((nodes[src], nodes[state.outgoing.col]))
            weights = state.values[state.outgoing.ids]
        else:
            dst = np.repeat(np.arange(graph.n), np.diff(graph.row))
            edges = np.column_stack((nodes[np.asarray(graph.col)], nodes[dst]))
            weights = graph.attributes['message_count']
        return self.replace(graph.name, nodes, edges, weights)

    @request_boundary
    def replace(self, gid, nodes, edges, weights=None, *, fail_after=None):
        started = time.perf_counter()
        if fail_after not in (None, 'after_workspace'):
            raise ValueError('Unknown replacement failure point')
        if gid not in self.refs: raise KeyError(gid)
        if any(a.pins for a in self.model.allocations.values()):
            raise RuntimeError('Replacement waits for current readers')
        # Preflight both success and rollback graph sets. Workspace is allowed
        # to remain larger after a failed replacement and stays fully charged.
        candidate = GraphAdapter(PreparedState(nodes, edges, weights), name=gid)
        needed = portfolio_workspace_bytes(max(self.executor.nmax, candidate.n), self.kernel_backend)
        # Pending committed edges also belong to the rollback contract. Their
        # last materialized CSR may be smaller than the current logical graph.
        from types import SimpleNamespace
        logical_shapes = [SimpleNamespace(n=g.n, row=ArrayShape(g.n+1),
                          col=ArrayShape(len(self.journal.edges(self.refs[name]))),
                          attributes={'message_count':None}) for name,g in self.graphs.items()]
        largest = max(portfolio_request_bytes(candidate),
                      *(portfolio_request_bytes(g) for g in logical_shapes))
        if needed + largest > self.arena.budget:
            raise MemoryError('Replacement and rollback requests exceed arena budget')
        if candidate.n > self.executor.nmax:
            self._grow_workspace(candidate.n, needed)
        if fail_after == 'after_workspace':
            raise RuntimeError('Injected after charged workspace growth')
        newref = self.journal.replace(self.refs[gid], nodes, edges, weights)
        snapshot = self.journal.snapshot(newref)
        old = self.graphs[gid]
        self.refs[gid] = newref
        self._retire_graph(gid)
        graph = GraphAdapter(snapshot.host_state, name=gid, generation=old.generation+1,
                             attribute_version=old.versions['message_count']+1)
        self.refs[gid] = newref; self.graphs[gid] = graph; self._bind_active_keys(graph)
        for suffix in ('topology', 'mapping', 'all', 'attr:message_count'):
            self.model.update(gid+':'+suffix)
        self._log_compaction_pending = True
        self.counters['graph_replacements'] += 1
        return {'seconds': time.perf_counter()-started, 'generation': newref.generation}

    def stats(self):
        self._reconcile_residence()
        return dict(self.counters, mode=self.mode, arena_reserved_bytes=self.arena.budget,
                    peak_capacity_bytes=self.arena.peak, live_capacity_bytes=self.arena.live,
                    workspace_bytes=self.executor.workspace_bytes,
                    host_array_bytes=unique_array_bytes([a for g in self.graphs.values()
                         for a in g.r2_state.arrays()+list(g.arrays.values())]),
                    compaction_d2d_bytes=self.arena.compaction_bytes,
                    journal=self.journal.stats(), pending_maintained_components=len(self.residence))
