"""Explicit typed-column query/adoption and unchanged recovery objective."""
from dataclasses import replace
from functools import partial
import time
import numpy as np
from easygraph.gpu._arena.session_manager import request_boundary,token
from easygraph.gpu._arena.factorized_session import portfolio_request_bytes,RepresentationSession
from easygraph.gpu._arena.gpu_runtime import Arena
from easygraph.gpu._arena.round2_topology.gpu_adapter import GraphAdapter
from easygraph.gpu._arena.round4_switch.session import SwitchingSession
from easygraph.gpu._arena.round4_switch.admitted_v2 import packet_payload_bytes
from easygraph.gpu._arena.round6_recovery.planner_v2 import Component,Query,CostModel,Problem,evaluate_subset,plan as reference_plan
from easygraph.gpu._arena.round6_recovery.profile import calibrate
from easygraph.gpu._arena.round7_recovery.packet_cost import PacketPrice as OriginalPacketPrice
from easygraph.gpu._arena.round8_lazy.session import LazySession
from easygraph.gpu._arena.round8_bounded_input.session_v2 import BoundedSession
from easygraph.gpu._arena.round8_startup.session import StartupTransferKernels
from easygraph.gpu._arena.round8_columns.authority import Store
from easygraph.gpu._arena.round8_columns.bulk import commit_add_edges
from easygraph.gpu._arena.round8_columns.columns import topology,deletion,insert_count,source_count
from easygraph.gpu._arena.round8_columns.transfer import ColumnsTransferMixin

class PacketPrice(OriginalPacketPrice):
    @staticmethod
    def features(old_edges,changes):
        return ('topology',max(1,old_edges+len(changes))) if insert_count(changes) else ('attribute',max(1,len(changes)))


class Session(ColumnsTransferMixin,BoundedSession):
    def __init__(self,*args,fast_input=True,lazy_module=True,change_budget_bytes=8<<20,
                 input_lut_budget_bytes=8<<20,packed_input_sort=True,
                 transfer_backend='nvrtc',transfer_resolution='demand',**kwargs):
        if fast_input is not True:raise ValueError('Typed transition candidate retains the qualified prepared-input path')
        if transfer_backend!='nvrtc' or transfer_resolution!='demand':
            raise ValueError('The candidate uses uniform NVRTC demand')
        kernels=StartupTransferKernels(transfer_backend,transfer_resolution)
        factory=partial(Store,change_budget_bytes=change_budget_bytes,
                        input_lut_budget_bytes=input_lut_budget_bytes,packed_input_sort=packed_input_sort)
        LazySession.__init__(self,*args,lazy_module=lazy_module,store_factory=factory,**kwargs)
        self.fast_input=True;self.input_lut_budget_bytes=input_lut_budget_bytes
        self.packed_input_sort=packed_input_sort;self.packet_price=PacketPrice()
        self.transfer_kernels=kernels

    @request_boundary
    def _query_columns(self, gid, function, attr=None, source=0, result='array', selected=None):
        started = time.perf_counter(); self.logical_tick += 1
        if self.last_graph is not None and self.last_graph != gid:
            self.counters['graph_switches'] += 1
        self.last_graph = gid
        ref = self.refs[gid]; changes = self.journal.peek_pending(ref)
        self._sync_snapshot(gid)
        changed_topology = topology(changes)
        needs_values = function == 'sssp' or (function == 'pr' and attr is not None)
        needed = bool(changes) and (self.mode != 'buffer_scoped' or changed_topology or needs_values)
        detail = None
        if needed:
            detail = self._adopt(self.journal.flush(ref), self.mode != 'buffer_refresh')
        elif changes:
            self.counters['dependency_scoped_deferred_queries'] += 1
        graph = self.graphs[gid]
        if self.executor.workspace_bytes + portfolio_request_bytes(graph) > self.arena.budget:
            raise MemoryError('Current graph request exceeds shared arena budget')
        with graph.r2_state.lease():
            answer = RepresentationSession.query(self, gid, function, attr, source, result, selected)
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

    def query(self,gid,function,attr=None,source=0,result='array',selected=None):
        with self.lock:
            if self.batch_requests is None:
                with self.batch([(gid,function,attr,source)]):
                    return self.query(gid,function,attr,source,result,selected)
            if self.batch_index>=len(self.batch_requests) or (gid,function,attr,source)!=self.batch_requests[self.batch_index]:
                raise ValueError('Call differs from the submitted epoch')
            value=self._query_columns(gid,function,attr,source,result,selected)
            self.last['declared_batch_index']=self.batch_index;self.batch_index+=1
            return value

    def _adopt_columns(self, flush, allow_advance):
        gid = flush.ref.graph_id
        old = self.graphs[gid]
        state = flush.new_snapshot.host_state
        changes = flush.changes
        changed_topology = topology(changes)
        allowed = {'row', 'col', 'out_row', 'out_col', 'weight', 'source_sum'}
        resident = {name: self.semantic[name] for name in self._graph_components(gid)
                    if self.model.valid(name) and self.semantic[name][1] in allowed
                    and self.model.allocations[self.model.components[name].allocation_id].owners == {name}
                    and (changed_topology or self.semantic[name][1:] in
                         (('weight','message_count'), ('source_sum','message_count')))}
        graph = GraphAdapter(state, name=gid, generation=old.generation+int(changed_topology),
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
            if changed_topology:
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

    def _physical_problem(self, flush):
        gid = flush.ref.graph_id; old = self.graphs[gid]
        state = flush.new_snapshot.host_state; changes = flush.changes
        graph = GraphAdapter(state, name=gid)
        if self.executor.workspace_bytes + portfolio_request_bytes(graph) > self.arena.budget:
            raise MemoryError('Current portfolio request exceeds the shared arena')
        changed_topology = topology(changes)
        has_deletions = deletion(changes)
        eligible, stale = set(), set()
        allowed = {'row', 'col', 'out_row', 'out_col', 'weight', 'source_sum'}
        catalog = {name: self.semantic[name] for name in self.model.components}
        for name in self._graph_components(gid):
            key = catalog[name]; c = self.model.components[name]
            affected = changed_topology or key[1] in ('weight', 'out_weight', 'source_sum') and key[2] is not None
            if affected or not self.model.valid(name): stale.add(name)
            if affected and not has_deletions and self.model.valid(name) and key[1] in allowed and \
                    self.model.allocations[c.allocation_id].owners == {name}: eligible.add(name)
        if eligible and self.profile is None and self.strategy in ('closed_form','bidirectional_greedy','exhaustive'):
            self.profile = calibrate(self, self.historical_profile)
        queries = []
        for g, f, a, source in self.batch_requests[self.batch_index:]:
            keys = self._requirements(graph if g == gid else self.graphs[g], f, a)
            for key in keys: catalog[token(key)] = key
            queries.append(Query(tuple(token(key) for key in keys), label=f+':'+str(a)))
        components = []
        has_insertions = bool(insert_count(changes))
        for name, key in catalog.items():
            current = graph if key[0] == gid else self.graphs[key[0]]
            kind = key[1]; item = 4 if kind in ('row', 'col', 'out_row', 'out_col') else 8
            count = current.n if kind == 'source_sum' else current.n+1 if kind in ('row','out_row') else current.r2_state.m
            c = self.model.components.get(name)
            allocation = None if c is None else self.model.allocations[c.allocation_id]
            old_payload = None if c is None else self.arena.slots[c.allocation_id].array.nbytes
            movement = 2*old.r2_state.m*item if name in eligible and has_insertions and kind in ('col','out_col','weight') else 0
            launches = 0; copies = 0
            if name in eligible:
                if has_insertions and kind in ('col','out_col','weight'):
                    copies = (old.r2_state.m*item + self.arena.slots['@native_bfs_output'].capacity-1)//self.arena.slots['@native_bfs_output'].capacity
                    launches += copies
                    launches += int(item == 4)
                if has_insertions and kind in ('row','out_row'): launches += 1
                if kind in ('weight','source_sum'): launches += 1
            components.append(Component(name, Arena.capacity((count,), np.int32 if item==4 else np.float64),
                count*item, kind=kind, allocation_id=None if c is None else c.allocation_id,
                old_capacity_bytes=None if allocation is None else allocation.capacity_bytes,
                old_payload_bytes=old_payload, advance_d2d_bytes=movement,
                advance_cost=0. if self.profile is None else launches*self.profile['kernel_launch_seconds']+copies*self.profile['copy_launch_seconds'],
                pinned=False if allocation is None else bool(allocation.pins),
                last_access=0 if c is None else c.last_access))
        by_name = {c.name: c for c in components}
        # Growth compacts surviving slots, then moves every expanded suffix.
        # The callback predicts both copy passes using actual arena offsets.
        def relocation(selected, retained):
            if not any(by_name[n].capacity_bytes > by_name[n].old_capacity for n in selected): return 0
            cursor = self.executor.workspace_bytes; shifted = 0; moved = 0
            physical = {self.model.components[n].allocation_id:n for n in retained}
            for n in sorted(physical.values(), key=lambda n:self.arena.slots[self.model.components[n].allocation_id].offset):
                slot = self.arena.slots[self.model.components[n].allocation_id]
                if slot.offset != cursor: moved += 2*slot.capacity
                if shifted: moved += 2*slot.capacity
                cursor += slot.capacity
                if n in selected: shifted += max(0, by_name[n].capacity_bytes-slot.capacity)
            return int(moved)
        new_count = insert_count(changes)
        source_rows = source_count(changes)
        def packet(selected):
            return packet_payload_bytes({catalog[n][1:] for n in selected}, new_count=new_count,
                change_count=len(changes), changed_source_rows=source_rows)
        costs = None if self.profile is None else CostModel(
            h2d_per_byte=self.profile['h2d_per_byte'], d2d_per_byte=self.profile['d2d_per_byte'],
            packet_fixed=self.profile['packet_fixed'], restore_fixed=self.profile['restore_fixed'],
            provenance='charged session calibration plus frozen R5 packet construction median')
        return Problem(tuple(components), frozenset(self.model.components), frozenset(eligible),
            tuple(queries), self.arena.budget, fixed_bytes=self.executor.workspace_bytes,
            packet_bytes=packet, cost_model=costs, stale_names=frozenset(stale), growth_d2d_bytes=relocation)

    def _problem(self, flush):
        # Cost-based R7 strategies receive the identical charged R6 calibration.
        saved = self.strategy
        try:
            self.strategy = saved if saved in ('full_refresh','all_advance') else 'exhaustive'
            problem = self._physical_problem(flush)
        finally:
            self.strategy = saved
        self.packet_features=self.packet_price.features(self.graphs[flush.ref.graph_id].r2_state.m,flush.changes)
        self.packet_prediction=None
        if self.packet_cost=='online' and saved!='adaptive_fixed':
            problem,self.packet_prediction=self.packet_price.price(problem,self.packet_features)
        if self.capture_problems:
            from easygraph.gpu._arena.round7_recovery.problem_io import encode
            slots = {name:dict(offset=int(slot.offset), capacity=int(slot.capacity))
                     for name,slot in self.arena.slots.items()}
            self.problem_records.append(encode(problem, slots, graph=flush.ref.graph_id,
                version=flush.new_snapshot.version, batch_index=self.batch_index))
        return problem

    def _adopt(self, flush, allow_advance):
        if self.strategy == 'ordinary' or not allow_advance or self.batch_requests is None:
            return self._adopt_columns( flush, allow_advance)
        started=time.perf_counter()
        problem=self._problem(flush)
        if self.strategy == 'preselected':
            key=(flush.ref.graph_id, int(flush.new_snapshot.version), self.batch_index)
            if key not in self.preselected: raise AssertionError(('No preregistered subset', key))
            replay_started=time.perf_counter()
            chosen=evaluate_subset(problem, self.preselected[key], trace=True)
            chosen.update(strategy='preselected',evaluations=1,simulation_passes=1,
                candidate_menu_size=1<<len(problem.eligible_names),pruned_candidates=[],
                candidate_summary=[],planner_seconds=time.perf_counter()-replay_started,
                visibility='offline chosen-subset trace; diagnostic headroom only')
        elif self.strategy in ('adaptive','adaptive_fixed','bounded','certified','previous'):
            from easygraph.gpu._arena.round7_recovery.planner import RecoveryContext, plan
            if self.planner_context is None: self.planner_context=RecoveryContext()
            solver='adaptive' if self.strategy=='adaptive_fixed' else self.strategy
            chosen=plan(problem, strategy=solver, context=self.planner_context)
            chosen.update(strategy=self.strategy,solver_strategy=solver)
        else:
            chosen=reference_plan(problem, self.strategy)
        if not chosen['feasible']: raise MemoryError(chosen['reason'])
        for name in chosen['retired_stale']+chosen['maintenance_victims']:
            if name in self.model.components:self._drop(name)
        elapsed=time.perf_counter()-started
        self.counters['recovery_planning_ns']+=int(elapsed*1e9)
        self.counters['recovery_plan_calls']+=1
        self.counters['recovery_candidates']+=chosen['evaluations']
        self.counters['recovery_components_avoided']+=len(problem.eligible_names)-len(chosen['selected'])
        chosen.update(counter_before_h2d=self.counters['h2d_bytes'],
            counter_before_builds=self.counters['view_builds'],batch_index=self.batch_index)
        detail=self._adopt_columns(flush,bool(chosen['selected']))
        if self.packet_cost=='online':self.packet_price.observe(self.packet_features,detail)
        if chosen['selected'] and detail['path']!='incremental':raise AssertionError('Selected action not executed')
        if chosen['selected'] and detail['packet_bytes']!=chosen['packet_payload_bytes']:raise AssertionError('Packet size mismatch')
        chosen.update(graph=flush.ref.graph_id,logical_version=flush.new_snapshot.version,
            complete_planning_seconds=elapsed,actual_maintenance=detail,packet_price_prediction=self.packet_prediction)
        if self.capture_problems:self.problem_records[-1]['chosen']=chosen
        self.recovery_plans.append(chosen)
        return dict(detail,recovery_plan_index=len(self.recovery_plans)-1,
                    recovery_planning_seconds=elapsed,selected_components=len(chosen['selected']))

    def stats(self):
        result=super().stats()
        result['transition_representation']='immutable typed journal columns'
        result['transfer_code_loading']=self.transfer_kernels.stats()
        return result

    def close(self):
        try:return super().close()
        finally:self.transfer_kernels=None

class TypedSession(Session):
    @request_boundary
    def mutate_add_edges(self,gid,edges,weights=None,*,fail_after=None):
        if self.batch_requests is not None:
            raise RuntimeError('Updates commit between analysis epochs')
        started=time.perf_counter();self.logical_tick+=1
        ref=self.refs[gid];self._sync_snapshot(gid)
        if fail_after=='before_commit':raise RuntimeError('Injected before logical mutation commit')
        if fail_after is not None:raise ValueError('Unknown mutation failure point')
        receipt=commit_add_edges(self.journal,ref,edges,weights)
        for flush in receipt.auto_flushed:
            self._retire_graph(flush.ref.graph_id);self._adopt(flush,False)
            self.counters['host_pressure_flushes']+=1
        detail=None
        if self.mode=='eager' and self.journal.peek_pending(ref):detail=self._adopt(self.journal.flush(ref),True)
        self.counters['logical_operations']+=len(edges);self.counters['mutation_batches']+=1
        self.counters['array_mutation_batches']+=1;self._reconcile_residence()
        return dict(seconds=time.perf_counter()-started,logical_version=receipt.version,
                    maintenance=detail,auto_flushed=[x.ref.graph_id for x in receipt.auto_flushed])
