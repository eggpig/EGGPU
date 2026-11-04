"""Version-fenced batch recovery on the frozen array-backed graph session.

All strategies share arrays, growth/packet kernels, next-use eviction, complete
host results and declared-batch information. No future batch is inspected.
"""
from contextlib import contextmanager
import time
import weakref
import numpy as np

from easygraph.gpu._arena.gpu_runtime import Arena
from easygraph.gpu._arena.factorized_session import portfolio_request_bytes
from easygraph.gpu._arena.session_manager import token
from easygraph.gpu._arena.round2_topology.gpu_adapter import GraphAdapter
from easygraph.gpu._arena.round4_switch.admitted_v2 import packet_payload_bytes
from easygraph.gpu._arena.round5_input.array_session_v2 import ArraySwitchingSession
from easygraph.gpu._arena.round6_recovery.epoch_model import EpochModel
from easygraph.gpu._arena.round6_recovery.planner_v2 import Component, Query, CostModel, Problem, plan
from easygraph.gpu._arena.round6_recovery.profile import calibrate

STRATEGIES = ('ordinary', 'full_refresh', 'all_advance', 'closed_form', 'bidirectional_greedy', 'exhaustive')


class RecoverySession(ArraySwitchingSession):
    def __init__(self, inputs, budget_bytes, journal_budget_bytes, strategy, historical):
        if strategy not in STRATEGIES: raise ValueError(strategy)
        self.strategy = strategy
        self.batch_requests = None
        self.batch_index = 0
        self.batch_keys = ()
        self.recovery_plans = []
        self.profile = None
        self.historical_profile = historical
        super().__init__(inputs, budget_bytes, journal_budget_bytes, 'buffer_scoped')
        ledger = EpochModel(budget_bytes, 'shared_component_LRU')
        ledger.__dict__.update(self.model.__dict__)
        ledger.epoch_owner = weakref.ref(self)
        self.model = ledger

    def next_use(self, name):
        for offset, names in enumerate(self.batch_keys[self.batch_index:]):
            if name in names: return offset
        return len(self.batch_keys)+1

    @contextmanager
    def batch(self, requests):
        """Submit ordered (graph, function, attribute, source) calls atomically."""
        with self.lock:
            if self.batch_requests is not None: raise RuntimeError('Nested batch')
            requests = tuple(tuple(r) for r in requests)
            if not requests or any(len(r) != 4 for r in requests):
                raise ValueError('A nonempty declared four-field call sequence is required')
            # This executable epoch is one graph/version; graph changes occur
            # between epochs. The session preserves other graph state throughout.
            if len({r[0] for r in requests}) != 1:
                raise ValueError('One graph/version per analysis epoch')
            for gid, function, attr, source in requests:
                if gid not in self.graphs or function not in ('bfs', 'pr', 'sssp'):
                    raise ValueError('Unknown graph or function')
            self.batch_requests = requests; self.batch_index = 0
            self.batch_keys = tuple(tuple(token(k) for k in self._requirements(self.graphs[g], f, a))
                                    for g, f, a, s in requests)
            try:
                first_plan = len(self.recovery_plans)
                yield self
                if self.batch_index != len(requests): raise RuntimeError('Incomplete declared batch')
                for decision in self.recovery_plans[first_plan:]:
                    actual_h2d = self.counters['h2d_bytes']-decision['counter_before_h2d']
                    actual_builds = self.counters['view_builds']-decision['counter_before_builds']
                    native_sources = 4*sum(f in ('bfs','sssp') for g,f,a,s in requests[decision['batch_index']:])
                    expected_h2d = decision['h2d_bytes']+native_sources
                    decision['observed_batch_suffix'] = dict(h2d_bytes=actual_h2d,
                        expected_h2d_bytes=expected_h2d, view_builds=actual_builds,
                        expected_view_builds=len(decision['restores']))
                    if actual_h2d != expected_h2d or actual_builds != len(decision['restores']):
                        raise AssertionError(('Recovery simulation/execution mismatch',decision['observed_batch_suffix']))
            finally:
                self.batch_requests = None; self.batch_keys = (); self.batch_index = 0

    def query(self, gid, function, attr=None, source=0, result='array', selected=None):
        with self.lock:
            if self.batch_requests is None:
                with self.batch([(gid, function, attr, source)]):
                    return self.query(gid, function, attr, source, result, selected)
            if self.batch_index >= len(self.batch_requests) or \
                    (gid, function, attr, source) != self.batch_requests[self.batch_index]:
                raise ValueError('Call differs from the submitted epoch')
            value = super().query(gid, function, attr, source, result, selected)
            self.last['declared_batch_index'] = self.batch_index
            self.batch_index += 1
            return value

    def mutate(self, *args, **kwargs):
        with self.lock:
            if self.batch_requests is not None: raise RuntimeError('Updates commit between analysis epochs')
            return super().mutate(*args, **kwargs)

    def replace(self, *args, **kwargs):
        with self.lock:
            if self.batch_requests is not None: raise RuntimeError('Replacement commits between analysis epochs')
            return super().replace(*args, **kwargs)

    def _problem(self, flush):
        gid = flush.ref.graph_id; old = self.graphs[gid]
        state = flush.new_snapshot.host_state; changes = flush.changes
        graph = GraphAdapter(state, name=gid)
        if self.executor.workspace_bytes + portfolio_request_bytes(graph) > self.arena.budget:
            raise MemoryError('Current portfolio request exceeds the shared arena')
        topology = any(c.base_weight is None or c.final_weight is None for c in changes)
        deletion = any(c.final_weight is None for c in changes)
        eligible, stale = set(), set()
        allowed = {'row', 'col', 'out_row', 'out_col', 'weight', 'source_sum'}
        catalog = {name: self.semantic[name] for name in self.model.components}
        for name in self._graph_components(gid):
            key = catalog[name]; c = self.model.components[name]
            affected = topology or key[1] in ('weight', 'out_weight', 'source_sum') and key[2] is not None
            if affected or not self.model.valid(name): stale.add(name)
            if affected and not deletion and self.model.valid(name) and key[1] in allowed and \
                    self.model.allocations[c.allocation_id].owners == {name}: eligible.add(name)
        if eligible and self.profile is None and self.strategy in ('closed_form','bidirectional_greedy','exhaustive'):
            self.profile = calibrate(self, self.historical_profile)
        queries = []
        for g, f, a, source in self.batch_requests[self.batch_index:]:
            keys = self._requirements(graph if g == gid else self.graphs[g], f, a)
            for key in keys: catalog[token(key)] = key
            queries.append(Query(tuple(token(key) for key in keys), label=f+':'+str(a)))
        components = []
        has_insertions = any(c.base_weight is None for c in changes)
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
        new_count = sum(c.base_weight is None for c in changes)
        source_rows = len({c.src for c in changes})
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

    def _adopt(self, flush, allow_advance):
        if self.strategy == 'ordinary' or not allow_advance or self.batch_requests is None:
            return super()._adopt(flush, allow_advance)
        started = time.perf_counter()
        problem = self._problem(flush)
        chosen = plan(problem, self.strategy)
        if not chosen['feasible']: raise MemoryError(chosen['reason'])
        for name in chosen['retired_stale'] + chosen['maintenance_victims']:
            if name in self.model.components: self._drop(name)
        elapsed = time.perf_counter()-started
        self.counters['recovery_planning_ns'] += int(elapsed*1e9)
        self.counters['recovery_plan_calls'] += 1
        self.counters['recovery_candidates'] += chosen['evaluations']
        self.counters['recovery_components_avoided'] += len(problem.eligible_names)-len(chosen['selected'])
        chosen.update(counter_before_h2d=self.counters['h2d_bytes'],
                      counter_before_builds=self.counters['view_builds'], batch_index=self.batch_index)
        detail = super()._adopt(flush, bool(chosen['selected']))
        if chosen['selected'] and detail['path'] != 'incremental':
            raise AssertionError('Admitted recovery plan did not execute its selected action')
        if chosen['selected'] and detail['packet_bytes'] != chosen['packet_payload_bytes']:
            raise AssertionError('Packet size differs from scalar physical plan')
        chosen.update(graph=flush.ref.graph_id, logical_version=flush.new_snapshot.version,
                      complete_planning_seconds=elapsed, actual_maintenance=detail)
        self.recovery_plans.append(chosen)
        return dict(detail, recovery_plan_index=len(self.recovery_plans)-1,
                    recovery_planning_seconds=elapsed, selected_components=len(chosen['selected']))

    def stats(self):
        return dict(super().stats(), recovery_strategy=self.strategy,
                    eviction='current-declared-epoch next use; recency then physical release tie break',
                    calibration=self.profile)
