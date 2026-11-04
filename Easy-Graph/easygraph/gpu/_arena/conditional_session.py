"""Independent v4 conditional-cost session; frozen v3 execution is inherited.

Both selections use the same component LRU, repair semantics, CUDA kernels,
output policy and legal construction alternatives. Only post-state valuation
differs. Costs must be supplied explicitly; unknown host preparation is rejected.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import time

import numpy as np

from easygraph.gpu._arena.conditional_cost import (Component as Definition, CostProfile, FutureRequest,
                              Recipe, Resolver, State, retention_loss)
from easygraph.gpu._arena.state_model import ComponentSpec
from easygraph.gpu._arena.session_manager import GraphSession, request_boundary, token, cp, Arena


@dataclass
class SessionCosts:
    transfer: CostProfile
    # Keys are semantic (graph, kind, attribute), and (graph, attribute).
    host_build_seconds: dict
    normalize_seconds: dict
    # Called with actual dirty edge/row/row-edge counts and current host readiness.
    repair_cpu_estimator: object
    provenance: dict
    # Explicit measured gather(outgoing->incoming)+normalize cost; optional.
    permute_normalize_seconds: dict = field(default_factory=dict)
    mapping_upload_seconds: dict = field(default_factory=dict)
    upload_seconds: dict = field(default_factory=dict)
    # Freshly materialized host output may have a different measured transfer
    # cost from a retained host array. Keep that source state explicit.
    cold_upload_seconds: dict = field(default_factory=dict)
    normalize_inplace_seconds: dict = field(default_factory=dict)

    @staticmethod
    def host_ready(graph, key):
        return key[1] in ('row', 'col', 'out_row', 'weight', 'in_to_out') or key in graph.arrays

    def host_extra(self, graph, key):
        if self.host_ready(graph, key):
            return 0.0
        if key not in self.host_build_seconds:
            raise ValueError('Missing explicit host-build cost: ' + str(key))
        return float(self.host_build_seconds[key])

    def upload_total(self, graph, key, payload, *, host_ready=True):
        if not host_ready and key in self.cold_upload_seconds:
            return self.cold_upload_seconds[key]
        if key in self.upload_seconds:
            return self.upload_seconds[key]
        if key[1] == 'in_to_out' and graph.name in self.mapping_upload_seconds:
            return self.mapping_upload_seconds[graph.name]
        return self.transfer.transfer(payload)

    def host_total(self, graph, key, payload):
        return self.host_extra(graph, key) + self.upload_total(
            graph, key, payload, host_ready=self.host_ready(graph, key))

    def repair_total(self, graph, key, positions):
        features = dict(entries=len(positions), rows=0, scanned_row_edges=0,
                        host_ready=self.host_ready(graph, key), kind=key[1])
        if key[1] == 'norm' and not features['host_ready'] and len(positions):
            rows = np.unique(graph.col[positions])
            features['rows'] = len(rows)
            features['scanned_row_edges'] = int(np.sum(graph.out_row[rows+1]-graph.out_row[rows], dtype=np.int64))
        cpu = float(self.repair_cpu_estimator(graph.name, key[2], features))
        if cpu < 0 or not np.isfinite(cpu):
            raise ValueError('Invalid host repair extraction cost')
        return self.transfer.patch(len(positions)) + cpu


@dataclass
class Candidate:
    label: str
    plan: object
    gpu_normalize: bool = False
    raw_name: str | None = None
    mapping_name: str | None = None
    upload_raw_to_target: bool = False


def shape_dtype(graph, key):
    kind = key[1]
    size = graph.n if kind == 'dangling' else graph.n+1 if kind in ('row', 'out_row') else len(graph.col)
    dtype = np.uint8 if kind == 'dangling' else np.int32 if kind in ('row', 'col', 'out_row', 'out_col', 'in_to_out') else np.float64
    return (size,), dtype


def host_array(graph, key):
    return graph.in_to_out if key[1] == 'in_to_out' else graph.get(key)


def permute_normalize(executor, graph, col, outgoing, permutation, destination, dangling):
    """Two-stage conversion with distinct source and destination allocations.

    The incoming destination is temporary raw storage until degree finishes,
    then normalized in place. The mapping and all arrays are arena residents.
    """
    if outgoing.data.ptr == destination.data.ptr:
        raise ValueError('Different edge orders require distinct source and destination')
    m = len(graph.col)
    if m:
        executor.kernels['gather'](((m+127)//128,), (128,),
            (np.int32(m), permutation, outgoing, destination))
    executor.normalize(graph, col, destination, destination, dangling)


def plan_lru_overwrite(model, specs, source, target, workspace_bytes, view_id):
    """Add legal destructive conversion without changing the shared LRU policy.

    The frozen planner only enables its native overwrite branch under adaptive.
    Plan an explicitly authorized physical alias, then exchange its semantic
    owner in the private candidate. The executor performs the conversion before
    commit; no user-visible state ever sees two different semantics as aliases.
    """
    entry = model.components.get(source)
    if entry is None or not model.valid(source):
        return None
    allocation = model.allocations[entry.allocation_id]
    target_spec = next(s for s in specs if s.key == target)
    if (allocation.pins or allocation.pending_release or allocation.owners != {source}
            or allocation.capacity_bytes < target_spec.capacity_bytes or source in {s.key for s in specs}):
        return None
    aliased = [replace(s, allocation_id=entry.allocation_id) if s.key == target else s for s in specs]
    plan = model.plan(aliased, workspace_bytes=workspace_bytes, view_id=view_id)
    if not plan.feasible:
        return plan
    action = next((a for a in plan.actions if a['kind'] == 'alias' and a['key'] == target), None)
    if action is None:
        return None
    plan._state.retire(source)
    plan._state.assert_invariants()
    action.update(kind='overwrite', source=source, capacity_bytes=allocation.capacity_bytes)
    plan.evict.append(source)
    plan.estimated_cost += target_spec.build_cost
    return plan


def conditional_problem(session, semantic_keys):
    """Snapshot all requested source closures without constructing host arrays."""
    keys = set(semantic_keys)
    # A PR representation can legally be recreated from matching incoming raw.
    for key in list(keys):
        if key[1] in ('norm', 'dangling'):
            graph = session.graphs[key[0]]
            keys.update(graph.key(kind, key[2] if kind in ('weight', 'norm', 'dangling') else None)
                        for kind in ('row', 'col', 'weight', 'norm', 'dangling'))
            if (key[0], key[2]) in session.costs.permute_normalize_seconds:
                keys.update((graph.key('out_weight', key[2]), graph.key('in_to_out')))
    if len(keys) > 64:
        raise ValueError('Conditional source closure exceeds the declared 64-component bound')
    definitions, recipes, semantics = {}, [], {}
    before = State.from_model(session.model)
    for key in sorted(keys, key=str):
        graph = session.graphs[key[0]]
        name = token(key)
        shape, dtype = shape_dtype(graph, key)
        definition = Definition(name, Arena.capacity(shape, dtype), graph.dependency(key))
        definitions[name], semantics[name] = definition, key
        payload = int(np.prod(shape))*np.dtype(dtype).itemsize
        host_key = 'host:' + name
        if session.costs.host_ready(graph, key):
            before.host_sources[host_key] = before.stamp(definition)
        recipes.append(Recipe('upload:' + name, (name,), host_inputs=(host_key,),
                              cost_seconds=session.costs.upload_total(graph, key, payload)))
        if not session.costs.host_ready(graph, key):
            # HostGraph stores raw attributes and canonical topology. This source
            # represents those host inputs at their current dependency versions.
            source = 'host-primary:' + name
            before.host_sources[source] = before.stamp(definition)
            cpu = session.costs.host_extra(graph, key)
            recipes.append(Recipe('host_build:' + name, (name,), host_inputs=(source,),
                                  host_outputs=((host_key, definition.dependencies),),
                                  cost_seconds=cpu+session.costs.upload_total(graph, key, payload, host_ready=False),
                                  cost_terms=(('host_build', cpu), ('h2d', session.costs.upload_total(graph, key, payload, host_ready=False)))))
        if key[1] in ('weight', 'out_weight', 'norm'):
            dep = graph.name + ':attr:' + str(key[2])
            offset = session.model.versions.get(dep, 0)-graph.versions[key[2]]
            memo = {}

            def positions(state, name=name, graph=graph, key=key, dep=dep, offset=offset, memo=memo):
                entry = state.residents.get(name)
                if entry is None:
                    return None
                stamp = dict(entry.stamp)
                if any(stamp.get(d, 0) != state.versions.get(d, 0) for d in graph.dependency(key) if d != dep):
                    return None
                old = (graph.generation, stamp.get(dep, 0)-offset)
                if old not in memo:
                    memo[old] = graph.dirty_positions(key, old)
                return memo[old]

            recipes.append(Recipe('repair:' + name, (name,), kind='repair', repair_dependencies=(dep,),
                enabled_when=lambda state, fn=positions: fn(state) is not None,
                cost_seconds=lambda state, fn=positions, graph=graph, key=key:
                    session.costs.repair_total(graph, key, fn(state))))
    for name, key in semantics.items():
        if key[1] != 'norm' or (key[0], key[2]) not in session.costs.normalize_seconds:
            continue
        graph = session.graphs[key[0]]
        raw, col, mask = (token(graph.key('weight', key[2])), token(graph.key('col')),
                          token(graph.key('dangling', key[2])))
        for overwrite in (False, True):
            recipes.append(Recipe('normalize:' + name + (':overwrite' if overwrite else ':retain'),
                                  (name, mask), (raw, col), kind='device',
                                  cost_seconds=(session.costs.normalize_inplace_seconds.get((key[0], key[2]),
                                      session.costs.normalize_seconds[(key[0], key[2])]) if overwrite else
                                      session.costs.normalize_seconds[(key[0], key[2])]),
                                  overwrite=((name, raw),) if overwrite else ()))
    for name, key in semantics.items():
        if key[1] != 'norm' or (key[0], key[2]) not in session.costs.normalize_inplace_seconds:
            continue
        graph = session.graphs[key[0]]
        raw_key=graph.key('weight',key[2])
        cpu_upload=session.costs.upload_total(graph,raw_key,len(graph.col)*8)
        gpu=session.costs.normalize_inplace_seconds[key[0],key[2]]
        recipes.append(Recipe('upload_inplace:' + name,
            (name,token(graph.key('dangling',key[2]))), (token(graph.key('col')),),
            host_inputs=('host:'+token(raw_key),), kind='device',
            cost_seconds=cpu_upload+gpu,cost_terms=(('h2d',cpu_upload),('normalize_inplace',gpu))))
    for name, key in semantics.items():
        if key[1] != 'norm' or (key[0], key[2]) not in session.costs.permute_normalize_seconds:
            continue
        graph = session.graphs[key[0]]
        recipes.append(Recipe('permute_normalize:' + name,
            (name, token(graph.key('dangling', key[2]))),
            (token(graph.key('out_weight', key[2])), token(graph.key('in_to_out')), token(graph.key('col'))),
            kind='device', cost_seconds=session.costs.permute_normalize_seconds[(key[0], key[2])]))
    return Resolver(list(definitions.values()), recipes, beam_width=8, max_expansions=128), before, semantics


class ConditionalGraphSession(GraphSession):
    def __init__(self, graphs, budget_bytes, *, costs: SessionCosts, selection='greedy',
                 kernel_backend='reference', decision_floor_seconds=0.00075,
                 overhead_multiplier=2.0, max_future_requests=4):
        if selection not in ('greedy', 'joint'):
            raise ValueError(selection)
        if decision_floor_seconds < 0 or overhead_multiplier < 0 or not 1 <= max_future_requests <= 4:
            raise ValueError('Invalid decision bound')
        # repair_lru maps to shared_component_LRU: victim ranking never consults
        # historical build_cost. All selections receive the same alternatives.
        super().__init__(graphs, budget_bytes, policy='repair_lru', kernel_backend=kernel_backend)
        self.costs, self.selection = costs, selection
        self.decision_floor_seconds = decision_floor_seconds
        self.overhead_multiplier, self.max_future_requests = overhead_multiplier, max_future_requests
        self.request_counts = Counter()
        self.decision_ewma_seconds = 0.0
        self.last_decision = None

    @request_boundary
    def query(self, graph_name, function, attr=None, source=0, result='array', selected=None):
        answer = super().query(graph_name, function, attr, source, result, selected)
        # The current request becomes history only after successful completion.
        self.request_counts[(graph_name, function, attr)] += 1
        self.last['conditional_selection'] = self.selection
        return answer

    def _score_candidates(self, candidates, required_keys):
        feasible = [c for c in candidates if c.plan.feasible]
        if not feasible:
            raise MemoryError(candidates[0].plan.reason)
        greedy = min(feasible, key=lambda c: (c.plan.estimated_cost, c.plan.predicted_peak_bytes))
        sources = {}
        for candidate in candidates:
            if candidate.upload_raw_to_target:
                source = 'upload_incoming_raw_to_target'
            elif not candidate.gpu_normalize:
                source = 'host'
            elif candidate.raw_name in candidate.plan.admit:
                source = 'upload_incoming_raw'
            elif candidate.raw_name in candidate.plan.repair:
                source = 'repair_incoming_raw'
            else:
                source = 'resident_incoming_raw'
            if candidate.mapping_name is not None:
                source = source.replace('incoming', 'outgoing')
            sources[candidate.label] = source
            self.counters['conditional_candidates_proposed'] += 1
            if candidate.plan.feasible:
                self.counters['conditional_candidates_feasible'] += 1
                self.counters['conditional_candidates_' + source] += 1
        self.last_decision = dict(selection=self.selection, selected=greedy.label,
                                  candidates={c.label: c.plan.estimated_cost for c in feasible},
                                  candidate_sources=sources,
                                  candidate_mapping={c.label: ('upload' if c.mapping_name in c.plan.admit else 'reuse')
                                      for c in candidates if c.mapping_name is not None},
                                  candidate_feasibility={c.label:c.plan.feasible for c in candidates},
                                  gate='greedy', scores={})
        if self.selection == 'greedy' or len(feasible) == 1 or not self.request_counts:
            return greedy
        changed = set().union(*(set(c.plan.evict)-set(c.plan.admit) for c in feasible))
        if not changed:
            self.last_decision['gate'] = 'same_retained_state'
            return greedy
        histories = self.request_counts.most_common(self.max_future_requests)
        total = sum(self.request_counts.values())
        relevant, potential = [], 0.0
        all_keys = set(required_keys)
        for request, count in histories:
            graph = self.graphs[request[0]]
            keys = self._requirements(graph, request[1], request[2])
            all_keys.update(keys)
            probability = count/total
            affected = [key for key in keys if token(key) in changed]
            if affected:
                relevant.append(FutureRequest(str(request), tuple(map(token, keys)), probability))
                for key in affected:
                    shape, dtype = shape_dtype(graph, key)
                    payload = int(np.prod(shape))*np.dtype(dtype).itemsize
                    potential += probability*self.costs.host_total(graph, key, payload)
        margin = max(self.decision_floor_seconds, self.overhead_multiplier*self.decision_ewma_seconds)
        self.last_decision.update(potential_seconds=potential, margin_seconds=margin)
        if not relevant or potential <= margin:
            self.last_decision['gate'] = 'restoration_potential_below_margin'
            return greedy
        started = time.perf_counter()
        all_keys.update(self.semantic[k] for k in self.model.components)
        resolver, before, semantics = conditional_problem(self, all_keys)
        scored = []
        for candidate in feasible:
            post = State.from_model(candidate.plan._state)
            post.host_sources = before.host_sources.copy()
            for action in candidate.plan.actions:
                if action['kind'] != 'admit':
                    continue
                name = action['key']
                key = self.semantic[name]
                if candidate.gpu_normalize and key[1] in ('norm', 'dangling'):
                    continue
                definition = resolver.components[name]
                post.host_sources['host:' + name] = post.stamp(definition)
            loss = retention_loss(resolver, before, post, relevant, self.arena.budget)
            if not loss['feasible'] or loss['truncated']:
                self.counters['conditional_unresolved'] += 1
                continue
            score = candidate.plan.estimated_cost + loss['cost_seconds']
            scored.append((score, candidate))
            self.last_decision['scores'][candidate.label] = dict(score_seconds=score, loss=loss)
        elapsed = time.perf_counter()-started
        self.counters['conditional_decision_ns'] += int(elapsed*1e9)
        self.decision_ewma_seconds = elapsed if self.decision_ewma_seconds == 0 else .8*self.decision_ewma_seconds+.2*elapsed
        self.last_decision['decision_seconds'] = elapsed
        greedy_score = next((s for s, c in scored if c is greedy), None)
        if greedy_score is None or not scored:
            self.last_decision['gate'] = 'unresolved_use_known_feasible'
            return greedy
        score, winner = min(scored, key=lambda item: item[0])
        if greedy_score-score <= margin:
            self.last_decision['gate'] = 'predicted_gain_below_margin'
            return greedy
        self.last_decision.update(gate='joint', selected=winner.label,
                                  predicted_gain_seconds=greedy_score-score)
        self.counters['conditional_joint_choices'] += 1
        return winner

    def prepare(self, graph, function, attr):
        started = time.perf_counter()
        keys = self._requirements(graph, function, attr)
        view = token((graph.name, function, attr))
        names = {key: token(key) for key in keys}
        if self.model.touch_if_valid(list(names.values()), workspace_bytes=self.executor.workspace_bytes, view_id=view):
            self.counters['planning_ns'] += int((time.perf_counter()-started)*1e9)
            self.counters['hits'] += len(names)
            self.counters['path_reuse'] += 1
            self.counters['conditional_fast_hits'] += 1
            self.last_decision = dict(selection=self.selection, selected='reuse', gate='valid_hit',
                                      candidates={}, candidate_sources={}, candidate_feasibility={}, scores={})
            return {key: self.arena.slots[self.model.components[name].allocation_id].array
                    for key, name in names.items()}, names
        self.counters['conditional_misses'] += 1
        all_keys = list(keys)
        can_normalize = (function == 'pr' and attr is not None
                         and (graph.name, attr) in self.costs.normalize_seconds)
        if can_normalize:
            all_keys.append(graph.key('weight', attr))
        can_permute = (function == 'pr' and attr is not None
                       and (graph.name, attr) in self.costs.permute_normalize_seconds)
        if can_permute:
            all_keys.extend((graph.key('out_weight', attr), graph.key('in_to_out')))
        shapes, dtypes, specs, repair_positions = {}, {}, {}, {}
        for key in dict.fromkeys(all_keys):
            name = token(key)
            self.semantic[name] = key
            shapes[name], dtypes[name] = shape_dtype(graph, key)
            capacity = Arena.capacity(shapes[name], dtypes[name])
            payload = int(np.prod(shapes[name]))*np.dtype(dtypes[name]).itemsize
            old = self.device_versions.get(name)
            positions = graph.dirty_positions(key, old) if old is not None else None
            repairable = key[1] in ('weight', 'out_weight', 'norm') and positions is not None
            repair_positions[name] = positions
            specs[name] = ComponentSpec(name, capacity, graph.dependency(key),
                build_cost=self.costs.host_total(graph, key, payload),
                repairable=repairable,
                repair_cost=self.costs.repair_total(graph, key, positions) if repairable else 0.0,
                repair_dependencies=(graph.name+':attr:'+str(key[2]),))
        requested_specs = [specs[name] for name in names.values()]
        plan = self.model.plan(requested_specs, workspace_bytes=self.executor.workspace_bytes, view_id=view)
        candidates = [Candidate('host', plan)]
        if function=='pr' and attr is not None and (graph.name,attr) in self.costs.normalize_inplace_seconds:
            norm,mask=token(graph.key('norm',attr)),token(graph.key('dangling',attr))
            if norm in plan.admit or norm in plan.repair or not plan.feasible:
                cost=(self.costs.upload_total(graph,graph.key('weight',attr),len(graph.col)*8)
                      + self.costs.normalize_inplace_seconds[graph.name,attr])
                inplace_specs=[replace(s,build_cost=cost,repairable=False) if s.key==norm
                    else replace(s,build_cost=0.0) if s.key==mask else s for s in requested_specs]
                inplace=self.model.plan(inplace_specs,workspace_bytes=self.executor.workspace_bytes,view_id=view)
                candidates.append(Candidate('upload_inplace',inplace,True,norm,upload_raw_to_target=True))
        if can_normalize:
            raw = token(graph.key('weight', attr))
            norm, mask = token(graph.key('norm', attr)), token(graph.key('dangling', attr))
            need_norm = norm in plan.admit or not plan.feasible
            if need_norm:
                gpu_specs = [replace(s, build_cost=self.costs.normalize_seconds[(graph.name, attr)]) if s.key == norm
                             else replace(s, build_cost=0.0) if s.key == mask else s for s in requested_specs]
                retain = self.model.plan(gpu_specs+[specs[raw]], workspace_bytes=self.executor.workspace_bytes, view_id=view)
                candidates.append(Candidate('device_retain', retain, True, raw))
                if raw in self.model.components and self.model.valid(raw):
                    overwrite_specs=[replace(s,build_cost=self.costs.normalize_inplace_seconds.get(
                        (graph.name,attr),s.build_cost)) if s.key==norm else s for s in gpu_specs]
                    overwrite = plan_lru_overwrite(self.model, overwrite_specs, raw, norm,
                                                    self.executor.workspace_bytes, view)
                    if overwrite is not None:
                        candidates.append(Candidate('device_overwrite', overwrite, True, raw))
        if can_permute:
            raw = token(graph.key('out_weight', attr))
            mapping = token(graph.key('in_to_out'))
            norm, mask = token(graph.key('norm', attr)), token(graph.key('dangling', attr))
            if norm in plan.admit or not plan.feasible:
                gpu_specs = [replace(s, build_cost=self.costs.permute_normalize_seconds[(graph.name, attr)])
                             if s.key == norm else replace(s, build_cost=0.0) if s.key == mask else s
                             for s in requested_specs]
                mapped = self.model.plan(gpu_specs+[specs[raw], specs[mapping]],
                                         workspace_bytes=self.executor.workspace_bytes, view_id=view)
                candidates.append(Candidate('device_permute', mapped, True, raw, mapping))
        candidate = self._score_candidates(candidates, all_keys)
        self.counters['conditional_gate_' + self.last_decision['gate']] += 1
        self.counters['conditional_selected_' + self.last_decision['candidate_sources'][candidate.label]] += 1
        self.counters['planning_ns'] += int((time.perf_counter()-started)*1e9)
        self._execute_candidate(candidate, graph, shapes, dtypes, repair_positions)
        arrays = {key: self.arena.slots[self.model.components[name].allocation_id].array
                  for key, name in names.items()}
        return arrays, names

    def _execute_candidate(self, candidate, graph, shapes, dtypes, repairs):
        plan = candidate.plan
        # Plan/source revision is checked before any side effect, then again at commit.
        if plan.base_revision != self.model.revision:
            raise RuntimeError('Conditional candidate became stale before execution')
        raw_allocation = (self.model.components[candidate.raw_name].allocation_id
                          if candidate.raw_name in self.model.components else None)
        for action in plan.actions:
            kind = action['kind']
            if kind == 'release':
                self.arena.release(action['allocation_id'])
            elif kind == 'evict':
                self.device_versions.pop(action['key'], None)
                self.counters['evictions'] += 1
            elif kind == 'admit':
                name, aid = action['key'], action['allocation_id']
                key = self.semantic[name]
                try:
                    array = self.arena.allocate(aid, shapes[name], dtypes[name])
                except MemoryError:
                    if any(a.pins for a in self.model.allocations.values()):
                        raise
                    self.arena.compact()
                    array = self.arena.allocate(aid, shapes[name], dtypes[name])
                upload_target=candidate.upload_raw_to_target and key[1]=='norm'
                if upload_target or not (candidate.gpu_normalize and key[1] in ('norm', 'dangling')):
                    t = time.perf_counter()
                    data = host_array(graph, graph.key('weight',key[2]) if upload_target else key)
                    self.counters['conditional_host_build_ns'] += int((time.perf_counter()-t)*1e9)
                    array.set(data)
                    self.counters['h2d_bytes'] += data.nbytes
                    if key[1] == 'in_to_out':
                        self.counters['permutation_h2d_bytes'] += data.nbytes
                self.device_versions[name] = graph.version(key)
                self.counters['view_builds'] += 1
                if name == candidate.raw_name:
                    raw_allocation = aid
            elif kind == 'repair':
                name = action['key']
                key, positions = self.semantic[name], repairs[name]
                t = time.perf_counter()
                values = graph.repair_values(key, positions)
                self.counters['conditional_host_repair_ns'] += int((time.perf_counter()-t)*1e9)
                self.executor.patch(self.arena.slots[action['allocation_id']].array, positions, values)
                self.counters['h2d_bytes'] += len(positions)*12
                self.counters['repairs'] += 1
                self.counters['repaired_values'] += len(positions)
                self.device_versions[name] = graph.version(key)
            elif kind == 'reuse':
                self.counters['hits'] += 1
            elif kind == 'overwrite':
                self.device_versions.pop(action['source'], None)
                self.device_versions[action['key']] = graph.version(self.semantic[action['key']])
                self.counters['overwrites'] += 1
            elif kind not in ('workspace_resize', 'drop_view'):
                raise RuntimeError(('Unsupported conditional plan action', action))
        if candidate.gpu_normalize:
            attr = self.semantic[candidate.raw_name][2]
            def array(kind, attribute=None):
                name = token(graph.key(kind, attribute))
                return self.arena.slots[plan._state.components[name].allocation_id].array
            if candidate.mapping_name is not None:
                permute_normalize(self.executor, graph, array('col'), self.arena.slots[raw_allocation].array,
                                  array('in_to_out'), array('norm', attr), array('dangling', attr))
                self.counters['permutation_conversions'] += 1
                self.counters['permutation_gather_bytes'] += 4*len(graph.col)+16*len(graph.col)
            else:
                self.executor.normalize(graph, array('col'), self.arena.slots[raw_allocation].array,
                                        array('norm', attr), array('dangling', attr))
            self.counters['device_conversions'] += 1
            self.counters['avoided_h2d_bytes'] += 8*len(graph.col)+graph.n
        cp.cuda.runtime.deviceSynchronize()
        self.model.commit(plan)
        if self.arena.live != self.model.live_bytes:
            raise AssertionError('Conditional execution capacity/ledger mismatch')
        self.counters['path_'+candidate.label] += 1

    def stats(self):
        result = super().stats()
        result.update(conditional_selection=self.selection,
                      conditional_cost_provenance=self.costs.provenance,
                      conditional_decision_ewma_seconds=self.decision_ewma_seconds,
                      conditional_history_requests=sum(self.request_counts.values()))
        return result
