"""Bounded conditional reconstruction planning, independent of the v3 runtime.

Historical construction costs are deliberately absent from resident entries.
Every recipe describes a presently executable construction/repair path.  Inputs
must be current before execution; alternatives are evaluated on candidate states.
This is a bounded heuristic with explicit truncation, not a global optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import ceil, isfinite
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Component:
    key: str
    capacity_bytes: int
    dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        if self.capacity_bytes < 0 or len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError('Invalid component capacity or dependencies')


@dataclass(frozen=True)
class Resident:
    allocation_id: str
    stamp: tuple[tuple[str, int], ...]
    # None means unknown/full change scope; missing dependency means unchanged.
    dirty: tuple[tuple[str, frozenset | None], ...] = ()


@dataclass(frozen=True)
class Allocation:
    capacity_bytes: int
    pins: int = 0
    pending_release: bool = False


@dataclass
class State:
    residents: dict[str, Resident]
    allocations: dict[str, Allocation]
    versions: dict[str, int]
    workspace_bytes: int = 0
    serial: int = 0
    # Host representations carry semantic versions but do not consume GPU arena
    # capacity. Their real host memory remains a separately reported metric.
    host_sources: dict[str, tuple[tuple[str, int], ...]] = field(default_factory=dict)

    @property
    def live_bytes(self):
        return self.workspace_bytes + sum(a.capacity_bytes for a in self.allocations.values())

    def copy(self):
        # Entries and dirty scopes are immutable, so their identities may be shared.
        return State(self.residents.copy(), self.allocations.copy(), self.versions.copy(),
                     self.workspace_bytes, self.serial, self.host_sources.copy())

    def owners(self, allocation_id):
        return {k for k, entry in self.residents.items() if entry.allocation_id == allocation_id}

    def stamp(self, component):
        return tuple((d, self.versions.get(d, 0)) for d in component.dependencies)

    def valid(self, component):
        entry = self.residents.get(component.key)
        if entry is None:
            return False
        alloc = self.allocations.get(entry.allocation_id)
        return (alloc is not None and not alloc.pending_release
                and alloc.capacity_bytes >= component.capacity_bytes
                and dict(entry.stamp) == dict(self.stamp(component)))

    def remove(self, key):
        entry = self.residents[key]
        alloc = self.allocations[entry.allocation_id]
        if alloc.pins or alloc.pending_release:
            raise ValueError('Cannot release a pinned or pending allocation')
        del self.residents[key]
        if not self.owners(entry.allocation_id):
            del self.allocations[entry.allocation_id]

    @classmethod
    def from_model(cls, model):
        """Snapshot the existing StateModel without importing CUDA or mutating it."""
        entries = {}
        for key, value in model.components.items():
            dirty = model.repair_scope(key) if not model.valid(key) else {}
            entries[key] = Resident(value.allocation_id, tuple(value.stamp.items()),
                                    tuple((d, None if scope is None else frozenset(scope))
                                          for d, scope in dirty.items()))
        return cls(entries, {key: Allocation(a.capacity_bytes, a.pins, a.pending_release)
                             for key, a in model.allocations.items()}, model.versions.copy(),
                   model.workspace_capacity, model.serial)


CostFunction = Callable[[State], float]


@dataclass(frozen=True)
class Recipe:
    name: str
    outputs: tuple[str, ...]
    inputs: tuple[str, ...] = ()
    cost_seconds: float | CostFunction = 0.0
    scratch_bytes: int = 0
    kind: str = 'host'
    # A repair updates its single existing output allocation in place.
    repair_dependencies: tuple[str, ...] = ()
    # Mapping output key -> unique source whose allocation may be overwritten.
    overwrite: tuple[tuple[str, str], ...] = ()
    enabled_when: Callable[[State], bool] | None = None
    host_inputs: tuple[str, ...] = ()
    host_outputs: tuple[tuple[str, tuple[str, ...]], ...] = ()
    cost_terms: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        if not self.outputs or len(set(self.outputs)) != len(self.outputs):
            raise ValueError('Recipe requires unique outputs')
        if self.scratch_bytes < 0 or self.kind not in {'host', 'device', 'repair'}:
            raise ValueError('Invalid recipe scratch/kind')
        if self.kind == 'repair' and (len(self.outputs) != 1 or self.overwrite):
            raise ValueError('Repair has one existing destination')
        pairs = dict(self.overwrite)
        if (len(pairs) != len(self.overwrite) or len(set(pairs.values())) != len(pairs)
                or not set(pairs) <= set(self.outputs) or not set(pairs.values()) <= set(self.inputs)):
            raise ValueError('Overwrite destinations/sources must be unique declared outputs/inputs')

    def cost(self, state):
        value = self.cost_seconds(state) if callable(self.cost_seconds) else self.cost_seconds
        value = float(value)
        if value < 0 or not isfinite(value):
            raise ValueError('Recipe cost must be finite and nonnegative')
        return value


@dataclass
class Plan:
    state: State
    cost_seconds: float = 0.0
    peak_bytes: int = 0
    actions: tuple[dict, ...] = ()


@dataclass
class Result:
    plans: list[Plan]
    expansions: int
    truncated: bool
    reason: str

    @property
    def best(self):
        return self.plans[0] if self.plans else None


class Resolver:
    """Resolve a bounded recipe graph using a beam of distinct resulting states.

    Fixed mode is an executable current-prepare plan: no implicit eviction.
    reclaim_unrelated mode releases unrelated, unpinned components as needed.
    The latter estimates future restoration with relaxed unrelated retention;
    actions expose every release but exclude further future opportunity loss.
    """
    def __init__(self, components: Sequence[Component], recipes: Sequence[Recipe], *,
                 beam_width=8, max_expansions=256, max_depth=12,
                 max_components=64, max_recipes_per_target=8,
                 snapshot_guard: Callable[[], None] | None = None):
        self.components = {c.key: c for c in components}
        if len(self.components) != len(components) or len(components) > max_components:
            raise ValueError('Duplicate component or component bound exceeded')
        if min(beam_width, max_expansions, max_depth) < 1:
            raise ValueError('Search bounds must be positive')
        self.by_output = {key: [] for key in self.components}
        for recipe in recipes:
            if not set(recipe.outputs + recipe.inputs) <= set(self.components):
                raise ValueError('Recipe references an unknown component')
            for key in recipe.outputs:
                self.by_output[key].append(recipe)
        if any(len(v) > max_recipes_per_target for v in self.by_output.values()):
            raise ValueError('Per-target recipe bound exceeded')
        self.beam_width, self.max_expansions, self.max_depth = beam_width, max_expansions, max_depth
        self.snapshot_guard = snapshot_guard

    def resolve(self, targets: Iterable[str], state: State, budget_bytes: int, *,
                reclaim_unrelated=False):
        if self.snapshot_guard is not None:
            self.snapshot_guard()
        targets = tuple(dict.fromkeys(targets))
        if not set(targets) <= set(self.components):
            raise ValueError('Unknown target')
        if budget_bytes < 0 or state.live_bytes > budget_bytes:
            return Result([], 0, False, 'Initial live capacity exceeds budget')
        expansions, truncated = 0, False

        def prune(plans):
            nonlocal truncated
            # Most session choices have <=3 alternatives. Canonicalizing an
            # entire retained ledger for an already bounded beam costs more than
            # keeping an occasional duplicate; deduplicate only under pressure.
            if len(plans) <= self.beam_width:
                return sorted(plans, key=lambda p: (p.cost_seconds, p.peak_bytes))
            # Distinguish retained sources, semantic stamps, alias ownership and capacity.
            unique = {}
            for plan in plans:
                signature = (tuple(sorted((k, e.allocation_id, e.stamp) for k, e in plan.state.residents.items())),
                             tuple(sorted((k, a.capacity_bytes, a.pins, a.pending_release)
                                          for k, a in plan.state.allocations.items())),
                             tuple(sorted(plan.state.host_sources.items())))
                old = unique.get(signature)
                if old is None or (plan.cost_seconds, plan.peak_bytes) < (old.cost_seconds, old.peak_bytes):
                    unique[signature] = plan
            candidates = sorted(unique.values(), key=lambda p: (p.cost_seconds, p.peak_bytes))
            if len(candidates) > self.beam_width:
                truncated = True
            return candidates[:self.beam_width]

        def execute(recipe, prior, protected):
            s = prior.state.copy()
            if not all(s.valid(self.components[key]) for key in recipe.inputs):
                return None
            if any(key not in s.host_sources or any(s.versions.get(dep, 0) != version
                   for dep, version in s.host_sources[key]) for key in recipe.host_inputs):
                return None
            if recipe.enabled_when is not None and not recipe.enabled_when(s):
                return None
            if recipe.kind == 'repair':
                key = recipe.outputs[0]
                entry = s.residents.get(key)
                if entry is None or s.valid(self.components[key]):
                    return None
                changed = {d for d in self.components[key].dependencies
                           if dict(entry.stamp).get(d) != s.versions.get(d, 0)}
                dirty = dict(entry.dirty)
                if (set(dict(entry.stamp)) != set(self.components[key].dependencies)
                        or not changed or not changed <= set(recipe.repair_dependencies)
                        or not changed <= set(dirty) or any(dirty[d] is None for d in changed)):
                    return None
                alloc = s.allocations[entry.allocation_id]
                if (alloc.pins or alloc.pending_release or s.owners(entry.allocation_id) != {key}
                        or alloc.capacity_bytes < self.components[key].capacity_bytes):
                    return None
            cost = recipe.cost(s)
            actions = list(prior.actions)
            overwrite = dict(recipe.overwrite)
            source_ids = {}
            for key, source in overwrite.items():
                entry = s.residents[source]
                alloc = s.allocations[entry.allocation_id]
                if (source in targets or source in protected or alloc.pins or alloc.pending_release
                        or s.owners(entry.allocation_id) != {source}
                        or alloc.capacity_bytes < self.components[key].capacity_bytes):
                    return None
                source_ids[key] = entry.allocation_id
            if recipe.kind != 'repair':
                for key in recipe.outputs:
                    if key in s.residents:
                        if key in recipe.inputs:
                            return None
                        try:
                            old_id = s.residents[key].allocation_id
                            s.remove(key)
                            actions.append(dict(kind='release_output', key=key, allocation_id=old_id))
                        except ValueError:
                            return None
            extra = (0 if recipe.kind == 'repair' else
                     sum(self.components[k].capacity_bytes for k in recipe.outputs if k not in overwrite))
            need = s.live_bytes + extra + recipe.scratch_bytes
            if need > budget_bytes and reclaim_unrelated:
                # Keep current input closure, all already-ready request outputs,
                # and ancestors' required inputs while building their siblings.
                keep = set(targets) | set(protected) | set(recipe.inputs)
                for aid, alloc in list(s.allocations.items()):
                    owners = s.owners(aid)
                    if alloc.pins or alloc.pending_release or not owners or owners & keep:
                        continue
                    for key in sorted(owners):
                        s.remove(key)
                        actions.append(dict(kind='reclaim_unrelated', key=key, allocation_id=aid))
                    need = s.live_bytes + extra + recipe.scratch_bytes
                    if need <= budget_bytes:
                        break
            if need > budget_bytes:
                return None
            peak = max(prior.peak_bytes, need)
            for key in recipe.outputs:
                component = self.components[key]
                if recipe.kind == 'repair':
                    aid = s.residents[key].allocation_id
                elif key in overwrite:
                    source = overwrite[key]
                    aid = source_ids[key]
                    del s.residents[source]
                else:
                    s.serial += 1
                    aid = f'conditional:{s.serial}:{key}'
                    while aid in s.allocations:
                        s.serial += 1
                        aid = f'conditional:{s.serial}:{key}'
                    s.allocations[aid] = Allocation(component.capacity_bytes)
                s.residents[key] = Resident(aid, s.stamp(component))
            for host_key, dependencies in recipe.host_outputs:
                s.host_sources[host_key] = tuple((dep, s.versions.get(dep, 0)) for dep in dependencies)
            actions.append(dict(kind=recipe.kind, recipe=recipe.name, outputs=list(recipe.outputs),
                                inputs=list(recipe.inputs), overwrite=dict(recipe.overwrite),
                                host_inputs=list(recipe.host_inputs), host_outputs=[key for key, _ in recipe.host_outputs],
                                cost_terms=dict(recipe.cost_terms),
                                scratch_bytes=recipe.scratch_bytes, cost_seconds=cost,
                                peak_bytes=need))
            return Plan(s, prior.cost_seconds + cost, peak, tuple(actions))

        def ensure(key, prior, ancestors, protected):
            nonlocal expansions, truncated
            if prior.state.valid(self.components[key]):
                return [prior]
            if key in ancestors or len(ancestors) >= self.max_depth:
                if len(ancestors) >= self.max_depth:
                    truncated = True
                return []
            out = []
            for recipe in self.by_output[key]:
                if expansions >= self.max_expansions:
                    truncated = True
                    break
                expansions += 1
                candidates = [prior]
                for source in recipe.inputs:
                    expanded = []
                    for candidate in candidates:
                        expanded.extend(ensure(source, candidate, ancestors + (key,),
                                               protected | (set(recipe.inputs) - {source})))
                    candidates = prune(expanded)
                    if not candidates:
                        break
                for candidate in candidates:
                    next_plan = execute(recipe, candidate, protected)
                    if next_plan is not None:
                        out.append(next_plan)
            return prune(out)

        plans = [Plan(state.copy(), peak_bytes=state.live_bytes)]
        for key in targets:
            expanded = []
            for prior in plans:
                expanded.extend(ensure(key, prior, (), set()))
            plans = prune(expanded)
            if not plans:
                break
        # A later overwrite must not invalidate an earlier requested output.
        plans = [p for p in plans if all(p.state.valid(self.components[k]) for k in targets)]
        reason = 'feasible' if plans else ('search bound reached' if truncated else 'no legal capacity-feasible recipe')
        return Result(plans, expansions, truncated, reason)


@dataclass(frozen=True)
class FutureRequest:
    name: str
    targets: tuple[str, ...]
    probability: float


def retention_loss(resolver: Resolver, before: State, post: State, requests: Sequence[FutureRequest],
                   budget_bytes: int, *, before_results: Mapping[tuple[str, ...], Result] | None = None):
    """One-step historical-request expectation using post-candidate source states.

    Shared sources are built once inside each logical request.  Requests are
    alternative possible next requests, not a fabricated future sequence.
    Unrelated reclaim is free here. This omits later opportunity losses; bounded
    search and estimated recipe costs do not provide a formal numerical bound.
    """
    if any(r.probability < 0 or not isfinite(r.probability) for r in requests):
        raise ValueError('Invalid request probability')
    if sum(r.probability for r in requests) > 1.0 + 1e-12:
        raise ValueError('One-step probabilities must sum to at most one')
    total, details = 0.0, []
    for request in requests:
        if request.probability == 0:
            continue
        old = (before_results[request.targets] if before_results is not None else
               resolver.resolve(request.targets, before, budget_bytes, reclaim_unrelated=True))
        new = resolver.resolve(request.targets, post, budget_bytes, reclaim_unrelated=True)
        if old.best is None or new.best is None:
            return dict(feasible=False, cost_seconds=None, details=details,
                        reason=f'Restoration unresolved for {request.name}',
                        truncated=old.truncated or new.truncated)
        increase = max(0.0, new.best.cost_seconds - old.best.cost_seconds)
        total += request.probability * increase
        details.append(dict(name=request.name, probability=request.probability,
                            before_seconds=old.best.cost_seconds, post_seconds=new.best.cost_seconds,
                            extra_seconds=increase, weighted_seconds=request.probability * increase,
                            post_actions=list(new.best.actions), truncated=old.truncated or new.truncated))
    return dict(feasible=True, cost_seconds=total, details=details,
                truncated=any(d['truncated'] for d in details),
                interpretation='one-step conditional restoration estimate; unrelated future retention relaxed')


def select_candidate(resolver: Resolver, before: State, candidates: Sequence[Plan],
                     budget_bytes: int, *, future_requests: Sequence[FutureRequest] = (), joint=False,
                     execution_cost: Callable[[Plan], float] | None = None):
    """Same legal prepare candidates for greedy composition and joint selection."""
    scored = []
    baseline = ({r.targets: resolver.resolve(r.targets, before, budget_bytes, reclaim_unrelated=True)
                 for r in future_requests if r.probability} if joint else None)
    for index, candidate in enumerate(candidates):
        execution = float(execution_cost(candidate)) if execution_cost else 0.0
        if execution < 0 or not isfinite(execution):
            raise ValueError('Invalid execution cost')
        loss = (retention_loss(resolver, before, candidate.state, future_requests, budget_bytes,
                               before_results=baseline)
                if joint else dict(feasible=True, cost_seconds=0.0, details=[]))
        if loss['feasible']:
            scored.append(dict(index=index, prepare_seconds=candidate.cost_seconds,
                               execution_seconds=execution, loss=loss,
                               score_seconds=candidate.cost_seconds + execution + loss['cost_seconds']))
    if not scored:
        return None, []
    winner = min(scored, key=lambda item: (item['score_seconds'], candidates[item['index']].peak_bytes))
    return candidates[winner['index']], scored


@dataclass(frozen=True)
class CostProfile:
    """Explicit caller-supplied costs; no implicit claims of measured calibration."""
    h2d_start_seconds: float
    h2d_bytes_per_second: float
    patch_chunk_start_seconds: float
    patch_bytes_per_second: float
    normalize_start_seconds: float
    normalize_weight_bytes_per_second: float
    patch_chunk_entries: int = 8192

    def __post_init__(self):
        if (min(self.h2d_bytes_per_second, self.patch_bytes_per_second,
                self.normalize_weight_bytes_per_second, self.patch_chunk_entries) <= 0
                or min(self.h2d_start_seconds, self.patch_chunk_start_seconds,
                       self.normalize_start_seconds) < 0):
            raise ValueError('Invalid explicit cost profile')

    def transfer(self, nbytes):
        return self.h2d_start_seconds + nbytes / self.h2d_bytes_per_second

    def patch(self, entries):
        return ceil(entries / self.patch_chunk_entries) * self.patch_chunk_start_seconds + 12 * entries / self.patch_bytes_per_second

    def normalize(self, raw_weight_bytes):
        return self.normalize_start_seconds + raw_weight_bytes / self.normalize_weight_bytes_per_second


def build_weight_problem(*, raw_key: str, norm_key: str, dangling_key: str,
                         col_key: str, definitions: Mapping[str, Component], n: int, m: int,
                         attribute_dependency: str, profile: CostProfile,
                         repair_entries: Callable[[str, State], int | None],
                         host_prepare_seconds: Mapping[str, float] | None = None,
                         repair_prepare_seconds: Callable[[str, int, State], float] | None = None,
                         allow_overwrite=True):
    """Session-ready factory for the existing incoming raw->norm path.

    Keys are the session's physical component names, not assumed semantic names.
    The caller supplies actual aligned capacities and current dirty-position
    counts.  Outgoing raw must use a separate layout key and is never implicitly
    accepted as incoming raw.  Other topology/attribute recipes may be appended.
    CostProfile values are supplied explicitly from calibration or configuration.
    """
    keys = (raw_key, norm_key, dangling_key, col_key)
    if len(set(keys)) != 4 or not set(keys) <= set(definitions):
        raise ValueError('Weight problem requires four distinct declared components')
    components = [definitions[k] for k in keys]
    payload = {raw_key: 8*m, norm_key: 8*m, dangling_key: n, col_key: 4*m}
    host_prepare_seconds = host_prepare_seconds or {}
    recipes = [Recipe('upload:' + key, (key,), host_inputs=('host:' + key,),
                      cost_seconds=profile.transfer(payload[key]),
                      cost_terms=(('h2d', profile.transfer(payload[key])),)) for key in keys]
    for key in (norm_key, dangling_key):
        if key in host_prepare_seconds:
            cpu = host_prepare_seconds[key]
            recipes.append(Recipe('host_build_upload:' + key, (key,),
                host_inputs=('host:' + raw_key, 'host:' + col_key),
                host_outputs=(('host:' + key, definitions[key].dependencies),),
                cost_seconds=cpu + profile.transfer(payload[key]),
                cost_terms=(('host_build', cpu), ('h2d', profile.transfer(payload[key])))))
    for key in (raw_key, norm_key):
        def repair_cost(state, target=key):
            count = repair_entries(target, state)
            extra = repair_prepare_seconds(target, count, state) if repair_prepare_seconds else 0.0
            return profile.patch(count) + extra
        recipes.append(Recipe('repair:' + key, (key,), cost_seconds=repair_cost,
                              kind='repair', repair_dependencies=(attribute_dependency,),
                              enabled_when=lambda state, target=key: repair_entries(target, state) is not None))
    kwargs = dict(outputs=(norm_key, dangling_key), inputs=(raw_key, col_key), kind='device',
                  cost_seconds=profile.normalize(8*m))
    recipes.append(Recipe('normalize:retain', **kwargs))
    if allow_overwrite:
        recipes.append(Recipe('normalize:overwrite', overwrite=((norm_key, raw_key),), **kwargs))
    return components, recipes


def session_weight_problem(session, graph, attr, profile: CostProfile, *, model=None,
                           host_prepare_seconds: Mapping[str, float] | None = None,
                           repair_prepare_seconds: Callable[[str, int, State], float] | None = None,
                           allow_overwrite=True, **resolver_limits):
    """Read-only adapter for GraphSession shared physical keys, with no GPU work.

    Returns (resolver, snapshot, targets, semantic_names).  The adapter does not
    call graph.get(), construct weights, or allocate device memory.  A caller may
    pass a candidate StateModel post-state as model.  Only state validity and
    current graph dirty history determine restoration costs, never old spec costs.
    Dirty-position enumeration must be charged to planning by the integrating
    caller; this adapter intentionally leaves the frozen runtime untouched.
    """
    if session.policy not in {'repair_lru', 'cost_size', 'adaptive'}:
        raise ValueError('Adapter is for shared-state strong comparison policies')
    model = session.model if model is None else model
    semantics = {kind: graph.key(kind, attr if kind in ('weight', 'norm', 'dangling') else None)
                 for kind in ('weight', 'norm', 'dangling', 'col', 'row')}
    names = {kind: json.dumps(key, separators=(',', ':')) for kind, key in semantics.items()}
    n, m = graph.n, len(graph.col)
    sizes = {'weight': (m, 'float64'), 'norm': (m, 'float64'), 'dangling': (n, 'uint8'),
             'col': (m, 'int32'), 'row': (n+1, 'int32')}
    definitions = {names[kind]: Component(names[kind], session.arena.capacity((size,), dtype),
                                          graph.dependency(semantics[kind]))
                   for kind, (size, dtype) in sizes.items()}
    by_name = {names[kind]: semantics[kind] for kind in names}
    host_prepare_seconds = host_prepare_seconds or {}
    for kind in ('norm', 'dangling'):
        if semantics[kind] not in getattr(graph, 'arrays', {}) and kind not in host_prepare_seconds:
            raise ValueError('Supply current host preparation cost for missing ' + kind + ' view')
    attribute_dependency = graph.name + ':attr:' + attr
    attribute_offset = model.versions.get(attribute_dependency, 0) - graph.versions[attr]
    fingerprint = (graph.generation, graph.versions[attr], graph.n, len(graph.col),
                   tuple((dep, model.versions.get(dep, 0)) for dep in graph.dependency(semantics['weight'])))
    cache = {}

    def guard():
        current = (graph.generation, graph.versions[attr], graph.n, len(graph.col),
                   tuple((dep, model.versions.get(dep, 0)) for dep in graph.dependency(semantics['weight'])))
        if current != fingerprint:
            raise RuntimeError('Conditional-cost snapshot expired; rebuild it after an update boundary')

    def repair_count(key, state):
        resident = state.residents.get(key)
        if resident is None:
            return None
        stamp = dict(resident.stamp)
        # Topology/mapping are never repaired by a weight patch.
        if any(stamp.get(d, 0) != state.versions.get(d, 0)
               for d in definitions[key].dependencies if d != attribute_dependency):
            return None
        old_version = (graph.generation, stamp.get(attribute_dependency, 0) - attribute_offset)
        cache_key = (key, old_version)
        if cache_key not in cache:
            positions = graph.dirty_positions(by_name[key], old_version)
            cache[cache_key] = None if positions is None else len(positions)
        return cache[cache_key]

    components, recipes = build_weight_problem(
        raw_key=names['weight'], norm_key=names['norm'], dangling_key=names['dangling'],
        col_key=names['col'], definitions=definitions, n=n, m=m,
        attribute_dependency=attribute_dependency, profile=profile,
        repair_entries=repair_count, allow_overwrite=allow_overwrite,
        host_prepare_seconds={names[kind]: cost for kind, cost in host_prepare_seconds.items()},
        repair_prepare_seconds=repair_prepare_seconds)
    components.append(definitions[names['row']])
    recipes.append(Recipe('host:' + names['row'], (names['row'],),
                          host_inputs=('host:' + names['row'],), cost_seconds=profile.transfer(4*(n+1))))
    snapshot = State.from_model(model)
    for kind in names:
        if kind in ('row', 'col', 'weight') or semantics[kind] in getattr(graph, 'arrays', {}):
            snapshot.host_sources['host:' + names[kind]] = snapshot.stamp(definitions[names[kind]])
    return (Resolver(components, recipes, snapshot_guard=guard, **resolver_limits), snapshot,
            tuple(names[k] for k in ('row', 'col', 'norm', 'dangling')), names)
