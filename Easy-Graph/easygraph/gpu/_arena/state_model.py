"""CPU reference ledger and online planner for shared graph state.

Capacities are allocator capacities supplied by the caller, not array payloads.
The executor must perform actions in order and synchronously confirm frees before
commit. Existing pending frees remain charged. This module never executes CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ComponentSpec:
    key: str
    capacity_bytes: int
    dependencies: tuple[str, ...] = ()
    build_cost: float = 1.0
    repair_cost: float = 1.0
    repairable: bool = False
    repair_dependencies: tuple[str, ...] | None = None
    overwrite_from: tuple[str, ...] = ()
    prefer_overwrite: bool = False
    # Bind a semantic alias to an existing allocation, or name a new allocation.
    allocation_id: str | None = None

    def __post_init__(self):
        if self.capacity_bytes < 0 or self.build_cost < 0 or self.repair_cost < 0:
            raise ValueError('Capacities and costs must be nonnegative')
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError('Duplicate dependencies')


@dataclass
class Allocation:
    allocation_id: str
    capacity_bytes: int
    owners: set[str] = field(default_factory=set)
    pins: int = 0
    pending_release: bool = False


@dataclass
class Component:
    spec: ComponentSpec
    allocation_id: str
    stamp: dict[str, int]
    last_access: int
    uses: int = 1
    updates_observed: int = 0


@dataclass
class Plan:
    feasible: bool
    reason: str
    actions: list[dict]
    reuse: list[str]
    repair: list[str]
    admit: list[str]
    evict: list[str]
    predicted_peak_bytes: int
    resident_bytes_after: int
    workspace_bytes: int
    estimated_cost: float
    base_revision: int
    _state: 'StateModel | None' = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {k: deepcopy(v) for k, v in self.__dict__.items() if k != '_state'}


class StateModel:
    POLICIES = {'whole_view_LRU', 'shared_component_LRU', 'cost_size', 'adaptive'}

    def __init__(self, budget_bytes: int, policy: str = 'adaptive'):
        if budget_bytes < 0 or policy not in self.POLICIES:
            raise ValueError('Invalid budget or policy')
        self.budget_bytes = budget_bytes
        self.policy = policy
        self.allocations: dict[str, Allocation] = {}
        self.components: dict[str, Component] = {}
        self.views: dict[str, set[str]] = {}
        self.versions: dict[str, int] = {}
        self.update_scopes: dict[str, list[tuple[int, frozenset | None]]] = {}
        self.workspace_capacity = 0
        self.revision = 0
        self.clock = 0
        self.serial = 0

    @property
    def live_bytes(self) -> int:
        return sum(a.capacity_bytes for a in self.allocations.values()) + self.workspace_capacity

    def stamp(self, dependencies: Iterable[str]) -> dict[str, int]:
        return {d: self.versions.get(d, 0) for d in dependencies}

    def _planning_copy(self) -> 'StateModel':
        """Copy mutable ledger metadata, share frozen specs and read-only history.

        Planning never appends update history. Concurrent mutation changes revision
        and invalidates the plan, so sharing history here cannot permit a stale
        commit. This avoids walking every frozen delta and ComponentSpec twice.
        """
        work = StateModel(self.budget_bytes, self.policy)
        work.allocations = {aid: Allocation(aid, a.capacity_bytes, set(a.owners), a.pins,
                                            a.pending_release) for aid, a in self.allocations.items()}
        work.components = {key: Component(c.spec, c.allocation_id, c.stamp.copy(),
                                          c.last_access, c.uses, c.updates_observed)
                           for key, c in self.components.items()}
        work.views = {key: set(members) for key, members in self.views.items()}
        work.versions = self.versions.copy()
        work.update_scopes = self.update_scopes.copy()
        work.workspace_capacity = self.workspace_capacity
        work.revision, work.clock, work.serial = self.revision, self.clock, self.serial
        return work

    def valid(self, key: str, spec: ComponentSpec | None = None) -> bool:
        c = self.components[key]
        expected = spec or c.spec
        return (c.stamp == self.stamp(expected.dependencies)
                and c.spec.dependencies == expected.dependencies
                and self.allocations[c.allocation_id].capacity_bytes >= expected.capacity_bytes)

    def touch_if_valid(self, keys: Sequence[str], workspace_bytes: int = 0,
                       view_id: str | None = None) -> bool:
        """Commit a request whose inputs and execution capacity are already ready.

        The caller holds its request-boundary lock. This path performs no CUDA
        operation or allocation. Validation precedes every metadata change, so a
        miss leaves the ledger untouched and can use the regular planner.
        """
        if workspace_bytes < 0 or len(set(keys)) != len(keys):
            raise ValueError('Invalid ready request')
        if self.policy == 'whole_view_LRU' and view_id is None:
            raise ValueError('whole_view_LRU requires a logical view_id')
        if workspace_bytes > self.workspace_capacity:
            return False
        if any(key not in self.components or not self.valid(key)
               or self.allocations[self.components[key].allocation_id].pending_release
               for key in keys):
            return False
        self.clock += 1
        for key in keys:
            component=self.components[key]
            component.last_access=self.clock
            component.uses += 1
        if view_id is not None:
            self.views[view_id]=set(keys)
        self.revision += 1
        return True

    def update(self, dependency: str, affected_scope: Iterable | None = None) -> int:
        """Commit an update at a request boundary; None means unknown/full scope."""
        if any(dependency in c.stamp and self.allocations[c.allocation_id].pins
               for c in self.components.values()):
            raise ValueError('Update waits for readers of the affected dependency')
        v = self.versions.get(dependency, 0) + 1
        self.versions[dependency] = v
        scope = None if affected_scope is None else frozenset(affected_scope)
        self.update_scopes.setdefault(dependency, []).append((v, scope))
        for c in self.components.values():
            if dependency in c.stamp:
                c.updates_observed += 1
        self.revision += 1
        return v

    def compact_update_log(self) -> None:
        """Keep only updates still needed by some retained stale component."""
        for dep, history in self.update_scopes.items():
            oldest = min((c.stamp[dep] for c in self.components.values() if dep in c.stamp),
                         default=self.versions.get(dep, 0))
            self.update_scopes[dep] = [(v, scope) for v, scope in history if v > oldest]

    def repair_scope(self, key: str) -> dict[str, frozenset | None]:
        """Union all modifications since construction; caller expands transform deps."""
        c = self.components[key]
        result = {}
        for dep, built in c.stamp.items():
            if self.versions.get(dep, 0) == built:
                continue
            scopes = [s for version, s in self.update_scopes[dep] if version > built]
            result[dep] = None if any(s is None for s in scopes) else frozenset().union(*scopes)
        return result

    def pin(self, key: str) -> None:
        self.allocations[self.components[key].allocation_id].pins += 1
        self.revision += 1

    def unpin(self, key: str) -> None:
        a = self.allocations[self.components[key].allocation_id]
        if a.pins == 0:
            raise ValueError('Unbalanced unpin')
        a.pins -= 1
        self.revision += 1

    def retire(self, key: str) -> int:
        """Detach one owner. Unowned memory is pending, not immediately free."""
        c = self.components[key]
        a = self.allocations[c.allocation_id]
        if a.pins:
            raise ValueError('Pinned allocation cannot be retired')
        del self.components[key]
        a.owners.remove(key)
        for members in self.views.values():
            members.discard(key)
        if not a.owners:
            a.pending_release = True
        self.revision += 1
        return a.capacity_bytes if a.pending_release else 0

    def complete_release(self, allocation_id: str) -> int:
        a = self.allocations[allocation_id]
        if a.owners or a.pins or not a.pending_release:
            raise ValueError('Release requires no owners, no pins, and a completed event')
        del self.allocations[allocation_id]
        self.revision += 1
        return a.capacity_bytes

    def marginal_release_bytes(self, keys: Iterable[str]) -> int:
        keys = set(keys)
        return sum(a.capacity_bytes for a in self.allocations.values()
                   if not a.pins and not a.pending_release and a.owners and a.owners <= keys)

    def _retention_value(self, key: str) -> float:
        c = self.components[key]
        age = max(1, self.clock - c.last_access + 1)
        if self.policy in {'shared_component_LRU', 'whole_view_LRU'}:
            return 1.0 / age
        repair_allowed = (c.spec.repairable and (c.spec.repair_dependencies is None
                          or set(self.repair_scope(key)) <= set(c.spec.repair_dependencies)))
        cost = (c.spec.build_cost if self.valid(key) or not repair_allowed
                else min(c.spec.build_cost, c.spec.repair_cost))
        if self.policy == 'adaptive':
            # Past-only valid reuse estimator, damped by observed invalidations.
            cost *= c.uses / (c.uses + c.updates_observed + 1.0)
        return cost / age

    def _victim_groups(self, protected: set[str]) -> list[set[str]]:
        if self.policy == 'whole_view_LRU':
            groups = []
            for view, keys in self.views.items():
                # Components shared with any surviving logical view remain live.
                others = set().union(*(v for k, v in self.views.items() if k != view))
                group = keys - others
                if group:
                    groups.append(group)
        else:
            # All aliases must be removed together to reclaim a physical block.
            groups = [set(a.owners) for a in self.allocations.values() if a.owners]
        return [g for g in groups if not (g & protected)
                and not any(self.allocations[self.components[k].allocation_id].pins for k in g)]

    def plan(self, required_specs: Sequence[ComponentSpec], workspace_bytes: int = 0,
             build_temp_bytes: int = 0, view_id: str | None = None) -> Plan:
        """Plan one sequential request without mutation; commit after actual execution.

        workspace_bytes is total execution capacity, not an increment. Temporary
        build capacity is conservatively reserved together with every final input.
        Retained workspace growth frees the old workspace before allocating its
        replacement; executors with overlapping growth must give both capacities.
        whole_view_LRU requires view_id and groups all request components.
        """
        if workspace_bytes < 0 or build_temp_bytes < 0:
            raise ValueError('Negative workspace or build capacity')
        if self.policy == 'whole_view_LRU' and view_id is None:
            raise ValueError('whole_view_LRU requires a logical view_id')
        specs = {s.key: s for s in required_specs}
        if len(specs) != len(required_specs):
            raise ValueError('Duplicate requested component keys')
        work = self._planning_copy()
        work.clock += 1
        actions, reuse, repairs, admissions, evictions = [], [], [], [], []
        protected = set(specs)
        peak = self.live_bytes
        cost = 0.0
        # Only one sequential request owns workspace. Growth frees first.
        target_workspace = max(work.workspace_capacity, workspace_bytes)

        def fail(reason: str) -> Plan:
            return Plan(False, reason, [], [], [], [], [], peak, self.live_bytes,
                        target_workspace, 0.0, self.revision)

        def remove(keys: set[str]):
            for key in sorted(keys):
                aid = work.components[key].allocation_id
                work.retire(key)
                actions.append(dict(kind='evict', key=key, allocation_id=aid))
                evictions.append(key)
                a = work.allocations[aid]
                if a.pending_release:
                    amount = work.complete_release(aid)
                    actions.append(dict(kind='release', allocation_id=aid, capacity_bytes=amount))

        # Resolve each component to reuse, repair, overwrite, alias, or allocation.
        todo = []
        reserved_ids = set(work.allocations)
        for key, spec in specs.items():
            old = work.components.get(key)
            if old and work.valid(key, spec):
                reuse.append(key)
                old.last_access = work.clock
                old.uses += 1
                actions.append(dict(kind='reuse', key=key, allocation_id=old.allocation_id))
                continue
            if old:
                alloc = work.allocations[old.allocation_id]
                can_repair = (spec.repairable and old.spec.dependencies == spec.dependencies
                              and alloc.capacity_bytes >= spec.capacity_bytes
                              and not alloc.pins and alloc.owners == {key})
                if spec.repair_dependencies is not None:
                    can_repair = can_repair and set(work.repair_scope(key)) <= set(spec.repair_dependencies)
                if can_repair and spec.repair_cost <= spec.build_cost:
                    scope = work.repair_scope(key)
                    actions.append(dict(kind='repair', key=key, allocation_id=old.allocation_id,
                                        changed_scope={d: None if s is None else sorted(s) for d, s in scope.items()}))
                    old.spec = spec
                    old.stamp = work.stamp(spec.dependencies)
                    old.last_access = work.clock
                    old.uses += 1
                    repairs.append(key)
                    cost += spec.repair_cost
                    continue
                if alloc.pins:
                    return fail(f'Stale requested state {key} is pinned')
                remove({key})
            alias = spec.allocation_id
            if alias in work.allocations:
                a = work.allocations[alias]
                if a.pending_release or a.capacity_bytes < spec.capacity_bytes:
                    return fail(f'Alias {alias} is unavailable or too small')
                # Semantic alias is explicitly authorized by the supplied spec.
                protected.update(a.owners)
                todo.append(('alias', spec, alias))
                continue
            prior_alias = next((s for kind, s, aid in todo if aid == alias), None) if alias else None
            if prior_alias is not None:
                if spec.capacity_bytes > prior_alias.capacity_bytes:
                    return fail(f'Alias {alias} requests incompatible capacity')
                todo.append(('alias', spec, alias))
                continue
            overwrite = None
            if self.policy == 'adaptive':
                for source in spec.overwrite_from:
                    if source in protected or source not in work.components:
                        continue
                    c = work.components[source]
                    a = work.allocations[c.allocation_id]
                    if a.pins or a.owners != {source} or a.capacity_bytes < spec.capacity_bytes:
                        continue
                    # Retain if there is room; otherwise overwrite an unpinned slot.
                    if spec.prefer_overwrite or work.live_bytes + spec.capacity_bytes + target_workspace - work.workspace_capacity + build_temp_bytes > self.budget_bytes:
                        overwrite = source
                        break
            if overwrite is not None:
                c = work.components.pop(overwrite)
                a = work.allocations[c.allocation_id]
                a.owners.remove(overwrite)
                for members in work.views.values():
                    members.discard(overwrite)
                actions.append(dict(kind='overwrite', source=overwrite, key=key,
                                    allocation_id=a.allocation_id, capacity_bytes=a.capacity_bytes))
                evictions.append(overwrite)
                todo.append(('overwrite', spec, a.allocation_id))
            else:
                if alias is None:
                    work.serial += 1
                    alias = f'alloc:{work.serial}:{key}'
                if alias in reserved_ids:
                    return fail(f'Allocation identity {alias} already reserved')
                reserved_ids.add(alias)
                todo.append(('admit', spec, alias))

        new_bytes = sum(s.capacity_bytes for kind, s, aid in todo if kind == 'admit')
        final_peak = work.live_bytes + new_bytes + target_workspace - work.workspace_capacity + build_temp_bytes
        while final_peak > self.budget_bytes:
            if work.policy == 'whole_view_LRU':
                candidates = []
                for view, members in work.views.items():
                    if not members:
                        continue
                    other_owners = set().union(*(v for name, v in work.views.items() if name != view))
                    releasable = (members - other_owners) - protected
                    if any(work.allocations[work.components[k].allocation_id].pins for k in releasable):
                        continue
                    age = max(work.components[k].last_access for k in members)
                    candidates.append((age, view, releasable))
                if not candidates:
                    return fail('All remaining views are required, pinned, or pending release')
                _, victim_view, victim = min(candidates, key=lambda item: item[:2])
                del work.views[victim_view]
                actions.append(dict(kind='drop_view', view_id=victim_view))
                released = work.marginal_release_bytes(victim)
                remove(victim)
                final_peak -= released
                # Dropping a view with no marginal bytes is meaningful: later
                # dropping its co-owner can finally release shared allocations.
                continue
            groups = work._victim_groups(protected)
            if not groups:
                return fail('Required inputs, pinned/pending allocations and workspace exceed budget')
            def rank(group):
                release = work.marginal_release_bytes(group)
                if work.policy in {'shared_component_LRU', 'whole_view_LRU'}:
                    return (max(work.components[k].last_access for k in group), -release, sorted(group))
                return (sum(work._retention_value(k) for k in group) / max(1, release), -release, sorted(group))
            victim = min(groups, key=rank)
            released = work.marginal_release_bytes(victim)
            remove(victim)
            if released == 0:
                return fail('Eviction cannot free any physical capacity')
            final_peak -= released

        if target_workspace != work.workspace_capacity:
            actions.append(dict(kind='workspace_resize', old_capacity_bytes=work.workspace_capacity,
                                capacity_bytes=target_workspace, free_before_allocate=True))
            work.workspace_capacity = target_workspace
        if build_temp_bytes:
            actions.append(dict(kind='reserve_build_temp', capacity_bytes=build_temp_bytes))
        for kind, spec, aid in todo:
            if kind == 'admit':
                work.allocations[aid] = Allocation(aid, spec.capacity_bytes)
                actions.append(dict(kind='admit', key=spec.key, allocation_id=aid,
                                    capacity_bytes=spec.capacity_bytes))
            elif kind == 'alias':
                actions.append(dict(kind='alias', key=spec.key, allocation_id=aid))
            a = work.allocations[aid]
            a.owners.add(spec.key)
            work.components[spec.key] = Component(spec, aid, work.stamp(spec.dependencies), work.clock)
            admissions.append(spec.key)
            cost += 0.0 if kind == 'alias' else spec.build_cost
        if build_temp_bytes:
            actions.append(dict(kind='release_build_temp', capacity_bytes=build_temp_bytes))
        if view_id is not None:
            work.views[view_id] = set(specs)
        # Repairs and overwrite kernels can require transient space. Perform all
        # physical frees and reserve capacity BEFORE these transformations.
        phases = {'drop_view': 0, 'evict': 0, 'release': 0, 'reuse': 1, 'workspace_resize': 2,
                  'reserve_build_temp': 3, 'repair': 4, 'overwrite': 4,
                  'admit': 4, 'alias': 4, 'release_build_temp': 5}
        actions.sort(key=lambda action: phases[action['kind']])
        peak = max(peak, final_peak)
        work.revision = self.revision + 1
        work.assert_invariants()
        return Plan(True, 'feasible', actions, reuse, repairs, admissions, evictions, peak,
                    work.live_bytes, target_workspace, cost, self.revision, work)

    def plan_candidates(self, alternatives: Sequence[Sequence[ComponentSpec]],
                        workspace_bytes: int = 0, build_temp_bytes: int = 0,
                        view_id: str | None = None) -> Plan:
        """Joint selection among caller-declared semantically equal representations.

        Every alternative carries its measured conditional build/repair cost. The
        score adds the past-only expected cost of destroying retained components.
        Unlike prepare-then-evict greedy, representation choice sees eviction loss.
        This is a heuristic, not a claimed optimal or competitive algorithm.
        """
        if not alternatives:
            raise ValueError('No representation alternatives supplied')
        candidates = [self.plan(specs, workspace_bytes, build_temp_bytes, view_id)
                      for specs in alternatives]
        feasible = [p for p in candidates if p.feasible]
        if not feasible:
            return candidates[0]
        def score(plan):
            destroyed = set(plan.evict) - set(plan.admit)
            lost = sum(self._retention_value(k) for k in destroyed if k in self.components)
            return (plan.estimated_cost + lost, plan.predicted_peak_bytes)
        return min(feasible, key=score)

    def commit(self, plan: Plan) -> None:
        if not plan.feasible or plan._state is None:
            raise ValueError('Cannot commit an infeasible plan')
        if plan.base_revision != self.revision:
            raise ValueError('Plan is stale; replan after updates, pins, or releases')
        if plan.predicted_peak_bytes > self.budget_bytes:
            raise ValueError('Plan exceeds budget')
        # The planned state is private and can be transferred, consuming the plan.
        self.__dict__ = plan._state.__dict__
        plan._state = None

    def assert_invariants(self) -> None:
        if self.live_bytes > self.budget_bytes:
            raise AssertionError('Capacity exceeds budget')
        for key, component in self.components.items():
            a = self.allocations[component.allocation_id]
            assert key in a.owners and not a.pending_release
        for aid, a in self.allocations.items():
            assert a.pins >= 0 and a.capacity_bytes >= 0
            assert not (a.pending_release and a.owners)
            for key in a.owners:
                assert self.components[key].allocation_id == aid


def normalized_weight_repair_edges(changed_edges: Iterable[int], edge_sources: Sequence[int]) -> list[int]:
    """A changed outgoing edge weight invalidates its entire source normalization.

    edge_sources refers to source IDs, even if storage is incoming CSC/CSR. This
    avoids the incorrect assumption that one update always changes one scalar or
    one physical storage row.
    """
    changed_sources = {edge_sources[e] for e in changed_edges}
    return [e for e, source in enumerate(edge_sources) if source in changed_sources]
