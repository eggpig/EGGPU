"""CPU-only R6 v2 planner with deterministic floating-point cost reductions.

V1 remains frozen. V2 retains strict score/trace equality; unordered component
sets are reduced in canonical name order using math.fsum. No tolerance replaces
the replay invariant.

CPU-only maintenance-subset planner; no CUDA or third-party imports.

Every policy evaluates the same declared request list and action menu. Costs are
caller supplied, never fabricated measurements. Capacity is actual allocation
capacity (256-byte aligned); payload alone determines transfer bytes. Aliased
unaffected components share capacity. In-place advance currently requires a
unique owner, matching the frozen R4 implementation.
"""
from dataclasses import dataclass
from itertools import combinations
from math import fsum, inf, isfinite
from time import perf_counter
from typing import Callable, Optional


@dataclass(frozen=True)
class Component:
    name: str
    capacity_bytes: int
    payload_bytes: int
    kind: str = ''
    allocation_id: Optional[str] = None
    old_capacity_bytes: Optional[int] = None
    old_payload_bytes: Optional[int] = None
    advance_d2d_bytes: int = 0
    advance_cost: float = 0.0
    restore_cost: float = 0.0
    restore_scratch_bytes: int = 0
    pinned: bool = False
    last_access: int = 0

    @property
    def allocation(self):
        return self.name if self.allocation_id is None else self.allocation_id

    @property
    def old_capacity(self):
        return self.capacity_bytes if self.old_capacity_bytes is None else self.old_capacity_bytes


@dataclass(frozen=True)
class Query:
    required: tuple
    transient_bytes: int = 0
    label: str = ''


@dataclass(frozen=True)
class CostModel:
    h2d_per_byte: float = 0.0
    d2d_per_byte: float = 0.0
    packet_fixed: float = 0.0
    restore_fixed: float = 0.0
    advance_fixed: float = 0.0
    unit: str = 'seconds'
    provenance: str = 'caller-supplied; not a planner measurement'


@dataclass(frozen=True)
class Problem:
    components: tuple
    resident_names: frozenset
    eligible_names: frozenset
    queries: tuple
    budget_bytes: int
    fixed_bytes: int = 0
    packet_bytes: Optional[Callable] = None
    packet_alignment: int = 256
    cost_model: Optional[CostModel] = None
    stale_names: Optional[frozenset] = None
    growth_d2d_bytes: Optional[Callable] = None


def aligned_packet_fields(fields, alignment=8):
    """Payload for an ordered sequence of field lengths; caller unions fields."""
    size = 0
    for length in fields:
        if length < 0 or int(length) != length:
            raise ValueError('field length must be a nonnegative integer')
        size = (size + alignment - 1) // alignment * alignment + int(length)
    return size


def _validate(p):
    comp = {c.name: c for c in p.components}
    if len(comp) != len(p.components):
        raise ValueError('duplicate component name')
    residents = set(p.resident_names)
    eligible = set(p.eligible_names)
    stale = eligible if p.stale_names is None else set(p.stale_names)
    if not eligible <= stale <= residents <= comp.keys():
        raise ValueError('eligible must be a subset of stale, resident, catalog')
    if len(eligible) > 7:
        raise ValueError('exact R6 menu supports at most seven eligible components')
    if p.budget_bytes <= 0 or p.fixed_bytes < 0 or p.packet_alignment <= 0:
        raise ValueError('invalid budget/fixed bytes/alignment')
    capacities = {}
    for c in p.components:
        if c.capacity_bytes <= 0 or c.capacity_bytes % 256 or c.old_capacity <= 0 or c.old_capacity % 256:
            raise ValueError('component capacities must be positive multiples of 256')
        if not 0 <= c.payload_bytes <= c.capacity_bytes:
            raise ValueError('payload exceeds current capacity')
        if c.old_payload_bytes is not None and not 0 <= c.old_payload_bytes <= c.old_capacity:
            raise ValueError('old payload exceeds old capacity')
        if min(c.advance_d2d_bytes, c.restore_scratch_bytes, c.advance_cost, c.restore_cost) < 0:
            raise ValueError('negative primitive cost/bytes')
        if not isfinite(c.advance_cost) or not isfinite(c.restore_cost):
            raise ValueError('nonfinite primitive cost')
        signature = (c.old_capacity, c.capacity_bytes, c.payload_bytes)
        if c.allocation in capacities and capacities[c.allocation] != signature:
            raise ValueError('shared allocation capacity/payload mismatch')
        capacities[c.allocation] = signature
    for name in eligible:
        if sum(comp[n].allocation == comp[name].allocation for n in residents) != 1:
            raise ValueError('advance requires a uniquely owned physical allocation')
    # A stale alias cannot share its physical value with an unaffected owner.
    for name in stale:
        owners = {n for n in residents if comp[n].allocation == comp[name].allocation}
        if not owners <= stale:
            raise ValueError('stale allocation has an unaffected resident alias')
    for q in p.queries:
        if not set(q.required) <= comp.keys() or q.transient_bytes < 0:
            raise ValueError('unknown query component or negative transient bytes')
    if p.cost_model is not None:
        values = [p.cost_model.h2d_per_byte, p.cost_model.d2d_per_byte,
                  p.cost_model.packet_fixed, p.cost_model.restore_fixed, p.cost_model.advance_fixed]
        if any(not isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid externally supplied cost model')
    return comp, residents, eligible, stale


def _prepare(p):
    comp, residents, eligible, stale = _validate(p)
    allocation = {n: c.allocation for n,c in comp.items()}
    owners = {}
    for n in residents:
        owners.setdefault(allocation[n], set()).add(n)
    capacities = {allocation[n]: comp[n].old_capacity if n in stale else comp[n].capacity_bytes
                  for n in residents}
    upcoming = {n: inf for n in comp}
    next_uses = [None]*len(p.queries)
    for i in reversed(range(len(p.queries))):
        for n in p.queries[i].required:
            upcoming[n] = i
        next_uses[i] = upcoming.copy()
    requested = {}
    for q in p.queries:
        for n in q.required:
            requested.setdefault(allocation[n], set()).add(n)
    return dict(comp=comp, residents=frozenset(residents), eligible=frozenset(eligible),
                stale=frozenset(stale), allocation=allocation, owners=owners,
                capacities=capacities, next_uses=next_uses, requested=requested,
                packet_cache={})


def _packet_payload(p, selected, prepared):
    cache = prepared['packet_cache']
    if selected not in cache:
        value = 0 if not selected or p.packet_bytes is None else p.packet_bytes(selected)
        if not isinstance(value, int) or value < 0:
            raise ValueError('packet callback must return a nonnegative Python int')
        cache[selected] = value
    return cache[selected]


def lower_bound(p, selected, *, _prepared=None):
    """Safe cost/byte bound: exact advance plus unavoidable first restores.

    Restores are charged once per unique physical allocation, with the minimum
    primitive cost among its queried aliases. Growth, capacity-driven eviction
    and repeat restores are omitted, all having nonnegative costs.
    """
    prepared = _prepare(p) if _prepared is None else _prepared
    selected = frozenset(selected)
    if not selected <= prepared['eligible']:
        raise ValueError('selected component is not eligible')
    comp, allocation = prepared['comp'], prepared['allocation']
    available = {allocation[n] for n in prepared['residents']-prepared['stale'] | selected}
    missing = [owners for aid,owners in prepared['requested'].items() if aid not in available]
    restore_bytes = sum(comp[min(owners)].payload_bytes for owners in missing)
    packet = _packet_payload(p, selected, prepared)
    movement = sum(comp[n].advance_d2d_bytes for n in selected)
    if p.cost_model is None:
        return packet+movement+restore_bytes
    model = p.cost_model
    return fsum([(packet+restore_bytes)*model.h2d_per_byte, movement*model.d2d_per_byte,
                 model.packet_fixed if selected else 0., len(selected)*model.advance_fixed,
                 fsum(comp[n].advance_cost for n in sorted(selected)),
                 model.restore_fixed*len(missing),
                 fsum(sorted(min(comp[n].restore_cost for n in owners) for owners in missing))])


def evaluate_subset(p, selected, *, trace=True, _prepared=None):
    """Simulate one advance subset using physical-owner next-use eviction.

    fixed_bytes denotes permanently resident workspace. Query.transient_bytes is
    additional simultaneous storage, including retained result pins if any.
    growth_d2d_bytes(selected, retained) optionally predicts arena relocation;
    selected and retained are frozensets after maintenance admission eviction.
    """
    prepared = _prepare(p) if _prepared is None else _prepared
    comp, eligible, stale = prepared['comp'], prepared['eligible'], prepared['stale']
    resident = set(prepared['residents'])
    allocation = prepared['allocation']
    owners_by_allocation = {aid:set(owners) for aid,owners in prepared['owners'].items()}
    selected = frozenset(selected)
    if not selected <= eligible:
        raise ValueError('selected component is not eligible')
    capacities = prepared['capacities'].copy()
    touched = {n: comp[n].last_access for n in comp}
    evictions, events, restores = [], [], []
    retired, maintenance_victims = [], []
    consumed, wasted = set(), set()
    peak = p.fixed_bytes + sum(capacities.values())
    live_bytes = peak
    packet = packet_capacity = advance_d2d = growth_d2d = restore_h2d = 0
    restore_primitive_costs = []
    admitted_advance = False
    failure = None

    def live():
        return live_bytes

    def record(kind, **fields):
        if trace:
            events.append(dict(kind=kind, live_bytes=live(), **fields))

    def remove(names, reason, query_index):
        nonlocal live_bytes
        names = sorted(set(names))
        old = live()
        for name in names:
            resident.remove(name)
            aid = allocation[name]
            owners_by_allocation[aid].remove(name)
            if not owners_by_allocation[aid]:
                del owners_by_allocation[aid]
                live_bytes -= capacities.pop(aid)
            if name in selected and admitted_advance and name not in consumed:
                wasted.add(name)
        released = old - live()
        record(reason, names=names, bytes=released, query_index=query_index)
        if reason == 'evict':
            item = dict(phase='maintenance' if query_index < 0 else 'query',
                        query_index=query_index, names=names, bytes=released)
            evictions.append(item)
            if query_index < 0:
                maintenance_victims.extend(names)

    def next_use(name, index):
        return prepared['next_uses'][max(index, 0)][name] if p.queries else inf

    def reserve(extra, protected, index, reason):
        nonlocal peak, failure
        while live() + extra > p.budget_bytes:
            choices = []
            for aid, owners in owners_by_allocation.items():
                if owners & protected or any(comp[n].pinned for n in owners):
                    continue
                # Farthest next physical use, then least-recently accessed owner
                # group, then largest true release; all controls share this rule.
                distances = {n: next_use(n, index) for n in owners}
                nearest = min(distances.values())
                # EpochModel encodes each owner's (next-use, real recency),
                # then takes the maximum over a shared physical allocation.
                # Only owners at its nearest next use decide the recency tie.
                key = (-nearest,
                       max(touched[n] for n in owners if distances[n] == nearest),
                       -capacities[aid], tuple(sorted(owners)))
                choices.append((key, owners))
            if not choices:
                failure = dict(reason=reason, needed_bytes=live()+extra,
                               budget_bytes=p.budget_bytes,
                               excess_bytes=live()+extra-p.budget_bytes,
                               query_index=index)
                record('infeasible', **failure)
                return False
            _, owners = min(choices, key=lambda pair: pair[0])
            remove(owners, 'evict', index)
        peak = max(peak, live()+extra)
        record('reserve', phase=reason, query_index=index, extra_bytes=extra,
               peak_at_event_bytes=live()+extra)
        return True

    def result():
        h2d = (packet if admitted_advance else 0) + restore_h2d
        d2d = advance_d2d + growth_d2d
        model = p.cost_model
        cost = None if model is None else fsum([
            h2d*model.h2d_per_byte, d2d*model.d2d_per_byte,
            model.packet_fixed if admitted_advance else 0.,
            model.advance_fixed*len(selected) if admitted_advance else 0.,
            fsum(comp[n].advance_cost for n in sorted(selected)) if admitted_advance else 0.,
            model.restore_fixed*len(restores), fsum(restore_primitive_costs)])
        return dict(feasible=failure is None, reason=None if failure is None else failure['reason'],
                    failure=failure, selected=sorted(selected), retired_stale=retired,
                    maintenance_victims=maintenance_victims, evictions=evictions,
                    restores=restores, events=events,
                    packet_payload_bytes=packet, packet_capacity_bytes=packet_capacity,
                    advance_d2d_bytes=advance_d2d, growth_d2d_bytes=growth_d2d,
                    restore_h2d_bytes=restore_h2d, h2d_bytes=h2d, d2d_bytes=d2d,
                    peak_bytes=peak, final_resident=sorted(resident),
                    advanced_then_consumed=sorted(consumed),
                    advanced_then_evicted_before_use=sorted(wasted),
                    pending_advanced=sorted(selected & resident - consumed - wasted),
                    cost=cost, objective=h2d+d2d if model is None else cost,
                    cost_unit='bytes' if model is None else model.unit,
                    cost_provenance='byte objective; no timing estimate' if model is None else model.provenance,
                    growth_d2d_model='caller callback' if p.growth_d2d_bytes else 'not supplied; zero relocation estimate')

    record('initial')
    if peak > p.budget_bytes:
        failure = dict(reason='initial_state_exceeds_budget', needed_bytes=peak,
                       budget_bytes=p.budget_bytes, excess_bytes=peak-p.budget_bytes, query_index=-1)
        return result()
    for name in sorted(stale-selected):
        if comp[name].pinned:
            failure = dict(reason='stale_pinned_state_requires_copy_on_write', needed_bytes=live(),
                           budget_bytes=p.budget_bytes, excess_bytes=0, query_index=-1)
            return result()
        retired.append(name)
        remove([name], 'retire_stale', -1)
    if selected:
        if any(comp[n].pinned for n in selected):
            failure = dict(reason='stale_pinned_state_requires_copy_on_write', needed_bytes=live(),
                           budget_bytes=p.budget_bytes, excess_bytes=0, query_index=-1)
            return result()
        packet = _packet_payload(p, selected, prepared)
        packet_capacity = ((packet+p.packet_alignment-1)//p.packet_alignment*p.packet_alignment) if packet else 0
        growth = sum(max(0, comp[n].capacity_bytes-capacities[allocation[n]]) for n in selected)
        if not reserve(growth+packet_capacity, set(selected), -1, 'advance_peak'):
            return result()
        growth_d2d = 0 if p.growth_d2d_bytes is None else p.growth_d2d_bytes(selected, frozenset(resident))
        if not isinstance(growth_d2d, int) or growth_d2d < 0:
            raise ValueError('growth callback must return a nonnegative Python int')
        advance_d2d = sum(comp[n].advance_d2d_bytes for n in selected)
        for name in selected:
            # The frozen grow path keeps existing capacity when no grow occurs.
            capacities[allocation[name]] = max(capacities[allocation[name]],comp[name].capacity_bytes)
        live_bytes += growth
        admitted_advance = True
        record('advance', names=sorted(selected), packet_payload_bytes=packet,
               packet_capacity_bytes=packet_capacity, growth_capacity_bytes=growth,
               advance_d2d_bytes=advance_d2d, growth_d2d_bytes=growth_d2d)
        record('release_packet', bytes=packet_capacity)
    for i, query in enumerate(p.queries):
        required = set(query.required)
        # All inputs required by this call remain protected during restore.
        for name in dict.fromkeys(query.required):
            if name in resident:
                continue
            c = comp[name]
            aid = allocation[name]
            if aid in capacities:
                resident.add(name)
                owners_by_allocation[aid].add(name)
                record('alias', name=name, allocation=aid, query_index=i)
                continue
            if not reserve(c.capacity_bytes+c.restore_scratch_bytes, required, i, 'restore_peak'):
                return result()
            resident.add(name)
            capacities[aid] = c.capacity_bytes
            owners_by_allocation[aid] = {name}
            live_bytes += c.capacity_bytes
            restore_h2d += c.payload_bytes
            restore_primitive_costs.append(c.restore_cost)
            restores.append(dict(name=name, query_index=i, h2d_bytes=c.payload_bytes))
            record('restore', name=name, query_index=i, h2d_bytes=c.payload_bytes,
                   capacity_bytes=c.capacity_bytes, scratch_bytes=c.restore_scratch_bytes)
        if not reserve(query.transient_bytes, required, i, 'query_peak'):
            return result()
        consumed.update(required & selected - wasted)
        query_clock = max(touched.values(), default=0)+1
        for name in required:
            touched[name] = query_clock
        record('execute', query_index=i, label=query.label, required=sorted(required),
               transient_bytes=query.transient_bytes)
    return result()


def plan(p, strategy='exhaustive'):
    """Return chosen simulation, policy overhead, and all evaluated candidates.

    Strategies: full_refresh, all_advance, bidirectional_greedy, closed_form,
    exhaustive. all_advance falls back to full_refresh if its action is infeasible.
    closed_form scores singleton incremental costs after subtracting the shared
    fixed packet charge; exact for one fixed setup and separable feasible actions.
    """
    started = perf_counter()
    prepared = _prepare(p)
    names = tuple(sorted(p.eligible_names))
    full = frozenset(names)
    cache = {}
    pruned = []

    def evaluate(subset):
        key = frozenset(subset)
        if key not in cache:
            cache[key] = evaluate_subset(p, key, trace=False, _prepared=prepared)
        return cache[key]

    def rank(r):
        return (not r['feasible'], r['failure']['excess_bytes'] if r['failure'] else 0,
                r['objective'], r['peak_bytes'], len(r['selected']), tuple(r['selected']))

    empty_result = evaluate(())
    if strategy == 'full_refresh':
        winner = empty_result
    elif strategy == 'all_advance':
        all_result = evaluate(full)
        winner = all_result if all_result['feasible'] else empty_result
    elif strategy == 'exhaustive':
        bounded = []
        for size in range(len(names)+1):
            for selected in combinations(names, size):
                selected = frozenset(selected)
                bounded.append((lower_bound(p, selected, _prepared=prepared),
                                len(selected), tuple(sorted(selected)), selected))
        winner = empty_result
        for bound, _, _, selected in sorted(bounded):
            if selected in cache:
                if rank(cache[selected]) < rank(winner):
                    winner = cache[selected]
                continue
            # Nonnegative lower bounds apply to feasible completed plans. The
            # margin conservatively keeps nearly equal floating-point costs;
            # all equal-cost choices still receive full peak/tie evaluation.
            margin = 1e-12*max(1., abs(winner['objective']))
            if winner['feasible'] and bound > winner['objective']+margin:
                pruned.append(dict(selected=sorted(selected), lower_bound=bound,
                                   incumbent_objective=winner['objective'], status='strict_lower_bound_pruned'))
                continue
            candidate = evaluate(selected)
            if rank(candidate) < rank(winner):
                winner = candidate
    elif strategy == 'bidirectional_greedy':
        candidates = [empty_result]
        for current, adding in [(frozenset(), True), (full, False)]:
            incumbent = evaluate(current)
            while True:
                edits = full-current if adding else current
                if not edits:
                    break
                choices = [evaluate(current | {n} if adding else current-{n}) for n in sorted(edits)]
                best = min(choices, key=rank)
                if rank(best) >= rank(incumbent):
                    break
                current = frozenset(best['selected'])
                incumbent = best
            candidates.append(incumbent)
        winner = min(candidates, key=rank)
    elif strategy == 'closed_form':
        # Subtract only the explicit shared setup. Shared fields, eviction,
        # restoration multiplicity and capacity remain measured by the same
        # simulator when evaluating the resulting candidate.
        setup = 0 if p.cost_model is None else p.cost_model.packet_fixed
        selected = frozenset(n for n in names
                             if evaluate({n})['feasible'] and
                             evaluate({n})['objective']-empty_result['objective']-setup < 0)
        winner = min([empty_result, evaluate(selected)], key=rank)
    else:
        raise ValueError('unknown strategy: '+str(strategy))
    # The selected policy outcome receives the original complete event API.
    # Pure callbacks see the identical immutable problem on this final pass.
    result = evaluate_subset(p, winner['selected'], trace=True, _prepared=prepared)
    if any(result[k] != winner[k] for k in ('feasible','objective','peak_bytes','h2d_bytes','d2d_bytes','selected')):
        raise AssertionError('Planner callbacks changed between scoring and execution trace')
    result.update(strategy=strategy, evaluations=len(cache), simulation_passes=len(cache)+1,
                  candidate_menu_size=1 << len(names), pruned_candidates=pruned,
                  planner_seconds=perf_counter()-started,
                  candidate_summary=[dict(selected=r['selected'], feasible=r['feasible'],
                                          cost=r['cost'], objective=r['objective'],
                                          peak_bytes=r['peak_bytes'], reason=r['reason'])
                                     for _,r in sorted(cache.items(),key=lambda pair:(len(pair[0]),tuple(sorted(pair[0]))))])
    return result
