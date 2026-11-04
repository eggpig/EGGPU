"""R7 cost-aware recovery search over the frozen R6 physical action menu.

A context stores only action-set hints, immutable menu structure and observed
CPU costs. No prior version, residency decision or physical result is reused.
Every returned action has one complete simulation against the current Problem.
Model regret bounds concern this finite action menu and fixed eviction policy;
they do not bound actual end-to-end wall time.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations
from math import fsum, isfinite
from statistics import median
from time import perf_counter

from easygraph.gpu._arena.round6_recovery import planner_v2 as reference

Component = reference.Component
Query = reference.Query
CostModel = reference.CostModel
Problem = reference.Problem
evaluate_subset = reference.evaluate_subset
lower_bound = reference.lower_bound
aligned_packet_fields = reference.aligned_packet_fields

STRATEGIES = ('bounded', 'adaptive', 'certified', 'previous')


@dataclass
class RecoveryContext:
    """Session-local, bounded hints; safe across changes to versions and sizes."""
    capacity: int = 32
    max_evaluations: int = 6
    cost_multiplier: float = 1.0
    hints: OrderedDict = field(default_factory=OrderedDict)
    menus: OrderedDict = field(default_factory=OrderedDict)
    simulation_seconds: list = field(default_factory=list)

    def __post_init__(self):
        if self.capacity < 1 or self.max_evaluations < 2 or \
                not isfinite(self.cost_multiplier) or self.cost_multiplier < 0:
            raise ValueError('Invalid recovery context resource limits')

    def remember(self, key, selected):
        self.hints[key] = tuple(selected)
        self.hints.move_to_end(key)
        while len(self.hints) > self.capacity:
            self.hints.popitem(last=False)

    def menu(self, names):
        if names not in self.menus:
            self.menus[names] = tuple(frozenset(s) for size in range(len(names)+1)
                                     for s in combinations(names, size))
        self.menus.move_to_end(names)
        while len(self.menus) > self.capacity:
            self.menus.popitem(last=False)
        return self.menus[names]

    def observed(self, seconds):
        self.simulation_seconds.append(seconds)
        del self.simulation_seconds[:-32]


def structure_key(p):
    """A reuse *hint*, never a validity certificate for an old execution plan.

    Capacity, payload, cost model and version changes deliberately retain the
    hint; the current simulation checks them afresh. Graph identities, query
    order and owner sharing prevent unrelated structures polluting each other.
    """
    return (tuple(sorted((c.name, c.kind, c.allocation) for c in p.components)),
            tuple(sorted(p.eligible_names)),
            tuple(sorted(p.eligible_names if p.stale_names is None else p.stale_names)),
            tuple(tuple(q.required) for q in p.queries))


def _rank(r):
    return (not r['feasible'], r['failure']['excess_bytes'] if r['failure'] else 0,
            r['objective'], r['peak_bytes'], len(r['selected']), tuple(r['selected']))


def _bounds(p, prepared, menu):
    """R6 safe bound with common alias/restore descriptors built only once.

    The expression deliberately retains R6's canonical fsum grouping. Only
    descriptor work is hoisted; no floating tolerance changes score semantics.
    """
    comp = prepared['comp']; allocation = prepared['allocation']
    available = {allocation[n] for n in prepared['residents']-prepared['stale']}
    missing = [(aid, comp[min(owners)].payload_bytes,
                min(comp[n].restore_cost for n in owners))
               for aid, owners in prepared['requested'].items() if aid not in available]
    selected_alloc = {n: allocation[n] for n in prepared['eligible']}
    model = p.cost_model
    rows = []
    for selected in menu:
        aids = {selected_alloc[n] for n in selected}
        absent = [item for item in missing if item[0] not in aids]
        restore_bytes = sum(item[1] for item in absent)
        packet = reference._packet_payload(p, selected, prepared)
        movement = sum(comp[n].advance_d2d_bytes for n in selected)
        if model is None:
            value = packet+movement+restore_bytes
        else:
            value = fsum([(packet+restore_bytes)*model.h2d_per_byte,
                          movement*model.d2d_per_byte,
                          model.packet_fixed if selected else 0.,
                          len(selected)*model.advance_fixed,
                          fsum(comp[n].advance_cost for n in sorted(selected)),
                          model.restore_fixed*len(absent),
                          fsum(sorted(item[2] for item in absent))])
        rows.append((value, len(selected), tuple(sorted(selected)), selected))
    return sorted(rows)


def _relaxed_global_bound(p, prepared):
    """O(components) relaxation: each missing physical input chooses its
    cheaper advance or first-restore path, ignoring shared setup, relocation,
    repeat restores and all resource conflicts. All omitted costs are >= 0.
    """
    comp=prepared['comp']; allocation=prepared['allocation']; model=p.cost_model
    available={allocation[n] for n in prepared['residents']-prepared['stale']}
    advancing={allocation[n]:n for n in prepared['eligible']}
    values=[]
    for aid,owners in prepared['requested'].items():
        if aid in available:
            continue
        first=comp[min(owners)]
        restore=first.payload_bytes if model is None else fsum([
            first.payload_bytes*model.h2d_per_byte,model.restore_fixed,
            min(comp[n].restore_cost for n in owners)])
        if aid in advancing:
            c=comp[advancing[aid]]
            advance=c.advance_d2d_bytes if model is None else fsum([
                c.advance_d2d_bytes*model.d2d_per_byte,model.advance_fixed,c.advance_cost])
            restore=min(restore,advance)
        values.append(restore)
    value=sum(values) if model is None else fsum(values)
    # A downward numerical margin keeps this bound conservative when its
    # differently grouped weighted byte terms meet the exact fsum objective.
    return value if model is None else max(0.,value-1e-12*max(1.,abs(value)))


def _sparse_menu(p, prepared, candidates, previous):
    """State-independent one-edit proposals plus observed wasted maintenance.

    This does not search graph names, known traces or future epochs. It spans
    both directions from the action extremes and from the previous action.
    Every proposal is rescored against current packet fields and physical state.
    """
    full=frozenset(prepared['eligible']); choices=set(candidates)
    centers=[frozenset(),full]
    if previous is not None:
        centers.append(frozenset(previous)&full)
    for center in centers:
        for n in sorted(full):
            choices.add(center ^ {n})
    for r in candidates.values():
        selected=frozenset(r['selected'])
        choices.add(selected-frozenset(r['advanced_then_evicted_before_use']))
        choices.add(selected-frozenset(r['pending_advanced']))
    required={n for q in p.queries for n in q.required}
    choices.add(full & required)
    # Packet-fixed setup is shared: remove it before testing isolated savings.
    singletons=tuple([frozenset()]+[frozenset((n,)) for n in sorted(full)])
    scalar={s:value for value,_,_,s in _bounds(p,prepared,singletons)}
    setup=0 if p.cost_model is None else p.cost_model.packet_fixed
    choices.add(frozenset(n for n in full if scalar[frozenset((n,))]-scalar[frozenset()]-setup<0))
    return tuple(sorted(choices,key=lambda s:(len(s),tuple(sorted(s)))))


def _no_eviction_certificate(p, prepared, menu):
    """Sufficient physical capacity certificate for every subset in the menu.

    All possibly live physical allocations plus the largest simultaneous
    scratch region fit. Overestimation only rejects the shortcut; it never
    admits an unsafe plan. Pinned stale values still require explicit rejection.
    """
    comp = prepared['comp']
    if any(comp[n].pinned for n in prepared['stale']):
        return None
    initial = p.fixed_bytes+sum(prepared['capacities'].values())
    if initial > p.budget_bytes:
        return None
    possible = dict(prepared['capacities'])
    for q in p.queries:
        for n in q.required:
            possible[comp[n].allocation] = max(possible.get(comp[n].allocation, 0), comp[n].capacity_bytes)
    for n in prepared['eligible']:
        possible[comp[n].allocation] = max(possible.get(comp[n].allocation, 0), comp[n].capacity_bytes)
    packet_peak = max(((reference._packet_payload(p, s, prepared)+p.packet_alignment-1)//p.packet_alignment)*p.packet_alignment
                      for s in menu)
    scratch_peak = max([0]+[q.transient_bytes for q in p.queries]+
                       [comp[n].restore_scratch_bytes for q in p.queries for n in q.required])
    bound = p.fixed_bytes+sum(possible.values())+max(packet_peak, scratch_peak)
    if bound > p.budget_bytes:
        return None
    return dict(kind='sufficient_no_eviction_capacity', conservative_peak_bytes=bound,
                budget_bytes=p.budget_bytes, initial_bytes=initial)


def _direct_score(p, selected, prepared):
    """Exact cost and peak when the no-eviction certificate holds.

    This is a straight scan of first physical uses, without owner eviction,
    touched clocks, event construction, or state copies. Restore aliases use
    the *first requested* owner, not the cheaper alias used by the lower bound.
    """
    comp = prepared['comp']; allocation = prepared['allocation']
    resident = prepared['residents']-prepared['stale'] | selected
    capacities = {allocation[n]: comp[n].old_capacity if n in selected else comp[n].capacity_bytes
                  for n in resident}
    initial = p.fixed_bytes+sum(prepared['capacities'].values())
    live = p.fixed_bytes+sum(capacities.values())
    packet = reference._packet_payload(p, selected, prepared)
    packet_capacity = (packet+p.packet_alignment-1)//p.packet_alignment*p.packet_alignment
    growth = sum(max(0, comp[n].capacity_bytes-capacities[allocation[n]]) for n in selected)
    peak = max(initial, live+growth+packet_capacity)
    movement = sum(comp[n].advance_d2d_bytes for n in selected)
    relocation = 0 if not selected or p.growth_d2d_bytes is None else p.growth_d2d_bytes(selected, resident)
    if not isinstance(relocation, int) or relocation < 0:
        raise ValueError('growth callback must return a nonnegative Python int')
    for n in selected:
        capacities[allocation[n]] = max(capacities[allocation[n]], comp[n].capacity_bytes)
    live += growth
    restore_bytes = 0; restore_costs = []
    for q in p.queries:
        for n in dict.fromkeys(q.required):
            c = comp[n]; aid = allocation[n]
            if aid in capacities:
                continue
            peak = max(peak, live+c.capacity_bytes+c.restore_scratch_bytes)
            live += c.capacity_bytes; capacities[aid] = c.capacity_bytes
            restore_bytes += c.payload_bytes; restore_costs.append(c.restore_cost)
        peak = max(peak, live+q.transient_bytes)
    h2d = packet+restore_bytes; d2d = movement+relocation
    model = p.cost_model
    objective = h2d+d2d if model is None else fsum([
        h2d*model.h2d_per_byte, d2d*model.d2d_per_byte,
        model.packet_fixed if selected else 0.,
        model.advance_fixed*len(selected) if selected else 0.,
        fsum(comp[n].advance_cost for n in sorted(selected)) if selected else 0.,
        model.restore_fixed*len(restore_costs), fsum(restore_costs)])
    return dict(selected=sorted(selected), objective=objective, peak_bytes=peak,
                h2d_bytes=h2d, d2d_bytes=d2d)


def plan(p, strategy='adaptive', context=None):
    """Choose a currently validated action and expose all decision overhead.

    ``bounded`` evaluates at most context.max_evaluations complete candidates.
    ``adaptive`` tests a sparse, symmetric neighborhood of empty/full/previous
    actions and stops when candidate opportunity is no larger than observed CPU
    simulation cost, in seconds-valued models only. A separate global relaxation
    bounds regret for the unsearched full menu.
    ``certified`` uses a sufficient no-eviction certificate to exactly score
    all subsets without state simulation, otherwise takes the adaptive path.
    ``previous`` is the explicit empty/full/previous-set reuse control.
    Frozen R6 strategy names delegate unchanged to the frozen implementation.
    """
    if strategy not in STRATEGIES:
        return reference.plan(p, strategy)
    started = perf_counter()
    context = RecoveryContext() if context is None else context
    prepared = reference._prepare(p)
    names = tuple(sorted(p.eligible_names)); full = frozenset(names)
    key = structure_key(p); previous = context.hints.get(key)
    menu = context.menu(names)
    candidates = {}; simulation_durations = []; bound_rows = []
    certificate = None; direct_count = 0; reason = None; relaxed_bound = None

    def evaluate(selected):
        selected = frozenset(selected)
        if selected not in candidates:
            clock = perf_counter()
            candidates[selected] = evaluate_subset(p, selected, trace=True, _prepared=prepared)
            seconds = perf_counter()-clock
            simulation_durations.append(seconds)
            context.observed(seconds)
        return candidates[selected]

    if strategy == 'certified':
        certificate = _no_eviction_certificate(p, prepared, menu)
        if certificate is not None:
            scored = [_direct_score(p, s, prepared) for s in menu]
            direct_count = len(scored)
            direct = min(scored, key=lambda r:(r['objective'],r['peak_bytes'],len(r['selected']),tuple(r['selected'])))
            winner = evaluate(direct['selected'])
            if not winner['feasible'] or any(winner[k] != direct[k] for k in ('objective','peak_bytes','h2d_bytes','d2d_bytes','selected')):
                raise AssertionError('No-eviction formula differs from current physical simulation')
            reason = 'exact_no_eviction_certificate'

    if reason is None:
        # Both feasible action extremes remain available to every new strategy.
        evaluate(()); evaluate(full)
        if previous is not None and frozenset(previous) <= full and len(candidates) < context.max_evaluations:
            evaluate(previous)
        winner = min(candidates.values(), key=_rank)
        if strategy == 'previous':
            reason = 'empty_full_previous_control'
        else:
            if strategy == 'adaptive':
                search_menu = _sparse_menu(p,prepared,candidates,previous)
                relaxed_bound = _relaxed_global_bound(p,prepared)
            else:
                search_menu = menu
            bound_rows = _bounds(p, prepared, search_menu)
            reason = 'sparse_pool_exhausted' if strategy == 'adaptive' else 'menu_exhausted'
            for bound, _, _, selected in bound_rows:
                if selected in candidates:
                    continue
                opportunity = max(0., winner['objective']-bound)
                margin = 1e-12*max(1., abs(winner['objective']))
                if winner['feasible'] and bound > winner['objective']+margin:
                    reason = 'strict_model_bound'; break
                if len(candidates) >= context.max_evaluations:
                    reason = 'evaluation_budget'; break
                estimated = median(context.simulation_seconds)*context.cost_multiplier
                if strategy in ('adaptive','certified') and p.cost_model is not None and p.cost_model.unit == 'seconds' \
                        and winner['feasible'] and opportunity <= estimated:
                    reason = 'model_opportunity_below_observed_simulation_cost'; break
                candidate = evaluate(selected)
                if _rank(candidate) < _rank(winner):
                    winner = candidate

    # Bounds are not computed for the deliberately cheap previous control.
    # Zero is safe since all primitive costs are nonnegative.
    if certificate is not None:
        model_lower = winner['objective']; regret = 0.
    elif relaxed_bound is not None:
        model_lower=relaxed_bound
        regret=max(0.,winner['objective']-model_lower) if winner['feasible'] else None
    elif bound_rows:
        remaining = [v for v,_,_,s in bound_rows if s not in candidates]
        completed = [r['objective'] for r in candidates.values() if r['feasible']]
        model_lower = min(remaining+completed) if remaining or completed else 0.
        regret = max(0., winner['objective']-model_lower) if winner['feasible'] else None
    else:
        model_lower = 0.; regret = winner['objective'] if winner['feasible'] else None
    context.remember(key, winner['selected'])
    result = dict(winner)
    result.update(strategy=strategy, evaluations=len(candidates), simulation_passes=len(candidates),
                  candidate_menu_size=len(menu), pruned_candidates=[],
                  planner_seconds=perf_counter()-started,
                  candidate_summary=[dict(selected=r['selected'], feasible=r['feasible'],
                                          cost=r['cost'], objective=r['objective'],
                                          peak_bytes=r['peak_bytes'], reason=r['reason'])
                                     for _,r in sorted(candidates.items(),key=lambda pair:(len(pair[0]),tuple(sorted(pair[0]))))],
                  recovery_context_hit=previous is not None,
                  prior_selected_hint=None if previous is None else list(previous),
                  stop_reason=reason, no_eviction_certificate=certificate,
                  direct_score_count=direct_count,
                  lower_bound_candidates=len(bound_rows),
                  model_menu_lower_bound=model_lower, model_regret_upper_bound=regret,
                  bound_scope='current finite recovery menu, supplied costs and fixed next-use eviction; not wall time',
                  trial_cost_estimate_seconds=median(context.simulation_seconds) if context.simulation_seconds else 0.,
                  simulation_seconds=sum(simulation_durations),
                  decision_cost_multiplier=context.cost_multiplier,
                  max_evaluations=context.max_evaluations,
                  version_validation='current Problem only; cached selected set is an untrusted hint')
    return result
