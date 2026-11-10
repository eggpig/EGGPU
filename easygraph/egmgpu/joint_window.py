"""Finite calibrated route selection, used by the real native controller.

No GPU work, profiling, view construction, or formal-result selection lives here.
Capacity means declared retained allocations plus declared residual/base bounds.
Set dominance preserves minimum makespan and its minimum union charge; it need
not preserve the route-id tie-break of an unpruned search. Runtime claims concern
the finite calibrated model, not arbitrary online arrivals or interference.
Native caches can retain allocations after a call, so charges last until shutdown.
"""
from dataclasses import dataclass, field
import itertools
import json
import math
import time

from .deployment_planner import DeploymentPlanningError, NoFeasiblePlanError
from .types import ExecutionMode


@dataclass(frozen=True)
class WindowCall:
    function: object
    args: tuple
    route_ids: tuple
    contract_id: str
    kwargs: dict = field(default_factory=dict)


def resource_key(resource):
    # Device is always part of identity, even if a producer omits it in its key.
    return json.dumps([int(resource['device']), resource['identity']], sort_keys=True)


def route_resources(route):
    out = {}
    for item in route['resources']:
        key = resource_key(item)
        value = (int(item['device']), int(item['bytes']))
        if key in out and out[key] != value:
            raise DeploymentPlanningError('Conflicting resource identity/bytes')
        out[key] = value
    for device, size in route['residual_bound_bytes'].items():
        key = resource_key(dict(device=device, identity=['route-residual', route['input_id'], route['function'], route['contract_id'], route['residual_identity']]))
        if key in out:
            raise DeploymentPlanningError('Residual identity overlaps a declared resource')
        out[key] = (int(device), int(size))
    return out


def dominates(a, b):
    """Preserve makespan/union-charge objectives, not all route-id tie-breaks.

    Sufficient only for identical footprints and the retained-resource model.
    A different GPU may hide a strict solo-cost reduction in the final makespan.
    """
    if a['devices'] != b['devices'] or a['seconds'] > b['seconds']:
        return False
    ra, rb = route_resources(a), route_resources(b)
    if not ra.keys() <= rb.keys() or any(ra[k] != rb[k] for k in ra):
        return False
    return a['seconds'] < b['seconds'] or ra.keys() != rb.keys()


class WindowDeploymentPlanner:
    """At most four calls; predicted completion with frozen per-route costs.

An existing DeploymentPlanner remains the single-request AUTO planner. This
explicit candidate only selects already registered, compatible native routes.
"""
    scope = 'declared-retained-resource-union-plus-residual-and-base-bounds'

    def __init__(self, profile, devices, *, accounting='union', allow_synthetic=False):
        self.profile = json.loads(json.dumps(profile))  # no mutable caller aliases
        p = self.profile
        if p.get('schema') != 'egmgpu.joint-window-profile.v1':
            raise DeploymentPlanningError('Missing joint-window profile schema')
        self.synthetic = p.get('synthetic') is True
        if self.synthetic and not allow_synthetic:
            raise DeploymentPlanningError('Synthetic profiles require explicit CPU-test opt-in')
        if p.get('capacity_scope') != self.scope:
            raise DeploymentPlanningError('Capacity scope must include base and residual bounds')
        if p.get('frozen_before_formal') is not True:
            raise DeploymentPlanningError('A pre-formal frozen calibration is required')
        if not p.get('artifact_id') or not p.get('environment_id'):
            raise DeploymentPlanningError('Artifact/environment profile binding is required')
        if accounting not in ('union', 'additive'):
            raise DeploymentPlanningError('Unknown accounting control')
        self.accounting = accounting
        self.devices = tuple(devices)
        if not 1 <= len(self.devices) <= 8:
            raise DeploymentPlanningError("Finite planner supports one to eight devices")
        self.budgets = {int(d): int(n) for d, n in p['capacity_bytes'].items()}
        self.base = {int(d): int(n) for d, n in p['base_bound_bytes'].items()}
        if set(self.devices) != set(self.budgets) or set(self.base) != set(self.devices):
            raise DeploymentPlanningError('Every configured device needs capacity/base bounds')
        if any(self.budgets[d] <= 0 or not 0 <= self.base[d] <= self.budgets[d] for d in self.devices):
            raise DeploymentPlanningError('Invalid capacity/base bounds')
        self._evidence(p.get('base_evidence'), {'diagnostic', 'implementation_bound'})
        self.routes = {}
        identities = {}
        self._resources = {}
        self._charges = {}
        self._empty_queue_schedule_cache = {}
        self._empty_queue_exact_order_cache = {}
        for route in p['routes']:
            name = route['id']
            if name in self.routes:
                raise DeploymentPlanningError('Duplicate route id')
            group = tuple(route['devices'])
            if not group or tuple(sorted(set(group))) != group or not set(group) <= set(self.devices):
                raise DeploymentPlanningError('Invalid route device group')
            mode = ExecutionMode.parse(route['mode'])
            if mode is ExecutionMode.AUTO:
                raise DeploymentPlanningError('A candidate must name an implemented mode')
            route['seconds'] = float(route['seconds'])
            if not math.isfinite(route['seconds']) or route['seconds'] <= 0:
                raise DeploymentPlanningError('A positive calibrated cost is required')
            for field_name in ('layout', 'function', 'contract_id', 'residual_identity', 'residual_coverage', 'input_id'):
                if not route.get(field_name):
                    raise DeploymentPlanningError('Missing route field: '+field_name)
            if set(map(int, route['residual_bound_bytes'])) != set(group):
                raise DeploymentPlanningError('Each route device needs a non-CSR residual bound')
            if 'arguments' not in route or 'vertices' not in route or 'directed' not in route:
                raise DeploymentPlanningError('Input dimensions and exact arguments must be profiled')
            self._evidence(route.get('latency_evidence'), {'pilot'})
            self._evidence(route.get('memory_evidence'), {'diagnostic', 'implementation_bound'})
            compiled = route_resources(route)
            self._resources[name] = compiled
            self._charges[name] = (compiled if accounting == "union" else
                {json.dumps([name, k]): v for k, v in compiled.items()})
            for key, value in compiled.items():
                if value[0] not in group or value[1] < 0:
                    raise DeploymentPlanningError('Invalid resource device/bytes')
                if key in identities and identities[key] != value:
                    raise DeploymentPlanningError('One resource identity has inconsistent bytes')
                identities[key] = value
            self.routes[name] = route
        if not self.routes:
            raise DeploymentPlanningError('No calibrated routes')

    def _evidence(self, evidence, phases):
        if not isinstance(evidence, dict) or evidence.get('phase') not in phases:
            raise DeploymentPlanningError('Missing independent pilot/memory evidence')
        if not evidence.get('record') or not evidence.get('scope'):
            raise DeploymentPlanningError('Evidence record and boundary are required')
        if not self.synthetic and evidence.get('synthetic'):
            raise DeploymentPlanningError('Synthetic evidence cannot calibrate a measured profile')

    def candidates(self, call):
        if not isinstance(call, WindowCall) or not callable(call.function) or not call.route_ids:
            raise DeploymentPlanningError('A WindowCall with explicit candidate routes is required')
        if len(call.route_ids) > 16 or len(set(call.route_ids)) != len(call.route_ids):
            raise DeploymentPlanningError("At most sixteen unique preprofiled routes per call")
        found = []
        for name in call.route_ids:
            if name not in self.routes:
                raise DeploymentPlanningError('Unprofiled route: '+str(name))
            r = self.routes[name]
            if r['function'] != call.function.__name__ or r['contract_id'] != call.contract_id:
                raise DeploymentPlanningError('Route/function/input-result contract mismatch')
            found.append(r)
        return found

    def validate_call_binding(self, call, snapshot):
        arguments = json.loads(json.dumps(dict(args=list(call.args[1:]), kwargs=call.kwargs)))
        for route in self.candidates(call):
            if (getattr(call.args[0], '_joint_input_id', None) != route['input_id']
                or snapshot.vertex_count != route['vertices']
                or snapshot.directed != route['directed']
                or arguments != route['arguments']):
                raise DeploymentPlanningError('Graph identity/direction/dimensions or arguments differ from profile')

    def charges(self, route):
        return self._charges[route['id']]

    def _dominates(self, a, b):
        if a['devices'] != b['devices'] or a['seconds'] > b['seconds']:
            return False
        ra, rb = self._resources[a['id']], self._resources[b['id']]
        return (ra.keys() <= rb.keys() and all(ra[k] == rb[k] for k in ra)
                and (a['seconds'] < b['seconds'] or ra.keys() != rb.keys()))

    def select(self, candidates, retained, queued_seconds):
        started = time.perf_counter()
        if not 1 <= len(candidates) <= 4:
            raise DeploymentPlanningError('Joint window must contain one to four calls')
        before = sum(map(len, candidates))
        if self.accounting == 'union':
            candidates = [[r for r in rs if not any(self._dominates(q, r) for q in rs if q is not r)] for rs in candidates]
        usage = dict(self.base)
        for device, size in retained.values():
            usage[device] += size
        usage_before = dict(usage)
        if any(usage[d] > self.budgets[d] for d in self.devices):
            raise NoFeasiblePlanError('Previously retained resources exceed capacity')
        resources = dict(retained)
        best = None
        considered = feasible = pruned = 0
        initial = tuple(float(queued_seconds.get(d, 0)) for d in self.devices)
        schedule_cache = self._empty_queue_schedule_cache if not any(initial) else {}
        candidate_signatures = [tuple(sorted(r["id"] for r in rs)) for rs in candidates]
        prior_equal = [max((j for j in range(i) if candidate_signatures[j] == candidate_signatures[i]), default=-1)
                       for i in range(len(candidates))]
        positions = {d: i for i, d in enumerate(self.devices)}
        compiled_schedule = {name: (tuple(positions[d] for d in r['devices']), r['seconds'])
                             for name, r in self.routes.items()}

        def schedule(routes):
            # Repeated requests share the same route costs. Evaluate each unique
            # route sequence once per multiset, then restore exact index tie-break.
            ids = tuple(r["id"] for r in routes)
            if not any(initial) and ids in self._empty_queue_exact_order_cache:
                return self._empty_queue_exact_order_cache[ids]
            multiset = tuple(sorted(ids))
            if multiset not in schedule_cache:
                optimum = float('inf'); winners = []
                for sequence in sorted(set(itertools.permutations(multiset))):
                    available = list(initial)
                    for name in sequence:
                        group, seconds = compiled_schedule[name]
                        end = max(available[d] for d in group) + seconds
                        for d in group: available[d] = end
                    makespan = max(available)
                    if makespan < optimum: optimum, winners = makespan, [sequence]
                    elif makespan == optimum: winners.append(sequence)
                schedule_cache[multiset] = optimum, winners
            optimum, winners = schedule_cache[multiset]
            orders = []
            for sequence in winners:
                remaining = list(range(len(routes))); order = []
                for name in sequence:
                    i = next(i for i in remaining if routes[i]['id'] == name)
                    remaining.remove(i); order.append(i)
                orders.append(tuple(order))
            answer = optimum, min(orders)
            if not any(initial): self._empty_queue_exact_order_cache[ids] = answer
            return answer

        def visit(prefix):
            nonlocal best, considered, feasible, pruned
            if len(prefix) == len(candidates):
                considered += 1; feasible += 1
                makespan, order = schedule(prefix)
                score = (makespan, sum(usage.values()), tuple(r['id'] for r in prefix), order)
                if best is None or score < best[0]:
                    best = score, tuple(prefix), order, dict(resources), dict(usage)
                return
            for route in candidates[len(prefix)]:
                previous = prior_equal[len(prefix)]
                if previous >= 0 and route["id"] < prefix[previous]["id"]:
                    # Swapping equal-candidate requests leaves capacities and
                    # best makespan unchanged. This symmetry step keeps the
                    # route-id tie-break within the surviving candidate sets;
                    # preceding dominance need not preserve the unpruned tie-break.
                    continue
                added = []
                for key, value in self.charges(route).items():
                    if key in resources:
                        if resources[key] != value:
                            raise DeploymentPlanningError('Retained resource identity conflict')
                    else:
                        resources[key] = value; added.append(key); usage[value[0]] += value[1]
                # All bounds are nonnegative and charges persist. A partial
                # assignment over capacity cannot regain feasibility downstream.
                if all(usage[d] <= self.budgets[d] for d in self.devices):
                    visit(prefix + [route])
                else:
                    pruned += 1
                    considered += math.prod(len(rs) for rs in candidates[len(prefix)+1:])
                for key in added:
                    device, size = resources.pop(key); usage[device] -= size
        visit([])
        if best is None:
            raise NoFeasiblePlanError('No calibrated joint route fits retained capacity; no fallback')
        score, routes, order, resources, usage = best
        return dict(routes=routes, order=order, retained=resources, usage=usage,
                    usage_before=usage_before, resource_union_before=sorted(retained),
                    resource_union_after=sorted(resources),
                    new_resource_keys=sorted(resources.keys()-retained.keys()),
                    predicted_makespan_seconds=score[0], combinations=considered,
                    feasible_combinations=feasible, capacity_pruned_branches=pruned,
                    unique_schedule_multisets=len(schedule_cache),
                    equivalent_call_symmetry_reduced=True,
                    selection_seconds=time.perf_counter()-started,
                    dominated_routes=before-sum(map(len,candidates)))
