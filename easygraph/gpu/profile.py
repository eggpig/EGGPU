"""Observed query profiles and a fail-closed bridge to exact memory accounting.

Complete-call times are observations, never invented preparation-route prices.
Snapshots are consumed without touching a GPU or importing a native backend.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any

from .contracts import Descriptor, GraphDescriptor, StateSnapshot


@dataclass(frozen=True)
class CallObservation(Descriptor):
    schema: str
    graph: GraphDescriptor
    algorithm: str
    parameters: dict[str, Any]
    backend: str
    logical_devices: tuple[int, ...]
    complete_host_output_seconds: float
    output_bytes: int
    before: StateSnapshot
    after: StateSnapshot
    adapter_result_seconds: float
    timing_scope: str = 'external timer around complete session.query; snapshots and outer profiling-lock acquisition excluded'
    kind: str = 'measured_complete_call'
    recovery_route_cost_available: bool = False

    def require_current(self, session):
        current = session.describe(self.graph.graph_id)
        if current.identity_scope != self.graph.identity_scope or current.version != self.graph.version:
            raise ValueError('Stale or foreign graph profile')
        if current.node_domain != self.graph.node_domain:
            raise ValueError('Profile node domain changed')
        if session.capability_descriptor().logical_devices != self.logical_devices:
            raise ValueError('Profile device placement changed')


def profile_query(session, graph_id, algorithm, parameters=None):
    with session._lock:
        before = session.snapshot()
        begin = perf_counter()
        result = session.query(graph_id, algorithm, parameters)
        elapsed = perf_counter() - begin
        after = session.snapshot()
        if not math.isfinite(result.seconds) or result.seconds < 0:
            raise ValueError('Invalid measured call duration')
        return result, CallObservation('eggpu.call_observation.v1', result.graph, algorithm,
                                       dict(parameters or {}), result.graph.backend,
                                       after.capabilities.logical_devices, elapsed,
                                       int(result.values.nbytes), before, after, result.seconds)


def planner_memory_inventory(snapshot: StateSnapshot):
    """Return actual allocation capacity/ownership, refusing native aggregates.

This is an input fragment, not a complete recovery_planner.Problem: future
routes need measured preparation costs, source conditions and stage peaks.
Host outputs are deliberately separate from the device capacity ledger.
"""
    if not snapshot.allocation_inventory_complete or not snapshot.consistency.get('consistent'):
        raise ValueError('Exact planner needs a complete, consistent allocation inventory; aggregate counters are insufficient')
    return {
        'schema': 'eggpu.planner_memory_inventory.v1',
        'session_id': snapshot.session_id,
        'allocations': [a.to_dict() for a in snapshot.allocations],
        'components': [c.to_dict() for c in snapshot.components],
        'pinned_device_allocation_ids': [a.id for a in snapshot.allocations if a.pins or a.pending_release],
        'workspace_allocation_ids': [a.id for a in snapshot.allocations if a.kind == 'workspace_extent'],
        'live_host_outputs': [o.to_dict() for o in snapshot.live_host_outputs],
        'cross_backend_owner_sharing': False,
        'route_costs_available': False,
        'stage_peak_measurements_available': False,
        'scope': snapshot.accounting_scope,
    }
