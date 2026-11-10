"""Serializable observations, with explicit identity and accounting boundaries.

An arena extent is a suballocation, not a cudaMalloc. Native aggregate counters
never become invented physical allocations. No descriptor grants a controller
permission to mutate an owner in another backend/session.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import numpy as np


class Descriptor:
    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class NodeDomain(Descriptor):
    count: int
    order: str
    labels_sha256: str
    dtype: str = 'int64'

    @classmethod
    def from_nodes(cls, nodes):
        values = np.asarray(nodes)
        if values.ndim != 1 or values.dtype.kind not in 'iu':
            raise ValueError('Node domain requires sorted unique integer labels')
        if len(values) and (int(values.max()) >= 1 << 63 or int(values.min()) < -(1 << 63)):
            raise ValueError('Node domain exceeds signed int64')
        values = np.asarray(values, dtype='<i8')
        if np.any(values[1:] <= values[:-1]):
            raise ValueError('Node domain requires sorted unique labels')
        dense = not len(values) or (values[0] == 0 and values[-1] == len(values) - 1)
        return cls(len(values), 'dense_global_id' if dense else 'sorted_original_label',
                   hashlib.sha256(values.tobytes()).hexdigest())


@dataclass(frozen=True)
class GraphVersion(Descriptor):
    ref_generation: int | None
    logical_revision: int | None
    materialized_generation: int | None
    attribute_revision: int | None
    source: str


@dataclass(frozen=True)
class GraphDescriptor(Descriptor):
    graph_id: str
    identity_scope: str
    backend: str
    version: GraphVersion
    node_domain: NodeDomain
    directed: bool | None
    weighted: bool | None
    provenance: dict[str, Any]


@dataclass(frozen=True)
class CapabilityDescriptor(Descriptor):
    backend: str
    algorithms: tuple[str, ...]
    logical_devices: tuple[int, ...]
    mutations: str
    node_domain: str
    output: str
    physical_ownership: str
    cross_backend_owner_sharing: bool
    control_routes: tuple[str, ...]
    qualification: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class QueryContract(Descriptor):
    algorithm: str
    source_label: int | None
    weight_projection: str
    damping: float | None
    max_iterations: int | None
    tolerance_per_vertex: float | None
    unreachable: str | None
    weighted_dangling_threshold: float | None
    output_order: str = 'values[i] belongs to node_ids[i]'
    result_dtype: str = 'float64'


def query_contract(algorithm, parameters, count):
    pagerank = algorithm in ('pr', 'weighted_pr')
    return QueryContract(algorithm, parameters.get('source'),
                         'message_count' if algorithm in ('sssp', 'weighted_pr') else 'unweighted',
                         .85 if pagerank else None, 10000 if pagerank else None,
                         1e-6 / count if pagerank and count else None,
                         None if pagerank else '+inf', 1e-15 if algorithm == 'weighted_pr' else None)


@dataclass(frozen=True)
class PhysicalAllocation(Descriptor):
    id: str
    backend_key: str
    logical_device: int
    kind: str
    capacity_bytes: int
    payload_bytes: int
    arena_offset: int
    owners: tuple[str, ...]
    pins: int
    pending_release: bool
    source: str


@dataclass(frozen=True)
class ComponentState(Descriptor):
    id: str
    backend_key: str
    graph_id: str | None
    semantic: tuple[Any, ...]
    allocation_id: str
    dependency_stamp: dict[str, int]
    valid: bool
    source: str


@dataclass(frozen=True)
class HostOutputState(Descriptor):
    owner_id: str
    graph: GraphDescriptor
    algorithm: str
    payload_bytes: int
    memory_space: str = 'host'
    source: str = 'live NumPy output allocation weak reference; independent owning values'


@dataclass(frozen=True)
class StateSnapshot(Descriptor):
    schema: str
    session_id: str
    backend: str
    graphs: tuple[GraphDescriptor, ...]
    capabilities: CapabilityDescriptor
    allocations: tuple[PhysicalAllocation, ...]
    components: tuple[ComponentState, ...]
    live_host_outputs: tuple[HostOutputState, ...]
    counters: dict[str, Any]
    allocation_inventory_complete: bool
    accounting_scope: str
    consistency: dict[str, Any]
    limitations: tuple[str, ...]


def managed_inventory(engine, namespace):
    """Read existing metadata under the engine lock; never request a graph array."""
    model, arena = engine.model, engine.arena
    allocations, components = [], []
    for key, slot in arena.slots.items():
        allocation = model.allocations.get(key)
        workspace = str(key).startswith('@')
        if allocation is None and not workspace:
            raise RuntimeError(f'Arena extent has no ledger owner: {key}')
        device = int(slot.array.device.id)
        allocations.append(PhysicalAllocation(
            f'{namespace}/extent/{key}', str(key), device,
            'workspace_extent' if workspace else 'arena_extent',
            int(slot.capacity), int(slot.array.nbytes), int(slot.offset),
            (f'{namespace}/workspace',) if workspace else tuple(sorted(
                f'{namespace}/component/{owner}' for owner in allocation.owners)),
            0 if workspace else int(allocation.pins),
            False if workspace else bool(allocation.pending_release),
            'backend.session.arena.slots + model.allocations; disjoint extents within one reserved CUDA arena'))
    for key, component in model.components.items():
        semantic = tuple(engine.semantic.get(key, ()))
        if component.allocation_id not in arena.slots:
            raise RuntimeError(f'Component has no physical extent: {key}')
        components.append(ComponentState(
            f'{namespace}/component/{key}', str(key), semantic[0] if semantic else None,
            semantic, f'{namespace}/extent/{component.allocation_id}',
            {str(k): int(v) for k, v in component.stamp.items()}, bool(model.valid(key)),
            'backend.session.model.components/stamp/valid and semantic'))
    charged = sum(a.capacity_bytes for a in allocations)
    consistency = {'extent_capacity_bytes': charged, 'arena_live_bytes': int(arena.live),
                   'ledger_live_bytes': int(model.live_bytes), 'arena_budget_bytes': int(arena.budget),
                   'consistent': charged == int(arena.live) == int(model.live_bytes),
                   'ledger_revision': int(model.revision)}
    if not consistency['consistent']:
        raise RuntimeError(f'Physical/ledger mismatch: {consistency}')
    return tuple(allocations), tuple(components), consistency
