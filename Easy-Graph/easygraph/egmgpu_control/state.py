"""Versioned multi-device graph-state metadata and CPU-side residency policy."""

from __future__ import annotations

import threading

from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Dict
from typing import Hashable
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple


MAX_LOCAL_ADJACENCY_ENTRIES = (1 << 31) - 1
# Manifest v1 and the current CUDA ABI store local IDs in signed int32.  A
# shard may contain 2^31 local vertices (IDs 0..2^31-1), while global IDs and
# aggregate counts remain uint64.
MAX_LOCAL_VERTICES = 1 << 31


class GraphStateError(RuntimeError):
    """Base class for invalid graph-state operations."""


class StaleSnapshotError(GraphStateError):
    """Raised when state is installed for an obsolete graph generation."""


class CapacityError(GraphStateError):
    """Raised when no evictable state can satisfy a device memory budget."""


@dataclass(frozen=True)
class GraphSnapshot:
    """Immutable host-graph identity observed at request submission.

    ``generation`` must advance whenever topology or relevant edge attributes
    change.  The two signatures let a native integration reject accidental
    metadata reuse even when a caller fails to advance the generation.
    """

    graph_id: Hashable
    generation: int
    vertex_count: int
    adjacency_entries: int
    topology_signature: str
    weight_signature: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            hash(self.graph_id)
        except TypeError as exc:
            raise TypeError("graph_id must be hashable") from exc
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if self.vertex_count < 0 or self.adjacency_entries < 0:
            raise ValueError("graph sizes must be non-negative")
        if not self.topology_signature:
            raise ValueError("topology_signature must be non-empty")

    def mutated(
        self,
        *,
        topology_signature: Optional[str] = None,
        weight_signature: Optional[str] = None,
        vertex_count: Optional[int] = None,
        adjacency_entries: Optional[int] = None,
    ) -> "GraphSnapshot":
        """Return the next immutable generation after a graph mutation."""

        return replace(
            self,
            generation=self.generation + 1,
            topology_signature=(
                self.topology_signature
                if topology_signature is None
                else topology_signature
            ),
            weight_signature=(
                self.weight_signature if weight_signature is None else weight_signature
            ),
            vertex_count=(self.vertex_count if vertex_count is None else vertex_count),
            adjacency_entries=(
                self.adjacency_entries
                if adjacency_entries is None
                else adjacency_entries
            ),
        )


@dataclass(frozen=True)
class ViewKey:
    """Graph generation plus the semantic signature of a prepared view."""

    graph_id: Hashable
    generation: int
    signature: str

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if not self.signature:
            raise ValueError("view signature must be non-empty")

    @classmethod
    def for_snapshot(cls, snapshot: GraphSnapshot, signature: str) -> "ViewKey":
        return cls(snapshot.graph_id, snapshot.generation, signature)


@dataclass(frozen=True)
class StateKey:
    """Identity of a reusable view on an ordered, canonical device group.

    The flattened key is exactly
    ``(graph_id, generation, view_signature, device_group, layout)``.
    """

    view: ViewKey
    device_group: Tuple[int, ...]
    layout: str

    def __post_init__(self) -> None:
        canonical = tuple(sorted(set(self.device_group)))
        if not canonical:
            raise ValueError("device_group must not be empty")
        if canonical != self.device_group:
            raise ValueError("device_group must be sorted and contain unique devices")
        if any(device < 0 for device in canonical):
            raise ValueError("device identifiers must be non-negative")
        if not self.layout:
            raise ValueError("layout must be non-empty")

    @property
    def graph_id(self) -> Hashable:
        return self.view.graph_id

    @property
    def generation(self) -> int:
        return self.view.generation

    @property
    def view_signature(self) -> str:
        return self.view.signature

    def as_tuple(self) -> Tuple[Any, ...]:
        return (
            self.graph_id,
            self.generation,
            self.view_signature,
            self.device_group,
            self.layout,
        )


@dataclass(frozen=True)
class PartitionMetadata:
    """Contiguous vertex-owner partition metadata for one device."""

    partition_id: int
    device_id: int
    vertex_begin: int
    vertex_end: int
    adjacency_entries: int
    halo_vertices: int = 0
    cut_edges: int = 0
    local_id_bits: int = 32

    def __post_init__(self) -> None:
        if self.partition_id < 0 or self.device_id < 0:
            raise ValueError("partition and device identifiers must be non-negative")
        if self.vertex_begin < 0 or self.vertex_end < self.vertex_begin:
            raise ValueError("invalid half-open vertex range")
        if self.adjacency_entries < 0 or self.halo_vertices < 0 or self.cut_edges < 0:
            raise ValueError("partition sizes must be non-negative")
        if self.local_id_bits != 32:
            raise ValueError("the current partition contract uses 32-bit local IDs")
        if self.local_vertex_count + self.halo_vertices > MAX_LOCAL_VERTICES:
            raise ValueError("partition exceeds the 32-bit local vertex-ID domain")
        if self.adjacency_entries > MAX_LOCAL_ADJACENCY_ENTRIES:
            raise ValueError("partition exceeds the signed 32-bit adjacency-entry limit")

    @property
    def local_vertex_count(self) -> int:
        return self.vertex_end - self.vertex_begin


def validate_partition_set(
    snapshot: GraphSnapshot,
    partitions: Sequence[PartitionMetadata],
    device_group: Optional[Sequence[int]] = None,
) -> None:
    """Validate complete, non-overlapping ownership of all global vertices."""

    if not partitions:
        raise ValueError("at least one partition is required")
    ordered = sorted(partitions, key=lambda item: item.vertex_begin)
    if ordered[0].vertex_begin != 0 or ordered[-1].vertex_end != snapshot.vertex_count:
        raise ValueError("partitions must cover the complete global vertex range")
    for left, right in zip(ordered, ordered[1:]):
        if left.vertex_end != right.vertex_begin:
            raise ValueError("partition vertex ranges must be contiguous and disjoint")
    partition_ids = [item.partition_id for item in partitions]
    if len(set(partition_ids)) != len(partition_ids):
        raise ValueError("partition identifiers must be unique")
    devices = [item.device_id for item in partitions]
    if len(set(devices)) != len(devices):
        raise ValueError("one partition per device is required")
    if device_group is not None and set(devices) != set(device_group):
        raise ValueError("partition devices do not match the state device group")


@dataclass(frozen=True)
class ResidencyRecord:
    """Read-only residency information returned to control-plane clients."""

    key: StateKey
    device_id: int
    size_bytes: int
    partition: Optional[PartitionMetadata]
    stale: bool
    lease_count: int
    evictable: bool
    last_access: int


@dataclass
class _ResidentShard:
    key: StateKey
    device_id: int
    size_bytes: int
    partition: Optional[PartitionMetadata]
    payload: Any
    stale: bool
    lease_count: int
    evictable: bool
    last_access: int

    def public(self) -> ResidencyRecord:
        return ResidencyRecord(
            key=self.key,
            device_id=self.device_id,
            size_bytes=self.size_bytes,
            partition=self.partition,
            stale=self.stale,
            lease_count=self.lease_count,
            evictable=self.evictable,
            last_access=self.last_access,
        )


class StateLease:
    """Pins all shards of one logical state until explicitly released."""

    def __init__(self, owner: "MultiDeviceGraphState", key: StateKey) -> None:
        self._owner = owner
        self.key = key
        self._released = False

    def payload(self, device_id: int) -> Any:
        return self._owner._leased_payload(self.key, device_id)

    def release(self) -> None:
        if not self._released:
            self._owner._release(self.key)
            self._released = True

    def __enter__(self) -> "StateLease":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


class MultiDeviceGraphState:
    """Thread-safe reference state store with generation invalidation and LRU.

    Memory budgets describe the portion of each device assigned to reusable
    graph state, not total device memory.  Admission and eviction are performed
    atomically across a logical multi-device view.
    """

    def __init__(self, memory_budgets: Mapping[int, int]) -> None:
        if not memory_budgets:
            raise ValueError("at least one device memory budget is required")
        self._budgets = dict(memory_budgets)
        if any(device < 0 for device in self._budgets):
            raise ValueError("device identifiers must be non-negative")
        if any(size < 0 for size in self._budgets.values()):
            raise ValueError("memory budgets must be non-negative")
        self._entries: Dict[Tuple[StateKey, int], _ResidentShard] = {}
        self._generation: Dict[Hashable, int] = {}
        self._snapshot: Dict[Tuple[Hashable, int], GraphSnapshot] = {}
        self._clock = 0
        self._lock = threading.RLock()

    @property
    def devices(self) -> Tuple[int, ...]:
        return tuple(sorted(self._budgets))

    def observe_snapshot(self, snapshot: GraphSnapshot) -> None:
        """Register a generation and invalidate older, unleased resident state."""

        with self._lock:
            current = self._generation.get(snapshot.graph_id)
            if current is not None and snapshot.generation < current:
                raise StaleSnapshotError(
                    f"graph {snapshot.graph_id!r} generation {snapshot.generation} "
                    f"is older than observed generation {current}"
                )
            existing = self._snapshot.get((snapshot.graph_id, snapshot.generation))
            if existing is not None and existing != snapshot:
                raise GraphStateError(
                    "snapshot metadata changed without advancing graph generation"
                )
            self._snapshot[(snapshot.graph_id, snapshot.generation)] = snapshot
            if current is None or snapshot.generation > current:
                self._generation[snapshot.graph_id] = snapshot.generation
                obsolete = {
                    key
                    for key, _device in self._entries
                    if key.graph_id == snapshot.graph_id
                    and key.generation < snapshot.generation
                }
                for key in obsolete:
                    for shard in self._shards(key):
                        shard.stale = True
                    if all(shard.lease_count == 0 for shard in self._shards(key)):
                        self._remove_key(key)

    def install(
        self,
        key: StateKey,
        snapshot: GraphSnapshot,
        size_bytes: Mapping[int, int],
        *,
        partitions: Optional[Sequence[PartitionMetadata]] = None,
        payloads: Optional[Mapping[int, Any]] = None,
        evictable: bool = True,
    ) -> None:
        """Atomically admit all device shards for one logical prepared view."""

        with self._lock:
            self.observe_snapshot(snapshot)
            if key.graph_id != snapshot.graph_id or key.generation != snapshot.generation:
                raise ValueError("state key does not identify the supplied snapshot")
            if self._generation[key.graph_id] != key.generation:
                raise StaleSnapshotError("cannot install state for an obsolete generation")
            existing_shards = self._shards(key)
            if any(shard.lease_count > 0 for shard in existing_shards):
                raise GraphStateError("cannot replace graph state with active leases")
            sizes = dict(size_bytes)
            if set(sizes) != set(key.device_group):
                raise ValueError("size_bytes must contain exactly the state device group")
            if any(size < 0 for size in sizes.values()):
                raise ValueError("resident sizes must be non-negative")
            unknown = set(sizes) - set(self._budgets)
            if unknown:
                raise ValueError(f"missing memory budgets for devices {sorted(unknown)}")

            partition_map: Dict[int, PartitionMetadata] = {}
            if partitions is not None:
                validate_partition_set(snapshot, partitions, key.device_group)
                partition_map = {part.device_id: part for part in partitions}
            supplied_payloads = {} if payloads is None else dict(payloads)
            if set(supplied_payloads) - set(key.device_group):
                raise ValueError("payload supplied for a device outside the state group")

            old_sizes = {shard.device_id: shard.size_bytes for shard in existing_shards}
            projected = {
                device: self.usage_bytes(device)
                - old_sizes.get(device, 0)
                + sizes.get(device, 0)
                for device in self._budgets
            }
            victims = self._plan_evictions(projected, protected=key)
            for victim in victims:
                self._remove_key(victim)
            self._remove_key(key)

            self._clock += 1
            for device in key.device_group:
                self._entries[(key, device)] = _ResidentShard(
                    key=key,
                    device_id=device,
                    size_bytes=sizes[device],
                    partition=partition_map.get(device),
                    payload=supplied_payloads.get(device),
                    stale=False,
                    lease_count=0,
                    evictable=evictable,
                    last_access=self._clock,
                )

    def acquire(self, key: StateKey) -> StateLease:
        """Lease a complete current-generation state for an in-flight request."""

        with self._lock:
            shards = self._shards(key)
            if len(shards) != len(key.device_group) or any(shard.stale for shard in shards):
                raise KeyError("state is not completely resident and current")
            self._clock += 1
            for shard in shards:
                shard.lease_count += 1
                shard.last_access = self._clock
            return StateLease(self, key)

    def _release(self, key: StateKey) -> None:
        with self._lock:
            shards = self._shards(key)
            if not shards or any(shard.lease_count <= 0 for shard in shards):
                raise GraphStateError("unbalanced state lease release")
            for shard in shards:
                shard.lease_count -= 1
            if all(shard.stale and shard.lease_count == 0 for shard in shards):
                self._remove_key(key)

    def payload(self, key: StateKey, device_id: int) -> Any:
        """Return payload only while the state is current and resident."""

        with self._lock:
            shard = self._entries.get((key, device_id))
            if shard is None or shard.stale:
                raise KeyError((key, device_id))
            return shard.payload

    def _leased_payload(self, key: StateKey, device_id: int) -> Any:
        """Permit an in-flight lease to finish on a now-stale generation."""

        with self._lock:
            shard = self._entries.get((key, device_id))
            if shard is None or shard.lease_count <= 0:
                raise KeyError((key, device_id))
            return shard.payload

    def is_resident(
        self, key: StateKey, devices: Optional[Iterable[int]] = None
    ) -> bool:
        requested = key.device_group if devices is None else tuple(devices)
        with self._lock:
            current = self._generation.get(key.graph_id)
            return current == key.generation and all(
                (key, device) in self._entries
                and not self._entries[(key, device)].stale
                for device in requested
            )

    def records(self, *, include_stale: bool = False) -> Tuple[ResidencyRecord, ...]:
        with self._lock:
            return tuple(
                shard.public()
                for shard in sorted(
                    self._entries.values(),
                    key=lambda item: (repr(item.key.as_tuple()), item.device_id),
                )
                if include_stale or not shard.stale
            )

    def usage_bytes(self, device_id: int) -> int:
        if device_id not in self._budgets:
            raise KeyError(device_id)
        return sum(
            shard.size_bytes
            for (_key, device), shard in self._entries.items()
            if device == device_id
        )

    def clear(self) -> None:
        with self._lock:
            if any(shard.lease_count for shard in self._entries.values()):
                raise GraphStateError("cannot clear graph state with active leases")
            self._entries.clear()

    def _shards(self, key: StateKey) -> list[_ResidentShard]:
        return [
            self._entries[(key, device)]
            for device in key.device_group
            if (key, device) in self._entries
        ]

    def _remove_key(self, key: StateKey) -> None:
        for device in key.device_group:
            self._entries.pop((key, device), None)

    def _plan_evictions(
        self, projected: Mapping[int, int], *, protected: StateKey
    ) -> Tuple[StateKey, ...]:
        usage = dict(projected)
        if all(usage[d] <= self._budgets[d] for d in self._budgets):
            return ()
        logical_keys = {key for key, _device in self._entries if key != protected}
        candidates = []
        for key in logical_keys:
            shards = self._shards(key)
            if shards and all(
                shard.evictable and shard.lease_count == 0 for shard in shards
            ):
                candidates.append(
                    (
                        0 if any(shard.stale for shard in shards) else 1,
                        max(shard.last_access for shard in shards),
                        repr(key.as_tuple()),
                        key,
                    )
                )
        candidates.sort(key=lambda item: item[:3])
        victims = []
        for _stale_rank, _last_access, _stable_key, key in candidates:
            if all(usage[d] <= self._budgets[d] for d in self._budgets):
                break
            shards = self._shards(key)
            if not any(
                usage[shard.device_id] > self._budgets[shard.device_id]
                for shard in shards
            ):
                continue
            victims.append(key)
            for shard in shards:
                usage[shard.device_id] -= shard.size_bytes
        if any(usage[d] > self._budgets[d] for d in self._budgets):
            details = ", ".join(
                f"gpu{device}: {usage[device]}/{self._budgets[device]} bytes"
                for device in sorted(self._budgets)
                if usage[device] > self._budgets[device]
            )
            raise CapacityError(f"insufficient evictable graph-state capacity ({details})")
        return tuple(victims)
