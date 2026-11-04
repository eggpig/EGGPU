"""Versioned graph snapshots with lease-based reclamation.

This module separates *observing* a mutable Python graph from owning an
immutable native representation.  With the production extension, the default
provider copies a regular EasyGraph object and builds a private native CSR
while request submission owns the Python interpreter lock.  Read-only bulk CSR
graphs retain their existing native handle.  CPU-only imports or unsupported
graph-like objects remain metadata-only instead of being presented as frozen.

The public fields shared with ``egmgpu_control.state.GraphSnapshot`` are kept
duck-type compatible: ``graph_id``, ``generation``, ``vertex_count``,
``adjacency_entries``, ``topology_signature``, and ``weight_signature``.
There is deliberately no import between the runtime and control-plane
packages.
"""

from __future__ import annotations

import threading
import weakref

from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from types import MappingProxyType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Hashable
from typing import Mapping
from typing import Optional
from typing import Tuple


class GraphSnapshotError(RuntimeError):
    """Base exception for graph snapshot lifecycle failures."""


class SnapshotCaptureError(GraphSnapshotError):
    """A stable graph generation could not be observed."""


class SnapshotConsistencyError(GraphSnapshotError):
    """Graph metadata changed without a corresponding generation change."""


class StaleGraphGenerationError(GraphSnapshotError):
    """A graph reported a generation older than one already observed."""


class NativePayloadUnavailableError(GraphSnapshotError):
    """A caller requested native state from a metadata-only snapshot."""


class GraphGenerationChangedError(GraphSnapshotError):
    """A mutable graph changed after request submission."""


@dataclass(frozen=True)
class SnapshotObservation:
    """Submission-time metadata read from one graph generation."""

    graph_id: Hashable
    generation: int
    directed: bool
    vertex_count: int
    edge_count: int
    adjacency_entries: int
    adjacency_entries_exact: bool
    generation_tracked: bool
    source_is_immutable: bool

    def __post_init__(self) -> None:
        try:
            hash(self.graph_id)
        except TypeError as exc:
            raise TypeError("graph_id must be hashable") from exc
        if self.generation < 0:
            raise ValueError("graph generation must be non-negative")
        if min(self.vertex_count, self.edge_count, self.adjacency_entries) < 0:
            raise ValueError("graph sizes must be non-negative")


@dataclass(frozen=True)
class NativeSnapshotPayload:
    """Provider result for an immutable native graph representation.

    Providers must set ``immutable=True`` before the registry marks the
    payload ready.  ``release`` is called once when the corresponding snapshot
    is reclaimed; bulk-CSR handles do not need one because their graph object
    owns the native handle.
    """

    payload: Any
    immutable: bool
    topology_signature: Optional[str] = None
    weight_signature: Optional[str] = None
    release: Optional[Callable[[Any], None]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.payload is None and self.immutable:
            raise ValueError("an immutable native snapshot payload cannot be None")
        if self.release is not None and not callable(self.release):
            raise TypeError("native payload release hook must be callable")


@dataclass(frozen=True)
class FrozenNativeGraph:
    """Private native CSR plus the public node order for one generation.

    ``native_graph`` is a pybind ``Graph`` whose CSR was fully built while
    the GIL was held.  Fixed GPU workers retain only native buffers from this
    object.  ``public_nodes`` freezes the index-to-node mapping so result
    construction never consults a Python graph that may have mutated while
    the request was running.
    """

    native_graph: Any = field(repr=False, compare=False)
    public_nodes: Tuple[Any, ...]
    public_index: Mapping[Any, int] = field(repr=False, compare=False)
    directed: bool

    def __post_init__(self) -> None:
        if self.native_graph is None:
            raise ValueError("a frozen native graph requires a native handle")
        index = dict(self.public_index)
        if len(index) != len(self.public_nodes):
            raise ValueError("frozen public node labels must be unique")
        for position, node in enumerate(self.public_nodes):
            if index.get(node) != position:
                raise ValueError("frozen public node index does not match node order")
        object.__setattr__(self, "public_index", MappingProxyType(index))


def native_graph_handle(payload: Any) -> Any:
    """Return the pybind graph carried by either snapshot payload form."""

    if isinstance(payload, FrozenNativeGraph):
        return payload.native_graph
    return payload


def snapshot_public_nodes(snapshot: "GraphSnapshot") -> Tuple[Any, ...]:
    """Return the immutable public node order retained by a snapshot."""

    payload = snapshot.require_native_payload()
    if isinstance(payload, FrozenNativeGraph):
        return payload.public_nodes
    return tuple(range(snapshot.vertex_count))


@dataclass(frozen=True)
class GraphSnapshot:
    """Immutable identity and metadata for one observed graph generation."""

    graph_id: Hashable
    generation: int
    vertex_count: int
    adjacency_entries: int
    topology_signature: str
    weight_signature: Optional[str] = None
    directed: bool = False
    edge_count: int = 0
    adjacency_entries_exact: bool = False
    generation_tracked: bool = True
    source_is_immutable: bool = False
    native_payload_ready: bool = False
    native_payload: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("graph generation must be non-negative")
        if min(self.vertex_count, self.edge_count, self.adjacency_entries) < 0:
            raise ValueError("graph sizes must be non-negative")
        if not self.topology_signature:
            raise ValueError("topology_signature must be non-empty")
        if self.native_payload_ready and self.native_payload is None:
            raise ValueError("a ready native snapshot requires a payload")

    def require_native_payload(self) -> Any:
        """Return immutable native state or fail instead of using Python state."""

        if not self.native_payload_ready:
            raise NativePayloadUnavailableError(
                "graph generation {} has metadata only; no immutable native "
                "snapshot is available".format(self.generation)
            )
        return self.native_payload


@dataclass(frozen=True)
class SnapshotRecord:
    """Read-only lifecycle information exposed for tests and observability."""

    graph_id: Hashable
    generation: int
    lease_count: int
    stale: bool
    native_payload_ready: bool


@dataclass
class _SnapshotEntry:
    snapshot: GraphSnapshot
    lease_count: int
    stale: bool
    cleanup: Optional[Callable[[Any], None]]


class _GraphRecord:
    def __init__(self, graph: Any, collected: Callable[[weakref.ref], None]) -> None:
        self.capture_lock = threading.Lock()
        self.execution_lock = threading.Lock()
        self.current_generation: Optional[int] = None
        self.entries: Dict[int, _SnapshotEntry] = {}
        try:
            self.weak_graph = weakref.ref(graph, collected)
            self.strong_graph = None
        except TypeError:
            # EasyGraph graph classes are weak-referenceable.  This fallback
            # keeps custom graph-like objects correct and makes their lifetime
            # explicit through GraphSnapshotRegistry.retire().
            self.weak_graph = None
            self.strong_graph = graph

    def refers_to(self, graph: Any) -> bool:
        if self.weak_graph is not None:
            return self.weak_graph() is graph
        return self.strong_graph is graph


class GraphSnapshotLease:
    """Pins one snapshot until a request completes or the context exits."""

    def __init__(
        self,
        owner: "GraphSnapshotRegistry",
        graph: Any,
        snapshot: GraphSnapshot,
    ) -> None:
        self._owner = owner
        # Prevent object-id reuse while a request still owns the snapshot.
        self._graph = graph
        self.snapshot = snapshot
        self._release_lock = threading.Lock()
        self._released = False

    @property
    def released(self) -> bool:
        with self._release_lock:
            return self._released

    @property
    def native_payload_ready(self) -> bool:
        return self.snapshot.native_payload_ready

    @property
    def concurrent_execution_safe(self) -> bool:
        """Whether requests can read this generation without the source graph.

        A mutable source is safe once capture produced a private immutable
        native payload: an in-flight request keeps that payload while a
        mutation advances the source graph to a new generation.
        """

        return bool(self.snapshot.native_payload_ready)

    def require_native_payload(self) -> Any:
        return self.snapshot.require_native_payload()

    def assert_generation_current(self) -> None:
        """Reject a mutable graph changed after this lease was acquired."""

        self._owner._assert_generation_current(self)

    @contextmanager
    def execution_guard(self, cancelled: Optional[Callable[[], bool]] = None):
        """Serialize metadata-only calls on one graph without reserving a GPU.

        The yielded value is ``False`` only when cancellation wins while the
        request is waiting for the per-graph lock.
        """

        if self.concurrent_execution_safe:
            yield True
            return
        lock = self._owner._execution_lock(self)
        acquired = False
        try:
            while not acquired:
                if cancelled is not None and cancelled():
                    yield False
                    return
                acquired = lock.acquire(timeout=0.05)
            yield True
        finally:
            if acquired:
                lock.release()

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            graph = self._graph
            try:
                self._owner._release(
                    graph, self.snapshot.graph_id, self.snapshot.generation
                )
            finally:
                # Reclamation is logically complete even if a provider's
                # native cleanup callback reports an error.
                self._released = True
                self._graph = None

    def __enter__(self) -> "GraphSnapshotLease":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except BaseException:
            # Destructors cannot safely report native cleanup errors.  Normal
            # request paths always release explicitly or through ``with``.
            pass


SnapshotProvider = Callable[
    [Any, SnapshotObservation], Optional[NativeSnapshotPayload]
]


class GraphSnapshotRegistry:
    """Thread-safe generation registry with lease-based stale reclamation."""

    def __init__(
        self,
        snapshot_provider: Optional[SnapshotProvider] = None,
        capture_retries: int = 3,
    ) -> None:
        if capture_retries < 1:
            raise ValueError("capture_retries must be at least one")
        self._provider = (
            self._default_snapshot_provider
            if snapshot_provider is None
            else snapshot_provider
        )
        if not callable(self._provider):
            raise TypeError("snapshot_provider must be callable")
        self._capture_retries = int(capture_retries)
        self._lock = threading.RLock()
        self._records: Dict[int, _GraphRecord] = {}

    def acquire(self, graph: Any) -> GraphSnapshotLease:
        """Capture the graph generation and return a lease for one request."""

        record = self._record_for(graph)
        with record.capture_lock:
            for _ in range(self._capture_retries):
                observation = self._observe(graph)
                existing = self._acquire_existing(graph, record, observation)
                if existing is not None:
                    return existing

                native = self._provider(graph, observation)
                if native is not None and not isinstance(native, NativeSnapshotPayload):
                    raise TypeError(
                        "snapshot_provider must return NativeSnapshotPayload or None"
                    )
                if native is not None and not native.immutable:
                    self._release_uninstalled(native)
                    raise SnapshotCaptureError(
                        "snapshot_provider returned a mutable native payload"
                    )

                generation_after = self._read_generation(graph)
                if generation_after != observation.generation:
                    self._release_uninstalled(native)
                    continue

                snapshot = self._make_snapshot(observation, native)
                return self._install_and_acquire(graph, record, snapshot, native)

        raise SnapshotCaptureError(
            "graph mutation prevented a stable snapshot after {} attempts".format(
                self._capture_retries
            )
        )

    capture = acquire

    def retain(self, lease: GraphSnapshotLease) -> GraphSnapshotLease:
        """Create another lease for the exact generation already pinned.

        Unlike :meth:`acquire`, this operation does not observe the mutable
        source graph.  It is used by an explicit workflow after one atomic
        snapshot capture so every submitted call reads the same generation,
        even when a later mutation makes that generation stale for new calls.
        """

        if not isinstance(lease, GraphSnapshotLease):
            raise TypeError("retain requires a GraphSnapshotLease")
        if lease._owner is not self:
            raise GraphSnapshotError("snapshot lease belongs to another registry")
        with lease._release_lock:
            if lease._released or lease._graph is None:
                raise GraphSnapshotError("cannot retain a released snapshot lease")
            graph = lease._graph
            snapshot = lease.snapshot
            with self._lock:
                record = self._records.get(int(snapshot.graph_id))
                if record is None or not record.refers_to(graph):
                    raise GraphSnapshotError("snapshot lease refers to an unknown graph")
                entry = record.entries.get(int(snapshot.generation))
                if entry is None or entry.snapshot is not snapshot:
                    raise GraphSnapshotError("pinned graph generation is no longer retained")
                entry.lease_count += 1
                return GraphSnapshotLease(self, graph, snapshot)

    def records(self, graph: Optional[Any] = None) -> Tuple[SnapshotRecord, ...]:
        """Return a stable lifecycle snapshot ordered by graph and generation."""

        with self._lock:
            if graph is None:
                records = tuple(self._records.values())
            else:
                record = self._records.get(id(graph))
                records = (
                    (record,)
                    if record is not None and record.refers_to(graph)
                    else ()
                )
            result = []
            for record in records:
                for generation, entry in record.entries.items():
                    result.append(
                        SnapshotRecord(
                            graph_id=entry.snapshot.graph_id,
                            generation=generation,
                            lease_count=entry.lease_count,
                            stale=entry.stale,
                            native_payload_ready=entry.snapshot.native_payload_ready,
                        )
                    )
        return tuple(sorted(result, key=lambda item: (repr(item.graph_id), item.generation)))

    def retire(self, graph: Any) -> None:
        """Forget unleased state for a graph-like object without weak references."""

        cleanup = []
        with self._lock:
            graph_id = id(graph)
            record = self._records.get(graph_id)
            if record is None or not record.refers_to(graph):
                return
            if any(entry.lease_count for entry in record.entries.values()):
                raise GraphSnapshotError("cannot retire a graph with active snapshot leases")
            self._records.pop(graph_id, None)
            cleanup.extend(self._reclaim_entries(record, tuple(record.entries)))
        self._run_cleanup(cleanup)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            entries = [
                entry
                for record in self._records.values()
                for entry in record.entries.values()
            ]
            return {
                "graphs": len(self._records),
                "snapshots": len(entries),
                "active_leases": sum(entry.lease_count for entry in entries),
                "stale_snapshots": sum(bool(entry.stale) for entry in entries),
                "native_snapshots": sum(
                    bool(entry.snapshot.native_payload_ready) for entry in entries
                ),
            }

    def _record_for(self, graph: Any) -> _GraphRecord:
        graph_id = id(graph)
        with self._lock:
            record = self._records.get(graph_id)
            if record is not None:
                if not record.refers_to(graph):
                    raise GraphSnapshotError("graph identity collision in snapshot registry")
                return record

            def collected(reference: weakref.ref) -> None:
                self._graph_collected(graph_id, reference)

            record = _GraphRecord(graph, collected)
            self._records[graph_id] = record
            return record

    def _graph_collected(self, graph_id: int, reference: weakref.ref) -> None:
        cleanup = []
        with self._lock:
            record = self._records.get(graph_id)
            if record is None or record.weak_graph is not reference:
                return
            if any(entry.lease_count for entry in record.entries.values()):
                # A lease holds a strong graph reference, so this branch is a
                # defensive invariant check rather than an expected path.
                return
            self._records.pop(graph_id, None)
            cleanup.extend(self._reclaim_entries(record, tuple(record.entries)))
        self._run_cleanup(cleanup)

    def _acquire_existing(
        self,
        graph: Any,
        record: _GraphRecord,
        observation: SnapshotObservation,
    ) -> Optional[GraphSnapshotLease]:
        with self._lock:
            entry = record.entries.get(observation.generation)
            if entry is None:
                return None
            self._validate_same_generation(entry.snapshot, observation)
            if entry.stale:
                raise StaleGraphGenerationError(
                    "generation {} is already stale".format(observation.generation)
                )
            entry.lease_count += 1
            return GraphSnapshotLease(self, graph, entry.snapshot)

    def _install_and_acquire(
        self,
        graph: Any,
        record: _GraphRecord,
        snapshot: GraphSnapshot,
        native: Optional[NativeSnapshotPayload],
    ) -> GraphSnapshotLease:
        cleanup = []
        with self._lock:
            current = record.current_generation
            if current is not None and snapshot.generation < current:
                cleanup.extend(self._cleanup_for_uninstalled(native))
                failure = StaleGraphGenerationError(
                    "generation {} is older than observed generation {}".format(
                        snapshot.generation, current
                    )
                )
            else:
                failure = None

            if failure is None and snapshot.generation in record.entries:
                entry = record.entries[snapshot.generation]
                self._validate_same_generation(entry.snapshot, snapshot)
                cleanup.extend(self._cleanup_for_uninstalled(native))
            elif failure is None:
                entry = _SnapshotEntry(
                    snapshot=snapshot,
                    lease_count=0,
                    stale=False,
                    cleanup=None if native is None else native.release,
                )
                record.entries[snapshot.generation] = entry

            if failure is None and (
                current is None or snapshot.generation > current
            ):
                record.current_generation = snapshot.generation
                obsolete = []
                for generation, old_entry in record.entries.items():
                    if generation < snapshot.generation:
                        old_entry.stale = True
                        if old_entry.lease_count == 0:
                            obsolete.append(generation)
                cleanup.extend(self._reclaim_entries(record, tuple(obsolete)))

            if failure is None:
                entry.lease_count += 1
                lease = GraphSnapshotLease(self, graph, entry.snapshot)
            else:
                lease = None
        self._run_cleanup(cleanup)
        if failure is not None:
            raise failure
        return lease

    def _release(self, graph: Any, graph_id: int, generation: int) -> None:
        cleanup = []
        with self._lock:
            record = self._records.get(int(graph_id))
            if record is None or not record.refers_to(graph):
                raise GraphSnapshotError("snapshot lease refers to an unknown graph")
            entry = record.entries.get(int(generation))
            if entry is None or entry.lease_count <= 0:
                raise GraphSnapshotError("snapshot lease was released without ownership")
            entry.lease_count -= 1
            if entry.lease_count == 0 and entry.stale:
                cleanup.extend(self._reclaim_entries(record, (generation,)))
        self._run_cleanup(cleanup)

    def _execution_lock(self, lease: GraphSnapshotLease) -> threading.Lock:
        graph = lease._graph
        with self._lock:
            record = self._records.get(int(lease.snapshot.graph_id))
            if record is None or not record.refers_to(graph):
                raise GraphSnapshotError("snapshot lease refers to an unknown graph")
            return record.execution_lock

    def _assert_generation_current(self, lease: GraphSnapshotLease) -> None:
        if lease.concurrent_execution_safe:
            return
        graph = lease._graph
        current = self._read_generation(graph)
        if current != lease.snapshot.generation:
            raise GraphGenerationChangedError(
                "graph generation changed from {} at submission to {} before "
                "request completion".format(lease.snapshot.generation, current)
            )

    def _observe(self, graph: Any) -> SnapshotObservation:
        generation, tracked, immutable = self._generation_info(graph)
        directed = self._directed(graph)
        vertices = self._vertex_count(graph)
        edges = self._edge_count(graph)
        if hasattr(graph, "num_entries"):
            adjacency_entries = int(graph.num_entries)
            adjacency_exact = True
        else:
            adjacency_entries = edges if directed else 2 * edges
            adjacency_exact = directed
        return SnapshotObservation(
            graph_id=id(graph),
            generation=generation,
            directed=directed,
            vertex_count=vertices,
            edge_count=edges,
            adjacency_entries=adjacency_entries,
            adjacency_entries_exact=adjacency_exact,
            generation_tracked=tracked,
            source_is_immutable=immutable,
        )

    def _make_snapshot(
        self,
        observation: SnapshotObservation,
        native: Optional[NativeSnapshotPayload],
    ) -> GraphSnapshot:
        default_signature = (
            "graph={};generation={};directed={};vertices={};edges={}".format(
                observation.graph_id,
                observation.generation,
                int(observation.directed),
                observation.vertex_count,
                observation.edge_count,
            )
        )
        ready = bool(native is not None and native.immutable)
        return GraphSnapshot(
            graph_id=observation.graph_id,
            generation=observation.generation,
            vertex_count=observation.vertex_count,
            adjacency_entries=observation.adjacency_entries,
            topology_signature=(
                default_signature
                if native is None or native.topology_signature is None
                else native.topology_signature
            ),
            weight_signature=None if native is None else native.weight_signature,
            directed=observation.directed,
            edge_count=observation.edge_count,
            adjacency_entries_exact=observation.adjacency_entries_exact,
            generation_tracked=observation.generation_tracked,
            source_is_immutable=observation.source_is_immutable,
            native_payload_ready=ready,
            native_payload=None if not ready else native.payload,
        )

    def _validate_same_generation(
        self,
        snapshot: GraphSnapshot,
        observed: Any,
    ) -> None:
        fields = (
            "graph_id",
            "generation",
            "directed",
            "vertex_count",
            "edge_count",
            "adjacency_entries",
        )
        conflicts = [
            name
            for name in fields
            if getattr(snapshot, name) != getattr(observed, name)
        ]
        if conflicts:
            raise SnapshotConsistencyError(
                "graph metadata changed without advancing generation: {}".format(
                    ", ".join(conflicts)
                )
            )

    def _generation_info(self, graph: Any) -> Tuple[int, bool, bool]:
        bulk = bool(getattr(graph, "_eggpu_bulk_csr", False))
        raw = getattr(graph, "_mutation_generation", None)
        tracked = raw is not None
        if raw is None and bulk:
            metadata = getattr(graph, "metadata", None)
            if isinstance(metadata, dict):
                raw = metadata.get("generation", 0)
                tracked = True
            else:
                signature = getattr(graph, "_eggpu_bulk_signature", ())
                raw = signature[1] if len(signature) > 1 else 0
                tracked = True
        generation = 0 if raw is None else int(raw)
        if generation < 0:
            raise SnapshotCaptureError("graph generation must be non-negative")
        return generation, tracked, bulk

    def _read_generation(self, graph: Any) -> int:
        return self._generation_info(graph)[0]

    @staticmethod
    def _directed(graph: Any) -> bool:
        value = getattr(graph, "is_directed", False)
        return bool(value() if callable(value) else value)

    @staticmethod
    def _vertex_count(graph: Any) -> int:
        method = getattr(graph, "number_of_nodes", None)
        if callable(method):
            return int(method())
        if hasattr(graph, "num_nodes"):
            return int(graph.num_nodes)
        return int(len(graph))

    @staticmethod
    def _edge_count(graph: Any) -> int:
        method = getattr(graph, "number_of_edges", None)
        if callable(method):
            return int(method())
        if hasattr(graph, "num_edges"):
            return int(graph.num_edges)
        edges = getattr(graph, "edges", None)
        if edges is None:
            raise SnapshotCaptureError("graph does not expose an edge count")
        return int(len(edges))

    @staticmethod
    def _default_snapshot_provider(
        graph: Any,
        observation: SnapshotObservation,
    ) -> Optional[NativeSnapshotPayload]:
        if observation.source_is_immutable:
            payload = getattr(graph, "_cpp_graph", None)
            if payload is None:
                return None
            signature = getattr(graph, "_eggpu_bulk_signature", None)
            return NativeSnapshotPayload(
                payload=payload,
                immutable=True,
                topology_signature=(
                    None if signature is None else "bulk-csr:{}".format(repr(signature))
                ),
            )

        # The production GPU extension supplies a generation-local native
        # snapshot builder.  CPU-only imports and older extensions retain the
        # metadata-only behavior instead of pretending a mutable graph was
        # frozen.  If the binding exists but capture fails, propagate that
        # failure: silently falling back would violate snapshot consistency.
        try:
            import cpp_easygraph
        except ImportError:
            return None
        if not (
            hasattr(graph, "_node")
            and hasattr(graph, "_adj")
            and hasattr(graph, "_mutation_generation")
        ):
            return None
        freeze = getattr(cpp_easygraph, "cpp_egmgpu_freeze_graph_snapshot", None)
        if not callable(freeze):
            return None
        public_nodes = tuple(graph)
        native = freeze(graph)
        if len(public_nodes) != observation.vertex_count:
            raise SnapshotCaptureError(
                "frozen public node order does not match the observed graph size"
            )
        return NativeSnapshotPayload(
            payload=FrozenNativeGraph(
                native_graph=native,
                public_nodes=public_nodes,
                public_index={node: index for index, node in enumerate(public_nodes)},
                directed=observation.directed,
            ),
            immutable=True,
            topology_signature=(
                "native-snapshot:graph={};generation={};directed={};vertices={};edges={}".format(
                    observation.graph_id,
                    observation.generation,
                    int(observation.directed),
                    observation.vertex_count,
                    observation.edge_count,
                )
            ),
            weight_signature="generation={}".format(observation.generation),
        )

    @staticmethod
    def _cleanup_for_uninstalled(
        native: Optional[NativeSnapshotPayload],
    ) -> Tuple[Tuple[Callable[[Any], None], Any], ...]:
        if native is None or native.release is None:
            return ()
        return ((native.release, native.payload),)

    def _release_uninstalled(
        self,
        native: Optional[NativeSnapshotPayload],
    ) -> None:
        self._run_cleanup(self._cleanup_for_uninstalled(native))

    @staticmethod
    def _reclaim_entries(
        record: _GraphRecord,
        generations: Tuple[int, ...],
    ) -> Tuple[Tuple[Callable[[Any], None], Any], ...]:
        cleanup = []
        for generation in generations:
            entry = record.entries.pop(generation, None)
            if entry is not None and entry.cleanup is not None:
                cleanup.append((entry.cleanup, entry.snapshot.native_payload))
        return tuple(cleanup)

    @staticmethod
    def _run_cleanup(
        callbacks: Tuple[Tuple[Callable[[Any], None], Any], ...],
    ) -> None:
        first_error = None
        for callback, payload in callbacks:
            try:
                callback(payload)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise GraphSnapshotError(
                "native snapshot cleanup failed: {}".format(first_error)
            ) from first_error
