"""Version-bound atomic requests over the shared state-retention contract.

An operator request captures semantic input identity. Physical residency remains
the executor's responsibility. This prevents requests prepared before a mutable
graph update from silently reading the new version; native immutable snapshots
use the same check. Backend numerical contracts remain attached to each result.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any
import hashlib
import json
import operator
import sys


def _require_gpu_python():
    if sys.version_info < (3, 10):
        raise RuntimeError("The optional GPU session backend requires Python >= 3.10")


class StaleRequestError(RuntimeError):
    """The requested semantic snapshot changed before execution."""


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    reads: tuple[str, ...]
    private_state: tuple[str, ...]
    completion: str = "complete owning host output before return"


OPERATORS = {
    "bfs": OperatorSpec("bfs", ("topology", "outgoing"), ("frontier", "distance")),
    "sssp": OperatorSpec("sssp", ("topology", "outgoing", "weights"), ("frontier", "distance")),
    "pr": OperatorSpec("pr", ("topology", "incoming", "out_degree"), ("rank", "residual")),
    "weighted_pr": OperatorSpec("weighted_pr", ("topology", "incoming", "weights", "out_strength"), ("rank", "residual")),
    "wcc": OperatorSpec("wcc", ("topology", "undirected"), ("labels",)),
}


def _identity(graph):
    """Authoritative semantic identity, independent of lazy materialization.

    Arena journal commits advance logical_revision before dependent device
    arrays are refreshed. Native input is an immutable registered manifest or
    an internally constructed memory snapshot with a real content signature.
    Observational provenance and materialized generations are deliberately
    excluded: preparing a valid representation must preserve request identity.
    """
    value = {"graph_id": graph.graph_id, "scope": graph.identity_scope,
             "backend": graph.backend, "nodes": graph.node_domain.to_dict(),
             "directed": graph.directed, "weighted": graph.weighted}
    if graph.backend == "managed":
        value["version"] = [graph.version.ref_generation,
                            graph.version.logical_revision]
    elif graph.backend == "native":
        value["version"] = graph.version.ref_generation
        if graph.provenance.get("snapshot_storage") == "native_memory":
            snapshot = graph.provenance.get("snapshot_document_sha256")
            if not isinstance(snapshot, str) or len(snapshot) != 64 or any(c not in "0123456789abcdef" for c in snapshot):
                raise ValueError("Native memory snapshot requires its real document SHA-256")
            value["native_memory_snapshot"] = snapshot
        else:
            value["manifest"] = graph.provenance["manifest_document_sha256"]
    else:
        raise ValueError("Unsupported semantic graph descriptor")
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class OperatorRequest:
    session_id: str
    graph_id: str
    snapshot_identity: str
    spec: OperatorSpec
    parameters: tuple[tuple[str, Any], ...]


class Session:
    """One owner for explicit graph operations and their lifetime checks.

Public factories never choose a different backend on failure. Native uses
partition manifests and per-device retained-state budgets. Arena uses owned
COO inputs and a single backing-allocation budget. The distinct budget meanings
are exposed by ``capabilities`` and ``inventory``.
    """

    def __init__(self, backend, *, kind):
        self._backend = backend
        self.kind = kind
        self._lock = RLock()
        self.closed = False
        self.session_id = backend.session_id
        self.dispatch_plan = None

    @classmethod
    def create(cls, inputs, *, devices=(0,), state_model="static",
               algorithms=("bfs", "sssp", "wcc", "pr"), budget_bytes=None, **options):
        """Create the route selected by a fixed workload capability policy.

        Use ``plan_session`` to inspect the decision without importing CUDA.
        A single-device static/journal session takes a mapping of COO inputs
        (nodes, edges, weights) and requires ``budget_bytes`` for its backing
        arena. Snapshot sessions and multi-device static sessions take partition
        manifests; their optional per-device budget uses ``budgets`` in options.
        These budget meanings are intentionally not silently converted.

        The selected backend is never changed after an error. This factory does
        not convert inputs or choose a different implementation for each graph.
        Fixed-backend scaling should use ``Session.native`` for every GPU count.
        """
        from ._dispatch import plan_session
        plan = plan_session(devices=devices, state_model=state_model, algorithms=algorithms)
        if plan.backend == "arena":
            if budget_bytes is None:
                raise ValueError("The single-device arena requires explicit budget_bytes")
            result = cls.arena(inputs, budget_bytes=budget_bytes,
                               device=plan.devices[0], **options)
        else:
            if budget_bytes is not None:
                raise ValueError("Native retained-state budgets use per-device budgets, not budget_bytes")
            result = cls.native(inputs, devices=plan.devices, **options)
        result.dispatch_plan = plan
        return result

    @classmethod
    def native(cls, manifests, *, devices=(0,), cpp=None, **options):
        _require_gpu_python()
        from ._dynamic import DynamicNativeSession
        if cpp is None:
            import cpp_easygraph as cpp
        return cls(DynamicNativeSession(cpp, manifests, devices=devices, **options),
                   kind="native")

    @classmethod
    def arena(cls, graphs, *, budget_bytes, device=0,
              pr_tolerance_l1=1e-6, pr_max_iter=10000):
        """Create one bounded arena with an explicit PageRank stopping contract.

        ``pr_tolerance_l1`` bounds the sum of consecutive-iterate differences,
        not a per-vertex tolerance or an independently certified final residual.
        Defaults preserve the historical arena behavior.
        """
        _require_gpu_python()
        if isinstance(device, bool):
            raise ValueError("device must be a nonnegative integer")
        device = operator.index(device)
        if device < 0:
            raise ValueError("device must be a nonnegative integer")
        import cupy as cp
        from ._core import CoreManagedSession
        with cp.cuda.Device(device):
            backend = CoreManagedSession(graphs, budget_bytes=budget_bytes,
                                     pr_tolerance_l1=pr_tolerance_l1,
                                     pr_max_iter=pr_max_iter)
        result = cls(backend, kind="arena")
        result._device = device
        return result

    def _open(self):
        if self.closed:
            raise RuntimeError("Session closed")

    def _device_context(self):
        if self.kind == "arena":
            import cupy as cp
            return cp.cuda.Device(self._device)
        from contextlib import nullcontext
        return nullcontext()

    def request(self, graph_id, algorithm, parameters=None):
        with self._lock:
            self._open()
            caps = self._backend.capability_descriptor()
            if algorithm not in caps.algorithms or algorithm not in OPERATORS:
                raise NotImplementedError(f"{self.kind} does not support {algorithm}")
            params = dict(parameters or {})
            if algorithm in ("bfs", "sssp"):
                from ._bridge import source_parameter
                params = {"source": source_parameter(params)}
            elif params:
                raise ValueError("This backend exposes its fixed PR/WCC contract")
            return OperatorRequest(self.session_id, graph_id,
                                   _identity(self._backend.describe(graph_id)),
                                   OPERATORS[algorithm], tuple(sorted(params.items())))

    def execute(self, request):
        with self._lock, self._device_context():
            self._open()
            if not isinstance(request, OperatorRequest):
                raise TypeError("execute expects an OperatorRequest")
            if request.session_id != self.session_id:
                raise ValueError("Request belongs to another session")
            if OPERATORS.get(request.spec.name) != request.spec:
                raise ValueError("Unknown operator dependency contract")
            if _identity(self._backend.describe(request.graph_id)) != request.snapshot_identity:
                raise StaleRequestError("Graph changed after request creation; create a new request")
            result = self._backend.query(request.graph_id, request.spec.name,
                                         dict(request.parameters))
            if result.graph is None or _identity(result.graph) != request.snapshot_identity:
                raise RuntimeError("Executor returned a different graph version")
            if result.values.flags.writeable or not result.values.flags.owndata:
                raise RuntimeError("Executor must return independent immutable host values")
            return result

    def query(self, graph_id, algorithm, parameters=None):
        with self._lock:
            return self.execute(self.request(graph_id, algorithm, parameters))

    def update(self, graph_id, batch):
        with self._lock, self._device_context():
            self._open()
            if self.kind != "arena":
                raise NotImplementedError("Native snapshots use commit_weights/register_snapshot")
            return self._backend.update(graph_id, batch)

    def commit_weights(self, base_name, new_name, updates, directory=None, *, storage='files'):
        with self._lock:
            self._open()
            if self.kind != "native":
                raise NotImplementedError("Arena graphs use update")
            return self._backend.commit_weights(base_name, new_name, updates, directory, storage=storage)

    def register_snapshot(self, name, manifest):
        with self._lock:
            self._open()
            if self.kind != "native":
                raise NotImplementedError("Native manifest registration requires native backend")
            return self._backend.register_snapshot(name, manifest)

    def retain_snapshot(self, name):
        with self._lock:
            self._open()
            if self.kind != "native":
                raise NotImplementedError("Explicit native view retention requires native backend")
            return self._backend.retain_snapshot(name)

    def release_snapshot(self, name):
        with self._lock:
            self._open()
            if self.kind != "native":
                raise NotImplementedError("Explicit native view retention requires native backend")
            return self._backend.release_snapshot(name)

    def capabilities(self):
        with self._lock:
            self._open()
            return self._backend.capability_descriptor()

    def inventory(self):
        with self._lock:
            self._open()
            return (self._backend.resource_inventory() if self.kind == "native"
                    else self._backend.snapshot())

    def close(self):
        with self._lock, self._device_context():
            if not self.closed:
                self._backend.close()
                self.closed = True

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()
