"""Pre-admission compatibility checks for production EGMGPU routes.

Native bindings intentionally keep their own defensive validation.  This
module adds an earlier, side-effect-free check used by the scheduler before a
plan is selected or a device reservation is queued.  It therefore prevents an
AUTO request from selecting a route that is known to reject the submitted
graph view or public-function arguments.

The checks preserve EasyGraph's public semantics.  In particular, routing a
decorated public function through EGMGPU must not bypass the directed-graph,
multigraph, weight, target, or algorithm restrictions of that function.
"""

from __future__ import annotations

import inspect
import math

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

from .types import ExecutionMode


@dataclass(frozen=True)
class CompatibilityResult:
    """Result of one pure executor/graph/argument compatibility check."""

    compatible: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if self.compatible and self.reason:
            raise ValueError("a compatible result cannot contain a failure reason")
        if not self.compatible and not str(self.reason).strip():
            raise ValueError("an incompatible result requires a reason")

    @classmethod
    def accept(cls) -> "CompatibilityResult":
        return cls(True, "")

    @classmethod
    def reject(cls, reason: str) -> "CompatibilityResult":
        return cls(False, str(reason).strip())


class ExecutorCompatibilityError(RuntimeError):
    """An explicit execution mode cannot preserve the submitted semantics."""


CompatibilityPredicate = Callable[
    [Any, Tuple[Any, ...], Mapping[str, Any], ExecutionMode], CompatibilityResult
]


_PRODUCTION_ATTESTATION = object()


class _ProductionExecutor:
    """Internal envelope for a route admitted by the strict bootstrap.

    The process-local token is deliberately unavailable through the public
    package namespace.  Ordinary ``register_native_executor`` callables remain
    valid explicit research routes, but cannot acquire production/AUTO status
    merely by setting an ``AVAILABLE`` string or a similarly named attribute.
    """

    __slots__ = (
        "_attestation",
        "_executor",
        "_predicate",
        "evidence_identity",
    )

    def __init__(
        self,
        executor: Callable[..., Any],
        predicate: CompatibilityPredicate,
        evidence_identity: str,
        token: object,
    ) -> None:
        if token is not _PRODUCTION_ATTESTATION:
            raise TypeError("production executors are created by bootstrap_production_bindings")
        if not callable(executor) or not callable(predicate):
            raise TypeError("production executor and compatibility predicate must be callable")
        identity = str(evidence_identity).strip()
        if not identity:
            raise ValueError("production executor requires an evidence identity")
        self._attestation = token
        self._executor = executor
        self._predicate = predicate
        self.evidence_identity = identity

    @property
    def eligible_device_groups(self):
        return getattr(self._executor, "eligible_device_groups", ())

    @property
    def spec(self):
        return getattr(self._executor, "spec", None)

    @property
    def evidence(self):
        return getattr(self._executor, "evidence", None)

    def compatibility(
        self,
        snapshot: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
        mode: ExecutionMode,
    ) -> CompatibilityResult:
        result = self._predicate(snapshot, args, kwargs, ExecutionMode.parse(mode))
        if not isinstance(result, CompatibilityResult):
            raise TypeError("executor compatibility predicate returned an invalid result")
        return result

    def __call__(self, context, args, kwargs):
        return self._executor(context, args, kwargs)


def _attest_production_executor(
    executor: Callable[..., Any],
    predicate: CompatibilityPredicate,
    evidence_identity: str,
) -> Callable[..., Any]:
    """Bootstrap-only constructor; intentionally omitted from ``__all__``."""

    return _ProductionExecutor(
        executor,
        predicate,
        evidence_identity,
        _PRODUCTION_ATTESTATION,
    )


def _unwrap(executor: Any) -> Any:
    current = executor
    seen = set()
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _ProductionExecutor):
            return current
        current = getattr(current, "_egmgpu_original_executor", None)
    return None


def is_production_executor(executor: Any) -> bool:
    """Return whether ``executor`` carries an in-process bootstrap attestation."""

    wrapped = _unwrap(executor)
    return bool(
        wrapped is not None
        and wrapped._attestation is _PRODUCTION_ATTESTATION
    )


def evaluate_executor_compatibility(
    executor: Any,
    snapshot: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    mode: ExecutionMode,
) -> CompatibilityResult:
    """Evaluate an executor predicate, defaulting research routes to explicit-only.

    Research executors do not need a predicate.  They remain compatible for an
    explicit call and are excluded from AUTO by ``is_production_executor``.
    """

    wrapped = _unwrap(executor)
    if wrapped is None:
        return CompatibilityResult.accept()
    return wrapped.compatibility(
        snapshot, tuple(args), dict(kwargs), ExecutionMode.parse(mode)
    )


def production_compatibility_predicate(
    function_name: str,
    mode: ExecutionMode,
    public_function: Callable[..., Any],
) -> CompatibilityPredicate:
    """Build the standard predicate for one evidence-backed public route."""

    name = str(function_name).strip().lower()
    parsed_mode = ExecutionMode.parse(mode)
    if parsed_mode is ExecutionMode.AUTO:
        raise ValueError("AUTO is a selection policy, not an executor mode")
    if not callable(public_function):
        raise TypeError("public_function must be callable")
    signature = inspect.signature(public_function)

    def predicate(snapshot, args, kwargs, observed_mode):
        if ExecutionMode.parse(observed_mode) is not parsed_mode:
            return CompatibilityResult.reject(
                "executor mode does not match the selected execution mode"
            )
        if snapshot is None or not bool(
            getattr(snapshot, "native_payload_ready", False)
        ):
            return CompatibilityResult.reject(
                "the route requires a frozen native graph snapshot"
            )
        try:
            bound = signature.bind(*tuple(args), **dict(kwargs))
        except TypeError as exc:
            return CompatibilityResult.reject(
                "arguments do not match the public function: {}".format(exc)
            )
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        graph = arguments.get(next(iter(signature.parameters), ""))
        if graph is None:
            return CompatibilityResult.reject("the public graph argument is missing")

        graph_kind = _graph_kind(graph, snapshot)
        if parsed_mode is ExecutionMode.PARTITIONED:
            if graph_kind != "partition_manifest":
                return CompatibilityResult.reject(
                    "partitioned execution requires a loaded multi-partition manifest"
                )
        elif parsed_mode is ExecutionMode.REPLICATED:
            if graph_kind != "immutable_bulk":
                return CompatibilityResult.reject(
                    "replicated execution requires an immutable bulk CSR graph"
                )
        elif graph_kind == "partition_manifest":
            return CompatibilityResult.reject(
                "single/request execution cannot consume a multi-partition manifest"
            )
        elif graph_kind not in {"immutable_bulk", "mutable_snapshot"}:
            return CompatibilityResult.reject("unsupported native graph snapshot kind")

        directed = bool(getattr(snapshot, "directed", False))
        multigraph = _is_multigraph(graph)
        if name == "scc" and not directed:
            return CompatibilityResult.reject(
                "strongly_connected_components requires a directed graph"
            )
        if name == "wcc" and directed:
            return CompatibilityResult.reject(
                "connected_components requires an undirected graph"
            )
        if name == "mst":
            if directed:
                return CompatibilityResult.reject(
                    "minimum_spanning_tree requires an undirected graph"
                )
            if multigraph:
                return CompatibilityResult.reject(
                    "minimum_spanning_tree does not support multigraphs"
                )
            if bool(arguments.get("ignore_nan", False)):
                return CompatibilityResult.reject(
                    "native minimum_spanning_tree does not implement ignore_nan=True"
                )
            algorithm = arguments.get("algorithm", "kruskal")
            if not isinstance(algorithm, str) or algorithm not in {
                "kruskal",
                "prim",
                "boruvka",
            }:
                return CompatibilityResult.reject(
                    "unknown minimum_spanning_tree algorithm {!r}".format(algorithm)
                )

        if name in {
            "lcc",
            "bc",
            "closeness",
            "effective_size",
            "efficiency",
            "constraint",
            "hierarchy",
        } and multigraph:
            return CompatibilityResult.reject(
                "{} does not support multigraphs".format(name)
            )
        if name in {"bfs", "dijkstra", "bellman_ford", "sssp"} and multigraph:
            return CompatibilityResult.reject(
                "{} path semantics do not support multigraphs".format(name)
            )

        if arguments.get("n_workers") is not None:
            return CompatibilityResult.reject(
                "native execution does not accept n_workers"
            )
        weight = arguments.get("weight")
        if weight is not None and not isinstance(weight, str):
            return CompatibilityResult.reject(
                "native weight parameters must be an attribute name or None"
            )
        if name == "lcc":
            if weight is not None:
                return CompatibilityResult.reject(
                    "the current native clustering path supports weight=None only"
                )
            if directed and not _has_undirected_projection(graph):
                return CompatibilityResult.reject(
                    "directed clustering requires a validated undirected projection"
                )
        if name in {"bellman_ford", "sssp", "bc", "closeness"}:
            sources = arguments.get("sources")
            if sources is not None and (
                not isinstance(sources, Sequence)
                or isinstance(sources, (str, bytes))
            ):
                return CompatibilityResult.reject(
                    "native sources must be a reusable finite sequence"
                )

        numeric = _validate_numeric_parameters(name, arguments)
        if numeric is not None:
            return CompatibilityResult.reject(numeric)

        if parsed_mode is ExecutionMode.PARTITIONED:
            if name == "pagerank":
                alpha = float(arguments.get("alpha", 0.85))
                if not 0.0 < alpha < 1.0:
                    return CompatibilityResult.reject(
                        "partitioned PageRank alpha must lie in (0, 1)"
                    )
            partitioned = _validate_partitioned_arguments(name, graph, arguments)
            if partitioned is not None:
                return CompatibilityResult.reject(partitioned)
        return CompatibilityResult.accept()

    return predicate


def _graph_kind(graph: Any, snapshot: Any) -> str:
    # Import lazily to avoid a binding-module cycle during package import.
    try:
        from .partitioned_bindings import ImmutableMultiPartitionGraph

        if isinstance(graph, ImmutableMultiPartitionGraph):
            return "partition_manifest"
    except ImportError:  # pragma: no cover - dependency-free package probes.
        pass
    if bool(getattr(snapshot, "source_is_immutable", False)) and bool(
        getattr(graph, "_eggpu_bulk_csr", False)
    ):
        return "immutable_bulk"
    if not bool(getattr(snapshot, "source_is_immutable", False)):
        return "mutable_snapshot"
    return "unknown"


def _is_multigraph(graph: Any) -> bool:
    value = getattr(graph, "is_multigraph", False)
    return bool(value() if callable(value) else value)


def _has_undirected_projection(graph: Any) -> bool:
    if getattr(graph, "undirected_projection_path", None) is not None:
        return True
    if getattr(graph, "undirected_projection", None) is not None:
        return True
    metadata = getattr(graph, "metadata", None)
    return bool(
        isinstance(metadata, Mapping)
        and metadata.get("undirected_projection_manifest")
    )


def _validate_numeric_parameters(name: str, arguments: Mapping[str, Any]) -> Optional[str]:
    try:
        if name == "pagerank":
            alpha = float(arguments.get("alpha", 0.85))
            maximum = arguments.get("max_iter")
            tolerance = arguments.get("tol")
            if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                return "PageRank alpha must be finite and lie in [0, 1]"
            if maximum is not None and int(maximum) < 1:
                return "PageRank max_iter must be positive"
            if tolerance is not None and (
                not math.isfinite(float(tolerance)) or float(tolerance) < 0.0
            ):
                return "PageRank tol must be finite and non-negative"
    except (TypeError, ValueError, OverflowError):
        return "{} contains an invalid numeric parameter".format(name)
    return None


def _validate_partitioned_arguments(
    name: str, graph: Any, arguments: Mapping[str, Any]
) -> Optional[str]:
    native = getattr(graph, "_cpp_graph", None)
    if name == "pagerank":
        expected = (
            str(getattr(native, "weight_key", ""))
            if bool(getattr(native, "weighted", False))
            else None
        )
        if arguments.get("weight") != expected:
            return "PageRank weight does not match the loaded partition view"
    elif name == "wcc":
        if bool(getattr(native, "directed", False)):
            return "partitioned WCC requires an undirected manifest view"
    elif name == "bfs":
        if arguments.get("target") is not None:
            return "partitioned BFS requires the complete distance result"
        if arguments.get("source") is None:
            return "partitioned BFS requires one source"
    elif name == "sssp":
        if arguments.get("target") is not None:
            return "partitioned SSSP requires the complete distance result"
        expected = str(getattr(native, "weight_key", ""))
        if arguments.get("weight", "weight") != expected:
            return "SSSP weight does not match the loaded partition view"
        if "source" in arguments:
            if arguments.get("source") is None:
                return "partitioned SSSP requires one source"
        else:
            sources = arguments.get("sources")
            if (
                not isinstance(sources, Sequence)
                or isinstance(sources, (str, bytes))
            ):
                return "partitioned SSSP requires exactly one source"
            if len(sources) != 1:
                return "partitioned SSSP currently supports exactly one source"
    return None


__all__ = [
    "CompatibilityPredicate",
    "CompatibilityResult",
    "ExecutorCompatibilityError",
    "evaluate_executor_compatibility",
    "is_production_executor",
    "production_compatibility_predicate",
]
