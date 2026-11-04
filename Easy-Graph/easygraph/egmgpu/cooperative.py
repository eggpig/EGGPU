"""Function-family execution descriptions for replicated multi-GPU calls.

The adapters in this module split already-supported ``sources`` or ``nodes``
parameters and describe deterministic result merging.  Their Python shard
callables are executable only by the explicit ``CPUTestRuntime``.  Production
cooperative paths use one synchronous pybind adapter and submit native-only
tasks to the C++ ``RuntimeManager``; they never send these Python callbacks to
device workers.

An adapter being ``WIREABLE`` means only that the audited EGGPU Python/backend
entry point accepts the required subset parameter.  It is not an availability
or performance claim.  Runtime installation requires explicit successful GPU
probe evidence, which this source tree does not fabricate.
"""

from __future__ import annotations

import inspect

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple

from .types import CooperativeContext
from .types import CooperativePlan
from .types import ExecutionMode


class CooperativeAdapterError(RuntimeError):
    """Base error for invalid cooperative execution descriptions."""


class AdapterNotAvailableError(CooperativeAdapterError):
    """A wireable adapter was used without successful GPU probe evidence."""


class ShardResultError(CooperativeAdapterError):
    """A shard returned a value incompatible with its merge contract."""


class ShardAxis(str, Enum):
    SOURCES = "sources"
    NODES = "nodes"


class MergeStrategy(str, Enum):
    DENSE_SUM = "dense_sum"
    ORDERED_SEQUENCE = "ordered_sequence"
    ORDERED_MAPPING = "ordered_mapping"


class AdapterReadiness(str, Enum):
    DESCRIBED = "described"
    WIREABLE = "wireable"
    AVAILABLE = "available"


@dataclass(frozen=True)
class FunctionExecutionDescriptor:
    """Audited decomposition and result contract for one graph function."""

    name: str
    function_family: str
    subset_parameter: str
    shard_axis: ShardAxis
    merge_strategy: MergeStrategy
    public_signature: str
    backend_entrypoint: str
    output_contract: str
    backend_subset_supported: bool
    subset_required: bool = False
    semantic_alias_of: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or not self.function_family:
            raise ValueError("descriptor name and function_family must be non-empty")
        if self.subset_parameter != self.shard_axis.value:
            raise ValueError("subset_parameter must match the declared shard axis")
        if not self.public_signature or not self.backend_entrypoint:
            raise ValueError("descriptor signatures must be non-empty")

    @property
    def initial_readiness(self) -> AdapterReadiness:
        return (
            AdapterReadiness.WIREABLE
            if self.backend_subset_supported
            else AdapterReadiness.DESCRIBED
        )


@dataclass(frozen=True)
class ShardAssignment:
    device: int
    ordinal: int
    items: Tuple[Any, ...]


@dataclass(frozen=True)
class ExecutionDescription:
    function: str
    mode: ExecutionMode
    shard_axis: ShardAxis
    merge_strategy: MergeStrategy
    original_items: Tuple[Any, ...]
    assignments: Tuple[ShardAssignment, ...]


@dataclass(frozen=True)
class GPUProbeEvidence:
    """Persisted evidence required before an adapter becomes available."""

    devices: Tuple[int, ...]
    backend: str
    result_path: str
    passed: bool

    def __post_init__(self) -> None:
        canonical = tuple(sorted(set(int(device) for device in self.devices)))
        if len(canonical) < 2:
            raise ValueError("GPU probe evidence requires at least two devices")
        object.__setattr__(self, "devices", canonical)
        if not self.backend or not self.result_path:
            raise ValueError("GPU probe evidence requires backend and raw result path")


@dataclass(frozen=True)
class AdapterStatus:
    descriptor: FunctionExecutionDescriptor
    readiness: AdapterReadiness
    probe: Optional[GPUProbeEvidence] = None


class _EmptyShardResult:
    pass


EMPTY_SHARD_RESULT = _EmptyShardResult()


def _graph_nodes(graph: Any) -> Tuple[Any, ...]:
    nodes = getattr(graph, "nodes", None)
    if nodes is None:
        if hasattr(graph, "num_nodes"):
            return tuple(range(int(graph.num_nodes)))
        return tuple(iter(graph))
    if hasattr(nodes, "keys"):
        return tuple(nodes.keys())
    return tuple(nodes)


def _balanced_contiguous_shards(
    items: Sequence[Any], devices: Sequence[int]
) -> Tuple[ShardAssignment, ...]:
    ordered_items = tuple(items)
    ordered_devices = tuple(int(device) for device in devices)
    if not ordered_devices:
        raise ValueError("at least one device is required for sharding")
    quotient, remainder = divmod(len(ordered_items), len(ordered_devices))
    assignments = []
    begin = 0
    for ordinal, device in enumerate(ordered_devices):
        size = quotient + (1 if ordinal < remainder else 0)
        end = begin + size
        assignments.append(
            ShardAssignment(device=device, ordinal=ordinal, items=ordered_items[begin:end])
        )
        begin = end
    return tuple(assignments)


class ReplicatedSubsetAdapter:
    """Build per-device subset calls and one deterministic host merge."""

    def __init__(self, descriptor: FunctionExecutionDescriptor) -> None:
        if not descriptor.backend_subset_supported:
            raise CooperativeAdapterError(
                "{} has no audited subset-capable backend".format(descriptor.name)
            )
        self.descriptor = descriptor

    def build_plan(
        self,
        context: CooperativeContext,
        function: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> CooperativePlan:
        if context.mode is not ExecutionMode.REPLICATED:
            raise CooperativeAdapterError(
                "{} adapter requires replicated execution".format(
                    self.descriptor.name
                )
            )
        signature = inspect.signature(function)
        try:
            bound = signature.bind_partial(*args, **kwargs)
        except TypeError as exc:
            raise CooperativeAdapterError(
                "arguments do not match {}: {}".format(
                    self.descriptor.public_signature, exc
                )
            ) from exc
        if not args:
            raise CooperativeAdapterError("a graph argument is required")
        graph = args[0]
        subset_name = self.descriptor.subset_parameter
        if subset_name not in signature.parameters:
            raise CooperativeAdapterError(
                "callable signature does not expose audited {!r} parameter".format(
                    subset_name
                )
            )
        subset_value = bound.arguments.get(subset_name)
        if self.descriptor.subset_required and subset_value is None:
            raise CooperativeAdapterError(
                "{} requires an explicit {!r} argument".format(
                    self.descriptor.name, subset_name
                )
            )
        items = _graph_nodes(graph) if subset_value is None else tuple(subset_value)
        if self.descriptor.merge_strategy is MergeStrategy.ORDERED_MAPPING:
            items = tuple(dict.fromkeys(items))
        assignments = _balanced_contiguous_shards(items, context.devices)
        description = ExecutionDescription(
            function=self.descriptor.name,
            mode=context.mode,
            shard_axis=self.descriptor.shard_axis,
            merge_strategy=self.descriptor.merge_strategy,
            original_items=items,
            assignments=assignments,
        )

        def shard_call(assignment: ShardAssignment) -> Callable[[], Any]:
            if not assignment.items:
                return lambda: EMPTY_SHARD_RESULT

            def invoke() -> Any:
                shard_bound = signature.bind_partial(*args, **kwargs)
                shard_bound.arguments[subset_name] = list(assignment.items)
                return function(*shard_bound.args, **shard_bound.kwargs)

            return invoke

        shards = {
            assignment.device: shard_call(assignment)
            for assignment in assignments
        }
        return CooperativePlan(
            shards=shards,
            merge=lambda results: self.merge(description, results),
            metadata=description,
        )

    def merge(
        self,
        description: ExecutionDescription,
        results: Sequence[Any],
    ) -> Any:
        if len(results) != len(description.assignments):
            raise ShardResultError(
                "expected {} shard results, received {}".format(
                    len(description.assignments), len(results)
                )
            )
        if self.descriptor.merge_strategy is MergeStrategy.DENSE_SUM:
            return self._merge_dense_sum(results)
        if self.descriptor.merge_strategy is MergeStrategy.ORDERED_SEQUENCE:
            return self._merge_ordered_sequence(results)
        if self.descriptor.merge_strategy is MergeStrategy.ORDERED_MAPPING:
            return self._merge_ordered_mapping(description, results)
        raise AssertionError("unhandled merge strategy")

    @staticmethod
    def _merge_dense_sum(results: Sequence[Any]) -> list:
        dense_results = [result for result in results if result is not EMPTY_SHARD_RESULT]
        if not dense_results:
            return []
        if any(isinstance(result, Mapping) for result in dense_results):
            raise ShardResultError("dense-sum shards must return ordered dense values")
        try:
            width = len(dense_results[0])
        except TypeError as exc:
            raise ShardResultError("dense-sum shard result has no length") from exc
        if any(len(result) != width for result in dense_results):
            raise ShardResultError("dense-sum shard widths do not match")
        merged = [0.0] * width
        # Runtime supplies results in assigned-device order.  Preserve that
        # order explicitly so floating-point accumulation is reproducible.
        for result in results:
            if result is EMPTY_SHARD_RESULT:
                continue
            for index in range(width):
                merged[index] += float(result[index])
        return merged

    @staticmethod
    def _merge_ordered_sequence(results: Sequence[Any]) -> list:
        merged = []
        for result in results:
            if result is EMPTY_SHARD_RESULT:
                continue
            if isinstance(result, (str, bytes, Mapping)):
                raise ShardResultError("ordered-sequence shard returned a non-sequence")
            try:
                merged.extend(result)
            except TypeError as exc:
                raise ShardResultError(
                    "ordered-sequence shard returned a non-iterable"
                ) from exc
        return merged

    @staticmethod
    def _merge_ordered_mapping(
        description: ExecutionDescription,
        results: Sequence[Any],
    ) -> Dict[Any, Any]:
        by_key = {}
        for result in results:
            if result is EMPTY_SHARD_RESULT:
                continue
            if not isinstance(result, Mapping):
                raise ShardResultError("ordered-mapping shard must return a mapping")
            for key, value in result.items():
                if key in by_key:
                    raise ShardResultError(
                        "multiple shards returned key {!r}".format(key)
                    )
                by_key[key] = value
        missing = [item for item in description.original_items if item not in by_key]
        if missing:
            raise ShardResultError(
                "shards omitted {} requested result keys".format(len(missing))
            )
        return {item: by_key[item] for item in description.original_items}


class CooperativeAdapterRegistry:
    """Capability state machine that never equates wiring with availability."""

    def __init__(self, descriptors: Iterable[FunctionExecutionDescriptor]) -> None:
        self._status = {}
        for descriptor in descriptors:
            if descriptor.name in self._status:
                raise ValueError(
                    "duplicate cooperative descriptor {}".format(descriptor.name)
                )
            self._status[descriptor.name] = AdapterStatus(
                descriptor=descriptor,
                readiness=descriptor.initial_readiness,
            )

    def status(self, name: str) -> AdapterStatus:
        try:
            return self._status[str(name)]
        except KeyError as exc:
            raise KeyError("unknown cooperative function {!r}".format(name)) from exc

    def record_gpu_probe(self, name: str, evidence: GPUProbeEvidence) -> AdapterStatus:
        current = self.status(name)
        if not current.descriptor.backend_subset_supported:
            raise AdapterNotAvailableError(
                "{} has no subset-capable backend to probe".format(name)
            )
        if evidence.passed and not Path(evidence.result_path).is_file():
            raise AdapterNotAvailableError(
                "successful GPU probe raw result does not exist: {}".format(
                    evidence.result_path
                )
            )
        readiness = (
            AdapterReadiness.AVAILABLE
            if evidence.passed
            else AdapterReadiness.WIREABLE
        )
        updated = AdapterStatus(current.descriptor, readiness, evidence)
        self._status[str(name)] = updated
        return updated

    def install_available(
        self,
        runtime: Any,
        functions: Mapping[str, Callable[..., Any]],
    ) -> Tuple[str, ...]:
        pending = []
        for name, function in functions.items():
            status = self.status(name)
            if status.readiness is not AdapterReadiness.AVAILABLE:
                raise AdapterNotAvailableError(
                    "{} is {}, not GPU-probed available".format(
                        name, status.readiness.value
                    )
                )
            pending.append((name, function, ReplicatedSubsetAdapter(status.descriptor)))

        installed = []
        for name, function, adapter in pending:
            runtime.register_cooperative_executor(
                function,
                ExecutionMode.REPLICATED,
                adapter.build_plan,
            )
            installed.append(name)
        return tuple(installed)

    def matrix(self) -> Dict[str, str]:
        return {
            name: status.readiness.value
            for name, status in sorted(self._status.items())
        }


def default_function_descriptors() -> Tuple[FunctionExecutionDescriptor, ...]:
    """Return the ten audited replicated-subset execution descriptions."""

    centrality = (
        FunctionExecutionDescriptor(
            "bc",
            "Centrality",
            "sources",
            ShardAxis.SOURCES,
            MergeStrategy.DENSE_SUM,
            (
                "betweenness_centrality(G, weight=None, sources=None, "
                "normalized=True, endpoints=False)"
            ),
            "gpu_eggpu_backend.betweenness_centrality(..., sources=...)",
            "dense values in graph node order",
            True,
        ),
        FunctionExecutionDescriptor(
            "closeness",
            "Centrality",
            "sources",
            ShardAxis.SOURCES,
            MergeStrategy.ORDERED_SEQUENCE,
            "closeness_centrality(G, weight=None, sources=None)",
            "gpu_eggpu_backend.closeness_centrality(..., sources=...)",
            "dense values in requested source order",
            True,
        ),
    )
    paths = (
        FunctionExecutionDescriptor(
            "bfs_batch",
            "Path & Spanning",
            "sources",
            ShardAxis.SOURCES,
            MergeStrategy.ORDERED_MAPPING,
            "multi_source_bfs(G, sources, target=None)",
            "gpu_eggpu_backend.multi_source_bfs(..., sources=...)",
            "source-to-distance-mapping in requested source order",
            True,
            subset_required=True,
        ),
        FunctionExecutionDescriptor(
            "dijkstra_batch",
            "Path & Spanning",
            "sources",
            ShardAxis.SOURCES,
            MergeStrategy.ORDERED_MAPPING,
            "multi_source_dijkstra(G, sources, weight='weight', target=None)",
            "gpu_eggpu_backend.multi_source_dijkstra(..., sources=...)",
            "source-to-distance-mapping in requested source order",
            True,
            subset_required=True,
        ),
        FunctionExecutionDescriptor(
            "bellman_ford_batch",
            "Path & Spanning",
            "sources",
            ShardAxis.SOURCES,
            MergeStrategy.ORDERED_MAPPING,
            "multi_source_bellman_ford(G, sources, weight='weight', target=None)",
            "gpu_eggpu_backend.multi_source_bellman_ford(..., sources=...)",
            "source-to-distance-mapping in requested source order",
            True,
            subset_required=True,
        ),
        FunctionExecutionDescriptor(
            "sssp",
            "Path & Spanning",
            "sources",
            ShardAxis.SOURCES,
            MergeStrategy.ORDERED_MAPPING,
            "multi_source_dijkstra(G, sources, weight='weight', target=None)",
            "gpu_eggpu_backend.multi_source_dijkstra(..., sources=...)",
            "source-to-distance-mapping in requested source order",
            True,
            subset_required=True,
            semantic_alias_of="dijkstra_batch",
        ),
    )
    structural = tuple(
        FunctionExecutionDescriptor(
            name,
            "Structural Holes",
            "nodes",
            ShardAxis.NODES,
            MergeStrategy.ORDERED_MAPPING,
            "{}(G, nodes=None, weight=None)".format(name),
            "gpu_eggpu_backend.{}(..., nodes=...)".format(name),
            "node-to-value mapping in requested node order",
            True,
        )
        for name in ("effective_size", "efficiency", "constraint", "hierarchy")
    )
    return centrality + paths + structural


def default_cooperative_adapter_registry() -> CooperativeAdapterRegistry:
    return CooperativeAdapterRegistry(default_function_descriptors())
