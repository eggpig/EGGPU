"""Function-family execution descriptors and truthful capability metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Tuple


class ExecutionMode(str, Enum):
    AUTO = "auto"
    SINGLE_DEVICE = "single"
    REPLICATED_TASK_PARALLEL = "replicated"
    PARTITIONED_ITERATIVE = "partitioned"
    REQUEST_PARALLEL = "request_parallel"

    @classmethod
    def parse(cls, value: object) -> "ExecutionMode":
        """Normalize public-runtime enums and wire-format strings."""

        if isinstance(value, cls):
            return value
        raw = getattr(value, "value", value)
        try:
            return cls(str(raw).strip().lower())
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"unknown EGMGPU execution mode {value!r}; expected {choices}"
            ) from exc


class CapabilityLevel(str, Enum):
    """Whether an execution path exists, is designed, or is unsupported."""

    UNAVAILABLE = "unavailable"
    REFERENCE = "reference"
    AVAILABLE = "available"

    @property
    def executable(self) -> bool:
        return self is CapabilityLevel.AVAILABLE


class FunctionFamily(str, Enum):
    CENTRALITY = "centrality"
    CONNECTIVITY = "connectivity"
    PATH_AND_SPANNING = "path_and_spanning"
    STRUCTURAL_HOLES = "structural_holes"


class DecompositionAxis(str, Enum):
    NONE = "none"
    SOURCE = "source"
    OUTPUT_VERTEX = "output_vertex"
    GRAPH_PARTITION = "graph_partition"
    REQUEST = "request"


@dataclass(frozen=True)
class FunctionDescriptor:
    """Stable control-plane description of one public graph function."""

    name: str
    public_name: str
    family: FunctionFamily
    decomposition: Mapping[ExecutionMode, DecompositionAxis]
    capabilities: Mapping[ExecutionMode, CapabilityLevel]
    required_view: str = "csr:outgoing:unweighted"
    combine: Optional[str] = None
    termination: Optional[str] = None
    result_merge: str = "coordinator_materialization"

    def capability(self, mode: ExecutionMode) -> CapabilityLevel:
        mode = ExecutionMode.parse(mode)
        return self.capabilities.get(mode, CapabilityLevel.UNAVAILABLE)

    @property
    def cooperative_modes(self) -> Tuple[ExecutionMode, ...]:
        return tuple(
            mode
            for mode in (
                ExecutionMode.REPLICATED_TASK_PARALLEL,
                ExecutionMode.PARTITIONED_ITERATIVE,
            )
            if self.capability(mode) is not CapabilityLevel.UNAVAILABLE
        )


class CapabilityRegistry:
    """Mutable deployment view over immutable function descriptors.

    The default registry records implemented single-device EGGPU paths as
    ``AVAILABLE``.  Planned EGMGPU paths are ``REFERENCE`` and therefore are
    never selected by an executable plan until a native integration explicitly
    promotes them after probing the backend.
    """

    def __init__(self, descriptors: Iterable[FunctionDescriptor]) -> None:
        self._descriptors: Dict[str, FunctionDescriptor] = {}
        self._aliases: Dict[str, str] = {}
        self._levels: Dict[Tuple[str, ExecutionMode], CapabilityLevel] = {}
        for descriptor in descriptors:
            canonical = descriptor.name.lower()
            if canonical in self._descriptors:
                raise ValueError(f"duplicate function descriptor: {descriptor.name}")
            self._descriptors[canonical] = descriptor
            for alias in {descriptor.name, descriptor.public_name}:
                normalized = alias.lower()
                previous = self._aliases.get(normalized)
                if previous is not None and previous != canonical:
                    raise ValueError(f"ambiguous function alias: {alias}")
                self._aliases[normalized] = canonical
            for mode, level in descriptor.capabilities.items():
                self._levels[(canonical, mode)] = level

    def descriptor(self, function: str) -> FunctionDescriptor:
        canonical = self._aliases.get(function.lower(), function.lower())
        try:
            return self._descriptors[canonical]
        except KeyError as exc:
            raise KeyError(f"unknown EGMGPU function {function!r}") from exc

    def level(self, function: str, mode: ExecutionMode) -> CapabilityLevel:
        mode = ExecutionMode.parse(mode)
        descriptor = self.descriptor(function)
        return self._levels.get(
            (descriptor.name.lower(), mode), CapabilityLevel.UNAVAILABLE
        )

    def set_level(
        self, function: str, mode: ExecutionMode, level: CapabilityLevel
    ) -> None:
        """Set a probed deployment capability without rewriting descriptors."""

        mode = ExecutionMode.parse(mode)
        descriptor = self.descriptor(function)
        declared = descriptor.capability(mode)
        if declared is CapabilityLevel.UNAVAILABLE and level is not declared:
            raise ValueError(
                f"{descriptor.name} does not declare {mode.value} as a valid path"
            )
        self._levels[(descriptor.name.lower(), mode)] = level

    def executable_modes(self, function: str) -> Tuple[ExecutionMode, ...]:
        return tuple(
            mode
            for mode in ExecutionMode
            if mode is not ExecutionMode.AUTO and self.level(function, mode).executable
        )

    def functions(self) -> Tuple[str, ...]:
        return tuple(sorted(self._descriptors))

    def support_matrix(self) -> Dict[str, Dict[str, str]]:
        return {
            name: {
                mode.value: self.level(name, mode).value
                for mode in ExecutionMode
                if mode is not ExecutionMode.AUTO
            }
            for name in self.functions()
        }


def _capabilities(cooperative: Optional[ExecutionMode]) -> Dict[ExecutionMode, CapabilityLevel]:
    capabilities = {
        ExecutionMode.SINGLE_DEVICE: CapabilityLevel.AVAILABLE,
        ExecutionMode.REQUEST_PARALLEL: CapabilityLevel.REFERENCE,
    }
    if cooperative is not None:
        capabilities[cooperative] = CapabilityLevel.REFERENCE
    return capabilities


def _decomposition(
    cooperative: Optional[ExecutionMode], axis: DecompositionAxis
) -> Dict[ExecutionMode, DecompositionAxis]:
    result = {
        ExecutionMode.SINGLE_DEVICE: DecompositionAxis.NONE,
        ExecutionMode.REQUEST_PARALLEL: DecompositionAxis.REQUEST,
    }
    if cooperative is not None:
        result[cooperative] = axis
    return result


def _descriptor(
    name: str,
    public_name: str,
    family: FunctionFamily,
    cooperative: Optional[ExecutionMode] = None,
    axis: DecompositionAxis = DecompositionAxis.NONE,
    **kwargs: object,
) -> FunctionDescriptor:
    return FunctionDescriptor(
        name=name,
        public_name=public_name,
        family=family,
        decomposition=_decomposition(cooperative, axis),
        capabilities=_capabilities(cooperative),
        **kwargs,
    )


def default_function_descriptors() -> Tuple[FunctionDescriptor, ...]:
    partitioned = ExecutionMode.PARTITIONED_ITERATIVE
    replicated = ExecutionMode.REPLICATED_TASK_PARALLEL
    return (
        _descriptor(
            "pagerank",
            "pagerank",
            FunctionFamily.CENTRALITY,
            partitioned,
            DecompositionAxis.GRAPH_PARTITION,
            required_view="csr:incoming:weighted-optional",
            combine="sum",
            termination="global_residual",
            result_merge="owned_vertex_concat",
        ),
        _descriptor(
            "mst", "minimum_spanning_tree", FunctionFamily.PATH_AND_SPANNING
        ),
        _descriptor("lcc", "clustering", FunctionFamily.CONNECTIVITY),
        _descriptor(
            "wcc",
            "connected_components",
            FunctionFamily.CONNECTIVITY,
            partitioned,
            DecompositionAxis.GRAPH_PARTITION,
            combine="min_label",
            termination="global_no_change",
            result_merge="component_canonicalization",
        ),
        _descriptor(
            "scc", "strongly_connected_components", FunctionFamily.CONNECTIVITY
        ),
        _descriptor(
            "bfs",
            "bfs",
            FunctionFamily.PATH_AND_SPANNING,
            partitioned,
            DecompositionAxis.GRAPH_PARTITION,
            combine="first_owner_discovery",
            termination="global_empty_frontier",
            result_merge="owned_vertex_concat",
        ),
        _descriptor(
            "dijkstra", "single_source_dijkstra", FunctionFamily.PATH_AND_SPANNING
        ),
        _descriptor(
            "bellman_ford",
            "multi_source_bellman_ford",
            FunctionFamily.PATH_AND_SPANNING,
        ),
        _descriptor(
            "sssp",
            "multi_source_dijkstra",
            FunctionFamily.PATH_AND_SPANNING,
            partitioned,
            DecompositionAxis.GRAPH_PARTITION,
            required_view="csr:outgoing:weighted",
            combine="min_distance",
            termination="global_empty_buckets",
            result_merge="owned_vertex_concat",
        ),
        _descriptor("kcore", "k_core", FunctionFamily.CONNECTIVITY),
        _descriptor(
            "bc",
            "betweenness_centrality",
            FunctionFamily.CENTRALITY,
            replicated,
            DecompositionAxis.SOURCE,
            combine="sum",
            result_merge="fixed_device_order_sum",
        ),
        _descriptor(
            "closeness",
            "closeness_centrality",
            FunctionFamily.CENTRALITY,
            replicated,
            DecompositionAxis.SOURCE,
            result_merge="source_order_concat",
        ),
        _descriptor(
            "effective_size",
            "effective_size",
            FunctionFamily.STRUCTURAL_HOLES,
            replicated,
            DecompositionAxis.OUTPUT_VERTEX,
            required_view="csr:ego:weighted-optional",
            result_merge="output_vertex_concat",
        ),
        _descriptor(
            "efficiency",
            "efficiency",
            FunctionFamily.STRUCTURAL_HOLES,
            replicated,
            DecompositionAxis.OUTPUT_VERTEX,
            required_view="csr:ego:weighted-optional",
            result_merge="output_vertex_concat",
        ),
        _descriptor(
            "constraint",
            "constraint",
            FunctionFamily.STRUCTURAL_HOLES,
            replicated,
            DecompositionAxis.OUTPUT_VERTEX,
            required_view="csr:ego:weighted-optional",
            result_merge="output_vertex_concat",
        ),
        _descriptor(
            "hierarchy",
            "hierarchy",
            FunctionFamily.STRUCTURAL_HOLES,
            replicated,
            DecompositionAxis.OUTPUT_VERTEX,
            required_view="csr:ego:weighted-optional",
            result_merge="output_vertex_concat",
        ),
    )


def default_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry(default_function_descriptors())
