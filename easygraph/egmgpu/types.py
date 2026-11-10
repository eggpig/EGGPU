"""Public value types used by the EGMGPU runtime.

The types in this module intentionally have no CUDA or EasyGraph imports.  A
workflow can therefore be assembled and validated before the GPU runtime is
initialized.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


class ExecutionMode(str, Enum):
    """Execution plans understood by the EGMGPU scheduler."""

    AUTO = "auto"
    SINGLE = "single"
    REQUEST_PARALLEL = "request_parallel"
    REPLICATED = "replicated"
    PARTITIONED = "partitioned"

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                "unknown EGMGPU execution mode {!r}; expected one of {}".format(
                    value, choices
                )
            ) from exc


class RequestSchedulingPolicy(str, Enum):
    """Controller policies for unconstrained production ``AUTO`` requests.

    ``FIFO`` preserves the original evidence-scoped reservation behavior.  The
    four remaining policies are explicit RQ3 treatments and require a measured
    deployment planner so device-memory feasibility is checked before a
    request enters the reservation queue.
    """

    FIFO = "fifo"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    REUSE_AFFINITY = "reuse_affinity"
    ADAPTIVE = "adaptive"

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                "unknown EGMGPU request scheduling policy {!r}; expected one of {}".format(
                    value, choices
                )
            ) from exc


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable configuration of one process-local EGMGPU runtime."""

    devices: Tuple[int, ...]
    mode: ExecutionMode = ExecutionMode.AUTO
    min_predicted_gain: float = 0.05
    scheduling_policy: RequestSchedulingPolicy = RequestSchedulingPolicy.FIFO
    reservation_policy: str = "fifo"

    def __post_init__(self):
        if self.reservation_policy not in {"fifo", "conflict_aware"}:
            raise ValueError("reservation_policy must be fifo or conflict_aware")
        normalized = tuple(int(device) for device in self.devices)
        if not normalized:
            raise ValueError("EGMGPU requires at least one CUDA device")
        if any(device < 0 for device in normalized):
            raise ValueError("CUDA device identifiers must be non-negative")
        if len(set(normalized)) != len(normalized):
            raise ValueError("CUDA device identifiers must be unique")
        # C++ DeviceGroup canonicalizes device order.  Mirror that rule so
        # owner validation, deterministic merging, and probe evidence all use
        # one stable device ordering.
        normalized = tuple(sorted(normalized))
        if not 0.0 <= float(self.min_predicted_gain) < 1.0:
            raise ValueError("min_predicted_gain must be in [0, 1)")
        object.__setattr__(self, "devices", normalized)
        object.__setattr__(self, "mode", ExecutionMode.parse(self.mode))
        object.__setattr__(self, "min_predicted_gain", float(self.min_predicted_gain))
        object.__setattr__(
            self,
            "scheduling_policy",
            RequestSchedulingPolicy.parse(self.scheduling_policy),
        )
        if (
            self.scheduling_policy is not RequestSchedulingPolicy.FIFO
            and self.mode is not ExecutionMode.AUTO
        ):
            raise ValueError(
                "round-robin, least-loaded, reuse-affinity, and adaptive "
                "scheduling apply only to AUTO mode; explicit execution modes "
                "already define their placement boundary"
            )


@dataclass(frozen=True)
class CallSpec:
    """A graph function and its arguments in an explicit workflow."""

    function: Callable[..., Any]
    args: Tuple[Any, ...]
    kwargs: Mapping[str, Any]
    mode: Optional[ExecutionMode] = None
    devices: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if not callable(self.function):
            raise TypeError("function must be callable")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        if self.mode is not None:
            object.__setattr__(self, "mode", ExecutionMode.parse(self.mode))
        if self.devices is not None:
            normalized = tuple(int(device) for device in self.devices)
            if not normalized:
                raise ValueError("a call-specific device set cannot be empty")
            if len(set(normalized)) != len(normalized):
                raise ValueError("call-specific CUDA devices must be unique")
            object.__setattr__(self, "devices", tuple(sorted(normalized)))


@dataclass(frozen=True)
class CooperativeContext:
    """Device assignment supplied to a cooperative-plan builder."""

    devices: Tuple[int, ...]
    mode: ExecutionMode
    request_id: int
    graph_snapshot: Optional[Any] = None


@dataclass(frozen=True)
class NativeCallContext:
    """Controller-side context for one native-runtime invocation.

    The registered adapter executes on a Python controller thread.  It may
    inspect Python arguments and construct an immutable native request, but
    only that native request may be submitted to the C++ device workers.
    """

    devices: Tuple[int, ...]
    mode: ExecutionMode
    request_id: int
    graph_snapshot: Optional[Any] = None
    route_id: Optional[str] = None
    route_layout: Optional[str] = None


@dataclass(frozen=True)
class CooperativePlan:
    """Per-device work and deterministic merge for one cooperative call.

    ``shards`` must contain exactly one zero-argument callable for every
    assigned device.  Device workers run those callables concurrently.  The
    scheduler passes results to ``merge`` in assigned-device order.
    """

    shards: Mapping[int, Callable[[], Any]]
    merge: Callable[[Sequence[Any]], Any]
    cleanup: Optional[Callable[[], None]] = None
    metadata: Optional[Any] = None

    def validate(self, devices):
        expected = tuple(devices)
        actual = tuple(sorted(int(device) for device in self.shards))
        if actual != tuple(sorted(expected)):
            raise ValueError(
                "cooperative plan shards {} do not match assigned devices {}".format(
                    actual, expected
                )
            )
        if not all(callable(task) for task in self.shards.values()):
            raise TypeError("every cooperative shard must be callable")
        if not callable(self.merge):
            raise TypeError("cooperative plan merge must be callable")


CooperativePlanBuilder = Callable[
    [CooperativeContext, Callable[..., Any], Tuple[Any, ...], Dict[str, Any]],
    CooperativePlan,
]

NativeExecutor = Callable[
    [NativeCallContext, Tuple[Any, ...], Dict[str, Any]],
    Any,
]
