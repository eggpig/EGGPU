"""Topology-aware analytical planner for the EGMGPU control plane."""

from __future__ import annotations

import itertools
import math

from dataclasses import dataclass
from typing import Dict
from typing import FrozenSet
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

from .calibration import COOPERATIVE_MODES
from .calibration import CalibrationProfile
from .descriptors import CapabilityLevel
from .descriptors import CapabilityRegistry
from .descriptors import ExecutionMode
from .state import GraphSnapshot


class PlannerError(RuntimeError):
    """Base class for control-plane planning errors."""


class CapabilityUnavailableError(PlannerError):
    """Raised when an explicitly requested path is not executable."""


class NoFeasiblePlanError(PlannerError):
    """Raised when available device memory cannot execute a workload."""


@dataclass(frozen=True)
class DeviceProfile:
    device_id: int
    memory_bytes: int
    reserved_bytes: int = 0
    compute_scale: float = 1.0
    h2d_bandwidth_bytes_per_second: float = 12.0e9
    d2h_bandwidth_bytes_per_second: float = 12.0e9
    queue_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.device_id < 0:
            raise ValueError("device_id must be non-negative")
        if self.memory_bytes <= 0:
            raise ValueError("memory_bytes must be positive")
        if self.reserved_bytes < 0 or self.reserved_bytes >= self.memory_bytes:
            raise ValueError("reserved_bytes must be in [0, memory_bytes)")
        if self.compute_scale <= 0:
            raise ValueError("compute_scale must be positive")
        if self.h2d_bandwidth_bytes_per_second <= 0:
            raise ValueError("H2D bandwidth must be positive")
        if self.d2h_bandwidth_bytes_per_second <= 0:
            raise ValueError("D2H bandwidth must be positive")
        if self.queue_seconds < 0:
            raise ValueError("queue_seconds must be non-negative")

    @property
    def usable_memory_bytes(self) -> int:
        return self.memory_bytes - self.reserved_bytes


@dataclass(frozen=True)
class TopologyProfile:
    """Measured link properties; absent peer links use host staging."""

    peer_bandwidth_bytes_per_second: Mapping[Tuple[int, int], float]
    host_staging_bandwidth_bytes_per_second: float = 10.0e9
    collective_bandwidth_bytes_per_second: Optional[float] = None
    message_latency_seconds: float = 5.0e-6

    def __post_init__(self) -> None:
        if self.host_staging_bandwidth_bytes_per_second <= 0:
            raise ValueError("host-staging bandwidth must be positive")
        if (
            self.collective_bandwidth_bytes_per_second is not None
            and self.collective_bandwidth_bytes_per_second <= 0
        ):
            raise ValueError("collective bandwidth must be positive")
        if self.message_latency_seconds < 0:
            raise ValueError("message latency must be non-negative")
        if any(value <= 0 for value in self.peer_bandwidth_bytes_per_second.values()):
            raise ValueError("peer bandwidths must be positive")

    def peer_bandwidth(self, left: int, right: int) -> float:
        if left == right:
            return math.inf
        return self.peer_bandwidth_bytes_per_second.get(
            (left, right),
            self.peer_bandwidth_bytes_per_second.get(
                (right, left), self.host_staging_bandwidth_bytes_per_second
            ),
        )

    def group_bandwidth(self, devices: Sequence[int]) -> float:
        if len(devices) < 2:
            return math.inf
        if self.collective_bandwidth_bytes_per_second is not None:
            return self.collective_bandwidth_bytes_per_second
        return min(
            self.peer_bandwidth(left, right)
            for left, right in itertools.combinations(devices, 2)
        )

    @classmethod
    def uniform(
        cls, devices: Sequence[int], bandwidth_bytes_per_second: float
    ) -> "TopologyProfile":
        links = {
            (left, right): bandwidth_bytes_per_second
            for left, right in itertools.combinations(sorted(set(devices)), 2)
        }
        return cls(peer_bandwidth_bytes_per_second=links)


@dataclass(frozen=True)
class WorkloadEstimate:
    function: str
    snapshot: GraphSnapshot
    graph_state_bytes: int
    workspace_bytes: int
    output_bytes: int
    single_compute_seconds: float
    communication_bytes: int = 0
    communication_rounds: int = 0
    state_preparation_seconds: float = 0.0
    materialization_seconds: float = 0.0
    merge_seconds: float = 0.0
    partition_overhead_fraction: float = 0.10
    concurrent_requests: int = 1
    resident_devices: FrozenSet[int] = frozenset()
    calibration_scale: float = 1.0

    def __post_init__(self) -> None:
        sizes = (self.graph_state_bytes, self.workspace_bytes, self.output_bytes)
        if any(value < 0 for value in sizes):
            raise ValueError("workload byte counts must be non-negative")
        if self.single_compute_seconds < 0 or self.materialization_seconds < 0:
            raise ValueError("workload times must be non-negative")
        if self.merge_seconds < 0 or self.state_preparation_seconds < 0:
            raise ValueError("state preparation and merge times must be non-negative")
        if self.communication_bytes < 0 or self.communication_rounds < 0:
            raise ValueError("communication estimates must be non-negative")
        if self.partition_overhead_fraction < 0:
            raise ValueError("partition_overhead_fraction must be non-negative")
        if self.concurrent_requests < 1:
            raise ValueError("concurrent_requests must be at least one")
        if not math.isfinite(self.calibration_scale) or self.calibration_scale <= 0:
            raise ValueError("calibration_scale must be finite and positive")


@dataclass(frozen=True)
class CostModelConfig:
    minimum_multigpu_gain: float = 0.05
    replicated_parallel_efficiency: float = 0.90
    partitioned_parallel_efficiency: float = 0.78
    request_parallel_efficiency: float = 0.95
    communication_overlap_fraction: float = 0.50

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_multigpu_gain < 1:
            raise ValueError("minimum_multigpu_gain must be in [0, 1)")
        for name in (
            "replicated_parallel_efficiency",
            "partitioned_parallel_efficiency",
            "request_parallel_efficiency",
        ):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.communication_overlap_fraction <= 1:
            raise ValueError("communication overlap must be in [0, 1]")


@dataclass(frozen=True)
class CostBreakdown:
    queue: float
    state: float
    compute: float
    communication: float
    overlap: float
    merge: float
    materialize: float
    calibrated_resident_total: Optional[float] = None

    @property
    def total(self) -> float:
        if self.calibrated_resident_total is not None:
            return self.queue + self.state + self.calibrated_resident_total
        return (
            self.queue
            + self.state
            + self.compute
            + self.communication
            - self.overlap
            + self.merge
            + self.materialize
        )


@dataclass(frozen=True)
class PlanOverride:
    mode: Optional[ExecutionMode] = None
    devices: Optional[Tuple[int, ...]] = None
    allow_reference: bool = False

    def __post_init__(self) -> None:
        if self.mode is not None:
            normalized_mode = ExecutionMode.parse(self.mode)
            object.__setattr__(
                self,
                "mode",
                None if normalized_mode is ExecutionMode.AUTO else normalized_mode,
            )
        if self.devices is not None:
            canonical = tuple(sorted(set(self.devices)))
            if not canonical:
                raise ValueError("an explicit device group must not be empty")
            if canonical != self.devices:
                raise ValueError("explicit devices must be sorted and unique")


@dataclass(frozen=True)
class ExecutionPlan:
    function: str
    mode: ExecutionMode
    devices: Tuple[int, ...]
    capability: CapabilityLevel
    cost: CostBreakdown
    required_memory_bytes: Mapping[int, int]
    reason: str
    explicit: bool = False

    @property
    def executable(self) -> bool:
        return self.capability.executable


class AnalyticalCostModel:
    """Deterministic implementation of the paper-facing latency equation."""

    def __init__(self, config: Optional[CostModelConfig] = None) -> None:
        self.config = CostModelConfig() if config is None else config

    def required_memory(
        self, mode: ExecutionMode, devices: Sequence[int], work: WorkloadEstimate
    ) -> Dict[int, int]:
        count = len(devices)
        if count == 0:
            raise ValueError("cannot size an empty device group")
        if mode in (ExecutionMode.SINGLE_DEVICE, ExecutionMode.REQUEST_PARALLEL):
            return {
                device: work.graph_state_bytes + work.workspace_bytes
                for device in devices
            }
        if mode is ExecutionMode.REPLICATED_TASK_PARALLEL:
            per_device_workspace = math.ceil(work.workspace_bytes / count)
            return {
                device: work.graph_state_bytes + per_device_workspace
                for device in devices
            }
        if mode is ExecutionMode.PARTITIONED_ITERATIVE:
            partition_state = math.ceil(
                work.graph_state_bytes * (1.0 + work.partition_overhead_fraction) / count
            )
            partition_workspace = math.ceil(work.workspace_bytes / count)
            return {
                device: partition_state + partition_workspace for device in devices
            }
        raise ValueError(f"cannot size execution mode {mode.value}")

    def estimate(
        self,
        mode: ExecutionMode,
        devices: Sequence[DeviceProfile],
        topology: TopologyProfile,
        work: WorkloadEstimate,
    ) -> CostBreakdown:
        if not devices:
            raise ValueError("cannot estimate an empty device group")
        ids = tuple(device.device_id for device in devices)
        count = len(devices)
        jobs = work.concurrent_requests
        queue = max(device.queue_seconds for device in devices)
        resident = all(device in work.resident_devices for device in ids)

        if mode in (ExecutionMode.SINGLE_DEVICE, ExecutionMode.REQUEST_PARALLEL):
            bytes_per_device = work.graph_state_bytes
        elif mode is ExecutionMode.REPLICATED_TASK_PARALLEL:
            bytes_per_device = work.graph_state_bytes
        else:
            bytes_per_device = math.ceil(
                work.graph_state_bytes * (1.0 + work.partition_overhead_fraction) / count
            )
        state = 0.0
        if not resident:
            state = work.state_preparation_seconds + max(
                bytes_per_device / device.h2d_bandwidth_bytes_per_second
                for device in devices
            )

        aggregate_compute = sum(device.compute_scale for device in devices)
        if mode is ExecutionMode.SINGLE_DEVICE:
            compute = work.single_compute_seconds * jobs / devices[0].compute_scale
            efficiency = 1.0
        elif mode is ExecutionMode.REPLICATED_TASK_PARALLEL:
            efficiency = self.config.replicated_parallel_efficiency
            compute = work.single_compute_seconds * jobs / (
                aggregate_compute * efficiency
            )
        elif mode is ExecutionMode.PARTITIONED_ITERATIVE:
            efficiency = self.config.partitioned_parallel_efficiency
            compute = work.single_compute_seconds * jobs / (
                aggregate_compute * efficiency
            )
        elif mode is ExecutionMode.REQUEST_PARALLEL:
            efficiency = self.config.request_parallel_efficiency
            compute = work.single_compute_seconds * jobs / (
                aggregate_compute * efficiency
            )
        else:
            raise ValueError(f"cannot estimate execution mode {mode.value}")

        communication = 0.0
        if mode in (
            ExecutionMode.REPLICATED_TASK_PARALLEL,
            ExecutionMode.PARTITIONED_ITERATIVE,
        ):
            bandwidth = topology.group_bandwidth(ids)
            communication = (
                0.0
                if math.isinf(bandwidth)
                else work.communication_bytes / bandwidth
            ) + work.communication_rounds * topology.message_latency_seconds
        overlap = self.config.communication_overlap_fraction * min(
            compute, communication
        )

        coordinator = min(devices, key=lambda device: device.device_id)
        returned_results = jobs if mode is ExecutionMode.REQUEST_PARALLEL else jobs
        merge = work.merge_seconds * returned_results
        merge += (
            work.output_bytes
            * returned_results
            / coordinator.d2h_bandwidth_bytes_per_second
        )
        materialize = work.materialization_seconds * returned_results
        return CostBreakdown(
            queue=queue,
            state=state,
            compute=compute,
            communication=communication,
            overlap=overlap,
            merge=merge,
            materialize=materialize,
        )


class AdaptivePlanner:
    """Select an executable plan while preserving explicit user choices."""

    def __init__(
        self,
        devices: Sequence[DeviceProfile],
        topology: TopologyProfile,
        capabilities: CapabilityRegistry,
        cost_model: Optional[AnalyticalCostModel] = None,
        calibration: Optional[CalibrationProfile] = None,
        topology_fingerprint: Optional[str] = None,
        available_device_groups: Optional[
            Mapping[Tuple[str, ExecutionMode], Sequence[Tuple[int, ...]]]
        ] = None,
    ) -> None:
        if not devices:
            raise ValueError("the planner requires at least one device")
        self._devices = {device.device_id: device for device in devices}
        if len(self._devices) != len(devices):
            raise ValueError("device identifiers must be unique")
        self.topology = topology
        self.capabilities = capabilities
        self.cost_model = AnalyticalCostModel() if cost_model is None else cost_model
        self.calibration = calibration
        self.topology_fingerprint = topology_fingerprint
        self._available_device_groups = self._normalize_device_groups(
            available_device_groups
        )

    def plan(
        self, work: WorkloadEstimate, override: Optional[PlanOverride] = None
    ) -> ExecutionPlan:
        override = PlanOverride() if override is None else override
        self.capabilities.descriptor(work.function)
        allowed_devices = (
            tuple(sorted(self._devices))
            if override.devices is None
            else override.devices
        )
        unknown = set(allowed_devices) - set(self._devices)
        if unknown:
            raise ValueError(f"unknown devices requested: {sorted(unknown)}")

        if override.mode is not None:
            return self._explicit_plan(work, override, allowed_devices)

        if override.devices is not None:
            return self._explicit_device_group_plan(work, allowed_devices)

        candidates = self._candidates(
            work,
            allowed_devices,
            include_reference=False,
            modes=None,
            explicit=False,
            require_calibration=True,
        )
        if not candidates:
            raise NoFeasiblePlanError(
                f"no available execution path can fit {work.function!r}"
            )

        singles = [p for p in candidates if p.mode is ExecutionMode.SINGLE_DEVICE]
        multigpu = [p for p in candidates if len(p.devices) > 1]
        if not singles:
            partitioned = [
                p for p in multigpu if p.mode is ExecutionMode.PARTITIONED_ITERATIVE
            ]
            if not partitioned:
                raise NoFeasiblePlanError(
                    "the graph exceeds single-device capacity and no executable "
                    "partitioned path is available"
                )
            minimum_devices = min(len(plan.devices) for plan in partitioned)
            capacity_plans = [
                plan for plan in partitioned if len(plan.devices) == minimum_devices
            ]
            chosen = min(capacity_plans, key=self._plan_order)
            return self._with_reason(
                chosen,
                "single-device capacity exceeded; selected the smallest feasible "
                "partitioned device group",
            )

        best_single = min(singles, key=self._plan_order)
        if not multigpu:
            available_cooperative = any(
                self.capabilities.level(work.function, mode).executable
                for mode in COOPERATIVE_MODES
            )
            reason = (
                "cooperative AUTO requires fresh, topology-matched calibration; "
                "selected the safe single-device path"
                if available_cooperative
                else "only single-device path is executable"
            )
            return self._with_reason(best_single, reason)
        best_multi = min(multigpu, key=self._plan_order)
        gate = 1.0 - self.cost_model.config.minimum_multigpu_gain
        if best_multi.cost.total <= best_single.cost.total * gate:
            return self._with_reason(
                best_multi,
                f"predicted latency clears the {self.cost_model.config.minimum_multigpu_gain:.0%} "
                "multi-device gain gate",
            )
        return self._with_reason(
            best_single,
            f"multi-device prediction does not clear the "
            f"{self.cost_model.config.minimum_multigpu_gain:.0%} gain gate",
        )

    def _explicit_device_group_plan(
        self, work: WorkloadEstimate, devices: Tuple[int, ...]
    ) -> ExecutionPlan:
        modes = (
            (ExecutionMode.SINGLE_DEVICE,)
            if len(devices) == 1
            else (
                ExecutionMode.REQUEST_PARALLEL,
                ExecutionMode.REPLICATED_TASK_PARALLEL,
                ExecutionMode.PARTITIONED_ITERATIVE,
            )
        )
        candidates = self._candidates(
            work,
            devices,
            include_reference=False,
            modes=modes,
            explicit=True,
            exact_group=True,
            require_calibration=True,
        )
        if not candidates:
            raise NoFeasiblePlanError(
                "no executable mode can use the complete explicit device group"
            )
        chosen = min(candidates, key=self._plan_order)
        return self._with_reason(chosen, "explicit device group preserved")

    def _explicit_plan(
        self,
        work: WorkloadEstimate,
        override: PlanOverride,
        allowed_devices: Tuple[int, ...],
    ) -> ExecutionPlan:
        assert override.mode is not None
        level = self.capabilities.level(work.function, override.mode)
        if level is CapabilityLevel.UNAVAILABLE:
            raise CapabilityUnavailableError(
                f"{work.function} does not support {override.mode.value}"
            )
        if level is CapabilityLevel.REFERENCE and not override.allow_reference:
            raise CapabilityUnavailableError(
                f"{work.function} {override.mode.value} is a reference design, not "
                "an executable CUDA path"
            )
        candidates = self._candidates(
            work,
            allowed_devices,
            include_reference=override.allow_reference,
            modes=(override.mode,),
            explicit=True,
            exact_group=override.devices is not None,
            require_calibration=False,
        )
        if not candidates:
            raise NoFeasiblePlanError(
                f"requested {override.mode.value} plan cannot fit the selected devices"
            )
        chosen = min(candidates, key=self._plan_order)
        return self._with_reason(chosen, "explicit user override preserved")

    def _candidates(
        self,
        work: WorkloadEstimate,
        allowed_devices: Tuple[int, ...],
        *,
        include_reference: bool,
        modes: Optional[Tuple[ExecutionMode, ...]],
        explicit: bool,
        exact_group: bool = False,
        require_calibration: bool = False,
    ) -> Tuple[ExecutionPlan, ...]:
        selected_modes = modes or tuple(
            mode
            for mode in ExecutionMode
            if mode is not ExecutionMode.AUTO
            and (
                self.capabilities.level(work.function, mode).executable
                or (
                    include_reference
                    and self.capabilities.level(work.function, mode)
                    is CapabilityLevel.REFERENCE
                )
            )
        )
        plans = []
        for mode in selected_modes:
            level = self.capabilities.level(work.function, mode)
            if level is CapabilityLevel.UNAVAILABLE:
                continue
            if level is CapabilityLevel.REFERENCE and not include_reference:
                continue
            for group in self._groups(mode, allowed_devices, work, exact_group):
                if not self._device_group_available(work.function, mode, group):
                    continue
                if (
                    require_calibration
                    and mode in COOPERATIVE_MODES
                    and not self._cooperative_calibrated(
                        work.function, mode, len(group), work.calibration_scale
                    )
                ):
                    continue
                profiles = tuple(self._devices[device] for device in group)
                required = self.cost_model.required_memory(mode, group, work)
                if any(
                    required[device] > self._devices[device].usable_memory_bytes
                    for device in group
                ):
                    continue
                cost = self.cost_model.estimate(mode, profiles, self.topology, work)
                cost = self._apply_calibration(
                    work.function, mode, len(group), work, profiles, cost
                )
                plans.append(
                    ExecutionPlan(
                        function=self.capabilities.descriptor(work.function).name,
                        mode=mode,
                        devices=group,
                        capability=level,
                        cost=cost,
                        required_memory_bytes=required,
                        reason="candidate",
                        explicit=explicit,
                    )
                )
        return tuple(plans)

    def _normalize_device_groups(self, value):
        if value is None:
            return None
        normalized = {}
        for (function, mode), groups in value.items():
            canonical = self.capabilities.descriptor(str(function)).name.lower()
            parsed_mode = ExecutionMode.parse(mode)
            canonical_groups = set()
            for group in groups:
                devices = tuple(int(device) for device in group)
                if not devices or tuple(sorted(set(devices))) != devices:
                    raise ValueError(
                        "available device groups must be non-empty, sorted, and unique"
                    )
                unknown = set(devices) - set(self._devices)
                if unknown:
                    raise ValueError(
                        "available device group contains unknown devices: {}".format(
                            sorted(unknown)
                        )
                    )
                canonical_groups.add(devices)
            normalized[(canonical, parsed_mode)] = frozenset(canonical_groups)
        return normalized

    def _device_group_available(
        self, function: str, mode: ExecutionMode, group: Tuple[int, ...]
    ) -> bool:
        if self._available_device_groups is None:
            return True
        canonical = self.capabilities.descriptor(function).name.lower()
        allowed = self._available_device_groups.get((canonical, mode))
        # A missing entry leaves ordinary single/request paths unconstrained.
        # Cooperative paths are exact evidence-scoped capabilities: when a
        # group map is supplied, an absent cooperative entry means no group
        # has been validated on this deployment.
        if allowed is None:
            return mode not in COOPERATIVE_MODES
        return group in allowed

    def _cooperative_calibrated(
        self,
        function: str,
        mode: ExecutionMode,
        device_count: int,
        calibration_scale: float,
    ) -> bool:
        if self.calibration is None:
            return False
        if not self.calibration.sufficient(
            function,
            ExecutionMode.SINGLE_DEVICE,
            1,
            topology_fingerprint=self.topology_fingerprint,
        ):
            return False
        if not self.calibration.sufficient(
            function,
            mode,
            device_count,
            topology_fingerprint=self.topology_fingerprint,
        ):
            return False
        coefficient = self.calibration.coefficient(function, mode, device_count)
        return bool(
            coefficient is not None
            and coefficient.break_even_scale is not None
            and calibration_scale >= coefficient.break_even_scale
        )

    def _apply_calibration(
        self,
        function: str,
        mode: ExecutionMode,
        device_count: int,
        work: WorkloadEstimate,
        devices: Sequence[DeviceProfile],
        analytical: CostBreakdown,
    ) -> CostBreakdown:
        if self.calibration is None or not self.calibration.sufficient(
            function,
            mode,
            device_count,
            topology_fingerprint=self.topology_fingerprint,
        ):
            return analytical
        coefficient = self.calibration.coefficient(function, mode, device_count)
        assert coefficient is not None
        resident = all(
            device.device_id in work.resident_devices for device in devices
        )
        scale = work.calibration_scale
        return CostBreakdown(
            queue=analytical.queue,
            state=0.0 if resident else coefficient.state * scale,
            compute=coefficient.compute * scale,
            communication=coefficient.communication * scale,
            overlap=0.0,
            merge=coefficient.merge * scale,
            materialize=coefficient.materialize * scale,
            # Batch coefficients drive planning; EMA remains online telemetry.
            calibrated_resident_total=coefficient.total * scale,
        )

    def _groups(
        self,
        mode: ExecutionMode,
        devices: Tuple[int, ...],
        work: WorkloadEstimate,
        exact_group: bool,
    ) -> Tuple[Tuple[int, ...], ...]:
        if exact_group:
            if mode is ExecutionMode.SINGLE_DEVICE and len(devices) != 1:
                raise ValueError("single mode requires exactly one explicit device")
            if mode is not ExecutionMode.SINGLE_DEVICE and len(devices) < 2:
                raise ValueError(
                    f"{mode.value} mode requires at least two explicit devices"
                )
            return (devices,)
        if mode is ExecutionMode.SINGLE_DEVICE:
            return tuple((device,) for device in devices)
        if mode is ExecutionMode.REQUEST_PARALLEL:
            if work.concurrent_requests < 2:
                return ()
            maximum = min(len(devices), work.concurrent_requests)
            return tuple(
                group
                for count in range(2, maximum + 1)
                for group in itertools.combinations(devices, count)
            )
        return tuple(
            group
            for count in range(2, len(devices) + 1)
            for group in itertools.combinations(devices, count)
        )

    @staticmethod
    def _plan_order(plan: ExecutionPlan) -> Tuple[float, int, Tuple[int, ...]]:
        return (plan.cost.total, len(plan.devices), plan.devices)

    @staticmethod
    def _with_reason(plan: ExecutionPlan, reason: str) -> ExecutionPlan:
        return ExecutionPlan(
            function=plan.function,
            mode=plan.mode,
            devices=plan.devices,
            capability=plan.capability,
            cost=plan.cost,
            required_memory_bytes=plan.required_memory_bytes,
            reason=reason,
            explicit=plan.explicit,
        )
