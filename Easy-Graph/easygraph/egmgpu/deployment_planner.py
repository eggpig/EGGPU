"""Evidence-gated deployment planning for the production EGMGPU runtime.

This module connects the CPU-testable :mod:`easygraph.egmgpu_control` model to
production request submission.  It accepts already parsed measurements and
already validated capability evidence.  It never discovers files, probes a
GPU, starts workers, or upgrades a reference implementation during import.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import threading
import time

from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

try:
    from easygraph.egmgpu_control.calibration import CalibrationProfile
    from easygraph.egmgpu_control.descriptors import CapabilityLevel
    from easygraph.egmgpu_control.descriptors import DecompositionAxis
    from easygraph.egmgpu_control.descriptors import ExecutionMode as ControlMode
    from easygraph.egmgpu_control.descriptors import default_capability_registry
    from easygraph.egmgpu_control.planner import AdaptivePlanner
    from easygraph.egmgpu_control.planner import AnalyticalCostModel
    from easygraph.egmgpu_control.planner import CostModelConfig
    from easygraph.egmgpu_control.planner import DeviceProfile
    from easygraph.egmgpu_control.planner import ExecutionPlan
    from easygraph.egmgpu_control.planner import NoFeasiblePlanError
    from easygraph.egmgpu_control.planner import PlanOverride
    from easygraph.egmgpu_control.planner import TopologyProfile
    from easygraph.egmgpu_control.planner import WorkloadEstimate
except ModuleNotFoundError as error:
    # Dependency-free control tests place ``easygraph/`` itself on
    # PYTHONPATH, exposing the two subpackages as top-level modules.
    if error.name != "easygraph":
        raise
    from easygraph.egmgpu_control.calibration import CalibrationProfile
    from easygraph.egmgpu_control.descriptors import CapabilityLevel
    from easygraph.egmgpu_control.descriptors import DecompositionAxis
    from easygraph.egmgpu_control.descriptors import ExecutionMode as ControlMode
    from easygraph.egmgpu_control.descriptors import default_capability_registry
    from easygraph.egmgpu_control.planner import AdaptivePlanner
    from easygraph.egmgpu_control.planner import AnalyticalCostModel
    from easygraph.egmgpu_control.planner import CostModelConfig
    from easygraph.egmgpu_control.planner import DeviceProfile
    from easygraph.egmgpu_control.planner import ExecutionPlan
    from easygraph.egmgpu_control.planner import NoFeasiblePlanError
    from easygraph.egmgpu_control.planner import PlanOverride
    from easygraph.egmgpu_control.planner import TopologyProfile
    from easygraph.egmgpu_control.planner import WorkloadEstimate

from .types import ExecutionMode


COOPERATIVE_MODES = frozenset(
    {ControlMode.REPLICATED_TASK_PARALLEL, ControlMode.PARTITIONED_ITERATIVE}
)


class DeploymentPlanningError(RuntimeError):
    """Base error for the production planning bridge."""


class DeploymentCapabilityError(DeploymentPlanningError):
    """A requested production path lacks current execution evidence."""


@dataclass(frozen=True)
class CooperativeCapability:
    """Topology-scoped evidence that one native cooperative route is usable.

    The caller must construct this value from a validated binding registry.
    Merely constructing it does not register an executor and does not make a
    route selectable: production planning also requires a registered route and
    sufficient topology-matched calibration.
    """

    function: str
    mode: ControlMode
    devices: Tuple[int, ...]
    topology_fingerprint: str
    level: CapabilityLevel = CapabilityLevel.AVAILABLE

    def __post_init__(self) -> None:
        function = str(self.function).strip().lower()
        mode = ControlMode.parse(self.mode)
        devices = tuple(int(device) for device in self.devices)
        if not function:
            raise ValueError("cooperative capability function cannot be empty")
        if mode not in COOPERATIVE_MODES:
            raise ValueError("cooperative capability requires a cooperative mode")
        if len(devices) < 2 or tuple(sorted(set(devices))) != devices:
            raise ValueError(
                "cooperative capability devices must be sorted, unique, and contain at least two devices"
            )
        if not self.topology_fingerprint:
            raise ValueError("cooperative capability requires a topology fingerprint")
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "devices", devices)
        object.__setattr__(self, "level", CapabilityLevel(self.level))

    @classmethod
    def from_validated_binding_status(cls, status: Any) -> "CooperativeCapability":
        """Translate an already validated native-binding status.

        This method performs no evidence I/O.  The binding registry remains
        responsible for checksum, result, and topology validation before it
        returns an ``AVAILABLE`` status.
        """

        readiness = getattr(getattr(status, "readiness", None), "value", None)
        evidence = getattr(status, "evidence", None)
        spec = getattr(status, "spec", None)
        if readiness != "available" or evidence is None or spec is None:
            raise DeploymentCapabilityError(
                "native binding status is not AVAILABLE with retained evidence"
            )
        if hasattr(evidence, "passed") and not bool(evidence.passed):
            raise DeploymentCapabilityError(
                "native binding evidence did not pass its correctness probe"
            )
        return cls(
            function=str(spec.name),
            mode=ControlMode.parse(spec.mode),
            devices=tuple(evidence.devices),
            topology_fingerprint=str(evidence.topology_fingerprint),
            level=CapabilityLevel.AVAILABLE,
        )

    @classmethod
    def all_from_validated_binding_status(
        cls, status: Any
    ) -> Tuple["CooperativeCapability", ...]:
        """Translate every exact device group installed for one function."""

        readiness = getattr(getattr(status, "readiness", None), "value", None)
        spec = getattr(status, "spec", None)
        groups = tuple(getattr(status, "evidence_groups", ()))
        if readiness != "available" or spec is None or not groups:
            raise DeploymentCapabilityError(
                "native binding status has no AVAILABLE retained device groups"
            )
        capabilities = []
        for evidence in groups:
            capabilities.append(
                cls(
                    function=str(spec.name),
                    mode=ControlMode.parse(spec.mode),
                    devices=tuple(evidence.devices),
                    topology_fingerprint=str(evidence.topology_fingerprint),
                    level=CapabilityLevel.AVAILABLE,
                )
            )
        return tuple(capabilities)


@dataclass(frozen=True)
class MeasuredDeployment:
    """Validated device and link measurements supplied by the caller."""

    devices: Tuple[DeviceProfile, ...]
    topology: TopologyProfile
    topology_fingerprint: str

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("a measured deployment requires at least one device")
        ids = tuple(device.device_id for device in self.devices)
        if tuple(sorted(set(ids))) != ids:
            raise ValueError("measured deployment devices must be sorted and unique")
        if not self.topology_fingerprint:
            raise ValueError("measured deployment requires a topology fingerprint")

    @classmethod
    def from_measurements(
        cls,
        hardware: Mapping[str, Any],
        transport: Mapping[str, Any],
        *,
        devices: Optional[Sequence[int]] = None,
        topology_fingerprint: Optional[str] = None,
        reserved_memory_bytes: Any = 0,
        compute_scales: Optional[Mapping[int, float]] = None,
    ) -> "MeasuredDeployment":
        """Build profiles from parsed ``results/hardware`` JSON objects.

        No path is accepted here deliberately.  The application decides which
        trusted records to parse and supplies their mappings explicitly.
        """

        if not isinstance(hardware, Mapping) or not isinstance(transport, Mapping):
            raise TypeError("hardware and transport measurements must be mappings")
        raw_devices = hardware.get("gpus", hardware.get("devices"))
        if not isinstance(raw_devices, Sequence) or isinstance(raw_devices, (str, bytes)):
            raise ValueError("hardware measurement requires a device list")
        records = {}
        for raw in raw_devices:
            if not isinstance(raw, Mapping):
                raise ValueError("each hardware device record must be a mapping")
            device_id = int(raw.get("index", raw.get("device_id", -1)))
            if device_id in records or device_id < 0:
                raise ValueError("hardware device IDs must be unique and non-negative")
            if "total_global_memory" in raw:
                memory_bytes = int(raw["total_global_memory"])
            elif "memory_total_mib" in raw:
                memory_bytes = int(raw["memory_total_mib"]) * (1 << 20)
            else:
                raise ValueError("hardware device record has no memory capacity")
            if memory_bytes <= 0:
                raise ValueError("device memory capacity must be positive")
            records[device_id] = (raw, memory_bytes)

        requested = tuple(records) if devices is None else tuple(int(v) for v in devices)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("selected devices must be non-empty and unique")
        selected = tuple(sorted(requested))
        unknown = set(selected) - set(records)
        if unknown:
            raise ValueError("hardware measurement lacks devices {}".format(sorted(unknown)))

        benchmark = transport.get("benchmark", transport)
        if not isinstance(benchmark, Mapping):
            raise ValueError("transport measurement has no benchmark object")
        host_links = benchmark.get("host_links", ())
        peer_links = benchmark.get("peer_links", ())
        host = {}
        for raw in host_links:
            if not isinstance(raw, Mapping):
                raise ValueError("host transport record must be a mapping")
            device_id = int(raw["device_id"])
            h2d = float(raw["host_to_device_gbps"]) * 1.0e9
            d2h = float(raw["device_to_host_gbps"]) * 1.0e9
            if not math.isfinite(h2d) or not math.isfinite(d2h) or min(h2d, d2h) <= 0:
                raise ValueError("host transport bandwidths must be finite and positive")
            host[device_id] = (h2d, d2h)
        missing_host = set(selected) - set(host)
        if missing_host:
            raise ValueError(
                "transport measurement lacks host links for devices {}".format(
                    sorted(missing_host)
                )
            )

        directional_bandwidth = {}
        peer_latencies = []
        peer_support = {}
        for raw in peer_links:
            if not isinstance(raw, Mapping):
                raise ValueError("peer transport record must be a mapping")
            left = int(raw["from_device"])
            right = int(raw["to_device"])
            supported = bool(raw.get("supported", False))
            peer_support[(left, right)] = supported
            if not supported or left not in selected or right not in selected:
                continue
            bandwidth = float(raw["bandwidth_gbps"]) * 1.0e9
            latency = float(raw["latency_microseconds"]) * 1.0e-6
            if (
                not math.isfinite(bandwidth)
                or bandwidth <= 0
                or not math.isfinite(latency)
                or latency < 0
            ):
                raise ValueError("peer transport measurements must be finite")
            directional_bandwidth[(left, right)] = bandwidth
            peer_latencies.append(latency)

        peer_bandwidth = {}
        for left_index, left in enumerate(selected):
            for right in selected[left_index + 1 :]:
                forward = directional_bandwidth.get((left, right))
                reverse = directional_bandwidth.get((right, left))
                if forward is not None and reverse is not None:
                    peer_bandwidth[(left, right)] = min(forward, reverse)

        def reserved(device_id: int) -> int:
            if isinstance(reserved_memory_bytes, Mapping):
                return int(reserved_memory_bytes.get(device_id, 0))
            return int(reserved_memory_bytes)

        scales = {} if compute_scales is None else dict(compute_scales)
        profiles = tuple(
            DeviceProfile(
                device_id=device_id,
                memory_bytes=records[device_id][1],
                reserved_bytes=reserved(device_id),
                compute_scale=float(scales.get(device_id, 1.0)),
                h2d_bandwidth_bytes_per_second=host[device_id][0],
                d2h_bandwidth_bytes_per_second=host[device_id][1],
            )
            for device_id in selected
        )
        # Staging performs a D2H and H2D leg.  Their harmonic throughput is a
        # conservative measured fallback for the complete transfer path.
        staging = min(
            1.0 / (1.0 / host[device][0] + 1.0 / host[device][1])
            for device in selected
        )
        collective_gbps = benchmark.get("collective_bandwidth_gbps")
        topology = TopologyProfile(
            peer_bandwidth_bytes_per_second=peer_bandwidth,
            host_staging_bandwidth_bytes_per_second=staging,
            collective_bandwidth_bytes_per_second=(
                None if collective_gbps is None else float(collective_gbps) * 1.0e9
            ),
            message_latency_seconds=max(peer_latencies, default=5.0e-6),
        )
        fingerprint = topology_fingerprint or _measurement_fingerprint(
            records, selected, peer_support
        )
        return cls(profiles, topology, str(fingerprint))


@dataclass(frozen=True)
class FunctionCostHints:
    """Analytical sizing priors, not measured performance coefficients."""

    graph_vertex_bytes: int = 8
    graph_adjacency_bytes: int = 4
    weighted_adjacency_extra_bytes: int = 8
    workspace_vertex_bytes: int = 16
    workspace_adjacency_bytes: int = 0
    output_item_bytes: int = 8
    seconds_per_work_unit: float = 1.0e-8
    state_seconds_per_work_unit: float = 0.0
    communication_bytes_per_adjacency: float = 0.0
    communication_rounds: int = 0
    partition_overhead_fraction: float = 0.10
    feature_units_per_adjacency: float = 1.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.graph_vertex_bytes,
            self.graph_adjacency_bytes,
            self.weighted_adjacency_extra_bytes,
            self.workspace_vertex_bytes,
            self.workspace_adjacency_bytes,
            self.output_item_bytes,
            self.communication_rounds,
        )
        if any(int(value) < 0 for value in integer_fields):
            raise ValueError("function cost byte and round hints must be non-negative")
        float_fields = (
            self.seconds_per_work_unit,
            self.state_seconds_per_work_unit,
            self.communication_bytes_per_adjacency,
            self.partition_overhead_fraction,
            self.feature_units_per_adjacency,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in float_fields):
            raise ValueError("function cost hints must be finite and non-negative")
        if self.feature_units_per_adjacency == 0:
            raise ValueError("feature_units_per_adjacency must be positive")


class WorkloadEstimator:
    """Derive a planner workload from immutable snapshot and call metadata."""

    def __init__(
        self,
        hints: Optional[Mapping[str, FunctionCostHints]] = None,
        default_hints: Optional[FunctionCostHints] = None,
    ) -> None:
        self._hints = {
            str(name).strip().lower(): value for name, value in (hints or {}).items()
        }
        if any(not isinstance(value, FunctionCostHints) for value in self._hints.values()):
            raise TypeError("workload hints must contain FunctionCostHints values")
        self._default = FunctionCostHints() if default_hints is None else default_hints
        if not isinstance(self._default, FunctionCostHints):
            raise TypeError("default_hints must be FunctionCostHints")

    def estimate(
        self,
        function: str,
        snapshot: Any,
        descriptor: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        *,
        concurrent_requests: int,
        resident_devices: FrozenSet[int],
    ) -> WorkloadEstimate:
        hints = self._hints.get(function, self._default)
        vertices = int(snapshot.vertex_count)
        adjacency = int(snapshot.adjacency_entries)
        if min(vertices, adjacency) < 0:
            raise ValueError("snapshot sizes must be non-negative")
        required_view = str(descriptor.required_view).lower()
        weighted = bool(getattr(snapshot, "weight_signature", None)) or (
            "weighted" in required_view and "weighted-optional" not in required_view
        )
        graph_adjacency_bytes = hints.graph_adjacency_bytes + (
            hints.weighted_adjacency_extra_bytes if weighted else 0
        )
        graph_state_bytes = (vertices + 1) * hints.graph_vertex_bytes
        graph_state_bytes += adjacency * graph_adjacency_bytes
        workspace_bytes = vertices * hints.workspace_vertex_bytes
        workspace_bytes += adjacency * hints.workspace_adjacency_bytes

        source_count = _request_cardinality(
            args, kwargs, ("sources", "source", "roots", "root"), default=1
        )
        output_count = _request_cardinality(
            args, kwargs, ("nodes", "nodelist", "output_nodes"), default=vertices
        )
        positional_source, positional_output = _positional_cardinalities(
            function, args
        )
        if positional_source is not None and not any(
            name in kwargs for name in ("sources", "source", "roots", "root")
        ):
            source_count = positional_source
        if positional_output is not None and not any(
            name in kwargs for name in ("nodes", "nodelist", "output_nodes")
        ):
            output_count = positional_output
        if descriptor.decomposition.get(
            ControlMode.REPLICATED_TASK_PARALLEL
        ) is DecompositionAxis.SOURCE:
            work_multiplier = max(source_count, 1)
        elif descriptor.decomposition.get(
            ControlMode.REPLICATED_TASK_PARALLEL
        ) is DecompositionAxis.OUTPUT_VERTEX:
            work_multiplier = output_count / max(vertices, 1)
        else:
            work_multiplier = 1.0
        work_units = max(vertices + adjacency, 1) * max(work_multiplier, 1.0 / max(vertices, 1))
        calibration_scale = max(
            adjacency * max(work_multiplier, 1.0 / max(vertices, 1))
            * hints.feature_units_per_adjacency,
            1.0,
        )
        return WorkloadEstimate(
            function=function,
            snapshot=snapshot,
            graph_state_bytes=int(graph_state_bytes),
            workspace_bytes=int(math.ceil(workspace_bytes * max(work_multiplier, 1.0))),
            output_bytes=int(max(output_count, 0) * hints.output_item_bytes),
            single_compute_seconds=work_units * hints.seconds_per_work_unit,
            communication_bytes=int(
                math.ceil(adjacency * hints.communication_bytes_per_adjacency)
            ),
            communication_rounds=hints.communication_rounds,
            state_preparation_seconds=work_units * hints.state_seconds_per_work_unit,
            partition_overhead_fraction=hints.partition_overhead_fraction,
            concurrent_requests=int(concurrent_requests),
            resident_devices=resident_devices,
            calibration_scale=calibration_scale,
        )


@dataclass(frozen=True)
class DeploymentDecision:
    """Runtime-ready decision plus its auditable control-plane plan."""

    function: str
    mode: ExecutionMode
    devices: Tuple[int, ...]
    service_device_group: Tuple[int, ...]
    control_plan: ExecutionPlan
    planning_seconds: float

    @property
    def estimated_seconds(self) -> float:
        return self.control_plan.cost.total

    def summary(self) -> Mapping[str, Any]:
        return {
            "function": self.function,
            "mode": self.mode.value,
            "devices": self.devices,
            "service_device_group": self.service_device_group,
            "estimated_seconds": self.estimated_seconds,
            "planning_seconds": self.planning_seconds,
            "reason": self.control_plan.reason,
            "explicit": self.control_plan.explicit,
        }


class DeploymentPlanner:
    """Thread-safe production bridge around :class:`AdaptivePlanner`.

    A residency provider returns the current complete set of resident devices,
    including an empty set after eviction, or None when residency is unknown.
    Placement history is used only when no current residency answer exists.
    """

    def __init__(
        self,
        deployment: MeasuredDeployment,
        *,
        calibration: Optional[CalibrationProfile] = None,
        cooperative_capabilities: Iterable[CooperativeCapability] = (),
        workload_estimator: Optional[WorkloadEstimator] = None,
        residency_provider: Optional[
            Callable[[Any, str, Sequence[Any], Mapping[str, Any]], Optional[Iterable[int]]]
        ] = None,
    ) -> None:
        if not isinstance(deployment, MeasuredDeployment):
            raise TypeError("deployment must be a MeasuredDeployment")
        if calibration is not None and not isinstance(calibration, CalibrationProfile):
            raise TypeError("calibration must be a CalibrationProfile")
        if residency_provider is not None and not callable(residency_provider):
            raise TypeError("residency_provider must be callable")
        self.deployment = deployment
        self.calibration = calibration
        self.workload_estimator = (
            WorkloadEstimator() if workload_estimator is None else workload_estimator
        )
        if not isinstance(self.workload_estimator, WorkloadEstimator):
            raise TypeError("workload_estimator must be a WorkloadEstimator")
        self._residency_provider = residency_provider
        self._capabilities = tuple(cooperative_capabilities)
        if any(not isinstance(item, CooperativeCapability) for item in self._capabilities):
            raise TypeError("cooperative_capabilities must contain CooperativeCapability")
        known = {device.device_id for device in deployment.devices}
        for item in self._capabilities:
            if not set(item.devices).issubset(known):
                raise ValueError("cooperative capability references an unknown device")
        self._stats_lock = threading.Lock()
        self._decision_count = 0
        self._failure_count = 0
        self._decision_seconds = 0.0
        self._maximum_decision_seconds = 0.0
        self._mode_counts: Dict[str, int] = {}
        self._last_decision = None
        self._candidate_evaluation_count = 0
        self._candidate_failure_count = 0
        self._candidate_evaluation_seconds = 0.0
        self._maximum_candidate_evaluation_seconds = 0.0

    def plan_request(
        self,
        function: Callable[..., Any],
        snapshot: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        *,
        requested_mode: ExecutionMode,
        requested_devices: Optional[Tuple[int, ...]],
        registered_modes: Iterable[ExecutionMode],
        eligible_device_groups: Optional[
            Mapping[ExecutionMode, Iterable[Tuple[int, ...]]]
        ] = None,
        queue_seconds: Optional[Mapping[int, float]] = None,
        resident_devices_hint: Iterable[int] = (),
        concurrent_requests: int = 1,
        minimum_multigpu_gain: float = 0.05,
    ) -> DeploymentDecision:
        return self._execute_planning_call(
            function,
            snapshot,
            args,
            kwargs,
            requested_mode=requested_mode,
            requested_devices=requested_devices,
            registered_modes=registered_modes,
            eligible_device_groups=eligible_device_groups,
            queue_seconds=queue_seconds,
            resident_devices_hint=resident_devices_hint,
            concurrent_requests=concurrent_requests,
            minimum_multigpu_gain=minimum_multigpu_gain,
            candidate_evaluation=False,
        )

    def evaluate_candidate(
        self,
        function: Callable[..., Any],
        snapshot: Any,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        *,
        requested_mode: ExecutionMode,
        requested_devices: Optional[Tuple[int, ...]],
        registered_modes: Iterable[ExecutionMode],
        eligible_device_groups: Optional[
            Mapping[ExecutionMode, Iterable[Tuple[int, ...]]]
        ] = None,
        queue_seconds: Optional[Mapping[int, float]] = None,
        resident_devices_hint: Iterable[int] = (),
        concurrent_requests: int = 1,
        minimum_multigpu_gain: float = 0.05,
    ) -> DeploymentDecision:
        """Evaluate one exact scheduler candidate without recording a decision.

        Request-placement controls compare every evidence-authorized singleton
        group before selecting one.  Keeping these feasibility/cost probes
        separate prevents planner ``decision_count`` from overstating the
        number of submitted requests.
        """

        return self._execute_planning_call(
            function,
            snapshot,
            args,
            kwargs,
            requested_mode=requested_mode,
            requested_devices=requested_devices,
            registered_modes=registered_modes,
            eligible_device_groups=eligible_device_groups,
            queue_seconds=queue_seconds,
            resident_devices_hint=resident_devices_hint,
            concurrent_requests=concurrent_requests,
            minimum_multigpu_gain=minimum_multigpu_gain,
            candidate_evaluation=True,
        )

    def _execute_planning_call(
        self,
        function,
        snapshot,
        args,
        kwargs,
        *,
        requested_mode,
        requested_devices,
        registered_modes,
        eligible_device_groups,
        queue_seconds,
        resident_devices_hint,
        concurrent_requests,
        minimum_multigpu_gain,
        candidate_evaluation,
    ):
        begin = time.perf_counter()
        try:
            decision = self._plan_request(
                function,
                snapshot,
                tuple(args),
                dict(kwargs),
                requested_mode=ExecutionMode.parse(requested_mode),
                requested_devices=requested_devices,
                registered_modes=tuple(registered_modes),
                eligible_device_groups=eligible_device_groups,
                queue_seconds={} if queue_seconds is None else queue_seconds,
                resident_devices_hint=tuple(resident_devices_hint),
                concurrent_requests=int(concurrent_requests),
                minimum_multigpu_gain=float(minimum_multigpu_gain),
                begin=begin,
            )
        except BaseException:
            elapsed = time.perf_counter() - begin
            with self._stats_lock:
                if candidate_evaluation:
                    self._candidate_failure_count += 1
                    self._candidate_evaluation_seconds += elapsed
                    self._maximum_candidate_evaluation_seconds = max(
                        self._maximum_candidate_evaluation_seconds, elapsed
                    )
                else:
                    self._failure_count += 1
                    self._decision_seconds += elapsed
                    self._maximum_decision_seconds = max(
                        self._maximum_decision_seconds, elapsed
                    )
            raise
        with self._stats_lock:
            if candidate_evaluation:
                self._candidate_evaluation_count += 1
                self._candidate_evaluation_seconds += decision.planning_seconds
                self._maximum_candidate_evaluation_seconds = max(
                    self._maximum_candidate_evaluation_seconds,
                    decision.planning_seconds,
                )
            else:
                self._decision_count += 1
                self._decision_seconds += decision.planning_seconds
                self._maximum_decision_seconds = max(
                    self._maximum_decision_seconds, decision.planning_seconds
                )
                self._mode_counts[decision.mode.value] = (
                    self._mode_counts.get(decision.mode.value, 0) + 1
                )
                self._last_decision = dict(decision.summary())
        return decision

    def _plan_request(
        self,
        function,
        snapshot,
        args,
        kwargs,
        *,
        requested_mode,
        requested_devices,
        registered_modes,
        eligible_device_groups,
        queue_seconds,
        resident_devices_hint,
        concurrent_requests,
        minimum_multigpu_gain,
        begin,
    ):
        if snapshot is None:
            raise NoFeasiblePlanError("adaptive graph planning requires a graph snapshot")
        if concurrent_requests < 1:
            raise ValueError("concurrent_requests must be at least one")
        if not 0 <= minimum_multigpu_gain < 1:
            raise ValueError("minimum_multigpu_gain must be in [0, 1)")
        registry = default_capability_registry()
        raw_name = getattr(function, "__name__", str(function)).strip().lower()
        descriptor = registry.descriptor(raw_name)
        canonical = descriptor.name.lower()
        registered = frozenset(ControlMode.parse(mode) for mode in registered_modes)
        executor_groups = self._normalize_executor_device_groups(
            eligible_device_groups, registered
        )
        requested_control = ControlMode.parse(requested_mode)
        selected_devices = (
            None
            if requested_devices is None
            else tuple(int(device) for device in requested_devices)
        )
        known = {device.device_id for device in self.deployment.devices}
        if selected_devices is not None:
            if not selected_devices or tuple(sorted(set(selected_devices))) != selected_devices:
                raise ValueError("requested devices must be sorted, unique, and non-empty")
            unknown = set(selected_devices) - known
            if unknown:
                raise ValueError("requested devices are outside the deployment")

        resident = frozenset(int(device) for device in resident_devices_hint)
        if not resident.issubset(known):
            raise ValueError("resident device hint contains an unknown device")
        if self._residency_provider is not None:
            measured = self._residency_provider(
                snapshot, canonical, args, kwargs
            )
            if measured is not None:
                measured_resident = frozenset(int(device) for device in measured)
                if not measured_resident.issubset(known):
                    raise ValueError("residency provider returned an unknown device")
                resident = measured_resident
        work = self.workload_estimator.estimate(
            canonical,
            snapshot,
            descriptor,
            args,
            kwargs,
            concurrent_requests=concurrent_requests,
            resident_devices=resident,
        )

        available_groups = self._eligible_cooperative_groups(
            canonical, registered, work.calibration_scale, executor_groups
        )
        # Start from the declared descriptor but expose only routes that are
        # both registered in this runtime and currently evidence-backed.
        for mode in descriptor.capabilities:
            registry.set_level(canonical, mode, CapabilityLevel.UNAVAILABLE)
        for mode in registered:
            if mode in (ControlMode.SINGLE_DEVICE, ControlMode.REQUEST_PARALLEL):
                if (
                    mode in descriptor.capabilities
                    and (mode not in executor_groups or executor_groups[mode])
                ):
                    registry.set_level(canonical, mode, CapabilityLevel.AVAILABLE)
            elif available_groups.get((canonical, mode)):
                registry.set_level(canonical, mode, CapabilityLevel.AVAILABLE)

        # A multi-device explicit group cannot be silently reinterpreted as a
        # one-device request-parallel execution.
        if (
            requested_control is ControlMode.AUTO
            and selected_devices is not None
            and len(selected_devices) > 1
            and ControlMode.REQUEST_PARALLEL in descriptor.capabilities
        ):
            registry.set_level(
                canonical, ControlMode.REQUEST_PARALLEL, CapabilityLevel.UNAVAILABLE
            )

        profiles = tuple(
            replace(
                device,
                queue_seconds=device.queue_seconds
                + _nonnegative_queue(queue_seconds.get(device.device_id, 0.0)),
            )
            for device in self.deployment.devices
        )
        model = AnalyticalCostModel(
            CostModelConfig(minimum_multigpu_gain=minimum_multigpu_gain)
        )
        calibration = (
            self.calibration
            if self.calibration is not None
            and self.calibration.topology_fingerprint
            == self.deployment.topology_fingerprint
            else None
        )
        planner_groups = dict(available_groups)
        single_groups = executor_groups.get(ControlMode.SINGLE_DEVICE)
        if single_groups is not None:
            planner_groups[(canonical, ControlMode.SINGLE_DEVICE)] = single_groups
        request_groups = executor_groups.get(ControlMode.REQUEST_PARALLEL)
        if request_groups is not None:
            request_devices = tuple(group[0] for group in request_groups)
            maximum = min(len(request_devices), concurrent_requests)
            planner_groups[(canonical, ControlMode.REQUEST_PARALLEL)] = tuple(
                group
                for count in range(2, maximum + 1)
                for group in itertools.combinations(request_devices, count)
            )
        adaptive = AdaptivePlanner(
            profiles,
            self.deployment.topology,
            registry,
            model,
            calibration=calibration,
            topology_fingerprint=self.deployment.topology_fingerprint,
            available_device_groups=planner_groups,
        )

        if requested_control is ControlMode.REQUEST_PARALLEL:
            plan = self._plan_one_device_request(
                canonical,
                selected_devices,
                profiles,
                registry,
                model,
                work,
                request_groups,
            )
        elif (
            requested_control is ControlMode.AUTO
            and selected_devices is not None
            and len(selected_devices) == 1
            and ControlMode.SINGLE_DEVICE not in registered
            and ControlMode.REQUEST_PARALLEL in registered
        ):
            plan = self._plan_one_device_request(
                canonical,
                selected_devices,
                profiles,
                registry,
                model,
                work,
                request_groups,
            )
        else:
            override = PlanOverride(
                mode=(
                    None
                    if requested_control is ControlMode.AUTO
                    else requested_control
                ),
                devices=selected_devices,
            )
            try:
                plan = adaptive.plan(work, override)
            except NoFeasiblePlanError:
                # REQUEST_PARALLEL remains a valid one-device execution when a
                # single call arrives and no separately named SINGLE route is
                # registered.  It does not gain fictitious multi-GPU capacity.
                if (
                    requested_control is ControlMode.AUTO
                    and selected_devices is None
                    and ControlMode.REQUEST_PARALLEL in registered
                ):
                    plan = self._plan_one_device_request(
                        canonical,
                        None,
                        profiles,
                        registry,
                        model,
                        work,
                        request_groups,
                    )
                else:
                    raise

        service_group = plan.devices
        if plan.mode is ControlMode.REQUEST_PARALLEL:
            one_request = replace(work, concurrent_requests=1)
            actual_devices = (
                min(
                    plan.devices,
                    key=lambda device_id: (
                        model.estimate(
                            ControlMode.REQUEST_PARALLEL,
                            tuple(
                                device
                                for device in profiles
                                if device.device_id == device_id
                            ),
                            self.deployment.topology,
                            one_request,
                        ).total,
                        device_id,
                    ),
                ),
            )
        else:
            actual_devices = plan.devices
        runtime_mode = ExecutionMode(plan.mode.value)
        elapsed = time.perf_counter() - begin
        return DeploymentDecision(
            function=canonical,
            mode=runtime_mode,
            devices=actual_devices,
            service_device_group=service_group,
            control_plan=plan,
            planning_seconds=elapsed,
        )

    @staticmethod
    def affinity_signature(function, snapshot, args=(), kwargs=None):
        """Return the graph/view identity used by controller placement history.

        The signature contains only immutable snapshot metadata and the
        descriptor's required graph view.  It is a placement hint, not a claim
        that a native cache entry is still resident.
        """

        del args, kwargs
        registry = default_capability_registry()
        raw_name = getattr(function, "__name__", str(function)).strip().lower()
        descriptor = registry.descriptor(raw_name)
        graph_key = (
            str(getattr(snapshot, "graph_id", "")),
            int(getattr(snapshot, "generation", 0)),
            str(getattr(snapshot, "topology_signature", "")),
            str(getattr(snapshot, "weight_signature", "")),
        )
        view_key = str(descriptor.required_view).strip().lower()
        return graph_key, view_key

    @staticmethod
    def _normalize_executor_device_groups(value, registered):
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("eligible_device_groups must be a mode-to-groups mapping")
        result = {}
        for raw_mode, raw_groups in value.items():
            mode = ControlMode.parse(raw_mode)
            if mode not in registered:
                raise ValueError(
                    "eligible device groups reference an unregistered mode"
                )
            groups = []
            for raw_group in raw_groups:
                group = tuple(int(device) for device in raw_group)
                if not group or tuple(sorted(set(group))) != group:
                    raise ValueError(
                        "eligible device groups must be non-empty, sorted, and unique"
                    )
                cooperative = mode in COOPERATIVE_MODES
                if cooperative and len(group) < 2:
                    raise ValueError(
                        "cooperative eligible groups require at least two devices"
                    )
                if not cooperative and len(group) != 1:
                    raise ValueError(
                        "single/request eligible groups require exactly one device"
                    )
                groups.append(group)
            if len(set(groups)) != len(groups):
                raise ValueError("eligible device groups contain duplicates")
            result[mode] = tuple(
                sorted(groups, key=lambda group: (len(group), group))
            )
        return result

    def _eligible_cooperative_groups(
        self, function, registered, scale, executor_groups
    ):
        result = {}
        calibration = self.calibration
        if (
            calibration is None
            or calibration.topology_fingerprint
            != self.deployment.topology_fingerprint
        ):
            return result
        for evidence in self._capabilities:
            if (
                evidence.function != function
                or evidence.mode not in registered
                or evidence.level is not CapabilityLevel.AVAILABLE
                or evidence.topology_fingerprint
                != self.deployment.topology_fingerprint
            ):
                continue
            permitted = executor_groups.get(evidence.mode)
            if permitted is not None and evidence.devices not in permitted:
                continue
            if not calibration.sufficient(
                function,
                ControlMode.SINGLE_DEVICE,
                1,
                topology_fingerprint=self.deployment.topology_fingerprint,
            ) or not calibration.sufficient(
                function,
                evidence.mode,
                len(evidence.devices),
                topology_fingerprint=self.deployment.topology_fingerprint,
            ):
                continue
            coefficient = calibration.coefficient(
                function, evidence.mode, len(evidence.devices)
            )
            if (
                coefficient is None
                or coefficient.break_even_scale is None
                or scale < coefficient.break_even_scale
            ):
                continue
            result.setdefault((function, evidence.mode), []).append(evidence.devices)
        return {
            key: tuple(sorted(set(groups), key=lambda group: (len(group), group)))
            for key, groups in result.items()
        }

    @staticmethod
    def _plan_one_device_request(
        function,
        requested_devices,
        profiles,
        registry,
        model,
        work,
        eligible_groups=None,
    ):
        if registry.level(function, ControlMode.REQUEST_PARALLEL) is not CapabilityLevel.AVAILABLE:
            raise DeploymentCapabilityError(
                "{} has no available request-parallel native route".format(function)
            )
        if requested_devices is not None and len(requested_devices) != 1:
            raise NoFeasiblePlanError(
                "request_parallel requires exactly one execution device per call"
            )
        allowed = (
            tuple(device for device in profiles if device.device_id in requested_devices)
            if requested_devices is not None
            else profiles
        )
        if eligible_groups is not None:
            eligible_devices = {group[0] for group in eligible_groups}
            allowed = tuple(
                device for device in allowed if device.device_id in eligible_devices
            )
        single_work = replace(work, concurrent_requests=1)
        candidates = []
        for device in allowed:
            group = (device.device_id,)
            required = model.required_memory(
                ControlMode.REQUEST_PARALLEL, group, single_work
            )
            if required[device.device_id] > device.usable_memory_bytes:
                continue
            candidates.append(
                ExecutionPlan(
                    function=function,
                    mode=ControlMode.REQUEST_PARALLEL,
                    devices=group,
                    capability=CapabilityLevel.AVAILABLE,
                    cost=model.estimate(
                        ControlMode.REQUEST_PARALLEL,
                        (device,),
                        # Request execution has no inter-device communication.
                        TopologyProfile.uniform(group, 1.0),
                        single_work,
                    ),
                    required_memory_bytes=required,
                    reason=(
                        "explicit request-parallel override preserved"
                        if requested_devices is not None
                        else "safe one-device request-parallel path"
                    ),
                    explicit=requested_devices is not None,
                )
            )
        if not candidates:
            raise NoFeasiblePlanError(
                "no single device can fit the request-parallel graph state"
            )
        return min(candidates, key=lambda plan: (plan.cost.total, plan.devices))

    def stats(self) -> Mapping[str, Any]:
        with self._stats_lock:
            total = self._decision_seconds
            attempts = self._decision_count + self._failure_count
            candidate_attempts = (
                self._candidate_evaluation_count + self._candidate_failure_count
            )
            return {
                "decision_count": self._decision_count,
                "failure_count": self._failure_count,
                "total_decision_seconds": total,
                "mean_decision_seconds": 0.0 if attempts == 0 else total / attempts,
                "maximum_decision_seconds": self._maximum_decision_seconds,
                "mode_counts": dict(self._mode_counts),
                "last_decision": (
                    None if self._last_decision is None else dict(self._last_decision)
                ),
                "candidate_evaluation_count": self._candidate_evaluation_count,
                "candidate_failure_count": self._candidate_failure_count,
                "total_candidate_evaluation_seconds": (
                    self._candidate_evaluation_seconds
                ),
                "mean_candidate_evaluation_seconds": (
                    0.0
                    if candidate_attempts == 0
                    else self._candidate_evaluation_seconds / candidate_attempts
                ),
                "maximum_candidate_evaluation_seconds": (
                    self._maximum_candidate_evaluation_seconds
                ),
                "topology_fingerprint": self.deployment.topology_fingerprint,
                "calibration_topology_match": bool(
                    self.calibration is not None
                    and self.calibration.topology_fingerprint
                    == self.deployment.topology_fingerprint
                ),
            }


def _nonnegative_queue(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("queue estimates must be finite and non-negative")
    return result


def _request_cardinality(args, kwargs, names, default):
    for name in names:
        if name not in kwargs or kwargs[name] is None:
            continue
        value = kwargs[name]
        if isinstance(value, (str, bytes)):
            return 1
        try:
            return len(value)
        except TypeError:
            return 1
    # Positional arguments are intentionally not guessed: public functions
    # use incompatible positional layouts.  Callers that need a source/output
    # cardinality in planning can pass the named public parameter.
    return int(default)


def _value_cardinality(value):
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return 1
    try:
        return len(value)
    except TypeError:
        return 1


def _positional_cardinalities(function, args):
    """Read only positions fixed by the existing EasyGraph public signatures."""

    # ``args[0]`` is the graph.  The remaining positions below are stable in
    # the inherited API and avoid signature inspection or function execution.
    if function in {"sssp", "dijkstra", "bellman_ford"} and len(args) > 1:
        return _value_cardinality(args[1]), None
    if function == "bc" and len(args) > 2:
        return _value_cardinality(args[2]), None
    if function == "closeness" and len(args) > 2:
        return _value_cardinality(args[2]), None
    if function in {"effective_size", "efficiency", "constraint", "hierarchy"}:
        if len(args) > 1:
            return None, _value_cardinality(args[1])
    return None, None


def _measurement_fingerprint(records, selected, peer_support):
    devices = []
    for device_id in selected:
        raw, memory_bytes = records[device_id]
        devices.append(
            {
                "device_id": device_id,
                "uuid": str(raw.get("uuid", "")),
                "name": str(raw.get("name", "")),
                "pci_bus_id": str(raw.get("pci_bus_id", "")),
                "compute_capability": str(raw.get("compute_capability", "")),
                "total_global_memory": memory_bytes,
            }
        )
    peer = [
        [left == right or bool(peer_support.get((left, right), False)) for right in selected]
        for left in selected
    ]
    payload = {"devices": devices, "peer_access": peer}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CooperativeCapability",
    "DeploymentCapabilityError",
    "DeploymentDecision",
    "DeploymentPlanner",
    "DeploymentPlanningError",
    "FunctionCostHints",
    "MeasuredDeployment",
    "WorkloadEstimator",
]
