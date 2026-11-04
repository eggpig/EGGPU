"""Production scheduler backed by the process-global C++ worker runtime.

This scheduler owns no Python device-worker pool.  Python controller threads
perform argument parsing, snapshot checks, and result materialization.  A
registered native executor must translate those arguments into a pybind call
whose C++ side submits only native value types and shared native buffers to
``RuntimeManager``.  Generic Python callbacks are never submitted to native
device workers.
"""

import concurrent.futures
import inspect
import itertools
import math
import threading
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass

from .compatibility import ExecutorCompatibilityError
from .compatibility import evaluate_executor_compatibility
from .compatibility import is_production_executor
from .deployment_planner import DeploymentPlanningError
from .deployment_planner import NoFeasiblePlanError
from .native_adapter import native_adapter
from .runtime import ConfigurationError
from .runtime import EGMGPUFuture
from .runtime import RuntimeClosedError
from .runtime import UnsupportedExecutionModeError
from .runtime import WorkflowExecutionError
from .runtime import _GangReservation
from .snapshot import GraphSnapshotRegistry
from .snapshot import GraphSnapshotLease
from .snapshot import NativePayloadUnavailableError
from .types import ExecutionMode
from .types import NativeCallContext
from .types import RequestSchedulingPolicy
from .types import RuntimeConfig


_MISSING_EXECUTOR_DEVICE_GROUPS = object()


@dataclass(frozen=True)
class _PolicyPlacement:
    mode: ExecutionMode
    devices: tuple
    executor: object
    predicted_service_seconds: float
    predicted_queue_seconds: float
    maximum_candidate_queue_seconds: float
    graph_affinity: bool
    view_affinity: bool
    planner_reason: str


@dataclass(frozen=True)
class _PlacementHistory:
    graph_key: tuple
    view_key: str


def _executor_device_groups(executor, mode):
    """Return one executor's exact evidence-authorized device groups.

    Public API registration wraps executors to guard direct-dispatch reentry.
    Follow that wrapper's explicit original-executor link so evidence metadata
    is not accidentally erased by the guard.
    """

    current = executor
    seen = set()
    raw = _MISSING_EXECUTOR_DEVICE_GROUPS
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        raw = getattr(
            current, "eligible_device_groups", _MISSING_EXECUTOR_DEVICE_GROUPS
        )
        if raw is not _MISSING_EXECUTOR_DEVICE_GROUPS:
            break
        current = getattr(current, "_egmgpu_original_executor", None)
    if raw is _MISSING_EXECUTOR_DEVICE_GROUPS:
        # Custom executors predate evidence-scoped scheduling and remain
        # unrestricted.  Production evidence-backed adapters expose metadata.
        return None

    parsed_mode = ExecutionMode.parse(mode)
    groups = []
    try:
        candidates = tuple(raw)
    except TypeError as exc:
        raise ConfigurationError(
            "native executor eligible_device_groups must be iterable"
        ) from exc
    for candidate in candidates:
        try:
            group = tuple(int(device) for device in candidate)
        except TypeError as exc:
            raise ConfigurationError(
                "native executor device groups must contain iterables"
            ) from exc
        if not group or tuple(sorted(set(group))) != group:
            raise ConfigurationError(
                "native executor device groups must be non-empty, sorted, and unique"
            )
        cooperative = parsed_mode in {
            ExecutionMode.REPLICATED,
            ExecutionMode.PARTITIONED,
        }
        if cooperative and len(group) < 2:
            raise ConfigurationError(
                "cooperative executor device groups require at least two devices"
            )
        if not cooperative and len(group) != 1:
            raise ConfigurationError(
                "single/request executor device groups require exactly one device"
            )
        groups.append(group)
    if len(set(groups)) != len(groups):
        raise ConfigurationError("native executor repeats an eligible device group")
    return tuple(sorted(groups, key=lambda group: (len(group), group)))


class NativeOwnedRuntime:
    """Single-owner Python control plane for native fixed-device workers."""

    worker_owner = "native"

    def __init__(
        self,
        config,
        snapshot_registry=None,
        snapshot_provider=None,
        adapter=None,
        deployment_planner=None,
    ):
        if not isinstance(config, RuntimeConfig):
            raise TypeError("config must be a RuntimeConfig")
        if snapshot_registry is not None and snapshot_provider is not None:
            raise ValueError("provide snapshot_registry or snapshot_provider, not both")
        if deployment_planner is not None and not (
            callable(getattr(deployment_planner, "plan_request", None))
            and callable(getattr(deployment_planner, "stats", None))
        ):
            raise TypeError(
                "deployment_planner must provide plan_request() and stats()"
            )
        if (
            config.scheduling_policy is not RequestSchedulingPolicy.FIFO
            and deployment_planner is None
        ):
            raise ConfigurationError(
                "{} scheduling requires a measured deployment planner".format(
                    config.scheduling_policy.value
                )
            )
        self.config = config
        self._snapshot_registry = (
            GraphSnapshotRegistry(snapshot_provider=snapshot_provider)
            if snapshot_registry is None
            else snapshot_registry
        )
        self._adapter = native_adapter() if adapter is None else adapter
        self._deployment_planner = deployment_planner
        self._joint_window_planner = None
        self._joint_publication_lock = threading.Lock()
        self._owner_token = object()
        self._state_lock = threading.RLock()
        self._accepting = True
        self._request_ids = itertools.count(1)
        self._stats = Counter()
        self._futures = set()
        self._native_executors = {}
        self._planned_device_seconds = Counter()
        self._placement_lock = threading.Lock()
        self._round_robin_cursor = 0
        self._last_placement_by_device = {}
        self._scheduler_mode_counts = Counter()
        self._scheduler_group_counts = Counter()
        self._scheduler_device_counts = Counter()
        self._scheduler_graph_affinity_hits = 0
        self._scheduler_view_affinity_hits = 0
        self._scheduler_queue_seconds_total = 0.0
        self._scheduler_queue_seconds_maximum = 0.0
        self._scheduler_candidate_queue_seconds_maximum = 0.0
        self._scheduler_wait_seconds_total = 0.0
        self._scheduler_wait_seconds_maximum = 0.0
        self._scheduler_wait_samples = 0
        self._scheduler_last_selection = None
        self._untracked_plans = 0
        self._reservations = _GangReservation(config.devices, config.reservation_policy)
        self._controller_local = threading.local()
        self._controllers = None
        claimed = False
        try:
            status = self._adapter.claim(
                self._owner_token, devices=config.devices, backend="auto"
            )
            claimed = True
            active_devices = tuple(int(item) for item in status.get("device_ids", ()))
            if active_devices != config.devices:
                raise ConfigurationError(
                    "native runtime started devices {}, expected {}".format(
                        active_devices, config.devices
                    )
                )
            self._controllers = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(4, 2 * len(config.devices)),
                thread_name_prefix="egmgpu-native-control",
            )
        except BaseException:
            self._accepting = False
            if self._controllers is not None:
                self._controllers.shutdown(wait=True, cancel_futures=True)
            if claimed:
                self._adapter.release(self._owner_token, drain=True)
            raise

    @property
    def closed(self):
        with self._state_lock:
            return not self._accepting

    def register_native_executor(self, function, mode, executor):
        """Register a controller-side adapter for one implemented native path."""

        if not callable(function) or not callable(executor):
            raise TypeError("function and native executor must be callable")
        parsed_mode = ExecutionMode.parse(mode)
        if parsed_mode is ExecutionMode.AUTO:
            raise ValueError("AUTO is a selection policy, not a native executor")
        _executor_device_groups(executor, parsed_mode)
        with self._state_lock:
            if not self._accepting:
                raise RuntimeClosedError("EGMGPU runtime is closed")
            self._native_executors[(function, parsed_mode)] = executor

    def register_native_executors_atomic(self, registrations):
        """Validate and publish a set of native routes in one state update."""

        normalized = []
        keys = set()
        for function, mode, executor in tuple(registrations):
            if not callable(function) or not callable(executor):
                raise TypeError("function and native executor must be callable")
            parsed_mode = ExecutionMode.parse(mode)
            if parsed_mode is ExecutionMode.AUTO:
                raise ValueError("AUTO is a selection policy, not a native executor")
            _executor_device_groups(executor, parsed_mode)
            key = (function, parsed_mode)
            if key in keys:
                raise ValueError("duplicate native executor registration")
            keys.add(key)
            normalized.append((key, executor))
        with self._state_lock:
            if not self._accepting:
                raise RuntimeClosedError("EGMGPU runtime is closed")
            self._native_executors.update(normalized)

    def unregister_native_executor(self, function, mode):
        parsed_mode = ExecutionMode.parse(mode)
        with self._state_lock:
            self._native_executors.pop((function, parsed_mode), None)

    def configure_joint_window(self, profile, *, accounting="union", allow_synthetic=False, observed_binding=None):
        """Explicit isolated experiment; default submit/AUTO policy is unchanged."""
        from .joint_window import WindowDeploymentPlanner
        planner = WindowDeploymentPlanner(profile, self.config.devices,
            accounting=accounting, allow_synthetic=allow_synthetic)
        if not planner.synthetic and observed_binding != {
            'artifact_id': planner.profile['artifact_id'],
            'environment_id': planner.profile['environment_id'],
        }:
            raise ConfigurationError("Observed artifact/environment differs from frozen profile")
        with self._state_lock:
            if not self._accepting or self._stats['submitted'] or self._joint_window_planner is not None:
                raise ConfigurationError("Configure joint window on a fresh accepting runtime")
            self._reservations.configure_joint(planner)
            self._joint_window_planner = planner
        return dict(capacity_scope=planner.scope, synthetic=planner.synthetic,
                    accounting=planner.accounting)

    def submit_window(self, calls):
        """Select routes then use submit -> controller -> registered native executor.

        No graph rewriting is performed. The adapter must consume context.route_id
        and route_layout and use the already imported matching native graph handle.
        """
        planner = self._joint_window_planner
        if planner is None:
            raise ConfigurationError("An independently calibrated joint profile is required")
        calls = tuple(calls)
        if not 1 <= len(calls) <= 4:
            raise ConfigurationError("A finite joint window contains one to four calls")
        if bool(getattr(self._controller_local, "inside_native_executor", False)):
            raise ConfigurationError("Native executors cannot submit joint windows")
        leases, candidates, tokens, futures = [], [], [], []
        try:
            # Full preflight before any resource or reservation publication.
            for call in calls:
                routes = planner.candidates(call)
                if not call.args or self._first_graph_like(call.args, call.kwargs) is not call.args[0]:
                    raise ConfigurationError("Joint calls require their immutable graph as first arg")
                graph = call.args[0]
                lease = self._snapshot_registry.acquire(graph)
                leases.append(lease)
                if not lease.concurrent_execution_safe or not lease.native_payload_ready:
                    raise ConfigurationError("Joint window requires immutable native snapshots")
                planner.validate_call_binding(call, lease.snapshot)
                for route in routes:
                    mode = ExecutionMode.parse(route['mode'])
                    group = self._validate_requested_devices(route['devices'])
                    self._validate_route(mode, group)
                    compatible, rejected = self._compatible_routes(call.function, lease.snapshot,
                        tuple(call.args), dict(call.kwargs), mode)
                    if mode not in compatible:
                        self._raise_incompatible_or_unregistered(call.function, mode, rejected)
                    executor = compatible[mode]
                    self._validate_executor_device_group(executor, mode, group)
                    original = getattr(executor, '_egmgpu_original_executor', executor)
                    if not getattr(original, 'joint_route_aware', False):
                        raise ConfigurationError("Joint executor must explicitly consume route_id/layout")
                candidates.append(routes)
            with self._joint_publication_lock:
                with self._state_lock:
                    if not self._accepting:
                        raise RuntimeClosedError("EGMGPU runtime is closed")
                    tokens, selection = self._reservations.admit_window(candidates)
                by_index = {}
                # Match controller publication order to the atomically queued order;
                # return futures in caller order, regardless of the chosen order.
                for i in selection['order']:
                    call, token = calls[i], tokens[i]
                    kwargs = dict(call.kwargs)
                    kwargs.update(_egmgpu_mode=token.route['mode'],
                                  _egmgpu_devices=tuple(token.route['devices']),
                                  _egmgpu_pinned_snapshot=leases[i], _egmgpu_joint_token=token)
                    f = self.submit(call.function, *call.args, **kwargs)
                    futures.append(f)
                    by_index[i] = f
            return tuple(by_index[i] for i in range(len(calls))), selection
        except BaseException:
            for f in futures:
                f.cancel()
            for token in tokens:
                if not token.published:
                    self._reservations.finish_token(token)
                elif token.future is None:
                    self._reservations.finish_token(token)
            raise
        finally:
            for lease in leases:
                lease.release()

    def submit(self, function, *args, **kwargs):
        if bool(getattr(self._controller_local, "inside_native_executor", False)):
            raise ConfigurationError(
                "a native executor cannot synchronously submit another EGMGPU request"
            )
        with self._state_lock:
            if not self._accepting:
                raise RuntimeClosedError("EGMGPU runtime is closed")
        joint_token = kwargs.pop("_egmgpu_joint_token", None)
        if self._joint_window_planner is not None and joint_token is None:
            raise ConfigurationError("Joint runtime requires submit_window; unprofiled submit rejected")
        if joint_token is not None:
            self._reservations.publish_token(joint_token)
        mode = kwargs.pop("_egmgpu_mode", None)
        devices = kwargs.pop("_egmgpu_devices", None)
        pinned_snapshot = kwargs.pop("_egmgpu_pinned_snapshot", None)
        if not callable(function):
            raise TypeError("function must be callable")
        requested_mode = self.config.mode if mode is None else ExecutionMode.parse(mode)
        selected_devices = self._validate_requested_devices(devices)
        requested_devices = selected_devices

        # An explicit mode and group have unambiguous executor identity.  Reject
        # evidence-ineligible devices before graph snapshot construction or
        # controller/future publication.
        if requested_mode is not ExecutionMode.AUTO and requested_devices is not None:
            explicit_executor = self._executor_for_mode(function, requested_mode)
            self._validate_route(requested_mode, requested_devices)
            self._validate_executor_device_group(
                explicit_executor, requested_mode, requested_devices
            )

        snapshot_lease = None
        graph = self._first_graph_like(args, kwargs)
        if pinned_snapshot is not None:
            if not isinstance(pinned_snapshot, GraphSnapshotLease):
                raise ConfigurationError("internal pinned snapshot is not a lease")
            if graph is None or pinned_snapshot._graph is not graph:
                raise ConfigurationError(
                    "pinned workflow snapshot does not belong to the submitted graph"
                )
            snapshot_lease = self._snapshot_registry.retain(pinned_snapshot)
        elif graph is not None:
            snapshot_lease = self._snapshot_registry.acquire(graph)
        planning_load = None
        planning_load_precharged = False
        reservation_groups = None
        placement_history = None
        submission_started = time.monotonic()
        try:
            snapshot = None if snapshot_lease is None else snapshot_lease.snapshot
            compatible, incompatibilities = self._compatible_routes(
                function,
                snapshot,
                tuple(args),
                dict(kwargs),
                requested_mode,
            )
            if requested_mode is not ExecutionMode.AUTO:
                executor = compatible.get(requested_mode)
                if executor is None:
                    self._raise_incompatible_or_unregistered(
                        function, requested_mode, incompatibilities
                    )
            if requested_mode is ExecutionMode.AUTO and not compatible:
                self._raise_no_auto_route(function, incompatibilities)

            # Explicit research/plugin routes remain executable, but they do
            # not become AVAILABLE planner capabilities.  Only a route
            # attested by the strict production bootstrap enters AUTO or the
            # evidence-backed deployment planner.
            use_planner = bool(
                joint_token is None
                and self._deployment_planner is not None
                and snapshot_lease is not None
                and (
                    requested_mode is ExecutionMode.AUTO
                    or is_production_executor(compatible[requested_mode])
                )
            )
            policy_active = bool(
                requested_mode is ExecutionMode.AUTO
                and requested_devices is None
                and self.config.scheduling_policy
                is not RequestSchedulingPolicy.FIFO
            )
            if policy_active and not use_planner:
                raise ConfigurationError(
                    "{} scheduling requires an immutable graph snapshot and a "
                    "measured deployment planner".format(
                        self.config.scheduling_policy.value
                    )
                )
            if policy_active:
                with self._placement_lock:
                    placement, placement_history = self._select_policy_placement(
                        function,
                        snapshot_lease.snapshot,
                        args,
                        kwargs,
                        compatible,
                    )
                    actual_mode = placement.mode
                    selected_devices = placement.devices
                    executor = placement.executor
                    planning_load = {
                        device: placement.predicted_service_seconds
                        for device in selected_devices
                    }
                    self._charge_planning_load(planning_load)
                    planning_load_precharged = True
                    self._record_policy_selection(placement)
            elif use_planner:
                with self._placement_lock:
                    decision = self._plan_submission(
                        function,
                        snapshot_lease.snapshot,
                        args,
                        kwargs,
                        requested_mode,
                        selected_devices,
                        compatible,
                    )
                actual_mode = ExecutionMode.parse(decision.mode)
                selected_devices = self._validate_requested_devices(
                    decision.devices
                )
                if (
                    requested_mode is not ExecutionMode.AUTO
                    and actual_mode is not requested_mode
                ):
                    raise ConfigurationError(
                        "deployment planner rewrote an explicit execution mode"
                    )
                if (
                    requested_devices is not None
                    and selected_devices != requested_devices
                ):
                    raise ConfigurationError(
                        "deployment planner rewrote an explicit device group"
                    )
                executor = compatible.get(actual_mode)
                if executor is None:
                    raise ExecutorCompatibilityError(
                        "deployment planner selected {} after that route was "
                        "excluded by executor/graph/argument compatibility".format(
                            actual_mode.value
                        )
                    )
                self._validate_executor_device_group(
                    executor, actual_mode, selected_devices
                )
                concurrent = max(1, self._planning_concurrency())
                per_request_seconds = self._decision_service_seconds(
                    decision, concurrent
                )
                if (
                    not math.isfinite(per_request_seconds)
                    or per_request_seconds < 0
                ):
                    raise ConfigurationError(
                        "deployment planner returned an invalid latency estimate"
                    )
                planning_load = {
                    device: per_request_seconds for device in selected_devices
                }
                with self._placement_lock:
                    self._charge_planning_load(planning_load)
                    planning_load_precharged = True
            else:
                actual_mode, executor = self._resolve_executor(
                    function, requested_mode, compatible
                )
                selected_devices, reservation_groups = self._reservation_policy(
                    executor, actual_mode, selected_devices
                )
            cooperative = self._validate_route(actual_mode, selected_devices)
            if cooperative and snapshot_lease is not None and not snapshot_lease.native_payload_ready:
                snapshot_lease.release()
                snapshot_lease = None
                raise NativePayloadUnavailableError(
                    "{} execution requires an immutable native graph snapshot; "
                    "the submitted graph provides metadata only".format(
                        actual_mode.value
                    )
                )
        except BaseException:
            if planning_load_precharged:
                self._release_untracked_planning_load(planning_load)
            if snapshot_lease is not None:
                snapshot_lease.release()
            raise

        if snapshot_lease is not None:
            with self._state_lock:
                counter = (
                    "snapshot_concurrent_requests"
                    if snapshot_lease.concurrent_execution_safe
                    else "snapshot_serialized_requests"
                )
                self._stats[counter] += 1

        future = EGMGPUFuture(next(self._request_ids), requested_mode)
        if joint_token is not None or "_calibration_route_id" in kwargs:
            future.add_done_callback(lambda f: setattr(f, "_joint_completed_ns", time.monotonic_ns()))
        if joint_token is not None:
            joint_token.future = future
            future.joint_route_id = joint_token.route['id']
            future.joint_route_layout = joint_token.route['layout']
        if snapshot_lease is not None:
            future._assign_snapshot(snapshot_lease.snapshot)
        try:
            self._track_future(
                future,
                planning_load,
                planning_load_precharged=planning_load_precharged,
            )
            self._controllers.submit(
                self._run_native,
                future,
                tuple(args),
                dict(kwargs),
                actual_mode,
                selected_devices,
                reservation_groups,
                executor,
                snapshot_lease,
                placement_history,
                submission_started,
                joint_token,
            )
        except RuntimeError as exc:
            if planning_load_precharged and future not in self._futures:
                self._release_untracked_planning_load(planning_load)
            if snapshot_lease is not None:
                snapshot_lease.release()
            if not future.done():
                future.set_exception(RuntimeClosedError("EGMGPU runtime is closed"))
            if joint_token is not None:
                self._reservations.finish_token(joint_token)
            raise RuntimeClosedError("EGMGPU runtime is closed") from exc
        return future

    def run_workflow(self, graph, calls, fail_fast=True):
        from .types import CallSpec

        specs = list(calls)
        if any(not isinstance(spec, CallSpec) for spec in specs):
            raise TypeError("run_workflow expects CallSpec objects created by egmgpu.call")
        workflow_lease = self._snapshot_registry.acquire(graph)
        try:
            # Preflight the complete workflow against one frozen generation
            # before any device reservation becomes visible.
            for spec in specs:
                requested_mode = self.config.mode if spec.mode is None else spec.mode
                requested_devices = self._validate_requested_devices(spec.devices)
                call_args = (graph,) + tuple(spec.args)
                compatible, rejected = self._compatible_routes(
                    spec.function,
                    workflow_lease.snapshot,
                    call_args,
                    dict(spec.kwargs),
                    requested_mode,
                )
                if requested_mode is ExecutionMode.AUTO:
                    if not compatible:
                        self._raise_no_auto_route(spec.function, rejected)
                elif requested_mode not in compatible:
                    self._raise_incompatible_or_unregistered(
                        spec.function, requested_mode, rejected
                    )
                if self._deployment_planner is None:
                    actual_mode, executor = self._resolve_executor(
                        spec.function, requested_mode, compatible
                    )
                    self._validate_route(actual_mode, requested_devices)
                    self._reservation_policy(
                        executor, actual_mode, requested_devices
                    )
                elif (
                    requested_mode is not ExecutionMode.AUTO
                    and requested_devices is not None
                ):
                    executor = compatible[requested_mode]
                    self._validate_executor_device_group(
                        executor, requested_mode, requested_devices
                    )
        except BaseException:
            workflow_lease.release()
            raise
        futures = []
        try:
            for spec in specs:
                submit_kwargs = dict(spec.kwargs)
                if spec.mode is not None:
                    submit_kwargs["_egmgpu_mode"] = spec.mode
                if spec.devices is not None:
                    submit_kwargs["_egmgpu_devices"] = spec.devices
                submit_kwargs["_egmgpu_pinned_snapshot"] = workflow_lease
                futures.append(
                    self.submit(spec.function, graph, *spec.args, **submit_kwargs)
                )
        except BaseException:
            for submitted in futures:
                submitted.cancel()
            raise
        finally:
            # Each submitted request owns a retained child lease.  Releasing
            # the workflow's capture permits reclamation after the last child.
            workflow_lease.release()

        pending = set(futures)
        results = [None] * len(futures)
        indexes = {future: index for index, future in enumerate(futures)}
        first_failure = None
        for completed in concurrent.futures.as_completed(futures):
            pending.discard(completed)
            index = indexes[completed]
            try:
                results[index] = completed.result()
            except BaseException as exc:
                if first_failure is None:
                    name = getattr(specs[index].function, "__name__", repr(specs[index].function))
                    first_failure = WorkflowExecutionError(index, name, exc)
                    if fail_fast:
                        for other in pending:
                            other.cancel()
        if first_failure is not None:
            raise first_failure from first_failure.cause
        return results

    def stats(self):
        with self._state_lock:
            totals = dict(self._stats)
            accepting = self._accepting
            live_futures = sum(not future.done() for future in self._futures)
            registered = tuple(
                sorted(
                    (getattr(function, "__name__", repr(function)), mode.value)
                    for function, mode in self._native_executors
                )
            )
            planned_queue = {
                device: seconds
                for device, seconds in sorted(self._planned_device_seconds.items())
                if seconds > 0
            }
        native_status = dict(self._adapter.status())
        with self._placement_lock:
            selection_count = sum(self._scheduler_mode_counts.values())
            wait_samples = self._scheduler_wait_samples
            scheduling = {
                "policy": self.config.scheduling_policy.value,
                "selection_count": selection_count,
                "mode_counts": dict(sorted(self._scheduler_mode_counts.items())),
                "group_counts": dict(sorted(self._scheduler_group_counts.items())),
                "device_counts": {
                    str(device): count
                    for device, count in sorted(self._scheduler_device_counts.items())
                },
                "graph_affinity_hits": self._scheduler_graph_affinity_hits,
                "view_affinity_hits": self._scheduler_view_affinity_hits,
                "mean_predicted_queue_seconds": (
                    0.0
                    if selection_count == 0
                    else self._scheduler_queue_seconds_total / selection_count
                ),
                "total_predicted_queue_seconds": (
                    self._scheduler_queue_seconds_total
                ),
                "maximum_predicted_queue_seconds": (
                    self._scheduler_queue_seconds_maximum
                ),
                "maximum_candidate_queue_seconds": (
                    self._scheduler_candidate_queue_seconds_maximum
                ),
                "reservation_wait_samples": wait_samples,
                "mean_reservation_wait_seconds": (
                    0.0
                    if wait_samples == 0
                    else self._scheduler_wait_seconds_total / wait_samples
                ),
                "total_reservation_wait_seconds": (
                    self._scheduler_wait_seconds_total
                ),
                "maximum_reservation_wait_seconds": (
                    self._scheduler_wait_seconds_maximum
                ),
                "last_selection": (
                    None
                    if self._scheduler_last_selection is None
                    else dict(self._scheduler_last_selection)
                ),
                "residency_signal": (
                    "last successful graph/view placement per device; a "
                    "controller hint, not a native cache-hit counter"
                ),
            }
        result = {
            "configured": True,
            "started": bool(native_status.get("running", False)),
            "accepting": accepting,
            "worker_owner": self.worker_owner,
            "devices": self.config.devices,
            "mode": self.config.mode.value,
            "scheduling_policy": self.config.scheduling_policy.value,
            "live_futures": live_futures,
            "totals": totals,
            "workers": native_status.get("workers", ()),
            "native_backend": native_status.get("communication_backend", "none"),
            "registered_native_executors": registered,
            "snapshots": self._snapshot_registry.stats(),
            "deployment_planner": (
                None
                if self._deployment_planner is None
                else dict(self._deployment_planner.stats())
            ),
            "planned_queue_seconds": planned_queue,
            "request_scheduler": scheduling,
        }
        result.update(self._reservations.snapshot())
        return result

    def shutdown(self, wait=True, cancel_futures=False):
        if bool(getattr(self._controller_local, "inside_native_executor", False)):
            raise ConfigurationError(
                "a native executor cannot shut down its owning EGMGPU runtime"
            )
        with self._state_lock:
            if not self._accepting:
                return
            self._accepting = False
            futures = tuple(self._futures)
        if cancel_futures:
            for future in futures:
                future.cancel()
            self._reservations.wake()

        def stop():
            try:
                self._stop_controllers(cancel_futures)
            finally:
                # Native entry points invoked by controller tasks are
                # synchronous.  Once all controllers have returned, draining
                # is cheap and avoids abandoning a C++ task even when the
                # Python shutdown request itself was non-blocking.
                self._adapter.release(self._owner_token, drain=True)

        if wait:
            stop()
        else:
            self._controllers.shutdown(wait=False)
            threading.Thread(
                target=stop,
                name="egmgpu-native-shutdown",
                daemon=True,
            ).start()

    def _stop_controllers(self, cancel_futures):
        # Never cancel the controller Future itself.  ``cancel_futures`` has
        # already marked each public EGMGPUFuture, so a queued controller will
        # exit before native execution.  Letting it run is nevertheless
        # essential: that controller owns the snapshot lease and releases it
        # in ``_run_native``'s finally block.
        try:
            self._controllers.shutdown(wait=True, cancel_futures=False)
        except TypeError:
            self._controllers.shutdown(wait=True)

    def _track_future(
        self,
        future,
        planning_load=None,
        *,
        planning_load_precharged=False,
    ):
        load = {} if planning_load is None else dict(planning_load)
        with self._state_lock:
            if not self._accepting:
                raise RuntimeClosedError("EGMGPU runtime is closed")
            if planning_load_precharged and self._untracked_plans <= 0:
                raise RuntimeError("precharged scheduler plan accounting underflow")
            self._futures.add(future)
            self._stats["submitted"] += 1
            if planning_load_precharged:
                self._untracked_plans -= 1
            else:
                for device, seconds in load.items():
                    self._planned_device_seconds[int(device)] += float(seconds)

        def completed_callback(completed):
            if completed.cancelled():
                outcome = "cancelled"
            elif completed.exception() is not None:
                outcome = "failed"
            else:
                outcome = "completed"
            with self._state_lock:
                self._stats[outcome] += 1
                self._futures.discard(completed)
                for device, seconds in load.items():
                    remaining = self._planned_device_seconds[int(device)] - float(
                        seconds
                    )
                    if remaining <= 1.0e-15:
                        self._planned_device_seconds.pop(int(device), None)
                    else:
                        self._planned_device_seconds[int(device)] = remaining
            if completed.cancelled():
                self._reservations.wake()

        future.add_done_callback(completed_callback)

    def _resolve_executor(self, function, requested_mode, compatible=None):
        routes = {} if compatible is None else dict(compatible)
        if compatible is None:
            with self._state_lock:
                routes = {
                    mode: executor
                    for (registered_function, mode), executor in self._native_executors.items()
                    if registered_function is function
                }
        if requested_mode is ExecutionMode.AUTO:
            for candidate in (
                ExecutionMode.REQUEST_PARALLEL,
                ExecutionMode.SINGLE,
            ):
                executor = routes.get(candidate)
                if executor is None or not is_production_executor(executor):
                    continue
                groups = self._configured_executor_groups(executor, candidate)
                if groups is None or groups:
                    return candidate, executor
            name = getattr(function, "__name__", repr(function))
            raise UnsupportedExecutionModeError(
                "{} has no compatible, evidence-authorized native single/request "
                "executor inside configured devices {}; AUTO does not select "
                "research registrations or an uncalibrated cooperative path".format(
                    name, self.config.devices
                )
            )
        executor = routes.get(requested_mode)
        if executor is None:
            name = getattr(function, "__name__", repr(function))
            raise UnsupportedExecutionModeError(
                "{} has no registered native {} executor".format(
                    name, requested_mode.value
                )
            )
        return requested_mode, executor

    def _executor_for_mode(self, function, mode):
        with self._state_lock:
            executor = self._native_executors.get((function, mode))
        if executor is None:
            name = getattr(function, "__name__", repr(function))
            raise UnsupportedExecutionModeError(
                "{} has no registered native {} executor".format(
                    name, mode.value
                )
            )
        return executor

    def _validate_registered_route(self, function, requested_mode):
        mode = ExecutionMode.parse(requested_mode)
        with self._state_lock:
            modes = {
                registered_mode
                for registered_function, registered_mode in self._native_executors
                if registered_function is function
            }
        if mode is ExecutionMode.AUTO:
            with self._state_lock:
                production = any(
                    registered_function is function
                    and is_production_executor(executor)
                    for (registered_function, _), executor in self._native_executors.items()
                )
            if production:
                return
        elif mode in modes:
            return
        name = getattr(function, "__name__", repr(function))
        raise UnsupportedExecutionModeError(
            "{} has no registered native {} executor".format(name, mode.value)
        )

    def _planning_concurrency(self):
        with self._state_lock:
            return (
                1
                + self._untracked_plans
                + sum(not future.done() for future in self._futures)
            )

    @staticmethod
    def _decision_service_seconds(decision, concurrent_requests):
        cost = getattr(getattr(decision, "control_plan", None), "cost", None)
        queue = float(getattr(cost, "queue", 0.0))
        total = float(decision.estimated_seconds)
        service = max(0.0, total - queue) / max(1, int(concurrent_requests))
        if not math.isfinite(service):
            raise ConfigurationError(
                "deployment planner returned a non-finite service estimate"
            )
        # This analytical floor is used only for queue admission.  It is never
        # reported as measured device time.
        return max(service, 1.0e-9)

    def _affinity_signature(self, function, snapshot, args, kwargs):
        provider = getattr(self._deployment_planner, "affinity_signature", None)
        if callable(provider):
            graph_key, view_key = provider(function, snapshot, args, kwargs)
            return _PlacementHistory(tuple(graph_key), str(view_key))
        return _PlacementHistory(
            (
                str(getattr(snapshot, "graph_id", "")),
                int(getattr(snapshot, "generation", 0)),
                str(getattr(snapshot, "topology_signature", "")),
                str(getattr(snapshot, "weight_signature", "")),
            ),
            getattr(function, "__name__", repr(function)),
        )

    def _history_affinity(self, signature, devices):
        histories = tuple(
            self._last_placement_by_device.get(device) for device in devices
        )
        graph_hit = bool(histories) and all(
            item is not None and item.graph_key == signature.graph_key
            for item in histories
        )
        view_hit = graph_hit and all(
            item.view_key == signature.view_key for item in histories
        )
        return graph_hit, view_hit

    def _select_policy_placement(self, function, snapshot, args, kwargs, compatible):
        """Select one exact evidence-backed group while holding placement lock."""

        policy = self.config.scheduling_policy
        signature = self._affinity_signature(function, snapshot, args, kwargs)
        if policy is RequestSchedulingPolicy.ADAPTIVE:
            resident = tuple(
                device
                for device in self.config.devices
                if self._history_affinity(signature, (device,))[1]
            )
            decision = self._plan_submission(
                function,
                snapshot,
                args,
                kwargs,
                ExecutionMode.AUTO,
                None,
                compatible,
                resident_devices_hint=resident,
            )
            mode = ExecutionMode.parse(decision.mode)
            executor = compatible.get(mode)
            if executor is None:
                raise ExecutorCompatibilityError(
                    "adaptive planner selected a route excluded by compatibility"
                )
            devices = tuple(int(device) for device in decision.devices)
            self._validate_executor_device_group(executor, mode, devices)
            graph_hit, view_hit = self._history_affinity(signature, devices)
            placement = _PolicyPlacement(
                mode=mode,
                devices=devices,
                executor=executor,
                predicted_service_seconds=self._decision_service_seconds(
                    decision, self._planning_concurrency()
                ),
                predicted_queue_seconds=max(
                    (
                        float(self._planned_device_seconds.get(device, 0.0))
                        for device in devices
                    ),
                    default=0.0,
                ),
                maximum_candidate_queue_seconds=max(
                    (
                        float(self._planned_device_seconds.get(device, 0.0))
                        for device in self.config.devices
                    ),
                    default=0.0,
                ),
                graph_affinity=graph_hit,
                view_affinity=view_hit,
                planner_reason=str(
                    getattr(getattr(decision, "control_plan", None), "reason", "")
                ),
            )
            return placement, signature

        # The scheduler baselines compare one-request-per-device placement.
        # RequestParallel is preferred; SINGLE is the exact semantic fallback.
        mode = next(
            (
                candidate
                for candidate in (
                    ExecutionMode.REQUEST_PARALLEL,
                    ExecutionMode.SINGLE,
                )
                if candidate in compatible
            ),
            None,
        )
        if mode is None:
            raise UnsupportedExecutionModeError(
                "{} scheduling requires an evidence-authorized one-device route".format(
                    policy.value
                )
            )
        executor = compatible[mode]
        groups = self._configured_executor_groups(executor, mode)
        if groups is None:
            raise ConfigurationError(
                "{} scheduling requires exact evidence-authorized device groups".format(
                    policy.value
                )
            )
        candidates = self._one_device_policy_candidates(
            function,
            snapshot,
            args,
            kwargs,
            mode,
            executor,
            groups,
            signature,
        )
        if policy is RequestSchedulingPolicy.ROUND_ROBIN:
            chosen = candidates[self._round_robin_cursor % len(candidates)][0]
            self._round_robin_cursor += 1
        elif policy is RequestSchedulingPolicy.LEAST_LOADED:
            chosen = min(
                candidates,
                key=lambda item: (
                    item[0].predicted_queue_seconds
                    + item[0].predicted_service_seconds,
                    sum(
                        self._scheduler_device_counts[device]
                        for device in item[0].devices
                    ),
                    item[0].devices,
                ),
            )[0]
        elif policy is RequestSchedulingPolicy.REUSE_AFFINITY:
            def reuse_score(item):
                placement, state_seconds = item
                reuse_credit = (
                    state_seconds
                    if placement.view_affinity
                    else (0.5 * state_seconds if placement.graph_affinity else 0.0)
                )
                return (
                    placement.predicted_queue_seconds
                    + placement.predicted_service_seconds
                    - reuse_credit,
                    not placement.view_affinity,
                    not placement.graph_affinity,
                    sum(
                        self._scheduler_device_counts[device]
                        for device in placement.devices
                    ),
                    placement.devices,
                )

            chosen = min(candidates, key=reuse_score)[0]
        else:
            raise AssertionError("unexpected request scheduling policy")
        return chosen, signature

    def _one_device_policy_candidates(
        self,
        function,
        snapshot,
        args,
        kwargs,
        mode,
        executor,
        groups,
        signature,
    ):
        candidates = []
        rejected = []
        for group in groups:
            if len(group) != 1:
                continue
            try:
                decision = self._plan_submission(
                    function,
                    snapshot,
                    args,
                    kwargs,
                    mode,
                    group,
                    {mode: executor},
                    concurrent_requests=1,
                    resident_devices_hint=(
                        group if self._history_affinity(signature, group)[1] else ()
                    ),
                    candidate_evaluation=True,
                )
            except (DeploymentPlanningError, NoFeasiblePlanError) as exc:
                # Candidate failures are retained for a fail-closed diagnostic;
                # another exact group may still satisfy the memory bound.
                rejected.append("{}: {}".format(group, exc))
                continue
            selected = tuple(int(device) for device in decision.devices)
            if selected != group:
                raise ConfigurationError(
                    "deployment planner rewrote an exact scheduler candidate"
                )
            graph_hit, view_hit = self._history_affinity(signature, group)
            queue_seconds = max(
                float(self._planned_device_seconds.get(device, 0.0))
                for device in group
            )
            state_seconds = float(
                getattr(
                    getattr(getattr(decision, "control_plan", None), "cost", None),
                    "state",
                    0.0,
                )
            )
            candidates.append(
                (
                    _PolicyPlacement(
                        mode=mode,
                        devices=tuple(group),
                        executor=executor,
                        predicted_service_seconds=self._decision_service_seconds(
                            decision, 1
                        ),
                        predicted_queue_seconds=queue_seconds,
                        maximum_candidate_queue_seconds=max(
                            (
                                float(
                                    self._planned_device_seconds.get(device, 0.0)
                                )
                                for device in self.config.devices
                            ),
                            default=0.0,
                        ),
                        graph_affinity=graph_hit,
                        view_affinity=view_hit,
                        planner_reason=str(
                            getattr(
                                getattr(decision, "control_plan", None),
                                "reason",
                                "",
                            )
                        ),
                    ),
                    state_seconds,
                )
            )
        if not candidates:
            detail = "; ".join(rejected)
            raise ConfigurationError(
                "no evidence-authorized one-device group is memory feasible{}".format(
                    "" if not detail else ": " + detail
                )
            )
        return sorted(candidates, key=lambda item: item[0].devices)

    def _charge_planning_load(self, planning_load):
        load = {} if planning_load is None else dict(planning_load)
        with self._state_lock:
            for device, seconds in load.items():
                self._planned_device_seconds[int(device)] += float(seconds)
            self._untracked_plans += 1

    def _release_untracked_planning_load(self, planning_load):
        load = {} if planning_load is None else dict(planning_load)
        with self._state_lock:
            if self._untracked_plans <= 0:
                raise RuntimeError("untracked scheduler plan accounting underflow")
            self._untracked_plans -= 1
            for device, seconds in load.items():
                remaining = self._planned_device_seconds[int(device)] - float(seconds)
                if remaining <= 1.0e-15:
                    self._planned_device_seconds.pop(int(device), None)
                else:
                    self._planned_device_seconds[int(device)] = remaining

    def _record_policy_selection(self, placement):
        self._scheduler_mode_counts[placement.mode.value] += 1
        group_name = ",".join(str(device) for device in placement.devices)
        self._scheduler_group_counts[group_name] += 1
        for device in placement.devices:
            self._scheduler_device_counts[int(device)] += 1
        if placement.graph_affinity:
            self._scheduler_graph_affinity_hits += 1
        if placement.view_affinity:
            self._scheduler_view_affinity_hits += 1
        queue_seconds = float(placement.predicted_queue_seconds)
        self._scheduler_queue_seconds_total += queue_seconds
        self._scheduler_queue_seconds_maximum = max(
            self._scheduler_queue_seconds_maximum, queue_seconds
        )
        self._scheduler_candidate_queue_seconds_maximum = max(
            self._scheduler_candidate_queue_seconds_maximum,
            float(placement.maximum_candidate_queue_seconds),
        )
        self._scheduler_last_selection = {
            "policy": self.config.scheduling_policy.value,
            "mode": placement.mode.value,
            "devices": placement.devices,
            "predicted_queue_seconds": queue_seconds,
            "maximum_candidate_queue_seconds": float(
                placement.maximum_candidate_queue_seconds
            ),
            "predicted_service_seconds": float(
                placement.predicted_service_seconds
            ),
            "graph_affinity": bool(placement.graph_affinity),
            "view_affinity": bool(placement.view_affinity),
            "planner_reason": placement.planner_reason,
        }

    def _record_reservation_wait(self, seconds):
        measured = max(0.0, float(seconds))
        with self._placement_lock:
            self._scheduler_wait_samples += 1
            self._scheduler_wait_seconds_total += measured
            self._scheduler_wait_seconds_maximum = max(
                self._scheduler_wait_seconds_maximum, measured
            )

    def _record_completed_placement(self, devices, history):
        if history is None:
            return
        with self._placement_lock:
            for device in devices:
                self._last_placement_by_device[int(device)] = history

    def _plan_submission(
        self,
        function,
        snapshot,
        args,
        kwargs,
        requested_mode,
        requested_devices,
        compatible_routes,
        concurrent_requests=None,
        resident_devices_hint=(),
        candidate_evaluation=False,
    ):
        with self._state_lock:
            registered = tuple(
                (mode, executor)
                for mode, executor in compatible_routes.items()
                if is_production_executor(executor)
            )
            registered_modes = tuple(mode for mode, _ in registered)
            eligible_device_groups = {}
            configured = set(self.config.devices)
            for registered_mode, executor in registered:
                groups = _executor_device_groups(executor, registered_mode)
                if groups is not None:
                    eligible_device_groups[registered_mode] = tuple(
                        group for group in groups if set(group).issubset(configured)
                    )
            queue_seconds = {
                device: float(self._planned_device_seconds.get(device, 0.0))
                for device in self.config.devices
            }
            observed_concurrency = (
                1
                + self._untracked_plans
                + sum(not future.done() for future in self._futures)
            )
            if concurrent_requests is None:
                concurrent_requests = observed_concurrency
        plan_request = self._deployment_planner.plan_request
        if candidate_evaluation:
            evaluator = getattr(self._deployment_planner, "evaluate_candidate", None)
            if callable(evaluator):
                plan_request = evaluator
        planner_kwargs = {
            "requested_mode": requested_mode,
            "requested_devices": requested_devices,
            "registered_modes": registered_modes,
            "queue_seconds": queue_seconds,
            "concurrent_requests": concurrent_requests,
            "minimum_multigpu_gain": self.config.min_predicted_gain,
        }
        try:
            signature = inspect.signature(plan_request)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and (
            "eligible_device_groups" in signature.parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        ):
            planner_kwargs["eligible_device_groups"] = eligible_device_groups
        if signature is not None and (
            "resident_devices_hint" in signature.parameters
            or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        ):
            planner_kwargs["resident_devices_hint"] = tuple(
                int(device) for device in resident_devices_hint
            )
        return plan_request(function, snapshot, args, kwargs, **planner_kwargs)

    def _compatible_routes(self, function, snapshot, args, kwargs, requested_mode):
        """Return routes compatible before planning or device admission."""

        parsed = ExecutionMode.parse(requested_mode)
        with self._state_lock:
            candidates = tuple(
                (mode, executor)
                for (registered_function, mode), executor in self._native_executors.items()
                if registered_function is function
                and (parsed is ExecutionMode.AUTO or mode is parsed)
            )
        compatible = {}
        rejected = {}
        for mode, executor in candidates:
            if parsed is ExecutionMode.AUTO and not is_production_executor(executor):
                rejected[mode] = (
                    "route was registered through the research interface and is "
                    "not production-evidence attested"
                )
                continue
            result = evaluate_executor_compatibility(
                executor, snapshot, args, kwargs, mode
            )
            if result.compatible:
                groups = self._configured_executor_groups(executor, mode)
                if groups is not None and not groups:
                    rejected[mode] = (
                        "route has no evidence-authorized device group inside "
                        "the configured runtime"
                    )
                else:
                    compatible[mode] = executor
            else:
                rejected[mode] = result.reason
        return compatible, rejected

    @staticmethod
    def _raise_incompatible_or_unregistered(function, mode, rejected):
        name = getattr(function, "__name__", repr(function))
        reason = rejected.get(mode)
        if reason is not None:
            raise ExecutorCompatibilityError(
                "{} cannot use explicit {} execution: {}".format(
                    name, ExecutionMode.parse(mode).value, reason
                )
            )
        raise UnsupportedExecutionModeError(
            "{} has no registered native {} executor".format(
                name, ExecutionMode.parse(mode).value
            )
        )

    @staticmethod
    def _raise_no_auto_route(function, rejected):
        name = getattr(function, "__name__", repr(function))
        detail = "; ".join(
            "{}: {}".format(mode.value, rejected[mode])
            for mode in sorted(rejected, key=lambda item: item.value)
        )
        suffix = "" if not detail else "; rejected routes: {}".format(detail)
        raise UnsupportedExecutionModeError(
            "{} has no compatible production route for AUTO{}".format(name, suffix)
        )

    def _configured_executor_groups(self, executor, mode):
        groups = _executor_device_groups(executor, mode)
        if groups is None:
            return None
        configured = set(self.config.devices)
        return tuple(group for group in groups if set(group).issubset(configured))

    def _validate_executor_device_group(self, executor, mode, devices):
        groups = self._configured_executor_groups(executor, mode)
        if groups is None:
            return
        selected = tuple(int(device) for device in devices)
        if selected not in groups:
            raise ConfigurationError(
                "{} executor has no evidence-authorized route for device group {}; "
                "eligible groups in this runtime are {}".format(
                    ExecutionMode.parse(mode).value, selected, groups
                )
            )

    def _reservation_policy(self, executor, mode, selected_devices):
        """Choose only evidence-authorized groups before entering the queue."""

        groups = self._configured_executor_groups(executor, mode)
        if groups is None:
            return selected_devices, None
        if selected_devices is not None:
            self._validate_executor_device_group(executor, mode, selected_devices)
            return selected_devices, None
        if not groups:
            raise UnsupportedExecutionModeError(
                "{} executor has no evidence-authorized device group inside the "
                "configured runtime {}".format(
                    ExecutionMode.parse(mode).value, self.config.devices
                )
            )
        parsed_mode = ExecutionMode.parse(mode)
        if parsed_mode in {ExecutionMode.SINGLE, ExecutionMode.REQUEST_PARALLEL}:
            # Let the FIFO reservation choose any currently free authorized GPU
            # so request parallelism is preserved without admitting other GPUs.
            return None, groups
        if self.config.devices in groups:
            return self.config.devices, None
        # With no calibrated planner, preserve as much of the configured gang
        # as retained evidence permits; ties are deterministic.
        selected = sorted(groups, key=lambda group: (-len(group), group))[0]
        return selected, None

    def _run_native(
        self,
        future,
        args,
        kwargs,
        mode,
        preferred_devices,
        eligible_device_groups,
        executor,
        snapshot_lease,
        placement_history,
        submission_started,
        joint_token=None,
    ):
        guard = (
            nullcontext(True)
            if snapshot_lease is None
            else snapshot_lease.execution_guard(cancelled=future.cancelled)
        )
        try:
            with guard as acquired:
                if not acquired:
                    return
                if snapshot_lease is not None and not snapshot_lease.concurrent_execution_safe:
                    snapshot_lease.assert_generation_current()
                cooperative = mode in {
                    ExecutionMode.REPLICATED,
                    ExecutionMode.PARTITIONED,
                }
                count = (
                    len(preferred_devices)
                    if preferred_devices is not None
                    else (
                        len(eligible_device_groups[0])
                        if eligible_device_groups is not None
                        else (len(self.config.devices) if cooperative else 1)
                    )
                )
                selected = self._reservations.acquire(
                    count,
                    preferred_devices,
                    cancelled=future.cancelled,
                    eligible=eligible_device_groups,
                    joint_token=joint_token,
                )
                if selected is None:
                    return
                self._record_reservation_wait(
                    time.monotonic() - float(submission_started)
                )
                try:
                    # The request may have waited behind another device gang
                    # after its initial submission-time check.  Revalidate a
                    # mutable graph immediately before invoking pybind so a
                    # stale request never touches its graph/cache state.
                    if (
                        snapshot_lease is not None
                        and not snapshot_lease.concurrent_execution_safe
                    ):
                        snapshot_lease.assert_generation_current()
                    future._assign(selected, mode)
                    if not future.set_running_or_notify_cancel():
                        return
                    context = NativeCallContext(
                        devices=selected,
                        mode=mode,
                        request_id=future.request_id,
                        route_id=None if joint_token is None else joint_token.route['id'],
                        route_layout=None if joint_token is None else joint_token.route['layout'],
                        graph_snapshot=(
                            None if snapshot_lease is None else snapshot_lease.snapshot
                        ),
                    )
                    self._controller_local.inside_native_executor = True
                    try:
                        result = executor(context, args, kwargs)
                    finally:
                        self._controller_local.inside_native_executor = False
                    if snapshot_lease is not None and not snapshot_lease.concurrent_execution_safe:
                        snapshot_lease.assert_generation_current()
                    self._record_completed_placement(selected, placement_history)
                    future.set_result(result)
                except BaseException as exc:
                    if not future.done():
                        future.set_exception(exc)
                finally:
                    self._reservations.release(selected)
        except BaseException as exc:
            if not future.done():
                try:
                    future.set_exception(exc)
                except concurrent.futures.InvalidStateError:
                    pass
        finally:
            self._reservations.finish_token(joint_token)
            if snapshot_lease is not None:
                snapshot_lease.release()

    def _validate_requested_devices(self, devices):
        if devices is None:
            return None
        normalized = tuple(int(device) for device in devices)
        if not normalized:
            raise ConfigurationError("a request-specific device set cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ConfigurationError("request-specific CUDA devices must be unique")
        unknown = set(normalized).difference(self.config.devices)
        if unknown:
            raise ConfigurationError(
                "CUDA devices {} are not managed by this runtime".format(sorted(unknown))
            )
        return tuple(sorted(normalized))

    def _validate_route(self, mode, selected_devices):
        cooperative = mode in {
            ExecutionMode.REPLICATED,
            ExecutionMode.PARTITIONED,
        }
        if cooperative:
            assigned = self.config.devices if selected_devices is None else selected_devices
            if len(assigned) < 2:
                raise ConfigurationError(
                    "{} execution requires at least two devices".format(mode.value)
                )
        elif selected_devices is not None and len(selected_devices) != 1:
            raise ConfigurationError(
                "{} execution accepts exactly one request-specific device".format(
                    mode.value
                )
            )
        return cooperative

    @staticmethod
    def _first_graph_like(args, kwargs=None):
        values = tuple(args)
        if kwargs:
            values += tuple(kwargs.values())
        for value in values:
            if bool(getattr(value, "_eggpu_bulk_csr", False)):
                return value
            if (
                callable(getattr(value, "number_of_nodes", None))
                and callable(getattr(value, "number_of_edges", None))
                and hasattr(value, "is_directed")
            ):
                return value
        return None
