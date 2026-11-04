"""Explicit CPU-test scheduler for EGMGPU control-plane contracts.

This module's Python ``_DeviceWorker`` pool exists only so CPU-only tests can
exercise reservations, cancellation, snapshots, and deterministic merging.
It must not be used as the production CUDA scheduler.  Production execution
is owned exclusively by the C++ ``RuntimeManager`` through
``NativeOwnedRuntime``; this distinction prevents two fixed-worker pools from
competing for the same devices.
"""

import concurrent.futures
import ctypes
import ctypes.util
import itertools
import queue
import threading
from contextlib import nullcontext
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from .types import CooperativeContext
from .types import CooperativePlan
from .types import ExecutionMode
from .types import RuntimeConfig
from .snapshot import GraphSnapshotRegistry
from .snapshot import NativePayloadUnavailableError


class EGMGPUError(RuntimeError):
    """Base exception for EGMGPU runtime errors."""


class ConfigurationError(EGMGPUError):
    """The requested runtime configuration is invalid or unavailable."""


class RuntimeClosedError(EGMGPUError):
    """A request was submitted after the runtime stopped accepting work."""


class UnsupportedExecutionModeError(EGMGPUError):
    """A function has no executor for a forced cooperative mode."""


class CooperativeExecutionError(EGMGPUError):
    """One device failed while executing a cooperative request."""

    def __init__(self, device, cause):
        super().__init__(
            "cooperative execution failed on CUDA device {}: {}".format(device, cause)
        )
        self.device = int(device)
        self.cause = cause


class WorkflowExecutionError(EGMGPUError):
    """One call in an explicit workflow failed."""

    def __init__(self, index, function_name, cause):
        super().__init__(
            "workflow call {} ({}) failed: {}".format(index, function_name, cause)
        )
        self.index = int(index)
        self.function_name = str(function_name)
        self.cause = cause


def _load_cudart():
    candidates = []
    discovered = ctypes.util.find_library("cudart")
    if discovered:
        candidates.append(discovered)
    candidates.extend(
        [
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0",
            "cudart64_12.dll",
            "cudart64_110.dll",
        ]
    )
    errors = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "CUDA runtime library was not found"
    raise ConfigurationError("unable to load CUDA runtime: {}".format(detail))


def cuda_device_count():
    """Return the number of CUDA devices visible to the current process."""

    cudart = _load_cudart()
    count = ctypes.c_int()
    status = int(cudart.cudaGetDeviceCount(ctypes.byref(count)))
    if status != 0:
        raise ConfigurationError("cudaGetDeviceCount failed with status {}".format(status))
    return int(count.value)


def bind_cuda_device(device):
    """Bind the calling worker thread to one CUDA device."""

    cudart = _load_cudart()
    status = int(cudart.cudaSetDevice(ctypes.c_int(int(device))))
    if status != 0:
        raise ConfigurationError(
            "cudaSetDevice({}) failed with status {}".format(device, status)
        )


class EGMGPUFuture(concurrent.futures.Future):
    """Future returned by :func:`easygraph.egmgpu.submit`."""

    def __init__(self, request_id, requested_mode):
        super().__init__()
        self.request_id = int(request_id)
        self.requested_mode = ExecutionMode.parse(requested_mode)
        self._assignment_lock = threading.Lock()
        self._assigned_devices = ()
        self._execution_mode = None
        self._graph_snapshot = None

    @property
    def assigned_devices(self):
        with self._assignment_lock:
            return self._assigned_devices

    @property
    def execution_mode(self):
        with self._assignment_lock:
            return self._execution_mode

    @property
    def graph_snapshot(self):
        """Submission-time graph metadata, or ``None`` for non-graph calls."""

        with self._assignment_lock:
            return self._graph_snapshot

    def _assign(self, devices, mode):
        with self._assignment_lock:
            self._assigned_devices = tuple(devices)
            self._execution_mode = ExecutionMode.parse(mode)

    def _assign_snapshot(self, snapshot):
        with self._assignment_lock:
            self._graph_snapshot = snapshot


@dataclass
class _WorkerTask:
    function: Callable[[], Any]
    future: concurrent.futures.Future


class _DeviceWorker:
    def __init__(self, device, binder):
        self.device = int(device)
        self._binder = binder
        self._queue = queue.Queue()
        self._started = threading.Event()
        self._startup_error = None
        self._stats_lock = threading.Lock()
        self._stats = Counter()
        self._thread = threading.Thread(
            target=self._run,
            name="egmgpu-gpu-{}".format(self.device),
            daemon=True,
        )
        self._thread.start()

    def await_start(self, timeout=10.0):
        if not self._started.wait(timeout):
            raise ConfigurationError(
                "CUDA device {} worker did not start within {:.1f}s".format(
                    self.device, timeout
                )
            )
        if self._startup_error is not None:
            raise ConfigurationError(
                "failed to initialize CUDA device {} worker: {}".format(
                    self.device, self._startup_error
                )
            ) from self._startup_error

    def submit(self, function):
        future = concurrent.futures.Future()
        with self._stats_lock:
            self._stats["submitted"] += 1
        self._queue.put(_WorkerTask(function=function, future=future))
        return future

    def shutdown(self, wait=True):
        self._queue.put(None)
        if wait:
            self._thread.join()

    def stats(self):
        with self._stats_lock:
            values = dict(self._stats)
        values.update(
            {
                "device": self.device,
                "queue_depth": self._queue.qsize(),
                "thread_ident": self._thread.ident,
                "alive": self._thread.is_alive(),
            }
        )
        return values

    def _run(self):
        try:
            self._binder(self.device)
        except BaseException as exc:
            self._startup_error = exc
            self._started.set()
            return
        self._started.set()
        while True:
            task = self._queue.get()
            if task is None:
                return
            future = task.future
            if not future.set_running_or_notify_cancel():
                with self._stats_lock:
                    self._stats["cancelled"] += 1
                continue
            try:
                result = task.function()
            except BaseException as exc:
                future.set_exception(exc)
                with self._stats_lock:
                    self._stats["failed"] += 1
            else:
                future.set_result(result)
                with self._stats_lock:
                    self._stats["completed"] += 1


@dataclass
class _ReservationWaiter:
    sequence: int
    count: int
    preferred: Optional[Tuple[int, ...]]
    eligible: Optional[Tuple[Tuple[int, ...], ...]]
    cancelled: Callable[[], bool]


class _JointReservationToken:
    def __init__(self, route):
        self.route = route
        self.future = None
        self.aborted = False
        self.published = False
        self.waiter = None

    def cancelled(self):
        return self.aborted or (self.future is not None and self.future.cancelled())


class _GangReservation:
    """Atomic reservations; FIFO by default, optional conflict-safe bypass."""

    def __init__(self, devices, policy="fifo"):
        if policy not in {"fifo", "conflict_aware"}:
            raise ValueError("unknown reservation policy")
        self._policy = policy
        self._bypass_count = 0
        self._devices = tuple(devices)
        self._busy = set()
        self._waiters = []
        self._sequence = itertools.count(1)
        self._condition = threading.Condition()
        self._joint_planner = None
        self._joint_retained = {}
        self._joint_tokens = set()
        self._joint_last = None

    def configure_joint(self, planner):
        with self._condition:
            if self._busy or self._waiters or self._joint_planner is not None:
                raise ConfigurationError("Joint reservation must be configured before submissions")
            self._joint_planner = planner

    def admit_window(self, candidates):
        # Selection, retained-capacity admission, and queue publication share
        # exactly the lock that protects GPU busy sets and reservation grants.
        with self._condition:
            planner = self._joint_planner
            if planner is None:
                raise ConfigurationError("Joint profile is not configured")
            queued = {d: 0.0 for d in self._devices}
            for token in self._joint_tokens:
                if not token.cancelled():
                    for device in token.route['devices']:
                        queued[device] += token.route['seconds']
            answer = planner.select(candidates, self._joint_retained, queued)
            tokens = [_JointReservationToken(route) for route in answer['routes']]
            for i in answer['order']:
                token = tokens[i]
                token.waiter = _ReservationWaiter(next(self._sequence),
                    len(token.route['devices']), tuple(token.route['devices']), None,
                    token.cancelled)
                self._waiters.append(token.waiter)
                self._joint_tokens.add(token)
            self._joint_retained = answer['retained']
            self._joint_last = {k: v for k, v in answer.items() if k not in ('routes','retained')}
            self._joint_last['route_ids'] = [r['id'] for r in answer['routes']]
            self._joint_last['capacity_scope'] = planner.scope
            self._joint_last['accounting'] = planner.accounting
            self._condition.notify_all()
            return tokens, self._joint_last.copy()

    def publish_token(self, token):
        with self._condition:
            if token not in self._joint_tokens or token.published or token.cancelled():
                raise ConfigurationError("Invalid, reused, or cancelled joint token")
            token.published = True

    def finish_token(self, token):
        if token is None:
            return
        with self._condition:
            token.aborted = True
            if token.waiter in self._waiters:
                self._waiters.remove(token.waiter)
            self._joint_tokens.discard(token)
            # Retained graph/workspace caches can outlive a future: no credit
            # is returned here, even for cancelled or failed native calls.
            self._condition.notify_all()

    def acquire(self, count, preferred, cancelled, eligible=None, joint_token=None):
        preferred_tuple = None if preferred is None else tuple(preferred)
        eligible_groups = (
            None
            if eligible is None
            else tuple(tuple(int(device) for device in group) for group in eligible)
        )
        waiter = _ReservationWaiter(
            sequence=next(self._sequence),
            count=int(count),
            preferred=preferred_tuple,
            eligible=eligible_groups,
            cancelled=cancelled,
        )
        with self._condition:
            if joint_token is None:
                if self._joint_planner is not None:
                    raise ConfigurationError("Unprofiled reservation in joint runtime")
                self._waiters.append(waiter)
            else:
                if joint_token not in self._joint_tokens or not joint_token.published:
                    raise ConfigurationError("Joint token was not admitted/published")
                waiter = joint_token.waiter
                if waiter.preferred != preferred_tuple or waiter.count != int(count):
                    raise ConfigurationError("Selected route differs from admitted footprint")
            while True:
                if cancelled():
                    self._waiters.remove(waiter)
                    self._condition.notify_all()
                    return None
                if self._waiters[0] is waiter or self._policy == "conflict_aware":
                    # Do not let a later request consume any GPU potentially
                    # needed by an earlier waiter. Explicit groups are exact;
                    # unplaced requests conservatively protect their eligible
                    # union (or all devices). Running work remains exclusive.
                    protected = set()
                    if self._policy == "conflict_aware":
                        for earlier in self._waiters[:self._waiters.index(waiter)]:
                            if earlier.cancelled():
                                continue
                            if earlier.preferred is not None:
                                protected.update(earlier.preferred)
                            elif earlier.eligible is not None:
                                for group in earlier.eligible:
                                    protected.update(group)
                            else:
                                protected.update(self._devices)
                    available = [
                        device for device in self._devices
                        if device not in self._busy and device not in protected
                    ]
                    if preferred_tuple is None:
                        if eligible_groups is None:
                            selected = tuple(available[: waiter.count])
                        else:
                            selected = next(
                                (
                                    group
                                    for group in eligible_groups
                                    if len(group) == waiter.count
                                    and all(device in available for device in group)
                                ),
                                (),
                            )
                    elif all(device in available for device in preferred_tuple):
                        selected = preferred_tuple
                    else:
                        selected = ()
                    if len(selected) == waiter.count:
                        if self._waiters[0] is not waiter:
                            self._bypass_count += 1
                        self._busy.update(selected)
                        self._waiters.remove(waiter)
                        self._condition.notify_all()
                        return selected
                self._condition.wait(timeout=0.05)

    def release(self, devices):
        with self._condition:
            for device in devices:
                if device not in self._busy:
                    raise RuntimeError(
                        "CUDA device {} was released without a reservation".format(device)
                    )
                self._busy.remove(device)
            self._condition.notify_all()

    def wake(self):
        with self._condition:
            self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return {
                "busy_devices": tuple(sorted(self._busy)),
                "waiting_reservations": len(self._waiters),
                "reservation_policy": self._policy,
                "reservation_bypass_count": self._bypass_count,
                "joint_last_selection": self._joint_last,
                "joint_live_tokens": len(self._joint_tokens),
                "joint_retained_declared_bytes": {
                    d: sum(n for device,n in self._joint_retained.values() if device == d)
                    for d in self._devices
                },
            }


class EGMGPURuntime:
    """CPU-test scheduler with fake per-device Python workers.

    Tests pass a non-CUDA ``device_binder`` and may execute Python callables on
    the fake workers.  Production code must use ``NativeOwnedRuntime``; native
    device workers must never receive Python objects or callbacks.
    """

    worker_owner = "python_cpu_test"

    def __init__(
        self,
        config,
        device_binder=bind_cuda_device,
        snapshot_registry=None,
        snapshot_provider=None,
    ):
        if not isinstance(config, RuntimeConfig):
            raise TypeError("config must be a RuntimeConfig")
        self.config = config
        if snapshot_registry is not None and snapshot_provider is not None:
            raise ValueError(
                "provide snapshot_registry or snapshot_provider, not both"
            )
        self._snapshot_registry = (
            GraphSnapshotRegistry(snapshot_provider=snapshot_provider)
            if snapshot_registry is None
            else snapshot_registry
        )
        self._state_lock = threading.RLock()
        self._accepting = True
        self._request_ids = itertools.count(1)
        self._stats = Counter()
        self._futures = set()
        self._cooperative_builders = {}
        self._workers = {
            device: _DeviceWorker(device, device_binder) for device in config.devices
        }
        try:
            for worker in self._workers.values():
                worker.await_start()
        except BaseException:
            for worker in self._workers.values():
                worker.shutdown(wait=False)
            raise
        self._reservations = _GangReservation(config.devices, config.reservation_policy)
        # Coordinator threads never execute CUDA.  They wait for reservations,
        # dispatch work, merge native results, and propagate failures.
        self._controllers = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(4, 2 * len(config.devices)),
            thread_name_prefix="egmgpu-control",
        )

    @property
    def closed(self):
        with self._state_lock:
            return not self._accepting

    def register_cooperative_executor(self, function, mode, builder):
        """Register a native/operator adapter for a cooperative mode."""

        if not callable(function):
            raise TypeError("function must be callable")
        parsed_mode = ExecutionMode.parse(mode)
        if parsed_mode not in {ExecutionMode.REPLICATED, ExecutionMode.PARTITIONED}:
            raise ValueError("cooperative executors require replicated or partitioned mode")
        if not callable(builder):
            raise TypeError("builder must be callable")
        with self._state_lock:
            self._cooperative_builders[(function, parsed_mode)] = builder

    def unregister_cooperative_executor(self, function, mode):
        parsed_mode = ExecutionMode.parse(mode)
        with self._state_lock:
            self._cooperative_builders.pop((function, parsed_mode), None)

    def submit(self, function, *args, **kwargs):
        mode = kwargs.pop("_egmgpu_mode", None)
        devices = kwargs.pop("_egmgpu_devices", None)
        if not callable(function):
            raise TypeError("function must be callable")
        requested_mode = (
            self.config.mode if mode is None else ExecutionMode.parse(mode)
        )
        selected_devices = self._validate_requested_devices(devices)
        actual_mode = self._resolve_mode(function, requested_mode)
        if (
            actual_mode in {ExecutionMode.SINGLE, ExecutionMode.REQUEST_PARALLEL}
            and selected_devices is not None
            and len(selected_devices) != 1
        ):
            raise ConfigurationError(
                "{} execution accepts exactly one request-specific device".format(
                    actual_mode.value
                )
            )
        if actual_mode in {ExecutionMode.REPLICATED, ExecutionMode.PARTITIONED}:
            cooperative_devices = (
                self.config.devices if selected_devices is None else selected_devices
            )
            if len(cooperative_devices) < 2:
                raise ConfigurationError(
                    "{} execution requires at least two devices".format(
                        actual_mode.value
                    )
                )
        snapshot_lease = None
        graph = self._first_graph_like(args, kwargs)
        if graph is not None:
            snapshot_lease = self._snapshot_registry.acquire(graph)
            if (
                actual_mode
                in {ExecutionMode.REPLICATED, ExecutionMode.PARTITIONED}
                and not snapshot_lease.native_payload_ready
            ):
                snapshot_lease.release()
                raise NativePayloadUnavailableError(
                    "{} execution requires an immutable native graph snapshot; "
                    "the submitted graph provides metadata only".format(
                        actual_mode.value
                    )
                )
        request_id = next(self._request_ids)
        future = EGMGPUFuture(request_id, requested_mode)
        if snapshot_lease is not None:
            future._assign_snapshot(snapshot_lease.snapshot)
        try:
            self._track_future(future)
        except BaseException:
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
        if actual_mode in {ExecutionMode.REPLICATED, ExecutionMode.PARTITIONED}:
            builder = self._cooperative_builders[(function, actual_mode)]
            controller = self._run_cooperative
            controller_args = (
                future,
                function,
                tuple(args),
                dict(kwargs),
                actual_mode,
                selected_devices,
                builder,
                snapshot_lease,
            )
        else:
            controller = self._run_independent
            controller_args = (
                future,
                function,
                tuple(args),
                dict(kwargs),
                actual_mode,
                selected_devices,
                snapshot_lease,
            )
        try:
            self._controllers.submit(controller, *controller_args)
        except RuntimeError as exc:
            future.set_exception(RuntimeClosedError("EGMGPU runtime is closed"))
            if snapshot_lease is not None:
                snapshot_lease.release()
            raise RuntimeClosedError("EGMGPU runtime is closed") from exc
        return future

    def run_workflow(self, graph, calls, fail_fast=True):
        from .types import CallSpec

        specs = list(calls)
        if any(not isinstance(spec, CallSpec) for spec in specs):
            raise TypeError("run_workflow expects CallSpec objects created by egmgpu.call")
        futures = []
        for spec in specs:
            submit_kwargs = dict(spec.kwargs)
            if spec.mode is not None:
                submit_kwargs["_egmgpu_mode"] = spec.mode
            if spec.devices is not None:
                submit_kwargs["_egmgpu_devices"] = spec.devices
            futures.append(
                self.submit(spec.function, graph, *spec.args, **submit_kwargs)
            )

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
                    name = getattr(
                        specs[index].function,
                        "__name__",
                        repr(specs[index].function),
                    )
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
        snapshot = {
            "configured": True,
            "accepting": accepting,
            "devices": self.config.devices,
            "mode": self.config.mode.value,
            "live_futures": live_futures,
            "totals": totals,
            "workers": {
                device: worker.stats() for device, worker in self._workers.items()
            },
            "snapshots": self._snapshot_registry.stats(),
        }
        snapshot.update(self._reservations.snapshot())
        return snapshot

    def shutdown(self, wait=True, cancel_futures=False):
        with self._state_lock:
            if not self._accepting:
                return
            self._accepting = False
            futures = tuple(self._futures)
        if cancel_futures:
            for future in futures:
                future.cancel()
            self._reservations.wake()
        if wait:
            self._stop_controllers(cancel_futures=cancel_futures)
            for worker in self._workers.values():
                worker.shutdown(wait=True)
        else:
            # Workers must outlive controller tasks that have not dispatched
            # yet.  A daemon reaper preserves non-blocking shutdown without
            # placing a stop marker ahead of those tasks in a worker queue.
            self._controllers.shutdown(wait=False)

            def reap():
                self._stop_controllers(cancel_futures=cancel_futures)
                for worker in self._workers.values():
                    worker.shutdown(wait=True)

            threading.Thread(
                target=reap,
                name="egmgpu-shutdown",
                daemon=True,
            ).start()

    def _stop_controllers(self, cancel_futures):
        try:
            self._controllers.shutdown(wait=True, cancel_futures=cancel_futures)
        except TypeError:  # Python 3.8 compatibility.
            self._controllers.shutdown(wait=True)

    def _track_future(self, future):
        with self._state_lock:
            if not self._accepting:
                raise RuntimeClosedError("EGMGPU runtime is closed")
            self._futures.add(future)
            self._stats["submitted"] += 1

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
            if completed.cancelled():
                self._reservations.wake()

        future.add_done_callback(completed_callback)

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

    def _resolve_mode(self, function, requested_mode):
        if requested_mode is ExecutionMode.AUTO:
            # Until a measured cost-model adapter is installed, AUTO favors
            # request-level throughput and never silently invokes an
            # unvalidated cooperative implementation.
            return ExecutionMode.REQUEST_PARALLEL
        if requested_mode in {ExecutionMode.REPLICATED, ExecutionMode.PARTITIONED}:
            if (function, requested_mode) not in self._cooperative_builders:
                name = getattr(function, "__name__", repr(function))
                raise UnsupportedExecutionModeError(
                    "{} has no registered {} executor".format(name, requested_mode.value)
                )
        return requested_mode

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

    def _run_independent(
        self,
        future,
        function,
        args,
        kwargs,
        mode,
        preferred_devices,
        snapshot_lease,
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
                checked_function = function
                if (
                    snapshot_lease is not None
                    and not snapshot_lease.concurrent_execution_safe
                ):
                    snapshot_lease.assert_generation_current()

                    def checked_function(*call_args, **call_kwargs):
                        snapshot_lease.assert_generation_current()
                        result = function(*call_args, **call_kwargs)
                        snapshot_lease.assert_generation_current()
                        return result

                self._run_independent_reserved(
                    future,
                    checked_function,
                    args,
                    kwargs,
                    mode,
                    preferred_devices,
                )
        except BaseException as exc:
            if not future.done():
                try:
                    future.set_exception(exc)
                except concurrent.futures.InvalidStateError:
                    pass
        finally:
            if snapshot_lease is not None:
                snapshot_lease.release()

    def _run_independent_reserved(
        self, future, function, args, kwargs, mode, preferred_devices
    ):
        selected = self._reservations.acquire(
            1, preferred_devices, cancelled=future.cancelled
        )
        if selected is None:
            return
        device = selected[0]
        future._assign(selected, mode)
        if not future.set_running_or_notify_cancel():
            self._reservations.release(selected)
            return
        try:
            worker_future = self._workers[device].submit(
                lambda: function(*args, **kwargs)
            )
            result = worker_future.result()
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)
        finally:
            self._reservations.release(selected)

    def _run_cooperative(
        self,
        future,
        function,
        args,
        kwargs,
        mode,
        preferred_devices,
        builder,
        snapshot_lease,
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
                self._run_cooperative_reserved(
                    future,
                    function,
                    args,
                    kwargs,
                    mode,
                    preferred_devices,
                    builder,
                    snapshot_lease,
                )
        except BaseException as exc:
            if not future.done():
                try:
                    future.set_exception(exc)
                except concurrent.futures.InvalidStateError:
                    pass
        finally:
            if snapshot_lease is not None:
                snapshot_lease.release()

    def _run_cooperative_reserved(
        self,
        future,
        function,
        args,
        kwargs,
        mode,
        preferred_devices,
        builder,
        snapshot_lease,
    ):
        count = (
            len(preferred_devices)
            if preferred_devices is not None
            else len(self.config.devices)
        )
        if count < 2:
            future.set_exception(
                ConfigurationError("cooperative execution requires at least two devices")
            )
            return
        selected = self._reservations.acquire(
            count, preferred_devices, cancelled=future.cancelled
        )
        if selected is None:
            return
        future._assign(selected, mode)
        if not future.set_running_or_notify_cancel():
            self._reservations.release(selected)
            return
        plan = None
        result = None
        failure = None
        try:
            context = CooperativeContext(
                devices=selected,
                mode=mode,
                request_id=future.request_id,
                graph_snapshot=(
                    None if snapshot_lease is None else snapshot_lease.snapshot
                ),
            )
            plan = builder(context, function, args, kwargs)
            if not isinstance(plan, CooperativePlan):
                raise TypeError("cooperative executor must return CooperativePlan")
            plan.validate(selected)
            shard_futures = {
                device: self._workers[device].submit(plan.shards[device])
                for device in selected
            }
            shard_results = []
            for device in selected:
                try:
                    shard_results.append(shard_futures[device].result())
                except BaseException as exc:
                    for other in shard_futures.values():
                        other.cancel()
                    # A running shard cannot be cancelled by Future.cancel().
                    # Keep the complete gang reserved until every shard has
                    # stopped; otherwise a later collective could start on a
                    # subset of devices while an earlier shard is still live.
                    for other in shard_futures.values():
                        try:
                            other.result()
                        except BaseException:
                            pass
                    raise CooperativeExecutionError(device, exc) from exc
            result = plan.merge(shard_results)
        except BaseException as exc:
            failure = exc
        finally:
            if plan is not None and plan.cleanup is not None:
                try:
                    plan.cleanup()
                except BaseException as cleanup_error:
                    # Keep the operator failure when both execution and
                    # cleanup fail; otherwise surface cleanup corruption.
                    if failure is None:
                        failure = cleanup_error
            self._reservations.release(selected)
        if failure is not None:
            future.set_exception(failure)
        else:
            future.set_result(result)


# Descriptive name for new code.  The historical name remains available so
# existing CPU-only scheduler tests do not break.
CPUTestRuntime = EGMGPURuntime
