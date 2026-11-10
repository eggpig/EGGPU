"""Public process-level EGMGPU API.

Configuration is lazy: importing this module, building ``CallSpec`` objects,
registering native adapters, and calling :func:`configure` create no worker
threads.  The first submitted request acquires the process-global C++
``RuntimeManager`` through :class:`NativeOwnedRuntime`.
"""

import os
import threading

from .deployment_planner import DeploymentPlanner
from .production_runtime import NativeOwnedRuntime
from .runtime import ConfigurationError
from .runtime import EGMGPURuntime
from .runtime import RuntimeClosedError
from .runtime import UnsupportedExecutionModeError
from .runtime import cuda_device_count
from .types import CallSpec
from .types import ExecutionMode
from .types import RequestSchedulingPolicy
from .types import RuntimeConfig


_CONFIG = None
_RUNTIME = None
_SNAPSHOT_PROVIDER = None
_DEPLOYMENT_PLANNER = None
_NATIVE_EXECUTORS = {}
_SHUTDOWN_EVENT = None
_SHUTDOWN_FAILURE = None
_RUNTIME_LOCK = threading.RLock()
_DIRECT_DISPATCH_LOCAL = threading.local()
_runtime_factory = NativeOwnedRuntime


def _direct_graph_argument(args, kwargs):
    if args:
        return args[0]
    return kwargs.get("G")


def _mutable_snapshot_graph(graph):
    return bool(
        graph is not None
        and not getattr(graph, "_eggpu_bulk_csr", False)
        and hasattr(graph, "_node")
        and hasattr(graph, "_adj")
        and hasattr(graph, "_mutation_generation")
        and callable(getattr(graph, "number_of_nodes", None))
        and callable(getattr(graph, "number_of_edges", None))
        and hasattr(graph, "is_directed")
    )


def _guard_native_executor(executor):
    """Mark controller-side native execution for direct-entry recursion safety."""

    if bool(getattr(executor, "_egmgpu_direct_reentry_guard", False)):
        return executor

    def guarded(context, args, kwargs):
        previous = bool(
            getattr(_DIRECT_DISPATCH_LOCAL, "inside_native_executor", False)
        )
        _DIRECT_DISPATCH_LOCAL.inside_native_executor = True
        try:
            return executor(context, args, kwargs)
        finally:
            _DIRECT_DISPATCH_LOCAL.inside_native_executor = previous

    guarded._egmgpu_direct_reentry_guard = True
    guarded._egmgpu_original_executor = executor
    return guarded


def _direct_dispatch_eligible(function, args, kwargs):
    """Return whether one existing synchronous API call may be redirected.

    This predicate is side-effect free: in particular, it never calls
    :func:`configure`, creates a runtime, probes CUDA, or acquires a graph
    snapshot.  Read-only bulk CSR graphs and generation-tracked EasyGraph
    objects are eligible graph boundaries.  The latter are copied into a
    private native snapshot before execution.
    """

    from .direct_dispatch import is_direct_public_function
    from .compatibility import is_production_executor

    if not is_direct_public_function(function):
        return False
    if bool(getattr(_DIRECT_DISPATCH_LOCAL, "inside_native_executor", False)):
        return False
    graph = _direct_graph_argument(args, kwargs)
    bulk = bool(getattr(graph, "_eggpu_bulk_csr", False))
    mutable = _mutable_snapshot_graph(graph)
    if not bulk and not mutable:
        return False
    with _RUNTIME_LOCK:
        if (
            _CONFIG is None
            or _SHUTDOWN_EVENT is not None
            or _SHUTDOWN_FAILURE is not None
        ):
            return False
        requested_mode = _CONFIG.mode
        registered = {
            mode: executor
            for (registered_function, mode), executor in _NATIVE_EXECUTORS.items()
            if registered_function is function
        }
        registered_modes = set(registered)
        if requested_mode is not ExecutionMode.AUTO:
            return requested_mode in registered_modes
        production_modes = {
            mode
            for mode, executor in registered.items()
            if is_production_executor(executor)
        }
        if mutable:
            return bool(
                production_modes.intersection(
                    {ExecutionMode.REQUEST_PARALLEL, ExecutionMode.SINGLE}
                )
            )
        if _DEPLOYMENT_PLANNER is not None:
            return bool(production_modes)
        return bool(
            production_modes.intersection(
                {
                    ExecutionMode.REQUEST_PARALLEL,
                    ExecutionMode.SINGLE,
                }
            )
        )


def _parse_device_list(value):
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "auto":
            return _detect_devices()
        if not stripped:
            raise ConfigurationError("EGMGPU device list cannot be empty")
        return tuple(int(item.strip()) for item in stripped.split(","))
    return tuple(int(device) for device in value)


def _detect_devices():
    configured = os.environ.get("EASYGRAPH_EGMGPU_DEVICES")
    if configured and configured.strip().lower() != "auto":
        return _parse_device_list(configured)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if not entries or entries == ["-1"]:
            raise ConfigurationError("no CUDA devices are visible to EGMGPU")
        # cudaSetDevice consumes process-visible logical identifiers.
        return tuple(range(len(entries)))

    count = cuda_device_count()
    if count <= 0:
        raise ConfigurationError("no CUDA devices are visible to EGMGPU")
    return tuple(range(count))


def configure(
    devices="auto",
    mode=None,
    min_predicted_gain=0.05,
    scheduling_policy=RequestSchedulingPolicy.FIFO.value,
    snapshot_provider=None,
    deployment_planner=None,
    deployment=None,
    calibration=None,
    cooperative_capabilities=(),
    workload_estimator=None,
    residency_provider=None,
    reservation_policy="fifo",
):
    """Store process configuration without starting scheduler workers.

    Repeating the same configuration is idempotent.  A different
    configuration requires :func:`shutdown` first so active requests are
    never cancelled implicitly.  Native workers start only on the first
    :func:`submit` or :func:`run_workflow`.  ``reservation_policy`` selects
    FIFO or conflict-safe bypass after a request has a placement; it does not
    select a device group or require an adaptive deployment planner.
    """

    if mode is None:
        mode = os.environ.get("EASYGRAPH_EGMGPU_MODE", ExecutionMode.AUTO.value)
    config = RuntimeConfig(
        devices=_parse_device_list(devices),
        mode=ExecutionMode.parse(mode),
        min_predicted_gain=min_predicted_gain,
        scheduling_policy=RequestSchedulingPolicy.parse(scheduling_policy),
        reservation_policy=reservation_policy,
    )
    capability_items = tuple(cooperative_capabilities)
    has_planning_parts = (
        deployment is not None
        or calibration is not None
        or bool(capability_items)
        or workload_estimator is not None
        or residency_provider is not None
    )
    if deployment_planner is not None and has_planning_parts:
        raise ConfigurationError(
            "provide deployment_planner or its explicit construction inputs, not both"
        )
    if deployment_planner is None and deployment is not None:
        deployment_planner = DeploymentPlanner(
            deployment,
            calibration=calibration,
            cooperative_capabilities=capability_items,
            workload_estimator=workload_estimator,
            residency_provider=residency_provider,
        )
    elif deployment_planner is None and has_planning_parts:
        raise ConfigurationError(
            "adaptive planning inputs require an explicit MeasuredDeployment"
        )
    if deployment_planner is not None:
        planner_deployment = getattr(deployment_planner, "deployment", None)
        if planner_deployment is None or not hasattr(planner_deployment, "devices"):
            raise ConfigurationError(
                "deployment_planner must expose its measured deployment"
            )
        planned_devices = tuple(
            device.device_id for device in planner_deployment.devices
        )
        if planned_devices != config.devices:
            raise ConfigurationError(
                "deployment planner devices {} do not match configured devices {}".format(
                    planned_devices, config.devices
                )
            )
    if (
        config.scheduling_policy is not RequestSchedulingPolicy.FIFO
        and deployment_planner is None
    ):
        raise ConfigurationError(
            "{} scheduling requires a measured deployment planner for memory "
            "feasibility and predicted queue cost".format(
                config.scheduling_policy.value
            )
        )

    global _CONFIG, _SNAPSHOT_PROVIDER, _DEPLOYMENT_PLANNER
    with _RUNTIME_LOCK:
        if _SHUTDOWN_EVENT is not None:
            raise ConfigurationError(
                "EGMGPU runtime shutdown is still in progress"
            )
        if _SHUTDOWN_FAILURE is not None:
            raise ConfigurationError(
                "the previous EGMGPU runtime shutdown failed: {}".format(
                    _SHUTDOWN_FAILURE
                )
            )
        if _CONFIG is not None:
            if (
                _CONFIG == config
                and _SNAPSHOT_PROVIDER is snapshot_provider
                and _DEPLOYMENT_PLANNER is deployment_planner
            ):
                return _CONFIG
            raise ConfigurationError(
                "EGMGPU is already configured; call eg.egmgpu.shutdown() "
                "before reconfiguration"
            )
        _CONFIG = config
        _SNAPSHOT_PROVIDER = snapshot_provider
        _DEPLOYMENT_PLANNER = deployment_planner
        return _CONFIG


def _get_runtime():
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _SHUTDOWN_EVENT is not None:
            raise RuntimeClosedError("EGMGPU runtime shutdown is in progress")
        if _SHUTDOWN_FAILURE is not None:
            raise RuntimeClosedError(
                "the previous EGMGPU runtime shutdown failed: {}".format(
                    _SHUTDOWN_FAILURE
                )
            )
        if _CONFIG is None:
            # The lock is reentrant.  Auto-configuration probes device count
            # but does not start either Python or native worker threads.
            configure()
        runtime = _RUNTIME
        if runtime is None:
            candidate = None
            try:
                runtime_kwargs = {"snapshot_provider": _SNAPSHOT_PROVIDER}
                if _DEPLOYMENT_PLANNER is not None:
                    runtime_kwargs["deployment_planner"] = _DEPLOYMENT_PLANNER
                candidate = _runtime_factory(_CONFIG, **runtime_kwargs)
                register = getattr(candidate, "register_native_executor", None)
                if _NATIVE_EXECUTORS and register is None:
                    raise ConfigurationError(
                        "the selected runtime does not accept native executors"
                    )
                if register is not None:
                    for (function, mode), executor in _NATIVE_EXECUTORS.items():
                        register(function, mode, executor)
            except BaseException:
                if candidate is not None:
                    try:
                        candidate.shutdown(wait=True, cancel_futures=True)
                    except BaseException:
                        # Preserve the construction/registration failure.  A
                        # best-effort shutdown already owns its cleanup path.
                        pass
                raise
            _RUNTIME = candidate
            runtime = candidate
        if runtime.closed:
            raise RuntimeClosedError("EGMGPU runtime is not accepting requests")
        return runtime


def call(function, *args, **kwargs):
    """Create one call in an explicit EGMGPU workflow.

    ``mode`` and ``devices`` are scheduler options.  Remaining keyword
    arguments are forwarded unchanged to the graph function.  This operation
    never initializes the runtime.
    """

    mode = kwargs.pop("mode", None)
    devices = kwargs.pop("devices", None)
    return CallSpec(function, tuple(args), kwargs, mode=mode, devices=devices)


def configure_joint_window(profile, *, accounting="union", allow_synthetic=False, observed_binding=None):
    """Explicit isolated profile-driven finite-window experiment."""
    return _get_runtime().configure_joint_window(profile, accounting=accounting,
                                                allow_synthetic=allow_synthetic, observed_binding=observed_binding)


def submit_window(calls):
    """Submit up to four preprofiled calls through the native control plane."""
    return _get_runtime().submit_window(calls)


def submit(function, *args, **kwargs):
    """Submit a graph function and return an :class:`EGMGPUFuture`.

    This is the first operation that may start the native worker runtime.
    Functions have no production execution path until a native executor is
    registered explicitly; registration alone does not start workers.
    """

    mode = kwargs.pop("mode", None)
    devices = kwargs.pop("devices", None)
    if mode is not None:
        kwargs["_egmgpu_mode"] = mode
    if devices is not None:
        kwargs["_egmgpu_devices"] = devices
    _preflight_default_native(function, mode)
    return _get_runtime().submit(function, *args, **kwargs)


def _submit_direct_configured(function, *args, **kwargs):
    """Submit one direct wrapper call without implicit auto-configuration.

    Eligibility and runtime acquisition are separate operations.  Holding the
    process lock across this second check prevents a concurrent shutdown from
    turning an explicitly configured direct call into the auto-configuring
    behavior intentionally retained by the explicit :func:`submit` API.
    """

    mode = kwargs.pop("mode", None)
    devices = kwargs.pop("devices", None)
    if mode is not None:
        kwargs["_egmgpu_mode"] = mode
    if devices is not None:
        kwargs["_egmgpu_devices"] = devices
    with _RUNTIME_LOCK:
        if _CONFIG is None:
            raise RuntimeClosedError(
                "EGMGPU direct dispatch requires explicit configure()"
            )
        if _SHUTDOWN_EVENT is not None:
            raise RuntimeClosedError("EGMGPU runtime shutdown is in progress")
        if _SHUTDOWN_FAILURE is not None:
            raise RuntimeClosedError(
                "the previous EGMGPU runtime shutdown failed: {}".format(
                    _SHUTDOWN_FAILURE
                )
            )
        graph = _direct_graph_argument(args, kwargs)
        if (
            mode is None
            and _CONFIG.mode is ExecutionMode.AUTO
            and _mutable_snapshot_graph(graph)
        ):
            # Cooperative bindings are not qualified for private EasyGraph
            # snapshots.  Keep AUTO on an exact one-device route even when an
            # adaptive deployment planner is active.
            for candidate in (
                ExecutionMode.REQUEST_PARALLEL,
                ExecutionMode.SINGLE,
            ):
                if (function, candidate) in _NATIVE_EXECUTORS:
                    mode = candidate
                    kwargs["_egmgpu_mode"] = candidate
                    break
        _preflight_default_native(function, mode)
        runtime = _get_runtime()
    return runtime.submit(function, *args, **kwargs)


def run_workflow(graph, calls, fail_fast=True):
    """Run workflow calls concurrently and return results in input order."""

    specs = tuple(calls)
    for spec in specs:
        if not isinstance(spec, CallSpec):
            raise TypeError("run_workflow expects CallSpec objects created by egmgpu.call")
        _preflight_default_native(spec.function, spec.mode)
    return _get_runtime().run_workflow(graph, specs, fail_fast=fail_fast)


def _preflight_default_native(function, requested_mode):
    """Reject unwired production calls before acquiring native workers."""

    with _RUNTIME_LOCK:
        if _RUNTIME is not None or _runtime_factory is not NativeOwnedRuntime:
            return
        if _CONFIG is None:
            configure()
        mode = _CONFIG.mode if requested_mode is None else ExecutionMode.parse(requested_mode)
        if mode is ExecutionMode.AUTO:
            registered = any(
                (function, candidate) in _NATIVE_EXECUTORS
                for candidate in (
                    tuple(ExecutionMode)
                    if _DEPLOYMENT_PLANNER is not None
                    else (
                        ExecutionMode.REQUEST_PARALLEL,
                        ExecutionMode.SINGLE,
                    )
                )
                if candidate is not ExecutionMode.AUTO
            )
        else:
            registered = (function, mode) in _NATIVE_EXECUTORS
        if not registered:
            name = getattr(function, "__name__", repr(function))
            raise UnsupportedExecutionModeError(
                "{} has no registered native {} executor".format(name, mode.value)
            )


def register_native_executor(function, mode, executor):
    """Register one implemented native execution path without starting it.

    ``executor`` runs on a Python controller thread.  It may parse Python
    arguments and invoke a pybind entry point, but the C++ entry point must
    submit only native values and native buffers to device workers.  A
    registration is wiring evidence, not a claim of measured performance.
    """

    if not callable(function) or not callable(executor):
        raise TypeError("function and native executor must be callable")
    from .compatibility import is_production_executor

    if is_production_executor(executor):
        raise ConfigurationError(
            "production-attested executors can only be published by the "
            "strict atomic bootstrap"
        )
    parsed_mode = ExecutionMode.parse(mode)
    if parsed_mode is ExecutionMode.AUTO:
        raise ValueError("AUTO is a selection policy, not a native executor")
    guarded = _guard_native_executor(executor)
    with _RUNTIME_LOCK:
        _NATIVE_EXECUTORS[(function, parsed_mode)] = guarded
        runtime = _RUNTIME
        if runtime is not None:
            runtime.register_native_executor(function, parsed_mode, guarded)


def _register_native_executors_atomic(registrations):
    """Publish an already validated binding set without partial visibility."""

    normalized = []
    keys = set()
    for function, mode, executor in tuple(registrations):
        if not callable(function) or not callable(executor):
            raise TypeError("function and native executor must be callable")
        parsed_mode = ExecutionMode.parse(mode)
        if parsed_mode is ExecutionMode.AUTO:
            raise ValueError("AUTO is a selection policy, not a native executor")
        key = (function, parsed_mode)
        if key in keys:
            raise ValueError("duplicate native executor registration")
        keys.add(key)
        normalized.append(
            (function, parsed_mode, _guard_native_executor(executor))
        )
    with _RUNTIME_LOCK:
        runtime = _RUNTIME
        if runtime is not None:
            register_many = getattr(
                runtime, "register_native_executors_atomic", None
            )
            if not callable(register_many):
                raise ConfigurationError(
                    "the active runtime cannot atomically install native bindings"
                )
            register_many(normalized)
        _NATIVE_EXECUTORS.update(
            ((function, mode), executor)
            for function, mode, executor in normalized
        )


def unregister_native_executor(function, mode):
    """Remove a native path registration without initializing the runtime."""

    parsed_mode = ExecutionMode.parse(mode)
    with _RUNTIME_LOCK:
        _NATIVE_EXECUTORS.pop((function, parsed_mode), None)
        runtime = _RUNTIME
        if runtime is not None:
            runtime.unregister_native_executor(function, parsed_mode)


def register_cooperative_executor(function, mode, builder):
    """Register a Python-shard builder on an explicit CPU-test runtime.

    ``CooperativePlan`` contains Python callbacks and is therefore forbidden
    on production native workers.  Tests that need it must construct
    :class:`EGMGPURuntime` directly (or inject that runtime factory) before
    registration.
    """

    with _RUNTIME_LOCK:
        runtime = _RUNTIME
    if runtime is None or not isinstance(runtime, EGMGPURuntime):
        raise ConfigurationError(
            "Python cooperative executors are CPU-test-only; construct an "
            "EGMGPURuntime test scheduler explicitly"
        )
    runtime.register_cooperative_executor(function, mode, builder)


def unregister_cooperative_executor(function, mode):
    """Remove a Python-shard builder from an explicit CPU-test runtime."""

    with _RUNTIME_LOCK:
        runtime = _RUNTIME
    if runtime is None or not isinstance(runtime, EGMGPURuntime):
        raise ConfigurationError(
            "Python cooperative executors are CPU-test-only; no test runtime is active"
        )
    runtime.unregister_cooperative_executor(function, mode)


def stats():
    """Return a side-effect-free snapshot of scheduler state and counters."""

    with _RUNTIME_LOCK:
        runtime = _RUNTIME
        config = _CONFIG
        shutting_down = _SHUTDOWN_EVENT is not None
        registered = tuple(
            sorted(
                (getattr(function, "__name__", repr(function)), mode.value)
                for function, mode in _NATIVE_EXECUTORS
            )
        )
        planner = _DEPLOYMENT_PLANNER
    if runtime is not None:
        return runtime.stats()
    return {
        "configured": config is not None,
        "started": False,
        "accepting": config is not None and not shutting_down,
        "shutting_down": shutting_down,
        "worker_owner": "native" if config is not None else None,
        "devices": () if config is None else config.devices,
        "mode": None if config is None else config.mode.value,
        "scheduling_policy": (
            None if config is None else config.scheduling_policy.value
        ),
        "live_futures": 0,
        "totals": {},
        "workers": (),
        "busy_devices": (),
        "waiting_reservations": 0,
        "registered_native_executors": registered,
        "snapshots": {
            "graphs": 0,
            "snapshots": 0,
            "active_leases": 0,
            "stale_snapshots": 0,
            "native_snapshots": 0,
        },
        "deployment_planner": (
            None if planner is None else dict(planner.stats())
        ),
    }


def shutdown(wait=True, cancel_futures=False):
    """Stop the process-local runtime; repeated calls are safe."""

    global _CONFIG, _RUNTIME, _SHUTDOWN_EVENT, _SHUTDOWN_FAILURE
    global _SNAPSHOT_PROVIDER, _DEPLOYMENT_PLANNER
    owns_shutdown = False
    with _RUNTIME_LOCK:
        event = _SHUTDOWN_EVENT
        if event is None:
            runtime = _RUNTIME
            _RUNTIME = None
            if runtime is None:
                _CONFIG = None
                _SNAPSHOT_PROVIDER = None
                _DEPLOYMENT_PLANNER = None
                if _SHUTDOWN_FAILURE is not None and wait:
                    raise RuntimeClosedError(
                        "EGMGPU runtime shutdown failed: {}".format(
                            _SHUTDOWN_FAILURE
                        )
                    ) from _SHUTDOWN_FAILURE
                return
            _SHUTDOWN_FAILURE = None
            event = threading.Event()
            _SHUTDOWN_EVENT = event
            owns_shutdown = True

    if not owns_shutdown:
        if wait:
            event.wait()
            with _RUNTIME_LOCK:
                failure = _SHUTDOWN_FAILURE
            if failure is not None:
                raise RuntimeClosedError(
                    "EGMGPU runtime shutdown failed: {}".format(failure)
                ) from failure
        return

    def stop_runtime():
        global _CONFIG, _SHUTDOWN_EVENT, _SHUTDOWN_FAILURE
        global _SNAPSHOT_PROVIDER, _DEPLOYMENT_PLANNER
        failure = None
        try:
            # The API owns asynchronous behavior.  The runtime itself always
            # reaches a completed native release before this event is set.
            runtime.shutdown(wait=True, cancel_futures=cancel_futures)
        except BaseException as exc:
            failure = exc
        finally:
            with _RUNTIME_LOCK:
                _CONFIG = None
                _SNAPSHOT_PROVIDER = None
                _DEPLOYMENT_PLANNER = None
                _SHUTDOWN_FAILURE = failure
                if _SHUTDOWN_EVENT is event:
                    _SHUTDOWN_EVENT = None
                event.set()
        if failure is not None and wait:
            raise failure

    if wait:
        stop_runtime()
    else:
        threading.Thread(
            target=stop_runtime,
            name="egmgpu-api-shutdown",
            daemon=True,
        ).start()
