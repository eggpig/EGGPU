"""Explicit lifecycle adapter for the native EGMGPU worker runtime.

Importing this module and probing capabilities are side-effect free.  Native
workers are created only by an explicit lifecycle operation.  The production
Python control plane acquires this adapter with an opaque owner token; direct
``start``/``shutdown`` calls cannot then create or stop a second worker owner.
"""

import importlib
import sys
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Tuple


_REQUIRED_BINDINGS = (
    "cpp_egmgpu_probe_topology",
    "cpp_egmgpu_benchmark_transports",
    "cpp_egmgpu_runtime_start",
    "cpp_egmgpu_runtime_status",
    "cpp_egmgpu_runtime_ping",
    "cpp_egmgpu_runtime_shutdown",
)

# Every compiled binding instance reaches the same process-global C++
# RuntimeManager.  Adapter-local locks are therefore insufficient: two
# adapter objects could otherwise race through ``status`` and both believe
# they owned an idempotently started runtime.
_PROCESS_OWNER_LOCK = threading.RLock()
_PROCESS_OWNER_ADAPTER = None


class NativeRuntimeUnavailableError(RuntimeError):
    """The compiled extension does not provide the native EGMGPU runtime."""


class NativeRuntimeConflictError(RuntimeError):
    """Starting native workers would overlap another EGMGPU worker pool."""


@dataclass(frozen=True)
class NativeCapabilities:
    """Side-effect-free CUDA topology and native-runtime availability."""

    binding_available: bool
    runtime_available: bool
    diagnostic: str
    nccl_compiled: bool
    device_count: int
    devices: Tuple[Mapping[str, Any], ...]
    peer_access: Tuple[Tuple[bool, ...], ...]


def _load_bindings():
    try:
        module = importlib.import_module("cpp_easygraph")
    except ImportError as error:
        raise NativeRuntimeUnavailableError(
            "cpp_easygraph is not importable; build EasyGraph with "
            "EASYGRAPH_ENABLE_GPU=ON"
        ) from error
    missing = [name for name in _REQUIRED_BINDINGS if not hasattr(module, name)]
    if missing:
        raise NativeRuntimeUnavailableError(
            "cpp_easygraph has no native EGMGPU runtime bindings: {}".format(
                ", ".join(missing)
            )
        )
    return module


def _capabilities_from_result(result, binding_available=True):
    devices = tuple(dict(device) for device in result.get("devices", ()))
    peer_access = tuple(
        tuple(bool(value) for value in row)
        for row in result.get("peer_access", ())
    )
    return NativeCapabilities(
        binding_available=bool(binding_available),
        runtime_available=bool(result.get("runtime_available", False)),
        diagnostic=str(result.get("diagnostic", "")),
        nccl_compiled=bool(result.get("nccl_compiled", False)),
        device_count=int(result.get("device_count", len(devices))),
        devices=devices,
        peer_access=peer_access,
    )


def probe_native():
    """Probe the native bindings and CUDA topology without starting workers."""

    try:
        bindings = _load_bindings()
    except NativeRuntimeUnavailableError as error:
        return NativeCapabilities(
            binding_available=False,
            runtime_available=False,
            diagnostic=str(error),
            nccl_compiled=False,
            device_count=0,
            devices=(),
            peer_access=(),
        )
    return _capabilities_from_result(bindings.cpp_egmgpu_probe_topology())


def _python_scheduler_active():
    # Do not import api.py here: capability probing must remain free of runtime
    # initialization and package-level side effects.
    api_module = sys.modules.get("easygraph.egmgpu.api")
    if api_module is None:
        return False
    runtime = getattr(api_module, "_RUNTIME", None)
    return runtime is not None and not bool(getattr(runtime, "closed", False))


class NativeRuntimeAdapter:
    """Thin owner for the process-global native runtime bindings.

    Constructing an adapter is side-effect free.  ``start`` is deliberately
    explicit and may be called repeatedly only for an identical native
    configuration, as enforced by the C++ RuntimeManager.
    """

    def __init__(self, bindings=None):
        self._bindings = bindings
        self._lock = threading.RLock()
        self._owner_token = None
        self._direct_owner_token = object()

    def _require_bindings(self):
        if self._bindings is None:
            self._bindings = _load_bindings()
        missing = [
            name for name in _REQUIRED_BINDINGS if not hasattr(self._bindings, name)
        ]
        if missing:
            raise NativeRuntimeUnavailableError(
                "native EGMGPU binding set is incomplete: {}".format(
                    ", ".join(missing)
                )
            )
        return self._bindings

    def capabilities(self):
        """Return topology capabilities without starting native workers."""

        with self._lock:
            bindings = self._require_bindings()
            return _capabilities_from_result(bindings.cpp_egmgpu_probe_topology())

    def status(self):
        """Return native runtime state without creating it."""

        with self._lock:
            bindings = self._require_bindings()
            return dict(bindings.cpp_egmgpu_runtime_status())

    def benchmark_transports(
        self,
        devices=None,
        *,
        bandwidth_bytes=64 << 20,
        bandwidth_iterations=20,
        latency_iterations=1000,
        warmup_iterations=3,
    ):
        """Explicitly measure P2P and pinned-host transport characteristics.

        Unlike :meth:`capabilities`, this operation allocates CUDA memory,
        enables supported peer paths, and synchronizes measured streams.  It
        is never invoked automatically during import or configuration.
        """

        with self._lock:
            bindings = self._require_bindings()
            normalized = (
                None if devices is None else [int(device) for device in devices]
            )
            return dict(
                bindings.cpp_egmgpu_benchmark_transports(
                    normalized,
                    int(bandwidth_bytes),
                    int(bandwidth_iterations),
                    int(latency_iterations),
                    int(warmup_iterations),
                )
            )

    @property
    def running(self):
        return bool(self.status().get("running", False))

    def start(self, devices=None, backend="auto"):
        """Start fixed-device native workers under direct lifecycle ownership."""

        with _PROCESS_OWNER_LOCK:
            with self._lock:
                if _python_scheduler_active():
                    raise NativeRuntimeConflictError(
                        "the Python EGMGPU scheduler is active; shut it down before "
                        "starting native workers"
                    )
                return self._start_owned(
                    self._direct_owner_token, devices=devices, backend=backend
                )

    def claim(self, owner, devices=None, backend="auto"):
        """Acquire the process-global native workers for one control plane.

        ``owner`` is compared by identity.  It is intentionally opaque so an
        unrelated scheduler cannot accidentally release the active workers.
        This method is the only startup path used by ``NativeOwnedRuntime``.
        """

        if owner is None:
            raise ValueError("native runtime owner token cannot be None")
        with _PROCESS_OWNER_LOCK:
            with self._lock:
                return self._start_owned(owner, devices=devices, backend=backend)

    def _start_owned(self, owner, devices=None, backend="auto"):
        global _PROCESS_OWNER_ADAPTER
        if (
            _PROCESS_OWNER_ADAPTER is not None
            and _PROCESS_OWNER_ADAPTER is not self
        ):
            raise NativeRuntimeConflictError(
                "the process-global native EGMGPU runtime belongs to another adapter"
            )
        if self._owner_token is not None and self._owner_token is not owner:
            raise NativeRuntimeConflictError(
                "the native EGMGPU worker runtime already has another owner"
            )
        bindings = self._require_bindings()
        if self._owner_token is None:
            status = dict(bindings.cpp_egmgpu_runtime_status())
            if bool(status.get("running", False)):
                raise NativeRuntimeConflictError(
                    "native EGMGPU workers are already running outside this adapter owner"
                )
        normalized = None if devices is None else [int(device) for device in devices]
        result = dict(bindings.cpp_egmgpu_runtime_start(normalized, str(backend)))
        self._owner_token = owner
        _PROCESS_OWNER_ADAPTER = self
        return result

    def ping(self):
        """Run a device-affinity check on every active native worker."""

        with self._lock:
            bindings = self._require_bindings()
            return tuple(int(device) for device in bindings.cpp_egmgpu_runtime_ping())

    def shutdown(self, drain=True):
        """Stop directly-owned native workers; repeated shutdown is safe."""

        with _PROCESS_OWNER_LOCK:
            with self._lock:
                if self._owner_token is None:
                    return
                if self._owner_token is not self._direct_owner_token:
                    raise NativeRuntimeConflictError(
                        "native workers belong to the production EGMGPU control plane"
                    )
                self._release_owned(self._direct_owner_token, drain=drain)

    def release(self, owner, drain=True):
        """Release workers acquired by :meth:`claim`."""

        if owner is None:
            raise ValueError("native runtime owner token cannot be None")
        with _PROCESS_OWNER_LOCK:
            with self._lock:
                self._release_owned(owner, drain=drain)

    def _release_owned(self, owner, drain=True):
        global _PROCESS_OWNER_ADAPTER
        if self._owner_token is None:
            return
        if self._owner_token is not owner:
            raise NativeRuntimeConflictError(
                "only the native EGMGPU worker owner may release the runtime"
            )
        bindings = self._require_bindings()
        bindings.cpp_egmgpu_runtime_shutdown(bool(drain))
        self._owner_token = None
        if _PROCESS_OWNER_ADAPTER is self:
            _PROCESS_OWNER_ADAPTER = None

    def __enter__(self):
        if not self.running:
            self.start()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.shutdown()
        return False


_DEFAULT_ADAPTER = NativeRuntimeAdapter()


def native_adapter():
    """Return the process-level adapter without starting native workers."""

    return _DEFAULT_ADAPTER


def use_native(devices=None, backend="auto"):
    """Explicitly start and return the process-level native runtime adapter."""

    _DEFAULT_ADAPTER.start(devices=devices, backend=backend)
    return _DEFAULT_ADAPTER


def shutdown_native(drain=True):
    """Stop the process-level native runtime if its bindings are available."""

    try:
        _DEFAULT_ADAPTER.shutdown(drain=drain)
    except NativeRuntimeUnavailableError:
        return
