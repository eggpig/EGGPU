#!/usr/bin/env python3
"""Device-interval timing for strict nx-cugraph public calls.

The public NetworkX call includes backend dispatch, optional graph conversion,
host-side result construction, and the GPU algorithm.  CUDA events placed
around that entire synchronous call are not a valid device timer because the
stop event is only enqueued after host-side work completes.  This module
instead wraps the concrete ``pylibcugraph`` algorithm entry points reached by
nx-cugraph and records events on the same explicit RAFT/CUDA stream.

The resulting interval is the backend device-execution boundary used by the
paper's ``Processing time`` panel.  Public-call wall time remains a separate
end-to-end metric.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass


DEFAULT_BACKEND_FUNCTIONS = (
    "pagerank",
    "personalized_pagerank",
    "triangle_count",
    "weakly_connected_components",
    "bfs",
    "sssp",
    "core_number",
)


@dataclass(frozen=True)
class DeviceInterval:
    backend: str
    seconds: float


class _BoundHandleFactory:
    """Create pylibcugraph handles backed by one explicit RAFT stream."""

    def __init__(self, original, raft_handle):
        self._original = original
        self._raft_handle = raft_handle
        self.calls = 0

    def __call__(self, handle=None):
        self.calls += 1
        if handle is not None:
            return self._original(handle)
        return self._original(self._raft_handle.getHandle())


class NxCugraphDeviceTimer:
    """Patch pylibcugraph within a worker and collect device intervals.

    Use one instance per isolated benchmark worker.  The context manager
    restores every patched symbol even when a backend call raises.
    """

    def __init__(self, backend_functions=DEFAULT_BACKEND_FUNCTIONS):
        import cupy as cp
        import pylibcugraph as plc
        from pylibraft.common import Handle

        self.cp = cp
        self.plc = plc
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.raft_handle = Handle(stream=self.stream.ptr, n_streams=0)
        self.backend_functions = tuple(backend_functions)
        self._original_resource_handle = plc.ResourceHandle
        self._handle_factory = _BoundHandleFactory(
            self._original_resource_handle, self.raft_handle
        )
        self._original_backend_functions = {}
        self._installed = False
        self._intervals = []
        self._interval_resource_handle_start = 0

    @property
    def resource_handle_calls(self):
        return int(self._handle_factory.calls)

    @property
    def intervals(self):
        return tuple(self._intervals)

    @property
    def total_seconds(self):
        return float(sum(interval.seconds for interval in self._intervals))

    def reset(self):
        self._intervals.clear()
        self._interval_resource_handle_start = self.resource_handle_calls

    def synchronize(self):
        self.raft_handle.sync()

    def call(self, callable_obj):
        """Run a public API call with the explicit stream current."""

        with self.stream:
            return callable_obj()

    def install(self):
        if self._installed:
            raise RuntimeError("nx-cugraph device timer is already installed")
        self.plc.ResourceHandle = self._handle_factory
        for name in self.backend_functions:
            original = getattr(self.plc, name, None)
            if original is None:
                continue
            self._original_backend_functions[name] = original

            @functools.wraps(original)
            def wrapped(*args, __name=name, __original=original, **kwargs):
                start = self.cp.cuda.Event()
                stop = self.cp.cuda.Event()
                start.record(self.stream)
                result = __original(*args, **kwargs)
                stop.record(self.stream)
                stop.synchronize()
                seconds = self.cp.cuda.get_elapsed_time(start, stop) / 1000.0
                if not math.isfinite(seconds) or seconds <= 0.0:
                    raise RuntimeError(
                        f"invalid nx-cugraph {__name} device interval: {seconds!r}"
                    )
                self._intervals.append(DeviceInterval(__name, float(seconds)))
                return result

            setattr(self.plc, name, wrapped)
        self._installed = True
        return self

    def restore(self):
        if not self._installed:
            return
        for name, original in self._original_backend_functions.items():
            setattr(self.plc, name, original)
        self._original_backend_functions.clear()
        self.plc.ResourceHandle = self._original_resource_handle
        self.raft_handle.sync()
        self._installed = False

    def validate_against_public_wall(self, public_wall_seconds):
        public_wall_seconds = float(public_wall_seconds)
        device_seconds = self.total_seconds
        if not math.isfinite(public_wall_seconds) or public_wall_seconds <= 0.0:
            raise RuntimeError(
                f"invalid nx-cugraph public wall time: {public_wall_seconds!r}"
            )
        if not self._intervals:
            raise RuntimeError(
                "nx-cugraph public call reached no instrumented "
                "pylibcugraph backend function"
            )
        if self.interval_resource_handle_calls <= 0:
            raise RuntimeError(
                "nx-cugraph public call did not create a ResourceHandle "
                "bound to the explicit timing stream"
            )
        tolerance = max(1.0e-6, public_wall_seconds * 1.0e-4)
        if device_seconds > public_wall_seconds + tolerance:
            raise RuntimeError(
                "nx-cugraph device interval exceeds public wall time: "
                f"device={device_seconds:.9g}, public={public_wall_seconds:.9g}"
            )
        return device_seconds

    @property
    def interval_resource_handle_calls(self):
        return int(
            self.resource_handle_calls - self._interval_resource_handle_start
        )

    def validate_provenance(
        self,
        *,
        expected_backends,
        expected_interval_count,
    ):
        """Reject a timing sample whose backend boundary is ambiguous.

        The benchmark has a fixed semantic adapter for each supported public
        NetworkX function.  A successful sample is admissible only if it
        reaches exactly the expected pylibcugraph algorithm entry points and
        every invocation is observed.
        """

        expected = set(expected_backends)
        observed = {interval.backend for interval in self._intervals}
        if observed != expected:
            raise RuntimeError(
                "unexpected nx-cugraph backend boundary: "
                f"expected={sorted(expected)}, observed={sorted(observed)}"
            )
        if len(self._intervals) != int(expected_interval_count):
            raise RuntimeError(
                "unexpected nx-cugraph backend interval count: "
                f"expected={int(expected_interval_count)}, "
                f"observed={len(self._intervals)}"
            )
        if self.interval_resource_handle_calls < len(self._intervals):
            raise RuntimeError(
                "fewer explicit-stream ResourceHandle constructions than "
                "instrumented backend invocations"
            )
        return self.provenance()

    def provenance(self):
        return {
            "timer": "cuda_events_at_pylibcugraph_backend_boundary",
            "stream": "explicit_cupy_stream_bound_to_pylibraft_handle",
            "backend_functions": sorted(
                {interval.backend for interval in self._intervals}
            ),
            "backend_interval_count": len(self._intervals),
            "resource_handle_factory_calls": self.interval_resource_handle_calls,
            "resource_handle_factory_calls_process_total": (
                self.resource_handle_calls
            ),
            "validation": "device_interval_nonzero_and_not_above_public_wall",
        }

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, traceback):
        self.restore()
        return False
