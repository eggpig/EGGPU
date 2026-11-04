"""Isolated, opt-in lifecycle remedies; production runtime is untouched.

Deferred kernels charge compilation/loading at their first actual invocation.
The arena backend owns one budgeted allocation across sequential sessions. Its
retained physical reservation is explicit, even while no session owns a lease.
"""
from __future__ import annotations

import threading
import time

import cupy as cp


class DeferredKernel:
    def __init__(self, module, name, record=None):
        self.module = module
        self.name = name
        self.kernel = None
        self.record = record
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        if self.kernel is None:
            with self.lock:
                if self.kernel is None:
                    start = time.perf_counter()
                    self.kernel = self.module.get_function(self.name)
                    if self.record:
                        self.record('deferred_kernel_resolution', time.perf_counter()-start,
                                    function=self.name, kernel_id=id(self.kernel))
        return self.kernel(*args, **kwargs)


class DeferredModule:
    def __init__(self, module, record=None):
        self.module = module
        self.record = record

    def get_function(self, name):
        return DeferredKernel(self.module, name, self.record)


class SequentialArenaBackend:
    """One fixed-capacity CUDA allocation with explicit sequential leases.

    Graph arrays and Python views belong to the lease and must be dropped before
    release. At most one lease is active: concurrent callers share the backend
    through the existing session lock/queue. A caller creating a second session
    before closing the first receives an error, never an unbudgeted allocation.
    """
    def __init__(self, budget_bytes, memory_factory=None, record=None):
        self.budget_bytes = int(budget_bytes)
        if self.budget_bytes < 256 or self.budget_bytes % 256:
            raise ValueError('budget must be a positive multiple of 256 bytes')
        self.memory_factory = memory_factory or cp.cuda.memory.Memory
        self.record = record
        self.memory = None
        self.active = False
        self.closed = False
        self.borrow_count = 0
        self.lock = threading.RLock()

    @property
    def reserved_bytes(self):
        return self.budget_bytes if self.memory is not None else 0

    def borrow(self, size):
        with self.lock:
            if self.closed:
                raise RuntimeError('backend is closed')
            if self.active:
                raise RuntimeError('backend budget already leased to an active session')
            if int(size) != self.budget_bytes:
                raise ValueError('session budget must equal backend fixed reservation')
            start = time.perf_counter()
            reused = self.memory is not None
            if self.memory is None:
                self.memory = self.memory_factory(self.budget_bytes)
            self.active = True
            self.borrow_count += 1
            if self.record:
                self.record('backend_borrow', time.perf_counter()-start,
                            reused=reused, reserved_bytes=self.reserved_bytes)
            return self.memory

    def release(self):
        with self.lock:
            if not self.active:
                raise RuntimeError('no active backend lease')
            # Existing Arena.close synchronizes before clearing its references.
            self.active = False

    def close(self):
        with self.lock:
            if self.active:
                raise RuntimeError('close the active session before its backend')
            start = time.perf_counter()
            cp.cuda.runtime.deviceSynchronize()
            self.memory = None
            self.closed = True
            if self.record:
                self.record('backend_final_release', time.perf_counter()-start,
                            reserved_bytes=self.reserved_bytes)
