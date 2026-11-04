"""Explicit transfer-module loading choices; frozen update kernels are inherited."""
import time

import cupy as cp

from easygraph.gpu._arena.round3_transfer.kernels import SOURCE, TransferKernels
from easygraph.gpu._arena.round8_bounded_input.session_v2 import BoundedSession, BoundedTypedSession

NAMES = ('move_int', 'move_double', 'insert_int', 'advance_row', 'patch_double')


class ResolvingFunctions:
    """Instance-owned resolution, charged to the first actual maintenance request."""
    def __init__(self, backend, resolution, module_factory):
        if backend not in ('nvcc', 'nvrtc'):
            raise ValueError('backend must be nvcc or nvrtc')
        if resolution not in ('eager', 'demand'):
            raise ValueError('resolution must be eager or demand')
        self.backend, self.resolution = backend, resolution
        self.module_factory = module_factory
        self.module = None
        self.cache = {}
        self.module_object_seconds = 0.
        self.resolutions = []

    def __getitem__(self, name):
        if name not in NAMES:
            raise KeyError(name)
        if name not in self.cache:
            if self.module is None:
                started = time.perf_counter()
                self.module = self.module_factory(code=SOURCE, backend=self.backend,
                                                  options=('--std=c++17',))
                self.module_object_seconds += time.perf_counter()-started
            names = NAMES if self.resolution == 'eager' else (name,)
            for requested in names:
                if requested not in self.cache:
                    started = time.perf_counter()
                    self.cache[requested] = self.module.get_function(requested)
                    self.resolutions.append(dict(name=requested,
                        seconds=time.perf_counter()-started))
        return self.cache[name]

    def stats(self):
        return dict(backend=self.backend, resolution=self.resolution,
            module_created=self.module is not None,
            module_object_seconds=self.module_object_seconds,
            resolution_count=len(self.resolutions),
            resolved_functions=list(self.cache),
            resolution_seconds=sum(r['seconds'] for r in self.resolutions),
            resolutions=[dict(r) for r in self.resolutions],
            scope='RawModule object creation plus get_function wall times, charged inside the first actual maintenance request; subsequent kernel execution is outside these nested timers.')


class StartupTransferKernels(TransferKernels):
    def __init__(self, backend='nvrtc', resolution='eager', *, module_factory=cp.RawModule):
        self.functions = ResolvingFunctions(backend, resolution, module_factory)

    def stats(self):
        return self.functions.stats()


class StartupSession(BoundedSession):
    def __init__(self, *args, transfer_backend='nvrtc', transfer_resolution='eager', **kwargs):
        # Reject an invalid option before any device allocation.
        kernels = StartupTransferKernels(transfer_backend, transfer_resolution)
        super().__init__(*args, **kwargs)
        self.transfer_kernels = kernels

    def stats(self):
        result = super().stats()
        result['transfer_code_loading'] = self.transfer_kernels.stats()
        return result

    def close(self):
        try:
            super().close()
        finally:
            self.transfer_kernels = None


class StartupTypedSession(StartupSession, BoundedTypedSession):
    pass
