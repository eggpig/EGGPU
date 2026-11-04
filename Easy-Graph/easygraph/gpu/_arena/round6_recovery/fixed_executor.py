"""R6 exact-iteration PR on the existing factorized kernels and workspace.

GraphSession has no iteration argument. Bind a request explicitly:
    session.executor = FixedIterationExecutor(session.executor, session.kernel_backend)
    with session.executor.request(iterations=20, alpha=.85):
        output = session.query(gid, 'pr', attr=None)

The default remains the original convergent executor. Per-iteration 16-byte
statistics downloads are preserved for a clean stopping-semantics comparison.
"""
from contextlib import contextmanager
import math
import operator
import time

import numpy as np
from easygraph.gpu._arena.factorized_runtime import FactorizedExecutor
from easygraph.gpu._arena.gpu_runtime import cp


class FixedIterationExecutor(FactorizedExecutor):
    def __init__(self, base, kernel_backend):
        if isinstance(base, FactorizedExecutor):
            previous = base
            super().__init__(previous.base, kernel_backend)
            self.factor_kernels = previous.factor_kernels
        else:
            super().__init__(base, kernel_backend)
        self.requested_iterations = None
        self.requested_alpha = None

    def set_request(self, *, iterations=None, alpha=None):
        """Set one serial request context; validation is atomic and eager."""
        if iterations is not None:
            if isinstance(iterations, (bool, np.bool_)):
                raise TypeError('iterations must be a nonnegative integer')
            iterations = operator.index(iterations)
            if iterations < 0:
                raise ValueError('iterations must be nonnegative')
        if alpha is not None:
            alpha = float(alpha)
            if not math.isfinite(alpha) or not 0 <= alpha <= 1:
                raise ValueError('alpha must be finite and in [0, 1]')
        self.requested_iterations, self.requested_alpha = iterations, alpha
        return self

    def clear_request(self):
        self.set_request()

    @contextmanager
    def request(self, *, iterations=None, alpha=None):
        old = self.requested_iterations, self.requested_alpha
        self.set_request(iterations=iterations, alpha=alpha)
        try:
            yield self
        finally:
            self.requested_iterations, self.requested_alpha = old

    def run(self, graph, arrays, function, attr=None, source=0, alpha=.85,
            tol=1e-8, max_iter=300, selected=None):
        iterations = self.requested_iterations
        if function != 'pr' or iterations is None:
            return super().run(graph, arrays, function, attr, source,
                               alpha, tol, max_iter, selected)
        if graph.key('source_sum', attr) not in arrays:
            raise ValueError('Exact-iteration PR requires the factorized source_sum view')
        if self.requested_alpha is not None:
            alpha = self.requested_alpha
        n = graph.n
        if not n:
            value, metrics = self.base.download(self.base.x[:0], selected)
            return value, dict(iterations=0, requested_iterations=iterations,
                               stopping='exact_iterations', empty_graph=True,
                               device_loop_seconds=0., d2h_bytes=0,
                               representation='factorized', alpha=float(alpha),
                               final_global_l1=None, **metrics)
        load_seconds = self.load_factorized_kernels()
        grid = ((n + 127) // 128,)
        x, y = self.base.x, self.base.y
        scaled = self.scaled_workspace()
        row, col = arrays[graph.key('row')], arrays[graph.key('col')]
        sums = arrays[graph.key('source_sum', attr)]
        weight = arrays[graph.key('weight', attr)] if attr is not None else scaled
        start, end = self.base.events
        t = time.perf_counter(); start.record(); event_start = time.perf_counter() - t
        t = time.perf_counter()
        self.base.kernels['init'](grid, (128,), (np.int32(n), x, np.int32(0), np.int32(source)))
        init_seconds = time.perf_counter() - t
        flag = np.empty(2, np.float64)
        for _ in range(iterations):
            cp.cuda.runtime.memsetAsync(self.base.stat.data.ptr, 0, 16, cp.cuda.get_current_stream().ptr)
            self.factor_kernels['scale_and_stats'](grid, (128,),
                (np.int32(n), x, sums, scaled, self.base.stat))
            self.factor_kernels['pagerank_factorized'](grid, (128,),
                (np.int32(n), row, col, weight, scaled, x, y, self.base.stat,
                 np.float64(alpha), np.int32(attr is not None)))
            self.base.stat.get(out=flag)
            x, y = y, x
        t = time.perf_counter(); end.record(); end.synchronize(); event_finish = time.perf_counter() - t
        value, metrics = self.base.download(x[:n], selected)
        return value, dict(iterations=iterations, requested_iterations=iterations,
            stopping='exact_iterations', empty_graph=False, alpha=float(alpha),
            device_loop_seconds=cp.cuda.get_elapsed_time(start, end) / 1000,
            host_event_start_seconds=event_start, host_init_launch_seconds=init_seconds,
            host_event_finish_seconds=event_finish,
            d2h_bytes=value.nbytes + 16 * iterations,
            factorized_kernel_load_seconds=load_seconds,
            representation='factorized', final_global_l1=float(flag[1]) if iterations else None,
            **metrics)
