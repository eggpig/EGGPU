"""Complete WCC using shared outgoing CSR and the session's existing workspace.

Each call recomputes a minimum-root disjoint set. We retain representations,
not previous answers. Directed edges induce weak connectivity; isolated nodes
remain singleton components. Atomic hooking strictly decreases root IDs.
"""
from time import perf_counter
from contextlib import contextmanager
import numpy as np
import cupy as cp

SOURCE = r'''
__device__ __forceinline__ int root(int *p, int x) {
    int y = atomicAdd(p + x, 0);
    while (y != x) { x = y; y = atomicAdd(p + x, 0); }
    return x;
}
extern "C" __global__ void init_parents(int n, int *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = i;
}
extern "C" __global__ void union_edges(int n, const int *row,
                                       const int *col, int *p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int u = tid >> 5, lane = tid & 31;
    if (u >= n) return;
    for (int e = row[u] + lane; e < row[u + 1]; e += 32) {
        int a = root(p, u), b = root(p, col[e]);
        while (a != b) {
            int hi = max(a, b), lo = min(a, b);
            int prior = atomicCAS(p + hi, hi, lo);
            if (prior == hi) break;
            a = root(p, lo); b = root(p, prior);
        }
    }
}
extern "C" __global__ void emit_roots(int n, int *p, double *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { int r = root(p, i); atomicExch(p + i, r); out[i] = r; }
}
'''


class ConnectivityExecutor:
    def __init__(self, base):
        self.base = base
        self.connectivity_kernels = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def pr(self):
        return self.base.pr

    @pr.setter
    def pr(self, value):
        self.base.pr = value

    def run(self, graph, arrays, function, attr=None, source=0, **kwargs):
        if function != 'wcc':
            return self.base.run(graph, arrays, function, attr, source, **kwargs)
        n = graph.n
        started = perf_counter()
        if self.connectivity_kernels is None:
            module = cp.RawModule(code=SOURCE, options=('--std=c++17',), backend='nvrtc')
            self.connectivity_kernels = {k: module.get_function(k) for k in
                                        ('init_parents', 'union_edges', 'emit_roots')}
        load_seconds = perf_counter() - started
        parent = self.base.y.view(np.int32)[:n]
        output = self.base.x[:n]
        start, end = self.base.events
        start.record()
        if n:
            self.connectivity_kernels['init_parents'](((n + 255)//256,), (256,), (np.int32(n), parent))
            self.connectivity_kernels['union_edges'](((n + 3)//4,), (128,),
                 (np.int32(n), arrays[graph.key('out_row')], arrays[graph.key('out_col')], parent))
            self.connectivity_kernels['emit_roots'](((n + 255)//256,), (256,), (np.int32(n), parent, output))
        end.record(); end.synchronize()
        value, metrics = self.base.download(output, kwargs.get('selected'))
        return value, dict(iterations=1, device_loop_seconds=cp.cuda.get_elapsed_time(start, end)/1000,
                           d2h_bytes=value.nbytes, connectivity_module_load_seconds=load_seconds,
                           connectivity='minimum_root_atomic_union', retained_algorithm_answers=False,
                           **metrics)


def connectivity_session(base_class):
    """Add one representation-sharing function to a compatible typed session."""
    class ConnectivitySession(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.executor = ConnectivityExecutor(self.executor)

        def _requirements(self, graph, function, attr):
            if function == 'wcc':
                return [graph.key('out_row'), graph.key('out_col')]
            return super()._requirements(graph, function, attr)

        def prepare(self, graph, function, attr):
            return super().prepare(graph, 'bfs' if function == 'wcc' else function, attr)

        @contextmanager
        def batch(self, requests):
            requests = tuple(tuple(r) for r in requests)
            if not any(r[1] == 'wcc' for r in requests):
                with super().batch(requests):
                    yield self
                return
            from easygraph.gpu._arena.session_manager import token
            with self.lock:
                if self.batch_requests is not None:
                    raise RuntimeError('Nested batch')
                if self.strategy != 'ordinary':
                    raise ValueError('WCC batch currently uses online ordinary retention')
                if len({r[0] for r in requests}) != 1:
                    raise ValueError('One graph/version per analysis epoch')
                for gid, function, attr, source in requests:
                    if gid not in self.graphs or function not in ('bfs','sssp','pr','wcc'):
                        raise ValueError('Unknown graph or function')
                self.batch_requests = requests; self.batch_index = 0
                self.batch_keys = tuple(tuple(token(k) for k in self._requirements(self.graphs[g], f, a))
                                        for g,f,a,s in requests)
                try:
                    yield self
                    if self.batch_index != len(requests):
                        raise RuntimeError('Incomplete declared batch')
                finally:
                    self.batch_requests = None; self.batch_keys = (); self.batch_index = 0
    return ConnectivitySession
