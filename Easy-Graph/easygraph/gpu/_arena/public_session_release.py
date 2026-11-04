"""Canonical public session: endpoint production plus public weighted PR semantics.

The frozen V2 cohort uses unweighted PageRank and therefore delegates to its
exact same execution path. Weighted requests use the public CPU dangling-row
threshold with the already reserved factorized workspace and unchanged gather.
"""
from time import perf_counter
from easygraph.gpu._arena.factorized_runtime import FACTOR_SOURCE
from easygraph.gpu._arena.gpu_runtime import cp
from easygraph.gpu._arena.public_session_final import EasyGraphSession as FinalSession
from easygraph.gpu._arena.public_session_final import PublicExecutor as FinalExecutor
from easygraph.gpu._arena.public_session_final import _names, _endpoint_records, _endpoint_lock

_OLD_SCALE = 'scaled[i]=sum[i] ? x[i]/sum[i] : 0.0; dangling=sum[i] ? 0.0 : x[i];'
_NEW_SCALE = 'const bool active = sum[i] > 1.0e-15; scaled[i]=active ? x[i]/sum[i] : 0.0; dangling=active ? 0.0 : x[i];'
assert FACTOR_SOURCE.count(_OLD_SCALE) == 1
PUBLIC_WEIGHTED_SOURCE = FACTOR_SOURCE.replace(_OLD_SCALE, _NEW_SCALE)


class PublicExecutor(FinalExecutor):
    def __init__(self, base):
        super().__init__(base)
        self.public_weighted_kernels = None

    def run(self, graph, arrays, function, attr=None, source=0, **kwargs):
        if function != 'pr' or attr is None or self.pr is None:
            return super().run(graph, arrays, function, attr, source, **kwargs)
        load_seconds = 0.
        if self.public_weighted_kernels is None:
            started = perf_counter()
            module = cp.RawModule(code=PUBLIC_WEIGHTED_SOURCE,
                                  options=('--std=c++17',), backend='nvcc')
            self.public_weighted_kernels = {name: module.get_function(name)
                for name in ('scale_and_stats', 'pagerank_factorized')}
            load_seconds = perf_counter() - started
        prior = self.base.factor_kernels
        self.base.factor_kernels = self.public_weighted_kernels
        try:
            value, metrics = super().run(graph, arrays, function, attr, source, **kwargs)
        finally:
            self.base.factor_kernels = prior
        metrics['public_weighted_kernel_load_seconds'] = load_seconds
        metrics['public_weighted_dangling_threshold'] = 1.e-15
        return value, metrics


def _install_structural_state(graph):
    """Exclude only this graph's scoped wrappers from copied/serialized state."""
    with _endpoint_lock:
        record = _endpoint_records[id(graph)]
        if '__getstate__' in record['methods']:
            return
        original = graph.__getstate__
        had_value = '__getstate__' in graph.__dict__
        value = graph.__dict__.get('__getstate__')
        def structural_state():
            state = original()
            if isinstance(state, dict):
                state = dict(state)
                for name, (wrapped, had_original, saved) in record['methods'].items():
                    if state.get(name) is wrapped:
                        if had_original:
                            state[name] = saved
                        else:
                            state.pop(name, None)
            return state
        graph.__getstate__ = structural_state
        # The existing endpoint owner count restores this hook when the last
        # session closes, including nested sessions sharing the same graph.
        record['methods']['__getstate__'] = (structural_state, had_value, value)


class EasyGraphSession(FinalSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine.executor = PublicExecutor(self.engine.executor.base)
        for graph in self.graphs.values():
            _install_structural_state(graph)

    def close(self):
        executor = getattr(self.engine, 'executor', None)
        if isinstance(executor, PublicExecutor):
            executor.public_weighted_kernels = None
        return super().close()
