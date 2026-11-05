"""Read already-computed arena metrics without materializing lazy graph views."""
from numbers import Real
from time import perf_counter_ns


def json_ready(value):
    """Convert scalar NumPy metadata only, after the measured workflow ends."""
    if isinstance(value,dict):
        return {str(k):json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_ready(v) for v in value]
    if type(value).__module__.startswith('numpy') and getattr(value,'ndim',None)==0:
        return value.item()
    return value


def snapshot(public_session):
    # Exact current d84 arena object chain; unsupported objects fail explicitly.
    managed = public_session._backend
    engine = managed.backend.session
    graph = engine.graphs['g']
    state = graph.r2_state
    stored = vars(state)
    views = stored.get('host_view_seconds', {})
    return dict(host_view_seconds=dict(views),
        materialized_host_views=[name for name in ('incoming','degree','sums','in_weight') if name in stored],
        graph_host_prepare_seconds=graph.host_prepare_seconds,
        counters={k:v for k,v in engine.counters.items() if isinstance(v,Real) and not isinstance(v,bool)},
        managed_initialization_seconds=managed.initialization_seconds)


def deltas(before, after):
    return {key:after.get(key,0)-before.get(key,0) for key in set(before)|set(after)}


class CaptureSequence:
    def __init__(self, original):
        self.original = original
        self.contract = original.contract
        self.captures = []
        self.construction_snapshot = snapshot(original.session)

    def query(self, request):
        start = perf_counter_ns()
        before = snapshot(self.original.session)
        prepared = perf_counter_ns()
        params = {'source':request['source']} if request['algorithm'] in ('bfs','sssp') else {}
        result = self.original.session.query('g',request['algorithm'],params)
        returned = perf_counter_ns()
        after = snapshot(self.original.session)
        captured = perf_counter_ns()
        self.captures.append(dict(index=request['index'], algorithm=request['algorithm'],
            source=request['source'], public_query_seconds=result.seconds,
            public_call_wall_seconds=(returned-prepared)/1e9,
            observation_overhead_seconds=(prepared-start+captured-returned)/1e9,
            backend_metrics=result.backend_metrics,
            host_view_seconds_delta=deltas(before['host_view_seconds'],after['host_view_seconds']),
            graph_host_prepare_seconds_delta=after['graph_host_prepare_seconds']-before['graph_host_prepare_seconds'],
            counters_delta=deltas(before['counters'],after['counters']),
            before=before,after=after))
        return result.values

    def close(self):
        self.original.close()


def diagnostic_execute(execute, factory, args, graph):
    holders = []
    def create(a,g):
        value = CaptureSequence(factory(a,g))
        holders.append(value)
        return value
    outputs, record = execute(args,graph,backend_factory=create)
    captures = holders[0].captures if holders else []
    record.update(diagnostic_only=True, performance_eligible=False, formal_measurement=False,
        arena_diagnostics=json_ready(captures),
        construction_snapshot=json_ready(holders[0].construction_snapshot) if holders else None,
        diagnostic_scope='Original 16 requests and full outputs; already-produced QueryResult metrics plus scalar host-state snapshots. Observation overhead is included in this diagnostic workflow, never a formal timing.',
        timer_relationship='Nested timers overlap: do not add device_loop, execute_and_download, download or prepare/host-view times into a new total.')
    if record.get('execution_complete') and len(captures)!=16:
        raise ValueError('Complete diagnostic needs exactly 16 unfiltered metric records')
    return outputs,record
