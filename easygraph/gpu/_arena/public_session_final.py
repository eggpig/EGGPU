"""Scoped EasyGraph public dispatch into the qualified versioned session.

The graph remains authoritative. Observed edits are committed before its next
public call. The context intercepts PR and complete BFS/Dijkstra calls on its
registered graphs; other public operations keep their existing implementation.
The fixed-capacity session accounts only for the intercepted execution path.
"""
from contextvars import ContextVar
from functools import wraps
from threading import RLock
import math
from time import perf_counter
import numpy as np
from easygraph.gpu._arena.replacement import ReplacementSession as TypedSession
from easygraph.gpu._arena.round6_recovery.fixed_executor import FixedIterationExecutor
from easygraph.gpu._arena.gpu_runtime import cp
from easygraph.gpu._arena.round4_switch.journal import Mutation

_active = ContextVar('eggpu_public_session', default=None)
_dispatch_lock = RLock()
_users = 0
_saved = {}
_installed = {}
_names = ('pagerank', 'single_source_bfs', 'multi_source_bfs',
          'single_source_dijkstra', 'multi_source_dijkstra')


class PublicExecutor:
    """Carry the public PR stopping contract into the unchanged GPU executor."""
    def __init__(self, base):
        self.base = FixedIterationExecutor(base, base.kernel_backend)
        self.pr = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def run(self, graph, arrays, function, attr=None, source=0, **kwargs):
        if function == 'pr' and self.pr is not None:
            alpha, iterations, eps = self.pr
            kwargs.update(alpha=alpha, max_iter=iterations, tol=eps * graph.n)
            if eps == 0:
                with self.base.request(iterations=iterations, alpha=alpha):
                    return self.base.run(graph, arrays, function, attr, source, **kwargs)
            return self._public_pagerank(graph, arrays, attr, source, **kwargs)
        return self.base.run(graph, arrays, function, attr, source, **kwargs)

    def _public_pagerank(self, graph, arrays, attr, source, alpha, max_iter, tol, selected=None):
        """Same factorized kernels; public PR returns the last allowed iterate.

        Public EasyGraph stops at err < n*eps or max_iter. The research
        convergent executor raises on the second condition; this adapter uses
        the public stopping rule without rerunning or relaxing its tolerance.
        """
        executor=self.base
        n=graph.n
        if not n:
            value,metrics=executor.base.download(executor.base.x[:0],selected)
            return value,dict(iterations=0,device_loop_seconds=0.,d2h_bytes=0,representation='factorized',**metrics)
        load_seconds=executor.load_factorized_kernels()
        grid=((n+127)//128,);x,y=executor.base.x,executor.base.y;scaled=executor.scaled_workspace()
        row,col=arrays[graph.key('row')],arrays[graph.key('col')]
        sums=arrays[graph.key('source_sum',attr)]
        weight=arrays[graph.key('weight',attr)] if attr is not None else scaled
        start,end=executor.base.events;t=perf_counter();start.record();event_start=perf_counter()-t
        t=perf_counter()
        executor.base.kernels['init'](grid,(128,),(np.int32(n),x,np.int32(0),np.int32(source)))
        init_seconds=perf_counter()-t;flag=np.empty(2,np.float64);converged=False
        for iteration in range(max_iter):
            cp.cuda.runtime.memsetAsync(executor.base.stat.data.ptr,0,16,cp.cuda.get_current_stream().ptr)
            executor.factor_kernels['scale_and_stats'](grid,(128,),(np.int32(n),x,sums,scaled,executor.base.stat))
            executor.factor_kernels['pagerank_factorized'](grid,(128,),(np.int32(n),row,col,weight,scaled,x,y,executor.base.stat,np.float64(alpha),np.int32(attr is not None)))
            executor.base.stat.get(out=flag);x,y=y,x
            if flag[1]<tol:converged=True;break
        t=perf_counter();end.record();end.synchronize();event_finish=perf_counter()-t
        value,metrics=executor.base.download(x[:n],selected)
        return value,dict(iterations=iteration+1,device_loop_seconds=cp.cuda.get_elapsed_time(start,end)/1000,
            host_event_start_seconds=event_start,host_init_launch_seconds=init_seconds,host_event_finish_seconds=event_finish,
            d2h_bytes=value.nbytes+16*(iteration+1),factorized_kernel_load_seconds=load_seconds,
            representation='factorized',final_global_l1=float(flag[1]),converged=converged,
            stopping='public_tolerance_or_iteration_budget',**metrics)


def _wrapper(name, original):
    @wraps(original)
    def dispatch(graph, *args, **kwargs):
        session = _active.get()
        if session is not None and session.accepts(graph, name, args, kwargs):
            return session._dispatch(graph, name, kwargs)
        return original(graph, *args, **kwargs)
    return dispatch


_endpoint_lock = RLock()
_endpoint_records = {}


def _subscribe_endpoints(session, name, graph):
    """One wrapper per graph, shared by nested sessions and removed by the last."""
    # Graph observers are needed only for the legacy mutable-graph context.
    from easygraph.classes.graph import Graph
    from easygraph.classes.directed_graph import DiGraph
    with _endpoint_lock:
        key = id(graph)
        record = _endpoint_records.get(key)
        if record is None:
            record = dict(graph=graph, users={}, methods={})
            _endpoint_records[key] = record
            operations = [('add_edge', ('u_of_edge', 'v_of_edge')), ('remove_edge', ('u', 'v'))]
            operations += [(op, None) for op in ('add_edges', 'add_edges_from', 'add_weighted_edge',
                          'add_weighted_edges_from', 'remove_edges', 'remove_edges_from',
                          'add_node', 'add_nodes', 'add_nodes_from', 'remove_node', 'remove_nodes')]
            for operation, parameters in operations:
                original = getattr(graph, operation, None)
                if original is None:
                    continue
                # Custom overrides retain the general mutation-observer path.
                allowed = (getattr(Graph, operation, None), getattr(DiGraph, operation, None))
                if getattr(original, '__func__', None) not in allowed:
                    continue
                had_instance_value = operation in graph.__dict__
                instance_value = graph.__dict__.get(operation)
                def make_wrapper(original, parameters):
                    @wraps(original)
                    def call(*args, **kwargs):
                        with _endpoint_lock:
                            if parameters is None:
                                try:
                                    return original(*args, **kwargs)
                                finally:
                                    for owner, graph_name in list(record['users'].items()):
                                        owner._force_full(graph_name)
                            try:
                                if len(args) > 2:
                                    raise ValueError('Unsupported argument shape')
                                endpoints = tuple(args[i] if len(args) > i else kwargs[parameters[i]]
                                                  for i in range(2))
                                hash(endpoints[0]); hash(endpoints[1])
                                if any(parameters[i] in kwargs for i in range(len(args))):
                                    raise ValueError('Duplicated endpoint argument')
                            except (KeyError, ValueError, TypeError):
                                try:
                                    return original(*args, **kwargs)
                                finally:
                                    for owner, graph_name in list(record['users'].items()):
                                        owner._force_full(graph_name)
                            listeners = list(record['users'].items())
                            for owner, graph_name in listeners:
                                owner._begin_endpoint(graph_name, endpoints)
                            succeeded = False
                            try:
                                result = original(*args, **kwargs)
                                succeeded = True
                                return result
                            finally:
                                for owner, graph_name in listeners:
                                    owner._end_endpoint(graph_name, endpoints, succeeded)
                    return call
                wrapped = make_wrapper(original, parameters)
                setattr(graph, operation, wrapped)
                record['methods'][operation] = (wrapped, had_instance_value, instance_value)
        assert record['graph'] is graph
        record['users'][session] = name

    def unsubscribe():
        with _endpoint_lock:
            record['users'].pop(session, None)
            if not record['users']:
                for operation, (wrapped, had_value, value) in record['methods'].items():
                    if getattr(graph, operation) is wrapped:
                        if had_value:
                            setattr(graph, operation, value)
                        else:
                            delattr(graph, operation)
                _endpoint_records.pop(key, None)
    return unsubscribe


class EasyGraphSession:
    def __init__(self, graphs, *, budget_bytes=128 << 20,
                 journal_budget_bytes=4 << 20, weight='weight', strategy='ordinary'):
        from easygraph.classes._state_tracking import subscribe
        self.graphs = dict(graphs)
        if len({id(g) for g in self.graphs.values()}) != len(self.graphs):
            raise ValueError('Register each graph object once')
        if any(g.is_multigraph() for g in self.graphs.values()):
            raise TypeError('A simple Graph or DiGraph is required')
        self.weight = weight
        self.lock = RLock()
        self.names = {id(g): name for name, g in self.graphs.items()}
        self.labels, self.index, self.tables = {}, {}, {}
        self.edge_positions = {}
        self.pending = {name: self._pending() for name in self.graphs}
        self.unsubscribe = []
        self.endpoint_unsubscribe = []
        self.endpoint_depth = {name: 0 for name in self.graphs}
        self.endpoint_syncs = self.full_scans = 0
        self.calls = {name: 0 for name in _names}
        self.syncs = self.logical_edits = self.replacements = 0
        self.producer_seconds = dict(scan=0., diff=0., engine_commit=0., synchronization=0.)
        self.closed = False
        self._context_token = None
        inputs = []
        for name, graph in self.graphs.items():
            labels, table, positions = self._scan(graph)
            self.edge_positions[name] = positions
            self.labels[name] = labels
            self.index[name] = {node: i for i, node in enumerate(labels)}
            self.tables[name] = table
            inputs.append(self._input(name, labels, table))
        self.engine = TypedSession(inputs, budget_bytes, journal_budget_bytes,
                                   strategy, None, host_index_budget_bytes=0,
                                   host_producer='shared', change_budget_bytes=8 << 20)
        self.engine.executor = PublicExecutor(self.engine.executor)
        for name, graph in self.graphs.items():
            self.unsubscribe.append(subscribe(graph, self._on_mutation))
            self.endpoint_unsubscribe.append(_subscribe_endpoints(self, name, graph))

    def _scan(self, graph):
        """One adjacency production supplies both values and observer positions."""
        started=perf_counter()
        self.full_scans += 1
        labels = tuple(graph.nodes)
        index = {node: i for i, node in enumerate(labels)}
        table,positions = {},{}
        for u in labels:
            source=index[u]
            for v, data in graph.adj[u].items():
                value = float(data.get(self.weight, 1.))
                if not math.isfinite(value) or value < 0:
                    raise ValueError('The session requires finite nonnegative edge weights')
                edge=(source,index[v])
                table[edge] = value
                positions.setdefault(id(data),[]).append(edge)
        self.producer_seconds['scan']+=perf_counter()-started
        return labels,table,positions

    @staticmethod
    def _pending():
        return dict(generation=None, generic=False, topology=False, edges={},
                    pairs=set(), retired_owners={})

    @staticmethod
    def _input(name, labels, table):
        return dict(name=name, nodes=np.arange(len(labels), dtype=np.int64),
                    edges=np.asarray(list(table), dtype=np.int64).reshape(-1, 2),
                    weights=np.asarray(list(table.values()), dtype=np.float64))

    def _on_mutation(self, graph, generation, data, keys):
        with self.lock:
            name = self.names[id(graph)]
            pending = self.pending[name]
            if self.endpoint_depth[name]:
                pending['generation'] = generation
                return
            if pending['generation'] != generation and pending['generic']:
                pending['topology'] = True
                pending['edges'].clear()
            pending['generation'] = generation
            pending['generic'] = data is None
            if data is not None and not pending['topology']:
                prior = pending['edges'].get(id(data))
                combined = keys if prior is None else None if keys is None or prior[1] is None else keys | prior[1]
                pending['edges'][id(data)] = (data, combined)

    def _force_full(self, name):
        with self.lock:
            pending = self.pending[name]
            pending['generation'] = self.graphs[name]._mutation_generation
            pending['generic'] = pending['topology'] = True

    def _begin_endpoint(self, name, endpoints):
        with self.lock:
            pending = self.pending[name]
            if pending['generic']:
                pending['topology'] = True
            self.endpoint_depth[name] += 1
            u, v = endpoints
            data = self.graphs[name].adj.get(u, {}).get(v)
            if data is not None and id(data) in self.edge_positions[name]:
                pending['retired_owners'][id(data)] = data

    def _end_endpoint(self, name, endpoints, succeeded):
        with self.lock:
            self.endpoint_depth[name] -= 1
            pending = self.pending[name]
            graph = self.graphs[name]
            pending['generation'] = graph._mutation_generation
            u, v = endpoints
            if not succeeded or u not in self.index[name] or v not in self.index[name]:
                pending['topology'] = pending['generic'] = True
                return
            pending['pairs'].add((u, v))
            if not graph.is_directed():
                pending['pairs'].add((v, u))

    def _sync_endpoint_pairs(self, name):
        """Use final edge values at the declared call boundary, with one commit."""
        buffer = self.pending[name]
        if not buffer['pairs'] or buffer['generic'] or buffer['topology']:
            return False
        graph, index = self.graphs[name], self.index[name]
        candidates = {(index[u], index[v]) for u, v in buffer['pairs']}
        fresh_positions = {}
        for u, v in buffer['pairs']:
            data = graph.adj.get(u, {}).get(v)
            if data is not None:
                fresh_positions.setdefault(id(data), []).append((index[u], index[v]))
        for data, keys in buffer['edges'].values():
            if keys is not None and self.weight not in keys:
                continue
            positions = self.edge_positions[name].get(id(data), fresh_positions.get(id(data)))
            if positions is None:
                return False
            candidates.update(positions)
        diff_started = perf_counter()
        changes, final_values = [], {}
        previous = self.tables[name]
        labels = self.labels[name]
        for u, v in sorted(candidates):
            data = graph.adj.get(labels[u], {}).get(labels[v])
            if data is None:
                if (u, v) in previous:
                    changes.append(Mutation('delete', u, v))
                final_values[u, v] = None
            else:
                value = float(data.get(self.weight, 1.))
                if not math.isfinite(value) or value < 0:
                    raise ValueError('The session requires finite nonnegative edge weights')
                if (u, v) not in previous or previous[u, v] != value:
                    changes.append(Mutation('set', u, v, value))
                final_values[u, v] = value
        self.producer_seconds['diff'] += perf_counter() - diff_started
        if changes:
            started = perf_counter()
            self.engine.mutate(name, changes)
            self.producer_seconds['engine_commit'] += perf_counter() - started
            self.logical_edits += len(changes)
        for edge, value in final_values.items():
            if value is None:
                previous.pop(edge, None)
            else:
                previous[edge] = value
        for owner in buffer['retired_owners']:
            self.edge_positions[name].pop(owner, None)
        self.edge_positions[name].update(fresh_positions)
        self.pending[name] = self._pending()
        self.syncs += 1
        self.endpoint_syncs += 1
        return True

    def _sync(self,name):
        started=perf_counter()
        try:
            return self._sync_impl(name)
        finally:
            self.producer_seconds['synchronization']+=perf_counter()-started

    def _sync_impl(self, name):
        buffer = self.pending[name]
        if buffer['generation'] is None:
            return
        if self._sync_endpoint_pairs(name):
            return
        pending = list(buffer['edges'].values())
        if buffer['generic'] or buffer['topology']:
            pending.append(None)
        # Refined attribute notifications overwrite their generic generation
        # notification. An unrelated attribute changes no registered dependency.
        if all(item is not None and item[1] is not None and self.weight not in item[1]
               for item in pending):
            self.pending[name] = self._pending()
            return
        if all(item is not None and id(item[0]) in self.edge_positions[name]
               for item in pending):
            changes = {}
            for data, keys in pending:
                if keys is not None and self.weight not in keys:
                    continue
                value = float(data.get(self.weight, 1.))
                if not math.isfinite(value) or value < 0:
                    raise ValueError('The session requires finite nonnegative edge weights')
                for edge in self.edge_positions[name][id(data)]:
                    if self.tables[name][edge] != value:
                        changes[edge] = value
            if changes:
                commit_started=perf_counter()
                self.engine.mutate(name, [Mutation('set', u, v, value) for (u, v), value in changes.items()])
                self.producer_seconds['engine_commit']+=perf_counter()-commit_started
                self.tables[name].update(changes)
                self.logical_edits += len(changes)
            self.pending[name] = self._pending()
            self.syncs += 1
            return
        labels, table, positions = self._scan(self.graphs[name])
        if labels != self.labels[name]:
            value = self._input(name, labels, table)
            commit_started=perf_counter()
            self.engine.replace(name, value['nodes'], value['edges'], value['weights'])
            self.producer_seconds['engine_commit']+=perf_counter()-commit_started
            self.labels[name] = labels
            self.index[name] = {node: i for i, node in enumerate(labels)}
            self.replacements += 1
        else:
            previous = self.tables[name]
            diff_started=perf_counter()
            edits = [Mutation('delete', u, v) for u, v in previous if (u, v) not in table]
            edits += [Mutation('set', u, v, value) for (u, v), value in table.items()
                      if (u, v) not in previous or previous[(u, v)] != value]
            self.producer_seconds['diff']+=perf_counter()-diff_started
            if edits:
                commit_started=perf_counter()
                self.engine.mutate(name, edits)
                self.producer_seconds['engine_commit']+=perf_counter()-commit_started
                self.logical_edits += len(edits)
        self.tables[name] = table
        self.edge_positions[name] = positions
        self.pending[name] = self._pending()
        self.syncs += 1

    def accepts(self, graph, name, args, kwargs):
        if self.closed or args or id(graph) not in self.names:
            return False
        if self.graphs[self.names[id(graph)]] is not graph:
            return False
        if kwargs.get('target') is not None:
            return False
        requested = kwargs.get('weight', self.weight if 'dijkstra' in name else None)
        if requested is not None and requested != self.weight:
            return False
        if name == 'pagerank' and not graph.is_directed() and any(u in graph.adj[u] for u in graph.nodes):
            return False  # Public undirected PR has its own loop projection.
        return True

    def _one(self, name, function, source=None, weight=None):
        if function == 'pr' and not self.labels[name]:
            return {}
        source_id = 0 if function == 'pr' else self.index[name][source]
        value = self.engine.query(name, function, None if weight is None else 'message_count', source_id)
        # Public paths omit unreachable nodes; each return is independently owned.
        return {node: float(value[i]) for i, node in enumerate(self.labels[name])
                if function == 'pr' or math.isfinite(float(value[i]))}

    def _dispatch(self, graph, function, kwargs):
        with self.lock:
            name = self.names[id(graph)]
            self._sync(name)
            self.calls[function] += 1
            if function == 'pagerank':
                old = self.engine.executor.pr
                self.engine.executor.pr = (kwargs.get('alpha', .85), kwargs.get('max_iter', 200), kwargs.get('eps', 1e-6))
                try:
                    return self._one(name, 'pr', weight=kwargs.get('weight'))
                finally:
                    self.engine.executor.pr = old
            operation = 'bfs' if 'bfs' in function else 'sssp'
            weight = None if operation == 'bfs' else kwargs.get('weight', self.weight)
            if operation == 'sssp' and weight is None:
                operation = 'bfs'
            if function.startswith('multi_'):
                return {source: self._one(name, operation, source, weight) for source in kwargs['sources']}
            return self._one(name, operation, kwargs['source'], weight)

    def stats(self):
        return dict(public_calls=dict(self.calls), syncs=self.syncs, logical_edits=self.logical_edits,
                    graph_replacements=self.replacements, engine=self.engine.stats(),
                    endpoint_syncs=self.endpoint_syncs, full_scans=self.full_scans,
                    pending_endpoint_pairs=sum(len(p['pairs']) for p in self.pending.values()),
                    pending_attribute_records=sum(len(p['edges']) for p in self.pending.values()),
                    producer_seconds=dict(self.producer_seconds),
                    timer_scope='Nested public adapter diagnostics: synchronization includes scan, diff and engine commit; initialization also includes one scan. Never add these to the outer complete session again.',
                    pending_topology_graphs=sum(bool(p['generic'] or p['topology']) for p in self.pending.values()),
                    integration='Scoped public dispatch; three algorithm families use the versioned session',
                    memory_scope='Owned arena covers session-dispatched calls; original backend operations are separate')

    def __enter__(self):
        from easygraph.utils import gpu_eggpu_backend as backend
        global _users
        if self.closed or self._context_token is not None:
            raise RuntimeError('Session is closed or already entered')
        with _dispatch_lock:
            if _users == 0:
                for name in _names:
                    _saved[name] = getattr(backend, name)
                    _installed[name] = _wrapper(name, _saved[name])
                    setattr(backend, name, _installed[name])
            _users += 1
        self._context_token = _active.set(self)
        return self

    def close(self):
        global _users
        if self.closed:
            return
        if self._context_token is not None:
            from easygraph.utils import gpu_eggpu_backend as backend
            _active.reset(self._context_token)
            self._context_token = None
            with _dispatch_lock:
                _users -= 1
                if _users == 0:
                    for name in _names:
                        if getattr(backend, name) is _installed[name]:
                            setattr(backend, name, _saved[name])
                    _saved.clear(); _installed.clear()
        for unsubscribe in self.endpoint_unsubscribe:
            unsubscribe()
        self.endpoint_unsubscribe.clear()
        for unsubscribe in self.unsubscribe:
            unsubscribe()
        self.unsubscribe.clear()
        self.engine.close()
        self.pending.clear(); self.tables.clear(); self.graphs.clear(); self.edge_positions.clear()
        self.labels.clear(); self.index.clear(); self.names.clear(); self.endpoint_depth.clear()
        self.closed = True

    def __exit__(self, *exc):
        self.close()
