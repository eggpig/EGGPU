"""Instance-local SSSP numeric contract; other algorithms use the frozen path.

The terminal call is a literal copy of GraphSession.query with IEEE SSSP
output interpretation, unchanged timer/counters/output policy. The typed-column
boundary preserves journal flush/lease/epoch handling. No global monkeypatch,
result cache, ndarray operator override, or untimed output repair is used.
"""
import time
from types import MethodType
import numpy as np
from easygraph.gpu._arena.gpu_runtime import cp
from easygraph.gpu._arena.session_manager import request_boundary, token
from easygraph.gpu._arena.factorized_session import portfolio_request_bytes
from easygraph.gpu._arena.round8_columns.columns import topology
from easygraph.gpu._arena.device_delta_ieee_v3 import DeviceResidentDeltaSSSP

@request_boundary
def _ieee_query(self, graph_name, function, attr=None, source=0, result='array', selected=None):
    if function != 'sssp' or getattr(self.native_sssp, 'ieee_distances', False) is not True:
        raise ValueError('IEEE query requires the qualified IEEE SSSP adapter')
    t=time.perf_counter()
    graph=self.graphs[graph_name]
    if not 0 <= source < max(1,graph.n): raise ValueError('source not in graph')
    if result not in ('array','dict','selected','view'):raise ValueError(result)
    indices=None
    if result=='selected':
        if selected is None:raise ValueError('Selected result requires vertex indices')
        values=np.asarray(selected)
        if values.ndim!=1 or (values.size and (values.dtype.kind not in 'iu' or np.any(values<0) or np.any(values>=graph.n))):
            raise ValueError('Selected indices must be a one-dimensional sequence of valid integers')
        indices=np.ascontiguousarray(values,dtype=np.int32)
    arrays,names=self.prepare(graph,function,attr)
    t1=time.perf_counter()
    for name in names.values(): self.model.pin(name)
    try:
        native=(self.native_bfs if function=='bfs' else self.native_sssp if function=='sssp' else None)
        if native is not None:
            self.sources.set(np.asarray([source],np.int32))
            self.counters['h2d_bytes']+=4
            start,end=self.executor.events
            host_time=time.perf_counter();start.record();event_start_seconds=time.perf_counter()-host_time
            args=[arrays[graph.key('out_row')],arrays[graph.key('out_col')]]
            if function=='sssp':args.append(arrays[graph.key('out_weight',attr)])
            value=native.run(*args,self.sources)
            host_time=time.perf_counter();end.record();end.synchronize();event_finish_seconds=time.perf_counter()-host_time
            output,download=self.executor.download(value[0],indices)
            # Native SSSP already uses IEEE +inf; all finite values are retained.
            metrics=dict(iterations=native.last['expansion_rounds'],
                device_loop_seconds=cp.cuda.get_elapsed_time(start,end)/1000,
                host_event_start_seconds=event_start_seconds,host_event_finish_seconds=event_finish_seconds,
                native_execute_seconds=native.last['native_execute_seconds'],
                d2h_bytes=output.nbytes+native.last['control_d2h_bytes'],**download)
        else:output,metrics=self.executor.run(graph,arrays,function,attr,source,selected=indices)
    finally:
        cp.cuda.runtime.deviceSynchronize()
        for name in names.values(): self.model.unpin(name)
    t2=time.perf_counter()
    if result=='dict': answer={node:float(v) for node,v in zip(graph.nodes,output)}
    elif result=='array': answer=output
    elif result=='selected': answer={graph.nodes[int(i)]:float(v) for i,v in zip(indices,output)}
    else:
        from easygraph.utils.gpu_eggpu_backend import _DenseValueDict
        answer=_DenseValueDict(graph.nodes,output,node_to_idx=graph.node_to_id,dtype=float)
    packaged=time.perf_counter()
    if self.policy=='no_reuse':
        for name in list(self.model.components): self._drop(name)
    self._compact_update_logs()
    end=time.perf_counter()
    self.last=dict(graph=graph_name,function=function,attribute=attr,seconds=end-t,
                   prepare_seconds=t1-t,execute_and_download_seconds=t2-t1,
                   packaging_seconds=packaged-t2,housekeeping_seconds=end-packaged,result_mode=result,**metrics)
    self.last['kernel_backend']='ieee_'+native.last.get('selected_kernel', 'parallel_delta')
    self.last['distance_encoding']=native.last['distance_encoding']
    self.counters['d2h_bytes']+=metrics['d2h_bytes']
    self.counters['h2d_bytes']+=metrics['result_indices_h2d_bytes']
    self.counters['result_d2h_bytes']+=metrics['result_d2h_bytes']
    self.counters['result_indices_h2d_bytes']+=metrics['result_indices_h2d_bytes']
    self.counters['result_gather_launches']+=metrics['result_gather_launches']
    self.events.append(self.last)
    return answer


@request_boundary
def _ieee_query_columns(self, gid, function, attr=None, source=0, result='array', selected=None):
    if function != 'sssp':
        return self._pre_ieee_query_columns(gid, function, attr, source, result, selected)
    started = time.perf_counter(); self.logical_tick += 1
    if self.last_graph is not None and self.last_graph != gid:
        self.counters['graph_switches'] += 1
    self.last_graph = gid
    ref = self.refs[gid]; changes = self.journal.peek_pending(ref)
    self._sync_snapshot(gid)
    changed_topology = topology(changes)
    needs_values = function == 'sssp' or (function == 'pr' and attr is not None)
    needed = bool(changes) and (self.mode != 'buffer_scoped' or changed_topology or needs_values)
    detail = None
    if needed:
        detail = self._adopt(self.journal.flush(ref), self.mode != 'buffer_refresh')
    elif changes:
        self.counters['dependency_scoped_deferred_queries'] += 1
    graph = self.graphs[gid]
    if self.executor.workspace_bytes + portfolio_request_bytes(graph) > self.arena.budget:
        raise MemoryError('Current graph request exceeds shared arena budget')
    with graph.r2_state.lease():
        answer = _ieee_query(self, gid, function, attr, source, result, selected)
        # Preserve ConditionalGraphSession's successful-query history.
        self.request_counts[(gid, function, attr)] += 1
        self.last['conditional_selection'] = self.selection
    self._reconcile_residence()
    from easygraph.gpu._arena.session_manager import token
    for key in self._requirements(graph, function, attr):
        name = token(key)
        if name in self.residence:
            self.counters['maintained_then_consumed'] += 1
            self.counters['maintenance_to_consumption_ticks'] += self.logical_tick - self.residence.pop(name)[1]
    self.last.update(seconds=time.perf_counter()-started, deferred_host_materialization=needed,
                     journal_maintenance=detail, graph_epoch=ref.generation)
    return answer


def install_device_delta(session):
    """Install before requests; the session keeps its arena and update protocol."""
    if hasattr(session, '_pre_ieee_query_columns'):
        raise ValueError('IEEE SSSP already installed')
    session.native_sssp = DeviceResidentDeltaSSSP(session.native_sssp)
    session.native_sssp.ieee_distances = True
    session._pre_ieee_query_columns = session._query_columns
    session._query_columns = MethodType(_ieee_query_columns, session)
    return session
