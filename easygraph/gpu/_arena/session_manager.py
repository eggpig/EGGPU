"""Executable shared-state session backed by an actual CUDA arena."""
from __future__ import annotations
from collections import defaultdict
import json
import time
from dataclasses import replace
import threading
from functools import wraps
import ctypes

import cupy as cp
import numpy as np

from easygraph.gpu._arena.gpu_runtime import Arena, Executor, HostGraph
from easygraph.gpu._arena.state_model import StateModel, ComponentSpec


POLICIES = ('no_reuse','whole_lru','single_weight','shared_lru',
            'selective_lru','repair_lru','cost_size','adaptive')
KERNEL_BACKENDS=('reference','native_bfs','native_paths')


def workspace_layout(nmax,kernel_backend='reference',patch_capacity=8192):
    if kernel_backend not in KERNEL_BACKENDS:raise ValueError(kernel_backend)
    if nmax<0:raise ValueError('Negative vertex capacity')
    layout=Executor.workspace_layout(nmax,patch_capacity)
    if kernel_backend!='reference':
        from easygraph.gpu._arena.native_bridge import DEFAULT_BUILD
        lib=ctypes.CDLL(str(DEFAULT_BUILD/'libeggpu_native_bfs.so'))
        kinds=['bfs']+(['sssp'] if kernel_backend=='native_paths' else [])
        for kind in kinds:
            query=getattr(lib,'eggpu_native_'+kind+'_workspace_bytes')
            query.argtypes=[ctypes.c_int,ctypes.c_int];query.restype=ctypes.c_size_t
            need=query(nmax,1)
            if not need:raise ValueError('Invalid native vertex capacity')
            layout.extend([('@native_'+kind+'_workspace',(need,),np.uint8),
                           ('@native_'+kind+'_output',(nmax,),np.float64)])
        layout.append(('@native_sources',(1,),np.int32))
    return layout


def workspace_bytes(nmax,kernel_backend='reference',patch_capacity=8192):
    return sum(Arena.capacity(shape,dtype) for _,shape,dtype in workspace_layout(nmax,kernel_backend,patch_capacity))


def largest_request_bytes(graph):
    """PR includes the largest supported individual view in either direction."""
    return (Arena.capacity(graph.row.shape,np.int32)+Arena.capacity(graph.col.shape,np.int32)+
            Arena.capacity(graph.col.shape,np.float64)+Arena.capacity((graph.n,),np.uint8))


def token(key): return json.dumps(key,separators=(',',':'))


def request_boundary(method):
    @wraps(method)
    def wrapped(self,*args,**kwargs):
        with self.lock:
            return method(self,*args,**kwargs)
    return wrapped


class GraphSession:
    def __init__(self, graphs, budget_bytes, policy='adaptive',kernel_backend='reference'):
        if policy not in POLICIES: raise ValueError(policy)
        if kernel_backend not in KERNEL_BACKENDS:raise ValueError(kernel_backend)
        self.graphs={g.name:g for g in graphs}
        if not self.graphs:raise ValueError('At least one graph is required')
        nmax=max(g.n for g in self.graphs.values())
        expected_workspace=workspace_bytes(nmax,kernel_backend)
        if expected_workspace>budget_bytes:raise MemoryError('Session workspace exceeds arena budget')
        self.policy=policy
        self.arena=Arena(budget_bytes)
        self.executor=Executor(self.arena,nmax)
        self.kernel_backend=kernel_backend
        self.native_bfs=self.native_sssp=None
        if kernel_backend!='reference':
            from easygraph.gpu._arena.native_bridge import NativeBFS,NativeSSSP
            self.native_bfs=NativeBFS(self.arena,nmax)
            if kernel_backend=='native_paths':self.native_sssp=NativeSSSP(self.arena,nmax)
            self.sources=self.arena.allocate('@native_sources',(1,),np.int32)
            self.executor.workspace_bytes=self.arena.live
        if expected_workspace!=self.executor.workspace_bytes:raise AssertionError('Native workspace query disagrees with allocations')
        model_policy={'whole_lru':'whole_view_LRU','cost_size':'cost_size','adaptive':'adaptive'}.get(policy,'shared_component_LRU')
        self.model=StateModel(budget_bytes,model_policy)
        self.model.workspace_capacity=self.executor.workspace_bytes
        self.device_versions={}
        self.semantic={}
        self.counters=defaultdict(int)
        self.events=[]
        self.last={}
        self.lock=threading.RLock()
        self._log_compaction_pending=False

    def _drop(self, key):
        aid=self.model.components[key].allocation_id
        self.model.retire(key)
        if self.model.allocations[aid].pending_release:
            self.model.complete_release(aid)
            self.arena.release(aid)
        self.device_versions.pop(key,None)
        self.counters['evictions']+=1

    @request_boundary
    def update(self, graph_name, attr, positions, values):
        t=time.perf_counter()
        graph=self.graphs[graph_name]
        graph.update(attr,positions,values)
        self.model.update(graph_name+':attr:'+attr,affected_scope=list(map(int,positions)))
        if self.policy in ('no_reuse','whole_lru','shared_lru'):
            self.model.update(graph_name+':all',affected_scope=None)
        self.counters['updates']+=len(positions)
        self._log_compaction_pending=True
        return time.perf_counter()-t

    @request_boundary
    def replace_graph(self, graph):
        t=time.perf_counter()
        old=self.graphs[graph.name]
        target=max(self.executor.nmax,graph.n)
        needed=workspace_bytes(target,self.kernel_backend,self.executor.patch_capacity)
        prospective=[graph if name==graph.name else g for name,g in self.graphs.items()]
        if needed+max(largest_request_bytes(g) for g in prospective)>self.arena.budget:
            raise MemoryError('Replacement requires more than the fixed arena budget for one complete request')
        if any(a.pins for a in self.model.allocations.values()):
            raise ValueError('Graph replacement waits for all active readers')
        if target>self.executor.nmax:self._grow_workspace(target,needed)
        graph.generation=old.generation+1
        self.graphs[graph.name]=graph
        self.model.update(graph.name+':topology')
        self.model.update(graph.name+':mapping')
        self.model.update(graph.name+':all')
        self.counters['topology_replacements']+=1
        self._log_compaction_pending=True
        self.counters['topology_replace_ns']+=int((time.perf_counter()-t)*1e9)
        return time.perf_counter()-t

    def _grow_workspace(self,nmax,needed):
        """Preflight a capacity transaction, then rebuild scratch in the arena."""
        t=time.perf_counter()
        plan=self.model.plan([],workspace_bytes=needed,view_id='@workspace_growth')
        if not plan.feasible:raise MemoryError(plan.reason)
        layout=workspace_layout(nmax,self.kernel_backend,self.executor.patch_capacity)
        host_partial=(np.empty(min(4096,max(1,(nmax+255)//256)),np.float64)
                      if self.native_sssp is not None else None)
        old=self.executor.workspace_bytes
        cp.cuda.runtime.deviceSynchronize()
        for action in plan.actions:
            if action['kind']=='release':self.arena.release(action['allocation_id'])
            elif action['kind']=='evict':
                self.device_versions.pop(action['key'],None)
                self.counters['evictions']+=1
                self.counters['workspace_growth_evictions']+=1
            elif action['kind'] not in ('workspace_resize','drop_view'):
                raise RuntimeError(('unexpected growth action',action))
        self.arena.resize_workspace(layout)
        self.executor.bind_workspace(nmax)
        for kind,native in (('bfs',self.native_bfs),('sssp',self.native_sssp)):
            if native is None:continue
            native.nmax=nmax
            native.workspace=self.arena.slots['@native_'+kind+'_workspace'].array
            native.output=self.arena.slots['@native_'+kind+'_output'].array
            native.workspace_bytes=sum(self.arena.slots['@native_'+kind+'_'+part].capacity for part in ('workspace','output'))
        if self.native_sssp is not None:self.native_sssp.host_partial=host_partial
        if self.native_bfs is not None:self.sources=self.arena.slots['@native_sources'].array
        self.model.commit(plan)
        if self.arena.live!=self.model.live_bytes:raise AssertionError('Workspace growth ledger mismatch')
        self.counters['workspace_resizes']+=1
        self.counters['workspace_growth_bytes']+=needed-old
        self.counters['workspace_resize_ns']+=int((time.perf_counter()-t)*1e9)

    def _requirements(self, graph, function, attr):
        if ((function=='bfs' and self.native_bfs is not None) or
                (function=='sssp' and self.native_sssp is not None)):
            keys=[graph.key('out_row'),graph.key('out_col')]
            return keys+([graph.key('out_weight',attr)] if function=='sssp' else [])
        keys=[graph.key('row'),graph.key('col')]
        if function=='pr': keys += [graph.key('norm',attr),graph.key('dangling',attr)]
        elif function=='sssp': keys += [graph.key('weight',attr)]
        elif function!='bfs': raise ValueError(function)
        return keys

    def _physical(self, key, view):
        return token((view,key)) if self.policy=='whole_lru' else token(key)

    def prepare(self, graph, function, attr):
        keys=self._requirements(graph,function,attr)
        view=token((graph.name,function,attr))
        names={key:self._physical(key,view) for key in keys}
        # Fully retained topology plus one overwriteable weight representation.
        if self.policy=='single_weight':
            kinds={k[1] for k in keys if k[2] is not None}
            for name in list(self.model.components):
                old=self.semantic[name]
                if old[0]==graph.name and old[1] in kinds and old[2]!=attr:
                    self._drop(name)
        t=time.perf_counter()
        if self.model.touch_if_valid(list(names.values()),workspace_bytes=self.executor.workspace_bytes,view_id=view):
            self.counters['planning_ns']+=int((time.perf_counter()-t)*1e9)
            self.counters['hits']+=len(names)
            self.counters['path_reuse']+=1
            arrays={key:self.arena.slots[self.model.components[name].allocation_id].array for key,name in names.items()}
            return arrays,names
        shapes={key:((graph.n,) if key[1]=='dangling' else
                     ((len(graph.row),) if key[1] in ('row','out_row') else (len(graph.col),))) for key in keys}
        dtypes={key:(np.uint8 if key[1]=='dangling' else np.int32 if key[1] in ('row','col','out_row','out_col') else np.float64) for key in keys}
        specs=[]
        repairs={}
        for key,name in names.items():
            self.semantic[name]=key
            capacity=Arena.capacity(shapes[key],dtypes[key])
            deps=graph.dependency(key)
            if self.policy in ('no_reuse','whole_lru','shared_lru'): deps += (graph.name+':all',)
            oldversion=self.device_versions.get(name)
            positions=graph.dirty_positions(key,oldversion) if oldversion is not None else None
            canrepair=(self.policy in ('repair_lru','cost_size','adaptive') and key[1] in ('norm','weight','out_weight')
                       and positions is not None)
            repair_bytes=len(positions)*12 if canrepair else capacity
            repairs[name]=positions
            # Bytes/bandwidth plus observed fixed launch cost, same units for all plans.
            payload=int(np.prod(shapes[key]))*np.dtype(dtypes[key]).itemsize
            build_cost=8e-6+payload/12e9
            repair_cost=12e-6+repair_bytes/12e9
            kwargs=dict(key=name,capacity_bytes=capacity,dependencies=deps,build_cost=build_cost,
                        repair_cost=repair_cost,repairable=canrepair)
            # Explicit repair dependencies keep topology changes on the rebuild path.
            if 'repair_dependencies' in ComponentSpec.__dataclass_fields__:
                kwargs['repair_dependencies']=(graph.name+':attr:'+str(key[2]),)
            specs.append(ComponentSpec(**kwargs))
        plan=self.model.plan(specs,workspace_bytes=self.executor.workspace_bytes,view_id=view)
        path='host'
        raw_name=token(graph.key('weight',attr))
        norm_name=names.get(graph.key('norm',attr))
        norm_needed=norm_name is not None and (norm_name in plan.admit or not plan.feasible)
        source_array=None
        if (self.policy in ('repair_lru','cost_size','adaptive') and attr is not None and norm_needed
                and raw_name in self.model.components and self.model.valid(raw_name)):
            raw=self.model.components[raw_name]
            source_array=self.arena.slots[raw.allocation_id].array
            gpu_cost=24e-6+len(graph.col)*32/350e9
            gpu_specs=[replace(s,build_cost=gpu_cost if s.key==norm_name else s.build_cost) for s in specs]
            # Retained source is explicitly protected throughout construction.
            retain=self.model.plan(gpu_specs+[raw.spec],workspace_bytes=self.executor.workspace_bytes,view_id=view)
            candidates=[('host',plan),('device_retain',retain)]
            if self.policy=='adaptive':
                overwrite_specs=[replace(s,overwrite_from=(raw_name,),prefer_overwrite=True)
                                 if s.key==norm_name else s for s in gpu_specs]
                overwrite=self.model.plan(overwrite_specs,workspace_bytes=self.executor.workspace_bytes,view_id=view)
                if any(a['kind']=='overwrite' and a['source']==raw_name for a in overwrite.actions):
                    candidates.append(('device_overwrite',overwrite))
            feasible=[(label,p) for label,p in candidates if p.feasible]
            def score(item):
                label,p=item
                lost=sum(self.model._retention_value(k) for k in set(p.evict)-set(p.admit)
                         if k in self.model.components) if self.policy=='adaptive' else 0.
                return p.estimated_cost+lost
            if feasible: path,plan=min(feasible,key=score)
        self.counters['planning_ns']+=int((time.perf_counter()-t)*1e9)
        if not plan.feasible: raise MemoryError(plan.reason)
        # The byte ledger plans physical capacity. Fragmentation uses the same
        # best-fit arena for every policy; failed allocations are reported.
        for action in plan.actions:
            kind=action['kind']
            if kind=='release':
                self.arena.release(action['allocation_id'])
            elif kind=='evict':
                self.device_versions.pop(action['key'],None)
                self.counters['evictions']+=1
            elif kind=='admit':
                name=action['key']; key=self.semantic[name]
                try:
                    arr=self.arena.allocate(action['allocation_id'],shapes[key],dtypes[key])
                except MemoryError:
                    if any(a.pins for a in self.model.allocations.values()): raise
                    self.arena.compact()
                    arr=self.arena.allocate(action['allocation_id'],shapes[key],dtypes[key])
                if not (path!='host' and key[1] in ('norm','dangling')):
                    data=graph.get(key)
                    arr.set(data)
                    self.counters['h2d_bytes']+=data.nbytes
                self.counters['view_builds']+=1
                self.device_versions[name]=graph.version(key)
            elif kind=='repair':
                name=action['key']; key=self.semantic[name]; pos=repairs[name]
                arr=self.arena.slots[action['allocation_id']].array
                self.executor.patch(arr,pos,graph.repair_values(key,pos))
                self.counters['h2d_bytes']+=len(pos)*12
                self.counters['repaired_values']+=len(pos)
                self.counters['repairs']+=1
                self.device_versions[name]=graph.version(key)
            elif kind=='reuse': self.counters['hits']+=1
            elif kind=='overwrite':
                self.device_versions.pop(action['source'],None)
                self.device_versions[action['key']]=graph.version(self.semantic[action['key']])
                self.counters['overwrites']+=1
            elif kind not in ('workspace_resize','drop_view'): raise RuntimeError(('unexpected planner action',kind))
        if path!='host':
            state=plan._state
            source_array=self.arena.slots[raw.allocation_id].array
            dest=self.arena.slots[state.components[norm_name].allocation_id].array
            dangling=self.arena.slots[state.components[names[graph.key('dangling',attr)]].allocation_id].array
            col=self.arena.slots[state.components[names[graph.key('col')]].allocation_id].array
            self.executor.normalize(graph,col,source_array,dest,dangling)
            self.counters['device_conversions']+=1
            self.counters['avoided_h2d_bytes']+=len(graph.col)*8+graph.n
        cp.cuda.runtime.deviceSynchronize()
        self.model.commit(plan)
        if self.arena.live != self.model.live_bytes:
            raise AssertionError(('physical/ledger mismatch',self.arena.live,self.model.live_bytes))
        arrays={key:self.arena.slots[self.model.components[name].allocation_id].array for key,name in names.items()}
        self.counters['path_'+path]+=1
        return arrays,names

    @request_boundary
    def query(self, graph_name, function, attr=None, source=0, result='array', selected=None):
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
                output[output>=1e99]=np.inf
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
        self.last['kernel_backend']=('original_native_'+function if native is not None else 'common_reference')
        self.counters['d2h_bytes']+=metrics['d2h_bytes']
        self.counters['h2d_bytes']+=metrics['result_indices_h2d_bytes']
        self.counters['result_d2h_bytes']+=metrics['result_d2h_bytes']
        self.counters['result_indices_h2d_bytes']+=metrics['result_indices_h2d_bytes']
        self.counters['result_gather_launches']+=metrics['result_gather_launches']
        self.events.append(self.last)
        return answer

    def _compact_update_logs(self):
        if not self._log_compaction_pending:return
        t=time.perf_counter();oldest={}
        for name,version in self.device_versions.items():
            if name not in self.model.components:continue
            key=self.semantic[name];graph=self.graphs[key[0]]
            if version[0]!=graph.generation:continue
            oldest[key]=min(oldest.get(key,version[1]),version[1])
        removed=sum(g.compact_changes(oldest) for g in self.graphs.values())
        before=sum(len(h) for h in self.model.update_scopes.values())
        self.model.compact_update_log()
        self.counters['host_update_log_entries_removed']+=removed
        self.counters['model_update_log_entries_removed']+=before-sum(len(h) for h in self.model.update_scopes.values())
        self.counters['update_log_compaction_ns']+=int((time.perf_counter()-t)*1e9)
        self._log_compaction_pending=(any(self.model.update_scopes.values()) or
            any(history for g in self.graphs.values() for history in g.changes.values()))

    def stats(self):
        return dict(self.counters,arena_reserved_bytes=self.arena.budget,
                    live_capacity_bytes=self.arena.live,peak_capacity_bytes=self.arena.peak,
                    suballocations=self.arena.allocations,workspace_bytes=self.executor.workspace_bytes,
                    compactions=self.arena.compactions,compaction_d2d_bytes=self.arena.compaction_bytes,
                    workspace_vertex_capacity=self.executor.nmax,
                    host_update_log_payload_bytes=sum(p.nbytes for g in self.graphs.values() for h in g.changes.values() for _,p in h),
                    host_update_log_entries=sum(len(h) for g in self.graphs.values() for h in g.changes.values()),
                    model_update_log_entries=sum(len(h) for h in self.model.update_scopes.values()),
                    host_array_bytes=sum(g.row.nbytes+g.col.nbytes+g.target.nbytes+g.out_row.nbytes+g.out_order.nbytes+g.in_to_out.nbytes
                        +sum(a.nbytes for a in g.attributes.values())+sum(a.nbytes for a in g.arrays.values())
                        +sum(p.nbytes for h in g.changes.values() for _,p in h) for g in self.graphs.values()),
                    native_host_workspace_bytes=0 if self.native_sssp is None else self.native_sssp.host_partial.nbytes)

    @request_boundary
    def close(self):
        self.executor=None
        self.native_bfs=None
        self.native_sssp=None
        self.sources=None
        self.model=None
        self.arena.close()
