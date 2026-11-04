"""Fixed materialized/factorized PR controls with identical session boundaries.

Representation selection is deliberately absent here. Each fixed control uses
shared-component LRU, dependency repair, original BFS/SSSP, and the same arena.
Factorization is prior capability, not an EGGPU-only option or novelty claim.
"""
import time
import weakref
import numpy as np
from easygraph.gpu._arena.conditional_session import ConditionalGraphSession
from easygraph.gpu._arena.factorized_graph import FactorizedGraph
from easygraph.gpu._arena.factorized_runtime import FactorizedExecutor
from easygraph.gpu._arena.session_manager import GraphSession,request_boundary,token,workspace_bytes,workspace_layout,largest_request_bytes
from easygraph.gpu._arena.state_model import ComponentSpec
from easygraph.gpu._arena.gpu_runtime import Arena,cp


def portfolio_workspace_layout(nmax,backend,patch_capacity=8192):
    layout=workspace_layout(nmax,backend,patch_capacity)
    if backend=='reference':layout.append(('@scaled_rank',(nmax,),np.float64))
    return layout


def portfolio_workspace_bytes(nmax,backend,patch_capacity=8192):
    return sum(Arena.capacity(shape,dtype) for _,shape,dtype in portfolio_workspace_layout(nmax,backend,patch_capacity))


def factorized_request_bytes(graph,weighted=True):
    cap=lambda a,d:Arena.capacity((a,),d)
    return cap(graph.n+1,np.int32)+cap(len(graph.col),np.int32)+cap(graph.n,np.float64)+(cap(len(graph.col),np.float64) if weighted else 0)


def portfolio_request_bytes(graph):
    return max(largest_request_bytes(graph),factorized_request_bytes(graph,bool(graph.attributes)))


class RepresentationSession(ConditionalGraphSession):
    def __init__(self,graphs,budget_bytes,*,costs,representation='factorized',sum_builder='host',selection='greedy',
                 kernel_backend='native_paths',**kwargs):
        if representation not in ('materialized','factorized'):raise ValueError('Unknown fixed representation')
        if sum_builder not in ('host','device'):raise ValueError('Unknown fixed sum builder')
        graphs=[FactorizedGraph.wrap(g) for g in graphs]
        nmax=max(g.n for g in graphs)
        required=portfolio_workspace_bytes(nmax,kernel_backend)+max(portfolio_request_bytes(g) for g in graphs)
        if required>budget_bytes:raise MemoryError('One complete representation portfolio request exceeds budget')
        self.representation,self.sum_builder=representation,sum_builder
        super().__init__(graphs,budget_bytes,costs=costs,selection=selection,kernel_backend=kernel_backend,**kwargs)
        if kernel_backend=='reference':
            self.arena.allocate('@scaled_rank',(nmax,),np.float64)
            self.executor.workspace_bytes=self.arena.live
            self.model.workspace_capacity=self.arena.live
        self.executor=FactorizedExecutor(self.executor,kernel_backend)
        for graph in self.graphs.values():self._bind_active_keys(graph)
        self.model.assert_invariants()

    def _bind_active_keys(self,graph):
        owner=weakref.ref(self);graph_ref=weakref.ref(graph)
        def active(key):
            session=owner();current=graph_ref()
            return (current is not None and key in current.arrays) or (session is not None and session.model is not None and token(key) in session.model.components)
        graph.set_active_key_callback(active)

    def _requirements(self,graph,function,attr):
        if function=='pr' and self.representation=='factorized':
            return [graph.key('row'),graph.key('col'),graph.key('source_sum',attr)]+([] if attr is None else [graph.key('weight',attr)])
        return super()._requirements(graph,function,attr)

    def prepare(self,graph,function,attr):
        # BFS/SSSP have a single required representation in this fixed portfolio.
        # Every control uses the same scoped cache path and needs no PR source
        # calibration to prepare these functions.
        if function!='pr':return GraphSession.prepare(self,graph,function,attr)
        if self.representation!='factorized':return super().prepare(graph,function,attr)
        started=time.perf_counter();keys=self._requirements(graph,function,attr);view=token((graph.name,'factorized_pr',attr))
        names={key:token(key) for key in keys}
        graph.register_keys(keys)
        if self.model.touch_if_valid(list(names.values()),workspace_bytes=self.executor.workspace_bytes,view_id=view):
            self.counters['planning_ns']+=int((time.perf_counter()-started)*1e9)
            self.counters['hits']+=len(names);self.counters['path_reuse']+=1
            self.last_decision=dict(selection=self.selection,selected='factorized_reuse',gate='valid_hit',candidates={},scores={})
            return {k:self.arena.slots[self.model.components[v].allocation_id].array for k,v in names.items()},names
        shapes={k:((graph.n,) if k[1]=='source_sum' else graph.row.shape if k[1]=='row' else graph.col.shape) for k in keys}
        dtypes={k:np.int32 if k[1] in ('row','col') else np.float64 for k in keys}
        specs=[];repairs={}
        for key,name in names.items():
            self.semantic[name]=key;old=self.device_versions.get(name)
            positions=graph.dirty_positions(key,old) if old is not None else None
            canrepair=key[1] in ('weight','source_sum') and positions is not None
            repairs[name]=positions
            # These numbers express the fixed control's preference for local
            # repair, not a measured time model or representation selector.
            specs.append(ComponentSpec(name,Arena.capacity(shapes[key],dtypes[key]),graph.dependency(key),
                build_cost=1.0,repairable=canrepair,repair_cost=0.0,
                repair_dependencies=(graph.name+':attr:'+str(key[2]),)))
        plan=self.model.plan(specs,workspace_bytes=self.executor.workspace_bytes,view_id=view)
        self.counters['planning_ns']+=int((time.perf_counter()-started)*1e9)
        if not plan.feasible:raise MemoryError(plan.reason)
        sum_name=names[graph.key('source_sum',attr)]
        build_device=self.sum_builder=='device' and attr is not None and sum_name in plan.admit
        for action in plan.actions:
            kind=action['kind']
            if kind=='release':self.arena.release(action['allocation_id'])
            elif kind=='evict':self.device_versions.pop(action['key'],None);self.counters['evictions']+=1
            elif kind=='admit':
                name=action['key'];key=self.semantic[name]
                try:array=self.arena.allocate(action['allocation_id'],shapes[key],dtypes[key])
                except MemoryError:
                    if any(a.pins for a in self.model.allocations.values()):raise
                    self.arena.compact();array=self.arena.allocate(action['allocation_id'],shapes[key],dtypes[key])
                if not (name==sum_name and build_device):
                    t=time.perf_counter();data=graph.get(key)
                    self.counters['factorized_host_build_ns']+=int((time.perf_counter()-t)*1e9)
                    array.set(data);self.counters['h2d_bytes']+=data.nbytes
                    if key[1]=='source_sum':self.counters['source_sum_host_builds']+=1
                self.device_versions[name]=graph.version(key);self.counters['view_builds']+=1
            elif kind=='repair':
                name=action['key'];key=self.semantic[name];positions=repairs[name]
                t=time.perf_counter();values=graph.repair_values(key,positions)
                self.counters['factorized_host_repair_ns']+=int((time.perf_counter()-t)*1e9)
                self.executor.patch(self.arena.slots[action['allocation_id']].array,positions,values)
                self.device_versions[name]=graph.version(key)
                self.counters['h2d_bytes']+=12*len(positions);self.counters['repairs']+=1;self.counters['repaired_values']+=len(positions)
                if key[1]=='source_sum':self.counters['source_sum_rows_repaired']+=len(positions)
            elif kind=='reuse':self.counters['hits']+=1
            elif kind not in ('workspace_resize','drop_view'):raise RuntimeError(('unsupported fixed factorized action',kind))
        def planned_array(key):return self.arena.slots[plan._state.components[names[key]].allocation_id].array
        if build_device:
            sums=planned_array(graph.key('source_sum',attr))
            cp.cuda.runtime.memsetAsync(sums.data.ptr,0,graph.n*8,cp.cuda.get_current_stream().ptr)
            m=len(graph.col)
            if m:self.executor.kernels['degree'](((m+127)//128,),(128,),
                (np.int32(m),planned_array(graph.key('col')),planned_array(graph.key('weight',attr)),sums))
            self.counters['source_sum_gpu_builds']+=1
        cp.cuda.runtime.deviceSynchronize();self.model.commit(plan)
        if self.arena.live!=self.model.live_bytes:raise AssertionError('factorized ledger/arena disagreement')
        self.last_decision=dict(selection=self.selection,selected='factorized_'+self.sum_builder,gate='fixed_representation',candidates={},scores={})
        self.counters['path_factorized_'+self.sum_builder]+=1
        return {k:self.arena.slots[self.model.components[v].allocation_id].array for k,v in names.items()},names

    @request_boundary
    def query(self,*args,**kwargs):
        value=super().query(*args,**kwargs)
        if self.last['function']=='pr':
            self.last['representation']=self.representation
            if self.representation=='factorized':self.last['kernel_backend']='common_factorized_pull'
        return value

    @request_boundary
    def replace_graph(self,graph):
        t=time.perf_counter();graph=FactorizedGraph.wrap(graph);old=self.graphs[graph.name]
        nmax=max(self.executor.nmax,graph.n)
        needed=portfolio_workspace_bytes(nmax,self.kernel_backend,self.executor.patch_capacity)
        prospective=[graph if name==graph.name else g for name,g in self.graphs.items()]
        if needed+max(portfolio_request_bytes(g) for g in prospective)>self.arena.budget:
            raise MemoryError('Replacement portfolio request exceeds fixed arena budget')
        if any(a.pins for a in self.model.allocations.values()):raise ValueError('Replacement waits for readers')
        if nmax>self.executor.nmax:self._grow_workspace(nmax,needed)
        self.counters['retired_source_sum_repaired_rows']+=old.sum_repair_rows
        self.counters['retired_source_sum_scanned_edges']+=old.sum_repair_scanned_edges
        self.counters['retired_norm_closure_entries']+=old.norm_closure_entries
        graph.generation=old.generation+1;self.graphs[graph.name]=graph;self._bind_active_keys(graph)
        for kind in ('topology','mapping','all'):self.model.update(graph.name+':'+kind)
        self.counters['topology_replacements']+=1;self._log_compaction_pending=True
        self.counters['topology_replace_ns']+=int((time.perf_counter()-t)*1e9)
        return time.perf_counter()-t

    def _grow_workspace(self,nmax,needed):
        if self.kernel_backend!='reference':return super()._grow_workspace(nmax,needed)
        started=time.perf_counter();plan=self.model.plan([],workspace_bytes=needed,view_id='@workspace_growth')
        if not plan.feasible:raise MemoryError(plan.reason)
        layout=portfolio_workspace_layout(nmax,self.kernel_backend,self.executor.patch_capacity)
        old=self.executor.workspace_bytes;cp.cuda.runtime.deviceSynchronize()
        for action in plan.actions:
            if action['kind']=='release':self.arena.release(action['allocation_id'])
            elif action['kind']=='evict':
                self.device_versions.pop(action['key'],None);self.counters['evictions']+=1;self.counters['workspace_growth_evictions']+=1
            elif action['kind'] not in ('workspace_resize','drop_view'):raise RuntimeError(('unexpected grow action',action))
        self.arena.resize_workspace(layout);self.executor.bind_workspace(nmax);self.model.commit(plan)
        if self.arena.live!=self.model.live_bytes:raise AssertionError('Growth ledger disagreement')
        self.counters['workspace_growths']+=1;self.counters['workspace_growth_bytes']+=needed-old
        self.counters['workspace_growth_ns']+=int((time.perf_counter()-started)*1e9)

    def stats(self):
        result=(GraphSession.stats(self) if self.costs is None else super().stats())
        if self.costs is None:
            result.update(conditional_selection='fixed_representation',
                conditional_cost_provenance={'status':'not required by fixed factorized control'})
        result.update(fixed_representation=self.representation,fixed_sum_builder=self.sum_builder,
            factorized_kernel_module_loaded=self.executor.factor_kernels is not None,
            scaled_workspace='@scaled_rank' if self.kernel_backend=='reference' else '@native_bfs_output',
            source_sum_repaired_rows=self.counters['retired_source_sum_repaired_rows']+sum(g.sum_repair_rows for g in self.graphs.values()),
            source_sum_scanned_edges=self.counters['retired_source_sum_scanned_edges']+sum(g.sum_repair_scanned_edges for g in self.graphs.values()),
            norm_closure_entries=self.counters['retired_norm_closure_entries']+sum(g.norm_closure_entries for g in self.graphs.values()))
        return result
