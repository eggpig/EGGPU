"""R2 host transitions under identical scoped GPU refresh and fixed kernels.

This first complete-session test isolates the host transition mechanism. The
three controls all upload missing/invalid GPU components through the same
RepresentationSession path; CPU position maps are charged but not passed off as
already executed device migration.
"""
from collections import defaultdict
from dataclasses import dataclass
import time
import numpy as np
from easygraph.gpu._arena.factorized_graph import FactorizedGraph
from easygraph.gpu._arena.factorized_session import RepresentationSession,portfolio_request_bytes
from easygraph.gpu._arena.session_manager import request_boundary
from easygraph.gpu._arena.round2_topology.state import State,unique_array_bytes


class PreparedState(State):
    def prepare_delta(self,*args,**kwargs):
        prepared=getattr(self,'_provided_delta',None)
        return prepared if prepared is not None else super().prepare_delta(*args,**kwargs)

    def apply_prepared(self,edges,delta,control,fail_after=None):
        self._provided_delta=delta
        try:return self.transition(edges,control=control,fail_after=fail_after)
        finally:del self._provided_delta


@dataclass
class ArrayShape:
    size:int
    @property
    def shape(self):return (self.size,)
    def __len__(self):return self.size


class GraphAdapter(FactorizedGraph):
    """Explicit adapter preserving stable-state methods, with no class rewrite."""
    def __init__(self,state,name='superuser',generation=0,attribute_version=0):
        self.r2_state=state;self.name=name;self.n=len(state.nodes)
        self.nodes=state.domain.nodes;self.node_to_id=state.domain.node_to_id
        self.row=state.incoming.row;self.col=state.incoming.col;self.out_row=state.outgoing.row
        self.attributes={'message_count':state.in_weight}
        self.generation=generation;self.versions=defaultdict(int,{'message_count':attribute_version})
        self.arrays={};self.changes=defaultdict(list);self.change_floors=defaultdict(int)
        self._registered_keys=set();self._active_key_callback=None
        self.host_prepare_seconds=0.;self.view_builds=0;self.repaired_values=0
        self.sum_repair_rows=0;self.sum_repair_scanned_edges=0;self.norm_closure_entries=0
        self.input_graph_class='R2 stable-edge compact state'

    def get(self,key):
        started=time.perf_counter();self.register_keys([key]);kind,attr=key[1:]
        if attr not in (None,'message_count'):raise KeyError(attr)
        if kind=='row':result=self.row
        elif kind=='col':result=self.col
        elif kind=='out_row':result=self.out_row
        elif kind=='out_col':result=self.r2_state.outgoing.col
        elif kind=='weight':result=self.r2_state.in_weight
        elif kind=='source_sum':result=self.r2_state.degree if attr is None else self.r2_state.sums
        elif kind=='out_weight':
            if key not in self.arrays:self.arrays[key]=self.r2_state.values[self.r2_state.outgoing.ids]
            result=self.arrays[key]
        else:raise KeyError('Representation not registered by this fixed-factorized experiment: '+str(key))
        self.host_prepare_seconds+=time.perf_counter()-started
        return result

    def dirty_positions(self,key,old_version):
        # All controls deliberately share the same GPU refresh contract here.
        # Equal semantic versions can hit; changed attributes/layouts rebuild.
        return np.empty(0,np.int32) if old_version==self.version(key) else None


class RawTransitionSession(RepresentationSession):
    def __init__(self,state,budget_bytes,control):
        self.control=control;self.state=state;self.raw_transition_records=[]
        super().__init__([GraphAdapter(state)],budget_bytes,costs=None,
            representation='factorized',sum_builder='host',selection='greedy',kernel_backend='native_paths')

    @request_boundary
    def apply_raw(self,edges,values=None,mode='add',fail_after=None):
        started=time.perf_counter();state=self.state
        if state.readers or any(a.pins for a in self.model.allocations.values()):raise RuntimeError('Update waits for readers')
        delta=state.prepare_delta(edges,values,mode)
        logical_done=time.perf_counter()
        new_m=state.m+int(delta.new.sum())
        shape=type('ShapeGraph',(),dict(n=len(state.nodes),row=ArrayShape(len(state.nodes)+1),
                   col=ArrayShape(new_m),attributes={'message_count':None}))()
        if self.executor.workspace_bytes+portfolio_request_bytes(shape)>self.arena.budget:
            raise MemoryError('Updated complete request does not fit fixed GPU arena')
        old=self.graphs['superuser']
        result=state.apply_prepared(edges,delta,self.control,fail_after)
        graph=GraphAdapter(state,generation=old.generation+int(result['new_edges']>0),
                           attribute_version=old.versions['message_count']+1)
        self.graphs['superuser']=graph;self._bind_active_keys(graph)
        if result['new_edges']:
            self.model.update('superuser:topology');self.counters['topology_replacements']+=1
        self.model.update('superuser:attr:message_count')
        self._log_compaction_pending=True
        maps=result.pop('old_to_new_maps');result.pop('delta')
        # These host maps were constructed/charged. No GPU migration is claimed.
        result['mapping_bytes']=sum(a.nbytes for a in maps.values() if a is not None)
        del maps
        result.update(raw_boundary_seconds=time.perf_counter()-started,
            logical_prepare_seconds=logical_done-started,device_transition='common scoped refresh on next request')
        self.raw_transition_records.append(result)
        self.counters['raw_update_events']+=len(edges)
        return result

    @request_boundary
    def query(self,*args,**kwargs):
        with self.state.lease():return super().query(*args,**kwargs)

    def stats(self):
        # Count actual R2 arrays. Calling legacy HostGraph.stats would require
        # materializing unused target/order arrays solely for the metric.
        return dict(self.counters,arena_reserved_bytes=self.arena.budget,
            live_capacity_bytes=self.arena.live,peak_capacity_bytes=self.arena.peak,
            suballocations=self.arena.allocations,workspace_bytes=self.executor.workspace_bytes,
            compactions=self.arena.compactions,compaction_d2d_bytes=self.arena.compaction_bytes,
            workspace_vertex_capacity=self.executor.nmax,
            host_array_bytes=unique_array_bytes(self.state.arrays()+[a for g in self.graphs.values() for a in g.arrays.values()]),
            host_update_log_payload_bytes=0,host_update_log_entries=0,
            model_update_log_entries=sum(len(v) for v in self.model.update_scopes.values()),
            native_host_workspace_bytes=self.native_sssp.host_partial.nbytes,
            fixed_representation='factorized',fixed_sum_builder='host',r2_control=self.control,
            factorized_kernel_module_loaded=self.executor.factor_kernels is not None,
            conditional_cost_provenance={'status':'fixed representation; no calibration required'},
            scaled_workspace='@native_bfs_output')
