"""Native outgoing CSR admission without rebuilding its outgoing CSR.

Initial CSR is copied and validated inside register().  The outgoing layout is
published directly; an incoming layout and immutable journal authority are
produced once.  All later mutation, budgeting, recovery and kernels are inherited.
"""
from types import MappingProxyType
from scipy.sparse import csr_matrix
from easygraph.gpu._arena.input_hotpath.csr_session import CSRInput
from functools import partial
from time import perf_counter
import weakref
import numpy as np
from easygraph.gpu._arena.replacement import ReplacementSession
from easygraph.gpu._arena.round8_columns.authority import Store
from easygraph.gpu._arena.round8_columns.session import PacketPrice
from easygraph.gpu._arena.round8_cold.authority import ColdPreparation
from easygraph.gpu._arena.round8_lazy.session import LazySession
from easygraph.gpu._arena.round8_startup.session import StartupTransferKernels
from easygraph.gpu._arena.round7_recovery.host_authority_v2 import IndexedEdgeMap
from easygraph.gpu._arena.round2_topology.gpu_adapter import PreparedState
from easygraph.gpu._arena.round2_topology.state import Direction
from easygraph.gpu._arena.retained_node_map import ImmutableNodeDomain


class CSRPreparation(ColdPreparation):
    def __init__(self, authority, labels, nodes, state):
        super().__init__(authority,labels,nodes,None)
        self.state=state
    @property
    def payload_bytes(self):
        if self.state is None:return 0
        state=self.state
        arrays=[state.nodes,state.outgoing.row,state.outgoing.col,state.outgoing.ids,
                state.incoming.row,state.incoming.col,state.incoming.ids,
                state.values,state.in_weight,state.sums,state.degree]
        return sum(x.nbytes for x in {id(x):x for x in arrays}.values())
    def release(self):
        self.state=None
        super().release()


class LinearCSRStore(Store):
    def _input(self,nodes,edges,weights):
        if not isinstance(edges,CSRInput):return super()._input(nodes,edges,weights)
        started=perf_counter()
        try:
            labels_array=np.asarray(nodes)
            row=np.asarray(edges.row);col=np.asarray(edges.col)
            if labels_array.ndim!=1 or labels_array.dtype.kind not in 'iu':raise ValueError('Sorted integer node labels required')
            if labels_array.dtype.kind=='u' and labels_array.size and int(labels_array.max())>=1<<63:raise ValueError('Node label exceeds int64')
            nodes=labels_array.astype(np.int64,copy=True);n=len(nodes)
            if n>=1<<31 or len(nodes)>1 and np.any(nodes[1:]<=nodes[:-1]):raise ValueError('Sorted unique int64 node labels required')
            if row.ndim!=1 or col.ndim!=1 or row.dtype.kind not in 'iu' or col.dtype.kind not in 'iu':raise ValueError('Integer CSR row/column arrays required')
            m=len(col)
            if m>=1<<31 or len(row)!=n+1 or int(row[0])!=0 or int(row[-1])!=m or np.any(row[1:]<row[:-1]):raise ValueError('Invalid CSR row offsets')
            if m and (int(col.min())<0 or int(col.max())>=n):raise ValueError('CSR destination outside node domain')
            if weights is not None and np.iscomplexobj(weights):raise ValueError('Real weights required')
            values=np.ones(m,np.float64) if weights is None else np.asarray(weights,np.float64).copy()
            if values.ndim!=1 or len(values)!=m or not np.all(np.isfinite(values)) or np.any(values<0):raise ValueError('Finite nonnegative edge weights required')
            row=row.astype(np.int32,copy=True);col=col.astype(np.int32,copy=True)
            counts=np.diff(row);src=np.repeat(np.arange(n,dtype=np.int32),counts)
            canonical=not m or not np.any((src[1:]==src[:-1]) & (col[1:]<=col[:-1]))
            pairs=np.empty((m,2),np.int64)
            pairs[:,0]=nodes[src];pairs[:,1]=nodes[col]
            if not canonical:
                self.cold_count['csr_noncanonical_fallbacks']+=1
                return super()._input(nodes,pairs,values)
            base=IndexedEdgeMap(pairs,values)
            labels=tuple(nodes.tolist())
            state=PreparedState.__new__(PreparedState)
            state.nodes=nodes
            domain=ImmutableNodeDomain.__new__(ImmutableNodeDomain)
            owned_mapping={v:i for i,v in enumerate(labels)}
            domain.nodes=labels;domain.n=n;domain.node_to_id=MappingProxyType(owned_mapping)
            domain.source_mapping_identity=id(owned_mapping);domain.registration_seconds=0.
            state.domain=domain
            state.m=m;state.capacity=max(1,2*m)
            state.values=np.empty(state.capacity,np.float64);state.values[:m]=values
            state.outgoing=Direction(row,col,np.arange(m,dtype=np.int32))
            incoming=csr_matrix((state.outgoing.ids,col,row),shape=(n,n),copy=False).tocsc()
            state.incoming=Direction(incoming.indptr,incoming.indices,incoming.data)
            state.in_weight=values[incoming.data].copy()
            state.degree=counts.astype(np.float64)
            state.sums=np.bincount(src,weights=values,minlength=n).astype(np.float64)
            state.graph=None;state.version=0;state.readers=0
            state.initialization_seconds=perf_counter()-started
            self._cold_frames[-1]=CSRPreparation(weakref.ref(base),labels,nodes,state)
            self.cold_count['csr_direct_admissions']+=1
            self.cold_count['csr_outgoing_rebuilds_avoided']+=1
            self.cold_count['csr_persistent_authority_pair_bytes']+=pairs.nbytes
            self._cold_observe()
            return labels,frozenset(labels),base
        finally:self.cold_count['csr_input_seconds']+=perf_counter()-started

    def _materialize(self,ref,nodes,edges,previous,changes,version):
        prep=self._cold_frames[-1] if self._cold_frames else None
        if not isinstance(prep,CSRPreparation) or not prep.matches(nodes,edges,previous,changes,version):
            return super()._materialize(ref,nodes,edges,previous,changes,version)
        self._remove_certificate(ref)
        state=prep.state;state._r8_authority=weakref.ref(edges);prep.consumed=True
        self.cold_count['csr_prepared_publications']+=1
        self.cold_last_identity=dict(authority=id(edges),labels=id(nodes),prepared_nodes=id(state.nodes),
            published_nodes=id(state.nodes),shared_domain_labels=state.domain.nodes is nodes,consumed_once=True)
        prep.release();self._cold_observe()
        return state


class LinearCSRSession(ReplacementSession):
    """Same typed-session API; each input may use COO or CSRInput in 'edges'."""
    def __init__(self,*args,lazy_module=True,change_budget_bytes=8<<20,
                 input_lut_budget_bytes=8<<20,packed_input_sort=True,**kwargs):
        kernels=StartupTransferKernels('nvrtc','demand')
        factory=partial(LinearCSRStore,change_budget_bytes=change_budget_bytes,
                        input_lut_budget_bytes=input_lut_budget_bytes,packed_input_sort=packed_input_sort)
        LazySession.__init__(self,*args,lazy_module=lazy_module,store_factory=factory,**kwargs)
        self.fast_input=True;self.input_lut_budget_bytes=input_lut_budget_bytes
        self.packed_input_sort=packed_input_sort;self.packet_price=PacketPrice()
        self.transfer_kernels=kernels
