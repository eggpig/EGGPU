"""Exact stable-edge transitions of compact CSR and factorized PR state.

All controls share aggregation, lookup, local insertion and immutable node IDs.
Publication follows all fallible preparation. Fixed-domain readers execute only
between updates; existing canonical/sum arrays may be patched at publication.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import time
import numpy as np
from easygraph.gpu._arena.factorized_graph import FactorizedGraph
from easygraph.gpu._arena.retained_node_map import ImmutableNodeDomain,RetainedNodeMapHostGraph


@dataclass
class Direction:
    row: np.ndarray
    col: np.ndarray
    ids: np.ndarray


@dataclass
class Delta:
    src: np.ndarray
    dst: np.ndarray
    ids: np.ndarray
    values: np.ndarray
    new: np.ndarray
    changed_rows: np.ndarray


def endpoints(edges,nodes):
    edges=np.asarray(edges)
    if edges.ndim!=2 or edges.shape[1]!=2 or not np.issubdtype(edges.dtype,np.integer):
        raise ValueError('Integer endpoint pairs required')
    ids=np.searchsorted(nodes,edges)
    if ids.size and (np.any(ids>=len(nodes)) or np.any(nodes[np.minimum(ids,len(nodes)-1)]!=edges)):
        raise ValueError('Endpoint outside fixed node domain')
    return ids.astype(np.int32)


def aggregate(edges,nodes,values=None,mode='add'):
    ids=endpoints(edges,nodes)
    if values is None:values=np.ones(len(ids),np.float64)
    values=np.asarray(values,np.float64)
    if values.shape!=(len(ids),) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Finite nonnegative edge values required')
    keys=(ids[:,0].astype(np.uint64)<<np.uint64(32))|ids[:,1].astype(np.uint64)
    unique,inverse=np.unique(keys,return_inverse=True)
    totals=np.zeros(len(unique),np.float64)
    if mode=='add':np.add.at(totals,inverse,values)
    elif mode=='set':
        last=np.full(len(unique),-1,np.int64);np.maximum.at(last,inverse,np.arange(len(ids)))
        totals=values[last]
    else:raise ValueError('Unknown update mode')
    return (unique>>np.uint64(32)).astype(np.int32),(unique&np.uint64(0xffffffff)).astype(np.int32),totals


def locate(layout,src,dst):
    positions=np.empty(len(src),np.int32);present=np.empty(len(src),bool)
    for j,(u,v) in enumerate(zip(src,dst)):
        lo,hi=int(layout.row[u]),int(layout.row[u+1])
        p=lo+int(np.searchsorted(layout.col[lo:hi],v))
        positions[j]=p;present[j]=p<hi and layout.col[p]==v
    return positions,present


def insert_layout(old,src,dst,ids,payload=None):
    """Same paid local merge primitive for ordinary and joint controls.

    The old->new map is explicitly allocated even though host contiguous copies
    can use runs. It is the concrete map needed by a future device migration.
    Payload is optional, and represents only a currently requested attribute.
    """
    if not len(src):return old,None,payload,0
    order=np.lexsort((dst,src));src,dst,ids=src[order],dst[order],ids[order]
    before,present=locate(old,src,dst)
    if np.any(present):raise AssertionError('Insertion primitive received an existing edge')
    m=len(old.col);k=len(src)
    row=np.zeros_like(old.row)
    np.add.at(row,src+1,1);np.cumsum(row,out=row);row+=old.row
    col=np.empty(m+k,np.int32);edge_ids=np.empty(m+k,np.int32)
    mapping=np.empty(m,np.int32)
    result_payload=None if payload is None else np.empty(m+k,np.float64)
    start=0;largest_range_temp=0
    for j,stop in enumerate([*map(int,before),m]):
        count=stop-start;target=start+j
        col[target:target+count]=old.col[start:stop]
        edge_ids[target:target+count]=old.ids[start:stop]
        positions=np.arange(target,target+count,dtype=np.int32)
        largest_range_temp=max(largest_range_temp,positions.nbytes)
        mapping[start:stop]=positions
        if payload is not None:result_payload[target:target+count]=payload[start:stop]
        if j<k:
            col[stop+j]=dst[j];edge_ids[stop+j]=ids[j]
        start=stop
    return Direction(row,col,edge_ids),mapping,result_payload,largest_range_temp


def unique_array_bytes(arrays):
    seen=set();total=0
    for a in arrays:
        while isinstance(a.base,np.ndarray):a=a.base
        if id(a) not in seen:seen.add(id(a));total+=a.nbytes
    return total


class State:
    def __init__(self,nodes,edges,values=None):
        started=time.perf_counter()
        self.nodes=np.asarray(nodes,np.int64).copy()
        if self.nodes.ndim!=1 or np.any(self.nodes[1:]<=self.nodes[:-1]):raise ValueError('Sorted unique node domain required')
        self.domain=ImmutableNodeDomain(tuple(map(int,self.nodes)),{int(v):i for i,v in enumerate(self.nodes)})
        src,dst,weight=aggregate(edges,self.nodes,values)
        graph=RetainedNodeMapHostGraph('r2',len(nodes),src,dst,{'message_count':weight},self.domain)
        m=len(src);self.m=m
        self.capacity=max(1,2*m)
        self.values=np.empty(self.capacity,np.float64);self.values[:m]=weight
        ids=np.arange(m,dtype=np.int32)
        self.outgoing=Direction(graph.out_row,graph.get(graph.key('out_col')),ids)
        self.incoming=Direction(graph.row,graph.col,ids[graph.in_to_out])
        self.in_weight=graph.attributes['message_count']
        self.degree=np.diff(graph.out_row).astype(np.float64)
        self.sums=np.bincount(src,weights=weight,minlength=len(nodes))
        self.graph=None;self.version=0;self.readers=0
        self.initialization_seconds=time.perf_counter()-started

    def arrays(self):
        arrays=[self.nodes,self.values,self.degree,self.sums,self.in_weight]
        for d in (self.outgoing,self.incoming):arrays.extend((d.row,d.col,d.ids))
        if self.graph is not None:
            g=self.graph;arrays += [g.row,g.col,g.target,g.out_row,g.out_order,g.in_to_out,*g.attributes.values(),*g.arrays.values()]
        return arrays

    def owned_array_bytes(self):return unique_array_bytes(self.arrays())

    @contextmanager
    def lease(self):
        self.readers+=1
        try:yield self
        finally:self.readers-=1

    def fingerprint(self):
        import hashlib
        h=hashlib.sha256()
        for a in [self.values[:self.m],self.degree,self.sums,self.in_weight,
                  self.outgoing.row,self.outgoing.col,self.outgoing.ids,self.incoming.row,self.incoming.col,self.incoming.ids]:
            h.update(a.tobytes())
        h.update(np.asarray([self.version,self.m,self.capacity],np.int64).tobytes())
        return h.hexdigest()

    def prepare_delta(self,edges,values=None,mode='add'):
        src,dst,weight=aggregate(edges,self.nodes,values,mode)
        positions,present=locate(self.outgoing,src,dst)
        if self.m+int(np.sum(~present))>=2**31:raise ValueError('Edge positions must fit signed int32')
        ids=np.empty(len(src),np.int32)
        ids[present]=self.outgoing.ids[positions[present]]
        ids[~present]=self.m+np.arange(np.sum(~present),dtype=np.int32)
        if mode=='add':weight[present]+=self.values[ids[present]]
        if not np.isfinite(weight).all():raise ValueError('Updated weight overflow')
        return Delta(src,dst,ids,weight,~present,np.unique(src))

    def transition(self,edges,*,control='shared_delta',values=None,mode='add',fail_after=None):
        if control not in ('global_rebuild','shared_delta','joint_order'):raise ValueError(control)
        if self.readers:raise RuntimeError('Topology transaction waits for active readers')
        start=last=time.perf_counter();phase={}
        def mark(name):
            nonlocal last
            now=time.perf_counter();phase[name]=now-last;last=now
            if fail_after==name:raise RuntimeError('Injected pre-publication failure: '+name)
        delta=self.prepare_delta(edges,values,mode);mark('aggregate_and_logical_lookup')
        count=int(delta.new.sum());new_m=self.m+count
        capacity=self.capacity;canonical=self.values
        if new_m>capacity:
            capacity=max(new_m,2*capacity);canonical=np.empty(capacity,np.float64);canonical[:self.m]=self.values[:self.m]
        rebuild=control=='global_rebuild' and count>0
        order=('outgoing',) if rebuild else ('outgoing','incoming')
        order_peaks={}
        if control=='joint_order' and count:
            sizes={};temporary={}
            for direction in ('outgoing','incoming'):
                old=getattr(self,direction)
                s,d=(delta.src,delta.dst) if direction=='outgoing' else (delta.dst,delta.src)
                positions,_=locate(old,s[delta.new],d[delta.new])
                temporary[direction]=int(np.diff(np.r_[0,np.sort(positions),self.m]).max(initial=0))*4
                sizes[direction]=(len(self.nodes)+1)*4+new_m*8+self.m*4+(new_m*8 if direction=='incoming' else 0)
            # Both old layouts and both old->new maps remain live until atomic
            # publication. Only the order of identical primitives changes.
            for candidate in (('outgoing','incoming'),('incoming','outgoing')):
                live=self.owned_array_bytes();peak=live
                for direction in candidate:
                    live+=sizes[direction];peak=max(peak,live+temporary[direction])
                order_peaks['/'.join(candidate)]=peak
            order=min((('outgoing','incoming'),('incoming','outgoing')),
                key=lambda candidate:order_peaks['/'.join(candidate)])
        mark('capacity_and_order')
        outputs={};maps={};scratch={}
        for direction in order:
            old=getattr(self,direction)
            s,d=(delta.src,delta.dst) if direction=='outgoing' else (delta.dst,delta.src)
            payload=self.in_weight if direction=='incoming' and not rebuild else None
            layout,mapping,newraw,temporary=insert_layout(old,s[delta.new],d[delta.new],delta.ids[delta.new],payload)
            outputs[direction]=layout;maps[direction]=mapping;scratch[direction]=temporary
            if direction=='incoming' and not rebuild:
                if newraw is self.in_weight:newraw=self.in_weight.copy()
                locations,present=locate(layout,delta.dst,delta.src)
                if not np.all(present):raise AssertionError('Changed edge absent after transition')
                newraw[locations]=delta.values
                in_weight=newraw
            mark(direction+'_layout_and_active_payload')
        outgoing=outputs['outgoing'];incoming=outputs.get('incoming',self.incoming)
        sorted_patch=np.argsort(delta.ids);patchids=delta.ids[sorted_patch];patchvalues=delta.values[sorted_patch]
        def values_for(ids):
            vals=canonical[ids].copy()
            if len(patchids):
                p=np.searchsorted(patchids,ids);present=p<len(patchids)
                present[present] &= patchids[p[present]]==ids[present]
                vals[present]=patchvalues[p[present]]
            return vals
        graph=None
        if rebuild:
            sources=np.repeat(np.arange(len(self.nodes),dtype=np.int32),np.diff(outgoing.row))
            weight=canonical[outgoing.ids]
            locations,present=locate(outgoing,delta.src,delta.dst)
            if not np.all(present):raise AssertionError('Changed edge absent from rebuild input')
            weight[locations]=delta.values
            graph=FactorizedGraph.wrap(RetainedNodeMapHostGraph('r2',len(self.nodes),sources,outgoing.col,
                {'message_count':weight},self.domain))
            outgoing=Direction(graph.out_row,graph.get(graph.key('out_col')),outgoing.ids)
            incoming=Direction(graph.row,graph.col,outgoing.ids[graph.in_to_out])
            inverse=np.empty(new_m,np.int32);inverse[incoming.ids]=np.arange(new_m,dtype=np.int32)
            maps['incoming']=inverse[self.incoming.ids]
            in_weight=graph.attributes['message_count']
            degree=graph.get(graph.key('source_sum'))
            sums=graph.get(graph.key('source_sum','message_count'))
            sum_values=sums[delta.changed_rows]
        else:
            sum_values=np.empty(len(delta.changed_rows),np.float64)
            for i,u in enumerate(delta.changed_rows):
                sum_values[i]=values_for(outgoing.ids[outgoing.row[u]:outgoing.row[u+1]]).sum(dtype=np.float64)
            degree,sums=self.degree,self.sums
        mark('active_state_and_source_sums')
        old_arrays=self.arrays()
        temporary_arrays=[a for a in maps.values() if a is not None]
        new_arrays=[outgoing.row,outgoing.col,outgoing.ids,incoming.row,incoming.col,incoming.ids,in_weight,canonical,degree,sums]
        checkpoint_bytes=unique_array_bytes(old_arrays+new_arrays+temporary_arrays)
        # Final boundary publication performs validated indexed assignments only.
        # No view is exposed until every allocation/merge/repair above succeeds.
        mark('prepublication_checks')
        canonical[delta.ids]=delta.values
        if not rebuild:
            sums[delta.changed_rows]=sum_values
            degree[delta.changed_rows]=(outgoing.row[delta.changed_rows+1]-outgoing.row[delta.changed_rows])
        self.values,self.capacity,self.m=canonical,capacity,new_m
        self.outgoing,self.incoming,self.in_weight=outgoing,incoming,in_weight
        self.degree,self.sums,self.graph=degree,sums,graph
        self.version+=1
        phase['publication']=time.perf_counter()-last
        return dict(seconds=time.perf_counter()-start,phases=phase,control=control,new_edges=count,
            changed_edges=len(delta.ids),changed_source_rows=len(delta.changed_rows),order=list(order),
            known_array_order_peak_estimates=order_peaks,
            old_to_new_maps=maps,delta=delta,
            old_new_maps_checkpoint_array_bytes=checkpoint_bytes,
            mapping_bytes=sum(a.nbytes for a in maps.values() if a is not None),
            largest_range_temporary_bytes=max(scratch.values(),default=0),
            persistent_array_bytes=self.owned_array_bytes(),canonical_capacity_bytes=canonical.nbytes,
            canonical_live_bytes=new_m*8)
