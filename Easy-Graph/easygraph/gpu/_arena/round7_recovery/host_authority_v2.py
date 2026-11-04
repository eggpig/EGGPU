"""Typed authority V2: indexed batches, direct cold CSR and atomic add batches.

V1 remains frozen. Optimizations depend on input semantics and numeric domains,
never graph identity. General ordered mutation batches retain V1's path.
"""
from collections import Counter
from dataclasses import replace
from time import perf_counter
import numpy as np

from easygraph.gpu._arena.round4_switch.journal import (Mutation,Snapshot,Receipt,_Journal,CHANGE_DTYPE,
    _changes,retained_footprint)
from easygraph.gpu._arena.round5_input.array_journal_v2 import ArrayEdgeMap,ArrayJournalStore,KEY_DTYPE
from easygraph.gpu._arena.round7_recovery import host_authority as v1


def packed_eligible(pairs):
    return not pairs.size or bool(np.all(pairs>=0) and np.all(pairs<1<<32))


def encode_keys(pairs,packed):
    return ((pairs[:,0].astype(np.uint64)<<np.uint64(32))|pairs[:,1].astype(np.uint64)) if packed else v1._keys(pairs)


class IndexedEdgeMap(ArrayEdgeMap):
    def __init__(self,pairs,values):
        # Called only for validated immutable buffers or freshly built arrays.
        self.pairs=pairs;self.values_array=values
        self.pairs.flags.writeable=False;self.values_array.flags.writeable=False
        self.packed=packed_eligible(pairs)
        self._keys=encode_keys(pairs,self.packed);self._keys.flags.writeable=False

    def positions(self,pairs):
        if self.packed:
            legal=np.all((pairs>=0)&(pairs<1<<32),axis=1)
            keys=encode_keys(pairs,True)
        else:
            legal=np.ones(len(pairs),bool);keys=encode_keys(pairs,False)
        locations=np.searchsorted(self._keys,keys);present=legal & (locations<len(self._keys))
        if len(self._keys):present[present]&=self._keys[locations[present]]==keys[present]
        return locations,present

    def position(self,key):
        hash(key)
        if not isinstance(key,tuple) or len(key)!=2:raise KeyError(key)
        try:
            u,v=map(int,key)
            if (u,v)!=key or not -(1<<63)<=u<1<<63 or not -(1<<63)<=v<1<<63:raise KeyError(key)
        except (TypeError,ValueError,OverflowError) as error:raise KeyError(key) from error
        locations,present=self.positions(np.asarray([(u,v)],np.int64))
        if not present[0]:raise KeyError(key)
        return int(locations[0])


def indexed(base):
    if isinstance(base,IndexedEdgeMap):return base
    pairs,values=base.to_arrays()
    return IndexedEdgeMap(pairs,values)


def fold(base,data):
    if not len(data):return base
    if isinstance(base,v1._BatchBase):base=base.base
    base=indexed(base);pairs,original=base.to_arrays()
    changes=np.column_stack((data['src'],data['dst']))
    positions,present=base.positions(changes);live=(data['flags']&2)!=0
    remove=present & ~live;modify=present & live;insert=~present & live
    values=original.copy();values[positions[modify]]=data['final'][modify]
    if np.any(remove):
        keep=np.ones(len(pairs),bool);keep[positions[remove]]=False
        pairs=pairs[keep];values=values[keep]
        keys=encode_keys(pairs,base.packed)
    else:keys=base._keys
    if np.any(insert):
        additions=changes[insert]
        # Existing packed domain can expand across signed/extreme labels.
        packed=base.packed and packed_eligible(additions)
        if packed!=base.packed:keys=encode_keys(pairs,packed)
        positions=np.searchsorted(keys,encode_keys(additions,packed))
        shifts=np.bincount(positions,minlength=len(pairs)+1)
        destinations=np.arange(len(pairs))+np.cumsum(shifts[:-1])
        targets=positions+np.arange(len(additions))
        result=np.empty((len(pairs)+len(additions),2),np.int64)
        result_values=np.empty(len(result),np.float64)
        result[destinations]=pairs;result_values[destinations]=values
        result[targets]=additions;result_values[targets]=data['final'][insert]
        pairs,values=result,result_values
    return IndexedEdgeMap(pairs,values)


def cold_state(nodes,edges):
    """Direct CSR from validated unique sorted authority; no duplicate sort."""
    from easygraph.gpu._arena.round2_topology.gpu_adapter import PreparedState
    from easygraph.gpu._arena.round2_topology.state import Direction,endpoints
    from easygraph.gpu._arena.retained_node_map import ImmutableNodeDomain
    started=perf_counter();nodes=np.asarray(nodes,np.int64).copy()
    pairs,weights=edges.to_arrays();mapped=endpoints(pairs,nodes)
    src,dst=mapped[:,0],mapped[:,1];n=len(nodes);m=len(src)
    if m>=2**31:raise ValueError('Edge positions must fit signed int32')
    out_count=np.bincount(src,minlength=n);in_count=np.bincount(dst,minlength=n)
    out_row=np.r_[0,np.cumsum(out_count)].astype(np.int32)
    in_row=np.r_[0,np.cumsum(in_count)].astype(np.int32)
    # Source order is already canonical; stable target sort gives (dst,src).
    incoming_order=np.argsort(dst,kind='stable').astype(np.int32)
    ids=np.arange(m,dtype=np.int32)
    state=PreparedState.__new__(PreparedState)
    state.nodes=nodes;state.domain=ImmutableNodeDomain(tuple(map(int,nodes)),{int(v):i for i,v in enumerate(nodes)})
    state.m=m;state.capacity=max(1,2*m);state.values=np.empty(state.capacity,np.float64)
    state.values[:m]=weights
    state.outgoing=Direction(out_row,dst.copy(),ids)
    state.incoming=Direction(in_row,src[incoming_order],incoming_order)
    state.in_weight=weights[incoming_order].copy()
    state.degree=out_count.astype(np.float64)
    state.sums=np.bincount(src,weights=weights,minlength=n).astype(np.float64)
    state.graph=None;state.version=0;state.readers=0
    state.initialization_seconds=perf_counter()-started
    return state


def vector_materializer(ref,nodes,edges,previous,changes,version):
    if previous is None or previous.ref!=ref or any(c.final_weight is None for c in changes):
        return cold_state(nodes,edges)
    return v1.vector_materializer(ref,nodes,edges,previous,changes,version)


class VectorJournalStore(v1.VectorJournalStore):
    def __init__(self,journal_budget_bytes,materializer=vector_materializer):
        super().__init__(journal_budget_bytes,materializer)

    @staticmethod
    def _input(nodes,edges,weights):
        nodes,allowed,base=ArrayJournalStore._input(nodes,edges,weights)
        return nodes,allowed,indexed(base)

    @classmethod
    def from_store(cls,old):
        result=super().from_store(old)
        result.materializer=vector_materializer
        for record in result._records.values():record.base_edges=indexed(record.base_edges)
        return result

    def _logical(self,record,journal=None):
        if journal is None:return record.base_edges
        started=perf_counter();result=fold(record.base_edges,journal.data)
        self._counters['authority_fold_seconds']+=perf_counter()-started
        self._counters['authority_fold_calls']+=1
        self._counters['authority_fold_base_edges']+=len(record.base_edges)
        self._counters['authority_fold_delta_edges']+=len(journal.data)
        return result

    def commit(self,ref,mutations):
        with self._lock:
            record=self._record(ref);mutations=list(mutations)
            # Input-domain validation stays equivalent to scalar _integer.
            if not mutations or any(not isinstance(m,Mutation) or m.mode!='add' or
                isinstance(m.src,(bool,np.bool_)) or isinstance(m.dst,(bool,np.bool_)) or
                not isinstance(m.src,(int,np.integer)) or not isinstance(m.dst,(int,np.integer)) or
                not -(1<<63)<=int(m.src)<1<<63 or not -(1<<63)<=int(m.dst)<1<<63 for m in mutations):
                return super().commit(ref,mutations)
            return self._add_commit(record,mutations)

    def _add_commit(self,record,mutations):
        """Vector sequential accumulation, followed by frozen atomic publication.

        np.add.at starts from each edge's *current* value and visits operations
        in their declared order, preserving floating addition and duplicates.
        """
        started=perf_counter();ref=record.ref;previous=self._journals.get(ref)
        pairs=np.asarray([(m.src,m.dst) for m in mutations],np.int64)
        try:weights=np.asarray([float(1. if m.value is None else m.value) for m in mutations],np.float64)
        except (TypeError,ValueError,OverflowError) as error:raise ValueError('Finite nonnegative weight required') from error
        if not np.all(np.isfinite(weights)) or np.any(weights<0):raise ValueError('Finite nonnegative weight required')
        nodes=np.asarray(record.nodes,np.int64);mapped=np.searchsorted(nodes,pairs)
        if np.any(mapped>=len(nodes)) or np.any(nodes[mapped]!=pairs):raise ValueError('Endpoint outside generation node domain')
        packed=packed_eligible(pairs)
        keys=encode_keys(pairs,packed)
        unique,first,inverse=np.unique(keys,return_index=True,return_inverse=True)
        requested=pairs[first]
        base=indexed(record.base_edges);positions,base_present=base.positions(requested)
        base_values=np.zeros(len(unique),np.float64)
        base_values[base_present]=base.values_array[positions[base_present]]
        current=base_values.copy();current_present=base_present.copy()
        untouched=np.empty(0,CHANGE_DTYPE)
        if previous is not None:
            prior_pairs=np.column_stack((previous.data['src'],previous.data['dst']))
            common_packed=packed and packed_eligible(prior_pairs)
            prior_keys=encode_keys(prior_pairs,common_packed);request_keys=encode_keys(requested,common_packed)
            pos=np.searchsorted(prior_keys,request_keys);found=pos<len(prior_keys)
            if len(prior_keys):found[found]&=prior_keys[pos[found]]==request_keys[found]
            current[found]=previous.data['final'][pos[found]]
            current_present[found]=(previous.data['flags'][pos[found]]&2)!=0
            current[~current_present]=0.
            touched=np.zeros(len(previous.data),bool);touched[pos[found]]=True
            untouched=previous.data[~touched]
        final=current.copy()
        with np.errstate(over='ignore'):np.add.at(final,inverse,weights)
        if not np.all(np.isfinite(final)):raise ValueError('Finite nonnegative weight required')
        changed=int(np.sum(~current_present | (final!=current)))
        if not changed:return Receipt(ref,record.version,0,(),retained_footprint(self._journals)['total_bytes'])
        keep=~base_present | (final!=base_values)
        data=np.empty(int(keep.sum()),CHANGE_DTYPE)
        data['src']=requested[keep,0];data['dst']=requested[keep,1]
        data['base']=base_values[keep];data['final']=final[keep]
        data['flags']=2+base_present[keep].astype(np.uint8)
        if len(untouched):
            data=np.concatenate((untouched,data));order=np.lexsort((data['dst'],data['src']));data=data[order]
        data.flags.writeable=False
        version=record.version+1;sequence=self._sequence+1;candidate=dict(self._journals)
        if len(data):candidate[ref]=_Journal(ref,record.base_version,version,
            previous.first_sequence if previous is not None else sequence,data)
        else:candidate.pop(ref,None)
        candidate=dict(candidate) if candidate else {}
        footprint=retained_footprint(candidate);candidate_peak=footprint['total_bytes']
        self._candidate_peak=max(self._candidate_peak,candidate_peak);plans=[]
        while footprint['total_bytes']>self.journal_budget_bytes:
            victim=min(candidate,key=lambda key:(candidate[key].first_sequence,key.graph_id,key.generation))
            item=self._record(victim);journal=candidate[victim];edges=self._logical(item,journal)
            plan=self._build(victim,item.nodes,edges,item.snapshot,_changes(journal),journal.version)
            plans.append((item,edges,plan));candidate.pop(victim);candidate=dict(candidate) if candidate else {}
            footprint=retained_footprint(candidate)
        record.version=version
        for item,edges,plan in plans:
            item.base_edges=edges;item.base_version=plan.new_snapshot.version;item.snapshot=plan.new_snapshot
        if ref not in candidate and all(plan.ref!=ref for _,_,plan in plans):
            record.base_version=version
            if record.snapshot is not None:record.snapshot=Snapshot(ref,version,record.snapshot.host_state)
        self._journals=candidate;self._sequence=sequence
        self._retained_peak=max(self._retained_peak,footprint['total_bytes'])
        self._candidate_peak=max(self._candidate_peak,candidate_peak)
        self._counters['commits']+=1;self._counters['operations']+=len(mutations)
        self._counters['pressure_flushes']+=len(plans)
        self._counters['commit_seconds']+=perf_counter()-started
        self._counters['typed_add_batches']+=1;self._counters['typed_add_operations']+=len(mutations)
        return Receipt(ref,version,changed,tuple(plan for _,_,plan in plans),footprint['total_bytes'])


def install_vector_authority(session):
    session.journal=VectorJournalStore.from_store(session.journal)
    return session
