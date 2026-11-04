"""Budgeted host indexes and shared immutable update certificates (R8).

Logical transactions retain R7 semantics. Index entries own only packed keys;
weak layout references cannot extend a graph snapshot's lifetime.
"""
from collections import OrderedDict, Counter
from dataclasses import dataclass
import sys
import time
import weakref
import numpy as np
from easygraph.gpu._arena.round7_recovery import host_authority_v2 as r7
from easygraph.gpu._arena.round7_recovery import host_authority as v1
from easygraph.gpu._arena.round2_topology.state import Direction, endpoints, locate as scalar_locate
from easygraph.gpu._arena.round2_topology.gpu_adapter import PreparedState
from easygraph.gpu._arena.round4_switch.journal import retained_footprint


def payload(arrays):
    seen=set();total=0
    for value in arrays:
        if not isinstance(value,np.ndarray):continue
        while isinstance(value.base,np.ndarray):value=value.base
        if id(value) not in seen:seen.add(id(value));total+=value.nbytes
    return total


class IndexCache:
    def __init__(self,budget):
        if budget<0:raise ValueError('Nonnegative index budget required')
        self.budget=int(budget);self.entries=OrderedDict();self.count=Counter()
        self.live=0;self.peak=0;self.temporary_peak=0;self.working_peak=0;self.on_change=None

    def _remove(self,key):
        entry=self.entries.pop(key,None)
        if entry is not None:
            self.live-=entry[1].nbytes
            if self.on_change is not None:self.on_change()

    def _keys(self,layout):
        key=id(layout);entry=self.entries.get(key)
        if entry is not None and entry[0]() is layout:
            self.entries.move_to_end(key);self.count['hits']+=1;return entry[1]
        if entry is not None:self._remove(key)
        started=time.perf_counter();m=len(layout.col);n=len(layout.row)-1
        row_ids=np.arange(n,dtype=np.uint64);counts=np.diff(layout.row)
        rows=np.repeat(row_ids,counts)
        temporary=payload([row_ids,counts,rows])
        self.temporary_peak=max(self.temporary_peak,temporary)
        self.working_peak=max(self.working_peak,self.live+temporary)
        del row_ids,counts
        keys=rows;keys<<=np.uint64(32)
        cols=layout.col.astype(np.uint64);keys|=cols
        self.temporary_peak=max(self.temporary_peak,keys.nbytes+cols.nbytes)
        self.working_peak=max(self.working_peak,self.live+keys.nbytes+cols.nbytes)
        self.count['builds']+=1;self.count['built_bytes']+=keys.nbytes
        self.count['build_nanoseconds']+=int((time.perf_counter()-started)*1e9)
        keys.flags.writeable=False
        if keys.nbytes<=self.budget:
            while self.entries and self.live+keys.nbytes>self.budget:
                victim=next(iter(self.entries));self._remove(victim);self.count['evictions']+=1
            owner=weakref.ref(self)
            def expire(ref,k=key):
                cache=owner()
                if cache is not None:
                    current=cache.entries.get(k)
                    if current is not None and current[0] is ref:cache._remove(k)
            self.entries[key]=(weakref.ref(layout,expire),keys)
            self.live+=keys.nbytes;self.peak=max(self.peak,self.live)
            if self.on_change is not None:self.on_change()
        else:self.count['uncached_builds']+=1
        return keys

    def locate(self,layout,src,dst):
        if len(src)<64:
            self.count['small_scalar_queries']+=len(src)
            return scalar_locate(layout,src,dst)
        keys=self._keys(layout)
        wanted=(src.astype(np.uint64)<<np.uint64(32))|dst.astype(np.uint64)
        positions=np.searchsorted(keys,wanted);present=positions<len(keys)
        if len(keys):present[present]&=keys[positions[present]]==wanted[present]
        self.count['batch_queries']+=1;self.count['queried_edges']+=len(src)
        return positions.astype(np.int32),present

    def clear(self):
        self.entries.clear();self.live=0
        if self.on_change is not None:self.on_change()

    def stats(self):
        footprint=retained_footprint(self.entries)
        if footprint['payload_bytes']!=self.live:raise AssertionError('Index byte ledger mismatch')
        return dict(self.count,budget_bytes=self.budget,retained_payload_bytes=self.live,
                    retained_payload_peak_bytes=self.peak,retained_metadata_bytes=footprint['metadata_bytes'],
                    explicit_index_build_temporary_peak_bytes=self.temporary_peak,
                    explicit_index_working_peak_bytes=self.working_peak,
                    entries=len(self.entries),scope='Budget limits retained packed-index payload; reachable metadata and explicit build temporaries separately measured; NumPy internal workspace/RSS separate')



@dataclass
class Certificate:
    ref:object
    source_version:int
    target_version:int
    old:object
    new:object
    changes:tuple
    src:np.ndarray
    dst:np.ndarray
    values:np.ndarray
    inserted:np.ndarray
    out_description:dict
    in_description:dict
    incoming_positions:np.ndarray

    def arrays(self):
        return [self.src,self.dst,self.values,self.inserted,self.incoming_positions,*self.out_description.values(),*self.in_description.values()]

    def matches(self,old,new,changes):
        return self.old() is old and self.new() is new and (self.changes is changes or self.changes==changes)


def description(src,dst,positions,new):
    order=np.lexsort((dst[new],src[new]));selected=np.flatnonzero(new)[order]
    return dict(before=np.ascontiguousarray(positions[selected],np.int32),
                values=np.ascontiguousarray(dst[selected],np.int32),
                rows=np.ascontiguousarray(src[selected],np.int32)),selected


def merge_direction(old,desc,new_ids,n):
    if not len(desc['before']):return old
    before=desc['before'];m=len(old.col);k=len(before)
    counts=np.bincount(before,minlength=m+1)
    existing=np.arange(m)+np.cumsum(counts[:-1]);inserted=before+np.arange(k)
    col=np.empty(m+k,np.int32);ids=np.empty(m+k,np.int32)
    col[existing]=old.col;ids[existing]=old.ids
    col[inserted]=desc['values'];ids[inserted]=new_ids
    increments=np.bincount(desc['rows'],minlength=n)
    row=old.row+np.r_[0,np.cumsum(increments)].astype(np.int32)
    return Direction(row,col,ids)


class Store(r7.VectorJournalStore):
    def __init__(self,journal_budget_bytes,host_index_budget_bytes=0,producer='shared',materializer=None):
        if callable(host_index_budget_bytes):materializer=host_index_budget_bytes;host_index_budget_bytes=0
        if producer not in ('cached','shared'):raise ValueError(producer)
        self.producer=producer;self.index=IndexCache(host_index_budget_bytes)
        self.certificates={};self.certificate_peak=0;self.certificate_count=Counter()
        self.extra_retained_peak=0;self.index.on_change=self._retained_change
        super().__init__(journal_budget_bytes,materializer=self._materialize if materializer is None else materializer)

    def _logical(self,record,journal=None):
        if journal is None or self.producer!='shared':return super()._logical(record,journal)
        data=journal.data
        # Inspect flags before any endpoint lookup: deleting batches use the
        # exact R7 cold path once, including its complete preparation timer.
        if np.any((data['flags']&2)==0):return super()._logical(record,journal)
        started=time.perf_counter();base=r7.indexed(record.base_edges)
        pairs,original=base.to_arrays();changes=np.column_stack((data['src'],data['dst']))
        if base.packed and not r7.packed_eligible(changes):
            # Packed lookup correctly reports absent out-of-domain keys but
            # its wrapped numeric rank is not a signed canonical insertion rank.
            positions,present=v1._positions(pairs,changes)
            self.certificate_count['signed_rank_domain_promotions']+=1
        else:positions,present=base.positions(changes)
        values=original.copy();values[positions[present]]=data['final'][present]
        insert=~present
        if np.any(insert):
            before=positions[insert];counts=np.bincount(before,minlength=len(pairs)+1)
            targets=np.arange(len(pairs))+np.cumsum(counts[:-1]);newpos=before+np.arange(int(insert.sum()))
            merged=np.empty((len(pairs)+int(insert.sum()),2),np.int64)
            merged_values=np.empty(len(merged),np.float64)
            merged[targets]=pairs;merged_values[targets]=values
            merged[newpos]=changes[insert];merged_values[newpos]=data['final'][insert]
            pairs,values=merged,merged_values
        result=r7.IndexedEdgeMap(pairs,values)
        result._r8_hint=(weakref.ref(base),positions.astype(np.int32),present.copy(),
                         weakref.ref(record.snapshot) if record.snapshot is not None else lambda:None)
        self._counters['authority_fold_seconds']+=time.perf_counter()-started
        self._counters['authority_fold_calls']+=1;self._counters['authority_fold_base_edges']+=len(base)
        self._counters['authority_fold_delta_edges']+=len(data)
        return result

    def _build(self,ref,nodes,edges,previous,changes,version):
        try:return super()._build(ref,nodes,edges,previous,changes,version)
        finally:
            if hasattr(edges,'_r8_hint'):del edges._r8_hint

    def _materialize(self,ref,nodes,edges,previous,changes,version):
        self._remove_certificate(ref)
        if previous is None or previous.ref!=ref or any(c.final_weight is None for c in changes):
            state=r7.cold_state(nodes,edges);state._r8_authority=weakref.ref(edges)
            return state
        old=previous.host_state
        if not changes:return old
        pairs=np.asarray([(c.src,c.dst) for c in changes],np.int64).reshape(-1,2)
        mapped=endpoints(pairs,old.nodes);src,dst=mapped[:,0],mapped[:,1]
        values=np.asarray([c.final_weight for c in changes],np.float64)
        if all(c.base_weight is not None for c in changes):
            state=v1.vector_materializer(ref,nodes,edges,previous,changes,version)
            state._r8_authority=weakref.ref(edges)
            if self.producer=='shared':
                empty={name:np.empty(0,np.int32) for name in ('before','values','rows')}
                self._publish_certificate(ref,previous,version,old,state,changes,src.copy(),dst.copy(),values,
                                          np.zeros(len(src),bool),empty,dict(empty),None)
            return state
        hint=getattr(edges,'_r8_hint',None)
        valid=(self.producer=='shared' and hint is not None and hint[3]() is previous and
               getattr(old,'_r8_authority',lambda:None)() is hint[0]())
        if valid:
            positions,present=hint[1],hint[2];self.certificate_count['fold_rank_reuses']+=1
        else:positions,present=self.index.locate(old.outgoing,src,dst)
        inserted=~present;n=len(nodes);count=int(inserted.sum());new_m=old.m+count
        if new_m>=2**31:raise ValueError('Edge positions must fit signed int32')
        ids=np.empty(len(src),np.int32);ids[present]=old.outgoing.ids[positions[present]]
        ids[inserted]=old.m+np.arange(count,dtype=np.int32)
        out_desc,out_selected=description(src,dst,positions,inserted)
        if self.producer=='shared':
            in_positions,in_present=self.index.locate(old.incoming,dst,src)
            if not np.array_equal(present,in_present):raise AssertionError('Direction membership mismatch')
            in_desc,in_selected=description(dst,src,in_positions,inserted)
            incoming_positions=in_positions+np.searchsorted(in_desc['before'],in_positions,side='right')
            incoming_positions[in_selected]=in_desc['before']+np.arange(count)
            incoming_positions=np.asarray(incoming_positions,np.int32)
        else:
            new_indices=np.flatnonzero(inserted)
            before,found=self.index.locate(old.incoming,dst[inserted],src[inserted])
            if found.any():raise AssertionError('New edge already present')
            in_desc,selected=description(dst[inserted],src[inserted],before,np.ones(count,bool))
            in_selected=new_indices[selected];incoming_positions=None
        if not count:
            # Keep sparse changed-row sum repair and exact R7 immutable semantics.
            state=v1.vector_materializer(ref,nodes,edges,previous,changes,version)
        else:
            capacity=old.capacity if new_m<=old.capacity else max(new_m,2*old.capacity)
            canonical=np.empty(capacity,np.float64);canonical[:old.m]=old.values[:old.m];canonical[ids]=values
            outgoing=merge_direction(old.outgoing,out_desc,ids[out_selected],n)
            incoming=merge_direction(old.incoming,in_desc,ids[in_selected],n)
            state=PreparedState.__new__(PreparedState);state.__dict__=old.__dict__.copy()
            state.values=canonical;state.m=new_m;state.capacity=capacity
            state.outgoing=outgoing;state.incoming=incoming;state.in_weight=canonical[incoming.ids]
            state.degree=np.diff(outgoing.row).astype(np.float64)
            sources=np.repeat(np.arange(n,dtype=np.int32),np.diff(outgoing.row))
            state.sums=np.bincount(sources,weights=canonical[outgoing.ids],minlength=n).astype(np.float64)
            state.graph=None;state.readers=0;state.version=old.version+1
        state._r8_authority=weakref.ref(edges)
        if self.producer=='shared':
            self._publish_certificate(ref,previous,version,old,state,changes,src.copy(),dst.copy(),values,
                                      inserted,out_desc,in_desc,incoming_positions)
        return state

    def _retained_change(self):
        certificate_live=payload([a for c in self.certificates.values() for a in c.arrays()])
        self.certificate_peak=max(self.certificate_peak,certificate_live)
        self.extra_retained_peak=max(self.extra_retained_peak,self.index.live+certificate_live)

    def _remove_certificate(self,ref):
        self.certificates.pop(ref,None);self._retained_change()

    def _publish_certificate(self,ref,previous,version,old,state,changes,src,dst,values,inserted,out_desc,in_desc,incoming_positions):
        owner=weakref.ref(self)
        def expire(token):
            store=owner()
            if store is not None:
                current=store.certificates.get(ref)
                if current is not None and token in (current.old,current.new):store._remove_certificate(ref)
        cert=Certificate(ref,previous.version,version,weakref.ref(old,expire),weakref.ref(state,expire),changes,
                         src,dst,values,inserted,out_desc,in_desc,incoming_positions)
        for a in cert.arrays():
            if isinstance(a,np.ndarray):a.flags.writeable=False
        self.certificates[ref]=cert;self.certificate_count['builds']+=1;self._retained_change()

    def certificate(self,old,new,changes):
        for cert in self.certificates.values():
            if cert.matches(old,new,changes):
                self.certificate_count['packet_reuses']+=1;return cert
        self.certificate_count['packet_misses']+=1;return None

    def certificate_bytes(self):
        for ref,c in list(self.certificates.items()):
            if c.old() is None or c.new() is None:self._remove_certificate(ref)
        return payload([a for c in self.certificates.values() for a in c.arrays()])

    @classmethod
    def from_store(cls,old):
        result=v1.VectorJournalStore.from_store.__func__(cls,old)
        for record in result._records.values():record.base_edges=r7.indexed(record.base_edges)
        result.materializer=result._materialize
        return result

    def discard(self,ref):
        result=super().discard(ref);self._remove_certificate(ref)
        return result

    def replace(self,ref,*args,**kwargs):
        result=super().replace(ref,*args,**kwargs);self._remove_certificate(ref)
        return result

    def stats(self):
        return dict(super().stats(),host_producer=self.producer,host_index=self.index.stats(),
                    extra_host_retained_payload_bytes=self.index.live+self.certificate_bytes(),
                    extra_host_retained_payload_peak_bytes=self.extra_retained_peak,
                    host_certificate=dict(self.certificate_count,retained_payload_bytes=self.certificate_bytes(),
                        retained_payload_peak_bytes=self.certificate_peak,
                        retained_metadata_bytes=retained_footprint(self.certificates)['metadata_bytes'],
                        scope='Identity-deduplicated certificate buffers and complete reachable metadata including EdgeChange tuples; weak snapshot/layout refs are not traversed'))


VectorJournalStore=Store
IndexedEdgeMap=r7.IndexedEdgeMap
