"""Vectorized host authority and stable-ID CSR materialization.

All graph policies may use this same implementation. Journal publication,
generation checks, operation ordering, cancellation and failure atomicity remain
the frozen transactional implementation. A flush folds the net delta into one
current immutable typed authority instead of carrying a growing overlay chain.
"""
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from time import perf_counter
import numpy as np

from easygraph.gpu._arena.round4_switch.journal import Mutation, _changes
from easygraph.gpu._arena.round5_input.array_journal_v2 import ArrayJournalStore, ArrayEdgeMap, KEY_DTYPE


def _keys(pairs):
    return np.ascontiguousarray(pairs,dtype=np.int64).view(KEY_DTYPE).reshape(-1)


def _positions(base_pairs, pairs):
    """One typed binary-search batch; no per-edge NumPy scalar dispatch."""
    keys=_keys(base_pairs);query=_keys(pairs)
    positions=np.searchsorted(keys,query)
    present=positions<len(keys)
    if len(keys):present[present]&=keys[positions[present]]==query[present]
    return positions,present


def _publish_arrays(pairs,values):
    """Transfer freshly owned buffers (or immutable unchanged pairs) to a map."""
    result=ArrayEdgeMap.__new__(ArrayEdgeMap)
    result.pairs=pairs;result.values_array=values
    result.pairs.flags.writeable=False;result.values_array.flags.writeable=False
    result._keys=result.pairs.view(KEY_DTYPE).reshape(-1)
    return result


def fold_authority(base, changes):
    """Immutable base + canonical net delta -> immutable current typed base."""
    if not len(changes):return base
    pairs,original=base.to_arrays()
    update=np.column_stack((changes['src'],changes['dst']))
    positions,present=_positions(pairs,update)
    final=(changes['flags']&2)!=0
    keep=np.ones(len(pairs),dtype=bool);values=original.copy()
    remove=present & ~final
    keep[positions[remove]]=False
    modify=present & final
    values[positions[modify]]=changes['final'][modify]
    new=~present & final
    if np.any(remove):pairs=pairs[keep];values=values[keep]
    if np.any(new):
        additions=update[new];new_values=changes['final'][new]
        # Both inputs are already sorted. One insertion rank per new edge,
        # followed by a prefix sum, replaces a full endpoint re-sort.
        insertion=np.searchsorted(_keys(pairs),_keys(additions))
        counts=np.bincount(insertion,minlength=len(pairs)+1)
        existing=np.arange(len(pairs))+np.cumsum(counts[:-1])
        targets=insertion+np.arange(len(additions))
        merged=np.empty((len(pairs)+len(additions),2),np.int64)
        merged_values=np.empty(len(merged),np.float64)
        merged[existing]=pairs;merged_values[existing]=values
        merged[targets]=additions;merged_values[targets]=new_values
        pairs,values=merged,merged_values
    return _publish_arrays(pairs,values)


class _BatchBase(Mapping):
    """Temporary lookup accelerator; authoritative arrays are unchanged."""
    def __init__(self,base,mutations):
        self.base=base
        keys=[]
        for m in mutations:
            if isinstance(m,Mutation) and isinstance(m.src,(int,np.integer)) and isinstance(m.dst,(int,np.integer)):
                u,v=int(m.src),int(m.dst)
                if -(1<<63)<=u<(1<<63) and -(1<<63)<=v<(1<<63):keys.append((u,v))
        keys=tuple(dict.fromkeys(keys))
        self.lookup={}
        if keys:
            pairs,values=base.to_arrays()
            positions,present=_positions(pairs,np.asarray(keys,np.int64))
            self.lookup={key:float(values[p]) if found else None
                         for key,p,found in zip(keys,positions,present)}
    def __len__(self):return len(self.base)
    def __iter__(self):return iter(self.base)
    def __getitem__(self,key):
        if key in self.lookup:
            value=self.lookup[key]
            if value is None:raise KeyError(key)
            return value
        return self.base[key]
    def to_arrays(self):return self.base.to_arrays()
    def to_dict(self):return self.base.to_dict()


def _packed(src,dst):
    return (src.astype(np.uint64)<<np.uint64(32))|dst.astype(np.uint64)


def _insert_direction(old,src,dst,ids,n):
    """Sorted vector merge preserving every existing canonical edge ID."""
    from easygraph.gpu._arena.round2_topology.state import Direction
    if not len(src):return old
    old_src=np.repeat(np.arange(n,dtype=np.int32),np.diff(old.row))
    old_keys=_packed(old_src,old.col)
    order=np.lexsort((dst,src));src=src[order];dst=dst[order];ids=ids[order]
    new_keys=_packed(src,dst)
    insertion=np.searchsorted(old_keys,new_keys)
    counts=np.bincount(insertion,minlength=len(old.col)+1)
    existing=np.arange(len(old.col))+np.cumsum(counts[:-1])
    insertion=insertion+np.arange(len(new_keys))
    col=np.empty(len(old.col)+len(src),np.int32);edge_ids=np.empty_like(col)
    col[existing]=old.col;edge_ids[existing]=old.ids
    col[insertion]=dst;edge_ids[insertion]=ids
    increments=np.bincount(src,minlength=n)
    row=old.row+np.r_[0,np.cumsum(increments)].astype(np.int32)
    return Direction(row,col,edge_ids)


def vector_materializer(ref,nodes,edges,previous,changes,version):
    """Current journal semantics, stable IDs and immutable published snapshots.

    Nondeleting updates share the node domain, preserve old edge IDs and append
    new IDs in canonical source/target order, exactly as frozen R2. A deletion
    rebuilds from current authoritative arrays, matching frozen R4 semantics.
    """
    from easygraph.gpu._arena.round2_topology.gpu_adapter import PreparedState
    from easygraph.gpu._arena.round2_topology.state import endpoints
    if previous is None or previous.ref!=ref or any(c.final_weight is None for c in changes):
        pairs,values=edges.to_arrays()
        state=PreparedState(np.asarray(nodes,np.int64),pairs,values)
        state.sums=np.asarray(state.sums,dtype=np.float64)
        return state
    old=previous.host_state
    if not changes:return old
    if all(c.base_weight is not None for c in changes):
        # A pure attribute update needs neither endpoint merge nor full sum
        # rebuild. Keep the established scoped source-row repair primitive.
        from easygraph.gpu._arena.round4_switch.journal import default_materializer
        return default_materializer(ref,nodes,edges,previous,changes,version)
    pairs=np.asarray([(c.src,c.dst) for c in changes],np.int64).reshape(-1,2)
    weight=np.asarray([c.final_weight for c in changes],np.float64)
    if not np.all(np.isfinite(weight)) or np.any(weight<0):raise ValueError('Finite nonnegative edge values required')
    mapped=endpoints(pairs,old.nodes);src,dst=mapped[:,0],mapped[:,1]
    keys=_packed(src,dst);order=np.argsort(keys,kind='stable')
    keys,src,dst,weight=keys[order],src[order],dst[order],weight[order]
    # Journal net deltas are unique. Keep prepare_delta(mode='set') semantics
    # when this callback is also used with a caller-supplied repeated list.
    last=np.r_[keys[1:]!=keys[:-1],True]
    keys,src,dst,weight=keys[last],src[last],dst[last],weight[last]
    n=len(old.nodes)
    old_src=np.repeat(np.arange(n,dtype=np.int32),np.diff(old.outgoing.row))
    old_keys=_packed(old_src,old.outgoing.col)
    locations=np.searchsorted(old_keys,keys);present=locations<len(old_keys)
    if len(old_keys):present[present]&=old_keys[locations[present]]==keys[present]
    count=int(np.sum(~present));new_m=old.m+count
    if new_m>=2**31:raise ValueError('Edge positions must fit signed int32')
    ids=np.empty(len(keys),np.int32)
    ids[present]=old.outgoing.ids[locations[present]]
    ids[~present]=old.m+np.arange(count,dtype=np.int32)
    capacity=old.capacity if new_m<=old.capacity else max(new_m,2*old.capacity)
    canonical=np.empty(capacity,np.float64);canonical[:old.m]=old.values[:old.m]
    canonical[ids]=weight
    outgoing=_insert_direction(old.outgoing,src[~present],dst[~present],ids[~present],n)
    incoming=_insert_direction(old.incoming,dst[~present],src[~present],ids[~present],n)
    state=PreparedState.__new__(PreparedState);state.__dict__=old.__dict__.copy()
    state.values=canonical;state.m=new_m;state.capacity=capacity
    state.outgoing=outgoing;state.incoming=incoming;state.in_weight=canonical[incoming.ids]
    state.degree=np.diff(outgoing.row).astype(np.float64)
    sources=np.repeat(np.arange(n,dtype=np.int32),np.diff(outgoing.row))
    state.sums=np.bincount(sources,weights=canonical[outgoing.ids],minlength=n).astype(np.float64)
    state.graph=None;state.readers=0;state.version=old.version+1
    return state


class VectorJournalStore(ArrayJournalStore):
    """Same transactional API with batch lookup and compact typed publication."""
    def __init__(self,journal_budget_bytes,materializer=vector_materializer):
        super().__init__(journal_budget_bytes,materializer)

    @classmethod
    def from_store(cls,old):
        """Explicit independent store migration; callers charge this operation.

        References, versions and immutable snapshots remain valid. The old store
        retains its own records, journals and locks and is not silently mutated.
        """
        with old._lock:
            result=cls(old.journal_budget_bytes)
            result._records={}
            for gid,record in old._records.items():
                base=record.base_edges
                if not isinstance(base,ArrayEdgeMap):
                    pairs,values=base.to_arrays();base=ArrayEdgeMap(pairs,values)
                result._records[gid]=replace(record,base_edges=base)
            result._generations=dict(old._generations);result._journals=dict(old._journals)
            result._sequence=old._sequence;result._counters=Counter(old._counters)
            result._retained_peak=old._retained_peak;result._candidate_peak=old._candidate_peak
            return result

    def _logical(self,record,journal=None):
        if journal is None:return record.base_edges
        started=perf_counter()
        result=fold_authority(record.base_edges,journal.data)
        self._counters['authority_fold_seconds']+=perf_counter()-started
        self._counters['authority_fold_calls']+=1
        self._counters['authority_fold_base_edges']+=len(record.base_edges)
        self._counters['authority_fold_delta_edges']+=len(journal.data)
        return result

    def commit(self,ref,mutations):
        with self._lock:
            record=self._record(ref)
            mutations=list(mutations)
            started=perf_counter();base=record.base_edges
            accelerated=_BatchBase(base,mutations)
            self._counters['batch_lookup_seconds']+=perf_counter()-started
            self._counters['batch_lookup_calls']+=1
            self._counters['batch_lookup_keys']+=len(accelerated.lookup)
            record.base_edges=accelerated
            try:return super().commit(ref,mutations)
            finally:
                if record.base_edges is accelerated:record.base_edges=base

    def stats(self):
        return dict(super().stats(),authority_format='current immutable typed base plus pending coalesced journal',
                    authority_transition='vector batch lookup, one delta fold per materialization, stable-ID direction merges')


def install_vector_authority(session):
    """Charge this inside session initialization for any recovery policy."""
    session.journal=VectorJournalStore.from_store(session.journal)
    return session
