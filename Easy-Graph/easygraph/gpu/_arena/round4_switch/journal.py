"""Transactional, per-generation mutation journals shared by all controls.

Committed logical state is immutable base edges plus a compact net journal.
CSR materialization is a separate operation. No discard operation deletes a
committed logical mutation. The default materializer uses R2 shared_delta.
"""
from dataclasses import dataclass,fields,is_dataclass
from collections import Counter
from types import MappingProxyType
import copy
import math
import sys
import threading
import time
import numpy as np


CHANGE_DTYPE=np.dtype([('src','<i8'),('dst','<i8'),('base','<f8'),('final','<f8'),('flags','u1')])


@dataclass(frozen=True)
class GraphRef:
    graph_id:str
    generation:int


@dataclass(frozen=True)
class Mutation:
    mode:str
    src:int
    dst:int
    value:float|None=None


@dataclass(frozen=True)
class EdgeChange:
    src:int
    dst:int
    base_weight:float|None
    final_weight:float|None


@dataclass(frozen=True)
class Snapshot:
    ref:GraphRef
    version:int
    host_state:object


@dataclass(frozen=True)
class FlushResult:
    ref:GraphRef
    old_snapshot:Snapshot|None
    new_snapshot:Snapshot
    changes:tuple[EdgeChange,...]
    reused:bool=False


@dataclass(frozen=True)
class Receipt:
    ref:GraphRef
    version:int
    changed_edges:int
    auto_flushed:tuple[FlushResult,...]
    journal_bytes:int


@dataclass(frozen=True)
class DiscardResult:
    ref:GraphRef
    version:int
    discarded_snapshot:Snapshot|None


@dataclass(frozen=True)
class _Journal:
    ref:GraphRef
    base_version:int
    version:int
    first_sequence:int
    data:np.ndarray


@dataclass
class _Record:
    ref:GraphRef
    nodes:tuple[int,...]
    node_set:frozenset[int]
    base_edges:dict
    version:int
    base_version:int
    snapshot:Snapshot|None


class StaleGraphError(KeyError):pass


def retained_footprint(journals):
    """Identity-deduplicated retained payload and metadata reachable from logs.

Array-owned bytes are separated from headers. Shared refs/keys are counted
once; allocator fragmentation and transient decode/materialization are outside
this retention metric, rather than silently classified as zero overhead.
"""
    seen=set();payload=0;metadata=0
    def visit(value):
        nonlocal payload,metadata
        if id(value) in seen:return
        seen.add(id(value));size=sys.getsizeof(value)
        if isinstance(value,np.ndarray):
            own=value.nbytes if value.flags.owndata else 0;payload+=own;metadata+=size-own
            if value.base is not None:visit(value.base)
        else:
            metadata+=size
            if isinstance(value,dict):
                for key,item in value.items():visit(key);visit(item)
            elif isinstance(value,(tuple,list,set,frozenset)):
                for item in value:visit(item)
            elif is_dataclass(value):
                visit(value.__dict__)
    visit(journals)
    return dict(payload_bytes=payload,metadata_bytes=metadata,total_bytes=payload+metadata,objects=len(seen))


def _changes(journal):
    if journal is None:return ()
    return tuple(EdgeChange(int(row['src']),int(row['dst']),float(row['base']) if row['flags']&1 else None,
                float(row['final']) if row['flags']&2 else None) for row in journal.data)


def _encode(ref,base_version,version,sequence,changes):
    array=np.empty(len(changes),dtype=CHANGE_DTYPE)
    for i,c in enumerate(sorted(changes,key=lambda c:(c.src,c.dst))):
        array[i]=(c.src,c.dst,c.base_weight or 0.,c.final_weight or 0.,
                  int(c.base_weight is not None)|2*int(c.final_weight is not None))
    array.flags.writeable=False
    return _Journal(ref,base_version,version,sequence,array)


def _integer(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError('Node labels must be integers')
    value=int(value)
    if not -(1<<63)<=value<(1<<63):raise ValueError('Node label exceeds int64')
    return value


def _weight(value):
    try:value=float(value)
    except (TypeError,ValueError,OverflowError) as error:raise ValueError('Finite nonnegative weight required') from error
    if not math.isfinite(value) or value<0:raise ValueError('Finite nonnegative weight required')
    return value


def default_materializer(ref,nodes,edges,previous_snapshot,changes,version):
    """Pure callback: old snapshots remain intact until a whole commit succeeds.

All methods use this same R2 primitive. Nondeleting changes preserve canonical
edge IDs by cloning only publication-mutated arrays. Deletes and cold states
rebuild from the authoritative final edge mapping.
"""
    from easygraph.gpu._arena.round2_topology.gpu_adapter import PreparedState
    if previous_snapshot is not None and previous_snapshot.ref==ref:
        if not changes:return previous_snapshot.host_state
        if all(c.final_weight is not None for c in changes):
            old=previous_snapshot.host_state
            state=PreparedState.__new__(PreparedState);state.__dict__=old.__dict__.copy()
            for name in ('values','sums','degree'):setattr(state,name,np.array(getattr(old,name),dtype=np.float64,copy=True))
            state.readers=0
            pairs=np.asarray([(c.src,c.dst) for c in changes],np.int64).reshape(-1,2)
            values=np.asarray([c.final_weight for c in changes],np.float64)
            delta=state.prepare_delta(pairs,values,'set')
            state.apply_prepared(pairs,delta,'shared_delta')
            return state
    pairs=sorted(edges)
    state=PreparedState(np.asarray(nodes,np.int64),np.asarray(pairs,np.int64).reshape(-1,2),
                        np.asarray([edges[pair] for pair in pairs],np.float64))
    # numpy.bincount([], weights=[]) yields integer zeros: normalize this cold
    # empty-graph case before later fractional shared_delta publication.
    state.sums=np.asarray(state.sums,dtype=np.float64)
    return state


class JournalStore:
    def __init__(self,journal_budget_bytes,materializer=default_materializer):
        if journal_budget_bytes<sys.getsizeof({}):raise ValueError('Journal budget must hold an empty ownership table')
        self.journal_budget_bytes=int(journal_budget_bytes);self.materializer=materializer
        self._records={};self._generations={};self._journals={};self._sequence=0
        self._lock=threading.RLock();self._counters=Counter();self._retained_peak=sys.getsizeof({})
        self._candidate_peak=sys.getsizeof({})

    def _record(self,ref):
        record=self._records.get(ref.graph_id)
        if record is None or record.ref!=ref:raise StaleGraphError('Graph generation has been retired: '+repr(ref))
        return record

    @staticmethod
    def _input(nodes,edges,weights):
        nodes=tuple(_integer(n) for n in nodes)
        if tuple(sorted(set(nodes)))!=nodes:raise ValueError('Sorted unique node labels required')
        allowed=frozenset(nodes);pairs=list(edges)
        if weights is None:weights=[1.]*len(pairs)
        else:weights=list(weights)
        if len(pairs)!=len(weights):raise ValueError('Initial edge/weight length mismatch')
        result={}
        for pair,value in zip(pairs,weights):
            if len(pair)!=2:raise ValueError('Endpoint pair required')
            src,dst=map(_integer,pair)
            if src not in allowed or dst not in allowed:raise ValueError('Endpoint outside generation node domain')
            result[(src,dst)]=_weight(result.get((src,dst),0.)+_weight(value))
        return nodes,allowed,result

    def register(self,graph_id,nodes,edges,weights=None,*,materialize=True):
        with self._lock:
            if not isinstance(graph_id,str) or not graph_id:raise ValueError('Nonempty string graph_id required')
            if graph_id in self._records:raise ValueError('Graph exists; use replace with its current reference')
            return self._register(graph_id,nodes,edges,weights,materialize)

    def _register(self,graph_id,nodes,edges,weights,materialize):
        nodes,allowed,base=self._input(nodes,edges,weights)
        ref=GraphRef(graph_id,self._generations.get(graph_id,0)+1)
        snapshot=None
        if materialize:snapshot=self._build(ref,nodes,base,None,(),0).new_snapshot
        record=_Record(ref,nodes,allowed,base,0,0,snapshot)
        self._records[graph_id]=record;self._generations[graph_id]=ref.generation
        return ref

    def replace(self,ref,nodes,edges,weights=None,*,materialize=True):
        with self._lock:
            self._record(ref)
            # Validate and materialize the new generation before publishing it.
            new_ref=self._register(ref.graph_id,nodes,edges,weights,materialize)
            self._journals={key:value for key,value in self._journals.items() if key!=ref}
            self._counters['replacements']+=1
            return new_ref

    def _logical(self,record,journal=None):
        edges=dict(record.base_edges)
        for c in _changes(journal):
            if c.final_weight is None:edges.pop((c.src,c.dst),None)
            else:edges[(c.src,c.dst)]=c.final_weight
        return edges

    def edges(self,ref):
        with self._lock:
            record=self._record(ref)
            return MappingProxyType(self._logical(record,self._journals.get(ref)))

    def peek_pending(self,ref):
        with self._lock:self._record(ref);return _changes(self._journals.get(ref))

    def snapshot(self,ref):
        with self._lock:return self._record(ref).snapshot

    def version(self,ref):
        with self._lock:return self._record(ref).version

    def _build(self,ref,nodes,edges,previous,changes,version):
        started=time.perf_counter();self._counters['materializer_attempts']+=1
        try:state=self.materializer(ref,nodes,MappingProxyType(edges),previous,changes,version)
        finally:self._counters['materializer_seconds']+=time.perf_counter()-started
        return FlushResult(ref,previous,Snapshot(ref,version,state),changes,
                           previous is not None and state is previous.host_state)

    def commit(self,ref,mutations):
        with self._lock:
            started=time.perf_counter();record=self._record(ref);previous=self._journals.get(ref)
            net={(c.src,c.dst):c for c in _changes(previous)};before={};operations=0
            for mutation in mutations:
                if not isinstance(mutation,Mutation):raise ValueError('Mutation objects required')
                mode=mutation.mode;src=_integer(mutation.src);dst=_integer(mutation.dst);key=(src,dst)
                if src not in record.node_set or dst not in record.node_set:raise ValueError('Endpoint outside generation node domain')
                current=net[key].final_weight if key in net else record.base_edges.get(key)
                before.setdefault(key,current)
                if mode=='delete':final=None
                elif mode in ('add','set','insert'):
                    if mode=='set' and mutation.value is None:raise ValueError('set requires an explicit weight')
                    value=_weight(1. if mutation.value is None else mutation.value)
                    if mode=='insert' and current is not None:raise ValueError('insert requires an absent edge')
                    final=_weight((current or 0.)+value) if mode=='add' else value
                else:raise ValueError('Unknown mutation mode')
                base=record.base_edges.get(key)
                if base==final:net.pop(key,None)
                else:net[key]=EdgeChange(src,dst,base,final)
                operations+=1
            changed=sum(old!=(net[key].final_weight if key in net else record.base_edges.get(key)) for key,old in before.items())
            if not changed:
                return Receipt(ref,record.version,0,(),retained_footprint(self._journals)['total_bytes'])
            version=record.version+1;sequence=self._sequence+1
            candidate=dict(self._journals)
            if net:candidate[ref]=_encode(ref,record.base_version,version,
                    previous.first_sequence if previous is not None else sequence,tuple(net.values()))
            else:candidate.pop(ref,None)
            candidate=dict(candidate) if candidate else {}
            footprint=retained_footprint(candidate);candidate_peak=footprint['total_bytes']
            self._candidate_peak=max(self._candidate_peak,candidate_peak)
            plans=[]
            while footprint['total_bytes']>self.journal_budget_bytes:
                victim=min(candidate,key=lambda key:(candidate[key].first_sequence,key.graph_id,key.generation))
                item=self._record(victim);journal=candidate[victim]
                edges=self._logical(item,journal)
                plan=self._build(victim,item.nodes,edges,item.snapshot,_changes(journal),journal.version)
                plans.append((item,edges,plan));candidate.pop(victim);candidate=dict(candidate) if candidate else {}
                footprint=retained_footprint(candidate)
            # All fallible callback work succeeded. Publish scalar versions,
            # prepared authority mappings and the prepared ownership dictionary.
            record.version=version
            for item,edges,plan in plans:
                item.base_edges=edges;item.base_version=plan.new_snapshot.version;item.snapshot=plan.new_snapshot
            if ref not in candidate and all(plan.ref!=ref for _,_,plan in plans):
                # Cancellation back to the last materialized logical state can
                # retag that exact state without running any materializer.
                record.base_version=version
                if record.snapshot is not None:record.snapshot=Snapshot(ref,version,record.snapshot.host_state)
            self._journals=candidate;self._sequence=sequence
            self._retained_peak=max(self._retained_peak,footprint['total_bytes'])
            self._candidate_peak=max(self._candidate_peak,candidate_peak)
            self._counters['commits']+=1;self._counters['operations']+=operations
            self._counters['pressure_flushes']+=len(plans)
            self._counters['commit_seconds']+=time.perf_counter()-started
            return Receipt(ref,version,changed,tuple(plan for _,_,plan in plans),footprint['total_bytes'])

    def flush(self,ref):
        with self._lock:
            record=self._record(ref);journal=self._journals.get(ref)
            if journal is None and record.snapshot is not None and record.snapshot.version==record.version:
                return FlushResult(ref,record.snapshot,record.snapshot,(),True)
            changes=_changes(journal);edges=self._logical(record,journal)
            plan=self._build(ref,record.nodes,edges,record.snapshot,changes,record.version)
            journals={key:value for key,value in self._journals.items() if key!=ref}
            record.base_edges=edges;record.base_version=record.version;record.snapshot=plan.new_snapshot
            self._journals=journals;self._counters['explicit_flushes']+=1
            return plan

    def discard(self,ref):
        """Retire only the cached CSR/host materialization, never logical edits."""
        with self._lock:
            record=self._record(ref);old=record.snapshot;record.snapshot=None
            self._counters['physical_discards']+=1
            return DiscardResult(ref,record.version,old)

    def stats(self):
        with self._lock:
            footprint=retained_footprint(self._journals)
            return dict(self._counters,journal_budget_bytes=self.journal_budget_bytes,
                retained_journal=footprint,retained_peak_bytes=self._retained_peak,
                transient_candidate_peak_bytes=self._candidate_peak,
                graphs=len(self._records),pending_graphs=len(self._journals),
                pending_edges=sum(len(journal.data) for journal in self._journals.values()),
                authority_base_edges=sum(len(record.base_edges) for record in self._records.values()),
                materialized_graphs=sum(record.snapshot is not None for record in self._records.values()),
                accounting_scope='Unique retained structured-array payload plus reachable journal/ref/container metadata; immutable authority base mappings, host snapshots, temporary decode/materialization and process allocator fragmentation are separate')
