"""Typed transitions through authority, CSR and certificate publication.

All transaction algorithms retain their frozen operation order. Only the
published change representation and its consumers differ. Descriptor ownership
includes the whole 33-byte journal allocation even after pending-log release.
"""
from collections import Counter
from dataclasses import replace
import sys,time,weakref
from time import perf_counter
import numpy as np
from easygraph.gpu._arena.round4_switch.journal import (Mutation,EdgeChange,Snapshot,Receipt,FlushResult,
    _Journal,CHANGE_DTYPE,_changes,_integer,_weight,_encode,retained_footprint)
from easygraph.gpu._arena.round7_recovery import host_authority as v1
from easygraph.gpu._arena.round7_recovery import host_authority_v2 as r7
from easygraph.gpu._arena.round7_recovery.host_authority_v2 import packed_eligible,encode_keys,indexed
from easygraph.gpu._arena.round8_bounded_input.authority_v2 import BoundedInputStoreV2
from easygraph.gpu._arena.round8_host.host_authority import Store as HostStore,description,merge_direction,payload
from easygraph.gpu._arena.round2_topology.state import Delta
from easygraph.gpu._arena.round2_topology.gpu_adapter import PreparedState
from easygraph.gpu._arena.round8_columns.columns import TransitionColumns,ColumnsCache,deletion,insert_count

class Store(BoundedInputStoreV2):
    def __init__(self,*args,change_budget_bytes=8<<20,**kwargs):
        self.issued={};self.transition_peak=0;self.extended_owner_peak=0
        owner=weakref.ref(self)
        def created(changes):
            store=owner()
            if store is not None:store._issued(changes)
        self.change_cache=ColumnsCache(change_budget_bytes,created)
        super().__init__(*args,**kwargs)

    def _issued(self,changes):
        owner=weakref.ref(self);key=id(changes)
        def released(token):
            store=owner()
            if store is not None and store.issued.get(key) is token:store.issued.pop(key,None)
        self.issued[key]=weakref.ref(changes,released)
        self._observe_owners()

    def _live_columns(self):
        return [c for token in tuple(self.issued.values()) if (c:=token()) is not None]

    def _owner_arrays(self):
        return [c.data for c in self._live_columns()]

    def _observe_owners(self):
        arrays=self._owner_arrays()
        self.transition_peak=max(self.transition_peak,payload(arrays))
        active={id(j.data) for j in getattr(self,'_journals',{}).values()}
        extended=payload([a for a in arrays if id(a) not in active])
        self.extended_owner_peak=max(self.extended_owner_peak,extended)

    def _columns(self,journal):
        snapshot=None if journal is None else self._record(journal.ref).snapshot
        return self.change_cache.get(journal,None if snapshot is None else snapshot.host_state)

    def peek_pending(self,ref):
        with self._lock:
            self._record(ref);self.change_cache.prune(self._journals)
            return self._columns(self._journals.get(ref))

    def flush(self,ref):
        with self._lock:
            record=self._record(ref);journal=self._journals.get(ref)
            self.change_cache.prune(self._journals)
            if journal is None and record.snapshot is not None and record.snapshot.version==record.version:
                return FlushResult(ref,record.snapshot,record.snapshot,(),True)
            changes=self._columns(journal);edges=self._logical(record,journal)
            plan=self._build(ref,record.nodes,edges,record.snapshot,changes,record.version)
            journals={key:value for key,value in self._journals.items() if key!=ref}
            record.base_edges=edges;record.base_version=record.version;record.snapshot=plan.new_snapshot
            self._journals=journals;self._counters['explicit_flushes']+=1
            self.change_cache.prune(self._journals);self._observe_owners()
            return plan

    def _sparse_attributes(self,old,src,dst,values):
        # Reuse the frozen sparse changed-row repair after supplying an already
        # canonical Delta. No EdgeChange, endpoint aggregation or duplicate sort.
        positions,present=self.index.locate(old.outgoing,src,dst)
        if not np.all(present):raise AssertionError('Attribute edge absent in source layout')
        state=PreparedState.__new__(PreparedState);state.__dict__=old.__dict__.copy()
        for name in ('values','sums','degree'):
            setattr(state,name,np.array(getattr(old,name),dtype=np.float64,copy=True))
        state.readers=0
        delta=Delta(src,dst,old.outgoing.ids[positions],values,np.zeros(len(src),bool),np.unique(src))
        state.apply_prepared(None,delta,'shared_delta')
        return state

    def _materialize(self,ref,nodes,edges,previous,changes,version):
        if not isinstance(changes,TransitionColumns):
            return super()._materialize(ref,nodes,edges,previous,changes,version)
        if changes.ref!=ref or changes.version!=version or (previous is not None and (previous.version!=changes.base_version or changes.source_state is None or changes.source_state() is not previous.host_state)):
            raise ValueError('Transition version/source mismatch')
        self._remove_certificate(ref)
        if previous is None or previous.ref!=ref or deletion(changes):
            state=r7.cold_state(nodes,edges);state._r8_authority=weakref.ref(edges)
            return state
        old=previous.host_state
        if not changes:return old
        src=np.searchsorted(old.nodes,changes.src).astype(np.int32)
        dst=np.searchsorted(old.nodes,changes.dst).astype(np.int32)
        values=changes.final
        if not changes.new_count:
            state=self._sparse_attributes(old,src,dst,values)
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
            state=self._sparse_attributes(old,src,dst,values)
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

    def commit(self,ref,mutations):
        with self._lock:
            record=self._record(ref);mutations=list(mutations)
            try:
                if mutations and all(isinstance(m,Mutation) and m.mode=='add' and
                    not isinstance(m.src,(bool,np.bool_)) and not isinstance(m.dst,(bool,np.bool_)) and
                    isinstance(m.src,(int,np.integer)) and isinstance(m.dst,(int,np.integer)) and
                    -(1<<63)<=int(m.src)<1<<63 and -(1<<63)<=int(m.dst)<1<<63 for m in mutations):
                    return self._add_commit(record,mutations)
                started=perf_counter();base=record.base_edges;accelerated=v1._BatchBase(base,mutations)
                self._counters['batch_lookup_seconds']+=perf_counter()-started
                self._counters['batch_lookup_calls']+=1;self._counters['batch_lookup_keys']+=len(accelerated.lookup)
                record.base_edges=accelerated
                try:return self._generic_commit(ref,mutations)
                finally:
                    if record.base_edges is accelerated:record.base_edges=base
            finally:
                self.change_cache.prune(self._journals);self._observe_owners()

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
            plan=self._build(victim,item.nodes,edges,item.snapshot,self._columns(journal),journal.version)
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

    def _generic_commit(self,ref,mutations):
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
                plan=self._build(victim,item.nodes,edges,item.snapshot,self._columns(journal),journal.version)
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

    def _retained_change(self):
        # Certificate final-value views may borrow the 33N owner. Count that
        # allocation once alongside the explicit dense certificate arrays.
        arrays=[a for c in self.certificates.values() for a in c.arrays()]
        arrays += [c.changes.data for c in self.certificates.values() if isinstance(c.changes,TransitionColumns)]
        certificate_live=payload(arrays)
        self.certificate_peak=max(self.certificate_peak,certificate_live)
        self.extra_retained_peak=max(self.extra_retained_peak,self.index.live+certificate_live)
        self._observe_owners()

    def certificate_bytes(self):
        for ref,c in list(self.certificates.items()):
            if c.old() is None or c.new() is None:self._remove_certificate(ref)
        arrays=[a for c in self.certificates.values() for a in c.arrays()]
        arrays += [c.changes.data for c in self.certificates.values() if isinstance(c.changes,TransitionColumns)]
        return payload(arrays)

    def replace(self,ref,*args,**kwargs):
        try:return super().replace(ref,*args,**kwargs)
        finally:self.change_cache.prune(self._journals);self._observe_owners()

    def discard(self,ref):
        result=super().discard(ref);self.change_cache.clear('discard');self._observe_owners()
        return result

    def stats(self):
        with self._lock:
            self.change_cache.prune(self._journals);self._observe_owners()
            result=super().stats();columns=self._live_columns()
            active={id(j.data) for j in self._journals.values()}
            owners=payload([c.data for c in columns]);extended=payload([c.data for c in columns if id(c.data) not in active])
            observed_columns=tuple(columns)
            members=(self._journals,self.index.entries,self.certificates,self.change_cache.entry,observed_columns,self.issued)
            union=retained_footprint(members)
            # The observation-only containers are not retained by the Store.
            observation_headers=sys.getsizeof(members)+sys.getsizeof(observed_columns)
            union['metadata_bytes']-=observation_headers
            union['total_bytes']-=observation_headers
            return dict(result,prepared_changes=self.change_cache.stats(),
                prepared_host_union=dict(union,scope='Unique reachable pending journals, typed descriptors, cache, certificates and index entries. Published external descriptors known through weak tracking count their full owner; weakrefs do not retain snapshots. CSR/authority arrays and RSS separate.'),
                typed_transitions=dict(live_descriptors=len(columns),owner_payload_bytes=owners,
                    owner_payload_peak_bytes=self.transition_peak,extended_owner_payload_bytes=extended,
                    extended_owner_payload_peak_bytes=self.extended_owner_peak,
                    pending_and_issued_owner_payload_bytes=payload([j.data for j in self._journals.values()]+[c.data for c in columns]),
                    row_object_decodes_in_publication=0,
                    scope='33N immutable journal owner counted once across all five column views and current pending journal; extended means not currently owned by published pending journals, including pressure-preparation candidates and externally held receipts/descriptors. Peaks observe creation/publication, not RSS. Generic object mutation validation remains its original scalar transaction.'))
