"""Explicit edge-array add API; independent of the frozen producer comparison.

Endpoints must be an integer NumPy (k, 2) array. Weights may be absent (unit
increments), a scalar, or a length-k array. No Mutation objects are constructed.
The publication body is the qualified R7 ordered-add transaction algorithm.
"""
import time
from time import perf_counter
import numpy as np
from easygraph.gpu._arena.round4_switch.journal import Receipt,Snapshot,_Journal,CHANGE_DTYPE,_changes,retained_footprint
from easygraph.gpu._arena.round7_recovery.host_authority_v2 import packed_eligible,encode_keys,indexed
from easygraph.gpu._arena.session_manager import request_boundary
from easygraph.gpu._arena.round8_host.session import Session


def commit_add_edges(store,ref,edges,weights=None):
    self=store
    with self._lock:
        started=perf_counter();record=self._record(ref);previous=self._journals.get(ref)
        if not isinstance(edges,np.ndarray) or edges.ndim!=2 or edges.shape[1]!=2 or edges.dtype.kind not in 'iu':
            raise ValueError('Integer NumPy endpoint array with shape (k,2) required')
        if edges.dtype.kind=='u' and edges.size and int(edges.max())>=1<<63:
            raise ValueError('Endpoint exceeds signed int64 domain')
        pairs=edges.astype(np.int64,copy=False)
        if weights is None:weights=np.ones(len(pairs),np.float64)
        else:
            if np.iscomplexobj(weights):raise ValueError('Real finite nonnegative weights required')
            try:weights=np.asarray(weights,np.float64)
            except (TypeError,ValueError,OverflowError) as error:raise ValueError('Real finite nonnegative weights required') from error
            if weights.ndim==0:weights=np.full(len(pairs),weights.item(),np.float64)
        if weights.ndim!=1 or len(weights)!=len(pairs):raise ValueError('Weight length mismatch')
        if not np.all(np.isfinite(weights)) or np.any(weights<0):raise ValueError('Finite nonnegative weight required')
        if not len(pairs):return Receipt(ref,record.version,0,(),retained_footprint(self._journals)['total_bytes'])
        # The snapshot already owns the immutable generation's validated node
        # domain; reusing it adds no buffer or cached state.
        nodes=(record.snapshot.host_state.nodes if record.snapshot is not None else np.asarray(record.nodes,np.int64))
        mapped=np.searchsorted(nodes,pairs)
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
        self._counters['commits']+=1;self._counters['operations']+=len(pairs)
        self._counters['pressure_flushes']+=len(plans)
        self._counters['commit_seconds']+=perf_counter()-started
        self._counters['typed_add_batches']+=1;self._counters['typed_add_operations']+=len(pairs)
        self._counters['array_add_batches']+=1;self._counters['array_add_operations']+=len(pairs)
        return Receipt(ref,version,changed,tuple(plan for _,_,plan in plans),footprint['total_bytes'])

class TypedSession(Session):
    @request_boundary
    def mutate_add_edges(self,gid,edges,weights=None,*,fail_after=None):
        if self.batch_requests is not None:
            raise RuntimeError('Updates commit between analysis epochs')
        started=time.perf_counter();self.logical_tick+=1
        ref=self.refs[gid];self._sync_snapshot(gid)
        if fail_after=='before_commit':raise RuntimeError('Injected before logical mutation commit')
        if fail_after is not None:raise ValueError('Unknown mutation failure point')
        receipt=commit_add_edges(self.journal,ref,edges,weights)
        for flush in receipt.auto_flushed:
            self._retire_graph(flush.ref.graph_id);self._adopt(flush,False)
            self.counters['host_pressure_flushes']+=1
        detail=None
        if self.mode=='eager' and self.journal.peek_pending(ref):detail=self._adopt(self.journal.flush(ref),True)
        self.counters['logical_operations']+=len(edges);self.counters['mutation_batches']+=1
        self.counters['array_mutation_batches']+=1;self._reconcile_residence()
        return dict(seconds=time.perf_counter()-started,logical_version=receipt.version,
                    maintenance=detail,auto_flushed=[x.ref.graph_id for x in receipt.auto_flushed])
