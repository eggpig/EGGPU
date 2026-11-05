"""CPU selector for a fixed-prefix partition update-locality intervention.

Select exactly floor(m*rho) arcs (rounded down to paired arcs for an
undirected input), touching all and only owners 0..K-1. Every undirected
edge includes its reverse. A topology-only coverage anchor per owner is
included before uniform sampling without replacement of remaining edges.
Insufficient capacity/owner coverage yields explicit INFEASIBLE.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def binding(path):
    path=Path(path).resolve();digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def select(offsets, indices, directed, edge_cuts, vertex_cuts, k, rho, seed=20261104, chunk=1 << 20):
    n,m=len(offsets)-1,len(indices)
    cuts=np.asarray(vertex_cuts,dtype=np.int64)
    edges=np.asarray(edge_cuts,dtype=np.int64)
    if (len(cuts)!=9 or len(edges)!=9 or k not in (1,2,4,8) or
        not np.array_equal(cuts,np.linspace(0,n,9,dtype=np.int64)) or
        not np.array_equal(edges,offsets[cuts]) or not 0<rho<=1 or chunk<=0):
        raise ValueError('Require the frozen eight-way vertex partition and positive density')
    requested=int(m*rho);count=requested if directed else 2*(requested//2)
    units=count if directed else count//2
    base=dict(schema='state.partition-hotspot.selector/v1',gpu_executed=False,performance_eligible=False,
        partitions=8,k=k,selected_owners=list(range(k)),rho_requested=rho,arcs_requested=count,
        seed=seed,sampling='one topology-only coverage anchor per owner, then uniform remaining logical edges',
        undirected_reverse_arcs_included=not directed)

    def batches():
        for lo in range(0,int(edges[k]),chunk):
            hi=min(int(edges[k]),lo+chunk)
            positions=np.arange(lo,hi,dtype=np.int64)
            # Preserve scalar/vector probe dtype and avoid repeated int64 CSR copies.
            rows=np.searchsorted(offsets,positions.astype(offsets.dtype),side='right')-1
            target=indices[lo:hi]
            valid=np.ones(hi-lo,dtype=bool) if directed else ((target<cuts[k]) & (rows<target))
            positions=positions[valid];rows=rows[valid];target=target[valid]
            owner=np.searchsorted(cuts,rows,side='right')-1
            other=np.searchsorted(cuts,target,side='right')-1
            yield positions,owner,other

    total=0;anchors={}
    for positions,owner,other in batches():
        for p in range(k):
            if p in anchors:continue
            eligible=(owner==p) if directed else ((owner==p)|(other==p))
            found=np.flatnonzero(eligible)
            if len(found):anchors[p]=total+int(found[0])
        total+=len(positions)
    ranks=np.array(sorted(set(anchors.values())),dtype=np.int64)
    if units>total or len(anchors)!=k or units<len(ranks):
        return np.empty(0,dtype='<i8'),dict(base,status='INFEASIBLE',eligible_logical_edges=total,
            owners_with_eligible_edges=sorted(anchors),minimum_coverage_edges=len(ranks),
            reason='Insufficient eligible edges, requested count, or per-owner topology coverage')
    rng=np.random.default_rng(seed)
    random=np.sort(rng.choice(total-len(ranks),units-len(ranks),replace=False))
    for blocked in ranks:random[random>=blocked]+=1
    chosen_ranks=np.sort(np.r_[ranks,random])
    selected=[];total_seen=0
    for positions,_,_ in batches():
        a,b=np.searchsorted(chosen_ranks,[total_seen,total_seen+len(positions)])
        selected.extend(positions[chosen_ranks[a:b]-total_seen].tolist())
        total_seen+=len(positions)
    if not directed:
        reverse=[]
        for position in selected:
            u=int(np.searchsorted(offsets,np.asarray(position,dtype=offsets.dtype),side='right')-1)
            v=int(indices[position]);a,b=int(offsets[v]),int(offsets[v+1])
            rev=a+int(np.searchsorted(indices[a:b],np.asarray(u,dtype=indices.dtype)))
            if rev>=b or int(indices[rev])!=u:raise ValueError('Missing undirected reverse arc')
            reverse.append(rev)
        selected+=reverse
    result=np.array(sorted(selected),dtype='<i8')
    affected=sorted(set((np.searchsorted(edges,result,side='right')-1).tolist()))
    if len(result)!=count or len(np.unique(result))!=count or affected!=list(range(k)):
        raise ValueError('Partition-hotspot output violated exact count/uniqueness/owner domain')
    return result,dict(base,status='PASS_CPU_SELECTION',eligible_logical_edges=total,
        changed_positions=count,actual_rho=count/m if m else 0,affected_owners=affected,
        coverage_anchor_logical_edges=len(ranks))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True,help='Existing P8 shared base.json')
    p.add_argument('--k',type=int,choices=(1,2,4,8),required=True)
    p.add_argument('--rho',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.canonical/'canonical.json').read_text());n,m=meta['num_nodes'],meta['num_entries']
    base=json.loads(a.base.read_text());row=base['row']
    if row['n']!=n or row['m']!=m:raise ValueError('Canonical/base domain mismatch')
    for role,name in [('offsets','offsets.bin'),('indices','indices.bin'),('weights','weights.bin'),('metadata','canonical.json'),('sources','sources.json')]:
        if binding(a.canonical/name)!=row['canonical'][role]:raise ValueError('Canonical/base payload differs: '+role)
    offsets=np.memmap(a.canonical/'offsets.bin',dtype='<i4',mode='r',shape=(n+1,))
    indices=np.memmap(a.canonical/'indices.bin',dtype='<i4',mode='r',shape=(m,))
    values,receipt=select(offsets,indices,meta['directed'],row['edge_cuts'],np.linspace(0,n,9,dtype=np.int64),a.k,a.rho,a.seed)
    receipt.update(base=binding(a.base),canonical=row['canonical'],selector=binding(__file__))
    if receipt['status']=='PASS_CPU_SELECTION':
        np.save(a.output/'positions.npy',values,allow_pickle=False)
        receipt['positions']=binding(a.output/'positions.npy')
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
