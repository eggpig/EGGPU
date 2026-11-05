#!/usr/bin/env python3
"""CPU-only range manifests and deterministic update positions for frozen CSR."""
from pathlib import Path
import argparse, copy, importlib.util, json, os, shutil, time
import state_protocol as c


def signer(source):
    path=source/'easygraph/gpu/_manifest_signatures.py';spec=importlib.util.spec_from_file_location('state_manifest_signer',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.sign


def serialize(root,out,name,parts,sign,weighted=True,identity=None):
    import numpy as np
    meta=json.loads((root/'canonical.json').read_text());n=meta['num_nodes'];m=meta['num_entries']
    off=np.memmap(root/'offsets.bin',dtype='<i4',mode='r',shape=(n+1,));ind=np.memmap(root/'indices.bin',dtype='<i4',mode='r',shape=(m,))
    w=np.memmap(root/'weights.bin',dtype='<f8',mode='r',shape=(m,)) if weighted else None
    cuts=np.linspace(0,n,parts+1,dtype=np.int64)
    if n<parts or not 0<m<2**31:raise ValueError('Unsupported node/arc domain')
    out.mkdir(parents=True,exist_ok=False);rows=[];edge_cuts=[]
    for rank,(first,last) in enumerate(zip(cuts[:-1],cuts[1:])):
        a,b=int(off[first]),int(off[last]);target=ind[a:b];mask=(target<first)|(target>=last);halo=np.unique(target[mask])
        mapping=np.r_[np.arange(first,last,dtype='<u8'),halo.astype('<u8')];inverse=np.full(n,-1,dtype='<i4');inverse[mapping]=np.arange(len(mapping),dtype='<i4')
        arrays={'offsets':(off[first:last+1]-a).astype('<i4'),'indices':inverse[target],
                'local_to_global':mapping,'owners':np.searchsorted(cuts[1:],mapping,side='right').astype('<i4')}
        if weighted:arrays['weights']=w[a:b]
        files={}
        for key,array in arrays.items():
            path=out/f'p{rank}.{key}.bin';array.tofile(path)
            files[key]={'path':path.name,'dtype':'uint64' if key=='local_to_global' else 'float64' if key=='weights' else 'int32',
                        'count':len(array),'byte_size':array.nbytes,'sha256':c.sha(path)}
        rows.append(dict(partition_id=rank,device_id=rank,owned_vertex_count=int(last-first),halo_vertex_count=len(halo),adjacency_count=b-a,
                         cut_edge_count=int(mask.sum()),ownership={'encoding':'range','vertex_begin':int(first),'vertex_end':int(last)},files=files))
        edge_cuts.append(a)
    edge_cuts.append(m)
    doc={'format':'egmgpu-partitioned-csr','version':1,
         'graph':{'id':identity or name+'@0','generation':1,'vertex_count':n,'adjacency_count':m,'edge_count':meta['num_edges'],'directed':meta['directed'],'weighted':weighted},
         'indexing':{'endianness':'little','global_vertex_dtype':'uint64','index_dtype':'int32','offset_dtype':'int32'},
         'partitioning':{'ownership_encoding':'range'},'view':{'orientation':'outgoing' if meta['directed'] else 'undirected','weight_key':'controlled_positive_integer' if weighted else None},'partitions':rows}
    top,weight,view=sign(doc);doc['graph']['topology_sha256']=top
    if weighted:doc['graph']['weight_sha256']=weight
    doc['view']['signature']=view
    c.save(out/'manifest.json',doc);return c.bind(out/'manifest.json'),edge_cuts


def projections(root,manifest,name,parts,sign):
    """Explicit immutable topology views accepted by the production API.

    BFS keeps canonical orientation; WCC consumes the deduplicated weak graph.
    These identities stay fixed through weight-only commits. Existing native
    compatibility keys decide physical sharing; no digest is forged to force it.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    base=Path(manifest['path']);doc=copy.deepcopy(json.loads(base.read_text()));directory=base.parent
    doc['graph'].update(id=name+'@topology',weighted=False);doc['graph'].pop('weight_sha256',None)
    doc['view']['weight_key']=None
    for part in doc['partitions']:part['files'].pop('weights')
    top,_,view=sign(doc);doc['graph']['topology_sha256']=top;doc['view']['signature']=view
    topology=directory/'topology_manifest.json';c.save(topology,doc)
    result={'bfs':{'manifest':c.bind(topology),'identity':name+'@topology'}}
    if not doc['graph']['directed']:
        weak=copy.deepcopy(doc);weak['graph']['id']=name+'@weak'
        path=directory/'weak_manifest.json';c.save(path,weak)
    else:
        n=doc['graph']['vertex_count'];m=doc['graph']['adjacency_count']
        off=np.memmap(root/'offsets.bin',dtype='<i4',mode='r',shape=(n+1,));ind=np.memmap(root/'indices.bin',dtype='<i4',mode='r',shape=(m,))
        matrix=csr_matrix((np.ones(m,dtype=np.uint8),ind,off),shape=(n,n));weak=matrix.maximum(matrix.T).tocsr();weak.sort_indices()
        if weak.nnz>=2**31:raise OverflowError('State weak projection exceeds current int32 serializer; no edges dropped')
        projected=directory/'weak_canonical';projected.mkdir()
        weak.indptr.astype('<i4').tofile(projected/'offsets.bin');weak.indices.astype('<i4').tofile(projected/'indices.bin')
        c.save(projected/'canonical.json',dict(num_nodes=n,num_entries=int(weak.nnz),num_edges=int(weak.nnz)//2,directed=False))
        item,_=serialize(projected,directory/'weak',name,parts,sign,weighted=False,identity=name+'@weak');path=Path(item['path'])
        # The partition files retain exact hashes; avoid a duplicate whole weak CSR.
        shutil.rmtree(projected)
    result['wcc']={'manifest':c.bind(path),'identity':name+'@weak'}
    for value in result.values():
        path=Path(value['manifest']['path']);document=json.loads(path.read_text())
        value['files']=[c.bind(path.parent/spec['path']) for part in document['partitions'] for spec in part['files'].values()]
    return result


def positions(root,rho,locality,seed):
    import numpy as np
    meta=json.loads((root/'canonical.json').read_text());n=meta['num_nodes'];m=meta['num_entries'];rng=np.random.default_rng(seed)
    count=int(m*rho);count=count if meta['directed'] else 2*(count//2)
    if count==0:return np.empty(0,dtype='<i8')
    if count==m:return np.arange(m,dtype='<i8')
    if meta['directed']:
        if locality=='uniform':return np.sort(rng.choice(m,count,replace=False)).astype('<i8')
        start=int(rng.integers(m-count+1));return np.arange(start,start+count,dtype='<i8')
    off=np.memmap(root/'offsets.bin',dtype='<i4',mode='r',shape=(n+1,));ind=np.memmap(root/'indices.bin',dtype='<i4',mode='r',shape=(m,))
    chosen=set();cursor=int(rng.integers(m))
    while len(chosen)<count:
        pos=int(rng.integers(m)) if locality=='uniform' else cursor%m;cursor+=1
        if pos in chosen:continue
        u=int(np.searchsorted(off,pos,side='right')-1);v=int(ind[pos]);a,b=int(off[v]),int(off[v+1])
        reverse=a+int(np.searchsorted(ind[a:b],u))
        if reverse>=b or ind[reverse]!=u:raise ValueError('Undirected reverse arc absent')
        chosen.update((pos,reverse))
    return np.array(sorted(chosen),dtype='<i8')


def tiny(root,n,seed):
    import numpy as np
    root.mkdir(parents=True,exist_ok=False)
    arcs=sorted({(u,(u+1)%(n-3)) for u in range(n-3)}|{(u,(u+5)%(n-3)) for u in range(0,n-3,2)})
    off=np.r_[0,np.cumsum(np.bincount([u for u,v in arcs],minlength=n))].astype('<i4');ind=np.array([v for u,v in arcs],dtype='<i4')
    w=((np.arange(len(arcs))+seed)%16+1).astype('<f8')
    for file,array in [('offsets.bin',off),('indices.bin',ind),('weights.bin',w)]:array.tofile(root/file)
    c.save(root/'canonical.json',dict(num_nodes=n,num_entries=len(arcs),num_edges=len(arcs),directed=True))
    c.save(root/'sources.json',dict(primary_sources=[0,1,2]))
    return root


def prepare(a):
    os.environ['CUDA_VISIBLE_DEVICES']='';import numpy as np
    if not 0<=a.rho<=1:raise ValueError('rho outside [0,1]')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);sign=signer(a.source);rows={};begin=time.monotonic()
    names=a.datasets.split(',') if not a.tiny else ['TINY_A','TINY_B','TINY_C']
    for index,name in enumerate(names):
        if not a.tiny and name not in set().union(*c.POOLS.values()):raise ValueError('State graph is outside frozen pools')
        root=tiny(out/(name+'_canonical'),31+6*index,index) if a.tiny else a.canonical/name
        if not a.tiny:
            receipt=json.loads((root/'receipt.json').read_text())
            if receipt['status']!='PASS_CANONICAL_INPUT' or receipt['dataset']!=name:raise ValueError('Unqualified canonical source')
            for item in receipt['files'].values():c.checked(item)
        meta=json.loads((root/'canonical.json').read_text());n,m=meta['num_nodes'],meta['num_entries']
        if shutil.disk_usage(out).free<28*m+48*n*a.partitions+8*int(m*a.rho)+(2<<30):raise RuntimeError('Insufficient manifest/output preparation disk reserve')
        manifest,cuts=serialize(root,out/name,name,a.partitions,sign)
        projected=projections(root,manifest,name,a.partitions,sign)
        pos=positions(root,a.rho,a.locality,20261104+index);path=out/name/'update_positions.npy';np.save(path,pos,allow_pickle=False)
        rows[name]={'canonical':{role:c.bind(root/file) for role,file in [('offsets','offsets.bin'),('indices','indices.bin'),('weights','weights.bin'),('metadata','canonical.json'),('sources','sources.json')]},
                    'manifest':manifest,'projections':projected,'n':n,'m':m,'edge_cuts':cuts,'positions':c.bind(path),
                    'requested_rho':a.rho,'actual_rho':len(pos)/m,'changed_positions':len(pos),'locality':a.locality,
                    'sources':json.loads((root/'sources.json').read_text())['primary_sources'],
                    'manifest_files':[c.bind(out/name/spec['path']) for part in json.loads(Path(manifest['path']).read_text())['partitions'] for spec in part['files'].values()]}
    record={'schema':'state.inputs.v2','status':'PASS_CPU_INPUTS_GPU_UNQUALIFIED','datasets':rows,'partitions':a.partitions,
            'qualification_only':a.tiny,'gpu_executed':False,'seconds':time.monotonic()-begin,
            'weight_update_rule':'changed values for version r are 1+((initial_weight-1+r) mod 16); undirected paired arcs change together',
            'source_signer':c.bind(a.source/'easygraph/gpu/_manifest_signatures.py')}
    c.save(out/'inputs.json',record);print(json.dumps({'status':record['status'],'inputs':str(out/'inputs.json')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--canonical',type=Path)
    p.add_argument('--datasets',default='US,IN,OR');p.add_argument('--partitions',type=int,default=1);p.add_argument('--rho',type=float,default=1e-4)
    p.add_argument('--locality',choices=['uniform','clustered'],default='uniform');p.add_argument('--output',type=Path,required=True);p.add_argument('--tiny',action='store_true')
    a=p.parse_args()
    if not a.tiny and a.canonical is None:p.error('--canonical required for frozen main graphs')
    prepare(a)
