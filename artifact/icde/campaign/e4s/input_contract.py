"""CPU-only frozen E4-S source sequences and author-required input layouts."""
from pathlib import Path
import hashlib, heapq, json, math, os, sys, time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import sg_worker as common

DATASETS={'US':'suitesparse-dimacs10-road_usa','OR':'nr-soc-orkut','IN':'law-indochina-2004','UK':'law-uk-2002'}
BATCHES=(32,128,512)
KGRAPH_PARTITIONS=8


def sources(graph,name,count=512):
    from scipy.sparse.csgraph import connected_components
    matrix=common.graph_matrix(graph);_,labels=connected_components(matrix,directed=graph['directed'],
        connection='strong' if graph['directed'] else 'weak',return_labels=True)
    sizes=np.bincount(labels);largest=np.flatnonzero(sizes==sizes.max())
    chosen=int(labels[np.flatnonzero(np.isin(labels,largest))[0]])
    eligible=np.flatnonzero(labels==chosen)
    if len(eligible)<count:raise ValueError('Largest component cannot provide frozen distinct batch; no source filtering/substitution')
    prefix=('eggpu-source-v1|20261104|'+DATASETS.get(name,name)+'|').encode()
    rank=lambda vertex:(hashlib.sha256(prefix+str(int(vertex)).encode()).digest(),int(vertex))
    selected=[vertex for _,vertex in heapq.nsmallest(count,map(rank,eligible))]
    if name in DATASETS and selected[:3]!=graph['sources']:raise ValueError('Extended source prefix differs from canonical primary three')
    return {'schema':'e4s.sources.v1','dataset':name,'sources':selected,'batches':list(BATCHES),
            'largest_component_vertices':len(eligible),'input_sha256':graph['input_sha256'],
            'rule':'Extend frozen largest-component SHA256 ordering, preserve prefix across batch lengths; no query results or compression statistics used',
            'selection_uses_timing':False,'implementation':common.content_binding(__file__)}


def load_sources(path,graph,count):
    record=json.loads(Path(path).read_text())
    if record.get('schema')!='e4s.sources.v1' or record['input_sha256']!=graph['input_sha256'] or record['implementation']!=common.content_binding(__file__):
        raise ValueError('Source sequence belongs to another canonical input/recipe')
    sequence=record['sources'][:count]
    if count not in BATCHES or len(sequence)!=count or len(set(sequence))!=count or any(type(v)is not int or not 0<=v<graph['n'] for v in sequence):
        raise ValueError('Invalid frozen E4-S batch sequence')
    return sequence


def write_vector(stream,values):
    for start in range(0,len(values),1<<20):np.savetxt(stream,values[start:start+(1<<20)],fmt='%d')


def kgraph_layout(graph,out,partitions=KGRAPH_PARTITIONS):
    """Paper VI-D legal fixed chunk configuration, not the unavailable optimizer.

    The full input preparation interval includes chunking, edge separation and
    author-format serialization. No vertex renumbering, edge loss or weights change.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=False);n,m=graph['n'],graph['m']
    if partitions<1 or partitions>n:raise ValueError('Invalid chunk partition count')
    if (n-1)*16>=min(1000000000,(2**31-1)//2):raise OverflowError('Exact w16 upper distance bound collides with author sentinel')
    cuts=np.linspace(0,n,partitions+1,dtype=np.int64)
    ids=np.searchsorted(cuts[1:],np.arange(n),side='right').astype(np.int32)
    src=np.repeat(np.arange(n,dtype=np.int32),np.diff(graph['offsets']))
    inter=ids[src]!=ids[graph['indices']]
    counts={}
    for name,mask in [('inter',inter),('intra',~inter)]:
        degree=np.bincount(src[mask],minlength=n);offset=np.r_[0,np.cumsum(degree[:-1])]
        count=int(mask.sum());counts[name]=count
        with (out/(name+'.adj')).open('w') as stream:
            stream.write('WeightedAdjacencyGraph\n'+str(n)+'\n'+str(count)+'\n')
            write_vector(stream,offset);write_vector(stream,graph['indices'][mask]);write_vector(stream,graph['weights'][mask])
    with (out/'parts.txt').open('w') as stream:write_vector(stream,ids)
    if counts['inter']+counts['intra']!=m:raise ValueError('Author partition split lost an arc')
    return {'partitions':partitions,'method':'fixed_contiguous_vertex_chunks','partition_cuts':cuts.tolist(),
            'inter_arcs':counts['inter'],'intra_arcs':counts['intra'],'all_vertices_preserved':True,
            'optimizer_used':False,'parameter_source':'KGraph ICDE2025 VI-D fixed eight-partition comparison; explicit chunk configuration'}


def c2_layout(graph,out):
    out=Path(out);(out/'origin').mkdir(parents=True,exist_ok=False);(out/'sssp').mkdir()
    if (graph['n']-1)*16>=1000000000:raise OverflowError('C2 PM int32 distance sentinel unsupported')
    # The common CSR has exactly the required offset/index representation. Keep
    # immutable canonical arrays; only the explicit int32 weight conversion copies.
    source=Path(graph['directory'])
    for filename,role in [('vlist.bin','offsets'),('elist.bin','indices')]:
        (out/'origin'/filename).symlink_to((source/(role+'.bin')).resolve())
    graph['weights'].astype('<i4').tofile(out/'origin/wlist.bin')
    return {'layout':'author_origin_int32_csr','vertices':graph['n'],'arcs':graph['m'],'weights_exact':True}


def tiny(root):
    """521 vertices: 513-node SCC, separate 3-node SCC, five isolates."""
    root=Path(root);common.make_tiny(root)
    for name,directed in [('TINY_D',True),('TINY_U',False)]:
        folder=root/'canonical'/name;n=521
        edges={(v,(v+1)%513) for v in range(513)}|{(v,(v+17)%513) for v in range(0,513,7)}|{(513,514),(514,515),(515,513)}
        if not directed:edges|={(v,u) for u,v in edges}
        edges=sorted(edges);off=np.r_[0,np.cumsum(np.bincount([u for u,_ in edges],minlength=n))].astype('<i4')
        ind=np.array([v for _,v in edges],dtype='<i4');weights=np.array([1+((min(u,v)*131+max(u,v)*17)%16) for u,v in edges],dtype='<f8')
        off.tofile(folder/'offsets.bin');ind.tofile(folder/'indices.bin');weights.tofile(folder/'weights.bin')
        meta=json.loads((folder/'canonical.json').read_text());meta.update(num_nodes=n,num_entries=len(edges),num_edges=len(edges) if directed else len(edges)//2)
        common.save(folder/'canonical.json',meta);receipt=json.loads((folder/'receipt.json').read_text())
        for role,item in receipt['files'].items():receipt['files'][role]=common.binding(folder/Path(item['path']).name)
        receipt['e4s_fixture']='521 vertices, up to 512 unique source requests, nonunit weights and disconnected/isolated vertices'
        common.save(folder/'receipt.json',receipt);graph=common.load_input(folder);record=sources(graph,name)
        common.save(folder/'e4s_sources.json',record)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input-root',type=Path,required=True);p.add_argument('--dataset',choices=DATASETS)
    p.add_argument('--make-tiny',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.make_tiny:tiny(a.input_root)
    else:
        if not a.dataset or not a.output:p.error('A frozen dataset and output path are required')
        graph=common.load_input(a.input_root/'canonical'/a.dataset);common.save(a.output,sources(graph,a.dataset))
