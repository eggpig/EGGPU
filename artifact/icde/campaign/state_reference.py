"""Independent CPU full-vector references for state workflows, outside timing.

Entries bind canonical bytes, deterministic mutation recipe, source vertex,
reference implementation/dependency versions and numerical quality contract.
They never consume EGGPU or another GPU implementation's output. Lossless NPZ
avoids duplicating large FP64 references for every arm/repetition. Every tested
output still receives a complete-vector audit and PR fixed-point residual.
"""
from pathlib import Path
import fcntl, hashlib, json, os, shutil, tempfile, time
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
import state_protocol as c

QUALITY={'version':'state-full-vector-cpu-v2','pr_alpha':.85,'reference_stop_l1':1e-12,
         'reference_cap':2000,'residual_l1':1e-6,'mass_error':1e-6,'reference_l1':1e-5,
         'sssp':'exact_fp64_plus_infinity','bfs':'exact_plus_infinity','wcc':'partition_equivalence'}


def content(item):return {k:item[k] for k in ('bytes','sha256')}


def key_document(row,version,function,source):
    weighted=function in ('sssp','weighted_pr')
    document={'schema':'state.reference.v2','n':row['n'],'m':row['m'],'function':function,
       'source':int(source) if function in ('bfs','sssp') else None,
       'version':int(version) if weighted else 0,
       'input':{k:content(v) for k,v in row['canonical'].items() if k!='weights' or weighted},
       'positions':content(row['positions']) if weighted and version else None,
       'weight_recipe':'base weights; if version>0 set selected positions to 1+(base-1+version)%16',
       'orientation':'canonical directed CSR; WCC weak connectivity',
       'quality':QUALITY,'reference_source':content(c.bind(__file__)),
       'recipe_source':content(c.bind(Path(c.__file__).with_name('state_inputs.py'))),
       'numpy':np.__version__,'scipy':scipy.__version__}
    digest=hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return document,digest


def matrix_for(row,version,function):
    n,m=row['n'],row['m'];paths=row['canonical']
    off=np.memmap(paths['offsets']['path'],dtype='<i4',mode='r',shape=(n+1,))
    ind=np.memmap(paths['indices']['path'],dtype='<i4',mode='r',shape=(m,))
    if function in ('sssp','weighted_pr'):
        weights=np.memmap(paths['weights']['path'],dtype='<f8',mode='r',shape=(m,))
        if version:
            weights=weights.copy();positions=np.load(row['positions']['path'],allow_pickle=False,mmap_mode='r')
            weights[positions]=1+(weights[positions]-1+version)%16
    else:weights=np.ones(m,dtype=np.float64)
    return csr_matrix((weights,ind,off),shape=(n,n),copy=False)


def pr_step(matrix):
    n=matrix.shape[0];strength=np.asarray(matrix.sum(axis=1)).reshape(-1);dangling=strength==0
    inv=np.divide(1.,strength,out=np.zeros(n),where=strength>0)
    transpose=matrix.T
    def step(values):return .15/n+.85*(transpose@(values*inv)+values[dangling].sum()/n)
    return step


def build_reference(matrix,function,source):
    if function in ('bfs','sssp'):
        return dijkstra(matrix,directed=True,indices=source,unweighted=function=='bfs'),{'solver':'scipy.csgraph.dijkstra'}
    if function=='wcc':
        count,labels=connected_components(matrix,directed=False,return_labels=True)
        return labels,{'solver':'scipy.csgraph.connected_components','components':int(count)}
    if function!='weighted_pr':raise ValueError('Unsupported independent reference')
    step=pr_step(matrix);reference=np.full(matrix.shape[0],1/matrix.shape[0])
    for iteration in range(QUALITY['reference_cap']):
        nxt=step(reference);delta=float(np.abs(nxt-reference).sum());reference=nxt
        if delta<QUALITY['reference_stop_l1']:
            return reference,{'solver':'independent_fp64_power_iteration','iterations':iteration+1,'last_update_l1':delta}
    raise ValueError('Independent PR reference did not converge')


class ReferenceCache:
    def __init__(self,path,max_bytes=4<<30,min_free_bytes=2<<30):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.max_bytes=int(max_bytes);self.min_free_bytes=int(min_free_bytes)
        if self.max_bytes<=0 or self.min_free_bytes<0:raise ValueError('Invalid independent-reference disk budget')

    def _make_space(self,requested,exclude=None):
        entries=sorted(self.path.glob('*.npz'),key=lambda p:p.stat().st_mtime_ns)
        used=sum(p.stat().st_size for p in entries)
        for path in entries:
            if used+requested<=self.max_bytes and shutil.disk_usage(self.path).free>=self.min_free_bytes+requested:break
            if path==exclude:continue
            used-=path.stat().st_size;path.unlink();path.with_suffix('.json').unlink(missing_ok=True)
        return used+requested<=self.max_bytes and shutil.disk_usage(self.path).free>=self.min_free_bytes+requested

    def get(self,row,version,function,source,matrix):
        document,key=key_document(row,version,function,source);data=self.path/(key+'.npz');meta=data.with_suffix('.json')
        with (self.path/'.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if data.exists() and meta.exists():
                record=json.loads(meta.read_text())
                if record['key']!=document or content(c.bind(data))!=record['array']:raise ValueError('Independent reference cache corruption')
                with np.load(data,allow_pickle=False) as archive:reference=archive['reference']
                if reference.shape!=(row['n'],):raise ValueError('Incomplete cached reference')
                data.touch();reference.flags.writeable=False
                return reference,{'key_sha256':key,'cache_hit':True,'file':c.bind(data),'independent':True,**record['solver']}
            started=time.perf_counter();reference,solver=build_reference(matrix,function,source)
            if reference.shape!=(row['n'],):raise ValueError('Incomplete CPU reference')
            metadata={'key':document,'solver':solver,'build_seconds':time.perf_counter()-started}
            # Reserve an uncompressed upper bound before writing. If disk cannot
            # hold it, keep correctness and record the uncached independent solve.
            if not self._make_space(reference.nbytes+(1<<20)):
                reference.flags.writeable=False
                return reference,{'key_sha256':key,'cache_hit':False,'stored':False,'reason':'reference_cache_disk_budget',
                                  'independent':True,'build_seconds':metadata['build_seconds'],**solver}
            fd,temporary=tempfile.mkstemp(prefix=key+'.',suffix='.partial',dir=self.path)
            try:
                with os.fdopen(fd,'wb') as stream:np.savez_compressed(stream,reference=reference)
                os.replace(temporary,data);metadata['array']=content(c.bind(data));c.save(meta,metadata)
            finally:
                if Path(temporary).exists():Path(temporary).unlink()
            reference.flags.writeable=False
            return reference,{'key_sha256':key,'cache_hit':False,'stored':True,'file':c.bind(data),'independent':True,
                              'build_seconds':metadata['build_seconds'],**solver}


def output_quality(values,row,version,function,source,cache=None):
    n=row['n']
    if values.shape!=(n,) or values.dtype!=np.float64 or values.flags.writeable or not values.flags.owndata or values.base is not None:
        raise ValueError('Incomplete or nonowning public result')
    matrix=matrix_for(row,version,function)
    if cache is None:
        reference,solver=build_reference(matrix,function,source);proof={'cache_hit':False,'stored':False,'independent':True,**solver}
    else:reference,proof=cache.get(row,version,function,source,matrix)
    if function in ('bfs','sssp'):
        np.testing.assert_array_equal(values,reference)
        return {'passed':True,'vertices':n,'exact':True,'unreachable':int(np.isposinf(values).sum()),'reference':proof}
    if function=='wcc':
        count=int(proof['components']);_,first=np.unique(reference,return_index=True);mapping=values[first]
        valid=np.all(np.isfinite(values)) and np.all(values==np.floor(values)) and len(np.unique(values))==count and np.all(values==mapping[reference])
        if not valid:raise ValueError('WCC partition differs')
        return {'passed':True,'vertices':n,'components':count,'partition_equivalent':True,'reference':proof}
    residual=float(np.abs(values-pr_step(matrix)(values)).sum());error=float(np.abs(values-reference).sum());mass=float(abs(values.sum()-1))
    if not np.all(np.isfinite(values)) or np.any(values<0) or residual>QUALITY['residual_l1'] or error>QUALITY['reference_l1'] or mass>QUALITY['mass_error']:
        raise ValueError('PR residual/reference/mass contract failed')
    return {'passed':True,'vertices':n,'residual_l1':residual,'reference_l1':error,'mass_error':mass,'reference':proof}
