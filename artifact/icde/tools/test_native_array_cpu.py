"""CPU-only exact file/array import equivalence and invalid-input contracts."""
from pathlib import Path
import copy,hashlib,json,os,sys,time
import numpy as np

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-dso',type=Path,required=True)
parser.add_argument('--build-receipt',type=Path,required=True)
parser.add_argument('--work',type=Path,required=True)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[3]
args.native_dso=args.native_dso.resolve();out=args.work.resolve()
freeze=json.loads((ROOT/'artifact/icde/provenance/runtime_source_freeze.json').read_text())
build=json.loads(args.build_receipt.read_text())
source_sha=build.get('source_tree_sha256') or build.get('source_sha256')
if source_sha!=freeze['source_tree_sha256'] or build.get('status') not in ['PASS','PASS_BUILD_GPU_UNQUALIFIED']:
 raise ValueError('Build receipt must bind the exact current source')
expected=build.get('native_dso')
if expected is None:
 candidates=[v for v in build.get('outputs',[]) if Path(v['path']).name==args.native_dso.name]
 if len(candidates)!=1:raise ValueError('Exactly one matching native extension build output required')
 expected=candidates[0]
if hashlib.sha256(args.native_dso.read_bytes()).hexdigest()!=expected['sha256']:
 raise ValueError('Native extension differs from its explicit build receipt')
out.mkdir(parents=True,exist_ok=False)
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[str(args.native_dso.parent),str(ROOT/'Easy-Graph')]
import cpp_easygraph as cpp
assert Path(cpp.__file__).resolve()==args.native_dso
tests=[]
def projection(n,edges):
 pairs=np.asarray(sorted({(min(u,v),max(u,v)) for u,v in edges if u!=v}),dtype=np.int64).reshape(-1,2)
 degree=np.bincount(pairs.ravel(),minlength=n)
 forward=pairs.copy();swap=(degree[pairs[:,0]]>degree[pairs[:,1]])|((degree[pairs[:,0]]==degree[pairs[:,1]])&(pairs[:,0]>pairs[:,1]));forward[swap]=forward[swap,::-1]
 p={'unique_edge_count':len(pairs),'degree':degree.astype(np.int32)}
 for key,rows in [('lower',pairs),('upper',pairs[:,::-1]),('forward',forward)]:
  rows=rows[np.lexsort((rows[:,1],rows[:,0]))];p[key+'_V']=np.r_[0,np.cumsum(np.bincount(rows[:,0],minlength=n))].astype(np.int32);p[key+'_E']=rows[:,1].astype(np.int32)
 return p
def fixture(n,edges,weighted=False):
 edges=sorted(edges);pairs=np.asarray(edges,dtype=np.int64).reshape(-1,2)
 return dict(offsets=np.r_[0,np.cumsum(np.bincount(pairs[:,0],minlength=n))].astype(np.int32),indices=pairs[:,1].astype(np.int32),weights=(np.arange(len(edges),dtype=np.float64)+0.5 if weighted else None),undirected_projection=projection(n,edges))
def files(d,name,validate=True):
 q=out/name;q.mkdir();v=d['offsets'];e=d['indices'];v.tofile(q/'V');e.tofile(q/'E');p=d.get('undirected_projection');fp=None
 if p is not None:
  fp={'unique_edge_count':p['unique_edge_count']}
  for key in ['lower_V','lower_E','upper_V','upper_E','degree','forward_V','forward_E']:
   if key in p:p[key].tofile(q/key);fp[key+'_path']=str(q/key)
  if 'weights' in p:p['weights'].tofile(q/'PW');fp['weights_path']=str(q/'PW');fp['weight_key']=p.get('weight_key','weight')
 weights=d.get('weights');wp=''
 if weights is not None:weights.tofile(q/'W');wp=str(q/'W')
 return cpp.cpp_graph_from_csr_files(str(q/'V'),str(q/'E'),len(v)-1,len(e),d.get('directed',True),validate,wp,d.get('weight_key','weight'),fp)
def arrays(d,validate=True):return cpp.cpp_graph_from_csr_arrays(d['offsets'],d['indices'],d.get('directed',True),validate,d.get('weights'),d.get('weight_key','weight'),d.get('undirected_projection'))
def equal(a,b):
 if isinstance(a,np.ndarray):assert a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 else:assert a==b,(a,b)
def check(name,d):
 a=cpp.cpp_graph_csr_arrays(arrays(d));b=cpp.cpp_graph_csr_arrays(files(d,name));equal(a,b)
 assert a['num_nodes']==len(d['offsets'])-1
 tests.append({'name':name,'status':'PASS','complete_arrays_compared':True})
 return a
for n,edges,weighted,name in [(0,[],False,'empty'),(1,[],False,'isolated'),(6,[(0,1),(1,0),(1,2),(2,1)],False,'isolates_and_asymmetry'),(4,[(0,1),(0,2),(1,2),(2,3),(3,1)],True,'weighted'),(4,[(0,0),(0,1),(0,1),(2,3)],False,'duplicates_and_selfloops')]:
 d=fixture(n,edges,weighted);check(name,d)
rng=np.random.default_rng(789)
for i in range(6):
 n=31+i;edges={(int(u),int(v)) for u,v in rng.integers(0,n,size=(100,2)) if u!=v};check('random_'+str(i),fixture(n,edges,i%2==0))
d=fixture(8,[(0,1),(1,0),(0,3),(3,0),(3,4)],True);native=arrays(d);before=cpp.cpp_graph_csr_arrays(native)
for value in [d['offsets'],d['indices'],d['weights'],*d['undirected_projection'].values()]:
 if isinstance(value,np.ndarray):value[:]=-1
equal(before,cpp.cpp_graph_csr_arrays(native))
snapshot=cpp.cpp_graph_csr_arrays(native);snapshot['offsets'][:]=-1;snapshot['undirected_projection']['degree'][:]=-1;equal(before,cpp.cpp_graph_csr_arrays(native));tests.append(dict(name='import_export_independent_ownership',status='PASS'))
base=fixture(5,[(0,1),(0,3),(1,2),(2,4)],True)
def error(call):
 try:call()
 except (TypeError,ValueError) as e:return type(e).__name__,str(e)
 raise AssertionError('invalid input was accepted')
def reject(name,mutate,same_file_error=True):
 d=copy.deepcopy(base);mutate(d);a=error(lambda:arrays(d))
 if same_file_error:
  b=error(lambda:files(d,'bad_'+name));assert a==b,(name,a,b)
 tests.append(dict(name=name,status='PASS',array_error=a,file_contract_exact=same_file_error))
reject('first_offset',lambda d:d['offsets'].__setitem__(0,1))
reject('last_offset',lambda d:d['offsets'].__setitem__(-1,3))
reject('nonmonotonic',lambda d:d['offsets'].__setitem__(2,0))
reject('endpoint_range',lambda d:d['indices'].__setitem__(0,99))
reject('nonfinite_weights',lambda d:d['weights'].__setitem__(0,np.inf))
reject('projection_degree',lambda d:d['undirected_projection']['degree'].__setitem__(0,0))
reject('projection_endpoint',lambda d:d['undirected_projection']['upper_E'].__setitem__(0,99))
reject('projection_orientation',lambda d:d['undirected_projection']['lower_E'].__setitem__(0,0))
reject('projection_forward',lambda d:d['undirected_projection']['forward_E'].__setitem__(0,4))
reject('int64_offsets',lambda d:d.update(offsets=d['offsets'].astype(np.int64)),False)
reject('big_endian_offsets',lambda d:d.update(offsets=d['offsets'].astype('>i4')),False)
reject('strided_indices',lambda d:d.update(indices=np.repeat(d['indices'],2)[::2]),False)
reject('matrix_offsets',lambda d:d.update(offsets=d['offsets'].reshape(2,3)),False)
reject('python_list',lambda d:d.update(indices=d['indices'].tolist()),False)
reject('float32_weights',lambda d:d.update(weights=d['weights'].astype(np.float32)),False)
reject('short_weights',lambda d:d.update(weights=d['weights'][:-1]),False)
reject('missing_projection',lambda d:d['undirected_projection'].pop('degree'),False)
reject('projection_dtype',lambda d:d['undirected_projection'].update(degree=d['undirected_projection']['degree'].astype(np.int64)),False)
reject('projection_count',lambda d:d['undirected_projection'].update(unique_edge_count=-1))
# Endpoint bounds are always checked even when deep validation is explicitly off.
d=copy.deepcopy(base);d['offsets'][-1]=0
assert error(lambda:arrays(d,False))==error(lambda:files(d,'unchecked_invalid_endpoint',False));tests.append(dict(name='validate_false_endpoint_guard',status='PASS'))
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
receipt=dict(status='PASS_CPU_FILE_ARRAY_EQUIVALENCE',gpu_executed=False,cases=len(tests),tests=tests,source_freeze=bind(ROOT/'artifact/icde/provenance/runtime_source_freeze.json'),build=bind(args.build_receipt),dso=bind(Path(cpp.__file__)),test=bind(Path(__file__)))
(out/'array_cpu_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'cases':len(tests),'receipt':bind(out/'array_cpu_receipt.json')}))
