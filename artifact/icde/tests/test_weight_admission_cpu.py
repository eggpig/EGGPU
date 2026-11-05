"""Read-only CPU contract comparison to frozen 447c admission; no CUDA imports."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,tempfile,types,unittest
from unittest import mock
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE/'fixtures/weight_admission_447c'
NEW=HERE.parents[2]/'Easy-Graph/easygraph/gpu'
def module_file(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
signer=module_file('original_weight_signer',OLD/'_manifest_signatures.py')
helper=module_file('candidate_weight_scan',NEW/'_weight_admission.py')
def admission(root,kind):
 path=root/('_native.py' if kind=='static' else '_dynamic.py');tree=ast.parse(path.read_text());names={'_sha','sha','verify_manifest'}
 tree.body=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in names]
 scope=dict(Path=Path,np=np,hashlib=hashlib,sign=signer.sign,scan_weight_payload=helper.scan_weight_payload)
 exec(compile(tree,str(path),'exec'),scope);return scope['verify_manifest']
FUNCTIONS={(root,kind):admission(root,kind) for root in (OLD,NEW) for kind in ('static','dynamic')}

class Admission(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='weight_admission_',dir=os.environ['EGGPU_TEST_SCRATCH']);self.base=Path(self.temp.name);self.serial=0
 def tearDown(self):self.temp.cleanup()
 def fixture(self,weights,n=4,dtype='<f8',parts=1):
  folder=self.base/str(self.serial);self.serial+=1;folder.mkdir();rows=[]
  for rank in range(parts):
   payload=folder/f'p{rank}.weights.bin';values=np.asarray(weights,dtype=dtype);values.tofile(payload)
   rows.append(dict(partition_id=rank,owned_vertex_count=n if rank==0 else 0,halo_vertex_count=0,adjacency_count=len(values),cut_edge_count=0,ownership=dict(encoding='range',vertex_begin=0,vertex_end=n if rank==0 else 0),files={'weights':{'path':payload.name,'dtype':{'<f8':'float64','<i4':'int32','<u8':'uint64'}[dtype],'count':len(values),'byte_size':payload.stat().st_size,'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}}))
  d={'format':'egmgpu-partitioned-csr','version':1,'graph':dict(id='contract',generation=1,directed=True,weighted=True,vertex_count=n,edge_count=len(weights)*parts,adjacency_count=len(weights)*parts),'view':dict(orientation='outgoing',weight_key='weight'),'partitioning':dict(ownership_encoding='range'),'partitions':rows}
  self.resign(d);p=folder/'manifest.json';p.write_text(json.dumps(d));return p,d
 def resign(self,d):
  t,w,v=signer.sign(d);d['graph']['topology_sha256']=t;d['graph']['weight_sha256']=w;d['view']['signature']=v
 def outcome(self,root,kind,p,d,compact):
  certificate={} if compact else None
  try:
   result=FUNCTIONS[root,kind](p,d,**({'numeric_certificate':certificate} if kind=='static' else {}));return ('PASS',result,certificate)
  except Exception as e:return ('ERROR',type(e).__name__,str(e),certificate)
 def parity(self,p,d,compact=False,expected=None):
  for kind in ('static','dynamic'):
   old=self.outcome(OLD,kind,p,d,compact);new=self.outcome(NEW,kind,p,d,compact);self.assertEqual(old,new,(kind,old,new))
   if expected is not None and (kind=='static' or not compact):self.assertEqual(new[0],expected)
  return self.outcome(NEW,'static',p,d,compact)
 def test_valid_empty_zero_negativezero_fractional_and_multipart(self):
  for values in ([],[0.0,-0.0,1,16,255],[0.25,2.5],[1e100],[np.finfo(np.float64).tiny]):
   for parts in (1,2,8):
    p,d=self.fixture(values,parts=parts);self.parity(p,d,expected='PASS');self.parity(p,d,compact=True)
 def test_invalid_values_and_error_priority_by_original_chunk(self):
  for value in (np.nan,np.inf,-np.inf,-1.0,-np.finfo(float).tiny,255.5,256.0):
   p,d=self.fixture([1,value,2]);self.parity(p,d);self.parity(p,d,compact=True,expected='ERROR')
  values=np.ones((1<<18)+8);values[0]=256;values[-1]=np.nan;p,d=self.fixture(values);r=self.parity(p,d,compact=True,expected='ERROR');self.assertIn('exact integer',r[2])
  values[0]=np.nan;values[-1]=256;p,d=self.fixture(values);r=self.parity(p,d,compact=True,expected='ERROR');self.assertIn('finite nonnegative',r[2])
 def test_complete_hash_failure_precedes_nan_or_range_rejection(self):
  for value in (np.nan,256.0,-1.0):
   values=np.ones((1<<18)+7);values[0]=value;p,d=self.fixture(values);payload=p.parent/d['partitions'][0]['files']['weights']['path']
   with payload.open('r+b') as stream:stream.seek(-1,2);b=stream.read(1);stream.seek(-1,2);stream.write(bytes([b[0]^1]))
   r=self.parity(p,d,compact=True,expected='ERROR');self.assertEqual(r[2],'Snapshot payload identity mismatch')
 def test_shape_dtype_missing_or_hash_corruption_and_path_escape(self):
  for change in ('bytes','count','sha','dtype','owned','adjacency','weighted','vertex','escape'):
   p,d=self.fixture([1,2]);part=d['partitions'][0];item=part['files']['weights']
   if change=='bytes':item['byte_size']+=8
   elif change=='count':item['count']+=1
   elif change=='sha':item['sha256']='0'*64
   elif change=='dtype':item['dtype']='float32'
   elif change=='owned':part['owned_vertex_count']+=1
   elif change=='adjacency':part['adjacency_count']+=1
   elif change=='weighted':d['graph']['weighted']=False
   elif change=='vertex':d['graph']['vertex_count']=0
   elif change=='escape':item['path']='../../outside.bin'
   self.parity(p,d);self.parity(p,d,compact=True,expected='ERROR')
  for dtype in ('<i4','<u8'):
   for values in ([1],[1,2],[]):
    p,d=self.fixture(values,dtype=dtype);self.parity(p,d);self.parity(p,d,compact=True,expected='ERROR')
 def test_exact_u32_sentinel_distance_bounds_and_certificate(self):
  for n,w,accepted in [(16843009,255,True),(16843010,255,False),(2**32-1,1,True),(2**32,1,False),(2**31-1,2,True),(2**31,2,True),(2**31+1,2,False)]:
   p,d=self.fixture([w],n=n);r=self.parity(p,d,compact=True,expected='PASS' if accepted else 'ERROR')
   if accepted:self.assertEqual(r[2]['finite_distance_upper_bound'],(n-1)*w)
 def test_no_mmap_or_second_scan_on_valid_fp64_public_admission(self):
  p,d=self.fixture([0,1,255])
  with mock.patch.object(np,'memmap',side_effect=AssertionError('second weight scan')):
   for kind in ('static','dynamic'):
    old=self.outcome(OLD,kind,p,d,True);new=self.outcome(NEW,kind,p,d,True)
    self.assertEqual(old[0],'ERROR');self.assertEqual(new[0],'PASS')
 def test_stream_is_bounded_and_hashes_all_bytes_despite_early_nan(self):
  values=np.ones((1<<18)*3+19);values[0]=np.nan;p,d=self.fixture(values);payload=p.parent/'p0.weights.bin';sizes=[];readbytes=[];original=Path.open
  class Stream:
   def __init__(self,inner):self.inner=inner
   def __enter__(self):self.inner.__enter__();return self
   def __exit__(self,*args):return self.inner.__exit__(*args)
   def read(self,size):
    sizes.append(size);data=self.inner.read(size);readbytes.append(len(data));return data
  def opened(path,*args,**kwargs):return Stream(original(path,*args,**kwargs))
  with mock.patch.object(helper.Path,'open',opened):digest,error,maximum,count=helper.scan_weight_payload(payload,certify=True)
  self.assertEqual(digest,d['partitions'][0]['files']['weights']['sha256']);self.assertEqual(error,'Weights must be finite nonnegative f64');self.assertEqual(sum(readbytes),payload.stat().st_size);self.assertTrue(all(0<size<=2**21 for size in sizes))
 def test_no_trusted_cache_accepts_replacement_after_admission(self):
  p,d=self.fixture([1,2,3]);self.parity(p,d,compact=True,expected='PASS');payload=p.parent/'p0.weights.bin';np.array([1,2,4],dtype='<f8').tofile(payload);self.parity(p,d,compact=True,expected='ERROR')
 def test_dynamic_content_signature_validation_is_not_bypassed(self):
  p,d=self.fixture([1,2,3]);d['graph']['weight_sha256']='0'*64
  for root in (OLD,NEW):self.assertEqual(self.outcome(root,'dynamic',p,d,False)[2],'Manifest content signatures disagree')
 def test_producer_signatures_do_not_bind_python_implementation(self):
  p,d=self.fixture([1,2,255]);before=copy.deepcopy(d);self.parity(p,d,compact=True,expected='PASS');self.assertEqual(d,before);self.assertEqual(signer.sign(d),(d['graph']['topology_sha256'],d['graph']['weight_sha256'],d['view']['signature']))

if __name__=='__main__':unittest.main(verbosity=2)
