from pathlib import Path
from unittest import mock
import errno,hashlib,json,os,tempfile,unittest
import numpy as np
import archive_outputs as a

class Recoverable(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);self.inputs=self.root/'inputs';self.store=self.root/'objects_store';folder=self.inputs/'canonical/TINY_U';folder.mkdir(parents=True)
  self.write(folder/'canonical.json',dict(num_nodes=3,num_entries=0,directed=False));self.write(folder/'sources.json',dict(primary_sources=[0,1,2]));(folder/'offsets.bin').write_bytes(np.zeros(4,dtype='<i4').tobytes());(folder/'indices.bin').write_bytes(b'');(folder/'weights.bin').write_bytes(b'')
  files={role:a.binding(folder/file) for role,file in [('canonical','canonical.json'),('sources','sources.json'),('offsets','offsets.bin'),('indices','indices.bin'),('weights','weights.bin')]};self.write(folder/'receipt.json',dict(status='PASS_CANONICAL_INPUT',files=files));input_sha,meta,canon=a.canonical_identity(self.inputs,'TINY_U')
  self.producer=self.root/'reference.py';self.producer.write_text('# independent CPU fixture');producer=a.binding(self.producer)
  self.sequence=[0,1];self.checks=[];self.arrays=[];self.archives=[]
  for i in self.sequence:
   array=np.full(3,np.inf,dtype='<f8');array[i]=0;self.arrays.append(array);p=self.root/('ref'+str(i)+'.npz');np.savez_compressed(p,reference=array);item=a.binding(p);self.archives.append(p)
   key=dict(schema='state.reference.v2',n=3,m=0,function='sssp',source=i,version=0,input={r:{k:files[r][k] for k in ['bytes','sha256']} for r in ['offsets','indices','weights']},reference_source={k:producer[k] for k in ['bytes','sha256']})
   self.write(p.with_suffix('.json'),dict(key=key,array={k:item[k] for k in ['bytes','sha256']}))
   self.checks.append(dict(sequence=i,source=i,values=3,output_sha256=hashlib.sha256(array.tobytes()).hexdigest(),quality=dict(passed=True,exact=True,vertices=3,reference=dict(independent=True,file=item,key_sha256=a.digest(key))),survives_session_close=True))
  ep=self.root/'epoch.json';self.write(ep,dict(variant='eggpu',fixture=True));self.result=self.root/'result.json';self.raw=dict(schema='e4s.complete.batch.v1',status='PASS_E4S_QUALIFICATION',phase='tiny',quality_pass=True,variant='eggpu',dataset='TINY_U',batch=2,input_sha256=input_sha,epoch=a.binding(ep),epoch_sha256=a.digest(a.load(ep)),source_sequence=self.sequence,checks=self.checks,measurements=dict(all_outputs_retained_through_close=True),reference_implementation=producer);self.write(self.result,self.raw);self.out=self.root/'recovery.json'
 def tearDown(self):self.t.cleanup()
 def write(self,p,x):Path(p).write_text(json.dumps(x,indent=2))
 def save_result(self):self.write(self.result,self.raw)
 def archive(self):return a.archive_result(self.result,self.inputs,self.store,self.out)
 def replace_archive(self,index,array):
  p=self.archives[index];np.savez_compressed(p,reference=array);item=a.binding(p);m=a.load(p.with_suffix('.json'));m['array']={k:item[k] for k in ['bytes','sha256']};self.write(p.with_suffix('.json'),m);self.raw['checks'][index]['quality']['reference']['file']=item;self.save_result()
 def test_full_hardlink_recovery_survives_reference_cache_unlink(self):
  r=self.archive();self.assertEqual(r['status'],'PASS_BYTE_IDENTICAL_RECOVERABLE_E4S_OUTPUTS');self.assertTrue(r['post_measurement']);self.assertFalse(r['gpu_executed'])
  for i,row in enumerate(r['vectors']):
   obj=Path(row['object']['path']);self.assertEqual(obj.stat().st_ino,self.archives[i].stat().st_ino);self.archives[i].unlink();self.archives[i].with_suffix('.json').unlink();a.checked(row['object'])
   with np.load(obj,allow_pickle=False) as z:restored=z['reference']
   self.assertEqual(hashlib.sha256(restored.tobytes()).hexdigest(),row['measured_output_sha256'])
 def test_two_jobs_reuse_same_objects_no_duplicate_vector_archive(self):
  first=self.archive();second=a.archive_result(self.result,self.inputs,self.store,self.root/'second.json');self.assertEqual([x['object'] for x in first['vectors']],[x['object'] for x in second['vectors']]);self.assertTrue(all(x['object_storage']=='existing_verified' for x in second['vectors']));self.assertEqual(len(list((self.store/'objects').glob('*/*.npz'))),2)
 def test_signed_zero_difference_rejected_despite_numeric_equality(self):
  arr=self.arrays[0].copy();arr[0]=-0.;self.assertTrue(np.array_equal(arr,self.arrays[0]));self.replace_archive(0,arr)
  with self.assertRaisesRegex(ValueError,'byte level'):self.archive()
  self.assertFalse(self.out.exists())
 def test_corrupt_archive_rejected_before_pin(self):
  self.archives[0].write_bytes(b'bad')
  with self.assertRaisesRegex(ValueError,'SHA or length'):self.archive()
  self.assertFalse(self.out.exists())
 def test_wrong_source_metadata_or_missing_stored_reference_rejected(self):
  self.raw['checks'][0]['quality']['reference']['independent']=False;self.save_result()
  with self.assertRaises(ValueError):self.archive()
  self.raw['checks'][0]['quality']['reference']['independent']=True;self.save_result();p=self.archives[0].with_suffix('.json');m=a.load(p);m['key']['source']=2;self.write(p,m)
  with self.assertRaisesRegex(ValueError,'exact graph/source'):self.archive()
 def test_dtype_and_shape_rejected_even_valid_archive_sha(self):
  self.replace_archive(0,self.arrays[0].astype('<f4'))
  with self.assertRaisesRegex(ValueError,'dtype/shape'):self.archive()
  self.replace_archive(0,np.zeros(2,dtype='<f8'))
  with self.assertRaisesRegex(ValueError,'dtype/shape'):self.archive()
 def test_exdev_only_fallback_copies_bounded_and_recovers(self):
  original=os.link;source_names={str(p) for p in self.archives}|{str(p.with_suffix('.json')) for p in self.archives}
  def cross(src,dst,**kw):
   if str(src) in source_names:raise OSError(errno.EXDEV,'cross device')
   return original(src,dst,**kw)
  with mock.patch.object(a.os,'link',side_effect=cross):r=self.archive()
  self.assertTrue(all(x['object_storage']=='copy_exdev' for x in r['vectors']));self.assertNotEqual(Path(r['vectors'][0]['object']['path']).stat().st_ino,self.archives[0].stat().st_ino)
 def test_permission_error_does_not_fall_back_or_publish_pass(self):
  with mock.patch.object(a.os,'link',side_effect=PermissionError(errno.EACCES,'denied')):
   with self.assertRaises(PermissionError):self.archive()
  self.assertFalse(self.out.exists())
 def test_fsync_error_does_not_publish_pass(self):
  with mock.patch.object(a.os,'fsync',side_effect=OSError(errno.EIO,'io failure')):
   with self.assertRaises(OSError):self.archive()
  self.assertFalse(self.out.exists())
 def test_existing_corrupted_object_is_never_overwritten(self):
  item=self.raw['checks'][0]['quality']['reference']['file'];obj=self.store/'objects'/item['sha256'][:2]/(item['sha256']+'.npz');obj.parent.mkdir(parents=True);obj.write_bytes(b'bad')
  with self.assertRaises(ValueError):self.archive()
  self.assertEqual(obj.read_bytes(),b'bad');self.assertFalse(self.out.exists())
 def test_pin_open_fd_rejects_replaced_source_path(self):
  p=self.archives[0];item=a.binding(p);fd=a.regular_fd(p)
  try:
   p.unlink();p.write_bytes(b'substitute')
   with self.assertRaisesRegex(ValueError,'pathname changed'):a.pin_fd(p,fd,item,self.root/'pin.npz')
  finally:os.close(fd)
  self.assertFalse((self.root/'pin.npz').exists())
 def test_original_result_is_preserved_and_second_sidecar_write_rejected(self):
  before=self.result.read_bytes();self.archive();self.assertEqual(before,self.result.read_bytes())
  with self.assertRaisesRegex(ValueError,'overwrite'):self.archive()
 def test_partial_or_wrong_epoch_is_not_recoverable_pass(self):
  self.raw['checks'].pop();self.save_result()
  with self.assertRaisesRegex(ValueError,'batch'):self.archive()
  self.raw['checks']=self.checks;self.raw['epoch_sha256']='wrong';self.save_result()
  with self.assertRaisesRegex(ValueError,'epoch'):self.archive()

if __name__=='__main__':unittest.main(verbosity=2)
