"""Post-timing E4S lossless recovery bindings, never synthetic measured outputs.

An independently produced reference may represent a measured output only after
its decoded float64 bytes SHA equals the measured full-vector SHA. Pinned NPZ
objects survive reference-cache eviction. No worker or epoch is modified.
"""
from pathlib import Path
import argparse,errno,fcntl,hashlib,json,os,shutil,stat,tempfile
import numpy as np

CHUNK=8<<20

def require(ok,message):
 if not ok:raise ValueError(message)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def fsync_dir(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def regular_fd(path):
 fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
 if not stat.S_ISREG(os.fstat(fd).st_mode):os.close(fd);raise ValueError('Expected regular archive file')
 return fd

def fd_sha(fd):
 os.lseek(fd,0,os.SEEK_SET);h=hashlib.sha256();n=0
 while True:
  block=os.read(fd,CHUNK)
  if not block:break
  n+=len(block);h.update(block)
 return dict(bytes=n,sha256=h.hexdigest())

def binding(path):
 path=Path(path).absolute();fd=regular_fd(path)
 try:return dict(path=str(path),**fd_sha(fd))
 finally:os.close(fd)
def read_bound_json(path):
 path=Path(path).absolute();fd=regular_fd(path)
 try:
  data=bytearray()
  while True:
   chunk=os.read(fd,CHUNK)
   if not chunk:break
   data.extend(chunk);require(len(data)<=128<<20,'JSON metadata unexpectedly exceeds bounded size')
  return json.loads(data),dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 finally:os.close(fd)

def checked(item):
 require(binding(item['path'])=={k:item[k] for k in ['path','bytes','sha256']},'File binding changed: '+item['path']);return Path(item['path'])

def immutable_json(path,value):
 """Atomically publish an exclusive, durable JSON object; never overwrite."""
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,temp=tempfile.mkstemp(prefix='.publish-',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.link(temp,path);fsync_dir(path.parent)
 finally:
  Path(temp).unlink(missing_ok=True)
 return binding(path)

def pin_fd(source,fd,item,destination):
 """Same-FS hardlink; EXDEV-only bounded copy. All other errors fail closed."""
 destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.exists():
  checked(dict(path=str(destination),bytes=item['bytes'],sha256=item['sha256']));existing=regular_fd(destination)
  try:os.fsync(existing)
  finally:os.close(existing)
  fsync_dir(destination.parent);return 'existing_verified'
 handle,tmp=tempfile.mkstemp(prefix='.pin-',dir=destination.parent);os.close(handle);Path(tmp).unlink();mode='hardlink'
 try:
  try:
   os.link(source,tmp,follow_symlinks=False)
   a,b=os.stat(tmp,follow_symlinks=False),os.fstat(fd);require(stat.S_ISREG(a.st_mode) and (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino),'Reference pathname changed after verified fd was opened')
   os.fsync(fd)
  except OSError as error:
   if error.errno!=errno.EXDEV:raise
   mode='copy_exdev';require(shutil.disk_usage(destination.parent).free>=item['bytes']+(2<<30),'Cross-device archive copy lacks required disk reserve');out=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC,0o600)
   try:
    os.lseek(fd,0,os.SEEK_SET)
    while True:
     block=os.read(fd,CHUNK)
     if not block:break
     view=memoryview(block)
     while view:
      count=os.write(out,view);require(count>0,'Short archive copy write');view=view[count:]
    os.fsync(out)
   finally:os.close(out)
  checked(dict(path=tmp,bytes=item['bytes'],sha256=item['sha256']))
  try:os.link(tmp,destination)
  except FileExistsError:checked(dict(path=str(destination),bytes=item['bytes'],sha256=item['sha256']));mode='raced_existing_verified'
  fsync_dir(destination.parent)
 finally:Path(tmp).unlink(missing_ok=True)
 return mode

def canonical_identity(root,dataset):
 folder=Path(root)/'canonical'/dataset;receipt=load(folder/'receipt.json');require(receipt['status']=='PASS_CANONICAL_INPUT','Canonical input not qualified')
 for role,name in [('canonical','canonical.json'),('sources','sources.json')]:
  actual=binding(folder/name);require(all(actual[k]==receipt['files'][role][k] for k in ['bytes','sha256']),'Canonical metadata changed')
 meta=load(folder/'canonical.json');sources=load(folder/'sources.json')['primary_sources'];r=binding(folder/'receipt.json')
 identity=dict(receipt={k:r[k] for k in ['bytes','sha256']},files={role:{k:i[k] for k in ['bytes','sha256']} for role,i in receipt['files'].items()},sources=sources,n=meta['num_nodes'],m=meta['num_entries'],directed=meta['directed'])
 return digest(identity),meta,receipt

def certify_one(row,source,meta,canonical,producer,store):
 q=row['quality'];require(q.get('passed') is True and q.get('exact') is True and q.get('vertices')==meta['num_nodes'],'Missing exact complete-vector audit')
 ref=q['reference'];require(ref.get('independent') is True and 'file' in ref,'Recoverable independent reference is not stored')
 item=ref['file'];metadata_path=Path(item['path']).with_suffix('.json');metadata,metadata_binding=read_bound_json(metadata_path);key=metadata['key'];expected_input={r:{k:canonical['files'][r][k] for k in ['bytes','sha256']} for r in ['offsets','indices','weights']}
 require(digest(key)==ref['key_sha256'] and key.get('schema')=='state.reference.v2' and key.get('function')=='sssp' and key.get('version')==0 and key.get('source')==source and key.get('n')==meta['num_nodes'] and key.get('m')==meta['num_entries'] and key.get('input')==expected_input,'Reference does not describe this exact graph/source task')
 require(key['reference_source']=={k:producer[k] for k in ['bytes','sha256']},'Reference producer source differs')
 require(metadata['array']=={k:item[k] for k in ['bytes','sha256']},'Reference metadata does not bind archive')
 fd=regular_fd(item['path'])
 try:
  require(fd_sha(fd)=={k:item[k] for k in ['bytes','sha256']},'Reference archive SHA or length differs')
  os.lseek(fd,0,os.SEEK_SET)
  with os.fdopen(os.dup(fd),'rb') as stream:
   with np.load(stream,allow_pickle=False) as archive:
    require(archive.files==['reference'],'Unexpected reference archive members');array=archive['reference']
  require(array.dtype==np.dtype('<f8') and array.shape==(meta['num_nodes'],) and array.flags.c_contiguous,'Wrong recoverable vector dtype/shape/order')
  raw=memoryview(array).cast('B');h=hashlib.sha256()
  for start in range(0,len(raw),CHUNK):h.update(raw[start:start+CHUNK])
  raw_sha=h.hexdigest();require(raw_sha==row['output_sha256'],'Measured output and reference differ at byte level (+0/-0 included)');del raw,array
  # Recheck the same opened fd after decoding; pathname races cannot substitute bytes.
  require(fd_sha(fd)=={k:item[k] for k in ['bytes','sha256']},'Reference archive changed during decode')
  dst=Path(store)/'objects'/item['sha256'][:2]/(item['sha256']+'.npz');mode=pin_fd(item['path'],fd,item,dst)
 finally:os.close(fd)
 metadata_dst=Path(store)/'metadata'/metadata_binding['sha256'][:2]/(metadata_binding['sha256']+'.json');mfd=regular_fd(metadata_path)
 try:
  require(fd_sha(mfd)=={k:metadata_binding[k] for k in ['bytes','sha256']},'Reference metadata changed');metadata_mode=pin_fd(metadata_path,mfd,metadata_binding,metadata_dst)
 finally:os.close(mfd)
 return dict(sequence=row['sequence'],source=source,values=meta['num_nodes'],dtype='<f8',order='C',measured_output_sha256=row['output_sha256'],restored_raw_sha256=raw_sha,byte_identical=True,object=binding(dst),reference_metadata=binding(metadata_dst),reference_key_sha256=ref['key_sha256'],object_storage=mode,metadata_storage=metadata_mode)

def archive_result(result_path,input_root,store,output):
 result_path=Path(result_path).absolute();output=Path(output).absolute();source_binding=binding(__file__);r,raw_binding=read_bound_json(result_path)
 require(r.get('status') in ['PASS_E4S_QUALIFICATION','PASS_E4S_FORMAL'] and r.get('quality_pass') is True,'Only successful complete E4S jobs may be archived')
 require(r.get('schema')=='e4s.complete.batch.v1' and r.get('phase') in ['tiny','target','formal'],'Unsupported E4S result schema')
 require(not output.exists(),'Refusing to overwrite recovery sidecar')
 input_sha,meta,canonical=canonical_identity(input_root,r['dataset']);require(input_sha==r['input_sha256'],'Canonical input differs from measured job')
 epoch=load(checked(r['epoch']));require(digest(epoch)==r['epoch_sha256'],'Measured epoch binding differs');producer=r['reference_implementation'];checked(producer)
 sequence=r['source_sequence'];checks=r['checks'];require(len(checks)==len(sequence)==r['batch'] and len(set(sequence))==len(sequence),'Missing or repeated distinct-source batch')
 require(all(row.get('sequence')==i and row.get('source')==source and row.get('values')==meta['num_nodes'] for i,(row,source) in enumerate(zip(checks,sequence))),'Measured source/shape sequence differs')
 if r['variant']=='eggpu':require(r['measurements'].get('all_outputs_retained_through_close') is True and all(row.get('survives_session_close') is True for row in checks),'Measured EGGPU output lifetime proof missing')
 store=Path(store).absolute();store.mkdir(parents=True,exist_ok=True);rows=[]
 with (store/'.archive.lock').open('a+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for row,source in zip(checks,sequence):rows.append(certify_one(row,source,meta,canonical,producer,store))
  require(binding(result_path)==raw_binding,'Measured result changed during archive audit');require(binding(__file__)==source_binding,'Recovery helper source changed during audit')
  document=dict(schema='e4s.recoverable.output.v1',status='PASS_BYTE_IDENTICAL_RECOVERABLE_E4S_OUTPUTS',original_result=raw_binding,epoch=r['epoch'],epoch_sha256=r['epoch_sha256'],input_sha256=input_sha,dataset=r['dataset'],variant=r['variant'],phase=r['phase'],batch=r['batch'],source_sequence=sequence,vectors=rows,source=source_binding,gpu_executed=False,post_measurement=True,measured_intervals_unchanged=True,retention='Pinned immutable-by-contract content-addressed NPZ and metadata; cache eviction does not remove the pinned hardlink',restore='np.load(object.path, allow_pickle=False)[reference]; verify full archive SHA, dtype/shape, then raw float64 bytes SHA equals measured_output_sha256')
  immutable_json(output,document)
 return document


def validate_sidecar(result_path,sidecar_path,expected_helper,*,require_formal=False):
 r,raw=read_bound_json(result_path);p,pb=read_bound_json(sidecar_path);checked(expected_helper)
 require(p.get('status')=='PASS_BYTE_IDENTICAL_RECOVERABLE_E4S_OUTPUTS' and p.get('schema')=='e4s.recoverable.output.v1' and p.get('original_result')==raw,'Recovery sidecar does not bind exact measured result')
 require(p.get('source')==expected_helper and p.get('post_measurement') is True and p.get('measured_intervals_unchanged') is True,'Recovery producer/measurement boundary differs')
 if require_formal:require(r.get('status')=='PASS_E4S_FORMAL' and r.get('phase')=='formal','Qualification cannot be exported as formal')
 require(r.get('quality_pass') is True and p.get('epoch_sha256')==r.get('epoch_sha256') and p.get('input_sha256')==r.get('input_sha256'),'Wrong recovery epoch/input/quality')
 require(p['source_sequence']==r['source_sequence'] and len(p['vectors'])==len(r['checks'])==r['batch'],'Recovery omitted a source/vector')
 for i,(v,c) in enumerate(zip(p['vectors'],r['checks'])):
  require(v['sequence']==c['sequence']==i and v['source']==c['source']==r['source_sequence'][i] and v['values']==c['values'],'Recovery vector identity differs')
  require(v.get('byte_identical') is True and v['restored_raw_sha256']==v['measured_output_sha256']==c['output_sha256'] and v.get('dtype')=='<f8' and v.get('order')=='C','Recovery is not byte-identical measured FP64 output')
  checked(v['object']);checked(v['reference_metadata'])
 return dict(status='PASS_VERIFIED_RECOVERABLE_E4S_OUTPUTS',sidecar=pb,original_result=raw,vectors=len(p['vectors']))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--store',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();r=archive_result(args.result,args.input_root,args.store,args.output);print(json.dumps(dict(status=r['status'],vectors=len(r['vectors']),gpu_executed=False)))
