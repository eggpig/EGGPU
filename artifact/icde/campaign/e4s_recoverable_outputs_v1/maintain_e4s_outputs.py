"""Mandatory post-timing recovery audit followed by the original maintenance.
A failure leaves the original worker receipt unchanged and never becomes a
formal usable sample. One new immutable wrapper receipt per attempt.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time
import archive_outputs as archive


def run(spec_path,output):
 spec=archive.load(spec_path);output=Path(output);archive.require(not output.exists(),'Maintenance attempt receipt already exists; no implicit retry')
 raw=archive.load(spec['result']);raw_binding=archive.binding(spec['result']);state=dict(schema='e4s.output.maintenance.v1',status='FAILED_E4S_OUTPUT_MAINTENANCE',original_result=raw_binding,source=archive.binding(__file__),archive_status='NOT_STARTED',original_maintenance_status='NOT_STARTED',worker_status=raw.get('status'),gpu_executed=False,post_measurement=True,attempt_spec=archive.binding(spec_path),started_unix=time.time())
 try:
  for item in spec['bindings']:archive.checked(item)
  if spec.get('recovery_of'):
   prior=archive.load(archive.checked(spec['recovery_of']));archive.require(prior.get('status')=='FAILED_E4S_OUTPUT_MAINTENANCE' and prior.get('original_result')==raw_binding,'Explicit recovery must bind prior failed postprocessing of this exact worker');state['recovery_of']=spec['recovery_of']
  worker_passed=raw.get('status') in ['PASS_E4S_QUALIFICATION','PASS_E4S_FORMAL']
  sidecar=Path(spec['archive_receipt'])
  if not worker_passed:state['archive_status']='NOT_APPLICABLE_WORKER_FAILED'
  elif sidecar.exists():
   archive.require(spec.get('recovery_of') is not None,'Existing per-job sidecar requires an explicit bound recovery attempt')
   # Explicit maintenance-only recovery may reuse a fully checked prior pin.
   archive.validate_sidecar(spec['result'],sidecar,spec['archive_helper'])
   fd=archive.regular_fd(sidecar)
   try:os.fsync(fd)
   finally:os.close(fd)
   archive.fsync_dir(sidecar.parent)
  else:archive.archive_result(spec['result'],spec['input_root'],spec['store'],sidecar)
  if worker_passed:state.update(archive_status='PASS',archive_receipt=archive.binding(sidecar))
  original=spec['original_maintenance']
  archive.require(original['command_sha256']==hashlib.sha256(json.dumps(original['command'],separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),'Original maintenance command digest differs')
  for item in original['bindings']:archive.checked(item)
  env={**os.environ,**original.get('environment',{}),'CUDA_VISIBLE_DEVICES':''};run=subprocess.run(original['command'],env=env,timeout=original.get('timeout',3600),check=False);state['original_maintenance_returncode']=run.returncode
  archive.require(run.returncode==0,'Original maintenance subprocess failed');receipt=archive.load(original['receipt']);archive.require(receipt['status'] in original['expected_statuses'],'Original maintenance receipt failed')
  state.update(original_maintenance_status='PASS',original_maintenance_receipt=archive.binding(original['receipt']),status='PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE' if worker_passed else 'PASS_FAILED_WORKER_CLEANUP')
 except BaseException as error:
  state['error']=type(error).__name__+': '+str(error)
  if state['archive_status']=='NOT_STARTED':state['archive_status']='FAIL'
  else:state['original_maintenance_status']='FAIL'
  raise
 finally:
  state['finished_unix']=time.time();archive.immutable_json(output,state)
 return state

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();r=run(args.spec,args.output);print(json.dumps({'status':r['status'],'gpu_executed':False}))
