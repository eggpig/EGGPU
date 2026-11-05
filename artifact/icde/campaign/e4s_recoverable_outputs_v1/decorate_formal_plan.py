"""Attach mandatory E4S postprocessing without modifying measured worker argv."""
from pathlib import Path
import copy,hashlib,json
import archive_outputs as archive

def command_digest(command):return hashlib.sha256(json.dumps(command,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def decorate(plan,spec,output):
 result=copy.deepcopy(plan);output=Path(output);result['postprocess_contract']='e4s_recoverable_outputs_v1';result['required_controller']=spec['controller'];result['archive_contract']=spec
 for item in [spec[k] for k in ['helper','wrapper','exporter','controller','supervisor']]:archive.checked(item)
 for job in result['jobs']:
  archive.require(job.get('phase')=='formal' and job['expected_statuses']==['PASS_E4S_FORMAL'],'Only frozen E4S formal jobs may be decorated')
  original=copy.deepcopy(job['maintenance']);archive.require(original['command_sha256']==command_digest(original['command']),'Original maintenance command digest differs')
  command=job['command'];input_root=command[command.index('--input-root')+1];parent=Path(job['receipt']).parent;sidecar=parent/'recoverable_outputs.json';post=parent/'recoverable_maintenance.json';spec_path=output/'postprocessing_specs'/(job['id']+'.json')
  wrapper_spec=dict(schema='e4s.required.postprocessing.v1',result=job['receipt'],input_root=input_root,store=spec['store'],archive_receipt=str(sidecar),archive_helper=spec['helper'],original_maintenance=original,bindings=[spec['helper'],spec['wrapper'],spec['exporter']],worker_command_sha256=command_digest(job['command']),worker_timeout_seconds=job['timeout'],timing='Original worker intervals/timeout unchanged. Archive and CPU maintenance occur after worker exit and outside measured intervals.')
  wrapper_binding=archive.immutable_json(spec_path,wrapper_spec);argv=[spec['python'],'-B',spec['wrapper']['path'],'--spec',str(spec_path),'--output',str(post)]
  bindings={x['path']:x for x in original['bindings']};bindings.update({x['path']:x for x in [spec['helper'],spec['wrapper'],spec['exporter'],wrapper_binding]})
  job['maintenance']=dict(command=argv,command_sha256=command_digest(argv),bindings=list(bindings.values()),receipt=str(post),expected_statuses=['PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE','PASS_FAILED_WORKER_CLEANUP'],gpu_lock=original['gpu_lock'],timeout=max(original.get('timeout',3600),spec.get('postprocess_timeout_seconds',86400)),environment={**original.get('environment',{}),'CUDA_VISIBLE_DEVICES':''})
  job['required_postprocess']='e4s_recoverable_outputs_v1';job['required_archive_receipt']=str(sidecar);job['original_maintenance_preserved_in']=wrapper_binding
 archive.require([j['command'] for j in result['jobs']]==[j['command'] for j in plan['jobs']],'Measured worker commands changed')
 return result
