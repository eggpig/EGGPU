"""Clone a fresh, explicit state policy epoch; CPU preparation only, never launch."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,shutil
import policy_guard
SOURCE='71ea757d372f8c1b969ea074caaa27d3a77d4cea6d48cd9c6ae9c4b02936d7ce'
HERE=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text())
def bind(path):
 p=Path(path).resolve();h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(item):
 if bind(item['path'])!=item:raise ValueError('Changed original binding: '+item['path'])
 return Path(item['path'])
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def rewrite(value,old,new):
 if isinstance(value,dict):return {k:rewrite(v,old,new) for k,v in value.items()}
 if isinstance(value,list):return [rewrite(v,old,new) for v in value]
 if isinstance(value,str) and (value==str(old) or value.startswith(str(old)+'/')):return str(new)+value[len(str(old)):]
 return value

def prepare(original,output,certificate,policy_cpu_proof):
 original=Path(original).resolve();output=Path(output).resolve();certificate=Path(certificate).resolve()
 selection=read(original/'selection.json');old_all=read(original/'all_plan.json')
 selected_original=read(original/'plan.json')
 authoritative={job['id']:job for job in selected_original['jobs']}
 if not set(authoritative)<={job['id'] for job in old_all['jobs']}:raise ValueError('Selected plan contains foreign jobs')
 # Retain any already-approved per-job maintenance/resource additions from
 # the actual selected plan, rather than an older all-plan template.
 old_all=copy.deepcopy(old_all);old_all['jobs']=[copy.deepcopy(authoritative.get(job['id'],job)) for job in old_all['jobs']]
 if selection['source_tree_sha256']!=SOURCE or len(old_all['jobs'])!=511:raise ValueError('Expected original source71 state511 jobs/504 targets')
 proof=read(policy_cpu_proof)
 if proof.get('schema')!='eggpu.source71.weighted-pr-packed-host-cpu.v1' or proof.get('status')!='PASS_WEIGHTED_PACKED_CPU_EQUIVALENCE' or proof.get('source_tree_sha256')!=SOURCE:raise ValueError('Exact source71 weighted policy CPU proof must pass')
 if proof.get('recommended_policy_candidate')!=policy_guard.EXPECTED_ENV:raise ValueError('CPU proof proposes a different policy')
 for name,item in proof['source_files'].items():
  actual=bind(Path(selection['source_root'])/name)
  if any(actual[k]!=item[k] for k in ('bytes','sha256')):raise ValueError('Policy CPU proof does not bind actual candidate source')
 for key in ('source_freeze','native_dso','native_build_receipt','runner_freeze','hash_cache_helper'):checked(selection[key])
 if Path(selection['runner']).resolve()!=original/'runner':raise ValueError('Original runner must be contained in original bundle')
 old_freeze=read(selection['runner_freeze']['path'])
 for name,entry in old_freeze['files'].items():checked(dict(entry,path=str(original/'runner'/name)))
 expected=read(certificate/'source_contract.json')['original_reference']
 if any(bind(original/'runner/state_reference.py')[k]!=expected[k] for k in ('bytes','sha256')):raise ValueError('Certificate/reference source differs')
 output.mkdir(parents=True,exist_ok=False)
 for name in old_freeze['files']:
  destination=output/'runner'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/'runner'/name,destination)
 shutil.copyfile(original/'runner/freeze.json',output/'runner/freeze.json')
 shutil.copyfile(original/'launch.py',output/'launch.py')
 policy_path=output/'measurement_policy.json'
 policy=dict(schema='state.explicit-pr-policy.v1',name='source71_state_auto_implicit_pull_only_v1',source_tree_sha256=SOURCE,
   environment=policy_guard.EXPECTED_ENV,expected_actual_metrics=policy_guard.EXPECTED_METRICS,guard=bind(HERE/'policy_guard.py'),
   cpu_proof=bind(policy_cpu_proof),original_bundle_selection=bind(original/'selection.json'),
   old_measurements_reused=False,new_focused_checks=1,new_physical_byte_pilots=6,
   original_controls_preserved=['transport','FP64 weights/wire','native_delta SSSP certification','arms','beta grid','trace/query counts','old reader interventions'],
   policy_changes_only=list(policy_guard.EXPECTED_ENV),gpu_qualified=False)
 save(policy_path,policy)
 selected=rewrite(selection,original,output);selected.update(runner_freeze=bind(output/'runner/freeze.json'),measurement_policy=bind(policy_path))
 save(output/'selection.json',selected)
 amendment_path=output/'audit_amendment.json'
 certificate_files=('pr_certificate.py','state_audit_migration_v1.py','PROOF.txt','cpu_receipt.json')
 amendment=dict(schema='state.audit.amendment.v1',method='state-pr-full-residual-certificate-v1',
   files={name:bind(certificate/name) for name in certificate_files},original_reference=bind(output/'runner/state_reference.py'),
   quality_thresholds_unchanged=True,measurement_code_and_timers_unchanged=True,full_vectors_retained=True,
   numerical_reference_error='Original independent oracle for tiny; explicit residual certificate or original-reference fallback for target/formal')
 amendment['files'].update({'launch_certificate.py':bind(HERE/'launch_state_policy.py'),'policy_guard.py':bind(HERE/'policy_guard.py'),'measurement_policy.json':bind(policy_path)})
 save(amendment_path,amendment)
 # Specs retain the original function/source/input parameters; only their fresh
 # output/spec/pilot paths and source binding paths differ in this new epoch.
 for job in old_all['jobs']:
  old_spec=Path(job['command'][job['command'].index('--spec')+1]);spec=rewrite(read(old_spec),original,output)
  spec['selection']=bind(output/'selection.json')
  if spec.get('paired_block'):
   spec['paired_block']['runner_freeze']=bind(output/'runner/freeze.json')
   spec['paired_block']['measurement_policy']=bind(policy_path)
   spec['paired_block_sha256']=hashlib.sha256(json.dumps(spec['paired_block'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
  save(Path(str(old_spec).replace(str(original),str(output),1)),spec)
 jobs=[]
 for old in old_all['jobs']:
  job=rewrite(old,original,output);job['environment'].update(policy_guard.EXPECTED_ENV)
  job['command'][2]=str(HERE/'launch_state_policy.py');job['command']+=['--audit-amendment',str(amendment_path)]
  job['bindings']=[bind(item['path']) if str(item['path']).startswith(str(output)+'/') else item for item in job['bindings']]
  job['bindings']+=[bind(amendment_path),*amendment['files'].values(),bind(policy_path),bind(policy_cpu_proof),bind(__file__)]
  jobs.append(job)
 by_id={j['id']:j for j in jobs}
 lineage=dict(original_selected_plan=bind(original/'plan.json'),original_plan=bind(original/'all_plan.json'),measurement_policy=bind(policy_path),audit_amendment=bind(amendment_path),fresh_observation_epoch=True,old_policy_results_reused=False)
 for path in sorted(original.glob('*_plan.json')):
  if path.name=='all_plan.json':doc=copy.deepcopy(old_all)
  else:doc=read(path)
  if 'jobs' not in doc:continue
  doc=rewrite(doc,original,output);doc['jobs']=[by_id[j['id']] for j in doc['jobs']];doc['new_policy_epoch']=lineage
  save(output/path.name,doc)
 pending=rewrite(read(original/'tail_pending.json'),original,output)
 pending['job_specs']=[by_id[j['id']] for j in pending['job_specs']];pending['core_plan']=bind(output/'paper_core_plan.json');pending['new_policy_epoch']=lineage
 save(output/'tail_pending.json',pending)
 chosen=read(original/'plan.json').get('selected_plan',{}).get('path','')
 if not chosen:raise ValueError('Original chosen plan must be explicit')
 chosen_name=Path(chosen).name;doc=read(output/chosen_name);save(output/'plan.json',dict(doc,selected_plan=bind(output/chosen_name)))
 coverage=rewrite(read(original/'coverage.json'),original,output);coverage.update(measurement_policy=bind(policy_path),reused_main_observations=0,reuse_reason='Fresh policy epoch; old general/materialized/full data retained only in original bundle')
 save(output/'coverage.json',coverage)
 result=dict(status='PREPARED_CPU_ONLY_NOT_GPU_QUALIFIED',selection=bind(output/'selection.json'),plan=bind(output/'plan.json'),
   targets=504,core_jobs=len(read(output/'paper_core_plan.json')['jobs']),tail_jobs=len(pending['job_specs']),initial_focused=1,physical_byte_pilots=6,
   source_tree_sha256=SOURCE,old_policy_results_reused=False,measurement_policy=bind(policy_path),gpu_executed=False)
 if result['core_jobs']!=82 or result['tail_jobs']!=429:raise ValueError('Original denominators changed')
 save(output/'stage_manifest.json',dict(result,queue=str(output/'queue'),plan=bind(output/'plan.json'),status='PREPARED_NOT_SUBMITTED'))
 save(output/'preparation.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('original-bundle','output','certificate','policy-cpu-proof'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();print(json.dumps(prepare(a.original_bundle,a.output,a.certificate,a.policy_cpu_proof)))
