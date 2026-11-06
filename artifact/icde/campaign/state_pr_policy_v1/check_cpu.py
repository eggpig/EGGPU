"""CPU-only planner and actual-policy receipt checks; never executes GPU workers."""
from pathlib import Path
import copy,importlib.util,json,os,shutil,tempfile,time
import prepare_fresh_policy as p
import policy_guard
HERE=Path(__file__).resolve().parent;F=HERE.parent
SCRATCH=Path(os.environ['EGGPU_POLICY_TEST_SCRATCH']);SCRATCH.mkdir(parents=True,exist_ok=True)

def main():
 begin=time.monotonic()
 with tempfile.TemporaryDirectory(dir=SCRATCH) as work:
  root=Path(work);old=root/'old';old.mkdir();runner=old/'runner';runner.mkdir()
  source_runner=F/'state_main_first_pass/runner';frozen=p.read(source_runner/'freeze.json')
  for name in frozen['files']:
   dest=runner/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source_runner/name,dest)
  shutil.copyfile(source_runner/'freeze.json',runner/'freeze.json')
  shutil.copyfile(F/'state_main_first_pass/launch.py',old/'launch.py')
  fake=root/'cpu_only_binding';fake.write_text('CPU planning fixture; never a native binary\n')
  selection=dict(source_tree_sha256=p.SOURCE,source_root=str(F/'unified_release_20261106/Easy-Graph'),source_freeze=p.bind(fake),native_dso=p.bind(fake),native_build_receipt=p.bind(fake),runner=str(runner),runner_freeze=p.bind(runner/'freeze.json'),hash_cache_helper=p.bind(F/'timing/cache_worker.py'))
  p.save(old/'selection.json',selection)
  jobs=[];core=[]
  identifiers=[('focused_source_p8','run','tiny')]+[('pilot'+str(i),'pilot','target') for i in range(6)]+[('target_'+str(i),'run','target') for i in range(504)]
  for index,(name,action,phase) in enumerate(identifiers):
   specpath=old/'specs'/(name+'.json');output=old/'results'/name
   spec=dict(id=name,action=action,phase=phase,selection=p.bind(old/'selection.json'),worker_arguments=[action,'--output',str(output),'--first-pass-spec',str(specpath)],paired_block=dict(runner_freeze=p.bind(runner/'freeze.json'),arm='test',inputs=p.bind(fake)))
   p.save(specpath,spec)
   job=dict(id=name,phase=phase,command=['python','-B',str(old/'launch.py'),'--selection',str(old/'selection.json'),'--spec',str(specpath)],receipt=str(output/'receipt.json'),expected_statuses=['PASS'],depends_on=[],timeout=3600,environment={'EGGPU_COMM_BACKEND':'host_staging','EGMGPU_SSSP_WEIGHT_ENCODING':'f64'},bindings=[p.bind(old/'launch.py'),p.bind(old/'selection.json'),p.bind(runner/'freeze.json'),p.bind(specpath)])
   jobs.append(job)
   if name=='focused_source_p8' or name in ('pilot0','pilot1','pilot2') or (name.startswith('target_') and int(name[7:])<78):core.append(job)
  assert len(jobs)==511 and len(core)==82
  p.save(old/'all_plan.json',dict(jobs=jobs,full_denominator=504))
  p.save(old/'paper_core_plan.json',dict(jobs=core,targets=78))
  selected_core=copy.deepcopy(core);selected_core[4]['maintenance_note']='approved selected-plan field retained'
  p.save(old/'plan.json',dict(jobs=selected_core,targets=78,selected_plan=p.bind(old/'paper_core_plan.json')))
  p.save(old/'tail_pending.json',dict(job_specs=[j for j in jobs if j not in core],core_plan=p.bind(old/'paper_core_plan.json')))
  p.save(old/'coverage.json',dict(configurations=[dict(id=j['id']) for j in jobs if j['id'].startswith('target_')],reused_main_observations=0))
  original={str(q.relative_to(old)):p.bind(q) for q in old.rglob('*') if q.is_file()}
  fresh=root/'fresh';result=p.prepare(old,fresh,F/'pr_residual_certificate',F/'pr_runtime_policy_audit/cpu_receipt.json')
  assert result['core_jobs']==82 and result['tail_jobs']==429 and result['targets']==504
  assert p.read(fresh/'paper_core_plan.json')['jobs'][4]['maintenance_note']=='approved selected-plan field retained'
  new=p.read(fresh/'all_plan.json')['jobs'];assert [j['id'] for j in new]==[j['id'] for j in jobs]
  for before,after in zip(jobs,new):
   assert after['environment']==dict(before['environment'],**policy_guard.EXPECTED_ENV)
   assert after['phase']==before['phase'] and after['timeout']==before['timeout'] and after['depends_on']==before['depends_on']
   assert after['receipt']==before['receipt'].replace(str(old),str(fresh),1)
   oldspec=p.read(before['command'][6]);newspec=p.read(after['command'][6])
   assert newspec['worker_arguments']==p.rewrite(oldspec['worker_arguments'],old,fresh)
   assert newspec['selection']==p.bind(fresh/'selection.json')
   for binding in after['bindings']:p.checked(binding)
  assert p.read(fresh/'tail_pending.json')['core_plan']==p.bind(fresh/'paper_core_plan.json')
  assert p.read(fresh/'plan.json')['selected_plan']==p.bind(fresh/'paper_core_plan.json')
  assert p.read(fresh/'runner/freeze.json')==p.read(old/'runner/freeze.json')
  assert {str(q.relative_to(old)):p.bind(q) for q in old.rglob('*') if q.is_file()}==original
  assert not (fresh/'results').exists() and not (fresh/'queue').exists()
  module_spec=importlib.util.spec_from_file_location('fresh_schedule',HERE/'after_main_supervise_v4.py');schedule=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(schedule)
  controller=F/'state_audit_scheduling/run_campaign_queue_v9.py';lock=root/'controller.lock';lock.touch()
  mechanism=root/'mechanism.json';p.save(mechanism,dict(jobs=[]));checkpoint=root/'mechanism_state.json';p.save(checkpoint,dict(status='WAITING_CONTROLLER_LOCK',plan=p.bind(mechanism),completed=[]))
  stages=[]
  for name,mode,plan in [('state_main','start',fresh/'plan.json'),('unified_mechanism_application','resume',mechanism),('scaling','start',mechanism)]:
   ancestor=root/(name+'_stage.json');p.save(ancestor,dict(queue='old_'+name,plan=p.bind(mechanism)))
   stage=dict(name=name,mode=mode,plan=str(plan),queue=str(root/(name+'_future_queue')),ancestor_stage_manifest=str(ancestor))
   if mode=='resume':stage.update(resume_state=str(checkpoint),resume_sha=p.bind(checkpoint)['sha256'],old_owner_pid=99999990,old_queue='old_mechanism',stop_receipt=str(checkpoint))
   if name=='state_main':stage.update(controller=str(controller),wall_timeout_floor=10800)
   stages.append(stage)
  config=root/'schedule.json';p.save(config,dict(schema='eggpu.after-main.resume.config.v2',fresh_state_policy=p.bind(fresh/'measurement_policy.json'),python='python',controller=str(controller),controller_lock=str(lock),prerequisite_queue_states=['repaired_main/state.json','missing_external_continuation/queue/state.json'],stages=stages))
  _,scheduled=schedule.prepare(config,root/'schedule_runtime')
  assert scheduled['stages'][0]['mode']=='start' and scheduled['stages'][1]['mode']=='resume'
  pointer=p.read(stages[0]['ancestor_stage_manifest']);assert pointer['plan']==p.bind(fresh/'plan.json') and pointer['old_policy_results_historical']

 for devices,method in [([0],'whole_contiguous'),(list(range(8)),'packed_destination_owned')]:
  record=dict(environment=policy_guard.EXPECTED_ENV,devices=devices,events=[dict(kind='query',function='weighted_pr',index=0,details=dict(metrics=dict(policy_guard.EXPECTED_METRICS,host_transpose='auto',host_transpose_method=method)))])
  assert policy_guard.inspect(record)['status']=='PASS_ACTUAL_PR_POLICY'
  record['events'][0]['details']['metrics']['host_transpose_method']='general_destination_owned'
  try:policy_guard.inspect(record)
  except ValueError:pass
  else:raise AssertionError('Old general host path accepted')
 pilot=policy_guard.inspect(dict(schema='state.byte.pilot.v4',environment=policy_guard.EXPECTED_ENV))
 assert pilot['physical_bytes_remeasured'] and not pilot['actual_query_route_claimed']
 receipt=dict(status='PASS_CPU_FRESH_STATE_POLICY_PREPARATION',source=p.bind(HERE/'prepare_fresh_policy.py'),jobs=511,targets=504,core_jobs=82,tail_jobs=429,checked_job_bindings=True,original_bundle_unchanged=True,new_focused=1,new_physical_pilots=6,actual_route_negative_checks=True,gpu_executed=False,seconds=time.monotonic()-begin)
 p.save(HERE/'cpu_receipt.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
