"""Replace idle scheduling owners; retain current measured workers and old results."""
from pathlib import Path
import sys,json,os,time,signal,subprocess,copy,hashlib
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--campaign',type=Path,required=True)
parser.add_argument('--python',required=True)
parser.add_argument('--old-supervisor-pid',type=int,required=True)
parser.add_argument('--old-tail-pid',type=int,required=True)
parser.add_argument('--old-additions-pid',type=int,required=True)
args=parser.parse_args()
p=args.campaign.resolve();h=p/'first_pass_20261106';py=args.python
sys.path.insert(0,str(p));import owned_process_supervisor_v2 as owned
def read(q):return json.loads(Path(q).read_text())
def bind(q):
 q=Path(q);b=q.read_bytes();return dict(path=str(q),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(q,v):
 q=Path(q);q.parent.mkdir(parents=True,exist_ok=True)
 with q.open('x') as s:json.dump(v,s,indent=2);s.write('\n')
out=h/'after_main_pr_v1';out.mkdir(exist_ok=True)
assert not list(out.iterdir()), 'Existing prepared schedule must not be rewritten'
assert not (h/'state_main_resumed/queue').exists()
assert not (h/'unified_mechanism_application_resumed/queue').exists()
assert not (h/'unified_scaling_continuation/queue').exists()
assert read(h/'after_main_resume_runtime/state.json')['status']=='WAITING_MAIN_AND_MISSING'
assert read(h/'state_tail_after_main/state.json')['status']=='WAITING_MAJOR_FIRST_PASS_STAGES'
assert read(h/'state_scope_additions_plan/successor_state.json')['status']=='WAITING_PRIMARY_STATE_QUEUE'
checks=[(args.old_supervisor_pid,h/'state_audit_scheduling/after_main_supervise_v2.py',h/'after_main_resume_runtime'),(args.old_tail_pid,h/'state_audit_scheduling/state_tail_successor_v2.py',h/'state_tail_after_main/manifest.json'),(args.old_additions_pid,h/'state_scope_additions/run_after_primary.py',h/'state_scope_additions/run_after_primary.py')]
identities=[]
for pid,script,scope in checks:
 try:raw=(Path('/proc')/str(pid)/'cmdline').read_bytes()
 except FileNotFoundError:
  for candidate in Path('/proc').iterdir():
   if not candidate.name.isdigit():continue
   try:other=(candidate/'cmdline').read_bytes()
   except (FileNotFoundError,ProcessLookupError):continue
   assert not (str(script).encode() in other and str(scope).encode() in other), 'Unexpected replacement scheduling owner'
  continue
 assert str(script).encode() in raw and str(scope).encode() in raw,(pid,raw)
 row=owned.identity(pid);assert row and row['state']!='Z'
 assert not [r for r in owned.snapshot().values() if r['ppid']==pid and r['state']!='Z'],'Only childless idle watchers may be replaced'
 identities.append(row)
for row in identities:
 result=owned.signal_identity(row,signal.SIGTERM)
 if row['state']=='T':owned.signal_identity(row,signal.SIGCONT)
 row['stop_signal']=result
limit=time.monotonic()+5
while time.monotonic()<limit:
 if all(owned.identity(r['pid']) is None or owned.identity(r['pid'])['state']=='Z' for r in identities):break
 time.sleep(.1)
assert all(owned.identity(r['pid']) is None or owned.identity(r['pid'])['state']=='Z' for r in identities)
save(out/'old_idle_watchers_stopped.json',dict(status='STOPPED_ONLY_IDLE_CHILDLESS_WATCHERS',owners=identities,active_workers_untouched=True,old_results_preserved=True))
config=copy.deepcopy(read(h/'first_table_priority/resume_config.json'))
config['schema']='eggpu.after-main.resume.config.v2'
config['prerequisite_queue_states']=[str(h/'unified_main_continuation/queue/state.json'),str(h/'missing_external_continuation/queue/state.json')]
config['fresh_state_policy']=bind(h/'state_main_pr_v1/measurement_policy.json')
config['stages'][0]=dict(name='state_main',mode='start',plan=str(h/'state_main_pr_v1/plan.json'),queue=str(h/'state_main_pr_v1/queue'),ancestor_stage_manifest=str(h/'state_main/stage_manifest.json'),controller=str(p/'run_campaign_queue_v9.py'),wall_timeout_floor=10800)
configpath=out/'config.json';save(configpath,config)
cmd=[py,'-B',str(h/'state_pr_policy_v1/after_main_supervise_v4.py'),'--config',str(configpath),'--output',str(out/'runtime')]
subprocess.run(cmd+['--prepare-only'],check=True)
with (out/'supervisor.log').open('xb') as log:child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
save(out/'launch.json',dict(status='FRESH_STATE_AND_UNCHANGED_MECHANISM_SCALING_SCHEDULED',pid=child.pid,command=cmd,config=bind(configpath),gpu_jobs_not_launched_by_preparation=True))
print(json.dumps(read(out/'launch.json')))
