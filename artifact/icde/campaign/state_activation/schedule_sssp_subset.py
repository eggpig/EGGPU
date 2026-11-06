from pathlib import Path
import fcntl,hashlib,json,os,subprocess,time
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--campaign',type=Path,required=True)
parser.add_argument('--python',required=True)
parser.add_argument('--expected-plan-sha256',required=True)
args=parser.parse_args()
p=args.campaign.resolve();h=p/'first_pass_20261106';out=h/'sssp_subset_plan';py=args.python
expected=args.expected_plan_sha256
def read(q):return json.loads(q.read_text())
def save(q,d):
 t=q.with_suffix('.partial');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(q)
with (out/'successor.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert hashlib.sha256((out/'plan.json').read_bytes()).hexdigest()==expected
 assert not (out/'controller_launch.json').exists()
 while True:
  prerequisites=[]
  for name in ('unified_main_continuation','missing_external_continuation'):
   q=h/name/'queue/state.json';s=read(q) if q.exists() else {};prerequisites.append(dict(path=str(q),status=s.get('status','NOT_STARTED')))
  if any(r['status'].startswith('FAILED_') for r in prerequisites):
   save(out/'successor_state.json',dict(status='BLOCKED_CONTROLLER_FAILURE',prerequisites=prerequisites));raise SystemExit(1)
  if all(r['status'] in ('PASS_REQUESTED_JOBS','COMPLETE_WITH_FAILURES') for r in prerequisites):break
  save(out/'successor_state.json',dict(status='WAITING_MAIN_AND_MISSING',prerequisites=prerequisites,updated_unix=time.time()));time.sleep(15)
 assert not (out/'queue').exists()
 assert hashlib.sha256((out/'plan.json').read_bytes()).hexdigest()==expected
 cmd=[py,'-B',str(p/'run_campaign_queue_v9.py'),'--plan',str(out/'plan.json'),'--output',str(out/'queue'),'--controller-lock',str(p/'root_controller.lock'),'--wall-timeout-floor','10800']
 receipt=dict(status='LAUNCH_INTENT',command=cmd,plan_sha256=expected)
 with (out/'controller_launch.json').open('x') as stream:json.dump(receipt,stream)
 with (out/'controller.log').open('xb') as log:child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 receipt.update(status='SUBMITTED',pid=child.pid);save(out/'controller_launch.json',receipt);save(out/'successor_state.json',receipt)
 manifest=read(out/'stage_manifest.json');manifest.update(queue_status='SUBMITTED',controller_pid=child.pid);save(out/'stage_manifest.json',manifest)
