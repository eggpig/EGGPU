"""Pause one exact owned author controller at its next complete-job boundary.

Root invokes this CPU coordinator. It takes no worker/GPU action. The unchanged
v8 controller admits the exact checkpoint and remaining original plan.
"""
from pathlib import Path
import argparse, copy, fcntl, hashlib, importlib.util, json, os, signal, subprocess, sys, time
TERMINAL={'PASS_REQUESTED_JOBS','COMPLETE_WITH_FAILURES'}
def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def save(p,value):
 p=Path(p);tmp=p.with_name(p.name+'.partial-'+str(os.getpid()));tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def argv(pid):return [x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
def assert_owner(sup,pid,expected,old,controller):
 now=sup.identity(pid)
 if now is None or now['state']=='Z' or now['starttime']!=expected['starttime']:raise RuntimeError('Exact old owner exited or identity changed')
 a=argv(pid)
 if str(controller) not in a or '--output' not in a or a[a.index('--output')+1]!=str(old):raise RuntimeError('Exact controller argv mismatch')
 return now
def descendants_empty(pid):
 for task in (Path('/proc')/str(pid)/'task').iterdir():
  if (task/'children').read_text().strip():return False
 return True
def rewrite_pointers(value,old,new):
 # Historical entries stay historical. Match only exact current queue pointers.
 count=0
 if isinstance(value,dict):
  out={}
  for k,v in value.items():
   if k in ('historical_stages','queue_history','history','previous_stage_manifest'):out[k]=v;continue
   if k=='queue' and v==old:out[k]=new;count+=1
   elif k in ('queues','stages'):
    out[k],n=rewrite_pointers(v,old,new);count+=n
   else:out[k]=v
  return out,count
 if isinstance(value,list):
  out=[]
  for v in value:
   if v==old:out.append(new);count+=1
   else:x,n=rewrite_pointers(v,old,new);out.append(x);count+=n
  return out,count
 return value,0

def run(a):
 c=a.campaign.resolve();h=c/'first_pass_20261106';old=(a.old_queue or h/'author_text_main/queue').resolve();out=(a.output or h/'author_text_after_main').resolve();queue=out/'queue';ancestor=(a.stage_manifest or h/'author_reservation_handoff/stage_manifest.json').resolve();controller=c/'run_campaign_queue_v8.py';lockpath=c/'root_controller.lock'
 sup=load('author_priority_owned',c/'owned_process_supervisor_v2.py');v8=load('author_priority_queue_v8',controller)
 if not lockpath.is_file():raise ValueError('Existing global controller lock required')
 out.mkdir(parents=True,exist_ok=True)
 with (out/'coordinator.lock').open('a+') as singleton:
  fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
  statepath=out/'priority_state.json'
  if statepath.exists():
   state=read(statepath)
   if state['owner_pid']!=a.pid or state['old_queue']!=str(old) or state['future_queue']!=str(queue):raise ValueError('Coordinator identity changed')
   if state['status'] not in ('PAUSED_WAITING_MAIN_AND_MISSING','RESUMED_AUTHOR_QUEUE'):raise RuntimeError('Ambiguous previous transition; inspect retained state')
  else:
   if a.mode=='resume':raise ValueError('No paused checkpoint')
   if queue.exists():raise FileExistsError('Future queue already exists')
   identity=sup.identity(a.pid)
   if identity is None:raise ValueError('Old owner absent')
   assert_owner(sup,a.pid,identity,old,controller)
   state=dict(schema='eggpu.author-main-priority.v1',status='WAITING_AUTHOR_COMPLETE_JOB_BOUNDARY',owner_pid=a.pid,owner_identity=identity,old_queue=str(old),future_queue=str(queue),controller=bind(controller),supervisor=bind(c/'owned_process_supervisor_v2.py'),stage_manifest=str(ancestor),controller_lock=str(lockpath),started_unix=time.time(),measurements_changed=False,completed_jobs_rerun=False)
   save(statepath,state)
   # Keep the pidfd from initial identity validation through the only signal.
   fd=sup.pidfd_open(a.pid)
   try:
    assert_owner(sup,a.pid,identity,old,controller)
    with lockpath.open('a+') as lock:
     fcntl.flock(lock,fcntl.LOCK_EX)
     for _ in range(600):
      assert_owner(sup,a.pid,identity,old,controller)
      oldstate=read(old/'state.json')
      if oldstate.get('status')=='WAITING_CONTROLLER_LOCK' and descendants_empty(a.pid):break
      time.sleep(.05)
     else:raise RuntimeError('Owned queue did not reach idle waiting checkpoint under lock')
     raw=(old/'state.json').read_bytes();oldstate=json.loads(raw);snapshot=old/'author_priority.waiting.json'
     if snapshot.exists():raise FileExistsError('Boundary checkpoint already exists; inspect prior transition')
     snapshot.write_bytes(raw);(out/'author.waiting.json').write_bytes(raw)
     plan=Path(oldstate['plan']['path'])
     # Validate no pending attempts in the ORIGINAL queue, not the archive dir.
     for job in oldstate['jobs'][len(oldstate['completed']):]:
      for suffix in ('.log','.log.supervision.json','.maintenance.log','.maintenance.log.supervision.json'):
       if (old/(job['id']+suffix)).exists():raise RuntimeError('Pending job already attempted')
     _,_,admission=v8.admit_unchanged_resume(plan,queue,snapshot,hashlib.sha256(raw).hexdigest(),c/'owned_process_supervisor_v2.py')
     prior_raw=ancestor.read_bytes();prior=json.loads(prior_raw);updated,count=rewrite_pointers(prior,str(old),str(queue))
     if not count:raise ValueError('Ancestor has no exact current old queue pointer')
     archive=out/'previous_stage_manifest.json';archive.write_bytes(prior_raw)
     updated.setdefault('historical_stages',[]).append(dict(queue=str(old),manifest=bind(archive),waiting_checkpoint=bind(snapshot)))
     updated.update(queue_status='PAUSING_AT_COMPLETE_JOB_BOUNDARY',deferred_resume_supervisor=str(statepath),future_plan=bind(plan))
     assert_owner(sup,a.pid,identity,old,controller)
     if not descendants_empty(a.pid) or (old/'state.json').read_bytes()!=raw:raise RuntimeError('Boundary changed under held lock')
     state.update(status='PAUSING_AT_COMPLETE_JOB_BOUNDARY',waiting_checkpoint=bind(snapshot),plan=bind(plan),completed=len(oldstate['completed']),attempted=oldstate['attempted'],admission=admission,previous_stage_manifest=bind(archive),pointer_replacements=count)
     save(statepath,state)
     # Publish the future pointer before SIGINT can make old state FAILED_CONTROLLER.
     save(ancestor,updated)
     sup.pidfd_send_signal(fd,signal.SIGINT)
     state['signal_sent']='SIGINT_EXACT_PIDFD';save(statepath,state)
     for _ in range(400):
      now=sup.identity(a.pid)
      if now is None or now['starttime']!=identity['starttime'] or now['state']=='Z':break
      time.sleep(.05)
     else:raise RuntimeError('Exact controller did not exit; no escalation is permitted')
     finalraw=(old/'state.json').read_bytes();final=json.loads(finalraw)
     if final['completed']!=oldstate['completed'] or final['attempted']!=oldstate['attempted']:raise RuntimeError('Completed prefix changed during pause')
     stopped=out/'author.stopped.json';stopped.write_bytes(finalraw)
     state.update(status='PAUSED_WAITING_MAIN_AND_MISSING',stopped_state=bind(stopped),paused_unix=time.time());save(statepath,state)
     updated['queue_status']='SCHEDULED_WAITING_MAIN_AND_MISSING';save(ancestor,updated)
   finally:os.close(fd)
  if a.mode=='pause' or state['status']=='RESUMED_AUTHOR_QUEUE':print(json.dumps(state));return
  prerequisite=[h/'unified_main_continuation/queue/state.json',h/'missing_external_continuation/queue/state.json']
  while True:
   statuses=[dict(path=str(p),status=read(p).get('status') if p.exists() else 'NOT_STARTED') for p in prerequisite]
   if all(r['status'] in TERMINAL for r in statuses):break
   if state.get('prerequisites')!=statuses:state['prerequisites']=statuses;save(statepath,state)
   time.sleep(a.poll_seconds)
  # Launch admission and state updates also serialize under the global lock.
  with lockpath.open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   now=sup.identity(a.pid)
   if now and now['starttime']==state['owner_identity']['starttime'] and now['state']!='Z':raise RuntimeError('Old exact controller still alive')
   for key in ('controller','supervisor','plan','waiting_checkpoint'):
    if bind(state[key]['path'])!=state[key]:raise RuntimeError('Bound '+key+' changed')
   if queue.exists():raise FileExistsError('Ambiguous existing future queue')
   snapshot=state['waiting_checkpoint'];v8.admit_unchanged_resume(state['plan']['path'],queue,snapshot['path'],snapshot['sha256'],c/'owned_process_supervisor_v2.py')
   for job in read(snapshot['path'])['jobs'][state['completed']:]:
    if (old/(job['id']+'.log')).exists():raise RuntimeError('Pending job subsequently attempted')
   command=[a.python,'-B',str(controller),'--plan',state['plan']['path'],'--output',str(queue),'--controller-lock',str(lockpath),'--resume-state',snapshot['path'],'--resume-sha',snapshot['sha256']]
   current=read(ancestor)
   _,still_old=rewrite_pointers(current,str(old),str(queue))
   if still_old or str(queue) not in json.dumps(current):raise RuntimeError('Future stage pointer changed')
   state.update(status='RESUME_LAUNCH_INTENT',command=command,prerequisites=statuses,launch_intent_unix=time.time());save(statepath,state)
   with (out/'resume.controller.log').open('xb') as log:
    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
   state.update(status='RESUMED_AUTHOR_QUEUE',controller_pid=child.pid,resumed_unix=time.time());save(statepath,state)
   current.update(queue_status='SUBMITTED',controller_pid=child.pid);save(ancestor,current)
  print(json.dumps(state))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--pid',type=int,required=True);p.add_argument('--old-queue',type=Path);p.add_argument('--output',type=Path);p.add_argument('--stage-manifest',type=Path);p.add_argument('--python',default=sys.executable);p.add_argument('--mode',choices=('run','pause','resume'),default='run');p.add_argument('--poll-seconds',type=float,default=10)
 a=p.parse_args()
 if not 1<=a.poll_seconds<=60:p.error('Poll must be 1..60 seconds')
 run(a)
