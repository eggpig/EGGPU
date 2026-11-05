"""Serial source-bound jobs with actual phase counts and non-owning storage waits.

Workers retain their GPU/resource gates. No failed worker is automatically retried.
Maintenance is explicit and argv-and-file-SHA-bound after worker exit. The
controller lock remains held; the worker GPU lock is checked free before the
maintenance command acquires its own optional GPU lock. This module does not submit any job merely by importing it.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,re,shutil,signal,subprocess,time,select,sys


class UnsafeProcessTreeError(BaseException):
 """A cleanup uncertainty stops the queue before any further worker admission."""


def bind(path):
 p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def command_digest(command):return hashlib.sha256(json.dumps(command,separators=(',',':')).encode()).hexdigest()

def phase_of(job):
 command=job['command'];phases=[]
 for index,arg in enumerate(command):
  if arg=='--phase':
   if index+1==len(command):raise ValueError('Missing --phase value')
   phases.append(command[index+1])
  elif arg.startswith('--phase='):phases.append(arg.split('=',1)[1])
 if len(phases)>1:raise ValueError('Duplicate phase flags')
 declared=job.get('phase')
 if declared is not None and phases and declared!=phases[0]:raise ValueError('job.phase and command phase differ')
 value=declared if declared is not None else phases[0] if phases else 'unspecified'
 if value not in ('tiny','target','formal','pilot','qualification','diagnostic','cpu','unspecified'):raise ValueError('Unknown phase')
 return value


def check_bindings(rows):
 for expected in rows:
  actual=bind(expected['path'])
  if actual['sha256']!=expected['sha256'] or ('bytes' in expected and actual['bytes']!=expected['bytes']):raise ValueError('Code/input binding changed: '+expected['path'])


def disk_available(guard):
 path=Path(guard['path']).resolve()
 while not path.exists():
  if path==path.parent:raise ValueError('Cannot locate storage filesystem')
  path=path.parent
 free=shutil.disk_usage(path).free
 return dict(path=guard['path'],observed_filesystem_path=str(path),free_bytes=free,min_free_bytes=guard['min_free_bytes'],ready=free>=guard['min_free_bytes'])


def check_maintenance(spec):
 command=spec['command'];bindings=spec.get('bindings',[])
 if not isinstance(command,list) or not command or any(not isinstance(x,str) for x in command):raise ValueError('Maintenance command must be explicit argv')
 if spec.get('command_sha256')!=command_digest(command) or not bindings:raise ValueError('Maintenance argv SHA and file bindings are required')
 executable=shutil.which(command[0]) if not Path(command[0]).is_absolute() else command[0]
 if executable is None:raise ValueError('Missing maintenance executable')
 required={str(Path(executable).resolve())}
 required.update(str(Path(x).resolve()) for x in command[1:] if x.endswith(('.py','.sh')) and Path(x).is_file())
 if not required<={str(Path(x['path']).resolve()) for x in bindings}:raise ValueError('Maintenance executable/direct script must be SHA-bound')
 if not spec.get('gpu_lock'):raise ValueError('Maintenance must verify released worker GPU lock')
 return spec


def run_process(command,env,log,timeout,on_start):
 supervisor=Path(__file__).with_name('owned_process_supervisor_v2.py')
 report=Path(str(log)+'.supervision.json')
 argv=[sys.executable,'-B',str(supervisor),'--command-json',json.dumps(command,separators=(',',':')),
       '--timeout',str(timeout),'--report',str(report),'--expected-parent',str(os.getpid())]
 with log.open('xb') as stream:
  process=subprocess.Popen(argv+['--log-fd',str(stream.fileno())],env=env,stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,stderr=stream,pass_fds=(stream.fileno(),),start_new_session=True,text=True)
  try:
   ready,_,_=select.select([process.stdout],[],[],min(10.,max(1.,timeout)))
   if not ready:raise RuntimeError('CPU supervisor failed to admit worker within startup deadline')
   line=process.stdout.readline()
   if not line:raise RuntimeError('CPU supervisor exited before worker admission')
   event=json.loads(line)
   if event.get('event')!='worker_started':raise ValueError('Unexpected supervisor handshake')
   on_start(event['pid'])
   code=process.wait(timeout=timeout+15)
   if not report.is_file():raise UnsafeProcessTreeError('Missing owned-process cleanup report')
   outcome=json.loads(report.read_text())
   if outcome.get('surviving_owned_processes'):raise UnsafeProcessTreeError('Owned descendants remain; inspect supervisor report')
   if outcome.get('status')=='TIMED_OUT':raise subprocess.TimeoutExpired(command,timeout)
   return code
  except BaseException:
   if process.poll() is None:
    process.terminate()
    try:process.wait(timeout=12)
    except subprocess.TimeoutExpired:
     # Never continue a campaign after an unresponsive supervisor. Killing its
     # process group cannot establish cleanup of separate-session descendants.
     raise UnsafeProcessTreeError('Supervisor cleanup unresponsive; manual owned-process review required')
   raise
  finally:
   process.stdout.close()


def run(plan_path,output,controller_lock):
 plan_path=Path(plan_path);output=Path(output);controller_lock=Path(controller_lock);plan=json.loads(plan_path.read_text())
 jobs=plan['jobs'];known=set();phases={}
 poll=plan.get('storage_poll_seconds',5)
 if not isinstance(poll,(int,float)) or not 0<poll<=60:raise ValueError('Storage poll interval must be in (0,60] seconds')
 for job in jobs:
  name=job['id']
  if not re.fullmatch(r'[A-Za-z0-9_.-]+',name) or name in known:raise ValueError('Unsafe/duplicate job identity')
  if not set(job.get('depends_on',[]))<=known:raise ValueError('Dependencies must precede consumer')
  if not isinstance(job['command'],list) or not job['command'] or any(not isinstance(x,str) for x in job['command']):raise ValueError('Explicit worker argv required')
  phases[name]=phase_of(job);known.add(name)
  if 'disk_guard' in job:
   guard=job['disk_guard']
   if not isinstance(guard.get('path'),str) or type(guard.get('min_free_bytes')) is not int or guard['min_free_bytes']<0:raise ValueError('Invalid disk guard')
  if job.get('maintenance'):check_maintenance(job['maintenance'])
 output.mkdir(parents=True,exist_ok=False)
 state=dict(schema='campaign.queue.v5',status='WAITING_CONTROLLER_LOCK',plan=bind(plan_path),controller=bind(__file__),supervisor=bind(Path(__file__).with_name('owned_process_supervisor_v2.py')),jobs=jobs,job_phases=phases,completed=[],attempted=0,passed=0,failed=0,skipped_dependency=0,failed_prelaunch=0,formal_planned=sum(p=='formal' for p in phases.values()),formal_attempted=0,formal_passed=0,formal_failed_attempts=0,formal_measurements=0,formal_measurements_unit='successful formal worker jobs; not vector count or statistical sub-samples',phase_counts={},maintenance_attempted=0,maintenance_passed=0,maintenance_failed=0,storage_wait_observations=0)
 for phase in set(phases.values()):state['phase_counts'][phase]=dict(planned=sum(p==phase for p in phases.values()),attempted=0,passed=0,failed=0,failed_prelaunch=0,skipped_dependency=0)
 def save():
  p=output/'state.partial.json';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(output/'state.json')
 def started(item,phase,pid):
  item.update(pid=pid,execution_started=True);state['attempted']+=1;state['phase_counts'][phase]['attempted']+=1
  if phase=='formal':state['formal_attempted']+=1
  save()
 def complete(item,phase):
  outcome=item['outcome'];counts=state['phase_counts'][phase]
  if outcome=='PASS':
   state['passed']+=1;counts['passed']+=1
   if phase=='formal':state['formal_passed']+=1;state['formal_measurements']=state['formal_passed']
  elif outcome=='SKIPPED_DEPENDENCY':state['skipped_dependency']+=1;counts['skipped_dependency']+=1
  else:
   state['failed']+=1;counts['failed']+=1
   if not item.get('execution_started'):state['failed_prelaunch']+=1;counts['failed_prelaunch']+=1
   elif phase=='formal':state['formal_failed_attempts']+=1
  state['completed'].append(item);save()
 save();outcomes={}
 try:
  for job in jobs:
   phase=phases[job['id']]
   while True:
    state.update(status='WAITING_CONTROLLER_LOCK',active_job=job['id']);save();waiting=False;abort=None
    with controller_lock.open('a+') as owner:
     fcntl.flock(owner,fcntl.LOCK_EX)
     if any(outcomes[d]!='PASS' for d in job.get('depends_on',[])):
      item=dict(id=job['id'],phase=phase,outcome='SKIPPED_DEPENDENCY',execution_started=False,gpu_submitted=False)
     else:
      guard=disk_available(job['disk_guard']) if job.get('disk_guard') else None
      if guard and not guard['ready']:
       state.update(status='WAITING_STORAGE',storage_guard=guard,worker_started_for_active_job=False);state['storage_wait_observations']+=1;save();waiting=True
      else:
       state.update(status='RUNNING',worker_started_for_active_job=False);state.pop('storage_guard',None);save()
       item=dict(id=job['id'],phase=phase,outcome='FAIL',execution_started=False,started_unix=time.time());begin=time.monotonic();abort=None
       try:
        check_bindings(job.get('bindings',[]));env={**os.environ,**plan.get('environment',{}),**job.get('environment',{})}
        def launched(pid):
         state['worker_started_for_active_job']=True;started(item,phase,pid)
        code=run_process(job['command'],env,output/(job['id']+'.log'),job.get('timeout',3600),launched);item['returncode']=code
        receipt=Path(job['receipt']);result=json.loads(receipt.read_text()) if receipt.exists() else {};item['status']=result.get('status','MISSING_RECEIPT');item['receipt']=bind(receipt) if receipt.exists() else None
        if phase!='unspecified' and result.get('phase',phase)!=phase:raise ValueError('Worker receipt phase differs from declared phase')
        if phase=='unspecified' and result.get('phase')=='formal':raise ValueError('Formal worker requires an explicit declared phase')
        if code==0 and item['status'] in job['expected_statuses']:item['outcome']='PASS'
       except BaseException as error:
        item['error']=type(error).__name__+': '+str(error)
        if not isinstance(error,Exception):abort=error
       item['seconds']=time.monotonic()-begin
       supervision=output/(job['id']+'.log.supervision.json')
       if supervision.is_file():item['process_supervision']=bind(supervision)
     if not waiting:
      outcomes[job['id']]=item['outcome'];complete(item,phase)
      if item.get('execution_started') and job.get('maintenance'):
       spec=job['maintenance'];state['status']='RUNNING_MAINTENANCE';save();maintenance=dict(status='FAIL',started_unix=time.time(),command_sha256=spec['command_sha256']);item['maintenance']=maintenance
       try:
        check_maintenance(spec);check_bindings(spec['bindings'])
        if not Path(spec['gpu_lock']).is_file():raise ValueError('Existing worker GPU lock file required')
        with Path(spec['gpu_lock']).open('a+') as gpu_guard:
         fcntl.flock(gpu_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
         maintenance['worker_gpu_lock_observed_released']=True
         fcntl.flock(gpu_guard,fcntl.LOCK_UN)
         env={**os.environ,**plan.get('environment',{}),**job.get('environment',{}),**spec.get('environment',{})}
         def maintain_started(pid):maintenance['pid']=pid;state['maintenance_attempted']+=1;save()
         code=run_process(spec['command'],env,output/(job['id']+'.maintenance.log'),spec.get('timeout',3600),maintain_started)
         maintenance['returncode']=code
         if code:raise RuntimeError('Maintenance command failed')
         if spec.get('receipt'):
          p=Path(spec['receipt']);r=json.loads(p.read_text());maintenance['receipt']=bind(p)
          if r.get('status') not in spec['expected_statuses']:raise ValueError('Maintenance receipt failed')
         maintenance['status']='PASS';state['maintenance_passed']+=1
       except BaseException as error:
        maintenance['error']=type(error).__name__+': '+str(error);state['maintenance_failed']+=1;state['status']='FAILED_MAINTENANCE';save();raise
       finally:maintenance['finished_unix']=time.time();save()
      if abort is not None:raise abort
    if waiting:
     # Controller lock is released before sleeping. No GPU launch/failure/retry occurred.
     time.sleep(poll);continue
    break
   time.sleep(.05)
  state['status']='PASS_REQUESTED_JOBS' if all(x=='PASS' for x in outcomes.values()) else 'COMPLETE_WITH_FAILURES'
 except BaseException as error:
  if state['status']!='FAILED_MAINTENANCE':state['status']='FAILED_CONTROLLER'
  state['error']=type(error).__name__+': '+str(error);raise
 finally:state['finished_unix']=time.time();save()
 return state

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['plan','output','controller-lock']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();run(a.plan,a.output,a.controller_lock)
