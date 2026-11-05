"""Serial source-bound jobs with strict maintenance-only plan migration and non-owning storage waits.

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



def _resume_counts(jobs, completed):
 phases={j['id']:phase_of(j) for j in jobs}
 result=dict(attempted=0,passed=0,failed=0,skipped_dependency=0,failed_prelaunch=0,
             formal_planned=sum(p=='formal' for p in phases.values()),formal_attempted=0,
             formal_passed=0,formal_failed_attempts=0,formal_measurements=0,
             maintenance_attempted=0,maintenance_passed=0,maintenance_failed=0,phase_counts={})
 for phase in set(phases.values()):
  result['phase_counts'][phase]=dict(planned=sum(p==phase for p in phases.values()),attempted=0,passed=0,failed=0,failed_prelaunch=0,skipped_dependency=0)
 by_id={j['id']:j for j in jobs}
 for item in completed:
  phase=phases[item['id']];c=result['phase_counts'][phase]
  if item.get('phase')!=phase:raise ValueError('Inherited phase mismatch')
  started=item.get('execution_started',False)
  if type(started) is not bool:raise ValueError('Invalid inherited execution marker')
  if started:
   result['attempted']+=1;c['attempted']+=1
   if phase=='formal':result['formal_attempted']+=1
  outcome=item.get('outcome')
  if outcome=='PASS':
   if not started:raise ValueError('Inherited PASS did not execute')
   result['passed']+=1;c['passed']+=1
   if phase=='formal':result['formal_passed']+=1;result['formal_measurements']+=1
  elif outcome=='SKIPPED_DEPENDENCY':
   if started:raise ValueError('Skipped dependency cannot have executed')
   result['skipped_dependency']+=1;c['skipped_dependency']+=1
  elif outcome=='FAIL':
   result['failed']+=1;c['failed']+=1
   if not started:result['failed_prelaunch']+=1;c['failed_prelaunch']+=1
   elif phase=='formal':result['formal_failed_attempts']+=1
  else:raise ValueError('Unknown inherited outcome')
  if started and by_id[item['id']].get('maintenance'):
   result['maintenance_attempted']+=1;result['maintenance_passed']+=1
 return result


def admit_unchanged_resume(plan_path, output, resume_state, resume_sha, supervisor):
 """Verify a terminal prefix without changing it or admitting any worker.

 Only an observed between-job WAITING_CONTROLLER_LOCK checkpoint is accepted.
 The root must additionally establish that the old controller and its workers
 have exited before invoking this entrypoint. An active, partial, failed or
 ambiguous maintenance attempt is never retried by this API.
 """
 if resume_state is None or resume_sha is None:raise ValueError('Both resume state and SHA are required')
 if not re.fullmatch(r'[0-9a-f]{64}',resume_sha):raise ValueError('Invalid resume SHA')
 old_path=Path(resume_state);raw=old_path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=resume_sha:raise ValueError('Resume state SHA changed')
 old=json.loads(raw);plan=json.loads(Path(plan_path).read_text());jobs=plan['jobs']
 if Path(output).exists():raise FileExistsError('Resume requires a new queue output')
 if old.get('status')!='WAITING_CONTROLLER_LOCK':raise ValueError('Resume is restricted to a between-job waiting checkpoint')
 if old.get('plan')!=bind(plan_path) or old.get('jobs')!=jobs:raise ValueError('Resume requires the unchanged complete original plan')
 check_bindings([old['controller'],old['supervisor']])
 if old['supervisor']!=bind(supervisor):raise ValueError('Supervisor changed across resume')
 completed=old.get('completed',[])
 if not isinstance(completed,list) or len(completed)>=len(jobs):raise ValueError('Resume requires a proper completed prefix and remaining jobs')
 if [r.get('id') for r in completed]!=[j['id'] for j in jobs[:len(completed)]]:raise ValueError('Completed jobs are not the exact original prefix')
 if old.get('job_phases')!={j['id']:phase_of(j) for j in jobs}:raise ValueError('Original phase map changed')
 if old.get('active_job')!=jobs[len(completed)]['id']:raise ValueError('Active identity is not the first unstarted job')
 calculated=_resume_counts(jobs,completed)
 for key,value in calculated.items():
  if old.get(key)!=value:raise ValueError('Inherited counter mismatch: '+key)
 proofs=[];outcomes={};maintenance_proofs={}
 inherited_maintenance=old.get('maintenance_supervision_bindings',{})
 if not isinstance(inherited_maintenance,dict):raise ValueError('Invalid inherited maintenance proof map')
 for job,item in zip(jobs,completed):
  prior_failed=any(outcomes[d]!='PASS' for d in job.get('depends_on',[]))
  if (item['outcome']=='SKIPPED_DEPENDENCY') != prior_failed:raise ValueError('Inherited dependency outcome mismatch')
  if item.get('receipt') is not None:
   if item['receipt']['path']!=job['receipt']:raise ValueError('Inherited receipt path changed')
   check_bindings([item['receipt']]);proofs.append(item['receipt']);r=json.loads(Path(job['receipt']).read_text())
   if r.get('status','MISSING_RECEIPT')!=item.get('status'):raise ValueError('Inherited worker status differs from receipt')
   phase=phase_of(job)
   if phase!='unspecified' and r.get('phase',phase)!=phase:raise ValueError('Inherited receipt phase changed')
  elif item['outcome']=='PASS':raise ValueError('Inherited PASS has no receipt')
  if item['outcome']=='PASS' and (item.get('returncode')!=0 or item.get('status') not in job['expected_statuses']):raise ValueError('Inherited PASS was not qualified')
  if item.get('execution_started'):
   supervision=item.get('process_supervision')
   if not supervision:raise ValueError('Executed inherited job lacks cleanup proof')
   check_bindings([supervision]);proofs.append(supervision)
   evidence=json.loads(Path(supervision['path']).read_text())
   if evidence.get('surviving_owned_processes') or 'finished_monotonic' not in evidence:raise ValueError('Inherited worker cleanup is incomplete')
   terminal={'PASS_PROCESS_EXIT','WORKER_FAILED','TIMED_OUT','CANCELLED','FAILED_WORKER_LEFT_DESCENDANTS'}
   if evidence.get('status') not in terminal:raise ValueError('Inherited worker cleanup is ambiguous')
   if item['outcome']=='PASS' and (evidence.get('status')!='PASS_PROCESS_EXIT' or evidence.get('supervisor_returncode')!=0):raise ValueError('Inherited PASS lacks successful supervisor exit')
   if job.get('maintenance'):
    m=item.get('maintenance',{});spec=job['maintenance']
    if m.get('status')!='PASS' or m.get('returncode')!=0 or not m.get('worker_gpu_lock_observed_released'):raise ValueError('Inherited maintenance did not complete')
    if m.get('command_sha256')!=spec['command_sha256']:raise ValueError('Inherited maintenance command changed')
    if not m.get('receipt') or m['receipt']['path']!=spec.get('receipt'):raise ValueError('Missing inherited maintenance receipt')
    check_bindings([m['receipt']]);proofs.append(m['receipt']);r=json.loads(Path(m['receipt']['path']).read_text())
    if r.get('status') not in spec['expected_statuses']:raise ValueError('Inherited maintenance receipt failed')
    if job['id'] in inherited_maintenance:
     q_binding=inherited_maintenance[job['id']];check_bindings([q_binding]);q=Path(q_binding['path'])
     if q.name!=job['id']+'.maintenance.log.supervision.json':raise ValueError('Inherited maintenance proof identity changed')
    else:
     if old.get('schema') in ('campaign.queue.v7','campaign.queue.v8'):raise ValueError('Resumed maintenance lacks its retained proof binding')
     q=old_path.parent/(job['id']+'.maintenance.log.supervision.json')
     if not q.is_file():raise ValueError('Inherited maintenance lacks cleanup report')
     q_binding=bind(q)
    e=json.loads(q.read_text())
    if e.get('status')!='PASS_PROCESS_EXIT' or e.get('supervisor_returncode')!=0 or e.get('surviving_owned_processes') or 'finished_monotonic' not in e:raise ValueError('Inherited maintenance cleanup is incomplete')
    proofs.append(q_binding);maintenance_proofs[job['id']]=q_binding
  elif item.get('maintenance') is not None:raise ValueError('Nonexecuted inherited job has maintenance')
  outcomes[item['id']]=item['outcome']
 for job in jobs[len(completed):]:
  # A partial receipt directory or log is evidence of a previous attempt.
  # This API only resumes work which never started, regardless of its outcome.
  if Path(job['receipt']).parent.exists():raise ValueError('Remaining job has an existing/partial output directory: '+job['id'])
  for suffix in ('.log','.log.supervision.json','.maintenance.log','.maintenance.log.supervision.json'):
   if (old_path.parent/(job['id']+suffix)).exists():raise ValueError('Remaining job has an old attempt log: '+job['id'])
  check_bindings(job.get('bindings',[]))
  if job.get('maintenance'):
   check_maintenance(job['maintenance']);check_bindings(job['maintenance']['bindings'])
   if job['maintenance'].get('receipt') and Path(job['maintenance']['receipt']).parent.exists():raise ValueError('Remaining job has partial maintenance output')
 admission=dict(schema='campaign.resume.admission.v1',status='PASS_EXACT_COMPLETED_PREFIX',parent_state=bind(old_path),
                original_plan=bind(plan_path),inherited_completed=len(completed),remaining_jobs=len(jobs)-len(completed),
                preserved_failed_jobs=sum(x['outcome']=='FAIL' for x in completed),proofs=proofs,maintenance_supervision_bindings=maintenance_proofs,
                retry_policy='No retries: only jobs with no receipt directory, attempt log or maintenance output may start',
                root_liveness_precondition='Root must confirm previous controller/workers exited and use the original global serial lock')
 return old,raw,admission


def canonical_bytes(value):
 return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')


def admit_resume(plan_path,output,resume_state,resume_sha,supervisor,maintenance_migration_from=None):
 if maintenance_migration_from is None:
  return admit_unchanged_resume(plan_path,output,resume_state,resume_sha,supervisor)
 if resume_state is None or resume_sha is None:raise ValueError('Maintenance migration requires an exact resume checkpoint')
 parent_path=Path(resume_state);raw=parent_path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=resume_sha:raise ValueError('Resume state SHA changed')
 parent=json.loads(raw);old_binding=parent['plan']
 if maintenance_migration_from!=old_binding['sha256']:raise ValueError('Explicit old plan SHA does not match checkpoint')
 check_bindings([old_binding]);old_plan=json.loads(Path(old_binding['path']).read_text());new_plan=json.loads(Path(plan_path).read_text())
 if Path(plan_path).resolve()==Path(old_binding['path']).resolve():raise ValueError('Migration requires a distinct new plan file')
 old_top={k:v for k,v in old_plan.items() if k not in ('jobs','maintenance_migration')};new_top={k:v for k,v in new_plan.items() if k not in ('jobs','maintenance_migration')}
 if canonical_bytes(old_top)!=canonical_bytes(new_top):raise ValueError('Migration changed non-maintenance plan metadata')
 if len(old_plan['jobs'])!=len(new_plan['jobs']):raise ValueError('Migration changed job count')
 n=len(parent['completed']);changes=[]
 for index,(old_job,new_job) in enumerate(zip(old_plan['jobs'],new_plan['jobs'])):
  if index<n:
   if canonical_bytes(old_job)!=canonical_bytes(new_job):raise ValueError('Migration changed a completed job')
   continue
  a={k:v for k,v in old_job.items() if k!='maintenance'};b={k:v for k,v in new_job.items() if k!='maintenance'}
  if canonical_bytes(a)!=canonical_bytes(b):raise ValueError('Migration changed an unstarted worker contract')
  old_spec=old_job.get('maintenance');new_spec=new_job.get('maintenance')
  if canonical_bytes(old_spec)==canonical_bytes(new_spec):continue
  if not old_spec or not new_spec:raise ValueError('Migration cannot add or remove required maintenance')
  check_maintenance(old_spec);check_maintenance(new_spec);check_bindings(new_spec['bindings'])
  if new_spec['gpu_lock']!=old_spec['gpu_lock']:raise ValueError('Migration changed the global GPU lock')
  if new_spec.get('receipt') and Path(new_spec['receipt']).parent.exists():raise ValueError('Migrated maintenance output already exists')
  changes.append(dict(id=new_job['id'],old_maintenance_sha256=hashlib.sha256(canonical_bytes(old_spec)).hexdigest(),new_maintenance_sha256=hashlib.sha256(canonical_bytes(new_spec)).hexdigest()))
 if not changes:raise ValueError('Explicit migration contains no maintenance changes')
 # First validate the complete original checkpoint, including all receipts,
 # counters, old helpers and no-start evidence. Never rewrite the parent.
 state,raw,admission=admit_unchanged_resume(old_binding['path'],output,resume_state,resume_sha,supervisor)
 migration=dict(schema='campaign.maintenance.migration.v1',status='PASS_MAINTENANCE_ONLY_MIGRATION',old_plan=old_binding,new_plan=bind(plan_path),
                parent_state=bind(parent_path),changed_jobs=changes,preserved_completed_jobs=n,
                invariant='Completed entire jobs unchanged; unstarted jobs differ only in maintenance; worker argv, env, bindings, receipt, phase, deps and every other field are canonical-byte identical.')
 admission['maintenance_migration']=migration
 state['plan']=bind(plan_path);state['jobs']=new_plan['jobs']
 return state,raw,admission


def run(plan_path,output,controller_lock,resume_state=None,resume_sha=None,maintenance_migration_from=None):
 if maintenance_migration_from is not None and (resume_state is None or resume_sha is None):raise ValueError('Maintenance migration requires a resume state and SHA')
 plan_path=Path(plan_path);output=Path(output);controller_lock=Path(controller_lock);plan=json.loads(plan_path.read_text())
 if plan.get('postprocess_contract')!='e4s_recoverable_outputs_v1':raise ValueError('This controller requires mandatory E4S recovery postprocessing')
 if resume_state is not None or resume_sha is not None:raise ValueError('Explicit reviewed recovery/continuation plan required; no automatic resume or worker replay')
 jobs=plan['jobs'];known=set();phases={}
 poll=plan.get('storage_poll_seconds',5)
 if not isinstance(poll,(int,float)) or not 0<poll<=60:raise ValueError('Storage poll interval must be in (0,60] seconds')
 for job in jobs:
  name=job['id']
  if Path(job['receipt']).parent.exists():raise ValueError('Existing worker output forbids replay; use explicit postprocessing recovery and reviewed continuation')
  if not re.fullmatch(r'[A-Za-z0-9_.-]+',name) or name in known:raise ValueError('Unsafe/duplicate job identity')
  if not set(job.get('depends_on',[]))<=known:raise ValueError('Dependencies must precede consumer')
  if not isinstance(job['command'],list) or not job['command'] or any(not isinstance(x,str) for x in job['command']):raise ValueError('Explicit worker argv required')
  phases[name]=phase_of(job);known.add(name)
  if phases[name]!='formal' or job.get('required_postprocess')!='e4s_recoverable_outputs_v1' or not job.get('maintenance'):raise ValueError('Every job must be formal E4S with required recovery maintenance')
  if 'disk_guard' in job:
   guard=job['disk_guard']
   if not isinstance(guard.get('path'),str) or type(guard.get('min_free_bytes')) is not int or guard['min_free_bytes']<0:raise ValueError('Invalid disk guard')
  if job.get('maintenance'):check_maintenance(job['maintenance'])
 resumed=None
 if resume_state is not None or resume_sha is not None:
  resumed=admit_resume(plan_path,output,resume_state,resume_sha,Path(__file__).with_name('owned_process_supervisor_v2.py'),maintenance_migration_from)
 output.mkdir(parents=True,exist_ok=False)
 state=dict(schema='campaign.e4s.recoverable.queue.v1',status='WAITING_CONTROLLER_LOCK',plan=bind(plan_path),controller=bind(__file__),supervisor=bind(Path(__file__).with_name('owned_process_supervisor_v2.py')),jobs=jobs,job_phases=phases,completed=[],attempted=0,passed=0,failed=0,skipped_dependency=0,failed_prelaunch=0,formal_planned=sum(p=='formal' for p in phases.values()),formal_attempted=0,formal_passed=0,formal_failed_attempts=0,formal_measurements=0,formal_measurements_unit='formal jobs with worker, byte-identical recoverable output, and original maintenance all successful',worker_passed=0,worker_failed=0,archive_passed=0,archive_failed=0,original_maintenance_passed=0,original_maintenance_failed=0,phase_counts={},maintenance_attempted=0,maintenance_passed=0,maintenance_failed=0,storage_wait_observations=0)
 for phase in set(phases.values()):state['phase_counts'][phase]=dict(planned=sum(p==phase for p in phases.values()),attempted=0,passed=0,failed=0,failed_prelaunch=0,skipped_dependency=0)
 if resumed is not None:
  state,parent_raw,admission=resumed
  (output/'resume_parent_state.json').write_bytes(parent_raw)
  (output/'resume_admission.json').write_text(json.dumps(admission,indent=2)+'\n')
  lineage=list(state.get('controller_lineage',[]))
  lineage.append(dict(controller=state['controller'],state=admission['parent_state']))
  state.update(schema='campaign.queue.v8',controller=bind(__file__),supervisor=bind(Path(__file__).with_name('owned_process_supervisor_v2.py')),
               controller_lineage=lineage,resume_admission=bind(output/'resume_admission.json'),inherited_completed=admission['inherited_completed'],maintenance_supervision_bindings=admission['maintenance_supervision_bindings'])
  state.pop('finished_unix',None)
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
 save();outcomes={item['id']:item['outcome'] for item in state['completed']}
 try:
  for job in jobs:
   if job['id'] in outcomes:continue
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
      item['worker_outcome']=item['outcome']
      if item.get('execution_started'):state['worker_passed' if item['outcome']=='PASS' else 'worker_failed']+=1
      state['pending_postprocessing']=item;save()
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
         if item['worker_outcome']=='PASS':
          if r.get('status')!='PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE' or r.get('archive_status')!='PASS' or r.get('original_maintenance_status')!='PASS':raise ValueError('Mandatory recoverable output/original maintenance did not both pass')
          if r.get('original_result')!=item['receipt']:raise ValueError('Postprocessing does not bind exact original worker receipt')
         maintenance['status']='PASS';state['maintenance_passed']+=1;state['archive_passed']+=int(r.get('archive_status')=='PASS');state['original_maintenance_passed']+=int(r.get('original_maintenance_status')=='PASS')
       except BaseException as error:
        maintenance['error']=type(error).__name__+': '+str(error);state['maintenance_failed']+=1;state['status']='FAILED_MAINTENANCE'
        p=Path(spec.get('receipt',''))
        try:post=json.loads(p.read_text()) if p.is_file() else {}
        except (ValueError,OSError):post={}
        maintenance['postprocess_status']=post.get('status','MISSING_OR_UNREADABLE');maintenance['postprocess_error']=post.get('error')
        state['archive_passed']+=int(post.get('archive_status')=='PASS');state['archive_failed']+=int(post.get('archive_status')!='PASS' and item['worker_outcome']=='PASS')
        state['original_maintenance_passed']+=int(post.get('original_maintenance_status')=='PASS');state['original_maintenance_failed']+=int(post.get('archive_status')=='PASS' and post.get('original_maintenance_status')!='PASS')
        item['outcome']='FAIL_REQUIRED_POSTPROCESSING';item['failure_stage']='worker' if item['worker_outcome']!='PASS' else 'archive' if post.get('archive_status')!='PASS' else 'original_maintenance';item['original_worker_status_preserved']=True
        outcomes[job['id']]=item['outcome'];state.pop('pending_postprocessing',None);complete(item,phase);save();raise
       finally:
        maintenance['finished_unix']=time.time()
        proof=output/(job['id']+'.maintenance.log.supervision.json')
        if proof.is_file():state.setdefault('maintenance_supervision_bindings',{})[job['id']]=bind(proof)
        save()
      outcomes[job['id']]=item['outcome'];state.pop('pending_postprocessing',None);complete(item,phase)
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
 p.add_argument('--resume-state',type=Path);p.add_argument('--resume-sha');p.add_argument('--maintenance-migration-from')
 a=p.parse_args();run(a.plan,a.output,a.controller_lock,a.resume_state,a.resume_sha,a.maintenance_migration_from)
