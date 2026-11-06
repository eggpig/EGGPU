"""One real CPU-only pause/resume fixture, including completed maintenance."""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, tempfile, time
HERE=Path(__file__).resolve().parent
C=HERE.parents[1]
def read(p):return json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def wait(predicate,seconds=30):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  if predicate():return
  time.sleep(.05)
 raise AssertionError('CPU fixture timed out')
def main():
 scratch=Path(os.environ['EGGPU_AUTHOR_TEST_SCRATCH']);scratch.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=scratch) as temp:
  p=Path(temp);h=p/'first_pass_20261106';h.mkdir();old=h/'author_text_main/queue';future=h/'author_text_after_main/queue';lock=p/'root_controller.lock';lock.touch();(p/'gpu.lock').touch()
  for name in ('run_campaign_queue_v8.py','owned_process_supervisor_v2.py'):shutil.copyfile(C/name,p/name)
  worker=p/'worker.py';worker.write_text("from pathlib import Path\nimport sys,json,time\ntime.sleep(float(sys.argv[2]));p=Path(sys.argv[1]);p.mkdir();(p/'result.json').write_text(json.dumps({'status':'PASS','phase':'cpu'}));Path(sys.argv[3]).open('a').write(p.name+'\\n')\n")
  counter=p/'counter';jobs=[]
  for i in (1,2):
   out=p/('job'+str(i));jobs.append(dict(id='job'+str(i),phase='cpu',command=[sys.executable,'-B',str(worker),str(out),'2' if i==1 else '0',str(counter)],receipt=str(out/'result.json'),expected_statuses=['PASS'],bindings=[bind(worker)],timeout=10))
  maintenance=p/'maint.py';maintenance.write_text("from pathlib import Path\nimport sys,json\np=Path(sys.argv[1]);p.mkdir();(p/'result.json').write_text(json.dumps({'status':'PASS'}))\n")
  mc=[sys.executable,'-B',str(maintenance),str(p/'maintenance')]
  jobs[0]['maintenance']=dict(command=mc,command_sha256=hashlib.sha256(json.dumps(mc,separators=(',',':')).encode()).hexdigest(),bindings=[bind(Path(sys.executable)),bind(maintenance)],gpu_lock=str(p/'gpu.lock'),receipt=str(p/'maintenance/result.json'),expected_statuses=['PASS'],timeout=10)
  plan=p/'plan.json';plan.write_text(json.dumps(dict(jobs=jobs)));before=plan.read_bytes()
  manifest=h/'author_reservation_handoff/stage_manifest.json';manifest.parent.mkdir();manifest.write_text(json.dumps(dict(queues=[str(h/'old_empty/queue'),str(old)],stages=[dict(queue=str(old))])))
  logs=[];children=[]
  try:
   log=(p/'old.log').open('w');logs.append(log);owner=subprocess.Popen([sys.executable,'-B',str(p/'run_campaign_queue_v8.py'),'--plan',str(plan),'--output',str(old),'--controller-lock',str(lock)],stdout=log,stderr=log);children.append(owner)
   wait(lambda:(old/'state.json').exists() and read(old/'state.json').get('worker_started_for_active_job'))
   log=(p/'helper.log').open('w');logs.append(log);helper=subprocess.Popen([sys.executable,'-B',str(HERE/'pause_then_resume_author.py'),'--campaign',str(p),'--pid',str(owner.pid),'--mode','run','--poll-seconds','1'],stdout=log,stderr=log);children.append(helper)
   statepath=h/'author_text_after_main/priority_state.json'
   wait(lambda:statepath.exists() and read(statepath)['status']=='PAUSED_WAITING_MAIN_AND_MISSING')
   paused=read(statepath);assert paused['completed']==1 and paused['attempted']==1;assert owner.wait(timeout=2)!=0
   assert read(old/'state.json')['completed']==read(Path(paused['waiting_checkpoint']['path']))['completed']
   assert read(manifest)['queues']==[str(h/'old_empty/queue'),str(future)] and read(manifest)['stages'][0]['queue']==str(future)
   assert not future.exists()
   for name in ('unified_main_continuation','missing_external_continuation'):
    path=h/name/'queue/state.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(dict(status='PASS_REQUESTED_JOBS')))
   assert helper.wait(timeout=20)==0,(p/'helper.log').read_text()
   wait(lambda:(future/'state.json').exists() and read(future/'state.json')['status']=='PASS_REQUESTED_JOBS')
   resumed=read(future/'state.json');assert resumed['inherited_completed']==1 and len(resumed['completed'])==2
   assert counter.read_text().splitlines()==['job1','job2'];assert plan.read_bytes()==before
   receipt=dict(status='PASS_CPU_EXACT_AUTHOR_PAUSE_RESUME',helper=bind(HERE/'pause_then_resume_author.py'),real_frozen_v8_and_supervisor=True,maintenance_evidence_parent_preserved=True,queue_list_and_stage_pointers_updated=True,completed_prefix_preserved=True,no_completed_job_rerun=True,original_plan_bytes_unchanged=True,main_and_missing_terminal_required=True,gpu_executed=False)
   (HERE/'cpu_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
  finally:
   for child in children:
    if child.poll() is None:child.terminate();child.wait(timeout=10)
   for log in logs:log.close()
if __name__=='__main__':main()
