"""CPU-only regression checks: real frozen supervisor, no graph/GPU imports."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile,time,unittest

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parents[1]
SCRATCH=Path(os.environ['EGGPU_TIMEOUT_TEST_SCRATCH'])
SCRATCH.mkdir(parents=True,exist_ok=True)
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Checks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=SCRATCH);self.p=Path(self.tmp.name)
  for name,source in [('run_campaign_queue_v9.py',HERE/'run_campaign_queue_v9.py'),('run_campaign_queue_v8.py',CAMPAIGN/'run_campaign_queue_v8.py'),('owned_process_supervisor_v2.py',CAMPAIGN/'owned_process_supervisor_v2.py')]:shutil.copyfile(source,self.p/name)
  self.q=module(self.p/'run_campaign_queue_v9.py','q9');self.old=module(self.p/'run_campaign_queue_v8.py','q8')
  self.worker=self.p/'worker.py';self.worker.write_text("from pathlib import Path\nimport json,sys,time\ntime.sleep(float(sys.argv[2]));p=Path(sys.argv[1]);p.mkdir();(p/'result.json').write_text(json.dumps({'status':'PASS','phase':'cpu'}));Path(sys.argv[3]).open('a').write(p.name+'\\n')\n")
  self.counter=self.p/'counter';self.lock=self.p/'controller.lock'
 def tearDown(self):self.tmp.cleanup()
 def job(self,name,delay=0,timeout=4):
  out=self.p/name
  return dict(id=name,phase='cpu',command=[sys.executable,'-B',str(self.worker),str(out),str(delay),str(self.counter)],receipt=str(out/'result.json'),expected_statuses=['PASS'],bindings=[self.q.bind(self.worker)],timeout=timeout)
 def plan(self,jobs,name='plan.json'):
  path=self.p/name;path.write_text(json.dumps(dict(jobs=jobs)));return path
 def test_default_and_floor_use_real_frozen_supervisor(self):
  plan=self.plan([self.job('default_timeout',.5,.1)]);original=plan.read_bytes()
  state=self.q.run(plan,self.p/'defaultq',self.lock)
  self.assertEqual(state['completed'][0]['wall_timeout']['effective_seconds'],.1)
  proof=json.loads(Path(state['completed'][0]['process_supervision']['path']).read_text());self.assertEqual(proof['status'],'TIMED_OUT')
  self.assertEqual(original,plan.read_bytes())
  plan=self.plan([self.job('floor_pass',.5,.1)],'floor.json');original=plan.read_bytes()
  state=self.q.run(plan,self.p/'floorq',self.lock,wall_timeout_floor=3)
  self.assertEqual(state['status'],'PASS_REQUESTED_JOBS');self.assertEqual(state['completed'][0]['wall_timeout']['original_seconds'],.1)
  self.assertEqual(state['completed'][0]['wall_timeout']['effective_seconds'],3)
  self.assertEqual(original,plan.read_bytes())
 def test_floor_does_not_lower_original_or_change_maintenance_timeout(self):
  job=self.job('long',timeout=4);gpu=self.p/'gpu.lock';gpu.touch();maint=self.job('maint',timeout=1)
  job['maintenance']=dict(command=maint['command'],command_sha256=self.q.command_digest(maint['command']),bindings=[self.q.bind(Path(sys.executable).resolve()),self.q.bind(self.worker)],gpu_lock=str(gpu),receipt=maint['receipt'],expected_statuses=['PASS'],timeout=1)
  seen=[];original=self.q.run_process
  def observe(command,env,log,timeout,on_start):
   seen.append(timeout);return original(command,env,log,timeout,on_start)
  self.q.run_process=observe
  state=self.q.run(self.plan([job]),self.p/'queue',self.lock,wall_timeout_floor=3)
  self.assertEqual(seen,[4,1]);self.assertEqual(state['maintenance_passed'],1)
 def test_exact_prefix_resume_preserves_plan_checkpoint_and_completed(self):
  jobs=[self.job('one'),self.job('two')];prefix=self.plan(jobs[:1],'prefix.json')
  old=self.old.run(prefix,self.p/'oldqueue',self.lock)
  plan=self.plan(jobs);old.update(status='WAITING_CONTROLLER_LOCK',plan=self.q.bind(plan),jobs=jobs,active_job='two',job_phases={'one':'cpu','two':'cpu'});old['phase_counts']['cpu']['planned']=2
  checkpoint=self.p/'checkpoint.json';checkpoint.write_text(json.dumps(old));raw=checkpoint.read_bytes();plan_raw=plan.read_bytes()
  state=self.q.run(plan,self.p/'resumed',self.lock,checkpoint,self.q.bind(checkpoint)['sha256'],wall_timeout_floor=5)
  self.assertEqual(state['completed'][:1],old['completed']);self.assertEqual(self.counter.read_text().splitlines(),['one','two'])
  self.assertEqual(state['controller_lineage'][0]['controller'],old['controller']);self.assertEqual(state['supervisor'],old['supervisor'])
  self.assertEqual(state['completed'][1]['wall_timeout']['effective_seconds'],5)
  self.assertEqual(checkpoint.read_bytes(),raw);self.assertEqual(plan.read_bytes(),plan_raw)
  self.assertEqual((self.p/'resumed/resume_parent_state.json').read_bytes(),raw)
 def test_invalid_floor_rejected_before_output(self):
  plan=self.plan([self.job('worker')])
  for value in (0,-1,float('nan'),float('inf'),21601,True):
   with self.assertRaises(ValueError):self.q.run(plan,self.p/'output',self.lock,wall_timeout_floor=value)
  self.assertFalse((self.p/'output').exists())
 def test_launchers_limit_floor_to_selected_stage(self):
  helper=module(HERE/'after_main_supervise_v2.py','resume_helper')
  config=dict(python=sys.executable,controller='v8.py',controller_lock='lock')
  stage=dict(mode='start',plan='plan',queue='queue')
  default=helper.command(config,stage);self.assertNotIn('--wall-timeout-floor',default);self.assertIn('v8.py',default)
  stage.update(controller='v9.py',wall_timeout_floor=10800)
  amended=helper.command(config,stage);self.assertIn('v9.py',amended);self.assertEqual(amended[-2:],['--wall-timeout-floor','10800'])
  for path,args in [(HERE/'run_campaign_queue_v9.py',[]),(HERE/'after_main_supervise_v2.py',[]),(HERE/'state_tail_successor_v2.py',['prepare'])]:
   result=subprocess.run([sys.executable,'-B',str(path),*args,'--help'],capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
  for name in ('run_campaign_queue_v9.py','after_main_supervise_v2.py','state_tail_successor_v2.py','bootstrap_after_main_v2.py'):compile((HERE/name).read_text(),name,'exec')

if __name__=='__main__':
 begin=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);r=unittest.TextTestRunner(verbosity=2).run(suite)
 receipt=dict(schema='campaign.timeout-floor.cpu-checks.v1',status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),elapsed_seconds=time.monotonic()-begin,gpu_executed=False,remote_changes=False)
 (HERE/'timeout_floor_cpu_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');raise SystemExit(0 if r.wasSuccessful() else 1)
