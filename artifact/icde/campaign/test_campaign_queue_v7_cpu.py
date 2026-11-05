"""CPU-only resume tests using real child processes and the frozen supervisor."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile,unittest
D=Path(__file__).resolve().parent

def module(name):
 spec=importlib.util.spec_from_file_location(name,D/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v6=module('run_campaign_queue_v6');v7=module('run_campaign_queue_v7')
ROOT=D.parents[2]/'EGGPU_TMP/EGGPU_ICDE_20261017/artifact_recovery_20261105_v5/resume_cpu'
# Tests may be invoked from a copied remote code directory; override only the scratch path.
ROOT=Path(os.environ.get('EGGPU_RESUME_TEST_SCRATCH',str(ROOT)))
ROOT.mkdir(parents=True,exist_ok=True)

class ResumeTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=ROOT);self.p=Path(self.tmp.name)
  self.worker=self.p/'cpu_worker.py'
  self.worker.write_text("from pathlib import Path\nimport sys,json\np=Path(sys.argv[1]);p.mkdir();c=Path(sys.argv[2]);c.open('a').write(sys.argv[3]+'\\n');fail=sys.argv[4]=='fail';(p/'result.json').write_text(json.dumps({'status':'FAIL' if fail else 'PASS','phase':'target'}));raise SystemExit(1 if fail else 0)\n")
  self.counter=self.p/'executed.txt';self.lock=self.p/'controller.lock'
 def tearDown(self):self.tmp.cleanup()
 def fixture(self,failed=False,dependent=False,count=2,maintenance=False):
  jobs=[]
  if maintenance:
   maint=self.p/'cpu_maint.py';maint.write_text("from pathlib import Path\nimport json,sys\np=Path(sys.argv[1]);p.mkdir();(p/'receipt.json').write_text(json.dumps({'status':'PASS_MAINT'}))\n")
   gpu=self.p/'gpu.lock';gpu.touch()
  for i in range(count):
   out=self.p/('result'+str(i));j=dict(id='j'+str(i),phase='target',command=[sys.executable,'-B',str(self.worker),str(out),str(self.counter),'j'+str(i),'fail' if failed and i==0 else 'pass'],receipt=str(out/'result.json'),expected_statuses=['PASS'],bindings=[v7.bind(self.worker)],timeout=15,depends_on=['j0'] if i and dependent else [])
   if maintenance:
    mo=self.p/('maint'+str(i));cmd=[sys.executable,'-B',str(maint),str(mo)]
    j['maintenance']=dict(command=cmd,command_sha256=v6.command_digest(cmd),bindings=[v6.bind(Path(sys.executable).resolve()),v6.bind(maint)],gpu_lock=str(gpu),receipt=str(mo/'receipt.json'),expected_statuses=['PASS_MAINT'],timeout=15)
   jobs.append(j)
  prefix=self.p/'prefix_plan.json';prefix.write_text(json.dumps({'jobs':jobs[:1]}));v6.run(prefix,self.p/'prefix',self.lock)
  plan=self.p/'plan.json';plan.write_text(json.dumps({'jobs':jobs}));old=json.loads((self.p/'prefix/state.json').read_text())
  old.update(status='WAITING_CONTROLLER_LOCK',plan=v7.bind(plan),jobs=jobs,active_job='j1',job_phases={j['id']:'target' for j in jobs})
  old['phase_counts']['target']['planned']=count
  checkpoint=self.p/'checkpoint';checkpoint.mkdir();state=checkpoint/'state.json';state.write_text(json.dumps(old,indent=2)+'\n')
  if maintenance:shutil.copyfile(self.p/'prefix/j0.maintenance.log.supervision.json',checkpoint/'j0.maintenance.log.supervision.json')
  return plan,state
 def resume(self,plan,state):return v7.run(plan,self.p/'resumed',self.lock,state,v7.bind(state)['sha256'])
 def mutate(self,state,fn):
  d=json.loads(state.read_text());fn(d);state.write_text(json.dumps(d))
 def test_real_success_prefix_is_not_rerun(self):
  plan,state=self.fixture();original=state.read_bytes();result=self.resume(plan,state)
  self.assertEqual(self.counter.read_text().splitlines(),['j0','j1']);self.assertEqual(result['passed'],2);self.assertEqual(result['attempted'],2)
  self.assertEqual(state.read_bytes(),original);self.assertEqual((self.p/'resumed/resume_parent_state.json').read_bytes(),original)
  self.assertEqual(result['completed'][:1],json.loads(original)['completed']);self.assertEqual(result['plan'],v7.bind(plan));self.assertEqual(result['inherited_completed'],1)
 def test_failed_prefix_is_not_retried_and_dependency_is_skipped(self):
  plan,state=self.fixture(failed=True,dependent=True);result=self.resume(plan,state)
  self.assertEqual(self.counter.read_text().splitlines(),['j0']);self.assertEqual(result['failed'],1);self.assertEqual(result['skipped_dependency'],1);self.assertEqual(result['attempted'],1)
  self.assertFalse((self.p/'result1').exists());self.assertEqual(result['status'],'COMPLETE_WITH_FAILURES')
 def test_modified_completed_proof_is_rejected(self):
  plan,state=self.fixture();(self.p/'result0/result.json').write_text('{"status":"PASS","phase":"target","changed":true}')
  with self.assertRaisesRegex(ValueError,'binding changed'):self.resume(plan,state)
  self.assertFalse((self.p/'resumed').exists());self.assertEqual(self.counter.read_text().splitlines(),['j0'])
 def test_partial_remaining_output_is_rejected(self):
  plan,state=self.fixture();(self.p/'result1').mkdir()
  with self.assertRaisesRegex(ValueError,'partial output'):self.resume(plan,state)
 def test_partial_remaining_log_is_rejected(self):
  plan,state=self.fixture();(state.parent/'j1.log').write_text('partial')
  with self.assertRaisesRegex(ValueError,'old attempt log'):self.resume(plan,state)
 def test_unknown_maintenance_is_rejected(self):
  plan,state=self.fixture();p=json.loads(plan.read_text());p['jobs'][0]['maintenance']={'command_sha256':'old','receipt':str(self.p/'unknown-maint/result.json')};plan.write_text(json.dumps(p))
  def edit(d):
   d['plan']=v7.bind(plan);d['jobs']=p['jobs'];d['maintenance_attempted']=1;d['maintenance_passed']=1;d['completed'][0]['maintenance']={'status':'RUNNING'}
  self.mutate(state,edit)
  with self.assertRaisesRegex(ValueError,'maintenance did not complete'):v7.admit_resume(plan,self.p/'resumed',state,v7.bind(state)['sha256'],D/'owned_process_supervisor_v2.py')
 def test_started_unrecorded_attempt_is_rejected(self):
  plan,state=self.fixture();self.mutate(state,lambda d:d.update(attempted=2))
  with self.assertRaisesRegex(ValueError,'counter mismatch'):self.resume(plan,state)
 def test_running_checkpoint_is_rejected(self):
  plan,state=self.fixture();self.mutate(state,lambda d:d.update(status='RUNNING'))
  with self.assertRaisesRegex(ValueError,'between-job'):self.resume(plan,state)
 def test_changed_plan_is_rejected(self):
  plan,state=self.fixture();p=json.loads(plan.read_text());p['jobs'][1]['timeout']=16;plan.write_text(json.dumps(p))
  with self.assertRaisesRegex(ValueError,'unchanged complete original plan'):self.resume(plan,state)
 def test_wrong_checkpoint_sha_is_rejected(self):
  plan,state=self.fixture()
  with self.assertRaisesRegex(ValueError,'SHA changed'):v7.run(plan,self.p/'resumed',self.lock,state,'0'*64)
 def test_existing_new_queue_is_rejected(self):
  plan,state=self.fixture();(self.p/'resumed').mkdir()
  with self.assertRaises(FileExistsError):self.resume(plan,state)
 def test_cleanup_uncertainty_is_rejected(self):
  plan,state=self.fixture();d=json.loads(state.read_text());p=Path(d['completed'][0]['process_supervision']['path']);e=json.loads(p.read_text());e['surviving_owned_processes']=[{'pid':123}];p.write_text(json.dumps(e));d['completed'][0]['process_supervision']=v7.bind(p);state.write_text(json.dumps(d))
  with self.assertRaisesRegex(ValueError,'cleanup is incomplete'):self.resume(plan,state)

 def test_unknown_supervisor_terminal_status_is_rejected(self):
  plan,state=self.fixture();d=json.loads(state.read_text());p=Path(d['completed'][0]['process_supervision']['path']);e=json.loads(p.read_text());e['status']='UNRECOGNIZED_TERMINAL';p.write_text(json.dumps(e));d['completed'][0]['process_supervision']=v7.bind(p);state.write_text(json.dumps(d))
  with self.assertRaisesRegex(ValueError,'cleanup is ambiguous'):self.resume(plan,state)
 def test_pass_with_nonzero_supervisor_exit_is_rejected(self):
  plan,state=self.fixture();d=json.loads(state.read_text());p=Path(d['completed'][0]['process_supervision']['path']);e=json.loads(p.read_text());e['supervisor_returncode']=1;p.write_text(json.dumps(e));d['completed'][0]['process_supervision']=v7.bind(p);state.write_text(json.dumps(d))
  with self.assertRaisesRegex(ValueError,'successful supervisor exit'):self.resume(plan,state)
 def test_two_real_resumes_preserve_prior_maintenance_proofs(self):
  plan,state=self.fixture(count=3,maintenance=True)
  # Inject abrupt process exit exactly after the second completed job has
  # finished maintenance and the controller has published its next WAITING
  # checkpoint. Real workers and the frozen supervisor execute both attempts.
  wrapper=self.p/'interrupt_at_wait.py'
  wrapper.write_text("from pathlib import Path\nimport importlib.util,sys,json,os,fcntl\ns=importlib.util.spec_from_file_location('queue',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\noutput=Path(sys.argv[3]);original=m.fcntl.flock\ndef at_wait(fd,op):\n p=output/'state.json'\n if op==fcntl.LOCK_EX and p.exists():\n  d=json.loads(p.read_text())\n  if d['status']=='WAITING_CONTROLLER_LOCK' and len(d['completed'])==2:os._exit(0)\n return original(fd,op)\nm.fcntl.flock=at_wait\nm.run(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])\n")
  first=self.p/'first_resume';r=subprocess.run([sys.executable,'-B',str(wrapper),str(D/'run_campaign_queue_v7.py'),str(plan),str(first),str(self.lock),str(state),v7.bind(state)['sha256']],timeout=30)
  self.assertEqual(r.returncode,0);second_state=first/'state.json';d=json.loads(second_state.read_text());self.assertEqual(d['status'],'WAITING_CONTROLLER_LOCK');self.assertEqual(len(d['completed']),2);self.assertFalse((self.p/'result2').exists())
  self.assertEqual(set(d['maintenance_supervision_bindings']),{'j0','j1'})
  self.assertNotEqual(Path(d['maintenance_supervision_bindings']['j0']['path']).parent,first)
  result=v7.run(plan,self.p/'second_resume',self.lock,second_state,v7.bind(second_state)['sha256'])
  self.assertEqual(self.counter.read_text().splitlines(),['j0','j1','j2']);self.assertEqual(result['passed'],3);self.assertEqual(result['maintenance_passed'],3);self.assertEqual(result['completed'][:2],d['completed'])
  self.assertEqual(set(result['maintenance_supervision_bindings']),{'j0','j1','j2'});self.assertEqual(len(result['controller_lineage']),2)

if __name__=='__main__':unittest.main(verbosity=2)
