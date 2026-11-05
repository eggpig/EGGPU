"""Real CPU subprocesses test ordering/accounting; fake workers never import GPU."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,unittest
import run_e4s_formal_queue_v1 as controller

class QueueOrdering(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.gpu_lock=self.root/'gpu.lock';self.gpu_lock.touch();self.worker=self.root/'worker.py';self.maint=self.root/'maint.py'
  self.worker.write_text('''from pathlib import Path
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--phase');p.add_argument('--output');p.add_argument('--status');a=p.parse_args();folder=Path(a.output);folder.mkdir(exist_ok=False);(folder/'result.json').write_text(json.dumps(dict(status=a.status,phase=a.phase,seconds=0.25)))
''')
  self.maint.write_text('''from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--raw');p.add_argument('--output');p.add_argument('--mode');a=p.parse_args();raw=Path(a.raw);data=raw.read_bytes();status='PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE' if a.mode=='pass' else 'PASS_FAILED_WORKER_CLEANUP' if a.mode=='worker_fail' else 'FAILED_E4S_OUTPUT_MAINTENANCE';r=dict(status=status,original_result=dict(path=str(raw.resolve()),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),archive_status='FAIL' if a.mode=='archive_fail' else 'NOT_APPLICABLE_WORKER_FAILED' if a.mode=='worker_fail' else 'PASS',original_maintenance_status='FAIL' if a.mode=='maintenance_fail' else 'NOT_STARTED' if a.mode=='archive_fail' else 'PASS',error=None if a.mode in ['pass','worker_fail'] else a.mode);Path(a.output).write_text(json.dumps(r));raise SystemExit(0 if a.mode in ['pass','worker_fail'] else 1)
''')
 def tearDown(self):self.tmp.cleanup()
 def run_case(self,mode='pass',worker_status='PASS_E4S_FORMAL',queue='queue'):
  raw=self.root/'work/result.json';post=self.root/'post.json';cmd=[sys.executable,str(self.maint),'--raw',str(raw),'--output',str(post),'--mode',mode]
  binding=controller.bind
  maintenance=dict(command=cmd,command_sha256=controller.command_digest(cmd),bindings=[binding(sys.executable),binding(self.maint)],receipt=str(post),expected_statuses=['PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE','PASS_FAILED_WORKER_CLEANUP'],gpu_lock=str(self.gpu_lock),timeout=10,environment={'CUDA_VISIBLE_DEVICES':''})
  job=dict(id='formal_fixture',phase='formal',command=[sys.executable,str(self.worker),'--phase','formal','--output',str(raw.parent),'--status',worker_status],receipt=str(raw),expected_statuses=['PASS_E4S_FORMAL'],bindings=[binding(sys.executable),binding(self.worker)],depends_on=[],maintenance=maintenance,required_postprocess='e4s_recoverable_outputs_v1',timeout=10)
  plan=self.root/'plan.json';plan.write_text(json.dumps(dict(postprocess_contract='e4s_recoverable_outputs_v1',jobs=[job],environment={'CUDA_VISIBLE_DEVICES':''})));command=[sys.executable,controller.__file__,'--plan',str(plan),'--output',str(self.root/queue),'--controller-lock',str(self.root/'controller.lock')]
  proc=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=25,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'});state_path=self.root/queue/'state.json';state=json.loads(state_path.read_text()) if state_path.exists() else None;return proc,state,job
 def test_success_counted_after_all_postprocessing_pass(self):
  proc,s,_=self.run_case();self.assertEqual(proc.returncode,0,proc.stderr);self.assertEqual(s['formal_passed'],1);self.assertEqual(s['worker_passed'],1);self.assertEqual(s['archive_passed'],1);self.assertEqual(s['original_maintenance_passed'],1);self.assertEqual(s['completed'][0]['worker_outcome'],'PASS');self.assertNotIn('pending_postprocessing',s)
 def test_archive_fail_keeps_worker_pass_but_formal_zero(self):
  proc,s,_=self.run_case('archive_fail');self.assertNotEqual(proc.returncode,0);self.assertEqual(s['formal_passed'],0);self.assertEqual(s['worker_passed'],1);self.assertEqual(s['archive_failed'],1);self.assertEqual(s['completed'][0]['failure_stage'],'archive');self.assertEqual(json.loads((self.root/'work/result.json').read_text())['status'],'PASS_E4S_FORMAL')
 def test_maintenance_fail_then_explicit_cpu_recovery_never_replays_worker(self):
  proc,s,job=self.run_case('maintenance_fail');raw=self.root/'work/result.json';before=raw.read_bytes();self.assertEqual(s['formal_passed'],0);self.assertEqual(s['archive_passed'],1);self.assertEqual(s['original_maintenance_failed'],1);self.assertEqual(s['completed'][0]['failure_stage'],'original_maintenance')
  # Root may explicitly recover CPU postprocessing with a new immutable receipt.
  cmd=list(job['maintenance']['command']);cmd[cmd.index('--output')+1]=str(self.root/'recovery.json');cmd[cmd.index('--mode')+1]='pass';r=subprocess.run(cmd,capture_output=True,text=True,timeout=10);self.assertEqual(r.returncode,0);self.assertEqual(raw.read_bytes(),before);self.assertEqual(json.loads((self.root/'recovery.json').read_text())['status'],'PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE')
  rerun,_,_=self.run_case('pass',queue='attempted_replay');self.assertNotEqual(rerun.returncode,0);self.assertIn('forbids replay',rerun.stderr);self.assertFalse((self.root/'attempted_replay').exists());self.assertEqual(raw.read_bytes(),before);self.assertEqual(json.loads((self.root/'queue/state.json').read_text())['formal_passed'],0)
 def test_worker_oom_keeps_worker_failure_category_not_archive_timeout(self):
  proc,s,_=self.run_case('worker_fail','CUDA_OOM');self.assertEqual(proc.returncode,0,proc.stderr);self.assertEqual(s['worker_failed'],1);self.assertEqual(s['worker_passed'],0);self.assertEqual(s['formal_passed'],0);self.assertEqual(s['archive_failed'],0);self.assertEqual(s['completed'][0]['status'],'CUDA_OOM');self.assertEqual(s['completed'][0]['outcome'],'FAIL')

if __name__=='__main__':unittest.main(verbosity=2)
