from pathlib import Path
from unittest import mock
import fcntl,json,os,subprocess,sys,tempfile,time,unittest
import run_campaign_queue_v4 as q

WORKER='''from pathlib import Path
import argparse,json,sys
p=argparse.ArgumentParser();p.add_argument('--phase');p.add_argument('--receipt');p.add_argument('--status',default='PASS');p.add_argument('--exit',type=int,default=0);a=p.parse_args()
Path(a.receipt).write_text(json.dumps(dict(status=a.status,phase=a.phase,gpu_executed=False)))
raise SystemExit(a.exit)
'''
MAINT='''from pathlib import Path
import fcntl,json,sys
# The child can acquire the GPU lock itself; controller must have released its probe.
with Path(sys.argv[1]).open('a+') as gpu:
 fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
 blocked=False
 with Path(sys.argv[2]).open('a+') as owner:
  try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:blocked=True
 assert blocked,'root controller lock must stay held'
 Path(sys.argv[3]).write_text(json.dumps(dict(status='PASS_MAINTENANCE',gpu_executed=False)))
'''

class QueueTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);self.worker=self.root/'worker.py';self.worker.write_text(WORKER);self.controller=self.root/'controller.lock';self.controller.touch();self.gpu=self.root/'gpu.lock';self.gpu.touch()
 def tearDown(self):self.t.cleanup()
 def job(self,name,phase='formal',code=0):
  receipt=self.root/(name+'.json');return dict(id=name,phase=phase,command=[sys.executable,'-B',str(self.worker),'--phase',phase,'--receipt',str(receipt),'--exit',str(code)],receipt=str(receipt),expected_statuses=['PASS'],timeout=5,bindings=[q.bind(self.worker)])
 def run_plan(self,jobs,name='run',**extras):
  plan=self.root/(name+'_plan.json');plan.write_text(json.dumps(dict(jobs=jobs,**extras)));return q.run(plan,self.root/name,self.controller)
 def test_actual_pass_failure_dependency_and_formal_counts(self):
  a=self.job('a');b=self.job('b',code=2);c=self.job('c');c['depends_on']=['b'];d=self.job('d','target')
  r=self.run_plan([a,b,c,d]);self.assertEqual((r['attempted'],r['passed'],r['failed'],r['skipped_dependency']),(3,2,1,1));self.assertEqual((r['formal_planned'],r['formal_attempted'],r['formal_passed'],r['formal_failed_attempts'],r['formal_measurements']),(3,2,1,1,1));self.assertFalse((self.root/'c.json').exists())
 def test_storage_wait_releases_controller_and_launches_once(self):
  job=self.job('wait');job['disk_guard']=dict(path=str(self.root),min_free_bytes=10);observations=[]
  def disk(guard):
   observations.append(1);return dict(ready=len(observations)>1,free_bytes=0 if len(observations)==1 else 20,**guard)
  real_sleep=time.sleep
  def pause(seconds):
   if seconds==.01:
    state=json.loads((self.root/'run/state.json').read_text());self.assertEqual(state['status'],'WAITING_STORAGE');self.assertEqual(state['formal_attempted'],0);self.assertFalse(Path(job['receipt']).exists())
    with self.controller.open('a+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   else:real_sleep(seconds)
  with mock.patch.object(q,'disk_available',side_effect=disk),mock.patch.object(q.time,'sleep',side_effect=pause):r=self.run_plan([job],storage_poll_seconds=.01)
  self.assertEqual((r['formal_attempted'],r['formal_measurements'],r['storage_wait_observations']),(1,1,1));self.assertEqual(len(r['completed']),1)
 def test_prelaunch_hash_error_is_not_attempted_measurement(self):
  job=self.job('bad');job['bindings'][0]['sha256']='0'*64;r=self.run_plan([job]);self.assertEqual((r['attempted'],r['formal_attempted'],r['formal_measurements'],r['failed_prelaunch']),(0,0,0,1))
 def test_phase_consistency_and_equal_style(self):
  job=self.job('mismatch');job['phase']='target'
  with self.assertRaises(ValueError):self.run_plan([job])
  self.assertEqual(q.phase_of(dict(command=['x','--phase=formal'])),'formal')
  with self.assertRaises(ValueError):q.phase_of(dict(command=['x','--phase','formal','--phase=formal']))
 def maintenance(self):
  script=self.root/'maint.py';script.write_text(MAINT);out=self.root/'maint.json';argv=[sys.executable,'-B',str(script),str(self.gpu),str(self.controller),str(out)]
  return dict(command=argv,command_sha256=q.command_digest(argv),bindings=[q.bind(sys.executable),q.bind(script)],gpu_lock=str(self.gpu),receipt=str(out),expected_statuses=['PASS_MAINTENANCE'])
 def test_real_maintenance_child_sees_locks_and_bound_code(self):
  job=self.job('with_maintenance');job['maintenance']=self.maintenance();r=self.run_plan([job]);self.assertEqual((r['maintenance_attempted'],r['maintenance_passed'],r['formal_measurements']),(1,1,1));self.assertTrue(r['completed'][0]['maintenance']['worker_gpu_lock_observed_released'])
 def test_bad_maintenance_sha_stops_without_relabeling_worker(self):
  job=self.job('with_bad_maintenance');job['maintenance']=self.maintenance();job['maintenance']['bindings'][1]['sha256']='0'*64
  with self.assertRaises(ValueError):self.run_plan([job])
  state=json.loads((self.root/'run/state.json').read_text());self.assertEqual(state['status'],'FAILED_MAINTENANCE');self.assertEqual((state['formal_attempted'],state['formal_measurements'],state['maintenance_attempted'],state['maintenance_failed']),(1,1,0,1))
 def test_missing_maintenance_binding_rejected_before_worker(self):
  job=self.job('unbound');job['maintenance']=self.maintenance();job['maintenance']['bindings']=[]
  with self.assertRaises(ValueError):self.run_plan([job])
  self.assertFalse(Path(job['receipt']).exists())
 def test_failed_dependency_never_waits_on_storage(self):
  a=self.job('bad',code=1);b=self.job('dependent');b['depends_on']=['bad'];b['disk_guard']=dict(path=str(self.root),min_free_bytes=1<<60)
  with mock.patch.object(q,'disk_available',side_effect=AssertionError('Must skip before storage')):r=self.run_plan([a,b])
  self.assertEqual(r['skipped_dependency'],1);self.assertEqual(r['storage_wait_observations'],0)

if __name__=='__main__':unittest.main()
