from pathlib import Path
import json,unittest
import archive_outputs as a,backfill_qualification as b
import test_archive_outputs

class Backfill(unittest.TestCase):
 def setUp(self):
  self.f=test_archive_outputs.Recoverable();self.f.setUp();self.root=self.f.root;self.plan=self.root/'qual.json';self.plan.write_text(json.dumps(dict(jobs=[dict(id='tiny',phase='tiny',expected_statuses=['PASS_E4S_QUALIFICATION'],receipt=str(self.f.result),command=['worker','--input-root',str(self.f.inputs)])])));self.sha=a.binding(self.plan)['sha256']
 def tearDown(self):self.f.tearDown()
 def test_read_only_inspect_has_no_lock_or_sidecar_writes(self):
  r=b.run(self.plan,self.sha,self.f.store,self.root/'controller.lock');self.assertEqual(r['rows'][0]['status'],'RECOVERY_REQUIRED');self.assertFalse((self.root/'controller.lock').exists());self.assertFalse((self.root/'recoverable_outputs.json').exists())
 def test_real_cpu_backfill_retains_original_and_is_idempotent(self):
  before=self.f.result.read_bytes();r=b.run(self.plan,self.sha,self.f.store,self.root/'controller.lock',True);self.assertEqual(r['rows'][0]['status'],'PASS_QUALIFICATION_RECOVERY');self.assertEqual(r['worker_reruns'],0);self.assertEqual(r['formal_admitted'],0);r2=b.run(self.plan,self.sha,self.f.store,self.root/'controller.lock',True);self.assertEqual(r2['rows'][0]['status'],'PASS_QUALIFICATION_RECOVERY');self.assertEqual(before,self.f.result.read_bytes())
 def test_wrong_plan_sha_and_formal_job_are_rejected(self):
  with self.assertRaises(ValueError):b.run(self.plan,'bad',self.f.store,self.root/'lock')
  plan=a.load(self.plan);plan['jobs'][0]['phase']='formal';self.plan.write_text(json.dumps(plan))
  with self.assertRaises(ValueError):b.run(self.plan,a.binding(self.plan)['sha256'],self.f.store,self.root/'lock')

if __name__=='__main__':unittest.main(verbosity=2)
