from pathlib import Path
import copy,json,sys,unittest
import archive_outputs as a,maintain_e4s_outputs as maintenance,export_eligible as export,decorate_formal_plan as decorate
import test_archive_outputs

class MaintenanceExport(unittest.TestCase):
 def setUp(self):
  self.f=test_archive_outputs.Recoverable();self.f.setUp();self.root=self.f.root;self.f.raw.update(phase='formal',status='PASS_E4S_FORMAL',repeat=0);self.f.raw['measurements']['first_complete_seconds']=.75;self.f.save_result()
  self.script=self.root/'original_maintenance.py';self.script.write_text("import sys,json\nfrom pathlib import Path\np=Path(sys.argv[1]);p.write_text(json.dumps({'status':'PASS_ORIGINAL_MAINTENANCE' if sys.argv[2]=='pass' else 'FAIL_ORIGINAL_MAINTENANCE'}));raise SystemExit(0 if sys.argv[2]=='pass' else 1)\n")
  self.original=dict(command=[sys.executable,str(self.script),str(self.root/'original.json'),'pass'],bindings=[a.binding(Path(sys.executable).resolve()),a.binding(self.script)],receipt=str(self.root/'original.json'),expected_statuses=['PASS_ORIGINAL_MAINTENANCE'],timeout=10,gpu_lock=str(self.root/'gpu.lock'))
  self.original['command_sha256']=decorate.command_digest(self.original['command']);self.helper=a.binding(a.__file__);self.spec=dict(result=str(self.f.result),input_root=str(self.f.inputs),store=str(self.f.store),archive_receipt=str(self.f.out),archive_helper=self.helper,bindings=[self.helper,a.binding(maintenance.__file__)],original_maintenance=self.original);self.path=self.root/'maintenance_spec.json';self.save_spec()
 def tearDown(self):self.f.tearDown()
 def save_spec(self):self.path.write_text(json.dumps(self.spec,indent=2))
 def test_raw_formal_pass_without_archive_and_maintenance_cannot_export(self):
  with self.assertRaises(FileNotFoundError):export.eligible(self.f.result,self.root/'absent.json',self.helper)
 def test_mandatory_archive_and_maintenance_make_measured_time_exportable(self):
  raw=self.f.result.read_bytes();r=maintenance.run(self.path,self.root/'post.json');self.assertEqual(r['status'],'PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE');entry=export.eligible(self.f.result,self.root/'post.json',self.helper);self.assertEqual(entry['first_complete_seconds'],.75);self.assertTrue(entry['postprocessing_excluded_from_measurement']);self.assertEqual(raw,self.f.result.read_bytes())
 def test_archive_failure_blocks_original_maintenance_and_export(self):
  array=self.f.arrays[0].copy();array[0]=-0.;self.f.replace_archive(0,array)
  with self.assertRaises(ValueError):maintenance.run(self.path,self.root/'failed.json')
  self.assertFalse((self.root/'original.json').exists());self.assertEqual(a.load(self.root/'failed.json')['archive_status'],'FAIL')
  with self.assertRaises(ValueError):export.eligible(self.f.result,self.root/'failed.json',self.helper)
 def test_original_failure_then_explicit_cpu_only_recovery_preserves_evidence(self):
  self.original['command'][-1]='fail';self.original['command_sha256']=decorate.command_digest(self.original['command']);self.save_spec();raw=self.f.result.read_bytes()
  with self.assertRaises(ValueError):maintenance.run(self.path,self.root/'failure.json')
  failure=(self.root/'failure.json').read_bytes();self.assertEqual(a.load(self.root/'failure.json')['archive_status'],'PASS')
  with self.assertRaises(ValueError):export.eligible(self.f.result,self.root/'failure.json',self.helper)
  self.original['command'][-1]='pass';self.original['command'][2]=str(self.root/'recovered_original.json');self.original['receipt']=str(self.root/'recovered_original.json');self.original['command_sha256']=decorate.command_digest(self.original['command']);self.path=self.root/'explicit_recovery_spec.json';self.spec['recovery_of']=a.binding(self.root/'failure.json');self.save_spec();r=maintenance.run(self.path,self.root/'recovered_maintenance.json');self.assertEqual(r['status'],'PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE');self.assertEqual(raw,self.f.result.read_bytes());self.assertEqual(failure,(self.root/'failure.json').read_bytes());self.assertEqual(export.eligible(self.f.result,self.root/'recovered_maintenance.json',self.helper)['first_complete_seconds'],.75)
 def test_corrupted_pinned_object_blocks_export(self):
  maintenance.run(self.path,self.root/'post.json');sidecar=a.load(self.f.out);Path(sidecar['vectors'][0]['object']['path']).write_bytes(b'broken')
  with self.assertRaises(ValueError):export.eligible(self.f.result,self.root/'post.json',self.helper)
 def test_missing_vector_or_changed_measured_hash_rejected(self):
  maintenance.run(self.path,self.root/'post.json');sidecar=a.load(self.f.out);sidecar['vectors'].pop();self.f.out.write_text(json.dumps(sidecar))
  with self.assertRaises(ValueError):a.validate_sidecar(self.f.result,self.f.out,self.helper,require_formal=True)
 def test_decorated_plan_keeps_all_ten_original_worker_commands_and_timeouts(self):
  folder=Path(a.__file__).parent;spec=dict(helper=self.helper,wrapper=a.binding(maintenance.__file__),exporter=a.binding(export.__file__),controller=a.binding(folder/'run_e4s_formal_queue_v1.py'),supervisor=a.binding(folder/'owned_process_supervisor_v2.py'),python=sys.executable,store=str(self.f.store),postprocess_timeout_seconds=86400)
  jobs=[dict(id='formal'+str(i),phase='formal',command=['python','frozen_worker.py','--input-root',str(self.f.inputs),'--phase','formal','--repeat',str(i)],receipt=str(self.root/('formal'+str(i))/'result.json'),expected_statuses=['PASS_E4S_FORMAL'],timeout=90120,maintenance=self.original) for i in range(10)];plan=dict(jobs=jobs);result=decorate.decorate(plan,spec,self.root/'admission')
  self.assertEqual([x['command'] for x in result['jobs']],[x['command'] for x in plan['jobs']]);self.assertEqual(len(result['jobs']),10);self.assertTrue(all(x['timeout']==90120 and x['required_postprocess']=='e4s_recoverable_outputs_v1' for x in result['jobs']));self.assertTrue(all(x['maintenance']['expected_statuses']==['PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE','PASS_FAILED_WORKER_CLEANUP'] for x in result['jobs']))

if __name__=='__main__':unittest.main(verbosity=2)
