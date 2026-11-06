"""Metadata-only exporter trigger and two-generation retention CPU fixtures."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, os, tempfile, types, unittest
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/'export_on_progress.py'
spec=importlib.util.spec_from_file_location('export_progress_cpu',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SCRATCH=Path(os.environ['EGGPU_EXPORT_TEST_SCRATCH']);SCRATCH.mkdir(parents=True,exist_ok=True)
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
def prior(p):
 return dict(last_export_completed=p['completed'],last_export_stage=p['stage'],last_export_remaining_completed=p['remaining_completed'],last_export_terminal_queues=p['terminal_queues'],last_export_target_completed_signature=p['target_completed_signature'],last_export_finished=p['finished'],last_export_stopped=p['stopped'])
class Checks(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=SCRATCH);self.c=Path(self.temp.name);self.h=self.c/'first_pass_20261106';self.q=self.h/'core/queue/state.json';self.item=dict(status='RUNNING',jobs=[],completed=[]);write(self.q,self.item)
 def tearDown(self):self.temp.cleanup()
 def snap(self):write(self.q,self.item);return m.snapshot(self.c,None)
 def test_each_target_outcome_but_no_tiny_or_pilot(self):
  p=self.snap();self.assertTrue(m.needs_export(p,{}));last=prior(p)
  for i in range(20):self.item['completed'].append(dict(id='tiny'+str(i),phase='tiny',outcome='PASS'))
  self.item['completed'].append(dict(id='pilot',phase='pilot',outcome='FAIL'))
  p=self.snap();self.assertFalse(m.needs_export(p,last));self.assertEqual(p['target_completed'],0)
  for outcome in ('PASS','FAIL','SKIPPED_DEPENDENCY','BLOCKED','TIMED_OUT'):
   self.item['completed'].append(dict(id=outcome,phase='target',outcome=outcome,status=outcome,receipt={'path':'/unused/'+outcome,'sha256':'abc'}))
   p=self.snap();self.assertTrue(m.needs_export(p,last));last=prior(p);self.assertFalse(m.needs_export(self.snap(),last))
 def test_phase_fallback_and_inherited_prefix_deduplication(self):
  self.item['jobs']=[dict(id='a',phase='formal',receipt='/not-read/a'),dict(id='b',command=['worker','--phase=target'],receipt='/not-read/b')]
  self.item['completed']=[dict(id='a',outcome='PASS'),dict(id='b',outcome='FAIL')]
  p=self.snap();self.assertEqual(p['target_completed'],2);last=prior(p)
  clone=self.h/'author_parallel/lane0_queue/state.json';write(clone,self.item)
  self.assertFalse(m.needs_export(self.snap(),last));self.assertEqual(self.snap()['target_completed'],2)
  self.item['completed'][0]['status']='CHANGED_RESULT_METADATA';self.assertTrue(m.needs_export(self.snap(),last))
 def test_remaining_stage_terminal_finish_stop_conditions(self):
  remaining=self.h/'unified_main_continuation/queue/state.json';write(remaining,self.item);p=self.snap();last=prior(p)
  write(remaining,dict(status='RUNNING',completed=[dict(id='tiny',phase='tiny',outcome='PASS')]))
  self.assertFalse(m.needs_export(self.snap(),last))
  write(remaining,dict(status='RUNNING',completed=[dict(id='target',phase='target',outcome='SKIPPED_DEPENDENCY')]))
  p=self.snap();self.assertTrue(m.needs_export(p,last));self.assertEqual(p['remaining_completed'][str(remaining)],1)
  last=prior(p)
  for field,value in [('stage','RUNNING_NEXT'),('terminal_queues',['q:FAILED_CONTROLLER']),('finished',True),('stopped',True)]:
   changed={**p,field:value};self.assertTrue(m.needs_export(changed,last));self.assertFalse(m.needs_export(changed,prior(changed)))
 def test_run_persists_signature_across_restarts(self):
  out=self.c/'export';authority=self.c/'authority.json';authority.write_text('{}');args=argparse.Namespace(campaign=self.c,output=out,authority=authority,plot_python=None,once=True,cpu_set=None)
  original=(m.export,m.os.sched_setaffinity,m.subprocess.run);calls=[]
  m.export=lambda *a:(calls.append(a[2]['target_completed_signature']) or {'status':'EXPORTED'})
  m.os.sched_setaffinity=lambda *a:None;m.subprocess.run=lambda *a,**kw:types.SimpleNamespace(returncode=1)
  try:
   m.run(args);self.assertEqual(len(calls),1)
   self.item['completed']=[dict(id='tiny'+str(i),phase='tiny',outcome='PASS') for i in range(12)];write(self.q,self.item);m.run(args);self.assertEqual(len(calls),1)
   self.item['completed'].append(dict(id='newtarget',phase='target',outcome='PASS'));write(self.q,self.item);m.run(args);self.assertEqual(len(calls),2)
   m.run(args);self.assertEqual(len(calls),2)
   self.assertEqual(m.read(out/'state.json')['last_export_target_completed_signature'],calls[-1])
  finally:m.export,m.os.sched_setaffinity,m.subprocess.run=original
 def test_actual_export_keeps_only_current_and_previous(self):
  # Exercise the unchanged writer/rotation with synthetic metadata builders.
  out=self.c/'exports';out.mkdir();authority=self.c/'authority.json';authority.write_text('{}');original=m.load_module
  def build(receipts,authority,generation):
   (generation/'first_pass.json').write_text('{}');return {'counts':{}}
  modules={'first_pass_collector':types.SimpleNamespace(collect=lambda *a,**k:{}),
   'first_pass_builder':types.SimpleNamespace(build=build),
   'remaining_results':types.SimpleNamespace(capture=lambda *a:None,build=lambda *a:{'counts':{}})}
  m.load_module=lambda name,path:modules[name]
  try:
   for _ in range(3):m.export(self.c,out,self.snap(),authority,Path('/not-used'),False)
   self.assertEqual(len(list(out.glob('.generation.*'))),2);self.assertTrue((out/'current').is_symlink());self.assertTrue((out/'previous').is_symlink());self.assertNotEqual((out/'current').resolve(),(out/'previous').resolve())
  finally:m.load_module=original
  before=ast.parse((HERE/'before.py').read_text());after=ast.parse(SOURCE.read_text())
  for name in ('export','point','clean_unpublished','discover'):
   a=next(x for x in before.body if isinstance(x,ast.FunctionDef) and x.name==name);b=next(x for x in after.body if isinstance(x,ast.FunctionDef) and x.name==name);self.assertEqual(ast.dump(a),ast.dump(b))
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 if not result.wasSuccessful():raise SystemExit(1)
 receipt=dict(status='PASS_METADATA_TARGET_EXPORT_TRIGGER_CPU',tests=result.testsRun,source=dict(path=str(SOURCE),bytes=SOURCE.stat().st_size,sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest()),target_outcomes=['PASS','FAIL','SKIPPED_DEPENDENCY','BLOCKED','TIMED_OUT'],tiny_and_pilot_do_not_trigger=True,remaining_target_progress_preserved=True,stage_terminal_final_stop_preserved=True,unchanged_restart_does_not_export=True,inherited_prefix_deduplicated=True,actual_current_previous_retention_checked=True,collector_builder_plotter_unmodified=True,array_payloads_read=False,gpu_executed=False)
 (HERE/'cpu_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
