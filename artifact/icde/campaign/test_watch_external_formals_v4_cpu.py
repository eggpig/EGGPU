import copy,json,os,sys,tempfile,unittest
from pathlib import Path
import watch_external_formals_v4 as w

class Gates(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
  def write(name,value):
   p=self.p/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return w.bind(p)
  self.write=write
  tool=write('tool.json',{});supervision=write('supervision.json',{'status':'PASS_PROCESS_EXIT','supervisor_returncode':0,'surviving_owned_processes':[],'finished_monotonic':2})
  jobs=[];rows=[]
  for i in range(36):
   receipt=write('native/'+str(i)+'.json',{'status':'PASS'})
   jobs.append({'id':str(i),'receipt':receipt['path'],'expected_statuses':['PASS']})
   rows.append({'id':str(i),'receipt':receipt,'status':'PASS','returncode':0,'outcome':'PASS','process_supervision':supervision})
  plan=write('native_plan.json',{'jobs':jobs});native=write('native_state.json',{'status':'PASS_REQUESTED_JOBS','plan':plan,'jobs':jobs,'completed':rows})
  refs={'cohort_controller':tool,'state':str(self.p/'cohort.json'),'datasets':{},'reference_source':tool,'preparer':tool}
  complete=[];archive=self.p/'ref.npz';archive.write_bytes(b'archive');file=w.bind(archive)
  for name in ['US','OR','IN','UK']:
   sources=write(name+'/sources.json',{'sources':list(range(512))});metadata=write(name+'/meta.json',{})
   receipt=write(name+'/receipt.json',{'status':'PASS_ALL_512_CPU_REFERENCES','dataset':name,'completed':512,'gpu_executed':False,'independent':True,'lossless':True,'complete_sha_bindings':True,'sources':sources,'reference_source':tool,'reference_source_after':tool,'tool':tool,'n':900,'full_source_sequence':list(range(512)),'entries':[{'sequence':i,'source':i,'values':900,'metadata':metadata,'reference':{'file':file}} for i in range(512)]})
   refs['datasets'][name]={'receipt':receipt['path'],'sources':sources};complete.append({'dataset':name,'returncode':0,'receipt':receipt['path']})
  write('cohort.json',{'status':'PASS_ALL_FOUR_REFERENCE_COHORTS','gpu_executed':False,'completed':complete})
  groups={}
  for name in ['cugraph_mg','c2p','gum']:
   plan=write(name+'/plan.json',{'environment':{},'jobs':[{'id':'test','phase':'formal','receipt':str(self.p/name/'runs/result.json'),'bindings':[]}]})
   groups[name]={'plan':plan,'count':1,'queue':str(self.p/name/'queue'),'log':str(self.p/name/'controller.log')}
  self.cfg={'schema':'external.formal.watcher.v4','controller':tool,'supervisor':tool,'watcher':tool,'native':{'state':native,'plan':w.bind(self.p/'native_plan.json'),'count':36},'full_grid':write('grid.json',{'status':'PASS_CPU_SHARED_GRID_PREPARED_GPU_UNQUALIFIED','cells':list(range(56)),'gpu_executed':False}),'references':refs,'groups':groups,'gum_release':None,'cpu_affinity':sorted(os.sched_getaffinity(0)),'python_invocation':sys.executable,'controller_lock':str(self.p/'lock')}
 def test_ready(self):self.assertEqual(w.inspect(self.cfg,'supported')['status'],'PASS_ALL_LAUNCH_GATES')
 def test_gum_hold_independent(self):self.assertEqual(w.inspect(self.cfg,'gum')['status'],'HOLD_GUM_PRECISION_AUDIT')
 def test_missing_cohort_waits(self):Path(self.cfg['references']['state']).unlink();self.assertEqual(w.inspect(self.cfg,'supported')['status'],'WAITING_CPU_REFERENCE_COHORT')
 def test_running_waits(self):self.write('cohort.json',{'status':'RUNNING'});self.assertEqual(w.inspect(self.cfg,'supported')['status'],'WAITING_CPU_REFERENCE_COHORT')
 def test_stopped_rejects(self):
  self.write('cohort.json',{'status':'STOPPED_AFTER_PREPARATION_FAILURE'})
  with self.assertRaises(ValueError):w.inspect(self.cfg,'supported')
 def test_incomplete_512_rejects(self):
  p=Path(self.cfg['references']['datasets']['US']['receipt']);r=w.load(p);r['entries'].pop();p.write_text(json.dumps(r))
  with self.assertRaises(ValueError):w.inspect(self.cfg,'supported')
 def test_wrong_source_order_rejects(self):
  p=Path(self.cfg['references']['datasets']['US']['receipt']);r=w.load(p);r['entries'][0]['source']=88;p.write_text(json.dumps(r))
  with self.assertRaises(ValueError):w.inspect(self.cfg,'supported')
 def test_archive_truncated_rejects(self):
  (self.p/'ref.npz').write_bytes(b'x')
  with self.assertRaises(ValueError):w.inspect(self.cfg,'supported')
 def test_native_changed_rejects(self):
  (self.p/'native/0.json').write_text('{}')
  with self.assertRaises(ValueError):w.inspect(self.cfg,'supported')
 def test_unknown_supervision_rejects(self):
  b=self.write('supervision.json',{'status':'MYSTERY','supervisor_returncode':0,'surviving_owned_processes':[],'finished_monotonic':2});p=self.p/'native_state.json';s=w.load(p)
  for r in s['completed']:r['process_supervision']=b
  self.cfg['native']['state']=self.write('native_state.json',s)
  with self.assertRaises(ValueError):w.inspect(self.cfg,'supported')
 def test_partial_formal_output_rejects(self):
  (self.p/'c2p/runs').mkdir()
  with self.assertRaises(ValueError):w.inspect(self.cfg,'supported')
 def test_one_shot_gum_receipt(self):
  config=self.write('config.json',self.cfg);out=self.p/'launch.json';self.assertEqual(w.run(config['path'],'gum',out,True)['status'],'HOLD_GUM_PRECISION_AUDIT')
  with self.assertRaises(FileExistsError):w.run(config['path'],'gum',out,True)
 def test_release_requires_exact_plan(self):
  self.cfg['gum_release']=self.write('release.json',{'status':'PASS_GUM_FORMAL_RELEASE','plan_sha256':'wrong','display_label':'GUM (standard)','precision_audit':self.cfg['watcher']})
  with self.assertRaises(ValueError):w.inspect(self.cfg,'gum')

if __name__=='__main__':unittest.main()
