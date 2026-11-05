"""Full CPU toy E4S proof -> byte recovery -> frozen-reader admission pipeline."""
from pathlib import Path
import copy,json,shutil,sys,unittest
import archive_outputs as archive,decorate_formal_plan as decorate
import test_archive_outputs
D=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(D/'formal_admission_d84_v2'))
import admit,test_admit

class FullAdmission(unittest.TestCase):
 def setUp(self):
  self.a=test_admit.Admission();self.a.setUp();self.f=test_archive_outputs.Recoverable();self.f.setUp();self.root=self.a.root
  shutil.copytree(self.f.inputs/'canonical/TINY_U',self.f.inputs/'canonical/US')
  epoch=dict(schema='e4s.epoch.v1',release_sources=self.a.files,native_dso=archive.binding(self.a.dso),budget_mib=22000)
  ep_path=self.root/'e4s_epoch.json';ep_path.write_text(json.dumps(epoch));ep=archive.binding(ep_path);self.jobs=[]
  for phase,dataset in [('tiny','TINY_U'),('target','US')]:
   parent=self.f.root/phase;parent.mkdir();p=parent/'result.json';r=copy.deepcopy(self.f.raw);r.update(phase=phase,dataset=dataset,directed=False,epoch=ep,epoch_sha256=archive.digest(epoch));r['measurements'].update(first_complete_seconds=.75,adapter_preparation_seconds=.25,query_batch_seconds=.5,requests=[dict(sequence=i,source=i) for i in range(2)])
   if phase=='target':r['tiny_proof']=archive.binding(self.f.root/'tiny/result.json')
   p.write_text(json.dumps(r));cmd=['python','worker.py','--variant','eggpu','--batch','2','--dataset',dataset,'--phase',phase,'--repeat','0','--output',str(parent),'--input-root',str(self.f.inputs),'--release',str(self.a.source)]
   if phase=='target':cmd+=['--tiny-proof',str(self.f.root/'tiny/result.json')]
   self.jobs.append(dict(id=phase,phase=phase,receipt=str(p),command=cmd,expected_statuses=['PASS_E4S_QUALIFICATION'],bindings=[],depends_on=[]));archive.archive_result(p,self.f.inputs,self.f.store,parent/'recoverable_outputs.json')
  target=self.jobs[-1];raw=archive.load(target['receipt']);self.formals=[]
  original=dict(command=[sys.executable,'-c','pass'],command_sha256=decorate.command_digest([sys.executable,'-c','pass']),bindings=[archive.binding(Path(sys.executable).resolve())],gpu_lock=str(self.root/'gpu.lock'),receipt=str(self.root/'original.json'),expected_statuses=['PASS'],timeout=10)
  for i in range(10):
   j=copy.deepcopy(target);j.update(id='formal'+str(i),phase='formal',expected_statuses=['PASS_E4S_FORMAL'],depends_on=['target'],timeout=90120,maintenance=original,target_contract=dict(source_sequence=[0,1],input_sha256=raw['input_sha256']));cmd=j['command'];cmd[cmd.index('--phase')+1]='formal';cmd[cmd.index('--repeat')+1]=str(i);cmd[cmd.index('--output')+1]=str(self.root/('formal'+str(i)));cmd+=['--target-proof',target['receipt']];j['receipt']=str(self.root/('formal'+str(i))/'result.json');self.formals.append(j)
  q=self.a.save('e4s_qual.json',dict(jobs=self.jobs,environment={}));f=self.a.save('e4s_pending.json',dict(source_tree_sha256='fixture_d84',jobs=self.formals,environment={}));folder=Path(archive.__file__).parent
  contract={k:archive.binding(folder/file) for k,file in dict(helper='archive_outputs.py',wrapper='maintain_e4s_outputs.py',exporter='export_eligible.py',controller='run_e4s_formal_queue_v1.py',supervisor='owned_process_supervisor_v2.py',decorator='decorate_formal_plan.py').items()};contract.update(python=sys.executable,store=str(self.f.store),postprocess_timeout_seconds=86400)
  self.a.manifest['groups']=dict(e4s=dict(pending=f,qualifications=[q],expected_formal_jobs=10));self.a.manifest['archive_contract']=contract;self.a.manifest['policy']=dict(e4s_formal_archive_repair_pending=False);self.a.write(self.a.manifest_path,self.a.manifest)
 def tearDown(self):self.f.tearDown();self.a.tearDown()
 def test_complete_recoverable_qualification_admits_all_ten_with_required_controller(self):
  r=admit.inspect(self.a.manifest_path,'e4s','admit',self.root/'admitted');self.assertEqual(r['status'],'PASS_ADMITTED_PLAN_ONLY_NOT_LAUNCHED');p=admit.load(r['plan']['path']);self.assertEqual(len(p['jobs']),10);self.assertEqual(p['postprocess_contract'],'e4s_recoverable_outputs_v1');self.assertTrue(all(j['required_postprocess']=='e4s_recoverable_outputs_v1' for j in p['jobs']));self.assertEqual([j['command'] for j in p['jobs']],[j['command'] for j in self.formals])
 def test_missing_one_recovery_sidecar_blocks_entire_formal_group(self):
  (self.f.root/'target/recoverable_outputs.json').unlink();r=admit.inspect(self.a.manifest_path,'e4s','admit',self.root/'admitted');self.assertEqual(r['status'],'NOT_ADMITTED_BLOCKED');self.assertFalse((self.root/'admitted').exists())

if __name__=='__main__':unittest.main(verbosity=2)
