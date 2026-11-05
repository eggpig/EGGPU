"""Meaningful CPU fixtures; no GPU import/worker execution or real admission."""
from pathlib import Path
import copy,fcntl,json,tempfile,unittest,os
from unittest import mock
import admit as a
D=Path(__file__).resolve().parents[1]

class Admission(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.b=a.Binder();self.source=self.root/'source';self.source.mkdir()
  self.files={}
  for name in ['easygraph/gpu/session.py','easygraph/gpu/_weight_admission.py','cpp_easygraph/classes/graph_convert.cpp']:
   p=self.source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture source');self.files[name]=self.b.content(p)
  self.dso=self.root/'native.so';self.dso.write_bytes(b'CPU fixture only, not a DSO')
  ep=dict(protocol='e1-host-csr-complete-v1',release_sources=self.files,files={'native_dso':self.b.bind(self.dso)},budget_mib=22000)
  self.epoch=self.save('epoch.json',ep);self.edigest=a.digest(ep);self.jobs=[]
  for phase,dataset in [('tiny','TINY_U'),('target','US')]:
   folder=self.root/'inputs/canonical'/dataset;folder.mkdir(parents=True);self.write(folder/'canonical.json',dict(num_nodes=16,num_entries=0,directed=False));self.write(folder/'sources.json',dict(primary_sources=[0,1,2]));self.write(folder/'receipt.json',dict(status='PASS_CANONICAL_INPUT',files={role:self.b.bind(folder/file) for role,file in [('canonical','canonical.json'),('sources','sources.json')]}));graph=a.canonical(folder,a.Binder());self.graph=graph
   source_rows=[]
   for index in range(3):
    calls={}
    for name in ['first','warmup','ready']:
     p=self.root/dataset/('out'+str(index)+name);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((str(index)+name).encode())
     calls[name]=dict(seconds=2 if name=='first' else 1,quality=dict(complete=True,dtype='float64',owning=True,immutable=True,**{'pass':True},exact_reference_match=True,unreachable_match=True),output=self.b.bind(p))
    child=dict(status='PASS',quality_pass=True,source=index,input_sha256=graph['input_sha256'],calls=calls)
    source_rows.append(dict(source=index,result=self.save(dataset+'/source'+str(index)+'.json',child)))
   proof=dict(status='PASS_QUALIFICATION',phase=phase,system='eggpu',algorithm='bfs',dataset=dataset,directed=False,quality_pass=True,input_sha256=graph['input_sha256'],epoch=self.epoch,epoch_sha256=self.edigest,sources=source_rows)
   if phase=='target':proof['tiny_proof']=self.b.bind(self.root/'TINY_U/result.json')
   receipt=self.save(dataset+'/result.json',proof);cmd=['python','worker.py','--system','eggpu','--algorithm','bfs','--dataset',dataset,'--phase',phase,'--repeat','0','--output',str(self.root/dataset),'--input-root',str(self.root/'inputs'),'--release',str(self.source),'--budget-mib','22000']
   if phase=='target':cmd+=['--tiny-proof',str(self.root/'TINY_U/result.json')]
   self.jobs.append(dict(id=phase,command=cmd,receipt=receipt['path'],expected_statuses=['PASS_QUALIFICATION'],phase=phase,depends_on=[],bindings=[]))
  self.qual=dict(source_tree_sha256='fixture_d84',jobs=self.jobs,environment={});self.qbinding=self.save('qualification.json',self.qual);target=self.jobs[-1];formals=[]
  for r in range(10):
   j=copy.deepcopy(target);j['id']='formal'+str(r);j['phase']='formal';j['depends_on']=['target'];c=j['command'];c[c.index('--phase')+1]='formal';c[c.index('--repeat')+1]=str(r);c[c.index('--output')+1]=str(self.root/('formal'+str(r)));c+=['--target-proof',target['receipt']];j['receipt']=str(self.root/('formal'+str(r))/'result.json');formals.append(j)
  self.formal=dict(source_tree_sha256='fixture_d84',jobs=formals,environment={});self.fbinding=self.save('pending.json',self.formal)
  # A bound stand-in isolates the admission helper from the separately tested
  # production references_gate (13 tests). Its waiting/pass behavior is real.
  watcher=self.root/'watcher.py';watcher.write_text("raise RuntimeError('entrypoint must never execute')\ndef references_gate(spec):\n r=load(spec['state'])\n return {'status':r['status']}\n")
  self.state=self.root/'reference_state.json';self.write(self.state,dict(status='PASS_ALL_REFERENCE_PREPARATION_COMPLETE'))
  cfg=self.save('watcher_config.json',dict(watcher=self.b.bind(watcher),references=dict(state=str(self.state))))
  cpu=self.save('cpu.json',dict(status='PASS_CPU',gpu_executed=False))
  frozen=self.save('freeze.json',dict(source_tree_sha256='fixture_d84',source_files=self.files))
  readers={k:self.b.bind(D/v) for k,v in dict(sg='sg_current/sg_worker.py',native_sg='native_load_p1/sg_worker.py',native='native_load_p1/native_worker.py',native_profile='native_load_p1/native_profile.py',sequence='sg_current/sequence16_worker.py',e4s='e4s/worker.py').items()}
  self.manifest=dict(schema='eggpu.d84.formal.admission.v1',source_sha256='fixture_d84',source_root=str(self.source),source_freeze=frozen,native_dso=self.b.bind(self.dso),proof_readers=readers,cpu_gate=dict(external_watcher_config=cfg,receipts=[dict(binding=cpu,status='PASS_CPU')]),groups=dict(sg=dict(pending=self.fbinding,qualifications=[self.qbinding],expected_formal_jobs=10)))
  self.manifest_path=self.root/'manifest.json';self.write(self.manifest_path,self.manifest)
 def tearDown(self):self.temp.cleanup()
 def write(self,p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2));return p
 def save(self,name,d):return self.b.bind(self.write(self.root/name,d))
 def test_all_pass_admits_all_ten_without_starting_and_preserves_commands(self):
  result=a.inspect(self.manifest_path,'sg','admit',self.root/'admitted');self.assertEqual(result['status'],'PASS_ADMITTED_PLAN_ONLY_NOT_LAUNCHED');self.assertEqual(result['workers_launched'],0);plan=a.load(result['plan']['path']);self.assertEqual([j['command'] for j in plan['jobs']],[j['command'] for j in self.formal['jobs']]);self.assertTrue(all(j['depends_on']==[] for j in plan['jobs']));self.assertTrue(all(j['external_qualification_dependencies']==['target'] for j in plan['jobs']));self.assertTrue(all(len(j['bindings'])>=6 for j in plan['jobs']))
 def test_inspection_never_admits_or_creates_output(self):
  r=a.inspect(self.manifest_path,'sg','inspect',self.root/'out');self.assertEqual(r['status'],'READY_FOR_FULL_PROOF_AUDIT_NOT_ADMITTED');self.assertFalse((self.root/'out').exists())
 def test_failed_or_absent_target_blocks_entire_group(self):
  p=Path(self.jobs[-1]['receipt']);d=a.load(p);d['status']='FAILED';self.write(p,d)
  for mode in ['inspect','admit']:
   r=a.inspect(self.manifest_path,'sg',mode,self.root/'out');self.assertEqual(r['status'],'NOT_ADMITTED_BLOCKED');self.assertFalse((self.root/'out').exists());self.assertEqual(r['formal_jobs'],10)
  p.unlink();self.assertEqual(a.inspect(self.manifest_path,'sg')['blockers'][0]['status'],'ABSENT')
 def test_cpu_reference_wait_even_all_qualifications_pass(self):
  self.write(self.state,dict(status='WAITING_CPU_REFERENCE_COHORT'));r=a.inspect(self.manifest_path,'sg','admit',self.root/'out');self.assertEqual(r['status'],'NOT_ADMITTED_BLOCKED');self.assertFalse((self.root/'out').exists())
 def test_corrupt_output_sha_is_fatal_before_plan_creation(self):
  p=self.root/'US/out0first';p.write_bytes(b'tampered');
  with self.assertRaises(ValueError):a.inspect(self.manifest_path,'sg','admit',self.root/'out')
  self.assertFalse((self.root/'out').exists())
 def test_changed_current_source_is_fatal(self):
  (self.source/'easygraph/gpu/session.py').write_text('changed')
  with self.assertRaisesRegex(ValueError,'Current source changed'):a.inspect(self.manifest_path,'sg','admit',self.root/'out')
 def test_no_overwrite_admission_artifact(self):
  (self.root/'out').mkdir()
  with self.assertRaisesRegex(ValueError,'overwrite'):a.inspect(self.manifest_path,'sg','admit',self.root/'out')
 def test_missing_duplicate_or_out_of_range_repeat_rejected(self):
  for kind in ['missing','duplicate','range']:
   f=copy.deepcopy(self.formal)
   if kind=='missing':f['jobs'].pop()
   else:j=f['jobs'][-1];j['command'][j['command'].index('--repeat')+1]='0' if kind=='duplicate' else '10'
   with self.assertRaisesRegex(ValueError,'ten repeats'):a.structural_gate(f,[self.qual])
 def test_configuration_worker_or_unknown_dependency_rejected(self):
  for kind in ['config','worker','dependency']:
   f=copy.deepcopy(self.formal);j=f['jobs'][0]
   if kind=='config':j['command']+=['--max-iter','17']
   elif kind=='worker':j['command'][1]='different.py'
   else:j['depends_on']=['unknown']
   with self.assertRaises(ValueError):a.structural_gate(f,[self.qual])
 def test_bound_epoch_tampering_rejected(self):
  (self.root/'epoch.json').write_text((self.root/'epoch.json').read_text()+' ')
  with self.assertRaises(ValueError):a.inspect(self.manifest_path,'sg','admit',self.root/'out')
 def test_common_quality_rejects_lax_pr_or_nonowning_outputs(self):
  q=dict(complete=True,dtype='float64',owning=True,immutable=True,**{'pass':True},residual_l1=1e-7,mass_error=1e-7,reference_l1=1e-6);self.assertTrue(a.quality(q,'pr'));q['residual_l1']=1e-4;self.assertFalse(a.quality(q,'pr'));q['residual_l1']=1e-7;q['owning']=False;self.assertFalse(a.quality(q,'pr'))
 def test_e4s_exact_hash_ownership_and_source_checks_without_fake_archive(self):
  checks=[dict(sequence=i,source=i,values=16,quality=dict(passed=True,exact=True,vertices=16),output_sha256='a'*64,survives_session_close=True) for i in range(2)]
  proof=dict(batch=2,variant='eggpu',source_sequence=[0,1],checks=checks,measurements=dict(first_complete_seconds=3,adapter_preparation_seconds=1,query_batch_seconds=2,all_outputs_retained_through_close=True,requests=[dict(source=i,sequence=i) for i in range(2)]))
  evidence=a.call_artifacts(proof,'e4s',dict(n=16),a.Binder());self.assertFalse(evidence['archive_rehashed']);checks[1]['source']=7
  with self.assertRaisesRegex(ValueError,'source/shape'):a.call_artifacts(proof,'e4s',dict(n=16),a.Binder())
 def test_binder_full_sha_catches_same_size_same_inode_modification(self):
  p=self.root/'payload';p.write_bytes(b'abc');b=a.Binder();old=b.bind(p);b.checked(old);self.assertEqual(b.hashed_bytes,6);p.write_bytes(b'xyz')
  with self.assertRaises(ValueError):b.checked(old)
 def test_e4s_formal_explicitly_held_until_archive_repair(self):
  self.manifest['groups']['e4s']=self.manifest['groups']['sg'];self.manifest['policy']={'e4s_formal_archive_repair_pending':True};self.write(self.manifest_path,self.manifest)
  r=a.inspect(self.manifest_path,'e4s','admit',self.root/'out');self.assertEqual(r['status'],'NOT_ADMITTED_BLOCKED');self.assertIn('e4s_output_archive',[x.get('stage') for x in r['blockers']]);self.assertFalse((self.root/'out').exists())
 def test_formal_must_cover_every_qualified_target_even_same_total_count(self):
  q=copy.deepcopy(self.qual);extra=copy.deepcopy(q['jobs'][-1]);extra['id']='omitted_target';extra['receipt']+='other';q['jobs'].append(extra)
  with self.assertRaisesRegex(ValueError,'every exact qualification target'):a.structural_gate(self.formal,[q])
 def test_full_admission_holds_global_controller_lock(self):
  lock=self.root/'root_controller.lock'
  def probe(*args):
   with lock.open('a+') as other:
    with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
   return {'status':'TEST_LOCKED_NO_WORKER'}
  with mock.patch.object(a,'_inspect_locked',side_effect=probe):self.assertEqual(a.inspect(self.manifest_path,'sg','admit',self.root/'out')['status'],'TEST_LOCKED_NO_WORKER')
  with lock.open('a+') as other:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(other,fcntl.LOCK_UN)
 @unittest.skipUnless(os.environ.get('EGGPU_PRIVATE_FROZEN_PLAN_ROOT'), 'Private deployment evidence is not distributed; supply its explicit directory to recheck original cardinalities')
 def test_real_frozen_plan_cardinality_and_dependency_mapping(self):
  D=Path(os.environ['EGGPU_PRIVATE_FROZEN_PLAN_ROOT'])
  counts={'sg':(72,560),'native':(124,880),'sequence':(10,60),'e4s':(60,360)}
  for name,counts_expected in counts.items():
   form=a.load(D/('unified_release_d84_v2/'+name+'_formal/pending.json'));plans=[a.load(D/('unified_release_d84_v2/'+name+'/plan.json'))]
   if name=='native':plans.append(a.load(D/'unified_release_d84_v2/preflight/plan.json'))
   q,m=a.structural_gate(form,plans);self.assertEqual((len(q),len(m)),counts_expected)

if __name__=='__main__':unittest.main(verbosity=2)
