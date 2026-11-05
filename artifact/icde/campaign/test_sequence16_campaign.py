"""CPU-only continuous timing, ownership, semantic trace and baseline state reuse."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,os,tempfile,unittest
import numpy as np
import sg_worker as shared
import sequence16_worker as worker
import sequence16_adapter as adapter
import run_sequence16 as controller
import cugraph_csr_adapter

class SequenceContracts(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);shared.make_tiny(self.root)
  self.graph=shared.load_input(self.root/'canonical/TINY_D')
 def tearDown(self):
  import gc
  for value in self.graph.values():
   if isinstance(value,np.memmap):value._mmap.close()
  del self.graph;gc.collect();self.temp.cleanup()
 def test_exact_16_trace_and_fixed_three_sources(self):
  trace=worker.trace(self.graph);self.assertEqual(len(trace),16)
  self.assertEqual([r['algorithm'] for r in trace],['bfs','sssp','wcc','pr']*4)
  self.assertEqual([r['source'] for r in trace if r['algorithm']=='bfs'],[0,1,2,0])
  self.assertEqual([r['source'] for r in trace if r['algorithm']=='sssp'],[0,1,2,0])
  self.assertTrue(all(r['source'] is None for r in trace if r['algorithm'] in ('wcc','pr')))
 def test_writeable_reused_api_buffer_is_copied_before_next_request(self):
  buffer=np.zeros(7);outputs=[]
  for i in range(16):
   buffer[:]=i;out,copied=worker.owned_output(buffer,outputs,7);outputs.append(out);self.assertTrue(copied)
  for i,out in enumerate(outputs):np.testing.assert_array_equal(out,np.full(7,i));self.assertFalse(out.flags.writeable)
 def test_borrowed_view_is_copied(self):
  base=np.arange(14.);view=base[::2];view.flags.writeable=False
  out,copied=worker.owned_output(view,[],7);base[:]=0
  self.assertTrue(copied);np.testing.assert_array_equal(out,np.arange(14.)[::2])
 def test_immutable_owner_is_preserved_and_alias_copy_is_made(self):
  base=np.arange(7.);base.flags.writeable=False
  out,copied=worker.owned_output(base,[],7);self.assertIs(out,base);self.assertFalse(copied)
  second,copied=worker.owned_output(base,[base],7);self.assertTrue(copied);self.assertFalse(np.shares_memory(second,base))
 def test_incomplete_host_domain_is_rejected(self):
  with self.assertRaises(ValueError):worker.owned_output(np.ones(6),[],7)
 def test_continuous_interval_includes_constructor_not_sum_of_request_times(self):
  events=[]
  class Fake:
   contract={'fake':True}
   def __init__(self,*a):events.append('construct')
   def query(self,request):events.append(request['index']);return np.full(7,request['index'],dtype=float)
   def close(self):events.append('close')
  with patch.object(worker.time,'perf_counter_ns',side_effect=range(100)):
   outputs,result=worker.execute_sequence(SimpleNamespace(),self.graph,Fake)
  self.assertEqual(result['T_workflow_seconds'],34/1e9)
  self.assertEqual(sum(c['seconds'] for c in result['calls']),16/1e9)
  self.assertEqual(events,['construct',*range(16),'close']);self.assertEqual(result['warmup_requests'],0)
  for i,out in enumerate(outputs):np.testing.assert_array_equal(out,np.full(7,i))
 def test_failed_sequence_retains_completed_outputs_without_reporting_total(self):
  class Fail:
   contract={'fake':True}
   def __init__(self,*args):pass
   def query(self,request):
    if request['index']==3:raise RuntimeError('intentional injected failure')
    return np.full(7,request['index'],dtype=float)
   def close(self):pass
  outputs,result=worker.execute_sequence(SimpleNamespace(),self.graph,Fail)
  self.assertEqual(len(outputs),3);self.assertEqual(len(result['calls']),3)
  self.assertIsNone(result['T_workflow_seconds']);self.assertFalse(result['execution_complete'])
  self.assertIn('injected failure',result['execution_error'])

 def test_all_outputs_independently_match_oracle_after_close(self):
  graph=self.graph;events=[]
  class CPUOracle:
   contract={'fake_cpu_only':True}
   def __init__(self,*args):pass
   def query(self,r):return shared.reference(graph,r['algorithm'],r['source'])
   def close(self):events.append('closed')
  outputs,result=worker.execute_sequence(SimpleNamespace(),graph,CPUOracle)
  self.assertEqual(events,['closed'])
  for call,out in zip(result['calls'],outputs):
   expected=shared.reference(graph,call['algorithm'],call['source'])
   self.assertTrue(shared.audit_output(out,expected,graph,call['algorithm'])['pass'])
 def test_cugraph_reuses_weighted_graph_for_bfs_and_sssp(self):
  constructed=[];calls=[]
  class Fake:
   def __init__(self,graph,function,source=0):
    self.cp=np;self.graph=graph;self.function=function;self.source=source;constructed.append(self)
   def complete_query(self):
    calls.append((id(self),self.function,self.source,self.graph['directed']))
    return np.zeros(self.graph['n'])
   def close(self):pass
  with patch.object(cugraph_csr_adapter,'CuGraphCSR',Fake):
   sequence=adapter.CuGraphSequence(SimpleNamespace(),self.graph)
   for request in worker.trace(self.graph):sequence.query(request)
   sequence.close()
  self.assertEqual(len(constructed),3)
  self.assertEqual(calls[0][0],calls[1][0]);self.assertEqual(calls[0][0],calls[4][0]);self.assertFalse(calls[2][3])
  self.assertEqual([x[2] for x in calls if x[1]=='bfs'],[0,1,2,0])
 def plan_args(self,**overrides):
  args=dict(systems='eggpu,cugraph',datasets='TINY_D,TINY_U',phase='tiny',repeat=None,proofs=None,
   input_root=self.root,source=Path('/candidate/source'),native_dso=Path('/dso.so'),arena_library_dir=Path('/arena'),
   setup=Path('/setup'),lock_file=Path('/lock'),reference_cache=Path('/refs'),output=self.root/'plan',
   python_eggpu=Path('/core/python'),python_cugraph=Path('/rapids/python'),gpu_uuid='GPU-TEST',
   cpu_set=','.join(map(str,sorted(os.sched_getaffinity(0))[:16])),timeout=1800,budget_mib=22000,min_host_free_gib=8)
  args.update(overrides);return SimpleNamespace(**args)
 def test_plan_never_executes_or_creates_output(self):
  args=self.plan_args()
  with patch('subprocess.run',side_effect=AssertionError('No execution')):jobs=controller.plan(args)
  self.assertEqual(len(jobs),4);self.assertFalse(args.output.exists())
  self.assertTrue(all('--source' in j['command'] for j in jobs))
 def test_single_query_proof_not_accepted(self):
  path=self.root/'proof';path.write_text(json.dumps(dict(status='PASS_QUALIFICATION',protocol=shared.PROTOCOL)))
  with self.assertRaises(ValueError):controller.proof_index([path])
 def test_no_gpu_without_optin(self):
  with self.assertRaisesRegex(ValueError,'execute-gpu'):worker.job(SimpleNamespace(execute_gpu=False))
 def test_child_needs_global_lease(self):
  with self.assertRaisesRegex(ValueError,'inherited'):worker.inherited_lease(SimpleNamespace(lease_fd=None))
 def test_formal_explicitly_plans_ten_repeats_and_rotates_system_order(self):
  directory=self.root/'canonical/EP';directory.mkdir();(directory/'canonical.json').write_text(json.dumps(dict(directed=True)))
  indexed={}
  for system in ('eggpu','cugraph'):
   for phase,dataset in [('tiny','TINY_D'),('target','EP')]:indexed[(phase,system,dataset)]=Path('/proof')
  with patch.object(controller,'proof_index',return_value=indexed):jobs=controller.plan(self.plan_args(datasets='EP',phase='formal'))
  self.assertEqual(len(jobs),20);self.assertEqual({j['repeat'] for j in jobs},set(range(10)))
  self.assertEqual([j['system'] for j in jobs[:4]],['eggpu','cugraph','cugraph','eggpu'])

if __name__=='__main__':unittest.main()
