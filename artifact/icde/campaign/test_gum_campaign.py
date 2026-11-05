"""CPU contracts only: descriptors, weights, native dispatch and truthful timing."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,os,tempfile,unittest
import numpy as np
import gum_worker as worker
import gum_reused as reused
import native_worker
import run_gum
import sg_worker as shared

class GumContracts(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  native_worker.make_tiny(self.root);self.graph=shared.load_input(self.root/'canonical/TINY_D')
 def tearDown(self):
  for value in self.graph.values():
   if isinstance(value,np.memmap):value._mmap.close()
  self.graph=None
  import gc;gc.collect()
  self.temp.cleanup()
 def test_weight_values_are_preserved(self):
  result=reused.weight_domain(self.graph);self.assertLess(result['simple_path_distance_bound'],2**32-1)
  for value in [0.,1.5,float('nan'),float('inf')]:
   g=dict(self.graph,weights=np.asarray([value]))
   with self.assertRaises(ValueError):reused.weight_domain(g)
 def test_weight_overflow_rejected(self):
  with self.assertRaises(ValueError):reused.weight_domain(dict(self.graph,n=2**30,weights=np.asarray([16.])))
 def test_descriptor_keeps_original_weight_file(self):
  p,info=worker.descriptor(self.graph,'sssp',self.root);rows=p.read_text().splitlines()
  self.assertEqual(p.name,worker.DESCRIPTOR_NAME);self.assertEqual(p.suffix,'.lawcsr');self.assertEqual(len(rows),7);self.assertEqual(rows[3],'1');self.assertEqual(rows[-1],str(self.root/'canonical/TINY_D/weights.bin'))
 def test_wcc_projection_preserves_isolates_and_reverse_edges(self):
  p,info=worker.descriptor(self.graph,'wcc',self.root);rows=p.read_text().splitlines()
  self.assertEqual(rows[1],'16');self.assertEqual(rows[-1],'-');self.assertEqual(rows[3],'0')
  offsets=np.fromfile(rows[4],dtype='<i4');indices=np.fromfile(rows[5],dtype='<i4')
  pairs={(u,int(v)) for u in range(16) for v in indices[offsets[u]:offsets[u+1]]}
  self.assertEqual(pairs,{(v,u) for u,v in pairs});self.assertEqual(offsets[-1],offsets[7])
 def test_author_standard_sssp_and_fixed_pr(self):
  args=SimpleNamespace(function='sssp',mpiexec=Path('/mpi'),core_python=Path('/python'))
  cmd=reused.native_command(args,'/gum','/csr',16,2,8,Path('/out'))
  self.assertIn('--ws=false',cmd);self.assertIn('--sssp_source=2',cmd);self.assertIn('gum_rank_exec.py',' '.join(cmd))
  args.function='pr';cmd=reused.native_command(args,'/gum','/csr',16,0,2,Path('/out'))
  self.assertIn('--pr_mr=117',cmd);self.assertIn('--pr_d=0.85',cmd)
 def test_native_sentinel_audit_preserves_raw(self):
  p=self.root/'native';v=np.asarray([0,1,2**32-1,-1],dtype='<i8');v.tofile(p);before=p.read_bytes()
  actual=reused.normalise_native(p,'bfs',4);self.assertTrue(np.isposinf(actual[2:]).all());self.assertEqual(p.read_bytes(),before)
 def test_output_domain_is_complete(self):
  p=self.root/'native';np.asarray([0],dtype='<i8').tofile(p)
  with self.assertRaises(ValueError):reused.normalise_native(p,'bfs',16)
 def test_rankmax_uses_complete_query_not_phase_maxima(self):
  runtime=self.root/'runtime.py';runtime.write_text('def rank_diagnostics(*a,**kw):\n return [{"rank":r,"rounds":[{"warm_query_seconds":v} for v in x]} for r,x in enumerate([[2.,4.,3.],[3.,2.,5.]])]\ndef setup_records(*a):return {"setup":999}\n')
  times,details=worker.parse_times(runtime,self.root,'bfs',2);self.assertEqual(times,[3.,4.,5.])
 def test_no_gpu_without_optin(self):
  with self.assertRaisesRegex(ValueError,'execute-gpu'):worker.job(SimpleNamespace(execute_gpu=False))
 def test_plan_is_complete_task_and_cpu_only(self):
  cpus=sorted(os.sched_getaffinity(0))[:16]
  args=SimpleNamespace(proofs=None,counts='1,2',datasets='TINY_D,TINY_U',functions='bfs,pr',phase='tiny',repeat=0,
    output=self.root/'planned',core_python=Path('/python'),gpu_uuids='GPU-a,GPU-b',cpu_set=','.join(map(str,cpus)))
  with patch('subprocess.Popen',side_effect=AssertionError('No execution during plan')):jobs=run_gum.plan(args)
  self.assertEqual(len(jobs),8);self.assertTrue(all(j['T_first_available'] is True for j in jobs));self.assertFalse(args.output.exists())
 def test_formal_requires_tiny_and_target(self):
  cpus=sorted(os.sched_getaffinity(0))[:16]
  args=SimpleNamespace(proofs=None,counts='8',datasets='EP',functions='pr',phase='formal',repeat=0,output=self.root/'planned',core_python=Path('/python'),gpu_uuids=','.join('GPU-'+str(i) for i in range(8)),cpu_set=','.join(map(str,cpus)))
  with self.assertRaisesRegex(ValueError,'Missing tiny proof'):run_gum.plan(args)

if __name__=='__main__':unittest.main()
