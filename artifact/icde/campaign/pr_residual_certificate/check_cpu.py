"""CPU full-vector regression checks against the unchanged independent oracle."""
from pathlib import Path
from fractions import Fraction as F
import importlib.util,json,math,os,sys,tempfile,time,unittest
import numpy as np
from scipy.sparse import csr_matrix
HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'state_main_first_pass/runner'
sys.path.insert(0,str(ORIGINAL));sys.path.insert(0,str(HERE))
import state_reference as old
import pr_certificate as cert


def owned(values):
 result=np.array(values,dtype=np.float64,copy=True);result.flags.writeable=False;return result


def exact_solution(matrix):
 n=matrix.shape[0];dense=matrix.toarray();a=F(.85);b=F(.15)
 strengths=[sum(F(float(x)) for x in dense[i]) for i in range(n)]
 eq=[]
 for j in range(n):
  eq.append([F(int(i==j))-a*(F(float(dense[i,j]))/strengths[i] if strengths[i] else F(1,n)) for i in range(n)]+[b/n])
 for i in range(n):
  pivot=eq[i][i];eq[i]=[x/pivot for x in eq[i]]
  for j in range(n):
   if j!=i:
    pivot=eq[j][i];eq[j]=[x-pivot*y for x,y in zip(eq[j],eq[i])]
 return [row[-1] for row in eq]

class Checks(unittest.TestCase):
 def setUp(self):self.factory=old.matrix_for
 def tearDown(self):old.matrix_for=self.factory
 def run_matrix(self,matrix,values=None,version=0):
  old.matrix_for=lambda row,ver,fn:matrix
  row=dict(n=matrix.shape[0],m=matrix.nnz)
  reference,_=old.build_reference(matrix,'weighted_pr',0)
  values=owned(reference if values is None else values)
  before=old.output_quality(values,row,version,'weighted_pr',0)
  result=cert.output_quality(old,values,row,version,'weighted_pr',0)
  self.assertTrue(result['passed']);self.assertTrue(before['passed'])
  self.assertNotIn('reference_l1',result)
  self.assertLessEqual(before['reference_l1'],result['reference_l1_upper_bound'])
  self.assertLessEqual(result['reference_l1_upper_bound'],1e-5)
  self.assertFalse(result['reference']['power_solve_executed'])
  return result,values
 def test_exact_rational_fixed_points_dangling_selfloops_hubs(self):
  fixtures=[np.zeros((1,1)),np.zeros((7,7)),np.eye(6)*16,
   np.array([[0,1,0],[0,0,16],[1,0,0]],dtype=float),
   np.array([[0,16,1,1],[0,0,0,0],[0,16,0,0],[0,16,0,0]],dtype=float)]
  for dense in fixtures:
   matrix=csr_matrix(dense);result,values=self.run_matrix(matrix)
   exact=exact_solution(matrix)
   error=sum(abs(F(float(v))-ref) for v,ref in zip(values,exact))
   self.assertLessEqual(error,F(result['reference']['proof']['exact_fixed_point_l1_upper_bound']))
 def test_seeded_sparse_weighted_graphs_and_mutations(self):
  rng=np.random.default_rng(20261106)
  for n in (2,7,29,97):
   for version in range(8):
    mask=rng.random((n,n))<.08;data=rng.integers(1,17,size=(n,n))*mask
    data=np.where(data,1+(data-1+version)%16,0)
    self.run_matrix(csr_matrix(data,dtype=np.float64),version=version)
 def test_near_residual_boundary_and_bad_candidates(self):
  matrix=csr_matrix(np.eye(7));reference,_=old.build_reference(matrix,'weighted_pr',0)
  values=reference.copy();delta=.999e-6/(2*.15);values[0]+=delta;values[1]-=delta
  result,_=self.run_matrix(matrix,values);self.assertGreater(result['residual_l1'],.998e-6)
  row=dict(n=7,m=7)
  for case in (np.full(7,np.nan),np.full(7,np.inf),np.array([-1.]+[1/3]*6),reference*1.00001,reference+np.array([1e-3,-1e-3,0,0,0,0,0])):
   with self.assertRaises(ValueError):cert.output_quality(old,owned(case),row,0,'weighted_pr',0)
 def test_nonowning_or_writeable_output_rejected(self):
  row=dict(n=2,m=2);values=np.array([.5,.5])
  with self.assertRaises(ValueError):cert.output_quality(old,values,row,0,'weighted_pr',0)
  view=values[:];view.flags.writeable=False
  with self.assertRaises(ValueError):cert.output_quality(old,view,row,0,'weighted_pr',0)
 def test_outside_weight_domain_falls_back_to_unchanged_reference(self):
  matrix=csr_matrix(np.array([[0,.5],[1,0]]));old.matrix_for=lambda *args:matrix
  reference,_=old.build_reference(matrix,'weighted_pr',0)
  result=cert.output_quality(old,owned(reference),dict(n=2,m=2),0,'weighted_pr',0)
  self.assertIn('reference_l1',result);self.assertEqual(result['audit_method']['used'],'original_independent_reference')
 def test_near_bound_falls_back_and_other_algorithms_remain_exact(self):
  matrix=csr_matrix(np.eye(2));old.matrix_for=lambda *args:matrix
  original=cert.bounds
  try:
   cert.bounds=lambda *args,**kwargs:dict(certificate_pass=False)
   result=cert.output_quality(old,owned([.5,.5]),dict(n=2,m=2),0,'weighted_pr',0)
   self.assertIn('reference_l1',result);self.assertEqual(result['audit_method']['used'],'original_independent_reference')
  finally:cert.bounds=original
  result=cert.output_quality(old,owned([0.,np.inf]),dict(n=2,m=2),0,'bfs',0)
  self.assertTrue(result['exact'])
 def test_large_count_bounds_and_upward_rounding_without_large_arrays(self):
  tw=cert.bounds(41652230,1468364884,1e-6,1.000001,old.QUALITY)
  self.assertTrue(tw['certificate_pass']);self.assertLess(tw['reference_l1_upper_bound'],8.57e-6)
  huge=cert.bounds(2**31-1,2**31-1,1e-6,1.000001,old.QUALITY)
  self.assertFalse(huge['certificate_pass'])
  for value in (F(1,3),F(1,10**20),F(10**30,3)):
   self.assertGreaterEqual(F(cert.upward(value)),value)
  for n,m in ((0,1),(1,2**31)):
   with self.assertRaises(ValueError):cert.bounds(n,m,1e-6,1,old.QUALITY)

if __name__=='__main__':
 begin=time.monotonic();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 receipt=dict(schema='state.pr-certificate.cpu-tests.v1',status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,errors=len(r.errors),failures=len(r.failures),seconds=time.monotonic()-begin,random_graph_cases=32,exact_rational_cases=5,gpu_executed=False,remote_changes=False,numpy=np.__version__)
 (HERE/'cpu_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');raise SystemExit(0 if r.wasSuccessful() else 1)
