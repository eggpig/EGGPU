from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,struct,os
import pytest
import gum_worker as worker
import gum_reused as reused
import native_worker,sg_worker as shared,run_gum

def test_native_precision_and_cli():
 a=SimpleNamespace(function='pr',mpiexec=Path('/mpi'),core_python=Path('/python'))
 c=reused.native_command(a,'/gum','/csr',16,0,8,Path('/out'))
 assert '--pr_mr=117' in c and '--ws=false' in c and '--pr_d=0.85' in c
 assert reused.TARGETS['pr']=='gum_pr_fp32'

def rows():
 return [dict(round=i,rank_bytes=4,message_bytes=8,iterations=117,alpha=struct.unpack('f',struct.pack('f',.85))[0]) for i in range(3)]

def test_native_observation_accepts_author_float_damping(tmp_path):
 (tmp_path/'timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows()))
 p=worker.precision_evidence(tmp_path)
 assert p['effective_author_alpha']!=.85 and p['quality_reference_alpha']==.85

@pytest.mark.parametrize('field,value',[('rank_bytes',8),('message_bytes',16),('iterations',64),('alpha',.85)])
def test_precision_substitution_rejected(tmp_path,field,value):
 r=rows();r[1][field]=value;(tmp_path/'timings.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in r))
 with pytest.raises(ValueError):worker.precision_evidence(tmp_path)

def test_missing_query_rejected(tmp_path):
 (tmp_path/'timings.jsonl').write_text(json.dumps(rows()[0])+'\n')
 with pytest.raises(ValueError):worker.precision_evidence(tmp_path)

def test_full_domain_and_quality_remain_common(tmp_path):
 native_worker.make_tiny(tmp_path);g=shared.load_input(tmp_path/'canonical/TINY_D')
 p,_=worker.descriptor(g,'pr',tmp_path);data=p.read_text().splitlines()
 assert data[1:4]==['16',str(g['m']),'0'] and data[-1]=='-'
 assert shared.QUALITY['pr_residual_l1']==1e-6 and shared.QUALITY['pr_mass_error']==1e-6 and shared.QUALITY['pr_reference_l1']==1e-5

def test_original_precision_candidate_rejects_formal():
 with pytest.raises(ValueError,match='qualification only'):worker.configure(SimpleNamespace(function='pr',phase='formal'))

def test_plan_has_no_gpu_execution(tmp_path):
 cpus=list(range(16))
 a=SimpleNamespace(proofs=None,counts='1,2,4,8',datasets='TINY_D,TINY_U',functions='pr',phase='tiny',repeat=0,output=tmp_path/'planned',core_python=Path('/python'),gpu_uuids=','.join('GPU-'+str(i) for i in range(8)),cpu_set=','.join(map(str,cpus)))
 with patch('subprocess.Popen',side_effect=AssertionError('No GPU during planning')), patch('os.sched_getaffinity',return_value=set(cpus)):jobs=run_gum.plan(a)
 assert len(jobs)==8 and not a.output.exists()

def test_no_gpu_without_optin():
 with pytest.raises(ValueError,match='execute-gpu'):worker.job(SimpleNamespace(execute_gpu=False))
