"""Explicit CPU placement must survive the fresh-process execution boundary."""
from pathlib import Path
import json,os,subprocess,sys
import pytest
HERE=Path(__file__).resolve().parent

def test_explicit_cpu_map_survives_profile_install_and_exec():
    code="""import native_profile as p,subprocess,sys,json
from types import SimpleNamespace
p.install(SimpleNamespace(native_profile=p.NAME,algorithm='bfs',count=8,communication='host_staging',sssp_encoding='u8_exact_u32_exact',sssp_delta_rule='certified_bound_plus_one'))
r=subprocess.check_output([sys.executable,'-B','-c',"import native_profile as p,json;print(json.dumps({'cpus':p.CPUS,'groups':p.environment('bfs',8)['EGMGPU_CORE_OUTPUT_CPUS']}))"],text=True)
print(r)
"""
    env=dict(os.environ,EGGPU_BENCH_CPU_SET=','.join(map(str,range(16))),CUDA_VISIBLE_DEVICES='')
    row=json.loads(subprocess.check_output([sys.executable,'-B','-c',code],cwd=HERE,env=env,text=True))
    assert row['cpus']==list(range(16)) and row['groups']=='0,1;2,3;4,5;6,7;8,9;10,11;12,13;14,15'

@pytest.mark.parametrize('value',['1,2,3',','.join(['1']*16)])
def test_invalid_cpu_map_rejected_on_import(value):
    r=subprocess.run([sys.executable,'-B','-c','import native_profile'],cwd=HERE,env=dict(os.environ,EGGPU_BENCH_CPU_SET=value),capture_output=True,text=True)
    assert r.returncode and 'exactly 16 distinct CPUs' in r.stderr
