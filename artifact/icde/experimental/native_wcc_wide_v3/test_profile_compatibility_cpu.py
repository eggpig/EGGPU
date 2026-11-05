"""Compile exact production WCC option validators; never initialize CUDA.

The owner-output header is included unchanged. The WCC structs/functions and
pre-device parameter block are extracted verbatim from the bound parent source.
Only hardware device discovery is replaced by the requested count; actual host
CPU affinity is observed by the unmodified production parser.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
PARENT=Path(os.environ['EGGPU_WCC_V2_SOURCE'])
SOURCE=PARENT/'Easy-Graph/cpp_easygraph/runtime'
sys.path.insert(0,str(HERE/'runner'))
import native_worker as worker


def extract_block(text,start):
    begin=text.index(start);brace=text.index('{',begin);depth=1;end=brace+1
    while depth:
        if text[end]=='{':depth+=1
        if text[end]=='}':depth-=1
        end+=1
    return text[begin:end]


@pytest.fixture(scope='session')
def compiled(tmp_path_factory):
    root=tmp_path_factory.mktemp('actual_wcc_config');source=(SOURCE/'egmgpu_partitioned_wcc.cpp').read_text()
    chunks=[extract_block(source,'struct WccOutputOptions')+';',
        extract_block(source,'WccOutputOptions wcc_output_options(bool forest)'),
        extract_block(source,'bool wcc_boundary_csr(bool forest)'),
        extract_block(source,'bool wcc_unique_local_edges(bool forest)'),
        extract_block(source,'bool single_partition_candidate_enabled()')]
    block=source[source.index('    const char* reuse_setting = std::getenv("EGMGPU_WCC_UPLOAD_REUSE");'):
                 source.index('    *stage = "resolve_devices";',source.index('py::dict run_partitioned_wcc_impl('))]
    body='\n'.join(['#include "egmgpu_owner_output_team.h"','#include <iostream>',*chunks,
        'int main(int argc,char** argv) { try {',
        'const bool dense_output=true;',
        'const bool public_fp64_output=std::string(std::getenv("EGGPU_NATIVE_WCC_WORKER_FP64_OUTPUT"))=="1";',
        'const std::size_t count=std::stoul(argv[1]);',block,
        'const auto placement=easygraph::egmgpu::owner_output::parse_placement(core_output_team16,count);',
        '(void)single_partition_candidate_enabled();',
        'std::cout << "PASS " << algorithm << " " << output.assembly << " " << output.merge << " " << (placement.explicit_binding?"explicit":"inherited") << "\\n";',
        'return 0; } catch(const std::exception& e) { std::cerr << e.what() << "\\n"; return 3; } }'])
    cpp=root/'actual_production_configuration.cpp';cpp.write_text(body);binary=root/'actual_production_configuration'
    command=['g++','-std=c++14','-O0','-fopenmp','-DEASYGRAPH_USE_OPENMP=1','-I',str(SOURCE),str(cpp),'-o',str(binary)]
    result=subprocess.run(command,text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stderr
    return binary


def configuration(variant,count=1,**updates):
    args=worker.parser().parse_args(['--algorithm','wcc','--dataset','TINY_D','--phase','tiny',
        '--count',str(count),'--gpu-uuids',','.join('GPU-'+str(i) for i in range(count)),
        '--wcc-qualification-variant',variant])
    for key,value in updates.items():setattr(args,key,value)
    return args


@pytest.fixture(autouse=True)
def restore_environment():
    before=dict(os.environ)
    yield
    os.environ.clear();os.environ.update(before)


def audit(compiled,count):
    # This tiny CPU process only parses placement; it never starts a team or GPU.
    # Observe the actual declared host budget, rather than stubbing affinity.
    return subprocess.run(['/usr/bin/taskset','-c','0,1,2,3,4,5,6,7,48,49,50,51,52,53,54,55',
                           str(compiled),str(count)],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},
                          text=True,capture_output=True,timeout=10)


@pytest.mark.parametrize('count',[1,2,4,8])
@pytest.mark.parametrize('variant',['forest_coo','forest_csr','label_coo'])
def test_complete_actual_cpp_configuration_accepts_all_tiny_variants(compiled,count,variant):
    args=worker.configure(configuration(variant,count));result=audit(compiled,count)
    assert result.returncode==0,result.stderr
    profile=worker.profile_record(args)
    assert profile['native_profile']['environment']==profile['qualification_effective_environment']
    assert profile['base_profile']==worker.native_profile.identity(args)
    if variant=='label_coo':
        assert result.stdout=='PASS label caller legacy inherited\n'
        assert profile['base_profile']['environment']['EGMGPU_CORE_OUTPUT_CPU_PLACEMENT']=='explicit'
        assert profile['qualification_effective_environment']['EGMGPU_CORE_OUTPUT_CPU_PLACEMENT']=='inherited'
        assert profile['qualification_effective_environment']['EGMGPU_CORE_OUTPUT_CPUS']==''
    else:assert profile['native_profile']['environment']==profile['base_profile']['environment'] or variant=='forest_csr'


@pytest.mark.parametrize('count',[1,2,4,8])
def test_real_cpp_parser_reproduces_v2_failure_then_accepts_v3(compiled,count):
    spec=importlib.util.spec_from_file_location('_old_wcc_v2',PARENT/'runner/native_worker.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.configure(configuration('label_coo',count));before=audit(compiled,count)
    assert before.returncode==3 and 'explicit output CPU placement requires team16' in before.stderr
    worker.configure(configuration('label_coo',count));after=audit(compiled,count)
    assert after.returncode==0,after.stderr


@pytest.mark.parametrize('key,value,reason',[
    ('EGMGPU_CORE_OUTPUT_CPU_PLACEMENT','explicit','requires team16'),
    ('EGMGPU_CORE_OUTPUT_TEAM','team16','requires pinned worker or tiled'),
    ('EGMGPU_WCC_HOST_ASSEMBLY','worker','requires forest and canonical_direct'),
    ('EGMGPU_WCC_RESULT_MERGE','canonical_direct','requires the converged forest'),
    ('EGMGPU_WCC_BOUNDARY_REPRESENTATION','csr','requires the forest'),
    ('EGMGPU_WCC_LOCAL_UNION','unique_edges','requires forest'),
    ('EGGPU_NATIVE_WCC_WORKER_FP64_OUTPUT','1','requires canonical forest'),
])
def test_each_forest_only_option_rejected_by_actual_cpp_parser(compiled,key,value,reason):
    worker.configure(configuration('label_coo'));os.environ[key]=value
    result=audit(compiled,1);assert result.returncode==3 and reason in result.stderr


@pytest.mark.parametrize('count',[1,2,4,8])
@pytest.mark.parametrize('function',['bfs','sssp','wcc','pr'])
def test_next_function_restores_entire_original_profile(count,function):
    worker.configure(configuration('label_coo',count))
    args=configuration('forest_coo',count,algorithm=function,communication=None)
    worker.configure(args);expected=worker.native_profile.environment(function,count)
    assert {key:os.environ[key] for key in expected}==expected
    assert worker.profile_record(args)['native_profile']['environment']==expected


@pytest.mark.parametrize('variant',['label_coo','forest_csr'])
@pytest.mark.parametrize('phase',['target','formal'])
def test_branch_overrides_remain_tiny_only(variant,phase):
    with pytest.raises(ValueError,match='tiny qualification only'):
        worker.configure(configuration(variant,phase=phase,dataset='TW'))


@pytest.mark.parametrize('fault',['none','placement','team','union','mutable','borrowed','dtype'])
def test_actual_label_metrics_and_public_ownership_are_required(fault):
    values=np.arange(4,dtype='f8');values.flags.writeable=False
    if fault=='mutable':values=values.copy()
    if fault=='borrowed':values=values.view()
    if fault=='dtype':values=np.arange(4,dtype='i8');values.flags.writeable=False
    layout=[dict(partition_id=0,dtype='int64',bytes=40,sha256='fixture')]
    metrics=dict(devices_used=[0],wcc_offset_layouts=layout,wcc_offset_policy='adaptive_signed_i32_i64_wcc_v1',
        wcc_algorithm='label',boundary_representation='coo',host_assembly_strategy='caller',result_merge_mode='legacy',
        core_output_team_policy='original',local_union_mode='all_arcs',host_output_mode='pinned',
        host_output_pinned_placement='numa_local',core_output_cpu_placement=dict(mode='inherited',worker_cpus=[]))
    if fault=='placement':metrics['core_output_cpu_placement']['mode']='explicit'
    if fault=='team':metrics['core_output_team_policy']='team16'
    if fault=='union':metrics['local_union_mode']='unique_edges'
    backend=worker.NativeBackend.__new__(worker.NativeBackend)
    backend.args=configuration('label_coo');backend.algorithm='wcc';backend.source=None
    backend.count=1;backend.expected_offset_layouts=layout;backend.last_metrics={}
    backend.session=SimpleNamespace(query=lambda *unused:SimpleNamespace(values=values,backend_metrics=metrics))
    if fault=='none':assert backend.query() is values
    else:
        with pytest.raises(ValueError):backend.query()
