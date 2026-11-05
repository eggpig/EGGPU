#!/usr/bin/env python3
"""Exercise the actual production query method through its first native call.

Only native CUDA calls and output helpers are replaced with CPU test sentinels;
the production method's weighted/directed/projection admission checks execute
unchanged. This cannot qualify CUDA algorithms or physical sharing.
"""
from pathlib import Path
import argparse, ast, importlib.util, json, os, subprocess, sys, tempfile, threading, time, types, unittest
import numpy as np
import state_inputs, state_worker, state_protocol as c


class NativeCallObserved(Exception):
    def __init__(self,name,args,kwargs):
        super().__init__(name);self.args_seen=args;self.kwargs_seen=kwargs


def actual_policy(source):
    path=source/'easygraph/gpu/_core.py';tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_policy')
    scope={'os':os,'sys':sys,'importlib':importlib,'POLICY_SOURCE':source/'easygraph/egmgpu/sssp_policy.py'}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    return scope['native_policy']('f64')[0]


def certification_contract(source):
    """Compile/execute the exact production C++ admission block with no CUDA."""
    path=source/'cpp_easygraph/runtime/egmgpu_partitioned_sssp.cpp';text=path.read_text()
    begin=text.index('    const char* certification_setting = std::getenv(')
    end=text.index('    const std::string plan_variant =',begin)
    unit='#include <cstdlib>\n#include <cstdint>\n#include <cerrno>\n#include <stdexcept>\n#include <string>\nint main(){try{\n'+text[begin:end]+'\nreturn 0;}catch(const std::exception&){return 7;}}\n'
    with tempfile.TemporaryDirectory(prefix='state-native-policy-cpu-') as directory:
        cpp=Path(directory)/'contract.cpp';binary=Path(directory)/'contract';cpp.write_text(unit)
        subprocess.run(['c++','-std=c++11','-Wall','-Wextra',str(cpp),'-o',str(binary)],check=True,capture_output=True)
        env=dict(os.environ,**state_worker.COMMON_ENV,CUDA_VISIBLE_DEVICES='')
        for key in ('EGGPU_SSSP_DELTA_MAX_DENSITY_PPM','EGGPU_SSSP_DELTA_CALIBRATION_SHA256'):env.pop(key,None)
        if subprocess.run([str(binary)],env=env).returncode!=0:raise AssertionError('Frozen runner certification rejected by actual native block')
        env['EGGPU_SSSP_WEIGHT_CERTIFICATION']='adaptive'
        if subprocess.run([str(binary)],env=env).returncode!=7:raise AssertionError('Uncalibrated adaptive policy unexpectedly admitted')
        env.update(EGGPU_SSSP_DELTA_MAX_DENSITY_PPM='250000',EGGPU_SSSP_DELTA_CALIBRATION_SHA256='a'*64)
        if subprocess.run([str(binary)],env=env).returncode!=0:raise AssertionError('Complete adaptive identity rejected')
    return {'production_source':c.bind(path),'actual_extracted_block_sha256':__import__('hashlib').sha256(text[begin:end].encode()).hexdigest(),
            'cases':['runner_full_without_calibration_accepted','adaptive_missing_calibration_rejected','adaptive_complete_identity_accepted'],
            'gpu_executed':False,'passed':True}


def method_from_source(source):
    path=source/'easygraph/gpu/_native.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UnifiedNativeSession')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='query')
    scope={'perf_counter':time.perf_counter,'source_parameter':lambda p:p['source'],'os':os}
    flag_names={'_bfs_output_flags','_bfs_worker_output_flag','_wcc_worker_output_flag','_sssp_output_flag','_pr_output_flag'}
    flags=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in flag_names]
    if len(flags)!=len(flag_names):raise AssertionError('Missing native output policy function')
    exec(compile(ast.Module(body=flags+[method],type_ignores=[]),str(path),'exec'),scope)
    return scope['query'],c.bind(path)


def backend(document,policy):
    class Calls:
        def __getattr__(self,name):
            if not name.startswith('cpp_egmgpu_partitioned_'):raise AssertionError('Unexpected native entry '+name)
            def observed(*args,**kwargs):raise NativeCallObserved(name,args,kwargs)
            return observed
    return types.SimpleNamespace(_lock=threading.RLock(),closed=False,graphs={'g':object()},
             metadata={'g':document['graph']},_manifest_documents={'g':document},devices=[0],cpp=Calls(),
             sssp_policy=policy)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();query,binding=method_from_source(a.source)
    os.environ.update(state_worker.COMMON_ENV,CUDA_VISIBLE_DEVICES='')
    policy=actual_policy(a.source)
    if policy.delta!=16 or policy.weight_encoding!='f64' or policy.wire_encoding!='f64' or policy.single_owner_execution!='arena_v3_if_eligible':raise AssertionError('Frozen native policy changed')
    native_certification=certification_contract(a.source)
    inputs=state_worker.load_inputs(a.inputs);checks=[]
    for name,row in inputs['datasets'].items():
        weighted=json.loads(Path(row['manifest']['path']).read_text())
        for function in ('bfs','wcc','sssp','weighted_pr'):
            item=row['projections'][function]['manifest'] if function in row['projections'] else row['manifest']
            document=json.loads(Path(item['path']).read_text());obj=backend(document,policy)
            try:query(obj,'g',function,{'source':0} if function in ('bfs','sssp') else {})
            except NativeCallObserved as call:
                args,kw=call.args_seen,call.kwargs_seen
                if function=='sssp':
                    if args[1:]!=(0,policy.delta,0,[0]) or kw!={'dense_output':True,'execution_policy':policy.native_options()}:raise AssertionError('SSSP call options changed')
                elif function=='weighted_pr':
                    if args[1:]!=(.85,1000,1e-9/row['n'],[0]) or kw:raise AssertionError('PR call options changed')
                elif function=='bfs':
                    if args[1:]!=(0,) or kw!={'devices':[0],'dense_output':True,'frontier_policy':'adaptive'}:raise AssertionError('BFS call options changed')
                elif function=='wcc':
                    if args[1:]!=(100000,[0]) or kw!={'dense_output':True}:raise AssertionError('WCC call options changed')
                checks.append({'dataset':name,'function':function,'native_entry':str(call),'kwargs':kw,'all_native_options_checked':True,'passed':True})
            else:raise AssertionError('Did not reach the native entry')
        for function in ('bfs','wcc'):
            try:query(backend(weighted,policy),'g',function,{'source':0} if function=='bfs' else {})
            except ValueError as error:
                if 'Weighted graph requires explicit' not in str(error):raise
                checks.append({'dataset':name,'function':function,'weighted_rejected':True,'passed':True})
            else:raise AssertionError('Original invalid weighted dispatch was accepted')
        # Same source arrays remain byte-bound across the topology companion;
        # native's stricter compatibility key decides whether they share on GPU.
        plain=json.loads(Path(row['projections']['bfs']['manifest']['path']).read_text())
        for left,right in zip(weighted['partitions'],plain['partitions']):
            for key in ('offsets','indices','owners','local_to_global'):
                if left['files'][key]['sha256']!=right['files'][key]['sha256']:raise AssertionError('Topology companion changed a source array')
        checks.append({'dataset':name,'topology_payload_identity':True,'passed':True})
    receipt={'status':'PASS_CPU_PROJECTION_AND_POLICY_ADMISSION','gpu_executed':False,'tests':len(checks)+4,'checks':checks,
             'native_certification':native_certification,'frozen_policy':policy.document(),'environment':state_worker.COMMON_ENV,
             'files':[c.bind(Path(__file__)),c.bind(Path(state_worker.__file__)),c.bind(a.source/'easygraph/egmgpu/sssp_policy.py')],
             'production_method':binding,'inputs':c.bind(a.inputs),'native_calls':'Sentinel only; no CUDA invocation',
             'scope':'Actual unmodified UnifiedNativeSession.query admission; topology companion hashes; not physical-sharing qualification'}
    c.save(a.output,receipt);print(json.dumps({'status':receipt['status'],'tests':receipt['tests'],'receipt':str(a.output)}))


if __name__=='__main__':main()
