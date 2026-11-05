#!/usr/bin/env python3
"""Portable clean build wrapper; no inherited machine paths or GPU execution."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
import verify
ROOT=Path(__file__).resolve().parents[3]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--cuda-root',type=Path,required=True);p.add_argument('--architecture',required=True)
    p.add_argument('--nccl-include',type=Path);p.add_argument('--nccl-library',type=Path)
    p.add_argument('--cmake',default='cmake');p.add_argument('--jobs',type=int,choices=(1,2),default=2)
    p.add_argument('--arena',action='store_true',help='Build the libraries required by the public single-GPU owned-CSR route as well as the native extension')
    p.add_argument('--execute',action='store_true');a=p.parse_args();m=verify.verify(ROOT)
    if bool(a.nccl_include)!=bool(a.nccl_library):p.error('NCCL requires both include and library')
    if not a.architecture.isdigit():p.error('Explicit numeric CUDA architecture required, for example 89')
    out=a.build_dir.resolve()
    if out.exists() or out==ROOT or ROOT in out.parents:raise ValueError('An absent build directory outside the package is required')
    command=[sys.executable,'-B',str(ROOT/'Easy-Graph/scripts/gpu/build.py'),'--build-dir',str(out),'--cuda','--cuda-root',str(a.cuda_root.resolve()),'--architecture',a.architecture,'--jobs',str(a.jobs),'--cmake',a.cmake,
        '--cmake-arg=-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','--cmake-arg=-DPYBIND11_TEST=OFF']
    if a.nccl_library:command+=['--nccl-include',str(a.nccl_include.resolve()),'--nccl-library',str(a.nccl_library.resolve())]
    if a.arena:command+=['--arena']
    result=dict(schema='eggpu.portable.clean_build/v2',status='PLAN_ONLY_NOT_BUILT',base_source_sha256=m['base_source_sha256'],source_sha256=m['current_source_sha256'],command=command,
        gpu_executed=False,new_binary_gpu_qualified=False,qualification='Packaged original source evidence does not qualify this rebuilt DSO on this host.')
    if a.execute:
        begin=time.perf_counter();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',CMAKE_BUILD_PARALLEL_LEVEL=str(a.jobs),OMP_NUM_THREADS=str(a.jobs))
        process=subprocess.run(command,env=env);result.update(returncode=process.returncode,seconds=time.perf_counter()-begin)
        verify.verify(ROOT)
        result.update(status='PASS_BUILD_GPU_UNQUALIFIED' if not process.returncode else 'FAIL_BUILD',outputs=[dict(path=str(f),sha256=verify.sha(f)) for f in sorted(out.rglob('*.so'))])
        out.mkdir(exist_ok=True);(out/'portable_build_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        if process.returncode:raise SystemExit(process.returncode)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
