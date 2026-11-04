#!/usr/bin/env python3
"""Build optional native GPU support and arena libraries, without running CUDA.

Build outputs must be outside the source checkout. A receipt and logs are saved
beside the build. Invoke with the Python interpreter of the target environment.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, required=True)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--arena', action='store_true')
    p.add_argument('--cuda-root', type=Path)
    p.add_argument('--architecture', default='80')
    p.add_argument('--nccl-include', type=Path)
    p.add_argument('--nccl-library', type=Path)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--cmake', default='cmake')
    p.add_argument('--cmake-arg', action='append', default=[])
    a = p.parse_args()
    if a.jobs < 1 or a.jobs > 2:
        p.error('--jobs must be 1 or 2; raise the limit explicitly for a dedicated build host')
    if a.arena and not a.cuda:
        p.error('--arena requires --cuda')
    if bool(a.nccl_include) != bool(a.nccl_library):
        p.error('provide both NCCL include and library paths')
    dest = a.build_dir.resolve()
    if dest == ROOT or ROOT in dest.parents:
        p.error('--build-dir must be outside the source checkout')
    dest.mkdir(parents=True, exist_ok=True)
    logdir = dest / ('logs-' + str(time.time_ns())); logdir.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    rows = []
    def run(argv, name):
        start = time.perf_counter()
        with (logdir / name).open('w') as stream:
            proc = subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT)
        rows.append(dict(argv=argv, returncode=proc.returncode,
                         seconds=time.perf_counter()-start, log=str(logdir/name)))
        if proc.returncode:
            raise RuntimeError('Build failed; see ' + str(logdir/name))
    receipt = dict(gpu_executed=False, source=str(ROOT), steps=rows, status='RUNNING')
    try:
        cmake = [a.cmake, '-S', str(ROOT), '-B', str(dest/'native'),
                 '-DEASYGRAPH_ENABLE_GPU=' + ('ON' if a.cuda else 'OFF'),
                 '-DPYBIND11_FINDPYTHON=ON', '-DPython_EXECUTABLE='+sys.executable,
                 '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF']
        if a.cuda:
            cmake += ['-DCMAKE_CUDA_ARCHITECTURES='+a.architecture]
            if a.cuda_root:
                cmake += ['-DCUDAToolkit_ROOT='+str(a.cuda_root.resolve()),
                          '-DCMAKE_CUDA_COMPILER='+str(a.cuda_root.resolve()/'bin/nvcc')]
        if a.nccl_library:
            cmake += ['-DEGMGPU_ENABLE_NCCL=ON',
                      '-DEGMGPU_NCCL_INCLUDE_DIR='+str(a.nccl_include.resolve()),
                      '-DEGMGPU_NCCL_LIBRARY='+str(a.nccl_library.resolve())]
        cmake += a.cmake_arg
        run(cmake, 'configure.log')
        run([a.cmake, '--build', str(dest/'native'), '--parallel', str(a.jobs)], 'build.log')
        if a.arena:
            nvcc = str(a.cuda_root.resolve()/'bin/nvcc') if a.cuda_root else shutil.which('nvcc')
            if not nvcc: raise RuntimeError('nvcc not found')
            arena = ROOT/'easygraph/gpu/_arena'
            spec=importlib.util.spec_from_file_location('arena_native_build',arena/'native_bridge.py')
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            previous=os.environ.get('EGGPU_CUDA_ARCH')
            try:
                os.environ['EGGPU_CUDA_ARCH']=a.architecture
                receipt['arena_bridge']=module.build(dest/'arena',nvcc)
            finally:
                if previous is None: os.environ.pop('EGGPU_CUDA_ARCH',None)
                else: os.environ['EGGPU_CUDA_ARCH']=previous
            run([nvcc,'-O3','--shared','-std=c++17','-arch=sm_'+a.architecture,
                 '-Xcompiler=-fPIC',str(arena/'native/device_delta_ieee_v3.cu'),
                 '-o',str(dest/'arena/libdevice_delta_ieee_v3.so')], 'arena_delta.log')
        outputs=[p for p in dest.rglob('*.so') if p.is_file()]
        # RTLD_NOW catches unresolved project symbols without launching kernels.
        run([sys.executable, '-c',
             'import ctypes,os,sys; [ctypes.CDLL(p,mode=os.RTLD_NOW) for p in sys.argv[1:]]',
             *map(str,outputs)], 'load_only.log')
        source_paths=[]
        for folder in ('cpp_easygraph','gpu_easygraph','easygraph/gpu/_arena/native'):
            source_paths.extend(p for p in (ROOT/folder).rglob('*')
                                if p.is_file() and p.suffix in ('.cu','.cuh','.h','.cpp'))
        receipt.update(status='PASS', dynamic_load_only='PASS; RTLD_NOW; CUDA_VISIBLE_DEVICES empty',
                       source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted(source_paths)},
                       outputs={str(p.relative_to(dest)):digest(p) for p in outputs})
    except Exception as exc:
        receipt.update(status='FAILED',error=str(exc))
        raise
    finally:
        (logdir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('Build receipt:',logdir/'receipt.json')
    print('Native module directory:',dest/'native/cpp_easygraph')
    if a.arena: print('EGGPU_ARENA_LIBRARY_DIR:',dest/'arena')

if __name__=='__main__':main()
