"""Build and call original native EGGPU BFS/SSSP kernels with borrowed storage.

The native translation unit is generated from an exact, hashed prefix of the
repository CUDA file. The wrapper owns no CUDA arrays and does not import CuPy
during --build. All output and scratch allocations use the caller's Arena.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'native/sssp_dijkstra.cu'
DEFAULT_BUILD = Path(os.environ.get('EGGPU_ARENA_LIBRARY_DIR', str(ROOT/'lib')))
DEFAULT_NVCC = os.environ.get('NVCC', 'nvcc')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output_dir=DEFAULT_BUILD, nvcc=DEFAULT_NVCC):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp = output_dir / 'tmp'; temp.mkdir(exist_ok=True)
    original = SOURCE.read_bytes()
    marker = b'static inline int ensure_device_buffer('
    if original.count(marker) != 1:
        raise RuntimeError('Kernel extraction marker changed; inspect source before building')
    prefix = original.split(marker)[0]
    expected = (b'd_sssp_unweighted_bfs', b'd_bfs_frontier_init',
                b'd_bfs_expand_frontier', b'd_bfs_int_to_double', b'd_calc_min_edge',
                b'd_sssp_delta_init', b'd_sssp_delta_relax',
                b'd_sssp_delta_reduce_partial', b'd_sssp_delta_build_frontier')
    if not all(name in prefix for name in expected):
        raise RuntimeError('Original BFS/SSSP kernel definitions missing')
    include = output_dir / 'native_kernel_prefix.inc'
    include.write_bytes(prefix + b'\n} // namespace gpu_easygraph\n')
    library = output_dir / 'libeggpu_native_bfs.so'
    candidate = output_dir / ('libeggpu_native_bfs.' + str(os.getpid()) + '.so')
    common = ROOT / 'native/common'
    command = [str(nvcc), '-O3', '--shared', '-std=c++17', '-arch=sm_'+os.environ.get('EGGPU_CUDA_ARCH','80'),
               '-Xcompiler=-fPIC', '-I'+str(output_dir), '-I'+str(common),
               str(ROOT/'native/native_bridge.cu'), '-o', str(candidate)]
    env = os.environ.copy(); env['TMPDIR'] = str(temp)
    start = time.perf_counter()
    done = subprocess.run(command, env=env, text=True, capture_output=True)
    (output_dir/'build.log').write_text(done.stdout + done.stderr)
    if done.returncode:
        raise RuntimeError(f'Native bridge build failed; see {output_dir / "build.log"}')
    version = subprocess.run([str(nvcc),'--version'], text=True, capture_output=True, check=True).stdout
    symbols = subprocess.run(['nm','-D',str(candidate)], text=True,capture_output=True,check=True).stdout
    if 'cudaMalloc' in symbols or 'cudaFree' in symbols:
        raise RuntimeError('Borrowed-storage bridge unexpectedly links an allocation operation')
    candidate.replace(library)  # Existing processes keep their mapped original inode.
    record = dict(source=str(SOURCE), source_sha256=hashlib.sha256(original).hexdigest(),
                  original_kernel_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
                  generated_include_sha256=sha(include), bridge_sha256=sha(ROOT/'native/native_bridge.cu'),
                  library=str(library), library_sha256=sha(library), compiler=version,
                  command=command, build_seconds=time.perf_counter()-start,
                  no_cuda_malloc_free_import=True, original_kernel_names=[name.decode() for name in expected])
    (output_dir/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


class NativeBFS:
    def __init__(self, arena, nmax, source_capacity=1, library=None):
        import numpy as np
        self.arena, self.nmax, self.source_capacity = arena, int(nmax), int(source_capacity)
        self.lib = ctypes.CDLL(str(library or DEFAULT_BUILD/'libeggpu_native_bfs.so'))
        self.lib.eggpu_native_bfs_workspace_bytes.argtypes=[ctypes.c_int,ctypes.c_int]
        self.lib.eggpu_native_bfs_workspace_bytes.restype=ctypes.c_size_t
        f=self.lib.eggpu_native_bfs_prepared
        f.argtypes=[ctypes.c_void_p]*3 + [ctypes.c_int]*5 + [ctypes.c_void_p,ctypes.c_size_t,
                    ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_int)]
        f.restype=ctypes.c_int
        need=self.lib.eggpu_native_bfs_workspace_bytes(nmax,source_capacity)
        if not need: raise ValueError('Invalid native BFS workspace shape')
        before=arena.live
        self.workspace=arena.allocate('@native_bfs_workspace',(need,),np.uint8)
        self.output=arena.allocate('@native_bfs_output',(source_capacity*nmax,),np.float64)
        self.workspace_bytes=arena.live-before
        self.last={}

    def run(self, row, col, sources, target=-1, scan_mode=-1):
        import cupy as cp
        import numpy as np
        n, m, count = row.size-1, col.size, sources.size
        if n<0 or n>self.nmax or count>self.source_capacity:
            raise ValueError('Request exceeds declared workspace capacity')
        for array in (row,col,sources):
            if array.dtype!=np.int32 or array.ndim!=1 or not array.flags.c_contiguous:
                raise TypeError('Native BFS requires contiguous one-dimensional int32 CSR/sources')
            if array.device.id!=cp.cuda.runtime.getDevice():
                raise ValueError('All borrowed arrays must belong to the active GPU')
        control=ctypes.c_uint64(); rounds=ctypes.c_int()
        stream=cp.cuda.get_current_stream()
        start=time.perf_counter()
        status=self.lib.eggpu_native_bfs_prepared(row.data.ptr,col.data.ptr,sources.data.ptr,
            n,m,count,target,scan_mode,self.workspace.data.ptr,self.workspace.nbytes,
            self.output.data.ptr,stream.ptr,ctypes.byref(control),ctypes.byref(rounds))
        if status:
            raise RuntimeError('Native BFS CUDA status '+str(status))
        self.last=dict(native_execute_seconds=time.perf_counter()-start,
                       control_d2h_bytes=control.value,expansion_rounds=rounds.value,
                       source_count=count,vertices=n,edges=m,scan_mode=scan_mode,
                       workspace_capacity_bytes=self.workspace_bytes)
        return self.output[:count*n].reshape((count,n))


class NativeSSSP:
    """Unchanged native parallel-delta kernels, with positive float64 weights.

    The graph producer validates CSR indices, sources, and finite nonnegative
    weights. Output is source-major float64 and uses native 1e100 for unreachable
    vertices. Each call rebuilds the min-edge vector in its reserved workspace.
    """
    def __init__(self, arena, nmax, source_capacity=1, library=None):
        import numpy as np
        self.arena, self.nmax, self.source_capacity = arena, int(nmax), int(source_capacity)
        self.lib = ctypes.CDLL(str(library or DEFAULT_BUILD/'libeggpu_native_bfs.so'))
        self.lib.eggpu_native_sssp_workspace_bytes.argtypes=[ctypes.c_int,ctypes.c_int]
        self.lib.eggpu_native_sssp_workspace_bytes.restype=ctypes.c_size_t
        f=self.lib.eggpu_native_sssp_prepared
        f.argtypes=([ctypes.c_void_p]*4 + [ctypes.c_int]*4 +
                    [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,
                     ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,
                     ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_int)])
        f.restype=ctypes.c_int
        need=self.lib.eggpu_native_sssp_workspace_bytes(self.nmax,self.source_capacity)
        if not need: raise ValueError('Invalid native SSSP workspace shape')
        before=arena.live
        self.workspace=arena.allocate('@native_sssp_workspace',(need,),np.uint8)
        self.output=arena.allocate('@native_sssp_output',(self.source_capacity*self.nmax,),np.float64)
        self.host_partial=np.empty(min(4096,max(1,(self.nmax+255)//256)),np.float64)
        self.workspace_bytes=arena.live-before
        self.last={}

    def run(self, row, col, weight, sources, warp_size=0):
        import cupy as cp
        import numpy as np
        n,m,count=row.size-1,col.size,sources.size
        if n<0 or n>self.nmax or count>self.source_capacity or weight.size!=m:
            raise ValueError('Request exceeds declared capacity or mismatches edge weights')
        for array,dtype in ((row,np.int32),(col,np.int32),(weight,np.float64),(sources,np.int32)):
            if array.dtype!=dtype or array.ndim!=1 or not array.flags.c_contiguous:
                raise TypeError('Native SSSP requires contiguous int32 CSR/sources and float64 weights')
            if array.device.id!=cp.cuda.runtime.getDevice():
                raise ValueError('All borrowed arrays must belong to the active GPU')
        if warp_size==0:
            avg=m//max(1,n)
            warp_size=max(v for v in (1,2,4,8,16,32) if v<=max(1,avg))
        control=ctypes.c_uint64();rounds=ctypes.c_int()
        stream=cp.cuda.get_current_stream()
        start=time.perf_counter()
        status=self.lib.eggpu_native_sssp_prepared(row.data.ptr,col.data.ptr,weight.data.ptr,sources.data.ptr,
            n,m,count,warp_size,self.workspace.data.ptr,self.workspace.nbytes,self.output.data.ptr,
            self.host_partial.ctypes.data,self.host_partial.nbytes,stream.ptr,
            ctypes.byref(control),ctypes.byref(rounds))
        if status: raise RuntimeError('Native SSSP CUDA status '+str(status))
        self.last=dict(native_execute_seconds=time.perf_counter()-start,control_d2h_bytes=control.value,
            expansion_rounds=rounds.value,source_count=count,vertices=n,edges=m,warp_size=warp_size,
            workspace_capacity_bytes=self.workspace_bytes,host_workspace_bytes=self.host_partial.nbytes,
            kernel_variant='original_parallel_delta',min_edge_rebuilds=1 if n and count else 0)
        return self.output[:count*n].reshape((count,n))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--build',action='store_true')
    p.add_argument('--output-dir',type=Path,default=DEFAULT_BUILD)
    p.add_argument('--nvcc',type=Path,default=DEFAULT_NVCC)
    a=p.parse_args()
    if a.build:
        result=build(a.output_dir,a.nvcc)
        print(json.dumps({k:result[k] for k in ('library','library_sha256','build_seconds','no_cuda_malloc_free_import')},indent=2))
    else:p.error('use --build; import NativeBFS/NativeSSSP with a caller-owned arena')
