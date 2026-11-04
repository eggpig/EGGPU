"""Independent borrowed-Arena SSSP candidate; preserve sparse v4 dispatch."""
import ctypes
import time
from pathlib import Path
import os
from easygraph.gpu._arena.gpu_runtime import cp
from easygraph.gpu._arena.sssp_frontier import NativeFrontierSSSP
from easygraph.gpu._arena.kernel_sssp_v4 import static_sm_count

DEFAULT_LIBRARY=Path(os.environ.get('EGGPU_ARENA_LIBRARY_DIR', str(Path(__file__).with_name('lib'))))/'libdevice_delta_ieee_v3.so'


class IEEEDeltaCore(NativeFrontierSSSP):
    def __init__(self,base,checkpoint_iterations=16,width=32,blocks_per_sm=8,force=True,library=DEFAULT_LIBRARY):
        super().__init__(base,checkpoint_iterations,1)
        self.width=int(width);self.launch_cap=int(blocks_per_sm)*static_sm_count();self.force=bool(force)
        self._delta_lib=ctypes.CDLL(str(library))
        f=self._delta_lib.eggpu_device_delta_ieee_v3
        f.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*6+[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64)]+[ctypes.POINTER(ctypes.c_int)]*3
        f.restype=ctypes.c_int

    @property
    def workspace_bytes(self):return self.base.workspace_bytes
    @workspace_bytes.setter
    def workspace_bytes(self,value):self.base.workspace_bytes=value

    def run(self,row,col,weight,sources,warp_size=0):
        n,m,count=row.size-1,col.size,sources.size
        if n>self.nmax or count>self.source_capacity or weight.size!=m:
            raise ValueError('Request exceeds original declared workspace or weight shape')
        for a,dtype in ((row,cp.int32),(col,cp.int32),(sources,cp.int32),(weight,cp.float64)):
            if a.dtype!=dtype or a.ndim!=1 or not a.flags.c_contiguous or a.device.id!=cp.cuda.runtime.getDevice():
                raise TypeError('Contiguous current-device int32 CSR/source and float64 weights required')
        width=self.width or warp_size
        if not width:
            width=max(w for w in (1,2,4,8,16,32) if w<=max(1,m//max(1,n)))
        d2h=ctypes.c_uint64();rounds=ctypes.c_int();queued=ctypes.c_int();checks=ctypes.c_int()
        start=time.perf_counter()
        code=self._delta_lib.eggpu_device_delta_ieee_v3(row.data.ptr,col.data.ptr,weight.data.ptr,sources.data.ptr,
            n,m,count,width,self.checkpoint_iterations,self.launch_cap,
            self.workspace.data.ptr,self.workspace.nbytes,self.output.data.ptr,cp.cuda.get_current_stream().ptr,
            ctypes.byref(d2h),ctypes.byref(rounds),ctypes.byref(queued),ctypes.byref(checks))
        if code:raise RuntimeError('Device delta CUDA status '+str(code))
        self.last=dict(native_execute_seconds=time.perf_counter()-start,control_d2h_bytes=d2h.value,
            expansion_rounds=rounds.value,queued_rounds=queued.value,control_checkpoints=checks.value,
            checkpoint_iterations=self.checkpoint_iterations,source_count=count,vertices=n,edges=m,
            warp_size=width,launch_cap=self.launch_cap,workspace_capacity_bytes=self.workspace_bytes,
            kernel_variant='device_control_ieee_parallel_delta',min_edge_rebuilds=int(bool(n and count)),
            additional_device_bytes=0,device_delta_used=True,result_cache_used=False,
            distance_encoding='ieee754_float64_positive_infinity',overflow_policy='IEEE754 positive infinity for unrepresentable path sums')
        return self.output[:n*count].reshape(count,n)


from easygraph.gpu._arena.kernel_sssp_v4 import SparseCooperativeSSSP


class DeviceResidentDeltaSSSP(SparseCooperativeSSSP):
    """Same sparse structural rule; every other request uses IEEE device delta."""
    def __init__(self, base):
        from easygraph.gpu._arena.kernel_sssp_v4 import SparseCooperativeSSSP
        # Existing candidates already use SparseCooperativeSSSP. Reuse its raw
        # storage holder, never its legacy finite-sentinel fallback algorithm.
        raw = base.base if isinstance(base, SparseCooperativeSSSP) else base
        super().__init__(IEEEDeltaCore(raw))

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        self.last['distance_encoding'] = 'ieee754_float64_positive_infinity'
        self.last['device_delta_used'] = self.last['selected_kernel'] == 'parallel_delta'
        self.last['overflow_policy'] = 'IEEE754 positive infinity for unrepresentable path sums'
        return result
