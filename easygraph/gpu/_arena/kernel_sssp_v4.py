"""Frozen-domain sparse SSSP with wider cooperation and cached static SM counts.

The original structural eligibility rule is unchanged: n>=256, m/n<=4,
maximum out-degree<=128. The eligible path uses4 lanes per vertex and a launch
cap of8 blocks per SM. All other inputs retain the original parallel-delta
implementation. Graph state/scratch accounting and exact output semantics are
inherited, with no final-answer cache and no extra explicit device array.
"""
import threading
from easygraph.gpu._arena.gpu_runtime import cp
from easygraph.gpu._arena.sssp_frontier import NativeFrontierSSSP
from easygraph.gpu._arena.sssp_frontier_warp_v2 import WarpFrontierSSSPV2
from easygraph.gpu._arena.sssp_final import QualifiedSSSP
_SM_COUNTS={}
_SM_LOCK=threading.Lock()
def static_sm_count():
    device=int(cp.cuda.Device().id)
    with _SM_LOCK:
        if device not in _SM_COUNTS:
            _SM_COUNTS[device]=int(cp.cuda.runtime.getDeviceProperties(device)['multiProcessorCount'])
        return _SM_COUNTS[device]

class SparseCooperativeSSSP(QualifiedSSSP):
    def __init__(self,base,checkpoint_iterations=16):
        NativeFrontierSSSP.__init__(self,base,checkpoint_iterations,1)
        sms=static_sm_count()
        self.fast=WarpFrontierSSSPV2(base,checkpoint_iterations,vector_width=4,grid_cap=8*sms)
        self.max_mean_degree=4.;self.max_out_degree=128;self.min_vertices=256;self._degree_kernel=None

    def run(self,*args,**kwargs):
        result=super().run(*args,**kwargs)
        self.last['sparse_launch_configuration']={'vector_width':4,'blocks_per_sm':8,'sm_count_cached_per_cuda_device':True}
        return result
