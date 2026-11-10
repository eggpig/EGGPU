"""Structural SSSP dispatch between exact frontier and native parallel delta.

Low-density bounded-degree graphs benefit from frontier propagation without
global delta scans. Hubs and denser graphs retain native parallel delta. The
predicate uses current CSR structure on every request, so updates and arena
address reuse cannot retain a stale dispatch decision. No graph-name lookup,
answer caching or additional explicit device allocation is used.
"""
import time
import numpy as np
from easygraph.gpu._arena.gpu_runtime import cp
from easygraph.gpu._arena.sssp_frontier import NativeFrontierSSSP
from easygraph.gpu._arena.sssp_frontier_warp_v2 import WarpFrontierSSSPV2

DEGREE_SOURCE=r'''
extern "C" __global__ void degree_bound(const int n,const int* row,int* answer) {
  __shared__ int values[128];
  int m=0;
  for(int i=threadIdx.x;i<n;i+=128) m=max(m,row[i+1]-row[i]);
  values[threadIdx.x]=m;__syncthreads();
  for(int stride=64;stride;stride>>=1) {
    if(threadIdx.x<stride) values[threadIdx.x]=max(values[threadIdx.x],values[threadIdx.x+stride]);
    __syncthreads();
  }
  if(threadIdx.x==0) answer[0]=values[0];
}
'''

class TunedFrontierSSSP(WarpFrontierSSSPV2):
    def __init__(self,base,checkpoint_iterations=16):
        # One lane per low-degree frontier vertex; one resident block per SM
        # supplies enough independent work while bounding launch overhead.
        props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        super().__init__(base,checkpoint_iterations,1,int(props['multiProcessorCount']))

class PolicyFrontierSSSP(NativeFrontierSSSP):
    def __init__(self,base,checkpoint_iterations=16,max_mean_degree=4.,max_out_degree=16,min_vertices=256):
        super().__init__(base,checkpoint_iterations,1)
        self.fast=TunedFrontierSSSP(base,checkpoint_iterations)
        self.max_mean_degree=float(max_mean_degree)
        self.max_out_degree=int(max_out_degree)
        self.min_vertices=int(min_vertices)
        self._degree_kernel=None

    def run(self,row,col,weight,sources,warp_size=0):
        started=time.perf_counter();n=row.size-1
        inspect=n>=self.min_vertices and col.size<=self.max_mean_degree*n
        degree=None;selection_d2h=0
        if inspect:
            if self._degree_kernel is None:
                self._degree_kernel=cp.RawKernel(DEGREE_SOURCE,'degree_bound',options=('--std=c++17',))
            control=self.workspace[:4].view(cp.int32)
            self._degree_kernel((1,),(128,),(np.int32(n),row,control))
            degree=int(control.get()[0]);selection_d2h=4
        use_frontier=inspect and degree<=self.max_out_degree
        selection=time.perf_counter()-started
        chosen=self.fast if use_frontier else self.base
        answer=chosen.run(row,col,weight,sources,warp_size)
        self.last=dict(chosen.last)
        self.last.update(native_execute_seconds=time.perf_counter()-started,
            control_d2h_bytes=self.last['control_d2h_bytes']+selection_d2h,
            selected_kernel='frontier' if use_frontier else 'parallel_delta',
            structural_selection_seconds=selection,max_current_out_degree=degree,
            selection_rule=dict(max_mean_degree=self.max_mean_degree,max_out_degree=self.max_out_degree,min_vertices=self.min_vertices),
            selection_cached=False,additional_device_bytes=0)
        return answer
