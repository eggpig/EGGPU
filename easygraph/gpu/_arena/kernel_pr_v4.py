"""Qualified PR candidate: warp-cooperative long rows without extra graph state.

All control/output scratch remains owned by the existing arena. No new device
arrays or result caches. Every PR step retains the original global-L1 stopping
rule. Row cooperation is selected from the current device CSR at execution.
"""
import time
from easygraph.gpu._arena.gpu_runtime import cp
from easygraph.gpu._arena.low_degree import LowDegreeExecutor,SOURCE

HYBRID=SOURCE+r'''
extern "C" __global__ void rank_hybrid(const int n,const int* row,const int* col,
 const double* weight,const double* scaled,const double* x,double* y,double* stat,
 const int* ctl,const double alpha,const int weighted) {
  if(ctl[0]) return;
  const int lane=threadIdx.x&31;
  const int base=((blockIdx.x*blockDim.x+threadIdx.x)>>5)*8;
  const int r=base+(lane>>2),sub=lane&3;
  const bool valid=r<n;
  const bool large=valid && row[r+1]-row[r]>128;
  double sum=0.,diff=0.;
  if(valid && !large)
    for(int e=row[r]+sub;e<row[r+1];e+=4)
      sum+=(weighted ? weight[e] : 1.)*scaled[col[e]];
  sum+=__shfl_down_sync(0xffffffff,sum,2,4);
  sum+=__shfl_down_sync(0xffffffff,sum,1,4);
  if(valid && !large && sub==0) {
    double value=(1.-alpha)/n+alpha*(sum+stat[0]/n);
    y[r]=value;diff=fabs(value-x[r]);
  }
  unsigned large_rows=__ballot_sync(0xffffffff,large && sub==0);
  while(large_rows) {
    const int leader=__ffs(large_rows)-1;
    const int v=base+(leader>>2);sum=0.;
    for(int e=row[v]+lane;e<row[v+1];e+=32)
      sum+=(weighted ? weight[e] : 1.)*scaled[col[e]];
    sum=warp_sum(sum);
    if(lane==0) {
      double value=(1.-alpha)/n+alpha*(sum+stat[0]/n);
      y[v]=value;diff+=fabs(value-x[v]);
    }
    large_rows&=large_rows-1;
  }
  diff=block_sum(diff);
  if(threadIdx.x==0) atomicAdd(stat+1,diff);
}
'''

class HybridRowsExecutor(LowDegreeExecutor):
    def __init__(self,base,checkpoint_iterations=8,max_in_degree=16):
        super().__init__(base,checkpoint_iterations=checkpoint_iterations,vector_width=4,max_in_degree=max_in_degree)

    def _load(self):
        if self._kernels is not None:return 0.
        start=time.perf_counter();module=cp.RawModule(code=HYBRID,options=('--std=c++17',),backend='nvrtc')
        self._kernels={n:module.get_function(n) for n in ('init_lean','scale','finish_lean','fused4')}
        self._kernels['rank4']=module.get_function('rank_hybrid')
        return time.perf_counter()-start

    def _rank(self,*args,**kwargs):
        output,metrics=super()._rank(*args,**kwargs)
        metrics['row_cooperation']='4 lanes for degree<=128;32 lanes for longer rows within same warp'
        metrics['additional_device_array_bytes']=0
        return output,metrics
