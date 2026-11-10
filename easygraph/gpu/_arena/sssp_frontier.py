"""Borrowed-workspace exact nonnegative SSSP with batched host notification.

This independent candidate replaces NativeSSSP.run, not a baseline algorithm.
Every strict relaxation schedules its target for a later frontier. One epoch
tag per vertex bounds each frontier by n. Empty-next-frontier is an exact fixed
point; delayed host observation changes neither values nor stopping semantics.
All scratch and output are views of the original native SSSP arena allocations.
"""
import time
import numpy as np
from easygraph.gpu._arena.gpu_runtime import cp

SOURCE = r'''
__device__ double minimum(double* p, double value) {
  unsigned long long* address=(unsigned long long*)p;
  unsigned long long old=atomicCAS(address,0ULL,0ULL), assumed;
  while(value<__longlong_as_double(old)) {
    assumed=old;
    old=atomicCAS(address,assumed,__double_as_longlong(value));
    if(old==assumed) break;
  }
  return __longlong_as_double(old);
}
extern "C" __global__ void initialize(const int n, const int* sources,
 const int source_index, double* distances, int* frontier, int* tags, int* ctl) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
    distances[i]=i==sources[source_index] ? 0. : __longlong_as_double(0x7ff0000000000000ULL);
    tags[i]=-1;
  }
  if(blockIdx.x==0 && threadIdx.x==0) {
    frontier[0]=sources[source_index];
    ctl[0]=1; ctl[1]=0; ctl[2]=0; ctl[3]=0;
  }
}
template<int WIDTH> __device__ void traverse(const int* row,const int* col,
 const double* weight, const int* frontier, int* next, double* distances,
 int* tags, int* ctl) {
  if(ctl[2]) return;
  int lane=threadIdx.x&(WIDTH-1);
  int stride=gridDim.x*blockDim.x/WIDTH;
  for(int i=(blockIdx.x*blockDim.x+threadIdx.x)/WIDTH;i<ctl[0];i+=stride) {
    int u=frontier[i];
    double du=__longlong_as_double(atomicCAS((unsigned long long*)(distances+u),0ULL,0ULL));
    for(int e=row[u]+lane;e<row[u+1];e+=WIDTH) {
      int v=col[e]; double candidate=du+weight[e];
      double prior=minimum(distances+v,candidate);
      if(candidate<prior && atomicExch(tags+v,ctl[3])!=ctl[3])
        next[atomicAdd(ctl+1,1)]=v;
    }
  }
}
#define STEP(W) extern "C" __global__ void step##W(const int* row,const int* col, \
 const double* weight,const int* frontier,int* next,double* distances,int* tags,int* ctl) \
 { traverse<W>(row,col,weight,frontier,next,distances,tags,ctl); }
STEP(1) STEP(2) STEP(4) STEP(8) STEP(16) STEP(32)
extern "C" __global__ void advance(int* ctl) {
  if(!ctl[2]) {
    ctl[3]++;
    ctl[0]=ctl[1]; ctl[1]=0;
    if(ctl[0]==0) ctl[2]=1;
  }
}
'''


class NativeFrontierSSSP:
    """NativeSSSP-compatible decorator; supports complete or selected downloads."""
    def __init__(self, base, checkpoint_iterations=16, vector_width=4):
        if checkpoint_iterations<1 or vector_width not in (1,2,4,8,16,32):
            raise ValueError('Invalid checkpoint interval or vector width')
        self.base=base
        self.checkpoint_iterations=int(checkpoint_iterations)
        self.vector_width=int(vector_width)
        self._kernels=None
        self.last={}

    def __getattr__(self,name):return getattr(self.base,name)

    @property
    def output(self):return self.base.output
    @output.setter
    def output(self,value):self.base.output=value
    @property
    def workspace(self):return self.base.workspace
    @workspace.setter
    def workspace(self,value):self.base.workspace=value
    @property
    def nmax(self):return self.base.nmax
    @nmax.setter
    def nmax(self,value):self.base.nmax=value
    @property
    def host_partial(self):return self.base.host_partial
    @host_partial.setter
    def host_partial(self,value):self.base.host_partial=value

    def _load(self):
        if self._kernels is not None:return 0.
        started=time.perf_counter()
        module=cp.RawModule(code=SOURCE,options=('--std=c++17',),backend='nvrtc')
        self._kernels={name:module.get_function(name) for name in
                       ('initialize','step'+str(self.vector_width),'advance')}
        props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        self._grid_cap=2*int(props['multiProcessorCount'])
        return time.perf_counter()-started

    def run(self,row,col,weight,sources,warp_size=0):
        n=row.size-1;count=sources.size
        if n>self.nmax or count>self.source_capacity:
            raise ValueError('Request exceeds reserved native capacity')
        if not n or not count:
            self.last=dict(expansion_rounds=0,control_d2h_bytes=0,native_execute_seconds=0.)
            return self.output[:n*count].reshape(count,n)
        if row.dtype!=cp.int32 or col.dtype!=cp.int32 or sources.dtype!=cp.int32 or weight.dtype!=cp.float64:
            raise TypeError('Requires int32 CSR/sources and float64 weights')
        load=self._load()
        # Three n-int arrays plus four counters fit the original two n-int,
        # n-double, reduction-partial and 256-byte native workspace reservation.
        if self.workspace.nbytes<12*n+16:raise AssertionError('Insufficient native workspace')
        ints=self.workspace[:12*n+16].view(cp.int32)
        f0=ints[:n];f1=ints[n:2*n];tags=ints[2*n:3*n];ctl=ints[3*n:3*n+4]
        grid=(min(self._grid_cap,max(1,(n*self.vector_width+127)//128)),)
        init_grid=(min(self._grid_cap,max(1,(n+127)//128)),)
        host=np.empty(4,np.int32);rounds=0;checks=0;queued=0
        control_seconds=0.;started=time.perf_counter()
        for index in range(count):
            distances=self.output[index*n:(index+1)*n]
            self._kernels['initialize'](init_grid,(128,),
                (np.int32(n),sources,np.int32(index),distances,f0,tags,ctl))
            current,next_=f0,f1
            for first in range(0,n+1,self.checkpoint_iterations):
                batch=min(self.checkpoint_iterations,n+1-first)
                for _ in range(batch):
                    self._kernels['step'+str(self.vector_width)](grid,(128,),
                        (row,col,weight,current,next_,distances,tags,ctl))
                    self._kernels['advance']((1,),(1,),(ctl,))
                    current,next_=next_,current
                queued+=batch;before=time.perf_counter();ctl.get(out=host)
                control_seconds+=time.perf_counter()-before;checks+=1
                if host[2]:break
            if not host[2]:raise RuntimeError('SSSP exceeded n+1 rounds; requires nonnegative finite weights')
            rounds+=int(host[3])
        self.last=dict(expansion_rounds=rounds,control_d2h_bytes=16*checks,
            native_execute_seconds=time.perf_counter()-started,
            kernel_load_seconds=load,control_checkpoint_seconds=control_seconds,
            control_checkpoints=checks,queued_rounds=queued,
            algorithm='exact_async_frontier',checkpoint_iterations=self.checkpoint_iterations,
            vector_width=self.vector_width,additional_device_bytes=0)
        return self.output[:n*count].reshape(count,n)
