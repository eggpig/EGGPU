"""R11 PageRank execution candidate; no algorithm-result cache.

Wrap either a public or typed executor. Graph representations, graph versions,
the output contract, and the arena remain owned by the wrapped implementation.
The candidate removes unnecessary fixed-iteration device-to-host handshakes and
checks convergence in batches. A device control word freezes the FIRST iterate
meeting the original tolerance, so checkpoint frequency changes no stopping
semantics. All arrays, including control words, reuse reserved arena scratch.
"""
from __future__ import annotations

import time
from collections import OrderedDict
import numpy as np
from easygraph.gpu._arena.gpu_runtime import cp

SOURCE = r'''
__device__ __forceinline__ double warp_sum(double x) {
  for(int d=16;d;d>>=1) x+=__shfl_down_sync(0xffffffff,x,d);
  return x;
}
__device__ __forceinline__ double block_sum(double x) {
  __shared__ double partial[4];
  x=warp_sum(x);
  if((threadIdx.x&31)==0) partial[threadIdx.x>>5]=x;
  __syncthreads();
  x=threadIdx.x<4 ? partial[threadIdx.x] : 0.;
  if(threadIdx.x<32) x=warp_sum(x);
  return x;
}
extern "C" __global__ void initialize(const int n, double *x, int *ctl) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) x[i]=1.0/n;
  if(i==0) { ctl[0]=0; ctl[1]=0; }
}
extern "C" __global__ void reset_stats(double *stat, const int *ctl) {
  if(!ctl[0]) { stat[0]=0.; stat[1]=0.; }
}
extern "C" __global__ void scale(const int n, const double *x,
 const double *sums, double *scaled, double *stat, const int *ctl,
 const double dangling_threshold) {
  if(ctl[0]) return;
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  double dangling=0.;
  if(i<n) {
    bool active=sums[i]>dangling_threshold;
    scaled[i]=active ? x[i]/sums[i] : 0.;
    dangling=active ? 0. : x[i];
  }
  dangling=block_sum(dangling);
  if(threadIdx.x==0) atomicAdd(stat,dangling);
}
template<int WIDTH> __device__ void rank_step(const int n, const int *row,
 const int *col, const double *weight, const double *scaled,
 const double *x, double *y, double *stat, const int *ctl,
 const double alpha, const int weighted) {
  if(ctl[0]) return;
  const int i=(blockIdx.x*blockDim.x+threadIdx.x)/WIDTH;
  const int lane=threadIdx.x&(WIDTH-1);
  double sum=0.,diff=0.;
  if(i<n) {
    for(int e=row[i]+lane;e<row[i+1];e+=WIDTH)
      sum+=(weighted ? weight[e] : 1.)*scaled[col[e]];
  }
  for(int delta=WIDTH/2;delta;delta>>=1)
    sum+=__shfl_down_sync(0xffffffff,sum,delta,WIDTH);
  if(i<n && lane==0) {
    double value=(1.-alpha)/n+alpha*(sum+stat[0]/n);
    y[i]=value;
    diff=fabs(value-x[i]);
  }
  diff=block_sum(diff);
  if(threadIdx.x==0) atomicAdd(stat+1,diff);
}
#define RANK(W) extern "C" __global__ void rank##W(const int n, const int *row, \
 const int *col, const double *weight, const double *scaled, const double *x, \
 double *y, double *stat, const int *ctl, const double alpha, const int weighted) \
 { rank_step<W>(n,row,col,weight,scaled,x,y,stat,ctl,alpha,weighted); }
RANK(1) RANK(2) RANK(4) RANK(8) RANK(16) RANK(32)
extern "C" __global__ void finish_step(const double *stat, int *ctl,
 const double tol, const int convergence) {
  if(!ctl[0]) { ctl[1]++; if(convergence && stat[1]<tol) ctl[0]=1; }
}
'''


def _factor_executor(base):
    """Find the factorized implementation without relying on wrapper count."""
    seen = set()
    while id(base) not in seen:
        seen.add(id(base))
        if 'kernel_backend' in base.__dict__ and hasattr(base, 'scaled_workspace'):
            return base
        base = getattr(base, 'base', None)
        if base is None:
            break
    raise TypeError('Expected an executor wrapping FactorizedExecutor')


class KernelExecutor:
    """Compatible run() decorator for public or typed graph sessions.

    `checkpoint_iterations` controls host notifications only. Every completed
    mathematical iteration still evaluates the same full global L1 criterion.
    Vector width is explicit; no data-dependent benchmark selection is hidden.
    """
    def __init__(self, base, checkpoint_iterations=8, vector_width=4,
                 use_cuda_graph=False, max_graphs=16):
        if checkpoint_iterations < 1 or vector_width not in (1,2,4,8,16,32):
            raise ValueError('Invalid checkpoint interval or CSR vector width')
        self.base = base
        self.checkpoint_iterations = int(checkpoint_iterations)
        self.vector_width = int(vector_width)
        self._kernels = None
        self.use_cuda_graph=bool(use_cuda_graph)
        self.max_graphs=int(max_graphs)
        if self.max_graphs<1:
            raise ValueError('max_graphs must be positive')
        self._graphs=OrderedDict()
        self.graph_captures=0
        self.graph_replays=0
        self.graph_evictions=0
        self._capture_stream=None
        # Public dispatch writes .pr on the topmost wrapper; forward through
        # decorators so mixed-function execution preserves one request contract.
        self.pr = getattr(base, 'pr', None)

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def pr(self):
        return getattr(self.base,'pr',None)

    @pr.setter
    def pr(self,value):
        self.base.pr=value

    @property
    def workspace_bytes(self):
        return self.base.workspace_bytes

    @workspace_bytes.setter
    def workspace_bytes(self, value):
        self.base.workspace_bytes = value

    def _load(self):
        if self._kernels is not None:
            return 0.
        start=time.perf_counter()
        module=cp.RawModule(code=SOURCE,options=('--std=c++17',),backend='nvrtc')
        self._kernels={name:module.get_function(name) for name in
            ('initialize','reset_stats','scale','finish_step','rank'+str(self.vector_width))}
        return time.perf_counter()-start

    def clear_graphs(self):
        self._graphs.clear()

    def _emit(self,kernels,count,n,x,y,sums,scaled,stat,ctl,threshold,row,col,weight,alpha,attr,tol,convergence):
        grid=((n+127)//128,)
        rank_grid=((n*self.vector_width+127)//128,)
        for _ in range(count):
            kernels['reset_stats']((1,),(1,),(stat,ctl))
            kernels['scale'](grid,(128,),(np.int32(n),x,sums,scaled,stat,ctl,np.float64(threshold)))
            kernels['rank'+str(self.vector_width)](rank_grid,(128,),
                (np.int32(n),row,col,weight,scaled,x,y,stat,ctl,np.float64(alpha),np.int32(attr is not None)))
            kernels['finish_step']((1,),(1,),(stat,ctl,np.float64(tol),np.int32(convergence)))
            x,y=y,x

    def _enqueue(self,count,n,x,y,sums,scaled,stat,ctl,threshold,row,col,weight,alpha,attr,tol,convergence):
        args=(self._kernels,count,n,x,y,sums,scaled,stat,ctl,threshold,row,col,weight,alpha,attr,tol,convergence)
        if not self.use_cuda_graph:
            self._emit(*args)
            return
        key=(count,n,self.vector_width,threshold,alpha,attr is not None,tol,convergence,
             *(int(a.data.ptr) for a in (x,y,sums,scaled,stat,ctl,row,col,weight)))
        graph=self._graphs.get(key)
        if graph is None:
            if self._capture_stream is None:
                self._capture_stream=cp.cuda.Stream(non_blocking=True)
            stream=self._capture_stream
            with stream:
                stream.begin_capture()
                self._emit(*args)
                graph=stream.end_capture()
            self.graph_captures+=1
            self._graphs[key]=graph
            while len(self._graphs)>self.max_graphs:
                self._graphs.popitem(last=False)
                self.graph_evictions+=1
        self._graphs.move_to_end(key)
        graph.launch(cp.cuda.get_current_stream())
        self.graph_replays+=1

    def run(self, graph, arrays, function, attr=None, source=0, alpha=.85,
            tol=1e-8, max_iter=300, selected=None):
        if function!='pr' or graph.key('source_sum',attr) not in arrays:
            return self.base.run(graph,arrays,function,attr,source,alpha=alpha,
                                 tol=tol,max_iter=max_iter,selected=selected)
        factor=_factor_executor(self.base)
        fixed=getattr(self.base,'requested_iterations',None)
        public=self.pr is not None
        if public:
            alpha,max_iter,eps=self.pr
            tol=eps*graph.n
            fixed=max_iter if eps==0 else None
        elif fixed is not None:
            override=getattr(self.base,'requested_alpha',None)
            if override is not None:
                alpha=override
        return self._rank(factor,graph,arrays,attr,alpha,tol,
                          fixed if fixed is not None else max_iter,
                          convergence=fixed is None, public=public, selected=selected)

    def _rank(self,factor,graph,arrays,attr,alpha,tol,limit,convergence,public,selected):
        engine=factor.base
        n=graph.n
        if n==0:
            output,metrics=engine.download(engine.x[:0],selected)
            return output,dict(iterations=0,device_loop_seconds=0.,d2h_bytes=0,
                               representation='factorized',**metrics)
        if limit < 0:
            raise ValueError('Iteration limit must be nonnegative')
        load_seconds=self._load()
        kernels=self._kernels
        x,y=engine.x,engine.y
        scaled=factor.scaled_workspace()
        stat=engine.stat
        # Patch scratch is not in use while a request owns execution workspace.
        ctl=engine.patch_i[:2]
        row,col=arrays[graph.key('row')],arrays[graph.key('col')]
        sums=arrays[graph.key('source_sum',attr)]
        weight=arrays[graph.key('weight',attr)] if attr is not None else scaled
        grid=((n+127)//128,)
        start,end=engine.events
        t=time.perf_counter();start.record();event_start=time.perf_counter()-t
        t=time.perf_counter();kernels['initialize'](grid,(128,),(np.int32(n),x,ctl))
        init_seconds=time.perf_counter()-t
        control=np.zeros(2,np.int32);flags=np.zeros(2,np.float64)
        checkpoints=0;launch_steps=0
        threshold=1e-15 if public and attr is not None else 0.
        for lo in range(0,limit,self.checkpoint_iterations if convergence else max(1,limit)):
            hi=min(limit,lo+(self.checkpoint_iterations if convergence else limit))
            self._enqueue(hi-lo,n,x,y,sums,scaled,stat,ctl,threshold,row,col,weight,alpha,attr,tol,convergence)
            if (hi-lo)%2:
                x,y=y,x
            launch_steps+=hi-lo
            ctl.get(out=control);checkpoints+=1
            if control[0]:
                break
        if limit:
            stat.get(out=flags)
        t=time.perf_counter();end.record();end.synchronize();event_finish=time.perf_counter()-t
        iterations=int(control[1])
        converged=bool(control[0])
        if convergence and not converged and not public:
            raise RuntimeError('R11 factorized PageRank failed convergence')
        # Guarded batches leave both arrays untouched after the first converged
        # iteration. Recover its buffer by ACTUAL, not enqueued, parity.
        result=engine.y if iterations%2 else engine.x
        output,metrics=engine.download(result[:n],selected)
        return output,dict(iterations=iterations,requested_iterations=None if convergence else limit,
            stopping='public_tolerance_or_iteration_budget' if public and convergence else
                     'global_l1_or_error' if convergence else 'exact_iterations',
            alpha=float(alpha),device_loop_seconds=cp.cuda.get_elapsed_time(start,end)/1000,
            host_event_start_seconds=event_start,host_init_launch_seconds=init_seconds,
            host_event_finish_seconds=event_finish,d2h_bytes=output.nbytes+8*checkpoints+(16 if limit else 0),
            factorized_kernel_load_seconds=load_seconds,representation='factorized',
            final_global_l1=float(flags[1]) if limit else None,converged=converged,
            convergence_checkpoints=checkpoints,queued_iterations=launch_steps,
            checkpoint_interval=self.checkpoint_iterations,csr_vector_width=self.vector_width,
            cuda_graph_enabled=self.use_cuda_graph,cuda_graph_entries=len(self._graphs),
            cuda_graph_captures=self.graph_captures,cuda_graph_replays=self.graph_replays,
            cuda_graph_evictions=self.graph_evictions,
            cuda_graph_memory_scope='CUDA driver graph metadata outside explicit array arena; process memory must be reported separately',
            result_cache_used=False,additional_device_array_bytes=0,**metrics)
