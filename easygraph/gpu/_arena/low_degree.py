"""Elide dangling/scale passes when a versioned low-degree predicate proves safe."""
import time
import weakref
from collections import OrderedDict
import numpy as np
from easygraph.gpu._arena.gpu_runtime import cp
from easygraph.gpu._arena.lean import LeanExecutor,LEAN_SOURCE

SOURCE=LEAN_SOURCE+r'''
template<int WIDTH> __device__ void fused_rank(const int n,const int *row,const int *col,
 const double *weight,const double *sums,const double *x,double *y,double *stat,
 const int *ctl,const double alpha,const int weighted) {
  if(ctl[0]) return;
  const int i=(blockIdx.x*blockDim.x+threadIdx.x)/WIDTH;
  const int lane=threadIdx.x&(WIDTH-1);
  double sum=0.,diff=0.;
  if(i<n) for(int e=row[i]+lane;e<row[i+1];e+=WIDTH)
    sum+=(weighted ? weight[e] : 1.)*(x[col[e]]/sums[col[e]]);
  for(int d=WIDTH/2;d;d>>=1) sum+=__shfl_down_sync(0xffffffff,sum,d,WIDTH);
  if(i<n && lane==0) {
    double value=(1.-alpha)/n+alpha*sum;
    y[i]=value;diff=fabs(value-x[i]);
  }
  diff=block_sum(diff);
  if(threadIdx.x==0) atomicAdd(stat+1,diff);
}
#define FUSED(W) extern "C" __global__ void fused##W(const int n,const int *row, \
 const int *col,const double *weight,const double *sums,const double *x,double *y, \
 double *stat,const int *ctl,const double alpha,const int weighted) \
 { fused_rank<W>(n,row,col,weight,sums,x,y,stat,ctl,alpha,weighted); }
FUSED(1) FUSED(2) FUSED(4) FUSED(8) FUSED(16) FUSED(32)
'''

class LowDegreeExecutor(LeanExecutor):
    def __init__(self,base,checkpoint_iterations=8,vector_width=4,max_in_degree=16):
        super().__init__(base,checkpoint_iterations,vector_width)
        self.max_in_degree=max_in_degree
        self.predicates=OrderedDict()

    def _load(self):
        if self._kernels is not None:return 0.
        t=time.perf_counter();module=cp.RawModule(code=SOURCE,options=('--std=c++17',),backend='nvrtc')
        self._kernels={name:module.get_function(name) for name in
            ('init_lean','scale','finish_lean','rank'+str(self.vector_width),'fused'+str(self.vector_width))}
        return time.perf_counter()-t

    def _eligible(self,graph,attr,public):
        key=graph.key('source_sum',attr)
        threshold=1e-15 if public and attr is not None else 0.
        # The predicate depends on the SAME versions as the requested source
        # sums. A weight edit/topology edit therefore cannot reuse stale proof.
        version=graph.version(key)
        cachekey=(id(graph),graph.name,graph.n,key,version,threshold,self.max_in_degree)
        if cachekey not in self.predicates or self.predicates[cachekey][0]() is not graph:
            sums=graph.get(key)
            self.predicates[cachekey]=(weakref.ref(graph), bool(graph.n and np.all(sums>threshold)
                and np.diff(graph.row).max(initial=0)<=self.max_in_degree))
            while len(self.predicates)>128:self.predicates.popitem(last=False)
        self.predicates.move_to_end(cachekey)
        return self.predicates[cachekey][1]

    def _rank(self,factor,graph,arrays,attr,alpha,tol,limit,convergence,public,selected):
        t=time.perf_counter();eligible=self._eligible(graph,attr,public);eligibility=time.perf_counter()-t
        if not eligible:
            result,metrics=super()._rank(factor,graph,arrays,attr,alpha,tol,limit,convergence,public,selected)
            metrics.update(low_degree_fused=False,eligibility_seconds=eligibility)
            return result,metrics
        engine=factor.base;n=graph.n
        if limit<0:raise ValueError('Iteration limit must be nonnegative')
        load=self._load();k=self._kernels;x,y=engine.x,engine.y
        stat=engine.stat;ctl=engine.patch_i[:2];last=engine.patch_v[:1]
        row,col=arrays[graph.key('row')],arrays[graph.key('col')];sums=arrays[graph.key('source_sum',attr)]
        weight=arrays[graph.key('weight',attr)] if attr is not None else sums
        grid=((n+127)//128,);rankgrid=((n*self.vector_width+127)//128,)
        start,end=engine.events;start.record()
        k['init_lean'](grid,(128,),(np.int32(n),x,stat,ctl,last))
        control=np.zeros(2,np.int32);residual=np.zeros(1,np.float64);checks=0;queued=0
        stride=self.checkpoint_iterations if convergence else max(1,limit)
        for lo in range(0,limit,stride):
            for _ in range(lo,min(limit,lo+stride)):
                k['fused'+str(self.vector_width)](rankgrid,(128,),
                    (np.int32(n),row,col,weight,sums,x,y,stat,ctl,np.float64(alpha),np.int32(attr is not None)))
                k['finish_lean']((1,),(1,),(stat,ctl,last,np.float64(tol),np.int32(convergence)))
                x,y=y,x;queued+=1
            ctl.get(out=control);checks+=1
            if control[0]:break
        last.get(out=residual);end.record();end.synchronize()
        iterations=int(control[1]);converged=bool(control[0])
        if convergence and not converged and not public:raise RuntimeError('Fused PageRank failed convergence')
        result=engine.y if iterations%2 else engine.x
        output,metrics=engine.download(result[:n],selected)
        return output,dict(iterations=iterations,requested_iterations=None if convergence else limit,
            stopping='public_tolerance_or_iteration_budget' if public and convergence else 'global_l1_or_error' if convergence else 'exact_iterations',
            alpha=float(alpha),device_loop_seconds=cp.cuda.get_elapsed_time(start,end)/1000,
            d2h_bytes=output.nbytes+8*(checks+1),factorized_kernel_load_seconds=load,
            representation='factorized',final_global_l1=float(residual[0]) if limit else None,
            converged=converged,convergence_checkpoints=checks,queued_iterations=queued,
            checkpoint_interval=self.checkpoint_iterations,csr_vector_width=self.vector_width,
            low_degree_fused=True,eligibility_seconds=eligibility,max_in_degree_threshold=self.max_in_degree,
            fused_stats_reset=True,result_cache_used=False,additional_device_array_bytes=0,**metrics)
