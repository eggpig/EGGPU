"""Three-kernel PR iterations: fuse the next reset into the convergence step."""
import time
import numpy as np
from easygraph.gpu._arena.gpu_runtime import cp
from easygraph.gpu._arena.executor import KernelExecutor,SOURCE

LEAN_SOURCE=SOURCE+r'''
extern "C" __global__ void init_lean(const int n,double *x,double *stat,int *ctl,double *last) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) x[i]=1./n;
  if(i==0) { ctl[0]=0;ctl[1]=0;stat[0]=0.;stat[1]=0.;last[0]=0.; }
}
extern "C" __global__ void finish_lean(double *stat,int *ctl,double *last,const double tol,const int convergence) {
  if(!ctl[0]) {
    ctl[1]++;last[0]=stat[1];
    if(convergence && stat[1]<tol) ctl[0]=1;
    if(!ctl[0]) { stat[0]=0.;stat[1]=0.; }
  }
}
'''

class LeanExecutor(KernelExecutor):
    def _load(self):
        if self._kernels is not None:return 0.
        start=time.perf_counter()
        module=cp.RawModule(code=LEAN_SOURCE,options=('--std=c++17',),backend='nvrtc')
        self._kernels={n:module.get_function(n) for n in ('init_lean','scale','finish_lean','rank'+str(self.vector_width))}
        return time.perf_counter()-start

    def _rank(self,factor,graph,arrays,attr,alpha,tol,limit,convergence,public,selected):
        engine=factor.base;n=graph.n
        if n==0:
            output,metrics=engine.download(engine.x[:0],selected)
            return output,dict(iterations=0,device_loop_seconds=0.,d2h_bytes=0.,representation='factorized',**metrics)
        if limit<0:raise ValueError('Iteration limit must be nonnegative')
        load=self._load();k=self._kernels;x,y=engine.x,engine.y
        scaled=factor.scaled_workspace();stat=engine.stat;ctl=engine.patch_i[:2];last=engine.patch_v[:1]
        row,col=arrays[graph.key('row')],arrays[graph.key('col')];sums=arrays[graph.key('source_sum',attr)]
        weight=arrays[graph.key('weight',attr)] if attr is not None else scaled
        grid=((n+127)//128,);rankgrid=((n*self.vector_width+127)//128,)
        start,end=engine.events;t=time.perf_counter();start.record();eventstart=time.perf_counter()-t
        k['init_lean'](grid,(128,),(np.int32(n),x,stat,ctl,last))
        control=np.zeros(2,np.int32);residual=np.zeros(1,np.float64);checks=0;queued=0
        stride=self.checkpoint_iterations if convergence else max(1,limit)
        for lo in range(0,limit,stride):
            for _ in range(lo,min(limit,lo+stride)):
                k['scale'](grid,(128,),(np.int32(n),x,sums,scaled,stat,ctl,np.float64(1e-15 if public and attr is not None else 0.)))
                k['rank'+str(self.vector_width)](rankgrid,(128,),
                    (np.int32(n),row,col,weight,scaled,x,y,stat,ctl,np.float64(alpha),np.int32(attr is not None)))
                k['finish_lean']((1,),(1,),(stat,ctl,last,np.float64(tol),np.int32(convergence)))
                x,y=y,x;queued+=1
            ctl.get(out=control);checks+=1
            if control[0]:break
        last.get(out=residual)
        t=time.perf_counter();end.record();end.synchronize();eventfinish=time.perf_counter()-t
        iterations=int(control[1]);converged=bool(control[0])
        if convergence and not converged and not public:raise RuntimeError('Lean PageRank failed convergence')
        result=engine.y if iterations%2 else engine.x
        output,metrics=engine.download(result[:n],selected)
        return output,dict(iterations=iterations,requested_iterations=None if convergence else limit,
            stopping='public_tolerance_or_iteration_budget' if public and convergence else 'global_l1_or_error' if convergence else 'exact_iterations',
            alpha=float(alpha),device_loop_seconds=cp.cuda.get_elapsed_time(start,end)/1000,
            host_event_start_seconds=eventstart,host_event_finish_seconds=eventfinish,
            d2h_bytes=output.nbytes+8*(checks+1),factorized_kernel_load_seconds=load,
            representation='factorized',final_global_l1=float(residual[0]) if limit else None,
            converged=converged,convergence_checkpoints=checks,queued_iterations=queued,
            checkpoint_interval=self.checkpoint_iterations,csr_vector_width=self.vector_width,
            fused_stats_reset=True,result_cache_used=False,additional_device_array_bytes=0,**metrics)
