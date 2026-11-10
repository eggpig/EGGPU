"""Existing pull traversal with node-factorized PageRank preparation."""
import time
import numpy as np
from easygraph.gpu._arena.gpu_runtime import cp,CUDA_SOURCE

FACTOR_SOURCE=CUDA_SOURCE[:CUDA_SOURCE.index('extern "C" __global__ void init')]+r'''
extern "C" __global__ void scale_and_stats(const int n, const double *x,
 const double *sum, double *scaled, double *stat) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 double dangling=0.0;
 if(i<n) { scaled[i]=sum[i] ? x[i]/sum[i] : 0.0; dangling=sum[i] ? 0.0 : x[i]; }
 dangling=block_sum(dangling);
 if(threadIdx.x==0) atomicAdd(stat,dangling);
}
extern "C" __global__ void pagerank_factorized(const int n, const int *row,
 const int *col, const double *weight, const double *scaled,
 const double *x, double *y, double *stat, const double alpha, const int weighted) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;
 double diff=0.0;
 if(i<n) {
  double z=0.0;
  for(int e=row[i];e<row[i+1];++e) z += weighted ? weight[e]*scaled[col[e]] : scaled[col[e]];
  y[i]=(1.0-alpha)/n+alpha*(z+stat[0]/n);
  diff=fabs(y[i]-x[i]);
 }
 diff=block_sum(diff);
 if(threadIdx.x==0) atomicAdd(stat+1,diff);
}
'''

class FactorizedExecutor:
    """Wrap an initialized executor; all original BFS/SSSP methods delegate."""
    def __init__(self,base,kernel_backend):
        self.base=base;self.kernel_backend=kernel_backend
        self.factor_kernels=None

    def load_factorized_kernels(self):
        if self.factor_kernels is not None:return 0.0
        started=time.perf_counter()
        module=cp.RawModule(code=FACTOR_SOURCE,options=('--std=c++17',),backend='nvcc')
        self.factor_kernels={name:module.get_function(name) for name in ('scale_and_stats','pagerank_factorized')}
        return time.perf_counter()-started

    def __getattr__(self,name):return getattr(self.base,name)
    @property
    def workspace_bytes(self):return self.base.workspace_bytes
    @workspace_bytes.setter
    def workspace_bytes(self,value):self.base.workspace_bytes=value
    def bind_workspace(self,nmax):self.base.bind_workspace(nmax)

    def scaled_workspace(self):
        key='@scaled_rank' if self.kernel_backend=='reference' else '@native_bfs_output'
        array=self.base.arena.slots[key].array
        if array.dtype!=cp.float64 or array.size<self.base.nmax:raise AssertionError('Invalid shared scaled-rank workspace')
        return array

    def run(self,graph,arrays,function,attr=None,source=0,alpha=.85,tol=1e-8,max_iter=300,selected=None):
        if function!='pr' or graph.key('source_sum',attr) not in arrays:
            return self.base.run(graph,arrays,function,attr,source,alpha,tol,max_iter,selected)
        n=graph.n
        if not n:
            value,metrics=self.base.download(self.base.x[:0],selected)
            return value,dict(iterations=0,device_loop_seconds=0.,d2h_bytes=0,representation='factorized',**metrics)
        load_seconds=self.load_factorized_kernels()
        grid=((n+127)//128,);x,y=self.base.x,self.base.y;scaled=self.scaled_workspace()
        row,col=arrays[graph.key('row')],arrays[graph.key('col')]
        sums=arrays[graph.key('source_sum',attr)]
        weight=arrays[graph.key('weight',attr)] if attr is not None else scaled
        start,end=self.base.events;t=time.perf_counter();start.record();event_start=time.perf_counter()-t
        t=time.perf_counter();self.base.kernels['init'](grid,(128,),(np.int32(n),x,np.int32(0),np.int32(source)))
        init_seconds=time.perf_counter()-t
        flag=np.empty(2,np.float64);converged=False
        for iteration in range(max_iter):
            cp.cuda.runtime.memsetAsync(self.base.stat.data.ptr,0,16,cp.cuda.get_current_stream().ptr)
            self.factor_kernels['scale_and_stats'](grid,(128,),(np.int32(n),x,sums,scaled,self.base.stat))
            self.factor_kernels['pagerank_factorized'](grid,(128,),(np.int32(n),row,col,weight,scaled,x,y,self.base.stat,
                                                                    np.float64(alpha),np.int32(attr is not None)))
            self.base.stat.get(out=flag);x,y=y,x
            if flag[1]<tol:converged=True;break
        t=time.perf_counter();end.record();end.synchronize();event_finish=time.perf_counter()-t
        if not converged:raise RuntimeError('factorized PageRank failed convergence')
        value,metrics=self.base.download(x[:n],selected)
        return value,dict(iterations=iteration+1,device_loop_seconds=cp.cuda.get_elapsed_time(start,end)/1000,
            host_event_start_seconds=event_start,host_init_launch_seconds=init_seconds,
            host_event_finish_seconds=event_finish,d2h_bytes=value.nbytes+16*(iteration+1),
            factorized_kernel_load_seconds=load_seconds,
            representation='factorized',final_global_l1=float(flag[1]),**metrics)
