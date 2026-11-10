"""Queue-aggregated alternative, preserving the v1 exploratory source."""
import time
from easygraph.gpu._arena.gpu_runtime import cp
from easygraph.gpu._arena.sssp_frontier import NativeFrontierSSSP,SOURCE

WARP_SOURCE=SOURCE.replace(
    'unsigned long long old=atomicCAS(address,0ULL,0ULL), assumed;',
    'unsigned long long old=*address, assumed;').replace(
    'double du=__longlong_as_double(atomicCAS((unsigned long long*)(distances+u),0ULL,0ULL));',
    '''double du=0.;
    if(lane==0) du=__longlong_as_double(atomicCAS((unsigned long long*)(distances+u),0ULL,0ULL));
    du=__shfl_sync(__activemask(),du,0,WIDTH);''').replace(
    '''if(candidate<prior && atomicExch(tags+v,ctl[3])!=ctl[3])
        next[atomicAdd(ctl+1,1)]=v;''',
    '''if(candidate<prior && atomicExch(tags+v,ctl[3])!=ctl[3]) {
        unsigned mask=__activemask();
        int leader=__ffs(mask)-1,warp_lane=threadIdx.x&31;
        int first=0;
        if(warp_lane==leader) first=atomicAdd(ctl+1,__popc(mask));
        first=__shfl_sync(mask,first,leader);
        int offset=__popc(mask&((1U<<warp_lane)-1));
        next[first+offset]=v;
      }''')

class WarpFrontierSSSP(NativeFrontierSSSP):
    def __init__(self,base,checkpoint_iterations=16,vector_width=4,grid_cap=216):
        super().__init__(base,checkpoint_iterations,vector_width)
        self.grid_cap=int(grid_cap)

    def _load(self):
        if self._kernels is not None:return 0.
        started=time.perf_counter()
        module=cp.RawModule(code=WARP_SOURCE,options=('--std=c++17',),backend='nvrtc')
        self._kernels={name:module.get_function(name) for name in
                       ('initialize','step'+str(self.vector_width),'advance')}
        self._grid_cap=self.grid_cap
        return time.perf_counter()-started
