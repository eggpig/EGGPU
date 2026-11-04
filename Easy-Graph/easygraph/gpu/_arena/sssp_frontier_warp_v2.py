"""Explicit logical-warp synchronization for shared source-distance reads.

v1 used an instantaneous active mask around a warp broadcast following an
atomic read; independent thread scheduling can exclude the source lane. Each
logical CSR group now supplies its complete mask to the synchronized shuffle.
"""
import time
from easygraph.gpu._arena.gpu_runtime import cp
from easygraph.gpu._arena.sssp_frontier_warp import WarpFrontierSSSP,WARP_SOURCE

CORRECTED_SOURCE=WARP_SOURCE.replace('__shfl_sync(__activemask(),du,0,WIDTH)',
    '__shfl_sync((0xffffffffU >> (32-WIDTH)) << ((threadIdx.x&31)&~(WIDTH-1)),du,0,WIDTH)')

class WarpFrontierSSSPV2(WarpFrontierSSSP):
    def _load(self):
        if self._kernels is not None:return 0.
        started=time.perf_counter()
        module=cp.RawModule(code=CORRECTED_SOURCE,options=('--std=c++17',),backend='nvrtc')
        self._kernels={name:module.get_function(name) for name in
                       ('initialize','step'+str(self.vector_width),'advance')}
        self._grid_cap=self.grid_cap
        return time.perf_counter()-started
