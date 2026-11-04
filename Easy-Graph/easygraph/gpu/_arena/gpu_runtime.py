"""Single-device graph representations and bounded execution storage.

Every algorithm and cache policy uses the same kernels. A single CUDA allocation
backs graph components, working vectors, and update staging. Results are copied
to independent host arrays before a request releases its workspace.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import time

import cupy as cp
import numpy as np
from scipy.sparse import coo_matrix


CUDA_SOURCE = r'''
__device__ __forceinline__ double block_sum(double value) {
  __shared__ double values[128];
  values[threadIdx.x]=value;
  __syncthreads();
  for(int delta=64;delta;delta>>=1) {
    if(threadIdx.x<delta) values[threadIdx.x]+=values[threadIdx.x+delta];
    __syncthreads();
  }
  return values[0];
}
extern "C" __global__ void init(const int n, double *x, const int mode,
                                const int source) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) x[i] = mode == 0 ? 1.0/n : (i == source ? 0.0 : __longlong_as_double(0x7ff0000000000000LL));
}
extern "C" __global__ void stats(const int n, const double *x,
 const unsigned char *dangling, double *out) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  double val=(i<n && dangling[i]) ? x[i] : 0.0;
  val=block_sum(val);
  if(threadIdx.x==0) atomicAdd(out,val);
}
extern "C" __global__ void pagerank(const int n, const int *row,
 const int *col, const double *w, const double *x, double *y,
 double *stat, const double alpha) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  double diff=0.0;
  if(i<n) {
    double z = 0.0;
    for (int e=row[i]; e<row[i+1]; ++e) z += w[e]*x[col[e]];
    y[i] = (1.0-alpha)/n + alpha*(z+stat[0]/n);
    diff=fabs(y[i]-x[i]);
  }
  diff=block_sum(diff);
  if(threadIdx.x==0) atomicAdd(stat+1,diff);
}
extern "C" __global__ void shortest(const int n, const int *row,
 const int *col, const double *w, const double *x, double *y,
 double *stat, const int weighted) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  double change=0.0;
  if(i<n) {
    double z = x[i];
    for (int e=row[i]; e<row[i+1]; ++e) {
      double a = x[col[e]] + (weighted ? w[e] : 1.0);
      z = fmin(z,a);
    }
    y[i] = z;
    change=z<x[i];
  }
  change=block_sum(change);
  if(threadIdx.x==0 && change) atomicAdd(stat+1,change);
}
extern "C" __global__ void patch(const int n, const int *pos,
 const double *value, double *out) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i<n) out[pos[i]]=value[i];
}
extern "C" __global__ void gather(const int n, const int *pos,
 const double *value, double *out) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i<n) out[i]=value[pos[i]];
}
extern "C" __global__ void degree(const int m, const int *col,
 const double *w, double *sum) {
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  if(i<m) atomicAdd(sum+col[i],w[i]);
}
extern "C" __global__ void normalize(const int m, const int *col,
 const double *w, const double *sum, double *out) {
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  if(i<m) out[i]=sum[col[i]] ? w[i]/sum[col[i]] : 0.0;
}
extern "C" __global__ void mask(const int n, const double *sum, unsigned char *out) {
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  if(i<n) out[i]=sum[i]==0.0;
}
'''


@dataclass
class Slot:
    offset: int
    capacity: int
    array: object


class Arena:
    """An owned CUDA allocation with explicit suballocation and coalescing."""

    def __init__(self, budget_bytes):
        self.budget = int(budget_bytes)
        if self.budget < 256 or self.budget % 256:
            raise ValueError('budget must be a positive multiple of 256 bytes')
        self.memory = cp.cuda.memory.Memory(self.budget)
        self.free = [(0, self.budget)]
        self.slots = {}
        self.live = self.peak = self.allocations = 0
        self.compactions = self.compaction_bytes = 0

    @staticmethod
    def capacity(shape, dtype):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        return max(256, (size + 255) // 256 * 256)

    def largest_free(self):
        return max((n for _, n in self.free), default=0)

    def allocate(self, key, shape, dtype):
        if key in self.slots:
            raise KeyError(key)
        size = self.capacity(shape, dtype)
        choices = [(n, i, off) for i, (off, n) in enumerate(self.free) if n >= size]
        if not choices:
            raise MemoryError(f'arena fragmentation/capacity: {key} needs {size}; largest={self.largest_free()}')
        n, i, off = min(choices)
        self.free.pop(i)
        if n > size:
            self.free.append((off + size, n - size))
        arr = cp.ndarray(shape, dtype=dtype,
                         memptr=cp.cuda.memory.MemoryPointer(self.memory, off))
        self.slots[key] = Slot(off, size, arr)
        self.live += size
        self.peak = max(self.peak, self.live)
        self.allocations += 1
        assert self.live <= self.budget
        return arr

    def release(self, key):
        slot = self.slots.pop(key)
        self.live -= slot.capacity
        self.free.append((slot.offset, slot.capacity))
        merged = []
        for off, size in sorted(self.free):
            if merged and merged[-1][0] + merged[-1][1] == off:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((off, size))
        self.free = merged

    def compact(self, staging_key='@patch_v'):
        """Pack graph allocations at a synchronized request boundary.

        Workspace addresses stay fixed. Two bounded device copies per chunk
        avoid undefined overlapping cudaMemcpy behavior.
        """
        cp.cuda.runtime.deviceSynchronize()
        stage=self.slots[staging_key]
        cursor=max(s.offset+s.capacity for k,s in self.slots.items() if str(k).startswith('@'))
        if any(s.offset<cursor for k,s in self.slots.items() if not str(k).startswith('@')):
            raise ValueError('Workspace must occupy the arena prefix before graph compaction')
        stream=cp.cuda.get_current_stream().ptr
        moved=0
        for key,slot in sorted(list(self.slots.items()),key=lambda kv:kv[1].offset):
            if str(key).startswith('@'): continue
            if cursor != slot.offset:
                for lo in range(0,slot.capacity,stage.capacity):
                    count=min(stage.capacity,slot.capacity-lo)
                    cp.cuda.runtime.memcpyAsync(stage.array.data.ptr,self.memory.ptr+slot.offset+lo,count,3,stream)
                    cp.cuda.runtime.memcpyAsync(self.memory.ptr+cursor+lo,stage.array.data.ptr,count,3,stream)
                arr=cp.ndarray(slot.array.shape,dtype=slot.array.dtype,
                               memptr=cp.cuda.memory.MemoryPointer(self.memory,cursor))
                self.slots[key]=Slot(cursor,slot.capacity,arr)
                moved+=slot.capacity
            cursor+=slot.capacity
        cp.cuda.runtime.deviceSynchronize()
        self.free=[(cursor,self.budget-cursor)] if cursor<self.budget else []
        if moved:
            self.compactions+=1
            self.compaction_bytes+=moved*2

    def resize_workspace(self, layout, staging_key='@patch_v'):
        """Grow scratch in the same CUDA allocation at a synchronized boundary.

        Scratch contents are disposable. Retained graph allocations are first
        packed, then shifted right through the old bounded staging buffer. All
        @ handles are rebound by the session after this method returns.
        """
        old_end=max(s.offset+s.capacity for k,s in self.slots.items() if str(k).startswith('@'))
        target=sum(self.capacity(shape,dtype) for _,shape,dtype in layout)
        graph_bytes=sum(s.capacity for k,s in self.slots.items() if not str(k).startswith('@'))
        if target<old_end or target+graph_bytes>self.budget:
            raise MemoryError('Workspace growth and retained graph state exceed arena budget')
        if len({k for k,_,_ in layout})!=len(layout) or any(not str(k).startswith('@') for k,_,_ in layout):
            raise ValueError('Workspace layout requires unique @ keys')
        # Construct every new workspace view before mutating arena metadata.
        workspace={};offset=0
        for key,shape,dtype in layout:
            capacity=self.capacity(shape,dtype)
            array=cp.ndarray(shape,dtype=dtype,memptr=cp.cuda.memory.MemoryPointer(self.memory,offset))
            workspace[key]=Slot(offset,capacity,array);offset+=capacity
        self.compact(staging_key)
        stage=self.slots[staging_key]
        graph_slots=sorted(((k,s) for k,s in self.slots.items() if not str(k).startswith('@')),key=lambda x:x[1].offset)
        delta=target-old_end
        stream=cp.cuda.get_current_stream().ptr
        if delta and graph_bytes:
            # memmove to higher addresses: copy chunks from the end backwards.
            remaining=graph_bytes
            while remaining:
                count=min(stage.capacity,remaining);lo=remaining-count
                cp.cuda.runtime.memcpyAsync(stage.array.data.ptr,self.memory.ptr+old_end+lo,count,3,stream)
                cp.cuda.runtime.memcpyAsync(self.memory.ptr+target+lo,stage.array.data.ptr,count,3,stream)
                remaining=lo
            cp.cuda.runtime.deviceSynchronize()
            self.compactions+=1;self.compaction_bytes+=2*graph_bytes
        for key,slot in graph_slots:
            off=slot.offset+delta
            array=cp.ndarray(slot.array.shape,dtype=slot.array.dtype,memptr=cp.cuda.memory.MemoryPointer(self.memory,off))
            workspace[key]=Slot(off,slot.capacity,array)
        self.slots=workspace
        self.live=target+graph_bytes
        self.peak=max(self.peak,self.live)
        self.allocations+=len(layout)
        self.free=[(self.live,self.budget-self.live)] if self.live<self.budget else []

    def close(self):
        cp.cuda.runtime.deviceSynchronize()
        self.slots.clear()
        self.memory = None


class HostGraph:
    """Incoming CSR plus an index of outgoing positions for bounded repair."""

    def __init__(self, name, n, src, dst, attributes, nodes=None):
        self.name = name
        self.n = int(n)
        self.nodes = list(range(n)) if nodes is None else list(nodes)
        if len(self.nodes) != n or len(set(self.nodes)) != n:
            raise ValueError('node labels must uniquely cover every vertex')
        self.node_to_id = {v: i for i, v in enumerate(self.nodes)}
        src, dst = np.asarray(src, np.int32), np.asarray(dst, np.int32)
        if src.shape != dst.shape or (src.size and (min(src.min(),dst.min()) < 0 or max(src.max(),dst.max()) >= n)):
            raise ValueError('invalid edge endpoints')
        order = np.lexsort((src, dst))
        self.col, self.target = src[order], dst[order]
        if len(order)>1 and np.any((self.col[1:]==self.col[:-1]) & (self.target[1:]==self.target[:-1])):
            raise ValueError('simple graph input must contain no duplicate edges')
        self.row = np.r_[0, np.cumsum(np.bincount(self.target, minlength=n))].astype(np.int32)
        self.attributes = {k: np.ascontiguousarray(np.asarray(v, np.float64)[order]) for k,v in attributes.items()}
        if any(len(v) != len(src) or not np.all(np.isfinite(v)) or np.any(v < 0) for v in self.attributes.values()):
            raise ValueError('weights must be finite nonnegative edge arrays')
        self.out_order = np.argsort(self.col, kind='stable').astype(np.int32)
        self.in_to_out = np.empty(len(self.col),np.int32)
        self.in_to_out[self.out_order] = np.arange(len(self.col),dtype=np.int32)
        self.out_row = np.r_[0, np.cumsum(np.bincount(self.col, minlength=n))].astype(np.int32)
        self.generation = 0
        self.versions = defaultdict(int)
        self.arrays = {}
        self.changes = defaultdict(list)
        self.change_floors = defaultdict(int)
        self.view_builds = self.repaired_values = 0
        self.host_prepare_seconds = 0.0

    @classmethod
    def from_easygraph(cls, graph, name='graph', attributes=('weight',)):
        nodes = list(graph.nodes)
        mapping = {v:i for i,v in enumerate(nodes)}
        src, dst = [], []
        weights = {k: [] for k in attributes}
        for u in nodes:
            for v, data in graph[u].items():
                src.append(mapping[u]); dst.append(mapping[v])
                for k in attributes:
                    weights[k].append(float(data.get(k, 1.0)))
        return cls(name, len(nodes), src, dst, weights, nodes=nodes)

    def key(self, kind, attr=None):
        return (self.name, kind, attr)

    def version(self, key):
        _, kind, attr = key
        return (self.generation, self.versions[attr] if kind in ('weight','out_weight','norm','dangling') and attr is not None else 0)

    def dependency(self, key):
        _, kind, attr = key
        deps = [self.name+':topology', self.name+':mapping']
        if kind in ('weight','out_weight','norm','dangling') and attr is not None:
            deps.append(self.name+':attr:'+attr)
        return tuple(deps)

    def get(self, key):
        start = time.perf_counter()
        _, kind, attr = key
        if kind == 'row': return self.row
        if kind == 'col': return self.col
        if kind == 'out_row': return self.out_row
        if kind == 'out_col':
            if key not in self.arrays:
                self.arrays[key]=np.ascontiguousarray(self.target[self.out_order],dtype=np.int32)
                self.view_builds+=1
            return self.arrays[key]
        if kind == 'weight': return self.attributes[attr]
        if kind == 'out_weight':
            if key not in self.arrays:
                self.arrays[key]=np.ascontiguousarray(self.attributes[attr][self.out_order])
                self.view_builds+=1
            return self.arrays[key]
        if key not in self.arrays:
            values = np.ones(len(self.col), np.float64) if attr is None else self.attributes[attr]
            sums = np.bincount(self.col, weights=values, minlength=self.n)
            if kind == 'norm':
                denom = sums[self.col]
                out = np.zeros(len(values), np.float64)
                np.divide(values, denom, out=out, where=denom!=0)
            elif kind == 'dangling': out = np.asarray(sums == 0, np.uint8)
            else: raise ValueError(kind)
            self.arrays[key] = out
            self.view_builds += 1
        self.host_prepare_seconds += time.perf_counter() - start
        return self.arrays[key]

    def update(self, attr, positions, values):
        positions = np.asarray(positions, np.int32)
        values = np.asarray(values, np.float64)
        if positions.shape != values.shape or np.any(positions < 0) or np.any(positions >= len(self.col)):
            raise ValueError('invalid update batch')
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError('nonnegative finite weights required')
        self.attributes[attr][positions] = values
        sources = np.unique(self.col[positions])
        affected = np.concatenate([self.out_order[self.out_row[u]:self.out_row[u+1]] for u in sources]) if sources.size else positions
        self.versions[attr] += 1
        version = self.versions[attr]
        for kind in ('weight','out_weight','norm','dangling'):
            key = self.key(kind, attr)
            touched = (positions if kind=='weight' else self.in_to_out[positions] if kind=='out_weight'
                       else sources if kind=='dangling' else affected)
            self.changes[key].append((version, touched.copy()))
            if kind=='out_weight' and key in self.arrays:
                self.arrays[key][touched]=self.attributes[attr][positions]
            if kind == 'norm' and key in self.arrays:
                a = self.attributes[attr]
                for u in sources:
                    e = self.out_order[self.out_row[u]:self.out_row[u+1]]
                    total = a[e].sum()
                    self.arrays[key][e] = a[e] / total if total else 0.0
                self.repaired_values += len(affected)
            if kind == 'dangling' and key in self.arrays:
                for u in sources:
                    e = self.out_order[self.out_row[u]:self.out_row[u+1]]
                    self.arrays[key][u] = self.attributes[attr][e].sum() == 0

    def dirty_positions(self, key, old_version):
        if old_version[0] != self.generation: return None
        if old_version[1] < self.change_floors[key]:return None
        pieces = [pos for ver,pos in self.changes[key] if ver > old_version[1]]
        return np.unique(np.concatenate(pieces)) if pieces else np.empty(0,np.int32)

    def compact_changes(self,oldest_versions):
        removed=0
        for key,history in self.changes.items():
            floor=oldest_versions.get(key,self.versions[key[2]])
            retained=[(version,positions) for version,positions in history if version>floor]
            removed+=len(history)-len(retained)
            self.changes[key]=retained
            self.change_floors[key]=max(self.change_floors[key],floor)
        return removed

    def repair_values(self, key, positions):
        _,kind,attr=key
        if kind=='weight': return self.attributes[attr][positions]
        if kind=='out_weight': return self.attributes[attr][self.out_order[positions]]
        if kind!='norm': return self.get(key)[positions]
        if key in self.arrays: return self.arrays[key][positions]
        values=self.attributes[attr]
        sources=self.col[positions]
        totals={int(u):values[self.out_order[self.out_row[u]:self.out_row[u+1]]].sum()
                for u in np.unique(sources)}
        den=np.asarray([totals[int(u)] for u in sources],np.float64)
        out=np.zeros(len(positions),np.float64)
        np.divide(values[positions],den,out=out,where=den!=0)
        return out

    def reference(self, function, attr=None, source=0, alpha=.85, tol=1e-8, max_iter=300):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import breadth_first_order, shortest_path
        if self.n==0:return np.empty(0,np.float64)
        if function=='pr':
            matrix=csr_matrix((self.get(self.key('norm',attr)), self.col, self.row),shape=(self.n,self.n))
            dangling=self.get(self.key('dangling',attr)).astype(bool)
            x=np.full(self.n,1/self.n,np.float64)
            for _ in range(max_iter):
                y=(1-alpha)/self.n+alpha*(matrix@x+x[dangling].sum()/self.n)
                if np.abs(y-x).sum()<tol: return y
                x=y
            raise RuntimeError('reference PageRank did not converge')
        data=np.ones(len(self.col)) if function=='bfs' else self.attributes[attr]
        matrix=csr_matrix((data,self.col,self.row),shape=(self.n,self.n)).transpose().tocsr()
        return shortest_path(matrix,directed=True,indices=source,unweighted=function=='bfs',method='D')


class Executor:
    @staticmethod
    def workspace_layout(nmax,patch_capacity=8192):
        return [('@x',(nmax,),np.float64),('@y',(nmax,),np.float64),
                ('@stat',(2,),np.float64),('@patch_i',(patch_capacity,),np.int32),
                ('@patch_v',(patch_capacity,),np.float64)]

    def bind_workspace(self,nmax):
        self.nmax=int(nmax)
        for name in ('x','y','stat','patch_i','patch_v'):
            setattr(self,name,self.arena.slots['@'+name].array)
        self.workspace_bytes=sum(s.capacity for k,s in self.arena.slots.items() if str(k).startswith('@'))

    def __init__(self, arena, nmax, patch_capacity=8192):
        self.arena, self.nmax = arena, nmax
        self.x = arena.allocate('@x',(nmax,),np.float64)
        self.y = arena.allocate('@y',(nmax,),np.float64)
        self.stat = arena.allocate('@stat',(2,),np.float64)
        self.patch_i=arena.allocate('@patch_i',(patch_capacity,),np.int32)
        self.patch_v=arena.allocate('@patch_v',(patch_capacity,),np.float64)
        self.patch_capacity=patch_capacity
        self.workspace_bytes=arena.live
        module = cp.RawModule(code=CUDA_SOURCE, options=('--std=c++17',),backend='nvcc')
        self.kernels = {name:module.get_function(name) for name in ('init','stats','pagerank','shortest','patch','gather','degree','normalize','mask')}
        self.events=(cp.cuda.Event(),cp.cuda.Event())
        self.result_events=(cp.cuda.Event(),cp.cuda.Event())

    def normalize(self, graph, col, raw, norm, dangling):
        cp.cuda.runtime.memsetAsync(self.x.data.ptr,0,graph.n*8,cp.cuda.get_current_stream().ptr)
        m=len(graph.col)
        if m:
            self.kernels['degree'](((m+127)//128,),(128,),(np.int32(m),col,raw,self.x))
            self.kernels['normalize'](((m+127)//128,),(128,),(np.int32(m),col,raw,self.x,norm))
        if graph.n:
            self.kernels['mask'](((graph.n+127)//128,),(128,),(np.int32(graph.n),self.x,dangling))

    def patch(self, array, positions, values):
        if array.dtype != cp.float64:  # small masks are copied as a whole
            raise TypeError('scatter patch expects float64')
        for lo in range(0,len(positions),self.patch_capacity):
            p=np.ascontiguousarray(positions[lo:lo+self.patch_capacity],dtype=np.int32)
            v=np.ascontiguousarray(values[lo:lo+self.patch_capacity],dtype=np.float64)
            count=len(p)
            self.patch_i[:count].set(p); self.patch_v[:count].set(v)
            self.kernels['patch'](((count+127)//128,),(128,),(np.int32(count),self.patch_i,self.patch_v,array))

    def download(self,device_array,selected=None):
        """Complete request-owned downloads using only reserved gather scratch."""
        t=time.perf_counter();d2h=0.;launches=0;gather_device=0.
        if selected is None:
            start=time.perf_counter();output=device_array.get();d2h=time.perf_counter()-start
            index_bytes=0
        else:
            selected=np.ascontiguousarray(selected,dtype=np.int32)
            output=np.empty(len(selected),np.float64)
            index_bytes=selected.nbytes
            for lo in range(0,len(selected),self.patch_capacity):
                count=min(self.patch_capacity,len(selected)-lo)
                self.patch_i[:count].set(selected[lo:lo+count])
                start_event,end_event=self.result_events;start_event.record()
                self.kernels['gather'](((count+127)//128,),(128,),
                    (np.int32(count),self.patch_i,device_array,self.patch_v))
                end_event.record()
                start=time.perf_counter();self.patch_v[:count].get(out=output[lo:lo+count])
                d2h+=time.perf_counter()-start;launches+=1
                gather_device+=cp.cuda.get_elapsed_time(start_event,end_event)/1000
        return output,dict(download_seconds=time.perf_counter()-t,d2h_seconds=d2h,
                           result_d2h_bytes=output.nbytes,result_indices_h2d_bytes=index_bytes,
                           result_gather_launches=launches,result_gather_device_seconds=gather_device,
                           d2h_timer_scope='host blocking get; selected mode also waits for preceding gather',
                           output_host_bytes=output.nbytes)

    def run(self, graph, arrays, function, attr=None, source=0, alpha=.85,tol=1e-8,max_iter=300,selected=None):
        n=graph.n
        if not n:
            output,metrics=self.download(self.x[:0],selected)
            return output,dict(iterations=0,device_loop_seconds=0.,d2h_bytes=0,**metrics)
        grid=((n+127)//128,)
        x,y=self.x,self.y
        start,end=self.events
        host_time=time.perf_counter();start.record();event_start_seconds=time.perf_counter()-host_time
        host_time=time.perf_counter()
        self.kernels['init'](grid,(128,),(np.int32(n),x,np.int32(0 if function=='pr' else 1),np.int32(source)))
        init_launch_seconds=time.perf_counter()-host_time
        row,col=arrays[graph.key('row')],arrays[graph.key('col')]
        weight=arrays[graph.key('norm' if function=='pr' else 'weight',attr)] if function!='bfs' else x
        flag=np.empty(2,np.float64)
        converged=False
        for iteration in range(max_iter if function=='pr' else max(1,n)):
            cp.cuda.runtime.memsetAsync(self.stat.data.ptr,0,16,cp.cuda.get_current_stream().ptr)
            if function=='pr':
                self.kernels['stats'](grid,(128,),(np.int32(n),x,arrays[graph.key('dangling',attr)],self.stat))
                self.kernels['pagerank'](grid,(128,),(np.int32(n),row,col,weight,x,y,self.stat,np.float64(alpha)))
            else:
                self.kernels['shortest'](grid,(128,),(np.int32(n),row,col,weight,x,y,self.stat,np.int32(function=='sssp')))
            self.stat.get(out=flag)
            x,y=y,x
            if (function=='pr' and flag[1]<tol) or (function!='pr' and flag[1]==0):
                converged=True; break
        host_time=time.perf_counter();end.record(); end.synchronize();event_finish_seconds=time.perf_counter()-host_time
        if not converged: raise RuntimeError(function+' failed convergence')
        output,metrics=self.download(x[:n],selected)
        return output,dict(iterations=iteration+1,device_loop_seconds=cp.cuda.get_elapsed_time(start,end)/1000,
                           host_event_start_seconds=event_start_seconds,host_init_launch_seconds=init_launch_seconds,
                           host_event_finish_seconds=event_finish_seconds,
                           d2h_bytes=output.nbytes+16*(iteration+1),**metrics)
