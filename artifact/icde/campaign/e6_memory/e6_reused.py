"""Existing E6 algorithm adapters, extracted without algorithm changes.

Source/function identities are recorded in e6_reuse_provenance.json.
Class/namespace adaptation plus explicit complete vertex-domain validation.
Validation is inside the measured complete-call interval and does not alter algorithms.
"""
from pathlib import Path
import sys,contextlib
K=32
QUANTUM=1e-6


def own(x):
 import numpy as np
 a=np.array(x,copy=True);a.flags.writeable=False;return a

def require_complete_ids(ids,n,xp):
 """Reject absent/duplicate/out-of-range vertices before scatter or reduction."""
 if ids.ndim!=1 or len(ids)!=n or ids.dtype.kind not in "iu":raise ValueError("Incomplete vertex output domain")
 if n and (bool(xp.any(ids<0)) or bool(xp.any(ids>=n)) or int(xp.unique(ids).size)!=n):raise ValueError("Duplicate or invalid output vertices")
 return ids

def canonical_labels(values):
 import numpy as np
 v=np.asarray(values);u,inv=np.unique(v,return_inverse=True);mins=np.full(len(u),len(v),dtype='i8');np.minimum.at(mins,inv,np.arange(len(v)));return mins[inv]

def select_candidates(effective,constraint,k=K):
 import numpy as np
 e=np.asarray(effective);c=np.asarray(constraint)
 if e.shape!=c.shape or e.ndim!=1:raise ValueError('Different score domains')
 ids=np.flatnonzero(np.isfinite(e)&np.isfinite(c))
 if not len(ids):raise ValueError('No non-isolated candidates')
 # Fixed task semantics: numerical scores are ranked at 1e-6 resolution,
 # then canonical vertex ID breaks ties. No CPU reference enters selection.
 qe=np.rint(e[ids]/QUANTUM).astype('i8');qc=np.rint(c[ids]/QUANTUM).astype('i8')
 return ids[np.lexsort((ids,qc,-qe))[:min(k,len(ids))]]

def select_winner(candidates,closeness):
 import numpy as np
 ids=np.asarray(candidates);v=np.asarray(closeness)
 if v.shape!=ids.shape or not np.isfinite(v).all():raise ValueError('Invalid candidate closeness')
 return int(ids[np.lexsort((ids,-np.rint(v/QUANTUM).astype('i8')))[0]])

class IgraphBackend:
 def __init__(self,p,j,data):
  import igraph as ig
  self.ig=ig;self.data=data;self.graph=None;self.p=p;self.j=j
 def build(self):
  if self.graph is None:self.graph=self.ig.Graph(n=self.data['n'],edges=self.data['undirected'],directed=False)
 def call(self,fn,selected=None):
  import numpy as np
  self.build();g=self.graph
  if fn=='effective_size':
   degree=np.asarray(g.degree(),dtype='f8');lcc=np.asarray(g.transitivity_local_undirected(mode='zero',weights=None));v=degree-lcc*(degree-1);v[degree==0]=np.nan
  elif fn=='constraint':v=g.constraint(weights=None)
  elif fn=='closeness':
   d=np.asarray(g.distances(source=list(map(int,selected)),target=None,mode='all'),dtype='f8');r=np.isfinite(d).sum(axis=1)-1;s=np.where(np.isfinite(d),d,0).sum(axis=1);v=np.divide(r.astype(float)**2,(self.data['n']-1)*s,out=np.zeros(len(selected)),where=s>0)
  elif fn=='bfs':v=np.asarray(g.distances(source=[int(selected)],target=None,mode='all'))[0]
  elif fn=='wcc':v=canonical_labels(g.connected_components(mode='weak').membership)
  else:raise ValueError(fn)
  return own(v),dict(mode='CPU derived from native transitivity and degree' if fn=='effective_size' else 'CPU igraph C',devices_used=[],igraph_version=self.ig.__version__)
 def close(self):self.graph=None

class NativeBackend:
 def __init__(self,p,j,data):
  sys.path[:0]=[p['source'],str(Path(p['native_dso']['path']).parent)]
  import easygraph as eg,cpp_easygraph as cpp
  if Path(cpp.__file__).resolve()!=Path(p['native_dso']['path']).resolve():raise ValueError('Wrong native extension')
  for name in ['cpp_egmgpu_single_execute','cpp_egmgpu_replicated_effective_size','cpp_egmgpu_replicated_constraint','cpp_egmgpu_replicated_closeness']:
   if not hasattr(cpp,name):raise RuntimeError('Missing existing ABI '+name)
  self.eg,self.cpp,self.p,self.j=eg,cpp,p,j;self.graph=None;self.devices=list(range(j['P']))
  cpp.cpp_egmgpu_runtime_start(self.devices,'host_staging')
 def build(self):
  if self.graph is None:self.graph=self.eg.read_eggpu_csr(self.p['datasets'][self.j['dataset']]['manifests']['projected']['path'],validate=True).cpp()
 def call(self,fn,selected=None):
  self.build();cpp=self.cpp;g=self.graph
  if len(self.devices)>1 and fn in ('effective_size','constraint'):
   raw=getattr(cpp,'cpp_egmgpu_replicated_'+fn)(g,None,None,self.devices,True);mode='replicated_graph_ego_vertex_shards';used=list(raw['devices_used'])
  elif len(self.devices)>1 and fn=='closeness':
   raw=cpp.cpp_egmgpu_replicated_closeness(g,list(map(int,selected)),None,self.devices);mode='replicated_graph_selected_source_shards';used=list(raw['devices_used'])
  else:
   params={'weight':None}
   if fn in ('effective_size','constraint'):params['nodes']=None
   elif fn=='closeness':params['sources']=list(map(int,selected))
   elif fn=='bfs':params['source']=int(selected)
   raw=cpp.cpp_egmgpu_single_execute(g,fn,params,0);mode='single_GPU_ABI';used=[int(raw['device'])]
  if used!=(self.devices if len(self.devices)>1 and fn in ('effective_size','constraint','closeness') else [0]):raise ValueError('Actual ABI device coverage differs')
  values=canonical_labels(raw['integer_values_dense']) if fn=='wcc' else raw['values_dense']
  return own(values),dict(mode=mode,devices_used=used,kernel_seconds=float(raw['kernel_seconds']),public_Session=False)
 def close(self):
  self.graph=None;self.cpp.cpp_egmgpu_runtime_shutdown()

class MixedBackend:
 def __init__(self,p,j,data):
  # RAPIDS imports remain in their original interpreter. Only the existing
  # CPU igraph package is appended as an explicit same-Python fallback.
  import cugraph as cg,cudf,cupy as cp,numpy as np
  sys.path.append(p['fallback_site'])
  self.cpu=IgraphBackend(p,j,data);self.cg,self.cudf,self.cp=cg,cudf,cp;self.data=data;self.graph=None;self.gpu_executed=False
 def build(self):
  if self.graph is None:
   import numpy as np
   offsets=np.r_[0,np.cumsum(self.data['degree'])].astype('i4');indices=self.data['edges'][:,1].astype('i4')
   self.graph=self.cg.Graph(directed=False);self.graph.from_cudf_adjlist(self.cudf.Series(offsets),self.cudf.Series(indices),symmetrize=False);self.cp.cuda.runtime.deviceSynchronize();self.gpu_executed=True
 def scatter(self,frame,key):
  import numpy as np
  self.cp.cuda.runtime.deviceSynchronize()
  ids=frame['vertex'].to_numpy();v=frame[key].to_numpy();require_complete_ids(ids,self.data['n'],np)
  if v.shape!=ids.shape:raise ValueError('Vertex/value domain mismatch')
  out=np.empty(self.data['n'],dtype=v.dtype);out[ids]=v
  self.cp.cuda.runtime.deviceSynchronize();return out,ids
 def call(self,fn,selected=None):
  import numpy as np
  if fn in ('constraint','closeness'):
   value,details=self.cpu.call(fn,selected);details['mode']='CPU igraph fallback in mixed backend';return value,details
  self.build()
  if fn=='effective_size':
   frame=self.cg.triangle_count(self.graph);counts,ids=self.scatter(frame,'counts');degree=self.data['degree'];v=degree-np.divide(2*counts,degree,out=np.zeros(self.data['n']),where=degree>0);v[degree==0]=np.nan
  elif fn=='bfs':
   start=self.cudf.DataFrame({'vertex':self.cudf.Series([int(selected)],dtype='int32')})
   frame=self.cg.bfs(self.graph,start=start,return_predecessors=False);v,ids=self.scatter(frame,'distance');v=np.asarray(v,dtype='f8');v[v>=self.data['n']]=np.inf
  elif fn=='wcc':
   frame=self.cg.weakly_connected_components(self.graph);v,ids=self.scatter(frame,'labels');v=canonical_labels(v)
  else:raise ValueError(fn)
  return own(v),dict(mode='GPU derived effective_size via triangle_count and degree' if fn=='effective_size' else 'GPU cuGraph SG',devices_used=[0],cugraph_version=self.cg.__version__,vertex_ids=own(ids))
 def close(self):
  self.graph=None;self.cpu.close();self.cp.get_default_memory_pool().free_all_blocks();self.cp.cuda.runtime.deviceSynchronize()

def product_blocks(degree,edges,product_budget_bytes):
 """Topology-derived upper bound, not a claim about cuSPARSE workspace."""
 import numpy as np
 n=len(degree)
 walks=np.bincount(edges[:,0],weights=degree[edges[:,1]],minlength=n)
 if np.any(walks>=2**53):raise ValueError('Two-hop count exceeds exact float64 integer range')
 bounds=np.minimum(n,walks.astype('i8'))
 row_bytes=bounds*12+4
 if int(row_bytes.max(initial=0))>product_budget_bytes:raise MemoryError('One row exceeds product planning budget')
 cuts=[0];total=0
 for i,cost in enumerate(row_bytes):
  cost=int(cost)
  if total and total+cost>product_budget_bytes:cuts.append(i);total=0
  total+=cost
 cuts.append(n)
 return list(zip(cuts[:-1],cuts[1:])),dict(two_hop_walks=int(walks.sum()),product_nnz_upper=int(bounds.sum()),product_storage_upper_bytes=int(row_bytes.sum()),product_budget_bytes=int(product_budget_bytes),block_count=len(cuts)-1,max_block_rows=max(b-a for a,b in zip(cuts[:-1],cuts[1:])),workspace_bound_proven=False)

class GPUCompositeBackend(MixedBackend):
 def __init__(self,p,j,data):
  super().__init__(p,j,data);self.composition=p['composition'];self.j=j
  self.adjacency=None;self.probability=None;self.device_degree=None;self.blocks=None;self.product_plan=None
  from pylibcugraph import ResourceHandle,bfs
  self.primitive_handle=ResourceHandle();self.primitive_bfs=bfs
 def bfs_arrays(self,source):
  self.build()
  if self.graph.renumbered:raise ValueError('Direct primitive requires the frozen canonical CSR vertex domain')
  result=self.primitive_bfs(handle=self.primitive_handle,graph=self.graph._plc_graph,sources=self.cp.asarray([int(source)],dtype='i4'),direction_optimizing=False,depth_limit=-1,compute_predecessors=False,do_expensive_check=False)
  require_complete_ids(result[2],self.data['n'],self.cp)
  if result[0].shape!=result[2].shape:raise ValueError('BFS vertex/value domain mismatch')
  return result
 def degree(self):
  if self.device_degree is None:self.device_degree=self.cp.asarray(self.data['degree'],dtype='f8')
  return self.device_degree
 def build_sparse(self):
  if self.adjacency is not None:return
  import numpy as np
  from cupyx.scipy.sparse import csr_matrix
  cp=self.cp;n=self.data['n'];edges=self.data['edges'];degree=self.data['degree']
  offsets=np.r_[0,np.cumsum(degree)].astype('i4');indices=cp.asarray(edges[:,1],dtype='i4');indptr=cp.asarray(offsets)
  self.adjacency=csr_matrix((cp.ones(len(edges),dtype='f8'),indices,indptr),shape=(n,n))
  rows=cp.asarray(edges[:,0],dtype='i4')
  self.probability=csr_matrix((1.0/self.degree()[rows],indices,indptr),shape=(n,n))
  free,total=cp.cuda.runtime.memGetInfo()
  # Permit a full product whenever its topology-derived bound fits this budget.
  # The remaining space serves intermediate matrices and library workspace;
  # the allocation contract is checked by real GPU qualification, not this estimate.
  budget=min(6*(1<<30),int(free)//4)
  self.blocks,self.product_plan=product_blocks(degree,edges,budget)
  self.product_plan.update(cuda_free_bytes_at_plan=int(free),cuda_total_bytes=int(total),route='full_product' if len(self.blocks)==1 else 'topology_bounded_row_blocks')
 def call(self,fn,selected=None):
  import numpy as np
  cp=self.cp
  if fn=='closeness':
   self.build();n=self.data['n'];values=cp.empty(len(selected),dtype='f8')
   for i,node in enumerate(selected):
    d,predecessors,ids=self.bfs_arrays(node);valid=(d>=0)&(d<n)
    reachable=cp.count_nonzero(valid)-1;distance_sum=cp.sum(cp.where(valid,d,0),dtype='f8')
    values[i]=cp.where(distance_sum>0,reachable.astype('f8')**2/((n-1)*distance_sum),0.0)
   result=own(cp.asnumpy(values));cp.cuda.runtime.deviceSynchronize()
   return result,dict(mode='GPU derived WF closeness via pylibcugraph BFS with reused ResourceHandle and GPU reductions',devices_used=[0],selected_source_count=len(selected),distance_vectors_copied_to_host=False,resource_handle_identity=id(self.primitive_handle),pylibcugraph_direct=True,cugraph_version=self.cg.__version__)
  if fn=='bfs':
   d,predecessors,ids=self.bfs_arrays(selected);values=cp.empty(self.data['n'],dtype='f8');values[ids]=d;values[values>=self.data['n']]=cp.inf
   result=own(cp.asnumpy(values));host_ids=own(cp.asnumpy(ids));cp.cuda.runtime.deviceSynchronize()
   return result,dict(mode='GPU pylibcugraph BFS with reused ResourceHandle',devices_used=[0],resource_handle_identity=id(self.primitive_handle),pylibcugraph_direct=True,cugraph_version=self.cg.__version__,vertex_ids=host_ids)
  if fn=='effective_size' and self.composition=='gpu_composite':
   self.build();frame=self.cg.triangle_count(self.graph)
   ids=frame['vertex'].to_cupy();counts=frame['counts'].to_cupy();degree=self.degree();require_complete_ids(ids,self.data['n'],cp)
   if counts.shape!=ids.shape:raise ValueError('Triangle vertex/value domain mismatch')
   values=cp.empty(self.data['n'],dtype='f8');values[ids]=cp.where(degree[ids]>0,degree[ids]-2*counts/degree[ids],cp.nan)
   result=own(cp.asnumpy(values));host_ids=own(cp.asnumpy(ids));cp.cuda.runtime.deviceSynchronize()
   return result,dict(mode='GPU derived effective_size via triangle_count and GPU normalization',devices_used=[0],cugraph_version=self.cg.__version__,vertex_ids=host_ids)
  if fn=='constraint' and self.composition=='gpu_composite':
   self.build();self.build_sparse();pool=cp.get_default_memory_pool();observed={'pool_used':0,'pool_backing':0,'allocations':0}
   allocator=cp.cuda.get_allocator()
   def tracked(size):
    pointer=allocator(size);observed['pool_used']=max(observed['pool_used'],pool.used_bytes());observed['pool_backing']=max(observed['pool_backing'],pool.total_bytes());observed['allocations']+=1;return pointer
   context=cp.cuda.using_allocator(tracked) if self.j['phase']=='qualification' else contextlib.nullcontext()
   with context:
    values=cp.empty(self.data['n'],dtype='f8')
    for lo,hi in self.blocks:
     a=self.adjacency if (lo==0 and hi==self.data['n']) else self.adjacency[lo:hi]
     p=self.probability if (lo==0 and hi==self.data['n']) else self.probability[lo:hi]
     two_hop=p@self.probability
     local=(p+two_hop).multiply(a)
     values[lo:hi]=cp.asarray(local.multiply(local).sum(axis=1)).ravel()
     del two_hop,local
    values[self.degree()==0]=cp.nan;result=own(cp.asnumpy(values));cp.cuda.runtime.deviceSynchronize()
   free,total=cp.cuda.runtime.memGetInfo()
   return result,dict(mode='GPU derived constraint via CuPy CSR SpGEMM, adjacency mask and row reduction',devices_used=[0],cupy_version=cp.__version__,product_plan=self.product_plan,allocator_instrumented=self.j['phase']=='qualification',observed_cupy_pool=observed if self.j['phase']=='qualification' else None,pool_used_bytes_at_stage_end=pool.used_bytes(),pool_backing_bytes_at_stage_end=pool.total_bytes(),cuda_free_bytes_at_stage_end=int(free),cuda_total_bytes=int(total),memory_scope='CuPy pool observations exclude cuDF/RMM and unexposed cuSPARSE internal allocations; CUDA free is a boundary observation, not peak process memory.')
  return super().call(fn,selected)
 def close(self):
  self.adjacency=None;self.probability=None;self.device_degree=None;super().close();self.primitive_handle=None
