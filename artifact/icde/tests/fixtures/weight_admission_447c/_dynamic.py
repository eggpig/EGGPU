"""Immutable snapshots with native weight maintenance and cache-first lookup.

The same full native WPR/SSSP kernels run on P1/P2. Sparse maintenance is a real
CUDA scatter; host diff, PR summary rebuilding and manifest costs remain paid.
Old semantic snapshots are host-backed. Only completed catalog-exclusive device
weights may be reused; live native view holders force COW.
"""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import copy,errno,hashlib,json,operator,os,shutil,tempfile
import numpy as np
from easygraph.gpu._manifest_signatures import sign
from easygraph.gpu._native import UnifiedNativeSession
from easygraph.gpu.contracts import NodeDomain
from easygraph.gpu._controller import StateSwitchController

ROUTES=('persistent_full','lazy_sparse_cow','lease_sparse')
RETIREMENT_POLICIES=('lineage','all_other_views')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def verify_manifest(path,document):
 path=Path(path).resolve();allowed={'int32':4,'uint64':8,'float64':8};total=0
 if int(document['graph']['vertex_count'])<=0:raise ValueError('Empty vertex domain')
 for part in document['partitions']:
  for name,item in part['files'].items():
   target=(path.parent/item['path']).resolve();target.relative_to(path.parent)
   if item['dtype'] not in allowed:raise ValueError('Unsupported dtype')
   if target.stat().st_size!=item['byte_size'] or item['count']*allowed[item['dtype']]!=item['byte_size'] or sha(target)!=item['sha256']:raise ValueError('Snapshot payload identity mismatch')
   if name=='weights' and item['count']:
    weights=np.memmap(target,mode='r',dtype='<f8')
    for first in range(0,len(weights),1<<18):
     block=weights[first:first+(1<<18)]
     if not np.all(np.isfinite(block)) or np.any(block<0):raise ValueError('Weights must be finite nonnegative f64')
    del weights
  total+=part['owned_vertex_count']
 if total!=document['graph']['vertex_count']:raise ValueError('Incomplete ownership')
 topology,weight,view=sign(document)
 if topology!=document['graph']['topology_sha256'] or weight!=document['graph'].get('weight_sha256') or view!=document['view']['signature']:raise ValueError('Manifest content signatures disagree')
 return sha(path)


def stage_weights(base_path,new_name,updates,directory):
 """CPU staging only. Validate every absolute set before writing any payload."""
 begin=perf_counter();base_path=Path(base_path).resolve();doc=json.loads(base_path.read_text());verify_manifest(base_path,doc)
 if not doc['graph']['weighted']:raise ValueError('Weight commits require weighted base')
 if not isinstance(new_name,str) or not new_name or new_name in ('.','..') or Path(new_name).name!=new_name:raise ValueError('Invalid snapshot name')
 parts={p['partition_id']:p for p in doc['partitions']};prepared={};seen=set()
 for item in updates:
  if len(item)!=3:raise ValueError('Require partition, position, absolute value')
  p,pos,value=item
  if isinstance(p,(bool,np.bool_)) or isinstance(pos,(bool,np.bool_)):raise ValueError('Integer positions required')
  try:p=operator.index(p);pos=operator.index(pos)
  except TypeError as e:raise ValueError('Integer positions required') from e
  if p not in parts or pos<0 or pos>=parts[p]['files']['weights']['count']:raise ValueError('Patch position outside partition')
  if (p,pos) in seen:raise ValueError('Duplicate absolute-set position')
  value=float(value)
  if not np.isfinite(value) or value<0:raise ValueError('Finite nonnegative weights required')
  seen.add((p,pos));prepared.setdefault(p,[]).append((pos,value))
 directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True);target=directory/new_name
 if target.exists():raise FileExistsError('Snapshot directory already exists')
 stage=Path(tempfile.mkdtemp(prefix='.stage-'+new_name+'-',dir=directory));written=0;copied=0
 try:
  doc=copy.deepcopy(doc)
  doc['graph']['id']=new_name;doc['graph']['generation']=int(doc['graph'].get('generation',0))+1
  for part in doc['partitions']:
   p=part['partition_id']
   for name,spec in part['files'].items():
    original=base_path.parent/spec['path'];destination=stage/f'p{p}.{name}.bin'
    if name=='weights' and p in prepared:
     values=np.fromfile(original,dtype='<f8')
     for pos,value in prepared[p]:values[pos]=value
     values.tofile(destination);written+=int(values.nbytes);spec['sha256']=sha(destination)
    else:
     try:os.link(original,destination)
     except OSError as exc:
      if exc.errno!=errno.EXDEV:raise
      shutil.copyfile(original,destination);copied+=destination.stat().st_size
    spec['path']=destination.name
  topology,weight,view=sign(doc)
  if topology!=doc['graph']['topology_sha256']:raise AssertionError('Weight commit changed topology')
  doc['graph']['weight_sha256']=weight;doc['view']['signature']=view
  manifest=stage/'manifest.json';manifest.write_text(json.dumps(doc,indent=2)+'\n');verify_manifest(manifest,doc)
  stage.rename(target)
 except BaseException:
  if stage.exists():shutil.rmtree(stage)
  raise
 return target/'manifest.json',dict(kind='absolute_weight_commit',positions=len(seen),semantic_value_bytes=8*len(seen),host_weight_payload_written_bytes=written,host_cross_filesystem_copy_bytes=copied,host_payload_written_bytes=written+copied,host_stage_seconds=perf_counter()-begin,manifest=str(target/'manifest.json'),topology_sha256=topology,weight_sha256=weight,gpu_maintenance='lazy on next full native query')


def _dynamic_totals(inventory):
 out={}
 for row in inventory['per_device']:
  out[int(row['device_id'])]={k.removeprefix('dynamic.'):int(v) for k,v in row['counters'].items() if k.startswith('dynamic.')}
 return out

class DynamicNativeSession(UnifiedNativeSession):
 capabilities=UnifiedNativeSession.capabilities|frozenset({'fixed_topology_absolute_weight_commit','native_memory_absolute_weight_commit','native_cuda_sparse_weight_patch','catalog_exclusive_weight_lease'})
 def __init__(self,cpp,manifests,*,devices=(0,),route='persistent_full',retirement_policy='lineage',**options):
  if route not in ROUTES:raise ValueError('Invalid dynamic route')
  if retirement_policy not in RETIREMENT_POLICIES:raise ValueError('Invalid snapshot retirement policy')
  self.retirement_policy=retirement_policy;self._snapshot_lineage={}
  self.route=route;self._retained_snapshots=set();self._route_before=os.environ.get('EGGPU_DYNAMIC_WEIGHT_ROUTE')
  os.environ['EGGPU_DYNAMIC_WEIGHT_ROUTE']=route
  self._options=options
  try:
   super().__init__(cpp,manifests,devices=devices,**options)
   # Independent input snapshots are separate families. Only commit_weights
   # establishes a version relationship; identical topology does not establish it.
   self._snapshot_lineage={name:name for name in self.graphs}
   state=self.resource_inventory()
   if any(row['counters'].get('dynamic.abi_version')!=1 for row in state['per_device']):raise RuntimeError('DSO lacks dynamic CUDA weight ABI')
  except BaseException:
   if hasattr(self,'closed') and not self.closed:super().close()
   self._restore_environment();raise
 def capability_descriptor(self):
  base=super().capability_descriptor()
  return replace(base,mutations='fixed-topology absolute weight commits into new immutable snapshots',
                 control_routes=base.control_routes+('commit_weights','register_snapshot',self.route,'retirement:'+self.retirement_policy),
                 limitations=base.limitations+('Topology changes require a separately constructed snapshot manifest',))
 def _restore_environment(self):
  if self._route_before is None:os.environ.pop('EGGPU_DYNAMIC_WEIGHT_ROUTE',None)
  else:os.environ['EGGPU_DYNAMIC_WEIGHT_ROUTE']=self._route_before
 def _validate_manifest_document(self,name,path,document):self.snapshot_hashes[name]=verify_manifest(path,document)
 def register_snapshot(self,name,manifest_path):
  with self._lock:
   if self.closed:raise RuntimeError('Session closed')
   path=Path(manifest_path).resolve();document=json.loads(path.read_text());digest=verify_manifest(path,document)
   if name in self.graphs:
    if self.snapshot_hashes[name]==digest:return self.describe(name)
    raise ValueError('Existing snapshot name is immutable')
   if sorted(p['device_id'] for p in document['partitions'])!=sorted(self.devices):raise ValueError('Placement differs from session')
   identity=(document['graph']['id'],document['graph'].get('generation'))
   if any(identity==(meta['id'],meta.get('generation')) for meta in self.metadata.values()):raise ValueError('Native snapshot identity already registered')
   native=self.cpp.cpp_egmgpu_load_partition_manifest(str(path),import_policy='structural')
   nodes=np.arange(document['graph']['vertex_count'],dtype=np.int64);nodes.flags.writeable=False
   domain=NodeDomain.from_nodes(nodes)
   self.graphs[name]=native;self.metadata[name]=document['graph'];self.nodes[name]=nodes
   self._manifest_documents[name]=document;self._manifest_paths[name]=str(path);self._domains[name]=domain;self.snapshot_hashes[name]=digest
   self._snapshot_lineage[name]=name
   if isinstance(self.controller,StateSwitchController):self.controller.graph_bindings[document['graph']['id']]=digest
   return self.describe(name)
 def _registered_host_payload(self):
  return dict(snapshots=len(self.graphs),array_payload_bytes=sum(spec['byte_size'] for document in self._manifest_documents.values() for part in document['partitions'] for spec in part['files'].values()),
   scope='Named immutable native snapshots retained for old-name queries; sum of array payload sizes only. Excludes host plans, labels, packets, allocator capacity and RSS. Lineage retirement concerns GPU views; host registry is O(RB) until close.')
 def _commit_native_memory(self,base_name,new_name,updates):
  begin=perf_counter()
  base=self.graphs[base_name]
  identity=(new_name,int(self.metadata[base_name].get('generation',0))+1)
  if any(identity==(meta['id'],meta.get('generation')) for meta in self.metadata.values()):raise ValueError('Native snapshot identity already registered')
  receipt=dict(self.cpp.cpp_egmgpu_commit_weights_memory(base,new_name,updates))
  native=receipt.pop('graph')
  document=copy.deepcopy(self._manifest_documents[base_name])
  document['snapshot_storage']='native_memory'
  document['graph']['id']=new_name;document['graph']['generation']=receipt['generation']
  by_part={item['partition_id']:item for item in receipt['partitions']}
  for part in document['partitions']:
   row=by_part[part['partition_id']]
   if row['device_id']!=part['device_id'] or row['weight_count']!=part['adjacency_count']:raise AssertionError('Native commit changed placement or geometry')
   part['files']['weights']['sha256']=row['weight_sha256']
   for spec in part['files'].values():
    spec['path']=None;spec['storage']='native_memory'
  topology,weight,view=sign(document)
  if topology!=self.metadata[base_name]['topology_sha256'] or (topology,weight,view)!=(receipt['topology_sha256'],receipt['weight_sha256'],receipt['view_signature']):
   raise AssertionError('Native/Python content signatures disagree')
  document['graph']['weight_sha256']=weight;document['view']['signature']=view
  identity=dict(kind='native_memory_snapshot_v1',base_snapshot_sha256=self.snapshot_hashes[base_name],document=document)
  digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()
  # All validations precede semantic publication. Node identity and placement
  # remain immutable, so the existing read-only node domain can be retained.
  self.graphs[new_name]=native;self.metadata[new_name]=document['graph'];self.nodes[new_name]=self.nodes[base_name]
  self._manifest_documents[new_name]=document;self._manifest_paths[new_name]=None
  self._domains[new_name]=self._domains[base_name];self.snapshot_hashes[new_name]=digest
  self._snapshot_lineage[new_name]=self._snapshot_lineage[base_name]
  if isinstance(self.controller,StateSwitchController):self.controller.graph_bindings[new_name]=digest
  receipt.update(kind='native_memory_absolute_weight_commit',positions=receipt['requested_positions'],
   semantic_value_bytes=8*receipt['requested_positions'],snapshot_storage='native_memory',snapshot_sha256=digest,
   snapshot_identity_kind='native_memory_snapshot_v1',base_snapshot_sha256=self.snapshot_hashes[base_name],
   snapshot_lineage=self._snapshot_lineage[new_name],manifest=None,host_weight_payload_written_bytes=0,
   host_payload_written_bytes=0,host_cross_filesystem_copy_bytes=0,host_stage_seconds=0.,
   registered_host_payload=self._registered_host_payload(),commit_seconds=perf_counter()-begin,gpu_maintenance='lazy on next full native query')
  return receipt
 def commit_weights(self,base_name,new_name,updates,manifest_directory=None,*,storage='files'):
  begin=perf_counter()
  with self._lock:
   if self.closed:raise RuntimeError('Session closed')
   if base_name not in self.graphs:raise ValueError('Unknown registered base snapshot')
   if new_name in self.graphs:raise ValueError('New name already registered')
   if not isinstance(new_name,str) or not new_name or new_name in ('.','..') or Path(new_name).name!=new_name:raise ValueError('Invalid snapshot name')
   if storage not in ('files','native_memory'):raise ValueError('Unknown commit storage')
   if storage=='native_memory':return self._commit_native_memory(base_name,new_name,updates)
   if manifest_directory is None or self._manifest_paths[base_name] is None:raise ValueError('File commit requires an on-disk base and output directory')
   path,receipt=stage_weights(self._manifest_paths[base_name],new_name,updates,manifest_directory)
   # Atomic host publication: no native query sees a partially registered name.
   try:self.register_snapshot(new_name,path)
   except BaseException:
    shutil.rmtree(path.parent)
    raise
   self._snapshot_lineage[new_name]=self._snapshot_lineage[base_name]
   receipt['snapshot_lineage']=self._snapshot_lineage[new_name]
   receipt['registered_host_payload']=self._registered_host_payload()
   receipt['commit_seconds']=perf_counter()-begin;receipt['snapshot_sha256']=self.snapshot_hashes[new_name]
   return receipt
 def retain_snapshot(self,name):
  """Protect existing views from this wrapper's proactive release (no raw pointer)."""
  with self._lock:
   if self.closed or name not in self.graphs:raise ValueError('Unknown live snapshot')
   if self.controller.budgets is not None:raise ValueError('Explicit retention qualification requires no automatic byte budget')
   self._retained_snapshots.add(name)
   return dict(snapshot=name,kind='manager_retention_request',reader_scope='Existing native registry view; completed calls only. Not a concurrent kernel lease.')
 def release_snapshot(self,name):
  with self._lock:self._retained_snapshots.discard(name)
 def query(self,graph_id,algorithm,parameters=None):
  begin=perf_counter()
  with self._lock:
   if self.closed:raise RuntimeError('Session closed')
   lineage=self._snapshot_lineage[graph_id]
   if os.environ.get('EGGPU_DYNAMIC_WEIGHT_ROUTE')!=self.route:raise RuntimeError('Dynamic route changed during active session')
   self.cpp.cpp_egmgpu_reset_allocation_observation(self.devices)
   before=self.resource_inventory();release=None
   # Every route receives exactly the same semantic-retirement permission.
   # The difference is full upload vs COW scatter vs exclusive lease scatter.
   protected={self.metadata[name]['id'] for name in self._retained_snapshots}|{self.metadata[graph_id]['id']}
   retireable={self.metadata[name]['id'] for name in self.graphs
               if self.retirement_policy=='all_other_views' or self._snapshot_lineage[name]==lineage}
   owner_ids=[o['owner_id'] for row in before['per_device'] for o in row['owners']
              if o['kind']=='view' and o.get('graph_id') in retireable
              and o.get('graph_id') not in protected and not o.get('active_leases',0)]
   if owner_ids:release=self.release(owner_ids,release_idle=False,release_workspaces=False)
   result=super().query(graph_id,algorithm,parameters)
   after=self.resource_inventory();left=_dynamic_totals(before);right=_dynamic_totals(after)
   delta={d:{k:right[d].get(k,0)-left.get(d,{}).get(k,0) for k in right[d] if k not in ('abi_version','host_mirror_live_bytes','peak_packet_device_bytes')} for d in right}
   result.backend_metrics['native_dynamic_patch']=dict(route=self.route,retirement_policy=self.retirement_policy,snapshot_lineage=lineage,per_device_delta=delta,after_counters=right,release=release,
    scope='Actual native contiguous weight arrays; full WPR and SSSP kernels. Source strength recomputed by original host preparation. No cross-device shared allocation or shared-indirect execution.',
    host_diff_scope='Full host comparison unless native-generated SSSP exact-base packet applies; PR retains full host discovery. All incurred work charged inside query.',
    device_patch_scope='Sparse CUDA scatter and actual H2D/D2D bytes; retained inventory excludes transient packet allocation',
    snapshot_scope='Old host snapshots remain readable; unrelated graph families use the ordinary retention budget; released same-family GPU views may require rebuilding')
   return replace(result,seconds=perf_counter()-begin)
 def close(self):
  try:
   with self._lock:
    if not self.closed and self.cpp.cpp_egmgpu_runtime_status()['running']:
     # Shutdown ends CUDA workers; the separate process-wide PR prepared cache
     # must also release graph_owner references before registered hosts go away.
     self.closed_native_cache_receipt=self.cpp.cpp_egmgpu_clear_partitioned_state(devices=self.devices)
    super().close()
    if self.closed:
     self.closed_host_registry=self._registered_host_payload()
     for name in ('graphs','metadata','nodes','_domains','_manifest_documents','_manifest_paths','snapshot_hashes','_snapshot_lineage','_retained_snapshots'):
      getattr(self,name,{}).clear()
  finally:self._restore_environment()
