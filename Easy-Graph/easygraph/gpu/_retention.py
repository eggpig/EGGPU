"""P1/P2 native view lifecycle control on actual per-device allocations.

The retention budget applies after a completed query, not to CUDA context,
unmanaged communication storage, or transient construction peaks. Old native
kernels and host plans remain in use. No future request information is read.
"""
from __future__ import annotations
from dataclasses import replace
import copy
from time import perf_counter
import numpy as np

from easygraph.gpu._bridge import NativePartitionSession, source_parameter
from easygraph.gpu.contracts import QueryContract


def algorithm_name(value):
    return 'pr' if value in ('pr', 'pagerank') else value


def check_inventory(snapshot, devices):
    if snapshot.get('has_untracked_shared_references', False):
        raise RuntimeError('Untracked native consumers invalidate the BFS/PR ownership contract')
    observed = snapshot['per_device']
    if sorted(int(d['device_id']) for d in observed) != sorted(devices):
        raise RuntimeError('Inventory devices differ from session devices')
    total = 0
    for device in observed:
        owners = {o['owner_id']: o for o in device['owners']}
        if len(owners) != len(device['owners']):
            raise RuntimeError('Repeated owner identity')
        seen = set(); size = 0
        for allocation in device['allocations']:
            aid = allocation['allocation_id']
            if aid in seen or allocation['device_id'] != device['device_id']:
                raise RuntimeError('Allocation duplicated or charged to wrong device')
            seen.add(aid)
            if allocation['bytes'] < 0 or not set(allocation['owners']) <= set(owners):
                raise RuntimeError('Invalid allocation size/owner reference')
            size += allocation['bytes']
        if size != device['managed_resident_bytes']:
            raise RuntimeError('Unique allocation union differs from recorded bytes')
        total += size
    if total != snapshot['managed_resident_bytes']:
        raise RuntimeError('Per-device allocation totals disagree')
    return snapshot


class BoundaryRetentionController:
    """Same current-call view-LRU with scratch fallback for both workspace policies.

scratch_first is a stronger alternative: release idle scratch before evicting
long-lived graph representations. All eviction decisions use current inventory.
"""
    def __init__(self, cpp, devices, budgets=None, retention='lru'):
        if retention not in ('lru', 'scratch_first'):
            raise ValueError('Unknown retention policy')
        self.cpp = cpp; self.devices = list(devices)
        self.budgets = None if budgets is None else {int(d):int(v) for d,v in budgets.items()}
        if self.budgets is not None and (set(self.budgets) != set(devices) or min(self.budgets.values()) <= 0):
            raise ValueError('A positive boundary budget is required on every device')
        self.retention = retention; self.clock = 0; self.last_use = {}

    def inventory(self):
        return check_inventory(self.cpp.cpp_egmgpu_managed_state_inventory(devices=self.devices), self.devices)

    def evict(self, owner_ids, devices, *, release_idle=True, release_workspaces=False):
        return self.cpp.cpp_egmgpu_managed_evict(owner_ids=list(owner_ids), devices=list(devices),
                                              release_idle=release_idle, release_workspaces=release_workspaces)

    def finish_query(self, graph_id, algorithm):
        begin=perf_counter(); self.clock += 1
        initial=self.inventory(); state=initial; actions=[]
        algorithm=algorithm_name(algorithm)
        for d in state['per_device']:
            for o in d['owners']:
                same_algorithm=algorithm_name(o.get('algorithm')) == algorithm
                if same_algorithm and (o.get('graph_id') == graph_id or
                                      (o.get('kind')=='workspace' and not o.get('graph_id'))):
                    self.last_use[o['owner_id']]=self.clock
        if self.budgets is not None:
            for device in self.devices:
                cap=self.budgets[device]
                row=next(d for d in state['per_device'] if d['device_id']==device)
                if row['managed_resident_bytes'] <= cap: continue
                # A directory-only owner is not a running query: trim idle arrays
                # first, without invalidating any live representation or host plan.
                action=self.evict([], [device]); actions.append({'reason':'idle_catalog','device':device,'receipt':action})
                state=self.inventory(); attempted=set()
                while True:
                    row=next(d for d in state['per_device'] if d['device_id']==device)
                    if row['managed_resident_bytes'] <= cap: break
                    candidates=[]
                    for o in row['owners']:
                        if o['owner_id'] in attempted or o.get('active_leases',0): continue
                        if o['kind']=='workspace':
                            candidates.append((0 if self.retention=='scratch_first' else 2,
                                               self.last_use.get(o['owner_id'],0),o['owner_id']))
                        elif o['kind']=='view':
                            current=o.get('graph_id')==graph_id and algorithm_name(o.get('algorithm'))==algorithm
                            candidates.append((3 if current else 1,self.last_use.get(o['owner_id'],0),o['owner_id']))
                    if not candidates:
                        raise RuntimeError(f'No legal owner release can satisfy device {device} retained budget {cap}')
                    _,_,owner=min(candidates); attempted.add(owner)
                    before=row['managed_resident_bytes']; action=self.evict([owner],[device])
                    state=self.inventory(); after=next(d['managed_resident_bytes'] for d in state['per_device'] if d['device_id']==device)
                    actions.append({'reason':self.retention,'device':device,'owner_id':owner,
                                    'before_bytes':before,'after_bytes':after,'freed_bytes':before-after,'receipt':action})
                    if after > before:
                        raise RuntimeError('Release increased managed resident storage')
            for d in state['per_device']:
                if d['managed_resident_bytes'] > self.budgets[d['device_id']]:
                    raise RuntimeError('Final per-device retention budget violated')
        live={o['owner_id'] for d in state['per_device'] for o in d['owners']}
        self.last_use={k:v for k,v in self.last_use.items() if k in live}
        return {'before_retention':initial,'after_retention':state,'actions':actions,
                'controller_seconds':perf_counter()-begin,'retention_policy':self.retention,
                'budget_scope':'Completed-query retained BFS/PR views, workspaces and shared catalog only; not transient or total CUDA peak',
                'budgets':self.budgets}


class ManagedNativeSession(NativePartitionSession):
    """Unified owning-output session with actual P1/P2 retention control.

The native workspace policy is selected before runtime construction by the
launcher. This adapter does not mutate process environment or change kernels.
"""
    def __init__(self, cpp, manifests, *, devices=(0,), budgets=None, retention='lru'):
        for name in ('cpp_egmgpu_managed_state_inventory','cpp_egmgpu_managed_evict'):
            if not hasattr(cpp,name): raise RuntimeError('Requires native_view_management_v1 artifact: '+name)
        controller=BoundaryRetentionController(cpp,devices,budgets,retention)
        super().__init__(cpp,manifests,devices=devices)
        self.controller=controller
        self.last_resource_record=None

    def query(self, graph_id, algorithm, parameters=None):
        begin=perf_counter()
        with self._lock:
            if self.closed: raise RuntimeError('Session closed')
            parameters=dict(parameters or {})
            graph=self.graphs[graph_id]; n=int(self.metadata[graph_id]['vertex_count'])
            if algorithm=='bfs':
                source=source_parameter(parameters)
                if not 0<=source<n: raise ValueError('Source outside manifest vertex domain')
                call_begin=perf_counter()
                raw=self.cpp.cpp_egmgpu_partitioned_bfs(graph,source,devices=self.devices,dense_output=True,frontier_policy='adaptive')
                native_seconds=perf_counter()-call_begin
                values=np.asarray(raw['values_dense']); values=np.where(values<0,np.inf,values)
                contract=QueryContract('bfs',source,'unweighted',None,None,None,'+inf',None)
            elif algorithm=='pr':
                if parameters: raise ValueError('PR contract is fixed to alpha=.85, max_iter=1000, tolerance=1e-9/n')
                call_begin=perf_counter()
                raw=self.cpp.cpp_egmgpu_partitioned_pagerank(graph,.85,1000,1e-9/n,self.devices)
                native_seconds=perf_counter()-call_begin
                if not raw['converged']: raise RuntimeError('Native PageRank did not converge')
                values=raw['values_dense']; contract=QueryContract('pr',None,'unweighted',.85,1000,1e-9/n,None,None)
            else: raise NotImplementedError('Resource-managed native qualification is BFS/PR only')
            metrics={k:v for k,v in raw.items() if k not in ('values','values_dense','labels_dense')}
            metrics['native_complete_output_seconds']=native_seconds
            # Copy to an independently owned host result before any native state release.
            result=self._own_result(graph_id,algorithm,values,begin,metrics,parameters,contract=contract)
            resource=self.controller.finish_query(self.metadata[graph_id]['id'],algorithm)
            metrics['resource_management']=resource
            self.last_resource_record=resource
            return replace(result,seconds=perf_counter()-begin)

    def resource_inventory(self):
        with self._lock:
            if self.closed: raise RuntimeError('Session closed')
            return self.controller.inventory()

    def capability_descriptor(self):
        return replace(super().capability_descriptor(),
                       physical_ownership='BFS/PR retained allocations and owners on each device via resource_inventory; snapshot remains legacy aggregate',
                       control_routes=('automatic_cache_reuse','boundary_inventory','selective_view_release','idle_workspace_release','per_device_retention_budget'),
                       qualification='Native view-management cohort records P1/P2 qualification separately',
                       limitations=('Immutable graph snapshots; no dynamic updates',
                                    'Request-boundary retained-state budget, not construction or total CUDA peak',
                                    'BFS/PR only; other native consumers are outside this ownership contract',
                                    'Pointer allocation IDs are boundary identities; lifetime evidence also requires stable owner IDs',
                                    'No cross-device physical allocation sharing; old exact planner is not a gang executor'))

    def release(self, owner_ids, *, devices=None, release_idle=True, release_workspaces=False):
        with self._lock:
            if self.closed: raise RuntimeError('Session closed')
            selected=self.devices if devices is None else list(devices)
            if not selected or not set(selected)<=set(self.devices): raise ValueError('Device outside session')
            return self.controller.evict(owner_ids,selected,release_idle=release_idle,release_workspaces=release_workspaces)
