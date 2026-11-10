"""Exact scalar packet admission; true packet remains one parent construction.

V1 sources and results are frozen. This subclass changes only the planner's
byte estimate, preserving all LRU decisions, transfer kernels and graph results.
"""
import numpy as np
from easygraph.gpu._arena.gpu_runtime import Arena
from easygraph.gpu._arena.session_manager import token
from easygraph.gpu._arena.round2_topology.gpu_adapter import GraphAdapter
from easygraph.gpu._arena.round2_topology.state import Delta
from easygraph.gpu._arena.round4_switch.admitted_session import AdmittedSwitchingSession


def packet_payload_bytes(kinds, *, new_count, change_count, changed_source_rows):
    """Mirror make_packet field order and its eight-byte starting alignment."""
    if not (0 <= new_count <= change_count and 0 <= changed_source_rows <= change_count):
        raise ValueError('Invalid mutation counts')
    lengths = []
    for direction in ('out', 'in'):
        relevant = {'out_row','out_col'} if direction == 'out' else {'row','col','weight'}
        if new_count and any(kind in relevant for kind, attr in kinds):
            lengths.extend([new_count*4]*3)  # before, values, rows: int32
    if ('weight','message_count') in kinds and change_count:
        lengths.extend((change_count*4,change_count*8))
    for attr in (None,'message_count'):
        if ('source_sum',attr) in kinds and (attr is not None or new_count) and changed_source_rows:
            lengths.extend((changed_source_rows*4,changed_source_rows*8))
    size = 0
    for length in lengths:
        size = (size+7)//8*8 + length
    return size


class AdmittedSwitchingSessionV2(AdmittedSwitchingSession):
    def __init__(self, inputs, budget_bytes, journal_budget_bytes, mode='admitted_v2'):
        if mode != 'admitted_v2': raise ValueError(mode)
        super().__init__(inputs,budget_bytes,journal_budget_bytes,'admitted')

    def _plan_admission(self, flush):
        gid, function, attr = self.request_context
        old = self.graphs[gid]; state = flush.new_snapshot.host_state; changes = flush.changes
        topology = any(c.base_weight is None or c.final_weight is None for c in changes)
        graph = GraphAdapter(state, name=gid)
        keys = self._requirements(graph, function, attr)
        requested = {token(key): key for key in keys}
        protected = set(requested)
        allowed = {'row','col','out_row','out_col','weight','source_sum'}
        selected = {name: self.semantic[name] for name in self._graph_components(gid)
                    if self.model.valid(name) and self.semantic[name][1] in allowed
                    and self.model.allocations[self.model.components[name].allocation_id].owners == {name}
                    and (topology or self.semantic[name][1:] in
                         (('weight','message_count'),('source_sum','message_count')))}
        initial_selected = set(selected)
        nodes = state.nodes
        src = np.searchsorted(nodes,[c.src for c in changes]).astype(np.int32)
        dst = np.searchsorted(nodes,[c.dst for c in changes]).astype(np.int32)
        delta = Delta(src,dst,np.arange(len(changes),dtype=np.int32),
                      np.asarray([c.final_weight for c in changes],np.float64),
                      np.asarray([c.base_weight is None for c in changes],bool),np.unique(src))
        work = self.model._planning_copy(); victims = []; packet_builds = 0
        if work.policy != 'shared_component_LRU': raise AssertionError('Common LRU policy changed')
        def remove(name):
            aid = work.components[name].allocation_id; work.retire(name)
            if work.allocations[aid].pending_release: work.complete_release(aid)
            victims.append(name); selected.pop(name,None)
        # A required stale/non-advanceable input is rebuilt by the same ordinary
        # planner. Retire it before estimating space, as that planner would.
        for name,key in requested.items():
            affected = topology or (key[1] in ('weight','out_weight','source_sum') and key[2] is not None)
            if name in work.components and name not in selected and (affected or not work.valid(name)):
                remove(name)
        def capacity(key):
            kind = key[1]
            shape = (graph.n,) if kind == 'source_sum' else (graph.n+1,) if kind in ('row','out_row') else (state.m,)
            dtype = np.int32 if kind in ('row','out_row','col','out_col') else np.float64
            return Arena.capacity(shape,dtype)
        def requirement():
            nonlocal packet_builds
            if selected:
                payload = packet_payload_bytes({key[1:] for key in selected.values()},
                    new_count=int(delta.new.sum()), change_count=len(delta.ids),
                    changed_source_rows=len(delta.changed_rows))
                packet_builds += 1
                reserve = Arena.capacity((payload,),np.uint8) if payload else 0
            else: payload = reserve = 0
            growth = sum(max(0,capacity(key)-work.allocations[work.components[name].allocation_id].capacity_bytes)
                         for name,key in selected.items() if key[1] in ('col','out_col','weight'))
            missing = sum(capacity(key) for name,key in requested.items() if name not in work.components)
            return reserve+growth+missing, dict(packet_payload_bytes=payload,packet_capacity_bytes=reserve,
                       growth_capacity_bytes=growth,missing_request_capacity_bytes=missing)
        while True:
            extra,detail = requirement()
            if work.live_bytes+extra <= self.arena.budget: break
            groups = work._victim_groups(protected)
            if not groups:
                return dict(feasible=False,path='ordinary_refresh_packet_does_not_fit',victims=[],
                            planned_victims=victims,selected_components=0,packet_estimates=packet_builds,
                            avoided_maintenance_components=0,**detail)
            group = min(groups,key=lambda group:(max(work.components[k].last_access for k in group),
                        -work.marginal_release_bytes(group),sorted(group)))
            for name in sorted(group): remove(name)
        return dict(feasible=True,path='current_request_admitted',victims=victims,
                    selected_components=len(selected),selected_names=sorted(selected),
                    avoided_maintenance_components=len(initial_selected-set(selected)),
                    packet_estimates=packet_builds,predicted_joint_capacity_bytes=work.live_bytes+extra,**detail)

    def stats(self):
        return dict(super().stats(),mode='admitted_v2',packet_budget_estimation='Exact field sizes and alignment; no candidate packet materialization')
