"""Admit the current request before advancing shared surviving device state.

The ordinary buffer_scoped session, journal, LRU and all movement kernels are
unchanged. No future request or per-consumer mutation chain is consulted.
"""
import time
import numpy as np
from easygraph.gpu._arena.gpu_runtime import Arena
from easygraph.gpu._arena.session_manager import request_boundary, token
from easygraph.gpu._arena.round2_topology.gpu_adapter import GraphAdapter
from easygraph.gpu._arena.round2_topology.state import Delta
from easygraph.gpu._arena.round3_transfer.packet import make_packet
from easygraph.gpu._arena.round4_switch.session import SwitchingSession


class AdmittedSwitchingSession(SwitchingSession):
    def __init__(self, inputs, budget_bytes, journal_budget_bytes, mode='admitted'):
        if mode != 'admitted': raise ValueError(mode)
        self.request_context = None
        self.admission_last = None
        super().__init__(inputs, budget_bytes, journal_budget_bytes, 'buffer_scoped')

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
                packet = make_packet(old.r2_state.outgoing,old.r2_state.incoming,state,delta,
                                     {key[1:] for key in selected.values()})
                payload = packet.nbytes; packet_builds += 1
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

    def _adopt(self, flush, allow_advance):
        context = self.request_context
        if (not allow_advance or context is None or context[0] != flush.ref.graph_id or not flush.changes
                or any(c.final_weight is None for c in flush.changes)):
            return super()._adopt(flush,allow_advance)
        started = time.perf_counter(); plan = self._plan_admission(flush)
        self.counters['admission_planning_ns'] += int((time.perf_counter()-started)*1e9)
        self.counters['admission_calls'] += 1
        self.counters['admission_packet_estimates'] += plan['packet_estimates']
        self.counters['admission_avoided_maintenance_components'] += plan['avoided_maintenance_components']
        if plan['feasible']:
            for name in plan['victims']: self._drop(name)
            self.counters['admission_early_evictions'] += len(plan['victims'])
        else: self.counters['admission_refresh_fallbacks'] += 1
        self.admission_last = plan
        detail = super()._adopt(flush,allow_advance=plan['feasible'])
        if plan['feasible']:
            # Missing requested representations fit beside every retained
            # allocation after the parent's single coalesced growth operation.
            assert self.arena.live + plan['missing_request_capacity_bytes'] <= self.arena.budget
        return dict(detail,admission=plan)

    @request_boundary
    def query(self,gid,function,attr=None,source=0,result='array',selected=None):
        started = time.perf_counter(); prior = self.request_context
        self.request_context = (gid,function,attr); self.admission_last = None
        try:
            answer = super().query(gid,function,attr,source,result,selected)
            self.last.update(admission=self.admission_last,seconds=time.perf_counter()-started)
            return answer
        finally: self.request_context = prior

    def stats(self):
        return dict(super().stats(),mode='admitted',admission_scope='current request only; common component LRU')
