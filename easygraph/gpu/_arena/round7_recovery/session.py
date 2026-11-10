"""R7 integration on immutable R6 execution primitives.

The diagnostic preselected policy receives an offline trace of chosen subsets.
It remains a labelled headroom experiment, never an online performance claim.
"""
import time

from easygraph.gpu._arena.round5_input.array_session_v2 import ArraySwitchingSession
from easygraph.gpu._arena.round6_recovery.session_v2 import RecoverySession
from easygraph.gpu._arena.round6_recovery.planner_v2 import evaluate_subset, plan as reference_plan
from easygraph.gpu._arena.round7_recovery.transfer import VectorTransferMixin
from easygraph.gpu._arena.round7_recovery.packet_cost import PacketPrice


class PlannedRecoverySession(VectorTransferMixin, RecoverySession):
    def __init__(self, inputs, budget_bytes, journal_budget_bytes, strategy, historical,
                 capture=False, preselected=None, authority='array',packet_engine='legacy',packet_cost='historical'):
        if authority=='vector2':
            from easygraph.gpu._arena.round7_recovery.initialization import initialize
            initialize(self,inputs,budget_bytes,journal_budget_bytes,historical)
        else:super().__init__(inputs, budget_bytes, journal_budget_bytes, 'ordinary', historical)
        if authority=='vector':
            from easygraph.gpu._arena.round7_recovery.host_authority import VectorJournalStore
            self.journal=VectorJournalStore.from_store(self.journal)
        elif authority not in ('array','vector2'):raise ValueError(authority)
        self.authority=authority
        if packet_engine not in ('legacy','vector'):raise ValueError(packet_engine)
        self.packet_engine=packet_engine
        if packet_cost not in ('historical','online'):raise ValueError(packet_cost)
        self.packet_cost=packet_cost
        self.packet_price=PacketPrice()
        self.packet_features=None
        self.packet_prediction=None
        self.strategy = strategy
        self.capture_problems = capture
        self.problem_records = []
        self.preselected = {} if preselected is None else preselected
        self.planner_context = None

    def _problem(self, flush):
        # Cost-based R7 strategies receive the identical charged R6 calibration.
        saved = self.strategy
        try:
            self.strategy = saved if saved in ('full_refresh','all_advance') else 'exhaustive'
            problem = super()._problem(flush)
        finally:
            self.strategy = saved
        self.packet_features=self.packet_price.features(self.graphs[flush.ref.graph_id].r2_state.m,flush.changes)
        self.packet_prediction=None
        if self.packet_cost=='online' and saved!='adaptive_fixed':
            problem,self.packet_prediction=self.packet_price.price(problem,self.packet_features)
        if self.capture_problems:
            from easygraph.gpu._arena.round7_recovery.problem_io import encode
            slots = {name:dict(offset=int(slot.offset), capacity=int(slot.capacity))
                     for name,slot in self.arena.slots.items()}
            self.problem_records.append(encode(problem, slots, graph=flush.ref.graph_id,
                version=flush.new_snapshot.version, batch_index=self.batch_index))
        return problem

    def _adopt(self, flush, allow_advance):
        if self.strategy == 'ordinary' or not allow_advance or self.batch_requests is None:
            return ArraySwitchingSession._adopt(self, flush, allow_advance)
        started=time.perf_counter()
        problem=self._problem(flush)
        if self.strategy == 'preselected':
            key=(flush.ref.graph_id, int(flush.new_snapshot.version), self.batch_index)
            if key not in self.preselected: raise AssertionError(('No preregistered subset', key))
            replay_started=time.perf_counter()
            chosen=evaluate_subset(problem, self.preselected[key], trace=True)
            chosen.update(strategy='preselected',evaluations=1,simulation_passes=1,
                candidate_menu_size=1<<len(problem.eligible_names),pruned_candidates=[],
                candidate_summary=[],planner_seconds=time.perf_counter()-replay_started,
                visibility='offline chosen-subset trace; diagnostic headroom only')
        elif self.strategy in ('adaptive','adaptive_fixed','bounded','certified','previous'):
            from easygraph.gpu._arena.round7_recovery.planner import RecoveryContext, plan
            if self.planner_context is None: self.planner_context=RecoveryContext()
            solver='adaptive' if self.strategy=='adaptive_fixed' else self.strategy
            chosen=plan(problem, strategy=solver, context=self.planner_context)
            chosen.update(strategy=self.strategy,solver_strategy=solver)
        else:
            chosen=reference_plan(problem, self.strategy)
        if not chosen['feasible']: raise MemoryError(chosen['reason'])
        for name in chosen['retired_stale']+chosen['maintenance_victims']:
            if name in self.model.components:self._drop(name)
        elapsed=time.perf_counter()-started
        self.counters['recovery_planning_ns']+=int(elapsed*1e9)
        self.counters['recovery_plan_calls']+=1
        self.counters['recovery_candidates']+=chosen['evaluations']
        self.counters['recovery_components_avoided']+=len(problem.eligible_names)-len(chosen['selected'])
        chosen.update(counter_before_h2d=self.counters['h2d_bytes'],
            counter_before_builds=self.counters['view_builds'],batch_index=self.batch_index)
        detail=ArraySwitchingSession._adopt(self,flush,bool(chosen['selected']))
        if self.packet_cost=='online':self.packet_price.observe(self.packet_features,detail)
        if chosen['selected'] and detail['path']!='incremental':raise AssertionError('Selected action not executed')
        if chosen['selected'] and detail['packet_bytes']!=chosen['packet_payload_bytes']:raise AssertionError('Packet size mismatch')
        chosen.update(graph=flush.ref.graph_id,logical_version=flush.new_snapshot.version,
            complete_planning_seconds=elapsed,actual_maintenance=detail,packet_price_prediction=self.packet_prediction)
        if self.capture_problems:self.problem_records[-1]['chosen']=chosen
        self.recovery_plans.append(chosen)
        return dict(detail,recovery_plan_index=len(self.recovery_plans)-1,
                    recovery_planning_seconds=elapsed,selected_components=len(chosen['selected']))

    def stats(self):
        return dict(super().stats(),authority_implementation=self.authority+'_journal_r7',packet_engine=self.packet_engine,
                    packet_cost=self.packet_cost,packet_cost_observations={k:len(v) for k,v in self.packet_price.samples.items()})
