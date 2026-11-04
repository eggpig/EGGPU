"""Explicit dependency loading with frozen workspace, kernels and state semantics.

No process-global replacement of CuPy or frozen constructor is performed.
First function resolution remains inside its actual calling operation. Loading
is an engineering improvement, independent of graph-state policy contributions.
"""
from collections import Counter,defaultdict
from functools import partial
import threading,weakref
import numpy as np
import cupy as cp
from easygraph.gpu._arena.gpu_runtime import Arena,Executor,CUDA_SOURCE
from easygraph.gpu._arena.lifecycle_adapters import DeferredModule
from easygraph.gpu._arena.session_manager import workspace_bytes
from easygraph.gpu._arena.state_model import StateModel
from easygraph.gpu._arena.native_bridge import NativeBFS,NativeSSSP
from easygraph.gpu._arena.factorized_graph import FactorizedGraph
from easygraph.gpu._arena.factorized_runtime import FactorizedExecutor
from easygraph.gpu._arena.factorized_session import portfolio_workspace_bytes,portfolio_request_bytes
from easygraph.gpu._arena.round2_topology.gpu_adapter import GraphAdapter
from easygraph.gpu._arena.round6_recovery.epoch_model import EpochModel
from easygraph.gpu._arena.round7_recovery.packet_cost import PacketPrice
from easygraph.gpu._arena.round8_cold.session import ColdSession
from easygraph.gpu._arena.round8_host.bulk import TypedSession
from easygraph.gpu._arena.round8_cold_fast.authority import FastColdStore
from easygraph.gpu._arena.round8_changes.authority import ChangePreparationMixin,PreparedStore,DEFAULT_CHANGE_BUDGET

KERNEL_NAMES=('init','stats','pagerank','shortest','patch','gather','degree','normalize','mask')


class LazyExecutor(Executor):
    """Executor's identical allocation layout, with explicitly supplied resolver."""
    def __init__(self,arena,nmax,patch_capacity=8192,*,module_factory=None,event_factory=None):
        self.arena,self.nmax=arena,nmax
        self.x=arena.allocate('@x',(nmax,),np.float64)
        self.y=arena.allocate('@y',(nmax,),np.float64)
        self.stat=arena.allocate('@stat',(2,),np.float64)
        self.patch_i=arena.allocate('@patch_i',(patch_capacity,),np.int32)
        self.patch_v=arena.allocate('@patch_v',(patch_capacity,),np.float64)
        self.patch_capacity=patch_capacity;self.workspace_bytes=arena.live
        self.resolutions=[]
        factory=cp.RawModule if module_factory is None else module_factory
        owner=weakref.ref(self)
        def resolved(event,seconds,**detail):
            executor=owner()
            if executor is not None:executor._resolved(event,seconds,**detail)
        module=DeferredModule(factory(code=CUDA_SOURCE,options=('--std=c++17',),backend='nvcc'),resolved)
        self.kernels={name:module.get_function(name) for name in KERNEL_NAMES}
        event_factory=cp.cuda.Event if event_factory is None else event_factory
        self.events=(event_factory(),event_factory());self.result_events=(event_factory(),event_factory())

    def _resolved(self,event,seconds,**detail):
        self.resolutions.append(dict(event=event,seconds=seconds,**detail))


def initialize_representation(session,graphs,budget_bytes,lazy_module):
    """Frozen native_paths/factorized/host/greedy configuration, explicit Executor.

Mirrors GraphSession, ConditionalGraphSession and RepresentationSession
constructor fields. All operation, growth and reclamation methods are inherited.
"""
    graphs=[FactorizedGraph.wrap(g) for g in graphs]
    if not graphs:raise ValueError('At least one graph is required')
    nmax=max(g.n for g in graphs);backend='native_paths'
    required=portfolio_workspace_bytes(nmax,backend)+max(map(portfolio_request_bytes,graphs))
    if required>budget_bytes:raise MemoryError('One complete representation portfolio request exceeds budget')
    session.representation='factorized';session.sum_builder='host'
    session.graphs={g.name:g for g in graphs};session.policy='repair_lru'
    expected=workspace_bytes(nmax,backend)
    if expected>budget_bytes:raise MemoryError('Session workspace exceeds arena budget')
    session.arena=Arena(budget_bytes)
    session.executor=(LazyExecutor if lazy_module else Executor)(session.arena,nmax)
    session.code_executor=session.executor;session.kernel_backend=backend
    session.native_bfs=NativeBFS(session.arena,nmax);session.native_sssp=NativeSSSP(session.arena,nmax)
    session.sources=session.arena.allocate('@native_sources',(1,),np.int32)
    session.executor.workspace_bytes=session.arena.live
    if expected!=session.executor.workspace_bytes:raise AssertionError('Native workspace query disagrees with allocations')
    session.model=StateModel(budget_bytes,'shared_component_LRU')
    session.model.workspace_capacity=session.executor.workspace_bytes
    session.device_versions={};session.semantic={};session.counters=defaultdict(int)
    session.events=[];session.last={};session.lock=threading.RLock();session._log_compaction_pending=False
    session.costs=None;session.selection='greedy';session.decision_floor_seconds=.00075
    session.overhead_multiplier=2.;session.max_future_requests=4
    session.request_counts=Counter();session.decision_ewma_seconds=0.;session.last_decision=None
    session.executor=FactorizedExecutor(session.executor,backend)
    for graph in session.graphs.values():session._bind_active_keys(graph)
    session.model.assert_invariants()


class LazySession(ColdSession):
    """Public ColdSession contract, with explicit store/executor factories."""
    def __init__(self,inputs,budget_bytes,journal_budget_bytes,strategy,historical,
                 host_index_budget_bytes=0,host_producer='shared',packet_cost='online',
                 capture=False,preselected=None,*,lazy_module=True,store_factory=None):
        if not isinstance(lazy_module,bool):raise ValueError('lazy_module must be bool')
        if len({x['name'] for x in inputs})!=len(inputs):raise ValueError('Graph identifiers must be unique')
        self.strategy='ordinary';self.batch_requests=None;self.batch_index=0;self.batch_keys=()
        self.recovery_plans=[];self.profile=None;self.historical_profile=historical;self.mode='buffer_scoped'
        factory=self.store_type if store_factory is None else store_factory
        self.journal=factory(journal_budget_bytes,host_index_budget_bytes,host_producer)
        self.refs={};self.transfer_kernels=None;self.last_graph=None;self.residence={};self.logical_tick=0
        graphs=[]
        for item in inputs:
            ref=self.journal.register(item['name'],item['nodes'],item['edges'],item.get('weights'))
            self.refs[item['name']]=ref
            graphs.append(GraphAdapter(self.journal.snapshot(ref).host_state,name=item['name']))
        initialize_representation(self,graphs,budget_bytes,lazy_module)
        ledger=EpochModel(budget_bytes,'shared_component_LRU');ledger.__dict__.update(self.model.__dict__)
        ledger.epoch_owner=weakref.ref(self);self.model=ledger
        self.authority='r8';self.packet_engine='vector';self.packet_cost=packet_cost
        self.packet_price=PacketPrice();self.packet_features=None;self.packet_prediction=None
        self.strategy=strategy;self.capture_problems=capture;self.problem_records=[]
        self.preselected={} if preselected is None else preselected;self.planner_context=None
        self.lazy_module=lazy_module

    def stats(self):
        resolutions=getattr(self.code_executor,'resolutions',[])
        return dict(super().stats(),kernel_loading=dict(lazy=self.lazy_module,
            resolved_functions=[x['function'] for x in resolutions],resolution_count=len(resolutions),
            resolution_seconds=sum(x['seconds'] for x in resolutions),records=list(resolutions),
            scope='Resolution is charged inside first actual operation; all complete call/session timers include it'))

    def close(self):
        try:return super().close()
        finally:self.code_executor=None


class LazyTypedSession(LazySession,TypedSession):
    pass


class FastPreparedStore(ChangePreparationMixin,FastColdStore):
    pass


class FinalSession(LazySession):
    """Composable full candidate with orthogonal fast-input/lazy-loader switches."""
    def __init__(self,*args,fast_input=True,lazy_module=True,
                 change_budget_bytes=DEFAULT_CHANGE_BUDGET,**kwargs):
        if not isinstance(fast_input,bool):raise ValueError('fast_input must be bool')
        store=FastPreparedStore if fast_input else PreparedStore
        super().__init__(*args,lazy_module=lazy_module,
                         store_factory=partial(store,change_budget_bytes=change_budget_bytes),**kwargs)
        self.fast_input=fast_input

    def stats(self):
        host=self.journal.stats()
        return dict(super().stats(),fast_input=self.fast_input,prepared_changes=host['prepared_changes'],
                    prepared_host_union=host['prepared_host_union'])

    def close(self):
        try:return super().close()
        finally:self.journal.change_cache.clear('session_close')


class FinalTypedSession(FinalSession,TypedSession):
    pass
