"""Explicit dependency injection for the qualified vector journal constructor."""
import weakref
from easygraph.gpu._arena.factorized_session import RepresentationSession
from easygraph.gpu._arena.round2_topology.gpu_adapter import GraphAdapter
from easygraph.gpu._arena.round6_recovery.epoch_model import EpochModel
from easygraph.gpu._arena.round7_recovery.host_authority_v2 import VectorJournalStore

def initialize(self,inputs,budget_bytes,journal_budget_bytes,historical):
    if len({item['name'] for item in inputs})!=len(inputs):raise ValueError('Graph identifiers must be unique')
    self.strategy='ordinary';self.batch_requests=None;self.batch_index=0;self.batch_keys=()
    self.recovery_plans=[];self.profile=None;self.historical_profile=historical
    self.mode='buffer_scoped';self.journal=VectorJournalStore(journal_budget_bytes)
    self.refs={};self.transfer_kernels=None;self.last_graph=None;self.residence={};self.logical_tick=0
    graphs=[]
    for item in inputs:
        ref=self.journal.register(item['name'],item['nodes'],item['edges'],item.get('weights'))
        self.refs[item['name']]=ref
        graphs.append(GraphAdapter(self.journal.snapshot(ref).host_state,name=item['name']))
    RepresentationSession.__init__(self,graphs,budget_bytes,costs=None,representation='factorized',
        sum_builder='host',selection='greedy',kernel_backend='native_paths')
    ledger=EpochModel(budget_bytes,'shared_component_LRU');ledger.__dict__.update(self.model.__dict__)
    ledger.epoch_owner=weakref.ref(self);self.model=ledger
