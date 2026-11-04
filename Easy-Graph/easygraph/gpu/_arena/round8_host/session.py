"""R8 explicit host-producer injection; R7 device execution remains unchanged."""
import weakref
from easygraph.gpu._arena.factorized_session import RepresentationSession
from easygraph.gpu._arena.round2_topology.gpu_adapter import GraphAdapter
from easygraph.gpu._arena.round6_recovery.epoch_model import EpochModel
from easygraph.gpu._arena.round7_recovery.session import PlannedRecoverySession
from easygraph.gpu._arena.round7_recovery.packet_cost import PacketPrice
from easygraph.gpu._arena.round8_host.host_authority import Store
from easygraph.gpu._arena.round8_host.transfer import VectorTransferMixin


class Session(VectorTransferMixin,PlannedRecoverySession):
    store_type=Store
    def __init__(self,inputs,budget_bytes,journal_budget_bytes,strategy,historical,
                 host_index_budget_bytes=0,host_producer='shared',packet_cost='online',
                 capture=False,preselected=None):
        if len({x['name'] for x in inputs})!=len(inputs):raise ValueError('Graph identifiers must be unique')
        self.strategy='ordinary';self.batch_requests=None;self.batch_index=0;self.batch_keys=()
        self.recovery_plans=[];self.profile=None;self.historical_profile=historical
        self.mode='buffer_scoped';self.journal=self.store_type(journal_budget_bytes,host_index_budget_bytes,host_producer)
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
        self.authority='r8';self.packet_engine='vector';self.packet_cost=packet_cost
        self.packet_price=PacketPrice();self.packet_features=None;self.packet_prediction=None
        self.strategy=strategy;self.capture_problems=capture;self.problem_records=[]
        self.preselected={} if preselected is None else preselected;self.planner_context=None

    def stats(self):
        return dict(super().stats(),authority_implementation='r8_shared_host_producer',
                    host_index=self.journal.index.stats(),host_certificate=self.journal.stats()['host_certificate'])

    def close(self):
        result=super().close()
        self.journal.index.clear()
        for ref in list(self.journal.certificates):self.journal._remove_certificate(ref)
        return result
