"""R4 execution and maintenance with typed logical authority at construction."""
from easygraph.gpu._arena.factorized_session import RepresentationSession
from easygraph.gpu._arena.round2_topology.gpu_adapter import GraphAdapter
from easygraph.gpu._arena.round4_switch.session import SwitchingSession, MODES
from easygraph.gpu._arena.round5_input.array_journal_v2 import ArrayJournalStore


class ArraySwitchingSession(SwitchingSession):
    def __init__(self, inputs, budget_bytes, journal_budget_bytes, mode):
        if mode not in MODES:
            raise ValueError(mode)
        if len({item['name'] for item in inputs}) != len(inputs):
            raise ValueError('Graph identifiers must be unique')
        self.mode = mode
        self.journal = ArrayJournalStore(journal_budget_bytes)
        self.refs = {}
        self.transfer_kernels = None
        self.last_graph = None
        self.residence = {}
        self.logical_tick = 0
        graphs = []
        for item in inputs:
            ref = self.journal.register(item['name'], item['nodes'], item['edges'], item.get('weights'))
            self.refs[item['name']] = ref
            graphs.append(GraphAdapter(self.journal.snapshot(ref).host_state, name=item['name']))
        RepresentationSession.__init__(self, graphs, budget_bytes, costs=None,
            representation='factorized', sum_builder='host', selection='greedy', kernel_backend='native_paths')

    def stats(self):
        return dict(super().stats(), authority_implementation='array_journal_v2')
