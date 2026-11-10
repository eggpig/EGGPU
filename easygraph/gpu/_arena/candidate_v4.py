"""Final kernels with demand-driven host state over owned canonical CSR.

Admission owns and validates its input. Reverse topology, edge-authority indexes,
and weight summaries are materialized on demand; mutations retain the qualified
typed journal and the same dependency/version semantics. Arena semantics are
unchanged, so code warmup does not introduce a new retained allocation pool.
"""
from easygraph.gpu._arena.trace_backends import EGGPUAdapter


class DemandCandidateAdapter(EGGPUAdapter):
    def __init__(self,data,*,budget_bytes,**kwargs):
        from easygraph.gpu._arena.input_hotpath.csr_session_v4 import DemandCSRSession
        from easygraph.gpu._arena.kernel_sssp_v4 import SparseCooperativeSSSP
        from easygraph.gpu._arena.public_session_release import PublicExecutor
        from easygraph.gpu._arena.connectivity import connectivity_session,ConnectivityExecutor
        from easygraph.gpu._arena.bulk_assign import assignment_session
        from easygraph.gpu._arena.kernel_pr_v4 import HybridRowsExecutor
        self.nodes={k:v['nodes'] for k,v in data.items()}
        cls=assignment_session(connectivity_session(DemandCSRSession))
        inputs=[dict(name=k,nodes=v['nodes'],edges=v['edges'],weights=v['weights']) for k,v in data.items()]
        self.session=cls(inputs,budget_bytes,4<<20,'ordinary',None,host_producer='shared',
            host_index_budget_bytes=0,change_budget_bytes=8<<20)
        self.session.executor=ConnectivityExecutor(HybridRowsExecutor(PublicExecutor(self.session.executor.base)))
        self.session.native_sssp=SparseCooperativeSSSP(self.session.native_sssp)
        self.candidate=True;self.last={}
