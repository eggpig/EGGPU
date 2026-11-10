"""Single-card adapter used by the owned-CSR executor."""
import time
import numpy as np
class EGGPUAdapter:
    def __init__(self,data,*,candidate=False,budget_bytes=1024<<20,**kwargs):
        from easygraph.gpu._arena.replacement import ReplacementSession
        from easygraph.gpu._arena.public_session_release import PublicExecutor
        from easygraph.gpu._arena.connectivity import connectivity_session, ConnectivityExecutor
        from easygraph.gpu._arena.bulk_assign import assignment_session
        self.nodes={k:v['nodes'] for k,v in data.items()}
        cls=connectivity_session(ReplacementSession)
        if candidate:cls=assignment_session(cls)
        self.candidate=candidate
        inputs=[dict(name=k,nodes=v['nodes'],edges=v['edges'],weights=v['weights']) for k,v in data.items()]
        self.session=cls(inputs,budget_bytes,4<<20,'ordinary',None,host_producer='shared',
                         host_index_budget_bytes=0,change_budget_bytes=8<<20)
        core=PublicExecutor(self.session.executor.base)
        if candidate:
            from easygraph.gpu._arena.low_degree import LowDegreeExecutor
            core=LowDegreeExecutor(core)
        self.session.executor=ConnectivityExecutor(core)
        self.last={}

    def query(self,gid,algorithm,parameters):
        function='pr' if algorithm=='weighted_pr' else algorithm
        attr='message_count' if algorithm in ('weighted_pr','sssp') else None
        source=int(np.searchsorted(self.nodes[gid],parameters['source'])) if algorithm in ('bfs','sssp') else 0
        self.session.executor.pr=(.85,getattr(self,'pr_max_iter',10000),
                                  getattr(self,'pr_tolerance_l1',1e-6)/max(1,len(self.nodes[gid])))
        value=self.session.query(gid,function,attr,source)
        self.last=dict(self.session.last);return value

    def update(self,gid,batch):
        from easygraph.gpu._arena.round4_switch.journal import Mutation
        if self.candidate and not len(batch.get('delete',[])):
            pieces=[batch[k] for k in ('set','insert') if len(batch.get(k,[]))]
            if not pieces:return None
            values=np.concatenate(pieces)
            return self.session.mutate_set_edges(gid,values[:,:2].astype(np.int64),values[:,2])
        operations=[]
        for kind in ('delete','set','insert'):
            for row in batch.get(kind,[]):
                operations.append(Mutation(kind,int(row[0]),int(row[1]),float(row[2]) if len(row)>2 else None))
        return self.session.mutate(gid,operations)

    def stats(self):return self.session.stats()
    def close(self):self.session.close()
