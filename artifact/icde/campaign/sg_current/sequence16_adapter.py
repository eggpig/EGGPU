"""Fixed public-API mappings for a continuous four-function SG request sequence."""
from pathlib import Path
import sys
import sg_worker as shared


class EGGPUSequence:
    def __init__(self,args,graph):
        import numpy as np
        sys.path.insert(0,str(args.source.resolve()));sys.path.insert(1,str(args.native_dso.resolve().parent))
        from easygraph.gpu import Session
        if Path(sys.modules[Session.__module__].__file__).resolve()!=(args.source/'easygraph/gpu/session.py').resolve():
            raise ValueError('Actual public Session source differs from --source')
        cpp=sys.modules.get('cpp_easygraph')
        if cpp is not None and Path(cpp.__file__).resolve()!=args.native_dso.resolve():raise ValueError('Wrong native extension')
        payload={'nodes':np.arange(graph['n'],dtype=np.int64),'offsets':graph['offsets'],
                 'indices':graph['indices'],'weights':graph['weights']}
        self.session=Session.create({'g':payload},devices=(0,),state_model='static',input_format='csr',
            algorithms=shared.ALGORITHMS,budget_bytes=args.budget_mib<<20,
            pr_tolerance_l1=shared.PR_TOLERANCE,pr_max_iter=shared.PR_MAX_ITER)
        self.contract={'system':'EGGPU','public_api':'Session.create(input_format=csr)',
            'dispatch':self.session.dispatch_plan.to_dict(),'graph_objects':1,'result_cache_added_by_runner':False}
    def query(self,request):
        params={'source':request['source']} if request['algorithm'] in ('bfs','sssp') else {}
        return self.session.query('g',request['algorithm'],params).values
    def close(self):self.session.close()


class CuGraphSequence:
    """Retain legal author SGGraph state; no new algorithm or result memoization.

    Official pylibcugraph BFS example accepts an SGGraph with weight_array.
    BFS and SSSP therefore share outgoing weighted CSR; BFS ignores edge costs.
    WCC needs weak topology; PR needs unit weights + transposed storage.
    """
    def __init__(self,args,graph):
        self.graph=graph;self.backends={};self.contract={'system':'cuGraph',
            'graph_mapping':{'bfs':'outgoing_weighted','sssp':'outgoing_weighted',
                             'wcc':'weak_unweighted','pr':'transposed_unit'},
            'lazy_preparation':'included in first request using that representation',
            'result_cache_added_by_runner':False,'pr_initial_guess':None,
            'bfs_weighted_graph_evidence':'https://docs.nvidia.com/cugraph/latest/api_docs/api/plc/pylibcugraph.bfs/'}
    def query(self,request):
        import numpy as np
        from cugraph_csr_adapter import CuGraphCSR
        algorithm=request['algorithm'];source=request['source'] or 0
        key=self.contract['graph_mapping'][algorithm]
        if key not in self.backends:
            graph=self.graph
            if algorithm=='wcc' and graph['directed']:
                matrix=shared.graph_matrix(graph);weak=matrix.maximum(matrix.T).tocsr();weak.sort_indices()
                from cugraph_csr_adapter import weak_csr_graph
                graph=weak_csr_graph(graph,weak)
            self.backends[key]=CuGraphCSR(graph,'sssp' if algorithm in ('bfs','sssp') else algorithm,source=source)
        backend=self.backends[key];backend.function=algorithm;backend.source=source
        if algorithm=='bfs':backend.sources=backend.cp.asarray([source],dtype=backend.index_dtype)
        result=backend.complete_query()
        # The bound SG adapter calls cp.asnumpy for each fresh native result;
        # it never retains a host output reference. Preserve that owned array.
        if not result.flags.owndata or not result.flags.c_contiguous:result=np.array(result,copy=True,order='C')
        result.flags.writeable=False
        return result
    def close(self):
        for backend in self.backends.values():backend.close()
        self.contract['constructed_graph_objects']=len(self.backends)
        self.backends.clear()


def build(args,graph):
    return EGGPUSequence(args,graph) if args.system=='eggpu' else CuGraphSequence(args,graph)
