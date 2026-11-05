"""Deferred-import SG adapter. CPU-tested contracts; GPU qualification pending.

Use inside the primary driver's existing GPU identity/occupancy/frozen-plan gates.
The constructor is preparation; complete_query includes D2H and canonical assembly.
"""
import numpy as np

ADAPTER_ORIGIN = 'experiments/rtx4090d_20261031/performance_revision_20261102/large_panel_v5/cugraph_csr_adapter.py'
ADAPTER_ORIGIN_SHA256 = '57afa35e4336a02d4ebeb0386929a664a8cbe1874ac6ae5163856238a32a01cd'
# Campaign revision: accept reciprocal undirected inputs for every operator;
# PR uses the frozen 2000-iteration cap. All conversion remains inside T_first.


SGGRAPH_SOURCE = 'https://raw.githubusercontent.com/rapidsai/cugraph/main/python/pylibcugraph/pylibcugraph/graphs.pyx'
WCC_SOURCE = 'https://docs.rapids.ai/api/cugraph/stable/api_docs/api/plc/pylibcugraph.weakly_connected_components/'


def ordered_output(ids, raw, n, function):
    ids, raw = np.asarray(ids), np.asarray(raw)
    if ids.shape != (n,) or raw.shape != (n,) or ids.dtype.kind not in 'iu':
        raise ValueError('Incomplete native output domain')
    ordered = all(np.array_equal(ids[first:first + (1 << 20)], np.arange(first, min(n, first + (1 << 20))))
                  for first in range(0, n, 1 << 20))
    if ordered:
        values = raw if raw.flags.owndata and raw.flags.c_contiguous else np.array(raw, copy=True, order='C')
    else:
        order = np.argsort(ids)
        if not np.array_equal(ids[order], np.arange(n)):
            raise ValueError('Missing/duplicated canonical vertex in output')
        values = raw[order]
    if function == 'wcc':
        # The common task asks for component partition equivalence. Native labels
        # are sufficient; audit-only canonicalization must not inflate call time.
        return values
    if function == 'pr':
        if raw.dtype != np.float64: raise ValueError('PageRank did not compute FP64')
        return values
    if function == 'bfs':
        if raw.dtype.kind != 'i': raise ValueError('BFS did not return integral distances')
        unreachable = values == np.iinfo(raw.dtype).max
    elif function == 'sssp':
        if raw.dtype != np.float64: raise ValueError('SSSP did not compute FP64')
        unreachable = values == np.finfo(raw.dtype).max
    else:
        raise ValueError('Unsupported function')
    result = values if values.dtype == np.float64 else values.astype(np.float64)
    result[unreachable] = np.inf
    return result


class CuGraphCSR:
    def __init__(self, graph, function, source=0, cp=None, plc=None):
        if cp is None:
            import cupy as cp
        if plc is None:
            import pylibcugraph as plc
        if function not in ('bfs', 'pr', 'sssp', 'wcc'): raise ValueError('Unsupported function')
        if not isinstance(graph['directed'], bool):
            raise ValueError('directed must be an explicit boolean')
        if function == 'wcc' and graph['directed']:
            raise ValueError('WCC needs explicit weak CSR')
        if not 0 <= source < graph['n']: raise ValueError('Source outside canonical domain')
        self.cp, self.plc, self.function, self.n, self.source = cp, plc, function, graph['n'], source
        self.handle = plc.ResourceHandle()
        self.offsets, self.indices = cp.asarray(graph['offsets']), cp.asarray(graph['indices'])
        self.weights = (cp.asarray(graph['weights']) if function == 'sssp' else
                        cp.ones(graph['m'], dtype=cp.float64) if function == 'pr' else None)
        self.graph = plc.SGGraph(self.handle, plc.GraphProperties(is_symmetric=not graph['directed'], is_multigraph=False),
                                self.offsets, self.indices, weight_array=self.weights,
                                input_array_format='CSR', store_transposed=function == 'pr', renumber=False,
                                do_expensive_check=False, symmetrize=False)
        self.vertices = cp.arange(self.n, dtype=cp.int32) if function == 'pr' else None
        self.outdegree = cp.asarray(np.diff(graph['offsets']), dtype=cp.float64) if function == 'pr' else None
        self.sources = cp.asarray([source], dtype=cp.int32) if function == 'bfs' else None
        cp.cuda.runtime.deviceSynchronize()
        self.contract = dict(status='GPU_QUALIFICATION_REQUIRED', backend='cuGraph', gpu_count=1,
                             input_array_format='CSR', topology_dtype='int32', vertices=self.n,
                             store_transposed=function == 'pr', complete_output=True,
                             vertices_domain='CSR offset length n+1; every output ID checked',
                             weight_semantics='canonical f64' if function == 'sssp' else 'unweighted')
        if function == 'sssp':
            self.contract['sssp_predecessors'] = dict(compute_predecessors=True,
                reason='Installed release requires True; confirmed by actual v3 tiny call rejection',
                returned_to_caller=False, timing='Required native predecessor computation remains inside complete-call timing; only distances are transferred to host')

    def complete_query(self):
        cp, plc = self.cp, self.plc
        if self.function == 'bfs':
            raw, _, ids = plc.bfs(self.handle, graph=self.graph, sources=self.sources,
                                 direction_optimizing=False, depth_limit=0,
                                 compute_predecessors=False, do_expensive_check=False)
        elif self.function == 'pr':
            ids, raw = plc.pagerank(self.handle, graph=self.graph,
                                   precomputed_vertex_out_weight_vertices=self.vertices,
                                   precomputed_vertex_out_weight_sums=self.outdegree,
                                   initial_guess_vertices=None, initial_guess_values=None,
                                   alpha=.85, epsilon=1e-9 / self.n, max_iterations=2000,
                                   do_expensive_check=False, fail_on_nonconvergence=True)
        elif self.function == 'sssp':
            ids, raw, _ = plc.sssp(self.handle, self.graph, source=self.source, cutoff=float('inf'),
                                  compute_predecessors=True, do_expensive_check=False)
        else:
            ids, raw = plc.weakly_connected_components(self.handle, self.graph, None, None, None, None, False)
        output = ordered_output(cp.asnumpy(ids), cp.asnumpy(raw), self.n, self.function)
        cp.cuda.runtime.deviceSynchronize()
        return output

    def close(self):
        self.graph = self.offsets = self.indices = self.weights = None
        self.vertices = self.outdegree = self.sources = self.handle = None
