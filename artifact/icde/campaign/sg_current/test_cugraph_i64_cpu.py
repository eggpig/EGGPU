"""No allocation proportional to the 2^31 boundary and no CUDA imports."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from cugraph_csr_adapter import CuGraphCSR, topology_dtype, weak_csr_graph


@pytest.mark.parametrize('n,m,expected', [(7, 9, 'int32'), (7, 2**31-1, 'int32'),
    (7, 2**31, 'int64'), (2**31, 0, 'int64')])
def test_count_boundary(n, m, expected):
    assert topology_dtype(n, m) == np.dtype(expected)


@pytest.mark.parametrize('n,m', [(-1, 1), (1, -1), (1, 2**63)])
def test_invalid_counts_rejected(n, m):
    with pytest.raises(OverflowError): topology_dtype(n, m)


def test_weak_projection_retains_full_structure_and_isolates():
    mat = sparse.csr_matrix(([1., 1., 1.], ([0, 1, 3], [1, 2, 4])), shape=(7, 7))
    weak = mat.maximum(mat.T).tocsr()
    graph = weak_csr_graph(dict(n=7, m=3, directed=True), weak)
    assert graph['m'] == 6 and graph['directed'] is False
    rebuilt = sparse.csr_matrix((np.ones(6), graph['indices'], graph['offsets']), shape=(7, 7))
    assert (rebuilt != weak).nnz == 0 and len(graph['offsets']) == 8


def test_author_constructor_receives_wide_offsets_and_indices_unchanged():
    observed = {}
    class CP:
        int32=np.int32; float64=np.float64
        asarray=staticmethod(np.asarray); ones=staticmethod(np.ones); arange=staticmethod(np.arange)
        @staticmethod
        def repeat(*args, **kwargs):
            raise AssertionError('CuPy device-array repeat is unsupported in installed release')
        diff=staticmethod(np.diff)
        cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None))
    class PLC:
        ResourceHandle=staticmethod(lambda: object())
        GraphProperties=staticmethod(lambda **kw: kw)
        @staticmethod
        def SGGraph(handle, properties, offsets, indices, **kw):
            observed.update(offsets=offsets, indices=indices, properties=properties, **kw)
            return object()
    offsets=np.array([0, 1, 2, 2], dtype=np.int64)
    indices=np.array([1, 0], dtype=np.int64)
    backend=CuGraphCSR(dict(n=3, m=2, directed=False, offsets=offsets, indices=indices),
                       'wcc', cp=CP, plc=PLC)
    np.testing.assert_array_equal(observed['offsets'], np.array([0, 1], dtype=np.int64))
    assert observed['indices'] is indices
    np.testing.assert_array_equal(observed['vertices_array'], np.array([0, 1, 2], dtype=np.int64))
    assert observed['input_array_format'] == 'COO' and observed['weight_array'] is None
    assert backend.contract['topology_dtype'] == 'int64'



def test_wide_coo_empty_rows_preserve_explicit_vertex_domain():
    seen = {}
    class CP:
        float64=np.float64
        asarray=staticmethod(np.asarray); ones=staticmethod(np.ones); arange=staticmethod(np.arange)
        cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None))
    class PLC:
        ResourceHandle=staticmethod(lambda: object())
        GraphProperties=staticmethod(lambda **kw: kw)
        @staticmethod
        def SGGraph(handle, properties, rows, columns, **kw):
            seen.update(rows=rows, columns=columns, **kw)
            return object()
    for offsets, indices in [([0,0,0,0], []), ([0,0,2,2], [0,2])]:
        g=dict(n=3,m=len(indices),directed=False,offsets=np.array(offsets,dtype=np.int64),
               indices=np.array(indices,dtype=np.int64))
        CuGraphCSR(g,'wcc',cp=CP,plc=PLC)
        np.testing.assert_array_equal(seen['rows'], np.repeat(np.arange(3),np.diff(offsets)))
        np.testing.assert_array_equal(seen['columns'], indices)
        np.testing.assert_array_equal(seen['vertices_array'], [0,1,2])
        assert seen['input_array_format']=='COO'
