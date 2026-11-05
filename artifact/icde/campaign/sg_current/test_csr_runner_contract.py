"""CPU-only positive check of the timed public CSR entry point."""
import sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest import mock
import numpy as np
import sg_worker


def test_eggpu_backend_calls_public_csr_without_coo_conversion(tmp_path):
    observed=[];expected=np.arange(7,dtype=np.float64)
    backend=SimpleNamespace(query=lambda *a,**kw:SimpleNamespace(values=expected))
    def create(graphs,**kwargs):
        observed.append((graphs,kwargs));return backend
    session=SimpleNamespace(__module__='easygraph.gpu.session',create=create)
    root=ModuleType('easygraph');root.__path__=[]
    gpu=ModuleType('easygraph.gpu');gpu.__path__=[];gpu.Session=session;root.gpu=gpu
    module=ModuleType('easygraph.gpu.session');module.__file__=str(tmp_path/'easygraph/gpu/session.py')
    graph=dict(n=7,m=3,directed=True,offsets=np.array([0,1,2,3,3,3,3,3],dtype=np.int32),
        indices=np.array([1,2,0],dtype=np.int32),weights=np.array([2.,3.,4.]))
    args=SimpleNamespace(system='eggpu',release=tmp_path,native_dso=tmp_path/'cpp_easygraph.so',
        budget_mib=4096,algorithm='pr')
    with mock.patch.dict(sys.modules,{'easygraph':root,'easygraph.gpu':gpu,
            'easygraph.gpu.session':module,'cpp_easygraph':None}), \
         mock.patch.object(sys,'path',list(sys.path)), \
         mock.patch.object(np,'repeat',side_effect=AssertionError('No CSR-to-COO row expansion')), \
         mock.patch.object(np,'column_stack',side_effect=AssertionError('No endpoint matrix')):
        got,query=sg_worker.build_backend(args,graph,None)
        assert got is backend and query() is expected
    payload=observed[0][0]['g'];kwargs=observed[0][1]
    assert kwargs['input_format']=='csr' and kwargs['state_model']=='static'
    assert set(payload)=={'nodes','offsets','indices','weights'}
    np.testing.assert_array_equal(payload['nodes'],np.arange(7))
    for key in ('offsets','indices','weights'):assert payload[key] is graph[key]
