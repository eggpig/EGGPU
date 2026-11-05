"""CPU-only protocol regression: inspect native arguments, never initialize CUDA."""
import unittest
from types import SimpleNamespace
from unittest import mock
import numpy as np
import cugraph_csr_adapter as adapter
import sequence16_adapter


class PRProtocolTests(unittest.TestCase):
    def test_native_epsilon_is_global_and_independent_of_graph_order(self):
        for n in (7, 75879, 23947347, 41652230):
            with self.subTest(n=n):
                calls=[]
                cp=SimpleNamespace(asnumpy=lambda v:v,
                    cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda:None)))
                def pagerank(*args, **kwargs):
                    calls.append(kwargs)
                    return np.arange(3,dtype=np.int32),np.full(3,1/3,dtype=np.float64)
                backend=adapter.CuGraphCSR.__new__(adapter.CuGraphCSR)
                backend.cp=cp;backend.plc=SimpleNamespace(pagerank=pagerank)
                backend.function='pr';backend.n=n;backend.handle=object();backend.graph=object()
                backend.vertices=object();backend.outdegree=object()
                # This call-argument test does not allocate a US/TW-size vector;
                # full domain validation is separately covered by SG CPU/GPU tests.
                with mock.patch.object(adapter,'ordered_output',return_value=np.ones(3)):
                    backend.complete_query()
                self.assertEqual(calls[0]['epsilon'],1e-9)
                self.assertEqual(calls[0]['max_iterations'],2000)
                self.assertEqual(calls[0]['alpha'],.85)
                self.assertTrue(calls[0]['fail_on_nonconvergence'])
                self.assertIsNone(calls[0]['initial_guess_values'])
                self.assertIsNone(calls[0]['initial_guess_vertices'])

    def test_constructor_exposes_actual_stopping_protocol(self):
        cp=SimpleNamespace(asarray=np.asarray,ones=np.ones,arange=np.arange,float64=np.float64,int32=np.int32,
            cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda:None)))
        plc=SimpleNamespace(ResourceHandle=lambda:object(),GraphProperties=lambda **kw:kw,
            SGGraph=lambda *a,**kw:object())
        graph=dict(n=3,m=2,directed=True,offsets=np.array([0,1,2,2],dtype=np.int32),
            indices=np.array([1,2],dtype=np.int32),weights=np.ones(2))
        backend=adapter.CuGraphCSR(graph,'pr',cp=cp,plc=plc)
        self.assertEqual(backend.contract['pr_epsilon_global_l1'],1e-9)
        self.assertEqual(backend.contract['pr_stopping_norm'],'sum(abs(new-old))')
        self.assertEqual(backend.contract['pr_value_dtype'],'float64')
        backend.close()

    def test_sequence16_uses_same_corrected_adapter(self):
        graph=dict(n=3,m=2,directed=True)
        seq=sequence16_adapter.CuGraphSequence(SimpleNamespace(),graph)
        created=[]
        class Fake(adapter.CuGraphCSR):
            def __init__(self,*args,**kwargs):
                self.contract={'pr_epsilon_global_l1':adapter.PR_EPSILON_GLOBAL_L1}
                created.append(self)
            def complete_query(self):return np.ones(3,dtype=np.float64)
            def close(self):pass
        with mock.patch.object(adapter,'CuGraphCSR',Fake):
            a=seq.query({'algorithm':'pr','source':None})
            b=seq.query({'algorithm':'pr','source':None})
        self.assertEqual(len(created),1)
        self.assertEqual(created[0].contract['pr_epsilon_global_l1'],1e-9)
        self.assertIsNot(a,b)
        self.assertFalse(a.flags.writeable)
        seq.close()

if __name__=='__main__':unittest.main()
