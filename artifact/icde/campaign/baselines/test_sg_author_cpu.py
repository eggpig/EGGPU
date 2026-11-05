"""CPU-only canonical input/launch/timer contracts; no GPU baseline invocation."""
from pathlib import Path
import argparse, json, tempfile, unittest
import numpy as np
import sg_author_worker as w


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);w.common.make_tiny(cls.root)
        cls.graph=w.common.load_input(cls.root/'canonical/TINY_D')
    @classmethod
    def tearDownClass(cls):
        for value in cls.graph.values():
            if isinstance(value,np.memmap):value._mmap.close()
        cls.graph=None
        import gc;gc.collect()
        cls.tmp.cleanup()
    def args(self,system,algorithm):
        return argparse.Namespace(system=system,algorithm=algorithm,root=self.root,gpu_uuid='GPU-test',timeout=3600)
    def test_direct_csr_retains_weights_and_full_vertex_domain(self):
        g=self.graph
        for system in w.SYSTEMS:
            for function in ('bfs','sssp'):
                layout=w.graph_layout(g,system,function)
                np.testing.assert_array_equal(layout['offsets'],g['offsets']);np.testing.assert_array_equal(layout['indices'],g['indices'])
                np.testing.assert_array_equal(layout['weights'],g['weights']);self.assertEqual(layout['n'],7)
    def test_wcc_weak_projection_is_symmetric_and_preserves_isolate(self):
        for system in w.SYSTEMS:
            layout=w.graph_layout(self.graph,system,'wcc')
            m=w.common.graph_matrix(layout)
            self.assertEqual((m!=m.T).nnz,0);self.assertEqual(layout['n'],7)
            if system!='cggraph':self.assertEqual(m.getrow(6).nnz,0)
    def test_cggraph_reorder_is_explicit_bijection(self):
        layout=w.graph_layout(self.graph,'cggraph','pr');order=layout['order'];m=w.common.graph_matrix(layout)
        reference=w.common.graph_matrix(self.graph)[order][:,order]
        self.assertEqual((m!=reference).nnz,0);self.assertEqual(sorted(order),list(range(7)))
        np.testing.assert_array_equal(layout['degrees'],np.diff(self.graph['offsets'])[order])
    def test_cggraph_uses_u32_not_misnamed_u64(self):
        layout=w.graph_layout(self.graph,'cggraph','sssp');self.assertEqual(layout['offsets'].dtype.itemsize,4)
        self.assertEqual(layout['weights_u32'].dtype,np.dtype('<u4'))
    def test_legacy_full_sssp_reorder_keeps_weights_and_source(self):
        layout=w.graph_layout(self.graph,'cggraph','sssp',legacy=True);order=layout['order']
        actual=w.common.graph_matrix(layout,weighted=True)
        expected=w.common.graph_matrix(self.graph,weighted=True)[order][:,order]
        self.assertEqual((actual!=expected).nnz,0)
        out=self.root/'legacy_call';out.mkdir();args=self.args('cggraph','sssp');args.variant='cggraph_legacy_full'
        spec=w.invocation(args,{'binary':{'path':'never_execute'}},layout,out/'stage',out,5)
        self.assertIn('--algorithm=1',spec['argv']);self.assertIn('--root=4',spec['argv'])
        self.assertEqual(spec['marker'],'RECENT_CG_JSON ')
    def test_text_preserves_explicit_integer_weights(self):
        for system in ('hytgraph','grapin'):
            layout=w.graph_layout(self.graph,system,'sssp');stage=self.root/(system+'_text');w.write_layout(layout,stage,system,'sssp')
            rows=np.loadtxt(stage/'graph.edges',dtype=np.int64);np.testing.assert_array_equal(rows[:,2],self.graph['weights'])
            np.testing.assert_array_equal(rows[:,1],self.graph['indices'])
    def test_market_upper_triangle_keeps_explicit_n(self):
        layout=w.graph_layout(self.graph,'gunrock','wcc');stage=self.root/'market';w.write_layout(layout,stage,'gunrock','wcc')
        lines=(stage/'graph.mtx').read_text().splitlines();self.assertTrue(lines[0].endswith('symmetric'))
        self.assertEqual([int(x) for x in lines[1].split()[:2]],[7,7])
        self.assertTrue(all(int(line.split()[0])<int(line.split()[1]) for line in lines[2:]))
    def test_source_bound_to_all_three_native_calls(self):
        for system in ('gunrock','cggraph','sygraph'):
            out=self.root/(system+'_trace');out.mkdir();layout=w.graph_layout(self.graph,system,'sssp')
            spec=w.invocation(self.args(system,'sssp'),{'binary':{'path':'never_execute'}},layout,out/'stage',out,2)
            calls=[x for x in (out/'trace.tsv').read_text().splitlines() if x.startswith('C ')]
            self.assertEqual(len(calls),3);self.assertTrue(all(x.split()[4]=='2' for x in calls))
            self.assertEqual(bool(spec['stdin']),system=='gunrock')
    def test_hytgraph_preserves_weights_and_exact_output_marker(self):
        out=self.root/'hyt_call';out.mkdir();layout=w.graph_layout(self.graph,'hytgraph','sssp')
        spec=w.invocation(self.args('hytgraph','sssp'),{'binary':{'path':'never_execute'}},layout,out/'stage',out,2)
        self.assertIn('--weight_num=0',spec['argv']);self.assertIn('--source_node=2',spec['argv'])
        self.assertEqual(spec['marker'],'HYTGRAPH_COMPLETE_JSON ');self.assertEqual(spec['environment']['EGGPU_HYTGRAPH_NODES'],'7')
    def test_grapin_only_source_functions_receive_source_node(self):
        for function in ('bfs','sssp','wcc','pr'):
            out=self.root/('grapin_cli_'+function);out.mkdir()
            layout=w.graph_layout(self.graph,'grapin',function)
            spec=w.invocation(self.args('grapin',function),{'binary':{'path':'never_execute'}},layout,out/'stage',out,2)
            self.assertEqual(any(x.startswith('--source_node=') for x in spec['argv']),function in ('bfs','sssp'))
    def test_sycl_preload_uses_its_abi_compatible_environment(self):
        env=w.environment(self.args('sygraph','bfs'))
        self.assertEqual(env['LD_PRELOAD'],str(self.root/'envs/sycl/lib/libstdc++.so.6'))
        self.assertEqual(env['LD_LIBRARY_PATH'].split(':')[0],str(self.root/'envs/sycl/lib'))
    def test_missing_wcc_first_timer_cannot_be_silently_inferred(self):
        rows=[{'complete_query_seconds':1.}]*3
        with self.assertRaises(ValueError):w.native_times(self.args('gunrock','wcc'),rows,'')
        log='\n'.join('CAMPAIGN_CC_FIRST_JSON '+json.dumps({'round':i,'first_request_seconds':10+i}) for i in range(3))
        self.assertEqual(w.native_times(self.args('gunrock','wcc'),rows,log),{'first':10,'ready':1.})
    def test_native_timer_components_not_double_counted(self):
        rows=[{'host_csr_read_seconds':2.,'runtime_init_seconds':3.,'complete_call_seconds':5.}]*3
        self.assertEqual(w.native_times(self.args('cggraph','bfs'),rows,''),{'first':10.,'ready':5.})
    def test_controller_keeps_full_and_modern_source_variants_separate(self):
        import run_sg_authors as controller
        args=argparse.Namespace(variants=','.join(controller.VARIANTS),datasets='TINY_D,TINY_U',algorithms='bfs,sssp,wcc,pr',
            proofs=None,phase='tiny',repeat=None,input_root=self.root,output=self.root/'planned',python=Path('/never/run'),
            root=self.root,clock_build_root=self.root/'build',setup=self.root/'setup.json',reference_cache=self.root/'ref',lock_file=self.root/'lock',
            gpu_uuid='GPU-test',cpus=','.join(str(x) for x in range(16)),timeout=3600,min_host_free_gib=8.)
        jobs=controller.plan(args);self.assertEqual(len(jobs),42)
        legacy=[j for j in jobs if j['variant']=='cggraph_legacy_full']
        modern=[j for j in jobs if j['variant']=='cggraph_v15']
        self.assertEqual({j['algorithm'] for j in legacy},{'bfs','sssp','wcc','pr'})
        self.assertEqual({j['algorithm'] for j in modern},{'bfs','sssp'})
        self.assertEqual(len({j['output'] for j in jobs}),len(jobs))
        self.assertFalse(args.output.exists())


if __name__=='__main__':unittest.main(verbosity=2)
