"""Actual tiny KGraph CPU executors and input/batch contracts; no GPU calls."""
from pathlib import Path
import argparse,json,os,tempfile,time,unittest
import numpy as np
from scipy.sparse.csgraph import dijkstra
import input_contract as inputs
import worker


class BatchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        inputs.tiny(cls.root)
        cls.graphs={name:inputs.common.load_input(cls.root/'canonical'/name) for name in ('TINY_D','TINY_U')}
        cls.author=json.loads(Path(os.environ['E4S_CPU_AUTHOR_BUILD']).read_text())
        if cls.author['status']!='PASS_BUILD_ONLY':raise ValueError('Fixed author CPU build required')
        cls.results=[]

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_source_sequence_prefix_unique_and_unmodified(self):
        for name,graph in self.graphs.items():
            source=Path(graph['directory'])/'e4s_sources.json'
            values=[inputs.load_sources(source,graph,n) for n in inputs.BATCHES]
            self.assertEqual([len(v) for v in values],[32,128,512])
            self.assertEqual(values[0],values[1][:32]);self.assertEqual(values[1],values[2][:128])
            self.assertTrue(all(v<513 for v in values[2]))
            self.assertEqual(len(set(values[2])),512)

    def test_chunk_partition_roundtrip_preserves_all_arcs_weights_and_isolates(self):
        for name,graph in self.graphs.items():
            out=self.root/(name+'_chunk');proof=inputs.kgraph_layout(graph,out)
            parts=np.loadtxt(out/'parts.txt',dtype=np.int64)
            self.assertEqual(len(parts),521);self.assertEqual(proof['partitions'],8);self.assertFalse(proof['optimizer_used'])
            reconstructed=[]
            for kind in ('inter','intra'):
                lines=(out/(kind+'.adj')).read_text().splitlines();self.assertEqual(lines[0],'WeightedAdjacencyGraph')
                n,m=map(int,lines[1:3]);self.assertEqual(n,graph['n'])
                row=np.r_[np.array(lines[3:3+n],dtype=np.int64),m]
                dst=np.array(lines[3+n:3+n+m],dtype=np.int64);weight=np.array(lines[3+n+m:],dtype=np.int64)
                self.assertEqual(len(weight),m)
                for node in range(n):
                    for k in range(row[node],row[node+1]):
                        self.assertEqual(parts[node]!=parts[dst[k]],kind=='inter')
                        reconstructed.append((node,int(dst[k]),int(weight[k])))
            expected=[(u,int(graph['indices'][k]),int(graph['weights'][k])) for u in range(graph['n'])
                for k in range(graph['offsets'][u],graph['offsets'][u+1])]
            self.assertEqual(sorted(reconstructed),expected)

    def test_c2_origin_uses_exact_w16_and_preserves_full_vertex_domain(self):
        for name,graph in self.graphs.items():
            out=self.root/(name+'_c2');inputs.c2_layout(graph,out)
            for file,role in [('vlist','offsets'),('elist','indices'),('wlist','weights')]:
                np.testing.assert_array_equal(np.fromfile(out/'origin'/(file+'.bin'),dtype='<i4'),graph[role])

    def test_real_kgraph_two_variants_two_orientations_32_distinct_requests(self):
        for name,graph in self.graphs.items():
            sequence=inputs.load_sources(Path(graph['directory'])/'e4s_sources.json',graph,32)
            expected=dijkstra(inputs.common.graph_matrix(graph,weighted=True),directed=True,indices=sequence)
            for variant in ('kgraph_sparse','kgraph_dense'):
                out=self.root/(name+'_'+variant);out.mkdir()
                args=argparse.Namespace(output=out,variant=variant,timeout=120,audit_timeout=120)
                entry={'binary':self.author['binaries'][variant]}
                def audit(values,index,source):
                    self.assertEqual(source,sequence[index]);self.assertEqual(len(values),graph['n'])
                    self.assertEqual(values.dtype,np.dtype('<f8'));self.assertFalse(values.flags.writeable)
                    self.assertTrue(values.flags.owndata)
                    np.testing.assert_array_equal(values,expected[index],err_msg=f'{variant} {name} source {source}')
                    return {'sequence':index,'source':source,'pass':True}
                details,checks=worker.author_batch(args,graph,sequence,entry,audit)
                self.assertEqual(len(checks),32)
                self.assertGreaterEqual(details['first_complete_seconds'],details['adapter_preparation_seconds']+details['query_batch_seconds'])
                self.assertEqual(details['summary']['queries'],32)
                self.results.append({'variant':variant,'orientation':name,'full_vectors':32,'values':32*graph['n'],
                    'first_interval_valid':True,'quality':True})

    def test_refuse_formal_execution_without_owner_flag(self):
        with self.assertRaisesRegex(ValueError,'primary agent'):
            worker.validate(argparse.Namespace(execute=False))

    def test_slow_independent_oracle_does_not_consume_native_timeout(self):
        graph=self.graphs['TINY_D'];sequence=inputs.load_sources(Path(graph['directory'])/'e4s_sources.json',graph,32)
        expected=dijkstra(inputs.common.graph_matrix(graph,weighted=True),indices=sequence,directed=True)
        out=self.root/'slow_oracle';out.mkdir()
        args=argparse.Namespace(output=out,variant='kgraph_sparse',timeout=1,audit_timeout=10)
        def audit(values,index,source):
            np.testing.assert_array_equal(values,expected[index]);time.sleep(.04)
            return {'sequence':index,'source':source,'pass':True}
        started=time.monotonic()
        detail,checks=worker.author_batch(args,graph,sequence,{'binary':self.author['binaries']['kgraph_sparse']},audit)
        self.assertGreater(time.monotonic()-started,1.2)
        self.assertLess(detail['first_complete_seconds'],1)
        self.assertEqual(len(checks),32)

    def test_actual_reference_cache_full_vector_audit(self):
        graph=self.graphs['TINY_D'];source=17
        args=argparse.Namespace(reference_module=Path(os.environ['E4S_CPU_REFERENCE']),
            reference_cache=self.root/'reference_cache',reference_cache_gib=.01)
        audit=worker.quality_context(args,graph)
        values=dijkstra(inputs.common.graph_matrix(graph,weighted=True),indices=source,directed=True)
        values=np.array(values,dtype='<f8',copy=True);values.flags.writeable=False
        first=audit(values,0,source);second=audit(values,1,source)
        self.assertTrue(first['quality']['passed']);self.assertTrue(second['quality']['reference']['cache_hit'])
        changed=values.copy();changed[source]=1;changed.flags.writeable=False
        with self.assertRaises(AssertionError):audit(changed,2,source)

    def test_controller_has_24_tiny_jobs_without_execution(self):
        import run_e4s
        sources=self.root/'controller_sources';sources.mkdir()
        for name,graph in self.graphs.items():
            (sources/(name+'.json')).write_bytes((Path(graph['directory'])/'e4s_sources.json').read_bytes())
        args=argparse.Namespace(proofs=None,phase='tiny',repeat=None,datasets='TINY_D,TINY_U',batches='32,128,512',
            input_root=self.root,sources_dir=sources,output=self.root/'plan_only',python=Path('/never/run'),
            root=self.root,author_build=Path(os.environ['E4S_CPU_AUTHOR_BUILD']),release=self.root/'release',
            native_dso=self.root/'not_loaded.so',arena_library_dir=self.root/'arena',reference_module=Path(os.environ['E4S_CPU_REFERENCE']),
            reference_cache=self.root/'refs',setup=self.root/'setup.json',lock_file=self.root/'lock',
            gpu_uuid='GPU-no-execution',cpus=','.join(map(str,range(16))),budget_mib=22000,reference_cache_gib=4.,timeout=30,audit_timeout=120,min_host_free_gib=8.)
        jobs=run_e4s.plan(args)
        self.assertEqual(len(jobs),24);self.assertEqual(len({j['id'] for j in jobs}),24)
        self.assertEqual(sum(j['gpu_count']==0 for j in jobs),12)
        self.assertFalse(args.output.exists())


if __name__=='__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    unittest.main(verbosity=2)
