"""Actual CPU compressor + author H-closure tests; never run GPU executor."""
from pathlib import Path
import argparse,heapq,json,os,subprocess,tempfile,unittest
import numpy as np
from scipy.sparse.csgraph import dijkstra
import c2p_worker as worker


class C2PContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.build=json.loads(Path(os.environ['C2P_CPU_BUILD']).read_text())
        if cls.build['status']!='PASS_BUILD_ONLY':raise ValueError('CPU author build required')
        worker.common.make_tiny(cls.root)
        cls.graphs={name:worker.common.load_input(cls.root/'canonical'/name) for name in ('TINY_D','TINY_U')}
        helper=cls.root/'closure.cpp';header=Path(cls.build['header']['path']).parent/'source_compensation.hpp'
        helper.write_text('#include "'+str(header)+'"\n#include <fstream>\n#include <iostream>\n'
            'std::vector<int> read(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);'
            'auto n=f.tellg();if(n<0||n%4)throw std::runtime_error("input");std::vector<int>v(n/4);f.seekg(0);'
            'f.read((char*)v.data(),n);return v;}\nint main(int argc,char**argv){'
            'std::string p=argv[1];auto s=c2_source_compensation::build(read(p+"/hvlist.bin"),'
            'read(p+"/helist.bin"),read(p+"/hwlist.bin"),std::stoi(argv[2]));'
            'for(size_t i=0;i<s.vertices.size();++i)std::cout<<s.vertices[i]<<" "<<s.distances[i]<<"\\n";return 0;}\n')
        cls.helper=cls.root/'closure'
        subprocess.run(['/usr/bin/g++','-std=c++17','-O2',str(helper),'-o',str(cls.helper)],check=True)
        cls.stages={}
        for name,graph in cls.graphs.items():
            stage=cls.root/(name+'_compressed');worker.origin_layout(graph,stage)
            with (cls.root/(name+'.compress.log')).open('w') as log:
                subprocess.run([cls.build['compressor']['path'],str(stage)],stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'),check=True,timeout=30)
            worker.validate_compression(stage,graph['n']);cls.stages[name]=stage

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_origin_exact_csr_and_explicit_weights(self):
        for name,stage in self.stages.items():
            graph=self.graphs[name]
            for file,key in [('vlist','offsets'),('elist','indices'),('wlist','weights')]:
                np.testing.assert_array_equal(np.fromfile(stage/'origin'/(file+'.bin'),dtype='<i4'),graph[key])

    def test_actual_compressor_preserves_all_vertex_ids(self):
        for name,stage in self.stages.items():
            proof=worker.validate_compression(stage,self.graphs[name]['n'])
            self.assertEqual(proof['compressed']['vertices'],7)
            self.assertEqual(proof['source_compensation']['vertices'],7)

    def test_actual_author_H_closure_plus_compressed_graph_all_sources(self):
        for name,stage in self.stages.items():
            graph=self.graphs[name];p=stage/'sssp'
            row=np.fromfile(p/'vlist.bin',dtype='<i4');col=np.fromfile(p/'elist.bin',dtype='<i4');weight=np.fromfile(p/'wlist.bin',dtype='<i4')
            for source in range(graph['n']):
                output=subprocess.check_output([str(self.helper),str(p),str(source)],text=True)
                seeds=[(0,source)]+[(int(line.split()[1]),int(line.split()[0])) for line in output.splitlines()]
                distance=np.full(graph['n'],np.inf);queue=[]
                for value,node in seeds:distance[node]=min(distance[node],value);heapq.heappush(queue,(value,node))
                while queue:
                    value,node=heapq.heappop(queue)
                    if value!=distance[node]:continue
                    for index in range(row[node],row[node+1]):
                        nxt=int(col[index]);candidate=value+int(weight[index])
                        if candidate<distance[nxt]:distance[nxt]=candidate;heapq.heappush(queue,(candidate,nxt))
                expected=dijkstra(worker.common.graph_matrix(graph,weighted=True),indices=source,directed=True)
                np.testing.assert_array_equal(distance,expected,err_msg=name+' source '+str(source))

    def test_cli_contains_three_independent_single_source_calls(self):
        out=self.root/'invocation';out.mkdir()
        args=argparse.Namespace(gpu_uuid='GPU-unexecuted')
        spec=worker.invocation(args,self.build,self.graphs['TINY_D'],self.stages['TINY_D'],out,5)
        calls=(out/'trace.tsv').read_text().splitlines()[1:]
        self.assertEqual([line.split()[4] for line in calls],['5','5','5'])
        self.assertEqual(spec['variant'],'P');self.assertEqual(spec['calls'],3)
        self.assertNotIn('pm',spec['argv'][0].lower())

    def test_no_execution_without_owner_flag(self):
        with self.assertRaisesRegex(ValueError,'primary agent'):
            worker.run(argparse.Namespace(execute_gpu=False))

    def test_controller_tiny_plan_has_no_execution_or_output_side_effect(self):
        import run_c2p
        args=argparse.Namespace(proofs=None,phase='tiny',repeat=None,datasets='TINY_D,TINY_U',
            input_root=self.root,output=self.root/'plan_only',python=Path('/never/run'),root=self.root,
            author_build=Path(os.environ['C2P_CPU_BUILD']),setup=self.root/'setup.json',
            reference_cache=self.root/'references',lock_file=self.root/'lock',gpu_uuid='GPU-never-run',
            cpus=','.join(map(str,range(16))),timeout=30,min_host_free_gib=8.)
        jobs=run_c2p.plan(args)
        self.assertEqual(len(jobs),2)
        self.assertTrue(all('sssp_c2graph_p' in row['id'] for row in jobs))
        self.assertFalse(args.output.exists())

    def test_exact_author_build_binding_and_cpu_ELF_resolution(self):
        args=argparse.Namespace(author_build=Path(os.environ['C2P_CPU_BUILD']))
        build=worker.artifact(args)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
            LD_LIBRARY_PATH='/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu')
        output=subprocess.check_output(['/usr/bin/ldd',build['binary']['path']],env=env,text=True)
        self.assertNotIn('not found',output)
        self.assertEqual(build['source_commit'],worker.COMMIT)


if __name__=='__main__':unittest.main(verbosity=2)
