"""CPU-only preserved public API and new canonical MG contracts."""
import copy,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import cugraph_mg_adapter as a
import cugraph_mg_worker as worker
import native_worker
import run_cugraph_mg
import sg_worker as shared


def fake_cp():
    return SimpleNamespace(asarray=np.asarray,asnumpy=lambda x:np.array(x,copy=True),ones=np.ones,
        int32=np.int32,float64=np.float64)

class PreservedAdapterContracts(unittest.TestCase):
    def test_01_nonzero_sources_actual_public_arguments(self):
            called=[]; ids=np.arange(10,dtype=np.int32)
            def bfs(handle,**kw):
                called.append(('bfs',kw));v=np.arange(10,dtype=np.int32);v[0]=np.iinfo(np.int32).max
                return v,None,ids
            def sssp(handle,**kw):
                self.assertTrue(kw['compute_predecessors']);called.append(('sssp',kw))
                v=np.arange(10,dtype=np.float64);v[0]=np.finfo(np.float64).max
                return ids,v,None
            def pr(handle,**kw):called.append(('pr',kw));return ids,np.full(10,.1)
            def wcc(handle,**kw):called.append(('wcc',kw));return ids,np.full(10,7,dtype=np.int32)
            plc=SimpleNamespace(bfs=bfs,sssp=sssp,pagerank=pr,weakly_connected_components=wcc)
            state=dict(handle=object(),graph=object(),vertices=ids,degree=np.ones(10))
            b=a.public_query(plc,fake_cp(),state,'bfs',True,9);self.assertTrue(np.isinf(b['values'][0]))
            np.testing.assert_array_equal(called[-1][1]['sources'],[9])
            a.public_query(plc,fake_cp(),state,'bfs',False,9);self.assertEqual(len(called[-1][1]['sources']),0)
            s=a.public_query(plc,fake_cp(),state,'sssp',False,9);self.assertTrue(np.isinf(s['values'][0]));self.assertEqual(called[-1][1]['source'],9)
            a.public_query(plc,fake_cp(),state,'pr',True,9)
            kw=called[-1][1];self.assertEqual(kw['epsilon'],1e-9);self.assertEqual(kw['max_iterations'],10000)
            self.assertEqual(kw['alpha'],.85);self.assertTrue(kw['fail_on_nonconvergence']);self.assertIsNone(kw['initial_guess_values'])
            self.assertIs(kw['precomputed_vertex_out_weight_sums'],state['degree'])
            a.public_query(plc,fake_cp(),state,'wcc',True,9);self.assertIsNone(called[-1][1]['weights'])

    def test_02_native_graph_arguments(self):
            observed=[]
            plc=SimpleNamespace(GraphProperties=lambda **kw:kw,MGGraph=lambda **kw:observed.append(kw) or object())
            for fn in ('bfs','sssp','wcc','pr'):
                state=a.make_graph(plc,fake_cp(),object(),np.array([[0,1],[1,2]],dtype=np.int32),
                    np.arange(4,dtype=np.int32),np.array([2.,3.]),np.array([1.,1.,0.,0.]),fn,True)
                kw=observed[-1];self.assertEqual(kw['num_arrays'],1);self.assertEqual(kw['store_transposed'],fn=='pr')
                self.assertEqual(kw['graph_properties']['is_symmetric'],fn=='wcc')
                self.assertFalse(kw['symmetrize']);np.testing.assert_array_equal(kw['vertices_array'][0],[0,1,2,3])
                if fn=='sssp':np.testing.assert_array_equal(kw['weight_array'][0],[2.,3.])
                if fn=='pr':np.testing.assert_array_equal(kw['weight_array'][0],[1.,1.]);self.assertEqual(kw['weight_array'][0].dtype,np.float64)

    def test_03_linear_scatter_preserves_native_wcc_labels_and_owns_output(self):
            pieces=[dict(vertices=np.array([3,1]),values=np.array([8,9],dtype=np.int32)),dict(vertices=np.array([0,2]),values=np.array([9,8],dtype=np.int32))]
            result=a.complete_output(pieces,4,'wcc');np.testing.assert_array_equal(result,[9,9,8,8])
            self.assertTrue(result.flags.owndata);self.assertTrue(result.flags.c_contiguous);self.assertFalse(result.flags.writeable)
            second=a.complete_output(pieces,4,'wcc');self.assertFalse(np.shares_memory(result,second))
            pieces[0]['values'][:]=0;np.testing.assert_array_equal(result,[9,9,8,8])
            del pieces;np.testing.assert_array_equal(second,[9,9,8,8])

    def test_04_full_domain_errors(self):
            for ids in ([0,0,2],[0,1,3],[-1,1,2],[0,1]):
                with self.assertRaises(ValueError):a.complete_output([dict(vertices=np.array(ids),values=np.ones(len(ids)))],3,'pr')
            with self.assertRaises(ValueError):a.complete_output([dict(vertices=np.arange(3),values=np.ones(3,dtype=np.float32))],3,'pr')
            with self.assertRaises(ValueError):a.complete_output([dict(vertices=np.arange(3),values=np.ones(3))],3,'wcc')
            with self.assertRaises(ValueError):a.complete_output([dict(vertices=np.arange(3,dtype=float),values=np.ones(3))],3,'pr')
            with self.assertRaises(ValueError):a.complete_output([dict(vertices=np.arange(3),values=np.ones((3,1)))],3,'pr')

    def test_05_rank_csr_conversion_exact_no_disk_coo(self):
            with tempfile.TemporaryDirectory() as name:
                root=Path(name);offsets=np.array([0,2,2,3,4,4],dtype=np.int32);indices=np.array([2,3,0,2],dtype=np.int32);weights=np.array([7.,2.,4.,3.])
                for role,value in [('offsets',offsets),('indices',indices),('weights',weights)]:value.tofile(root/(role+'.bin'))
                descriptor=dict(n=5,m=4,directed=True,files={role:str(root/(role+'.bin')) for role in ('offsets','indices','weights')})
                initial=set(root.iterdir())
                for count in (2,4,8):
                    pieces=[a.rank_host(descriptor,r,count,'sssp') for r in range(count)]
                    np.testing.assert_array_equal(np.concatenate([p[0] for p in pieces]),[[0,2],[0,3],[2,0],[3,2]])
                    np.testing.assert_array_equal(np.concatenate([p[2] for p in pieces]),weights)
                    np.testing.assert_array_equal(np.sort(np.concatenate([p[1] for p in pieces])),np.arange(5))
                    pr=[a.rank_host(descriptor,r,count,'pr') for r in range(count)]
                    full=np.empty(5)
                    for _,nodes,_,degree in pr:full[nodes]=degree
                    np.testing.assert_array_equal(full,[2,0,1,1,0])
                self.assertEqual(set(root.iterdir()),initial)
                with self.assertRaises(ValueError):a.rank_host(descriptor,0,2,'wcc')
                descriptor['directed']=False
                a.rank_host(descriptor,0,2,'wcc')
                with self.assertRaises(ValueError):a.rank_host(descriptor,0,2,'bfs')
                del pieces, pr
                import gc; gc.collect()

    def test_08_device_and_cpu_binding(self):
            obj=object.__new__(a.CuGraphMG);obj.uuids=['GPU-a','GPU-b'];obj.cpus=[0,1]
            bindings={str(i):dict(uuid='GPU-'+u,host_pid=i+10,own_context_observed=True,current_device=0,cpu_affinity=[0,1]) for i,u in enumerate(('a','b'))}
            obj._check_bindings(bindings)
            bad=copy.deepcopy(bindings);bad['1']['uuid']='GPU-a'
            with self.assertRaises(RuntimeError):obj._check_bindings(bad)
            bad=copy.deepcopy(bindings);bad['1']['host_pid']=10
            with self.assertRaises(RuntimeError):obj._check_bindings(bad)
            bad=copy.deepcopy(bindings);bad['1']['own_context_observed']=False
            with self.assertRaises(RuntimeError):obj._check_bindings(bad)
            bad=copy.deepcopy(bindings);bad['1']['cpu_affinity']=[1]
            obj._check_bindings(bad)
            with self.assertRaises(RuntimeError):obj._check_bindings(bad, require_context=False)
            bad['1']['cpu_affinity']=[2]
            with self.assertRaises(RuntimeError):obj._check_bindings(bad)
            bad['1']['cpu_affinity']=[]
            with self.assertRaises(RuntimeError):obj._check_bindings(bad)

    def test_09_loopback_cluster_shape_and_no_dashboard(self):
            address='tcp://127.0.0.1:1000';wa='tcp://127.0.0.1:1001'
            def node(addr,port):return dict(address=addr,http_bound_addresses=[dict(host='127.0.0.1',port=port)],http_service_port=port,
                application_types=[],bokeh_application_types=[],worker_dashboard_flag=False)
            scheduler=node(address,5000);ranks={wa:node(wa,5001)}
            info=dict(address=address,workers={wa:dict(memory_limit=32<<30,nthreads=1)},services=dict(dashboard=5000))
            a.check_cluster_shape(info,scheduler,ranks,1,32<<30)
            bad=copy.deepcopy(ranks);bad[wa]['http_bound_addresses'][0]['host']='0.0.0.0'
            with self.assertRaises(RuntimeError):a.check_cluster_shape(info,scheduler,bad,1,32<<30)
            bad=copy.deepcopy(ranks);bad[wa]['application_types']=['bokeh.server']
            with self.assertRaises(RuntimeError):a.check_cluster_shape(info,scheduler,bad,1,32<<30)

    def test_12_complete_query_lifecycle_and_close(self):
            released=[];closed=[];observed=[]
            class Future:
                def __init__(self,value):self.value=value
                def release(self):released.append(self.value)
            class Client:
                def submit(self,fn,state,function,owner,source,**kw):
                    observed.append((state,function,owner,source,kw))
                    return Future(dict(vertices=np.array([state]),values=np.array([float(state)])))
                def gather(self,fs):return [x.value for x in fs]
                def close(self,timeout):closed.append('client')
            obj=object.__new__(a.CuGraphMG);obj.closed=False;obj.function='sssp';obj.source=9;obj.n=2
            obj.uuids=['GPU-a','GPU-b'];obj.client=Client();obj.futures=[0,1];obj.workers=['worker-a','worker-b']
            obj.metrics={};held=[obj.complete_query() for _ in range(3)]
            for x in held:np.testing.assert_array_equal(x,[0.,1.]);self.assertTrue(x.flags.owndata)
            self.assertTrue(all(not np.shares_memory(x,y) for i,x in enumerate(held) for y in held[:i]))
            self.assertTrue(all(row[3]==9 for row in observed));self.assertEqual([x[2] for x in observed],[True,False]*3)
            obj.futures=[Future('graph-a'),Future('graph-b')];obj.own_comms=True
            obj.comms=SimpleNamespace(destroy=lambda:closed.append('comms'))
            obj.cluster=SimpleNamespace(close=lambda timeout:closed.append('cluster'));obj.observation={}
            obj.close();self.assertEqual(closed,['comms','client','cluster'])
            for x in held:np.testing.assert_array_equal(x,[0.,1.])
            with self.assertRaises(RuntimeError):obj.complete_query()
            obj.close();self.assertEqual(len(closed),3)

class CanonicalContracts(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);native_worker.make_tiny(self.root)
  self.graph=shared.load_input(self.root/'canonical/TINY_D')
 def tearDown(self):
  for value in self.graph.values():
   if isinstance(value,np.memmap):value._mmap.close()
  self.graph=None
  import gc;gc.collect()
  self.temp.cleanup()
 def test_weak_mmap_retains_full_domain(self):
  graph,info=worker.prepared_graph(self.graph,'wcc',self.root);self.assertFalse(graph['directed'])
  self.assertEqual(graph['n'],16);self.assertIsInstance(graph['offsets'],np.memmap);self.assertIsNone(graph['weights'])
  desc=a.describe_graph(graph);pieces=[a.rank_host(desc,r,8,'wcc') for r in range(8)]
  np.testing.assert_array_equal(np.sort(np.concatenate([v[1] for v in pieces])),np.arange(16))
 def test_unweighted_directedness_does_not_resymmetrize(self):
  graph,info=worker.prepared_graph(self.graph,'bfs',self.root);self.assertTrue(graph['directed'])
  self.assertEqual(graph['m'],self.graph['m']);self.assertIs(graph['indices'],self.graph['indices'])
 def test_weighted_canonical_values_reach_rank_loader(self):
  graph,info=worker.prepared_graph(self.graph,'sssp',self.root);desc=a.describe_graph(graph)
  pieces=[a.rank_host(desc,r,2,'sssp') for r in range(2)]
  np.testing.assert_array_equal(np.concatenate([v[2] for v in pieces]),self.graph['weights'])
 def test_no_gpu_without_optin(self):
  with self.assertRaisesRegex(ValueError,'execute-gpu'):worker.job(SimpleNamespace(execute_gpu=False))
 def test_plan_has_only_multigpu_configs_and_no_execution(self):
  cpus=sorted(os.sched_getaffinity(0))[:16]
  args=SimpleNamespace(proofs=None,counts='2,4,8',datasets='TINY_D,TINY_U',algorithms='bfs,sssp,wcc,pr',phase='tiny',repeat=0,
   gpu_uuids=','.join('GPU-'+str(i) for i in range(8)),cpu_set=','.join(map(str,cpus)),output=self.root/'plan',rapids_python=Path('/rapids/python'))
  with patch('subprocess.Popen',side_effect=AssertionError('CPU plan')):jobs=run_cugraph_mg.plan(args)
  self.assertEqual(len(jobs),24);self.assertFalse(args.output.exists())
  args.counts='1'
  with self.assertRaises(ValueError):run_cugraph_mg.plan(args)
 def test_target_proof_gate(self):
  cpus=sorted(os.sched_getaffinity(0))[:16]
  args=SimpleNamespace(proofs=None,counts='2',datasets='EP',algorithms='pr',phase='target',repeat=0,
   gpu_uuids='GPU-a,GPU-b',cpu_set=','.join(map(str,cpus)),output=self.root/'plan',rapids_python=Path('/rapids/python'))
  with self.assertRaisesRegex(ValueError,'Missing tiny proof'):run_cugraph_mg.plan(args)

 def test_complete_source_child_audits_owned_outputs_after_close(self):
  import fcntl,sys
  folder=self.root/'job';folder.mkdir();expected=shared.reference(self.graph,'bfs',0);events=[]
  cpus=sorted(os.sched_getaffinity(0))[:16]
  class FakeBackend:
   def __init__(self,graph,*rest):
    self.metrics={'devices_used':[0,1]};self.contract={'fake':'CPU contract'}
    stage=json.loads((folder/'stage.json').read_text());events.append(('construct',stage['label'],stage['active']))
   def complete_query(self):
    events.append(('query',));value=expected.copy();value.flags.writeable=False;return value
   def observe(self):return {'fake':'no GPU'}
   def close(self):events.append(('close',))
  args=SimpleNamespace(output=folder,input_root=self.root,dataset='TINY_D',phase='tiny',repeat=0,algorithm='bfs',
   count=2,gpu_uuids='GPU-a,GPU-b',cpu_set=','.join(map(str,cpus)),source_index=0,epoch_sha256='cpu',
   timeout_seconds=10,reference_cache=self.root/'refs',lock_file=self.root/'global.lock')
  original_audit=shared.audit_output
  def after_close(*values):
   self.assertEqual(events[-1],('close',));return original_audit(*values)
  fake_pickle=SimpleNamespace(register_pickle_by_value=lambda module:None)
  with args.lock_file.open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);args.lease_fd=lock.fileno()
   with patch.object(worker,'epoch',return_value=({},'cpu')),patch.object(a,'CuGraphMG',FakeBackend),patch.object(shared,'audit_output',side_effect=after_close),patch.dict(sys.modules,{'cloudpickle':fake_pickle}):
    worker.source_child(args)
  result=json.loads((folder/'result.json').read_text());self.assertEqual(result['status'],'PASS')
  self.assertEqual(events[0],('construct','first',True));self.assertEqual(sum(e==('query',) for e in events),3)
  self.assertTrue(result['cleanup_completed']);self.assertEqual(set(result['calls']),{'first','warmup','ready'})
  self.assertGreater(result['T_first_seconds'],0)

 def test_child_requires_inherited_global_lock(self):
  with self.assertRaisesRegex(ValueError,'inherited global GPU lease'):
   worker.source_child(SimpleNamespace(lease_fd=None))

if __name__=='__main__':unittest.main()
