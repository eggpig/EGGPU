"""CPU checks for semantic identity, failure paths and snapshot portability."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import errno
import hashlib
import importlib.util
import json
import numpy as np
import pytest
from easygraph.gpu import Session, StaleRequestError
from easygraph.gpu.contracts import GraphDescriptor, GraphVersion, NodeDomain
from easygraph.gpu._dynamic import stage_weights, verify_manifest

class LazyBackend:
    session_id = 'test-session'
    def __init__(self):
        self.version = 1
        self.materialized = 0
        self.closed = False
        self.writable = False
    def describe(self, name):
        return GraphDescriptor(name, 'test-session/'+name, 'managed',
             GraphVersion(0,self.version,self.materialized,self.materialized,'observed'),
             NodeDomain.from_nodes(np.arange(3)),True,True,{})
    def capability_descriptor(self):
        return SimpleNamespace(algorithms=('bfs','sssp'))
    def query(self,name,algorithm,parameters):
        self.materialized=self.version
        values=np.arange(3,dtype=float);values.flags.writeable=self.writable
        return SimpleNamespace(graph=self.describe(name),values=values)
    def close(self):self.closed=True

def test_prepare_does_not_change_semantic_identity():
    backend=LazyBackend();session=Session(backend,kind='test')
    request=session.request('g','bfs',{'source':0})
    first=session.execute(request)
    second=session.execute(request)
    assert backend.materialized==1
    assert not np.shares_memory(first.values,second.values)
    session.close()
    np.testing.assert_array_equal(first.values,[0,1,2])
    with pytest.raises(RuntimeError,match='closed'):session.execute(request)

def test_committed_mutation_invalidates_old_request():
    backend=LazyBackend();session=Session(backend,kind='test')
    old=session.request('g','sssp',{'source':0});backend.version+=1
    with pytest.raises(StaleRequestError):session.execute(old)
    session.execute(session.request('g','sssp',{'source':0}))

def test_contract_rejects_wrong_session_and_mutable_output():
    backend=LazyBackend();session=Session(backend,kind='test')
    request=session.request('g','bfs',{'source':0})
    with pytest.raises(ValueError):session.execute(replace(request,session_id='foreign'))
    with pytest.raises(ValueError):session.request('g','bfs',{'source':True})
    backend.writable=True
    with pytest.raises(RuntimeError,match='immutable'):session.execute(request)

def fixture_manifest(tmp_path):
    script=Path(__file__).resolve().parents[2]/'scripts/gpu/qualification_inputs.py'
    spec=importlib.util.spec_from_file_location('qualification_inputs',script)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    edges=np.array([[0,1],[1,2],[2,0],[2,1]],dtype=np.int32)
    raw=mod.canonical(3,edges,np.array([1.,2.,3.,4.]))
    item,_=mod.manifest(raw,tmp_path/'base','base',1)
    return Path(item['manifest'])

def test_weight_commit_across_filesystems(tmp_path,monkeypatch):
    base=fixture_manifest(tmp_path)
    original=json.loads(base.read_text())
    before={x['path']:(base.parent/x['path']).read_bytes() for x in original['partitions'][0]['files'].values()}
    def cross_device(*a,**k):raise OSError(errno.EXDEV,'fixture different filesystem')
    monkeypatch.setattr('easygraph.gpu._dynamic.os.link',cross_device)
    target,receipt=stage_weights(base,'next',[(0,1,7.)],tmp_path/'commits')
    doc=json.loads(target.read_text());verify_manifest(target,doc)
    assert receipt['host_cross_filesystem_copy_bytes']>0
    assert receipt['host_payload_written_bytes']==receipt['host_weight_payload_written_bytes']+receipt['host_cross_filesystem_copy_bytes']
    assert doc['graph']['topology_sha256']==original['graph']['topology_sha256']
    values=np.fromfile(target.parent/doc['partitions'][0]['files']['weights']['path'],dtype='<f8')
    np.testing.assert_array_equal(values,[1.,7.,3.,4.])
    for name,value in before.items():assert (base.parent/name).read_bytes()==value

def test_invalid_weight_commit_never_publishes(tmp_path):
    base=fixture_manifest(tmp_path)
    with pytest.raises(ValueError):stage_weights(base,'bad',[(0,0,-1.)],tmp_path/'commits')
    assert not (tmp_path/'commits/bad').exists()


def test_pr_self_loop_and_explicit_native_options():
    import easygraph as eg
    graph=eg.Graph();graph.add_edge(0,0);graph.add_edge(0,1)
    expected=eg.pagerank(graph)
    actual=eg.pagerank(graph,max_iter=1000,tol=1e-13)
    np.testing.assert_allclose([actual[0],actual[1]],[expected[0],expected[1]],atol=1e-11)
    native=graph.cpp()
    actual_native=eg.pagerank(native,max_iter=1000,tol=1e-13,weight=None)
    assert len(actual_native)==2


def test_upstream_cpu_dijkstra_precision_and_new_functions():
    import cpp_easygraph as cpp
    import easygraph as eg
    graph=eg.DiGraph();graph.add_edge('s','a',weight=16777217.0);graph.add_edge('a','t',weight=.25)
    native=graph.cpp()
    values=cpp.cpp_dijkstra_multisource(native,['s'],weight='weight')
    assert float(values[0][2])==16777217.25
    for name in ['cpp_boruvka_mst_edges','cpp_average_shortest_path_length','cpp_eccentricity','cpp_louvain_communities']:
        assert hasattr(cpp,name),name
