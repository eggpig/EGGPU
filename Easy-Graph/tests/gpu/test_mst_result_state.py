"""MST result factories and mutable output attributes, without GPU execution."""
import copy
import importlib
import pickle
import sys
from collections import OrderedDict, UserDict
from types import SimpleNamespace

import easygraph as eg
import networkx as nx
import numpy as np
import pytest

from easygraph.classes._state_tracking import subscribe
from easygraph.utils import gpu_eggpu_backend as backend

mst=importlib.import_module('easygraph.functions.path.mst')


class ValidatedAttrs(dict):
    def __init__(self):
        super().__init__(factory_marker='kept')

    def __setitem__(self,key,value):
        if key=='cost' and value<0:raise ValueError('negative cost')
        super().__setitem__(key,value)

    def update(self,*args,**kwargs):
        for key,value in dict(*args,**kwargs).items():self[key]=value


class FactoryGraph(eg.Graph):
    edge_attr_dict_factory=ValidatedAttrs


def source_graph(factory=dict,graph_type=eg.Graph):
    graph=graph_type()
    if graph_type is not FactoryGraph:graph.edge_attr_dict_factory=factory
    graph.add_nodes_from(['a','b','c','isolated'])
    for u,v,w in [('a','b',1.5),('a','c',4.75),('b','c',2.25)]:
        graph.add_edge(u,v,cost=w,metadata='edge')
    return graph


def assert_mutation_contract(tree,source,base):
    edges=tree.edges;u,v,attrs=edges[0]
    assert isinstance(attrs,base)
    assert attrs is tree[u][v]
    if tree.is_directed():assert attrs is tree._pred[v][u]
    else:assert attrs is tree[v][u]
    source_before=source._mutation_generation
    before=tree._mutation_generation
    seen=[]
    def observe(graph,generation,mapping,keys):
        if mapping is not None:seen.append((graph,generation,mapping,keys))
    unsubscribe=subscribe(tree,observe)
    tree.cache['__eggpu_prepared_weight']=object()
    old=attrs['cost'];attrs['cost']=old+.5
    assert tree._mutation_generation==before+1
    assert source._mutation_generation==source_before
    assert '__eggpu_prepared_weight' not in tree.cache
    assert len(seen)==1 and seen[0][0] is tree and seen[0][2] is attrs
    # UserDict's custom lookup follows the existing conservative edge-scope
    # contract; ordinary dict lookup can identify the changed attribute.
    assert seen[0][3]==(None if base is UserDict else frozenset({'cost'}))
    assert tree[u][v]['cost']==old+.5
    assert len(tree.edges)==len(edges) and list(tree.nodes)==['a','b','c','isolated']
    unsubscribe()
    if base is ValidatedAttrs:
        generation=tree._mutation_generation
        with pytest.raises(ValueError,match='negative cost'):attrs['cost']=-1
        assert tree._mutation_generation==generation


@pytest.mark.parametrize('factory', [dict,OrderedDict,UserDict,ValidatedAttrs])
@pytest.mark.parametrize('maximum', [False,True])
@pytest.mark.parametrize('graph_type,algorithm', [(eg.Graph,'kruskal'),(eg.Graph,'prim'),(eg.DiGraph,'kruskal')])
def test_cpu_tree_factory_mutation_and_reference(monkeypatch,factory,graph_type,maximum,algorithm):
    monkeypatch.setattr(mst,'gpu_runtime_enabled',lambda:False)
    graph=source_graph(factory,graph_type)
    function=eg.maximum_spanning_tree if maximum else eg.minimum_spanning_tree
    tree=function(graph,weight='cost',algorithm=algorithm)
    oracle=nx.Graph();oracle.add_nodes_from(graph.nodes)
    for u,v,attrs in graph.edges:oracle.add_edge(u,v,**dict(attrs))
    expected=(nx.maximum_spanning_tree if maximum else nx.minimum_spanning_tree)(oracle,weight='cost',algorithm=algorithm)
    assert {frozenset((u,v)) for u,v,_ in tree.edges}=={frozenset(e) for e in expected.edges}
    assert sum(attrs['cost'] for _,_,attrs in tree.edges)==sum(d['cost'] for _,_,d in expected.edges(data=True))
    assert isinstance(tree,graph_type)
    assert_mutation_contract(tree,graph,factory)


@pytest.mark.parametrize('factory', [dict,OrderedDict,UserDict,ValidatedAttrs])
@pytest.mark.parametrize('route', ['dispatch_fast','dispatch_directed_fallback','backend_fallback','native_lazy'])
def test_cpu_mocked_gpu_result_routes(monkeypatch,factory,route):
    graph=source_graph(factory,eg.DiGraph if route=='dispatch_directed_fallback' else eg.Graph)
    edges=[('a','b',1.5),('b','c',2.25)]
    prepared={'nodes':list(graph.nodes),'directed':graph.is_directed(),'stamp':(False,4,3,0)}
    monkeypatch.setattr(mst,'gpu_runtime_enabled',lambda:True)
    monkeypatch.setattr(backend,'eggpu_backend_enabled',lambda:True)
    monkeypatch.setattr(backend,'minimum_spanning_tree_edges',lambda G,weight:edges)
    monkeypatch.setattr(backend,'_graph_context',lambda G:prepared)
    monkeypatch.setattr(backend,'_result_cache_get',lambda *a:None)
    monkeypatch.setattr(backend,'_result_cache_put',lambda *a:None)
    if route.startswith('dispatch_'):
        monkeypatch.setattr(backend,'minimum_spanning_tree',lambda *a,**k:None)
        tree=mst._minimum_spanning_tree_gpu_runtime_dispatch(graph,weight='cost')
    elif route=='backend_fallback':
        monkeypatch.setattr(backend,'_cpp_gpu_mst_tree',lambda *a,**k:None)
        tree=backend.minimum_spanning_tree(graph,weight='cost')
    else:
        lazy=lazy_tree(factory=dict)
        monkeypatch.setattr(backend,'_get_cached_cpp_graph',lambda *a,**k:object())
        monkeypatch.setitem(sys.modules,'cpp_easygraph',SimpleNamespace(
            cpp_gpu_mst_tree=lambda *a,**k:{'tree':lazy,'kernel_seconds':0.}))
        tree=backend.minimum_spanning_tree(graph,weight='cost')
    assert [(u,v,d['cost']) for u,v,d in tree.edges]==edges
    assert_mutation_contract(tree,graph,factory)


def lazy_tree(factory=dict,payload='arrays'):
    tree=eg.Graph();tree.edge_attr_dict_factory=factory
    tree.add_nodes_from(['a','b','c','isolated'])
    if payload=='arrays':
        tree.cache['_lazy_edge_arrays']=(np.array([0,1],np.int32),np.array([1,2],np.int32),
                                         np.array([1.5,2.25]),'cost')
    else:
        tree.cache['_lazy_edges']=[('a','b',{'cost':1.5}),('b','c',{'cost':2.25})]
    return tree


@pytest.mark.parametrize('factory', [dict,OrderedDict,UserDict,ValidatedAttrs])
@pytest.mark.parametrize('payload', ['arrays','edge_list'])
@pytest.mark.parametrize('restore', ['edges_first','adj_first','copy','deepcopy','pickle'])
def test_lazy_result_public_mapping_identity_and_restore(factory,payload,restore):
    original=lazy_tree(factory,payload)
    if restore=='copy':tree=copy.copy(original)
    elif restore=='deepcopy':tree=copy.deepcopy(original)
    elif restore=='pickle':tree=pickle.loads(pickle.dumps(original))
    else:tree=original
    if restore=='adj_first':assert tree['a']['b']['cost']==1.5
    assert [(u,v,d['cost']) for u,v,d in tree.edges]==[('a','b',1.5),('b','c',2.25)]
    assert len(tree.adj['isolated'])==0
    # For an un-copied tree use a separate source to test mutation ownership.
    source=original if tree is not original else eg.Graph()
    assert_mutation_contract(tree,source,factory)
    if tree is not original:assert original['a']['b']['cost']==1.5


def test_class_factory_is_preserved_by_cpu_and_fast_tree(monkeypatch):
    graph=source_graph(graph_type=FactoryGraph)
    monkeypatch.setattr(mst,'gpu_runtime_enabled',lambda:False)
    cpu=eg.minimum_spanning_tree(graph,weight='cost')
    fast=FactoryGraph()
    assert mst._populate_mst_tree_fast(fast,graph,[('a','b',1.5),('b','c',2.25)],weight='cost')
    for tree in (cpu,fast):assert_mutation_contract(tree,graph,ValidatedAttrs)


@pytest.mark.parametrize('factory',[dict,UserDict,ValidatedAttrs])
@pytest.mark.parametrize('payload',['arrays','edge_list'])
def test_edit_returned_lazy_edge_before_any_adjacency_access(factory,payload):
    tree=lazy_tree(factory,payload)
    edges=tree.edges
    before=tree._mutation_generation
    data=edges[0][2]
    data['cost']=9.125
    assert tree._mutation_generation==before+1
    assert tree['a']['b'] is data and tree['b']['a'] is data
    assert tree['a']['b']['cost']==9.125
    assert [(u,v,d['cost']) for u,v,d in tree.edges]==[('a','b',9.125),('b','c',2.25)]
