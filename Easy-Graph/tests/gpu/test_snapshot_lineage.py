"""CPU mocks: semantic retirement permissions, independent of GPU performance."""
import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from easygraph.gpu import _dynamic as dy
from easygraph.gpu._bridge import QueryResult
from easygraph.gpu._native import UnifiedNativeSession


@pytest.fixture
def factory(tmp_path, monkeypatch):
    script=Path(__file__).resolve().parents[2]/'scripts/gpu/qualification_inputs.py'
    spec=importlib.util.spec_from_file_location('lineage_cpu_inputs',script)
    qi=importlib.util.module_from_spec(spec);spec.loader.exec_module(qi)
    raw=qi.canonical(3,np.array([[0,1],[1,2],[2,0]],dtype=np.int32),np.array([1.,2.,3.]))
    paths={name:qi.manifest(raw,tmp_path/name,name,1)[0]['manifest'] for name in ('A','B')}
    def initialize(self,cpp,manifests,*,devices=(0,),budgets=None,**kwargs):
        self.cpp=cpp;self.devices=list(devices);self.closed=False;self._lock=threading.RLock()
        self.graphs={name:object() for name in manifests};self._manifest_paths=dict(manifests)
        self._manifest_documents={name:json.loads(Path(path).read_text()) for name,path in manifests.items()}
        self.metadata={name:doc['graph'] for name,doc in self._manifest_documents.items()}
        self.snapshot_hashes={name:dy.sha(path) for name,path in manifests.items()}
        self.nodes={name:np.arange(3) for name in manifests};self._domains={}
        self.controller=SimpleNamespace(budgets=budgets);self._owners=[];self._releases=[];self._rebuilds=0
    def inventory(self):
        return {'per_device':[{'device_id':0,'owners':[dict(o) for o in self._owners],
                'counters':{'dynamic.abi_version':1}}]}
    def query(self,name,algorithm,parameters=None):
        gid=self.metadata[name]['id'];owner=f'{gid}/{algorithm}'
        if not any(o['owner_id']==owner for o in self._owners):
            self._owners.append(dict(owner_id=owner,kind='view',graph_id=gid,active_leases=0));self._rebuilds+=1
        values=np.array([1.,2.,3.]);values.flags.writeable=False
        return QueryResult(name,algorithm,values,0.,{})
    def release(self,owner_ids,**kwargs):
        self._releases.extend(owner_ids);self._owners=[o for o in self._owners if o['owner_id'] not in owner_ids]
        return dict(owner_ids=list(owner_ids))
    monkeypatch.setattr(UnifiedNativeSession,'__init__',initialize)
    monkeypatch.setattr(UnifiedNativeSession,'resource_inventory',inventory)
    monkeypatch.setattr(UnifiedNativeSession,'query',query)
    monkeypatch.setattr(UnifiedNativeSession,'release',release)
    monkeypatch.setattr(UnifiedNativeSession,'describe',lambda self,name:SimpleNamespace(name=name))
    monkeypatch.setattr(UnifiedNativeSession,'close',lambda self:setattr(self,'closed',True))
    cpp=SimpleNamespace(cpp_egmgpu_reset_allocation_observation=lambda devices:None,
                        cpp_egmgpu_load_partition_manifest=lambda *args,**kwargs:object())
    def create(**kwargs):return dy.DynamicNativeSession(cpp,paths,**kwargs)
    return create,tmp_path


@pytest.mark.parametrize('route',dy.ROUTES)
def test_independent_graphs_survive_switch_under_budget(factory,route):
    create,_=factory;s=create(route=route,budgets={0:4096})
    try:
        first=s.query('A','sssp');s.query('B','sssp');s.query('A','sssp')
        assert s._snapshot_lineage=={'A':'A','B':'B'}
        assert s._rebuilds==2 and not s._releases
    finally:s.close()
    np.testing.assert_array_equal(first.values,[1,2,3])
    assert not first.values.flags.writeable


def test_legacy_diagnostic_releases_unrelated_graph(factory):
    create,_=factory;s=create(retirement_policy='all_other_views')
    try:
        s.query('A','sssp');s.query('B','sssp');s.query('A','sssp')
        assert s._rebuilds==3 and len(s._releases)==2
    finally:s.close()


@pytest.mark.parametrize('retain',[False,True])
def test_commit_inherits_lineage_and_old_host_remains_queryable(factory,retain):
    create,tmp=factory;s=create(route='lease_sparse')
    try:
        old=s.query('A','sssp');s.query('B','sssp')
        if retain:s.retain_snapshot('A')
        receipt=s.commit_weights('A','A1',[(0,1,7.)],tmp/'commits')
        assert receipt['snapshot_lineage']=='A' and s._snapshot_lineage['A1']=='A'
        s.query('A1','sssp')
        a_id=s.metadata['A']['id'];b_id=s.metadata['B']['id']
        assert (f'{a_id}/sssp' in s._releases)==(not retain)
        assert f'{b_id}/sssp' not in s._releases
        s.release_snapshot('A');s.query('A','sssp')
        assert 'A1/sssp' in s._releases
        np.testing.assert_array_equal(old.values,[1,2,3])
    finally:s.close()


def test_invalid_commit_does_not_publish_lineage(factory):
    create,tmp=factory;s=create()
    try:
        before=dict(s._snapshot_lineage)
        with pytest.raises(ValueError):s.commit_weights('A','bad',[(0,0,-1.)],tmp/'commits')
        assert s._snapshot_lineage==before and 'bad' not in s.graphs
        assert not (tmp/'commits/bad').exists()
    finally:s.close()


def test_native_import_failure_does_not_publish_lineage(factory,monkeypatch):
    create,tmp=factory;s=create()
    def fail(*args,**kwargs):raise RuntimeError('mock native importer failure')
    monkeypatch.setattr(s.cpp,'cpp_egmgpu_load_partition_manifest',fail)
    try:
        before=dict(s._snapshot_lineage)
        with pytest.raises(RuntimeError,match='mock native importer'):
            s.commit_weights('A','bad',[(0,0,4.)],tmp/'commits')
        assert s._snapshot_lineage==before and 'bad' not in s.graphs
        assert not (tmp/'commits/bad').exists()
    finally:s.close()


def test_active_owner_and_retained_snapshot_are_never_retired(factory):
    create,tmp=factory;s=create()
    try:
        s.query('A','sssp');s._owners[0]['active_leases']=1
        s.commit_weights('A','A1',[(0,0,4.)],tmp/'commits');s.query('A1','sssp')
        assert not s._releases
    finally:s.close()


def test_unknown_policy_has_no_environment_mutation(factory,monkeypatch):
    create,_=factory;monkeypatch.setenv('EGGPU_DYNAMIC_WEIGHT_ROUTE','sentinel')
    with pytest.raises(ValueError,match='retirement'):create(retirement_policy='unknown')
    import os
    assert os.environ['EGGPU_DYNAMIC_WEIGHT_ROUTE']=='sentinel'
