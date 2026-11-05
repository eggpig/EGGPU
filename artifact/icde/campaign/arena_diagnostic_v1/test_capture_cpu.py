from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys

import numpy as np
import pytest

import capture
import worker as diagnostic
import prepare


class LazyState:
    def __init__(self):
        self.host_view_seconds={}
    def __getattr__(self,key):
        raise AssertionError('Observing must not materialize '+key)


class FakeOriginal:
    def __init__(self):
        self.state=LazyState()
        self.graph=SimpleNamespace(r2_state=self.state,host_prepare_seconds=0.)
        self.engine=SimpleNamespace(graphs={'g':self.graph},counters={'h2d_bytes':0})
        self.session=SimpleNamespace(_backend=SimpleNamespace(backend=SimpleNamespace(session=self.engine),
            initialization_seconds=.1),query=self.query)
        self.contract={'route':'CPU fixture'}
        self.count=0;self.closed=False;self.outputs=[]
    def query(self,gid,algorithm,params):
        self.count+=1
        if algorithm=='pr' and 'incoming' not in vars(self.state):
            self.state.incoming=object()
            self.state.host_view_seconds['incoming']=.7
            self.graph.host_prepare_seconds+=.7
            self.engine.counters['h2d_bytes']+=12
        values=np.arange(2,dtype=np.float64)+self.count
        values.flags.writeable=False
        self.outputs.append(values)
        return SimpleNamespace(values=values,seconds=.01,backend_metrics={'iterations':7,'result_cache_used':False})
    def close(self):self.closed=True


def test_snapshot_reads_only_existing_scalar_state():
    original=FakeOriginal();first=capture.snapshot(original.session)
    assert first['materialized_host_views']==[] and first['host_view_seconds']=={}
    assert 'incoming' not in vars(original.state)


def test_capture_returns_identical_complete_owner_and_exactly_one_query():
    original=FakeOriginal();probe=capture.CaptureSequence(original)
    value=probe.query(dict(index=0,algorithm='pr',source=None))
    assert value is original.outputs[0] and original.count==1
    row=probe.captures[0]
    assert row['host_view_seconds_delta']=={'incoming':.7}
    assert row['counters_delta']=={'h2d_bytes':12}
    assert row['backend_metrics']['iterations']==7
    probe.close();assert original.closed


def test_frozen_sixteen_request_executor_keeps_trace_and_all_owned_outputs():
    frozen=diagnostic.load_base()
    original=FakeOriginal()
    outputs,record=capture.diagnostic_execute(frozen.execute_sequence,lambda *unused:original,
        SimpleNamespace(),{'n':2,'sources':[1,2,3]})
    assert original.count==16 and original.closed
    assert len(outputs)==len(record['arena_diagnostics'])==16
    assert [x['algorithm'] for x in record['calls']]==['bfs','sssp','wcc','pr']*4
    assert [x['source'] for x in record['calls'] if x['algorithm']=='bfs']==[1,2,3,1]
    assert all(x.flags.owndata and not x.flags.writeable for x in outputs)
    assert not record['performance_eligible'] and not record['formal_measurement']
    assert sum(x['host_view_seconds_delta'].get('incoming',0) for x in record['arena_diagnostics'])==.7
    assert not record['sum_of_query_times_used_as_total']


def test_wrapper_binds_own_epoch_and_routes_child_back_to_wrapper(tmp_path):
    base=SimpleNamespace(__file__=str(diagnostic.BASE/'sequence16_worker.py'),PROTOCOL='old')
    base.configure=lambda args:args
    base.epoch=lambda args:({'files':{},'protocol':base.PROTOCOL},'old')
    base.child_command=lambda *args:[sys.executable,'-B',base.__file__,'--child']
    base.execute_sequence=lambda *args,**kwargs:None
    base.job=lambda args:None;base.source_child=lambda args:None
    base.adapter=SimpleNamespace(EGGPUSequence=None)
    base.shared=SimpleNamespace(binding=lambda p:{'path':str(p)},digest_json=lambda x:json.dumps(x,sort_keys=True))
    diagnostic.install(base)
    args=SimpleNamespace(phase='target',repeat=0,system='eggpu')
    assert base.configure(args) is args
    with pytest.raises(ValueError,match='no formal'):
        base.configure(SimpleNamespace(phase='formal',repeat=0,system='eggpu'))
    data,_=base.epoch(args)
    assert data['protocol']==diagnostic.PROTOCOL and data['formal_repetitions']==0
    assert set(data['files'])=={'arena_diagnostic_worker','arena_diagnostic_capture'}
    assert base.child_command(None,None,None,None)[2]==str(Path(diagnostic.__file__).resolve())


def test_numpy_scalar_metrics_are_serialized_after_capture_without_output_copy():
    values={'iterations':np.int64(7),'duration':np.float64(.5),'flag':np.bool_(True)}
    assert json.loads(json.dumps(capture.json_ready(values)))=={'iterations':7,'duration':.5,'flag':True}


def test_deployment_transform_keeps_every_resource_flag_environment_and_timeout(tmp_path):
    base=diagnostic.HERE/'fixtures/sequence_plan.json'
    plan=json.loads(base.read_text())
    originals=[j for j in plan['jobs'] if prepare.option(j['command'],'system')=='eggpu']
    transformed=prepare.transform(originals,tmp_path/'out',diagnostic.HERE/'worker.py')
    assert len(transformed)==5
    known=set()
    for original,new in zip(originals,transformed):
        assert new.get('environment',{})==original.get('environment',{})
        assert new['timeout']==original['timeout']
        assert new['phase']==original['phase']
        assert set(new['depends_on'])<=known
        known.add(new['id'])
        for flag in ['cpu-set','gpu-uuid','budget-mib','source','native-dso','reference-cache',
                     'input-root','phase','repeat','lock-file','timeout']:
            assert prepare.option(new['command'],flag)==prepare.option(original['command'],flag)
        if '--tiny-proof' in new['command']:
            assert prepare.option(new['command'],'tiny-proof') in {j['receipt'] for j in transformed[:2]}
