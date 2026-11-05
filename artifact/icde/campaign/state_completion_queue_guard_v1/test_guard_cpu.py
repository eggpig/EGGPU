from pathlib import Path
from types import SimpleNamespace
import copy
import json
import os
import sys

import pytest
import derive


RUNNER = Path(__file__).resolve().parents[1] / 'state_protocol_completion_v1'


def inputs(tiny=False):
    return dict(qualification_only=tiny, datasets={'US':dict(n=10, m=30,
        manifest_files=[dict(bytes=100)], projections={'bfs':dict(files=[dict(bytes=200)]),
                                                     'wcc':dict(files=[dict(bytes=300)])})})


def spec(arm='Rebuild', tiny=False, action='run'):
    return dict(phase='tiny' if tiny else 'target', datasets=['US'],
                experiment='E3', order='interleaved', arm=arm, action=action)


def trace(*unused):
    return [dict(kind='commit', dataset='US')] + [dict(kind='query',dataset='US')]*80


def test_rebuild_reserves_all_versions_plus_outputs_oracle_and_temporary():
    r=derive.capacity(spec(),inputs(),trace)
    assert r['rebuild_temporary_payload_upper_bytes']==20*600
    assert r['min_free_bytes']==80*10*8*2+10*8+20*600+(2<<30)
    old=derive.capacity(spec('D0C1'),inputs(),trace)
    assert r['worker_required_host_free_gib']-old['worker_required_host_free_gib']==22*600/(1<<30)


def test_tiny_rebuild_uses_two_versions():
    r=derive.capacity(spec(tiny=True),inputs(tiny=True),trace)
    assert r['rebuild_temporary_payload_upper_bytes']==1200
    assert r['rebuild_versions']==2


def test_pilot_reserves_no_unexecuted_vectors_or_rebuild_files():
    r=derive.capacity(spec('D1C1',action='pilot'),inputs(),trace)
    assert r['complete_query_outputs']==0
    assert r['min_free_bytes']==2<<30


def test_input_phase_mismatch_fails():
    with pytest.raises(ValueError,match='phase'):
        derive.capacity(spec(),inputs(tiny=True),trace)


def test_derive_preserves_original_contract_and_binds_incremental_chain(tmp_path):
    campaign=tmp_path/'experimental/campaign';campaign.mkdir(parents=True)
    for name in ['run_campaign_queue_v8.py','owned_process_supervisor_v2.py',
                 'completed_job_audit_maintenance_v2.py','completed_output_maintenance_incremental_v1.py',
                 'incremental_completed_output_v1.py']:
        (campaign/name).write_text('# CPU fixture; never executed\n')
    inp=tmp_path/'inputs.json';derive.save(inp,inputs())
    s=spec();s.update(id='target_US_Rebuild',receipt=str(tmp_path/'raw/receipt.json'),
        inputs=derive.bind(inp),dependencies=[])
    path=tmp_path/'job.json';derive.save(path,s)
    original=dict(id=s['id'],command=[sys.executable,'guard.py','guard','--spec',str(path),'--execute-gpu'],
        receipt=s['receipt'],depends_on=[],bindings=[derive.bind(path)],environment={'A':'B'},
        timeout=3600,expected_statuses=['PASS_STATE_GPU_QUALIFICATION'])
    base=tmp_path/'base.json';derive.save(base,dict(jobs=[original],environment={'FROZEN':'1'}))
    args=SimpleNamespace(campaign=campaign,base_plan=base,base_sha256=derive.bind(base)['sha256'],
        expected_jobs=1,runner=RUNNER,output=tmp_path/'derived',python=Path(sys.executable))
    result=derive.derive(args)
    plan=derive.read(result['plan']['path']);job=plan['jobs'][0]
    assert {k:v for k,v in job.items() if k not in ('phase','disk_guard','maintenance')}==original
    assert plan['environment']=={'FROZEN':'1'}
    assert job['phase']=='target' and job['disk_guard']['rebuild_temporary_payload_upper_bytes']==12000
    assert job['maintenance']['expected_statuses']==['PASS_INCREMENTAL_HARDLINKED_OUTPUTS']
    assert derive.bind(result['registry']['path']) in job['maintenance']['bindings']
    registry=derive.read(result['registry']['path'])
    assert registry['schema']=='completed.output.incremental.registry.v1'
    assert registry['queue_states']==[str(args.output/'queue/state.json')]
    assert not (args.output/'queue').exists()
    assert result['workers_launched']==result['formal_measurements']==0
    with pytest.raises(ValueError,match='overwrite'):
        derive.derive(args)
