"""CPU capacity-evidence checks; all inputs and failures are synthetic fixtures."""
from pathlib import Path
import json
import pytest
import capacity as c
import normalize as n
from test_paper_results import bind, write


def test_device_oom_needs_exact_allocation_evidence():
    raw={'status':'FAILED','quality_pass':False}
    line='RuntimeError: cudaErrorMemoryAllocation: out of memory'
    c.classify('OOM',raw,line,line)
    for line in ('EasyGraph GPU: gpu error occurred','std::bad_alloc','No space left on device',
                 'Index count exceeds int32'):
        with pytest.raises(n.core.EvidenceError):c.classify('OOM',raw,line,line)


def test_pending_and_complete_are_distinct():
    c.classify('Pending',None)
    with pytest.raises(n.core.EvidenceError):c.classify('Pending',{'status':'FAILED'})
    with pytest.raises(n.core.EvidenceError):c.classify('Complete',{'status':'FAILED','quality_pass':True})
    with pytest.raises(n.core.EvidenceError):c.classify('Complete',{'status':'PASS_QUALIFICATION'})
    c.classify('Complete',{'status':'PASS_QUALIFICATION','quality_pass':True})


def test_sampled_memory_binds_exact_attempt_and_is_not_allocation_size(tmp_path):
    receipt=write(tmp_path/'receipt.json',{'status':'FAILED'})
    samples=write(tmp_path/'memory.json',dict(kind='device_memory_samples',execution_receipt=receipt,
        samples=[dict(used_bytes={'GPU-TEST':100}),dict(used_bytes={'GPU-TEST':150})],covers_complete_task=False))
    e=n.core.Evidence(tmp_path,[])
    assert c.sampled_peak(e,samples,receipt)==(150,'partial sampling')
    assert c.sampled_peak(e,None,receipt)==(None,'not collected')
    with pytest.raises(n.core.EvidenceError):c.sampled_peak(e,samples,dict(receipt,sha256='wrong'))


def test_sampled_jsonl_summary_and_uuid_are_verified(tmp_path):
    receipt=write(tmp_path/'receipt.json',{'status':'FAILED'})
    raw=tmp_path/'memory.jsonl';raw.write_text('\n'.join(json.dumps(dict(used_bytes={'GPU-TEST':x})) for x in (100,150))+'\n')
    record=dict(kind='device_memory_samples',execution_receipt=receipt,samples_file=bind(raw),
                covers_complete_task=True,formal_measurement=False,performance_eligible=False,
                sample_count=2,busiest_device_peak_bytes=150)
    e=n.core.Evidence(tmp_path,[]);samples=write(tmp_path/'memory.json',record)
    assert c.sampled_peak(e,samples,receipt)==(150,'complete task')
    for change in ({'formal_measurement':True},{'sample_count':1},{'busiest_device_peak_bytes':300}):
        with pytest.raises(n.core.EvidenceError):c.sampled_peak(e,write(tmp_path/'changed.json',dict(record,**change)),receipt)


def test_normalized_failure_has_no_time_or_speedup(tmp_path,monkeypatch):
    source=tmp_path/'source.py';source.write_text('synthetic only')
    files={'source.py':{k:v for k,v in bind(source).items() if k!='path'}}
    freeze=write(tmp_path/'freeze.json',dict(source_files=files,source_tree_sha256=n.digest(files)))
    ep=write(tmp_path/'epoch.json',{'adapter':bind(source)})
    epoch=json.loads(Path(ep['path']).read_text());epoch_sha=n.digest(epoch)
    receipt=write(tmp_path/'failure.json',dict(status='FAILED',quality_pass=False,phase='target',
        input_sha256='synthetic-input',epoch=ep,epoch_sha256=epoch_sha))
    log=tmp_path/'failure.log';log.write_text('cudaErrorMemoryAllocation: out of memory')
    cell=dict(system='cuGraph',algorithm='sssp',gpu_count=1,datasets=['TW'],scope='single_task',
        adapter='sg',role='external',epoch=ep,epoch_sha256=epoch_sha,receipt=receipt,status='OOM',
        classification=dict(evidence=bind(log),needle=log.read_text(),stage='graph construction',failed_allocation_bytes=1024))
    selection=write(tmp_path/'selection.json',dict(schema='eggpu.capacity.selection/v1',source_freeze=freeze,
        input_registry={},cells=[cell],synthetic_fixture=True))
    monkeypatch.setattr(n.core,'input_domains',lambda *args:{'TW':dict(sha256='synthetic-input',
        identity={'files':{x:{'bytes':v} for x,v in [('offsets',10),('indices',20),('weights',40)]}})})
    result=c.normalize(selection['path']);row=result['rows'][0]
    assert row['timing_mean'] is None and row['speedup'] is None
    assert row['sampled_peak_vram_bytes'] is None and row['failed_allocation_bytes']==1024
    assert row['canonical_csr_bytes']==70
    c.render(result,tmp_path/'preview')
    assert 'SYNTHETIC TEST DATA' in (tmp_path/'preview/E1_capacity_p1.svg').read_text()
    assert '0.0' not in (tmp_path/'preview/capacity.csv').read_text()


def test_resident_capacity_renders_separately(tmp_path):
    rows=[]
    for arm in ('D1C0','D1C1'):
        for beta in (0,.25,.5,.75,1):
            rows.append(dict(system='EGGPU',arm=arm,gpus=1,scope='multi_graph_state',
                budget_fraction=beta,status='Complete',datasets='US,OR,IN,UK'))
    c.render({'rows':rows,'synthetic_fixture':True},tmp_path/'preview')
    assert not list((tmp_path/'preview').glob('E1*'))
    text=(tmp_path/'preview/E4_residency_capacity_p1.svg').read_text()
    assert 'Budget fraction' in text and 'SYNTHETIC TEST DATA' in text
