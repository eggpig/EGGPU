"""Synthetic CPU-only continuous-clock evidence checks; not measured performance."""
from pathlib import Path
import copy,json,hashlib
import numpy as np
import pytest
import normalize as n
import sequence as s

def bind(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,value):p.write_text(json.dumps(value));return bind(p)
@pytest.fixture
def fixture(tmp_path):
    cell=dict(id='sequence_US_eggpu',experiment='E1_SEQUENCE',adapter='sequence',system='eggpu',role='eggpu',dataset='US',gpu_count=1)
    domain=dict(sha256='synthetic_input',n=4,sources=[0,1,2],directed=False)
    trace=[dict(index=r*4+i,round=r,algorithm=f,source=domain['sources'][s.SOURCE_INDICES[r]] if f in ('bfs','sssp') else None,source_index=s.SOURCE_INDICES[r] if f in ('bfs','sssp') else None) for r in range(4) for i,f in enumerate(s.FUNCTIONS)]
    output=tmp_path/'array.npy';np.save(output,np.full(4,.25))
    quality=dict(pass_=True,complete=True,dtype='float64',owning=True,immutable=True,exact_reference_match=True,unreachable_match=True,partition_equivalent=True,residual_l1=0,mass_error=0,reference_l1=0);quality['pass']=quality.pop('pass_')
    epoch=dict(protocol=s.PROTOCOL,workflow=list(s.FUNCTIONS)*4,source_indices=list(s.SOURCE_INDICES),formal_repetitions=10,warmups=0)
    eb=write(tmp_path/'epoch.json',epoch);cell['epoch_sha256']=n.digest(epoch)
    child=dict(status='PASS',quality_pass=True,execution_complete=True,outputs_survive_close=True,protocol=s.PROTOCOL,system='eggpu',epoch_sha256=n.digest(epoch),input_sha256=domain['sha256'],continuous_interval=True,sum_of_query_times_used_as_total=False,warmup_requests=0,requests=16,workflow_start_ns=100,workflow_complete_ns=3400,T_workflow_seconds=3300/1e9,calls=[dict(row,begin_ns=200+i*200,end_ns=300+i*200,seconds=100/1e9,quality=quality,output=bind(output)) for i,row in enumerate(trace)])
    wb=write(tmp_path/'workflow.json',child)
    raw=dict(protocol=s.PROTOCOL,system='eggpu',request_count=16,quality_pass=True,dataset='US',directed=False,input_sha256=domain['sha256'],trace=trace,workflow=wb,epoch_sha256=n.digest(epoch),epoch=eb,T_workflow_seconds=child['T_workflow_seconds'],phase='formal',status='PASS_FORMAL')
    raw['qualification']={kind:write(tmp_path/(kind+'.json'),dict(raw,status='PASS_QUALIFICATION',phase=kind,dataset='TINY_U' if kind=='tiny' else 'US')) for kind in ['tiny','target']}
    return tmp_path,cell,domain,epoch,raw,child,n.core.Evidence(tmp_path,[])

def test_actual_prefixes_preserve_preparation_time(fixture):
    p,c,d,epoch,raw,child,e=fixture;metrics,_=s.run(e,raw,c,epoch,{'US':d},'formal')
    assert metrics['prefix_01']==200/1e9 and metrics['workflow']==3300/1e9
    assert metrics['workflow']>sum(row['seconds'] for row in child['calls'])

@pytest.mark.parametrize('field,value',[('continuous_interval',False),('sum_of_query_times_used_as_total',True),('warmup_requests',1),('outputs_survive_close',False),('requests',15)])
def test_invalid_workflow_contract_rejected(fixture,field,value):
    p,c,d,epoch,raw,child,e=fixture;child[field]=value;raw['workflow']=write(p/'changed.json',child)
    with pytest.raises(n.core.EvidenceError):s.run(e,raw,c,epoch,{'US':d},'formal')

def test_changed_source_order_rejected(fixture):
    p,c,d,epoch,raw,child,e=fixture;raw['trace'][4]['source']=2
    with pytest.raises(n.core.EvidenceError):s.run(e,raw,c,epoch,{'US':d},'formal')

def test_overlapping_request_clock_rejected(fixture):
    p,c,d,epoch,raw,child,e=fixture;child['calls'][1]['begin_ns']=100;raw['workflow']=write(p/'changed.json',child)
    with pytest.raises(n.core.EvidenceError):s.run(e,raw,c,epoch,{'US':d},'formal')

def test_tampered_complete_output_rejected(fixture):
    p,c,d,epoch,raw,child,e=fixture;(p/'array.npy').write_bytes(b'tampered')
    with pytest.raises(n.core.EvidenceError):s.run(e,raw,c,epoch,{'US':d},'formal')

def test_missing_formal_proof_rejected(fixture):
    p,c,d,epoch,raw,child,e=fixture;raw['qualification'].pop('target')
    with pytest.raises((KeyError,n.core.EvidenceError)):s.run(e,raw,c,epoch,{'US':d},'formal')

def test_qualification_is_not_formal(fixture):
    p,c,d,epoch,raw,child,e=fixture;raw['status']='PASS_QUALIFICATION'
    with pytest.raises(n.core.EvidenceError):s.run(e,raw,c,epoch,{'US':d},'formal')

def test_fixed_six_cells_required():
    rows=[dict(experiment='E1_SEQUENCE',dataset=g,system=system,gpu_count=1) for g in ('US','IN','TW') for system in ('eggpu','cugraph')]
    n.validate_grid(rows,'formal')
    with pytest.raises(n.core.EvidenceError):n.validate_grid(rows[:-1],'formal')

def test_sequence_vector_preview_is_watermarked(tmp_path):
    cells=[]
    for graph in ('US','IN','TW'):
        for system in ('eggpu','cugraph'):
            metrics={f'prefix_{i:02d}':{'mean':float(i),'ci95':[i-.1,i+.1]} for i in range(1,17)}
            cells.append(dict(experiment='E1_SEQUENCE',definition=dict(dataset=graph,system=system),complete_formal=True,valid_repetitions=10,metrics=metrics))
    result=dict(cells=cells,synthetic_fixture=True);captions=[]
    s.render(result,tmp_path,captions)
    assert (tmp_path/'Fig3_continuous_sequence.pdf').is_file()
    text=(tmp_path/'Fig3_continuous_sequence.svg').read_text()
    assert 'SYNTHETIC TEST DATA' in text and 'Cumulative time (s)' in text and len(captions)==1
