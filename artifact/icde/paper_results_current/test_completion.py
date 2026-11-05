"""CPU-only synthetic admission/render tests; no experimental performance claim."""
from pathlib import Path
from collections import Counter
import copy
import json
import numpy as np
import pytest
import normalize as n
import protocol
import archive_gate
import analysis
import render
import synthetic_fixtures as f
import prepare_selection


def test_full_latest_grids():
    state=protocol.state_cells();e6=protocol.e6_cells()
    assert Counter(r['experiment'] for r in state)=={'E3':300,'E4':160,'E3R':20}
    assert len(e6)==45 and Counter(r['dataset'] for r in e6)=={'EP':15,'LJ':15,'OR':15}
    n.validate_grid(state+e6,'formal')


def test_scaffold_is_pending_not_a_selection(tmp_path):
    plan=prepare_selection.scaffold({'E3','E4','E3R','E4S','E6'})
    assert plan['expected_counts']==dict(E3=300,E4=160,E3R=20,E4S=36,E6=45)
    assert len(plan['cells'])==561 and all(not r['runs'] for r in plan['cells'])
    path=tmp_path/'pending.json';f.write(path,plan)
    with pytest.raises(ValueError):n.normalize(path,'formal')


def test_selection_index_cannot_shrink_or_redefine_grid():
    plan=prepare_selection.scaffold({'E6'})
    with pytest.raises(ValueError):prepare_selection.bind_index(plan,dict(status='FROZEN_EXPLICIT_RECEIPT_INDEX',cells=[]))
    rows=copy.deepcopy(plan['cells']);rows[0]['gpu_count']=8
    with pytest.raises(ValueError):prepare_selection.bind_index(plan,dict(status='FROZEN_EXPLICIT_RECEIPT_INDEX',cells=rows))


@pytest.mark.parametrize('experiment',['E3','E4','E3R','E6'])
@pytest.mark.parametrize('damage',['missing','duplicate'])
def test_incomplete_or_duplicate_grid_rejected(experiment,damage):
    cells=[r for r in protocol.state_cells()+protocol.e6_cells() if r['experiment']==experiment]
    if damage=='missing':cells.pop()
    else:cells.append(copy.deepcopy(cells[0]))
    with pytest.raises(ValueError):n.validate_grid(cells,'formal')


@pytest.mark.parametrize('backend',['igraph','cugraph_composite'])
def test_both_2x2_groups_have_four_cells(backend):
    rows=[r for r in protocol.e6_cells() if r['dataset']=='EP' and r['system']=='eggpu' and r['gpu_count']==1 and
          (r['ablate']=='none' or r['coverage_backend']==backend and r['ablate'] in ('structural','closeness','structural_closeness'))]
    assert {r['ablate'] for r in rows}=={'none','structural','closeness','structural_closeness'}


@pytest.fixture
def e6(tmp_path):return f.e6_selection(tmp_path/'synthetic_e6')


def test_full_grid_source_bound_normalization(e6):
    path,_=e6;result=n.normalize(path,'formal')
    assert result['paper_ready'] and result['synthetic_fixture'] and result['formal_cells']==45
    assert len(result['records'])==450
    assert all(x['valid_repetitions']==10 for x in result['cells'])
    assert len(analysis.e6_factorial(result))==6
    assert all(x['complete'] for x in analysis.e6_factorial(result))


@pytest.mark.parametrize('mutation',['missing_repeat','duplicate_repeat','quality','epoch','phase','schema','backend','worker_source','vector_bytes','vector_shape','lifetime','quality_detail','timed','stage_device','stage_backend','proof_protocol'])
def test_formal_gates(e6,mutation,tmp_path):
    path,selection=e6;cell=selection['cells'][0]
    if mutation=='missing_repeat':
        cell['runs'].pop();f.write(path,selection);result=n.normalize(path,'formal')
        assert not result['paper_ready'] and result['formal_cells']==44
        with pytest.raises(ValueError):render.render(result,tmp_path/'no_figure')
        assert not (tmp_path/'no_figure').exists();return
    if mutation=='duplicate_repeat':cell['runs'].append(cell['runs'][0]);f.write(path,selection)
    else:
        raw=json.loads(Path(cell['runs'][0]['path']).read_text())
        if mutation=='quality':raw['quality_pass']=False
        elif mutation=='epoch':raw['epoch_sha256']='0'*64
        elif mutation=='phase':raw['phase']='target'
        elif mutation=='schema':raw['schema']='eggpu.e6.job/v1'
        elif mutation=='backend':raw['coverage_backend']='igraph'
        elif mutation=='proof_protocol':
            proof=json.loads(Path(raw['target_proof']['path']).read_text());proof['protocol']='old'
            raw['target_proof']=f.write(path.parent/'bad_proof.json',proof)
        elif mutation=='worker_source':
            epoch=json.loads(Path(raw['epoch']['path']).read_text());fake=path.parent/'fake_worker.py';fake.write_text('synthetic wrong worker')
            epoch['files_e6']['e6_worker.py']=f.bind(fake);eb=f.write(path.parent/'bad_epoch.json',epoch)
            raw.update(epoch=eb,epoch_sha256=n.digest(epoch));cell.update(epoch=eb,epoch_sha256=n.digest(epoch))
        else:
            child=json.loads(Path(raw['child']['path']).read_text());call=child['calls']['first']
            if mutation=='vector_bytes':Path(call['outputs']['bfs']['path']).write_bytes(b'changed')
            elif mutation=='vector_shape':
                fp=path.parent/'short.npy';np.save(fp,np.zeros(2));call['outputs']['bfs']=f.bind(fp)
            elif mutation=='lifetime':child['outputs_survive_close']=False
            elif mutation=='quality_detail':call['quality']['vectors']['constraint']['nan_mask_equal']=False
            elif mutation=='timed':call['timed']=False
            elif mutation=='stage_device':call['trace']['stages'][0]['metrics']['devices_used']=[]
            elif mutation=='stage_backend':call['trace']['stages'][0]['metrics']['coverage_backend']='igraph'
            raw['child']=f.write(path.parent/'bad_child.json',child)
        cell['runs'][0]=f.write(path.parent/'bad_run.json',raw);f.write(path,selection)
    with pytest.raises((ValueError,FileNotFoundError)):n.normalize(path,'formal')


def test_qualification_diagnostics_never_publish(e6,tmp_path):
    path,s=e6;cell=s['cells'][0];raw=json.loads(Path(cell['runs'][0]['path']).read_text())
    s['mode']='qualification';cell['runs']=[raw['target_proof']];s['cells']=[cell];s['required_cells']=[cell['id']];f.write(path,s)
    result=n.normalize(path,'qualification');assert not result['paper_ready']
    render.diagnostics(result,tmp_path/'diagnostics')
    assert not list((tmp_path/'diagnostics').glob('*.pdf'))
    with pytest.raises(ValueError):render.publish(result,tmp_path/'published',tmp_path/'scratch')


def test_synthetic_never_promotes_current(e6,tmp_path):
    result=n.normalize(e6[0],'formal')
    with pytest.raises(ValueError):render.publish(result,tmp_path/'published',tmp_path/'scratch')
    assert not (tmp_path/'published').exists()


def test_synthetic_marker_cannot_be_stripped(e6):
    path,s=e6;s['synthetic_fixture']=False;f.write(path,s)
    with pytest.raises(ValueError):n.normalize(path,'formal')


@pytest.mark.parametrize('ex,arm',[('E3R','D1C1'),('E4','D1Cnone'),('E3','Rebuild')])
def test_new_state_payload(tmp_path,ex,arm):
    raw,config=f.state_payload(tmp_path,ex,arm)
    metrics,_=n.state_payload(n.core.Evidence(tmp_path,[]),raw,config,{},tiny=True)
    count=len(raw['checks']);assert metrics['maintenance_h2d_bytes']==count*3
    assert metrics['commit_rebuild_freed_bytes']==(6 if arm=='Rebuild' else 0)
    if ex=='E3R':assert count==16 and not any(x['kind']=='commit' for x in raw['events'])


@pytest.mark.parametrize('mutation',['old_protocol','output_sha','negative_quality','semantic_version','no_retention','zero_release_progress','rebuild_missing','trace_missing'])
def test_state_semantic_archive_gates(tmp_path,mutation):
    ex,arm=('E3','Rebuild') if mutation=='rebuild_missing' else ('E4','D1Cnone') if mutation=='no_retention' else ('E3R','D1C1')
    raw,c=f.state_payload(tmp_path,ex,arm)
    query=next(x for x in raw['events'] if x['kind']=='query')
    if mutation=='old_protocol':c['protocol']='old'
    elif mutation=='output_sha':raw['checks'][0]['output_sha256']='0'*64
    elif mutation=='negative_quality':next(x for x in raw['checks'] if x['event']['function']=='weighted_pr')['quality']['residual_l1']=-1.
    elif mutation=='semantic_version':query['details']['semantic_snapshot']='other@0'
    elif mutation=='no_retention':query['details']['metrics']['resource_management']['after_retention']['per_device'][0]['managed_resident_bytes']=1
    elif mutation=='zero_release_progress':query['details']['metrics']['resource_management']['actions']=[dict(freed_bytes=0,prediction_error_bytes=0,capacity_progress=True)]
    elif mutation=='rebuild_missing':raw['rebuild_control']=False
    elif mutation=='trace_missing':raw['events'].pop()
    with pytest.raises(ValueError):n.state_payload(n.core.Evidence(tmp_path,[]),raw,c,{},tiny=True)


def test_e4s_full_archive_decoded(tmp_path):
    raw,cell,_,_=f.e4s_archive(tmp_path)
    assert archive_gate.verify(n.core.Evidence(tmp_path,[]),raw,cell,formal=True)['vectors']==1


@pytest.mark.parametrize('mutation',['hash_only','missing_archive','missing_object','decoded_bytes','quality','wrong_epoch','missing_vector','wrong_dtype','bad_maintenance','wrong_result','false_measurement_boundary','qualification_as_formal'])
def test_e4s_archive_gates(tmp_path,mutation):
    raw,cell,sidecar,post=f.e4s_archive(tmp_path);entry=cell['recoverable_outputs'][0]
    if mutation=='hash_only':cell['accept_execution_hash_attestation']=True
    elif mutation=='missing_archive':cell['recoverable_outputs']=[]
    elif mutation=='missing_object':Path(sidecar['vectors'][0]['object']['path']).unlink()
    elif mutation=='decoded_bytes':
        obj=tmp_path/'replaced.npz';np.savez_compressed(obj,reference=np.array([0.,2.,np.inf],dtype='<f8'))
        sidecar['vectors'][0]['object']=f.bind(obj)
    elif mutation=='quality':raw['quality_pass']=False
    elif mutation=='wrong_epoch':sidecar['epoch_sha256']='wrong'
    elif mutation=='missing_vector':sidecar['vectors']=[]
    elif mutation=='wrong_dtype':sidecar['vectors'][0]['dtype']='<i8'
    elif mutation=='bad_maintenance':post['original_maintenance_status']='FAIL'
    elif mutation=='wrong_result':sidecar['original_result']['sha256']='0'*64
    elif mutation=='false_measurement_boundary':sidecar['measured_intervals_unchanged']=False
    elif mutation=='qualification_as_formal':raw['phase']='target'
    entry['result']=f.write(tmp_path/'result.json',raw)
    entry['archive']=f.write(tmp_path/'archive.json',sidecar);post['archive_receipt']=entry['archive']
    entry['maintenance']=f.write(tmp_path/'maintenance.json',post)
    with pytest.raises((ValueError,FileNotFoundError)):archive_gate.verify(n.core.Evidence(tmp_path,[]),raw,cell,formal=True)


def test_watermarked_latest_e6_render(e6,tmp_path):
    result=n.normalize(e6[0],'formal');out=tmp_path/'SYNTHETIC_NOT_RESULTS';out.mkdir()
    captions=[];render.render_e6(result,out,captions)
    assert len(list(out.glob('*2x2*.svg')))==4
    assert len(list(out.glob('*.svg')))==14
    for path in out.glob('*.svg'):assert 'SYNTHETIC TEST DATA' in path.read_text()
    assert all('hash attestation' not in text for _,text in captions)
    comparisons=analysis.analyze(result)
    assert len(comparisons['e6_factorial'])==6


def test_upstream_sources_unchanged():
    for relative in protocol.PROVENANCE:protocol.checked_source(relative)
