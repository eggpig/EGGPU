"""CPU-only synthetic evidence and presentation tests; no performance results."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import pytest
import normalize as n
import analysis as a
import render as r


def bind(path):
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return bind(path)


@pytest.fixture
def full_e6(tmp_path,monkeypatch):
    source=tmp_path/'easygraph/example.py';source.parent.mkdir();source.write_text('SYNTHETIC = True\n')
    files={'easygraph/example.py':{k:v for k,v in bind(source).items() if k!='path'}}
    freeze=write(tmp_path/'freeze.json',dict(source_files=files,source_tree_sha256=n.digest(files)))
    epoch={'release_sources':{'easygraph/example.py':bind(source)},'protocol':'synthetic-test-only'}
    eb=write(tmp_path/'epoch.json',epoch)
    outputs={}
    for name in (*n.E6_FUNCTIONS,'candidates','winner'):
        length=32 if name in ('closeness','candidates') else 1 if name=='winner' else 40
        path=tmp_path/(name+'.npy');np.save(path,np.arange(length,dtype=np.float64));outputs[name]=bind(path)
    stages=[dict(function=f,start_seconds=float(i),end_seconds=float(i+1),complete_seconds=1.,
                 metrics=dict(devices_used=[0],coverage_fallback=False)) for i,f in enumerate(n.E6_FUNCTIONS)]
    call=dict(seconds=6.,quality={'pass':True},outputs=outputs,
              trace=dict(stages=stages,candidate_seconds=.1,winner_seconds=.1))
    child=dict(status='PASS',quality_pass=True,outputs_survive_close=True,input_sha256='input',
               weak_projection=dict(n=40,source_vertices_preserved=True,weights_used=False),
               calls={label:copy.deepcopy(call) for label in ('first','warmup','ready')})
    cb=write(tmp_path/'child.json',child)
    base=dict(schema='eggpu.e6.job/v1',system='eggpu',ablate='none',dataset='EP',repeat=0,gpu_count=1,
              epoch=eb,epoch_sha256=n.digest(epoch),input_sha256='input',quality_pass=True,child=cb,
              T_first_seconds=6.,T_ready_seconds=6.)
    proofs={phase:write(tmp_path/(phase+'.json'),dict(base,phase=phase,status='PASS_QUALIFICATION',performance_eligible=False)) for phase in ('tiny','target')}
    cell=dict(id='E6_EP_eggpu',experiment='E6',adapter='e6',role='eggpu',system='eggpu',ablate='none',dataset='EP',gpu_count=1,
              epoch=eb,epoch_sha256=n.digest(epoch),runs=[])
    for i in range(10):
        raw=dict(base,repeat=i,phase='formal',status='PASS',performance_eligible=True,
                 tiny_proof=proofs['tiny'],target_proof=proofs['target'])
        cell['runs'].append(write(tmp_path/f'run{i}.json',raw))
    selection=dict(schema='eggpu.paper.selection.v1',status='FROZEN_EXPLICIT_SELECTION',mode='formal',
                   final_eggpu_release=freeze,input_registry={},required_cells=[cell['id']],cells=[cell])
    sp=tmp_path/'selection.json';write(sp,selection)
    monkeypatch.setattr(n.core,'input_domains',lambda *args:{'EP':dict(n=40,sha256='input')})
    # This fixture isolates receipt normalization. Full protocol-grid rejection
    # is tested independently below; it does not pretend one cell is E6 coverage.
    monkeypatch.setattr(n,'validate_grid',lambda *args:None)
    return sp,selection,base,child


def test_full_formal_chain(full_e6):
    path,_,_,_=full_e6;result=n.normalize(path,'formal')
    assert result['paper_ready'] and result['formal_cells']==1
    assert result['cells'][0]['metrics']['first']['mean']==6
    assert len(result['records'])==10


def test_missing_repetition_never_paper(full_e6):
    path,s,_,_=full_e6;s['cells'][0]['runs'].pop();write(path,s)
    result=n.normalize(path,'formal');assert not result['paper_ready']
    assert result['cells'][0]['metrics']=={}


def test_failure_preserved_no_zero(full_e6):
    path,s,base,_=full_e6
    failed=write(path.parent/'failed.json',dict(base,repeat=9,phase='formal',status='OOM',quality_pass=False))
    s['cells'][0]['runs'][-1]=failed;s['cells'][0]['terminal']={'evidence':failed,'reason':'Explicit OOM after required qualification'}
    write(path,s);result=n.normalize(path,'formal')
    assert result['cells'][0]['terminal_status']=='OOM'
    assert result['records'][-1]['metrics']=={} and not result['paper_ready']


def test_prelaunch_attempt_retained_without_coverage(full_e6):
    path,s,_,_=full_e6
    failed=write(path.parent/'prelaunch.json',dict(status='FAILED',repeat=0,error='environment import failed before epoch'))
    s['attempt_history']=[dict(cell=s['cells'][0]['id'],evidence=failed,reason='Prelaunch attempt; replacement chosen by predeclared retry policy')]
    write(path,s);result=n.normalize(path,'formal')
    assert result['paper_ready'] and len(result['records'])==11
    assert result['records'][-1]['selected'] is False and result['cells'][0]['observed_repetitions']==10


def test_e6_split_gpu_source_binding(tmp_path):
    files={};observed={}
    for name in ('easygraph/x.py','gpu_easygraph/k.cu'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic')
        observed[name]=bind(p);files[name]={k:v for k,v in bind(p).items() if k!='path'}
    epoch=dict(release_sources={'easygraph/x.py':observed['easygraph/x.py']},
               gpu_kernel_sources={'gpu_easygraph/k.cu':observed['gpu_easygraph/k.cu']})
    eb=write(tmp_path/'epoch.json',epoch);cell=dict(adapter='e6',role='eggpu',epoch=eb,epoch_sha256=n.digest(epoch))
    n.bind_identity(n.core.Evidence(tmp_path,[]),cell,dict(epoch=eb,epoch_sha256=n.digest(epoch)),dict(source_files=files))
    del epoch['gpu_kernel_sources'];eb=write(tmp_path/'bad_epoch.json',epoch)
    cell.update(epoch=eb,epoch_sha256=n.digest(epoch))
    with pytest.raises(n.core.EvidenceError):
        n.bind_identity(n.core.Evidence(tmp_path,[]),cell,dict(epoch=eb,epoch_sha256=n.digest(epoch)),dict(source_files=files))


@pytest.mark.parametrize('mutation',['source','proof_output','duplicate','wrong_epoch','wrong_phase','wrong_length','wrong_device'])
def test_evidence_rejections(full_e6,mutation):
    path,s,base,child=full_e6
    if mutation=='source':(path.parent/'easygraph/example.py').write_text('changed')
    elif mutation=='proof_output':(path.parent/'winner.npy').write_bytes(b'changed')
    elif mutation=='duplicate':s['cells'][0]['runs'].append(s['cells'][0]['runs'][0]);write(path,s)
    elif mutation=='wrong_epoch':
        raw=json.loads(Path(s['cells'][0]['runs'][0]['path']).read_text());raw['epoch_sha256']='0'*64
        s['cells'][0]['runs'][0]=write(path.parent/'wrong.json',raw);write(path,s)
    elif mutation=='wrong_phase':
        raw=json.loads(Path(s['cells'][0]['runs'][0]['path']).read_text());raw['phase']='target'
        s['cells'][0]['runs'][0]=write(path.parent/'wrong.json',raw);write(path,s)
    else:
        if mutation=='wrong_length':
            file=path.parent/'short.npy';np.save(file,np.arange(3));child['calls']['first']['outputs']['bfs']=bind(file)
        else:child['calls']['first']['trace']['stages'][0]['metrics']['devices_used']=[]
        cb=write(path.parent/'changed_child.json',child)
        raw=json.loads(Path(s['cells'][0]['runs'][0]['path']).read_text());raw['child']=cb
        s['cells'][0]['runs'][0]=write(path.parent/'wrong.json',raw);write(path,s)
    with pytest.raises(n.core.EvidenceError):n.normalize(path,'formal')


def test_qualification_only(full_e6,tmp_path):
    path,s,base,_=full_e6;s['mode']='qualification';s['cells'][0]['runs']=[base['child']]
    target=write(path.parent/'qual.json',dict(base,repeat=0,phase='target',status='PASS_QUALIFICATION',performance_eligible=False))
    s['cells'][0]['runs']=[target];write(path,s)
    result=n.normalize(path,'qualification');assert not result['paper_ready']
    r.diagnostics(result,tmp_path/'diagnostics')
    assert 'NOT PAPER RESULTS' in (tmp_path/'diagnostics/checks.html').read_text()
    assert not list((tmp_path/'diagnostics').glob('*.pdf'))
    with pytest.raises(n.core.EvidenceError):n.normalize(path,'formal')


def test_exact_cas_mapping(tmp_path):
    p=tmp_path/'object';p.write_bytes(b'original payload');item=bind(p);item['path']='/remote/formal/answer.npy'
    e=n.core.Evidence(tmp_path,[{'from':item['path'],'to':str(p)}])
    assert e.verify(item)==p
    p.write_bytes(b'wrong payload')
    with pytest.raises(n.core.EvidenceError):e.verify(item)


def test_fixed_grid():
    e2=[dict(id=f'{g}-{f}-{p}',experiment='E2',role='eggpu',dataset=g,algorithm=f,gpu_count=p)
        for g in n.E2_GRAPHS for f in n.core.ALGORITHMS for p in (1,2,4,8)]
    n.validate_grid(e2,'formal')
    with pytest.raises(n.core.EvidenceError):n.validate_grid(e2[:-1],'formal')
    with pytest.raises(n.core.EvidenceError):n.validate_grid([dict(experiment='E6',dataset='EP',system='eggpu',ablate='none',gpu_count=1)],'formal')
    n.validate_grid([dict(experiment='E6')],'qualification')


@pytest.mark.parametrize('ex,names,order,commits,queries',[
    ('E3',['US'],'interleaved',20,80),('E4',['US','OR','IN','UK'],'grouped',0,128),
    ('E4',['US','OR','IN','UK'],'once',0,16),('E5',['IN','UK','OR'],'interleaved',24,96)])
def test_frozen_trace(ex,names,order,commits,queries):
    trace=n.state_trace(ex,names,order)
    assert sum(x['kind']=='commit' for x in trace)==commits
    assert sum(x['kind']=='query' for x in trace)==queries
    assert n.state_trace(ex,names,order)==trace


def test_state_actual_counter_boundaries(tmp_path):
    inputs=write(tmp_path/'input.json',dict(schema='state.inputs.v2',qualification_only=True,partitions=1,datasets={'US':{'n':5}}))
    trace=n.state_trace('E3',['US'],'interleaved',tiny=True)
    values=np.zeros(5,dtype=np.float64);file=tmp_path/'values.npz';np.savez_compressed(file,values=values)
    output=dict(blob=bind(file),shape=[5],dtype=values.dtype.str,vector_sha256=hashlib.sha256(memoryview(values).cast('B')).hexdigest())
    checks=[];events=[]
    for event in trace:
        measured=dict(event)
        if event['kind']=='query':
            measured['details']={'metrics':dict(shared_array_new_bytes=5,
                native_dynamic_patch=dict(route='lease_sparse',retirement_policy='lineage',per_device_delta={'0':dict(weight_full_h2d_bytes=1,patch_h2d_bytes=2,cow_d2d_bytes=2,dense_lease_h2d_bytes=100)}),
                resource_management=dict(state_switch_policy='closure',controller_version='state_switch_v2_closure_lru',actions=[]))}
            checks.append(dict(event=event,output=output,survives_session_close=True,
                               quality=dict(passed=True,residual_l1=0,mass_error=0,reference_l1=0)))
        events.append(measured)
    raw=dict(inputs=inputs,datasets=['US'],experiment='E3',arm='D1C1',old_readers='none',execution_complete=True,
             quality_pass=True,continuous_interval=True,sum_of_phases_used_as_total=False,events=events,checks=checks,
             first_complete_workflow_seconds=2,workflow_complete_host_seconds=1,budget=None)
    config=dict(experiment='E3',arm='D1C1',old_readers='none',gpu_uuids=['GPU0'],order='interleaved',route='lease_sparse',policy='closure')
    metrics,_=n.state_payload(n.core.Evidence(tmp_path,[]),raw,config,{},tiny=True)
    assert metrics['maintenance_h2d_bytes']==24 and metrics['cow_d2d_bytes']==16 and metrics['shared_array_new_bytes']==40
    del events[1]['details']['metrics']['shared_array_new_bytes']
    with pytest.raises(n.core.EvidenceError):n.state_payload(n.core.Evidence(tmp_path,[]),raw,config,{},tiny=True)


def synthetic():
    return dict(mode='formal',paper_ready=True,synthetic_fixture=True,source_tree_sha256='SYNTHETIC',
        selection_binding={'path':'synthetic-selection'},selection={},cells=[],records=[],formal_cells=0,expected_cells=0,limitations=[])


def cell(result,identity,definition,metrics):
    row=dict(cell=identity,experiment=definition['experiment'],definition=definition,valid_repetitions=10,
             observed_repetitions=10,expected_repetitions=10,complete_formal=True,terminal_status=None,resolved=True,
             metrics={key:n.core.mean_ci(values) for key,values in metrics.items()})
    result['cells'].append(row);result['formal_cells']+=1;result['expected_cells']+=1
    for i in range(10):result['records'].append(dict(cell=identity,repeat=i,valid=True,status='PASS',
        binding={'path':identity+f'/r{i}'},metrics={k:v[i] for k,v in metrics.items()}))
    return row


def test_scaling_nonmonotonic_and_native_reference():
    result=synthetic()
    for p,value in ((1,10),(2,6),(4,7),(8,3)):
        cell(result,str(p),dict(experiment='E2',role='eggpu',dataset='US',algorithm='bfs',gpu_count=p),
             {'first':[value]*10,'ready':[value/2]*10})
    scaling=a.scaling(result)
    assert not scaling[0]['mean_monotone_nonincreasing']
    assert scaling[0]['points']['1']['speedup']['ci95']==[1,1]
    assert scaling[0]['points']['8']['speedup']['mean']==10/3
    assert scaling[0]['adjacent_comparisons'][1]['time_saved']['mean']==-1


def test_joint_is_not_interaction(monkeypatch):
    result=synthetic();mapped={}
    for arm,value in [('D0C0',10),('D1C0',7),('D0C1',7),('D1C1',5)]:
        cell(result,arm,dict(experiment='E5',datasets=['IN','UK','OR'],gpu_count=1,rho=1e-4,beta=.25,order='interleaved',
             arm=arm,inputs={'sha256':'input'}),{'workflow':[value]*10})
        for i in range(10):mapped[arm+f'/r{i}']=(str(i),arm)
    item=a.joint_effects(result)[0]
    assert item['joint_mean_faster_than_both_single'] and item['interaction'] is None
    monkeypatch.setattr(a,'paired_evidence',lambda result:(mapped,'VERIFIED_SYNTHETIC_TEST'))
    item=a.joint_effects(result)[0]
    assert item['interaction']['mean']==1 and not item['interaction_established']


def test_ratio_groups_not_pseudo_pairs():
    x=[1+i*.1 for i in range(10)];y=[2+i*.1 for i in range(10)]
    assert a.ratio_ci(x,y)==a.ratio_ci(x,list(reversed(y)))
    with pytest.raises(n.core.EvidenceError):a.ratio_ci(x[:9],y)


def test_e4s_attestation_and_domain(tmp_path):
    sources=write(tmp_path/'sources.json',[1,2]);e=n.core.Evidence(tmp_path,[])
    raw=dict(source_sequence=[1,2],sources=sources,checks=[dict(sequence=i,source=s,values=40,
             output_sha256='a'*64,quality=dict(passed=True,exact=True,vertices=40)) for i,s in enumerate([1,2])],
             measurements=dict(first_complete_seconds=2,query_batch_seconds=1,adapter_preparation_seconds=1))
    assert n.e4s_payload(e,raw,dict(batch=2),40)['first']==2
    with pytest.raises(n.core.EvidenceError):n.e4s_payload(e,raw,dict(batch=2),41)
    raw['checks'].pop()
    with pytest.raises(n.core.EvidenceError):n.e4s_payload(e,raw,dict(batch=2),40)


def test_render_vectors_and_no_publish_synthetic(tmp_path):
    result=synthetic()
    for function in n.core.ALGORITHMS:
        for p in (1,2,4,8):
            cell(result,f'{function}{p}',dict(experiment='E2',role='eggpu',dataset='US',algorithm=function,gpu_count=p),
                 {'first':[10/p+i*.001 for i in range(10)],'ready':[2/p+i*.001 for i in range(10)]})
    r.render(result,tmp_path/'preview')
    assert len(list((tmp_path/'preview').glob('*.pdf')))==4
    assert 'SYNTHETIC TEST DATA' in (tmp_path/'preview/Fig4_E2_first_time.svg').read_text()
    assert 'independent-group log delta' in (tmp_path/'preview/captions.md').read_text()
    assert len((tmp_path/'preview/repetitions.csv').read_text().splitlines())==321
    with pytest.raises(n.core.EvidenceError):r.publish(result,tmp_path/'paper',tmp_path/'scratch')
    result['paper_ready']=False
    with pytest.raises(n.core.EvidenceError):r.render(result,tmp_path/'invalid')


def test_managed_artifact_removal_protects_edits(tmp_path):
    out=tmp_path/'previous';out.mkdir();item=out/'figure.svg';item.write_text('synthetic')
    write(out/'artifact_manifest.json',dict(schema='eggpu.paper.artifacts.v1',files={'figure.svg':{k:v for k,v in bind(item).items() if k!='path'}}))
    r.managed_check(out);item.write_text('user annotation')
    with pytest.raises(n.core.EvidenceError):r.managed_remove(out)
    assert item.read_text()=='user annotation'


def test_matched_block_requires_bound_execution(tmp_path):
    result=synthetic();result['selection_binding']={'path':str(tmp_path/'selection.json')}
    jobs=[];outcomes=[]
    for arm in ('D0C0','D1C0','D0C1','D1C1'):
        receipt=write(tmp_path/(arm+'.json'),{'synthetic':True})
        block={'repeat':0,'inputs':'same frozen input'}
        spec=write(tmp_path/(arm+'_spec.json'),dict(paired_block=block,paired_block_sha256=n.digest(block),receipt=receipt['path'],arm=arm))
        jobs.append(dict(id=arm,command=['python','guard','--spec',spec['path']],bindings=[spec]))
        outcomes.append(dict(id=arm,outcome='PASS',receipt=receipt))
        result['records'].append(dict(binding=receipt,repeat=0))
    schedule=write(tmp_path/'schedule.json',dict(jobs=jobs))
    execution=write(tmp_path/'execution.json',dict(plan=schedule,completed=outcomes))
    result['selection']['pairing']=dict(schedule=schedule,execution=execution)
    mapping,status=a.paired_evidence(result);assert len(mapping)==4 and status.startswith('EXPLICIT')
    wrong=write(tmp_path/'wrong_execution.json',dict(plan={'sha256':'wrong'},completed=outcomes))
    result['selection']['pairing']['execution']=wrong
    with pytest.raises(n.core.EvidenceError):a.paired_evidence(result)


def test_e6_stage_and_batch_figures(tmp_path):
    result=synthetic()
    for graph in ('EP','LJ','OR'):
        choices=[('eggpu',g,1) for g in ('none','structural','closeness','bfs','wcc')]
        choices += [('eggpu','none',p) for p in (2,4,8)]
        choices += [('cugraph_composite','none',1),('cugraph_igraph','none',1),('igraph','none',0)]
        for system,ablate,p in choices:
            metrics={}
            for timing in ('first','ready'):
                for stage in ('structural','ranking','closeness','bfs','wcc','preparation_and_handoff'):
                    metrics[timing+'__'+stage]=[1.]*10
                metrics[timing]=[6.]*10
            cell(result,f'{graph}-{system}-{ablate}-{p}',dict(experiment='E6',role='eggpu' if system=='eggpu' else 'external',
                 dataset=graph,system=system,ablate=ablate,gpu_count=p),metrics)
    for graph in ('US','OR','IN','UK'):
        for q in (32,128,512):
            for variant in ('eggpu','c2graph_pm','kgraph_sparse' if graph=='US' else 'kgraph_dense'):
                cell(result,f'{graph}-{q}-{variant}',dict(experiment='E4S',dataset=graph,batch=q,variant=variant),
                     {'first':[q/4]*10,'ready':[q/8]*10,'preparation':[q/8]*10})
    out=tmp_path/'extended_preview';out.mkdir();captions=[]
    r.render_e6(result,out,captions);r.render_e4s(result,out,captions)
    assert len(list(out.glob('*.pdf')))==10
    assert 'SYNTHETIC TEST DATA' in (out/'Fig9_E6_EP_first.svg').read_text()
    assert any('not a repeated Q-by-n archive' in x[1] for x in captions)


def test_two_generations_only_after_verified_stage(tmp_path,monkeypatch):
    counter=[0]
    def fake_render(result,out):
        out.mkdir();counter[0]+=1;file=out/'artifact.txt';file.write_text(str(counter[0]))
        write(out/'artifact_manifest.json',dict(schema='eggpu.paper.artifacts.v1',files={'artifact.txt':{k:v for k,v in bind(file).items() if k!='path'}}))
    monkeypatch.setattr(r,'render',fake_render)
    result={'paper_ready':True}
    for _ in range(3):r.publish(result,tmp_path/'published',tmp_path/'scratch')
    assert sorted(x.name for x in (tmp_path/'published').iterdir())==['current','previous']
    assert (tmp_path/'published/current/artifact.txt').read_text()=='3'
    assert (tmp_path/'published/previous/artifact.txt').read_text()=='2'
    (tmp_path/'published/current/artifact.txt').write_text('user edit')
    with pytest.raises(n.core.EvidenceError):r.publish(result,tmp_path/'published',tmp_path/'scratch')
    assert (tmp_path/'published/previous/artifact.txt').read_text()=='2'


def test_state_main_controls_and_vectors_render(tmp_path):
    result=synthetic()
    for graph in ('US','IN','OR'):
        for rho in (0.,1e-4,1e-2):
            for arm,value in [('D0C1',10.),('D1C1',7.)]:
                cell(result,f'E3-{graph}-{rho}-{arm}',dict(experiment='E3',datasets=[graph],gpu_count=1,
                    old_readers='none',locality='uniform',rho=rho,arm=arm),
                    {'workflow':[value]*10,'maintenance_h2d_bytes':[value*1e9]*10,'cow_d2d_bytes':[1e8]*10})
        for arm in ('D0C1','D1C1'):
            cell(result,f'PR-{graph}-{arm}',dict(experiment='E3_PR_CONTROL',dataset=graph,arm=arm),{'workflow':[2.]*10})
    for beta in (0.,.25,1.):
        for arm,value in [('D1C0',12.),('D1C1',8.)]:
            cell(result,f'E4-{beta}-{arm}',dict(experiment='E4',datasets=['US','OR','IN','UK'],gpu_count=1,
                order='interleaved',beta=beta,arm=arm),{'workflow':[value]*10,'shared_array_new_bytes':[value*1e9]*10,'freed_bytes':[1e9]*10})
    for arm,value in [('D0C0',10.),('D1C0',7.),('D0C1',7.),('D1C1',5.)]:
        cell(result,'E5-'+arm,dict(experiment='E5',datasets=['IN','UK','OR'],gpu_count=1,rho=1e-4,beta=.25,arm=arm),
            {'workflow':[value]*10,'maintenance_h2d_bytes':[value*1e9]*10})
    out=tmp_path/'state_preview';out.mkdir();captions=[]
    r.render_state(result,out,captions,{'joint':[]})
    assert len(list(out.glob('*.pdf')))==12
    for path in out.glob('*.svg'):
        assert 'SYNTHETIC TEST DATA' in path.read_text()
    assert any('not an estimate of all reload' in text for _,text in captions)
