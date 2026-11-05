"""Synthetic CPU-only fixtures. No fixture is a measured research result."""
from pathlib import Path
import hashlib
import json
import statistics
import pytest
import e1_summary as e1


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return bind(path)


def bind(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def quality(algorithm):
    return dict(pass_=True, complete=True, dtype='float64', residual_l1=1e-10,
                mass_error=0., reference_l1=1e-9, exact_reference_match=True,
                unreachable_match=True, partition_equivalent=True, **{'pass': True})


class Fixture:
    def __init__(self, root, mode='formal', count=1, external=1):
        self.root = root; self.mode = mode; self.count = count
        payload = root/'SYNTHETIC_CPU_TEST_DO_NOT_PUBLISH.bin'
        payload.write_bytes(b'SYNTHETIC CPU CONTRACT TEST; NOT GRAPH EXPERIMENT DATA')
        self.payload = bind(payload)
        source = {'easygraph/gpu/session.py': {k: self.payload[k] for k in ('bytes', 'sha256')}}
        freeze = write(root/'freeze.json', dict(source_files=source, source_tree_sha256=e1.digest(source)))
        records = {}
        for name in e1.DATASETS:
            folder=root/'inputs'/name
            meta = write(folder/'canonical.json', dict(num_nodes=7, num_entries=3, directed=name!='US'))
            sources = write(folder/'sources.json', dict(primary_sources=[0,1,2]))
            files=dict(offsets=self.payload,indices=self.payload,weights=self.payload,canonical=meta,sources=sources)
            receipt=dict(status='PASS_CANONICAL_INPUT',dataset=name,files=files)
            receipt_binding=write(folder/'receipt.json',receipt)
            records[name]=dict(receipt_path=receipt_binding['path'],receipt_sha256=receipt_binding['sha256'],receipt=receipt)
        registry=write(root/'registry.json',dict(status='ALL_SEVEN_CANONICAL_READY',records=records))
        self.selection=dict(schema='eggpu.e1.selection.v1',status='FROZEN_EXPLICIT_SELECTION',mode=mode,
                            final_eggpu_release=freeze,input_registry=registry,systems=[])
        self.domains=e1.input_domains(e1.Evidence(root),registry)
        for i in range(external+1):
            name='eggpu' if i==0 else 'baseline'+str(i)
            epoch=dict(protocol='test-explicit-contract',files={'worker':self.payload},
                       release_sources=source if i==0 else {},system=name)
            epoch_binding=write(root/name/'epoch.json',epoch)
            schema='eggpu.e1.sg.job.v1' if count==1 else 'eggpu.native.job.v1'
            entries={algorithm:dict(epoch=epoch_binding,protocol=epoch['protocol'],schema=schema) for algorithm in e1.ALGORITHMS}
            self.selection['systems'].append(dict(id=name,label=name,role='eggpu' if i==0 else 'external',
                system=name,gpu_count=count,algorithms=entries,not_applicable={},runs=[]))
        self.path=root/'selection.json'

    def spec(self, name):
        return next(s for s in self.selection['systems'] if s['id']==name)

    def add(self, name, dataset='US', algorithm='bfs', repeat=0, timings=None, failure=None):
        spec=self.spec(name);entry=spec['algorithms'][algorithm];domain=self.domains[dataset]
        epoch=json.loads(Path(entry['epoch']['path']).read_text());epoch_sha=e1.digest(epoch)
        folder=self.root/'runs'/name/dataset/algorithm/str(repeat)
        raw=dict(schema=entry['schema'],protocol=entry['protocol'],system=name,dataset=dataset,algorithm=algorithm,
            gpu_count=self.count,repeat=repeat,phase=self.mode if self.mode=='formal' else 'target',
            status='PASS' if self.mode=='formal' else 'PASS_QUALIFICATION',quality_pass=True,
            performance_eligible=self.mode=='formal',epoch=entry['epoch'],epoch_sha256=epoch_sha,
            input_sha256=domain['sha256'],directed=domain['directed'],sources=[])
        for kind in ('tiny','target'):
            proof=dict(status='PASS_QUALIFICATION',phase=kind,quality_pass=True,epoch_sha256=epoch_sha,
                system=name,algorithm=algorithm,gpu_count=self.count,directed=domain['directed'],epoch=entry['epoch'],
                dataset=dataset if kind=='target' else 'TINY_D' if domain['directed'] else 'TINY_U',
                input_sha256=domain['sha256'])
            raw[kind+'_proof']=write(folder/(kind+'.json'),proof)
        if failure:
            raw.update(status=failure,quality_pass=False,performance_eligible=False,error='Synthetic explicit '+failure)
        else:
            sources=domain['sources'] if algorithm in ('bfs','sssp') else [None]
            for index,source in enumerate(sources):
                calls={}
                for metric in ('first','warmup','ready'):
                    value=timings(repeat,index,metric) if timings else (1.+repeat*.01+index)*(.5 if metric=='ready' else 1.)
                    calls[metric]=dict(seconds=value,quality=quality(algorithm),output=self.payload,
                        continuous_interval=dict(status='PASS_CONTINUOUS_MONOTONIC_INTERVAL'))
                child=dict(status='PASS',quality_pass=True,source=source,input_sha256=domain['sha256'],calls=calls)
                child_binding=write(folder/f'source_{index}.json',child)
                raw['sources'].append(dict(source=source,result=child_binding,calls=calls))
        for kind in ('tiny','target'):
            proof=json.loads(Path(raw[kind+'_proof']['path']).read_text());proof['sources']=raw['sources']
            raw[kind+'_proof']=write(Path(raw[kind+'_proof']['path']),proof)
        ref=write(folder/'result.json',raw);spec['runs'].append(ref)
        return ref

    def cell(self,name,dataset='US',algorithm='bfs',count=10,**kwargs):
        return [self.add(name,dataset,algorithm,i,**kwargs) for i in range(count)]

    def save(self):
        write(self.path,self.selection);return self.path

    def summary(self):
        return e1.summarize(self.save(),self.mode)

    def mutate(self,reference,change):
        path=Path(reference['path']);value=json.loads(path.read_text());change(value)
        new=write(path,value);reference.clear();reference.update(new)


def test_three_sources_are_one_repeat_and_denominator_stays_28(tmp_path):
    f=Fixture(tmp_path)
    f.cell('eggpu',timings=lambda r,s,m: [1,4,10][s]+r)
    f.cell('baseline1',timings=lambda r,s,m: [2,8,20][s]+r)
    summary=f.summary();cell=next(c for c in summary['cells'] if c['system_id']=='eggpu' and c['qualified'])
    expected=[5+r for r in range(10)]
    assert cell['metrics']['first']['mean']==statistics.mean(expected)
    assert cell['metrics']['first']['sd']==pytest.approx(statistics.stdev(expected))
    assert cell['metrics']['first']['n']==10
    assert summary['totals'][0]['planned_cells']==28
    assert summary['totals'][0]['paired_cells']==1
    assert summary['totals'][0]['mean_leads']==1
    assert summary['totals'][0]['leads_over_28']==1/28
    assert len(summary['source_rows'])==60
    per_graph=next(x for x in summary['breakdowns'] if x['group']=='dataset' and x['name']=='US' and x['metric']=='first')
    per_function=next(x for x in summary['breakdowns'] if x['group']=='algorithm' and x['name']=='bfs' and x['metric']=='first')
    assert per_graph['planned_cells']==4 and per_graph['mean_leads']==1
    assert per_function['planned_cells']==7 and per_function['valid_coverage']==1/7


def test_fastest_is_minimum_external_mean_not_fastest_single_run(tmp_path):
    f=Fixture(tmp_path,external=2)
    f.cell('eggpu',timings=lambda r,s,m:2.)
    f.cell('baseline1',timings=lambda r,s,m:3.)
    f.cell('baseline2',timings=lambda r,s,m:.1 if r==0 else 6.)
    rows=[r for r in f.summary()['comparisons'] if r['paired']]
    assert all(r['fastest_qualified_measured']=='baseline1' and r['speedup']==1.5 for r in rows)


def test_first_and_ready_are_never_combined(tmp_path):
    f=Fixture(tmp_path)
    f.cell('eggpu',timings=lambda r,s,m:4. if m=='first' else 1.)
    f.cell('baseline1',timings=lambda r,s,m:2.)
    totals=f.summary()['totals']
    assert totals[0]['mean_leads']==0 and totals[1]['mean_leads']==1


def test_welch_is_independent_and_exploratory():
    a=[10.+i for i in range(10)];b=[9.99+i for i in range(10)]
    result=e1.welch_difference(a,b)
    assert result['degrees_of_freedom']==pytest.approx(18.)
    assert result['ci95'][0]<0<result['ci95'][1]
    assert 'not multiplicity-adjusted' in result['interpretation']


def test_nine_repeats_have_no_formal_mean_or_latex(tmp_path):
    f=Fixture(tmp_path);f.cell('eggpu',count=9);f.cell('baseline1')
    summary=f.summary();cell=next(c for c in summary['cells'] if c['system_id']=='eggpu' and c['dataset']=='US' and c['algorithm']=='bfs')
    assert 'metrics' not in cell and not summary['paper_ready']
    e1.render(summary,tmp_path/'review')
    assert not list((tmp_path/'review').glob('*.tex'))


@pytest.mark.parametrize('failure',['FAILED','OOM','INDEX_UNSUPPORTED','TIMEOUT','FAIL_QUALITY'])
def test_failure_is_not_an_automatic_win(tmp_path,failure):
    f=Fixture(tmp_path);f.cell('eggpu');ref=f.add('baseline1',failure=failure)
    f.spec('baseline1')['terminal_cells']=[dict(evidence=ref,reason='Explicit terminal synthetic test')]
    summary=f.summary();row=next(r for r in summary['comparisons'] if r['dataset']=='US' and r['algorithm']=='bfs')
    assert not row['paired'] and not row['mean_leads'] and row['strict_full_baseline_coverage']
    assert summary['totals'][0]['mean_leads']==0 and summary['totals'][0]['leads_over_28']==0


def test_valid_comparison_keeps_missing_baseline_visible(tmp_path):
    f=Fixture(tmp_path,external=2);f.cell('eggpu');f.cell('baseline1',timings=lambda r,s,m:10.)
    row=next(r for r in f.summary()['comparisons'] if r['paired'])
    assert row['applicable_external']==2 and row['qualified_external']==1
    assert row['mean_leads'] and not row['strict_full_baseline_coverage']


@pytest.mark.parametrize('field,value',[('epoch_sha256','wrong'),('input_sha256','wrong'),('directed',True),
    ('gpu_count',8),('phase','target'),('performance_eligible',False),('repeat',10),('schema','unknown')])
def test_mixed_or_unqualified_identity_is_rejected(tmp_path,field,value):
    f=Fixture(tmp_path);ref=f.add('eggpu');f.mutate(ref,lambda r:r.update({field:value}))
    with pytest.raises(e1.EvidenceError):f.summary()


def test_duplicate_attempt_cannot_select_the_faster_one(tmp_path):
    f=Fixture(tmp_path);f.add('eggpu');f.add('eggpu')
    with pytest.raises(e1.EvidenceError,match='Duplicate repeat'):f.summary()


def test_wrong_source_order_is_rejected(tmp_path):
    f=Fixture(tmp_path);ref=f.add('eggpu');f.mutate(ref,lambda r:r['sources'].reverse())
    with pytest.raises(e1.EvidenceError,match='source IDs'):f.summary()


def test_changed_output_bytes_are_rejected(tmp_path):
    f=Fixture(tmp_path);f.add('eggpu')
    Path(f.payload['path']).write_bytes(b'tampered')
    with pytest.raises(e1.EvidenceError,match='SHA changed'):f.summary()


def test_reported_top_level_mean_is_not_trusted(tmp_path):
    f=Fixture(tmp_path);ref=f.add('eggpu');f.mutate(ref,lambda r:r.update(T_first_seconds=1000.))
    with pytest.raises(e1.EvidenceError,match='three-source arithmetic'):f.summary()


def test_qualification_checks_are_marked_and_cannot_enter_paper(tmp_path):
    f=Fixture(tmp_path,mode='qualification');f.add('eggpu');f.add('baseline1')
    summary=f.summary();assert not summary['paper_ready']
    with pytest.raises(e1.EvidenceError,match='paper/current'):e1.render(summary,tmp_path/'paper')
    e1.render(summary,tmp_path/'checks')
    assert 'QUALIFICATION' in (tmp_path/'checks/checks.html').read_text()
    assert not list((tmp_path/'checks').glob('*.tex'))
    summary['paper_ready']=True
    e1.render(summary,tmp_path/'checks')
    assert not list((tmp_path/'checks').glob('*.tex'))
    with pytest.raises(e1.EvidenceError,match='mode differs'):e1.summarize(f.path,'formal')


def test_complete_formal_matrix_can_export_tables(tmp_path):
    f=Fixture(tmp_path)
    for name in ('eggpu','baseline1'):
        for dataset in e1.DATASETS:
            for algorithm in e1.ALGORITHMS:
                f.cell(name,dataset,algorithm,timings=lambda r,s,m:1.+r*.01+s)
    summary=f.summary();assert summary['paper_ready']
    assert summary['totals'][0]['paired_cells']==28 and summary['totals'][0]['mean_leads']==0
    e1.render(summary,tmp_path/'synthetic_formal_export')
    assert len(list((tmp_path/'synthetic_formal_export').glob('*.tex')))==2


@pytest.mark.parametrize('schema',['eggpu.e1.sg.author.v1','eggpu.e1.c2graph_p.v1'])
def test_author_clock_requires_continuous_first_interval(tmp_path,schema):
    f=Fixture(tmp_path);f.spec('baseline1')['algorithms']['sssp']['schema']=schema
    if schema=='eggpu.e1.sg.author.v1':
        entry=f.spec('baseline1')['algorithms']['sssp']
        epoch=json.loads(Path(entry['epoch']['path']).read_text());epoch.pop('files')
        epoch.update({name:f.payload for name in ('worker','common_worker','clock_helper','launcher','libstdcxx','setup')})
        epoch.update(build={'binary':f.payload},runtime_linker={'libraries':{'lib':f.payload}})
        entry['epoch']=write(tmp_path/'author_epoch.json',epoch)
    ref=f.add('baseline1',algorithm='sssp');raw=json.loads(Path(ref['path']).read_text())
    child_ref=raw['sources'][0]['result'];child=json.loads(Path(child_ref['path']).read_text())
    child['calls']['first']['continuous_interval']['status']='PHASE_SUM'
    raw['sources'][0]['result']=write(Path(child_ref['path']),child)
    for kind in ('tiny','target'):
        proof=json.loads(Path(raw[kind+'_proof']['path']).read_text());proof['sources']=raw['sources']
        raw[kind+'_proof']=write(Path(raw[kind+'_proof']['path']),proof)
    new=write(Path(ref['path']),raw);ref.clear();ref.update(new)
    with pytest.raises(e1.EvidenceError,match='continuous clock'):f.summary()


def test_author_epoch_accepts_its_actual_top_level_bindings(tmp_path):
    f=Fixture(tmp_path);entry=f.spec('baseline1')['algorithms']['bfs']
    entry['schema']='eggpu.e1.sg.author.v1'
    epoch=json.loads(Path(entry['epoch']['path']).read_text());epoch.pop('files')
    epoch.update({name:f.payload for name in ('worker','common_worker','clock_helper','launcher','libstdcxx','setup')})
    epoch.update(build={'binary':f.payload},runtime_linker={'libraries':{'lib':f.payload}})
    entry['epoch']=write(tmp_path/'author_epoch.json',epoch)
    f.add('baseline1')
    assert any(c.get('valid_repeats')==1 for c in f.summary()['cells'])


def test_gum_rounds_and_embedded_result_are_supported(tmp_path):
    f=Fixture(tmp_path,count=8)
    f.spec('baseline1')['algorithms']['bfs']['schema']=e1.GUM_SCHEMA
    ref=f.add('baseline1');raw=json.loads(Path(ref['path']).read_text())
    raw.update(formal=True,status='PASS_FORMAL')
    for item in raw['sources']:
        child=json.loads(Path(item['result']['path']).read_text())
        child.update(T_first_seconds=2.,T_ready_seconds=.5,contiguous_canonical_to_first_result=True,
            continuous_first={'status':'PASS_CONTINUOUS_MONOTONIC_INTERVAL'},rounds=[
                dict(round=i,quality=quality('bfs'),raw_output=f.payload) for i in range(3)])
        item['result']=child
    for kind in ('tiny','target'):
        proof=json.loads(Path(raw[kind+'_proof']['path']).read_text());proof['sources']=raw['sources']
        raw[kind+'_proof']=write(Path(raw[kind+'_proof']['path']),proof)
    new=write(Path(ref['path']),raw);ref.clear();ref.update(new)
    summary=f.summary();cell=next(c for c in summary['cells'] if c['system_id']=='baseline1' and c['selected_repeats'])
    assert cell['runs'][0]['first']==2. and cell['runs'][0]['ready']==.5


def test_cugraph_mg_source_zero_and_no_performance_flag(tmp_path):
    f=Fixture(tmp_path,count=8)
    f.spec('baseline1')['algorithms']['pr']['schema']='cugraph-mg-host-csr-complete-v1'
    ref=f.add('baseline1',algorithm='pr');raw=json.loads(Path(ref['path']).read_text())
    raw.pop('performance_eligible');raw.update(status='PASS_FORMAL')
    item=raw['sources'][0];item['source']=0
    child=json.loads(Path(item['result']['path']).read_text());child['source']=0
    item['result']=write(Path(item['result']['path']),child)
    for kind in ('tiny','target'):
        proof=json.loads(Path(raw[kind+'_proof']['path']).read_text());proof['sources']=raw['sources']
        raw[kind+'_proof']=write(Path(raw[kind+'_proof']['path']),proof)
    new=write(Path(ref['path']),raw);ref.clear();ref.update(new)
    assert any(c.get('valid_repeats')==1 for c in f.summary()['cells'])


def test_changed_qualification_only_output_is_rejected(tmp_path):
    f=Fixture(tmp_path);ref=f.add('eggpu');raw=json.loads(Path(ref['path']).read_text())
    proof_ref=raw['tiny_proof'];proof=json.loads(Path(proof_ref['path']).read_text())
    child=json.loads(Path(proof['sources'][0]['result']['path']).read_text())
    proof_output=tmp_path/'only_qualification.bin';proof_output.write_bytes(b'proof only')
    child['calls']['first']['output']=bind(proof_output)
    proof['sources'][0]['result']=write(tmp_path/'proof_child.json',child)
    raw['tiny_proof']=write(Path(proof_ref['path']),proof)
    new=write(Path(ref['path']),raw);ref.clear();ref.update(new)
    proof_output.write_bytes(b'qualification changed after measurement')
    with pytest.raises(e1.EvidenceError,match='SHA changed'):f.summary()


def test_pin_explicit_paths_and_preserve_qualification_mode(tmp_path):
    import bind_selection
    f=Fixture(tmp_path,mode='qualification');f.add('eggpu')
    f.selection['status']='DRAFT_EXPLICIT_SELECTION'
    # An explicit path without a hash is pinned; no directory search is needed.
    f.selection['final_eggpu_release'].pop('sha256')
    draft=f.save();output=tmp_path/'elsewhere/locked.json'
    bind_selection.pin(draft,output)
    assert json.loads(output.read_text())['mode']=='qualification'
    assert not e1.summarize(output,'qualification')['paper_ready']
    bind_selection.pin(draft,output)


def test_pin_refuses_stale_hash_and_overwrite(tmp_path):
    import bind_selection
    f=Fixture(tmp_path,mode='qualification');f.selection['status']='DRAFT_EXPLICIT_SELECTION'
    f.selection['input_registry']['sha256']='0'*64
    with pytest.raises(e1.EvidenceError,match='SHA changed'):
        bind_selection.pin(f.save(),tmp_path/'frozen.json')
    assert not (tmp_path/'frozen.json').exists()
