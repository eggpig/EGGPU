"""CPU-only runner-v3 binding and new complete 80-job WCC qualification plan."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys

P=C=OLD=Q=PARENT_PLAN=PARENT_REGISTRY=BUILD=None
SOURCE='6cce329e95f6fb10831d698845677579cfd05bfbf50c899132d03ed20cecab81'


def load(p):return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''):h.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def save(p,x):
    with Path(p).open('x') as stream:json.dump(x,stream,indent=2);stream.write('\n')
def digest(x):return hashlib.sha256(json.dumps(x,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def replace(command,flag,value):command[command.index(flag)+1]=str(value)


def validate_parent_plan(parent):
    recipe=load(Path(__file__).with_name('qualification_recipe.json'))
    expected={row['id']:row for row in recipe['jobs']}
    observed={}
    for job in parent['jobs']:
        name='wide3_'+job['id'].removeprefix('wide_')
        assert name not in observed and name in expected,'Unknown or duplicate WCC qualification cell'
        observed[name]=job;row=expected[name]
        assert all(job[key]==row[key] for key in ('phase','gpu_count','algorithm','dataset','timeout','expected_statuses')),'Parent qualification dimensions differ'
        assert ['wide3_'+x.removeprefix('wide_') for x in job['depends_on']]==row['depends_on'],'Parent prerequisite order/scope differs'
        assert job.get('maintenance'),'Every parent qualification needs its original source-bound maintenance'
        for flag,value in row['algorithm_flags'].items():
            assert flag in job['command'] and job['command'][job['command'].index(flag)+1]==value,'Parent algorithm intervention differs: '+flag
    assert set(observed)==set(expected),'Exactly the complete 80-cell recipe is required'
    return True


def main():
    import argparse
    global P,C,OLD,Q,PARENT_PLAN,PARENT_REGISTRY,BUILD
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','candidate','parent-candidate','output','parent-plan','parent-registry','build-receipt'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--controller-python',type=Path,required=True)
    parser.add_argument('--cpu-set',required=True)
    args=parser.parse_args()
    P=args.campaign.resolve();C=args.candidate.resolve();OLD=args.parent_candidate.resolve();Q=args.output.resolve()
    PARENT_PLAN=args.parent_plan.resolve();PARENT_REGISTRY=args.parent_registry.resolve();BUILD=args.build_receipt.resolve()
    published=Path(__file__).resolve().parents[4]
    assert not C.is_relative_to(published) and not Q.is_relative_to(published),'Use fresh candidate/output directories outside the published checkout'
    assert not Q.exists() and not (C/'freeze.json').exists(),'Never overwrite frozen candidates or qualification attempts'
    old=load(OLD/'freeze.json');build=load(BUILD);cpu=load(C/'cpu_receipt.json')
    assert old['source_tree_sha256']==build['source_tree_sha256']==SOURCE
    assert bind(build['native_dso']['path'])==build['native_dso'],'Actual freshly source-bound DSO differs'
    for rel,expected in old['source_files'].items():
        actual=bind(OLD/'Easy-Graph'/rel)
        assert {k:actual[k] for k in ('bytes','sha256')}==expected,rel
    assert cpu['status']=='PASS_WCC_RUNNER_CPU_ONLY' and cpu['tests']['passed']==50 and not cpu['gpu_executed']
    files={p.name:bind(p) for p in sorted((C/'runner').iterdir()) if p.is_file() and (p.suffix=='.py' or p.name=='historical_profile.json')}
    assert set(files)==set(cpu['runner_files'])
    for name,b in files.items():assert {k:b[k] for k in ('bytes','sha256')}==cpu['runner_files'][name],name
    changed=[]
    for name,b in files.items():
        before=bind(OLD/'runner'/name)
        if any(before[k]!=b[k] for k in ('bytes','sha256')):changed.append(name)
    assert changed==['native_worker.py'],'Only the requested runner configuration/audit changes are allowed'
    Q.mkdir()
    frozen=dict(schema='eggpu.candidate.freeze.v1',status='FROZEN_CPU_VALIDATED_GPU_UNQUALIFIED',
        source_tree_sha256=SOURCE,source_root=str(OLD/'Easy-Graph'),source_files=old['source_files'],
        parent_source_freeze=bind(OLD/'freeze.json'),native_dso=build['native_dso'],
        build=bind(BUILD),compilation_performed=False,changed_source_files=[],
        changed_runner_files=changed,runner_files=files,cpu_evidence=bind(C/'cpu_receipt.json'),
        gpu_executed=False,formal_measurements=0,
        scope='Runner-only label CPU placement correction plus effective-profile/actual-metrics audit; original 6cce source and explicitly supplied freshly bound DSO reused exactly. No previous GPU PASS transfers.')
    save(C/'freeze.json',frozen)
    parent_path=PARENT_PLAN;parent=load(parent_path)
    assert len(parent['jobs'])==80 and parent['source_tree_sha256']==SOURCE
    validate_parent_plan(parent)
    registry=load(PARENT_REGISTRY)
    registry['queue_states']=list(dict.fromkeys(registry['queue_states']+[str(Q/'queue/state.json')]))
    registry['parent_registry']=bind(PARENT_REGISTRY)
    registry['scope']='Previous completed immutable queues plus new runner-v3 complete80 queue; no failed job replacement.'
    save(Q/'queue_registry.json',registry);registry_binding=bind(Q/'queue_registry.json')
    common=[bind(C/'freeze.json'),bind(C/'cpu_receipt.json'),bind(C/'test_profile_compatibility_cpu.py'),
            bind(__file__),build['native_dso'],bind(BUILD),bind(OLD/'freeze.json'),
            bind(parent_path),*files.values(),bind(P/'setup.json'),bind(P/'run_campaign_queue_v8.py'),
            bind(P/'owned_process_supervisor_v2.py')]
    mapping={j['id']:'wide3_'+j['id'].removeprefix('wide_') for j in parent['jobs']}
    outputs={j['receipt']:str(Q/'runs'/mapping[j['id']]/Path(j['receipt']).name) for j in parent['jobs']}
    jobs=[]
    for original in parent['jobs']:
        job=copy.deepcopy(original);command=job['command'];jid=mapping[job['id']]
        command[1]=str(C/'runner'/Path(command[1]).name)
        # Production source/DSO and every algorithm option are unchanged.
        assert command[command.index('--release')+1]==str(OLD/'Easy-Graph')
        assert command[command.index('--native-dso')+1]==build['native_dso']['path']
        replace(command,'--output',Path(outputs[job['receipt']]).parent)
        if '--tiny-proof' in command:
            replace(command,'--tiny-proof',outputs[command[command.index('--tiny-proof')+1]])
        job.update(id=jid,receipt=outputs[job['receipt']],bindings=copy.deepcopy(common),
                   depends_on=[mapping[x] for x in job['depends_on']],command_sha256=digest(command))
        if job.get('maintenance'):
            maintenance=job['maintenance'];argv=maintenance['command'];out=Q/'maintenance'/jid
            replace(argv,'--registry',Q/'queue_registry.json');replace(argv,'--output',out)
            replace(argv,'--job-result',job['receipt']);maintenance['receipt']=str(out/'receipt.json')
            maintenance['command_sha256']=digest(argv)
            maintenance['bindings']=[registry_binding if Path(b['path']).name=='queue_registry.json' else b for b in maintenance['bindings']]
        jobs.append(job)
    assert len({j['receipt'] for j in jobs})==len({j['id'] for j in jobs})==80
    prefix=[j['id'] for j in jobs if j['phase']!='target'];assert len(prefix)==76
    assert all(j['depends_on']==prefix for j in jobs if j['phase']=='target')
    known=set()
    for job in jobs:
        assert not Path(job['receipt']).parent.exists()
        assert set(job['depends_on'])<=known;known.add(job['id'])
        for b in job['bindings']+job.get('maintenance',{}).get('bindings',[]):
            actual=bind(b['path']);assert all(actual[k]==v for k,v in b.items()),b['path']
    plan=copy.deepcopy(parent);plan.update(schema='eggpu.wcc.wide.complete-qualification/v3',jobs=jobs,
        source=bind(C/'freeze.json'),build=bind(BUILD),runner=bind(C/'cpu_receipt.json'),
        generator=bind(__file__),parent_plans=[bind(parent_path)],maintenance_registry=registry_binding,
        scope='4 numeric +48 WCC branch/width +24 actual BFS/SSSP/PR regressions +4 TW; all newly bound outputs/proofs; no inherited GPU qualification.',
        admission='Only primary agent may launch. TW targets require all76 current-run prerequisites; full proof/timeout/numerical contracts unchanged.')
    plan['environment']['PYTHONPATH']=':'.join([str(C/'runner'),str(OLD/'native'),str(OLD/'Easy-Graph')])
    save(Q/'plan.json',plan)
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(C/'runner'));import native_worker as worker
    worker.shared.source_epoch=lambda proxy:(dict(files={},release_sources={},environment={},fixed_cpu_inventory=True),None)
    worker.runtime_dependencies=lambda path:dict(cpu_check_bound_dso=build['native_dso'])
    worker.shared.cgroup_memory=lambda:dict(cgroup_limits=[])
    parsed={}
    for job in jobs:
        args=worker.parser().parse_args(job['command'][2:]);worker.configure(args);parsed[job['id']]=args
    epochs=[]
    for count in (1,2,4,8):
        tiny=f'wide3_tiny_TINY_D_wcc_auto_forest_coo_p{count}';target=f'wide3_target_TW_wcc_auto_forest_coo_p{count}'
        a=parsed[tiny];worker.configure(a);one=worker.epoch(a)[1]
        b=parsed[target];worker.configure(b);two=worker.epoch(b)[1];assert one==two
        label=parsed[f'wide3_tiny_TINY_D_wcc_auto_label_coo_p{count}'];worker.configure(label)
        effective=worker.profile_record(label)
        assert effective['qualification_effective_environment']['EGMGPU_CORE_OUTPUT_CPU_PLACEMENT']=='inherited'
        assert effective['qualification_effective_environment']['EGMGPU_CORE_OUTPUT_CPUS']==''
        assert worker.epoch(label)[1]!=two
        epochs.append(dict(count=count,matching_target_tiny_epoch=True,label_epoch_distinct=True,base_and_effective_profile_separate=True))
    receipt=dict(schema='eggpu.wcc.wide.complete-preparation/v3',status='PASS_CPU_PREPARED_NOT_STARTED',
        gpu_executed=False,formal_measurements=0,jobs=80,prerequisites=76,numeric_jobs=4,wcc_tiny_jobs=48,
        public_regression_jobs=24,target_jobs=4,source_tree_sha256=SOURCE,native_dso=build['native_dso'],
        production_source_changed=False,compilation_performed=False,plan=bind(Q/'plan.json'),freeze=bind(C/'freeze.json'),
        generator=bind(__file__),epoch_checks=epochs,
        startup=['/usr/bin/taskset','-c',args.cpu_set,
            str(args.controller_python.absolute()),'-B',str(P/'run_campaign_queue_v8.py'),
            '--plan',str(Q/'plan.json'),'--output',str(Q/'queue'),'--controller-lock',str(P/'root_controller.lock')])
    save(Q/'preparation_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','jobs','prerequisites','source_tree_sha256','plan')}))


if __name__=='__main__':main()
