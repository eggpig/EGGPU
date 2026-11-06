"""CPU-only additive first observations: 32 hotspot cells and six PR controls."""
from pathlib import Path
from collections import Counter
import argparse
import copy
import hashlib
import importlib.util
import json
import sys

SOURCE='71ea757d372f8c1b969ea074caaa27d3a77d4cea6d48cd9c6ae9c4b02936d7ce'


def read(path):return json.loads(Path(path).read_text())
def bind(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())
def checked(item):
    if bind(item['path'])!=item:raise ValueError('Changed bound file: '+item['path'])
    return Path(item['path'])
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def option(values,name):
    if values.count(name)!=1:raise ValueError('Expected exactly one '+name)
    return values[values.index(name)+1]
def replace(values,name,value):values[values.index(name)+1]=str(value)


def inventory(campaign):
    campaign=Path(campaign)
    support=load('_additive_state_support',campaign/'state_protocol_completion_v1/state_plan.py')
    rows=read(campaign/'state_unified_full_v1/hotspot/coverage.json')['configurations']
    wanted={row['id']:row for row in support.hotspot_grid()}
    if len(rows)!=32 or {row['id'] for row in rows}!=set(wanted):raise ValueError('Hotspot denominator differs')
    for row in rows:
        if any(row[key]!=value for key,value in wanted[row['id']].items()):raise ValueError('Hotspot intervention changed')
    controls=read(campaign/'e3_unweighted_control_v4/unified_memory_v1/qualification_guarded.json')['jobs']
    if Counter(j['phase'] for j in controls)!={'tiny':2,'target':6}:raise ValueError('PR qualification scope differs')
    if {(j['dataset'],j['arm']) for j in controls if j['phase']=='target'}!={(d,a) for d in ('US','IN','OR') for a in ('D0C1','D1C1')}:
        raise ValueError('PR negative-control cells differ')
    return rows,controls


def prepare(campaign,state_main,output,expected_source=SOURCE):
    campaign,state_main,output=map(lambda p:Path(p).resolve(),(campaign,state_main,output))
    rows,controls=inventory(campaign)
    selection_path=state_main/'selection.json';selection=read(selection_path)
    if selection['source_tree_sha256']!=expected_source:raise ValueError('Additions must use the explicitly frozen source71')
    for key in ('source_freeze','native_dso','native_build_receipt','runner_freeze','hash_cache_helper'):checked(selection[key])
    freeze=read(selection['source_freeze']['path'])
    support=load('_additive_state_deployment',selection['source_guard'])
    support.source_files(Path(selection['source_root']),freeze)
    dep=read(campaign/'state_unified_full_v1/main/deployment.json')
    if str(Path(sys.executable).absolute())!=selection['python_invocation']:raise ValueError('Use original core venv Python')
    registry=read(checked(dep['registry']))
    focus=state_main/'results/focused_source_p8/receipt.json';focus_data=read(focus)
    if focus_data.get('status')!='PASS_STATE_GPU_QUALIFICATION' or focus_data.get('phase')!='tiny':
        raise ValueError('Retained current-source P8 focused check must finish first')
    for name,item in read(selection['runner_freeze']['path'])['files'].items():checked(dict(item,path=str(Path(selection['runner'])/name)))
    main=read(state_main/'all_plan.json');templates={}
    for job in main['jobs']:
        spec=read(option(job['command'],'--spec'));block=spec.get('paired_block') or {}
        if spec.get('action')=='run' and spec.get('phase')=='target' and block.get('experiment')=='E3':
            key=(tuple(block['datasets']),block['partitions'] if 'partitions' in block else len(option(spec['worker_arguments'],'--gpu-uuids').split(',')),block['rho'],block['locality'],block['old_readers'],spec['arm'])
            templates[key]=(job,spec)
    output.mkdir(parents=True,exist_ok=False)
    protocol=load('_additive_same_protocol',Path(selection['runner'])/'state_protocol.py')
    capacity=load('_additive_same_capacity',campaign/'state_completion_queue_guard_v1/derive.py').capacity
    jobs=[]
    for row in rows:
        inp=checked(row['inputs']);inputs=read(inp);support.validate_input_metadata(inputs,freeze,registry)
        key=(tuple(row['datasets']),8,1e-4,'uniform',row['old_readers'],row['arm'])
        original,base=templates[key];job=copy.deepcopy(original);spec=copy.deepcopy(base)
        identifier='hotspot_'+row['id'];spec_path=output/'specs'/(identifier+'.json');target=output/'results'/identifier
        args=spec['worker_arguments'];replace(args,'--inputs',inp);replace(args,'--output',target);replace(args,'--first-pass-spec',spec_path)
        block=spec['paired_block'];block.update(inputs=row['inputs'],locality=row['locality'],hotspot_k=row['hotspot_k'])
        spec.update(id=identifier,inputs=row['inputs'],priority_stage='supplement_partition_hotspot_p8',
            paired_block_sha256=hashlib.sha256(json.dumps(block,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            original_main_spec=bind(option(original['command'],'--spec')),additive_n1=True)
        save(spec_path,spec);replace(job['command'],'--spec',spec_path)
        job.update(id=identifier,receipt=str(target/'receipt.json'),depends_on=[],first_pass_stage=spec['priority_stage'],
            observations=1,source_tree_sha256=expected_source,
            disk_guard=dict(path=str(campaign),**capacity(spec,inputs,protocol.trace)),
            bindings=job['bindings']+[bind(spec_path),row['inputs'],bind(focus),bind(__file__)])
        jobs.append(job)
    # Unchanged negative-control code receives an isolated deployment pointing
    # to the same current candidate and same complete state runner. Its two
    # exact-arm tiny checks are one initial round; six targets are the n=1 cells.
    pr_dep=copy.deepcopy(dep)
    pr_dep.update(source_root=selection['source_root'],source_freeze=selection['source_freeze'],
        source_tree_sha256=expected_source,native_dso=selection['native_dso'],native_build_receipt=selection['native_build_receipt'],
        runner_root=selection['runner'],runner_freeze=selection['runner_freeze'],planner=bind(selection['source_guard']),
        first_observation_scope='Six distinct target repeat-zero PR controls; two exact-arm tiny checks; no duplicate targets or formal relabeling.')
    pr_dep['environment']['PYTHONPATH']=str(Path(selection['native_dso']['path']).parent)+':'+selection['source_root']
    pr_dep_path=output/'pr_deployment.json';save(pr_dep_path,pr_dep)
    idmap={j['id']:'unweighted_pr_'+j['id'] for j in controls}
    for original in controls:
        job=copy.deepcopy(original);values=job['command'];identifier=idmap[job['id']];target=output/'results'/identifier
        worker=Path(values[2]);checked(bind(worker));support.validate_input_metadata(read(checked(job['input_binding'])),freeze,registry,tiny=job['phase']=='tiny')
        replace(values,'--deployment',pr_dep_path);replace(values,'--support',selection['source_guard']);replace(values,'--output',target)
        replace(values,'--output-store',output/'objects')
        if '--tiny-proof' in values:
            prerequisite=original['depends_on'][0];replace(values,'--tiny-proof',output/'results'/idmap[prerequisite]/'receipt.json')
        for field in ('maintenance','deferred_maintenance'):
            if field in job:job['historical_'+field]=job.pop(field)
        job.update(id=identifier,receipt=str(target/'receipt.json'),depends_on=[idmap[x] for x in original['depends_on']],
            environment=support.environment_for(pr_dep,1),observations=1 if job['phase']=='target' else 0,
            source_tree_sha256=expected_source,first_pass_stage='unweighted_pr_negative_control',
            bindings=[bind(worker),bind(worker.with_name('pr_plan.py')),bind(pr_dep_path),pr_dep['planner'],job['input_binding'],
                selection['source_freeze'],selection['native_dso'],selection['native_build_receipt'],selection['runner_freeze'],bind(__file__)])
        jobs.append(job)
    if len(jobs)!=40 or len({j['id'] for j in jobs})!=40:raise ValueError('Additive plan must contain 38 observations and two tiny checks')
    plan=dict(schema='eggpu.state.additive-first-observation.v1',jobs=jobs,environment={},observations=1,
        target_cells=38,hotspot_cells=32,unweighted_pr_cells=6,initial_pr_tiny_checks=2,
        source_tree_sha256=expected_source,existing_state_denominator=504,combined_state_denominator=542,
        original_state_plan=bind(state_main/'all_plan.json'),selection=bind(selection_path),full_numerical_checks_unchanged=True,
        raw_phase_preserved=True,second_repetitions=0,storage_poll_seconds=5,gpu_executed=False)
    save(output/'plan.json',plan)
    save(output/'stage_manifest.json',dict(queue=str(output/'queue'),plan=str(output/'plan.json'),target_cells=38,initial_checks=2))
    save(output/'coverage.json',dict(hotspot=rows,unweighted_pr=[dict(dataset=j['dataset'],arm=j['arm'],partitions=1,phase='target',repeat=0) for j in controls if j['phase']=='target'],source_tree_sha256=expected_source,gpu_executed=False))
    return dict(status='PREPARED_CPU_ONLY',jobs=40,target_cells=38,source_tree_sha256=expected_source,plan=bind(output/'plan.json'),gpu_executed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','state-main','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-source-sha256',default=SOURCE)
    a=p.parse_args();print(json.dumps(prepare(a.campaign,a.state_main,a.output,a.expected_source_sha256)))
