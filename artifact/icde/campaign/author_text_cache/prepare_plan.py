"""Prepare cached-input successors for the explicitly deferred eight TW cells."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
from disk_admission import prepared_bound

HERE=Path(__file__).resolve().parent


def bind(path):
    path=Path(path).resolve();data=path.read_bytes()
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def option(command,key):
    if command.count(key)!=1:raise ValueError('Expected one '+key)
    return command[command.index(key)+1]


def prepare(deferred,preparation,converter_build,text_cache,output):
    deferred,preparation,converter_build,text_cache,output=map(lambda p:Path(p).resolve(),(deferred,preparation,converter_build,text_cache,output))
    data=json.loads(deferred.read_text());ready=json.loads(preparation.read_text())
    if ready['status']!='PASS_CPU_PREPARED_TEXT_INPUTS' or ready['converter_build']!=bind(converter_build) or ready['text_cache']!=str(text_cache):
        raise ValueError('Completed exact CPU input preparation required')
    oldjobs=data['jobs']
    cells={(option(j['command'],'--system'),option(j['command'],'--algorithm')) for j in oldjobs}
    if len(oldjobs)!=8 or cells!={(s,f) for s in ('grapin','hytgraph') for f in ('bfs','sssp','wcc','pr')}:
        raise ValueError('Expected exactly the eight deferred TW Grapin/HyTGraph cells')
    output.mkdir(parents=True,exist_ok=False);jobs=[];unavailable=[]
    for original in oldjobs:
        command=original['command']
        if original.get('depends_on'):raise ValueError('Deferred queue prerequisites must already have explicit reused receipts')
        if option(command,'--dataset')!='TW' or option(command,'--phase')!='target' or option(command,'--repeat')!='0':
            raise ValueError('Deferred scope is not target TW r0')
        index=next(i for i,v in enumerate(command) if v.endswith('.py'))
        helper=Path(command[index])
        if helper.name!='placement_worker.py' or command.count('--')!=1:
            raise ValueError('Expected the original placement/hash wrapper')
        worker=Path(option(command,'--worker'))
        if bind(worker)!=ready['original_worker'] or option(command,'--worker-sha256')!=ready['original_worker']['sha256']:
            raise ValueError('Deferred worker differs from byte-equivalence authority')
        algorithm=option(command,'--algorithm');kind='wcc' if algorithm=='wcc' else 'canonical'
        if ready['layouts'][kind]['status']!='PASS':
            unavailable.append(dict(id=original['id'],system=option(command,'--system'),algorithm=algorithm,dataset='TW',
                status='INPUT_LAYOUT_UNSUPPORTED_CPU',evidence=bind(preparation),detail=ready['layouts'][kind],
                gpu_measurement=False,original_attempt=original['receipt']))
            continue
        job=copy.deepcopy(original);identifier='cached_text_'+original['id'];destination=output/'runs'/identifier
        if destination.exists():raise ValueError('New output already exists')
        values=job['command'];values[index]=str(HERE/'cached_worker.py')
        split=values.index('--')
        values[split:split]=['--placement-helper',str(helper),'--text-cache',str(text_cache),
            '--converter-build',str(converter_build),'--preparation',str(preparation)]
        values[values.index('--output')+1]=str(destination)
        job.update(id=identifier,receipt=str(destination/'result_cached_input.json'),original_receipt=original['receipt'],
            measurement_protocol='author-ready-native-call2-cached-input-first-supplement-v1',
            canonical_csr_cold_first_comparable=False,ready_measurement_unchanged=True)
        storage=prepared_bound(ready,algorithm,bind)
        storage['source_audit']=bind(HERE/'disk_source_audit.json')
        job['original_disk_guard']=copy.deepcopy(job['disk_guard'])
        job['disk_guard']['min_free_bytes']=storage['min_free_bytes']
        job['cached_input_storage_admission']=storage
        job['bindings']=job.get('bindings',[])+[bind(HERE/name) for name in ('cached_worker.py','text_cache.py','prepare_plan.py','disk_admission.py')]+[bind(preparation),bind(converter_build),storage['cache_metadata'],storage['source_audit']]
        if 'command_sha256' in job:
            job['command_sha256']=hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
        for key in ('maintenance','deferred_maintenance'):
            if key in job:job['historical_'+key]=job.pop(key)
        jobs.append(job)
    plan=dict(schema='eggpu.author.cached-input-first-pass.v1',jobs=jobs,environment=data.get('environment',{}),storage_poll_seconds=5,
        cells=8,pending=len(jobs),unavailable=unavailable,source_deferred=bind(deferred),preparation=bind(preparation),
        protocol='Ready comparison uses unchanged complete native call 2. First times are cached-input supplements; one-time CPU cache construction is separately retained.',
        gpu_executed=False)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (output/'stage_manifest.json').write_text(json.dumps(dict(queue=str(output/'queue'),plan=str(output/'plan.json'),
        cells=8,pending=len(jobs),unavailable=unavailable),indent=2)+'\n')
    return dict(status='PREPARED_NOT_EXECUTED',jobs=len(jobs),cells=8,unavailable=unavailable,plan=bind(output/'plan.json'),gpu_executed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('deferred','preparation','converter-build','text-cache','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.deferred,a.preparation,a.converter_build,a.text_cache,a.output)))
