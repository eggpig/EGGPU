"""CPU-only exact-source deployment of five independent arena diagnostic jobs."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os

HERE=Path(__file__).resolve().parent
SOURCE='d84ff388e1e40d1fd6ed83fcaceecd0491fc0451adffaff657a95940b1803f84'
DSO='6dfc1c03f249900afc216163d7b59d40da3dce19efdacc6eb60d198724801bac'


def require(value,message):
    if not value:raise ValueError(message)


def bind(path):
    path=Path(path).resolve();before=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'File changed while hashing')
    return dict(path=str(path),bytes=before.st_size,sha256=h.hexdigest())


def checked(item):
    require(bind(item['path'])==item,'Bound file changed: '+item['path'])
    return Path(item['path'])


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def option(command,key):return command[command.index('--'+key)+1]


def change(command,key,value):command[command.index('--'+key)+1]=str(value)


def transform(originals,output,worker_path):
    """Exactly preserve computation flags, environments and source rotation."""
    mapping={j['id']:'arena_diagnostic_v1_'+j['id'] for j in originals}
    receipts={j['receipt']:str(output/'runs'/mapping[j['id']]/'result.json') for j in originals}
    jobs=[]
    for old in originals:
        j=copy.deepcopy(old);j['id']=mapping[old['id']]
        old_script=next(x for x in old['command'] if x.endswith('.py'))
        require(Path(old_script).name=='sequence16_worker.py','Wrong frozen sequence entrypoint')
        j['command'][j['command'].index(old_script)]=str(worker_path)
        change(j['command'],'output',Path(receipts[old['receipt']]).parent)
        if '--tiny-proof' in j['command']:
            change(j['command'],'tiny-proof',receipts[option(old['command'],'tiny-proof')])
        j['receipt']=receipts[old['receipt']]
        j['depends_on']=[mapping[x] for x in old.get('depends_on',[])]
        # All remaining original CLI items retain their original order/value.
        normalized=list(j['command']);normalized[normalized.index(str(worker_path))]=old_script
        change(normalized,'output',option(old['command'],'output'))
        if '--tiny-proof' in normalized:change(normalized,'tiny-proof',option(old['command'],'tiny-proof'))
        require(normalized==old['command'],'Diagnostic changed a computation/resource argument')
        require(j.get('environment',{})==old.get('environment',{}),'Diagnostic changed job environment')
        jobs.append(j)
    return jobs


def prepare(args):
    from worker import EXPECTED,PROTOCOL
    campaign=args.campaign.resolve();base=args.base_plan.resolve();output=args.output.resolve()
    base_binding=bind(base);require(base_binding['sha256']==args.base_sha256,'Base sequence plan SHA differs')
    plan=read(base);require(plan['source_tree_sha256']==SOURCE,'Not current d84 sequence plan')
    freeze=read(checked(plan['source']));checked(plan['build'])
    require(freeze['source_tree_sha256']==SOURCE,'Source freeze differs')
    originals=[j for j in plan['jobs'] if option(j['command'],'system')=='eggpu']
    require(len(originals)==5,'Expected exactly two tiny and three target EGGPU jobs')
    require({(j['phase'],option(j['command'],'dataset')) for j in originals}==
            {('tiny','TINY_D'),('tiny','TINY_U'),('target','US'),('target','IN'),('target','TW')},'Diagnostic graph/phase scope differs')
    source=Path(option(originals[0]['command'],'source'))
    for rel,expected in freeze['source_files'].items():
        actual=bind(source/rel)
        require({k:actual[k] for k in ('bytes','sha256')}==expected,'Frozen source changed: '+rel)
    seen={}
    for job in originals:
        require(option(job['command'],'repeat')=='0' and option(job['command'],'phase')==job['phase'],'Only source-bound single qualifications')
        require(option(job['command'],'source')==str(source),'Mixed source roots')
        require(bind(option(job['command'],'native-dso'))['sha256']==DSO,'Native DSO differs')
        for item in job['bindings']:
            # One full byte observation per immutable dependency in this CPU preparation.
            if item['path'] not in seen:checked(item);seen[item['path']]=item
            else:require(seen[item['path']]==item,'Conflicting frozen bindings')
    for name,expected in EXPECTED.items():
        require(bind(campaign/'sg_current'/name)['sha256']==expected,'Frozen runner changed')
    require(not output.exists(),'Never overwrite any diagnostic output/queue')
    require(not output.is_relative_to(source.resolve()),'Do not write frozen source')
    jobs=transform(originals,output,HERE/'worker.py')
    python=originals[0]['command'][0]
    require(all(j['command'][0]==python for j in originals),'Mixed interpreter invocations')
    interpreter=bind(python)
    helpers={name:bind(campaign/name) for name in ('run_campaign_queue_v8.py','owned_process_supervisor_v2.py',
        'completed_job_audit_maintenance_v2.py','completed_output_maintenance_incremental_v1.py','incremental_completed_output_v1.py')}
    controller_lock=str(campaign/'root_controller.lock');gpu_lock=str(campaign.parents[1]/'gpu_experiment.lock')
    require(all(option(j['command'],'lock-file')==gpu_lock for j in jobs),'Original GPU lock differs')
    registry_path=output/'queue_registry.json'
    registry=dict(schema='completed.output.incremental.registry.v1',
        deduplicator=helpers['incremental_completed_output_v1.py'],queue_states=[str(output/'queue/state.json')],
        index=str(output/'completed_output_index.json'),controller_lock=controller_lock,gpu_lock=gpu_lock)
    extra=[bind(HERE/name) for name in ('worker.py','capture.py','prepare.py')]
    for job in jobs:
        require(not Path(job['receipt']).parent.exists(),'Diagnostic output was already attempted')
        command=job['command'];folder=Path(option(command,'input-root'))/'canonical'/option(command,'dataset')
        meta=read(folder/'canonical.json');receipt=read(folder/'receipt.json')
        require(receipt['status']=='PASS_CANONICAL_INPUT','Canonical metadata not qualified')
        require({k:bind(folder/'canonical.json')[k] for k in ('bytes','sha256')}==
                {k:receipt['files']['canonical'][k] for k in ('bytes','sha256')},'Canonical metadata changed')
        n=int(meta['num_nodes']);vectors=16*n*8
        job['disk_guard']=dict(path=str(campaign),min_free_bytes=2*vectors+8*n+(2<<30),
            complete_output_upper_bytes=vectors,new_oracle_upper_bytes=vectors,
            serialization_temporary_upper_bytes=8*n,safety_reserve_bytes=2<<30,
            compression_savings_assumed=False,dedup_savings_assumed=False)
        job['bindings']+=extra+[bind(folder/'canonical.json'),bind(folder/'receipt.json')]
        destination=output/'maintenance'/job['id']
        maintain=[python,'-B',str(campaign/'completed_job_audit_maintenance_v2.py'),'--registry',str(registry_path),
            '--output',str(destination),'--gpu-lock',gpu_lock,'--job-result',job['receipt'],
            '--dedup-wrapper',str(campaign/'completed_output_maintenance_incremental_v1.py')]
        job['maintenance']=dict(command=maintain,command_sha256=hashlib.sha256(json.dumps(maintain,separators=(',',':')).encode()).hexdigest(),
            bindings=[interpreter]+[helpers[k] for k in ('completed_job_audit_maintenance_v2.py',
                'completed_output_maintenance_incremental_v1.py','incremental_completed_output_v1.py')],
            gpu_lock=gpu_lock,environment={'CUDA_VISIBLE_DEVICES':''},receipt=str(destination/'receipt.json'),
            expected_statuses=['PASS_INCREMENTAL_HARDLINKED_OUTPUTS'],timeout=3600)
        job.update(diagnostic_only=True,performance_eligible=False,formal_measurement=False)
    output.mkdir(parents=True,exist_ok=False);save(registry_path,registry)
    for job in jobs:job['maintenance']['bindings'].append(bind(registry_path))
    derived=dict(plan,jobs=jobs,schema='eggpu.arena.diagnostic.qualification.v1',protocol=PROTOCOL,
        diagnostic_only=True,performance_eligible=False,formal_measurements=0,original_plan=base_binding,
        controller=helpers['run_campaign_queue_v8.py'],supervisor=helpers['owned_process_supervisor_v2.py'],
        deployment=bind(__file__),storage_poll_seconds=15,
        admission='CPU-only preparation. New diagnostic tiny proofs required; root alone may explicitly launch these five jobs.')
    save(output/'qualification_plan.json',derived)
    result=dict(status='PASS_CPU_DEPLOYED_ARENA_DIAGNOSTIC_NOT_LAUNCHED',gpu_executed=False,workers_launched=0,
        formal_measurements=0,jobs=5,protocol=PROTOCOL,plan=bind(output/'qualification_plan.json'),registry=bind(registry_path),
        source_sha256=SOURCE,source_freeze=plan['source'],native_dso_sha256=DSO,
        startup=[python,'-B',str(campaign/'run_campaign_queue_v8.py'),'--plan',str(output/'qualification_plan.json'),
                 '--output',str(output/'queue'),'--controller-lock',controller_lock])
    save(output/'receipt.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','base-plan','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--base-sha256',required=True)
    print(json.dumps(prepare(parser.parse_args())))
