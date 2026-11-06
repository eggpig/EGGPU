"""Wait for the first comparison table, then start the three deferred queues.

Root supplies exact stopped snapshots and owns invocation. This script never
interrupts old work or rewrites plans. v8 performs its existing resume checks.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import subprocess
import time

TERMINAL = {'PASS_REQUESTED_JOBS', 'COMPLETE_WITH_FAILURES'}


def read(path):return json.loads(Path(path).read_text())
def bind(path):
    path=Path(path).resolve();data=path.read_bytes()
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.partial-'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def command(config,stage):
    values=[config['python'],'-B',stage.get('controller',config['controller']),'--plan',stage['plan'],
            '--output',stage['queue'],'--controller-lock',config['controller_lock']]
    if stage['mode']=='resume':
        values+=['--resume-state',stage['resume_state'],'--resume-sha',stage['resume_sha']]
    elif stage['mode']!='start':raise ValueError('Stage mode must be start or resume')
    floor=stage.get('wall_timeout_floor')
    if floor is not None:
        if type(floor) not in (int,float) or not math.isfinite(floor) or not 0<floor<=21600:raise ValueError('Invalid per-stage wall timeout floor')
        values+=['--wall-timeout-floor',str(floor)]
    return values


def old_owner_alive(stage):
    if stage['mode']!='resume':return False
    path=Path('/proc')/str(stage['old_owner_pid'])/'cmdline'
    try:raw=path.read_bytes()
    except FileNotFoundError:return False
    return any(name in raw for name in (b'run_campaign_queue_v8.py',b'run_campaign_queue_v9.py',b'run_campaign_queue_v10.py')) and stage['old_queue'].encode() in raw

def process_running(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def prepare(config_path,output):
    config_path=Path(config_path).resolve();output=Path(output).resolve();config=read(config_path)
    if config.get('schema') not in ('eggpu.after-main.resume.config.v1','eggpu.after-main.resume.config.v2'):raise ValueError('Explicit scheduling configuration required')
    if len(config['prerequisite_queue_states'])!=2 or len(config['stages'])!=3:
        raise ValueError('Expected main + missing prerequisites and exactly three deferred stages')
    if config['schema']=='eggpu.after-main.resume.config.v2':
        if sum(stage['mode']=='resume' for stage in config['stages'])!=1 or sum(stage['mode']=='start' for stage in config['stages'])!=2:raise ValueError('Fresh-state schedule requires one mechanism resume and two new queues')
        fresh=[stage for stage in config['stages'] if stage['name']=='state_main']
        if len(fresh)!=1 or fresh[0]['mode']!='start':raise ValueError('Fresh state must start a new epoch')
        proof=config['fresh_state_policy']
        if bind(proof['path'])!=proof:raise ValueError('Fresh policy binding changed')
        if read(fresh[0]['plan']).get('new_policy_epoch',{}).get('measurement_policy')!=proof:raise ValueError('Fresh state plan does not bind the selected new policy')
        if read(proof['path']).get('old_measurements_reused') is not False:raise ValueError('Old-policy state results cannot be reused')
    elif {stage['mode'] for stage in config['stages']}!={'resume','start'} or sum(stage['mode']=='resume' for stage in config['stages'])!=2:
        raise ValueError('Expected two unchanged-plan resumes and one new scaling queue')
    if not Path(config['controller_lock']).is_file():raise ValueError('Existing global per-job controller lock required')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'state.json').exists():
        state=read(output/'state.json')
        if state['config']!=bind(config_path):raise ValueError('Supervisor configuration changed')
        return config,state
    stages=[];pointer_updates=[];archives=[]
    for stage in config['stages']:
        if old_owner_alive(stage):raise ValueError('Old exact controller still alive: '+stage['name'])
        if Path(stage['queue']).exists():raise FileExistsError('New queue directory already exists: '+stage['queue'])
        plan=bind(stage['plan'])
        record=dict(name=stage['name'],mode=stage['mode'],queue=stage['queue'],plan=plan,
                    command=command(config,stage),controller=bind(stage.get('controller',config['controller'])),
                    wall_timeout_floor=stage.get('wall_timeout_floor'),status='WAITING_MAIN_AND_MISSING')
        if stage['mode']=='resume':
            snapshot=bind(stage['resume_state'])
            old=read(stage['resume_state'])
            if snapshot['sha256']!=stage['resume_sha'] or old.get('status')!='WAITING_CONTROLLER_LOCK':
                raise ValueError('Use the exact between-job waiting snapshot and SHA')
            if old.get('plan')!=plan:raise ValueError('Resume must use the unchanged original plan')
            record.update(resume_state=snapshot,inherited_completed=len(old.get('completed',[])),
                          stop_receipt=bind(stage['stop_receipt']))
        ancestor=Path(stage['ancestor_stage_manifest'])
        prior=read(ancestor)
        archive=output/(stage['name']+'_previous_stage_manifest.json')
        original_bytes=ancestor.read_bytes()
        archive_binding=dict(path=str(archive),bytes=len(original_bytes),sha256=hashlib.sha256(original_bytes).hexdigest())
        archives.append((archive,original_bytes))
        updated=copy.deepcopy(prior)
        old_queue=updated.get('queue')
        if not isinstance(old_queue,str):raise ValueError('Expected one original stage queue pointer')
        updated['queue']=stage['queue']
        updated.setdefault('historical_stages',[]).append(dict(queue=old_queue,manifest=archive_binding))
        updated.update(queue_status='SCHEDULED_WAITING_MAIN_AND_MISSING',future_plan=plan,
                       deferred_resume_supervisor=str(output/'state.json'))
        if config['schema']=='eggpu.after-main.resume.config.v2' and stage['name']=='state_main':
            updated['plan']=plan
            updated['measurement_policy']=config['fresh_state_policy']
            updated['old_policy_results_historical']=True
        record['ancestor_stage_manifest']=str(ancestor)
        record['previous_stage_manifest']=archive_binding
        stages.append(record)
        pointer_updates.append((ancestor,updated))
    state=dict(schema='eggpu.after-main.resume.supervisor.v4',status='WAITING_MAIN_AND_MISSING',
        config=bind(config_path),controller=bind(config['controller']),supervisor=bind(__file__),
        prerequisites=config['prerequisite_queue_states'],stages=stages,started_unix=time.time(),
        plans_modified=False,completed_jobs_rerun=False)
    for archive,original_bytes in archives:archive.write_bytes(original_bytes)
    for ancestor,updated in pointer_updates:atomic(ancestor,updated)
    atomic(output/'state.json',state)
    atomic(output/'stage_manifest.json',dict(stages=[dict(queue=stage['queue'],name=stage['name']) for stage in stages]))
    return config,state


def run(config_path,output,poll,prepare_only=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'supervisor.lock').open('a+') as singleton:
        fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config,state=prepare(config_path,output)
        def update(**values):
            state.update(values,updated_unix=time.time());atomic(output/'state.json',state)
        if prepare_only:
            print(json.dumps(dict(status='PREPARED_WAITING_MAIN_AND_MISSING',stages=len(state['stages']))));return
        while True:
            prerequisites=[]
            for value in config['prerequisite_queue_states']:
                path=Path(value);row=read(path) if path.is_file() else {}
                prerequisites.append(dict(path=value,status=row.get('status','NOT_STARTED'),
                                          completed=len(row.get('completed',[])),jobs=len(row.get('jobs',[]))))
            if all(row['status'] in TERMINAL for row in prerequisites):break
            if state.get('prerequisite_status')!=prerequisites:update(status='WAITING_MAIN_AND_MISSING',prerequisite_status=prerequisites)
            time.sleep(poll)
        update(status='STARTING_DEFERRED_QUEUES',prerequisite_status=prerequisites)
        for original,stage in zip(config['stages'],state['stages']):
            queue=Path(stage['queue']);queue_state=queue/'state.json'
            if queue_state.exists():
                if read(queue_state).get('plan')!=stage['plan']:raise ValueError('Existing resumed queue has a different plan')
                stage['status']='SUBMITTED_ALREADY';update();continue
            if stage.get('pid') or stage.get('launch_intent_unix'):
                raise RuntimeError('Ambiguous prior launch; preserve logs and inspect instead of resubmitting')
            if old_owner_alive(original):raise ValueError('Old exact controller reappeared')
            if bind(stage['controller']['path'])!=stage['controller']:raise ValueError('Deferred controller changed')
            if bind(original['plan'])!=stage['plan']:raise ValueError('Deferred plan changed')
            if original['mode']=='resume' and bind(original['resume_state'])!=stage['resume_state']:
                raise ValueError('Paused snapshot changed')
            stage.update(status='STARTING',launch_intent_unix=time.time());update()
            with (output/(stage['name']+'.controller.log')).open('xb') as log:
                child=subprocess.Popen(stage['command'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                    start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            stage.update(status='SUBMITTED',pid=child.pid,submitted_unix=time.time());update()
            ancestor=Path(stage['ancestor_stage_manifest']);manifest=read(ancestor)
            if manifest.get('queue')!=stage['queue']:raise ValueError('Ancestor stage pointer changed unexpectedly')
            manifest['queue_status']='SUBMITTED';manifest['controller_pid']=child.pid;atomic(ancestor,manifest)
        update(status='RUNNING_DEFERRED_QUEUES')
        while True:
            complete=True;failed=[]
            for stage in state['stages']:
                path=Path(stage['queue'])/'state.json';row=read(path) if path.is_file() else {}
                stage.update(status=row.get('status','STARTING'),completed=len(row.get('completed',[])))
                if stage['status'] not in TERMINAL:complete=False
                if stage['status'].startswith('FAILED_'):failed.append(stage['name'])
                if not path.exists() and stage.get('pid') and not process_running(stage['pid']):
                    failed.append(stage['name'])
            if failed:update(status='DEFERRED_CONTROLLER_FAILURE_REQUIRES_INSPECTION',failed_stages=failed);return
            if complete:update(status='DEFERRED_QUEUES_FINISHED',finished_unix=time.time());return
            update();time.sleep(poll)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--poll-seconds',type=float,default=15);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.poll_seconds<=60:parser.error('Poll interval must be 1..60 seconds')
    try:run(args.config,args.output,args.poll_seconds,args.prepare_only)
    except BaseException as error:
        path=args.output/'state.json'
        if path.exists():
            state=read(path);state.update(status='SUPERVISOR_FAILED_REQUIRES_INSPECTION',error=type(error).__name__+': '+str(error));atomic(path,state)
        raise
