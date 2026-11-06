"""Wait for first-pass stages, then admit the unchanged 426-target state tail."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

TERMINAL = {'PASS_REQUESTED_JOBS', 'COMPLETE_WITH_FAILURES'}
DEFAULT_STAGES = ('unified_main', 'state_main', 'missing_external', 'unified_scaling',
                  'unified_mechanism_application', 'author_reservation_handoff')


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def checked(item):
    if bind(item['path']) != item:
        raise ValueError('Frozen scheduler input changed: '+item['path'])
    return Path(item['path'])


def atomic(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def exclusive(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())


def option(command, flag):
    if command.count(flag) != 1:
        raise ValueError('Expected one '+flag)
    return command[command.index(flag)+1]


def queue_path(value):
    path = Path(value['path'] if isinstance(value,dict) else value).resolve()
    return path if path.name=='state.json' else path/'state.json'


def queue_snapshot(path, expected_plan=None):
    path=Path(path)
    if not path.exists():
        return dict(status='PENDING_QUEUE',queue=str(path),terminal=False)
    state=read(path)
    actual_plan=state['plan'];checked(actual_plan)
    if expected_plan is not None:
        expected=bind(expected_plan['path'] if isinstance(expected_plan,dict) else expected_plan)
        if isinstance(expected_plan,dict) and expected!=expected_plan:
            raise ValueError('Stage manifest plan changed')
        if actual_plan!=expected:
            raise ValueError('Stage queue belongs to another plan: '+str(path))
    plan=read(actual_plan['path'])
    if state.get('jobs')!=plan['jobs']:
        raise ValueError('Stage state jobs differ from their immutable plan')
    identifiers=[j['id'] for j in plan['jobs']]
    completed=state.get('completed',[]);completed_ids=[j['id'] for j in completed]
    if len(set(identifiers))!=len(identifiers) or len(set(completed_ids))!=len(completed_ids) or not set(completed_ids)<=set(identifiers):
        raise ValueError('Duplicate or foreign job in stage state')
    terminal=state['status'] in TERMINAL
    if terminal and set(completed_ids)!=set(identifiers):
        raise ValueError('Terminal stage omitted jobs')
    return dict(status=state['status'],queue=str(path),terminal=terminal,
        plan=actual_plan,jobs=len(identifiers),completed=len(completed),
        outcomes=dict(Counter(x['outcome'] for x in completed)),active_job=state.get('active_job'),
        completed_worker_executions=sum(bool(x.get('execution_started')) for x in completed),
        active_worker_started=state['status']=='RUNNING' and state.get('worker_started_for_active_job') is True,
        state_binding=bind(path) if terminal else None)


def manifest_snapshot(path):
    path=Path(path)
    if not path.exists():
        return dict(status='PENDING_STAGE_MANIFEST',manifest=str(path),terminal=False,queues=[])
    manifest=read(path)
    entries=manifest.get('queues')
    if entries is None:
        value=manifest.get('queue',manifest.get('queue_state'))
        if value is None:
            raise ValueError('Stage manifest needs queue/queue_state or queues: '+str(path))
        entries=[dict(queue=value,plan=manifest.get('plan'))]
    if not isinstance(entries,list) or not entries:
        raise ValueError('Stage manifest has no queues')
    queues=[]
    for entry in entries:
        if isinstance(entry,str):
            entry=dict(queue=entry)
        value=entry.get('queue',entry.get('queue_state'))
        if value is None:
            raise ValueError('Stage queue entry has no queue path')
        queues.append(queue_snapshot(queue_path(value),entry.get('plan')))
    return dict(status='STAGE_TERMINAL' if all(q['terminal'] for q in queues) else 'WAITING_STAGE',
        manifest=str(path),manifest_binding=bind(path),terminal=all(q['terminal'] for q in queues),queues=queues)


def observed_wall_times(bundle, core_state, tail_state=None):
    coverage=read(Path(bundle)/'coverage.json')
    cells={'target_'+r['id']:r for r in coverage['configurations']}
    groups=defaultdict(list);total=0
    for state_path in (core_state,tail_state):
        if state_path is None or not Path(state_path).exists():
            continue
        for item in read(state_path).get('completed',[]):
            if item['id'] not in cells or item.get('outcome')!='PASS':
                continue
            seconds=item.get('seconds')
            if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
                continue
            cell=cells[item['id']]
            key=(cell['experiment'],cell['partitions'],tuple(cell['datasets']),cell['arm'],cell['old_readers'],cell['order'])
            groups[key].append(seconds);total+=1
    rows=[dict(experiment=k[0],partitions=k[1],datasets=list(k[2]),arm=k[3],old_readers=k[4],order=k[5],
        observations=len(v),job_wall_seconds_median=statistics.median(v),job_wall_seconds_min=min(v),job_wall_seconds_max=max(v))
        for k,v in sorted(groups.items())]
    return dict(successful_target_job_observations=total,observed_groups=rows,
        estimated_remaining_seconds=None,estimated_finish_unix=None,
        reason='No target job wall-time observations yet.' if not rows else 'Observed job wall times are descriptive. Unmeasured P1/old-reader/order/input-update configurations have no validated total-time model.',
        excludes_tiny_and_pilots=True,continuous_query_times_are_not_job_wall_times=True,
        no_cross_workload_extrapolation=True)


def inspect(manifest):
    for name in ('pending','coverage','core_plan','helper','helper_prepare','controller','supervisor','selection','source'):
        checked(manifest[name])
    stages=[manifest_snapshot(path) for path in manifest['wait_stage_manifests']]
    core=queue_snapshot(manifest['core_queue_state'],manifest['core_plan'])
    failures=[q for stage in stages for q in stage['queues'] if q['status'].startswith('FAILED_')]
    waiting=[s['manifest'] for s in stages if not s['terminal']]
    if core['status'].startswith('FAILED_'):
        failures.append(core)
    status='BLOCKED_STAGE_CONTROLLER_FAILURE' if failures else 'WAITING_MAJOR_FIRST_PASS_STAGES' if waiting or not core['terminal'] else 'READY_TO_ADMIT'
    prerequisite_failures=[]
    if core['terminal']:
        state=read(manifest['core_queue_state']);completed={j['id']:j for j in state['completed']}
        for identifier in manifest['core_prerequisites']:
            row=completed[identifier]
            if row['outcome']!='PASS' or not row.get('receipt'):
                prerequisite_failures.append(dict(id=identifier,outcome=row['outcome']))
            else:
                checked(row['receipt'])
        if prerequisite_failures:
            status='BLOCKED_CORE_PREREQUISITE'
    return dict(status=status,stages=stages,core=core,waiting_stage_manifests=waiting,
        failed_stage_controllers=failures,core_prerequisite_failures=prerequisite_failures,
        time_evidence=observed_wall_times(manifest['bundle'],manifest['core_queue_state']),
        observed_unix=time.time(),gpu_launched_by_inspection=False)


def prepare(args):
    campaign=args.campaign.resolve();bundle=args.bundle.resolve();out=args.output.resolve()
    floor=args.wall_timeout_floor
    if floor is not None and (not math.isfinite(floor) or not 0<floor<=21600):raise ValueError('Wall timeout floor must be finite and in (0,21600]')
    if not args.wait_stage:
        args.wait_stage=[campaign/'first_pass_20261106'/name/'stage_manifest.json' for name in DEFAULT_STAGES]
    waits=[str(Path(p).resolve()) for p in args.wait_stage]
    if len(waits)<6 or len(set(waits))!=len(waits):
        raise ValueError('The six named major first-pass stage manifests are required')
    if str(out/'stage_manifest.json') in waits:
        raise ValueError('Tail cannot wait for itself')
    core_plan=args.core_plan.resolve() if args.core_plan else bundle/'plan.json'
    core=read(core_plan);pending=read(bundle/'tail_pending.json')
    checked(pending['core_plan'])
    if core['jobs']!=read(pending['core_plan']['path'])['jobs'] or len(core['jobs'])!=82:
        raise ValueError('Core must be exactly the original 82-job priority plan')
    if len(pending['job_specs'])!=429:
        raise ValueError('Tail must retain 426 targets and three P1 byte pilots')
    coreids={j['id'] for j in core['jobs']};tailids={j['id'] for j in pending['job_specs']}
    if len(tailids)!=429 or coreids&tailids:
        raise ValueError('Tail repeats core jobs')
    if sum(j['id'].startswith('target_') for j in pending['job_specs'])!=426:
        raise ValueError('Tail target denominator changed')
    specifications=[]
    for job in [*core['jobs'],*pending['job_specs']]:
        spec_path=Path(option(job['command'],'--spec'))
        specifications.append(read(spec_path))
    output_stores={option(s['worker_arguments'],'--output-store') for s in specifications}
    reference_caches={option(s['worker_arguments'],'--reference-cache') for s in specifications}
    if output_stores!={str(bundle/'objects')} or len(reference_caches)!=1:
        raise ValueError('Core/tail must share original content-addressed outputs and reference cache')
    helper=args.admit_helper.resolve()
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema='eggpu.state-tail.successor.v1',bundle=str(bundle),output=str(out),
        wait_stage_manifests=waits,core_queue_state=str(queue_path(args.core_queue or bundle/'queue')),
        core_plan=bind(core_plan),pending=bind(bundle/'tail_pending.json'),coverage=bind(bundle/'coverage.json'),
        selection=bind(bundle/'selection.json'),helper=bind(helper),helper_prepare=bind(helper.with_name('prepare.py')),
        source=bind(__file__),controller=bind(args.controller.resolve() if args.controller else campaign/'run_campaign_queue_v8.py'),
        wall_timeout_floor=floor,
        supervisor=bind(campaign/'owned_process_supervisor_v2.py'),controller_lock=str(campaign/'root_controller.lock'),
        python_invocation=read(bundle/'selection.json')['python_invocation'],poll_seconds=args.poll_seconds,
        tail_plan=str(out/'plan.json'),tail_queue=str(out/'queue'),
        core_prerequisites=sorted({d for j in pending['job_specs'] for d in j['depends_on'] if d in coreids}),
        full_denominator=504,priority_targets=78,remaining_targets=426,new_byte_pilots=3,
        output_store=next(iter(output_stores)),reference_cache=next(iter(reference_caches)),
        storage_policy='Original immutable content-addressed lossless vectors and existing independent-reference cache; no raw deletion, copied payloads, numerical changes, or reduced disk guards.',
        outstanding_outside_504=['partition_hotspot_32','unweighted_PR_invariance_6','Grapin_update_adapter'],
        gpu_executed=False)
    exclusive(out/'manifest.json',manifest)
    exclusive(out/'stage_manifest.json',dict(schema='eggpu.first-pass.stage.v1',queue=str(out/'queue'),plan=str(out/'plan.json'),
        scheduler_manifest=bind(out/'manifest.json'),scheduler_state=str(out/'state.json'),
        source_tree_sha256=read(bundle/'selection.json')['source_tree_sha256'],targets=426,full_denominator=504,
        status='WAITING_MAJOR_FIRST_PASS_STAGES',gpu_executed=False))
    exclusive(out/'state.json',dict(status='PREPARED_NOT_RUNNING',manifest=bind(out/'manifest.json'),gpu_launched=False))
    return dict(status='PREPARED_CPU_ONLY',manifest=bind(out/'manifest.json'),stage_manifest=bind(out/'stage_manifest.json'),
        targets=426,jobs=429,wait_stage_manifests=waits,gpu_executed=False)


def validate_tail(manifest):
    pending=read(checked(manifest['pending']));plan=read(manifest['tail_plan'])
    if len(plan['jobs'])!=429 or plan.get('targets')!=426 or plan.get('no_repeated_core_jobs') is not True:
        raise ValueError('Original tail admission changed scope')
    for old,new in zip(pending['job_specs'],plan['jobs']):
        for key in set(old)|set(new):
            if key in ('depends_on','bindings','reused_completed_prerequisites'):
                continue
            if old.get(key)!=new.get(key):
                raise ValueError('Tail admission changed worker/measurement field: '+key)
        if new['bindings'][:len(old['bindings'])]!=old['bindings']:
            raise ValueError('Tail dropped original bindings')
    return bind(manifest['tail_plan'])


def run(path,execute_launch=False):
    if not execute_launch:
        raise ValueError('Primary agent must explicitly request --execute-launch')
    path=Path(path).resolve();manifest=read(path);out=Path(manifest['output'])
    if str(Path(sys.executable).absolute())!=manifest['python_invocation']:
        raise ValueError('Use the original core Python invocation')
    with (out/'watcher.lock').open('a+') as own_lock:
        fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # This private watcher lock is never the shared controller/GPU lock.
        def update(**values):
            atomic(out/'state.json',dict(manifest=bind(path),updated_unix=time.time(),**values))
        try:
            queue_state=Path(manifest['tail_queue'])/'state.json'
            intent_path=out/'launch_intent.json'
            if not intent_path.exists():
                while True:
                    snapshot=inspect(manifest)
                    update(**snapshot,gpu_launched=False)
                    if snapshot['status'].startswith('BLOCKED_'):
                        return snapshot
                    if snapshot['status']=='READY_TO_ADMIT':
                        break
                    time.sleep(manifest['poll_seconds'])
                if Path(manifest['tail_plan']).exists():
                    raise ValueError('Unclaimed tail plan already exists; preserve it for inspection')
                for job in read(manifest['pending']['path'])['job_specs']:
                    if Path(job['receipt']).parent.exists():
                        raise ValueError('Tail output already exists; no automatic retry: '+job['id'])
                command=[manifest['python_invocation'],'-B',manifest['helper']['path'],'--pending',manifest['pending']['path'],
                    '--core-state',manifest['core_queue_state'],'--output',manifest['tail_plan']]
                update(status='ADMITTING_TAIL_CPU_ONLY',gpu_launched=False,command=command)
                subprocess.run(command,check=True,stdin=subprocess.DEVNULL)
                plan_binding=validate_tail(manifest)
                gate=inspect(manifest)
                if gate['status']!='READY_TO_ADMIT':
                    raise ValueError('Major-stage gate changed during CPU admission')
                command=[manifest['python_invocation'],'-B',manifest['controller']['path'],'--plan',manifest['tail_plan'],
                    '--output',manifest['tail_queue'],'--controller-lock',manifest['controller_lock']]
                if manifest.get('wall_timeout_floor') is not None:
                    command+=['--wall-timeout-floor',str(manifest['wall_timeout_floor'])]
                intent=dict(command=command,plan=plan_binding,prerequisite_gate=gate,created_unix=time.time(),
                    policy='Exactly one submission; unresolved launch intent is never retried automatically.')
                exclusive(intent_path,intent)
                with (out/'queue.log').open('xb') as log:
                    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,
                        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                exclusive(out/'launch.json',dict(pid=child.pid,command=command,created_unix=time.time(),intent=bind(intent_path)))
            else:
                intent=read(intent_path);checked(intent['plan'])
                if not queue_state.exists() and not (out/'launch.json').exists():
                    update(status='BLOCKED_UNRESOLVED_LAUNCH_INTENT',gpu_launched=None)
                    return dict(status='BLOCKED_UNRESOLVED_LAUNCH_INTENT')
            while True:
                snapshot=queue_snapshot(queue_state,manifest['tail_plan'])
                update(status='TAIL_'+snapshot['status'],tail=snapshot,controller_submitted=True,
                    gpu_launched=bool(snapshot.get('completed_worker_executions',0) or snapshot.get('active_worker_started')),
                    time_evidence=observed_wall_times(manifest['bundle'],manifest['core_queue_state'],queue_state))
                if snapshot['terminal'] or snapshot['status'].startswith('FAILED_'):
                    return snapshot
                if not queue_state.exists():
                    launch=read(out/'launch.json')
                    proc=Path('/proc')/str(launch['pid'])/'cmdline'
                    if not proc.exists() or manifest['tail_plan'].encode() not in proc.read_bytes().split(b'\0'):
                        update(status='BLOCKED_LAUNCH_WITHOUT_QUEUE_STATE',gpu_launched=None,launch=launch)
                        return dict(status='BLOCKED_LAUNCH_WITHOUT_QUEUE_STATE')
                time.sleep(manifest['poll_seconds'])
        except BaseException as error:
            update(status='STOPPED_REQUIRES_INSPECTION',error=type(error).__name__+': '+str(error),
                controller_submitted=(out/'launch.json').exists(),gpu_launched=None if (out/'launch.json').exists() else False)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    for name in ('campaign','bundle','admit-helper','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--core-plan',type=Path);p.add_argument('--core-queue',type=Path)
    p.add_argument('--controller',type=Path);p.add_argument('--wall-timeout-floor',type=float)
    p.add_argument('--wait-stage',type=Path,action='append',default=[])
    p.add_argument('--poll-seconds',type=int,default=15,choices=range(1,61))
    for name in ('inspect','run'):
        p=sub.add_parser(name);p.add_argument('--manifest',type=Path,required=True)
        if name=='run':p.add_argument('--execute-launch',action='store_true')
    args=parser.parse_args()
    result=prepare(args) if args.action=='prepare' else inspect(read(args.manifest)) if args.action=='inspect' else run(args.manifest,args.execute_launch)
    print(json.dumps(result,allow_nan=False))
