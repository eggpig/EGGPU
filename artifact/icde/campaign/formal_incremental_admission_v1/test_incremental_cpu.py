"""Real frozen proof readers plus synthetic CPU-only admission/queue fixtures."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import concurrent.futures
import ast
import copy
import fcntl
import importlib.util
import json
import os
import sys

import pytest

import admit_incremental as inc
import run_incremental_queue as controller

LEDGER, _ = inc.inspector()
spec = importlib.util.spec_from_file_location('_incremental_ledger_fixture',
    Path(LEDGER.__file__).with_name('test_inspect_ledger.py'))
fixture_module = importlib.util.module_from_spec(spec)
sys.modules['inspect_ledger'] = LEDGER
spec.loader.exec_module(fixture_module)


@pytest.fixture
def case():
    fixture = fixture_module.Ledger('runTest'); fixture.setUp()
    f = fixture.f; root = f.root
    spec = f.manifest['groups']['sg']
    spec.update(admitted_output=str(root/'old_admitted'), queue=str(root/'old_queue'),
                controller_log=str(root/'old_controller.log'))
    f.manifest['controller_lock'] = str(root/'controller.lock')
    f.write(f.manifest_path,f.manifest)
    parent = Path(inc.__file__).resolve().parents[1]
    request = dict(schema='eggpu.formal.incremental.request.v1', group='sg',
        parent_manifest=inc.bind(f.manifest_path), namespace_root=str(root/'incremental'),
        controller_lock=f.manifest['controller_lock'],
        controller=inc.bind(parent/'run_campaign_queue_v8.py'),
        supervisor=inc.bind(parent/'owned_process_supervisor_v2.py'),
        python=inc.bind(sys.executable), python_invocation=sys.executable,
        additional_prior_states=[], additional_legacy_directories=[], dispositions=[])
    request_path = root/'incremental_request.json'; f.write(request_path,request)
    obj = SimpleNamespace(f=f,fixture=fixture,root=root,request=request,request_path=request_path,n=0)

    def run(admit=True):
        f.write(request_path,request); out=root/f'observation{obj.n}';obj.n+=1
        return inc.run(request_path,out,admit=admit)
    obj.run = run
    try: yield obj
    finally: fixture.tearDown()


def second_cell(case, status='PASS_QUALIFICATION'):
    f=case.f
    target=copy.deepcopy(f.qual['jobs'][-1]); target['id']='second_target'
    target['receipt']=str(case.root/'second_target/result.json')
    target['command'][target['command'].index('--output')+1]=str(case.root/'second_target')
    proof=inc.load(f.qual['jobs'][-1]['receipt']);proof['status']=status
    f.write(target['receipt'],proof)
    f.qual['jobs'].append(target)
    originals=copy.deepcopy(f.formal['jobs'])
    extra=[]
    for original in originals:
        job=copy.deepcopy(original);job['id']='second_'+original['id'];job['depends_on']=['second_target']
        job['receipt']=str(case.root/job['id']/'result.json')
        command=job['command'];command[command.index('--output')+1]=str(case.root/job['id'])
        command[command.index('--target-proof')+1]=target['receipt'];extra.append(job)
    f.formal['jobs']=[j for pair in zip(originals,extra) for j in pair]
    case.fixture.sync_plans();case.request['parent_manifest']=inc.bind(f.manifest_path)
    return target


def batch_report(result): return result['admission']


def test_one_qualified_cell_admits_exact_original_ten_jobs_and_cpu_gate(case):
    result=case.run(); admission=batch_report(result); plan=inc.load(admission['plan']['path'])
    assert result['original_cell_count']==1 and result['original_formal_slots']==10
    assert result['workers_launched']==0 and result['formal_measurements_created']==0
    assert [j for j in plan['jobs'] if j['phase']=='formal']==case.f.formal['jobs']
    assert plan['jobs'][0]['id']=='target' and plan['jobs'][0]['phase']=='cpu'
    assert all(j['depends_on']==['target'] for j in plan['jobs'][1:])
    assert not (Path(admission['plan']['path']).parent/'queue').exists()
    assert (case.root/'old_queue/incremental_fence.json').is_file()
    assert not (case.root/'old_admitted/plan.json').exists()


@pytest.mark.parametrize('status', ['FAILED','RUNNING','FAIL_QUALITY'])
def test_unresolved_peer_remains_in_complete_matrix_without_blocking_qualified_cell(case,status):
    second=second_cell(case,status)
    result=case.run();assert result['selected_cell_ids']==['target']
    assert result['original_cell_count']==2 and result['original_formal_slots']==20
    assert sum(len(c['formal_slots']) for c in result['complete_ledger']['cells'])==20
    peer=next(c for c in result['complete_ledger']['cells'] if c['cell_id']==second['id'])
    assert peer['status']!='QUALIFIED' and not peer['counts_as_win'] and peer['time_seconds'] is None
    assert len(peer['formal_slots'])==10


def test_later_qualified_cell_is_new_batch_without_replaying_first(case):
    second=second_cell(case,'RUNNING');first=case.run()
    proof=inc.load(second['receipt']);proof['status']='PASS_QUALIFICATION';case.f.write(second['receipt'],proof)
    second_result=case.run();assert second_result['selected_cell_ids']==['second_target']
    plan=inc.load(batch_report(second_result)['plan']['path'])
    expected=[j for j in case.f.formal['jobs'] if j['id'].startswith('second_')]
    assert [j for j in plan['jobs'] if j['phase']=='formal']==expected
    third=case.run();assert third['selected_formal_slots']==0 and 'admission' not in third
    registry=inc.load(case.root/'incremental/sg/registry.json')
    assert len(registry['batches'])==2 and len(registry['cells'])==2


def test_fixed_matrix_order_and_interleaving_are_preserved(case):
    second_cell(case);result=case.run();plan=inc.load(batch_report(result)['plan']['path'])
    assert [j for j in plan['jobs'] if j['phase']=='formal']==case.f.formal['jobs']
    assert result['selected_cell_ids']==['target','second_target']


@pytest.mark.parametrize('where',['worker_output','maintenance_output','prior_queue'])
def test_any_attempt_or_queue_claim_blocks_all_ten_of_that_cell(case,where):
    if where=='worker_output':Path(inc.option(case.f.formal['jobs'][3]['command'],'output')).mkdir()
    if where=='maintenance_output':
        path=case.root/'maintenance0';path.mkdir()
        for j in case.f.formal['jobs']:j['maintenance']=dict(receipt=str(path/'receipt.json'),command=['python','m.py','--output',str(path)])
        case.fixture.sync_plans();case.request['parent_manifest']=inc.bind(case.f.manifest_path)
    if where=='prior_queue':
        p=case.root/'prior.json';case.f.write(p,dict(status='WAITING_CONTROLLER_LOCK',jobs=case.f.formal['jobs'],completed=[]))
        case.request['additional_prior_states']=[str(p)]
    result=case.run();assert result['selected_formal_slots']==0
    assert result['complete_ledger']['cells'][0]['incremental_scheduling']['status']=='EXISTING_ATTEMPT_OR_CLAIM'
    assert not (case.root/'incremental').exists()


@pytest.mark.parametrize('where',['old_admitted','old_queue','old_controller.log'])
def test_existing_legacy_namespace_never_replaced_or_reused(case,where):
    path=case.root/where
    if where.endswith('.log'):path.write_text('prior launch')
    else:path.mkdir();(path/'existing.json').write_text('{}')
    result=case.run();assert result['selected_formal_slots']==0
    assert any(b.get('stage')=='legacy_namespace' for b in result['blockers'])


def test_inspection_does_not_fence_or_reserve(case):
    result=case.run(admit=False);assert result['selected_formal_slots']==10 and 'admission' not in result
    assert not (case.root/'old_queue').exists() and not (case.root/'incremental').exists()


def test_source_drift_globally_blocks_admission(case):
    (case.f.source/'easygraph/gpu/session.py').write_text('changed')
    result=case.run();assert result['selected_formal_slots']==0 and result['blockers']


def test_failed_numeric_or_cpu_global_gate_is_not_cell_selectable(case):
    module,_=inc.inspector();real=module.inspect
    def blocked(path):
        report=real(path);report['global_blockers'].append(dict(stage='global_qualification',job='numeric_p4',reason='missing exact current output'))
        return report
    with mock.patch.object(module,'inspect',side_effect=blocked),mock.patch.object(inc,'inspector',return_value=(module,Path(module.__file__))):
        result=case.run()
    assert result['selected_formal_slots']==0 and not (case.root/'incremental').exists()


def test_missing_repeat_still_rejected_before_selection(case):
    case.f.formal['jobs'].pop();case.fixture.sync_plans();case.request['parent_manifest']=inc.bind(case.f.manifest_path)
    with pytest.raises(ValueError,match='ten repeats'):case.run()


def test_corrupt_actual_output_prevents_that_cells_admission(case):
    (case.root/'US/out0first').write_bytes(b'wrong actual bytes')
    result=case.run();assert result['selected_formal_slots']==0
    assert result['complete_ledger']['cells'][0]['status']=='INFRA'


def test_admission_and_inspection_hold_original_controller_lock(case):
    module,_=inc.inspector();real=module.inspect
    def probe(path):
        with Path(case.request['controller_lock']).open('a+') as lock:
            with pytest.raises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return real(path)
    with mock.patch.object(module,'inspect',side_effect=probe),mock.patch.object(inc,'inspector',return_value=(module,Path(module.__file__))):
        assert case.run()['selected_formal_slots']==10


def test_concurrent_admission_reserves_each_cell_once(case):
    def invoke(i):return inc.run(case.request_path,case.root/f'concurrent{i}',admit=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(invoke,[0,1]))
    assert sorted(r['selected_formal_slots'] for r in results)==[0,10]
    assert len(inc.load(case.root/'incremental/sg/registry.json')['batches'])==1


def test_crash_reservation_is_fail_closed_and_never_retried(case):
    real=inc.save_new
    def fail(path,value):
        if Path(path).name=='gate.json':raise RuntimeError('synthetic crash after reservation')
        return real(path,value)
    with mock.patch.object(inc,'save_new',side_effect=fail):
        with pytest.raises(RuntimeError,match='crash'):case.run()
    result=case.run();assert result['selected_formal_slots']==0
    registry=inc.load(case.root/'incremental/sg/registry.json')
    assert registry['batches'][0]['status']=='RESERVED'


@pytest.mark.parametrize('change',['source','receipt','output_attempt'])
def test_prelaunch_cpu_gate_rejects_changed_state_or_replay(case,change):
    result=case.run();gate=batch_report(result)['gate']['path']
    if change=='source':(case.f.source/'easygraph/gpu/session.py').write_text('changed')
    if change=='receipt':Path(case.f.qual['jobs'][-1]['receipt']).write_text('{}')
    if change=='output_attempt':Path(inc.option(case.f.formal['jobs'][0]['command'],'output')).mkdir()
    with pytest.raises(ValueError):inc.verify_cell(gate,'target',case.root/'gate_result.json')


def test_prelaunch_cpu_gate_is_not_a_performance_result(case):
    result=case.run();gate=batch_report(result)['gate']['path']
    proof=inc.verify_cell(gate,'target',case.root/'gate_result.json')
    assert proof['phase']=='cpu' and not proof['gpu_executed'] and not proof['performance_eligible']


def write_completed_state(case,admission,*,failure=False):
    plan_path=Path(admission['plan']['path']);plan=inc.load(plan_path);queue=plan_path.parent/'queue';queue.mkdir(exist_ok=True)
    rows=[]
    for job in plan['jobs']:
        receipt=None
        if job['phase']=='formal':
            case.f.write(job['receipt'],dict(status='PASS',phase='formal',synthetic_cpu_fixture=True))
            receipt=inc.bind(job['receipt'])
        rows.append(dict(id=job['id'],phase=job['phase'],outcome='PASS',receipt=receipt,execution_started=True,
                         maintenance={'status':'PASS'} if job['phase']=='formal' else None))
    if failure:rows[-1]['maintenance']={'status':'FAIL','error':'synthetic final maintenance failure'}
    state=dict(schema='campaign.queue.v5',plan=admission['plan'],jobs=plan['jobs'],completed=rows,
               status='FAILED_MAINTENANCE' if failure else 'PASS_REQUESTED_JOBS')
    case.f.write(queue/'state.json',state)
    return queue,state


def projection_locked(case,gate):
    with Path(case.request['controller_lock']).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return controller.project_completed(gate)


def test_projection_has_no_second_attempt_or_formal_count_and_preserves_failure(case):
    result=case.run();admission=batch_report(result);write_completed_state(case,admission,failure=True)
    gate=inc.load(admission['gate']['path']);projection=projection_locked(case,gate)
    assert projection['projection_only'] and projection['physical_worker_runs']==0 and not projection['performance_eligible']
    assert 'formal_measurements' not in projection and 'attempted' not in projection
    assert len(projection['completed'])==10
    assert projection['completed'][-1]['maintenance']['status']=='FAIL'
    assert [r['receipt']['path'] for r in projection['completed']]==[j['receipt'] for j in case.f.formal['jobs']]
    assert case.run()['selected_formal_slots']==0


def test_projection_requires_global_controller_lock(case):
    result=case.run();gate=inc.load(batch_report(result)['gate']['path'])
    with pytest.raises(ValueError,match='original controller lock'):controller.project_completed(gate)


def test_projection_rejects_nonprefix_or_forged_receipts(case):
    result=case.run();admission=batch_report(result);queue,state=write_completed_state(case,admission)
    gate=inc.load(admission['gate']['path']);state['completed'][2],state['completed'][3]=state['completed'][3],state['completed'][2]
    case.f.write(queue/'state.json',state)
    with pytest.raises(ValueError,match='prefix'):projection_locked(case,gate)


def test_projection_rejects_duplicate_ids_across_batches(case):
    result=case.run();admission=batch_report(result);_,state=write_completed_state(case,admission)
    gate=inc.load(admission['gate']['path']);registry=inc.registry_read(Path(gate['registry']).parent,gate['owner'])
    second=case.root/'synthetic_duplicate';second.mkdir();plan=inc.load(admission['plan']['path'])
    case.f.write(second/'plan.json',plan);copy_binding=inc.bind(second/'plan.json')
    state=copy.deepcopy(state);state['plan']=copy_binding;case.f.write(second/'queue/state.json',state)
    registry['batches'].append(dict(status='COMMITTED',plan=copy_binding))
    with mock.patch.object(inc,'registry_read',return_value=registry):
        with pytest.raises(ValueError,match='Duplicate completed'):projection_locked(case,gate)


def test_last_maintenance_failure_is_refreshed_by_wrapper_finally(case):
    result=case.run();admission=batch_report(result);plan=inc.load(admission['plan']['path'])
    gate=inc.load(admission['gate']['path'])
    fake=SimpleNamespace(run_process=lambda *unused:None)
    def failing_base(*unused):
        write_completed_state(case,admission,failure=True)
        raise RuntimeError('synthetic final maintenance failure')
    fake.run=failing_base
    loader=SimpleNamespace(exec_module=lambda module:module.__dict__.update(fake.__dict__))
    spec=SimpleNamespace(loader=loader)
    with mock.patch.object(controller.importlib.util,'spec_from_file_location',return_value=spec), \
         mock.patch.object(controller.importlib.util,'module_from_spec',return_value=SimpleNamespace()):
        with pytest.raises(RuntimeError,match='final maintenance failure'):
            controller.run(admission['gate']['path'],admission['plan']['path'],Path(admission['plan']['path']).parent/'queue',case.request['controller_lock'])
    projection=inc.load(gate['completed_projection'])
    assert projection['completed'][-1]['maintenance']['status']=='FAIL'
    assert projection['physical_worker_runs']==0


@pytest.mark.parametrize('change',['plan_bytes','copied_plan','copied_gate','other_output'])
def test_runtime_cannot_bypass_registered_plan_or_reserved_queue(case,change):
    result=case.run();admission=batch_report(result)
    plan_path=Path(admission['plan']['path']);spec_path=Path(admission['gate']['path']);output=plan_path.parent/'queue'
    if change=='plan_bytes':
        plan=inc.load(plan_path);plan['jobs']=plan['jobs'][1:];case.f.write(plan_path,plan)
    if change=='copied_plan':
        other=case.root/'copied_plan.json';other.write_bytes(plan_path.read_bytes());plan_path=other
    if change=='copied_gate':
        other=case.root/'copied_gate.json';other.write_bytes(spec_path.read_bytes());spec_path=other
    if change=='other_output':output=case.root/'unreserved_queue'
    with mock.patch.object(controller.importlib.util,'spec_from_file_location',side_effect=AssertionError('Must reject before loading controller')):
        with pytest.raises(ValueError):controller.run(spec_path,plan_path,output,case.request['controller_lock'])
    assert not output.exists()


def test_e4s_is_held_for_dedicated_recovery_controller(case):
    case.f.manifest['groups']['e4s']=copy.deepcopy(case.f.manifest['groups']['sg'])
    case.f.write(case.f.manifest_path,case.f.manifest)
    case.request.update(group='e4s',parent_manifest=inc.bind(case.f.manifest_path))
    result=case.run()
    assert result['selected_formal_slots']==0
    assert any(b.get('stage')=='e4s_incremental_controller' for b in result['blockers'])
    assert 'admission' not in result


def test_full_sha_rereads_every_time_and_same_size_mutation_is_rehashed(case):
    parent,_=LEDGER.parent_helper();cls=parent.Binder
    path=case.root/'cache_payload';path.write_bytes(b'abc');before=path.stat()
    binder=cls();original=binder.bind(path);binder.checked(original)
    assert binder.hashed_bytes==6
    path.write_bytes(b'xyz');os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(ValueError):binder.checked(original)
    assert binder.hashed_bytes==9
    other=cls();other.bind(path);assert other.hashed_bytes==3


def test_namespace_cannot_contain_legacy_queue(case):
    case.f.manifest['groups']['sg']['queue']=str(case.root/'incremental/sg/legacy_queue')
    case.f.write(case.f.manifest_path,case.f.manifest)
    case.request['parent_manifest']=inc.bind(case.f.manifest_path)
    with pytest.raises(ValueError,match='overlaps'):case.run()


@pytest.mark.parametrize('corrupt_last_array',[False,True])
def test_real_v8_supervisor_and_frozen_maintenance_preserve_receipts_and_final_failure(case,corrupt_last_array):
    """Actual subprocesses and frozen maintenance, using private CPU-only files.

    Only the synthetic fixture worker entrypoint is substituted. Its proof
    function is copied verbatim from the bound production SG proof reader, and
    the entire fixture matrix is admitted before executing any CPU process.
    """
    parent=Path(inc.__file__).resolve().parent.parent
    original=Path(case.f.manifest['proof_readers']['sg']['path']).read_text()
    definition=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='validate_proof')
    worker=case.root/'synthetic_cpu_worker.py'
    worker.write_text(ast.get_source_segment(original,definition)+'''\n
if __name__ == '__main__':
    from pathlib import Path
    import hashlib,json,os,sys
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output=Path(sys.argv[sys.argv.index('--output')+1]);output.mkdir()
    arrays=[]
    for name in ('a.f64','b.f64'):
        path=output/name;path.write_bytes(b'0123456789abcdef')
        arrays.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (output/'result.json').write_text(json.dumps(dict(status='PASS_CPU_FIXTURE',phase='formal',synthetic_cpu_fixture=True,arrays=arrays)))
    if os.environ.get('CORRUPT_LAST_ARRAY') == '1' and sys.argv[sys.argv.index('--repeat')+1] == '9':
        (output/'b.f64').write_bytes(b'fedcba9876543210')
''')
    case.f.manifest['proof_readers']['sg']=inc.bind(worker)
    gpu_lock=case.root/'private_cpu_only_gpu.lock';gpu_lock.touch()
    registry=case.root/'private_maintenance_registry.json'
    case.f.write(registry,dict(schema='completed.output.incremental.registry.v1',
        deduplicator=inc.bind(parent/'incremental_completed_output_v1.py'),
        gpu_lock=str(gpu_lock),controller_lock=case.request['controller_lock'],
        queue_states=[str(case.root/'old_queue/state.json')],index=str(case.root/'private_array_index.json')))
    helpers=[parent/name for name in ('completed_job_audit_maintenance_v2.py',
        'completed_output_maintenance_incremental_v1.py','incremental_completed_output_v1.py')]
    for job in case.f.qual['jobs']+case.f.formal['jobs']:
        job['command'][:2]=[sys.executable,str(worker)]
        job['environment']={'CUDA_VISIBLE_DEVICES':'','CORRUPT_LAST_ARRAY':'1' if corrupt_last_array else '0'}
        job['bindings']=[inc.bind(worker),inc.bind(sys.executable)]
        job['timeout']=30
        if job['phase']!='formal':continue
        job['expected_statuses']=['PASS_CPU_FIXTURE']
        output=case.root/'real_maintenance'/job['id']
        command=[sys.executable,'-B',str(helpers[0]),'--registry',str(registry),'--output',str(output),
            '--gpu-lock',str(gpu_lock),'--job-result',job['receipt'],'--dedup-wrapper',str(helpers[1])]
        job['maintenance']=dict(command=command,command_sha256=inc.digest(command),
            bindings=[inc.bind(p) for p in [Path(sys.executable),registry]+helpers],
            gpu_lock=str(gpu_lock),environment={'CUDA_VISIBLE_DEVICES':''},receipt=str(output/'receipt.json'),
            expected_statuses=['PASS_INCREMENTAL_HARDLINKED_OUTPUTS'],timeout=30)
    case.fixture.sync_plans();case.request['parent_manifest']=inc.bind(case.f.manifest_path)
    result=case.run();admission=batch_report(result)
    output=Path(admission['plan']['path']).parent/'queue'
    if corrupt_last_array:
        with pytest.raises(RuntimeError,match='Maintenance command failed'):
            controller.run(admission['gate']['path'],admission['plan']['path'],output,case.request['controller_lock'])
    else:
        controller.run(admission['gate']['path'],admission['plan']['path'],output,case.request['controller_lock'])
    actual=inc.load(output/'state.json')
    projection=inc.load(case.root/'old_queue/state.json')
    assert actual['attempted']==11 and len(actual['completed'])==11
    assert actual['maintenance_passed']==(9 if corrupt_last_array else 10)
    assert actual['maintenance_failed']==int(corrupt_last_array)
    assert actual['status']==('FAILED_MAINTENANCE' if corrupt_last_array else 'PASS_REQUESTED_JOBS')
    assert projection['completed']==actual['completed'][1:]
    assert projection['physical_worker_runs']==0 and projection['projection_only'] and not projection['performance_eligible']
    assert all(row['maintenance']['worker_gpu_lock_observed_released'] for row in projection['completed'])
    for row in projection['completed']:
        inc.checked(row['receipt'])
        supervision=inc.load(inc.checked(row['process_supervision']))
        assert supervision['status']=='PASS_PROCESS_EXIT' and not supervision.get('surviving_owned_processes')
    first=Path(case.f.formal['jobs'][0]['receipt']).parent
    assert (first/'a.f64').stat().st_ino==(first/'b.f64').stat().st_ino
    assert projection['completed'][-1]['maintenance']['status']==('FAIL' if corrupt_last_array else 'PASS')
    assert 'finished_unix' in projection['completed'][-1]['maintenance']
    if not corrupt_last_array:
        last=case.f.formal['jobs'][-1]['maintenance']['receipt']
        discovery=inc.load(Path(last).parent/'discovery.json')
        assert len(discovery['completed_receipts'])==10 and discovery['array_paths']==20
        assert len({Path(j['receipt']).with_name('a.f64').stat().st_ino for j in case.f.formal['jobs']})==1
