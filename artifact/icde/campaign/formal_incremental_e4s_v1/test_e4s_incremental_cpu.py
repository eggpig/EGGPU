"""Private CPU fixtures exercise actual E4S recovery, controller and maintenance."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import ast
import concurrent.futures
import copy
import fcntl
import importlib.util
import json
import os
import sys

import pytest
import admit_e4s_incremental as inc
import run_incremental_queue as controller

HERE=Path(__file__).resolve().parent
C=HERE.parent
sys.path.insert(0,str(C/'e4s_recoverable_outputs_v1'))
sys.path.insert(0,str(C/'formal_admission_d84_v2'))
import test_admission_integration as fixtures
import archive_outputs as archive
import export_eligible


@pytest.fixture
def case():
    f=fixtures.FullAdmission('runTest');f.setUp();root=f.root
    group=f.a.manifest['groups']['e4s']
    group.update(admitted_output=str(root/'old_admitted'),queue=str(root/'old_queue'),controller_log=str(root/'old_controller.log'))
    f.a.manifest['controller_lock']=str(root/'controller.lock')
    worker=f.a.manifest['proof_readers']['e4s']['path']
    for job in f.jobs+f.formals:
        job['command'][1]=worker
        job['timeout']=3600
    request=dict(schema='eggpu.formal.incremental.request.v1',group='e4s',
        namespace_root=str(root/'incremental'),controller_lock=str(root/'controller.lock'),
        controller=inc.bind(HERE/'run_e4s_formal_queue_v1.py'),
        supervisor=inc.bind(HERE/'owned_process_supervisor_v2.py'),
        python=inc.bind(sys.executable),python_invocation=sys.executable,
        additional_prior_states=[],additional_legacy_directories=[],dispositions=[],archive_qualified_outputs=True)
    obj=SimpleNamespace(f=f,root=root,request=request,n=0)
    def sync():
        group.update(pending=f.a.save('e4s_pending.json',dict(source_tree_sha256='fixture_d84',jobs=f.formals,environment={})),
                     qualifications=[f.a.save('e4s_qual.json',dict(source_tree_sha256='fixture_d84',jobs=f.jobs,environment={}))],expected_formal_jobs=len(f.formals))
        f.a.write(f.a.manifest_path,f.a.manifest)
        request['parent_manifest']=inc.bind(f.a.manifest_path)
        f.a.write(root/'request.json',request)
    def run(admit=True):
        sync();out=root/f'observation_{obj.n}';obj.n+=1
        return inc.run(root/'request.json',out,admit=admit)
    obj.sync=sync;obj.run=run;sync()
    try:yield obj
    finally:f.tearDown()


def formal_jobs(plan):return [j for j in plan['jobs'] if j['phase']=='formal']


def test_original_ten_worker_jobs_preserved_and_mandatory_decorator_applied(case):
    report=case.run();assert report['selected_formal_slots']==10
    plan=inc.load(report['admission']['plan']['path'])
    assert report['original_formal_slots']==10 and report['workers_launched']==0
    assert plan['postprocess_contract']=='e4s_recoverable_outputs_v1'
    allowed={'maintenance','required_postprocess','required_archive_receipt','original_maintenance_preserved_in'}
    for original,actual in zip(case.f.formals,formal_jobs(plan)):
        assert {k:v for k,v in original.items() if k not in allowed}=={k:v for k,v in actual.items() if k not in allowed}
        post=inc.load(inc.checked(actual['original_maintenance_preserved_in']))
        assert post['original_maintenance']==original['maintenance']
        assert actual['required_postprocess']=='e4s_recoverable_outputs_v1'
    assert plan['jobs'][0]['phase']=='cpu' and plan['jobs'][0]['required_cpu_gate']=='e4s_incremental_cell_v1'


def test_new_qualified_sidecars_created_before_deep_audit_under_original_lock(case):
    for job in case.f.jobs:Path(job['receipt']).with_name('recoverable_outputs.json').unlink()
    raw={j['receipt']:Path(j['receipt']).read_bytes() for j in case.f.jobs}
    original=inc.prepare_qualification_archives
    def probe(*args):
        with Path(case.request['controller_lock']).open('a+') as lock:
            with pytest.raises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return original(*args)
    with mock.patch.object(inc,'prepare_qualification_archives',side_effect=probe):report=case.run()
    assert report['selected_formal_slots']==10
    recovery=inc.load(report['qualification_recovery']['path'])
    assert [r['status'] for r in recovery['rows']]==['PASS_QUALIFICATION_RECOVERY']*2
    assert all(Path(p).read_bytes()==data for p,data in raw.items())
    assert recovery['worker_reruns']==0 and not recovery['gpu_executed']


def test_missing_sidecar_without_archive_authorization_stays_unqualified(case):
    Path(case.f.jobs[-1]['receipt']).with_name('recoverable_outputs.json').unlink()
    case.request['archive_qualified_outputs']=False
    report=case.run();assert report['selected_formal_slots']==0 and report['original_formal_slots']==10
    assert report['complete_ledger']['cells'][0]['status']!='QUALIFIED'
    assert 'admission' not in report


def test_changed_recovery_object_blocks_current_qualified_cell(case):
    sidecar=inc.load(Path(case.f.jobs[-1]['receipt']).with_name('recoverable_outputs.json'))
    path=Path(sidecar['vectors'][0]['object']['path']);path.chmod(0o600);path.write_bytes(b'corrupt')
    report=case.run();assert report['selected_formal_slots']==0
    recovery=inc.load(report['qualification_recovery']['path'])
    assert any(r['status']=='FAIL_RECOVERY' for r in recovery['rows'])


def test_gate_revalidates_recoverable_objects_after_admission(case):
    report=case.run();sidecar=inc.load(Path(case.f.jobs[-1]['receipt']).with_name('recoverable_outputs.json'))
    path=Path(sidecar['vectors'][0]['object']['path']);path.chmod(0o600);path.write_bytes(b'corrupt')
    with pytest.raises(ValueError):inc.verify_cell(report['admission']['gate']['path'],'target',case.root/'gate.json')


def test_gate_preserves_exact_admitted_sidecar_bytes_not_only_parsed_fields(case):
    report=case.run();path=Path(case.f.jobs[-1]['receipt']).with_name('recoverable_outputs.json')
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='Bound file changed'):
        inc.verify_cell(report['admission']['gate']['path'],'target',case.root/'gate.json')


def test_ordinary_controller_cannot_replace_dedicated_recovery_controller(case):
    case.request['controller']=inc.bind(C/'run_campaign_queue_v8.py')
    case.request['supervisor']=inc.bind(C/'owned_process_supervisor_v2.py')
    with pytest.raises(ValueError,match='dedicated'):case.run()


def test_integer_is_not_an_explicit_boolean_archive_policy(case):
    case.request['archive_qualified_outputs']=1
    with pytest.raises(ValueError,match='boolean'):case.run()


def test_failed_prior_worker_or_maintenance_output_never_retried(case):
    Path(case.f.formals[0]['receipt']).parent.mkdir()
    report=case.run();assert report['selected_formal_slots']==0 and 'admission' not in report
    assert report['complete_ledger']['cells'][0]['incremental_scheduling']['status']=='EXISTING_ATTEMPT_OR_CLAIM'


def second_cell(case,status='RUNNING'):
    target=copy.deepcopy(case.f.jobs[-1]);target['id']='second_target'
    output=case.root/'second_target';target['receipt']=str(output/'result.json')
    target['command'][target['command'].index('--output')+1]=str(output)
    raw=inc.load(case.f.jobs[-1]['receipt']);raw['status']=status;case.f.a.write(target['receipt'],raw)
    originals=copy.deepcopy(case.f.formals);extra=[]
    for job in originals:
        job['id']='second_'+job['id'];output=case.root/job['id'];job['receipt']=str(output/'result.json');job['depends_on']=['second_target']
        job['command'][job['command'].index('--output')+1]=str(output)
        job['command'][job['command'].index('--target-proof')+1]=target['receipt'];extra.append(job)
    case.f.jobs.append(target);case.f.formals=[j for pair in zip(case.f.formals,extra) for j in pair]
    return target


@pytest.mark.parametrize('status',['RUNNING','FAILED','FAIL_QUALITY'])
def test_unresolved_peer_keeps_twenty_slots_without_blocking_qualified_cell(case,status):
    second_cell(case,status);report=case.run()
    assert report['original_formal_slots']==20 and report['selected_formal_slots']==10
    assert report['selected_cell_ids']==['target']
    assert len(report['complete_ledger']['cells'])==2


def test_later_qualified_cell_archive_and_ten_repeats_never_replay_old_cell(case):
    second=second_cell(case);first=case.run()
    raw=inc.load(second['receipt']);raw['status']='PASS_E4S_QUALIFICATION';case.f.a.write(second['receipt'],raw)
    report=case.run();assert report['selected_cell_ids']==['second_target']
    assert Path(second['receipt']).with_name('recoverable_outputs.json').is_file()
    assert report['admission']['plan']!=first['admission']['plan']
    assert case.run()['selected_formal_slots']==0


def test_concurrent_admissions_reserve_cell_once(case):
    def run(i):return inc.run(case.root/'request.json',case.root/f'concurrent{i}',admit=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:reports=list(pool.map(run,[0,1]))
    assert sorted(r['selected_formal_slots'] for r in reports)==[0,10]


@pytest.mark.parametrize('change',['plan','output','gate','source','worker_output'])
def test_runtime_or_cell_gate_rejects_changed_reserved_identity(case,change):
    report=case.run();admission=report['admission'];plan=Path(admission['plan']['path']);gate=Path(admission['gate']['path']);output=plan.parent/'queue'
    if change=='plan':p=inc.load(plan);p['jobs']=p['jobs'][1:];case.f.a.write(plan,p)
    if change=='output':output=case.root/'other_queue'
    if change=='gate':other=case.root/'copied_gate.json';other.write_bytes(gate.read_bytes());gate=other
    if change=='source':(case.f.a.source/'easygraph/gpu/session.py').write_text('changed')
    if change=='worker_output':Path(case.f.formals[0]['receipt']).parent.mkdir()
    if change in ('source','worker_output'):
        with pytest.raises(ValueError):inc.verify_cell(gate,'target',case.root/'gate.json')
    else:
        with mock.patch.object(controller.importlib.util,'spec_from_file_location',side_effect=AssertionError('Must reject before loading controller')):
            with pytest.raises(ValueError):controller.run(gate,plan,output,case.request['controller_lock'])


def prepare_real_cpu_workers(case,mode):
    production=Path(case.f.a.manifest['proof_readers']['e4s']['path']).read_text()
    definition=next(n for n in ast.parse(production).body if isinstance(n,ast.FunctionDef) and n.name=='proof')
    worker=case.root/'synthetic_e4s_cpu_worker.py'
    worker.write_text(ast.get_source_segment(production,definition)+'''\n
if __name__ == '__main__':
    from pathlib import Path
    import json,os,sys
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output=Path(sys.argv[sys.argv.index('--output')+1]);output.mkdir()
    target=Path(sys.argv[sys.argv.index('--target-proof')+1]);raw=json.loads(target.read_text())
    repeat=int(sys.argv[sys.argv.index('--repeat')+1]);raw.update(phase='formal',status='PASS_E4S_FORMAL',repeat=repeat,synthetic_cpu_fixture=True)
    if repeat==9 and os.environ['FIXTURE_MODE']=='archive_fail':raw['checks'][0]['output_sha256']='0'*64
    if repeat==9 and os.environ['FIXTURE_MODE']=='worker_fail':raw.update(status='FAILED',quality_pass=False)
    (output/'result.json').write_text(json.dumps(raw))
''')
    case.f.a.manifest['proof_readers']['e4s']=inc.bind(worker)
    gpu=case.root/'private_cpu_only_gpu.lock';gpu.touch()
    registry=case.root/'maintenance_registry.json'
    case.f.a.write(registry,dict(schema='completed.output.incremental.registry.v1',
        deduplicator=inc.bind(C/'incremental_completed_output_v1.py'),gpu_lock=str(gpu),
        controller_lock=case.request['controller_lock'],queue_states=[str(case.root/'old_queue/state.json')],
        index=str(case.root/'array_index.json')))
    helpers=[C/name for name in ['completed_job_audit_maintenance_v2.py',
        'completed_output_maintenance_incremental_v1.py','incremental_completed_output_v1.py']]
    fail=case.root/'fail_original_maintenance.py';fail.write_text("raise SystemExit('synthetic original maintenance failure')\n")
    for job in case.f.jobs+case.f.formals:
        job['command'][:2]=[sys.executable,str(worker)];job['bindings']=[inc.bind(worker),inc.bind(sys.executable)]
        job['environment']={'CUDA_VISIBLE_DEVICES':'','FIXTURE_MODE':mode};job['timeout']=30
        if job['phase']!='formal':continue
        output=case.root/'original_maintenance'/job['id']
        argv=[sys.executable,'-B',str(helpers[0]),'--registry',str(registry),'--output',str(output),
              '--gpu-lock',str(gpu),'--job-result',job['receipt'],'--dedup-wrapper',str(helpers[1])]
        if mode=='original_fail' and job['id']=='formal9':argv=[sys.executable,str(fail),'--output',str(output)]
        job['maintenance']=dict(command=argv,command_sha256=inc.digest(argv),gpu_lock=str(gpu),
            bindings=[inc.bind(p) for p in [Path(sys.executable),registry,fail]+helpers],
            receipt=str(output/'receipt.json'),expected_statuses=['PASS_INCREMENTAL_HARDLINKED_OUTPUTS'],
            environment={'CUDA_VISIBLE_DEVICES':''},timeout=30)


@pytest.mark.parametrize('mode',['success','archive_fail','original_fail','worker_fail'])
def test_real_dedicated_controller_archives_maintenance_and_final_failure(case,mode):
    prepare_real_cpu_workers(case,mode);report=case.run();admission=report['admission']
    plan=inc.load(admission['plan']['path']);queue=Path(admission['plan']['path']).parent/'queue'
    if mode in ('archive_fail','original_fail'):
        with pytest.raises(RuntimeError,match='Maintenance command failed'):
            controller.run(admission['gate']['path'],admission['plan']['path'],queue,case.request['controller_lock'])
    else:controller.run(admission['gate']['path'],admission['plan']['path'],queue,case.request['controller_lock'])
    actual=inc.load(queue/'state.json');projection=inc.load(case.root/'old_queue/state.json')
    assert actual['attempted']==11 and actual['formal_attempted']==10
    assert actual['formal_measurements']==(10 if mode=='success' else 9)
    assert actual['worker_passed']==(9 if mode=='worker_fail' else 10)
    assert projection['completed']==actual['completed'][1:] and projection['physical_worker_runs']==0
    assert not projection['performance_eligible']
    for row,job in zip(projection['completed'],formal_jobs(plan)):
        inc.checked(row['receipt'])
        assert 'finished_unix' in row['maintenance']
        if row['outcome']=='PASS':
            proof=export_eligible.eligible(job['receipt'],job['maintenance']['receipt'],case.f.a.manifest['archive_contract']['helper'])
            assert proof['status']=='PASS_E4S_FORMAL_EXPORT_ELIGIBILITY'
        else:
            with pytest.raises(ValueError):export_eligible.eligible(job['receipt'],job['maintenance']['receipt'],case.f.a.manifest['archive_contract']['helper'])
    if mode in ('archive_fail','original_fail'):
        last=projection['completed'][-1]
        assert last['worker_outcome']=='PASS' and last['outcome']=='FAIL_REQUIRED_POSTPROCESSING'
        assert last['failure_stage']==('archive' if mode=='archive_fail' else 'original_maintenance')
        assert actual['status']=='FAILED_MAINTENANCE'
    elif mode=='worker_fail':
        assert actual['worker_failed']==1 and actual['archive_failed']==0
        assert projection['completed'][-1]['outcome']=='FAIL'
    assert case.run()['selected_formal_slots']==0


def test_dedicated_controller_changes_only_cpu_gate_admission_and_counts():
    old=(C/'e4s_recoverable_outputs_v1/run_e4s_formal_queue_v1.py').read_text()
    new=(HERE/'run_e4s_formal_queue_v1.py').read_text()
    old=old.replace("  if phases[name]!='formal' or job.get('required_postprocess')!='e4s_recoverable_outputs_v1' or not job.get('maintenance'):raise ValueError('Every job must be formal E4S with required recovery maintenance')",
'''  if phases[name]=='cpu':
   if job.get('required_cpu_gate')!='e4s_incremental_cell_v1' or job.get('maintenance') or job.get('depends_on') or job.get('expected_statuses')!=['PASS_CURRENT_INCREMENTAL_CELL_GATE']:raise ValueError('Only explicit E4S incremental CPU cell gates are allowed')
   if job['command'][2]!=str(Path(__file__).with_name('cell_gate.py')) or job.get('environment',{}).get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU gate entrypoint or CUDA visibility differs')
  elif phases[name]!='formal' or job.get('required_postprocess')!='e4s_recoverable_outputs_v1' or not job.get('maintenance'):raise ValueError('Every formal job must be E4S with required recovery maintenance')''')
    old=old.replace("if item.get('execution_started'):state['worker_passed' if item['outcome']=='PASS' else 'worker_failed']+=1",
                    "if item.get('execution_started') and phase=='formal':state['worker_passed' if item['outcome']=='PASS' else 'worker_failed']+=1")
    assert old==new
    assert (HERE/'owned_process_supervisor_v2.py').read_bytes()==(C/'e4s_recoverable_outputs_v1/owned_process_supervisor_v2.py').read_bytes()
