"""CPU-only controller envelope for the frozen state completion qualification.

Does not launch a worker. Original job argv/env/receipt/dependencies stay equal.
The worker remains responsible for its bound host/device resource admission.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os


def require(value, message):
    if not value:
        raise ValueError(message)


def bind(path):
    path = Path(path).resolve()
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'File changed while hashing')
    return dict(path=str(path), bytes=before.st_size, sha256=h.hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def option(command, name):
    return command[command.index('--' + name) + 1]


def command_digest(command):
    return hashlib.sha256(json.dumps(command, separators=(',', ':')).encode()).hexdigest()


def capacity(spec, inputs, trace):
    """Conservative uncompressed bytes, including independent oracle and rebuild.

    Rebuild payload uses exact bound host partition/projection files and follows
    the worker's version-retention counts. Never credit compression/dedup.
    """
    require(spec['phase'] in ('tiny', 'target'), 'Qualification envelope only')
    require(inputs['qualification_only'] == (spec['phase'] == 'tiny'), 'Input phase differs')
    names = spec['datasets']
    require(names and len(names) == len(set(names)) and set(names) <= set(inputs['datasets']), 'Invalid dataset pool')
    rounds = 2 if inputs['qualification_only'] else 20 if spec['experiment'] == 'E3' else 4 if spec['experiment'] == 'E3R' else 8
    events = [] if spec['action'] == 'pilot' else trace(spec['experiment'], names, spec['order'], spec['phase'] == 'tiny')
    sizes = [int(inputs['datasets'][e['dataset']]['n']) * 8 for e in events if e['kind'] == 'query']
    host = 4. + sum(row['n'] * 8 * 4 * rounds + row['m'] * 8 *
        (1 if spec['experiment'] == 'E4' else rounds + 1) + row['m'] * 4 + row['n'] * 24
        for name, row in inputs['datasets'].items() if name in names) / (1 << 30)
    payload = 0
    if spec['arm'] == 'Rebuild':
        require(spec['experiment'] == 'E3', 'Rebuild only supported for E3')
        payload = sum(sum(x['bytes'] for x in row['manifest_files']) +
            sum(x['bytes'] for projection in row['projections'].values() for x in projection['files'])
            for name, row in inputs['datasets'].items() if name in names)
        host += (rounds + 2) * payload / (1 << 30)
    output = sum(sizes)
    oracle = output  # Every query could need a distinct independent reference.
    temporary = max(sizes, default=0)
    rebuild = rounds * payload if spec['action'] != 'pilot' else 0
    reserve = dict(complete_output_upper_bytes=output, new_oracle_upper_bytes=oracle,
        serialization_temporary_upper_bytes=temporary,
        rebuild_temporary_payload_upper_bytes=rebuild, rebuild_per_version_payload_bytes=payload,
        rebuild_versions=rounds if rebuild else 0, safety_reserve_bytes=2 << 30,
        dedup_savings_assumed=False, compression_savings_assumed=False,
        complete_query_outputs=len(sizes), worker_required_host_free_gib=host)
    reserve['min_free_bytes'] = output + oracle + temporary + rebuild + (2 << 30)
    return reserve


def derive(args):
    campaign = args.campaign.resolve()
    original_path = args.base_plan.resolve()
    original_binding = bind(original_path)
    require(original_binding['sha256'] == args.base_sha256, 'Original qualification plan SHA differs')
    plan = read(original_path)
    require(len(plan['jobs']) == args.expected_jobs, 'Unexpected complete qualification matrix size')
    runner = args.runner.resolve()
    require(not args.output.resolve().is_relative_to(runner), 'Envelope must be outside frozen runner directory')
    freeze_path = runner / 'freeze.json'
    freeze = read(freeze_path)
    # The runner is immutable; validate the exact protocol file against its freeze.
    protocol_path = runner / 'state_protocol.py'
    frozen_files = freeze.get('runner_files', freeze.get('files', {}))
    protocol_binding = bind(protocol_path)
    expected = frozen_files.get('state_protocol.py')
    require(expected is not None, 'Runner freeze lacks state_protocol.py')
    require(all(protocol_binding[k] == expected[k] for k in ('bytes', 'sha256')), 'Frozen protocol changed')
    module_spec = importlib.util.spec_from_file_location('_state_completion_capacity_trace', protocol_path)
    protocol = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(protocol)
    controller = campaign / 'run_campaign_queue_v8.py'
    supervisor = campaign / 'owned_process_supervisor_v2.py'
    audit = campaign / 'completed_job_audit_maintenance_v2.py'
    wrapper = campaign / 'completed_output_maintenance_incremental_v1.py'
    deduplicator = campaign / 'incremental_completed_output_v1.py'
    python = str(args.python)
    interpreter_binding = bind(python)
    helpers = {x.name: bind(x) for x in (controller, supervisor, audit, wrapper, deduplicator)}
    out = args.output.resolve()
    require(not out.exists(), 'Never overwrite a derived plan or attempt namespace')
    queue = out / 'queue'
    registry_path = out / 'queue_registry.json'
    controller_lock = str(campaign / 'root_controller.lock')
    gpu_lock = str(campaign.parents[1] / 'gpu_experiment.lock')
    registry = dict(schema='completed.output.incremental.registry.v1', deduplicator=helpers[deduplicator.name],
        queue_states=[str(queue / 'state.json')], index=str(out / 'completed_output_index.json'),
        controller_lock=controller_lock, gpu_lock=gpu_lock,
        scope='Only this new qualification queue; shared state outputs already use lossless content-addressed OutputStore.')
    jobs, resources, inputs_cache, known = [], {}, {}, set()
    for old in plan['jobs']:
        require(old['id'] not in known and set(old.get('depends_on', [])) <= known, 'Invalid frozen dependency order')
        known.add(old['id'])
        spec_path = Path(option(old['command'], 'spec')).resolve()
        spec_binding = bind(spec_path)
        require(spec_binding in old['bindings'], 'Job spec is not bound by original job')
        spec = read(spec_path)
        require(spec['id'] == old['id'] and spec['receipt'] == old['receipt'], 'Job/spec identity differs')
        require(spec['dependencies'] == old.get('depends_on', []), 'Job/spec dependencies differ')
        require(not Path(old['receipt']).parent.exists(), 'Qualification attempt output already exists')
        key = spec['inputs']['path']
        if key not in inputs_cache:
            require(bind(key) == spec['inputs'], 'State input metadata binding differs')
            inputs_cache[key] = read(key)
        reserve = capacity(spec, inputs_cache[key], protocol.trace)
        job = copy.deepcopy(old)
        job['phase'] = spec['phase']
        job['disk_guard'] = dict(path=str(campaign), **reserve)
        resources[old['id']] = reserve
        destination = out / 'maintenance' / old['id']
        command = [python, '-B', str(audit), '--registry', str(registry_path), '--output', str(destination),
            '--gpu-lock', gpu_lock, '--job-result', old['receipt'], '--dedup-wrapper', str(wrapper)]
        job['maintenance'] = dict(command=command, command_sha256=command_digest(command),
            bindings=[interpreter_binding, helpers[audit.name], helpers[wrapper.name], helpers[deduplicator.name]],
            gpu_lock=gpu_lock, environment={'CUDA_VISIBLE_DEVICES': ''}, receipt=str(destination / 'receipt.json'),
            expected_statuses=['PASS_INCREMENTAL_HARDLINKED_OUTPUTS'], timeout=3600)
        require({k:v for k,v in job.items() if k not in ('phase','disk_guard','maintenance')} == old,
                'Original qualification execution contract changed')
        jobs.append(job)
    out.mkdir(parents=True, exist_ok=False)
    save(registry_path, registry)
    registry_binding = bind(registry_path)
    for job in jobs:
        job['maintenance']['bindings'].append(registry_binding)
    derived = dict(plan, jobs=jobs, original_plan=original_binding, storage_poll_seconds=15,
        queue_controller=helpers[controller.name], queue_supervisor=helpers[supervisor.name],
        derivation=bind(__file__), frozen_protocol=protocol_binding, runner_freeze=bind(freeze_path),
        storage_guard_rule='complete uncompressed outputs + all independent oracle vectors + largest serialization temporary + Rebuild full-version temporary payloads + 2 GiB; no dedup/compression credit',
        worker_resource_policy='Original worker host/device admission unchanged, including Rebuild retained host versions.')
    save(out / 'qualification_plan.json', derived)
    receipt = dict(status='PASS_CPU_DERIVED_STATE_QUALIFICATION_NOT_LAUNCHED', gpu_executed=False,
        workers_launched=0, formal_measurements=0, jobs=len(jobs),
        phases={phase:sum(j['phase']==phase for j in jobs) for phase in ('tiny','target')},
        plan=bind(out / 'qualification_plan.json'), registry=registry_binding,
        controller=helpers[controller.name], supervisor=helpers[supervisor.name],
        capacities=resources, original_job_contract='All original fields equal; only phase/disk_guard/maintenance added.',
        startup=[python,'-B',str(controller),'--plan',str(out/'qualification_plan.json'),
                 '--output',str(queue),'--controller-lock',controller_lock])
    save(out / 'receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','base-plan','runner','output','python'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--base-sha256', required=True)
    parser.add_argument('--expected-jobs', type=int, default=664)
    args = parser.parse_args()
    result = derive(args)
    print(json.dumps({k:result[k] for k in ('status','jobs','phases','plan','registry','startup')}))
