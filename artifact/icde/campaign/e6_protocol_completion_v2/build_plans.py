"""CPU-only candidate plan generator. --draft emits non-executable job_specs.

Run without --draft on the experiment host after copying this candidate. All
bound old/new files are checked before producing a fresh qualification plan.
It neither launches a controller nor admits formal measurements.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import e6_cpu_reference as reference
import e6_worker as worker

HERE = Path(__file__).resolve().parent
RUNNER_FILES = ('e6_worker.py', 'e6_reused.py', 'e6_reuse_provenance.json',
                'sg_worker.py', 'cugraph_csr_adapter.py', 'e6_cpu_reference.py',
                'legacy_cpu_reference.py', 'legacy_contract.json', 'build_plans.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def option(command, name):
    return command[command.index('--' + name) + 1]


def set_option(command, name, value):
    flag = '--' + name
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def configurations():
    rows = [dict(system='eggpu', ablate='none', coverage_backend='none', gpu_count=p,
                 panel='shared_native_control' if p == 1 else 'fixed_backend_scaling') for p in (1, 2, 4, 8)]
    for backend in ('igraph', 'cugraph_composite'):
        for ablate in ('structural', 'closeness', 'structural_closeness'):
            rows.append(dict(system='eggpu', ablate=ablate, coverage_backend=backend,
                             gpu_count=1, panel='coverage_2x2_' + backend))
    rows.extend(dict(system='eggpu', ablate=ablate, coverage_backend='igraph', gpu_count=1,
                     panel='legacy_traversal_supplement') for ablate in ('bfs', 'wcc'))
    rows.extend(dict(system=system, ablate='none', coverage_backend='none',
                     gpu_count=0 if system == 'igraph' else 1, panel='complete_external')
                for system in ('igraph', 'cugraph_igraph', 'cugraph_composite'))
    return rows


def config_key(row):
    return f"{row['system']}_{row['ablate']}_{row['coverage_backend']}_p{row['gpu_count']}"


def validate_matrix(jobs, formal=False):
    expected_phase = {'formal'} if formal else {'tiny', 'target'}
    if any(j['phase'] not in expected_phase for j in jobs):
        raise ValueError('Only the declared qualification or formal phases are allowed')
    wanted = {(d, config_key(c), r) for d in worker.DATASETS
              for c in configurations() for r in (range(10) if formal else [0])}
    target = [j for j in jobs if j['phase'] != 'tiny']
    got = [(j['dataset'], config_key(j), j['repeat']) for j in target]
    if len(got) != len(set(got)) or set(got) != wanted:
        raise ValueError('Fixed graph/configuration/repeat ledger has omissions or duplicates')
    if not formal:
        tiny = [config_key(j) for j in jobs if j['phase'] == 'tiny']
        if len(tiny) != 15 or set(tiny) != {config_key(c) for c in configurations()}:
            raise ValueError('Every configuration requires exactly one current tiny')
    for job in jobs:
        command = job['command']
        if job.get('command_sha256') != digest(command):
            raise ValueError('Command binding changed')
        parsed = worker.parser().parse_args(command[3:])
        parsed.gpu_uuids = parsed.gpu_uuids.split(',') if parsed.gpu_uuids else []
        worker.options_valid(parsed)
        if option(command, 'timeout') != '3600':
            raise ValueError('Every complete application has the identical 3600 second limit')
        if job['timeout'] < 3 * 3600:
            raise ValueError('Supervisor must permit all three independent complete calls')
        if (option(command, 'system'), option(command, 'ablate'), option(command, 'coverage-backend'),
                int(option(command, 'gpu-count')), option(command, 'dataset'), option(command, 'phase'),
                int(option(command, 'repeat'))) != (
                job['system'], job['ablate'], job['coverage_backend'], job['gpu_count'],
                job['dataset'], job['phase'], job['repeat']):
            raise ValueError('Command/ledger identity mismatch')
        root = Path(option(command, 'output')).parent
        tiny_id = f"tiny_{worker.TINY}_{config_key(job)}_r00"
        target_id = f"target_{job['dataset']}_{config_key(job)}_r00"
        requirements = [] if job['phase'] == 'tiny' else [('tiny-proof', tiny_id)]
        if job['phase'] == 'formal':
            requirements.append(('target-proof', target_id))
        if job['depends_on'] != [ident for _, ident in requirements]:
            raise ValueError('Dependencies must be this exact configuration and current protocol')
        for flag, ident in requirements:
            if option(command, flag) != str(root / ident / 'result.json'):
                raise ValueError('A qualification proof points to another configuration or protocol')
        for flag in ('tiny-proof', 'target-proof'):
            if flag not in {f for f, _ in requirements} and '--' + flag in command:
                raise ValueError('Unexpected inherited qualification proof')
        if job.get('expected_statuses') != ['PASS' if formal else 'PASS_QUALIFICATION']:
            raise ValueError('Failures and qualification are not successful formal measurements')
    return True


def check_template_binding(item):
    # Preserve virtual-environment invocation paths. Hash dereferenced bytes,
    # but do not turn envs/core/bin/python into the base interpreter command.
    if (set(item) != {'path','bytes','sha256'} or not Path(item['path']).is_absolute()
            or type(item['bytes']) is not int or item['bytes'] < 0):
        raise ValueError('Invalid inherited template binding')
    actual = reference.binding(item['path'])
    if any(actual[k] != item[k] for k in ('bytes','sha256')):
        raise ValueError('Inherited template bytes changed: ' + item['path'])
    return Path(item['path'])


def build(args):
    original_path = args.original.resolve()
    original = json.loads(original_path.read_text())
    templates = [j for j in original['jobs'] if j['phase'] == 'tiny']
    if len(templates) != 11:
        raise ValueError('Expected the complete original 11-configuration source-bound plan')
    native = {j['gpu_count']: j for j in templates if j['system'] == 'eggpu' and j['ablate'] == 'none'}
    external = {j['system']: j for j in templates if j['system'] != 'eggpu'}
    if set(native) != {1, 2, 4, 8} or set(external) != {'igraph', 'cugraph_igraph', 'cugraph_composite'}:
        raise ValueError('Original system/resource matrix is incomplete')
    campaign = args.campaign
    runtime_runner = campaign / HERE.name
    out = args.output.resolve()
    if not out.is_relative_to(HERE):
        raise ValueError('Candidate artifacts must stay under its independent directory')
    if not args.draft and runtime_runner.resolve() != HERE:
        raise ValueError('Executable generation must run at its bound deployment path')
    runtime_out = runtime_runner / out.relative_to(HERE)
    legacy_ref = Path(option(native[1]['command'], 'reference-manifest'))
    if args.draft:
        local_ref = original_path.parents[2] / 'e6_reference_binding_v2/independent_reference_manifest.json'
        if not local_ref.is_file():
            raise ValueError('Cannot locate the explicitly bound original oracle manifest')
    else:
        local_ref = legacy_ref
    old_ref = reference.binding(local_ref)
    expected_ref = [b for b in native[1]['bindings'] if b['path'] == str(legacy_ref)]
    if len(expected_ref) != 1 or any(old_ref[k] != expected_ref[0][k] for k in ('bytes', 'sha256')):
        raise ValueError('Original template oracle binding changed')
    bridge = reference.manifest_record(local_ref, worker.CONTRACT)
    bridge['legacy_manifest']['path'] = str(legacy_ref)
    # Fail before creating any executable plan if any inherited input/source/DSO/
    # maintenance/reference evidence is absent or changed.
    if not args.draft:
        seen = set()
        for template in templates:
            for item in template['bindings'] + template.get('maintenance', {}).get('bindings', []):
                if item['path'].endswith('queue_registry.json'):
                    continue  # Replaced by this candidate's independent registry.
                if item['path'] not in seen:
                    check_template_binding(item); seen.add(item['path'])
        old = json.loads(local_ref.read_text())
        reference.check(old['producer']['worker']); reference.check(old['producer']['reused'])
        for row in old['references']:
            reference.check(row['receipt']); reference.check(row['output'])
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'reference_bridge.json', bridge)
    if not args.draft:
        reference.validate_manifest(out / 'reference_bridge.json', worker.CONTRACT, verify_payloads=True)
    current_bindings = []
    for name in RUNNER_FILES:
        item = reference.binding(HERE / name); item['path'] = str(runtime_runner / name)
        current_bindings.append(item)
    bridge_binding = reference.binding(out / 'reference_bridge.json')
    bridge_binding['path'] = str(runtime_out / 'reference_bridge.json')
    current_bindings.append(bridge_binding)
    # New isolated registry: old queues and maintenance are not mutated.
    old_registry_path = Path(option(native[1]['maintenance']['command'], 'registry'))
    registry_file = original_path.parent / 'queue_registry.json' if args.draft else old_registry_path
    registry = json.loads(registry_file.read_text())
    if registry.get('schema') != 'completed.output.incremental.registry.v1':
        raise ValueError('Expected the current completed-output maintenance contract')
    registry['index'] = str(runtime_out / 'completed_output_index.json')
    registry['queue_states'] = [str(runtime_out / 'qualification_queue/state.json'),
                                str(runtime_out / 'formal_queue/state.json')]
    registry['scope'] = 'Independent coverage2x2-v2 completed-output index; inherited parent registries remain read-only evidence.'
    save(out / 'queue_registry.json', registry)
    registry_binding = reference.binding(out / 'queue_registry.json')
    registry_binding['path'] = str(runtime_out / 'queue_registry.json')
    rapids_python = external['cugraph_composite']['command'][0]
    common_environment = copy.deepcopy(original['environment'])

    def make(config, phase, dataset, repeat=0, tiny=None, target=None):
        template = native[config['gpu_count']] if config['system'] == 'eggpu' else external[config['system']]
        job = copy.deepcopy(template)
        command = list(job['command'])
        command[2] = str(runtime_runner / 'e6_worker.py')
        if config['system'] == 'eggpu':
            command[0] = rapids_python
        ident = f"{phase}_{dataset}_{config_key(config)}_r{repeat:02}"
        output = runtime_out / 'runs' / ident
        changes = dict(system=config['system'], ablate=config['ablate'],
                       coverage_backend=config['coverage_backend'], phase=phase, dataset=dataset,
                       repeat=repeat, output=output,
                       input_root=campaign / ('e6_tiny_inputs' if phase == 'tiny' else 'inputs'),
                       reference_manifest=runtime_out / 'reference_bridge.json', timeout=3600)
        for key, value in changes.items():
            set_option(command, key.replace('_', '-'), value)
        for name in ('tiny-proof', 'target-proof'):
            while '--' + name in command:
                i = command.index('--' + name); del command[i:i + 2]
        dependencies = []
        for name, proof in (('tiny-proof', tiny), ('target-proof', target)):
            if proof:
                set_option(command, name, proof['receipt']); dependencies.append(proof['id'])
        # Target metadata is copied from the exact same original dataset/config.
        target_template = next((j for j in original['jobs'] if j['dataset'] == dataset and
                                j['system'] == template['system'] and j['ablate'] == template['ablate'] and
                                j['gpu_count'] == template['gpu_count']), template)
        job.update(config, id=ident, command=command, command_sha256=digest(command),
                   receipt=str(output / 'result.json'), phase=phase, dataset=dataset, repeat=repeat,
                   depends_on=dependencies, expected_statuses=['PASS' if phase == 'formal' else 'PASS_QUALIFICATION'],
                   timeout=86400, disk_guard=copy.deepcopy(target_template['disk_guard']))
        job['bindings'] = copy.deepcopy(template['bindings']) + current_bindings
        if not args.draft:
            job['bindings'].append(reference.binding(command[0]))
        maintenance = job.get('maintenance')
        if maintenance:
            mcommand = maintenance['command']
            set_option(mcommand, 'registry', runtime_out / 'queue_registry.json')
            set_option(mcommand, 'output', runtime_out / 'maintenance' / ident)
            if '--job-result' in mcommand:
                set_option(mcommand, 'job-result', output / 'result.json')
            maintenance['command_sha256'] = digest(mcommand)
            maintenance['receipt'] = str(runtime_out / 'maintenance' / ident / 'receipt.json')
            maintenance['bindings'] = [b for b in maintenance['bindings'] if not b['path'].endswith('queue_registry.json')] + [registry_binding]
        return job

    configs = configurations()
    tiny = [make(c, 'tiny', worker.TINY) for c in configs]
    qualification = tiny[:]; formal = []
    for dataset in worker.DATASETS:
        targets = [make(c, 'target', dataset, tiny=t) for c, t in zip(configs, tiny)]
        qualification += targets
        for repeat in range(10):
            order = list(zip(configs, tiny, targets))
            if repeat % 2:
                order.reverse()
            formal += [make(c, 'formal', dataset, repeat, t, target) for c, t, target in order]
    validate_matrix(qualification); validate_matrix(formal, formal=True)
    common = dict(schema='eggpu.e6.coverage2x2.qualification-plan/v1', protocol=worker.PROTOCOL,
                  contract=worker.CONTRACT, environment=common_environment, original_template=reference.binding(original_path),
                  gpu_executed=False, formal_measurements=0, runner_bindings=current_bindings,
                  shared_control='eggpu_none_none_p1 measured once per repeat and referenced by both 2x2 panels',
                  environment_policy='All EGGPU arms use the same existing RAPIDS interpreter; current tiny/target required.',
                  numerical_policy='Preserve stricter current atol=1e-10 rtol=1e-8; this is explicitly stricter than original selection freeze.',
                  storage_poll_seconds=15)
    key = 'job_specs' if args.draft else 'jobs'
    save(out / 'qualification_plan.json', dict(common, status='DRAFT_REQUIRES_DEPLOYMENT_PREFLIGHT' if args.draft else 'CPU_PLAN_ONLY_NOT_EXECUTED', **{key: qualification}))
    # Formal remains a non-executable ledger even after qualification-plan materialization.
    save(out / 'formal_pending.json', dict(common, status='NOT_ADMITTED_REQUIRES_CURRENT_PROOFS_AND_SEPARATE_REVIEW', job_specs=formal))
    receipt = dict(status='PASS_CPU_DRAFT' if args.draft else 'PASS_CPU_QUALIFICATION_PLAN',
                   gpu_executed=False, qualification_jobs=len(qualification), formal_pending_cells=len(formal),
                   configurations_per_graph=len(configs), admitted=False,
                   plans={n: reference.binding(out / n) for n in ('qualification_plan.json', 'formal_pending.json', 'reference_bridge.json')})
    save(out / 'receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draft', action='store_true')
    print(json.dumps(build(p.parse_args())))
