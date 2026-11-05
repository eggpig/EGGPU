#!/usr/bin/env python3
"""CPU-only source-bound E3/E4/E5 plan builder and admission guard.

No discovery of newest files, no performance-based selection, no GPU imports.
Only `guard --execute-gpu` replaces this process with the explicitly frozen
worker; the root campaign controller is the sole owner of that operation.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys

RHO = (0., 1e-6, 1e-4, 1e-2, .1, 1.)
BETA = (0., .25, .5, .75, 1.)
ORDERS = ('grouped', 'interleaved', 'permuted')
SOURCE_SUFFIXES = {'.py', '.cpp', '.cc', '.c', '.cu', '.cuh', '.h', '.hpp', '.cmake'}
SIGNER = 'easygraph/gpu/_manifest_signatures.py'
BENCHMARK = ('state_worker.py', 'state_contract.py', 'state_protocol.py',
             'state_reference.py', 'state_inputs.py', 'baselines/hytgraph_qualify.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def checked(item):
    path = Path(item['path'])
    require(path.is_file(), 'Missing bound file: ' + str(path))
    require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
            'Bound file changed: ' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


RUNTIME_PROBE = '''import json,sys,hashlib
from pathlib import Path
import numpy,scipy
def file(p):
 p=Path(p);return {'path':str(p.absolute()),'resolved_path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps({'invocation':str(Path(sys.executable).absolute()),'prefix':sys.prefix,'base_prefix':sys.base_prefix,'python_version':sys.version,'numpy_version':numpy.__version__,'scipy_version':scipy.__version__,'numpy':file(numpy.__file__),'scipy':file(scipy.__file__)}))
'''


def runtime_probe(invocation, environment):
    """Preserve argv[0]/venv semantics; ELF identity alone is insufficient."""
    invocation = str(Path(invocation).absolute())  # Intentionally NOT resolve().
    env = dict(os.environ, **environment, CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([invocation, '-B', '-c', RUNTIME_PROBE], env=env,
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
    runtime = json.loads(result.stdout)
    require(runtime['invocation'] == invocation, 'Python invocation identity differs')
    return runtime


def verify_runtime(dep):
    require('python_invocation' in dep and 'python_runtime' in dep, 'Legacy resolved-interpreter deployment is inadmissible')
    require(str(Path(sys.executable).absolute()) == dep['python_invocation'], 'Wrong lexical Python/venv invocation')
    require(sys.prefix == dep['python_runtime']['prefix'] and
            sys.base_prefix == dep['python_runtime']['base_prefix'], 'Wrong Python virtual environment prefix')
    actual = runtime_probe(dep['python_invocation'], dep['environment'])
    require(actual == dep['python_runtime'], 'Actual selected interpreter/dependency environment changed')
    return actual


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def option(command, name, default=None):
    flag = '--' + name
    return command[command.index(flag) + 1] if flag in command else default


def source_files(root, freeze):
    require(digest(freeze['source_files']) == freeze['source_tree_sha256'], 'Invalid source freeze digest')
    for relative, expected in freeze['source_files'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Unsafe source path')
        checked(dict(expected, path=str(Path(root) / relative)))
    # Extra executable source is also an epoch change, even if omitted from freeze.
    actual = set()
    for directory in ('easygraph', 'cpp_easygraph', 'gpu_easygraph'):
        for path in (Path(root) / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and (
                    path.suffix in SOURCE_SUFFIXES or path.name == 'CMakeLists.txt'):
                actual.add(str(path.relative_to(root)))
    require(actual <= set(freeze['source_files']), 'Unbound executable source added')
    require(SIGNER in freeze['source_files'], 'Current manifest signer missing')


def grid():
    """Frozen main grid; unresolved controls are reported separately, not invented."""
    result = []
    for p, names in ((1, ('US', 'IN', 'OR')), (8, ('IN', 'OR'))):
        for name in names:
            for rho in RHO:
                for locality in (('uniform',) if rho in (0., 1.) else ('uniform', 'clustered')):
                    for old in ('none', 'per_update'):
                        for arm in ('D0C1', 'D1C1'):
                            result.append(dict(experiment='E3', datasets=[name], partitions=p,
                                rho=rho, locality=locality, old_readers=old, arm=arm,
                                beta=.25, order='interleaved', role='density_reader_control'))
    for p in (1, 8):
        for beta in BETA:
            for order in ORDERS:
                for arm in ('D1C0', 'D1C1'):
                    result.append(dict(experiment='E4', datasets=['US', 'OR', 'IN', 'UK'],
                        partitions=p, rho=0., locality='uniform', old_readers='none',
                        arm=arm, beta=beta, order=order, role='budget_order_control'))
        for arm in ('D1C0', 'D1C1'):
            result.append(dict(experiment='E4', datasets=['US', 'OR', 'IN', 'UK'], partitions=p,
                rho=0., locality='uniform', old_readers='none', arm=arm,
                beta=.25, order='once', role='no_revisit_control'))
        for rho, beta, role in ((1e-4, .25, 'main'), (0., .25, 'no_update_control'),
                                (1e-4, 1., 'full_residency_control')):
            for arm in ('D0C0', 'D1C0', 'D0C1', 'D1C1'):
                result.append(dict(experiment='E5', datasets=['IN', 'UK', 'OR'], partitions=p,
                    rho=rho, locality='uniform', old_readers='none', arm=arm,
                    beta=beta, order='interleaved', role=role))
    for row in result:
        row['id'] = (f"{row['experiment']}_{'-'.join(row['datasets'])}_p{row['partitions']}"
            f"_rho{row['rho']:g}_{row['locality']}_{row['arm']}_{row['old_readers']}"
            f"_b{row['beta']:g}_{row['order']}")
    require(len({row['id'] for row in result}) == len(result), 'Duplicate grid identity')
    return result


def hotspot_grid():
    """Predeclared E3 intervention; fixed density, explicit touched owner count."""
    rows=[]
    for name in ('IN','OR'):
        for k in (1,2,4,8):
            for old in ('none','per_update'):
                for arm in ('D0C1','D1C1'):
                    row=dict(experiment='E3',datasets=[name],partitions=8,rho=1e-4,
                        locality='partition_hotspot_k'+str(k),old_readers=old,arm=arm,
                        beta=.25,order='interleaved',role='fixed_density_partition_coverage',hotspot_k=k)
                    row['id']=f"E3_{name}_p8_rho0.0001_{row['locality']}_{arm}_{old}_b0.25_interleaved"
                    rows.append(row)
    return rows


def validate_hotspot(record):
    cell=record['cell']
    require(cell['experiment']=='E3' and cell['partitions']==8 and cell['rho']==1e-4 and
            len(cell['datasets'])==1 and cell['datasets'][0] in ('IN','OR'), 'Hotspot input domain differs')
    k=cell['hotspot_k']
    require(k in (1,2,4,8) and cell['locality']=='partition_hotspot_k'+str(k),'Hotspot owner intervention differs')
    prep=record['hotspot_preparation'];checked(prep['controller']);checked(prep['selector'])
    row=record['datasets'][cell['datasets'][0]];proof=read(checked(row['hotspot_selection']))
    require(proof['status']=='PASS_CPU_SELECTION' and proof['k']==k and proof['rho_requested']==1e-4 and
            proof['selected_owners']==list(range(k)) and proof['affected_owners']==list(range(k)), 'Hotspot owner coverage differs')
    require(proof['positions']==row['positions'] and proof['base']==row['shared_base'] and
            proof['canonical']==row['canonical'] and proof['selector']==prep['selector'], 'Hotspot payload provenance differs')
    require(proof['seed']==20261104 and proof['changed_positions']==row['changed_positions'] and
            proof['actual_rho']==row['actual_rho'], 'Hotspot count/seed differs')
    # The frozen selector tests reverse arcs; admission additionally verifies
    # actual stored positions, exact count, strict order, and owner domain.
    import numpy as np
    positions=np.load(checked(row['positions']),allow_pickle=False,mmap_mode='r')
    require(positions.dtype==np.dtype('<i8') and positions.ndim==1 and len(positions)==proof['changed_positions'], 'Hotspot positions dtype/count differs')
    require(not len(positions) or (positions[0]>=0 and positions[-1]<row['m'] and np.all(positions[1:]>positions[:-1])), 'Hotspot positions are not unique/in range')
    actual=np.unique(np.searchsorted(row['edge_cuts'],positions,side='right')-1).tolist()
    require(actual==list(range(k)), 'Actual hotspot owner coverage differs')


def input_key(cell):
    return (cell['experiment'], tuple(cell['datasets']), cell['partitions'],
            cell['rho'], cell['locality'])


def validate_input_metadata(record, freeze, registry, *, tiny=False):
    require(record.get('schema') == 'state.inputs.v2' and
            record.get('status') == 'PASS_CPU_INPUTS_GPU_UNQUALIFIED', 'Invalid state input receipt')
    require(record.get('qualification_only') is tiny, 'Tiny/target input domain mismatch')
    signer = record.get('source_signer')
    if signer is None:
        # Original state_inputs v2 predated the explicit source_signer field.
        # Such legacy tiny fixtures need explicit signer provenance in deployment.
        require(tiny, 'Real input lacks manifest signer provenance')
    else:
        require({k: signer[k] for k in ('sha256', 'bytes')} == freeze['source_files'][SIGNER],
                'Input signer differs from final source; regenerate inputs')
    if not tiny and record.get('cell',{}).get('locality','').startswith('partition_hotspot_'):
        validate_hotspot(record)
    if tiny:
        require(set(record['datasets']) == {'TINY_A', 'TINY_B', 'TINY_C'}, 'Wrong tiny graph pool')
        return
    require(registry['status'] == 'ALL_SEVEN_CANONICAL_READY', 'Seven-input registry incomplete')
    require(len(record['cell']['datasets']) == len(record['datasets']) and
            set(record['cell']['datasets']) == set(record['datasets']), 'Input pool identity differs')
    require(record['partitions'] == record['cell']['partitions'], 'Input partition identity differs')
    prep = record['shared_grid_preparation']
    require(prep['source_signer'] == signer, 'Conflicting signer provenance')
    for name, row in record['datasets'].items():
        canonical = registry['records'][name]['receipt']
        aliases = {'metadata': 'canonical' if 'canonical' in canonical['files'] else 'metadata'}
        expected = {key: canonical['files'][aliases.get(key, key)]
                    for key in ('offsets', 'indices', 'weights', 'metadata', 'sources')}
        require(row['canonical'] == expected, 'Canonical input binding differs from registry')
        require(row['n'] == canonical['vertices'] and row['m'] == canonical['stored_arcs'], 'Input dimensions differ')
        require(row['sources'] == canonical['source_selection']['primary_sources'], 'Input primary sources differ')
        require(row['requested_rho'] == record['cell']['rho'] and
                row['locality'] == record['cell']['locality'], 'Position intervention differs')
        require(len(row['edge_cuts']) == record['partitions'] + 1 and
                row['edge_cuts'][0] == 0 and row['edge_cuts'][-1] == row['m'], 'Bad partition edge cuts')


def initial(row):
    if row['partitions'] != 1 or row['order'] != 'interleaved' or row['beta'] != .25:
        return False
    return row['experiment'] == 'E4' or (row['rho'] == 1e-4 and row['locality'] == 'uniform')


def validate_algorithm_environment(environment):
    """Validate actual C++ PR option prerequisites before any GPU admission.

    This checks deployment compatibility only. It does not select a faster
    strategy or alter the frozen state worker COMMON_ENV / D/C interventions.
    """
    env = environment
    require(env.get('EGMGPU_CORE_OUTPUT_CPU_PLACEMENT', 'inherited') == 'inherited',
            'Four-function state workflow requires inherited output CPU placement; PR lacks explicit placement support')
    streamed = env.get('EGMGPU_PAGERANK_SCHEDULE', 'legacy') == 'streamed'
    if env.get('EGMGPU_PAGERANK_OUTPUT', 'copy') == 'pinned':
        require(streamed, 'Pinned PageRank output requires streamed schedule')
    if env.get('EGMGPU_CORE_OUTPUT_TEAM', 'original') == 'team16':
        require(streamed, 'Core PR output team16 requires streamed execution')
    halo = env.get('EGMGPU_PAGERANK_HALO_EXCHANGE', 'peer_gather')
    explicit_halo = halo in ('nccl_allgather', 'nccl_variable_gather', 'nccl_sparse_halo', 'host_pinned_parallel')
    if explicit_halo:
        require(streamed, 'Explicit bulk halo requires streamed PageRank')
        require(env.get('EGMGPU_PAGERANK_SCALAR_REDUCTION', 'auto') == 'host',
                'Explicit bulk halo requires host scalar reduction')
    if halo.startswith('nccl_'):
        require(env.get('EGGPU_COMM_BACKEND') == 'nccl', 'NCCL halo requires an NCCL deployment')
    if env.get('EGMGPU_PAGERANK_SPARSE_STATE', 'call') == 'retained':
        require(halo == 'nccl_sparse_halo' and env.get('EGMGPU_WORKSPACE_POLICY') == 'per_worker',
                'Retained sparse state requires nccl_sparse_halo and per_worker')
    if halo == 'host_pinned_parallel':
        require(env.get('OMP_WAIT_POLICY') == 'PASSIVE', 'Pinned parallel halo requires passive OpenMP wait')


def deployment(a):
    freeze = read(a.freeze)
    require(freeze['source_tree_sha256'] == a.expected_source_sha256, 'Explicit final source SHA differs')
    source_files(a.source.resolve(), freeze)
    runner_freeze = read(a.runner / 'freeze.json')
    for name, expected in runner_freeze['files'].items():
        checked(dict(expected, path=str(a.runner / name)))
    require(set(BENCHMARK) <= set(runner_freeze['files']), 'Incomplete runner freeze')
    template = read(a.deployment_template)
    resources = {}
    for p in (1, 8):
        matches = [j for j in template['jobs'] if j['id'] == f'pilot_p{p}']
        require(len(matches) == 1, 'Missing exact deployment profile')
        command = matches[0]['command']
        resources[str(p)] = {k: option(command, k) for k in ('gpu-uuids', 'cpu-set', 'lock', 'inputs')}
        resource = resources[str(p)]
        require(len(resource['gpu-uuids'].split(',')) == p, 'GPU profile size differs')
        require(len(set(resource['cpu-set'].split(','))) == 16, 'Expected exact 16 CPU cores')
        resource['tiny_inputs'] = bind(resource.pop('inputs'))
        resource['tiny_signer_provenance'] = 'Manifest signatures revalidated against selected final source at admission'
        if p == 1:
            python = command[0]
            dso = option(command, 'native-dso')
        else:
            require(command[0] == python and option(command, 'native-dso') == dso, 'Runtime differs across profiles')
    env = dict(template['environment'])
    profile_path = getattr(a, 'profile_environment', None)
    profile = read(profile_path) if profile_path else None
    if profile:
        common = imports(a.runner)[0].COMMON_ENV
        for count, resource in resources.items():
            extra = profile['environment_additions_by_count'][count]
            require(not any(key in common and common[key] != value for key, value in extra.items()),
                    'Profile must not override frozen state COMMON_ENV')
            require(not any(key in env and env[key] != value for key, value in extra.items()),
                    'Profile must not override the deployment template environment')
            validate_algorithm_environment(dict(env, **extra))
            resource['environment_additions'] = extra
    env['PYTHONPATH'] = str(Path(dso).parent) + ':' + str(a.source.resolve())
    invocation = str(Path(python).absolute())
    runtime = runtime_probe(invocation, env)
    return dict(schema='state.final.deployment.v4', source_root=str(a.source.resolve()),
        source_freeze=bind(a.freeze), source_tree_sha256=freeze['source_tree_sha256'],
        runner_root=str(a.runner.resolve()), runner_freeze=bind(a.runner / 'freeze.json'),
        registry=bind(a.registry), python=bind(python), python_invocation=invocation,
        python_runtime=runtime, native_dso=bind(dso),
        native_build_receipt=bind(a.native_build_receipt),
        environment=env, resources=resources, template=bind(a.deployment_template),
        profile_environment=bind(profile_path) if profile_path else None,
        planner=bind(__file__), timeout_seconds=3600)


def environment_for(dep, partitions):
    """One predeclared per-resource profile, shared by every graph/arm."""
    profile = dep.get('profile_environment')
    extra = dep['resources'][str(partitions)].get('environment_additions', {})
    if profile:
        record = read(checked(profile))
        require(extra == record['environment_additions_by_count'][str(partitions)],
                'Per-resource environment differs from bound profile')
    environment = dict(dep['environment'], **extra)
    validate_algorithm_environment(environment)
    return environment


def build(a):
    require(not a.output.exists(), 'Use a new immutable plan directory')
    dep = deployment(a)
    freeze, registry = read(a.freeze), read(a.registry)
    catalog = {}
    for path in a.inputs:
        record = read(path)
        validate_input_metadata(record, freeze, registry)
        key = input_key(record['cell'])
        require(key not in catalog, 'Duplicate input identity; select one explicit receipt')
        catalog[key] = (bind(path), record)
    for resource in dep['resources'].values():
        validate_input_metadata(read(resource['tiny_inputs']['path']), freeze, registry, tiny=True)
    rows = hotspot_grid() if getattr(a,'grid_domain','main') == 'partition-hotspot' else grid()
    selected = []
    for row in rows:
        key = input_key(row)
        row['input_status'] = ('MISSING_INPUT' if key not in catalog else
            'P8_DISK_HOLD' if row['partitions'] == 8 and not a.enable_p8_targets else 'READY_CPU_INPUTS_UNQUALIFIED')
        row['selected_for_this_plan'] = (a.scope == 'available' or initial(row)) and row['input_status'] == 'READY_CPU_INPUTS_UNQUALIFIED'
        if key in catalog:
            row['inputs'] = catalog[key][0]
        if row['selected_for_this_plan']:
            selected.append(row)
    require(selected, 'No selected ready target configurations')
    a.output.mkdir(parents=True)
    save_new(a.output / 'deployment.json', dep)
    dep_binding = bind(a.output / 'deployment.json')
    qualify, formal = [], []
    specs = {}

    def add_job(identity, action, phase, experiment, arm, old, beta, order,
                p, inputs, names, *, repeat=0, pilot=None, tiny=None, target=None, depends=()):
        require(identity not in specs, 'Duplicate job id')
        output = a.results_root / identity
        command = [dep['python_invocation'], '-B', str(a.runner / 'state_worker.py'), action,
            '--execute-gpu', '--phase', phase, '--repeat', str(repeat), '--inputs', inputs['path'],
            '--source', dep['source_root'], '--native-dso', dep['native_dso']['path'],
            '--gpu-uuids', dep['resources'][str(p)]['gpu-uuids'],
            '--cpu-set', dep['resources'][str(p)]['cpu-set'],
            '--lock', dep['resources'][str(p)]['lock'],
            '--output-store', str(a.results_root / 'objects'), '--reference-cache', str(a.results_root / 'references'),
            '--output', str(output), '--experiment', experiment, '--arm', arm, '--old-readers', old,
            '--beta', str(beta), '--order', order, '--datasets', ','.join(names)]
        for flag, dependency in (('pilot', pilot), ('tiny-proof', tiny), ('target-proof', target)):
            if dependency:
                command += ['--' + flag, specs[dependency]['receipt']]
        spec = dict(schema='state.final.job.v1', id=identity, deployment=dep_binding,
            command=command, inputs=inputs, datasets=names, partitions=p, action=action, phase=phase,
            experiment=experiment, arm=arm, old_readers=old, beta=beta, order=order, repeat=repeat,
            receipt=str(output / 'receipt.json'), dependencies=list(depends),
            runtime_admission=str(a.results_root / 'admissions' / (identity + '.json')),
            proof_specs={k: bind(a.output / 'jobs' / (v + '.json'))
                         for k, v in (('pilot', pilot), ('tiny', tiny), ('target', target)) if v})
        if phase == 'formal':
            block = dict(experiment=experiment, inputs_sha256=inputs['sha256'],
                         datasets=names, partitions=p, old_readers=old,
                         beta=beta, order=order, repeat=repeat)
            spec['paired_block'] = block
            spec['paired_block_sha256'] = digest(block)
        save_new(a.output / 'jobs' / (identity + '.json'), spec)
        specs[identity] = spec
        item = dict(id=identity, command=[dep['python_invocation'], '-B', str(Path(__file__).resolve()),
            'guard', '--spec', str(a.output / 'jobs' / (identity + '.json')), '--execute-gpu'],
            receipt=spec['receipt'], expected_statuses=[
                'PASS_PHYSICAL_BYTE_PILOT' if action == 'pilot' else
                'PASS_STATE_COMPLETE_WORKFLOW' if phase == 'formal' else 'PASS_STATE_GPU_QUALIFICATION'],
            depends_on=list(depends), timeout=dep['timeout_seconds'],
            environment=environment_for(dep, p),
            bindings=[bind(__file__), bind(a.output / 'jobs' / (identity + '.json')), dep_binding])
        (formal if phase == 'formal' else qualify).append(item)
        return identity

    # Requalify the existing 22 main-point tiny jobs on both resource profiles.
    # Extra sweep tiny proofs are generated only when needed, keyed by exact configuration.
    tiny_ids = {}
    pilot_ids = {}
    for p in (1, 8):
        tiny_input = dep['resources'][str(p)]['tiny_inputs']
        tiny_names = list(read(tiny_input['path'])['datasets'])
        pilot = add_job(f'tiny_pilot_p{p}', 'pilot', 'tiny', 'E5', 'D1C1', 'none', .25,
                        'interleaved', p, tiny_input, tiny_names)
        pilot_ids[('tiny', p)] = pilot
        defaults = [dict(experiment='E3', arm=arm, old_readers=old, beta=.25, order='interleaved', partitions=p)
                    for arm in ('D0C1', 'D1C1') for old in ('none', 'per_update')]
        defaults += [dict(experiment=ex, arm=arm, old_readers='none', beta=.25, order='interleaved', partitions=p)
                     for ex, arms in (('E4', ('D1C0', 'D1C1')), ('E5', ('D0C0','D1C0','D0C1','D1C1')))
                     for arm in arms]
        configs = defaults + [row for row in selected if row['partitions'] == p]
        for row in configs:
            key = tuple(row[k] for k in ('experiment', 'arm', 'old_readers', 'beta', 'order', 'partitions'))
            if key in tiny_ids:
                continue
            ex, arm, old, beta, order, _ = key
            ident = f'tiny_{ex}_{arm}_{old}_b{beta:g}_{order}_p{p}'
            need_pilot = None if ex == 'E3' else pilot
            tiny_ids[key] = add_job(ident, 'run', 'tiny', ex, arm, old, beta, order, p,
                tiny_input, ['TINY_A'] if ex == 'E3' else tiny_names, pilot=need_pilot,
                depends=() if need_pilot is None else (pilot,))
    for row in selected:
        ex, p = row['experiment'], row['partitions']
        pilot = None
        if ex != 'E3':
            key = ('target', row['inputs']['sha256'], ex, p)
            if key not in pilot_ids:
                pilot_ids[key] = add_job(f'target_pilot_{ex}_p{p}_{row["inputs"]["sha256"][:12]}',
                    'pilot', 'target', ex, 'D1C1', 'none', .25, 'interleaved', p,
                    row['inputs'], row['datasets'])
            pilot = pilot_ids[key]
        key = tuple(row[k] for k in ('experiment', 'arm', 'old_readers', 'beta', 'order', 'partitions'))
        tiny = tiny_ids[key]
        target = add_job('target_' + row['id'], 'run', 'target', ex, row['arm'], row['old_readers'],
            row['beta'], row['order'], p, row['inputs'], row['datasets'], pilot=pilot, tiny=tiny,
            depends=tuple(x for x in (pilot, tiny) if x))
        for repeat in range(10):
            add_job('formal_' + row['id'] + f'_r{repeat:02}', 'run', 'formal', ex, row['arm'],
                row['old_readers'], row['beta'], row['order'], p, row['inputs'], row['datasets'],
                repeat=repeat, pilot=pilot, tiny=tiny, target=target)
    save_new(a.output / 'qualification_plan.json', dict(jobs=qualify, environment=dep['environment'],
        schema='state.final.queue.v1', source_tree_sha256=dep['source_tree_sha256'], formal_measurements=0))
    # Complete one matched repetition block before the next. Rotate arm order
    # deterministically across repetitions, without inspecting any timings.
    def formal_order(job):
        spec = specs[job['id']]
        arms = {'E3': ('D0C1','D1C1'), 'E4': ('D1C0','D1C1'),
                'E5': ('D0C0','D1C0','D0C1','D1C1')}[spec['experiment']]
        return (spec['experiment'], spec['partitions'], spec['inputs']['sha256'],
                tuple(spec['datasets']), spec['old_readers'], spec['beta'], spec['order'],
                spec['repeat'], (arms.index(spec['arm']) - spec['repeat']) % len(arms))
    formal.sort(key=formal_order)
    save_new(a.output / 'formal_pending.json', dict(status='NOT_ADMITTED_REQUIRES_EXACT_QUALIFICATIONS',
        jobs=formal, environment=dep['environment'], schema='state.final.formal.pending.v1',
        formal_measurements=0, note='Not a queue authorization. Use admit-formal after all selected proofs pass.'))
    save_new(a.output / 'coverage.json', dict(schema='state.final.coverage.v1', status='CPU_PLAN_ONLY',
        source_tree_sha256=dep['source_tree_sha256'], configurations=rows,
        selected_configurations=len(selected), planned_formal_repetitions=len(formal),
        qualification_jobs=len(qualify), gpu_executed=False, formal_measurements=0,
        pending_protocol_controls=[
            dict(control='unweighted_PR_weight_invariance', status='PENDING_SEPARATE_RUNNER',
                 reason='Frozen state_v4 has BFS/SSSP/WCC/weighted_PR only; adding PR changes query counts.'),
            dict(control='partition_concentration', status='SEPARATE_SOURCE_BOUND_HOTSPOT_PLAN',
                 reason='32 predeclared P8 configurations at fixed density use exact K=1/2/4/8 touched owners; same D and old-reader arms.'),
            dict(control='cold_full_graph_rebuild', status='NOT_IMPLEMENTED_IN_STATE_V4',
                 reason='D0 persistent_full reuses unchanged topology and fully uploads new uncached weight identities.')]))
    return dict(status='CPU_PLAN_ONLY', output=str(a.output), configurations=len(selected),
                qualification_jobs=len(qualify), pending_formal_jobs=len(formal), gpu_executed=False)


def imports(runner):
    sys.path.insert(0, str(runner))
    modules = tuple(importlib.import_module(name) for name in ('state_worker', 'state_contract', 'state_protocol'))
    for module in modules:
        require(Path(module.__file__).resolve().parent == Path(runner).resolve(), 'Wrong frozen state runner imported')
    return modules


def controlled_environment(environment, common, uuids):
    expected = dict(environment, **common, CUDA_VISIBLE_DEVICES=uuids, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    relevant = lambda key: key.startswith(('EGGPU_', 'EGMGPU_', 'NCCL_', 'UCX_', 'CUDA_', 'CUPY_')) or key in common or key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH')
    actual = {k: v for k, v in os.environ.items() if relevant(k)}
    for key, value in expected.items():
        if relevant(key):
            actual[key] = value
    require(set(actual) == {k for k in expected if relevant(k)}, 'Unbound controlled environment variable inherited')
    return actual


def check_pilot(proof, config, inputs, names):
    require(proof.get('schema') == 'state.byte.pilot.v4' and
            proof.get('status') == 'PASS_PHYSICAL_BYTE_PILOT' and proof.get('gpu_executed') is True,
            'Missing successful actual physical-byte pilot')
    require(proof.get('selection_uses_timing') is False, 'Budget must not be selected using timing')
    for key in ('source_binding', 'devices', 'environment', 'gpu_uuids', 'cpu_affinity'):
        require(proof.get(key) == config[key], 'Pilot final source/resource mismatch: ' + key)
    require(proof['inputs'] == inputs and set(proof['resident_view_bytes']) == set(names), 'Pilot input/pool differs')
    require(proof.get('phase') == ('tiny' if all(x.startswith('TINY_') for x in names) else 'target'), 'Pilot phase differs')
    for row in proof['resident_view_bytes'].values():
        require(set(row) == {str(x) for x in config['devices']} and
                all(type(x) is int and x > 0 for x in row.values()), 'Invalid actual per-device view bytes')


def check_prior_admission(spec_binding, dep):
    selected = read(checked(spec_binding))
    prior = read(selected['runtime_admission'])
    require(prior['status'] == 'PASS_CPU_ADMISSION' and prior['spec'] == spec_binding,
            'Qualification lacks exact launcher admission')
    require(prior['python_runtime'] == dep['python_runtime'] and
            prior['source_tree_sha256'] == dep['source_tree_sha256'],
            'Qualification launcher/venv/source identity differs')
    return selected, bind(selected['runtime_admission'])


def check_spec(spec_path):
    spec = read(spec_path)
    require(spec.get('schema') == 'state.final.job.v1', 'Wrong job schema')
    dep = read(checked(spec['deployment']))
    checked(dep['planner'])
    require(Path(dep['planner']['path']).resolve() == Path(__file__).resolve(), 'Wrong admission guard imported')
    freeze = read(checked(dep['source_freeze']))
    require(freeze['source_tree_sha256'] == dep['source_tree_sha256'], 'Final source selection changed')
    source_files(Path(dep['source_root']), freeze)
    runner_freeze = read(checked(dep['runner_freeze']))
    for name, expected in runner_freeze['files'].items():
        checked(dict(expected, path=str(Path(dep['runner_root']) / name)))
    for key in ('python', 'native_dso', 'native_build_receipt'):
        checked(dep[key])
    require(bind(sys.executable)['sha256'] == dep['python']['sha256'], 'Admission must use selected worker Python binary')
    runtime = verify_runtime(dep)
    registry = read(checked(dep['registry']))
    inputs = read(checked(spec['inputs']))
    validate_input_metadata(inputs, freeze, registry, tiny=spec['phase'] == 'tiny')
    worker, contract, protocol = imports(dep['runner_root'])
    worker.load_inputs(Path(spec['inputs']['path']))  # Complete canonical + projection payload hashes.
    if not inputs['qualification_only']:
        for name, row in inputs['datasets'].items():
            checked(row['shared_base'])
            entry = registry['records'][name]
            require(sha(entry['receipt_path']) == entry['receipt_sha256'], 'Canonical receipt changed')
            require(read(entry['receipt_path']) == entry['receipt'], 'Canonical receipt content differs')
            for item in entry['receipt']['files'].values():
                checked(item)
        for key in ('controller', 'state_inputs', 'state_protocol', 'source_signer', 'source_freeze'):
            checked(inputs['shared_grid_preparation'][key])
    # Manifest loader is CPU-only. It verifies the actual final source's signature
    # algorithm for old tiny receipts that predate explicit signer provenance.
    signer_path = Path(dep['source_root']) / SIGNER
    signer_spec = importlib.util.spec_from_file_location('_state_selected_signer', signer_path)
    signer = importlib.util.module_from_spec(signer_spec)
    signer_spec.loader.exec_module(signer)
    for row in inputs['datasets'].values():
        for item in [row['manifest'], *(x['manifest'] for x in row['projections'].values())]:
            manifest = read(checked(item))
            topology, weights, view = signer.sign(manifest)
            require(topology == manifest['graph']['topology_sha256'] and
                    weights == manifest['graph'].get('weight_sha256') and
                    view == manifest['view']['signature'], 'Manifest signature fails selected signer')
    args = worker.parser().parse_args(spec['command'][3:])
    require(spec['command'][:3] == [dep['python_invocation'], '-B', str(Path(dep['runner_root']) / 'state_worker.py')],
            'Command must execute the selected public frozen worker')
    require(inputs['partitions'] == spec['partitions'], 'Prepared partitions differ from resource profile')
    require(str(args.inputs) == spec['inputs']['path'] and args.datasets.split(',') == spec['datasets'], 'Job input differs')
    for key in ('phase', 'experiment', 'arm', 'old_readers', 'beta', 'order', 'repeat', 'action'):
        require(getattr(args, key) == spec[key], 'Command/spec differs: ' + key)
    require(str(args.source) == dep['source_root'] and str(args.native_dso) == dep['native_dso']['path'], 'Command source/DSO differs')
    resource = dep['resources'][str(spec['partitions'])]
    require(args.gpu_uuids == resource['gpu-uuids'] and args.cpu_set == resource['cpu-set'] and
            str(args.lock) == resource['lock'], 'Command resources differ')
    contract.validate_arguments(args, inputs, spec['datasets'])
    selected_environment = environment_for(dep, spec['partitions'])
    environment = controlled_environment(selected_environment, worker.COMMON_ENV, args.gpu_uuids)
    affinity = sorted(int(x) for x in args.cpu_set.split(','))
    require(set(affinity) <= set(os.sched_getaffinity(0)), 'Requested CPU affinity unavailable')
    binding = contract.source_binding(Path(dep['source_root']), SimpleNamespace(__file__=dep['native_dso']['path']))
    config = contract.configuration(args, binding, list(range(spec['partitions'])), environment, affinity)
    budget = None
    evidence = []
    if spec['action'] == 'run' and spec['experiment'] != 'E3':
        pilot_spec, launcher_evidence = check_prior_admission(spec['proof_specs']['pilot'], dep)
        evidence.append(launcher_evidence)
        require(str(args.pilot) == pilot_spec['receipt'], 'Pilot path differs from selected plan')
        proof = read(args.pilot)
        check_pilot(proof, config, spec['inputs'], spec['datasets'])
        budget = protocol.budgets(proof, spec['beta'])
        budget['pilot'] = bind(args.pilot)
        evidence.append(budget['pilot'])
    for phase in ('tiny', 'target'):
        if phase == 'tiny' and spec['phase'] in ('target', 'formal') and spec['action'] == 'run' or phase == 'target' and spec['phase'] == 'formal':
            selected, launcher_evidence = check_prior_admission(spec['proof_specs'][phase], dep)
            evidence.append(launcher_evidence)
            path = getattr(args, phase + '_proof')
            require(str(path) == selected['receipt'], 'Proof path differs from selected plan')
            proof = read(path)
            require(proof['inputs'] == selected['inputs'], 'Proof input differs from selected job')
            # The frozen contract validates every event and complete durable output.
            evidence.append(contract.validate_proof(path, phase, config, spec['inputs'], spec['datasets'], budget))
    require(not args.output.exists(), 'Output attempt already exists; never overwrite a run')
    return dict(status='PASS_CPU_ADMISSION', gpu_executed=False, formal_measurements=0,
                spec=bind(spec_path), source_tree_sha256=dep['source_tree_sha256'],
                configuration_sha256=contract.digest(config), proof_bindings=evidence,
                physical_budget=budget, command=spec['command'], environment=selected_environment,
                python_runtime=runtime)


def guard(a):
    result = check_spec(a.spec)
    if a.receipt:
        save_new(a.receipt, result)
    if not a.execute_gpu:
        return result
    # Invocation is exclusively submitted by the parent/root campaign owner.
    # The worker still acquires the global GPU lock and checks host/disk/GPU state.
    env = dict(os.environ, **result['environment'])
    # Frozen state_v4 is unchanged. This immutable launcher sidecar records the
    # exact venv admission and is mandatory for every future pilot/tiny/target proof.
    save_new(read(a.spec)['runtime_admission'], result)
    os.execve(result['command'][0], result['command'], env)


def admit_formal(a):
    pending = read(a.bundle / 'formal_pending.json')
    require(pending['status'] == 'NOT_ADMITTED_REQUIRES_EXACT_QUALIFICATIONS', 'Wrong formal request schema')
    evidence, configs, per_config_proofs = [], {}, {}
    for job in pending['jobs']:
        spec_path = Path(option(job['command'], 'spec'))
        spec = read(spec_path)
        key = tuple(spec[k] if not isinstance(spec[k], list) else tuple(spec[k])
                    for k in ('experiment', 'datasets', 'arm', 'old_readers', 'beta', 'order', 'partitions')) + (spec['inputs']['sha256'],)
        configs.setdefault(key, []).append(spec['repeat'])
        if len(configs[key]) == 1:
            admitted = check_spec(spec_path)
            evidence.extend(admitted['proof_bindings'])
            per_config_proofs[key] = admitted['proof_bindings']
        require(not Path(spec['receipt']).parent.exists(), 'Formal attempt already exists')
    require(configs and all(sorted(repeats) == list(range(10)) for repeats in configs.values()), 'Every selected cell requires ten independent repetitions')
    proofs = {x['path']: x for x in evidence}
    jobs = []
    for row in pending['jobs']:
        spec = read(option(row['command'], 'spec'))
        key = tuple(spec[k] if not isinstance(spec[k], list) else tuple(spec[k])
                    for k in ('experiment', 'datasets', 'arm', 'old_readers', 'beta', 'order', 'partitions')) + (spec['inputs']['sha256'],)
        jobs.append(dict(row, depends_on=[], bindings=row['bindings'] + per_config_proofs[key]))
    save_new(a.output, dict(schema='state.final.formal.admitted.v1', jobs=jobs,
        environment=pending['environment'], admission='CPU_PROOFS_VALIDATED_NOT_MEASURED',
        pending=bind(a.bundle / 'formal_pending.json'), proof_bindings=list(proofs.values()),
        configurations=len(configs), repetitions_per_configuration=10, formal_measurements=0))
    return dict(status='PASS_FORMAL_PLAN_ADMISSION', output=str(a.output),
                configurations=len(configs), jobs=len(jobs), gpu_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build')
    for name in ('source', 'freeze', 'runner', 'registry', 'deployment-template', 'native-build-receipt', 'output', 'results-root'):
        b.add_argument('--' + name, type=Path, required=True)
    b.add_argument('--profile-environment', type=Path)
    b.add_argument('--expected-source-sha256', required=True)
    b.add_argument('--inputs', type=Path, nargs='+', required=True)
    b.add_argument('--scope', choices=('initial', 'available'), default='initial')
    b.add_argument('--grid-domain', choices=('main','partition-hotspot'), default='main')
    b.add_argument('--enable-p8-targets', action='store_true')
    g = sub.add_parser('guard')
    g.add_argument('--spec', type=Path, required=True)
    g.add_argument('--receipt', type=Path)
    g.add_argument('--execute-gpu', action='store_true')
    f = sub.add_parser('admit-formal')
    f.add_argument('--bundle', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'build': build, 'guard': guard, 'admit-formal': admit_formal}[args.action](args)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
