"""CPU-only, complete-matrix incremental admission; never launches any worker.

Original formal jobs stay byte-for-byte equal as JSON objects. A CPU proof gate
using the original target ID satisfies their unchanged controller dependencies.
"""
from pathlib import Path
import argparse
import collections
import copy
import fcntl
import hashlib
import importlib.util
import json
import os

HERE = Path(__file__).resolve().parent
INSPECTOR_SHA = '0d2c0343caec9f6cac08c5ef5aae621e5cf45812e8f33e58deac40923de21e75'
SUPPORTED = {'sg', 'sequence', 'native'}


def require(condition, message):
    if not condition: raise ValueError(message)


def load(path): return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bind(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    after = path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'File changed while hashing')
    return dict(path=str(path), bytes=before.st_size, sha256=h.hexdigest())


def checked(item):
    actual = bind(item['path'])
    require(actual == item, 'Bound file changed: ' + item['path'])
    return Path(item['path'])


def save_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def replace_json(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.partial')
    save_new(temporary, value); temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def inspector():
    path = HERE.parent / 'formal_admission_disposition_v1/inspect_ledger.py'
    require(bind(path)['sha256'] == INSPECTOR_SHA, 'Frozen ledger inspector changed')
    spec = importlib.util.spec_from_file_location('_incremental_frozen_inspector', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, path


def option(command, name): return command[command.index('--' + name) + 1]


def complete_job_outputs(job):
    paths = {str(Path(job['receipt']).resolve()), str(Path(option(job['command'], 'output')).resolve())}
    if job.get('maintenance'):
        paths.add(str(Path(job['maintenance']['receipt']).resolve()))
        command = job['maintenance']['command']
        if '--output' in command: paths.add(str(Path(option(command, 'output')).resolve()))
    return sorted(paths)


def original_identity(manifest, group, formal):
    return dict(source_sha256=manifest['source_sha256'], group=group,
                pending=manifest['groups'][group]['pending'],
                matrix_sha256=digest(formal['jobs']), original_formal_slots=len(formal['jobs']))


def registry_read(folder, owner):
    path = folder / 'registry.json'
    if not folder.exists(): return dict(schema='eggpu.formal.incremental.registry.v1', owner=owner, batches=[], cells={})
    require(path.is_file(), 'Existing namespace without registry needs explicit recovery')
    state = load(path)
    require(state.get('schema') == 'eggpu.formal.incremental.registry.v1' and state.get('owner') == owner,
            'Namespace belongs to another source, matrix or manifest')
    batches = folder / 'batches'
    directories = {p.name for p in batches.iterdir()} if batches.exists() else set()
    require(all(p.is_dir() for p in batches.iterdir()) if batches.exists() else True, 'Unexpected batch artifact')
    require(directories == {x['batch_id'] for x in state['batches']}, 'Orphan or missing batch requires explicit recovery')
    claimed = {}
    for row in state['batches']:
        reservation = load(checked(row['reservation']))
        require(reservation['owner'] == owner and reservation['batch_id'] == row['batch_id'], 'Reservation identity changed')
        for cell, identity in reservation['cells'].items():
            require(cell not in claimed, 'A cell is reserved twice')
            claimed[cell] = dict(batch_id=row['batch_id'], cell_identity_sha256=identity)
        if row['status'] == 'COMMITTED':
            checked(row['plan']); checked(row['receipt']); checked(row['gate'])
        else:
            require(row['status'] == 'RESERVED', 'Unknown reservation state')
    require(claimed == state['cells'], 'Cell reservation index differs from immutable batch history')
    return state


def legacy_guard_paths(spec, request):
    require(all(spec.get(k) for k in ('admitted_output', 'queue', 'controller_log')),
            'Explicit original admitted directory, queue and log are required')
    paths = [Path(spec['admitted_output']).resolve(), Path(spec['queue']).resolve()]
    paths += [Path(p).resolve() for p in request.get('additional_legacy_directories', [])]
    require(len(paths) == len(set(paths)), 'Duplicate legacy namespaces')
    return paths


def verify_fences(paths, log, fence):
    require(not Path(log).exists(), 'Original controller log exists; review prior admission/attempt before migration')
    for path in paths:
        if not path.exists(): continue
        permitted = {'incremental_fence.json'}
        if str(path / 'state.json') == fence['completed_projection']: permitted.add('state.json')
        require(path.is_dir() and {p.name for p in path.iterdir()} <= permitted and (path / 'incremental_fence.json').is_file(),
                'Existing legacy plan, queue, state or output forbids migration: ' + str(path))
        require(load(path / 'incremental_fence.json') == fence, 'Legacy namespace is fenced by another owner')
        if (path / 'state.json').exists():
            projected = load(path / 'state.json')
            require(projected.get('schema') == 'eggpu.incremental.completed.projection.v1' and
                    projected.get('owner') == fence['owner'] and projected.get('projection_only') is True and
                    projected.get('physical_worker_runs') == 0 and projected.get('performance_eligible') is False,
                    'Legacy state is not our completed-only projection')


def fence_legacy(paths, fence):
    for path in paths:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=False)
            save_new(path / 'incremental_fence.json', fence)


def prior_claims(paths, formal_jobs):
    known = {str(Path(j['receipt']).resolve()): j['id'] for j in formal_jobs}
    result = collections.defaultdict(list); observations = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            observations.append(dict(path=str(path), observed='ABSENT')); continue
        state = load(path); require(isinstance(state.get('jobs'), list), 'Prior queue lacks explicit job inventory')
        bound = bind(path); observations.append(dict(binding=bound, status=state.get('status')))
        for job in state['jobs']:
            receipt = str(Path(job['receipt']).resolve())
            if receipt in known:
                result[known[receipt]].append(dict(kind='PRIOR_QUEUE_CLAIM', state=bound, prior_job_id=job['id']))
    return result, observations


def runtime_check(request, manifest):
    for key in ('controller', 'supervisor', 'python'): checked(request[key])
    require(Path(request['python_invocation']).resolve() == Path(request['python']['path']),
            'Invocation interpreter differs from its exact executable binding')
    require(Path(request['controller']['path']).name == 'run_campaign_queue_v8.py',
            'Initial incremental route requires the existing serial v8 controller')
    require(Path(request['supervisor']['path']) == Path(request['controller']['path']).with_name('owned_process_supervisor_v2.py'),
            'Controller sibling supervisor differs')
    require(request.get('controller_lock') == manifest.get('controller_lock'), 'Use the original global controller lock')


def run(request_path, observation, *, admit=False):
    request_path, observation = Path(request_path).resolve(), Path(observation).resolve()
    request_binding = bind(request_path); request = load(request_path)
    require(request.get('schema') == 'eggpu.formal.incremental.request.v1', 'Unknown incremental request')
    manifest = load(checked(request['parent_manifest'])); group = request['group']
    require(group in {'sg','sequence','native','e4s'}, 'Unknown group')
    lock = Path(request['controller_lock']); lock.parent.mkdir(parents=True, exist_ok=True)
    observation.mkdir(parents=True, exist_ok=False)
    with lock.open('a+') as owner_lock:
        fcntl.flock(owner_lock, fcntl.LOCK_EX)
        checked(request_binding); checked(request['parent_manifest']); runtime_check(request, manifest)
        module, inspector_path = inspector()
        inspect_request = dict(schema='eggpu.formal.disposition.inspect.v1', group=group,
            manifest=request['parent_manifest'], dispositions=request.get('dispositions', []))
        internal = observation / 'inspect_request.json'; save_new(internal, inspect_request)
        ledger = module.inspect(internal)
        spec = manifest['groups'][group]; formal = load(checked(spec['pending']))
        identity = original_identity(manifest, group, formal)
        root = Path(request['namespace_root']).resolve(); folder = root / group
        require(not folder.is_relative_to(Path(manifest['source_root']).resolve()), 'Never write into the frozen source')
        owner = dict(identity, namespace=str(folder))
        state = registry_read(folder, owner)
        paths = legacy_guard_paths(spec, request)
        require(all(not folder.is_relative_to(p) and not p.is_relative_to(folder) for p in paths),
                'Incremental namespace overlaps legacy namespace')
        fence = dict(schema='eggpu.formal.incremental.legacy-fence.v1', owner=owner,
                     completed_projection=str(Path(spec['queue']).resolve() / 'state.json'),
                     meaning='No old whole-group plan/controller may start in this namespace; formal cells are reserved in the stated incremental registry.')
        namespace_errors = []
        try: verify_fences(paths, spec['controller_log'], fence)
        except ValueError as error: namespace_errors.append(str(error))
        claims, observations = prior_claims(request.get('additional_prior_states', []), formal['jobs'])
        selected = []
        limit = request.get('max_new_cells')
        require(limit is None or type(limit) is int and limit > 0, 'max_new_cells must be a positive integer')
        for cell in ledger['cells']:
            jobs = [s['original_job'] for s in cell['formal_slots']]
            seen = []
            for job in jobs:
                seen += claims[job['id']]
                seen += [dict(kind='EXISTING_OUTPUT', path=p) for p in complete_job_outputs(job) if Path(p).exists()]
            reserved = state['cells'].get(cell['cell_id'])
            if reserved:
                require(reserved['cell_identity_sha256'] == cell['cell_identity_sha256'], 'Reserved cell identity changed')
            schedule = ('RESERVED' if reserved else 'EXISTING_ATTEMPT_OR_CLAIM' if seen else
                        'ELIGIBLE_UNRESERVED' if cell['status'] == 'QUALIFIED' else 'NOT_QUALIFIED')
            cell['incremental_scheduling'] = dict(status=schedule, reservation=reserved, prior_evidence=seen)
            if schedule == 'ELIGIBLE_UNRESERVED' and (limit is None or len(selected) < limit): selected.append(cell)
        blockers = list(ledger['global_blockers']) + [dict(stage='legacy_namespace', reason=e) for e in namespace_errors]
        if group not in SUPPORTED:
            blockers.append(dict(stage='e4s_incremental_controller', reason='HOLD: dedicated recoverable-output controller rejects CPU gate prelude; no ordinary-controller fallback.'))
        # Shared source/CPU/numeric gates block all new admissions. Unresolved
        # individual target cells remain in the denominator without blocking peers.
        if blockers: selected = []
        save_new(observation / 'complete_ledger.json', ledger)
        report = dict(schema='eggpu.formal.incremental.observation.v1',
            status='READY_NOT_ADMITTED' if selected else 'NO_NEW_ADMISSION',
            request=request_binding, group=group, mode='admit' if admit else 'inspect',
            workers_launched=0, gpu_executed=False, formal_measurements_created=0,
            original_cell_count=ledger['original_cell_count'], original_formal_slots=ledger['original_formal_slots'],
            selected_cell_ids=[c['cell_id'] for c in selected], selected_formal_slots=10*len(selected),
            scheduling_policy='Original fixed-matrix order, never observed performance; complete ten-repeat cells only.',
            blockers=blockers, complete_ledger=ledger, prior_queue_observations=observations,
            parent_manifest=request['parent_manifest'], inspector=bind(inspector_path), helper=bind(__file__))
        if admit and selected:
            report['admission'] = commit(request, request_binding, manifest, formal, owner, folder,
                                         state, paths, fence, selected, ledger, observation)
            report['status'] = 'PASS_INCREMENTAL_PLAN_ONLY_NOT_LAUNCHED'
        save_new(observation / 'receipt.json', report)
        return report


def commit(request, request_binding, manifest, formal, owner, folder, state, paths, fence, cells, ledger, observation):
    selected = {cell['cell_id']:cell for cell in cells}
    jobs = [slot['original_job'] for cell in cells for slot in cell['formal_slots']]
    job_ids = {j['id'] for j in jobs}
    # Preserve original interleaving/repeat ordering from the full plan.
    jobs = [copy.deepcopy(j) for j in formal['jobs'] if j['id'] in job_ids]
    require(len(jobs) == len(cells)*10 and len(job_ids) == len(jobs), 'Only entire ten-repeat cells may be admitted')
    require(all(not Path(p).exists() for j in jobs for p in complete_job_outputs(j)), 'Formal output appeared during admission')
    require(all(j.get('depends_on') == [next(c['cell_id'] for c in cells if j['id'] in
        {s['original_job']['id'] for s in c['formal_slots']})] for j in jobs), 'Unsupported cross-cell formal dependency')
    batch_id = f"batch_{len(state['batches']):04d}_{digest(sorted(selected))[:12]}"
    folder.mkdir(parents=True, exist_ok=True); batch = folder / 'batches' / batch_id
    batch.mkdir(parents=True, exist_ok=False)
    reservation = dict(schema='eggpu.formal.incremental.reservation.v1', owner=owner, batch_id=batch_id,
        cells={c['cell_id']:c['cell_identity_sha256'] for c in cells},
        original_jobs={j['id']:digest(j) for j in jobs}, receipts=[j['receipt'] for j in jobs],
        performance_measurements=0, workers_launched=0)
    save_new(batch / 'reservation.json', reservation); reservation_binding = bind(batch / 'reservation.json')
    fence_legacy(paths, fence)
    entry = dict(batch_id=batch_id, status='RESERVED', reservation=reservation_binding)
    state['batches'].append(entry)
    for cell in cells:
        state['cells'][cell['cell_id']] = dict(batch_id=batch_id, cell_identity_sha256=cell['cell_identity_sha256'])
    replace_json(folder / 'registry.json', state)
    # Reservation persists if the following preparation fails; no automatic retry.
    gate = dict(schema='eggpu.formal.incremental.batch-gate.v1', owner=owner, batch_id=batch_id,
        reservation=reservation_binding, registry=str(folder / 'registry.json'), parent_manifest=request['parent_manifest'],
        pending=manifest['groups'][request['group']]['pending'], source_freeze=manifest['source_freeze'],
        source_root=manifest['source_root'], native_dso=manifest['native_dso'], request=request_binding,
        complete_matrix_request=bind(observation / 'inspect_request.json'),
        complete_ledger=bind(observation / 'complete_ledger.json'),
        completed_projection=fence['completed_projection'], legacy_fence=fence,
        base_controller=request['controller'], supervisor=request['supervisor'],
        cells={c['cell_id']:dict(identity=c['cell_identity_sha256'], proof_audit=c['proof_audit'],
               jobs=[s['original_job'] for s in c['formal_slots']]) for c in cells})
    save_new(batch / 'gate.json', gate); gate_binding = bind(batch / 'gate.json')
    gate_jobs = []
    for cell in cells:
        result = batch / 'cpu_gates' / cell['cell_id'] / 'result.json'
        command = [request['python_invocation'],'-B',str(HERE / 'cell_gate.py'), '--spec',gate_binding['path'],
                   '--cell',cell['cell_id'],'--output',str(result),'--phase','cpu']
        bindings = [gate_binding, reservation_binding, request_binding, request['parent_manifest'], gate['complete_ledger'],
                    manifest['groups'][request['group']]['pending'], bind(__file__), bind(HERE / 'cell_gate.py'),
                    bind(HERE / 'run_incremental_queue.py'),
                    request['python'], request['controller'], request['supervisor']]
        gate_jobs.append(dict(id=cell['cell_id'], phase='cpu', command=command, receipt=str(result),
            expected_statuses=['PASS_CURRENT_INCREMENTAL_CELL_GATE'], timeout=86400,
            bindings=bindings, depends_on=[], environment={'CUDA_VISIBLE_DEVICES':''}))
    plan = copy.deepcopy(formal); plan['jobs'] = gate_jobs + jobs
    plan.update(schema='eggpu.formal.incremental.plan.v1', admission='DEEP_CURRENT_PROOFS_TEN_REPEATS_PER_SELECTED_CELL',
        original_complete_matrix=manifest['groups'][request['group']]['pending'], original_cell_count=ledger['original_cell_count'],
        original_formal_slots=ledger['original_formal_slots'], selected_cell_ids=list(selected), gate=gate_binding,
        required_controller=bind(HERE / 'run_incremental_queue.py'), base_controller=request['controller'], required_supervisor=request['supervisor'],
        complete_matrix_status='Unselected cells remain PENDING/FAIL/capability in complete ledger; no denominator change.',
        formal_job_mutation='None: original command, env, bindings, dependencies, maintenance, timeout, output path and all other fields are identical.')
    require([j for j in plan['jobs'] if j['phase']=='formal'] == jobs, 'Formal job mutation forbidden')
    save_new(batch / 'plan.json', plan)
    receipt = dict(schema='eggpu.formal.incremental.batch-receipt.v1', status='PASS_INCREMENTAL_PLAN_ONLY_NOT_LAUNCHED',
        workers_launched=0, gpu_executed=False, original_jobs=len(formal['jobs']), admitted_formal_jobs=len(jobs),
        cpu_gate_jobs=len(gate_jobs), plan=bind(batch / 'plan.json'), gate=gate_binding,
        reservation=reservation_binding, complete_ledger_counts=ledger['counts'],
        startup=[request['python_invocation'],'-B',str(HERE/'run_incremental_queue.py'), '--spec',str(batch/'gate.json'), '--plan',str(batch/'plan.json'),
                 '--output',str(batch/'queue'),'--controller-lock',request['controller_lock']])
    save_new(batch / 'receipt.json', receipt)
    entry.update(status='COMMITTED', plan=bind(batch / 'plan.json'), gate=gate_binding, receipt=bind(batch/'receipt.json'))
    replace_json(folder / 'registry.json', state)
    return receipt


def verify_cell(spec_path, cell_id, output):
    gate = load(spec_path); require(gate.get('schema') == 'eggpu.formal.incremental.batch-gate.v1', 'Unknown batch gate')
    require(cell_id in gate['cells'], 'Cell outside admitted batch')
    reservation = load(checked(gate['reservation'])); cell = gate['cells'][cell_id]
    require(reservation['owner'] == gate['owner'] and reservation['cells'].get(cell_id) == cell['identity'], 'Reservation identity differs')
    state = registry_read(Path(gate['registry']).parent, gate['owner'])
    require(state['cells'].get(cell_id) == dict(batch_id=gate['batch_id'], cell_identity_sha256=cell['identity']), 'Cell not exclusively reserved')
    checked(gate['parent_manifest']); checked(gate['pending']); checked(gate['native_dso']); checked(gate['request'])
    freeze = load(checked(gate['source_freeze']))
    require(freeze['source_tree_sha256'] == gate['owner']['source_sha256'], 'Source epoch changed')
    for rel, expected in freeze['source_files'].items():
        actual = bind(Path(gate['source_root'])/rel)
        require({k:actual[k] for k in ('bytes','sha256')} == expected, 'Current source changed: '+rel)
    for kind in ('tiny','target'):
        proof = cell['proof_audit'][kind]
        checked(proof['receipt']); checked(proof['epoch'])
    for job in cell['jobs']:
        require(reservation['original_jobs'][job['id']] == digest(job), 'Original formal job changed')
        require(all(not Path(p).exists() for p in complete_job_outputs(job)), 'Cell was attempted after reservation; no replay')
    result = dict(status='PASS_CURRENT_INCREMENTAL_CELL_GATE', phase='cpu', cell_id=cell_id,
        batch_id=gate['batch_id'], gpu_executed=False, performance_eligible=False,
        spec=bind(spec_path), reservation=gate['reservation'],
        meaning='Admission already performed full frozen worker proof/output audits; current source and exact proof identities remain bound. Formal workers recheck their unchanged own proofs.')
    save_new(output, result); return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--admit',action='store_true');args=parser.parse_args()
    report=run(args.request,args.output,admit=args.admit)
    print(json.dumps({k:report[k] for k in ('status','original_cell_count','original_formal_slots','selected_formal_slots','workers_launched')}))
