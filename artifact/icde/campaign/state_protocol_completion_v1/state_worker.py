#!/usr/bin/env python3
"""Primary-only real E3/E4/E5 public-session worker and physical-byte pilot.

Importing/preparing this program is CPU-only. GPU actions require --execute-gpu,
explicit UUIDs and the shared experiment lock. Use the campaign supervisor's
60-minute process-group deadline around each invocation.
"""
from pathlib import Path
import argparse, fcntl, hashlib, json, math, os, sys, time
import state_protocol as c

COMMON_ENV={'EGGPU_COMM_BACKEND':'host_staging','EGMGPU_WORKSPACE_POLICY':'per_worker',
 'EGGPU_DENSE_WEIGHT_LEASE':'0','EGGPU_HOST_SNAPSHOT_STORAGE':'shared',
 'EGMGPU_SSSP_WEIGHT_ENCODING':'f64','EGMGPU_SSSP_WIRE_ENCODING':'f64',
 # Full certification is the frozen, calibration-free native default. Adaptive
 # certification requires an independently frozen density/calibration pair.
 'EGGPU_SSSP_WEIGHT_CERTIFICATION':'full','EGGPU_SINGLE_OWNER_SSSP':'1',
 'EGGPU_NATIVE_SSSP_OWNED_OUTPUT':'1','EGGPU_SSSP_DIRECT_FRONTIER':'1',
 'EGGPU_NATIVE_PR_OWNED_OUTPUT':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}


def strict(value):
    if isinstance(value,dict):return {str(k):strict(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [strict(x) for x in value]
    if hasattr(value,'item'):return strict(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        if value==math.inf:return {'special_value':'+inf'}
        raise ValueError('Unexpected nonfinite metric')
    return value


def load_inputs(path):
    record=json.loads(path.read_text())
    if record['status']!='PASS_CPU_INPUTS_GPU_UNQUALIFIED':raise ValueError('Unqualified input preparation')
    if record.get('schema')!='state.inputs.v2':raise ValueError('Prepare state.inputs.v2 with explicit BFS/WCC projections')
    for row in record['datasets'].values():
        for item in [row['manifest'],row['positions'],*row['canonical'].values(),*row['manifest_files']]:c.checked(item)
        for item in row['projections'].values():
            for binding in [item['manifest'],*item['files']]:c.checked(binding)
    return record


from state_contract import source_binding

def view_bytes(state):
    result={}
    for row in state['per_device']:
        owners={o['owner_id']:o for o in row['owners']}
        result[str(row['device_id'])]=sum(a['bytes'] for a in row['allocations'] if any(owners[o]['kind']=='view' for o in a['owners']))
    return result


def source_for(row,index):return int(row['sources'][index%len(row['sources'])])


def manifests_for(inputs,names):
    result={}
    for name in names:
        row=inputs['datasets'][name];result[name+'@0']=row['manifest']['path']
        for value in row['projections'].values():result[value['identity']]=value['manifest']['path']
    return result


def execution_snapshot(row,snapshot,function):
    return row['projections'][function]['identity'] if function in ('bfs','wcc') else snapshot


def output_quality(values,row,version,function,source,cache=None):
    from state_reference import output_quality as independent_quality
    return independent_quality(values,row,version,function,source,cache)


def packet_for(row, version):
    import numpy as np
    positions = np.load(row['positions']['path'], allow_pickle=False, mmap_mode='r')
    weights = np.memmap(row['canonical']['weights']['path'], dtype='<f8', mode='r', shape=(row['m'],))
    cuts = row['edge_cuts']
    for position in positions:
        rank = int(np.searchsorted(cuts, position, side='right') - 1)
        yield rank, int(position) - cuts[rank], float(1 + (weights[position] - 1 + version) % 16)


def run_workflow(a, inputs, names, create_session, acquire, budget):
    """One continuous application interval, retaining every complete result."""
    import numpy as np
    import state_contract as contract
    route, policy = c.ARMS[a.arm]
    events = c.trace(a.experiment, names, a.order, a.phase == 'tiny')
    held, records, leases = [], [], []
    rebuilt = {}
    rebuild_directory = a.output / 'temporary_rebuild_inputs'
    session = reader = None
    initialized = begin = None
    failure = None
    final_inventory = None
    start = time.perf_counter_ns()
    try:
        session = create_session()
        # E3 has the same completed base queries in both reader treatments.
        if a.experiment == 'E3':
            name = names[0]
            row = inputs['datasets'][name]
            for function in c.FUNCTIONS:
                params = {'source': source_for(row, 0)} if function in ('bfs', 'sssp') else {}
                session.query(execution_snapshot(row, name + '@0', function), function, params)
        initialized = begin = time.perf_counter_ns()
        for index, event in enumerate(events):
            tick = time.perf_counter_ns()
            row = inputs['datasets'][event['dataset']]
            if event['kind'] == 'commit':
                if reader is not None:
                    raise ValueError('The preceding four-query batch did not release its old reader')
                if a.old_readers == 'per_update':
                    reader = contract.OldReader(session, event['base'], acquire,
                        inputs['partitions'], a.phase == 'tiny')
                    reader.receipt['commit_event_index'] = index
                    leases.append(reader.receipt)
                if a.arm == 'Rebuild':
                    from state_rebuild import rebuild
                    selected, detail = rebuild(session, row, event, inputs['partitions'],
                        a.source, rebuild_directory)
                    rebuilt[event['snapshot']] = selected
                else:
                    detail = session.commit_weights(event['base'], event['snapshot'],
                                                    packet_for(row, event['version']), storage='native_memory')
            else:
                source = source_for(row, event['source_index'])
                params = {'source': source} if event['function'] in ('bfs', 'sssp') else {}
                executed = execution_snapshot(row, event['snapshot'], event['function'])
                if event['snapshot'] in rebuilt:
                    key = event['function'] if event['function'] in ('bfs','wcc') else 'weighted'
                    executed = rebuilt[event['snapshot']][key]
                result = session.query(executed, event['function'], params)
                values = result.values
                if (values.shape != (row['n'],) or values.dtype != np.float64 or
                        values.flags.writeable or not values.flags.owndata or values.base is not None):
                    raise ValueError('The public result is not a complete owning readonly FP64 vector')
                if any(np.shares_memory(values, prior) for _, prior, _ in held):
                    raise ValueError('Two completed request outputs share writable backing storage')
                detail = {'session_seconds': result.seconds, 'metrics': result.backend_metrics,
                    'execution_snapshot': executed, 'semantic_snapshot': event['snapshot'],
                    'intervention': c.validate_intervention(result.backend_metrics, route, policy, budget)}
                held.append((event, values, source))
                if event['function'] == c.FUNCTIONS[-1] and reader is not None:
                    reader.receipt['last_query_event_index'] = index
                    reader.close()
                    reader = None
            records.append({'index': index, **event,
                            'public_seconds': (time.perf_counter_ns() - tick) / 1e9, 'details': detail})
        if reader is not None:
            raise ValueError('The workflow ended with an unreleased old reader')
        complete = time.perf_counter_ns()
        # Inventory and hashing are audit work after the actual continuous end.
        final_inventory = session.inventory()
    except BaseException as error:
        complete = time.perf_counter_ns()
        failure = type(error).__name__ + ': ' + str(error)
    finally:
        before_close = [hashlib.sha256(memoryview(values).cast('B')).hexdigest() for _, values, _ in held]
        if reader is not None:
            try:
                reader.close()
            except BaseException as error:
                failure = (failure or '') + '; reader close: ' + type(error).__name__ + ': ' + str(error)
        if session is not None:
            try:
                session.close()
            except BaseException as error:
                failure = (failure or '') + '; Session close: ' + type(error).__name__ + ': ' + str(error)
        if a.arm == 'Rebuild' and session is not None and session.closed:
            from state_rebuild import cleanup
            cleanup(rebuild_directory)
        closed = time.perf_counter_ns()
    complete_trace = failure is None and len(records) == len(events)
    counters = {}
    for event in records:
        if event['kind'] != 'query':
            continue
        for delta in event['details']['metrics']['native_dynamic_patch']['per_device_delta'].values():
            for key, value in delta.items():
                if isinstance(value, (int, float)):
                    counters[key] = counters.get(key, 0) + value
    return held, before_close, {'execution_complete': complete_trace, 'execution_error': failure,
        'events': records, 'leases': leases, 'final_inventory': final_inventory,
        'maintenance_counters': counters, 'old_reader_batches': len(leases),
        'old_reader_readback_in_application': a.phase == 'tiny' and a.old_readers == 'per_update',
        'initialization_seconds': (initialized - start) / 1e9 if initialized is not None else None,
        'workflow_complete_host_seconds': (complete - begin) / 1e9 if complete_trace else None,
        'first_complete_workflow_seconds': (complete - start) / 1e9 if complete_trace else None,
        'attempt_seconds': (complete - start) / 1e9,
        'audit_and_close_seconds': (closed - complete) / 1e9,
        'continuous_interval': True, 'sum_of_phases_used_as_total': False,
        'rebuild_control': a.arm == 'Rebuild',
        'rebuild_temporary_inputs_removed_after_session_close': a.arm == 'Rebuild' and not rebuild_directory.exists(),
        'timing': 'Continuous public calls, commits, reader acquire/close, and equal event recording; '
                  'E3 base initialization also reported separately; all hashes/references/output disk/Session close '
                  'outside the interval. Tiny-only full old-view readbacks are diagnostic, never paper timings.'}


def audit_outputs(a, inputs, held, before_close):
    from state_reference import ReferenceCache
    from state_contract import OutputStore
    reference_cache = ReferenceCache(a.reference_cache or a.output.parent / 'independent_references',
                                    int(a.reference_cache_gib * (1 << 30)))
    store = OutputStore(a.output_store)
    checks = []
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / 'output_index.jsonl').open('x') as journal:
        for index, (event, values, source) in enumerate(held):
            item = {'event': event, 'source': source,
                    'output_sha256': hashlib.sha256(memoryview(values).cast('B')).hexdigest()}
            # Save completed outputs even if their later numerical audit fails.
            item['output'] = store.put(values)
            item['survives_session_close'] = item['output_sha256'] == before_close[index]
            journal.write(json.dumps(strict(dict(kind='complete_output', index=index, **item)), allow_nan=False) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            try:
                if not item['survives_session_close']:
                    raise ValueError('Owning output changed after Session close')
                item['quality'] = output_quality(values, inputs['datasets'][event['dataset']],
                    event['version'], event['function'], source, reference_cache)
            except BaseException as error:
                item['quality'] = {'passed': False, 'error': type(error).__name__ + ': ' + str(error)}
            journal.write(json.dumps(strict(dict(kind='quality', index=index, quality=item['quality'])), allow_nan=False) + '\n')
            journal.flush()
            checks.append(item)
    return checks, {'source': c.bind(Path(__file__).with_name('state_reference.py')),
        'directory': str(reference_cache.path), 'disk_budget_bytes': reference_cache.max_bytes,
        'quality_scope': 'Every output audited in full; every PR residual recomputed; CPU references never consume GPU answers'}


def execute(a):
    import state_contract as contract
    if not a.execute_gpu:
        raise ValueError('--execute-gpu is required; only the primary agent owns GPU execution')
    inputs = load_inputs(a.inputs)
    names = a.datasets.split(',') if a.datasets else list(inputs['datasets'])
    contract.validate_arguments(a, inputs, names)
    uuids = a.gpu_uuids.split(',')
    devices = list(range(len(uuids)))
    if (len(set(uuids)) != len(uuids) or inputs['partitions'] != len(uuids) or
            any(not value.startswith('GPU-') for value in uuids)):
        raise ValueError('Explicit GPU UUIDs must match prepared partitions')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).with_name('baselines')))
    from hytgraph_qualify import resource_gate
    record = {'schema': 'state.complete.workflow.v4', 'status': 'RUNNING', 'gpu_executed': False,
        'performance_eligible': False, 'phase': a.phase, 'repeat': a.repeat, 'experiment': a.experiment,
        'arm': a.arm, 'old_readers': a.old_readers, 'datasets': names, 'quality_pass': False}
    try:
        with a.lock.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.cpu_set:
                cores = [int(value) for value in a.cpu_set.split(',')]
                if len(cores) != 16 or len(set(cores)) != 16 or not set(cores) <= set(os.sched_getaffinity(0)):
                    raise ValueError('Exactly 16 currently available CPU cores required')
                os.sched_setaffinity(0, cores)
            rounds = 2 if inputs['qualification_only'] else 20 if a.experiment == 'E3' else 4 if a.experiment == 'E3R' else 8
            required_host = 4. + sum(inputs['datasets'][name]['n'] * 8 * 4 * rounds +
                inputs['datasets'][name]['m'] * 8 * (1 if a.experiment == 'E4' else rounds + 1) +
                inputs['datasets'][name]['m'] * 4 + inputs['datasets'][name]['n'] * 24 for name in names) / (1024 ** 3)
            if a.arm == 'Rebuild':
                # Immutable host snapshots remain registered to preserve old
                # names. Reserve full host partition/projection payloads for
                # every version, not just changed weights; this may expose a
                # genuine host-capacity boundary.
                payload = sum(sum(x['bytes'] for x in row['manifest_files']) +
                    sum(x['bytes'] for projection in row['projections'].values() for x in projection['files'])
                    for name, row in inputs['datasets'].items() if name in names)
                required_host += (rounds + 2) * payload / (1 << 30)
            resources = [resource_gate(uuid, required_host) for uuid in uuids]
            import shutil
            a.output_store.mkdir(parents=True, exist_ok=True)
            output_reserve = (sum(inputs['datasets'][event['dataset']]['n'] * 8
                for event in c.trace(a.experiment, names, a.order, a.phase == 'tiny') if event['kind'] == 'query')
                if a.action == 'run' else 0)
            free_disk = shutil.disk_usage(a.output_store).free
            rebuild_reserve = (rounds * payload if a.arm == 'Rebuild' else 0)
            if free_disk < output_reserve + rebuild_reserve + (2 << 30):
                raise RuntimeError('Insufficient disk for complete output vectors plus 2 GiB reserve')
            record['output_admission'] = {'available_bytes': free_disk, 'worst_case_vector_bytes': output_reserve,
                                          'rebuild_temporary_payload_reserve': rebuild_reserve,
                                          'safety_reserve_bytes': 2 << 30, 'dedup_savings_assumed': False}
            os.environ.update(COMMON_ENV, **c.arm_environment(a.arm), CUDA_VISIBLE_DEVICES=','.join(uuids), CUDA_DEVICE_ORDER='PCI_BUS_ID')
            actual_environment = {key: value for key, value in os.environ.items()
                if key.startswith(('EGGPU_', 'EGMGPU_', 'NCCL_', 'UCX_', 'CUDA_', 'CUPY_')) or
                key in COMMON_ENV or key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH')}
            sys.path.insert(0, str(a.source.resolve()))
            if a.native_dso:
                sys.path.insert(1, str(a.native_dso.resolve().parent))
            import cpp_easygraph as cpp
            from easygraph.gpu import Session, _controller
            from easygraph.gpu._snapshot_lease import acquire_snapshot_lease
            if Path(sys.modules[Session.__module__].__file__).resolve() != a.source.resolve() / 'easygraph/gpu/session.py':
                raise ValueError('Wrong public Session source imported')
            if a.native_dso and Path(cpp.__file__).resolve() != a.native_dso.resolve():
                raise ValueError('Wrong native DSO imported')
            if 'closure_lru' not in _controller.POLICIES:
                raise ValueError('Source lacks strong C0 control')
            binding = source_binding(a.source, cpp)
            config = contract.configuration(a, binding, devices, actual_environment, sorted(os.sched_getaffinity(0)))
            record.update(source_binding=binding, configuration=config, configuration_sha256=contract.digest(config),
                inputs=c.bind(a.inputs), devices=devices, gpu_uuids=uuids, resources=resources,
                environment=actual_environment, cpu_affinity=sorted(os.sched_getaffinity(0)))
            manifests = manifests_for(inputs, names)
            def session(selected, route='persistent_full', policy='closure', budget=None):
                from state_controls import install_controller
                result = Session.native(selected, devices=devices, cpp=cpp, route=route,
                    policy='closure' if policy in ('no_retention','view_lru') else policy,
                    budgets=budget, retirement_policy='lineage', inventory_mode='recheck', batch_devices=True)
                install_controller(result, policy)
                return result
            if a.action == 'pilot':
                rows, inventories = {}, {}
                record['gpu_executed'] = True
                for name in names:
                    with session(manifests_for(inputs, [name])) as state:
                        row = inputs['datasets'][name]
                        for function in c.FUNCTIONS:
                            params = {'source': source_for(row, 0)} if function in ('bfs', 'sssp') else {}
                            state.query(execution_snapshot(row, name + '@0', function), function, params)
                        inventories[name] = state.inventory()
                        rows[name] = view_bytes(inventories[name])
                record.update(schema='state.byte.pilot.v4', status='PASS_PHYSICAL_BYTE_PILOT',
                    resident_view_bytes=rows, inventories=inventories, selection_uses_timing=False)
                return
            route, policy = c.ARMS[a.arm]
            budget = budget_record = None
            if a.experiment not in ('E3','E3R'):
                if a.pilot is None:
                    raise ValueError('E4/E5 require the frozen physical-byte pilot')
                pilot = json.loads(a.pilot.read_text())
                if (pilot.get('schema') != 'state.byte.pilot.v4' or pilot.get('status') != 'PASS_PHYSICAL_BYTE_PILOT' or
                        pilot['source_binding'] != binding or pilot['inputs'] != c.bind(a.inputs) or
                        pilot['devices'] != devices or pilot['environment'] != actual_environment or
                        set(pilot['resident_view_bytes']) != set(names)):
                    raise ValueError('Byte pilot differs from current source/input/resource/pool')
                budget_record = c.budgets(pilot, a.beta)
                budget_record['pilot'] = c.bind(a.pilot)
                budget = budget_record['per_device']
            record['budget'] = budget_record
            record['qualification'] = {}
            for phase, path in (('tiny', a.tiny_proof), ('target', a.target_proof)):
                if (phase == 'tiny' and a.phase in ('target', 'formal')) or (phase == 'target' and a.phase == 'formal'):
                    record['qualification'][phase] = contract.validate_proof(path, phase, config,
                        c.bind(a.inputs), names, budget_record)
            record['gpu_executed'] = True
            create = lambda: session(manifests, route, policy,
                                     None if budget is None else {int(k): v for k, v in budget.items()})
            held, before_close, measured = run_workflow(a, inputs, names, create, acquire_snapshot_lease, budget)
            record.update(measured)
            checks, reference = audit_outputs(a, inputs, held, before_close)
            record.update(checks=checks, independent_reference=reference, output_store=str(a.output_store.resolve()))
            record['quality_pass'] = bool(measured['execution_complete'] and checks and all(x['quality']['passed'] for x in checks))
            if not record['quality_pass']:
                raise ValueError('Incomplete state workflow or failed full-output audit: ' + str(measured['execution_error']))
            record.update(status='PASS_STATE_COMPLETE_WORKFLOW' if a.phase == 'formal' else 'PASS_STATE_GPU_QUALIFICATION',
                          performance_eligible=a.phase == 'formal')
    except BaseException as error:
        record.update(status='FAIL_STATE_WORKFLOW', error=type(error).__name__ + ': ' + str(error), performance_eligible=False)
        raise
    finally:
        c.save(a.output / 'receipt.json', strict(record))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['pilot', 'run'])
    for name in ('inputs', 'source', 'output', 'lock', 'output-store'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--native-dso', type=Path)
    p.add_argument('--datasets')
    p.add_argument('--experiment', choices=['E3', 'E3R', 'E4', 'E5'], required=True)
    p.add_argument('--arm', choices=c.ARMS, default='D1C1')
    p.add_argument('--order', choices=['grouped', 'interleaved', 'permuted', 'once'], default='interleaved')
    p.add_argument('--beta', type=float, default=.25)
    p.add_argument('--phase', choices=['tiny', 'target', 'formal'], default='tiny')
    p.add_argument('--repeat', type=int, default=0)
    p.add_argument('--old-readers', choices=['none', 'per_update'], default='none')
    for name in ('pilot', 'tiny-proof', 'target-proof', 'reference-cache'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--reference-cache-gib', type=float, default=4.)
    p.add_argument('--gpu-uuids', required=True)
    p.add_argument('--cpu-set', help='Optional exact 16-core affinity, bound into all proofs')
    p.add_argument('--execute-gpu', action='store_true')
    return p


if __name__ == '__main__':
    execute(parser().parse_args())

