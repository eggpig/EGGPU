#!/usr/bin/env python3
"""Independent E3 unweighted-PR negative control: 21 real PR queries, 20 commits.

This module imports no GPU library. Execution needs an explicit --execute-gpu
and is exclusively submitted by the root campaign owner. Frozen state_v4 and
the main 80-query E3 trace are not modified.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

PROTOCOL = 'e3-unweighted-pr-negative-v3-native-device-keys'
QUALITY = {'alpha': .85, 'residual_l1': 1e-6, 'mass_error': 1e-6,
           'reference_l1': 1e-5, 'reference_stop_l1': 1e-12,
           'reference_cap': 2000, 'invariance_l1': 2e-8}
COMMITS = 20
ARMS = {'D0C1': 'persistent_full', 'D1C1': 'lease_sparse'}
FILES = ('pr_control.py', 'pr_plan.py')
WEIGHT_TRANSFER_COUNTERS = ('weight_full_h2d_bytes', 'patch_h2d_bytes',
                            'dense_lease_h2d_bytes', 'cow_d2d_bytes')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def checked(item):
    path = Path(item['path'])
    need(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Bound file changed: ' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def load_support(path):
    spec = importlib.util.spec_from_file_location('_pr_control_frozen_support', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace(name):
    result = [dict(kind='query', dataset=name, version=0, function='pr')]
    for version in range(1, COMMITS + 1):
        result.append(dict(kind='commit', dataset=name, version=version,
                           base=f'{name}@{version-1}', snapshot=f'{name}@{version}'))
        result.append(dict(kind='query', dataset=name, version=version, function='pr'))
    return result


def validate_arguments(a, inputs):
    need(a.old_readers == 'none', 'per_update is inapplicable: existing readonly lease requires resident SSSP/weighted-PR views; this workload only runs unweighted PR')
    need(a.arm in ARMS, 'C must remain fixed at C1')
    need(inputs['qualification_only'] == (a.phase == 'tiny'), 'Tiny/target domain mismatch')
    need(a.repeat in range(10) and (a.phase == 'formal' or a.repeat == 0), 'Require ten formal repeats 0..9; qualification repeat0')
    need(a.dataset in inputs['datasets'], 'Dataset not in explicit input')
    need((a.phase == 'tiny' and a.dataset == 'TINY_A') or
         (a.phase != 'tiny' and a.dataset in ('US','IN','OR')), 'Frozen negative-control graph differs')
    row = inputs['datasets'][a.dataset]
    need(row['changed_positions'] > 0 and row['requested_rho'] > 0, 'Negative control must actually change weights')
    if a.phase != 'tiny':
        need(row['requested_rho'] == 1e-4 and row['locality'] == 'uniform', 'Main negative control fixes rho1e-4 uniform')
    need(a.phase == 'tiny' or a.tiny_proof is not None, 'Exact tiny qualification required')
    need(a.phase != 'formal' or a.target_proof is not None, 'Exact target qualification required')
    need(inputs['partitions'] == len(a.gpu_uuids.split(',')), 'Prepared partitions differ from GPUs')


def unit_oracle(row):
    """Independent outgoing-unit CSR oracle: never opens weight payloads."""
    import numpy as np
    from scipy.sparse import csr_matrix
    n, m = row['n'], row['m']
    offsets = np.memmap(row['canonical']['offsets']['path'], dtype='<i4', mode='r', shape=(n+1,))
    indices = np.memmap(row['canonical']['indices']['path'], dtype='<i4', mode='r', shape=(m,))
    unit = np.ones(m, dtype=np.float64)
    matrix = csr_matrix((unit, indices, offsets), shape=(n,n), copy=False)
    degree = np.diff(offsets).astype(np.float64)
    inverse = np.divide(1., degree, out=np.zeros(n), where=degree > 0)
    dangling = degree == 0
    transpose = matrix.T
    def step(values):
        return (1 - QUALITY['alpha']) / n + QUALITY['alpha'] * (
            transpose @ (values * inverse) + values[dangling].sum() / n)
    reference = np.full(n, 1 / n)
    for iteration in range(QUALITY['reference_cap']):
        nxt = step(reference)
        delta = float(np.abs(nxt-reference).sum())
        reference = nxt
        if delta < QUALITY['reference_stop_l1']:
            reference.flags.writeable = False
            return step, reference, dict(solver='independent_unit_outgoing_power_iteration',
                iterations=iteration+1, last_update_l1=delta, input_weight_payload_read=False,
                canonical_topology={k: row['canonical'][k] for k in ('offsets','indices')})
    raise ValueError('Independent unweighted PR oracle failed to converge')


def vector_hash(values):
    return hashlib.sha256(memoryview(values).cast('B')).hexdigest()


def numerical_quality(values, step, reference, baseline):
    import numpy as np
    need(values.dtype == np.float64 and values.shape == reference.shape and
         values.flags.owndata and not values.flags.writeable and values.base is None,
         'Complete independent owning readonly FP64 output required')
    residual = float(np.abs(values-step(values)).sum())
    error = float(np.abs(values-reference).sum())
    mass = float(abs(values.sum()-1))
    invariant = float(np.abs(values-baseline).sum())
    need(np.all(np.isfinite(values)) and np.all(values >= 0) and
         residual <= QUALITY['residual_l1'] and error <= QUALITY['reference_l1'] and
         mass <= QUALITY['mass_error'] and invariant <= QUALITY['invariance_l1'],
         'Unweighted PR quality or weight-invariance failed')
    return dict(passed=True, residual_l1=residual, reference_l1=error, mass_error=mass,
                invariance_l1=invariant, bitwise_equal_to_base=vector_hash(values)==vector_hash(baseline))


def validate_commit(receipt, expected_topology, previous_weight, changed_positions):
    need(receipt['kind'] == 'native_memory_absolute_weight_commit' and
         receipt['snapshot_storage'] == 'native_memory', 'Expected real native FP64 commit')
    need(receipt['positions'] == changed_positions, 'Weight updates were omitted')
    need(receipt['topology_sha256'] == expected_topology, 'Commit changed topology')
    need(receipt['weight_sha256'] != previous_weight, 'Commit did not change weighted snapshot content')
    return receipt['weight_sha256']


def query_evidence(result, topology_name, arm, partitions, version):
    need(result.algorithm == 'pr' and result.graph_id == topology_name, 'Must actually query the original unweighted topology projection')
    need(result.contract is not None and result.contract.algorithm == 'pr' and
         result.contract.weight_projection == 'unweighted', 'Public result has wrong weight semantics')
    metrics = result.backend_metrics
    need(metrics.get('iterations', 0) > 0 and metrics.get('converged') is True,
         'Every PR query must execute and converge; cached answer replay is not allowed')
    dynamic = metrics['native_dynamic_patch']
    need(dynamic['route'] == ARMS[arm] and dynamic['retirement_policy'] == 'lineage', 'Wrong D route')
    resource = metrics['resource_management']
    need(resource['state_switch_policy'] == 'closure', 'C must remain fixed')
    delta = dynamic['per_device_delta']
    need(all(type(device) is int for device in delta) and set(delta) == set(range(partitions)),
         'Incomplete actual device counters: expected native integer device IDs, got '+repr(list(delta)))
    totals = {key: sum(row.get(key, 0) for row in delta.values()) for key in WEIGHT_TRANSFER_COUNTERS}
    need(all(type(value) in (int,float) and math.isfinite(value) and value >= 0
             for value in totals.values()), 'Invalid actual weighted-view byte counter')
    if version:
        need(all(value == 0 for value in totals.values()), 'Weight-only commit unexpectedly caused unweighted PR to maintain GPU weight arrays')
    return dict(weighted_view_transfer_counters=totals,
        counter_scope='Native actual weighted-view full upload, scatter packet and COW counters; absent counter key means zero in this frozen instrumentation',
        total_cuda_copy_bytes='NOT_FULLY_INSTRUMENTED_DO_NOT_INFER_ZERO',
        device_view_cache_hit=metrics.get('device_view_cache_hit'),
        shared_array_new_bytes=metrics.get('shared_array_new_bytes'),
        communication_bytes=metrics.get('communication_bytes'),
        communication_bytes_scope=metrics.get('communication_bytes_scope'))


def run_workflow(a, row, make_session, packet):
    import numpy as np
    held, events, session = [], [], None
    topology_name = row['projections']['bfs']['identity']
    manifest = read(checked(row['manifest']))
    expected_topology = manifest['graph']['topology_sha256']
    previous_weight = manifest['graph']['weight_sha256']
    started = time.perf_counter_ns()
    initialized = None
    failure = None
    final_inventory = None
    failed_event = None
    try:
        session = make_session()
        initialized = time.perf_counter_ns()
        for event in trace(a.dataset):
            begin = time.perf_counter_ns()
            failed_event = dict(event)
            if event['kind'] == 'commit':
                details = session.commit_weights(event['base'], event['snapshot'],
                    packet(row, event['version']), storage='native_memory')
                previous_weight = validate_commit(details, expected_topology, previous_weight,
                                                  row['changed_positions'])
            else:
                result = session.query(topology_name, 'pr', {})  # New native invocation on every event.
                values = result.values
                failed_event['observed_query'] = dict(public_seconds=result.seconds,
                    algorithm=result.algorithm, graph_id=result.graph_id,
                    output_shape=list(values.shape), output_dtype=str(values.dtype),
                    metrics=result.backend_metrics)
                need(values.shape == (row['n'],) and values.dtype == np.float64 and
                     values.flags.owndata and not values.flags.writeable and values.base is None,
                     'PR did not return a complete independent public vector')
                need(not any(np.shares_memory(values, old) for _, old in held), 'Output buffer reused across requests')
                held.append((event, values))
                dependency = query_evidence(result, topology_name, a.arm, a.partitions, event['version'])
                details = dict(public_seconds=result.seconds, execution_snapshot=topology_name,
                               algorithm='pr', dependency_audit=dependency, metrics=result.backend_metrics)
            events.append(dict(event, interval_seconds=(time.perf_counter_ns()-begin)/1e9, details=details))
            failed_event = None
        ended = time.perf_counter_ns()
        final_inventory = session.inventory()
    except BaseException as error:
        ended = time.perf_counter_ns()
        failure = type(error).__name__ + ': ' + str(error)
    finally:
        before_close = [vector_hash(values) for _, values in held]
        if session is not None:
            try:
                session.close()
            except BaseException as error:
                failure = (failure or '') + '; close: ' + type(error).__name__ + ': ' + str(error)
    complete = failure is None and len(events) == 41 and len(held) == 21
    return held, before_close, dict(execution_complete=complete, execution_error=failure,
        events=events, failed_event=failed_event, final_inventory=final_inventory,
        first_complete_workflow_seconds=(ended-started)/1e9 if complete else None,
        workflow_complete_host_seconds=(ended-initialized)/1e9 if complete else None,
        continuous_interval=True, sum_of_phases_used_as_total=False,
        timing='Continuous Session construction through 21 fresh full-host PR queries and20 native weight commits; hashes, oracle, disk, final inventory and close are outside the interval',
        weighted_view_transfer_scope='Weight maintenance only; total CUDA transfer is not inferred')


def audit_outputs(a, row, held, before_close, contract):
    store = contract.OutputStore(a.output_store)
    saved = []
    # Save every completed output before computing an oracle that might fail.
    with (a.output / 'output_index.jsonl').open('x') as journal:
        for index, (event, values) in enumerate(held):
            item = dict(event=event, output=store.put(values),
                        survives_session_close=vector_hash(values)==before_close[index])
            journal.write(json.dumps(item)+'\n'); journal.flush(); os.fsync(journal.fileno())
            saved.append(item)
    step, reference, solver = unit_oracle(row)
    for item, (_, values) in zip(saved, held):
        try:
            need(item['survives_session_close'], 'Output changed after Session close')
            item['quality'] = numerical_quality(values, step, reference, held[0][1])
        except BaseException as error:
            item['quality'] = dict(passed=False, error=type(error).__name__+': '+str(error))
    return saved, solver


def validate_proof(path, phase, config, inputs, name, contract):
    proof = read(path)
    need(proof.get('schema') == PROTOCOL and proof.get('phase') == phase and
         proof.get('status') == 'PASS_UNWEIGHTED_PR_QUALIFICATION' and
         proof.get('configuration') == config and proof.get('configuration_sha256') == digest(config),
         'Negative-control proof source/protocol/resource/phase differs')
    need(proof.get('execution_complete') and proof.get('quality_pass') is True,
         'Incomplete negative-control qualification')
    checked(proof['inputs'])
    if phase == 'target':
        need(proof['inputs'] == inputs and proof['dataset'] == name, 'Target proof input differs')
    else:
        need(proof['dataset'] == 'TINY_A' and proof['inputs'] == config['tiny_inputs'], 'Wrong tiny domain/input binding')
    expected = trace(proof['dataset'])
    need(len(proof.get('events',[])) == 41 and len(proof.get('checks',[])) == 21, 'Incomplete proof trace')
    for exp, measured in zip(expected, proof['events']):
        need(all(measured.get(k) == v for k,v in exp.items()), 'Proof event differs')
    for exp, item in zip([x for x in expected if x['kind']=='query'], proof['checks']):
        need(item['event'] == exp and item.get('quality',{}).get('passed') is True and
             item.get('survives_session_close'), 'Missing full quality/lifetime proof')
        contract.check_output(item['output'])
    return bind(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('deployment', 'support', 'inputs', 'output', 'output-store'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--phase', choices=('tiny','target','formal'), required=True)
    p.add_argument('--repeat', type=int, default=0)
    p.add_argument('--gpu-uuids', required=True)
    p.add_argument('--cpu-set', required=True)
    p.add_argument('--old-readers', choices=('none','per_update'), default='none')
    p.add_argument('--tiny-proof', type=Path)
    p.add_argument('--target-proof', type=Path)
    p.add_argument('--execute-gpu', action='store_true')
    return p


def execute(a):
    need(a.execute_gpu, 'GPU execution is owned by the root campaign controller')
    support = load_support(a.support)
    dep = read(a.deployment)
    need(bind(a.support) == dep['planner'], 'Must reuse the explicitly bound admission support')
    freeze = read(checked(dep['source_freeze']))
    support.source_files(Path(dep['source_root']), freeze)
    runner_freeze = read(checked(dep['runner_freeze']))
    for name, expected in runner_freeze['files'].items():
        checked(dict(expected, path=str(Path(dep['runner_root'])/name)))
    for key in ('python','native_dso','native_build_receipt','registry'):
        checked(dep[key])
    need(bind(sys.executable)['sha256'] == dep['python']['sha256'], 'Wrong worker Python')
    python_runtime = support.verify_runtime(dep)
    worker, contract, protocol = support.imports(dep['runner_root'])
    inputs = worker.load_inputs(a.inputs)
    support.validate_input_metadata(inputs, freeze, read(dep['registry']['path']), tiny=a.phase=='tiny')
    a.partitions = inputs['partitions']
    validate_arguments(a, inputs)
    resource = dep['resources'][str(a.partitions)]
    need(a.gpu_uuids == resource['gpu-uuids'] and a.cpu_set == resource['cpu-set'], 'Resource profile differs')
    if a.phase == 'tiny':
        need(bind(a.inputs) == resource['tiny_inputs'], 'Use the explicitly selected tiny input')
    row = inputs['datasets'][a.dataset]
    topology = read(checked(row['projections']['bfs']['manifest']))
    weighted = read(checked(row['manifest']))
    need(weighted['graph']['weighted'] is True and topology['graph']['weighted'] is False and
         all('weights' not in part['files'] for part in topology['partitions']), 'Use original explicit unweighted projection; never edit weight fields')
    a.output.mkdir(parents=True, exist_ok=False)
    record = dict(schema=PROTOCOL, status='RUNNING', phase=a.phase, repeat=a.repeat,
        experiment='E3_UNWEIGHTED_PR_CONTROL', dataset=a.dataset, arm=a.arm, old_readers='none',
        gpu_executed=False, quality_pass=False, performance_eligible=False)
    try:
        with Path(resource['lock']).open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            affinity = [int(x) for x in a.cpu_set.split(',')]
            need(len(set(affinity)) == 16 and set(affinity) <= set(os.sched_getaffinity(0)), 'Unavailable16-core affinity')
            os.sched_setaffinity(0, affinity)
            sys.path.insert(0, str(Path(dep['runner_root'])/'baselines'))
            from hytgraph_qualify import resource_gate
            # 21 outputs + immutable weight generations + independent unit CSR/oracle reserve.
            host = (4<<30) + 21*row['n']*8 + 21*row['m']*8 + row['m']*12 + row['n']*64
            record['resources'] = [resource_gate(uuid, host/(1<<30)) for uuid in a.gpu_uuids.split(',')]
            import shutil
            a.output_store.mkdir(parents=True,exist_ok=True)
            reserve = 21*row['n']*8 + (2<<30)
            need(shutil.disk_usage(a.output_store).free >= reserve, 'Insufficient complete-output disk reserve')
            record['output_admission'] = dict(vector_bytes=21*row['n']*8,reserve_bytes=2<<30,
                                             dedup_savings_assumed=False,minimum_host_bytes=host)
            environment = support.controlled_environment(support.environment_for(dep, a.partitions), worker.COMMON_ENV, a.gpu_uuids)
            os.environ.update(support.environment_for(dep, a.partitions), **worker.COMMON_ENV,
                              CUDA_VISIBLE_DEVICES=a.gpu_uuids,CUDA_DEVICE_ORDER='PCI_BUS_ID')
            sys.path.insert(0, dep['source_root']);sys.path.insert(1,str(Path(dep['native_dso']['path']).parent))
            import cpp_easygraph as cpp
            from easygraph.gpu import Session
            need(Path(cpp.__file__).resolve() == Path(dep['native_dso']['path']).resolve(), 'Wrong actual DSO')
            need(Path(sys.modules[Session.__module__].__file__).resolve() == Path(dep['source_root'])/'easygraph/gpu/session.py', 'Wrong public Session')
            binding = contract.source_binding(Path(dep['source_root']),cpp)
            binding['negative_control'] = {name:bind(Path(__file__).with_name(name)) for name in FILES}
            binding['admission_support'] = bind(a.support)
            config = dict(protocol=PROTOCOL, source_binding=binding, source_tree_sha256=freeze['source_tree_sha256'],
                deployment=bind(a.deployment), arm=a.arm, route=ARMS[a.arm],policy='closure',old_readers='none',
                tiny_inputs=resource['tiny_inputs'],
                python_runtime=python_runtime,
                gpu_uuids=a.gpu_uuids.split(','), cpu_affinity=sorted(affinity), environment=environment,
                quality=QUALITY, commits=COMMITS, queries=COMMITS+1,
                old_reader_limitation='Weighted input-view lease is inapplicable to this21-unweighted-PR-only trace')
            record.update(configuration=config, configuration_sha256=digest(config),inputs=bind(a.inputs),qualification={})
            for phase,path in (('tiny',a.tiny_proof),('target',a.target_proof)):
                if phase=='tiny' and a.phase in ('target','formal') or phase=='target' and a.phase=='formal':
                    record['qualification'][phase] = validate_proof(path,phase,config,bind(a.inputs),a.dataset,contract)
            manifests={a.dataset+'@0':row['manifest']['path'],row['projections']['bfs']['identity']:row['projections']['bfs']['manifest']['path']}
            def session():
                return Session.native(manifests,devices=list(range(a.partitions)),cpp=cpp,
                    route=ARMS[a.arm],policy='closure',budgets=None,retirement_policy='lineage',
                    inventory_mode='recheck',batch_devices=True)
            record['gpu_executed']=True
            held,before_close,measured=run_workflow(a,row,session,worker.packet_for)
            record.update(measured)
            checks,oracle=audit_outputs(a,row,held,before_close,contract)
            record.update(checks=checks,independent_reference=oracle)
            record['quality_pass']=measured['execution_complete'] and len(checks)==21 and all(x['quality']['passed'] for x in checks)
            need(record['quality_pass'],'Negative control execution/quality failed')
            record.update(status='PASS_UNWEIGHTED_PR_FORMAL' if a.phase=='formal' else 'PASS_UNWEIGHTED_PR_QUALIFICATION',performance_eligible=a.phase=='formal')
    except BaseException as error:
        record.update(status='FAIL_UNWEIGHTED_PR_CONTROL',error=type(error).__name__+': '+str(error),performance_eligible=False)
        raise
    finally:
        protocol.save(a.output/'receipt.json',worker.strict(record))


if __name__ == '__main__':
    execute(parser().parse_args())
