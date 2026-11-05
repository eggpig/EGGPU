"""Real protocol/cache code with a CPU Session double; never CUDA execution."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import os
import sys

import numpy as np
import pytest

import state_protocol as protocol
import state_contract as contract
import state_inputs
import state_reference as reference
import state_worker as worker


SOURCE = Path(os.environ.get('EGGPU_TEST_SOURCE', str(
    Path(__file__).resolve().parents[1] / 'candidates/cold_import_v1/Easy-Graph')))


def args(tmp_path, **changes):
    base = dict(action='run', phase='tiny', repeat=0, experiment='E3', arm='D1C1',
        old_readers='per_update', order='interleaved', beta=.25, tiny_proof=None, target_proof=None,
        output=tmp_path/'run', output_store=tmp_path/'outputs', reference_cache=tmp_path/'references',
        reference_cache_gib=.1, gpu_uuids='GPU-test', datasets='TINY_A')
    return SimpleNamespace(**(base | changes))


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp('state-v4')
    out = directory / 'inputs'
    state_inputs.prepare(SimpleNamespace(tiny=True, source=SOURCE, output=out,
        rho=.1, locality='uniform', partitions=1, datasets='TINY_A'))
    return worker.load_inputs(out/'inputs.json')


class FakeLease:
    def __init__(self, session, snapshot):
        self.session, self.snapshot = session, snapshot
        self.ids = [snapshot + ':sssp', snapshot + ':pr']
        self.closed = False
        for key in self.ids:
            session.views[key]['active_leases'] += 1
        session.log.append(('acquire', snapshot))

    def read_weights(self):
        self.session.readbacks += 1
        if self.session.forbid_readback:
            raise AssertionError('Formal/target cannot read back old weights')
        row = self.session.row
        values = np.fromfile(row['canonical']['weights']['path'], dtype='<f8')
        version = int(self.snapshot.rsplit('@', 1)[1])
        positions = np.load(row['positions']['path'])
        values[positions] = 1 + (values[positions] - 1 + version) % 16
        result = []
        for key in self.ids:
            item = values.copy()
            if self.session.corrupt_old and self.session.commits:
                item[0] += 1
            item.flags.writeable = False
            result.append(dict(owner_id=key, algorithm='sssp' if key.endswith('sssp') else 'weighted_pr',
                               device_id=0, values=item))
        return {'per_view': result}

    def close(self):
        if not self.closed:
            self.session.log.append(('release', self.snapshot))
            for key in self.ids:
                self.session.views[key]['active_leases'] -= 1
            self.closed = True


class FakeSession:
    def __init__(self, row, route='lease_sparse', policy='closure', *, forbid_readback=False, corrupt_old=False):
        self.row, self.route, self.policy = row, route, policy
        self.views, self.log, self.commits = {}, [], []
        self._backend = self
        self.metadata = {}
        self.readbacks, self.closed = 0, False
        self.forbid_readback, self.corrupt_old = forbid_readback, corrupt_old

    def query(self, snapshot, function, params):
        self.log.append(('query', snapshot, function))
        version = 0 if function in ('bfs', 'wcc') else int(snapshot.rsplit('@', 1)[1])
        if function in ('sssp', 'weighted_pr'):
            self.metadata[snapshot] = {'id': snapshot, 'generation': version + 1}
            key = snapshot + (':sssp' if function == 'sssp' else ':pr')
            self.views.setdefault(key, dict(owner_id=key, kind='view', algorithm='sssp' if function == 'sssp' else 'pr',
                device_id=0, graph_id=snapshot, generation=version+1, active_leases=0))
        matrix = reference.matrix_for(self.row, version, function)
        values, _ = reference.build_reference(matrix, function, params.get('source', 0))
        values = values.astype(np.float64, copy=True)
        values.flags.writeable = False
        cow = int(self.route == 'lease_sparse' and any(x['active_leases'] for x in self.views.values()) and function in ('sssp', 'weighted_pr'))
        metrics = {'native_dynamic_patch': {'route': self.route, 'retirement_policy': 'lineage',
            'per_device_delta': {'0': {'cow_copies': cow, 'patch_packets': 0, 'lease_reuses': 0}}},
            'resource_management': {'state_switch_policy': self.policy, 'controller_version': 'state_switch_v2_closure_lru',
                'after_retention': {'per_device': [{'device_id': 0, 'managed_resident_bytes': 64}]}, 'actions': []}}
        return SimpleNamespace(values=values, seconds=0., backend_metrics=metrics)

    def commit_weights(self, base, snapshot, packet, storage):
        assert storage == 'native_memory'
        self.commits.append((base, snapshot, list(packet)))
        self.log.append(('commit', base, snapshot))
        return {'committed': snapshot}

    def inventory(self):
        return {'per_device': [{'device_id': 0, 'owners': copy.deepcopy(list(self.views.values())), 'allocations': []}]}

    def close(self):
        assert not any(row['active_leases'] for row in self.views.values())
        self.closed = True
        self.log.append(('close',))


@pytest.mark.parametrize('arm', ['D0C0','D1C0','D0C1','D1C1'])
@pytest.mark.parametrize('old_readers', ['none', 'per_update'])
def test_real_workflow_trace_old_reader_lifetime_and_full_outputs(inputs, tmp_path, arm, old_readers):
    a = args(tmp_path, arm=arm, old_readers=old_readers)
    s = FakeSession(inputs['datasets']['TINY_A'], *protocol.ARMS[arm])
    held, hashes, record = worker.run_workflow(a, inputs, ['TINY_A'], lambda: s,
                                              lambda state, name: FakeLease(state, name), None)
    assert record['execution_complete'] and len(record['events']) == 10 and len(held) == 8
    assert s.closed and record['continuous_interval'] and not record['sum_of_phases_used_as_total']
    assert record['old_reader_batches'] == (2 if old_readers == 'per_update' else 0)
    if old_readers == 'per_update':
        assert s.readbacks == 4
        for index, item in enumerate(s.log):
            if item[0] != 'acquire':
                continue
            assert s.log[index + 1][0] == 'commit'
            assert [entry[2] for entry in s.log[index + 2:index + 6]] == list(protocol.FUNCTIONS)
            assert s.log[index + 6][0] == 'release'
        assert all(x['released'] and x['old_weights_bitwise_unchanged'] for x in record['leases'])
    checks, _ = worker.audit_outputs(a, inputs, held, hashes)
    assert all(x['quality']['passed'] and x['survives_session_close'] for x in checks)
    assert len(checks) == 8
    for (_, values, _), check in zip(held, checks):
        np.testing.assert_array_equal(contract.check_output(check['output']), values)


def test_target_reader_does_not_add_qualification_readback(inputs, tmp_path):
    a = args(tmp_path, phase='target')
    s = FakeSession(inputs['datasets']['TINY_A'], forbid_readback=True)
    held, hashes, record = worker.run_workflow(a, inputs, ['TINY_A'], lambda: s,
                                              lambda state, name: FakeLease(state, name), None)
    assert record['execution_complete'] and len(held) == 80 and len(s.commits) == 20
    assert s.readbacks == 0 and len(record['leases']) == 20
    assert record['maintenance_counters']['cow_copies'] == 40
    assert not record['old_reader_readback_in_application']


def test_changed_old_native_view_fails_and_still_releases(inputs, tmp_path):
    a = args(tmp_path)
    s = FakeSession(inputs['datasets']['TINY_A'], corrupt_old=True)
    _, _, record = worker.run_workflow(a, inputs, ['TINY_A'], lambda: s,
                                      lambda state, name: FakeLease(state, name), None)
    assert not record['execution_complete'] and 'Readonly old FP64 view changed' in record['execution_error']
    assert s.closed and not any(x['active_leases'] for x in s.views.values())


@pytest.mark.parametrize('field,value', [('algorithm', 'sssp'), ('device_id', 1),
    ('graph_id', 'wrong-base'), ('generation', 99)])
def test_lease_checks_per_device_algorithm_and_current_base(inputs, field, value):
    s = FakeSession(inputs['datasets']['TINY_A'])
    s.query('TINY_A@0', 'sssp', {'source': 0})
    s.query('TINY_A@0', 'weighted_pr', {})
    s.views['TINY_A@0:pr'][field] = value
    with pytest.raises(ValueError):
        contract.OldReader(s, 'TINY_A@0', lambda state, name: FakeLease(state, name), 1, False)
    assert not any(x['active_leases'] for x in s.views.values())


def test_output_cas_deduplicates_bitwise_and_rejects_corruption(tmp_path):
    store = contract.OutputStore(tmp_path/'blobs')
    a = np.array([0., np.inf, -0.])
    first = store.put(a)
    second = store.put(a.copy())
    assert first['object_id'] == second['object_id'] and second['reused_existing_object']
    assert len(list((tmp_path/'blobs').rglob('*.npz'))) == 1
    different = a.copy(); different[2] = 0.
    assert store.put(different)['object_id'] != first['object_id']
    Path(first['blob']['path']).write_bytes(b'corrupt')
    with pytest.raises((ValueError, OSError)):
        store.put(a)


@pytest.mark.parametrize('experiment,names,queries,commits', [
    ('E3', ['IN'], 80, 20), ('E4', ['US', 'OR', 'IN', 'UK'], 128, 0),
    ('E5', ['IN', 'UK', 'OR'], 96, 24)])
def test_frozen_formal_counts(experiment, names, queries, commits):
    events = protocol.trace(experiment, names)
    assert sum(x['kind'] == 'query' for x in events) == queries
    assert sum(x['kind'] == 'commit' for x in events) == commits


@pytest.mark.parametrize('changes', [dict(repeat=10), dict(repeat=-1), dict(phase='target'),
    dict(experiment='E4'), dict(old_readers='busy_kernel'), dict(beta=1.01)])
def test_invalid_phases_or_reader_scope_rejected(inputs, tmp_path, changes):
    with pytest.raises(ValueError):
        contract.validate_arguments(args(tmp_path, **changes), inputs, ['TINY_A'])


def proof_fixture(tmp_path):
    a = args(tmp_path, old_readers='none')
    config = contract.configuration(a, {'sha': 'source'}, [0], {}, [0])
    source = tmp_path/'inputs.json'; source.write_text('{}')
    inp = protocol.bind(source)
    store = contract.OutputStore(tmp_path/'store')
    item = store.put(np.array([0., np.inf]))
    events = protocol.trace('E3', ['TINY_A'], qualification=True)
    checks = [{'event': event, 'quality': {'passed': True}, 'survives_session_close': True, 'output': item}
              for event in events if event['kind'] == 'query']
    proof = dict(schema='state.complete.workflow.v4', status='PASS_STATE_GPU_QUALIFICATION', phase='tiny',
        configuration=config, configuration_sha256=contract.digest(config), inputs=inp, datasets=['TINY_A'],
        events=events, checks=checks, execution_complete=True, quality_pass=True, budget=None)
    path = tmp_path/'proof.json'; protocol.save(path, proof)
    return a, config, inp, proof, path


def test_proof_accepts_exact_protocol_and_rechecks_full_output(tmp_path):
    _, config, inp, proof, path = proof_fixture(tmp_path)
    assert contract.validate_proof(path, 'tiny', config, inp, ['IN'], None) == protocol.bind(path)
    Path(proof['checks'][0]['output']['blob']['path']).write_bytes(b'bad')
    with pytest.raises(ValueError):
        contract.validate_proof(path, 'tiny', config, inp, ['IN'], None)


@pytest.mark.parametrize('key,value', [('arm', 'D0C0'), ('old_readers', 'per_update'), ('order', 'grouped'),
    ('beta', .5), ('experiment', 'E5'), ('source_binding', {'sha': 'changed'})])
def test_proof_rejects_cross_configuration(tmp_path, key, value):
    _, config, inp, _, path = proof_fixture(tmp_path)
    changed = config | {key: value}
    with pytest.raises(ValueError):
        contract.validate_proof(path, 'tiny', changed, inp, ['IN'], None)


def test_target_proof_requires_exact_input_and_budget(tmp_path):
    _, config, inp, proof, path = proof_fixture(tmp_path)
    proof['phase'] = 'target'; proof['datasets'] = ['IN']; proof['budget'] = {'per_device': {'0': 1}}
    events = protocol.trace('E3', ['IN'])
    item = proof['checks'][0]['output']
    proof['events'] = events
    proof['checks'] = [{'event': e, 'quality': {'passed': True}, 'survives_session_close': True, 'output': item}
                       for e in events if e['kind'] == 'query']
    protocol.save(path, proof)
    contract.validate_proof(path, 'target', config, inp, ['IN'], proof['budget'])
    for target, budget in [(dict(inp, sha256='bad'), proof['budget']), (inp, {'per_device': {'0': 2}})]:
        with pytest.raises(ValueError):
            contract.validate_proof(path, 'target', config, target, ['IN'], budget)


def test_binding_covers_snapshot_session_native_and_benchmark(tmp_path):
    dso = tmp_path/'native.so'; dso.write_bytes(b'cpu-fixture-not-an-extension')
    result = contract.source_binding(SOURCE, SimpleNamespace(__file__=str(dso)))
    assert result['dso'] == protocol.bind(dso)
    for name in ('easygraph/gpu/_snapshot_lease.py', 'easygraph/gpu/session.py',
                 'gpu_easygraph/runtime/partitioned_sssp_state.cpp',
                 'gpu_easygraph/runtime/partitioned_device_state.cpp'):
        assert name in result['files']
    assert set(result['benchmark']) == set(contract.BENCHMARK_FILES)


def test_formal_requires_both_exact_proofs(tmp_path):
    real = {'qualification_only': False, 'datasets': {'IN': {}}}
    with pytest.raises(ValueError, match='tiny'):
        contract.validate_arguments(args(tmp_path, phase='formal'), real, ['IN'])
    with pytest.raises(ValueError, match='target'):
        contract.validate_arguments(args(tmp_path, phase='formal', tiny_proof=Path('tiny')), real, ['IN'])
    for repeat in range(10):
        contract.validate_arguments(args(tmp_path, phase='formal', repeat=repeat,
                                    tiny_proof=Path('tiny'), target_proof=Path('target')), real, ['IN'])
