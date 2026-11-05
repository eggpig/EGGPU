"""Real Session.native_static facade with CPU-only native ABI doubles.

These tests cover actual constructor/validation/query/ownership/control paths;
they do not claim native GPU kernel execution or compact performance.
"""
from pathlib import Path
import hashlib
import heapq
import importlib.util
import json
import numpy as np
import pytest

from easygraph.gpu import Session
from easygraph.gpu._native import verify_manifest


class NativeDouble:
    def __init__(self):
        self.running = False
        self.starts = 0
        self.devices = []
        self.policies = []
        self.deltas = []
        self.corrupt_observation = False

    def cpp_egmgpu_runtime_status(self):
        return {'running': self.running}

    def cpp_egmgpu_runtime_start(self, *, devices, backend):
        self.running = True
        self.starts += 1
        self.devices = list(devices)

    def cpp_egmgpu_runtime_shutdown(self, *, drain):
        self.running = False

    def cpp_egmgpu_clear_partitioned_state(self, *, devices):
        assert self.running
        return {'cleared': True, 'devices': list(devices)}

    def cpp_egmgpu_managed_state_inventory(self, *, devices):
        return dict(algorithms=['bfs', 'pr', 'sssp', 'wcc'], managed_resident_bytes=0,
                    per_device=[dict(device_id=d, owners=[], allocations=[], managed_resident_bytes=0,
                                     counters={'dynamic.abi_version': 1}) for d in devices])

    def cpp_egmgpu_managed_evict(self, **kwargs):
        raise AssertionError('No retained allocations in this CPU ABI double')

    def cpp_egmgpu_load_partition_manifest(self, path, *, import_policy):
        return Path(path)

    def cpp_egmgpu_partitioned_sssp(self, path, source, delta, limit, devices, *, dense_output, execution_policy):
        self.policies.append(dict(execution_policy))
        self.deltas.append(delta)
        doc = json.loads(path.read_text())
        n = doc['graph']['vertex_count']
        adjacency = [[] for _ in range(n)]
        maximum = 0
        noninteger = 0
        for part in doc['partitions']:
            arrays = {key: np.fromfile(path.parent / item['path'], dtype={'int32': '<i4', 'uint64': '<u8', 'float64': '<f8'}[item['dtype']])
                      for key, item in part['files'].items()}
            maximum = max(maximum, float(arrays['weights'].max(initial=0)))
            noninteger += int(np.count_nonzero(arrays['weights'] != np.floor(arrays['weights'])))
            for local in range(part['owned_vertex_count']):
                for edge in range(arrays['offsets'][local], arrays['offsets'][local + 1]):
                    adjacency[int(arrays['local_to_global'][local])].append((
                        int(arrays['local_to_global'][arrays['indices'][edge]]), arrays['weights'][edge]))
        result = np.full(n, np.inf)
        result[source] = 0.
        pending = [(0., source)]
        while pending:
            distance, vertex = heapq.heappop(pending)
            if distance != result[vertex]:
                continue
            for neighbor, weight in adjacency[vertex]:
                candidate = distance + weight
                if candidate < result[neighbor]:
                    result[neighbor] = candidate
                    heapq.heappush(pending, (candidate, neighbor))
        compact = execution_policy['weight_encoding'] == 'u8_exact'
        all_light = maximum <= delta
        route = 'requested_all_light' if all_light else 'requested_exact_integer_delta16'
        reason = 'current_maximum_weight_at_most_requested_delta' if all_light else 'integer_simple_path_bound_below_2p53_with_exact_binary_bucket_scaling'
        return dict(values_dense=result, devices_used=list(devices), converged=True,
            delta=delta, execution_policy_version='egmgpu.sssp-execution-policy/v1',
            execution_policy_source='explicit', execution_policy=dict(execution_policy),
            output_representation='owning_fp64_dense', host_assembly_policy='worker',
            sssp_native_output_ownership=dict(version='native_sssp_host_ownership_v1', requested='0',
                eligible=True, used=False, reason='disabled', producer='capsule_fp64_worker_scatter',
                complete_vertices=n, values_bytes=n * 8),
            weight_encoding='f64' if self.corrupt_observation else execution_policy['weight_encoding'],
            wire_encoding=execution_policy['wire_encoding'], wire_integer_bound_certified=compact,
            wire_maximum_weight=maximum, wire_distance_bound=(n - 1) * maximum,
            numeric_domain_version='nonnegative_fp64_all_light_fallback_v1',
            numeric_statistics_available=True, numeric_vertex_count=n,
            numeric_maximum_weight=maximum, numeric_noninteger_weight_count=noninteger,
            requested_delta=delta, effective_delta=delta, numeric_delta_changed=False,
            numeric_domain_route=route, numeric_domain_reason=reason,
            numeric_all_edges_light=all_light, numeric_integer_bound_used=not all_light,
            execution_policy_effective=dict(requested_delta=delta, effective_delta=delta,
                requested_delta_applied=True, numeric_domain_route=route),
            single_owner_execution_used=False, bucket_rounds=1)


@pytest.fixture(autouse=True)
def controlled_environment(monkeypatch):
    for name in ('EGGPU_DYNAMIC_WEIGHT_ROUTE', 'EGGPU_SSSP_WEIGHT_CERTIFICATION', 'EGGPU_SINGLE_OWNER_SSSP'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('EGGPU_NATIVE_SSSP_OWNED_OUTPUT', '0')


def fixture_manifest(tmp_path, weights=(0., 2., 255.), *, count=1, n=16, weighted=True):
    script = Path(__file__).resolve().parents[2] / 'scripts/gpu/qualification_inputs.py'
    spec = importlib.util.spec_from_file_location('static_native_inputs', script)
    qi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qi)
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)[:len(weights)]
    if n == 1:
        edges = np.array([[0, 0]], dtype=np.int32)[:len(weights)]
    raw = qi.canonical(n, edges, np.asarray(weights, dtype=np.float64))
    result, _ = qi.manifest(raw, tmp_path / 'input', 'static-fixture', count, weighted=weighted)
    return Path(result['manifest'])


@pytest.mark.parametrize('count', [1, 2, 4, 8])
def test_real_public_facade_exact_policy_and_independent_outputs(tmp_path, count, monkeypatch):
    monkeypatch.setenv('EGGPU_SINGLE_OWNER_SSSP', '1')
    path = fixture_manifest(tmp_path, count=count)
    cpp = NativeDouble()
    session = Session.native_static({'g': path}, devices=tuple(range(count)), cpp=cpp)
    try:
        first = session.query('g', 'sssp', {'source': 0})
        other = session.query('g', 'sssp', {'source': 2})
        np.testing.assert_array_equal(first.values[:4], [0., 0., 2., 257.])
        assert np.isposinf(first.values[4:]).all()
        assert not np.shares_memory(first.values, other.values)
        assert first.values.dtype == np.float64 and first.values.flags.owndata and not first.values.flags.writeable
        assert all(p['weight_encoding'] == 'u8_exact' and p['wire_encoding'] == 'u32_exact'
                   and p['single_owner_execution'] == 'off' for p in cpp.policies)
        cert = first.backend_metrics['static_sssp_input_certificate']
        assert cert['vertex_count'] == 16 and cert['weights_count'] == 3 and cert['finite_distance_upper_bound'] == 15 * 255
        assert session.capabilities().mutations.startswith('none')
    finally:
        session.close()
    np.testing.assert_array_equal(first.values[:4], [0., 0., 2., 257.])
    assert cpp.starts == 1 and not cpp.running


@pytest.mark.parametrize('weight', [.25, 256., -1., np.nan, np.inf])
def test_invalid_original_weights_rejected_before_runtime_start(tmp_path, weight):
    path = fixture_manifest(tmp_path, (1.,))
    doc = json.loads(path.read_text())
    item = doc['partitions'][0]['files']['weights']
    payload = path.parent / item['path']
    np.asarray([weight], dtype='<f8').tofile(payload)
    item['sha256'] = hashlib.sha256(payload.read_bytes()).hexdigest()
    path.write_text(json.dumps(doc))
    cpp = NativeDouble()
    with pytest.raises(ValueError):
        Session.native_static({'g': path}, cpp=cpp)
    assert cpp.starts == 0 and not cpp.running


def test_n1_zero_edges_zero_bound(tmp_path):
    path = fixture_manifest(tmp_path, (), n=1)
    cpp = NativeDouble()
    session = Session.native_static({'g': path}, cpp=cpp)
    try:
        result = session.query('g', 'sssp', {'source': 0})
        np.testing.assert_array_equal(result.values, [0.])
        assert result.backend_metrics['static_sssp_input_certificate']['finite_distance_upper_bound'] == 0
    finally:
        session.close()


@pytest.mark.parametrize('count', [1, 2, 4, 8])
def test_certified_bound_plus_one_uses_current_graph_and_fixed_policy(tmp_path, count):
    first = fixture_manifest(tmp_path / 'a', (0., 2., 255.), count=count)
    second = fixture_manifest(tmp_path / 'b', (1., 2., 3.), count=count)
    cpp = NativeDouble()
    session = Session.native_static({'a': first, 'b': second}, devices=tuple(range(count)), cpp=cpp,
                                   sssp_delta_rule='certified_bound_plus_one')
    try:
        outputs = [session.query(name, 'sssp', {'source': 0}) for name in ('a', 'b', 'a')]
        assert cpp.deltas == [3826., 46., 3826.]
        for result, delta in zip(outputs, (3826., 46., 3826.)):
            m = result.backend_metrics
            assert m['static_sssp_delta_rule'] == 'certified_bound_plus_one'
            assert m['static_sssp_requested_delta'] == delta and m['bucket_rounds'] == 1
        assert all(p['single_owner_execution'] == 'off' for p in cpp.policies)
        assert not np.shares_memory(outputs[0].values, outputs[2].values)
    finally:
        session.close()


def test_n1_zero_edges_certified_bound_plus_one(tmp_path):
    cpp = NativeDouble()
    session = Session.native_static({'g': fixture_manifest(tmp_path, (), n=1)}, cpp=cpp,
                                   sssp_delta_rule='certified_bound_plus_one')
    try:
        result = session.query('g', 'sssp', {'source': 0})
        assert result.values.tolist() == [0.] and cpp.deltas[-1] == 1.
    finally:
        session.close()


@pytest.mark.parametrize('options', [dict(sssp_delta_rule='unknown'),
    dict(sssp_encoding='f64', sssp_delta_rule='certified_bound_plus_one')])
def test_bad_delta_rule_or_f64_combination_rejected_before_start(tmp_path, options):
    cpp = NativeDouble()
    with pytest.raises(ValueError):
        Session.native_static({'g': fixture_manifest(tmp_path)}, cpp=cpp, **options)
    assert cpp.starts == 0


def test_n1_positive_weight_rejects_bound_rule_before_start(tmp_path):
    path = fixture_manifest(tmp_path, (2.,), n=1)
    cpp = NativeDouble()
    with pytest.raises(ValueError, match='B\\+1 >= maxW'):
        Session.native_static({'g': path}, cpp=cpp, sssp_delta_rule='certified_bound_plus_one')
    assert cpp.starts == 0
    session = Session.native_static({'g': path}, cpp=cpp, sssp_delta_rule='fixed16')
    try:
        assert session.query('g', 'sssp', {'source': 0}).values.tolist() == [0.]
    finally:
        session.close()


@pytest.mark.parametrize('field,bad', [('requested_delta',16.), ('effective_delta',16.),
    ('bucket_rounds',2), ('numeric_delta_changed',True), ('numeric_all_edges_light',False)])
def test_wrong_actual_bound_schedule_is_rejected(tmp_path, field, bad):
    cpp = NativeDouble()
    original = cpp.cpp_egmgpu_partitioned_sssp
    def wrong(*args, **kwargs):
        result = original(*args, **kwargs)
        result[field] = bad
        return result
    cpp.cpp_egmgpu_partitioned_sssp = wrong
    session = Session.native_static({'g': fixture_manifest(tmp_path)}, cpp=cpp,
                                   sssp_delta_rule='certified_bound_plus_one')
    try:
        with pytest.raises((RuntimeError, ValueError)):
            session.query('g', 'sssp', {'source': 0})
    finally:
        session.close()


@pytest.mark.parametrize('n,valid', [(16843009, True), (16843010, False)])
def test_uint32_strict_global_bound_before_runtime(tmp_path, n, valid):
    path = fixture_manifest(tmp_path, (255.,))
    doc = json.loads(path.read_text())
    doc['graph']['vertex_count'] = n
    doc['partitions'][0]['owned_vertex_count'] = n
    path.write_text(json.dumps(doc))
    if valid:
        cert = {}
        verify_manifest(path, doc, numeric_certificate=cert)
        assert cert['finite_distance_upper_bound'] < 2**32 - 1
    else:
        cpp = NativeDouble()
        with pytest.raises(ValueError, match='sentinel'):
            Session.native_static({'g': path}, cpp=cpp)
        assert cpp.starts == 0


@pytest.mark.parametrize('mode', ['dtype', 'missing_count', 'byte_size'])
def test_malformed_empty_metadata_is_not_accepted(tmp_path, mode):
    path = fixture_manifest(tmp_path, (), n=1)
    doc = json.loads(path.read_text())
    weights = doc['partitions'][0]['files']['weights']
    if mode == 'dtype':
        weights['dtype'] = 'int32'
    elif mode == 'missing_count':
        doc['partitions'][0]['adjacency_count'] = 1
    else:
        weights['byte_size'] = 8
    path.write_text(json.dumps(doc))
    cpp = NativeDouble()
    with pytest.raises(ValueError):
        Session.native_static({'g': path}, cpp=cpp)
    assert cpp.starts == 0


def test_default_verifier_identity_unchanged_for_nonempty_inputs(tmp_path):
    path = fixture_manifest(tmp_path)
    doc = json.loads(path.read_text())
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_manifest(path, doc) == expected
    assert verify_manifest(path, doc, numeric_certificate={}) == expected


def test_unweighted_manifest_does_not_become_implicit_sssp_weights(tmp_path):
    path = fixture_manifest(tmp_path, weighted=False)
    cpp = NativeDouble()
    session = Session.native_static({'g': path}, cpp=cpp)
    try:
        with pytest.raises(ValueError, match='certified weighted manifest'):
            session.query('g', 'sssp', {'source': 0})
        assert cpp.policies == []
    finally:
        session.close()


def test_general_f64_remains_explicit_static_option(tmp_path):
    path = fixture_manifest(tmp_path, (.25,))
    cpp = NativeDouble()
    session = Session.native_static({'g': path}, cpp=cpp, sssp_encoding='f64')
    try:
        result = session.query('g', 'sssp', {'source': 0})
        assert result.values[1] == .25 and cpp.policies[-1]['weight_encoding'] == 'f64'
    finally:
        session.close()


def test_original_dynamic_factory_keeps_f64_policy(tmp_path):
    path = fixture_manifest(tmp_path, (.25,))
    cpp = NativeDouble()
    session = Session.native({'g': path}, cpp=cpp)
    try:
        assert session._backend.sssp_policy.weight_encoding == 'f64'
        assert session._backend.sssp_policy.wire_encoding == 'f64'
        assert session.capabilities().mutations.startswith('fixed-topology')
    finally:
        session.close()


def test_all_public_dynamic_operations_reject_but_outputs_survive(tmp_path):
    path = fixture_manifest(tmp_path)
    session = Session.native_static({'g': path}, cpp=NativeDouble())
    try:
        result = session.query('g', 'sssp', {'source': 0})
        calls = [lambda: session.commit_weights('g', 'new', []),
                 lambda: session.register_snapshot('new', path),
                 lambda: session.retain_snapshot('g'), lambda: session.release_snapshot('g'),
                 lambda: session._backend.acquire_readonly_view('g')]
        for call in calls:
            with pytest.raises(NotImplementedError, match='Static native'):
                call()
    finally:
        session.close()
    assert result.values[3] == 257.


def test_unrequested_native_encoding_observation_rejected(tmp_path):
    path = fixture_manifest(tmp_path)
    cpp = NativeDouble()
    session = Session.native_static({'g': path}, cpp=cpp)
    cpp.corrupt_observation = True
    try:
        with pytest.raises(RuntimeError, match='selected static SSSP encoding'):
            session.query('g', 'sssp', {'source': 0})
    finally:
        session.close()


@pytest.mark.parametrize('name,value', [('EGGPU_DYNAMIC_WEIGHT_ROUTE', 'lease_sparse'),
                                      ('EGGPU_SSSP_WEIGHT_CERTIFICATION', 'native_delta')])
def test_dynamic_environment_rejected_before_runtime(tmp_path, monkeypatch, name, value):
    path = fixture_manifest(tmp_path)
    monkeypatch.setenv(name, value)
    cpp = NativeDouble()
    with pytest.raises(ValueError, match='Static native input'):
        Session.native_static({'g': path}, cpp=cpp)
    assert cpp.starts == 0
