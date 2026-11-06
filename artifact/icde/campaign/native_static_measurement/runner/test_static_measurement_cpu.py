"""CPU checks for the new timing boundary and preserved public result contract."""
import ast
import copy
import inspect
import json
from types import SimpleNamespace
import numpy as np
import pytest
import native_worker as worker


@pytest.mark.parametrize('function', ['bfs', 'sssp', 'wcc', 'pr'])
@pytest.mark.parametrize('count', [1, 2, 4, 8])
def test_query_returns_original_owned_output_before_runner_audit(function, count, monkeypatch):
    backend = worker.NativeBackend.__new__(worker.NativeBackend)
    backend.algorithm, backend.source, backend.count = function, 0, count
    backend.sssp_encoding, backend.last_metrics = 'u8_exact_u32_exact', {}
    metrics = copy.deepcopy(json.loads(worker.native_profile.REFERENCE.read_text())['records'][f'{function}_p{count}']['observed_example'])
    metrics.update(devices_used=list(range(count)), static_sssp_delta_rule='certified_bound_plus_one')
    values = np.array([0., 1., np.inf], dtype=np.float64)
    values.flags.writeable = False
    result = SimpleNamespace(values=values, backend_metrics=metrics,
        contract=SimpleNamespace(max_iterations=1000, damping=.85, tolerance_per_vertex=worker.shared.PR_TOLERANCE / len(values)))
    backend.session = SimpleNamespace(query=lambda *args: result)
    original_observe, original_json = worker.native_profile.observe, worker.json_metric_values
    def unexpected(*args, **kwargs):
        raise AssertionError('Runner audit entered the timed query')
    monkeypatch.setattr(worker.native_profile, 'observe', unexpected)
    monkeypatch.setattr(worker, 'json_metric_values', unexpected)
    assert backend.query() is values
    assert values.dtype == np.float64 and values.flags.owndata and not values.flags.writeable
    monkeypatch.setattr(worker.native_profile, 'observe', original_observe)
    monkeypatch.setattr(worker, 'json_metric_values', original_json)
    audited = backend.audit_last_query()
    assert audited['frozen_native_profile_observation']['actual_native_route_checks_passed']
    metrics['devices_used'] = []
    with pytest.raises(ValueError, match='devices_used'):
        backend.audit_last_query()


def test_uniform_factory_and_timer_order():
    init = ast.parse(inspect.getsource(worker))
    backend = next(n for n in init.body if isinstance(n, ast.ClassDef) and n.name == 'NativeBackend')
    factories = [n.func.attr for n in ast.walk(backend) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == 'Session']
    assert factories == ['native_static']
    source = inspect.getsource(worker.source_child)
    assert source.index('outputs[label] = backend.query()') < source.index('seconds = time.perf_counter() - started') < source.index('backend.audit_last_query()')
    assert 'shared.audit_output(values, expected, graph, args.algorithm)' in source
    assert 'np.shares_memory' in source and 'backend.close(); backend = None' in source
