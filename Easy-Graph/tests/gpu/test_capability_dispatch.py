"""CPU qualification of dispatch, complete WCC output, and source validation."""
from contextlib import nullcontext
from dataclasses import asdict
from threading import RLock
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from easygraph.gpu import Session, plan_session, StaleRequestError
from easygraph.gpu._bridge import ManagedSession
from easygraph.gpu._core import CoreManagedSession, canonical_wcc
from easygraph.gpu.contracts import GraphDescriptor, GraphVersion, NodeDomain


@pytest.mark.parametrize('devices,state_model,expected', [
    ((0,), 'static', 'arena'), ((3,), 'journal', 'arena'),
    ((0,), 'snapshots', 'native'), ((0, 1), 'static', 'native'),
    ((2, 5, 6, 7), 'snapshots', 'native'),
])
def test_plan_uses_resources_and_state_contract(devices, state_model, expected):
    plan = plan_session(devices=devices, state_model=state_model)
    assert plan.backend == expected
    assert plan.devices == devices
    assert plan.to_dict() == asdict(plan)
    assert plan.input_kind == ('owned_coo' if expected == 'arena' else 'partition_manifest')


@pytest.mark.parametrize('devices', [(), (0, 0), (-1,), (True,), (1.5,), (np.bool_(True),)])
def test_invalid_devices_rejected_without_gpu(devices):
    with pytest.raises(ValueError):
        plan_session(devices=devices)


def test_unsupported_requirement_never_silently_changes_semantics():
    with pytest.raises(NotImplementedError):
        plan_session(devices=(0, 1), state_model='journal')
    with pytest.raises(NotImplementedError):
        plan_session(algorithms=('sssp', 'ppr'))
    with pytest.raises(ValueError):
        plan_session(state_model='automatic')
    with pytest.raises(ValueError):
        plan_session(algorithms='bfs')


def test_factory_records_plan_and_does_not_retry_another_backend(monkeypatch):
    calls = []
    def arena(cls, inputs, **options):
        calls.append(('arena', inputs, options))
        return SimpleNamespace(dispatch_plan=None)
    def native(cls, inputs, **options):
        calls.append(('native', inputs, options))
        return SimpleNamespace(dispatch_plan=None)
    monkeypatch.setattr(Session, 'arena', classmethod(arena))
    monkeypatch.setattr(Session, 'native', classmethod(native))
    first = Session.create({'graph-with-arbitrary-name': 'coo'}, devices=(4,), budget_bytes=4096)
    assert first.dispatch_plan.backend == 'arena'
    second = Session.create({'g': 'manifest'}, devices=(0,), state_model='snapshots', budgets={0: 4096})
    assert second.dispatch_plan.backend == 'native'
    assert calls[0][2] == dict(budget_bytes=4096, device=4)
    assert calls[1][2] == dict(devices=(0,), budgets={0: 4096})
    def fail(*args, **kwargs):
        raise MemoryError('admission failure')
    monkeypatch.setattr(Session, 'arena', classmethod(fail))
    with pytest.raises(MemoryError, match='admission failure'):
        Session.create({'g': 'coo'}, budget_bytes=4096)
    assert len(calls) == 2


def test_budget_meanings_are_not_silently_converted():
    with pytest.raises(ValueError, match='explicit budget_bytes'):
        Session.create({})
    with pytest.raises(ValueError, match='per-device budgets'):
        Session.create({}, devices=(0, 1), budget_bytes=4096)


@pytest.mark.parametrize('raw,expected', [
    ([0, 0, 2, 2, 4], [0, 0, 2, 2, 4]),
    ([3, 3, 4, 3, 4], [0, 0, 2, 0, 2]),
    ([1, 0], [0, 1]),
    ([], []),
])
def test_wcc_certificate_preserves_general_canonicalization(raw, expected):
    values = np.asarray(raw, dtype=np.float64)
    certified = canonical_wcc(values, len(values), certify_minimum=True)
    ordinary = canonical_wcc(values, len(values), certify_minimum=False)
    np.testing.assert_array_equal(certified, expected)
    np.testing.assert_array_equal(certified, ordinary)
    assert certified.flags.owndata
    assert not np.shares_memory(certified, values)


def test_canonical_roots_and_fallback_preserve_minima_without_global_sorting(monkeypatch):
    def unexpected_unique(*args, **kwargs):
        raise AssertionError('Dense-index labels do not need global re-encoding')
    monkeypatch.setattr(np, 'unique', unexpected_unique)
    np.testing.assert_array_equal(
        canonical_wcc(np.array([0., 0., 2., 2.]), 4, certify_minimum=True),
        [0., 0., 2., 2.])
    np.testing.assert_array_equal(
        canonical_wcc(np.array([1., 1., 3., 3.]), 4, certify_minimum=True),
        [0., 0., 2., 2.])


@pytest.mark.parametrize('labels', [[0., np.nan], [0., -1.], [0., 2.], [0., .5]])
def test_invalid_wcc_output_is_not_certified(labels):
    with pytest.raises(ValueError):
        canonical_wcc(np.array(labels), 2, certify_minimum=True)


def _fake_managed(cls, nodes, values):
    backend = cls.__new__(cls)
    backend.closed = False
    backend.nodes = {'g': np.asarray(nodes, dtype=np.int64)}
    backend._lock = RLock()
    backend.pr_tolerance_l1, backend.pr_max_iter = 1e-6, 2000
    backend._initialize_bridge('managed')
    backend._test_version = 0
    backend.backend = SimpleNamespace(last={}, query=lambda *args: values)
    backend.describe = lambda name: GraphDescriptor(
        name, 'test/g', 'managed',
        GraphVersion(0, backend._test_version, 0, 0, 'cpu-test'),
        NodeDomain.from_nodes(backend.nodes[name]), True, True, {})
    return backend


def test_wcc_public_output_remains_independent_across_queries_and_updates():
    raw = np.array([0., 0., 2., 2.])
    backend = _fake_managed(CoreManagedSession, [10, 20, 30, 40], raw)
    backend.capability_descriptor = lambda: SimpleNamespace(algorithms=('bfs', 'sssp', 'pr', 'wcc'))
    session = Session(backend, kind='cpu-test')
    request = session.request('g', 'wcc')
    first, second = session.execute(request), session.execute(request)
    assert first.values.flags.owndata and not first.values.flags.writeable
    assert not np.shares_memory(first.values, second.values)
    assert not np.shares_memory(first.values, raw)
    raw[:] = 0
    np.testing.assert_array_equal(first.values, [0, 0, 2, 2])
    backend._test_version += 1
    with pytest.raises(StaleRequestError):
        session.execute(request)
    assert first.backend_metrics['wcc_public_output']['bridge_copy_bytes'] == 0


@pytest.mark.parametrize('source,valid', [(-5, True), (50, True), (7, True),
                                         (-6, False), (51, False), (8, False),
                                         (1 << 80, False)])
def test_source_lookup_preserves_original_sparse_labels(source, valid):
    backend = _fake_managed(ManagedSession, [-5, 7, 50], np.array([0., 1., 2.]))
    if valid:
        result = backend.query('g', 'bfs', {'source': source})
        assert len(result.values) == 3
    else:
        with pytest.raises(ValueError, match='Source absent'):
            backend.query('g', 'bfs', {'source': source})


def test_public_arena_factory_exposes_core_implementation(monkeypatch):
    calls = []
    cp = ModuleType('cupy')
    cp.cuda = SimpleNamespace(Device=lambda d: nullcontext())
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    def backend(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(session_id='cpu-test')
    monkeypatch.setattr('easygraph.gpu._core.CoreManagedSession', backend)
    session = Session.arena({}, budget_bytes=4096, device=2)
    assert session.kind == 'arena' and session._device == 2
    assert len(calls) == 1
