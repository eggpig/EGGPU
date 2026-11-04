"""CPU checks: public arena precision reaches execution and result provenance."""
from types import ModuleType, SimpleNamespace
from contextlib import nullcontext
import sys
import numpy as np
import pytest

from easygraph.gpu import Session
from easygraph.gpu._bridge import ManagedSession
from easygraph.gpu._arena.trace_backends import EGGPUAdapter


@pytest.mark.parametrize('options', [
    {'pr_tolerance_l1': 0}, {'pr_tolerance_l1': -1},
    {'pr_tolerance_l1': float('nan')}, {'pr_tolerance_l1': float('inf')},
    {'pr_tolerance_l1': True}, {'pr_max_iter': 0},
    {'pr_max_iter': -1}, {'pr_max_iter': 1.5}, {'pr_max_iter': True},
])
def test_invalid_options_rejected_before_gpu_import(options):
    with pytest.raises(ValueError):
        ManagedSession({}, budget_bytes=1024, **options)


@pytest.mark.parametrize('explicit', [False, True])
def test_adapter_uses_actual_requested_precision(explicit):
    adapter = EGGPUAdapter.__new__(EGGPUAdapter)
    adapter.nodes = {'g': np.arange(4)}
    calls = []
    engine = SimpleNamespace(executor=SimpleNamespace(pr=None), last={})
    def query(*args):
        calls.append((args, engine.executor.pr))
        return np.full(4, .25)
    engine.query = query
    adapter.session = engine
    if explicit:
        adapter.pr_max_iter = 1000
        adapter.pr_tolerance_l1 = 1e-9
    adapter.query('g', 'weighted_pr', {})
    assert calls[0][0] == ('g', 'pr', 'message_count', 0)
    assert calls[0][1] == (.85, 1000 if explicit else 10000,
                            (1e-9 if explicit else 1e-6) / 4)


@pytest.mark.parametrize('algorithm', ['pr', 'weighted_pr'])
def test_result_contract_records_actual_precision(algorithm):
    managed = ManagedSession.__new__(ManagedSession)
    managed.closed = False
    managed.nodes = {'g': np.arange(4)}
    managed.pr_tolerance_l1, managed.pr_max_iter = 1e-9, 1000
    managed.backend = SimpleNamespace(last={}, query=lambda *args: np.full(4, .25))
    managed._own_result = lambda *args, **kwargs: kwargs['contract']
    contract = managed._query('g', algorithm, {})
    assert contract.max_iterations == 1000
    assert contract.tolerance_per_vertex == 1e-9 / 4
    assert contract.damping == .85
    assert contract.weight_projection == ('message_count' if algorithm == 'weighted_pr' else 'unweighted')


def test_public_factory_forwards_numerical_options(monkeypatch):
    calls = []
    def backend(graphs, **kwargs):
        calls.append((graphs, kwargs))
        return SimpleNamespace(session_id='test-only')
    cp = ModuleType('cupy')
    cp.cuda = SimpleNamespace(Device=lambda device: nullcontext())
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    monkeypatch.setattr('easygraph.gpu._core.CoreManagedSession', backend)
    result = Session.arena({'g': 'fixture'}, budget_bytes=1024,
                           pr_tolerance_l1=1e-9, pr_max_iter=1000)
    assert result.kind == 'arena'
    assert calls == [({'g': 'fixture'}, {'budget_bytes': 1024,
                      'pr_tolerance_l1': 1e-9, 'pr_max_iter': 1000})]
