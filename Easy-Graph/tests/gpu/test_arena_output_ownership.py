"""CPU contracts for real download code and the public bridge ownership path."""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import gc
from pathlib import Path
from threading import RLock
from time import perf_counter
import time
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from easygraph.gpu._arena_output import _download_request, _register_download, _consume_download
from easygraph.gpu._bridge import ManagedSession, owning_result
from easygraph.gpu._core import CoreManagedSession, canonical_wcc
from easygraph.gpu.contracts import GraphDescriptor, GraphVersion


class Device:
    """CPU-only blocking device-copy double; no CuPy or CUDA initialization."""
    def __init__(self, values): self.values = values
    def __getitem__(self, item): return Device(self.values[item])
    def set(self, values): self.values[...] = values
    def get(self, out=None):
        if out is None: return self.values.copy()
        out[...] = self.values
        return out


def download(values, selected=None):
    # Execute the production method body, not a duplicate of its allocation hook.
    import easygraph.gpu._bridge as bridge
    source = Path(bridge.__file__).parent / '_arena/gpu_runtime.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Executor')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'download')
    namespace = {'np': np, 'time': time, 'cp': SimpleNamespace(cuda=SimpleNamespace(get_elapsed_time=lambda *a: 0.))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    event = SimpleNamespace(record=lambda: None)
    engine = SimpleNamespace(patch_capacity=2, patch_i=Device(np.empty(2, np.int32)),
                             patch_v=Device(np.empty(2, np.float64)), result_events=(event, event))
    def gather(grid, block, args):
        count, indices, device, target = args
        target.values[:count] = device.values[indices.values[:count]]
    engine.kernels = {'gather': gather}
    return namespace['download'](engine, Device(values), selected)[0]


def publish(values, algorithm='bfs', scope=None, graph='g'):
    return owning_result(graph, algorithm, values, perf_counter(), {}, _arena_scope=scope)


@pytest.mark.parametrize('algorithm', ['bfs', 'sssp', 'pr', 'weighted_pr'])
def test_real_complete_download_is_adopted_once(algorithm):
    device_values = np.array([0., 1., np.inf, -0.])
    with _download_request('g', algorithm) as scope:
        values = download(device_values)
        result = publish(values, algorithm, scope)
        replay = publish(values, algorithm, scope)
    assert result.values is values and replay.values is not values
    assert not result.values.flags.writeable and result.values.flags.owndata
    assert not np.shares_memory(result.values, replay.values)
    device_values[:] = 17.
    np.testing.assert_array_equal(result.values, [0., 1., np.inf, -0.])
    key = 'pr' if algorithm == 'weighted_pr' else algorithm
    metrics = result.backend_metrics[key + '_public_output_ownership']
    assert metrics['route'] == 'adopt_arena_download' and metrics['bridge_copy_bytes'] == 0
    assert metrics['bridge_copy_bytes_avoided'] == values.nbytes
    assert replay.backend_metrics[key + '_public_output_ownership']['bridge_copy_bytes'] == values.nbytes


@pytest.mark.parametrize('selected', [[], [3, 1, 3], [0, 1, 2, 3]])
def test_real_selected_download_owns_independent_full_allocation(selected):
    source = np.array([0., 2., 5., np.inf])
    with _download_request('g', 'sssp') as scope:
        values = download(source, selected)
        result = publish(values, 'sssp', scope)
    assert result.values is values and values.base is None
    np.testing.assert_array_equal(values, source[selected])


@pytest.mark.parametrize('algorithm', ['bfs', 'sssp', 'pr', 'weighted_pr'])
def test_caller_owned_array_and_metrics_cannot_certify(algorithm):
    values = np.arange(4, dtype=np.float64)
    with _download_request('g', algorithm) as scope:
        result = publish(values, algorithm, scope)
    assert result.values is not values
    values[:] = -8.
    np.testing.assert_array_equal(result.values, np.arange(4))
    assert publish(values, algorithm, object()).values is not values


@pytest.mark.parametrize('factory', [
    lambda: np.arange(4, dtype=np.float32),
    lambda: np.arange(4, dtype=np.int64),
    lambda: np.arange(8, dtype=np.float64)[::2],
    lambda: np.arange(4, dtype=np.float64).reshape(2, 2),
    lambda: np.arange(4, dtype=np.float64).view(),
    lambda: np.arange(4, dtype=np.float64).view(type('SubArray', (np.ndarray,), {})),
])
def test_non_exact_owned_float64_producer_falls_back(factory):
    values = factory()
    with _download_request('g', 'bfs') as scope:
        _register_download(values)
        result = publish(values, scope=scope)
    assert result.values is not values and result.values.dtype == np.float64
    assert not np.shares_memory(result.values, values)


def test_replaced_array_even_with_matching_values_falls_back():
    with _download_request('g', 'bfs') as scope:
        original = download(np.arange(4, dtype=np.float64))
        copied = original.copy()
        assert publish(copied, scope=scope).values is not copied
        assert publish(original, scope=scope).values is not original


def test_multiple_downloads_fail_closed():
    with _download_request('g', 'bfs') as scope:
        first = download(np.arange(4, dtype=np.float64))
        second = download(np.arange(4, dtype=np.float64))
        assert publish(first, scope=scope).values is not first
        assert publish(second, scope=scope).values is not second


def test_registry_is_weak_and_does_not_hold_result():
    with _download_request('g', 'bfs'):
        values = download(np.arange(4, dtype=np.float64))
        reference = weakref.ref(values)
        del values
        gc.collect()
        assert reference() is None


def test_bitwise_special_values_and_empty_output_are_preserved():
    bits = np.array([0, 1 << 63, 0x7ff0000000000000, 0x7ff8000000001234, 0x3ff0000000000001], dtype=np.uint64)
    for values in (bits.view(np.float64), np.empty(0, np.float64)):
        with _download_request('g', 'sssp') as scope:
            host = download(values)
            result = publish(host, 'sssp', scope)
        assert result.values is host
        np.testing.assert_array_equal(result.values.view(np.uint64), values.view(np.uint64))


def test_scope_cleanup_on_exception_and_stale_token():
    with pytest.raises(RuntimeError, match='after download'):
        with _download_request('g', 'bfs') as scope:
            values = download(np.arange(4, dtype=np.float64))
            raise RuntimeError('after download')
    assert publish(values, scope=scope).values is not values
    with _download_request('g', 'bfs') as new_scope:
        assert publish(values, scope=new_scope).values is not values


def test_nested_queries_and_graph_algorithm_binding():
    with _download_request('outer', 'bfs') as outer:
        values = download(np.arange(4, dtype=np.float64))
        with _download_request('inner', 'pr') as inner:
            nested = download(np.full(4, .25))
            assert publish(values, scope=outer, graph='outer').values is not values
            assert publish(nested, 'pr', inner, 'inner').values is nested
        assert publish(values, scope=outer, graph='wrong').values is not values
        assert publish(values, 'sssp', outer, 'outer').values is not values
        assert publish(values, scope=outer, graph='outer').values is values


def test_thread_context_cannot_replay_capability():
    with _download_request('g', 'bfs') as scope:
        values = download(np.arange(4, dtype=np.float64))
        context = copy_context()
        with ThreadPoolExecutor(1) as pool:
            result = pool.submit(context.run, publish, values, 'bfs', scope).result()
        assert result.values is not values
        assert publish(values, scope=scope).values is values


def test_independent_thread_scopes():
    def run(seed):
        with _download_request('g', 'bfs') as scope:
            values = download(np.arange(4, dtype=np.float64) + seed)
            return publish(values, scope=scope).values
    with ThreadPoolExecutor(2) as pool:
        first, second = list(pool.map(run, [0, 10]))
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, np.arange(4))
    np.testing.assert_array_equal(second, np.arange(4) + 10)


class HostSession(CoreManagedSession):
    def __init__(self, *, certified=True):
        self.nodes = {'g': np.arange(4, dtype=np.int64)}
        self._lock = RLock()
        self.closed = False
        self.pr_tolerance_l1, self.pr_max_iter = 1e-9, 1000
        self.sequence = 0
        def query(*args):
            self.sequence += 1
            values = np.arange(4, dtype=np.float64) + self.sequence
            self.backend.last = {'sequence': self.sequence}
            return download(values) if certified else values
        self.backend = SimpleNamespace(query=query, last={}, close=lambda: None)
        self._initialize_bridge('managed')

    def describe(self, name):
        return GraphDescriptor(name, self.session_id + '/' + name, 'managed',
                               GraphVersion(0, 0, 0, 0, 'CPU fixture'),
                               self._domains[name], True, True, {})


@pytest.mark.parametrize('algorithm', ['bfs', 'sssp', 'pr', 'weighted_pr'])
@pytest.mark.parametrize('certified', [False, True])
def test_real_managed_bridge_old_outputs_survive_later_queries_and_close(algorithm, certified):
    session = HostSession(certified=certified)
    parameters = {'source': 0} if algorithm in ('bfs', 'sssp') else {}
    first = session.query('g', algorithm, parameters)
    second = session.query('g', algorithm, parameters)
    assert len(session._live_outputs()) == 2
    assert not np.shares_memory(first.values, second.values)
    key = 'pr' if algorithm == 'weighted_pr' else algorithm
    assert first.backend_metrics[key + '_public_output_ownership']['route'] == (
        'adopt_arena_download' if certified else 'copy_to_public_numpy')
    session.close()
    np.testing.assert_array_equal(first.values, np.arange(4) + 1)
    np.testing.assert_array_equal(second.values, np.arange(4) + 2)
    assert not first.values.flags.writeable and not second.values.flags.writeable


def test_real_bridge_shape_exception_does_not_leak_producer_scope():
    session = HostSession()
    def malformed(*args):
        return download(np.arange(3, dtype=np.float64))
    session.backend.query = malformed
    with pytest.raises(RuntimeError, match='one complete value'):
        session.query('g', 'bfs', {'source': 0})
    values = np.arange(4, dtype=np.float64)
    _register_download(values)  # outside any live request: cannot register
    assert publish(values, scope=object()).values is not values


def test_wcc_certified_download_consumed_once_and_caller_always_copied():
    labels = np.array([0., 0., 2., 2., 4.])
    with _download_request('g', 'wcc') as scope:
        values = download(labels)
        first = canonical_wcc(values, len(values), certify_minimum=True, _arena_scope=scope, _arena_graph='g')
        second = canonical_wcc(values, len(values), certify_minimum=True, _arena_scope=scope, _arena_graph='g')
    assert first is values and second is not values
    ordinary = canonical_wcc(values, len(values), certify_minimum=True)
    assert ordinary is not values and not np.shares_memory(ordinary, values)
    with _download_request('g', 'wcc') as scope:
        caller = canonical_wcc(labels, len(labels), certify_minimum=True, _arena_scope=scope, _arena_graph='g')
    assert caller is not labels


def test_wcc_nonminimum_roots_force_independent_canonicalization():
    labels = np.array([1., 1., 3., 3., 4.])
    with _download_request('g', 'wcc') as scope:
        values = download(labels)
        result = canonical_wcc(values, len(values), certify_minimum=True, _arena_scope=scope, _arena_graph='g')
    assert not np.shares_memory(result, values)
    np.testing.assert_array_equal(values, labels)
    np.testing.assert_array_equal(result, [0., 0., 2., 2., 4.])


def test_wcc_late_invalid_label_cannot_transfer(monkeypatch):
    from easygraph.gpu import _core
    monkeypatch.setattr(_core, '_WCC_CANONICAL_CHUNK', 2)
    with _download_request('g', 'wcc') as scope:
        values = download(np.array([0., 0., 2., 2., np.nan]))
        with pytest.raises(ValueError, match='exact dense indices'):
            canonical_wcc(values, len(values), certify_minimum=True, _arena_scope=scope, _arena_graph='g')


@pytest.mark.parametrize('certified', [False, True])
@pytest.mark.parametrize('minimum', [False, True])
def test_real_wcc_bridge_owns_outputs_after_next_query_close(certified, minimum):
    session = HostSession()
    labels = np.array([0., 0., 2., 2.] if minimum else [1., 1., 3., 3.])
    def query(*args):
        return download(labels) if certified else labels
    session.backend.query = query
    first = session.query('g', 'wcc')
    second = session.query('g', 'wcc')
    session.close()
    labels[:] = 3.
    np.testing.assert_array_equal(first.values, [0., 0., 2., 2.])
    np.testing.assert_array_equal(second.values, [0., 0., 2., 2.])
    assert not np.shares_memory(first.values, second.values) and not first.values.flags.writeable
    metrics = first.backend_metrics['wcc_public_output']
    adopted = certified and minimum
    assert metrics['ownership'] == ('adopt_certified_arena_download' if adopted else 'adopt_private_canonical_array')
    assert metrics['canonical_output_write_bytes'] == (0 if adopted else first.values.nbytes)


@pytest.mark.parametrize('algorithm', ['bfs', 'sssp', 'pr', 'weighted_pr', 'wcc'])
@pytest.mark.parametrize('certified', [False, True])
def test_real_core_managed_mro_all_five_functions(algorithm, certified):
    session = HostSession(certified=certified)
    expected = np.array([0., 0., 2., 2.]) if algorithm == 'wcc' else np.arange(4, dtype=np.float64)
    def query(*args):
        return download(expected) if certified else expected
    session.backend.query = query
    assert type(session)._own_result is CoreManagedSession._own_result
    parameters = {'source': 0} if algorithm in ('bfs', 'sssp') else {}
    result = session.query('g', algorithm, parameters)
    session.close()
    np.testing.assert_array_equal(result.values, expected)
    assert not result.values.flags.writeable
    if algorithm == 'sssp':
        assert result.contract.weight_projection == 'input edge weights'
    if algorithm in ('pr', 'weighted_pr'):
        assert result.contract.max_iterations == 1000
        assert result.contract.tolerance_per_vertex == 1e-9 / 4
