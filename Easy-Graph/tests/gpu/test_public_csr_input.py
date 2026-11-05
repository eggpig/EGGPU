"""CPU qualification of public CSR admission using the real owned host store.

CuPy's import-time RawModule default is replaced with a function that raises if
called. No CUDA operation or kernel construction is permitted in this suite.
"""
from contextlib import nullcontext
from threading import RLock
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from easygraph.gpu import Session, plan_session
from easygraph.gpu._bridge import ManagedSession
from easygraph.gpu._csr_input import prepare_csr


@pytest.fixture
def host_modules(monkeypatch):
    cp = ModuleType('cupy')
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU execution is forbidden in CPU CSR tests')
    cp.RawModule = cp.RawKernel = forbidden
    cp.cuda = SimpleNamespace(Device=lambda device: nullcontext())
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    from easygraph.gpu._arena.input_hotpath.owned_csr_v6 import OwnedCSRStore, _ProducedCSR
    from easygraph.gpu._arena.round4_switch.journal import Mutation
    return OwnedCSRStore, _ProducedCSR, Mutation


def fixture():
    return dict(nodes=np.array([-9, 4, 10, 11, 13, 99], dtype=np.int64),
                offsets=np.array([0, 1, 2, 4, 4, 5, 5], dtype=np.int32),
                indices=np.array([1, 2, 0, 3, 5], dtype=np.int32),
                weights=np.array([1.25, 2.5, 4.75, 0., 3.125]))


def register(Store, payload):
    prepared, profiles = prepare_csr({'g': payload})
    store = Store(4 << 20, 0, 'shared')
    item = prepared['g']
    ref = store.register('g', item['nodes'], item['edges'], item['weights'])
    return store, ref, profiles


def edges_of(state):
    outgoing = state.outgoing
    return {(int(state.nodes[u]), int(state.nodes[outgoing.col[pos]])): float(state.values[outgoing.ids[pos]])
            for u in range(len(state.nodes)) for pos in range(int(outgoing.row[u]), int(outgoing.row[u+1]))}


def test_explicit_format_plan_keeps_one_resource_semantic_route():
    plan = plan_session(input_format='csr', state_model='journal')
    assert plan.backend == 'arena' and plan.input_kind == 'owned_csr'
    assert plan.policy == 'capability_v2'
    assert plan_session().input_kind == 'owned_coo'
    assert plan_session(devices=(0, 1)).input_kind == 'partition_manifest'
    with pytest.raises(ValueError):
        plan_session(devices=(0, 1), input_format='csr')
    with pytest.raises(ValueError):
        plan_session(input_format='auto_fastest')


def test_factory_forwards_explicit_csr_without_retry(monkeypatch):
    calls = []
    def arena(cls, inputs, **options):
        calls.append(options); return SimpleNamespace(dispatch_plan=None)
    monkeypatch.setattr(Session, 'arena', classmethod(arena))
    session = Session.create({'g': fixture()}, input_format='csr', budget_bytes=4096)
    assert session.dispatch_plan.input_kind == 'owned_csr'
    assert calls == [dict(budget_bytes=4096, device=0, input_format='csr')]
    def fail(*args, **kwargs):
        raise MemoryError('admission failure')
    monkeypatch.setattr(Session, 'arena', classmethod(fail))
    with pytest.raises(MemoryError):
        Session.create({'g': fixture()}, input_format='csr', budget_bytes=4096)


def test_real_store_owns_all_arrays_and_avoids_initial_endpoint_table(host_modules):
    Store, PrivateCapability, _ = host_modules
    raw = fixture(); original = {key: value.copy() for key, value in raw.items()}
    prepared, _ = prepare_csr({'g': raw})
    assert not isinstance(prepared['g']['edges'], PrivateCapability)
    store, ref, profiles = register(Store, raw)
    state = store.snapshot(ref).host_state
    base = store._records['g'].base_edges
    assert base._expanded_pairs is None and base._expanded_keys is None
    assert profiles['g']['dense_endpoint_temporary_bytes'] == 0
    assert profiles['g']['canonical_dense_endpoint_bytes_avoided'] == 16 * len(raw['indices'])
    for owned, caller in [(state.nodes, raw['nodes']), (state.outgoing.row, raw['offsets']),
                          (state.outgoing.col, raw['indices']), (state.values, raw['weights'])]:
        assert not np.shares_memory(owned, caller)
        assert not owned.flags.writeable and caller.flags.writeable
    raw['nodes'][:] = 777; raw['offsets'][:] = 0; raw['indices'][:] = 0; raw['weights'][:] = 99
    np.testing.assert_array_equal(state.nodes, original['nodes'])
    np.testing.assert_array_equal(state.outgoing.row, original['offsets'])
    np.testing.assert_array_equal(state.outgoing.col, original['indices'])
    np.testing.assert_array_equal(state.values, original['weights'])
    assert len(state.nodes) == 6 and state.outgoing.row[-1] == 5
    assert base[(-9, 4)] == 1.25
    with pytest.raises(KeyError):
        base[(4, -9)]  # Admission never symmetrizes directed adjacency.


def test_reciprocal_and_zero_arc_csr_preserve_all_isolates(host_modules):
    Store, _, _ = host_modules
    for row, col in [([0, 1, 2, 2, 2], [1, 0]), ([0, 0, 0, 0, 0], [])]:
        payload = dict(nodes=np.arange(4), offsets=np.array(row, np.int32), indices=np.array(col, np.int32))
        store, ref, _ = register(Store, payload)
        state = store.snapshot(ref).host_state
        assert len(state.nodes) == 4 and state.m == len(col)
        np.testing.assert_array_equal(state.values, np.ones(len(col)))
        assert store._records['g'].base_edges._expanded_pairs is None


@pytest.mark.parametrize('change', [
    lambda g: g.update(nodes=np.array([5, 4, 10, 11, 13, 99])),
    lambda g: g.update(nodes=np.array([0, 1, 2, 3, 4, 1 << 63], np.uint64)),
    lambda g: g.update(offsets=np.array([0, 1, 2, 4, 4, 6, 5])),
    lambda g: g.update(offsets=np.array([0, 1, 2, 4, 4, 5])),
    lambda g: g.update(indices=np.array([1, 2, 0, 9, 5])),
    lambda g: g.update(indices=np.array([1., 2., 0., 3., 5.])),
    lambda g: g.update(indices=np.array([1, 2, 3, 0, 5])),
    lambda g: g.update(indices=np.array([1, 2, 0, 0, 5])),
    lambda g: g.update(weights=np.array([1., 2., np.nan, 0., 3.])),
    lambda g: g.update(weights=np.array([1., 2., -1., 0., 3.])),
    lambda g: g.update(weights=np.ones(4)),
    lambda g: g.update(weights=np.ones(5, np.complex128)),
    lambda g: g.update(directed=False),
    lambda g: g.update(edges=np.array([[0, 1]])),
])
def test_invalid_or_ambiguous_csr_is_rejected_without_fallback(host_modules, change):
    raw = fixture(); change(raw)
    with pytest.raises(ValueError):
        prepare_csr({'g': raw})


def test_byte_order_and_sparse_original_labels_use_existing_copy_validation(host_modules):
    Store, _, _ = host_modules
    raw = fixture()
    raw['nodes'] = np.array([-900000000000, 4, 10, 11, 13, 900000000000], dtype='>i8')
    raw['offsets'] = raw['offsets'].astype('>i4')
    raw['indices'] = raw['indices'].astype('>i4')
    raw['weights'] = raw['weights'].astype('>f8')
    store, ref, _ = register(Store, raw)
    state = store.snapshot(ref).host_state
    assert state.nodes.dtype == np.int64 and state.outgoing.row.dtype == np.int32
    assert state.values.dtype == np.float64
    assert edges_of(state)[(-900000000000, 4)] == 1.25


def test_chunk_boundaries_preserve_row_boundaries_and_reject_internal_duplicates(host_modules, monkeypatch):
    import easygraph.gpu._csr_input as module
    monkeypatch.setattr(module, '_CHUNK', 2)
    prepare_csr({'g': fixture()})
    raw = fixture(); raw['indices'][3] = raw['indices'][2]
    with pytest.raises(ValueError, match='sorted unique destination'):
        prepare_csr({'g': raw})


def test_existing_journal_set_insert_delete_preserves_prior_host_snapshot(host_modules):
    Store, _, Mutation = host_modules
    store, ref, _ = register(Store, fixture())
    old = store.snapshot(ref)
    before = edges_of(old.host_state)
    receipt = store.commit(ref, [Mutation('set', -9, 4, 5.5), Mutation('insert', 99, -9, 2.),
                                 Mutation('delete', 10, 11, None)])
    assert receipt.version > old.version
    store.flush(ref)
    current = store.snapshot(ref)
    expected = dict(before); expected[-9, 4] = 5.5; expected[99, -9] = 2.; del expected[10, 11]
    assert edges_of(current.host_state) == expected
    assert edges_of(old.host_state) == before
    assert len(current.host_state.nodes) == 6


def test_managed_bridge_queries_keep_owned_source_labels(host_modules, monkeypatch):
    Store, _, _ = host_modules
    import easygraph.gpu._arena.input_hotpath.confirm_candidate_v1 as module
    class HostOnlyAdapter:
        def __init__(self, data, *, budget_bytes):
            journal = Store(4 << 20, 0, 'shared'); refs = {}
            self.nodes = {name: value['nodes'] for name, value in data.items()}
            for name, value in data.items():
                refs[name] = journal.register(name, value['nodes'], value['edges'], value['weights'])
            self.session = SimpleNamespace(journal=journal, refs=refs, lock=RLock())
    monkeypatch.setattr(module, 'ConfirmCandidateAdapter', HostOnlyAdapter)
    raw = fixture()
    managed = ManagedSession({'g': raw}, budget_bytes=4096, input_format='csr')
    raw['nodes'][:] = 777
    np.testing.assert_array_equal(managed.nodes['g'], [-9, 4, 10, 11, 13, 99])
    assert managed.backend.nodes is managed.nodes
    assert not managed.nodes['g'].flags.writeable
