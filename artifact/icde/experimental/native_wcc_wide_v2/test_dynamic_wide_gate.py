"""Exercise actual public dynamic input validation with real signed tiny files."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import sys

import numpy as np
import pytest

from easygraph.gpu import _dynamic as dynamic
from test_wide_offsets import fixture, widen, read, save, resign

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'runner'))
import native_worker as worker


def make_f64(path):
    doc = read(path)
    for part in doc['partitions']:
        item = part['files'].get('weights')
        if item:
            payload = np.fromfile(path.parent / item['path'], dtype='<f4').astype('<f8').tobytes()
            (path.parent / item['path']).write_bytes(payload)
            item.update(dtype='float64', byte_size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    save(path, resign(doc))


@pytest.mark.parametrize('wide', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_public_dynamic_gate_accepts_actual_int64_offsets_without_cast(tmp_path, wide, weighted):
    path = fixture(tmp_path / 'graph', weighted=weighted)
    if weighted: make_f64(path)
    if wide: widen(path)
    before = {p: p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
    assert dynamic.verify_manifest(path, read(path)) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == {p: p.read_bytes() for p in path.parent.iterdir() if p.is_file()}


@pytest.mark.parametrize('failure', ['payload', 'truncation', 'byte_width', 'signature'])
def test_public_dynamic_gate_rejects_wide_payload_identity_failures(tmp_path, failure):
    path = fixture(tmp_path / 'graph'); widen(path); doc = read(path)
    item = doc['partitions'][0]['files']['offsets']; payload = path.parent / item['path']
    if failure == 'payload':
        changed = bytearray(payload.read_bytes()); changed[-1] ^= 1; payload.write_bytes(changed)
    if failure == 'truncation': payload.write_bytes(payload.read_bytes()[:-1])
    if failure == 'byte_width': item['byte_size'] = item['count'] * 4
    if failure == 'signature': doc['view']['signature'] = '0' * 64
    with pytest.raises(ValueError): dynamic.verify_manifest(path, doc)


@pytest.mark.parametrize('field', ['indices', 'owners', 'local_to_global', 'weights'])
def test_int64_is_not_enabled_for_other_public_arrays(tmp_path, field):
    path = fixture(tmp_path / 'graph', weighted=True); make_f64(path); widen(path); doc = read(path)
    item = doc['partitions'][0]['files'][field]
    payload = np.zeros(item['count'], dtype='<i8').tobytes()
    (path.parent / item['path']).write_bytes(payload)
    item.update(dtype='int64', byte_size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    save(path, resign(doc))
    with pytest.raises(ValueError, match='Unsupported dtype'): dynamic.verify_manifest(path, read(path))


def test_real_weight_stage_preserves_wide_topology_and_old_snapshot(tmp_path):
    path = fixture(tmp_path / 'base', weighted=True); make_f64(path); widen(path)
    before = read(path); old_weight = (path.parent / before['partitions'][0]['files']['weights']['path']).read_bytes()
    target, receipt = dynamic.stage_weights(path, 'new', [(0, 0, 7.25)], tmp_path / 'versions')
    after = read(target)
    assert before['graph']['topology_sha256'] == after['graph']['topology_sha256']
    assert before['graph']['weight_sha256'] != after['graph']['weight_sha256']
    assert receipt['semantic_value_bytes'] == 8
    assert (path.parent / before['partitions'][0]['files']['weights']['path']).read_bytes() == old_weight
    for a, b in zip(before['partitions'], after['partitions']):
        assert a['files']['offsets']['sha256'] == b['files']['offsets']['sha256']
        assert b['files']['offsets']['dtype'] == 'int64'
    assert dynamic.verify_manifest(target, after)


@pytest.fixture(autouse=True)
def restore_environment():
    saved = dict(os.environ)
    yield
    os.environ.clear(); os.environ.update(saved)


def configuration(variant, count=1, **updates):
    argv = ['--algorithm', 'wcc', '--dataset', 'TINY_D', '--phase', 'tiny',
            '--count', str(count), '--gpu-uuids', ','.join('GPU-' + str(i) for i in range(count)),
            '--wcc-qualification-variant', variant]
    result = worker.parser().parse_args(argv)
    for key, value in updates.items(): setattr(result, key, value)
    return result


@pytest.mark.parametrize('count', [1, 2, 4, 8])
def test_label_disables_all_forest_only_options_and_next_job_restores_them(count):
    worker.configure(configuration('label_coo', count))
    expected = dict(EGMGPU_WCC_ALGORITHM='label', EGMGPU_WCC_BOUNDARY_REPRESENTATION='coo',
                    EGMGPU_WCC_HOST_ASSEMBLY='caller', EGMGPU_WCC_RESULT_MERGE='legacy',
                    EGMGPU_CORE_OUTPUT_TEAM='original', EGGPU_NATIVE_WCC_WORKER_FP64_OUTPUT='0',
                    EGMGPU_WCC_LOCAL_UNION='all_arcs')
    assert {k: os.environ[k] for k in expected} == expected
    worker.configure(configuration('forest_coo', count))
    assert os.environ['EGMGPU_WCC_LOCAL_UNION'] == 'unique_edges'
    assert os.environ['EGGPU_NATIVE_WCC_WORKER_FP64_OUTPUT'] == '1'
    assert os.environ['EGMGPU_CORE_OUTPUT_TEAM'] == 'team16'
    assert os.environ['EGMGPU_WCC_HOST_ASSEMBLY'] == ('tiled' if count == 1 else 'worker')


@pytest.mark.parametrize('variant', ['label_coo', 'forest_csr'])
def test_alternate_variant_is_never_target_or_formal_configuration(variant):
    for phase in ('target', 'formal'):
        with pytest.raises(ValueError, match='tiny qualification only'):
            worker.configure(configuration(variant, phase=phase, dataset='TW'))


@pytest.mark.parametrize('fault', ['none', 'mutable', 'borrowed', 'dtype', 'assembly', 'offsets'])
def test_label_query_executes_full_public_result_validation(fault):
    values = np.arange(4, dtype='f8'); values.flags.writeable = False
    if fault == 'mutable': values = values.copy()
    if fault == 'borrowed': values = values.view()
    if fault == 'dtype': values = np.arange(4, dtype='i8'); values.flags.writeable = False
    layouts = [{'partition_id': 0, 'dtype': 'int64', 'bytes': 40, 'sha256': 'fixture'}]
    metrics = dict(devices_used=[0], wcc_offset_layouts=layouts,
        wcc_offset_policy='adaptive_signed_i32_i64_wcc_v1', wcc_algorithm='label',
        boundary_representation='coo', host_assembly_strategy='caller', result_merge_mode='legacy')
    if fault == 'assembly': metrics['host_assembly_strategy'] = 'worker'
    if fault == 'offsets': metrics['wcc_offset_layouts'] = []
    backend = worker.NativeBackend.__new__(worker.NativeBackend)
    backend.args = configuration('label_coo'); backend.algorithm = 'wcc'; backend.source = None
    backend.count = 1; backend.expected_offset_layouts = layouts; backend.last_metrics = {}
    backend.session = SimpleNamespace(query=lambda *unused: SimpleNamespace(values=values, backend_metrics=metrics))
    if fault == 'none':
        assert backend.query() is values
        assert backend.last_metrics['frozen_native_profile_observation']['qualification_only']
    else:
        with pytest.raises(ValueError): backend.query()
