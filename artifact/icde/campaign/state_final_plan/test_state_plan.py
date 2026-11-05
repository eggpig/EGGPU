"""CPU-only protocol/admission tests; no GPU library imports or executions."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import venv
import pytest
import state_plan as p


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return p.bind(path)


@pytest.fixture
def metadata():
    signer = {'path': '/old/signer.py', 'bytes': 3, 'sha256': 'a' * 64}
    freeze = {'source_files': {p.SIGNER: {k: signer[k] for k in ('bytes', 'sha256')}}}
    payload = {key: {'path': '/' + key, 'bytes': 4, 'sha256': 'b' * 64}
               for key in ('offsets','indices','weights','metadata','sources')}
    canonical = {'files': payload,
                 'vertices': 7, 'stored_arcs': 9, 'source_selection': {'primary_sources': [1, 3, 5]}}
    registry = {'status': 'ALL_SEVEN_CANONICAL_READY', 'records': {'US': {'receipt': canonical}}}
    row = {'canonical': canonical['files'], 'n': 7, 'm': 9, 'sources': [1, 3, 5],
           'requested_rho': 1e-4, 'locality': 'uniform', 'edge_cuts': [0, 9]}
    record = {'schema': 'state.inputs.v2', 'status': 'PASS_CPU_INPUTS_GPU_UNQUALIFIED',
              'qualification_only': False, 'partitions': 1, 'source_signer': signer,
              'shared_grid_preparation': {'source_signer': signer}, 'datasets': {'US': row},
              'cell': {'experiment': 'E3', 'datasets': ['US'], 'partitions': 1,
                       'rho': 1e-4, 'locality': 'uniform'}}
    return record, freeze, registry


def test_grid_complete_and_nonredundant():
    rows = p.grid()
    assert len(rows) == 288
    assert len({x['id'] for x in rows}) == 288
    assert sum(p.initial(x) for x in rows) == 18
    e3 = [x for x in rows if x['experiment'] == 'E3']
    assert len(e3) == 200
    assert {x['arm'] for x in e3} == {'D0C1', 'D1C1'}
    assert all(x['locality'] == 'uniform' for x in e3 if x['rho'] in (0, 1))
    assert {x['datasets'][0] for x in e3 if x['partitions'] == 8} == {'IN', 'OR'}
    assert {x['arm'] for x in rows if x['experiment'] == 'E4'} == {'D1C0', 'D1C1'}
    assert {x['arm'] for x in rows if x['experiment'] == 'E5'} == {'D0C0','D1C0','D0C1','D1C1'}


def test_equivalent_signer_allows_old_provenance(metadata):
    record, freeze, registry = metadata
    old = deepcopy(record)
    p.validate_input_metadata(record, freeze, registry)
    assert record == old  # Never rewrite source signer/original input receipt.


def test_registry_canonical_metadata_alias_is_exact(metadata):
    record, freeze, registry = metadata
    registry = deepcopy(registry)
    files = registry['records']['US']['receipt']['files']
    files['canonical'] = files.pop('metadata')
    files['id_mapping'] = {'path':'/ids', 'bytes':7, 'sha256':'d'*64}
    p.validate_input_metadata(record, freeze, registry)
    files['canonical']['sha256'] = 'e'*64
    with pytest.raises(ValueError, match='Canonical'):
        p.validate_input_metadata(record, freeze, registry)


@pytest.mark.parametrize('mutation', [
    lambda r, f, g: f['source_files'][p.SIGNER].update(sha256='c' * 64),
    lambda r, f, g: r['datasets']['US'].update(sources=[3, 1, 5]),
    lambda r, f, g: r['datasets']['US'].update(requested_rho=.1),
    lambda r, f, g: r['datasets']['US'].update(locality='clustered'),
    lambda r, f, g: r.update(partitions=8),
    lambda r, f, g: r.update(qualification_only=True),
    lambda r, f, g: r['datasets']['US'].update(n=8),
    lambda r, f, g: r['datasets']['US'].update(edge_cuts=[1, 9]),
    lambda r, f, g: r['datasets']['US'].update(canonical={}),
    lambda r, f, g: g.update(status='TW_PENDING'),
])
def test_metadata_rejects_mismatch(metadata, mutation):
    record, freeze, registry = metadata
    mutation(record, freeze, registry)
    with pytest.raises(ValueError):
        p.validate_input_metadata(record, freeze, registry)


def test_real_input_requires_signer(metadata):
    record, freeze, registry = metadata
    record.pop('source_signer')
    with pytest.raises(ValueError, match='signer'):
        p.validate_input_metadata(record, freeze, registry)


def test_source_freeze_detects_mutation_and_unbound_source(tmp_path):
    file = tmp_path / p.SIGNER
    file.parent.mkdir(parents=True)
    file.write_text('test signer')
    table = {p.SIGNER: {k: p.bind(file)[k] for k in ('bytes','sha256')}}
    freeze = {'source_files': table, 'source_tree_sha256': p.digest(table)}
    p.source_files(tmp_path, freeze)
    extra = file.with_name('unbound.py')
    extra.write_text('print(1)')
    with pytest.raises(ValueError, match='Unbound'):
        p.source_files(tmp_path, freeze)
    extra.unlink()
    file.write_text('mutated')
    with pytest.raises(ValueError, match='changed'):
        p.source_files(tmp_path, freeze)


def test_pilot_requires_actual_same_resource_and_source():
    config = {'source_binding': {'files': 'final'}, 'devices': [0], 'environment': {'A': 'b'},
              'gpu_uuids': ['GPU-one'], 'cpu_affinity': [1, 2]}
    proof = dict(config, schema='state.byte.pilot.v4', status='PASS_PHYSICAL_BYTE_PILOT',
        gpu_executed=True, selection_uses_timing=False, inputs={'sha256': 'input'},
        resident_view_bytes={'IN': {'0': 17}, 'UK': {'0': 23}}, phase='target')
    p.check_pilot(proof, config, {'sha256': 'input'}, ['IN', 'UK'])
    for key, value in [('gpu_uuids', ['GPU-other']), ('cpu_affinity', [3,4]),
                       ('source_binding', {'files': 'old'}), ('selection_uses_timing', True),
                       ('resident_view_bytes', {'IN': {'0': 0}, 'UK': {'0': 23}})]:
        bad = deepcopy(proof)
        bad[key] = value
        with pytest.raises(ValueError):
            p.check_pilot(bad, config, {'sha256': 'input'}, ['IN', 'UK'])


def test_environment_rejects_hidden_treatment(monkeypatch):
    for key in list(os.environ):
        if key.startswith(('EGGPU_', 'EGMGPU_', 'NCCL_', 'UCX_', 'CUDA_', 'CUPY_')) or key in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH'):
            monkeypatch.delenv(key)
    result = p.controlled_environment({'PYTHONPATH': 'bound'}, {'OMP_NUM_THREADS':'1'}, 'GPU-one')
    assert result['CUDA_VISIBLE_DEVICES'] == 'GPU-one'
    monkeypatch.setenv('EGGPU_UNBOUND_SWITCH', '1')
    with pytest.raises(ValueError, match='Unbound'):
        p.controlled_environment({}, {}, 'GPU-one')


def test_plan_no_overwrite(tmp_path):
    path = tmp_path / 'evidence.json'
    p.save_new(path, {'original': True})
    with pytest.raises(FileExistsError):
        p.save_new(path, {'original': False})
    assert p.read(path) == {'original': True}


def formal_fixture(tmp_path, repeats):
    jobs = []
    for repeat in repeats:
        spec = {'experiment':'E3', 'datasets':['US'], 'arm':'D1C1', 'old_readers':'none',
                'beta':.25, 'order':'interleaved', 'partitions':1, 'inputs':{'sha256':'input'},
                'repeat':repeat, 'receipt':str(tmp_path / 'results' / f'r{repeat}' / 'receipt.json')}
        file = tmp_path / 'jobs' / f'r{repeat}.json'
        write(file, spec)
        jobs.append({'id':f'r{repeat}', 'command':['python','state_plan.py','guard','--spec',str(file)],
                     'depends_on':[], 'bindings':[]})
    write(tmp_path / 'formal_pending.json', {'status':'NOT_ADMITTED_REQUIRES_EXACT_QUALIFICATIONS',
                                            'jobs':jobs, 'environment':{}})


@pytest.mark.parametrize('repeats', [list(range(9)), list(range(1,11)), [0]*10])
def test_formal_requires_ten_exact_independent_repeats(tmp_path, monkeypatch, repeats):
    formal_fixture(tmp_path, repeats)
    monkeypatch.setattr(p, 'check_spec', lambda _: {'proof_bindings':[]})
    with pytest.raises(ValueError, match='ten independent'):
        p.admit_formal(SimpleNamespace(bundle=tmp_path, output=tmp_path/'formal.json'))


def test_formal_rejects_missing_qualification(tmp_path, monkeypatch):
    formal_fixture(tmp_path, range(10))
    def missing(_):
        raise FileNotFoundError('target qualification absent')
    monkeypatch.setattr(p, 'check_spec', missing)
    with pytest.raises(FileNotFoundError):
        p.admit_formal(SimpleNamespace(bundle=tmp_path, output=tmp_path/'formal.json'))
    assert not (tmp_path / 'formal.json').exists()


def test_formal_admission_is_not_measurement(tmp_path, monkeypatch):
    formal_fixture(tmp_path, range(10))
    proof = write(tmp_path / 'target_receipt.json', {'status':'TEST_ONLY'})
    monkeypatch.setattr(p, 'check_spec', lambda _: {'proof_bindings':[proof]})
    output = tmp_path / 'formal.json'
    result = p.admit_formal(SimpleNamespace(bundle=tmp_path, output=output))
    assert result['gpu_executed'] is False
    admitted = p.read(output)
    assert admitted['formal_measurements'] == 0 and len(admitted['jobs']) == 10
    assert admitted['jobs'][0]['bindings'] == [proof]


def test_readonly_guard_never_executes(monkeypatch):
    monkeypatch.setattr(p, 'check_spec', lambda _: {'status':'PASS_CPU_ADMISSION', 'gpu_executed':False})
    monkeypatch.setattr(os, 'execve', lambda *_: pytest.fail('GPU process must not launch in CPU admission'))
    assert p.guard(SimpleNamespace(spec=Path('synthetic'),receipt=None,execute_gpu=False))['gpu_executed'] is False


def test_real_symlink_venv_must_preserve_invocation(tmp_path):
    root=tmp_path/'venv'
    venv.EnvBuilder(with_pip=False,symlinks=True).create(root)
    invocation=str(root/'bin/python')
    runtime=p.runtime_probe(invocation,{})
    assert runtime['invocation']==invocation
    assert runtime['prefix']==str(root)
    assert runtime['prefix']!=runtime['base_prefix']
    resolved=p.bind(invocation)['path']
    assert resolved!=invocation
    base=p.runtime_probe(resolved,{})
    assert p.bind(invocation)['sha256']==p.bind(resolved)['sha256']
    assert base['prefix']!=runtime['prefix']  # Same ELF is insufficient.
    with pytest.raises(ValueError,match='lexical'):
        p.verify_runtime({'python_invocation':invocation,'python_runtime':runtime,'environment':{}})


def test_legacy_python_identity_cannot_admit():
    with pytest.raises(ValueError,match='Legacy'):
        p.verify_runtime({'python':p.bind(sys.executable)})


def test_old_qualification_without_launcher_identity_rejected(tmp_path):
    spec=write(tmp_path/'spec.json',{'runtime_admission':str(tmp_path/'missing.json')})
    with pytest.raises(FileNotFoundError):
        p.check_prior_admission(spec,{'python_runtime':{},'source_tree_sha256':'new'})
