"""CPU-only tests of the added coverage menu, causal routing and proof gates."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import shutil

import numpy as np
import pytest

import build_plans as plans
import e6_cpu_reference as reference
import e6_reused as reused
import e6_worker as worker


HERE = Path(__file__).resolve().parent


def args(**updates):
    values = dict(system='eggpu', ablate='none', coverage_backend='none',
                  gpu_count=1, gpu_uuids=['GPU-fixture'], dataset=worker.TINY,
                  phase='tiny', repeat=0, timeout=3600., release=Path('/release'),
                  native_dso=Path('/native/cpp.so'), fallback_site=None)
    values.update(updates)
    return SimpleNamespace(**values)


def vectors():
    scores = np.arange(40, dtype='f8') + 1
    scores[36:] = np.nan
    values = dict(effective_size=scores, constraint=scores / 100)
    values['candidates'] = reused.select_candidates(scores, values['constraint'])
    values['closeness'] = np.linspace(.1, .5, 32)
    values['winner'] = [reused.select_winner(values['candidates'], values['closeness'])]
    values['bfs'] = np.r_[np.ones(36), np.full(4, np.inf)]
    values['bfs'][values['winner'][0]] = 0.
    values['wcc'] = np.r_[np.zeros(36, dtype='i8'), np.arange(36, 40)]
    return {name: reused.own(value) for name, value in values.items()}


def trace_for(config):
    stages = []
    for function in worker.FUNCTIONS:
        delegated = function in worker.GROUPS.get(config.ablate, ())
        used = [] if delegated and config.coverage_backend == 'igraph' else (
            list(range(config.gpu_count)) if not delegated and function in
            ('effective_size', 'constraint', 'closeness') else [0])
        stages.append(dict(function=function, metrics=dict(devices_used=used,
            coverage_fallback=delegated, coverage_backend=config.coverage_backend if delegated else 'eggpu')))
    return {'stages': stages}


@pytest.mark.parametrize('backend', ['igraph', 'cugraph_composite'])
@pytest.mark.parametrize('group', ['structural', 'closeness', 'structural_closeness'])
def test_dispatches_actual_selected_functions_with_native_traversal_tail(monkeypatch, backend, group):
    instances = {}
    expected = vectors()

    class Fake:
        def __init__(self, kind):
            self.kind, self.calls, self.closed = kind, [], False
            self.eg = SimpleNamespace(__file__='/release/easygraph/__init__.py')
            instances[kind] = self

        def call(self, function, selected=None):
            self.calls.append((function, selected))
            used = [] if self.kind == 'igraph' else [0]
            return reused.own(expected[function]), {'devices_used': used}

        def close(self):
            self.closed = True

    monkeypatch.setattr(worker, 'DenseNative', lambda *a: Fake('eggpu'))
    monkeypatch.setattr(reused, 'IgraphBackend', lambda *a: Fake('igraph'))
    monkeypatch.setattr(reused, 'GPUCompositeBackend', lambda *a: Fake('cugraph_composite'))
    config = args(ablate=group, coverage_backend=backend)
    wrapper = worker.CoverageBackend(config, {'n': 40}, Path('/unused'))
    values, trace = worker.flow(wrapper, 40)
    assert worker.audit_execution(trace, config)
    assert worker.audit_output(values, expected, 40)['pass']
    assert [f for f, _ in instances[backend].calls] == list(worker.GROUPS[group])
    native_calls = [f for f, _ in instances['eggpu'].calls]
    assert native_calls[-2:] == ['bfs', 'wcc']
    close_backend = instances[backend] if group in ('closeness', 'structural_closeness') else instances['eggpu']
    selected = next(s for f, s in close_backend.calls if f == 'closeness')
    assert np.array_equal(selected, values['candidates'])
    bfs_source = next(s for f, s in instances['eggpu'].calls if f == 'bfs')
    assert bfs_source == int(values['winner'][0])
    wrapper.close()
    assert all(instance.closed for instance in instances.values())


def test_gpu_external_failure_is_not_cpu_or_native_fallback():
    wrapper = worker.CoverageBackend.__new__(worker.CoverageBackend)
    wrapper.fallback = worker.GROUPS['structural_closeness']
    wrapper.external_kind = 'cugraph_composite'
    calls = []

    def failing(*unused):
        calls.append('gpu'); raise MemoryError('fixture GPU capacity failure')

    def unexpected(*unused):
        raise AssertionError('A GPU failure must not choose another implementation')

    wrapper.external = SimpleNamespace(call=failing)
    wrapper.native = SimpleNamespace(call=unexpected)
    with pytest.raises(MemoryError, match='capacity'):
        wrapper.call('constraint')
    assert calls == ['gpu']


@pytest.mark.parametrize('change', ['device', 'backend', 'fallback', 'missing_stage', 'reorder'])
def test_actual_coverage_trace_tampering_rejected(change):
    config = args(ablate='structural_closeness', coverage_backend='cugraph_composite')
    trace = trace_for(config)
    assert worker.audit_execution(trace, config)
    if change == 'device': trace['stages'][0]['metrics']['devices_used'] = []
    if change == 'backend': trace['stages'][0]['metrics']['coverage_backend'] = 'igraph'
    if change == 'fallback': trace['stages'][0]['metrics']['coverage_fallback'] = False
    if change == 'missing_stage': trace['stages'].pop()
    if change == 'reorder': trace['stages'].reverse()
    with pytest.raises(ValueError): worker.audit_execution(trace, config)


@pytest.mark.parametrize('p', [1, 2, 4, 8])
def test_fixed_backend_mg_only_parallelizes_supported_stages(p):
    config = args(gpu_count=p, gpu_uuids=[f'GPU-{i}' for i in range(p)])
    assert worker.audit_execution(trace_for(config), config)


@pytest.mark.parametrize('changes', [
    dict(ablate='structural_closeness'), dict(coverage_backend='igraph'),
    dict(ablate='bfs', coverage_backend='cugraph_composite'),
    dict(ablate='structural', coverage_backend='cugraph_composite', gpu_count=2, gpu_uuids=['a', 'b']),
    dict(system='cugraph_composite', ablate='closeness', coverage_backend='cugraph_composite'),
    dict(system='igraph'), dict(gpu_count=0, gpu_uuids=[]),
    dict(timeout=1800.), dict(dataset='US'), dict(gpu_uuids=[]),
    dict(phase='formal', dataset='EP', repeat=10)])
def test_invalid_menu_resources_or_timeout_rejected(changes):
    with pytest.raises(ValueError): worker.options_valid(args(**changes))


@pytest.fixture
def bridge_fixture(tmp_path):
    producer_file = tmp_path / 'producer.py'; producer_file.write_text('independent CPU formula')
    formula_file = tmp_path / 'formula.py'; formula_file.write_text('unchanged selection formula')
    legacy_contract = json.loads((HERE / 'legacy_contract.json').read_text())
    producer = dict(worker=reference.binding(producer_file), reused=reference.binding(formula_file),
                    numpy='preserved-producer-version', scipy='preserved-scipy-version')
    rows = []
    for dataset in (worker.TINY, *worker.DATASETS):
        blob = tmp_path / (dataset + '.npz'); np.savez(blob, **vectors())
        receipt = tmp_path / (dataset + '.json')
        identity = dict(input_sha256='input-' + dataset, protocol=legacy_contract,
            code={k: producer['worker'][k] for k in ('bytes', 'sha256')},
            reused={k: producer['reused'][k] for k in ('bytes', 'sha256')},
            numpy=producer['numpy'], scipy=producer['scipy'])
        receipt.write_text(json.dumps(dict(identity=identity, gpu_executed=False, performance_eligible=False,
            output={k: reference.binding(blob)[k] for k in ('bytes', 'sha256')})))
        rows.append(dict(dataset=dataset, input_sha256='input-' + dataset,
                         receipt=reference.binding(receipt), output=reference.binding(blob)))
    old = dict(schema=reference.legacy.SCHEMA, protocol=legacy_contract, producer=producer,
               gpu_executed=False, performance_eligible=False, references=rows)
    old_path = tmp_path / 'legacy_manifest.json'; old_path.write_text(json.dumps(old))
    bridge = reference.manifest_record(old_path, worker.CONTRACT)
    path = tmp_path / 'bridge.json'; path.write_text(json.dumps(bridge))
    return dict(path=path, bridge=bridge, old_path=old_path, old=old,
                graph=dict(n=40, input_sha256='input-EP'), formula=formula_file)


def test_reference_bridge_reuses_exact_outputs_and_preserves_producer(bridge_fixture):
    f = bridge_fixture
    reference.validate_manifest(f['path'], worker.CONTRACT, verify_payloads=True)
    actual, receipt = reference.load(f['graph'], f['path'], worker.CONTRACT, f['formula'])
    for name, expected in vectors().items():
        np.testing.assert_equal(actual[name], expected)
    assert json.loads(Path(receipt['path']).read_text())['identity']['numpy'] == 'preserved-producer-version'
    assert f['old']['protocol']['timeout_seconds'] == 1800
    assert f['bridge']['protocol']['timeout_seconds'] == 3600
    assert not f['bridge']['performance_eligible']


@pytest.mark.parametrize('change', ['quality', 'formula', 'selection', 'gpu', 'timing', 'repetitions'])
def test_bridge_rejects_numerical_or_timing_scope_changes(bridge_fixture, change):
    f = bridge_fixture
    contract = copy.deepcopy(worker.CONTRACT)
    keys = {'quality': 'quality', 'formula': 'closeness', 'selection': 'selection',
            'gpu': 'mg', 'timing': 'timing', 'repetitions': 'formal_repetitions'}
    contract[keys[change]] = 'changed'
    with pytest.raises(ValueError): reference.validate_contract(f['old']['protocol'], contract)


@pytest.mark.parametrize('change', ['producer', 'formula', 'payload', 'receipt', 'manifest', 'input', 'duplicate'])
def test_bridge_rejects_stale_or_wrong_full_evidence(bridge_fixture, change):
    f = bridge_fixture
    if change == 'producer': Path(f['old']['producer']['worker']['path']).write_text('changed')
    if change == 'formula': f['formula'].write_text('changed')
    if change == 'payload': Path(f['old']['references'][1]['output']['path']).write_bytes(b'changed')
    if change == 'receipt': Path(f['old']['references'][1]['receipt']['path']).write_text('{}')
    if change == 'manifest': f['old_path'].write_text('{}')
    if change == 'input': f['graph']['input_sha256'] = 'different'
    if change == 'duplicate':
        f['old']['references'].append(copy.deepcopy(f['old']['references'][1]))
        f['old_path'].write_text(json.dumps(f['old']))
        f['bridge']['legacy_manifest'] = reference.binding(f['old_path'])
        f['path'].write_text(json.dumps(f['bridge']))
    with pytest.raises(ValueError): reference.load(f['graph'], f['path'], worker.CONTRACT, f['formula'])


@pytest.fixture(scope='module')
def drafted(tmp_path_factory):
    root = tmp_path_factory.mktemp('plan')
    candidate = root / 'candidate'; candidate.mkdir()
    for name in plans.RUNNER_FILES: shutil.copyfile(HERE / name, candidate / name)
    original = HERE / 'fixtures/qualification_plan.json'
    original_copy = root / 'e6_memory_input_v1/full_profile/qualification_plan.json'
    original_copy.parent.mkdir(parents=True)
    shutil.copyfile(original, original_copy)
    shutil.copyfile(original.parent / 'queue_registry.json', original_copy.parent / 'queue_registry.json')
    manifest = root / 'e6_reference_binding_v2/independent_reference_manifest.json'
    manifest.parent.mkdir()
    shutil.copyfile(HERE / 'fixtures/independent_reference_manifest.json', manifest)
    old_here = plans.HERE
    try:
        plans.HERE = candidate
        receipt = plans.build(SimpleNamespace(original=original_copy, campaign=Path('/bound/campaign'),
                          output=candidate / 'draft', draft=True))
    finally:
        plans.HERE = old_here
    qualification = json.loads((candidate / 'draft/qualification_plan.json').read_text())
    formal = json.loads((candidate / 'draft/formal_pending.json').read_text())
    return receipt, qualification, formal


def test_complete_2x2s_share_only_native_control_and_keep_legacy_supplement(drafted):
    receipt, qualification, formal = drafted
    assert len(plans.configurations()) == 15
    assert receipt['qualification_jobs'] == 60 and receipt['formal_pending_cells'] == 450
    assert 'jobs' not in qualification and 'jobs' not in formal
    assert not receipt['admitted'] and not receipt['gpu_executed']
    assert plans.validate_matrix(qualification['job_specs'])
    assert plans.validate_matrix(formal['job_specs'], formal=True)
    native = [j for j in qualification['job_specs'] if j['system'] == 'eggpu']
    assert len({j['command'][0] for j in native}) == 1
    assert '/envs/rapids/bin/python' in native[0]['command'][0]
    for dataset in worker.DATASETS:
        for repeat in range(10):
            block = [j for j in formal['job_specs'] if j['dataset'] == dataset and j['repeat'] == repeat]
            assert len(block) == 15
            assert sum(j['system'] == 'eggpu' and j['ablate'] == 'none' and j['gpu_count'] == 1 for j in block) == 1


@pytest.mark.parametrize('change', ['omission', 'duplicate', 'command', 'timeout', 'proof', 'dependency', 'phase', 'formal_status'])
def test_plan_ledger_rejects_coverage_or_proof_substitution(drafted, change):
    _, qualification, formal = drafted
    jobs = copy.deepcopy(formal['job_specs'] if change == 'formal_status' else qualification['job_specs'])
    if change == 'omission': jobs.pop()
    if change == 'duplicate': jobs.append(copy.deepcopy(jobs[-1]))
    if change == 'command': jobs[0]['command'][0] = '/other/python'
    if change == 'timeout':
        plans.set_option(jobs[0]['command'], 'timeout', 1800)
        jobs[0]['command_sha256'] = plans.digest(jobs[0]['command'])
    if change == 'proof':
        job = next(j for j in jobs if j['phase'] == 'target')
        plans.set_option(job['command'], 'tiny-proof', '/old/protocol/result.json')
        job['command_sha256'] = plans.digest(job['command'])
    if change == 'dependency': next(j for j in jobs if j['phase'] == 'target')['depends_on'] = []
    if change == 'phase': jobs[0]['phase'] = 'diagnostic'
    if change == 'formal_status': jobs[0]['expected_statuses'] = ['PASS_QUALIFICATION']
    with pytest.raises(ValueError): plans.validate_matrix(jobs, formal=change == 'formal_status')


def test_reused_algorithm_source_remains_identical():
    upstream = json.loads((HERE / 'upstream_bindings.json').read_text())
    assert reference.binding(HERE / 'e6_reused.py')['sha256'] == upstream['e6_reused.py']['sha256']
    assert reference.binding(HERE / 'legacy_cpu_reference.py')['sha256'] == upstream['e6_cpu_reference.py']['sha256']


@pytest.mark.parametrize('change', ['none', 'backend', 'protocol', 'omitted_call', 'output_bytes', 'failed_quality'])
def test_qualification_is_current_backend_and_full_output_bound(tmp_path, change):
    config = args(ablate='structural_closeness', coverage_backend='cugraph_composite')
    epoch = tmp_path / 'epoch.json'; epoch.write_text(json.dumps({'current': 'coverage2x2'}))
    epoch_sha = worker.sg.digest_json(json.loads(epoch.read_text()))
    blob = tmp_path / 'output.bin'; blob.write_bytes(b'synthetic-test-proof-only')
    call = dict(quality={'pass': True}, trace=trace_for(config),
                outputs={name: worker.sg.binding(blob) for name in reference.legacy.FIELDS})
    child = dict(status='PASS', quality_pass=True, input_sha256='synthetic-input',
                 calls={name: copy.deepcopy(call) for name in ('first', 'warmup', 'ready')})
    if change == 'omitted_call': child['calls'].pop('ready')
    if change == 'failed_quality': child['calls']['first']['quality']['pass'] = False
    child_path = tmp_path / 'child.json'; child_path.write_text(json.dumps(child))
    record = dict(phase='tiny', system=config.system, ablate=config.ablate, gpu_count=config.gpu_count,
        coverage_backend=config.coverage_backend, epoch_sha256=epoch_sha, protocol=worker.PROTOCOL,
        contract=worker.CONTRACT, status='PASS_QUALIFICATION', quality_pass=True, dataset=worker.TINY,
        input_sha256='synthetic-input', epoch=worker.sg.binding(epoch), child=worker.sg.binding(child_path))
    if change == 'backend': record['coverage_backend'] = 'igraph'
    if change == 'protocol': record['protocol'] = 'legacy-1800s'
    if change == 'output_bytes': blob.write_bytes(b'changed')
    path = tmp_path / 'proof.json'; path.write_text(json.dumps(record))
    if change == 'none':
        assert worker.validate_proof(path, 'tiny', config, epoch_sha, {}) == worker.sg.binding(path)
    else:
        with pytest.raises(ValueError): worker.validate_proof(path, 'tiny', config, epoch_sha, {})


def test_inherited_binding_keeps_virtual_environment_path(tmp_path):
    import build_plans as plans
    real=tmp_path/'base-python';real.write_bytes(b'python-binary')
    invocation=tmp_path/'env-python';invocation.symlink_to(real)
    item=plans.reference.binding(real);item['path']=str(invocation)
    assert plans.check_template_binding(item)==invocation
    assert str(invocation)!=str(real)
    real.write_bytes(b'changed binary')
    with pytest.raises(ValueError,match='Inherited template bytes changed'):
        plans.check_template_binding(item)


def test_inherited_binding_declared_length_is_not_ignored(tmp_path):
    import build_plans as plans
    p=tmp_path/'file';p.write_bytes(b'abc');item=plans.reference.binding(p);item['bytes']+=1
    with pytest.raises(ValueError,match='Inherited template bytes changed'):
        plans.check_template_binding(item)
