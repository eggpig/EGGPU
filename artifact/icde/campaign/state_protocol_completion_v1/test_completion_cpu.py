"""Independent CPU ownership/CSR/protocol checks; no CUDA execution."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import os
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import state_controls as controls
import state_contract as contract
import state_inputs
import state_plan
import state_protocol as protocol
import state_rebuild

W = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get('EGGPU_TEST_SOURCE', str(
    Path(__file__).resolve().parents[1] / 'candidates/native_memory_input_v1/Easy-Graph')))
from cpu_ownership_oracle import NativeReferenceCpp
from easygraph.gpu._controller import StateSwitchController
from easygraph.gpu._dynamic import verify_manifest


def controller(cpp, policy, budgets):
    backend = SimpleNamespace(controller=StateSwitchController(cpp, cpp.devices, budgets,
        policy='closure', inventory_mode='recheck', batch_devices=True))
    controls.install_controller(SimpleNamespace(_backend=backend), policy)
    return backend.controller


@pytest.mark.parametrize('count', [1, 2, 4, 8])
def test_view_lru_zero_marginal_delete_does_not_satisfy_budget(count):
    cpp = NativeReferenceCpp(range(count))
    for d in cpp.devices:
        cpp.add(f'b{d}', d, {'shared': 100}, algorithm='bfs', shared=['shared'])
        cpp.add(f'p{d}', d, {'shared': 100}, algorithm='pr', shared=['shared'])
    c = controller(cpp, 'view_lru', {d: 1 for d in cpp.devices})
    result = c.finish_query('G', 'bfs')
    assert [a['freed_bytes'] for a in result['actions']] == [0, 100 * count]
    assert result['after_retention']['managed_resident_bytes'] == 0
    assert all(a['prediction_error_bytes'] == 0 for a in result['actions'])


@pytest.mark.parametrize('policy', ['view_lru', 'no_retention'])
def test_active_child_protects_parent(policy):
    cpp = NativeReferenceCpp()
    cpp.add('old', 0, {'x': 8}, graph='old')
    cpp.add('leased-child', 0, {'y': 2}, kind='workspace', graph='old', parent='old', active=1)
    cpp.add('free', 0, {'z': 6}, graph='new')
    c = controller(cpp, policy, {0: 10})
    if policy == 'no_retention':
        with pytest.raises(RuntimeError, match='still has physical'):
            c.finish_query('new', 'bfs')
    else:
        result = c.finish_query('new', 'bfs')
        assert result['after_retention']['managed_resident_bytes'] == 10
    assert set(cpp.owners) == {'old', 'leased-child'}


@pytest.mark.parametrize('count', [1, 8])
def test_no_retention_releases_actual_shared_union_and_catalog(count):
    cpp = NativeReferenceCpp(range(count))
    for d in cpp.devices:
        cpp.add(f'a{d}', d, {'shared': 8, 'private-a': 1}, graph='A', shared=['shared'])
        cpp.add(f'b{d}', d, {'shared': 8, 'private-b': 1}, graph='B', shared=['shared'])
        cpp.catalog.add((d, 'idle')); cpp.atoms[d, 'idle'] = 3
    result = controller(cpp, 'no_retention', {d: 100 for d in cpp.devices}).finish_query('A','bfs')
    assert result['actions'][0]['freed_bytes'] == count * 13
    assert result['actions'][0]['capacity_progress'] is True
    assert result['actions'][0]['zero_release_is_capacity_progress'] is False
    assert result['after_retention']['managed_resident_bytes'] == 0
    assert result['budgets'] == {d: 100 for d in cpp.devices}


def test_rebuild_release_preserves_real_old_leases():
    cpp = NativeReferenceCpp()
    cpp.add('old', 0, {'old': 8}, active=1, graph='old', shared=['old'])
    cpp.add('free', 0, {'private': 3}, graph='free')
    c = StateSwitchController(cpp, [0], None, policy='closure', inventory_mode='recheck')
    result = controls.retire_for_rebuild(SimpleNamespace(_backend=SimpleNamespace(controller=c)))
    assert set(cpp.owners) == {'old'}
    assert result['freed_bytes'] == 3
    assert result['after_retention']['managed_resident_bytes'] == 8


def test_corrupt_native_release_receipt_is_not_accepted():
    cpp = NativeReferenceCpp()
    cpp.add('free', 0, {'private': 3}, graph='free')
    cpp.mismatch = True
    with pytest.raises(RuntimeError):
        controller(cpp, 'no_retention', {0: 100}).finish_query('free','bfs')


def test_unknown_external_references_fail_closed():
    cpp = NativeReferenceCpp(); cpp.add('a', 0, {'x': 1}, shared=['x'])
    c = StateSwitchController(cpp, [0], None, policy='closure')
    original = c.inventory
    def broken():
        snapshot = original()
        snapshot['per_device'][0]['allocations'][0]['untracked_shared_references'] = 1
        return snapshot
    with pytest.raises(RuntimeError, match='Untracked'):
        controls.release_legal(broken, c.evict, [0])


@pytest.fixture(scope='module')
def fixture_inputs(tmp_path_factory):
    out = tmp_path_factory.mktemp('completion-inputs') / 'prepared'
    state_inputs.prepare(SimpleNamespace(tiny=True, source=SOURCE, output=out,
        rho=.1, locality='uniform', partitions=1, datasets='TINY_A'))
    return json.loads((out/'inputs.json').read_text())


def global_matrix(path):
    document = json.loads(path.read_text()); n = document['graph']['vertex_count']
    edges = []
    for part in document['partitions']:
        def arr(name, dtype):
            return np.fromfile(path.parent / part['files'][name]['path'], dtype=dtype)
        mapping = arr('local_to_global', '<u8')
        offsets, indices = arr('offsets','<i4'), arr('indices','<i4')
        weights = arr('weights','<f8') if document['graph']['weighted'] else np.ones(len(indices))
        for u in range(part['owned_vertex_count']):
            edges.extend((int(mapping[u]), int(mapping[indices[k]]), float(weights[k]))
                         for k in range(offsets[u], offsets[u+1]))
    return sorted(edges), document


@pytest.mark.parametrize('partitions', [1,8])
@pytest.mark.parametrize('version', [1,20])
@pytest.mark.parametrize('projection', ['weighted','bfs','wcc'])
def test_rebuild_reconstructs_full_canonical_host_csr(fixture_inputs,tmp_path,partitions,version,projection):
    row = fixture_inputs['datasets']['TINY_A']
    _, path, receipt = state_rebuild.make_manifest(row, f'TINY_A@{version}',version,partitions,
        state_inputs.signer(SOURCE),tmp_path/'rebuilt',projection=projection)
    observed, doc = global_matrix(path)
    verify_manifest(path, doc)
    off = np.fromfile(row['canonical']['offsets']['path'],dtype='<i4')
    ind = np.fromfile(row['canonical']['indices']['path'],dtype='<i4')
    weights = np.fromfile(row['canonical']['weights']['path'],dtype='<f8')
    positions = np.load(row['positions']['path'])
    weights[positions] = 1+(weights[positions]-1+version)%16
    matrix = csr_matrix((weights if projection=='weighted' else np.ones(len(ind)),ind,off),shape=(row['n'],row['n']))
    if projection=='wcc': matrix=matrix.maximum(matrix.T).tocsr()
    expected=sorted((u,int(matrix.indices[k]),float(matrix.data[k])) for u in range(row['n']) for k in range(matrix.indptr[u],matrix.indptr[u+1]))
    assert observed==expected
    assert doc['graph']['generation']==version+1
    assert receipt['payload_bytes'] == sum(p.stat().st_size for p in path.parent.glob('*.bin'))
    base_cuts = json.loads(Path(row['manifest']['path']).read_text())['partitions']
    if partitions == 1:
        assert [p['ownership'] for p in doc['partitions']] == [p['ownership'] for p in base_cuts]
    expected_cuts = np.linspace(0,row['n'],partitions+1,dtype=np.int64)
    assert [(p['ownership']['vertex_begin'],p['ownership']['vertex_end']) for p in doc['partitions']] == list(zip(expected_cuts[:-1],expected_cuts[1:]))
    assert all((path.parent/f['path']).stat().st_ino != Path(row['canonical']['offsets']['path']).stat().st_ino
               for part in doc['partitions'] for f in part['files'].values())


def test_rebuild_uses_real_registration_and_keeps_lease(fixture_inputs,tmp_path):
    cpp=NativeReferenceCpp();cpp.add('old',0,{'old':8},active=1,graph='old')
    c=StateSwitchController(cpp,[0],None,policy='closure')
    class Session:
        _backend=SimpleNamespace(controller=c)
        def __init__(self):self.registered={}
        def register_snapshot(self,name,path):
            assert name not in self.registered
            verify_manifest(path,json.loads(path.read_text()))
            self.registered[name]=global_matrix(path)
    session=Session()
    selected,receipt=state_rebuild.rebuild(session,fixture_inputs['datasets']['TINY_A'],
        dict(snapshot='TINY_A@1',version=1),1,SOURCE,tmp_path/'attempt')
    assert set(session.registered)==set(selected.values()) and len(selected)==3
    assert set(cpp.owners)=={'old'}
    assert receipt['host_payload_written_bytes']>0
    with pytest.raises(FileExistsError):
        state_rebuild.rebuild(session,fixture_inputs['datasets']['TINY_A'],
            dict(snapshot='TINY_A@1',version=1),1,SOURCE,tmp_path/'attempt')


def test_successive_rebuilds_do_not_overwrite_old_weight_payload(fixture_inputs,tmp_path):
    row=fixture_inputs['datasets']['TINY_A']
    canonical=Path(row['canonical']['weights']['path'])
    original=canonical.read_bytes()
    _,first,_=state_rebuild.make_manifest(row,'TINY_A@1',1,1,state_inputs.signer(SOURCE),
        tmp_path/'first',projection='weighted')
    old={p.name:p.read_bytes() for p in first.parent.glob('*') if p.is_file()}
    _,second,_=state_rebuild.make_manifest(row,'TINY_A@2',2,1,state_inputs.signer(SOURCE),
        tmp_path/'second',projection='weighted')
    assert canonical.read_bytes()==original
    assert {p.name:p.read_bytes() for p in first.parent.glob('*') if p.is_file()}==old
    values1=np.fromfile(first.parent/'p0.weights.bin',dtype='<f8')
    values2=np.fromfile(second.parent/'p0.weights.bin',dtype='<f8')
    base=np.frombuffer(original,dtype='<f8');positions=np.load(row['positions']['path'])
    unchanged=np.ones(len(base),dtype=bool);unchanged[positions]=False
    np.testing.assert_array_equal(values1[positions],1+(base[positions]-1+1)%16)
    np.testing.assert_array_equal(values2[positions],1+(base[positions]-1+2)%16)
    np.testing.assert_array_equal(values1[unchanged],base[unchanged])
    np.testing.assert_array_equal(values2[unchanged],base[unchanged])


@pytest.mark.parametrize('tiny',[False,True])
def test_readonly_orders_have_identical_multiset(tiny):
    grouped=protocol.trace('E3R',['IN'],'grouped',tiny)
    mixed=protocol.trace('E3R',['IN'],'interleaved',tiny)
    encode=lambda rows:Counter(json.dumps(r,sort_keys=True) for r in rows)
    assert len(grouped)==len(mixed)==16 and grouped!=mixed and encode(grouped)==encode(mixed)
    assert all(r['kind']=='query' and r['version']==0 for r in mixed)
    assert Counter(r['function'] for r in mixed)=={f:4 for f in protocol.FUNCTIONS}


def test_new_grid_has_complete_cells_and_reuses_exact_input_identity():
    rows=state_plan.grid()
    assert Counter(r['experiment'] for r in rows)=={'E3':300,'E4':160,'E3R':20}
    assert len({r['id'] for r in rows})==480
    assert all(state_plan.input_key(r)[0]=='E3' and r['rho']==0 for r in rows if r['experiment']=='E3R')
    for experiment,arms in [('E3',{'Rebuild','D0C1','D1C1'}),('E4',{'D1Cnone','D1CviewLRU','D1C0','D1Clogical','D1C1'})]:
        blocks={}
        for row in rows:
            if row['experiment']!=experiment:continue
            key=tuple(json.dumps(row[k],sort_keys=True) for k in ('datasets','partitions','rho','locality','old_readers','beta','order'))
            blocks.setdefault(key,set()).add(row['arm'])
        assert blocks and all(value==arms for value in blocks.values())


def test_rebuild_environment_is_explicit_and_other_routes_share_default():
    assert protocol.arm_environment('Rebuild')['EGMGPU_SHARED_DEVICE_CACHE_BYTES']=='0'
    assert len({json.dumps(protocol.arm_environment(arm),sort_keys=True) for arm in protocol.ARMS if arm!='Rebuild'})==1
    assert {'state_controls.py','state_rebuild.py'}<=set(contract.BENCHMARK_FILES)


def test_formal_templates_cannot_be_submitted_as_queue():
    text=Path(state_plan.__file__).read_text()
    assert "job_specs=formal, jobs=[]" in text
    assert "pending['job_specs']" in text


def test_actual_plan_builder_preserves_480_cells_and_664_qualifications(tmp_path,monkeypatch):
    # Exercise the real builder with local immutable CPU fixtures. Deployment
    # and canonical-source validation are tested separately, never bypassed in
    # produced plans or the real guard.
    runner=Path(__file__).resolve().parent
    tiny=tmp_path/'tiny.json'
    tiny.write_text(json.dumps({'datasets':{'TINY_A':{},'TINY_B':{},'TINY_C':{}}}))
    freeze=tmp_path/'source.json';freeze.write_text('{}')
    registry=tmp_path/'registry.json';registry.write_text('{}')
    dso=tmp_path/'cpu-dso-placeholder';dso.write_bytes(b'CPU fixture, never imported')
    dep=dict(source_tree_sha256='f'*64,python_invocation=sys.executable,
        source_root=str(tmp_path/'source'),native_dso=state_plan.bind(dso),
        environment={},timeout_seconds=3600,
        resources={str(p):{'gpu-uuids':','.join(f'GPU-fixture-{n}' for n in range(p)),
            'cpu-set':','.join(map(str,range(16))),'lock':str(tmp_path/'gpu.lock'),
            'tiny_inputs':state_plan.bind(tiny)} for p in (1,8)})
    monkeypatch.setattr(state_plan,'deployment',lambda args:dep)
    monkeypatch.setattr(state_plan,'validate_input_metadata',lambda *args,**kwargs:None)
    unique={}
    for row in state_plan.grid():
        key=state_plan.input_key(row)
        if key in unique:continue
        path=tmp_path/f'input-{len(unique)}.json'
        input_row=dict(row,experiment=key[0])
        path.write_text(json.dumps({'cell':input_row,'datasets':{n:{} for n in row['datasets']}}))
        unique[key]=path
    args=SimpleNamespace(output=tmp_path/'plan',results_root=tmp_path/'results',
        freeze=freeze,registry=registry,runner=runner,inputs=list(unique.values()),
        enable_p8_targets=True,scope='available',grid_domain='main')
    result=state_plan.build(args)
    assert result['configurations']==480 and result['qualification_jobs']==664
    assert result['pending_formal_jobs']==4800
    formal=state_plan.read(args.output/'formal_pending.json')
    assert formal['jobs']==[] and len(formal['job_specs'])==4800
    qualification=state_plan.read(args.output/'qualification_plan.json')
    assert len(qualification['jobs'])==664
    from state_worker import parser
    readonly_blocks={}
    for file in (args.output/'jobs').glob('*.json'):
        spec=state_plan.read(file)
        options=parser().parse_args(spec['command'][3:])
        assert options.experiment==spec['experiment'] and options.arm==spec['arm']
        assert options.repeat==spec['repeat']
        if spec['experiment']=='E3R':
            assert '--pilot' not in spec['command']
            if spec['phase']=='formal':
                readonly_blocks.setdefault(spec['paired_block_sha256'],set()).add((spec['order'],spec['arm']))
    assert len(readonly_blocks)==50
    assert all(len(variants)==4 for variants in readonly_blocks.values())
