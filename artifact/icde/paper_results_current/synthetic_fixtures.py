"""Explicit synthetic test evidence only. These objects are not GPU observations."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import normalize as n
import protocol


def bind(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + '\n')
    return bind(path)


def registry(root):
    records = {}
    for graph in n.DATASETS:
        folder = root / graph
        files = {'canonical': write(folder/'canonical.json', dict(num_nodes=40, num_entries=0, directed=False)),
                 'sources': write(folder/'sources.json', dict(primary_sources=[0, 1, 2]))}
        for name, data in (('offsets', np.zeros(41, dtype='<i4')), ('indices', np.array([], dtype='<i4')), ('weights', np.array([], dtype='<f8'))):
            path = folder/(name+'.bin'); data.tofile(path); files[name] = bind(path)
        receipt = dict(status='PASS_CANONICAL_INPUT', synthetic_fixture=True, files=files)
        item = write(folder/'receipt.json', receipt)
        records[graph] = dict(receipt=receipt, receipt_path=item['path'], receipt_sha256=item['sha256'])
    return write(root/'registry.json', dict(status='ALL_SEVEN_CANONICAL_READY', records=records, synthetic_fixture=True))


def e6_selection(root):
    """45 configurations × 10 repeats, all explicitly synthetic, real hash chains."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    input_registry = registry(root/'inputs')
    e = n.core.Evidence(root, []); domains = n.core.input_domains(e, input_registry)
    source = root/'easygraph/synthetic.py'; source.parent.mkdir(); source.write_text('SYNTHETIC_ONLY = True\n')
    source_files = {'easygraph/synthetic.py': {k: bind(source)[k] for k in ('bytes', 'sha256')}}
    freeze = write(root/'freeze.json', dict(source_files=source_files, source_tree_sha256=n.digest(source_files)))
    outputs = {}
    for name in (*n.E6_FUNCTIONS, 'candidates', 'winner'):
        length = 32 if name in ('candidates', 'closeness') else 1 if name == 'winner' else 40
        dtype = np.int64 if name in ('candidates', 'winner') else np.float64
        path = root/(name+'.npy'); np.save(path, np.arange(length, dtype=dtype)); outputs[name] = bind(path)
    vectors = {}
    for name, item in outputs.items():
        value = np.load(item['path'])
        vectors[name] = dict(shape=list(value.shape), dtype=str(value.dtype), owning=True, immutable=True, **{'pass': True})
        if name in n.E6_FUNCTIONS[:3]:
            vectors[name].update(nan_mask_equal=True, finite_mask_equal=True, rmse_finite=0., max_absolute_error_finite=0.)
    quality = dict(vectors=vectors, actual_candidate_selection=True, actual_winner_selection=True, bfs_wcc_membership_exact=True, **{'pass': True})
    files_e6 = {name: bind(protocol.CAMPAIGN/'e6_protocol_completion_v2'/name) for name in
                ('e6_worker.py', 'e6_reused.py', 'sg_worker.py', 'e6_cpu_reference.py', 'legacy_cpu_reference.py', 'e6_reuse_provenance.json', 'legacy_contract.json')}
    cells = []
    for index, d in enumerate(protocol.e6_cells()):
        folder = root/f'cell{index:02d}'
        epoch = dict(protocol=protocol.E6_PROTOCOL, contract=protocol.E6_CONTRACT, files_e6=files_e6,
                     release_sources={'easygraph/synthetic.py': bind(source)}, **{k: d[k] for k in ('system', 'ablate', 'coverage_backend', 'gpu_count')})
        eb = write(folder/'epoch.json', epoch)
        stages = []
        for j, function in enumerate(n.E6_FUNCTIONS):
            fallback = function in protocol.E6_GROUPS.get(d['ablate'], ())
            stages.append(dict(function=function, start_seconds=float(j), end_seconds=float(j+1), complete_seconds=1.,
                               metrics=dict(devices_used=protocol.e6_expected_devices(d, function), coverage_fallback=fallback,
                                            coverage_backend=d['coverage_backend'] if fallback else 'eggpu')))
        calls = {label: dict(seconds=6., timed=label!='warmup', quality=copy.deepcopy(quality), outputs=outputs,
                            trace=dict(stages=stages, candidate_seconds=.1, winner_seconds=.1)) for label in ('first', 'warmup', 'ready')}
        child = dict(status='PASS', quality_pass=True, outputs_survive_close=True, input_sha256=domains[d['dataset']]['sha256'],
                     weak_projection=dict(n=40, source_vertices_preserved=True, weights_used=False), calls=calls)
        cb = write(folder/'child.json', child)
        base = dict(schema='eggpu.e6.job/coverage2x2-v2', protocol=protocol.E6_PROTOCOL, contract=protocol.E6_CONTRACT,
                    **{k:d[k] for k in ('dataset','system','ablate','coverage_backend','gpu_count')},
                    epoch=eb, epoch_sha256=n.digest(epoch), input_sha256=child['input_sha256'], quality_pass=True,
                    child=cb, T_first_seconds=6., T_ready_seconds=6., synthetic_fixture=True)
        proofs = {phase: write(folder/(phase+'.json'), dict(base, repeat=0, dataset='E6_TINY_U' if phase=='tiny' else d['dataset'],
                           phase=phase, status='PASS_QUALIFICATION', performance_eligible=False)) for phase in ('tiny','target')}
        runs = [write(folder/f'r{repeat:02d}.json', dict(base, phase='formal', status='PASS', performance_eligible=True,
                      repeat=repeat, tiny_proof=proofs['tiny'], target_proof=proofs['target'])) for repeat in range(10)]
        cells.append(dict(d, id=f'E6_synthetic_{index:02d}', adapter='e6', role='eggpu' if d['system']=='eggpu' else 'external',
                          epoch=eb, epoch_sha256=n.digest(epoch), runs=runs))
    selection = dict(schema='eggpu.paper.selection.v1', export_protocol='paper-results-protocol-completion-v1',
                     status='FROZEN_EXPLICIT_SELECTION', mode='formal', synthetic_fixture=True,
                     final_eggpu_release=freeze, input_registry=input_registry, required_cells=[x['id'] for x in cells], cells=cells)
    path = root/'selection.json'; write(path, selection)
    return path, selection


def state_payload(root, experiment='E3R', arm='D1C1'):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    inputs = write(root/'inputs.json', dict(schema='state.inputs.v2', qualification_only=True, partitions=1, datasets={'US':{'n':5}}))
    trace = n.state_trace(experiment, ['US'], 'interleaved', tiny=True)
    values = np.zeros(5, dtype=np.float64); path=root/'output.npz'; np.savez_compressed(path, values=values)
    sha = hashlib.sha256(memoryview(values).cast('B')).hexdigest()
    output = dict(blob=bind(path), dtype=values.dtype.str, shape=[5], vector_sha256=sha, lossless=True, format='npz:values')
    route, policy = protocol.STATE.ARMS[arm]
    events=[]; checks=[]
    for event in trace:
        measured=dict(event)
        if event['kind']=='query':
            resource=dict(state_switch_policy=policy, controller_version='state_switch_v2_closure_lru', actions=[],
                          after_retention={'per_device':[dict(device_id=0, managed_resident_bytes=0)]})
            measured['details']=dict(semantic_snapshot=event['snapshot'], metrics=dict(shared_array_new_bytes=5,
                native_dynamic_patch=dict(route=route, retirement_policy='lineage', per_device_delta={'0':dict(weight_full_h2d_bytes=1, patch_h2d_bytes=2, cow_d2d_bytes=2)}),
                resource_management=resource))
            quality=dict(passed=True, vertices=5, exact=True, partition_equivalent=True, residual_l1=0., mass_error=0., reference_l1=0., reference={'independent':True})
            checks.append(dict(event=event, output=output, output_sha256=sha, survives_session_close=True, quality=quality))
        elif arm=='Rebuild':
            measured['details']=dict(kind='canonical_host_partition_and_device_rebuild', rebuild_generation=event['version'],
                 release=dict(prediction_error_bytes=0,zero_release_is_capacity_progress=False,freed_bytes=3,capacity_progress=True),
                 host_reconstruction_seconds=.01, host_payload_written_bytes=40)
        events.append(measured)
    raw=dict(inputs=inputs, datasets=['US'], experiment=experiment, arm=arm, old_readers='none', execution_complete=True,
             quality_pass=True, continuous_interval=True, sum_of_phases_used_as_total=False, events=events, checks=checks,
             first_complete_workflow_seconds=2., workflow_complete_host_seconds=1., budget=None,
             rebuild_control=arm=='Rebuild',rebuild_temporary_inputs_removed_after_session_close=arm=='Rebuild')
    config=dict(protocol=protocol.STATE_PROTOCOL, experiment=experiment, arm=arm, old_readers='none', gpu_uuids=['SYNTHETIC_DEVICE'],
                order='interleaved',route=route, policy=policy,
                rebuild_scope='canonical host partitions, file-manifest import, full device state; catalog disabled')
    return raw, config


def e4s_archive(root):
    """One synthetic batch with a complete content-bound formal recovery chain."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    helper=bind(protocol.CAMPAIGN/'e4s_recoverable_outputs_v1/archive_outputs.py')
    wrapper=bind(protocol.CAMPAIGN/'e4s_recoverable_outputs_v1/maintain_e4s_outputs.py')
    reference=bind(protocol.CAMPAIGN/'state_protocol_completion_v1/state_reference.py')
    values=np.array([0.,1.,np.inf],dtype='<f8'); sha=hashlib.sha256(memoryview(values).cast('B')).hexdigest()
    path=root/'values.npz';np.savez_compressed(path,reference=values);obj=bind(path)
    key=dict(schema='state.reference.v2',function='sssp',version=0,source=0,n=3,
             reference_source={k:reference[k] for k in ('bytes','sha256')})
    metadata=write(root/'metadata.json',dict(key=key,array={k:obj[k] for k in ('bytes','sha256')}))
    check=dict(sequence=0,source=0,values=3,output_sha256=sha,survives_session_close=True,
               quality=dict(passed=True,exact=True,vertices=3,reference=dict(independent=True,key_sha256=n.digest(key))))
    raw=dict(schema='e4s.complete.batch.v1',phase='formal',status='PASS_E4S_FORMAL',performance_eligible=True,quality_pass=True,
             epoch_sha256='SYNTHETIC_EPOCH',input_sha256='SYNTHETIC_INPUT',source_sequence=[0],checks=[check],batch=1,
             dataset='SYNTHETIC',variant='eggpu',repeat=0,reference_implementation=reference,
             measurements=dict(first_complete_seconds=2.,all_outputs_retained_through_close=True))
    result=write(root/'result.json',raw)
    vector=dict(sequence=0,source=0,values=3,byte_identical=True,dtype='<f8',order='C',restored_raw_sha256=sha,
                measured_output_sha256=sha,object=obj,reference_metadata=metadata,reference_key_sha256=n.digest(key))
    sidecar=dict(schema='e4s.recoverable.output.v1',status='PASS_BYTE_IDENTICAL_RECOVERABLE_E4S_OUTPUTS',original_result=result,
                 source=helper,post_measurement=True,measured_intervals_unchanged=True,epoch_sha256=raw['epoch_sha256'],
                 input_sha256=raw['input_sha256'],source_sequence=[0],vectors=[vector])
    archive=write(root/'archive.json',sidecar)
    original=write(root/'original_maintenance.json',dict(status='PASS_SYNTHETIC_MAINTENANCE'))
    spec=write(root/'spec.json',dict(result=result['path'],archive_helper=helper,bindings=[wrapper,helper],
               original_maintenance=dict(receipt=original['path'],expected_statuses=['PASS_SYNTHETIC_MAINTENANCE'])))
    post=dict(status='PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE',archive_status='PASS',original_maintenance_status='PASS',
              original_result=result,archive_receipt=archive,attempt_spec=spec,source=wrapper,original_maintenance_receipt=original)
    maintenance=write(root/'maintenance.json',post)
    cell=dict(archive_helper=helper,recoverable_outputs=[dict(result=result,archive=archive,maintenance=maintenance)])
    return raw,cell,sidecar,post
