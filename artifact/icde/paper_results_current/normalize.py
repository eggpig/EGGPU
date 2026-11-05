"""Explicit source-bound E2--E6 receipts to auditable per-repetition metrics.

Reads only explicitly selected bindings. No latest-run search, winner selection,
or qualification-to-formal promotion. All stored output bytes are verified.
E4-S requires byte-identical recoverable outputs and formal maintenance proof.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import random
import statistics
import protocol

_path = Path(__file__).resolve().parents[1] / 'presentation' / 'e1_summary.py'
_spec = importlib.util.spec_from_file_location('_paper_frozen_e1_evidence', _path)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
need, digest = core.require, core.digest
DATASETS = core.DATASETS
E2_GRAPHS = ('US', 'OR', 'IN', 'UK', 'TW')
FUNCTIONS = ('bfs', 'sssp', 'wcc', 'weighted_pr')
E6_FUNCTIONS = ('effective_size','constraint','closeness','bfs','wcc')
FAILURES = core.FAILURES | {'FAIL_STATE_WORKFLOW','FAIL_UNWEIGHTED_PR_CONTROL','MISSING_RECEIPT','SKIPPED_DEPENDENCY'}
WEIGHT_METRICS = ('weight_full_h2d_bytes', 'patch_h2d_bytes', 'cow_d2d_bytes')


def number(value, *, positive=False):
    need(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
         and value >= 0 and (not positive or value > 0), 'Invalid timing/byte value')
    return float(value)


def recursively_verify(e, value):
    if isinstance(value,dict):
        if {'path','sha256'} <= value.keys():
            e.verify(value)
        else:
            for child in value.values():
                recursively_verify(e,child)
    elif isinstance(value,list):
        for child in value:
            recursively_verify(e,child)


def compare_release(observed, freeze, require_complete=True):
    expected=freeze['source_files']
    for name,item in observed.items():
        need(name in expected and {k:item[k] for k in ('bytes','sha256')}==expected[name],
             'Actual EGGPU source differs from selected final release: '+name)
    required={name for name in expected if name.startswith(('easygraph/','cpp_easygraph/','gpu_easygraph/'))
              and Path(name).suffix in ('.py','.cpp','.h','.hpp','.c','.cc','.cu','.cuh')}
    if require_complete:need(required<=set(observed),'Incomplete final-source coverage')


def bind_identity(e, cell, raw, freeze):
    if cell['adapter'] in ('state','pr_control'):
        selected=e.read(cell['configuration_receipt'])
        config=selected['configuration']
        need(digest(config)==selected['configuration_sha256']==cell['configuration_sha256'], 'Selected configuration digest differs')
        need(raw.get('configuration')==config and raw.get('configuration_sha256')==cell['configuration_sha256'], 'Mixed state/control configurations')
        recursively_verify(e,config)
        compare_release(config['source_binding']['files'],freeze)
        if cell['adapter']=='state' and cell['experiment'] in ('E3','E4','E3R'):
            benchmark=config['source_binding']['benchmark']
            for name in ('state_worker.py','state_contract.py','state_protocol.py','state_reference.py','state_inputs.py','state_controls.py','state_rebuild.py','baselines/hytgraph_qualify.py'):
                need({k:benchmark[name][k] for k in ('bytes','sha256')}==protocol.PROVENANCE['state_protocol_completion_v1/'+name],
                     'State benchmark producer differs from the current protocol: '+name)
        return config
    epoch=e.read(cell['epoch'])
    need(digest(epoch)==cell['epoch_sha256']==raw.get('epoch_sha256'), 'Mixed or unbound source/runner epoch')
    need(e.read(raw['epoch'])==epoch,'Receipt epoch bytes differ')
    recursively_verify(e,epoch)
    if cell['role']=='eggpu':need('release_sources' in epoch,'EGGPU epoch lacks complete source binding')
    if 'release_sources' in epoch:
        observed=dict(epoch['release_sources'])
        for name,item in epoch.get('gpu_kernel_sources',{}).items():
            need(name not in observed or observed[name]==item,'Contradictory native kernel source binding')
            observed[name]=item
        compare_release(observed,freeze,require_complete=cell['role']=='eggpu')
    return epoch


def npz_output(e,item):
    import numpy as np
    path=e.verify(item['blob'])
    with np.load(path,allow_pickle=False) as archive:
        need(archive.files==['values'],'Unexpected state output archive members')
        values=archive['values']
    need(values.dtype==np.dtype('<f8'),'State output must preserve full FP64 values')
    need(values.dtype.str==item['dtype'] and list(values.shape)==item['shape'],'Stored full-vector domain differs')
    need(hashlib.sha256(memoryview(values).cast('B')).hexdigest()==item['vector_sha256'],'Stored full vector bytes differ')
    return len(values)


def state_trace(experiment,names,order,tiny=False):
    if experiment in ('E3','E4','E3R'):
        return protocol.STATE.trace(experiment,names,order,qualification=tiny)
    rounds=2 if tiny else 20 if experiment=='E3' else 8
    visits=[(name,r) for r in range(rounds) for name in names]
    if order=='grouped':visits=[(name,r) for name in names for r in range(rounds)]
    elif order=='permuted':
        need(experiment=='E4','Version-changing trace cannot permute')
        random.Random(20261104).shuffle(visits)
    elif order=='once':
        need(experiment=='E4','Only E4 has once control')
        visits=[(name,0) for name in names]
    else:need(order=='interleaved','Unknown trace ordering')
    versions={name:0 for name in names};events=[]
    for name,r in visits:
        if experiment!='E4':
            old=versions[name];versions[name]+=1
            events.append(dict(kind='commit',dataset=name,base=f'{name}@{old}',snapshot=f'{name}@{old+1}',version=old+1))
        for function in FUNCTIONS:
            events.append(dict(kind='query',dataset=name,snapshot=f'{name}@{versions[name]}',function=function,source_index=r%3,version=versions[name]))
    return events


def validate_state_input(e, raw, domains, *, tiny=False):
    inputs=e.read(raw['inputs'])
    need(inputs['schema']=='state.inputs.v2' and inputs['qualification_only']==tiny,'Wrong state input domain')
    recursively_verify(e,inputs)
    if not tiny:
        for name,row in inputs['datasets'].items():
            need(name in domains and row['n']==domains[name]['n'],'Wrong state canonical graph')
            canonical=domains[name]['identity']['files']
            for key,item in row['canonical'].items():
                alias='canonical' if key=='metadata' else key
                need({k:item[k] for k in ('bytes','sha256')}==canonical[alias],'State input differs from selected seven-graph registry')
            need(row['sources']==domains[name]['sources'],'State source rotation differs')
    return inputs


def state_payload(e,raw,config,domains,*,tiny=False):
    if raw['experiment'] in ('E3','E4','E3R'):
        need(config.get('protocol')==protocol.STATE_PROTOCOL,'Old state protocol cannot supply current experiments')
    inputs=validate_state_input(e,raw,domains,tiny=tiny)
    names=raw['datasets'];ex=raw['experiment']
    need(ex==config['experiment'] and raw['arm']==config['arm'] and raw['old_readers']==config['old_readers'],'State treatment identity differs')
    need(inputs['partitions']==len(config['gpu_uuids']),'State partitions/resource identity differs')
    expected=state_trace(ex,names,config['order'],tiny)
    need(raw.get('execution_complete') is True and raw.get('quality_pass') is True and
         raw.get('continuous_interval') is True and raw.get('sum_of_phases_used_as_total') is False,'State workflow is incomplete or total is reconstructed')
    need(len(raw['events'])==len(expected),'State event count differs')
    totals={name:0. for name in WEIGHT_METRICS}
    totals.update(shared_array_new_bytes=0.,freed_bytes=0.,commit_rebuild_freed_bytes=0.)
    if config['arm']=='Rebuild':
        need(raw.get('rebuild_control') is True and raw.get('rebuild_temporary_inputs_removed_after_session_close') is True,
             'Rebuild was not executed or its temporary inputs survived Session close')
        need(config.get('rebuild_scope')=='canonical host partitions, file-manifest import, full device state; catalog disabled',
             'Rebuild scope differs from the actual full rebuild control')
    for event,measured in zip(expected,raw['events']):
        need(all(measured.get(k)==v for k,v in event.items()),'State trace order/semantics differ')
        if event['kind']!='query':
            if config['arm']=='Rebuild':
                details=measured['details'];release=details['release']
                need(details.get('kind')=='canonical_host_partition_and_device_rebuild' and details['rebuild_generation']==event['version'],
                     'Rebuild commit omitted actual canonical host reconstruction')
                need(release['prediction_error_bytes']==0 and release.get('zero_release_is_capacity_progress') is False,
                     'Rebuild physical release evidence differs')
                freed=number(release['freed_bytes'])
                need(release['capacity_progress']==(freed>0),'Rebuild zero-byte release counted as progress')
                totals['commit_rebuild_freed_bytes']+=freed
                number(details['host_reconstruction_seconds']);number(details['host_payload_written_bytes'])
            continue
        metrics=measured['details']['metrics'];dynamic=metrics['native_dynamic_patch'];resource=metrics['resource_management']
        need(dynamic['route']==config['route'] and resource['state_switch_policy']==config['policy'],'Measured D/C intervention differs')
        need(dynamic['retirement_policy']=='lineage' and resource['controller_version']=='state_switch_v2_closure_lru',
             'Different maintenance retirement or weaker C0 control')
        if raw['experiment'] in ('E3','E4','E3R'):
            need(measured['details'].get('semantic_snapshot')==event['snapshot'],'Query semantic graph version differs')
        if config['policy']=='no_retention':
            need(all(row['managed_resident_bytes']==0 for row in resource['after_retention']['per_device']),
                 'No-retention left live managed allocation bytes')
        for row in dynamic['per_device_delta'].values():
            if config['route']=='persistent_full':
                need(not any(row.get(k,0) for k in ('patch_packets','lease_reuses','cow_copies')),
                     'D0 unexpectedly executed D1 maintenance')
            for key in WEIGHT_METRICS:totals[key]+=number(row.get(key,0))
        need('shared_array_new_bytes' in metrics,'Missing allocation counter is not measured zero')
        totals['shared_array_new_bytes']+=number(metrics['shared_array_new_bytes'])
        for action in resource['actions']:
            need(action.get('prediction_error_bytes',0)==0,'Physical closure accounting mismatch')
            freed=number(action['freed_bytes'])
            need(freed>0 or action.get('capacity_progress') is not True,'Zero physical release counted as capacity progress')
            totals['freed_bytes']+=freed
        if raw.get('budget'):
            for device in resource['after_retention']['per_device']:
                need(device['managed_resident_bytes']<=raw['budget']['per_device'][str(device['device_id'])],'Retained physical budget exceeded')
    queries=[x for x in expected if x['kind']=='query']
    need(len(raw['checks'])==len(queries),'Incomplete full output audit')
    for event,check in zip(queries,raw['checks']):
        need(check['event']==event and check['survives_session_close'] and check['quality']['passed'],'Bad state output or lifetime')
        need(check['output_sha256']==check['output']['vector_sha256'] and check['output'].get('lossless') is True and
             check['output'].get('format')=='npz:values','State complete output archive/hash differs')
        need(npz_output(e,check['output'])==inputs['datasets'][event['dataset']]['n'],'Incomplete state output length')
        q=check['quality'];need(q.get('vertices')==inputs['datasets'][event['dataset']]['n'],'Incomplete quality vector domain')
        need(q.get('reference',{}).get('independent') is True,'Missing independent state reference provenance')
        if event['function'] in ('bfs','sssp'):need(q.get('exact') is True,'Traversal full vector is not exact')
        if event['function']=='wcc':need(q.get('partition_equivalent') is True,'WCC partition audit failed')
        if event['function']=='weighted_pr':
            q=check['quality']
            need(number(q['residual_l1'])<=1e-6 and number(q['mass_error'])<=1e-6 and number(q['reference_l1'])<=1e-5,'WPR common quality differs')
    if config['old_readers']=='per_update':
        need(len(raw['leases'])==sum(x['kind']=='commit' for x in expected),'Missing real old-reader lifetime')
        need(all(x['released'] and (not tiny or x['old_weights_bitwise_unchanged']) for x in raw['leases']),'Old reader was not protected/released')
    if raw.get('budget'):
        budget=raw['budget'];pilot=e.read(budget['pilot']);recursively_verify(e,pilot['source_binding']);e.verify(pilot['inputs'])
        need(pilot['status']=='PASS_PHYSICAL_BYTE_PILOT' and pilot['selection_uses_timing'] is False and
             pilot['source_binding']==config['source_binding'] and pilot['inputs']==raw['inputs'] and
             pilot['gpu_uuids']==config['gpu_uuids'],'Wrong actual physical-byte pilot')
        for device in pilot['devices']:
            values=[r[str(device)] for r in pilot['resident_view_bytes'].values()]
            need(budget['per_device'][str(device)]==math.floor(max(values)+config['beta']*(sum(values)-max(values))),'Budget differs from physical-byte rule')
    return dict(first=number(raw['first_complete_workflow_seconds'],positive=True),
        workflow=number(raw['workflow_complete_host_seconds'],positive=True),
        maintenance_h2d_bytes=totals['weight_full_h2d_bytes']+totals['patch_h2d_bytes'],
        cow_d2d_bytes=totals['cow_d2d_bytes'],shared_array_new_bytes=totals['shared_array_new_bytes'],
        freed_bytes=totals['freed_bytes'],commit_rebuild_freed_bytes=totals['commit_rebuild_freed_bytes']), inputs


def state_run(e,raw,cell,config,domains,mode):
    need(raw['schema']=='state.complete.workflow.v4' and raw['experiment']==cell['experiment'],'Wrong state schema/experiment')
    need(raw['datasets']==cell['datasets'],'Wrong graph pool/order')
    for key in ('arm','old_readers','order','beta'):
        need(config[key]==cell[key],'State cell dimension differs: '+key)
    need(len(config['gpu_uuids'])==cell['gpu_count'],'Wrong resource count')
    need(raw['inputs']==cell['inputs'],'Exact selected input/positions differ')
    metrics,inputs=state_payload(e,raw,config,domains,tiny=False)
    for row in inputs['datasets'].values():
        need(row['requested_rho']==cell['rho'] and row['locality']==cell['locality'],'Density/locality differs')
    if mode=='formal':
        need(raw['status']=='PASS_STATE_COMPLETE_WORKFLOW' and raw['phase']=='formal' and raw['performance_eligible'] is True,'State qualification is not formal')
        for phase in ('tiny','target'):
            proof=e.read(raw['qualification'][phase]);need(proof['phase']==phase and proof['status']=='PASS_STATE_GPU_QUALIFICATION','Wrong state proof phase')
            need(proof['configuration']==config and proof['configuration_sha256']==digest(config),'State proof epoch differs')
            if phase=='target':need(proof['inputs']==raw['inputs'] and proof['datasets']==raw['datasets'] and proof['budget']==raw['budget'],'State target scope differs')
            state_payload(e,proof,config,domains,tiny=phase=='tiny')
    else:need(raw['status']=='PASS_STATE_GPU_QUALIFICATION','Expected qualification only')
    return metrics, 'complete_archived_vectors_verified'


def native_payload(e,raw,cell,domain,*,tiny=False):
    algorithm=cell['algorithm'];schema=raw['schema']
    rows=raw['sources'];expected=domain['sources'] if algorithm in ('bfs','sssp') else [None]
    need(len(rows)==len(expected),'Incomplete three-source/native task')
    if not tiny and algorithm in ('bfs','sssp'):need([x['source'] for x in rows]==expected,'Source IDs or order differ')
    collected=[]
    for item in rows:
        child=core.source_result(e,item)
        need(child.get('source')==item.get('source'),'Native child source differs')
        timings=core.read_source(e,child,schema,algorithm)
        if cell['role']=='eggpu':
            need(child['partition_audit']['status']=='PASS_FULL_PARTITION_ROUNDTRIP','Native full partition audit missing')
            e.verify(child['preparation']['manifest'])
            for label in ('first','warmup','ready'):
                need(child['calls'][label]['native_metrics']['devices_used']==list(range(cell['gpu_count'])),'Native query used different devices')
        collected.append(timings)
    metrics={metric:statistics.mean(x[metric] for x in collected) for metric in ('first','ready')}
    for metric,value in metrics.items():
        if 'T_'+metric+'_seconds' in raw:need(math.isclose(value,raw['T_'+metric+'_seconds'],rel_tol=1e-12),'Stored source mean differs')
    return metrics


def native_run(e,raw,cell,epoch,domains,mode):
    need(raw.get('experiment','E2')=='E2' and cell['experiment']=='E2','Strong-scaling panel cannot consume optimized E1-SG')
    need(raw.get('system')==cell['system'] and raw.get('algorithm',raw.get('function'))==cell['algorithm'] and
         raw.get('gpu_count',raw.get('count'))==cell['gpu_count'] and raw['dataset']==cell['dataset'],'Native E2 cell differs')
    if cell['role']=='eggpu':need(raw.get('implementation')=='forced_native','E2 requires native P1/P2/P4/P8 fixed route')
    domain=domains[cell['dataset']]
    need(raw['input_sha256']==domain['sha256'] and raw['directed']==domain['directed'],'E2 canonical identity differs')
    metrics=native_payload(e,raw,cell,domain)
    if mode=='formal':
        need(raw['status'] in ('PASS','PASS_FORMAL') and raw.get('performance_eligible',raw.get('formal')) is True,'Nonformal native result')
        for phase in ('tiny','target'):
            proof=e.read(raw.get(phase+'_proof',raw.get('qualification',{}).get(phase)))
            need(proof['phase']==phase and proof['status']=='PASS_QUALIFICATION' and proof['quality_pass'] and
                 proof['epoch_sha256']==cell['epoch_sha256'] and e.read(proof['epoch'])==epoch,'Native qualification differs')
            need(proof.get('gpu_count',proof.get('count'))==cell['gpu_count'] and proof.get('algorithm',proof.get('function'))==cell['algorithm'],'Native proof resources/function differ')
            need(proof.get('system')==cell['system'] and proof['directed']==domain['directed'],'Native proof system/directedness differs')
            if phase=='target':need(proof['input_sha256']==domain['sha256'],'Native target input differs')
            native_payload(e,proof,cell,domain,tiny=phase=='tiny')
    else:need(raw['status']=='PASS_QUALIFICATION','Expected native target qualification')
    return metrics,'complete_archived_vectors_verified'


def e6_payload(e,raw,cell,vertices=None):
    import numpy as np
    child=e.read(raw['child'])
    need(child['status']=='PASS' and child['quality_pass'] and child['outputs_survive_close'],'Bad E6 full output/lifetime')
    need(child['input_sha256']==raw['input_sha256'] and set(child['calls'])=={'first','warmup','ready'},'Incomplete E6 calls')
    n=child['weak_projection']['n']
    need(child['weak_projection']['source_vertices_preserved'] and not child['weak_projection']['weights_used'],
         'E6 must preserve every source vertex in the unweighted weak projection')
    if vertices is not None:need(n==vertices,'E6 canonical vertex domain differs')
    metrics={}
    for label,call in child['calls'].items():
        need(call['quality']['pass'] and set(call['outputs'])==set(E6_FUNCTIONS)|{'candidates','winner'},'Incomplete E6 stages/outputs')
        need(call.get('timed') is (label!='warmup'),'E6 warmup/measurement boundary differs')
        quality=call['quality']
        need(set(quality.get('vectors',{}))==set(call['outputs']) and all(quality.get(key) is True for key in
             ('actual_candidate_selection','actual_winner_selection','bfs_wcc_membership_exact')),
             'E6 complete dependent selection/lifetime audit missing')
        for name,output in call['outputs'].items():
            value=np.load(e.verify(output),mmap_mode='r',allow_pickle=False)
            expected=32 if name in ('candidates','closeness') else 1 if name=='winner' else n
            need(value.shape==(expected,) and value.dtype.kind in 'iuf','Incomplete E6 stage output domain')
            q=quality['vectors'][name]
            need(q.get('pass') is True and q.get('owning') is True and q.get('immutable') is True and
                 q.get('shape')==list(value.shape) and q.get('dtype')==str(value.dtype),'E6 owning output audit differs from stored vector')
            if name in E6_FUNCTIONS[:3]:
                need(value.dtype==np.float64 and q.get('nan_mask_equal') is True and q.get('finite_mask_equal') is True,
                     'E6 FP64 finite/NaN quality contract differs')
                number(q['rmse_finite']);number(q['max_absolute_error_finite'])
        stages=call['trace']['stages']
        need(tuple(x['function'] for x in stages)==E6_FUNCTIONS,'E6 causal stage ordering differs')
        expected_last=0.
        for stage in stages:
            function=stage['function'];system=cell['system']
            fallback=protocol.E6_GROUPS.get(cell['ablate'],())
            devices=protocol.e6_expected_devices(cell,function)
            need(stage['metrics']['devices_used']==devices,'E6 actual device composition differs')
            if system=='eggpu':
                need(stage['metrics']['coverage_fallback']==(function in fallback),'E6 actual coverage fallback differs')
                need(stage['metrics']['coverage_backend']==(cell['coverage_backend'] if function in fallback else 'eggpu'),
                     'E6 actual fallback implementation differs')
            need(stage['start_seconds']>=expected_last-1e-8 and stage['end_seconds']>=stage['start_seconds'],'E6 stage intervals overlap or reverse')
            need(math.isclose(stage['end_seconds']-stage['start_seconds'],stage['complete_seconds'],rel_tol=1e-8,abs_tol=1e-8),'E6 stage interval differs')
            expected_last=stage['end_seconds']
        if label=='warmup':continue
        total=number(call['seconds'],positive=True)
        stage_values={x['function']:number(x['complete_seconds']) for x in stages}
        ranking=number(call['trace']['candidate_seconds'])+number(call['trace']['winner_seconds'])
        remainder=total-sum(stage_values.values())-ranking
        need(remainder>=-1e-6,'E6 stage sum exceeds continuous total')
        metrics[label]=total
        for key,value in dict(structural=stage_values['effective_size']+stage_values['constraint'],
            ranking=ranking,closeness=stage_values['closeness'],bfs=stage_values['bfs'],wcc=stage_values['wcc'],
            preparation_and_handoff=max(0.,remainder)).items():metrics[label+'__'+key]=value
        need(math.isclose(total,raw['T_'+label+'_seconds'],rel_tol=1e-12),'E6 parent total differs')
    return metrics


def e6_run(e,raw,cell,epoch,domains,mode):
    need(raw['schema']=='eggpu.e6.job/coverage2x2-v2' and raw['protocol']==protocol.E6_PROTOCOL and raw['contract']==protocol.E6_CONTRACT,
         'Wrong E6 complete 2x2 protocol/contract')
    need(raw['dataset']==cell['dataset'] and raw['system']==cell['system'] and raw['ablate']==cell['ablate'] and
         raw['coverage_backend']==cell['coverage_backend'] and raw['gpu_count']==cell['gpu_count'],'Wrong E6 configuration')
    need(epoch.get('protocol')==protocol.E6_PROTOCOL and epoch.get('contract')==protocol.E6_CONTRACT,
         'E6 source epoch uses a different workflow protocol')
    need(all(epoch.get(key)==cell[key] for key in ('system','ablate','coverage_backend','gpu_count')),'E6 epoch composition differs')
    for name in ('e6_worker.py','e6_reused.py','sg_worker.py','e6_cpu_reference.py','legacy_cpu_reference.py','e6_reuse_provenance.json','legacy_contract.json'):
        need({k:epoch['files_e6'][name][k] for k in ('bytes','sha256')}==protocol.PROVENANCE['e6_protocol_completion_v2/'+name],
             'E6 benchmark producer differs from the current protocol: '+name)
    need(raw['input_sha256']==domains[cell['dataset']]['sha256'],'E6 canonical input differs')
    metrics=e6_payload(e,raw,cell,domains[cell['dataset']]['n'])
    if mode=='formal':
        need(raw['status']=='PASS' and raw['phase']=='formal' and raw['performance_eligible'] is True,'E6 qualification is not formal')
        for phase in ('tiny','target'):
            proof=e.read(raw[phase+'_proof'])
            need(proof['status']=='PASS_QUALIFICATION' and proof['phase']==phase and proof['quality_pass'] and
                 proof['epoch_sha256']==cell['epoch_sha256'] and e.read(proof['epoch'])==epoch,'E6 proof epoch differs')
            need(proof.get('protocol')==protocol.E6_PROTOCOL and proof.get('contract')==protocol.E6_CONTRACT,
                 'E6 proof protocol differs')
            need((proof['system'],proof['ablate'],proof['gpu_count'],proof['coverage_backend'])==
                 (cell['system'],cell['ablate'],cell['gpu_count'],cell['coverage_backend']),'E6 proof composition differs')
            if phase=='tiny':need(proof['dataset']=='E6_TINY_U','E6 tiny proof uses another graph')
            if phase=='target':need(proof['input_sha256']==raw['input_sha256'],'E6 target input differs')
            e6_payload(e,proof,cell,None if phase=='tiny' else domains[cell['dataset']]['n'])
    else:need(raw['status']=='PASS_QUALIFICATION','Expected E6 target qualification')
    return metrics,'complete_archived_stage_vectors_verified'


def e4s_payload(e,raw,cell,vertices=None):
    sequence=raw['source_sequence'];need(len(sequence)==cell['batch'] and len(set(sequence))==len(sequence),'Wrong frozen E4-S source batch')
    e.verify(raw['sources']);need(len(raw['checks'])==cell['batch'],'Incomplete E4-S full-vector checks')
    for index,(source,check) in enumerate(zip(sequence,raw['checks'])):
        need(check['sequence']==index and check['source']==source and check['quality']['passed'] and check['quality']['exact'],'E4-S exact full-vector audit differs')
        need(check['values']==check['quality']['vertices'] and check['values']>0,'E4-S incomplete vector domain')
        if vertices is not None:need(check['values']==vertices,'E4-S vector length differs from canonical input')
        need(isinstance(check['output_sha256'],str) and len(check['output_sha256'])==64,'Missing E4-S full-vector hash')
        # The frozen reference cache is explicitly evictable. Its transient
        # filename is not an archival output promise; algorithm/input/version
        # bindings and the full-vector execution audit remain in the epoch.
    measurement=raw['measurements']
    return dict(first=number(measurement['first_complete_seconds'],positive=True),
                ready=number(measurement['query_batch_seconds'],positive=True),
                preparation=number(measurement['adapter_preparation_seconds']))


def validate_grid(cells,mode):
    """Do not let a reduced selection silently redefine complete paper coverage."""
    if mode!='formal':return
    experiments={cell['experiment'] for cell in cells}
    if 'E2' in experiments:
        own=[x for x in cells if x['experiment']=='E2' and x['role']=='eggpu']
        observed=[(x['dataset'],x['algorithm'],x['gpu_count']) for x in own]
        expected={(g,f,p) for g in E2_GRAPHS for f in core.ALGORITHMS for p in (1,2,4,8)}
        need(len(observed)==len(set(observed)) and set(observed)==expected,'E2 fixed native five-graph/four-function/P1/P2/P4/P8 coverage required')
    if experiments & {'E3','E4','E3R','E5'}:
        fields=('experiment','datasets','gpu_count','rho','locality','arm','old_readers','beta','order')
        def key(row):return tuple(tuple(row[k]) if isinstance(row[k],list) else row[k] for k in fields)
        for ex in experiments & {'E3','E4','E3R','E5'}:
            if ex=='E5':
                path=Path(__file__).resolve().parents[1]/'campaign/state_final_plan/state_plan.py'
                spec=importlib.util.spec_from_file_location('_paper_fixed_unchanged_e5_grid',path)
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                grid=[dict(row,gpu_count=row['partitions']) for row in module.grid()]
            else:grid=protocol.state_cells()
            required={key(row) for row in grid if row['experiment']==ex}
            observed=[key(row) for row in cells if row['experiment']==ex]
            need(len(observed)==len(set(observed)) and set(observed)==required,'Incomplete frozen state grid for '+ex)
    if 'E4S' in experiments:
        expected={(g,v,q) for g in ('US','OR','IN','UK') for q in (32,128,512)
                  for v in ('eggpu','c2graph_pm','kgraph_sparse' if g=='US' else 'kgraph_dense')}
        observed=[(x['dataset'],x['variant'],x['batch']) for x in cells if x['experiment']=='E4S']
        need(len(observed)==len(set(observed)) and expected==set(observed),'E4-S fixed three-system batch coverage missing or duplicated')
    if 'E6' in experiments:
        def key(row):return tuple(row[k] for k in ('dataset','system','ablate','coverage_backend','gpu_count'))
        expected={key(row) for row in protocol.e6_cells()}
        observed=[key(x) for x in cells if x['experiment']=='E6']
        need(len(observed)==len(set(observed)) and expected==set(observed),'E6 requires 15 exact configurations per graph, including both 2x2 groups')
    if 'E3_PR_CONTROL' in experiments:
        expected={(g,arm,1) for g in ('US','IN','OR') for arm in ('D0C1','D1C1')}
        observed={(x['dataset'],x['arm'],x['gpu_count']) for x in cells if x['experiment']=='E3_PR_CONTROL'}
        need(expected<=observed,'PR negative control must cover three graphs/two arms')


def raw_cell_identity(raw,cell):
    if cell['adapter']=='state':
        need(raw.get('experiment')==cell['experiment'] and raw.get('datasets')==cell['datasets'] and
             raw.get('arm')==cell['arm'] and raw.get('old_readers')==cell['old_readers'],'Failure/success state cell identity differs')
    else:
        need(raw.get('dataset')==cell['dataset'],'Failure/success dataset identity differs')
        if cell['adapter']=='native':
            need(raw.get('system')==cell['system'] and raw.get('algorithm',raw.get('function'))==cell['algorithm'] and
                 raw.get('gpu_count',raw.get('count'))==cell['gpu_count'],'Failure/success E2 identity differs')
        if cell['adapter']=='e6':
            need((raw.get('system'),raw.get('ablate'),raw.get('gpu_count'))==(cell['system'],cell['ablate'],cell['gpu_count']),'Failure/success E6 identity differs')
            need(raw.get('coverage_backend')==cell['coverage_backend'],'Failure/success E6 coverage backend differs')
        if cell['adapter']=='e4s':
            need((raw.get('variant'),raw.get('batch'))==(cell['variant'],cell['batch']),'Failure/success batch identity differs')
        if cell['adapter']=='pr_control':need(raw.get('arm')==cell['arm'],'Failure/success PR control arm differs')


def e4s_run(e,raw,cell,epoch,domains,mode):
    import archive_gate
    need(raw['schema']=='e4s.complete.batch.v1' and raw['dataset']==cell['dataset'] and
         raw['variant']==cell['variant'] and raw['batch']==cell['batch'],'Wrong E4-S batch task')
    need(raw['input_sha256']==domains[cell['dataset']]['sha256'],'E4-S input differs')
    metrics=e4s_payload(e,raw,cell,domains[cell['dataset']]['n'])
    archive_gate.verify(e,raw,cell,formal=mode=='formal')
    if mode=='formal':
        need(raw['status']=='PASS_E4S_FORMAL' and raw['performance_eligible'],'E4-S result is not formal')
        for phase in ('tiny','target'):
            proof=e.read(raw[phase+'_proof'])
            need(proof['status']=='PASS_E4S_QUALIFICATION' and proof['phase']==phase and proof['quality_pass'] and
                 proof['epoch_sha256']==cell['epoch_sha256'] and e.read(proof['epoch'])==epoch,'E4-S proof epoch differs')
            need(proof['variant']==cell['variant'] and proof['batch']==cell['batch'],'E4-S proof batch differs')
            if phase=='target':need(proof['input_sha256']==raw['input_sha256'] and proof['source_sequence']==raw['source_sequence'],'E4-S target scope differs')
            e4s_payload(e,proof,cell,None if phase=='tiny' else domains[cell['dataset']]['n'])
            archive_gate.verify(e,proof,cell,formal=False)
    else:need(raw['status']=='PASS_E4S_QUALIFICATION','Expected E4-S qualification')
    return metrics,'full_recoverable_batch_decoded_and_verified_with_maintenance'


def pr_control_payload(e,raw,config,domains,tiny=False):
    inputs=validate_state_input(e,raw,domains,tiny=tiny)
    name=raw['dataset'];row=inputs['datasets'][name]
    expected=[dict(kind='query',dataset=name,version=0,function='pr')]
    for version in range(1,21):
        expected += [dict(kind='commit',dataset=name,version=version,base=f'{name}@{version-1}',snapshot=f'{name}@{version}'),
                     dict(kind='query',dataset=name,version=version,function='pr')]
    need(raw['execution_complete'] and raw['quality_pass'] and raw['continuous_interval'] is True and
         raw['sum_of_phases_used_as_total'] is False,'Incomplete PR control workflow')
    need(len(raw['events'])==41 and len(raw['checks'])==21,'Wrong independent control query count')
    h2d=0.
    for exp,actual in zip(expected,raw['events']):
        need(all(actual.get(k)==v for k,v in exp.items()),'PR control trace differs')
        if exp['kind']=='query':
            detail=actual['details'];need(detail['algorithm']=='pr' and detail['execution_snapshot']==row['projections']['bfs']['identity'],'PR control changed to weighted graph')
            counters=detail['dependency_audit']['weighted_view_transfer_counters']
            if exp['version']:
                need(all(number(v)==0 for v in counters.values()),'Weight change caused unweighted GPU maintenance')
                h2d+=number(counters['weight_full_h2d_bytes'])+number(counters['patch_h2d_bytes'])
    for event,check in zip([x for x in expected if x['kind']=='query'],raw['checks']):
        need(check['event']==event and check['survives_session_close'] and check['quality']['passed'],'Bad PR control output')
        need(npz_output(e,check['output'])==row['n'],'Incomplete control vector')
        q=check['quality'];need(q['residual_l1']<=1e-6 and q['mass_error']<=1e-6 and
            q['reference_l1']<=1e-5 and q['invariance_l1']<=2e-8,'PR invariance or quality differs')
    return dict(first=number(raw['first_complete_workflow_seconds'],positive=True),
                workflow=number(raw['workflow_complete_host_seconds'],positive=True),
                maintenance_h2d_bytes=h2d,max_invariance_l1=max(x['quality']['invariance_l1'] for x in raw['checks']))


def pr_control_run(e,raw,cell,config,domains,mode):
    need(raw['schema']=='e3-unweighted-pr-negative-v3-native-device-keys' and config['protocol']==raw['schema'],'Wrong control protocol')
    need(raw['dataset']==cell['dataset'] and raw['arm']==cell['arm'] and raw['inputs']==cell['inputs'] and
         config['old_readers']=='none' and config['policy']=='closure' and len(config['gpu_uuids'])==cell['gpu_count'],'Wrong PR control intervention')
    metrics=pr_control_payload(e,raw,config,domains)
    if mode=='formal':
        need(raw['status']=='PASS_UNWEIGHTED_PR_FORMAL' and raw['performance_eligible'],'PR control qualification is not formal')
        for phase in ('tiny','target'):
            proof=e.read(raw['qualification'][phase])
            need(proof['phase']==phase and proof['status']=='PASS_UNWEIGHTED_PR_QUALIFICATION' and
                 proof['configuration']==config and proof['configuration_sha256']==digest(config),'PR control proof source differs')
            need(proof['inputs']==(config['tiny_inputs'] if phase=='tiny' else raw['inputs']),'PR control proof input differs')
            pr_control_payload(e,proof,config,domains,tiny=phase=='tiny')
    else:need(raw['status']=='PASS_UNWEIGHTED_PR_QUALIFICATION','Expected PR target qualification')
    return metrics,'complete_archived_vectors_verified'


def normalize(selection_path,mode):
    selection_path=Path(selection_path);selection=json.loads(selection_path.read_text())
    need(selection.get('schema')=='eggpu.paper.selection.v1' and selection.get('status')=='FROZEN_EXPLICIT_SELECTION','Explicit frozen selection required')
    need(selection.get('export_protocol')=='paper-results-protocol-completion-v1','Explicit current export protocol required')
    need(selection['mode']==mode and mode in ('formal','qualification'),'Cannot promote qualification')
    e=core.Evidence(selection_path.parent,selection.get('path_remap',[]))
    freeze=e.read(selection['final_eggpu_release']);need(digest(freeze['source_files'])==freeze['source_tree_sha256'],'Release digest differs')
    domains=core.input_domains(e,selection['input_registry'])
    cells=selection['cells'];need(len({x['id'] for x in cells})==len(cells),'Duplicate cell ID')
    need(set(selection['required_cells'])=={x['id'] for x in cells},'Required cell coverage differs from explicit pre-registration')
    validate_grid(cells,mode)
    # The transport may differ by function in the frozen native profile, but
    # one function/count must never select a different epoch by graph.
    native_epochs={};native_binaries=set()
    for cell in cells:
        if cell['adapter']!='native' or cell['role']!='eggpu':continue
        key=(cell['algorithm'],cell['gpu_count'])
        need(key not in native_epochs or native_epochs[key]==cell['epoch_sha256'],
             'Native profile cannot vary by graph')
        native_epochs[key]=cell['epoch_sha256']
        epoch=e.read(cell['epoch'])
        native_binaries.add(digest({k:epoch['files'][k] for k in ('native_worker','native_input','native_dso')}))
    need(len(native_binaries)<=1,'Fixed native scaling cannot mix runner/DSO versions')
    records=[];summaries=[];seen_paths=set()
    adapters={'native':native_run,'state':state_run,'e6':e6_run,'e4s':e4s_run,'pr_control':pr_control_run}
    for cell in cells:
        need(cell['adapter'] in adapters,'Unsupported explicit receipt adapter')
        repeats={};valid=[]
        for binding in cell['runs']:
            logical=binding['path'];need(logical not in seen_paths,'One result cannot count twice');seen_paths.add(logical)
            raw=e.read(binding)
            need(raw.get('synthetic_fixture') is not True or selection.get('synthetic_fixture') is True,
                 'Synthetic measurement cannot be relabeled as real evidence')
            raw_cell_identity(raw,cell)
            identity=bind_identity(e,cell,raw,freeze)
            repeat=raw['repeat'];need(type(repeat) is int and repeat in range(10) and repeat not in repeats,'Duplicate/invalid independent repetition')
            need(raw['phase']==('formal' if mode=='formal' else 'target'),'Wrong selected measurement phase')
            item=dict(cell=cell['id'],repeat=repeat,status=raw['status'],binding=binding,selected=True,valid=False,metrics={})
            repeats[repeat]=item
            if raw['status'] not in FAILURES:
                need(raw.get('quality_pass') is True,'Successful result lacks quality')
                item['metrics'],item['output_evidence']=adapters[cell['adapter']](e,raw,cell,identity,domains,mode)
                item['valid']=True;valid.append(item)
            else:item['error']=raw.get('error','Explicit unsuccessful run; no timing assigned')
            records.append(item)
        terminal=None
        if cell.get('terminal'):
            terminal=e.read(cell['terminal']['evidence'])
            need(terminal.get('status') in FAILURES and cell['terminal']['reason'],'Invalid explicit terminal status')
            need(any(x['binding']==cell['terminal']['evidence'] and not x['valid'] for x in repeats.values()),
                 'Terminal must bind an observed unsuccessful run in this exact cell')
        complete=set(repeats)==set(range(10)) and len(valid)==10 and mode=='formal'
        metrics={}
        if complete:
            keys=set(valid[0]['metrics']);need(all(set(x['metrics'])==keys for x in valid),'Metric availability changed across repeats')
            for metric in keys:metrics[metric]=core.mean_ci([x['metrics'][metric] for x in sorted(valid,key=lambda x:x['repeat'])])
        summaries.append(dict(cell=cell['id'],experiment=cell['experiment'],definition={k:v for k,v in cell.items() if k not in ('runs','terminal')},
            valid_repetitions=len(valid),observed_repetitions=len(repeats),expected_repetitions=10,
            complete_formal=complete,terminal_status=terminal.get('status') if terminal else None,
            resolved=complete or terminal is not None,metrics=metrics))
    for history in selection.get('attempt_history',[]):
        # Keep explicitly declared failed/retried or prelaunch attempts without
        # treating a missing epoch as a source-qualified terminal observation.
        raw=e.read(history['evidence'])
        need(raw.get('status') in FAILURES and history.get('reason'),'History is only for explicit unsuccessful attempts')
        need(history['cell'] in selection['required_cells'],'History has no predeclared cell')
        need(history['evidence']['path'] not in seen_paths,'History duplicates a selected attempt')
        seen_paths.add(history['evidence']['path'])
        records.append(dict(cell=history['cell'],repeat=raw.get('repeat'),status=raw['status'],
            binding=history['evidence'],selected=False,valid=False,metrics={},
            error=raw.get('error'),exclusion_reason=history['reason'],
            source_status='not_admitted_to_selected_epoch_or_cell_coverage'))
    return dict(schema='eggpu.paper.normalized.v1',export_protocol='paper-results-protocol-completion-v1',
        synthetic_fixture=selection.get('synthetic_fixture') is True,mode=mode,source_tree_sha256=freeze['source_tree_sha256'],
        selection_binding={'path':str(selection_path.resolve()),'sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest()},
        selection=selection,records=records,cells=summaries,
        paper_ready=mode=='formal' and all(x['resolved'] for x in summaries) and any(x['complete_formal'] for x in summaries),
        evidence_verified=e.verified,formal_cells=sum(x['complete_formal'] for x in summaries),expected_cells=len(summaries),
        limitations=['E4-S requires full pinned recoverable vectors, exact decoded byte identity and formal archive/maintenance evidence; hash-only attestation is rejected.',
                    'Weight-maintenance H2D excludes other CUDA copies. New shared-array bytes are allocation evidence, not total reload traffic.'])
