"""Prepare 36 matched SSSP-only target observations and one initial tiny round."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,sys
from observe_build import bind

HERE=Path(__file__).resolve().parent
SOURCE='71ea757d372f8c1b969ea074caaa27d3a77d4cea6d48cd9c6ae9c4b02936d7ce'
RHO=(0.,1e-6,1e-4,1e-2,.1,1.)
SYSTEMS=('D0C1','D1C1','Grapin')


def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def prepare(campaign,state_main,grapin_build,formatter_build,output):
    campaign,state_main,grapin_build,formatter_build,output=map(lambda p:Path(p).resolve(),(campaign,state_main,grapin_build,formatter_build,output))
    selection=read(state_main/'selection.json')
    if selection['source_tree_sha256']!=SOURCE:raise ValueError('Only source71 is admitted')
    if read(grapin_build)['status']!='PASS_OBSERVATION_ONLY_CPU_BUILD' or read(formatter_build)['status']!='PASS_CPU_FORMATTER_BUILD':raise ValueError('Both isolated CPU builds required')
    dep=read(campaign/'state_unified_full_v1/main/deployment.json');resource=dep['resources']['1']
    if str(Path(sys.executable).absolute())!=selection['python_invocation']:raise ValueError('Original core venv Python required')
    coverage=read(state_main/'coverage.json')['configurations'];rows={}
    for row in coverage:
        if row['experiment']=='E3' and row['partitions']==1 and len(row['datasets'])==1 and row['datasets'][0] in ('IN','OR') and row['rho'] in RHO and row['locality']=='uniform':
            key=(row['datasets'][0],row['rho'])
            if key in rows and rows[key]['inputs']!=row['inputs']:raise ValueError('Factor arms do not share exact input')
            rows[key]=row
    if set(rows)!={(d,r) for d in ('IN','OR') for r in RHO}:raise ValueError('Full planned density coverage missing')
    output.mkdir(parents=True,exist_ok=False);jobs=[];cells=[];unavailable=[]
    env=dict(dep['environment'],**resource.get('environment_additions',{}))
    env['PYTHONPATH']=str(Path(selection['native_dso']['path']).parent)+':'+selection['source_root']
    common=dict(source_tree_sha256=SOURCE,selection=bind(state_main/'selection.json'),gpu_uuids=resource['gpu-uuids'],cpu_set=resource['cpu-set'],
        lock=resource['lock'],environment=env,grapin_observation_build=bind(grapin_build),
        output_store=str(output/'objects'),reference_cache=str(campaign/'state_completion_results_v1/references'),
        artifact_root=str(campaign.parents[1]))
    def add(name,rho,inputs,system,tiny=False):
        record=read(inputs['path'])
        if bind(inputs['path'])!=inputs:raise ValueError('Frozen input changed')
        row=record['datasets'][name];phase='tiny' if tiny else 'target'
        identifier=f'{phase}_{name}_rho{rho:g}_{system}_r00';destination=output/'results'/identifier
        cell=dict(id=identifier,dataset=name,rho=rho,system=system,inputs=inputs,source=int(row['sources'][0]),
            changed_positions=row['changed_positions'],planned_update_records_per_kind=20*row['changed_positions'])
        if not tiny:cells.append(cell)
        if system=='Grapin' and 20*row['changed_positions']>2**32-1:
            unavailable.append(dict(cell,status='AUTHOR_UPDATE_COUNT_UNSUPPORTED',gpu_measurement=False,
                reason='Frozen Loader total_add/total_del and read cursors are uint32; the complete 20-commit trace exceeds its exact representable per-kind count.',
                source=bind(HERE/'grapin_update_Loader.h')))
            return
        spec_path=output/'specs'/(identifier+'.json');prepared=output/'prepared'/f'{name}_rho{rho:g}'
        input_spec=output/'input_specs'/f'{name}_rho{rho:g}.json'
        if not input_spec.exists():
            save(input_spec,dict(inputs=inputs,dataset=name,author_text_helpers=str(campaign/'first_pass_20261106/author_text_cache'),
                hash_cache_helper=selection['hash_cache_helper']['path'],hash_cache=selection['hash_cache'],
                update_formatter_build=str(formatter_build),base_formatter_build=str(campaign/'first_pass_20261106/author_text_cache_build/receipt.json'),
                base_text_cache=str(output/'base_text_objects'),author_worker=str(campaign/'runner_candidates/sg_author_clock_v3/baselines/sg_author_worker.py')))
        spec=dict(common,system=system,dataset=name,rho=rho,phase=phase,repeat=0,inputs=inputs,n=row['n'],m=row['m'],
            output=str(destination),prepared_inputs=str(prepared),input_preparation_spec=str(input_spec),
            tiny_proof=None if tiny else str(output/'results'/f'tiny_TINY_A_rho0.0001_{system}_r00'/'receipt.json'))
        save(spec_path,spec)
        jobs.append(dict(id=identifier,command=[selection['python_invocation'],'-B',str(HERE/'worker.py'),'--spec',str(spec_path),'--execute-gpu'],
            receipt=str(destination/'receipt.json'),expected_statuses=['PASS_E3_SSSP_SUBSET'],timeout=7200,
            depends_on=[] if tiny else [f'tiny_TINY_A_rho0.0001_{system}_r00'],environment=env,
            bindings=[bind(spec_path),inputs,common['selection'],common['grapin_observation_build'],bind(formatter_build)]+[bind(HERE/name) for name in ('worker.py','observe_build.py','prepare_inputs.py','update_text.cpp','prepare_plan.py')],
            resource_admission=dict(output_reserve_bytes=21*row['n']*8*3+(2<<30),fail_on_resource=True,
                scope='Worker checks full raw/archive/reference reserve before preparation and before native execution; exact input text is added before writing. No queue WAIT_STORAGE while holding controller lock.'),
            observations=0 if tiny else 1,phase=phase,system=system,dataset=name,rho=rho))
    for system in SYSTEMS:add('TINY_A',1e-4,resource['tiny_inputs'],system,True)
    for dataset in ('IN','OR'):
        for rho in RHO:
            for system in SYSTEMS:add(dataset,rho,rows[(dataset,rho)]['inputs'],system)
    plan=dict(schema='e3.sssp.common-update.first-observation.plan.v1',jobs=jobs,environment={},target_cells=36,initial_tiny_checks=3,unavailable=unavailable,
        source_tree_sha256=SOURCE,observations=1,storage_poll_seconds=5,full_four_function_E3_comparison=False,
        scope='P1 fixed first primary source; 20 identical weight commits and 21 complete SSSP outputs; D0/D1 versus original Grapin incremental update API.',
        timings=['prepared_service_seconds','post_base_update_workflow_seconds'],cold_first_comparable=False,
        baseline_scope='Grapin maintains SSSP incrementally. EGGPU retains its existing full-query algorithms. No result is inferred from another function or workflow.',gpu_executed=False)
    save(output/'plan.json',plan);save(output/'coverage.json',dict(cells=cells,target_cells=36,unavailable=unavailable,source_tree_sha256=SOURCE,gpu_executed=False))
    save(output/'stage_manifest.json',dict(queue=str(output/'queue'),plan=str(output/'plan.json'),target_cells=36,executable_target_cells=len(jobs)-3,unavailable=unavailable))
    return dict(status='PREPARED_CPU_ONLY',jobs=len(jobs),targets=36,plan=bind(output/'plan.json'),gpu_executed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('campaign','state-main','grapin-build','formatter-build','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.campaign,a.state_main,a.grapin_build,a.formatter_build,a.output)))
