"""Incremental recorded-evidence export for the separate 32+6 state additions."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,os,shutil,time
from prepare import bind,read,SOURCE


def fingerprint(path):
    value=Path(path).stat()
    return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns]
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def need(value,message):
    if not value:raise ValueError(message)
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def option(command,name):return command[command.index(name)+1]


def validate_record(data,job,source_binding,selection_binding,spec=None,deployment=None):
    need(data.get('phase')=='target' and data.get('repeat')==0 and data.get('execution_complete') is True and data.get('quality_pass') is True,'Missing complete target repeat-zero numerical pass')
    config=data['configuration'];need(data['configuration_sha256']==digest(config),'Configuration digest differs')
    need(data.get('continuous_interval') is True and data.get('sum_of_phases_used_as_total') is False,'Timing must be an actual continuous interval')
    checks=data['checks'];events=data['events']
    for item in checks:
        need(item.get('quality',{}).get('passed') is True and item.get('survives_session_close') is True,'Output numerical/lifetime check missing')
        blob=item['output']['blob'];path=Path(blob['path'])
        need(path.is_file() and path.stat().st_size==blob['bytes'],'Retained complete output is unavailable')
    result=dict(id=job['id'],status='PASS_RECORDED_SOURCE71_N1',phase='target',repeat=0,observations=1,
        time_seconds=data['workflow_complete_host_seconds'],primary_time_boundary='recorded_ready_continuous_workflow',
        first_complete_workflow_seconds=data['first_complete_workflow_seconds'],workflow_complete_host_seconds=data['workflow_complete_host_seconds'],
        quality_pass=True,complete_output_count=len(checks),validation='Source/configuration/trace/recorded full numerical and lifetime checks plus retained output availability; no repeated numerical computation or source parsing.')
    need(all(finite(result[key]) and result[key]>0 for key in ('first_complete_workflow_seconds','workflow_complete_host_seconds')),'Invalid measured interval')
    if job['first_pass_stage']=='unweighted_pr_negative_control':
        need(data['schema']=='e3-unweighted-pr-negative-v3-native-device-keys' and data['status']=='PASS_UNWEIGHTED_PR_QUALIFICATION','Wrong negative-control protocol')
        need(config['source_tree_sha256']==SOURCE and config['deployment']==deployment,'Wrong current PR deployment/source')
        need(data['dataset']==job['dataset'] and data['arm']==job['arm'] and config['arm']==job['arm'] and data['inputs']==job['input_binding'],'PR planned cell/input differs')
        actual=dict(config['source_binding']);actual.pop('negative_control');actual.pop('admission_support')
        need(actual==source_binding,'PR source/DSO/current state runner differs from retained focused source check')
        need(len(events)==41 and len(checks)==21 and config['commits']==20 and config['queries']==21,'Incomplete PR trace')
        expected=[dict(kind='query',dataset=data['dataset'],version=0,function='pr')]
        for version in range(1,21):expected.extend([dict(kind='commit',dataset=data['dataset'],version=version,base=f"{data['dataset']}@{version-1}",snapshot=f"{data['dataset']}@{version}"),dict(kind='query',dataset=data['dataset'],version=version,function='pr')])
        need(all(all(event.get(k)==v for k,v in wanted.items()) for wanted,event in zip(expected,events)),'PR commit/query identities differ')
        queries=[event for event in events if event['kind']=='query']
        need(all(item['event']=={k:v for k,v in event.items() if k not in ('interval_seconds','details')} for item,event in zip(checks,queries)),'PR output/event association differs')
        counters=('weight_full_h2d_bytes','patch_h2d_bytes','dense_lease_h2d_bytes','cow_d2d_bytes')
        for event in queries[1:]:
            totals=event['details']['dependency_audit']['weighted_view_transfer_counters']
            need(all(key in totals and finite(totals[key]) and totals[key]==0 for key in counters),'Nonzero/missing unweighted-PR weighted-view maintenance')
        bounds=dict(residual_l1=1e-6,mass_error=1e-6,reference_l1=1e-5,invariance_l1=2e-8)
        maxima={key:max(item['quality'][key] for item in checks) for key in bounds}
        need(all(finite(maxima[key]) and 0<=maxima[key]<=bounds[key] for key in bounds),'PR numerical/invariance proof differs')
        result.update(scope='unweighted_PR_negative_control',dataset=data['dataset'],arm=data['arm'],partitions=1,
            commits=20,queries=21,quality_maxima=maxima,post_commit_weighted_view_bytes={key:0 for key in counters},
            total_cuda_copy_bytes='NOT_FULLY_INSTRUMENTED_DO_NOT_INFER_ZERO',
            workflow_scope='21 full unweighted PR queries and 20 weight commits after Session creation; not an individual ready-query time')
    else:
        need(data['schema']=='state.complete.workflow.v4' and data['status']=='PASS_STATE_GPU_QUALIFICATION','Wrong complete state protocol')
        need(data['source_binding']==source_binding and config['source_binding']==source_binding,'Hotspot source/DSO/runner differs')
        need(spec['selection']==selection_binding and spec['paired_block']['source_tree_sha256']==SOURCE,'Wrong hotspot selection')
        need(data['first_pass']['spec']==bind(option(job['command'],'--spec')) and data['first_pass']['paired_block_sha256']==spec['paired_block_sha256'],'Hotspot spec/block differs')
        need(data['inputs']==spec['inputs'] and len(events)==100 and len(checks)==80,'Incomplete hotspot update trace')
        need(config['arm']==spec['arm'] and config['old_readers']==spec['paired_block']['old_readers'] and config['experiment']=='E3' and len(config['gpu_uuids'])==8,'Hotspot intervention/resources differ')
        need(sum(e['kind']=='commit' for e in events)==20 and sum(e['kind']=='query' for e in events)==80,'Hotspot counts differ')
        for version in range(1,21):
            group=events[(version-1)*5:version*5]
            need(group[0]['kind']=='commit' and group[0]['version']==version,'Commit version differs')
            need([e['function'] for e in group[1:]]==['bfs','sssp','wcc','weighted_pr'] and all(e['version']==version for e in group[1:]),'Function/version sequence differs')
        need(all(item['event']=={k:v for k,v in event.items() if k not in ('index','public_seconds','details')} for item,event in zip(checks,[e for e in events if e['kind']=='query'])),'Hotspot output/event association differs')
        if config['old_readers']=='per_update':need(len(data['leases'])==20 and all(x['released'] for x in data['leases']),'Missing old-reader completion')
        block=spec['paired_block']
        result.update(scope='partition_hotspot',dataset=spec['datasets'][0],arm=spec['arm'],partitions=8,
            owner_count=block['hotspot_k'],old_readers=block['old_readers'],rho=block['rho'],
            maintenance_counters=data['maintenance_counters'],commits=20,queries=80,
            workflow_scope='20 four-function update batches after the common completed base queries; first interval also includes base initialization')
    return result


def export(plan_path,output):
    plan_path=Path(plan_path).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    plan=read(plan_path);need(plan['target_cells']==38 and plan['source_tree_sha256']==SOURCE,'Wrong additive plan')
    state_main=Path(plan['original_state_plan']['path']).parent
    paths=[plan_path,state_main/'selection.json',state_main/'results/focused_source_p8/receipt.json',plan_path.parent/'pr_deployment.json']
    identity=[fingerprint(path) for path in paths]
    cache_path=output/'cache.json';cache=read(cache_path) if cache_path.exists() else {}
    if cache.get('identity')!=identity:
        selection=read(paths[1]);focus=read(paths[2]);need(selection['source_tree_sha256']==SOURCE and focus['status']=='PASS_STATE_GPU_QUALIFICATION','Focused source evidence missing')
        cache=dict(identity=identity,selection_binding=bind(paths[1]),source_binding=focus['source_binding'],deployment=bind(paths[3]),results={})
    rows=[];initial=[]
    for job in plan['jobs']:
        if job.get('phase')=='tiny':
            path=Path(job['receipt']);initial.append(dict(id=job['id'],status=read(path).get('status') if path.exists() else 'PENDING'));continue
        path=Path(job['receipt'])
        if not path.exists():rows.append(dict(id=job['id'],status='PENDING',observations=0));continue
        stamp=fingerprint(path);entry=cache['results'].get(job['id'])
        if entry is None or entry['fingerprint']!=stamp:
            data=read(path)
            try:
                spec=None if job['first_pass_stage']=='unweighted_pr_negative_control' else read(option(job['command'],'--spec'))
                value=validate_record(data,job,cache['source_binding'],cache['selection_binding'],spec,cache['deployment'])
                value['receipt']=bind(path)
            except (KeyError,ValueError,TypeError,OSError) as error:
                value=dict(id=job['id'],status=data.get('status','INCOMPLETE'),export_status='WITHHELD',reason=str(error),observations=0,receipt=bind(path))
            entry=dict(fingerprint=stamp,row=value,outputs=[x['output']['blob'] for x in data.get('checks',[]) if 'output' in x and 'blob' in x['output']]);cache['results'][job['id']]=entry
        value=entry['row']
        if value['status']=='PASS_RECORDED_SOURCE71_N1':
            try:
                need(all(Path(x['path']).is_file() and Path(x['path']).stat().st_size==x['bytes'] for x in entry['outputs']),'Retained complete output is unavailable')
            except (OSError,ValueError) as error:
                value=dict(value,status='RETAINED_OUTPUT_UNAVAILABLE',export_status='WITHHELD',reason=str(error),observations=0)
        rows.append(value)
    need(len(rows)==38 and len(initial)==2,'The additive plan must retain exactly 38 target cells and two initial PR checks')
    summary=dict(schema='eggpu.state.additions.progress.v1',source_tree_sha256=SOURCE,main_state_denominator=504,
        additive_denominator=38,hotspot_denominator=32,unweighted_pr_denominator=6,
        eligible_n1=sum(row['status']=='PASS_RECORDED_SOURCE71_N1' for row in rows),status_counts=dict(Counter(row['status'] for row in rows)),
        initial_checks=initial,rows=rows,updated_unix=time.time(),eta_seconds=None,
        timing_note='Primary time_seconds is the directly recorded workflow_complete_host_seconds ready continuous interval. first_complete_workflow_seconds is supplemental. No phase subtraction, whole-job wall time or 504-main denominator substitution.')
    current=output/'current.json'
    if current.exists():shutil.copyfile(current,output/'previous.json')
    for path,value in ((current,summary),(cache_path,cache)):
        temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch-seconds',type=int,default=0);a=p.parse_args()
    while True:
        result=export(a.plan,a.output);print(json.dumps({key:result[key] for key in ('eligible_n1','additive_denominator','status_counts')}),flush=True)
        if not a.watch_seconds:break
        time.sleep(max(15,a.watch_seconds))
