"""One-shot root launch; immutable native-36 proof plus complete CPU I/O gates.

Default inspection is read-only. GUM is independently held unless a separately
bound release names its plan and audited precision/display policy. No retries.
CPU-reference archive bytes are not rehashed by this scheduling gate: per-entry
metadata and payload lengths are checked; the frozen validation reader checks
actual reference payloads. This gate establishes preparation quiescence.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time


def require(ok,message):
    if not ok:raise ValueError(message)


def bind(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())


def checked(item):
    # Historical worker bindings carry path+SHA; newer bindings also record size.
    # A missing optional size is not a content change. Always check the full SHA.
    require(isinstance(item, dict) and set(item) in ({'path','sha256'}, {'path','sha256','bytes'}), 'Invalid file binding fields')
    require(isinstance(item['path'], str) and Path(item['path']).is_absolute(), 'Binding path must be absolute')
    require(isinstance(item['sha256'], str) and len(item['sha256']) == 64 and all(c in '0123456789abcdef' for c in item['sha256']), 'Invalid binding SHA256')
    if 'bytes' in item:
        require(type(item['bytes']) is int and item['bytes'] >= 0, 'Invalid binding size')
    actual=bind(item['path'])
    require(all(actual[key] == value for key,value in item.items()), 'Bound file changed: '+item['path'])
    return Path(item['path'])


def load(path):return json.loads(Path(path).read_text())
def save(path,obj):
    p=Path(path);tmp=p.with_name(p.name+'.partial')
    with tmp.open('w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def native_gate(spec):
    state=load(checked(spec['state']));plan=load(checked(spec['plan']))
    require(state['plan']==spec['plan'] and state['jobs']==plan['jobs'],'Native gate plan differs')
    jobs=plan['jobs'];rows=state['completed'];require(len(jobs)==spec['count']==36,'Expected exact 36 native qualifications')
    require(state['status']=='PASS_REQUESTED_JOBS' and [x['id'] for x in rows]==[x['id'] for x in jobs],'Native gate incomplete or changed')
    for job,row in zip(jobs,rows):
        require(row['outcome']=='PASS' and row['returncode']==0 and row['status'] in job['expected_statuses'],'Native qualification failed')
        require(row['receipt']['path']==job['receipt'],'Native receipt path differs');r=load(checked(row['receipt']))
        require(r['status']==row['status'],'Native receipt status differs')
        s=load(checked(row['process_supervision']))
        require(s['status']=='PASS_PROCESS_EXIT' and s['supervisor_returncode']==0 and not s.get('surviving_owned_processes') and 'finished_monotonic' in s,'Native worker not cleanly terminated')
    return spec['state']


def references_gate(spec):
    checked(spec['cohort_controller']);path=Path(spec['state'])
    if not path.exists():return dict(status='WAITING_CPU_REFERENCE_COHORT',observed='ABSENT')
    try:state=load(path)
    except json.JSONDecodeError:return dict(status='WAITING_CPU_REFERENCE_COHORT',observed='IN_PROGRESS_WRITE')
    status=state.get('status')
    if status=='RUNNING':return dict(status='WAITING_CPU_REFERENCE_COHORT',observed=status)
    require(status=='PASS_ALL_FOUR_REFERENCE_COHORTS','CPU reference cohort ended without complete PASS: '+str(status))
    expected=list(spec['datasets']);rows=state['completed']
    require(state['gpu_executed'] is False and [x['dataset'] for x in rows]==expected,'Reference cohort identities differ')
    proofs=[]
    for row,name in zip(rows,expected):
        require(row['returncode']==0 and row['receipt']==spec['datasets'][name]['receipt'],'Reference cohort child failed or path changed')
        r=load(row['receipt']);entry=spec['datasets'][name]
        require(r['status']=='PASS_ALL_512_CPU_REFERENCES' and r['dataset']==name and r['completed']==512 and len(r['entries'])==512,'Incomplete reference dataset')
        require(r['gpu_executed'] is False and r['independent'] is True and r['lossless'] is True and r['complete_sha_bindings'] is True,'Reference preparation contract differs')
        require(r['sources']==entry['sources'] and r['reference_source']==spec['reference_source'] and r['reference_source_after']==spec['reference_source'] and r['tool']==spec['preparer'],'Reference code/source binding differs')
        sources=load(checked(entry['sources']))['sources'];require(len(sources)==len(set(sources))==512 and sources==r['full_source_sequence'],'Reference sequence differs')
        for i,e in enumerate(r['entries']):
            require(e['sequence']==i and e['source']==sources[i] and e['values']==r['n'],'Reference entry identity differs')
            checked(e['metadata']);f=e['reference']['file'];p=Path(f['path'])
            require(p.is_file() and p.stat().st_size==f['bytes'],'Missing or truncated reference archive')
        proofs.append(bind(row['receipt']))
    checked(spec['reference_source']);checked(spec['preparer'])
    return dict(status='PASS_ALL_REFERENCE_PREPARATION_COMPLETE',cohort=bind(path),receipts=proofs,payload_validation='Lengths here; full payload checks remain in frozen worker validator')


def inspect(config,group):
    require(config['schema']=='external.formal.watcher.v4','Wrong watcher config')
    for key in ('controller','supervisor','watcher'):checked(config[key])
    native=native_gate(config['native'])
    full=load(checked(config['full_grid']))
    require(full['status']=='PASS_CPU_SHARED_GRID_PREPARED_GPU_UNQUALIFIED' and len(full['cells'])==56 and full['gpu_executed'] is False,'Full grid preparation incomplete')
    selected=['cugraph_mg','c2p'] if group=='supported' else ['gum']
    if group=='gum':
        release=config.get('gum_release')
        if release is None:return dict(status='HOLD_GUM_PRECISION_AUDIT',gpu_queues_launched=False)
        r=load(checked(release));require(r['status']=='PASS_GUM_FORMAL_RELEASE' and r['plan_sha256']==config['groups']['gum']['plan']['sha256'] and r['display_label']=='GUM (standard)','GUM release differs')
        checked(r['precision_audit'])
    refs=references_gate(config['references'])
    if refs['status']!='PASS_ALL_REFERENCE_PREPARATION_COMPLETE':return dict(refs,gpu_queues_launched=False)
    plans=[]
    for name in selected:
        g=config['groups'][name];p=load(checked(g['plan']))
        require(len(p['jobs'])==g['count'],'Formal count changed')
        require(not Path(g['queue']).exists() and not Path(g['log']).exists(),'Formal queue already attempted')
        for j in p['jobs']:
            require(j['phase']=='formal' and not Path(j['receipt']).parent.exists(),'Formal output already attempted')
            for b in j.get('bindings',[]):checked(b)
            m=j.get('maintenance')
            if m:
                for b in m['bindings']:checked(b)
                require(not Path(m['receipt']).parent.exists(),'Formal maintenance already attempted')
        plans.append(g['plan'])
    return dict(status='PASS_ALL_LAUNCH_GATES',gpu_queues_launched=False,native=native,references=refs,plans=plans,selected_groups=selected)


def run(config_path,group,output,execute=False,poll=15):
    cfg_binding=bind(config_path);cfg=load(config_path)
    if not execute:return inspect(cfg,group)
    require(output is not None,'Explicit unused launch receipt is required');output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    state=dict(status='WAITING_CPU_PREPARATION',config=cfg_binding,generator=bind(__file__),group=group,launched=[],gpu_queues_launched=False,started_unix=time.time())
    with output.open('x') as f:json.dump(state,f);f.flush();os.fsync(f.fileno())
    try:
        while True:
            checked(cfg_binding);observation=inspect(cfg,group);state['gate_observation']=observation;save(output,state)
            if observation['status']=='HOLD_GUM_PRECISION_AUDIT':state['status']=observation['status'];return state
            if observation['status']=='PASS_ALL_LAUNCH_GATES':break
            time.sleep(poll)
        os.sched_setaffinity(0,set(cfg['cpu_affinity']))
        for name in observation['selected_groups']:
            g=cfg['groups'][name];plan=load(checked(g['plan']))
            command=[cfg['python_invocation'],'-B',cfg['controller']['path'],'--plan',g['plan']['path'],'--output',g['queue'],'--controller-lock',cfg['controller_lock']]
            with Path(g['log']).open('xb') as log:
                child=subprocess.Popen(command,env={**os.environ,**plan['environment']},stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            state['launched'].append(dict(group=name,pid=child.pid,plan=g['plan'],queue=g['queue']));state.update(status='LAUNCHING_SERIAL_FORMAL_QUEUES',gpu_queues_launched=True);save(output,state)
        state['status']='LAUNCHED_SERIAL_FORMAL_QUEUES';return state
    except BaseException as e:state.update(status='FAILED_CLOSED_NO_RETRY',error=type(e).__name__+': '+str(e));raise
    finally:state['finished_unix']=time.time();save(output,state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--group',choices=('supported','gum'),default='supported');p.add_argument('--output',type=Path);p.add_argument('--execute-gpu',action='store_true');a=p.parse_args();print(json.dumps(run(a.config,a.group,a.output,a.execute_gpu)))
