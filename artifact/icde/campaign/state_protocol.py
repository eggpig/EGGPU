"""Frozen E3/E4/E5 interventions and CPU-only trace/budget helpers."""
from pathlib import Path
import hashlib, json, math, random

ARMS={'D0C0':('persistent_full','closure_lru'),'D1C0':('lease_sparse','closure_lru'),
      'D0C1':('persistent_full','closure'),'D1C1':('lease_sparse','closure')}
FUNCTIONS=('bfs','sssp','wcc','weighted_pr')
POOLS={'E3':('US','IN','OR'),'E4':('US','OR','IN','UK'),'E5':('IN','UK','OR')}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def bind(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}


def checked(item):
    p=Path(item['path'])
    if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Frozen file changed: '+str(p))
    return p


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_suffix(p.suffix+'.partial');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(p)


def trace(experiment,datasets,order='interleaved',qualification=False):
    names=list(datasets);events=[];rounds=2 if qualification else (20 if experiment=='E3' else 8)
    if experiment=='E3' and len(names)!=1:raise ValueError('Each E3 trace has exactly one graph')
    visits=[(name,r) for r in range(rounds) for name in names]
    if order=='grouped':visits=[(name,r) for name in names for r in range(rounds)]
    elif order=='permuted':
        if experiment!='E4':raise ValueError('Version-changing traces cannot reorder visits')
        random.Random(20261104).shuffle(visits)
    elif order not in ('interleaved','once'):raise ValueError('Unknown order')
    if order=='once':
        if experiment!='E4':raise ValueError('No-revisit control is E4 only')
        visits=[(name,0) for name in names]
    versions={name:0 for name in names}
    for name,r in visits:
        if experiment!='E4':
            old=versions[name];versions[name]+=1
            events.append(dict(kind='commit',dataset=name,base=f'{name}@{old}',snapshot=f'{name}@{old+1}',version=old+1))
        for function in FUNCTIONS:
            events.append(dict(kind='query',dataset=name,snapshot=f'{name}@{versions[name]}',function=function,
                               source_index=r%3,version=versions[name]))
    return events


def budgets(pilot,beta):
    if not 0<=beta<=1:raise ValueError('beta outside [0,1]')
    rows=pilot['resident_view_bytes'];devices=pilot['devices'];result={};bounds={}
    for device in devices:
        sizes=[r[str(device)] for r in rows.values()]
        if not sizes or any(type(x)is not int or x<=0 for x in sizes):raise ValueError('Invalid physical view byte pilot')
        lower=max(sizes);upper=sum(sizes);result[str(device)]=math.floor(lower+beta*(upper-lower))
        bounds[str(device)]={'B_min':lower,'B_all':upper}
    return {'per_device':result,'beta':beta,'bounds':bounds,'rule':'floor(max_g S_gd + beta*(sum_g S_gd-max_g S_gd))','selection_uses_timing':False}


def validate_intervention(metrics,route,policy,budget):
    dynamic=metrics['native_dynamic_patch'];resource=metrics['resource_management']
    if dynamic['route']!=route or dynamic['retirement_policy']!='lineage':raise ValueError('Actual D treatment differs')
    if resource['state_switch_policy']!=policy:raise ValueError('Actual C treatment differs')
    if resource['controller_version']!='state_switch_v2_closure_lru':raise ValueError('Old controller is not the strong control')
    if budget is not None:
        for row in resource['after_retention']['per_device']:
            if row['managed_resident_bytes']>budget[str(row['device_id'])]:raise ValueError('Physical budget exceeded')
    for action in resource['actions']:
        if action.get('prediction_error_bytes',0):raise ValueError('Physical closure accounting mismatch')
    if route=='persistent_full':
        for row in dynamic['per_device_delta'].values():
            if any(row.get(key,0) for key in ('patch_packets','lease_reuses','cow_copies')):
                raise ValueError('D0 unexpectedly used D1 maintenance')
    return {'D':route,'C':policy,'physical_release_receipts_checked':len(resource['actions'])}
