"""Inference keeps scaling, joint benefit, and factorial interaction separate."""
from pathlib import Path
import math
import statistics
import normalize as n


def difference(control,treatment):
    result=n.core.welch_difference(control,treatment)
    result['interpretation']='predeclared independent-group mean difference; unadjusted95% interval, not a familywise claim'
    return result


def values(result,cell_id,metric):
    rows=sorted((x for x in result['records'] if x['cell']==cell_id and x['valid']),key=lambda x:x['repeat'])
    n.need(len(rows)==10 and [x['repeat'] for x in rows]==list(range(10)),'Ten exact independent repetitions required')
    return [row['metrics'][metric] for row in rows]


def ratio_ci(numerator,denominator,same=False):
    from scipy.stats import t
    n.need(len(numerator)==len(denominator)==10,'Ratio requires10 independent repeats per resource count')
    if same:return dict(mean=1.,ci95=[1.,1.],method='same exact P1 sample')
    a,b=statistics.mean(numerator),statistics.mean(denominator)
    va,vb=statistics.variance(numerator)/(10*a*a),statistics.variance(denominator)/(10*b*b)
    df=(va+vb)**2/(va*va/9+vb*vb/9) if va+vb else None
    radius=float(t.ppf(.975,df))*math.sqrt(va+vb) if df else 0.
    mean=a/b
    return dict(mean=mean,ci95=[mean*math.exp(-radius),mean*math.exp(radius)],
                method='independent-group log delta interval; not paired',degrees_of_freedom=df)


def scaling(result):
    groups={}
    for row in result['cells']:
        d=row['definition']
        if d['experiment']=='E2' and d['role']=='eggpu':
            groups.setdefault((d['dataset'],d['algorithm']),{})[d['gpu_count']]=row
    answer=[]
    for (dataset,algorithm),cells in sorted(groups.items()):
        for metric in ('first','ready'):
            complete=set(cells)=={1,2,4,8} and all(x['complete_formal'] for x in cells.values())
            item=dict(dataset=dataset,algorithm=algorithm,metric=metric,complete=complete)
            if complete:
                base=values(result,cells[1]['cell'],metric)
                means=[cells[p]['metrics'][metric]['mean'] for p in (1,2,4,8)]
                item['mean_monotone_nonincreasing']=all(y<=x for x,y in zip(means,means[1:]))
                item['points']={str(p):dict(time=cells[p]['metrics'][metric],speedup=ratio_ci(base,values(result,cells[p]['cell'],metric),same=p==1),
                    efficiency=statistics.mean(base)/(p*cells[p]['metrics'][metric]['mean'])) for p in (1,2,4,8)}
                item['adjacent_comparisons']=[dict(from_gpus=a,to_gpus=b,
                    time_saved=difference(values(result,cells[a]['cell'],metric),values(result,cells[b]['cell'],metric)))
                    for a,b in ((1,2),(2,4),(4,8))]
            answer.append(item)
    return answer


def paired_evidence(result):
    """A repeat number alone never establishes a matched execution block."""
    pairing=result['selection'].get('pairing')
    if not pairing:return {},'NO_EXPLICIT_PAIRED_SCHEDULE'
    e=n.core.Evidence(Path(result['selection_binding']['path']).parent,result['selection'].get('path_remap',[]))
    schedule=e.read(pairing['schedule']);execution=e.read(pairing['execution'])
    n.need(execution['plan']['sha256']==pairing['schedule']['sha256'],'Execution used another paired schedule')
    jobs={x['id']:x for x in schedule['jobs']};mapped={}
    selected={x['binding']['path']:x for x in result['records']}
    for outcome in execution['completed']:
        if outcome.get('outcome')!='PASS':continue
        receipt=outcome.get('receipt')
        if not receipt or receipt['path'] not in selected:continue
        e.verify(receipt)
        job=jobs[outcome['id']];command=job['command']
        n.need('--spec' in command,'Paired state execution lacks exact job spec')
        path=command[command.index('--spec')+1]
        candidates=[x for x in job['bindings'] if x['path']==path]
        n.need(len(candidates)==1,'Paired job spec must be explicitly hashed')
        spec=e.read(candidates[0]);block=spec['paired_block']
        n.need(spec['receipt']==receipt['path'] and spec['paired_block_sha256']==n.digest(block),'Paired block identity differs')
        n.need(selected[receipt['path']]['repeat']==block['repeat'],'Paired repeat differs')
        mapped[receipt['path']]=(spec['paired_block_sha256'],spec['arm'])
    return mapped,'EXPLICIT_SCHEDULE_AND_EXECUTION_VERIFIED'


def joint_effects(result):
    groups={}
    for row in result['cells']:
        d=row['definition']
        if d['experiment']=='E5':
            key=(tuple(d['datasets']),d['gpu_count'],d['rho'],d['beta'],d['order'],d['inputs']['sha256'])
            groups.setdefault(key,{})[d['arm']]=row
    mapped,pair_status=paired_evidence(result)
    output=[]
    for key,cells in groups.items():
        complete=set(cells)=={'D0C0','D1C0','D0C1','D1C1'} and all(x['complete_formal'] for x in cells.values())
        item=dict(datasets=list(key[0]),gpu_count=key[1],rho=key[2],beta=key[3],order=key[4],complete=complete,
                  pairing_status=pair_status)
        if complete:
            data={arm:values(result,row['cell'],'workflow') for arm,row in cells.items()}
            item['joint_benefit_vs_D0C0']=difference(data['D0C0'],data['D1C1'])
            item['joint_benefit_vs_each_single']={arm:difference(data[arm],data['D1C1']) for arm in ('D1C0','D0C1')}
            item['joint_mean_faster_than_both_single']=statistics.mean(data['D1C1'])<min(statistics.mean(data[x]) for x in ('D1C0','D0C1'))
            blocks={}
            ids={row['cell'] for row in cells.values()}
            for run in result['records']:
                if run['cell'] not in ids or not run['valid'] or run['binding']['path'] not in mapped:continue
                block,arm=mapped[run['binding']['path']]
                n.need(arm not in blocks.setdefault(block,{}),'Duplicate arm within paired block')
                blocks[block][arm]=run['metrics']['workflow']
            if len(blocks)==10 and all(set(row)==set(cells) for row in blocks.values()):
                contrasts=[row['D1C1']-row['D1C0']-row['D0C1']+row['D0C0'] for row in blocks.values()]
                item['interaction']=dict(n.core.mean_ci(contrasts),formula='T11-T10-T01+T00',
                    method='ten predeclared matched execution blocks',negative_means_superadditive_time_saving=True)
                item['interaction_established']=item['interaction']['ci95'][1]<0
            else:
                item['interaction']=None
                item['interaction_established']=False
                item['interaction_unavailable_reason']='Need all four arms in each of10 explicitly verified execution blocks; do not pseudo-pair by repeat number'
        output.append(item)
    return output


def predeclared_contrasts(result):
    output=[];groups={}
    for row in result['cells']:
        d=row['definition'];ex=d['experiment']
        if ex in ('E3','E4','E3_PR_CONTROL'):
            key=(ex,tuple(d.get('datasets',[d.get('dataset')])),d['gpu_count'],d.get('rho'),d.get('locality'),
                 d.get('old_readers'),d.get('beta'),d.get('order'))
            groups.setdefault(key,{})[d['arm']]=row
        elif ex=='E6' and d['system']=='eggpu' and d['gpu_count']==1:
            groups.setdefault(('E6',d['dataset']),{})[d['ablate']]=row
    for key,rows in sorted(groups.items(),key=str):
        pairs=[(x,'none') for x in ('structural','closeness','bfs','wcc')] if key[0]=='E6' else [('D1C0','D1C1')] if key[0]=='E4' else [('D0C1','D1C1')]
        for before,after in pairs:
            if before not in rows or after not in rows:continue
            a,b=rows[before],rows[after]
            if not(a['complete_formal'] and b['complete_formal']):continue
            metrics=('first','ready') if key[0]=='E6' else ('workflow','maintenance_h2d_bytes') if key[0]!='E4' else ('workflow','shared_array_new_bytes')
            for metric in metrics:
                av,bv=values(result,a['cell'],metric),values(result,b['cell'],metric)
                output.append(dict(experiment=key[0],control=a['cell'],treatment=b['cell'],metric=metric,
                    control_mean=statistics.mean(av),treatment_mean=statistics.mean(bv),saved=difference(av,bv),
                    ratio=ratio_ci(av,bv) if min(statistics.mean(av),statistics.mean(bv))>0 else None))
    return output


def analyze(result):
    return dict(scaling=scaling(result),joint=joint_effects(result),contrasts=predeclared_contrasts(result),
                confidence='Means: Student-t95% over10 independent complete repetitions. Pairing only with verified schedule/execution.',
                exclusions='Failure, timeout, OOM, unsupported index, and missing repeats do not receive zero time or an inferred win.')
