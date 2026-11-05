"""Continuous sixteen-request evidence; never sum independent function runs."""
import math
import numpy as np
import normalize as n
PROTOCOL='e1-continuous-16-public-csr-v1'
FUNCTIONS=('bfs','sssp','wcc','pr')
SOURCE_INDICES=(0,1,2,0)

def payload(e,raw,cell,domain,tiny=False):
    n.need(raw.get('protocol')==PROTOCOL and raw.get('system')==cell['system'] and raw.get('request_count')==16 and raw.get('quality_pass') is True,'Wrong continuous workflow identity')
    if tiny:
        n.need(raw.get('dataset') in ('TINY_D','TINY_U') and raw['directed']==domain['directed'],'Wrong tiny workflow directedness')
    else:n.need(raw['input_sha256']==domain['sha256'] and raw['dataset']==cell['dataset'],'Wrong continuous workflow input')
    trace=raw['trace'];n.need(len(trace)==16,'Continuous workflow must contain sixteen requests')
    sources=[trace[i*4]['source'] for i in range(3)] if tiny else domain['sources']
    n.need(len(set(sources))==3,'Three independent frozen source IDs required')
    expected=[dict(index=r*4+i,round=r,algorithm=algorithm,source=sources[SOURCE_INDICES[r]] if algorithm in ('bfs','sssp') else None,source_index=SOURCE_INDICES[r] if algorithm in ('bfs','sssp') else None) for r in range(4) for i,algorithm in enumerate(FUNCTIONS)]
    n.need(trace==expected,'Frozen continuous request/source order changed')
    child=e.read(raw['workflow'])
    n.need(child.get('status')=='PASS' and child.get('quality_pass') is True and child.get('execution_complete') is True and child.get('outputs_survive_close') is True,'Incomplete continuous workflow output')
    n.need(child['protocol']==PROTOCOL and child['system']==cell['system'] and child['epoch_sha256']==raw['epoch_sha256'] and child['input_sha256']==raw['input_sha256'],'Continuous child source/input differs')
    n.need(child.get('continuous_interval') is True and child.get('sum_of_query_times_used_as_total') is False and child.get('warmup_requests')==0,'Continuous time must not be a sum of standalone calls')
    n.need(child.get('requests')==16 and len(child['calls'])==16,'Missing continuous request output')
    start,complete=child['workflow_start_ns'],child['workflow_complete_ns']
    n.need(type(start) is int and type(complete) is int and complete>start,'Invalid continuous monotonic interval')
    total=(complete-start)/1e9
    n.need(math.isclose(n.number(child['T_workflow_seconds'],positive=True),total,rel_tol=1e-9,abs_tol=1e-9) and math.isclose(n.number(raw['T_workflow_seconds'],positive=True),total,rel_tol=1e-9,abs_tol=1e-9),'Continuous parent/child interval differs')
    metrics={'workflow':total};previous=start;size=domain['n'] if not tiny else None
    for i,(request,row) in enumerate(zip(expected,child['calls'])):
        n.need(all(row.get(k)==v for k,v in request.items()),'Measured request order differs')
        begin,end=row['begin_ns'],row['end_ns']
        n.need(type(begin) is int and type(end) is int and previous<=begin<end<=complete,'Overlapping or out-of-interval request clock')
        n.need(math.isclose(n.number(row['seconds'],positive=True),(end-begin)/1e9,rel_tol=1e-9,abs_tol=1e-9),'Request seconds differ from continuous clock')
        n.core.verify_quality(row['quality'],request['algorithm'])
        n.need(row['quality'].get('owning') is True and row['quality'].get('immutable') is True,'Workflow output ownership missing')
        path=e.verify(row['output']);values=np.load(path,mmap_mode='r',allow_pickle=False)
        if size is None:size=len(values)
        n.need(values.shape==(size,) and values.dtype.str=='<f8','Workflow complete output domain differs')
        n.need(all(0<=source<size for source in sources),'Frozen source out of domain')
        metrics[f'prefix_{i+1:02d}']=(end-start)/1e9;previous=end
    return metrics

def run(e,raw,cell,epoch,domains,mode):
    n.need(epoch.get('protocol')==PROTOCOL and epoch.get('workflow')==list(FUNCTIONS)*4 and epoch.get('source_indices')==list(SOURCE_INDICES) and epoch.get('formal_repetitions')==10 and epoch.get('warmups')==0,'Wrong continuous epoch protocol')
    domain=domains[cell['dataset']];metrics=payload(e,raw,cell,domain)
    n.need(raw['status']==('PASS_FORMAL' if mode=='formal' else 'PASS_QUALIFICATION'),'Qualification is not a formal continuous workflow')
    if mode=='formal':
        for phase in ('tiny','target'):
            proof=e.read(raw['qualification'][phase])
            n.need(proof.get('status')=='PASS_QUALIFICATION' and proof.get('phase')==phase and proof.get('epoch_sha256')==cell['epoch_sha256'] and e.read(proof['epoch'])==epoch,'Continuous proof epoch/phase differs')
            payload(e,proof,cell,domain,tiny=phase=='tiny')
    return metrics,'sixteen_complete_archived_vectors_and_continuous_timestamps_verified'

def render(result,out,captions):
    import render as r
    rows=[x for x in result['cells'] if x['experiment']=='E1_SEQUENCE']
    if not rows:return
    plt=r.pyplot();fig,axes=plt.subplots(1,3,figsize=(6.89,2.3),constrained_layout=True,squeeze=False);handles=[];labels=[]
    for ax,dataset in zip(axes.flat,('US','IN','TW')):
        for system,label,color in [('eggpu','EGGPU',r.PALETTE['blue']),('cugraph','cuGraph',r.PALETTE['orange'])]:
            selected=[x for x in rows if x['definition']['dataset']==dataset and x['definition']['system']==system]
            n.need(len(selected)==1,'Exactly one continuous cell per system/graph required')
            row=selected[0]
            if not row['complete_formal']:
                ax.text(.02,.98-(.12 if system=='cugraph' else 0),label+': '+str(row['terminal_status'] or 'INCOMPLETE'),transform=ax.transAxes,va='top',fontsize=6);continue
            stats=[r.stat(row,f'prefix_{i:02d}') for i in range(1,17)]
            h=ax.errorbar(range(1,17),[s[0] for s in stats],yerr=[[s[1][0] for s in stats],[s[1][1] for s in stats]],color=color,marker='o' if system=='eggpu' else 's',capsize=1.5,label=label)
            if label not in labels:handles.append(h);labels.append(label)
        ax.set_title(dataset);ax.set_xlabel('Requests');ax.set_ylabel('Cumulative time (s)');ax.set_xticks([1,4,8,12,16]);ax.set_ylim(bottom=0);ax.grid(axis='y')
    if handles:fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    r.figure_save(fig,out,'Fig3_continuous_sequence','One fresh process executes [BFS, SSSP, WCC, PR] four times on US/IN/TW. Points use actual monotonic request completion timestamps from the single common-CSR-ready interval; source indices are0,1,2,0. The complete continuous workflow time is retained separately. No standalone function means are summed and no warmup requests are discarded.',captions,result,rows)
