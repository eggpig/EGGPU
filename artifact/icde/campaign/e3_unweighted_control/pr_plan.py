#!/usr/bin/env python3
"""CPU-only explicit plans for the separate21-query unweighted PR control."""
from pathlib import Path
import argparse
import json
import os
import sys
from types import SimpleNamespace
import pr_control as c


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')


def build(a):
    c.need(not a.output.exists(), 'Use a new immutable control plan directory')
    dep = c.read(a.deployment)
    support = c.load_support(c.checked(dep['planner']))
    c.need(support.runtime_probe(dep['python_invocation'],dep['environment'])==dep['python_runtime'],
           'Actual invocation/virtual environment differs from deployment')
    freeze = c.read(c.checked(dep['source_freeze']))
    support.source_files(Path(dep['source_root']),freeze)
    registry = c.read(c.checked(dep['registry']))
    inputs = {}
    for path in a.inputs:
        record = c.read(path)
        support.validate_input_metadata(record,freeze,registry)
        for name,row in record['datasets'].items():
            c.need(name in ('US','IN','OR') and record['partitions']==a.partitions and
                   row['requested_rho']==1e-4 and row['locality']=='uniform', 'Explicit E3rho1e-4 uniform inputs required')
            c.need(name not in inputs,'Duplicate dataset input')
            inputs[name]=c.bind(path)
    c.need(set(inputs)=={'US','IN','OR'},'Control retains all three fixed graphs')
    resource=dep['resources'][str(a.partitions)]
    tiny=c.read(c.checked(resource['tiny_inputs']))
    support.validate_input_metadata(tiny,freeze,registry,tiny=True)
    worker=Path(__file__).with_name('pr_control.py').resolve()
    bindings=[c.bind(worker),c.bind(__file__),c.bind(a.deployment),dep['planner']]
    tiny_jobs=[];target_jobs=[];formal=[]
    def job(identity,phase,name,arm,input_binding,repeat=0,tiny_proof=None,target_proof=None,depends=()):
        out=a.results_root/identity
        cmd=[dep['python_invocation'],'-B',str(worker),'--execute-gpu','--deployment',str(a.deployment.resolve()),
             '--support',dep['planner']['path'],'--inputs',input_binding['path'],'--dataset',name,
             '--arm',arm,'--phase',phase,'--repeat',str(repeat),'--gpu-uuids',resource['gpu-uuids'],
             '--cpu-set',resource['cpu-set'],'--old-readers','none','--output',str(out),
             '--output-store',str(a.results_root/'objects')]
        if tiny_proof:cmd+=['--tiny-proof',tiny_proof]
        if target_proof:cmd+=['--target-proof',target_proof]
        return dict(id=identity,command=cmd,receipt=str(out/'receipt.json'),
            expected_statuses=['PASS_UNWEIGHTED_PR_FORMAL' if phase=='formal' else 'PASS_UNWEIGHTED_PR_QUALIFICATION'],
            depends_on=list(depends),timeout=3600,bindings=bindings+[input_binding],
            environment=support.environment_for(dep,a.partitions),
            arm=arm,dataset=name,phase=phase,repeat=repeat,input_binding=input_binding)
    for arm in c.ARMS:
        tiny_jobs.append(job(f'tiny_{arm}_p{a.partitions}','tiny','TINY_A',arm,resource['tiny_inputs']))
    for name in ('US','IN','OR'):
        for arm in c.ARMS:
            tiny_job=next(x for x in tiny_jobs if x['arm']==arm)
            target_jobs.append(job(f'target_{name}_{arm}_p{a.partitions}','target',name,arm,inputs[name],
                tiny_proof=tiny_job['receipt'],depends=(tiny_job['id'],)))
        for repeat in range(10):
            arms=list(c.ARMS)
            if repeat%2:arms.reverse()
            for arm in arms:
                tiny_job=next(x for x in tiny_jobs if x['arm']==arm)
                target=next(x for x in target_jobs if x['arm']==arm and x['dataset']==name)
                formal.append(job(f'formal_{name}_{arm}_p{a.partitions}_r{repeat:02}','formal',name,arm,inputs[name],repeat,
                    tiny_proof=tiny_job['receipt'],target_proof=target['receipt']))
    a.output.mkdir(parents=True)
    common=dict(schema='e3.unweighted.pr.plan.v1',source_tree_sha256=freeze['source_tree_sha256'],
                environment=dep['environment'],gpu_executed=False,formal_measurements=0)
    save(a.output/'qualification_plan.json',dict(common,jobs=tiny_jobs+target_jobs,status='GPU_NOT_EXECUTED'))
    save(a.output/'formal_pending.json',dict(common,jobs=formal,status='NOT_ADMITTED',
        deployment=c.bind(a.deployment),worker=c.bind(worker),partitions=a.partitions,
        note='Requires fresh exact-control tiny and target proofs; no main-trace/v4proof substitution'))
    return dict(status='CPU_PLAN_ONLY',qualification_jobs=8,pending_formal_jobs=60,output=str(a.output),gpu_executed=False)


def admit(a):
    pending=c.read(a.bundle/'formal_pending.json')
    c.need(pending['status']=='NOT_ADMITTED','Wrong control plan')
    dep=c.read(c.checked(pending['deployment']))
    c.checked(pending['worker'])
    support=c.load_support(c.checked(dep['planner']))
    python_runtime=support.verify_runtime(dep)
    freeze=c.read(c.checked(dep['source_freeze']))
    support.source_files(Path(dep['source_root']),freeze)
    for key in ('python','native_dso','native_build_receipt','registry'):c.checked(dep[key])
    worker,contract,_=support.imports(dep['runner_root'])
    c.need(c.bind(sys.executable)['sha256']==dep['python']['sha256'],'Wrong admission Python')
    binding=contract.source_binding(Path(dep['source_root']),SimpleNamespace(__file__=dep['native_dso']['path']))
    binding['negative_control']={name:c.bind(Path(__file__).with_name(name)) for name in c.FILES}
    binding['admission_support']=dep['planner']
    seen={};validated={};jobs=[]
    for job in pending['jobs']:
        for item in job['bindings']:c.checked(item)
        args=c.parser().parse_args(job['command'][3:])
        record=worker.load_inputs(args.inputs)
        c.validate_arguments(args,record)
        key=(job['dataset'],job['arm'])
        seen.setdefault(key,[]).append(job['repeat'])
        if key not in validated:
            env=support.controlled_environment(support.environment_for(dep,pending['partitions']),worker.COMMON_ENV,args.gpu_uuids)
            config=dict(protocol=c.PROTOCOL,source_binding=binding,source_tree_sha256=freeze['source_tree_sha256'],
                deployment=pending['deployment'],arm=args.arm,route=c.ARMS[args.arm],policy='closure',old_readers='none',
                tiny_inputs=dep['resources'][str(pending['partitions'])]['tiny_inputs'],
                python_runtime=python_runtime,
                gpu_uuids=args.gpu_uuids.split(','),cpu_affinity=sorted(int(x) for x in args.cpu_set.split(',')),environment=env,
                quality=c.QUALITY,commits=c.COMMITS,queries=c.COMMITS+1,
                old_reader_limitation='Weighted input-view lease is inapplicable to this21-unweighted-PR-only trace')
            validated[key]=[c.validate_proof(path,phase,config,c.bind(args.inputs),args.dataset,contract)
                for phase,path in (('tiny',args.tiny_proof),('target',args.target_proof))]
        c.need(not args.output.exists(),'Existing formal attempt cannot be overwritten')
        jobs.append(dict(job,bindings=job['bindings']+validated[key]))
    c.need(set(seen)=={(name,arm) for name in ('US','IN','OR') for arm in c.ARMS} and
           all(sorted(v)==list(range(10)) for v in seen.values()),'Require six cells and ten independent repeats each')
    save(a.output,dict(pending,jobs=jobs,status='CPU_ADMITTED_GPU_NOT_EXECUTED',pending=c.bind(a.bundle/'formal_pending.json')))
    return dict(status='CPU_ADMITTED_GPU_NOT_EXECUTED',jobs=len(jobs),gpu_executed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    b=s.add_parser('build')
    for name in ('deployment','output','results-root'):b.add_argument('--'+name,type=Path,required=True)
    b.add_argument('--inputs',type=Path,nargs='+',required=True);b.add_argument('--partitions',type=int,choices=(1,8),default=1)
    r=s.add_parser('admit-formal');r.add_argument('--bundle',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a) if a.action=='build' else admit(a)))


if __name__=='__main__':main()
