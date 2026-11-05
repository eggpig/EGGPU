"""Plan GUM canonical qualification or complete-task formal runs; execute serially on explicit opt-in."""
from pathlib import Path
from types import SimpleNamespace
import json,os,signal,subprocess,sys
import gum_worker as worker
import sg_worker as shared


def plan(args):
    proofs=json.loads(args.proofs.read_text()) if args.proofs else {}
    jobs=[]
    for count in [int(x) for x in args.counts.split(',')]:
        for dataset in args.datasets.split(','):
            for function in args.functions.split(','):
                key=f'{dataset}:{function}:p{count}';name=f'{args.phase}_{dataset}_{function}_p{count}_r{args.repeat}'
                local=SimpleNamespace(**vars(args));local.dataset=dataset;local.function=function;local.count=count
                local.gpu_uuids=','.join(args.gpu_uuids.split(',')[:count]);local.output=args.output/name
                local.tiny_proof=proofs.get(key,{}).get('tiny');local.target_proof=proofs.get(key,{}).get('target')
                worker.configure(local)
                if args.phase in ('target','formal') and not local.tiny_proof:raise ValueError('Missing tiny proof '+key)
                if args.phase=='formal' and not local.target_proof:raise ValueError('Missing target proof '+key)
                command=[str(args.core_python),'-B',str(Path(worker.__file__).resolve()),'--execute-gpu']
                for action in worker.parser()._actions:
                    if action.dest in ('help','execute_gpu','make_tiny'):continue
                    value=getattr(local,action.dest,None)
                    if value is not None:command.extend([action.option_strings[0],str(value)])
                jobs.append(dict(id=name,command=command,output=str(local.output),function=function,dataset=dataset,count=count,
                    T_first_available=True,timing_scope='continuous T_first and native T_ready'))
    return jobs[::-1] if args.repeat%2 else jobs


def main():
    p=worker.parser();p.description=__doc__
    p.add_argument('--datasets',default='TINY_D,TINY_U');p.add_argument('--functions',default='pr')
    p.add_argument('--counts',default='1,2,4,8');p.add_argument('--proofs',type=Path)
    args=p.parse_args()
    for key in ('gpu_uuids','cpu_set','output','core_python'):
        if getattr(args,key) is None:p.error('--'+key.replace('_','-')+' required')
    jobs=plan(args)
    if not args.execute_gpu:
        print(json.dumps(dict(status='PLAN_ONLY',gpu_executed=False,jobs=jobs),indent=2));return
    args.output.mkdir(parents=True,exist_ok=False);record=dict(status='RUNNING',jobs=jobs,completed=[])
    try:
        for job in jobs:
            child=None
            try:
                with (args.output/(job['id']+'.log')).open('x') as log:
                    child=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    code=child.wait(timeout=(args.timeout_seconds+300)*(3 if job['function'] in ('bfs','sssp') else 1)+300)
                    if code:raise RuntimeError('GUM job failed: '+job['id'])
                record['completed'].append(shared.binding(Path(job['output'])/'result.json'))
            finally:
                if child is not None and child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            shared.save(args.output/'controller.json',record)
        record['status']='PASS'
    except BaseException as error:record.update(status='FAIL',error=type(error).__name__+': '+str(error));raise
    finally:shared.save(args.output/'controller.json',record)

if __name__=='__main__':main()
