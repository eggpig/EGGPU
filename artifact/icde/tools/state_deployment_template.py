"""Create an explicit local P1/P8 resource template; no GPU discovery/execution."""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parents[3]
def create(a):
    cpus=a.cpu_set.split(',');uuids=a.gpu_uuids.split(',')
    if len(cpus)!=16 or len(set(cpus))!=16 or any(not x.isdigit() for x in cpus):raise ValueError('Exactly sixteen distinct CPU IDs required')
    if len(uuids)!=8 or len(set(uuids))!=8 or any(not x.startswith('GPU-') for x in uuids):raise ValueError('Eight explicit ordered GPU UUIDs required')
    env=json.loads(a.environment.read_text())
    if not isinstance(env,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in env.items()):raise ValueError('Explicit string-to-string environment JSON required')
    jobs=[]
    for count,path in [(1,a.tiny_p1),(8,a.tiny_p8)]:
        if not path.is_file():raise ValueError('Prepare explicit CPU tiny inputs first')
        command=[str(a.python.resolve()),'-B',str(ROOT/'artifact/icde/campaign/state_worker.py'),'--native-dso',str(a.native_dso.resolve()),'--gpu-uuids',','.join(uuids[:count]),'--cpu-set',a.cpu_set,'--lock',str(a.lock.resolve()),'--inputs',str(path.resolve())]
        jobs.append(dict(id='pilot_p'+str(count),command=command))
    return dict(schema='eggpu.public.state.deployment-template.v1',environment=env,jobs=jobs,gpu_executed=False,scope='Resource template only; source-bound planner constructs and admits fresh execution jobs.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['python','native-dso','lock','tiny-p1','tiny-p8','environment','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cpu-set',required=True);p.add_argument('--gpu-uuids',required=True);a=p.parse_args();record=create(a)
    with a.output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
