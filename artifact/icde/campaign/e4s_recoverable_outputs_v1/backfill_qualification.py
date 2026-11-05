"""Explicit frozen E4S qualification plan -> post-timing recovery sidecars.
No GPU, no worker reruns, no conversion of qualification into formal evidence.
"""
from pathlib import Path
import argparse,contextlib,fcntl,json
import archive_outputs as archive

def run(plan_path,plan_sha,store,controller_lock,execute=False):
 plan_binding=archive.binding(plan_path);archive.require(plan_binding['sha256']==plan_sha,'Frozen qualification plan SHA differs');plan=archive.load(plan_path);jobs=plan['jobs'];archive.require(all(j.get('phase') in ['tiny','target'] and j['expected_statuses']==['PASS_E4S_QUALIFICATION'] for j in jobs),'Only explicit E4S qualification plans accepted')
 rows=[]
 with (Path(controller_lock).open('a+') if execute else contextlib.nullcontext()) as lock:
  if execute:fcntl.flock(lock,fcntl.LOCK_EX)
  for job in jobs:
   p=Path(job['receipt']);r=archive.load(p) if p.exists() else {};row=dict(job=job['id'],worker_status=r.get('status','ABSENT'),sidecar=str(p.parent/'recoverable_outputs.json'))
   if r.get('status')!='PASS_E4S_QUALIFICATION':row['status']='NO_QUALIFIED_COMPLETE_OUTPUT';rows.append(row);continue
   if not execute:row['status']='RECOVERY_PRESENT_UNVERIFIED' if Path(row['sidecar']).exists() else 'RECOVERY_REQUIRED';rows.append(row);continue
   try:
    cmd=job['command'];root=cmd[cmd.index('--input-root')+1]
    if not Path(row['sidecar']).exists():archive.archive_result(p,root,store,row['sidecar'])
    row['proof']=archive.validate_sidecar(p,row['sidecar'],archive.binding(archive.__file__));row['status']='PASS_QUALIFICATION_RECOVERY'
   except Exception as error:row.update(status='FAIL_RECOVERY',error=type(error).__name__+': '+str(error))
   rows.append(row)
 return dict(schema='e4s.qualification.recovery.backfill.v1',plan=plan_binding,gpu_executed=False,worker_reruns=0,formal_admitted=0,executed=execute,rows=rows)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--store',type=Path,required=True);p.add_argument('--controller-lock',type=Path,required=True);p.add_argument('--archive',action='store_true');args=p.parse_args();r=run(args.plan,args.plan_sha,args.store,args.controller_lock,args.archive);print(json.dumps(r,indent=2));raise SystemExit(2 if any(x['status']=='FAIL_RECOVERY' for x in r['rows']) else 0)
