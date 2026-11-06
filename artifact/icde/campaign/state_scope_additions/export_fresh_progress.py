"""Recorded-evidence 38-cell exporter with explicit fresh-policy checks."""
from pathlib import Path
import argparse,json,time
import export_progress as base
from fresh_policy_support import checked_policy,inspect_pr

def install(plan_path):
    plan=base.read(plan_path);selection=base.read(Path(plan['original_state_plan']['path']).parent/'selection.json')
    policy,guard=checked_policy(selection)
    if plan['measurement_policy']!=selection['measurement_policy'] or plan['old_policy_results_reused'] is not False:
        raise ValueError('Additive plan is not the fresh policy epoch')
    original=base.validate_record
    def validate(data,job,*args,**kwargs):
        result=original(data,job,*args,**kwargs)
        result['policy_check']=inspect_pr(data) if job['first_pass_stage']=='unweighted_pr_negative_control' else guard.inspect(data)
        result['measurement_policy']=selection['measurement_policy'];return result
    base.validate_record=validate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--watch-seconds',type=int,default=0);a=p.parse_args()
    install(a.plan)
    # A separate output directory prevents cache reuse from the previous policy.
    marker=a.output/'policy_export_identity.json';identity=dict(exporter=base.bind(__file__),plan=base.bind(a.plan))
    a.output.mkdir(parents=True,exist_ok=True)
    if marker.exists() and base.read(marker)!=identity:raise ValueError('Use a fresh export directory for a changed policy/exporter')
    if not marker.exists():
        if (a.output/'cache.json').exists():raise ValueError('Fresh policy exporter cannot reuse unverified old cache')
        marker.write_text(json.dumps(identity,indent=2)+'\n')
    while True:
        result=base.export(a.plan,a.output);print(json.dumps({k:result[k] for k in ('eligible_n1','additive_denominator','status_counts')}),flush=True)
        if not a.watch_seconds:break
        time.sleep(max(15,a.watch_seconds))
