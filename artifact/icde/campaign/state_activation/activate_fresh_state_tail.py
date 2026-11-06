from pathlib import Path
import subprocess,json,os
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--campaign',type=Path,required=True)
parser.add_argument('--python',required=True)
args=parser.parse_args()
p=args.campaign.resolve();h=p/'first_pass_20261106';py=args.python;out=h/'state_tail_pr_v1'
cmd=[py,'-B',str(h/'state_audit_scheduling/state_tail_successor_v2.py'),'prepare','--campaign',str(p),'--bundle',str(h/'state_main_pr_v1'),'--core-plan',str(h/'state_main_pr_v1/plan.json'),'--core-queue',str(h/'state_main_pr_v1/queue'),'--admit-helper',str(h/'state_main_first_pass/admit_tail.py'),'--output',str(out),'--controller',str(p/'run_campaign_queue_v9.py'),'--wall-timeout-floor','18000']
for name in ('unified_main','state_main','missing_external','unified_scaling','unified_mechanism_application','author_reservation_handoff','state_scope_additions_pr_v1','sssp_subset_plan'):
 cmd+=['--wait-stage',str(h/name/'stage_manifest.json')]
subprocess.run(cmd,check=True)
run=[py,'-B',str(h/'state_audit_scheduling/state_tail_successor_v2.py'),'run','--manifest',str(out/'manifest.json'),'--execute-launch']
with (out/'successor.log').open('xb') as log:child=subprocess.Popen(run,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
receipt=dict(status='FRESH_POLICY_TAIL_SCHEDULED',pid=child.pid,command=run,targets=426,core_and_tail_targets=504,major_wait_stages=8,old_policy_results_reused=False)
with (out/'launch.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
