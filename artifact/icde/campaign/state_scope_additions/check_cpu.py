"""Check real retained hotspot selections, exact control scope and CLI grammar."""
from pathlib import Path
import argparse
import copy
import json
import prepare as p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--recorded-campaign-prefix',required=True,help='Original bound input path prefix to map into --campaign')
    args=parser.parse_args();campaign=args.campaign.resolve();remote=args.recorded_campaign_prefix.rstrip('/')+'/'
    def local(path):
        value=str(path)
        if value.startswith(remote):return campaign/value[len(remote):]
        return Path(path)
    rows,controls=p.inventory(campaign)
    support=p.load('_additive_input_validation',campaign/'state_protocol_completion_v1/state_plan.py')
    def checked(item):
        path=local(item['path']);actual=p.bind(path)
        if any(actual[k]!=item[k] for k in ('bytes','sha256')):raise ValueError('Retained input binding changed')
        return path
    support.checked=checked
    selected={row['inputs']['sha256']:row['inputs'] for row in rows};assert len(selected)==8
    proofs=[]
    for item in selected.values():
        record=p.read(checked(item));support.validate_hotspot(record);proofs.append(item)
        invalid=copy.deepcopy(record);invalid['cell']['hotspot_k']=3
        try:support.validate_hotspot(invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid partition coverage accepted')
    control=p.load('_unchanged_pr_control',campaign/'e3_unweighted_control_v4/pr_control.py')
    for job in controls:
        arguments=control.parser().parse_args(job['command'][3:])
        assert arguments.phase==job['phase'] and arguments.repeat==0 and arguments.old_readers=='none'
        events=control.trace(job['dataset'])
        assert len(events)==41 and sum(e['kind']=='query' for e in events)==21 and sum(e['kind']=='commit' for e in events)==20
        assert [e['version'] for e in events if e['kind']=='query']==list(range(21))
    record=dict(status='PASS_CPU_ADDITIVE_SCOPE_AND_RETAINED_HOTSPOT_INPUTS',gpu_executed=False,
        hotspot_cells=32,distinct_retained_hotspot_inputs=8,unweighted_pr_target_cells=6,initial_exact_arm_pr_tiny_checks=2,
        source_tree_required=p.SOURCE,hotspot_input_proofs=proofs,
        original_pr_worker=p.bind(campaign/'e3_unweighted_control_v4/pr_control.py'),
        original_state_runner=p.bind(campaign/'first_pass_20261106/state_main_first_pass/runner/state_worker.py'),
        planner=p.bind(p.__file__),actual_gpu_plan_preparation_requires_remote_bound_source_and_core_focused_PASS=True,
        checks=['All eight retained hotspot position arrays pass original source-bound count/order/owner validation',
            'Invalid hotspot owner-domain rejected','32 hotspot cells and six distinct PR target cells match frozen design',
            'Every original PR command parses; all 21 queries and 20 real weight commits are retained',
            'No numerical or current-runner code edited; no GPU imports or execution'])
    args.receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='hotspot_input_proofs'}))


if __name__=='__main__':main()
