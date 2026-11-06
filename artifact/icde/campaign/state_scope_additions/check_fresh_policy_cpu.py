"""CPU-only explicit-policy admission checks; no GPU result is created."""
from pathlib import Path
import argparse,ast,copy,difflib,json,tempfile
import fresh_policy_support as support
from prepare import bind,SOURCE

def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    here=Path(__file__).resolve().parent;guard=here.parent/'state_pr_policy_v1/policy_guard.py';checks=[]
    for name in ('prepare_fresh_policy.py','fresh_policy_support.py','export_fresh_progress.py'):ast.parse((here/name).read_text())
    with tempfile.TemporaryDirectory(prefix='fresh-additions-',dir=a.scratch) as directory:
        path=Path(directory)/'policy.json';policy=dict(source_tree_sha256=SOURCE,environment=support.EXPECTED,old_measurements_reused=False,guard=bind(guard));path.write_text(json.dumps(policy));selection={'measurement_policy':bind(path)}
        actual,module=support.checked_policy(selection);assert actual==policy
        events=[dict(kind='query',function='weighted_pr',index=i,details={'metrics':dict(module.EXPECTED_METRICS,host_transpose='auto',host_transpose_method='packed_destination_owned')}) for i in range(8)]
        assert module.inspect(dict(environment=support.EXPECTED,devices=list(range(8)),events=events))['status']=='PASS_ACTUAL_PR_POLICY'
        checks.append('Fresh source71 policy and actual P8 focused route admitted')
        pr=dict(configuration={'environment':support.EXPECTED},events=[dict(kind='query',details={'metrics':dict(module.EXPECTED_METRICS,host_transpose='auto',host_transpose_method='whole_contiguous')}) for _ in range(21)])
        assert support.inspect_pr(pr)['queries']==21;checks.append('All21 P1 control query policy metrics admitted')
        for key,value in [('device_metadata_policy','full'),('host_transpose_method','general')]:
            bad=copy.deepcopy(pr);bad['events'][7]['details']['metrics'][key]=value
            try:support.inspect_pr(bad)
            except ValueError:checks.append(key+' mismatch rejected')
            else:raise AssertionError('Old policy admitted')
        policy['old_measurements_reused']=True;path.write_text(json.dumps(policy))
        try:support.checked_policy({'measurement_policy':bind(path)})
        except ValueError:checks.append('Old measurement reuse rejected')
        else:raise AssertionError('Reuse admitted')
    text=(here/'prepare_fresh_policy.py').read_text()
    assert "guard.inspect(focus_data)" in text and "pr_dep['environment'].update(policy['environment'])" in text
    assert 'validate_input_metadata' in text and 'support.source_files' in text and 'expected_source' in text
    delta=''.join(difflib.unified_diff((here/'prepare.py').read_text().splitlines(True),text.splitlines(True),fromfile='prepare.py',tofile='prepare_fresh_policy.py'))
    (here/'fresh_planner.diff').write_text(delta)
    record=dict(status='PASS_CPU_FRESH_ADDITIVE_POLICY_ADMISSION',gpu_executed=False,fixture_only=True,checks=checks,
        files={name:bind(here/name) for name in ('prepare_fresh_policy.py','fresh_policy_support.py','export_fresh_progress.py','fresh_planner.diff')},
        source71=SOURCE,target_cells=38,initial_PR_checks=2,hotspot_requires_new_focused_PASS=True)
    a.receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

if __name__=='__main__':main()
