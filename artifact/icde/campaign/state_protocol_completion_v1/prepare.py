#!/usr/bin/env python3
"""CPU-only draft/deployment preparation. This script never executes GPU jobs."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import argparse
import hashlib
import json
import sys
import state_plan as plan


ROOT = Path(__file__).resolve().parent


def source_freeze():
    files = {}
    for name in (*plan.BENCHMARK, 'state_plan.py', 'prepare.py'):
        binding = plan.bind(ROOT / name)
        files[name] = {k: binding[k] for k in ('bytes','sha256')}
    return dict(schema='state.completion.runner.freeze.v1', files=files,
        runner_sha256=plan.digest(files), gpu_executed=False)


def draft(base_coverage, output):
    coverage = plan.read(base_coverage)
    by_key = {plan.input_key(row): row['inputs'] for row in coverage['configurations']
              if row.get('inputs') and row['experiment'] in ('E3','E4')}
    cells = plan.grid()
    for row in cells:
        row['inputs'] = by_key.get(plan.input_key(row))
        row['input_status'] = 'BOUND_PATH_NOT_RECHECKED_HERE' if row['inputs'] else 'MISSING_INPUT'
    if any(row['inputs'] is None for row in cells):
        raise ValueError('Missing required canonical state input; do not silently drop cells')
    tiny_keys = sorted({tuple(row[k] for k in ('experiment','arm','old_readers','beta','order','partitions'))
                       for row in cells})
    qualification_specs = [dict(phase='tiny', experiment=e, arm=a, old_readers=o,
        beta=b, order=order, partitions=p) for e,a,o,b,order,p in tiny_keys]
    qualification_specs += [dict(phase='target',cell=row) for row in cells]
    qualification_specs += [dict(phase=phase, action='pilot',experiment='E4',partitions=p)
                             for phase in ('tiny','target') for p in (1,8)]
    formal = [dict(id=row['id']+f'_r{repeat:02}',cell=row,repeat=repeat,
                   status='NOT_ADMITTED_REQUIRES_EXACT_TINY_TARGET_AND_OUTPUT_PROOFS')
              for row in cells for repeat in range(10)]
    output.mkdir(parents=True,exist_ok=False)
    plan.save_new(output/'qualification_draft.json',dict(schema='state.completion.draft.v1',
        status='DRAFT_NOT_DEPLOYED_NOT_A_GPU_QUEUE',jobs=[],job_specs=qualification_specs,
        base_coverage=plan.bind(base_coverage),runner=source_freeze(),
        qualification_jobs_planned=len(qualification_specs),gpu_executed=False,formal_measurements=0))
    plan.save_new(output/'formal_pending.json',dict(schema='state.completion.formal.draft.v1',
        status='DRAFT_NOT_ADMITTED_NOT_A_GPU_QUEUE',jobs=[],job_specs=formal,
        configurations=len(cells),repetitions_per_configuration=10,formal_measurements=0))
    plan.save_new(output/'coverage.json',dict(status='CPU_DRAFT_ONLY',configurations=cells,
        counts=dict(Counter(r['experiment'] for r in cells)),
        qualification_jobs_planned=len(qualification_specs),formal_repetitions_planned=len(formal),
        all_baseline_and_control_cells_retained=True,gpu_executed=False,formal_measurements=0))
    return dict(status='CPU_DRAFT_ONLY',configurations=len(cells),
        qualification_jobs_planned=len(qualification_specs),formal_repetitions_planned=len(formal))


def deploy(base_deployment, base_coverage, output, results_root):
    # Run on the selected machine through its exact existing venv. All inputs,
    # source files, DSO, environment and source signer are checked by plan.build.
    dep = plan.read(base_deployment)
    if str(Path(sys.executable).absolute()) != dep['python_invocation']:
        raise ValueError('Use the exact lexical Python invocation from base deployment')
    freeze_path = ROOT/'freeze.json'
    actual = source_freeze()
    if not freeze_path.exists() or plan.read(freeze_path) != actual:
        raise ValueError('Runner freeze is missing or changed')
    coverage = plan.read(base_coverage)
    inputs = sorted({row['inputs']['path'] for row in coverage['configurations']
                     if row.get('inputs') and row['experiment'] in ('E3','E4')})
    for field in ('source_freeze','registry','template','native_build_receipt','profile_environment'):
        if dep.get(field): plan.checked(dep[field])
    args = SimpleNamespace(source=Path(dep['source_root']),freeze=Path(dep['source_freeze']['path']),
        runner=ROOT,registry=Path(dep['registry']['path']),
        deployment_template=Path(dep['template']['path']),
        native_build_receipt=Path(dep['native_build_receipt']['path']),
        profile_environment=Path(dep['profile_environment']['path']) if dep.get('profile_environment') else None,
        expected_source_sha256=dep['source_tree_sha256'],output=output,results_root=results_root,
        inputs=[Path(p) for p in inputs],scope='available',grid_domain='main',enable_p8_targets=True)
    result=plan.build(args)
    if result['configurations'] != 480 or result['pending_formal_jobs'] != 4800:
        raise ValueError('Full completion grid missing; preserve failed plan and investigate')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    d=sub.add_parser('draft');d.add_argument('--base-coverage',type=Path,required=True);d.add_argument('--output',type=Path,required=True)
    sub.add_parser('freeze')
    d=sub.add_parser('deploy')
    for name in ('base-deployment','base-coverage','output','results-root'):
        d.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.action=='freeze':
        plan.save_new(ROOT/'freeze.json',source_freeze());result={'status':'CPU_RUNNER_FROZEN'}
    elif args.action=='draft':result=draft(args.base_coverage,args.output)
    else:result=deploy(args.base_deployment,args.base_coverage,args.output,args.results_root)
    print(json.dumps(result))


if __name__=='__main__':main()
