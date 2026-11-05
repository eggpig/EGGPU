#!/usr/bin/env python3
"""Run packaged campaign CPU contracts, with all temporary output outside the checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT/'Easy-Graph')
    args = p.parse_args(); work = args.work.resolve()
    if work.exists() or work == ROOT or ROOT in work.parents:
        raise ValueError('Choose an absent test directory outside the checkout')
    work.mkdir(parents=True); campaign = ROOT/'artifact/icde/campaign'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               EGGPU_TEST_SOURCE=str(args.source.resolve()), TMPDIR=str(work),
               PYTHONPATH=str(campaign)+os.pathsep+os.environ.get('PYTHONPATH',''))
    commands = [
        [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(campaign), '-p', 'test_*campaign.py', '-v'],
        [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(campaign), '-p', 'test_gum_clock.py', '-v'],
        [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(campaign), '-p', 'test_pr_global_l1.py', '-v'],
        [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--confcutdir='+str(campaign),
         '--basetemp='+str(work/'csr_entry_tmp'), str(campaign/'test_csr_runner_contract.py')],
        [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(campaign/'baselines'), '-p', 'test_sg*cpu.py', '-v'],
        [sys.executable, '-B', str(campaign/'state_cpu_tests.py'), '--source', str(args.source.resolve()), '--output', str(work/'state_cpu.json')],
        [sys.executable, '-B', str(campaign/'state_inputs.py'), '--source', str(args.source.resolve()), '--tiny', '--rho', '.1', '--output', str(work/'state_inputs')],
        [sys.executable, '-B', str(campaign/'state_projection_cpu_tests.py'), '--source', str(args.source.resolve()), '--inputs', str(work/'state_inputs/inputs.json'), '--output', str(work/'state_projection_cpu.json')],
        [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--confcutdir='+str(campaign),
         '--basetemp='+str(work/'state_v4_tmp'), str(campaign/'test_state_v4_cpu.py')],
        [sys.executable, '-B', str(campaign/'test_state_reference_cpu.py'), '--inputs', str(work/'state_inputs/inputs.json'), '--output', str(work/'state_reference_cpu.json')],
        [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--confcutdir='+str(ROOT/'artifact/icde'),
         '--basetemp='+str(work/'planning_tmp'), str(ROOT/'artifact/icde/presentation/test_e1_summary.py'),
         str(campaign/'state_final_plan/test_state_plan.py'), str(campaign/'test_campaign_queue_v4_cpu.py'),
         str(campaign/'e3_unweighted_control/test_pr_control.py')],
    ]
    record = {'schema': 'eggpu.public_campaign_cpu/v1', 'status': 'RUNNING', 'gpu_executed': False,
        'scope': 'SG/native/E6/sequence/MG CPU contracts and state v4/controls/reference checks; no GPU or network requests',
        'files': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(campaign.rglob('*.py'))}, 'steps': []}
    try:
        for index, command in enumerate(commands):
            with (work/f'cpu_{index}.log').open('x') as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            record['steps'].append({'command': command, 'returncode': result.returncode})
            if result.returncode:
                raise RuntimeError('CPU contract suite failed; inspect its retained log')
        record['status'] = 'PASS_CPU_ONLY'
    except BaseException as error:
        record.update(status='FAIL_CPU', error=type(error).__name__+': '+str(error))
        raise
    finally:
        (work/'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'status': record['status'], 'gpu_executed': False}))


if __name__ == '__main__':
    main()
