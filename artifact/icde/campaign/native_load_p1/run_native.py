"""Serial fixed-native controller for E1-MG and E2. Default is plan-only.

Choose explicit qualification receipts with --proofs JSON_PATH. No result-based
backend selection, replacement of native P1, or automatic rerun selection occurs.
"""
from pathlib import Path
import argparse
import json
import subprocess
import time

import sg_worker as shared
import native_worker
import native_profile
from run_sg import split_choice


def proof_index(paths):
    proofs = {}
    for value in paths:
        path = Path(value).resolve(); data = json.loads(path.read_text())
        if data.get('status') != 'PASS_QUALIFICATION' or data.get('implementation') != 'forced_native':
            raise ValueError('Only explicit passing forced-native qualification receipts are accepted')
        key = (data['phase'], data['algorithm'], data['dataset'], data['gpu_count'])
        if key in proofs:
            raise ValueError('Choose one explicit proof per identity; no newest/fastest auto-selection')
        proofs[key] = path
    return proofs


def plan(args):
    communication = getattr(args, 'communication', 'profile')
    if communication not in ('profile', 'host_staging', 'nccl'):
        raise ValueError('Use one explicit communication option for all counts')
    datasets = split_choice(args.datasets, shared.DATASETS + shared.TINY_DATASETS)
    algorithms = split_choice(args.algorithms, shared.ALGORITHMS)
    counts = [int(value) for value in split_choice(args.counts, ('1', '2', '4', '8'))]
    uuids = args.gpu_uuids.split(',')
    if len(uuids) < max(counts) or len(set(uuids)) != len(uuids):
        raise ValueError('Provide enough distinct ordered UUIDs for the largest requested P')
    if args.experiment == 'E1-MG' and counts != [8]:
        raise ValueError('E1-MG main panel requires --counts 8')
    if args.experiment == 'E2' and any(d not in native_worker.E2_DATASETS + shared.TINY_DATASETS for d in datasets):
        raise ValueError('E2 is frozen to US/OR/IN/UK/TW')
    if any((d in shared.TINY_DATASETS) != (args.phase == 'tiny') for d in datasets):
        raise ValueError('Tiny fixture and real graph phases are separate')
    repeats = list(range(10)) if args.phase == 'formal' and args.repeat is None else [args.repeat or 0]
    if any(r < 0 or (args.phase == 'formal' and r >= 10) for r in repeats):
        raise ValueError('Formal repeats are 0..9')
    proofs = proof_index(json.loads(args.proofs.read_text()) if args.proofs else [])
    jobs = []
    for repeat in repeats:
        for dataset in datasets:
            meta = json.loads((args.input_root / 'canonical' / dataset / 'canonical.json').read_text())
            tiny = 'TINY_D' if meta['directed'] else 'TINY_U'
            for algorithm in algorithms:
                order = counts if repeat % 2 == 0 else list(reversed(counts))
                for count in order:
                    output = args.output / f'{args.phase}_{dataset}_{algorithm}_native_p{count}_r{repeat:02d}'
                    command = [str(args.python), str(Path(__file__).with_name('native_worker.py')),
                        '--execute-gpu', '--experiment', args.experiment, '--phase', args.phase,
                        '--dataset', dataset, '--algorithm', algorithm, '--count', str(count),
                        '--gpu-uuids', ','.join(uuids[:count]), '--repeat', str(repeat), '--output', str(output)]
                    selected_communication = native_profile.environment(algorithm,count)['EGGPU_COMM_BACKEND'] if communication == 'profile' else communication
                    command += ['--communication', selected_communication]
                    for name in ('input_root', 'release', 'native_dso', 'setup', 'lock_file', 'reference_cache',
                                 'budget_mib', 'retained_budget_mib', 'min_host_free_gib'):
                        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
                    if args.keep_prepared:
                        command.append('--keep-prepared')
                    if args.phase in ('target', 'formal'):
                        key = ('tiny', algorithm, tiny, count)
                        if key not in proofs:
                            raise ValueError('Missing fixed-native tiny proof: ' + str(key))
                        command += ['--tiny-proof', str(proofs[key])]
                    if args.phase == 'formal':
                        key = ('target', algorithm, dataset, count)
                        if key not in proofs:
                            raise ValueError('Missing fixed-native target proof: ' + str(key))
                        command += ['--target-proof', str(proofs[key])]
                    jobs.append(dict(experiment=args.experiment, system='eggpu', implementation='forced_native',
                                     dataset=dataset, algorithm=algorithm, gpu_count=count, repeat=repeat,
                                     communication=selected_communication,
                                     output=str(output), command=command))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-gpu', action='store_true')
    p.add_argument('--experiment', choices=('E1-MG', 'E2'), default='E2')
    p.add_argument('--phase', choices=('tiny', 'diagnostic', 'target', 'formal'), default='diagnostic')
    p.add_argument('--datasets', default='US,OR,IN,UK,TW')
    p.add_argument('--algorithms', default='bfs,sssp,wcc,pr')
    p.add_argument('--counts', default='1,2,4,8')
    p.add_argument('--gpu-uuids', required=True)
    p.add_argument('--communication', choices=('profile', 'host_staging', 'nccl'), default='profile')
    p.add_argument('--repeat', type=int)
    for name in ('input-root', 'release', 'native-dso', 'setup', 'lock-file', 'reference-cache', 'output', 'python'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--proofs', type=Path)
    p.add_argument('--budget-mib', type=int, default=22000)
    p.add_argument('--retained-budget-mib', type=int, default=0)
    p.add_argument('--min-host-free-gib', type=float, default=8)
    p.add_argument('--keep-prepared', action='store_true')
    args = p.parse_args()
    jobs = plan(args)
    if not args.execute_gpu:
        print(json.dumps({'gpu_executed': False, 'jobs': jobs}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(schema='eggpu.native.controller.v1', status='RUNNING', experiment=args.experiment,
        phase=args.phase, implementation='forced_native', controller=shared.binding(__file__), jobs=jobs,
        completed=[], started_unix=time.time(), whole_suite_claim=False,
        formal_repetitions_required=10,
        comparison_boundaries=['T_first: common host CSR through partition/import/full output',
                               'T_ready: retained native state, fresh query/full output after one warmup'])
    shared.save(args.output / 'controller.json', state)
    try:
        for index, job in enumerate(jobs):
            state['active_job'] = index; shared.save(args.output / 'controller.json', state)
            with (args.output / f'job_{index:04d}.log').open('wb') as log:
                completed = subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT)
            path = Path(job['output']) / 'result.json'
            record = json.loads(path.read_text()) if path.exists() else {}
            state['completed'].append(dict(index=index, returncode=completed.returncode,
                status=record.get('status', 'FAILED_PROCESS'), result=shared.binding(path) if path.exists() else None))
            if completed.returncode or record.get('status') not in ('PASS', 'PASS_QUALIFICATION'):
                raise RuntimeError(f'Native job {index} failed; retained evidence is not overwritten')
            shared.save(args.output / 'controller.json', state)
        state['status'] = 'PASS_REQUESTED_JOBS'
    except BaseException as error:
        state.update(status='FAILED', error=type(error).__name__+': '+str(error))
        raise
    finally:
        state['finished_unix'] = time.time()
        shared.save(args.output / 'controller.json', state)


if __name__ == '__main__':
    main()
