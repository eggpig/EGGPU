"""Serial E1 launcher; plan-only unless --execute-gpu is explicitly supplied.

Use --repeat N for one repeat block, or omit it for all ten formal repeats.
Each job writes a new directory. A failed job stops this controller; retained
measurements are never replaced or filtered by speed. Root owns execution.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

from sg_worker import ALGORITHMS, DATASETS, TINY_DATASETS, binding, save


def split_choice(value, allowed):
    values = value.split(',')
    if not values or len(set(values)) != len(values) or any(x not in allowed for x in values):
        raise argparse.ArgumentTypeError('Expected unique comma-separated values from ' + ','.join(allowed))
    return values


def proof_index(paths):
    result = {}
    for path in paths:
        path = Path(path).resolve()
        record = json.loads(path.read_text())
        if record.get('status') != 'PASS_QUALIFICATION':
            raise ValueError('Qualification list contains a nonpassing receipt: ' + str(path))
        key = (record['phase'], record['system'], record['algorithm'], record['dataset'])
        if key in result:
            raise ValueError('Choose exactly one proof per qualification identity; no automatic latest/fastest selection')
        result[key] = path
    return result


def plan(args):
    datasets = split_choice(args.datasets, DATASETS + TINY_DATASETS)
    systems = split_choice(args.systems, ('eggpu', 'cugraph'))
    algorithms = split_choice(args.algorithms, ALGORITHMS)
    repeats = list(range(10)) if args.phase == 'formal' and args.repeat is None else [args.repeat or 0]
    if any(r < 0 or (args.phase == 'formal' and r >= 10) for r in repeats):
        raise ValueError('Formal repetitions are 0..9')
    if any((d in TINY_DATASETS) != (args.phase == 'tiny') for d in datasets):
        raise ValueError('Tiny fixtures and the seven main graphs have distinct phase scopes')
    proofs = proof_index(json.loads(args.proofs.read_text()) if args.proofs else [])
    jobs = []
    for repeat in repeats:
        for dataset in datasets:
            meta = json.loads((args.input_root / 'canonical' / dataset / 'canonical.json').read_text())
            tiny = 'TINY_D' if meta['directed'] else 'TINY_U'
            for algorithm in algorithms:
                # Balance ordering without inspecting measured time or selecting a backend.
                ordered = systems if repeat % 2 == 0 else list(reversed(systems))
                for system in ordered:
                    out = args.output / f'{args.phase}_{dataset}_{algorithm}_{system}_r{repeat:02d}'
                    executable = args.python_eggpu if system == 'eggpu' else args.python_cugraph
                    command = [str(executable), str(Path(__file__).with_name('sg_worker.py')),
                               '--execute-gpu', '--system', system, '--dataset', dataset,
                               '--algorithm', algorithm, '--phase', args.phase, '--repeat', str(repeat),
                               '--output', str(out)]
                    for name in ('input_root', 'release', 'arena_library_dir', 'native_dso', 'setup',
                                 'gpu_uuid', 'lock_file', 'reference_cache', 'budget_mib', 'min_host_free_gib'):
                        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
                    if args.phase in ('target', 'formal'):
                        key = ('tiny', system, algorithm, tiny)
                        if key not in proofs:
                            raise ValueError('Missing explicit tiny qualification: ' + str(key))
                        command += ['--tiny-proof', str(proofs[key])]
                    if args.phase == 'formal':
                        key = ('target', system, algorithm, dataset)
                        if key not in proofs:
                            raise ValueError('Missing explicit target qualification: ' + str(key))
                        command += ['--target-proof', str(proofs[key])]
                    jobs.append(dict(system=system, dataset=dataset, algorithm=algorithm,
                                     repeat=repeat, output=str(out), command=command))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-gpu', action='store_true')
    p.add_argument('--phase', choices=('tiny', 'diagnostic', 'target', 'formal'), default='diagnostic')
    p.add_argument('--datasets', default='EP')
    p.add_argument('--systems', default='eggpu,cugraph')
    p.add_argument('--algorithms', default='bfs,sssp,wcc,pr')
    p.add_argument('--repeat', type=int)
    for name in ('input-root', 'release', 'arena-library-dir', 'native-dso', 'setup', 'lock-file',
                 'reference-cache', 'output', 'python-eggpu', 'python-cugraph'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--proofs', type=Path, help='JSON list of explicit worker result.json qualification paths')
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--budget-mib', type=int, default=22000)
    p.add_argument('--min-host-free-gib', type=float, default=8.)
    args = p.parse_args()
    jobs = plan(args)
    if not args.execute_gpu:
        print(json.dumps({'gpu_executed': False, 'jobs': jobs}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'schema': 'eggpu.e1.sg.controller.v1', 'status': 'RUNNING', 'phase': args.phase,
             'controller': binding(__file__), 'jobs': jobs, 'completed': [],
             'started_unix': time.time(), 'whole_suite_claim': False,
             'formal_repetitions_required': 10, 'error_bars': 'aggregate only after all 10 independent repeat blocks pass'}
    save(args.output / 'controller.json', state)
    try:
        for index, item in enumerate(jobs):
            state['active_job'] = index
            save(args.output / 'controller.json', state)
            with (args.output / f'job_{index:04d}.log').open('wb') as log:
                completed = subprocess.run(item['command'], stdout=log, stderr=subprocess.STDOUT)
            result_path = Path(item['output']) / 'result.json'
            receipt = json.loads(result_path.read_text()) if result_path.exists() else {}
            state['completed'].append({'index': index, 'returncode': completed.returncode,
                'result': binding(result_path) if result_path.exists() else None,
                'status': receipt.get('status', 'FAILED_PROCESS')})
            if completed.returncode or receipt.get('status') not in ('PASS', 'PASS_QUALIFICATION'):
                raise RuntimeError(f'Job {index} failed; original output and log retained')
            save(args.output / 'controller.json', state)
        state['status'] = 'PASS_REQUESTED_JOBS'
    except BaseException as error:
        state.update(status='FAILED', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        state['finished_unix'] = time.time()
        save(args.output / 'controller.json', state)


if __name__ == '__main__':
    main()
