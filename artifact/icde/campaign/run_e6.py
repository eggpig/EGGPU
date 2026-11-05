"""Plan or execute serial E6 workflows. Root alone supplies --execute-gpu.

EP/LJ are main workloads; OR is a separately labelled pressure workload.
No fallback, candidate, dataset or implementation is selected from timings.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import e6_worker as e6
import sg_worker as sg


def split(value, allowed):
    values = value.split(',') if value else []
    if not values or len(set(values)) != len(values) or any(v not in allowed for v in values):
        raise ValueError('Expected unique choices from '+','.join(map(str, allowed)))
    return values


def proof_index(paths):
    proofs = {}
    for name in paths:
        path = Path(name).resolve(); record = json.loads(path.read_text())
        if record.get('status') != 'PASS_QUALIFICATION' or record.get('quality_pass') is not True:
            raise ValueError('Only successful E6 qualifications may be selected')
        key = (record['phase'], record['dataset'], record['system'], record['ablate'], record['gpu_count'])
        if key in proofs:
            raise ValueError('Choose exactly one proof per identity; no automatic latest or fastest selection')
        proofs[key] = path
    return proofs


def plan(args):
    datasets = split(args.datasets, (*e6.DATASETS, e6.TINY))
    systems = split(args.systems, e6.SYSTEMS)
    counts = list(map(int, split(args.native_gpu_counts, ('1', '2', '4', '8'))))
    if any((dataset == e6.TINY) != (args.phase == 'tiny') for dataset in datasets):
        raise ValueError('Tiny and real datasets require their respective phases')
    if len(set(args.gpu_uuids)) != len(args.gpu_uuids):
        raise ValueError('GPU UUID list contains duplicate physical devices')
    repeats = list(range(10)) if args.phase == 'formal' and args.repeat is None else [args.repeat or 0]
    if any(r < 0 or (args.phase == 'formal' and r >= 10) for r in repeats):
        raise ValueError('Formal repeats are 0..9')
    configs = []
    for system in systems:
        if system == 'eggpu':
            configs.extend((system, 'none', count) for count in counts)
            if args.coverage_ablation:
                if 1 not in counts:
                    raise ValueError('Four-group coverage ablation needs the unchanged EGGPU SG control')
                configs.extend((system, group, 1) for group in e6.GROUPS)
        else:
            configs.append((system, 'none', 0 if system == 'igraph' else 1))
    if args.coverage_ablation and 'eggpu' not in systems:
        raise ValueError('Coverage ablation requires EGGPU')
    if len(args.gpu_uuids) < max(count for _, _, count in configs):
        raise ValueError('Explicit UUID list is shorter than the requested GPU count')
    proofs = proof_index(json.loads(args.proofs.read_text()) if args.proofs else [])
    jobs = []
    for repeat in repeats:
        for dataset in datasets:
            ordered = configs if repeat % 2 == 0 else list(reversed(configs))
            for system, ablate, count in ordered:
                output = args.output/f'{args.phase}_{dataset}_{system}_{ablate}_p{count}_r{repeat:02d}'
                python = args.python_cugraph if system.startswith('cugraph') else args.python_core
                command = [str(python), '-B', str(Path(__file__).with_name('e6_worker.py')), '--execute-gpu',
                    '--system', system, '--ablate', ablate, '--dataset', dataset, '--phase', args.phase,
                    '--repeat', str(repeat), '--gpu-count', str(count), '--gpu-uuids', ','.join(args.gpu_uuids[:count]),
                    '--output', str(output)]
                for name in ('input_root', 'release', 'native_dso', 'arena_library_dir', 'setup', 'lock_file',
                             'reference_cache', 'budget_mib', 'min_host_free_gib'):
                    command += ['--'+name.replace('_', '-'), str(getattr(args, name))]
                if args.fallback_site:
                    command += ['--fallback-site', str(args.fallback_site)]
                if args.build_receipt:
                    command += ['--build-receipt', str(args.build_receipt)]
                if system == 'eggpu' and args.build_receipt is None:
                    raise ValueError('Native jobs require an explicit successful build receipt')
                for kind in (('tiny',) if args.phase == 'target' else ('tiny', 'target') if args.phase == 'formal' else ()):
                    key = (kind, e6.TINY if kind == 'tiny' else dataset, system, ablate, count)
                    if key not in proofs:
                        raise ValueError('Missing explicit E6 qualification '+str(key))
                    command += ['--'+kind+'-proof', str(proofs[key])]
                jobs.append(dict(dataset=dataset, dataset_role='pressure' if dataset == 'OR' else
                    'qualification' if dataset == e6.TINY else 'main', system=system, ablate=ablate,
                    gpu_count=count, repeat=repeat, output=str(output), command=command))
    return jobs


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-gpu', action='store_true')
    p.add_argument('--phase', choices=('tiny', 'diagnostic', 'target', 'formal'), default='diagnostic')
    p.add_argument('--datasets', default='EP,LJ')
    p.add_argument('--systems', default=','.join(e6.SYSTEMS))
    p.add_argument('--native-gpu-counts', default='1')
    p.add_argument('--coverage-ablation', action='store_true')
    p.add_argument('--gpu-uuids', default='')
    p.add_argument('--repeat', type=int)
    p.add_argument('--budget-mib', type=int, default=22000)
    p.add_argument('--min-host-free-gib', type=float, default=8.)
    for name in ('input-root', 'release', 'native-dso', 'arena-library-dir', 'setup', 'lock-file',
                 'reference-cache', 'output', 'python-core', 'python-cugraph'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('proofs', 'fallback-site', 'build-receipt'):
        p.add_argument('--'+name, type=Path)
    return p


def main():
    args = parser().parse_args(); args.gpu_uuids = args.gpu_uuids.split(',') if args.gpu_uuids else []
    jobs = plan(args)
    if not args.execute_gpu:
        print(json.dumps({'gpu_executed': False, 'jobs': jobs, 'contract': e6.CONTRACT}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(schema='eggpu.e6.controller/v1', status='RUNNING', controller=sg.binding(__file__),
        protocol=e6.PROTOCOL, jobs=jobs, completed=[], requested_scope_only=True,
        formal_repetitions_required=10, started_unix=time.time())
    sg.save(args.output/'controller.json', state)
    try:
        for index, item in enumerate(jobs):
            state['active_job'] = index; sg.save(args.output/'controller.json', state)
            with (args.output/f'job_{index:04d}.log').open('wb') as log:
                result = subprocess.run(item['command'], stdout=log, stderr=subprocess.STDOUT)
            path = Path(item['output'])/'result.json'
            receipt = json.loads(path.read_text()) if path.exists() else {}
            state['completed'].append(dict(index=index, returncode=result.returncode,
                receipt=sg.binding(path) if path.exists() else None, status=receipt.get('status', 'FAILED_PROCESS')))
            if result.returncode or receipt.get('status') not in ('PASS', 'PASS_QUALIFICATION'):
                raise RuntimeError('E6 job failed; retained output/log preserved, no automatic substitution')
            sg.save(args.output/'controller.json', state)
        state['status'] = 'PASS_REQUESTED_JOBS'
    except BaseException as error:
        state.update(status='FAILED', error=type(error).__name__+': '+str(error))
        raise
    finally:
        state['finished_unix'] = time.time(); sg.save(args.output/'controller.json', state)


if __name__ == '__main__':
    main()
