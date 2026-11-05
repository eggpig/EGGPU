#!/usr/bin/env python3
"""Isolated CPU-only rebuilds adding complete-result monotonic observations.

Fixed author algorithm, numeric type, compile flags, preparation, and output
contracts are inherited from the qualified v2 build receipts. Nothing executes
the resulting GPU binaries. Each unit retains exact source and patch evidence.
"""
from pathlib import Path
import argparse, difflib, hashlib, json, os, re, shutil, subprocess, time


def bind(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def units(root, campaign):
    return {
        'gunrock_essentials': root/'build/gunrock/receipt.json',
        'gunrock_legacy_cc': campaign/'baselines/gunrock_cc_timer_build_v1/receipt.json',
        'cggraph_v15': root/'build/cggraph/receipt.json',
        'cggraph_legacy_integer': campaign/'baselines/cggraph_legacy_traversal_build_v2/receipt.json',
        'cggraph_legacy_pr': campaign/'baselines/cggraph_legacy_full_build_v1/receipt.json',
        'sygraph_traversal': root/'build/sygraph_v6/receipt.json',
        'sygraph_cc': root/'build/sygraph_coverage_v2/receipt.json',
        'grapin_static': root/'build/grapin_pma_boundary_v1/static/receipt.json',
        'grapin_cc': root/'build/grapin_pma_boundary_v1/cc/receipt.json',
        'grapin_sssp': root/'build/grapin_static_sssp_v1/receipt.json',
        'hytgraph': campaign/'baselines/hytgraph_build_v4/receipt.json',
    }


def commands(raw):
    if raw.get('command'):
        return [raw['command']]
    if raw.get('commands'):
        return raw['commands']
    return [row.get('argv', row.get('command')) for row in raw['steps']]


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected completion observation context: ' + old[:100])
    return text.replace(old, new, 1)


def observe(text, unit):
    if unit == 'gunrock_essentials':
        old = 'active+=sec(start,end);dumpOutput(argv[1],c.sequence,v);'
        return once(text, old, 'active+=sec(start,end);eggpu_sg_complete_clock(c.sequence,v.size());dumpOutput(argv[1],c.sequence,v);')
    if unit == 'cggraph_v15':
        return once(text, '  dumpOutput(argv[2],c.sequence,v);', '  eggpu_sg_complete_clock(c.sequence,v.size());\n  dumpOutput(argv[2],c.sequence,v);')
    if unit == 'sygraph_traversal':
        old = 'active+=sec(start,end);dumpOutput(argv[2],c.sequence,values);'
        return once(text, old, 'active+=sec(start,end);eggpu_sg_complete_clock(c.sequence,values.size());dumpOutput(argv[2],c.sequence,values);')
    if unit == 'sygraph_cc':
        old = '  queue.wait_and_throw();auto end=Clock::now();'
        return once(text, old, old+'\n  eggpu_sg_complete_clock(r,values.size());')
    if unit.startswith('cggraph_legacy'):
        old = '            auto recent_b=RecentClock::now();'
        return once(text, old, old+'\n            eggpu_sg_complete_clock(runId,complete.size());')
    if unit == 'gunrock_legacy_cc':
        old = '    auto complete_end = std::chrono::steady_clock::now();'
        return once(text, old, old+'\n    eggpu_sg_complete_clock(iter,owned_outputs.back().size());')
    if unit == 'grapin_sssp':
        old = '  GROUTE_CUDA_CHECK(cudaDeviceSynchronize());const auto end=EggpuClock::now();'
        return once(text, old, old+'\n  eggpu_sg_complete_clock(call,outputs.back().size());')
    if unit.startswith('grapin_'):
        old = '        const auto query_end=Clock::now();'
        return once(text, old, old+'\n        eggpu_sg_complete_clock(call,outputs.back().size());')
    if unit == 'hytgraph':
        old = '        times.push_back(std::chrono::duration<double>(Clock::now()-query_begin).count());masses.push_back(mass);'
        return once(text, old, old+'\n        eggpu_sg_complete_clock(call,outputs.back().size());')
    raise ValueError(unit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    base = units(args.root, args.campaign)[args.unit]
    raw = json.loads(base.read_text())
    if raw['status'] not in ('PASS', 'PASS_BUILD_ONLY'):
        raise ValueError('Base author build has not passed')
    original_binaries = raw.get('binaries') or {'native': raw['binary'] if isinstance(raw['binary'], dict)
        else {'path': raw['binary'], 'sha256': raw['binary_sha256']}}
    for item in original_binaries.values():
        if bind(item['path'])['sha256'] != item['sha256']:
            raise ValueError('Frozen author binary changed')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    header = output/'sg_complete_clock.h'
    shutil.copy2(Path(__file__).with_name(header.name), header)
    argv_list = commands(raw)
    is_cmake = '-S' in argv_list[0]
    tree = is_cmake or args.unit.startswith('cggraph_legacy')
    files = []
    if tree:
        source = output/'source'
        shutil.copytree(base.parent/'source', source,
            ignore=shutil.ignore_patterns('.git', '__pycache__', 'build'))
        if is_cmake:
            prefix = 'samples' if args.unit == 'hytgraph' else 'source/samples'
            algorithms = {'hytgraph': ('bfs','sssp','cc','pr'), 'grapin_static': ('bfs','pr'),
                          'grapin_cc': ('cc',), 'grapin_sssp': ('sssp',)}[args.unit]
            files = [source/prefix/('hybrid_'+algorithm)/('hybrid_'+algorithm+'.cu') for algorithm in algorithms]
        else:
            files = [source/'project/main_enginer.hpp']
        argv_list = [[x.replace(str(base.parent), str(output)) for x in command] for command in argv_list]
    else:
        original = next(Path(x) for x in argv_list[0] if x.endswith(('.cu', '.cpp')))
        target = output/original.name
        text = original.read_text()
        # Keep relative include resolution identical after relocating only this
        # wrapper; all author headers remain untouched and are receipt-bound.
        text = re.sub(r'#include "([^"]+)"', lambda m: '#include "'+str((original.parent/m[1]).resolve())+'"', text)
        target.write_text(text)
        files = [target]
        changed_commands = []
        for command in argv_list:
            command = [str(target) if x == str(original) else x.replace(str(base.parent), str(output)) for x in command]
            changed_commands.append(command)
        argv_list = changed_commands
        # sygraph coverage includes are part of its original build directory,
        # not outputs; retain those frozen include files.
        for command in argv_list:
            for i, token in enumerate(command):
                if token == '-I'+str(output/'include'):
                    command[i] = '-I'+str(base.parent/'include')
    changes = []
    for file in files:
        before = file.read_text()
        after = '#include "'+str(header)+'"\n'+observe(before, args.unit)
        file.write_text(after)
        relative = file.relative_to(output)
        patch = output/(file.stem+'.clock.patch')
        patch.write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='before/'+str(relative), tofile='after/'+str(relative))))
        changes.append({'source': bind(file), 'before_sha256': hashlib.sha256(before.encode()).hexdigest(), 'patch': bind(patch)})
    affinity = {84,85} & os.sched_getaffinity(0)
    if affinity:
        os.sched_setaffinity(0, affinity)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1')
    env.pop('LD_PRELOAD', None)
    env.pop('LD_LIBRARY_PATH', None)
    if args.unit.startswith('sygraph_'):
        env['LD_LIBRARY_PATH'] = str(args.root/'envs/sycl/lib')
    steps = []
    code = 0
    for i, command in enumerate(argv_list):
        start = time.monotonic()
        with (output/('build_'+str(i)+'.log')).open('w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        steps.append({'argv': command, 'returncode': code, 'seconds': time.monotonic()-start})
        if code:
            break
    binaries = {name: str(Path(item['path'].replace(str(base.parent), str(output)))) for name,item in original_binaries.items()}
    receipt = {'status': 'PASS_BUILD_ONLY' if not code else 'FAIL_BUILD', 'gpu_executed': False,
        'unit': args.unit, 'builder': bind(__file__), 'base': bind(base), 'changes': changes,
        'header': bind(header), 'steps': steps, 'binaries': {name: bind(path) for name,path in binaries.items() if Path(path).exists()},
        'algorithm_changed': False, 'numeric_types_changed': False, 'compile_flags_changed': False,
        'observation': 'CLOCK_MONOTONIC timestamp after existing full owning host output/synchronization, before archive; original duration metrics preserved',
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'build_environment': {'CUDA_VISIBLE_DEVICES': '', 'LD_LIBRARY_PATH': env.get('LD_LIBRARY_PATH')}}
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'status': receipt['status'], 'receipt': str(output/'receipt.json')}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
