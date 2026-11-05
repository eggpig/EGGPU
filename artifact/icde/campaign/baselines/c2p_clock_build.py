#!/usr/bin/env python3
"""CPU-only complete-host observation build of existing C2graph-P, not PM."""
from pathlib import Path
import argparse,difflib,hashlib,json,os,shutil,subprocess,time


def bind(p):
    p=Path(p);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.root/'build/c2graph/receipt.json';raw=json.loads(base.read_text())
    if raw['status']!='PASS_BUILD_ONLY' or bind(raw['binary'])['sha256']!=raw['binary_sha256'] or bind(raw['compressor'])['sha256']!=raw['compressor_sha256']:
        raise ValueError('Existing C2graph-P build changed')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    command=raw['commands'][1].copy();original=next(Path(x) for x in command if x.endswith('.cu'))
    dependency=next(Path(x[2:]) for x in command if x.startswith('-I'))
    shutil.copytree(dependency,out/'source')
    trace=original.parent/'trace_input_all.hpp';shutil.copy2(trace,out/'source'/trace.name)
    provenance=json.loads((out/'source/provenance.json').read_text())
    if provenance['upstream_commit']!='97af1ebd935a03f5adc180935644e786a624bdbe':raise ValueError('Wrong author revision')
    header=out/'source/sg_complete_clock.h';shutil.copy2(Path(__file__).with_name('sg_complete_clock.h'),header)
    before=original.read_text();old='cumulative+=sec(a,e);dumpOutput(argv[2],c.sequence,v);'
    if before.count(old)!=1:raise ValueError('Missing existing complete FP64 host-output point')
    after='#include "sg_complete_clock.h"\n'+before.replace(old,'cumulative+=sec(a,e);eggpu_sg_complete_clock(c.sequence,v.size());dumpOutput(argv[2],c.sequence,v);')
    adapter=out/'source/c2graph_p_clock.cu';adapter.write_text(after)
    patch=out/'clock.patch';patch.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(original),tofile=str(adapter))))
    binary=out/'native';command=[str(adapter) if x==str(original) else '-I'+str(out/'source') if x=='-I'+str(dependency) else x for x in command]
    command[command.index('-o')+1]=str(binary)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1');env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    os.sched_setaffinity(0,{84,85});started=time.monotonic()
    with (out/'build.log').open('w') as log:code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    record={'status':'PASS_BUILD_ONLY' if not code else 'FAIL_BUILD','gpu_executed':False,'variant':'c2graph_p',
        'source_commit':provenance['upstream_commit'],'base':bind(base),'builder':bind(__file__),
        'binary':bind(binary) if binary.exists() else None,'compressor':bind(raw['compressor']),
        'command':command,'returncode':code,'seconds':time.monotonic()-started,'patch':bind(patch),
        'source_files':[bind(p) for p in sorted((out/'source').rglob('*')) if p.is_file()],
        'header':bind(header),'original_adapter':bind(original),'algorithm_changed':False,
        'prior_correctness_patches':provenance['repairs'],
        'scope':'Existing single-source P executor, compressed input retained across three independent solves; adds only complete-host monotonic marker. PM batch timings are not used.'}
    (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'status':record['status'],'receipt':str(out/'receipt.json')}));return code


if __name__=='__main__':raise SystemExit(main())
