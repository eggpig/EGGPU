#!/usr/bin/env python3
"""CPU-only Gunrock CC observation patch: complete first-request wall time.

Reuses the exact prior flags and unchanged dependency objects. No algorithm,
partition, convergence, memory layout or execution policy is altered.
"""
from pathlib import Path
import argparse, difflib, hashlib, json, os, subprocess, time


def binding(p):
    p=Path(p);return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old=json.loads(a.base.read_text());out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if old['status']!='PASS' or binding(old['binary'])['sha256']!=old['binary_sha256']:raise ValueError('Frozen author build changed')
    command=old['steps'][0]['argv'].copy();source=Path(command[command.index('-dc')+1]);before=source.read_text()
    text=before.replace('using namespace gunrock;','static std::chrono::steady_clock::time_point campaign_first_begin;\nusing namespace gunrock;',1)
    text=text.replace('int main(int argc, char **argv) {','int main(int argc, char **argv) {\n  campaign_first_begin = std::chrono::steady_clock::now();',1)
    needle='    fflush(stdout);\n    total_elapsed += single_elapsed;'
    addition='''    printf("CAMPAIGN_CC_FIRST_JSON {\\"round\\":%lld,\\"first_request_seconds\\":%.17g}\\n",
           (long long)iter, std::chrono::duration<double>(complete_end-campaign_first_begin).count());
'''
    if text.count(needle)!=1:raise ValueError('Missing exact CC timer context')
    text=text.replace(needle,addition+needle)
    target=out/'gunrock_cc_timer.cu';target.write_text(text)
    patch=out/'timer.patch';patch.write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='before/adapter.cu',tofile='after/adapter.cu')))
    original_object=command[command.index('-o')+1];obj=out/'adapter.o';command[command.index('-dc')+1]=str(target);command[command.index('-o')+1]=str(obj)
    link=old['steps'][-1]['argv'].copy();link=[str(obj) if x==original_object else x for x in link];binary=out/'gunrock_wcc';link[link.index('-o')+1]=str(binary)
    unchanged=[binding(x) for x in link if x.endswith('.o') and x!=str(obj)]
    affinity={84,85}&os.sched_getaffinity(0)
    if affinity:os.sched_setaffinity(0,affinity)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1');env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    steps=[];code=0
    for i,argv in enumerate((command,link)):
        begin=time.monotonic()
        with (out/f'build_{i}.log').open('w') as log:code=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        steps.append({'argv':argv,'returncode':code,'seconds':time.monotonic()-begin})
        if code:break
    receipt={'status':'PASS_BUILD_ONLY' if code==0 else 'FAIL_BUILD','gpu_executed':False,'base':binding(a.base),
             'before':binding(source),'after':binding(target),'patch':binding(patch),'unchanged_dependency_objects':unchanged,
             'binary':str(binary),'binary_sha256':binding(binary)['sha256'] if binary.exists() else None,'steps':steps,
             'algorithm_changed':False,'timing_changed':'Adds first request from main entry through first complete owning output and synchronization; existing complete query timer unchanged'}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'receipt':str(out/'receipt.json')}));return code


if __name__=='__main__':raise SystemExit(main())
