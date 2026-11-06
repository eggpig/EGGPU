"""Build a separate Grapin binary with one additional continuous timing field."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess

HERE=Path(__file__).resolve().parent
ORIGINAL='ede123860f5090090c89292fac1d3b0d957234c5c488f655e9507ebf775c3856'
RELATIVE='source/samples/hybrid_sssp/hybrid_sssp.cu'


def bind(path):
    path=Path(path).resolve();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8<<20),b''):digest.update(data)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def build_commands(original,original_receipt,output):
    return [[str(x).replace(str(Path(original_receipt).parent),str(output)) for x in step['command']] for step in original['steps']]


def observation_patch(text):
    changes=[(' double calls=0;',' double calls=0;\n EggpuClock::time_point eggpu_base_complete;'),
      ('  if(qualification)engine.EggpuCheckPartialCacheV2(snapshot);','  if(qualification)engine.EggpuCheckPartialCacheV2(snapshot);\n  if(snapshot==0)eggpu_base_complete=EggpuClock::now();'),
      ('  <<",\\\"active_workflow_wall_seconds\\\":"<<eggpu_seconds(prepared,service_end)',
       '  <<",\\\"active_workflow_wall_seconds\\\":"<<eggpu_seconds(prepared,service_end)\n   <<",\\\"post_base_update_workflow_seconds\\\":"<<eggpu_seconds(eggpu_base_complete,service_end)')]
    original=text
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Frozen observation anchor differs')
        text=text.replace(before,after)
    restored=text
    for before,after in reversed(changes):restored=restored.replace(after,before)
    if restored!=original:raise ValueError('Unexpected source edit')
    return text


def build(original_receipt,output,cpu_set):
    original_receipt=Path(original_receipt).resolve();output=Path(output).resolve()
    original=json.loads(original_receipt.read_text());source=Path(original['source'])
    if original['status']!='PASS_BUILD_ONLY' or original['unit']!='update':raise ValueError('Need frozen corrected update unit')
    if bind(source/RELATIVE)['sha256']!=ORIGINAL:raise ValueError('Original incremental author source changed')
    for relative,expected in original['source_file_sha256'].items():
        if bind(source/relative)['sha256']!=expected:raise ValueError('Original author source tree changed')
    output.mkdir(parents=True,exist_ok=False)
    selected={int(x) for x in cpu_set.split(',')}
    if not selected<=os.sched_getaffinity(0):raise ValueError('Unavailable build CPUs')
    os.sched_setaffinity(0,selected)
    shadow=output/'source';shutil.copytree(source,shadow,copy_function=shutil.copy2,symlinks=True)
    target=shadow/RELATIVE;modified=observation_patch(target.read_text());target.unlink();target.write_text(modified)
    record=dict(status='BUILDING_OBSERVATION_ONLY',original_build=bind(original_receipt),original_binary=original['binary'],
        original_source=bind(source/RELATIVE),observed_source=bind(target),builder=bind(__file__),
        source_only_additions='One host time_point, one timestamp after completed base request, one JSON elapsed field; native kernels, input parsing, update API, output ownership and required parent repair unchanged.',
        algorithms_changed=False,numerical_types_changed=False,gpu_executed=False,steps=[])
    env=dict(os.environ);env.update(original['build_environment']);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for key in original.get('cleared_environment',[]):env.pop(key,None)
    env['CUDA_VISIBLE_DEVICES']=''
    for index,command in enumerate(build_commands(original,original_receipt,output)):
        with (output/f'build_{index}.log').open('xb') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        record['steps'].append(dict(argv=command,returncode=result.returncode))
        if result.returncode:
            record['status']='FAILED_OBSERVATION_ONLY_CPU_BUILD';(output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
            raise RuntimeError('Observation-only build failed')
    binary=output/'build/grapin_sssp'
    record.update(status='PASS_OBSERVATION_ONLY_CPU_BUILD',binary=bind(binary))
    (output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-set',default='16,17,18,19,20,21,22,23,64,65,66,67,68,69,70,71');a=p.parse_args()
    print(json.dumps(build(a.original_receipt,a.output,a.cpu_set)))
