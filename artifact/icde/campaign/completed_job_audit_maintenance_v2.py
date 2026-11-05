"""After-job evidence classification plus lossless output deduplication; CPU only."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys


def bind(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())


def classify(messages,success):
    if success:return 'PASS'
    s='\n'.join(messages).lower()
    if any(x in s for x in ['cudaerrormemoryallocation','cuda_error_out_of_memory','out_of_memory: cuda','cuda out of memory','cupy.cuda.memory.outofmemoryerror','cuda memory allocation error']):return 'CUDA_OOM'
    if any(x in s for x in ['no space left on device','insufficient disk space','disk quota exceeded']):return 'DISK_CAPACITY'
    if 'index_unsupported' in s:return 'ADAPTER_INDEX_LIMIT'
    if any(x in s for x in ['int64 not allowed','unsupported_type','unsupported type']):return 'AUTHOR_TYPE_CAPABILITY_FAILURE'
    if any(x in s for x in ['host/cgroup headroom insufficient','host memory','memoryerror','cannot allocate memory']):return 'HOST_MEMORY_CAPACITY'
    if any(x in s for x in ['timeout','timed_out','timed out']):return 'TIMEOUT'
    if 'execution failed' in s or 'execution_error' in s:return 'BACKEND_EXECUTION_FAILURE'
    if any(x in s for x in ['quality failed','quality check','output quality']):return 'OUTPUT_QUALITY_FAILURE'
    return 'OTHER_FAILURE_REQUIRES_REVIEW'


def inspect(result):
    result=Path(result);root=result.parent;files=[];messages=[];peaks=[];stages=[];status=None
    candidates={result}
    candidates.update(root.rglob('result.json'));candidates.update(root.rglob('stage.json'))
    candidates.update(root.rglob('*observer*.json'));candidates.update(root.rglob('watchdog.json'))
    def walk(x,path,key=''):
        if isinstance(x,dict):
            for k,v in x.items():
                label=(key+'.'+k).strip('.')
                if k in ('error','execution_error','close_error') and v:messages.append(label+': '+str(v))
                if ('peak' in k.lower() and any(t in label.lower() for t in ['vram','gpu','device'])
                        and not any(t in label.lower() for t in ['host','rss'])):
                    peaks.append(dict(file=str(path),field=label,value=v))
                walk(v,path,label)
        elif isinstance(x,list):
            for i,v in enumerate(x):walk(v,path,key+f'[{i}]')
    for path in sorted(candidates):
        if not path.is_file():continue
        files.append(bind(path))
        try:data=json.loads(path.read_text())
        except (ValueError,UnicodeError):continue
        if path==result:status=data.get('status')
        if path.name=='stage.json':stages.append(dict(file=str(path),data=data))
        walk(data,path)
    for path in sorted(root.rglob('*.log')):
        files.append(bind(path))
        with path.open(errors='replace') as f:
            for line in f:
                if any(x in line.lower() for x in ['error','out of memory','bad_alloc','no space left','index_unsupported','timed out','traceback']):
                    messages.append(str(path)+': '+line.rstrip()[:4000])
    success=bool(status and status.startswith('PASS'))
    return dict(schema='eggpu.complete.job.capacity.audit.v1',status='PASS_AUDIT_RECORDED',
        original_result_status=status,classification=classify(messages,success),
        source_evidence=files,messages=messages,stages=stages,
        actual_peak_vram_fields=peaks,actual_peak_vram_available=bool(peaks),
        peak_vram_policy='Only existing recorded peak metrics; failed requested allocation size and pre-run free memory are not peak usage.',
        presentation_policy='CUDA OOM is capacity evidence with no numeric runtime or infinite speedup. Adapter, disk and host capacity failures are separate.',
        missing_result=not result.is_file())


def main():
    p=argparse.ArgumentParser()
    for n in ['registry','output','gpu-lock','job-result','dedup-wrapper']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();audit=inspect(a.job_result)
    # The deduplicator owns output/receipt.json. Preserve a separate immutable audit sibling.
    audit_path=a.output.parent/(a.output.name+'.capacity_audit.json')
    audit_path.parent.mkdir(parents=True,exist_ok=True)
    with audit_path.open('x') as f:f.write(json.dumps(audit,indent=2)+'\n')
    subprocess.run([sys.executable,'-B',str(a.dedup_wrapper),'--registry',str(a.registry),
                    '--output',str(a.output),'--gpu-lock',str(a.gpu_lock)],check=True)


if __name__=='__main__':main()
