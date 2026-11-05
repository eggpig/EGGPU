"""Bounded HTTP-range downloader with verified offsets and durable progress.

Four connections per public file by default; eight must be explicit. A prior sequential .partial is reused as a
verified prefix; after preallocation, progress comes only from the sidecar.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, os, re, threading, time, urllib.request

class RangeNotSupported(RuntimeError):
    pass


def download_file(url, dest, timeout=45, retries=4, *, connections=4):
    if type(connections) is not int or connections not in (4, 8):
        raise ValueError('connections must be explicitly 4 or 8')
    dest=Path(dest)
    if dest.exists(): return {'status':'EXISTING_REQUIRES_VALIDATION','url':url,'bytes':dest.stat().st_size}
    headers={'User-Agent':'EGGPU-public-artifact/2','Accept-Encoding':'identity','Range':'bytes=0-0'}
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=timeout) as probe:
        match=re.fullmatch(r'bytes 0-0/(\d+)',probe.headers.get('Content-Range',''))
        if probe.status!=206 or not match:
            raise RangeNotSupported('Source does not support checked HTTP ranges; use sequential downloader')
        total=int(match[1]);identity={'etag':probe.headers.get('ETag'),'last_modified':probe.headers.get('Last-Modified')}
        if len(probe.read(2))!=1:raise ValueError('Range probe length differs')
    partial=dest.with_name(dest.name+'.partial');progress=dest.with_name(dest.name+'.ranges.json')
    if progress.exists():
        state=json.loads(progress.read_text())
        if state['url']!=url or state['total_bytes']!=total or state['identity']!=identity:
            raise ValueError('Remote object changed during resume')
        if not partial.exists() or partial.stat().st_size!=total:raise ValueError('Range payload/progress mismatch')
    else:
        prefix=partial.stat().st_size if partial.exists() else 0
        if prefix>total:raise ValueError('Sequential prefix exceeds official object size')
        block=2<<20
        tasks=[[start,min(total,start+block)] for start in range(prefix,total,block)]
        state={'schema':'eggpu.checked_ranges.v1','url':url,'identity':identity,'total_bytes':total,
               'initial_prefix_bytes':prefix,'ranges':tasks,'completed':[]}
        partial.parent.mkdir(parents=True,exist_ok=True)
        with partial.open('ab') as f:f.truncate(total)
    lock=threading.Lock()
    def save():
        state['downloaded_bytes']=state['initial_prefix_bytes']+sum(state['ranges'][i][1]-state['ranges'][i][0] for i in state['completed'])
        tmp=progress.with_name(progress.name+'.partial')
        with tmp.open('w') as f:json.dump(state,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        tmp.replace(progress)
    save();fd=os.open(partial,os.O_WRONLY)
    def fetch(index):
        start,end=state['ranges'][index]
        for attempt in range(retries+1):
            try:
                h=dict(headers,Range=f'bytes={start}-{end-1}')
                with urllib.request.urlopen(urllib.request.Request(url,headers=h),timeout=timeout) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end-1}/{total}':
                        raise ValueError('Response does not match requested byte range')
                    if identity['etag'] and response.headers.get('ETag')!=identity['etag']:
                        raise ValueError('Remote ETag changed')
                    position=start
                    while position<end:
                        data=response.read(min(64<<10,end-position))
                        if not data:raise OSError('Incomplete HTTP range')
                        written=0
                        while written<len(data):
                            count=os.pwrite(fd,data[written:],position+written)
                            if count<=0:raise OSError('Short local write')
                            written+=count
                        position+=len(data)
                    if response.read(1):raise ValueError('Unexpected trailing range bytes')
                with lock:
                    os.fsync(fd);state['completed'].append(index);save()
                return
            except (OSError,TimeoutError):
                if attempt==retries:raise
                time.sleep(min(2**attempt,8))
    try:
        done=set(state['completed'])
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures=[pool.submit(fetch,i) for i in range(len(state['ranges'])) if i not in done]
            for future in as_completed(futures):future.result()
    finally:os.close(fd)
    if state['downloaded_bytes']!=total:raise ValueError('Not all ranges completed')
    partial.replace(dest);state['status']='PASS_COMPLETE_RANGES';save()
    return {'status':'DOWNLOADED_CHECKED_RANGES_REQUIRES_VALIDATION','url':url,'bytes':total,
            'resumed_from':state['initial_prefix_bytes'],'connections':connections,'progress':str(progress)}
