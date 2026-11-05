#!/usr/bin/env python3
"""Fetch or verify the exact author-source blobs used by the baseline adapters."""
from pathlib import Path
import argparse, hashlib, json, urllib.request

LOCK = Path(__file__).with_name('upstream_lock.json')


def blob_id(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def verify_group(root, group, fetch=False):
    root=Path(root); records=[]
    for item in json.loads(LOCK.read_text())['files']:
        if item['local_group']!=group:continue
        path=root/item['path']
        if not path.exists() and fetch:
            url='https://raw.githubusercontent.com/'+item['repo']+'/'+item['commit']+'/'+item['path']
            with urllib.request.urlopen(url,timeout=45) as response:data=response.read()
            if len(data)!=item['bytes'] or blob_id(data)!=item['git_blob_sha1']:
                raise ValueError('Downloaded source differs from fixed Git blob: '+item['path'])
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        data=path.read_bytes()
        if len(data)!=item['bytes'] or blob_id(data)!=item['git_blob_sha1']:
            raise ValueError('Source differs from fixed Git blob: '+str(path))
        records.append(dict(path=item['path'],git_blob_sha1=item['git_blob_sha1'],bytes=len(data)))
    if not records:raise ValueError('Unknown source group: '+group)
    return dict(group=group,files_checked=len(records),lock_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest(),
                status='PASS_EXACT_AUTHOR_BLOBS',commits=sorted({x['commit'] for x in json.loads(LOCK.read_text())['files'] if x['local_group']==group}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=['hytgraph','kgraph'],required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--fetch',action='store_true',help='Download missing fixed blobs; never overwrite existing differing files')
    a=p.parse_args();print(json.dumps(verify_group(a.source,a.group,a.fetch),indent=2))


if __name__=='__main__':main()
