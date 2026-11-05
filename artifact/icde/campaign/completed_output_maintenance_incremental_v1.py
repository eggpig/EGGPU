"""Drop-in CPU maintenance wrapper with an explicit immutable-output index.

Its argv matches completed_output_maintenance.py, preserving the existing
capacity-audit wrapper. Use only as a root-controller maintenance child.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    for n in ('registry','output','gpu-lock'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=json.loads(a.registry.read_text())
    if r.get('schema')!='completed.output.incremental.registry.v1':
        raise ValueError('Explicit incremental maintenance registry required')
    code=Path(r['deduplicator']['path'])
    if hashlib.sha256(code.read_bytes()).hexdigest()!=r['deduplicator']['sha256']:
        raise ValueError('Incremental helper changed')
    if str(a.gpu_lock)!=r['gpu_lock']:
        raise ValueError('Global GPU lock changed')
    if not any(Path(x).is_file() for x in r['queue_states']):
        raise ValueError('No explicitly registered completed queue state available')
    subprocess.run([sys.executable,'-B',str(code),'--apply','--registry',str(a.registry),
                    '--index',r['index'],'--output',str(a.output),
                    '--controller-lock',r['controller_lock'],'--controller-lock-held',
                    '--gpu-lock',str(a.gpu_lock)],check=True)


if __name__=='__main__':main()
