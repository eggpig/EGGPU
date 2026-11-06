"""Compile only the CPU text serializer into a fresh explicit build directory."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent


def bind(path):
    path=Path(path).resolve();data=path.read_bytes()
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def build(output,compiler='/usr/bin/g++'):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    library=output/'libcsr_text.so'
    command=[compiler,'-O3','-std=c++17','-shared','-fPIC','-fno-gnu-unique',str(HERE/'csr_text.cpp'),'-o',str(library)]
    with (output/'build.log').open('xb') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    record=dict(status='PASS_CPU_BUILD_ONLY',source=bind(HERE/'csr_text.cpp'),library=bind(library),
        compiler=bind(compiler),command=command,gpu_executed=False)
    (output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--compiler',default='/usr/bin/g++')
    a=p.parse_args();print(json.dumps(build(a.output,a.compiler)))
