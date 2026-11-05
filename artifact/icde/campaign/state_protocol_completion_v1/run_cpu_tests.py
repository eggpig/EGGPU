#!/usr/bin/env python3
"""Run self-contained CPU checks using an explicitly selected source checkout."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time


def bind(path):
    path=Path(path)
    return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve();root=Path(__file__).resolve().parent
    args.output.mkdir(parents=True,exist_ok=False)
    args.scratch.mkdir(parents=True,exist_ok=False)
    if not (source/'easygraph/gpu/_controller.py').is_file():
        raise ValueError('Provide the actual selected Easy-Graph source')
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        EGGPU_TEST_SOURCE=str(source))
    env['PYTHONPATH']=os.pathsep.join([str(root),str(source),env.get('PYTHONPATH','')])
    command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',
        '--basetemp='+str(args.scratch/'pytest'),str(root/'test_completion_cpu.py'),
        str(root/'test_original_contract_cpu.py')]
    def runner_bindings():
        return {str(p.relative_to(root)):bind(p) for p in sorted(root.rglob('*.py'))
                if '__pycache__' not in p.parts}
    before=runner_bindings()
    begin=time.monotonic()
    result=subprocess.run(command,env=env,capture_output=True,text=True)
    log=result.stdout+result.stderr
    log_path=args.output/'cpu_test.log';log_path.write_text(log)
    import numpy,scipy
    after=runner_bindings()
    unchanged=before==after
    receipt=dict(status='PASS_CPU_ONLY' if result.returncode==0 and unchanged else 'FAIL_CPU_ONLY',
        returncode=result.returncode,elapsed_seconds=time.monotonic()-begin,
        gpu_executed=False,formal_measurements=0,command=command,
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,
        source_root=str(source),source_files={str(p.relative_to(source)):bind(p)
            for p in sorted((source/'easygraph/gpu').glob('*.py'))},
        runner_files=after,runner_files_before=before,runner_unchanged_during_tests=unchanged,
        log=bind(log_path),scope='Independent CPU semantic tests; not native GPU qualification')
    (args.output/'cpu_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(log)
    return result.returncode if result.returncode else 0 if unchanged else 1


if __name__=='__main__':raise SystemExit(main())
