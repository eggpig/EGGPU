"""Run isolated CPU synthetic tests and record their exact source/dependency identity."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scratch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dependency-path',type=Path,action='append',default=[])
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        # Keep the environment's numpy instead of accidentally replacing it
        # with a second numpy bundled alongside a plotting-only dependency.
        import numpy
        sys.path[1:1]=[str(path.resolve()) for path in a.dependency_path]
        import scipy,matplotlib,pytest
        (a.output/'dependencies.json').write_text(json.dumps(dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,pytest=pytest.__version__),indent=2)+'\n')
        return pytest.main(['-q',str(HERE/'test_completion.py'),'--rootdir='+str(HERE),
            '--basetemp='+str(a.scratch/'pytest'),'--junitxml='+str(a.output/'tests.xml'),'-p','no:cacheprovider','--tb=short'])
    if a.output.exists() or a.scratch.exists():raise FileExistsError('Use new explicit output and scratch directories; never overwrite evidence')
    a.output.mkdir(parents=True);a.scratch.mkdir(parents=True)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_MAX_THREADS='2',
                     PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(a.scratch/'mpl'))
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--scratch',str(a.scratch),'--output',str(a.output)]
    for path in a.dependency_path:command+=['--dependency-path',str(path)]
    with (a.output/'stdout.log').open('w') as log:
        done=subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=False)
    suites=ET.parse(a.output/'tests.xml').getroot().findall('testsuite') if (a.output/'tests.xml').exists() else []
    counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.md') and p.is_file() and p.name!='freeze.json'}
    receipt=dict(schema='eggpu.paper.protocol.cpu.validation.v1',status='PASS_CPU_SYNTHETIC_ONLY' if done.returncode==0 else 'FAIL',
                 exit_code=done.returncode,counts=counts,source_files=files,gpu_executed=False,performance_evidence=False,
                 formal_figures_generated=False,synthetic_figures_only=True,dependency_versions=json.loads((a.output/'dependencies.json').read_text()) if (a.output/'dependencies.json').exists() else None)
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status=receipt['status'],counts=counts)))
    return done.returncode


if __name__=='__main__':raise SystemExit(main())
