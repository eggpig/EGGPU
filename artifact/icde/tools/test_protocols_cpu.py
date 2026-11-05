"""Public CPU recheck of protocol/admission candidates. No GPU initialization."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'artifact/icde';C=P/'campaign'
GROUPS={
 'incremental':(C/'formal_incremental_admission_v1',['test_incremental_cpu.py']),
 'e4s_incremental':(C/'formal_incremental_e4s_v1',['test_e4s_incremental_cpu.py']),
 'state_guard':(C/'state_completion_queue_guard_v1',['test_guard_cpu.py']),
 'arena_diagnostic':(C/'arena_diagnostic_v1',['test_capture_cpu.py']),
 'paper':(P/'paper_results_current',['test_completion.py']),
 'wcc_v3':(P/'experimental/native_wcc_wide_v3',['test_profile_compatibility_cpu.py']),
 'wcc_plan':(P/'experimental/native_wcc_wide_v3',['test_portable_plan_cpu.py']),
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True)
 p.add_argument('--groups',default=','.join(GROUPS));p.add_argument('--dependency-path',type=Path,action='append',default=[])
 a=p.parse_args();work=a.work.resolve()
 if work.exists() or work.is_relative_to(ROOT):raise ValueError('Choose a new work directory outside checkout')
 work.mkdir(parents=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_MAX_THREADS='2',MPLCONFIGDIR=str(work/'mpl'),EGGPU_TEST_TMP=str(work),TMPDIR=str(work))
 if a.dependency_path:env['PYTHONPATH']=os.pathsep.join(str(p.resolve()) for p in a.dependency_path)
 rows=[]
 for group in a.groups.split(','):
  if group not in GROUPS:raise ValueError('Unknown group')
  cwd,names=GROUPS[group]
  if group=='wcc_v3':
   target=work/'wcc_v2'
   command=[sys.executable,'-B',str(P/'experimental/native_wcc_wide_v2/materialize.py'),'--output',str(target)]
   subprocess.run(command,env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
   env['EGGPU_WCC_V2_SOURCE']=str(target)
  command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--confcutdir='+str(cwd),'--basetemp='+str(work/(group+'_tmp')),'--junitxml='+str(work/(group+'.xml')),*names]
  log=work/(group+'.log')
  with log.open('x') as stream:done=subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
  suites=ET.parse(work/(group+'.xml')).getroot().findall('testsuite') if (work/(group+'.xml')).exists() else []
  counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
  row=dict(group=group,returncode=done.returncode,counts=counts,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest());rows.append(row);print(json.dumps(row),flush=True)
 report=dict(schema='eggpu.public.protocols-cpu-validation/v1',status='PASS_PUBLIC_CPU_ONLY' if all(x['returncode']==0 for x in rows) else 'FAIL_PUBLIC_CPU',groups=rows,gpu_executed=False,performance_evidence=False,synthetic_figures_published=False)
 (work/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');return int(any(x['returncode'] for x in rows))
if __name__=='__main__':raise SystemExit(main())
