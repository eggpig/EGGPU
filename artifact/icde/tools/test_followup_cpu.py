"""Run the public followup suites in isolated processes without CUDA."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3]; C=ROOT/'artifact/icde/campaign'
GROUPS={
 'recovery':(C/'e4s_recoverable_outputs_v1',['test_archive_outputs.py','test_maintenance_export.py','test_backfill.py','test_queue_postprocessing.py','test_admission_integration.py']),
 'disposition':(C/'formal_admission_disposition_v1',['test_inspect_ledger.py']),
 'watcher':(C,['test_watch_external_formals_v4_cpu.py','test_watch_binding_v4_cpu.py']),
 'e6':(C/'e6_protocol_completion_v2',['test_protocol_cpu.py']),
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--groups',default=','.join(GROUPS));a=p.parse_args();work=a.work.resolve()
 if work.exists() or work.is_relative_to(ROOT):raise ValueError('Choose a new work directory outside checkout')
 work.mkdir(parents=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',EGGPU_TEST_TMP=str(work),TMPDIR=str(work));rows=[]
 for group in a.groups.split(','):
  cwd,names=GROUPS[group];command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--confcutdir='+str(cwd),'--basetemp='+str(work/(group+'_tmp')),*names]
  start=time.monotonic();log=work/(group+'.log')
  with log.open('x') as stream:r=subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
  row={'group':group,'returncode':r.returncode,'seconds':time.monotonic()-start,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()};rows.append(row);print(json.dumps(row),flush=True)
 report={'status':'PASS_PUBLIC_FOLLOWUP_CPU' if all(x['returncode']==0 for x in rows) else 'FAIL_PUBLIC_FOLLOWUP_CPU','gpu_executed':False,'groups':rows}
 (work/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');return int(report['status'].startswith('FAIL'))
if __name__=='__main__':raise SystemExit(main())
