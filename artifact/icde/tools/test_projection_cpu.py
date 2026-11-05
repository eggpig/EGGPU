#!/usr/bin/env python3
"""Run independent public-source and runner CPU suites; no GPU execution."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3]
C=ROOT/'artifact/icde/campaign'
GROUPS={
'native_load_p1':(C/'native_load_p1',['test_static_runner_cpu.py','test_native_profile_cpu.py','test_metric_json_cpu.py','test_load_preparation_cpu.py','test_public_cpu_mapping.py']),
'sg_current':(C/'sg_current',['test_cugraph_i64_cpu.py','test_csr_runner_contract.py','test_pr_global_l1.py','test_sg_campaign.py','test_sequence16_campaign.py']),
'state_planner':(C/'state_final_plan',['test_state_plan.py']),
'state_hotspot':(C/'state_partition_hotspot',['test_hotspot_positions.py','test_hotspot_integration.py']),
'state_grid':(C/'state_grid_prepare_v3',['test_state_grid_cpu.py']),
'pr_control':(C/'e3_unweighted_control',['test_pr_control.py']),
'gum_fp32':(C/'gum_fp32_author',['test_fp32_contract.py','test_gum_clock.py']),
'e6_reference':(C/'e6_memory',['test_reference_binding.py']),
'queue_v7':(C,['test_campaign_queue_v7_cpu.py']),
'queue_v8':(C,['test_campaign_queue_v8_cpu.py']),
'incremental_maintenance':(C,['test_incremental_completed_output_cpu.py']),
'public_projection':(ROOT/'artifact/icde/tools',['test_d84_projection.py']),
'presentation':(ROOT/'artifact/icde/presentation',['test_e1_summary.py']),
'e4s':(C/'e4s',['test_e4s_cpu.py']),
'state_worker':(C,['test_state_v4_cpu.py']),
'weight_admission':(ROOT/'artifact/icde/tests',['test_weight_admission_cpu.py']),
'native_diagnostic':(C/'native_diagnostic',['test_diagnostic_scope_cpu.py']),
'paper_e2_e6':(ROOT/'artifact/icde/paper_results',['test_paper_results.py','test_capacity.py','test_public_paper_gates.py','test_sequence.py']),
}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--groups',default='source,'+','.join(n for n in GROUPS if n!='e4s'));a=p.parse_args()
 if 'e4s' in a.groups.split(',') and not all(os.environ.get(k) for k in ['E4S_CPU_AUTHOR_BUILD','E4S_CPU_REFERENCE']):raise ValueError('E4S CPU author tests require explicit E4S_CPU_AUTHOR_BUILD and E4S_CPU_REFERENCE')
 a.work=a.work.resolve();a.work.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',EGGPU_TEST_SOURCE=str(ROOT/'Easy-Graph'),EGGPU_TEST_FREEZE=str(ROOT/'artifact/icde/provenance/runtime_source_freeze.json'),EGGPU_TEST_STATE_CODE=str(C),EGGPU_TEST_SCRATCH=str(a.work),EGGPU_RESUME_TEST_SCRATCH=str(a.work/'resume'),EGGPU_INCREMENTAL_TEST_SCRATCH=str(a.work/'incremental'))
 rows=[]
 expanded=[]
 for group in a.groups.split(','):
  if group=='source':expanded.append((group,None,None))
  else:
   folder,names=GROUPS[group]
   expanded.extend((group+'.'+Path(test).stem,folder,[test]) for test in names)
 for name,folder,tests in expanded:
  if name=='source':command=[sys.executable,'-B',str(ROOT/'artifact/icde/tools/test_cpu.py'),'--work',str(a.work/'source')];cwd=ROOT
  else:
   cwd=folder;command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--confcutdir='+str(cwd),'--basetemp='+str(a.work/(name+'_temporary')),*tests]
  started=time.time();log=a.work/(name+'.log')
  with log.open('x') as f:r=subprocess.run(command,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  row=dict(group=name,returncode=r.returncode,seconds=time.time()-started,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest());rows.append(row)
  print(json.dumps(row),flush=True)
  (a.work/'receipt.json').write_text(json.dumps(dict(status='RUNNING',gpu_executed=False,groups=rows),indent=2)+'\n')
 status='PASS_PUBLIC_PROJECTED_CPU' if all(x['returncode']==0 for x in rows) else 'FAIL_PUBLIC_PROJECTED_CPU'
 (a.work/'receipt.json').write_text(json.dumps(dict(status=status,gpu_executed=False,groups=rows),indent=2)+'\n');return 0 if status.startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
