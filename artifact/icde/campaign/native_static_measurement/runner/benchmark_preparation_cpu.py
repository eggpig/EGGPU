"""CPU-only diagnostic; never publishes an end-to-end or paper speedup."""
from pathlib import Path
import argparse,hashlib,json,os,resource,subprocess,sys,time,shutil

def binding(path):
 path=Path(path);h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--child',choices=['parent','candidate']);p.add_argument('--pair',type=int);p.add_argument('--full-audit',action='store_true');a=p.parse_args()
 if a.child:
  runner=Path(__file__).resolve().parents[2] / ('native_parent' if a.child=='parent' else 'native_load_p1');sys.path.insert(0,str(runner))
  import native_input as ni
  from sg_worker import load_input
  import numpy as np
  load_start=time.perf_counter();g=load_input(a.campaign/'inputs/canonical/US');prefault={role:float(g[role].sum(dtype=np.float64)) for role in ('offsets','indices','weights')};input_seconds=time.perf_counter()-load_start
  graph=ni.original_graph(g,'US');sign=ni.load_signer(Path(__file__).resolve().parents[5]/'Easy-Graph')
  output=a.output/f'pair{a.pair}_{a.child}';before=time.perf_counter();result=ni.native_partitions(graph,output,1,sign,weighted=True);elapsed=time.perf_counter()-before
  audit=ni.audit_partitions(graph,result['path']) if a.full_audit else None
  doc=json.loads(Path(result['path']).read_text());files={k:{key:v[key] for key in ['sha256','byte_size','dtype','count']} for k,v in doc['partitions'][0]['files'].items()}
  r={'variant':a.child,'pair':a.pair,'preparation_seconds':elapsed,'common_validation_prefault_seconds':input_seconds,'prefault_checksums':prefault,'manifest_sha256':result['sha256'],'payloads':files,'input_identity':g['input_sha256'],'source':binding(runner/'native_input.py'),'audit':audit,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'all_format_hashing_in_preparation_interval':True,'newly_written_owner_mapping_bytes':12*g['n'],'canonical_inputs_modified':False}
  (a.output/f'pair{a.pair}_{a.child}.json').write_text(json.dumps(r,indent=2)+'\n');shutil.rmtree(output);return
 a.output.mkdir(exist_ok=False);records=[];environment={'cpu_affinity':sorted(os.sched_getaffinity(0)),'loadavg_before':Path('/proc/loadavg').read_text().strip(),'background_load':'Concurrent state fullgrid CPU preparation and four-graph reference cohort; GPU qualification/diagnostic activity may be present; no formal timings scheduled.'}
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 for pair,order in enumerate([('parent','candidate'),('candidate','parent'),('parent','candidate')]):
  for variant in order:
   cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--campaign',str(a.campaign),'--output',str(a.output),'--child',variant,'--pair',str(pair)]
   if pair==0 and variant=='candidate':cmd.append('--full-audit')
   child=subprocess.run(cmd,env=env,capture_output=True,text=True)
   if child.returncode:raise RuntimeError(child.stdout+child.stderr)
   r=json.loads((a.output/f'pair{pair}_{variant}.json').read_text());records.append(r);print(json.dumps({k:r[k] for k in ['variant','pair','preparation_seconds','common_validation_prefault_seconds']}),flush=True)
  old,new=(json.loads((a.output/f'pair{pair}_{v}.json').read_text()) for v in ['parent','candidate']);assert old['manifest_sha256']==new['manifest_sha256'] and old['payloads']==new['payloads'] and old['prefault_checksums']==new['prefault_checksums'] and old['input_identity']==new['input_identity']
 environment['loadavg_after']=Path('/proc/loadavg').read_text().strip()
 report={'schema':'eggpu.native.load.preparation.diagnostic.v1','status':'PASS_CPU_PREPARATION_BYTE_PARITY','gpu_executed':False,'dataset':'US','weighted':True,'count':1,'pairs':3,'records':records,'environment':environment,'not_paper_performance':True,'scope':'Exact byte-equivalent P1 format preparation including full output SHA; common validated/prefaulted host CSR. Background CPU/I/O load means numbers are diagnostic only. Full task/GPU qualification remains required.','script':binding(__file__)}
 (a.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'receipt':binding(a.output/'receipt.json'),'status':report['status']}),flush=True)
if __name__=='__main__':main()
