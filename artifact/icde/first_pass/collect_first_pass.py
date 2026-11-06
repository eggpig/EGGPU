"""Read JSON receipts and failed-job log tails; no GPU, array rehash, replay, or qualification."""
from pathlib import Path
import argparse,datetime,hashlib,json

PLANS=(
 'unified_release_d84_v2/sg/plan.json','unified_release_d84_v2/native/plan.json',
 'unified_release_d84_v2/sg_formal/pending.json','unified_release_d84_v2/native_formal/pending.json',
 'final_external_plans_ownership_v2/author_formal_plan.json',
 'final_external_plans_ownership_v2/c2p_formal_plan.json',
 'final_external_plans_ownership_v2/gum_formal_plan.json',
 'final_external_plans_ownership_v2/cugraph_mg_formal_plan.json',
 'maintenance_incremental_v1/plans/author/plan.json',
 'maintenance_incremental_v1/plans/cugraph_mg/plan.json',
 'maintenance_incremental_v1/plans/c2p/plan.json',
 'gum_fp32_author_v1/qualification_guarded.json',
 'unified_release_d84_v2/sequence/plan.json',
 'cugraph_mg_tw_narrow_guarded_v2/plan.json','cugraph_mg_wide_qualification_v5/plan.json',
)
QUEUES=(
 'first_pass_20261106/core/queue/state.json','first_pass_20261106/first_page/queue/state.json',
 'first_pass_20261106/successors/missing_queue/state.json','first_pass_20261106/successors/scaling_queue/state.json',
 'first_pass_20261106/successors/e4s_queue/state.json','first_pass_20261106/successors/e6_queue/state.json',
 'first_pass_20261106/author_parallel/lane0_queue/state.json','first_pass_20261106/author_parallel/lane1_queue/state.json','first_pass_20261106/author_parallel/serial_tail_queue/state.json',
 'unified_release_d84_v2/sg/queue/state.json','unified_release_d84_v2/native/queue/state.json',
 'unified_release_d84_v2/preflight/queue/state.json',
 'maintenance_incremental_v1/plans/author/queue/state.json',
 'maintenance_incremental_v1/plans/cugraph_mg/queue/state.json',
 'maintenance_incremental_v1/plans/c2p/queue/state.json',
 'two_repeat_queues_v1/cugraph_mg/queue/state.json','two_repeat_queues_v1/c2p/queue/state.json',
 'gum_fp32_author_v1/queue/state.json','unified_release_d84_v2/sequence/queue/state.json',
 'state_completion_qualification_guarded_v1/queue/state.json',
 'state_unified_full_v1/hotspot/qualification_queue/state.json',
 'e6_protocol_completion_v2/deployed/qualification_queue/state.json',
 'unified_release_d84_v2/e4s/queue/state.json',
 'capacity_diagnostics_d84_v2/gum_standard_qualification/queue/state.json',
 'cugraph_mg_tw_narrow_guarded_v2/queue/state.json','cugraph_mg_wide_qualification_v5/queue/state.json',
 'e1_missing_first_pass_queue_20261106_v1/queue/state.json',
)

def collect(campaign,extra_queue_states=(),extra_plans=(),source_freeze='candidates/unified_release_20261106/freeze.json',native_protocol='native-host-csr-static-all4-public-return-v2'):
 root=Path(campaign).resolve();objects={};missing=[];records=[];queues=[];plans=[];seen=set();failed_attempt_evidence=[]
 def read(path):
  p=Path(path)
  if not p.is_absolute():p=root/p
  key=str(p)
  if key in objects:return objects[key]['value']
  if not p.is_file():
   if key not in missing:missing.append(key)
   return None
  if p.suffix!='.json' or p.stat().st_size>32*1024*1024:return None
  b=p.read_bytes()
  try:d=json.loads(b)
  except (ValueError,UnicodeDecodeError):return None
  objects[key]={'binding':{'path':key,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()},'value':d}
  return d
 def refs(value):
  if isinstance(value,dict):
   if isinstance(value.get('path'),str) and 'sha256' in value:
    p=Path(value['path'])
    if p.suffix=='.json' and p.is_relative_to(root):
     existed=str(p) in objects;d=read(p)
     if d is not None and not existed:refs(d)
   else:
    for k,v in value.items():
     if k not in ('tiny_proof','target_proof','qualification','release_sources','gpu_kernel_sources','source_files','files'):refs(v)
  elif isinstance(value,list):
   for x in value:refs(x)
 def record(path,origin,job=None,completed=None):
  if not path:return
  key=str(Path(path))
  if key in seen:
   if completed is not None:
    old=next(x for x in records if x['path']==key)
    old.update(completed=completed,job_id=job.get('id') if job else None,worker_command=job.get('command') if job else None,completed_origin=origin)
   return
  d=read(key)
  if d is None:return
  if d.get('phase')=='tiny' or str(d.get('dataset','')).startswith('TINY'):return
  seen.add(key);refs(d);records.append({'path':key,'origin':origin,'job_id':job.get('id') if job else None,'worker_command':job.get('command') if job else None,'completed':completed})
 for rel in dict.fromkeys((*PLANS,*extra_plans)):
  d=read(rel)
  if d is None:continue
  plans.append(objects[str(root/rel)]['binding'])
  for job in d.get('jobs',[]):
   cmd=job.get('command',[])
   if '--target-proof' in cmd:record(cmd[cmd.index('--target-proof')+1],rel,job)
   if job.get('phase') in ('target','formal'):record(job.get('receipt'),rel,job)
  objects.pop(str(root/rel),None)
 author_queues=sorted(str(p.relative_to(root)) for p in (root/'first_pass_20261106/author_parallel').glob('*/queue/state.json'))
 for rel in dict.fromkeys((*QUEUES,*author_queues,*(str(Path(x).relative_to(root)) if Path(x).is_absolute() else str(x) for x in extra_queue_states))):
  d=read(rel)
  if d is None:continue
  queues.append({'binding':objects[str(root/rel)]['binding'],'status':d.get('status'),'active_job':d.get('active_job'),'completed':d.get('completed',[])});jobs={j['id']:j for j in d.get('jobs',[])}
  for row in d.get('completed',[]):
   if row.get('phase')=='tiny':continue
   b=row.get('receipt')
   job=jobs.get(row['id'])
   if b:record(b['path'],rel,job,row)
   if row.get('outcome')=='FAIL':
    evidence={'queue':objects[str(root/rel)]['binding'],'completed':row,'job':{k:job[k] for k in ('id','command','receipt','phase') if k in job} if job else None}
    log=root/Path(rel).parent/(row['id']+'.log')
    if log.is_file():
     payload=log.read_bytes();evidence['log']={'path':str(log),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'tail_utf8':payload[-131072:].decode('utf-8',errors='replace'),'tail_truncated':len(payload)>131072}
    failed_attempt_evidence.append(evidence)
  objects.pop(str(root/rel),None)
 record(str(root/'first_pass_20261106/first_page/trace/receipt.json'),'first_page_motivation_metadata')
 for graph in ('US','EP','LJ','OR','IN','UK','TW'):
  for name in ('receipt.json','canonical.json','sources.json'):read('inputs/canonical/'+graph+'/'+name)
 for rel in ('setup.json',source_freeze,'formal_admission_d84_v2/manifest.json'):read(rel)
 scheduler_deferrals={}
 for kind,rel in [('event','first_pass_20261106/deferred_preparation_event.json'),('plan','first_pass_20261106/author_reservation_handoff/deferred_preparation.json')]:
  item=read(rel)
  if item is not None:
   scheduler_deferrals[kind]={**objects[str(root/rel)]['binding'],'canonical_sha256':hashlib.sha256(json.dumps(item,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}
 author_text_input_plan=None
 rel='first_pass_20261106/author_text_main/plan.json';item=read(rel)
 if item is not None:
  author_text_input_plan=objects[str(root/rel)]['binding'];refs(item.get('preparation'));refs(item.get('source_deferred'))
  for job in item.get('jobs',[]):record(job.get('receipt'),rel,job)
 return {'schema':'eggpu.first-pass.receipts.v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'author_text_input_plan':author_text_input_plan,
  'campaign':str(root),'selected_source_freeze':source_freeze,'selected_native_protocol':native_protocol,'scheduler_deferrals':scheduler_deferrals,'scope':'Explicit current plans plus their target proofs and completed queue receipts; JSON metadata plus failed-job log tails. No output vectors reread or measurements created.',
  'records':records,'plans':plans,'queues':queues,'objects':objects,'missing':missing,'failed_attempt_evidence':failed_attempt_evidence,'gpu_executed':False}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path)
 p.add_argument('--plan',action='append',default=[]);p.add_argument('--queue-state',action='append',default=[]);p.add_argument('--source-freeze',default='candidates/unified_release_20261106/freeze.json')
 p.add_argument('--native-protocol',default='native-host-csr-static-all4-public-return-v2')
 a=p.parse_args();d=collect(a.campaign,a.queue_state,a.plan,a.source_freeze,a.native_protocol);text=json.dumps(d,ensure_ascii=False,separators=(',',':'))
 if a.output:a.output.write_text(text+'\n')
 else:print(text)
