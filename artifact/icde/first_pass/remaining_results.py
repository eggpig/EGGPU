"""Source-bound n=1 E2/E4S/E6 tables in the existing export generation.

Only bounded JSON receipts are read. Recorded full-output audits are checked;
no algorithm, GPU operation, graph array, or output vector is replayed/read.
"""
from pathlib import Path
import argparse,collections,csv,functools,hashlib,html,importlib.util,json,math

HERE=Path(__file__).resolve().parent
SOURCE='71ea757d372f8c1b969ea074caaa27d3a77d4cea6d48cd9c6ae9c4b02936d7ce'
PROTOCOL='native-host-csr-static-all4-public-return-v2'
ROOT='first_pass_20261106/'
PATHS={'scaling':ROOT+'unified_scaling_continuation/plan.json',
 'main':ROOT+'unified_main_continuation/plan.json',
 'mechanism':ROOT+'unified_mechanism_application/plan.json',
 'cells':ROOT+'unified_mechanism_application/receipt.json',
 'selection':ROOT+'unified_mechanism_application/selection.json',
 'freeze':'candidates/unified_release_20261106/freeze.json'}
FUNCTIONS=('effective_size','constraint','closeness','bfs','wcc')
GROUPS={'structural':FUNCTIONS[:2],'closeness':('closeness',),
 'structural_closeness':FUNCTIONS[:3],'bfs':('bfs',),'wcc':('wcc',)}
LABELS={'eggpu':'EGGPU','kgraph_sparse':'KGraph','kgraph_dense':'KGraph',
 'c2graph_pm':'C²graph-PM','igraph':'igraph (CPU)',
 'cugraph_igraph':'cuGraph + igraph','cugraph_composite':'cuGraph + CuPy'}

@functools.lru_cache(maxsize=1)
def core():
 spec=importlib.util.spec_from_file_location('remaining_core',HERE/'build_first_pass.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def need(ok,message):
 if not ok:raise ValueError(message)

def arg(job,flag,default=None):
 command=job['command'];need(command.count(flag)<=1,'Ambiguous plan option '+flag)
 return command[command.index(flag)+1] if flag in command else default

def binding(value):
 need(isinstance(value,dict) and isinstance(value.get('path'),str) and
      type(value.get('bytes')) is int and value['bytes']>0 and
      isinstance(value.get('sha256'),str) and len(value['sha256'])==64,'Missing complete recorded file binding')
 return value

def same(a,b):
 return all(a.get(k)==b.get(k) for k in ('path','bytes','sha256'))

def bindings(value):
 if isinstance(value,dict):
  if {'path','bytes','sha256'}<=value.keys():yield value
  else:
   for item in value.values():yield from bindings(item)
 elif isinstance(value,list):
  for item in value:yield from bindings(item)

def capture(campaign,bundle):
 """Add only the explicitly selected plans and their exact output paths."""
 root=Path(campaign).resolve();objects=bundle['objects'];missing=bundle['missing']
 def read(path,recurse=False):
  path=Path(path);path=path if path.is_absolute() else root/path
  need(path.is_relative_to(root),'Remaining metadata escaped campaign')
  key=str(path)
  if key in objects:return objects[key]['value']
  if not path.is_file():
   if key not in missing:missing.append(key)
   return None
  need(path.suffix=='.json' and path.stat().st_size<=32*1024*1024,'Unbounded remaining metadata')
  raw=path.read_bytes()
  try:value=json.loads(raw)
  except (ValueError,UnicodeDecodeError):return None # A live partial write is pending.
  objects[key]={'binding':{'path':key,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'value':value}
  if recurse:refs(value)
  return value
 def refs(value):
  if isinstance(value,dict):
   if {'path','sha256'}<=value.keys():
    path=Path(value['path'])
    if path.suffix=='.json' and path.is_relative_to(root):read(path,True)
   else:
    for key,item in value.items():
     if key not in ('jobs','source_files','release_sources','gpu_kernel_sources','files','files_e6','bindings','tiny_proof','target_proof','qualification'):refs(item)
  elif isinstance(value,list):
   for item in value:refs(item)
 selected={name:read(path,name=='selection') for name,path in PATHS.items()}
 if selected['scaling']:read(selected['scaling']['main_p8_plan']['path'])
 for name in ('scaling','main','mechanism'):
  for job in (selected[name] or {}).get('jobs',[]):
   if job.get('phase')=='target':read(job['receipt'],True)
 for cell in (selected['cells'] or {}).get('cells',[]):read(cell['receipt'],True)
 # A reused P8 may already have passed before the continuation was prepared.
 for item in (selected['scaling'] or {}).get('p8_reuse',[]):read(item['receipt'],True)
 bundle['remaining_sources']={name:objects[str(root/path)]['binding'] for name,path in PATHS.items() if str(root/path) in objects}
 return bundle

def source_check(bundle,raw,epoch,freeze,selection):
 identity=raw.get('first_pass_source',{})
 need(identity.get('source_tree_sha256')==SOURCE,'EGGPU source identity differs from selected source71')
 need(bundle.get(identity['selection'])==selection,'EGGPU selection receipt differs')
 need(same(identity['runner_freeze'],selection['runner_freeze']) and
      same(identity['native_dso'],selection['native_dso']),'EGGPU runner/DSO selection differs')
 bundle.get(identity['runner_freeze'])
 observed={**epoch.get('release_sources',{}),**epoch.get('gpu_kernel_sources',{})}
 expected=freeze['source_files'];required={k for k in expected if k.startswith(('easygraph/','cpp_easygraph/')) and Path(k).suffix in ('.py','.cpp','.cu','.h','.hpp','.c','.cuh')}
 required.update(k for k in expected if k.startswith('gpu_easygraph/') and Path(k).suffix in ('.cu','.cpp','.h','.hpp','.cuh'))
 need(required<=observed.keys(),'EGGPU measured source coverage incomplete')
 for name,item in observed.items():
  need(name in expected and all(item[k]==expected[name][k] for k in ('bytes','sha256')),'EGGPU measured source differs: '+name)
 need(same(epoch.get('native_dso',epoch.get('files',{}).get('native_dso',{})),selection['native_dso']),'EGGPU measured DSO differs')
 if 'build_receipt' in epoch:need(same(epoch['build_receipt'],selection['build_receipt']),'EGGPU measured build receipt differs')

def producer_check(bundle,job,epoch,selection,own):
 worker=arg(job,'--worker') or next(x for x in job['command'] if x.endswith('.py'))
 expected={b['path']:b for b in job.get('bindings',[])}
 if own:expected.update({b['path']:b for b in bundle.get(selection['runner_freeze'])['files'].values()})
 observed={b['path']:b for b in bindings({k:epoch[k] for k in ('files','files_e6') if k in epoch})}
 need(worker in expected and worker in observed and same(expected[worker],observed[worker]),'Measured worker differs from selected job')
 for path,item in observed.items():
  if path in expected:need(same(item,expected[path]),'Measured helper differs from selected job: '+path)

def e4s_check(bundle,raw,job,cell,domain):
 need(raw.get('schema')=='e4s.complete.batch.v1','Wrong E4S schema')
 need(raw.get('variant')==cell['system'] and raw.get('batch')==cell['batch'],'Wrong E4S configuration')
 contract=job['target_contract'];need(raw['input_sha256']==contract['input_sha256'] and raw['directed']==contract['directed']==domain['directed'],'E4S canonical input identity differs')
 sequence=raw['source_sequence'];batch=cell['batch'];n=domain['identity']['n']
 need(sequence==contract['source_sequence'] and len(sequence)==len(set(sequence))==batch,'E4S frozen source sequence differs')
 bundle.get(raw['sources']);checks=raw['checks'];need(len(checks)==batch,'E4S incomplete output checks')
 for i,(source,check) in enumerate(zip(sequence,checks)):
  q=check['quality'];need(check['sequence']==i and check['source']==source and check['values']==q['vertices']==n and q.get('passed') is True and q.get('exact') is True,'E4S full output quality differs')
  need(q['reference'].get('independent') is True and len(check.get('output_sha256',''))==64,'E4S independent reference/output hash missing')
  if cell['system']=='eggpu':need(check.get('survives_session_close') is True,'E4S output lifetime failed')
 measurement=raw['measurements']
 first,ready=(core().num(measurement[k]) for k in ('first_complete_seconds','query_batch_seconds'))
 need(first>=ready,'E4S first interval shorter than query batch')
 if cell['system']=='eggpu':
  need(measurement.get('all_outputs_retained_through_close') is True,'E4S owning outputs not retained')
  requests=measurement['requests'];need(len(requests)==batch,'E4S incomplete timed requests')
  for i,(source,request) in enumerate(zip(sequence,requests)):
   need(request['sequence']==i and request['source']==source,'E4S timed source sequence differs');core().num(request['seconds'])
 else:
  interval=measurement['continuous_interval'];launch=bundle.get(interval['launch'])
  need(interval['clock_domain']['clock']=='CLOCK_MONOTONIC' and interval.get('phase_durations_summed') is False and interval['start_ns']<=launch['monotonic_launch_ns']<interval['end_ns']<=interval['parent_end_ns'],'E4S author continuous interval differs')
  need(launch['clock_domain']==interval['clock_domain'] and launch['parent_monotonic_start_ns']==interval['start_ns'] and launch['pid']==interval['pid'],'E4S author monotonic clock/PID differs')
  need(math.isclose((interval['end_ns']-interval['start_ns'])/1e9,first,rel_tol=1e-12),'E4S author complete interval differs')
  summary=measurement['summary']
  if cell['system'].startswith('kgraph'):need(summary['queries']==batch and summary['n']==n,'E4S KGraph complete request count differs')
  else:
   requests=measurement['requests'];need(summary['requested']==summary['completed']==batch and [r['request_id'] for r in requests]==list(range(batch)) and [r['source'] for r in requests]==sequence,'E4S C2 PM complete request sequence differs')
 return {'first_seconds':first,'ready_seconds':ready,'adapter_preparation_seconds':measurement['adapter_preparation_seconds'],'full_output_checks':batch,'first_boundary':raw['timing'],'ready_boundary':'complete batch after each system normal graph preparation; all owning FP64 outputs; no second-batch warmup, lazy per-request work included'}

def e6_devices(cell,function):
 system=cell['system'];fallback=function in GROUPS.get(cell['ablate'],())
 if system=='igraph':return []
 if system=='cugraph_igraph':return [] if function in ('constraint','closeness') else [0]
 if system=='cugraph_composite':return [0]
 if fallback:return [] if cell['coverage_backend']=='igraph' else [0]
 return list(range(cell['gpu_count'])) if function in FUNCTIONS[:3] else [0]

def e6_check(bundle,raw,epoch,cell,domain):
 authority=json.loads((HERE/'remaining_protocol.json').read_text())
 need(raw.get('schema')=='eggpu.e6.job/coverage2x2-v2','Wrong E6 schema')
 for value in (raw,epoch):
  need(value.get('protocol')==authority['E6_PROTOCOL'] and value.get('contract')==authority['E6_CONTRACT'],'Wrong E6 complete workflow protocol/contract')
  need(all(value.get(k)==cell[k] for k in ('system','ablate','coverage_backend','gpu_count')),'E6 resource/fallback configuration differs')
 need(raw['input_sha256']==domain['sha256'],'E6 canonical input identity differs')
 child=bundle.get(raw['child']);need(child.get('status')=='PASS' and child.get('quality_pass') is True and child.get('outputs_survive_close') is True,'E6 full output or lifetime check failed')
 need(child['input_sha256']==raw['input_sha256'] and set(child['calls'])=={'first','warmup','ready'},'E6 incomplete calls/input')
 weak=child['weak_projection'];n=domain['identity']['n']
 need(weak['n']==n and weak.get('source_vertices_preserved') is True and weak.get('weights_used') is False,'E6 weak projection contract differs')
 details={};names=set(FUNCTIONS)|{'candidates','winner'}
 for label,call in child['calls'].items():
  q=call['quality'];need(q.get('pass') is True and set(call['outputs'])==names and set(q['vectors'])==names,'E6 incomplete full-stage quality')
  need(call.get('timed') is (label!='warmup') and all(q.get(k) is True for k in ('actual_candidate_selection','actual_winner_selection','bfs_wcc_membership_exact')),'E6 dependent selection/measurement differs')
  for name,out in call['outputs'].items():
   binding(out);v=q['vectors'][name];size=32 if name in ('candidates','closeness') else 1 if name=='winner' else n
   need(all(v.get(k) is True for k in ('pass','owning','immutable')) and v.get('shape')==[size],'E6 output domain/ownership differs')
   if name in FUNCTIONS[:3]:
    need(v.get('dtype')=='float64' and v.get('nan_mask_equal') is True and v.get('finite_mask_equal') is True,'E6 FP64 reference masks differ')
    for key in ('rmse_finite','max_absolute_error_finite'):
     need(type(v.get(key)) in (int,float) and math.isfinite(v[key]) and v[key]>=0,'E6 numerical audit missing')
  trace=call['trace'];stages=trace['stages'];need(tuple(s['function'] for s in stages)==FUNCTIONS,'E6 causal stage order differs');last=0.
  for stage in stages:
   function=stage['function'];metrics=stage['metrics'];fallback=function in GROUPS.get(cell['ablate'],())
   need(metrics['devices_used']==e6_devices(cell,function),'E6 actual device composition differs')
   if cell['system']=='eggpu':need(metrics.get('coverage_fallback')==fallback and metrics.get('coverage_backend')==(cell['coverage_backend'] if fallback else 'eggpu'),'E6 fallback implementation differs')
   start,end=stage['start_seconds'],stage['end_seconds']
   need(start>=last-1e-8 and end>=start and math.isclose(end-start,stage['complete_seconds'],rel_tol=1e-8,abs_tol=1e-8),'E6 stage interval differs');last=end
  total=core().num(call['seconds']);ranking=trace['candidate_seconds']+trace['winner_seconds']
  need(ranking>=0 and total-sum(s['complete_seconds'] for s in stages)-ranking>=-1e-6,'E6 stages exceed complete interval')
  if label!='warmup':
   need(math.isclose(total,raw['T_'+label+'_seconds'],rel_tol=1e-12),'E6 recorded parent interval differs')
   details[label+'_seconds']=total
 details.update(full_output_checks=21,first_boundary=authority['E6_CONTRACT']['timing'],ready_boundary='same complete dependent workflow with retained graph state after one full warmup')
 return details

def planned_cells(bundle):
 plans={k:bundle.get(v) for k,v in PATHS.items()};scale,main,mechanism=plans['scaling'],plans['main'],plans['mechanism']
 need(scale.get('source_tree_sha256')==main.get('source_tree_sha256')==mechanism.get('own_source_tree_sha256')==SOURCE and scale.get('protocol')==PROTOCOL,'Selected plans do not bind source71/static protocol')
 cells=[]
 for job in scale['jobs']:
  if job.get('phase')=='target':cells.append({'experiment':'E2','dataset':arg(job,'--dataset'),'algorithm':arg(job,'--algorithm'),'gpu_count':int(arg(job,'--count')),'system':'eggpu','job':job,'raw_receipt':job['receipt'],'p8_reused':False})
 mapping={}
 for job in main['jobs']:
  original=job.get('continuation',{}).get('original_receipt')
  if original:need(original not in mapping,'Ambiguous continuation receipt mapping');mapping[original]=job
 original_main=bundle.get(scale['main_p8_plan'])
 originals={job['receipt']:job for job in original_main['jobs']}
 for item in scale['p8_reuse']:
  need(item['source_tree_sha256']==SOURCE and item['protocol']==PROTOCOL,'P8 reuse identity differs')
  original=item['receipt'];job=mapping.get(original) or originals[original]
  need(job.get('phase')=='target' and arg(job,'--dataset')==item['dataset'] and arg(job,'--algorithm')==item['algorithm'] and arg(job,'--count')=='8','P8 continuation mapping changes configuration')
  cells.append({'experiment':'E2','dataset':item['dataset'],'algorithm':item['algorithm'],'gpu_count':8,'system':'eggpu','job':job,'raw_receipt':job['receipt'],'p8_reused':True,'original_receipt':original})
 expected={(g,a,p) for g in ('US','OR','IN','UK','TW') for a in ('bfs','sssp','wcc','pr') for p in (1,2,4,8)}
 keys=[(x['dataset'],x['algorithm'],x['gpu_count']) for x in cells];need(len(keys)==len(set(keys)) and set(keys)==expected,'E2 frozen 80-cell grid differs')
 jobs={j['id']:j for j in mechanism['jobs']}
 for reused in plans['cells'].get('external_reused',[]):jobs[reused['original_job']['id']]=reused['original_job']
 for declared in plans['cells']['cells']:
  job=jobs[declared['id']];need(job['receipt']==declared['receipt'] and job.get('phase')=='target','Mechanism output path/phase differs')
  cell={'experiment':declared['stage'].upper(),'dataset':arg(job,'--dataset'),'gpu_count':job['gpu_count'],'system':arg(job,'--variant') or arg(job,'--system'),'job':job,'raw_receipt':job['receipt']}
  if cell['experiment']=='E4S':cell['batch']=int(arg(job,'--batch'))
  else:cell.update(ablate=arg(job,'--ablate'),coverage_backend=arg(job,'--coverage-backend'))
  cells.append(cell)
 e4=[(c['dataset'],c['system'],c['batch']) for c in cells if c['experiment']=='E4S']
 need(len(e4)==36 and len(set(e4))==36 and set(e4)=={(g,s,q) for g in ('US','OR','IN','UK') for s in ('eggpu','c2graph_pm','kgraph_sparse' if g=='US' else 'kgraph_dense') for q in (32,128,512)},'E4S frozen 36-cell grid differs')
 configurations={('eggpu','none','none',p) for p in (1,2,4,8)}|{('eggpu',a,b,1) for a in ('structural','closeness','structural_closeness') for b in ('igraph','cugraph_composite')}|{('eggpu',a,'igraph',1) for a in ('bfs','wcc')}|{(s,'none','none',0 if s=='igraph' else 1) for s in ('igraph','cugraph_igraph','cugraph_composite')}
 e6=[(c['dataset'],c['system'],c['ablate'],c['coverage_backend'],c['gpu_count']) for c in cells if c['experiment']=='E6']
 need(len(e6)==45 and len(set(e6))==45 and set(e6)=={(g,*c) for g in ('EP','LJ','OR') for c in configurations},'E6 frozen 45-cell grid differs')
 return cells,plans

def build(bundle_path,output):
 c=core();bundle=c.Bundle(bundle_path);cells,plans=planned_cells(bundle);freeze=plans['freeze'];selection=plans['selection']
 need(freeze.get('source_tree_sha256')==selection.get('source_tree_sha256')==SOURCE,'Remaining source authority differs')
 domains={g:bundle.domain(g) for g in c.GRAPHS};rows=[]
 attempts={row['id']:row for queue in bundle.data.get('queues',[]) for row in queue.get('completed',[])}
 for cell in cells:
  job=cell['job'];row={k:v for k,v in cell.items() if k!='job'}
  row.update(job_id=job['id'],label=LABELS[cell['system']],status='PENDING',first_seconds=None,ready_seconds=None,n_independent=0,source_tree_sha256=SOURCE if cell['system']=='eggpu' else None,measurement_protocol=PROTOCOL if cell['experiment']=='E2' else None)
  attempt=attempts.get(job['id']);row['attempt']=attempt
  try:
   if cell['raw_receipt'] not in bundle.objects:
    if attempt and attempt.get('outcome')=='FAIL':row.update(status='FAILED',reason=attempt.get('status','FAILED_ATTEMPT'))
    rows.append(row);continue
   raw=bundle.get(cell['raw_receipt']);row.update(raw_status=raw.get('status'),raw_binding=bundle.objects[cell['raw_receipt']]['binding'],phase=raw.get('phase'),repeat=raw.get('repeat'),original_performance_eligible=raw.get('performance_eligible'))
   if raw.get('status') in ('RUNNING','WAITING'):rows.append(row);continue
   if raw.get('status') not in ('PASS','PASS_QUALIFICATION','PASS_FORMAL','PASS_E4S_QUALIFICATION','PASS_E4S_FORMAL'):
    row.update(status='FAILED',reason=raw.get('error',raw.get('status')));rows.append(row);continue
   need(raw.get('phase')=='target' and raw.get('repeat')==0 and raw.get('dataset')==cell['dataset'] and raw.get('quality_pass') is True,'Wrong target/r0/full-output quality identity')
   if attempt and attempt.get('receipt'):need(same(row['raw_binding'],attempt['receipt']),'Completed queue receipt binding differs')
   epoch=bundle.get(raw['epoch']);need(c.digest(epoch)==raw['epoch_sha256'],'Recorded epoch digest differs');row['epoch']=raw['epoch']
   if cell['experiment']=='E2':
    need(raw.get('implementation')=='forced_native' and raw.get('protocol')==PROTOCOL and raw.get('gpu_count',raw.get('count'))==cell['gpu_count'] and raw.get('algorithm',raw.get('function'))==cell['algorithm'],'Wrong fixed native E2 route/protocol/configuration')
    observation=c.observation(bundle,{'path':cell['raw_receipt'],'origin':'explicit_current_E2_plan','completed':attempt},freeze,domains)
    need(observation['status']=='FIRST_OBSERVATION_AVAILABLE','E2 full observation not available')
    need(same(epoch.get('native_dso',epoch.get('files',{}).get('native_dso',{})),selection['native_dso']),'E2 measured native DSO differs')
    producer_check(bundle,job,epoch,selection,False)
    for item in raw['sources']:
     child=bundle.get(item['result']);need(child['partition_audit']['status']=='PASS_FULL_PARTITION_ROUNDTRIP' and child.get('outputs_survive_close') is True,'E2 full partition/lifetime audit missing');bundle.get(child['preparation']['manifest'])
     for call in child['calls'].values():need(call['native_metrics']['devices_used']==list(range(cell['gpu_count'])),'E2 actual native devices differ')
    row.update({k:observation[k] for k in ('first_seconds','ready_seconds','source_values','output_bindings')})
    row.update(first_boundary='canonical host CSR ready to complete owning FP64 result, averaged across fixed sources',ready_boundary='retained static graph, complete owning FP64 result after warmup')
   else:
    if cell['system']=='eggpu':source_check(bundle,raw,epoch,freeze,selection)
    producer_check(bundle,job,epoch,selection,cell['system']=='eggpu')
    details=e4s_check(bundle,raw,job,cell,domains[cell['dataset']]) if cell['experiment']=='E4S' else e6_check(bundle,raw,epoch,cell,domains[cell['dataset']])
    row.update(details);row['measurement_protocol']=raw.get('protocol',raw.get('schema'))
   row.update(status='AVAILABLE',n_independent=1,quality_pass=True,qualification_promoted_to_formal=False,payload_revalidated_by_this_export=False)
  except (KeyError,ValueError,TypeError,StopIteration) as exc:row.update(status='REVIEW',reason=str(exc),first_seconds=None,ready_seconds=None,n_independent=0)
  rows.append(row)
 # Scaling never uses the optimized Arena E1/P1 denominator.
 p1={(r['dataset'],r.get('algorithm')):r for r in rows if r['experiment']=='E2' and r['gpu_count']==1 and r['status']=='AVAILABLE'}
 for row in rows:
  base=p1.get((row['dataset'],row.get('algorithm')))
  if row['experiment']=='E2' and row['status']=='AVAILABLE' and base:row['ready_speedup_from_native_p1']=base['ready_seconds']/row['ready_seconds'];row['speedup_denominator_receipt']=base['raw_receipt']
 counts={ex:dict(collections.Counter(r['status'] for r in rows if r['experiment']==ex)) for ex in ('E2','E4S','E6')}
 report={'schema':'eggpu.remaining-first-observation.v1','primary_metric':'ready_seconds','supplementary_metric':'first_seconds','primary_timing':{'E2':'retained static graph call','E4S':'normal graph preparation then complete batch; not warmed second batch; lazy work included','E6':'retained graph after one full warmup, complete dependent workflow'},'source_tree_sha256':SOURCE,'static_protocol':PROTOCOL,'observed_utc':bundle.data.get('observed_utc'),'counts':counts,'cells':rows,'selection_bindings':bundle.data.get('remaining_sources',{}),'n_independent_max':1,'error_bars':False,'missing_rendered_as_zero':False,'array_payloads_read':False,'gpu_executed':False,'validation_scope':'Recorded complete-output numerical, lifetime, source, input, resource and timing audits; no new array verification or measurements.'}
 out=Path(output);out.mkdir(parents=True,exist_ok=True);(out/'remaining_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 fields=('experiment','dataset','label','system','algorithm','batch','gpu_count','ablate','coverage_backend','status','first_seconds','ready_seconds','ready_speedup_from_native_p1','n_independent','source_tree_sha256','measurement_protocol','raw_receipt','original_receipt','p8_reused','epoch','reason','first_boundary','ready_boundary','original_performance_eligible')
 with (out/'remaining_results.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
  for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
 text=['<!doctype html><meta charset="utf-8"><title>E2 / E4S / E6</title><style>body{font:14px system-ui;margin:28px;color:#172033}table{border-collapse:collapse}td,th{padding:6px 9px;border-bottom:1px solid #ddd;text-align:right}details{margin:15px 0}p{max-width:1100px;line-height:1.7}</style><p><a href="../first_pass.html">返回主表</a> · <a href="remaining_results.csv">完整 CSV</a> · <a href="remaining_results.json">证据与计时边界</a></p><h1>E2 / E4S / E6 首轮观测</h1><p>EGGPU 固定源码 71ea757；E2 固定原生静态协议，P8 按原计划到续跑计划的精确路径映射复用。n=1，无误差条；PASS target 按原身份保留，未改称正式多次实验。待完成、失败、复核格均无时间值。这里只复核已有完整数值校验，不重新读取向量。主图统一使用实际 ready 后区间，first 为补充。E4S 从各系统正常图准备后计完整批次，不是第二次热批次，按需准备仍计入；E6 是完整热身后包含五个算子和依赖选择的独立完整工作流。</p>']
 for ex in counts:
  text.append('<h2>'+ex+' · '+html.escape(str(counts[ex]))+'</h2><table><tr><th>Graph</th><th>System/configuration</th><th>Ready Time(s)</th><th>First Time(s), supplement</th><th>Status</th></tr>')
  for row in rows:
   if row['experiment']!=ex:continue
   config=' / '.join(str(row[k]) for k in ('label','algorithm','batch','gpu_count','ablate','coverage_backend') if k in row)
   value=lambda k:format(row[k],'.6g') if row[k] is not None else '—'
   title=html.escape(row.get('reason') or row['raw_receipt'],quote=True)
   text.append('<tr title="'+title+'"><td>'+row['dataset']+'</td><td>'+html.escape(config)+'</td><td>'+value('ready_seconds')+'</td><td>'+value('first_seconds')+'</td><td>'+row['status']+'</td></tr>')
  text.append('</table>')
 text.append('<p>图形由同一次导出的 render_icde 公共样式生成。E6 的 CPU 对手和混合实现按其真实资源单列；扩展图是复制图上的分工扩展，不宣称分区容量扩展。详细原始身份、原 target 标签和边界见 CSV/JSON。</p>')
 (out/'remaining_results.html').write_text('\n'.join(text));return report

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();print(build(a.bundle,a.output)['counts'])
