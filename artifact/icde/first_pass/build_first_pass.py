"""First complete observations from existing full target or r0 receipts.

This presentation reuses recorded correctness and source contracts. It does not
rerun algorithms, reread array payloads, promote target to formal, or wait for
statistical repetitions. A failed or missing cell remains visible.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,html,json,math,statistics
from datetime import datetime
from zoneinfo import ZoneInfo

GRAPHS=('US','EP','LJ','OR','IN','UK','TW')
ALGORITHMS=('bfs','sssp','wcc','pr')
GOOD={'PASS','PASS_QUALIFICATION','PASS_FORMAL'}
LABELS={'eggpu':'EGGPU','cugraph':'cuGraph','cugraph_mg':'cuGraph','gum':'GUM',
 'gunrock':'Gunrock','cggraph':'CGgraph','sygraph':'SYgraph','grapin':'Grapin',
 'hytgraph':'HyTGraph','c2graph':'C²graph-P'}


def digest(x):
 return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def need(ok,message):
 if not ok:raise ValueError(message)


def key_of(raw):
 system=raw.get('system');variant=raw.get('variant');count=raw.get('gpu_count',raw.get('count'))
 if system=='c2graph_p':system,variant='c2graph','c2graph_p'
 if system=='c2graph' and variant is None:variant='c2graph_p'
 return (raw.get('dataset'),raw.get('algorithm',raw.get('function')),count,system,variant)


class Bundle:
 def __init__(self,path,supplement=None):
  self.path=Path(path)
  with (gzip.open(self.path,'rt') if self.path.suffix=='.gz' else self.path.open()) as stream:self.data=json.load(stream)
  self.objects=self.data['objects'];self.root=self.data['campaign'];self.supplement=supplement
  if supplement:
   extra=json.loads(Path(supplement).read_text());self.objects.update(extra.get('objects',extra))
 def get(self,item):
  path=item['path'] if isinstance(item,dict) else str(item)
  if not path.startswith('/'):path=self.root+'/'+path
  obj=self.objects[path]
  if isinstance(item,dict):
   for k in ('bytes','sha256'):
    if k in item:need(obj['binding'][k]==item[k],'Captured JSON binding differs: '+path)
  return obj['value']
 def domain(self,graph):
  receipt=self.get('inputs/canonical/'+graph+'/receipt.json');meta=self.get('inputs/canonical/'+graph+'/canonical.json')
  sources=self.get('inputs/canonical/'+graph+'/sources.json')['primary_sources']
  b=self.objects[self.root+'/inputs/canonical/'+graph+'/receipt.json']['binding']
  identity=dict(receipt={k:b[k] for k in ('bytes','sha256')},
   files={name:{k:item[k] for k in ('bytes','sha256')} for name,item in receipt['files'].items()},
   sources=sources,n=meta['num_nodes'],m=meta['num_entries'],directed=meta['directed'])
  return dict(identity=identity,sha256=digest(identity),sources=sources,directed=meta['directed'])


def quality(q,algorithm):
 need(q.get('pass') is True and q.get('complete') is True,'Existing complete-output check did not pass')
 if algorithm in ('bfs','sssp'):
  need(q.get('exact_reference_match') is True and q.get('unreachable_match') is True,'Existing distance reference check did not pass')
 elif algorithm=='wcc':need(q.get('partition_equivalent') is True,'Existing component partition check did not pass')
 else:
  for k,limit in [('residual_l1',1e-6),('mass_error',1e-6),('reference_l1',1e-5)]:
   v=q.get(k);need(type(v) in (int,float) and math.isfinite(v) and 0<=v<=limit,'Existing PR '+k+' threshold did not pass')


def num(x):
 need(type(x) in (int,float) and math.isfinite(x) and x>0,'Missing/nonpositive complete timing')
 return float(x)


def observation(bundle,entry,freeze,domains):
 raw=bundle.get(entry['path']);dataset,algorithm,count,system,variant=key_of(raw)
 out=dict(dataset=dataset,algorithm=algorithm,gpu_count=count,system=system,variant=variant,
  phase=raw.get('phase'),repeat=raw.get('repeat'),measurement_protocol=raw.get('protocol'),raw_status=raw.get('status'),raw_receipt=entry['path'],experiment=raw.get('experiment','E1'),implementation=raw.get('implementation'),
  origin=entry['origin'],input_sha256=raw.get('input_sha256'),epoch_sha256=raw.get('epoch_sha256'),
  first_seconds=None,ready_seconds=None,job_wall_seconds=(entry.get('completed') or {}).get('seconds'),
  status='UNAVAILABLE',quality_pass=raw.get('quality_pass'),selection_order=0)
 need(dataset in GRAPHS and algorithm in ALGORITHMS and count in (1,2,4,8),'Outside first-pass core-function scope')
 need(raw.get('phase') in ('target','formal'),'Only full target or formal receipts can supply first pass')
 need(raw.get('repeat',0)==0 or raw.get('phase')=='target','Later independent repeats are reserved for the second pass')
 domain=domains[dataset]
 if raw.get('status') not in GOOD:
  errors=[raw.get('error')] if raw.get('error') else []
  for child in raw.get('sources',[]):
   if child.get('error'):errors.append(child['error'])
   link=child.get('result')
   if isinstance(link,dict) and link.get('path') in bundle.objects:
    detail=bundle.get(link)
    if detail.get('error'):errors.append(detail['error'])
  out.update(status='IN_PROGRESS' if raw.get('status') in ('RUNNING','WAITING') else 'FAILED_ATTEMPT',error='; '.join(dict.fromkeys(errors)) or raw.get('status'));return out
 need(raw.get('input_sha256')==domain['sha256'] and raw.get('directed')==domain['directed'],'Different frozen graph/input identity')
 need(raw.get('quality_pass') is True,'Existing full job quality did not pass')
 epoch=bundle.get(raw['epoch']);need(digest(epoch)==raw['epoch_sha256'],'Recorded epoch identity differs')
 if system=='eggpu':
  selected_protocol=bundle.data.get('selected_native_protocol')
  if selected_protocol and (raw.get('implementation')=='forced_native' or raw.get('schema')=='eggpu.native.job.v1'):
   need(raw.get('protocol')==selected_protocol,'Native measurement endpoint differs from selected protocol')
  observed=dict(epoch.get('release_sources',{}));observed.update(epoch.get('gpu_kernel_sources',{}));expected=freeze['source_files']
  required={name for name in expected if name.startswith(('easygraph/','cpp_easygraph/')) and Path(name).suffix in ('.py','.cpp','.cu','.h','.hpp','.c','.cuh')}
  need(required<=observed.keys(),'EGGPU measured runtime source coverage incomplete')
  for name,b in observed.items():
   need(name in expected and all(expected[name][k]==b[k] for k in ('bytes','sha256')),'EGGPU source differs from current frozen release: '+name)
 sources=raw.get('sources',[]);expected=domain['sources'] if algorithm in ('bfs','sssp') else [None]
 need(len(sources)==len(expected),'Incomplete fixed-source task')
 if len(expected)==3:need([x.get('source') for x in sources]==expected,'Fixed sources/order differ')
 values=[];warmup=[];outputs=[]
 for item in sources:
  child=item.get('result',item)
  if isinstance(child,dict) and 'path' in child:child=bundle.get(child)
  need(child.get('status')=='PASS' and child.get('quality_pass',True) is True,'Child source did not pass')
  need(child.get('source')==item.get('source'),'Child source identity differs')
  if 'calls' in child:
   calls=child['calls'];need({'first','ready','warmup'}<=calls.keys(),'Incomplete first/warmup/ready task')
   for c in calls.values():quality(c['quality'],algorithm)
   first=num(calls['first']['seconds']);ready=num(calls['ready']['seconds'])
   if raw.get('schema') in ('eggpu.e1.sg.author.v1','eggpu.e1.c2graph_p.v1','eggpu.e1.gunrock-legacy-p8-complete.v1'):
    need(calls['first'].get('continuous_interval',{}).get('status')=='PASS_CONTINUOUS_MONOTONIC_INTERVAL','Author first time is not the original continuous interval')
   for c in calls.values():
    b=c.get('output');need(isinstance(b,dict) and 'sha256' in b and b.get('bytes',0)>0,'Complete output binding missing');outputs.append(b)
   warmup.append(calls['warmup'].get('seconds'))
  else:
   if system=='gum' and algorithm=='pr':
    precision=child.get('precision_evidence',{})
    need(precision.get('compute_dtype')=='float32' and precision.get('final_owned_dtype')=='float64' and precision.get('output_normalization_inside_query') is True,'Current GUM PR must retain original FP32 compute and timed owned output normalization')
   rounds=child.get('rounds',[]);need(len(rounds)==3,'GUM initial/warmup/ready rounds missing')
   for row in rounds:quality(row['quality'],algorithm);outputs.append(row['raw_output'])
   need(child.get('contiguous_canonical_to_first_result') is True,'GUM first interval is not continuous')
   first=num(child['T_first_seconds']);ready=num(child['T_ready_seconds'])
  values.append(dict(source=item.get('source'),first_seconds=first,ready_seconds=ready))
 first=statistics.mean(x['first_seconds'] for x in values);ready=statistics.mean(x['ready_seconds'] for x in values)
 for label,value in [('first',first),('ready',ready)]:
  if 'T_'+label+'_seconds' in raw:need(math.isclose(value,raw['T_'+label+'_seconds'],rel_tol=1e-10),'Recorded source average differs')
 out.update(status='FIRST_OBSERVATION_AVAILABLE',first_seconds=first,ready_seconds=ready,n_independent=1,
  source_values=values,output_bindings=outputs,epoch=raw['epoch'],source_tree_sha256=freeze.get('source_tree_sha256') if system=='eggpu' else None,
  measurement_kind='existing_full_target_observation' if raw['phase']=='target' else 'existing_formal_r0_observation',
  qualification_promoted_to_formal=False,payload_revalidated_by_this_export=False)
 return out


def label(row):
 name=LABELS.get(row.get('system'),row.get('system') or 'Gunrock (8 GPU)')
 if row.get('variant')=='cggraph_v15':name+=' v1.5'
 elif row.get('variant')=='cggraph_legacy_full':name+=' (legacy)'
 return name


def fmt(x):
 if x is None:return '—'
 return f'{x:.4g}'


def build(bundle_path,authority_path,output,supplement=None):
 b=Bundle(bundle_path,supplement);authority=json.loads(Path(authority_path).read_text());freeze_key=b.data.get('selected_source_freeze','candidates/native_memory_input_v1/freeze.json');freeze=b.get(freeze_key)
 domains={g:b.domain(g) for g in GRAPHS};attempts=[];excluded=[]
 for i,entry in enumerate(b.data['records']):
  raw=b.get(entry['path']);key=key_of(raw)
  if key[0] not in GRAPHS or key[1] not in ALGORITHMS or raw.get('phase') not in ('target','formal'):continue
  if raw.get('phase')=='formal' and raw.get('repeat')!=0:continue
  if key[3]=='gum' and key[1]=='pr' and raw.get('epoch'):
   epoch=b.get(raw['epoch'])
   if epoch.get('precision',{}).get('compute')!='author float32':
    excluded.append(dict(raw_receipt=entry['path'],cell=list(key),reason='Historical GUM FP64 PR; current baseline requires original author FP32',classification='historical_variant',raw_status=raw.get('status'),phase=raw.get('phase')));continue
  try:r=observation(b,entry,freeze,domains);r['selection_order']=i;attempts.append(r)
  except (KeyError,ValueError,TypeError) as e:
   excluded.append(dict(raw_receipt=entry['path'],cell=list(key),reason=str(e),raw_status=raw.get('status'),phase=raw.get('phase')))
 cells=[]
 for original_spec in authority['rows']:
  spec=dict(original_spec)
  if spec.get('system') is None and spec['gpu_count']==8 and spec['algorithm']=='wcc':
   spec.update(system='gunrock',variant='gunrock_legacy_cc')
  for graph in GRAPHS:
   key=(graph,spec['algorithm'],spec['gpu_count'],spec['system'],spec.get('variant'))
   rows=[r for r in attempts if (r['dataset'],r['algorithm'],r['gpu_count'],r['system'],r['variant'])==key]
   if spec['role']=='eggpu' and spec['gpu_count']==1:rows=[r for r in rows if r['experiment']=='E1' and r.get('implementation')!='forced_native']
   # Predeclared preference: first successful original target in frozen plan order;
   # only if no completed target exists, use original formal r0. Never compare time.
   valid=sorted([r for r in rows if r['status']=='FIRST_OBSERVATION_AVAILABLE'],key=lambda r:(r['phase']!='target',r['selection_order']))
   selected=valid[0] if valid else None
   cell=dict(dataset=graph,algorithm=spec['algorithm'],gpu_count=spec['gpu_count'],system=spec['system'],variant=spec.get('variant'),label=label(spec),role=spec['role'],
    status='AVAILABLE' if selected else ('FAILED' if any(r['status']=='FAILED_ATTEMPT' for r in rows) else 'PENDING'),first_seconds=selected['first_seconds'] if selected else None,
    ready_seconds=selected['ready_seconds'] if selected else None,measurement_kind=selected['measurement_kind'] if selected else None,
    measurement_protocol=selected.get('measurement_protocol') if selected else None,raw_receipt=selected['raw_receipt'] if selected else None,attempts=[r['raw_receipt'] for r in rows],
    excluded_attempts=[r for r in excluded if tuple(r['cell'])==key])
   if selected:cell['observation']=selected
   failure_rows=[r for r in rows if r['status']=='FAILED_ATTEMPT']
   cell['failure_reasons']=[r.get('error') or r.get('raw_status') for r in failure_rows]
   cell['failure_class']=None
   if cell['status']=='FAILED':
    reason=' '.join(str(x) for x in cell['failure_reasons'])
    cell['failure_class']=('OUT_OF_MEMORY' if any(x in reason.lower() for x in ('out_of_memory','out of memory','outofmemory','cudaerrormemoryallocation')) else 'INDEX_LIMIT' if 'IndexUnsupported' in reason or 'INDEX_UNSUPPORTED' in reason else 'QUALITY_FAILED' if 'quality failed' in reason.lower() else 'PREPROCESSING_TIMEOUT' if 'TimeoutExpired' in reason and '/compress' in reason else 'TIMEOUT' if 'TimeoutExpired' in reason else 'PROCESS_FAILURE' if 'executable returned' in reason or 'executable failed' in reason else 'FAILED_UNCLASSIFIED')
   if not selected and any(x.get('classification')!='historical_variant' for x in cell['excluded_attempts']):cell['status']='REVIEW_NEEDED'
   cells.append(cell)
 comparisons=[]
 for count in (1,8):
  for graph in GRAPHS:
   for algorithm in ALGORITHMS:
    group=[r for r in cells if (r['gpu_count'],r['dataset'],r['algorithm'])==(count,graph,algorithm)]
    own=next(r for r in group if r['role']=='eggpu');external=[r for r in group if r['role']=='external'];available=[r for r in external if r['status']=='AVAILABLE']
    row=dict(dataset=graph,algorithm=algorithm,gpu_count=count,eggpu_first_seconds=own['first_seconds'],eggpu_ready_seconds=own['ready_seconds'],
     available_baselines=len(available),required_baselines=len(external),matrix_resolved=all(r['status'] in ('AVAILABLE','FAILED') for r in group),all_systems_have_usable_times=all(r['status']=='AVAILABLE' for r in group),strongest_observed_baseline=None,baseline_ready_seconds=None,speedup_ready=None,strongest_observed_baseline_first=None,baseline_first_seconds=None,speedup_first=None)
    if available:
     fastest=min(available,key=lambda r:r['ready_seconds']);row.update(strongest_observed_baseline=fastest['label'],baseline_ready_seconds=fastest['ready_seconds'])
     if own['ready_seconds'] is not None:row['speedup_ready']=fastest['ready_seconds']/own['ready_seconds']
     first_best=min(available,key=lambda r:r['first_seconds']);row.update(strongest_observed_baseline_first=first_best['label'],baseline_first_seconds=first_best['first_seconds'])
     if own['first_seconds'] is not None:row['speedup_first']=first_best['first_seconds']/own['first_seconds']
    comparisons.append(row)
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 report=dict(schema='eggpu.first-pass.e1.presentation.v1',status='PARTIAL_FIRST_OBSERVATIONS_NOT_COMPLETE_EVALUATION',observed_utc=b.data['observed_utc'],
  source_bundle={'path':str(Path(bundle_path).resolve()),'sha256':hashlib.sha256(Path(bundle_path).read_bytes()).hexdigest()},
  metadata_supplement={'path':str(Path(supplement).resolve()),'sha256':hashlib.sha256(Path(supplement).read_bytes()).hexdigest()} if supplement else None,
  selected_native_protocol=b.data.get('selected_native_protocol'),input_domains={k:v['sha256'] for k,v in domains.items()},source_freeze=b.objects[b.root+'/'+freeze_key]['binding'],
  selection='Existing complete target first in original plan order, otherwise formal r0; never fastest replicate. Every failure and unavailable cell retained.',
  timings='Seconds per complete host-owned result; first includes preparation, ready retains prepared state. BFS/SSSP mean over fixed three sources is one observation; queue wall is excluded.',
  evidence='Recorded full-vector correctness and source/input JSON evidence reused. No new array reads or GPU execution; target receipts remain target, not formal.',
  cells=cells,comparisons=comparisons,excluded=excluded,all_attempts=attempts,
  counts={'required_cells':len(cells),'available_cells':sum(x['status']=='AVAILABLE' for x in cells),'pending_cells':sum(x['status']=='PENDING' for x in cells),'failed_cells':sum(x['status']=='FAILED' for x in cells),'review_cells':sum(x['status']=='REVIEW_NEEDED' for x in cells)})
 if report['counts']['pending_cells']==0 and report['counts']['review_cells']==0:
  report['status']='COMPLETE_FIRST_PASS_WITH_RECORDED_FAILURES' if report['counts']['failed_cells'] else 'COMPLETE_FIRST_PASS_ALL_CELLS_HAVE_USABLE_OBSERVATIONS'
 report['complete_first_pass']=report['counts']['pending_cells']==0 and report['counts']['review_cells']==0
 report['all_cells_have_usable_times']=report['counts']['available_cells']==report['counts']['required_cells']
 (out/'first_pass.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 missing=[{k:x[k] for k in ('gpu_count','dataset','algorithm','system','variant','label','status','failure_class','failure_reasons','attempts','excluded_attempts')} for x in cells if x['status']!='AVAILABLE']
 (out/'missing_cells.json').write_text(json.dumps(missing,ensure_ascii=False,indent=2)+'\n')
 fields=['gpu_count','dataset','algorithm','label','status','first_seconds','ready_seconds','measurement_kind','measurement_protocol','raw_receipt','failure_class','failure_reasons']
 with (out/'all_systems.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(cells)
 with (out/'comparison.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(comparisons[0]),extrasaction='ignore');writer.writeheader();writer.writerows(comparisons)
 observed_local=datetime.fromisoformat(b.data['observed_utc']).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S（北京时间）')
 markdown=['# 首轮结果：既有完整目标测量，n=1','',f"采集时间：{observed_local}。固定矩阵 {len(cells)} 格，已有 {report['counts']['available_cells']} 格可用；其余保留缺项/失败。",'',
  '优先复用原完整 target 的已记录测量；它们仍标为 target，不重标 formal。当前不能写最终 SOTA 比例或置信区间。主表执行时间（原字段 T_ready）覆盖准备完成后的计算、同步及返回完整主机结果所需的传输；首次任务时间保留在补充表。BFS/SSSP 三个固定源的等权均值仍只算一次观测。',
  '', 'T_first、T_ready 均从原结果读取；80/200 秒一类队列整作业时间不进入此表。详见 all_systems.csv 的所有系统、原始路径和缺项。','']
 for metric,title,name in [('ready','执行时间','strongest_observed_baseline'),('first','补充：首次任务时间','strongest_observed_baseline_first')]:
  for count in (1,8):
   markdown+=['## '+('单卡' if count==1 else '8 卡')+' · '+title,'','| 图 | 算法 | EGGPU (s) | 已测最快对手 | 对手 (s) | 相对速度 | 对手覆盖 |','|---|---|---:|---|---:|---:|---:|']
   for r in comparisons:
    if r['gpu_count']==count:markdown.append('| '+ ' | '.join([r['dataset'],r['algorithm'].upper(),fmt(r['eggpu_'+metric+'_seconds']),r[name] or '—',fmt(r.get('baseline_'+metric+'_seconds')),fmt(r['speedup_'+metric])+'×' if r['speedup_'+metric] is not None else '—',f"{r['available_baselines']}/{r['required_baselines']}"])+' |')
   markdown+=['','相对速度仅针对该计时边界下的已测对手；覆盖未齐时不代表完整对手集合上的领先。','']
 markdown.extend(['## 实际失败分类','', '| 资源 | 图 | 算法 | 系统 | 原因 |','|---|---|---|---|---|'])
 for cell in cells:
  if cell['status']=='FAILED':markdown.append(f"| {cell['gpu_count']} GPU | {cell['dataset']} | {cell['algorithm'].upper()} | {cell['label']} | {cell['failure_class']} |")
 markdown.extend(['', '完整错误、失败原件及同格全部尝试保留在 first_pass.json；失败不补造时间，不算自动胜利。'])
 (out/'FIRST_PASS_CN.md').write_text('\n'.join(markdown)+'\n')
 render_html(report,out)
 return report


def render_html(report,out):
 sections=[]
 for metric,title in [('ready_seconds','Time(s)'),('first_seconds','Supplement: first-task time (s)')]:
  for count in (1,8):
   rows=[r for r in report['cells'] if r['gpu_count']==count];labels=list(dict.fromkeys(r['label'] for r in rows))
   table=['<table><thead><tr><th>Graph</th><th>Algorithm</th>'+''.join('<th>'+html.escape(x)+'</th>' for x in labels)+'</tr></thead><tbody>']
   for graph in GRAPHS:
    for alg in ALGORITHMS:
     table.append('<tr><th>'+graph+'</th><th>'+alg.upper()+'</th>')
     for name in labels:
      r=next((x for x in rows if x['dataset']==graph and x['algorithm']==alg and x['label']==name),None)
      text=fmt(r[metric]) if r and r['status']=='AVAILABLE' else {'FAILED':'FAIL','REVIEW_NEEDED':'CHECK','PENDING':'—'}.get(r['status'],'N/A') if r else 'N/A'
      if r and r['status']=='FAILED':text={'INDEX_LIMIT':'IDX','QUALITY_FAILED':'QF','PREPROCESSING_TIMEOUT':'TO-prep','TIMEOUT':'TO','PROCESS_FAILURE':'FAIL-proc'}.get(r.get('failure_class'),'FAIL')
      tooltip=html.escape((r.get('raw_receipt') or '; '.join(str(x) for x in [*r.get('failure_reasons',[]),*r.get('attempts',[])] if x) or r['status']) if r else 'Not applicable',quote=True)
      table.append('<td title="'+tooltip+'">'+text+'</td>')
     table.append('</tr>')
   table.append('</tbody></table>');sections.append(f'<h2>{count} GPU · {title}</h2>'+''.join(table))
 text='''<!doctype html><meta charset="utf-8"><title>EGGPU first-pass measurements</title><style>body{font:14px system-ui,sans-serif;color:#172033;margin:28px}h1{font-size:24px}h2{font-size:18px;margin-top:32px}p{max-width:1000px;line-height:1.7}table{border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:7px 10px;border-bottom:1px solid #e4e7eb;text-align:right}th{background:#f3f5f7}tr:nth-child(4n+1) td,tr:nth-child(4n+1) th{border-top:2px solid #b7bec9}td:hover{background:#e4eef7}</style><h1>首轮比较 · 每格一次完整观测</h1><p>固定七图与全部适用 baseline。已有完整 target 测量按原身份复用，不重标为 formal；数值使用原始 T_first/T_ready，不使用队列总耗时。— 待完成；IDX 索引限制；QF 输出质量未过；TO-prep 准备超时；FAIL-proc 进程失败；FAIL 其他失败；CHECK 来源或合同需核对；N/A 不适用。鼠标停留显示原始证据路径。</p>'''+''.join(sections)
 (out/'first_pass.html').write_text(text)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--authority',type=Path,default=Path(__file__).resolve().with_name('e1_scope_authority.json'));p.add_argument('--output',type=Path,required=True);p.add_argument('--supplement',type=Path)
 a=p.parse_args();r=build(a.bundle,a.authority,a.output,a.supplement);print(json.dumps(r['counts']))
