"""Fail-closed, CPU-only complete-group admission; this tool never starts workers.

Inspect mode only reports blockers. Admit mode derives a separate queue plan
only after every fixed qualification and CPU-reference preparation gate passes.
Frozen worker proof readers are reused by extracting only their pure validator
functions; their entrypoints/configure methods are never executed.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,collections,copy,fcntl,hashlib,importlib.util,json,math,os,tempfile


def require(condition,message):
 if not condition:raise ValueError(message)
def load(path):return json.loads(Path(path).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def options(argv):
 result={};i=0
 while i<len(argv):
  arg=argv[i]
  if arg.startswith('--'):
   if '=' in arg:key,value=arg[2:].split('=',1);result[key]=value
   elif i+1<len(argv) and not argv[i+1].startswith('--'):result[arg[2:]]=argv[i+1];i+=1
   else:result[arg[2:]]=True
  i+=1
 return result

class Binder:
 """Full SHA validation, bounded RAM; stat timestamps never substitute for bytes."""
 def __init__(self):self.cache={};self.seen={};self.hashed_bytes=0
 def bind(self,path):
  p=Path(path).resolve();before=p.stat();key=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
  after=p.stat();require(key==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'File changed while hashing: '+str(p));self.hashed_bytes+=before.st_size
  value=dict(path=str(p),bytes=before.st_size,sha256=h.hexdigest());self.seen[str(p)]=value;return value
 def checked(self,item):
  observed=self.bind(item['path']);require(all(observed[k]==item[k] for k in ['path','bytes','sha256']),'Bound file changed: '+item['path']);return Path(item['path'])
 def content(self,path):return {k:v for k,v in self.bind(path).items() if k!='path'}


def extract(path,names,namespace):
 """Compile only named definitions from an already SHA-bound frozen reader."""
 tree=ast.parse(Path(path).read_text(),filename=str(path));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
 require({n.name for n in nodes}==set(names),'Missing frozen proof functions')
 code=compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec');exec(code,namespace);return namespace


def workers(spec,binder):
 for item in spec.values():binder.checked(item)
 common=dict(Path=Path,json=json,binding=binder.bind,digest_json=digest,TINY_DATASETS=('TINY_D','TINY_U'))
 extract(spec['sg']['path'],['validate_proof'],common);shared=SimpleNamespace(**common)
 profile=dict(NAME='static_full_profile_v1');extract(spec['native_profile']['path'],['require_fields','observe'],profile)
 native_common=dict(Path=Path,json=json,binding=binder.bind,digest_json=digest,TINY_DATASETS=('TINY_D','TINY_U'));extract(spec['native_sg']['path'],['validate_proof'],native_common)
 native=dict(Path=Path,json=json,shared=SimpleNamespace(**native_common),native_profile=SimpleNamespace(**profile));extract(spec['native']['path'],['validate_proof'],native)
 sequence=dict(Path=Path,json=json,shared=shared,PROTOCOL='e1-continuous-16-public-csr-v1');extract(spec['sequence']['path'],['validate_proof'],sequence)
 e4s=dict(Path=Path,json=json,common=shared);extract(spec['e4s']['path'],['proof'],e4s)
 return dict(sg=common['validate_proof'],native=native['validate_proof'],sequence=sequence['validate_proof'],e4s=e4s['proof'],native_observe=profile['observe'])


def canonical(directory,binder):
 directory=Path(directory);receipt=load(directory/'receipt.json');require(receipt['status']=='PASS_CANONICAL_INPUT','Canonical input not qualified')
 for role,name in [('canonical','canonical.json'),('sources','sources.json')]:
  actual=binder.content(directory/name);require(actual=={k:receipt['files'][role][k] for k in ['bytes','sha256']},'Canonical metadata changed')
 meta=load(directory/'canonical.json');sources=load(directory/'sources.json')['primary_sources'];require(len(sources)==len(set(sources))==3,'Frozen primary sources missing')
 identity=dict(receipt=binder.content(directory/'receipt.json'),files={r:{k:i[k] for k in ['bytes','sha256']} for r,i in receipt['files'].items()},sources=sources,n=meta['num_nodes'],m=meta['num_entries'],directed=meta['directed'])
 return dict(n=meta['num_nodes'],m=meta['num_entries'],directed=meta['directed'],sources=sources,input_sha256=digest(identity))


def quality(q,function):
 if not all(q.get(k) is True for k in ['pass','complete','owning','immutable']):return False
 if q.get('dtype') not in (['float64','int64'] if function=='wcc' else ['float64']):return False
 if function=='pr':return all(isinstance(q.get(k),(int,float)) and math.isfinite(q[k]) and 0<=q[k]<=v for k,v in [('residual_l1',1e-6),('mass_error',1e-6),('reference_l1',1e-5)])
 if function=='wcc':return q.get('partition_equivalent') is True
 return q.get('exact_reference_match') is True and q.get('unreachable_match') is True


def current_epoch(proof,job,manifest,freeze,binder):
 item=proof['epoch'];ep=load(binder.checked(item));require(digest(ep)==proof['epoch_sha256'],'Epoch digest changed');epoch_bound_artifacts(ep,binder)
 opt=options(job['command']);system=opt.get('system',opt.get('variant','eggpu'))
 if system=='eggpu':
  sources=ep.get('release_sources',{});require(bool(sources),'Missing current source observation')
  for rel,seen in sources.items():require(rel in freeze['source_files'] and seen==freeze['source_files'][rel],'Old/different observed source: '+rel)
  for rel in ['easygraph/gpu/session.py','easygraph/gpu/_weight_admission.py','cpp_easygraph/classes/graph_convert.cpp']:require(rel in sources,'Required source absent from epoch: '+rel)
  native=ep.get('native_dso',ep.get('files',{}).get('native_dso',{}));require(native.get('sha256')==manifest['native_dso']['sha256'],'Qualification used different native DSO')
  require(ep.get('budget_mib',22000)==int(opt.get('budget-mib',22000)),'Budget differs from frozen command')
 return ep


def call_artifacts(proof,family,graph,binder):
 if family=='e4s':
  checks=proof.get('checks',[]);sequence=proof['source_sequence'];require(len(checks)==proof['batch']==len(sequence),'Incomplete E4S full-vector audits')
  for index,(row,source) in enumerate(zip(checks,sequence)):
   require(row.get('sequence')==index and row.get('source')==source and row.get('values')==graph['n'],'E4S complete-output source/shape differs')
   q=row.get('quality',{});require(q.get('passed') is True and q.get('exact') is True and q.get('vertices')==graph['n'],'E4S exact complete quality failure')
   require(isinstance(row.get('output_sha256'),str) and len(row['output_sha256'])==64 and all(c in '0123456789abcdef' for c in row['output_sha256']),'Missing E4S complete-vector SHA')
  m=proof['measurements'];require(isinstance(m.get('first_complete_seconds'),(int,float)) and 0<m['first_complete_seconds']<=3600,'E4S first complete interval absent or over limit')
  if proof['variant']=='eggpu':
   require(m.get('all_outputs_retained_through_close') is True and all(x.get('survives_session_close') is True for x in checks),'EGGPU batch output ownership/lifetime evidence missing')
   requests=m.get('requests',[]);require([x['source'] for x in requests]==sequence and [x['sequence'] for x in requests]==list(range(len(sequence))),'EGGPU batch request sequence incomplete')
   require(abs(m['first_complete_seconds']-m['adapter_preparation_seconds']-m['query_batch_seconds'])<1e-5,'EGGPU continuous perf_counter endpoints inconsistent')
  else:
   interval=m.get('continuous_interval',{});require(interval.get('phase_durations_summed') is False,'E4S author clock not continuous')
   require(isinstance(interval.get('start_ns'),int) and interval['end_ns']>interval['start_ns'],'E4S author completion timestamps incomplete')
   if interval.get('launch'):binder.checked(interval['launch'])
  return dict(output_proof='All complete vectors audited and hashed; frozen E4S protocol does not archive batch vectors',vectors=len(checks),archive_rehashed=False)
 if family=='sequence':
  child=load(binder.checked(proof['workflow']));calls=child['calls'];require(len(calls)==16 and child.get('continuous_interval') is True and child.get('sum_of_query_times_used_as_total') is False and child.get('execution_complete') is True,'Incomplete continuous 16-query workflow')
  for row in calls:require(quality(row['quality'],row['algorithm']),'Sequence common quality failure');binder.checked(row['output'])
  return dict(output_proof='16 complete owned host vectors, full archived SHA checked',vectors=16,archive_rehashed=True)
 function=proof['algorithm'];sources=proof['sources'];expected=graph['sources'] if function in ['bfs','sssp'] else [None]
 require([x['source'] for x in sources]==expected,'Qualification does not cover fixed source sequence')
 for source in sources:
  child=load(binder.checked(source['result']));require(child.get('input_sha256')==proof['input_sha256'],'Child canonical input differs')
  for name in ['first','warmup','ready']:
   c=child['calls'][name];require(quality(c['quality'],function),'Common quality/ownership contract failure');binder.checked(c['output'])
  for name in ['first','ready']:
   sec=child['calls'][name]['seconds'];require(isinstance(sec,(float,int)) and math.isfinite(sec) and 0<sec<=1800,'Complete-call time missing or over frozen timeout')
 return dict(output_proof='All fixed-source first/warmup/ready complete outputs, full archived SHA checked',vectors=len(sources)*3,archive_rehashed=True)


def numeric_artifacts(proof,job,binder,observe):
 opt=options(job['command']);count=int(opt['count']);require(proof.get('count')==count and proof.get('gpu_executed') is True and proof.get('performance_evidence') is False,'Wrong numeric-domain execution')
 require(proof.get('all_full_vectors_exact') is True and proof.get('bitwise_outputs_survive_later_queries_and_close') is True,'Numeric-domain complete ownership proof missing')
 expected=['zero_and_255','all_isolates']+(['single_vertex_no_edges'] if count==1 else []);cases=proof.get('cases',[]);require([x['name'] for x in cases]==expected,'Numeric-domain cases incomplete')
 for row in cases:
  require(row['status']=='PASS_STATIC_NUMERIC_DOMAIN' and row['n']==(1 if row['name']=='single_vertex_no_edges' else 16),'Wrong numeric-domain shape/status');binder.checked(row['manifest']);calls=row['calls'];require([c['repeat'] for c in calls]==[0,1,2],'Numeric-domain repeats incomplete')
  for call in calls:require(call.get('quality_pass') is True,'Numeric-domain quality failed');binder.checked(call['output']);observe(call['metrics'],'sssp',count)
 return dict(output_proof='Numeric zero/255/isolates/single-vertex full exact vectors and cross-close bitwise lifetime',vectors=len(cases)*3,archive_rehashed=True)


def epoch_bound_artifacts(value,binder):
 if isinstance(value,dict):
  if set(['path','bytes','sha256'])<=set(value):binder.checked(value)
  else:
   for item in value.values():epoch_bound_artifacts(item,binder)
 elif isinstance(value,list):
  for item in value:epoch_bound_artifacts(item,binder)


def normalized_command(job):
 opt=options(job['command']);return {k:v for k,v in opt.items() if k not in ['phase','repeat','output','target-proof']}


def structural_gate(formal,qualification_plans):
 jobs=formal['jobs'];require(bool(jobs),'Empty formal plan forbidden');ids=[j['id'] for j in jobs];require(len(ids)==len(set(ids)),'Duplicate formal job id')
 qualifications={}
 for p in qualification_plans:
  for job in p['jobs']:
   require(job['id'] not in qualifications,'Duplicate qualification id');qualifications[job['id']]=job
 targets={j['receipt']:j for j in qualifications.values()};groups=collections.defaultdict(list);mapping={}
 for job in jobs:
  opt=options(job['command']);require(opt.get('phase')=='formal' and job.get('phase')=='formal','Non-formal job in pending plan')
  require(opt.get('tiny-proof') and opt.get('target-proof'),'Missing worker proof flags')
  target=targets.get(opt['target-proof']);require(target is not None,'Target proof outside exact qualification plans')
  require(opt['tiny-proof'] in targets,'Tiny proof outside exact qualification plans')
  require(normalized_command(job)==normalized_command(target),'Formal computation/configuration differs from qualified target')
  require(job.get('environment',{})==target.get('environment',{}),'Formal job environment differs from target')
  require(job['command'][0]==target['command'][0] and next(x for x in job['command'] if x.endswith('.py'))==next(x for x in target['command'] if x.endswith('.py')),'Python executable or worker differs')
  require({x for x in job.get('depends_on',[]) if x not in ids}<={target['id']},'Unknown external controller dependency')
  require(target['id'] in job.get('depends_on',[]),'Missing intended target dependency')
  groups[opt['target-proof']].append(int(opt['repeat']));mapping[job['id']]=target
 for target,repeats in groups.items():require(sorted(repeats)==list(range(10)),'Every exact cell requires all ten repeats 0..9: '+target)
 require(set(groups)=={j['receipt'] for j in qualifications.values() if options(j['command']).get('phase')=='target'},'Formal matrix must cover every exact qualification target cell')
 return qualifications,mapping


def cpu_gate(manifest,binder):
 spec=manifest['cpu_gate'];config=load(binder.checked(spec['external_watcher_config']));watcher_path=binder.checked(config['watcher'])
 namespace={'Path':Path,'json':json,'require':require,'checked':binder.checked,'load':load,'bind':binder.bind}
 extract(watcher_path,['references_gate'],namespace);references=namespace['references_gate'](config['references'])
 require(references.get('status')=='PASS_ALL_REFERENCE_PREPARATION_COMPLETE','CPU references not fully prepared: '+str(references.get('status')))
 receipts=[]
 for item in spec['receipts']:
  r=load(binder.checked(item['binding']));require(r.get('status')==item['status'] and r.get('gpu_executed') is False,'Required CPU preparation/check failed');receipts.append(item['binding'])
 return dict(status='PASS_CPU_PREPARATION_COMPLETE',references=references,receipts=receipts)


def recovery_modules(spec,binder):
 for key in ['helper','wrapper','exporter','controller','supervisor','decorator']:binder.checked(spec[key])
 module_spec=importlib.util.spec_from_file_location('_bound_e4s_archive',spec['helper']['path']);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
 namespace=dict(Path=Path,copy=copy,hashlib=hashlib,json=json,archive=module);extract(spec['decorator']['path'],['command_digest','decorate'],namespace)
 return module,namespace['decorate']


def inspect(manifest_path,group,mode='inspect',output=None):
 if mode=='admit':
  manifest=load(manifest_path);lock_path=Path(manifest.get('controller_lock',str(Path(manifest_path).parent/'root_controller.lock')))
  with lock_path.open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   return _inspect_locked(manifest_path,group,mode,output)
 return _inspect_locked(manifest_path,group,mode,output)

def _inspect_locked(manifest_path,group,mode='inspect',output=None):
 binder=Binder();manifest=load(manifest_path);require(manifest['schema']=='eggpu.d84.formal.admission.v1','Unknown admission schema');spec=manifest['groups'][group]
 formal=load(binder.checked(spec['pending']));plans=[load(binder.checked(p)) for p in spec['qualifications']];qualifications,mapping=structural_gate(formal,plans)
 require(formal['source_tree_sha256']==manifest['source_sha256'],'Pending source epoch differs');require(len(formal['jobs'])==spec['expected_formal_jobs'],'Frozen formal matrix cardinality differs')
 blockers=[];states=[]
 if group=='e4s' and manifest.get('policy',{}).get('e4s_formal_archive_repair_pending',False):blockers.append(dict(stage='e4s_output_archive',reason='HOLD: mandatory byte-identical recoverable-output sidecar not yet integrated; SHA-only qualification is not formal archive compliance'))
 for job in qualifications.values():
  p=Path(job['receipt']);observed=load(p).get('status') if p.exists() else 'ABSENT';require(len(job['expected_statuses'])==1,'Ambiguous expected qualification status');expected=job['expected_statuses'][0];require(expected in ['PASS_QUALIFICATION','PASS_E4S_QUALIFICATION','PASS_STATIC_NATIVE_NUMERIC_QUALIFICATION'],'Unsupported qualification schema');states.append(dict(id=job['id'],status=observed,expected=expected,path=str(p)))
  if observed!=expected:blockers.append(dict(job=job['id'],status=observed,reason='Entire frozen qualification group is required; no automatic PASS-subset selection'))
  if group=='e4s' and not manifest.get('policy',{}).get('e4s_formal_archive_repair_pending',False):
   sidecar=p.parent/'recoverable_outputs.json'
   if not sidecar.exists() or load(sidecar).get('status')!='PASS_BYTE_IDENTICAL_RECOVERABLE_E4S_OUTPUTS':blockers.append(dict(job=job['id'],stage='recoverable_outputs',reason='Exact qualification recovery sidecar is required before formal admission'))
 cpu=None
 try:cpu=cpu_gate(manifest,binder)
 except (ValueError,OSError,KeyError,json.JSONDecodeError) as error:blockers.append(dict(stage='cpu_preparation',reason=str(error)))
 report=dict(schema='eggpu.d84.formal.admission.receipt.v1',group=group,mode=mode,status='NOT_ADMITTED',formal_jobs=len(formal['jobs']),qualification_jobs=len(qualifications),repetitions_per_cell=10,blockers=blockers,qualification_states=states,cpu_preparation=cpu,source_sha256=manifest['source_sha256'],gpu_executed=False,workers_launched=0,manifest=binder.bind(manifest_path),pending=spec['pending'],intended_formal_queue=spec.get('queue'),intended_controller_log=spec.get('controller_log'))
 if blockers or mode=='inspect':
  report['status']='NOT_ADMITTED_BLOCKED' if blockers else 'READY_FOR_FULL_PROOF_AUDIT_NOT_ADMITTED';return report
 require(mode=='admit' and output is not None,'Explicit output directory required; admission never starts workers')
 out=Path(output);require('admitted_output' not in spec or str(out)==spec['admitted_output'],'Use fixed manifest admission output path');require(not out.exists(),'Refusing to overwrite admission artifact')
 freeze=load(binder.checked(manifest['source_freeze']));require(freeze['source_tree_sha256']==manifest['source_sha256'],'Wrong source freeze');binder.checked(manifest['native_dso'])
 for rel,b in freeze['source_files'].items():require(binder.content(Path(manifest['source_root'])/rel)==b,'Current source changed: '+rel)
 readers=workers(manifest['proof_readers'],binder);proofs={};audit={};epochs={};graphs={}
 recovery,decorate=(recovery_modules(manifest['archive_contract'],binder) if group=='e4s' else (None,None))
 for plan in plans:require(plan.get('environment',{})==formal.get('environment',{}),'Qualification/formal plan environment differs')
 for job in qualifications.values():
  for item in job.get('bindings',[]):binder.checked(item)
  opt=options(job['command']);proof=load(job['receipt']);ep=current_epoch(proof,job,manifest,freeze,binder);epochs[job['receipt']]=ep
  expected_protocol={'sg':'e1-host-csr-complete-v1','native':'native-host-csr-complete-static-full-profile-v1','sequence':'e1-continuous-16-public-csr-v1','e4s':'e4s.epoch.v1'}[group]
  require(ep.get('schema' if group=='e4s' else 'protocol')==expected_protocol,'Qualification timing/protocol epoch differs')
  if proof['status']=='PASS_STATIC_NATIVE_NUMERIC_QUALIFICATION':
   require(group=='native','Numeric-domain receipt outside native gate');audit[job['receipt']]=numeric_artifacts(proof,job,binder,readers['native_observe']);proofs[job['receipt']]=binder.bind(job['receipt']);continue
  graph=canonical(Path(opt['input-root'])/'canonical'/opt['dataset'],binder);graphs[job['receipt']]=graph
  args=SimpleNamespace(system=opt.get('system','eggpu'),algorithm=opt.get('algorithm'),count=int(opt.get('count',1)),sssp_encoding=opt.get('sssp-encoding','u8_exact_u32_exact'),variant=opt.get('variant'),batch=int(opt.get('batch',0)))
  phase=opt['phase'];require(proof.get('phase')==phase and proof.get('dataset')==opt['dataset'],'Proof job identity differs')
  if group=='sg':readers[group](job['receipt'],kind=phase,system=args.system,algorithm=args.algorithm,epoch_sha=proof['epoch_sha256'],graph=graph)
  elif group=='e4s':readers[group](Path(job['receipt']),phase,args,proof['epoch_sha256'],graph,proof['source_sequence'])
  else:readers[group](job['receipt'],phase,args,proof['epoch_sha256'],graph)
  audit[job['receipt']]=call_artifacts(proof,group,graph,binder);proofs[job['receipt']]=binder.bind(job['receipt'])
  if group=='e4s':
   sidecar=Path(job['receipt']).parent/'recoverable_outputs.json';audit[job['receipt']]['recoverable_outputs']=recovery.validate_sidecar(job['receipt'],sidecar,manifest['archive_contract']['helper'])
   pinned=load(sidecar);binder.seen[str(sidecar)]=recovery.binding(sidecar)
   for vector in pinned['vectors']:
    for kind in ['object','reference_metadata']:binder.seen[vector[kind]['path']]=vector[kind]
 for job in formal['jobs']:
  for item in job.get('bindings',[]):binder.checked(item)
  opt=options(job['command']);tiny=load(opt['tiny-proof']);target=load(opt['target-proof'])
  require(tiny['epoch_sha256']==target['epoch_sha256'],'Tiny and target are not the same exact epoch')
  require(tiny['directed']==target['directed'],'Tiny does not cover target directedness')
  require(target.get('tiny_proof',target.get('qualification',{}).get('tiny'))==proofs[opt['tiny-proof']],'Target did not bind exact intended tiny proof')
  if group=='e4s':require(target['source_sequence']==job['target_contract']['source_sequence'] and target['input_sha256']==job['target_contract']['input_sha256'],'E4S target source-sequence contract differs')
 # Bind all exact proof, epoch, full-output/manifest and CPU evidence. Worker
 # validators still recheck their own proofs when the root later starts jobs.
 proof_bindings=list(binder.seen.values());derived=copy.deepcopy(formal);formal_ids={j['id'] for j in derived['jobs']}
 out.mkdir(parents=True,exist_ok=False)
 gate_path=out/'gate.json';gate_path.write_text(json.dumps(dict(status='PASS_FULL_GROUP_PROOF_AND_CPU_AUDIT',manifest=binder.bind(manifest_path),pending=spec['pending'],source_sha256=manifest['source_sha256'],proof_bindings=proof_bindings,qualification_audits=audit,cpu_preparation=cpu,workers_launched=0),indent=2)+'\n');gate_binding=binder.bind(gate_path)
 for job in derived['jobs']:
  job['external_qualification_dependencies']=job.get('depends_on',[]);job['depends_on']=[x for x in job.get('depends_on',[]) if x in formal_ids]
  opt=options(job['command']);relevant=[proofs[opt['tiny-proof']],proofs[opt['target-proof']],gate_binding]
  for key in ['tiny-proof','target-proof']:
   qualified=load(opt[key]);relevant.append(qualified['epoch'])
   if group=='sequence':relevant.append(qualified['workflow'])
   elif group!='e4s':relevant.extend(x['result'] for x in qualified['sources'])
  combined={i['path']:i for i in job.get('bindings',[])};combined.update({i['path']:i for i in relevant});job['bindings']=list(combined.values())
 derived['admission']='PASS_COMPLETE_FIXED_GROUP_QUALIFICATION_AND_CPU_PREPARATION';derived['admission_helper']=binder.bind(__file__);derived['admission_manifest']=binder.bind(manifest_path);derived['qualification_audits']=audit;derived['cpu_preparation_gate']=cpu
 if group=='e4s':derived=decorate(derived,manifest['archive_contract'],out)
 # No worker command, repetitions, resource guard, maintenance, or proof flag changes.
 require([j['command'] for j in derived['jobs']]==[j['command'] for j in formal['jobs']],'Unexpected worker mutation')
 p=out/'plan.json';p.write_text(json.dumps(derived,indent=2)+'\n');report.update(status='PASS_ADMITTED_PLAN_ONLY_NOT_LAUNCHED',plan=binder.bind(p),full_proof_audits=audit,bindings=proof_bindings,hashed_bytes=binder.hashed_bytes,selection_policy='Whole requested group; zero performance-based exclusions; all ten repeats retained');(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--group',choices=['sg','native','sequence','e4s'],required=True);p.add_argument('--mode',choices=['inspect','admit'],default='inspect');p.add_argument('--output',type=Path);a=p.parse_args();r=inspect(a.manifest,a.group,a.mode,a.output);print(json.dumps(r,indent=2));raise SystemExit(0 if r['status'] in ['READY_FOR_FULL_PROOF_AUDIT_NOT_ADMITTED','PASS_ADMITTED_PLAN_ONLY_NOT_LAUNCHED'] else 2)
