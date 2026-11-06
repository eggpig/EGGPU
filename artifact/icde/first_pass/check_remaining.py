"""CPU regression using captured real JSON; metadata negative fixtures are never performance data."""
from pathlib import Path
import argparse,copy,datetime,gzip,hashlib,importlib.util,json

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def run(a):
 f=a.runtime.resolve();m=module('remaining_check',f/'remaining_results.py');c=m.core();out=a.scratch.resolve();out.mkdir(parents=True,exist_ok=True);checks=[]
 def passed(name):checks.append(name)
 def rejects(name,call):
  try:call()
  except (ValueError,KeyError,TypeError):passed(name);return
  raise AssertionError('Accepted invalid metadata: '+name)
 x=json.loads(a.snapshot.read_text());x['observed_utc']=datetime.datetime.fromtimestamp(a.snapshot.stat().st_mtime,datetime.timezone.utc).isoformat();x['selected_native_protocol']=m.PROTOCOL;x['records']=[];x['plans']=[];x['missing']=[];x['queues']=[v['value'] for p,v in x['objects'].items() if p.endswith('/state.json')]
 bundle_path=out/'actual_snapshot.json';bundle_path.write_text(json.dumps(x));r=m.build(bundle_path,out/'remaining')
 assert r['counts']=={'E2':{'PENDING':80},'E4S':{'AVAILABLE':1,'PENDING':35},'E6':{'PENDING':45}};passed('actual_source71_e4s_only_one_available_no_invented_E2_E6')
 assert all(row['first_seconds'] is None and row['ready_seconds'] is None for row in r['cells'] if row['status']!='AVAILABLE');passed('all_160_unavailable_cells_have_null_times')
 reuse=[row for row in r['cells'] if row.get('p8_reused')];assert len(reuse)==20 and all('/unified_main_continuation/' in row['raw_receipt'] for row in reuse);passed('all_20_p8_receipts_follow_explicit_continuation_mapping')
 b=c.Bundle(bundle_path);cells,plans=m.planned_cells(b);cell=next(row for row in cells if row['raw_receipt'] in b.objects and row['experiment']=='E4S');raw=b.get(cell['raw_receipt']);epoch=b.get(raw['epoch']);domain=b.domain('US')
 bad=copy.deepcopy(raw);bad['checks'][-1]['quality']['exact']=False;rejects('last_E4S_full_output_mismatch_rejected',lambda:m.e4s_check(b,bad,cell['job'],cell,domain))
 bad=copy.deepcopy(raw);bad['first_pass_source']['source_tree_sha256']='0'*64;rejects('mixed_own_source_rejected',lambda:m.source_check(b,bad,epoch,plans['freeze'],plans['selection']))
 b2=c.Bundle(bundle_path);main=b2.get(m.PATHS['main']);j=next(j for j in main['jobs'] if j.get('phase')=='target' and '--count' in j['command']);main['jobs'].append(copy.deepcopy(j));rejects('ambiguous_p8_mapping_rejected',lambda:m.planned_cells(b2))
 compat=c.Bundle(a.compatibility);compat_count=0
 for path,obj in compat.objects.items():
  raw=obj['value']
  if not path.endswith('/result.json') or raw.get('status') not in ('PASS_QUALIFICATION','PASS_E4S_QUALIFICATION'):continue
  if raw.get('schema')=='eggpu.e6.job/coverage2x2-v2':
   cell={k:raw[k] for k in ('dataset','system','ablate','coverage_backend','gpu_count')};domain={'sha256':raw['input_sha256'],'identity':{'n':40}};m.e6_check(compat,raw,compat.get(raw['epoch']),cell,domain);compat_count+=1
   child=compat.get(raw['child']);saved=copy.deepcopy(child['calls']['ready']['trace']['stages'][0]['metrics']['devices_used']);child['calls']['ready']['trace']['stages'][0]['metrics']['devices_used']=[99]
   rejects('E6_actual_devices_rejected_'+str(compat_count),lambda:m.e6_check(compat,raw,compat.get(raw['epoch']),cell,domain));child['calls']['ready']['trace']['stages'][0]['metrics']['devices_used']=saved
  elif raw.get('schema')=='e4s.complete.batch.v1' and raw['variant']!='eggpu':
   cell={'system':raw['variant'],'batch':raw['batch']};job={'target_contract':{k:raw[k] for k in ('input_sha256','directed','source_sequence')}};domain={'directed':raw['directed'],'identity':{'n':raw['checks'][0]['values']}};m.e4s_check(compat,raw,job,cell,domain);compat_count+=1
 assert compat_count==7;passed('seven_actual_historical_tiny_schema_checks_not_admitted_as_target_results')
 live=c.Bundle(a.current_bundle);authority=f/'unified_export_runtime_protocol_v2/e1_scope_authority.json'
 if not authority.exists():authority=f/'e1_scope_authority.json'
 baseline=c.build(a.current_bundle,authority,out/'actual_E1');assert baseline['counts']==dict(required_cells=329,available_cells=217,pending_cells=92,deferred_preparation_cells=8,input_layout_unsupported_cells=0,failed_cells=20,review_cells=0);passed('actual_online_28_historical_reviews_removed_all_20_failures_preserved')
 domains={g:live.domain(g) for g in c.GRAPHS};freeze=live.get(json.loads(authority.read_text())['eggpu_source_freeze'])
 entry=next(e for e in live.data['records'] if (lambda r:r.get('system')=='grapin' and r.get('phase')=='target' and r.get('status')=='PASS_QUALIFICATION')(live.get(e['path'])))
 original=copy.deepcopy(live.get(entry['path']));old=c.observation(live,entry,freeze,domains)
 def attach(path,value):
  raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode();link=dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());live.objects[path]={'binding':link,'value':value};return link
 original_link=live.objects[entry['path']]['binding'];usage=attach(live.root+'/metadata_fixture/cache_usage.json',{'status':'METADATA_FIXTURE_ONLY'})
 derivative={**original,'measurement_protocol':'author-ready-native-call2-cached-input-first-supplement-v1','cached_input_first_supplement':True,'ready_measurement_unchanged':True,'canonical_csr_cold_first_comparable':False,'original_author_result':original_link,'text_cache_usage':usage};link=attach(live.root+'/metadata_fixture/result_cached_input.json',derivative)
 new=c.observation(live,{**entry,'path':link['path']},freeze,domains);assert new['first_seconds'] is None and new['cached_input_first_seconds']==old['first_seconds'] and new['ready_seconds']==old['ready_seconds'];passed('cached_input_metadata_keeps_ready_exact_and_excludes_cold_first')
 derivative['T_ready_seconds']=-1;rejects('cached_derivative_cannot_change_original_fields',lambda:c.observation(live,{**entry,'path':link['path']},freeze,domains))
 deferred=c.preparation_deferrals(live,domains);oldplan=live.get(live.data['scheduler_deferrals']['plan']);j=oldplan['jobs'][0];cmd=j['command'];worker=next(v for v in j['bindings'] if v['path']==cmd[cmd.index('--worker')+1]);detail=dict(status='UNSUPPORTED_FROZEN_INT32_LAYOUT',error='INDEX_UNSUPPORTED: weak projection exceeds int32',original_graph_layout_executed=True,gpu_executed=False)
 prep=dict(status='PASS_CPU_PREPARED_TEXT_INPUTS',gpu_executed=False,input_sha256=domains['TW']['sha256'],canonical_receipt=live.objects[live.root+'/inputs/canonical/TW/receipt.json']['binding'],layouts={'wcc':detail},original_worker=worker);plink=attach(live.root+'/metadata_fixture/preparation.json',prep)
 unavailable=[dict(id=d['job_id'],system=d['system'],algorithm='wcc',dataset='TW',status='INPUT_LAYOUT_UNSUPPORTED_CPU',evidence=plink,detail=detail,gpu_measurement=False,original_attempt=d['receipt']) for d in deferred if d['algorithm']=='wcc']
 plan=dict(schema='eggpu.author.cached-input-first-pass.v1',preparation=plink,source_deferred=live.data['scheduler_deferrals']['plan'],unavailable=unavailable);live.data['author_text_input_plan']=attach(live.root+'/metadata_fixture/plan.json',plan)
 limits=c.input_layout_limits(live,domains,deferred);assert len(limits)==2;passed('two_declared_TW_WCC_CPU_limits_schema_accepted_without_timings')
 prep['layouts']['wcc']['original_graph_layout_executed']=False;rejects('unexecuted_CPU_layout_cannot_close_deferred_cells',lambda:c.input_layout_limits(live,domains,deferred))
 receipt=dict(status='PASS_CPU_REGRESSION',checks=checks,check_count=len(checks),actual_remaining_counts=r['counts'],actual_E1_counts=baseline['counts'],historical_tiny_schema_only=7,metadata_fixture_timings_created=False,fixtures_not_published=True,gpu_executed=False,array_payloads_read=False)
 (out/'validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();
 for k in ('runtime','snapshot','current-bundle','compatibility','scratch'):p.add_argument('--'+k,type=Path,required=True)
 run(p.parse_args())
