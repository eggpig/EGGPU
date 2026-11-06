"""Presentation-only check: direct ready/first fields, CSV, captions and E5 interaction."""
from pathlib import Path
import csv,importlib.util,json,os,tempfile
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ready',HERE/'summarize_state_ready.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SCRATCH=Path(os.environ['EGGPU_READY_TEST_SCRATCH']);SCRATCH.mkdir(parents=True,exist_ok=True)
def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
 p=Path(tmp);bundle=p/'bundle';bundle.mkdir();policy=p/'policy.json';save(policy,{});pb=m.bind(policy)
 freeze=p/'freeze.json';save(freeze,dict(source_files={}))
 runner=p/'runner.json';save(runner,dict(files={}))
 selection=dict(native_dso={},measurement_policy=pb,source_tree_sha256='CPU_FIXTURE',source_freeze=m.bind(freeze),runner_freeze=m.bind(runner))
 save(bundle/'selection.json',selection)
 times={'D0C0':10.,'D0C1':20.,'D1C0':30.,'D1C1':45.};firsts={'D0C0':100.,'D0C1':200.,'D1C0':300.,'D1C1':400.};rows=[]
 for arm in times:
  row=dict(id=arm,experiment='E5',datasets=['IN','UK','OR'],arm=arm,partitions=8,rho=1e-4,beta=.25,old_readers='none',order='interleaved',locality='uniform',priority_stage='paper_main_p8',inputs={});rows.append(row)
  event=dict(kind='query',function='bfs',details=dict(metrics=dict(native_dynamic_patch=dict(per_device_delta={}),resource_management=dict(actions=[]),shared_array_new_bytes=0)))
  check=dict(event=dict(kind='query',function='bfs'),quality=dict(passed=True,exact=True,reference=dict(independent=True)),survives_session_close=True,output=dict(lossless=True,format='npz:values',vector_sha256='fixture'),output_sha256='fixture')
  record=dict(status='PASS_STATE_GPU_QUALIFICATION',phase='target',repeat=0,execution_complete=True,quality_pass=True,datasets=row['datasets'],experiment='E5',arm=arm,inputs={},devices=list(range(8)),configuration={k:row[k] for k in ('old_readers','beta','order')},source_binding=dict(dso={},files={},benchmark={}),continuous_interval=True,sum_of_phases_used_as_total=False,events=[event]*96,checks=[check]*96,workflow_complete_host_seconds=times[arm],first_complete_workflow_seconds=firsts[arm],first_pass=dict(paired_block_sha256='same_block'),measurement_policy=dict(binding=pb,actual=dict(status='PASS_ACTUAL_PR_POLICY')))
  save(bundle/'results'/('target_'+arm)/'receipt.json',record)
 save(bundle/'coverage.json',dict(configurations=rows,stages=['paper_main_p8']))
 output=p/'report';output.mkdir();oldcache=output/'receipt_cache.json';oldcache.write_text('{"historical_first_cache":true}');rawcache=oldcache.read_bytes()
 m.export(bundle,output,False);report=m.read(output/'report.json')
 assert all(r['status']=='COMPLETE_N1' and r['time_seconds']==times[r['arm']] and r['first_complete_workflow_seconds']==firsts[r['arm']] for r in report['rows'])
 assert report['interactions'][0]['interaction_seconds']==5. # First values would give -0, not5.
 csvrows=list(csv.DictReader((output/'all_cells.csv').open()))
 assert all(float(r['time_seconds'])==times[r['arm']] and float(r['first_complete_workflow_seconds'])==firsts[r['arm']] for r in csvrows)
 assert 'first-complete time is supplemental' in (output/'FIGURE_CAPTION.txt').read_text()
 assert (output/'receipt_cache_ready_v1.json').exists() and oldcache.read_bytes()==rawcache
 # Actual historical field values are passed directly, never subtracted.
 path=bundle/'results/target_D0C0/receipt.json';record=m.read(path);record.update(workflow_complete_host_seconds=1251.345050105,first_complete_workflow_seconds=1284.460626153);save(path,record)
 actual=m.assess(path,rows[0],selection,dict(source_files={}),dict(files={}))
 assert actual['time_seconds']==1251.345050105 and actual['first_complete_workflow_seconds']==1284.460626153
receipt=dict(status='PASS_CPU_READY_PRESENTATION',source=m.bind(HERE/'summarize_state_ready.py'),primary='workflow_complete_host_seconds',supplement='first_complete_workflow_seconds',checked=['direct actual1251.345/1284.461 field mapping','all4 E5arms','CSV bothfields','ready interaction5 vsfirst0','ready figurecaption','oldfirst cache not reused'],plot_series='All existing E3/E4/E5 draw paths read the checked primary time_seconds; no measurement code changes',gpu_executed=False)
save(HERE/'cpu_receipt.json',receipt);print(json.dumps(receipt))
