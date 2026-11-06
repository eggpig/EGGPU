"""Read-only progress export. No GPU, input arrays, qualification, or installation."""
from pathlib import Path
import argparse,fcntl,gzip,hashlib,importlib.util,json,os,shutil,subprocess,sys,time,traceback

HERE=Path(__file__).resolve().parent
QUEUE_END={'PASS_REQUESTED_JOBS','COMPLETE_WITH_FAILURES','COMPLETE','COMPLETED','CANCELLED','STOPPED'}
DRIVER_END={'FIRST_PASS_SCHEDULED_STAGES_FINISHED','STOPPED_REQUIRES_INSPECTION'}
QUEUE_PATHS=('core/queue','author_parallel/lane0_queue','author_parallel/lane1_queue',
 'author_parallel/serial_tail_queue','successors/missing_queue','successors/scaling_queue',
 'successors/e4s_queue','successors/e6_queue','first_page/queue',
 'unified_main_continuation/queue','unified_scaling_continuation/queue','unified_mechanism_application/queue')


def save(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)


def read(path):
 try:
  # Queue metadata only; never enumerate or read graph/output arrays.
  if path.stat().st_size>32*1024*1024:raise ValueError('Queue/manifest JSON exceeds 32 MiB: '+str(path))
  return json.loads(path.read_text())
 except FileNotFoundError:return None


def terminal(status):
 return status in QUEUE_END or str(status).startswith(('FAILED_','STOPPED_'))


def discover(campaign):
 root=campaign/'first_pass_20261106';queues={root/x/'state.json' for x in QUEUE_PATHS}
 manifests={root/'author_parallel/manifest.json'}
 # Bounded to this run directory and its immediate children, never the campaign.
 manifests.update(root.glob('*manifest.json'));manifests.update(root.glob('*/manifest.json'))
 manifests.update(root.glob('*/stage_manifest.json'))
 driver=read(root/'successors/state.json') or {}
 def add(value,base):
  if isinstance(value,dict):value=value.get('path')
  if not isinstance(value,str):return
  path=Path(value);path=path if path.is_absolute() else base/path
  if path.name!='state.json':path=path/'state.json'
  path=path.resolve()
  if path.is_relative_to(campaign):queues.add(path)
 def walk(value,base):
  if isinstance(value,dict):
   for key,child in value.items():
    if key in ('queue','queue_state','queue_path'):add(child,base)
    elif key=='queue_states' and isinstance(child,list):
     for item in child:add(item,base)
    else:walk(child,base)
  elif isinstance(value,list):
   for child in value:walk(child,base)
 for path in sorted(manifests):
  if 'preview' in path.parent.name.lower():continue
  item=read(path)
  if item:walk(item,path.parent)
 walk(driver,root/'successors')
 return sorted(queues),driver


def completed_targets(item):
 # Use only queue metadata. Resumed queues can repeat an inherited completed
 # prefix, so the signature below omits the queue path and deduplicates rows.
 jobs={job.get('id'):job for job in item.get('jobs',[])};result=[]
 for row in item.get('completed',[]):
  job=jobs.get(row.get('id'),{});phase=row.get('phase')
  if phase in (None,'unspecified'):phase=item.get('job_phases',{}).get(row.get('id'),job.get('phase'))
  if phase is None:
   command=job.get('command',[])
   for index,arg in enumerate(command):
    if arg=='--phase' and index+1<len(command):phase=command[index+1];break
    if arg.startswith('--phase='):phase=arg.split('=',1)[1];break
  if phase not in ('target','formal'):continue
  receipt=row.get('receipt');receipt=receipt if isinstance(receipt,dict) else {}
  result.append(dict(id=row.get('id'),phase=phase,outcome=row.get('outcome'),status=row.get('status'),
   receipt={key:receipt[key] for key in ('path','sha256','bytes') if key in receipt},
   expected_receipt=job.get('receipt')))
 return result


def needs_export(progress,previous):
 return (previous.get('last_export_completed',-1)<0
  or progress['target_completed_signature']!=previous.get('last_export_target_completed_signature')
  or progress['remaining_completed']!=previous.get('last_export_remaining_completed',{})
  or progress['stage']!=previous.get('last_export_stage')
  or progress['terminal_queues']!=previous.get('last_export_terminal_queues',[])
  or (progress['finished'] and not previous.get('last_export_finished',False))
  or (progress['stopped'] and not previous.get('last_export_stopped',False)))


def snapshot(campaign,prior_stage):
 paths,driver=discover(campaign);rows=[];targets=set()
 for path in paths:
  item=read(path)
  if item is None:rows.append(dict(path=str(path),status='NOT_STARTED',completed=0,target_completed=0,passed=0,failed=0));continue
  completed=item.get('completed',[]);target_rows=completed_targets(item)
  targets.update(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False) for row in target_rows)
  rows.append(dict(path=str(path),status=item.get('status','UNKNOWN'),completed=len(completed),target_completed=len(target_rows),
   passed=sum(x.get('outcome')=='PASS' for x in completed),failed=sum(x.get('outcome')=='FAIL' for x in completed),active_job=item.get('active_job')))
 status=driver.get('status','WAITING_CORE_FIRST_PASS')
 # Ignore per-job RUNNING/WAITING and temporary archive/memory waits.
 stage=status if status.startswith('RUNNING_') or status in DRIVER_END else prior_stage or 'CORE_FIRST_PASS'
 terminal_queues=sorted(x['path']+':'+x['status'] for x in rows if terminal(x['status']))
 all_ended=bool(rows) and all(terminal(x['status']) for x in rows)
 return dict(observed_unix=time.time(),queues=rows,completed=sum(x['completed'] for x in rows),
  target_completed=len(targets),target_completed_signature=hashlib.sha256(json.dumps(sorted(targets),ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
  remaining_completed={x['path']:x['target_completed'] for x in rows if any(name in x['path'] for name in ('unified_main_continuation/','unified_scaling_continuation/','unified_mechanism_application/'))},
  stage=stage,driver_status=status,terminal_queues=terminal_queues,
  finished=status=='FIRST_PASS_SCHEDULED_STAGES_FINISHED' and all_ended,
  stopped=status=='STOPPED_REQUIRES_INSPECTION' and all(x['status']=='NOT_STARTED' or terminal(x['status']) for x in rows))


def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def point(link,target):
 tmp=link.with_name(link.name+'.next')
 if tmp.is_symlink():tmp.unlink()
 tmp.symlink_to(target.name,target_is_directory=True);os.replace(tmp,link)


def export(campaign,output,progress,authority,plot_python,plot_enabled):
 generation=output/('.generation.'+str(time.time_ns()));generation.mkdir()
 collect=load_module('first_pass_collector',HERE/'collect_first_pass.py')
 build=load_module('first_pass_builder',HERE/'build_first_pass.py')
 remaining=load_module('remaining_results',HERE/'remaining_results.py')
 bundle=collect.collect(campaign,extra_queue_states=[r['path'] for r in progress['queues']])
 remaining.capture(campaign,bundle)
 with (generation/'receipts.json.gz').open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as packed:
   packed.write(json.dumps(bundle,ensure_ascii=False,separators=(',',':')).encode())
 del bundle
 report=build.build(generation/'receipts.json.gz',authority,generation)
 remaining_report=remaining.build(generation/'receipts.json.gz',generation/'remaining')
 with (generation/'first_pass.html').open('a') as stream:
  stream.write('<p><a href="remaining/remaining_results.html">E2 固定后端扩展、E4S 请求批次与 E6 完整工作流：当前结果、图形及完整 CSV</a></p>')
 save(generation/'stage_progress.json',progress)
 plot_status='SKIPPED_MATPLOTLIB_UNAVAILABLE'
 if plot_enabled:
  try:
   with (generation/'plot.log').open('w') as log:
    done=subprocess.run([str(plot_python),'-B',str(HERE/'render_icde.py'),'--report',str(generation/'first_pass.json'),'--output',str(generation)],stdout=log,stderr=subprocess.STDOUT,timeout=180)
   plot_status='PASS' if done.returncode==0 else 'FAILED_OPTIONAL_PLOT_SEE_LOG'
  except Exception as exc:
   plot_status='FAILED_OPTIONAL_PLOT: '+repr(exc)
 receipt=dict(status='EXPORTED',observed_unix=time.time(),counts=report['counts'],
  source_scope='E1 static matrix and exact source71 plans for E2 fixed-native scaling, E4S batches, E6 workflows; E3/E4/E5 retain their existing state_main summarizer.',
  remaining_counts=remaining_report['counts'],
  plot_status=plot_status,cpu_affinity=sorted(os.sched_getaffinity(0)),gpu_executed=False,
  authority={'path':str(authority),'sha256':hashlib.sha256(authority.read_bytes()).hexdigest()},
  array_payloads_read=False,qualification_launched=False,
  source_files={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('export_on_progress.py','collect_first_pass.py','build_first_pass.py','render_icde.py','remaining_results.py','remaining_protocol.json')})
 save(generation/'export_receipt.json',receipt)
 current,previous=output/'current',output/'previous'
 for link in (current,previous):
  if link.exists() and not link.is_symlink():raise RuntimeError('Export path must be a managed symlink: '+str(link))
 old_current=current.resolve() if current.exists() else None
 old_previous=previous.resolve() if previous.exists() else None
 # The current pointer changes atomically only after all text tables are complete.
 point(current,generation)
 if old_current:point(previous,old_current)
 if old_previous and old_previous not in (generation,old_current):
  if old_previous.parent==output and old_previous.name.startswith('.generation.'):shutil.rmtree(old_previous)
 return receipt


def clean_unpublished(output):
 keep={p.resolve() for p in (output/'current',output/'previous') if p.is_symlink()}
 for path in output.glob('.generation.*'):
  if path not in keep and path.is_dir():shutil.rmtree(path)


def run(args):
 campaign=args.campaign.resolve();output=(args.output or campaign/'first_pass_20261106/export').resolve();output.mkdir(parents=True,exist_ok=True)
 if args.cpu_set:
  selected={int(x) for x in args.cpu_set.split(',')}
  if not selected or not selected<=os.sched_getaffinity(0):raise ValueError('Unavailable exporter CPU placement')
  os.sched_setaffinity(0,selected)
 lease=(output/'watcher.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 authority=(args.authority or HERE/'e1_scope_authority.json').resolve()
 if not authority.is_file():raise FileNotFoundError(authority)
 plot_python=Path(args.plot_python or sys.executable)
 try:
  probe=subprocess.run([str(plot_python),'-c','import matplotlib,numpy'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
  plot_enabled=probe.returncode==0
 except (OSError,subprocess.TimeoutExpired):plot_enabled=False
 clean_unpublished(output)
 previous=read(output/'state.json') or {};last_count=previous.get('last_export_completed',-1)
 last_stage=previous.get('last_export_stage');last_terminal=previous.get('last_export_terminal_queues',[])
 last_remaining=previous.get('last_export_remaining_completed',{})
 last_targets=previous.get('last_export_target_completed_signature')
 last_finished=previous.get('last_export_finished',False);last_stopped=previous.get('last_export_stopped',False)
 stage=previous.get('stage');failures=0
 while True:
  try:
   progress=snapshot(campaign,stage);stage=progress['stage']
   due=needs_export(progress,dict(last_export_completed=last_count,last_export_stage=last_stage,
    last_export_terminal_queues=last_terminal,last_export_remaining_completed=last_remaining,
    last_export_target_completed_signature=last_targets,last_export_finished=last_finished,last_export_stopped=last_stopped))
   state={**progress,'status':'WATCHING','pid':os.getpid(),'last_export_completed':last_count,
    'last_export_stage':last_stage,'last_export_terminal_queues':last_terminal,'last_export_remaining_completed':last_remaining,
    'last_export_target_completed_signature':last_targets,'last_export_finished':last_finished,'last_export_stopped':last_stopped,'plot_available':plot_enabled}
   if due:
    receipt=export(campaign,output,progress,authority,plot_python,plot_enabled)
    last_count=progress['completed'];last_stage=stage;last_terminal=progress['terminal_queues'];last_remaining=progress['remaining_completed'];failures=0
    last_targets=progress['target_completed_signature'];last_finished=progress['finished'];last_stopped=progress['stopped']
    state.update(last_export_completed=last_count,last_export_stage=last_stage,last_export_terminal_queues=last_terminal,last_export_remaining_completed=last_remaining,last_export_target_completed_signature=last_targets,
     last_export_finished=last_finished,last_export_stopped=last_stopped,last_export=receipt)
   if progress['finished']:state['status']='FINISHED_FINAL_EXPORT'
   elif progress['stopped']:state['status']='STOPPED_WITH_INCOMPLETE_STAGES_FINAL_EXPORT'
   save(output/'state.json',state);clean_unpublished(output)
   if progress['finished'] or progress['stopped'] or args.once:return
  except Exception as exc:
   clean_unpublished(output)
   failures+=1;save(output/'state.json',dict(status='EXPORT_ERROR_RETRYING',error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid(),observed_unix=time.time(),
    last_export_completed=last_count,last_export_stage=last_stage,last_export_terminal_queues=last_terminal,last_export_remaining_completed=last_remaining,
    last_export_target_completed_signature=last_targets,last_export_finished=last_finished,last_export_stopped=last_stopped,stage=stage,consecutive_errors=failures))
   if args.once:raise
  time.sleep(60)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--output',type=Path);parser.add_argument('--authority',type=Path);parser.add_argument('--plot-python');parser.add_argument('--once',action='store_true');parser.add_argument('--cpu-set',help='Optional comma-separated available CPU IDs; default inherits current affinity')
 run(parser.parse_args())
