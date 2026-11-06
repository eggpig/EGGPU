"""CPU bootstrap and root-started successor for the explicit new 38-cell epoch."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,time
from prepare import bind,read,checked

HERE=Path(__file__).resolve().parent
TERMINAL={'PASS_REQUESTED_JOBS','COMPLETE_WITH_FAILURES'}

def atomic(path,value):
    path=Path(path);temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def bootstrap(campaign,fresh_state,output,python):
    campaign,fresh_state,output=map(lambda x:Path(x).resolve(),(campaign,fresh_state,output))
    config=dict(schema='state.additive.fresh.bootstrap.v1',campaign=str(campaign),fresh_state=str(fresh_state),output=str(output),
        python=str(Path(python).absolute()),core_queue=str(fresh_state/'queue/state.json'),focused=str(fresh_state/'results/focused_source_p8/receipt.json'),
        queue=str(output/'queue'),plan=str(output/'plan.json'),controller_lock=str(campaign/'root_controller.lock'),
        target_cells=38,initial_checks=2,main_denominator=504,gpu_submission_requires_root_start_flag=True,
        files={name:bind(HERE/name) for name in ('watch_fresh_additions.py','prepare.py','prepare_fresh_policy.py','fresh_policy_support.py','export_progress.py','export_fresh_progress.py')},
        controller=bind(campaign/'run_campaign_queue_v9.py'))
    if output.exists():
        if read(output/'bootstrap.json')!=config:raise ValueError('Existing bootstrap identity differs')
        return config
    output.mkdir(parents=True,exist_ok=False)
    atomic(output/'bootstrap.json',config)
    atomic(output/'stage_manifest.json',dict(status='WAIT_FOCUSED',queue=config['queue'],plan=config['plan'],target_cells=38,initial_checks=2,
        prerequisite_focused=config['focused'],prerequisite_core_queue=config['core_queue'],bootstrap=bind(output/'bootstrap.json'),gpu_submitted=False))
    return config

def dependency_state(config):
    path=Path(config['focused'])
    if not path.exists():return 'WAIT_FOCUSED'
    focused=read(path)
    if focused.get('status')!='PASS_STATE_GPU_QUALIFICATION' or focused.get('phase')!='tiny':return 'FOCUSED_NOT_PASS'
    return 'FOCUSED_PASS'

def watch(config_path,execute_gpu_after_core,poll_seconds):
    config_path=Path(config_path).resolve();config=read(config_path);output=Path(config['output'])
    for item in [*config['files'].values(),config['controller']]:checked(item)
    python=config['python'];state_path=output/'watcher_state.json'
    def report(status,**details):atomic(state_path,dict(status=status,updated_unix=time.time(),gpu_launch_authorized_by_root_flag=execute_gpu_after_core,**details))
    with (output/'watcher.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=Path(config['plan'])
        if not plan.exists():
            while True:
                status=dependency_state(config);report(status)
                if status=='FOCUSED_PASS':break
                if status=='FOCUSED_NOT_PASS':
                    # A RUNNING record may precede its atomic final write.
                    raw=read(config['focused']).get('status','')
                    if raw.startswith(('FAIL','ERROR','CANCEL')):raise RuntimeError('Fresh focused check failed: '+raw)
                time.sleep(poll_seconds)
            command=[python,'-B',str(HERE/'prepare_fresh_policy.py'),'--campaign',config['campaign'],'--state-main',config['fresh_state'],'--output',str(output)]
            report('CPU_PREPARING',command=command)
            with (output/'prepare.log').open('xb') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
            if result.returncode:report('FAILED_CPU_PREPARATION',returncode=result.returncode);raise RuntimeError('Fresh additive CPU preparation failed')
        document=read(plan)
        if document.get('target_cells')!=38 or document.get('old_policy_results_reused') is not False:raise ValueError('Wrong new plan')
        # Post-prepare metadata only; this exporter never opens a GPU context.
        exporter=[python,'-B',str(HERE/'export_fresh_progress.py'),'--plan',str(plan),'--output',str(output/'progress')]
        launch=output/'controller_launch.json';child=None
        while True:
            result=subprocess.run(exporter,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
            if result.returncode:report('EXPORT_FAILED',error=result.stderr[-4000:]);raise RuntimeError('Fresh additive export failed')
            core=Path(config['core_queue']);core_status=read(core).get('status','NOT_STARTED') if core.exists() else 'NOT_STARTED'
            if not launch.exists():
                if core_status not in TERMINAL:
                    if core_status.startswith('FAILED_'):report('CORE_CONTROLLER_FAILED',core_status=core_status);raise RuntimeError('Core controller needs root inspection')
                    report('WAIT_CORE_TERMINAL',core_status=core_status);time.sleep(poll_seconds);continue
                command=[python,'-B',config['controller']['path'],'--plan',str(plan),'--output',config['queue'],'--controller-lock',config['controller_lock'],'--wall-timeout-floor','10800']
                intent=dict(status='ROOT_LAUNCH_READY',command=command,plan=bind(plan),core_terminal_receipt=bind(core),bootstrap=bind(config_path),gpu_submitted=False)
                atomic(output/'root_launch_request.json',intent)
                if not execute_gpu_after_core:
                    report('ROOT_LAUNCH_READY',request=str(output/'root_launch_request.json'));return intent
                if Path(config['queue']).exists():raise ValueError('Existing queue without launch intent requires root inspection')
                intent.update(status='LAUNCH_INTENT',created_unix=time.time(),gpu_submitted=True)
                with launch.open('x') as stream:json.dump(intent,stream,indent=2)
                with (output/'controller.log').open('xb') as log:
                    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
                intent.update(status='SUBMITTED',pid=child.pid);atomic(launch,intent)
            queue=Path(config['queue'])/'state.json';queue_status=read(queue).get('status','STARTING') if queue.exists() else 'STARTING'
            report('EXPORTING_SUBMITTED_QUEUE',queue_status=queue_status,launch=read(launch))
            if queue_status in TERMINAL:
                subprocess.run(exporter,check=True,stdout=subprocess.DEVNULL,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
                return dict(status='TERMINAL_EXPORTED',queue_status=queue_status)
            if child is not None and child.poll() is not None and queue_status not in TERMINAL:
                report('CONTROLLER_EXITED_NONTERMINAL',returncode=child.returncode,queue_status=queue_status);raise RuntimeError('Controller exited without terminal queue')
            time.sleep(poll_seconds)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    b=sub.add_parser('bootstrap')
    for name in ('campaign','fresh-state','output','python'):b.add_argument('--'+name,required=True)
    w=sub.add_parser('watch');w.add_argument('--config',type=Path,required=True);w.add_argument('--execute-gpu-after-core',action='store_true');w.add_argument('--poll-seconds',type=int,default=30)
    a=p.parse_args();result=bootstrap(a.campaign,a.fresh_state,a.output,a.python) if a.action=='bootstrap' else watch(a.config,a.execute_gpu_after_core,max(10,a.poll_seconds));print(json.dumps(result))
