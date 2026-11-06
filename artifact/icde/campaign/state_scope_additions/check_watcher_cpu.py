"""CPU fixture checks; every launch/export subprocess is intercepted."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys,tempfile
import watch_fresh_additions as watcher

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scratch',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args();checks=[]
    original_run,original_popen=watcher.subprocess.run,watcher.subprocess.Popen
    with tempfile.TemporaryDirectory(prefix='fresh-watcher-',dir=args.scratch) as temporary:
        base=Path(temporary);campaign=base/'campaign';campaign.mkdir();(campaign/'run_campaign_queue_v9.py').write_text('# fixture controller only\n')
        fresh=base/'fresh';focused=fresh/'results/focused_source_p8/receipt.json';focused.parent.mkdir(parents=True)
        core=fresh/'queue/state.json';core.parent.mkdir(parents=True)
        for execute in (False,True):
            output=base/('executing' if execute else 'preparing');config=watcher.bootstrap(campaign,fresh,output,sys.executable)
            assert watcher.dependency_state(config)=='WAIT_FOCUSED';focused.write_text(json.dumps(dict(status='RUNNING',phase='tiny')))
            assert watcher.dependency_state(config)=='FOCUSED_NOT_PASS';focused.write_text(json.dumps(dict(status='PASS_STATE_GPU_QUALIFICATION',phase='tiny')))
            core.write_text(json.dumps(dict(status='PASS_REQUESTED_JOBS')));calls=[]
            def run(command,**kwargs):
                calls.append(('run',command))
                if 'prepare_fresh_policy.py' in command[2]:
                    (output/'plan.json').write_text(json.dumps(dict(target_cells=38,old_policy_results_reused=False)))
                return SimpleNamespace(returncode=0,stderr='')
            def popen(command,**kwargs):
                assert execute and command[2]==str(campaign/'run_campaign_queue_v9.py')
                assert core.exists() and json.loads(core.read_text())['status'] in watcher.TERMINAL
                calls.append(('popen',command));queue=output/'queue';queue.mkdir();(queue/'state.json').write_text(json.dumps(dict(status='PASS_REQUESTED_JOBS')))
                return SimpleNamespace(pid=12345,poll=lambda:0,returncode=0)
            watcher.subprocess.run=run;watcher.subprocess.Popen=popen
            result=watcher.watch(output/'bootstrap.json',execute,10)
            assert sum(kind=='popen' for kind,_ in calls)==int(execute)
            assert result['status']==('TERMINAL_EXPORTED' if execute else 'ROOT_LAUNCH_READY')
            assert (output/'root_launch_request.json').exists()
            if execute:
                assert sum('export_fresh_progress.py' in command[2] for kind,command in calls if kind=='run')==2
                # Restart observes durable intent and never launches twice.
                calls.clear();watcher.watch(output/'bootstrap.json',True,10);assert not any(kind=='popen' for kind,_ in calls)
            focused.unlink()
        checks.extend(['Missing/RUNNING focused does not admit CPU preparation','Bootstrap publishes exact future queue before preparation',
            'Without root flag terminal core produces launch request only','Root-flag mocked submission occurs only after terminal core',
            'Terminal final export executes; durable launch intent prevents restart resubmission'])
    watcher.subprocess.run=original_run;watcher.subprocess.Popen=original_popen
    record=dict(status='PASS_CPU_WATCHER_TRANSITIONS',fixture_only=True,subprocesses_mocked=True,gpu_executed=False,checks=checks,watcher=watcher.bind(watcher.__file__),test=watcher.bind(__file__))
    args.receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

if __name__=='__main__':main()
