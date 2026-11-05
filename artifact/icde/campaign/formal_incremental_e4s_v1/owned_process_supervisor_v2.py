"""Linux CPU-only subreaper for one explicitly submitted worker process tree.

Each worker has a dedicated supervisor. Orphaned setsid descendants are adopted
by this process, never inferred from a global PID range or a process name. Signals
use pidfds after exact /proc starttime validation; process-group IDs are recorded
but unrelated group members are never signalled.
"""
from pathlib import Path
import argparse,ctypes,errno,json,os,platform,signal,subprocess,sys,time


def identity(pid):
    try:
        raw=(Path('/proc')/str(pid)/'stat').read_text();rest=raw[raw.rfind(')')+2:].split()
        return dict(pid=int(pid),state=rest[0],ppid=int(rest[1]),pgrp=int(rest[2]),session=int(rest[3]),starttime=int(rest[19]))
    except (FileNotFoundError,ProcessLookupError):return None


def snapshot():
    result={}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:row=identity(int(p.name))
            except PermissionError:continue
            if row:result[row['pid']]=row
    return result


def pidfd_open(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid,0)
    # Some Conda Python builds omit os.pidfd_open although the Linux kernel
    # supports it. x86-64 syscall 434 is specified in asm/unistd_64.h; do not
    # guess an ABI for other architectures.
    if platform.machine() not in ('x86_64','amd64'):raise RuntimeError('Python pidfd_open or verified x86-64 ABI required')
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    fd=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
    if fd<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
    return int(fd)


def pidfd_send_signal(fd,signum):
    if hasattr(signal,'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd,signum,None,0)
    # Linux x86-64 asm/unistd_64.h defines pidfd_send_signal as syscall 424.
    # The target interpreter may omit the Python wrapper even with kernel support.
    if platform.machine() not in ('x86_64','amd64'):
        raise RuntimeError('Python pidfd_send_signal or verified x86-64 ABI required')
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    result=libc.syscall(ctypes.c_long(424),ctypes.c_int(fd),ctypes.c_int(signum),ctypes.c_void_p(),ctypes.c_uint(0))
    if result<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
    return None


def signal_identity(row,signum):
    """A pidfd prevents a signal reaching a replacement process with reused PID."""
    try:fd=pidfd_open(row['pid'])
    except ProcessLookupError:return 'already_exited'
    try:
        now=identity(row['pid'])
        if now is None or now['starttime']!=row['starttime']:return 'identity_changed_not_signalled'
        if now['state']=='Z':return 'zombie_not_signalled'
        pidfd_send_signal(fd,signum);return 'sent'
    except ProcessLookupError:return 'already_exited'
    finally:os.close(fd)


class OwnedTree:
    def __init__(self,supervisor_pid):self.supervisor_pid=supervisor_pid;self.known={};self.events=[]
    def refresh(self):
        current=snapshot();owned={pid for pid,row in current.items() if row['ppid']==self.supervisor_pid}
        while True:
            extended=owned|{pid for pid,row in current.items() if row['ppid'] in owned}
            if extended==owned:break
            owned=extended
        for pid in owned:
            row=current[pid];self.known[(pid,row['starttime'])]=row
        return [row for row in current.values() if (row['pid'],row['starttime']) in self.known]
    def living(self):return [r for r in self.refresh() if r['state']!='Z']
    def reap_orphans(self,worker):
        worker.poll()
        for row in self.refresh():
            if row['pid']==worker.pid:continue
            try:os.waitpid(row['pid'],os.WNOHANG)
            except (ChildProcessError,ProcessLookupError):pass
    def cleanup(self,worker,term_grace=2.,kill_grace=3.):
        sent=set()
        for signum,grace in ((signal.SIGTERM,term_grace),(signal.SIGKILL,kill_grace)):
            deadline=time.monotonic()+grace
            while True:
                live=self.living()
                if not live:break
                # Observe all descendants before signalling any parent. New
                # descendants created by signal handlers are adopted next pass.
                for row in live:
                    key=(row['pid'],row['starttime'],signum)
                    if key not in sent:
                        outcome=signal_identity(row,signum);sent.add(key)
                        self.events.append(dict(pid=row['pid'],starttime=row['starttime'],pgrp=row['pgrp'],signal=int(signum),outcome=outcome))
                self.reap_orphans(worker)
                if time.monotonic()>=deadline:break
                time.sleep(.02)
        self.reap_orphans(worker)
        return self.living()


def write_report(path,value):
    path=Path(path);temporary=path.with_suffix('.partial.json')
    with temporary.open('w') as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def supervise(command,log_fd,timeout,report_path,expected_parent):
    if int(Path('/proc/self/stat').read_text().split(' ',1)[0])!=os.getpid():raise RuntimeError('Visible /proc must match this PID namespace')
    probe=pidfd_open(os.getpid())
    try:pidfd_send_signal(probe,0)
    finally:os.close(probe)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER')
    interrupted=[]
    def cancel(signum,frame):interrupted.append(signum)
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
    if os.getppid()!=expected_parent:raise RuntimeError('Queue parent exited before worker admission')
    tree=OwnedTree(os.getpid());worker=None;report=dict(schema='campaign.owned_process_supervision.v1',supervisor_pid=os.getpid(),gpu_imported_by_supervisor=False,status='STARTING',signals=[])
    code=125
    try:
        worker=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log_fd,stderr=log_fd,start_new_session=True)
        initial=identity(worker.pid);tree.refresh();report['worker_pid']=worker.pid;report['worker_identity']=initial
        print(json.dumps(dict(event='worker_started',pid=worker.pid)),flush=True)
        start=time.monotonic();report['started_monotonic']=start
        while worker.poll() is None:
            tree.refresh()
            if interrupted or time.monotonic()-start>=timeout:
                report['status']='CANCELLED' if interrupted else 'TIMED_OUT';code=128+interrupted[-1] if interrupted else 124
                survivors=tree.cleanup(worker);report['surviving_owned_processes']=survivors
                if survivors:report['status']='FAILED_OWNED_PROCESS_CLEANUP';code=126
                break
            time.sleep(.02)
        else:
            code=worker.returncode;report['worker_returncode']=code
            # Executors may complete a short asynchronous shutdown after their
            # parent exits. Persistent orphans are a controller failure.
            end=time.monotonic()+2.
            while tree.living() and time.monotonic()<end:
                tree.reap_orphans(worker);time.sleep(.02)
            if tree.living():
                report['status']='FAILED_WORKER_LEFT_DESCENDANTS';survivors=tree.cleanup(worker);report['surviving_owned_processes']=survivors;code=126 if survivors else 125
            else:report['status']='PASS_PROCESS_EXIT' if code==0 else 'WORKER_FAILED'
        return code
    finally:
        if worker is not None:
            if tree.living():report['surviving_owned_processes']=tree.cleanup(worker)
            tree.reap_orphans(worker)
        report.update(supervisor_returncode=code,owned_identities=list(tree.known.values()),signals=tree.events,finished_monotonic=time.monotonic())
        write_report(report_path,report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--command-json',required=True);p.add_argument('--log-fd',type=int,required=True);p.add_argument('--timeout',type=float,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--expected-parent',type=int,required=True);a=p.parse_args()
    command=json.loads(a.command_json)
    if not isinstance(command,list) or not command or any(not isinstance(x,str) for x in command):raise ValueError('Explicit command argv required')
    raise SystemExit(supervise(command,a.log_fd,a.timeout,a.report,a.expected_parent))
