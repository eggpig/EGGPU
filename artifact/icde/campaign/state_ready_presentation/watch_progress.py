from pathlib import Path
import json,time,os,sys,fcntl,subprocess
import argparse
parser=argparse.ArgumentParser()
for name in ('bundle','output','python'):parser.add_argument('--'+name,required=True)
args=parser.parse_args();out=Path(args.output);bundle=Path(args.bundle);py=args.python
summarizer=Path(__file__).resolve().with_name('summarize_state_ready.py')
out.mkdir(exist_ok=True)
with (out/'watcher.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);last=None
 while True:
  signatures=[]
  for p in sorted((bundle/'results').glob('*/receipt.json')):
   s=p.stat();signatures.append((str(p),s.st_size,s.st_mtime_ns))
  if signatures!=last:
   cmd=[py,'-B',str(summarizer),'--bundle',str(bundle),'--output',str(out/'current'),'--render']
   r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
   state=dict(status='WATCHING' if r.returncode==0 else 'EXPORT_FAILED',returncode=r.returncode,receipt_count=len(signatures),updated_unix=time.time(),stderr=r.stderr[-4000:])
   temporary=out/'state.partial.json';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(out/'state.json')
   print(r.stdout,flush=True)
   if r.returncode:print(r.stderr,flush=True);raise SystemExit(r.returncode)
   last=signatures
  time.sleep(60)
