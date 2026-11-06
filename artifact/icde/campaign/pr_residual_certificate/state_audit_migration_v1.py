"""Strict CPU-only audit-wrapper transformation; original measurement jobs stay intact."""
from pathlib import Path
import copy,hashlib,json
METHOD='state-pr-full-residual-certificate-v1'
def read(path):return json.loads(Path(path).read_text())
def bind(path):
 p=Path(path).resolve();raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(binding):
 if bind(binding['path'])!=binding:raise ValueError('Audit binding changed: '+binding['path'])
 return Path(binding['path'])
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def amendment(path):
 result=read(path)
 if result.get('schema')!='state.audit.amendment.v1' or result.get('method')!=METHOD:raise ValueError('Unknown audit amendment')
 for b in result['files'].values():checked(b)
 if bind(__file__)['sha256']!=result['files']['state_audit_migration_v1.py']['sha256']:raise ValueError('Audit migration policy source differs')
 return result

def transformed(job,path):
 record=amendment(path);command=job['command']
 if len(command)!=7 or command[1]!='-B' or command[3]!='--selection' or command[5]!='--spec':raise ValueError('Unknown original state launch argv')
 spec=read(command[6])
 if spec.get('action')!='run' or spec.get('phase') not in ('target','formal'):return copy.deepcopy(job)
 if spec['id']!=job['id']:raise ValueError('Job/spec identity differs')
 if '--first-pass-spec' not in spec['worker_arguments'] or spec['worker_arguments'][spec['worker_arguments'].index('--first-pass-spec')+1]!=command[6]:raise ValueError('Bound worker spec differs')
 if '--output' not in spec['worker_arguments'] or str(Path(spec['worker_arguments'][spec['worker_arguments'].index('--output')+1])/'receipt.json')!=job['receipt']:raise ValueError('Worker output/receipt mismatch')
 result=copy.deepcopy(job)
 result['command']=[*command[:2],record['files']['launch_certificate.py']['path'],*command[3:],'--audit-amendment',str(Path(path).resolve())]
 result['bindings']=[*job.get('bindings',[]),bind(path),*record['files'].values()]
 return result

def validate_job(old,new,path,completed=False):
 expected=old if completed else transformed(old,path)
 if canonical(expected)!=canonical(new):raise ValueError('Audit migration changed a measurement job outside wrapper argv/appended audit bindings: '+old['id'])
 return canonical(old)!=canonical(new)

def validate_plan(old,new,path,prefix):
 a={k:v for k,v in old.items() if k!='jobs'};b={k:v for k,v in new.items() if k not in ('jobs','audit_migration')}
 if canonical(a)!=canonical(b):raise ValueError('Audit migration changed original plan metadata')
 if len(old['jobs'])!=len(new['jobs']):raise ValueError('Audit migration changed job count')
 changes=[]
 for index,(before,after) in enumerate(zip(old['jobs'],new['jobs'])):
  if validate_job(before,after,path,index<prefix):changes.append(before['id'])
 if not changes:raise ValueError('Audit migration has no future target changes')
 return changes
