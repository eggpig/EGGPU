"""Bind freshly produced local CPU oracles; never relabel a moved old proof."""
from pathlib import Path
import argparse,json
import e6_worker as worker
import e6_cpu_reference as reference

def create(paths):
 producer={'worker':reference.binding(worker.__file__),'reused':reference.binding(Path(worker.__file__).with_name('e6_reused.py'))};rows=[];identities=set()
 for path in paths:
  path=Path(path).resolve();record=json.loads(path.read_text());identity=record['identity'];blob=path.with_suffix('.npz')
  if record.get('gpu_executed') is not False or record.get('performance_eligible') is not False or identity['protocol']!=worker.CONTRACT:raise ValueError('Only newly computed independent CPU references are accepted')
  for name,field in [('worker','code'),('reused','reused')]:
   if identity[field]!={k:producer[name][k] for k in ('bytes','sha256')}:raise ValueError('Oracle producer differs from this public worker')
  for key in ['numpy','scipy']:
   if key in producer and producer[key]!=identity[key]:raise ValueError('Mixed oracle runtime')
   producer[key]=identity[key]
  output=reference.binding(blob)
  if record['output']!={k:output[k] for k in ('bytes','sha256')}:raise ValueError('Oracle data hash differs')
  if identity['input_sha256'] in identities:raise ValueError('Duplicate oracle input')
  identities.add(identity['input_sha256']);rows.append(dict(input_sha256=identity['input_sha256'],receipt=reference.binding(path),output=output))
 if not rows:raise ValueError('Explicit complete oracle receipts required')
 return dict(schema=reference.SCHEMA,protocol=worker.CONTRACT,gpu_executed=False,performance_eligible=False,producer=producer,references=rows)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipts',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.output.open('x') as f:json.dump(create(a.receipts),f,indent=2);f.write('\n')
