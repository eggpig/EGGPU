"""Regenerate the frozen state input/configuration matrices without GPU work."""
from pathlib import Path
import argparse,importlib.util,json
ROOT=Path(__file__).resolve().parents[3]

def generate(kind):
 path=ROOT/'artifact/icde/campaign/state_final_plan/state_plan.py';spec=importlib.util.spec_from_file_location('public_state_plan',path);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
 rows=p.hotspot_grid() if kind=='hotspot' else p.grid()
 if kind=='state-inputs':
  unique={}
  for row in rows:
   key=p.input_key(row)
   if key not in unique:unique[key]={k:row[k] for k in ['experiment','datasets','partitions','rho','locality']}
  rows=[]
  for index,row in enumerate(unique.values()):rows.append(dict(row,id='input_'+str(index).zfill(3)))
 return dict(status='CPU_PROTOCOL_ONLY_NOT_RESULTS',source_sha256=json.loads((ROOT/'artifact/icde/provenance/runtime_source_freeze.json').read_text())['source_tree_sha256'],gpu_executed=False,kind=kind,cells=rows)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=('state-inputs','state-configurations','hotspot'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.output.open('x') as f:json.dump(generate(a.kind),f,indent=2);f.write('\n')
