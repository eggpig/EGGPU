"""Materialize the experimental overlay outside the checkout; no build or GPU."""
from pathlib import Path
import argparse,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def files(root):
 out={}
 for p in root.rglob('*'):
  if p.is_symlink():raise ValueError('Symlink source is not the frozen tree')
  if p.is_file():
   b=p.read_bytes();out[str(p.relative_to(root))]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 return out
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
 if out.exists() or out.is_relative_to(ROOT):raise ValueError('Choose a new directory outside this checkout')
 patch=json.loads((HERE/'source_patch.json').read_text())
 if files(ROOT/'Easy-Graph')!=patch['base_source_files']:raise ValueError('Published d84 base changed')
 for rel in patch['changed_files']:
  b=(HERE/'source_overlay'/rel).read_bytes()
  if {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}!=patch['candidate_source_files'][rel]:raise ValueError('Candidate overlay changed')
 shutil.copytree(ROOT/'Easy-Graph',out/'Easy-Graph')
 for rel in patch['changed_files']:
  target=out/'Easy-Graph'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/'source_overlay'/rel,target)
 if files(out/'Easy-Graph')!=patch['candidate_source_files']:raise ValueError('Materialized candidate tree differs')
 for source in HERE.glob('test_*'):shutil.copyfile(source,out/source.name)
 shutil.copytree(HERE/'runner',out/'runner')
 (out/'source_identity.json').write_text(json.dumps({'source_sha256':patch['candidate_source_sha256'],'base_source_sha256':patch['base_source_sha256'],'promoted_to_main':False,'gpu_executed':False},indent=2)+'\n')
 print(json.dumps({'status':'PASS_EXACT_EXPERIMENTAL_SOURCE_MATERIALIZED','source_sha256':patch['candidate_source_sha256'],'source_files':len(patch['candidate_source_files']),'changed_files':len(patch['changed_files']),'gpu_executed':False}))
if __name__=='__main__':main()
