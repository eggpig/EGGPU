from pathlib import Path
import importlib.util,json,tempfile,unittest,sys
ROOT=Path(__file__).resolve().parents[3]
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class Projection(unittest.TestCase):
 def test_exact_grid_counts(self):
  m=load(ROOT/'artifact/icde/tools/protocol_grid.py','grid_projection')
  self.assertEqual(len(m.generate('state-configurations')['cells']),288);self.assertEqual(len(m.generate('state-inputs')['cells']),56);self.assertEqual(len(m.generate('hotspot')['cells']),32)
 def test_source_freeze_matches_runtime_source(self):
  import hashlib
  r=json.loads((ROOT/'artifact/icde/provenance/runtime_source_freeze.json').read_text());actual={}
  for name,b in r['source_files'].items():
   data=(ROOT/'Easy-Graph'/name).read_bytes();actual[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
  self.assertEqual(actual,r['source_files'])
 def test_paper_qualification_rejected_without_output(self):
  sys.path.insert(0,str(ROOT/'artifact/icde/presentation'));m=load(ROOT/'artifact/icde/presentation/paper_results.py','paper_projection')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'selection.json').write_text(json.dumps({'mode':'qualification'}))
   with self.assertRaises(ValueError):m.export(p/'selection.json',p/'paper')
   self.assertFalse((p/'paper').exists())
 def test_paper_missing_formal_rejected_without_output(self):
  sys.path.insert(0,str(ROOT/'artifact/icde/presentation'));m=load(ROOT/'artifact/icde/presentation/paper_results.py','paper_projection2')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'selection.json').write_text(json.dumps({'mode':'formal'}))
   with self.assertRaises((ValueError,KeyError,FileNotFoundError)):m.export(p/'selection.json',p/'paper')
   self.assertFalse((p/'paper').exists())
if __name__=='__main__':unittest.main()
