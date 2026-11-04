"""CPU tests for byte-exact protocol weights and source-format normalization."""
from pathlib import Path
import argparse, hashlib, json, os, struct, subprocess

def main():
 p=argparse.ArgumentParser();p.add_argument('--tools',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
 a.work.mkdir(parents=True,exist_ok=False);checks=[]
 cases=[('snap',4,5,'# Nodes: 4 Edges: 5\n0 1\n0 1\n1 1\n1 2\n2 1\n',[(0,1),(1,2),(2,1)]),
        ('mm',4,4,'%%MatrixMarket matrix coordinate pattern symmetric\n% fixture\n4 4 2\n1 2\n2 3\n',[(0,1),(1,0),(1,2),(2,1)])]
 for mode,n,m,text,expected in cases:
  d=a.work/mode;d.mkdir()
  r=subprocess.run([str(a.tools/'edge_list_to_csr'),mode,str(n),str(m),str(d)],input=text,text=True,capture_output=True,check=True)
  offsets=struct.unpack('<'+'i'*(n+1),(d/'offsets.bin').read_bytes())
  raw=(d/'indices.bin').read_bytes();indices=struct.unpack('<'+'i'*(len(raw)//4),raw)
  edges=[(u,indices[j]) for u in range(n) for j in range(offsets[u],offsets[u+1])]
  assert edges==expected and offsets[-1]==len(expected)
  for directed in (0,1):
   out=d/f'weights_{directed}.bin'
   subprocess.run([str(a.tools/'weights'),str(d/'offsets.bin'),str(d/'indices.bin'),str(out),str(n),str(len(edges)),'test-protocol',str(directed)],check=True,env=dict(os.environ,OMP_NUM_THREADS='2'),capture_output=True)
   actual=struct.unpack('<'+'d'*len(edges),out.read_bytes())
   wanted=[]
   for u,v in edges:
    if not directed:u,v=min(u,v),max(u,v)
    wanted.append(1+int.from_bytes(hashlib.sha256(f'eggpu-w16-v1|20261104|test-protocol|{u}|{v}'.encode()).digest()[:8],'big')%16)
   assert list(actual)==wanted
   checks.append({'format':mode,'directed_weights':bool(directed),'full_weights_match_python':True,'isolates_preserved':True})
 d=a.work/'invalid';d.mkdir()
 r=subprocess.run([str(a.tools/'edge_list_to_csr'),'snap','4','1',str(d)],input='0 4\n',text=True,capture_output=True)
 assert r.returncode!=0
 receipt={'status':'PASS_CPU_INPUT_TOOLS','checks':checks,'out_of_domain_rejected':True,'gpu_executed':False}
 (a.work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
