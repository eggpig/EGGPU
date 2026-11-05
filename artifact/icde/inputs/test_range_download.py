from pathlib import Path
from unittest.mock import patch
import io, json, tempfile
from range_download import download_file, RangeNotSupported

def main(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=False)
 payload=bytes(range(256))*16384
 class Reply(io.BytesIO):
  status=206
  def __init__(self,begin,end):
   super().__init__(payload[begin:end+1]);self.headers={'Content-Range':f'bytes {begin}-{end}/{len(payload)}','ETag':'fixture','Last-Modified':'fixture'}
 def request(req,timeout=None):
  bounds=req.headers['Range'].split('=')[1].split('-');return Reply(int(bounds[0]),int(bounds[1]))
 dest=root/'source.gz';dest.with_name(dest.name+'.partial').write_bytes(payload[:17])
 with patch('urllib.request.urlopen',request):result=download_file('https://fixture/source.gz',dest,retries=0)
 assert dest.read_bytes()==payload and result['resumed_from']==17
 state=json.loads(dest.with_name(dest.name+'.ranges.json').read_text());assert state['downloaded_bytes']==len(payload)
 class Unsupported(io.BytesIO):
  status=200;headers={}
 try:
  with patch('urllib.request.urlopen',lambda *a,**k:Unsupported(payload)):download_file('https://fixture/no-range',root/'unsupported',retries=0)
 except RangeNotSupported:pass
 else:raise AssertionError('Unverified nonrange response accepted')
 assert not (root/'unsupported.partial').exists()
 record={'status':'PASS_RANGE_DOWNLOAD_CPU','prefix_resume':True,'complete_bytes_equal':True,'nonrange_rejected_before_preallocation':True}
 (root/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':
 import sys;main(sys.argv[1])
