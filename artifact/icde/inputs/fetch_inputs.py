"""Fetch only the three missing frozen inputs; no GPU or normalization."""
import argparse, hashlib, http.client, json, re, shutil, time
from pathlib import Path
import urllib.error, urllib.parse, urllib.request

def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def download_file(url, dest, timeout=45, retries=4):
    """Resume only when HTTP Content-Range proves the returned start offset."""
    if dest.exists():
        return {"status": "EXISTING_REQUIRES_MD5", "url": url, "bytes": dest.stat().st_size}
    part = dest.with_name(dest.name + ".partial")
    for attempt in range(retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "EGGPU-public-artifact/2", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = "bytes=" + str(offset) + "-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as source:
                status = source.status
                if status not in (200, 206):
                    raise ValueError("Unexpected download status " + str(status))
                length = source.headers.get("Content-Length")
                expected = int(length) if length is not None else None
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", source.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[2]) < offset:
                        raise ValueError("Invalid or mismatched resume Content-Range")
                    range_length = int(match[2]) - offset + 1
                    if expected is not None and expected != range_length:
                        raise ValueError("Content-Length and Content-Range disagree")
                    expected = range_length
                    mode = "ab" if offset else "wb"
                    total = None if match[3] == "*" else int(match[3])
                else:
                    # Server may legally ignore Range: restart, never append a
                    # complete 200 response to existing partial content.
                    mode, offset, total = "wb", 0, expected
                received = 0
                with part.open(mode) as target:
                    while True:
                        block = source.read(8 << 20)
                        if not block:
                            break
                        target.write(block)
                        received += len(block)
                if expected is not None and received != expected:
                    raise OSError("Incomplete response: expected %d, received %d" % (expected, received))
                if total is not None and part.stat().st_size != total:
                    raise OSError("Incomplete object after partial response")
            part.replace(dest)
            return {"status": "DOWNLOADED_REQUIRES_MD5", "url": url,
                    "bytes": dest.stat().st_size, "attempts": attempt + 1, "resumed_from": offset}
        except urllib.error.HTTPError as error:
            match = re.fullmatch(r"bytes \*/(\d+)", error.headers.get("Content-Range", ""))
            if error.code == 416 and match and offset == int(match[1]) and offset > 0:
                part.replace(dest)
                return {"status": "COMPLETE_PARTIAL_REQUIRES_MD5", "url": url, "bytes": offset}
            if attempt == retries or error.code not in (408, 429, 500, 502, 503, 504):
                raise
        except (OSError, http.client.IncompleteRead):
            if attempt == retries:
                raise
        time.sleep(min(2 ** attempt, 8))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 a.root.mkdir(parents=True,exist_ok=True)
 state={'status':'RUNNING','gpu_executed':False,'datasets':{}}
 def save():
  tmp=a.root/'fetch_state.partial';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.root/'fetch_state.json')
 save()
 specs={
 'LJ':('https://snap.stanford.edu/data/soc-LiveJournal1.html',['https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz']),
 'US':('https://sparse.tamu.edu/DIMACS10/road_usa',['https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/road_usa.tar.gz']),
 'TW':('https://law.di.unimi.it/webdata/twitter-2010/',['https://data.law.di.unimi.it/webdata/twitter-2010/twitter-2010.md5sums','https://data.law.di.unimi.it/webdata/twitter-2010/twitter-2010.properties','https://data.law.di.unimi.it/webdata/twitter-2010/twitter-2010.graph'])}
 for name,(page,urls) in specs.items():
  row={'status':'FETCHING','source_page':page,'files':[]};state['datasets'][name]=row;save()
  try:
   out=a.root/name;out.mkdir(exist_ok=True)
   required=(4<<30) if name=='TW' else (1<<30)
   if shutil.disk_usage(out).free<required+(6<<30):raise RuntimeError('Insufficient disk plus 6GiB reserve')
   for url in urls:
    path=out/urllib.parse.urlsplit(url).path.rsplit('/',1)[-1]
    row['current_file']=path.name;save()
    info=download_file(url,path,timeout=45,retries=4)
    info.update(path=str(path),sha256=digest(path));row['files'].append(info);save()
   if name=='TW':
    expected={}
    for line in (out/'twitter-2010.md5sums').read_text().splitlines():
     m=re.match(r'^([0-9a-fA-F]{32})\s+\*?(.+)$',line.strip())
     if m:expected[Path(m[2]).name]=m[1].lower()
    for f in row['files']:
     path=Path(f['path'])
     if path.suffix in ('.graph','.properties'):
      f['official_md5']=expected[path.name]
      if digest(path,'md5')!=expected[path.name]:raise ValueError('Official checksum mismatch')
   row['status']='PASS_SOURCE_FILES';row.pop('current_file',None)
  except Exception as e:row.update(status='FAILED',error=type(e).__name__+': '+str(e))
  save()
 state['status']='PASS_ALL_SOURCE_FILES' if all(x['status']=='PASS_SOURCE_FILES' for x in state['datasets'].values()) else 'INCOMPLETE_SOURCE_FILES'
 save();print(json.dumps(state,indent=2))
if __name__=='__main__':main()
