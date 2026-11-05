"""Pin only explicitly listed evidence paths; never glob, search or choose a run."""
import argparse
import hashlib
import json
from pathlib import Path
from e1_summary import Evidence, EvidenceError, require, summarize


def pin(draft_path, output):
    draft_path=Path(draft_path);output=Path(output)
    value=json.loads(draft_path.read_text())
    require(value.get('status')=='DRAFT_EXPLICIT_SELECTION', 'Expected an explicit draft selection')
    reader=Evidence(draft_path.parent,value.get('path_remap',[]))
    def visit(node):
        if isinstance(node,list):return [visit(item) for item in node]
        if not isinstance(node,dict):return node
        if 'path' in node:
            if 'sha256' in node:
                reader.verify(node)
            p=reader.path(node['path']).resolve();h=hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
            return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
        return {key:visit(item) for key,item in node.items()}
    value=visit(value);value['status']='FROZEN_EXPLICIT_SELECTION'
    # Paths in the pinned selection are absolute after any explicit relocation.
    # Embedded legacy record paths still use the same explicit remapping table.
    for item in value.get('path_remap',[]):
        target=Path(item['to'])
        if not target.is_absolute():item['to']=str((draft_path.parent/target).resolve())
    payload=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if output.exists():
        require(output.read_text()==payload,'Refuse to replace a different frozen selection')
        return
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.pending')
    require(not temporary.exists(),'A pending selection already exists')
    temporary.write_text(payload)
    try:
        summarize(temporary,value['mode'])
        temporary.replace(output)
    except BaseException:
        temporary.unlink()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();pin(args.draft,args.output)
    print(json.dumps({'status':'FROZEN_EXPLICIT_SELECTION','path':str(args.output.resolve()),
                      'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
