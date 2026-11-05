"""CPU-only controller prelude, with no subprocess/GPU dispatch."""
from pathlib import Path
import argparse
import json
from admit_incremental import verify_cell

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--cell',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase',choices=['cpu'],required=True)
    args=parser.parse_args();print(json.dumps(verify_cell(args.spec,args.cell,args.output)))
