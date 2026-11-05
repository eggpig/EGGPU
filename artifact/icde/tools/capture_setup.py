#!/usr/bin/env python3
"""Capture a local runner setup file without executing CUDA or installing software.

The generated file contains your local GPU UUIDs; keep it local. The repository
contains only the separate, sanitized setup_public.json publication summary.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new setup file; existing run bindings are immutable')
    command = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv']
    output = subprocess.check_output(command, text=True)
    record = {'schema': 'eggpu.local_runner_setup/v1', 'gpu_executed': False,
              'gpus': {'command': command, 'output': output},
              'scope': 'Local resource identity used by SG/native/E6 admission; not a benchmark result'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': 'PASS_LOCAL_SETUP_CAPTURE', 'gpu_executed': False}))


if __name__ == '__main__':
    main()
