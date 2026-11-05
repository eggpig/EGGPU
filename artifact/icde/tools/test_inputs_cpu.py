#!/usr/bin/env python3
"""Run the portable input format/resume CPU contracts without downloading data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    if work.exists() or work == ROOT or ROOT in work.parents:
        raise ValueError('Choose an absent work directory outside the checkout')
    work.mkdir(parents=True)
    inputs = ROOT / 'artifact/icde/inputs'
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               '--confcutdir=' + str(inputs), '--basetemp=' + str(work/'temporary'),
               str(inputs/'test_portable_inputs.py'), str(inputs/'test_download_resume.py')]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               CUDA_VISIBLE_DEVICES='', TMPDIR=str(work))
    with (work/'checks.log').open('x') as log:
        status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (work/'receipt.json').write_text(json.dumps(dict(status='PASS_CPU_ONLY' if not status else 'FAIL_CPU',
        gpu_executed=False, returncode=status, scope='Input format, mappings, cache/resume contracts; no whole graph download'))+'\n')
    raise SystemExit(status)


if __name__ == '__main__':
    main()
