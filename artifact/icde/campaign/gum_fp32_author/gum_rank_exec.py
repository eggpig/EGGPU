#!/usr/bin/env python3
"""Record MPI rank launch, then exec the unchanged author binary in the same PID."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import gum_clock


def main():
    rank = int(os.environ['OMPI_COMM_WORLD_RANK']); size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    cpus = [int(x) for x in os.environ['EGGPU_GUM_CPUSET'].split(',')]
    if len(cpus) != 16 or len(set(cpus)) != 16 or set(os.sched_getaffinity(0)) != set(cpus):
        raise ValueError('MPI rank did not inherit the fixed 16-CPU allocation')
    path = Path(sys.argv[1]).resolve(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    clock = gum_clock.clock_domain(); launch_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    record = dict(clock_domain=clock, monotonic_launch_ns=launch_ns,
        parent_monotonic_start_ns=int(os.environ['EGGPU_GUM_MONOTONIC_START_NS']), rank=rank, size=size, pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        binary=dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest()),
        interpreter=dict(executable=str(Path(sys.executable).absolute()), prefix=sys.prefix, base_prefix=sys.base_prefix),
        argv=sys.argv[1:])
    output = Path(os.environ['GUM_RANK_EVIDENCE']) / f'launch-rank-{rank}.json'
    with output.open('x') as stream: json.dump(record, stream, indent=2); stream.write('\n')
    os.execv(str(path), [str(path), *sys.argv[2:]])


if __name__ == '__main__': main()
