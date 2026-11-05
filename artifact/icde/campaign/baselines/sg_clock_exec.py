#!/usr/bin/env python3
"""Bind native PID and monotonic namespace, then exec exactly the chosen binary."""
from pathlib import Path
import json, os, sys, time
import sg_clock


def main():
    binary = Path(sys.argv[1]).resolve()
    record = {'pid': os.getpid(), 'clock_domain': sg_clock.clock_domain(),
        'monotonic_launch_ns': time.clock_gettime_ns(time.CLOCK_MONOTONIC),
        'parent_monotonic_start_ns': int(os.environ['EGGPU_SG_START_NS']),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'argv': [str(binary), *sys.argv[2:]]}
    with Path(os.environ['EGGPU_SG_LAUNCH']).open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    os.execv(str(binary), [str(binary), *sys.argv[2:]])


if __name__ == '__main__':
    main()
