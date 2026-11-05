"""Validate one continuous parent-to-native CLOCK_MONOTONIC interval."""
from pathlib import Path
import json, os, time


def clock_domain():
    offsets = Path('/proc/self/timens_offsets')
    return {'clock': 'CLOCK_MONOTONIC',
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'time_namespace': os.readlink('/proc/self/ns/time'),
        'time_namespace_for_children': os.readlink('/proc/self/ns/time_for_children'),
        'time_offsets': offsets.read_text() if offsets.exists() else None,
        'resolution_seconds': time.clock_getres(time.CLOCK_MONOTONIC)}


def first_result(folder, start_ns, parent_end_ns, parent_domain):
    if type(start_ns) is not int or type(parent_end_ns) is not int or not 0 < start_ns < parent_end_ns:
        raise ValueError('Invalid parent CLOCK_MONOTONIC bracket')
    folder = Path(folder)
    launch = json.loads((folder/'launch.json').read_text())
    if launch['clock_domain'] != parent_domain or launch['parent_monotonic_start_ns'] != start_ns:
        raise ValueError('Author process does not share the parent monotonic domain/origin')
    if not start_ns <= launch['monotonic_launch_ns'] <= parent_end_ns:
        raise ValueError('Author process launch is outside the parent bracket')
    marker = 'SG_COMPLETE_MONOTONIC_JSON '
    rows = [json.loads(line.split(marker, 1)[1]) for line in (folder/'native.log').read_text().splitlines() if marker in line]
    if len(rows) != 3 or [row.get('sequence') for row in rows] != [0, 1, 2]:
        raise ValueError('Exactly three ordered complete-output monotonic events are required')
    previous = launch['monotonic_launch_ns']
    for row in rows:
        if (row.get('pid') != launch['pid'] or row.get('clock') != 'CLOCK_MONOTONIC' or
                type(row.get('t_ns')) is not int or not previous < row['t_ns'] <= parent_end_ns or
                type(row.get('values')) is not int or row['values'] <= 0):
            raise ValueError('Native complete-output marker PID/clock/time/count mismatch')
        previous = row['t_ns']
    if len({row['values'] for row in rows}) != 1:
        raise ValueError('Complete-output domain changed between the three calls')
    end_ns = rows[0]['t_ns']
    return {'status': 'PASS_CONTINUOUS_MONOTONIC_INTERVAL', 'clock_domain': parent_domain,
        'start_ns': start_ns, 'end_ns': end_ns, 'parent_process_end_ns': parent_end_ns,
        'seconds': (end_ns-start_ns)/1e9, 'values': rows[0]['values'], 'pid': launch['pid'],
        'events': rows, 'phase_durations_summed': False,
        'start_boundary': 'Canonical CSR already resident; before required weak projection, reorder, format conversion, serialization, trace creation and process startup',
        'end_boundary': 'Author first complete owning host vector and existing completion synchronization; fixed first-call cache preparation included where the author requires it',
        'excluded': 'Canonical input loading and hash verification before start; output disk archival, provenance hashing and independent correctness reference after endpoint'}
