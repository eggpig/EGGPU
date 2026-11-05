"""Incremental immutable-output maintenance; never changes measured values.

Only completed receipt-bound arrays are eligible. New/changed inodes receive a
full SHA check. A prior check can be reused only for a read-only inode with the
same device, inode, size, mtime, ctime, mode and link count. No writable file is
trusted from a prior fingerprint. This is a cooperative-writer optimization,
not a replacement for full artifact integrity checks at publication/export.
"""
from pathlib import Path
from contextlib import ExitStack
import argparse
import collections
import fcntl
import hashlib
import json
import os
import stat
import time

SUFFIXES = {'.npy', '.npz', '.i64', '.f64'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(path):
    p = Path(path)
    if p.is_symlink() or p.resolve() != p.absolute():
        raise ValueError('Symlink or noncanonical output path')
    s = p.stat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError('Only regular output arrays are allowed')
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                mode=stat.S_IMODE(s.st_mode), nlink=s.st_nlink)


def binding(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def write_atomic(path, value):
    p = Path(path)
    temporary = p.with_name(p.name + '.partial-' + str(os.getpid()))
    with temporary.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, p)
    fd = os.open(p.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def completed_arrays(states):
    arrays, proofs = {}, []
    for state_path in states:
        state = json.loads(Path(state_path).read_text())
        for item in state['completed']:
            top = item.get('receipt')
            if not top:
                continue
            p = Path(top['path']).absolute()
            if binding(p) != top:
                raise ValueError('Completed receipt changed')
            proofs.append(top)
            scope, seen = p.parent, set()

            def visit(x):
                if isinstance(x, dict):
                    if {'path', 'sha256', 'bytes'} <= x.keys():
                        q = Path(x['path']).absolute()
                        if q.is_relative_to(scope) and q.resolve() == q:
                            if q.suffix in SUFFIXES:
                                if not q.is_file():
                                    raise ValueError('A completed bound output is missing')
                                expected = (x['sha256'], x['bytes'])
                                if q in arrays and arrays[q] != expected:
                                    raise ValueError('Conflicting output bindings')
                                arrays[q] = expected
                            elif q.suffix == '.json' and q not in seen:
                                if binding(q) != {k: x[k] for k in ('path', 'bytes', 'sha256')}:
                                    raise ValueError('Bound child receipt changed')
                                read(q)
                    for child in x.values():
                        visit(child)
                elif isinstance(x, list):
                    for child in x:
                        visit(child)

            def read(q):
                seen.add(q)
                visit(json.loads(q.read_text()))

            read(p)
    return arrays, proofs


def writer_audit(fingerprints):
    selected = {(x['device'], x['inode']) for x in fingerprints}
    invisible = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.geteuid():
                continue
            fds = list((proc / 'fd').iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            invisible.append(int(proc.name))
            continue
        for fd in fds:
            try:
                s = fd.stat()
                if (s.st_dev, s.st_ino) not in selected:
                    continue
                flags = next(x.split()[1] for x in (proc / 'fdinfo' / fd.name).read_text().splitlines() if x.startswith('flags:'))
                if int(flags, 8) & os.O_ACCMODE:
                    raise ValueError('Writable descriptor on a completed output')
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                invisible.append(int(proc.name))
    return dict(unobservable_same_uid_pids=sorted(set(invisible)),
                scope='Cooperative completed jobs under controller/GPU locks; no claim of visibility into restricted unrelated processes.')


def deduplicate(arrays, index_path, output, apply=False):
    """Caller holds the controller and GPU locks; tests use private CPU files."""
    index_path, output = Path(index_path), Path(output)
    old = json.loads(index_path.read_text()) if index_path.exists() else dict(schema='completed.output.index.v1', paths={})
    if old.get('schema') != 'completed.output.index.v1':
        raise ValueError('Unexpected immutable-output index schema')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', apply=apply, started_unix=time.time(),
                  paths_checked=0, unique_inodes_hashed=0, bytes_hashed=0,
                  cached_paths=0, paths_replaced=0, storage_bytes_released=0,
                  skipped_external_hardlinks=[], all_run_paths_retained=True,
                  measurement_values_modified=False, gpu_executed=False)
    groups = collections.defaultdict(list)
    observed, checked = {}, {}
    try:
        for p, expected in arrays.items():
            p = Path(p)
            if p.suffix not in SUFFIXES:
                raise ValueError('Unapproved array role/suffix')
            fp = fingerprint(p)
            if fp['bytes'] != expected[1]:
                raise ValueError('Output size changed')
            observed[p] = fp
            groups[(fp['device'], fp['inode'])].append(p)
        report['writer_audit'] = writer_audit(observed.values())
        eligible = collections.defaultdict(list)
        for inode, aliases in groups.items():
            expected_set = {arrays[p] for p in aliases}
            if len(expected_set) != 1:
                raise ValueError('One inode has conflicting bindings')
            expected, size = next(iter(expected_set))
            anchor = aliases[0]
            fp = observed[anchor]
            hit = False
            for p in aliases:
                previous = old['paths'].get(str(p))
                if previous and (previous['sha256'], previous['bytes']) != (expected, size):
                    raise ValueError('Immutable path binding changed')
                if previous and not (fp['mode'] & 0o222) and previous['fingerprint'] == fp:
                    hit = True
            if hit:
                report['cached_paths'] += len(aliases)
            else:
                if sha(anchor) != expected or fingerprint(anchor) != fp:
                    raise ValueError('Completed output changed during full SHA')
                report['unique_inodes_hashed'] += 1
                report['bytes_hashed'] += size
            for p in aliases:
                if fingerprint(p) != fp:
                    raise ValueError('Output alias changed during validation')
                checked[p] = expected, size
                report['paths_checked'] += 1
            if fp['nlink'] != len(aliases):
                report['skipped_external_hardlinks'].append(str(anchor))
                continue
            if apply and fp['mode'] & 0o222:
                anchor.chmod(fp['mode'] & ~0o222)
                fp = fingerprint(anchor)
            eligible[(expected, size, inode[0], fp['mode'])].append((aliases, fp))
        with (output/'events.jsonl').open('x') as journal:
            for (expected, size, device, mode), copies in eligible.items():
                if len(copies) < 2:
                    continue
                canonical = copies[0][0][0]
                source = fingerprint(canonical)
                for aliases, old_fp in copies[1:]:
                    allocated = aliases[0].stat().st_blocks * 512
                    for p in aliases:
                        if fingerprint(p) != old_fp or fingerprint(canonical) != source:
                            raise ValueError('Writer or link set changed before replacement')
                        event = dict(path=str(p), canonical=str(canonical), sha256=expected, bytes=size, apply=apply)
                        journal.write(json.dumps(event)+'\n');journal.flush();os.fsync(journal.fileno())
                        if apply:
                            temporary = p.with_name('.incremental-dedup-'+str(os.getpid())+'-'+p.name)
                            if temporary.exists():
                                raise ValueError('Unresolved prior temporary link')
                            try:
                                os.link(canonical, temporary)
                                os.replace(temporary, p)
                            finally:
                                if temporary.exists():
                                    temporary.unlink()
                            report['paths_replaced'] += 1
                            source = fingerprint(canonical)
                            # Removing one alias changes the old inode's ctime/link count.
                            remaining = [q for q in aliases if q != p and q.stat().st_ino == old_fp['inode']]
                            if remaining:
                                old_fp = fingerprint(remaining[0])
                    if apply:
                        report['storage_bytes_released'] += allocated
                if apply:
                    if sha(canonical) != expected:
                        raise ValueError('Canonical bytes changed after sharing')
                    report['bytes_hashed'] += size
        if apply:
            paths = dict(old['paths'])
            for p, (expected, size) in checked.items():
                paths[str(p)] = dict(sha256=expected, bytes=size, fingerprint=fingerprint(p))
            index_path.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(index_path, dict(schema='completed.output.index.v1', paths=paths,
                                         policy='Only read-only unchanged full filesystem fingerprints may reuse prior full SHA checks.'))
            report['index'] = binding(index_path)
        report['status'] = 'PASS_INCREMENTAL_HARDLINKED_OUTPUTS' if apply else 'PASS_INCREMENTAL_READ_ONLY_PLAN'
    except BaseException as e:
        report.update(status='FAILED_PRESERVE_JOURNAL', error=type(e).__name__+': '+str(e))
        raise
    finally:
        report['finished_unix'] = time.time()
        (output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('registry', 'index', 'output', 'controller-lock', 'gpu-lock'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--controller-lock-held', action='store_true', help='Only for a SHA-bound controller maintenance child; verify the serial lock is already held')
    a = p.parse_args()
    with ExitStack() as stack:
        controller = stack.enter_context(a.controller_lock.open('a+'))
        if a.controller_lock_held:
            try:
                fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(controller, fcntl.LOCK_UN)
                raise ValueError('Declared inherited controller lock is not held')
        else:
            fcntl.flock(controller, fcntl.LOCK_EX)
        gpu = stack.enter_context(a.gpu_lock.open('a+'))
        fcntl.flock(gpu, fcntl.LOCK_EX)
        registry = json.loads(a.registry.read_text())
        arrays, proofs = completed_arrays([Path(x) for x in registry['queue_states'] if Path(x).is_file()])
        result = deduplicate(arrays, a.index, a.output, a.apply)
        (a.output/'discovery.json').write_text(json.dumps(dict(registry=binding(a.registry), completed_receipts=proofs, array_paths=len(arrays)),indent=2)+'\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
