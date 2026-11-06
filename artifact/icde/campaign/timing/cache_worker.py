"""Run an unchanged worker with a shared once-per-file-identity SHA cache.

Only filesystem binding helpers are replaced; algorithm, numerical checks,
measurement intervals, arguments and environment remain unchanged. Parent/source
children share the cache. Each process leaves an explicit wrapper sidecar.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
import stat
import subprocess
import sys
import time


def full_binding(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def fingerprint(path):
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Hash binding requires a regular file: ' + str(path))
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class BindingCache:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS sha_bindings (path TEXT PRIMARY KEY, identity TEXT NOT NULL, sha256 TEXT NOT NULL)')
        self.memory = {}
        self.hits = self.misses = self.hashed_bytes = 0
        self.hash_seconds = 0.0

    def binding(self, path):
        path = Path(path).resolve()
        name = str(path)
        for _ in range(3):
            before = fingerprint(path)
            identity = json.dumps(before, separators=(',', ':'))
            row = self.memory.get(name)
            if row is None or row[0] != identity:
                row = self.db.execute('SELECT identity,sha256 FROM sha_bindings WHERE path=?', (name,)).fetchone()
            if row is not None and row[0] == identity:
                # Catch ordinary replacement/edit between the initial stat and
                # cache lookup; content changes always invalidate the identity.
                if fingerprint(path) != before:
                    continue
                self.memory[name] = row
                self.hits += 1
                return dict(path=name, bytes=before[2], sha256=row[1])
            started = time.perf_counter()
            actual = full_binding(path)
            self.hash_seconds += time.perf_counter() - started
            if fingerprint(path) != before:
                continue
            self.db.execute('INSERT OR REPLACE INTO sha_bindings VALUES (?,?,?)', (name, identity, actual['sha256']))
            self.memory[name] = (identity, actual['sha256'])
            self.misses += 1
            self.hashed_bytes += before[2]
            return actual
        raise RuntimeError('File changed while establishing its binding: ' + name)

    def report(self):
        return dict(database=str(self.path), cache_hits=self.hits, full_hashes=self.misses,
                    full_hash_bytes=self.hashed_bytes, full_hash_seconds=self.hash_seconds,
                    identity_fields=['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'],
                    policy='Full SHA on first observation or any ordinary file-identity change; unchanged files reuse exact original SHA.')

    def close(self):
        self.db.close()


def rewrite_child(command, worker, wrapper, cache, expected):
    if not isinstance(command, (list, tuple)):
        return command
    values = list(command)
    # Only the same source worker's subprocess entry is routed through this
    # wrapper. Native executables, library subprocesses and other tools stay exact.
    indices = [i for i,x in enumerate(values) if isinstance(x, str) and x == str(worker)]
    if len(indices) != 1:
        return command
    i = indices[0]
    return values[:i] + [str(wrapper), '--worker', str(worker), '--worker-sha256', expected,
                        '--cache', str(cache), '--'] + values[i+1:]


def run(worker, expected, cache_path, arguments):
    worker = Path(worker).resolve()
    original_identity = full_binding(worker)
    if original_identity['sha256'] != expected:
        raise ValueError('Original worker changed')
    wrapper = Path(__file__).resolve()
    wrapper_identity = full_binding(wrapper)
    cache = BindingCache(cache_path)
    old_argv = sys.argv
    old_path = list(sys.path)
    original_popen = subprocess.Popen
    output = None
    if '--output' in arguments:
        output = Path(arguments[arguments.index('--output')+1]).resolve()
    status = 'FAILED'
    patched = []
    started = time.time()
    try:
        sys.argv = [str(worker), *arguments]
        sys.path.insert(0, str(worker.parent))
        spec = importlib.util.spec_from_file_location('_first_pass_original_worker', worker)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        candidates = [module]
        for attribute in ('shared','common'):
            value = getattr(module, attribute, None)
            if value is not None:
                candidates.append(value)
        seen = set()
        for value in candidates:
            if id(value) in seen:
                continue
            seen.add(id(value))
            if callable(getattr(value, 'binding', None)):
                patched.append(dict(module=value.__name__, file=str(getattr(value,'__file__','')), function='binding'))
                value.binding = cache.binding
        if not patched:
            raise ValueError('No supported binding helper in the original worker')
        def cached_child(command, *args, **kwargs):
            return original_popen(rewrite_child(command, worker, wrapper, cache.path, expected), *args, **kwargs)
        subprocess.Popen = cached_child
        module.main()
        status = 'COMPLETE'
    finally:
        subprocess.Popen = original_popen
        sys.argv = old_argv
        sys.path[:] = old_path
        record = dict(schema='eggpu.first_pass.hash_cache.v1', status=status, original_worker=original_identity,
                      wrapper=wrapper_identity, pid=os.getpid(), started_unix=started, finished_unix=time.time(),
                      patched_helpers=patched, algorithm_modified=False, numerical_checks_modified=False,
                      measured_intervals_modified=False, worker_arguments=arguments, cache=cache.report())
        if output is not None and output.is_dir():
            sidecar = output/'hash_cache_usage.json'
            with sidecar.open('x') as stream:
                json.dump(record,stream,indent=2);stream.write('\n')
        cache.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--worker-sha256',required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    run(args.worker,args.worker_sha256,args.cache,arguments)


if __name__ == '__main__':
    main()
