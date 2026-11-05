"""Fetch/cache all seven frozen original sources with per-dataset locked receipts."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import time

from download_common import download_file

HERE = Path(__file__).resolve().parent


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def binding(path):
    path = Path(path)
    return dict(path=str(path.absolute()), bytes=path.stat().st_size, sha256=digest(path))


def merge(root, dataset, record):
    directory = root / 'source'; directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'fetch_state.json'
    with (directory / 'fetch_state.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(target.read_text()) if target.exists() else {'datasets': {}}
        state.setdefault('datasets', {})[dataset] = record
        temporary = directory / ('fetch_state.' + dataset + '.partial')
        with temporary.open('w') as stream:
            json.dump(state, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(target)


def validate_payload(path, spec):
    observed = binding(path)
    if observed['bytes'] != spec['bytes']:
        raise ValueError('Frozen compressed source size differs: ' + path.name)
    if spec.get('sha256') and observed['sha256'] != spec['sha256']:
        raise ValueError('Frozen compressed source SHA256 differs: ' + path.name)
    if spec.get('official_md5') and digest(path, 'md5') != spec['official_md5']:
        raise ValueError('Publisher MD5 differs: ' + path.name)
    return observed


def fetch(root, dataset, source, *, law_http=False, cache_import=None, connections=4):
    directory = root / 'source' / dataset; directory.mkdir(parents=True, exist_ok=True)
    record = dict(status='RUNNING', dataset=dataset, protocol='fresh-source-v4', files=[],
                  source_spec=source, source_catalog=binding(HERE / 'source_catalog.json'),
                  fetch_code={name: binding(HERE / name) for name in
                              ('fetch.py', 'download_common.py', 'range_download.py')},
                  gpu_executed=False, started_unix=time.time())
    with (directory / 'fetch_dataset.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior_path = directory / 'source_receipt.json'
        if prior_path.exists():
            prior = json.loads(prior_path.read_text())
            if prior.get('status') == 'PASS_SOURCE_FILES':
                if prior.get('source_spec') != source:
                    raise ValueError('Qualified source specification changed; use a new source attempt')
                previous = {Path(item['path']).name: item for item in prior['files']}
                if set(previous) != {item['name'] for item in source['files']}:
                    raise ValueError('Qualified source file inventory changed')
                for spec in source['files']:
                    observed = validate_payload(directory / spec['name'], spec)
                    if any(observed[k] != previous[spec['name']][k] for k in ('bytes', 'sha256')):
                        raise ValueError('Qualified cached source changed')
                # A canonical receipt may bind this source receipt's exact bytes.
                # Keep those bytes stable on an idempotent cache hit or relocation.
                merge(root, dataset, prior)
                return prior
        merge(root, dataset, record)
        try:
            needed = sum(f['bytes'] for f in source['files'] if not (directory / f['name']).exists())
            if shutil.disk_usage(directory).free < needed + (6 << 30):
                raise RuntimeError('Insufficient source-cache disk plus 6 GiB reserve')
            for spec in source['files']:
                path = directory / spec['name']; url = spec['url']
                if law_http and source['format'] == 'law_webgraph':
                    url = url.replace('https://data.law.di.unimi.it/', 'http://data.law.di.unimi.it/', 1)
                if not path.exists() and cache_import and spec['name'] in cache_import:
                    cached = Path(cache_import[spec['name']]); validate_payload(cached, spec)
                    try: os.link(cached, path)
                    except OSError: shutil.copyfile(cached, path)
                    acquisition = {'status': 'VALIDATED_EXISTING_CACHE', 'path': str(cached)}
                else:
                    acquisition = download_file(url, path, connections=connections)
                item = validate_payload(path, spec)
                item.update(url=url, acquisition=acquisition)
                record['files'].append(item); merge(root, dataset, record)
            if source['format'] == 'law_webgraph':
                lines = (directory / (source['basename'] + '.md5sums')).read_text().splitlines()
                official = {}
                for line in lines:
                    match = re.fullmatch(r'([0-9a-fA-F]{32})\s+\*?([^\s]+)', line.strip())
                    if match: official[match[2]] = match[1].lower()
                for suffix in ('.graph', '.properties'):
                    path = directory / (source['basename'] + suffix)
                    if path.name not in official or digest(path, 'md5') != official[path.name]:
                        raise ValueError('Full payload differs from downloaded publisher MD5 list')
                    next(item for item in record['files'] if Path(item['path']).name == path.name)['official_md5'] = official[path.name]
            record.update(status='PASS_SOURCE_FILES', completed_unix=time.time())
            if dataset == 'US':
                record.update(identity_basis=source['identity_basis'], suitesparse_byte_hash_verified=False)
        except BaseException as error:
            record.update(status='FAILED', error=type(error).__name__ + ': ' + str(error)); raise
        finally:
            (directory / 'source_receipt.json').write_text(json.dumps(record, indent=2) + '\n')
            merge(root, dataset, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--datasets', default='EP,LJ,US,OR,IN,UK,TW')
    parser.add_argument('--connections', type=int, choices=(4, 8), default=4)
    parser.add_argument('--law-http', action='store_true', help='Explicit official HTTP transport; publisher MD5 remains required')
    parser.add_argument('--cache-import', type=Path, help='JSON filename to previously downloaded local file mapping')
    args = parser.parse_args()
    catalog = json.loads((HERE / 'source_catalog.json').read_text())['sources']
    cached = json.loads(args.cache_import.read_text()) if args.cache_import else None
    for name in args.datasets.split(','):
        result = fetch(args.root, name, catalog[name], law_http=args.law_http, cache_import=cached, connections=args.connections)
        print(json.dumps({'dataset': name, 'status': result['status']}), flush=True)


if __name__ == '__main__':
    main()
