"""Print exact payload sizes and conservative CPU admission budgets; no GPU calls."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import shutil

HERE = Path(__file__).resolve().parent


def budgets(dataset, source):
    n = dataset['canonical_vertices']
    known = dataset.get('canonical_stored_arcs')
    m = known if known is not None else dataset['source_stored_arcs']
    return dict(dataset=dataset['abbreviation'], vertices=n, stored_arcs_bound=m,
                canonical_size_exact=known is not None,
                offsets_bytes=4 * (n + 1), indices_bytes=4 * m, weights_bytes=8 * m,
                canonical_payload_bytes=4 * (n + 1) + 12 * m,
                compressed_source_bytes=sum(f['bytes'] for f in source['files']),
                # Includes graph/SCC/transpose working arrays and a 2 GiB margin.
                # This is an admission bound, not a measured high-water mark.
                host_admission_bytes=64 * m + 128 * n + (2 << 30))


def headroom():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].split()[0]) * 1024
    candidates = {Path('/sys/fs/cgroup')}
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            path = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
            candidates.update([path, *path.parents])
    observed = []
    for path in sorted(candidates):
        if not (path / 'memory.max').exists(): continue
        maximum = (path / 'memory.max').read_text().strip()
        current = int((path / 'memory.current').read_text())
        if maximum != 'max':
            remaining = max(0, int(maximum) - current); available = min(available, remaining)
            observed.append(dict(path=str(path), max_bytes=int(maximum), current_bytes=current))
    return dict(available_bytes=available, finite_cgroup_limits=observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--datasets', default='EP,LJ,US,OR,IN,UK,TW')
    parser.add_argument('--enforce', action='store_true')
    args = parser.parse_args(); args.root.mkdir(parents=True, exist_ok=True)
    registry = {d['abbreviation']: d for d in json.loads((HERE / 'dataset_registry.json').read_text())['datasets']}
    sources = json.loads((HERE / 'source_catalog.json').read_text())['sources']
    table = [budgets(registry[name], sources[name]) for name in args.datasets.split(',')]
    jars = sum(item['bytes'] for item in json.loads((HERE / 'java_dependencies.json').read_text())['jars'])
    disk_bound = sum(row['canonical_payload_bytes'] + row['compressed_source_bytes'] for row in table) + jars + (6 << 30)
    host_bound = max(row['host_admission_bytes'] for row in table)
    memory = headroom()
    result = dict(status='CPU_PREFLIGHT_PLAN', gpu_executed=False, datasets=table,
                  jars_bytes=jars, disk_fresh_directory_bound_bytes=disk_bound,
                  disk_bound_note='Stream compressed archives; no expanded text copy. Includes all retained sources, canonical arrays, pinned jars and 6 GiB reserve; excludes a separately installed JDK, Python environment, benchmark outputs and old versions.',
                  host_admission_bound_bytes=host_bound, memory=memory,
                  free_disk_bytes=shutil.disk_usage(args.root).free,
                  python_dependencies={name: importlib.metadata.version(name) for name in ('numpy', 'scipy')})
    if args.enforce and (result['free_disk_bytes'] < disk_bound or memory['available_bytes'] < host_bound):
        result['status'] = 'INSUFFICIENT_FRESH_INPUT_ADMISSION'
        print(json.dumps(result, indent=2)); raise SystemExit(2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
