"""Build CPU input tools and compile the pinned WebGraph exporter from source."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import struct

from download_common import download_file
from fetch import binding, digest

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools', type=Path, required=True)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--java', default='java')
    parser.add_argument('--javac', default='javac')
    parser.add_argument('--jar-cache', type=Path, help='Optional existing official Maven cache; each hash is required')
    args = parser.parse_args()
    args.tools.mkdir(parents=True, exist_ok=True)
    if (args.tools / 'toolchain.json').exists():
        raise ValueError('Use a fresh tools directory; do not silently rebuild an already bound toolchain')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    versions = {}
    for name, executable, option in [('cxx', args.cxx, '--version'), ('java', args.java, '-version'), ('javac', args.javac, '-version')]:
        proc = subprocess.run([executable, option], capture_output=True, text=True, check=True, env=env)
        versions[name] = dict(executable=str(Path(shutil.which(executable) or executable).resolve()),
                              version=(proc.stdout + proc.stderr).strip())
    for name in ['edge_list_to_csr', 'metis_to_csr', 'weights']:
        cmd = [args.cxx, '-O3', '-std=c++17', str(HERE / (name + '.cpp'))]
        if name == 'weights': cmd += ['-fopenmp', '-lcrypto']
        subprocess.run([*cmd, '-o', str(args.tools / name)], check=True, env=env)
    jars = args.tools / 'jars'; jars.mkdir(exist_ok=True)
    deps = json.loads((HERE / 'java_dependencies.json').read_text())
    for spec in deps['jars']:
        target = jars / spec['file']
        cached = args.jar_cache / spec['file'] if args.jar_cache else None
        if not target.exists() and cached and cached.exists():
            if digest(cached) != spec['sha256']: raise ValueError('Cached jar differs: ' + spec['file'])
            shutil.copyfile(cached, target)
        if not target.exists(): download_file(spec['url'], target)
        if target.stat().st_size != spec['bytes'] or digest(target) != spec['sha256']:
            raise ValueError('Official Maven dependency hash mismatch: ' + spec['file'])
    classes = args.tools / 'classes'; classes.mkdir(exist_ok=True)
    subprocess.run([args.javac, '--release', '17', '-d', str(classes), str(HERE / 'LawToCsr.java')], check=True, env=env)
    selftest = args.tools / 'java_self_test'
    subprocess.run([args.java, '-cp', str(classes), 'LawToCsr', '--self-test', str(selftest)], check=True, env=env)
    if struct.unpack('<6i', (selftest / 'offsets.bin').read_bytes()) != (0, 2, 3, 3, 4, 4):
        raise ValueError('Java exporter self-test offsets differ')
    if struct.unpack('<4i', (selftest / 'indices.bin').read_bytes()) != (1, 3, 2, 0):
        raise ValueError('Java exporter self-test complete output differs')
    result = dict(status='PASS_CPU_TOOLCHAIN', gpu_executed=False, versions=versions,
                  dependencies=binding(HERE / 'java_dependencies.json'),
                  source_files={p.name: binding(p) for p in HERE.iterdir() if p.suffix in ('.cpp', '.java')},
                  files={str(p.relative_to(args.tools)): binding(p) for p in args.tools.rglob('*') if p.is_file()})
    (args.tools / 'toolchain.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'tools': str(args.tools)}))


if __name__ == '__main__':
    main()
