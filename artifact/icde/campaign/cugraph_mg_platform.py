"""Target-host observations; no old server paths or UUIDs."""
import hashlib
import importlib.metadata
import os
from pathlib import Path


def resolve_distribution(name):
    candidates = [name]
    if name in ('cugraph', 'pylibcugraph', 'dask-cudf', 'cudf', 'rmm', 'raft-dask'):
        candidates += [name+'-cu13', name+'-cu12']
    elif name == 'cupy':
        candidates += ['cupy-cuda13x', 'cupy-cuda12x']
    installed = []
    for candidate in candidates:
        try:
            installed.append((candidate, importlib.metadata.version(candidate)))
        except importlib.metadata.PackageNotFoundError:
            pass
    if len(installed) != 1:
        raise RuntimeError(f'Expected one installed distribution for {name}: {installed}')
    return installed[0]


def set_cpu_affinity(cpus):
    """Dask-CUDA installs its own NUMA affinity; restore frozen experiment mask."""
    os.sched_setaffinity(0, set(cpus))
    observed=sorted(os.sched_getaffinity(0))
    if observed!=sorted(cpus):raise RuntimeError('Cannot apply declared CPU mask')
    return observed


def worker_binding():
    import cupy as cp
    import pynvml
    pynvml.nvmlInit()
    try:
        current = cp.cuda.runtime.getDevice()
        if current != 0:
            raise RuntimeError("Dask rank must use assigned logical GPU 0")
        bus = cp.cuda.Device(current).pci_bus_id
        handle = pynvml.nvmlDeviceGetHandleByPciBusId(bus)
        identity = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(identity, bytes):
            identity = identity.decode()
        host_pid = int(os.readlink('/proc/self'))
        owned = [p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                 if p.pid == host_pid]
        return dict(pid=os.getpid(), host_pid=host_pid,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    current_device=current, pci_bus_id=bus, uuid=identity,
                    process_device_bytes_at_observation=sum(owned),
                    own_context_observed=bool(owned),
                    cpu_affinity=sorted(os.sched_getaffinity(0)))
    finally:
        pynvml.nvmlShutdown()


def toolkit_probe(expected_versions):
    """Charge imports/probes to setup, use target's installed compatible CUDA stack.

    No inherited custom NVRTC override or copied host shared library is used.
    GPU query qualification is still mandatory on each target and P.
    """
    versions = {name: resolve_distribution(name)[1] for name in expected_versions}
    if versions != expected_versions:
        raise RuntimeError('Target package versions changed after freeze')
    import cupy as cp
    import pylibcugraph  # noqa: F401
    mapped = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        path = Path(fields[5])
        if not path.is_file() or not any(name in path.name for name in
                ('libcudart', 'libnvrtc', 'libcugraph', 'libraft', 'libnccl', 'librmm')):
            continue
        path = path.resolve()
        if str(path) in mapped:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        mapped[str(path)] = digest.hexdigest()
    return dict(versions=versions, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                cuda_driver=cp.cuda.runtime.driverGetVersion(), mapped_libraries=mapped,
                environment={key: os.environ.get(key) for key in
                             ('CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH',
                              'CUDA_VISIBLE_DEVICES')})
