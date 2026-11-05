"""No GPU: runner policy propagation and source-epoch separation."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import native_worker as worker


@pytest.mark.parametrize('count', [1, 2, 4, 8])
def test_static_policy_is_the_same_across_native_gpu_counts(count):
    args = worker.parser().parse_args(['--algorithm', 'sssp', '--dataset', 'TINY_D', '--phase', 'tiny',
        '--count', str(count), '--gpu-uuids', ','.join('GPU-' + str(x) for x in range(count))])
    worker.configure(args)
    assert args.sssp_encoding == 'u8_exact_u32_exact'
    forwarded = worker.forward_options(args)
    assert forwarded[forwarded.index('--sssp-encoding') + 1] == args.sssp_encoding
    assert forwarded[forwarded.index('--communication') + 1] == 'host_staging'


def test_explicit_encoding_changes_production_epoch(tmp_path, monkeypatch):
    args = worker.parser().parse_args(['--algorithm', 'sssp', '--dataset', 'TINY_D', '--phase', 'tiny',
        '--count', '1', '--gpu-uuids', 'GPU-a'])
    args.release = tmp_path / 'release'
    args.release.mkdir()
    args.native_dso = tmp_path / 'native.so'
    args.native_dso.write_bytes(b'CPU epoch fixture, never loaded')
    worker.configure(args)
    monkeypatch.setattr(worker.shared, 'source_epoch', lambda proxy: (dict(files={}, release_sources={}, environment={}), None))
    monkeypatch.setattr(worker, 'runtime_dependencies', lambda path: {})
    monkeypatch.setattr(worker.shared, 'cgroup_memory', lambda: dict(cgroup_limits=[]))
    first, digest = worker.epoch(args)
    args.sssp_encoding = 'f64'
    second, different = worker.epoch(args)
    assert digest != different
    assert first['sssp_encoding'] == 'u8_exact_u32_exact' and second['sssp_encoding'] == 'f64'
    assert first['sssp_single_owner_execution'] == 'off'
    assert first['sssp_session'] == 'Session.native_static'
    assert first['native_route'] == 'static_immutable'
