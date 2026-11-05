"""Owner-only GPU semantics: unchanged CSR fixture plus live ownership checks.

The owner must hold the shared GPU lock and apply the recorded resource gate.
This file is import-safe and never executes GPU work without --execute-gpu.
"""
from pathlib import Path
import argparse
import hashlib
import json


def run(device):
    import numpy as np
    from easygraph.gpu._bridge import _Bridge
    from qualify_public_csr import run as csr_semantics
    original = _Bridge._own_result
    held, observations = [], []
    def observe(self, graph_id, algorithm, values, begin, metrics, parameters, **kwargs):
        result = original(self, graph_id, algorithm, values, begin, metrics, parameters, **kwargs)
        if algorithm == 'wcc':
            ownership = result.backend_metrics['wcc_public_output']
            assert ownership['ownership'] == 'adopt_certified_arena_download'
            assert ownership['canonical_output_write_bytes'] == 0
            avoided = ownership['canonical_copy_bytes_avoided']
        else:
            key = 'pr' if algorithm == 'weighted_pr' else algorithm
            ownership = result.backend_metrics[key + '_public_output_ownership']
            assert ownership['route'] == 'adopt_arena_download'
            assert ownership['bridge_copy_bytes'] == 0
            avoided = ownership['bridge_copy_bytes_avoided']
        assert avoided == result.values.nbytes
        assert result.values.dtype == np.float64 and result.values.flags.owndata
        assert not result.values.flags.writeable
        assert not any(np.shares_memory(result.values, old.values) for old, _ in held)
        held.append((result, result.values.view(np.uint64).copy()))
        observations.append(dict(algorithm=algorithm, values=len(result.values), ownership=ownership,
                                 output_sha256=hashlib.sha256(result.values.tobytes()).hexdigest()))
        return result
    _Bridge._own_result = observe
    try:
        reference = csr_semantics(device)
    finally:
        _Bridge._own_result = original
    assert len(held) == len(reference['checks']) == 33
    for result, bits in held:
        np.testing.assert_array_equal(result.values.view(np.uint64), bits)
        assert not result.values.flags.writeable
    source = Path(__file__).resolve().parents[2]
    files = ['easygraph/gpu/_arena_output.py', 'easygraph/gpu/_arena/gpu_runtime.py',
             'easygraph/gpu/_bridge.py', 'easygraph/gpu/_core.py',
             'tests/gpu/qualify_arena_output_ownership.py', 'tests/gpu/qualify_public_csr.py']
    return dict(status='PASS_GPU_ARENA_OUTPUT_OWNERSHIP', performance_eligible=False,
                gpu_device=device, values_bitwise_unchanged_after_updates_queries_close=True,
                checks=observations, complete_semantic_qualification=reference,
                source_bindings={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-gpu', action='store_true', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('A fresh qualification output is required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='FAILED', performance_eligible=False)
    try:
        result = run(args.device)
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks': len(result['checks'])}))


if __name__ == '__main__':
    main()
