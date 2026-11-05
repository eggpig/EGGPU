#!/usr/bin/env python3
"""Owner-only GPU semantics for exact static SSSP zero/empty domains.

This is qualification, never a performance sample or a formal target proof.
The normal frozen native worker still supplies D/U x P qualification receipts.
"""
from pathlib import Path
import fcntl
import importlib
import importlib.util
import json
import os
import sys
import numpy as np
import native_worker as worker


def main():
    args = worker.parser().parse_args()
    if not args.execute_gpu:
        raise ValueError('Only the primary experiment owner may pass --execute-gpu')
    args.algorithm, args.dataset, args.phase, args.experiment = 'sssp', 'TINY_D', 'tiny', 'E2'
    if args.sssp_encoding != 'u8_exact_u32_exact':
        raise ValueError('This qualifier specifically checks the exact compact representation')
    worker.configure(args)
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=','.join(args.uuids),
                      EGGPU_COMM_BACKEND=args.communication, EGGPU_SSSP_WEIGHT_CERTIFICATION='full',
                      EGGPU_DYNAMIC_WEIGHT_ROUTE='persistent_full')
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status='RUNNING', gpu_executed=False, performance_evidence=False,
                  purpose='Independent static numeric-domain semantics', count=args.count, cases=[])
    worker.shared.save(args.output / 'receipt.json', record)
    try:
        with args.lock_file.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            record['admission'] = worker.admission(args, dict(n=16, m=3))
            entry, identity = worker.epoch(args)
            worker.shared.save(args.output / 'epoch.json', entry)
            record.update(epoch_sha256=identity, epoch=worker.shared.binding(args.output / 'epoch.json'))
            sys.path.insert(0, str(args.release.resolve()))
            sys.path.insert(1, str(args.native_dso.resolve().parent))
            from easygraph.gpu import Session
            cpp = importlib.import_module('cpp_easygraph')
            if Path(cpp.__file__).resolve() != args.native_dso.resolve():
                raise ValueError('Loaded DSO differs from the exact qualification binding')
            script = args.release / 'scripts/gpu/qualification_inputs.py'
            spec = importlib.util.spec_from_file_location('static_numeric_qualification_inputs', script)
            qi = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(qi)
            cases = [('zero_and_255', 16, np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32), [0., 255., 0.]),
                     ('all_isolates', 16, np.empty((0, 2), dtype=np.int32), [])]
            if args.count == 1:
                cases.append(('single_vertex_no_edges', 1, np.empty((0, 2), dtype=np.int32), []))
            retained = []
            for name, n, edges, weights in cases:
                raw = qi.canonical(n, edges, np.asarray(weights, dtype=np.float64))
                manifest, _ = qi.manifest(raw, args.output / name, name, args.count)
                expected = qi.heap_sssp(raw)
                session = None
                row = dict(name=name, n=n, m=len(weights), manifest=worker.shared.binding(manifest['manifest']), calls=[])
                record['cases'].append(row)
                record['gpu_executed'] = True
                try:
                    session = Session.native_static({'g': manifest['manifest']}, devices=tuple(range(args.count)),
                        cpp=cpp, sssp_encoding=args.sssp_encoding, sssp_delta_rule=args.sssp_delta_rule,
                        inventory_mode='recheck')
                    for repeat in range(3):
                        result = session.query('g', 'sssp', {'source': 0})
                        worker.native_profile.observe(result.backend_metrics, 'sssp', args.count)
                        np.testing.assert_array_equal(result.values, expected)
                        if result.values.flags.writeable or not result.values.flags.owndata:
                            raise AssertionError('Public complete host output is not independent readonly storage')
                        if any(np.shares_memory(result.values, old) for old, _ in retained):
                            raise AssertionError('A later query reused a previous output buffer')
                        bits = result.values.view(np.uint64).copy()
                        retained.append((result.values, bits))
                        row['calls'].append(dict(repeat=repeat, quality_pass=True,
                            output=worker.shared.persist_output(args.output, result.values), metrics=result.backend_metrics))
                finally:
                    if session is not None:
                        session.close()
                for values, bits in retained:
                    np.testing.assert_array_equal(values.view(np.uint64), bits)
                row['status'] = 'PASS_STATIC_NUMERIC_DOMAIN'
                worker.shared.save(args.output / 'receipt.json', record)
            record.update(status='PASS_STATIC_NATIVE_NUMERIC_QUALIFICATION',
                          all_full_vectors_exact=True, bitwise_outputs_survive_later_queries_and_close=True)
            if worker.epoch(args)[1] != identity:
                raise ValueError('Static numeric qualification source/epoch changed during execution')
    except BaseException as error:
        record.update(status='FAILED_STATIC_NATIVE_NUMERIC_QUALIFICATION', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        worker.shared.save(args.output / 'receipt.json', record)


if __name__ == '__main__':
    main()
