"""Full recovery gate for E4-S. Explicit bindings support archive relocation."""
import hashlib
import math
import protocol


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def same_content(binding, expected):
    return {key: binding[key] for key in ('bytes', 'sha256')} == expected


def verify(e, raw, cell, *, formal):
    """Decode each pinned NPZ and check its raw bytes against measured outputs.

    No lookup of latest runs or inferred sidecar names is permitted. Every
    formal run and both qualifications need an explicit recovery association.
    """
    import numpy as np
    require(cell.get('accept_execution_hash_attestation') is not True,
            'Hash-only attestation is forbidden by the current E4-S protocol')
    require(raw.get('quality_pass') is True and all(c['quality'].get('passed') is True and
            c['quality'].get('exact') is True for c in raw['checks']), 'Complete exact output quality did not pass')
    helper = cell['archive_helper']
    e.verify(helper)
    require(same_content(helper, protocol.PROVENANCE['e4s_recoverable_outputs_v1/archive_outputs.py']),
            'Recovery helper is not the selected frozen producer')
    matches = [entry for entry in cell.get('recoverable_outputs', []) if e.read(entry['result']) == raw]
    require(len(matches) == 1, 'Each E4-S measured receipt needs exactly one explicit recoverable archive')
    entry = matches[0]; result_binding = entry['result']; sidecar = e.read(entry['archive'])
    require(sidecar.get('schema') == 'e4s.recoverable.output.v1' and
            sidecar.get('status') == 'PASS_BYTE_IDENTICAL_RECOVERABLE_E4S_OUTPUTS', 'Recovery did not pass')
    require(sidecar['original_result'] == result_binding and sidecar['source'] == helper,
            'Recovery is bound to another measured result or helper')
    require(sidecar.get('post_measurement') is True and sidecar.get('measured_intervals_unchanged') is True,
            'Recovery changed the measured interval')
    require(sidecar['epoch_sha256'] == raw['epoch_sha256'] and sidecar['input_sha256'] == raw['input_sha256'] and
            sidecar['source_sequence'] == raw['source_sequence'], 'Recovery input/epoch/source sequence differs')
    require(len(sidecar['vectors']) == len(raw['checks']) == raw['batch'], 'Recovery omitted batch outputs')
    e.verify(raw['reference_implementation'])
    if raw['variant'] == 'eggpu':
        require(raw['measurements'].get('all_outputs_retained_through_close') is True and
                all(c.get('survives_session_close') is True for c in raw['checks']), 'EGGPU owning-output lifetime not verified')
    for index, (vector, check) in enumerate(zip(sidecar['vectors'], raw['checks'])):
        require(vector['sequence'] == check['sequence'] == index and
                vector['source'] == check['source'] == raw['source_sequence'][index] and
                vector['values'] == check['values'], 'Recovery vector identity differs')
        require(vector.get('byte_identical') is True and vector.get('dtype') == '<f8' and vector.get('order') == 'C' and
                vector['restored_raw_sha256'] == vector['measured_output_sha256'] == check['output_sha256'],
                'Recovery is not a byte-identical measured FP64 vector')
        with np.load(e.verify(vector['object']), allow_pickle=False) as archive:
            require(archive.files == ['reference'], 'Unexpected recoverable archive members')
            values = archive['reference']
        require(values.dtype == np.dtype('<f8') and values.shape == (check['values'],) and values.flags.c_contiguous,
                'Recovered vector dtype/shape/order differs')
        actual = hashlib.sha256(memoryview(values).cast('B')).hexdigest()
        require(actual == check['output_sha256'], 'Decoded bytes differ from measured output hash')
        metadata = e.read(vector['reference_metadata']); key = metadata['key']
        require(digest(key) == vector['reference_key_sha256'] == check['quality']['reference']['key_sha256'],
                'Recovered independent-reference key differs')
        require(key['schema'] == 'state.reference.v2' and key['function'] == 'sssp' and key['version'] == 0 and
                key['source'] == vector['source'] and key['n'] == vector['values'], 'Recovered reference task differs')
        require(metadata['array'] == {k: vector['object'][k] for k in ('bytes', 'sha256')} and
                key['reference_source'] == {k: raw['reference_implementation'][k] for k in ('bytes', 'sha256')},
                'Reference producer/archive binding differs')
    if formal:
        require(raw.get('status') == 'PASS_E4S_FORMAL' and raw.get('phase') == 'formal' and
                raw.get('performance_eligible') is True, 'Qualification cannot supply formal E4-S performance')
        post = e.read(entry['maintenance'])
        require(post.get('status') == 'PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE' and
                post.get('archive_status') == post.get('original_maintenance_status') == 'PASS' and
                post['original_result'] == result_binding and post['archive_receipt'] == entry['archive'],
                'Mandatory recovery and original maintenance did not pass for this result')
        spec = e.read(post['attempt_spec'])
        require(spec['result'] == result_binding['path'] and spec['archive_helper'] == helper and
                post['source'] in spec['bindings'], 'Maintenance spec/helper/source differs')
        for binding in spec['bindings']: e.verify(binding)
        require(same_content(post['source'], protocol.PROVENANCE['e4s_recoverable_outputs_v1/maintain_e4s_outputs.py']),
                'Maintenance wrapper source is not the frozen producer')
        original = e.read(post['original_maintenance_receipt'])
        require(post['original_maintenance_receipt']['path'] == spec['original_maintenance']['receipt'] and
                original.get('status') in spec['original_maintenance']['expected_statuses'],
                'Original maintenance did not satisfy the frozen attempt')
        seconds = raw['measurements']['first_complete_seconds']
        require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 3600,
                'Invalid complete-task interval')
    return {'vectors': len(sidecar['vectors']), 'status': 'FULL_RECOVERABLE_VECTORS_DECODED_AND_VERIFIED'}
