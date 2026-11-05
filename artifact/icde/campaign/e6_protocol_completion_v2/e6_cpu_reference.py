"""Explicit, fail-closed consumption of the unchanged independent CPU oracle.

The bridge changes only the execution menu and 1800 -> 3600 second limit.
Original producer receipts, outputs, formulas and numerical gates stay intact.
"""
from pathlib import Path
import json
import legacy_cpu_reference as legacy

SCHEMA = 'eggpu.e6.execution-contract-reference-bridge/v1'
HERE = Path(__file__).resolve().parent
binding, check = legacy.binding, legacy.check


def validate_contract(legacy_contract, consumer_contract):
    frozen = json.loads((HERE / 'legacy_contract.json').read_text())
    if legacy_contract != frozen:
        raise ValueError('Independent oracle is not the exact frozen legacy contract')
    # Import lazily: e6_worker itself imports this consumer.
    from e6_worker import CONTRACT
    if consumer_contract != CONTRACT:
        raise ValueError('Wrong candidate reference consumer contract')
    delta = {k for k in set(frozen) | set(consumer_contract)
             if frozen.get(k) != consumer_contract.get(k)}
    if delta != {'ablation', 'timeout_seconds'} or frozen['timeout_seconds'] != 1800 or consumer_contract['timeout_seconds'] != 3600:
        raise ValueError('Only the reviewed execution-menu and timeout bridge is allowed')
    return True


def manifest_record(legacy_path, contract):
    old = json.loads(Path(legacy_path).read_text())
    validate_contract(old['protocol'], contract)
    return dict(schema=SCHEMA, protocol=contract, legacy_manifest=binding(legacy_path),
                changes=['ablation', 'timeout_seconds'], gpu_executed=False,
                performance_eligible=False,
                meaning='Reuses identical independent numerical oracle bytes; does not transfer any GPU qualification or timing.')


def validate_manifest(path, contract, *, verify_payloads=False):
    bridge = json.loads(Path(path).read_text())
    if (bridge.get('schema') != SCHEMA or bridge.get('protocol') != contract or
            bridge.get('changes') != ['ablation', 'timeout_seconds'] or
            bridge.get('gpu_executed') is not False or bridge.get('performance_eligible') is not False):
        raise ValueError('Wrong explicit E6 execution-contract reference bridge')
    old_path = check(bridge['legacy_manifest'])
    old = json.loads(old_path.read_text())
    validate_contract(old['protocol'], contract)
    if old.get('schema') != legacy.SCHEMA or old.get('gpu_executed') is not False or old.get('performance_eligible') is not False:
        raise ValueError('Wrong legacy oracle identity')
    if verify_payloads:
        for item in old['producer']['worker'], old['producer']['reused']:
            check(item)
        if {r['dataset'] for r in old['references']} != {'E6_TINY_U', 'EP', 'LJ', 'OR'} or len(old['references']) != 4:
            raise ValueError('Oracle bridge must retain all four fixed inputs exactly once')
        for row in old['references']:
            check(row['receipt']); check(row['output'])
    return old_path, old


def load(graph, manifest_path, contract, reused_path):
    old_path, old = validate_manifest(manifest_path, contract)
    # The original consumer still checks producer/source/input/runtime identity,
    # every payload hash, all seven fields and full vector domains.
    return legacy.load(graph, old_path, old['protocol'], reused_path)
