"""Explicit future-only residual-certificate audit over the unchanged measured worker."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import sys


def read(path):
    return json.loads(Path(path).read_text())


def checked(item):
    path=Path(item['path'])
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    if path.stat().st_size != item['bytes'] or h.hexdigest()!=item['sha256']:
        raise ValueError('Changed immutable file: '+str(path))
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--audit-amendment',type=Path,required=True)
    args=parser.parse_args()
    selection=read(args.selection)
    policy_binding=selection['measurement_policy']
    policy_record=read(checked(policy_binding))
    policy_module_spec=importlib.util.spec_from_file_location('_state_explicit_pr_policy',checked(policy_record['guard']))
    policy_guard=importlib.util.module_from_spec(policy_module_spec);policy_module_spec.loader.exec_module(policy_guard)
    if policy_record['environment']!=policy_guard.EXPECTED_ENV:raise ValueError('Unknown state policy')
    spec=read(args.spec)
    amendment_binding=dict(path=str(args.audit_amendment.resolve()),bytes=args.audit_amendment.stat().st_size,sha256=hashlib.sha256(args.audit_amendment.read_bytes()).hexdigest())
    amendment=read(checked(amendment_binding))
    if amendment['method']!='state-pr-full-residual-certificate-v1':raise ValueError('Unknown audit amendment')
    for binding in amendment['files'].values():checked(binding)
    if checked(amendment['files']['launch_certificate.py'])!=Path(__file__).resolve():raise ValueError('Wrong audit launcher')
    assert str(Path(sys.executable).absolute())==selection['python_invocation']
    freeze=read(checked(selection['source_freeze']))
    assert freeze['source_tree_sha256']==selection['source_tree_sha256']
    modspec=importlib.util.spec_from_file_location('_state_first_pass_source_guard',selection['source_guard'])
    module=importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(module)
    module.source_files(Path(selection['source_root']),freeze)
    checked(selection['native_dso']);checked(selection['native_build_receipt'])
    runner=Path(selection['runner'])
    for name,binding in read(checked(selection['runner_freeze']))['files'].items():
        checked(dict(binding,path=str(runner/name)))
    checked(spec['inputs'])
    assert checked(spec['selection'])==args.selection.resolve()
    # Reuse the already reviewed full-SHA identity cache. It only replaces
    # file hashing; each complete numerical output audit still executes.
    helper=checked(selection['hash_cache_helper'])
    helper_spec=importlib.util.spec_from_file_location('_state_first_pass_hash_cache',helper)
    helper_module=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper_module)
    cache=helper_module.BindingCache(selection['hash_cache'])
    sys.path.insert(0,str(runner))
    import state_protocol as protocol
    protocol.sha=lambda path:cache.binding(path)['sha256']
    protocol.bind=cache.binding
    sys.argv=[str(runner/'state_worker.py'),*spec['worker_arguments']]
    output=Path(spec['worker_arguments'][spec['worker_arguments'].index('--output')+1])
    checked(dict(amendment['original_reference'],path=str(runner/'state_reference.py')))
    worker_spec=importlib.util.spec_from_file_location('_state_original_measured_worker',runner/'state_worker.py')
    worker=importlib.util.module_from_spec(worker_spec);worker_spec.loader.exec_module(worker)
    auditor=[]
    def certified_quality(values,row,version,function,source,cache=None):
        # Delay all new NumPy/SciPy imports and certificate setup until the
        # original post-workflow audit hook. Environment/init ordering stays
        # exactly as in the original measured worker.
        import state_reference as reference
        if Path(reference.__file__).resolve()!=(runner/'state_reference.py').resolve():raise ValueError('Wrong original reference source')
        if spec['phase']=='tiny':
            return reference.output_quality(values,row,version,function,source,cache)
        if not auditor:
            module_spec=importlib.util.spec_from_file_location('_state_pr_certificate',checked(amendment['files']['pr_certificate.py']))
            module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module);auditor.append(module)
        return auditor[0].output_quality(reference,values,row,version,function,source,cache)
    worker.output_quality=certified_quality
    original_save=protocol.save
    def save_with_amendment(path,value):
        if Path(path).resolve()==(output/'receipt.json').resolve():
            value=dict(value,audit_amendment=dict(manifest=amendment_binding,method=amendment['method'],
                original_measurement_runner=selection['runner_freeze'],measurement_source_and_timers_unchanged=True,
                thresholds_unchanged=True,reference_distance_kind='measured original reference or certified upper bound, identified per output'))
            if 'independent_reference' in value:
                value['independent_reference']=dict(value['independent_reference'],audit_amendment=amendment_binding,
                    quality_scope='Every output audited in full; PR full residual and mass checked; original reference comparison or rigorous residual certificate; bound is not a measured difference')
            value['measurement_policy']=dict(binding=policy_binding,environment=policy_guard.EXPECTED_ENV,
                old_policy_observations_reused=False,tiny_uses_original_independent_reference=spec['phase']=='tiny')
            if value.get('status') in ('PASS_STATE_GPU_QUALIFICATION','PASS_STATE_COMPLETE_WORKFLOW','PASS_PHYSICAL_BYTE_PILOT'):
                try:value['measurement_policy']['actual']=policy_guard.inspect(value)
                except ValueError as error:
                    value.update(status='FAIL_STATE_POLICY',performance_eligible=False,error=str(error))
                    original_save(path,value)
                    raise
        return original_save(path,value)
    protocol.save=save_with_amendment
    try:
        worker.execute(worker.parser().parse_args())
    finally:
        if output.is_dir():
            with (output/'hash_cache_usage.json').open('x') as stream:
                json.dump(dict(helper=selection['hash_cache_helper'],cache=cache.report(),
                    numerical_thresholds_unchanged=True,audit_method_amended=True,audit_amendment=amendment_binding),stream,indent=2)
        cache.close()
