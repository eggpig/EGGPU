"""Admit only the explicit new policy epoch; old general data stay historical."""
from pathlib import Path
from prepare import checked,read,load,SOURCE
EXPECTED={'EGMGPU_PAGERANK_HOST_TRANSPOSE':'auto','EGMGPU_PAGERANK_HOST_WEIGHT_STORAGE':'implicit',
    'EGMGPU_PAGERANK_DEVICE_METADATA':'pull_only','EGMGPU_PAGERANK_TRANSPOSE_ENGINE':'cpu','EGMGPU_PAGERANK_VIEW':'transpose'}

def checked_policy(selection):
    policy=read(checked(selection['measurement_policy']))
    if policy['source_tree_sha256']!=SOURCE or policy['environment']!=EXPECTED or policy['old_measurements_reused'] is not False:
        raise ValueError('Fresh explicit source71 PR policy required')
    return policy,load('_additive_policy_guard',checked(policy['guard']))

def inspect_pr(record):
    environment=record['configuration']['environment']
    if any(environment.get(key)!=value for key,value in EXPECTED.items()):raise ValueError('PR control fresh policy environment differs')
    queries=[event for event in record['events'] if event['kind']=='query']
    expected=dict(preparation_method='transpose',host_weight_storage='implicit',device_metadata_policy='pull_only',host_transpose='auto',host_transpose_method='whole_contiguous')
    if len(queries)!=21 or any(event['details']['metrics'].get(key)!=value for event in queries for key,value in expected.items()):
        raise ValueError('PR control actual policy differs')
    return dict(status='PASS_ACTUAL_PR_POLICY',queries=21,expected_metrics=expected)
