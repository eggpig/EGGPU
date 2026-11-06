"""Verify the explicit state PR policy only after the original timed workflow."""
EXPECTED_ENV={
 'EGMGPU_PAGERANK_HOST_TRANSPOSE':'auto',
 'EGMGPU_PAGERANK_HOST_WEIGHT_STORAGE':'implicit',
 'EGMGPU_PAGERANK_DEVICE_METADATA':'pull_only',
 'EGMGPU_PAGERANK_TRANSPOSE_ENGINE':'cpu',
 'EGMGPU_PAGERANK_VIEW':'transpose'}
EXPECTED_METRICS={'preparation_method':'transpose','host_weight_storage':'implicit','device_metadata_policy':'pull_only'}

def inspect(record):
 environment=record.get('environment',{})
 mismatch={k:dict(expected=v,actual=environment.get(k)) for k,v in EXPECTED_ENV.items() if environment.get(k)!=v}
 if mismatch:raise ValueError('State PR policy environment differs: '+str(mismatch))
 if record.get('schema')=='state.byte.pilot.v4':
  return dict(status='BOUND_POLICY_NEW_PHYSICAL_BYTE_PILOT',environment=EXPECTED_ENV,
              actual_query_route_claimed=False,physical_bytes_remeasured=True)
 rows=[event for event in record.get('events',[]) if event.get('kind')=='query' and event.get('function')=='weighted_pr']
 if not rows:raise ValueError('Complete state workflow has no weighted PR route evidence')
 expected_method='whole_contiguous' if len(record.get('devices',[]))==1 else 'packed_destination_owned'
 for row in rows:
  metrics=row['details']['metrics']
  missing={k:dict(expected=v,actual=metrics.get(k)) for k,v in EXPECTED_METRICS.items() if metrics.get(k)!=v}
  if metrics.get('host_transpose')!='auto' or metrics.get('host_transpose_method')!=expected_method:
   missing['host_transpose_method']=dict(expected=expected_method,actual=metrics.get('host_transpose_method'))
  if missing:raise ValueError('Actual PR policy differs at '+str(row['index'])+': '+str(missing))
 return dict(status='PASS_ACTUAL_PR_POLICY',weighted_pr_queries=len(rows),expected_metrics=EXPECTED_METRICS,
             measurement_interval_unchanged=True,checked_after_workflow=True)
