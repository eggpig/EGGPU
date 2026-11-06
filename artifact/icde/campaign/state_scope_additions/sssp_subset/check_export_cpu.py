"""Admission fixtures only; test values are not target measurements."""
from pathlib import Path
import argparse,copy,json,tempfile
import export_progress as e

def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    checks=[]
    with tempfile.TemporaryDirectory(prefix='sssp-export-',dir=a.scratch) as directory:
        root=Path(directory);output=root/'output.bin';output.write_bytes(b'fixture-only');blob=e.bind(output)
        worker={'fixture_worker':True};selection={'native_dso':{'fixture_dso':True}}
        job=dict(system='D1C1',dataset='IN',rho=1e-4);spec=dict(job,inputs={},source_tree_sha256=e.SOURCE,selection={},gpu_uuids='fixture',cpu_set='0',environment={},grapin_observation_build={})
        configuration={k:spec[k] for k in ('system','source_tree_sha256','selection','gpu_uuids','cpu_set','environment','grapin_observation_build')};configuration['worker']=worker
        data=dict(job,schema='e3-fixed-source-sssp-common-weight-update-v1',status='PASS_E3_SSSP_SUBSET',source_tree_sha256=e.SOURCE,phase='target',repeat=0,observations=1,gpu_executed=True,quality_pass=True,execution_complete=True,
            spec={},inputs={},configuration=configuration,configuration_sha256=e.digest(configuration),commits=20,queries=21,output_survives_teardown=True,cold_first_comparable=False,
            checks=[dict(version=i,source=3,survives_teardown=True,quality={'passed':True},output={'blob':blob}) for i in range(21)],
            post_base_update_workflow_seconds=2.,prepared_service_seconds=7.,wall_seconds=19.,source_binding={'dso':selection['native_dso']},events=[dict(version=i,source=3,commit=None if not i else {}) for i in range(21)])
        result=e.validate(data,job,spec,{},selection,3,worker)
        assert result['time_seconds']==2. and result['prepared_service_seconds']==7. and result['time_seconds']!=7.-2.
        checks.append('Primary copies direct post-base interval; prepared service and job wall remain distinct')
        for label,change in [('missing numerical check',lambda d:d['checks'][6]['quality'].update(passed=False)),('wrong source',lambda d:d.update(source_tree_sha256='old')),('missing commit',lambda d:d['events'][3].update(commit=None)),('invalid interval',lambda d:d.update(post_base_update_workflow_seconds=8.))]:
            bad=copy.deepcopy(data);change(bad)
            try:e.validate(bad,job,spec,{},selection,3,worker)
            except ValueError:checks.append(label+' rejected')
            else:raise AssertionError(label)
        output.unlink()
        try:e.validate(data,job,spec,{},selection,3,worker)
        except ValueError:checks.append('Missing retained complete vector rejected')
        else:raise AssertionError('Missing output admitted')
    record=dict(status='PASS_CPU_SSSP_EXPORT_ADMISSION',fixture_only=True,gpu_executed=False,checks=checks,exporter=e.bind(e.__file__))
    a.receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

if __name__=='__main__':main()
