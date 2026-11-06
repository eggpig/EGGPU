"""Schema fixtures exercise export admission only; these are not measurements."""
from pathlib import Path
import argparse,copy,importlib.util,json,sys,tempfile
import export_progress as export

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    campaign=Path(__file__).resolve().parents[2]
    protocol=load('fixture_state_protocol',campaign/'first_pass_20261106/state_main_first_pass/runner/state_protocol.py')
    pr=load('fixture_pr_protocol',campaign/'e3_unweighted_control_v4/pr_control.py');checks=[]
    with tempfile.TemporaryDirectory(prefix='state-export-',dir=a.scratch) as directory:
        root=Path(directory);blob=root/'output';blob.write_bytes(b'fixture-only');bound=export.bind(blob)
        source={'fixture_only':True};selection={'fixture_selection':True};deployment={'fixture_deployment':True}
        common=dict(phase='target',repeat=0,execution_complete=True,quality_pass=True,continuous_interval=True,
            sum_of_phases_used_as_total=False,first_complete_workflow_seconds=2.,workflow_complete_host_seconds=1.,inputs={'fixture_input':True})
        def audit(event):return dict(event=copy.deepcopy(event),output={'blob':bound},survives_session_close=True,quality=dict(passed=True,residual_l1=0.,mass_error=0.,reference_l1=0.,invariance_l1=0.))
        config=dict(source_binding=dict(source,negative_control={},admission_support={}),source_tree_sha256=export.SOURCE,deployment=deployment,arm='D1C1',commits=20,queries=21)
        events=pr.trace('IN');data=dict(common,schema=pr.PROTOCOL,status='PASS_UNWEIGHTED_PR_QUALIFICATION',configuration=config,configuration_sha256=export.digest(config),dataset='IN',arm='D1C1',events=copy.deepcopy(events),checks=[audit(e) for e in events if e['kind']=='query'])
        for event in data['events']:
            if event['kind']=='query':event['details']={'dependency_audit':{'weighted_view_transfer_counters':{key:0 for key in pr.WEIGHT_TRANSFER_COUNTERS}}}
        job=dict(id='pr-fixture',first_pass_stage='unweighted_pr_negative_control',dataset='IN',arm='D1C1',input_binding=common['inputs'])
        assert export.validate_record(data,job,source,selection,deployment=deployment)['complete_output_count']==21;checks.append('PR real producer trace and schema accepted')
        for label,modify in (
            ('source mismatch',lambda d:d['configuration']['source_binding'].update(fixture_only=False)),
            ('nonzero byte counter',lambda d:d['events'][2]['details']['dependency_audit']['weighted_view_transfer_counters'].update(patch_h2d_bytes=8)),
            ('invariance failure',lambda d:d['checks'][4]['quality'].update(invariance_l1=1e-3)),
            ('output-event swap',lambda d:d['checks'][4].update(event=d['checks'][3]['event']))):
            bad=copy.deepcopy(data);modify(bad);bad['configuration_sha256']=export.digest(bad['configuration'])
            try:export.validate_record(bad,job,source,selection,deployment=deployment)
            except ValueError:checks.append(label+' rejected')
            else:raise AssertionError(label)
        spec=dict(selection=selection,paired_block=dict(source_tree_sha256=export.SOURCE,old_readers='per_update',hotspot_k=4,rho=1e-4),paired_block_sha256='fixture',inputs=common['inputs'],datasets=['IN'],arm='D1C1')
        specpath=root/'spec.json';specpath.write_text(json.dumps(spec));events=protocol.trace('E3',['IN'])
        config=dict(source_binding=source,old_readers='per_update',arm='D1C1',experiment='E3',gpu_uuids=list(range(8)))
        hotspot=dict(common,schema='state.complete.workflow.v4',status='PASS_STATE_GPU_QUALIFICATION',source_binding=source,configuration=config,configuration_sha256=export.digest(config),
            first_pass=dict(spec=export.bind(specpath),paired_block_sha256='fixture'),events=events,checks=[audit(e) for e in events if e['kind']=='query'],leases=[{'released':True}]*20,maintenance_counters={})
        job=dict(id='hotspot-fixture',first_pass_stage='supplement_partition_hotspot_p8',command=['--spec',str(specpath)])
        assert export.validate_record(hotspot,job,source,selection,spec)['complete_output_count']==80;checks.append('Hotspot real producer trace and schema accepted')
        bad=copy.deepcopy(hotspot);bad['checks'][0]['event']=bad['checks'][1]['event']
        try:export.validate_record(bad,job,source,selection,spec)
        except ValueError:checks.append('Hotspot output-event mismatch rejected')
        else:raise AssertionError('Hotspot mismatch admitted')
        state=root/'state_main';focus=state/'results/focused_source_p8/receipt.json';focus.parent.mkdir(parents=True)
        (state/'selection.json').write_text(json.dumps({'source_tree_sha256':export.SOURCE}));focus.write_text(json.dumps(dict(status='PASS_STATE_GPU_QUALIFICATION',source_binding=source)))
        addition=root/'addition';addition.mkdir();depfile=addition/'pr_deployment.json';depfile.write_text('{}')
        data['configuration']['deployment']=export.bind(depfile);data['configuration_sha256']=export.digest(data['configuration'])
        receipt=addition/'actual_fixture.json';receipt.write_text(json.dumps(data))
        prjob=dict(id='pr-fixture',first_pass_stage='unweighted_pr_negative_control',dataset='IN',arm='D1C1',input_binding=common['inputs'],receipt=str(receipt),phase='target')
        jobs=[prjob]+[dict(id=f'pending_{i}',phase='target',receipt=str(root/f'pending_{i}')) for i in range(37)]+[dict(id=f'tiny_{i}',phase='tiny',receipt=str(root/f'tiny_{i}')) for i in range(2)]
        plan=addition/'plan.json';plan.write_text(json.dumps(dict(target_cells=38,source_tree_sha256=export.SOURCE,original_state_plan={'path':str(state/'all_plan.json')},jobs=jobs)))
        assert export.export(plan,addition/'progress')['eligible_n1']==1
        original_read=export.read;reads=[]
        def recording_read(path):reads.append(str(path));return original_read(path)
        export.read=recording_read
        assert export.export(plan,addition/'progress')['eligible_n1']==1
        assert str(receipt) not in reads and str(focus) not in reads;checks.append('Unchanged receipt/source evidence reused without reparsing; current and previous retained')
        blob.unlink()
        assert export.export(plan,addition/'progress')['eligible_n1']==0;checks.append('Cached result withdrawn after retained output disappears')
        export.read=original_read
        try:export.validate_record(hotspot,job,source,selection,spec)
        except ValueError:checks.append('Missing retained output rejected')
        else:raise AssertionError('Missing output admitted')
    record=dict(status='PASS_CPU_EXPORT_ADMISSION',fixture_only=True,gpu_executed=False,measurements_created=False,checks=checks,exporter=export.bind(export.__file__),test=export.bind(__file__))
    a.receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

if __name__=='__main__':main()
