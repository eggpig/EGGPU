"""E4S formal export gate: raw worker PASS alone is insufficient."""
from pathlib import Path
import argparse,json,math
import archive_outputs as archive

def eligible(result_path,maintenance_path,archive_helper):
 r,raw=archive.read_bound_json(result_path);post,pb=archive.read_bound_json(maintenance_path)
 archive.require(r.get('status')=='PASS_E4S_FORMAL' and r.get('phase')=='formal' and r.get('quality_pass') is True,'Not a successful formal worker measurement')
 archive.require(post.get('status')=='PASS_E4S_RECOVERABLE_OUTPUT_AND_MAINTENANCE' and post.get('archive_status')=='PASS' and post.get('original_maintenance_status')=='PASS' and post.get('original_result')==raw,'Mandatory archive and maintenance have not both passed for this exact worker result')
 spec=archive.load(archive.checked(post['attempt_spec']));archive.require(spec['result']==raw['path'] and spec['archive_helper']==archive_helper,'Postprocessing spec differs from this measurement/helper')
 archive.require(post.get('source') in spec['bindings'],'Maintenance wrapper source was not frozen in its spec')
 original=archive.load(archive.checked(post['original_maintenance_receipt']));archive.require(post['original_maintenance_receipt']['path']==spec['original_maintenance']['receipt'] and original.get('status') in spec['original_maintenance']['expected_statuses'],'Original maintenance did not pass its frozen contract')
 archive.checked(post['archive_receipt']);proof=archive.validate_sidecar(result_path,post['archive_receipt']['path'],archive_helper,require_formal=True)
 seconds=r['measurements']['first_complete_seconds'];archive.require(isinstance(seconds,(int,float)) and math.isfinite(seconds) and 0<seconds<=3600,'Invalid measured complete-task interval')
 return dict(status='PASS_E4S_FORMAL_EXPORT_ELIGIBILITY',dataset=r['dataset'],variant=r['variant'],batch=r['batch'],repeat=r.get('repeat'),input_sha256=r['input_sha256'],epoch_sha256=r['epoch_sha256'],first_complete_seconds=seconds,worker_result=raw,postprocessing_receipt=pb,recoverable_outputs=proof,postprocessing_excluded_from_measurement=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--maintenance',type=Path,required=True);p.add_argument('--archive-helper',type=Path,required=True);args=p.parse_args();print(json.dumps(eligible(args.result,args.maintenance,archive.binding(args.archive_helper)),indent=2))
