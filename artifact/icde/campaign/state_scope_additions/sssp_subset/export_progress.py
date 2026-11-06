"""Recorded-only 36-cell SSSP export; primary is the direct post-base interval."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,html,json,math,os,shutil,time

SOURCE='71ea757d372f8c1b969ea074caaa27d3a77d4cea6d48cd9c6ae9c4b02936d7ce'
SYSTEMS=('D0C1','D1C1','Grapin');RHO=(0.,1e-6,1e-4,1e-2,.1,1.)
def read(path):return json.loads(Path(path).read_text())
def need(value,message):
    if not value:raise ValueError(message)
def stamp(path):
    try:s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    except FileNotFoundError:return None
def bind(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(path,value):
    temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)
def option(values,name):return values[values.index(name)+1]
def positive(value):return type(value) in (int,float) and math.isfinite(value) and value>0

def validate(data,job,spec,spec_binding,selection,source,worker_binding):
    need(data.get('schema')=='e3-fixed-source-sssp-common-weight-update-v1' and data.get('status')=='PASS_E3_SSSP_SUBSET','No complete subset PASS')
    need(data.get('source_tree_sha256')==SOURCE and data.get('phase')=='target' and data.get('repeat')==0 and data.get('observations')==1,'Wrong source/phase/observation')
    need(data.get('gpu_executed') is True and data.get('quality_pass') is True and data.get('execution_complete') is True,'Execution/numerical check incomplete')
    need(data['spec']==spec_binding and data['inputs']==spec['inputs'],'Bound target spec/input differs')
    need(all(data[key]==job[key]==spec[key] for key in ('system','dataset','rho')),'Planned target cell differs')
    config=data['configuration'];need(data['configuration_sha256']==digest(config),'Configuration digest differs')
    need(all(config[key]==spec[key] for key in ('system','source_tree_sha256','selection','gpu_uuids','cpu_set','environment','grapin_observation_build')),'Selected configuration differs')
    need(config['worker']==worker_binding and data['commits']==20 and data['queries']==21 and len(data['checks'])==21,'Wrong worker or incomplete20/21 trace')
    need(data.get('output_survives_teardown') is True and data.get('cold_first_comparable') is False,'Ownership/timing contract differs')
    checks=data['checks'];need([item['version'] for item in checks]==list(range(21)),'Output version association differs')
    for item in checks:
        need(item['source']==source and item.get('survives_teardown') is True and item.get('quality',{}).get('passed') is True,'Full output quality/lifetime check missing')
        blob=item['output']['blob'];need(Path(blob['path']).is_file() and Path(blob['path']).stat().st_size==blob['bytes'],'Retained output unavailable')
    primary=data['post_base_update_workflow_seconds'];prepared=data['prepared_service_seconds']
    need(positive(primary) and positive(prepared) and primary<=prepared,'Invalid direct continuous measured interval')
    if job['system']=='Grapin':
        session=data['native_session']
        need(session['snapshots']==21 and session['policy']=='keep' and session['parent_mode']=='repair' and session['B_outputs']==0 and session['H_outputs']==0 and session['extra_validation_inside_interval'] is False,'Wrong author workload/mode')
        need(primary==session['post_base_update_workflow_seconds'] and prepared==session['complete_service_wall_seconds'],'Native timing fields differ')
        need([event['snapshot'] for event in data['events']]==list(range(21)),'Native update trace incomplete')
    else:
        need(data['source_binding']['dso']==selection['native_dso'],'Actual native DSO differs')
        need([event['version'] for event in data['events']]==list(range(21)) and all(event['source']==source for event in data['events']),'Own update/source trace incomplete')
        need(all(event['commit'] is not None for event in data['events'][1:]),'A weight commit was omitted')
    return dict(status='COMPLETE_N1',observations=1,time_seconds=primary,post_base_update_workflow_seconds=primary,
        prepared_service_seconds=prepared,wall_seconds=data.get('wall_seconds'),commits=20,queries=21,quality_pass=True,
        timing_boundary='Direct continuous post-base20commit+SSSP interval; prepared service supplementary; never canonical cold-first')

class Exporter:
    def __init__(self,plan,output):
        self.plan_path=Path(plan).resolve();self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.plan=read(self.plan_path);need(self.plan['target_cells']==36 and self.plan['source_tree_sha256']==SOURCE,'Wrong subset plan')
        self.identity=bind(self.plan_path);self.context={};self.rows={};self.fingerprints={};self.outputs={};self.last_signature=None
        target_jobs=[j for j in self.plan['jobs'] if j['phase']=='target'];wanted={(d,r,s) for d in ('IN','OR') for r in RHO for s in SYSTEMS}
        unavailable=self.plan.get('unavailable',[])
        need({(j['dataset'],j['rho'],j['system']) for j in target_jobs+unavailable}==wanted,'Coverage differs from36 planned cells')
        bindings={}
        for job in target_jobs:
            specpath=Path(option(job['command'],'--spec'));spec=read(specpath)
            need(bind(specpath) in job['bindings'],'Plan spec binding differs')
            for key in ('selection','inputs'):
                item=spec[key]
                if item['path'] not in bindings:
                    need(bind(item['path'])==item,'Source/input binding changed');bindings[item['path']]=read(item['path'])
            selection=bindings[spec['selection']['path']];inputs=bindings[spec['inputs']['path']]
            need(selection['source_tree_sha256']==SOURCE,'Selection source differs')
            worker=[b for b in job['bindings'] if b['path']==job['command'][2]];need(len(worker)==1,'Worker binding missing')
            self.context[job['id']]=(spec,bind(specpath),selection,int(inputs['datasets'][job['dataset']]['sources'][0]),worker[0])
        self.target_jobs=target_jobs
    def signature(self):
        return [(job['id'],stamp(job['receipt'])) for job in self.plan['jobs']]+[(path,stamp(path)) for paths in self.outputs.values() for path in paths]
    def run(self):
        signature=self.signature()
        if signature==self.last_signature:return None
        self.last_signature=signature;rows=[]
        for job in self.target_jobs:
            key=job['id'];receipt=Path(job['receipt']);fingerprint=stamp(receipt)
            row=dict(id=key,dataset=job['dataset'],rho=job['rho'],system=job['system'],status='PENDING',observations=0,time_seconds=None,
                post_base_update_workflow_seconds=None,prepared_service_seconds=None)
            if fingerprint is not None:
                if self.fingerprints.get(key)!=fingerprint:
                    data=read(receipt)
                    try:value=validate(data,job,*self.context[key])
                    except (ValueError,KeyError,TypeError,OSError) as error:
                        value=dict(status='FAILED' if str(data.get('status','')).startswith('FAIL') else 'WITHHELD',
                            raw_status=data.get('status'),reason=str(error),worker_error=data.get('error'),observations=0)
                    value['receipt']=bind(receipt);self.rows[key]=value;self.fingerprints[key]=fingerprint
                    self.outputs[key]={item['output']['blob']['path']:item['output']['blob']['bytes'] for item in data.get('checks',[]) if 'output' in item and 'blob' in item['output']}
                row.update(self.rows[key])
                if row['status']=='COMPLETE_N1' and any(stamp(path) is None or Path(path).stat().st_size!=size for path,size in self.outputs.get(key,{}).items()):
                    row.update(status='WITHHELD',reason='Retained output unavailable',observations=0,time_seconds=None,post_base_update_workflow_seconds=None,prepared_service_seconds=None)
            rows.append(row)
        for cell in self.plan.get('unavailable',[]):rows.append(dict(cell,observations=0,time_seconds=None,post_base_update_workflow_seconds=None,prepared_service_seconds=None))
        summary=dict(schema='e3.sssp.subset.recorded-progress.v1',source_tree_sha256=SOURCE,plan=self.identity,target_cells=36,
            main_state_denominator=504,separate_state_additions=38,rows=rows,status_counts=dict(Counter(row['status'] for row in rows)),
            complete_n1=sum(row['status']=='COMPLETE_N1' for row in rows),updated_unix=time.time(),eta_seconds=None,
            primary='post_base_update_workflow_seconds',supplement='prepared_service_seconds',cold_first_comparable=False,
            timing_note='Both systems: actual continuous interval after completed base through20weight commits and20complete SSSP outputs. Never phase subtraction; no cold-first comparison.',
            initial_checks=[dict(id=j['id'],status=read(j['receipt']).get('status') if Path(j['receipt']).exists() else 'PENDING') for j in self.plan['jobs'] if j['phase']=='tiny'])
        current=self.output/'current.json'
        if current.exists():shutil.copyfile(current,self.output/'previous.json')
        atomic(current,summary)
        with (self.output/'cells.csv').open('w',newline='') as stream:
            fields=['id','dataset','rho','system','status','observations','time_seconds','post_base_update_workflow_seconds','prepared_service_seconds'];writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        self.plot(rows);self.last_signature=self.signature();return summary
    def plot(self,rows):
        parts=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="390" viewBox="0 0 900 390"><rect width="900" height="390" fill="white"/><g font-family="sans-serif" fill="#222">']
        colors={'D0C1':'#79acd1','D1C1':'#126bb0','Grapin':'#888888'}
        for panel,name in enumerate(('IN','OR')):
            left=65+panel*445;top=40;bottom=300;width=350;values=[row['time_seconds'] for row in rows if row['dataset']==name and row['status']=='COMPLETE_N1'];maximum=max(values)*1.1 if values else 1.
            parts.append(f'<text x="{left+175}" y="25" text-anchor="middle">{name}</text><path d="M{left} {top}V{bottom}H{left+width}" fill="none" stroke="#333"/>')
            for step in range(5):
                value=maximum*step/4;y=bottom-(bottom-top)*step/4;parts.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11">{value:.2g}</text>')
            for i,rho in enumerate(RHO):parts.append(f'<text x="{left+i*width/5}" y="320" text-anchor="middle" font-size="11">{rho:g}</text>')
            parts.append(f'<text x="{left+175}" y="341" text-anchor="middle" font-size="12">Update density</text>')
            for system in SYSTEMS:
                for row in rows:
                    if row['dataset']==name and row['system']==system and row['status']=='COMPLETE_N1':
                        x=left+RHO.index(row['rho'])*width/5;y=bottom-(bottom-top)*row['time_seconds']/maximum
                        parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[system]}"><title>{html.escape(system)}: {row["time_seconds"]:.6g} s</title></circle>')
        parts.append('<text transform="translate(14 180) rotate(-90)" text-anchor="middle" font-size="12">Time (s)</text>')
        for i,system in enumerate(SYSTEMS):parts.append(f'<circle cx="{210+i*200}" cy="365" r="4" fill="{colors[system]}"/><text x="{220+i*200}" y="369" font-size="12">{html.escape("EGGPU "+system if system!="Grapin" else system)}</text>')
        parts.append('</g></svg>');(self.output/'post_base_sssp.svg').write_text(''.join(parts)+'\n')
        (self.output/'FIGURE_CAPTION.txt').write_text('P1 SSSP-only, n=1: directly recorded continuous20commit+SSSP interval after completed base. Every complete point has21 full-output numerical checks. Missing cells remain blank and are listed in cells.csv/current.json. Prepared-input service time is supplemental; this is not canonical cold-first timing.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--watch-seconds',type=int,default=0);a=p.parse_args();exporter=Exporter(a.plan,a.output)
    while True:
        result=exporter.run()
        if result is not None:print(json.dumps({k:result[k] for k in ('complete_n1','target_cells','status_counts')}),flush=True)
        if not a.watch_seconds:break
        time.sleep(max(15,a.watch_seconds))
