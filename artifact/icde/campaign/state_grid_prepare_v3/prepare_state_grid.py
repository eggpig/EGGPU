#!/usr/bin/env python3
"""Plan or CPU-prepare shared immutable state bases without altering frozen runners."""
from pathlib import Path
import argparse,ast,copy,fcntl,hashlib,importlib.util,json,math,os,shutil,sys,time

ROLES={'offsets':'offsets.bin','indices':'indices.bin','weights':'weights.bin','metadata':'canonical.json','sources':'sources.json'}
POOLS={'E3':('US','IN','OR'),'E4':('US','OR','IN','UK'),'E5':('IN','UK','OR')}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8<<20),b''):h.update(data)
    return h.hexdigest()


def content(item):return {k:item[k] for k in ['bytes','sha256']}

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def bind(path):
    path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def load(path):return json.loads(Path(path).read_text())
def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def checked(item):
    if bind(item['path'])!=item:raise ValueError('Immutable file changed: '+item['path'])


def initial_cells():
    """First real targets only; full density/partition-locality axes are not claimed."""
    rows=[]
    for parts in (1,8):
        for name in POOLS['E3']:
            if parts==8 and name=='US':continue
            rows.append(dict(id=f'E3_{name}_p{parts}_rho1e-4_uniform',experiment='E3',datasets=[name],partitions=parts,rho=1e-4,locality='uniform'))
        for experiment,rho in [('E4',0.),('E5',1e-4)]:
            rows.append(dict(id=f'{experiment}_p{parts}',experiment=experiment,datasets=list(POOLS[experiment]),partitions=parts,rho=rho,locality='uniform'))
    return rows


def validate_cells(rows,qualification=False):
    ids=set()
    for row in rows:
        name=row['id'];parts=row['partitions'];names=row['datasets'];experiment=row['experiment']
        if not isinstance(name,str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in name) or name in ids:raise ValueError('Unsafe/duplicate cell id')
        ids.add(name)
        if parts not in (1,2,4,8) or experiment not in POOLS:raise ValueError('Unsupported partitions/experiment')
        if not names or len(set(names))!=len(names):raise ValueError('Empty/duplicate dataset pool')
        if not qualification and ((experiment=='E3' and (len(names)!=1 or names[0] not in POOLS['E3'])) or (experiment!='E3' and tuple(names)!=POOLS[experiment])):raise ValueError('Frozen dataset pool/order differs')
        if row['locality'] not in ('uniform','clustered') or not math.isfinite(row['rho']) or not 0<=row['rho']<=1:raise ValueError('Invalid update positions recipe')
    return rows


def code_identity(state_code,source,source_freeze):
    import numpy,scipy
    state_code=Path(state_code).resolve();source=Path(source).resolve();freeze=load(source_freeze)
    if digest(freeze['source_files'])!=freeze['source_tree_sha256']:raise ValueError('Source freeze digest differs')
    # The complete source identity is frozen, and all actual preparation imports are byte checked.
    signer=source/'easygraph/gpu/_manifest_signatures.py'
    if freeze['source_files']['easygraph/gpu/_manifest_signatures.py']!={'sha256':sha(signer),'bytes':signer.stat().st_size}:raise ValueError('Signer differs from source freeze')
    for name in ['state_inputs.py','state_protocol.py']:
        if not (state_code/name).is_file():raise ValueError('Missing frozen preparation dependency')
    identity=dict(controller=bind(__file__),state_inputs=bind(state_code/'state_inputs.py'),state_protocol=bind(state_code/'state_protocol.py'),source_signer=bind(signer),source_freeze=bind(source_freeze),source_sha256=freeze['source_tree_sha256'],numpy=numpy.__version__,scipy=scipy.__version__)
    sys.path.insert(0,str(state_code))
    for name in ('state_protocol','state_inputs'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=state_code/(name+'.py'):raise ValueError('Different state module already imported')
    import state_inputs
    return identity,state_inputs


def inspect_input(canonical,name,qualification=False):
    directory=(Path(canonical)/name).resolve();meta=load(directory/'canonical.json')
    if not qualification:
        receipt=load(directory/'receipt.json')
        if receipt.get('status')!='PASS_CANONICAL_INPUT' or receipt['dataset']!=name:raise ValueError('Unqualified canonical input')
        for role,file in ROLES.items():
            item=receipt['files']['canonical' if role=='metadata' else role]
            if (directory/file).stat().st_size!=item['bytes']:raise ValueError('Canonical size mismatch')
        if meta.get('weight_rule')!='w16-v1':raise ValueError('Wrong canonical weight version')
    files={role:bind(directory/file) for role,file in ROLES.items()}
    if not qualification:
        for role,item in files.items():
            expected=receipt['files']['canonical' if role=='metadata' else role]
            if any(item[k]!=expected[k] for k in ['bytes','sha256']):raise ValueError('Canonical payload SHA differs')
    return dict(directory=str(directory),metadata=meta,canonical=files,sources=load(directory/'sources.json')['primary_sources'],receipt=bind(directory/'receipt.json') if not qualification else None)


def preparation_extension(code, impl, legacy_controller, selector):
    legacy_controller=Path(legacy_controller);selector=Path(selector)
    if sha(legacy_controller) != '7afdfd44ad0c8395e0e3a8609f7891a4c5a191887aa3d25997140cf20452aae4':raise ValueError('Wrong frozen base serializer controller')
    if sha(selector) != '7a2d158ab8b0dca6601acaf465fe61a35557e155f34d18a4f78a9daae96b6a99':raise ValueError('Wrong typed position selector')
    oldfn=next(x for x in ast.parse(Path(impl.__file__).read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='positions')
    newfn=next(x for x in ast.parse(selector.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='positions')
    casts=0
    for node in ast.walk(newfn):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='searchsorted':
            probe=node.args[1]
            if isinstance(probe,ast.Call) and isinstance(probe.func,ast.Attribute) and probe.func.attr=='asarray':node.args[1]=probe.args[0];casts+=1
    if casts!=2 or ast.dump(oldfn,include_attributes=False)!=ast.dump(newfn,include_attributes=False):raise ValueError('Position selector changes the frozen recipe')
    spec=importlib.util.spec_from_file_location('_state_typed_positions_v3',selector);typed=importlib.util.module_from_spec(spec);spec.loader.exec_module(typed)
    code.update(base_serializer_controller=bind(legacy_controller),position_selector=bind(selector),position_selector_changes='Only two integer CSR search probe dtype casts; identical RNG/selection recipe')
    # New metadata explicitly binds the wrapper and selector. Only topology cache
    # keys retain the old controller because prepare_base/serialize are unchanged.
    class Interface:
        serialize=staticmethod(impl.serialize)
        projections=staticmethod(impl.projections)
        signer=staticmethod(impl.signer)
        positions=staticmethod(typed.positions)
    return code,Interface


def base_key(name,parts,entry,code):
    # Pool position/rho/locality never enter the immutable topology key.
    code=dict(code,controller=code.get('base_serializer_controller',code['controller']))
    return digest(dict(dataset=name,parts=parts,canonical={k:{f:v[f] for f in ['bytes','sha256']} for k,v in entry['canonical'].items()},preparation={k:content(code[k]) if isinstance(code[k],dict) else code[k] for k in ['controller','state_inputs','state_protocol','source_signer','numpy','scipy']}))


def position_key(entry,rho,locality,index,code):
    return digest(dict(canonical={k:{f:v[f] for f in ['bytes','sha256']} for k,v in entry['canonical'].items()},rho=rho,locality=locality,seed=20261104+index,preparation=content(code['state_inputs']),selector=content(code['position_selector']),numpy=code['numpy']))


def base_bound(meta,parts):
    n,m=meta['num_nodes'],meta['num_entries'];halos=min(m,(parts-1)*n)
    weighted=12*m+4*(n+parts)+12*(n+halos)
    weak=0 if not meta['directed'] else 8*m+4*(n+parts)+12*(n+min(2*m,(parts-1)*n))
    temporary=0 if not meta['directed'] else 8*m+4*(n+1)
    return dict(persistent_bytes_upper=weighted+weak+(4<<20),temporary_bytes_upper=temporary,estimate='Exact serializer types; halo <= min(arcs,(P-1)*n), directed weak arcs <= 2m; metadata allowance4MiB')


def positions_count(meta,rho):
    count=int(meta['num_entries']*rho)
    return count if meta['directed'] else 2*(count//2)


def verified_base(path):
    doc=load(path/'base.json')
    if doc['status']!='PASS_IMMUTABLE_STATE_BASE':raise ValueError('Incomplete base cache')
    body={k:v for k,v in doc.items() if k!='receipt_payload_sha256'}
    if doc.get('receipt_payload_sha256')!=digest(body):raise ValueError('Base metadata changed')
    for item in doc['owned_files']:checked(item)
    return doc


def prepare_base(path,name,parts,entry,code,impl,source):
    if path.exists():
        if not (path/'base.json').exists():raise ValueError('Incomplete previous base retained; inspect it before retry: '+str(path))
        return verified_base(path),False
    manifest,cuts=impl.serialize(Path(entry['directory']),path,name,parts,impl.signer(source))
    projections=impl.projections(Path(entry['directory']),manifest,name,parts,impl.signer(source))
    files=[bind(path/spec['path']) for part in load(manifest['path'])['partitions'] for spec in part['files'].values()]
    owned=[bind(f) for f in sorted(path.rglob('*')) if f.is_file()]
    row=dict(canonical=entry['canonical'],manifest=manifest,projections=projections,n=entry['metadata']['num_nodes'],m=entry['metadata']['num_entries'],edge_cuts=cuts,sources=entry['sources'],manifest_files=files)
    doc=dict(schema='state.shared_base.v1',status='PASS_IMMUTABLE_STATE_BASE',dataset=name,partitions=parts,preparation=code,canonical_receipt=entry['receipt'],row=row,owned_files=owned)
    doc['receipt_payload_sha256']=digest(doc)
    for item in owned:Path(item['path']).chmod(0o444)
    save(path/'base.json',doc);(path/'base.json').chmod(0o444)
    return doc,True


def prepare_positions(directory,key,entry,rho,locality,index,code,impl):
    import numpy as np
    folder=directory/key;meta=folder/'receipt.json'
    if folder.exists():
        if not meta.exists():raise ValueError('Incomplete positions retained: '+str(folder))
        record=load(meta);checked(record['positions']);return record,False
    folder.mkdir(parents=True)
    # Exact frozen selection recipe with dtype-preserving scalar searches.
    values=impl.positions(Path(entry['directory']),rho,locality,20261104+index)
    if len(values)!=positions_count(entry['metadata'],rho):raise ValueError('Position selector count differs')
    file=folder/'update_positions.npy';np.save(file,values,allow_pickle=False)
    record=dict(schema='state.shared_positions.v1',positions=bind(file),requested_rho=rho,actual_rho=len(values)/entry['metadata']['num_entries'],changed_positions=len(values),locality=locality,seed=20261104+index,pool_index=index,selector=code['position_selector'],original_selector=code['state_inputs'])
    file.chmod(0o444);save(meta,record);meta.chmod(0o444);return record,True


def reserve_evidence(a):
    """Dropping the TW reserve requires completed canonical payload evidence."""
    if a.qualification_only:return dict(mode='synthetic_CPU_fixture',reserved_bytes=a.reserve_bytes)
    if a.reserve_bytes >= 29 << 30:return dict(mode='TW_pending_reserve',reserved_bytes=a.reserve_bytes)
    if a.reserve_bytes < 2 << 30:raise ValueError('Keep at least 2 GiB margin after TW completion')
    receipt=getattr(a,'completed_tw_receipt',None)
    expected=(a.canonical/'TW/receipt.json').resolve()
    if receipt is None or Path(receipt).resolve()!=expected:raise ValueError('Lower TW reserve requires explicit canonical TW receipt')
    tw=inspect_input(a.canonical,'TW',False)
    raw=load(expected)
    # Include auxiliary payloads too; PASS text alone cannot release reservation.
    for item in raw['files'].values():
        file=expected.parent/Path(item['path']).name
        if content(bind(file))!=content(item):raise ValueError('Completed TW payload changed')
    return dict(mode='TW_complete_full_payload_verified',reserved_bytes=a.reserve_bytes,
                canonical_receipt=bind(expected),canonical=tw['canonical'])


def execute(a):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if a.reserve_bytes<0:raise ValueError('Negative disk reserve')
    cells=validate_cells(load(a.spec)['cells'] if a.spec else initial_cells(),a.qualification_only)
    tw_reserve=reserve_evidence(a)
    if a.plan.exists():raise FileExistsError('Preserve the previous preparation plan')
    if a.execute and (not a.run_receipt or a.run_receipt.exists()):raise FileExistsError('An unused explicit run receipt is required')
    code,impl=code_identity(a.state_code,a.source,a.source_freeze)
    code,impl=preparation_extension(code,impl,a.legacy_controller,a.position_selector)
    entries={name:inspect_input(a.canonical,name,a.qualification_only) for name in sorted({n for r in cells for n in r['datasets']})}
    root=a.output.resolve();cache=root/'bases';position_dir=root/'positions';bases={};positions={}
    for cell in cells:
        for index,name in enumerate(cell['datasets']):
            entry=entries[name];key=base_key(name,cell['partitions'],entry,code);path=cache/key
            existing=path.exists();doc=verified_base(path) if existing else None
            bases[key]=dict(dataset=name,partitions=cell['partitions'],path=str(path),exists=existing,**base_bound(entry['metadata'],cell['partitions']))
            keyp=position_key(entry,cell['rho'],cell['locality'],index,code);pathp=position_dir/keyp
            if pathp.exists():
                record=load(pathp/'receipt.json');checked(record['positions'])
            positions[keyp]=dict(path=str(pathp),exists=pathp.exists(),bytes_upper=8*positions_count(entry['metadata'],cell['rho'])+4096)
    persistent=sum(x['persistent_bytes_upper'] for x in bases.values() if not x['exists'])+sum(x['bytes_upper'] for x in positions.values() if not x['exists'])
    transient=max([x['temporary_bytes_upper'] for x in bases.values() if not x['exists']]+[0])
    ancestor=root
    while not ancestor.exists():ancestor=ancestor.parent
    free=shutil.disk_usage(ancestor).free;required=persistent+transient+a.reserve_bytes
    plan=dict(schema='state.shared_grid_plan.v1',status='PLAN_ONLY',gpu_executed=False,cells=cells,bases=bases,positions=positions,preparation=code,disk=dict(free_bytes=free,persistent_bytes_upper=persistent,temporary_peak_bytes_upper=transient,reserved_bytes=a.reserve_bytes,required_bytes_upper=required,admitted=free>=required,tw_reservation=tw_reserve),limitations=['Prepared inputs are not GPU qualification or measured results; explicit partition-hotspot controls require their own selector.','Position selection is byte-identical to frozen state_inputs.py, including undirected paired arcs and pool-index seed.','Physical-byte GPU pilot and exact tiny/target gates are still mandatory.'])
    save(a.plan,plan)
    if not a.execute:return plan
    if free<required:raise RuntimeError('Insufficient disk after explicit reserve; plan retained')
    root.mkdir(parents=True,exist_ok=True);cache.mkdir(exist_ok=True);position_dir.mkdir(exist_ok=True)
    record=dict(schema='state.shared_grid_receipt.v1',status='RUNNING',gpu_executed=False,plan=bind(a.plan),new_bases=[],reused_bases=[],new_positions=[],reused_positions=[],cells=[])
    begin=time.monotonic()
    with (root/'.prepare.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done_bases={};done_positions={}
        try:
            for cell in cells:
                rows={}
                for index,name in enumerate(cell['datasets']):
                    entry=entries[name];key=base_key(name,cell['partitions'],entry,code)
                    if key not in done_bases:
                        # Recheck current free bytes before each serialized allocation.
                        bound=bases[key]
                        if not (cache/key).exists() and shutil.disk_usage(root).free<bound['persistent_bytes_upper']+bound['temporary_bytes_upper']+a.reserve_bytes:raise RuntimeError('Disk headroom changed before base preparation')
                        doc,created=prepare_base(cache/key,name,cell['partitions'],entry,code,impl,a.source)
                        done_bases[key]=doc;record['new_bases' if created else 'reused_bases'].append(bind(cache/key/'base.json'))
                    keyp=position_key(entry,cell['rho'],cell['locality'],index,code)
                    if keyp not in done_positions:
                        if not (position_dir/keyp).exists() and shutil.disk_usage(root).free<positions[keyp]['bytes_upper']+a.reserve_bytes:raise RuntimeError('Disk headroom changed before position preparation')
                        pos,created=prepare_positions(position_dir,keyp,entry,cell['rho'],cell['locality'],index,code,impl)
                        done_positions[keyp]=pos;record['new_positions' if created else 'reused_positions'].append(bind(position_dir/keyp/'receipt.json'))
                    row=copy.deepcopy(done_bases[key]['row']);pos=done_positions[keyp]
                    row.update({k:pos[k] for k in ['positions','requested_rho','actual_rho','changed_positions','locality']});row['position_seed']=pos['seed'];row['shared_base']=bind(cache/key/'base.json');rows[name]=row
                out=root/'cells'/cell['id']/'inputs.json'
                obj=dict(schema='state.inputs.v2',status='PASS_CPU_INPUTS_GPU_UNQUALIFIED',datasets=rows,partitions=cell['partitions'],qualification_only=a.qualification_only,gpu_executed=False,weight_update_rule='changed values for version r are 1+((initial_weight-1+r) mod 16); undirected paired arcs change together',source_signer=code['source_signer'],shared_grid_preparation=code,cell=cell)
                if out.exists():
                    if load(out)!=obj:raise ValueError('Refuse changed cell overwrite: '+str(out))
                else:save(out,obj);out.chmod(0o444)
                record['cells'].append(bind(out));save(a.run_receipt,record)
            record['status']='PASS_CPU_SHARED_GRID_PREPARED_GPU_UNQUALIFIED'
        except BaseException as error:record.update(status='FAIL_PRESERVED_PARTIAL',error=type(error).__name__+': '+str(error));raise
        finally:
            record['seconds']=time.monotonic()-begin;record['disk_free_after']=shutil.disk_usage(root).free;save(a.run_receipt,record)
    return record


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','source-freeze','state-code','canonical','output','plan']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--legacy-controller',type=Path,required=True);p.add_argument('--position-selector',type=Path,required=True);p.add_argument('--run-receipt',type=Path)
    p.add_argument('--spec',type=Path);p.add_argument('--reserve-bytes',type=int,default=29<<30)
    p.add_argument('--completed-tw-receipt',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--qualification-only',action='store_true')
    return p

if __name__=='__main__':
    result=execute(parser().parse_args());print(json.dumps({k:result[k] for k in ['status','gpu_executed']}))
