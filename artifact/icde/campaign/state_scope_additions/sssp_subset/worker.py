"""Matched SSSP-only E3: one base result and twenty fixed-topology commits."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ctypes,fcntl,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
from observe_build import bind

PROTOCOL='e3-fixed-source-sssp-common-weight-update-v1'
ROUTES={'D0C1':'persistent_full','D1C1':'lease_sparse'}


def read(path):return json.loads(Path(path).read_text())
def checked(item):
    value=bind(item['path'])
    if any(value[k]!=item[k] for k in item if k in ('bytes','sha256','path')):raise ValueError('Changed bound file: '+item['path'])
    return Path(item['path'])
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def vector_sha(values):return hashlib.sha256(memoryview(values).cast('B')).hexdigest()


def own_workflow(spec,row,make_session,packet,intervention):
    import numpy as np
    source=int(row['sources'][0]);name=spec['dataset'];held=[];events=[];session=None
    started=time.perf_counter_ns();base_completed=None;failure=None
    try:
        session=make_session()
        for version in range(21):
            tick=time.perf_counter_ns();commit=None
            if version:
                commit=session.commit_weights(f'{name}@{version-1}',f'{name}@{version}',packet(row,version),storage='native_memory')
            result=session.query(f'{name}@{version}','sssp',{'source':source});values=result.values
            if values.shape!=(row['n'],) or values.dtype!=np.float64 or values.flags.writeable or not values.flags.owndata or values.base is not None:
                raise ValueError('Incomplete/nonowning result')
            if any(np.shares_memory(values,old) for old in held):raise ValueError('Reused answer/output storage')
            detail=intervention(result.backend_metrics,ROUTES[spec['system']],'closure',None)
            held.append(values)
            events.append(dict(version=version,source=source,commit=commit,metrics=result.backend_metrics,
                intervention=detail,complete_commit_query_seconds=(time.perf_counter_ns()-tick)/1e9))
            if version==0:base_completed=time.perf_counter_ns()
        completed=time.perf_counter_ns()
    except BaseException as error:
        completed=time.perf_counter_ns();failure=type(error).__name__+': '+str(error)
    finally:
        before=[vector_sha(values) for values in held]
        if session is not None:
            try:session.close()
            except BaseException as error:failure=(failure or '')+'; close: '+type(error).__name__+': '+str(error)
    survives=all(vector_sha(value)==sha for value,sha in zip(held,before))
    if not survives:failure=(failure or '')+'; result changed after Session close'
    complete=failure is None and len(held)==21
    return held,dict(prepared_service_seconds=(completed-started)/1e9 if complete else None,
        post_base_update_workflow_seconds=(completed-base_completed)/1e9 if complete else None,
        events=events,output_survives_teardown=survives,execution_complete=complete,execution_error=failure)


def native_workflow(spec,row,prepared,output,binding=bind):
    build=read(checked(spec['grapin_observation_build']))
    if build['status']!='PASS_OBSERVATION_ONLY_CPU_BUILD' or build['algorithms_changed'] is not False:raise ValueError('Reviewed observation-only build required')
    binary=checked(build['binary']);data=read(prepared)
    if data['status']!='PASS_SSSP_COMMON_INPUTS' or data['inputs']!=spec['inputs'] or data['source']!=int(row['sources'][0]):raise ValueError('Common author input differs')
    for item in data['files'].values():
        if binding(item['path'])!=item:raise ValueError('Prepared author input changed')
    prefix=output/'native_output'
    command=[str(binary),'--format=market_big','--graphfile='+data['files']['initial.edges']['path'],
        '--update_size='+data['files']['counts.txt']['path'],'--updatefile='+data['files']['updates.edges']['path'],
        '--eggpu_nodes='+str(row['n']),'--eggpu_cache_edges='+str(row['m']),'--weight_num=0','--weight=true',
        '--SEGMENT=1','--n_stream=1','--wl_alloc_factor=4','--source_node='+str(row['sources'][0]),'--output='+str(prefix)]
    env=dict(os.environ,EGGPU_GRAPIN_CACHE_POLICY='keep',EGGPU_GRAPIN_H_N='1',EGGPU_GRAPIN_B_N='1',
        EGGPU_GRAPIN_MODE='qualification' if spec['phase']=='tiny' else 'formal',EGGPU_GRAPIN_PARENT_MODE='repair',
        LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
        LD_LIBRARY_PATH=spec['artifact_root']+'/envs/cuda128/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu')
    with (output/'native.log').open('xb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:process.communicate(b'Q\n',timeout=3600)
        except BaseException:
            if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    if process.returncode:raise RuntimeError('Author native process failed')
    lines=(output/'native.log').read_text().splitlines()
    rows=[json.loads(line.split('EGGPU_GRAPIN_JSON ',1)[1]) for line in lines if line.startswith('EGGPU_GRAPIN_JSON ')]
    sessions=[json.loads(line.split('EGGPU_GRAPIN_SESSION_V11_JSON ',1)[1]) for line in lines if line.startswith('EGGPU_GRAPIN_SESSION_V11_JSON ')]
    if len(rows)!=21 or [r['snapshot'] for r in rows]!=list(range(21)) or len(sessions)!=1:raise ValueError('Missing complete twenty-commit native trace')
    session=sessions[0]
    if session['snapshots']!=21 or session['policy']!='keep' or session['parent_mode']!='repair' or session['B_outputs'] or session['H_outputs']:
        raise ValueError('Unexpected author transfer/parent-maintenance workload')
    if session['extra_validation_inside_interval']!=(spec['phase']=='tiny'):raise ValueError('Author mode changed')
    for key in ('complete_service_wall_seconds','post_base_update_workflow_seconds'):
        if not 0<float(session[key])<=3600:raise ValueError('Invalid native continuous interval')
    if any(row['nodes']!=spec['n'] for row in rows):raise ValueError('Native omitted vertices')
    return [Path(str(prefix)+f'.{v}.f64') for v in range(21)],dict(
        prepared_service_seconds=session['complete_service_wall_seconds'],
        post_base_update_workflow_seconds=session['post_base_update_workflow_seconds'],events=rows,
        native_session=session,invocation=dict(argv=command,environment={k:env[k] for k in ('EGGPU_GRAPIN_CACHE_POLICY','EGGPU_GRAPIN_H_N','EGGPU_GRAPIN_B_N','EGGPU_GRAPIN_MODE','EGGPU_GRAPIN_PARENT_MODE','LD_PRELOAD','LD_LIBRARY_PATH')}),
        output_survives_teardown=True,execution_complete=True,execution_error=None,native_log=bind(output/'native.log'))


def warm_own_runtime(native_dso):
    lines=subprocess.check_output(['ldd',str(native_dso)],text=True).splitlines()
    paths=[line.split('=>',1)[1].strip().split()[0] for line in lines if 'libcudart.so' in line and '=>' in line]
    if len(paths)!=1:raise ValueError('Cannot identify frozen native CUDA runtime')
    runtime=ctypes.CDLL(paths[0]);runtime.cudaFree.argtypes=[ctypes.c_void_p];runtime.cudaFree.restype=ctypes.c_int
    runtime.cudaDeviceSynchronize.restype=ctypes.c_int
    if runtime.cudaFree(None) or runtime.cudaDeviceSynchronize():raise RuntimeError('Common pre-service CUDA initialization failed')
    return dict(runtime=bind(paths[0]),outside_service=True,scope='Same cudaFree(nullptr)+device synchronize already present before Grapin service_begin')


def execute(spec_path,execute_gpu):
    if not execute_gpu:raise ValueError('Only root may execute GPU work')
    spec_path=Path(spec_path);spec=read(spec_path);selection=read(checked(spec['selection']));output=Path(spec['output'])
    if spec['phase'] not in ('tiny','target') or spec['repeat']!=0:raise ValueError('First-observation scope only')
    if selection['source_tree_sha256']!=spec['source_tree_sha256']:raise ValueError('Selected source differs')
    if str(Path(sys.executable).absolute())!=selection['python_invocation']:raise ValueError('Wrong Python invocation')
    for key in ('source_freeze','native_dso','native_build_receipt','runner_freeze'):checked(selection[key])
    support=load('_sssp_scope_guard',selection['source_guard']);support.source_files(Path(selection['source_root']),read(selection['source_freeze']['path']))
    for name,item in read(selection['runner_freeze']['path'])['files'].items():checked(dict(item,path=str(Path(selection['runner'])/name)))
    hash_helper=load('_sssp_hash_cache',checked(selection['hash_cache_helper']));cache=hash_helper.BindingCache(selection['hash_cache'])
    sys.path.insert(0,selection['runner']);import state_protocol as protocol
    protocol.bind=cache.binding;protocol.sha=lambda path:cache.binding(path)['sha256']
    import state_worker as worker
    import state_contract as contract
    inputs=worker.load_inputs(checked(spec['inputs']));row=inputs['datasets'][spec['dataset']]
    if inputs['partitions']!=1 or row['n']!=spec['n'] or row['m']!=spec['m'] or len(spec['gpu_uuids'].split(','))!=1:raise ValueError('Same-resource P1 domain differs')
    output.mkdir(parents=True,exist_ok=False);started=time.time()
    record=dict(schema=PROTOCOL,status='RUNNING',spec=bind(spec_path),source_tree_sha256=spec['source_tree_sha256'],
        system=spec['system'],dataset=spec['dataset'],rho=spec['rho'],phase=spec['phase'],repeat=0,inputs=spec['inputs'],
        commits=20,queries=21,gpu_executed=False,quality_pass=False,performance_eligible=False,
        cold_first_comparable=False,timing_contract='Prepared-input service and continuous post-base update window; 20 commits/21 full SSSP outputs; runtime initialization, text serialization and independent audit separately recorded.')
    configuration={key:spec[key] for key in ('system','source_tree_sha256','selection','gpu_uuids','cpu_set','environment','grapin_observation_build')}
    configuration.update(protocol=PROTOCOL,worker=bind(__file__),fixed_source_policy='first frozen primary source',commits=20,queries=21)
    record.update(configuration=configuration,configuration_sha256=digest(configuration))
    try:
        if spec['phase']=='target':
            proof=read(spec['tiny_proof'])
            if proof['status']!='PASS_E3_SSSP_SUBSET' or proof['phase']!='tiny' or proof['configuration']!=configuration or proof['quality_pass'] is not True or len(proof['checks'])!=21:
                raise ValueError('Exact initial subset qualification required')
            for item in proof['checks']:
                if item['quality']['passed'] is not True:raise ValueError('Tiny quality missing')
                contract.check_output(item['output'])
            record['tiny_proof']=bind(spec['tiny_proof'])
        selected={int(v) for v in spec['cpu_set'].split(',')}
        if len(selected)!=16 or not selected<=os.sched_getaffinity(0):raise ValueError('Exactly sixteen available CPUs required')
        os.sched_setaffinity(0,selected)
        os.environ.update(spec['environment'],**worker.COMMON_ENV,CUDA_VISIBLE_DEVICES=spec['gpu_uuids'],CUDA_DEVICE_ORDER='PCI_BUS_ID')
        sys.path.insert(0,str(Path(selection['runner'])/'baselines'))
        from hytgraph_qualify import resource_gate
        host=(4<<30)+21*row['n']*8+21*row['m']*8+row['m']*12+row['n']*64
        if spec['system']=='Grapin':
            # Author Loader stores both complete d/a lists, pinned copies and
            # all added-weight batches. Include vector-capacity slack and CSR.
            total=20*row['changed_positions']
            if total>2**32-1:raise ValueError('AUTHOR_UPDATE_COUNT_UNSUPPORTED: frozen uint32 Loader total')
            capacity=1 if not total else 1<<(total-1).bit_length()
            host=max(host,(4<<30)+row['m']*64+12*(5*total+2*capacity)+21*row['n']*8)
        record['resources']=resource_gate(spec['gpu_uuids'],host/(1<<30))
        storage_required=21*row['n']*8*3+(2<<30)
        record['storage_admission']=dict(actual_free_bytes=shutil.disk_usage(output).free,required_incremental_bytes=storage_required,
            scope='All 21 raw, archived and independent reference vectors simultaneously plus 2 GiB; no compression/dedup savings assumed',fail_on_resource=True)
        if record['storage_admission']['actual_free_bytes']<storage_required:raise RuntimeError('CPU_OUTPUT_STORAGE_UNAVAILABLE')
        prepared=None
        if spec['system']=='Grapin':
            prepared=Path(spec['prepared_inputs'])/'receipt.json'
            if not prepared.exists():
                formatter=load('_sssp_exact_input_adapter',Path(__file__).with_name('prepare_inputs.py'))
                formatter.prepare(spec['input_preparation_spec'],spec['prepared_inputs'])
            record['offline_author_text_preparation']=bind(prepared)
        with Path(spec['lock']).open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            record['storage_admission']['actual_free_bytes_before_native']=shutil.disk_usage(output).free
            if record['storage_admission']['actual_free_bytes_before_native']<storage_required:raise RuntimeError('CPU_OUTPUT_STORAGE_UNAVAILABLE_AFTER_PREPARATION')
            record['resources']=resource_gate(spec['gpu_uuids'],host/(1<<30));record['gpu_executed']=True
            if spec['system']=='Grapin':held,measured=native_workflow(spec,row,prepared,output,cache.binding)
            else:
                sys.path[:0]=[str(Path(selection['native_dso']['path']).parent),selection['source_root']]
                import cpp_easygraph as cpp
                from easygraph.gpu import Session
                if Path(cpp.__file__).resolve()!=Path(selection['native_dso']['path']).resolve():raise ValueError('Wrong actual DSO')
                record['source_binding']=contract.source_binding(Path(selection['source_root']),cpp)
                record['runtime_initialization']=warm_own_runtime(selection['native_dso']['path'])
                def session():
                    from state_controls import install_controller
                    result=Session.native({spec['dataset']+'@0':row['manifest']['path']},devices=[0],cpp=cpp,
                        route=ROUTES[spec['system']],policy='closure',budgets=None,retirement_policy='lineage',inventory_mode='recheck',batch_devices=True)
                    install_controller(result,'closure');return result
                held,measured=own_workflow(spec,row,session,worker.packet_for,protocol.validate_intervention)
            record.update(measured)
            import numpy as np
            from state_reference import ReferenceCache
            references=ReferenceCache(spec['reference_cache']);store=contract.OutputStore(spec['output_store']);checks=[];values_list=[]
            record['checks']=checks
            journal=(output/'output_index.jsonl').open('x')
            for version,item in enumerate(held):
                raw=item if isinstance(item,Path) else None
                values=np.fromfile(raw,dtype='<f8') if raw is not None else item;values.setflags(write=False)
                saved=store.put(values);values_list.append(values)
                checks.append(dict(version=version,source=int(row['sources'][0]),output=saved,survives_teardown=measured['output_survives_teardown']))
                if raw is not None:
                    checks[-1]['original_native_output']=cache.binding(raw);raw.unlink()
                journal.write(json.dumps(checks[-1])+'\n');journal.flush();os.fsync(journal.fileno())
            journal.close()
            for version,values in enumerate(values_list):
                checks[version]['quality']=worker.output_quality(values,row,version,'sssp',int(row['sources'][0]),references)
            if measured['execution_complete'] is not True or len(checks)!=21 or not all(x['quality']['passed'] and x['survives_teardown'] for x in checks):raise ValueError('Incomplete common full-output audit: '+str(measured.get('execution_error')))
            record.update(checks=checks,status='PASS_E3_SSSP_SUBSET',quality_pass=True,performance_eligible=spec['phase']=='target',observations=1 if spec['phase']=='target' else 0)
    except BaseException as error:
        record.update(status='FAIL_E3_SSSP_SUBSET',error=type(error).__name__+': '+str(error));raise
    finally:
        record.update(wall_seconds=time.time()-started,hash_cache=cache.report())
        protocol.save(output/'receipt.json',worker.strict(record));cache.close()
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--execute-gpu',action='store_true');a=p.parse_args()
    execute(a.spec,a.execute_gpu)
