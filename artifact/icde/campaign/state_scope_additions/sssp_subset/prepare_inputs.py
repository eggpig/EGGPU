"""Exact author update packets; immutable base text shared across rho cells."""
from pathlib import Path
import argparse,ctypes,hashlib,importlib.util,json,os,shutil,subprocess,sys,time
from observe_build import bind

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


class UpdateWriter:
    def __init__(self,receipt):
        self.receipt=Path(receipt);self.record=json.loads(self.receipt.read_text())
        if self.record['source']!=bind(HERE/'update_text.cpp') or self.record['library']!=bind(self.record['library']['path']):raise ValueError('CPU formatter changed')
        self.library=ctypes.CDLL(self.record['library']['path']);self.call=self.library.write_updates
        self.call.argtypes=[ctypes.c_uint64]*3+[ctypes.c_void_p]*4+[ctypes.c_char_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_char_p,ctypes.c_uint64]
        self.call.restype=ctypes.c_int
    def write(self,n,m,off,ind,weights,positions,path=None):
        import numpy as np
        arrays=[np.asarray(x) for x in (off,ind,weights,positions)]
        for value,dtype,count in zip(arrays,('<i4','<i4','<f8','<i8'),(n+1,m,m,len(positions))):
            if value.dtype.str!=dtype or value.shape!=(count,) or not value.flags.c_contiguous:raise ValueError('Exact canonical contiguous arrays required')
        length=ctypes.c_uint64();error=ctypes.create_string_buffer(512)
        code=self.call(n,m,len(positions),*(ctypes.c_void_p(v.ctypes.data) for v in arrays),None if path is None else os.fsencode(path),ctypes.byref(length),error,len(error))
        if code:raise ValueError(error.value.decode())
        return length.value


def build(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);library=output/'libupdate_text.so'
    command=['/usr/bin/g++','-O3','-std=c++17','-shared','-fPIC','-fno-gnu-unique',str(HERE/'update_text.cpp'),'-o',str(library)]
    subprocess.run(command,check=True)
    record=dict(status='PASS_CPU_FORMATTER_BUILD',source=bind(HERE/'update_text.cpp'),library=bind(library),command=command,gpu_executed=False)
    (output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');return record


def prepare(spec_path,output):
    spec_path=Path(spec_path);spec=json.loads(spec_path.read_text());output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    helper=Path(spec['author_text_helpers']);sys.path.insert(0,str(helper))
    from text_cache import Converter,TextCache
    cache_module=load('_sssp_input_hash_cache',spec['hash_cache_helper'])
    cache=cache_module.BindingCache(spec['hash_cache'])
    record=dict(status='PREPARING_CPU_ONLY',spec=bind(spec_path),gpu_executed=False)
    try:
        inputs=json.loads(Path(spec['inputs']['path']).read_text())
        if cache.binding(spec['inputs']['path'])!=spec['inputs']:raise ValueError('Input manifest changed')
        row=inputs['datasets'][spec['dataset']]
        for item in [row['positions'],*row['canonical'].values()]:
            if cache.binding(item['path'])!=item:raise ValueError('Canonical bytes changed')
        import numpy as np
        n,m=row['n'],row['m'];off=np.memmap(row['canonical']['offsets']['path'],dtype='<i4',mode='r',shape=(n+1,))
        ind=np.memmap(row['canonical']['indices']['path'],dtype='<i4',mode='r',shape=(m,));weights=np.memmap(row['canonical']['weights']['path'],dtype='<f8',mode='r',shape=(m,))
        positions=np.load(row['positions']['path'],allow_pickle=False,mmap_mode='r')
        if 20*len(positions)>2**32-1:raise ValueError('AUTHOR_UPDATE_COUNT_UNSUPPORTED: frozen Loader accumulates per-kind counts in uint32')
        if (n-1)*16>=2**32-1:raise ValueError('Frozen positive uint32 SSSP domain exceeded')
        formatter=UpdateWriter(spec['update_formatter_build']);needed=formatter.write(n,m,off,ind,weights,positions)
        converter=Converter(spec['base_formatter_build'],cache.binding)
        store=TextCache(spec['base_text_cache'],converter,cache,cache_module,cache.binding(spec['author_worker']),True,nonblocking=True)
        layout=dict(n=n,m=m,offsets=off,indices=ind,weights=weights)
        entry,base=store.ensure(layout)
        reserve=21*n*8*3+(2<<30)
        available=shutil.disk_usage(output).free
        record['storage_admission']=dict(actual_free_bytes=available,exact_update_text_bytes=needed,output_reserve_bytes=reserve,fail_on_resource=True)
        if available<needed+reserve:raise RuntimeError('CPU_INPUT_STORAGE_UNAVAILABLE: exact author update text requires '+str(needed)+' bytes plus full output reserve '+str(reserve))
        store.write_layout(layout,output/'base','grapin','sssp')
        actual=formatter.write(n,m,off,ind,weights,positions,output/'updates.edges')
        if actual!=needed:raise ValueError('Count-only and actual update text sizes differ')
        (output/'counts.txt').write_text((str(len(positions))+' '+str(len(positions))+'\n')*20)
        for name in ('updates.edges','counts.txt'):os.chmod(output/name,0o444)
        record.update(status='PASS_SSSP_COMMON_INPUTS',inputs=spec['inputs'],dataset=spec['dataset'],n=n,m=m,
            source=int(row['sources'][0]),commits=20,queries=21,changed_positions=len(positions),
            source_vertex_policy='First frozen primary source, held fixed for all 21 calls in every system.',
            files={'initial.edges':cache.binding(output/'base/graph.edges'),**{name:cache.binding(output/name) for name in ('updates.edges','counts.txt')}},
            base_text_metadata=cache.binding(entry/'metadata.json'),exact_update_text_bytes=needed,
            update_recipe='Each committed existing position receives 1+(base_weight-1+version)%16; delete previous weight then add same endpoints with new weight; all committed topologies identical.',
            formatter=bind(HERE/'update_text.cpp'),seconds=time.monotonic()-started)
    except BaseException as error:
        record.update(status='FAILED_CPU_INPUT_PREPARATION',error=type(error).__name__+': '+str(error));raise
    finally:
        (output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');cache.close()
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
    b=s.add_parser('build');b.add_argument('--output',type=Path,required=True)
    r=s.add_parser('prepare');r.add_argument('--spec',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.output) if a.action=='build' else prepare(a.spec,a.output)))
