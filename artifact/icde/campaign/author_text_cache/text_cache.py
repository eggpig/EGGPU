"""Exact input serialization cache; no graph, native-call or output changes."""
from pathlib import Path
import ctypes
import fcntl
import hashlib
import json
import os
import shutil
import stat
import sys
import time

FORMAT='src dst uint32_weight, decimal ASCII, one space, LF, no header; CSR arc order'


def read(path):return json.loads(Path(path).read_text())


def atomic(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def content_hash(data):
    digest=hashlib.sha256();view=memoryview(data).cast('B')
    for first in range(0,len(view),16<<20):digest.update(view[first:first+(16<<20)])
    return digest.hexdigest()


def file_identity(path):
    path=Path(path)
    if path.is_symlink():raise ValueError('Cache blob must not be a symlink')
    value=path.stat()
    if not stat.S_ISREG(value.st_mode):raise ValueError('Cache blob must be a regular file')
    return dict(device=value.st_dev,inode=value.st_ino,bytes=value.st_size,mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,mode=stat.S_IMODE(value.st_mode),nlink=value.st_nlink)


class Converter:
    def __init__(self,receipt,binding):
        self.receipt=Path(receipt);self.record=read(receipt)
        if self.record['status']!='PASS_CPU_BUILD_ONLY':raise ValueError('CPU converter build did not pass')
        if binding(Path(__file__).with_name('csr_text.cpp'))!=self.record['source']:
            raise ValueError('Converter does not bind this reviewed serializer source')
        for name in ('source','library'):
            if binding(self.record[name]['path'])!=self.record[name]:raise ValueError('Converter source/library changed')
        if sys.byteorder!='little':raise ValueError('Frozen CSR requires little-endian host')
        self.library=ctypes.CDLL(self.record['library']['path'])
        self.call=self.library.csr_text_write
        self.call.argtypes=[ctypes.c_uint64,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
            ctypes.c_char_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_char_p,ctypes.c_uint64]
        self.call.restype=ctypes.c_int

    def write(self,layout,path=None):
        import numpy as np
        n,m=int(layout['n']),int(layout['m']);arrays=[]
        for name,dtype,count in [('offsets','<i4',n+1),('indices','<i4',m),('weights','<f8',m)]:
            value=np.asarray(layout[name])
            if value.dtype.str!=dtype or value.shape!=(count,) or not value.flags.c_contiguous:
                raise ValueError('Converter requires original complete contiguous '+name)
            arrays.append(value)
        written=ctypes.c_uint64();error=ctypes.create_string_buffer(512)
        code=self.call(n,m,*(ctypes.c_void_p(v.ctypes.data) for v in arrays),
            None if path is None else os.fsencode(path),ctypes.byref(written),error,len(error))
        if code:raise ValueError('Exact text conversion failed: '+error.value.decode())
        return written.value


class TextCache:
    def __init__(self,root,converter,hash_cache,cache_module,worker,allow_build=False,nonblocking=False):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.converter=converter;self.hash_cache=hash_cache;self.cache_module=cache_module
        self.worker=worker;self.allow_build=allow_build;self.nonblocking=nonblocking;self.events=[];self.stages={}
        self.original_rmtree=shutil.rmtree

    def array_identity(self,value):
        import numpy as np
        array=np.asarray(value)
        if not array.flags.c_contiguous:raise ValueError('Original layout array is non-contiguous')
        base=value;seen=set()
        while base is not None and id(base) not in seen:
            seen.add(id(base))
            if isinstance(base,np.memmap) and getattr(base,'filename',None):
                path=Path(base.filename)
                offset=int(getattr(base,'offset',0))+int(array.ctypes.data)-int(base.ctypes.data)
                if offset==0 and array.nbytes==path.stat().st_size and not array.flags.writeable:
                    record=self.hash_cache.binding(path)
                    return dict(dtype=array.dtype.str,shape=list(array.shape),bytes=array.nbytes,sha256=record['sha256'])
            base=getattr(base,'base',None)
        return dict(dtype=array.dtype.str,shape=list(array.shape),bytes=array.nbytes,sha256=content_hash(array))

    def identity(self,layout):
        record=dict(format=FORMAT,n=int(layout['n']),m=int(layout['m']),
            arrays={name:self.array_identity(layout[name]) for name in ('offsets','indices','weights')})
        key=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return key,record

    def seed_alias(self,path,sha):
        # This is used only immediately after creating or removing our hardlink,
        # whose inode and read-only bytes were checked while holding the entry lock.
        path=Path(path).resolve();identity=json.dumps(self.cache_module.fingerprint(path),separators=(',',':'))
        self.hash_cache.db.execute('INSERT OR REPLACE INTO sha_bindings VALUES (?,?,?)',(str(path),identity,sha))
        self.hash_cache.memory[str(path)]=(identity,sha)

    def verify(self,metadata,blob):
        actual=file_identity(blob)
        if actual['mode']&0o222:raise ValueError('Cached text unexpectedly writable')
        verification=blob.with_name('verification.json')
        previous=read(verification) if verification.exists() else dict(blob_identity=metadata['blob_identity'],blob_sha256=metadata['blob']['sha256'])
        if previous.get('blob_sha256')!=metadata['blob']['sha256']:raise ValueError('Cache verification identity changed')
        if actual!=previous['blob_identity']:
            value=self.cache_module.full_binding(blob)
            if any(value[k]!=metadata['blob'][k] for k in ('bytes','sha256')):
                raise ValueError('Cached text payload changed')
        atomic(verification,dict(blob_identity=actual,blob_sha256=metadata['blob']['sha256']))
        self.seed_alias(blob,metadata['blob']['sha256'])

    def ensure(self,layout):
        started=time.monotonic();key,identity=self.identity(layout)
        entry=self.root/key;entry.mkdir(exist_ok=True)
        blob=entry/'graph.edges';metadata_path=entry/'metadata.json'
        with (entry/'.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|(fcntl.LOCK_NB if self.nonblocking else 0))
            if metadata_path.exists():
                metadata=read(metadata_path)
                if metadata['identity']!=identity:raise ValueError('Cache identity differs')
                if metadata['converter_source']!=self.converter.record['source'] or metadata['converter_library']!=self.converter.record['library'] or metadata['original_worker']!=self.worker:
                    raise ValueError('Cache producer provenance differs')
                self.verify(metadata,blob);reused=True
            else:
                if not self.allow_build:raise ValueError('CPU input cache not prepared; never format inside a GPU job')
                if blob.exists():raise ValueError('Uncommitted cache blob exists; preserve for inspection')
                count_start=time.monotonic();expected_bytes=self.converter.write(layout)
                if shutil.disk_usage(entry).free<expected_bytes+(2<<30):raise ValueError('Insufficient disk for exact prepared text and reserve')
                temporary=entry/'graph.edges.partial'
                actual_bytes=self.converter.write(layout,temporary)
                if actual_bytes!=expected_bytes:raise ValueError('Exact count/write sizes differ')
                os.chmod(temporary,0o444);os.rename(temporary,blob)
                binding=self.cache_module.full_binding(blob)
                metadata=dict(schema='eggpu.author.text-cache.v1',identity=identity,key=key,blob=binding,
                    blob_identity=file_identity(blob),converter_build=self.hash_cache.binding(self.converter.receipt),
                    converter_source=self.converter.record['source'],converter_library=self.converter.record['library'],
                    original_worker=self.worker,format=FORMAT,rows=identity['m'],
                    count_write_seconds=time.monotonic()-count_start,gpu_executed=False)
                atomic(metadata_path,metadata);self.seed_alias(blob,binding['sha256']);reused=False
                atomic(entry/'verification.json',dict(blob_identity=file_identity(blob),blob_sha256=binding['sha256']))
        event=dict(key=key,metadata=str(metadata_path),blob=metadata['blob'],cache_hit=reused,
            cpu_seconds=time.monotonic()-started,bytes=metadata['blob']['bytes'])
        self.events.append(event)
        return entry,metadata

    def write_layout(self,layout,stage,system,algorithm):
        if system not in ('grapin','hytgraph'):raise ValueError('Text cache only covers the two exact author loaders')
        stage=Path(stage).resolve();entry,metadata=self.ensure(layout)
        with (entry/'.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            metadata=read(entry/'metadata.json');blob=entry/'graph.edges';self.verify(metadata,blob)
            stage.mkdir(parents=True,exist_ok=False)
            alias=stage/'graph.edges';os.link(blob,alias)
            if os.stat(alias).st_ino!=os.stat(blob).st_ino or os.stat(alias).st_dev!=os.stat(blob).st_dev:
                raise ValueError('Prepared text stage is not an exact hardlink')
            (stage/'empty_updates.txt').write_bytes(b'')
            atomic(entry/'verification.json',dict(blob_identity=file_identity(blob),blob_sha256=metadata['blob']['sha256']))
            self.seed_alias(blob,metadata['blob']['sha256']);self.seed_alias(alias,metadata['blob']['sha256'])
            self.stages[str(stage)]=entry
            self.events[-1].update(stage=str(stage),system=system,algorithm=algorithm,hardlink=True)
        return sorted(stage.iterdir())

    def rmtree(self,path,*args,**kwargs):
        key=str(Path(path).resolve());entry=self.stages.get(key)
        if entry is None:return self.original_rmtree(path,*args,**kwargs)
        with (entry/'.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            metadata=read(entry/'metadata.json');blob=entry/'graph.edges';self.verify(metadata,blob)
            self.original_rmtree(path,*args,**kwargs)
            atomic(entry/'verification.json',dict(blob_identity=file_identity(blob),blob_sha256=metadata['blob']['sha256']))
            self.seed_alias(blob,metadata['blob']['sha256'])


class ShutilProxy:
    def __init__(self,cache):self.cache=cache
    def __getattr__(self,name):return self.cache.rmtree if name=='rmtree' else getattr(shutil,name)
