"""Byte-equivalence to the frozen NumPy serializer, plus cache lifecycle tests."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace
import numpy as np
import build
import prepare_plan
import cached_worker
import prepare_inputs
from text_cache import Converter,TextCache,ShutilProxy


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def rejected(call):
    try:call()
    except ValueError:return
    raise AssertionError('Invalid serialization accepted')


def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();args.scratch.mkdir(parents=True,exist_ok=True)
    memory_calls=[]
    try:
        prepare_inputs.memory_gate(SimpleNamespace(cgroup_memory=lambda:dict(available_bytes=1)),2,
            lambda **fields:memory_calls.append(fields),15,True)
    except RuntimeError as error:assert str(error).startswith('CPU_RESOURCE_UNAVAILABLE:')
    else:raise AssertionError('Memory shortage did not fail immediately')
    assert len(memory_calls)==1
    campaign=Path(__file__).resolve().parents[1]
    original=campaign/'baselines/sg_author_worker.py'
    sys.path.insert(0,str(original.parent))
    worker=load('_exact_author_serializer',original)
    cache_module=load('_text_cache_test_binding',campaign/'timing/cache_worker.py')
    with tempfile.TemporaryDirectory(prefix='text-cache-cpu-',dir=args.scratch) as temporary:
        root=Path(temporary);built=build.build(root/'build')
        bindings=cache_module.BindingCache(root/'bindings.sqlite')
        converter=Converter(root/'build/receipt.json',bindings.binding)
        fixtures=[]
        fixtures.append(dict(n=1,m=0,offsets=np.array([0,0],dtype='<i4'),indices=np.array([],dtype='<i4'),weights=np.array([],dtype='<f8')))
        fixtures.append(dict(n=7,m=8,offsets=np.array([0,0,3,3,4,7,8,8],dtype='<i4'),
            indices=np.array([0,1,6,2,0,4,6,1],dtype='<i4'),weights=np.array([0,1,9,10,16,4294967295,1,16],dtype='<f8')))
        generator=np.random.default_rng(2718)
        for n,m in [(19,(1<<18)+3),(1024,1000003)]:
            counts=np.zeros(n,dtype='<i4');counts[1:-1]=np.bincount(generator.integers(0,n-2,size=m),minlength=n-2)
            offsets=np.r_[np.int32(0),np.cumsum(counts,dtype=np.int32)].astype('<i4')
            fixtures.append(dict(n=n,m=m,offsets=offsets,indices=generator.integers(0,n,size=m,dtype=np.int32),
                weights=np.resize(np.array([1,9,10,16,4294967295],dtype='<f8'),m)))
        observations=[]
        for i,layout in enumerate(fixtures):
            for system in ('grapin','hytgraph'):
                golden=root/f'golden-{i}-{system}';started=time.monotonic()
                worker.write_layout(layout,golden,system,'sssp')
                old_seconds=time.monotonic()-started
                result=root/f'compiled-{i}-{system}.edges';started=time.monotonic()
                written=converter.write(layout,result);new_seconds=time.monotonic()-started
                assert result.read_bytes()==(golden/'graph.edges').read_bytes()
                assert written==result.stat().st_size==converter.write(layout)
                observations.append(dict(fixture=i,system=system,n=layout['n'],m=layout['m'],bytes=written,
                    sha256=build.bind(result)['sha256'],numpy_serialization_seconds=old_seconds,compiled_serialization_seconds=new_seconds))
        bad=dict(fixtures[1]);bad['weights']=bad['weights'].copy();bad['weights'][0]=1.5
        rejected(lambda:converter.write(bad))
        bad=dict(fixtures[1]);bad['indices']=bad['indices'].copy();bad['indices'][0]=7
        rejected(lambda:converter.write(bad))
        bad=dict(fixtures[1]);bad['offsets']=bad['offsets'].copy();bad['offsets'][2]=-1
        rejected(lambda:converter.write(bad))
        existing=root/'existing.edges';existing.write_bytes(b'preserve')
        rejected(lambda:converter.write(fixtures[1],existing));assert existing.read_bytes()==b'preserve'
        # A read-only canonical memmap uses the exact input-file SHA cache;
        # all systems/functions share this content key independent of source.
        layout=dict(fixtures[1])
        for name in ('offsets','indices','weights'):
            path=root/(name+'.bin');layout[name].tofile(path)
            layout[name]=np.memmap(path,dtype=layout[name].dtype,mode='r',shape=layout[name].shape)
        store=TextCache(root/'objects',converter,bindings,cache_module,build.bind(original),True)
        entry,metadata=store.ensure(layout);metadata_before=build.bind(entry/'metadata.json')
        store.allow_build=False
        converter_write=converter.write
        converter.write=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Prepared text serialized again'))
        for i,(system,algorithm) in enumerate((s,f) for s in ('grapin','hytgraph') for f in ('bfs','sssp','pr')):
            stage=root/f'alias-{i}';paths=store.write_layout(layout,stage,system,algorithm)
            assert set(p.name for p in paths)=={'graph.edges','empty_updates.txt'}
            assert os.stat(stage/'graph.edges').st_ino==os.stat(entry/'graph.edges').st_ino
            assert bindings.binding(stage/'graph.edges')['sha256']==metadata['blob']['sha256']
            ShutilProxy(store).rmtree(stage)
            assert not stage.exists() and (entry/'graph.edges').exists()
            assert build.bind(entry/'metadata.json')==metadata_before
        converter.write=converter_write
        preparation=root/'prepared.json'
        prep=dict(status='PASS_CPU_PREPARED_TEXT_INPUTS',converter_build=build.bind(root/'build/receipt.json'),
            text_cache=str((root/'objects').resolve()),original_worker=build.bind(original),
            layouts={kind:dict(status='PASS',n=layout['n'],m=layout['m'],metadata=build.bind(entry/'metadata.json'),blob=metadata['blob']) for kind in ('canonical','wcc')})
        preparation.write_text(json.dumps(prep))
        deferred=root/'deferred.json';oldjobs=[]
        for system in ('grapin','hytgraph'):
            for algorithm in ('bfs','sssp','wcc','pr'):
                identifier=f'target_TW_{algorithm}_{system}_author_r00'
                command=[sys.executable,'-B',str(campaign/'timing/placement_worker.py'),
                    '--worker',str(original),'--worker-sha256',build.bind(original)['sha256'],'--cache',str(root/'bindings.sqlite'),
                    '--','--system',system,'--algorithm',algorithm,'--dataset','TW','--phase','target','--repeat','0',
                    '--output',str(root/'old'/identifier),'--reference-cache',str(root/'references'),'--timeout','1800','--tiny-proof','unchanged-original-proof.json']
                oldjobs.append(dict(id=identifier,command=command,depends_on=[],receipt=str(root/'old'/identifier/'result.json'),
                    environment={'UNCHANGED':'1'},bindings=[],disk_guard={'path':str(root),'min_free_bytes':123},expected_statuses=['PASS_QUALIFICATION']))
        deferred.write_text(json.dumps(dict(jobs=oldjobs)))
        plan_info=prepare_plan.prepare(deferred,preparation,root/'build/receipt.json',root/'objects',root/'planned')
        assert plan_info['jobs']==8
        planned=json.loads((root/'planned/plan.json').read_text())
        for old,new in zip(oldjobs,planned['jobs']):
            before=old['command'][old['command'].index('--')+1:]
            after=new['command'][new['command'].index('--')+1:]
            before[before.index('--output')+1]=after[after.index('--output')+1]
            assert before==after and old['environment']==new['environment'] and old['disk_guard']==new['original_disk_guard']
            assert new['disk_guard']['min_free_bytes']==new['cached_input_storage_admission']['min_free_bytes']
            assert new['receipt'].endswith('/result_cached_input.json')
        prep['layouts']['wcc']={'status':'UNSUPPORTED_FROZEN_INT32_LAYOUT'}
        unsupported=root/'prepared-unsupported.json';unsupported.write_text(json.dumps(prep))
        rejected_plan=prepare_plan.prepare(deferred,unsupported,root/'build/receipt.json',root/'objects',root/'planned-unsupported')
        assert rejected_plan['jobs']==6 and len(rejected_plan['unavailable'])==2
        # Exercise receipt wrapping with a CPU-only stand-in for the native job.
        # No author executable or numerical timing is claimed by this fixture.
        class FakePolicy:
            def install(self,module):pass
        wrapped=root/'wrapped-output'
        fixture_result=dict(status='PASS_QUALIFICATION',quality_pass=True,phase='target',repeat=0,
            sources=[dict(calls={'first':{'seconds':123},'warmup':{'timed':False},'ready':{'seconds':456}})],
            gpu_executed=False,cpu_fixture=True)
        def fake_run(worker_path,expected,cache_path,arguments,cpus):
            wrapped.mkdir()
            policy=FakePolicy();policy.cache=bindings
            module=SimpleNamespace(__file__=str(original),run=worker.run)
            FakePolicy.install(policy,module)
            module.write_layout(layout,wrapped/'stage','grapin','bfs')
            module.shutil.rmtree(wrapped/'stage')
            (wrapped/'result.json').write_text(json.dumps(fixture_result))
        fake=SimpleNamespace(PlacementPolicy=FakePolicy,cache_worker=cache_module,run=fake_run)
        original_load=cached_worker.load;original_converter=cached_worker.Converter
        try:
            cached_worker.load=lambda *a:fake
            cached_worker.Converter=lambda *a:converter
            cached_worker.run(SimpleNamespace(arguments=['--system','grapin','--dataset','TW','--phase','target',
                '--repeat','0','--algorithm','bfs','--output',str(wrapped),'--reference-cache',str(root/'references')],worker=original,worker_sha256=build.bind(original)['sha256'],
                preparation=preparation,placement_helper=campaign/'timing/placement_worker.py',
                converter_build=root/'build/receipt.json',text_cache=root/'objects',cache=root/'bindings.sqlite',cpu_set=None))
        finally:
            cached_worker.load=original_load;cached_worker.Converter=original_converter
        assert json.loads((wrapped/'result.json').read_text())==fixture_result
        amended=json.loads((wrapped/'result_cached_input.json').read_text())
        assert amended['sources']==fixture_result['sources'] and amended['ready_measurement_unchanged'] is True
        assert amended['canonical_csr_cold_first_comparable'] is False and amended['original_author_result']==build.bind(wrapped/'result.json')
        missing=TextCache(root/'unprepared',converter,bindings,cache_module,build.bind(original),False)
        rejected(lambda:missing.write_layout(layout,root/'must-not-run','grapin','bfs'))
        blob=entry/'graph.edges';os.chmod(blob,0o644)
        rejected(lambda:store.ensure(layout))
        checks=['Compiled serializer byte-identical to original NumPy write_layout for both loaders',
            'Fail-on-resource memory preflight exits after one check without a polling wait; disk shortage always fails directly',
            'Empty graph, empty CSR rows, uint32 limits, 2^18 chunk boundary and >8 MiB buffered output checked',
            'Fractional weights, invalid destination/offsets and overwrite rejected',
            'Six BFS/SSSP/PR system/function consumers reuse one exact cache blob without conversion',
            'Read-only hardlink cleanup retains cache; immutable producer metadata binding remains unchanged',
            'Unprepared cache and writable cached payload rejected',
            'Eight successor jobs preserve every original worker argument except new output path; unsupported weak layouts remain two explicit CPU-evidence cells',
            'CPU-only wrapper fixture preserves raw receipt and every call value; derived receipt explicitly separates cached first and unchanged ready',
            'Original worker, graph_layout, native invocations, algorithms and numerical checks not edited',
            'Only CPU serializer and file lifecycle exercised; no GPU calls or author binary invocation']
        record=dict(status='PASS_CPU_EXACT_TEXT_CACHE',checks=checks,source=build.bind(Path(__file__).with_name('csr_text.cpp')),
            cache=build.bind(Path(__file__).with_name('text_cache.py')),original_worker=build.bind(original),
            fixtures=observations,synthetic_fixture_timings_not_TW_ETA=True,gpu_executed=False)
        bindings.close()
        for name in ('offsets','indices','weights'):layout[name]._mmap.close()
        # This workspace uses a network filesystem; release the test library's
        # mapping before TemporaryDirectory removes its inode.
        import _ctypes
        _ctypes.dlclose(converter.library._handle)
        with args.receipt.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
        print(json.dumps({k:v for k,v in record.items() if k!='fixtures'}))


if __name__=='__main__':main()
