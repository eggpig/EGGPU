"""Account for cached adapter staging using real free bytes, outside timing."""
from pathlib import Path
import ast
import copy
import hashlib
import inspect
import json
import os
import shutil

RESERVE=2<<30


def bound(n,algorithm):
    n=int(n)
    if n<1 or algorithm not in ('bfs','sssp','pr','wcc'):raise ValueError('Invalid output domain')
    sources=3 if algorithm in ('bfs','sssp') else 1
    # Per source: all three f64 native files, all three distinct f64 npy
    # archives, and one independent f64 reference, simultaneously budgeted.
    # Actual archive processing unlinks each raw file and deduplicates equal
    # arrays; neither reduction is assumed. Four npy headers get 256 B each.
    return dict(n=n,source_count=sources,native_raw_bytes=3*8*n*sources,
        archived_output_bytes=3*8*n*sources,reference_bytes=8*n*sources,
        npy_header_bytes=4*256*sources,logs_metadata_reserve_bytes=RESERVE,
        cache_hardlink_incremental_payload_bytes=0,
        min_free_bytes=7*8*n*sources+4*256*sources+RESERVE)


def prepared_bound(preparation,algorithm,binding):
    kind='wcc' if algorithm=='wcc' else 'canonical';layout=preparation['layouts'][kind]
    if layout['status']!='PASS':raise ValueError('Cannot admit unprepared layout')
    metadata_item=layout['metadata']
    if binding(metadata_item['path'])!=metadata_item:raise ValueError('Prepared text metadata changed')
    metadata=json.loads(Path(metadata_item['path']).read_text())
    if metadata['blob']!=layout['blob'] or metadata['identity']['n']!=layout['n'] or metadata['identity']['m']!=layout['m']:
        raise ValueError('Prepared text dimensions/blob differ')
    blob=Path(metadata['blob']['path']);info=blob.stat()
    if blob.is_symlink() or info.st_size!=metadata['blob']['bytes'] or info.st_mode&0o222:
        raise ValueError('Prepared text must remain a read-only exact-size file')
    result=bound(layout['n'],algorithm)
    result.update(m=layout['m'],cached_text_bytes=metadata['blob']['bytes'],cached_text=metadata['blob'],
        cache_metadata=metadata_item,original_threshold_bytes=48*layout['m']+72*layout['n']+RESERVE,
        policy='Adapter storage admission only: existing read-only text is hardlinked, native and reference outputs retain a full conservative bound; real filesystem free bytes are unchanged.')
    return result


class Admission:
    def __init__(self,record,output,reference_cache):
        self.record=record;self.output=Path(output).resolve();self.events=[]
        base=self.output
        while not base.exists():base=base.parent
        reference=Path(reference_cache).resolve()
        while not reference.exists():reference=reference.parent
        devices={os.stat(path).st_dev for path in (base,reference,record['cached_text']['path'])}
        if len(devices)!=1:raise ValueError('This bound requires output/reference/cache on the same filesystem')

    def insufficient(self,graph,out,index):
        if Path(out).resolve()!=self.output or graph['n']!=self.record['n'] or graph['m']!=self.record['m']:
            raise ValueError('Cached storage guard scope differs')
        if index not in range(self.record['source_count']):raise ValueError('Unexpected source scope')
        actual=shutil.disk_usage(out)
        event=dict(source_index=index,path=str(self.output),actual_free_bytes=actual.free,
            actual_total_bytes=actual.total,actual_used_bytes=actual.used,
            original_threshold_bytes=self.record['original_threshold_bytes'],
            incremental_threshold_bytes=self.record['min_free_bytes'],cached_text_bytes=self.record['cached_text_bytes'],
            admitted=actual.free>=self.record['min_free_bytes'],scope='Before original layout and continuous first timer; full job output budget retained at every source')
        self.events.append(event)
        return not event['admitted']


def install(module,admission):
    original=ast.parse(inspect.getsource(module.run));changed=copy.deepcopy(original)
    expected=ast.parse("shutil.disk_usage(out).free<48*graph['m']+72*graph['n']+(2<<30)",mode='eval').body
    target_dump=ast.dump(expected,include_attributes=False);matched=[]
    class Replace(ast.NodeTransformer):
        def visit_Compare(self,node):
            if ast.dump(node,include_attributes=False)==target_dump:
                matched.append(node)
                return ast.copy_location(ast.Call(func=ast.Name(id='_cached_staging_insufficient',ctx=ast.Load()),
                    args=[ast.Name(id=name,ctx=ast.Load()) for name in ('graph','out','index')],keywords=[]),node)
            return self.generic_visit(node)
    Replace().visit(changed)
    if len(matched)!=1:raise ValueError('Frozen worker no longer has exactly one known staging guard')
    restored=copy.deepcopy(changed)
    class Restore(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='_cached_staging_insufficient':return copy.deepcopy(expected)
            return self.generic_visit(node)
    Restore().visit(restored)
    if ast.dump(restored,include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise ValueError('Unexpected author function change outside storage guard')
    namespace=dict(module.run.__globals__,_cached_staging_insufficient=admission.insufficient)
    ast.fix_missing_locations(changed);exec(compile(changed,str(Path(__file__).resolve())+':storage_guard_only','exec'),namespace)
    module.run=namespace['run']
    return dict(original_function_ast_sha256=hashlib.sha256(ast.dump(original,include_attributes=False).encode()).hexdigest(),
        guard_replacements=1,all_other_function_ast_unchanged=True,physical_free_bytes_unmodified=True)
