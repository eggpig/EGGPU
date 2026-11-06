"""Focused real-free-byte admission and exact one-guard AST check; no GPU."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import disk_admission as disk
from build import bind


def main():
    p=argparse.ArgumentParser();p.add_argument('--scratch',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    campaign=Path(__file__).resolve().parents[1]
    worker=campaign/'baselines/sg_author_worker.py'
    sys.path.insert(0,str(worker.parent))
    spec=importlib.util.spec_from_file_location('_disk_guard_author',worker);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    n,m=41652230,1468364884
    with tempfile.TemporaryDirectory(prefix='disk-admission-',dir=a.scratch) as directory:
        root=Path(directory);blob=root/'edges';blob.write_bytes(b'1 2 1\n')
        record=disk.bound(n,'bfs');record.update(m=m,cached_text=bind(blob),cached_text_bytes=30835662564,
            original_threshold_bytes=48*m+72*n+disk.RESERVE)
        admission=disk.Admission(record,root,root)
        physical=shutil.disk_usage;usage=type(physical(root))
        try:
            disk.shutil.disk_usage=lambda path:usage(100_000_000_000,62_700_000_000,37_300_000_000)
            assert 37_300_000_000<record['original_threshold_bytes']
            assert not admission.insufficient(dict(n=n,m=m),root,0)
            assert admission.events[-1]['actual_free_bytes']==37_300_000_000
            disk.shutil.disk_usage=lambda path:usage(100_000_000_000,100_000_000_000-(record['min_free_bytes']-1),record['min_free_bytes']-1)
            assert admission.insufficient(dict(n=n,m=m),root,1)
            assert admission.events[-1]['actual_free_bytes']==record['min_free_bytes']-1
        finally:disk.shutil.disk_usage=physical
        patch=disk.install(module,admission)
        assert patch['guard_replacements']==1 and patch['all_other_function_ast_unchanged'] and patch['physical_free_bytes_unmodified']
        result=dict(status='PASS_CPU_CACHED_STORAGE_ADMISSION',gpu_executed=False,
            simulated_filesystem_readings=True,
            original_worker=bind(worker),helper=bind(disk.__file__),bound_bfs_sssp=disk.bound(n,'bfs'),bound_pr=disk.bound(n,'pr'),
            original_threshold_bytes=record['original_threshold_bytes'],checks=admission.events,patch=patch,
            native_executed=False,numerical_guard_changes=False,host_gpu_guard_changes=False)
        a.receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
