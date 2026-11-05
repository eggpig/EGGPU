"""Four-core native lifecycle management for committed immutable snapshots.

Native allocations (including SSSP/WCC) are inventoried and released on owning
CUDA workers. This wrapper never substitutes predicted Python allocation bytes.
"""
from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from easygraph.gpu._bridge import NativePartitionSession, source_parameter, _NATIVE_WCC_OWNED_OUTPUT, _NATIVE_BFS_OWNED_OUTPUT, _NATIVE_SSSP_OWNED_OUTPUT, _NATIVE_PR_OWNED_OUTPUT
from easygraph.gpu.contracts import NodeDomain, QueryContract
from easygraph.gpu._core import native_policy, canonical_wcc, wcc_contract
from easygraph.gpu._weight_admission import scan_weight_payload
from easygraph.gpu import _core as _wcc_core
from easygraph.gpu._receipts import ManagedNativeSession, ReceiptRetentionController
from easygraph.gpu._controller import StateSwitchController, POLICIES


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def verify_manifest(path, document, *, numeric_certificate=None):
    """Bind immutable topology/value bytes; native importer also checks structure."""
    path=Path(path).resolve(); graph=document['graph']; total=0; maximum=0; weight_count=0
    if int(graph['vertex_count'])<=0:raise ValueError('Empty vertex domain')
    allowed={'int32':4,'uint64':8,'float64':8}
    for part in document['partitions']:
        for name,item in part['files'].items():
            payload=(path.parent/item['path']).resolve();payload.relative_to(path.parent)
            if item['dtype'] not in allowed:raise ValueError('Unsupported manifest dtype')
            if (payload.stat().st_size!=item['byte_size'] or item['count']*allowed[item['dtype']]!=item['byte_size']):
                raise ValueError('Snapshot payload identity mismatch')
            fused = None
            if name == 'weights' and item['dtype'] == 'float64':
                fused = scan_weight_payload(payload, certify=numeric_certificate is not None)
                observed_sha = fused[0]
            else:
                observed_sha = _sha(payload)
            if observed_sha != item['sha256']:
                raise ValueError('Snapshot payload identity mismatch')
            if name=='weights':
                if numeric_certificate is not None and (item['dtype']!='float64' or item['count']!=part['adjacency_count']):
                    raise ValueError('Static compact SSSP requires complete original float64 weights')
                if not item['count']:continue
                if fused is not None:
                    if fused[1] is not None:
                        raise ValueError(fused[1])
                    maximum = max(maximum, fused[2])
                    weight_count += fused[3]
                    continue
                # Preserve legacy rejection behavior for malformed non-FP64
                # declarations; valid native FP64 payloads take the fused path.
                weights=np.memmap(payload,mode='r',dtype='<f8')
                for first in range(0,len(weights),1<<18):
                    block=weights[first:first+(1<<18)]
                    if not np.all(np.isfinite(block)) or np.any(block<0):raise ValueError('Weights must be finite nonnegative f64')
                    if numeric_certificate is not None:
                        if np.any(block>255) or np.any(block!=np.floor(block)):
                            raise ValueError('Static compact SSSP requires exact integer weights in [0,255]')
                        maximum=max(maximum,int(block.max(initial=0)));weight_count+=len(block)
                del weights
        total+=part['owned_vertex_count']
    if total!=graph['vertex_count']:raise ValueError('Incomplete snapshot ownership')
    if numeric_certificate is not None:
        if not graph.get('weighted',False) or weight_count!=graph['adjacency_count']:
            raise ValueError('Static compact SSSP requires complete weighted input')
        bound=(int(graph['vertex_count'])-1)*maximum
        if bound>=2**32-1:raise ValueError('Static compact finite distance bound collides with u32 unreachable sentinel')
        numeric_certificate.update(encoding='u8_exact_u32_exact',vertex_count=int(graph['vertex_count']),
            weights_count=weight_count,max_weight=maximum,finite_distance_upper_bound=bound,
            public_input_dtype='float64',public_output_dtype='float64',unreachable='+inf',
            rule='Every original weight is an integer in [0,255]; (N-1)*maxW < UINT32_MAX.',
            timing='Full certification in manifest admission inside session initialization; native plan encoding also remains inside the first query.')
    return _sha(path)


def _native_wcc_output(cpp, labels, count):
    requested = os.environ.get('EGGPU_WCC_OUTPUT_FINALIZER', '0')
    if requested not in ('0', '1'):
        raise ValueError('EGGPU_WCC_OUTPUT_FINALIZER must be 0 or 1')
    eligible = (os.environ.get('EGGPU_WCC_CANONICAL_CERTIFICATE', '0') == '1'
        and type(labels) is np.ndarray and labels.dtype == np.dtype(np.uint64)
        and labels.shape == (count,) and labels.flags.c_contiguous and labels.flags.aligned
        and 0 <= count <= 2**53)
    proof = dict(version='native_uint64_canonical_materialize_v1',requested=requested,
        eligible=bool(eligible),used=False,adopted_private_owned=False,checked_vertices=0,
        route='legacy_canonical_wcc',reason='disabled' if requested=='0' else 'unsupported_input_or_certificate_off')
    if requested == '1' and eligible:
        # The binding always allocates a new private owning array. Nothing
        # retains or exposes that array before the _own_result publication.
        result = cpp.cpp_egmgpu_finalize_wcc_uint64(labels, count)
        if result['version'] != proof['version'] or result['checked_vertices'] != count or type(result['certified']) is not bool:
            raise RuntimeError('Wrong WCC output-finalizer ABI')
        proof['checked_vertices'] = count
        if result['certified']:
            _wcc_core._WCC_CANONICAL_CERTIFICATE_AUDIT['calls'] += 1
            _wcc_core._WCC_CANONICAL_CERTIFICATE_AUDIT['certified'] += 1
            proof.update(used=True,adopted_private_owned=True,route='fused_uint64_certification_and_materialization',reason='exact_canonical_labels')
            return result['values'], True, proof
        # Keep generic partition normalization and its exact counter semantics.
        proof['reason'] = 'noncanonical_fallback'
    return canonical_wcc(labels, count), False, proof


def _wcc_worker_output_flag():
    requested=os.environ.get('EGGPU_NATIVE_WCC_WORKER_FP64_OUTPUT','0')
    if requested not in ('0','1'):
        raise ValueError('EGGPU_NATIVE_WCC_WORKER_FP64_OUTPUT must be 0 or 1')
    if requested=='1' and (os.environ.get('EGGPU_WCC_OUTPUT_FINALIZER','0')!='1' or os.environ.get('EGGPU_WCC_CANONICAL_CERTIFICATE','0')!='1'):
        raise ValueError('WCC worker FP64 output requires finalizer=1 and canonical certificate=1')
    return requested


def _native_wcc_worker_output(raw,count):
    threads=16 if raw.get('core_output_team_policy')=='team16' else 1
    expected=dict(version='native_wcc_worker_public_fp64_v1',requested='1',eligible=True,used=True,
        reason='complete_private_worker_scatter_and_certificate',producer='numpy_owned_fp64_worker_scatter',
        complete_vertices=count,converted_vertices=count,certified_vertices=count,invalid_vertices=0,noncanonical_vertices=0,
        certificate_threads=threads,certificate_preserved=True,values_bytes=count*8,
        intermediate_uint64_bytes_avoided=count*8,intermediate_payload_write_bytes_avoided=count*8)
    proof=raw.get('wcc_worker_public_output');observation=raw.get('wcc_public_certificate_thread_observation')
    if (not 0<=count<=2**53 or raw.get('output_representation')!='owning_fp64_dense'
            or raw.get('wcc_algorithm')!='forest' or raw.get('host_output_mode')!='pinned'
            or raw.get('result_merge_mode')!='canonical_direct' or raw.get('host_assembly_strategy') not in ('worker','tiled')
            or raw.get('core_output_team_policy') not in ('team16','original')
            or not isinstance(proof,dict) or set(proof)!=set(expected)
            or any(type(proof[k]) is not type(v) or proof[k]!=v for k,v in expected.items())
            or not isinstance(observation,dict) or type(observation.get('actual')) is not int
            or observation['actual']!=threads or len(observation.get('threads',[]))!=threads):
        raise RuntimeError('WCC worker output proof differs')
    values=raw['labels_dense']
    if (type(values) is not np.ndarray or values.dtype!=np.dtype(np.float64) or values.shape!=(count,)
            or not values.flags.c_contiguous or not values.flags.aligned or not values.flags.owndata or values.base is not None):
        raise RuntimeError('WCC worker producer did not return complete private FP64 storage')
    _wcc_core._WCC_CANONICAL_CERTIFICATE_AUDIT['calls']+=1
    _wcc_core._WCC_CANONICAL_CERTIFICATE_AUDIT['certified']+=1
    return values,dict(proof,adopted_private_owned=True,route='adopt_private_worker_fp64_output')


def _bfs_output_flags():
    owned = os.environ.get('EGGPU_NATIVE_BFS_OWNED_OUTPUT', '0')
    fused = os.environ.get('EGGPU_NATIVE_BFS_FUSED_OUTPUT', '0')
    if owned not in ('0', '1'):
        raise ValueError('EGGPU_NATIVE_BFS_OWNED_OUTPUT must be 0 or 1')
    if fused not in ('0', '1'):
        raise ValueError('EGGPU_NATIVE_BFS_FUSED_OUTPUT must be 0 or 1')
    if fused == '1' and owned != '1':
        raise ValueError('EGGPU_NATIVE_BFS_FUSED_OUTPUT=1 requires EGGPU_NATIVE_BFS_OWNED_OUTPUT=1')
    return owned, fused


def _native_bfs_output(cpp, raw_values, count, owned_output, fused_output):
    # Only complete native query-owned arrays enter this integration. Public
    # caller arrays and capsule storage never receive an ownership capability.
    eligible = (type(raw_values) is np.ndarray and raw_values.dtype == np.dtype(np.int64)
        and raw_values.shape == (count,) and raw_values.flags.c_contiguous
        and raw_values.flags.aligned and 0 <= count <= 2**53)
    proof = dict(version='native_bfs_output_materialization_v2',requested=owned_output,
        used=owned_output=='1',adopted_private_owned=owned_output=='1',
        route='adopt_private_where_output' if owned_output=='1' else 'legacy_bridge_copy',
        producer='numpy_where_int64_unreachable_to_float64',
        fused_requested=fused_output,fused_eligible=bool(eligible),fused_used=False,
        fused_reason='disabled' if fused_output=='0' else 'unsupported_input',
        checked_vertices=0,negative_vertices=None,invalid_nonnegative_vertices=None,
        finalizer_version=None,finalizer_candidate_bytes=0,
        accounting_scope='Known host output payload sizes; not measured elapsed time, GPU allocation, or peak memory')
    values = None
    if fused_output == '1' and eligible:
        result = cpp.cpp_egmgpu_finalize_bfs_int64(raw_values, count)
        if (result['version'] != 'native_int64_bfs_materialize_v1'
                or result['checked_vertices'] != count or type(result['certified']) is not bool
                or not 0 <= result['negative_vertices'] <= count
                or not 0 <= result['invalid_nonnegative_vertices'] <= count-result['negative_vertices']
                or result['certified'] != (result['invalid_nonnegative_vertices'] == 0)):
            raise RuntimeError('Wrong BFS output-finalizer ABI')
        proof.update(finalizer_version=result['version'],checked_vertices=count,
            negative_vertices=result['negative_vertices'],
            invalid_nonnegative_vertices=result['invalid_nonnegative_vertices'],
            finalizer_candidate_bytes=8*count)
        if result['certified']:
            values = result['values']
            if values is None:
                raise RuntimeError('Certified BFS finalizer returned no output')
            proof.update(fused_used=True,fused_reason='exact_bfs_distance_domain',
                route='adopt_private_fused_output',producer='native_int64_bfs_materialize_v1')
        else:
            if result['values'] is not None:
                raise RuntimeError('Uncertified BFS finalizer exposed a candidate')
            proof['fused_reason'] = 'nonnegative_distance_outside_graph_fallback'
    if values is None:
        # Keep the original value and rounding semantics on unsupported inputs;
        # the existing bridge still checks public shape, dtype, and ownership.
        values = np.asarray(raw_values)
        values = np.where(values < 0, np.inf, values)
    proof.update(values_bytes=int(values.nbytes),
        bridge_copy_bytes_avoided=int(values.nbytes) if owned_output=='1' else 0,
        numpy_mask_bytes_avoided=count if proof['fused_used'] else 0)
    return values, owned_output=='1', proof


def _bfs_worker_output_flag(owned, fused):
    requested = os.environ.get('EGGPU_NATIVE_BFS_WORKER_FP64_OUTPUT', '0')
    if requested not in ('0', '1'):
        raise ValueError('EGGPU_NATIVE_BFS_WORKER_FP64_OUTPUT must be 0 or 1')
    if requested == '1' and (owned != '1' or fused != '1'):
        raise ValueError('BFS worker FP64 output requires owned=1 and fused=1')
    return requested


def _native_bfs_worker_output(raw, count):
    proof = raw.get('bfs_worker_public_output')
    unreachable = raw.get('unreachable_vertex_count')
    if type(unreachable) is not int or not 0 <= unreachable <= count:
        raise RuntimeError('Invalid BFS worker unreachable count')
    expected = dict(version='native_bfs_worker_public_fp64_v1',requested='1',eligible=True,used=True,
        reason='complete_private_worker_scatter',producer='numpy_owned_fp64_worker_scatter_normalize',
        complete_vertices=count,checked_vertices=count,unreachable_vertices=unreachable,invalid_vertices=0,
        values_bytes=count*8,intermediate_int64_bytes_avoided=count*8,
        intermediate_payload_read_write_bytes_avoided=count*16)
    if (not 0 <= count <= 2**53 or raw.get('output_representation') != 'owning_fp64_dense'
            or raw.get('host_assembly') != 'workers' or raw.get('result_assembly') != 'direct'
            or raw.get('output_mode') != 'move' or not isinstance(proof,dict)
            or set(proof) != set(expected) or any(type(proof[k]) is not type(v) or proof[k] != v for k,v in expected.items())):
        raise RuntimeError('BFS worker output proof differs')
    values = raw['values_dense']
    if (type(values) is not np.ndarray or values.dtype != np.dtype(np.float64) or values.shape != (count,)
            or not values.flags.c_contiguous or not values.flags.aligned
            or not values.flags.owndata or values.base is not None):
        raise RuntimeError('BFS worker producer did not return complete private FP64 storage')
    return values, dict(proof,adopted_private_owned=True,bridge_copy_bytes_avoided=count*8,
        numpy_mask_bytes_avoided=count,finalizer_candidate_bytes=0,
        route='adopt_private_worker_fp64_output',accounting_scope='Host payload sizes; not measured traffic, peak memory or elapsed time')


def _sssp_output_flag():
    value = os.environ.get('EGGPU_NATIVE_SSSP_OWNED_OUTPUT', '0')
    if value not in ('0', '1'):
        raise ValueError('EGGPU_NATIVE_SSSP_OWNED_OUTPUT must be 0 or 1')
    return value


def _native_sssp_output(raw, count, requested):
    eligible = raw.get('host_assembly_policy') == 'worker'
    used = requested == '1' and eligible
    expected = dict(version='native_sssp_host_ownership_v1',requested=requested,
        eligible=eligible,used=used,reason='private_numpy_owner' if used else 'unsupported_output_route' if requested=='1' else 'disabled',
        producer='numpy_owned_fp64_worker_scatter' if used else 'capsule_fp64_worker_scatter' if eligible else 'caller_vector_capsule',
        complete_vertices=count,values_bytes=count*8)
    proof = raw.get('sssp_native_output_ownership')
    if (raw.get('output_representation') != 'owning_fp64_dense' or not isinstance(proof,dict)
            or set(proof) != set(expected) or any(type(proof[k]) is not type(v) or proof[k] != v for k,v in expected.items())):
        raise RuntimeError('SSSP native ownership proof differs')
    values = raw['values_dense']
    if (type(values) is not np.ndarray or values.dtype != np.dtype(np.float64) or values.shape != (count,)
            or not values.flags.c_contiguous or not values.flags.aligned
            or (used and (not values.flags.owndata or values.base is not None))):
        raise RuntimeError('SSSP producer did not return the declared complete private FP64 storage')
    return values, used, proof


def _pr_output_flag():
    value = os.environ.get('EGGPU_NATIVE_PR_OWNED_OUTPUT', '0')
    if value not in ('0', '1'):
        raise ValueError('EGGPU_NATIVE_PR_OWNED_OUTPUT must be 0 or 1')
    return value


def _native_pr_output(raw, count, requested):
    if raw.get('result_assembly') not in ('direct', 'zeroed') or raw.get('output_mode') not in ('copy', 'move', 'pinned'):
        raise RuntimeError('Unknown PR output route')
    eligible = raw['result_assembly'] == 'direct' and raw['output_mode'] != 'copy'
    used = requested == '1' and eligible
    expected = dict(version='native_pr_host_ownership_v1',requested=requested,
        eligible=eligible,used=used,reason='private_numpy_owner' if used else 'unsupported_output_route' if requested=='1' else 'disabled',
        producer='numpy_owned_fp64_canonical_scatter' if used else 'capsule_fp64_canonical_scatter' if eligible else 'legacy_pr_materialization',
        complete_vertices=count,values_bytes=count*8)
    proof = raw.get('pr_native_output_ownership')
    if (not isinstance(proof,dict) or set(proof) != set(expected)
            or any(type(proof[k]) is not type(v) or proof[k] != v for k,v in expected.items())):
        raise RuntimeError('PR native ownership proof differs')
    values = raw['values_dense']
    if (type(values) is not np.ndarray or values.dtype != np.dtype(np.float64) or values.shape != (count,)
            or not values.flags.c_contiguous or not values.flags.aligned
            or (used and (not values.flags.owndata or values.base is not None))):
        raise RuntimeError('PR producer did not return the declared complete private FP64 storage')
    return values, used, proof


class UnifiedNativeSession(ManagedNativeSession):
    capabilities=frozenset({'bfs','pr','weighted_pr','sssp','wcc','immutable_partitions','multiple_devices','native_allocation_inventory','register_snapshot'})

    def __init__(self,cpp,manifests,*,devices=(0,),budgets=None,retention='scratch_first',inventory_mode='recheck',batch_devices=True,policy=None):
        if policy is not None and policy not in POLICIES:raise ValueError('Unknown unified retention policy')
        self.snapshot_hashes={}
        self.sssp_policy,self.verify_sssp=native_policy('f64')
        super().__init__(cpp,manifests,devices=devices,budgets=budgets,retention=retention,inventory_mode=inventory_mode,batch_devices=batch_devices)
        if policy is not None:
            bindings={self.metadata[name]['id']:digest for name,digest in self.snapshot_hashes.items()}
            self.controller=StateSwitchController(cpp,devices,budgets,policy=policy,inventory_mode=inventory_mode,batch_devices=batch_devices,graph_bindings=bindings)
        inventory=self.resource_inventory()
        if inventory.get('algorithms')!=['bfs','pr','sssp','wcc']:
            self.close();raise RuntimeError('Require unified four-core native allocation ABI')

    def _validate_manifest_document(self,name,path,document):
        self.snapshot_hashes[name]=verify_manifest(path,document)

    def register_snapshot(self,name,manifest_path):
        """Publish a fully committed host manifest, never mutate an active view."""
        with self._lock:
            if self.closed:raise RuntimeError('Session closed')
            path=Path(manifest_path).resolve();document=json.loads(path.read_text())
            digest=verify_manifest(path,document)
            if name in self.graphs:
                if self.snapshot_hashes[name]==digest:return self.describe(name)
                raise ValueError('Existing snapshot name is immutable; register a new name')
            if sorted(p['device_id'] for p in document['partitions'])!=sorted(self.devices):raise ValueError('Snapshot placement differs from session')
            identity=(document['graph']['id'],document['graph'].get('generation'))
            for old in self.metadata:
                previous=(self.metadata[old]['id'],self.metadata[old].get('generation'))
                if identity==previous and self.snapshot_hashes[old]!=digest:
                    raise ValueError('Different content must use a distinct graph identity or generation')
            # Import first. Failures do not publish a partial Python registration.
            native=self.cpp.cpp_egmgpu_load_partition_manifest(str(path),import_policy='structural')
            nodes=np.arange(document['graph']['vertex_count'],dtype=np.int64);nodes.flags.writeable=False
            self.graphs[name]=native;self.metadata[name]=document['graph'];self.nodes[name]=nodes
            self._manifest_documents[name]=document;self._manifest_paths[name]=str(path)
            self._domains[name]=NodeDomain.from_nodes(nodes);self.snapshot_hashes[name]=digest
            if isinstance(self.controller,StateSwitchController):self.controller.graph_bindings[document['graph']['id']]=digest
            return self.describe(name)

    def query(self,graph_id,algorithm,parameters=None):
        begin=perf_counter()
        with self._lock:
            if self.closed:raise RuntimeError('Session closed')
            parameters=dict(parameters or {});graph=self.graphs[graph_id];meta=self.metadata[graph_id];n=int(meta['vertex_count'])
            weighted=meta.get('weighted',False)
            if algorithm in ('bfs','pr','wcc') and weighted:raise ValueError('Weighted graph requires explicit sssp/weighted_pr projection')
            if algorithm in ('weighted_pr','sssp') and not weighted:raise ValueError('Weighted operator requires explicit weighted snapshot')
            start=perf_counter();private_owned=None;wcc_output=None;bfs_output=None;sssp_output=None;pr_output=None
            if algorithm=='bfs':
                owned_output,fused_output=_bfs_output_flags()
                worker_output=_bfs_worker_output_flag(owned_output,fused_output)
                source=source_parameter(parameters)
                if not 0<=source<n:raise ValueError('Source outside snapshot domain')
                cpp_begin=perf_counter()
                if worker_output == '1':
                    raw=self.cpp.cpp_egmgpu_partitioned_bfs(graph,source,devices=self.devices,dense_output=True,frontier_policy='adaptive',public_fp64_output=True)
                else:
                    raw=self.cpp.cpp_egmgpu_partitioned_bfs(graph,source,devices=self.devices,dense_output=True,frontier_policy='adaptive')
                cpp_seconds=perf_counter()-cpp_begin;output_begin=perf_counter()
                if worker_output == '1':
                    values,bfs_output=_native_bfs_worker_output(raw,n);bfs_owned=True
                else:
                    values,bfs_owned,bfs_output=_native_bfs_output(self.cpp,raw['values_dense'],n,owned_output,fused_output)
                output_seconds=perf_counter()-output_begin
                if bfs_owned:private_owned=_NATIVE_BFS_OWNED_OUTPUT
                contract=QueryContract('bfs',source,'unweighted',None,None,None,'+inf',None)
            elif algorithm in ('pr','weighted_pr'):
                if parameters:raise ValueError('Fixed PR parameters: alpha=.85,max_iter=1000,tol=1e-9/n')
                requested=_pr_output_flag();pr_cpp_begin=perf_counter()
                raw=self.cpp.cpp_egmgpu_partitioned_pagerank(graph,.85,1000,1e-9/n,self.devices)
                pr_cpp_seconds=perf_counter()-pr_cpp_begin;pr_proof_begin=perf_counter()
                values,pr_owned,pr_output=_native_pr_output(raw,n,requested)
                pr_proof_seconds=perf_counter()-pr_proof_begin
                if pr_owned:private_owned=_NATIVE_PR_OWNED_OUTPUT
                contract=QueryContract(algorithm,None,'input edge weights' if weighted else 'unweighted',.85,1000,1e-9/n,None,None)
            elif algorithm=='sssp':
                source=source_parameter(parameters)
                if not 0<=source<n:raise ValueError('Source outside snapshot domain')
                requested=_sssp_output_flag();cpp_begin=perf_counter()
                raw=self.cpp.cpp_egmgpu_partitioned_sssp(graph,source,self.sssp_policy.delta,0,self.devices,dense_output=True,execution_policy=self.sssp_policy.native_options())
                cpp_seconds=perf_counter()-cpp_begin;policy_begin=perf_counter()
                self.verify_sssp(raw,self.sssp_policy)
                policy_seconds=perf_counter()-policy_begin;proof_begin=perf_counter()
                values,sssp_owned,sssp_output=_native_sssp_output(raw,n,requested)
                output_proof_seconds=perf_counter()-proof_begin
                if sssp_owned:private_owned=_NATIVE_SSSP_OWNED_OUTPUT
                contract=QueryContract('sssp',source,'input edge weights',None,None,None,'+inf',None)
            elif algorithm=='wcc':
                if parameters or meta.get('directed',False) or self._manifest_documents[graph_id]['view']['orientation']!='undirected':raise ValueError('WCC requires explicit unweighted weak projection')
                worker_output=_wcc_worker_output_flag();cpp_begin=perf_counter()
                if worker_output=='1':
                    raw=self.cpp.cpp_egmgpu_partitioned_wcc(graph,100000,self.devices,dense_output=True,public_fp64_output=True)
                else:
                    raw=self.cpp.cpp_egmgpu_partitioned_wcc(graph,100000,self.devices,dense_output=True)
                wcc_cpp_seconds=perf_counter()-cpp_begin;output_begin=perf_counter()
                if worker_output=='1':
                    values,wcc_output=_native_wcc_worker_output(raw,n);wcc_owned=True
                else:
                    values,wcc_owned,wcc_output=_native_wcc_output(self.cpp,raw['labels_dense'],n)
                wcc_output_seconds=perf_counter()-output_begin;contract=wcc_contract()
                if wcc_owned:private_owned=_NATIVE_WCC_OWNED_OUTPUT
            else:raise NotImplementedError(algorithm)
            native_seconds=perf_counter()-start
            if raw.get('converged',True) is not True or list(raw.get('devices_used',()))!=self.devices:raise RuntimeError('Native convergence or participation failed')
            metrics={k:v for k,v in raw.items() if k not in ('values','values_dense','labels_dense')}
            metrics['native_complete_output_seconds']=native_seconds
            if wcc_output is not None:
                metrics['wcc_output_finalization']=wcc_output
                metrics['wcc_cpp_call_seconds']=wcc_cpp_seconds
                metrics['wcc_output_finalization_seconds']=wcc_output_seconds if worker_output=='0' else 0.0
                metrics['wcc_output_proof_seconds']=wcc_output_seconds if worker_output=='1' else 0.0
            if bfs_output is not None:
                metrics['bfs_output_ownership']=bfs_output
                metrics['bfs_cpp_call_seconds']=cpp_seconds
                metrics['bfs_output_finalization_seconds']=output_seconds if worker_output=='0' else 0.0
                metrics['bfs_output_proof_seconds']=output_seconds if worker_output=='1' else 0.0
            if sssp_output is not None:
                metrics['sssp_output_ownership']=sssp_output
                metrics['sssp_cpp_call_seconds']=cpp_seconds
                metrics['sssp_policy_validation_seconds']=policy_seconds
                metrics['sssp_output_proof_seconds']=output_proof_seconds
            if pr_output is not None:
                metrics['pr_output_ownership']=pr_output
                metrics['pr_cpp_call_seconds']=pr_cpp_seconds
                metrics['pr_output_proof_seconds']=pr_proof_seconds
            own_key='sssp' if algorithm=='sssp' else 'pr' if pr_output is not None else None
            own_result_begin=perf_counter() if own_key is not None else None
            bfs_own_begin=perf_counter() if algorithm=='bfs' else None
            wcc_own_begin=perf_counter() if algorithm=='wcc' else None
            result=self._own_result(graph_id,algorithm,values,begin,metrics,parameters,contract=contract,_private_owned=private_owned)
            if bfs_own_begin is not None:metrics['bfs_own_result_seconds']=perf_counter()-bfs_own_begin
            if wcc_own_begin is not None:metrics['wcc_own_result_seconds']=perf_counter()-wcc_own_begin
            if own_result_begin is not None:metrics[own_key+'_own_result_seconds']=perf_counter()-own_result_begin
            resource_begin=perf_counter() if own_key is not None else None
            managed_algorithm='pr' if algorithm=='weighted_pr' else algorithm
            if isinstance(self.controller,StateSwitchController):self.controller.set_request(meta['id'],managed_algorithm,parameters)
            resource=self.controller.finish_query(meta['id'],managed_algorithm)
            resource['budget_scope']='Completed-query retained BFS/PR/SSSP/WCC native allocations and shared catalog; excludes transient construction peak and total CUDA process memory'
            metrics['resource_management']=resource;self.last_resource_record=resource
            if resource_begin is not None:metrics[own_key+'_resource_tail_seconds']=perf_counter()-resource_begin
            return replace(result,seconds=perf_counter()-begin)

    def release_all(self):
        """Release all currently idle native graph views/workspaces, keep host plans."""
        with self._lock:
            state=self.resource_inventory()
            owners=[o['owner_id'] for row in state['per_device'] for o in row['owners'] if o['kind'] in ('view','workspace')]
            return self.release(owners,release_idle=True,release_workspaces=True)

    def capability_descriptor(self):
        return replace(super().capability_descriptor(),algorithms=('bfs','pr','sssp','wcc','weighted_pr'),
            physical_ownership='Actual native four-core per-device allocation union, views and workspaces; owning host outputs',
            control_routes=('register_committed_snapshot','automatic_cache_reuse','boundary_inventory','selective_view_release','idle_workspace_release','per_device_retention_budget'),
            qualification='P1/P2 qualification evidence is separate from API availability',
            limitations=('Committed immutable snapshots; no in-place dynamic topology update',
                         'Request-boundary retained-state budgets; no construction-peak or total-VRAM bound',
                         'No cross-device physical allocation sharing',
                         'No simultaneous mutable workspace borrowing; pooling is per algorithm and fixed worker',
                         'Shared extent range leases from SG v11 are not implemented in this native executor'))
