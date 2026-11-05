"""Native cuGraph MG adapter. CSR mmap -> rank-local COO in RAM only."""
from __future__ import annotations
import gc
import hashlib
import ipaddress
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
import numpy as np
from cugraph_mg_platform import worker_binding, toolkit_probe, set_cpu_affinity, resolve_distribution

PACKAGES=("cugraph","pylibcugraph","dask","distributed","dask-cuda","dask-cudf","cudf","rmm","raft-dask","numpy","cupy")

def make_graph(plc, cp, handle, edges, nodes, weights, degree, function, multi):
    src = cp.asarray(edges[:, 0], dtype=cp.int32)
    dst = cp.asarray(edges[:, 1], dtype=cp.int32)
    vertices = cp.asarray(nodes, dtype=cp.int32)
    edge_weights = (cp.asarray(weights, dtype=cp.float64) if function == "sssp"
                    else cp.ones(len(edges), dtype=cp.float64) if function == "pr" else None)
    properties = plc.GraphProperties(is_symmetric=function == "wcc", is_multigraph=False)
    options = dict(store_transposed=function == "pr", do_expensive_check=False,
                   drop_self_loops=False, drop_multi_edges=False, symmetrize=False)
    if multi:
        graph = plc.MGGraph(resource_handle=handle, graph_properties=properties,
            src_array=[src], dst_array=[dst],
            weight_array=[edge_weights] if edge_weights is not None else None,
            vertices_array=[vertices], num_arrays=1, **options)
    else:
        graph = plc.SGGraph(handle, properties, src, dst,
            weight_array=edge_weights, vertices_array=vertices,
            renumber=False, input_array_format="COO", **options)
    return dict(graph=graph, handle=handle, vertices=vertices,
                degree=cp.asarray(degree, dtype=cp.float64) if degree is not None else None)


def public_query(plc, cp, state, function, source_owner=True, source=0):
    """Actual public dispatch, also dependency-injected in CPU contract checks."""
    handle, graph = state["handle"], state["graph"]
    if function == "bfs":
        values, _, vertices = plc.bfs(handle, graph=graph,
            sources=cp.asarray([source] if source_owner else [], dtype=cp.int32),
            direction_optimizing=False, depth_limit=0,
            compute_predecessors=False, do_expensive_check=False)
    elif function == "sssp":
        vertices, values, _ = plc.sssp(handle, graph=graph, source=source,
            cutoff=float("inf"), compute_predecessors=True, do_expensive_check=False)
    elif function == "wcc":
        vertices, values = plc.weakly_connected_components(handle, graph=graph,
            offsets=None, indices=None, weights=None, labels=None, do_expensive_check=False)
    elif function == "pr":
        vertices, values = plc.pagerank(handle, graph=graph,
            precomputed_vertex_out_weight_vertices=state["vertices"],
            precomputed_vertex_out_weight_sums=state["degree"],
            initial_guess_vertices=None, initial_guess_values=None,
            alpha=.85, epsilon=1e-9, max_iterations=10000,
            do_expensive_check=False, fail_on_nonconvergence=True)
    else:
        raise ValueError(function)
    ids, host = cp.asnumpy(vertices), cp.asnumpy(values)
    if function == "bfs":
        if host.dtype.kind != "i":
            raise ValueError("BFS must return integer hop counts")
        unreachable = host == np.iinfo(host.dtype).max
        host = host.astype(np.float64)
        host[unreachable] = np.inf
    elif function == "sssp":
        if host.dtype != np.float64:
            raise ValueError("SSSP must use the explicit FP64 weights")
        host[host == np.finfo(host.dtype).max] = np.inf
    elif function == "pr" and host.dtype != np.float64:
        raise ValueError("PR must execute with native FP64 ranks")
    elif function == "wcc" and host.dtype.kind not in "iu":
        raise ValueError("WCC must return integer labels")
    return dict(vertices=ids, values=host)




def cluster_options(protocol, job, uuids):
    return dict(CUDA_VISIBLE_DEVICES=uuids, n_workers=job["gpu_count"], threads_per_worker=1,
        protocol='tcp',host='127.0.0.1',scheduler_port=0,dashboard_address=None,
        scheduler_kwargs=dict(dashboard=False,dashboard_address='127.0.0.1:0'),
        worker_dashboard_address=None,worker_port=0,
        local_directory=protocol['scratch']+'/'+job['id'],
        memory_limit=protocol["resources"]["host_bytes_per_rank"],
        device_memory_limit=protocol["resources"]["device_memory_limit_bytes"],
        rmm_pool_size=protocol["resources"]["rmm_pool_bytes"],
        rmm_maximum_pool_size=protocol["resources"]["rmm_max_bytes"],
        rmm_allocator_external_lib_list=['cupy'],enable_cudf_spill=False,silence_logs=30)


def observe_services(dask_scheduler=None,dask_worker=None):
    """Inspect this owned Dask process; no network requests or external services."""
    node=dask_scheduler if dask_scheduler is not None else dask_worker
    if node is None:raise ValueError('Owned scheduler/worker required')
    server=node.http_server
    addresses=[dict(host=str(sock.getsockname()[0]),port=int(sock.getsockname()[1]))
               for sock in server._sockets.values()]
    applications=[type(app).__module__+'.'+type(app).__qualname__
                  for app in node.http_application.applications]
    return dict(pid=os.getpid(),address=node.address,http_bound_addresses=addresses,
        http_service_port=int(server.port),application_types=applications,
        bokeh_application_types=[name for name in applications if name.startswith('bokeh.')],
        worker_dashboard_flag=getattr(node,'_dashboard',None),
        services={name:int(service.port) for name,service in node.services.items()
                  if hasattr(service,'port')})


def check_cluster_shape(info,scheduler,workers,expected_count,host_bytes):
    if len(info['workers'])!=expected_count or set(workers)!=set(info['workers']):
        raise RuntimeError('Incorrect owned cluster worker shape')
    if scheduler['address']!=info['address']:
        raise RuntimeError('Scheduler address observation changed')
    for address,metadata in info['workers'].items():
        if metadata['memory_limit']!=host_bytes or metadata['nthreads']!=1:
            raise RuntimeError('Worker resource options differ from protocol')
        if workers[address]['address']!=address or workers[address]['worker_dashboard_flag'] is not False:
            raise RuntimeError('Worker identity/dashboard configuration changed')
    for value in [scheduler,*workers.values()]:
        host=urlparse(value['address']).hostname
        if host is None or not ipaddress.ip_address(host).is_loopback:
            raise RuntimeError('Owned Dask RPC is not loopback')
        if value['bokeh_application_types'] or value['application_types']:
            raise RuntimeError('Unexpected attached HTTP application; Bokeh must be disabled')
        if not value['http_bound_addresses']:
            raise RuntimeError('Missing actual HTTP socket observation')
        for bound in value['http_bound_addresses']:
            if not ipaddress.ip_address(bound['host']).is_loopback or not 0<bound['port']<65536:
                raise RuntimeError('Owned HTTP service is not loopback with a live port')
        if value['http_service_port'] not in [x['port'] for x in value['http_bound_addresses']]:
            raise RuntimeError('HTTP service port differs from bound sockets')
    if info.get('services',{}).get('dashboard')!=scheduler['http_service_port']:
        raise RuntimeError('Scheduler HTTP service metadata differs from observed socket')




def complete_output(pieces, n, function):
    """Linear complete-domain scatter; WCC labels remain native partition IDs."""
    if function not in ('bfs','sssp','wcc','pr') or n <= 0: raise ValueError('Invalid output contract')
    ids = np.concatenate([p['vertices'] for p in pieces])
    raw = np.concatenate([p['values'] for p in pieces])
    if ids.shape != (n,) or raw.shape != (n,) or ids.dtype.kind not in 'iu':
        raise ValueError('Missing full vertex-domain output')
    if np.any(ids < 0) or np.any(ids >= n): raise ValueError('Out-of-domain vertex ID')
    # Check duplicates within and across pieces before any indexed assignment.
    if not np.all(np.bincount(ids.astype(np.int64, copy=False), minlength=n) == 1):
        raise ValueError('Duplicate or missing vertex ID')
    if (function == 'wcc' and raw.dtype.kind not in 'iu') or (function != 'wcc' and raw.dtype != np.float64):
        raise ValueError('Wrong native output precision/type')
    result = np.empty(n, dtype=np.int64 if function == 'wcc' else np.float64)
    result[ids] = raw
    result.flags.writeable = False
    return result


def describe_graph(graph):
    files = {}
    for role in ('offsets','indices','weights'):
        value = graph.get(role)
        if value is not None:
            if not isinstance(value, np.memmap): raise ValueError('Use canonical mmap CSR input')
            files[role] = str(Path(value.filename).resolve())
    return dict(n=int(graph['n']), m=int(graph['m']), directed=bool(graph['directed']), files=files)


def rank_host(descriptor, rank, count, function):
    """Generate only this rank's contiguous edge interval, never a disk COO."""
    n, m = descriptor['n'], descriptor['m']; paths = descriptor['files']
    if not 0 <= rank < count or function not in ('bfs','sssp','wcc','pr'): raise ValueError('Invalid rank/function')
    if (function == 'wcc') == descriptor['directed']: raise ValueError('WCC requires explicit weak CSR; others directed CSR')
    offsets = np.memmap(paths['offsets'], mode='r', dtype='<i4', shape=(n + 1,))
    indices = np.memmap(paths['indices'], mode='r', dtype='<i4', shape=(m,)) if m else np.empty(0,dtype=np.int32)
    begin, end = m * rank // count, m * (rank + 1) // count
    edges = np.empty((end - begin, 2), dtype=np.int32)
    if end > begin:
        lo = int(np.searchsorted(offsets, begin, side='right')) - 1
        hi = int(np.searchsorted(offsets, end - 1, side='right')) - 1
        bounds = np.clip(offsets[lo:hi+2], begin, end)
        lengths = np.diff(bounds)
        edges[:,0] = np.repeat(np.arange(lo,hi+1,dtype=np.int32),lengths)
        edges[:,1] = indices[begin:end]
    nodes = np.arange(rank, n, count, dtype=np.int32)
    degree = (offsets[nodes+1].astype(np.float64) - offsets[nodes]) if function == 'pr' else None
    weights = None
    if function == 'sssp':
        weights = np.memmap(paths['weights'], mode='r', dtype='<f8', shape=(m,))[begin:end]
    return edges, nodes, weights, degree


def build_rank(session_id, descriptor, rank, count, function):
    import cupy as cp
    import pylibcugraph as plc
    import cugraph.dask.comms.comms as comms
    if cp.cuda.runtime.getDevice() != 0: raise RuntimeError('Rank must use assigned logical device zero')
    handle = plc.ResourceHandle(comms.get_handle(session_id).getHandle())
    arrays = rank_host(descriptor, rank, count, function)
    state = make_graph(plc, cp, handle, *arrays, function, True)
    cp.cuda.runtime.deviceSynchronize()
    return state


def query_rank(state, function, source_owner, source):
    import cupy as cp
    import pylibcugraph as plc
    if cp.cuda.runtime.getDevice() != 0: raise RuntimeError('Rank moved off assigned logical device zero')
    return public_query(plc, cp, state, function, source_owner, source)


def loaded_libraries():
    result = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/') or '.so' not in fields[5]: continue
        p = Path(fields[5]).resolve()
        if not p.is_file() or not any(s in str(p) for s in ('pylibcugraph','libcugraph','libraft','libnccl','librmm','/rmm/')): continue
        if str(p) in result: continue
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8 << 20), b''): digest.update(block)
        result[str(p)] = dict(path=str(p),bytes=p.stat().st_size,sha256=digest.hexdigest())
    if not any('libcugraph' in p for p in result): raise ValueError('Actual loaded cuGraph library missing')
    return result


class CuGraphMG:
    def __init__(self, graph, function, source, uuids, cpus, scratch):
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client, wait
        import cugraph.dask.comms.comms as comms
        if type(source) is not int or not 0 <= source < graph['n']: raise ValueError('Source outside canonical domain')
        if len(uuids) not in (2,4,8) or len(set(uuids)) != len(uuids): raise ValueError('Require unique explicit MG UUIDs')
        self.cluster = self.client = None; self.futures = []; self.own_comms = False; self.closed = False
        self.function, self.source, self.n = function, source, graph['n']
        self.uuids, self.cpus, self.comms = list(uuids), sorted(cpus), comms
        self.metrics = {}; self.observation = {}
        self.contract = dict(implementation='native pylibcugraph.MGGraph with RAFT collective',
            source=source, graph_direction='weak' if function == 'wcc' else 'directed',
            sssp_weights='canonical float64',
            sssp_predecessors=dict(compute_predecessors=True,returned_to_caller=False,
                reason='Installed pylibcugraph Python release requires True; v3 tiny rejection and official API documentation agree'),
            pr_weighting='unweighted FP64',pr_alpha=.85,
            pr_epsilon_global_l1=1e-9,pr_max_iterations=10000,pr_initial_guess_reused=False,
            host_output='linear scatter after complete unique canonical vertex domain validation',
            cpu_binding_contract='Exactly 16 allowed logical CPU IDs at setup; backend-selected nonempty affinity subsets remain valid; no rank may escape that domain',
            wcc_label_contract='partition equivalence',graph_disk_copies=0,comms_p2p=False)
        versions = {name: resolve_distribution(name)[1] for name in PACKAGES}
        resources = dict(host_bytes_per_rank=32 << 30, device_memory_limit_bytes=20 << 30,
            rmm_pool_bytes=256 << 20, rmm_max_bytes=16 << 30)
        options = cluster_options(dict(scratch=str(scratch),resources=resources),
                                  dict(gpu_count=len(uuids),id='owned'), uuids)
        self.observation['cluster_options'] = options
        start = time.perf_counter()
        try:
            if comms.is_initialized(): raise RuntimeError('Existing communicator is not owned by this job')
            self.cluster = LocalCUDACluster(**options); self.client = Client(self.cluster)
            self.client.wait_for_workers(len(uuids), timeout=120)
            info = self.client.scheduler_info(n_workers=-1); self.workers = sorted(info['workers'])
            affinity = self.client.run(set_cpu_affinity, self.cpus)
            scheduler = self.client.run_on_scheduler(observe_services); services = self.client.run(observe_services)
            check_cluster_shape(info,scheduler,services,len(uuids),resources['host_bytes_per_rank'])
            self.observation.update(worker_cpu_affinity=affinity,scheduler_service_observation=scheduler,
                worker_service_observations=services,scheduler_info_observed=info)
            self.observation['toolkit_workers'] = self.client.run(toolkit_probe, versions)
            bindings = self.client.run(worker_binding)
            self._check_bindings(bindings, require_context=False)
            self.observation['setup_bindings'] = bindings
            self.own_comms = True; comms.initialize(p2p=False)
            sid = comms.get_session_id()
            self.observation['cluster_and_comms_setup_seconds'] = time.perf_counter() - start
            prepare_start = time.perf_counter(); descriptor = describe_graph(graph)
            self.futures = [self.client.submit(build_rank,sid,descriptor,rank,len(uuids),function,
                workers=[address],allow_other_workers=False,pure=False) for rank,address in enumerate(self.workers)]
            wait(self.futures)
            for future in self.futures:
                if future.status == 'error': raise future.exception()
                if future.status != 'finished': raise RuntimeError('MG construction not complete')
            self.observation['rank_csr_conversion_and_graph_preparation_seconds'] = time.perf_counter() - prepare_start
            bindings = self.client.run(worker_binding); self._check_bindings(bindings)
            self.observation['initial_bindings'] = bindings
        except BaseException:
            self.close(); raise

    def _check_bindings(self, bindings, require_context=True):
        if len(bindings) != len(self.uuids) or {x['uuid'] for x in bindings.values()} != set(self.uuids):
            raise RuntimeError('Incorrect actual MG device domain')
        if len({x['host_pid'] for x in bindings.values()}) != len(self.uuids): raise RuntimeError('MG ranks share a process')
        def affinity_valid(observed):
            if not require_context:
                return observed == self.cpus
            return bool(observed) and set(observed) <= set(self.cpus)
        if any((require_context and not x['own_context_observed']) or x['current_device'] != 0 or not affinity_valid(x['cpu_affinity']) for x in bindings.values()):
            raise RuntimeError('MG context/device/CPU affinity observation failed: '+repr({'observed':bindings,'expected_cpus':self.cpus,'require_context':require_context}))

    def complete_query(self):
        if self.closed: raise RuntimeError('Closed owned MG graph')
        outputs = [self.client.submit(query_rank,future,self.function,rank == 0,self.source,
            workers=[address],allow_other_workers=False,pure=False)
            for rank,(address,future) in enumerate(zip(self.workers,self.futures))]
        try: result = complete_output(self.client.gather(outputs),self.n,self.function)
        finally:
            for output in outputs: output.release()
        self.metrics = dict(devices_used=list(range(len(self.uuids))),native_mg=True,
            csr_layout='explicit weak' if self.function == 'wcc' else 'directed canonical',
            complete_host_output=True,pr_epsilon_global_l1=1e-9 if self.function == 'pr' else None)
        return result

    def observe(self):
        after = self.client.run(worker_binding); self._check_bindings(after)
        for address, previous in self.observation['initial_bindings'].items():
            for key in ('uuid','host_pid','current_device','pci_bus_id'):
                if previous[key] != after[address][key]: raise RuntimeError('Rank identity changed')
        per_rank = self.client.run(loaded_libraries)
        libraries = next(iter(per_rank.values()))
        if any(value != libraries for value in per_rank.values()): raise RuntimeError('MG ranks loaded different binaries')
        self.observation.update(final_bindings=after,worker_loaded_libraries=per_rank,
            loaded_backend_binaries=libraries,cleanup_completed=False)
        return self.observation

    def close(self):
        if self.closed: return
        errors = []
        for future in self.futures:
            try: future.release()
            except BaseException as exc: errors.append(repr(exc))
        for enabled, operation in ((self.own_comms,lambda:self.comms.destroy()),
            (self.client is not None,lambda:self.client.close(timeout=30)),
            (self.cluster is not None,lambda:self.cluster.close(timeout=30))):
            if enabled:
                try: operation()
                except BaseException as exc: errors.append(repr(exc))
        self.closed = True
        self.observation['cleanup_errors'] = errors
        if errors: raise RuntimeError('Owned MG cleanup failed: ' + repr(errors))
