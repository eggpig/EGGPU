"""Immutable, explicitly qualified SSSP policies; no environment mutation or GPU import."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from types import MappingProxyType

VERSION='egmgpu.sssp-execution-policy/v1'
ROUTES_SCHEMA='egmgpu.sssp-execution-policy-routes/v1'
_CHOICES={
    'single_owner_execution':('off','arena_v3_if_eligible'),
    'weight_encoding':('f64','u8_exact'), 'wire_encoding':('f64','u32_exact'),
    'output_team':('worker1','team16'), 'atomic_min':('cas','native_u64'),
    'exchange_sync':('legacy','trim'), 'pinned_placement':('caller','numa_local'),
    'host_assembly':('caller','worker'), 'relax_policy':('thread','warp'),
    'control_policy':('atomic','warp'), 'weight_identity':('compute','manifest_f64'),
}
_BOOLS=('profile','pinned_reuse','prepared_uploads','allow_undirected')

@dataclass(frozen=True)
class SsspExecutionPolicy:
    delta: float=1.0
    weight_encoding: str='f64'
    wire_encoding: str='f64'
    output_team: str='worker1'
    atomic_min: str='cas'
    exchange_sync: str='legacy'
    pinned_placement: str='caller'
    host_assembly: str='caller'
    relax_policy: str='thread'
    control_policy: str='atomic'
    weight_identity: str='manifest_f64'
    profile: bool=True
    pinned_reuse: bool=True
    prepared_uploads: bool=True
    allow_undirected: bool=True
    single_owner_execution: str='off'

    def __post_init__(self):
        if isinstance(self.delta,bool) or not isinstance(self.delta,(int,float)) or not math.isfinite(self.delta) or self.delta<=0:
            raise ValueError('SSSP delta must be finite and positive')
        for field,allowed in _CHOICES.items():
            if not isinstance(getattr(self,field),str) or getattr(self,field) not in allowed:
                raise ValueError('Invalid SSSP policy '+field)
        for field in _BOOLS:
            if type(getattr(self,field)) is not bool:raise TypeError('SSSP '+field+' must be bool')
        if self.output_team=='team16' and self.host_assembly!='worker':
            raise ValueError('SSSP team16 requires worker host assembly')

    @classmethod
    def from_mapping(cls,value):
        if isinstance(value,cls):return value
        if not isinstance(value,Mapping):raise TypeError('SSSP policy must be a mapping')
        return cls(**dict(value))

    def native_options(self):
        result=asdict(self);del result['delta'];return result

    def document(self):return asdict(self)


def verify_policy_observation(raw,policy):
    """Request identity and actual engine are independently checked."""
    if (raw.get('execution_policy_version')!=VERSION or raw.get('execution_policy_source')!='explicit'
            or raw.get('execution_policy')!=policy.native_options()):
        raise ValueError('Native SSSP request policy differs')
    maximum=raw.get('numeric_maximum_weight')
    vertices=raw.get('numeric_vertex_count')
    noninteger=raw.get('numeric_noninteger_weight_count')
    if (raw.get('numeric_domain_version')!='nonnegative_fp64_all_light_fallback_v1'
            or raw.get('numeric_statistics_available') is not True
            or type(vertices) is not int or vertices<=0
            or type(noninteger) is not int or noninteger<0
            or type(maximum) not in (int,float) or not math.isfinite(maximum) or maximum<0
            or raw.get('requested_delta')!=policy.delta):
        raise ValueError('SSSP numerical-domain current-plan metadata differs')
    if raw.get('single_owner_execution_used'):
        observed=raw.get('execution_policy_effective',{})
        engine=raw.get('single_owner_engine')
        if (policy.single_owner_execution!='arena_v3_if_eligible' or not raw.get('single_owner_execution_eligible')
                or raw.get('partition_count')!=1 or len(raw.get('devices_used',[]))!=1
                or raw.get('requested_delta')!=policy.delta or raw.get('delta') is not None
                or engine not in ('corrected_warp_frontier_v2','device_delta_ieee_v3')
                or observed.get('engine')!=engine or observed.get('fixed_delta_applied') is not False
                or observed.get('legacy_kernel_controls_applied') is not False
                or observed.get('checkpoint_iterations')!=16 or observed.get('blocks_per_sm')!=8
                or observed.get('vector_width')!=(4 if engine=='corrected_warp_frontier_v2' else 32)
                or raw.get('weight_encoding')!='f64' or raw.get('wire_encoding')!='f64'
                or raw.get('bucket_rounds') is not None or raw.get('light_relaxation_rounds') is not None
                or raw.get('atomic_min_policy')!=('cas' if engine=='corrected_warp_frontier_v2' else 'native_u64')
                or raw.get('control_policy')!='device_resident_checkpoint16'
                or raw.get('single_owner_scratch_capacity_bytes',0)<=0
                or raw.get('effective_delta') is not None
                or raw.get('numeric_domain_route')!='single_owner_ieee'
                or raw.get('numeric_domain_reason')!='existing_single_owner_ieee_engine'
                or raw.get('numeric_integer_bound_used') is not False
                or raw.get('numeric_all_edges_light') is not False
                or raw.get('numeric_delta_changed') is not False
                or observed.get('requested_delta')!=policy.delta
                or observed.get('effective_delta') is not None
                or observed.get('requested_delta_applied') is not False
                or observed.get('numeric_domain_route')!='single_owner_ieee'):
            raise ValueError('Single-owner SSSP actual execution contract differs')
    else:
        maximum=raw.get('numeric_maximum_weight')
        route=raw.get('numeric_domain_route')
        effective=raw.get('effective_delta')
        if (raw.get('numeric_domain_version')!='nonnegative_fp64_all_light_fallback_v1'
                or raw.get('requested_delta')!=policy.delta
                or not isinstance(maximum,(int,float)) or not math.isfinite(maximum) or maximum<0
                or not isinstance(effective,(int,float)) or not math.isfinite(effective) or effective<=0
                or raw.get('delta')!=effective
                or raw.get('numeric_delta_changed')!=(effective!=policy.delta)):
            raise ValueError('Partitioned SSSP requested/effective numerical domain differs')
        if route=='requested_all_light':
            valid=(effective==policy.delta and maximum<=effective and raw.get('numeric_all_edges_light') is True
                and raw.get('numeric_integer_bound_used') is False
                and raw.get('numeric_domain_reason')=='current_maximum_weight_at_most_requested_delta')
        elif route=='requested_exact_integer_delta16':
            valid=(effective==policy.delta==16 and maximum>effective and raw.get('numeric_noninteger_weight_count')==0
                and raw.get('numeric_integer_bound_used') is True and raw.get('numeric_all_edges_light') is False
                and maximum<2**53 and float(maximum).is_integer() and (vertices-1)*int(maximum)<2**53
                and raw.get('numeric_domain_reason')=='integer_simple_path_bound_below_2p53_with_exact_binary_bucket_scaling')
        elif route=='certified_maximum_all_light':
            valid=(effective==maximum>policy.delta and raw.get('numeric_all_edges_light') is True
                and raw.get('numeric_integer_bound_used') is False
                and raw.get('numeric_domain_reason')=='fixed_requested_bucket_domain_unproved_use_all_edges_light')
        else:valid=False
        observed=raw.get('execution_policy_effective',{})
        if (not valid or observed.get('requested_delta')!=policy.delta or observed.get('effective_delta')!=effective
                or observed.get('requested_delta_applied')!=(effective==policy.delta)
                or observed.get('numeric_domain_route')!=route):
            raise ValueError('Partitioned SSSP actual effective-delta route differs')


def invoke_sssp_policy(binding,native,source,devices,policy,output_options=None):
    options=dict(output_options or {})
    if policy is None:
        # Exact positional compatibility with pre-policy extensions and evidence.
        return binding(native,source,1.0,0,list(devices),**options)
    if not isinstance(policy,SsspExecutionPolicy):raise TypeError('Expected immutable SSSP policy')
    options['execution_policy']=policy.native_options()
    raw=binding(native,source,policy.delta,0,list(devices),**options)
    verify_policy_observation(raw,policy)
    return raw


def _pci_key(value):
    domain,bus,slot=str(value).lower().split(':')
    device,function=slot.split('.')
    return tuple(int(x,16) for x in (domain,bus,device,function))


def normalize_device_binding(rows):
    """IDs in service routes are logical; physical labels and UUIDs are explicit."""
    result={}
    for row in rows:
        logical=row['logical'];physical=row['physical'];uuid=row['uuid']
        if type(logical) is not int or logical<0 or type(physical) is not int or physical<0:
            raise ValueError('Invalid logical/physical device ID')
        if not isinstance(uuid,str) or not uuid.startswith('GPU-'):
            raise ValueError('Full GPU UUID required')
        if logical in result:raise ValueError('Repeated logical device')
        result[logical]=(physical,uuid,_pci_key(row['pci_bus_id']))
    if len({x[0] for x in result.values()})!=len(result) or len({x[1] for x in result.values()})!=len(result):
        raise ValueError('Repeated physical device or UUID')
    return result


def load_sssp_policy_routes(path,*,extension_path,eligible_groups,observed_device_binding=None,cuda_topology=None):
    """Read prospective policy choices only after explicit-policy GPU qualification.

    Existing extension/topology/function evidence remains enforced by the caller.
    This additional contract authorizes an exact policy for each device group;
    it does not infer an online optimum or reuse legacy environment-only evidence.
    """
    if path is None:return MappingProxyType({})
    if not isinstance(path,(str,Path)):raise TypeError('SSSP policy routes must be a JSON path')
    document=json.loads(Path(path).read_text())
    if document.get('schema')!=ROUTES_SCHEMA or document.get('frozen_before_formal') is not True:
        raise ValueError('SSSP policy routes require a prospectively frozen schema')
    if observed_device_binding is None or cuda_topology is None:
        raise ValueError('Explicit SSSP routes require an observed logical/physical/UUID binding and CUDA topology')
    observed=normalize_device_binding(observed_device_binding)
    cuda={int(d['device_id']):_pci_key(d['pci_bus_id']) for d in cuda_topology['devices']}
    if any(cuda.get(logical)!=identity[2] for logical,identity in observed.items()):
        raise ValueError('Observed device binding disagrees with process CUDA logical topology')
    groups={tuple(x) for x in eligible_groups};policies={}
    for route in document['routes']:
        devices=tuple(route['devices'])
        if devices not in groups or devices in policies:raise ValueError('Unqualified or duplicate SSSP device group')
        qualified_binding=normalize_device_binding(route['device_binding'])
        if set(qualified_binding)!=set(devices) or any(observed.get(d)!=qualified_binding[d] for d in devices):
            raise ValueError('SSSP route logical/physical/UUID binding differs from observed deployment')
        policy=SsspExecutionPolicy.from_mapping(route['policy'])
        if policy.profile:raise ValueError('Production SSSP route must disable diagnostic event profiling')
        qualification=json.loads(Path(route['qualification']).read_text())
        if not (qualification.get('status')=='passed' and qualification.get('phase')=='qualification'
                and qualification.get('function')=='sssp' and qualification.get('execution')=='parallel'
                and qualification.get('logical_devices')==list(devices)
                and qualification.get('physical_devices')==[qualified_binding[d][0] for d in devices]
                and normalize_device_binding(qualification.get('logical_device_mapping',[]))==qualified_binding
                and Path(qualification['loaded_extension']).resolve()==Path(extension_path).resolve()
                and qualification.get('full_outputs')==3 and qualification.get('outputs_survive_shutdown') is True
                and len(qualification.get('calls',[]))==3):
            raise ValueError('SSSP policy needs matching new-artifact explicit qualification')
        for call in qualification['calls']:
            if not call.get('full_vector_correct') or call.get('full_vector_size')!=qualification['graph']['vertex_count']:
                raise ValueError('SSSP policy qualification lacks complete correct output')
            if call.get('logical_devices')!=list(devices) or call['metrics'].get('devices_used')!=list(devices):
                raise ValueError('SSSP qualification actual logical devices differ from route')
            verify_policy_observation(call['metrics'],policy)
        policies[devices]=policy
    if not policies:raise ValueError('An explicit SSSP policy document must contain routes')
    return MappingProxyType(policies)
