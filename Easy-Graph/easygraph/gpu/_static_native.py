"""Explicit immutable native sessions with exact compact SSSP representation.

No automatic route choice uses graph names or measured speed. All GPU counts
use the same selected policy. The original inputs and public outputs remain
FP64, and compact eligibility is fully certified before runtime startup.
"""
from dataclasses import replace
import copy
import os
from time import perf_counter

from ._core import native_policy
from ._native import UnifiedNativeSession, verify_manifest

ENCODINGS = ('f64', 'u8_exact_u32_exact')
DELTA_RULES = ('fixed16', 'certified_bound_plus_one')


class StaticNativeSession(UnifiedNativeSession):
    capabilities = (UnifiedNativeSession.capabilities - {'register_snapshot'}) | {'static_exact_sssp_encoding'}

    def __init__(self, cpp, manifests, *, devices=(0,), sssp_encoding='u8_exact_u32_exact',
                 sssp_delta_rule='fixed16', **options):
        if sssp_encoding not in ENCODINGS:
            raise ValueError('Static SSSP encoding must be f64 or u8_exact_u32_exact')
        if sssp_delta_rule not in DELTA_RULES:
            raise ValueError('Static SSSP delta rule must be fixed16 or certified_bound_plus_one')
        if sssp_delta_rule == 'certified_bound_plus_one' and sssp_encoding != 'u8_exact_u32_exact':
            raise ValueError('Certified bound-plus-one scheduling requires exact compact integer encoding')
        if os.environ.get('EGGPU_DYNAMIC_WEIGHT_ROUTE', 'persistent_full') != 'persistent_full':
            raise ValueError('Static native input requires persistent_full; dynamic weight routes are unsupported')
        if os.environ.get('EGGPU_SSSP_WEIGHT_CERTIFICATION', 'full') != 'full':
            raise ValueError('Static native input requires full weight certification')
        self.sssp_encoding = sssp_encoding
        self.sssp_delta_rule = sssp_delta_rule
        self.numeric_certificates = {}
        policy, verify = native_policy(sssp_encoding)
        # Strong scaling keeps the same native delta-stepping policy for every
        # P, including P1. Never select the separate single-owner engine here.
        policy = replace(policy, single_owner_execution='off')
        super().__init__(cpp, manifests, devices=devices, **options)
        self.sssp_policy, self.verify_sssp = policy, verify

    def _validate_manifest_document(self, name, path, document):
        begin = perf_counter()
        compact = self.sssp_encoding == 'u8_exact_u32_exact' and document['graph'].get('weighted', False)
        certificate = {} if compact else None
        digest = verify_manifest(path, document, numeric_certificate=certificate)
        self.snapshot_hashes[name] = digest
        if certificate is not None:
            delta = certificate['finite_distance_upper_bound'] + 1
            if self.sssp_delta_rule == 'certified_bound_plus_one' and delta < certificate['max_weight']:
                raise ValueError('Certified bound-plus-one requires B+1 >= maxW; select fixed16 explicitly for this domain')
            certificate.update(manifest_document_sha256=digest,
                               admission_identity_and_numeric_check_seconds=perf_counter() - begin)
            self.numeric_certificates[name] = certificate

    def query(self, graph_id, algorithm, parameters=None):
        with self._lock:
            if algorithm == 'sssp' and self.sssp_encoding == 'u8_exact_u32_exact' and graph_id not in self.numeric_certificates:
                raise ValueError('Static compact SSSP needs its certified weighted manifest')
            if algorithm == 'sssp':
                # Select only by the explicit rule and this immutable graph's
                # fully scanned certificate, while holding the session lock.
                delta = (self.numeric_certificates[graph_id]['finite_distance_upper_bound'] + 1
                         if self.sssp_delta_rule == 'certified_bound_plus_one' else 16)
                self.sssp_policy = replace(self.sssp_policy, delta=float(delta))
            result = super().query(graph_id, algorithm, parameters)
            if algorithm == 'sssp':
                metrics = result.backend_metrics
                expected = ('u8_exact', 'u32_exact') if self.sssp_encoding != 'f64' else ('f64', 'f64')
                if (metrics.get('weight_encoding'), metrics.get('wire_encoding')) != expected:
                    raise RuntimeError('Native execution did not use the selected static SSSP encoding')
                if metrics.get('single_owner_execution_used') is not False:
                    raise RuntimeError('Static fixed-native policy must keep the single-owner engine off')
                if self.sssp_delta_rule == 'certified_bound_plus_one':
                    if (metrics.get('requested_delta') != delta or metrics.get('effective_delta') != delta
                            or metrics.get('delta') != delta or metrics.get('numeric_delta_changed') is not False
                            or metrics.get('numeric_all_edges_light') is not True
                            or metrics.get('bucket_rounds') != 1):
                        raise RuntimeError('Native execution did not honor certified B+1 single-bucket scheduling')
                if self.sssp_encoding != 'f64':
                    certificate = self.numeric_certificates[graph_id]
                    if (metrics.get('wire_integer_bound_certified') is not True
                            or metrics.get('wire_maximum_weight') != certificate['max_weight']
                            or metrics.get('wire_distance_bound') != certificate['finite_distance_upper_bound']):
                        raise RuntimeError('Native compact bound differs from original FP64 certification')
                    metrics['static_sssp_input_certificate'] = copy.deepcopy(certificate)
                metrics['static_sssp_encoding'] = self.sssp_encoding
                metrics['static_sssp_delta_rule'] = self.sssp_delta_rule
                metrics['static_sssp_requested_delta'] = float(delta)
                metrics['static_sssp_scope'] = 'Immutable inputs; identical fixed native policy across P; FP64 public values.'
            return result

    def _immutable_only(self, *args, **kwargs):
        raise NotImplementedError('Static native sessions do not support dynamic commits, snapshot registration or readonly device leases; use Session.native with FP64 for those operations')

    commit_weights = _immutable_only
    register_snapshot = _immutable_only
    retain_snapshot = _immutable_only
    release_snapshot = _immutable_only
    acquire_readonly_view = _immutable_only

    def close(self):
        with self._lock:
            if not self.closed and self.cpp.cpp_egmgpu_runtime_status()['running']:
                # Match the existing dynamic wrapper's release of process-wide
                # prepared caches before shutting down their owning workers.
                self.closed_native_cache_receipt = self.cpp.cpp_egmgpu_clear_partitioned_state(devices=self.devices)
            super().close()
            if self.closed:
                for name in ('graphs', 'metadata', 'nodes', '_domains', '_manifest_documents',
                             '_manifest_paths', 'snapshot_hashes', 'numeric_certificates'):
                    getattr(self, name, {}).clear()

    def capability_descriptor(self):
        base = super().capability_descriptor()
        return replace(base, mutations='none; immutable manifests fixed at creation',
            control_routes=tuple(route for route in base.control_routes if route != 'register_committed_snapshot') + ('static_exact_sssp_encoding',),
            qualification='Static exact encoding requires separate source-bound native GPU qualification for P1/P2/P4/P8.',
            limitations=base.limitations + (
                'Dynamic weight commits, registration, snapshot retention requests and readonly device leases are unsupported.',
                'Compact SSSP requires all original FP64 weights to be integers in [0,255] and (N-1)*maxW < UINT32_MAX.',
                'Optional certified_bound_plus_one requires compact input and B+1 >= maxW; no implicit fallback. Default fixed16 remains available.',
                'Compact storage changes resident weights and wire representation only; distance arithmetic and public outputs remain FP64.'))
