"""Full-vector FP64 PageRank certificate for the frozen positive-integer CSR domain.

This audit executes outside all query/workflow timers. A failed domain/bound
certificate falls back to the unchanged independent reference implementation.
A certificate never substitutes a measured reference error with an estimate.
"""
from fractions import Fraction as F
from pathlib import Path
import ctypes,hashlib,math,sys
import numpy as np

METHOD='state-pr-full-residual-certificate-v1'
UNIT=F(1,2**53)
TINY=F(1,2**1022)
SUPPORTED_QUALITY={'pr_alpha':.85,'reference_stop_l1':1e-12,'reference_cap':2000,
                   'residual_l1':1e-6,'mass_error':1e-6,'reference_l1':1e-5}


def upward(value):
    result=float(value)
    return math.nextafter(result,math.inf) if F(result)<value else result


def gamma(count):
    value=count*UNIT
    if value>=1:raise ValueError('Roundoff accumulation domain exceeded')
    return value/(1-value)


def bounds(n,m,residual_sum,mass_sum,quality):
    """Exact rational arithmetic until the outward-rounded report conversion."""
    if not 0<n<2**31 or not 0<=m<2**31:raise ValueError('Certificate requires supported int32 CSR counts')
    if any(quality.get(k)!=v for k,v in SUPPORTED_QUALITY.items()):raise ValueError('Unknown numerical contract')
    if not all(math.isfinite(x) and x>=0 for x in (residual_sum,mass_sum)):raise ValueError('Invalid full-vector reductions')
    alpha=F(quality['pr_alpha']);teleport=F(.15);gap=1-alpha;gn=gamma(n);g=gamma(m+n+32);gt=gamma(32)
    # tiny, rather than half the least subnormal, also covers flush-to-zero.
    tau=64*(m+n+1)*TINY
    reduction=(1-gn)*(1-UNIT)
    residual=F(residual_sum)/reduction+tau
    candidate_mass=F(mass_sum)/(1-gn)+tau
    # Every iterate of the original independent positive FP64 iteration lies
    # below this mass envelope, including its uniform starting vector.
    q=alpha*(1+g)
    if q>=1:raise ValueError('Rounded operator mass bound is not contractive')
    reference_mass=max(1+UNIT+n*TINY,(teleport*(1+gt)+tau)/(1-q))
    candidate_roundoff=alpha*g*candidate_mass+teleport*gt+tau
    reference_roundoff=alpha*g*reference_mass+teleport*gt+tau
    last_update=F(quality['reference_stop_l1'])/reduction+tau
    candidate_fixed=(residual+candidate_roundoff)/gap
    reference_fixed=(alpha*last_update+reference_roundoff)/gap
    exact_distance=candidate_fixed+reference_fixed
    # Include the rounding of the ORIGINAL reference comparison itself.
    computed_distance=(1+gn)*(1+UNIT)*exact_distance+tau
    return dict(reference_l1_upper_bound=upward(computed_distance),
        exact_fixed_point_l1_upper_bound=upward(candidate_fixed),
        converged_original_reference_l1_upper_bound=upward(reference_fixed),
        measured_residual_true_l1_upper_bound=upward(residual),
        candidate_step_roundoff_l1_upper_bound=upward(candidate_roundoff),
        reference_step_roundoff_l1_upper_bound=upward(reference_roundoff),
        reference_iterate_mass_upper_bound=upward(reference_mass),
        rounding_unit=float(UNIT),roundoff_operation_bound=m+n+32,
        rounding_assumption='IEEE binary64 round-to-nearest; nonnegative reductions; positive integer weights1..16',
        original_reference_stop_l1=quality['reference_stop_l1'],
        certificate_pass=computed_distance<=F(quality['reference_l1']))


def domain(matrix):
    n,m=matrix.shape[0],matrix.nnz
    if matrix.shape!=(n,n) or matrix.dtype!=np.float64 or not 0<n<2**31 or not 0<=m<2**31:
        return 'Unsupported square FP64 int32-count CSR domain'
    if matrix.format!='csr':return 'CSR format required'
    f=np.finfo(np.float64)
    if f.nmant!=52 or f.eps!=2.**-52 or f.maxexp!=1024:return 'Unsupported floating format'
    try:
        fn=ctypes.CDLL(None).fegetround;fn.restype=ctypes.c_int
        if fn()!=0:return 'Round-to-nearest is required on this Linux ABI'
    except (AttributeError,OSError):return 'Cannot establish rounding mode'
    if not sys.platform.startswith('linux'):return 'Rounding-mode ABI not established'
    if len(matrix.indptr)!=n+1 or int(matrix.indptr[0])!=0 or int(matrix.indptr[-1])!=m:
        return 'Invalid CSR offsets'
    block=1<<20
    for start in range(0,n,block):
        part=matrix.indptr[start:min(n,start+block)+1]
        if np.any(part[1:]<part[:-1]):return 'Nonmonotone CSR offsets'
    for start in range(0,m,block):
        values=matrix.data[start:start+block];indices=matrix.indices[start:start+block]
        if not (np.all(np.isfinite(values)) and np.all(values>=1) and np.all(values<=16) and np.all(values==np.floor(values))):
            return 'Weight domain is not positive integer1..16'
        if np.any(indices<0) or np.any(indices>=n):return 'CSR endpoint outside vertex domain'
    # Each row strength is an exact integer <=16*m<2**35<2**53.
    return None


def output_quality(original,values,row,version,function,source,cache=None):
    if function!='weighted_pr':return original.output_quality(values,row,version,function,source,cache)
    n=row['n']
    if values.shape!=(n,) or values.dtype!=np.float64 or values.flags.writeable or not values.flags.owndata or values.base is not None:
        raise ValueError('Incomplete or nonowning public result')
    if not np.all(np.isfinite(values)) or np.any(values<0):raise ValueError('PR finite/nonnegative contract failed')
    matrix=original.matrix_for(row,version,function)
    reason=domain(matrix)
    if reason is None:
        step=original.pr_step(matrix)(values)
        residual=float(np.abs(values-step).sum());mass_sum=float(values.sum());mass=abs(mass_sum-1)
        if not np.all(np.isfinite(step)) or residual>original.QUALITY['residual_l1'] or mass>original.QUALITY['mass_error']:
            raise ValueError('PR residual/mass contract failed')
        try:proof=bounds(n,matrix.nnz,residual,mass_sum,original.QUALITY)
        except ValueError as error:reason=str(error)
        else:
            if proof['certificate_pass']:
                return dict(passed=True,vertices=n,residual_l1=residual,mass_error=mass,
                    reference_l1_upper_bound=proof['reference_l1_upper_bound'],
                    reference_l1_kind='certified_upper_bound_not_measured_difference',
                    reference=dict(independent=True,solver=METHOD,full_vector=True,power_solve_executed=False,
                        reference_vector_generated=False,original_reference_convergence_required=False,
                        meaning='Full residual certifies fixed-point error and distance to any original reference satisfying its unchanged stop criterion',
                        proof=proof,certificate_source=source_binding()))
            reason='Conservative roundoff bound is too near the unchanged reference_l1 limit'
    result=original.output_quality(values,row,version,function,source,cache)
    result['audit_method']=dict(requested=METHOD,used='original_independent_reference',fallback_reason=reason,
                               certificate_source=source_binding())
    return result


def source_binding():
    p=Path(__file__).resolve();data=p.read_bytes()
    return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
