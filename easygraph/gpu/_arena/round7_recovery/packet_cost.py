"""Causal packet preparation price shared by all cost-based recovery controls.

Observations come only from already executed maintenance in this session.
The GPU byte model remains unchanged. This is an estimated service price,
not a guarantee about actual elapsed time or a future-query oracle.
"""
from collections import deque
from dataclasses import replace
import math
import statistics
class PacketPrice:
    def __init__(self):self.samples={'topology':deque(maxlen=8),'attribute':deque(maxlen=8)}
    @staticmethod
    def features(old_edges,changes):
        topology=any(c.base_weight is None for c in changes)
        return ('topology',max(1,old_edges+len(changes))) if topology else ('attribute',max(1,len(changes)))
    def price(self,problem,features):
        if problem.cost_model is None:return problem,None
        kind,work=features;prior=problem.cost_model.packet_fixed
        rates=self.samples[kind]
        host=statistics.median(rates)*work if rates else max(0.,prior-problem.cost_model.restore_fixed)
        estimate=problem.cost_model.restore_fixed+host
        cost=replace(problem.cost_model,packet_fixed=estimate,
            provenance='Charged device calibration + causal median packet-preparation service rate, last eight same update-kind observations')
        return replace(problem,cost_model=cost),dict(kind=kind,work=work,observations=len(rates),
            packet_fixed_seconds=estimate,historical_packet_fixed_seconds=prior,
            rate_seconds_per_work=None if not rates else statistics.median(rates))
    def observe(self,features,detail):
        if detail.get('path')!='incremental':return
        value=detail.get('host_advance_preparation_seconds',detail.get('packet_seconds'))
        if value is None:return
        if not math.isfinite(value) or value<0:raise ValueError('Invalid measured packet preparation cost')
        kind,work=features;self.samples[kind].append(value/work)
